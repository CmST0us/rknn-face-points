#include "face_tracker.h"

#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace rknn_face {
namespace {

std::vector<unsigned char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  std::vector<unsigned char> data(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(data.data()), data.size());
  return data;
}

class RknnModel {
 public:
  explicit RknnModel(const std::string& path) {
    auto bytes = read_file(path);
    if (bytes.empty()) throw std::runtime_error("cannot read " + path);
    check(rknn_init(&ctx_, bytes.data(), bytes.size(), 0, nullptr), "rknn_init");
    rknn_input_output_num io{};
    check(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "query io");
    inputs_.resize(io.n_input);
    outputs_.resize(io.n_output);
    for (uint32_t i = 0; i < io.n_input; ++i) {
      inputs_[i].index = i;
      check(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inputs_[i], sizeof(inputs_[i])), "query input");
    }
    for (uint32_t i = 0; i < io.n_output; ++i) {
      outputs_[i].index = i;
      check(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &outputs_[i], sizeof(outputs_[i])), "query output");
    }
  }

  ~RknnModel() { if (ctx_) rknn_destroy(ctx_); }

  std::vector<std::vector<float>> run(const std::vector<float>& input, double* ms) {
    if (inputs_.size() != 1 || input.size() != inputs_[0].n_elems)
      throw std::runtime_error("bad input element count");
    rknn_input in{};
    in.index = 0;
    in.buf = const_cast<float*>(input.data());
    in.size = static_cast<uint32_t>(input.size() * sizeof(float));
    in.type = RKNN_TENSOR_FLOAT32;
    in.fmt = inputs_[0].n_dims == 4 ? RKNN_TENSOR_NHWC : inputs_[0].fmt;
    check(rknn_inputs_set(ctx_, 1, &in), "inputs_set");
    auto start = std::chrono::steady_clock::now();
    check(rknn_run(ctx_, nullptr), "rknn_run");
    *ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    std::vector<rknn_output> desc(outputs_.size());
    for (auto& out : desc) out.want_float = 1;
    check(rknn_outputs_get(ctx_, desc.size(), desc.data(), nullptr), "outputs_get");
    std::vector<std::vector<float>> result(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto* p = static_cast<float*>(desc[i].buf);
      result[i].assign(p, p + outputs_[i].n_elems);
    }
    rknn_outputs_release(ctx_, desc.size(), desc.data());
    return result;
  }

 private:
  static void check(int rc, const char* what) {
    if (rc != RKNN_SUCC) throw std::runtime_error(std::string(what) + ": " + std::to_string(rc));
  }
  rknn_context ctx_ = 0;
  std::vector<rknn_tensor_attr> inputs_, outputs_;
};

void sample_rgb(const Image& image, float x, float y, float* out) {
  if (x < -0.5f || y < -0.5f || x > image.w - 0.5f || y > image.h - 0.5f) {
    out[0] = out[1] = out[2] = 0;
    return;
  }
  x = std::max(0.0f, std::min(x, image.w - 1.0f));
  y = std::max(0.0f, std::min(y, image.h - 1.0f));
  int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  int x1 = std::min(x0 + 1, image.w - 1), y1 = std::min(y0 + 1, image.h - 1);
  float ax = x - x0, ay = y - y0;
  for (int c = 0; c < 3; ++c) {
    float a = image.rgb[(y0 * image.w + x0) * 3 + c];
    float b = image.rgb[(y0 * image.w + x1) * 3 + c];
    float d = image.rgb[(y1 * image.w + x0) * 3 + c];
    float e = image.rgb[(y1 * image.w + x1) * 3 + c];
    out[c] = (a + (b - a) * ax) * (1 - ay) + (d + (e - d) * ax) * ay;
  }
}

struct Letterbox {
  float scale;
  int pad_x, pad_y, resized_w, resized_h;
};

std::vector<float> detector_input(const Image& image, Letterbox& box) {
  constexpr int size = 128;
  box.scale = std::min(size / static_cast<float>(image.w), size / static_cast<float>(image.h));
  box.resized_w = std::max(1, static_cast<int>(std::round(image.w * box.scale)));
  box.resized_h = std::max(1, static_cast<int>(std::round(image.h * box.scale)));
  box.pad_x = (size - box.resized_w) / 2;
  box.pad_y = (size - box.resized_h) / 2;
  std::vector<float> input(size * size * 3, -1.0f);
  for (int y = 0; y < box.resized_h; ++y) {
    for (int x = 0; x < box.resized_w; ++x) {
      float rgb[3];
      sample_rgb(image, (x + 0.5f) / box.scale - 0.5f,
                 (y + 0.5f) / box.scale - 0.5f, rgb);
      size_t dst = ((y + box.pad_y) * size + x + box.pad_x) * 3;
      for (int c = 0; c < 3; ++c) input[dst + c] = rgb[c] / 127.5f - 1.0f;
    }
  }
  return input;
}

struct Anchor { float x, y; };

std::vector<Anchor> make_anchors() {
  const int strides[4] = {8, 16, 16, 16};
  std::vector<Anchor> anchors;
  for (int layer = 0; layer < 4;) {
    int last = layer;
    while (last + 1 < 4 && strides[last + 1] == strides[layer]) ++last;
    int anchors_per_cell = (last - layer + 1) * 2;
    int grid = static_cast<int>(std::ceil(128.0f / strides[layer]));
    for (int y = 0; y < grid; ++y)
      for (int x = 0; x < grid; ++x)
        for (int n = 0; n < anchors_per_cell; ++n)
          anchors.push_back({(x + 0.5f) / grid, (y + 0.5f) / grid});
    layer = last + 1;
  }
  if (anchors.size() != 896) throw std::runtime_error("anchor self-check failed");
  return anchors;
}

struct Detection {
  float score, x, y, w, h;
  std::array<Point3, 6> keypoints;
};

Detection decode_detection(const std::vector<float>& raw_boxes,
                           const std::vector<float>& raw_scores,
                           const Letterbox& letterbox) {
  static const std::vector<Anchor> anchors = make_anchors();
  size_t best = 0;
  for (size_t i = 1; i < anchors.size(); ++i)
    if (raw_scores[i] > raw_scores[best]) best = i;
  float score_logit = std::max(-100.0f, std::min(raw_scores[best], 100.0f));
  const float* r = raw_boxes.data() + best * 16;
  auto image_x = [&](float normalized) { return (normalized * 128 - letterbox.pad_x) / letterbox.scale; };
  auto image_y = [&](float normalized) { return (normalized * 128 - letterbox.pad_y) / letterbox.scale; };
  float cx = r[0] / 128 + anchors[best].x;
  float cy = r[1] / 128 + anchors[best].y;
  Detection d{};
  d.score = 1.0f / (1.0f + std::exp(-score_logit));
  d.w = r[2] / letterbox.scale;
  d.h = r[3] / letterbox.scale;
  d.x = image_x(cx) - d.w / 2;
  d.y = image_y(cy) - d.h / 2;
  for (int i = 0; i < 6; ++i) {
    d.keypoints[i] = {image_x(r[4 + i * 2] / 128 + anchors[best].x),
                      image_y(r[5 + i * 2] / 128 + anchors[best].y), 0};
  }
  return d;
}

struct Roi { float cx, cy, w, h, rotation; };

std::vector<float> landmark_input(const Image& image, const Roi& roi) {
  constexpr int size = 256;
  float cs = std::cos(roi.rotation), sn = std::sin(roi.rotation);
  std::vector<float> input(size * size * 3);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      float local_x = (x + 0.5f) / size - 0.5f;
      float local_y = (y + 0.5f) / size - 0.5f;
      float src_x = roi.cx + cs * local_x * roi.w - sn * local_y * roi.h;
      float src_y = roi.cy + sn * local_x * roi.w + cs * local_y * roi.h;
      float rgb[3];
      sample_rgb(image, src_x, src_y, rgb);
      size_t dst = (y * size + x) * 3;
      for (int c = 0; c < 3; ++c) input[dst + c] = rgb[c] / 255.0f;
    }
  }
  return input;
}

std::vector<Point3> project_landmarks(const std::vector<float>& raw,
                                      const Roi& roi, int image_w, int image_h) {
  if (raw.size() < 478 * 3) throw std::runtime_error("short landmark output");
  float cs = std::cos(roi.rotation), sn = std::sin(roi.rotation);
  std::vector<Point3> points(478);
  for (int i = 0; i < 478; ++i) {
    float x = raw[i * 3] / 256.0f - 0.5f;
    float y = raw[i * 3 + 1] / 256.0f - 0.5f;
    points[i] = {(roi.cx + cs * x * roi.w - sn * y * roi.h) / image_w,
                 (roi.cy + sn * x * roi.w + cs * y * roi.h) / image_h,
                 raw[i * 3 + 2] / 256.0f * roi.w / image_w};
  }
  return points;
}

const int kBlendLandmarks[146] = {
  0,1,4,5,6,7,8,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,63,65,66,67,70,
  78,80,81,82,84,87,88,91,93,95,103,105,107,109,127,132,133,136,144,145,146,148,
  149,150,152,153,154,155,157,158,159,160,161,162,163,168,172,173,176,178,181,185,
  191,195,197,234,246,249,251,263,267,269,270,276,282,283,284,285,288,291,293,295,
  296,297,300,308,310,311,312,314,317,318,321,323,324,332,334,336,338,356,361,362,
  365,373,374,375,377,378,379,380,381,382,384,385,386,387,388,389,390,397,398,400,
  402,405,409,415,454,466,468,469,470,471,472,473,474,475,476,477
};

const char* kBlendNames[52] = {
  "_neutral","browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
  "cheekPuff","cheekSquintLeft","cheekSquintRight","eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft",
  "eyeLookDownRight","eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight","eyeLookUpLeft",
  "eyeLookUpRight","eyeSquintLeft","eyeSquintRight","eyeWideLeft","eyeWideRight","jawForward","jawLeft",
  "jawOpen","jawRight","mouthClose","mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight",
  "mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft","mouthPressRight",
  "mouthPucker","mouthRight","mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
  "mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight","mouthUpperUpLeft",
  "mouthUpperUpRight","noseSneerLeft","noseSneerRight"
};

std::vector<float> blend_input(const std::vector<Point3>& points, int w, int h) {
  std::vector<float> input(146 * 2);
  for (int i = 0; i < 146; ++i) {
    input[i * 2] = points[kBlendLandmarks[i]].x * w;
    input[i * 2 + 1] = points[kBlendLandmarks[i]].y * h;
  }
  return input;
}

Point3 sub(Point3 a, Point3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Point3 cross(Point3 a, Point3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Point3 normalized(Point3 p) {
  float n = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
  return n > 1e-6f ? Point3{p.x / n, p.y / n, p.z / n} : Point3{0, 0, 0};
}

std::array<float, 3> head_pose(const std::vector<Point3>& p, int w, int h) {
  auto px = [&](int i) { return Point3{p[i].x * w, p[i].y * h, p[i].z * w}; };
  Point3 x_axis = normalized(sub(px(263), px(33)));
  Point3 y_hint = normalized(sub(px(152), px(10)));
  Point3 z_axis = normalized(cross(x_axis, y_hint));
  Point3 y_axis = normalized(cross(z_axis, x_axis));
  constexpr float rad_to_deg = 57.2957795f;
  float pitch = std::atan2(y_axis.z, z_axis.z) * rad_to_deg;
  float yaw = std::asin(std::max(-1.0f, std::min(1.0f, -x_axis.z))) * rad_to_deg;
  float roll = std::atan2(x_axis.y, x_axis.x) * rad_to_deg;
  // ponytail: monocular axes; use calibrated canonical-mesh Procrustes for metric AR parity.
  return {yaw, pitch, roll};
}

void pixel(Image& image, int x, int y, unsigned char r, unsigned char g, unsigned char b) {
  if (x < 0 || y < 0 || x >= image.w || y >= image.h) return;
  size_t i = (y * image.w + x) * 3;
  image.rgb[i] = r; image.rgb[i + 1] = g; image.rgb[i + 2] = b;
}

void draw_point(Image& image, int x, int y) {
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) pixel(image, x + dx, y + dy, 30, 255, 90);
}

}  // namespace

struct Tracker::Impl {
  explicit Impl(const std::string& dir)
      : detector(dir + "/rknn_face_detector.rknn"),
        landmarker(dir + "/rknn_face_landmarks.rknn"),
        blendshapes(dir + "/tf2onnx_face_blendshapes.rknn") {}
  RknnModel detector, landmarker, blendshapes;
};

Tracker::Tracker(const std::string& model_dir) : impl_(new Impl(model_dir)) {}
Tracker::~Tracker() = default;

bool Tracker::process(const Image& image, Result& result) {
  if (image.w <= 0 || image.h <= 0 || image.rgb.size() != static_cast<size_t>(image.w) * image.h * 3)
    throw std::runtime_error("invalid RGB image");
  result = Result{};
  Letterbox letterbox{};
  auto detector_outputs = impl_->detector.run(detector_input(image, letterbox), &result.detector_ms);
  Detection detection = decode_detection(detector_outputs[0], detector_outputs[1], letterbox);
  result.score = detection.score;
  if (detection.score < 0.5f) return false;

  Roi roi{detection.x + detection.w / 2, detection.y + detection.h / 2,
          detection.w * 1.5f, detection.h * 1.5f,
          std::atan2(detection.keypoints[1].y - detection.keypoints[0].y,
                     detection.keypoints[1].x - detection.keypoints[0].x)};
  auto landmark_outputs = impl_->landmarker.run(landmark_input(image, roi), &result.landmark_ms);
  result.landmark_presence = 1.0f / (1.0f + std::exp(-landmark_outputs[1][0]));
  result.tongue_out = landmark_outputs[2][0];
  result.landmarks = project_landmarks(landmark_outputs[0], roi, image.w, image.h);
  auto blend_outputs = impl_->blendshapes.run(
      blend_input(result.landmarks, image.w, image.h), &result.blendshape_ms);
  if (blend_outputs[0].size() < result.blendshapes.size())
    throw std::runtime_error("short blendshape output");
  std::copy_n(blend_outputs[0].begin(), result.blendshapes.size(), result.blendshapes.begin());
  result.head_pose_deg = head_pose(result.landmarks, image.w, image.h);
  result.bbox_x = detection.x;
  result.bbox_y = detection.y;
  result.bbox_w = detection.w;
  result.bbox_h = detection.h;
  return true;
}

void draw(Image& image, const Result& result) {
  int x0 = static_cast<int>(result.bbox_x), y0 = static_cast<int>(result.bbox_y);
  int x1 = static_cast<int>(result.bbox_x + result.bbox_w);
  int y1 = static_cast<int>(result.bbox_y + result.bbox_h);
  for (int x = x0; x <= x1; ++x) {
    pixel(image, x, y0, 255, 60, 30); pixel(image, x, y1, 255, 60, 30);
  }
  for (int y = y0; y <= y1; ++y) {
    pixel(image, x0, y, 255, 60, 30); pixel(image, x1, y, 255, 60, 30);
  }
  for (const auto& p : result.landmarks)
    draw_point(image, static_cast<int>(p.x * image.w), static_cast<int>(p.y * image.h));
}

const char* blendshape_name(std::size_t index) {
  return index < 52 ? kBlendNames[index] : "unknown";
}

}  // namespace rknn_face
