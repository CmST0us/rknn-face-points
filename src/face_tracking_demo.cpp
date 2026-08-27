#include "face_tracker.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

using rknn_face::Image;
using rknn_face::Result;

static Image load_image(const char* path) {
  Image image;
  int channels = 0;
  unsigned char* data = stbi_load(path, &image.w, &image.h, &channels, 3);
  if (!data) throw std::runtime_error(std::string("cannot decode ") + path);
  image.rgb.assign(data, data + static_cast<size_t>(image.w) * image.h * 3);
  stbi_image_free(data);
  return image;
}

static void save_ppm(const Image& image, const char* path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(std::string("cannot write ") + path);
  f << "P6\n" << image.w << " " << image.h << "\n255\n";
  f.write(reinterpret_cast<const char*>(image.rgb.data()), image.rgb.size());
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::fprintf(stderr, "usage: %s IMAGE [OUTPUT.ppm] [MODEL_DIR]\n", argv[0]);
    return 2;
  }
  const char* output_path = argc >= 3 ? argv[2] : "face_result.ppm";
  const char* model_dir = argc >= 4 ? argv[3] : "models";
  try {
    Image image = load_image(argv[1]);
    rknn_face::Tracker tracker(model_dir);
    Result result;
    if (!tracker.process(image, result))
      throw std::runtime_error("no face above score 0.5 (best=" + std::to_string(result.score) + ")");

    std::vector<int> order(52);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return result.blendshapes[a] > result.blendshapes[b];
    });
    std::printf("face_score: %.6f\n", result.score);
    std::printf("bbox_px: %.2f %.2f %.2f %.2f\n",
                result.bbox_x, result.bbox_y, result.bbox_w, result.bbox_h);
    std::printf("landmark_presence: %.6f\n", result.landmark_presence);
    std::printf("landmarks: %zu\n", result.landmarks.size());
    std::printf("first_landmark: %.7f %.7f %.7f\n",
                result.landmarks[0].x, result.landmarks[0].y, result.landmarks[0].z);
    std::printf("head_pose_deg: yaw=%.2f pitch=%.2f roll=%.2f\n",
                result.head_pose_deg[0], result.head_pose_deg[1], result.head_pose_deg[2]);
    std::printf("tongueOut: %.6f\n", result.tongue_out);
    std::printf("top_blendshapes:\n");
    int shown = 0;
    for (int i : order) {
      if (i == 0) continue;
      std::printf("  %-20s %.6f\n", rknn_face::blendshape_name(i), result.blendshapes[i]);
      if (++shown == 10) break;
    }
    std::printf("npu_ms: detector=%.3f landmarks=%.3f blendshapes=%.3f total=%.3f\n",
                result.detector_ms, result.landmark_ms, result.blendshape_ms,
                result.detector_ms + result.landmark_ms + result.blendshape_ms);
    rknn_face::draw(image, result);
    save_ppm(image, output_path);
    std::printf("output: %s\n", output_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
