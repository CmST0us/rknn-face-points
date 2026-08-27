#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rknn_face {

struct Point3 {
  float x, y, z;
};

struct Image {
  int w = 0, h = 0;
  std::vector<unsigned char> rgb;
};

struct Result {
  float score = 0;
  float bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;
  float landmark_presence = 0;
  float tongue_out = 0;
  std::array<float, 3> head_pose_deg{};
  std::array<float, 52> blendshapes{};
  std::vector<Point3> landmarks;
  double detector_ms = 0, landmark_ms = 0, blendshape_ms = 0;
};

class Tracker {
 public:
  explicit Tracker(const std::string& model_dir);
  ~Tracker();
  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;

  bool process(const Image& image, Result& result);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void draw(Image& image, const Result& result);
const char* blendshape_name(std::size_t index);

}  // namespace rknn_face
