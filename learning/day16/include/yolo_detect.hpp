/**
 * day15/include/yolo_detect.hpp
 *
 * YOLOv5 Detection Pipeline — standalone implementation.
 *
 * This header implements the key YOLOv5 post-processing steps:
 *   1. YOLO output tensor format parsing
 *   2. Bounding box decoding (from network outputs to pixel coords)
 *   3. NMS (Non-Maximum Suppression)
 *   4. Coordinate scaling (letterbox reverse transform)
 *
 * YOLOv5 output format (per batch item):
 *   [1, elements, num_info]  where
 *   elements = num_anchors * anchor_h * anchor_w  (all scales combined)
 *   num_info = 4 (bbox) + 1 (objectness) + num_classes
 *
 * Each detection element:
 *   [cx, cy, w, h, obj_conf, cls0_prob, cls1_prob, ...]
 *
 * After sigmoid (in the network), all values are in [0, 1].
 * The detect layer decodes these into real bounding boxes.
 */

#pragma once

#include "tensor.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>

namespace learn_infer {

/**
 * A single detection result.
 */
struct Detection {
  int x, y, width, height;   // Bounding box in original image coords
  float confidence;           // obj_conf * class_prob
  int class_id;               // Predicted class index
};

/**
 * Decode raw YOLO output values into a bounding box.
 *
 * The network outputs (after sigmoid) are normalized:
 *   cx, cy ∈ [0, 1]  relative to the feature map grid cell
 *   w, h ∈ [0, 1]    scaled by anchor dimensions
 *
 * This function converts them to absolute coordinates on the
 * scaled input image (e.g. 640×640 after letterbox).
 *
 * @param cx         Center x from network output [0, 1]
 * @param cy         Center y from network output [0, 1]
 * @param w          Width from network output [0, 1]
 * @param h          Height from network output [0, 1]
 * @param grid_w     Width of the feature map grid
 * @param grid_h     Height of the feature map grid
 * @param anchor_w   Anchor box width (on scaled image)
 * @param anchor_h   Anchor box height (on scaled image)
 * @param stride     Stride of this detection scale
 * @param image_w    Scaled image width (e.g. 640)
 * @param image_h    Scaled image height (e.g. 640)
 * @param out_box    Output [center_x, center_y, width, height] in pixels
 */
inline void DecodeBox(float cx, float cy, float w, float h,
                      int grid_w, int grid_h,
                      float anchor_w, float anchor_h,
                      int stride,
                      int image_w, int image_h,
                      std::vector<float>& out_box) {
  // YOLOv5 decoding:
  // xy = (sigmoid(xy) * 2.0 - 0.5 + grid_coords) * stride
  // wh = (sigmoid(wh) * 2.0) ^ 2 * anchor_sizes
  //
  // Since sigmoid is already applied, cx/cy/w/h are in [0,1].

  // Box center in pixel coords
  float out_cx = (cx * 2.0f - 0.5f) * stride + stride * 0.5f;
  float out_cy = (cy * 2.0f - 0.5f) * stride + stride * 0.5f;

  // Box size in pixel coords
  float out_w = std::pow(w * 2.0f, 2.0f) * anchor_w;
  float out_h = std::pow(h * 2.0f, 2.0f) * anchor_h;

  out_box[0] = out_cx;
  out_box[1] = out_cy;
  out_box[2] = out_w;
  out_box[3] = out_h;
}

/**
 * Clip value to [lower, upper].
 */
template <typename T>
inline T Clip(T val, T lower, T upper) {
  return std::max(lower, std::min(val, upper));
}

/**
 * Scale bounding box coordinates from the letterboxed image back
 * to the original image size.
 *
 * @param box            Box [x, y, w, h] on the scaled image (in-place update)
 * @param scaled_w       Width of the letterboxed image (e.g. 640)
 * @param scaled_h       Height of the letterboxed image (e.g. 640)
 * @param orig_w         Original image width
 * @param orig_h         Original image height
 * @param gain           Scale factor (how much the image was shrunk)
 * @param pad_x          Horizontal padding added by letterbox
 * @param pad_y          Vertical padding added by letterbox
 */
inline void ScaleBox(std::vector<float>& box,
                     int scaled_w, int scaled_h,
                     int orig_w, int orig_h,
                     float gain, int pad_x, int pad_y) {
  // Remove padding and scale down
  box[0] = (box[0] - pad_x) / gain;  // x
  box[1] = (box[1] - pad_y) / gain;  // y
  box[2] = box[2] / gain;            // w
  box[3] = box[3] / gain;            // h

  // Convert center-x, center-y to top-left
  float left = box[0] - box[2] / 2.0f;
  float top = box[1] - box[3] / 2.0f;

  // Clip to image bounds
  left = Clip(left, 0.0f, static_cast<float>(orig_w));
  top = Clip(top, 0.0f, static_cast<float>(orig_h));
  box[2] = Clip(box[2], 0.0f, static_cast<float>(orig_w));
  box[3] = Clip(box[3], 0.0f, static_cast<float>(orig_h));

  box[0] = left;
  box[1] = top;
}

/**
 * Compute Intersection over Union (IoU) between two boxes.
 *
 * Boxes are [left, top, width, height].
 */
inline float ComputeIoU(const std::vector<float>& box_a,
                        const std::vector<float>& box_b) {
  // Intersection rectangle
  int left_a = static_cast<int>(box_a[0]);
  int top_a = static_cast<int>(box_a[1]);
  int right_a = left_a + static_cast<int>(box_a[2]);
  int bottom_a = top_a + static_cast<int>(box_a[3]);

  int left_b = static_cast<int>(box_b[0]);
  int top_b = static_cast<int>(box_b[1]);
  int right_b = left_b + static_cast<int>(box_b[2]);
  int bottom_b = top_b + static_cast<int>(box_b[3]);

  int inter_left = std::max(left_a, left_b);
  int inter_top = std::max(top_a, top_b);
  int inter_right = std::min(right_a, right_b);
  int inter_bottom = std::min(bottom_a, bottom_b);

  int inter_w = std::max(0, inter_right - inter_left);
  int inter_h = std::max(0, inter_bottom - inter_top);
  int inter_area = inter_w * inter_h;

  int area_a = (right_a - left_a) * (bottom_a - top_a);
  int area_b = (right_b - left_b) * (bottom_b - top_b);

  float union_area = area_a + area_b - inter_area;
  return union_area > 0 ? static_cast<float>(inter_area) / union_area : 0.0f;
}

/**
 * Non-Maximum Suppression (NMS).
 *
 * Given a list of detections, remove those whose bounding box
 * overlaps too much (IoU > threshold) with a higher-confidence box.
 *
 * @param detections     Input detections (sorted by confidence desc)
 * @param iou_threshold  Boxes with IoU above this are suppressed
 * @return               Filtered detection list
 */
inline std::vector<Detection> NMS(
    std::vector<Detection>& detections, float iou_threshold = 0.45f) {
  // Sort by confidence (descending)
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
              return a.confidence > b.confidence;
            });

  std::vector<bool> suppressed(detections.size(), false);
  std::vector<Detection> result;

  for (size_t i = 0; i < detections.size(); i++) {
    if (suppressed[i]) continue;

    result.push_back(detections[i]);

    // Build box for IoU comparison [left, top, w, h]
    std::vector<float> box_i{
        static_cast<float>(detections[i].x),
        static_cast<float>(detections[i].y),
        static_cast<float>(detections[i].width),
        static_cast<float>(detections[i].height)
    };

    for (size_t j = i + 1; j < detections.size(); j++) {
      if (suppressed[j]) continue;

      std::vector<float> box_j{
          static_cast<float>(detections[j].x),
          static_cast<float>(detections[j].y),
          static_cast<float>(detections[j].width),
          static_cast<float>(detections[j].height)
      };

      float iou = ComputeIoU(box_i, box_j);
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

/**
 * Parse YOLO output tensor into detections.
 *
 * The output tensor has shape [1, elements, 4 + 1 + num_classes].
 * Each element is [cx, cy, w, h, obj_conf, cls0, cls1, ...].
 * These values have already passed through sigmoid in the network.
 *
 * @param output         YOLO output tensor
 * @param num_classes    Number of object classes (e.g. 80 for COCO)
 * @param conf_threshold Minimum confidence to keep a detection
 * @return               Raw detections (before NMS)
 */
inline std::vector<Detection> ParseYOLOOutput(
    const std::shared_ptr<Tensor<float>>& output,
    int num_classes, float conf_threshold = 0.25f) {
  auto shapes = output->shapes();
  assert(shapes.size() == 3 || shapes.size() == 1);

  uint32_t batch = 1;
  uint32_t elements = shapes[0];
  uint32_t num_info = shapes.size() == 3 ? shapes[1] * shapes[2] : shapes[1];

  // If 3D [1, E, N], batch=shapes[0]
  if (shapes.size() == 3) {
    batch = shapes[0];
    elements = shapes[1];
    num_info = shapes[2];
  }

  assert(num_info >= 5);  // At least cx, cy, w, h, obj_conf
  int cls_offset = 4;  // Index where class probs start

  std::vector<Detection> detections;

  for (uint32_t b = 0; b < batch; b++) {
    for (uint32_t e = 0; e < elements; e++) {
      // Objectness confidence
      float obj_conf = output->at(b, e, cls_offset);
      if (obj_conf < conf_threshold) continue;

      // Find best class
      float best_cls_prob = -1.0f;
      int best_cls_id = 0;
      for (int c = 0; c < num_classes; c++) {
        float cls_prob = output->at(b, e, cls_offset + 1 + c);
        if (cls_prob > best_cls_prob) {
          best_cls_prob = cls_prob;
          best_cls_id = c;
        }
      }

      float confidence = obj_conf * best_cls_prob;
      if (confidence < conf_threshold) continue;

      // Bounding box coordinates (already in pixel coords from YoloDetect layer)
      Detection det;
      det.width = static_cast<int>(output->at(b, e, 2));
      det.height = static_cast<int>(output->at(b, e, 3));
      det.x = static_cast<int>(output->at(b, e, 0)) - det.width / 2;
      det.y = static_cast<int>(output->at(b, e, 1)) - det.height / 2;
      det.confidence = confidence;
      det.class_id = best_cls_id;

      detections.push_back(det);
    }
  }

  return detections;
}

/**
 * Print detections in a human-readable format.
 */
inline void PrintDetections(const std::vector<Detection>& detections,
                            const std::vector<std::string>* class_names = nullptr) {
  std::cout << "  Found " << detections.size() << " detection(s):\n";
  for (const auto& det : detections) {
    std::cout << "    ["
              << (class_names && det.class_id < (int)class_names->size()
                      ? class_names->at(det.class_id)
                      : "cls" + std::to_string(det.class_id))
              << "]  conf=" << std::fixed << std::setprecision(2) << det.confidence
              << "  box=(" << det.x << ", " << det.y << ", "
              << det.width << "x" << det.height << ")\n";
  }
}

}  // namespace learn_infer
