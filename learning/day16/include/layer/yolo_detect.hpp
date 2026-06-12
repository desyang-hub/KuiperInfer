/**
 * day15/include/layer/yolo_detect.hpp
 *
 * YoloDetect Layer — decodes YOLO output tensors into bounding boxes.
 *
 * In a real YOLOv5 model, the final layer is "op_detect" which:
 *   1. Applies sigmoid to the raw output
 *   2. Decodes each anchor's box using grid coords + anchors
 *   3. Produces [1, elements, 4 + 1 + num_classes] output
 *
 * This implementation handles step 2 (box decoding). Sigmoid is a
 * separate layer (nn.Sigmoid). Step 3 (flattening) is handled by
 * Flatten/View layers.
 */

#pragma once

#include "layer.hpp"
#include "layer_factory.hpp"
#include "yolo_detect.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

namespace learn_infer {

/**
 * YoloDetect layer for YOLOv5.
 *
 * Input tensor shape: [batch, num_anchors, 6 + num_classes, 1, 1]
 *   — 6 = 4 bbox + 1 obj + (num_classes class probs)
 *   — num_anchors = anchors across all 3 scales combined
 *
 * The layer decodes the bbox coordinates using anchor boxes and strides.
 * Output shape: [batch, num_anchors, 4 + 1 + num_classes, 1, 1]
 *   — bbox is now in absolute pixel coords
 */
class YoloDetectLayer final : public Layer<float> {
 public:
  /**
   * @param anchors      List of anchor box dimensions {w1, h1, w2, h2, ...}
   * @param stride       Stride value for this detection scale
   * @param grid_w       Grid width for this scale
   * @param grid_h       Grid height for this scale
   * @param img_w        Target image width (e.g. 640)
   * @param img_h        Target image height (e.g. 640)
   * @param num_classes  Number of object classes
   */
  YoloDetectLayer(const std::vector<float>& anchors,
                  int stride, int grid_w, int grid_h,
                  int img_w, int img_h, int num_classes = 80)
      : anchors_(anchors), stride_(stride), grid_w_(grid_w), grid_h_(grid_h),
        img_w_(img_w), img_h_(img_h), num_classes_(num_classes) {
    assert(anchors_.size() % 2 == 0);
  }

  void Forward(const std::vector<STensor>& inputs,
               std::vector<STensor>&& outputs) override {
    auto& input = inputs[0];
    auto input_shapes = input->shapes();

    // Input: [batch, num_anchors, num_info, 1, 1]
    uint32_t batch = input_shapes[0];
    uint32_t num_anchors = input_shapes[1];
    uint32_t num_info = input_shapes[2];

    int cls_offset = 5;  // 4 bbox + 1 obj

    // Output has same shape
    std::vector<uint32_t> out_shapes = input_shapes;
    auto out = std::make_shared<Tensor<float>>(out_shapes);

    int num_anchors_per_scale = anchors_.size() / 2;

    for (uint32_t b = 0; b < batch; b++) {
      for (uint32_t a = 0; a < num_anchors; a++) {
        // Copy class probs (unchanged)
        for (int c = 0; c < num_classes_; c++) {
          out->at(b, a, cls_offset, 0, 0) = input->at(b, a, cls_offset, 0, 0);
        }
        // Copy objectness (unchanged)
        out->at(b, a, 4, 0, 0) = input->at(b, a, 4, 0, 0);

        // Decode bounding box
        // Anchor index for this detection
        int anchor_idx = a % num_anchors_per_scale;
        float anchor_w = anchors_[anchor_idx * 2];
        float anchor_h = anchors_[anchor_idx * 2 + 1];

        float cx = input->at(b, a, 0, 0, 0);
        float cy = input->at(b, a, 1, 0, 0);
        float w = input->at(b, a, 2, 0, 0);
        float h = input->at(b, a, 3, 0, 0);

        std::vector<float> decoded(4);
        DecodeBox(cx, cy, w, h,
                  grid_w_, grid_h_,
                  anchor_w, anchor_h,
                  stride_,
                  img_w_, img_h_,
                  decoded);

        out->at(b, a, 0, 0, 0) = decoded[0];  // cx in pixels
        out->at(b, a, 1, 0, 0) = decoded[1];  // cy in pixels
        out->at(b, a, 2, 0, 0) = decoded[2];  // w in pixels
        out->at(b, a, 3, 0, 0) = decoded[3];  // h in pixels
      }
    }

    outputs[0] = out;
  }

  std::string Type() const override { return "op_detect"; }
};

}  // namespace learn_infer
