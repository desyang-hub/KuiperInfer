/**
 * day15/include/layer/flatten.hpp
 *
 * Flatten layer — reshapes a tensor into a 1D vector.
 *
 * In YOLOv5, after the backbone + FPN + PAN, the feature maps at each
 * detection head are flattened before being fed to the detect layer.
 *
 * Also implements a simple "View/Reshape" layer.
 *
 * Registered: "nn.Flatten", "nn.View"
 */

#pragma once

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cassert>

namespace learn_infer {

/**
 * FlattenLayer — collapses all dimensions into a single 1D tensor.
 *
 * Input:  [C, H, W]  →  Output:  [1, 1, C*H*W]
 */
class FlattenLayer : public Layer<float> {
 public:
  using value_type = float;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto inp = inputs[0];
    uint64_t total = inp->size();
    auto out = std::make_shared<Tensor<float>>(1, 1, static_cast<uint32_t>(total));

    // Copy data (already contiguous in row-major)
    const float* pin = inp->data();
    float* pout = out->data();
    for (uint64_t i = 0; i < total; i++) {
      pout[i] = pin[i];
    }

    outputs.clear();
    outputs.push_back(out);
    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.Flatten"; }
};

static LayerRegistererWrapper<FlattenLayer> g_flatten("nn.Flatten");

/**
 * ViewLayer — reshapes tensor to given target shape.
 *
 * Constructor takes target [C, H, W]. Total element count must match.
 */
class ViewLayer : public Layer<float> {
 public:
  using value_type = float;

  ViewLayer(uint32_t c, uint32_t h, uint32_t w)
      : target_c_(c), target_h_(h), target_w_(w) {}

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto inp = inputs[0];
    uint64_t target_size = static_cast<uint64_t>(target_c_) * target_h_ * target_w_;
    assert(inp->size() == target_size && "ViewLayer: element count mismatch");

    auto out = std::make_shared<Tensor<float>>(
        target_c_, target_h_, target_w_);

    const float* pin = inp->data();
    float* pout = out->data();
    for (uint64_t i = 0; i < target_size; i++) {
      pout[i] = pin[i];
    }

    outputs.clear();
    outputs.push_back(out);
    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.View"; }

 private:
  uint32_t target_c_;
  uint32_t target_h_;
  uint32_t target_w_;
};

// Note: ViewLayer requires constructor args, so it can't use the simple
// factory (which creates with zero args). We'll create it manually.

}  // namespace learn_infer
