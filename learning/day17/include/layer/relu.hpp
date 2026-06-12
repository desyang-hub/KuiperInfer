/**
 * day15/include/layer/relu.hpp
 *
 * ReLU (Rectified Linear Unit) activation layer.
 *
 * relu(x) = max(0, x)
 *
 * Applied element-wise. Used extensively in YOLOv5's backbone
 * (CSPDarknet) after each convolution + batch norm.
 *
 * Registered: "nn.ReLU"
 */

#pragma once

#include "layer.hpp"
#include "layer_factory.hpp"
#include <algorithm>

namespace learn_infer {

/**
 * ReLU activation layer.
 *
 * Input and output tensors have the same shape.
 * Output is a new tensor (inputs are not modified).
 */
class ReluLayer : public Layer<float> {
 public:
  using value_type = float;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto inp = inputs[0];
    auto out = std::make_shared<Tensor<float>>(
        inp->channels(), inp->rows(), inp->cols());

    const float* pin = inp->data();
    float* pout = out->data();
    uint64_t n = inp->size();

    for (uint64_t i = 0; i < n; i++) {
      pout[i] = std::max(0.0f, pin[i]);
    }

    outputs.clear();
    outputs.push_back(out);
    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.ReLU"; }
};

static LayerRegistererWrapper<ReluLayer> g_relu("nn.ReLU");

}  // namespace learn_infer
