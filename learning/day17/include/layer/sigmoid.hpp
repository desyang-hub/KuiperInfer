/**
 * day15/include/layer/sigmoid.hpp
 *
 * Sigmoid activation layer — element-wise sigmoid(x) = 1 / (1 + exp(-x))
 *
 * In YOLOv5, sigmoid is applied to the raw detection outputs to constrain:
 *   - bbox coordinates (x, y, w, h) to [0, 1] range
 *   - objectness confidence to [0, 1]
 *   - per-class probabilities to [0, 1]
 *
 * This layer is registered with the factory under "nn.Sigmoid".
 */

#pragma once

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

namespace learn_infer {

/**
 * Sigmoid activation layer.
 *
 * Applies sigmoid(x) = 1 / (1 + exp(-x)) to every element.
 * Input and output tensors have the same shape.
 */
class SigmoidLayer : public Layer<float> {
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
      pout[i] = 1.0f / (1.0f + std::exp(-pin[i]));
    }

    outputs.clear();
    outputs.push_back(out);
    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.Sigmoid"; }
};

// Auto-register with the factory
static LayerRegistererWrapper<SigmoidLayer> g_sigmoid("nn.Sigmoid");

}  // namespace learn_infer
