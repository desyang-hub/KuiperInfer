/**
 * day5/include/layer/sigmoid.hpp
 *
 * SigmoidLayer -- Sigmoid activation layer.
 *
 * Implements the logistic sigmoid function: output = 1 / (1 + exp(-x)).
 * Operates element-wise via Tensor::transform.
 *
 * Auto-registers itself as "nn.Sigmoid" through a static
 * LayerRegistererWrapper instance.
 */

#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

namespace learn_infer {

/**
 * SigmoidLayer<T> -- Sigmoid activation.
 *
 * output = 1 / (1 + exp(-x))
 *
 * Maps any real value to the range (0, 1).
 */
template <typename T>
class SigmoidLayer : public Layer<T> {
 public:
  using value_type = T;

  SigmoidLayer() = default;
  ~SigmoidLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto out = inputs[0]->clone();
    out->transform([](T x) { return T{1} / (T{1} + std::exp(-x)); });

    outputs.clear();
    outputs.push_back(std::move(out));
    return StatusCode::kSuccess;
  }

  std::string Type() const override {
    return "nn.Sigmoid";
  }
};

static LayerRegistererWrapper<SigmoidLayer<float>>
    g_sigmoid_registrar("nn.Sigmoid");

}  // namespace learn_infer

#endif  // SIGMOID_HPP
