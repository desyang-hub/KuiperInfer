/**
 * day5/include/layer/hardsigmoid.hpp
 *
 * HardSigmoidLayer -- HardSigmoid activation layer.
 *
 * Implements the piecewise-linear approximation of sigmoid:
 *   output = 0                    if x <= -2.5
 *   output = 1                    if x >=  2.5
 *   output = x/6 + 0.5            otherwise
 *
 * Operates element-wise via Tensor::transform.
 *
 * Auto-registers itself as "nn.HardSigmoid" through a static
 * LayerRegistererWrapper instance.
 */

#ifndef HARDSIGMOID_HPP
#define HARDSIGMOID_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

namespace learn_infer {

/**
 * HardSigmoidLayer<T> -- HardSigmoid activation.
 *
 * A fast, piecewise-linear approximation of the sigmoid function.
 * Commonly used in mobile/edge models (MobileNetV3) where exp() is
 * too expensive.
 *
 *   output = 0            for x <= -2.5
 *   output = 1            for x >=  2.5
 *   output = x/6 + 0.5    for -2.5 < x < 2.5
 */
template <typename T>
class HardSigmoidLayer : public Layer<T> {
 public:
  using value_type = T;

  HardSigmoidLayer() = default;
  ~HardSigmoidLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto out = inputs[0]->clone();
    out->transform([](T x) {
      if (x <= T{-2.5}) return T{0};
      if (x >= T{2.5})  return T{1};
      return x / T{6} + T{0.5};
    });

    outputs.clear();
    outputs.push_back(std::move(out));
    return StatusCode::kSuccess;
  }

  std::string Type() const override {
    return "nn.HardSigmoid";
  }
};

static LayerRegistererWrapper<HardSigmoidLayer<float>>
    g_hardsigmoid_registrar("nn.HardSigmoid");

}  // namespace learn_infer

#endif  // HARDSIGMOID_HPP
