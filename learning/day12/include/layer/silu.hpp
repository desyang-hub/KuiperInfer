/**
 * day5/include/layer/silu.hpp
 *
 * SiluLayer -- SiLU (Sigmoid Linear Unit) activation layer.
 *
 * Implements: output = x * sigmoid(x) = x / (1 + exp(-x))
 * Also known as Swish with beta=1.
 * Operates element-wise via Tensor::transform.
 *
 * Auto-registers itself as "nn.SiLU" through a static
 * LayerRegistererWrapper instance.
 */

#ifndef SILU_HPP
#define SILU_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

namespace learn_infer {

/**
 * SiluLayer<T> -- SiLU (Sigmoid Linear Unit) activation.
 *
 * output = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * A smooth, non-monotonic activation that often outperforms ReLU in
 * deep networks. Introduced in "Gaussian Error Linear Units" (GELUs)
 * follow-up work.
 */
template <typename T>
class SiluLayer : public Layer<T> {
 public:
  using value_type = T;

  SiluLayer() = default;
  ~SiluLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto out = inputs[0]->clone();
    out->transform([](T x) {
      return x / (T{1} + std::exp(-x));
    });

    outputs.clear();
    outputs.push_back(std::move(out));
    return StatusCode::kSuccess;
  }

  std::string Type() const override {
    return "nn.SiLU";
  }
};

static LayerRegistererWrapper<SiluLayer<float>>
    g_silu_registrar("nn.SiLU");

}  // namespace learn_infer

#endif  // SILU_HPP
