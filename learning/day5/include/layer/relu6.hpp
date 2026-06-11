/**
 * day5/include/layer/relu6.hpp
 *
 * Relu6Layer -- ReLU6 activation layer.
 *
 * Implements the clipped ReLU: output = min(max(0, x), 6).
 * Operates element-wise via Tensor::transform.
 *
 * Auto-registers itself as "nn.ReLU6" through a static
 * LayerRegistererWrapper instance.
 */

#ifndef RELU6_HPP
#define RELU6_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

namespace learn_infer {

/**
 * Relu6Layer<T> -- ReLU6 activation.
 *
 * output = min(max(0, x), 6)
 *
 * Clips values to the range [0, 6]. Useful for quantization-aware
 * networks where bounded activations improve fixed-point accuracy.
 */
template <typename T>
class Relu6Layer : public Layer<T> {
 public:
  using value_type = T;

  Relu6Layer() = default;
  ~Relu6Layer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto out = inputs[0]->clone();
    out->transform([](T x) { return std::min(std::max(T{0}, x), T{6}); });

    outputs.clear();
    outputs.push_back(std::move(out));
    return StatusCode::kSuccess;
  }

  std::string Type() const override {
    return "nn.ReLU6";
  }
};

static LayerRegistererWrapper<Relu6Layer<float>>
    g_relu6_registrar("nn.ReLU6");

}  // namespace learn_infer

#endif  // RELU6_HPP
