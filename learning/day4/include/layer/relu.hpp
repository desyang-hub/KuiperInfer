/**
 * day4/include/layer/relu.hpp
 *
 * ReluLayer -- ReLU activation layer.
 *
 * Implements the Rectified Linear Unit: output = max(0, input).
 * Operates element-wise via Tensor::transform.
 *
 * Auto-registers itself as "nn.ReLU" through a static
 * LayerRegistererWrapper instance.
 */

#ifndef RELU_HPP
#define RELU_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

namespace learn_infer {

/**
 * ReluLayer<T> -- ReLU activation.
 *
 * Requires exactly one input tensor; produces exactly one output tensor
 * of the same shape with all negative values clipped to zero.
 */
template <typename T>
class ReluLayer : public Layer<T> {
 public:
  // Required by LayerRegistererWrapper so it can deduce T from L.
  using value_type = T;

  ReluLayer() = default;
  ~ReluLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    // Clone the input and apply ReLU transform
    auto out = inputs[0]->clone();
    out->transform([](T x) { return std::max(T{0}, x); });

    outputs.clear();
    outputs.push_back(std::move(out));
    return StatusCode::kSuccess;
  }

  std::string Type() const override {
    return "nn.ReLU";
  }
};

// Auto-registration: creates a static wrapper that registers ReluLayer<float>
// under the type name "nn.ReLU" at program startup.
static LayerRegistererWrapper<ReluLayer<float>>
    g_relu_registrar("nn.ReLU");

}  // namespace learn_infer

#endif  // RELU_HPP
