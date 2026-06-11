/**
 * day5/include/layer/hardswish.hpp
 *
 * HardSwishLayer -- HardSwish activation layer.
 *
 * Implements: output = x * hard_sigmoid(x)
 * where hard_sigmoid is the piecewise-linear approximation.
 *
 *   hard_sigmoid(x) = 0            for x <= -3
 *                    = 1            for x >=  3
 *                    = (x + 3)/6    for -3 < x < 3
 *
 * HardSwish: x * hard_sigmoid(x)
 *
 * Operates element-wise via Tensor::transform.
 *
 * Auto-registers itself as "nn.HardSwish" through a static
 * LayerRegistererWrapper instance.
 */

#ifndef HARDSWISH_HPP
#define HARDSWISH_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

namespace learn_infer {

/**
 * HardSwishLayer<T> -- HardSwish activation.
 *
 * output = x * hard_sigmoid(x)
 *
 * where hard_sigmoid(x) =
 *     0            for x <= -3
 *     1            for x >=  3
 *     (x + 3)/6    for -3 < x < 3
 *
 * This is the cheap approximation of Swish (x * sigmoid(x)), used in
 * MobileNetV3. The bounds are shifted to [-3, 3] compared to
 * HardSigmoid's [-2.5, 2.5] because HardSwish applies a shifted
 * hard-sigmoid internally.
 */
template <typename T>
class HardSwishLayer : public Layer<T> {
 public:
  using value_type = T;

  HardSwishLayer() = default;
  ~HardSwishLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    auto out = inputs[0]->clone();
    out->transform([](T x) {
      // hard_sigmoid for hardswish: clamped to [0,1] with range [-3,3]
      T hs;
      if (x <= T{-3}) {
        hs = T{0};
      } else if (x >= T{3}) {
        hs = T{1};
      } else {
        hs = (x + T{3}) / T{6};
      }
      return x * hs;
    });

    outputs.clear();
    outputs.push_back(std::move(out));
    return StatusCode::kSuccess;
  }

  std::string Type() const override {
    return "nn.HardSwish";
  }
};

static LayerRegistererWrapper<HardSwishLayer<float>>
    g_hardswish_registrar("nn.HardSwish");

}  // namespace learn_infer

#endif  // HARDSWISH_HPP
