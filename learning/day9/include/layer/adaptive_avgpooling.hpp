/**
 * day8/include/layer/adaptive_avgpooling.hpp
 *
 * AdaptiveAvgPoolingLayer -- Adaptive average pooling for 2D feature maps.
 *
 * Takes desired output height and width as parameters. Dynamically computes
 * stride and kernel size from input dimensions:
 *
 *   stride_h = input_h / output_h
 *   stride_w = input_w / output_w
 *   kernel_h = input_h - (output_h - 1) * stride_h
 *   kernel_w = input_w - (output_w - 1) * stride_w
 *
 * Average over the adaptive window at each output position.
 * Registered as "nn.AdaptiveAvgPool2d".
 */

#ifndef ADAPTIVE_AVGPOOLING_HPP
#define ADAPTIVE_AVGPOOLING_HPP

#include "layer.hpp"
#include "layer_factory.hpp"

namespace learn_infer {

/**
 * AdaptiveAvgPoolingLayer<T> -- Adaptive 2D average pooling.
 *
 * Unlike standard pooling, the kernel size and stride are computed
 * from the input dimensions and the desired output dimensions.
 * This ensures the output always has the specified size regardless
 * of input size.
 */
template <typename T>
class AdaptiveAvgPoolingLayer : public Layer<T> {
 public:
  using value_type = T;

  explicit AdaptiveAvgPoolingLayer(uint32_t output_h, uint32_t output_w)
      : output_h_(output_h), output_w_(output_w) {}

  ~AdaptiveAvgPoolingLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    outputs.clear();
    outputs.resize(inputs.size());

    for (size_t b = 0; b < inputs.size(); b++) {
      const auto& input = inputs[b];
      uint32_t C = input->channels();
      uint32_t input_h = input->rows();
      uint32_t input_w = input->cols();

      // Compute adaptive stride and kernel size
      uint32_t stride_h = input_h / output_h_;
      uint32_t stride_w = input_w / output_w_;
      uint32_t kernel_h = input_h - (output_h_ - 1) * stride_h;
      uint32_t kernel_w = input_w - (output_w_ - 1) * stride_w;

      auto out = std::make_shared<Tensor<T>>(C, output_h_, output_w_);

      for (uint32_t c = 0; c < C; c++) {
        for (uint32_t oh = 0; oh < output_h_; oh++) {
          for (uint32_t ow = 0; ow < output_w_; ow++) {
            T sum = T{0};
            uint32_t count = 0;

            // Compute the starting position for this output cell
            uint32_t h_start = oh * stride_h;
            uint32_t w_start = ow * stride_w;

            // For the last output cell, extend kernel to cover remaining input
            uint32_t cur_kernel_h = kernel_h;
            uint32_t cur_kernel_w = kernel_w;
            if (oh == output_h_ - 1) {
              cur_kernel_h = input_h - h_start;
            }
            if (ow == output_w_ - 1) {
              cur_kernel_w = input_w - w_start;
            }

            for (uint32_t kh = 0; kh < cur_kernel_h; kh++) {
              for (uint32_t kw = 0; kw < cur_kernel_w; kw++) {
                uint32_t ih = h_start + kh;
                uint32_t iw = w_start + kw;
                if (ih < input_h && iw < input_w) {
                  sum += input->at(c, ih, iw);
                  count++;
                }
              }
            }

            out->at(c, oh, ow) = count > 0 ? sum / static_cast<T>(count) : T{0};
          }
        }
      }

      outputs[b] = std::move(out);
    }

    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.AdaptiveAvgPool2d"; }

  uint32_t output_h() const { return output_h_; }
  uint32_t output_w() const { return output_w_; }

 private:
  uint32_t output_h_, output_w_;
};

/**
 * AdaptiveAvgPoolingRegister -- register a parameterized AdaptiveAvgPool with the factory.
 */
template <typename T>
Layer<T>* AdaptiveAvgPoolingRegister(
    const std::string& type_name,
    uint32_t output_h, uint32_t output_w) {
  struct Params {
    uint32_t output_h, output_w;
  };
  auto params = std::make_shared<Params>(Params{output_h, output_w});

  LayerRegisterer<T>::RegisterCreator(
      type_name,
      [params]() -> Layer<T>* {
        return new AdaptiveAvgPoolingLayer<T>(
            params->output_h, params->output_w);
      });

  return new AdaptiveAvgPoolingLayer<T>(output_h, output_w);
}

}  // namespace learn_infer

#endif  // ADAPTIVE_AVGPOOLING_HPP
