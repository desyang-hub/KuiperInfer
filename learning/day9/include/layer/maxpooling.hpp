/**
 * day8/include/layer/maxpooling.hpp
 *
 * MaxPoolingLayer -- Max pooling for 2D feature maps.
 *
 * Implements the standard max pooling operation: for each spatial
 * position, find the maximum value in the kernel-sized window.
 *
 * Supports kernel size, stride, and padding.
 * Registered as "nn.MaxPool2d".
 */

#ifndef MAXPOOLING_HPP
#define MAXPOOLING_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <algorithm>

namespace learn_infer {

/**
 * MaxPoolingLayer<T> -- 2D max pooling.
 *
 * Sliding window max operation over [kernel_h, kernel_w] windows
 * with configurable stride and padding.
 */
template <typename T>
class MaxPoolingLayer : public Layer<T> {
 public:
  using value_type = T;

  MaxPoolingLayer(uint32_t kernel_h, uint32_t kernel_w,
                  uint32_t stride_h, uint32_t stride_w,
                  uint32_t pad_h, uint32_t pad_w)
      : kernel_h_(kernel_h), kernel_w_(kernel_w),
        stride_h_(stride_h), stride_w_(stride_w),
        pad_h_(pad_h), pad_w_(pad_w) {}

  ~MaxPoolingLayer() override = default;

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
      uint32_t H = input->rows() + 2 * pad_h_;
      uint32_t W = input->cols() + 2 * pad_w_;

      uint32_t out_h = (H - kernel_h_) / stride_h_ + 1;
      uint32_t out_w = (W - kernel_w_) / stride_w_ + 1;

      auto out = std::make_shared<Tensor<T>>(C, out_h, out_w);

      for (uint32_t c = 0; c < C; c++) {
        for (uint32_t oh = 0; oh < out_h; oh++) {
          for (uint32_t ow = 0; ow < out_w; ow++) {
            T max_val = (std::numeric_limits<T>::min)();

            for (uint32_t kh = 0; kh < kernel_h_; kh++) {
              for (uint32_t kw = 0; kw < kernel_w_; kw++) {
                uint32_t ih = oh * stride_h_ + kh - pad_h_;
                uint32_t iw = ow * stride_w_ + kw - pad_w_;

                // Handle padding: if out of bounds, skip (treat as -inf)
                if (ih < input->rows() && iw < input->cols()) {
                  T val = input->at(c, ih, iw);
                  if (val > max_val) {
                    max_val = val;
                  }
                }
              }
            }

            out->at(c, oh, ow) = max_val;
          }
        }
      }

      outputs[b] = std::move(out);
    }

    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.MaxPool2d"; }

  uint32_t kernel_h() const { return kernel_h_; }
  uint32_t kernel_w() const { return kernel_w_; }
  uint32_t stride_h() const { return stride_h_; }
  uint32_t stride_w() const { return stride_w_; }
  uint32_t pad_h() const { return pad_h_; }
  uint32_t pad_w() const { return pad_w_; }

 private:
  uint32_t kernel_h_, kernel_w_;
  uint32_t stride_h_, stride_w_;
  uint32_t pad_h_, pad_w_;
};

/**
 * MaxPoolingRegister -- register a parameterized MaxPooling with the factory.
 */
template <typename T>
Layer<T>* MaxPoolingRegister(
    const std::string& type_name,
    uint32_t kernel_h, uint32_t kernel_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w) {
  struct Params {
    uint32_t kernel_h, kernel_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
  };
  auto params = std::make_shared<Params>(Params{
      kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w});

  LayerRegisterer<T>::RegisterCreator(
      type_name,
      [params]() -> Layer<T>* {
        return new MaxPoolingLayer<T>(
            params->kernel_h, params->kernel_w,
            params->stride_h, params->stride_w,
            params->pad_h, params->pad_w);
      });

  return new MaxPoolingLayer<T>(
      kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
}

}  // namespace learn_infer

#endif  // MAXPOOLING_HPP
