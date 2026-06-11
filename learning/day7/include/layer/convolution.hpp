/**
 * day7/include/layer/convolution.hpp
 *
 * ConvolutionLayer -- 2D convolution via im2col + GEMM optimization.
 *
 * Key concepts:
 *   1. im2col: transform input spatial regions into columns of a matrix
 *      For input [C_in, H, W] and kernel [C_out, C_in/G, kh, kw],
 *      im2col output shape is [C_in/G * kh * kw, H_out * W_out]
 *
 *   2. Weight reshape: kernel weights from [C_out, C_in/G, kh, kw]
 *      reshaped to [C_out, C_in/G * kh * kw]
 *
 *   3. GEMM: output_matrix = weight_matrix x im2col_matrix
 *      output_matrix shape: [C_out, H_out * W_out]
 *      Reshape back to [C_out, H_out, W_out]
 *
 *   4. Simple manual GEMM without BLAS:
 *      for (c_out) for (h_out*w_out) for (accum) { C += A*B }
 *
 * Registered as "nn.Conv2d".
 */

#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>
#include <algorithm>

namespace learn_infer {

/**
 * ConvolutionLayer<T> -- 2D convolution using im2col + GEMM.
 *
 * Weight layout (NCHW): [C_out, C_in/groups, kernel_h, kernel_w]
 * Bias layout: [C_out]
 *
 * Constructor parameters define the layer's shape and weights.
 * Forward takes a batch of input tensors and returns a batch of
 * output tensors.
 */
template <typename T>
class ConvolutionLayer : public Layer<T> {
 public:
  using value_type = T;

  /**
   * Constructor.
   *
   * @param ch        Output channels (C_out)
   * @param rows      Input height
   * @param cols      Input width
   * @param kernel_h  Kernel height
   * @param kernel_w  Kernel width
   * @param stride_h  Stride height
   * @param stride_w  Stride width
   * @param pad_h     Padding height
   * @param pad_w     Padding width
   * @param groups    Number of groups (1 = standard conv)
   * @param weights   Weight data in [C_out, C_in/G, kh, kw] order
   * @param bias      Bias data in [C_out] order
   */
  ConvolutionLayer(
      uint32_t ch, uint32_t rows, uint32_t cols,
      uint32_t kernel_h, uint32_t kernel_w,
      uint32_t stride_h, uint32_t stride_w,
      uint32_t pad_h, uint32_t pad_w,
      uint32_t groups,
      const std::vector<T>& weights,
      const std::vector<T>& bias)
      : out_channels_(ch),
        in_rows_(rows), in_cols_(cols),
        kernel_h_(kernel_h), kernel_w_(kernel_w),
        stride_h_(stride_h), stride_w_(stride_w),
        pad_h_(pad_h), pad_w_(pad_w),
        groups_(groups),
        weights_(weights), bias_(bias) {
    // Compute input channels from weight size
    // weights size = C_out * (C_in / groups) * kh * kw
    uint64_t kernel_elements =
        static_cast<uint64_t>(kernel_h) * kernel_w;
    uint64_t total_weights = weights.size();
    uint64_t per_out_ch = total_weights / ch;
    uint64_t c_in_per_group = per_out_ch / kernel_elements;
    in_channels_ = static_cast<uint32_t>(c_in_per_group * groups);

    // Compute output spatial dimensions
    out_rows_ = (rows + 2 * pad_h - kernel_h) / stride_h + 1;
    out_cols_ = (cols + 2 * pad_w - kernel_w) / stride_w + 1;

    // im2col column count: (C_in / groups) * kh * kw
    im2col_depth_ = static_cast<uint32_t>(c_in_per_group * kernel_elements);
    // im2col spatial count: H_out * W_out
    im2col_spatial_ = out_rows_ * out_cols_;

    // Reshape weights from [C_out, C_in/G, kh, kw] to [C_out, im2col_depth_]
    // The weight data is already in the right order for row-major GEMM,
    // so we just need to flatten correctly.
    weight_matrix_.resize(ch * im2col_depth_);
    for (uint32_t co = 0; co < ch; co++) {
      for (uint64_t idx = 0; idx < per_out_ch; idx++) {
        uint64_t src = co * per_out_ch + idx;
        uint64_t dst = co * im2col_depth_ + idx;
        weight_matrix_[dst] = weights_[src];
      }
    }

    // Default bias to zeros if not provided
    if (bias.empty()) {
      bias_.resize(ch, T{0});
    }
  }

  ~ConvolutionLayer() override = default;

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

      // Step 1: Pad input
      auto padded = input->clone();
      padded->padding({pad_h_, pad_h_, pad_w_, pad_w_}, T{0});

      // Step 2: im2col -> [im2col_depth_, H_out*W_out] stored as [1, depth, spatial]
      auto col = im2col(padded);

      // Step 3: GEMM: [C_out, depth] x [depth, spatial] = [C_out, spatial]
      auto out_matrix = manual_gemm(
          weight_matrix_.data(), col->data(),
          out_channels_, im2col_depth_, im2col_spatial_);

      // Step 4: Add bias (broadcast across spatial positions)
      // out_matrix is [1, C_out, spatial], so access via at(0, co, pw)
      for (uint32_t co = 0; co < out_channels_; co++) {
        for (uint32_t pw = 0; pw < im2col_spatial_; pw++) {
          out_matrix->at(0, co, pw) += bias_[co];
        }
      }

      // Step 5: Reshape to [C_out, H_out, W_out]
      out_matrix->reshape(out_channels_, out_rows_, out_cols_);

      outputs[b] = std::move(out_matrix);
    }

    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.Conv2d"; }

  uint32_t in_channels() const { return in_channels_; }
  uint32_t out_channels() const { return out_channels_; }
  uint32_t kernel_h() const { return kernel_h_; }
  uint32_t kernel_w() const { return kernel_w_; }
  uint32_t stride_h() const { return stride_h_; }
  uint32_t stride_w() const { return stride_w_; }
  uint32_t pad_h() const { return pad_h_; }
  uint32_t pad_w() const { return pad_w_; }
  uint32_t groups() const { return groups_; }

  /**
   * Access the im2col result for debugging/inspection.
   * Returns the last computed im2col matrix, or nullptr if not yet computed.
   */
  const std::shared_ptr<Tensor<T>>& last_im2col() const { return last_col_; }

 private:
  /**
   * im2col: extract all spatial patches and arrange as columns.
   *
   * For each output position (oh, ow), extract the kernel-sized patch
   * from the (padded) input and flatten it into a column.
   *
   * Output shape: [1, im2col_depth_, H_out*W_out] where
   * im2col_depth_ = C_in/G * kh * kw
   */
  std::shared_ptr<Tensor<T>> im2col(
      const std::shared_ptr<Tensor<T>>& padded) const {
    auto col = std::make_shared<Tensor<T>>(1, im2col_depth_, im2col_spatial_);
    T* col_data = col->data();

    uint32_t c_in_per_group = in_channels_ / groups_;

    for (uint32_t oh = 0; oh < out_rows_; oh++) {
      for (uint32_t ow = 0; ow < out_cols_; ow++) {
        uint32_t col_idx = oh * out_cols_ + ow;

        for (uint32_t g = 0; g < groups_; g++) {
          for (uint32_t ci = 0; ci < c_in_per_group; ci++) {
            uint32_t c_in = g * c_in_per_group + ci;

            for (uint32_t kh = 0; kh < kernel_h_; kh++) {
              for (uint32_t kw = 0; kw < kernel_w_; kw++) {
                uint32_t ih = oh * stride_h_ + kh;
                uint32_t iw = ow * stride_w_ + kw;

                uint32_t row_idx =
                    c_in * kernel_h_ * kernel_w_ + kh * kernel_w_ + kw;
                uint64_t idx =
                    static_cast<uint64_t>(row_idx) * im2col_spatial_ + col_idx;

                col_data[idx] = padded->at(c_in, ih, iw);
              }
            }
          }
        }
      }
    }

    last_col_ = col;
    return col;
  }

  /**
   * Manual GEMM: C = A * B
   *
   * A: [M, K] stored row-major
   * B: [K, N] stored row-major
   * C: [M, N] stored as Tensor [1, M, N]
   *
   * Loop order: i(j(k)) for simplicity.
   */
  static std::shared_ptr<Tensor<T>> manual_gemm(
      const T* __restrict__ a,
      const T* __restrict__ b,
      uint32_t M, uint32_t K, uint32_t N) {
    auto c = std::make_shared<Tensor<T>>(1, M, N);
    T* c_data = c->data();

    for (uint32_t i = 0; i < M; i++) {
      for (uint32_t j = 0; j < N; j++) {
        T sum = T{0};
        for (uint32_t k = 0; k < K; k++) {
          sum += a[i * K + k] * b[k * N + j];
        }
        c_data[i * N + j] = sum;
      }
    }

    return c;
  }

  uint32_t in_channels_, out_channels_;
  uint32_t in_rows_, in_cols_;
  uint32_t out_rows_, out_cols_;
  uint32_t kernel_h_, kernel_w_;
  uint32_t stride_h_, stride_w_;
  uint32_t pad_h_, pad_w_;
  uint32_t groups_;

  uint32_t im2col_depth_;   // C_in/G * kh * kw
  uint32_t im2col_spatial_; // H_out * W_out

  std::vector<T> weights_;
  std::vector<T> bias_;
  std::vector<T> weight_matrix_; // [C_out, im2col_depth_]

  mutable std::shared_ptr<Tensor<T>> last_col_;
};

/**
 * ConvolutionRegister -- register a parameterized Convolution with the factory.
 */
template <typename T>
Layer<T>* ConvolutionRegister(
    const std::string& type_name,
    uint32_t ch, uint32_t rows, uint32_t cols,
    uint32_t kernel_h, uint32_t kernel_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w,
    uint32_t groups,
    const std::vector<T>& weights,
    const std::vector<T>& bias = {}) {
  struct Params {
    uint32_t ch, rows, cols;
    uint32_t kernel_h, kernel_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
    uint32_t groups;
    std::vector<T> weights;
    std::vector<T> bias;
  };
  auto params = std::make_shared<Params>(Params{
      ch, rows, cols, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w,
      groups, weights, bias});

  LayerRegisterer<T>::RegisterCreator(
      type_name,
      [params]() -> Layer<T>* {
        return new ConvolutionLayer<T>(
            params->ch, params->rows, params->cols,
            params->kernel_h, params->kernel_w,
            params->stride_h, params->stride_w,
            params->pad_h, params->pad_w,
            params->groups, params->weights, params->bias);
      });

  return new ConvolutionLayer<T>(
      ch, rows, cols, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w,
      groups, weights, bias);
}

}  // namespace learn_infer

#endif  // CONVOLUTION_HPP
