/**
 * day6/include/layer/batchnorm2d.hpp
 *
 * BatchNorm2DLayer -- 2D Batch Normalization (inference mode).
 *
 * Implements the standard batch norm inference formula:
 *   output = weight * (x - running_mean) / sqrt(running_var + eps) + bias
 *
 * Stores running_mean, running_var, weight, bias, and eps as member data.
 * Parameters are per-channel vectors of length C, broadcast across H x W.
 *
 * Inference optimization: precomputes fused scale and offset so the
 * forward pass is just a single multiply-add per element:
 *   y = scale[c] * x + offset[c]
 *
 * Uses #pragma omp parallel for for batch-level parallelism.
 *
 * Registered as "nn.BatchNorm2d".
 */

#ifndef BATCHNORM2D_HPP
#define BATCHNORM2D_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace learn_infer {

/**
 * BatchNorm2DLayer<T> -- Batch Normalization for 2D feature maps.
 *
 * Inference only.  Constructor stores all parameters; forward pass
 * applies the fused linear transform per channel.
 */
template <typename T>
class BatchNorm2DLayer : public Layer<T> {
 public:
  using value_type = T;

  /**
   * Constructor.
   *
   * @param channels     Number of channels C
   * @param running_mean Per-channel running mean (size C)
   * @param running_var  Per-channel running variance (size C)
   * @param weight       Per-channel scale gamma (size C, default all 1)
   * @param bias         Per-channel shift beta (size C, default all 0)
   * @param eps          Numerical stability constant (default 1e-5)
   */
  BatchNorm2DLayer(
      uint32_t channels,
      const std::vector<T>& running_mean,
      const std::vector<T>& running_var,
      const std::vector<T>& weight = {},
      const std::vector<T>& bias = {},
      T eps = T{1e-5})
      : channels_(channels),
        running_mean_(running_mean),
        running_var_(running_var),
        eps_(eps) {
    assert(running_mean.size() == channels &&
           "running_mean size must match channels");
    assert(running_var.size() == channels &&
           "running_var size must match channels");

    // Default weight = 1, bias = 0
    weight_.resize(channels, T{1});
    bias_.resize(channels, T{0});
    if (!weight.empty()) {
      assert(weight.size() == channels);
      weight_ = weight;
    }
    if (!bias.empty()) {
      assert(bias.size() == channels);
      bias_ = bias;
    }

    // Precompute fused inference parameters
    //   inv_stddev[c] = weight[c] / sqrt(running_var[c] + eps)
    //   offset[c]     = bias[c] - inv_stddev[c] * running_mean[c]
    // Forward becomes:  y[c,h,w] = inv_stddev[c] * x[c,h,w] + offset[c]
    inv_stddev_.resize(channels);
    offset_.resize(channels);
    for (uint32_t c = 0; c < channels; c++) {
      inv_stddev_[c] = weight_[c] / std::sqrt(running_var_[c] + eps_);
      offset_[c] = bias_[c] - inv_stddev_[c] * running_mean_[c];
    }
  }

  ~BatchNorm2DLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    outputs.clear();
    outputs.resize(inputs.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1)
#endif
    for (int64_t b = 0; b < static_cast<int64_t>(inputs.size()); b++) {
      const auto& input = inputs[static_cast<size_t>(b)];
      assert(input->channels() == channels_ &&
             "input channels must match layer channels");

      uint32_t R = input->rows();
      uint32_t W = input->cols();
      uint64_t plane_size = static_cast<uint64_t>(R) * W;

      auto out = std::make_shared<Tensor<T>>(channels_, R, W);
      const T* __restrict__ idata = input->data();
      T* __restrict__ odata = out->data();

      // Fused apply: y = inv_stddev * x + offset, per channel
      for (uint32_t c = 0; c < channels_; c++) {
        T inv_std = inv_stddev_[c];
        T off = offset_[c];
        uint64_t ch_off = static_cast<uint64_t>(c) * plane_size;
        for (uint64_t i = 0; i < plane_size; i++) {
          odata[ch_off + i] = inv_std * idata[ch_off + i] + off;
        }
      }

      outputs[b] = std::move(out);
    }

    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.BatchNorm2d"; }

  uint32_t channels() const { return channels_; }
  const std::vector<T>& running_mean() const { return running_mean_; }
  const std::vector<T>& running_var() const { return running_var_; }
  const std::vector<T>& weight() const { return weight_; }
  const std::vector<T>& bias() const { return bias_; }
  T eps() const { return eps_; }

 private:
  uint32_t channels_;
  std::vector<T> running_mean_;
  std::vector<T> running_var_;
  std::vector<T> weight_;
  std::vector<T> bias_;
  T eps_;

  // Precomputed fused parameters
  std::vector<T> inv_stddev_;
  std::vector<T> offset_;
};

/**
 * BatchNorm2DRegister -- register a parameterized BatchNorm2D with the factory.
 *
 * Because BatchNorm2D stores parameters (mean, var, weight, bias),
 * it cannot use the parameterless LayerRegistererWrapper.  This helper
 * registers a lambda creator that captures the parameters by shared_ptr.
 *
 * Usage:
 *   Layer<float>* bn = BatchNorm2DRegister<float>(
 *       "my_bn", 2, {0.0f, 1.0f}, {1.0f, 2.0f}, {1.0f, 1.0f}, {0.0f, 0.0f});
 */
template <typename T>
Layer<T>* BatchNorm2DRegister(
    const std::string& type_name,
    uint32_t channels,
    const std::vector<T>& running_mean,
    const std::vector<T>& running_var,
    const std::vector<T>& weight = {},
    const std::vector<T>& bias = {},
    T eps = T{1e-5}) {
  struct Params {
    uint32_t channels;
    std::vector<T> running_mean;
    std::vector<T> running_var;
    std::vector<T> weight;
    std::vector<T> bias;
    T eps;
  };
  auto params = std::make_shared<Params>(
      Params{channels, running_mean, running_var, weight, bias, eps});

  LayerRegisterer<T>::RegisterCreator(
      type_name,
      [params]() -> Layer<T>* {
        return new BatchNorm2DLayer<T>(
            params->channels, params->running_mean, params->running_var,
            params->weight, params->bias, params->eps);
      });

  return new BatchNorm2DLayer<T>(
      channels, running_mean, running_var, weight, bias, eps);
}

}  // namespace learn_infer

#endif  // BATCHNORM2D_HPP
