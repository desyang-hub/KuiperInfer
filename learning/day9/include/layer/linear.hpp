/**
 * day9/include/layer/linear.hpp
 *
 * LinearLayer -- Fully connected (linear) layer.
 *
 * Implements: output = input * weight^T + bias
 *
 * Input: [..., in_features] (flattened to [..., 1, in_features])
 * Weight: [out_features, in_features]
 * Bias: [out_features]
 *
 * Internally uses the same GEMM as convolution: C = A * B
 * where A is weight [out_features, in_features] and B is input
 * [in_features, batch_size].
 *
 * Registered as "nn.Linear".
 */

#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "layer.hpp"
#include "layer_factory.hpp"

namespace learn_infer {

/**
 * LinearLayer<T> -- Fully connected layer.
 *
 * Input tensor is flattened to [1, 1, in_features] if needed,
 * then matrix-multiplied with weight^T and bias added.
 */
template <typename T>
class LinearLayer : public Layer<T> {
 public:
  using value_type = T;

  LinearLayer(uint32_t in_features, uint32_t out_features,
              const std::vector<T>& weight,
              const std::vector<T>& bias = {})
      : in_features_(in_features), out_features_(out_features),
        weight_(weight), bias_(bias) {
    // Default bias to zeros if not provided
    if (bias.empty()) {
      bias_.resize(out_features, T{0});
    }
  }

  ~LinearLayer() override = default;

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

      // Flatten input to [1, 1, in_features]
      assert(input->size() == in_features_ &&
             "Linear: input size must match in_features");

      auto flat = input->clone();
      flat->reshape(1, 1, in_features_);

      // GEMM: [out_features, in_features] * [in_features, 1] = [out_features, 1]
      // Weight layout: [out_features, in_features]
      // Input layout: [in_features, 1] stored as flat data
      // Result: [out_features, 1] stored as [1, out_features, 1]
      auto result = gemm_compute(
          weight_.data(), flat->data(),
          out_features_, in_features_, 1);

      // Add bias
      for (uint32_t o = 0; o < out_features_; o++) {
        result->at(0, o, 0) += bias_[o];
      }

      // Reshape to 1D for convenience
      result->reshape(1, 1, out_features_);

      outputs[b] = std::move(result);
    }

    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.Linear"; }

  uint32_t in_features() const { return in_features_; }
  uint32_t out_features() const { return out_features_; }

 private:
  /**
   * Manual GEMM: C = A * B
   *
   * A: [M, K] stored row-major
   * B: [K, N] stored row-major
   * C: [M, N] stored as Tensor [1, M, N]
   */
  static std::shared_ptr<Tensor<T>> gemm_compute(
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

  uint32_t in_features_, out_features_;
  std::vector<T> weight_;  // [out_features, in_features]
  std::vector<T> bias_;    // [out_features]
};

/**
 * LinearRegister -- register a parameterized Linear layer with the factory.
 */
template <typename T>
Layer<T>* LinearRegister(
    const std::string& type_name,
    uint32_t in_features, uint32_t out_features,
    const std::vector<T>& weight,
    const std::vector<T>& bias = {}) {
  struct Params {
    uint32_t in_features, out_features;
    std::vector<T> weight;
    std::vector<T> bias;
  };
  auto params = std::make_shared<Params>(Params{
      in_features, out_features, weight, bias});

  LayerRegisterer<T>::RegisterCreator(
      type_name,
      [params]() -> Layer<T>* {
        return new LinearLayer<T>(
            params->in_features, params->out_features,
            params->weight, params->bias);
      });

  return new LinearLayer<T>(
      in_features, out_features, weight, bias);
}

}  // namespace learn_infer

#endif  // LINEAR_HPP
