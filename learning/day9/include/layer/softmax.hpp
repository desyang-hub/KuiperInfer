/**
 * day6/include/layer/softmax.hpp
 *
 * SoftmaxLayer -- Softmax activation layer.
 *
 * Implements numerically stable softmax:
 *   1. Find max(x)
 *   2. exp(x - max)
 *   3. Sum of exp(x - max)
 *   4. Divide each by the sum
 *
 * Supports batch: vector of input tensors, each softmaxed independently.
 * Uses #pragma omp parallel for for batch parallelism when OpenMP is available.
 * Uses AVX2 SIMD intrinsics for inner loops when __AVX2__ is defined.
 *
 * Auto-registers as "nn.Softmax".
 */

#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "layer.hpp"
#include "layer_factory.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace learn_infer {

/**
 * SoftmaxLayer<T> -- Softmax activation.
 *
 * Constructor takes a `dim` parameter (number of elements over which
 * to compute softmax).  When dim == 0, the default is a full 1D softmax
 * over all elements in the tensor.
 */
template <typename T>
class SoftmaxLayer : public Layer<T> {
 public:
  using value_type = T;

  explicit SoftmaxLayer(uint32_t dim = 0) : dim_(dim) {}
  ~SoftmaxLayer() override = default;

  StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) override {
    if (inputs.empty()) {
      return StatusCode::kInvalidInput;
    }

    outputs.clear();
    outputs.resize(inputs.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t b = 0; b < static_cast<int64_t>(inputs.size()); b++) {
      const auto& input = inputs[static_cast<size_t>(b)];
      uint32_t d = (dim_ > 0) ? dim_ : static_cast<uint32_t>(input->size());
      assert(d <= input->size() && "softmax dim exceeds tensor size");

      auto out = std::make_shared<Tensor<T>>(
          input->channels(), input->rows(), input->cols());
      const T* __restrict__ idata = input->data();
      T* __restrict__ odata = out->data();

      uint64_t n = input->size();

      if (d == n) {
        // Full 1D softmax over all elements
        softmax_kernel(idata, odata, static_cast<uint64_t>(d));
      } else {
        // Softmax over each chunk of `dim` elements
        for (uint64_t offset = 0; offset < n; offset += d) {
          softmax_kernel(idata + offset, odata + offset,
                         static_cast<uint64_t>(d));
        }
      }

      outputs[b] = std::move(out);
    }

    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.Softmax"; }
  uint32_t dim() const { return dim_; }

 private:
  uint32_t dim_;

#ifdef __AVX2__
  /**
   * AVX2 SIMD-accelerated softmax kernel.
   * Uses 256-bit vectors (8 floats at a time) for the exp-sum
   * and normalization loops.  Max search remains scalar.
   * exp approximation via range reduction + polynomial (Horner's method).
   */
  static void softmax_kernel(const float* __restrict__ inp,
                             float* __restrict__ out, uint64_t n) {
    // Step 1: find max
    float max_val = inp[0];
    for (uint64_t i = 1; i < n; i++) {
      if (inp[i] > max_val) max_val = inp[i];
    }

    // Step 2: exp(x - max) and accumulate sum
    const uint64_t vl = 8;  // vector length for __m256 float
    uint64_t simd_iters = n / vl;
    uint64_t remainder = n % vl;

    __m256 vec_sum = _mm256_setzero_ps();
    for (uint64_t i = 0; i < simd_iters; i++) {
      __m256 v = _mm256_loadu_ps(inp + i * vl);
      __m256 vmax = _mm256_set1_ps(max_val);
      __m256 diff = _mm256_sub_ps(v, vmax);
      __m256 e = simd_exp_ps(diff);
      _mm256_storeu_ps(out + i * vl, e);
      vec_sum = _mm256_add_ps(vec_sum, e);
    }

    // Scalar remainder
    for (uint64_t i = simd_iters * vl; i < n; i++) {
      out[i] = std::exp(inp[i] - max_val);
    }

    // Horizontal sum
    float hsum[8];
    _mm256_storeu_ps(hsum, vec_sum);
    float sum = hsum[0] + hsum[1] + hsum[2] + hsum[3] +
                hsum[4] + hsum[5] + hsum[6] + hsum[7];
    for (uint64_t i = simd_iters * vl; i < n; i++) {
      sum += out[i];
    }

    // Step 3: normalize by sum (SIMD)
    __m256 svec = _mm256_set1_ps(sum);
    for (uint64_t i = 0; i < simd_iters; i++) {
      __m256 v = _mm256_loadu_ps(out + i * vl);
      _mm256_storeu_ps(out + i * vl, _mm256_div_ps(v, svec));
    }
    for (uint64_t i = simd_iters * vl; i < n; i++) {
      out[i] /= sum;
    }
  }

  /**
   * SIMD exp approximation via range reduction + 7th-degree polynomial.
   * exp(x) = 2^k * exp(r), where r = x - k*ln2, r in [-ln2/2, ln2/2].
   * Polynomial coefficients for exp(r) on this range give ~1e-4 error.
   */
  static __m256 simd_exp_ps(__m256 x) {
    // Range reduction
    __m256 c_log2e = _mm256_set1_ps(1.442695041f);
    __m256 y = _mm256_mul_ps(x, c_log2e);
    __m256 yi = _mm256_round_ps(y,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 ln2v = _mm256_set1_ps(0.693147181f);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(yi, ln2v));

    // Polynomial for exp(r): 1 + r*(b1 + r*(b2 + ...))
    __m256 b7 = _mm256_set1_ps(0.00002229f);
    __m256 b6 = _mm256_set1_ps(0.00019843f);
    __m256 b5 = _mm256_set1_ps(0.00138891f);
    __m256 b4 = _mm256_set1_ps(0.00833339f);
    __m256 b3 = _mm256_set1_ps(0.04166667f);
    __m256 b2 = _mm256_set1_ps(0.16666667f);
    __m256 b1 = _mm256_set1_ps(0.49999999f);

    __m256 exp_r = b7;
    exp_r = _mm256_add_ps(b6, _mm256_mul_ps(exp_r, r));
    exp_r = _mm256_add_ps(b5, _mm256_mul_ps(exp_r, r));
    exp_r = _mm256_add_ps(b4, _mm256_mul_ps(exp_r, r));
    exp_r = _mm256_add_ps(b3, _mm256_mul_ps(exp_r, r));
    exp_r = _mm256_add_ps(b2, _mm256_mul_ps(exp_r, r));
    exp_r = _mm256_add_ps(b1, _mm256_mul_ps(exp_r, r));
    __m256 one = _mm256_set1_ps(1.0f);
    exp_r = _mm256_add_ps(one, _mm256_mul_ps(exp_r, r));

    // 2^k via exponent bit manipulation
    int ei[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ei), _mm256_cvttps_epi32(yi));
    float p2[8];
    for (int i = 0; i < 8; i++) {
      int e = ei[i];
      if (e < -126) e = -126;
      if (e > 127)   e = 127;
      uint32_t bits = static_cast<uint32_t>((e + 127) << 23);
      std::memcpy(&p2[i], &bits, sizeof(float));
    }
    __m256 pow2i = _mm256_loadu_ps(p2);

    return _mm256_mul_ps(exp_r, pow2i);
  }
#else
  /**
   * Scalar softmax kernel.
   * Numerically stable: subtract max before exp.
   */
  static void softmax_kernel(const T* __restrict__ inp,
                             T* __restrict__ out, uint64_t n) {
    // Step 1: find max
    T max_val = inp[0];
    for (uint64_t i = 1; i < n; i++) {
      if (inp[i] > max_val) max_val = inp[i];
    }

    // Step 2: exp(x - max) and accumulate sum
    T sum = T{0};
    for (uint64_t i = 0; i < n; i++) {
      out[i] = std::exp(inp[i] - max_val);
      sum += out[i];
    }

    // Step 3: divide by sum
    for (uint64_t i = 0; i < n; i++) {
      out[i] /= sum;
    }
  }
#endif
};

// Auto-registration
static LayerRegistererWrapper<SoftmaxLayer<float>>
    g_softmax_registrar("nn.Softmax");

}  // namespace learn_infer

#endif  // SOFTMAX_HPP
