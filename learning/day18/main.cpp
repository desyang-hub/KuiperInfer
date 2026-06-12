/**
 * day18/main.cpp
 *
 * Day 18: OpenMP 多线程并行策略 — 从 SIMD 到多核全面加速
 *
 * Day 17 实现了 SIMD 向量化（单核内多数据并行）。本 day 在此基础上
 * 引入 OpenMP 多线程并行（多核并行），两者结合构成深度学习推理的
 * 最大加速手段：
 *
 *   SIMD (AVX2)  → 单核 8x 加速（向量化）
 *   OpenMP       → 多核 Nx 加速（线程化）
 *   组合         → 8 * N 倍理论加速
 *
 * 本 demo 在 day17 所有 SIMD 代码的基础上，逐层叠加 OpenMP 并行：
 *   1. OpenMP 基础：线程数、区域、环境变量
 *   2. SIMD + OpenMP 组合加速 ReLU
 *   3. SIMD + OpenMP 组合加速 Sigmoid/SiLU
 *   4. SIMD + OpenMP 组合加速 HardSwish/HardSigmoid
 *   5. 6 种激活函数的统一调度器（SIMD + OpenMP 双层）
 *   6. MatMul 的 OpenMP 并行（M 维度拆分）
 *   7. MatMul 的 OpenMP + SIMD 双层并行
 *   8. 完整卷积管线：im2col + SIMD_GEMM + OpenMP
 *   9. Element-wise 加法的 SIMD + OpenMP
 *  10. 并行陷阱：false sharing、负载均衡、过度并行
 *  11. 编译器自动向量化 + OpenMP 对比
 *  12. 总结：SIMD + OpenMP 双层加速模式
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <memory>
#include <algorithm>
#include <string>
#include <immintrin.h>
#include <omp.h>
#include <atomic>

// =====================================================================
// 简易 Tensor 类（复用 day17 设计，支持 3D + 4D）
// =====================================================================
template<typename T = float>
class Tensor {
 public:
  Tensor() = default;
  Tensor(uint32_t c, uint32_t h, uint32_t w)
      : shapes_{static_cast<int32_t>(c), static_cast<int32_t>(h), static_cast<int32_t>(w)} {
    data_.resize(static_cast<uint64_t>(c) * h * w, 0);
  }
  Tensor(uint32_t c, uint32_t d, uint32_t h, uint32_t w)
      : shapes_{static_cast<int32_t>(c), static_cast<int32_t>(d),
                static_cast<int32_t>(h), static_cast<int32_t>(w)} {
    data_.resize(static_cast<uint64_t>(c) * d * h * w, 0);
  }

  uint32_t channels() const { return shapes_.empty() ? 1 : static_cast<uint32_t>(shapes_[0]); }
  uint32_t rows() const     { return shapes_.size() < 2 ? 1 : static_cast<uint32_t>(shapes_[1]); }
  uint32_t cols() const     { return shapes_.size() < 3 ? 1 : static_cast<uint32_t>(shapes_[2]); }
  uint64_t size() const     { return data_.size(); }

  T* data()       { return data_.data(); }
  const T* data() const { return data_.data(); }

  T& at(uint32_t c, uint32_t r, uint32_t col) {
    return data_[static_cast<uint64_t>(c) * rows() * cols() +
                 static_cast<uint64_t>(r) * cols() + col];
  }
  const T& at(uint32_t c, uint32_t r, uint32_t col) const {
    return data_[static_cast<uint64_t>(c) * rows() * cols() +
                 static_cast<uint64_t>(r) * cols() + col];
  }
  T& at(uint32_t c, uint32_t d, uint32_t h, uint32_t w) {
    return data_[static_cast<uint64_t>(c) * shapes_[1] * shapes_[2] * shapes_[3] +
                 static_cast<uint64_t>(d) * shapes_[2] * shapes_[3] +
                 static_cast<uint64_t>(h) * shapes_[3] + w];
  }
  const T& at(uint32_t c, uint32_t d, uint32_t h, uint32_t w) const {
    return data_[static_cast<uint64_t>(c) * shapes_[1] * shapes_[2] * shapes_[3] +
                 static_cast<uint64_t>(d) * shapes_[2] * shapes_[3] +
                 static_cast<uint64_t>(h) * shapes_[3] + w];
  }

  void fill(T val) { std::fill(data_.begin(), data_.end(), val); }

  void randn(float mean = 0.0f, float std_val = 1.0f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, std_val);
    for (auto& v : data_) v = dist(rng);
  }

 private:
  std::vector<T> data_;
  std::vector<int32_t> shapes_;
};

// =====================================================================
// Benchmark & 精度验证工具
// =====================================================================
template<typename Func>
static double Benchmark(Func&& func, int iterations = 100) {
  func();  // warmup
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) func();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

static float MaxDiff(const float* a, const float* b, uint64_t n) {
  float max_d = 0.0f;
  for (uint64_t i = 0; i < n; i++) {
    float d = std::abs(a[i] - b[i]);
    if (d > max_d) max_d = d;
  }
  return max_d;
}

// =====================================================================
// Part 1: OpenMP 基础检测
// =====================================================================
static void PrintOmpSupport() {
  int max_threads = omp_get_max_threads();
  int hardware_threads = omp_get_num_procs();
  int physical_cores = 0;

  std::cout << "  OpenMP Support:\n";
  std::cout << "    Max threads:      " << max_threads << "\n";
  std::cout << "    Hardware threads: " << hardware_threads << "\n";

#if _OPENMP
  std::cout << "    OpenMP version:   " << _OPENMP << " (20" << (_OPENMP - 2011) / 10 << ")\n";
#else
  std::cout << "    OpenMP: NOT enabled\n";
#endif

  std::cout << "\n  OpenMP 基础概念:\n";
  std::cout << "    #pragma omp parallel for  → 将 for 循环分配到多个线程\n";
  std::cout << "    #pragma omp parallel for num_threads(N) → 指定线程数\n";
  std::cout << "    OMP_NUM_THREADS=N  环境变量 → 运行时控制线程数\n";
  std::cout << "\n  SIMD + OpenMP 组合:\n";
  std::cout << "    外层: OpenMP 拆分到多个核心（多核并行）\n";
  std::cout << "    内层: SIMD 在每个核心内 8 路并行（向量化）\n";
  std::cout << "    总加速 ≈ 线程数 × 8（理论上限）\n\n";
}

// ═══════════════════════════════════════════════════
// SIMD 内核函数（复用 day17 代码，不重复实现）
// ═══════════════════════════════════════════════════

// --- ReLU ---
static void SimdReLU(const float* in, float* out, uint64_t n) {
  __m256 zero = _mm256_setzero_ps();
  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    _mm256_storeu_ps(out + i, _mm256_max_ps(zero, v));
  }
  for (; i < n; i++)
    out[i] = std::max(0.0f, in[i]);
}

// --- Sigmoid (Taylor exp approximation) ---
static void SimdSigmoid(const float* in, float* out, uint64_t n) {
  __m256 one      = _mm256_set1_ps(1.0f);
  __m256 zero     = _mm256_setzero_ps();
  __m256 ln2      = _mm256_set1_ps(0.69314718056f);
  __m256 inv_ln2  = _mm256_set1_ps(1.44269504089f);
  __m256 p1 = _mm256_set1_ps(1.0f);
  __m256 p2 = _mm256_set1_ps(0.5f);
  __m256 p3 = _mm256_set1_ps(0.16666667f);
  __m256 p4 = _mm256_set1_ps(0.04166667f);
  __m256 p5 = _mm256_set1_ps(0.00833333f);

  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(in + i);
    __m256 neg_x = _mm256_sub_ps(zero, x);

    __m256 ir = _mm256_round_ps(_mm256_mul_ps(neg_x, inv_ln2),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 fr = _mm256_sub_ps(neg_x, _mm256_mul_ps(ir, ln2));

    __m256 expf = _mm256_add_ps(one,
        _mm256_mul_ps(fr, _mm256_add_ps(p1,
            _mm256_mul_ps(fr, _mm256_add_ps(p2,
                _mm256_mul_ps(fr, _mm256_add_ps(p3,
                    _mm256_mul_ps(fr, _mm256_add_ps(p4,
                        _mm256_mul_ps(fr, p5))))))))));

    __m256 two_pow_i = _mm256_castsi256_ps(
        _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvttps_epi32(ir), _mm256_set1_epi32(127)),
            23));

    __m256 exp_neg_x = _mm256_mul_ps(expf, two_pow_i);
    _mm256_storeu_ps(out + i, _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x)));
  }
  for (; i < n; i++)
    out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

// --- HardSwish (blend) ---
static void SimdHardSwish(const float* in, float* out, uint64_t n) {
  __m256 zero   = _mm256_set1_ps(0.0f);
  __m256 three  = _mm256_set1_ps(3.0f);
  __m256 six    = _mm256_set1_ps(6.0f);
  __m256 mthree = _mm256_set1_ps(-3.0f);

  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(in + i);

    __m256 le_mask = _mm256_cmp_ps(x, mthree, _CMP_LE_OS);
    __m256 ge_mask = _mm256_cmp_ps(x, three,  _CMP_GE_OS);
    __m256 mid_mask = _mm256_and_ps(
        _mm256_cmp_ps(x, mthree, _CMP_GT_OS),
        _mm256_cmp_ps(x, three,  _CMP_LT_OS));

    __m256 f_high = x;
    __m256 f_mid  = _mm256_div_ps(_mm256_mul_ps(x, _mm256_add_ps(x, three)), six);

    __m256 result = _mm256_add_ps(
        _mm256_and_ps(f_high, ge_mask),
        _mm256_and_ps(f_mid, mid_mask));
    _mm256_storeu_ps(out + i, result);
  }
  for (; i < n; i++) {
    float x = in[i];
    if (x <= -3.0f) out[i] = 0.0f;
    else if (x >= 3.0f) out[i] = x;
    else out[i] = x * (x + 3.0f) / 6.0f;
  }
}

// --- ReLU6 ---
static void SimdReLU6(const float* in, float* out, uint64_t n) {
  __m256 zero = _mm256_setzero_ps();
  __m256 six  = _mm256_set1_ps(6.0f);
  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(in + i);
    _mm256_storeu_ps(out + i, _mm256_min_ps(six, _mm256_max_ps(zero, v)));
  }
  for (; i < n; i++)
    out[i] = std::min(6.0f, std::max(0.0f, in[i]));
}

// --- SiLU ---
static void SimdSiLU(const float* in, float* out, uint64_t n) {
  __m256 one      = _mm256_set1_ps(1.0f);
  __m256 zero     = _mm256_setzero_ps();
  __m256 ln2      = _mm256_set1_ps(0.69314718056f);
  __m256 inv_ln2  = _mm256_set1_ps(1.44269504089f);
  __m256 p1 = _mm256_set1_ps(1.0f);
  __m256 p2 = _mm256_set1_ps(0.5f);
  __m256 p3 = _mm256_set1_ps(0.16666667f);
  __m256 p4 = _mm256_set1_ps(0.04166667f);
  __m256 p5 = _mm256_set1_ps(0.00833333f);
  __m256 ten   = _mm256_set1_ps(10.0f);
  __m256 mfive = _mm256_set1_ps(-5.0f);

  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(in + i);
    __m256 neg_x = _mm256_sub_ps(zero, x);

    __m256 ir = _mm256_round_ps(_mm256_mul_ps(neg_x, inv_ln2),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 fr = _mm256_sub_ps(neg_x, _mm256_mul_ps(ir, ln2));
    __m256 expf = _mm256_add_ps(one,
        _mm256_mul_ps(fr, _mm256_add_ps(p1,
            _mm256_mul_ps(fr, _mm256_add_ps(p2,
                _mm256_mul_ps(fr, _mm256_add_ps(p3,
                    _mm256_mul_ps(fr, _mm256_add_ps(p4,
                        _mm256_mul_ps(fr, p5))))))))));
    __m256 two_pow_i = _mm256_castsi256_ps(
        _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvttps_epi32(ir), _mm256_set1_epi32(127)),
            23));
    __m256 exp_neg_x = _mm256_mul_ps(expf, two_pow_i);

    __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
    __m256 silu = _mm256_mul_ps(x, sig);

   // Clamp: x > 10 -> x,  x < -5 -> 0, else silu
   __m256 gt_mask = _mm256_cmp_ps(x, ten, _CMP_GT_OS);
   __m256 lt_mask = _mm256_cmp_ps(x, mfive, _CMP_LT_OS);

   __m256 result = _mm256_blendv_ps(silu, x, gt_mask);
   __m256 zero_v = _mm256_setzero_ps();
   result = _mm256_blendv_ps(result, zero_v, lt_mask);

   _mm256_storeu_ps(out + i, result);
  }
  for (; i < n; i++) {
    float x = in[i];
    float fx = (x > 10) ? x : (x < -5) ? 0 : x / (1 + std::exp(-x));
    out[i] = fx;
  }
}

// --- HardSigmoid ---
static void SimdHardSigmoid(const float* in, float* out, uint64_t n) {
  __m256 zero   = _mm256_set1_ps(0.0f);
  __m256 one    = _mm256_set1_ps(1.0f);
  __m256 three  = _mm256_set1_ps(3.0f);
  __m256 mthree = _mm256_set1_ps(-3.0f);
  __m256 six    = _mm256_set1_ps(6.0f);
  __m256 half   = _mm256_set1_ps(0.5f);

  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(in + i);

    __m256 le_mask = _mm256_cmp_ps(x, mthree, _CMP_LE_OS);
    __m256 ge_mask = _mm256_cmp_ps(x, three,  _CMP_GE_OS);
    __m256 mid_mask = _mm256_and_ps(
        _mm256_cmp_ps(x, mthree, _CMP_GT_OS),
        _mm256_cmp_ps(x, three,  _CMP_LT_OS));

    __m256 f_high = one;
    __m256 f_mid  = _mm256_add_ps(_mm256_div_ps(x, six), half);

    __m256 result = _mm256_add_ps(
        _mm256_and_ps(f_high, ge_mask),
        _mm256_and_ps(f_mid, mid_mask));
    _mm256_storeu_ps(out + i, result);
  }
  for (; i < n; i++) {
    float x = in[i];
    if (x <= -3.0f) out[i] = 0.0f;
    else if (x >= 3.0f) out[i] = 1.0f;
    else out[i] = x / 6.0f + 0.5f;
  }
}

// --- MatMul SIMD8 ---
static void SimdMatMul8(const float* A, const float* B, float* C,
                         uint32_t M, uint32_t K, uint32_t N) {
  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j + 8 <= N; j += 8) {
      __m256 acc = _mm256_setzero_ps();
      for (uint32_t k = 0; k < K; k++) {
        __m256 bv = _mm256_loadu_ps(B + k * N + j);
        __m256 av = _mm256_set1_ps(A[i * K + k]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
      }
      _mm256_storeu_ps(C + i * N + j, acc);
    }
    for (uint32_t j = (N + 7) / 8 * 8; j < N; j++) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < K; k++)
        sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }
}

// --- Scalar implementations for reference ---
static void ScalarMatMul(const float* A, const float* B, float* C,
                          uint32_t M, uint32_t K, uint32_t N) {
  for (uint32_t i = 0; i < M; i++)
    for (uint32_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < K; k++)
        sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
}

// =====================================================================
// Part 2: 激活函数统一调度器（复用 day17 设计）
// =====================================================================
enum class ActivationType {
  kReLU = 0,
  kReLU6,
  kSigmoid,
  kHardSwish,
  kSiLU,
  kHardSigmoid,
  kCount
};

static void SimdActiv(const float* in, float* out, uint64_t n, ActivationType type) {
  switch (type) {
    case ActivationType::kReLU:        SimdReLU(in, out, n); break;
    case ActivationType::kReLU6:       SimdReLU6(in, out, n); break;
    case ActivationType::kSigmoid:     SimdSigmoid(in, out, n); break;
    case ActivationType::kHardSwish:   SimdHardSwish(in, out, n); break;
    case ActivationType::kSiLU:        SimdSiLU(in, out, n); break;
    case ActivationType::kHardSigmoid: SimdHardSigmoid(in, out, n); break;
    default: break;
  }
}

// =====================================================================
// Part 3: SIMD + OpenMP 组合加速激活函数
//
// 核心思路：将总数据量按线程数拆分，每个线程在自己的数据块上
// 运行 SIMD 内核。这就是生产代码中 ActivationLayer::Forward()
// 的做法：
//
//   #pragma omp parallel for num_threads(batch)
//   for (int b = 0; b < batch; b++) {
//       func(inputs[b], outputs[b]);   // func = SIMD 函数指针
//   }
//
// 这里我们用更通用的方式：对任意大小 N 做 OpenMP 拆分。
// =====================================================================

// 通用模板：将 SIMD 内核包装成 OpenMP 并行版本
// chunk_size = 每个线程处理的元素数（默认让 OpenMP 自动分块）
static void OmpSimdActiv(const float* in, float* out, uint64_t n,
                          ActivationType type, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int64_t t = 0; t < num_threads; t++) {
    // 计算本线程的数据范围
    int64_t chunk = static_cast<int64_t>(n) / num_threads;
    int64_t remainder = static_cast<int64_t>(n) % num_threads;
    int64_t start = t * chunk + std::min(static_cast<int64_t>(t), remainder);
    int64_t end = start + chunk + (t < remainder ? 1 : 0);

    // 在本线程的数据块上运行 SIMD
    SimdActiv(in + start, out + start, end - start, type);
  }
}

// =====================================================================
// Part 4: MatMul 的 OpenMP 并行（M 维度拆分）
//
// SimdMatMul8 串行处理所有 M 行。并行版本将 M 维度拆分到线程：
//
//   #pragma omp parallel for num_threads(T)
//   for (int i = 0; i < M; i++) {
//       // 每个线程处理一行，行内用 SIMD8
//   }
//
// 这是生产代码中 ConvolutionLayer 的典型并行策略：
// 输出 channel 维度并行 + 内层 SIMD。
// =====================================================================
static void OmpSimdMatMul(const float* A, const float* B, float* C,
                           uint32_t M, uint32_t K, uint32_t N,
                           int num_threads) {
  // 为每个线程分配局部输出缓冲区，避免 false sharing
  // 然后合并到 C（生产代码直接写不同行，无冲突）
#pragma omp parallel for num_threads(num_threads)
  for (int32_t i = 0; i < static_cast<int32_t>(M); i++) {
    // 每行的 SIMD 计算（行之间无数据依赖，天然线程安全）
    uint32_t j = 0;
    for (; j + 8 <= N; j += 8) {
      __m256 acc = _mm256_setzero_ps();
      for (uint32_t k = 0; k < K; k++) {
        __m256 bv = _mm256_loadu_ps(B + k * N + j);
        __m256 av = _mm256_set1_ps(A[i * K + k]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
      }
      _mm256_storeu_ps(C + i * N + j, acc);
    }
    for (; j < N; j++) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < K; k++)
        sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }
}

// =====================================================================
// 主函数
// =====================================================================
int main() {
  std::cout << "================================================================\n";
  std::cout << "  Day 18: OpenMP 多线程并行 — SIMD + 多核全面加速\n";
  std::cout << "================================================================\n\n";

  // ═══════════════════════════════════════════════════
  // Part 1: OpenMP 支持检测
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 1: OpenMP 基础检测 ===\n";
  PrintOmpSupport();

  // ═══════════════════════════════════════════════════
  // Part 2: 激活函数 — Scalar vs SIMD vs SIMD+OpenMP
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 2: 激活函数 Scalar vs SIMD vs SIMD+OpenMP ===\n";
  {
    constexpr uint64_t N = 10000000;  // 增大到 10M 以体现多核优势
    std::vector<float> in(N), out_s(N), out_simd(N), out_omp(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(-3, 3);
    for (auto& v : in) v = dist(rng);

    int threads = std::min(omp_get_max_threads(), 8);

    // Sigmoid — 三种实现对比
    {
      auto scalar_sig = [&]{
        for (uint64_t i = 0; i < N; i++)
          out_s[i] = 1.0f / (1.0f + std::exp(-in[i]));
      };
      auto simd_sig = [&]{ SimdSigmoid(in.data(), out_simd.data(), N); };
      auto ompsig = [&]{ OmpSimdActiv(in.data(), out_omp.data(), N, ActivationType::kSigmoid, threads); };

      scalar_sig(); simd_sig(); ompsig();
      float diff = MaxDiff(out_s.data(), out_omp.data(), N);
      std::cout << "  Sigmoid (N=" << N << ", threads=" << threads << ")\n";
      std::cout << "    OMP+SIMD vs Scalar max_diff: " << std::setprecision(6) << diff
                << "  " << (diff < 5e-3f ? "✓" : "✗") << "\n";

      auto ts = Benchmark(scalar_sig, 50);
      auto ti = Benchmark(simd_sig, 50);
      auto to = Benchmark(ompsig, 50);
      std::cout << "    Scalar:   " << std::fixed << std::setprecision(3) << ts << " ms\n";
      std::cout << "    SIMD:     " << std::setprecision(3) << ti << " ms  (speedup: "
                << std::setprecision(2) << (ts/ti) << "x)\n";
      std::cout << "    SIMD+OMP: " << std::setprecision(3) << to << " ms  (speedup: "
                << std::setprecision(2) << (ts/to) << "x, vs SIMD: " << (ti/to) << "x)\n\n";
    }

    // HardSwish — 三种实现对比
    {
      auto scalar_hs = [&]{
        for (uint64_t i = 0; i < N; i++) {
          float x = in[i];
          out_s[i] = (x <= -3) ? 0 : (x >= 3) ? x : x*(x+3)/6;
        }
      };
      auto simd_hs = [&]{ SimdHardSwish(in.data(), out_simd.data(), N); };
      auto omp_hs = [&]{ OmpSimdActiv(in.data(), out_omp.data(), N, ActivationType::kHardSwish, threads); };

      scalar_hs(); simd_hs(); omp_hs();
      float diff = MaxDiff(out_s.data(), out_omp.data(), N);
      std::cout << "  HardSwish (N=" << N << ", threads=" << threads << ")\n";
      std::cout << "    OMP+SIMD vs Scalar max_diff: " << std::setprecision(10) << diff
                << "  " << (diff < 1e-5f ? "✓" : "✗") << "\n";

      auto ts = Benchmark(scalar_hs, 50);
      auto ti = Benchmark(simd_hs, 50);
      auto to = Benchmark(omp_hs, 50);
      std::cout << "    Scalar:   " << std::fixed << std::setprecision(3) << ts << " ms\n";
      std::cout << "    SIMD:     " << std::setprecision(3) << ti << " ms  (speedup: "
                << std::setprecision(2) << (ts/ti) << "x)\n";
      std::cout << "    SIMD+OMP: " << std::setprecision(3) << to << " ms  (speedup: "
                << std::setprecision(2) << (ts/to) << "x, vs SIMD: " << (ti/to) << "x)\n\n";
    }

    // SiLU — 三种实现对比
    {
      auto scalar_s = [&]{
        for (uint64_t i = 0; i < N; i++) {
          float x = in[i];
          if (x > 10) out_s[i] = x;
          else if (x < -5) out_s[i] = 0;
          else out_s[i] = x / (1 + std::exp(-x));
        }
      };
      auto simd_s = [&]{ SimdSiLU(in.data(), out_simd.data(), N); };
      auto omp_s = [&]{ OmpSimdActiv(in.data(), out_omp.data(), N, ActivationType::kSiLU, threads); };

      scalar_s(); simd_s(); omp_s();
      float diff = MaxDiff(out_s.data(), out_omp.data(), N);
      std::cout << "  SiLU (N=" << N << ", threads=" << threads << ")\n";
      std::cout << "    OMP+SIMD vs Scalar max_diff: " << std::setprecision(6) << diff
                << "  " << (diff < 5e-3f ? "✓" : "✗") << "\n";

      auto ts = Benchmark(scalar_s, 50);
      auto ti = Benchmark(simd_s, 50);
      auto to = Benchmark(omp_s, 50);
      std::cout << "    Scalar:   " << std::fixed << std::setprecision(3) << ts << " ms\n";
      std::cout << "    SIMD:     " << std::setprecision(3) << ti << " ms  (speedup: "
                << std::setprecision(2) << (ts/ti) << "x)\n";
      std::cout << "    SIMD+OMP: " << std::setprecision(3) << to << " ms  (speedup: "
                << std::setprecision(2) << (ts/to) << "x, vs SIMD: " << (ti/to) << "x)\n\n";
    }
  }

  // ═══════════════════════════════════════════════════
  // Part 3: 6 种激活函数全面对比 — Scalar / SIMD / SIMD+OpenMP
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 3: 6 种激活函数全面加速对比 ===\n";
  {
    constexpr uint64_t N = 10000000;
    std::vector<float> in(N), out_s(N), out_simd(N), out_omp(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(-3, 3);
    for (auto& v : in) v = dist(rng);

    int threads = std::min(omp_get_max_threads(), 8);

    std::cout << "  Benchmark (N=" << N << ", threads=" << threads << ", iterations=50):\n";
    std::cout << "  " << std::left << std::setw(14) << "Activation"
              << std::setw(14) << "Scalar(ms)" << std::setw(14) << "SIMD(ms)"
              << std::setw(14) << "SIMD+OMP(ms)\n";
    std::cout << "  " << std::string(56, '-') << "\n";

    auto bench = [&](const char* name, auto&& scalar_fn, ActivationType type) {
      scalar_fn();
      SimdActiv(in.data(), out_simd.data(), N, type);
      OmpSimdActiv(in.data(), out_omp.data(), N, type, threads);

      auto ts = Benchmark(scalar_fn, 50);
      auto ti = Benchmark([&]{ SimdActiv(in.data(), out_simd.data(), N, type); }, 50);
      auto to = Benchmark([&]{ OmpSimdActiv(in.data(), out_omp.data(), N, type, threads); }, 50);

      std::cout << "  " << std::left << std::setw(14) << name
                << std::right << std::fixed << std::setprecision(3)
                << std::setw(13) << ts << " ms"
                << std::setw(13) << ti << " ms"
                << std::setw(13) << to << " ms"
                << "  (" << std::setprecision(2) << (ts/to) << "x total)\n";
    };

    bench("ReLU",  [&]{ for (uint64_t i=0;i<N;i++) out_s[i]=std::max(0.0f,in[i]); },
          ActivationType::kReLU);
    bench("ReLU6", [&]{ for (uint64_t i=0;i<N;i++) out_s[i]=std::min(6.0f,std::max(0.0f,in[i])); },
          ActivationType::kReLU6);
    bench("Sigmoid", [&]{ for (uint64_t i=0;i<N;i++) out_s[i]=1.0f/(1.0f+std::exp(-in[i])); },
          ActivationType::kSigmoid);
    bench("HardSwish", [&]{ for (uint64_t i=0;i<N;i++){float x=in[i];out_s[i]=(x<=-3)?0:(x>=3)?x:x*(x+3)/6;} },
          ActivationType::kHardSwish);
    bench("SiLU", [&]{ for (uint64_t i=0;i<N;i++){float x=in[i];if(x>10)out_s[i]=x;else if(x<-5)out_s[i]=0;else out_s[i]=x/(1+std::exp(-x));} },
          ActivationType::kSiLU);
    bench("HardSigmoid", [&]{ for (uint64_t i=0;i<N;i++){float x=in[i];out_s[i]=(x<=-3)?0:(x>=3)?1:x/6+0.5f;} },
          ActivationType::kHardSigmoid);

    std::cout << "\n  观察:\n";
    std::cout << "    - SIMD 加速来自向量化（单核内 8 路并行）\n";
    std::cout << "    - OMP 加速来自多线程（多核并行）\n";
    std::cout << "    - 组合加速 ≈ SIMD_speedup × OMP_speedup\n";
    std::cout << "    - 数据量足够大时，OMP 加速接近线性（threads 倍）\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 4: MatMul — Scalar vs SIMD vs SIMD+OpenMP
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 4: MatMul 三层加速对比 ===\n";
  {
    struct Case { uint32_t M, K, N; std::string name; };
    std::vector<Case> cases = {
      {128, 256, 256, "FC: [128,256] x [256,128]"},
      {256, 512, 512, "FC: [256,512] x [512,256]"},
      {512, 1024, 1024, "FC: [512,1024] x [1024,512]"},
    };

    int threads = std::min(omp_get_max_threads(), 8);

    for (const auto& c : cases) {
      uint64_t size_a = static_cast<uint64_t>(c.M) * c.K;
      uint64_t size_b = static_cast<uint64_t>(c.K) * c.N;
      uint64_t size_c = static_cast<uint64_t>(c.M) * c.N;

      std::vector<float> A(size_a), B(size_b);
      std::vector<float> Cs(size_c), Csimd(size_c), Comp(size_c);
      std::mt19937 rng(42);
      std::normal_distribution<float> dist(0, 0.1f);
      for (auto& v : A) v = dist(rng);
      for (auto& v : B) v = dist(rng);

      ScalarMatMul(A.data(), B.data(), Cs.data(), c.M, c.K, c.N);
      SimdMatMul8(A.data(), B.data(), Csimd.data(), c.M, c.K, c.N);
      OmpSimdMatMul(A.data(), B.data(), Comp.data(), c.M, c.K, c.N, threads);

      float diff = MaxDiff(Cs.data(), Comp.data(), size_c);
      std::cout << "  " << c.name << "  (threads=" << threads << ")\n";
      std::cout << "    max_diff: " << std::setprecision(8) << diff
                << "  " << (diff < 1e-1f ? "✓" : "✗") << "\n";

      auto t_s = Benchmark([&]{ ScalarMatMul(A.data(), B.data(), Cs.data(), c.M, c.K, c.N); }, 10);
      auto t_i = Benchmark([&]{ SimdMatMul8(A.data(), B.data(), Csimd.data(), c.M, c.K, c.N); }, 10);
      auto t_o = Benchmark([&]{ OmpSimdMatMul(A.data(), B.data(), Comp.data(), c.M, c.K, c.N, threads); }, 10);
      std::cout << "    Scalar:   " << std::fixed << std::setprecision(2) << t_s << " ms\n";
      std::cout << "    SIMD8:    " << std::setprecision(2) << t_i << " ms  (speedup: "
                << std::setprecision(2) << (t_s/t_i) << "x)\n";
      std::cout << "    SIMD+OMP: " << std::setprecision(2) << t_o << " ms  (speedup: "
                << std::setprecision(2) << (t_s/t_o) << "x, vs SIMD: " << (t_i/t_o) << "x)\n";
    }
    std::cout << "\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 5: 完整卷积管线 — im2col + SIMD_GEMM + OpenMP
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 5: Conv Pipeline — im2col + SIMD_GEMM + OpenMP ===\n";
  {
    auto input = std::make_shared<Tensor<float>>(64, 28, 28);
    input->randn(0, 0.5f);
    auto weight = std::make_shared<Tensor<float>>(128, 64, 3, 3);
    weight->randn(0, 0.1f);

    uint32_t in_c = 64, in_h = 28, in_w = 28;
    uint32_t out_c = 128, kh = 3, kw = 3;
    uint32_t stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    uint32_t out_h = (in_h + 2*pad_h - kh)/stride_h + 1;
    uint32_t out_w = (in_w + 2*pad_w - kw)/stride_w + 1;

    uint32_t col_rows = in_c * kh * kw;  // 576
    uint32_t col_cols = out_h * out_w;   // 784

    // im2col
    std::vector<float> im2col_mat(col_rows * col_cols, 0.0f);
    {
      uint32_t col_idx = 0;
      for (uint32_t oh = 0; oh < out_h; oh++)
        for (uint32_t ow = 0; ow < out_w; ow++) {
          uint32_t row_idx = 0;
          for (uint32_t ic = 0; ic < in_c; ic++)
            for (uint32_t jkh = 0; jkh < kh; jkh++)
              for (uint32_t jkw = 0; jkw < kw; jkw++) {
                int ih = (int)(oh*stride_h + jkh - pad_h);
                int iw = (int)(ow*stride_w + jkw - pad_w);
                if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w)
                  im2col_mat[row_idx * col_cols + col_idx] =
                      input->at(ic, (uint32_t)ih, (uint32_t)iw);
                row_idx++;
              }
          col_idx++;
        }
    }

    // Weight matrix
    std::vector<float> wt_mat(out_c * col_rows);
    for (uint32_t k = 0; k < out_c; k++)
      for (uint32_t ic = 0; ic < in_c; ic++)
        for (uint32_t jkh = 0; jkh < kh; jkh++)
          for (uint32_t jkw = 0; jkw < kw; jkw++) {
            uint32_t col = ic * kh * kw + jkh * kw + jkw;
            wt_mat[k * col_rows + col] = weight->at(k, ic, jkh, jkw);
          }

    // GEMM with three implementations
    std::vector<float> gemm_s(out_c * col_cols), gemm_i(out_c * col_cols), gemm_o(out_c * col_cols);
    int threads = std::min(omp_get_max_threads(), 8);

    ScalarMatMul(wt_mat.data(), im2col_mat.data(), gemm_s.data(), out_c, col_rows, col_cols);
    SimdMatMul8(wt_mat.data(), im2col_mat.data(), gemm_i.data(), out_c, col_rows, col_cols);
    OmpSimdMatMul(wt_mat.data(), im2col_mat.data(), gemm_o.data(), out_c, col_rows, col_cols, threads);

    float diff = MaxDiff(gemm_s.data(), gemm_o.data(), out_c * col_cols);
    std::cout << "  Conv: [" << in_c << "," << in_h << "," << in_w << "] x ["
              << out_c << "," << in_c << "," << kh << "x" << kw << "]\n";
    std::cout << "  GEMM: [" << out_c << ", " << col_rows << "] x [" << col_rows << ", " << col_cols << "]\n";
    std::cout << "  OMP+SIMD vs Scalar max_diff: " << std::setprecision(8) << diff
              << "  " << (diff < 1e-1f ? "✓" : "✗") << "\n";

    auto t_s = Benchmark([&]{
      ScalarMatMul(wt_mat.data(), im2col_mat.data(), gemm_s.data(), out_c, col_rows, col_cols);
    }, 10);
    auto t_i = Benchmark([&]{
      SimdMatMul8(wt_mat.data(), im2col_mat.data(), gemm_i.data(), out_c, col_rows, col_cols);
    }, 10);
    auto t_o = Benchmark([&]{
      OmpSimdMatMul(wt_mat.data(), im2col_mat.data(), gemm_o.data(), out_c, col_rows, col_cols, threads);
    }, 10);
    std::cout << "  Scalar GEMM:   " << std::fixed << std::setprecision(2) << t_s << " ms/iter\n";
    std::cout << "  SIMD8  GEMM:   " << std::setprecision(2) << t_i << " ms/iter  (speedup: "
              << std::setprecision(2) << (t_s/t_i) << "x)\n";
    std::cout << "  SIMD+OMP GEMM: " << std::setprecision(2) << t_o << " ms/iter  (speedup: "
              << std::setprecision(2) << (t_s/t_o) << "x, vs SIMD: " << (t_i/t_o) << "x)\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 6: OpenMP 线程数对性能的影响
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 6: 线程数 vs 加速比 ===\n";
  {
    constexpr uint64_t N = 20000000;
    std::vector<float> in(N), out1(N), out2(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(-3, 3);
    for (auto& v : in) v = dist(rng);

    // Sigmoid 基准（单线程 SIMD）
    SimdSigmoid(in.data(), out1.data(), N);
    auto t_base = Benchmark([&]{ SimdSigmoid(in.data(), out1.data(), N); }, 50);

    std::cout << "  Sigmoid (N=" << N << "), iterations=50:\n";
    std::cout << "  " << std::left << std::setw(10) << "Threads"
              << std::setw(14) << "Time(ms)" << std::setw(14) << "Speedup\n";
    std::cout << "  " << std::string(38, '-') << "\n";
    std::cout << "  " << std::left << std::setw(10) << "1"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(13) << t_base << " ms"
              << std::setw(13) << "1.00" << "x\n";

    for (int t : {2, 4, 8}) {
      if (t > omp_get_max_threads()) break;
      auto t_omp = Benchmark([&]{
        OmpSimdActiv(in.data(), out2.data(), N, ActivationType::kSigmoid, t);
      }, 50);
      std::cout << "  " << std::left << std::setw(10) << t
                << std::right << std::fixed << std::setprecision(3)
                << std::setw(13) << t_omp << " ms"
                << std::setw(12) << (t_base/t_omp) << "x  ";
      // 计算线性加速比
      float linear = t_base / t_omp / t;
      std::cout << "(linear: " << std::setprecision(1) << (linear*100) << "%)\n";
    }
    std::cout << "  (线性加速 = 实际加速 / 线程数，越接近 100% 越好)\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 7: 并行陷阱演示
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 7: 并行陷阱 ===\n";

  // 陷阱 1: 数据竞争（故意制造 bug）
  std::cout << "  陷阱 1: 数据竞争（Data Race）\n";
  {
    constexpr uint64_t N = 1000000;
    std::vector<float> in(N, 1.0f);
    std::vector<float> out_correct(N), out_race(N);

    // 正确：每个线程写不同位置
    {
#pragma omp parallel for num_threads(4)
      for (int64_t i = 0; i < static_cast<int64_t>(N); i++)
        out_correct[i] = in[i] * 2.0f;
    }

    // 错误：多个线程写同一个累加器（演示用小规模）
    std::atomic<float> atomic_sum(0.0f);  // 用 atomic 修复
    float naive_sum = 0.0f;

    // 错误示范：非 atomic 累加
    {
#pragma omp parallel for num_threads(4) reduction(+:naive_sum)
      for (int64_t i = 0; i < 100000; i++)
        naive_sum += 1.0f;
    }
    std::cout << "    正确累加（reduction）: " << naive_sum << "\n";
    std::cout << "    每个线程独立，最后 reduction 合并 — 无竞争\n";
    std::cout << "    如果没有 reduction，多个线程同时写 naive_sum → 数据竞争（结果错误）\n\n";
  }

  // 陷阱 2: False Sharing
  std::cout << "  陷阱 2: False Sharing（伪共享）\n";
  {
    // 当多个线程写不同的变量，但这些变量在同一个缓存行（64 bytes）上，
    // CPU 会在核心之间频繁同步缓存行，导致性能大幅下降。
    //
    // 示例：每个线程一个计数器，但计数器相邻存放
    //
    // 错误写法: float counters[8];  // 8 个 float = 32 bytes < 64 byte cache line
    // 正确写法: 用 __attribute__((aligned(64))) 或 padding

    struct alignas(64) Counter {  // 每个 Counter 独占一个缓存行
      float value = 0;
    };

    constexpr int ITERS = 10000000;
    int threads = std::min(omp_get_max_threads(), 4);
    std::vector<Counter> counters(threads);

    // 有 padding 的版本
    auto padded_fn = [&] {
      for (auto& c : counters) c.value = 0;
#pragma omp parallel for num_threads(threads)
      for (int64_t t = 0; t < threads; t++) {
        float local = 0;
        for (int i = 0; i < ITERS; i++) local += 1.0f;
        counters[t].value = local;
      }
    };

    // 无 padding 的版本（模拟 false sharing）
    std::vector<float> tight_counters(threads, 0);
    auto tight_fn = [&] {
      for (auto& c : tight_counters) c = 0;
#pragma omp parallel for num_threads(threads)
      for (int64_t t = 0; t < threads; t++) {
        float local = 0;
        for (int i = 0; i < ITERS; i++) local += 1.0f;
        tight_counters[t] = local;
      }
    };

    auto t_padded = Benchmark(padded_fn, 20);
    auto t_tight = Benchmark(tight_fn, 20);

    std::cout << "    有 padding (alignas(64)): " << std::fixed << std::setprecision(3)
              << t_padded << " ms\n";
    std::cout << "    无 padding (可能 false sharing): " << std::setprecision(3)
              << t_tight << " ms\n";
    std::cout << "    ratio: " << std::setprecision(2) << (std::max(t_padded, t_tight) /
              std::min(t_padded, t_tight)) << "x\n";
    std::cout << "    (如果 ratio > 1.2，说明 false sharing 有影响)\n\n";
  }

  // 陷阱 3: 过度并行
  std::cout << "  陷阱 3: 过度并行（线程数 > 数据量）\n";
  {
    // 当数据量很小，线程数很多时，线程创建 + 同步的开销 > 计算本身
    // 生产代码对策：判断 N / threads > 阈值，否则退化为单线程
    constexpr uint64_t SMALL_N = 128;
    std::vector<float> small_in(SMALL_N, 1.0f);
    std::vector<float> small_out1(SMALL_N), small_out2(SMALL_N);

    int threads = std::min(omp_get_max_threads(), 8);

    auto t_single = Benchmark([&]{
      SimdSigmoid(small_in.data(), small_out1.data(), SMALL_N);
    }, 10000);
    auto t_omp = Benchmark([&]{
      OmpSimdActiv(small_in.data(), small_out2.data(), SMALL_N, ActivationType::kSigmoid, threads);
    }, 10000);

    std::cout << "    N=" << SMALL_N << " (远小于线程数 " << threads << ")\n";
    std::cout << "    单线程 SIMD: " << std::fixed << std::setprecision(4) << t_single << " ms\n";
    std::cout << "    OMP+SIMD:    " << std::setprecision(4) << t_omp << " ms\n";
    std::cout << "    ratio: " << std::setprecision(2) << (t_omp/t_single) << "x\n";
    std::cout << "    (OMP 变慢！因为线程同步开销 > 计算量)\n";
    std::cout << "    生产代码策略: if (N / threads < 4096) 使用单线程 SIMD\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 8: 生产代码中的并行模式
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 8: 生产代码并行模式总结 ===\n";
  {
    std::cout << "  KuiperInfer 中的并行模式:\n\n";
    std::cout << "  1. ActivationLayer::Forward() — batch 维度并行\n";
    std::cout << "     #pragma omp parallel for num_threads(batch)\n";
    std::cout << "     for (int b = 0; b < batch; b++) {\n";
    std::cout << "         simd_func(inputs[b], outputs[b]);  // SIMD 内核\n";
    std::cout << "     }\n\n";
    std::cout << "  2. ConvolutionLayer — group + batch 并行\n";
    std::cout << "     #pragma omp parallel for num_threads(batch)\n";
    std::cout << "     for (int b = 0; b < batch; b++) {\n";
    std::cout << "         #pragma omp parallel for  // group 维度（嵌套）\n";
    std::cout << "         for (int g = 0; g < groups; g++) {\n";
    std::cout << "             // im2col + GEMM（内层 SIMD）\n";
    std::cout << "         }\n";
    std::cout << "     }\n\n";
    std::cout << "  3. MaxPooling — batch 维度并行\n";
    std::cout << "     #pragma omp parallel for num_threads(batch)\n";
    std::cout << "     for (int b = 0; b < batch; b++) {\n";
    std::cout << "         // 逐 channel 逐位置取最大值\n";
    std::cout << "     }\n\n";
    std::cout << "  4. Softmax — channel 维度 SIMD，batch 维度 OpenMP\n";
    std::cout << "     #pragma omp parallel for\n";
    std::cout << "     for (int b = 0; b < batch; b++) {\n";
    std::cout << "         // AVX2 softmax 内核（2 遍扫描）\n";
    std::cout << "     }\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 9: 编译器自动向量化 vs 手写 SIMD + OpenMP
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 9: 编译器自动并行 vs 手写 ===\n";
  {
    // -O3 + -ffast-math 可以让编译器自动向量化简单循环
    // #pragma omp parallel for 可以让编译器自动并行循环
    // 但手写 SIMD + OpenMP 在复杂场景（exp, blend）仍然不可替代

    constexpr uint64_t N = 10000000;
    std::vector<float> a(N), b(N), out_auto(N), out_hand(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    int threads = std::min(omp_get_max_threads(), 8);

    // 编译器自动：简单加法
    auto auto_add = [&]{
#pragma omp parallel for num_threads(threads)
      for (int64_t i = 0; i < static_cast<int64_t>(N); i++)
        out_auto[i] = a[i] + b[i];
    };

    // 手写：SIMD + OpenMP
    auto hand_add = [&]{
#pragma omp parallel for num_threads(threads)
      for (int64_t t = 0; t < threads; t++) {
        int64_t chunk = N / threads;
        int64_t rem = N % threads;
        int64_t start = t * chunk + std::min(t, rem);
        int64_t end = start + chunk + (t < rem ? 1 : 0);
        uint64_t i = start;
        __m256 zero = _mm256_setzero_ps();
        for (; i + 8 <= (uint64_t)end; i += 8) {
          __m256 va = _mm256_loadu_ps(a.data() + i);
          __m256 vb = _mm256_loadu_ps(b.data() + i);
          _mm256_storeu_ps(out_hand.data() + i, _mm256_add_ps(va, vb));
        }
        for (; i < (uint64_t)end; i++) out_hand[i] = a[i] + b[i];
      }
    };

    auto_add(); hand_add();
    float diff = MaxDiff(out_auto.data(), out_hand.data(), N);
    std::cout << "  Add (N=" << N << ", threads=" << threads << ")\n";
    std::cout << "    max_diff: " << std::setprecision(10) << diff << "\n";

    auto ta = Benchmark(auto_add, 200);
    auto th = Benchmark(hand_add, 200);
    std::cout << "    编译器自动 (OMP+auto-vector): " << std::fixed << std::setprecision(3) << ta << " ms\n";
    std::cout << "    手写 (OMP+SIMD):              " << std::setprecision(3) << th << " ms\n";
    std::cout << "    ratio: " << std::setprecision(2) << (std::max(ta,th)/std::min(ta,th)) << "x\n";
    std::cout << "    -> 简单加法：编译器自动 + OpenMP 已经足够好\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 10: 总结
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 10: SIMD + OpenMP 双层加速总结 ===\n\n";
  std::cout << "  +------------------------------------------------------------------+\n";
  std::cout << "  |              双层加速模式：SIMD + OpenMP                          |\n";
  std::cout << "  +------------------------------------------------------------------+\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  |  外层: OpenMP 并行（多核）                                         |\n";
  std::cout << "  |    #pragma omp parallel for num_threads(T)                        |\n";
  std::cout << "  |    for (int t = 0; t < T; t++) {  ← 每线程处理 N/T 个元素         |\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  |  内层: SIMD 向量化（单核内 8 路）                                  |\n";
  std::cout << "  |    for (; i + 8 <= local_n; i += 8) {                             |\n";
  std::cout << "  |        __m256 v = _mm256_loadu_ps(...);                           |\n";
  std::cout << "  |        v = compute(v);                                            |\n";
  std::cout << "  |        _mm256_storeu_ps(..., v);                                  |\n";
  std::cout << "  |    }                                                              |\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  |  总加速 ≈ 8 × T  理论上限                                          |\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  +------------------------------------------------------------------+\n\n";
  std::cout << "  并行三原则:\n";
  std::cout << "    1. 数据并行 > 任务并行 — 每个线程处理不同的数据，无共享状态\n";
  std::cout << "    2. 减少同步 — 能不用 reduction 就不用，避免 #pragma omp critical\n";
  std::cout << "    3. 线程粒度 — 每个线程至少处理 4KB 数据（否则线程开销占主导）\n\n";
  std::cout << "  并行陷阱:\n";
  std::cout << "    1. 数据竞争 — 多个线程写同一位置 → 用 reduction 或分片写\n";
  std::cout << "    2. False sharing — 不同线程写相邻缓存行 → alignas(64)\n";
  std::cout << "    3. 过度并行 — 线程数 > 数据量 → 判断阈值后退化单线程\n";
  std::cout << "    4. 负载不均衡 — 用 schedule(dynamic) 或手动分片\n\n";
  std::cout << "  生产代码对应:\n";
  std::cout << "    ActivationLayer::Forward() → batch 级 OpenMP + SIMD 内核\n";
  std::cout << "    ConvolutionLayer::Forward() → batch × group 嵌套 OpenMP + GEMM\n";
  std::cout << "    SoftmaxLayer::Forward() → batch 级 OpenMP + AVX2 内核\n";
  std::cout << "    MaxPooling::Forward() → batch 级 OpenMP + 标量循环\n\n";

  std::cout << "All Day 18 demos completed successfully!\n";
  return 0;
}
