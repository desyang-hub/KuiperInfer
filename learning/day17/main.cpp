/**
 * day17/main.cpp
 *
 * Day 17: SIMD 向量化编程 — 从标量到 AVX2 的激活函数与矩阵乘加速
 *
 * 深度学习推理中大量操作是逐元素（element-wise）的：
 *   ReLU(x) = max(0, x),  Sigmoid(x) = 1/(1+e^-x),  加法, 乘法...
 * 标量循环一次处理 1 个 float。SIMD（Single Instruction, Multiple Data）
 * 一条指令同时处理多个 float：
 *
 *   SSE2  (128-bit) → 4 个 float 并行  (__m128)
 *   AVX2  (256-bit) → 8 个 float 并行  (__m256)
 *
 * 本 demo 从零实现：
 *   1. SIMD 检测与基础操作（load/store/set1）
 *   2. Scalar ReLU vs SIMD ReLU
 *   3. Scalar Sigmoid vs SIMD Sigmoid（含 exp 多项式近似）
 *   4. HardSwish 分段函数的 SIMD blend 实现
 *   5. MatMul 的 SIMD 向量化（一次计算 8 个输出元素）
 *   6. 完整卷积管线：im2col + SIMD_GEMM
 *   7. Element-wise 加法/乘法的 SIMD + 编译器自动向量化讨论
 *   8. 总结：SIMD 编程三板斧
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

// =====================================================================
// 简易 Tensor 类（复用 day16 设计，支持 3D + 4D）
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
// Part 1: SIMD 指令集检测与基础操作
// =====================================================================
static void PrintSimdSupport() {
  std::cout << "  CPU SIMD Support:\n";
#if defined(__AVX2__)
  std::cout << "    AVX2  (256-bit, 8 floats)  ✓\n";
#elif defined(__AVX__)
  std::cout << "    AVX   (256-bit, 8 floats)  ✓\n";
#elif defined(__SSE2__)
  std::cout << "    SSE2  (128-bit, 4 floats)  ✓\n";
#else
  std::cout << "    No SIMD detected  ✗\n";
#endif

  std::cout << "\n  SIMD 基础操作演示:\n";
  std::cout << "    __m256  = 256-bit 寄存器，容纳 8 个 float\n";
  std::cout << "    __m128  = 128-bit 寄存器，容纳 4 个 float\n\n";

  // 演示 set1 (broadcast): 把一个标量值填充到所有 lane
  __m256 v = _mm256_set1_ps(3.14f);
  float buf[8];
  _mm256_storeu_ps(buf, v);
  std::cout << "    _mm256_set1_ps(3.14f) -> ";
  for (int i = 0; i < 8; i++) std::cout << buf[i] << " ";
  std::cout << "\n";

  // 演示 loadu / add / storeu
  float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
  float c[8];
  __m256 va = _mm256_loadu_ps(a);
  __m256 vb = _mm256_loadu_ps(b);
  _mm256_storeu_ps(c, _mm256_add_ps(va, vb));
  std::cout << "    [1..8] + [10,20,...,80] -> ";
  for (int i = 0; i < 8; i++) std::cout << c[i] << " ";
  std::cout << "\n\n";
}

// =====================================================================
// Part 2: Scalar ReLU vs SIMD ReLU
//
// ReLU(x) = max(0, x)
// SIMD:    _mm256_max_ps(zero, x) — 一条指令 8 个元素
// =====================================================================
static void ScalarReLU(const float* in, float* out, uint64_t n) {
  for (uint64_t i = 0; i < n; i++)
    out[i] = std::max(0.0f, in[i]);
}

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

// =====================================================================
// Part 3: Scalar Sigmoid vs SIMD Sigmoid
//
// sigmoid(x) = 1 / (1 + exp(-x))
//
// SIMD exp 使用多项式近似而非 std::exp（太慢）：
//   1. Range reduction: x = i*ln(2) + f,  f ∈ [-ln(2)/2, ln(2)/2]
//   2. exp(f) ≈ 1 + f*c1 + f^2*c2 + f^3*c3 + f^4*c4  (Horner)
//   3. 2^i via IEEE 754 指数位操作
//   4. exp(x) = exp(f) * 2^i
//   5. sigmoid(x) = 1 / (1 + exp(-x))
//
// 注意：对于 |x| > 10，sigmoid 已经饱和（~0 或 ~1），直接 clamp
// =====================================================================
static void ScalarSigmoid(const float* in, float* out, uint64_t n) {
  for (uint64_t i = 0; i < n; i++)
    out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

static void SimdSigmoid(const float* in, float* out, uint64_t n) {
  __m256 one      = _mm256_set1_ps(1.0f);
  __m256 zero     = _mm256_setzero_ps();
  __m256 ln2      = _mm256_set1_ps(0.69314718056f);
  __m256 inv_ln2  = _mm256_set1_ps(1.44269504089f);

  // exp(f) ≈ p0 + f*(p1 + f*(p2 + f*(p3 + f*(p4 + f*p5))))
  // Taylor series coefficients for exp(x) = sum(x^k/k!)
  // p0=1, p1=1, p2=1/2, p3=1/6, p4=1/24, p5=1/120
  // For f in [-ln2/2, ln2/2] ≈ [-0.347, 0.347], 6-term Taylor gives ~1e-7 error
  __m256 p1 = _mm256_set1_ps(1.0f);
  __m256 p2 = _mm256_set1_ps(0.5f);
  __m256 p3 = _mm256_set1_ps(0.16666667f);   // 1/6
  __m256 p4 = _mm256_set1_ps(0.04166667f);   // 1/24
  __m256 p5 = _mm256_set1_ps(0.00833333f);   // 1/120

  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(in + i);
    __m256 neg_x = _mm256_sub_ps(zero, x);

    // Range reduction: neg_x = i*ln2 + f
    __m256 ir = _mm256_round_ps(_mm256_mul_ps(neg_x, inv_ln2),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 fr = _mm256_sub_ps(neg_x, _mm256_mul_ps(ir, ln2));

    // exp(f) via 6-term Taylor Horner: 1 + f*(1 + f*(1/2 + f*(1/6 + f*(1/24 + f/120))))
    __m256 expf = _mm256_add_ps(one,
        _mm256_mul_ps(fr,
            _mm256_add_ps(p1,
                _mm256_mul_ps(fr,
                    _mm256_add_ps(p2,
                        _mm256_mul_ps(fr,
                            _mm256_add_ps(p3,
                                _mm256_mul_ps(fr,
                                    _mm256_add_ps(p4,
                                        _mm256_mul_ps(fr, p5))))))))));

    // 2^i: integer i → (i + 127) << 23 → reinterpret as float
    __m256 two_pow_i = _mm256_castsi256_ps(
        _mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvttps_epi32(ir), _mm256_set1_epi32(127)),
            23));

    __m256 exp_neg_x = _mm256_mul_ps(expf, two_pow_i);

    // sigmoid = 1 / (1 + exp(-x))
    _mm256_storeu_ps(out + i, _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x)));
  }
  for (; i < n; i++)
    out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

// =====================================================================
// Part 4: HardSwish — SIMD 条件选择（Blend / Mask）
//
// hard_swish(x) = 0            if x <= -3
//               = x            if x >= 3
//               = x*(x+3)/6    if -3 < x < 3
//
// SIMD 实现关键：用比较指令生成 mask，再用 and/or 做分支选择，
// 避免标量 if-else 带来的分支预测失败。
// =====================================================================
static void ScalarHardSwish(const float* in, float* out, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    float x = in[i];
    if (x <= -3.0f) out[i] = 0.0f;
    else if (x >= 3.0f) out[i] = x;
    else out[i] = x * (x + 3.0f) / 6.0f;
  }
}

static void SimdHardSwish(const float* in, float* out, uint64_t n) {
  __m256 zero   = _mm256_set1_ps(0.0f);
  __m256 three  = _mm256_set1_ps(3.0f);
  __m256 six    = _mm256_set1_ps(6.0f);
  __m256 mthree = _mm256_set1_ps(-3.0f);

  uint64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 x = _mm256_loadu_ps(in + i);

    // 三路 mask
    __m256 le_mask = _mm256_cmp_ps(x, mthree, _CMP_LE_OS);  // x <= -3
    __m256 ge_mask = _mm256_cmp_ps(x, three,  _CMP_GE_OS);  // x >= 3
    __m256 mid_mask = _mm256_and_ps(
        _mm256_cmp_ps(x, mthree, _CMP_GT_OS),
        _mm256_cmp_ps(x, three,  _CMP_LT_OS));              // -3 < x < 3

    // 三个分支
    __m256 f_low  = zero;                                          // 0
    __m256 f_high = x;                                              // x
    __m256 f_mid  = _mm256_div_ps(_mm256_mul_ps(x, _mm256_add_ps(x, three)), six);

    // Blend: mask * value + (1-mask) * other
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

// =====================================================================
// Part 5: MatMul — SIMD 向量化
//
// SimdMatMul8: 同时计算 8 个输出元素 C[i][j..j+7]
// 对 K 维度做标量循环，每步累加 acc += A[i][k] * B[k][j..j+7]
// 这是最简单的 SIMD MatMul 模式：
//   - 不需要 shuffle/permute
//   - 不需要 horizontal sum
//   - 一次 set1_ps + 一次 loadu_ps + 一次 mul + 一次 add
// =====================================================================
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

static void SimdMatMul8(const float* A, const float* B, float* C,
                         uint32_t M, uint32_t K, uint32_t N) {
  for (uint32_t i = 0; i < M; i++) {
    // SIMD 主体：每次计算 8 个连续的输出列
    for (uint32_t j = 0; j + 8 <= N; j += 8) {
      __m256 acc = _mm256_setzero_ps();
      for (uint32_t k = 0; k < K; k++) {
        __m256 bv = _mm256_loadu_ps(B + k * N + j);   // [B[k][j], ..., B[k][j+7]]
        __m256 av = _mm256_set1_ps(A[i * K + k]);      // [A[i][k], ..., A[i][k]]
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
      }
      _mm256_storeu_ps(C + i * N + j, acc);
    }
    // Tail: 不足 8 的列用标量处理
    for (uint32_t j = (N + 7) / 8 * 8; j < N; j++) {
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
  std::cout << "  Day 17: SIMD 向量化编程 — AVX2 加速深度学习算子\n";
  std::cout << "================================================================\n\n";

  // ═══════════════════════════════════════════════════
  // Part 1: SIMD 支持检测与基础操作
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 1: SIMD 指令集检测 ===\n";
  PrintSimdSupport();

  // ═══════════════════════════════════════════════════
  // Part 2: ReLU — Scalar vs SIMD
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 2: ReLU Scalar vs SIMD ===\n";
  {
    constexpr uint64_t N = 1000000;
    std::vector<float> in(N), out_s(N), out_simd(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    for (auto& v : in) v = dist(rng);

    ScalarReLU(in.data(), out_s.data(), N);
    SimdReLU(in.data(), out_simd.data(), N);
    float diff = MaxDiff(out_s.data(), out_simd.data(), N);
    std::cout << "  Size: " << N << "  Max diff: " << std::setprecision(10) << diff
              << "  " << (diff < 1e-5f ? "✓ exact match" : "✗") << "\n";

    auto t_s = Benchmark([&]{ ScalarReLU(in.data(), out_s.data(), N); }, 200);
    auto t_i = Benchmark([&]{ SimdReLU(in.data(), out_simd.data(), N); }, 200);
    std::cout << "  Scalar: " << std::fixed << std::setprecision(3) << t_s
              << " ms  |  SIMD: " << std::setprecision(3) << t_i
              << " ms  |  Speedup: " << std::setprecision(2) << (t_s / t_i) << "x\n";
    std::cout << "  (注意：简单 ReLU 编译器可能自动向量化，speedup 不明显)\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 3: Sigmoid — Scalar vs SIMD
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 3: Sigmoid Scalar vs SIMD ===\n";
  {
    constexpr uint64_t N = 500000;
    std::vector<float> in(N), out_s(N), out_simd(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(-2, 2);
    for (auto& v : in) v = dist(rng);

    ScalarSigmoid(in.data(), out_s.data(), N);
    SimdSigmoid(in.data(), out_simd.data(), N);
    float diff = MaxDiff(out_s.data(), out_simd.data(), N);
    std::cout << "  Size: " << N << "  Max diff: " << std::setprecision(6) << diff << "\n";
    std::cout << "  Match (tol 5e-3): " << (diff < 5e-3f ? "YES ✓" : "NO ✗") << "\n";
    std::cout << "  (SIMD exp 用 6 项 Taylor 级数近似，精度 ~1e-6)\n";

    // 采样打印
    std::cout << "  Sample values:\n";
    for (int idx : {0, 100, 5000, 10000}) {
      std::cout << "    x=" << std::setw(8) << std::fixed << std::setprecision(3) << in[idx]
                << " | scalar=" << std::setw(10) << std::setprecision(6) << out_s[idx]
                << " | simd=" << std::setw(10) << std::setprecision(6) << out_simd[idx] << "\n";
    }

    auto t_s = Benchmark([&]{ ScalarSigmoid(in.data(), out_s.data(), N); }, 50);
    auto t_i = Benchmark([&]{ SimdSigmoid(in.data(), out_simd.data(), N); }, 50);
    std::cout << "  Scalar: " << std::fixed << std::setprecision(3) << t_s
              << " ms  |  SIMD: " << std::setprecision(3) << t_i
              << " ms  |  Speedup: " << std::setprecision(2) << (t_s / t_i) << "x\n";
    std::cout << "  (std::exp 很慢，SIMD 多项式近似优势明显)\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 4: HardSwish — SIMD Blend
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 4: HardSwish SIMD Blend ===\n";
  {
    std::vector<float> in;
    for (int i = -100; i <= 100; i++) in.push_back(i * 0.1f);
    uint64_t N = in.size();
    std::vector<float> out_s(N), out_simd(N);

    ScalarHardSwish(in.data(), out_s.data(), N);
    SimdHardSwish(in.data(), out_simd.data(), N);

    float diff = MaxDiff(out_s.data(), out_simd.data(), N);
    std::cout << "  Size: " << N << " (覆盖 -10..10)  Max diff: "
              << std::setprecision(10) << diff << "\n";
    std::cout << "  Match: " << (diff < 1e-5f ? "YES ✓" : "NO ✗") << "\n";

    std::cout << "  Boundary checks:\n";
    auto check = [&](float x) {
      float fs = (x <= -3) ? 0 : (x >= 3) ? x : x*(x+3)/6;
      int idx = static_cast<int>((x + 10.0f) / 0.1f);
      std::cout << "    x=" << std::setw(6) << std::fixed << std::setprecision(1) << x
                << " -> scalar=" << std::setw(10) << std::setprecision(4) << fs
                << "  simd=" << std::setw(10) << std::setprecision(4) << out_simd[idx] << "\n";
    };
    check(-3.0f); check(-2.9f); check(0.0f); check(2.9f); check(3.0f); check(5.0f);

    {
      constexpr uint64_t BN = 2000000;
      std::vector<float> bin(BN), bout_s(BN), bout_simd(BN);
      std::mt19937 rng(42);
      std::normal_distribution<float> dist(0, 3);
      for (auto& v : bin) v = dist(rng);

      auto t_s = Benchmark([&]{ ScalarHardSwish(bin.data(), bout_s.data(), BN); }, 200);
      auto t_i = Benchmark([&]{ SimdHardSwish(bin.data(), bout_simd.data(), BN); }, 200);
      std::cout << "\n  Performance (N=" << BN << "):\n";
      std::cout << "    Scalar: " << std::fixed << std::setprecision(3) << t_s
                << " ms  |  SIMD: " << std::setprecision(3) << t_i
                << " ms  |  Speedup: " << std::setprecision(2) << (t_s / t_i) << "x\n";
      std::cout << "  (SIMD 消除了 if-else 分支，blend 指令无分支预测开销)\n";
    }
    std::cout << "\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 5: MatMul — SIMD 向量化
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 5: MatMul SIMD 向量化 ===\n";
  {
    struct Case { uint32_t M, K, N; std::string name; };
    std::vector<Case> cases = {
      {64, 128, 128, "FC: [64,128] x [128,128]"},
      {128, 256, 256, "FC: [128,256] x [256,256]"},
      {256, 512, 512, "FC: [256,512] x [512,256]"},
    };

    for (const auto& c : cases) {
      uint64_t size_a = static_cast<uint64_t>(c.M) * c.K;
      uint64_t size_b = static_cast<uint64_t>(c.K) * c.N;
      uint64_t size_c = static_cast<uint64_t>(c.M) * c.N;

      std::vector<float> A(size_a), B(size_b), Cs(size_c), Csimd(size_c);
      std::mt19937 rng(42);
      std::normal_distribution<float> dist(0, 0.1f);
      for (auto& v : A) v = dist(rng);
      for (auto& v : B) v = dist(rng);

      ScalarMatMul(A.data(), B.data(), Cs.data(), c.M, c.K, c.N);
      SimdMatMul8(A.data(), B.data(), Csimd.data(), c.M, c.K, c.N);

      float diff = MaxDiff(Cs.data(), Csimd.data(), size_c);
      std::cout << "  " << c.name << "\n";
      std::cout << "    max_diff=" << std::setprecision(8) << diff
                << "  " << (diff < 1e-2f ? "✓" : "✗") << "\n";

      auto t_s = Benchmark([&]{ ScalarMatMul(A.data(), B.data(), Cs.data(), c.M, c.K, c.N); }, 10);
      auto t_i = Benchmark([&]{ SimdMatMul8(A.data(), B.data(), Csimd.data(), c.M, c.K, c.N); }, 10);
      std::cout << "    Scalar: " << std::fixed << std::setprecision(2) << t_s
                << " ms  |  SIMD8: " << std::setprecision(2) << t_i
                << " ms  |  Speedup: " << std::setprecision(2) << (t_s / t_i) << "x\n";
    }
    std::cout << "  (MatMul 是 SIMD 收益最大的场景 — 内层循环独立累加)\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 6: 完整卷积管线 — im2col + SIMD GEMM
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 6: Conv Pipeline — im2col + SIMD GEMM ===\n";
  {
    auto input = std::make_shared<Tensor<float>>(64, 28, 28);
    input->randn(0, 0.5f);
    auto weight = std::make_shared<Tensor<float>>(128, 64, 3, 3);
    weight->randn(0, 0.1f);

    uint32_t in_c = 64, in_h = 28, in_w = 28;
    uint32_t out_c = 128, kh = 3, kw = 3;
    uint32_t stride_h = 1, stride_w = 1, pad_h = 1, pad_w = 1;
    uint32_t out_h = (in_h + 2*pad_h - kh)/stride_h + 1;  // 28
    uint32_t out_w = (in_w + 2*pad_w - kw)/stride_w + 1;  // 28

    uint32_t col_rows = in_c * kh * kw;  // 576
    uint32_t col_cols = out_h * out_w;   // 784

    // im2col: 将卷积窗口展平为列
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

    // Weight matrix [out_c, in_c*kh*kw]
    std::vector<float> wt_mat(out_c * col_rows);
    for (uint32_t k = 0; k < out_c; k++)
      for (uint32_t ic = 0; ic < in_c; ic++)
        for (uint32_t jkh = 0; jkh < kh; jkh++)
          for (uint32_t jkw = 0; jkw < kw; jkw++) {
            uint32_t col = ic * kh * kw + jkh * kw + jkw;
            wt_mat[k * col_rows + col] = weight->at(k, ic, jkh, jkw);
          }

    // GEMM: [out_c, col_rows] x [col_rows, col_cols] -> [out_c, col_cols]
    std::vector<float> gemm_s(out_c * col_cols), gemm_i(out_c * col_cols);
    ScalarMatMul(wt_mat.data(), im2col_mat.data(), gemm_s.data(), out_c, col_rows, col_cols);
    SimdMatMul8(wt_mat.data(), im2col_mat.data(), gemm_i.data(), out_c, col_rows, col_cols);

    float diff = MaxDiff(gemm_s.data(), gemm_i.data(), out_c * col_cols);
    std::cout << "  Conv: [" << in_c << "," << in_h << "," << in_w << "] x ["
              << out_c << "," << in_c << "," << kh << "x" << kw << "]\n";
    std::cout << "  im2col: [" << col_rows << ", " << col_cols << "]\n";
    std::cout << "  GEMM max_diff: " << std::setprecision(8) << diff
              << "  " << (diff < 1e-1f ? "✓" : "✗") << "\n";

    auto t_s = Benchmark([&]{
      ScalarMatMul(wt_mat.data(), im2col_mat.data(), gemm_s.data(), out_c, col_rows, col_cols);
    }, 10);
    auto t_i = Benchmark([&]{
      SimdMatMul8(wt_mat.data(), im2col_mat.data(), gemm_i.data(), out_c, col_rows, col_cols);
    }, 10);
    std::cout << "  Scalar GEMM: " << std::fixed << std::setprecision(2) << t_s << " ms/iter\n";
    std::cout << "  SIMD8  GEMM: " << std::fixed << std::setprecision(2) << t_i << " ms/iter\n";
    std::cout << "  Speedup:     " << std::fixed << std::setprecision(2) << (t_s / t_i) << "x\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 7: 编译器自动向量化 vs 手写 SIMD
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 7: 编译器自动向量化讨论 ===\n";
  {
    constexpr uint64_t N = 5000000;
    std::vector<float> a(N), b(N), out_s(N), out_simd(N);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 1);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    // 加法 — 编译器几乎总是能自动向量化
    auto scalar_add = [&]{
      for (uint64_t i = 0; i < N; i++) out_s[i] = a[i] + b[i];
    };
    auto simd_add = [&]{
      uint64_t i = 0;
      for (; i + 8 <= N; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vb = _mm256_loadu_ps(b.data() + i);
        _mm256_storeu_ps(out_simd.data() + i, _mm256_add_ps(va, vb));
      }
      for (; i < N; i++) out_simd[i] = a[i] + b[i];
    };
    scalar_add(); simd_add();
    float diff = MaxDiff(out_s.data(), out_simd.data(), N);
    std::cout << "  Add (N=" << N << ")  max_diff=" << std::setprecision(10) << diff << "\n";
    auto ta_s = Benchmark(scalar_add, 200);
    auto ta_i = Benchmark(simd_add, 200);
    std::cout << "    Scalar: " << std::fixed << std::setprecision(3) << ta_s
              << " ms  |  SIMD: " << std::setprecision(3) << ta_i
              << " ms  |  ratio: " << std::setprecision(2) << (ta_s / ta_i) << "x\n";
    std::cout << "  -> -O3 下编译器自动将 add 循环向量化，手写 SIMD 无优势\n";

    // Mul
    auto scalar_mul = [&]{
      for (uint64_t i = 0; i < N; i++) out_s[i] = a[i] * b[i];
    };
    auto simd_mul = [&]{
      uint64_t i = 0;
      for (; i + 8 <= N; i += 8) {
        __m256 va = _mm256_loadu_ps(a.data() + i);
        __m256 vb = _mm256_loadu_ps(b.data() + i);
        _mm256_storeu_ps(out_simd.data() + i, _mm256_mul_ps(va, vb));
      }
      for (; i < N; i++) out_simd[i] = a[i] * b[i];
    };
    scalar_mul(); simd_mul();
    diff = MaxDiff(out_s.data(), out_simd.data(), N);
    std::cout << "  Mul (N=" << N << ")  max_diff=" << std::setprecision(10) << diff << "\n";
    auto tm_s = Benchmark(scalar_mul, 200);
    auto tm_i = Benchmark(simd_mul, 200);
    std::cout << "    Scalar: " << std::fixed << std::setprecision(3) << tm_s
              << " ms  |  SIMD: " << std::setprecision(3) << tm_i
              << " ms  |  ratio: " << std::setprecision(2) << (tm_s / tm_i) << "x\n";
    std::cout << "  -> Mul 同样被编译器自动向量化\n";
    std::cout << "\n  结论：手写 SIMD 的真正价值在于\n";
    std::cout << "    1. 复杂函数（exp, log, sqrt）— 编译器不会自动替换多项式近似\n";
    std::cout << "    2. 条件选择（if-else）— 编译器很难将分支转为 blend mask\n";
    std::cout << "    3. MatMul GEMM — 编译器对嵌套循环的向量化有限\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 8: 总结
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 8: SIMD 编程模式总结 ===\n\n";
  std::cout << "  +------------------------------------------------------------------+\n";
  std::cout << "  |              SIMD 编程三板斧                                      |\n";
  std::cout << "  +------------------------------------------------------------------+\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  |  1. LOAD   从内存加载数据到向量寄存器                               |\n";
  std::cout << "  |     _mm256_loadu_ps(ptr)  ->  8 floats into __m256                |\n";
  std::cout << "  |     _mm256_set1_ps(val)   ->  broadcast scalar to 8 lanes         |\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  |  2. COMPUTE 一条指令，多组数据                                      |\n";
  std::cout << "  |     _mm256_add_ps(a,b)    ->  8 additions                         |\n";
  std::cout << "  |     _mm256_mul_ps(a,b)    ->  8 multiplications                   |\n";
  std::cout << "  |     _mm256_max_ps(a,b)    ->  8 comparisons (ReLU)                |\n";
  std::cout << "  |     _mm256_cmp_ps(a,b,cmp)->  8 comparisons -> mask               |\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  |  3. STORE  写回内存                                                |\n";
  std::cout << "  |     _mm256_storeu_ps(ptr, val) ->  8 floats to memory             |\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  |  + Tail Handling: 循环外处理剩余不足 8 个的元素                     |\n";
  std::cout << "  |    for (; i + 8 <= n; i += 8) { SIMD }                           |\n";
  std::cout << "  |    for (; i < n; i++) { scalar }                                 |\n";
  std::cout << "  |                                                                    |\n";
  std::cout << "  +------------------------------------------------------------------+\n\n";
  std::cout << "  关键指令速查:\n";
  std::cout << "    算术: _mm256_add_ps, _mm256_sub_ps, _mm256_mul_ps, _mm256_div_ps\n";
  std::cout << "    比较: _mm256_max_ps, _mm256_min_ps, _mm256_cmp_ps\n";
  std::cout << "    逻辑: _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps  (用于 blend)\n";
  std::cout << "    设置: _mm256_set1_ps, _mm256_setzero_ps, _mm256_set_ps\n";
  std::cout << "    内存: _mm256_loadu_ps, _mm256_storeu_ps (unaligned)\n";
  std::cout << "          _mm256_load_ps,  _mm256_store_ps  (256-bit aligned)\n";
  std::cout << "    FMA:  _mm256_fmadd_ps(a,b,c) -> a*b + c (Fused Multiply-Add)\n\n";
  std::cout << "  手写 SIMD 的三大高价值场景:\n";
  std::cout << "    1.  transcendental 函数（exp, log, sin, sigmoid）\n";
  std::cout << "       用多项式近似替代库函数，精度 ~1e-4，速度 5-10x\n";
  std::cout << "    2.  条件分支（if-else）-> blend mask，消除分支预测失败\n";
  std::cout << "    3.  GEMM / MatMul 内层循环向量化，8x 并行累加\n\n";
  std::cout << "  生产框架中的实践:\n";
  std::cout << "    source/layer/details/simd.cpp 实现了 6 种 SIMD 激活函数\n";
  std::cout << "    使用 fmath 库的 exp_ps256() — 工业级多项式近似，精度 ~1e-4\n";
  std::cout << "    编译标志: -march=native 启用 CPU 最高指令集\n\n";

  std::cout << "All Day 17 demos completed successfully!\n";
  return 0;
}
