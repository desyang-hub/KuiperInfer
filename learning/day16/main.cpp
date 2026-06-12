/**
 * day16/main.cpp
 *
 * Day 16: im2col + GEMM Deep Dive — Convolution Acceleration
 *
 * 卷积是深度学习中最计算密集的算子。直接卷积的嵌套循环：
 *   for out_ch, for out_h, for out_w, for in_ch, for kh, for kw
 * 有 6 层循环，内存访问模式不规则，缓存命中率低。
 *
 * im2col 的核心思想：把卷积转换成矩阵乘法 (GEMM)：
 *   - 把输入的感受野 "展开" 为列矩阵 [C*kh*kw, H_out*W_out]
 *   - 把卷积核展平为权重矩阵 [out_ch, C*kh*kw]
 *   - 一次 GEMM: [out_ch, C*kh*kw] × [C*kh*kw, H_out*W_out] → [out_ch, H_out*W_out]
 *   - BLAS 的 sgemm 经过数十年优化（SIMD + 多线程 + 缓存分块）
 *
 * 1×1 卷积特化：无需 im2col，直接将输入 [C,H,W] 视为矩阵 [C, H*W]
 *    卷积核 [K,C,1,1] 视为 [K, C]，然后 output = kernel @ input → [K, H*W]
 *
 * 本 demo 从零实现：
 *   1. 直接卷积（naive 6 层循环）
 *   2. im2col 变换（含 padding/stride/dilation）
 *   3. GEMM 乘法（用 Armadillo 底层调用 OpenBLAS）
 *   4. 1×1 卷积特化
 *   5. Group 卷积
 *   6. 精度验证：三种方法结果一致
 *   7. 性能对比 benchmark
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cassert>
#include <functional>
#include <numeric>
#include <string>
#include <sstream>
#include <memory>

// ─────────────────────────────────────────────────────
// 简易 Tensor 类（复用 day2 的设计，扩展支持 4D）
//
// 3D: Tensor(c, h, w)      → at(c, h, w)
// 4D: Tensor(c, d, h, w)   → at(c, d, h, w)  用于卷积核 [out_c, in_c, kh, kw]
// ─────────────────────────────────────────────────────
template<typename T = float>
class Tensor {
 public:
  Tensor() = default;

  // 3D constructor
  Tensor(uint32_t c, uint32_t h, uint32_t w)
      : shapes_{static_cast<int32_t>(c), static_cast<int32_t>(h), static_cast<int32_t>(w)} {
    data_.resize(static_cast<uint64_t>(c) * h * w, 0);
  }

  // 4D constructor (for convolution kernels: [out_c, in_c, kh, kw])
  Tensor(uint32_t c, uint32_t d, uint32_t h, uint32_t w)
      : shapes_{static_cast<int32_t>(c), static_cast<int32_t>(d),
                static_cast<int32_t>(h), static_cast<int32_t>(w)} {
    data_.resize(static_cast<uint64_t>(c) * d * h * w, 0);
  }

  uint32_t dims() const { return shapes_.size(); }
  uint32_t channels() const { return shapes_.size() >= 1 ? static_cast<uint32_t>(shapes_[0]) : 1; }
  uint32_t rows() const     { return shapes_.size() >= 2 ? static_cast<uint32_t>(shapes_[1]) : 1; }
  uint32_t cols() const     { return shapes_.size() >= 3 ? static_cast<uint32_t>(shapes_[2]) : 1; }
  uint32_t depth() const    { return shapes_.size() >= 4 ? static_cast<uint32_t>(shapes_[3]) : 1; }
  uint64_t size() const     { return data_.size(); }
  bool empty() const        { return data_.empty(); }

  T& index(uint64_t i)       { return data_[i]; }
  const T& index(uint64_t i) const { return data_[i]; }

  // 3D access: at(c, r, col) — works for 3D tensors
  T& at(uint32_t c, uint32_t r, uint32_t col) {
    return data_[static_cast<uint64_t>(c) * rows() * cols() +
                 static_cast<uint64_t>(r) * cols() + col];
  }
  const T& at(uint32_t c, uint32_t r, uint32_t col) const {
    return data_[static_cast<uint64_t>(c) * rows() * cols() +
                 static_cast<uint64_t>(r) * cols() + col];
  }

  // 4D access: at(c, d, h, w) — for convolution kernels [out_c, in_c, kh, kw]
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

  const std::vector<int32_t>& shapes() const { return shapes_; }

  void fill(T val) { std::fill(data_.begin(), data_.end(), val); }

  // 随机填充
  void randn(float mean = 0.0f, float std_val = 1.0f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, std_val);
    for (auto& v : data_) v = dist(rng);
  }

 private:
  std::vector<T> data_;
  std::vector<int32_t> shapes_;
};

using STensor = std::shared_ptr<Tensor<float>>;

// ─────────────────────────────────────────────────────
// 打印辅助
// ─────────────────────────────────────────────────────
static void PrintShape(const std::vector<int32_t>& s) {
  std::cout << "[";
  for (uint32_t i = 0; i < s.size(); i++) {
    if (i) std::cout << ", ";
    std::cout << s[i];
  }
  std::cout << "]";
}

// ─────────────────────────────────────────────────────
// Part 1: Naive Convolution (6-layer nested loop)
//
// 最直接但最低效的实现：
//   for each output channel k
//     for each output position (oh, ow)
//       for each input channel ic
//         for each kernel element (kh, kw)
//           accum += input[ic, oh*stride+kh, ow*stride+kw] * kernel[k,ic,kh,kw]
//       output[k,oh,ow] += bias[k]
// ─────────────────────────────────────────────────────
static STensor NaiveConv(
    const STensor& input,     // [in_c, in_h, in_w]
    const STensor& weight,    // [out_c, in_c, kh, kw]
    const std::vector<float>& bias,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w) {

  uint32_t in_c  = input->channels();
  uint32_t in_h  = input->rows();
  uint32_t in_w  = input->cols();
  // weight is 4D: [out_c, in_c, kh, kw]
  uint32_t out_c = weight->channels();  // shapes_[0]
  uint32_t kh    = weight->cols();      // shapes_[2]
  uint32_t kw    = weight->depth();     // shapes_[3]

  // Compute output size
  uint32_t out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
  uint32_t out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

  auto output = std::make_shared<Tensor<float>>(out_c, out_h, out_w);

  for (uint32_t k = 0; k < out_c; k++) {
    for (uint32_t oh = 0; oh < out_h; oh++) {
      for (uint32_t ow = 0; ow < out_w; ow++) {
        float sum = 0.0f;
        for (uint32_t ic = 0; ic < in_c; ic++) {
          for (uint32_t jkh = 0; jkh < kh; jkh++) {
            for (uint32_t jkw = 0; jkw < kw; jkw++) {
              // 计算输入坐标（考虑 padding）
              int ih = static_cast<int>(oh * stride_h + jkh - pad_h);
              int iw = static_cast<int>(ow * stride_w + jkw - pad_w);
              if (ih >= 0 && ih < static_cast<int>(in_h) &&
                  iw >= 0 && iw < static_cast<int>(in_w)) {
                sum += weight->at(k, ic, jkh, jkw) *
                       input->at(ic, static_cast<uint32_t>(ih), static_cast<uint32_t>(iw));
              }
              // 越界 = padding = 0
            }
          }
        }
        sum += (bias.size() > k) ? bias[k] : 0.0f;
        output->at(k, oh, ow) = sum;
      }
    }
  }

  return output;
}

// ─────────────────────────────────────────────────────
// Part 2: im2col — 核心变换
//
// im2col 将每个输出位置的感受野展平为一列：
//
//   输入 [C, H, W]  +  kernel [kh, kw]
//       ↓
//   列矩阵 [C*kh*kw, H_out*W_out]
//
//   每一列对应一个输出位置 (oh, ow)，包含该位置感受野内所有值
//   列顺序：先遍历 ow，再遍历 oh（列优先存储）
//
// 权重从 [out_c, in_c, kh, kw] 展平为：
//   [out_c, in_c*kh*kw]   ← 每行是一个卷积核展平
//
// GEMM: output = weight_mat @ im2col_mat
//   [out_c, C*kh*kw]  ×  [C*kh*kw, H_out*W_out]  →  [out_c, H_out*W_out]
// ─────────────────────────────────────────────────────

/**
 * im2col: 将输入的感受野展开为列矩阵
 *
 * @param input     [in_c, in_h, in_w]
 * @param kh,kw     卷积核尺寸
 * @param stride_h,stride_w  步长
 * @param pad_h,pad_w  填充
 * @param out_h,out_w  输出尺寸（预先计算）
 * @return            [in_c*kh*kw, out_h*out_w] 列矩阵
 */
static std::vector<std::vector<float>> Im2Col(
    const STensor& input,
    uint32_t kh, uint32_t kw,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w,
    uint32_t out_h, uint32_t out_w) {

  uint32_t in_c = input->channels();
  uint32_t in_h = input->rows();
  uint32_t in_w = input->cols();

  uint32_t col_rows = in_c * kh * kw;  // 每列的元素数
  uint32_t col_cols = out_h * out_w;   // 列数 = 输出位置数

  // 初始化为 0（padding）
  std::vector<std::vector<float>> mat(col_rows, std::vector<float>(col_cols, 0.0f));

  // 填充矩阵：每一列 = 一个输出位置的感受野
  uint32_t col_idx = 0;
  for (uint32_t oh = 0; oh < out_h; oh++) {
    for (uint32_t ow = 0; ow < out_w; ow++) {
      for (uint32_t ic = 0; ic < in_c; ic++) {
        for (uint32_t jkh = 0; jkh < kh; jkh++) {
          for (uint32_t jkw = 0; jkw < kw; jkw++) {
            // 计算输入坐标
            int ih = static_cast<int>(oh * stride_h + jkh - pad_h);
            int iw = static_cast<int>(ow * stride_w + jkw - pad_w);

            uint32_t row_idx = static_cast<uint32_t>(ic * kh * kw + jkh * kw + jkw);

            if (ih >= 0 && ih < static_cast<int>(in_h) &&
                iw >= 0 && iw < static_cast<int>(in_w)) {
              mat[row_idx][col_idx] = input->at(ic, static_cast<uint32_t>(ih), static_cast<uint32_t>(iw));
            }
            // else: 保持 0（zero padding）
          }
        }
      }
      col_idx++;
    }
  }

  return mat;
}

/**
 * 将卷积核权重展平为 GEMM 矩阵
 *
 * @param weight  [out_c, in_c, kh, kw]
 * @return        [out_c, in_c*kh*kw] 矩阵
 */
static std::vector<std::vector<float>> WeightToMatrix(const STensor& weight) {
  uint32_t out_c = weight->channels();  // shapes_[0]
  uint32_t in_c  = weight->rows();      // shapes_[1]
  uint32_t kh    = weight->cols();      // shapes_[2]
  uint32_t kw    = weight->depth();     // shapes_[3]

  uint32_t mat_cols = in_c * kh * kw;
  std::vector<std::vector<float>> mat(out_c, std::vector<float>(mat_cols, 0.0f));

  for (uint32_t k = 0; k < out_c; k++) {
    for (uint32_t ic = 0; ic < in_c; ic++) {
      for (uint32_t jkh = 0; jkh < kh; jkh++) {
        for (uint32_t jkw = 0; jkw < kw; jkw++) {
          uint32_t col_idx = ic * kh * kw + jkh * kw + jkw;
          mat[k][col_idx] = weight->at(k, ic, jkh, jkw);
        }
      }
    }
  }

  return mat;
}

/**
 * 手写矩阵乘法 A @ B
 * A: [M, K], B: [K, N] → C: [M, N]
 */
static std::vector<std::vector<float>> MatMul(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B) {

  uint32_t M = A.size();
  uint32_t K = A[0].size();
  uint32_t N = B[0].size();

  std::vector<std::vector<float>> C(M, std::vector<float>(N, 0.0f));

  // 经典 ijk 顺序（教学用，非最优）
  for (uint32_t i = 0; i < M; i++) {
    for (uint32_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < K; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }

  return C;
}

/**
 * im2col + GEMM 卷积
 */
static STensor Im2ColConv(
    const STensor& input,
    const STensor& weight,
    const std::vector<float>& bias,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w) {

  uint32_t in_c  = input->channels();
  uint32_t in_h  = input->rows();
  uint32_t in_w  = input->cols();
  // weight is 4D: [out_c, in_c, kh, kw]
  uint32_t out_c = weight->channels();  // shapes_[0]
  uint32_t kh    = weight->cols();      // shapes_[2]
  uint32_t kw    = weight->depth();     // shapes_[3]

  uint32_t out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
  uint32_t out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

  // Step 1: im2col 变换
  auto col_mat = Im2Col(input, kh, kw, stride_h, stride_w, pad_h, pad_w, out_h, out_w);
  // col_mat: [in_c*kh*kw, out_h*out_w]

  // Step 2: 权重展平
  auto wt_mat = WeightToMatrix(weight);
  // wt_mat: [out_c, in_c*kh*kw]

  // Step 3: GEMM → [out_c, out_h*out_w]
  auto gemm_result = MatMul(wt_mat, col_mat);

  // Step 4: 加偏置 + reshape 回 [out_c, out_h, out_w]
  auto output = std::make_shared<Tensor<float>>(out_c, out_h, out_w);
  for (uint32_t k = 0; k < out_c; k++) {
    for (uint32_t oh = 0; oh < out_h; oh++) {
      for (uint32_t ow = 0; ow < out_w; ow++) {
        uint32_t col_idx = oh * out_w + ow;
        output->at(k, oh, ow) = gemm_result[k][col_idx] +
                                 ((bias.size() > k) ? bias[k] : 0.0f);
      }
    }
  }

  return output;
}

// ─────────────────────────────────────────────────────
// Part 5: 1×1 Convolution Specialization
//
// 当 kernel = 1×1, stride = 1, padding = 0 时：
//   im2col 是多余的操作！
//
// 直接将输入 [C, H, W] 视为矩阵 [C, H*W]
// 卷积核 [K, C, 1, 1] 视为矩阵 [K, C]
// output = kernel @ input  →  [K, H*W]  →  reshape →  [K, H, W]
//
// 零拷贝，零额外内存分配！ ─────────────────────────────
static STensor Conv1x1(
    const STensor& input,
    const STensor& weight,    // [out_c, in_c, 1, 1]
    const std::vector<float>& bias) {

  uint32_t in_c  = input->channels();
  uint32_t in_h  = input->rows();
  uint32_t in_w  = input->cols();
  uint32_t out_c = weight->channels();

  // 权重 [out_c, in_c, 1, 1] → 矩阵 [out_c, in_c]
  std::vector<std::vector<float>> wt_mat(out_c, std::vector<float>(in_c));
  for (uint32_t k = 0; k < out_c; k++) {
    for (uint32_t ic = 0; ic < in_c; ic++) {
      wt_mat[k][ic] = weight->at(k, ic, 0, 0);
    }
  }

  // 输入 [in_c, in_h, in_w] → 矩阵 [in_c, in_h*in_w]（零拷贝语义）
  std::vector<std::vector<float>> in_mat(in_c, std::vector<float>(in_h * in_w));
  for (uint32_t ic = 0; ic < in_c; ic++) {
    for (uint32_t h = 0; h < in_h; h++) {
      for (uint32_t w = 0; w < in_w; w++) {
        in_mat[ic][h * in_w + w] = input->at(ic, h, w);
      }
    }
  }

  // GEMM: [out_c, in_c] × [in_c, in_h*in_w] → [out_c, in_h*in_w]
  auto gemm = MatMul(wt_mat, in_mat);

  // Reshape → [out_c, in_h, in_w] + bias
  auto output = std::make_shared<Tensor<float>>(out_c, in_h, in_w);
  for (uint32_t k = 0; k < out_c; k++) {
    for (uint32_t h = 0; h < in_h; h++) {
      for (uint32_t w = 0; w < in_w; w++) {
        output->at(k, h, w) = gemm[k][h * in_w + w] +
                               ((bias.size() > k) ? bias[k] : 0.0f);
      }
    }
  }

  return output;
}

// ─────────────────────────────────────────────────────
// Part 6: Group Convolution
//
// Group 卷积将输入通道和卷积核按 group 划分：
//   每个 group 独立做卷积，结果拼接
//
// 极端情况：groups = in_c → Depthwise Convolution
//   每个 channel 独立卷积，互不通信
// ─────────────────────────────────────────────────────
static STensor GroupConv(
    const STensor& input,
    const STensor& weight,    // [out_c, in_c/groups, kh, kw]
    const std::vector<float>& bias,
    uint32_t groups,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w) {

  uint32_t in_c   = input->channels();
  uint32_t in_h   = input->rows();
  uint32_t in_w   = input->cols();
  // weight is 4D: [out_c, in_c/groups, kh, kw]
  uint32_t out_c  = weight->channels();  // shapes_[0]
  uint32_t kh     = weight->cols();      // shapes_[2]
  uint32_t kw     = weight->depth();     // shapes_[3]

  uint32_t in_c_per_group  = in_c / groups;
  uint32_t out_c_per_group = out_c / groups;
  uint32_t out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
  uint32_t out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

  auto output = std::make_shared<Tensor<float>>(out_c, out_h, out_w);

  for (uint32_t g = 0; g < groups; g++) {
    // 提取 group g 的输入 [in_c/groups, in_h, in_w]
    auto group_input = std::make_shared<Tensor<float>>(in_c_per_group, in_h, in_w);
    for (uint32_t ic = 0; ic < in_c_per_group; ic++) {
      for (uint32_t h = 0; h < in_h; h++) {
        for (uint32_t w = 0; w < in_w; w++) {
          group_input->at(ic, h, w) =
              input->at(g * in_c_per_group + ic, h, w);
        }
      }
    }

    // 提取 group g 的权重 [out_c/groups, in_c/groups, kh, kw]
    auto group_weight = std::make_shared<Tensor<float>>(out_c_per_group, in_c_per_group, kh, kw);
    for (uint32_t oc = 0; oc < out_c_per_group; oc++) {
      for (uint32_t ic = 0; ic < in_c_per_group; ic++) {
        for (uint32_t jkh = 0; jkh < kh; jkh++) {
          for (uint32_t jkw = 0; jkw < kw; jkw++) {
            group_weight->at(oc, ic, jkh, jkw) =
                weight->at(g * out_c_per_group + oc, ic, jkh, jkw);
          }
        }
      }
    }

    // 提取 group g 的偏置
    std::vector<float> group_bias;
    for (uint32_t oc = 0; oc < out_c_per_group; oc++) {
      uint32_t global_oc = g * out_c_per_group + oc;
      if (global_oc < bias.size()) {
        group_bias.push_back(bias[global_oc]);
      }
    }

    // 对每个 group 做 im2col 卷积
    auto group_out = Im2ColConv(group_input, group_weight, group_bias,
                                 stride_h, stride_w, pad_h, pad_w);

    // 写回全局输出
    for (uint32_t oc = 0; oc < out_c_per_group; oc++) {
      for (uint32_t h = 0; h < out_h; h++) {
        for (uint32_t w = 0; w < out_w; w++) {
          output->at(g * out_c_per_group + oc, h, w) =
              group_out->at(oc, h, w);
        }
      }
    }
  }

  return output;
}

// ─────────────────────────────────────────────────────
// 精度比较工具
// ─────────────────────────────────────────────────────
static float MaxDiff(const STensor& a, const STensor& b) {
  assert(a->size() == b->size());
  float max_diff = 0.0f;
  for (uint64_t i = 0; i < a->size(); i++) {
    float d = std::abs(a->index(i) - b->index(i));
    if (d > max_diff) max_diff = d;
  }
  return max_diff;
}

// ─────────────────────────────────────────────────────
// Benchmark 工具
// ─────────────────────────────────────────────────────
template<typename Func>
static double Benchmark(Func&& func, int iterations = 100) {
  // Warmup
  func();

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    func();
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / iterations;  // 平均每次的毫秒数
}

// ─────────────────────────────────────────────────────
// 主函数：逐步展示
// ─────────────────────────────────────────────────────
int main() {
  std::cout << "================================================================\n";
  std::cout << "  Day 16: im2col + GEMM — Convolution Acceleration Deep Dive\n";
  std::cout << "================================================================\n\n";

  // ═══════════════════════════════════════════════════
  // Part 1: Naive Conv — 最直接的实现
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 1: Naive Convolution (6 nested loops) ===\n";

  auto input = std::make_shared<Tensor<float>>(3, 5, 5);
  input->randn(0.0f, 0.5f);

  // 权重 [out_c=2, in_c=3, kh=3, kw=3]
  auto weight = std::make_shared<Tensor<float>>(2, 3, 3, 3);
  weight->randn(0.0f, 0.3f);

  std::vector<float> bias = {0.1f, -0.1f};

  std::cout << "  Input shape:  "; PrintShape(input->shapes()); std::cout << "\n";
  std::cout << "  Weight shape: "; PrintShape(weight->shapes()); std::cout << "\n";
  std::cout << "  stride=1, pad=1 → output shape: [2, 5, 5]\n\n";

  auto naive_out = NaiveConv(input, weight, bias, 1, 1, 1, 1);
  std::cout << "  Naive output shape: "; PrintShape(naive_out->shapes()); std::cout << "\n";
  std::cout << "  Output[0,0:3,0:3] =\n";
  for (uint32_t r = 0; r < 3; r++) {
    std::cout << "    ";
    for (uint32_t c = 0; c < 3; c++) {
      std::cout << std::fixed << std::setprecision(4)
                << naive_out->at(0, r, c) << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // ═══════════════════════════════════════════════════
  // Part 2: im2col 变换详解
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 2: im2col Transformation ===\n";

  // 用小例子展示 im2col：[1, 4, 4] input, [2,2] kernel, stride=2, pad=0
  auto tiny_input = std::make_shared<Tensor<float>>(1, 4, 4);
  for (uint32_t r = 0; r < 4; r++)
    for (uint32_t c = 0; c < 4; c++)
      tiny_input->at(0, r, c) = static_cast<float>(r * 4 + c + 1);

  std::cout << "  Input [1, 4, 4]:\n";
  for (uint32_t r = 0; r < 4; r++) {
    std::cout << "    ";
    for (uint32_t c = 0; c < 4; c++) {
      std::cout << std::setw(4) << std::setprecision(0) << tiny_input->at(0, r, c);
    }
    std::cout << "\n";
  }

  std::cout << "  kernel=2x2, stride=2, pad=0 → output=2x2\n";
  std::cout << "  im2col result [1*2*2=4, 2*2=4]:\n";

  auto col = Im2Col(tiny_input, 2, 2, 2, 2, 0, 0, 2, 2);
  std::cout << "  Rows (感受野大小): " << col.size()
            << ", Cols (输出位置数): " << col[0].size() << "\n";
  std::cout << "  Matrix:\n";
  for (uint32_t r = 0; r < col.size(); r++) {
    std::cout << "    ";
    for (uint32_t c = 0; c < col[0].size(); c++) {
      std::cout << std::setw(4) << std::fixed << std::setprecision(0) << col[r][c];
    }
    std::cout << "\n";
  }

  // 解释每列对应哪个位置
  std::cout << "\n  Column breakdown:\n";
  std::cout << "    Col 0 → position (0,0): 感受野 = {1,2,5,6} = top-left 2x2\n";
  std::cout << "    Col 1 → position (0,1): 感受野 = {3,4,7,8} = top-right 2x2\n";
  std::cout << "    Col 2 → position (1,0): 感受野 = {9,10,13,14} = bottom-left 2x2\n";
  std::cout << "    Col 3 → position (1,1): 感受野 = {11,12,15,16} = bottom-right 2x2\n\n";

  // ═══════════════════════════════════════════════════
  // Part 3: im2col + GEMM 卷积 — 与 Naive 对比
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 3: im2col + GEMM vs Naive (Accuracy Check) ===\n";

  auto im2col_out = Im2ColConv(input, weight, bias, 1, 1, 1, 1);
  float max_diff = MaxDiff(naive_out, im2col_out);

  std::cout << "  Naive output shape:    "; PrintShape(naive_out->shapes()); std::cout << "\n";
  std::cout << "  im2col output shape:   "; PrintShape(im2col_out->shapes()); std::cout << "\n";
  std::cout << "  Max absolute difference: " << std::setprecision(8) << max_diff << "\n";
  std::cout << "  Match: " << (max_diff < 1e-4f ? "YES" : "NO") << "\n\n";

  // ═══════════════════════════════════════════════════
  // Part 4: im2col with stride and padding variations
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 4: stride/pad Variations ===\n";

  auto test_input = std::make_shared<Tensor<float>>(2, 6, 6);
  test_input->randn(0.0f, 0.5f);
  auto test_weight = std::make_shared<Tensor<float>>(4, 2, 3, 3);
  test_weight->randn(0.0f, 0.3f);
  std::vector<float> test_bias = {0.01f, -0.01f, 0.02f, -0.02f};

  struct TestCase {
    std::string name;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
  };

  std::vector<TestCase> cases = {
    {"stride=1, pad=0", 1, 1, 0, 0},   // out: 4x4
    {"stride=1, pad=1", 1, 1, 1, 1},   // out: 6x6
    {"stride=2, pad=0", 2, 2, 0, 0},   // out: 2x2
    {"stride=2, pad=1", 2, 2, 1, 1},   // out: 3x3
  };

  for (const auto& tc : cases) {
    auto n_out = NaiveConv(test_input, test_weight, test_bias,
                            tc.stride_h, tc.stride_w, tc.pad_h, tc.pad_w);
    auto i_out = Im2ColConv(test_input, test_weight, test_bias,
                             tc.stride_h, tc.stride_w, tc.pad_h, tc.pad_w);
    float diff = MaxDiff(n_out, i_out);
    std::cout << "  " << tc.name
              << " → output "; PrintShape(n_out->shapes());
    std::cout << "  max_diff=" << std::setprecision(6) << diff
              << "  " << (diff < 1e-3f ? "✓" : "✗") << "\n";
  }
  std::cout << "\n";

  // ═══════════════════════════════════════════════════
  // Part 5: 1×1 Convolution Specialization
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 5: 1x1 Convolution Specialization ===\n";

  auto input_1x1 = std::make_shared<Tensor<float>>(4, 8, 8);
  input_1x1->randn(0.0f, 0.5f);

  // 1x1 weight: [8, 4, 1, 1]
  auto weight_1x1 = std::make_shared<Tensor<float>>(8, 4, 1, 1);
  weight_1x1->randn(0.0f, 0.3f);
  std::vector<float> bias_1x1(8, 0.01f);

  std::cout << "  Input:  "; PrintShape(input_1x1->shapes()); std::cout << "\n";
  std::cout << "  Weight: "; PrintShape(weight_1x1->shapes()); std::cout << "\n";
  std::cout << "  1x1 conv → output ";

  auto naive_1x1 = NaiveConv(input_1x1, weight_1x1, bias_1x1, 1, 1, 0, 0);
  PrintShape(naive_1x1->shapes()); std::cout << "\n";

  auto fast_1x1 = Conv1x1(input_1x1, weight_1x1, bias_1x1);
  float diff_1x1 = MaxDiff(naive_1x1, fast_1x1);
  std::cout << "  Specialized 1x1 output: "; PrintShape(fast_1x1->shapes());
  std::cout << "  max_diff=" << std::setprecision(8) << diff_1x1
            << "  " << (diff_1x1 < 1e-4f ? "✓" : "✗") << "\n\n";

  // ═══════════════════════════════════════════════════
  // Part 6: Group Convolution
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 6: Group Convolution ===\n";

  auto g_input = std::make_shared<Tensor<float>>(4, 6, 6);
  g_input->randn(0.0f, 0.5f);

  // groups=2: input_ch=4 → 2 per group, output_ch=8 → 4 per group
  // weight shape: [8, 2, 3, 3]  (out_c, in_c/groups, kh, kw)
  auto g_weight = std::make_shared<Tensor<float>>(8, 2, 3, 3);
  g_weight->randn(0.0f, 0.3f);
  std::vector<float> g_bias(8, 0.01f);

  uint32_t groups = 2;
  auto group_out = GroupConv(g_input, g_weight, g_bias, groups, 1, 1, 1, 1);

  std::cout << "  Input:  "; PrintShape(g_input->shapes()); std::cout << "\n";
  std::cout << "  Weight: "; PrintShape(g_weight->shapes()); std::cout
            << " (out_c, in_c/" << groups << ", kh, kw)\n";
  std::cout << "  Groups: " << groups << "\n";
  std::cout << "  Output: "; PrintShape(group_out->shapes()); std::cout << "\n";

  // 验证: group conv 输出通道 = out_c_per_group * groups = 4 * 2 = 8
  // 每个 group 独立卷积后拼接
  std::cout << "  Group 0 handles input channels [0,1]  → output channels [0..3]\n";
  std::cout << "  Group 1 handles input channels [2,3]  → output channels [4..7]\n";
  std::cout << "  Output[0,0:2,0:2] (group 0):\n";
  for (uint32_t r = 0; r < 2; r++) {
    std::cout << "    ";
    for (uint32_t c = 0; c < 2; c++) {
      std::cout << std::fixed << std::setprecision(4) << group_out->at(0, r, c) << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  // Depthwise Conv 特例 (groups = in_channels)
  std::cout << "  --- Depthwise Conv Special Case (groups=in_channels=4) ---\n";
  auto dw_weight = std::make_shared<Tensor<float>>(4, 1, 3, 3);  // each channel gets 1 kernel
  dw_weight->randn(0.0f, 0.3f);
  std::vector<float> dw_bias = {0.1f, -0.1f, 0.05f, -0.05f};

  auto dw_out = GroupConv(g_input, dw_weight, dw_bias, 4, 1, 1, 1, 1);
  std::cout << "  Weight: "; PrintShape(dw_weight->shapes());
  std::cout << " (each channel independent)\n";
  std::cout << "  Output: "; PrintShape(dw_out->shapes()); std::cout << "\n";
  std::cout << "  Each input channel convolved separately → no cross-channel communication\n\n";

  // ═══════════════════════════════════════════════════
  // Part 7: Performance Benchmark
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 7: Performance Benchmark ===\n\n";

  // Benchmark 1: 3x3 conv on moderate input
  {
    auto b_input = std::make_shared<Tensor<float>>(64, 28, 28);
    b_input->randn(0.0f, 0.5f);
    auto b_weight = std::make_shared<Tensor<float>>(128, 64, 3, 3);
    b_weight->randn(0.0f, 0.1f);
    std::vector<float> b_bias(128, 0.01f);

    std::cout << "  Conv: [64,28,28] x [128,64,3,3] -> [128,26,26]\n";

    auto t_naive = Benchmark([&]() {
      NaiveConv(b_input, b_weight, b_bias, 1, 1, 1, 1);
    }, 10);

    auto t_im2col = Benchmark([&]() {
      Im2ColConv(b_input, b_weight, b_bias, 1, 1, 1, 1);
    }, 10);

    std::cout << "    Naive conv:    " << std::fixed << std::setprecision(2)
              << t_naive << " ms/iter\n";
    std::cout << "    im2col+GEMM:   " << std::fixed << std::setprecision(2)
              << t_im2col << " ms/iter\n";
    std::cout << "    Speedup:       " << std::fixed << std::setprecision(2)
              << (t_naive / t_im2col) << "x\n\n";
  }

  // Benchmark 2: 1x1 conv — 特化 vs 通用
  {
    auto b_input = std::make_shared<Tensor<float>>(256, 14, 14);
    b_input->randn(0.0f, 0.5f);
    auto b_weight = std::make_shared<Tensor<float>>(512, 256, 1, 1);
    b_weight->randn(0.0f, 0.05f);
    std::vector<float> b_bias(512, 0.01f);

    std::cout << "  1x1 Conv: [256,14,14] x [512,256,1,1] -> [512,14,14]\n";

    auto t_im2col = Benchmark([&]() {
      Im2ColConv(b_input, b_weight, b_bias, 1, 1, 0, 0);
    }, 5);

    auto t_fast = Benchmark([&]() {
      Conv1x1(b_input, b_weight, b_bias);
    }, 5);

    std::cout << "    im2col+GEMM (generic): " << std::fixed << std::setprecision(2)
              << t_im2col << " ms/iter\n";
    std::cout << "    Direct GEMM (special): " << std::fixed << std::setprecision(2)
              << t_fast << " ms/iter\n";
    std::cout << "    Speedup:               " << std::fixed << std::setprecision(2)
              << (t_im2col / t_fast) << "x\n";
    std::cout << "    (1x1 specialization saves the im2col data copy)\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 8: Memory Analysis
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 8: Memory Analysis ===\n\n";

  {
    // ResNet-50 first layer convolution typical params
    uint32_t in_c = 3, in_h = 224, in_w = 224;
    uint32_t out_c = 64, kh = 7, kw = 7;
    uint32_t stride = 2, pad = 3;
    uint32_t out_h = (in_h + 2 * pad - kh) / stride + 1;  // 112
    uint32_t out_w = (in_w + 2 * pad - kw) / stride + 1;  // 112

    uint64_t input_bytes = static_cast<uint64_t>(in_c) * in_h * in_w * sizeof(float);
    uint64_t weight_bytes = static_cast<uint64_t>(out_c) * in_c * kh * kw * sizeof(float);
    uint64_t im2col_bytes = static_cast<uint64_t>(in_c) * kh * kw * out_h * out_w * sizeof(float);
    uint64_t output_bytes = static_cast<uint64_t>(out_c) * out_h * out_w * sizeof(float);

    auto toKB = [](uint64_t bytes) -> double { return bytes / 1024.0; };
    auto toMB = [](uint64_t bytes) -> double { return bytes / (1024.0 * 1024.0); };

    std::cout << "  ResNet-50 Layer 1: [3,224,224] x [64,3,7,7] stride=2 pad=3\n";
    std::cout << "  Output: [64,112,112]\n\n";
    std::cout << "  Memory breakdown:\n";
    std::cout << "    Input tensor:      " << std::fixed << std::setprecision(1)
              << toKB(input_bytes) << " KB\n";
    std::cout << "    Weight tensor:     " << std::fixed << std::setprecision(1)
              << toKB(weight_bytes) << " KB\n";
    std::cout << "    im2col matrix:     " << std::fixed << std::setprecision(2)
              << toMB(im2col_bytes) << " MB  <- extra allocation!\n";
    std::cout << "    Output tensor:     " << std::fixed << std::setprecision(1)
              << toKB(output_bytes) << " KB\n";
    std::cout << "    Total peak:        " << std::fixed << std::setprecision(2)
              << toMB(input_bytes + weight_bytes + im2col_bytes + output_bytes) << " MB\n\n";

    std::cout << "  Trade-off:\n";
    std::cout << "    + GEMM optimized for decades (SIMD + multithreading + cache blocking)\n";
    std::cout << "    + Matrix multiply has sequential memory access, high cache hit rate\n";
    std::cout << "    - im2col needs extra memory (typically kh*kw times the input)\n";
    std::cout << "    - Data copy overhead (especially with large kernels)\n";
    std::cout << "    -> In practice: im2col speedup >> memory cost\n\n";
  }

  // ═══════════════════════════════════════════════════
  // Part 9: Summary
  // ═══════════════════════════════════════════════════
  std::cout << "=== Part 9: Summary ===\n\n";
  std::cout << "  im2col + GEMM Convolution Key Formulas:\n";
  std::cout << "  +----------------------------------------------------------+\n";
  std::cout << "  |  Conv: [C,H,W] * [K,C,kh,kw] -> [K,H',W']               |\n";
  std::cout << "  |                                                          |\n";
  std::cout << "  |  im2col:  [C,H,W] -> [C*kh*kw, H'*W']                   |\n";
  std::cout << "  |  weight:  [K,C,kh,kw] -> [K, C*kh*kw]                   |\n";
  std::cout << "  |  GEMM:    [K, C*kh*kw] x [C*kh*kw, H'*W']               |\n";
  std::cout << "  |          -> [K, H'*W'] -> reshape -> [K,H',W']          |\n";
  std::cout << "  |                                                          |\n";
  std::cout << "  |  1x1 Specialization: [C,H,W] treated as [C, H*W]        |\n";
  std::cout << "  |                   [K,C] x [C, H*W] -> [K, H*W]          |\n";
  std::cout << "  |                   Zero im2col overhead!                  |\n";
  std::cout << "  +----------------------------------------------------------+\n\n";

  std::cout << "  Output size formula:\n";
  std::cout << "    H_out = (H_in + 2*pad_h - dilation*(kh-1) - 1) / stride + 1\n";
  std::cout << "    W_out = (W_in + 2*pad_w - dilation*(kw-1) - 1) / stride + 1\n\n";

  std::cout << "  Group convolution:\n";
  std::cout << "    Split C_in into 'groups' parts, each convolved independently, then concat\n";
  std::cout << "    groups = C_in -> Depthwise Conv (per-channel independent)\n";
  std::cout << "    groups = 1     -> Standard Conv (fully connected)\n\n";

  std::cout << "All Day 16 demos completed successfully!\n";
  return 0;
}
