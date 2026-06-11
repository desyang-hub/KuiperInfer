/**
 * day1/include/tensor.hpp
 *
 * Tensor 张量类 — 深度学习推理框架的数据基石
 *
 * 核心设计：
 * - 用一维 vector 存储所有数据（行优先 row-major）
 * - 支持 1D / 2D / 3D 张量
 * - 自动维度简化：[1,1,N]→1D, [1,H,W]→2D
 * - 提供 channels(), rows(), cols(), size() 等访问方法
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstdint>
#include <iostream>
#include <numeric>  // std::accumulate
#include <cassert>
#include <iomanip>
#include <cstring>
#include <memory>

namespace learn_infer {

template <typename T = float>
class Tensor {
 public:
  // ==================== 构造函数 ====================

  /// 创建 3D 张量 [channels, rows, cols]，自动分配内存（零初始化）
  Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  /// 创建 1D 张量 [size]
  Tensor(uint32_t size);

  /// 创建 2D 张量 [rows, cols]
  Tensor(uint32_t rows, uint32_t cols);

  /// 默认构造函数：创建一个空张量
  Tensor() = default;

  /// 从已有数据创建（不拷贝，只包装指针）
  static std::shared_ptr<Tensor<T>> Wrap(T* data, uint32_t channels,
                                         uint32_t rows, uint32_t cols);

  /// 从 vector 创建新张量（拷贝数据）
  static std::shared_ptr<Tensor<T>> Create(uint32_t channels, uint32_t rows,
                                           uint32_t cols,
                                           const std::vector<T>& values);

  // ==================== 基本信息 ====================

  /// 获取 channels（3D 时返回 channels，2D 时返回 rows，1D 时返回 1）
  uint32_t channels() const;

  /// 获取 rows
  uint32_t rows() const;

  /// 获取 cols
  uint32_t cols() const;

  /// 总元素个数
  uint64_t size() const;

  /// 单个 channel 的元素数（rows × cols）
  uint64_t plane_size() const;

  /// 是否为空
  bool empty() const;

  /// 获取逻辑形状 [c, r, col]（已做维度简化）
  std::vector<uint32_t> shapes() const;

  // ==================== 数据访问 ====================

  /// 获取原始数据指针
  T* data();
  const T* data() const;

  /// 获取指定 channel 的数据起始指针
  T* channel_data(uint32_t ch);
  const T* channel_data(uint32_t ch) const;

  /// 通过 (channel, row, col) 访问元素
  T& at(uint32_t channel, uint32_t row, uint32_t col);
  const T& at(uint32_t channel, uint32_t row, uint32_t col) const;

  /// 通过一维偏移访问元素
  T& index(uint32_t offset);
  const T& index(uint32_t offset) const;

  /// 将数据导出为 vector（行优先）
  std::vector<T> to_vector() const;

  // ==================== 基本操作 ====================

  /// 用指定值填充整个张量
  void fill(T value);

  /// 用 vector 中的数据填充张量（行优先）
  void fill(const std::vector<T>& values);

  /// 打印张量
  void print(const std::string& name = "tensor") const;

 private:
  // 内部存储：用 3D 维度来组织一维数据
  uint32_t channels_ = 0;
  uint32_t rows_ = 0;
  uint32_t cols_ = 0;

  // 逻辑形状（已做维度简化，用于 shapes() 返回）
  std::vector<uint32_t> logical_shapes_;

  // 实际数据存储
  std::vector<T> data_;
};

// ==================== 类型别名 ====================

using FTensor = Tensor<float>;
using STensor = std::shared_ptr<Tensor<float>>;

// ==================== 实现 ====================

/*
 * 【理解要点】
 *
 * 3D 张量的行优先内存布局：
 *
 * 形状 [2, 3, 4]  →  channels=2, rows=3, cols=4
 *
 *  内存中：
 *  [ ch0_row0_col0, ch0_row0_col1, ch0_row0_col2, ch0_row0_col3,
 *    ch0_row1_col0, ch0_row1_col1, ...               , ch0_row1_col3,
 *    ...
 *    ch0_row2_col3,
 *    ch1_row0_col0, ...
 *    ch1_row2_col3 ]
 *
 *  索引公式：index = (ch × rows × cols) + (r × cols) + c
 */

// ---- 构造函数实现 ----

template <typename T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
    : channels_(channels), rows_(rows), cols_(cols) {
  // 分配内存并初始化为 0
  data_.resize(channels * rows * cols, T{0});

  // 计算逻辑形状（维度简化）
  if (channels == 1 && rows == 1) {
    logical_shapes_ = {cols};              // [1,1,N] → [N]  视为 1D
  } else if (channels == 1) {
    logical_shapes_ = {rows, cols};        // [1,H,W] → [H,W] 视为 2D
  } else {
    logical_shapes_ = {channels, rows, cols};
  }
}

template <typename T>
Tensor<T>::Tensor(uint32_t size)
    : channels_(1), rows_(1), cols_(size) {
  data_.resize(size, T{0});
  logical_shapes_ = {size};
}

template <typename T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols)
    : channels_(1), rows_(rows), cols_(cols) {
  data_.resize(rows * cols, T{0});
  if (rows == 1) {
    logical_shapes_ = {cols};
  } else {
    logical_shapes_ = {rows, cols};
  }
}

// ---- 静态工厂方法 ----

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::Wrap(
    T* data, uint32_t channels, uint32_t rows, uint32_t cols) {
  auto tensor = std::make_shared<Tensor<T>>();
  tensor->channels_ = channels;
  tensor->rows_ = rows;
  tensor->cols_ = cols;
  tensor->data_.assign(data, data + channels * rows * cols);  // 拷贝数据
  tensor->logical_shapes_ = {channels, rows, cols};
  return tensor;
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::Create(
    uint32_t channels, uint32_t rows, uint32_t cols,
    const std::vector<T>& values) {
  assert(values.size() == static_cast<uint64_t>(channels) * rows * cols);
  auto tensor = std::make_shared<Tensor<T>>(channels, rows, cols);
  tensor->fill(values);
  return tensor;
}

// ---- 基本信息访问 ----

template <typename T>
uint32_t Tensor<T>::channels() const {
  return channels_;
}

template <typename T>
uint32_t Tensor<T>::rows() const {
  return rows_;
}

template <typename T>
uint32_t Tensor<T>::cols() const {
  return cols_;
}

template <typename T>
uint64_t Tensor<T>::size() const {
  return static_cast<uint64_t>(channels_) * rows_ * cols_;
}

template <typename T>
uint64_t Tensor<T>::plane_size() const {
  return static_cast<uint64_t>(rows_) * cols_;
}

template <typename T>
bool Tensor<T>::empty() const {
  return data_.empty();
}

template <typename T>
std::vector<uint32_t> Tensor<T>::shapes() const {
  return logical_shapes_;
}

// ---- 数据访问 ----

template <typename T>
T* Tensor<T>::data() {
  return data_.data();
}

template <typename T>
const T* Tensor<T>::data() const {
  return data_.data();
}

template <typename T>
T* Tensor<T>::channel_data(uint32_t ch) {
  assert(ch < channels_);
  return data_.data() + ch * plane_size();
}

template <typename T>
const T* Tensor<T>::channel_data(uint32_t ch) const {
  assert(ch < channels_);
  return data_.data() + ch * plane_size();
}

/**
 * at(ch, row, col) — 通过三维坐标访问元素
 *
 * 行优先索引：index = (ch × rows × cols) + (row × cols) + col
 */
template <typename T>
T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) {
  assert(channel < channels_ && row < rows_ && col < cols_);
  uint32_t index = (channel * rows_ + row) * cols_ + col;
  return data_[index];
}

template <typename T>
const T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  assert(channel < channels_ && row < rows_ && col < cols_);
  uint32_t index = (channel * rows_ + row) * cols_ + col;
  return data_[index];
}

template <typename T>
T& Tensor<T>::index(uint32_t offset) {
  assert(offset < data_.size());
  return data_[offset];
}

template <typename T>
const T& Tensor<T>::index(uint32_t offset) const {
  assert(offset < data_.size());
  return data_[offset];
}

/**
 * to_vector() — 将张量导出为行优先的一维 vector
 *
 * 因为我们内部就是行优先存储，所以直接返回 data_ 即可
 */
template <typename T>
std::vector<T> Tensor<T>::to_vector() const {
  return data_;
}

// ---- 基本操作 ----

template <typename T>
void Tensor<T>::fill(T value) {
  std::fill(data_.begin(), data_.end(), value);
}

/**
 * fill(vector) — 用 vector 中的数据填充张量
 *
 * 按行优先顺序逐元素赋值，与内部的存储顺序一致
 */
template <typename T>
void Tensor<T>::fill(const std::vector<T>& values) {
  assert(values.size() == data_.size());
  std::copy(values.begin(), values.end(), data_.begin());
}

/**
 * print() — 打印张量内容
 *
 * 对每个 channel 打印一个矩阵
 */
template <typename T>
void Tensor<T>::print(const std::string& name) const {
  auto shape = shapes();
  std::cout << name << " shape=[";
  for (uint32_t i = 0; i < shape.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << shape[i];
  }
  std::cout << "]\n";

  // 对小尺寸张量，打印具体内容
  if (size() <= 24) {
    for (uint32_t ch = 0; ch < channels_; ch++) {
      if (channels_ > 1) {
        std::cout << "  Channel " << ch << ":\n";
      }
      for (uint32_t r = 0; r < rows_; r++) {
        std::cout << "    ";
        for (uint32_t c = 0; c < cols_; c++) {
          std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                    << at(ch, r, c) << " ";
        }
        std::cout << "\n";
      }
    }
  } else {
    std::cout << "  (too large to print, " << size() << " elements)\n";
  }
  std::cout << "\n";
}

}  // namespace learn_infer

#endif  // TENSOR_HPP
