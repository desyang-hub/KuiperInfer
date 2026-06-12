/**
 * day3/include/tensor.hpp
 *
 * Tensor -- header-only N-dimensional tensor (C/H/W layout).
 *
 * Copied from day2/include/tensor.hpp and kept identical so that
 * day3's tensor_util.hpp can build on top of it.
 *
 * Pure header-only, no external dependencies.
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <cassert>
#include <iomanip>
#include <cstring>
#include <memory>
#include <cmath>
#include <functional>
#include <random>

namespace learn_infer {

template <typename T = float>
class Tensor {
 public:
  // ==================== Construction ====================

  Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
  Tensor(uint32_t size);
  Tensor(uint32_t rows, uint32_t cols);
  Tensor() = default;

  static std::shared_ptr<Tensor<T>> Wrap(T* data, uint32_t channels,
                                         uint32_t rows, uint32_t cols);
  static std::shared_ptr<Tensor<T>> Create(uint32_t channels, uint32_t rows,
                                           uint32_t cols,
                                           const std::vector<T>& values);

  // ==================== Info ====================

  uint32_t channels() const;
  uint32_t rows() const;
  uint32_t cols() const;
  uint64_t size() const;
  uint64_t plane_size() const;
  bool empty() const;
  std::vector<uint32_t> shapes() const;

  // ==================== Data access ====================

  T* data();
  const T* data() const;
  T* channel_data(uint32_t ch);
  const T* channel_data(uint32_t ch) const;
  T& at(uint32_t channel, uint32_t row, uint32_t col);
  const T& at(uint32_t channel, uint32_t row, uint32_t col) const;
  T& index(uint32_t offset);
  const T& index(uint32_t offset) const;
  std::vector<T> to_vector() const;

  // ==================== Basic ops ====================

  void fill(T value);
  void fill(const std::vector<T>& values);
  void print(const std::string& name = "tensor") const;

  // ==================== Day 2 ops ====================

  void reshape(uint32_t channels, uint32_t rows, uint32_t cols,
               bool row_major = true);
  void flatten(bool row_major = true);
  void transform(std::function<T(T)> func);
  void padding(const std::vector<uint32_t>& pads, T value);
  std::shared_ptr<Tensor<T>> clone() const;
  void ones();
  void randn(T mean = 0.0f, T var = 1.0f);

 private:
  uint32_t channels_ = 0;
  uint32_t rows_ = 0;
  uint32_t cols_ = 0;
  std::vector<uint32_t> logical_shapes_;
  std::vector<T> data_;
};

using FTensor = Tensor<float>;
using STensor = std::shared_ptr<Tensor<float>>;

// ==================== Implementation ====================

// ---- Construction ----

template <typename T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
    : channels_(channels), rows_(rows), cols_(cols) {
  data_.resize(channels * rows * cols, T{0});
  if (channels == 1 && rows == 1) {
    logical_shapes_ = {cols};
  } else if (channels == 1) {
    logical_shapes_ = {rows, cols};
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

// ---- Factories ----

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::Wrap(
    T* data, uint32_t channels, uint32_t rows, uint32_t cols) {
  auto tensor = std::make_shared<Tensor<T>>();
  tensor->channels_ = channels;
  tensor->rows_ = rows;
  tensor->cols_ = cols;
  tensor->data_.assign(data, data + channels * rows * cols);
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

// ---- Info ----

template <typename T>
uint32_t Tensor<T>::channels() const { return channels_; }

template <typename T>
uint32_t Tensor<T>::rows() const { return rows_; }

template <typename T>
uint32_t Tensor<T>::cols() const { return cols_; }

template <typename T>
uint64_t Tensor<T>::size() const {
  return static_cast<uint64_t>(channels_) * rows_ * cols_;
}

template <typename T>
uint64_t Tensor<T>::plane_size() const {
  return static_cast<uint64_t>(rows_) * cols_;
}

template <typename T>
bool Tensor<T>::empty() const { return data_.empty(); }

template <typename T>
std::vector<uint32_t> Tensor<T>::shapes() const { return logical_shapes_; }

// ---- Data access ----

template <typename T>
T* Tensor<T>::data() { return data_.data(); }

template <typename T>
const T* Tensor<T>::data() const { return data_.data(); }

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

template <typename T>
std::vector<T> Tensor<T>::to_vector() const {
  return data_;
}

// ---- Basic ops ----

template <typename T>
void Tensor<T>::fill(T value) {
  std::fill(data_.begin(), data_.end(), value);
}

template <typename T>
void Tensor<T>::fill(const std::vector<T>& values) {
  assert(values.size() == data_.size());
  std::copy(values.begin(), values.end(), data_.begin());
}

template <typename T>
void Tensor<T>::print(const std::string& name) const {
  auto shape = shapes();
  std::cout << name << " shape=[";
  for (uint32_t i = 0; i < shape.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << shape[i];
  }
  std::cout << "]\n";

  if (size() <= 48) {
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

// =====================================================================
// Day 2 ops implementation
// =====================================================================

template <typename T>
void Tensor<T>::reshape(uint32_t c, uint32_t r, uint32_t col, bool row_major) {
  uint64_t new_size = static_cast<uint64_t>(c) * r * col;
  assert(new_size == size() && "Reshape: element count must match");

  if (row_major) {
    channels_ = c;
    rows_ = r;
    cols_ = col;
  } else {
    channels_ = c;
    rows_ = r;
    cols_ = col;
  }

  if (channels_ == 1 && rows_ == 1) {
    logical_shapes_ = {cols_};
  } else if (channels_ == 1) {
    logical_shapes_ = {rows_, cols_};
  } else {
    logical_shapes_ = {channels_, rows_, cols_};
  }
}

template <typename T>
void Tensor<T>::flatten(bool row_major) {
  uint64_t s = size();
  reshape(1, 1, static_cast<uint32_t>(s), row_major);
}

template <typename T>
void Tensor<T>::transform(std::function<T(T)> func) {
  for (uint64_t i = 0; i < data_.size(); i++) {
    data_[i] = func(data_[i]);
  }
}

template <typename T>
void Tensor<T>::padding(const std::vector<uint32_t>& pads, T value) {
  assert(pads.size() == 4 && "pads must be [top, bottom, left, right]");
  uint32_t pad_top = pads[0];
  uint32_t pad_bottom = pads[1];
  uint32_t pad_left = pads[2];
  uint32_t pad_right = pads[3];

  uint32_t new_rows = rows_ + pad_top + pad_bottom;
  uint32_t new_cols = cols_ + pad_left + pad_right;
  uint64_t new_plane = static_cast<uint64_t>(new_rows) * new_cols;
  uint64_t old_plane = plane_size();

  std::vector<T> new_data;
  new_data.resize(channels_ * new_plane, value);

  for (uint32_t ch = 0; ch < channels_; ch++) {
    const T* src = data_.data() + ch * old_plane;
    T* dst = new_data.data() + ch * new_plane;

    for (uint32_t r = 0; r < rows_; r++) {
      for (uint32_t c = 0; c < cols_; c++) {
        uint64_t src_idx = r * cols_ + c;
        uint64_t dst_idx = (r + pad_top) * new_cols + (c + pad_left);
        dst[dst_idx] = src[src_idx];
      }
    }
  }

  data_ = std::move(new_data);
  rows_ = new_rows;
  cols_ = new_cols;

  if (channels_ == 1 && rows_ == 1) {
    logical_shapes_ = {cols_};
  } else if (channels_ == 1) {
    logical_shapes_ = {rows_, cols_};
  } else {
    logical_shapes_ = {channels_, rows_, cols_};
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::clone() const {
  auto t = std::make_shared<Tensor<T>>();
  t->channels_ = channels_;
  t->rows_ = rows_;
  t->cols_ = cols_;
  t->data_ = data_;
  t->logical_shapes_ = logical_shapes_;
  return t;
}

template <typename T>
void Tensor<T>::ones() {
  fill(static_cast<T>(1));
}

template <typename T>
void Tensor<T>::randn(T mean, T var) {
  T stddev = std::sqrt(var);
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<T> dist(mean, stddev);
  for (uint64_t i = 0; i < data_.size(); i++) {
    data_[i] = dist(gen);
  }
}

}  // namespace learn_infer

#endif  // TENSOR_HPP
