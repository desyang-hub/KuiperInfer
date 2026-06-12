/**
 * day3/include/tensor_util.hpp
 *
 * Tensor utility functions -- free functions that operate on Tensor
 * objects and return new shared_ptr<Tensor> results.
 *
 * Functions:
 *   TensorElementAdd       -- element-wise addition  a + b
 *   TensorElementMultiply  -- element-wise multiply  a * b
 *   TensorBroadcast        -- broadcast [C,1,1] to [C,H,W] by repeating
 *   TensorIsSame           -- compare two tensors within a threshold
 *   TensorClone            -- convenience wrapper for t->clone()
 *   TensorCreate           -- factory from shapes vector + values
 *
 * Header-only, no external dependencies beyond tensor.hpp.
 */

#ifndef TENSOR_UTIL_HPP
#define TENSOR_UTIL_HPP

#include "tensor.hpp"
#include "src/parser.hpp"
#include <cmath>

namespace learn_infer {
namespace util {

/**
 * TensorElementAdd(a, b)
 *
 * Element-wise addition.  Both tensors must have the same shape.
 * Returns a new shared_ptr<Tensor<T>>; inputs are not modified.
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(
    const std::shared_ptr<Tensor<T>>& a,
    const std::shared_ptr<Tensor<T>>& b) {
  assert(a->channels() == b->channels() &&
         a->rows() == b->rows() &&
         a->cols() == b->cols() &&
         "TensorElementAdd: shapes must match");

  auto out = std::make_shared<Tensor<T>>(a->channels(), a->rows(), a->cols());
  const T* pa = a->data();
  const T* pb = b->data();
  T* po = out->data();
  uint64_t n = a->size();
  for (uint64_t i = 0; i < n; i++) {
    po[i] = pa[i] + pb[i];
  }
  return out;
}

/**
 * TensorElementMultiply(a, b)
 *
 * Element-wise multiplication.  Both tensors must have the same shape.
 * Returns a new shared_ptr<Tensor<T>>; inputs are not modified.
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(
    const std::shared_ptr<Tensor<T>>& a,
    const std::shared_ptr<Tensor<T>>& b) {
  assert(a->channels() == b->channels() &&
         a->rows() == b->rows() &&
         a->cols() == b->cols() &&
         "TensorElementMultiply: shapes must match");

  auto out = std::make_shared<Tensor<T>>(a->channels(), a->rows(), a->cols());
  const T* pa = a->data();
  const T* pb = b->data();
  T* po = out->data();
  uint64_t n = a->size();
  for (uint64_t i = 0; i < n; i++) {
    po[i] = pa[i] * pb[i];
  }
  return out;
}

/**
 * TensorBroadcast(a, b)
 *
 * Broadcast tensor `a` to the shape of tensor `b`.
 *
 * Supported pattern:  a is [C, 1, 1] and b is [C, H, W] with the same C.
 * Every spatial position in each channel gets the single scalar value
 * stored in a's [C, 0, 0].
 *
 * Returns a new tensor of shape [C, H, W].
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorBroadcast(
    const std::shared_ptr<Tensor<T>>& a,
    const std::shared_ptr<Tensor<T>>& b) {
  uint32_t C = b->channels();
  uint32_t H = b->rows();
  uint32_t W = b->cols();

  assert(a->channels() == C &&
         a->rows() == 1 &&
         a->cols() == 1 &&
         "TensorBroadcast: a must be [C, 1, 1] matching b's channel count");

  auto out = std::make_shared<Tensor<T>>(C, H, W);
  T* po = out->data();
  for (uint32_t c = 0; c < C; c++) {
    T val = a->at(c, 0, 0);
    uint64_t plane_off = static_cast<uint64_t>(c) * H * W;
    for (uint64_t i = 0; i < static_cast<uint64_t>(H) * W; i++) {
      po[plane_off + i] = val;
    }
  }
  return out;
}

/**
 * TensorIsSame(a, b, threshold)
 *
 * Returns true if every corresponding element of a and b differs by
 * less than `threshold`.  Both tensors must have the same shape.
 */
template <typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a,
                  const std::shared_ptr<Tensor<T>>& b,
                  T threshold = 1e-6f) {
  assert(a->channels() == b->channels() &&
         a->rows() == b->rows() &&
         a->cols() == b->cols() &&
         "TensorIsSame: shapes must match");

  const T* pa = a->data();
  const T* pb = b->data();
  uint64_t n = a->size();
  for (uint64_t i = 0; i < n; i++) {
    if (std::abs(pa[i] - pb[i]) > threshold) {
      return false;
    }
  }
  return true;
}

/**
 * TensorClone(t)
 *
 * Convenience free-function wrapper for t->clone().
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorClone(const std::shared_ptr<Tensor<T>>& t) {
  return t->clone();
}

/**
 * TensorCreate(shapes, values)
 *
 * Factory: create a tensor from a shapes vector and flat values.
 *
 * Supported shape lengths:
 *   size 1  -> [1, 1, shapes[0]]
 *   size 2  -> [1, shapes[0], shapes[1]]
 *   size 3  -> [shapes[0], shapes[1], shapes[2]]
 *
 * values must contain exactly shapes[0]*shapes[1]*... elements.
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(
    const std::vector<uint32_t>& shapes, const std::vector<T>& values) {
  assert(shapes.size() >= 1 && shapes.size() <= 3);

  uint32_t c, r, col;
  if (shapes.size() == 1) {
    c = 1; r = 1; col = shapes[0];
  } else if (shapes.size() == 2) {
    c = 1; r = shapes[0]; col = shapes[1];
  } else {
    c = shapes[0]; r = shapes[1]; col = shapes[2];
  }

  uint64_t expected = static_cast<uint64_t>(c) * r * col;
  assert(values.size() == expected &&
         "TensorCreate: values count must match shapes product");

  return Tensor<T>::Create(c, r, col, values);
}

/**
 * EvaluateRPN(rpn, inputs)
 *
 * Evaluate an RPN (Reverse Polish Notation) sequence of AST nodes
 * using real Tensor<float> objects.
 *
 * Uses an operand stack:
 *   - Input node (num_index >= 0): push the corresponding input tensor
 *   - Op node (num_index < 0): pop two tensors, apply op, push result
 *     (-1 = ADD, -2 = MUL)
 *
 * @param rpn      RPN sequence from Parser::to_rpn()
 * @param inputs   Input tensors, indexed by @N reference
 * @return         Resulting output tensors as a vector (usually one tensor)
 */
template <typename T>
std::vector<std::shared_ptr<Tensor<T>>> EvaluateRPN(
    const std::vector<std::shared_ptr<tutorial::ASTNode>>& rpn,
    const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
  std::vector<std::shared_ptr<Tensor<T>>> stack;

  for (const auto& node : rpn) {
    if (node->is_input()) {
      int idx = node->num_index;
      assert(idx >= 0 && idx < static_cast<int>(inputs.size()));
      stack.push_back(inputs[idx]);
    } else {
      assert(stack.size() >= 2);
      auto rhs = stack.back(); stack.pop_back();
      auto lhs = stack.back(); stack.pop_back();

      std::shared_ptr<Tensor<T>> result;
      if (node->num_index == -1) {  // ADD
        result = TensorElementAdd(lhs, rhs);
      } else {  // MUL (num_index == -2)
        result = TensorElementMultiply(lhs, rhs);
      }
      stack.push_back(result);
    }
  }

  assert(stack.size() == 1);
  return stack;
}

}  // namespace util

}  // namespace learn_infer

#endif  // TENSOR_UTIL_HPP
