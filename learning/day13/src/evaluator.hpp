/**
 * src/evaluator.hpp
 *
 * Day 13: Expression Evaluator
 *
 * Evaluates an RPN (Reverse Polish Notation) sequence of AST nodes
 * using actual tensor data. This is the final stage of the pipeline:
 *
 *   Expression String -> Tokenizer -> Tokens -> Parser -> AST -> RPN
 *     -> Evaluator -> Output Tensor values
 *
 * The evaluator uses a simple operand stack:
 *   - For input nodes: push the corresponding input tensor onto the stack
 *   - For ADD node: pop two tensors, push element-wise sum
 *   - For MUL node: pop two tensors, push element-wise product
 *
 * After processing all RPN nodes, exactly one tensor remains on
 * the stack — that's the result.
 */

#pragma once

#include "src/parser.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

namespace learn_infer {
namespace tutorial {

/**
 * A simple 1D tensor for demonstration.
 *
 * In the full KuiperInfer framework, this uses arma::Cube<float>.
 * Here we use std::vector<float> to keep things self-contained.
 */
struct SimpleTensor {
  std::vector<float> data;
  int size;

  SimpleTensor() : size(0) {}
  explicit SimpleTensor(const std::vector<float>& d)
      : data(d), size(static_cast<int>(d.size())) {}

  float operator[](int i) const { return data[i]; }
  float& operator[](int i) { return data[i]; }

  /**
   * Print tensor contents.
   */
  void print(const std::string& label = "") const;
};

/**
 * Expression evaluator.
 *
 * Takes an RPN sequence and a set of input tensors,
 * evaluates the expression, and returns the output tensor.
 */
class Evaluator {
 public:
  /**
   * Evaluate an RPN sequence with the given inputs.
   *
   * @param rpn  Reverse Polish Notation sequence from Parser::to_rpn()
   * @param inputs  Input tensors, indexed by @N reference
   * @return  Result tensor
   */
  static SimpleTensor evaluate(
      const std::vector<std::shared_ptr<ASTNode>>& rpn,
      const std::vector<SimpleTensor>& inputs);

 public:
  /**
   * Element-wise addition of two tensors.
   * Broadcasts if sizes differ (smaller tensor repeats).
   */
  static SimpleTensor element_wise_add(const SimpleTensor& a, const SimpleTensor& b);

  /**
   * Element-wise multiplication of two tensors.
   */
  static SimpleTensor element_wise_mul(const SimpleTensor& a, const SimpleTensor& b);
};

}  // namespace tutorial
}  // namespace learn_infer
