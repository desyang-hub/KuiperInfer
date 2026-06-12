/**
 * src/evaluator.cpp
 *
 * Implementation of the expression evaluator.
 */

#include "src/evaluator.hpp"
#include <iostream>
#include <sstream>
#include <cassert>

namespace learn_infer {
namespace tutorial {

/* ------------------------------------------------------------------ */
/* SimpleTensor helpers                                                */
/* ------------------------------------------------------------------ */
void SimpleTensor::print(const std::string& label) const {
  if (!label.empty()) {
    std::cout << label << " = ";
  }
  std::cout << "[";
  for (int i = 0; i < size; i++) {
    if (i > 0) std::cout << ", ";
    std::cout << data[i];
  }
  std::cout << "]\n";
}

/* ------------------------------------------------------------------ */
/* Evaluator implementation                                            */
/* ------------------------------------------------------------------ */
SimpleTensor Evaluator::evaluate(
    const std::vector<std::shared_ptr<ASTNode>>& rpn,
    const std::vector<SimpleTensor>& inputs) {
  std::vector<SimpleTensor> stack;

  for (const auto& node : rpn) {
    if (node->is_input()) {
      // Push the referenced input tensor onto the stack
      int idx = node->num_index;
      if (idx < 0 || idx >= static_cast<int>(inputs.size())) {
        std::ostringstream msg;
        msg << "Evaluator: input index @" << idx << " out of range "
            << "(have " << inputs.size() << " inputs)";
        throw std::runtime_error(msg.str());
      }
      stack.push_back(inputs[idx]);
    } else if (node->num_index == -1) {
      // ADD operator
      if (stack.size() < 2) {
        throw std::runtime_error("Evaluator: ADD needs 2 operands");
      }
      auto b = stack.back(); stack.pop_back();
      auto a = stack.back(); stack.pop_back();
      stack.push_back(element_wise_add(a, b));
    } else if (node->num_index == -2) {
      // MUL operator
      if (stack.size() < 2) {
        throw std::runtime_error("Evaluator: MUL needs 2 operands");
      }
      auto b = stack.back(); stack.pop_back();
      auto a = stack.back(); stack.pop_back();
      stack.push_back(element_wise_mul(a, b));
    } else {
      std::ostringstream msg;
      msg << "Evaluator: unknown operator code " << node->num_index;
      throw std::runtime_error(msg.str());
    }
  }

  if (stack.size() != 1) {
    std::ostringstream msg;
    msg << "Evaluator: expected 1 result on stack, got " << stack.size();
    throw std::runtime_error(msg.str());
  }

  return stack[0];
}

SimpleTensor Evaluator::element_wise_add(const SimpleTensor& a, const SimpleTensor& b) {
  int size = std::max(a.size, b.size);
  std::vector<float> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = a[i % a.size] + b[i % b.size];
  }
  return SimpleTensor(result);
}

SimpleTensor Evaluator::element_wise_mul(const SimpleTensor& a, const SimpleTensor& b) {
  int size = std::max(a.size, b.size);
  std::vector<float> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = a[i % a.size] * b[i % b.size];
  }
  return SimpleTensor(result);
}

}  // namespace tutorial
}  // namespace learn_infer
