/**
 * src/layer/expression.cpp
 *
 * ExpressionLayer implementation:
 *   Constructor: tokenize → parse → to_rpn (cache the result)
 *   Forward: evaluate RPN with input tensors
 */

#include "include/layer/expression.hpp"
#include <stdexcept>

namespace learn_infer {

ExpressionLayer::ExpressionLayer(std::string statement)
    : statement_(std::move(statement)) {
  // Parse once at construction for efficiency
  try {
    tutorial::Tokenizer tokenizer(statement_);
    tokenizer.tokenize();
    tutorial::Parser parser(tokenizer.tokens());
    auto ast = parser.parse();
    rpn_ = parser.to_rpn(ast);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse expression '" + statement_ +
                             "': " + e.what());
  }
}

StatusCode ExpressionLayer::Forward(
    const std::vector<STensor>& inputs,
    std::vector<STensor>& outputs) {
  // Evaluate the pre-parsed RPN using tensor utilities
  outputs = util::EvaluateRPN<float>(rpn_, inputs);
  return StatusCode::kSuccess;
}

}  // namespace learn_infer
