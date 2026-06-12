/**
 * include/layer/expression.hpp
 *
 * Day 14: Expression Layer
 *
 * Integrates the expression parser (Day 12/13) with the real
 * Tensor and RuntimeGraph system (Day 11).
 *
 * The ExpressionLayer evaluates PNNX expression strings like
 * "add(@0,@1)" using real Tensor<float> objects.
 *
 * Pipeline inside Forward():
 *   1. Tokenize expression string  (cached at construction)
 *   2. Parse tokens into AST        (cached at construction)
 *   3. Convert AST to RPN            (cached at construction)
 *   4. Evaluate RPN with input tensors on a stack
 *   5. Return result tensor
 */

#pragma once

#include "layer.hpp"
#include "../../src/tokenizer.hpp"
#include "../../src/parser.hpp"
#include "../tensor_util.hpp"
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>

namespace learn_infer {

/**
 * Expression layer that evaluates PNNX expression strings.
 *
 * Supports:
 *   - add(@0,@1)       element-wise addition
 *   - mul(@0,@1)       element-wise multiplication
 *   - Nested: add(@0,mul(@1,@2))
 *   - @0 through @N    references to input tensors
 */
class ExpressionLayer : public Layer<float> {
 public:
  explicit ExpressionLayer(std::string statement);

  StatusCode Forward(const std::vector<STensor>& inputs,
                     std::vector<STensor>& outputs) override;

  std::string Type() const override { return "nn.Expression"; }

 private:
  std::string statement_;
  // Pre-computed RPN for efficiency (parsed once at construction)
  std::vector<std::shared_ptr<tutorial::ASTNode>> rpn_;
};

}  // namespace learn_infer
