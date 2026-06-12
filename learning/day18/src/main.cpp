/**
 * day14/main.cpp
 *
 * Day 14: Expression Layer — Integrating PNNX expressions with real Tensors
 *
 * This demo shows how the expression parser from Day 12/13 integrates
 * with the real Tensor<float> system and RuntimeGraph from Day 11.
 *
 * Concepts covered:
 *   1. ExpressionLayer — a Layer<float> that evaluates expression strings
 *   2. Tokenize → Parse → RPN → Evaluate pipeline on real Tensors
 *   3. Manual graph execution with ExpressionLayer
 *   4. Full pipeline demo from expression string to tensor result
 *
 * Why ExpressionLayer matters:
 *   PNNX .param files contain expressions like:
 *     "add(@0,@1)"
 *     "add(@0,mul(@1,@2))"
 *     "cat(@0,@1,@2,@3,0)"
 *   These express complex output computations without needing a dedicated
 *   layer type. The ExpressionLayer brings this capability to the learning
 *   framework.
 */

#include <iostream>
#include <iomanip>
#include <vector>

#include "include/tensor.hpp"
#include "include/tensor_util.hpp"
#include "include/layer/layer.hpp"
#include "include/layer/layer_factory.hpp"
#include "include/layer/expression.hpp"
#include "src/tokenizer.hpp"
#include "src/parser.hpp"

using namespace learn_infer;

// ============================================================================
// Part 1: ExpressionLayer standalone demo
// ============================================================================

void demo_expression_layer_standalone() {
  std::cout << "==========================================\n";
  std::cout << "Day 14: Expression Layer Integration Demo\n";
  std::cout << "==========================================\n\n";

  std::cout << "--- Part 1: ExpressionLayer Standalone ---\n\n";

  // Create input tensors — Create(channels, rows, cols, values)
  auto input0 = Tensor<float>::Create(1, 2, 3,
    {1, 2, 3, 4, 5, 6});
  auto input1 = Tensor<float>::Create(1, 2, 3,
    {10, 20, 30, 40, 50, 60});

  std::cout << "Input @0:";
  input0->print();
  std::cout << "Input @1:";
  input1->print();

  // Create an ExpressionLayer for "add(@0,@1)"
  ExpressionLayer layer("add(@0,@1)");
  std::cout << "ExpressionLayer: \"add(@0,@1)\"\n";

  // Forward pass
  std::vector<STensor> inputs = {input0, input1};
  std::vector<STensor> outputs;
  auto status = layer.Forward(inputs, outputs);

  if (status == StatusCode::kSuccess && !outputs.empty()) {
    std::cout << "\nOutput:";
    outputs[0]->print();
    std::cout << "Expected: [11, 22, 33, 44, 55, 66] -- ";
    // Verify
    auto expected = Tensor<float>::Create(1, 2, 3,
      {11, 22, 33, 44, 55, 66});
    if (util::TensorIsSame(outputs[0], expected, 1e-5f)) {
      std::cout << "PASS\n";
    } else {
      std::cout << "FAIL\n";
    }
  }

  // Now try a two-step expression
  std::cout << "\n--- Part 1b: Two-Step ExpressionLayer ---\n\n";

  auto input2 = Tensor<float>::Create(1, 2, 3,
    {2, 2, 2, 2, 2, 2});

  std::cout << "Input @0:";
  input0->print();
  std::cout << "Input @1:";
  input1->print();
  std::cout << "Input @2:";
  input2->print();

  // Expression: add(@0, mul(@1, @2))
  // = [1,2,3,4,5,6] + ([10,20,30,40,50,60] * [2,2,2,2,2,2])
  // = [1,2,3,4,5,6] + [20,40,60,80,100,120]
  // = [21, 42, 63, 84, 105, 126]
  ExpressionLayer layer2("add(@0,mul(@1,@2))");
  std::cout << "ExpressionLayer: \"add(@0,mul(@1,@2))\"\n";

  std::vector<STensor> inputs2 = {input0, input1, input2};
  std::vector<STensor> outputs2;
  layer2.Forward(inputs2, outputs2);

  if (!outputs2.empty()) {
    std::cout << "\nOutput:";
    outputs2[0]->print();
    std::cout << "Expected: [21, 42, 63, 84, 105, 126] -- ";
    auto expected2 = Tensor<float>::Create(1, 2, 3,
      {21, 42, 63, 84, 105, 126});
    if (util::TensorIsSame(outputs2[0], expected2, 1e-5f)) {
      std::cout << "PASS\n";
    } else {
      std::cout << "FAIL\n";
    }
  }
}

// ============================================================================
// Part 2: Mini manual graph with ExpressionLayer
// ============================================================================

void demo_manual_graph() {
  std::cout << "\n--- Part 2: Manual Graph with ExpressionLayer ---\n\n";

  // Simulate a ResNet-style residual connection:
  //   output = add(identity, mul(identity, scale))
  //
  // This mimics patterns seen in PNNX .param files where
  // expressions combine multiple intermediate results.
  //
  // Graph layout:
  //
  //   [input] ──@0──►                 ◄──@0─── [input]
  //                       add(@0,mul(@0,@1))    [input]
  //   [scale] ──@1──► mul             ◄──@1─── [scale]
  //                        │
  //                        ▼
  //                     [result]
  //
  // Computation:
  //   identity = [1, 2, 3, 4]
  //   scale    = [0.5, 0.5, 0.5, 0.5]
  //   mul(@0,@1) = [0.5, 1.0, 1.5, 2.0]
  //   add(@0, mul) = [1.5, 3.0, 4.5, 6.0]

  auto identity = Tensor<float>::Create(1, 1, 4,
    {1, 2, 3, 4});
  auto scale = Tensor<float>::Create(1, 1, 4,
    {0.5f, 0.5f, 0.5f, 0.5f});

  std::cout << "Residual connection: output = add(identity, mul(identity, scale))\n\n";
  std::cout << "  identity:";
  identity->print();
  std::cout << "  scale:";
  scale->print();

  // Step 1: mul(@0, @1) — scale the identity
  ExpressionLayer mul_layer("mul(@0,@1)");
  std::vector<STensor> mul_inputs = {identity, scale};
  std::vector<STensor> mul_outputs;
  mul_layer.Forward(mul_inputs, mul_outputs);

  std::cout << "  mul(@0,@1) result:";
  mul_outputs[0]->print();

  // Step 2: add(@0, @1) — add scaled result back to identity
  ExpressionLayer add_layer("add(@0,@1)");
  std::vector<STensor> add_inputs = {identity, mul_outputs[0]};
  std::vector<STensor> add_outputs;
  add_layer.Forward(add_inputs, add_outputs);

  std::cout << "  add(@0,@1) final output:";
  add_outputs[0]->print();

  std::cout << "  Expected: [1.5, 3.0, 4.5, 6.0] -- ";
  auto expected = Tensor<float>::Create(1, 1, 4,
    {1.5f, 3.0f, 4.5f, 6.0f});
  if (util::TensorIsSame(add_outputs[0], expected, 1e-5f)) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }
}

// ============================================================================
// Part 3: Parser pipeline demo with real Tensors
// ============================================================================

void demo_parser_pipeline() {
  std::cout << "\n--- Part 3: Full Parser Pipeline with Tensors ---\n\n";

  // Show the full pipeline: string -> tokens -> AST -> RPN -> Tensor result
  std::string expr_str = "add(@0,@1)";

  std::cout << "Expression: \"" << expr_str << "\"\n";

  // Step 1: Tokenize (using the Tokenizer class)
  tutorial::Tokenizer tokenizer(expr_str);
  tokenizer.tokenize();
  const auto& tokens = tokenizer.tokens();
  std::cout << "  Tokens: ";
  for (const auto& t : tokens) {
    std::cout << t.text << " ";
  }
  std::cout << "\n";

  // Step 2: Parse to AST
  tutorial::Parser parser(tokens);
  auto ast = parser.parse();
  std::cout << "  AST:\n";
  ast->print(4);

  // Step 3: Convert to RPN
  auto rpn = parser.to_rpn(ast);
  std::cout << "  RPN sequence (" << rpn.size() << " nodes):\n";
  for (const auto& node : rpn) {
    if (node->is_input()) {
      std::cout << "    - @" << node->num_index << "\n";
    } else {
      std::cout << "    - " << node->op_name() << "\n";
    }
  }

  // Step 4: Evaluate with real tensors
  auto t0 = Tensor<float>::Create(1, 1, 4, {1, 2, 3, 4});
  auto t1 = Tensor<float>::Create(1, 1, 4, {10, 20, 30, 40});

  std::vector<STensor> inputs = {t0, t1};
  auto result = util::EvaluateRPN<float>(rpn, inputs);

  std::cout << "\n  Evaluated with:\n";
  std::cout << "    @0 =";
  t0->print();
  std::cout << "    @1 =";
  t1->print();
  std::cout << "  Result =";
  result[0]->print();

  std::cout << "  Expected: [11, 22, 33, 44] -- ";
  auto expected = Tensor<float>::Create(1, 1, 4,
    {11, 22, 33, 44});
  if (util::TensorIsSame(result[0], expected, 1e-5f)) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }
}

// ============================================================================
// Part 4: Deeply nested expression
// ============================================================================

void demo_nested_expression() {
  std::cout << "\n--- Part 4: Deeply Nested Expression ---\n\n";

  // Test a deeply nested expression:
  //   add(@0, mul(@1, add(@2, @3)))
  //
  // With inputs:
  //   @0 = [1, 1, 1, 1]
  //   @1 = [2, 2, 2, 2]
  //   @2 = [3, 3, 3, 3]
  //   @3 = [4, 4, 4, 4]
  //
  // Evaluation:
  //   add(@2,@3) = [7, 7, 7, 7]
  //   mul(@1,[7,7,7,7]) = [14, 14, 14, 14]
  //   add(@0,[14,14,14,14]) = [15, 15, 15, 15]

  std::string expr = "add(@0,mul(@1,add(@2,@3)))";
  std::cout << "Expression: \"" << expr << "\"\n\n";

  auto t0 = Tensor<float>::Create(1, 1, 4, {1, 1, 1, 1});
  auto t1 = Tensor<float>::Create(1, 1, 4, {2, 2, 2, 2});
  auto t2 = Tensor<float>::Create(1, 1, 4, {3, 3, 3, 3});
  auto t3 = Tensor<float>::Create(1, 1, 4, {4, 4, 4, 4});

  std::cout << "  @0 ="; t0->print();
  std::cout << "  @1 ="; t1->print();
  std::cout << "  @2 ="; t2->print();
  std::cout << "  @3 ="; t3->print();

  ExpressionLayer layer(expr);
  std::vector<STensor> inputs = {t0, t1, t2, t3};
  std::vector<STensor> outputs;
  layer.Forward(inputs, outputs);

  std::cout << "  Result:";
  outputs[0]->print();

  std::cout << "  Expected: [15, 15, 15, 15] -- ";
  auto expected = Tensor<float>::Create(1, 1, 4,
    {15, 15, 15, 15});
  if (util::TensorIsSame(outputs[0], expected, 1e-5f)) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }
}

// ============================================================================

int main() {
  demo_expression_layer_standalone();
  demo_manual_graph();
  demo_parser_pipeline();
  demo_nested_expression();

  std::cout << "\n==========================================\n";
  std::cout << "Day 14 Complete!\n";
  std::cout << "==========================================\n";
  return 0;
}
