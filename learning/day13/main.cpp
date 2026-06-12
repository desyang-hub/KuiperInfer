/**
 * day13/main.cpp
 *
 * Day 13 Demo: Expression Parser + Evaluator
 *
 * Builds on Day 12 (Tokenizer) and adds:
 *   - Recursive descent parser: Tokens -> AST
 *   - AST -> RPN (Reverse Polish Notation) conversion
 *   - RPN evaluation with actual tensor values
 *   - Full pipeline: "add(@0,mul(@1,@2))" + data -> result
 *
 * Pipeline:
 *   Expression String -> Tokenizer -> Tokens -> Parser -> AST
 *     -> RPN -> Evaluator -> Output Tensor
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day13
 */

#include "src/tokenizer.hpp"
#include "src/parser.hpp"
#include "src/evaluator.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <vector>

using namespace learn_infer;
using namespace learn_infer::tutorial;

/* ------------------------------------------------------------------ */
/* Helper: pretty-print a token stream                                 */
/* ------------------------------------------------------------------ */
static void print_tokens(const std::vector<Token>& tokens) {
  std::cout << "  " << std::string(55, '-') << "\n";
  std::cout << "  #" << "  " << std::setw(12) << "Type"
            << "  " << std::setw(8) << "Text"
            << "  " << std::setw(10) << "Position" << "\n";
  std::cout << "  " << std::string(55, '-') << "\n";
  for (size_t i = 0; i < tokens.size(); i++) {
    std::cout << "  " << std::setw(2) << i << "  "
              << std::setw(12) << token_type_to_string(tokens[i].type) << "  "
              << std::setw(8) << "\"" << tokens[i].text << "\"  "
              << "[" << tokens[i].start_pos << "-" << tokens[i].end_pos << "]";
    if (tokens[i].type == TokenType::InputNumber) {
      std::cout << "  (index=" << tokens[i].input_index << ")";
    }
    std::cout << "\n";
  }
  std::cout << "  " << std::string(55, '-') << "\n";
  std::cout << "  Total: " << tokens.size() << " tokens\n\n";
}

/* ------------------------------------------------------------------ */
/* Demo 1: Basic parsing of "add(@0,@1)"                               */
/* ------------------------------------------------------------------ */
void demo_basic_parse() {
  std::cout << "========== Demo 1: Basic Parse ==========\n\n";
  std::cout << "  Expression: \"add(@0,@1)\"\n";
  std::cout << "  This means: output = input_0 + input_1\n\n";

  // Stage 1: Tokenize
  std::string expr = "add(@0,@1)";
  std::cout << "  Stage 1: Tokenize\n";
  Tokenizer tokenizer(expr);
  tokenizer.tokenize();
  print_tokens(tokenizer.tokens());

  // Stage 2: Parse to AST
  std::cout << "  Stage 2: Parse to AST\n";
  Parser parser(tokenizer.tokens());
  auto ast = parser.parse();

  std::cout << "  AST:\n";
  ast->print(4);
  std::cout << "\n";

  // Verify AST structure
  bool pass = true;
  pass &= ast->is_operator();
  pass &= (ast->num_index == -1); // ADD
  pass &= ast->left->is_input() && (ast->left->num_index == 0);
  pass &= ast->right->is_input() && (ast->right->num_index == 1);

  std::cout << "  AST structure: " << (pass ? "PASS" : "FAIL") << "\n\n";

  // Stage 3: AST -> RPN
  std::cout << "  Stage 3: Convert to RPN\n";
  auto rpn = Parser::to_rpn(ast);
  std::cout << "  RPN: " << Parser::rpn_to_string(rpn) << "\n";
  std::cout << "  (post-order traversal: left, right, operator)\n\n";

  // Stage 4: Evaluate
  std::cout << "  Stage 4: Evaluate\n";
  std::vector<SimpleTensor> inputs = {
    SimpleTensor({1.0f, 2.0f, 3.0f}),
    SimpleTensor({4.0f, 5.0f, 6.0f}),
  };
  std::cout << "  @0 = "; inputs[0].print();
  std::cout << "  @1 = "; inputs[1].print();

  auto result = Evaluator::evaluate(rpn, inputs);
  std::cout << "  result = @0 + @1 = ";
  result.print();
  std::cout << "\n";

  // Verify: [5, 7, 9]
  bool eval_pass = (result.size == 3);
  eval_pass &= (std::abs(result[0] - 5.0f) < 1e-5);
  eval_pass &= (std::abs(result[1] - 7.0f) < 1e-5);
  eval_pass &= (std::abs(result[2] - 9.0f) < 1e-5);
  std::cout << "  Evaluation: " << (eval_pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* Demo 2: Nested expression "add(@0,mul(@1,@2))"                      */
/* ------------------------------------------------------------------ */
void demo_nested_parse() {
  std::cout << "========== Demo 2: Nested Expression ==========\n\n";
  std::cout << "  Expression: \"add(@0,mul(@1,@2))\"\n";
  std::cout << "  This means: output = @0 + (@1 * @2)\n\n";

  std::string expr = "add(@0,mul(@1,@2))";

  std::cout << "  Stage 1: Tokenize\n";
  Tokenizer tokenizer(expr);
  tokenizer.tokenize();
  print_tokens(tokenizer.tokens());

  std::cout << "  Stage 2: Parse to AST\n";
  Parser parser(tokenizer.tokens());
  auto ast = parser.parse();
  std::cout << "  AST:\n";
  ast->print(4);
  std::cout << "\n";

  // Verify: root is ADD, left is @0, right is MUL(@1, @2)
  bool pass = true;
  pass &= ast->num_index == -1; // root is ADD
  pass &= ast->left->num_index == 0; // left child is @0
  pass &= ast->right->num_index == -2; // right child is MUL
  pass &= ast->right->left->num_index == 1;
  pass &= ast->right->right->num_index == 2;
  std::cout << "  AST structure: " << (pass ? "PASS" : "FAIL") << "\n\n";

  std::cout << "  Stage 3: Convert to RPN\n";
  auto rpn = Parser::to_rpn(ast);
  std::cout << "  RPN: " << Parser::rpn_to_string(rpn) << "\n";
  std::cout << "  Reading RPN left to right:\n";
  std::cout << "    Push @0, Push @1, Push @2,\n";
  std::cout << "    MUL => pop @2, @1 => push (@1 * @2),\n";
  std::cout << "    ADD => pop (@1*@2), @0 => push (@0 + @1*@2)\n\n";

  std::cout << "  Stage 4: Evaluate\n";
  std::vector<SimpleTensor> inputs = {
    SimpleTensor({10.0f, 20.0f, 30.0f}),
    SimpleTensor({2.0f, 3.0f, 4.0f}),
    SimpleTensor({5.0f, 5.0f, 5.0f}),
  };
  std::cout << "  @0 = "; inputs[0].print();
  std::cout << "  @1 = "; inputs[1].print();
  std::cout << "  @2 = "; inputs[2].print();

  auto result = Evaluator::evaluate(rpn, inputs);
  std::cout << "  result = @0 + (@1 * @2) = ";
  result.print();
  std::cout << "\n";

  // Verify: [10+10, 20+15, 30+20] = [20, 35, 50]
  bool eval_pass = (result.size == 3);
  eval_pass &= (std::abs(result[0] - 20.0f) < 1e-5);
  eval_pass &= (std::abs(result[1] - 35.0f) < 1e-5);
  eval_pass &= (std::abs(result[2] - 50.0f) < 1e-5);
  std::cout << "  Expected: [20, 35, 50]\n";
  std::cout << "  Evaluation: " << (eval_pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* Demo 3: Complex expression "mul(@0,add(@1,mul(@2,@3)))"             */
/* ------------------------------------------------------------------ */
void demo_complex_parse() {
  std::cout << "========== Demo 3: Deeply Nested Expression ==========\n\n";
  std::cout << "  Expression: \"mul(@0,add(@1,mul(@2,@3)))\"\n";
  std::cout << "  This means: output = @0 * (@1 + (@2 * @3))\n\n";

  std::string expr = "mul(@0,add(@1,mul(@2,@3)))";

  std::cout << "  Stage 1: Tokenize\n";
  Tokenizer tokenizer(expr);
  tokenizer.tokenize();
  print_tokens(tokenizer.tokens());

  std::cout << "  Stage 2: Parse to AST\n";
  Parser parser(tokenizer.tokens());
  auto ast = parser.parse();
  std::cout << "  AST:\n";
  ast->print(4);
  std::cout << "\n";

  std::cout << "  Stage 3: Convert to RPN\n";
  auto rpn = Parser::to_rpn(ast);
  std::cout << "  RPN: " << Parser::rpn_to_string(rpn) << "\n\n";

  std::cout << "  Stage 4: Evaluate\n";
  std::vector<SimpleTensor> inputs = {
    SimpleTensor({2.0f, 2.0f, 2.0f}),
    SimpleTensor({1.0f, 2.0f, 3.0f}),
    SimpleTensor({4.0f, 3.0f, 2.0f}),
    SimpleTensor({5.0f, 4.0f, 3.0f}),
  };
  std::cout << "  @0 = "; inputs[0].print();
  std::cout << "  @1 = "; inputs[1].print();
  std::cout << "  @2 = "; inputs[2].print();
  std::cout << "  @3 = "; inputs[3].print();

  auto result = Evaluator::evaluate(rpn, inputs);
  std::cout << "  result = @0 * (@1 + (@2 * @3)) = ";
  result.print();
  std::cout << "\n";

  // @2*@3 = [20, 12, 6]
  // @1 + (@2*@3) = [21, 14, 9]
  // @0 * (@1+@2*@3) = [42, 28, 18]
  bool eval_pass = (result.size == 3);
  eval_pass &= (std::abs(result[0] - 42.0f) < 1e-5);
  eval_pass &= (std::abs(result[1] - 28.0f) < 1e-5);
  eval_pass &= (std::abs(result[2] - 18.0f) < 1e-5);
  std::cout << "  Step by step:\n";
  std::cout << "    @2 * @3    = [20, 12, 6]\n";
  std::cout << "    @1 + above = [21, 14, 9]\n";
  std::cout << "    @0 * above = [42, 28, 18]\n";
  std::cout << "  Evaluation: " << (eval_pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* Demo 4: Step-by-step RPN evaluation trace                           */
/* ------------------------------------------------------------------ */
void demo_rpn_trace() {
  std::cout << "========== Demo 4: RPN Evaluation Trace ==========\n\n";
  std::cout << "  Let's trace the stack during evaluation of\n";
  std::cout << "  \"add(@0,mul(@1,@2))\" with inputs:\n";
  std::cout << "    @0 = [1, 2, 3]\n";
  std::cout << "    @1 = [4, 5, 6]\n";
  std::cout << "    @2 = [7, 8, 9]\n\n";

  std::string expr = "add(@0,mul(@1,@2))";
  Tokenizer tokenizer(expr);
  tokenizer.tokenize();
  Parser parser(tokenizer.tokens());
  auto ast = parser.parse();
  auto rpn = Parser::to_rpn(ast);

  std::cout << "  RPN: " << Parser::rpn_to_string(rpn) << "\n\n";

  // Manual trace
  std::cout << "  Evaluation trace (stack grows to the right):\n";
  std::cout << "  " << std::string(50, '-') << "\n";

  std::vector<SimpleTensor> inputs = {
    SimpleTensor({1.0f, 2.0f, 3.0f}),
    SimpleTensor({4.0f, 5.0f, 6.0f}),
    SimpleTensor({7.0f, 8.0f, 9.0f}),
  };

  std::vector<SimpleTensor> stack;
  for (size_t i = 0; i < rpn.size(); i++) {
    const auto& node = rpn[i];
    if (node->is_input()) {
      stack.push_back(inputs[node->num_index]);
      std::cout << "  Step " << i << ": Push @" << node->num_index
                << " " << std::string(15, ' ');
    } else if (node->num_index == -1) {
      auto b = stack.back(); stack.pop_back();
      auto a = stack.back(); stack.pop_back();
      auto r = Evaluator::element_wise_add(a, b);
      stack.push_back(r);
      std::cout << "  Step " << i << ": ADD  = a + b       ";
    } else if (node->num_index == -2) {
      auto b = stack.back(); stack.pop_back();
      auto a = stack.back(); stack.pop_back();
      auto r = Evaluator::element_wise_mul(a, b);
      stack.push_back(r);
      std::cout << "  Step " << i << ": MUL  = a * b       ";
    }
    std::cout << "  Stack: [" << stack.size() << " tensors]";
    if (!stack.empty()) {
      std::cout << "  top = ";
      stack.back().print();
    }
    std::cout << "\n";
  }

  std::cout << "  " << std::string(50, '-') << "\n";
  std::cout << "  Final result: ";
  stack.back().print();
  std::cout << "\n";

  // Verify: @0 + (@1 * @2) = [1,2,3] + [28,40,54] = [29, 42, 57]
  auto& result = stack.back();
  bool pass = (result.size == 3);
  pass &= (std::abs(result[0] - 29.0f) < 1e-5);
  pass &= (std::abs(result[1] - 42.0f) < 1e-5);
  pass &= (std::abs(result[2] - 57.0f) < 1e-5);
  std::cout << "  Expected: [29, 42, 57]\n";
  std::cout << "  Trace: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* Demo 5: Error handling                                              */
/* ------------------------------------------------------------------ */
void demo_errors() {
  std::cout << "========== Demo 5: Error Handling ==========\n\n";

  // Test 1: Unbalanced parentheses
  {
    std::cout << "  --- Test 1: Unbalanced parens \"add(@0,@1)\" ---\n";
    // Actually this is balanced; let's use "add(@0,@1"
    std::string expr = "add(@0,@1";
    try {
      Tokenizer tokenizer(expr);
      tokenizer.tokenize();
      Parser parser(tokenizer.tokens());
      parser.parse();
      std::cout << "  Result: NO ERROR (unexpected!)\n";
    } catch (const std::runtime_error& e) {
      std::cout << "  Caught: " << e.what() << "\n";
      std::cout << "  Result: PASS (error detected)\n";
    }
  }

  // Test 2: Missing operand
  {
    std::cout << "\n  --- Test 2: Missing operand \"add(@0)\" ---\n";
    std::string expr = "add(@0)";
    try {
      Tokenizer tokenizer(expr);
      tokenizer.tokenize();
      Parser parser(tokenizer.tokens());
      parser.parse();
      std::cout << "  Result: NO ERROR (unexpected!)\n";
    } catch (const std::runtime_error& e) {
      std::cout << "  Caught: " << e.what() << "\n";
      std::cout << "  Result: PASS (error detected)\n";
    }
  }

  // Test 3: Valid but tricky — parenthesized group
  {
    std::cout << "\n  --- Test 3: Paren group \"(add(@0,@1))\" ---\n";
    std::string expr = "(add(@0,@1))";
    try {
      Tokenizer tokenizer(expr);
      tokenizer.tokenize();
      Parser parser(tokenizer.tokens());
      auto ast = parser.parse();
      std::cout << "  Parsed successfully!\n";
      std::cout << "  AST:\n";
      ast->print(4);

      // Evaluate
      auto rpn = Parser::to_rpn(ast);
      std::vector<SimpleTensor> inputs = {
        SimpleTensor({1.0f, 2.0f}),
        SimpleTensor({3.0f, 4.0f}),
      };
      auto result = Evaluator::evaluate(rpn, inputs);
      std::cout << "  Result = "; result.print();
      std::cout << "  Result: PASS\n";
    } catch (const std::runtime_error& e) {
      std::cout << "  Caught: " << e.what() << "\n";
      std::cout << "  Result: FAIL (should have parsed)\n";
    }
  }
}

/* ------------------------------------------------------------------ */
/* Demo 6: Summary                                                     */
/* ------------------------------------------------------------------ */
void demo_summary() {
  std::cout << "========== Demo 6: Pipeline Summary ==========\n\n";
  std::cout << "  Complete pipeline for PNNX expression evaluation:\n\n";
  std::cout << "  1. Expression String\n";
  std::cout << "     \"add(@0,mul(@1,@2))\"\n\n";
  std::cout << "  2. Tokenizer (Day 12)\n";
  std::cout << "     [ADD, LPAREN, INPUT_NUM(0), COMMA, MUL,\n";
  std::cout << "      LPAREN, INPUT_NUM(1), COMMA, INPUT_NUM(2),\n";
  std::cout << "      RPAREN, RPAREN]\n\n";
  std::cout << "  3. Parser (Day 13) — AST\n";
  std::cout << "     ADD\n";
  std::cout << "     / \\\n";
  std::cout << "    @0  MUL\n";
  std::cout << "       /   \\\n";
  std::cout << "      @1    @2\n\n";
  std::cout << "  4. AST -> RPN (Day 13)\n";
  std::cout << "     [@0, @1, @2, MUL, ADD]\n\n";
  std::cout << "  5. Evaluate (Day 13)\n";
  std::cout << "     Stack trace:\n";
  std::cout << "       Push @0        => [@0]\n";
  std::cout << "       Push @1        => [@0, @1]\n";
  std::cout << "       Push @2        => [@0, @1, @2]\n";
  std::cout << "       MUL            => [@0, (@1*@2)]\n";
  std::cout << "       ADD            => [@0+@1*@2]\n\n";
  std::cout << "  6. Output Tensor\n";
  std::cout << "     The single tensor left on the stack!\n\n";

  std::cout << "  This is how PNNX expression operators work!\n";
  std::cout << "  In the full KuiperInfer framework:\n";
  std::cout << "    - Tensors are N-dimensional (arma::Cube)\n";
  std::cout << "    - Expressions are evaluated during graph build\n";
  std::cout << "    - The RPN determines execution order\n";
  std::cout << "    - Results connect to the next layer\n\n";
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 13: Expression Parser &\n";
  std::cout << "  Evaluator (Syntax Analysis)\n";
  std::cout << "========================================\n\n";

  demo_basic_parse();
  demo_nested_parse();
  demo_complex_parse();
  demo_rpn_trace();
  demo_errors();
  demo_summary();

  std::cout << "========================================\n";
  std::cout << "  Day 13 Complete!\n";
  std::cout << "  Learned:\n";
  std::cout << "    - Recursive descent parser\n";
  std::cout << "      (grammar-driven)\n";
  std::cout << "    - Abstract Syntax Tree (AST)\n";
  std::cout << "      (tree representation)\n";
  std::cout << "    - RPN conversion\n";
  std::cout << "      (post-order traversal)\n";
  std::cout << "    - Stack-based evaluation\n";
  std::cout << "      (RPN + inputs = result)\n";
  std::cout << "    - Full pipeline:\n";
  std::cout << "      String -> Tokens -> AST ->\n";
  std::cout << "      RPN -> Output Tensor\n";
  std::cout << "    - Error detection:\n";
  std::cout << "      unbalanced parens,\n";
  std::cout << "      missing operands\n";
  std::cout << "    - Parenthesized groups\n";
  std::cout << "      as expression wrappers\n";
  std::cout << "========================================\n";

  return 0;
}
