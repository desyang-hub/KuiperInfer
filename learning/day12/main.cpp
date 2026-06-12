/**
 * day12/main.cpp
 *
 * Day 12 Demo: Expression Tokenizer (Lexical Analysis)
 *
 * Builds on Day 11 (forward execution) and adds:
 *   - A hand-written tokenizer for PNNX expression strings
 *   - Token types: ADD, MUL, INPUT_NUM, COMMA, LPAREN, RPAREN
 *   - Scanning "add(@0,mul(@1,@2))" into a token stream
 *   - Error detection (unknown chars, incomplete tokens)
 *
 * The tokenizer is the first stage of the expression parser pipeline:
 *   Expression String -> Tokenizer -> Tokens -> Parser -> AST -> RPN -> Evaluate
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day12
 */

#include "src/tokenizer.hpp"
#include "include/layer/relu.hpp"
#include "include/layer/sigmoid.hpp"
#include "include/layer/softmax.hpp"
#include "include/layer/batchnorm2d.hpp"
#include "include/layer/convolution.hpp"
#include "include/layer/maxpooling.hpp"
#include "include/layer/linear.hpp"
#include "include/runtime/runtime_graph.hpp"
#include "include/tensor_util.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <chrono>
#include <sstream>

using namespace learn_infer;
using namespace learn_infer::util;
using namespace learn_infer::tutorial;

/* ------------------------------------------------------------------ */
/* Day 11 demos (kept for continuity) — simplified to save space       */
/* ------------------------------------------------------------------ */
void demo_linear_chain();
void demo_branching();
void demo_forward_with_tracking();

void demo_linear_chain() {
  std::cout << "========== Demo 1: Linear Chain ==========\n\n";
  ConvolutionRegister<float>("nn.Conv2d",
      1, 4, 4, 3, 3, 1, 1, 0, 0, 1,
      std::vector<float>(9, 1.0f), {0.0f});
  MaxPoolingRegister<float>("nn.MaxPool2d", 2, 2, 2, 2, 0, 0);

  RuntimeGraph<float> graph;
  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("conv1", "nn.Conv2d", {"input_data"}, {"conv_out"});
  graph.AddOperator("relu1", "nn.ReLU", {"conv_out"}, {"relu_out"});
  graph.AddOperator("pool1", "nn.MaxPool2d", {"relu_out"}, {"pool_out"});
  graph.AddOperator("output", "Output", {"pool_out"}, {});

  graph.Build();
  graph.ReverseTopoSort();

  auto input_tensor = Tensor<float>::Create(1, 4, 4, {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
  });
  graph.SetInput("input_data", input_tensor);
  graph.Forward();

  auto output = graph.GetOutput("pool_out");
  if (output) output->print("    output");
}

void demo_branching() {
  std::cout << "========== Demo 2: Branching ==========\n\n";
  LinearRegister<float>("branch_a", 4, 2,
      {1.0f, 0.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.5f, -1.0f},
      {0.0f, 0.0f});
  LinearRegister<float>("branch_b", 4, 2,
      {0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f},
      {0.0f, 0.0f});

  RuntimeGraph<float> graph;
  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("relu_branch", "nn.ReLU", {"input_data"}, {"relu_out"});
  graph.AddOperator("sigmoid_branch", "nn.Sigmoid", {"input_data"}, {"sigmoid_out"});
  graph.AddOperator("linear_a", "branch_a", {"relu_out"}, {"fc_a_out"});
  graph.AddOperator("linear_b", "branch_b", {"sigmoid_out"}, {"fc_b_out"});
  graph.AddOperator("output", "Output", {"fc_a_out", "fc_b_out"}, {});

  graph.Build();
  graph.ReverseTopoSort();

  graph.SetInput("input_data", Tensor<float>::Create(1, 1, 4, {-1.0f, 2.0f, -3.0f, 4.0f}));
  graph.Forward();

  if (graph.GetOutput("fc_a_out")) graph.GetOutput("fc_a_out")->print("    branch_a");
  if (graph.GetOutput("fc_b_out")) graph.GetOutput("fc_b_out")->print("    branch_b");
}

void demo_forward_with_tracking() {
  std::cout << "========== Demo 3: Forward with Shape Tracking ==========\n\n";
  ConvolutionRegister<float>("track_conv",
      1, 6, 6, 3, 3, 1, 1, 0, 0, 1,
      std::vector<float>(9, 0.5f), {0.0f});
  MaxPoolingRegister<float>("track_pool", 2, 2, 2, 2, 0, 0);
  BatchNorm2DRegister<float>("track_bn", 1, {0.0f}, {1.0f}, {1.0f}, {0.0f});

  RuntimeGraph<float> graph;
  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("conv1", "track_conv", {"input_data"}, {"conv_out"});
  graph.AddOperator("relu1", "nn.ReLU", {"conv_out"}, {"relu_out"});
  graph.AddOperator("pool1", "track_pool", {"relu_out"}, {"pool_out"});
  graph.AddOperator("bn1", "track_bn", {"pool_out"}, {"bn_out"});
  graph.AddOperator("output", "Output", {"bn_out"}, {});

  graph.Build();
  graph.ReverseTopoSort();

  std::vector<float> input_data(36);
  for (int i = 0; i < 36; i++) input_data[i] = static_cast<float>(i + 1);
  graph.SetInput("input_data", Tensor<float>::Create(1, 6, 6, input_data));
  graph.Forward();

  auto bn_out = graph.GetOutput("bn_out");
  if (bn_out) {
    auto s = bn_out->shapes();
    std::cout << "  Output shape=[" << s[0] << ", " << s[1] << ", " << s[2] << "]\n";
    bn_out->print("    bn_output");
  }
}

/* ------------------------------------------------------------------ */
/* Day 12 New Demos: Tokenizer                                         */
/* ------------------------------------------------------------------ */

static void print_tokens(const std::vector<Token>& tokens,
                         const std::string& label,
                         const std::string& expression) {
  std::cout << "\n  " << label << "\n";
  std::cout << "  Expression: \"" << expression << "\"\n";
  std::cout << "  " << std::string(60, '-') << "\n";
  std::cout << "  #" << "  " << std::setw(12) << "Type"
            << "  " << std::setw(8) << "Text"
            << "  " << std::setw(10) << "Position" << "\n";
  std::cout << "  " << std::string(60, '-') << "\n";
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
  std::cout << "  " << std::string(60, '-') << "\n";
  std::cout << "  Total tokens: " << tokens.size() << "\n\n";
}

void demo_tokenizer_basic() {
  std::cout << "========== Day 12 Demo 4: Basic Tokenizer ==========\n\n";
  std::cout << "  The tokenizer breaks an expression string into tokens.\n";
  std::cout << "  This is the FIRST stage of the expression parser.\n\n";

  std::string expr = "add(@0,@1)";

  std::cout << "  --- Tokenizing: \"" << expr << "\" ---\n";
  Tokenizer tokenizer(expr);
  tokenizer.tokenize();
  print_tokens(tokenizer.tokens(), "Tokens", expr);

  // Verify: add ( @0 , @1 )
  const auto& tokens = tokenizer.tokens();
  bool pass = (tokens.size() == 6);
  pass &= (tokens[0].type == TokenType::Add && tokens[0].text == "add");
  pass &= (tokens[1].type == TokenType::LParen && tokens[1].text == "(");
  pass &= (tokens[2].type == TokenType::InputNumber && tokens[2].input_index == 0);
  pass &= (tokens[3].type == TokenType::Comma && tokens[3].text == ",");
  pass &= (tokens[4].type == TokenType::InputNumber && tokens[4].input_index == 1);
  pass &= (tokens[5].type == TokenType::RParen && tokens[5].text == ")");

  std::cout << "  Verification: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

void demo_tokenizer_nested() {
  std::cout << "========== Day 12 Demo 5: Nested Expression ==========\n\n";
  std::cout << "  Nested expressions require the tokenizer to handle\n";
  std::cout << "  multiple levels of parentheses.\n\n";

  // add(@0,mul(@1,@2))  =>  @0 + (@1 * @2)
  std::string expr = "add(@0,mul(@1,@2))";

  std::cout << "  --- Tokenizing: \"" << expr << "\" ---\n";
  Tokenizer tokenizer(expr);
  tokenizer.tokenize();
  print_tokens(tokenizer.tokens(), "Tokens", expr);

  const auto& tokens = tokenizer.tokens();
  bool pass = (tokens.size() == 11);
  if (pass) {
    pass &= tokens[0].type == TokenType::Add;
    pass &= tokens[1].type == TokenType::LParen;
    pass &= tokens[2].type == TokenType::InputNumber && tokens[2].input_index == 0;
    pass &= tokens[3].type == TokenType::Comma;
    pass &= tokens[4].type == TokenType::Mul;
    pass &= tokens[5].type == TokenType::LParen;
    pass &= tokens[6].type == TokenType::InputNumber && tokens[6].input_index == 1;
    pass &= tokens[7].type == TokenType::Comma;
    pass &= tokens[8].type == TokenType::InputNumber && tokens[8].input_index == 2;
    pass &= tokens[9].type == TokenType::RParen;
    pass &= tokens[10].type == TokenType::RParen;
  }

  std::cout << "  Verification: " << (pass ? "PASS" : "FAIL") << "\n";
  std::cout << "  This corresponds to: @0 + (@1 * @2)\n";
  std::cout << "  The tokenizer produces a flat token stream.\n";
  std::cout << "  The NEXT step (Day 13) builds a tree from this stream.\n\n";
}

void demo_tokenizer_complex() {
  std::cout << "========== Day 12 Demo 6: Complex Expression ==========\n\n";
  std::cout << "  Real-world PNNX expressions can be quite complex.\n\n";

  // mul(@0,add(@1,mul(@2,@3)))  =>  @0 * (@1 + (@2 * @3))
  std::string expr = "mul(@0,add(@1,mul(@2,@3)))";

  std::cout << "  --- Tokenizing: \"" << expr << "\" ---\n";
  Tokenizer tokenizer(expr);
  tokenizer.tokenize();
  print_tokens(tokenizer.tokens(), "Tokens", expr);

  const auto& tokens = tokenizer.tokens();
  bool pass = (tokens.size() == 16);
  if (pass) {
    pass &= tokens[0].type == TokenType::Mul;
    pass &= tokens[4].type == TokenType::Add;
    pass &= tokens[8].type == TokenType::Mul;
    // Check parentheses balance
    int paren_depth = 0;
    bool balanced = true;
    for (const auto& t : tokens) {
      if (t.type == TokenType::LParen) paren_depth++;
      if (t.type == TokenType::RParen) paren_depth--;
      if (paren_depth < 0) { balanced = false; break; }
    }
    pass &= (paren_depth == 0 && balanced);
  }

  std::cout << "  Verification: " << (pass ? "PASS" : "FAIL") << "\n";
  int depth = 0;
  for (const auto& t : tokens) {
    if (t.type == TokenType::LParen) depth++;
    if (t.type == TokenType::RParen) depth--;
  }
  std::cout << "  Parentheses balanced: " << (depth == 0 ? "YES" : "NO") << "\n\n";
}

void demo_tokenizer_error() {
  std::cout << "========== Day 12 Demo 7: Error Handling ==========\n\n";
  std::cout << "  The tokenizer should report errors for invalid input.\n\n";

  // Test 1: Unknown character
  {
    std::string expr = "add(@0,@x)";
    std::cout << "  --- Test 1: Unknown char 'x' in \"" << expr << "\" ---\n";
    try {
      Tokenizer tokenizer(expr);
      tokenizer.tokenize();
      std::cout << "  Result: NO ERROR (unexpected!)\n";
    } catch (const std::runtime_error& e) {
      std::cout << "  Caught error: " << e.what() << "\n";
      std::cout << "  Result: PASS (error detected)\n";
    }
  }

  // Test 2: Incomplete "add" -> "ad"
  {
    std::string expr = "ad(@0,@1)";
    std::cout << "\n  --- Test 2: Incomplete keyword \"" << expr << "\" ---\n";
    try {
      Tokenizer tokenizer(expr);
      tokenizer.tokenize();
      std::cout << "  Result: NO ERROR (unexpected!)\n";
    } catch (const std::runtime_error& e) {
      std::cout << "  Caught error: " << e.what() << "\n";
      std::cout << "  Result: PASS (error detected)\n";
    }
  }

  // Test 3: Whitespace is handled
  {
    std::string expr = "add ( @0 , @1 )";
    std::cout << "\n  --- Test 3: Whitespace tolerance \"" << expr << "\" ---\n";
    try {
      Tokenizer tokenizer(expr);
      tokenizer.tokenize();
      std::cout << "  Tokenized successfully: " << tokenizer.tokens().size() << " tokens\n";
      print_tokens(tokenizer.tokens(), "Tokens with whitespace stripped", expr);
      std::cout << "  Result: PASS (whitespace handled)\n";
    } catch (const std::runtime_error& e) {
      std::cout << "  Caught error: " << e.what() << "\n";
      std::cout << "  Result: FAIL (should have succeeded)\n";
    }
  }

  // Test 4: Multi-digit input index
  {
    std::string expr = "add(@10,@21)";
    std::cout << "\n  --- Test 4: Multi-digit index \"" << expr << "\" ---\n";
    try {
      Tokenizer tokenizer(expr);
      tokenizer.tokenize();
      const auto& tokens = tokenizer.tokens();
      bool pass = (tokens[2].input_index == 10 && tokens[4].input_index == 21);
      std::cout << "  @10 index=" << tokens[2].input_index
                << "  @21 index=" << tokens[4].input_index << "\n";
      std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    } catch (const std::runtime_error& e) {
      std::cout << "  Caught error: " << e.what() << "\n";
      std::cout << "  Result: FAIL\n";
    }
  }
}

void demo_pipeline_overview() {
  std::cout << "========== Day 12 Demo 8: Full Pipeline Overview ==========\n\n";
  std::cout << "  The expression parser pipeline has these stages:\n\n";
  std::cout << "  Stage 1: Tokenizer   (Day 12) — DONE!\n";
  std::cout << "    \"add(@0,mul(@1,@2))\" --> [ADD, LPAREN, INPUT_NUM(0), COMMA, MUL, ...]\n\n";
  std::cout << "  Stage 2: Parser      (Day 13) — NEXT\n";
  std::cout << "    Tokens --> AST (Abstract Syntax Tree)\n";
  std::cout << "          --> RPN (Reverse Polish Notation)\n\n";
  std::cout << "  Stage 3: Evaluator   (Day 13) — NEXT\n";
  std::cout << "    RPN + Input Tensors --> Output Tensor values\n\n";

  std::cout << "  Token types in our expression language:\n";
  std::cout << "  " << std::string(50, '-') << "\n";
  std::cout << "  " << std::setw(12) << token_type_to_string(TokenType::Add)
            << "  -> \"add\" keyword (element-wise addition)\n";
  std::cout << "  " << std::setw(12) << token_type_to_string(TokenType::Mul)
            << "  -> \"mul\" keyword (element-wise multiplication)\n";
  std::cout << "  " << std::setw(12) << token_type_to_string(TokenType::InputNumber)
            << "  -> \"@N\" (reference to input tensor N)\n";
  std::cout << "  " << std::setw(12) << token_type_to_string(TokenType::Comma)
            << "  -> \",\" (separates arguments)\n";
  std::cout << "  " << std::setw(12) << token_type_to_string(TokenType::LParen)
            << "  -> \"(\" (open group)\n";
  std::cout << "  " << std::setw(12) << token_type_to_string(TokenType::RParen)
            << "  -> \")\" (close group)\n";
  std::cout << "  " << std::string(50, '-') << "\n";
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 12: Expression Tokenizer\n";
  std::cout << "  (Lexical Analysis)\n";
  std::cout << "========================================\n\n";

  // Day 11 demos (continuity)
  demo_linear_chain();
  demo_branching();
  demo_forward_with_tracking();

  // Day 12 new demos
  demo_tokenizer_basic();
  demo_tokenizer_nested();
  demo_tokenizer_complex();
  demo_tokenizer_error();
  demo_pipeline_overview();

  std::cout << "========================================\n";
  std::cout << "  Day 12 Complete!\n";
  std::cout << "  Learned:\n";
  std::cout << "    - Expression string format\n";
  std::cout << "      \"add(@0,mul(@1,@2))\"\n";
  std::cout << "    - Token types: ADD, MUL,\n";
  std::cout << "      INPUT_NUM, COMMA, LPAREN, RPAREN\n";
  std::cout << "    - Scanner implementation\n";
  std::cout << "      (character-by-character)\n";
  std::cout << "    - Whitespace handling\n";
  std::cout << "    - Error detection & reporting\n";
  std::cout << "    - Multi-digit input index\n";
  std::cout << "      (@0 through @N)\n";
  std::cout << "    - Pipeline: Tokenizer ->\n";
  std::cout << "      Parser -> Evaluator\n";
  std::cout << "========================================\n";

  return 0;
}
