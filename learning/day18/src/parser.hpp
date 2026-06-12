/**
 * src/parser.hpp
 *
 * Day 13: Expression Parser (Syntax Analysis)
 *
 * Builds on Day 12's tokenizer to parse tokens into an
 * Abstract Syntax Tree (AST), then convert the AST to
 * Reverse Polish Notation (RPN) for evaluation.
 *
 * Grammar:
 *   Expr     -> Term { ('+' | 'add') Term }
 *   Term     -> Factor { ('*' | 'mul') Factor }
 *   Factor   -> '@' NUMBER  |  '(' Expr ')'
 *              | 'add' '(' Expr ',' Expr ')'
 *              | 'mul' '(' Expr ',' Expr ')'
 *
 * For PNNX expressions, we only support add and mul,
 * so the grammar simplifies to:
 *   Expr   -> Primary { op Primary }
 *   Primary -> '@' NUMBER | '(' Expr ')'
 *                | 'add' '(' Expr ',' Expr ')'
 *                | 'mul' '(' Expr ',' Expr ')'
 */

#pragma once

#include "src/tokenizer.hpp"
#include <memory>
#include <string>
#include <vector>

namespace learn_infer {
namespace tutorial {

/**
 * Node in the expression syntax tree.
 *
 * Each node is either:
 *   - A leaf (input tensor reference): num_index >= 0, children null
 *   - An operator: num_index < 0, left/right children set
 *     num_index == 0 -> ADD, num_index == 1 -> MUL
 */
struct ASTNode {
  /// If >= 0: this is an input tensor reference to @num_index
  /// If < 0: this is an operator (-1=ADD, -2=MUL)
  int32_t num_index = -1;

  /// Left child (first operand for operators)
  std::shared_ptr<ASTNode> left = nullptr;

  /// Right child (second operand for operators)
  std::shared_ptr<ASTNode> right = nullptr;

  /**
   * Create an input tensor reference node (@index).
   */
  static std::shared_ptr<ASTNode> make_input(int32_t index);

  /**
   * Create an ADD operator node.
   */
  static std::shared_ptr<ASTNode> make_add(
      std::shared_ptr<ASTNode> left,
      std::shared_ptr<ASTNode> right);

  /**
   * Create a MUL operator node.
   */
  static std::shared_ptr<ASTNode> make_mul(
      std::shared_ptr<ASTNode> left,
      std::shared_ptr<ASTNode> right);

  /**
   * Check if this node is a leaf (input reference).
   */
  bool is_input() const { return num_index >= 0; }

  /**
   * Check if this node is an operator.
   */
  bool is_operator() const { return num_index < 0; }

  /**
   * Get operator name for debugging.
   */
  std::string op_name() const;

  /**
   * Print the AST subtree in a readable format.
   */
  void print(int indent = 0) const;
};

/**
 * Recursive descent parser.
 *
 * Takes the tokens produced by Day 12's tokenizer and
 * builds an AST using recursive descent parsing.
 *
 * Then converts the AST to RPN (post-order traversal),
 * which can be evaluated in a single left-to-right pass.
 */
class Parser {
 public:
  explicit Parser(const std::vector<Token>& tokens);

  /**
   * Parse the token stream into an AST.
   * Returns the root node of the syntax tree.
   */
  std::shared_ptr<ASTNode> parse();

  /**
   * Convert an AST to RPN (Reverse Polish Notation).
   *
   * RPN is a post-order traversal of the tree:
   *   left_subtree, right_subtree, operator
   *
   * For "add(@0,mul(@1,@2))" the RPN is:
   *   [@0, @1, @2, MUL, ADD]
   *
   * This order allows stack-based evaluation in one pass.
   */
  static std::vector<std::shared_ptr<ASTNode>> to_rpn(
      const std::shared_ptr<ASTNode>& root);

  /**
   * Pretty-print the RPN sequence.
   */
  static std::string rpn_to_string(
      const std::vector<std::shared_ptr<ASTNode>>& rpn);

 private:
  const std::vector<Token>& tokens_;
  size_t pos_;

  /**
   * Parse the current expression starting at pos_.
   * Advances pos_ past the parsed expression.
   */
  std::shared_ptr<ASTNode> parse_expression();

  /**
   * Parse a primary expression:
   *   - Input number: @0, @1, @10, ...
   *   - Parenthesized: ( expr )
   *   - Function call: add(expr, expr) or mul(expr, expr)
   */
  std::shared_ptr<ASTNode> parse_primary();

  /**
   * Look at the current token without consuming it.
   */
  const Token& peek() const;

  /**
   * Consume the current token and advance.
   */
  Token consume();

  /**
   * Consume a token of expected type, or throw.
   */
  void expect(TokenType type);
};

}  // namespace tutorial
}  // namespace learn_infer
