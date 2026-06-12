/**
 * src/parser.cpp
 *
 * Implementation of the expression parser.
 */

#include "src/parser.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cassert>
#include <functional>

namespace learn_infer {
namespace tutorial {

/* ------------------------------------------------------------------ */
/* ASTNode helpers                                                     */
/* ------------------------------------------------------------------ */
std::shared_ptr<ASTNode> ASTNode::make_input(int32_t index) {
  auto node = std::make_shared<ASTNode>();
  node->num_index = index;
  return node;
}

std::shared_ptr<ASTNode> ASTNode::make_add(
    std::shared_ptr<ASTNode> left,
    std::shared_ptr<ASTNode> right) {
  auto node = std::make_shared<ASTNode>();
  node->num_index = -1; // -1 means ADD
  node->left = left;
  node->right = right;
  return node;
}

std::shared_ptr<ASTNode> ASTNode::make_mul(
    std::shared_ptr<ASTNode> left,
    std::shared_ptr<ASTNode> right) {
  auto node = std::make_shared<ASTNode>();
  node->num_index = -2; // -2 means MUL
  node->left = left;
  node->right = right;
  return node;
}

std::string ASTNode::op_name() const {
  if (num_index == -1) return "ADD";
  if (num_index == -2) return "MUL";
  return "???";
}

void ASTNode::print(int indent) const {
  std::string prefix(indent, ' ');
  if (is_input()) {
    std::cout << prefix << "@input[" << num_index << "]\n";
  } else {
    std::cout << prefix << "op: " << op_name() << "\n";
    std::cout << prefix << "  left:\n";
    if (left) left->print(indent + 4);
    std::cout << prefix << "  right:\n";
    if (right) right->print(indent + 4);
  }
}

/* ------------------------------------------------------------------ */
/* Parser implementation                                               */
/* ------------------------------------------------------------------ */
Parser::Parser(const std::vector<Token>& tokens)
    : tokens_(tokens), pos_(0) {}

const Token& Parser::peek() const {
  assert(pos_ < tokens_.size());
  return tokens_[pos_];
}

Token Parser::consume() {
  assert(pos_ < tokens_.size());
  return tokens_[pos_++];
}

void Parser::expect(TokenType type) {
  if (pos_ >= tokens_.size()) {
    throw std::runtime_error("Parser: unexpected end of tokens, "
                             "expected " + token_type_to_string(type));
  }
  if (tokens_[pos_].type != type) {
    std::ostringstream msg;
    msg << "Parser: expected " << token_type_to_string(type)
        << " but got " << token_type_to_string(tokens_[pos_].type)
        << " (\"" << tokens_[pos_].text << "\")";
    throw std::runtime_error(msg.str());
  }
  pos_++;
}

std::shared_ptr<ASTNode> Parser::parse() {
  if (tokens_.empty()) {
    throw std::runtime_error("Parser: token list is empty");
  }
  auto root = parse_expression();
  if (pos_ != tokens_.size()) {
    std::ostringstream msg;
    msg << "Parser: unexpected tokens after expression at position "
        << pos_ << " / " << tokens_.size();
    throw std::runtime_error(msg.str());
  }
  return root;
}

std::shared_ptr<ASTNode> Parser::parse_expression() {
  // Every expression starts with a primary (input, func call, or paren group)
  auto node = parse_primary();

  // The expression is fully parsed; no infix operators in PNNX format.
  return node;
}

std::shared_ptr<ASTNode> Parser::parse_primary() {
  const Token& t = peek();

  // Case 1: Input tensor reference @N
  if (t.type == TokenType::InputNumber) {
    consume();
    return ASTNode::make_input(t.input_index);
  }

  // Case 2: Function call — add(...) or mul(...)
  if (t.type == TokenType::Add || t.type == TokenType::Mul) {
    TokenType op = t.type;
    consume(); // consume 'add' or 'mul'

    expect(TokenType::LParen);

    auto left = parse_expression();
    expect(TokenType::Comma);
    auto right = parse_expression();

    expect(TokenType::RParen);

    if (op == TokenType::Add) {
      return ASTNode::make_add(left, right);
    } else {
      return ASTNode::make_mul(left, right);
    }
  }

  // Case 3: Parenthesized expression ( expr )
  if (t.type == TokenType::LParen) {
    consume(); // consume '('
    auto node = parse_expression();
    expect(TokenType::RParen);
    return node;
  }

  // Unknown token
  std::ostringstream msg;
  msg << "Parser: unexpected token " << token_type_to_string(t.type)
      << " (\"" << t.text << "\") at position " << t.start_pos;
  throw std::runtime_error(msg.str());
}

/* ------------------------------------------------------------------ */
/* RPN conversion                                                      */
/* ------------------------------------------------------------------ */
std::vector<std::shared_ptr<ASTNode>> Parser::to_rpn(
    const std::shared_ptr<ASTNode>& root) {
  std::vector<std::shared_ptr<ASTNode>> result;
  if (!root) return result;

  // Post-order traversal: left, right, node
  // Recursive approach is simpler and correct
  std::function<void(const std::shared_ptr<ASTNode>&)> traverse;
  traverse = [&](const std::shared_ptr<ASTNode>& node) {
    if (!node) return;
    if (node->left) traverse(node->left);
    if (node->right) traverse(node->right);
    result.push_back(node);
  };
  traverse(root);

  return result;
}

std::string Parser::rpn_to_string(
    const std::vector<std::shared_ptr<ASTNode>>& rpn) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < rpn.size(); i++) {
    if (i > 0) oss << ", ";
    if (rpn[i]->is_input()) {
      oss << "@" << rpn[i]->num_index;
    } else {
      oss << rpn[i]->op_name();
    }
  }
  oss << "]";
  return oss.str();
}

}  // namespace tutorial
}  // namespace learn_infer
