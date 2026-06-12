/**
 * src/tokenizer.cpp
 *
 * Implementation of the expression tokenizer.
 */

#include "src/tokenizer.hpp"
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <sstream>

namespace learn_infer {
namespace tutorial {

/* ------------------------------------------------------------------ */
/* Token helpers                                                       */
/* ------------------------------------------------------------------ */
std::string Token::to_string() const {
  std::ostringstream oss;
  oss << "[" << token_type_to_string(type)
      << " \"" << text << "\""
      << " pos=" << start_pos << "-" << end_pos;
  if (type == TokenType::InputNumber) {
    oss << " index=" << input_index;
  }
  oss << "]";
  return oss.str();
}

/* ------------------------------------------------------------------ */
/* Tokenizer implementation                                            */
/* ------------------------------------------------------------------ */
Tokenizer::Tokenizer(const std::string& expression)
    : expr_(expression), pos_(0) {}

void Tokenizer::tokenize() {
  // Strip whitespace first
  std::string stripped;
  for (char c : expr_) {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      stripped += c;
    }
  }
  if (stripped.empty()) {
    throw std::runtime_error("Tokenizer: expression is empty or contains only whitespace");
  }
  expr_ = stripped;
  tokens_.clear();
  pos_ = 0;

  // Scan character by character
  while (pos_ < static_cast<int32_t>(expr_.size())) {
    char c = peek();

    if (c == 'a') {
      consume_add();
    } else if (c == 'm') {
      consume_mul();
    } else if (c == '@') {
      consume_input_number();
    } else if (c == ',') {
      consume_char(TokenType::Comma, ',');
    } else if (c == '(') {
      consume_char(TokenType::LParen, '(');
    } else if (c == ')') {
      consume_char(TokenType::RParen, ')');
    } else {
      std::ostringstream msg;
      msg << "Tokenizer: unknown character '" << c << "' at position " << pos_
          << " in expression: " << expr_;
      throw std::runtime_error(msg.str());
    }
  }
}

char Tokenizer::peek() const {
  if (pos_ >= static_cast<int32_t>(expr_.size())) {
    throw std::runtime_error("Tokenizer: unexpected end of expression");
  }
  return expr_[pos_];
}

char Tokenizer::advance() {
  char c = expr_[pos_];
  ++pos_;
  return c;
}

void Tokenizer::skip_whitespace() {
  // Whitespace already stripped, but kept for extensibility
}

void Tokenizer::consume_add() {
  int32_t start = pos_;
  // Check "add"
  if (pos_ + 2 >= static_cast<int32_t>(expr_.size()) ||
      expr_[pos_ + 1] != 'd' || expr_[pos_ + 2] != 'd') {
    throw std::runtime_error("Tokenizer: expected 'add' at position " + std::to_string(pos_));
  }
  pos_ += 3;
  tokens_.emplace_back(TokenType::Add, "add", start, pos_);
}

void Tokenizer::consume_mul() {
  int32_t start = pos_;
  // Check "mul"
  if (pos_ + 2 >= static_cast<int32_t>(expr_.size()) ||
      expr_[pos_ + 1] != 'u' || expr_[pos_ + 2] != 'l') {
    throw std::runtime_error("Tokenizer: expected 'mul' at position " + std::to_string(pos_));
  }
  pos_ += 3;
  tokens_.emplace_back(TokenType::Mul, "mul", start, pos_);
}

void Tokenizer::consume_input_number() {
  int32_t start = pos_;
  advance(); // skip '@'

  // Must be followed by digits
  if (pos_ >= static_cast<int32_t>(expr_.size()) ||
      !std::isdigit(static_cast<unsigned char>(expr_[pos_]))) {
    throw std::runtime_error("Tokenizer: expected digit after '@' at position " +
                             std::to_string(pos_));
  }

  int32_t num_start = pos_;
  while (pos_ < static_cast<int32_t>(expr_.size()) &&
         std::isdigit(static_cast<unsigned char>(expr_[pos_]))) {
    ++pos_;
  }

  std::string num_str = expr_.substr(num_start, pos_ - num_start);
  int32_t index = std::stoi(num_str);

  tokens_.emplace_back(TokenType::InputNumber, "@" + num_str, start, pos_, index);
}

void Tokenizer::consume_char(TokenType expected_type, char expected_char) {
  int32_t start = pos_;
  char c = advance();
  if (c != expected_char) {
    throw std::runtime_error("Tokenizer: expected '" + std::string(1, expected_char) +
                             "' at position " + std::to_string(start));
  }
  tokens_.emplace_back(expected_type, std::string(1, c), start, pos_);
}

}  // namespace tutorial
}  // namespace learn_infer
