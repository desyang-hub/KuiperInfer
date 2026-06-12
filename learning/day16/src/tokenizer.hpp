/**
 * src/tokenizer.hpp
 *
 * Day 12: Expression Tokenizer
 *
 * Implements a lexical analyzer (tokenizer) for PNNX-style
 * expression strings like:
 *   "add(@0,mul(@1,@2))"
 *   "mul(@0,add(@1,@2))"
 *
 * Token types:
 *   - ADD        ("add")       binary addition operator
 *   - MUL        ("mul")       binary multiplication operator
 *   - INPUT_NUM  ("@N")        input tensor reference (@0, @1, ...)
 *   - COMMA      (",")         argument separator
 *   - LPAREN     ("(")         left parenthesis
 *   - RPAREN     (")")         right parenthesis
 *
 * The tokenizer scans the input string character by character,
 * strips whitespace, and emits a stream of tokens with
 * start/end positions in the original string.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace learn_infer {
namespace tutorial {

enum class TokenType {
  Unknown = 0,
  Add,
  Mul,
  InputNumber,
  Comma,
  LParen,
  RParen,
};

// Convert token type to a human-readable string
inline std::string token_type_to_string(TokenType type) {
  switch (type) {
    case TokenType::Unknown:     return "UNKNOWN";
    case TokenType::Add:         return "ADD";
    case TokenType::Mul:         return "MUL";
    case TokenType::InputNumber: return "INPUT_NUM";
    case TokenType::Comma:       return "COMMA";
    case TokenType::LParen:      return "LPAREN";
    case TokenType::RParen:      return "RPAREN";
  }
  return "???";
}

/**
 * A single token produced by the tokenizer.
 */
struct Token {
  TokenType   type;
  std::string text;        // The raw text of this token
  int32_t     start_pos;   // Start position in the original string
  int32_t     end_pos;     // End position (exclusive)
  int32_t     input_index; // For INPUT_NUM: the number after @, else -1

  Token() : type(TokenType::Unknown), start_pos(0), end_pos(0), input_index(-1) {}
  Token(TokenType type, const std::string& text, int32_t start, int32_t end, int32_t idx = -1)
      : type(type), text(text), start_pos(start), end_pos(end), input_index(idx) {}

  std::string to_string() const;
};

/**
 * Simple expression tokenizer.
 *
 * Scans an expression string like "add(@0,mul(@1,@2))" and
 * produces a list of tokens.
 */
class Tokenizer {
 public:
  explicit Tokenizer(const std::string& expression);

  /**
   * Tokenize the expression string.
   * Call this once after construction.
   */
  void tokenize();

  /** Get the produced tokens. */
  const std::vector<Token>& tokens() const { return tokens_; }

  /** Get the original (whitespace-stripped) expression. */
  const std::string& expression() const { return expr_; }

 private:
  std::string expr_;
  std::vector<Token> tokens_;

  // Internal scan position
  int32_t pos_;

  // Peek at current character without consuming
  char peek() const;
  // Consume current character and advance
  char advance();
  // Skip whitespace
  void skip_whitespace();

  // Recognize specific token patterns
  void consume_add();
  void consume_mul();
  void consume_input_number();
  void consume_char(TokenType expected_type, char expected_char);
};

}  // namespace tutorial
}  // namespace learn_infer
