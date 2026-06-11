/**
 * day4/include/layer/layer.hpp
 *
 * Layer -- abstract base class for neural network layers.
 *
 * Each layer implements a Forward pass: given a vector of input tensors,
 * it computes and writes output tensors.  The virtual Type() method
 * returns the layer's string identifier (e.g. "nn.ReLU").
 *
 * Design:
 *   - Template parameter T is the data type (typically float).
 *   - Forward takes inputs by const reference and writes outputs by pointer
 *     reference so that the caller controls tensor lifecycle.
 *   - Returns StatusCode for error handling.
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include "../tensor.hpp"
#include <string>
#include <vector>
#include <memory>

namespace learn_infer {

/**
 * Simple enum for return codes.
 * In a real inference engine this would integrate with a broader status
 * system (error messages, stack traces, etc.).
 */
enum class StatusCode {
  kSuccess = 0,
  kError = 1,
  kNotImplemented = 2,
  kInvalidInput = 3,
  kShapeMismatch = 4
};

/**
 * Layer<T> -- abstract base class for all layers.
 *
 * Template parameter T is the element type of input/output tensors.
 *
 * Subclasses MUST implement:
 *   - virtual StatusCode Forward(const std::vector<std::shared_ptr<Tensor<T>>>&,
 *                                std::vector<std::shared_ptr<Tensor<T>>>&)
 *   - virtual std::string Type() const
 */
template <typename T>
class Layer {
 public:
  virtual ~Layer() = default;

  /**
   * Forward pass.
   *
   * @param inputs   Vector of input tensors.
   * @param outputs  Reference to output tensor vector; the layer writes
   *                 results here (may replace the entire vector).
   * @return StatusCode indicating success or failure.
   */
  virtual StatusCode Forward(
      const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
      std::vector<std::shared_ptr<Tensor<T>>>& outputs) = 0;

  /**
   * Return the layer type string, e.g. "nn.ReLU", "nn.Conv2D".
   */
  virtual std::string Type() const = 0;

 protected:
  Layer() = default;
  Layer(const Layer&) = default;
  Layer& operator=(const Layer&) = default;
};

}  // namespace learn_infer

#endif  // LAYER_HPP
