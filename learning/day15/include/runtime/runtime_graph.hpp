/**
 * day10/include/runtime/runtime_graph.hpp
 *
 * Mini runtime graph for learning topological sort.
 *
 * Concepts:
 *   - RuntimeOperator: a node in the computation graph
 *   - RuntimeGraph: owns all operators, builds connections, performs topo sort
 *   - Topological sort determines execution order so every operator runs
 *     only after all its inputs are ready.
 *
 * This is a simplified version of KuiperInfer's full runtime graph,
 * focused on demonstrating the topological sort algorithm.
 */

#ifndef RUNTIME_GRAPH_HPP
#define RUNTIME_GRAPH_HPP

#include "../layer/layer.hpp"
#include "../layer/layer_factory.hpp"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>

namespace learn_infer {

/**
 * RuntimeOperator<T> — one node in the computation graph.
 *
 * Each operator has:
 *   - name: unique identifier (e.g. "conv1", "bn1")
 *   - type: layer type string (e.g. "nn.Conv2d", "nn.ReLU")
 *   - input_names: names of operators whose outputs feed into this one
 *   - output_names: names given to this operator's outputs
 *   - start_time: execution order index (assigned by topo sort)
 *   - has_forward: flag used during DFS-based topo sort
 *   - layer: the concrete Layer<T>* (created by factory after topo sort)
 */
template <typename T>
class RuntimeOperator {
 public:
  std::string name;
  std::string type;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Topo sort state
  int start_time = -1;  // execution order, -1 = unassigned
  bool has_forward = false;  // DFS visited flag

  // Layer instance (created by factory)
  std::shared_ptr<Layer<T>> layer;

  // Input/output tensors (set during graph execution)
  std::map<std::string, std::shared_ptr<Tensor<T>>> input_tensors;
  std::map<std::string, std::shared_ptr<Tensor<T>>> output_tensors;
};

/**
 * RuntimeGraph<T> — owns all operators and manages graph execution.
 *
 * Pipeline:
 *   1. AddOperator() — register operators with their types and connections
 *   2. Build() — create connections, instantiate layers via factory
 *   3. ReverseTopoSort() — determine execution order via DFS post-order
 *   4. Forward() — execute operators in topo order
 */
template <typename T>
class RuntimeGraph {
 public:
  /**
   * Add an operator to the graph.
   *
   * @param name         Unique operator name
   * @param type         Layer type string (e.g. "nn.ReLU", "nn.Conv2d")
   * @param input_names  Names of upstream operators feeding into this one
   * @param output_names Names assigned to this operator's outputs
   */
  void AddOperator(const std::string& name,
                   const std::string& type,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names) {
    auto op = std::make_shared<RuntimeOperator<T>>();
    op->name = name;
    op->type = type;
    op->input_names = input_names;
    op->output_names = output_names;
    operators_.push_back(op);
    op_map_[name] = op;
  }

  /**
   * Build the graph: create layer instances via factory.
   *
   * For each operator (except Input/Output), look up the type string
   * in the factory registry and create a concrete layer.
   */
  void Build() {
    for (auto& op : operators_) {
      if (op->type == "Input" || op->type == "Output") {
        continue;  // No layer for Input/Output sentinels
      }
      try {
        op->layer.reset(LayerRegisterer<T>::CreateLayer(op->type));
      } catch (const std::exception& e) {
        std::cerr << "Error creating layer for \"" << op->name
                  << "\" (type=\"" << op->type << "\"): " << e.what() << "\n";
      }
    }
    std::cout << "  Build: created layer instances for " << operators_.size()
              << " operators.\n";
  }

  /**
   * Reverse topological sort via DFS post-order traversal.
   *
   * Algorithm:
   *   1. Reset all operators' has_forward and start_time
   *   2. For each operator with no inputs (entry points), start DFS
   *   3. DFS visits all downstream operators recursively
   *   4. After visiting all children, assign start_time = ++counter
   *
   * Post-order ensures: an operator's start_time is always greater than
   * all its inputs' start_times, so ascending order = valid execution order.
   */
  void ReverseTopoSort() {
    // Reset state
    for (auto& op : operators_) {
      op->has_forward = false;
      op->start_time = -1;
    }

    int counter = 0;

    // Build adjacency: operator -> list of downstream operators
    std::map<std::string, std::vector<std::string>> adjacency;
    std::set<std::string> has_inputs;

    for (auto& op : operators_) {
      for (const auto& in_name : op->input_names) {
        adjacency[in_name].push_back(op->name);
        has_inputs.insert(op->name);
      }
    }

    // DFS post-order from entry points (operators with no inputs)
    std::function<void(const std::string&)> dfs =
        [this, &dfs, &adjacency, &counter](const std::string& name) {
      auto op = op_map_[name];
      if (op->has_forward) return;
      op->has_forward = true;

      // Visit all downstream operators
      if (adjacency.count(name)) {
        for (const auto& next_name : adjacency[name]) {
          dfs(next_name);
        }
      }

      // Post-order: assign time after all children are visited
      op->start_time = ++counter;
    };

    // Start DFS from entry points
    for (auto& op : operators_) {
      if (has_inputs.find(op->name) == has_inputs.end()) {
        dfs(op->name);
      }
    }

    // Also visit any disconnected operators
    for (auto& op : operators_) {
      if (!op->has_forward) {
        dfs(op->name);
      }
    }

    std::cout << "  TopoSort: assigned execution order for "
              << operators_.size() << " operators.\n";
  }

  /**
   * Print the execution order after topo sort.
   */
  void PrintExecutionOrder() const {
    std::cout << "\n  Execution Order (after topological sort):\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    // Sort operators by start_time
    std::vector<std::shared_ptr<RuntimeOperator<T>>> sorted = operators_;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                return a->start_time < b->start_time;
              });

    for (const auto& op : sorted) {
      std::cout << "  [" << std::setw(3) << op->start_time << "] "
                << std::setw(20) << op->name << "  (type=" << op->type << ")\n";
    }
    std::cout << "  " << std::string(60, '-') << "\n\n";
  }

  /**
   * Set an input tensor for the graph.
   *
   * @param input_name  Name of the input (matches an operator's output_name)
   * @param tensor      The input tensor
   */
  void SetInput(const std::string& input_name,
                const std::shared_ptr<Tensor<T>>& tensor) {
    input_tensors_[input_name] = tensor;
  }

  /**
   * Forward pass: execute all operators in topo order.
   *
   * Data flows through the graph via shared_ptr<Tensor> — zero copy.
   * Each operator reads its inputs from input_tensors, computes,
   * and writes results to output_tensors.
   */
  void Forward() {
    // Sort by execution order
    std::vector<std::shared_ptr<RuntimeOperator<T>>> sorted = operators_;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                return a->start_time < b->start_time;
              });

    std::cout << "  Forward pass:\n";
    for (const auto& op : sorted) {
      if (op->type == "Input") {
        // Input node: copy external inputs to global output map
        for (const auto& out_name : op->output_names) {
          if (input_tensors_.count(out_name)) {
            auto t = input_tensors_[out_name];
            op->output_tensors[out_name] = t;
            all_outputs_[out_name] = t;
            auto s = t->shapes();
            std::cout << "    [Input] " << op->name << " -> " << out_name
                      << " shape=["
                      << (s.size() > 0 ? std::to_string(s[0]) : "0") << ", "
                      << (s.size() > 1 ? std::to_string(s[1]) : "0") << ", "
                      << (s.size() > 2 ? std::to_string(s[2]) : "0")
                      << "]\n";
          }
        }
        continue;
      }

      if (op->type == "Output") {
        // Output node: collect final results
        continue;
      }

      // Collect inputs from upstream operators' outputs (all_outputs_)
      std::vector<std::shared_ptr<Tensor<T>>> inputs;
      for (const auto& in_name : op->input_names) {
        if (all_outputs_.count(in_name)) {
          inputs.push_back(all_outputs_[in_name]);
        }
      }

      if (inputs.empty()) {
        std::cerr << "    [WARN] " << op->name << " has no inputs!\n";
        continue;
      }

      // Execute layer
      std::vector<std::shared_ptr<Tensor<T>>> outputs;
      op->layer->Forward(inputs, outputs);

      // Propagate outputs to downstream (store in global map)
      for (size_t i = 0; i < op->output_names.size() && i < outputs.size(); i++) {
        const auto& out_name = op->output_names[i];
        op->output_tensors[out_name] = outputs[i];
        all_outputs_[out_name] = outputs[i];

        auto s = outputs[i]->shapes();
        std::cout << "    [" << op->name << "] " << out_name
                  << " shape=["
                  << (s.size() > 0 ? std::to_string(s[0]) : "0") << ", "
                  << (s.size() > 1 ? std::to_string(s[1]) : "0") << ", "
                  << (s.size() > 2 ? std::to_string(s[2]) : "0")
                  << "]\n";
      }
    }
    std::cout << "\n";
  }

  /**
   * Get an output tensor by name.
   */
  std::shared_ptr<Tensor<T>> GetOutput(const std::string& name) const {
    auto it = all_outputs_.find(name);
    if (it != all_outputs_.end()) {
      return it->second;
    }
    return nullptr;
  }

  /**
   * Get all operators (useful for inspection).
   */
  const std::vector<std::shared_ptr<RuntimeOperator<T>>>& operators() const {
    return operators_;
  }

 private:
  std::vector<std::shared_ptr<RuntimeOperator<T>>> operators_;
  std::map<std::string, std::shared_ptr<RuntimeOperator<T>>> op_map_;
  std::map<std::string, std::shared_ptr<Tensor<T>>> input_tensors_;
  std::map<std::string, std::shared_ptr<Tensor<T>>> all_outputs_;
};

}  // namespace learn_infer

#endif  // RUNTIME_GRAPH_HPP
