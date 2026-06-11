/**
 * day10/main.cpp
 *
 * Day 10 Demo: Topological Sort for Computation Graphs
 *
 * Demonstrates:
 *   - Building a mini computation graph with RuntimeGraph
 *   - Adding operators (layers) with input/output connections
 *   - Reverse topological sort (DFS post-order) to determine execution order
 *   - Forward data propagation through the sorted graph
 *   - Multiple graph topologies: linear, branching, skip connections
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day10
 */

#include "include/layer/relu.hpp"
#include "include/layer/sigmoid.hpp"
#include "include/layer/relu6.hpp"
#include "include/layer/silu.hpp"
#include "include/layer/hardsigmoid.hpp"
#include "include/layer/hardswish.hpp"
#include "include/layer/softmax.hpp"
#include "include/layer/batchnorm2d.hpp"
#include "include/layer/convolution.hpp"
#include "include/layer/maxpooling.hpp"
#include "include/layer/adaptive_avgpooling.hpp"
#include "include/layer/linear.hpp"
#include "include/runtime/runtime_graph.hpp"
#include "include/tensor_util.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace learn_infer;
using namespace learn_infer::util;

/* ------------------------------------------------------------------ */
/* Helper: print a graph's structure before topo sort                  */
/* ------------------------------------------------------------------ */
template <typename T>
void print_graph_structure(const RuntimeGraph<T>& graph) {
  std::cout << "  Graph Structure:\n";
  std::cout << "  " << std::string(60, '-') << "\n";
  for (const auto& op : graph.operators()) {
    std::cout << "  " << std::setw(20) << op->name
              << "  type=" << std::setw(20) << op->type << "\n";
    std::cout << "       inputs:  ";
    for (const auto& n : op->input_names) std::cout << n << " ";
    std::cout << "\n";
    std::cout << "       outputs: ";
    for (const auto& n : op->output_names) std::cout << n << " ";
    std::cout << "\n";
  }
  std::cout << "  " << std::string(60, '-') << "\n\n";
}

/* ------------------------------------------------------------------ */
/* Demo 1: Simple linear chain — Input -> Conv -> ReLU -> Pool -> Out */
/* ------------------------------------------------------------------ */
void demo_linear_chain() {
  std::cout << "========== Demo 1: Linear Chain ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input ---> conv1 ---> relu1 ---> pool1 ---> output\n\n";

  // Register layers with the factory
  ConvolutionRegister<float>("nn.Conv2d",
      /*ch=*/1, /*rows=*/4, /*cols=*/4,
      /*kernel_h=*/3, /*kernel_w=*/3,
      /*stride_h=*/1, /*stride_w=*/1,
      /*pad_h=*/0, /*pad_w=*/0,
      /*groups=*/1,
      std::vector<float>(9, 1.0f),  // all-ones 3x3 kernel
      {0.0f});                       // zero bias
  MaxPoolingRegister<float>("nn.MaxPool2d", 2, 2, 2, 2, 0, 0);

  // Build the graph
  RuntimeGraph<float> graph;

  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("conv1", "nn.Conv2d", {"input_data"}, {"conv_out"});
  graph.AddOperator("relu1", "nn.ReLU", {"conv_out"}, {"relu_out"});
  graph.AddOperator("pool1", "nn.MaxPool2d", {"relu_out"}, {"pool_out"});
  graph.AddOperator("output", "Output", {"pool_out"}, {});

  print_graph_structure(graph);

  graph.Build();
  graph.ReverseTopoSort();
  graph.PrintExecutionOrder();

  // Run forward
  auto input_tensor = Tensor<float>::Create(1, 4, 4, {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f
  });
  graph.SetInput("input_data", input_tensor);
  graph.Forward();

  auto output = graph.GetOutput("pool_out");
  if (output) {
    std::cout << "  Final output shape: ["
              << output->shapes()[0] << ", "
              << output->shapes()[1] << ", "
              << output->shapes()[2] << "]\n";
    output->print("    output");
  }
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* Demo 2: Branching — two branches from one input                     */
/* ------------------------------------------------------------------ */
void demo_branching() {
  std::cout << "========== Demo 2: Branching Graph ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input --+---> relu ---> linear_a ---> output\n";
  std::cout << "            +---> sigmoid ---> linear_b ---> output\n\n";

  LinearRegister<float>("branch_a", 4, 2,
      {1.0f, 0.0f, -1.0f, 0.5f,
       0.0f, 1.0f, 0.5f, -1.0f},
      {0.0f, 0.0f});
  LinearRegister<float>("branch_b", 4, 2,
      {0.5f, 0.5f, 0.5f, 0.5f,
       -0.5f, -0.5f, -0.5f, -0.5f},
      {0.0f, 0.0f});

  RuntimeGraph<float> graph;

  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("relu_branch", "nn.ReLU", {"input_data"}, {"relu_out"});
  graph.AddOperator("sigmoid_branch", "nn.Sigmoid", {"input_data"}, {"sigmoid_out"});
  graph.AddOperator("linear_a", "branch_a", {"relu_out"}, {"fc_a_out"});
  graph.AddOperator("linear_b", "branch_b", {"sigmoid_out"}, {"fc_b_out"});
  graph.AddOperator("output", "Output", {"fc_a_out", "fc_b_out"}, {});

  print_graph_structure(graph);

  graph.Build();
  graph.ReverseTopoSort();
  graph.PrintExecutionOrder();

  // Verify: relu and sigmoid should both come before their respective fc layers
  auto ops = graph.operators();
  std::map<std::string, int> time_map;
  for (const auto& op : ops) {
    time_map[op->name] = op->start_time;
  }

  bool pass = true;
  pass &= time_map["relu_branch"] < time_map["linear_a"];
  pass &= time_map["sigmoid_branch"] < time_map["linear_b"];
  pass &= time_map["input"] < time_map["relu_branch"];
  pass &= time_map["input"] < time_map["sigmoid_branch"];

  std::cout << "  Topo order correctness: " << (pass ? "PASS" : "FAIL") << "\n\n";

  // Run forward
  auto input_tensor = Tensor<float>::Create(1, 1, 4, {
      -1.0f, 2.0f, -3.0f, 4.0f
  });
  graph.SetInput("input_data", input_tensor);
  graph.Forward();

  auto out1 = graph.GetOutput("fc_a_out");
  auto out2 = graph.GetOutput("fc_b_out");
  if (out1) out1->print("    branch_a_output");
  if (out2) out2->print("    branch_b_output");
}

/* ------------------------------------------------------------------ */
/* Demo 3: Skip connection (ResNet-style residual block)               */
/* ------------------------------------------------------------------ */
void demo_skip_connection() {
  std::cout << "========== Demo 3: Skip Connection (ResNet Block) ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input ---> conv1 ---> bn1 ---> relu ---> output\n";
  std::cout << "    input -------------------^^^^^^^^^^ (skip)\n\n";

  ConvolutionRegister<float>("skip_conv",
      /*ch=*/3, /*rows=*/3, /*cols=*/3,
      /*kernel_h=*/3, /*kernel_w=*/3,
      /*stride_h=*/1, /*stride_w=*/1,
      /*pad_h=*/1, /*pad_w=*/1,
      /*groups=*/1,
      std::vector<float>(27, 0.1f),  // 3*3*3 = 27 weights
      std::vector<float>(3, 0.0f));

  BatchNorm2DRegister<float>("skip_bn",
      3,
      {0.0f, 0.0f, 0.0f},   // mean
      {1.0f, 1.0f, 1.0f},   // var
      {1.0f, 1.0f, 1.0f},   // weight
      {0.0f, 0.0f, 0.0f});  // bias

  RuntimeGraph<float> graph;

  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("conv1", "skip_conv", {"input_data"}, {"conv_out"});
  graph.AddOperator("bn1", "skip_bn", {"conv_out"}, {"bn_out"});
  graph.AddOperator("relu1", "nn.ReLU", {"bn_out"}, {"relu_out"});
  graph.AddOperator("output", "Output", {"relu_out"}, {});

  print_graph_structure(graph);

  graph.Build();
  graph.ReverseTopoSort();
  graph.PrintExecutionOrder();

  // Verify topo order: input -> conv -> bn -> relu -> output
  auto ops = graph.operators();
  std::map<std::string, int> time_map;
  for (const auto& op : ops) {
    time_map[op->name] = op->start_time;
  }

  bool pass = true;
  pass &= time_map["input"] < time_map["conv1"];
  pass &= time_map["conv1"] < time_map["bn1"];
  pass &= time_map["bn1"] < time_map["relu1"];
  pass &= time_map["relu1"] < time_map["output"];

  std::cout << "  Topo order correctness (strict chain): "
            << (pass ? "PASS" : "FAIL") << "\n\n";

  // Run forward
  auto input_tensor = Tensor<float>::Create(3, 3, 3, {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      // channel 1
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f,
      0.7f, 0.8f, 0.9f,
      // channel 2
      -1.0f, -2.0f, -3.0f,
      -4.0f, -5.0f, -6.0f,
      -7.0f, -8.0f, -9.0f
  });
  graph.SetInput("input_data", input_tensor);
  graph.Forward();

  auto output = graph.GetOutput("relu_out");
  if (output) {
    std::cout << "  Residual block output shape: ["
              << output->shapes()[0] << ", "
              << output->shapes()[1] << ", "
              << output->shapes()[2] << "]\n";
    output->print("    relu_output");
  }
}

/* ------------------------------------------------------------------ */
/* Demo 4: DAG with multiple entry points                             */
/* ------------------------------------------------------------------ */
void demo_multi_entry() {
  std::cout << "========== Demo 4: Multiple Entry Points ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input_a ---> relu ---> linear_a ---> output\n";
  std::cout << "    input_b ---> sigmoid ---> linear_b ---> output\n\n";

  LinearRegister<float>("merge_a", 3, 2,
      {1.0f, -1.0f, 0.0f,
       0.0f, 1.0f, -1.0f},
      {0.0f, 0.0f});
  LinearRegister<float>("merge_b", 3, 2,
      {0.5f, 0.5f, 0.5f,
       -0.5f, -0.5f, -0.5f},
      {0.0f, 0.0f});

  RuntimeGraph<float> graph;

  graph.AddOperator("input_a", "Input", {}, {"data_a"});
  graph.AddOperator("input_b", "Input", {}, {"data_b"});
  graph.AddOperator("relu_a", "nn.ReLU", {"data_a"}, {"relu_a_out"});
  graph.AddOperator("sigmoid_b", "nn.Sigmoid", {"data_b"}, {"sigmoid_b_out"});
  graph.AddOperator("linear_a", "merge_a", {"relu_a_out"}, {"linear_a_out"});
  graph.AddOperator("linear_b", "merge_b", {"sigmoid_b_out"}, {"linear_b_out"});
  graph.AddOperator("output", "Output", {"linear_a_out", "linear_b_out"}, {});

  print_graph_structure(graph);

  graph.Build();
  graph.ReverseTopoSort();
  graph.PrintExecutionOrder();

  // Verify both entry points are properly ordered
  auto ops = graph.operators();
  std::map<std::string, int> time_map;
  for (const auto& op : ops) {
    time_map[op->name] = op->start_time;
  }

  bool pass = true;
  pass &= time_map["input_a"] < time_map["relu_a"];
  pass &= time_map["relu_a"] < time_map["linear_a"];
  pass &= time_map["input_b"] < time_map["sigmoid_b"];
  pass &= time_map["sigmoid_b"] < time_map["linear_b"];

  std::cout << "  Topo order correctness (dual entry): "
            << (pass ? "PASS" : "FAIL") << "\n\n";

  // Run forward
  graph.SetInput("data_a", Tensor<float>::Create(1, 1, 3, {-1.0f, 2.0f, -3.0f}));
  graph.SetInput("data_b", Tensor<float>::Create(1, 1, 3, {0.5f, -0.5f, 1.0f}));
  graph.Forward();

  auto out_a = graph.GetOutput("linear_a_out");
  auto out_b = graph.GetOutput("linear_b_out");
  if (out_a) out_a->print("    linear_a_output");
  if (out_b) out_b->print("    linear_b_output");
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 10: Topological Sort\n";
  std::cout << "  DFS Post-Order for Computation\n";
  std::cout << "  Graph Execution Ordering\n";
  std::cout << "========================================\n\n";

  demo_linear_chain();
  demo_branching();
  demo_skip_connection();
  demo_multi_entry();

  std::cout << "========================================\n";
  std::cout << "  Day 10 Complete!\n";
  std::cout << "  Learned:\n";
  std::cout << "    - Computation graph construction\n";
  std::cout << "    - DFS post-order topological sort\n";
  std::cout << "    - Execution order determination\n";
  std::cout << "    - Linear, branching, skip connection\n";
  std::cout << "      graph topologies\n";
  std::cout << "    - Zero-copy data propagation via\n";
  std::cout << "      shared_ptr<Tensor>\n";
  std::cout << "========================================\n";

  return 0;
}
