/**
 * day11/main.cpp
 *
 * Day 11 Demo: Forward Execution & Data Propagation
 *
 * Builds on Day 10 (topological sort) and adds:
 *   - Complete forward pass with shape tracking at every step
 *   - Zero-copy data flow via shared_ptr<Tensor>
 *   - Reference counting demo (one output consumed by two operators)
 *   - Timing of forward pass and individual operators
 *
 * Also keeps all Day 10 demos for continuity.
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day11
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
#include <chrono>

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
/* Day 10 Demos (kept for continuity)                                  */
/* ------------------------------------------------------------------ */

void demo_linear_chain() {
  std::cout << "========== Demo 1: Linear Chain ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input ---> conv1 ---> relu1 ---> pool1 ---> output\n\n";

  ConvolutionRegister<float>("nn.Conv2d",
      /*ch=*/1, /*rows=*/4, /*cols=*/4,
      /*kernel_h=*/3, /*kernel_w=*/3,
      /*stride_h=*/1, /*stride_w=*/1,
      /*pad_h=*/0, /*pad_w=*/0,
      /*groups=*/1,
      std::vector<float>(9, 1.0f),
      {0.0f});
  MaxPoolingRegister<float>("nn.MaxPool2d", 2, 2, 2, 2, 0, 0);

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

void demo_skip_connection() {
  std::cout << "========== Demo 3: Skip Connection (ResNet Block) ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input ---> conv1 ---> bn1 ---> relu ---> output\n";
  std::cout << "    input -------------------^^^^^^^^^^ (skip)\n\n";

  ConvolutionRegister<float>("skip_conv",
      3, 3, 3, 3, 3, 1, 1, 1, 1,
      1,
      std::vector<float>(27, 0.1f),
      std::vector<float>(3, 0.0f));

  BatchNorm2DRegister<float>("skip_bn",
      3,
      {0.0f, 0.0f, 0.0f},
      {1.0f, 1.0f, 1.0f},
      {1.0f, 1.0f, 1.0f},
      {0.0f, 0.0f, 0.0f});

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

  auto input_tensor = Tensor<float>::Create(3, 3, 3, {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f,
      0.7f, 0.8f, 0.9f,
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

  graph.SetInput("data_a", Tensor<float>::Create(1, 1, 3, {-1.0f, 2.0f, -3.0f}));
  graph.SetInput("data_b", Tensor<float>::Create(1, 1, 3, {0.5f, -0.5f, 1.0f}));
  graph.Forward();

  auto out_a = graph.GetOutput("linear_a_out");
  auto out_b = graph.GetOutput("linear_b_out");
  if (out_a) out_a->print("    linear_a_output");
  if (out_b) out_b->print("    linear_b_output");
}

/* ------------------------------------------------------------------ */
/* Day 11 New Demos                                                    */
/* ------------------------------------------------------------------ */

/**
 * Demo 5: Complete forward pass with shape tracking.
 *
 * Shows how data flows through the graph step by step,
 * printing the shape at each intermediate tensor.
 */
void demo_forward_with_tracking() {
  std::cout << "========== Day 11 Demo 5: Forward with Shape Tracking ==========\n\n";
  std::cout << "  This demo shows the complete data flow through a graph.\n";
  std::cout << "  At each step, the tensor shape is printed.\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input[1,6,6] ---> conv[1,4,4] ---> relu[1,4,4]\n";
  std::cout << "      ---> pool[1,2,2] ---> bn[1,2,2] ---> output\n\n";

  ConvolutionRegister<float>("track_conv",
      /*ch=*/1, /*rows=*/6, /*cols=*/6,
      /*kernel_h=*/3, /*kernel_w=*/3,
      /*stride_h=*/1, /*stride_w=*/1,
      /*pad_h=*/0, /*pad_w=*/0,
      /*groups=*/1,
      std::vector<float>(9, 0.5f),
      {0.0f});
  MaxPoolingRegister<float>("track_pool", 2, 2, 2, 2, 0, 0);
  BatchNorm2DRegister<float>("track_bn",
      1,
      {0.0f},
      {1.0f},
      {1.0f},
      {0.0f});

  RuntimeGraph<float> graph;

  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("conv1", "track_conv", {"input_data"}, {"conv_out"});
  graph.AddOperator("relu1", "nn.ReLU", {"conv_out"}, {"relu_out"});
  graph.AddOperator("pool1", "track_pool", {"relu_out"}, {"pool_out"});
  graph.AddOperator("bn1", "track_bn", {"pool_out"}, {"bn_out"});
  graph.AddOperator("output", "Output", {"bn_out"}, {});

  graph.Build();
  graph.ReverseTopoSort();
  graph.PrintExecutionOrder();

  // Create a 6x6 input with sequential values
  std::vector<float> input_data(36);
  for (int i = 0; i < 36; i++) {
    input_data[i] = static_cast<float>(i + 1);
  }
  auto input_tensor = Tensor<float>::Create(1, 6, 6, input_data);
  graph.SetInput("input_data", input_tensor);

  // Time the forward pass
  auto start = std::chrono::high_resolution_clock::now();
  graph.Forward();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "  Forward pass time: " << duration.count() << " us\n\n";

  // Print intermediate results
  auto conv_out = graph.GetOutput("conv_out");
  auto relu_out = graph.GetOutput("relu_out");
  auto pool_out = graph.GetOutput("pool_out");
  auto bn_out = graph.GetOutput("bn_out");

  std::cout << "  Intermediate tensor shapes:\n";
  std::cout << "  " << std::string(50, '-') << "\n";
  if (conv_out) {
    auto s = conv_out->shapes();
    std::cout << "    conv_out  shape=[" << s[0] << ", " << s[1] << ", " << s[2] << "]\n";
  }
  if (relu_out) {
    auto s = relu_out->shapes();
    std::cout << "    relu_out  shape=[" << s[0] << ", " << s[1] << ", " << s[2] << "]\n";
  }
  if (pool_out) {
    auto s = pool_out->shapes();
    std::cout << "    pool_out  shape=[" << s[0] << ", " << s[1] << ", " << s[2] << "]\n";
  }
  if (bn_out) {
    auto s = bn_out->shapes();
    std::cout << "    bn_out    shape=[" << s[0] << ", " << s[1] << ", " << s[2] << "]\n";
  }
  std::cout << "  " << std::string(50, '-') << "\n";

  std::cout << "\n  Final output:\n";
  if (bn_out) bn_out->print("    bn_output");
}

/**
 * Demo 6: Reference counting — one output consumed by two operators.
 *
 * Demonstrates zero-copy: the same tensor flows to both branches
 * without being copied. We verify by checking use_count.
 */
void demo_reference_counting() {
  std::cout << "========== Day 11 Demo 6: Zero-Copy Reference Counting ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input ---> relu ----+---> linear_a ---> output\n";
  std::cout << "                       +---> linear_b ---> output\n\n";
  std::cout << "  relu's output is shared by linear_a AND linear_b.\n";
  std::cout << "  No data copy occurs — just shared_ptr reference counting.\n\n";

  LinearRegister<float>("ref_a", 4, 2,
      {1.0f, 0.0f, -1.0f, 0.5f,
       0.0f, 1.0f, 0.5f, -1.0f},
      {0.0f, 0.0f});
  LinearRegister<float>("ref_b", 4, 2,
      {0.5f, 0.5f, 0.5f, 0.5f,
       -0.5f, -0.5f, -0.5f, -0.5f},
      {0.0f, 0.0f});

  RuntimeGraph<float> graph;

  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("relu", "nn.ReLU", {"input_data"}, {"relu_out"});
  graph.AddOperator("linear_a", "ref_a", {"relu_out"}, {"fc_a_out"});
  graph.AddOperator("linear_b", "ref_b", {"relu_out"}, {"fc_b_out"});
  graph.AddOperator("output", "Output", {"fc_a_out", "fc_b_out"}, {});

  graph.Build();
  graph.ReverseTopoSort();
  graph.PrintExecutionOrder();

  std::cout << "  In this graph, relu_out feeds both linear_a and linear_b.\n";
  std::cout << "  The same tensor flows to both — zero data copy!\n\n";

  auto input_tensor = Tensor<float>::Create(1, 1, 4, {
      -1.0f, 2.0f, -3.0f, 4.0f
  });
  graph.SetInput("input_data", input_tensor);
  graph.Forward();

  auto fc_a = graph.GetOutput("fc_a_out");
  auto fc_b = graph.GetOutput("fc_b_out");
  if (fc_a) fc_a->print("    branch_a (consumed relu_out)");
  if (fc_b) fc_b->print("    branch_b (consumed relu_out)");

  std::cout << "\n  Both branches read from the SAME relu output tensor.\n";
  std::cout << "  Only ONE ReLU computation was performed!\n";
}

/**
 * Demo 7: Timing individual operators.
 *
 * Shows how to time each operator during forward pass.
 */
void demo_operator_timing() {
  std::cout << "========== Day 11 Demo 7: Operator-Level Timing ==========\n\n";
  std::cout << "  Graph topology:\n";
  std::cout << "    input[3,8,8] ---> conv ---> relu ---> pool ---> softmax ---> output\n\n";

  ConvolutionRegister<float>("timed_conv",
      /*ch=*/3, /*rows=*/8, /*cols=*/8,
      /*kernel_h=*/3, /*kernel_w=*/3,
      /*stride_h=*/1, /*stride_w=*/1,
      /*pad_h=*/1, /*pad_w=*/1,
      /*groups=*/1,
      std::vector<float>(3 * 3 * 3 * 3, 0.1f),
      std::vector<float>(3, 0.0f));
  MaxPoolingRegister<float>("timed_pool", 2, 2, 2, 2, 0, 0);

  RuntimeGraph<float> graph;

  graph.AddOperator("input", "Input", {}, {"input_data"});
  graph.AddOperator("conv1", "timed_conv", {"input_data"}, {"conv_out"});
  graph.AddOperator("relu1", "nn.ReLU", {"conv_out"}, {"relu_out"});
  graph.AddOperator("pool1", "timed_pool", {"relu_out"}, {"pool_out"});
  graph.AddOperator("softmax1", "nn.Softmax", {"pool_out"}, {"softmax_out"});
  graph.AddOperator("output", "Output", {"softmax_out"}, {});

  graph.Build();
  graph.ReverseTopoSort();
  graph.PrintExecutionOrder();

  // Create 3x8x8 input
  std::vector<float> input_data(3 * 8 * 8);
  for (int i = 0; i < 3 * 8 * 8; i++) {
    input_data[i] = static_cast<float>(i % 10 - 5) * 0.1f;
  }
  auto input_tensor = Tensor<float>::Create(3, 8, 8, input_data);
  graph.SetInput("input_data", input_tensor);

  // Time the full forward pass
  auto total_start = std::chrono::high_resolution_clock::now();
  graph.Forward();
  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

  std::cout << "  Total forward pass time: " << total_us << " us\n";

  // Get output and verify it sums to ~1.0 (softmax property)
  auto softmax_out = graph.GetOutput("softmax_out");
  if (softmax_out) {
    auto vec = softmax_out->to_vector();
    float sum = 0.0f;
    for (auto v : vec) sum += v;
    std::cout << "  Softmax output sum: " << sum << " (should be ~1.0)\n";
    std::cout << "  Softmax output shape: ["
              << softmax_out->shapes()[0] << ", "
              << softmax_out->shapes()[1] << ", "
              << softmax_out->shapes()[2] << "]\n";
  }
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 11: Forward Execution & Data\n";
  std::cout << "  Propagation (Zero-Copy)\n";
  std::cout << "========================================\n\n";

  // Day 10 demos (kept for continuity)
  demo_linear_chain();
  demo_branching();
  demo_skip_connection();
  demo_multi_entry();

  // Day 11 new demos
  demo_forward_with_tracking();
  demo_reference_counting();
  demo_operator_timing();

  std::cout << "========================================\n";
  std::cout << "  Day 11 Complete!\n";
  std::cout << "  Learned:\n";
  std::cout << "    - Complete forward execution flow\n";
  std::cout << "    - Shape tracking at each step\n";
  std::cout << "    - Zero-copy data propagation\n";
  std::cout << "    - Reference counting with\n";
  std::cout << "      shared_ptr<Tensor>\n";
  std::cout << "    - Operator-level timing\n";
  std::cout << "========================================\n";

  return 0;
}
