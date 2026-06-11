/**
 * day5/main.cpp
 *
 * Day 5 Demo: Activation Layers
 *
 * Demonstrates 6 activation layers registered through the factory:
 *   - nn.ReLU       : max(0, x)
 *   - nn.Sigmoid    : 1 / (1 + exp(-x))
 *   - nn.ReLU6      : min(max(0, x), 6)
 *   - nn.SiLU       : x * sigmoid(x)
 *   - nn.HardSigmoid: piecewise-linear sigmoid approx
 *   - nn.HardSwish  : x * hard_sigmoid(x)
 *
 * Each layer is created via the factory, run with test inputs,
 * and compared against expected values.
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day5
 */

#include "include/layer/relu.hpp"
#include "include/layer/sigmoid.hpp"
#include "include/layer/relu6.hpp"
#include "include/layer/silu.hpp"
#include "include/layer/hardsigmoid.hpp"
#include "include/layer/hardswish.hpp"
#include "include/tensor_util.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace learn_infer;
using namespace learn_infer::util;

/* ------------------------------------------------------------------ */
/* Helper: run a layer by name and compare against expected values    */
/* ------------------------------------------------------------------ */
void test_layer(const std::string& layer_type,
                const std::vector<float>& input_vals,
                const std::vector<float>& expected_vals,
                uint32_t channels, uint32_t rows, uint32_t cols) {
  std::cout << "Testing: " << layer_type << "\n";

  Layer<float>* layer = LayerRegisterer<float>::CreateLayer(layer_type);
  std::cout << "  Factory created: " << layer->Type() << "\n";

  auto input = Tensor<float>::Create(channels, rows, cols, input_vals);
  auto expected = Tensor<float>::Create(channels, rows, cols, expected_vals);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;

  StatusCode status = layer->Forward(inputs, outputs);
  assert(status == StatusCode::kSuccess);
  assert(outputs.size() == 1);

  bool match = TensorIsSame(outputs[0], expected, 1e-4f);
  std::cout << "  Result: " << (match ? "PASS" : "FAIL") << "\n";

  if (!match) {
    std::cout << "  Output:\n";
    outputs[0]->print("    actual");
    std::cout << "  Expected:\n";
    expected->print("    expected");
  }

  delete layer;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 1. List all registered layers                                      */
/* ------------------------------------------------------------------ */
void demo_list_all() {
  std::cout << "========== 1. All Registered Activation Layers ==========\n\n";
  auto types = LayerRegisterer<float>::ListTypes();
  std::cout << "Count: " << types.size() << "\n";
  for (const auto& t : types) {
    std::cout << "  - " << t << "\n";
  }
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 2. Test ReLU                                                       */
/* ------------------------------------------------------------------ */
void demo_relu() {
  std::cout << "========== 2. nn.ReLU ==========\n\n";

  // Input: [-2, -1, 0, 1, 2]
  // Expected: [0, 0, 0, 1, 2]
  test_layer("nn.ReLU",
             {-2, -1, 0, 1, 2},
             { 0,  0, 0, 1, 2},
             1, 1, 5);
}

/* ------------------------------------------------------------------ */
/* 3. Test Sigmoid                                                    */
/* ------------------------------------------------------------------ */
void demo_sigmoid() {
  std::cout << "========== 3. nn.Sigmoid ==========\n\n";

  // sigmoid(0) = 0.5
  // sigmoid(1) = 0.731059
  // sigmoid(-1) = 0.268941
  // sigmoid(2) = 0.880797
  // sigmoid(-2) = 0.119203
  test_layer("nn.Sigmoid",
             { 0,  1, -1,  2, -2},
             {0.5f, 0.731059f, 0.268941f, 0.880797f, 0.119203f},
             1, 1, 5);

  // Edge cases: large positive and negative
  std::cout << "  Edge case: sigmoid(10) should be ~1.0\n";
  Layer<float>* layer = LayerRegisterer<float>::CreateLayer("nn.Sigmoid");
  auto input = Tensor<float>::Create(1, 1, 2, {10.0f, -10.0f});
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  layer->Forward(inputs, outputs);
  std::cout << "    sigmoid(10)  = " << outputs[0]->index(0) << " (expected ~1.0)\n";
  std::cout << "    sigmoid(-10) = " << outputs[0]->index(1) << " (expected ~0.0)\n";
  delete layer;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 4. Test ReLU6                                                      */
/* ------------------------------------------------------------------ */
void demo_relu6() {
  std::cout << "========== 4. nn.ReLU6 ==========\n\n";

  // Input: [-3, -1, 0, 3, 6, 9]
  // Expected: [0, 0, 0, 3, 6, 6]
  test_layer("nn.ReLU6",
             {-3, -1, 0, 3, 6, 9},
             { 0,  0, 0, 3, 6, 6},
             1, 1, 6);
}

/* ------------------------------------------------------------------ */
/* 5. Test SiLU                                                       */
/* ------------------------------------------------------------------ */
void demo_silu() {
  std::cout << "========== 5. nn.SiLU ==========\n\n";

  // SiLU(x) = x / (1 + exp(-x))
  // SiLU(0)  = 0
  // SiLU(1)  = 1 / (1 + exp(-1)) = 0.731059
  // SiLU(-1) = -1 / (1 + exp(1)) = -0.268941
  // SiLU(2)  = 2 / (1 + exp(-2)) = 1.761594
  // SiLU(-2) = -2 / (1 + exp(2)) = -0.238406
  test_layer("nn.SiLU",
             { 0,  1, -1,  2, -2},
             {0.0f, 0.731059f, -0.268941f, 1.761594f, -0.238406f},
             1, 1, 5);
}

/* ------------------------------------------------------------------ */
/* 6. Test HardSigmoid                                                */
/* ------------------------------------------------------------------ */
void demo_hardsigmoid() {
  std::cout << "========== 6. nn.HardSigmoid ==========\n\n";

  // HardSigmoid: 0 if x<=-2.5, 1 if x>=2.5, x/6+0.5 otherwise
  // x=-3:   0
  // x=-2.5: 0
  // x=0:    0/6+0.5 = 0.5
  // x=2.5:  1
  // x=3:    1
  test_layer("nn.HardSigmoid",
             {-3, -2.5f, 0, 2.5f, 3},
             { 0,    0,  0.5f,    1, 1},
             1, 1, 5);

  // Intermediate value: x=1.2 -> 1.2/6 + 0.5 = 0.7
  std::cout << "  Intermediate: hardsigmoid(1.2) should be 0.7\n";
  Layer<float>* layer = LayerRegisterer<float>::CreateLayer("nn.HardSigmoid");
  auto input = Tensor<float>::Create(1, 1, 1, {1.2f});
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  layer->Forward(inputs, outputs);
  std::cout << "    hardsigmoid(1.2) = " << outputs[0]->index(0)
            << " (expected 0.7)\n";
  delete layer;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 7. Test HardSwish                                                  */
/* ------------------------------------------------------------------ */
void demo_hardswish() {
  std::cout << "========== 7. nn.HardSwish ==========\n\n";

  // HardSwish: x * hard_sigmoid(x) where hard_sigmoid uses [-3,3] range
  // x=-4: -4 * 0 = 0
  // x=-3: -3 * 0 = 0
  // x=0:  0 * 0.5 = 0
  // x=3:  3 * 1 = 3
  // x=4:  4 * 1 = 4
  test_layer("nn.HardSwish",
             {-4, -3, 0, 3, 4},
             { 0,  0, 0, 3, 4},
             1, 1, 5);

  // Intermediate: x=0 -> (0+3)/6 = 0.5, output = 0 * 0.5 = 0
  // x=1 -> (1+3)/6 = 0.6667, output = 1 * 0.6667 = 0.6667
  std::cout << "  Intermediate: hardswish(1) should be ~0.6667\n";
  Layer<float>* layer = LayerRegisterer<float>::CreateLayer("nn.HardSwish");
  auto input = Tensor<float>::Create(1, 1, 1, {1.0f});
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  layer->Forward(inputs, outputs);
  std::cout << "    hardswish(1) = " << outputs[0]->index(0)
            << " (expected ~0.6667)\n";
  delete layer;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 8. Compare SiLU vs HardSwish (approximation quality)               */
/* ------------------------------------------------------------------ */
void demo_compare_silu_hardswish() {
  std::cout << "========== 8. SiLU vs HardSwish Comparison ==========\n\n";

  std::vector<float> test_vals = {-4, -3, -2, -1, 0, 1, 2, 3, 4};

  Layer<float>* silu = LayerRegisterer<float>::CreateLayer("nn.SiLU");
  Layer<float>* hardswish = LayerRegisterer<float>::CreateLayer("nn.HardSwish");

  auto input = Tensor<float>::Create(1, 1, 9, test_vals);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;

  inputs.push_back(input->clone());
  silu->Forward(inputs, outputs);
  auto silu_out = outputs[0];

  inputs.clear();
  outputs.clear();
  inputs.push_back(input->clone());
  hardswish->Forward(inputs, outputs);
  auto hs_out = outputs[0];

  std::cout << "  x    | SiLU       | HardSwish  | diff\n";
  std::cout << "  -----+------------+------------+------\n";
  for (uint32_t i = 0; i < 9; i++) {
    float s = silu_out->index(i);
    float h = hs_out->index(i);
    float diff = std::abs(s - h);
    std::cout << "  " << std::setw(4) << test_vals[i]
              << " | " << std::setw(9) << std::fixed << std::setprecision(6) << s
              << " | " << std::setw(10) << std::fixed << std::setprecision(6) << h
              << " | " << std::setw(5) << std::fixed << std::setprecision(4) << diff
              << "\n";
  }

  delete silu;
  delete hardswish;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 9. Multi-channel activation                                        */
/* ------------------------------------------------------------------ */
void demo_multichannel() {
  std::cout << "========== 9. Multi-Channel Activation ==========\n\n";

  Layer<float>* sigmoid = LayerRegisterer<float>::CreateLayer("nn.Sigmoid");

  // 2 channels, each 1x3
  auto input = Tensor<float>::Create(2, 1, 3,
      { -2.0f, 0.0f, 2.0f,   // channel 0
        -1.0f, 1.0f, 3.0f }); // channel 1

  std::cout << "Input [2,1,3]:\n";
  input->print("input");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  sigmoid->Forward(inputs, outputs);

  std::cout << "After Sigmoid:\n";
  outputs[0]->print("sigmoid_out");

  // Verify channel 0, element 1 (x=0): sigmoid(0)=0.5
  std::cout << "  sigmoid(0) = " << outputs[0]->at(0, 0, 1)
            << " (expected 0.5)\n";
  std::cout << "  sigmoid(2) = " << outputs[0]->at(0, 0, 2)
            << " (expected 0.880797)\n";
  std::cout << "  sigmoid(3) = " << outputs[0]->at(1, 0, 2)
            << " (expected 0.952574)\n";

  delete sigmoid;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 5: Activation Layers\n";
  std::cout << "========================================\n\n";

  demo_list_all();
  demo_relu();
  demo_sigmoid();
  demo_relu6();
  demo_silu();
  demo_hardsigmoid();
  demo_hardswish();
  demo_compare_silu_hardswish();
  demo_multichannel();

  std::cout << "========================================\n";
  std::cout << "  Day 5 Complete!\n";
  std::cout << "  Learned: Sigmoid, ReLU6, SiLU,\n";
  std::cout << "           HardSigmoid, HardSwish\n";
  std::cout << "           (all via factory pattern)\n";
  std::cout << "========================================\n";

  return 0;
}
