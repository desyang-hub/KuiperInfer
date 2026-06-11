/**
 * day6/main.cpp
 *
 * Day 6 Demo: Softmax and BatchNorm2D
 *
 * Demonstrates:
 *   - nn.Softmax  : numerically stable softmax over chunks of `dim` elements
 *   - nn.BatchNorm2d : inference-mode batch normalization with fused params
 *   - All activation layers from day5 (for registration count)
 *   - Multi-channel operations and numerical verification
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day6
 */

#include "include/layer/relu.hpp"
#include "include/layer/sigmoid.hpp"
#include "include/layer/relu6.hpp"
#include "include/layer/silu.hpp"
#include "include/layer/hardsigmoid.hpp"
#include "include/layer/hardswish.hpp"
#include "include/layer/softmax.hpp"
#include "include/layer/batchnorm2d.hpp"
#include "include/tensor_util.hpp"
#include <iostream>
#include <iomanip>
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
  std::cout << "========== 1. All Registered Layers ==========\n\n";
  auto types = LayerRegisterer<float>::ListTypes();
  std::cout << "Count: " << types.size() << "\n";
  for (const auto& t : types) {
    std::cout << "  - " << t << "\n";
  }
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 2. Softmax: basic 1D softmax over all elements (dim=0 means full)  */
/* ------------------------------------------------------------------ */
void demo_softmax_basic() {
  std::cout << "========== 2. nn.Softmax (1D basic) ==========\n\n";

  // softmax([1,2,3]) = exp([1,2,3]-3)/sum = exp([-2,-1,0])/(e^-2+e^-1+1)
  // = [0.090031, 0.244728, 0.665241]
  test_layer("nn.Softmax",
             {1.0f, 2.0f, 3.0f},
             {0.090031f, 0.244728f, 0.665241f},
             1, 1, 3);
}

/* ------------------------------------------------------------------ */
/* 3. Softmax: uniform input                                          */
/* ------------------------------------------------------------------ */
void demo_softmax_uniform() {
  std::cout << "========== 3. nn.Softmax (uniform input) ==========\n\n";

  Layer<float>* softmax = LayerRegisterer<float>::CreateLayer("nn.Softmax");
  auto input = Tensor<float>::Create(1, 1, 4, {3.0f, 3.0f, 3.0f, 3.0f});

  std::cout << "  Input: [3, 3, 3, 3]\n";

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  softmax->Forward(inputs, outputs);

  std::cout << "  Output: " << std::fixed << std::setprecision(4);
  for (uint32_t i = 0; i < 4; i++) {
    std::cout << outputs[0]->index(i) << " ";
  }
  std::cout << "(each should be 0.25)\n";

  bool pass = true;
  for (uint32_t i = 0; i < 4; i++) {
    if (std::abs(outputs[0]->index(i) - 0.25f) > 1e-4f) pass = false;
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";

  delete softmax;
}

/* ------------------------------------------------------------------ */
/* 4. Softmax: numerical stability with large values                  */
/* ------------------------------------------------------------------ */
void demo_softmax_stability() {
  std::cout << "========== 4. nn.Softmax (numerical stability) ==========\n\n";

  // [1000, 1001, 1002] -- without subtracting max, exp(1002) overflows
  SoftmaxLayer<float> softmax(0);
  auto input = Tensor<float>::Create(1, 1, 3, {1000.0f, 1001.0f, 1002.0f});

  std::cout << "  Input: [1000, 1001, 1002]\n";

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  softmax.Forward(inputs, outputs);

  std::cout << "  Output: [" << std::fixed << std::setprecision(6);
  std::cout << outputs[0]->index(0) << ", "
            << outputs[0]->index(1) << ", "
            << outputs[0]->index(2) << "]\n";
  std::cout << "  (expected: ~[0.090031, 0.244728, 0.665241])\n";

  // Verify all finite
  bool all_finite = true;
  for (uint64_t i = 0; i < outputs[0]->size(); i++) {
    if (std::isinf(outputs[0]->index(i)) || std::isnan(outputs[0]->index(i))) {
      all_finite = false;
    }
  }
  std::cout << "  All finite: " << (all_finite ? "PASS" : "FAIL") << "\n";

  // Verify values match expected (same as softmax([0,1,2]))
  bool match = TensorIsSame(outputs[0],
      Tensor<float>::Create(1, 1, 3,
          {0.090031f, 0.244728f, 0.665241f}), 1e-4f);
  std::cout << "  Result: " << (match ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 5. Softmax: batch processing                                       */
/* ------------------------------------------------------------------ */
void demo_softmax_batch() {
  std::cout << "========== 5. nn.Softmax (batch) ==========\n\n";

  SoftmaxLayer<float> softmax(0);

  // Batch of 3 tensors
  auto t1 = Tensor<float>::Create(1, 1, 3, {1.0f, 2.0f, 3.0f});
  auto t2 = Tensor<float>::Create(1, 1, 3, {0.0f, 0.0f, 0.0f});
  auto t3 = Tensor<float>::Create(1, 1, 3, {10.0f, 0.0f, -10.0f});

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(t1);
  inputs.push_back(t2);
  inputs.push_back(t3);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  softmax.Forward(inputs, outputs);

  std::cout << "  Batch size: " << inputs.size() << "\n";
  std::cout << "  Batch 1 [1,2,3]: " << std::fixed << std::setprecision(4);
  for (uint32_t i = 0; i < 3; i++) std::cout << outputs[0]->index(i) << " ";
  std::cout << "\n";

  std::cout << "  Batch 2 [0,0,0]: ";
  for (uint32_t i = 0; i < 3; i++) std::cout << outputs[1]->index(i) << " ";
  std::cout << "(each 0.3333)\n";

  std::cout << "  Batch 3 [10,0,-10]: ";
  for (uint32_t i = 0; i < 3; i++) std::cout << outputs[2]->index(i) << " ";
  std::cout << "(near [1,0,0])\n";

  // Verify batch 3: softmax([10,0,-10]) should be near [1,0,0]
  bool pass = outputs[2]->index(0) > 0.999f &&
              std::abs(outputs[2]->index(1)) < 0.001f &&
              std::abs(outputs[2]->index(2)) < 0.001f;
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 6. Softmax: sum=1 check on 10-element vector                       */
/* ------------------------------------------------------------------ */
void demo_softmax_sum() {
  std::cout << "========== 6. nn.Softmax (sum=1 check) ==========\n\n";

  Layer<float>* softmax = LayerRegisterer<float>::CreateLayer("nn.Softmax");
  auto input = Tensor<float>::Create(1, 1, 10,
      {3.0f, 1.0f, 5.0f, 2.0f, 4.0f,
       1.5f, 3.5f, 6.0f, 0.5f, 2.5f});

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  softmax->Forward(inputs, outputs);

  std::cout << "  Output (10 values): " << std::fixed << std::setprecision(4);
  for (uint32_t i = 0; i < 10; i++) {
    std::cout << outputs[0]->index(i) << " ";
  }
  std::cout << "\n";

  float sum = 0;
  for (uint32_t i = 0; i < 10; i++) sum += outputs[0]->index(i);
  std::cout << "  Sum: " << std::fixed << std::setprecision(6)
            << sum << " (expected 1.0)\n";
  std::cout << "  Result: " << (std::abs(sum - 1.0f) < 1e-4f ? "PASS" : "FAIL")
            << "\n\n";

  delete softmax;
}

/* ------------------------------------------------------------------ */
/* 7. BatchNorm2D: basic inference                                    */
/* ------------------------------------------------------------------ */
void demo_batchnorm_basic() {
  std::cout << "========== 7. nn.BatchNorm2d (basic) ==========\n\n";

  // 2-channel, 2x2 spatial
  auto input = Tensor<float>::Create(2, 2, 2,
      {1.0f, 2.0f, 3.0f, 4.0f,   // ch0
       0.0f, 1.0f, 2.0f, 3.0f}); // ch1

  // Params: mean=[2.5, 1.5], var=[1.25, 1.25]
  // weight=[2.0, 0.5], bias=[0.1, -0.2], eps=1e-5
  BatchNorm2DLayer<float> bn(2,
      {2.5f, 1.5f},   // mean
      {1.25f, 1.25f}, // var
      {2.0f, 0.5f},   // weight
      {0.1f, -0.2f},  // bias
      1e-5f);

  std::cout << "  Input [2,2,2]:\n";
  input->print("    input");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  bn.Forward(inputs, outputs);

  std::cout << "  Output [2,2,2]:\n";
  outputs[0]->print("    output");

  // Manual verification: ch0, [0,0]: x=1.0
  // inv_stddev = 2.0 / sqrt(1.25+1e-5) = 1.78886
  // offset = 0.1 - 1.78886 * 2.5 = -4.37214
  // y = 1.78886 * 1.0 + (-4.37214) = -2.58328
  // But the implementation does: y = inv_stddev * (x - mean) + weight * x + bias... no
  // Actually: y = weight/sqrt(var+eps) * x + (bias - weight*mean/sqrt(var+eps))
  float inv_std = 2.0f / std::sqrt(1.25f + 1e-5f);
  float off = 0.1f - inv_std * 2.5f;
  float expected = inv_std * 1.0f + off;
  std::cout << "  Verification: output[0,0,0] = " << std::fixed << std::setprecision(4)
            << outputs[0]->at(0, 0, 0) << " (expected " << expected << ")\n";
  std::cout << "  Result: "
            << (std::abs(outputs[0]->at(0, 0, 0) - expected) < 1e-4f ? "PASS" : "FAIL")
            << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 8. BatchNorm2D: batch processing                                   */
/* ------------------------------------------------------------------ */
void demo_batchnorm_batch() {
  std::cout << "========== 8. nn.BatchNorm2d (batch) ==========\n\n";

  BatchNorm2DLayer<float> bn(1,
      {5.0f},    // mean
      {1.0f},    // var
      {2.0f},    // weight
      {-0.0f});  // bias

  // Two feature maps
  auto f1 = Tensor<float>::Create(1, 1, 4, {4.0f, 5.0f, 6.0f, 7.0f});
  auto f2 = Tensor<float>::Create(1, 1, 4, {0.0f, 10.0f, 100.0f, -10.0f});

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(f1);
  inputs.push_back(f2);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  bn.Forward(inputs, outputs);

  // inv_stddev = 2.0 / sqrt(1.0+1e-5) = 2.0
  // offset = 0.0 - 2.0 * 5.0 = -10.0
  // y = 2.0 * x - 10.0
  std::cout << "  Batch 1 [4,5,6,7]: ";
  for (uint32_t i = 0; i < 4; i++)
    std::cout << std::fixed << std::setprecision(4) << outputs[0]->index(i) << " ";
  std::cout << "(expected: [-2.0, 0.0, 2.0, 4.0])\n";

  std::cout << "  Batch 2 [0,10,100,-10]: ";
  for (uint32_t i = 0; i < 4; i++)
    std::cout << std::fixed << std::setprecision(4) << outputs[1]->index(i) << " ";
  std::cout << "(expected: [-10.0, 10.0, 190.0, -30.0])\n";

  float expected0[4] = {-2.0f, 0.0f, 2.0f, 4.0f};
  float expected1[4] = {-10.0f, 10.0f, 190.0f, -30.0f};
  bool pass = true;
  for (uint32_t i = 0; i < 4; i++) {
    if (std::abs(outputs[0]->index(i) - expected0[i]) > 1e-3f) pass = false;
    if (std::abs(outputs[1]->index(i) - expected1[i]) > 1e-3f) pass = false;
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 9. BatchNorm2D: via factory registration                           */
/* ------------------------------------------------------------------ */
void demo_batchnorm_factory() {
  std::cout << "========== 9. nn.BatchNorm2d (factory) ==========\n\n";

  // Register a specific BatchNorm2D with the factory using identity params
  // mean=0, var=1, weight=1, bias=0 => output = 1/sqrt(1+eps) * x + 0
  BatchNorm2DRegister<float>(
      "nn.BatchNorm2d",   // type name
      1,                  // channels
      {0.0f},             // mean
      {1.0f},             // var
      {1.0f},             // weight
      {0.0f});            // bias

  // Now create via factory
  Layer<float>* bn = LayerRegisterer<float>::CreateLayer("nn.BatchNorm2d");
  std::cout << "  Factory created: " << bn->Type() << "\n";

  // With mean=0, var=1, weight=1, bias=0:
  // inv_stddev = 1/sqrt(1+eps) ~ 1.0 (slightly less due to eps)
  // offset = 0 - inv_stddev * 0 = 0
  // So output ~ input (very close to identity)
  auto input = Tensor<float>::Create(1, 1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  bn->Forward(inputs, outputs);

  // Compare with tolerance: output ~ input * 1/sqrt(1+eps) ~ input * 0.999995
  bool close = true;
  for (uint64_t i = 0; i < input->size(); i++) {
    if (std::abs(outputs[0]->index(i) - input->index(i)) > 1e-4f) {
      close = false;
    }
  }
  std::cout << "  Near-identity transform (tol=1e-4): " << (close ? "PASS" : "FAIL") << "\n";
  std::cout << "  Result: " << (close ? "PASS" : "FAIL") << "\n\n";

  delete bn;
}

/* ------------------------------------------------------------------ */
/* 10. Softmax: multi-channel (per-tensor softmax)                    */
/* ------------------------------------------------------------------ */
void demo_softmax_multichannel() {
  std::cout << "========== 10. nn.Softmax (multi-channel) ==========\n\n";

  Layer<float>* softmax = LayerRegisterer<float>::CreateLayer("nn.Softmax");

  auto input = Tensor<float>::Create(2, 1, 3,
      {1.0f, 2.0f, 3.0f,   // ch0
       4.0f, 5.0f, 6.0f}); // ch1

  std::cout << "  Input [2,1,3]:\n";
  input->print("    input");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  softmax->Forward(inputs, outputs);

  // Full tensor softmax (all 6 elements normalized together)
  std::cout << "  Softmax (full tensor, all 6 elements):\n";
  outputs[0]->print("    output");

  float sum = 0;
  for (uint64_t i = 0; i < outputs[0]->size(); i++) sum += outputs[0]->index(i);
  std::cout << "  Sum: " << std::fixed << std::setprecision(6)
            << sum << " (expected 1.0)\n";
  std::cout << "  Result: " << (std::abs(sum - 1.0f) < 1e-4f ? "PASS" : "FAIL")
            << "\n\n";

  delete softmax;
}

/* ------------------------------------------------------------------ */
/* 11. Softmax: explicit dim parameter (chunk softmax)                */
/* ------------------------------------------------------------------ */
void demo_softmax_dim() {
  std::cout << "========== 11. nn.Softmax (explicit dim) ==========\n\n";

  // dim=3: softmax over chunks of 3 elements
  // Input: [1,2,3, 4,5,6] -> softmax([1,2,3]), softmax([4,5,6])
  SoftmaxLayer<float> softmax(3);
  auto input = Tensor<float>::Create(1, 1, 6,
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  std::cout << "  Input [1,1,6]: [1, 2, 3, 4, 5, 6], dim=3\n";

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  softmax.Forward(inputs, outputs);

  std::cout << "  Output: ";
  for (uint32_t i = 0; i < 6; i++) {
    std::cout << std::fixed << std::setprecision(6)
              << outputs[0]->index(i) << " ";
  }
  std::cout << "\n";

  // Each group of 3 should sum to 1
  float sum0 = outputs[0]->index(0) + outputs[0]->index(1) + outputs[0]->index(2);
  float sum1 = outputs[0]->index(3) + outputs[0]->index(4) + outputs[0]->index(5);
  std::cout << "  Group 0 sum: " << std::fixed << std::setprecision(6)
            << sum0 << " (expected 1.0)\n";
  std::cout << "  Group 1 sum: " << std::fixed << std::setprecision(6)
            << sum1 << " (expected 1.0)\n";

  bool pass = std::abs(sum0 - 1.0f) < 1e-4f && std::abs(sum1 - 1.0f) < 1e-4f;
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 12. Re-test all Day 5 activations (still registered)               */
/* ------------------------------------------------------------------ */
void demo_day5_activations() {
  std::cout << "========== 12. Re-test Day 5 Activations ==========\n\n";

  // ReLU
  test_layer("nn.ReLU", {-2,-1,0,1,2}, {0,0,0,1,2}, 1,1,5);
  // Sigmoid
  test_layer("nn.Sigmoid", {0,1,-1,2,-2},
             {0.5f,0.731059f,0.268941f,0.880797f,0.119203f}, 1,1,5);
  // ReLU6
  test_layer("nn.ReLU6", {-3,-1,0,3,6,9}, {0,0,0,3,6,6}, 1,1,6);
  // SiLU
  test_layer("nn.SiLU", {0,1,-1,2,-2},
             {0.0f,0.731059f,-0.268941f,1.761594f,-0.238406f}, 1,1,5);
  // HardSigmoid
  test_layer("nn.HardSigmoid", {-3,-2.5f,0,2.5f,3},
             {0,0,0.5f,1,1}, 1,1,5);
  // HardSwish
  test_layer("nn.HardSwish", {-4,-3,0,3,4}, {0,0,0,3,4}, 1,1,5);
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 6: Softmax and BatchNorm2D\n";
  std::cout << "========================================\n\n";

  demo_list_all();
  demo_softmax_basic();
  demo_softmax_uniform();
  demo_softmax_stability();
  demo_softmax_batch();
  demo_softmax_sum();
  demo_batchnorm_basic();
  demo_batchnorm_batch();
  demo_batchnorm_factory();
  demo_softmax_multichannel();
  demo_softmax_dim();
  demo_day5_activations();

  std::cout << "========================================\n";
  std::cout << "  Day 6 Complete!\n";
  std::cout << "  Learned: Softmax (1D, batch,\n";
  std::cout << "           numerical stability, dim),\n";
  std::cout << "           BatchNorm2D (inference,\n";
  std::cout << "           fused params, batch)\n";
  std::cout << "========================================\n";

  return 0;
}
