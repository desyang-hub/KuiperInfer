/**
 * day3/main.cpp
 *
 * Day 3 Demo: Tensor utility functions
 *
 * Demonstrates:
 *   - TensorElementAdd
 *   - TensorElementMultiply
 *   - TensorBroadcast  (broadcast [C,1,1] to [C,H,W])
 *   - TensorIsSame
 *   - TensorClone
 *   - TensorCreate
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day3
 */

#include "include/tensor_util.hpp"
#include <iostream>
#include <cmath>

using namespace learn_infer;
using namespace learn_infer::util;

/* ------------------------------------------------------------------ */
/* 1. Element-wise Addition                                           */
/* ------------------------------------------------------------------ */
void demo_element_add() {
  std::cout << "========== 1. Element-wise Add ==========\n\n";

  auto a = FTensor::Create(
      1, 3, 3,
      {1, 2, 3,
       4, 5, 6,
       7, 8, 9});

  auto b = FTensor::Create(
      1, 3, 3,
      {9, 8, 7,
       6, 5, 4,
       3, 2, 1});

  std::cout << "A:\n";
  a->print("A");
  std::cout << "B:\n";
  b->print("B");

  auto c = TensorElementAdd(a, b);
  std::cout << "A + B (every element should be 10):\n";
  c->print("A+B");

  // Multi-channel add
  auto mc_a = FTensor::Create(
      2, 2, 2,
      {1, 2, 3, 4,
       5, 6, 7, 8});
  auto mc_b = FTensor::Create(
      2, 2, 2,
      {10, 20, 30, 40,
       50, 60, 70, 80});

  auto mc_c = TensorElementAdd(mc_a, mc_b);
  std::cout << "Multi-channel [2,2,2] add:\n";
  mc_c->print("mc_a+mc_b");
}

/* ------------------------------------------------------------------ */
/* 2. Element-wise Multiply                                           */
/* ------------------------------------------------------------------ */
void demo_element_multiply() {
  std::cout << "========== 2. Element-wise Multiply ==========\n\n";

  auto a = FTensor::Create(
      1, 3, 3,
      {1, 2, 3,
       4, 5, 6,
       7, 8, 9});

  // Multiplier: all 2s
  auto b = FTensor::Create(
      1, 3, 3,
      {2, 2, 2,
       2, 2, 2,
       2, 2, 2});

  std::cout << "A:\n";
  a->print("A");
  std::cout << "B (all 2s):\n";
  b->print("B");

  auto c = TensorElementMultiply(a, b);
  std::cout << "A * B (each element doubled):\n";
  c->print("A*B");

  // Multiply with different values
  auto d = FTensor::Create(
      1, 1, 4,
      {1, 2, 3, 4});
  auto e = FTensor::Create(
      1, 1, 4,
      {1, 1, 1, 1});
  auto f = TensorElementMultiply(d, e);
  std::cout << "[1,2,3,4] * [1,1,1,1]:\n";
  f->print("d*e");
}

/* ------------------------------------------------------------------ */
/* 3. Broadcast [C,1,1] to [C,H,W]                                   */
/* ------------------------------------------------------------------ */
void demo_broadcast() {
  std::cout << "========== 3. Broadcast [C,1,1] -> [C,H,W] ==========\n\n";

  // Per-channel scale factors: [3, 1, 1]
  auto scale = FTensor::Create(
      3, 1, 1,
      {1.0f, 2.0f, 3.0f});

  // Target shape: [3, 3, 3]
  auto target = FTensor::Create(
      3, 3, 3,
      {0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0});

  std::cout << "scale [3,1,1]:\n";
  scale->print("scale");
  std::cout << "Broadcast scale -> [3,3,3]:\n";

  auto broadcasted = TensorBroadcast(scale, target);
  broadcasted->print("broadcasted");
  std::cout << "  Channel 0 should be all 1.0000\n";
  std::cout << "  Channel 1 should be all 2.0000\n";
  std::cout << "  Channel 2 should be all 3.0000\n";

  // Realistic use case: batch norm gamma/beta broadcast then element-wise
  std::cout << "\n--- Realistic use case: per-channel scale + shift ---\n";
  auto gamma = FTensor::Create(
      2, 1, 1,
      {1.5f, 0.5f});
  auto features = FTensor::Create(
      2, 2, 3,
      {1, 2, 3,
       4, 5, 6,
       7, 8, 9,
       10, 11, 12});

  auto gamma_broadcast = TensorBroadcast(gamma, features);
  auto scaled = TensorElementMultiply(features, gamma_broadcast);

  std::cout << "features [2,2,3]:\n";
  features->print("features");
  std::cout << "gamma [2,1,1]:\n";
  gamma->print("gamma");
  std::cout << "features * broadcasted_gamma:\n";
  scaled->print("scaled");
  std::cout << "  Channel 0 multiplied by 1.5, channel 1 by 0.5\n";
}

/* ------------------------------------------------------------------ */
/* 4. TensorIsSame                                                    */
/* ------------------------------------------------------------------ */
void demo_is_same() {
  std::cout << "========== 4. TensorIsSame ==========\n\n";

  auto a = FTensor::Create(1, 2, 3, {1, 2, 3, 4, 5, 6});
  auto b = FTensor::Create(1, 2, 3, {1, 2, 3, 4, 5, 6});
  auto c = FTensor::Create(1, 2, 3, {1, 2, 3, 4, 5, 7});

  std::cout << "a vs b (identical): "
            << (TensorIsSame(a, b) ? "SAME" : "DIFFERENT") << "\n";
  std::cout << "a vs c (last elt differs): "
            << (TensorIsSame(a, c) ? "SAME" : "DIFFERENT") << "\n";

  // Floating-point tolerance
  auto fp1 = FTensor::Create(1, 1, 4, {1.0f, 2.0f, 3.0f, 4.0f});
  auto fp2 = FTensor::Create(1, 1, 4, {1.0000001f, 2.0f, 3.0f, 4.0f});

  std::cout << "fp1 vs fp2 (tiny diff, default threshold 1e-6): "
            << (TensorIsSame(fp1, fp2) ? "SAME" : "DIFFERENT") << "\n";
  std::cout << "fp1 vs fp2 (threshold=1e-8): "
            << (TensorIsSame(fp1, fp2, 1e-8f) ? "SAME" : "DIFFERENT") << "\n";
}

/* ------------------------------------------------------------------ */
/* 5. TensorClone                                                     */
/* ------------------------------------------------------------------ */
void demo_clone() {
  std::cout << "========== 5. TensorClone ==========\n\n";

  auto original = FTensor::Create(2, 2, 3,
                                   {1, 2, 3,
                                    4, 5, 6,
                                    7, 8, 9,
                                    10, 11, 12});

  auto cloned = TensorClone(original);

  std::cout << "Original:\n";
  original->print("original");
  std::cout << "Cloned:\n";
  cloned->print("cloned");

  // Modify clone, verify original is unchanged
  cloned->fill(0.0f);
  std::cout << "After cloned.fill(0), original should be unchanged:\n";
  original->print("original");

  std::cout << "IsSame(original, cloned) after fill: "
            << (TensorIsSame(original, cloned) ? "SAME" : "DIFFERENT") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 6. TensorCreate from shapes vector                                 */
/* ------------------------------------------------------------------ */
void demo_create() {
  std::cout << "========== 6. TensorCreate(shapes, values) ==========\n\n";

  // 1-D shape
  auto t1 = TensorCreate<float>(
      {4}, {10, 20, 30, 40});
  std::cout << "Create shape=[4]:\n";
  t1->print("t1");

  // 2-D shape
  auto t2 = TensorCreate<float>(
      {2, 3}, {1, 2, 3, 4, 5, 6});
  std::cout << "Create shape=[2,3]:\n";
  t2->print("t2");

  // 3-D shape
  auto t3 = TensorCreate<float>(
      {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  std::cout << "Create shape=[2,2,2]:\n";
  t3->print("t3");
}

/* ------------------------------------------------------------------ */
/* 7. Full pipeline: ResNet-style block simulation                    */
/* ------------------------------------------------------------------ */
void demo_pipeline() {
  std::cout << "========== 7. Mini Pipeline (ResNet-style block) ==========\n\n";

  // Input: [2, 3, 3] -- 2 channels, 3x3 spatial
  auto input = TensorCreate<float>(
      {2, 3, 3},
      {1, 2, 3, 4, 5, 6, 7, 8, 9,
       9, 8, 7, 6, 5, 4, 3, 2, 1});

  std::cout << "Step 1 - Input [2,3,3]:\n";
  input->print("input");

  // "Activation": ReLU (clip negatives to 0) -- all positive here
  auto after_relu = TensorClone(input);
  after_relu->transform([](float x) { return std::max(x, 0.0f); });
  std::cout << "Step 2 - After ReLU (no change since all positive):\n";
  after_relu->print("relu_out");

  // Per-channel gamma/beta (like batch norm affine)
  auto gamma = FTensor::Create(2, 1, 1, {2.0f, 0.5f});
  auto beta  = FTensor::Create(2, 1, 1, {1.0f, -1.0f});

  // affine = input * gamma + beta
  auto gamma_bc = TensorBroadcast(gamma, after_relu);
  auto scaled   = TensorElementMultiply(after_relu, gamma_bc);
  auto beta_bc  = TensorBroadcast(beta, after_relu);
  auto affine   = TensorElementAdd(scaled, beta_bc);

  std::cout << "Step 3 - After affine (x*gamma + beta):\n";
  std::cout << "  gamma=[2.0, 0.5], beta=[1.0, -1.0]\n";
  affine->print("affine");

  // Residual connection: output = affine + input
  auto output = TensorElementAdd(affine, input);
  std::cout << "Step 4 - Residual: output = affine + input:\n";
  output->print("output");

  // Verify with direct computation for channel 0, element [0,0]:
  // input[0,0,0]=1 -> relu=1 -> affine = 1*2.0+1.0=3.0 -> output = 3.0+1.0=4.0
  std::cout << "  Verification: output[0,0,0] should be 4.0: "
            << output->at(0, 0, 0) << "\n";
  // channel 1, element [0,0]:
  // input[1,0,0]=9 -> relu=9 -> affine = 9*0.5+(-1.0)=3.5 -> output = 3.5+9=12.5
  std::cout << "  Verification: output[1,0,0] should be 12.5: "
            << output->at(1, 0, 0) << "\n";
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 3: Tensor Utility Functions\n";
  std::cout << "========================================\n\n";

  demo_element_add();
  std::cout << "\n";
  demo_element_multiply();
  std::cout << "\n";
  demo_broadcast();
  std::cout << "\n";
  demo_is_same();
  std::cout << "\n";
  demo_clone();
  std::cout << "\n";
  demo_create();
  std::cout << "\n";
  demo_pipeline();

  std::cout << "========================================\n";
  std::cout << "  Day 3 Complete!\n";
  std::cout << "  Learned: ElementAdd, ElementMul,\n";
  std::cout << "           Broadcast, IsSame, Clone,\n";
  std::cout << "           TensorCreate\n";
  std::cout << "========================================\n";

  return 0;
}
