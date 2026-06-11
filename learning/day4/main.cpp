/**
 * day4/main.cpp
 *
 * Day 4 Demo: Layer system and Factory pattern
 *
 * Demonstrates:
 *   - Layer<T> abstract base class with virtual Forward() and Type()
 *   - LayerRegisterer factory with static registration map
 *   - LayerRegistererWrapper RAII auto-registration
 *   - ReluLayer as a concrete implementation
 *   - Creating layers by name through the factory
 *   - Running a forward pass through a chain of factory-created layers
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day4
 */

#include "include/layer/relu.hpp"
#include "include/tensor_util.hpp"
#include <iostream>
#include <cassert>

using namespace learn_infer;
using namespace learn_infer::util;

/* ------------------------------------------------------------------ */
/* 1. Demo: List registered layers                                    */
/* ------------------------------------------------------------------ */
void demo_registered_types() {
  std::cout << "========== 1. Registered Layer Types ==========\n\n";

  auto types = LayerRegisterer<float>::ListTypes();
  std::cout << "Registered layer count: " << types.size() << "\n";
  for (const auto& t : types) {
    std::cout << "  - " << t << "\n";
  }
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 2. Demo: Create layer via factory and run ReLU                     */
/* ------------------------------------------------------------------ */
void demo_relu_factory() {
  std::cout << "========== 2. Factory: Create and Run nn.ReLU ==========\n\n";

  // Create a layer by string name through the factory
  Layer<float>* layer = LayerRegisterer<float>::CreateLayer("nn.ReLU");
  std::cout << "Factory created layer type: " << layer->Type() << "\n\n";

  // Input tensor with positive and negative values
  auto input = FTensor::Create(
      1, 3, 3,
      {-3.0f, -1.0f,  2.0f,
       -2.0f,  0.0f,  4.0f,
        1.0f, -4.0f,  5.0f});

  std::cout << "Input (has negatives):\n";
  input->print("input");

  // Run forward pass
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;

  StatusCode status = layer->Forward(inputs, outputs);

  std::cout << "Forward status: "
            << (status == StatusCode::kSuccess ? "SUCCESS" : "FAILED") << "\n\n";

  assert(outputs.size() == 1);
  std::cout << "Output (negatives clipped to 0):\n";
  outputs[0]->print("relu_output");

  // Verify: [-3,-1,2; -2,0,4; 1,-4,5] -> [0,0,2; 0,0,4; 1,0,5]
  std::cout << "Verification:\n";
  std::cout << "  output[0,0,0] should be 0.0000: "
            << outputs[0]->at(0, 0, 0) << "\n";
  std::cout << "  output[0,0,2] should be 2.0000: "
            << outputs[0]->at(0, 0, 2) << "\n";
  std::cout << "  output[0,1,2] should be 4.0000: "
            << outputs[0]->at(0, 1, 2) << "\n";
  std::cout << "  output[0,2,2] should be 5.0000: "
            << outputs[0]->at(0, 2, 2) << "\n";

  delete layer;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 3. Demo: Multi-channel ReLU                                       */
/* ------------------------------------------------------------------ */
void demo_relu_multichannel() {
  std::cout << "========== 3. Multi-Channel ReLU ==========\n\n";

  Layer<float>* relu = LayerRegisterer<float>::CreateLayer("nn.ReLU");

  // 2-channel input with mixed signs
  auto input = FTensor::Create(
      2, 2, 3,
      {-1,  2, -3,
        4, -5,  6,
        7, -8,  9,
       -10, 11, -12});

  std::cout << "Input [2,2,3] with mixed signs:\n";
  input->print("input");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;

  relu->Forward(inputs, outputs);
  std::cout << "After ReLU:\n";
  outputs[0]->print("relu_out");

  // Verify channel 0: [0,2,0; 4,0,6]
  // Verify channel 1: [7,0,9; 0,11,0]
  std::cout << "Verification:\n";
  std::cout << "  Channel 0, [0,0] should be 0: " << outputs[0]->at(0, 0, 0) << "\n";
  std::cout << "  Channel 0, [0,1] should be 2: " << outputs[0]->at(0, 0, 1) << "\n";
  std::cout << "  Channel 1, [1,1] should be 11: " << outputs[0]->at(1, 1, 1) << "\n";
  std::cout << "  Channel 1, [1,2] should be 0: " << outputs[0]->at(1, 1, 2) << "\n";

  delete relu;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 4. Demo: Chain layers via factory (double ReLU)                    */
/* ------------------------------------------------------------------ */
void demo_layer_chain() {
  std::cout << "========== 4. Layer Chain (double ReLU) ==========\n\n";

  // Create two ReLU layers independently via factory
  Layer<float>* relu1 = LayerRegisterer<float>::CreateLayer("nn.ReLU");
  Layer<float>* relu2 = LayerRegisterer<float>::CreateLayer("nn.ReLU");

  std::cout << "Layer 1 type: " << relu1->Type() << "\n";
  std::cout << "Layer 2 type: " << relu2->Type() << "\n\n";

  // Input
  auto input = FTensor::Create(
      1, 2, 4,
      {-2, -1,  0,  1,
       -3,  2, -1,  3});

  std::cout << "Input:\n";
  input->print("input");

  // Pass 1: ReLU
  std::vector<std::shared_ptr<Tensor<float>>> inputs1;
  inputs1.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  relu1->Forward(inputs1, outputs1);
  std::cout << "After first ReLU:\n";
  outputs1[0]->print("relu1_out");

  // Pass 2: ReLU again (idempotent -- output should be identical)
  std::vector<std::shared_ptr<Tensor<float>>> inputs2;
  inputs2.push_back(outputs1[0]);
  std::vector<std::shared_ptr<Tensor<float>>> outputs2;
  relu2->Forward(inputs2, outputs2);
  std::cout << "After second ReLU (idempotent, same result):\n";
  outputs2[0]->print("relu2_out");

  // Verify idempotence: ReLU(ReLU(x)) == ReLU(x)
  bool same = TensorIsSame(outputs1[0], outputs2[0]);
  std::cout << "Idempotence check (ReLU(ReLU(x)) == ReLU(x)): "
            << (same ? "PASS" : "FAIL") << "\n";

  delete relu1;
  delete relu2;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 5. Demo: Error handling                                            */
/* ------------------------------------------------------------------ */
void demo_error_handling() {
  std::cout << "========== 5. Error Handling ==========\n\n";

  // Test: unknown layer type throws
  std::cout << "Test: CreateLayer with unknown type...\n";
  try {
    Layer<float>* bad = LayerRegisterer<float>::CreateLayer("nn.Unknown");
    std::cout << "FAIL: should have thrown\n";
    delete bad;
  } catch (const std::runtime_error& e) {
    std::cout << "PASS: caught exception: " << e.what() << "\n";
  }

  // Test: empty input returns kInvalidInput
  std::cout << "\nTest: Forward with empty inputs...\n";
  Layer<float>* relu = LayerRegisterer<float>::CreateLayer("nn.ReLU");
  std::vector<std::shared_ptr<Tensor<float>>> empty_inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  StatusCode status = relu->Forward(empty_inputs, outputs);
  std::cout << "Status: "
            << (status == StatusCode::kInvalidInput ? "kInvalidInput (PASS)"
                                                    : "unexpected") << "\n";

  delete relu;
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
/* 6. Demo: HasType introspection                                     */
/* ------------------------------------------------------------------ */
void demo_introspection() {
  std::cout << "========== 6. Factory Introspection ==========\n\n";

  std::cout << "HasType(\"nn.ReLU\"): "
            << (LayerRegisterer<float>::HasType("nn.ReLU") ? "true" : "false")
            << " (expected: true)\n";
  std::cout << "HasType(\"nn.Conv2D\"): "
            << (LayerRegisterer<float>::HasType("nn.Conv2D") ? "true" : "false")
            << " (expected: false)\n";
  std::cout << "HasType(\"nn.BatchNorm\"): "
            << (LayerRegisterer<float>::HasType("nn.BatchNorm") ? "true" : "false")
            << " (expected: false)\n";
  std::cout << "\n";
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 4: Layer System and Factory\n";
  std::cout << "========================================\n\n";

  demo_registered_types();
  demo_relu_factory();
  demo_relu_multichannel();
  demo_layer_chain();
  demo_error_handling();
  demo_introspection();

  std::cout << "========================================\n";
  std::cout << "  Day 4 Complete!\n";
  std::cout << "  Learned: Layer<T> base class,\n";
  std::cout << "           LayerRegisterer factory,\n";
  std::cout << "           LayerRegistererWrapper RAII,\n";
  std::cout << "           ReluLayer concrete impl\n";
  std::cout << "========================================\n";

  return 0;
}
