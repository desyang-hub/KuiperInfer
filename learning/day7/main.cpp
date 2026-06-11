/**
 * day7/main.cpp
 *
 * Day 7 Demo: Convolution via im2col + GEMM
 *
 * Demonstrates:
 *   - nn.Conv2d: 2D convolution using im2col + GEMM optimization
 *   - im2col process step by step
 *   - 3x3 convolution on a small image
 *   - 1x1 convolution (pointwise)
 *   - All layers from day6 remain registered
 *
 * Build:  mkdir build && cd build && cmake .. && make && ./day7
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
#include "include/tensor_util.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace learn_infer;
using namespace learn_infer::util;

/* ------------------------------------------------------------------ */
/* Helper: print im2col intermediate result                           */
/* ------------------------------------------------------------------ */
void print_im2col_step(const std::shared_ptr<Tensor<float>>& col,
                       uint32_t depth, uint32_t spatial,
                       const std::string& label) {
  std::cout << "  " << label << " shape=[" << depth << ", " << spatial << "]:\n";
  std::cout << std::fixed << std::setprecision(4);
  for (uint32_t r = 0; r < depth; r++) {
    std::cout << "    row " << std::setw(3) << r << ": ";
    for (uint32_t c = 0; c < spatial; c++) {
      std::cout << std::setw(8) << col->at(0, r, c) << " ";
    }
    std::cout << "\n";
  }
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
/* 2. 3x3 Conv: step-by-step im2col demonstration                     */
/* ------------------------------------------------------------------ */
void demo_conv_im2col() {
  std::cout << "========== 2. Conv im2col Step-by-Step ==========\n\n";

  // Input: 1 channel, 4x4
  // Values 1..16 arranged as a 4x4 grid
  auto input = Tensor<float>::Create(1, 4, 4, {
      1.0f,  2.0f,  3.0f,  4.0f,
      5.0f,  6.0f,  7.0f,  8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f
  });

  std::cout << "  Input [1,4,4]:\n";
  input->print("    input");

  // Weights: 1 out channel, 1 in channel, 3x3 kernel (all ones for simplicity)
  // This computes a sum of each 3x3 window
  std::vector<float> weights(9, 1.0f);
  std::vector<float> bias = {0.0f};

  ConvolutionLayer<float> conv(
      1, 4, 4,  // ch, rows, cols
      3, 3,     // kernel_h, kernel_w
      1, 1,     // stride_h, stride_w
      0, 0,     // pad_h, pad_w
      1,        // groups
      weights, bias);

  std::cout << "  Kernel: 3x3 all-ones\n";
  std::cout << "  Stride: 1x1, Pad: 0\n";
  std::cout << "  Expected output: [1, 2, 2]\n\n";

  // Show im2col manually for the first few positions
  // After padding (pad=0, so same as input):
  // Output positions: (0,0), (0,1), (1,0), (1,1)
  // For (0,0): extract [0:3, 0:3] = {1,2,3, 5,6,7, 9,10,11} -> sum=54
  // For (0,1): extract [0:3, 1:4] = {2,3,4, 6,7,8, 10,11,12} -> sum=65

  std::cout << "  Manual im2col for position (0,0):\n";
  std::cout << "    Patch: {1,2,3, 5,6,7, 9,10,11}\n";
  std::cout << "    Sum: 54\n\n";

  std::cout << "  Manual im2col for position (0,1):\n";
  std::cout << "    Patch: {2,3,4, 6,7,8, 10,11,12}\n";
  std::cout << "    Sum: 65\n\n";

  // Run forward
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  conv.Forward(inputs, outputs);

  std::cout << "  Output [1,2,2]:\n";
  outputs[0]->print("    output");

  // Show the im2col matrix
  const auto& col = conv.last_im2col();
  if (col) {
    print_im2col_step(col, conv.in_channels() * conv.kernel_h() * conv.kernel_w(),
                      2 * 2, "im2col matrix");
  }

  // Verify
  float expected[] = {54.0f, 63.0f, 90.0f, 99.0f};
  bool pass = true;
  for (uint32_t i = 0; i < 4; i++) {
    if (std::abs(outputs[0]->index(i) - expected[i]) > 1e-4f) {
      pass = false;
    }
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 3. 3x3 Conv: with padding                                          */
/* ------------------------------------------------------------------ */
void demo_conv_padded() {
  std::cout << "========== 3. Conv 3x3 with Padding ==========\n\n";

  // Input: 1 channel, 3x3
  auto input = Tensor<float>::Create(1, 3, 3, {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f
  });

  // All-ones 3x3 kernel, pad=1 -> output is 3x3
  std::vector<float> weights(9, 1.0f);

  ConvolutionLayer<float> conv(
      1, 3, 3,  // ch, rows, cols
      3, 3,     // kernel
      1, 1,     // stride
      1, 1,     // pad (same padding for 3x3)
      1,        // groups
      weights, {0.0f});

  std::cout << "  Input [1,3,3]:\n";
  input->print("    input");
  std::cout << "  Kernel: 3x3 all-ones, pad=1\n";

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  conv.Forward(inputs, outputs);

  std::cout << "  Output [1,3,3]:\n";
  outputs[0]->print("    output");

  // After pad=1, input becomes:
  //  0  0  0  0  0
  //  0  1  2  3  0
  //  0  4  5  6  0
  //  0  7  8  9  0
  //  0  0  0  0  0
  // Output[0,0] = rows 0-2, cols 0-2: 0+0+0+0+1+2+0+4+5 = 12
  // Output[0,1] = rows 0-2, cols 1-3: 0+0+0+1+2+3+4+5+6 = 21
  // Output[0,2] = rows 0-2, cols 2-4: 0+0+0+2+3+0+5+6+0 = 16
  // Output[1,0] = rows 1-3, cols 0-2: 0+1+2+0+4+5+0+7+8 = 27
  // Output[1,1] = rows 1-3, cols 1-3: 1+2+3+4+5+6+7+8+9 = 45
  // Output[1,2] = rows 1-3, cols 2-4: 2+3+0+5+6+0+8+9+0 = 33
  // Output[2,0] = rows 2-4, cols 0-2: 0+4+5+0+7+8+0+0+0 = 24
  // Output[2,1] = rows 2-4, cols 1-3: 4+5+6+7+8+9+0+0+0 = 39
  // Output[2,2] = rows 2-4, cols 2-4: 5+6+0+8+9+0+0+0+0 = 28
  float expected[] = {12, 21, 16, 27, 45, 33, 24, 39, 28};
  bool pass = true;
  for (uint32_t i = 0; i < 9; i++) {
    if (std::abs(outputs[0]->index(i) - expected[i]) > 1e-4f) {
      pass = false;
      std::cout << "  Mismatch at " << i << ": got "
                << std::fixed << std::setprecision(4)
                << outputs[0]->index(i) << " expected " << expected[i] << "\n";
    }
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 4. 1x1 Conv: pointwise convolution                                 */
/* ------------------------------------------------------------------ */
void demo_conv_1x1() {
  std::cout << "========== 4. Conv 1x1 (Pointwise) ==========\n\n";

  // Input: 1 channel, 3x3
  auto input = Tensor<float>::Create(1, 3, 3, {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f
  });

  // 1x1 conv: 1 input channel -> 1 output channel
  // Weight: [1, 1, 1, 1] = [out_ch=1, in_ch=1, kh=1, kw=1]
  // weight = 3.0, bias = 1.0
  // output[h,w] = 3.0 * input[h,w] + 1.0
  std::vector<float> weights = {3.0f};
  std::vector<float> bias = {1.0f};

  ConvolutionLayer<float> conv(
      1, 3, 3,  // ch, rows, cols
      1, 1,     // kernel
      1, 1,     // stride
      0, 0,     // pad
      1,        // groups
      weights, bias);

  std::cout << "  Input [1,3,3]: values 1..9\n";
  std::cout << "  Weight: [3.0], bias: [1.0]\n";
  std::cout << "  output = 3*x + 1\n";

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  conv.Forward(inputs, outputs);

  std::cout << "  Output [1,3,3]:\n";
  outputs[0]->print("    output");

  // Expected: 3*1+1=4, 3*2+1=7, 3*3+1=10, ...
  float expected[] = {
      4.0f, 7.0f, 10.0f,
      13.0f, 16.0f, 19.0f,
      22.0f, 25.0f, 28.0f
  };
  bool pass = true;
  for (uint32_t i = 0; i < 9; i++) {
    if (std::abs(outputs[0]->index(i) - expected[i]) > 1e-4f) pass = false;
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 5. Multi-output-channel conv                                       */
/* ------------------------------------------------------------------ */
void demo_conv_multiout() {
  std::cout << "========== 5. Conv Multi-Output Channels ==========\n\n";

  // Input: 2 channels, 3x3
  auto input = Tensor<float>::Create(2, 3, 3, {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, // ch0
      10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f // ch1
  });

  // 2 output channels, 1x1 kernel, 2 input channels
  // Weight layout: [C_out=2, C_in=2, kh=1, kw=1] = 8 weights
  // out_ch0: w[0,0]=0.5, w[0,1]=0.1 => 0.5*ch0 + 0.1*ch1
  // out_ch1: w[1,0]=-1.0, w[1,1]=0.0 => -1.0*ch0 + 0*ch1
  std::vector<float> weights = {0.5f, 0.1f, -1.0f, 0.0f};
  std::vector<float> bias = {0.0f, 10.0f};

  ConvolutionLayer<float> conv(
      2, 3, 3,  // 2 output channels, input 3x3
      1, 1,     // kernel
      1, 1,     // stride
      0, 0,     // pad
      1,        // groups
      weights, bias);

  std::cout << "  Input [2,3,3]: 2 channels\n";
  std::cout << "  out_ch0 = 0.5*ch0 + 0.1*ch1\n";
  std::cout << "  out_ch1 = -1.0*ch0 + 10.0 (bias)\n";

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  conv.Forward(inputs, outputs);

  std::cout << "  Output [2,3,3]:\n";
  outputs[0]->print("    output");

  bool pass = true;
  float ch0[] = {1,2,3,4,5,6,7,8,9};
  float ch1[] = {10,20,30,40,50,60,70,80,90};
  for (uint32_t i = 0; i < 9; i++) {
    float e0 = 0.5f * ch0[i] + 0.1f * ch1[i];
    float e1 = -1.0f * ch0[i] + 10.0f;
    if (std::abs(outputs[0]->at(0, i / 3, i % 3) - e0) > 1e-4f) pass = false;
    if (std::abs(outputs[0]->at(1, i / 3, i % 3) - e1) > 1e-4f) pass = false;
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 6. Conv + ReLU pipeline                                            */
/* ------------------------------------------------------------------ */
void demo_conv_relu() {
  std::cout << "========== 6. Conv + ReLU Pipeline ==========\n\n";

  // Input: 1 channel, 4x4 with some negative values
  auto input = Tensor<float>::Create(1, 4, 4, {
      -1.0f,  2.0f, -3.0f,  4.0f,
       5.0f, -6.0f,  7.0f, -8.0f,
      -9.0f, 10.0f,-11.0f, 12.0f,
      13.0f,-14.0f, 15.0f,-16.0f
  });

  // 2x2 all-ones kernel (sum of 2x2 windows)
  std::vector<float> weights(4, 1.0f);

  // Conv: 1ch, 4x4 -> stride=2, no pad -> 2x2 output
  ConvolutionLayer<float> conv(
      1, 4, 4,  // ch, rows, cols
      2, 2,     // kernel
      2, 2,     // stride
      0, 0,     // pad
      1,        // groups
      weights, {0.0f});

  std::cout << "  Input [1,4,4] with negative values:\n";
  input->print("    input");
  std::cout << "  Kernel: 2x2 all-ones, stride=2\n";

  // Conv
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> conv_outputs;
  conv.Forward(inputs, conv_outputs);

  std::cout << "  After Conv [1,2,2]:\n";
  conv_outputs[0]->print("    conv_output");

  // ReLU
  ReluLayer<float> relu;
  std::vector<std::shared_ptr<Tensor<float>>> relu_outputs;
  relu.Forward(conv_outputs, relu_outputs);

  std::cout << "  After ReLU [1,2,2]:\n";
  relu_outputs[0]->print("    relu_output");

  // Verify conv: 4 2x2 windows with stride 2
  // [0,0]: -1+2+5-6 = 0
  // [0,1]: -3+4+7-8 = 0
  // [1,0]: -9+10+13-14 = 0
  // [1,1]: -11+12+15-16 = 0
  float conv_expected[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float relu_expected[] = {0.0f, 0.0f, 0.0f, 0.0f};
  bool pass = true;
  for (uint32_t i = 0; i < 4; i++) {
    if (std::abs(conv_outputs[0]->index(i) - conv_expected[i]) > 1e-4f) pass = false;
    if (std::abs(relu_outputs[0]->index(i) - relu_expected[i]) > 1e-4f) pass = false;
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
/* 7. Conv via factory registration                                   */
/* ------------------------------------------------------------------ */
void demo_conv_factory() {
  std::cout << "========== 7. Conv via Factory ==========\n\n";

  // Register a conv layer
  ConvolutionRegister<float>(
      "nn.Conv2d",
      1, 3, 3,  // ch, rows, cols
      3, 3,     // kernel
      1, 1,     // stride
      1, 1,     // pad
      1,        // groups
      std::vector<float>(9, 1.0f), {0.0f});

  Layer<float>* conv = LayerRegisterer<float>::CreateLayer("nn.Conv2d");
  std::cout << "  Factory created: " << conv->Type() << "\n";

  auto input = Tensor<float>::Create(1, 3, 3, {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f
  });

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  conv->Forward(inputs, outputs);

  std::cout << "  Output [1,3,3]:\n";
  outputs[0]->print("    output");

  // Same as demo_conv_padded expected: [12,21,16,27,45,33,24,39,28]
  float expected[] = {12, 21, 16, 27, 45, 33, 24, 39, 28};
  bool pass = true;
  for (uint32_t i = 0; i < 9; i++) {
    if (std::abs(outputs[0]->index(i) - expected[i]) > 1e-4f) pass = false;
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";

  delete conv;
}

/* ------------------------------------------------------------------ */
/* 8. Batch processing                                                */
/* ------------------------------------------------------------------ */
void demo_conv_batch() {
  std::cout << "========== 8. Conv Batch Processing ==========\n\n";

  // Two different inputs
  auto i1 = Tensor<float>::Create(1, 3, 3, {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f
  });
  auto i2 = Tensor<float>::Create(1, 3, 3, {
      0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f
  });

  // 3x3 all-ones kernel, pad=1
  ConvolutionLayer<float> conv(
      1, 3, 3, 3, 3, 1, 1, 1, 1, 1,
      std::vector<float>(9, 1.0f), {0.0f});

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(i1);
  inputs.push_back(i2);
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  conv.Forward(inputs, outputs);

  std::cout << "  Batch size: " << inputs.size() << "\n";
  std::cout << "  Batch 1 (all 1..9):\n";
  outputs[0]->print("    output");
  std::cout << "  Batch 2 (center=1, rest=0):\n";
  outputs[1]->print("    output");

  // Batch 2 with pad=1: only center element is 1, surrounded by 0s
  // After pad: 5x5 with center=1
  // With 3x3 kernel stride=1, all 9 output windows overlap the center
  // So all output values should be 1.0
  bool pass = true;
  for (uint32_t r = 0; r < 3 && pass; r++) {
    for (uint32_t c = 0; c < 3 && pass; c++) {
      if (outputs[1]->at(0, r, c) != 1.0f) pass = false;
    }
  }
  std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
}

/* ------------------------------------------------------------------ */
int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 7: Convolution (im2col + GEMM)\n";
  std::cout << "========================================\n\n";

  demo_list_all();
  demo_conv_im2col();
  demo_conv_padded();
  demo_conv_1x1();
  demo_conv_multiout();
  demo_conv_relu();
  demo_conv_factory();
  demo_conv_batch();

  std::cout << "========================================\n";
  std::cout << "  Day 7 Complete!\n";
  std::cout << "  Learned: Convolution via im2col,\n";
  std::cout << "           weight reshape, manual GEMM,\n";
  std::cout << "           padding, stride, groups,\n";
  std::cout << "           batch processing\n";
  std::cout << "========================================\n";

  return 0;
}
