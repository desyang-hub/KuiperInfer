/**
 * day2/main.cpp
 *
 * Day 2 演示：Tensor 张量类的扩展操作
 *
 * 新增操作：Reshape / Flatten / Transform / Padding / Clone / Ones / RandN
 *
 * 编译：mkdir build && cd build && cmake .. && make && ./day2
 */

#include "include/tensor.hpp"
#include <iostream>

using namespace learn_infer;

void demo_reshape() {
  std::cout << "========== 1. Reshape ==========\n\n";

  // 创建 [2, 3, 4] 张量，填入 1~24
  auto t = FTensor::Create(
      2, 3, 4,
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  std::cout << "Before reshape:\n";
  t->print("t");

  // Reshape 到 [4, 3, 2] — 元素总数 24 不变
  t->reshape(4, 3, 2);
  std::cout << "After reshape(4, 3, 2):\n";
  t->print("t");

  // 再 reshape 回 [2, 3, 4]
  t->reshape(2, 3, 4);
  std::cout << "After reshape back to (2, 3, 4):\n";
  t->print("t");
}

void demo_flatten() {
  std::cout << "========== 2. Flatten ==========\n\n";

  auto t = FTensor::Create(
      2, 2, 3,
      {1, 2, 3, 4, 5, 6,
       7, 8, 9, 10, 11, 12});

  std::cout << "Before flatten:\n";
  t->print("t");

  t->flatten();
  std::cout << "After flatten:\n";
  t->print("t");

  std::cout << "  shapes() = [";
  auto s = t->shapes();
  for (uint32_t i = 0; i < s.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << s[i];
  }
  std::cout << "]  (auto-simplified to 1D)\n\n";
}

void demo_transform() {
  std::cout << "========== 3. Transform ==========\n\n";

  auto t = FTensor::Create(
      1, 3, 3,
      {1, 2, 3,
       4, 5, 6,
       7, 8, 9});

  std::cout << "Before transform:\n";
  t->print("t");

  // 平方：x -> x*x
  t->transform([](float x) { return x * x; });
  std::cout << "After transform(x -> x*x):\n";
  t->print("t");

  // 再应用 sin
  t->transform([](float x) { return std::sin(x); });
  std::cout << "After transform(x -> sin(x)):\n";
  t->print("t");

  // 再演示 scale + shift：x -> 2*x + 1
  auto t2 = FTensor::Create(1, 2, 4, {0, 1, 2, 3, 4, 5, 6, 7});
  std::cout << "t2 before:\n";
  t2->print("t2");

  t2->transform([](float x) { return 2.0f * x + 1.0f; });
  std::cout << "After transform(x -> 2x+1):\n";
  t2->print("t2");
}

void demo_padding() {
  std::cout << "========== 4. Padding ==========\n\n";

  // 创建一个 [1, 3, 3] 的张量
  auto t = FTensor::Create(
      1, 3, 3,
      {1, 2, 3,
       4, 5, 6,
       7, 8, 9});

  std::cout << "Before padding:\n";
  t->print("t");

  // pads=[1, 1, 1, 1] -> 上下各 1 行，左右各 1 列
  t->padding({1, 1, 1, 1}, 0.0f);
  std::cout << "After padding([1,1,1,1], value=0):\n";
  t->print("t");

  // 再演示 asymmetric padding
  auto t2 = FTensor::Create(
      1, 2, 2,
      {10, 20,
       30, 40});

  std::cout << "Before padding (asymmetric):\n";
  t2->print("t2");

  // pads=[2, 0, 1, 3] -> 上 2 行，下 0 行，左 1 列，右 3 列
  t2->padding({2, 0, 1, 3}, -1.0f);
  std::cout << "After padding([2,0,1,3], value=-1):\n";
  t2->print("t2");

  // 演示 3D padding (多 channel)
  auto t3 = FTensor::Create(
      2, 2, 2,
      {1, 2, 3, 4,
       10, 20, 30, 40});

  std::cout << "Before padding (3D):\n";
  t3->print("t3");

  t3->padding({1, 0, 1, 0}, 0.0f);
  std::cout << "After padding([1,0,1,0], value=0):\n";
  t3->print("t3");
}

void demo_clone() {
  std::cout << "========== 5. Clone ==========\n\n";

  auto original = FTensor::Create(
      2, 2, 2,
      {1, 2, 3, 4,
       5, 6, 7, 8});

  std::cout << "Original:\n";
  original->print("original");

  // Clone 得到深拷贝
  auto cloned = original->clone();
  std::cout << "Cloned (deep copy):\n";
  cloned->print("cloned");

  // 修改 clone 不影响 original
  cloned->fill(99.0f);
  std::cout << "After cloned.fill(99.0f):\n";
  std::cout << "Original (should be unchanged):\n";
  original->print("original");
  std::cout << "Cloned:\n";
  cloned->print("cloned");

  // 验证数据指针不同（深拷贝）
  std::cout << "Original data ptr: " << (const void*)original->data() << "\n";
  std::cout << "Cloned   data ptr: " << (const void*)cloned->data() << "\n";
  std::cout << "(different pointers = deep copy confirmed)\n\n";
}

void demo_ones() {
  std::cout << "========== 6. Ones ==========\n\n";

  FTensor t1(3, 3);
  std::cout << "Created [3, 3] tensor (zeros by default):\n";
  t1.print("t1");

  t1.ones();
  std::cout << "After ones():\n";
  t1.print("t1");

  // 3D ones
  FTensor t2(2, 2, 3);
  t2.ones();
  std::cout << "Created [2, 2, 3] tensor, ones():\n";
  t2.print("t2");
}

void demo_randn() {
  std::cout << "========== 7. RandN ==========\n\n";

  // 标准正态分布 N(0, 1)
  FTensor t1(2, 5);
  t1.randn(0.0f, 1.0f);
  std::cout << "RandN(mean=0, var=1), shape=[2, 5]:\n";
  t1.print("t1");

  // 自定义均值和方差 N(5, 4) => stddev=2
  FTensor t2(1, 3, 4);
  t2.randn(5.0f, 4.0f);
  std::cout << "RandN(mean=5, var=4 => stddev=2), shape=[3, 4]:\n";
  t2.print("t2");

  // 小方差 N(0, 0.01) => stddev=0.1
  FTensor t3(1, 1, 6);
  t3.randn(0.0f, 0.01f);
  std::cout << "RandN(mean=0, var=0.01 => stddev=0.1), shape=[6]:\n";
  t3.print("t3");
}

void demo_pipeline() {
  std::cout << "========== Pipeline: Combining Operations ==========\n\n";

  // 模拟一个小型图像处理 pipeline
  std::cout << "Simulating a mini image-processing pipeline:\n\n";

  // Step 1: create a 3x3 "image" with values
  auto img = FTensor::Create(1, 3, 3,
                              {0, 0, 0,
                               0, 1, 0,
                               0, 0, 0});
  std::cout << "Step 1 - Original 3x3 image:\n";
  img->print("img");

  // Step 2: pad with zeros
  img->padding({1, 1, 1, 1}, 0.0f);
  std::cout << "Step 2 - After padding([1,1,1,1]):\n";
  img->print("img");

  // Step 3: apply a transform (sigmoid-like)
  img->transform([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  std::cout << "Step 3 - After sigmoid(x):\n";
  img->print("img");

  // Step 4: clone for two branches
  auto branch_a = img->clone();
  auto branch_b = img->clone();

  branch_a->transform([](float x) { return x * 2.0f; });
  branch_b->transform([](float x) { return x * x; });

  std::cout << "Step 4 - Branch A (x*2):\n";
  branch_a->print("branch_a");

  std::cout << "Step 4 - Branch B (x^2):\n";
  branch_b->print("branch_b");

  // Step 5: flatten both
  branch_a->flatten();
  branch_b->flatten();

  std::cout << "Step 5 - Flattened branch A shape: ";
  auto sa = branch_a->shapes();
  for (uint32_t i = 0; i < sa.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << sa[i];
  }
  std::cout << "\n";

  std::cout << "Step 5 - Flattened branch B shape: ";
  auto sb = branch_b->shapes();
  for (uint32_t i = 0; i < sb.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << sb[i];
  }
  std::cout << "\n\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 2: Tensor Extended Operations\n";
  std::cout << "========================================\n\n";

  demo_reshape();
  demo_flatten();
  demo_transform();
  demo_padding();
  demo_clone();
  demo_ones();
  demo_randn();
  demo_pipeline();

  std::cout << "========================================\n";
  std::cout << "  Day 2 Complete!\n";
  std::cout << "  Learned: Reshape, Flatten, Transform,\n";
  std::cout << "           Padding, Clone, Ones, RandN\n";
  std::cout << "========================================\n";

  return 0;
}
