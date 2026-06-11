/**
 * day1/main.cpp
 *
 * Day 1 演示：Tensor 张量类的基本使用
 *
 * 这个文件既是演示代码，也是测试代码。
 * 每一段演示一个功能，配有详细注释。
 *
 * 编译：mkdir build && cd build && cmake .. && make && ./day1
 */

#include "include/tensor.hpp"
#include <iostream>
#include <cmath>

using namespace learn_infer;

void demo_1d_tensor() {
  std::cout << "========== 1D Tensor ==========\n\n";

  // 创建 1D 张量 [5]，初始化为 0
  FTensor t1(5);
  std::cout << "创建 1D Tensor，大小=5，初始值全为 0:\n";
  t1.print("t1");

  // 用 fill 填充值
  t1.fill(1.0f);
  std::cout << "fill(1.0f) 后:\n";
  t1.print("t1");

  // 用 vector 填充
  std::vector<float> values = {10, 20, 30, 40, 50};
  t1.fill(values);
  std::cout << "fill({10, 20, 30, 40, 50}) 后:\n";
  t1.print("t1");

  // 通过 index() 访问元素
  std::cout << "t1.index(0) = " << t1.index(0) << "\n";
  std::cout << "t1.index(2) = " << t1.index(2) << "\n";
  std::cout << "t1.index(4) = " << t1.index(4) << "\n\n";
}

void demo_2d_tensor() {
  std::cout << "========== 2D Tensor ==========\n\n";

  // 创建 2D 张量 [2, 3]（2 行 3 列）
  // 内部存储：channels=1, rows=2, cols=3
  FTensor t2(2, 3);

  // 用 at() 赋值
  for (uint32_t r = 0; r < t2.rows(); r++) {
    for (uint32_t c = 0; c < t2.cols(); c++) {
      t2.at(0, r, c) = static_cast<float>(r * t2.cols() + c + 1);
    }
  }

  std::cout << "2D Tensor [2, 3]，填入 1~6:\n";
  t2.print("t2");

  std::cout << "shapes() = [";
  auto shapes = t2.shapes();
  for (uint32_t i = 0; i < shapes.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << shapes[i];
  }
  std::cout << "]  (自动简化为 2D)\n\n";
}

void demo_3d_tensor() {
  std::cout << "========== 3D Tensor ==========\n\n";

  // 创建 3D 张量 [2, 2, 3]（2 个 channel，每个 2×3）
  FTensor t3(2, 2, 3);

  std::cout << "channels=" << t3.channels()
            << ", rows=" << t3.rows()
            << ", cols=" << t3.cols()
            << ", size=" << t3.size() << "\n";

  // 按 channel 填充不同值
  for (uint32_t ch = 0; ch < t3.channels(); ch++) {
    for (uint32_t r = 0; r < t3.rows(); r++) {
      for (uint32_t c = 0; c < t3.cols(); c++) {
        t3.at(ch, r, c) = static_cast<float>(ch * 100 + r * 10 + c);
      }
    }
  }

  std::cout << "\n3D Tensor [2, 2, 3] 内容：\n";
  t3.print("t3");

  std::cout << "shapes() = [";
  auto shapes = t3.shapes();
  for (uint32_t i = 0; i < shapes.size(); i++) {
    if (i > 0) std::cout << ", ";
    std::cout << shapes[i];
  }
  std::cout << "]\n\n";
}

void demo_image_tensor() {
  std::cout << "========== 模拟图像 Tensor ==========\n\n";

  // 模拟一张 3×3 的 RGB 图片
  // 形状: [channels=3, rows=3, cols=3]
  uint32_t H = 3, W = 3, C = 3;
  FTensor img(C, H, W);

  // R channel 全 1，G channel 全 2，B channel 全 3
  for (uint32_t r = 0; r < H; r++) {
    for (uint32_t c = 0; c < W; c++) {
      img.at(0, r, c) = 1.0f;  // R
      img.at(1, r, c) = 2.0f;  // G
      img.at(2, r, c) = 3.0f;  // B
    }
  }

  std::cout << "模拟 3×3 RGB 图像:\n";
  img.print("image");

  // 演示 to_vector()
  auto vec = img.to_vector();
  std::cout << "to_vector() 前 9 个元素（R channel）: ";
  for (int i = 0; i < 9; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << "\n";
  std::cout << "to_vector() 后 9 个元素（B channel）: ";
  for (int i = 18; i < 27; i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << "\n\n";
}

void demo_memory_layout() {
  std::cout << "========== 内存布局理解 ==========\n\n";

  // 创建一个小张量来理解内存布局
  // 形状 [2, 2, 2]
  FTensor t(2, 2, 2);

  // 填入 0~7 来观察存储顺序
  std::vector<float> vals = {0, 1, 2, 3, 4, 5, 6, 7};
  t.fill(vals);

  std::cout << "Tensor [2, 2, 2] 填入 {0,1,2,3,4,5,6,7}\n";
  std::cout << "\n内存中（行优先）：\n";

  std::cout << "  索引  值    坐标\n";
  for (uint32_t ch = 0; ch < t.channels(); ch++) {
    for (uint32_t r = 0; r < t.rows(); r++) {
      for (uint32_t c = 0; c < t.cols(); c++) {
        uint32_t idx = (ch * t.rows() + r) * t.cols() + c;
        std::cout << "  " << idx << "     " << std::setw(2)
                  << t.index(idx) << "    [" << ch << "," << r << "," << c
                  << "]\n";
      }
    }
  }

  std::cout << "\n验证 at() 访问：\n";
  std::cout << "  at(0,0,0) = " << t.at(0, 0, 0) << "  (应该是 0)\n";
  std::cout << "  at(0,0,1) = " << t.at(0, 0, 1) << "  (应该是 1)\n";
  std::cout << "  at(0,1,0) = " << t.at(0, 1, 0) << "  (应该是 2)\n";
  std::cout << "  at(1,1,1) = " << t.at(1, 1, 1) << "  (应该是 7)\n\n";
}

void demo_factory_create() {
  std::cout << "========== 工厂方法 Create ==========\n\n";

  // 使用静态工厂方法 Create 一步创建并填充张量
  auto t = FTensor::Create(
      2, 3, 4,   // [channels=2, rows=3, cols=4]
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
  );

  std::cout << "FTensor::Create(2, 3, 4, {1~24}):\n";
  t->print("created_tensor");
}

int main() {
  std::cout << "========================================\n";
  std::cout << "  Day 1: Tensor 张量类 — 从零开始\n";
  std::cout << "========================================\n\n";

  demo_1d_tensor();
  demo_2d_tensor();
  demo_3d_tensor();
  demo_image_tensor();
  demo_memory_layout();
  demo_factory_create();

  std::cout << "========================================\n";
  std::cout << "  Day 1 完成！\n";
  std::cout << "  明天学习：Fill/Reshape/Flatten 等操作\n";
  std::cout << "========================================\n";

  return 0;
}
