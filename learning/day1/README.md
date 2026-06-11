# Day 1：从零开始构建 Tensor 张量类

## 学习目标

今天，我们从零开始手写推理框架最核心的数据结构——**Tensor（张量）**。

Tensor 是深度学习中一切计算的基础载体。无论是输入图像、卷积核权重、还是中间激活值，都以 Tensor 的形式存在。

### 今天你将学到

1. 什么是 Tensor？为什么深度学习需要张量？
2. 如何用 C++ 实现一个支持 1D/2D/3D 的 Tensor 类
3. 张量的内存布局（channels, rows, cols）
4. 基本的张量操作：创建、访问、填充、打印

### 前置知识

- C++ 基础：类、模板、智能指针、vector
- 了解深度学习中张量的概念（类似 NumPy 的 ndarray）

## 项目结构

```
day1/
├── CMakeLists.txt
├── include/
│   └── tensor.hpp      # Tensor 类定义
├── source/
│   └── tensor.cpp      # Tensor 类实现
└── main.cpp            # 演示程序
```

## 编译与运行

```bash
mkdir build && cd build
cmake ..
make
./day1
```

## 核心概念

### Tensor 是什么？

Tensor（张量）就是 N 维数组。在深度学习推理框架中，我们主要用到 3D 张量：

```
形状 [channels, rows, cols]

例如一张 RGB 图片 224×224：
├─ channel 0 (R)  →  224×224 矩阵
├─ channel 1 (G)  →  224×224 矩阵
└─ channel 2 (B)  →  224×224 矩阵
```

### 内部存储

我们的 Tensor 内部用一维 `std::vector<float>` 存储所有数据，通过 `(channel, row, col)` 三个坐标来计算元素的实际位置：

```
线性索引 = (channel × rows × cols) + (row × cols) + col
```

这就是所谓的**行优先（row-major）**内存布局，与 PyTorch/NumPy 一致。

### 维度简化

为了方便使用，我们的 Tensor 会自动简化维度：
- 创建 `[1, 1, 5]` → 逻辑上视为 1D，`shapes()` 返回 `[5]`
- 创建 `[1, 3, 4]` → 逻辑上视为 2D，`shapes()` 返回 `[3, 4]`
- 创建 `[2, 3, 4]` → 保持 3D，`shapes()` 返回 `[2, 3, 4]`

## 下一步

Day 2 将学习：Tensor 的 Fill（填充）、Reshape（变形）、Flatten（展平）、Transform（逐元素变换）等操作。
