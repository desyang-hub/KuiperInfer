# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KuiperInfer is a hand-written C++17 deep learning inference framework that loads models in PNNX format (`.param` + `.bin` files exportable from PyTorch via torch2pnnx). It builds a runtime computation graph and executes forward inference on CPU using Armadillo + OpenBLAS. Supported models include ResNet, YOLOv5, MobileNet, and U-Net.

Additionally, `demos/kuiper_llama/` is a separate git submodule implementing a LLM inference framework supporting Llama2/3.x and Qwen2.5 with CUDA kernels and Int8 quantization.

## Build System

- **CMake** (min 3.16) with **vcpkg** for dependency management (`vcpkg.json`)
- **Dependencies**: armadillo, openblas, opencv, lapack, superlu, glog, gtest, benchmark, OpenMP

### Build Commands

```bash
git clone --recursive https://github.com/zjhellofss/KuiperInfer.git
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DDEVELOPMENT=ON ..
make -j$(nproc)
```

### CMake Options

| Flag | Description |
|------|-------------|
| `-DDEVELOPMENT=ON` | Enable tests and benchmarks (default: ON) |
| `-DBUILD_DEMO=ON` | Build demo applications (YOLO, ResNet, etc.) |

### Key Targets

| Target | Output | Location |
|--------|--------|----------|
| `kuiper` | Shared library (`libkuiper.so`) | `lib/` |
| `test_kuiper` | Test executable | `build/test/` |
| `bench_kuiper` | Benchmark executable | `build/bench/` |

### Run Tests

```bash
./build/test/test_kuiper
```

Tests use **Google Test** and cover data layer, layer operators, runtime graph, and full network inference. Test files are organized in `test/test_data/`, `test/test_layer/`, `test/test_net/`, `test/test_runtime/`.

### Run Benchmarks

```bash
./build/bench/bench_kuiper
```

Benchmarks use **Google Benchmark** and cover Conv, Reshape, RMSNorm, SIMD, and full models (ResNet, MobileNet, YOLO, U-Net).

## Architecture

### Four-Layer Design

1. **Data Layer** (`include/data/`, `source/data/`) — `Tensor<T>` wraps `arma::Cube<T>` for N-dimensional arrays with reshape, slice, transform, etc.

2. **Layer/Operator Layer** (`include/layer/`, `source/layer/`) — `Layer<float>` is the abstract base with `Forward(inputs, outputs)`. Concrete operators (Convolution, MaxPool, ReLU, Softmax, etc.) register themselves via `LayerRegisterer` factory pattern at static init time.

3. **Parser Layer** (`include/parser/`, `source/parser/`) — `ExpressionParser` is a lexer + recursive descent parser that parses PNNX expression strings into an AST for computing complex operator expressions.

4. **Runtime/Graph Layer** (`include/runtime/`, `source/runtime/`) — `RuntimeGraph` is the top-level class. Pipeline: `Init()` (parse PNNX) → `Build()` (construct graph with topological sort) → `set_inputs()` → `Forward()` → `get_outputs()`. Graph nodes are `RuntimeOperator<T>`, connected by `RuntimeOperand<T>`, with weights as `RuntimeAttribute`.

### PNNX IR

The PNNX intermediate representation (`include/runtime/pnnx/ir.h`) is borrowed from NCNN (BSD-3 licensed). KuiperInfer maps PNNX `Graph`/`Operator`/`Operand` to its own `RuntimeGraph`/`RuntimeOperator`/`RuntimeOperand`.

### Data Flow

```
.param + .bin → PNNX IR → RuntimeGraph::Build() → Topological Sort →
Create Layer objects via Factory → Propagate data in order → Output tensors
```

### Supported Operators

Convolution, Deconvolution (ConvTranspose), MaxPooling, AdaptiveAvgPooling, BatchNorm2D, ReLU, ReLU6, Sigmoid, HardSigmoid, HardSwish, SiLU, Softmax, Linear, MatMul, Flatten, View, Concat, Upsample, Expression, YoloDetect, RMSNorm

## Code Style

- **clang-format**: Google style, pointer alignment left, column limit 100
- Format with: `clang-format -i <file>`

## Git Submodules

| Submodule | Path | Purpose |
|-----------|------|---------|
| model_zoo | `tmp/` | Pre-trained PNNX model files |
| KuiperLLama | `demos/kuiper_llama/` | LLM inference framework (Llama, Qwen) with CUDA support |

Always clone with `--recursive` to include submodules.

## CI/CD

- `.github/workflows/cmake.yml` — Linux build in Docker, runs `test_kuiper`
- `.github/workflows/cmake-windows.yml` — Windows build with vcpkg, runs `test_kuiper.exe`
- Docker image: `hellofss/kuiperinfer:latest` (Ubuntu 22.04 with all dependencies pre-installed)
