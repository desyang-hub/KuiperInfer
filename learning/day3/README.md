# Day 3: Tensor Utility Functions

## Overview

Day 3 introduces a set of free utility functions that operate on `Tensor` objects and return new `shared_ptr<Tensor>` results. These functions are the building blocks for implementing neural network layers (addition, multiplication, broadcasting).

The `Tensor` class itself is copied from Day 2 (unchanged), and all new functionality lives in `tensor_util.hpp`.

## New Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| **TensorElementAdd** | `TensorElementAdd(a, b)` | Element-wise addition `a + b`. Returns a new tensor. |
| **TensorElementMultiply** | `TensorElementMultiply(a, b)` | Element-wise multiplication `a * b`. Returns a new tensor. |
| **TensorBroadcast** | `TensorBroadcast(a, b)` | Broadcast tensor `a` (shape `[C,1,1]`) to shape of `b` (`[C,H,W]`). Repeats each channel's single value across all spatial positions. |
| **TensorIsSame** | `TensorIsSame(a, b, threshold)` | Returns `true` if every corresponding element differs by less than `threshold` (default `1e-6`). |
| **TensorClone** | `TensorClone(t)` | Free-function wrapper for `t->clone()`. Deep copy. |
| **TensorCreate** | `TensorCreate(shapes, values)` | Factory from a `std::vector<uint32_t>` shapes spec (length 1, 2, or 3) and flat values. |

## Design Notes

- **Header-only**: Both `tensor.hpp` and `tensor_util.hpp` are pure header files with template implementations. No `.cpp` source files needed.
- **No external dependencies**: Only C++ standard library.
- **Non-mutating**: `TensorElementAdd`, `TensorElementMultiply`, and `TensorBroadcast` return new tensors; they do not modify inputs.
- **Broadcast semantics**: The broadcast function supports the `[C,1,1] -> [C,H,W]` pattern commonly used for per-channel parameters (batch norm gamma/beta, channel-wise scaling).

## File Structure

```
day3/
  CMakeLists.txt
  main.cpp                 # Demo code for all utility functions
  README.md
  include/
    tensor.hpp             # Tensor class (copied from day2, unchanged)
    tensor_util.hpp        # New: utility functions
  build/                   # Build directory (created by cmake)
```

## Build and Run

```bash
mkdir -p build && cd build
cmake ..
make
./day3
```

## Demo Sections

1. **Element-wise Add**: `[1,3,3]` add with single and multi-channel tensors
2. **Element-wise Multiply**: Scale by constant, multiply vectors
3. **Broadcast**: `[C,1,1]` to `[C,H,W]`, then a realistic per-channel scale + shift
4. **IsSame**: Exact comparison and floating-point tolerance tests
5. **Clone**: Deep copy with mutation independence verification
6. **TensorCreate**: Factory with 1-D, 2-D, and 3-D shape specs
7. **Mini Pipeline**: A ResNet-style block simulating ReLU, batch norm affine, and residual connection
