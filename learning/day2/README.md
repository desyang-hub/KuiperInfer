# Day 2: Tensor Extended Operations

## Overview

Day 2 extends the `Tensor` class from Day 1 with seven new operations essential for building a deep learning inference framework.

## New Operations

| Operation | Signature | Description |
|-----------|-----------|-------------|
| **Reshape** | `reshape(channels, rows, cols, row_major)` | Change tensor shape while preserving element count. Row-major mode keeps the linear element order. |
| **Flatten** | `flatten(row_major)` | Flatten to 1D tensor via Reshape. |
| **Transform** | `transform(func)` | Apply an element-wise lambda function in-place. |
| **Padding** | `padding(pads, value)` | Pad spatial dimensions. `pads = [top, bottom, left, right]`. |
| **Clone** | `clone()` | Deep copy returning `shared_ptr<Tensor>`. |
| **Ones** | `ones()` | Fill all elements with 1. |
| **RandN** | `randn(mean, var)` | Fill with normal distribution random values. |

## Design Notes

- **Header-only**: All implementations are in `include/tensor.hpp`. No separate `.cpp` files.
- **No external dependencies**: Uses only C++ standard library (`<vector>`, `<random>`, `<functional>`, etc.).
- **In-place operations**: Reshape, Flatten, Transform, Padding, Ones, and RandN all modify the tensor in-place.
- **Clone** is the only operation that creates a new tensor.
- **Padding** works per-channel: it pads each channel's spatial dimensions independently, preserving the channel count.

## File Structure

```
day2/
  CMakeLists.txt
  main.cpp               # Demo code showing each operation
  README.md
  include/
    tensor.hpp           # Extended Tensor class (header-only)
  build/                 # Build directory (created by cmake)
```

## Build and Run

```bash
mkdir -p build && cd build
cmake ..
make
./day2
```

## Demo Output

The `main.cpp` demonstrates each operation with clear before/after output:

1. **Reshape**: `[2,3,4] -> [4,3,2] -> [2,3,4]` with values 1-24
2. **Flatten**: `[2,2,3] -> [12]` with auto-dimension simplification
3. **Transform**: Apply `x*x`, `sin(x)`, and `2x+1` lambdas
4. **Padding**: Zero-padding and asymmetric padding with custom value
5. **Clone**: Deep copy verification (different data pointers, independent mutation)
6. **Ones**: Fill with 1s for both 2D and 3D tensors
7. **RandN**: Three examples with different mean/variance settings
8. **Pipeline**: A mini image-processing pipeline combining multiple operations
