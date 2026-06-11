# Day 4: Layer System and Factory Pattern

## Overview

Day 4 introduces the **Layer system** and **Factory pattern**, the architectural
foundation for building composable neural network models.  Instead of calling
tensor utility functions directly in a script, layers encapsulate their forward
logic behind a common interface, and a factory creates layers by name (string).

The `Tensor` and `TensorUtil` classes are copied from Day 3 (unchanged).

## New Files

| File | Description |
|------|-------------|
| `include/layer/layer.hpp` | Abstract base class `Layer<T>` with virtual `Forward()` and `Type()` |
| `include/layer/layer_factory.hpp` | `LayerRegisterer<T>` static factory + `LayerRegistererWrapper` RAII auto-registration |
| `include/layer/relu.hpp` | `ReluLayer<T>` -- first concrete layer; auto-registers as `nn.ReLU` |

## Design

### Layer<T> (Abstract Base)

```cpp
template <typename T>
class Layer {
 public:
  virtual StatusCode Forward(const std::vector<STensor<T>>& inputs,
                             std::vector<STensor<T>>& outputs) = 0;
  virtual std::string Type() const = 0;
};
```

- Template parameter `T` is the element type (e.g., `float`).
- `Forward()` takes input tensors, computes, and writes output tensors.
- `Type()` returns a string identifier like `"nn.ReLU"`.
- `StatusCode` enum provides simple return codes (`kSuccess`, `kInvalidInput`, etc.).

### LayerRegisterer<T> (Factory)

```cpp
template <typename T>
class LayerRegisterer {
 public:
  static bool RegisterCreator(const std::string& type, Creator creator);
  static Layer<T>* CreateLayer(const std::string& type);
  static bool HasType(const std::string& type);
  static std::vector<std::string> ListTypes();
};
```

- Holds a `static std::map<string, Creator>` registry.
- `CreateLayer("nn.ReLU")` returns a heap-allocated `Layer<float>*`.
- Throws `std::runtime_error` for unknown types.

### LayerRegistererWrapper (RAII Auto-Registration)

```cpp
template <typename L>
class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(const std::string& type);  // registers L under type
};
```

- Declaring a `static` instance of this wrapper automatically registers a layer type.
- In `relu.hpp`:
  ```cpp
  static LayerRegistererWrapper<ReluLayer<float>> g_relu_registrar("nn.ReLU");
  ```
  This registers `ReluLayer<float>` under `"nn.ReLU"` at program startup.
- The concrete layer type `L` must define `using value_type = T;` so the wrapper can deduce the template parameter.

### ReluLayer (Concrete Layer)

```cpp
template <typename T>
class ReluLayer : public Layer<T> {
  using value_type = T;

  StatusCode Forward(...) override {
    auto out = inputs[0]->clone();
    out->transform([](T x) { return std::max(T{0}, x); });
    outputs = {out};
    return StatusCode::kSuccess;
  }

  std::string Type() const override { return "nn.ReLU"; }
};
```

- Uses `Tensor::transform()` (from Day 2) to apply `max(0, x)` element-wise.
- Returns a cloned tensor (non-mutating).

## File Structure

```
day4/
  CMakeLists.txt
  main.cpp                     # Demo: factory creation, ReLU, chaining, error handling
  README.md
  include/
    tensor.hpp                 # Copied from day3 (unchanged)
    tensor_util.hpp            # Copied from day3 (unchanged)
    layer/
      layer.hpp                # Abstract Layer<T> base class
      layer_factory.hpp        # LayerRegisterer + LayerRegistererWrapper
      relu.hpp                 # ReluLayer + auto-registration as "nn.ReLU"
  build/                       # Build directory (created by cmake)
```

## Build and Run

```bash
mkdir -p build && cd build
cmake ..
make
./day4
```

## Demo Sections

1. **Registered Types**: List all factory-registered layer names
2. **Factory ReLU**: Create a ReLU layer by name, run forward pass on a [1,3,3] tensor
3. **Multi-Channel ReLU**: Apply ReLU to a [2,2,3] tensor
4. **Layer Chain**: Chain two ReLU layers via factory; verify idempotence
5. **Error Handling**: Unknown type throws, empty input returns `kInvalidInput`
6. **Introspection**: `HasType()` checks for registered/unknown types
