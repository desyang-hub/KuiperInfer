/**
 * day4/include/layer/layer_factory.hpp
 *
 * Layer Factory -- registration-based factory for creating layers by name.
 *
 * Pattern:
 *   1. LayerRegisterer holds a static map from string type names to
 *      callable creator functions (std::function<Layer*()>).
 *   2. RegisterCreator adds a name -> creator mapping.
 *   3. CreateLayer looks up a name and returns a fresh Layer*.
 *   4. LayerRegistererWrapper is an RAII helper: its constructor calls
 *      RegisterCreator, so simply declaring a static instance in a
 *      translation unit auto-registers the layer at program start.
 *
 * Usage example (in a .hpp file):
 *   static LayerRegistererWrapper<ReluLayer<float>>
 *       g_relu("nn.ReLU", []() { return new ReluLayer<float>(); });
 *
 * Then at runtime:
 *   auto layer = LayerRegisterer<float>::CreateLayer("nn.ReLU");
 */

#ifndef LAYER_FACTORY_HPP
#define LAYER_FACTORY_HPP

#include "layer.hpp"
#include <string>
#include <map>
#include <functional>
#include <stdexcept>

namespace learn_infer {

/**
 * LayerRegisterer -- static registration map and factory methods.
 */
template <typename T>
class LayerRegisterer {
 public:
  using Creator = std::function<Layer<T>*(void)>;

  /**
   * Register a creator function for the given type string.
   * Returns true on success, false if the name is already registered.
   */
  static bool RegisterCreator(const std::string& type, Creator creator) {
    auto& registry = GetRegistry();
    if (registry.count(type)) {
      return false;  // already registered
    }
    registry[type] = std::move(creator);
    return true;
  }

  /**
   * Create a Layer<T> instance by type string.
   * Throws std::runtime_error if the type is not found.
   */
  static Layer<T>* CreateLayer(const std::string& type) {
    auto& registry = GetRegistry();
    auto it = registry.find(type);
    if (it == registry.end()) {
      throw std::runtime_error(
          "LayerRegisterer::CreateLayer: unknown type \"" + type + "\"");
    }
    return it->second();
  }

  /**
   * Check whether a type string has been registered.
   */
  static bool HasType(const std::string& type) {
    auto& registry = GetRegistry();
    return registry.count(type) > 0;
  }

  /**
   * List all registered type names (for debugging/introspection).
   */
  static std::vector<std::string> ListTypes() {
    auto& registry = GetRegistry();
    std::vector<std::string> types;
    types.reserve(registry.size());
    for (auto& p : registry) {
      types.push_back(p.first);
    }
    return types;
  }

 private:
  static std::map<std::string, Creator>& GetRegistry() {
    static std::map<std::string, Creator> registry;
    return registry;
  }
};

/**
 * LayerRegistererWrapper -- RAII auto-registration helper.
 *
 * Template parameter L is the concrete layer type.
 *
 * Constructor registers a lambda that creates a new L instance.
 * Because the wrapper is typically declared as a static/global variable,
 * registration happens automatically at program startup (before main()).
 */
template <typename L>
class LayerRegistererWrapper {
 public:
  using T = typename L::value_type;  // L must typedef value_type

  LayerRegistererWrapper(const std::string& type) {
    LayerRegisterer<T>::RegisterCreator(
        type, []() -> Layer<T>* { return new L(); });
  }
};

}  // namespace learn_infer

#endif  // LAYER_FACTORY_HPP
