cmake_minimum_required(VERSION 3.16)
project(day4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only project: include/ contains tensor.hpp, tensor_util.hpp,
# and the layer system (layer/layer.hpp, layer/layer_factory.hpp, layer/relu.hpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(day4 main.cpp)
