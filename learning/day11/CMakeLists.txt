cmake_minimum_required(VERSION 3.16)
project(day11 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only project: include/ contains tensor.hpp, tensor_util.hpp,
# and the layer system (layer/layer.hpp, layer/layer_factory.hpp,
# layer/relu.hpp, layer/sigmoid.hpp, layer/relu6.hpp,
# layer/silu.hpp, layer/hardsigmoid.hpp, layer/hardswish.hpp,
# layer/softmax.hpp, layer/batchnorm2d.hpp, layer/convolution.hpp,
# layer/maxpooling.hpp, layer/adaptive_avgpooling.hpp, layer/linear.hpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# OpenMP for batch parallelism in softmax and batchnorm
find_package(OpenMP COMPONENTS CXX)
if(OPENMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# AVX2 for SIMD intrinsics in softmax
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

add_executable(day11 main.cpp)
if(OPENMP_CXX_FOUND)
  target_link_libraries(day11 OpenMP::OpenMP_CXX)
endif()
