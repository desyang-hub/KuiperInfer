cmake_minimum_required(VERSION 3.16)
project(day17 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Day 17: SIMD 向量化编程 — AVX2/SSE 指令集加速深度学习算子
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 使用本机最高指令集，启用 SIMD
add_executable(day17
  main.cpp
)

target_compile_options(day17 PRIVATE -march=native -mfpmath=sse -Ofast)
target_link_options(day17 PRIVATE -lpthread)
