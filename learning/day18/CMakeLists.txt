cmake_minimum_required(VERSION 3.16)
project(day18 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP 多线程并行 — SIMD + 多核全面加速
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(day18
  main.cpp
)
target_link_libraries(day18 OpenMP::OpenMP_CXX)
