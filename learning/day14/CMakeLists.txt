cmake_minimum_required(VERSION 3.16)
project(day14 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expression layer integrates parser (day12/13) with real Tensor system (day11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(day14
  src/main.cpp
  src/tokenizer.cpp
  src/parser.cpp
  src/layer/expression.cpp
)
