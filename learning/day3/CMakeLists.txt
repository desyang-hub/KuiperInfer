cmake_minimum_required(VERSION 3.16)
project(day3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only project: include/ contains tensor.hpp and tensor_util.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(day3 main.cpp)
