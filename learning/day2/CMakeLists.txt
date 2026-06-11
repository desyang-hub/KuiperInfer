cmake_minimum_required(VERSION 3.16)
project(day2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 创建可执行文件（tensor.hpp 是头文件实现，只需 main.cpp）
add_executable(day2 main.cpp)
