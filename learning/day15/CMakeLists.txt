cmake_minimum_required(VERSION 3.16)
project(day15 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# YOLOv5 Object Detection — sigmoid, flatten/view, bbox decoding, NMS, coordinate scaling
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(day15
  main.cpp
)
