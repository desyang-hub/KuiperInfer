cmake_minimum_required(VERSION 3.16)
project(day13 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    src/tokenizer.cpp
    src/tokenizer.hpp
    src/parser.cpp
    src/parser.hpp
    src/evaluator.cpp
    src/evaluator.hpp
    main.cpp
)

add_executable(day13 ${SOURCES})

# Include directories
target_include_directories(day13 PRIVATE ${CMAKE_SOURCE_DIR})

# Compiler warnings
target_compile_options(day13 PRIVATE -Wall -Wextra)
