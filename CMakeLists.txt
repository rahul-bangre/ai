cmake_minimum_required(VERSION 3.16)
project(qwen_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(qwen_infer
    src/tensor.cpp
    src/weights.cpp
    src/main.cpp
)

target_include_directories(qwen_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(qwen_infer PRIVATE -Wall -Wextra -Wpedantic)
