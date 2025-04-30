cmake_minimum_required(VERSION 3.15)

# Set project
project(ssapy VERSION 1.0 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: Hide unused symbols unless explicitly exported
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Required for linking static libs into shared modules
set(CMAKE_MACOSX_RPATH 1)

# Find Python with development headers and module support
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find pybind11 (assumes you vendored or added as submodule)
add_subdirectory(pybind11)

# Optional: Add your own library to link
add_library(ssapy STATIC
    src/ssapy.cpp
    include/ssapy.h
)
target_include_directories(ssapy PUBLIC include)

# Include the Python-specific logic
include(python.cmake)