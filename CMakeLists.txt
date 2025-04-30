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

# Python extension target
pybind11_add_module(_ssapy
    pysrc/ssapy.cpp        # your pybind11 bindings file
)

# Link your custom library to the Python module
target_link_libraries(_ssapy PRIVATE ssapy)

# Place the shared object directly into the ssapy Python package
set_target_properties(_ssapy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/ssapy"
)

# Optional: install step (not needed for pip wheels)
install(TARGETS _ssapy DESTINATION ssapy)
