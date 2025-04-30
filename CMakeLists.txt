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

configure_file(
  ${PROJECT_SOURCE_DIR}/python/setup.py.in
  ${PROJECT_BINARY_DIR}/python/setup.py.in
  @ONLY)
file(GENERATE
  OUTPUT ${PROJECT_BINARY_DIR}/python/setup.py
  INPUT ${PROJECT_BINARY_DIR}/python/setup.py.in)

set(PYTHON_PROJECT_DIR ${PROJECT_BINARY_DIR}/python/ssapy)
file(GENERATE OUTPUT ${PYTHON_PROJECT_DIR}/__init__.py CONTENT "__version__ = \"${PROJECT_VERSION}\"\n")
file(GENERATE OUTPUT ${PYTHON_PROJECT_DIR}/ssapy/__init__.py CONTENT "")

add_custom_command(
  OUTPUT python/dist/timestamp
  COMMAND ${CMAKE_COMMAND} -E remove_directory dist
  ...
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_ssapy>   ${PYTHON_PROJECT}/ssapy
  COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel
  COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/python/dist/timestamp
  MAIN_DEPENDENCY
    python/setup.py.in
  WORKING_DIRECTORY python)

# Main Target
add_custom_target(python_package ALL
  DEPENDS
    python/dist/timestamp
  WORKING_DIRECTORY python)
