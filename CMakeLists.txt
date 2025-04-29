cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0063 NEW)

project(ssapy VERSION 1.0 LANGUAGES CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MACOSX_RPATH 1)

# Output directory
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Bring in pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Include your headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build static C++ logic (optional)
add_library(ssapy STATIC src/ssapy.cpp)
set_target_properties(ssapy PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Build Python extension module
pybind11_add_module(_ssapy pysrc/ssapy.cpp)
target_link_libraries(_ssapy PUBLIC ssapy)

# Install into the ssapy/ Python package directory
install(TARGETS _ssapy
    LIBRARY DESTINATION ssapy
)