cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)  # set VERSION
cmake_policy(SET CMP0063 NEW)  # Compile the static lib with hidden visibility.

project(_ssapy CXX)
# set(CMAKE_VERBOSE_MAKEFILE True)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)
add_subdirectory(pybind11)
set(PYSRC_FILES
  pysrc/ssapy.cpp
)
set(SRC_FILES
  src/ssapy.cpp
)
set(CMAKE_POSITION_INDEPENDENT_CODE True)
add_library(ssapy STATIC ${SRC_FILES})
pybind11_add_module(_ssapy ${PYSRC_FILES})
target_link_libraries(_ssapy PUBLIC ssapy)
