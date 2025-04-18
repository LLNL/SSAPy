cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0063 NEW)

project(_ssapy VERSION 1.0 LANGUAGES CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MACOSX_RPATH 1)

# Output directories
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Include pybind11 via FetchContent
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(_ssapy src/ssapy.cpp)

install(TARGETS _ssapy
    LIBRARY DESTINATION ssapy
)

install(TARGETS _ssapy
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)