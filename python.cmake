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

# Generate setup.py file
configure_file(
  ${PROJECT_SOURCE_DIR}/python/setup.py.in
  ${PROJECT_BINARY_DIR}/setup.py
  @ONLY
)

# Generate Python package structure
set(PYTHON_PROJECT_DIR ${PROJECT_BINARY_DIR}/ssapy)
file(GENERATE OUTPUT ${PYTHON_PROJECT_DIR}/__init__.py CONTENT "__version__ = \"${PROJECT_VERSION}\"\n")
file(GENERATE OUTPUT ${PYTHON_PROJECT_DIR}/ssapy/__init__.py CONTENT "")

# Build Python package
add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/dist/timestamp
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_BINARY_DIR}/dist
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_ssapy> ${PYTHON_PROJECT_DIR}/ssapy
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_BINARY_DIR}/setup.py bdist_wheel
  COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/dist/timestamp
  MAIN_DEPENDENCY ${PROJECT_BINARY_DIR}/setup.py
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# Main Target
add_custom_target(python_package ALL
  DEPENDS ${PROJECT_BINARY_DIR}/dist/timestamp
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)