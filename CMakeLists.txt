cmake_minimum_required(VERSION 2.8)

project(openmoc)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonInterp)
find_package(PythonLibs)
execute_process(COMMAND
  ${PYTHON_EXECUTABLE} -c
  "import site; print(site.USER_SITE)"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND
  ${PYTHON_EXECUTABLE} -c
  "import numpy as np; print(np.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(cpp_source
  src/Cell.cpp
  src/Geometry.cpp
  src/LocalCoords.cpp
  src/log.cpp
  src/Material.cpp
  src/MOCKernel.cpp
  src/Point.cpp
  src/Quadrature.cpp
  src/ExpEvaluator.cpp
  src/Solver.cpp
  src/CPUSolver.cpp
  src/Surface.cpp
  src/Timer.cpp
  src/Track.cpp
  src/TrackGenerator.cpp
  src/TrackTraversingAlgorithms.cpp
  src/TraverseTracks.cpp
  src/Universe.cpp
  src/Vector.cpp
  src/Matrix.cpp
  src/Cmfd.cpp
  src/linalg.cpp)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CMAKE_SWIG_FLAGS "")
set(CMAKE_CXX_FLAGS "-DFP_PRECISION=double -DDOUBLE -DGCC -DVEC_LENGTH=8 -DVEC_ALIGNMENT=16 -DOPENMP -DSWIG -ffast-math -fopenmp -std=c++11")

add_library(${PROJECT_NAME}_cpp SHARED ${cpp_source})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/openmoc)
set_source_files_properties(openmoc/openmoc.i PROPERTIES
  CPLUSPLUS ON
  SWIG_FLAGS "-keyword;-DDOUBLE;-DFP_PRECISION=double;-DGCC"
  )
swig_add_module(${PROJECT_NAME} python openmoc/openmoc.i)
swig_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES} ${PROJECT_NAME}_cpp)

install(TARGETS _${PROJECT_NAME} ${PROJECT_NAME}_cpp
  DESTINATION ${PYTHON_SITE_PACKAGES})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/openmoc
  DESTINATION ${PYTHON_SITE_PACKAGES})

#===============================================================================
# Regression tests
#===============================================================================

# This allows for dashboard configuration
include(CTest)

# Get a list of all the tests to run
file(GLOB_RECURSE TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py)

# Loop through all the tests
foreach(test ${TESTS})

  # Get test information
  get_filename_component(TEST_NAME ${test} NAME)
  get_filename_component(TEST_PATH ${test} PATH)

  add_test(NAME ${TEST_NAME}
    WORKING_DIRECTORY ${TEST_PATH}
    COMMAND ${PYTHON_EXECUTABLE} ${TEST_NAME}
    )

endforeach(test)
