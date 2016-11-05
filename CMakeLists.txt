cmake_minimum_required(VERSION 2.8)

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CMAKE_SWIG_FLAGS "")
set(CMAKE_CXX_FLAGS "-DFP_PRECISION=double -DDOUBLE -DGCC -DVEC_LENGTH=8 -DVEC_ALIGNMENT=16 -DOPENMP -DSWIG -ffast-math -fopenmp -std=c++11 -fpic")

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

add_library(openmoc SHARED ${cpp_source})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/openmoc)

set_source_files_properties(openmoc/openmoc.i PROPERTIES
  CPLUSPLUS ON
  SWIG_FLAGS "-keyword;-DDOUBLE;-DFP_PRECISION=double;-DGCC"
  )
swig_add_module(openmoc python openmoc/openmoc.i)
swig_link_libraries(openmoc ${PYTHON_LIBRARIES} openmoc)

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
