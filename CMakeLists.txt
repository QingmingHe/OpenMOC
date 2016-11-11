cmake_minimum_required(VERSION 2.8)
project(openmoc CXX)

#===============================================================================
# Command line options
#===============================================================================
option(double_fp "Enable double precision" OFF)
option(root_install "Install openmoc for root user" OFF)
option(debug "Enable debug mode" OFF)
option(profile "Enable profile mode" OFF)
option(cuda "Compile openmoc.cuda module with nvcc compiler" OFF)

#===============================================================================
# Find SWIG package
#===============================================================================
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

#===============================================================================
# Find Python package and obtain Python related variables
#===============================================================================
find_package(PythonInterp)
find_package(PythonLibs)
if(${root_install})
  execute_process(COMMAND
    ${PYTHON_EXECUTABLE} -c
    "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  execute_process(COMMAND
    ${PYTHON_EXECUTABLE} -c
    "import site; print(site.USER_SITE)"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
execute_process(COMMAND
  ${PYTHON_EXECUTABLE} -c
  "import numpy as np; print(np.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

#===============================================================================
# CXX source files
#===============================================================================
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)

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

elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)

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
    src/Cmfd.cpp
    src/Vector.cpp
    src/Matrix.cpp
    src/linalg.cpp)

else(CMAKE_CXX_COMPILER_ID STREQUAL Intel)

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
    src/VectorizedSolver.cpp
    src/Surface.cpp
    src/Timer.cpp
    src/Track.cpp
    src/TrackGenerator.cpp
    src/TrackTraversingAlgorithms.cpp
    src/TraverseTracks.cpp
    src/Universe.cpp
    src/Cmfd.cpp
    src/Vector.cpp
    src/Matrix.cpp
    src/linalg.cpp)

else(CMAKE_CXX_COMPILER_ID STREQUAL XL)

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
    src/Cmfd.cpp
    src/Vector.cpp
    src/Matrix.cpp
    src/linalg.cpp)

endif()

#===============================================================================
# Include directories
#===============================================================================
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

#===============================================================================
# CXX and SWIG flags
#===============================================================================
set(CMAKE_SWIG_FLAGS "")
set(vec_length 8)
set(vec_alignment 16)
list(APPEND swigflags -keyword)

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)

  list(APPEND cxxflags -ffast-math -O3 -fopenmp -std=c++11 -fpic)
  list(APPEND cxxflags -DGCC -DOPENMP -DSWIG -DVEC_LENGTH=${vec_length}
    -DVEC_ALIGNMENT=${vec_alignment})
  list(APPEND swigflags -DGCC)
  list(APPEND ldflags -fopenmp)

  if(double_fp)
    list(APPEND cxxflags -DDOUBLE -DFP_PRECISION=double)
    list(APPEND swigflags -DDOUBLE -DFP_PRECISION=double)
  else()
    list(APPEND cxxflags -DSINGLE -DFP_PRECISION=float)
    list(APPEND swigflags -DSINGLE -DFP_PRECISION=float)
  endif()

  if(debug)
    list(APPEND cxxflags -g)
  endif()

  if(profile)
    list(APPEND cxxflags -pg)
    if(NOT debug)
      list(APPEND cxxflags -g)
    endif()
  endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)

  list(APPEND cxxflags -ffast-math -O3 -fopenmp -std=c++11 -fpic -Qunused-arguments
    -Wno-deprecated-register -Wno-parentheses-equality)
  list(APPEND cxxflags -DCLANG -DOPENMP -DSWIG -DVEC_LENGTH=${vec_length}
    -DVEC_ALIGNMENT=${vec_alignment})
  list(APPEND swigflags -DCLANG)
  list(APPEND ldflags -fopenmp)

  if(double_fp)
    list(APPEND cxxflags -DDOUBLE -DFP_PRECISION=double)
    list(APPEND swigflags -DDOUBLE -DFP_PRECISION=double)
  else()
    list(APPEND cxxflags -DSINGLE -DFP_PRECISION=float)
    list(APPEND swigflags -DSINGLE -DFP_PRECISION=float)
  endif()

  if(debug)
    list(APPEND cxxflags -g)
  endif()

  if(profile)
    list(APPEND cxxflags -pg)
    if(NOT debug)
      list(APPEND cxxflags -g)
    endif()
  endif()

else(CMAKE_CXX_COMPILER_ID STREQUAL Intel)

  list(APPEND cxxflags -O3 -fast --ccache-skip -openmp -xhost -std=c++11 -fpic
    -openmp-report -vec-report)
  list(APPEND cxxflags -DICPC -DOPENMP -DSWIG -DVEC_LENGTH=${vec_length}
    -DVEC_ALIGNMENT=${vec_alignment})
  list(APPEND swigflags -DICPC)
  list(APPEND ldflags -openmp)

  if(double_fp)
    list(APPEND cxxflags -DDOUBLE -DFP_PRECISION=double)
    list(APPEND swigflags -DDOUBLE -DFP_PRECISION=double)
  else()
    list(APPEND cxxflags -DSINGLE -DFP_PRECISION=float)
    list(APPEND swigflags -DSINGLE -DFP_PRECISION=float)
  endif()

  if(debug)
    list(APPEND cxxflags -g)
  endif()

  if(profile)
    list(APPEND cxxflags -pg)
    if(NOT debug)
      list(APPEND cxxflags -g)
    endif()
  endif()

else(CMAKE_CXX_COMPILER_ID STREQUAL XL)

  list(APPEND cxxflags -O2 -qarch=qp -qreport -qsimd=auto -qtune=qp -qunroll=auto
    -qsmp=omp -qpic)
  list(APPEND cxxflags -DBGXLC -DOPENMP -DSWIG -DVEC_LENGTH=${vec_length}
    -DVEC_ALIGNMENT=${vec_alignment})
  list(APPEND swigflags -DBGXLC)
  list(APPEND ldflags -qmkshrobj -R/soft/compilers/ibmcmp-may2013/lib64/bg/bglib64)

  if(double_fp)
    list(APPEND cxxflags -DDOUBLE -DFP_PRECISION=double)
    list(APPEND swigflags -DDOUBLE -DFP_PRECISION=double)
  else()
    list(APPEND cxxflags -DSINGLE -DFP_PRECISION=float)
    list(APPEND swigflags -DSINGLE -DFP_PRECISION=float)
  endif()

  if(debug)
    list(APPEND cxxflags -g)
  endif()

  if(profile)
    list(APPEND cxxflags -pg)
    if(NOT debug)
      list(APPEND cxxflags -g)
    endif()
  endif()

endif()

#===============================================================================
# Add SWIG module
#===============================================================================

# Set linker flags
string(REPLACE ";" " " ldflags "${ldflags}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${ldflags}")

# Set SWIG flags
set_source_files_properties(${PROJECT_NAME}/${PROJECT_NAME}.i
  PROPERTIES
  CPLUSPLUS ON
  SWIG_FLAGS "${swigflags}"
  )

# Make openmoc directory under current binary path
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME})

# Add openmoc module
swig_add_module(${PROJECT_NAME} python
  ${PROJECT_NAME}/${PROJECT_NAME}.i ${cpp_source})

# Link to Python libraries
swig_link_libraries(${PROJECT_NAME} ${PythonLibs})

# Set target CXX compile flags
if (CMAKE_VERSION VERSION_LESS 2.8.12)
  string(REPLACE ";" " " cxxflags "${cxxflags}")
  set_property(TARGET _${PROJECT_NAME} PROPERTY COMPILE_FLAGS "${cxxflags}")
else()
  target_compile_options(_${PROJECT_NAME} PUBLIC ${cxxflags})
endif()

#===============================================================================
# Install Python packages
#===============================================================================
install(TARGETS _${PROJECT_NAME}
  DESTINATION ${PYTHON_SITE_PACKAGES})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}
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
