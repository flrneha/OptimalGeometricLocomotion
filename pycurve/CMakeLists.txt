cmake_minimum_required(VERSION 3.18...3.22)
set(CMAKE_BUILD_TYPE_INIT Release)
project(curve-energy)


################################################################################
# General setup
################################################################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

set(CMAKE_CXX_STANDARD 14)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") # Path with CMake scripts
set(CURVE_EXTERNAL "${CMAKE_CURRENT_LIST_DIR}/external") # Path for downloads

# produce position independent code for ALL targets (often necessary for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
################################################################################
# Options
################################################################################
option(BUILD_EXAMPLES "Build examples" ON)
##option(BUILD_MEX "Build .mex files for MATLAB" OFF)
option(BUILD_PYTHON "Build library for Python " OFF)

################################################################################
# Dependencies
################################################################################
# Eigen, adapted from libigl
find_package(Eigen3 QUIET)

if(NOT TARGET Eigen3::Eigen)
  include(DownloadProject)
  message(STATUS "CURVE_EXTERNAL ${CURVE_EXTERNAL}")
  # Download Eigen
  download_project(
			PROJ         "eigen"
			SOURCE_DIR   "${CURVE_EXTERNAL}/eigen"
			DOWNLOAD_DIR "${CURVE_EXTERNAL}/.cache/eigen"
			GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
			GIT_TAG 3.3.9
		)
		
  
  # Create Eigen targets  
  add_library(curve_eigen INTERFACE)
  target_include_directories(curve_eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CURVE_EXTERNAL}/eigen>
    $<INSTALL_INTERFACE:include>
  )
  set_property(TARGET curve_eigen PROPERTY EXPORT_NAME Eigen3::Eigen)
  add_library(Eigen3::Eigen ALIAS curve_eigen)
endif()

# libigl is needed in the examples for general mesh processing
# if (BUILD_EXAMPLES OR BUILD_MEX)
#   if (BUILD_MEX)
#     option(LIBIGL_WITH_MATLAB "Use Matlab" ON)
#   endif ()
#   include(libigl)
# endif ()

################################################################################
# Build library
################################################################################
# shared
add_library(curve-energy SHARED
            src/auxiliary.cpp
            src/membrane_energy.cpp
            src/bending_energy.cpp
            src/membrane_gradient.cpp
            src/bending_gradient.cpp
            src/membrane_hessian.cpp
            src/curve_energy.cpp
            src/bending_hessian.cpp)
target_include_directories(curve-energy PUBLIC include)
target_link_libraries(curve-energy PUBLIC Eigen3::Eigen)

# static
add_library(curve-energy_static STATIC
            src/auxiliary.cpp
            src/membrane_energy.cpp
            src/bending_energy.cpp
            src/membrane_gradient.cpp
            src/bending_gradient.cpp
            src/membrane_hessian.cpp
            src/curve_energy.cpp
            src/bending_hessian.cpp)
target_include_directories(curve-energy_static PUBLIC include)
target_link_libraries(curve-energy_static PUBLIC Eigen3::Eigen)

################################################################################
# Additional builds
################################################################################
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

# if (BUILD_MEX)
#   add_subdirectory(mex)
# endif ()

if (BUILD_PYTHON)
  add_subdirectory(python)
endif ()