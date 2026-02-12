project(mex)

################################################################################
# Dependencies and options
################################################################################
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Look for Python
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Download nanobind
include(DownloadProject)
download_project(
        PROJ         "nanobind"
        SOURCE_DIR   "${CURVE_EXTERNAL}/nanobind"
        DOWNLOAD_DIR "${CURVE_EXTERNAL}/.cache/nanobind"
        GIT_REPOSITORY https://github.com/wjakob/nanobind
)

# Add nanobind
list(APPEND CMAKE_PREFIX_PATH "${CURVE_EXTERNAL}/nanobind/cmake")
find_package(nanobind CONFIG REQUIRED)

################################################################################
# Build MEX files
################################################################################
file(GLOB SOURCE_FILES *.cpp)

nanobind_add_module(pycurve pycurve.cpp)

target_link_libraries(pycurve PUBLIC curve-energy_static )
