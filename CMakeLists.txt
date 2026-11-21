cmake_minimum_required(VERSION 3.18)
project(yolov9_sahi CXX)

# Default to Release: a Debug build makes ncnn many times slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ncnn: built from the ncnn/ submodule by default, or use an installed one.
#   -DSAHI_SYSTEM_NCNN=ON  use find_package(ncnn) (set ncnn_DIR to <prefix>/lib/cmake/ncnn)
#   -DSAHI_VULKAN=ON       build ncnn with Vulkan GPU support (run with --gpu).
#                          Needs the ncnn/glslang submodule; very slow under WSL2's Dozen driver.
option(SAHI_SYSTEM_NCNN "Use an installed ncnn instead of the ncnn/ submodule" OFF)
option(SAHI_VULKAN "Build ncnn with Vulkan GPU support" OFF)

if(SAHI_SYSTEM_NCNN)
    find_package(ncnn REQUIRED)
else()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/CMakeLists.txt")
        message(FATAL_ERROR "ncnn submodule missing: run 'git submodule update --init --recursive'")
    endif()
    set(NCNN_VULKAN          ${SAHI_VULKAN} CACHE BOOL "" FORCE)
    set(NCNN_OPENMP          ON  CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
    set(NCNN_PYTHON          OFF CACHE BOOL "" FORCE)
    set(NCNN_INSTALL_SDK     OFF CACHE BOOL "" FORCE)
    set(NCNN_SHARED_LIB      OFF CACHE BOOL "" FORCE)
    if(SAHI_VULKAN AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/glslang/CMakeLists.txt")
        message(FATAL_ERROR "SAHI_VULKAN=ON needs ncnn's glslang submodule: run 'git submodule update --init --recursive'")
    endif()
    add_subdirectory(ncnn EXCLUDE_FROM_ALL)
endif()
if(SAHI_SYSTEM_NCNN)
    set(_ncnn_src "installed")
else()
    set(_ncnn_src "submodule")
endif()
message(STATUS "yolov9_sahi: ncnn ${_ncnn_src} | Vulkan ${SAHI_VULKAN} | ${CMAKE_BUILD_TYPE}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(yolov9_sahi main.cpp)

target_include_directories(yolov9_sahi PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolov9_sahi PRIVATE
    ncnn
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)
