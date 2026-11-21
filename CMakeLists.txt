cmake_minimum_required(VERSION 3.18)
project(yolov9_sahi CXX)

# Default to Release: a Debug build makes inference many times slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inference backend:
#   ORT_CUDA (default)  ONNX Runtime with the CUDA execution provider; tile
#                       preprocessing and decoding run as CUDA kernels (nvcc).
#                       Needs the ONNX Runtime GPU package (-DONNXRUNTIME_ROOT=...),
#                       the CUDA toolkit and cuDNN 9.
#   ORT_CPU             ONNX Runtime on the CPU (ONNX Runtime CPU or GPU package).
#   NCNN                ncnn from the ncnn/ submodule (.param/.bin models), CPU or Vulkan.
set(SAHI_BACKEND "ORT_CUDA" CACHE STRING "Inference backend: ORT_CUDA, ORT_CPU or NCNN")
set_property(CACHE SAHI_BACKEND PROPERTY STRINGS ORT_CUDA ORT_CPU NCNN)
string(TOUPPER "${SAHI_BACKEND}" SAHI_BACKEND)
if(NOT SAHI_BACKEND MATCHES "^(ORT_CUDA|ORT_CPU|NCNN)$")
    message(FATAL_ERROR "SAHI_BACKEND must be ORT_CUDA, ORT_CPU or NCNN (got '${SAHI_BACKEND}')")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(SAHI_BACKEND STREQUAL "NCNN")
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

    add_executable(yolov9_sahi main.cpp)
    target_compile_definitions(yolov9_sahi PRIVATE SAHI_NCNN=1)
    target_link_libraries(yolov9_sahi PRIVATE ncnn)
    message(STATUS "yolov9_sahi: backend NCNN | Vulkan ${SAHI_VULKAN} | ${CMAKE_BUILD_TYPE}")
else()
    # ONNX Runtime: the unpacked release package (the GPU package for ORT_CUDA),
    # with include/ and lib/ inside.
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime package directory (include/, lib/)")
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
              HINTS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
                    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
    find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64")
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime not found: pass -DONNXRUNTIME_ROOT=<unpacked onnxruntime package> "
                            "(the directory containing include/ and lib/)")
    endif()
    get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)

    set(_sources main.cpp ort_detector.cpp)
    if(SAHI_BACKEND STREQUAL "ORT_CUDA")
        if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_cuda.so")
            message(WARNING "${ONNXRUNTIME_LIB_DIR} has no libonnxruntime_providers_cuda.so: "
                            "this looks like the CPU package; ORT_CUDA needs the GPU package")
        endif()
        # nvcc builds the kernels for the GPU in this machine (CMake >= 3.24),
        # else for common NVIDIA architectures; override with -DCMAKE_CUDA_ARCHITECTURES=89.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
                set(CMAKE_CUDA_ARCHITECTURES native)
            else()
                set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
            endif()
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        list(APPEND _sources gpu_ops_cuda.cu)
    else()
        list(APPEND _sources gpu_ops_cpu.cpp)
    endif()

    add_executable(yolov9_sahi ${_sources})
    target_include_directories(yolov9_sahi PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(yolov9_sahi PRIVATE "${ONNXRUNTIME_LIBRARY}")
    # Find libonnxruntime*.so at run time without LD_LIBRARY_PATH
    set_target_properties(yolov9_sahi PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
    if(SAHI_BACKEND STREQUAL "ORT_CUDA")
        target_compile_definitions(yolov9_sahi PRIVATE SAHI_ORT=1 SAHI_ORT_CUDA=1 SAHI_ORT_DEVICE="CUDA")
        target_link_libraries(yolov9_sahi PRIVATE CUDA::cudart)
        message(STATUS "yolov9_sahi: backend ORT_CUDA | ONNX Runtime ${ONNXRUNTIME_LIB_DIR} | "
                       "CUDA ${CUDAToolkit_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES} | ${CMAKE_BUILD_TYPE}")
    else()
        target_compile_definitions(yolov9_sahi PRIVATE SAHI_ORT=1 SAHI_ORT_CPU=1 SAHI_ORT_DEVICE="CPU")
        message(STATUS "yolov9_sahi: backend ORT_CPU | ONNX Runtime ${ONNXRUNTIME_LIB_DIR} | ${CMAKE_BUILD_TYPE}")
    endif()
endif()

target_include_directories(yolov9_sahi PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov9_sahi PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
