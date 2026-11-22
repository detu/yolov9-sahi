cmake_minimum_required(VERSION 3.18)
project(yolov9_sahi CXX)

# Default to Release: a Debug build makes inference many times slower.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GPU backend -- one of them, or none for a CPU-only build (the CPU backend is
# always there):
#   YOLOV9_CUDA   MNN with CUDA (NVIDIA)                       (default)
#   YOLOV9_VULKAN MNN with Vulkan (any GPU with a Vulkan driver)
# Turning Vulkan on switches the CUDA default off; use a separate build
# directory for each.
option(YOLOV9_VULKAN "Build MNN's Vulkan backend (needs only the Vulkan loader at run time)" OFF)
if(YOLOV9_VULKAN)
    set(_cuda_default OFF)
else()
    set(_cuda_default ON)
endif()
option(YOLOV9_CUDA "Build MNN's CUDA backend (needs the CUDA toolkit / nvcc)" ${_cuda_default})
if(YOLOV9_CUDA AND YOLOV9_VULKAN)
    message(FATAL_ERROR "YOLOV9_CUDA and YOLOV9_VULKAN cannot both be ON; choose one:\n"
                        "  CUDA:   -DYOLOV9_CUDA=ON\n"
                        "  Vulkan: -DYOLOV9_CUDA=OFF -DYOLOV9_VULKAN=ON\n"
                        "(use a separate build directory for each)")
endif()
option(YOLOV9_CUDA_NATIVE_ARCH "Compile MNN's CUDA kernels only for the GPU in this machine" ON)
# GPU architecture for the CUDA kernels, e.g. 8.9 (RTX 40xx / RTX Ada), 8.6 (RTX 30xx), 12.0 (RTX 50xx).
# Empty = detect the GPU in this machine while configuring.
set(YOLOV9_CUDA_ARCH "" CACHE STRING "CUDA compute capability to build for (empty = detect)")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MNN/CMakeLists.txt")
    message(FATAL_ERROR "MNN submodule missing: run 'git submodule update --init'")
endif()

# ---- MNN (submodule), built as its own project: its CMake files expect to be
#      the top-level project. Also builds MNNConvert, used to turn .onnx into .mnn.
set(_mnn_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5        # CMake 4 refuses MNN's old minimum otherwise
    -DCMAKE_POLICY_DEFAULT_CMP0146=OLD        # MNN's CUDA backend uses FindCUDA
    -DMNN_BUILD_SHARED_LIBS=ON
    -DMNN_SEP_BUILD=OFF                       # backends inside libMNN.so
    -DMNN_BUILD_CONVERTER=ON
    -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_TEST=OFF -DMNN_BUILD_BENCHMARK=OFF
    -DMNN_VULKAN=${YOLOV9_VULKAN}
    -DMNN_CUDA=${YOLOV9_CUDA})

if(YOLOV9_CUDA)
    # nvcc is often not on PATH (e.g. WSL2 without /usr/local/cuda/bin in PATH):
    # look in the standard install locations too.
    file(GLOB _cuda_dirs LIST_DIRECTORIES true "/usr/local/cuda-*")
    list(SORT _cuda_dirs COMPARE NATURAL ORDER DESCENDING)  # newest versioned install first
    find_program(YOLOV9_NVCC nvcc
                 HINTS "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin" /usr/local/cuda/bin
                 PATHS ${_cuda_dirs} PATH_SUFFIXES bin)
    if(NOT YOLOV9_NVCC)
        message(FATAL_ERROR "nvcc not found. Install the CUDA toolkit (cuda-toolkit), add its bin/ to PATH "
                            "or pass -DYOLOV9_NVCC=/path/to/nvcc; or build without CUDA: -DYOLOV9_CUDA=OFF")
    endif()
    get_filename_component(_cuda_root "${YOLOV9_NVCC}" DIRECTORY)
    get_filename_component(_cuda_root "${_cuda_root}" DIRECTORY)
    list(APPEND _mnn_args
         "-DCUDA_TOOLKIT_ROOT_DIR=${_cuda_root}"
         "-DCUDA_NVCC_EXECUTABLE=${YOLOV9_NVCC}"
         -DMNN_CUDA_NATIVE_ARCH=${YOLOV9_CUDA_NATIVE_ARCH})
    # GPU architecture: -DYOLOV9_CUDA_ARCH, else ask nvidia-smi. Without it MNN
    # runs its own detection and, if that fails, compiles for old GPUs
    # (sm_60...), which CUDA 13 rejects ("Unsupported gpu architecture").
    set(_cuda_arch "${YOLOV9_CUDA_ARCH}")
    if(NOT _cuda_arch)
        find_program(YOLOV9_NVIDIA_SMI nvidia-smi HINTS /usr/lib/wsl/lib /usr/bin)
        if(YOLOV9_NVIDIA_SMI)
            execute_process(COMMAND "${YOLOV9_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
                            OUTPUT_VARIABLE _smi OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _smi_rc
                            ERROR_QUIET)
            if(_smi_rc EQUAL 0 AND _smi MATCHES "^([0-9]+\\.[0-9]+)")
                set(_cuda_arch "${CMAKE_MATCH_1}")  # first GPU
                message(STATUS "yolov9_sahi: GPU compute capability ${_cuda_arch} (nvidia-smi)")
            endif()
        endif()
    endif()
    # MNN's bundled CUTLASS v2.9 has a typo (set_slice3x3) in templates nobody
    # instantiates; GCC 14+ rejects such template bodies (-Wtemplate-body).
    # Older GCC ignores this -Wno-... flag.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND _mnn_args "-DCUDA_NVCC_FLAGS=-Xcompiler=-Wno-template-body")
    endif()
    if(_cuda_arch)
        list(APPEND _mnn_args "-DCUDA_ARCHS=${_cuda_arch}")
    else()
        execute_process(COMMAND "${YOLOV9_NVCC}" --version OUTPUT_VARIABLE _nvcc_ver ERROR_QUIET)
        if(_nvcc_ver MATCHES "release ([0-9]+)\\." AND CMAKE_MATCH_1 GREATER_EQUAL 13)
            message(FATAL_ERROR "Cannot determine the GPU architecture (nvidia-smi not found or failed), and CUDA "
                                "13 cannot build MNN's fallback list of old GPUs. Pass it, e.g. -DYOLOV9_CUDA_ARCH=8.9")
        endif()
    endif()
    message(STATUS "yolov9_sahi: CUDA toolkit ${_cuda_root}")
endif()

include(ExternalProject)
set(MNN_BUILD_DIR "${CMAKE_BINARY_DIR}/mnn")
set(MNN_LIBRARY "${MNN_BUILD_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}MNN${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(MNN_CONVERTER "${MNN_BUILD_DIR}/MNNConvert")
ExternalProject_Add(mnn_build
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/MNN"
    BINARY_DIR "${MNN_BUILD_DIR}"
    CMAKE_ARGS ${_mnn_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target MNN MNNConvert
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${MNN_LIBRARY}" "${MNN_CONVERTER}"
    USES_TERMINAL_BUILD TRUE)

add_library(mnn_lib SHARED IMPORTED)
set_target_properties(mnn_lib PROPERTIES
    IMPORTED_LOCATION "${MNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/MNN/include")
add_dependencies(mnn_lib mnn_build)

# ---- The program
add_executable(yolov9_sahi main.cpp region_detector.cpp mnn_runner.cpp onnx_meta.cpp)
add_dependencies(yolov9_sahi mnn_build)
target_include_directories(yolov9_sahi PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov9_sahi PRIVATE mnn_lib ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_compile_definitions(yolov9_sahi PRIVATE
    YOLOV9_WITH_CUDA=$<BOOL:${YOLOV9_CUDA}>
    YOLOV9_WITH_VULKAN=$<BOOL:${YOLOV9_VULKAN}>
    YOLOV9_MNNCONVERT="${MNN_CONVERTER}")
# Find libMNN.so at run time without LD_LIBRARY_PATH
set_target_properties(yolov9_sahi PROPERTIES BUILD_RPATH "${MNN_BUILD_DIR}")

if(YOLOV9_CUDA)
    set(_gpu "CUDA")
elseif(YOLOV9_VULKAN)
    set(_gpu "Vulkan")
else()
    set(_gpu "no GPU")
endif()
message(STATUS "yolov9_sahi: MNN backends: ${_gpu} + CPU | ${CMAKE_BUILD_TYPE}")
