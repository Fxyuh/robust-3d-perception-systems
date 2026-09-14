cmake_minimum_required(VERSION 3.22)
project(centerpoint_deployment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

option(CPD_BUILD_TOOLS "Build standalone validation and benchmark tools" ON)

if(CPD_BUILD_TOOLS)
  add_executable(centerpoint_preprocess pillar_preprocess.cu)
  target_link_libraries(centerpoint_preprocess PRIVATE CUDA::cudart)
  target_compile_options(centerpoint_preprocess PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)
endif()

add_library(centerpoint_preprocess_lib STATIC pillar_preprocess.cu)
target_compile_definitions(centerpoint_preprocess_lib PRIVATE CPD_NO_MAIN)
target_link_libraries(centerpoint_preprocess_lib PUBLIC CUDA::cudart)
target_include_directories(centerpoint_preprocess_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(centerpoint_preprocess_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)

get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  ABSOLUTE)
set(TENSORRT_ROOT "" CACHE PATH "Optional TensorRT installation prefix")
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInferRuntime.h
  HINTS
    "${REPOSITORY_ROOT}/work_dirs/centerpoint_deployment/deps/TensorRT/include"
    "${TENSORRT_ROOT}"
    "$ENV{TENSORRT_ROOT}"
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY
  NAMES nvinfer libnvinfer.so.11
  HINTS "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}"
  PATH_SUFFIXES lib lib64 tensorrt_libs)

if(EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferRuntime.h" AND
   EXISTS "${TENSORRT_LIBRARY}")
  add_library(centerpoint_detector STATIC centerpoint_detector.cpp)
  target_include_directories(centerpoint_detector
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(centerpoint_detector PUBLIC
    centerpoint_preprocess_lib CUDA::cudart ${TENSORRT_LIBRARY})
  target_compile_options(centerpoint_detector PRIVATE
    -O3 -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
  get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
  target_link_directories(centerpoint_detector PUBLIC ${TENSORRT_LIBRARY_DIR})

  if(CPD_BUILD_TOOLS)
    add_executable(centerpoint_infer centerpoint_infer.cpp)
    target_link_libraries(centerpoint_infer PRIVATE centerpoint_detector)
    target_compile_options(centerpoint_infer PRIVATE
      -O3 -Wall -Wextra -Wpedantic)
    set_target_properties(centerpoint_infer PROPERTIES
      BUILD_RPATH "${TENSORRT_LIBRARY_DIR}")

    find_path(NVML_INCLUDE_DIR nvml.h
      HINTS ${CUDAToolkit_INCLUDE_DIRS}
      PATH_SUFFIXES include)
    find_library(NVML_LIBRARY
      NAMES nvidia-ml libnvidia-ml.so.1
      HINTS /usr/lib/wsl/lib)
    if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
      get_filename_component(NVML_LIBRARY_DIR "${NVML_LIBRARY}" DIRECTORY)
      add_executable(centerpoint_dataset_eval centerpoint_dataset_eval.cpp)
      target_include_directories(centerpoint_dataset_eval PRIVATE
        ${NVML_INCLUDE_DIR})
      target_link_libraries(centerpoint_dataset_eval PRIVATE
        centerpoint_detector Threads::Threads ${NVML_LIBRARY})
      target_compile_options(centerpoint_dataset_eval PRIVATE
        -O3 -Wall -Wextra -Wpedantic)
      set_target_properties(centerpoint_dataset_eval PROPERTIES
        BUILD_RPATH "${TENSORRT_LIBRARY_DIR};${NVML_LIBRARY_DIR}")
    else()
      message(WARNING
        "NVML headers/library not found; centerpoint_dataset_eval disabled")
    endif()
  endif()
else()
  message(WARNING
    "TensorRT headers/library not found; centerpoint_infer will not be built")
endif()
