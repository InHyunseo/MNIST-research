cmake_minimum_required(VERSION 3.16)
project(static_sequence_inference LANGUAGES CXX)

set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnxruntime")

add_library(static_sequence_core
  src/csv_logger.cpp
  src/onnx_sequence_recognizer.cpp
  src/sequence_types.cpp
)
target_compile_features(static_sequence_core PUBLIC cxx_std_17)
target_include_directories(static_sequence_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(static_sequence_core SYSTEM
  PUBLIC
    "${ORT_DIR}/include"
)
target_link_libraries(static_sequence_core PRIVATE "${ORT_DIR}/lib/libonnxruntime.so")
target_compile_options(static_sequence_core PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
  -Wnon-virtual-dtor
  -Wold-style-cast
)
set_target_properties(static_sequence_core PROPERTIES
  BUILD_RPATH "${ORT_DIR}/lib"
)

add_executable(static_sequence_bench src/main.cpp)
target_compile_features(static_sequence_bench PRIVATE cxx_std_17)
target_link_libraries(static_sequence_bench PRIVATE static_sequence_core)
target_compile_options(static_sequence_bench PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
)
