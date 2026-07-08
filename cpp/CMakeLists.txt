cmake_minimum_required(VERSION 3.16)
project(mnist_cpp_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 다운로드한 ONNX Runtime C++ (third_party/onnxruntime)
set(ORT_DIR ${CMAKE_SOURCE_DIR}/../third_party/onnxruntime)

add_executable(bench src/main.cpp src/onnx_infer.cpp)
target_include_directories(bench PRIVATE include ${ORT_DIR}/include)
target_link_libraries(bench PRIVATE ${ORT_DIR}/lib/libonnxruntime.so)
set_target_properties(bench PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
