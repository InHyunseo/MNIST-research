// ONNX Runtime 추론 엔진.
// 모델 graph가 normalization을 포함하므로, 호출부는 raw uint8 784바이트를 넣고
// 클래스(0..9)를 돌려받는다.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <string>

struct OnnxInferOptions {
  int threads = 1;
  bool set_graph_optimization = true;
  bool graph_optimization = false;
  bool set_memory_reuse = true;
  bool memory_reuse = false;
};

class OnnxInfer {
 public:
  OnnxInfer(const std::string& model_path, int threads);
  OnnxInfer(const std::string& model_path, const OnnxInferOptions& options);
  int predict(const uint8_t* image);   // image: 784 uint8 -> class 0..9

 private:
  Ort::Env env_;
  Ort::SessionOptions so_;
  Ort::Session session_;
  Ort::MemoryInfo mem_;
  std::string in_name_, out_name_;
  std::array<int64_t, 4> shape_{1, 1, 28, 28};
};
