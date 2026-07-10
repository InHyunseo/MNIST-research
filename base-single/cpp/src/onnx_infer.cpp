#include "onnx_infer.hpp"

#include <algorithm>

constexpr int kPixels = 1 * 28 * 28;

OnnxInfer::OnnxInfer(const std::string& model_path, int threads)
    : OnnxInfer(model_path, OnnxInferOptions{threads}) {}

OnnxInfer::OnnxInfer(const std::string& model_path, const OnnxInferOptions& options)
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx_infer"),
      session_(nullptr),
      mem_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
  so_.SetIntraOpNumThreads(options.threads);
  so_.SetInterOpNumThreads(1);
  if (options.set_graph_optimization) {
    so_.SetGraphOptimizationLevel(options.graph_optimization
                                      ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                      : GraphOptimizationLevel::ORT_DISABLE_ALL);
  }
  if (options.set_memory_reuse) {
    if (options.memory_reuse) {
      so_.EnableCpuMemArena();
      so_.EnableMemPattern();
    } else {
      so_.DisableCpuMemArena();
      so_.DisableMemPattern();
    }
  }
  session_ = Ort::Session(env_, model_path.c_str(), so_);

  Ort::AllocatorWithDefaultOptions alloc;
  in_name_ = session_.GetInputNameAllocated(0, alloc).get();
  out_name_ = session_.GetOutputNameAllocated(0, alloc).get();
}

int OnnxInfer::predict(const uint8_t* image) {
  const char* in_names[] = {in_name_.c_str()};
  const char* out_names[] = {out_name_.c_str()};
  Ort::Value input = Ort::Value::CreateTensor<uint8_t>(
      mem_, const_cast<uint8_t*>(image), kPixels, shape_.data(), shape_.size());
  auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
  float* logits = out[0].GetTensorMutableData<float>();
  return static_cast<int>(std::max_element(logits, logits + 10) - logits);
}
