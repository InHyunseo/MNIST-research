#include "static_sequence/onnx_sequence_recognizer.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace static_sequence {
namespace {

void validate_dimensions(const std::vector<std::int64_t>& actual,
                         const std::array<std::int64_t, 4>& expected,
                         const std::string& tensor_name) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(tensor_name + " rank mismatch");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (index == 0 && actual[index] == -1) {
      continue;
    }
    if (actual[index] != expected[index]) {
      throw std::runtime_error(tensor_name + " shape mismatch");
    }
  }
}

}  // namespace

OnnxSequenceRecognizer::OnnxSequenceRecognizer(
    const std::filesystem::path& model_path, const int threads)
    : environment_(ORT_LOGGING_LEVEL_WARNING, "static_sequence"),
      session_(nullptr),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
  if (threads <= 0) {
    throw std::invalid_argument("threads must be positive");
  }
  session_options_.SetIntraOpNumThreads(threads);
  session_options_.SetInterOpNumThreads(1);
  session_ = Ort::Session(environment_, model_path.c_str(), session_options_);

  Ort::AllocatorWithDefaultOptions allocator;
  input_name_ = session_.GetInputNameAllocated(0, allocator).get();
  output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
  validate_model_contract();
}

void OnnxSequenceRecognizer::validate_model_contract() {
  if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
    throw std::runtime_error("model must expose exactly one input and one output");
  }

  // Keep TypeInfo alive while using its non-owning tensor shape view.
  const Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
  const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
  if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    throw std::runtime_error("model input must be uint8");
  }
  validate_dimensions(input_info.GetShape(), input_shape_, "model input");

  const Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
  const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
  if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("model output must be float32 logits");
  }
  const std::array<std::int64_t, 3> expected_output{1, 3, 10};
  const std::vector<std::int64_t> output_shape = output_info.GetShape();
  if (output_shape.size() != expected_output.size()) {
    throw std::runtime_error("model output rank mismatch");
  }
  for (std::size_t index = 0; index < expected_output.size(); ++index) {
    if (index == 0 && output_shape[index] == -1) {
      continue;
    }
    if (output_shape[index] != expected_output[index]) {
      throw std::runtime_error("model output shape mismatch");
    }
  }
}

SequencePrediction OnnxSequenceRecognizer::predict(const SequenceImage& image) {
  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};
  Ort::Value input = Ort::Value::CreateTensor<std::uint8_t>(
      memory_info_, const_cast<std::uint8_t*>(image.data()), image.size(),
      input_shape_.data(), input_shape_.size());
  std::vector<Ort::Value> outputs = session_.Run(
      Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
  const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
  const std::vector<std::int64_t> shape = output_info.GetShape();
  if (shape != std::vector<std::int64_t>({1, 3, 10})) {
    throw std::runtime_error("runtime output shape is not [1, 3, 10]");
  }

  const float* logits = outputs[0].GetTensorData<float>();
  SequencePrediction prediction;
  for (std::size_t position = 0; position < kSequenceLength; ++position) {
    const float* begin = logits + position * kDigitClasses;
    prediction.digits[position] = static_cast<std::uint8_t>(
        std::max_element(begin, begin + kDigitClasses) - begin);
  }
  return prediction;
}

}  // namespace static_sequence
