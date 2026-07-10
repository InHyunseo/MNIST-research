#ifndef STATIC_SEQUENCE_ONNX_SEQUENCE_RECOGNIZER_HPP_
#define STATIC_SEQUENCE_ONNX_SEQUENCE_RECOGNIZER_HPP_

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

#include "static_sequence/sequence_types.hpp"

namespace static_sequence {

class OnnxSequenceRecognizer final {
 public:
  explicit OnnxSequenceRecognizer(const std::filesystem::path& model_path, int threads);

  OnnxSequenceRecognizer(const OnnxSequenceRecognizer&) = delete;
  OnnxSequenceRecognizer& operator=(const OnnxSequenceRecognizer&) = delete;
  OnnxSequenceRecognizer(OnnxSequenceRecognizer&&) = delete;
  OnnxSequenceRecognizer& operator=(OnnxSequenceRecognizer&&) = delete;
  ~OnnxSequenceRecognizer() = default;

  [[nodiscard]] SequencePrediction predict(const SequenceImage& image);

 private:
  void validate_model_contract();

  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;
  std::array<std::int64_t, 4> input_shape_{1, 1, 32, 96};
};

}  // namespace static_sequence

#endif  // STATIC_SEQUENCE_ONNX_SEQUENCE_RECOGNIZER_HPP_
