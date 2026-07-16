#include "static_sequence/csv_logger.hpp"
#include "static_sequence/onnx_sequence_recognizer.hpp"
#include "static_sequence/sequence_types.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
  std::filesystem::path model_path{"models/onnx/static_cnn_s0.onnx"};
  std::filesystem::path images_path{"data/static_test_images.u8"};
  std::filesystem::path labels_path{"data/static_test_labels.u8"};
  std::filesystem::path log_directory{"logs"};
  int threads{1};
  int warmup{100};
  int iterations{2000};
};

std::string require_value(const int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}

Arguments parse_arguments(const int argc, char** argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--model") {
      arguments.model_path = require_value(argc, argv, index);
    } else if (flag == "--images") {
      arguments.images_path = require_value(argc, argv, index);
    } else if (flag == "--labels") {
      arguments.labels_path = require_value(argc, argv, index);
    } else if (flag == "--logdir") {
      arguments.log_directory = require_value(argc, argv, index);
    } else if (flag == "--threads") {
      arguments.threads = std::stoi(require_value(argc, argv, index));
    } else if (flag == "--warmup") {
      arguments.warmup = std::stoi(require_value(argc, argv, index));
    } else if (flag == "--n") {
      arguments.iterations = std::stoi(require_value(argc, argv, index));
    } else {
      throw std::invalid_argument("unknown argument: " + flag);
    }
  }
  if (arguments.threads <= 0 || arguments.warmup < 0 || arguments.iterations <= 0) {
    throw std::invalid_argument("threads and n must be positive; warmup must be non-negative");
  }
  return arguments;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open input file: " + path.string());
  }
  const std::streamsize byte_count = input.tellg();
  input.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byte_count));
  if (!input.read(reinterpret_cast<char*>(bytes.data()), byte_count)) {
    throw std::runtime_error("cannot read input file: " + path.string());
  }
  return bytes;
}

std::vector<static_sequence::SequenceImage> load_images(
    const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_bytes(path);
  if (bytes.empty() || bytes.size() % static_sequence::kImagePixels != 0) {
    throw std::runtime_error("image dump size is not a positive multiple of 3072 bytes");
  }
  const std::size_t image_count = bytes.size() / static_sequence::kImagePixels;
  std::vector<static_sequence::SequenceImage> images(image_count);
  for (std::size_t index = 0; index < image_count; ++index) {
    const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(
        index * static_sequence::kImagePixels);
    std::copy_n(begin, static_sequence::kImagePixels, images[index].begin());
  }
  return images;
}

std::vector<static_sequence::SequencePrediction> load_labels(
    const std::filesystem::path& path, const std::size_t expected_count) {
  const std::vector<std::uint8_t> bytes = read_bytes(path);
  if (bytes.size() != expected_count * static_sequence::kSequenceLength) {
    throw std::runtime_error("label dump size does not match image count");
  }
  std::vector<static_sequence::SequencePrediction> labels(expected_count);
  for (std::size_t index = 0; index < expected_count; ++index) {
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(
                    index * static_sequence::kSequenceLength),
                static_sequence::kSequenceLength, labels[index].digits.begin());
  }
  return labels;
}

double percentile(const std::vector<double>& sorted_values, const double quantile) {
  const std::size_t index = std::min(
      sorted_values.size() - 1,
      static_cast<std::size_t>(quantile * static_cast<double>(sorted_values.size())));
  return sorted_values[index];
}

int run(const Arguments& arguments) {
  const std::vector<static_sequence::SequenceImage> images =
      load_images(arguments.images_path);
  const std::vector<static_sequence::SequencePrediction> labels =
      load_labels(arguments.labels_path, images.size());
  static_sequence::OnnxSequenceRecognizer recognizer(arguments.model_path, arguments.threads);

  std::vector<static_sequence::SequencePrediction> predictions;
  predictions.reserve(images.size());
  std::size_t correct_digits = 0;
  std::size_t exact_sequences = 0;
  for (std::size_t index = 0; index < images.size(); ++index) {
    const auto prediction = recognizer.predict(images[index]);
    predictions.push_back(prediction);
    std::size_t matched_positions = 0;
    for (std::size_t position = 0; position < static_sequence::kSequenceLength; ++position) {
      if (prediction.digits[position] == labels[index].digits[position]) {
        ++correct_digits;
        ++matched_positions;
      }
    }
    if (matched_positions == static_sequence::kSequenceLength) {
      ++exact_sequences;
    }
  }

  std::vector<double> latency_us;
  latency_us.reserve(static_cast<std::size_t>(arguments.iterations));
  for (int index = 0; index < arguments.warmup + arguments.iterations; ++index) {
    const auto& image = images[static_cast<std::size_t>(index) % images.size()];
    const auto start = std::chrono::steady_clock::now();
    static_cast<void>(recognizer.predict(image));
    const auto end = std::chrono::steady_clock::now();
    if (index >= arguments.warmup) {
      latency_us.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
  }
  std::sort(latency_us.begin(), latency_us.end());
  const double mean = std::accumulate(latency_us.begin(), latency_us.end(), 0.0) /
                      static_cast<double>(latency_us.size());
  const double digit_accuracy = static_cast<double>(correct_digits) /
                                static_cast<double>(labels.size() * static_sequence::kSequenceLength);
  const double exact_match = static_cast<double>(exact_sequences) /
                             static_cast<double>(labels.size());

  std::cout << "[cpp_onnx] digit_accuracy=" << digit_accuracy
            << " exact_match=" << exact_match << '\n';
  std::cout << "  latency us : mean=" << mean
            << " median=" << percentile(latency_us, 0.50)
            << " p95=" << percentile(latency_us, 0.95)
            << " min=" << latency_us.front() << '\n';
  std::cout << "  throughput = " << (1e6 / mean) << " seq/s\n";

  static_sequence::csv_logger::write_latency(
      arguments.log_directory / "cpp_onnx_latency.csv", latency_us);
  static_sequence::csv_logger::write_predictions(
      arguments.log_directory / "cpp_onnx_predictions.csv", labels, predictions);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_arguments(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "static_sequence_bench: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
