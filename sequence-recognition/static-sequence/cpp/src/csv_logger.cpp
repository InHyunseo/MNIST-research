#include "static_sequence/csv_logger.hpp"

#include <fstream>
#include <stdexcept>

namespace static_sequence::csv_logger {
namespace {

std::ofstream open_output(const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path.string());
  }
  return output;
}

}  // namespace

void write_latency(const std::filesystem::path& path,
                   const std::vector<double>& latency_us) {
  std::ofstream output = open_output(path);
  output << "latency_us\n";
  for (const double latency : latency_us) {
    output << latency << '\n';
  }
}

void write_predictions(const std::filesystem::path& path,
                       const std::vector<SequencePrediction>& labels,
                       const std::vector<SequencePrediction>& predictions) {
  if (labels.size() != predictions.size()) {
    throw std::invalid_argument("label and prediction counts differ");
  }
  std::ofstream output = open_output(path);
  output << "true,pred,correct\n";
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const std::string truth = format_sequence(labels[index]);
    const std::string prediction = format_sequence(predictions[index]);
    output << truth << ',' << prediction << ',' << (truth == prediction ? 1 : 0) << '\n';
  }
}

}  // namespace static_sequence::csv_logger
