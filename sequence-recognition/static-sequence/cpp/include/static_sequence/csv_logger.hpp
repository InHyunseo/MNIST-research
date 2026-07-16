#ifndef STATIC_SEQUENCE_CSV_LOGGER_HPP_
#define STATIC_SEQUENCE_CSV_LOGGER_HPP_

#include <filesystem>
#include <vector>

#include "static_sequence/sequence_types.hpp"

namespace static_sequence::csv_logger {

void write_latency(const std::filesystem::path& path,
                   const std::vector<double>& latency_us);

void write_predictions(const std::filesystem::path& path,
                       const std::vector<SequencePrediction>& labels,
                       const std::vector<SequencePrediction>& predictions);

}  // namespace static_sequence::csv_logger

#endif  // STATIC_SEQUENCE_CSV_LOGGER_HPP_
