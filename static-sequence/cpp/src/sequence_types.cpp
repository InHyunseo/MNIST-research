#include "static_sequence/sequence_types.hpp"

#include <stdexcept>

namespace static_sequence {

std::string format_sequence(const SequencePrediction& prediction) {
  std::string text;
  text.reserve(kSequenceLength);
  for (const std::uint8_t digit : prediction.digits) {
    if (digit >= kDigitClasses) {
      throw std::invalid_argument("sequence digit is outside [0, 9]");
    }
    text.push_back(static_cast<char>('0' + digit));
  }
  return text;
}

}  // namespace static_sequence
