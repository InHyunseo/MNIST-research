#ifndef STATIC_SEQUENCE_SEQUENCE_TYPES_HPP_
#define STATIC_SEQUENCE_SEQUENCE_TYPES_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace static_sequence {

inline constexpr std::size_t kImageHeight = 32;
inline constexpr std::size_t kImageWidth = 96;
inline constexpr std::size_t kImagePixels = kImageHeight * kImageWidth;
inline constexpr std::size_t kSequenceLength = 3;
inline constexpr std::size_t kDigitClasses = 10;

using SequenceImage = std::array<std::uint8_t, kImagePixels>;

struct SequencePrediction {
  std::array<std::uint8_t, kSequenceLength> digits{};

  [[nodiscard]] bool operator==(const SequencePrediction& other) const noexcept {
    return digits == other.digits;
  }
};

[[nodiscard]] std::string format_sequence(const SequencePrediction& prediction);

}  // namespace static_sequence

#endif  // STATIC_SEQUENCE_SEQUENCE_TYPES_HPP_
