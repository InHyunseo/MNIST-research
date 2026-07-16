// 벤치마크 결과 CSV 기록 (latency, 예측). header-only.
#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace csv_logger {

inline void write_latency(const std::string& path, const std::vector<double>& times) {
  std::ofstream f(path);
  f << "latency_us\n";
  for (double t : times) f << t << "\n";
}

inline void write_preds(const std::string& path, const std::vector<uint8_t>& trues,
                        const std::vector<int>& preds) {
  std::ofstream f(path);
  f << "true,pred\n";
  for (size_t k = 0; k < preds.size(); ++k)
    f << static_cast<int>(trues[k]) << "," << preds[k] << "\n";
}

}  // namespace csv_logger
