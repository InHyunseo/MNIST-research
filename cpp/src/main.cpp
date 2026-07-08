// ONNX Runtime(C++) 추론 벤치마크.
// 전체 test set으로 accuracy와 예측을 구한 뒤, 단일 이미지 latency를 측정한다
// (thread=1, batch=1, warmup 제외 후 n회, predict 호출 구간만).
// data/*.u8을 읽고 logs/cpp_onnx_s{seed}_*.csv를 쓴다.
// 실행: bench <seed> <threads> <n> [logdir]
#include "csv_logger.hpp"
#include "onnx_infer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
const std::string kRoot = "/home/hyunseo/Research/mid_project/";
const char* kLabel = "cpp_onnx";
constexpr int kPixels = 1 * 28 * 28;

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(n);
  f.read(reinterpret_cast<char*>(buf.data()), n);
  return buf;
}

double percentile(const std::vector<double>& sorted_vals, double q) {
  size_t idx = std::min(sorted_vals.size() - 1,
                        static_cast<size_t>(q * sorted_vals.size()));
  return sorted_vals[idx];
}
}  // namespace

int main(int argc, char** argv) {
  int seed = (argc > 1) ? std::atoi(argv[1]) : 0;
  int threads = (argc > 2) ? std::atoi(argv[2]) : 1;
  int n = (argc > 3) ? std::atoi(argv[3]) : 2000;
  std::string logdir = (argc > 4) ? argv[4] : "";
  int warmup = 100;

  std::string model = kRoot + "models/onnx/mnist_cnn_s" + std::to_string(seed) + ".onnx";
  std::vector<uint8_t> images = read_file(kRoot + "data/mnist_test_images.u8");
  std::vector<uint8_t> labels = read_file(kRoot + "data/mnist_test_labels.u8");
  size_t M = labels.size();

  OnnxInfer engine(model, threads);

  // 전체 testset 예측 (accuracy + preds 로그)
  std::vector<int> preds(M);
  size_t correct = 0;
  for (size_t k = 0; k < M; ++k) {
    preds[k] = engine.predict(images.data() + k * kPixels);
    if (preds[k] == static_cast<int>(labels[k])) ++correct;
  }
  double acc = static_cast<double>(correct) / M;

  // latency: batch=1, warmup 제외 후 N회, predict 구간만
  std::vector<double> times_us;
  times_us.reserve(n);
  for (int i = 0; i < warmup + n; ++i) {
    size_t k = i % M;
    auto t0 = std::chrono::steady_clock::now();
    engine.predict(images.data() + k * kPixels);
    auto t1 = std::chrono::steady_clock::now();
    if (i >= warmup)
      times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(times_us.begin(), times_us.end());
  double mean = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();

  std::cout << "[" << kLabel << "] seed=" << seed << " threads=" << threads
            << " warmup=" << warmup << " n=" << n << "\n";
  std::cout << "  accuracy   = " << acc << "\n";
  std::cout << "  latency us : mean=" << mean
            << " median=" << percentile(times_us, 0.50)
            << " p95=" << percentile(times_us, 0.95)
            << " min=" << times_us.front() << "\n";
  std::cout << "  throughput = " << (1e6 / mean) << " inf/s\n";

  if (!logdir.empty()) {
    std::string base = logdir + "/" + kLabel + "_s" + std::to_string(seed);
    csv_logger::write_latency(base + "_latency.csv", times_us);
    csv_logger::write_preds(base + "_preds.csv", labels, preds);
  }
  return 0;
}
