// ONNX Runtime(C++) 추론 벤치마크.
// 기본 실행은 기존 baseline 파일명(logs/cpp_onnx_s{seed}_*.csv)을 유지한다.
// --variant를 주면 8종 tuning ablation 로그를 생성한다.
#include "csv_logger.hpp"
#include "onnx_infer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {
const char* kLabel = "cpp_onnx";
constexpr int kPixels = 1 * 28 * 28;

struct Variant {
  bool graph_opt;
  bool named_output;
  bool memory_reuse;
};

struct Args {
  bool has_variant = false;
  std::string variant_name;
  Variant variant{false, false, false};
  int seed = 0;
  int threads = 1;
  int n = 2000;
  int warmup = 100;
  std::string logdir;
};

bool parse_variant(const std::string& name, Variant& out) {
  if (name == "none") out = {false, false, false};
  else if (name == "graph") out = {true, false, false};
  else if (name == "named") out = {false, true, false};
  else if (name == "memory") out = {false, false, true};
  else if (name == "graph_named") out = {true, true, false};
  else if (name == "graph_memory") out = {true, false, true};
  else if (name == "named_memory") out = {false, true, true};
  else if (name == "all") out = {true, true, true};
  else return false;
  return true;
}

void print_usage() {
  std::cout
      << "Usage:\n"
      << "  bench [seed] [threads] [n] [logdir]\n"
      << "  bench --variant <none|graph|named|memory|graph_named|graph_memory|"
         "named_memory|all> --seed <int> --threads <int> --n <int> --logdir <dir>\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_value = [&](const std::string& flag) {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << flag << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };

    if (a == "--help" || a == "-h") {
      print_usage();
      std::exit(0);
    } else if (a == "--variant") {
      args.variant_name = need_value(a);
      if (!parse_variant(args.variant_name, args.variant)) {
        std::cerr << "unknown variant: " << args.variant_name << "\n";
        std::exit(2);
      }
      args.has_variant = true;
    } else if (a == "--seed") {
      args.seed = std::atoi(need_value(a).c_str());
    } else if (a == "--threads") {
      args.threads = std::atoi(need_value(a).c_str());
    } else if (a == "--n") {
      args.n = std::atoi(need_value(a).c_str());
    } else if (a == "--warmup") {
      args.warmup = std::atoi(need_value(a).c_str());
    } else if (a == "--logdir") {
      args.logdir = need_value(a);
    } else {
      pos.push_back(a);
    }
  }

  if (!pos.empty()) {
    Variant v{};
    size_t offset = 0;
    if (parse_variant(pos[0], v)) {
      args.has_variant = true;
      args.variant_name = pos[0];
      args.variant = v;
      offset = 1;
    }
    if (pos.size() > offset) args.seed = std::atoi(pos[offset].c_str());
    if (pos.size() > offset + 1) args.threads = std::atoi(pos[offset + 1].c_str());
    if (pos.size() > offset + 2) args.n = std::atoi(pos[offset + 2].c_str());
    if (pos.size() > offset + 3) args.logdir = pos[offset + 3];
  }

  return args;
}

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
  Args args = parse_args(argc, argv);
  std::filesystem::path root = std::filesystem::current_path();

  std::string model = (root / "models" / "onnx" /
                       ("mnist_cnn_s" + std::to_string(args.seed) + ".onnx")).string();
  std::vector<uint8_t> images =
      read_file((root / "data" / "mnist_test_images.u8").string());
  std::vector<uint8_t> labels =
      read_file((root / "data" / "mnist_test_labels.u8").string());
  size_t M = labels.size();

  OnnxInferOptions options;
  options.threads = args.threads;
  if (args.has_variant) {
    options.set_graph_optimization = true;
    options.graph_optimization = args.variant.graph_opt;
    options.set_memory_reuse = true;
    options.memory_reuse = args.variant.memory_reuse;
  }
  OnnxInfer engine(model, options);

  std::string label = kLabel;
  if (args.has_variant) {
    label += std::string("_ablation_") + args.variant_name +
             "_t" + std::to_string(args.threads);
  }

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
  times_us.reserve(args.n);
  for (int i = 0; i < args.warmup + args.n; ++i) {
    size_t k = i % M;
    auto t0 = std::chrono::steady_clock::now();
    engine.predict(images.data() + k * kPixels);
    auto t1 = std::chrono::steady_clock::now();
    if (i >= args.warmup)
      times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(times_us.begin(), times_us.end());
  double mean = std::accumulate(times_us.begin(), times_us.end(), 0.0) / times_us.size();

  std::cout << "[" << label << "] seed=" << args.seed << " threads=" << args.threads
            << " warmup=" << args.warmup << " n=" << args.n << "\n";
  std::cout << "  accuracy   = " << acc << "\n";
  std::cout << "  latency us : mean=" << mean
            << " median=" << percentile(times_us, 0.50)
            << " p95=" << percentile(times_us, 0.95)
            << " min=" << times_us.front() << "\n";
  std::cout << "  throughput = " << (1e6 / mean) << " inf/s\n";

  if (!args.logdir.empty()) {
    std::filesystem::create_directories(args.logdir);
    std::string base = args.logdir + "/" + label + "_s" + std::to_string(args.seed);
    csv_logger::write_latency(base + "_latency.csv", times_us);
    csv_logger::write_preds(base + "_preds.csv", labels, preds);
  }
  return 0;
}
