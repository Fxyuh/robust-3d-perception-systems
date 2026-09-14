#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpd {

enum class DType : std::uint32_t { kFloat32 = 1, kInt32 = 2 };

#pragma pack(push, 1)
struct ArrayHeader {
  char magic[8];
  std::uint32_t dtype;
  std::uint32_t ndim;
  std::uint64_t shape[4];
};
#pragma pack(pop)

static_assert(sizeof(ArrayHeader) == 48, "unexpected CPD array header size");

template <typename T> constexpr DType dtypeOf();
template <> constexpr DType dtypeOf<float>() { return DType::kFloat32; }
template <> constexpr DType dtypeOf<std::int32_t>() { return DType::kInt32; }

template <typename T> struct Array {
  std::vector<std::uint64_t> shape;
  std::vector<T> values;
};

inline std::uint64_t elementCount(const std::vector<std::uint64_t>& shape) {
  std::uint64_t count = 1;
  for (const auto dimension : shape) {
    count *= dimension;
  }
  return count;
}

template <typename T> Array<T> readArray(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + path);
  }
  ArrayHeader header{};
  stream.read(reinterpret_cast<char*>(&header), sizeof(header));
  const char expected[8] = {'C', 'P', 'D', 'A', 'R', 'R', '1', '\0'};
  if (!stream || std::memcmp(header.magic, expected, sizeof(expected)) != 0) {
    throw std::runtime_error("invalid CPD array header in " + path);
  }
  if (header.dtype != static_cast<std::uint32_t>(dtypeOf<T>())) {
    throw std::runtime_error("unexpected dtype in " + path);
  }
  if (header.ndim == 0 || header.ndim > 4) {
    throw std::runtime_error("invalid rank in " + path);
  }
  Array<T> result;
  result.shape.assign(header.shape, header.shape + header.ndim);
  result.values.resize(elementCount(result.shape));
  stream.read(reinterpret_cast<char*>(result.values.data()),
              static_cast<std::streamsize>(result.values.size() * sizeof(T)));
  if (!stream) {
    throw std::runtime_error("truncated data in " + path);
  }
  return result;
}

template <typename T>
void writeArray(const std::string& path, const std::vector<std::uint64_t>& shape,
                const std::vector<T>& values) {
  if (shape.empty() || shape.size() > 4 || elementCount(shape) != values.size()) {
    throw std::runtime_error("invalid output shape for " + path);
  }
  ArrayHeader header{};
  const char magic[8] = {'C', 'P', 'D', 'A', 'R', 'R', '1', '\0'};
  std::memcpy(header.magic, magic, sizeof(magic));
  header.dtype = static_cast<std::uint32_t>(dtypeOf<T>());
  header.ndim = static_cast<std::uint32_t>(shape.size());
  for (std::size_t index = 0; index < shape.size(); ++index) {
    header.shape[index] = shape[index];
  }
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot create " + path);
  }
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream) {
    throw std::runtime_error("failed writing " + path);
  }
}

}  // namespace cpd
