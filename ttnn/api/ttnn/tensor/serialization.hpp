#pragma once

#include <tt-metalium/tensor/serialization.hpp>

namespace ttnn {
// Functions to load and dump tensor to file using FlatBuffer format with inline file storage.
// Only inline file storage (data stored in same file) is currently supported:
// 1. Tensor metadata is serialized and stored as file "header", while the rest of the file is used as a data region for
//    tensor data.
// 2. Metadata includes data offsets and sizes for tensor / tensor shards (multi device context).
void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor);
}  // namespace ttnn
