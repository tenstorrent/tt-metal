// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

struct OverlappedTensorView {
    std::string name;
    ttnn::Tensor fused_tensor;
    std::array<uint32_t, 2> tensor_shape;
    std::array<uint32_t, 2> shard_shape;
    tt::tt_metal::CoreRangeSet core_range_set;
    tt::tt_metal::DataType dtype;
    std::array<uint32_t, 2> tile_shape;
    uint64_t byte_offset = 0;
    uint64_t total_size = 0;
};

void dump_overlapped_tensors(const std::string& file_name, const std::vector<OverlappedTensorView>& views);
std::vector<OverlappedTensorView> load_overlapped_tensors(
    const std::string& file_name, tt::tt_metal::distributed::MeshDevice* device = nullptr);

}  // namespace ttnn

// Compatibility aliases - ttnn tensor infrastructure has moved to the ttnn namespace.
namespace tt::tt_metal {

using OverlappedTensorView
    [[deprecated("use ttnn::OverlappedTensorView instead. This alias may be removed after Jun 2026.")]] =
        ttnn::OverlappedTensorView;

template <int = 0>
[[deprecated("use ttnn::dump_overlapped_tensors instead. This alias may be removed after Jun 2026.")]]
inline void dump_overlapped_tensors(
    const std::string& file_name, const std::vector<ttnn::OverlappedTensorView>& views) {
    ttnn::dump_overlapped_tensors(file_name, views);
}

template <int = 0>
[[deprecated("use ttnn::load_overlapped_tensors instead. This alias may be removed after Jun 2026.")]]
inline std::vector<ttnn::OverlappedTensorView> load_overlapped_tensors(
    const std::string& file_name, distributed::MeshDevice* device = nullptr) {
    return ttnn::load_overlapped_tensors(file_name, device);
}

}  // namespace tt::tt_metal
