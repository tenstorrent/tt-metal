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
    Tensor fused_tensor;
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

namespace tt::tt_metal {

// TODO(deprecate): temporary backward-compat aliases while call sites migrate to ttnn::.
using ttnn::dump_overlapped_tensors;
using ttnn::load_overlapped_tensors;
using ttnn::OverlappedTensorView;

}  // namespace tt::tt_metal
