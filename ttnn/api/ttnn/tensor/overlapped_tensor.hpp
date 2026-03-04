// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

struct OverlappedTensorView {
    std::string name;
    Tensor fused_tensor;
    std::array<uint32_t, 2> tensor_shape;
    std::array<uint32_t, 2> shard_shape;
    CoreRangeSet core_range_set;
    DataType dtype;
    std::array<uint32_t, 2> tile_shape;
    uint64_t byte_offset = 0;
};

void dump_overlapped_tensors(const std::string& file_name, const std::vector<OverlappedTensorView>& views);
std::vector<OverlappedTensorView> load_overlapped_tensors(
    const std::string& file_name, distributed::MeshDevice* device = nullptr);

}  // namespace tt::tt_metal
