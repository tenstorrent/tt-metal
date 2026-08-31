// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode.hpp"

#include "device/matmul_decode_device_operation.hpp"

namespace ttnn::experimental {

Tensor matmul_decode(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool partial_width_sharded,
    std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t global_cb_k_blocks,
    const std::optional<PackedWeightSpec>& packed_weight,
    bool all_gather,
    const std::optional<std::vector<ttnn::MeshCoordinate>>& mesh_coords,
    bool ring_gather) {
    return ttnn::prim::matmul_decode(
        input_tensor_a,
        input_tensor_b,
        partial_width_sharded,
        dtype,
        output_mem_config,
        global_cb,
        global_cb_k_blocks,
        packed_weight,
        all_gather,
        mesh_coords,
        ring_gather);
}

}  // namespace ttnn::experimental
