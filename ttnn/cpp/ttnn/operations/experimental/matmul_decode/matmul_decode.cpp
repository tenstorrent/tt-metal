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
    bool ring_gather,
    const std::optional<tt::tt_metal::CoreRangeSet>& output_core_grid,
    bool output_mcast_two_hub,
    bool rms_norm,
    std::optional<float> rms_norm_gamma,
    float rms_norm_epsilon) {
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
        ring_gather,
        output_core_grid,
        output_mcast_two_hub,
        rms_norm,
        rms_norm_gamma,
        rms_norm_epsilon);
}

}  // namespace ttnn::experimental
