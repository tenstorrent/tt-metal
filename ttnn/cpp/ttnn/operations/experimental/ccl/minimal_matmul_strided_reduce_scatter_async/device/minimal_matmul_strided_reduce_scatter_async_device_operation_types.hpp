// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

struct MinimalMatmulStridedReduceScatterAsyncParams {
    /* Matmul Params */
    const MinimalMatmulParams matmul_struct;

    /* Fused addcmul params (applied at the RS final write step, not in the MM kernel) */
    const std::optional<float> fused_ternary_scalar = std::nullopt;

    /* Reduce Scatter Params */
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig rs_output_mem_config;
    const std::optional<MemoryConfig> rs_intermediate_mem_config;
    const ttnn::ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    const std::optional<GlobalSemaphore> barrier_semaphore;
    const bool using_persistent_buffers;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const std::optional<uint32_t> cluster_axis;
    const std::optional<uint32_t> num_workers_per_link;
    const std::optional<uint32_t> num_buffers_per_channel;
    const std::optional<uint32_t> chunk_width_in_mm_blocks;

    const CoreCoord reduce_scatter_core_grid_offset;

    // Compile-time attributes select exactly the program-structure-affecting fields for the default
    // program-cache reflection hash + canonical key.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "matmul_struct",
        "dim",
        "num_links",
        "ring_size",
        "rs_output_mem_config",
        "rs_intermediate_mem_config",
        "topology",
        "has_barrier_semaphore",
        "using_persistent_buffers",
        "has_sub_device_id",
        "cluster_axis",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "chunk_width_in_mm_blocks",
        "reduce_scatter_core_grid_offset");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(matmul_struct),
            dim,
            num_links,
            ring_size,
            std::cref(rs_output_mem_config),
            std::cref(rs_intermediate_mem_config),
            topology,
            barrier_semaphore.has_value(),
            using_persistent_buffers,
            sub_device_id.has_value(),
            std::cref(cluster_axis),
            std::cref(num_workers_per_link),
            std::cref(num_buffers_per_channel),
            std::cref(chunk_width_in_mm_blocks),
            std::cref(reduce_scatter_core_grid_offset));
    }
};

struct MinimalMatmulStridedReduceScatterAsyncInputs {
    const Tensor input_tensor;
    const Tensor weight_tensor;
    const std::optional<Tensor> optional_rs_intermediate_tensor;
    const std::optional<Tensor> optional_rs_output_tensor;
    const std::optional<const Tensor> bias = std::nullopt;

    /* Fused addcmul inputs: output = addcmul_a + scalar * mm_output * addcmul_b */
    const std::optional<const Tensor> addcmul_input_tensor1 = std::nullopt;  // residual/base
    const std::optional<const Tensor> addcmul_input_tensor2 = std::nullopt;  // gate/multiplier
};

}  // namespace ttnn::experimental::prim
