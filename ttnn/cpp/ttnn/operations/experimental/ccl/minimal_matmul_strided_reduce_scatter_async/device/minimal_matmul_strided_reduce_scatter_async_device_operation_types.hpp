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

    // Rolling L1 window over the MM output, in M blocks. Unset (the default) keeps the whole MM
    // output resident in L1, which costs Mt_per_core * Nt_per_core tiles on every matmul core and
    // caps how large M can get before the shard crowds out the programs' circular buffers. When set
    // to W, only W M blocks per core are resident and slot m % W is recycled; the RS reader signals
    // back per M block so the matmul stalls rather than overwriting a block still to be read.
    // Windowing makes the MM output tensor smaller than [M, N] — it no longer holds the full matmul
    // result, so callers that read it must leave this unset.
    const std::optional<uint32_t> mm_window_blocks;

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
        "mm_window_blocks",
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
            std::cref(mm_window_blocks),
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

    /* Caller-owned scratch for the per-MM-core progress counters the MM uses to signal the RS
       (one uint32 slot per MM core, one row per core, height-sharded in L1 so the row sits at the
       same local address everywhere). Supplying it lets one device-lifetime allocation serve every
       MMRS program; when omitted the RS factory allocates a private array per program, which
       permanently lowers the device's L1 floor and starves later ops of circular-buffer space. */
    const std::optional<const Tensor> mm_progress_counters = std::nullopt;
    // Caller-owned RS->MM credit array for the rolling window; see mm_window_blocks. Allocated per
    // compiled program when absent, which permanently lowers the device's L1 floor.
    const std::optional<const Tensor> mm_credit_counters = std::nullopt;

    /* Fused concatenation: the second in0 source (suffix K-tiles). */
    const std::optional<const Tensor> mm_optional_input_tensor = std::nullopt;
};

}  // namespace ttnn::experimental::prim
