// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct StridedAllGatherMinimalMatmulAsyncParams {
    /* All Gather Params */
    const StridedAllGatherAsyncParams strided_all_gather_async_struct;

    /* Matmul Params */
    const MinimalMatmulParams matmul_struct;

    const CoreCoord all_gather_core_grid_offset;
    const bool read_local_slice_from_input;
    const std::vector<tt::tt_metal::IDevice*> devices;
    const StridedAllGatherAsync ag_op;
    const MMSignalAggregatorMode mm_signal_aggregator_mode = MMSignalAggregatorMode::Auto;

    // Compile-time attributes select exactly the program-structure-affecting fields for the default
    // program-cache reflection hash + canonical key
    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "output_mem_config",
        "topology",
        "cluster_axis",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "mm_cores_y",
        "mm_block_ht",
        "mm_block_wt",
        "matmul_struct",
        "all_gather_core_grid_offset",
        "read_local_slice_from_input",
        "mm_signal_aggregator_mode");
    auto attribute_values() const {
        return std::make_tuple(
            strided_all_gather_async_struct.dim,
            strided_all_gather_async_struct.num_links,
            strided_all_gather_async_struct.ring_size,
            std::cref(strided_all_gather_async_struct.output_mem_config),
            strided_all_gather_async_struct.topology,
            std::cref(strided_all_gather_async_struct.cluster_axis),
            std::cref(strided_all_gather_async_struct.num_workers_per_link),
            std::cref(strided_all_gather_async_struct.num_buffers_per_channel),
            std::cref(strided_all_gather_async_struct.mm_cores_y),
            std::cref(strided_all_gather_async_struct.mm_block_ht),
            std::cref(strided_all_gather_async_struct.mm_block_wt),
            std::cref(matmul_struct),
            std::cref(all_gather_core_grid_offset),
            read_local_slice_from_input,
            mm_signal_aggregator_mode);
    }
};

struct StridedAllGatherMinimalMatmulAsyncInputs {
    const Tensor input_tensor;
    const Tensor weight_tensor;
    const std::optional<Tensor> persistent_output_buffer;
    const std::optional<const Tensor> bias = std::nullopt;

    // Fused addcmul: matmul_output = ternary_a + fused_ternary_scalar * matmul_out * ternary_b
    const std::optional<const Tensor> fused_ternary_input_a = std::nullopt;
    const std::optional<const Tensor> fused_ternary_input_b = std::nullopt;
};

}  // namespace ttnn::experimental::prim
