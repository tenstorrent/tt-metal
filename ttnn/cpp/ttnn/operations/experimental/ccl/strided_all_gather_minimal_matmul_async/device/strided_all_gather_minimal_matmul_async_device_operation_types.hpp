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

    // Reflection (logging, graph capture) only; the program-cache key is StridedAllGatherMinimalMatmulAsync::
    // compute_program_hash. ag_op is left out because it carries no state.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "strided_all_gather_async_struct",
        "matmul_struct",
        "all_gather_core_grid_offset",
        "read_local_slice_from_input",
        "devices",
        "mm_signal_aggregator_mode");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(strided_all_gather_async_struct),
            std::cref(matmul_struct),
            std::cref(all_gather_core_grid_offset),
            read_local_slice_from_input,
            std::cref(devices),
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
