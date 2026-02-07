// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include <tuple>

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
};

struct StridedAllGatherMinimalMatmulAsyncInputs {
    const Tensor input_tensor;
    const Tensor weight_tensor;
    const std::optional<Tensor> persistent_output_buffer;
    const std::optional<const Tensor> bias = std::nullopt;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_tensor", "weight_tensor", "persistent_output_buffer", "bias");
    auto attribute_values() const {
        return std::forward_as_tuple(input_tensor, weight_tensor, persistent_output_buffer, bias);
    }
};

}  // namespace ttnn::experimental::prim
