// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::experimental::prim {

struct AllGatherMatmulAsyncParams {
    /* All Gather Params */
    AllGatherAsyncParams all_gather_async_attributes;
    AllGatherAsyncInputs all_gather_async_tensor_args;

    /* Matmul Params */
    ttnn::prim::MatmulParams matmul{};
    /* Fusion params */
    CoreCoord all_gather_core_grid_offset;

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "all_gather_core_grid_offset");
    auto attribute_values() const { return std::forward_as_tuple(this->matmul, this->all_gather_core_grid_offset); }
};

struct AllGatherMatmulAsyncInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<const Tensor> bias;
    std::optional<Tensor> persistent_output_buffer;
};

using AllGatherMatmulAsyncResult = std::vector<Tensor>;
using AllGatherMatmulAsyncResultSpec = std::vector<ttnn::TensorSpec>;

}  // namespace ttnn::experimental::prim
