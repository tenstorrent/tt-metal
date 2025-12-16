// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

/* All Gather Matmul fusion includes */
#include "ttnn/operations/matmul/device/matmul_op.hpp"  //TODO: migrate this code to use new matmul API. This code relies on the old matmul struct
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"  //TODO: migrate this code to use new all_gather_async API. This code relies on the old all_gather_async device_operation header

#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::experimental::ccl::all_gather_matmul_async {

struct operation_attributes_t {
    /* All Gather Params */
    ttnn::AllGatherAsync all_gather_async;  // TODO: migrate this code to use new all_gather_async API. This code relies
                                            // on the old all_gather_async struct
    /* Matmul Params */
    operations::matmul::Matmul
        matmul{};  // TODO: migrate this code to use new matmul API. This code relies on the old matmul struct
    /* Fusion params */
    CoreCoord all_gather_core_grid_offset;

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "all_gather_core_grid_offset");
    auto attribute_values() const { return std::forward_as_tuple(this->matmul, this->all_gather_core_grid_offset); }
};

struct tensor_args_t {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<const Tensor> bias;
    std::optional<Tensor> persistent_output_buffer;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::ccl::all_gather_matmul_async
