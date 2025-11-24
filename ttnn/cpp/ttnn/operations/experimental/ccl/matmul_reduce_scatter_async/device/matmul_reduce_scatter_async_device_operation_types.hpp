// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_types.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include <tt-metalium/core_coord.hpp>
#include <optional>
#include <tuple>

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async {

struct operation_attributes_t {
    const ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::operation_attributes_t
        reduce_scatter_minimal_async_struct;
    const operations::matmul::Matmul matmul_struct;
    const CoreCoord reduce_scatter_core_grid_offset;
};

struct tensor_args_t {
    Tensor input_tensor;
    Tensor weight_tensor;
    Tensor persistent_intermediate_buffer;
    Tensor persistent_output_buffer;
    std::optional<Tensor> bias;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;

using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async
