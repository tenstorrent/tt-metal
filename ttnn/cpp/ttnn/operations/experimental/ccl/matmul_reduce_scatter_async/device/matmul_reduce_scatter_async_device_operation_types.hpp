// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async {

// Type alias for the reduce scatter operation attributes used in fusion
using ReduceScatterMinimalAsyncParams =
    ttnn::operations::experimental::ccl::reduce_scatter_minimal_async::detail::operation_attributes_t;

struct operation_attributes_t {
    ReduceScatterMinimalAsyncParams reduce_scatter_params;
    ttnn::operations::matmul::operation_attributes_t matmul_struct;
    CoreCoord reduce_scatter_core_grid_offset;
    std::vector<IDevice*> devices;

    // Constructor required because operation structs are not default constructible.
    operation_attributes_t(
        ReduceScatterMinimalAsyncParams reduce_scatter_params,
        ttnn::operations::matmul::operation_attributes_t matmul_struct,
        CoreCoord reduce_scatter_core_grid_offset,
        std::vector<IDevice*> devices) :
        reduce_scatter_params(std::move(reduce_scatter_params)),
        matmul_struct(std::move(matmul_struct)),
        reduce_scatter_core_grid_offset(reduce_scatter_core_grid_offset),
        devices(std::move(devices)) {}

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "reduce_scatter_core_grid_offset");
    auto attribute_values() const {
        return std::forward_as_tuple(this->matmul_struct, this->reduce_scatter_core_grid_offset);
    }
};

struct tensor_return_value_t {
    Tensor mm;
    Tensor reduce_scatter;
};

struct spec_return_value_t {
    TensorSpec mm;
    TensorSpec reduce_scatter;
};

struct tensor_args_t {
    Tensor input;
    Tensor weight;
    std::optional<Tensor> bias;
    Tensor persistent_intermediate;
    Tensor persistent_output;
};

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async
