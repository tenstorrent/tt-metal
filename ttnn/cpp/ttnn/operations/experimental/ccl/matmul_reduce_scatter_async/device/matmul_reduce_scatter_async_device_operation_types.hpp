// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async {

struct operation_attributes_t {
    /* All Gather Params */
    ttnn::ReduceScatterMinimalAsync reduce_scatter_minimal_async_struct;
    /* Matmul Params */
    operations::matmul::Matmul matmul_struct;
    /* Fusion Params */
    CoreCoord reduce_scatter_core_grid_offset;
    /* Physical Devices this op runs on*/
    std::vector<IDevice*> devices;

    Tensor& persistent_intermediate_buffer;
    Tensor& persistent_output_buffer;

    // Constructor required because operation structs are not default constructible.
    operation_attributes_t(
        ttnn::ReduceScatterMinimalAsync reduce_scatter_minimal_async_struct,
        operations::matmul::Matmul matmul_struct,
        CoreCoord reduce_scatter_core_grid_offset,
        std::vector<IDevice*> devices,
        Tensor& persistent_intermediate_buffer,
        Tensor& persistent_output_buffer) :
        reduce_scatter_minimal_async_struct(std::move(reduce_scatter_minimal_async_struct)),
        matmul_struct(std::move(matmul_struct)),
        reduce_scatter_core_grid_offset(reduce_scatter_core_grid_offset),
        devices(std::move(devices)),
        persistent_intermediate_buffer(persistent_intermediate_buffer),
        persistent_output_buffer(persistent_output_buffer) {}

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
};

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_async
