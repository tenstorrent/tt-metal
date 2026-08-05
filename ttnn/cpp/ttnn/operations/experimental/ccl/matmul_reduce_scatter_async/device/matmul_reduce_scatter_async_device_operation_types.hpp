// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Type alias for the reduce scatter operation attributes used in fusion
using ReduceScatterMinimalAsyncParams = ttnn::experimental::prim::ReduceScatterMinimalAsyncParams;

struct MatmulReduceScatterAsyncParams {
    ReduceScatterMinimalAsyncParams reduce_scatter_params;
    ttnn::prim::MatmulParams matmul_struct;
    CoreCoord reduce_scatter_core_grid_offset;
    // Optional grid confining the reduce-scatter worker cores. Unlike reduce_scatter_core_grid_offset
    // (which is *added* to every enumerated worker core, and so overflows a grid that already reaches the
    // last column), this restricts choose_worker_cores to a sub-set of the worker sub-device. Used by the
    // BH prefetcher fused path to keep the reduce-scatter off the ring-matmul columns (1-3).
    std::optional<CoreRangeSet> reduce_scatter_sub_core_grid;
    std::vector<IDevice*> devices;

    // Constructor required because operation structs are not default constructible.
    MatmulReduceScatterAsyncParams(
        ReduceScatterMinimalAsyncParams reduce_scatter_params,
        ttnn::prim::MatmulParams matmul_struct,
        CoreCoord reduce_scatter_core_grid_offset,
        std::optional<CoreRangeSet> reduce_scatter_sub_core_grid,
        std::vector<IDevice*> devices) :
        reduce_scatter_params(std::move(reduce_scatter_params)),
        matmul_struct(std::move(matmul_struct)),
        reduce_scatter_core_grid_offset(reduce_scatter_core_grid_offset),
        reduce_scatter_sub_core_grid(std::move(reduce_scatter_sub_core_grid)),
        devices(std::move(devices)) {}

    static constexpr auto attribute_names = std::forward_as_tuple("matmul_struct", "reduce_scatter_core_grid_offset");
    auto attribute_values() const {
        return std::forward_as_tuple(this->matmul_struct, this->reduce_scatter_core_grid_offset);
    }
};

struct MatmulReduceScatterAsyncResult {
    Tensor mm;
    Tensor reduce_scatter;
};

struct MatmulReduceScatterAsyncResultSpec {
    tt::tt_metal::TensorSpec mm;
    tt::tt_metal::TensorSpec reduce_scatter;
};

struct MatmulReduceScatterAsyncInputs {
    Tensor input;
    Tensor weight;
    std::optional<Tensor> bias;
    Tensor persistent_intermediate;
    Tensor persistent_output;
};

}  // namespace ttnn::experimental::prim
