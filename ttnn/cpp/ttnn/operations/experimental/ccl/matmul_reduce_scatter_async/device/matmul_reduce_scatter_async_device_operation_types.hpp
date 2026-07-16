// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include <tt_stl/reflection.hpp>
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
    std::vector<IDevice*> devices;

    // Constructor required because operation structs are not default constructible.
    MatmulReduceScatterAsyncParams(
        ReduceScatterMinimalAsyncParams reduce_scatter_params,
        ttnn::prim::MatmulParams matmul_struct,
        CoreCoord reduce_scatter_core_grid_offset,
        std::vector<IDevice*> devices) :
        reduce_scatter_params(std::move(reduce_scatter_params)),
        matmul_struct(std::move(matmul_struct)),
        reduce_scatter_core_grid_offset(reduce_scatter_core_grid_offset),
        devices(std::move(devices)) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "output_mem_config",
        "optional_intermediate_mem_config",
        "topology",
        "has_barrier_semaphore",
        "using_persistent_buffers",
        "sub_device_id",
        "cluster_axis",
        "chunks_per_sync",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "matmul_struct",
        "reduce_scatter_core_grid_offset");
    auto attribute_values() const {
        return std::make_tuple(
            reduce_scatter_params.dim,
            reduce_scatter_params.num_links,
            reduce_scatter_params.ring_size,
            std::cref(reduce_scatter_params.output_mem_config),
            std::cref(reduce_scatter_params.optional_intermediate_mem_config),
            reduce_scatter_params.topology,
            reduce_scatter_params.barrier_semaphore.has_value(),
            reduce_scatter_params.using_persistent_buffers,
            reduce_scatter_params.sub_device_id,
            std::cref(reduce_scatter_params.cluster_axis),
            std::cref(reduce_scatter_params.chunks_per_sync),
            std::cref(reduce_scatter_params.num_workers_per_link),
            std::cref(reduce_scatter_params.num_buffers_per_channel),
            std::cref(matmul_struct),
            std::cref(reduce_scatter_core_grid_offset));
    }
};

struct MatmulReduceScatterAsyncResult {
    Tensor mm;
    Tensor reduce_scatter;
};

struct MatmulReduceScatterAsyncResultSpec {
    TensorSpec mm;
    TensorSpec reduce_scatter;
};

struct MatmulReduceScatterAsyncInputs {
    Tensor input;
    Tensor weight;
    std::optional<Tensor> bias;
    Tensor persistent_intermediate;
    Tensor persistent_output;
};

}  // namespace ttnn::experimental::prim
