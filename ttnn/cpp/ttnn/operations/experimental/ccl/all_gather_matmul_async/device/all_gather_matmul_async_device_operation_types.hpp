// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "output_mem_config",
        "topology",
        "cluster_axis",
        "barrier_semaphore_present",
        "using_persistent_buffers",
        "chunks_per_sync",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "sub_device_id",
        "matmul",
        "all_gather_core_grid_offset");
    auto attribute_values() const {
        return std::make_tuple(
            all_gather_async_attributes.dim,
            all_gather_async_attributes.num_links,
            all_gather_async_attributes.ring_size,
            std::cref(all_gather_async_attributes.output_mem_config),
            all_gather_async_attributes.topology,
            std::cref(all_gather_async_attributes.cluster_axis),
            all_gather_async_attributes.barrier_semaphore.has_value(),
            all_gather_async_attributes.using_persistent_buffers,
            std::cref(all_gather_async_attributes.chunks_per_sync),
            std::cref(all_gather_async_attributes.num_workers_per_link),
            std::cref(all_gather_async_attributes.num_buffers_per_channel),
            all_gather_async_attributes.sub_device_id,
            std::cref(matmul),
            std::cref(all_gather_core_grid_offset));
    }
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
