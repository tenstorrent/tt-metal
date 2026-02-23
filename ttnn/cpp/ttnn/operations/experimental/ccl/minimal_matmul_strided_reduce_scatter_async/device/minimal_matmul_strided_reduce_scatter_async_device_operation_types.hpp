// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

struct MinimalMatmulStridedReduceScatterAsyncParams {
    /* Matmul Params */
    const MinimalMatmulParams matmul_struct;

    /* Reduce Scatter Params */
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig rs_output_mem_config;
    const std::optional<MemoryConfig> rs_intermediate_mem_config;
    const ttnn::ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    const std::optional<GlobalSemaphore> barrier_semaphore;
    const bool using_persistent_buffers;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const std::optional<uint32_t> cluster_axis;
    const std::optional<uint32_t> chunks_per_sync;
    const std::optional<uint32_t> num_workers_per_link;
    const std::optional<uint32_t> num_buffers_per_channel;
    const std::optional<uint32_t> chunk_width_in_mm_blocks;

    const CoreCoord reduce_scatter_core_grid_offset;
    const std::vector<tt::tt_metal::IDevice*> devices;
};

struct MinimalMatmulStridedReduceScatterAsyncInputs {
    const Tensor input_tensor;
    const Tensor weight_tensor;
    const std::optional<Tensor> optional_rs_intermediate_tensor;
    const std::optional<Tensor> optional_rs_output_tensor;
    const std::optional<const Tensor> bias = std::nullopt;
};

}  // namespace ttnn::experimental::prim
