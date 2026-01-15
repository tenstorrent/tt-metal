// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#pragma once

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp"
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/distributed/types.hpp"

#include <tt-metalium/core_coord.hpp>

namespace composite_common {

std::tuple<uint32_t, int32_t> normalize_dim_4d(uint32_t dim, uint32_t rank);

bool use_composite_reduce_scatter(const ttnn::Tensor& input_tensor, int32_t dim, std::optional<uint32_t> cluster_axis);
bool use_all_gather_async_llama_sharded(const ttnn::Tensor& input_tensor, const ttnn::MemoryConfig& output_mem_config);
bool use_composite_all_gather(
    const ttnn::Tensor& input_tensor, int32_t dim, const std::optional<ttnn::MemoryConfig>& memory_config);
bool use_composite_all_to_all(
    const ttnn::Tensor& input_tensor,
    int32_t in_dim,
    int32_t out_dim,
    const std::optional<ttnn::MemoryConfig>& memory_config);

ttnn::Tensor composite_reduce_scatter(
    ttnn::Tensor input_tensor,
    int32_t dim,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel);

ttnn::Tensor composite_all_gather(
    ttnn::Tensor input_tensor,
    int32_t dim,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis);
// same as above but for vector of mesh
std::vector<ttnn::Tensor> composite_all_gather(
    const std::vector<ttnn::Tensor>& input_tensors,
    int32_t dim,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis);

ttnn::Tensor composite_all_to_all(
    ttnn::Tensor input_tensor,
    int32_t in_dim,
    int32_t out_dim,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id);

}  // namespace composite_common
