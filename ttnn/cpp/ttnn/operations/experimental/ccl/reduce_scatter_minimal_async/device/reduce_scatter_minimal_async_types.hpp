// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device.hpp>
#include <optional>
#include <vector>

namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async {

struct operation_attributes_t {
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const std::optional<MemoryConfig> optional_intermediate_mem_config;
    const ::ttnn::ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    const std::optional<GlobalSemaphore> barrier_semaphore;
    const bool using_persistent_buffers;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const std::optional<uint32_t> cluster_axis;
    const std::optional<uint32_t> chunks_per_sync;
    const std::optional<uint32_t> num_workers_per_link;
    const std::optional<uint32_t> num_buffers_per_channel;
};

struct tensor_args_t {
    Tensor input_tensor;
    std::optional<std::vector<Tensor>> persistent_output_buffers;
};

using tensor_return_value_t = std::vector<Tensor>;

using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::ccl::reduce_scatter_minimal_async
