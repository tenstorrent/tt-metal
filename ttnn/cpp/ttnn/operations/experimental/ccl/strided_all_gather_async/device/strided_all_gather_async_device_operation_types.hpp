// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::ccl::strided_all_gather_async {

struct operation_attributes_t {
    const std::vector<IDevice*> devices; // not used, could be deleted?
    const uint32_t dim;
    const uint32_t num_links;
    const uint32_t ring_size;
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::vector<GlobalSemaphore> semaphore;
    const std::optional<uint32_t> cluster_axis;
    const std::optional<uint32_t> tiles_per_chunk;
    const std::optional<uint32_t> num_workers_per_link;
    const std::optional<uint32_t> num_buffers_per_channel;
    const std::optional<uint32_t> mm_cores_y;
    const std::optional<uint32_t> mm_block_ht;
    const std::optional<uint32_t> mm_block_wt;
};

struct tensor_args_t {
    const Tensor input_tensor;
    const std::optional<Tensor> persistent_output_buffer;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::ccl::strided_all_gather_async
