// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operation.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async {

struct operation_attributes_t {
    std::vector<IDevice*> devices;
    int32_t dim = 0;
    uint32_t num_links = 1;
    uint32_t ring_size = 0;
    MemoryConfig output_mem_config;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<uint32_t> cluster_axis;
};

struct tensor_args_t {
    std::vector<Tensor> input_tensor;
    std::vector<std::optional<Tensor>> persistent_output_buffer;
};

using tensor_return_value_t = std::vector<Tensor>;

using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async
