// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/sub_device.hpp>
#include <optional>

namespace ttnn::operations::experimental::ccl::all_to_all_async_generic {

struct operation_attributes_t {
    const uint32_t in_dim;
    const uint32_t out_dim;
    const uint32_t num_links;
    const uint32_t num_devices;
    const ttnn::MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    const std::optional<uint32_t> cluster_axis;
};

struct tensor_args_t {
    Tensor input_tensor;
    std::optional<Tensor> persistent_output_buffer;
};

}  // namespace ttnn::operations::experimental::ccl::all_to_all_async_generic
