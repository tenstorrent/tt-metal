// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail {

// Shared struct for program artifacts - used for caching kernel handles and core info
struct DeepseekReduceScatterProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> all_cores;
    uint32_t num_directions_per_link;
    uint32_t num_workers_per_direction;
    uint32_t num_mux_cores_per_direction_per_link;
    uint32_t num_cores_per_link;
};

struct operation_attributes_t {
    ttnn::MemoryConfig output_memory_config;
    uint32_t num_links;
    std::optional<uint32_t> cluster_axis;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("sub_device_id", sub_device_id);
        return attrs;
    }
};

struct tensor_args_t {
    ttnn::Tensor input_tensor;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<Tensor>;

// Common validation function
void deepseek_reduce_scatter_common_validates(
    const ttnn::Tensor& input_tensor,
    const ttnn::MemoryConfig& output_memory_config,
    uint32_t num_links,
    uint32_t ring_size);

}  // namespace ttnn::operations::experimental::ccl::deepseek_reduce_scatter::detail
