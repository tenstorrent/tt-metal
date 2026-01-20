// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::operations::experimental::ccl::deepseek_moe_reduce_scatter::detail {

struct DeepseekMoEReductScatterProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> all_cores;
    uint32_t clamped_num_links;
    uint32_t num_directions_per_link;
    std::vector<tt::tt_metal::CBHandle> input_cb_handles;
    std::vector<tt::tt_metal::CBHandle> intermediate_cb_handles;
};

struct operation_attributes_t {
    ttnn::MemoryConfig output_memory_config;
    uint32_t dim;
    uint32_t num_links;
    std::optional<uint32_t> cluster_axis;

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("output_memory_config", output_memory_config);
        attrs.emplace_back("dim", dim);
        attrs.emplace_back("num_links", num_links);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct tensor_args_t {
    std::vector<ttnn::Tensor> input_tensors;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttnn::operations::experimental::ccl::deepseek_moe_reduce_scatter::detail
