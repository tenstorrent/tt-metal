// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEReduceScatterProgramArtifacts {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::KernelHandle writer_kernel_id;
    std::vector<tt::tt_metal::CoreCoord> all_cores;
    uint32_t clamped_num_links;
    uint32_t num_directions_per_link;
    std::vector<tt::tt_metal::CBHandle> input_cb_handles;
    std::vector<tt::tt_metal::CBHandle> intermediate_cb_handles;
};

struct DeepseekMoEReduceScatterParams {
    tt::tt_metal::MemoryConfig output_memory_config;
    uint32_t dim;
    uint32_t num_links;
    std::optional<uint32_t> cluster_axis;

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

struct DeepseekMoEReduceScatterInputs {
    std::vector<ttnn::Tensor> input_tensors;
};

}  // namespace ttnn::experimental::prim
