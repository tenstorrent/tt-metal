// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEReduceScatterParams {
    tt::tt_metal::MemoryConfig output_memory_config;
    uint32_t dim;
    uint32_t num_links;
    std::optional<uint32_t> cluster_axis;

    auto attributes() const {
        using ttsl::reflection::Attribute;
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
