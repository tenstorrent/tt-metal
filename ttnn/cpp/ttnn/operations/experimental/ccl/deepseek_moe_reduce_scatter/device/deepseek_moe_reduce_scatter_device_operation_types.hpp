// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental::prim {

// The op is hardcoded for a ring of 8: it takes one input slice per ring position, and create_output_tensors
// allocates one intermediate slice buffer per ring position ahead of the output, each wired to its own circular
// buffer by the program factory. validate_on_program_cache_miss, create_output_tensors, compute_output_topologies
// and the factory all read this one constant so they cannot disagree on the count.
inline constexpr size_t kNumIntermediateSlices = 8;

struct DeepseekMoEReduceScatterParams {
    tt::tt_metal::MemoryConfig output_memory_config;
    uint32_t dim;
    uint32_t num_links;
    std::optional<uint32_t> cluster_axis;

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.reserve(4);
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
