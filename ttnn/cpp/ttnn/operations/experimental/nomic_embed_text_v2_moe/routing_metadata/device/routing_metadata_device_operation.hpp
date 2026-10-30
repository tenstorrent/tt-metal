// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Sizes of one call, derived from its inputs and attributes.
struct RoutingMetadataLayout {
    uint32_t top_k = 0;
    // Every expert's region starts on a 32-row block. Rounding a region up wastes at most 31 rows,
    // so ceil(top_k * num_tokens / 32) + num_experts - 1 blocks hold any routing.
    uint32_t num_blocks = 0;
    uint32_t gathered_rows = 0;
    uint32_t padded_tokens = 0;
};

struct RoutingMetadataDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_experts = 0;
        uint32_t token_start = 0;
        uint32_t num_tokens = 0;
    };

    struct tensor_args_t {
        const Tensor& indices;
        const Tensor& values;
    };

    // gather_index, combine_index, combine_weights, expert_counts.
    using spec_return_value_t = std::
        tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor, Tensor>;

    // Validates the call and derives its sizes. compute_output_specs calls it first because the
    // framework allocates the outputs before validate_on_program_cache_miss runs.
    static RoutingMetadataLayout compute_layout(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Every runtime argument that is not a buffer address comes from a hashed attribute, so the
    // default program hash keeps a cache hit correct; the framework re-binds the buffers.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor, Tensor> routing_metadata(
    const Tensor& indices, const Tensor& values, uint32_t num_experts, uint32_t token_start, uint32_t num_tokens);

}  // namespace ttnn::prim
