// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/operations/experimental/fused_msda/fused_msda.hpp"

namespace ttnn::operations::experimental::fused_msda {

using ttnn::experimental::MSDAReferenceMode;

// Problem geometry, derived once from the tensor shapes + spatial_shapes and
// shared by validation, the output spec and the program factory. Keeping the
// derivation in one place is what stops the three from drifting.
struct MSDAShapes {
    uint32_t batch = 0;             // B
    uint32_t num_keys = 0;          // S = sum_l H_l * W_l
    uint32_t num_heads = 0;         // H
    uint32_t head_dim = 0;          // D
    uint32_t num_queries = 0;       // Q
    uint32_t num_levels = 0;        // L
    uint32_t num_points = 0;        // P
    uint32_t num_refs = 0;          // R (V2 only; 0 for V1)
    bool locations_packed = false;  // sampling_locations/offsets given as (B, Q, H, L*P*2)
    bool weights_packed = false;    // attention_weights given as (B, Q, H, L*P)
    bool value_packed = false;      // value given as (B, S, H*D); heads recovered from attention_weights
};

struct FusedMSDAOperation {
    struct operation_attributes_t {
        MemoryConfig output_memory_config;
        // (H_0, W_0, H_1, W_1, ...) — flattened so the attribute hashes with the
        // framework's default reflection-based hasher.
        ttsl::SmallVector<uint32_t> spatial_shapes_hw;
        bool align_corners = false;
        bool locations_in_grid_space = false;
        bool from_offsets = false;
        MSDAReferenceMode reference_mode = MSDAReferenceMode::Level;
    };

    struct tensor_args_t {
        const Tensor& value;
        const Tensor& attention_weights;
        // V1 supplies sampling_locations; V2 supplies reference_points +
        // sampling_offsets. Exactly one of the two groups is engaged, selected
        // by operation_attributes_t::from_offsets.
        std::optional<Tensor> sampling_locations;
        std::optional<Tensor> reference_points;
        std::optional<Tensor> sampling_offsets;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Derives (and fully validates) the problem geometry. Throws with a descriptive
// TT_FATAL on any inconsistency.
MSDAShapes derive_shapes(
    const FusedMSDAOperation::operation_attributes_t& attrs, const FusedMSDAOperation::tensor_args_t& args);

}  // namespace ttnn::operations::experimental::fused_msda

namespace ttnn::prim {

ttnn::Tensor fused_msda(
    const Tensor& value,
    const Tensor& sampling_locations,
    const Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    bool align_corners,
    bool locations_in_grid_space,
    const std::optional<MemoryConfig>& memory_config);

ttnn::Tensor fused_msda_from_offsets(
    const Tensor& value,
    const Tensor& reference_points,
    const Tensor& sampling_offsets,
    const Tensor& attention_weights,
    const std::vector<std::array<uint32_t, 2>>& spatial_shapes,
    ttnn::experimental::MSDAReferenceMode reference_mode,
    bool align_corners,
    const std::optional<MemoryConfig>& memory_config);

}  // namespace ttnn::prim
