// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ReshapeCodegenDeviceOperation {
    using operation_attributes_t = ReshapeCodegenParams;
    using tensor_args_t = ReshapeCodegenInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ReshapeCodegenTileRowsProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // The gate's verdict is part of the key, so a hit only needs a device input.
    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // The plan adds nothing to the key: the tile-rows program has fixed CB depths and its geometry follows from the
    // shapes.
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor reshape_codegen(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    ttnn::operations::data_movement::reshape_codegen::LaunchPlan plan);

}  // namespace ttnn::prim
