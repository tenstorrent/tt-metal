// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation_types.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_row_major_program_factory.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_tiled_program_factory.hpp"

namespace ttnn::operations::data_movement::reshape {

struct ReshapeDeviceOperation {
    using operation_attributes_t = reshape::operation_attributes_t;
    using tensor_args_t = reshape::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ReshapeRMProgramFactory, ReshapeTiledProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::data_movement::reshape

namespace ttnn::prim {
ttnn::operations::data_movement::reshape::ReshapeDeviceOperation::tensor_return_value_t reshape(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const ttnn::Shape& padded_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool recreate_mapping_tensor,
    const std::optional<CoreRangeSet>& sub_core_grid);
}  // namespace ttnn::prim
