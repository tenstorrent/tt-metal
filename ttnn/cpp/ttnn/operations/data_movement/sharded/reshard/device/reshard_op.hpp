// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation_types.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_width.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_height.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_generic.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp"

namespace ttnn::operations::data_movement {

struct ReshardDeviceOperation {
    using operation_attributes_t = reshard::operation_attributes_t;
    using tensor_args_t = reshard::tensor_args_t;
    using spec_return_value_t = reshard::spec_return_value_t;
    using tensor_return_value_t = reshard::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::ReshardSameWidthFactory</*local_is_output*/ true>,
        program::ReshardSameWidthFactory</*local_is_output*/ false>,
        program::ReshardSameHeightFactory</*local_is_output*/ true>,
        program::ReshardSameHeightFactory</*local_is_output*/ false>,
        program::ReshardGenericFactory,
        program::NdReshardCopyPagesFactory,
        program::NdReshardCopyLocalShardFactory</*local_is_input*/ true>,
        program::NdReshardCopyLocalShardFactory</*local_is_input*/ false>>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const tt::tt_metal::MemoryConfig& memory_config,
        const std::optional<Tensor>& optional_output_tensor);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto reshard =
    ttnn::register_operation<"ttnn::prim::reshard", ttnn::operations::data_movement::ReshardDeviceOperation>();
}  // namespace ttnn::prim
