// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation_types.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_width.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_same_height.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_program_factory_generic.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_pages.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement::reshard {

struct ReshardDeviceOperation {
    using operation_attributes_t = ReshardParams;
    using tensor_args_t = ReshardInputs;
    using spec_return_value_t = reshard::spec_return_value_t;
    using tensor_return_value_t = reshard::tensor_return_value_t;
    using program_factory_t = std::variant<
        reshard::program::ReshardSameWidthFactory</*local_is_output*/ true>,
        reshard::program::ReshardSameWidthFactory</*local_is_output*/ false>,
        reshard::program::ReshardSameHeightFactory</*local_is_output*/ true>,
        reshard::program::ReshardSameHeightFactory</*local_is_output*/ false>,
        reshard::program::ReshardGenericFactory,
        reshard::program::NdReshardCopyPagesFactory,
        reshard::program::NdReshardCopyLocalShardFactory</*local_is_input*/ true>,
        reshard::program::NdReshardCopyLocalShardFactory</*local_is_input*/ false>>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor) const;
};

}  // namespace ttnn::operations::data_movement::reshard

namespace ttnn::prim {
ttnn::operations::data_movement::reshard::ReshardDeviceOperation::tensor_return_value_t reshard(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor);
}  // namespace ttnn::prim
