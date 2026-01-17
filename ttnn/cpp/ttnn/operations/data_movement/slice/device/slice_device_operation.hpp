// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include <optional>
#include <variant>

namespace ttnn::operations::data_movement {

uint32_t get_rm_start_offset(const Tensor& tensor, const ttnn::Shape& slice_start);
uint32_t get_tiled_start_offset(const Tensor& input_tensor, const ttnn::Shape& slice_start, bool round_up = false);
uint32_t get_tiled_start_offset(const ttnn::Shape& input_shape, const ttnn::Shape& slice_start, bool round_up = false);

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

struct SliceDeviceOperation {
    using operation_attributes_t = SliceParams;
    using tensor_args_t = SliceInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        SliceRmProgramFactory,
        SliceRmShardedProgramFactory,
        SliceRmStrideProgramFactory,
        SliceTileProgramFactory,
        SliceTileTensorArgsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

SliceDeviceOperation::tensor_return_value_t slice(
    const Tensor& input,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& step,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool use_tensor_args,
    std::optional<Tensor> start_tensor = std::nullopt,
    std::optional<Tensor> end_tensor = std::nullopt,
    const std::optional<uint32_t>& slice_dim = std::nullopt,
    const std::optional<uint32_t>& num_devices = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<Tensor>& preallocated_output = std::nullopt);
}  // namespace ttnn::prim
