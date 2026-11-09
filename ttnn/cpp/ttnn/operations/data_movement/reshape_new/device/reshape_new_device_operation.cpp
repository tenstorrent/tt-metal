// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_new/device/reshape_new_device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_new/reshape_new_topology.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

using ttnn::operations::data_movement::reshape_new::ReshapeKind;

ReshapeNewCoreSplit split_reshape_units(
    tt::tt_metal::distributed::MeshDevice* device,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grid,
    uint32_t num_units) {
    tt::tt_metal::CoreRangeSet grid;
    if (sub_core_grid.has_value()) {
        grid = *sub_core_grid;
    } else {
        const auto size = device->compute_with_storage_grid_size();
        grid = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {size.x - 1, size.y - 1}));
    }
    const auto [num_cores, all_cores, group_1, group_2, units_per_core_1, units_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, num_units);

    ReshapeNewCoreSplit split;
    split.all_cores = all_cores;
    split.cores = tt::tt_metal::corerange_to_cores(group_1);
    const auto group_2_cores = tt::tt_metal::corerange_to_cores(group_2);
    split.cores.insert(split.cores.end(), group_2_cores.begin(), group_2_cores.end());
    const size_t group_1_size = group_1.num_cores();
    split.unit_begin.reserve(split.cores.size() + 1);
    uint32_t begin = 0;
    for (size_t i = 0; i < split.cores.size(); ++i) {
        split.unit_begin.push_back(begin);
        begin += i < group_1_size ? units_per_core_1 : units_per_core_2;
    }
    split.unit_begin.push_back(begin);
    TT_FATAL(begin == num_units, "reshape: split {} work units into {}", num_units, begin);
    return split;
}

ReshapeNewDeviceOperation::program_factory_t ReshapeNewDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    if (operation_attributes.kind == ReshapeKind::PageCopy) {
        return ReshapeNewPageCopyProgramFactory{};
    }
    return ReshapeNewGatherProgramFactory{};
}

void ReshapeNewDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& output_spec = operation_attributes.output_spec;
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "reshape: the input must be on device");
    TT_FATAL(input.is_allocated(), "reshape: the input must be allocated");
    TT_FATAL(
        operation_attributes.kind == ReshapeKind::PageCopy || operation_attributes.kind == ReshapeKind::Gather,
        "reshape: the device operation only runs PageCopy and Gather programs");
    TT_FATAL(
        input.logical_volume() == output_spec.logical_shape().volume(),
        "reshape: cannot reshape {} ({} elements) to {} ({} elements)",
        input.logical_shape(),
        input.logical_volume(),
        output_spec.logical_shape(),
        output_spec.logical_shape().volume());
    TT_FATAL(
        input.dtype() == output_spec.data_type() && input.layout() == output_spec.layout(),
        "reshape: the device operation does not change dtype or layout. Input spec: {}, output spec: {}",
        input.tensor_spec(),
        output_spec);
    if (operation_attributes.kind == ReshapeKind::PageCopy) {
        TT_FATAL(
            ttnn::operations::data_movement::reshape_new::page_sequences_identical(input.tensor_spec(), output_spec),
            "reshape: a page copy needs identical page sequences. Input spec: {}, output spec: {}",
            input.tensor_spec(),
            output_spec);
    }
}

ReshapeNewDeviceOperation::spec_return_value_t ReshapeNewDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    return operation_attributes.output_spec;
}

ReshapeNewDeviceOperation::tensor_return_value_t ReshapeNewDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(operation_attributes.output_spec, tensor_args.input.device());
}

ReshapeNewDeviceOperation::topology_return_value_t ReshapeNewDeviceOperation::compute_output_topologies(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return {ttnn::operations::data_movement::reshape_new::remap_topology(
        input.tensor_topology(), input.logical_shape(), operation_attributes.output_spec.logical_shape())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ReshapeNewDeviceOperation::tensor_return_value_t>
ReshapeNewDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args, const Tensor& output) {
    const auto& input = tensor_args.input;
    const int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input, output);
    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        {input}, {output}, ideal_dev_clock_cycles);
}

Tensor reshape_new(
    const Tensor& input,
    const tt::tt_metal::TensorSpec& output_spec,
    ttnn::operations::data_movement::reshape_new::ReshapeKind kind,
    uint32_t pad_bits,
    bool fill_padding,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grid) {
    return ttnn::device_operation::launch<ReshapeNewDeviceOperation>(
        ReshapeNewParams{
            .output_spec = output_spec,
            .kind = kind,
            .pad_bits = pad_bits,
            .fill_padding = fill_padding,
            .sub_core_grid = sub_core_grid},
        ReshapeNewInputs{input});
}

}  // namespace ttnn::prim
