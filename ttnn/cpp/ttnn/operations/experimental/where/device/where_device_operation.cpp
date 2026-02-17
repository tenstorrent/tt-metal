
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operation_concepts.hpp"

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

#include <tracy/Tracy.hpp>
#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

static_assert(
    ttnn::device_operation::DeviceOperationConcept<WhereDeviceOperation>,
    "WhereDeviceOperation must satisfy DeviceOperationConcept");

template <typename... Tensors>
static void fail_on_shape_mismatch(const Tensor& tensor_a, const Tensors&... other_tensors) {
    const auto& shape_a = tensor_a.logical_shape();

    bool all_shapes_match = ((shape_a == other_tensors.logical_shape()) && ...);

    TT_FATAL(all_shapes_match, "Not all input shapes match tensor_a's shape");
}

WhereDeviceOperation::program_factory_t WhereDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    ZoneScopedN("WhereDeviceOperation::select_program_factory");
    return ElementWiseMultiCoreWhereProgram{};
}

static void validate_memory_config(
    const WhereDeviceOperation::operation_attributes_t& attributes, const WhereDeviceOperation::tensor_args_t& args) {
    TT_FATAL(
        args.condition_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "condition_tensor memory layout is required to be INTERLEAVED.");
    TT_FATAL(
        attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "attributes memory layout is required to be INTERLEAVED.");

    TT_FATAL(
        (args.true_value_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED),
        "true_value_tensor memory layout is required to be INTERLEAVED.");

    TT_FATAL(
        (args.false_value_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED),
        "false_value_tensor memory layout is required to be INTERLEAVED.");
}

void WhereDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& args) {

    validate_memory_config(attributes, args);
    WhereDeviceOperation::validate_on_program_cache_hit(attributes, args);

    TT_FATAL(
        args.condition_tensor.layout() == Layout::TILE,
        "Condition tensor used in the where operation is required to be tiled!");
}

void WhereDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& args) {
    fail_on_shape_mismatch(args.condition_tensor, args.true_value_tensor, args.false_value_tensor);
}

WhereDeviceOperation::spec_return_value_t WhereDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& args) {
    if (args.output_tensor.has_value()) {
        return args.output_tensor->tensor_spec();
    }

    fail_on_shape_mismatch(args.condition_tensor, args.true_value_tensor, args.false_value_tensor);
    return TensorSpec(
        args.condition_tensor.logical_shape(),
        TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), operation_attributes.memory_config));
}

WhereDeviceOperation::tensor_return_value_t WhereDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& args) {
    if (args.output_tensor.has_value()) {
        return *args.output_tensor;
    }
    return create_device_tensor(compute_output_specs(operation_attributes, args), args.condition_tensor.device());
}

tt::stl::hash::hash_t WhereDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    TT_FATAL(
        std::holds_alternative<DeviceStorage>(args.condition_tensor.storage()),
        "Unexpected type {} for condition_tensor storage",
        tt::stl::get_active_type_name_in_variant(args.condition_tensor.storage()));
    TT_FATAL(
        std::holds_alternative<DeviceStorage>(args.true_value_tensor.storage()),
        "Unexpected type {} for true_value_tensor storage",
        tt::stl::get_active_type_name_in_variant(args.true_value_tensor.storage()));
    TT_FATAL(
        std::holds_alternative<DeviceStorage>(args.false_value_tensor.storage()),
        "Unexpected type {} for false_value_tensor storage",
        tt::stl::get_active_type_name_in_variant(args.false_value_tensor.storage()));

    auto program_factory = select_program_factory(attributes, args);
    return operation::hash_operation<WhereDeviceOperation>(
        attributes,
        program_factory.index(),
        args.condition_tensor.memory_config(),
        args.condition_tensor.dtype(),
        args.true_value_tensor.memory_config(),
        args.true_value_tensor.dtype(),
        args.false_value_tensor.memory_config(),
        args.false_value_tensor.dtype());
}

bool WhereDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& condition_tensor,
    const Tensor& true_value_tensor,
    const Tensor& false_value_tensor,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output_tensor) {
    CoreRangeSet worker_grid;
    auto* device = condition_tensor.device();
    for (const auto& sub_device_id : device->get_sub_device_ids()) {
        const auto& sub_device_workers =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
        worker_grid = worker_grid.merge(sub_device_workers);
    }

    return {
        operation_attributes_t{
            .memory_config = memory_config.value_or(
                output_tensor.has_value() ? output_tensor->memory_config() : condition_tensor.memory_config()),
            .dtype = dtype.value_or(condition_tensor.dtype()),
            .worker_grid = std::move(worker_grid),
            .compute_kernel_config = std::nullopt},
        tensor_args_t{
            .condition_tensor = condition_tensor,
            .true_value_tensor = true_value_tensor,
            .false_value_tensor = false_value_tensor,
            .output_tensor = output_tensor}};
}

}  // namespace ttnn::experimental::prim
