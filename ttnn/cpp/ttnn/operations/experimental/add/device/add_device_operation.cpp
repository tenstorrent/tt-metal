
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/add/device/add_device_operation.hpp"
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
    ttnn::device_operation::DeviceOperationConcept<AddDeviceOperation>,
    "AddDeviceOperation must satisfy DeviceOperationConcept");

template <typename... Tensors>
static void fail_on_shape_mismatch(const Tensor& tensor_a, const Tensors&... other_tensors) {
    const auto& shape_a = tensor_a.logical_shape();

    bool all_shapes_match = ((shape_a == other_tensors.logical_shape()) && ...);

    TT_FATAL(all_shapes_match, "Not all input shapes match tensor_a's shape");
}

AddDeviceOperation::program_factory_t AddDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    bool is_sharded = tensor_args.a_tensor.memory_config().nd_shard_spec().has_value() &&
                      tensor_args.b_tensor.memory_config().nd_shard_spec().has_value();

    // bool is_sharded =
    //     tensor_args.a_tensor.memory_config().is_sharded() && tensor_args.b_tensor.memory_config().is_sharded();
    if (is_sharded) {
        return EltNDShardedAddProgram{};
    }
    return ElementWiseMultiCoreAddProgram{};
}

static void validate_memory_config(
    const AddDeviceOperation::operation_attributes_t& attributes, const AddDeviceOperation::tensor_args_t& args) {
    (void)attributes;
    TT_FATAL(
        args.a_tensor.dtype() == args.b_tensor.dtype(),
        "Mismatched data types: 'a_tensor' and 'b_tensor' must have the same dtype.");

    // TT_FATAL(
    //     args.a_tensor.dtype() == output.dtype(),
    //     "Mismatched data types: 'a_tensor' and 'output' tensor must have the same dtype.");
    TT_FATAL(
        args.a_tensor.dtype() == args.b_tensor.dtype(),
        "Mismatched data types: 'a_tensor' and 'b_tensor' must have the same dtype.");

    // TT_FATAL(
    //     args.a_tensor.dtype() == output.dtype(),
    //     "Mismatched data types: 'a_tensor' and 'output' tensor must have the same dtype.");
}

void AddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    validate_memory_config(attributes, args);
    AddDeviceOperation::validate_on_program_cache_hit(attributes, args);

    // TT_FATAL(args.a_tensor.layout() == Layout::TILE, "a_tensor used in the add operation is required to be tiled!");
}

void AddDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& args) {
    fail_on_shape_mismatch(args.a_tensor, args.b_tensor);
}

AddDeviceOperation::spec_return_value_t AddDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& args) {
    if (args.output_tensor.has_value()) {
        return args.output_tensor->tensor_spec();
    }

    fail_on_shape_mismatch(args.a_tensor, args.b_tensor);
    return TensorSpec(
        args.a_tensor.logical_shape(),
        TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), operation_attributes.memory_config));
}

AddDeviceOperation::tensor_return_value_t AddDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& args) {
    if (args.output_tensor.has_value()) {
        return *args.output_tensor;
    }
    return create_device_tensor(compute_output_specs(operation_attributes, args), args.a_tensor.device());
}

tt::stl::hash::hash_t AddDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    TT_FATAL(
        std::holds_alternative<DeviceStorage>(args.a_tensor.storage()),
        "Unexpected type {} for a_tensor storage",
        tt::stl::get_active_type_name_in_variant(args.a_tensor.storage()));
    TT_FATAL(
        std::holds_alternative<DeviceStorage>(args.b_tensor.storage()),
        "Unexpected type {} for b_tensor storage",
        tt::stl::get_active_type_name_in_variant(args.b_tensor.storage()));

    auto program_factory = select_program_factory(attributes, args);
    return operation::hash_operation<AddDeviceOperation>(
        attributes,
        program_factory.index(),
        args.a_tensor.memory_config(),
        args.a_tensor.dtype(),
        args.b_tensor.memory_config(),
        args.b_tensor.dtype());
}

bool AddDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<AddDeviceOperation::operation_attributes_t, AddDeviceOperation::tensor_args_t> AddDeviceOperation::invoke(
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output_tensor) {
    CoreRangeSet worker_grid;
    auto* device = a_tensor.device();
    for (const auto& sub_device_id : device->get_sub_device_ids()) {
        const auto& sub_device_workers =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
        worker_grid = worker_grid.merge(sub_device_workers);
    }

    return {
        operation_attributes_t{
            .memory_config = memory_config.value_or(
                output_tensor.has_value() ? output_tensor->memory_config() : a_tensor.memory_config()),
            .dtype = dtype.value_or(a_tensor.dtype()),
            .worker_grid = std::move(worker_grid),
            .compute_kernel_config = std::nullopt},
        tensor_args_t{.a_tensor = a_tensor, .b_tensor = b_tensor, .output_tensor = output_tensor}};
}

}  // namespace ttnn::experimental::prim
