
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/operation_concepts.hpp"

#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/command_queue.hpp>

#include <tracy/Tracy.hpp>
#include "tt-metalium/assert.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::where {

static_assert(
    ttnn::device_operation::DeviceOperationConcept<WhereDeviceOperation>,
    "WhereDeviceOperation must satisfy DeviceOperationConcept");

static_assert(
    ttnn::decorators::PrimitiveOperationConcept<WhereDeviceOperation>,
    "WhereDeviceOperation must satisfy PrimitiveOperationConcept");

template <typename... Tensors>
static void fail_on_shape_mismatch(const Tensor& tensor_a, const Tensors&... other_tensors) {
    const auto& shape_a = tensor_a.logical_shape();

    bool all_shapes_match = ((shape_a == other_tensors.logical_shape()) && ...);

    TT_FATAL(all_shapes_match, "Not all input shapes match tensor_a's shape");
}

WhereDeviceOperation::program_factory_t WhereDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    ZoneScopedN("WhereDeviceOperation::select_program_factory");
    return ElementWiseMultiCoreWhereProgram{};
}

static void validate_memory_config(
    const WhereDeviceOperation::operation_attributes_t& attributes,
    const WhereDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& input_tensor_c = tensor_args.input_tensor_c;
    const auto& output_tensor = tensor_args.output_tensor;

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor 'a' memory layout is required to be INTERLEAVED.");
    TT_FATAL(
        attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "attributes memory layout is required to be INTERLEAVED.");

    TT_FATAL(
        (input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED),
        "Input tensor 'b' memory layout is required to be INTERLEAVED.");

    TT_FATAL(
        (input_tensor_c.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED),
        "Input tensor 'c' memory layout is required to be INTERLEAVED.");
}

void WhereDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor_a = tensor_args.input_tensor_a;

    validate_memory_config(attributes, tensor_args);
    WhereDeviceOperation::validate_on_program_cache_hit(attributes, tensor_args);

    TT_FATAL(
        input_tensor_a.get_layout() == Layout::TILE,
        "Condition tensor used in the where operation is required to be tiled!");
}

void WhereDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    fail_on_shape_mismatch(tensor_args.input_tensor_a, tensor_args.input_tensor_b, tensor_args.input_tensor_c);
}

WhereDeviceOperation::spec_return_value_t WhereDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    if (output_tensor.has_value()) {
        return output_tensor->get_tensor_spec();
    }

    fail_on_shape_mismatch(tensor_args.input_tensor_a, tensor_args.input_tensor_b, tensor_args.input_tensor_c);
    return TensorSpec(
        tensor_args.input_tensor_a.logical_shape(),
        TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), operation_attributes.memory_config));
}

WhereDeviceOperation::tensor_return_value_t WhereDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return *tensor_args.output_tensor;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t WhereDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& input_tensor_c = tensor_args.input_tensor_c;

    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_a.get_storage()),
        "Unexpected type {} for tensor 'a' storage",
        tt::stl::get_active_type_name_in_variant(input_tensor_a.get_storage()));
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_b.get_storage()),
        "Unexpected type {} for tensor 'b' storage",
        tt::stl::get_active_type_name_in_variant(input_tensor_b.get_storage()));
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_c.get_storage()),
        "Unexpected type {} for tensor 'c' storage",
        tt::stl::get_active_type_name_in_variant(input_tensor_c.get_storage()));

    auto program_factory = select_program_factory(attributes, tensor_args);
    return operation::hash_operation<WhereDeviceOperation>(
        attributes,
        program_factory.index(),
        input_tensor_a.dtype(),
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
        input_tensor_b.dtype(),
        std::get<DeviceStorage>(input_tensor_b.storage()).memory_config(),
        input_tensor_c.dtype(),
        std::get<DeviceStorage>(input_tensor_c.storage()).memory_config());
}

operation::OpPerformanceModel WhereDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& input_tensor_c = tensor_args.input_tensor_c;
    const auto& output_tensor = tensor_return_value;
    // GS specific parameters
    // 80 B/cycle unpacker BW shared
    // 128 datums per cycle math, but unpacker cant keep up
    constexpr uint32_t unpacker_byte_per_cycle = 80;
    uint32_t num_cores = attributes.worker_grid.num_cores();

    uint32_t total_bytes = 0;
    std::vector<Tensor> input_tensors = {input_tensor_a};
    total_bytes += input_tensor_a.volume() * input_tensor_a.element_size();

    input_tensors.push_back(input_tensor_b);
    total_bytes += input_tensor_b.volume() * input_tensor_b.element_size();

    input_tensors.push_back(input_tensor_c);
    total_bytes += input_tensor_c.volume() * input_tensor_c.element_size();

    uint32_t ideal_eltwise_cycles = total_bytes / unpacker_byte_per_cycle / num_cores;

    operation::OpPerformanceModel result(input_tensors, {output_tensor}, ideal_eltwise_cycles);
    return result;
}

bool WhereDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<WhereDeviceOperation::operation_attributes_t, WhereDeviceOperation::tensor_args_t>
WhereDeviceOperation::invoke(
    const Tensor& a_tensor,
    const Tensor& b_tensor,
    const Tensor& c_tensor,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> output_tensor) {
    CoreRangeSet worker_grid;
    auto device = a_tensor.device();
    for (const auto& sub_device_id : device->get_sub_device_ids()) {
        const auto& sub_device_workers =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
        worker_grid = worker_grid.merge(sub_device_workers);
    }

    return {
        operation_attributes_t{
            .memory_config = memory_config.value_or(
                output_tensor.has_value() ? output_tensor->memory_config() : a_tensor.memory_config()),
            .dtype = dtype.value_or(a_tensor.get_dtype()),
            .worker_grid = std::move(worker_grid),
            .compute_kernel_config = std::nullopt},
        tensor_args_t{
            .input_tensor_a = a_tensor,
            .input_tensor_b = b_tensor,
            .input_tensor_c = c_tensor,
            .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::experimental::where
