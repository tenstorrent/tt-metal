// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "copy_new_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::operations::data_movement::copy_new {

namespace {

bool can_use_specialized_factory(
    const CopyNewDeviceOperation::operation_attributes_t& operation_attributes,
    const CopyNewDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED ||
        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        return false;
    }
    if (input_tensor.memory_config() != operation_attributes.output_mem_config) {
        return false;
    }

    const bool tilized = input_tensor.layout() == Layout::TILE;
    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(operation_attributes.output_dtype);
    const bool convert_dtype = input_cb_data_format != output_cb_data_format;

    uint32_t input_unit_size =
        tilized ? tt::tile_size(input_cb_data_format) : input_tensor.padded_shape()[-1] * input_tensor.element_size();
    const bool sharded = input_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    if (sharded && !tilized) {
        input_unit_size = input_tensor.memory_config().shard_spec()->shape[1] * input_tensor.element_size();
    }

    const uint32_t input_alignment = input_tensor.buffer()->alignment();
    const uint32_t aligned_input_unit_size = tt::align(input_unit_size, input_alignment);
    constexpr uint32_t num_input_units = 2;
    uint32_t total_cb_size = num_input_units * aligned_input_unit_size;

    if (convert_dtype) {
        uint32_t output_unit_size = tilized ? tt::tile_size(output_cb_data_format)
                                            : input_tensor.padded_shape()[-1] * tt::datum_size(output_cb_data_format);
        if (sharded && !tilized) {
            output_unit_size =
                input_tensor.memory_config().shard_spec()->shape[1] * tt::datum_size(output_cb_data_format);
        }
        const uint32_t output_alignment = input_alignment;
        const uint32_t aligned_output_unit_size = tt::align(output_unit_size, output_alignment);
        constexpr uint32_t num_output_units = 2;
        total_cb_size += num_output_units * aligned_output_unit_size;
    }

    IDevice* device = input_tensor.device();
    const uint32_t max_l1_size =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    return total_cb_size < max_l1_size;
}

}  // namespace

CopyNewDeviceOperation::program_factory_t CopyNewDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (can_use_specialized_factory(operation_attributes, tensor_args)) {
        return SameMemoryConfig{};
    }
    if (tensor_args.input.layout() == Layout::ROW_MAJOR) {
        return DefaultRowMajor{};
    }
    return DefaultTilized{};
}

void CopyNewDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const Tensor& input_tensor_a = tensor_args.input;
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::BFLOAT8_B or
            input_tensor_a.dtype() == DataType::FLOAT32 or input_tensor_a.dtype() == DataType::BFLOAT4_B or
            input_tensor_a.dtype() == DataType::UINT32 or input_tensor_a.dtype() == DataType::INT32,
        "ttnn.copy_new only supports float, bfloat and int32 inputs but got {}",
        input_tensor_a.dtype());
    TT_FATAL(
        operation_attributes.output_dtype == DataType::BFLOAT16 or
            operation_attributes.output_dtype == DataType::BFLOAT8_B or
            operation_attributes.output_dtype == DataType::FLOAT32 or
            operation_attributes.output_dtype == DataType::BFLOAT4_B or
            operation_attributes.output_dtype == DataType::UINT32 or
            operation_attributes.output_dtype == DataType::INT32,
        "ttnn.copy_new only supports float, bfloat and int32 output tensors but got {}",
        operation_attributes.output_dtype);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy_new need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to copy_new need to be allocated in buffers on device!");

    DataType output_dtype = operation_attributes.output_dtype;
    if (tensor_args.preallocated_output.has_value()) {
        const Tensor& out_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            out_tensor.logical_shape() == input_tensor_a.logical_shape(),
            "Input tensor shape {} does not match output tensor shape {}",
            input_tensor_a.logical_shape(),
            out_tensor.logical_shape());
        TT_FATAL(
            input_tensor_a.layout() == out_tensor.layout(),
            "Input tensor layout ({}) must equal output tensor layout ({})",
            input_tensor_a.layout(),
            out_tensor.layout());

        TT_FATAL(
            out_tensor.memory_config() == operation_attributes.output_mem_config,
            "Mismatched output memory config. Check to see if the preallocated output tensor has a different shard "
            "spec than the one specified in the passed-in memory config.");
        TT_FATAL(out_tensor.dtype() == operation_attributes.output_dtype, "Mismatched output dtype");
        TT_FATAL(out_tensor.storage_type() == StorageType::DEVICE, "Output tensor needs to be on device!");
        TT_FATAL(out_tensor.buffer() != nullptr, "Output tensor needs to be allocated in buffers on device!");
        TT_FATAL(
            out_tensor.device() == input_tensor_a.device(),
            "Output tensor needs to be on the same device as the input tensor!");
        output_dtype = out_tensor.dtype();
    }

    if (output_dtype != input_tensor_a.dtype()) {
        TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Only tile layout supports dtype conversion");
    }

    if (input_tensor_a.layout() == Layout::TILE) {
        const auto output_tile = tensor_args.preallocated_output.has_value()
                                     ? tensor_args.preallocated_output.value().tensor_spec().tile()
                                     : tt::tt_metal::TensorLayout(
                                           output_dtype,
                                           tt::tt_metal::PageConfig(input_tensor_a.layout()),
                                           operation_attributes.output_mem_config)
                                           .get_tile();
        TT_FATAL(
            input_tensor_a.tensor_spec().tile().get_tile_shape() == output_tile.get_tile_shape(),
            "Input and output tensors must have the same tile shape when layout is TILE");
    }
}

CopyNewDeviceOperation::spec_return_value_t CopyNewDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const Tensor& input_tensor = tensor_args.input;

    auto output_layout = tt::tt_metal::TensorLayout(
        operation_attributes.output_dtype,
        tt::tt_metal::PageConfig(input_tensor.layout()),
        operation_attributes.output_mem_config);
    auto output_padded_shape = output_layout.compute_padded_shape(input_tensor.logical_shape());
    return {TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            output_padded_shape))};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
CopyNewDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& /*optional_input_tensors*/,
    std::vector<Tensor>& output_tensors) {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    const int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

CopyNewDeviceOperation::tensor_return_value_t CopyNewDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const Tensor& input_tensor = tensor_args.input;
    const spec_return_value_t spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

}  // namespace ttnn::operations::data_movement::copy_new

namespace ttnn::prim {

Tensor copy_new(
    const Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output,
    bool backwards) {
    return ttnn::device_operation::launch<ttnn::operations::data_movement::copy_new::CopyNewDeviceOperation>(
        ttnn::prim::CopyParams{output_mem_config, output_dtype, backwards},
        ttnn::prim::CopyInputs{input, preallocated_output});
}

}  // namespace ttnn::prim
