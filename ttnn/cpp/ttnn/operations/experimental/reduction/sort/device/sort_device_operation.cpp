// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::reduction::sort {

SortDeviceOperation::program_factory_t SortDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return sort::program::SortProgramFactory{};
}

void SortDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return validate_on_program_cache_miss(attributes, tensor_args);
}

void SortDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Validate shapes of input and output tensors
    const auto input_tensor_shape = tensor_args.input_tensor.get_padded_shape();
    const uint32_t Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;

    const auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.get_dtype());
    const auto input_data_format_size_bytes = tt::datum_size(input_data_format);

    const uint32_t input_tensor_tile_size = tt::constants::TILE_HW * input_data_format_size_bytes;
    const uint32_t value_tensor_tile_size = tt::constants::TILE_HW * input_data_format_size_bytes;
    const uint32_t index_tensor_tile_size = tt::constants::TILE_HW * sizeof(uint16_t);
    const uint32_t row_memory_size_bytes =
        (input_tensor_tile_size + value_tensor_tile_size + index_tensor_tile_size) * Wt;

    const auto device = tensor_args.input_tensor.device();
    const auto l1_mem_size_bytes = device->l1_size_per_core();

    // NOTE: This will be updated when support for sorting a single row on multicore is implemented.
    // Issue: https://github.com/tenstorrent/tt-metal/issues/21187
    TT_FATAL(
        row_memory_size_bytes < l1_mem_size_bytes,
        "Row memory size {} bytes exceeds L1 memory size {} bytes. "
        "Consider using a smaller input tensor or increasing the L1 memory size.",
        row_memory_size_bytes,
        l1_mem_size_bytes);

    TT_FATAL(
        tensor_args.input_tensor.buffer() != nullptr,
        "Operands need to be allocated in buffers on the device. Buffer is null.");
    TT_FATAL(
        tensor_args.input_tensor.storage_type() == StorageType::DEVICE,
        "Operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(tensor_args.input_tensor.storage_type()));

    TT_FATAL(input_tensor_shape.rank() == 4, "Input shape must be 4D, got {}", input_tensor_shape.rank());

    TT_FATAL(
        input_tensor_shape[-1] % 64 == 0,
        "Input shape inner dim {} must be a multiple of 64, pad with +/-infinity if necessary",
        input_tensor_shape[-1]);
    TT_FATAL(
        (input_tensor_shape[0] * input_tensor_shape[1] * input_tensor_shape[2]) % 32 == 0,
        "Input height (combined input_shape[0-3]) {} must be a multiple of 32",
        input_tensor_shape[0] * input_tensor_shape[1] * input_tensor_shape[2]);

    TT_FATAL(attributes.output_mem_config.is_sharded() == false, "Sharded implementation not supported yet");

    TT_FATAL(tensor_args.input_tensor.get_layout() == Layout::TILE, "The input must be in tiled format");

    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            const auto output_tensor_shape = tensor_args.output_tensors.at(0)->get_padded_shape();
            TT_FATAL(
                output_tensor_shape == input_tensor_shape,
                "Output tensor shape must be the same as input tensor shape. Got output tensor shape: {} and input "
                "tensor shape: {}",
                output_tensor_shape,
                input_tensor_shape);
            const auto output_indices_shape = tensor_args.output_tensors.at(1)->get_padded_shape();
            TT_FATAL(
                output_indices_shape == input_tensor_shape,
                "Output tensor indices shape must be the same as input tensor shape. Got output indices tensor shape: "
                "{} and "
                "input tensor shape: {}",
                output_indices_shape,
                input_tensor_shape);
        }
    }
}

SortDeviceOperation::spec_return_value_t SortDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            return {tensor_args.output_tensors[0]->get_tensor_spec(), tensor_args.output_tensors[1]->get_tensor_spec()};
        }
    }
    // Create output tensors specs
    auto output_shape = tensor_args.input_tensor.get_logical_shape();
    auto values_spec = TensorSpec(
        output_shape,
        TensorLayout(tensor_args.input_tensor.get_dtype(), PageConfig(Layout::TILE), attributes.output_mem_config));
    auto index_spec = TensorSpec(
        output_shape, TensorLayout(DataType::UINT16, PageConfig(Layout::TILE), attributes.output_mem_config));

    return {values_spec, index_spec};
}

SortDeviceOperation::tensor_return_value_t SortDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            return {tensor_args.output_tensors[0].value(), tensor_args.output_tensors[1].value()};
        }
    }
    auto output_specs = compute_output_specs(attributes, tensor_args);
    return {
        create_device_tensor(output_specs[0], tensor_args.input_tensor.device()),
        create_device_tensor(output_specs[1], tensor_args.input_tensor.device()),
    };
}

std::tuple<SortDeviceOperation::operation_attributes_t, SortDeviceOperation::tensor_args_t> SortDeviceOperation::invoke(
    const Tensor& input_tensor,
    const int8_t dim,
    const bool descending,
    const bool stable,
    const MemoryConfig& output_memory_config,
    const std::vector<std::optional<Tensor>>& output_tensors) {
    return {
        operation_attributes_t{dim, descending, stable, output_memory_config},
        tensor_args_t{input_tensor, output_tensors}};
}

}  // namespace ttnn::operations::experimental::reduction::sort
