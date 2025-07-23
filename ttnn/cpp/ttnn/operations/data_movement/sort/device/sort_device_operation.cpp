// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sort_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::sort {

constexpr uint32_t WT_THRESHOLD = 64;

SortDeviceOperation::program_factory_t SortDeviceOperation::select_program_factory(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const uint32_t Wt = input_tensor_shape[3] / tile_width;

    // Device number of cores
    const auto device = tensor_args.input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t total_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;

    const auto input_dtype = tensor_args.input_tensor.dtype();
    const auto output_specs = compute_output_specs(attributes, tensor_args);
    const auto index_dtype = output_specs[1].data_type();

    const uint32_t total_number_of_tiles_for_hybrid_approach =
        total_number_of_cores * program::SortProgramFactoryCrossCoreDataExchange::get_number_of_tiles_per_core(
                                    total_number_of_cores,
                                    Wt,
                                    input_dtype,
                                    index_dtype,
                                    program::SortProgramFactoryCrossCoreDataExchange::
                                        CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);

    if (Wt <= WT_THRESHOLD) {
        // Single-core implementation
        return program::SortProgramFactorySingleRowSingleCore{};
    } else if (Wt <= total_number_of_tiles_for_hybrid_approach) {
        // Hybrid implementation
        return program::SortProgramFactoryCrossCoreDataExchange{};
    }
    // DRAM implementation
    return program::SortProgramFactorySingleRowMultiCore{};
}

void SortDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return validate_on_program_cache_miss(attributes, tensor_args);
}

void SortDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Validate shapes of input and output tensors
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const uint32_t Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;

    const auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    const auto input_data_format_size_bytes = tt::datum_size(input_data_format);

    const uint32_t input_tensor_tile_size = tt::constants::TILE_HW * input_data_format_size_bytes;
    const uint32_t value_tensor_tile_size = tt::constants::TILE_HW * input_data_format_size_bytes;
    const uint32_t index_tensor_tile_size = tt::constants::TILE_HW * sizeof(uint16_t);
    const uint32_t row_memory_size_bytes =
        (input_tensor_tile_size + value_tensor_tile_size + index_tensor_tile_size) * Wt;

    const auto device = tensor_args.input_tensor.device();
    const auto l1_mem_size_bytes = device->l1_size_per_core();

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

    TT_FATAL(tensor_args.input_tensor.layout() == Layout::TILE, "The input must be in tiled format");

    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            const auto output_tensor_shape = tensor_args.output_tensors.at(0)->padded_shape();
            TT_FATAL(
                output_tensor_shape == input_tensor_shape,
                "Output tensor shape must be the same as input tensor shape. Got output tensor shape: {} and input "
                "tensor shape: {}",
                output_tensor_shape,
                input_tensor_shape);
            const auto output_indices_shape = tensor_args.output_tensors.at(1)->padded_shape();
            TT_FATAL(
                output_indices_shape == input_tensor_shape,
                "Output tensor indices shape must be the same as input tensor shape. Got output indices tensor shape: "
                "{} and "
                "input tensor shape: {}",
                output_indices_shape,
                input_tensor_shape);
            TT_FATAL(
                tensor_args.output_tensors.at(0)->dtype() == tensor_args.input_tensor.dtype(),
                "Output values tensor dtype must be the same as input tensor dtype. Got output values tensor dtype: {} "
                "and input tensor dtype: {}",
                tensor_args.output_tensors.at(0)->dtype(),
                tensor_args.input_tensor.dtype());
            TT_FATAL(
                tensor_args.output_tensors.at(1)->dtype() == DataType::UINT16 ||
                    tensor_args.output_tensors.at(1)->dtype() == DataType::UINT32,
                "Output indices tensor dtype must be UINT16 or UINT32. Got output indices tensor dtype: {}",
                tensor_args.output_tensors.at(1)->dtype());
        }
    }
}

SortDeviceOperation::spec_return_value_t SortDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensors.size() == 2) {
        if (tensor_args.output_tensors.at(0).has_value() && tensor_args.output_tensors.at(1).has_value()) {
            return {tensor_args.output_tensors[0]->tensor_spec(), tensor_args.output_tensors[1]->tensor_spec()};
        }
    }
    // Create output tensors specs
    auto output_shape = tensor_args.input_tensor.logical_shape();
    auto values_spec = TensorSpec(
        output_shape,
        TensorLayout(tensor_args.input_tensor.dtype(), PageConfig(Layout::TILE), attributes.output_mem_config));

    DataType index_dtype = DataType::UINT16;
    if (output_shape[-1] >= std::numeric_limits<uint16_t>::max()) {
        index_dtype = DataType::UINT32;
    }
    auto index_spec =
        TensorSpec(output_shape, TensorLayout(index_dtype, PageConfig(Layout::TILE), attributes.output_mem_config));

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
        create_device_tensor(output_specs[0], tensor_args.input_tensor.device()),  // Value tensor
        create_device_tensor(output_specs[1], tensor_args.input_tensor.device()),  // Index tensor
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

}  // namespace ttnn::operations::data_movement::sort
