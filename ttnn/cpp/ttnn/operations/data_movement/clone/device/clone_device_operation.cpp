// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::clone {
void CloneOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    if (operation_attributes.dtype != input.dtype()) {
        TT_FATAL(input.layout() == Layout::TILE, "Clone: data type conversion is only supported with tile layout");
    }
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Clone: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Clone: input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Clone: not currently support sharding");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Clone: not currently support sharding");
}

CloneOperation::program_factory_t CloneOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void CloneOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void CloneOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

CloneOperation::spec_return_value_t CloneOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype, tt::tt_metal::PageConfig(input.layout()), operation_attributes.memory_config));
};

CloneOperation::tensor_return_value_t CloneOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input.device());
}

std::tuple<CloneOperation::operation_attributes_t, CloneOperation::tensor_args_t> CloneOperation::invoke(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            dtype.value_or(input.dtype()),
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{input},
    };
}

tt::tt_metal::operation::OpPerformanceModelGeneral<CloneOperation::tensor_return_value_t>
CloneOperation::create_op_performance_model(
    const operation_attributes_t& op_attr, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input;
    printf("In create_op_performance_model for reshard op\n");
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Input tensor not on DEVICE?!");
    }
    const auto& input_shape = input_tensor.logical_shape();
    auto element_size_bytes = input_tensor.element_size();
    printf("element size bytes: %u\n", element_size_bytes);
    uint32_t input_size_bytes = input_shape.volume() * element_size_bytes;
    printf("input size bytes: %u\n", input_size_bytes);
    bool is_sharded = input_tensor.memory_config().shard_spec().has_value();
    printf("is sharded: %s\n", is_sharded ? "true" : "false");

    bool is_tiled = input_tensor.layout() == Layout::TILE;
    printf("is tiled: %s\n", is_tiled ? "true" : "false");
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    printf("single tile size: %u\n", single_tile_size);
    uint32_t input_transaction_size = is_tiled ? single_tile_size : input_shape[-1] * element_size_bytes;
    if (is_sharded) {
        const auto& input_shard_shape = input_tensor.memory_config().shard_spec().value().shape;
        printf("input shard shape: %u %u\n", input_shard_shape[0], input_shard_shape[1]);
        input_transaction_size = is_tiled ? single_tile_size : input_shard_shape[-1] * element_size_bytes;
    }
    printf("input transaction size: %u\n", input_transaction_size);
    uint32_t num_read_transactions = std::ceil((float)input_size_bytes / (float)input_transaction_size);
    printf("num read transactions: %u\n", num_read_transactions);
    bool is_dram = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    printf("is dram: %s\n", is_dram ? "true" : "false");
    // How to check if one DRAM or all DRAMs are used?
    //  for now assuming we are using all cores, different DRAM channels might be used
    auto arch = input_tensor.device()->arch();
    int num_cores = (arch == tt::ARCH::WORMHOLE_B0) ? 64 : 108;
    printf("num cores: %d\n", num_cores);
    // initial assumptions: divide transactions over all cores
    uint32_t total_read_cycles = get_cycles_for_read_transaction_size(
        input_transaction_size, is_dram, false, std::ceil((float)num_read_transactions / (float)num_cores));
    printf("total read cycles: %u\n", total_read_cycles);
    const auto& output_tensor = output;
    // Assuming parallelization over shard grid cores:
    // First pick the worker cores to be the max between the input and output shard grid cores
    auto is_local = false;
    if (is_sharded) {
        auto input_num_cores = input_tensor.memory_config().shard_spec().value().grid.num_cores();
        auto output_num_cores = output_tensor.memory_config().shard_spec().value().grid.num_cores();
        printf("input num cores: %u\n", input_num_cores);
        printf("output num cores: %u\n", output_num_cores);
        num_cores = std::max(input_num_cores, output_num_cores);

        if (input_num_cores > output_num_cores) {
            is_local = true;
        }
    }
    total_read_cycles = get_cycles_for_read_transaction_size(
        input_transaction_size, is_dram, is_local, std::ceil((float)num_read_transactions / (float)num_cores));

    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }
    const auto& output_shape = output_tensor.logical_shape();
    uint32_t output_size_bytes = output_shape.volume() * element_size_bytes;
    printf("output size bytes: %u\n", output_size_bytes);
    uint32_t output_transaction_size = is_tiled ? single_tile_size : output_shape[-1] * element_size_bytes;
    printf("output transaction size: %u\n", output_transaction_size);
    if (is_sharded) {
        const auto& output_shard_shape = output_tensor.memory_config().shard_spec().value().shape;
        printf("output shard shape: %u %u\n", output_shard_shape[0], output_shard_shape[1]);
        output_transaction_size = is_tiled ? single_tile_size : output_shard_shape[-1] * element_size_bytes;
    }

    uint32_t num_write_transactions = std::ceil((float)output_size_bytes / (float)output_transaction_size);
    printf("num write transactions: %u\n", num_write_transactions);

    uint32_t total_write_cycles = get_cycles_for_read_transaction_size(
        output_transaction_size, is_dram, !is_local, std::ceil((float)num_write_transactions / (float)num_cores));

    printf("total write cycles: %u\n", total_write_cycles);

    // do we just add cycles for read and write?
    int ideal_dev_clock_cycles = total_read_cycles + total_write_cycles;
    printf("ideal dev clock cycles: %d\n", ideal_dev_clock_cycles);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}
}  // namespace ttnn::operations::data_movement::clone
