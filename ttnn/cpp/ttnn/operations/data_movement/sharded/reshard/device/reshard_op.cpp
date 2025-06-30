// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_op.hpp"

#include <magic_enum/magic_enum.hpp>

#include "reshard_program_factory.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void ReshardDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");

    bool has_output_tensor = output_tensors.size() == 1 && output_tensors[0].has_value();
    if (has_output_tensor) {
        const auto& output_tensor = output_tensors[0].value();
        TT_FATAL(input_tensor.logical_shape() == output_tensor.logical_shape(), "Error");
        TT_FATAL(input_tensor.dtype() == output_tensor.dtype(), "Error");
        TT_FATAL(input_tensor.layout() == output_tensor.layout(), "Error");
    }
    const auto& out_mem_config =
        has_output_tensor ? output_tensors[0].value().memory_config() : this->output_mem_config;
    TT_FATAL(out_mem_config.is_sharded(), "output must be sharded");

    if ((input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
         out_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED)) {
        TT_FATAL(
            (input_tensor.memory_config().buffer_type() == BufferType::L1 ||
             out_mem_config.buffer_type() == BufferType::L1),
            "Resharding height shard to height shard must have at least one buffer in L1");
    } else if ((input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                out_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED)) {
        TT_FATAL(
            (input_tensor.memory_config().buffer_type() == BufferType::L1 ||
             out_mem_config.buffer_type() == BufferType::L1),
            "Resharding width shard to width shard must have at least one buffer in L1");
    } else {
        TT_FATAL(out_mem_config.buffer_type() == BufferType::L1, "Resharding requires output buffer to be in L1");
    }

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            bool same_row_size = input_tensor.memory_config().shard_spec().value().shape[0] ==
                                 out_mem_config.shard_spec().value().shape[0];
            TT_FATAL(same_row_size, "row major must have shard_spec[0] be the same on both input and output");
        } else {
            bool same_height_size = input_tensor.memory_config().shard_spec().value().shape[1] ==
                                    out_mem_config.shard_spec().value().shape[1];
            TT_FATAL(same_height_size, "row major must have shard_spec[1] be the same on both input and output");
        }
    }
}

std::vector<ttnn::TensorSpec> ReshardDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }

    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.layout(),
            output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
ReshardDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    printf("create_op_performance_model\n");
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Input tensor not on DEVICE?!");
    }
    const auto& input_shape = input_tensor.logical_shape();
    auto element_size_bytes = input_tensor.element_size();
    uint32_t input_size_bytes = input_shape.volume() * element_size_bytes;
    const auto& input_shard_shape = input_tensor.memory_config().shard_spec().value().shape;
    bool is_tiled = input_tensor.layout() == Layout::TILE;
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    uint32_t input_transaction_size = is_tiled ? single_tile_size : input_shard_shape[-1] * element_size_bytes;
    uint32_t num_read_transactions = std::ceil((float)input_size_bytes / (float)input_transaction_size);
    bool is_dram = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    // How to check if one DRAM or all DRAMs are used?
    //  for now assuming we are using all cores, different DRAM channels might be used
    auto arch = input_tensor.device()->arch();
    const int num_cores = (arch == tt::ARCH::WORMHOLE_B0) ? 64 : 108;

    // initial assumptions: divide transactions over all cores
    // alternative: use only shard grid cores: read transactions are now near, write transactions still far
    uint32_t total_read_cycles = get_cycles_for_read_transaction_size(
        input_transaction_size, is_dram, std::ceil((float)num_read_transactions / (float)num_cores));

    const auto& output_tensor = output_tensors.at(0);
    if (output_tensor.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }
    const auto& output_shape = output_tensor.logical_shape();
    uint32_t output_size_bytes = output_shape.volume() * element_size_bytes;
    const auto& output_shard_shape = output_tensor.memory_config().shard_spec().value().shape;
    uint32_t output_transaction_size = is_tiled ? single_tile_size : output_shard_shape[-1] * element_size_bytes;
    uint32_t num_write_transactions = std::ceil((float)output_size_bytes / (float)output_transaction_size);
    uint32_t total_write_cycles = get_cycles_for_write_transaction_size(
        output_transaction_size, is_dram, std::ceil((float)num_write_transactions / (float)num_cores));

    // do we just add cycles for read and write?
    int ideal_dev_clock_cycles = total_read_cycles + total_write_cycles;

    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

operation::ProgramWithCallbacks ReshardDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    // each tensor has its respective shard_spec within its memory_config
    return detail::reshard_multi_core(input_tensor, output_tensor);
}

std::vector<Tensor> ReshardDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors.at(0).device())};
}

}  // namespace ttnn::operations::data_movement
