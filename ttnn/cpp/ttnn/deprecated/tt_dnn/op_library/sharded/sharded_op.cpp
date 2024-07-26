// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/sharded/sharded_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Sharded::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        TT_FATAL(this->output_mem_config.is_sharded());
        TT_FATAL(this->output_mem_config.buffer_type == BufferType::L1);
        if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
            TT_FATAL((*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % L1_ALIGNMENT == 0, "Shard page size must currently have L1 aligned page size");
        }
    } else if (this->sharded_op_type == ShardedOpType::ShardedToInterleaved) {
        TT_FATAL(input_tensor.memory_config().is_sharded());
        TT_FATAL(input_tensor.memory_config().buffer_type == BufferType::L1);
        TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
            TT_FATAL((*input_tensor.memory_config().shard_spec).shape[1] * input_tensor.element_size() % (this->output_mem_config.buffer_type == BufferType::DRAM ? DRAM_ALIGNMENT : L1_ALIGNMENT) == 0, "Shard page size must be aligned to output buffer type alignment");
        }
    }
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE);
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(this->grid_size.x <= device_grid.x && this->grid_size.y <= device_grid.y);
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<Shape> Sharded::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> Sharded::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks Sharded::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        return interleaved_to_sharded_multi_core(input_tensor, output_tensor);
    } else {
        return sharded_to_interleaved_multi_core(input_tensor, output_tensor);
    }
}

std::string Sharded::get_type_name() const { return magic_enum::enum_name(this->sharded_op_type).data(); }

ShardedOpParallelizationStrategy Sharded::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    return ShardedOpParallelizationStrategy::MULTI_CORE;
}


void Reshard::validate_with_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");
    bool has_output_tensor = output_tensors.size() == 1 && output_tensors[0].has_value();
    if (has_output_tensor) {
        const auto& output_tensor = output_tensors[0].value();
        TT_FATAL(input_tensor.get_shape() == output_tensor.get_shape());
        TT_FATAL(input_tensor.get_dtype() == output_tensor.get_dtype());
        TT_FATAL(input_tensor.get_layout() == output_tensor.get_layout());
    }
    const auto& out_mem_config = has_output_tensor ? output_tensors[0].value().memory_config() : this->output_mem_config;
    TT_FATAL(out_mem_config.is_sharded(), "output must be sharded");
    TT_FATAL(out_mem_config.buffer_type == BufferType::L1);
    if(input_tensor.get_layout() == Layout::ROW_MAJOR) {
        bool same_row_size = input_tensor.memory_config().shard_spec.value().shape[1] == out_mem_config.shard_spec.value().shape[1];
        TT_FATAL(same_row_size, "row major must have shard_spec[1] be the same on both input and output");
    }
}

std::vector<Shape> Reshard::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

operation::ProgramWithCallbacks Reshard::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    //each tensor has its respective shard_spec within its memory_config
    return reshard_multi_core(input_tensor, output_tensor);
}

std::vector<Tensor> Reshard::create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    } else {
        auto mem_config = this->output_mem_config;

        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            mem_config
            )};
    }
}

ShardedOpParallelizationStrategy Reshard::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    return ShardedOpParallelizationStrategy::MULTI_CORE;
}

}  // namespace tt_metal

}  // namespace tt
