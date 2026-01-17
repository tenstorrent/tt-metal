// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/work_split.hpp>

#include <enchantum/enchantum.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace CMAKE_UNIQUE_NAMESPACE {
bool is_valid_for_legacy_reshard(const Tensor& input_tensor, const MemoryConfig& out_mem_config) {
    auto inp_mem_layout = input_tensor.memory_config().memory_layout();
    auto out_mem_layout = out_mem_config.memory_layout();

    auto inp_buffer_type = input_tensor.memory_config().buffer_type();
    auto out_buffer_type = out_mem_config.buffer_type();

    if (!input_tensor.memory_config().shard_spec().has_value() || !out_mem_config.shard_spec().has_value()) {
        // If shard_spec has no value, then we can only use nd resharding
        return false;
    }

    if (inp_mem_layout == out_mem_layout && inp_mem_layout != TensorMemoryLayout::BLOCK_SHARDED) {
        // Resharding must have at least one buffer in L1
        return inp_buffer_type == BufferType::L1 || out_buffer_type == BufferType::L1;
    }  // Resharding requires output buffer to be in L1
    return out_mem_config.buffer_type() == BufferType::L1;

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        if (inp_mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            // row major must have shard_spec[0] be the same on both input and output
            return input_tensor.memory_config().shard_spec().value().shape[0] ==
                   out_mem_config.shard_spec().value().shape[0];
        }  // row major must have shard_spec[1] be the same on both input and output
        return input_tensor.memory_config().shard_spec().value().shape[1] ==
               out_mem_config.shard_spec().value().shape[1];
    }
    return true;
}
}  // namespace CMAKE_UNIQUE_NAMESPACE

ReshardDeviceOperation::program_factory_t ReshardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto output_tensor_spec = compute_output_specs(args, tensor_args);
    const auto& out_mem_config = output_tensor_spec.memory_config();

    if (CMAKE_UNIQUE_NAMESPACE::is_valid_for_legacy_reshard(input_tensor, out_mem_config)) {
        if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            out_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            if (out_mem_config.buffer_type() == BufferType::L1) {
                return ReshardSameWidthFactory</*local_is_output*/ true>{};
            }
            return ReshardSameWidthFactory</*local_is_output*/ false>{};
        }
        if (input_tensor.layout() == Layout::ROW_MAJOR &&
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            out_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            if (out_mem_config.buffer_type() == BufferType::L1) {
                bool has_padding = false;
                CoreCoord input_shard_grid = input_tensor.buffer()->shard_spec().grid().ranges()[0].grid_size();
                CoreCoord output_shard_grid = out_mem_config.shard_spec().value().grid.ranges()[0].grid_size();
                uint32_t input_num_shard_cores = input_shard_grid.x == 1 ? input_shard_grid.y : input_shard_grid.x;
                uint32_t output_num_shard_cores = output_shard_grid.x == 1 ? output_shard_grid.y : output_shard_grid.x;
                uint32_t input_shard_width = input_tensor.buffer()->shard_spec().shape()[1];
                uint32_t output_shard_width = out_mem_config.shard_spec().value().shape[1];
                has_padding = input_num_shard_cores * input_shard_width > input_tensor.logical_shape()[-1];
                has_padding =
                    has_padding || output_num_shard_cores * output_shard_width > input_tensor.logical_shape()[-1];
                if (has_padding) {
                    return ReshardGenericFactory{};
                }
                return ReshardSameHeightFactory</*local_is_output*/ true>{};
            }
            return ReshardSameHeightFactory</*local_is_output*/ false>{};
        }
        return ReshardGenericFactory{};
    }
    auto input_buffer_type = input_tensor.memory_config().buffer_type();
    auto output_buffer_type = out_mem_config.buffer_type();
    auto input_page_size = input_tensor.buffer()->page_size();
    auto output_page_size = output_tensor_spec.compute_page_size_bytes();

    TT_FATAL(
        input_buffer_type == BufferType::DRAM || input_buffer_type == BufferType::L1,
        "Input buffer type must be DRAM or L1");
    TT_FATAL(
        output_buffer_type == BufferType::DRAM || output_buffer_type == BufferType::L1,
        "Output buffer type must be DRAM or L1");

    if (input_buffer_type == BufferType::DRAM && output_buffer_type == BufferType::DRAM) {
        return NdReshardCopyPagesFactory{};
    }
    if (input_buffer_type == BufferType::L1 && output_buffer_type == BufferType::L1 &&
        input_page_size != output_page_size) {
        return NdReshardCopyLocalShardFactory</*local_is_input*/ true>{};
    }

    if (input_buffer_type == BufferType::DRAM) {
        return NdReshardCopyLocalShardFactory</*local_is_input*/ false>{};
    }
    return NdReshardCopyLocalShardFactory</*local_is_input*/ true>{};
}

void ReshardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ReshardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");

    if (tensor_args.preallocated_output.has_value()) {
        const auto& output_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            input_tensor.logical_shape() == output_tensor.logical_shape(),
            "Input tensor logical shape ({}) must equal output tensor logical shape ({})",
            input_tensor.logical_shape(),
            output_tensor.logical_shape());
        TT_FATAL(
            input_tensor.dtype() == output_tensor.dtype(),
            "Input tensor dtype ({}) must equal output tensor dtype ({})",
            input_tensor.dtype(),
            output_tensor.dtype());
        TT_FATAL(
            input_tensor.layout() == output_tensor.layout(),
            "Input tensor layout ({}) must equal output tensor layout ({})",
            input_tensor.layout(),
            output_tensor.layout());
    }

    const auto& out_mem_config = tensor_args.preallocated_output.has_value()
                                     ? tensor_args.preallocated_output->memory_config()
                                     : args.output_mem_config;
    TT_FATAL(out_mem_config.is_sharded(), "output must be sharded");

    auto output_tensor_spec = compute_output_specs(args, tensor_args);
    if (!CMAKE_UNIQUE_NAMESPACE::is_valid_for_legacy_reshard(input_tensor, output_tensor_spec.memory_config())) {
        auto out_distribution_spec = output_tensor_spec.compute_buffer_sharding_args().buffer_distribution_spec();
        auto input_distribution_spec = input_tensor.buffer()->buffer_distribution_spec();

        auto n_logical_input_pages = input_distribution_spec->tensor_shape_in_pages().volume();
        auto n_logical_output_pages = out_distribution_spec->tensor_shape_in_pages().volume();

        auto input_page_size = input_tensor.tensor_spec().compute_page_size_bytes();
        auto output_page_size = output_tensor_spec.compute_page_size_bytes();

        TT_FATAL(
            n_logical_input_pages == n_logical_output_pages,
            "Number of input ({}) and output ({}) pages must match",
            n_logical_input_pages,
            n_logical_output_pages);
        TT_FATAL(
            input_page_size == output_page_size,
            "Input and output tensors must have the same page size. Input page size: {}, Output page size: {}",
            input_page_size,
            output_page_size);
        TT_FATAL(
            input_tensor.layout() == output_tensor_spec.tensor_layout().get_layout(),
            "Input and output tensors must have the same layout. Input layout: {}, Output layout: {}",
            enchantum::to_string(input_tensor.layout()),
            enchantum::to_string(output_tensor_spec.tensor_layout().get_layout()));
    }
}

TensorSpec ReshardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.layout(),
            args.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

Tensor ReshardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> ReshardDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) const {
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(tensor_args.input, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> result(
        {tensor_args.input}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

Tensor reshard(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<ReshardDeviceOperation>(
        ReshardParams{.output_mem_config = memory_config},
        ReshardInputs{.input = input_tensor, .preallocated_output = preallocated_output});
}

}  // namespace ttnn::prim
