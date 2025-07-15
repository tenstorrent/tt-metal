// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_op.hpp"

#include <magic_enum/magic_enum.hpp>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

#include "reshard_program_factory.hpp"
#include "nd_reshard_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

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
    } else {
        // Resharding requires output buffer to be in L1
        return out_mem_config.buffer_type() == BufferType::L1;
    }

    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        if (inp_mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            // row major must have shard_spec[0] be the same on both input and output
            return input_tensor.memory_config().shard_spec().value().shape[0] ==
                   out_mem_config.shard_spec().value().shape[0];
        } else {
            // row major must have shard_spec[1] be the same on both input and output
            return input_tensor.memory_config().shard_spec().value().shape[1] ==
                   out_mem_config.shard_spec().value().shape[1];
        }
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE

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

    if (!CMAKE_UNIQUE_NAMESPACE::is_valid_for_legacy_reshard(input_tensor, out_mem_config)) {
        auto output_tensor_spec = compute_output_specs(input_tensors, output_tensors).front();
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
            magic_enum::enum_name(input_tensor.layout()),
            magic_enum::enum_name(output_tensor_spec.tensor_layout().get_layout()));
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

operation::ProgramWithCallbacks ReshardDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (CMAKE_UNIQUE_NAMESPACE::is_valid_for_legacy_reshard(input_tensor, output_tensor.memory_config())) {
        return detail::reshard_multi_core(input_tensor, output_tensor);
    } else {
        return detail::nd_reshard_multi_core(input_tensor, output_tensor);
    }
}

std::vector<Tensor> ReshardDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors.at(0).device())};
}

}  // namespace ttnn::operations::data_movement
