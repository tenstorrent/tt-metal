// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_op.hpp"

#include <enchantum/enchantum.hpp>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

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
            enchantum::to_string(input_tensor.layout()),
            enchantum::to_string(output_tensor_spec.tensor_layout().get_layout()));
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
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

Tensor construct_per_core_host_tensor(
    const std::unordered_map<CoreCoord, std::vector<uint32_t>>& core_to_data, uint32_t MAX_RT_ARGS_WIDTH) {
    uint32_t max_width = MAX_RT_ARGS_WIDTH;

    // Create shape based on number of cores and max width
    ttnn::Shape tensor_shape({static_cast<uint32_t>(core_to_data.size()), static_cast<uint32_t>(max_width)});

    // Sort cores to ensure consistent ordering
    std::vector<CoreCoord> ordered_cores;
    ordered_cores.reserve(core_to_data.size());
    for (const auto& [core, _] : core_to_data) {
        ordered_cores.push_back(core);
    }
    std::sort(ordered_cores.begin(), ordered_cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return (a.y < b.y) || (a.y == b.y && a.x < b.x);
    });

    // Flatten data from all cores into single vector with padding
    std::vector<uint32_t> flattened_data;
    flattened_data.reserve(core_to_data.size() * max_width);

    for (const auto& core : ordered_cores) {
        const auto& data = core_to_data.at(core);
        flattened_data.insert(flattened_data.end(), data.begin(), data.end());

        // Add padding if needed
        if (data.size() < max_width) {
            flattened_data.insert(flattened_data.end(), max_width - data.size(), 0);
        }
    }

    // Create host buffer and tensor
    auto config_buffer = tt::tt_metal::HostBuffer(std::move(flattened_data));
    return Tensor(std::move(config_buffer), tensor_shape, DataType::UINT32, Layout::ROW_MAJOR);
}

Tensor move_per_core_config_to_device(
    const Tensor& host_tensor, const CoreRangeSet& grid, distributed::MeshDevice* device) {
    // Create shard spec for the config tensor
    // Each core gets a row of the tensor
    const std::array<uint32_t, 2> shard_shape = {1, host_tensor.logical_shape()[1]};
    auto shard_spec = tt::tt_metal::ShardSpec(grid, shard_shape, ShardOrientation::ROW_MAJOR);

    // Create memory config for device tensor
    auto mem_config = MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec);

    return host_tensor.to_device(device, mem_config);
}

operation::ProgramWithCallbacks ReshardDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    std::unordered_map<CoreCoord, std::vector<uint32_t>> empty_data;
    Tensor local_rt_args_config_0, local_rt_args_config_1;
    if (output_tensor.shard_spec().has_value()) {
        auto output_grid = output_tensor.shard_spec().value().grid;
        auto output_cores = corerange_to_cores(output_grid);

        // Estimate number of RT args: Each page stride range needs:
        // 1. Header (4 values): input_addr, num_output_pages, num_ranges, output_page_offset
        // 2. Each range has around 9 values
        uint32_t header_size = 4;
        uint32_t values_per_range = 9;

        uint32_t num_cores = output_cores.size();
        uint32_t pages_per_shard =
            output_tensor.shard_spec().value().shape[0] * output_tensor.shard_spec().value().shape[1];
        uint32_t estimated_ranges = pages_per_shard / output_tensor.buffer()->page_size();

        uint32_t estimated_args = header_size + (estimated_ranges * values_per_range);
        uint32_t MAX_RT_ARGS_WIDTH = ((estimated_args + 31) / 32) * 32;

        for (auto core : output_cores) {
            empty_data[core].resize(MAX_RT_ARGS_WIDTH, 0);
        }
        auto rt_args_host_0 = construct_per_core_host_tensor(empty_data, MAX_RT_ARGS_WIDTH);
        auto rt_args_host_1 = construct_per_core_host_tensor(empty_data, MAX_RT_ARGS_WIDTH);

        local_rt_args_config_0 = move_per_core_config_to_device(rt_args_host_0, output_grid, output_tensor.device());
        local_rt_args_config_1 = move_per_core_config_to_device(rt_args_host_1, output_grid, output_tensor.device());
    }
    if (CMAKE_UNIQUE_NAMESPACE::is_valid_for_legacy_reshard(input_tensor, output_tensor.memory_config())) {
        return detail::reshard_multi_core(input_tensor, output_tensor, local_rt_args_config_0, local_rt_args_config_1);
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
