// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "device/reshard_op.hpp"
#include "reshard.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor ReshardOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TensorSpec output_tensor_spec = optional_output_tensor.has_value() ? optional_output_tensor->tensor_spec()
                                                                       : TensorSpec(
                                                                             input_tensor.logical_shape(),
                                                                             TensorLayout::fromPaddedShape(
                                                                                 input_tensor.dtype(),
                                                                                 input_tensor.layout(),
                                                                                 memory_config,
                                                                                 input_tensor.logical_shape(),
                                                                                 input_tensor.padded_shape()));

    auto device = input_tensor.device();
    auto output_tensor = create_device_tensor(output_tensor_spec, device);

    bool is_valid_2d = is_valid_for_2d_reshard(input_tensor, memory_config);
    bool not_generic =
        (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
         memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED);
    not_generic = not_generic || (input_tensor.layout() == Layout::ROW_MAJOR &&
                                  input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                                  memory_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);
    std::unordered_map<CoreCoord, std::vector<uint32_t>> rt_config_map_0;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> rt_config_map_1;
    std::vector<Tensor> inputs;
    inputs.push_back(input_tensor);
    if (is_valid_2d && !not_generic && output_tensor.shard_spec().has_value()) {
        std::unordered_map<CoreCoord, std::vector<detail::PageStride>> output_core_to_page_range_pair;
        if (input_tensor.buffer()->page_size() != output_tensor.buffer()->page_size()) {
            output_core_to_page_range_pair =
                get_core_page_ranges_diff_width(input_tensor.buffer(), output_tensor.buffer(), input_tensor);
        } else {
            output_core_to_page_range_pair = get_core_page_ranges(input_tensor.buffer(), output_tensor.buffer());
        }
        const auto& input = input_tensor;
        const auto& output = output_tensor;
        auto output_shard_spec = output.shard_spec().value();
        auto all_cores = output_shard_spec.grid;
        auto grid = input.buffer()->buffer_type() == BufferType::DRAM ? device->dram_grid_size()
                                                                      : device->compute_with_storage_grid_size();
        auto input_core_type = input.buffer()->core_type();
        auto cores =
            corerange_to_cores(all_cores, std::nullopt, output_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

        uint32_t total_size, page_size, unit_size;
        auto output_shard_shape = output_shard_spec.shape;
        auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

        if (input.layout() == Layout::TILE) {
            page_size = tt::tt_metal::detail::TileSize(data_format);
            unit_size = page_size;
            total_size = output_shard_spec.numel() / tt::constants::TILE_HW * unit_size;
        } else {
            // For ROW_MAJOR, use base page size from GCD calculation
            uint32_t input_page_size = input.buffer()->page_size();
            uint32_t output_page_size = output.buffer()->page_size();
            uint32_t base_page_size = std::gcd(input_page_size, output_page_size);

            unit_size = base_page_size;
            page_size = base_page_size;
            total_size = output_shard_shape[0] * output_shard_shape[1] * output.element_size();
        }
        std::vector<uint32_t> physical_core_coords;
        physical_core_coords.reserve(grid.x * grid.y);
        for (uint32_t i = 0; i < grid.x; i++) {
            auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(i, 0), input_core_type);
            physical_core_coords.push_back(physical_input_core.x);
        }
        for (uint32_t i = 0; i < grid.y; i++) {
            auto physical_input_core = device->virtual_core_from_logical_core(CoreCoord(0, i), input_core_type);
            physical_core_coords.push_back(physical_input_core.y);
        }
        for (const auto& core : cores) {
            const auto& page_stride_vector = output_core_to_page_range_pair.at(core);
            auto runtime_args_0 = get_runtime_args_for_given_ranges(
                physical_core_coords,
                page_stride_vector,
                0,
                input.buffer()->address(),
                0,
                tt::div_up(page_stride_vector.size(), 2));
            auto output_page_offset =
                runtime_args_0[physical_core_coords.size() + 1];  // offset is equivalent to number of pages output in
                                                                  // previous risc core
            rt_config_map_0[core] = runtime_args_0;
            auto runtime_args_1 = get_runtime_args_for_given_ranges(
                physical_core_coords,
                page_stride_vector,
                output_page_offset,
                input.buffer()->address(),
                tt::div_up(page_stride_vector.size(), 2),
                page_stride_vector.size());
            rt_config_map_1[core] = runtime_args_1;
        }
        bool rt_gt_341 = false;
        for (const auto& [core, rt_args] : rt_config_map_0) {
            if (rt_args.size() > MAX_RUNTIME_ARGS || rt_config_map_1[core].size() > MAX_RUNTIME_ARGS) {
                rt_gt_341 = true;
                break;
            }
        }
        if (rt_gt_341) {
            auto runtime_args_tensor_0 = construct_per_core_host_tensor(rt_config_map_0);
            auto runtime_args_tensor_1 = construct_per_core_host_tensor(rt_config_map_1);

            auto device_runtime_args_0 =
                move_per_core_config_to_device(runtime_args_tensor_0, output_shard_spec.grid, device);
            auto device_runtime_args_1 =
                move_per_core_config_to_device(runtime_args_tensor_1, output_shard_spec.grid, device);
            inputs.push_back(device_runtime_args_0);
            inputs.push_back(device_runtime_args_1);
        }
    }
    bool has_output_tensor = optional_output_tensor.has_value();
    if (has_output_tensor) {
        return operation::run(
                   ReshardDeviceOperation{.output_mem_config = memory_config}, inputs, {}, {optional_output_tensor})
            .at(0);
    }
    return operation::run(ReshardDeviceOperation{.output_mem_config = memory_config}, inputs, {}, {output_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
