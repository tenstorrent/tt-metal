// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "clone_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::data_movement::clone {
tt::tt_metal::ProgramDescriptor CloneOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    ZoneScopedN("CloneOperation::ProgramFactory::create");
    using namespace tt::constants;
    using namespace tt::tt_metal::detail;
    using namespace tt::tt_metal;
    using namespace tt;

    ProgramDescriptor program;

    const auto& input = tensor_args.input;
    auto input_data_format = datatype_to_dataformat_converter(input.get_dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.get_dtype());
    bool convert_dtype = input_data_format != output_data_format;
    bool tilized = output.get_layout() == Layout::TILE;
    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tilized ? TileSize(data_format) : tensor.get_logical_shape()[-1] * tensor.element_size();
    };
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    uint32_t num_units = tilized ? output.volume() / TILE_HW : output.volume() / output.get_logical_shape()[-1];

    auto compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_units);

    auto src_cb_id = CBIndex::c_4;
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    program.cbs.push_back(CBDescriptor{
        .total_size = 2 * aligned_input_unit_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = src_cb_id,
            .data_format = input_data_format,
            .page_size = aligned_input_unit_size,
        }},
    });

    auto dst_cb_id = src_cb_id;
    if (convert_dtype) {
        dst_cb_id = CBIndex::c_20;
        uint32_t aligned_output_unit_size = round_up_to_mul32(output_unit_size);
        program.cbs.push_back(CBDescriptor{
            .total_size = 2 * aligned_output_unit_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = dst_cb_id,
                .data_format = output_data_format,
                .page_size = aligned_output_unit_size,
            }},
        });
    }

    auto input_buffer = input.buffer();
    auto output_buffer = output.buffer();
    bool input_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    constexpr size_t max_num_kernels = 4;
    program.kernels.resize(max_num_kernels);
    size_t num_kernels = 0;

    auto& read_kernel_descriptor = program.kernels[num_kernels++];
    read_kernel_descriptor.kernel_source =
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm.cpp",
    read_kernel_descriptor.core_ranges = all_cores.ranges(), read_kernel_descriptor.config = ReaderConfigDescriptor{},
    read_kernel_descriptor.reserve_runtime_args();

    auto& write_kernel_descriptor = program.kernels[num_kernels++];
    write_kernel_descriptor.kernel_source =
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp",
    write_kernel_descriptor.core_ranges = all_cores.ranges(), write_kernel_descriptor.config = WriterConfigDescriptor{};
    write_kernel_descriptor.reserve_runtime_args();

    if (tilized) {
        read_kernel_descriptor.compile_time_args = {
            (uint32_t)src_cb_id,
            (uint32_t)input_is_dram,
        };
        write_kernel_descriptor.compile_time_args = {
            (uint32_t)dst_cb_id,
            (uint32_t)output_is_dram,
        };
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_unit_size);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (uint32_t)log2(input_unit_size) : 0;
        read_kernel_descriptor.compile_time_args = {
            (uint32_t)src_cb_id,
            (uint32_t)input_is_dram,
            (uint32_t)src_stick_size_is_power_of_two,
            (uint32_t)src_log2_stick_size};
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_unit_size);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (uint32_t)log2(output_unit_size) : 0;
        write_kernel_descriptor.compile_time_args = {
            (uint32_t)dst_cb_id,
            (uint32_t)output_is_dram,
            (uint32_t)dst_stick_size_is_power_of_two,
            (uint32_t)dst_log2_stick_size};
    }

    if (convert_dtype) {
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
        auto create_compute_kernel = [&](const auto& core_group, uint32_t num_units_per_core) {
            if (!core_group.ranges().empty()) {
                auto& compute_kernel_descriptor = program.kernels[num_kernels++];
                compute_kernel_descriptor.kernel_source =
                    "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp";
                compute_kernel_descriptor.core_ranges = core_group.ranges();
                compute_kernel_descriptor.compile_time_args = {
                    (uint32_t)src_cb_id,
                    (uint32_t)dst_cb_id,
                    (uint32_t)num_units_per_core,
                };
                compute_kernel_descriptor.config = ComputeConfigDescriptor{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .math_approx_mode = math_approx_mode,
                };
            }
        };
        create_compute_kernel(core_group_1, num_units_per_core_group_1);
        create_compute_kernel(core_group_2, num_units_per_core_group_2);
    }

    uint32_t start_id = 0;
    uint32_t num_cores_group_1 = core_group_1.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    for (size_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t num_units_per_core = i < num_cores_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;
        if (tilized) {
            read_kernel_descriptor.runtime_args[core.x][core.y] = {
                (uint32_t)input_buffer->address(),
                (uint32_t)num_units_per_core,
                (uint32_t)start_id,
            };
            write_kernel_descriptor.runtime_args[core.x][core.y] = {
                (uint32_t)output_buffer->address(),
                (uint32_t)num_units_per_core,
                (uint32_t)start_id,
            };
        } else {
            read_kernel_descriptor.runtime_args[core.x][core.y] = {
                (uint32_t)input_buffer->address(),
                (uint32_t)input_unit_size,
                (uint32_t)num_units_per_core,
                (uint32_t)start_id,
            };
            write_kernel_descriptor.runtime_args[core.x][core.y] = {
                (uint32_t)output_buffer->address(),
                (uint32_t)output_unit_size,
                (uint32_t)num_units_per_core,
                (uint32_t)start_id,
            };
        }
        start_id += num_units_per_core;
    }

    program.kernels.resize(num_kernels);
    return program;
}
}  // namespace ttnn::operations::data_movement::clone
