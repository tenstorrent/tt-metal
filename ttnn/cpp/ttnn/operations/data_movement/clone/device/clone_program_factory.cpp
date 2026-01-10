// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "clone_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"

namespace ttnn::operations::data_movement::clone {
CloneOperation::ProgramFactory::cached_program_t CloneOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal::detail;
    using namespace tt::tt_metal;
    using namespace tt;

    Program program;

    const auto& input = tensor_args.input;
    auto input_data_format = datatype_to_dataformat_converter(input.dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    bool convert_dtype = input_data_format != output_data_format;
    bool tilized = output.layout() == Layout::TILE;
    auto compute_unit_size = [&](const auto& tensor, const auto& data_format) {
        return tilized ? tt::tile_size(data_format) : tensor.logical_shape()[-1] * tensor.element_size();
    };
    uint32_t input_unit_size = compute_unit_size(input, input_data_format);
    uint32_t output_unit_size = compute_unit_size(output, output_data_format);
    uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.logical_shape()[-1];

    auto output_memory_layout = output.memory_config().memory_layout();
    bool is_sharded = output_memory_layout != TensorMemoryLayout::INTERLEAVED;

    uint32_t num_cores;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_units_per_core_group_1;
    uint32_t num_units_per_core_group_2;
    uint32_t num_cores_x;
    uint32_t num_cores_y;

    if (is_sharded) {
        auto shard_spec = output.buffer()->shard_spec();
        all_cores = shard_spec.grid();
        num_cores = all_cores.num_cores();

        auto shard_shape = shard_spec.shape();
        uint32_t shard_height = shard_shape[0];
        uint32_t shard_width = shard_shape[1];

        if (tilized) {
            num_units_per_core_group_1 = (shard_height * shard_width) / TILE_HW;
        } else {
            num_units_per_core_group_1 = shard_height;
        }

        num_units_per_core_group_2 = 0;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();

        auto grid_size = all_cores.bounding_box();
        num_cores_x = grid_size.end_coord.x + 1;
        num_cores_y = grid_size.end_coord.y + 1;
    } else {
        auto compute_with_storage_grid_size = output.device()->compute_with_storage_grid_size();
        num_cores_x = compute_with_storage_grid_size.x;
        num_cores_y = compute_with_storage_grid_size.y;
        auto
            [num_cores_result,
             all_cores_result,
             core_group_1_result,
             core_group_2_result,
             num_units_per_core_group_1_result,
             num_units_per_core_group_2_result] = split_work_to_cores(compute_with_storage_grid_size, num_units);
        num_cores = num_cores_result;
        all_cores = all_cores_result;
        core_group_1 = core_group_1_result;
        core_group_2 = core_group_2_result;
        num_units_per_core_group_1 = num_units_per_core_group_1_result;
        num_units_per_core_group_2 = num_units_per_core_group_2_result;
    }

    auto alignment = input.buffer()->alignment();

    uint32_t src_cb_id = CBIndex::c_4;
    uint32_t aligned_input_unit_size = tt::align(input_unit_size, alignment);
    auto src_cb_config = CircularBufferConfig(2 * aligned_input_unit_size, {{src_cb_id, input_data_format}})
                             .set_page_size(src_cb_id, aligned_input_unit_size);
    CreateCircularBuffer(program, all_cores, src_cb_config);

    uint32_t dst_cb_id = src_cb_id;
    if (convert_dtype) {
        dst_cb_id = CBIndex::c_20;
        uint32_t aligned_output_unit_size = tt::align(output_unit_size, alignment);
        auto dst_cb_config = CircularBufferConfig(2 * aligned_output_unit_size, {{dst_cb_id, output_data_format}})
                                 .set_page_size(dst_cb_id, aligned_output_unit_size);
        CreateCircularBuffer(program, all_cores, dst_cb_config);
    }

    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    if (tilized) {
        reader_compile_time_args = {(uint32_t)src_cb_id};
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        writer_compile_time_args = {(uint32_t)dst_cb_id};
        TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);
    } else {
        reader_compile_time_args = {(uint32_t)src_cb_id, (uint32_t)input_unit_size};
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);
        writer_compile_time_args = {(uint32_t)dst_cb_id, (uint32_t)output_unit_size};
        TensorAccessorArgs(*output_buffer).append_to(writer_compile_time_args);
    }

    std::string read_kernel_path;
    std::string write_kernel_path;

    if (is_sharded) {
        read_kernel_path =
            tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_sharded.cpp"
                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm_sharded.cpp";
        write_kernel_path =
            tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_sharded.cpp"
                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm_sharded.cpp";
    } else {
        read_kernel_path = tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel.cpp"
                                   : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/read_kernel_rm.cpp";
        write_kernel_path = tilized ? "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel.cpp"
                                    : "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/write_kernel_rm.cpp";
    }

    auto read_kernel_id =
        CreateKernel(program, read_kernel_path, all_cores, ReaderDataMovementConfig(reader_compile_time_args, {}));

    auto write_kernel_id =
        CreateKernel(program, write_kernel_path, all_cores, WriterDataMovementConfig(writer_compile_time_args, {}));

    if (convert_dtype) {
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
        auto create_compute_kernel = [&](const auto& core_group, uint32_t num_units_per_core) {
            if (!core_group.ranges().empty()) {
                std::vector<uint32_t> compute_kernel_args = {
                    (uint32_t)src_cb_id,
                    (uint32_t)dst_cb_id,
                    (uint32_t)num_units_per_core,
                };
                CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp",
                    core_group,
                    ComputeConfig{
                        .math_fidelity = math_fidelity,
                        .fp32_dest_acc_en = fp32_dest_acc_en,
                        .dst_full_sync_en = dst_full_sync_en,
                        .math_approx_mode = math_approx_mode,
                        .compile_args = compute_kernel_args,
                    });
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

        if (is_sharded) {
            if (tilized) {
                SetRuntimeArgs(
                    program,
                    read_kernel_id,
                    core,
                    {
                        (uint32_t)input_buffer->address(),
                        (uint32_t)num_units_per_core,
                    });
                SetRuntimeArgs(
                    program,
                    write_kernel_id,
                    core,
                    {
                        (uint32_t)output_buffer->address(),
                        (uint32_t)num_units_per_core,
                    });
            } else {
                SetRuntimeArgs(
                    program,
                    read_kernel_id,
                    core,
                    {
                        (uint32_t)input_buffer->address(),
                        (uint32_t)input_unit_size,
                        (uint32_t)num_units_per_core,
                    });
                SetRuntimeArgs(
                    program,
                    write_kernel_id,
                    core,
                    {
                        (uint32_t)output_buffer->address(),
                        (uint32_t)output_unit_size,
                        (uint32_t)num_units_per_core,
                    });
            }
        } else {
            if (tilized) {
                SetRuntimeArgs(
                    program,
                    read_kernel_id,
                    core,
                    {
                        (uint32_t)input_buffer->address(),
                        (uint32_t)num_units_per_core,
                        (uint32_t)start_id,
                    });
                SetRuntimeArgs(
                    program,
                    write_kernel_id,
                    core,
                    {
                        (uint32_t)output_buffer->address(),
                        (uint32_t)num_units_per_core,
                        (uint32_t)start_id,
                    });
            } else {
                SetRuntimeArgs(
                    program,
                    read_kernel_id,
                    core,
                    {
                        (uint32_t)input_buffer->address(),
                        (uint32_t)input_unit_size,
                        (uint32_t)num_units_per_core,
                        (uint32_t)start_id,
                    });
                SetRuntimeArgs(
                    program,
                    write_kernel_id,
                    core,
                    {
                        (uint32_t)output_buffer->address(),
                        (uint32_t)output_unit_size,
                        (uint32_t)num_units_per_core,
                        (uint32_t)start_id,
                    });
            }
            start_id += num_units_per_core;
        }
    }
    return {std::move(program), {read_kernel_id, write_kernel_id, cores}};
}

void CloneOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& program = cached_program.program;
    const auto& read_kernel_id = cached_program.shared_variables.read_kernel_id;
    const auto& write_kernel_id = cached_program.shared_variables.write_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto input_buffer_address = tensor_args.input.buffer()->address();
    auto output_buffer_address = output.buffer()->address();
    for (const auto& core : cores) {
        GetRuntimeArgs(program, read_kernel_id, core)[0] = input_buffer_address;
        GetRuntimeArgs(program, write_kernel_id, core)[0] = output_buffer_address;
    }
}
}  // namespace ttnn::operations::data_movement::clone
