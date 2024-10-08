// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "common/constants.hpp"
#include "common/core_coord.h"
#include "tt_metal/common/work_split.hpp"
#include "uniform_device_operation.hpp"

namespace ttnn::operations::uniform {

using namespace tt;
using namespace tt::tt_metal;

UniformDeviceOperation::Factory::cached_program_t UniformDeviceOperation::Factory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    Device* device = output.device();
    auto grid = device->compute_with_storage_grid_size();
    int core_h = grid.y;

    uint32_t units_to_divide = output.volume() / constants::TILE_HEIGHT / constants::TILE_WIDTH;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    CommandQueue& cq = device->command_queue();
    Program program = Program();

    tt::DataFormat data_format = datatype_to_dataformat_converter(output.dtype());
    constexpr uint32_t single_tile_size = 4 * 1024;

    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Int32}})
            .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float32}})
            .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/uniform/device/kernels/";
    const std::vector<uint32_t> writer_compile_time_args{};
    const std::string writer_file_path = kernels_dir_path + "writer.cpp";
    const std::vector<uint32_t> compute_compile_time_args{};
    const std::string compute_file_path = kernels_dir_path + "uniform.cpp";

    KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program, writer_file_path, all_cores, WriterDataMovementConfig(writer_compile_time_args));
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        compute_file_path,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
        });

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::vector<uint32_t> compute_runtime_args = {tile_offset, units_per_core};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        union {
            float f;
            uint32_t u;
        } f2u_from, f2u_to;
        f2u_from.f = operation_attributes.from;
        f2u_to.f = operation_attributes.to;

        std::vector<uint32_t> writer_runtime_args = {
            output.buffer()->address(), f2u_from.u, f2u_to.u, tile_offset, units_per_core};
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        tile_offset += units_per_core;
    }

    return {std::move(program), {.writer_kernel_id = writer_kernel_id, .num_cores = num_cores, .num_cores_y = core_h}};
}

void UniformDeviceOperation::Factory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const uint32_t output_addr = tensor_return_value.buffer()->address();

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::operations::uniform
