// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "uniform_device_operation.hpp"

namespace ttnn::operations::uniform {

using namespace tt;
using namespace tt::tt_metal;

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed() -> uint32_t { return distribution(rng); }

UniformDeviceOperation::ProgramFactory::cached_program_t UniformDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();
    auto core_h = grid.y;

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    Program program = Program();

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);

    constexpr uint32_t in_out_num_tiles = 1;
    constexpr uint32_t intermed_num_tiles = 2;

    constexpr uint32_t intermed_cb_id = CBIndex::c_24;
    CircularBufferConfig cb_intermed_config =
        CircularBufferConfig(intermed_num_tiles * intermed_tile_size, {{intermed_cb_id, tt::DataFormat::Float32}})
            .set_page_size(intermed_cb_id, intermed_tile_size);
    CBHandle cb_intermed = tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed_config);

    constexpr uint32_t dst_cb_id = CBIndex::c_0;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(in_out_num_tiles * dtype_tile_size, {{dst_cb_id, out_data_format}})
            .set_page_size(dst_cb_id, dtype_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/uniform/device/kernels/";
    const uint32_t output_is_dram = output.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
    const std::vector<uint32_t> writer_compile_time_args{intermed_cb_id, dst_cb_id, output_is_dram};
    const std::string writer_file_path = kernels_dir_path + "writer_uniform.cpp";
    const std::vector<uint32_t> compute_compile_time_args{intermed_cb_id};
    const std::string compute_file_path = kernels_dir_path + "compute_uniform.cpp";

    std::map<string, string> writer_defines;
    switch (output_dtype) {
        case DataType::BFLOAT16: writer_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::FLOAT32: writer_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program, writer_file_path, all_cores, WriterDataMovementConfig(writer_compile_time_args, writer_defines));
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        compute_file_path,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                       // generated number out of range [from, to)
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
        });

    uint32_t tile_offset = 0;
    for (int i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        const float eps = 1e-6;
        union {
            float f;
            uint32_t u;
        } f2u_from, f2u_to;
        f2u_from.f = operation_attributes.from;
        f2u_to.f = operation_attributes.to - eps;  // -eps make sure that generated number is < operation_attributes.to

        // Each core has its own seed to increase the number of generated random numbers
        uint32_t seed = operation_attributes.seed != 0 ? operation_attributes.seed + i : get_random_seed();

        std::vector<uint32_t> compute_runtime_args = {seed, f2u_from.u, f2u_to.u, tile_offset, units_per_core};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        std::vector<uint32_t> writer_runtime_args = {output.buffer()->address(), tile_offset, units_per_core};
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.compute_kernel_id = compute_kernel_id, .writer_kernel_id = writer_kernel_id, .cores = cores}};
}

void UniformDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    const uint32_t output_addr = output.buffer()->address();

    for (int i = 0; i < cores.size(); ++i) {
        {
            auto& runtime_args = GetRuntimeArgs(program, compute_kernel_id, cores[i]);
            runtime_args[0] = operation_attributes.seed != 0 ? operation_attributes.seed + i : get_random_seed();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, cores[i]);
            runtime_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::operations::uniform
