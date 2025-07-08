// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "bernoulli_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::bernoulli {

using namespace tt;
using namespace tt::tt_metal;

std::mt19937 rng(std::time(0));
std::uniform_int_distribution d(1, 1 << 20);

uint32_t get_random_seed() { return d(rng); }

BernoulliDeviceOperation::ProgramFactory::cached_program_t BernoulliDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& input = tensor_args.input;

    IDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    uint32_t num_cores_x = grid.x;
    uint32_t num_cores_y = grid.y;
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    Program program = Program();

    constexpr uint32_t num_tiles = 2;
    auto in_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t in_dtype_tile_size = tile_size(in_data_format);
    constexpr uint32_t in_cb_id = CBIndex::c_0;
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(num_tiles * in_dtype_tile_size, {{in_cb_id, in_data_format}})
            .set_page_size(in_cb_id, in_dtype_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_in_config);

    const uint32_t float32_tile_size = tile_size(tt::DataFormat::Float32);
    constexpr uint32_t intermed_cb_id = CBIndex::c_24;
    CircularBufferConfig cb_intermed_config =
        CircularBufferConfig(num_tiles * float32_tile_size, {{intermed_cb_id, tt::DataFormat::Float32}})
            .set_page_size(intermed_cb_id, float32_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed_config);

    auto out_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t out_dtype_tile_size = tile_size(out_data_format);
    constexpr uint32_t intermed1_cb_id = CBIndex::c_25;
    CircularBufferConfig cb_intermed1_config =
        CircularBufferConfig(1 * out_dtype_tile_size, {{intermed1_cb_id, out_data_format}})
            .set_page_size(intermed1_cb_id, out_dtype_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed1_config);

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/bernoulli/device/kernels/";
    const uint32_t input_is_dram = input.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
    const std::vector<uint32_t> reader_compile_time_args{in_cb_id, input_is_dram};
    const std::string reader_file_path = kernels_dir_path + "reader_bernoulli.cpp";
    const std::vector<uint32_t> compute_compile_time_args{intermed_cb_id};
    const std::string compute_file_path = kernels_dir_path + "compute_bernoulli.cpp";
    const uint32_t output_is_dram = output.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
    const std::vector<uint32_t> writer_compile_time_args{in_cb_id, intermed_cb_id, intermed1_cb_id, output_is_dram};
    const std::string writer_file_path = kernels_dir_path + "writer_bernoulli.cpp";

    std::map<std::string, std::string> writer_defines;
    switch (input.dtype()) {
        case DataType::BFLOAT16: writer_defines["INPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::FLOAT32: writer_defines["INPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }
    switch (output.dtype()) {
        case DataType::BFLOAT16: writer_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::FLOAT32: writer_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program, reader_file_path, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
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
            .fp32_dest_acc_en =
                true,  // must always be true otherwise, generated float number are always in range of [0.4, 0.5]
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

        std::vector<uint32_t> reader_runtime_args = {input.buffer()->address(), tile_offset, units_per_core};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        // Each core has its own seed to increase the number of generated random numbers
        uint32_t seed = operation_attributes.seed != 0 ? operation_attributes.seed + i : get_random_seed();

        std::vector<uint32_t> compute_runtime_args = {seed, tile_offset, units_per_core};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        std::vector<uint32_t> writer_runtime_args = {output.buffer()->address(), tile_offset, units_per_core};
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        tile_offset += units_per_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = cores}};
}

void BernoulliDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t output_addr = output.buffer()->address();

    for (int i = 0; i < cores.size(); ++i) {
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
            runtime_args[0] = input_addr;
        }
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

}  // namespace ttnn::operations::bernoulli
