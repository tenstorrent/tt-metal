// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/work_split.hpp"
#include "uniform_device_operation.hpp"

namespace ttnn::operations::uniform {

uint32_t get_random_seed() {
    std::mt19937 rng(std::time(0));
    std::uniform_int_distribution d(1, 1 << 20);
    return d(rng);
}

UniformDeviceOperation::ProgramFactory::cached_program_t UniformDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt;

    Program program;
    const auto& device = output.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_units = output.volume() / TILE_HW;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_units);

    DataType output_dtype = output.get_dtype();
    auto dst_cb_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(dst_cb_data_format);
    const uint32_t uint32_tile_size = tile_size(tt::DataFormat::UInt32);

    constexpr uint32_t in_out_num_tiles = 1;
    constexpr uint32_t intermed_num_tiles = 2;

    // TODO(Shaw): Eliminate the need for this trigger. I tested changing CB::c_out0 to CB::c_in0, and it still works
    // fine.
    uint32_t src_cb_id = CB::c_in0;  // Required to trigger hardware PRNG, set to the lowest possible value
    auto src_cb_config = CircularBufferConfig(4, {{src_cb_id, tt::DataFormat::UInt32}}).set_page_size(src_cb_id, 4);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);

    uint32_t intermed_cb_id = CB::c_intermed0;
    auto intermed_cb_config =
        CircularBufferConfig(intermed_num_tiles * uint32_tile_size, {{intermed_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(intermed_cb_id, uint32_tile_size);
    auto intermed_cb = CreateCircularBuffer(program, all_cores, intermed_cb_config);

    uint32_t dst_cb_id = CB::c_out0;
    auto dst_cb_config = CircularBufferConfig(in_out_num_tiles * dtype_tile_size, {{dst_cb_id, dst_cb_data_format}})
                             .set_page_size(dst_cb_id, dtype_tile_size);
    auto dst_cb = CreateCircularBuffer(program, all_cores, dst_cb_config);

    bool output_is_dram = output.buffer()->buffer_type() == BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {intermed_cb_id, dst_cb_id, output_is_dram};
    std::vector<uint32_t> compute_compile_time_args = {intermed_cb_id};

    std::map<std::string, std::string> writer_defines;
    switch (output_dtype) {
        case DataType::BFLOAT16: writer_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::FLOAT32: writer_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/uniform/device/kernels/writer_uniform.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = true,  // Must always be set to `true`; otherwise, the generated float numbers will be
                                       // constrained to the range [0.4, 0.5]
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
        });

    uint32_t start_id = 0;
    uint32_t num_cores_group_1 = core_group_1.num_cores();
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);

    for (size_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        uint32_t num_units_per_core = i < num_cores_group_1 ? num_units_per_core_group_1 : num_units_per_core_group_2;

        std::vector<uint32_t> compute_runtime_args = {get_random_seed(), start_id, num_units_per_core};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        union {
            float f;
            uint32_t u;
        } f2u_from, f2u_to;
        f2u_from.f = operation_attributes.from;
        f2u_to.f = operation_attributes.to - 1e-10f;  // Subtract small value to ensure < 'to'

        std::vector<uint32_t> writer_runtime_args = {
            output.buffer()->address(), f2u_from.u, f2u_to.u, start_id, num_units_per_core};
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        start_id += num_units_per_core;
    }

    return {
        std::move(program),
        {
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cores = cores,
        }};
}

void UniformDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    auto output_buffer_address = output.buffer()->address();

    for (const auto& core : cores) {
        GetRuntimeArgs(program, compute_kernel_id, core)[0] = get_random_seed();
        GetRuntimeArgs(program, writer_kernel_id, core)[0] = output_buffer_address;
    }
}

}  // namespace ttnn::operations::uniform
