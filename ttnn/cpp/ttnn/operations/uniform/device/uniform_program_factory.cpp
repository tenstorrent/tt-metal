// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "uniform_device_operation.hpp"

namespace ttnn::operations::uniform {

using namespace tt;
using namespace tt::tt_metal;

UniformDeviceOperation::Factory::cached_program_t UniformDeviceOperation::Factory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input) {
    Device* device = input.device();
    auto grid = CoreCoord(0, 0);
    int core_h = 1;

    uint32_t units_to_divide = input.volume() / constants::TILE_HEIGHT / constants::TILE_WIDTH;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    CommandQueue& cq = device->command_queue();
    Program program = Program();

    tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    constexpr uint32_t single_tile_size = 4 * 1024;

    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Int32}})
            .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, grid, cb_src0_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float32}})
            .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, grid, cb_output_config);

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/uniform/device/kernels/";
    const std::vector<uint32_t> reader_compile_time_args{};
    const std::string reader_file_path = kernels_dir_path + "reader.cpp";
    const std::vector<uint32_t> writer_compile_time_args{};
    const std::string writer_file_path = kernels_dir_path + "writer.cpp";
    const std::vector<uint32_t> compute_compile_time_args{};
    const std::string compute_file_path = kernels_dir_path + "uniform.cpp";

    KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program, reader_file_path, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
    KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program, writer_file_path, all_cores, WriterDataMovementConfig(writer_compile_time_args));
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        compute_file_path,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
        });

    CoreCoord core = {0, 0};
    SetRuntimeArgs(program, reader_kernel_id, core, {});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, writer_kernel_id, core, {input.buffer()->address()});
    // tt::operations::primary::CreateCircularBuffer(program, all_cores, data_format, {{CB::c_in0, 2, data}});

    return {
        std::move(program),
        {.unary_reader_kernel_id = reader_kernel_id,
         .unary_writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = core_h}};
}

void UniformDeviceOperation::Factory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    // const uint32_t target_addr = tensor_args.target_tensor.buffer()->address();
    // const uint32_t output_grad_addr = tensor_args.output_grad_tensor.buffer()->address();
    // const uint32_t weight_addr =
    //     tensor_args.weight_tensor.has_value() ? tensor_args.weight_tensor.value().buffer()->address() : 0;
    // const uint32_t ignore_index = operation_attributes.ignore_index;

    // const uint32_t input_grad_addr = tensor_return_value.buffer()->address();

    // for (uint32_t i = 0; i < num_cores; ++i) {
    //     CoreCoord core = {i / num_cores_y, i % num_cores_y};
    //     {
    //         auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
    //         runtime_args[0] = target_addr;
    //         runtime_args[1] = output_grad_addr;
    //         runtime_args[2] = weight_addr;
    //         runtime_args[3] = ignore_index;
    //     }

    //     {
    //         auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
    //         runtime_args[0] = input_grad_addr;
    //     }
    // }
}

}  // namespace ttnn::operations::uniform
