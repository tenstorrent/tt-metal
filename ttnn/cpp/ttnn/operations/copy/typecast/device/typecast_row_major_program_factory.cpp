// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_row_major_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::copy::program {

using namespace tt::constants;

TypecastRowMajorProgramFactory::cached_program_t TypecastRowMajorProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const DataType input_dtype = args.input_dtype;
    const DataType output_dtype = args.output_dtype;

    Program program{};

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t page_size = input.buffer()->page_size();
    const tt::DataFormat cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    const uint32_t page_size_output = output.buffer()->page_size();

    // For row-major layout, each row is a page
    // Use buffer's num_pages to get the correct count (handles padding correctly)
    const uint32_t num_pages = input.buffer()->num_pages();

    auto* device = input.device();
    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_pages);

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_pages = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_pages * page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, page_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);

    const uint32_t output_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t num_output_pages = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_pages * page_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, page_size_output);
    CreateCircularBuffer(program, all_cores, cb_output_config);

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    reader_compile_time_args.push_back(page_size);
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    writer_compile_time_args.push_back(page_size_output);
    writer_compile_time_args.push_back(output_cb_index);

    KernelHandle typecast_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_row_major.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle typecast_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_row_major.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_pages_per_core_group_1,  // per_core_block_cnt (number of pages)
        src0_cb_index,
        output_cb_index};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    constexpr bool math_approx_mode = false;

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const char* path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast_row_major.cpp";

    CreateKernel(
        program,
        path,
        core_group_1,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_pages_per_core_group_2,  // per_core_block_cnt
            src0_cb_index,
            output_cb_index};

        CreateKernel(
            program,
            path,
            core_group_2,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = unary_defines});
    }

    for (uint32_t i = 0, num_pages_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        SetRuntimeArgs(
            program, typecast_reader_kernel_id, core, {src_buffer->address(), num_pages_per_core, num_pages_written});

        SetRuntimeArgs(
            program, typecast_writer_kernel_id, core, {dst_buffer->address(), num_pages_per_core, num_pages_written});
        num_pages_written += num_pages_per_core;
    }

    return cached_program_t{
        std::move(program), {typecast_reader_kernel_id, typecast_writer_kernel_id, num_cores, num_cores_y}};
}

void TypecastRowMajorProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& typecast_reader_kernel_id = cached_program.shared_variables.typecast_reader_kernel_id;
    auto& typecast_writer_kernel_id = cached_program.shared_variables.typecast_writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;

    Program& program = cached_program.program;

    const Tensor& input = tensor_args.input;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            tt::tt_metal::RuntimeArgsData& runtime_args = GetRuntimeArgs(program, typecast_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            tt::tt_metal::RuntimeArgsData& runtime_args = GetRuntimeArgs(program, typecast_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::copy::program
