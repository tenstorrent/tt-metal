// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/operations/experimental/where/device/program_factory/element_wise_multi_core_where_program.hpp"
#include "ttnn/operations/experimental/where/device/program_factory/elemwise_factory_common.hpp"
#include "ttnn/operations/experimental/where/device/kernel/dataflow/elemwise_reader_kernel_args.hpp"
#include "ttnn/operations/experimental/where/device/kernel/dataflow/elemwise_writer_kernel_args.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::ternary::experimental {

ElementWiseMultiCoreWhereProgram::cached_program_t ElementWiseMultiCoreWhereProgram::create(
    const where_ttt_args::operation_attributes_type& operation_attributes,
    const where_ttt_args::tensor_args_type& tensor_args,
    where_ttt_args::tensor_return_value_type& tensor_return_value) {
    using namespace ttnn::kernel::eltwise::where_args;
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& conditional_tensor = tensor_args.input_tensor_a;
    const auto& true_values_tensor = tensor_args.input_tensor_b;
    const auto& false_values_tensor = tensor_args.input_tensor_c;
    auto& output = tensor_return_value;

    Program program{};
    const auto& all_device_cores = operation_attributes.worker_grid;

    TT_ASSERT(
        conditional_tensor.get_dtype() == true_values_tensor.get_dtype(),
        "Mismatched data types: 'conditional_tensor' and 'true_values_tensor' must have the same dtype.");

    TT_ASSERT(
        conditional_tensor.get_dtype() == false_values_tensor.get_dtype(),
        "Mismatched data types: 'conditional_tensor' and 'false_values_tensor' must have the same dtype.");

    TT_ASSERT(
        conditional_tensor.get_dtype() == output.get_dtype(),
        "Mismatched data types: 'conditional_tensor' and 'output' must have the same dtype.");

    TT_ASSERT(
        conditional_tensor.get_dtype() == DataType::BFLOAT16,
        "Invalid data type: expected BFLOAT16 for 'conditional_tensor'.");

    uint32_t single_tile_size =
        tt_metal::detail::TileSize(tt_metal::datatype_to_dataformat_converter(conditional_tensor.get_dtype()));
    constexpr uint32_t num_input_tiles = 1;

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    auto cb_src1_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src1_cb_index, single_tile_size);
    CBHandle cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    constexpr uint32_t src2_cb_index = tt::CBIndex::c_2;
    auto cb_src2_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src2_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src2_cb_index, single_tile_size);
    CBHandle cb_src2 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src2_config);

    constexpr uint32_t src3_cb_index = tt::CBIndex::c_3;
    auto cb_src3_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src3_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src3_cb_index, single_tile_size);
    CBHandle cb_src3 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src3_config);

    constexpr uint32_t src4_cb_index = tt::CBIndex::c_4;
    auto cb_src4_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src4_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src4_cb_index, single_tile_size);
    CBHandle cb_src4 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src4_config);

    constexpr uint32_t src5_cb_index = tt::CBIndex::c_5;
    auto cb_src5_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src5_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src5_cb_index, single_tile_size);
    CBHandle cb_src5 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src5_config);

    constexpr uint32_t src6_cb_index = tt::CBIndex::c_6;
    auto cb_src6_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src6_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src6_cb_index, single_tile_size);
    CBHandle cb_src6 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src6_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_7;
    constexpr uint32_t num_output_tiles = 2;
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(
                                num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                                .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = CreateCircularBuffer(program, all_device_cores, cb_output_config);

    CompileTimeReaderKernelArgs reader_compile_time_args = {
        .cond_tensor_cb = src0_cb_index,
        .true_tensor_cb = src1_cb_index,
        .false_tensor_cb = src2_cb_index,
        .is_cond_tensor_in_dram = conditional_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM,
        .is_true_tensor_in_dram = true_values_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM,
        .is_false_tensor_in_dram = false_values_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM};

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    std::map<string, string> reader_defines;
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernel/dataflow/elemwise_reader_kernel.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(ttnn::kernel_utils::to_vector(reader_compile_time_args), reader_defines));

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    CompileTimeWriterKernelArgs writer_compile_time_args = {
        .cb_dst = output_cb_index, .is_dst_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM};
    std::map<string, string> writer_defines;
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernel/dataflow/elemwise_writer_kernel.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(ttnn::kernel_utils::to_vector(writer_compile_time_args), writer_defines));

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernel/compute/elemwise_where_kernel.cpp",
        all_device_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {}});

    set_eltwise_ternary_runtime_args<true>(
        program,
        conditional_tensor,
        true_values_tensor,
        false_values_tensor,
        output,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        all_device_cores);
    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         cb_src0,
         cb_src1,
         cb_src2,
         cb_output,
         all_device_cores,
         single_tile_size,
         single_tile_size,
         single_tile_size,
         single_tile_size}};
}

void ElementWiseMultiCoreWhereProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const where_ttt_args::operation_attributes_type& operation_attributes,
    const where_ttt_args::tensor_args_type& tensor_args,
    where_ttt_args::tensor_return_value_type& tensor_return_value) {
    const auto& sh_var = cached_program.shared_variables;
    set_eltwise_ternary_runtime_args<false>(
        cached_program.program,
        tensor_args.input_tensor_a,
        tensor_args.input_tensor_b,
        tensor_args.input_tensor_c,
        tensor_return_value,
        sh_var.reader_kernel_id,
        sh_var.writer_kernel_id,
        sh_var.eltwise_kernel_id,
        sh_var.all_device_cores);
}
}  // namespace ttnn::operations::ternary::experimental
