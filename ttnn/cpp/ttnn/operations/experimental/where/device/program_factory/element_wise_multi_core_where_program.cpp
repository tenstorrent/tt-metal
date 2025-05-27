// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/operations/experimental/where/device/where_device_operation.hpp"
#include "ttnn/operations/experimental/where/device/program_factory/elemwise_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::experimental::where {

WhereDeviceOperation::ElementWiseMultiCoreWhereProgram::cached_program_t
WhereDeviceOperation::ElementWiseMultiCoreWhereProgram::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    const auto& c = tensor_args.input_tensor_c;
    auto& output = tensor_return_value;

    Program program{};
    IDevice* device = a.device();
    const auto& all_device_cores = operation_attributes.worker_grid;

    // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
    uint32_t src0_bank_id = 0;
    uint32_t src1_bank_id = 0;
    uint32_t src2_bank_id = 0;
    uint32_t dst_bank_id = 0;

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b->get_dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat src2_cb_data_format = tt_metal::datatype_to_dataformat_converter(c->get_dtype());
    uint32_t src2_single_tile_size = tt_metal::detail::TileSize(src2_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 1;
    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src0_single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src0_cb_index, src0_single_tile_size);
    CBHandle cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    auto cb_src1_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src1_single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src1_cb_index, src1_single_tile_size);
    CBHandle cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    constexpr uint32_t src2_cb_index = tt::CBIndex::c_2;
    auto cb_src2_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src2_single_tile_size, {{src2_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src2_cb_index, src2_single_tile_size);
    CBHandle cb_src2 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src2_config);

    constexpr uint32_t src3_cb_index = tt::CBIndex::c_3;
    auto cb_src3_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src2_single_tile_size, {{src3_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src3_cb_index, src2_single_tile_size);
    CBHandle cb_src3 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src3_config);

    constexpr uint32_t src4_cb_index = tt::CBIndex::c_4;
    auto cb_src4_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src2_single_tile_size, {{src4_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src4_cb_index, src2_single_tile_size);
    CBHandle cb_src4 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src4_config);

    constexpr uint32_t src5_cb_index = tt::CBIndex::c_5;
    auto cb_src5_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src2_single_tile_size, {{src5_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src5_cb_index, src2_single_tile_size);
    CBHandle cb_src5 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src5_config);

    constexpr uint32_t src6_cb_index = tt::CBIndex::c_6;
    auto cb_src6_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src2_single_tile_size, {{src6_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src6_cb_index, src2_single_tile_size);
    CBHandle cb_src6 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src6_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_7;
    constexpr uint32_t num_output_tiles = 2;
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(
                                num_output_tiles * dst_single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                                .set_page_size(output_cb_index, dst_single_tile_size);
    CBHandle cb_output = CreateCircularBuffer(program, all_device_cores, cb_output_config);

    std::map<string, string> reader_defines;
    bool block_or_width_sharded = false;
    bool src0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    bool src1_is_dram = b->buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    bool src2_is_dram = c->buffer()->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_is_dram,
        (std::uint32_t)src2_is_dram,
        (std::uint32_t)block_or_width_sharded};

    std::map<string, string> writer_defines;
    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernel/dataflow/elemwise_reader_kernel.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernel/dataflow/writer.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};
    /* Use the add_tiles operation in the compute kernel */
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/where/device/kernel/compute/elemwise_where_kernel.cpp",
        all_device_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args});

    set_eltwise_ternary_runtime_args<true>(
        program,
        a,
        *b,
        *c,
        output,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        cb_src0,
        cb_src1,
        cb_src2,
        cb_output,
        all_device_cores,
        src0_single_tile_size,
        src1_single_tile_size,
        src2_single_tile_size,
        dst_single_tile_size);
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
         src0_single_tile_size,
         src1_single_tile_size,
         src2_single_tile_size,
         dst_single_tile_size}};
}

void WhereDeviceOperation::ElementWiseMultiCoreWhereProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {};
}  // namespace ttnn::operations::experimental::where
