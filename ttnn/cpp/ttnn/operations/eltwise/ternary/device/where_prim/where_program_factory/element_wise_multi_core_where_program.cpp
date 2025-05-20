// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/operations/eltwise/ternary/device/where_prim/where_device_operation.hpp"
#include "ttnn/operations/eltwise/ternary/device/where_prim/where_program_factory/elemwise_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::ternary {

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
    const auto& op_type = operation_attributes.ternary_op_type;

    Program program{};
    IDevice* device = a.device();
    CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;
    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src2_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

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
                              num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    auto cb_src1_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src1_cb_index, single_tile_size);
    CBHandle cb_src1 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t src2_cb_index = tt::CBIndex::c_2;
    auto cb_src2_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * single_tile_size, {{src2_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src2_cb_index, single_tile_size);
    CBHandle cb_src2 = tt::tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    auto cb_output_config = tt::tt_metal::CircularBufferConfig(
                                num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                                .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = CreateCircularBuffer(program, core, cb_output_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/ternary/device/where_prim/kernel/dataflow/elemwise_reader_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/ternary/device/where_prim/kernel/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};
    /* Use the add_tiles operation in the compute kernel */
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/ternary/device/where_prim/kernel/compute/elemwise_where_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_dram_buffer->address(),
         src1_dram_buffer->address(),
         src2_dram_buffer->address(),
         src0_bank_id,
         src1_bank_id,
         src2_bank_id});
    SetRuntimeArgs(program, compute_kernel_id, core, {});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address(), dst_bank_id});

    const auto& all_device_cores = operation_attributes.worker_grid;
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
}  // namespace ttnn::operations::ternary
