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

WhereDeviceOperation::BroadcastScalarsWhereProgram::cached_program_t
WhereDeviceOperation::BroadcastScalarsWhereProgram::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& a = tensor_args.input_tensor_a;

    auto& output = tensor_return_value;
    const auto& op_type = operation_attributes.ternary_op_type;

    Program program{};
    IDevice* device = a.device();
    const auto& all_device_cores = operation_attributes.worker_grid;

    // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
    uint32_t src0_bank_id = 0;
    uint32_t dst_bank_id = 0;

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 1;
    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(
                              num_input_tiles * src0_single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                              .set_page_size(src0_cb_index, src0_single_tile_size);
    CBHandle cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    auto cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * src0_single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, src0_single_tile_size);
    CBHandle cb_output = CreateCircularBuffer(program, all_device_cores, cb_output_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/ternary/device/where_prim/kernel/dataflow/"
        "broadcast_scalars_reader_kernel.cpp",
        all_device_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/ternary/device/where_prim/kernel/dataflow/writer.cpp",
        all_device_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};
    /* Use the add_tiles operation in the compute kernel */
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/ternary/device/where_prim/kernel/compute/broadcast_scalars_where_kernel.cpp",
        all_device_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });

    set_eltwise_unary_runtime_args<true>(
        program,
        a,
        output,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        cb_src0,
        cb_output,
        all_device_cores,
        src0_single_tile_size,
        dst_single_tile_size);
    return {
        std::move(program),
        {reader_kernel_id,
         writer_kernel_id,
         compute_kernel_id,
         cb_src0,
         cb_output,
         all_device_cores,
         src0_single_tile_size,
         dst_single_tile_size}};
}

void WhereDeviceOperation::BroadcastScalarsWhereProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {};
}  // namespace ttnn::operations::ternary
