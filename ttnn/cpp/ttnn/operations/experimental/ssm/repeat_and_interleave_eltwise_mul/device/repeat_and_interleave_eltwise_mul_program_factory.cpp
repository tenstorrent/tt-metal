// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_and_interleave_eltwise_mul_program_factory.hpp"

#include <map>
#include <string>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {
constexpr uint32_t ONE_TILE = 1;
}  // namespace

tt::tt_metal::ProgramDescriptor RepeatAndInterleaveEltwiseMulProgramFactory::create_descriptor(
    const RepeatMulParams& operation_attributes, const RepeatMulInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    auto& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();

    Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    // Parallelize on bshape[-1]
    auto num_output_blocks_total = bshape[-1] / TILE_WIDTH;
    const bool row_major = false;
    auto device_compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(device_compute_with_storage_grid_size, num_output_blocks_total, row_major);

    uint32_t g1_numcores = core_group_1.num_cores();
    std::vector<CoreCoord> cores = grid_to_cores(
        num_cores, device_compute_with_storage_grid_size.x, device_compute_with_storage_grid_size.y, row_major);

    // Circular buffer indices
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t cb0_tiles = ONE_TILE * 2;  // double buffer

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t cb1_tiles = ONE_TILE * 2;  // double buffer

    uint32_t output_cb_index = 16;
    uint32_t output_cb_tiles = ONE_TILE * 2;  // double buffer

    uint32_t interm_num_tiles = ONE_TILE * 2;  // double buffer
    uint32_t interm_cb_size = interm_num_tiles * interm_single_tile_size;
    uint32_t cb_intermed0_index = tt::CBIndex::c_24;  // cb_in0_transposed
    uint32_t cb_intermed1_index = tt::CBIndex::c_25;  // cb_in1_transposed
    uint32_t cb_intermed2_index = tt::CBIndex::c_26;  // cb_in1_bcast_row
    uint32_t cb_intermed3_index = tt::CBIndex::c_27;  // cb_out_transposed

    // Compile time args
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(src1_cb_index),
        static_cast<uint32_t>(cb_intermed1_index),
        static_cast<uint32_t>(cb_intermed2_index),
    };
    tt::tt_metal::TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(src1_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(output_cb_index),
    };
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args);
    std::vector<uint32_t> compute_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(src1_cb_index),
        static_cast<uint32_t>(output_cb_index),
        static_cast<uint32_t>(cb_intermed0_index),
        static_cast<uint32_t>(cb_intermed1_index),
        static_cast<uint32_t>(cb_intermed2_index),
        static_cast<uint32_t>(cb_intermed3_index),
    };

    std::map<std::string, std::string> ssm_eltwise_defines;
    if (ashape[-1] == TILE_WIDTH) {
        ssm_eltwise_defines["REPEAT_IN0"] = "1";
    }
    if (bshape[-1] == HIDDEN_SIZE) {
        ssm_eltwise_defines["REPEAT_INTERLEAVE_IN1"] = "1";
    }

    // Convert std::map to KernelDescriptor::Defines (vector of pairs, deterministic order from map).
    KernelDescriptor::Defines descriptor_defines;
    descriptor_defines.reserve(ssm_eltwise_defines.size());
    for (const auto& [k, v] : ssm_eltwise_defines) {
        descriptor_defines.emplace_back(k, v);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    // Create circular buffers
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb0_tiles * in0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb1_tiles * in1_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_data_format,
            .page_size = output_single_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_intermed0_index),
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_intermed1_index),
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_intermed2_index),
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size}}}});

    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_cb_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_intermed3_index),
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size}}}});

    // Load kernels
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
        "reader_ssm_eltwise_mul.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.defines = descriptor_defines;
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    reader_kernel_desc.runtime_args.reserve(cores.size());

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
        "writer_ssm_eltwise_mul.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};
    writer_kernel_desc.runtime_args.reserve(cores.size());

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/"
        "ssm_eltwise_mul.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_args);
    compute_kernel_desc.defines = std::move(descriptor_defines);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = operation_attributes.math_fidelity, .fp32_dest_acc_en = false, .math_approx_mode = false};
    compute_kernel_desc.runtime_args.reserve(cores.size());

    // Set runtime args per core
    uint32_t num_blocks_per_core = 0;
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        // reader: (src0_addr, src1_addr, num_blocks_per_core, num_blocks_written,
        //          bshape[2]/TILE_HEIGHT, bshape[-1]/TILE_WIDTH, ashape[-1]/TILE_WIDTH)
        reader_kernel_desc.emplace_runtime_args(
            cores[i],
            {src0_buffer,
             src1_buffer,
             num_blocks_per_core,
             num_blocks_written,
             static_cast<uint32_t>(bshape[2] / TILE_HEIGHT),
             static_cast<uint32_t>(bshape[-1] / TILE_WIDTH),
             static_cast<uint32_t>(ashape[-1] / TILE_WIDTH)});

        // writer: (dst_addr, num_tiles, start_id, bshape[2]/TILE_HEIGHT, hidden_size)
        // update writer's num_tiles based on input_b already repeat_interleaved or not
        uint32_t writer_num_tiles = num_blocks_per_core;
        uint32_t writer_start_id = num_blocks_written;
        if (bshape[-1] == HIDDEN_SIZE) {
            writer_num_tiles = num_blocks_per_core * TILE_WIDTH;
            writer_start_id = num_blocks_written * TILE_WIDTH;
        }
        writer_kernel_desc.emplace_runtime_args(
            cores[i],
            {out_buffer,
             writer_num_tiles,
             writer_start_id,
             static_cast<uint32_t>(bshape[2] / TILE_HEIGHT),
             HIDDEN_SIZE});

        // compute: (num_blocks_per_core, bshape[2]/TILE_HEIGHT)
        compute_kernel_desc.emplace_runtime_args(
            cores[i], {num_blocks_per_core, static_cast<uint32_t>(bshape[2] / TILE_HEIGHT)});

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
