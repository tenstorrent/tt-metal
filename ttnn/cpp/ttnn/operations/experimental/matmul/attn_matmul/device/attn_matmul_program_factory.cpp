// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_matmul_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor AttnMatmulProgramFactory::create_descriptor(
    const AttnMatmulParams& operation_attributes, const AttnMatmulInputs& tensor_args, Tensor& tensor_return_value) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::tt_metal::IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32
                                            ? tt::DataFormat::Float32
                                            : tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    if (in0_data_format == tt::DataFormat::Float32 or in1_data_format == tt::DataFormat::Float32 or
        output_data_format == tt::DataFormat::Float32) {
        TT_FATAL(fp32_dest_acc_en == true, "when inputs/output are in fp32 format, fp32_dest_acc_en must be set");
    }

    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug(tt::LogOp, "in0_data_format: {}", in0_data_format);
    log_debug(tt::LogOp, "in1_data_format: {}", in1_data_format);
    log_debug(tt::LogOp, "interm_data_format: {}", interm_data_format);
    log_debug(tt::LogOp, "output_data_format: {}", output_data_format);

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* src1_buffer = b.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // A block of work is one MtNt
    uint32_t num_cores_y = operation_attributes.compute_with_storage_grid_size.y;
    auto num_output_blocks_total = ashape[1];  // ashape[1] is Q num_heads; only parallelize on this
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_blocks_per_core_group_1,
         num_output_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(
                operation_attributes.compute_with_storage_grid_size, num_output_blocks_total);

    CoreRangeSet all_device_cores{CoreRange(
        {0, 0}, {device->compute_with_storage_grid_size().x - 1, device->compute_with_storage_grid_size().y - 1})};
    uint32_t total_num_cores = device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;

    // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
    const bool transpose_hw_bool = operation_attributes.transpose_hw.value_or(false);
    const uint32_t num_tokens_val =
        operation_attributes.num_tokens.value_or(0);  // should not be nullopt if transpose_hw=true
    constexpr uint32_t num_rows_in_one_tile = 32;

    uint32_t Mt = ashape[2] / TILE_HEIGHT;
    uint32_t Kt = ashape[3] / TILE_WIDTH;
    // For transpose_hw=true, in1_Kt is same as in0_Kt but on bshape[3]
    // For transpose_hw=false, in1_Kt is on bshape[2] but represents the max cache length to read from (ie. may not
    // equal in0_Kt)
    uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2] / TILE_HEIGHT;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val / TILE_HEIGHT : bshape[3] / TILE_WIDTH;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;
    uint32_t in1_KtNt_stride = transpose_hw_bool ? bshape[2] / TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
    uint32_t in1_KtNt_skip = transpose_hw_bool ? (bshape[2] / TILE_HEIGHT - 1) * in1_Kt : (in1_Kt - Kt) * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    // ---- Circular buffers ----
    // cb_src0's total_size = Kt * in0_single_tile_size depends on the input shape;
    // padded_shape is folded into compute_program_hash() so each unique Kt keeps
    // its own cache entry.  CB total_size/page_size are not patched on cache hit
    // — the cached descriptor already carries the correct values.
    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = Kt * in0_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
        }}},
    });

    constexpr uint8_t src1_cb_index = tt::CBIndex::c_1;
    const uint32_t cb1_num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb1_num_input_tiles * in1_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
        }}},
    });

    constexpr uint8_t cb_intermed0_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_intermed0_index,
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });

    constexpr uint8_t cb_intermed1_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_intermed1_index,
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });

    constexpr uint8_t cb_intermed2_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_intermed2_index,
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_5;
    const uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    // ---- Kernels ----
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(transpose_hw_bool),
        static_cast<uint32_t>(fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32)};
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/dataflow/"
        "reader_transformer_attn_matmul.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_args = {
        static_cast<uint32_t>(transpose_hw_bool),  // transpose_hw for matmul_init
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul/device/kernels/compute/transformer_attn_matmul.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // ---- Per-core runtime args (idle cores get zeroed args) ----
    reader_desc.runtime_args.reserve(total_num_cores);
    writer_desc.runtime_args.reserve(total_num_cores);
    compute_desc.runtime_args.reserve(total_num_cores);

    uint32_t num_output_blocks_per_core = 0;
    for (uint32_t i = 0, num_blocks_written = 0; i < total_num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        if (core_group_1.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_2;
        } else {
            reader_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            compute_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{0, 0, 0, 0});
            writer_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{0, 0, 0});
            continue;
        }

        reader_desc.runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                src0_addr,
                src1_addr,
                Mt,
                Kt,
                Nt,
                MtKt,
                in1_KtNt_skip,
                in1_KtNt_stride * num_rows_in_one_tile,
                num_output_blocks_per_core,
                num_blocks_written * MtKt,  // itileA_start
                0,                          // itileB_start; always read same in1 per core
            });
        compute_desc.runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                1,                                  // B
                1,                                  // Mt
                Kt,                                 // Kt
                num_output_blocks_per_core * MtNt,  // Nt
            });
        writer_desc.runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                dst_addr,
                num_output_blocks_per_core * MtNt,
                num_blocks_written * MtNt,
            });
        num_blocks_written += num_output_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
