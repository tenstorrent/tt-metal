// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_post_all_gather_welford_program_factory.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/math.hpp"

#include <optional>
#include <string>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

ProgramDescriptor PostAllGatherWelfordProgramFactory::create_descriptor(
    const DitLayernormPostAllGatherParams& operation_attributes,
    const DitLayernormPostAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& stats = tensor_args.stats;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t stats_tiles_cols = stats.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t tile_cols_per_device = 2;  // sum and sumsq
    const uint32_t num_devices = stats_tiles_cols / tile_cols_per_device;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");
    TT_FATAL(
        num_devices * tile_cols_per_device == stats_tiles_cols, "Number of devices must divide number of stats tiles");

    uint32_t num_tile_rows = NC * Ht;

    IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const uint32_t dst_reg_count = get_dest_reg_count(operation_attributes.compute_kernel_config);
    uint32_t block_size = dst_reg_count;  // stride by dest regs, like fused rmsnorm

    tt::DataFormat in_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat stats_data_format = datatype_to_dataformat_converter(stats.dtype());
    tt::DataFormat out_data_format = datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format =
        gamma.has_value() ? datatype_to_dataformat_converter(gamma.value().dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format =
        beta.has_value() ? datatype_to_dataformat_converter(beta.value().dtype()) : tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);

    Buffer* a_buffer = a.buffer();
    Buffer* stats_buffer = stats.buffer();
    Buffer* gamma_buffer = gamma.has_value() ? gamma.value().buffer() : nullptr;
    Buffer* beta_buffer = beta.has_value() ? beta.value().buffer() : nullptr;
    Buffer* dst_buffer = output.buffer();

    uint32_t Wt_round_up_block_size = tt::round_up(Wt, block_size);

    // Size circular buffers in terms of dst_reg_count (like fused rmsnorm)
    const uint32_t double_buffer = 2;
    const uint32_t in0_tiles = dst_reg_count * double_buffer;
    const uint32_t in1_tiles = stats_tiles_cols * double_buffer;
    const uint32_t in2_tiles = gamma.has_value() ? Wt_round_up_block_size : 0;
    const uint32_t in3_tiles = beta.has_value() ? Wt_round_up_block_size : 0;
    const uint32_t in4_tiles = 1;  // epsilon

    const uint32_t intermed0_tiles = tile_cols_per_device;
    const uint32_t intermed1_tiles = 1;
    const uint32_t intermed2_tiles = dst_reg_count;
    const uint32_t out0_tiles = dst_reg_count * double_buffer;

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    ProgramDescriptor desc;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)block_size,
        (std::uint32_t)stats_tiles_cols,
    };

    uint32_t gamma_page_size = 0;
    uint32_t gamma_is_row_major = 0;
    uint32_t beta_is_row_major = 0;
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        gamma_page_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        bool gamma_page_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(gamma_page_size);
        TT_FATAL(gamma_page_size_is_power_of_two, "Only power of 2 gammas are supported");
        gamma_is_row_major = 1;
    } else if (gamma.has_value() and gamma.value().layout() == Layout::TILE) {
        gamma_page_size = gamma_single_tile_size;
    }
    uint32_t beta_page_size = 0;
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        beta_page_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        bool beta_page_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(beta_page_size);
        TT_FATAL(beta_page_size_is_power_of_two, "Only power of 2 betas are supported");
        beta_is_row_major = 1;
    } else if (beta.has_value() and beta.value().layout() == Layout::TILE) {
        beta_page_size = beta_single_tile_size;
    }
    uint32_t gamma_element_size = gamma.has_value() ? gamma.value().element_size() : 0;
    uint32_t beta_element_size = beta.has_value() ? beta.value().element_size() : 0;

    // Determine if gamma/beta are batched (have batch dimension > 1)
    // Gamma/beta can have shapes: [W], [1, W], [1, 1, W], or [batch, 1, W]
    // When batch > 1, we need to re-read gamma/beta for each batch
    // Note: Batch broadcasting is only supported for TILE layout (validated in device_operation.cpp)
    uint32_t gamma_is_batched = 0;
    uint32_t gamma_batch_stride_tiles = 0;
    if (gamma.has_value() && gamma.value().layout() == Layout::TILE) {
        const auto& gamma_shape = gamma.value().padded_shape();
        if (gamma_shape.rank() >= 3 && gamma_shape[-3] > 1) {
            gamma_is_batched = 1;
            // Stride in tiles between batches = Wt (one row of tiles per batch)
            gamma_batch_stride_tiles = Wt;
        }
    }
    uint32_t beta_is_batched = 0;
    uint32_t beta_batch_stride_tiles = 0;
    if (beta.has_value() && beta.value().layout() == Layout::TILE) {
        const auto& beta_shape = beta.value().padded_shape();
        if (beta_shape.rank() >= 3 && beta_shape[-3] > 1) {
            beta_is_batched = 1;
            beta_batch_stride_tiles = Wt;
        }
    }

    reader_compile_time_args.push_back((std::uint32_t)gamma_page_size);
    reader_compile_time_args.push_back((std::uint32_t)beta_page_size);
    reader_compile_time_args.push_back((std::uint32_t)gamma_is_row_major);
    reader_compile_time_args.push_back((std::uint32_t)beta_is_row_major);
    reader_compile_time_args.push_back((std::uint32_t)Wt);
    reader_compile_time_args.push_back((std::uint32_t)gamma_element_size);
    reader_compile_time_args.push_back((std::uint32_t)beta_element_size);
    reader_compile_time_args.push_back((std::uint32_t)gamma_is_batched);
    reader_compile_time_args.push_back((std::uint32_t)beta_is_batched);
    reader_compile_time_args.push_back((std::uint32_t)gamma_batch_stride_tiles);
    reader_compile_time_args.push_back((std::uint32_t)beta_batch_stride_tiles);
    reader_compile_time_args.push_back((std::uint32_t)Ht);

    TensorAccessorArgs(a_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(stats_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(gamma_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(beta_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)Wt, (std::uint32_t)block_size};
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/dataflow/"
        "reader_layernorm_postallgather_dit.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/dataflow/"
        "writer_layernorm_postallgather_dit.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_args = {
        Wt,
        W,
        block_size,
        num_devices,
        gamma.has_value(),
        beta.has_value(),
        gamma_is_batched,
        beta_is_batched,
        Ht,
        Wt_round_up_block_size};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/"
        "layernorm_post_allgather_welford.cpp";

    // Float32 input on the welford path requires fp32_dest_acc_en=true as a prerequisite for
    // UnpackToDestFp32 (set below). UnpackToDestFp32 is what bypasses the unpacker's
    // Float32 → TF32 truncation in SrcA; fp32_dest_acc_en provides the 32-bit DEST that
    // UnpackToDestFp32 writes into. Without fp32 DEST, UnpackToDestFp32 can't be enabled
    // and inputs are silently truncated to TF32 (10 mantissa bits) on the way through SrcA.
    TT_FATAL(
        !(in_data_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "dit_layernorm_post_all_gather with Float32 input requires fp32_dest_acc_en=true in the "
        "compute kernel config; otherwise precision is silently lost in the unpacker format "
        "conversion.");

    // UnpackToDestFp32 only helps for CBs whose only consumer is an op that supports the
    // unpack-to-DEST path (copy_tile or transpose_wh_tile in fp32 mode). For those, setting
    // the flag preserves the full 23-mantissa fp32 by bypassing SrcA.
    // c_1 (stats) is consumed only by copy_tile inside combine_welford_partials.
    // Set the flag so the per-row mean/M2 recombine reads full mantissa.
    //
    // Note that setting the flag on a CB consumed by any FPU op (mul_tiles, add_tiles,*_bcast_*, ...)
    // is unsafe: per base_types.hpp the CB is "incompatible with unpacking to SRCA/B", and
    // on Wormhole/Blackhole that combination produces garbage in SrcA (not silent TF32
    // truncation as one might assume).
    // c_0 (input) is consumed only by sub_tiles_bcast_cols (FPU), so we must not enable the flag on it.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en && stats_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_1)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.defines = {compute_defines.begin(), compute_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = math_approx_mode,
    };

    /**
     * c_0 -> a
     * c_1 -> stats
     * c_2 -> gamma
     * c_3 -> beta
     * c_4 -> epsilon
     * c_5 -> [mean(x**2), mean(x)] stats reduced
     * c_6 -> 1/sqrt(var + epsilon)
     * c_7 -> a intermediate
     * c_8 -> output
     */

    // A input
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * in_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in_data_format,
            .page_size = in_single_tile_size,
        }}},
    });
    // Stats input
    desc.cbs.push_back(CBDescriptor{
        .total_size = in1_tiles * stats_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = stats_data_format,
            .page_size = stats_single_tile_size,
        }}},
    });
    if (gamma.has_value()) {
        // Gamma input
        desc.cbs.push_back(CBDescriptor{
            .total_size = in2_tiles * gamma_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
                .data_format = gamma_cb_data_format,
                .page_size = gamma_single_tile_size,
            }}},
        });
    }
    if (beta.has_value()) {
        // Beta input
        desc.cbs.push_back(CBDescriptor{
            .total_size = in3_tiles * beta_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = beta_cb_data_format,
                .page_size = beta_single_tile_size,
            }}},
        });
    }
    // Epsilon input
    desc.cbs.push_back(CBDescriptor{
        .total_size = in4_tiles * bfloat16_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
            .data_format = tt::DataFormat::Float16_b,
            .page_size = bfloat16_tile_size,
        }}},
    });
    // stats reduced
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed0_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    // recip_sqrt_var
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed1_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    // a intermediate
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed2_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_7),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    // output
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_tiles * out_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
            .data_format = out_data_format,
            .page_size = out_single_tile_size,
        }}},
    });

    union {
        float f;
        uint32_t u;
    } e{};
    e.f = operation_attributes.eps;  // epsilon

    uint32_t curr_row = 0;
    for (const auto& core : cores) {
        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t tile_row_start = curr_row;
        uint32_t tile_row_end = tile_row_start + num_tile_rows_per_core;

        TT_FATAL(tile_row_start < tile_row_end, "Tile row start must be less than tile row end");
        TT_FATAL(tile_row_end <= num_tile_rows, "Tile row end must be less than or equal to number of tile rows");

        KernelDescriptor::RTArgList reader_args;
        reader_args.reserve(7);
        reader_args.push_back(a_buffer);
        reader_args.push_back(stats_buffer);
        if (gamma_buffer != nullptr) {
            reader_args.push_back(gamma_buffer);
        } else {
            reader_args.push_back(uint32_t{0});
        }
        if (beta_buffer != nullptr) {
            reader_args.push_back(beta_buffer);
        } else {
            reader_args.push_back(uint32_t{0});
        }
        reader_args.push_back(tile_row_start);
        reader_args.push_back(tile_row_end);
        reader_args.push_back(e.u);
        reader_desc.emplace_runtime_args(core, reader_args);
        compute_desc.emplace_runtime_args(core, {num_tile_rows_per_core, tile_row_start});
        writer_desc.emplace_runtime_args(core, {dst_buffer, tile_row_start, tile_row_end});
        curr_row += num_tile_rows_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
