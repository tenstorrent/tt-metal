// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_pre_all_gather_device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/math.hpp"

#include <bit>
#include <map>
#include <numeric>
#include <string>

using uint32_t = std::uint32_t;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor LayerNormPreAllGatherWelfordProgramFactory::create_descriptor(
    const LayerNormPreAllGatherParams& operation_attributes,
    const LayerNormPreAllGatherInputs& tensor_args,
    Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const bool fuse_pre_add = b.has_value();
    const bool is_rmsnorm = operation_attributes.norm_type == LayerNormDistributedType::RMSNORM;
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const auto& logical_shape = a.logical_shape();
    const auto& padded_shape = a.padded_shape();
    const uint32_t W = logical_shape[-1];
    const uint32_t padded_W = padded_shape[-1], padded_H = padded_shape[-2];
    const uint32_t padded_HW = padded_H * padded_W;
    const uint32_t NC = a.physical_volume() / padded_HW;

    const uint32_t Wt = padded_W / tile_width;
    const uint32_t Ht = padded_H / tile_height;

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    TT_FATAL(!is_rmsnorm, "rms_norm is not compatible with welford, please disable welford flag to use rms norm");

    uint32_t num_tile_rows = NC * Ht;

    log_debug(tt::LogOp, "is_rmsnorm: {}", is_rmsnorm);
    log_debug(tt::LogOp, "W: {}", W);
    log_debug(tt::LogOp, "padded_W: {}", padded_W);
    log_debug(tt::LogOp, "padded_H: {}", padded_H);
    log_debug(tt::LogOp, "num_tile_rows: {}", num_tile_rows);
    log_debug(tt::LogOp, "Wt: {}", Wt);
    log_debug(tt::LogOp, "Ht: {}", Ht);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // The welford kernel does the pre-add and welford passes in blk-sized chunks, spilling the
    // welford accumulator to a small CB between chunks. Per-chunk overhead (state spill +
    // tile_regs scope switch) amortizes over more tiles when blk is larger; the upper bound is
    // how many tiles fit in DST in a single tile_regs scope (4 in fp32_dest_acc, 8 otherwise).
    // Constrain blk to divide Wt so the reader and compute kernel stay aligned without a
    // partial last block.
    const uint32_t dst_capacity = fp32_dest_acc_en ? 4u : 8u;
    uint32_t block_size = std::gcd(Wt, dst_capacity);
    uint32_t writer_block_size = 1;

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat inb_data_format = tt::DataFormat::Invalid;
    uint32_t inb_single_tile_size = 0;
    if (fuse_pre_add) {
        inb_data_format = tt::tt_metal::datatype_to_dataformat_converter(b->dtype());
        inb_single_tile_size = tt::tile_size(inb_data_format);
    }
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    log_debug(tt::LogOp, "in_data_format: {}", in_data_format);
    log_debug(tt::LogOp, "out_data_format: {}", out_data_format);

    // Sized for double-buffered block-sized chunks: the welford compute kernel waits on
    // block_size tiles at a time, so the reader must be able to fill that many while the
    // compute side processes the previous batch.
    const uint32_t in0_tiles = block_size * 2;
    const uint32_t res_tiles = block_size * 2;
    // The pre-add and welford passes are interleaved in block_size-sized chunks in the welford
    // kernel: each chunk's tiles are added into cb_fused and then immediately consumed by
    // welford, with the welford accumulator spilled to cb_welford_mean / cb_welford_m2 between
    // chunks. So cb_fused only needs to hold block_size * 2 tiles (double-buffered for producer/
    // consumer overlap), not the full Wt row.
    const uint32_t fused_tiles = block_size * 2;
    const uint32_t welford_spill_tiles = 1;

    uint32_t out0_tiles = 1;
    if (!is_rmsnorm) {
        out0_tiles = 2;
    }

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);
    log_debug(tt::LogOp, "core_group_1: {}", core_group_1.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_1: {}", num_tile_rows_per_core_group_1);
    log_debug(tt::LogOp, "core_group_2: {}", core_group_2.str());
    log_debug(tt::LogOp, "num_tile_rows_per_core_group_2: {}", num_tile_rows_per_core_group_2);

    std::vector<uint32_t> reader_compile_time_args = {
        block_size,
    };
    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    if (fuse_pre_add) {
        tt::tt_metal::TensorAccessorArgs(b->buffer()).append_to(reader_compile_time_args);
    }

    std::vector<uint32_t> writer_compile_time_args = {writer_block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;
    reader_defines["FUSE_PRE_ADD"] = fuse_pre_add ? "1" : "0";
    compute_defines["FUSE_PRE_ADD"] = fuse_pre_add ? "1" : "0";

    // UnpackToDestFp32 routes the unpack to DEST instead of SrcA, preserving FP32 precision.
    // That path uses the math-thread replay buffer, which collides with Welford's recurrence
    // slots; welford_unpack_fp32_active gates welford_init<WelfordInitMode::PreserveStats>()
    // after each transpose_tile to re-record the SFPU replay buffer.
    //
    // On the FUSE path, pre-add uses copy_tile + add_binary_tile (SFPU), not add_tiles, so
    // c_0/c_5 can use UnpackToDestFp32 for the copy_tile unpack and c_3 for Welford's
    // transpose_tile read of the post-add result.
    bool welford_unpack_fp32_active = (in_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en);
    std::vector<uint32_t> compute_args = {Wt, W, block_size};
    KernelDescriptor::NamedCompileTimeArgs compute_named_args = {
        {"welford_unpack_fp32_active", welford_unpack_fp32_active ? 1u : 0u},
    };

    const auto* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/"
        "layernorm_pre_allgather_welford.cpp";

    TT_FATAL(
        tensor_args.recip_tensor.has_value(),
        "Welford algorithm requires recip_tensor. Use ttnn.create_layer_norm_reciprocals() to create it.");
    const auto& recip_tensor = tensor_args.recip_tensor.value();
    const uint32_t reciprocal_CB_size_bytes = recip_tensor.buffer()->aligned_size_per_bank();
    constexpr tt::DataFormat reciprocal_cb_data_format = tt::DataFormat::Float32;

    // Build runtime args per core.  Buffer base addresses are bound via
    // emplace_runtime_args() so the framework patches them on cache hits.
    KernelDescriptor reader_kernel_desc;
    KernelDescriptor writer_kernel_desc;
    KernelDescriptor compute_kernel_desc;
    reader_kernel_desc.runtime_args.reserve(num_cores);
    writer_kernel_desc.runtime_args.reserve(num_cores);
    compute_kernel_desc.runtime_args.reserve(num_cores);

    uint32_t curr_row = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t in_tile_offset = curr_row * Wt;
        uint32_t out_tile_offset = curr_row * out0_tiles;

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(a.buffer());
        reader_args.push_back(num_tile_rows_per_core);
        reader_args.push_back(Wt);
        reader_args.push_back(in_tile_offset);
        if (fuse_pre_add) {
            reader_args.push_back(b->buffer());
        }
        reader_kernel_desc.emplace_runtime_args(core, reader_args);
        compute_kernel_desc.emplace_runtime_args(core, {num_tile_rows_per_core});
        writer_kernel_desc.emplace_runtime_args(
            core, {output.buffer(), num_tile_rows_per_core * out0_tiles, out_tile_offset});

        curr_row += num_tile_rows_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Reader kernel
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.defines = KernelDescriptor::Defines(reader_defines.begin(), reader_defines.end());
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));

    // Writer kernel
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_blocked.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};
    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));

    // Float32 input on the welford path requires fp32_dest_acc_en=true as a prerequisite for
    // UnpackToDestFp32 (set below). UnpackToDestFp32 is what bypasses the unpacker's
    // Float32 → TF32 truncation in SrcA; fp32_dest_acc_en provides the 32-bit DEST that
    // UnpackToDestFp32 writes into. Without fp32 DEST, UnpackToDestFp32 can't be enabled
    // and inputs are silently truncated to TF32 (10 mantissa bits) on the way through SrcA.
    TT_FATAL(
        !(in_data_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "layer_norm_pre_all_gather with Float32 input requires fp32_dest_acc_en=true in the "
        "compute kernel config; otherwise precision is silently lost in the unpacker format "
        "conversion.");

    // When welford_unpack_fp32_active:
    //   !fuse_pre_add -> Set UnpackToDestFp32 on c_0 only (input read by transpose_tile in the Welford loop).
    //   fuse_pre_add  -> Set UnpackToDestFp32 on c_0, c_5, c_3 (copy_tile pre-add unpack + transpose_tile on post-add
    //   cb_inp).
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (welford_unpack_fp32_active) {
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_0)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        if (fuse_pre_add) {
            unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_5)] =
                tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_3)] =
                tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }

    // Intermediate scratch CB (c_1) holds data only for the final transpose operation,
    // so its format mirrors out_data_format. When both that format is FP32 and DEST
    // is in FP32 mode, force UnpackToDestFp32 on c_1 too so the read-back doesn't
    // truncate to TF32. For non-FP32 outputs the final pack to c_14 truncates
    // anyway, so unpacking to FP32 would not be useful.
    if (out_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en) {
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_1)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Welford spill CBs (c_4 = running mean, c_6 = running M2) hold the FP32 accumulator
    // between block iterations and are reloaded into DEST via copy_tile. In Default
    // unpack-to-dest mode that round-trip routes through SrcA, which truncates FP32 to
    // TF32 on every block iteration. Force UnpackToDestFp32 on these CBs so the FP32 precision
    // survives the spill cycle.
    if (fuse_pre_add && fp32_dest_acc_en) {
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_4)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[static_cast<uint32_t>(tt::CBIndex::c_6)] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Compute kernel
    // Welford uses fp32 accumulation; preserve fp32_dest_acc_en from compute config
    compute_kernel_desc.kernel_source = compute_kernel_file;
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_args);
    compute_kernel_desc.named_compile_time_args = std::move(compute_named_args);
    compute_kernel_desc.defines = KernelDescriptor::Defines(compute_defines.begin(), compute_defines.end());
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = math_approx_mode};
    program_descriptor.kernels.push_back(std::move(compute_kernel_desc));

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    // c_in0 -> a
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * in_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in_data_format,
            .page_size = in_single_tile_size}}}});

    if (fuse_pre_add) {
        // c_5 -> residual b. Sized in residual's own data format so a residual with a different
        // dtype than the input is read correctly.
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = res_tiles * inb_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
                .data_format = inb_data_format,
                .page_size = inb_single_tile_size}}}});
        // c_3 -> fused a + b (compute kernel writes here, welford consumes from it)
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = fused_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = cb_data_format,
                .page_size = single_tile_size}}}});
        // c_4 -> welford mean accumulator spill (one tile, ping-pongs each iteration)
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = welford_spill_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
                .data_format = cb_data_format,
                .page_size = single_tile_size}}}});
        // c_6 -> welford M2 accumulator spill (one tile, ping-pongs each iteration)
        program_descriptor.cbs.push_back(CBDescriptor{
            .total_size = welford_spill_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_6),
                .data_format = cb_data_format,
                .page_size = single_tile_size}}}});
    }

    // Intermediate scratch for the post-Welford transpose round-trip (CB 1).
    // Used only for the last transpose operation before copying data into the
    // output CB, which is why its data format is tied to the output format.
    // Anything wider would waste SRAM and gain no precision (the read-back
    // unpack truncates to TF32 unless the output is Float32, in which case
    // UnpackToDestFp32 above preserves it).
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * out_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = out_data_format,
            .page_size = out_single_tile_size}}}});

    // Output (CB 14)
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = in0_tiles * out_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_14),
            .data_format = out_data_format,
            .page_size = out_single_tile_size}}}});

    // Reciprocal LUT (CB 2) - sharded, with backing buffer
    program_descriptor.cbs.push_back(CBDescriptor{
        .total_size = reciprocal_CB_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
            .data_format = reciprocal_cb_data_format,
            .page_size = reciprocal_CB_size_bytes}}},
        .buffer = recip_tensor.buffer()});

    return program_descriptor;
}

}  // namespace ttnn::prim
