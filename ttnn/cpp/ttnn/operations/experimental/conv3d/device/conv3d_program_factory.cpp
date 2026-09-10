// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_program_factory.hpp"
#include "conv3d_device_operation_types.hpp"
#include "kernels/conv3d_gather_tuning.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include <algorithm>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <hostdevcommon/common_values.hpp>

namespace ttnn::experimental::prim {

namespace mcast = ttnn::kernel_lib::host;

// Largest divisor of n that is <= cap. Always returns at least 1.
static uint32_t largest_divisor_up_to(uint32_t n, uint32_t cap) {
    for (uint32_t d = std::min(n, cap); d >= 1; d--) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

tt::tt_metal::ProgramDescriptor Conv3dProgramFactory::create_descriptor(
    const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args, Tensor& tensor_return_value) {
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ProgramDescriptor;
    using tt::tt_metal::ReaderConfigDescriptor;
    using tt::tt_metal::SemaphoreDescriptor;
    using tt::tt_metal::WriterConfigDescriptor;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& output_tensor = tensor_return_value;

    // Extract config from operation_attributes
    const auto& config = operation_attributes.config;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    ProgramDescriptor desc;

    auto grid_size = config.compute_with_storage_grid_size;
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();
    auto input_tensor_shape = input_tensor.logical_shape();
    uint32_t N = input_tensor_shape[0];
    uint32_t T_in = input_tensor_shape[1];
    uint32_t H_in = input_tensor_shape[2];
    uint32_t W_in = input_tensor_shape[3];
    uint32_t C_in = input_tensor_shape[4];
    auto [T_out, H_out, W_out] = detail::compute_output_dims(
        T_in,
        H_in,
        W_in,
        operation_attributes.padding,
        operation_attributes.stride,
        operation_attributes.kernel_size,
        operation_attributes.dilation);
    uint32_t C_out = operation_attributes.output_channels;
    uint32_t padded_C_out = tt::round_up(C_out, tt::constants::TILE_WIDTH);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto dtype_bytes = input_tensor.element_size();
    auto tile_size = tt::tile_size(data_format);

    bool use_bias = bias_tensor.has_value();

    // Extract compute kernel config early (needed for CB format decisions)
    [[maybe_unused]] auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), compute_kernel_config);

    /* Shapes/sizes needed in the kernel
        Reader does volume2column to convert some `T_block x H_block x W_block` of activation
        to `T_block x H_block x W_block, kD x kH x kW x C_in` patches.
        Compute takes this `num_patches x patch_size` CB and tilizes it.

        Writer reads the weights of size `kD x kH x kW x C_in, C_out`, tilized.
        Writer reads the bias of size `1, C_out`, tilized.
        Compute runs matmul on `patches @ kernel` and adds bias.
        Compute untilizes the result.
        Writer writes the result to the output tensor.


    Padding/tilizing constraints:
        - ceil(num_patches / TILE_HEIGHT) is number of tile rows of matmul
        - `kD x kH x kW x C_in` of the kernel weight is padded to tile size (since it's tilized)
            and must be padded with zeros so the MM result is correct.
    */

    // If C_out_block is set, use it. Otherwise, use the full number of output channels.
    uint32_t C_out_block = config.C_out_block > 0 ? config.C_out_block : padded_C_out;
    uint32_t C_in_block = config.C_in_block > 0 ? config.C_in_block : C_in;

    uint32_t patch_size = operation_attributes.kernel_size[0] * operation_attributes.kernel_size[1] *
                          operation_attributes.kernel_size[2] * C_in_block;
    uint32_t padded_patch_size = tt::round_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t num_patches = config.T_out_block * config.H_out_block * config.W_out_block;

    uint32_t C_in_num_blocks = tt::div_up(C_in, C_in_block);
    TT_FATAL(C_in_num_blocks * C_in_block == C_in, "C_in_num_blocks * C_in_block must equal C_in");
    uint32_t C_out_num_blocks = tt::div_up(padded_C_out, C_out_block);
    TT_FATAL(
        C_out_num_blocks * C_out_block == padded_C_out,
        "C_out_num_blocks * C_out_block must equal padded_C_out ({}). Got C_out_num_blocks={}, C_out_block={}.",
        padded_C_out,
        C_out_num_blocks,
        C_out_block);

    uint32_t matmul_M_t = tt::div_up(num_patches, tt::constants::TILE_HEIGHT);
    uint32_t matmul_K_t = tt::div_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t matmul_N_t = tt::div_up(C_out_block, tt::constants::TILE_WIDTH);

    // Matmul subblock sizing. out_subblock_w fills the dst register row; out_subblock_h
    // batches multiple tile-rows per matmul call for weight reuse.
    // On Wormhole B0 the matmul unit benefits from sub_h > 1 (preferred 2x4 subblock).
    // On Blackhole the row-by-row fused tilize+matmul is faster with the current
    // row-major subblock layout, so keep sub_h = 1 with optimized blockings.
    const uint32_t dst_size = ttnn::get_dest_reg_count(compute_kernel_config);
    const uint32_t out_subblock_w = std::min(matmul_N_t, dst_size);
    const auto arch = tt::tt_metal::hal::get_arch();
    const bool scale_subblock_h = arch == tt::ARCH::WORMHOLE_B0 && out_subblock_w == matmul_N_t;
    const uint32_t out_subblock_h = scale_subblock_h ? largest_divisor_up_to(matmul_M_t, dst_size / out_subblock_w) : 1;
    const uint32_t output_write_bytes_per_transaction = C_out_block * dtype_bytes;
    const bool small_output_write_transactions =
        output_write_bytes_per_transaction <= tt::constants::TILE_WIDTH * dtype_bytes;
    const bool enable_streaming_output = C_in_num_blocks == 1 && matmul_M_t > 1 && small_output_write_transactions;

    uint32_t num_patches_tile_padded = tt::round_up(num_patches, tt::constants::TILE_HEIGHT);

    uint32_t patch_size_bytes = patch_size * dtype_bytes;                // bytes of actual data per patch row
    uint32_t padded_patch_size_bytes = padded_patch_size * dtype_bytes;  // bytes per CB page (tile-aligned)
    uint32_t patch_pad_bytes = padded_patch_size_bytes - patch_size_bytes;
    uint32_t C_out_block_bytes = C_out_block * dtype_bytes;  // bytes per output channel row
    uint32_t C_in_block_bytes = C_in_block * dtype_bytes;    // bytes per input channel row

    log_debug(tt::LogOp, "Block sizes:");
    log_debug(tt::LogOp, "  T_out_block: {}", config.T_out_block);
    log_debug(tt::LogOp, "  H_out_block: {}", config.H_out_block);
    log_debug(tt::LogOp, "  W_out_block: {}", config.W_out_block);
    log_debug(tt::LogOp, "  C_out_block: {}", C_out_block);
    log_debug(tt::LogOp, "  C_out_num_blocks: {}", C_out_num_blocks);
    log_debug(tt::LogOp, "Patch size: {}", patch_size);
    log_debug(tt::LogOp, "Num patches: {}", num_patches);
    log_debug(tt::LogOp, "Patch size bytes: {}", patch_size_bytes);
    log_debug(tt::LogOp, "C_out block bytes: {}", C_out_block_bytes);
    log_debug(tt::LogOp, "Num patches tile padded: {}", num_patches_tile_padded);
    log_debug(tt::LogOp, "Matmul M_t: {}", matmul_M_t);
    log_debug(tt::LogOp, "Matmul K_t: {}", matmul_K_t);
    log_debug(tt::LogOp, "Matmul N_t: {}", matmul_N_t);
    // Create circular buffers for vol2col, weights, bias and matmul intermediates
    uint32_t next_cb_index = tt::CBIndex::c_0;

    // Fused tilize+matmul: compute tilizes row-by-row but batches out_subblock_h
    // tile-rows before each matmul call, so vol2col_tiled needs out_subblock_h*K_t
    // tiles instead of the full M_t*K_t.
    // vol2col_rm only needs TILE_HEIGHT pages since tilize consumes each row before
    // the next is pushed.
    // Double-buffer (2x) when num_patches isn't tile-aligned to avoid CB deadlock
    // between reader pushes and compute tilize pops on the partial last row.
    uint32_t vol2col_rm_pages = (num_patches % tt::constants::TILE_HEIGHT == 0)
                                    ? std::min(num_patches, (uint32_t)tt::constants::TILE_HEIGHT)
                                    : std::min(num_patches, 2 * tt::constants::TILE_HEIGHT);
    uint32_t cb_vol2col_rm_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = vol2col_rm_pages * padded_patch_size_bytes,
        .core_ranges = CoreRangeSet(core_grid),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_vol2col_rm_id),
            .data_format = data_format,
            .page_size = padded_patch_size_bytes,
        }}},
    });

    uint32_t cb_vol2col_tiled_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_subblock_h * matmul_K_t * tile_size,
        .core_ranges = CoreRangeSet(core_grid),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_vol2col_tiled_id),
            .data_format = data_format,
            .page_size = tile_size,
        }}},
    });

    uint32_t cb_weight_tiled_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = matmul_K_t * matmul_N_t * tile_size,
        .core_ranges = CoreRangeSet(core_grid),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_weight_tiled_id),
            .data_format = data_format,
            .page_size = tile_size,
        }}},
    });

    // Use fp32 partials whenever we have multiple C_in blocks and fp32 dest is enabled.
    // This eliminates bf16 truncation between C_in block partial sums.
    bool use_fp32_partials = fp32_dest_acc_en && C_in_num_blocks > 1;
    auto partial_data_format = use_fp32_partials ? tt::DataFormat::Float32 : data_format;
    auto partial_tile_size = tt::tile_size(partial_data_format);

    uint32_t cb_matmul_interm_tiled_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = matmul_M_t * matmul_N_t * partial_tile_size,
        .core_ranges = CoreRangeSet(core_grid),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_matmul_interm_tiled_id),
            .data_format = partial_data_format,
            .page_size = partial_tile_size,
        }}},
    });

    // NOTE: Most kernels create RM CB with tile_size pages and num_tile number of pages.
    // Using stick pages led to PCC issues.
    uint32_t cb_matmul_result_rm_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        // untilize will write padded rows, so this must be sized to avoid overflowing CB
        .total_size = matmul_M_t * matmul_N_t * tile_size,
        .core_ranges = CoreRangeSet(core_grid),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_matmul_result_rm_id),
            .data_format = data_format,
            .page_size = tile_size,
        }}},
    });

    uint32_t cb_reduction_tiled_id =
        32;  // Invalid value for cb index since there is only 32 of them and the indices go from 0 to 31
    uint32_t cb_worker_ack_back_id =
        32;  // Invalid value for cb index since there is only 32 of them and the indices go from 0 to 31
    if (C_in_num_blocks > 1) {
        // Multi-core reduction step: each core computes a partial sum, then they reduce
        // Use same format as partials CB so reduction adds matching formats
        cb_reduction_tiled_id = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = matmul_M_t * matmul_N_t * partial_tile_size,
            .core_ranges = CoreRangeSet(core_grid),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_reduction_tiled_id),
                .data_format = partial_data_format,
                .page_size = partial_tile_size,
            }}},
        });

        cb_worker_ack_back_id = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = tile_size,
            .core_ranges = CoreRangeSet(core_grid),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_worker_ack_back_id),
                .data_format = data_format,
                .page_size = tile_size,
            }}},
        });
    }

    uint32_t cb_bias_tiled_id =
        32;  // Invalid value for cb index since there is only 32 of them and the indices go from 0 to 31
    if (use_bias) {
        cb_bias_tiled_id = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = matmul_N_t * tile_size,
            .core_ranges = CoreRangeSet(core_grid),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_bias_tiled_id),
                .data_format = data_format,
                .page_size = tile_size,
            }}},
        });
    }

    log_debug(
        tt::LogOp,
        "CB vol2col_rm: page_size={} bytes (padded from {}), num_pages={}",
        padded_patch_size_bytes,
        patch_size_bytes,
        vol2col_rm_pages);
    log_debug(tt::LogOp, "CB vol2col_tiled: page_size={} bytes, num_pages={}", tile_size, out_subblock_h * matmul_K_t);
    log_debug(tt::LogOp, "CB weight_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_K_t * matmul_N_t);
    log_debug(
        tt::LogOp, "CB matmul_interm_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_N_t);
    log_debug(tt::LogOp, "CB matmul_result_rm: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_N_t);

    bool is_padding_zeros = operation_attributes.padding_mode == "zeros";
    // validate() fatals on a halo buffer with any other padding mode, so presence is the whole test.
    const bool halo_mode = tensor_args.halo_buffer.has_value();
    // Logical-pad masking: opt-in.
    const bool mask_mode = tensor_args.pad_offset_tensor.has_value() &&
                           (operation_attributes.logical_h_mask != 0 || operation_attributes.logical_w_mask != 0);

    uint32_t in_row_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_row_size_bytes = output_tensor.buffer()->aligned_page_size();

    const uint32_t device_num_dram_banks = static_cast<uint32_t>(input_tensor.device()->num_dram_channels());
    TT_FATAL(device_num_dram_banks > 0, "Device must report at least one DRAM channel");
    const bool input_is_dram_interleaved =
        input_tensor.buffer()->is_dram() &&
        input_tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
        !input_tensor.buffer()->buffer_distribution_spec().has_value();
    const uint32_t dram_read_alignment = tt::tt_metal::hal::get_dram_alignment();
    const bool input_pages_are_dram_read_aligned = in_row_size_bytes % dram_read_alignment == 0;
    const bool c_in_slice_is_dram_read_aligned = C_in_block_bytes % dram_read_alignment == 0;
    const bool enable_dram_read_staging =
        input_is_dram_interleaved && input_pages_are_dram_read_aligned && !c_in_slice_is_dram_read_aligned;
    // The staged reader rounds the DRAM source down by at most alignment - 1 bytes and reads
    // a full aligned window.  The scratch CB itself may only be L1-aligned, so reserve one
    // extra alignment chunk to let the kernel round its scratch base up safely.
    const uint32_t max_staged_dram_window_bytes =
        tt::round_up(C_in_block_bytes + dram_read_alignment - 1, dram_read_alignment);
    const uint32_t dram_read_scratch_page_bytes =
        enable_dram_read_staging ? max_staged_dram_window_bytes + dram_read_alignment : 0;
    uint32_t cb_dram_read_scratch_id = 32;  // Invalid; set below if DRAM read staging is needed
    if (enable_dram_read_staging) {
        cb_dram_read_scratch_id = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = dram_read_scratch_page_bytes,
            .core_ranges = CoreRangeSet(core_grid),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_dram_read_scratch_id),
                .data_format = data_format,
                .page_size = dram_read_scratch_page_bytes,
            }}},
        });
    }

    // Tiny landing CB for the per-device [h_start, w_start] offset page (mask mode only).
    uint32_t cb_pad_offset_id = 32;  // Invalid; set below if masking is enabled
    if (mask_mode) {
        const uint32_t pad_offset_page_bytes =
            tt::round_up(2u * static_cast<uint32_t>(sizeof(uint32_t)), dram_read_alignment);
        cb_pad_offset_id = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = pad_offset_page_bytes,
            .core_ranges = CoreRangeSet(core_grid),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_pad_offset_id),
                .data_format = tt::DataFormat::UInt32,
                .page_size = pad_offset_page_bytes,
            }}},
        });
    }

    // L1 pre-fetch buffer for kernels > 1x1x1 with no dilation.
    // Gathers the spatial receptive field from DRAM once per spatial block, then vol2col reads from L1.
    // Budget: remaining L1 after other CBs and kernel code/stack, capped at 500 KB.
    // hal::get_max_worker_l1_unreserved_size() gives total L1 for CBs + kernel code;
    // subtract a conservative 200 KB reserve for kernel code/stack.
    constexpr uint32_t L1_KERNEL_CODE_RESERVE = 200 * 1024;
    constexpr uint32_t L1_PREFETCH_HARD_CAP = 500 * 1024;
    const uint32_t l1_usable_for_cbs = tt::tt_metal::hal::get_max_worker_l1_unreserved_size() - L1_KERNEL_CODE_RESERVE;

    uint32_t other_cbs_bytes = (padded_patch_size_bytes * vol2col_rm_pages) +   // vol2col_rm
                               (tile_size * out_subblock_h * matmul_K_t) +      // vol2col_tiled
                               (tile_size * matmul_K_t * matmul_N_t) +          // weight_tiled
                               (partial_tile_size * matmul_M_t * matmul_N_t) +  // matmul_interm (may be fp32)
                               (tile_size * matmul_M_t * matmul_N_t);           // matmul_result_rm
    if (enable_dram_read_staging) {
        other_cbs_bytes += dram_read_scratch_page_bytes;
    }
    if (C_in_num_blocks > 1) {
        other_cbs_bytes += partial_tile_size * matmul_M_t * matmul_N_t;  // reduction (same format as partials)
        other_cbs_bytes += tile_size;                                    // worker_ack
    }
    if (use_bias) {
        other_cbs_bytes += tile_size * matmul_N_t;  // bias
    }
    uint32_t l1_prefetch_max_bytes =
        (other_cbs_bytes < l1_usable_for_cbs) ? std::min(l1_usable_for_cbs - other_cbs_bytes, L1_PREFETCH_HARD_CAP) : 0;

    const uint32_t kT = operation_attributes.kernel_size[0];
    const uint32_t kH = operation_attributes.kernel_size[1];
    const uint32_t kW = operation_attributes.kernel_size[2];
    const uint32_t W_shard_full_for_coalesce =
        (config.W_out_block - 1) * operation_attributes.stride[2] + operation_attributes.kernel_size[2];
    const uint32_t coalesced_read_row_bytes = C_in_block_bytes;
    // Coalescing pays for a scratch L1 reorder pass; require enough columns to give each
    // DRAM bank multiple pages so the larger bank-local bursts amortize the extra L1 traffic.
    const uint32_t coalesced_min_w_shard = 2 * device_num_dram_banks;
    const bool coalesced_shard_reads_candidate = input_is_dram_interleaved && C_in_num_blocks == 1 &&
                                                 C_in_block_bytes == in_row_size_bytes &&
                                                 W_shard_full_for_coalesce >= coalesced_min_w_shard;

    uint32_t cb_input_shard_id = 32;  // Invalid; set below if using L1 prefetch
    uint32_t T_shard_max = 0;
    uint32_t H_shard_max = 0;
    uint32_t W_shard_max = 0;
    bool enable_coalesced_shard_reads = false;
    uint32_t coalesced_scratch_rows = 0;

    const bool has_spatial_reuse = (kT > 1 || kH > 1 || kW > 1);
    const bool has_no_dilation =
        (operation_attributes.dilation[0] == 1 && operation_attributes.dilation[1] == 1 &&
         operation_attributes.dilation[2] == 1);

    if (has_spatial_reuse && has_no_dilation) {
        // Shard covers the full receptive field span for one spatial block, including padding positions.
        // Do NOT cap at T_in/H_in/W_in — padding positions outside input bounds are stored in the shard
        // (zero-filled or clamped) so that Phase 2 can index without boundary checks.
        T_shard_max = (config.T_out_block - 1) * operation_attributes.stride[0] + kT;
        H_shard_max = (config.H_out_block - 1) * operation_attributes.stride[1] + kH;
        W_shard_max = (config.W_out_block - 1) * operation_attributes.stride[2] + kW;
        uint32_t shard_positions_max = T_shard_max * H_shard_max * W_shard_max;
        uint32_t shard_bytes = shard_positions_max * C_in_block_bytes;
        uint32_t shard_rows_max = T_shard_max * H_shard_max;
        uint32_t coalesced_scratch_pages_per_row = W_shard_max;
        uint32_t coalesced_scratch_row_bytes = coalesced_scratch_pages_per_row * C_in_block_bytes;
        uint32_t coalesced_scratch_rows_fit = (coalesced_scratch_row_bytes > 0 && shard_bytes < l1_prefetch_max_bytes)
                                                  ? (l1_prefetch_max_bytes - shard_bytes) / coalesced_scratch_row_bytes
                                                  : 0;
        uint32_t coalesced_scratch_rows_candidate =
            coalesced_shard_reads_candidate ? std::min(shard_rows_max, coalesced_scratch_rows_fit) : 0;
        // Keep at least one row per DRAM bank in scratch when possible; smaller batches underfill the
        // coalesced gather and tend to lose to the direct reader after the L1 reorder cost.
        uint32_t coalesced_scratch_rows_min =
            coalesced_shard_reads_candidate ? std::min(shard_rows_max, device_num_dram_banks) : 0;
        uint32_t coalesced_scratch_positions = coalesced_scratch_rows_candidate * coalesced_scratch_pages_per_row;
        uint32_t shard_positions_with_coalesced_scratch = shard_positions_max + coalesced_scratch_positions;
        uint32_t shard_bytes_with_coalesced_scratch = shard_positions_with_coalesced_scratch * C_in_block_bytes;

        if (shard_bytes <= l1_prefetch_max_bytes) {
            enable_coalesced_shard_reads = coalesced_shard_reads_candidate &&
                                           coalesced_scratch_rows_candidate >= coalesced_scratch_rows_min &&
                                           shard_bytes_with_coalesced_scratch <= l1_prefetch_max_bytes;
            coalesced_scratch_rows = enable_coalesced_shard_reads ? coalesced_scratch_rows_candidate : 0;
            const uint32_t shard_positions_alloc =
                shard_positions_max + coalesced_scratch_rows * coalesced_scratch_pages_per_row;
            const uint32_t shard_bytes_alloc = shard_positions_alloc * C_in_block_bytes;
            cb_input_shard_id = next_cb_index++;
            desc.cbs.push_back(CBDescriptor{
                .total_size = shard_positions_alloc * C_in_block_bytes,
                .core_ranges = CoreRangeSet(core_grid),
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(cb_input_shard_id),
                    .data_format = data_format,
                    .page_size = C_in_block_bytes,
                }}},
            });

            log_debug(
                tt::LogOp,
                "L1 prefetch: T_shard_max={}, H_shard_max={}, W_shard_max={}, shard_positions={}, "
                "scratch_positions={}, shard_bytes={}, cb_id={}",
                T_shard_max,
                H_shard_max,
                W_shard_max,
                shard_positions_max,
                coalesced_scratch_rows * coalesced_scratch_pages_per_row,
                shard_bytes_alloc,
                cb_input_shard_id);
        } else {
            log_debug(
                tt::LogOp,
                "L1 prefetch shard ({} bytes) exceeds limit ({} bytes), falling back to direct reader",
                shard_bytes,
                l1_prefetch_max_bytes);
            T_shard_max = 0;
            H_shard_max = 0;
            W_shard_max = 0;
        }
    }

    const uint32_t coalesced_max_chunk_bytes =
        enable_coalesced_shard_reads
            ? tt::div_up(W_shard_full_for_coalesce, device_num_dram_banks) * coalesced_read_row_bytes
            : 0;

    log_debug(tt::LogOp, "Input tensor shape: N={}, T={}, H={}, W={}, C={}", N, T_in, H_in, W_in, C_in);
    log_debug(tt::LogOp, "Output tensor shape: T={}, H={}, W={}, C={}", T_out, H_out, W_out, C_out);
    log_debug(
        tt::LogOp,
        "Kernel size: {}x{}x{}",
        operation_attributes.kernel_size[0],
        operation_attributes.kernel_size[1],
        operation_attributes.kernel_size[2]);
    log_debug(
        tt::LogOp,
        "Stride: {}x{}x{}",
        operation_attributes.stride[0],
        operation_attributes.stride[1],
        operation_attributes.stride[2]);
    log_debug(
        tt::LogOp,
        "Dilation: {}x{}x{}",
        operation_attributes.dilation[0],
        operation_attributes.dilation[1],
        operation_attributes.dilation[2]);
    log_debug(
        tt::LogOp,
        "Padding: {}x{}x{}",
        operation_attributes.padding[0],
        operation_attributes.padding[1],
        operation_attributes.padding[2]);
    log_debug(tt::LogOp, "Groups: {}", operation_attributes.groups);
    log_debug(tt::LogOp, "Patch size: {}", patch_size);
    log_debug(tt::LogOp, "Input row size (bytes): {}", in_row_size_bytes);
    log_debug(tt::LogOp, "Output row size (bytes): {}", out_row_size_bytes);
    log_debug(tt::LogOp, "Data format: {}", data_format);
    log_debug(
        tt::LogOp,
        "Coalesced shard reads: enable={}, dram_banks={}, W_shard_full={}, scratch_rows={}, read_row_bytes={}, "
        "max_chunk_bytes={}, streaming_output={}, out_subblock={}x{}",
        enable_coalesced_shard_reads,
        device_num_dram_banks,
        W_shard_full_for_coalesce,
        coalesced_scratch_rows,
        coalesced_read_row_bytes,
        coalesced_max_chunk_bytes,
        enable_streaming_output,
        out_subblock_h,
        out_subblock_w);
    log_debug(
        tt::LogOp,
        "DRAM read staging: enable={}, alignment={}, scratch_page_bytes={}, cb_id={}",
        enable_dram_read_staging,
        dram_read_alignment,
        dram_read_scratch_page_bytes,
        cb_dram_read_scratch_id);

    /**
     * Compute parallelism for multi-core.
     * We now parallelize across C_in as the outermost dimension, followed by
     * C_out, T_out, H_out, and W_out dimensions. Cores working on the same output block
     * but different C_in ranges will need to synchronize for reduction.
     */

    // Calculate number of blocks along each dimension
    uint32_t T_out_blocks = tt::div_up(T_out, config.T_out_block);
    uint32_t H_out_blocks = tt::div_up(H_out, config.H_out_block);
    uint32_t W_out_blocks = tt::div_up(W_out, config.W_out_block);

    // Define parallelization factors for each dimension
    // C_in is the outermost parallelization dimension
    uint32_t c_in_parallel_factor = std::min(C_in_num_blocks, (uint32_t)num_cores);

    // Remaining cores per output block
    uint32_t cores_per_output = std::max(1u, (uint32_t)(num_cores / c_in_parallel_factor));

    // Distribute output parallelism across dimensions
    uint32_t c_out_parallel_factor = std::min(C_out_num_blocks, cores_per_output);
    uint32_t remaining_parallel = cores_per_output / c_out_parallel_factor;

    uint32_t t_out_parallel_factor = std::min(T_out_blocks, remaining_parallel);
    remaining_parallel = remaining_parallel / t_out_parallel_factor;

    uint32_t h_out_parallel_factor = std::min(H_out_blocks, remaining_parallel);
    remaining_parallel = remaining_parallel / h_out_parallel_factor;

    uint32_t w_out_parallel_factor = std::min(W_out_blocks, remaining_parallel);

    // Calculate total output blocks that will be processed in parallel
    uint32_t total_output_parallel =
        c_out_parallel_factor * t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor;

    // Verify parallelization is valid
    TT_FATAL(
        c_in_parallel_factor * total_output_parallel <= num_cores,
        "Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        c_in_parallel_factor * total_output_parallel,
        num_cores);

    log_debug(tt::LogOp, "Parallelization scheme:");
    log_debug(tt::LogOp, "C_in_num_blocks: {}, C_in_parallel_factor: {}", C_in_num_blocks, c_in_parallel_factor);
    log_debug(tt::LogOp, "C_out_parallel_factor: {}", c_out_parallel_factor);
    log_debug(tt::LogOp, "T_out_parallel_factor: {}", t_out_parallel_factor);
    log_debug(tt::LogOp, "H_out_parallel_factor: {}", h_out_parallel_factor);
    log_debug(tt::LogOp, "W_out_parallel_factor: {}", w_out_parallel_factor);
    log_debug(tt::LogOp, "Total output parallel blocks: {}", total_output_parallel);

    // Calculate blocks per core using ceiling division
    const uint32_t c_in_per_core = tt::div_up(C_in_num_blocks, c_in_parallel_factor);

    // When c_in_per_core > 1, a single core processes multiple C_in blocks sequentially.
    // The writer overwrites (not accumulates) each block's output at the same DRAM address,
    // and the bias is re-added on each iteration. Until the kernel supports accumulation
    // across C_in blocks on a single core, restrict to 1 block per core.
    TT_FATAL(
        c_in_per_core == 1,
        "Each core must handle exactly 1 C_in block, but got c_in_per_core={}. "
        "C_in_num_blocks={}, c_in_parallel_factor={}, num_cores={}",
        c_in_per_core,
        C_in_num_blocks,
        c_in_parallel_factor,
        num_cores);

    const uint32_t c_out_per_core = tt::div_up(C_out_num_blocks, c_out_parallel_factor);
    const uint32_t t_out_per_core = tt::div_up(T_out_blocks, t_out_parallel_factor);
    const uint32_t h_out_per_core = tt::div_up(H_out_blocks, h_out_parallel_factor);
    const uint32_t w_out_per_core = tt::div_up(W_out_blocks, w_out_parallel_factor);

    // Weight groups have the same (C_in, C_out) slice. Prefer one full-width row strip per
    // group when all strips fit; otherwise pack active assignments consecutively in row-major
    // order. The family chooses the transport from the resulting logical receiver sets.
    // Ceiling-sized chunks can leave empty spatial partitions, so placement uses active counts.
    auto active_slots = [](uint32_t blocks, uint32_t factor) { return tt::div_up(blocks, tt::div_up(blocks, factor)); };
    const uint32_t group_size = active_slots(T_out_blocks, t_out_parallel_factor) *
                                active_slots(H_out_blocks, h_out_parallel_factor) *
                                active_slots(W_out_blocks, w_out_parallel_factor);
    const uint32_t num_groups = c_in_parallel_factor * c_out_parallel_factor;
    const bool share_weights = group_size > 1;
    const uint32_t rows_per_group = tt::div_up(group_size, (uint32_t)grid_size.x);
    const bool place_in_rectangles = share_weights && (uint64_t)num_groups * rows_per_group <= grid_size.y;
    log_debug(
        tt::LogOp,
        "Weight placement: row_strips={}, active group_size={}, num_groups={}",
        place_in_rectangles,
        group_size,
        num_groups);

    // Set up semaphore for synchronization. It is dual-purpose.
    // On the reducer core, it tracks the number of workers that are done with an output block.
    // On the worker core, it is a valid bit indicating the worker can continue.
    uint32_t semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = semaphore_id,
        .core_ranges = CoreRangeSet(core_grid),
        .initial_value = 0,
    });

    // Trid-ring depth for gather_rows_to_shard.  Per-shape autotune (see
    // conv3d_trid_pipeline_findings.md).  Cutoff constants live in
    // kernels/conv3d_gather_tuning.hpp so the kernel-side per-call fast-path stays
    // pinned to the same numbers.  Two data-movement metrics gate the ring:
    //
    //   1. Reader-vs-compute balance — bytes per matmul tile op:
    //      intensity = T_shard * H_shard * W_shard * C_in_block_bytes / (M_t * K_t * N_t)
    //      Below kGatherIntensityCutoffBytes the kernel is compute-bound; ring overhead
    //      exceeds reader gain.
    //
    //   2. Representative gather burst size — reads per inner gather:
    //      inner_burst = T_shard * W_shard
    //      Below kGatherInnerBurstCutoff the host does not compile ring support. The
    //      reader still applies the stricter 2 * selected_trid_depth per-call guard
    //      before using the ring, so small edge gathers fall back to a single barrier.
    //
    // When either threshold fails, gather_trids = 0 and all ring code in the kernel is
    // constexpr-elided.
    const uint32_t k_T = operation_attributes.kernel_size[0];
    const uint32_t k_H = operation_attributes.kernel_size[1];
    const uint32_t k_W = operation_attributes.kernel_size[2];
    const uint32_t T_shard = (config.T_out_block - 1) * operation_attributes.stride[0] + k_T;
    const uint32_t H_shard = (config.H_out_block - 1) * operation_attributes.stride[1] + k_H;
    const uint32_t W_shard = (config.W_out_block - 1) * operation_attributes.stride[2] + k_W;
    const uint64_t reader_bytes_per_block = static_cast<uint64_t>(T_shard) * H_shard * W_shard * C_in_block_bytes;
    const uint64_t matmul_tiles = static_cast<uint64_t>(matmul_M_t) * matmul_K_t * matmul_N_t;
    const uint64_t bytes_per_tile = matmul_tiles == 0 ? 0 : (reader_bytes_per_block / matmul_tiles);
    const uint32_t inner_gather_burst = T_shard * W_shard;
    // Adaptive depth: shapes with inner_burst >= kGatherTridDepthHigh fill the deeper
    // ring; smaller bursts that still clear the lower cutoff use the shallower ring
    // (depth-8 drain on a small burst would barrier-on-(i-N) before earlier reads
    // had time to drain — same anti-pattern that the cutoff guards against, just at
    // a finer granularity). Below the lower cutoff or below the intensity floor,
    // ring is fully off.
    const bool intensity_pass = bytes_per_tile >= conv3d_gather_tuning::kGatherIntensityCutoffBytes;
    // Scratch-backed reader modes already issue larger or serialized reads; the trid ring
    // only affects fallback edge gathers there and measured as overhead.
    const bool gather_trid_ring_allowed = !enable_coalesced_shard_reads && !enable_dram_read_staging;
    uint32_t gather_trids = 0;
    if (gather_trid_ring_allowed && intensity_pass) {
        if (inner_gather_burst >= conv3d_gather_tuning::kGatherTridDepthHigh) {
            gather_trids = conv3d_gather_tuning::kGatherTridDepthHigh;
        } else if (inner_gather_burst >= conv3d_gather_tuning::kGatherInnerBurstCutoff) {
            gather_trids = conv3d_gather_tuning::kGatherTridDepthLow;
        }
    }

    log_debug(
        tt::LogOp,
        "gather trid ring: bytes_per_tile={}, inner_burst={}, gather_trids={}",
        bytes_per_tile,
        inner_gather_burst,
        gather_trids);

    std::vector<uint32_t> reader_compile_time_args = {
        cb_vol2col_rm_id,
        N,
        T_in,
        H_in,
        W_in,
        C_in,
        T_out,
        H_out,
        W_out,
        C_out,
        operation_attributes.padding[0],
        operation_attributes.padding[1],
        operation_attributes.padding[2],
        operation_attributes.kernel_size[0],
        operation_attributes.kernel_size[1],
        operation_attributes.kernel_size[2],
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        in_row_size_bytes,
        C_in_block_bytes,
        out_row_size_bytes,
        is_padding_zeros,
        semaphore_id,
        operation_attributes.stride[0],
        operation_attributes.stride[1],
        operation_attributes.stride[2],
        operation_attributes.dilation[0],
        operation_attributes.dilation[1],
        operation_attributes.dilation[2],
        cb_input_shard_id,
        T_shard_max,
        H_shard_max,
        W_shard_max,
        patch_pad_bytes,
        gather_trids,
        static_cast<uint32_t>(enable_coalesced_shard_reads),
        coalesced_scratch_rows,
        cb_dram_read_scratch_id,
        static_cast<uint32_t>(enable_dram_read_staging),
        dram_read_alignment,
        static_cast<uint32_t>(halo_mode),
        static_cast<uint32_t>(mask_mode),
        operation_attributes.logical_h_mask,
        operation_attributes.logical_w_mask,
        cb_pad_offset_id};
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);
    // Halo buffer accessor follows the input accessor (nullptr when not in halo mode, like bias).
    tt::tt_metal::TensorAccessorArgs(halo_mode ? tensor_args.halo_buffer.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);
    // Per-device offset accessor follows the halo accessor (nullptr when not masking).
    tt::tt_metal::TensorAccessorArgs(mask_mode ? tensor_args.pad_offset_tensor.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet(core_grid);
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // Matmul parameters (out_subblock_h, out_subblock_w, dst_size computed earlier for CB sizing)
    const uint32_t in0_block_w = matmul_K_t;

    TT_FATAL(matmul_N_t % out_subblock_w == 0, "matmul_N_t must be divisible by out_subblock_w");
    TT_FATAL(
        matmul_M_t % out_subblock_h == 0,
        "matmul_M_t ({}) must be divisible by out_subblock_h ({})",
        matmul_M_t,
        out_subblock_h);
    const uint32_t in0_num_subblocks = 1;
    const uint32_t in1_num_subblocks = matmul_N_t / out_subblock_w;

    log_debug(tt::LogOp, "Matmul parameters:");
    log_debug(tt::LogOp, "  matmul_M_t: {}", matmul_M_t);
    log_debug(tt::LogOp, "  matmul_K_t: {}", matmul_K_t);
    log_debug(tt::LogOp, "  matmul_N_t: {}", matmul_N_t);
    log_debug(tt::LogOp, "  dst_size: {}", dst_size);
    log_debug(tt::LogOp, "  in0_block_w: {}", in0_block_w);
    log_debug(tt::LogOp, "  out_subblock_w: {}", out_subblock_w);
    log_debug(tt::LogOp, "  out_subblock_h: {}", out_subblock_h);
    log_debug(tt::LogOp, "  in0_num_subblocks: {}", in0_num_subblocks);
    log_debug(tt::LogOp, "  in1_num_subblocks: {}", in1_num_subblocks);

    std::vector<uint32_t> compute_compile_time_args = {
        cb_vol2col_rm_id,
        cb_vol2col_tiled_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_matmul_result_rm_id,
        cb_reduction_tiled_id,
        cb_worker_ack_back_id,
        N,
        num_patches,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        (uint32_t)use_bias,
        T_out,
        H_out,
        W_out,
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        in0_num_subblocks,
        in1_num_subblocks,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        semaphore_id,
        (uint32_t)use_fp32_partials,
        // Stream final output rows only for many small output writes when there is a writer tail to overlap.
        (uint32_t)(enable_streaming_output ? 1 : 0)};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = CoreRangeSet(core_grid);
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    std::vector<uint32_t> writer_compile_time_args = {
        cb_matmul_result_rm_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_reduction_tiled_id,
        cb_worker_ack_back_id,
        N,
        T_out,
        H_out,
        W_out,
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        num_patches_tile_padded,
        out_row_size_bytes,
        C_out_block_bytes,
        (uint32_t)use_bias,
        semaphore_id,
        static_cast<uint32_t>(enable_streaming_output),
        operation_attributes.output_pad_h,
        operation_attributes.output_pad_w};
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_tensor.has_value() ? bias_tensor.value().buffer() : nullptr)
        .append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = CoreRangeSet(core_grid);
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    tt::tt_metal::Buffer* input_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* halo_buffer_ptr = halo_mode ? tensor_args.halo_buffer.value().buffer() : nullptr;
    tt::tt_metal::Buffer* pad_offset_buffer_ptr = mask_mode ? tensor_args.pad_offset_tensor.value().buffer() : nullptr;
    tt::tt_metal::Buffer* out_buffer = output_tensor.buffer();
    tt::tt_metal::Buffer* weight_buffer = weight_tensor.buffer();
    tt::tt_metal::Buffer* bias_buffer = bias_tensor.has_value() ? bias_tensor.value().buffer() : nullptr;

    // Per-core work assignments. Unassigned cores retain empty work ranges.
    struct CoreWork {
        bool has_work = false;
        bool is_reducer = false;
        uint32_t c_in_idx = 0;
        uint32_t c_out_idx = 0;
        uint32_t t_out_idx = 0;
        uint32_t h_out_idx = 0;
        uint32_t w_out_idx = 0;
        uint32_t reduction_group_id = 0;
        uint32_t mcast_group_id = 0;
        uint32_t c_in_block_start = 0, c_in_block_end = 0;
        uint32_t c_out_block_start = 0, c_out_block_end = 0;
        uint32_t t_out_start = 0, t_out_end = 0;
        uint32_t h_out_start = 0, h_out_end = 0;
        uint32_t w_out_start = 0, w_out_end = 0;
    };

    auto cores = corerange_to_cores(core_grid, num_cores, true);
    auto* device = input_tensor.device();
    std::vector<CoreWork> core_work;
    std::vector<mcast::McastGroup> weights_mcast_groups;

    auto compute_block_ranges = [&](CoreWork& cw) {
        cw.c_in_block_start = cw.c_in_idx * c_in_per_core;
        cw.c_in_block_end = std::min(cw.c_in_block_start + c_in_per_core, C_in_num_blocks);
        cw.c_out_block_start = cw.c_out_idx * c_out_per_core;
        cw.c_out_block_end = std::min(cw.c_out_block_start + c_out_per_core, C_out_num_blocks);
        const uint32_t t_block_start = cw.t_out_idx * t_out_per_core;
        const uint32_t t_block_end = std::min(t_block_start + t_out_per_core, T_out_blocks);
        const uint32_t h_block_start = cw.h_out_idx * h_out_per_core;
        const uint32_t h_block_end = std::min(h_block_start + h_out_per_core, H_out_blocks);
        const uint32_t w_block_start = cw.w_out_idx * w_out_per_core;
        const uint32_t w_block_end = std::min(w_block_start + w_out_per_core, W_out_blocks);
        cw.t_out_start = t_block_start * config.T_out_block;
        cw.t_out_end = std::min(t_block_end * config.T_out_block, T_out);
        cw.h_out_start = h_block_start * config.H_out_block;
        cw.h_out_end = std::min(h_block_end * config.H_out_block, H_out);
        cw.w_out_start = w_block_start * config.W_out_block;
        cw.w_out_end = std::min(w_block_end * config.W_out_block, W_out);
        cw.has_work = (cw.c_in_block_end > cw.c_in_block_start) && (cw.c_out_block_end > cw.c_out_block_start) &&
                      (cw.t_out_end > cw.t_out_start) && (cw.h_out_end > cw.h_out_start) &&
                      (cw.w_out_end > cw.w_out_start);
        cw.is_reducer = cw.has_work && cw.c_in_idx == 0;
        cw.reduction_group_id = cw.c_out_idx * (t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor) +
                                cw.t_out_idx * (h_out_parallel_factor * w_out_parallel_factor) +
                                cw.h_out_idx * w_out_parallel_factor + cw.w_out_idx;
        cw.mcast_group_id = cw.c_in_idx * c_out_parallel_factor + cw.c_out_idx;
    };

    // Collect active assignments in C_in -> C_out -> T -> H -> W order. Compact placement retains
    // this order; rectangular placement arranges each weight-sharing group into a full-row strip.
    const uint32_t hw_par = h_out_parallel_factor * w_out_parallel_factor;
    const uint32_t num_slots = c_in_parallel_factor * total_output_parallel;
    std::vector<CoreWork> active_work;
    active_work.reserve(num_slots);
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        CoreWork cw;
        const uint32_t output_idx = slot % total_output_parallel;
        cw.c_in_idx = slot / total_output_parallel;
        cw.c_out_idx = output_idx / (t_out_parallel_factor * hw_par);
        const uint32_t rem0 = output_idx % (t_out_parallel_factor * hw_par);
        cw.t_out_idx = rem0 / hw_par;
        const uint32_t rem1 = rem0 % hw_par;
        cw.h_out_idx = rem1 / w_out_parallel_factor;
        cw.w_out_idx = rem1 % w_out_parallel_factor;
        compute_block_ranges(cw);
        if (cw.has_work) {
            active_work.push_back(cw);
        }
    }
    log_debug(
        tt::LogOp,
        "Core placement: {} active assignments out of {} nominal slots on {} cores",
        active_work.size(),
        num_slots,
        num_cores);
    if (!place_in_rectangles) {
        core_work = std::move(active_work);
    }
    // Runtime args are emitted for every core. Compact placement appends idle entries;
    // rectangular placement fills only the active slots in each row strip below.
    core_work.resize(num_cores);

    // Prepare exact logical receiver sets. Row-major groups retain their first member as sender.
    if (share_weights && !place_in_rectangles) {
        std::vector<std::vector<uint32_t>> mcast_groups(num_groups);
        for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
            const CoreWork& cw = core_work[core_id];
            if (!cw.has_work) {
                continue;
            }
            mcast_groups[cw.mcast_group_id].push_back(core_id);
        }
        for (uint32_t gid = 0; gid < num_groups; ++gid) {
            const auto& members = mcast_groups[gid];
            if (members.empty()) {
                continue;
            }
            std::vector<CoreRange> receivers;
            for (uint32_t cid : members) {
                receivers.emplace_back(cores.at(cid), cores.at(cid));
            }
            weights_mcast_groups.emplace_back(
                CoreRangeSet(std::move(receivers)), std::vector<CoreCoord>{cores.at(members.front())});
        }
    } else if (place_in_rectangles) {
        // Row-strip placement: each (c_in_idx, c_out_idx) group occupies
        // ceil(members / grid.x) contiguous full-width rows; strips stack along Y in group order.
        // Members fill their strip row-major; unused tail slots receive writes but do not acknowledge.
        // Every group has the same
        // member count by the partitioning invariants above, so reduction partners (same slot,
        // different c_in strip) stay vertically aligned. Cores beyond the last strip get no work
        // and no role.
        //
        // Sender column staggering (SDPA-style): for each group we pick the sender slot inside
        // the bbox whose physical column is furthest from the columns already chosen by
        // previous groups' senders. This spreads DRAM weight reads (and ack convergence)
        // across columns / DRAM channels instead of stacking every sender on column 0. The
        // chosen slot still runs compute as a normal group member.

        std::vector<std::vector<uint32_t>> group_members(num_groups);  // indices into active_work
        for (uint32_t i = 0; i < active_work.size(); ++i) {
            group_members[active_work[i].mcast_group_id].push_back(i);
        }

        // Track physical-x columns already used as senders so we can max-min the next pick.
        std::vector<uint32_t> used_sender_phys_xs;
        used_sender_phys_xs.reserve(num_groups);

        auto pick_sender_within_idx = [&](uint32_t bbox_y_start_log, uint32_t num_members) {
            // Return a within-bbox active slot.  The first group keeps the historical top-left
            // sender; later groups choose the active slot whose physical column is furthest from
            // already-used sender columns.
            uint32_t sender_within_idx = 0;
            if (!used_sender_phys_xs.empty()) {
                uint32_t best_min_dist = 0;
                for (uint32_t cand_idx = 0; cand_idx < num_members; ++cand_idx) {
                    const uint32_t cand_x = cand_idx % grid_size.x;
                    const uint32_t cand_y = bbox_y_start_log + cand_idx / grid_size.x;
                    const uint32_t cand_phys_x =
                        (uint32_t)device->worker_core_from_logical_core(CoreCoord{cand_x, cand_y}).x;
                    uint32_t min_dist = UINT32_MAX;
                    for (uint32_t used_x : used_sender_phys_xs) {
                        const uint32_t d = cand_phys_x > used_x ? cand_phys_x - used_x : used_x - cand_phys_x;
                        min_dist = std::min(min_dist, d);
                    }
                    if (min_dist > best_min_dist) {
                        best_min_dist = min_dist;
                        sender_within_idx = cand_idx;
                    }
                }
            }
            return sender_within_idx;
        };

        uint32_t next_row = 0;
        for (uint32_t gid = 0; gid < num_groups; ++gid) {
            const auto& members = group_members[gid];
            const uint32_t num_members = (uint32_t)members.size();
            const uint32_t rows_this_group = tt::div_up(num_members, (uint32_t)grid_size.x);
            const uint32_t c_in_idx = gid / c_out_parallel_factor;
            const uint32_t c_out_idx = gid % c_out_parallel_factor;

            const uint32_t bbox_y_start_log = next_row;
            const uint32_t bbox_y_end_log = bbox_y_start_log + rows_this_group - 1;
            const uint32_t bbox_x_end_log = grid_size.x - 1;
            const uint32_t bbox_num_cores = grid_size.x * rows_this_group;
            next_row += rows_this_group;

            const uint32_t sender_within_idx = pick_sender_within_idx(bbox_y_start_log, num_members);
            const uint32_t sender_x_log = sender_within_idx % grid_size.x;
            const uint32_t sender_y_log = bbox_y_start_log + sender_within_idx / grid_size.x;
            const auto sender_phys = device->worker_core_from_logical_core(CoreCoord{sender_x_log, sender_y_log});
            used_sender_phys_xs.push_back((uint32_t)sender_phys.x);

            weights_mcast_groups.emplace_back(
                CoreRangeSet(CoreRange({0, bbox_y_start_log}, {bbox_x_end_log, bbox_y_end_log})),
                std::vector<CoreCoord>{{sender_x_log, sender_y_log}},
                num_members - 1);  // Only active receivers acknowledge; the sender is an active member.

            // Weight CB storage is allocated across core_grid, including the unused row tails.
            // Leave those cores idle: hardware can land payloads there without a receiving kernel.
            for (uint32_t within_idx = 0; within_idx < num_members; ++within_idx) {
                const uint32_t x = within_idx % grid_size.x;
                const uint32_t y = bbox_y_start_log + within_idx / grid_size.x;
                const uint32_t target_core_id = y * grid_size.x + x;
                core_work[target_core_id] = active_work[members[within_idx]];
            }

            log_debug(
                tt::LogOp,
                "Weight group {} (c_in={}, c_out={}): {} members + {} passive; bbox logical(0,{})..({},{}); "
                "sender logical({},{}) phys_x={} within_idx={}",
                gid,
                c_in_idx,
                c_out_idx,
                num_members,
                bbox_num_cores - num_members,
                bbox_y_start_log,
                bbox_x_end_log,
                bbox_y_end_log,
                sender_x_log,
                sender_y_log,
                sender_phys.x,
                sender_within_idx);
        }
    }

    std::optional<mcast::McastFamily> weights_mcast_family;
    if (share_weights) {
        mcast::McastConfig weight_config;
        weight_config.noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
        weight_config.base_sem_id = desc.semaphores.size();
        weights_mcast_family.emplace(
            device, std::move(weights_mcast_groups), weight_config, mcast::IrregularReceiverSetMode::ChainLink);
        for (const auto& semaphore : weights_mcast_family->owned_semaphores()) {
            desc.semaphores.push_back(semaphore);
        }
        weights_mcast_family->append_compile_time_args_to(writer_desc.compile_time_args);
    } else {
        mcast::append_absent_mcast_compile_time_args_to(writer_desc.compile_time_args);
    }

    // Build reduction groups from logical reduction keys (c_out_idx, t_out_idx, h_out_idx, w_out_idx).
    const uint32_t num_reduction_groups = total_output_parallel;
    std::vector<std::vector<uint32_t>> reduction_groups(num_reduction_groups);
    std::vector<uint32_t> reducer_core_ids(num_reduction_groups, UINT32_MAX);
    std::vector<std::vector<uint32_t>> worker_core_ids(num_reduction_groups);
    std::vector<uint32_t> reducer_core_physical_xs(num_reduction_groups, 0);
    std::vector<uint32_t> reducer_core_physical_ys(num_reduction_groups, 0);
    std::vector<std::vector<uint32_t>> worker_core_physical_xs(num_reduction_groups);
    std::vector<std::vector<uint32_t>> worker_core_physical_ys(num_reduction_groups);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        const CoreWork& cw = core_work[core_id];
        if (!cw.has_work) {
            continue;
        }
        CoreCoord core = cores.at(core_id);
        const auto core_physical = device->worker_core_from_logical_core(core);
        const uint32_t group_id = cw.reduction_group_id;
        reduction_groups[group_id].push_back(core_id);
        if (cw.is_reducer) {
            reducer_core_ids[group_id] = core_id;
            reducer_core_physical_xs[group_id] = (uint32_t)core_physical.x;
            reducer_core_physical_ys[group_id] = (uint32_t)core_physical.y;
        } else {
            worker_core_ids[group_id].push_back(core_id);
            worker_core_physical_xs[group_id].push_back((uint32_t)core_physical.x);
            worker_core_physical_ys[group_id].push_back((uint32_t)core_physical.y);
        }
    }

    // Log reduction groups.
    for (uint32_t group_id = 0; group_id < reduction_groups.size(); group_id++) {
        const auto& group = reduction_groups[group_id];
        if (!group.empty()) {
            std::string cores_str;
            for (uint32_t core_id : group) {
                CoreCoord core = cores.at(core_id);
                if (!cores_str.empty()) {
                    cores_str += ", ";
                }
                cores_str += "(" + std::to_string(core.x) + "," + std::to_string(core.y) + ")";
            }
            log_debug(
                tt::LogOp,
                "Reduction Group {}: {} cores [{}], ReducerPhysical: ({},{})",
                group_id,
                group.size(),
                cores_str,
                reducer_core_physical_xs[group_id],
                reducer_core_physical_ys[group_id]);
        }
    }

    // Build and set runtime args.
    reader_desc.runtime_args.reserve(num_cores);
    compute_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        const CoreWork& cw = core_work[core_id];
        const uint32_t group_id = cw.reduction_group_id;

        const uint32_t num_workers = cw.has_work ? (uint32_t)worker_core_ids[group_id].size() : 0u;

        // Reader pos[0] is the input buffer address (patched on cache hit via emplace_runtime_args).
        // Trailing halo buffer address (halo mode only) is read by the reader when its halo_mode CT flag set.
        KernelDescriptor::RTArgList reader_args;
        reader_args.reserve(12);
        reader_args.push_back(input_buffer);
        reader_args.push_back(cw.c_in_block_start);
        reader_args.push_back(cw.c_in_block_end);
        reader_args.push_back(cw.c_out_block_start);
        reader_args.push_back(cw.c_out_block_end);
        reader_args.push_back(cw.t_out_start);
        reader_args.push_back(cw.t_out_end);
        reader_args.push_back(cw.h_out_start);
        reader_args.push_back(cw.h_out_end);
        reader_args.push_back(cw.w_out_start);
        reader_args.push_back(cw.w_out_end);
        if (halo_mode) {
            reader_args.push_back(halo_buffer_ptr);
        }
        if (mask_mode) {
            reader_args.push_back(pad_offset_buffer_ptr);
        }
        reader_desc.emplace_runtime_args(core, reader_args);

        compute_desc.runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                cw.c_in_block_start,
                cw.c_in_block_end,
                cw.c_out_block_start,
                cw.c_out_block_end,
                cw.t_out_start,
                cw.t_out_end,
                cw.h_out_start,
                cw.h_out_end,
                cw.w_out_start,
                cw.w_out_end,
                (uint32_t)cw.is_reducer,
                num_workers});

        // Writer pos[0..2] are the output, weight, and bias buffer addresses. nullptr bias becomes
        // an embedded 0 so the kernel-side address is still well-defined.
        KernelDescriptor::RTArgList writer_args;
        writer_args.reserve(
            15 + (weights_mcast_family ? weights_mcast_family->runtime_args(core).size() : 0) +
            (num_workers > 0 ? 2 + 2 * num_workers : 0));
        writer_args.push_back(out_buffer);
        writer_args.push_back(weight_buffer);
        if (bias_buffer != nullptr) {
            writer_args.push_back(bias_buffer);
        } else {
            writer_args.push_back(uint32_t{0});
        }
        writer_args.push_back(cw.c_in_block_start);
        writer_args.push_back(cw.c_in_block_end);
        writer_args.push_back(cw.c_out_block_start);
        writer_args.push_back(cw.c_out_block_end);
        writer_args.push_back(cw.t_out_start);
        writer_args.push_back(cw.t_out_end);
        writer_args.push_back(cw.h_out_start);
        writer_args.push_back(cw.h_out_end);
        writer_args.push_back(cw.w_out_start);
        writer_args.push_back(cw.w_out_end);
        writer_args.push_back((uint32_t)cw.is_reducer);
        writer_args.push_back(num_workers);
        if (weights_mcast_family) {
            weights_mcast_family->append_runtime_args_to(writer_args, core);
        }

        if (num_workers > 0) {
            writer_args.push_back(reducer_core_physical_xs[group_id]);
            writer_args.push_back(reducer_core_physical_ys[group_id]);
            for (uint32_t v : worker_core_physical_xs[group_id]) {
                writer_args.push_back(v);
            }
            for (uint32_t v : worker_core_physical_ys[group_id]) {
                writer_args.push_back(v);
            }
        }

        log_debug(
            tt::LogOp,
            "Core ({},{}): HasWork={}, IsReducer={}, "
            "ReductionGroup={}, C_in_idx={}, C_out_idx={}, NumWorkers={}",
            core.x,
            core.y,
            cw.has_work,
            cw.is_reducer,
            group_id,
            cw.c_in_idx,
            cw.c_out_idx,
            num_workers);

        writer_desc.emplace_runtime_args(core, writer_args);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
