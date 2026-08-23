// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_program_factory.hpp"
#include "conv3d_device_operation_types.hpp"
#include "kernels/conv3d_gather_tuning.hpp"
#include "kernels/conv3d_weight_share.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <algorithm>
#include <filesystem>
#include <ranges>
#include <string>
#include <tt-metalium/hal.hpp>

namespace ttnn::experimental::prim {

// Largest divisor of n that is <= cap. Always returns at least 1.
static uint32_t largest_divisor_up_to(uint32_t n, uint32_t cap) {
    for (uint32_t d = std::min(n, cap); d >= 1; d--) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

ttnn::device_operation::ProgramArtifacts Conv3dProgramFactory::create_program_artifacts(
    const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& output_tensor = tensor_return_value;
    auto* device = input_tensor.device();

    // Extract config from operation_attributes
    const auto& config = operation_attributes.config;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const tt::tt_metal::experimental::DFBSpecName dfb_vol2col_rm{"vol2col_rm"};
    const tt::tt_metal::experimental::DFBSpecName dfb_vol2col_tiled{"vol2col_tiled"};
    const tt::tt_metal::experimental::DFBSpecName dfb_weight_tiled{"weight_tiled"};
    const tt::tt_metal::experimental::DFBSpecName dfb_matmul_interm_tiled{"matmul_interm_tiled"};
    const tt::tt_metal::experimental::DFBSpecName dfb_matmul_result_rm{"matmul_result_rm"};
    const tt::tt_metal::experimental::DFBSpecName dfb_reduction_tiled{"reduction_tiled"};
    const tt::tt_metal::experimental::DFBSpecName dfb_worker_ack_back{"worker_ack_back"};
    const tt::tt_metal::experimental::DFBSpecName dfb_bias_tiled{"bias_tiled"};
    const tt::tt_metal::experimental::DFBSpecName dfb_dram_read_scratch{"dram_read_scratch"};
    const tt::tt_metal::experimental::DFBSpecName dfb_pad_offset{"pad_offset"};
    const tt::tt_metal::experimental::DFBSpecName dfb_input_shard{"input_shard"};
    const tt::tt_metal::experimental::TensorParamName input_tensor_parameter{"input"};
    const tt::tt_metal::experimental::TensorParamName output_tensor_parameter{"output"};
    const tt::tt_metal::experimental::TensorParamName weight_tensor_parameter{"weight"};
    const tt::tt_metal::experimental::TensorParamName bias_tensor_parameter{"bias"};
    const tt::tt_metal::experimental::TensorParamName halo_tensor_parameter{"halo"};
    const tt::tt_metal::experimental::TensorParamName pad_offset_tensor_parameter{"pad_offset"};
    const tt::tt_metal::experimental::SemaphoreSpecName reduction_done_semaphore{"reduction_done"};
    const tt::tt_metal::experimental::SemaphoreSpecName weight_sender_semaphore{"weight_sender"};
    const tt::tt_metal::experimental::SemaphoreSpecName weight_receiver_semaphore{"weight_receiver"};
    const tt::tt_metal::experimental::KernelSpecName reader_kernel{"reader"};
    const tt::tt_metal::experimental::KernelSpecName compute_kernel{"compute"};
    const tt::tt_metal::experimental::KernelSpecName writer_kernel{"writer"};

    auto add_dfb_accessor_alias = [](auto& bindings,
                                     const tt::tt_metal::experimental::DFBSpecName& target_dfb,
                                     tt::tt_metal::experimental::DFBEndpointType endpoint_type,
                                     std::string alias) {
        auto binding = std::ranges::find_if(bindings, [&](const auto& candidate) {
            return candidate.dfb_spec_name == target_dfb && candidate.endpoint_type == endpoint_type;
        });
        TT_FATAL(
            binding != bindings.end(),
            "Missing Conv3D {} binding for DFB '{}' while adding accessor alias '{}'",
            endpoint_type == tt::tt_metal::experimental::DFBEndpointType::PRODUCER ? "producer" : "consumer",
            target_dfb,
            alias);
        binding->accessor_aliases.push_back(std::move(alias));
    };

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dfbs;

    auto grid_size = config.compute_with_storage_grid_size;
    auto core_grid = tt::tt_metal::CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
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
    // Stream one C_out row at a time only when there is no later C_in reduction and several
    // small writes can overlap the compute tail. Larger writes are more efficient as one drain.
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
    dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
        .unique_id = dfb_vol2col_rm,
        .entry_size = padded_patch_size_bytes,
        .num_entries = vol2col_rm_pages,
        .data_format_metadata = data_format,
    });

    dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
        .unique_id = dfb_vol2col_tiled,
        .entry_size = tile_size,
        .num_entries = out_subblock_h * matmul_K_t,
        .data_format_metadata = data_format,
    });

    dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
        .unique_id = dfb_weight_tiled,
        .entry_size = tile_size,
        .num_entries = matmul_K_t * matmul_N_t,
        .data_format_metadata = data_format,
    });

    // Use fp32 partials whenever we have multiple C_in blocks and fp32 dest is enabled.
    // This eliminates bf16 truncation between C_in block partial sums.
    bool use_fp32_partials = fp32_dest_acc_en && C_in_num_blocks > 1;
    auto partial_data_format = use_fp32_partials ? tt::DataFormat::Float32 : data_format;
    auto partial_tile_size = tt::tile_size(partial_data_format);

    dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
        .unique_id = dfb_matmul_interm_tiled,
        .entry_size = partial_tile_size,
        .num_entries = matmul_M_t * matmul_N_t,
        .data_format_metadata = partial_data_format,
        .advanced_options = {.allow_instance_multi_binding = C_in_num_blocks > 1},
    });

    // NOTE: Most kernels create RM CB with tile_size pages and num_tile number of pages.
    // Using stick pages led to PCC issues.
    dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
        .unique_id = dfb_matmul_result_rm,
        .entry_size = tile_size,
        // Untilize writes padded rows, so this must hold the full output tile block.
        .num_entries = matmul_M_t * matmul_N_t,
        .data_format_metadata = data_format,
    });

    if (C_in_num_blocks > 1) {
        // Multi-core reduction step: each core computes a partial sum, then they reduce
        // Use same format as partials CB so reduction adds matching formats
        dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = dfb_reduction_tiled,
            .entry_size = partial_tile_size,
            .num_entries = matmul_M_t * matmul_N_t,
            .data_format_metadata = partial_data_format,
            .advanced_options = {.allow_instance_multi_binding = true},
        });

        dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = dfb_worker_ack_back,
            .entry_size = tile_size,
            .num_entries = 1,
            .data_format_metadata = data_format,
        });
    }

    if (use_bias) {
        dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = dfb_bias_tiled,
            .entry_size = tile_size,
            .num_entries = matmul_N_t,
            .data_format_metadata = data_format,
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
        tt::LogOp,
        "CB matmul_interm_tiled: page_size={} bytes, num_pages={}",
        partial_tile_size,
        matmul_M_t * matmul_N_t);
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
    if (enable_dram_read_staging) {
        dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = dfb_dram_read_scratch,
            .entry_size = dram_read_scratch_page_bytes,
            .num_entries = 1,
            .data_format_metadata = data_format,
        });
    }

    // Tiny landing CB for the per-device [h_start, w_start] offset page (mask mode only).
    if (mask_mode) {
        const uint32_t pad_offset_page_bytes =
            tt::round_up(2u * static_cast<uint32_t>(sizeof(uint32_t)), dram_read_alignment);
        dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = dfb_pad_offset,
            .entry_size = pad_offset_page_bytes,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
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
            dfbs.push_back(tt::tt_metal::experimental::DataflowBufferSpec{
                .unique_id = dfb_input_shard,
                .entry_size = C_in_block_bytes,
                .num_entries = shard_positions_alloc,
                .data_format_metadata = data_format,
            });

            log_debug(
                tt::LogOp,
                "L1 prefetch: T_shard_max={}, H_shard_max={}, W_shard_max={}, shard_positions={}, "
                "scratch_positions={}, shard_bytes={}",
                T_shard_max,
                H_shard_max,
                W_shard_max,
                shard_positions_max,
                coalesced_scratch_rows * coalesced_scratch_pages_per_row,
                shard_bytes_alloc);
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
        "DRAM read staging: enable={}, alignment={}, scratch_page_bytes={}",
        enable_dram_read_staging,
        dram_read_alignment,
        dram_read_scratch_page_bytes);

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

    // Weight sharing: all cores with the same (c_in_idx, c_out_idx) read identical weights.
    // weight_share_mode (see WeightShareMode in conv3d_weight_share.hpp):
    //   Disabled — single-core group: each active core reads its own weight slice.
    //   Chain    — per-group forwarding chain (SDPA-style hop chain).
    //   Mcast    — each group multicasts over its own row-strip rectangle.
    //
    // Layout for mcast: each group occupies `rows_per_group = ceil(group_size / grid.x)`
    // contiguous rows, full grid width. The `num_groups` strips stack along Y. Within a strip,
    // active cores fill row-major; the trailing slots become passive participants. This makes
    // every group a clean rectangle, lets each one fire its own hardware multicast, and keeps
    // reduction-pair members vertically aligned (same x, y differing by a multiple of strip
    // height) so reduction reads stay short. Falls back to chain if the strips don't fit.
    const uint32_t group_size = t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor;
    const uint32_t num_groups = c_in_parallel_factor * c_out_parallel_factor;
    WeightShareMode weight_share_mode = WeightShareMode::Disabled;
    uint32_t mcast_rows_per_group = 0;
    if (group_size > 1) {
        const uint32_t rows_per_group = (group_size + grid_size.x - 1) / grid_size.x;
        const bool mcast_fits = (uint64_t)num_groups * rows_per_group <= grid_size.y;
        if (mcast_fits) {
            weight_share_mode = WeightShareMode::Mcast;
            mcast_rows_per_group = rows_per_group;
        } else {
            weight_share_mode = WeightShareMode::Chain;
        }
    }
    log_debug(
        tt::LogOp,
        "Weight share: mode={}, group_size={}, num_groups={}, rows_per_group={}",
        static_cast<uint32_t>(weight_share_mode),
        group_size,
        num_groups,
        mcast_rows_per_group);

    // All three legacy initial values are zero (INVALID == 0). The reduction semaphore is
    // dual-purpose: reducer instances count ready workers, while workers wait for the reducer ack.
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::SemaphoreSpec> semaphores = {
        tt::tt_metal::experimental::SemaphoreSpec{
            .unique_id = reduction_done_semaphore, .target_nodes = tt::tt_metal::CoreRangeSet(core_grid)},
        tt::tt_metal::experimental::SemaphoreSpec{
            .unique_id = weight_sender_semaphore, .target_nodes = tt::tt_metal::CoreRangeSet(core_grid)},
        tt::tt_metal::experimental::SemaphoreSpec{
            .unique_id = weight_receiver_semaphore, .target_nodes = tt::tt_metal::CoreRangeSet(core_grid)},
    };

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

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DFBBinding> reader_bindings = {
        tt::tt_metal::experimental::ProducerOf(dfb_vol2col_rm, "vol2col_rm"),
    };
    if (T_shard_max > 0) {
        reader_bindings.push_back(tt::tt_metal::experimental::ProducerOf(dfb_input_shard, "input_shard"));
        reader_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(dfb_input_shard, "input_shard"));
    } else {
        add_dfb_accessor_alias(
            reader_bindings, dfb_vol2col_rm, tt::tt_metal::experimental::DFBEndpointType::PRODUCER, "input_shard");
    }
    if (enable_dram_read_staging) {
        reader_bindings.push_back(tt::tt_metal::experimental::ProducerOf(dfb_dram_read_scratch, "dram_read_scratch"));
        reader_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(dfb_dram_read_scratch, "dram_read_scratch"));
    } else if (T_shard_max > 0) {
        add_dfb_accessor_alias(
            reader_bindings,
            dfb_input_shard,
            tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            "dram_read_scratch");
    } else {
        add_dfb_accessor_alias(
            reader_bindings,
            dfb_vol2col_rm,
            tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            "dram_read_scratch");
    }
    if (mask_mode) {
        reader_bindings.push_back(tt::tt_metal::experimental::ProducerOf(dfb_pad_offset, "pad_offset"));
        reader_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(dfb_pad_offset, "pad_offset"));
    } else {
        add_dfb_accessor_alias(
            reader_bindings, dfb_vol2col_rm, tt::tt_metal::experimental::DFBEndpointType::PRODUCER, "pad_offset");
    }
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::TensorBinding> reader_tensor_bindings = {
        tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = input_tensor_parameter, .accessor_name = "input"}};
    // Keep every accessor name resolvable in all specializations. Disabled halo/mask paths alias
    // their unused accessor to input; the compile-time mode guarantees it is never dereferenced.
    if (halo_mode) {
        reader_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = halo_tensor_parameter, .accessor_name = "halo"});
    } else {
        reader_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = input_tensor_parameter, .accessor_name = "halo"});
    }
    if (mask_mode) {
        reader_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = pad_offset_tensor_parameter, .accessor_name = "pad_offset"});
    } else {
        reader_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = input_tensor_parameter, .accessor_name = "pad_offset"});
    }
    tt::tt_metal::experimental::KernelSpec reader_spec{
        .unique_id = reader_kernel,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp"},
        .dfb_bindings = std::move(reader_bindings),
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args =
            {{"N", N},
             {"T_in", T_in},
             {"H_in", H_in},
             {"W_in", W_in},
             {"padding_t", operation_attributes.padding[0]},
             {"padding_h", operation_attributes.padding[1]},
             {"padding_w", operation_attributes.padding[2]},
             {"kT", operation_attributes.kernel_size[0]},
             {"kH", operation_attributes.kernel_size[1]},
             {"kW", operation_attributes.kernel_size[2]},
             {"T_block_size", config.T_out_block},
             {"H_block_size", config.H_out_block},
             {"W_block_size", config.W_out_block},
             {"in_row_size_bytes", in_row_size_bytes},
             {"C_in_block_bytes", C_in_block_bytes},
             {"is_padding_zeros_value", static_cast<uint32_t>(is_padding_zeros)},
             {"stride_t", operation_attributes.stride[0]},
             {"stride_h", operation_attributes.stride[1]},
             {"stride_w", operation_attributes.stride[2]},
             {"dilation_t", operation_attributes.dilation[0]},
             {"dilation_h", operation_attributes.dilation[1]},
             {"dilation_w", operation_attributes.dilation[2]},
             {"T_shard_max", T_shard_max},
             {"H_shard_max", H_shard_max},
             {"W_shard_max", W_shard_max},
             {"patch_pad_bytes", patch_pad_bytes},
             {"gather_trids", gather_trids},
             {"enable_coalesced_shard_reads_value", static_cast<uint32_t>(enable_coalesced_shard_reads)},
             {"coalesced_scratch_rows", coalesced_scratch_rows},
             {"enable_dram_read_staging_value", static_cast<uint32_t>(enable_dram_read_staging)},
             {"dram_read_alignment", dram_read_alignment},
             {"halo_mode_value", static_cast<uint32_t>(halo_mode)},
             {"mask_mode_value", static_cast<uint32_t>(mask_mode)},
             {"logical_h_mask", operation_attributes.logical_h_mask},
             {"logical_w_mask", operation_attributes.logical_w_mask}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"c_in_block_start",
                  "c_in_block_end",
                  "c_out_block_start",
                  "c_out_block_end",
                  "t_out_start",
                  "t_out_end",
                  "h_out_start",
                  "h_out_end",
                  "w_out_start",
                  "w_out_end"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

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

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DFBBinding> compute_bindings = {
        tt::tt_metal::experimental::ConsumerOf(dfb_vol2col_rm, "vol2col_rm"),
        tt::tt_metal::experimental::ProducerOf(dfb_vol2col_tiled, "vol2col_tiled"),
        tt::tt_metal::experimental::ConsumerOf(dfb_vol2col_tiled, "vol2col_tiled"),
        tt::tt_metal::experimental::ConsumerOf(dfb_weight_tiled, "weight_tiled"),
        tt::tt_metal::experimental::ProducerOf(dfb_matmul_interm_tiled, "matmul_interm_tiled"),
        tt::tt_metal::experimental::ConsumerOf(dfb_matmul_interm_tiled, "matmul_interm_tiled"),
        tt::tt_metal::experimental::ProducerOf(dfb_matmul_result_rm, "matmul_result_rm"),
    };
    if (use_bias) {
        compute_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(dfb_bias_tiled, "bias_tiled"));
    }
    if (C_in_num_blocks > 1) {
        compute_bindings.push_back(tt::tt_metal::experimental::ProducerOf(dfb_reduction_tiled, "reduction_tiled"));
        compute_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(dfb_reduction_tiled, "reduction_tiled"));
        compute_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(dfb_worker_ack_back, "worker_ack_back"));
    }
    if (!use_bias) {
        add_dfb_accessor_alias(
            compute_bindings, dfb_weight_tiled, tt::tt_metal::experimental::DFBEndpointType::CONSUMER, "bias_tiled");
    }
    if (C_in_num_blocks == 1) {
        add_dfb_accessor_alias(
            compute_bindings,
            dfb_matmul_interm_tiled,
            tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            "reduction_tiled");
        add_dfb_accessor_alias(
            compute_bindings,
            dfb_matmul_interm_tiled,
            tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            "worker_ack_back");
    }
    auto compute_hw = ttnn::to_compute_hardware_config(
        device->arch(),
        ttnn::ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .math_approx_mode = math_approx_mode,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en});
    auto& modes = unpack_modes(compute_hw);
    if (fp32_dest_acc_en && data_format == tt::DataFormat::Float32) {
        // Legacy left unpack_to_dest_mode at its default, so FP32 operands are unpacked into
        // SrcA/B. Metalium 2.0 requires that default to be explicit with a 32-bit Dest.
        modes.emplace(dfb_vol2col_rm, tt::tt_metal::UnpackMode::UnpackToSrc);
        modes.emplace(dfb_vol2col_tiled, tt::tt_metal::UnpackMode::UnpackToSrc);
        modes.emplace(dfb_weight_tiled, tt::tt_metal::UnpackMode::UnpackToSrc);
        modes.emplace(dfb_matmul_interm_tiled, tt::tt_metal::UnpackMode::UnpackToSrc);
        if (use_bias) {
            modes.emplace(dfb_bias_tiled, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        if (C_in_num_blocks > 1) {
            modes.emplace(dfb_reduction_tiled, tt::tt_metal::UnpackMode::UnpackToSrc);
            modes.emplace(dfb_worker_ack_back, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
    }
    if (use_fp32_partials) {
        modes.emplace(dfb_matmul_interm_tiled, tt::tt_metal::UnpackMode::UnpackToSrc);
        modes.emplace(dfb_reduction_tiled, tt::tt_metal::UnpackMode::UnpackToSrc);
    }
    tt::tt_metal::experimental::KernelSpec compute_spec{
        .unique_id = compute_kernel,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp"},
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_bindings),
        .compile_time_args =
            {{"N", N},
             {"num_patches", num_patches},
             {"matmul_M_t", matmul_M_t},
             {"matmul_K_t", matmul_K_t},
             {"matmul_N_t", matmul_N_t},
             {"use_bias_value", static_cast<uint32_t>(use_bias)},
             {"T_block_size", config.T_out_block},
             {"H_block_size", config.H_out_block},
             {"W_block_size", config.W_out_block},
             {"in0_num_subblocks", in0_num_subblocks},
             {"in1_num_subblocks", in1_num_subblocks},
             {"in0_block_w", in0_block_w},
             {"subblock_h", out_subblock_h},
             {"subblock_w", out_subblock_w},
             {"use_fp32_partials_value", static_cast<uint32_t>(use_fp32_partials)},
             {"enable_streaming_output_value", static_cast<uint32_t>(enable_streaming_output)},
             {"has_reduction_value", static_cast<uint32_t>(C_in_num_blocks > 1)}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"c_in_block_start",
                  "c_in_block_end",
                  "c_out_block_start",
                  "c_out_block_end",
                  "t_out_start",
                  "t_out_end",
                  "h_out_start",
                  "h_out_end",
                  "w_out_start",
                  "w_out_end",
                  "is_reducer",
                  "num_workers"}},
        .hw_config = compute_hw,
    };

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DFBBinding> writer_bindings = {
        tt::tt_metal::experimental::ConsumerOf(dfb_matmul_result_rm, "matmul_result_rm"),
        tt::tt_metal::experimental::ProducerOf(dfb_weight_tiled, "weight_tiled")};
    if (use_bias) {
        writer_bindings.push_back(tt::tt_metal::experimental::ProducerOf(dfb_bias_tiled, "bias_tiled"));
    }
    if (C_in_num_blocks > 1) {
        // The writer needs the local partials address as the template for remote-worker reads.
        // Gen1 multi-binding lowers this compute/writer shared resource to one plain explicit-sync CB.
        writer_bindings.push_back(
            tt::tt_metal::experimental::ProducerOf(dfb_matmul_interm_tiled, "matmul_interm_tiled"));
        writer_bindings.push_back(
            tt::tt_metal::experimental::ConsumerOf(dfb_matmul_interm_tiled, "matmul_interm_tiled"));
        writer_bindings.push_back(tt::tt_metal::experimental::ProducerOf(dfb_reduction_tiled, "reduction_tiled"));
        writer_bindings.push_back(tt::tt_metal::experimental::ConsumerOf(dfb_reduction_tiled, "reduction_tiled"));
        writer_bindings.push_back(tt::tt_metal::experimental::ProducerOf(dfb_worker_ack_back, "worker_ack_back"));
    }
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::TensorBinding> writer_tensor_bindings = {
        tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = output_tensor_parameter, .accessor_name = "output"},
        tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = weight_tensor_parameter, .accessor_name = "weight"},
    };
    if (use_bias) {
        writer_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = bias_tensor_parameter, .accessor_name = "bias"});
    } else {
        add_dfb_accessor_alias(
            writer_bindings, dfb_weight_tiled, tt::tt_metal::experimental::DFBEndpointType::PRODUCER, "bias_tiled");
        writer_tensor_bindings.push_back(tt::tt_metal::experimental::TensorBinding{
            .tensor_parameter_name = weight_tensor_parameter, .accessor_name = "bias"});
    }
    if (C_in_num_blocks == 1) {
        add_dfb_accessor_alias(
            writer_bindings,
            dfb_weight_tiled,
            tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            "matmul_interm_tiled");
        add_dfb_accessor_alias(
            writer_bindings,
            dfb_weight_tiled,
            tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            "reduction_tiled");
        add_dfb_accessor_alias(
            writer_bindings,
            dfb_weight_tiled,
            tt::tt_metal::experimental::DFBEndpointType::PRODUCER,
            "worker_ack_back");
    }
    const uint32_t num_worker_coord_varargs = C_in_num_blocks > 1 ? 2 + 2 * (C_in_num_blocks - 1) : 0;
    const auto writer_hw_config = ttnn::create_writer_datamovement_config(device->arch());
    std::optional<tt::tt_metal::NOC> writer_mcast_noc;
    if (weight_share_mode == WeightShareMode::Mcast) {
        TT_FATAL(
            std::holds_alternative<tt::tt_metal::experimental::DataMovementGen1Config>(writer_hw_config),
            "Conv3D weight multicast currently requires Gen1 data-movement hardware");
        writer_mcast_noc = std::get<tt::tt_metal::experimental::DataMovementGen1Config>(writer_hw_config).noc;
    }
    tt::tt_metal::experimental::KernelSpec writer_spec{
        .unique_id = writer_kernel,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp"},
        .dfb_bindings = std::move(writer_bindings),
        .semaphore_bindings =
            {tt::tt_metal::experimental::SemaphoreBinding{
                 .semaphore_spec_name = reduction_done_semaphore, .accessor_name = "reduction_done"},
             tt::tt_metal::experimental::SemaphoreBinding{
                 .semaphore_spec_name = weight_sender_semaphore, .accessor_name = "weight_sender"},
             tt::tt_metal::experimental::SemaphoreBinding{
                 .semaphore_spec_name = weight_receiver_semaphore, .accessor_name = "weight_receiver"}},
        .tensor_bindings = std::move(writer_tensor_bindings),
        .compile_time_args =
            {{"N", N},
             {"T_out", T_out},
             {"H_out", H_out},
             {"W_out", W_out},
             {"T_block_size", config.T_out_block},
             {"H_block_size", config.H_out_block},
             {"W_block_size", config.W_out_block},
             {"C_out_num_blocks", C_out_num_blocks},
             {"matmul_M_t", matmul_M_t},
             {"matmul_K_t", matmul_K_t},
             {"matmul_N_t", matmul_N_t},
             {"C_out_block_bytes", C_out_block_bytes},
             {"use_bias_value", static_cast<uint32_t>(use_bias)},
             {"weight_share_mode_value", static_cast<uint32_t>(weight_share_mode)},
             {"enable_streaming_output_value", static_cast<uint32_t>(enable_streaming_output)},
             {"output_pad_h", operation_attributes.output_pad_h},
             {"output_pad_w", operation_attributes.output_pad_w},
             {"has_reduction_value", static_cast<uint32_t>(C_in_num_blocks > 1)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"c_in_block_start", "c_in_block_end",     "c_out_block_start",
                                   "c_out_block_end",  "t_out_start",        "t_out_end",
                                   "h_out_start",      "h_out_end",          "w_out_start",
                                   "w_out_end",        "is_reducer",         "weight_share_role_value",
                                   "weight_src_noc_x", "weight_src_noc_y",   "chain_succ_noc_x",
                                   "chain_succ_noc_y", "mcast_bbox_start_x", "mcast_bbox_start_y",
                                   "mcast_bbox_end_x", "mcast_bbox_end_y",   "mcast_num_dests",
                                   "mcast_num_iters",  "num_workers"}},
        .hw_config = writer_hw_config,
        .advanced_options = {.num_runtime_varargs = num_worker_coord_varargs},
    };

    // Per-core work assignment via the original core_id row-major mapping. See WeightShareRole
    // in conv3d_weight_share.hpp for the role values.
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
        WeightShareRole weight_share_role = WeightShareRole::Local;
        // Where this core receives weights from: chain predecessor (chain roles) or mcast sender
        // (mcast receiver/passive). McastSender carries its own coord for uniform runtime args.
        uint32_t weight_src_noc_x = 0, weight_src_noc_y = 0;
        // Chain forwarding target (chain injector/middle). Unused for other roles.
        uint32_t chain_succ_noc_x = 0, chain_succ_noc_y = 0;
        // Mcast bbox in physical NoC coords (already swapped for NOC_1). Sender role only.
        uint32_t mcast_bbox_start_x = 0, mcast_bbox_start_y = 0;
        uint32_t mcast_bbox_end_x = 0, mcast_bbox_end_y = 0;
        uint32_t mcast_num_dests = 0;
        // Iterations for passive participation: matches active receivers' loop count.
        uint32_t mcast_num_iters = 0;
    };

    auto cores = corerange_to_cores(core_grid, num_cores, true);
    std::vector<CoreWork> core_work(num_cores);

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

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreWork& cw = core_work[core_id];
        const uint32_t output_idx = core_id % total_output_parallel;
        cw.c_in_idx = core_id / total_output_parallel;
        const uint32_t hw_par = h_out_parallel_factor * w_out_parallel_factor;
        cw.c_out_idx = output_idx / (t_out_parallel_factor * hw_par);
        const uint32_t rem0 = output_idx % (t_out_parallel_factor * hw_par);
        cw.t_out_idx = rem0 / hw_par;
        const uint32_t rem1 = rem0 % hw_par;
        cw.h_out_idx = rem1 / w_out_parallel_factor;
        cw.w_out_idx = rem1 % w_out_parallel_factor;
        compute_block_ranges(cw);
    }

    // Per-mode setup: chain (multi-group) builds per-group forwarding chains; mcast (single
    // group) computes a logical bbox and assigns roles to all cores within it (active and
    // passive participants).
    if (weight_share_mode == WeightShareMode::Chain) {
        // Build per-group chain: order cores by core_id, link each one's predecessor and successor.
        // Chain ordering by core_id keeps the chain "physically nearby" since core_id maps row-major
        // onto the grid, which keeps each hop short on the NoC.
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
            if (members.size() < 2) {
                continue;  // single-core group: leave role=0 (local DRAM read).
            }
            for (size_t i = 0; i < members.size(); ++i) {
                const uint32_t cid = members[i];
                CoreWork& cw = core_work[cid];
                const bool is_injector = (i == 0);
                const bool is_tail = (i + 1 == members.size());
                cw.weight_share_role = is_injector
                                           ? WeightShareRole::ChainInjector
                                           : (is_tail ? WeightShareRole::ChainTail : WeightShareRole::ChainMiddle);
                if (!is_injector) {
                    const auto pred_phys = device->worker_core_from_logical_core(cores.at(members[i - 1]));
                    cw.weight_src_noc_x = (uint32_t)pred_phys.x;
                    cw.weight_src_noc_y = (uint32_t)pred_phys.y;
                }
                if (!is_tail) {
                    const auto succ_phys = device->worker_core_from_logical_core(cores.at(members[i + 1]));
                    cw.chain_succ_noc_x = (uint32_t)succ_phys.x;
                    cw.chain_succ_noc_y = (uint32_t)succ_phys.y;
                }
            }
        }
    } else if (weight_share_mode == WeightShareMode::Mcast) {
        // Row-strip placement: each (c_in_idx, c_out_idx) group occupies `mcast_rows_per_group`
        // contiguous rows of the worker grid. The default row-major core_id assignment above
        // doesn't match this layout, so reassign every CoreWork from the rectangle.
        //
        // Sender column staggering (SDPA-style): for each group we pick the sender slot inside
        // the bbox whose physical column is furthest from the columns already chosen by
        // previous groups' senders. This spreads DRAM weight reads (and ack convergence)
        // across columns / DRAM channels instead of stacking every sender on column 0. The
        // chosen slot still runs compute as a normal mcast member (role 4 = sender + work).
        const uint32_t rows_per_group = mcast_rows_per_group;
        const uint32_t bbox_num_cores = grid_size.x * rows_per_group;
        const uint32_t hw_par = h_out_parallel_factor * w_out_parallel_factor;

        // Reset assignments before re-laying out.
        for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
            core_work[core_id] = CoreWork{};
        }

        // Track physical-x columns already used as senders so we can max-min the next pick.
        std::vector<uint32_t> used_sender_phys_xs;
        used_sender_phys_xs.reserve(num_groups);

        auto pick_sender_within_idx = [&](uint32_t bbox_y_start_log) {
            // Return a within-bbox active slot.  The first group keeps the historical top-left
            // sender; later groups choose the active slot whose physical column is furthest from
            // already-used sender columns.
            uint32_t sender_within_idx = 0;
            if (!used_sender_phys_xs.empty()) {
                uint32_t best_min_dist = 0;
                for (uint32_t cand_idx = 0; cand_idx < group_size; ++cand_idx) {
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

        for (uint32_t gid = 0; gid < num_groups; ++gid) {
            // mcast_group_id ordering matches the default: c_in_idx * c_out_par + c_out_idx.
            const uint32_t c_in_idx = gid / c_out_parallel_factor;
            const uint32_t c_out_idx = gid % c_out_parallel_factor;

            // Per-group iteration count must match the active receivers' writer loop.
            // Sharing requires group_size > 1, which implies c_out_parallel_factor equals
            // C_out_num_blocks and c_out_per_core is one.  The invariant above likewise pins
            // c_in_per_core to one, so every logical sharing group owns one valid block in
            // each channel dimension.  Rectangle-padding slots use this same positive count.
            const uint32_t this_c_in_blocks = std::min(c_in_per_core, C_in_num_blocks - c_in_idx * c_in_per_core);
            const uint32_t this_c_out_blocks = std::min(c_out_per_core, C_out_num_blocks - c_out_idx * c_out_per_core);
            const uint32_t mcast_iters = N * this_c_in_blocks * this_c_out_blocks;

            const uint32_t bbox_y_start_log = gid * rows_per_group;
            const uint32_t bbox_y_end_log = bbox_y_start_log + rows_per_group - 1;
            const uint32_t bbox_x_end_log = grid_size.x - 1;

            auto bbox_start_phys = device->worker_core_from_logical_core(CoreCoord{0, bbox_y_start_log});
            auto bbox_end_phys = device->worker_core_from_logical_core(CoreCoord{bbox_x_end_log, bbox_y_end_log});
            // Conv2d-style swap so the multicast hardware sees the rect in NOC_1's orientation.
            if (*writer_mcast_noc == tt::tt_metal::NOC::NOC_1) {
                std::swap(bbox_start_phys, bbox_end_phys);
            }

            const uint32_t sender_within_idx = pick_sender_within_idx(bbox_y_start_log);
            const uint32_t sender_x_log = sender_within_idx % grid_size.x;
            const uint32_t sender_y_log = bbox_y_start_log + sender_within_idx / grid_size.x;
            const auto sender_phys = device->worker_core_from_logical_core(CoreCoord{sender_x_log, sender_y_log});
            used_sender_phys_xs.push_back((uint32_t)sender_phys.x);

            // Sender is inside the bbox; EXCLUDE_SRC mcast → num_dests = bbox_cores - 1.
            const uint32_t num_receivers = bbox_num_cores - 1;

            for (uint32_t y_off = 0; y_off < rows_per_group; ++y_off) {
                for (uint32_t x = 0; x < grid_size.x; ++x) {
                    const uint32_t y = bbox_y_start_log + y_off;
                    const uint32_t within_idx = y_off * grid_size.x + x;
                    const uint32_t target_core_id = y * grid_size.x + x;
                    CoreWork& cw = core_work[target_core_id];

                    const bool is_sender_slot = within_idx == sender_within_idx;

                    if (within_idx < group_size) {
                        cw.c_in_idx = c_in_idx;
                        cw.c_out_idx = c_out_idx;
                        cw.t_out_idx = within_idx / hw_par;
                        const uint32_t rem = within_idx % hw_par;
                        cw.h_out_idx = rem / w_out_parallel_factor;
                        cw.w_out_idx = rem % w_out_parallel_factor;
                        compute_block_ranges(cw);
                        cw.weight_share_role =
                            is_sender_slot ? WeightShareRole::McastSender : WeightShareRole::McastReceiver;
                    } else {
                        cw.weight_share_role = WeightShareRole::McastPassive;
                    }
                    // Receivers/passives source weights from the sender. Sender carries its own
                    // coord for uniform runtime args; writer.cpp ignores it for McastSender.
                    cw.weight_src_noc_x = (uint32_t)sender_phys.x;
                    cw.weight_src_noc_y = (uint32_t)sender_phys.y;
                    cw.mcast_bbox_start_x = (uint32_t)bbox_start_phys.x;
                    cw.mcast_bbox_start_y = (uint32_t)bbox_start_phys.y;
                    cw.mcast_bbox_end_x = (uint32_t)bbox_end_phys.x;
                    cw.mcast_bbox_end_y = (uint32_t)bbox_end_phys.y;
                    cw.mcast_num_dests = num_receivers;
                    cw.mcast_num_iters = mcast_iters;
                }
            }

            log_debug(
                tt::LogOp,
                "Mcast group {} (c_in={}, c_out={}): bbox logical(0,{})..({},{}); "
                "phys swapped({},{})..({},{}); sender logical({},{}) phys_x={} within_idx={}; "
                "num_receivers={}, mcast_iters={}",
                gid,
                c_in_idx,
                c_out_idx,
                bbox_y_start_log,
                bbox_x_end_log,
                bbox_y_end_log,
                bbox_start_phys.x,
                bbox_start_phys.y,
                bbox_end_phys.x,
                bbox_end_phys.y,
                sender_x_log,
                sender_y_log,
                sender_phys.x,
                sender_within_idx,
                num_receivers,
                mcast_iters);
        }
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

    tt::tt_metal::experimental::KernelRunArgs reader_run{.kernel = reader_kernel};
    tt::tt_metal::experimental::KernelRunArgs compute_run{.kernel = compute_kernel};
    tt::tt_metal::experimental::KernelRunArgs writer_run{.kernel = writer_kernel};
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        const CoreWork& cw = core_work[core_id];
        const uint32_t group_id = cw.reduction_group_id;

        const uint32_t num_workers = cw.has_work ? (uint32_t)worker_core_ids[group_id].size() : 0u;
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"c_in_block_start", cw.c_in_block_start},
             {"c_in_block_end", cw.c_in_block_end},
             {"c_out_block_start", cw.c_out_block_start},
             {"c_out_block_end", cw.c_out_block_end},
             {"t_out_start", cw.t_out_start},
             {"t_out_end", cw.t_out_end},
             {"h_out_start", cw.h_out_start},
             {"h_out_end", cw.h_out_end},
             {"w_out_start", cw.w_out_start},
             {"w_out_end", cw.w_out_end}});
        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            core,
            {{"c_in_block_start", cw.c_in_block_start},
             {"c_in_block_end", cw.c_in_block_end},
             {"c_out_block_start", cw.c_out_block_start},
             {"c_out_block_end", cw.c_out_block_end},
             {"t_out_start", cw.t_out_start},
             {"t_out_end", cw.t_out_end},
             {"h_out_start", cw.h_out_start},
             {"h_out_end", cw.h_out_end},
             {"w_out_start", cw.w_out_start},
             {"w_out_end", cw.w_out_end},
             {"is_reducer", static_cast<uint32_t>(cw.is_reducer)},
             {"num_workers", num_workers}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"c_in_block_start", cw.c_in_block_start},
             {"c_in_block_end", cw.c_in_block_end},
             {"c_out_block_start", cw.c_out_block_start},
             {"c_out_block_end", cw.c_out_block_end},
             {"t_out_start", cw.t_out_start},
             {"t_out_end", cw.t_out_end},
             {"h_out_start", cw.h_out_start},
             {"h_out_end", cw.h_out_end},
             {"w_out_start", cw.w_out_start},
             {"w_out_end", cw.w_out_end},
             {"is_reducer", static_cast<uint32_t>(cw.is_reducer)},
             {"weight_share_role_value", static_cast<uint32_t>(cw.weight_share_role)},
             {"weight_src_noc_x", cw.weight_src_noc_x},
             {"weight_src_noc_y", cw.weight_src_noc_y},
             {"chain_succ_noc_x", cw.chain_succ_noc_x},
             {"chain_succ_noc_y", cw.chain_succ_noc_y},
             {"mcast_bbox_start_x", cw.mcast_bbox_start_x},
             {"mcast_bbox_start_y", cw.mcast_bbox_start_y},
             {"mcast_bbox_end_x", cw.mcast_bbox_end_x},
             {"mcast_bbox_end_y", cw.mcast_bbox_end_y},
             {"mcast_num_dests", cw.mcast_num_dests},
             {"mcast_num_iters", cw.mcast_num_iters},
             {"num_workers", num_workers}});

        if (num_worker_coord_varargs > 0) {
            auto& varargs = writer_run.advanced_options.runtime_varargs[core];
            varargs.assign(num_worker_coord_varargs, 0u);
            if (num_workers > 0) {
                varargs[0] = reducer_core_physical_xs[group_id];
                varargs[1] = reducer_core_physical_ys[group_id];
                for (uint32_t worker = 0; worker < num_workers; ++worker) {
                    varargs[2 + worker] = worker_core_physical_xs[group_id][worker];
                    varargs[2 + num_workers + worker] = worker_core_physical_ys[group_id][worker];
                }
            }
        }

        log_debug(
            tt::LogOp,
            "Core ({},{}): HasWork={}, IsReducer={}, ChainRole={}, "
            "ReductionGroup={}, C_in_idx={}, C_out_idx={}, NumWorkers={}",
            core.x,
            core.y,
            cw.has_work,
            cw.is_reducer,
            static_cast<uint32_t>(cw.weight_share_role),
            group_id,
            cw.c_in_idx,
            cw.c_out_idx,
            num_workers);
    }

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::TensorParameter> tensor_parameters = {
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = input_tensor_parameter, .spec = input_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = output_tensor_parameter, .spec = output_tensor.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{
            .unique_id = weight_tensor_parameter, .spec = weight_tensor.tensor_spec()},
    };
    if (use_bias) {
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = bias_tensor_parameter, .spec = bias_tensor->tensor_spec()});
    }
    if (halo_mode) {
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = halo_tensor_parameter, .spec = tensor_args.halo_buffer->tensor_spec()});
    }
    if (mask_mode) {
        tensor_parameters.push_back(tt::tt_metal::experimental::TensorParameter{
            .unique_id = pad_offset_tensor_parameter, .spec = tensor_args.pad_offset_tensor->tensor_spec()});
    }

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "conv3d",
        .kernels = {std::move(reader_spec), std::move(compute_spec), std::move(writer_spec)},
        .dataflow_buffers = std::move(dfbs),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {tt::tt_metal::experimental::WorkUnitSpec{
            .name = "conv3d",
            .kernels = {reader_kernel, compute_kernel, writer_kernel},
            .target_nodes = tt::tt_metal::CoreRangeSet(core_grid)}},
    };
    tt::tt_metal::experimental::ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run), std::move(compute_run), std::move(writer_run)},
        .tensor_args =
            {{input_tensor_parameter,
              tt::tt_metal::experimental::TensorArgument{std::cref(input_tensor.mesh_tensor())}},
             {output_tensor_parameter,
              tt::tt_metal::experimental::TensorArgument{std::cref(output_tensor.mesh_tensor())}},
             {weight_tensor_parameter,
              tt::tt_metal::experimental::TensorArgument{std::cref(weight_tensor.mesh_tensor())}}},
    };
    if (use_bias) {
        run_args.tensor_args.insert(
            {bias_tensor_parameter, tt::tt_metal::experimental::TensorArgument{std::cref(bias_tensor->mesh_tensor())}});
    }
    if (halo_mode) {
        run_args.tensor_args.insert(
            {halo_tensor_parameter,
             tt::tt_metal::experimental::TensorArgument{std::cref(tensor_args.halo_buffer->mesh_tensor())}});
    }
    if (mask_mode) {
        run_args.tensor_args.insert(
            {pad_offset_tensor_parameter,
             tt::tt_metal::experimental::TensorArgument{std::cref(tensor_args.pad_offset_tensor->mesh_tensor())}});
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
