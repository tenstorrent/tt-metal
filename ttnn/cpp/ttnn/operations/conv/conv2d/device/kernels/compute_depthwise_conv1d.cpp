// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// Compute one block (one kernel-tap slice) of a 1D depthwise conv.
//
// Per output tile:
//   dst[0]  = in0 * in1                                                 (FPU mul)
//   if idx > 0: dst[0] += prior partial loaded from scratch_dfb          (FPU add via DST reuse)
//   pack dst[0] -> out_dfb on the last tap, else scratch_dfb
//
// `idx` is the kernel-tap block index (0 .. num_taps-1, num_taps == filter_h*filter_w). The very
// first call (idx == 0) initializes the partial with the tap-0 product; subsequent calls accumulate
// via the DST_TO_SRCB dest-reuse pattern, which keeps the running partial in DST and only pulls the
// prior partial from L1. This gives a single pack per output tile and avoids the pack-format flips
// that corrupt block-float (BFLOAT8_B/BFLOAT4_B) outputs in the round-tripped variant — while
// still using FPU (not SFPU) for the add.
//
// The partial lives in scratch_dfb; only the last tap (idx == num_taps-1) packs into out_dfb. The host
// aliases scratch_dfb to out_dfb for a single height block (in-place), but uses a separate buffer when
// in0_num_blocks_h > 1, where out_dfb (the persistent sharded output) cannot double as the read-back
// scratch — block N would otherwise read back block N-1's already-written output.
//
// srcB (cfg92) tile descriptor: must match in1 for the mul, and is repopulated from DST for the
// dest-reuse add. We force srcB back to in1's format every iteration so block-float weights are
// decoded correctly.
inline void mul_and_accumulate_block(
    DataflowBuffer in0_dfb,
    DataflowBuffer in1_dfb,
    DataflowBuffer scratch_dfb,
    DataflowBuffer out_dfb,
    uint32_t block_num_tiles,
    uint32_t idx,
    uint32_t num_taps) {
    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t scratch_cb_id = scratch_dfb.get_id();
    // The last tap writes the finished output to out_dfb; earlier taps write the partial to scratch_dfb.
    const bool is_last_tap = (idx + 1 == num_taps);
    DataflowBuffer dst_dfb = is_last_tap ? out_dfb : scratch_dfb;
    const uint32_t dst_cb_id = dst_dfb.get_id();

    for (uint32_t i = 0; i < block_num_tiles; i++) {
        in1_dfb.wait_front(1);
        in0_dfb.wait_front(1);

        tile_regs_acquire();
        // mul: srcA = in0 (bf16), srcB = in1 (bf8/bf16) -> dst[0]
        reconfig_data_format_srcb(in1_cb_id);
        mul_init(in0_cb_id, in1_cb_id);
        mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);

        if (idx != 0) {
            // dest-reuse add: dst[0] += scratch_dfb (the prior tap's partial). srcA gets scratch_dfb
            // (cfg52 must match its format); srcB is filled from dst[0] by the dest-reuse path.
            reconfig_data_format_srca(scratch_cb_id);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                scratch_cb_id);
            scratch_dfb.wait_front(1);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                scratch_cb_id, 0, 0);
            scratch_dfb.pop_front(1);

            // Restore srcA to in0's format for the next iteration's mul unpack.
            reconfig_data_format_srca(in0_cb_id);
        }
        tile_regs_commit();

        // scratch_dfb and out_dfb share the output data format, so packing to either target needs no
        // pack reconfig.
        dst_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dst_cb_id);
        dst_dfb.push_back(1);
        tile_regs_release();

        in0_dfb.pop_front(1);
        in1_dfb.pop_front(1);
    }
}

// SFPU form of `mul_and_accumulate_block`, for genuinely fp32 operands.
//
// SrcA/SrcB truncate fp32 to ~9/~13 significand bits regardless of MathFidelity, which is why the
// FPU form above measures ~1.6e-03 against a float64 golden. `copy_tile` + `mul_binary_tile`/
// `add_binary_tile` keep everything in DST instead, avoiding that truncation on both the multiply
// and the partial reload. Needs `unpack_to_dest_mode = UnpackToDestFp32` on ACT/ACT_TILIZED/WEIGHTS/
// scratch (set in build_program_descriptor_sharded) or the tiles are already truncated before this
// runs. DST usage: 2 tiles (running value + staged operand).
inline void mul_and_accumulate_block_sfpu(
    DataflowBuffer in0_dfb,
    DataflowBuffer in1_dfb,
    DataflowBuffer scratch_dfb,
    DataflowBuffer out_dfb,
    uint32_t block_num_tiles,
    uint32_t idx,
    uint32_t num_taps) {
    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t scratch_cb_id = scratch_dfb.get_id();
    const bool is_last_tap = (idx + 1 == num_taps);
    DataflowBuffer dst_dfb = is_last_tap ? out_dfb : scratch_dfb;
    const uint32_t dst_cb_id = dst_dfb.get_id();

    // DST slots: 0 holds the running value, 1 stages the incoming operand.
    constexpr uint32_t DST_ACC = 0;
    constexpr uint32_t DST_OPERAND = 1;

    for (uint32_t i = 0; i < block_num_tiles; i++) {
        in1_dfb.wait_front(1);
        in0_dfb.wait_front(1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(in0_cb_id);
        copy_tile(in0_cb_id, 0, DST_ACC);
        copy_tile_to_dst_init_short(in1_cb_id);
        copy_tile(in1_cb_id, 0, DST_OPERAND);
        mul_binary_tile_init();
        mul_binary_tile(DST_ACC, DST_OPERAND, DST_ACC);

        if (idx != 0) {
            scratch_dfb.wait_front(1);
            copy_tile_to_dst_init_short(scratch_cb_id);
            copy_tile(scratch_cb_id, 0, DST_OPERAND);
            add_binary_tile_init();
            add_binary_tile(DST_ACC, DST_OPERAND, DST_ACC);
            scratch_dfb.pop_front(1);
        }
        tile_regs_commit();

        dst_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(DST_ACC, dst_cb_id);
        dst_dfb.push_back(1);
        tile_regs_release();

        in0_dfb.pop_front(1);
        in1_dfb.pop_front(1);
    }
}

// SFPU form of `mul_and_accumulate_coalesced_block`, for genuinely fp32 operands. See
// `mul_and_accumulate_block_sfpu` for why: same multiply-truncation defect, same fix.
template <uint32_t in0_block_w, uint32_t kernel_width, uint32_t block_num_tiles>
inline void mul_and_accumulate_coalesced_block_sfpu(
    DataflowBuffer in0_dfb, DataflowBuffer in1_dfb, DataflowBuffer out_dfb) {
    static_assert(kernel_width > 1);
    static_assert(in0_block_w % kernel_width == 0);
    static_assert(block_num_tiles % in0_block_w == 0);

    constexpr uint32_t in_channels_ntiles = in0_block_w / kernel_width;
    constexpr uint32_t act_block_h_ntiles = block_num_tiles / in0_block_w;

    // DST slots: 0 accumulates, 1 and 2 hold the tap's activation and weight.
    constexpr uint32_t DST_ACC = 0;
    constexpr uint32_t DST_ACT = 1;
    constexpr uint32_t DST_WEIGHT = 2;

    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t out_cb_id = out_dfb.get_id();

    in0_dfb.wait_front(block_num_tiles);
    in1_dfb.wait_front(block_num_tiles);

    for (uint32_t h = 0; h < act_block_h_ntiles; ++h) {
        for (uint32_t c = 0; c < in_channels_ntiles; ++c) {
            tile_regs_acquire();

            for (uint32_t tap = 0; tap < kernel_width; ++tap) {
                const uint32_t act_tile_idx = h * in0_block_w + tap * in_channels_ntiles + c;
                const uint32_t weight_tile_idx =
                    tap * act_block_h_ntiles * in_channels_ntiles + h * in_channels_ntiles + c;

                copy_tile_to_dst_init_short(in0_cb_id);
                copy_tile(in0_cb_id, act_tile_idx, DST_ACT);
                copy_tile_to_dst_init_short(in1_cb_id);
                copy_tile(in1_cb_id, weight_tile_idx, DST_WEIGHT);

                // The first tap seeds the accumulator directly; later taps multiply into the
                // scratch slot and add.
                mul_binary_tile_init();
                mul_binary_tile(DST_ACT, DST_WEIGHT, tap == 0 ? DST_ACC : DST_ACT);
                if (tap != 0) {
                    add_binary_tile_init();
                    add_binary_tile(DST_ACC, DST_ACT, DST_ACC);
                }
            }
            tile_regs_commit();

            out_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(DST_ACC, out_cb_id);
            out_dfb.push_back(1);
            tile_regs_release();
        }
    }

    in0_dfb.pop_front(block_num_tiles);
    in1_dfb.pop_front(block_num_tiles);
}

template <uint32_t in0_block_w, uint32_t kernel_width, uint32_t block_num_tiles>
inline void mul_and_accumulate_coalesced_block(DataflowBuffer in0_dfb, DataflowBuffer in1_dfb, DataflowBuffer out_dfb) {
    static_assert(kernel_width > 1);
    static_assert(in0_block_w % kernel_width == 0);
    static_assert(block_num_tiles % in0_block_w == 0);

    constexpr uint32_t in_channels_ntiles = in0_block_w / kernel_width;
    constexpr uint32_t act_block_h_ntiles = block_num_tiles / in0_block_w;

    const uint32_t in0_cb_id = in0_dfb.get_id();
    const uint32_t in1_cb_id = in1_dfb.get_id();
    const uint32_t out_cb_id = out_dfb.get_id();

    in0_dfb.wait_front(block_num_tiles);
    in1_dfb.wait_front(block_num_tiles);

    for (uint32_t h = 0; h < act_block_h_ntiles; ++h) {
        for (uint32_t c = 0; c < in_channels_ntiles; ++c) {
            tile_regs_acquire();
            reconfig_data_format_srca(in0_cb_id);
            reconfig_data_format_srcb(in1_cb_id);

            for (uint32_t tap = 0; tap < kernel_width; ++tap) {
                const uint32_t act_tile_idx = h * in0_block_w + tap * in_channels_ntiles + c;
                const uint32_t weight_tile_idx =
                    tap * act_block_h_ntiles * in_channels_ntiles + h * in_channels_ntiles + c;
                mul_init(in0_cb_id, in1_cb_id, tap != 0 ? 1U : 0U, __builtin_LINE());
                mul_tiles(in0_cb_id, in1_cb_id, act_tile_idx, weight_tile_idx, 0);
            }
            tile_regs_commit();

            out_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, out_cb_id);
            out_dfb.push_back(1);
            tile_regs_release();
        }
    }

    in0_dfb.pop_front(block_num_tiles);
    in1_dfb.pop_front(block_num_tiles);
}

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_num_blocks_h = get_compile_time_arg_val(3);
    constexpr uint32_t in0_num_blocks_w = get_compile_time_arg_val(4);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t kernel_width = get_compile_time_arg_val(9);
    constexpr bool coalesce_kw_reads = get_compile_time_arg_val(10) == 1;
    // Read-back scratch for the dest-reuse accumulation. The host points this at out_dfb for a single
    // height block (in-place) or at a dedicated scratch CB for multiple blocks.
    constexpr uint32_t partials_cb_id = get_compile_time_arg_val(11);

    // SFPU tap accumulation, selected by the host (use_fp32_sfpu_depthwise in the factory). The same
    // host condition delivers ACT/ACT_TILIZED/WEIGHTS/scratch UnpackToDestFp32, so dispatch and CB
    // modes can never disagree; when false (any non-fp32 operand or output), every caller stays
    // bit-exact on the stock FPU path.
    constexpr bool fp32_operands = get_compile_time_arg_val(12) == 1;

    DataflowBuffer dfb_tilized_in0(tilized_in0_cb_id);
    DataflowBuffer dfb_in1(in1_cb_id);
    DataflowBuffer dfb_out(out_cb_id);
    DataflowBuffer dfb_partials(partials_cb_id);

    // compute_kernel_hw_startup configures pack for out_dfb, math for in0/in1, and unpack for in0/in1.
    // The pack target never changes (we only ever pack to out_dfb), so no further pack reconfig is
    // needed for the lifetime of the kernel.
    compute_kernel_hw_startup(in0_cb_id, in1_cb_id, out_cb_id);

    for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
        for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
            // Tilize the full activation block height. The number of tile-rows is
            // in0_block_num_tiles / in0_block_w (== act_block_h_ntiles); this must match the tile
            // count mul_and_accumulate_block(_coalesced) consumes below. Using in0_num_subblocks
            // here under-produces by out_subblock_h_ntiles when it is > 1, deadlocking the CB.
            compute_kernel_lib::tilize<in0_block_w, in0_cb_id, tilized_in0_cb_id>(in0_block_num_tiles / in0_block_w);
            reconfig_data_format_srca(tilized_in0_cb_id);
            if constexpr (coalesce_kw_reads && fp32_operands) {
                mul_and_accumulate_coalesced_block_sfpu<in0_block_w, kernel_width, in0_block_num_tiles>(
                    dfb_tilized_in0, dfb_in1, dfb_out);
            } else if constexpr (coalesce_kw_reads) {
                mul_and_accumulate_coalesced_block<in0_block_w, kernel_width, in0_block_num_tiles>(
                    dfb_tilized_in0, dfb_in1, dfb_out);
            } else if constexpr (fp32_operands) {
                mul_and_accumulate_block_sfpu(
                    dfb_tilized_in0,
                    dfb_in1,
                    dfb_partials,
                    dfb_out,
                    in0_block_num_tiles,
                    in0_block_w_i,
                    in0_num_blocks_w);
            } else {
                // Accumulate kernel-tap in0_block_w_i of in0_num_blocks_w through dfb_partials, writing
                // the final tap to dfb_out. The host points dfb_partials at dfb_out for a single height
                // block (in-place, no extra buffer) or at a dedicated scratch CB for multiple blocks.
                mul_and_accumulate_block(
                    dfb_tilized_in0,
                    dfb_in1,
                    dfb_partials,
                    dfb_out,
                    in0_block_num_tiles,
                    in0_block_w_i,
                    in0_num_blocks_w);
            }
        }
    }
}
