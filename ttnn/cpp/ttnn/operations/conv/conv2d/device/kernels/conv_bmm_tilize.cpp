// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "internal/mod_div_lib.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// #include "api/debug/dprint.h"

#define DEBUG_PRINT 0

#ifdef SPLIT_READER
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true>
__attribute__((noinline)) void tilize_in(
#else
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true>
void tilize_in(
#endif
    uint32_t in_num_subblocks) {
    constexpr compute_kernel_lib::tilize_config::InitUninitMode init_uninit_mode =
        init_tilize ? (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit
                                     : compute_kernel_lib::tilize_config::InitUninitMode::InitOnly)
                    : (uninit_tilize ? compute_kernel_lib::tilize_config::InitUninitMode::UninitOnly
                                     : compute_kernel_lib::tilize_config::InitUninitMode::Neither);
    // Split-reader fires tilize_in twice back-to-back (first: init=true+uninit=false,
    // second: init=false+uninit=true). The second call must NOT reconfig datatypes —
    // doing so clobbers the bf16 SrcA override that fast_tilize_init installs for
    // fp32 input on BH (see _llk_unpack_fast_tilize_init_), breaking the second
    // tilize's MOP. Only reconfig on the init call; continuation reuses that state.
    constexpr auto reconfig_mode =
        init_tilize ? compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure
                    : compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure;
    compute_kernel_lib::tilize<
        in_block_w,
        in_cb_id,
        out_cb_id,
        init_uninit_mode,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        reconfig_mode>(in_num_subblocks);
}  // tilize_in()

template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id>
inline void tilize_single_block(experimental::CB in_cb) {
    in_cb.wait_front(in_block_w);
    fast_tilize_block(in_cb_id, in_block_w, out_cb_id);
    in_cb.pop_front(in_block_w);
}

template <uint32_t in_cb_id, uint32_t window_reuse_offset>
inline uint32_t update_in_cb(uint32_t in_cb_addr) {
    UNPACK((get_local_cb_interface(in_cb_id).fifo_rd_ptr = in_cb_addr));
    return in_cb_addr + window_reuse_offset;
}

template <uint32_t in_cb_id, uint32_t in_block_w, uint32_t out_cb_id, uint32_t tilized_cb_row_offset>
inline void tilize_single_block_with_out_cb_update(experimental::CB in_cb, uint32_t& out_cb_addr) {
    PACK((get_local_cb_interface(out_cb_id).fifo_wr_ptr = out_cb_addr));
    PACK((out_cb_addr += tilized_cb_row_offset));
    tilize_single_block<in_cb_id, in_block_w, out_cb_id>(in_cb);
}

template <
    uint32_t in1_cb_id,
    uint32_t in2_cb_id,
    uint32_t in_block_w,
    uint32_t in1_num_subblocks,
    uint32_t in2_num_subblocks,
    uint32_t out_cb_id,
    uint32_t out_cb_tiles,
    uint32_t window_reuse_offset,
    uint32_t tilized_cb_row_offset,
    uint32_t tilized_cb_second_reader_offset,
    uint32_t image_width_in_tiles>
inline void tilize_in_reuse_split_reader(
    experimental::CB in1_cb,
    experimental::CB in2_cb,
    experimental::CB out_cb,
    uint32_t act_cb_start_address,
    uint32_t act_cb_second_reader_start_address) {
    // with activation reuse, the activation buffers are sized to fit one output image width only,
    // so we need to interleave waits and pops on the two buffers to allow parallelization;
    // we reserve back tilized CB to store whole act block h - and then we update write pointers so that
    // we fill in first row of the first half (NCRISC), first row of the second half (BRISC) and so on
    out_cb.reserve_back(out_cb_tiles);
    fast_tilize_init_with_dt(in1_cb_id, in_block_w, out_cb_id);

    uint32_t in1_cb_addr = act_cb_start_address;
    uint32_t in2_cb_addr = act_cb_second_reader_start_address;

    uint32_t out_cb_addr, out_cb_addr_second_reader, out_cb_addr_init;
    PACK((out_cb_addr_init = get_local_cb_interface(out_cb_id).fifo_wr_ptr));
    PACK((out_cb_addr = out_cb_addr_init));
    PACK((out_cb_addr_second_reader = out_cb_addr_init + tilized_cb_second_reader_offset));

    constexpr uint32_t min_num_subblocks =
        in1_num_subblocks > in2_num_subblocks ? in2_num_subblocks : in1_num_subblocks;
    constexpr uint32_t min_num_image_rows = min_num_subblocks / image_width_in_tiles;
    constexpr uint32_t leftover_in1 = in1_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t leftover_in2 = in2_num_subblocks - min_num_image_rows * image_width_in_tiles;
    constexpr uint32_t max_leftover = leftover_in1 > leftover_in2 ? leftover_in1 : leftover_in2;

    // process minimum number of image rows for both readers in the same loop
    for (uint32_t image_row = 0; image_row < min_num_image_rows; ++image_row) {
        // each time we start processing a new image row, we need to update the read pointer to the appropriate offset
        // from the start of the CB to match the reader behavior
        in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
        in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
        for (uint32_t image_col = 0; image_col < image_width_in_tiles; ++image_col) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in1_cb, out_cb_addr);
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in2_cb, out_cb_addr_second_reader);
        }
    }

    // leftover image rows if one reader had more rows than the other
    in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
    in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
    for (uint32_t image_col = 0; image_col < max_leftover; ++image_col) {
        if (image_col < leftover_in1) {
            tilize_single_block_with_out_cb_update<in1_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in1_cb, out_cb_addr);

            if (image_col == image_width_in_tiles - 1) {
                in1_cb_addr = update_in_cb<in1_cb_id, window_reuse_offset>(in1_cb_addr);
            }
        }

        if (image_col < leftover_in2) {
            tilize_single_block_with_out_cb_update<in2_cb_id, in_block_w, out_cb_id, tilized_cb_row_offset>(
                in2_cb, out_cb_addr_second_reader);

            if (image_col == image_width_in_tiles - 1) {
                in2_cb_addr = update_in_cb<in2_cb_id, window_reuse_offset>(in2_cb_addr);
            }
        }
    }

    // Restore fifo_wr_ptr to the reserved-region base so push_back advances from a
    // known starting point. Without this, push_back's fifo_wr_ptr += num_words
    // starts from whichever mid-region offset the last tilize_single_block left
    // and trips the LLK bounds assert (see GH #42510).
    PACK((get_local_cb_interface(out_cb_id).fifo_wr_ptr = out_cb_addr_init));
    out_cb.push_back(out_cb_tiles);
    fast_tilize_uninit(in2_cb_id, out_cb_id, in_block_w);
}

// Tilize phase as a PreKBlockFn for compute_kernel_lib::matmul_block. Captures the
// conv2d per-K-block tilize-then-srcA-reconfig pattern. Compile-time constants are
// hoisted into template params; runtime addresses for activation_reuse live in fields.
template <
    bool height_sharded,
    bool packer_l1_acc,
    bool pack_relu,
    bool fuse_bias,
    bool split_reader,
    bool split_reader_cb_shared,
    bool activation_reuse,
    uint32_t in0_block_w_,
    uint32_t in0_cb_id_,
    uint32_t in0_pretilize_cb_id_,
    uint32_t in0_cb_second_reader_id_,
    uint32_t tilized_in0_cb_id_,
    uint32_t in1_cb_id_,
    uint32_t matmul_partials_cb_,
    uint32_t in0_num_subblocks_read_,
    uint32_t in0_num_subblocks_read_last_,
    uint32_t in0_nblocks_w_tilize_,
    uint32_t out_subblock_w_,
    uint32_t out_subblock_h_,
    uint32_t image_width_in_tiles_,
    uint32_t window_reuse_offset_,
    uint32_t tilized_cb_row_offset_,
    uint32_t tilized_cb_second_reader_offset_>
struct ConvTilizePreKBlock {
    experimental::CB cb_in0;
    experimental::CB cb_in0_second_reader;
    experimental::CB cb_tilized_in0;
    uint32_t act_cb_start_address;
    uint32_t act_cb_second_reader_start_address;
    uint32_t tilized_cb_start_address;

    ALWI void operator()(uint32_t in0_block_w_i, uint32_t /*num_k_blocks*/, bool last_inner_dim_block) const {
        if constexpr (!height_sharded) {
            // Block-sharded: tilize only every in0_nblocks_w_tilize K-blocks (one tilize result
            // feeds multiple K-block iterations when conv_act_c_blocks > 1).
            if (in0_block_w_i % in0_nblocks_w_tilize_ == 0) {
                if constexpr (pack_relu && !fuse_bias) {
                    if (last_inner_dim_block) {
                        // Disable RELU for the tilize. The matmul helper re-enables on the
                        // actual last K-block via LastBlockTarget::OutWithRelu.
                        PACK((llk_pack_relu_config(ReluConfig::none())));
                    }
                }
                if constexpr (packer_l1_acc) {
                    pack_reconfig_data_format(matmul_partials_cb_, tilized_in0_cb_id_);
                    pack_reconfig_l1_acc(0);
                }
                tilize_in<
                    in0_block_w_,
                    in0_pretilize_cb_id_,
                    tilized_in0_cb_id_,
                    true,
                    !split_reader || split_reader_cb_shared>(in0_num_subblocks_read_);

                if constexpr (split_reader && !split_reader_cb_shared) {
                    tilize_in<in0_block_w_, in0_cb_second_reader_id_, tilized_in0_cb_id_, false, true>(
                        in0_num_subblocks_read_last_);
                }
                // Matmul-state restore is now the helper's job: matmul_block is invoked with
                // InitMode::ShortAfterPreKBlock, so it reconfigs srcA/srcB and re-issues
                // mm_block_init_short itself after this functor returns. This functor only
                // tilizes (plus the relu / packer-l1_acc config above, which it owns).
            }
        } else {
            // Height-sharded: tilize every K-block.
            if constexpr (pack_relu && !fuse_bias) {
                if (last_inner_dim_block) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
            }
            if constexpr (packer_l1_acc) {
                pack_reconfig_data_format(matmul_partials_cb_, tilized_in0_cb_id_);
                pack_reconfig_l1_acc(0);
            }

            if constexpr (!activation_reuse) {
                tilize_in<in0_block_w_, in0_cb_id_, tilized_in0_cb_id_, true, !split_reader>(in0_num_subblocks_read_);
            }

            if constexpr (split_reader) {
                if constexpr (!activation_reuse) {
                    tilize_in<in0_block_w_, in0_cb_second_reader_id_, tilized_in0_cb_id_, false, true>(
                        in0_num_subblocks_read_last_);
                } else {
                    PACK((get_local_cb_interface(tilized_in0_cb_id_).fifo_wr_ptr = tilized_cb_start_address));
                    tilize_in_reuse_split_reader<
                        in0_cb_id_,
                        in0_cb_second_reader_id_,
                        in0_block_w_,
                        in0_num_subblocks_read_,
                        in0_num_subblocks_read_last_,
                        tilized_in0_cb_id_,
                        in0_block_w_*(in0_num_subblocks_read_ + in0_num_subblocks_read_last_),
                        window_reuse_offset_,
                        tilized_cb_row_offset_,
                        tilized_cb_second_reader_offset_,
                        image_width_in_tiles_>(
                        cb_in0,
                        cb_in0_second_reader,
                        cb_tilized_in0,
                        act_cb_start_address,
                        act_cb_second_reader_start_address);
                }
            }

            // Matmul-state restore is now the helper's job (InitMode::ShortAfterPreKBlock) —
            // see the block-sharded branch above. This functor only tilizes.
        }
    }
};

#ifdef SFPU_OP_INIT_ACTIVATION
// SFPU activation applied per output sub-block on the last K-block, before packing.
// Used as the matmul_block PostComputeFn in the no-bias path.
struct ConvSFPUPostCompute {
    ALWI void operator()(uint32_t num_tiles) const {
        for (uint32_t i = 0; i < num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
#endif

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    [[maybe_unused]] constexpr uint32_t in0_subblock_num_tiles =
        get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t reader_num_h_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(5);  // outer column block size (in inner column blocks)
    [[maybe_unused]] constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(6);                               // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(7);  // out_subblock_w*in1_num_subblocks
    // if these are not defined as volatile, it causes code size for TRISC2 to be too large if num_blocks > 1
    constexpr uint32_t in0_num_blocks_h = get_compile_time_arg_val(8);
    constexpr uint32_t in0_num_blocks_w = get_compile_time_arg_val(9);
    constexpr uint32_t in1_num_blocks_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(11);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(13);  // out_subblock_h * out_subblock_w;
    constexpr bool height_sharded = get_compile_time_arg_val(14);
    constexpr bool untilize_out = get_compile_time_arg_val(15);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in0_pretilize_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t in0_cb_second_reader_id = get_compile_time_arg_val(21);
    constexpr uint32_t matmul_partials_cb = get_compile_time_arg_val(22);
    constexpr uint32_t tilized_in0_cb_id = get_compile_time_arg_val(23);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(24);
    // Always false now (factory dedicates matmul_partials_cb); kept to hold compile-arg slot 25.
    [[maybe_unused]] constexpr bool partials_cb_uses_output = get_compile_time_arg_val(25);
    constexpr uint32_t in0_nblocks_w_tilize = get_compile_time_arg_val(26);
    constexpr bool check_skip_compute = get_compile_time_arg_val(27);
    constexpr bool pack_relu = get_compile_time_arg_val(28);
    constexpr bool packer_untilize = get_compile_time_arg_val(29);
    constexpr bool packer_l1_acc = get_compile_time_arg_val(30);
    constexpr bool fuse_bias = get_compile_time_arg_val(31);
    constexpr bool split_reader = get_compile_time_arg_val(32);
    constexpr bool activation_reuse = get_compile_time_arg_val(33);

    constexpr uint32_t image_width_in_tiles = get_compile_time_arg_val(34);
    constexpr uint32_t window_reuse_offset = get_compile_time_arg_val(35);
    constexpr uint32_t tilized_cb_row_offset = get_compile_time_arg_val(36);
    constexpr uint32_t tilized_cb_second_reader_offset = get_compile_time_arg_val(37);
    constexpr bool split_reader_cb_shared = get_compile_time_arg_val(38) == 1;

    // "Dedicate partials, match matmul" (GH#45995): the factory now sizes matmul_partials_cb to ONE
    // output block and gives it its own L1 region (is_globally_allocated=false → partials_cb_uses_output
    // is always false here). With a one-block dedicated region the helper's NON-pin FIFO wraps back to
    // the same base every K-block, so packer_l1_acc (and the software spill/reload when l1_acc is off)
    // operates at a fixed L1 base for multi-output-block convs too — exactly how matmul's dedicated
    // single-block interm0 behaves. There is NO pin anywhere on this path.
    //
    // M2 — re-enable TileRowMajor (TRM) on the NON-pin dedicated-partials base. The factory
    // (conv2d_op_sharded_program_factory.cpp) emits CONV_TILE_PACK_ROW_MAJOR for eligible
    // HEIGHT_SHARDED no-bias convs whose per_core_N is stranded by the SubblockMajor
    // out_subblock_w == per_core_N constraint, and folds a LARGER (relaxed) out_subblock into the
    // compile args. TRM lifts that constraint, so a relaxed subblock with out_subblock_h > 1 (fewer
    // matmul_block_init / pack passes per output block) becomes legal. With the define:
    //   • the matmul helper packs the LAST K-block row-major (pack_subblock_row_strided) instead of
    //     subblock-major, on the SAME non-pin dedicated-partials FIFO. l1_acc ON accumulates per-address
    //     across K-blocks (spill_row_grouped=true: non-last spills land row-strided, last-block reload
    //     gathers via copy_subblock_row_strided). l1_acc OFF software-spills/reloads: non-last spills are
    //     subblock-major and the reload is contiguous (copy_block_matmul_partials), but the LAST block's
    //     reserve_back lands one M-row-group on top of a still-fronted full block of spills, so the factory
    //     sizes matmul_partials_cb to 2*out_block_num_tiles for l1_acc-OFF (M3) — that headroom is what
    //     keeps the ROW_MAJOR/untilize Interm-target path from self-deadlocking on reserve_back.
    //   • the untilize phase (ROW_MAJOR output) reads the row-major interm via plain `untilize` (the
    //     row strip is already contiguous tile-row order), NOT reblock_and_untilize (SubblockMajor only).
    //   • TILE output packs the last K-block straight to out_cb in row-major tile order (== the tiled
    //     shard layout), no reblock.
    // Without the define this is byte-identical to the SBM path. TRM + fuse_bias degrades to
    // SubblockMajor (TileRowMajor + bias deadlocks the shared partials CB); the factory's eligibility
    // forbids bias on this path, so the degrade is defensive and never reached in production.
#ifdef CONV_TILE_PACK_ROW_MAJOR
    constexpr bool tile_pack_row_major = true;
#else
    constexpr bool tile_pack_row_major = false;
#endif
    // PoC (TT_CONV_TRM_CALLER_OWNS, via CONV_CALLER_OWNS_PACK_TARGET define): switch the matmul interm
    // pack onto the caller_owns_pack_target contract — the caller does ONE reserve_back before the
    // matmul and ONE push_back after, and the helper skips its own per-block reserve/push/drain. Under
    // this flag the TileRowMajor layout is also used WITH fuse_bias (see conv_output_layout below): the
    // bias-add reads the dedicated partials and writes a DISTINCT OUT buffer, so the TileRowMajor bias
    // path balances the single caller reserve/push.
#ifdef CONV_CALLER_OWNS_PACK_TARGET
    constexpr bool caller_owns_pack_target = true;
#else
    constexpr bool caller_owns_pack_target = false;
#endif
    // Production TRM degrades to SubblockMajor when fuse_bias (TRM+bias deadlocks the helper-owned shared
    // partials CB). The caller_owns path lifts that degrade: partials is dedicated and the caller owns the
    // single reserve/push, so TileRowMajor + bias is safe.
    constexpr auto conv_output_layout =
        (tile_pack_row_major && (caller_owns_pack_target || !fuse_bias))
            ? compute_kernel_lib::OutputCBLayout::TileRowMajor
            : compute_kernel_lib::OutputCBLayout::SubblockMajor;

    // One full output block in tiles = per_core_M (act_block_h_ntiles = in0_num_subblocks*out_subblock_h)
    // × per_core_N (in1_block_w = in1_num_subblocks*out_subblock_w). The caller_owns path reserves/pushes
    // exactly this many tiles on the dedicated partials CB once per outer iter (= L4a's 4×2 = 8).
    constexpr uint32_t out_block_num_tiles = (in0_num_subblocks * out_subblock_h) * in1_block_w;

    // ── PIN (legacy perf feature being eliminated; GH#45995) ──────────────────────────────────────
    // pin_interm_to_captured_base recovered the packer_l1_acc per-K-block DRAIN-SKIP win by reserving
    // the whole out_block ONCE at K-loop entry, packing each K-block's subblock partials to FIXED
    // offsets (L1_ACC integrates per-address), skipping the per-K-block interm reserve/push/wait/pop,
    // and pushing once at exit. The matmul-helper caller_owns_pack_target contract recovers the EXACT
    // same drain-skip without the bespoke pin machinery (one caller reserve_back before the K-loop +
    // one push_back after; the helper skips its own per-block reserve/push/drain).
    //
    // DROP-PIN (step a + a′): conv no longer selects pin for ANY conv. The factory routes the deep-K
    // packer_l1_acc INTERM-target classes (fuse_bias [step a] AND untilize_out [step a′]) through
    // caller_owns_pack_target + TileRowMajor instead — those convs are SBM (in1_num_subblocks==1) so
    // TileRowMajor is BYTE-IDENTICAL to SubblockMajor, and caller_owns forces TileRowMajor →
    // conv_output_layout != SubblockMajor → conv_pin computes false. The only remaining pin trigger was
    // the OUT target (no-bias, no-untilize, DEDICATED multi-output-block partials); step a′ drops that
    // trigger from conv_pin so those convs run the helper's NON-pin FIFO (the dedicated one-block
    // partials makes the non-pin FIFO wrap to base every K-block, so L1_ACC still accumulates at a
    // fixed base — see the dedicate-partials note at the top of kernel_main). The OUT target was NOT
    // migrated to caller_owns because it has no production-trace vehicle (every real no-bias TILE conv
    // is single-output-block → partials aliased onto OUT → already non-pin; the multi-output-block OUT
    // pin path is only reachable via a synthetic act_block_h override) AND caller_owns + the Out-target
    // software-reload would need new shared-helper logic untested by any shipped caller. The synthetic-
    // only non-pin perf delta is accepted.
    //
    // conv_pin is now ALWAYS false: fuse_bias/untilize_out go caller_owns (TileRowMajor), and the OUT
    // trigger is gone. The pin_interm_to_captured_base helper feature is still wired (passed below) for
    // a clean step (b) deletion once no caller uses it; here it always resolves to false.
    constexpr bool conv_pin_interm_target = fuse_bias || untilize_out;
    constexpr bool conv_pin = packer_l1_acc &&
                              (conv_output_layout == compute_kernel_lib::OutputCBLayout::SubblockMajor) &&
                              conv_pin_interm_target;

    constexpr uint32_t out_block_w = in1_block_w;

    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;

    uint32_t bias_block_offset = 0;
    constexpr uint32_t bias_ntiles_w = get_compile_time_arg_val(16);
    constexpr uint32_t bias_cb_id = get_compile_time_arg_val(17);

    constexpr uint32_t mm_in0_cb_id = height_sharded ? tilized_in0_cb_id : in0_cb_id;

    constexpr uint32_t in0_num_subblocks_read_last =
        (split_reader && !split_reader_cb_shared) ? reader_num_h_subblocks / 2 : 0;
    constexpr uint32_t in0_num_subblocks_read = reader_num_h_subblocks - in0_num_subblocks_read_last;

    // if activation reuse is enabled, we need to update read pointers of the act buffers
    // each time we pass on to the new output image row to match the reader behavior
    uint32_t act_cb_start_address = activation_reuse ? get_local_cb_interface(in0_cb_id).fifo_rd_ptr : 0;
    const uint32_t out_cb_tiles =
        activation_reuse ? in0_block_w * (in0_num_subblocks_read + in0_num_subblocks_read_last) : 0;
    const uint32_t tilized_cb_start_address =
        activation_reuse ? get_local_cb_interface(tilized_in0_cb_id).fifo_wr_ptr : 0;
    const uint32_t act_cb_second_reader_start_address =
        activation_reuse ? get_local_cb_interface(in0_cb_second_reader_id).fifo_rd_ptr : 0;

    // For block sharded conv2d, compute kernels may be scheduled on cores that only need tilize
    // operations (input grid) while actual matmul occurs on different cores (output grid).
    // Skip dummy compute operations when possible to reduce di/dt issues, but allow dummy
    // tilize operations on cores without input data for code simplicity.
    bool skip_compute = false;
    if constexpr (check_skip_compute) {
        skip_compute = (bool)get_arg_val<uint32_t>(0);
    }

    experimental::CB cb_in0(in0_cb_id);
    experimental::CB cb_in0_second_reader(in0_cb_second_reader_id);
    experimental::CB cb_tilized_in0(tilized_in0_cb_id);
    experimental::CB cb_mm_in0(mm_in0_cb_id);
    experimental::CB cb_in1(in1_cb_id);
    experimental::CB cb_matmul_partials(matmul_partials_cb);
    experimental::CB cb_out(out_cb_id);
    experimental::CB cb_bias(bias_cb_id);
    experimental::CB cb_untilize_mode_out(untilize_mode_out_cb_id);

    mm_block_init(mm_in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif
    // Kernel-entry base of the dedicated matmul_partials_cb. We reset rd/wr back here at the top of
    // each outer iter so every output block's L1_ACC accumulation lands at the same fixed base — the
    // dedicated one-block region then makes the helper's non-pin FIFO wrap to this base each K-block.
    UNPACK(const uint32_t partials_cb_read_ptr = get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr;)
    PACK(const uint32_t partials_cb_write_ptr = get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr;)

    // Last-block pack target for the matmul helper:
    //   FUSE_BIAS                → Interm: pack to matmul_partials_cb so bias-add can read it.
    //   PACK_RELU && !untilize_out → OutWithRelu: relu directly on the matmul output.
    //   untilize_out (no fuse)   → Interm: pack to matmul_partials_cb (= untilize_mode_out_cb)
    //                              so reblock_and_untilize / untilize can consume it.
    //   else                     → Out: pack straight to out_cb.
    constexpr compute_kernel_lib::LastBlockTarget last_block_target =
        fuse_bias                      ? compute_kernel_lib::LastBlockTarget::Interm
        : (pack_relu && !untilize_out) ? compute_kernel_lib::LastBlockTarget::OutWithRelu
        : untilize_out                 ? compute_kernel_lib::LastBlockTarget::Interm
                                       : compute_kernel_lib::LastBlockTarget::Out;

    using PreKBlockFn = ConvTilizePreKBlock<
        height_sharded,
        packer_l1_acc,
        pack_relu,
        fuse_bias,
        split_reader,
        split_reader_cb_shared,
        activation_reuse,
        in0_block_w,
        in0_cb_id,
        in0_pretilize_cb_id,
        in0_cb_second_reader_id,
        tilized_in0_cb_id,
        in1_cb_id,
        matmul_partials_cb,
        in0_num_subblocks_read,
        in0_num_subblocks_read_last,
        in0_nblocks_w_tilize,
        out_subblock_w,
        out_subblock_h,
        image_width_in_tiles,
        window_reuse_offset,
        tilized_cb_row_offset,
        tilized_cb_second_reader_offset>;

    PreKBlockFn pre_k_block{
        cb_in0,
        cb_in0_second_reader,
        cb_tilized_in0,
        act_cb_start_address,
        act_cb_second_reader_start_address,
        tilized_cb_start_address};

#ifdef SFPU_OP_INIT_ACTIVATION
    using MatmulPostFn = std::conditional_t<fuse_bias, compute_kernel_lib::NoPostCompute, ConvSFPUPostCompute>;
#else
    using MatmulPostFn = compute_kernel_lib::NoPostCompute;
#endif

    // in1 num blocks w is the outer loop. Output blocks are computed in col major order.
    for (uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
        for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
            if constexpr (pack_relu) {
                // for each output block we start we relu disabled so that intermediate results are not relu'd
                PACK((llk_pack_relu_config(ReluConfig::none())));
            }
            // Dedicated one-block matmul_partials_cb: force its rd/wr ptrs back to the kernel-entry
            // base each outer iter so every output block's L1_ACC accumulation lands at the same
            // fixed L1 address. Within the K-loop the helper's non-pin FIFO reserves/pushes/pops in
            // one-block increments, which — because the region holds exactly one output block —
            // wraps back to this base each K-block (matmul's dedicated-partials reset model).
            //
            // caller_owns_pack_target PoC: the caller's own reserve_back/push_back (below, around the
            // matmul call) advance the partials FIFO by exactly one output block per outer iter, and
            // the bias-add's pop drains it — so the FIFO wraps to its base naturally and this manual
            // rewind is both unnecessary and would desync the caller-owned wr_ptr from the reserve.
            if constexpr (!caller_owns_pack_target) {
                UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr;)
                PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr;)
            }

            // ── skip-compute fast path: tilize each K-block, drop the activation, skip the
            // matmul + bias + untilize. Used on input-grid cores that only drive tilize for
            // downstream cores.
            if constexpr (check_skip_compute) {
                if (skip_compute) {
                    for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                        const bool last_inner_dim_block = (in0_block_w_i == in0_num_blocks_w - 1);
                        pre_k_block(in0_block_w_i, in0_num_blocks_w, last_inner_dim_block);
                        cb_mm_in0.wait_front(in0_block_num_tiles);
                        cb_mm_in0.pop_front(in0_block_num_tiles);
                    }
                    continue;
                }
            }

            // ── K-loop matmul ──────────────────────────────────────────────
            // Pick the buffer the last K-block packs to. Mirrors last_block_target above.
            auto& matmul_out_buf = fuse_bias ? cb_matmul_partials : cb_untilize_mode_out;

            const auto shape = compute_kernel_lib::MatmulBlockShape::of(
                in0_num_subblocks,
                in1_num_subblocks,
                out_subblock_h,
                out_subblock_w,
                in0_block_w,
                in0_num_blocks_w,
                /*batch=*/1);

            // init_mode=ShortAfterPreKBlock: the kernel-entry mm_block_init at the top of
            // kernel_main covers initial state; thereafter the helper itself reconfigs srcA/srcB
            // and re-issues mm_block_init_short after each per-K-block tilize (ConvTilizePreKBlock),
            // so the functor no longer carries the matmul-state restore. The downstream bias-add and
            // untilize phases reconfig data formats, and the helper's per-iter restore re-establishes
            // matmul state on the next call. reconfig stays at its INPUT_AND_OUTPUT default — the
            // per-K-block srcA/srcB reconfig matches the old functor restore, and the per-K-block
            // pack reconfig to interm is the same one the pre-loop path used to issue once.
            //
            // matmul_partials_cb is a DEDICATED one-block region (the factory dropped the out_cb
            // alias + sized it to one output block). The helper's FIFO reserves/pushes/pops in
            // one-block increments, which wrap back to the per-iter base (reset at the top of this
            // loop) every K-block — so packer_l1_acc accumulates at a fixed address across K-blocks
            // for multi-output-block convs too, matching matmul's dedicated single-block interm0.
            // caller_owns_pack_target PoC: single outer reserve over the whole output block. The helper
            // packs all K-blocks into this region with L1_ACC accumulating (it manages the per-K-block
            // llk_pack_reconfig_l1_acc itself); no per-block reserve/push/drain. Mirrors the CCL
            // all_gather_minimal_matmul_async compute kernel's caller_owns call (compute.cpp:366-471).
            if constexpr (caller_owns_pack_target) {
                PACK((pack_reconfig_data_format(matmul_partials_cb)));
                cb_matmul_partials.reserve_back(out_block_num_tiles);
            }
            compute_kernel_lib::matmul_block<
                /*transpose=*/false,
                packer_l1_acc,
                last_block_target,
                conv_output_layout,
                compute_kernel_lib::matmul_config::InitMode::ShortAfterPreKBlock,
                compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
                compute_kernel_lib::InputPolicy::WaitAndPopPerKBlock,
                MatmulPostFn,
                PreKBlockFn,
                /*pin_interm_to_captured_base=*/conv_pin,
                compute_kernel_lib::NoPostKBlock,
                /*untilize_block_ct_dim=*/0,
                compute_kernel_lib::NoKBlockInnerDimFn,
                compute_kernel_lib::NoIn0Source,
                compute_kernel_lib::NoIn1BaseOffset,
                /*caller_owns_pack_target=*/caller_owns_pack_target>(
                cb_mm_in0,
                cb_in1,
                matmul_out_buf,
                cb_matmul_partials,
                shape,
                MatmulPostFn{},
                pre_k_block,
                /*in1_per_core_w=*/0,
                /*out_row_width=*/0,
                compute_kernel_lib::NoPostKBlock{},
                compute_kernel_lib::NoKBlockInnerDimFn{},
                compute_kernel_lib::NoIn0Source{},
                compute_kernel_lib::NoIn1BaseOffset{});
            if constexpr (caller_owns_pack_target) {
                cb_matmul_partials.push_back(out_block_num_tiles);
                PACK((llk_pack_reconfig_l1_acc(0)));
            }

            if constexpr (check_skip_compute) {
                if (skip_compute) {
                    continue;
                }
            }
            if constexpr (fuse_bias) {
                if constexpr (pack_relu) {
                    // if last block we pack the final result with relu enabled
                    PACK((llk_pack_relu_config(ReluConfig::zero())));
                }
                if constexpr (packer_l1_acc) {
                    pack_reconfig_l1_acc(0);
                }

                // Bias is row-broadcast on the conv2d path (single bias tile per output column).
                // Caller owns bias CB lifecycle: writer pushes bias_ntiles_w once at startup
                // (load_bias=true→false), and the compute kernel walks through it via
                // bias_block_offset advancing by in1_block_w per outer w iteration. The helper
                // reads bias at offset bias_block_offset+in1_index_subblock_offset and never
                // pops bias_buf, matching that lifecycle.
                cb_bias.wait_front(bias_ntiles_w);
                const auto bias_shape = compute_kernel_lib::BiasAddShape::of(
                    in0_num_subblocks,
                    in1_num_subblocks,
                    out_subblock_h,
                    out_subblock_w,
                    /*out_row_width=*/0);  // 0 => derive from out_subblock_w * in1_num_subblocks
                // conv_output_layout is always SubblockMajor (SBM-only after the dedicate-partials change),
                // so this is the legacy bias path.
#ifdef SFPU_OP_INIT_ACTIVATION
                compute_kernel_lib::add_bias_bcast_rows<
                    compute_kernel_lib::BiasBroadcast::RowBroadcast,
                    conv_output_layout,
                    ConvSFPUPostCompute>(
                    cb_matmul_partials,
                    cb_bias,
                    cb_untilize_mode_out,
                    bias_shape,
                    ConvSFPUPostCompute{},
                    bias_block_offset);
#else
                compute_kernel_lib::
                    add_bias_bcast_rows<compute_kernel_lib::BiasBroadcast::RowBroadcast, conv_output_layout>(
                        cb_matmul_partials, cb_bias, cb_untilize_mode_out, bias_shape, {}, bias_block_offset);
#endif

                // Dedicated one-block partials: when fuse_bias + untilize_out, bias-add writes back into
                // matmul_partials_cb (== untilize_mode_out_cb) in balanced one-block reserve/push, so its
                // FIFO wraps to base; reset rd/wr to the per-iter base so the untilize below reads the
                // freshly-written block from the same base. (Mirrors main's fuse_bias+untilize rewind.)
                if constexpr (untilize_out) {
                    UNPACK(get_local_cb_interface(matmul_partials_cb).fifo_rd_ptr = partials_cb_read_ptr);
                    PACK(get_local_cb_interface(matmul_partials_cb).fifo_wr_ptr = partials_cb_write_ptr);
                }
            }
            if constexpr (untilize_out) {
                if constexpr (packer_l1_acc) {
                    pack_reconfig_data_format(matmul_partials_cb, out_cb_id);
                    pack_reconfig_l1_acc(0);
                }
                if constexpr (pack_relu) {
                    PACK((llk_pack_relu_config(ReluConfig::none())));
                }
                if constexpr (packer_untilize && !tile_pack_row_major) {
                    // Narrow SubblockMajor output (define-absent path): gather subblock-major matmul
                    // output into row-major and untilize via pack_untilize_dest. One call —
                    // reblock_and_untilize loops over all in0_num_subblocks internally and owns the
                    // data-format reconfig (srcA=matmul_partials, pack=out) + the pack_untilize init/uninit.
                    compute_kernel_lib::reblock_and_untilize<out_subblock_w, out_block_w>(
                        in0_num_subblocks,
                        in1_num_subblocks,
                        out_subblock_num_tiles,
                        out_subblock_h,
                        cb_matmul_partials,
                        cb_out);
                } else {
                    // Wide SubblockMajor output (packer_untilize=false, !tile_pack_row_major): plain
                    // untilize reads the row strip sequentially and converts to row-major. ALSO the
                    // tile_pack_row_major path (M2 TRM, ROW_MAJOR output): the matmul already packed the
                    // interm row-major (contiguous tile-row order), so no reblock gather is needed — plain
                    // untilize reads the row strip sequentially. srcA reconfig to matmul_partials is handled
                    // externally here because untilize is invoked with NoReconfigure (the pack format came
                    // from the packer_l1_acc reconfig above).
                    if constexpr (!fuse_bias) {
                        reconfig_data_format_srca(in1_cb_id, matmul_partials_cb);
                    }
                    // Flatten nested loops into single iteration count: in0_num_subblocks * out_subblock_h
                    compute_kernel_lib::untilize<
                        out_block_w,
                        matmul_partials_cb,
                        out_cb_id,
                        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                        in0_num_subblocks * out_subblock_h);
                }
            }
            if constexpr ((in1_num_blocks_w > 1 || in0_num_blocks_h > 1)) {
                if constexpr (fuse_bias) {
                    reconfig_data_format(matmul_partials_cb, in1_cb_id, bias_cb_id, mm_in0_cb_id);
                } else {
                    reconfig_data_format_srca(matmul_partials_cb, in1_cb_id);
                }
            }
        }  // for in0_num_blocks_h
        if constexpr (fuse_bias) {
            bias_block_offset += in1_block_w;
        }
    }  // for in1_num_blocks_w
}  // void kernel_main()
