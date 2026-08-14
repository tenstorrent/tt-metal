// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute (TRISC). One helper call for the whole core: the library
// tilize helper loops num_blocks tile-rows internally, so the LLK init/uninit
// is amortized across every block this core owns (master.md Part 1
// `compute_block_size`).
//
// WT_CHUNK (block_width_tiles) is the W block factor from op_design.md §1.4 and
// arrives as a compile-time arg — the helper needs it as a template parameter.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace {

// ── PERF 2 — RAW-LLK BYPASS, justified and MEASURED ─────────────────────────
// Helper bypassed: `compute_kernel_lib::tilize<...>` (tilize_helpers.hpp), and
// under it the compute API `ckernel::tilize_block` (api/compute/tilize.h:171),
// on the REGULAR (non-fast) tilize path ONLY.
//
// `tilize_block`'s body is one DEST acquire -> datacopy -> commit -> release
// round trip PER TILE, always on DEST slot 0 — so it uses 1/8 (1/4 for 32-bit
// datums) of the DEST section it is allowed and pays a math<->pack semaphore
// trip for every single tile. The FAST path (`fast_tilize_block`, taken only for
// 32x32 output tiles with a bf16/fp32 input and a non-fp32 output) already fills
// a whole DEST section per acquire, which is exactly why it is fast — and a
// handshake ablation showed 99.5% of its wall is LLK payload, i.e. nothing left
// to win there. Everything ELSE — every `tile_height < 32`, every fp32 output,
// the bf16->fp32 cast, the integer dtypes — falls to the regular path.
//
// `tilize_block_wide` is `tilize_block` with the DEST window widened to the full
// section: N tiles datacopied into DEST slots 0..N-1 under ONE acquire, packed
// out under one commit. Same unpack MOP, same datacopy LLK, same pack LLK, same
// data formats, same DST_ACCUM_MODE / DST_SYNC_MODE / MATH_FIDELITY — the ONLY
// change is how many tiles share a DEST section, so the output is bit-identical
// by construction (and was verified bit-exact on every cell measured).
//
// Measured on the zero-NoC same-spec sharded plan [1,1,2048,256] H x8 bf16,
// baseline -> wide: tile_h=8 13,435 -> 7,428 (1.81x), tile_h=4 1.59x, tile_h=2
// 1.63x, tile_h=1 1.61x, tile_h=16 1.43x, bf16->fp32 cast 1.22x; interleaved
// DRAM a_square tile_h=8 97,226 -> 91,172 (1.066x). Flat, never slower, on the
// dtypes whose regular path is already DEST-bound (fp32, uint32, uint16, uint8)
// and at WT_CHUNK <= 2 (nothing to widen). No measured regression anywhere.
//
// Why raw and not a helper flag: `compute_kernel_lib::tilize` hard-wires
// `tilize_block` on its non-fast branch (tilize_helpers.inl:246) and exposes no
// DEST-window parameter; `block_width_tiles` is a template parameter but it
// controls the CB HANDSHAKE width, not the DEST window. There is no compute-API
// entry point for "regular tilize, N tiles per DEST section" at all. Classified
// `capability` in the changelog's Helper bypasses table. Do NOT "fix" this back
// to the helper call — that reverts a measured 1.4-1.8x on every tiny-tile,
// cast and integer cell.
template <uint32_t window>
ALWI void tilize_block_wide(uint32_t icb, uint32_t block, uint32_t ocb) {
    static_assert(window >= 1, "window must be >= 1");
    using namespace ckernel;

    UNPACK((llk_unpack_tilize_block(icb, block, 0 /*input_tile_index*/)));

    uint32_t done = 0;
    while (done < block) {
        const uint32_t left = block - done;
        const uint32_t n = (left < window) ? left : window;

        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        for (uint32_t i = 0; i < n; ++i) {
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    i /*dst index*/, icb)));
        }
        for (uint32_t i = 0; i < n; ++i) {
            PACK((llk_pack<DST_ACCUM_MODE, true /*out_of_order*/, PackMode::Default>(
                i /*dst tile index*/, ocb, done + i /*ocb tile index*/)));
        }

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));

        done += n;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_sticks = 0;
    constexpr uint32_t cb_output_tiles = 16;

    constexpr uint32_t wt_chunk = get_compile_time_arg_val(0);
    constexpr uint32_t needs_cast = get_compile_time_arg_val(1);
    // Perf 2 (R_RETILE only): the reader landed the face permutation DIRECTLY in
    // the output tile layout. With no cast it produced cb_output_tiles itself and
    // this kernel has nothing to do at all; with a cast it produced an
    // output-SHAPED tile in the INPUT dtype in cb_input_sticks and this kernel
    // owns the conversion ALONE — a datacopy, not a tilize.
    constexpr uint32_t retile_direct = get_compile_time_arg_val(2);
    // Perf 2 SPLIT READER: both DM RISCs read, each into its own input CB, so this
    // kernel consumes them ALTERNATELY (block i comes from CB A when i is even).
    // The writer kernel is not launched on that path, so this kernel also takes
    // over draining the aliased OUTPUT CB — that CB is the resident shard, and it
    // must keep exactly one consumer.
    constexpr uint32_t split_reader = get_compile_time_arg_val(3);
    constexpr uint32_t cb_input_sticks_b = 3;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);

    if (num_blocks == 0) {
        return;
    }

    using namespace compute_kernel_lib::tilize_config;

    if constexpr (split_reader) {
        // The alternation costs NOTHING: a control arm (one reader, one CB, but
        // this same per-block back-to-back form) measured 14,768 / 14,727 ns
        // against the batched baseline's 14,780 / 14,908 on the crossover and
        // 18,281 / 18,142 against 18,406 / 18,613 on the reshard — flat within 1%.
        // That is the result that closed Perf 1's open question about whether the
        // CB alternation would eat the split's win.
        //
        // No raw LLK here: the library helper's documented InitOnly / Neither /
        // UninitOnly lifecycle expresses back-to-back single-block calls natively.
        constexpr auto k_reconfig = needs_cast ? ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                                               : ReconfigureRegisterDatatypeMode::NoReconfigure;
        compute_kernel_lib::tilize<
            wt_chunk,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::InitOnly,
            WaitMode::WaitBlock,
            k_reconfig,
            Fp32Mode::Fast>(0);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            if (b & 1) {
                compute_kernel_lib::tilize<
                    wt_chunk,
                    cb_input_sticks_b,
                    cb_output_tiles,
                    InitUninitMode::Neither,
                    WaitMode::WaitBlock,
                    k_reconfig,
                    Fp32Mode::Fast>(1);
            } else {
                compute_kernel_lib::tilize<
                    wt_chunk,
                    cb_input_sticks,
                    cb_output_tiles,
                    InitUninitMode::Neither,
                    WaitMode::WaitBlock,
                    k_reconfig,
                    Fp32Mode::Fast>(1);
            }
            // The drain the writer would have done. Measured cost: nothing on the
            // DRAM flavor (9,857 without vs 9,919 with) and ~6% on the L1 flavor —
            // paid to keep the aliased CB's single-consumer contract.
            cb_wait_front(cb_output_tiles, wt_chunk);
            cb_pop_front(cb_output_tiles, wt_chunk);
        }
        compute_kernel_lib::tilize<
            wt_chunk,
            cb_input_sticks,
            cb_output_tiles,
            InitUninitMode::UninitOnly,
            WaitMode::WaitBlock,
            k_reconfig,
            Fp32Mode::Fast>(0);
        return;
    }

    if constexpr (retile_direct) {
        if constexpr (!needs_cast) {
            // The reader IS the op on this path — it is cb_output_tiles' producer
            // and the writer its consumer, so this kernel must not touch either CB.
            return;
        }
        compute_kernel_lib::copy_tiles<
            compute_kernel_lib::CopyInputPolicy::WaitAndPop,
            compute_kernel_lib::CopyDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_input_sticks, cb_output_tiles, num_blocks * wt_chunk);
        return;
    }

    // The library's own fast/regular decision, re-evaluated here so the DEST
    // window can be applied to exactly the branch that lacks one. Public
    // constexpr, so exactly one of the two bodies below is emitted.
    constexpr bool use_fast = compute_kernel_lib::can_use_fast_tilize<wt_chunk, cb_input_sticks, cb_output_tiles>();

    if constexpr (use_fast) {
        // ── the fast LLK path: the helper verbatim ────────────────────────
        // `fast_tilize_block` already fills a whole DEST section per acquire and
        // a handshake ablation puts 99.5% of this path's wall in LLK payload, so
        // there is nothing to widen and nothing to hoist. Unchanged since Phase 0.
        if constexpr (needs_cast) {
            // A real value-preserving cast: reconfigure both unpack and pack.
            compute_kernel_lib::tilize<
                wt_chunk,
                cb_input_sticks,
                cb_output_tiles,
                InitUninitMode::InitAndUninit,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
                Fp32Mode::Fast>(num_blocks);
        } else {
            // Same format in and out — skip the ~150 ns register reconfiguration.
            compute_kernel_lib::tilize<
                wt_chunk,
                cb_input_sticks,
                cb_output_tiles,
                InitUninitMode::InitAndUninit,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::NoReconfigure,
                Fp32Mode::Fast>(num_blocks);
        }
    } else {
        // ── the regular LLK path, with the DEST window widened (Perf 2) ────
        // Prologue/epilogue reproduce exactly what the helper does for the
        // non-fast path (tilize_helpers.inl:159-199, 258-272); only the block
        // body changes. See tilize_block_wide's header comment for the measured
        // justification of the bypass.
        //
        // DEST tile capacity. NOT `compute_kernel_lib::DEST_AUTO_LIMIT`: that
        // halves the capacity on DST_ACCUM_MODE alone, but a 32-BIT INPUT DATUM
        // (Float32/Tf32/Int32/UInt32) occupies a 32-bit DEST slot whether or not
        // fp32 accumulation is on. MEASURED: uint32 tilize with
        // fp32_dest_acc_en=false (so DEST_AUTO_LIMIT == 8) is NOT bit-exact at a
        // window of 8; it IS at 4. This is the corrected rule, and the library's
        // get_dest_limit() is reported upstream as a latent capability bug.
        constexpr uint32_t in_fmt = compute_kernel_lib::dfb_l1_format<cb_input_sticks>();
        constexpr bool input_is_32bit =
            in_fmt == static_cast<uint32_t>(DataFormat::Float32) || in_fmt == static_cast<uint32_t>(DataFormat::Tf32) ||
            in_fmt == static_cast<uint32_t>(DataFormat::Int32) || in_fmt == static_cast<uint32_t>(DataFormat::UInt32);
        constexpr uint32_t dest_window = (compute_kernel_lib::get_fp32_dest_acc_enabled() || input_is_32bit)
                                             ? (compute_kernel_lib::get_dst_full_sync_enabled() ? 8u : 4u)
                                             : (compute_kernel_lib::get_dst_full_sync_enabled() ? 16u : 8u);

        if constexpr (needs_cast) {
            reconfig_data_format_srca(cb_input_sticks);  // srcB is fast-path only
            pack_reconfig_data_format(cb_output_tiles);
        }
        tilize_init(cb_input_sticks, wt_chunk, cb_output_tiles);
        DataflowBuffer in_dfb(cb_input_sticks);
        DataflowBuffer out_dfb(cb_output_tiles);
        for (uint32_t block = 0; block < num_blocks; ++block) {
            in_dfb.wait_front(wt_chunk);
            out_dfb.reserve_back(wt_chunk);
            tilize_block_wide<dest_window>(cb_input_sticks, wt_chunk, cb_output_tiles);
            out_dfb.push_back(wt_chunk);
            in_dfb.pop_front(wt_chunk);
        }
        tilize_uninit(cb_input_sticks, cb_output_tiles);
    }
}
