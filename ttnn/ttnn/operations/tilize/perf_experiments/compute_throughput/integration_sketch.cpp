// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ============================================================================
// INTEGRATION SKETCH — not compiled, not included by anything.
// Proposed replacement body for ttnn/ttnn/operations/tilize/kernels/
// tilize_compute.cpp, plus the `tilize_block_wide` block body that moves with it.
// ============================================================================
//
// HOST CHANGES: NONE.
//   * `derive_blocking()` / `derive_shard_blocking()` / `wt_cap()` are unchanged.
//     The DEST window is NOT the block factor: it is derived on-device from
//     DST_ACCUM_MODE (already in the JIT header) and the input CB's L1 format
//     (already in `unpack_src_format[]`), neither of which the host has to
//     compute or pass.  WT_CHUNK keeps its current meaning (CB handshake width /
//     tiles per `tilize_block` call) and its current L1-bounded derivation.
//   * The measured caveat the host does NOT need to act on: the win only appears
//     for WT_CHUNK >= 4 (at WT_CHUNK 1 or 2 there is nothing to widen, and the
//     arms are exactly flat, never slower).  `derive_blocking()` already picks
//     the COARSEST chunk that fits, so it lands >= 4 wherever WT allows.
//
// READER / WRITER CONTRACT: unchanged.  Same per-block cb_wait_front /
// cb_reserve_back / cb_push_back / cb_pop_front counts, same block geometry,
// same page sizes.  Only the math<->pack DEST windowing inside one block moves.
//
// PRECISION CONTRACT: untouched.  fp32_dest_acc_en, dst_full_sync_en,
// math_fidelity, math_approx_mode and every dtype are read, never written.
// Output is BIT-EXACT with today on every cell measured (torch.equal).
//
// PREFERRED LONG-TERM FIX: this belongs in the LIBRARY, not the op.  Replacing
// `tilize_block` with the windowed form inside `compute_kernel_lib::tilize`'s
// non-fast branch (tilize_helpers.inl:246) would give every tilize caller the
// win with no op change at all.  The op-local form below is what to graduate
// until the helper takes it.

// ---------------------------------------------------------------------------
// (1) The block body.  Same unpack MOP, same datacopy LLK, same pack LLK as
//     `ckernel::tilize_block` (api/compute/tilize.h:171) — the ONLY difference
//     is that N tiles share one DEST section instead of each tile taking its own
//     acquire/release round trip on slot 0.
// ---------------------------------------------------------------------------
template <uint32_t window>
ALWI void tilize_block_wide(uint32_t icb, uint32_t block, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_block(icb, block, 0)));
    uint32_t done = 0;
    while (done < block) {
        const uint32_t left = block - done;
        const uint32_t n = (left < window) ? left : window;
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        for (uint32_t i = 0; i < n; ++i) {
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    i, icb)));
        }
        for (uint32_t i = 0; i < n; ++i) {
            PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, ocb, done + i)));
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        done += n;
    }
}

// ---------------------------------------------------------------------------
// (2) The DEST tile capacity.  DO NOT use `compute_kernel_lib::DEST_AUTO_LIMIT`
//     here: it halves the capacity on DST_ACCUM_MODE alone, but a 32-BIT INPUT
//     DATUM occupies a 32-bit DEST slot regardless of the accumulation flag.
//     MEASURED: uint32 tilize (fp32_dest_acc_en=false, so DEST_AUTO_LIMIT==8)
//     is NOT bit-exact at a window of 8; it IS bit-exact at 4.
// ---------------------------------------------------------------------------
constexpr bool k_input_is_32bit = []() {
    constexpr uint32_t f = compute_kernel_lib::dfb_l1_format<cb_input_sticks>();
    return f == static_cast<uint32_t>(DataFormat::Float32) || f == static_cast<uint32_t>(DataFormat::Tf32) ||
           f == static_cast<uint32_t>(DataFormat::Int32) || f == static_cast<uint32_t>(DataFormat::UInt32);
}();

constexpr uint32_t k_dest_window = (compute_kernel_lib::get_fp32_dest_acc_enabled() || k_input_is_32bit)
                                       ? (compute_kernel_lib::get_dst_full_sync_enabled() ? 8u : 4u)
                                       : (compute_kernel_lib::get_dst_full_sync_enabled() ? 16u : 8u);

// ---------------------------------------------------------------------------
// (3) The kernel body.  The FAST tilize path is left exactly as it is today —
//     `fast_tilize_block` already fills a whole DEST section per acquire, and
//     the ablation says that path is 100% LLK payload with < 1% of the wall in
//     the CB handshake.  Only the REGULAR path changes.
//
//     `can_use_fast_tilize` is a public constexpr of the library, so the split
//     is a compile-time `if constexpr` and exactly one of the two bodies is
//     emitted per program.
// ---------------------------------------------------------------------------
void kernel_main_sketch() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_input_sticks, cb_output_tiles);
    if (num_blocks == 0) {
        return;
    }
    MaybeDeviceZoneScope("compute_tilize");

    using namespace compute_kernel_lib::tilize_config;
    constexpr bool use_fast = compute_kernel_lib::can_use_fast_tilize<wt_chunk, cb_input_sticks, cb_output_tiles>();

    if constexpr (use_fast) {
        // ---- unchanged: today's helper call, both cast and no-cast forms ----
        if constexpr (needs_cast) {
            compute_kernel_lib::tilize<
                wt_chunk,
                cb_input_sticks,
                cb_output_tiles,
                InitUninitMode::InitAndUninit,
                WaitMode::WaitBlock,
                ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
                Fp32Mode::Fast>(num_blocks);
        } else {
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
        // ---- new: the regular path, with the DEST window widened -------------
        // Prologue/epilogue reproduce exactly what the helper does for the
        // non-fast path (tilize_helpers.inl:159-199, 258-272).
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
            tilize_block_wide<k_dest_window>(cb_input_sticks, wt_chunk, cb_output_tiles);
            out_dfb.push_back(wt_chunk);
            in_dfb.pop_front(wt_chunk);
        }
        tilize_uninit(cb_input_sticks, cb_output_tiles);
    }
}

// ---------------------------------------------------------------------------
// MEASURED (n150 Wormhole b0, bit-exact everywhere, ONE fresh-cache run/arm
// unless noted).  Same-spec HEIGHT-sharded L1 [1,1,2048,256] on 8 cores — the
// zero-NoC plan where the TRISC is the wall:
//
//   regime               baseline    wide_dest    speedup
//   tile_h = 8            13,435       7,428       1.81x
//   tile_h = 2            42,994      26,302       1.63x
//   tile_h = 1            83,682      51,827       1.61x
//   tile_h = 4            21,643      13,600       1.59x
//   tile_h = 16            9,337       6,513       1.43x
//   bf16 -> fp32 cast      8,448       6,914       1.22x
//   uint16                 5,942       5,797       flat
//   uint8                  4,653       4,727       flat
//   fp32 -> fp32          12,597      12,548       flat
//   uint32 -> uint32      12,624      12,514       flat
//   bf16 tile_h=32 (fast)  5,114       5,104       identical code
//
// Interleaved DRAM->DRAM [1,1,2048,2048] on 64 cores (median of 3):
//   tile_h = 8            97,226      91,172       1.066x
//   tile_h = 1           250,898     252,861       flat (NoC-bound)
//   tile_h = 32           86,250      86,758       flat (noise control, same code)
// ---------------------------------------------------------------------------
