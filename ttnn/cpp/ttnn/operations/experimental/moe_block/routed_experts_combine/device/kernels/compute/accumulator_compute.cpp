// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Accumulator compute: sums each token's routed rows, weighted by their gate weights.
//
// For every row the reader hands over (the row, its gate weight, the token's local index) it computes
//
//     sum[token] += row * weight
//
// The multiply is a broadcast multiply: srcA is the row, srcB the gate weight (element 0 of its tile, broadcast to
// every element), the product lands in DEST. The add is done by the packer: packing with L1 accumulation on adds DEST
// onto the token's sum already in cb_acc (a token's first row packs without it, so stale L1 is overwritten). Every
// real token gets exactly top_k rows (the router's top-k). A token is finished as soon as its top_k-th row is in: its
// sum goes out as bf16 (cb_out) and its index to the writer (cb_finished), so output rows are written while other
// tokens' rows still land, in the order tokens finish. Padding tokens get no rows; once the last row is in, each gets a
// zero row, in token order.
//
// The rows are row-major data, but they are processed as if they were tiles: a W-wide bf16 row is W / 1024 "tiles".
// That is fine because everything here is elementwise - the order of the elements inside a tile does not matter, and
// they come back out in the same order.
//
// combine_snap (snap_step_bits != 0, the step q a power of two): the sums are int32 counts of q. Each product v becomes
// round(v / q) as a sign-magnitude int32 on the SFPU (snap_to_sign_magnitude_count_tile), and the packer accumulates
// cb_acc (an Int32 CB) in INT32, which adds sign-magnitude. Integer adds are exact, so a token's sum does not depend on
// the order its rows land in, up to (2^31 - 1) q, where the L1 add saturates. The readout converts each count back
// to fp32 times q (sign_magnitude_count_to_float_tile). Without the snap the sums are fp32, order-dependent in the last
// bits.
//
// The reader's meta pages (cb_meta, one uint32 each):
//
//   page        | value
//   ------------+--------------------------------------------------------------------------------------------
//   first       | 0x80000000 | this accumulator's tokens, or 0: no accumulator on this core on this chip (return)
//   one per row | the row's local token index (token / A: tokens are dealt to the accumulators round-robin)
//   last        | 0xFFFFFFFF: no more rows (the compute does not know the row count)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compile_time_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "../combine_names.hpp"
#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {
// A DEST face: fp32 v -> round(v / q) (ties to even) as a sign-magnitude int32, |v / q| clamped below 2^31. No compute
// API does this: typecast_tile's fp32 -> int32 truncates and gives two's complement.
template <uint32_t inverse_step_bits>
inline void snap_to_sign_magnitude_count() {
    const sfpi::vFloat largest_count = 2147483520.0f;  // the largest fp32 below 2^31
    const sfpi::vInt two_to_23_bits = 0x4B000000;
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        const sfpi::vFloat scaled = sfpi::vFloat(sfpi::dst_reg[0]) * __builtin_bit_cast(float, inverse_step_bits);
        const sfpi::vFloat magnitude = sfpi::min(sfpi::abs(scaled), largest_count);
        // below 2^23, adding 2^23 rounds the magnitude to an integer, which is then the sum's bits less 2^23's bits
        sfpi::vInt count_magnitude = sfpi::as<sfpi::vInt>(magnitude + 8388608.0f) - two_to_23_bits;
        const sfpi::vInt mantissa_shift = sfpi::exexp(magnitude) - 23;
        v_if(mantissa_shift >= 0) {  // from 2^23 on the magnitude is an integer: its mantissa, shifted up
            count_magnitude = sfpi::as<sfpi::vInt>(sfpi::shft(
                sfpi::as<sfpi::vUInt>(sfpi::exman(magnitude, sfpi::MantissaMode::WithUnitBit)), mantissa_shift));
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::copysgn(sfpi::as<sfpi::vSMag>(count_magnitude), scaled);
        sfpi::dst_reg++;
    }
}

// A DEST face: sign-magnitude int32 count -> fp32 count x q
template <uint32_t step_bits>
inline void sign_magnitude_count_to_float() {
#pragma GCC unroll 8
    for (int face_row = 0; face_row < 8; face_row++) {
        const sfpi::vSMag count = sfpi::dst_reg[0];
        sfpi::dst_reg[0] =
            sfpi::convert<sfpi::vFloat>(count, sfpi::RoundMode::Nearest) * __builtin_bit_cast(float, step_bits);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

ALWI void snap_sfpu_init() { MATH((SFPU_UNARY_INIT(unused))); }
template <uint32_t inverse_step_bits>
ALWI void snap_to_sign_magnitude_count_tile(uint32_t dst_index) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, snap_to_sign_magnitude_count, (inverse_step_bits), dst_index, VectorMode::RC)));
}
template <uint32_t step_bits>
ALWI void sign_magnitude_count_to_float_tile(uint32_t dst_index) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, sign_magnitude_count_to_float, (step_bits), dst_index, VectorMode::RC)));
}

// fp32 tiles DEST holds at once (fp32 accumulation, half sync: 4)
constexpr uint32_t kDestTiles = compute_kernel_lib::DEST_AUTO_LIMIT;
constexpr uint32_t kMetaActive = 0x80000000u;
constexpr uint32_t kMetaEnd = 0xFFFFFFFFu;

void kernel_main() {
    // (all compile-time args are named: combine_names.hpp)
    constexpr uint32_t cb_row = get_named_compile_time_arg_val(combine_names::kRowCb);
    constexpr uint32_t cb_meta = get_named_compile_time_arg_val(combine_names::kMetaCb);
    constexpr uint32_t cb_acc = get_named_compile_time_arg_val(combine_names::kAccumulatorCb);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val(combine_names::kOutputCb);
    constexpr uint32_t tiles_per_row = get_named_compile_time_arg_val(combine_names::kRowTiles);
    constexpr uint32_t max_tokens = get_named_compile_time_arg_val(combine_names::kMaxTokensPerAccumulator);
    constexpr uint32_t cb_w = get_named_compile_time_arg_val(combine_names::kWeightCb);
    // (fp32 bits of the snap step q, a power of two; 0: off - fp32 sums)
    constexpr uint32_t snap_step_bits = get_named_compile_time_arg_val(combine_names::kSnapStepBits);
    constexpr bool snap = snap_step_bits != 0;
    // (1 / q, a power of two too: its exponent field is 254 less q's)
    constexpr uint32_t inverse_snap_step_bits = snap ? 0x7F000000u - snap_step_bits : 0u;
    // the router's top-k: a token is written out as soon as its top_k-th row has been added
    constexpr uint32_t top_k = get_named_compile_time_arg_val(combine_names::kTopK);
    // cb_finished: each finished token's accumulator-local index, in the order its row goes into cb_out (to the
    // writer); cb_sync: the pack -> unpack handshake before a finished token's sum is read back
    constexpr uint32_t cb_finished = get_named_compile_time_arg_val(combine_names::kFinishedCb);
    constexpr uint32_t cb_sync = get_named_compile_time_arg_val(combine_names::kSyncCb);
    static_assert(top_k >= 1 && top_k < 256, "top_k must be in [1, 255]: rows_seen counts a token's rows in a byte");
    compute_kernel_hw_startup(cb_row, cb_w, cb_acc);
    // (no runtime args: whether this core accumulates differs by chip, and one program for the mesh needs them alike)
    cb_wait_front(cb_meta, 1);
    const uint32_t first_page = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    if (first_page == 0) {
        return;
    }
    const uint32_t n_tokens = first_page & ~kMetaActive;
    mul_bcast_scalar_init(cb_row, cb_w);
    uint8_t rows_seen[max_tokens];  // (each token's rows added so far)
    for (uint32_t i = 0; i < max_tokens; ++i) {
        rows_seen[i] = 0;
    }
    cb_reserve_back(cb_acc, n_tokens * tiles_per_row);

    // The unpack -> math -> pack pipeline runs in one of two setups. Accumulating (the row loop below): the unpacker
    // reads a row and its gate weight, the packer adds their product onto the token's sum in cb_acc (L1 accumulate
    // on). Writing (output below): the unpacker reads a token's sum from cb_acc, the packer writes it to cb_out as
    // bf16 (L1 accumulate off).
    //
    // begin_output() switches the pipeline from accumulating to writing. It writes nothing itself. Call it ONCE before
    // a run of output() calls, not once per token: every output() after it uses the same setup. A token finished
    // early is a run of one (switch, write it, switch back); the end of the call is one run for all the tokens left.
    auto begin_output = [&]() {
        pack_reconfig_l1_acc(0);
        reconfig_data_format_srca(cb_acc);
        pack_reconfig_data_format(cb_out);
        // (the copy is an unpack straight to DEST - no math datacopy - so the SFPU init alongside it holds)
        copy_init(cb_acc);
        if constexpr (snap) {
            snap_sfpu_init();
        }
    };
    // output(token) writes one finished token, with the pipeline already switched by begin_output(): its sum as a bf16
    // row to cb_out (a zero row if the token got no rows - not the stale L1 of its cb_acc slot), then its id to
    // cb_finished, which tells the writer where the row goes.
    auto output = [&](uint32_t token) {
        cb_reserve_back(cb_out, tiles_per_row);
        if (rows_seen[token] == 0) {
            fill_tile_init();
            for (uint32_t first = 0; first < tiles_per_row; first += kDestTiles) {
                const uint32_t count = tiles_per_row - first < kDestTiles ? tiles_per_row - first : kDestTiles;
                tile_regs_acquire();
                for (uint32_t j = 0; j < count; ++j) {
                    fill_tile(j, 0.0f);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < count; ++j) {
                    pack_tile(j, cb_out);
                }
                tile_regs_release();
            }
            // (fill_tile_init replaced begin_output()'s copy / SFPU setup: put it back for the next output())
            copy_init(cb_acc);
            if constexpr (snap) {
                snap_sfpu_init();
            }
        } else {
            // (cb_acc is reserved once at the start and pushed only at the call's end: its read pointer stays at token
            // 0's slot, so a token's sum is token * tiles_per_row tiles from it, before and after the push)
            for (uint32_t first = 0; first < tiles_per_row; first += kDestTiles) {
                const uint32_t count = tiles_per_row - first < kDestTiles ? tiles_per_row - first : kDestTiles;
                tile_regs_acquire();
                for (uint32_t j = 0; j < count; ++j) {
                    copy_tile(cb_acc, token * tiles_per_row + first + j, j);
                }
                if constexpr (snap) {  // the int32 counts back to fp32: count x q
                    for (uint32_t j = 0; j < count; ++j) {
                        sign_magnitude_count_to_float_tile<snap_step_bits>(j);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < count; ++j) {
                    pack_tile(j, cb_out);
                }
                tile_regs_release();
            }
        }
        cb_push_back(cb_out, tiles_per_row);
        cb_reserve_back(cb_finished, 1);
        PACK((*reinterpret_cast<volatile uint32_t*>(get_local_cb_interface(cb_finished).fifo_wr_ptr << 4) = token));
        cb_push_back(cb_finished, 1);
    };

    while (true) {
        cb_wait_front(cb_meta, 1);
        const uint32_t token = read_tile_value(cb_meta, 0, 0);
        cb_pop_front(cb_meta, 1);
        if (token == kMetaEnd) {
            break;
        }
        cb_wait_front(cb_row, tiles_per_row);
        cb_wait_front(cb_w, 1);
        pack_reconfig_l1_acc(rows_seen[token] ? 1 : 0);
        for (uint32_t first = 0; first < tiles_per_row; first += kDestTiles) {
            const uint32_t count = tiles_per_row - first < kDestTiles ? tiles_per_row - first : kDestTiles;
            tile_regs_acquire();
            if constexpr (snap) {
                mul_bcast_scalar_init(cb_row, cb_w);
            }
            for (uint32_t j = 0; j < count; ++j) {
                mul_tiles_bcast_scalar(cb_row, cb_w, first + j, 0, j);
            }
            if constexpr (snap) {  // every term to its int32 count of q
                snap_sfpu_init();
                for (uint32_t j = 0; j < count; ++j) {
                    snap_to_sign_magnitude_count_tile<inverse_snap_step_bits>(j);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < count; ++j) {
                pack_tile<true>(j, cb_acc, token * tiles_per_row + first + j);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_row, tiles_per_row);
        cb_pop_front(cb_w, 1);
        if (++rows_seen[token] == top_k) {
            // The token's last row is in: write it now. First the packer must have added that row into L1 before the
            // unpacker reads the sum (a CB push from the PACK thread waits for the packer); then switch to writing,
            // write the token, and switch back to accumulating for the next row.
            cb_reserve_back(cb_sync, 1);
            cb_push_back(cb_sync, 1);
            cb_wait_front(cb_sync, 1);
            begin_output();
            output(token);
            cb_pop_front(cb_sync, 1);
            // back to accumulating (begin_output()'s reverse)
            reconfig_data_format_srca(cb_row);
            pack_reconfig_data_format(cb_acc);
            mul_bcast_scalar_init(cb_row, cb_w);
        }
    }
    // The end of the call: every row is in. Once all the packs are in L1, switch to writing once and write every token
    // that did not finish early - the padding tokens, which got no rows (a zero row each) - in token order. When every
    // token finished early, nothing is written here.
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, n_tokens * tiles_per_row);
    cb_wait_front(cb_acc, n_tokens * tiles_per_row);
    begin_output();
    for (uint32_t token = 0; token < n_tokens; ++token) {
        if (rows_seen[token] < top_k) {
            output(token);
        }
    }
}
