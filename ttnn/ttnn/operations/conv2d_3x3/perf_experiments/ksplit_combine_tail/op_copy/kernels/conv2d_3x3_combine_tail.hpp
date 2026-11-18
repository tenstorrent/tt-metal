// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// ksplit_combine_tail experiment (perf round 4): the K split's owner-side combine tail in one DEST pass.
//
// COMBINE_TAIL (a define, 0 = the committed path) is a bit set:
//   CT_FUSED      sum the k_cols slots, add the bias and pack (ReLU) to the output shard in ONE DEST pass, DEST-limit
//                 output tiles per section, inits hoisted per section. No cb_combined fp32 round trip.
//   CT_EXACT      (with CT_FUSED) the bias goes in after the whole sum through dest reuse (DEST -> SrcA, +
//   row-broadcast
//                 bias in SrcB): the sum takes the same SrcA rounding as the committed path's cb_combined unpack, and
//                 the slot adds run in the committed order (odd k_cols: copy slot 0, then pairs (1,2), (3,4), ...;
//                 even: pairs (0,1), (2,3), ...). Without it the bias is folded into the first operand pair (odd
//                 k_cols: DEST = slot 0 + bias; even: DEST = slot 0, then slot 1 + bias), so the k-way sum never
//                 leaves fp32 DEST.
//   CT_OWN_LOCAL  (N-slice-outer walk) the owner's own slice is packed by the matmul straight into cb_combined (free,
//                 since the fused tail no longer stages the sum there) and read from there by the combine; the writer
//                 never copies it into the own slot over NoC1, and the owner pushes its slots after k_cols - 1
//                 arrivals.
//
// Raw LLK (helper bypassed: ckl::reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip> + ckl::add_bias_bcast_rows): the
// reduce helper packs one fp32 tile per DEST section and cannot add a row-broadcast operand inside its accumulation;
// the bias helper reads its addend from a CB. Neither can (a) take operand tiles from two CBs (own slice + slots) per
// pair, (b) fold a row-broadcast add into an accumulating DEST pass (ELWADD acc_to_dest with SrcB row broadcast, or
// dest reuse DEST -> SrcA with SrcB row broadcast), or (c) pack several reduced tiles per section.

#pragma once

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

#ifndef COMBINE_TAIL
#define COMBINE_TAIL 0
#endif

namespace conv2d_3x3::combine_tail {

constexpr uint32_t CT_OWN_LOCAL = 1;
constexpr uint32_t CT_FUSED = 2;
constexpr uint32_t CT_EXACT = 4;

constexpr uint32_t mode = COMBINE_TAIL;
constexpr bool fused = (mode & CT_FUSED) != 0;
constexpr bool exact = (mode & CT_EXACT) != 0;
static_assert(!(mode & CT_OWN_LOCAL) || fused, "the own slice lives in cb_combined, which only the fused tail frees");

// Operand access with compile-time CB ids in every LLK call: a runtime CB id costs each add's unpack / math call its
// constant-folded operand lookup (tensor shape, CB interface), measured ~0.6 us over a res5 combine. So the own slice
// (cb_own, virtual slot my_col) is resolved by branching to a call whose CB ids are constants.
template <bool own_local, uint32_t cb_slots, uint32_t cb_own, uint32_t slot_tiles>
struct Operands {
    uint32_t my_col;
    ALWI uint32_t idx(uint32_t c, uint32_t t) const {
        if constexpr (own_local) {
            if (c == my_col) {
                return t;
            }
        }
        return c * slot_tiles + t;
    }
    ALWI void copy(uint32_t c, uint32_t t, uint32_t dst) const {
        if (own_local && c == my_col) {
            copy_tile(cb_own, t, dst);
        } else {
            copy_tile(cb_slots, c * slot_tiles + t, dst);
        }
    }
    // DEST[dst] += slot c + slot c + 1 (acc_to_dest add_init).
    ALWI void add_pair(uint32_t c, uint32_t t, uint32_t dst) const {
        if (own_local && c == my_col) {
            add_tiles(cb_own, cb_slots, t, (c + 1) * slot_tiles + t, dst);
        } else if (own_local && c + 1 == my_col) {
            add_tiles(cb_slots, cb_own, c * slot_tiles + t, t, dst);
        } else {
            add_tiles(cb_slots, cb_slots, c * slot_tiles + t, (c + 1) * slot_tiles + t, dst);
        }
    }
};

// DEST[i] (+)= slot a + row-broadcast bias tile n, acc_to_dest.
template <uint32_t cb_bias>
ALWI void bcast_add_acc_init(uint32_t cb_a) {
    UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(cb_a, cb_bias)));
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::ROW, MathFidelity::LoFi>(
        cb_a, cb_bias, 1 /*acc_to_dest*/)));
}
template <uint32_t cb_bias, uint32_t cb_a>
ALWI void bcast_add_acc(uint32_t ia, uint32_t n, uint32_t dst) {
    UNPACK((llk_unpack_AB<BroadcastType::ROW>(cb_a, cb_bias, ia, n)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(cb_a, cb_bias, dst, false /*clear_fp32_dst_acc*/)));
}

// DEST[i] = SrcA(DEST[i]) + row-broadcast bias tile n (dest reuse, DEST -> SrcA).
template <uint32_t cb_bias>
ALWI void bias_reuse_init() {
    UNPACK((llk_unpack_A_init<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(0, 0, cb_bias)));
    MATH((llk_math_eltwise_binary_init<
          EltwiseBinaryType::ELWADD,
          BroadcastType::ROW,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_bias, cb_bias, 0)));
}
template <uint32_t cb_bias>
ALWI void bias_reuse(uint32_t n, uint32_t dst) {
    UNPACK((llk_unpack_A<BroadcastType::ROW, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_bias, n)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::ROW,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_bias, cb_bias, dst, true)));
}

// One M-block's combine: every slot (and the own slice) is resident; the output block is [bm, out_n] row-major tiles,
// slot_tiles of them. Packs into cb_out at the block's write pointer (caller reserves / pushes).
template <
    bool own_local,
    bool has_bias,
    uint32_t k_cols,
    uint32_t slot_tiles,
    uint32_t out_n_tiles,
    uint32_t cb_slots,
    uint32_t cb_own,
    uint32_t cb_bias,
    uint32_t cb_out>
ALWI void fused_combine(uint32_t my_col) {
    constexpr uint32_t G = compute_kernel_lib::DEST_AUTO_LIMIT;
    static_assert(G >= 1);
    const Operands<own_local, cb_slots, cb_own, slot_tiles> op{my_col};
    reconfig_data_format(cb_slots, cb_slots);
    pack_reconfig_data_format(cb_out);
    for (uint32_t t0 = 0; t0 < slot_tiles; t0 += G) {
        const uint32_t g = (slot_tiles - t0) < G ? (slot_tiles - t0) : G;
        tile_regs_acquire();
        if constexpr (exact || !has_bias) {
            // The committed order: odd k_cols seeds DEST with slot 0, then pairwise acc_to_dest adds.
            uint32_t c0 = 0;
            if constexpr (k_cols & 1u) {
                copy_init(cb_slots);
                for (uint32_t i = 0; i < g; ++i) {
                    op.copy(0, t0 + i, i);
                }
                c0 = 1;
            }
            if constexpr (k_cols > 1) {
                add_init(cb_slots, cb_slots, true);
                for (uint32_t i = 0; i < g; ++i) {
                    for (uint32_t c = c0; c < k_cols; c += 2) {
                        op.add_pair(c, t0 + i, i);
                    }
                }
            }
            if constexpr (has_bias) {
                reconfig_data_format_srcb(cb_slots, cb_bias);
                bias_reuse_init<cb_bias>();
                for (uint32_t i = 0; i < g; ++i) {
                    bias_reuse<cb_bias>((t0 + i) % out_n_tiles, i);
                }
                reconfig_data_format_srcb(cb_bias, cb_slots);
            }
        } else {
            // Bias folded into the first pair: DEST never leaves fp32 until the output pack.
            uint32_t c_bias = 0;
            if constexpr ((k_cols & 1u) == 0) {
                copy_init(cb_slots);
                for (uint32_t i = 0; i < g; ++i) {
                    op.copy(0, t0 + i, i);
                }
                c_bias = 1;
            }
            reconfig_data_format_srcb(cb_slots, cb_bias);
            bcast_add_acc_init<cb_bias>(cb_slots);
            for (uint32_t i = 0; i < g; ++i) {
                if (own_local && c_bias == my_col) {
                    bcast_add_acc<cb_bias, cb_own>(op.idx(c_bias, t0 + i), (t0 + i) % out_n_tiles, i);
                } else {
                    bcast_add_acc<cb_bias, cb_slots>(op.idx(c_bias, t0 + i), (t0 + i) % out_n_tiles, i);
                }
            }
            reconfig_data_format_srcb(cb_bias, cb_slots);
            if constexpr (k_cols > 2) {
                add_init(cb_slots, cb_slots, true);
                for (uint32_t i = 0; i < g; ++i) {
                    for (uint32_t c = c_bias + 1; c < k_cols; c += 2) {
                        op.add_pair(c, t0 + i, i);
                    }
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < g; ++i) {
            pack_tile<true>(i, cb_out, t0 + i);
        }
        tile_regs_release();
    }
}

}  // namespace conv2d_3x3::combine_tail
