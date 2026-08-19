// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_nhwc compute (TRISC) — tilize -> blocked matmul -> (bias) -> untilize.
//
// The implicit im2col rows arrive in cb_act_rm as row-major sticks. Tilizing
// them is the matmul's per-K-block preprocessing step, so it runs as
// matmul_block's PreKBlockFn with InitMode::ShortAfterPreKBlock — the matmul
// helper owns the state restore after the tilize (contract (A) in
// matmul_block_helpers.hpp), so the functor does nothing but its own tilize.
//
// Advisory deviation from op_design.md: the design's API table names the
// deprecated mm_block_init() for the boot-time init. That symbol is marked
// [[deprecated]] in api/compute/matmul.h in favour of
// compute_kernel_hw_startup<SrcOrder::Reverse>() + matmul_block_init(), which
// is what this kernel uses. Semantics are identical.
//
// Refinement 5 — K-accumulation precision. matmul_block's default (software)
// K-accumulation spills each partial block to cb_partials and RELOADS it into
// DEST through copy_block_matmul_partials, i.e. through SrcA, which carries
// only ~11 mantissa bits on Wormhole. That reload truncates the running sum
// once per K-block, and because truncation is biased the error accumulates
// LINEARLY in num_k_blocks (measured: rel_rms = 0.0013 + 1.9e-4 * Kt, exact fit
// over Kt = 1..121). `packer_l1_acc` replaces the spill/reload with a hardware
// fp32 accumulate in the packer, so no partial sum ever round-trips through
// SrcA and only the per-product SrcA/SrcB floor remains.
//
// Two pieces of packer state the caller owns when packer_l1_acc is on (the
// helper arms L1_ACC before each of its OWN packs but never disarms it):
//   * the tilize inside PreKBlockFn packs into cb_act_tiles and must OVERWRITE
//     — it runs before the helper re-arms, so it clears L1_ACC first;
//   * after matmul_block returns with LastBlockTarget::Interm, L1_ACC is still
//     latched on, so the bias / untilize phases would accumulate too.
// Both mirror bmm_large_block_zm_fused_bias_activation.cpp's PACKER_L1_ACC path.

#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {

constexpr uint32_t cb_act_rm = 0;
constexpr uint32_t cb_weight_tiles = 1;
constexpr uint32_t cb_bias_tiles = 2;
constexpr uint32_t cb_out_rm = 16;
constexpr uint32_t cb_act_tiles = 24;
constexpr uint32_t cb_partials = 25;
constexpr uint32_t cb_mm_out = 26;

// M is zero-padded up to a whole number of `Mt` tile-rows per m_block; the
// number of m_blocks *this core* owns is a runtime arg (the grid split).
constexpr uint32_t Mt = get_compile_time_arg_val(0);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(1);
constexpr uint32_t num_k_blocks = get_compile_time_arg_val(2);
constexpr uint32_t Kb = get_compile_time_arg_val(3);
constexpr uint32_t Nt_b = get_compile_time_arg_val(4);
constexpr uint32_t out_subblock_w = get_compile_time_arg_val(5);
constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(6);
constexpr bool fuse_bias = get_compile_time_arg_val(7) == 1;
// Hardware fp32 K-accumulation in the packer instead of the software
// spill/reload through SrcA. Decided host-side (see the descriptor's
// `packer_l1_acc_en`) — it needs an fp32 cb_partials sized to EXACTLY one
// output block so the CB's write pointer wraps back onto the same L1 region
// every K-block.
constexpr bool packer_l1_acc = get_compile_time_arg_val(8) == 1;

constexpr uint32_t TILE_H = 32;

// PreKBlockFn: tilize this K-block's im2col rows out of cb_act_rm.
// Per the MATMUL-STATE-RESTORE CONTRACT (pattern A) this functor must NOT
// restore matmul state — InitMode::ShortAfterPreKBlock makes the helper do it.
struct TilizeActBlock {
    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool /*is_last*/) const {
        if constexpr (packer_l1_acc) {
            // The previous K-block's pack left L1_ACC armed; this tilize must
            // overwrite cb_act_tiles, not accumulate into it. Safe to clear —
            // matmul_block re-arms L1_ACC before every one of its own packs.
            pack_reconfig_l1_acc(0);
        }
        compute_kernel_lib::tilize<Kb, cb_act_rm, cb_act_tiles>(Mt, Mt * TILE_H);
    }
};

}  // namespace

void kernel_main() {
    using namespace compute_kernel_lib;

    // Per-core M-block count (grid split). Compute never needs the *start*
    // index — it consumes from CBs, and the reader/writer own the addressing.
    const uint32_t num_m_blocks_here = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_mm_out);
    matmul_block_init(cb_act_tiles, cb_weight_tiles, /*transpose=*/0, out_subblock_w, /*rt_dim=*/1, Kb);

    CircularBuffer act_buf(cb_act_tiles);
    CircularBuffer weight_buf(cb_weight_tiles);
    CircularBuffer bias_buf(cb_bias_tiles);
    CircularBuffer mm_out_buf(cb_mm_out);
    CircularBuffer partials_buf(cb_partials);

    constexpr LastBlockTarget mm_target = fuse_bias ? LastBlockTarget::Interm : LastBlockTarget::Out;

    // Every m_block is a FULL Mt tile-rows (the reader zero-pads the tail), so every CB
    // transaction stays size-aligned — metal CBs cannot straddle fifo_limit.
    for (uint32_t m_block = 0; m_block < num_m_blocks_here; ++m_block) {
        for (uint32_t n_block = 0; n_block < num_n_blocks; ++n_block) {
            matmul_block<
                /*transpose=*/false,
                packer_l1_acc,
                mm_target,
                OutputCBLayout::SubblockMajor,
                matmul_config::InitMode::ShortAfterPreKBlock,
                InputPolicy::WaitAndPopPerKBlock,
                InputPolicy::WaitAndPopPerKBlock,
                NoPostCompute,
                TilizeActBlock>(
                act_buf,
                weight_buf,
                mm_out_buf,
                partials_buf,
                MatmulBlockShape::of(Mt, in1_num_subblocks, /*out_subblock_h=*/1, out_subblock_w, Kb, num_k_blocks),
                /*post_compute=*/{},
                TilizeActBlock{});

            if constexpr (packer_l1_acc) {
                // LastBlockTarget::Interm leaves L1_ACC armed (the last K-block
                // accumulates in place); Out disarms it. Clear unconditionally
                // so the bias add and the untilize below always overwrite.
                pack_reconfig_l1_acc(0);
            }

            if constexpr (fuse_bias) {
                bias_buf.wait_front(Nt_b);
                add_bias_bcast_rows<BiasBroadcast::RowBroadcast, OutputCBLayout::SubblockMajor>(
                    partials_buf,
                    bias_buf,
                    mm_out_buf,
                    BiasAddShape::of(Mt, in1_num_subblocks, /*out_subblock_h=*/1, out_subblock_w));
                bias_buf.pop_front(Nt_b);
            }

            untilize<Nt_b, cb_mm_out, cb_out_rm>(Mt);
        }
    }
}
