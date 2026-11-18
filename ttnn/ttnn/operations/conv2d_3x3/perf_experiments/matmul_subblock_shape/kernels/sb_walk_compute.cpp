// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated conv2d_3x3 K-blocked matmul walk (single core, compute only): fp32 packer-L1 accumulation into a
// [bm, bn] Interm partials CB, one PreKBlockFn per K-block. Bench-only (perf_experiments/matmul_subblock_shape).
//
// impl 0 = compute_kernel_lib::matmul_block exactly as conv2d_3x3 instantiates it on the K-blocked walk
//          (Interm, packer_l1_acc, TileRowMajor, ShortAfterPreKBlock, WaitAndPopPerKBlock x2), uniform
//          out_subblock (sbh x sbw).
// impl 1 = raw row-group walk: the M-block is split into up to three row groups, each with its own subblock
//          shape (h x w), so a bm that no 4-tall subblock divides (bm 5) can still run 4x1 subblocks on most rows.
//
//   RAW LLK / HELPER BYPASS (why impl 1 exists): matmul_block takes ONE MatmulBlockShape per call and runs the
//   whole K walk inside the call, so it cannot give rows 0..3 a 4x1 subblock and row 4 a 1x4 one while sharing the
//   K-blocks' L1 accumulation. Calling it once per row group is not equivalent either: each call would wait/pop the
//   whole in0/in1 K-block and run its own K loop. impl 1 re-issues ckernel::matmul_block_init when the subblock
//   shape changes (the unpack reuse direction, reuse_a = ct_dim >= rt_dim, and the math MOP depend on it) and
//   otherwise performs the helper's exact LLK sequence (same per-tile matmul calls in the same K order, same
//   per-tile packs to the same TileRowMajor offsets), so the output is expected bit-identical.
//   Pack-state (format + L1-acc enable) is issued once per K-block (constant within it).
//
// tilize = 1: in0 arrives as row-major bf16 im2col rows and is tilized per K-block in the PreKBlockFn, exactly like
// the op's TilizeIm2colKBlock (one K-block = bm blocks of [32, bk*32]); otherwise the hook only pushes resident
// bf8 in0 tiles. in1 (the weights) is always pushed by the hook (the op's weight-writer stand-in).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t CB_IN0 = 0;  // bf8 tiles (resident when !tilize; the tilize target otherwise)
constexpr uint32_t CB_IN1 = 1;  // bf8 weight tiles (resident)
constexpr uint32_t CB_RM = 2;   // bf16 row-major im2col rows (resident; tilize only)
constexpr uint32_t CB_PARTIALS = 5;

template <uint32_t bm, uint32_t bk, uint32_t bn, bool tilize>
struct Hook {
    DataflowBuffer* rm;
    DataflowBuffer* tiles;
    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool /*is_last*/) const {
        cb_reserve_back(CB_IN1, bk * bn);
        cb_push_back(CB_IN1, bk * bn);
        if constexpr (tilize) {
            cb_reserve_back(CB_RM, bm * bk);
            cb_push_back(CB_RM, bm * bk);
            // The op's TilizeIm2colKBlock (no ReLU on the Interm target).
            pack_reconfig_l1_acc(0);
            ckl::tilize<bk, CB_RM, CB_IN0>(*rm, *tiles, bm);
        } else {
            cb_reserve_back(CB_IN0, bm * bk);
            cb_push_back(CB_IN0, bm * bk);
        }
    }
};

struct Group {
    uint32_t rows, h, w;
};

// impl 1: raw row-group K walk, Interm target, packer-L1 accumulation over K-blocks.
// own: the partials are reserved once before the walk and pushed once after it (no per-K-block push + drain);
// the per-K-block llk_pack_reconfig_l1_acc's STALLWAIT(CFG, PACK) plus an explicit packer drain order K-block k's
// L1-acc packs before K-block k+1's.
// init_once: matmul state set once per M-block (only legal when the hook does not dirty it, i.e. no tilize, and
// one group).
template <
    uint32_t bm,
    uint32_t bk,
    uint32_t bn,
    uint32_t nkb,
    uint32_t ngroups,
    bool own,
    bool init_once,
    typename HookT>
ALWI void raw_group_walk(const Group (&groups)[3], const HookT& hook) {
    static_assert(!init_once || ngroups == 1);
    if constexpr (own) {
        cb_reserve_back(CB_PARTIALS, bm * bn);
    }
    for (uint32_t block = 0; block < nkb; ++block) {
        const bool last = block == nkb - 1;
        hook(block, nkb, last);
        if (!init_once || block == 0) {
            reconfig_data_format(CB_IN1, CB_IN0);
            PACK((pack_reconfig_data_format(CB_PARTIALS)));
        }
        if constexpr (own) {
            PACK((TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK)));
        }
        PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
        cb_wait_front(CB_IN0, bm * bk);
        cb_wait_front(CB_IN1, bk * bn);
        if constexpr (!own) {
            cb_reserve_back(CB_PARTIALS, bm * bn);
        }

        uint32_t row0 = 0;
        for (uint32_t g = 0; g < ngroups; ++g) {
            const uint32_t h = groups[g].h;
            const uint32_t w = groups[g].w;
            if (!init_once || block == 0) {
                ckernel::matmul_block_init(CB_IN0, CB_IN1, false, w, h, bk);
            }
            for (uint32_t r = 0; r < groups[g].rows; r += h, row0 += h) {
                for (uint32_t c = 0; c < bn; c += w) {
                    tile_regs_acquire();
                    uint32_t in0_index = row0 * bk;
                    uint32_t in1_index = c;
                    for (uint32_t k = 0; k < bk; ++k) {
#ifndef SKIP_COMPUTE
                        ckernel::matmul_block(CB_IN0, CB_IN1, in0_index, in1_index, 0, false, w, h, bk);
#endif
                        in0_index++;
                        in1_index += bn;
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    ckl::pack_subblock_row_strided(0, CB_PARTIALS, row0 * bn + c, bn, h, w);
                    tile_regs_release();
                }
            }
        }

        if constexpr (!own) {
            cb_push_back(CB_PARTIALS, bm * bn);
            if (!last) {
                cb_wait_front(CB_PARTIALS, bm * bn);
                cb_pop_front(CB_PARTIALS, bm * bn);
            }
        }
        cb_pop_front(CB_IN0, bm * bk);
        cb_pop_front(CB_IN1, bk * bn);
    }
    if constexpr (own) {
        cb_push_back(CB_PARTIALS, bm * bn);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t bm = get_compile_time_arg_val(0);
    constexpr uint32_t bk = get_compile_time_arg_val(1);
    constexpr uint32_t bn = get_compile_time_arg_val(2);
    constexpr uint32_t nkb = get_compile_time_arg_val(3);
    constexpr uint32_t m_iters = get_compile_time_arg_val(4);
    constexpr uint32_t impl = get_compile_time_arg_val(5);
    constexpr uint32_t sbh = get_compile_time_arg_val(6);
    constexpr uint32_t sbw = get_compile_time_arg_val(7);
    constexpr uint32_t ngroups = get_compile_time_arg_val(8);
    constexpr Group groups[3] = {
        {get_compile_time_arg_val(9), get_compile_time_arg_val(10), get_compile_time_arg_val(11)},
        {get_compile_time_arg_val(12), get_compile_time_arg_val(13), get_compile_time_arg_val(14)},
        {get_compile_time_arg_val(15), get_compile_time_arg_val(16), get_compile_time_arg_val(17)},
    };
    constexpr bool tilize = get_compile_time_arg_val(18) != 0;
    constexpr uint32_t flags = get_compile_time_arg_val(19);
    constexpr bool own = (flags & 1) != 0;
    constexpr bool init_once = (flags & 2) != 0;

    static_assert(impl == 1 || (bm % sbh == 0 && bn % sbw == 0 && sbh * sbw <= 4));
    static_assert(ngroups >= 1 && ngroups <= 3);

    if constexpr (tilize) {
        compute_kernel_hw_startup(CB_RM, CB_IN0);
    } else {
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_IN1, CB_PARTIALS);
    }

    DataflowBuffer rm_dfb(CB_RM);
    DataflowBuffer tiles_dfb(CB_IN0);
    CircularBuffer in0(CB_IN0);
    CircularBuffer in1(CB_IN1);
    CircularBuffer partials(CB_PARTIALS);
    using HookT = Hook<bm, bk, bn, tilize>;
    const HookT hook{&rm_dfb, &tiles_dfb};

    for (uint32_t m = 0; m < m_iters; ++m) {
        if constexpr (impl == 2) {
            // The helper's caller-owned pack target (the op's K-split walk uses it): one reserve / push per M-block.
            partials.reserve_back(bm * bn);
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/true,
                ckl::LastBlockTarget::Interm,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::NoPostCompute,
                HookT,
                ckl::NoPostKBlock,
                /*untilize_block_ct_dim=*/0,
                ckl::NoKBlockInnerDimFn,
                ckl::NoIn0Source,
                ckl::NoIn1BaseOffset,
                /*caller_owns_pack_target=*/true>(
                in0,
                in1,
                partials,
                partials,
                ckl::MatmulBlockShape::of(bm / sbh, bn / sbw, sbh, sbw, bk, nkb),
                {},
                hook,
                /*in1_per_core_w=*/bn,
                /*out_row_width=*/bn);
            partials.push_back(bm * bn);
        } else if constexpr (impl == 0) {
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/true,
                ckl::LastBlockTarget::Interm,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::NoPostCompute,
                HookT>(
                in0,
                in1,
                partials,
                partials,
                ckl::MatmulBlockShape::of(bm / sbh, bn / sbw, sbh, sbw, bk, nkb),
                {},
                hook,
                /*in1_per_core_w=*/bn,
                /*out_row_width=*/bn);
        } else {
            raw_group_walk<bm, bk, bn, nkb, ngroups, own, init_once>(groups, hook);
        }
        // The op's bias add stands here: consume the M-block's partials (they stay in the shard for the check).
        partials.wait_front(bm * bn);
        partials.pop_front(bm * bn);
        pack_reconfig_l1_acc(0);
    }
}
