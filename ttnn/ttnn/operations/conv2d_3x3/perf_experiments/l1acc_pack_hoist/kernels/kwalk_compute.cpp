// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated K-blocked fp32 packer-L1-accumulation matmul walk (single core, compute only).
// impl 0 = compute_kernel_lib::matmul_block exactly as conv2d_3x3 instantiates it (baseline);
// impl 1 = kwalk::matmul_block_l1acc_hoisted, per-tile packs (pack-state hoisted per K-block);
// impl 2 = same + one multi-tile pack MOP per subblock row (RAW LLK, see kwalk_hoisted.hpp).
// The PreKBlockFn pushes this K-block's in0/in1 (the op's tilize/writer stand-in) and, when dirty_hook,
// leaves the packer the way the op's tilize hook does (l1-acc off, bf8 format, 1-tile default MOP),
// so every variant must restore its own pack state after it.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "kwalk_hoisted.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t CB_IN0 = 0;
constexpr uint32_t CB_IN1 = 1;
constexpr uint32_t CB_PARTIALS = 5;
constexpr uint32_t CB_OUT = 16;

template <uint32_t in0_blk, uint32_t in1_blk, bool dirty>
struct PushAndDirtyHook {
    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool /*is_last*/) const {
        cb_reserve_back(CB_IN0, in0_blk);
        cb_push_back(CB_IN0, in0_blk);
        cb_reserve_back(CB_IN1, in1_blk);
        cb_push_back(CB_IN1, in1_blk);
        if constexpr (dirty) {
            pack_reconfig_l1_acc(0);
            pack_reconfig_data_format(CB_OUT);
            PACK((llk_pack_init(CB_OUT)));
        }
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t bm = get_compile_time_arg_val(0);
    constexpr uint32_t bk = get_compile_time_arg_val(1);
    constexpr uint32_t bn = get_compile_time_arg_val(2);
    constexpr uint32_t nkb = get_compile_time_arg_val(3);
    constexpr uint32_t sbh = get_compile_time_arg_val(4);
    constexpr uint32_t sbw = get_compile_time_arg_val(5);
    constexpr bool caller_owns = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t target_code = get_compile_time_arg_val(7);
    constexpr uint32_t impl = get_compile_time_arg_val(8);
    constexpr uint32_t m_iters = get_compile_time_arg_val(9);
    constexpr bool dirty_hook = get_compile_time_arg_val(10) != 0;
    constexpr bool force_no_l1_acc = get_compile_time_arg_val(11) != 0;
    constexpr bool keep_fast_drain = get_compile_time_arg_val(12) != 0;

    constexpr ckl::LastBlockTarget target =
        target_code == 0 ? ckl::LastBlockTarget::Interm
                         : (target_code == 1 ? ckl::LastBlockTarget::Out : ckl::LastBlockTarget::OutWithRelu);
    static_assert(bm % sbh == 0 && bn % sbw == 0);
    constexpr auto shape = ckl::MatmulBlockShape::of(bm / sbh, bn / sbw, sbh, sbw, bk, nkb);
    constexpr uint32_t out_tiles = bm * bn;

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_IN1, CB_PARTIALS);

    CircularBuffer in0(CB_IN0);
    CircularBuffer in1(CB_IN1);
    CircularBuffer partials(CB_PARTIALS);
    CircularBuffer out(CB_OUT);
    using Hook = PushAndDirtyHook<bm * bk, bk * bn, dirty_hook>;
    const Hook hook{};

    for (uint32_t m = 0; m < m_iters; ++m) {
        if constexpr (caller_owns) {
            partials.reserve_back(out_tiles);
        }
        if constexpr (impl == 0) {
            ckl::matmul_block<
                /*transpose=*/false,
                /*packer_l1_acc=*/true,
                target,
                ckl::OutputCBLayout::TileRowMajor,
                ckl::matmul_config::InitMode::ShortAfterPreKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::InputPolicy::WaitAndPopPerKBlock,
                ckl::NoPostCompute,
                Hook,
                ckl::NoPostKBlock,
                /*untilize_block_ct_dim=*/0,
                ckl::NoKBlockInnerDimFn,
                ckl::NoIn0Source,
                ckl::NoIn1BaseOffset,
                caller_owns>(in0, in1, out, partials, shape, {}, hook, /*in1_per_core_w=*/bn, /*out_row_width=*/bn);
        } else {
            kwalk::matmul_block_l1acc_hoisted<
                caller_owns,
                target,
                impl == 2 ? kwalk::PackPath::RowMop : kwalk::PackPath::PerTile,
                force_no_l1_acc,
                keep_fast_drain>(in0, in1, out, partials, shape, hook, bn, bn);
        }
        if constexpr (caller_owns) {
            partials.push_back(out_tiles);
        }
        // Consume the M-block's result (the op's bias add / writer stand-in); data stays in the shard.
        if constexpr (target == ckl::LastBlockTarget::Interm) {
            partials.wait_front(out_tiles);
            partials.pop_front(out_tiles);
        } else {
            out.wait_front(out_tiles);
            out.pop_front(out_tiles);
        }
        // What the op does after the walk (bias add prologue / end of M-block).
        pack_reconfig_l1_acc(0);
        if constexpr (target == ckl::LastBlockTarget::OutWithRelu) {
            pack_relu_config(ReluConfig::none());
        }
    }
}
