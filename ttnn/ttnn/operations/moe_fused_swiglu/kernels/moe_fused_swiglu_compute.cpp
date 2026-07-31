// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — COMPUTE.
//
// Per M-block, on every one of the HGROUPS x KGROUPS worker cores:
//   1. fused tilize of the row-major x slice this core injects (bf16 path only);
//   2. gate matmul and up matmul over the SAME resident x block (one K-block == the whole
//      per-row K extent, so `in0_policy = WaitAndRetainOnLastBlock` retains it for both and the
//      kernel pops it once at the end — the "cb_x_tiles consumed twice" contract, and NOT a
//      second multicast of x);
//   3. the cross-column reduce adds (in-place FPU add per child), with the ROOT's final gate add
//      carrying SiLU on the PACKER thread, then the SwiGLU multiply through L1;
//   4. the `down` matmul over HGROUPS phase-2 K-blocks with packer L1 accumulation, then the one
//      genuine dtype boundary (bf16 partials -> bfp8 output).
//
// Everything here is a kernel_lib helper. The ONE raw access is the L1 mailbox read of the
// device-resident token count: the M-block trip count must be identical on all three TRISCs, and
// `cb_wait_front` in a compute kernel is UNPACK-only, so a CB handoff would let MATH/PACK diverge.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

#include "moe_fused_swiglu_common.hpp"  // the ONE definition of the mailbox word layout

using namespace compute_kernel_lib;

constexpr uint32_t M_BLOCK = get_compile_time_arg_val(0);
constexpr uint32_t KR_PAD = get_compile_time_arg_val(1);
constexpr uint32_t HN_PAD = get_compile_time_arg_val(2);
constexpr uint32_t EC_MAX = get_compile_time_arg_val(3);  // phase-2 N stride (uniform CB increment)
constexpr uint32_t HGROUPS = get_compile_time_arg_val(4);
constexpr uint32_t HID_T = get_compile_time_arg_val(5);
constexpr uint32_t INPUT_FORMAT = get_compile_time_arg_val(6);
constexpr uint32_t OUT_SUBBLOCK_H = get_compile_time_arg_val(7);
constexpr uint32_t MAILBOX_MAGIC = get_compile_time_arg_val(8);
// Smallest legal `m_eff` (= OUT_SUBBLOCK_H rounded up to a power of two, so `m_eff /
// OUT_SUBBLOCK_H` below is always exact). One host-side definition, identical in all three
// kernels — see m_tiles_eff().
constexpr uint32_t M_EFF_MIN = get_compile_time_arg_val(9);

constexpr uint32_t cb_x_in = get_compile_time_arg_val(10);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(11);
constexpr uint32_t cb_x_stage = get_compile_time_arg_val(12);
constexpr uint32_t cb_w_gate = get_compile_time_arg_val(13);
constexpr uint32_t cb_w_up = get_compile_time_arg_val(14);
constexpr uint32_t cb_w_down = get_compile_time_arg_val(15);
constexpr uint32_t cb_gate_acc = get_compile_time_arg_val(16);
constexpr uint32_t cb_up_acc = get_compile_time_arg_val(17);
constexpr uint32_t cb_gate_send = get_compile_time_arg_val(18);
constexpr uint32_t cb_up_send = get_compile_time_arg_val(19);
constexpr uint32_t cb_gate_silu = get_compile_time_arg_val(20);
constexpr uint32_t cb_reduce_gate_in = get_compile_time_arg_val(21);
constexpr uint32_t cb_reduce_up_in = get_compile_time_arg_val(22);
constexpr uint32_t cb_h_local = get_compile_time_arg_val(23);
constexpr uint32_t cb_h = get_compile_time_arg_val(24);
constexpr uint32_t cb_out_interm = get_compile_time_arg_val(25);
constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(26);

constexpr uint32_t TILE_H = 32;

// Per-K-block FMA step count for the gate/up matmul: the padded K slot is KR_PAD tiles wide but
// only `kr` of them are real, so the loop bound shrinks and the pad tiles are never touched.
struct KrSteps {
    uint32_t kr;
    ALWI uint32_t operator()(uint32_t, uint32_t) const { return kr; }
};

// Per-K-block FMA step count for the `down` matmul: every h round carries HN_PAD hidden tiles,
// except the last column-group, which owns fewer (HGROUPS * HN_PAD >= HID_T by construction).
struct HnSteps {
    uint32_t last;
    ALWI uint32_t operator()(uint32_t block, uint32_t block_k) const { return (block == HGROUPS - 1) ? last : block_k; }
};

void kernel_main() {
    const uint32_t mailbox_addr = get_arg_val<uint32_t>(0);
    const uint32_t kr = get_arg_val<uint32_t>(1);
    const uint32_t hn = get_arg_val<uint32_t>(2);
    const uint32_t ec = get_arg_val<uint32_t>(3);
    const uint32_t is_root = get_arg_val<uint32_t>(4);
    const uint32_t num_children = get_arg_val<uint32_t>(5);
    const uint32_t my_col = get_arg_val<uint32_t>(6);  // grid column == this core's x-injection slot

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_tiles, cb_w_gate, cb_gate_acc);
    // SiLU rides the packer thread of the root's final reduce add; the helpers never issue this.
    ActivationInitHelper<KernelActivation::SILU>::init();

    // Device-resident token count. All three TRISCs spin here independently so the M-block trip
    // count is thread-uniform (see the file header). The `fence` is exactly what
    // `invalidate_l1_cache()` compiles to on Blackhole (risc_common.h) — spelled out here because
    // that helper lives behind a dataflow-only include.
    volatile tt_l1_ptr uint32_t* mbox = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mailbox_addr);
    while (mbox[moe_fused_swiglu::MBOX_READY] != MAILBOX_MAGIC) {
        asm volatile("fence" ::: "memory");
    }
    const uint32_t m_t = mbox[moe_fused_swiglu::MBOX_M_T];
    const uint32_t m_blocks = mbox[moe_fused_swiglu::MBOX_M_BLOCKS];

    CircularBuffer x_buf(cb_x_tiles);
    CircularBuffer wg_buf(cb_w_gate);
    CircularBuffer wu_buf(cb_w_up);
    CircularBuffer wd_buf(cb_w_down);
    CircularBuffer gate_buf(cb_gate_acc);
    CircularBuffer up_buf(cb_up_acc);
    CircularBuffer rg_buf(cb_reduce_gate_in);
    CircularBuffer ru_buf(cb_reduce_up_in);
    CircularBuffer silu_buf(cb_gate_silu);
    CircularBuffer h_buf(cb_h);
    CircularBuffer out_interm_buf(cb_out_interm);
    CircularBuffer out_tiles_buf(cb_out_tiles);

    // The reduce slots are the ONE M-scaled pair that is always pushed WHOLE: the child unicasts
    // to its own cb_reduce_*_in write pointer as a proxy for the parent's, which only holds while
    // every push wraps back to the CB base. Live tokens occupy the first m_eff*HN_PAD tiles.
    constexpr uint32_t REDUCE_SLOT_TILES = M_BLOCK * HN_PAD;

    const uint32_t hn_last = HID_T - (HGROUPS - 1) * HN_PAD;

    for (uint32_t b = 0; b < m_blocks; ++b) {
        // The RUNTIME token tile-rows this block works on — the same number the reader uses for its
        // x-multicast rounds and the writer for its CB waits (moe_fused_swiglu_common.hpp). Every
        // shape and trip count below is derived from it, so count 128 does HALF the gate/up matmul,
        // reduce and `down` work of count 256 instead of the same amount.
        const uint32_t m_eff = moe_fused_swiglu::m_tiles_eff(m_t, b, M_BLOCK, M_EFF_MIN);
        const uint32_t x_slot_tiles = m_eff * KR_PAD;
        const uint32_t gu_block_tiles = m_eff * HN_PAD;
        const uint32_t out_block_tiles = m_eff * EC_MAX;

        // gate/up: [m_eff, HN_PAD] = x[m_eff, kr] @ W[kr, HN_PAD]. ONE K-block whose width is the
        // whole per-row K extent, which is what lets both matmuls read the same resident in0.
        MatmulBlockShape shape_gu = MatmulBlockShape::of(m_eff / OUT_SUBBLOCK_H, 1, OUT_SUBBLOCK_H, HN_PAD, KR_PAD, 1);
        shape_gu.last_in1_subblock_w_valid = (hn < HN_PAD) ? hn : 0;

        // down: [m_eff, ec] = h[m_eff, HGROUPS*HN_PAD] @ W_down[.., ec], HGROUPS K-blocks.
        // The FMA width is the real `ec`, but the in1 read stride and the output row stride are the
        // uniform EC_MAX so every phase-2 CB increment is core-independent.
        const MatmulBlockShape shape_dn =
            MatmulBlockShape::of(m_eff / OUT_SUBBLOCK_H, 1, OUT_SUBBLOCK_H, ec, HN_PAD, HGROUPS);

        // ---- 1. fused tilize of the x tile-rows this core injects (bf16 ROW_MAJOR only) ----
        if constexpr (INPUT_FORMAT == 0) {
            const uint32_t n_inject = moe_fused_swiglu::inject_rows(m_eff, my_col, HGROUPS);
            for (uint32_t i = 0; i < n_inject; ++i) {
                // Asymmetric page mode: TILE_H row-major stick slices in -> KR_PAD bfp8 tiles out.
                tilize<KR_PAD, cb_x_in, cb_x_stage>(1, TILE_H);
            }
        }

        // ---- 2. gate and up over the same resident x block ----
        matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            LastBlockTarget::Out,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndRetainOnLastBlock,
            InputPolicy::WaitAndPopPerKBlock,
            NoPostCompute,
            NoPreKBlock,
            NoPostKBlock,
            /*untilize_block_ct_dim=*/0,
            KrSteps>(x_buf, wg_buf, gate_buf, gate_buf, shape_gu, {}, {}, 0, 0, {}, KrSteps{kr});

        matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/false,
            LastBlockTarget::Out,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndRetainOnLastBlock,
            InputPolicy::WaitAndPopPerKBlock,
            NoPostCompute,
            NoPreKBlock,
            NoPostKBlock,
            /*untilize_block_ct_dim=*/0,
            KrSteps>(x_buf, wu_buf, up_buf, up_buf, shape_gu, {}, {}, 0, 0, {}, KrSteps{kr});

        // ---- 3. cross-column reduce + SwiGLU ----
        for (uint32_t c = 0; c < num_children; ++c) {
            const bool final_child = (c + 1 == num_children);
            if (is_root && final_child) {
                // Root's last gate add: SiLU is fused on the PACKER thread, so the activation
                // overlaps the math thread instead of costing a separate SFPU pass.
                //
                // One call per token tile-row: the helper's bias index does not advance with
                // in0_subblock (bias_add_helpers.inl:141), so an Elementwise bias spanning
                // M_BLOCK tile-rows is walked with bias_offset instead, one M-row per call.
                // The slot arrives WHOLE (see REDUCE_SLOT_TILES) but only its first m_eff tile-rows
                // carry live tokens, so the bias walk stops at m_eff and the tail is dropped.
                rg_buf.wait_front(REDUCE_SLOT_TILES);
                for (uint32_t m = 0; m < m_eff; ++m) {
                    add_bias_bcast_rows<
                        BiasBroadcast::Elementwise,
                        OutputCBLayout::SubblockMajor,
                        bias_add_config::NoPostBias,
                        SiluActivation>(
                        gate_buf, rg_buf, silu_buf, BiasAddShape::of(1, 1, OUT_SUBBLOCK_H, HN_PAD), {}, m * HN_PAD);
                }
                rg_buf.pop_front(REDUCE_SLOT_TILES);
            } else {
                add<input(cb_gate_acc), input(cb_reduce_gate_in), output(cb_gate_acc)>(
                    EltwiseShape::tiles(gu_block_tiles));
                if (gu_block_tiles < REDUCE_SLOT_TILES) {
                    rg_buf.pop_front(REDUCE_SLOT_TILES - gu_block_tiles);
                }
            }
            add<input(cb_up_acc), input(cb_reduce_up_in), output(cb_up_acc)>(EltwiseShape::tiles(gu_block_tiles));
            if (gu_block_tiles < REDUCE_SLOT_TILES) {
                ru_buf.pop_front(REDUCE_SLOT_TILES - gu_block_tiles);
            }
        }

        if (is_root) {
            // FPU multiply through L1 (deliberately not SFPU and not DEST-reuse — the L1
            // round-trip measured faster for an FPU consumer, examples/compute_fusion).
            mul<input(cb_gate_silu), input(cb_up_acc), output(cb_h_local)>(EltwiseShape::tiles(gu_block_tiles));
        } else {
            copy<input(cb_gate_acc), output(cb_gate_send)>(EltwiseShape::tiles(gu_block_tiles));
            copy<input(cb_up_acc), output(cb_up_send)>(EltwiseShape::tiles(gu_block_tiles));
        }

        // ---- 4. down matmul: HGROUPS K-blocks, packer L1 accumulation into a caller-owned
        // interm region (so every K-block accumulates at the SAME L1 address) ----
        out_interm_buf.reserve_back(out_block_tiles);
        matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/true,
            LastBlockTarget::Interm,
            OutputCBLayout::TileRowMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndPopPerKBlock,
            InputPolicy::WaitAndPopPerKBlock,
            NoPostCompute,
            NoPreKBlock,
            NoPostKBlock,
            /*untilize_block_ct_dim=*/0,
            HnSteps,
            NoIn0Source,
            NoIn1BaseOffset,
            /*caller_owns_pack_target=*/true>(
            h_buf,
            wd_buf,
            out_tiles_buf,
            out_interm_buf,
            shape_dn,
            {},
            {},
            /*in1_per_core_w=*/EC_MAX,
            /*out_row_width=*/EC_MAX,
            {},
            HnSteps{hn_last});
        out_interm_buf.push_back(out_block_tiles);
        // matmul_block leaves packer L1 accumulation ENABLED after its last K-block, and neither
        // the eltwise chain (L1Accumulation::Disabled is a compile-time no-op) nor a
        // packer_l1_acc=false matmul resets it. Without this the copy below — and the next
        // M-block's gate matmul — would ACCUMULATE onto stale L1 instead of overwriting.
        pack_reconfig_l1_acc(0);

        // The one genuine dtype boundary: bf16 accumulation -> bfp8 output tiles.
        copy<input(cb_out_interm), output(cb_out_tiles)>(EltwiseShape::tiles(out_block_tiles));

        // The resident x block was retained by both matmuls; release it now.
        x_buf.pop_front(x_slot_tiles);
    }
}
