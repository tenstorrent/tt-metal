#!/usr/bin/env python3
"""passb_fusion — generate the fused pass-B compute-kernel variants.

Baseline (k_base) is the shipped compute kernel, byte-for-byte.
k_fuse      : normalize x gamma [x bias] fused into ONE DEST window (one pack/tile).
k_fuse_ng   : normalize x gamma fused, bias left as its own pass (option (a) alone).

Only the compute kernel differs; reader/writer are the shipped ones.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "k_base" / "rms_norm_ttnn_compute.cpp"

HELPERS = r"""
// ============================================================================
// passb_fusion (perf experiment) — RAW LLK: a dest-reuse binary WITH a broadcast
// ============================================================================
// HELPER BYPASSED: `ckl::DestReuseBinary` (kernel_lib/eltwise/core/chain.hpp:526)
// and, with it, the metal wrapper pair `{mul,add}_reuse_dest_{init,tiles}`
// (api/compute/eltwise_binary.h:107-202 / :403-430).
//
// CLASS: **capability**, not ergonomics.
//   `DestReuseBinary<Input, Op, ReuseType, Dst>` takes a plain `InputSpec`, which
//   carries no `BroadcastDim` — unlike `BinaryFpu`, whose B side takes a
//   `BinaryFpuInputSpec` (`input(cb, BroadcastDim::Row)`).  It therefore always
//   emits `llk_unpack_A<BroadcastType::NONE, …>` +
//   `llk_math_eltwise_binary<…, BroadcastType::NONE, …>`, i.e. a FULL-tile second
//   operand.  Pass B's second and third operands are a 1xW gamma and a 1xW bias,
//   which are row-shaped and MUST broadcast down rows, so the fused chain
//   `((x * stat<Col>) * gamma<Row>) + bias<Row>` is inexpressible with the helper.
//
// WHAT IS MISSING is exactly ONE template parameter, not a hardware path:
//   * `llk_unpack_A` is already templated on `BroadcastType` AND on
//     `EltwiseBinaryReuseDestType`, and its ROW arm
//     (tt_llk_blackhole/llk_lib/llk_unpack_A.h, `_llk_unpack_A_mop_config_`) is
//     structurally identical to `llk_unpack_AB`'s ROW arm — same
//     (num_faces_r_dim x num_faces_c_dim) MOP over `unpack_srcb`, same
//     `srcb_clear_z` end-op — with the srcA slot replaced by the DEST-reuse dummy
//     `UNPACR_NOP(SrcA, …, SET_DVALID, UNP_ZEROSRC)`.  Its init comment states the
//     routing outright: "acc_to_dest DEST_TO_SRCA -> SrcB only (SrcA comes from
//     DEST)" and "broadcast -> SrcB".
//   * `_llk_math_eltwise_binary_with_dest_reuse_` has explicit COL / ROW / SCALAR
//     arms (llk_math_eltwise_binary.h:505+).
//   So the hardware does `DEST (op) bcast(CB)` natively; only the kernel_lib
//   surface refuses to ask for it.  The fix upstream is to let
//   `DestReuseBinary` take a `BinaryFpuInputSpec` exactly as `BinaryFpu`'s B side
//   does.
//
// WHY DEST_TO_SRCA IS FORCED: a broadcast operand always lands in SrcB, and
// `_llk_unpack_A_mop_config_` static_asserts
// `!(BType != NONE && acc_to_dest && reuse == DEST_TO_SRCB)`.  DEST therefore goes
// to SrcA.  Both operands here share the activation's format, so the mixed-dtype
// DEST_TO_SRCA caveat recorded at `ckl::DestReuseType` does not bite.
//
// PRECISION: strictly >= the unfused pair.  The unfused pair PACKS `x * stat`
// through cb_normalized (a bf16 CB) and unpacks it again; the fused window keeps
// it in DEST.  Same MATH_FIDELITY, same fp32_dest_acc_en, one fewer rounding.
namespace passb_fusion {

template <ckernel::EltwiseBinaryType OP, ckernel::BroadcastType BT>
ALWI void bcast_reuse_dest_init(uint32_t icb) {
    constexpr auto reuse = ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    UNPACK((llk_unpack_A_init<BT, /*acc_to_dest=*/true, reuse>(false, false, icb)));
    // MATH_FIDELITY is forwarded verbatim: llk_math_binary_api's
    // get_effective_math_fidelity<OP, F>() already forces LoFi for ELWADD/ELWSUB,
    // which is what the helper's own non-Mul path does.
    MATH((llk_math_eltwise_binary_init<OP, BT, MATH_FIDELITY, reuse>(icb, icb, /*acc_to_dest=*/0)));
}

template <ckernel::EltwiseBinaryType OP, ckernel::BroadcastType BT>
ALWI void bcast_reuse_dest_tiles(uint32_t icb, uint32_t itile, uint32_t idst) {
    constexpr auto reuse = ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    UNPACK((llk_unpack_A<BT, /*acc_to_dest=*/true, reuse>(icb, itile)));
    MATH((llk_math_eltwise_binary<OP, BT, DST_ACCUM_MODE, MATH_FIDELITY, reuse>(
        icb, icb, idst, true /* clear_fp32_dst_acc */)));
}

}  // namespace passb_fusion
"""

FUSED = r"""            // ================= passb_fusion: ONE DEST window ==================
            // Fuses `normalize_block` (x * stat<Col>) with `scale_block`
            // (* gamma<Row>){FUSED_BIAS_DOC} into a single DEST window: each tile is
            // unpacked once, multiplied {NMUL} times inside DEST, and packed ONCE
            // straight to cb_output_tiles.  cb_normalized is never touched.
            //
            // The synchronization is a hand-rolled transcription of the two (three)
            // chains it replaces: the same Upfront waits, the same PerBlockSize
            // reserve/push on the output, the same AtEnd pop of CB_T, the same
            // (rows x WT_CHUNK) walk at PASS_B_BLK DEST lanes, the same
            // element-major order inside a block.
            if constexpr (HAS_G || HAS_B) {
                MaybeDeviceZoneScope("compute_passb_fused");
                cb_wait_front(CB_T, hold_base + rows * WT_CHUNK);
                cb_wait_front(CB_STAT_B, rows);
                if constexpr (HAS_G) {
                    cb_wait_front(cb_gamma_tiles, hold_base + WT_CHUNK);
                }
                if constexpr (FUSE_BIAS && HAS_B) {
                    cb_wait_front(cb_bias_tiles, hold_base + WT_CHUNK);
                }
                constexpr uint32_t FUSED_OUT = (FUSE_BIAS || !HAS_B) ? cb_output_tiles : cb_normalized;
                // MEASURED LLK EDGE (perf_experiments/passb_fusion): at
                // dst_index == MAX_TILES_IN_HALF_DEST-1 the dest-reuse runner's
                // per-face TT_ZEROACC does not clear DEST face
                // (DEST_AUTO_LIMIT-1)*4+3 -- the LAST face of the half-bank -- so
                // the accumulating ELWMUL MOP returns y = n*(g+1) there
                // (llk_math_eltwise_binary.h `eltwise_binary_run_with_dest_reuse`).
                // Reproduced on the focus shape at PASS_B_BLK=8, BIT-IDENTICAL to
                // the unfused pair at 4/2/1.  So the fused window never uses the
                // top DEST lane.
                constexpr uint32_t FBLK = (PASS_B_BLK >= ckl::DEST_AUTO_LIMIT) ? (PASS_B_BLK / 2) : PASS_B_BLK;
                static_assert(FBLK >= 1 && WT_CHUNK % FBLK == 0, "passb_fusion: FBLK must divide WT_CHUNK");
                pack_reconfig_data_format(FUSED_OUT);
                for (uint32_t ht = 0; ht < rows; ++ht) {
                    for (uint32_t wt = 0; wt < WT_CHUNK; wt += FBLK) {
                        cb_reserve_back(FUSED_OUT, FBLK);
                        reconfig_data_format(CB_T, CB_STAT_B);
                        mul_bcast_cols_init(CB_T, CB_STAT_B);
                        tile_regs_acquire();
                        for (uint32_t j = 0; j < FBLK; ++j) {
                            mul_tiles_bcast_cols(
                                CB_T, CB_STAT_B, hold_base + ht * WT_CHUNK + wt + j, ht, j);
                        }
                        if constexpr (HAS_G) {
                            reconfig_data_format_srcb(CB_STAT_B, cb_gamma_tiles);
                            passb_fusion::bcast_reuse_dest_init<
                                ckernel::EltwiseBinaryType::ELWMUL,
                                ckernel::BroadcastType::ROW>(cb_gamma_tiles);
                            for (uint32_t j = 0; j < FBLK; ++j) {
                                passb_fusion::bcast_reuse_dest_tiles<
                                    ckernel::EltwiseBinaryType::ELWMUL,
                                    ckernel::BroadcastType::ROW>(cb_gamma_tiles, hold_base + wt + j, j);
                            }
                        }
                        if constexpr (FUSE_BIAS && HAS_B) {
                            reconfig_data_format_srcb(HAS_G ? cb_gamma_tiles : CB_STAT_B, cb_bias_tiles);
                            passb_fusion::bcast_reuse_dest_init<
                                ckernel::EltwiseBinaryType::ELWADD,
                                ckernel::BroadcastType::ROW>(cb_bias_tiles);
                            for (uint32_t j = 0; j < FBLK; ++j) {
                                passb_fusion::bcast_reuse_dest_tiles<
                                    ckernel::EltwiseBinaryType::ELWADD,
                                    ckernel::BroadcastType::ROW>(cb_bias_tiles, hold_base + wt + j, j);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t j = 0; j < FBLK; ++j) {
                            pack_tile(j, FUSED_OUT);
                        }
                        tile_regs_release();
                        cb_push_back(FUSED_OUT, FBLK);
                    }
                }
                if constexpr (PASS_B_X_POP == ckl::PopPolicy::AtEnd) {
                    cb_pop_front(CB_T, rows * WT_CHUNK);
                }
                if constexpr (!FUSE_BIAS && HAS_B) {
                    // option (a) only: the SHIFT stays its own pass, reading the
                    // fused (normalize x gamma) result out of cb_normalized.
                    MaybeDeviceZoneScope("compute_bias_add");
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                        ckl::BinaryFpu<
                            ckl::BinaryFpuOp::Add,
                            ckl::input(
                                cb_normalized,
                                ckl::WaitPolicy::Upfront,
                                ckl::PopPolicy::AtEnd,
                                ckl::OperandKind::Block),
                            ckl::input(B_IN, ckl::BroadcastDim::Row)>{0u, hold_base},
                        ckl::PackTile<PASS_B_OUT_GAMMA>{});
                }
            } else {
"""


def build(fuse_bias: bool) -> str:
    text = SRC.read_text()
    lines = text.split("\n")

    # ---- 1. helpers + the bcast include, right after the `namespace ckl` alias
    anchor = "namespace ckl = compute_kernel_lib;"
    assert text.count(anchor) == 1
    text = text.replace(
        anchor,
        '#include "api/compute/bcast.h"  // mul_bcast_cols_init / mul_tiles_bcast_cols\n\n'
        + anchor
        + "\n"
        + HELPERS
        + f"\n// option (b) when 1, option (a) when 0.\nstatic constexpr bool FUSE_BIAS = {int(fuse_bias)};\n",
    )

    # ---- 2. wrap the three pass-B stages
    lines = text.split("\n")
    start = next(
        i
        for i, l in enumerate(lines)
        if l.strip() == "// x * (1/rms). The stat is a REDUCE_ROW result: column-shaped, so it"
    )
    # end == the closing brace of the `if constexpr (HAS_B)` block, i.e. the line
    # right before `            if constexpr (RM) {`
    end = next(i for i, l in enumerate(lines) if l.strip() == "if constexpr (RM) {" and i > start)
    while lines[end - 1].strip() == "":
        end -= 1
    assert lines[end - 1].strip() == "}", lines[end - 1]

    body = lines[start:end]
    doc = " and `bias_block` (+ bias<Row>)" if fuse_bias else ""
    nmul = "twice/thrice" if fuse_bias else "twice"
    head = FUSED.replace("{FUSED_BIAS_DOC}", doc).replace("{NMUL}", nmul)
    new = head.split("\n")[:-1] + body + ["            }"]
    out = lines[:start] + new + lines[end:]
    return "\n".join(out)


for name, fb in (("k_fuse", True), ("k_fuse_ng", False)):
    d = HERE / name
    d.mkdir(exist_ok=True)
    (d / "rms_norm_ttnn_compute.cpp").write_text(build(fb))
    print("wrote", d / "rms_norm_ttnn_compute.cpp")
