"""passb_op_count -- build the forked pass-B compute kernels from the shipped one.

Every variant lives at its OWN PATH (k_<name>/) because the JIT kernel cache key
does not hash kernel source CONTENT: editing one file in place and re-running is a
cache HIT on the previous binary.  Re-run this script after touching the shipped
kernel; it always starts from a fresh copy.

Variants
  base  the shipped op, byte-for-byte (the honest baseline)
  swap  pass B in the OTHER order: out = (x * gamma<Row>) * stat<Col>
  ceil  ABLATION, wrong on purpose: the gamma traversal is DELETED (the scale packs
        cb_output_tiles direct).  Upper bound on what ANY route that removes one
        pass-B traversal can be worth on this shape.
  ua    the two pass-B packs go Upfront/AtEnd instead of PerBlockSize/PerBlockSize
        (one reserve+push per BLOCK instead of one per DEST-lane block)
  uan   Upfront/AtEnd on the cb_normalized pack only; cb_output_tiles keeps the
        per-DEST-block handover the writer overlaps against
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "kernels"

SCALE_START = "            // x * (1/rms). The stat is a REDUCE_ROW result: column-shaped, so it"
BIAS_START = "            if constexpr (HAS_B) {\n                // A2: the per-channel SHIFT"

STAT_B = """ckl::input(
                            CB_STAT_B,
                            ckl::BroadcastDim::Col,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Col)"""

SWAP = """            // passb_op_count / swap: THE SAME TWO MULS IN THE OTHER ORDER.
            //   shipped:  (x * stat<Col>) * gamma<Row>
            //   here:     (x * gamma<Row>) * stat<Col>
            // Op count is identical -- two broadcast muls over rows x WT_CHUNK tiles --
            // so this is not a fusion; it tests whether the ORDER of the two operand
            // shapes costs anything (which broadcast axis reads x, which reads the
            // materialized intermediate, and which stage carries the fp32 stat's
            // format reconfig).
            if constexpr (HAS_G) {
                {
                    MaybeDeviceZoneScope("compute_gamma_mul");
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_B, ckl::input(G_IN, ckl::BroadcastDim::Row)>{
                            hold_base, pc_base},
                        ckl::PackTile<PASS_B_OUT_NORM>{});
                }
                {
                    MaybeDeviceZoneScope("compute_scale");
                    if constexpr (HAS_B) {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                ckl::input(
                                    cb_normalized,
                                    ckl::WaitPolicy::PerBlockSize,
                                    ckl::PopPolicy::PerBlockSize,
                                    ckl::OperandKind::Block),
                                STAT_B_SPEC>{0u, 0u},
                            ckl::PackTile<ckl::output(
                                cb_normalized, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
                    } else {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                ckl::input(
                                    cb_normalized,
                                    ckl::WaitPolicy::Upfront,
                                    ckl::PopPolicy::AtEnd,
                                    ckl::OperandKind::Block),
                                STAT_B_SPEC>{0u, 0u},
                            ckl::PackTile<PASS_B_OUT_GAMMA>{});
                    }
                }
            } else {
                MaybeDeviceZoneScope("compute_scale");
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_B, STAT_B_SPEC>{hold_base},
                    ckl::PackTile<PASS_B_OUT_NORM>{});
            }

""".replace(
    "STAT_B_SPEC", STAT_B
)

CEIL = """            // passb_op_count / ceil: ABLATION, DELIBERATELY WRONG.  The gamma
            // traversal is gone and the scale packs cb_output_tiles direct, so the
            // output is missing its gamma factor.  This is the CEILING for any route
            // that removes one of pass B's two block traversals (a perfect fusion
            // removes at most this much: one unpack, one pack, one set of flow
            // control and the mul itself).
            {
                MaybeDeviceZoneScope("compute_scale");
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_B, STAT_B_SPEC>{hold_base},
                    ckl::PackTile<PASS_B_OUT_NORM>{});
            }
            if constexpr (HAS_G) {
                // The gamma traversal is DELETED in every operand set, bias included
                // (with a bias the shipped stage transforms cb_normalized in place, so
                // dropping it just leaves the block the scale pushed for the bias stage
                // to read).  Keep the per-channel ring's flow control alive: the chain
                // that used to wait it is exactly what is being ablated.
                if constexpr (PC_CHUNKED) {
                    cb_wait_front(cb_gamma_tiles, WT_CHUNK);
                }
            }

""".replace(
    "STAT_B_SPEC", STAT_B
)

NORM_OUT_BASE = "    constexpr uint32_t NORM_OUT = (HAS_G || HAS_B) ? cb_normalized : cb_output_tiles;"
NORM_OUT_CEIL = "    constexpr uint32_t NORM_OUT = HAS_B ? cb_normalized : cb_output_tiles;  // ceil: gamma deleted"

PACKS_BASE = """    constexpr auto PASS_B_OUT_NORM =
        ckl::output(NORM_OUT, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    constexpr auto PASS_B_OUT_GAMMA =
        ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);"""

# `Upfront` reserve on an IN-PLACE target DEADLOCKS (chain.inl:82-85), so the
# cb_normalized pack keeps PerBlockSize whenever a bias makes the next stage
# transform it in place.  RM keeps PerBlockSize on both: the untilize consumer
# needs the per-block page handover.
PACKS_UA = """    constexpr bool PB_UA = !RM && !HAS_B;   // ua: one reserve/push per BLOCK
    constexpr auto PASS_B_OUT_NORM = ckl::output(
        NORM_OUT,
        PB_UA ? ckl::ReservePolicy::Upfront : ckl::ReservePolicy::PerBlockSize,
        PB_UA ? ckl::PushPolicy::AtEnd : ckl::PushPolicy::PerBlockSize);
    constexpr auto PASS_B_OUT_GAMMA = ckl::output(
        cb_output_tiles,
        (!RM) ? ckl::ReservePolicy::Upfront : ckl::ReservePolicy::PerBlockSize,
        (!RM) ? ckl::PushPolicy::AtEnd : ckl::PushPolicy::PerBlockSize);"""

PACKS_UAN = """    constexpr bool PB_UA = !RM && !HAS_B;   // uan: block-granular pack on cb_normalized only
    constexpr auto PASS_B_OUT_NORM = ckl::output(
        NORM_OUT,
        PB_UA ? ckl::ReservePolicy::Upfront : ckl::ReservePolicy::PerBlockSize,
        PB_UA ? ckl::PushPolicy::AtEnd : ckl::PushPolicy::PerBlockSize);
    constexpr auto PASS_B_OUT_GAMMA =
        ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);"""


# ---------------------------------------------------------------------------
# swapx / nomul -- added after the first sweep (their own dirs, so their JIT
# cache keys are their own; k_swap's source is left EXACTLY as measured).
# ---------------------------------------------------------------------------

# The gamma-first body, without the !HAS_G fallback (its callers guard that).
SWAP_INNER = SWAP[SWAP.index("            if constexpr (HAS_G) {") :]
SWAP_INNER = SWAP_INNER[SWAP_INNER.index("{") + 1 : SWAP_INNER.rindex("\n            } else {")]

SWAPX_HEAD = """            // passb_op_count / swapx: the swap, GATED on CROSS_CORE.
            //
            // Pass B's first op is the one that needs the finalized stat.  On a
            // cross-core plan (`combine`) that stat arrives by gather -> root fold ->
            // multicast, and the shipped order stalls pass B on its arrival with the
            // gamma traversal still to do.  Doing the gamma mul FIRST -- it depends on
            // x and gamma only, never on the stat -- fills that wait with the traversal
            // instead of idling through it.  MEASURED 1.046x-1.125x on every
            // combine=True plan with a one-tile-row block, and a reproducible -1.7%
            // on `scheme=rows` case 05 (combine=False, where there is no wait to hide),
            // which is why the reorder is gated rather than unconditional.
            if constexpr (HAS_G && CROSS_CORE) {
"""


def compose(head, base_region):
    return head + SWAP_INNER + "            } else {\n" + base_region + "            }\n\n"


NOMUL_HEAD = """            // passb_op_count / nomul: ABLATION, DELIBERATELY WRONG.  The gamma
            // traversal keeps its unpack, its pack and its whole CB lifecycle but
            // drops the MULTIPLY (CopyTile instead of BinaryFpu).  base -> nomul is
            // therefore the cost of the mul itself (fidelity phases + the second
            // operand's unpack); nomul -> ceil is the cost of the traversal's
            // scaffolding.  That split says what a fusion could ever recover.
"""

NOMUL = (
    NOMUL_HEAD
    + """            {
                MaybeDeviceZoneScope("compute_scale");
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_B, STAT_B_SPEC>{hold_base},
                    ckl::PackTile<PASS_B_OUT_NORM>{});
            }
            if constexpr (HAS_G) {
                MaybeDeviceZoneScope("compute_gamma_mul");
                if constexpr (PC_CHUNKED) {
                    cb_wait_front(cb_gamma_tiles, WT_CHUNK);
                }
                if constexpr (HAS_B) {
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                        ckl::CopyTile<ckl::input(
                            cb_normalized,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::OperandKind::Block)>{0u},
                        ckl::PackTile<ckl::output(
                            cb_normalized, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
                } else {
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                        ckl::CopyTile<ckl::input(
                            cb_normalized, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block)>{
                            0u},
                        ckl::PackTile<PASS_B_OUT_GAMMA>{});
                }
            }

""".replace(
        "STAT_B_SPEC", STAT_B
    )
)


def fork(name):
    d = HERE / f"k_{name}"
    d.mkdir(exist_ok=True)
    for f in SRC.iterdir():
        if f.suffix in (".cpp", ".hpp"):
            shutil.copy2(f, d / f.name)
    return d / "rms_norm_ttnn_compute.cpp"


def swap_region(path, new):
    s = path.read_text()
    a = s.index(SCALE_START)
    b = s.index(BIAS_START)
    path.write_text(s[:a] + new + s[b:])


def sub(path, old, new):
    s = path.read_text()
    assert s.count(old) == 1, (path, old[:60])
    path.write_text(s.replace(old, new))


def main():
    fork("base")  # untouched

    p = fork("swap")
    swap_region(p, SWAP)

    p = fork("ceil")
    swap_region(p, CEIL)
    sub(p, NORM_OUT_BASE, NORM_OUT_CEIL)

    p = fork("ua")
    sub(p, PACKS_BASE, PACKS_UA)

    p = fork("uan")
    sub(p, PACKS_BASE, PACKS_UAN)

    base_region = (HERE / "k_base" / "rms_norm_ttnn_compute.cpp").read_text()
    base_region = base_region[base_region.index(SCALE_START) : base_region.index(BIAS_START)]

    p = fork("swapx")
    swap_region(p, compose(SWAPX_HEAD, base_region))

    p = fork("nomul")
    swap_region(p, NOMUL)

    print("variants built:", ", ".join(sorted(x.name for x in HERE.glob("k_*"))))


main()
