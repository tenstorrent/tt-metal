"""Build k_base/ (byte copy) and k_grp/ (the GROUPED DEST fold) from the shipped kernels.

Re-runnable from the pristine source, so the experiment's device edits stay a readable diff.

k_grp's ONE change: `X_SQUARED_WT` -- already a compile-time arg, already the reduce's
per-call width -- is allowed to be any DIVISOR of WT_CHUNK instead of only 1 or WT_CHUNK.
The square then accumulates `SQ_GROUP = WT_CHUNK / X_SQUARED_WT` width tiles into ONE DEST
slot and packs X_SQUARED_WT tiles per tile-row.  Expressed purely by RESHAPING the chain's
iteration grid -- `grid(rows * X_SQUARED_WT, SQ_GROUP)` with the SAME
`DestAccumulation::PerRow` output spec the shipped fold uses -- because
`DestAccumulation::PerRow` acquires/packs/clears DEST once per row of the grid
(chain.inl's driver loop, `per_row_dest_accumulation`), and `OperandKind::Block`'s index is
`base + r * Wt + c`, so the tile WALK ORDER over the (rows x WT_CHUNK) block is bit-for-bit
the one the flat shape produces.  No new helper, no raw LLK.

At X_SQUARED_WT == 1 the reshaped grid IS the shipped flat grid (`grid(rows, WT_CHUNK)`),
and at X_SQUARED_WT == WT_CHUNK the packed branch is kept verbatim (including its SQ_BLK
DEST blocking), so k_grp reduces to the shipped kernel on every shipped build.
"""

import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[1] / "kernels"

for d in ("k_base", "k_grp"):
    dst = HERE / d
    dst.mkdir(exist_ok=True)
    for f in SRC.iterdir():
        if f.is_file():
            shutil.copy(f, dst / f.name)

n = [0]


class F:
    def __init__(self, p):
        self.p = p
        self.s = p.read_text()

    def sub(self, old, new, count=1):
        assert self.s.count(old) == count, f"{self.p.name}: expected {count} of:\n{old[:300]}\ngot {self.s.count(old)}"
        self.s = self.s.replace(old, new)
        n[0] += 1

    def save(self):
        self.p.write_text(self.s)


c = F(HERE / "k_grp" / "rms_norm_ttnn_compute.cpp")

# ---- 1. the gate: any divisor of WT_CHUNK, and SQ_GROUP is the fold's depth ----
c.sub(
    """    constexpr bool SQ_FOLD = (X_SQUARED_WT == 1) && (WT_CHUNK > 1);
    static_assert(
        X_SQUARED_WT == 1 || X_SQUARED_WT == WT_CHUNK,
        "rms_norm_ttnn: X_SQUARED_WT must be 1 (the DEST fold) or WT_CHUNK (the packed path)");""",
    """    // perf_experiments/square_fold_ceiling -- THE GROUPED FOLD.
    // X_SQUARED_WT is any DIVISOR of WT_CHUNK.  SQ_GROUP == WT_CHUNK / X_SQUARED_WT is
    // how many width tiles are accumulated into ONE DEST slot before a pack, i.e. it is
    // BOTH the pack/unpack saving (SQ_GROUP:1) AND the serial 16-bit accumulation depth
    // the descriptor's DEST_ACC_SQUARE_MAX_WT ceiling exists to bound.  The two are now
    // decoupled from WT_CHUNK: a chunk of 32 can fold in groups of 8 -- exactly the depth
    // the shipped ceiling already permits -- and still delete 7 of every 8 packs.
    //   X_SQUARED_WT == 1          the shipped FLAT fold (SQ_GROUP == WT_CHUNK)
    //   1 < X_SQUARED_WT < WT_CHUNK the GROUPED fold
    //   X_SQUARED_WT == WT_CHUNK   the shipped PACKED path (SQ_GROUP == 1, no fold)
    constexpr uint32_t SQ_GROUP = WT_CHUNK / X_SQUARED_WT;
    constexpr bool SQ_FOLD = (SQ_GROUP > 1);
    static_assert(
        X_SQUARED_WT >= 1 && WT_CHUNK % X_SQUARED_WT == 0,
        "rms_norm_ttnn: X_SQUARED_WT must be a divisor of WT_CHUNK (1 == the flat DEST fold)");""",
)

# ---- 2. the iteration shape: ONE definition, used by both square chains -------
c.sub(
    """                MaybeDeviceZoneScope("compute_square");
#ifdef RMS_ABLATE_COMPUTE""",
    """                MaybeDeviceZoneScope("compute_square");
                // The fold's grid is (rows * X_SQUARED_WT) rows of SQ_GROUP tiles, so
                // DestAccumulation::PerRow acquires/packs/clears DEST once per GROUP.
                // `OperandKind::Block` indexes `base + r * Wt + c`, so the walk over the
                // (rows x WT_CHUNK) block is the flat shape's walk, tile for tile.  The
                // packed branch keeps grid(rows, WT_CHUNK) AND its SQ_BLK DEST blocking:
                // block_size applies to the INNER extent, so folding it into the reshaped
                // grid would silently drop D21's blocking on the un-folded path.
                const auto sq_shape = SQ_FOLD
                                          ? ckl::IterationShape::grid(rows * X_SQUARED_WT, SQ_GROUP)
                                          : ckl::IterationShape::grid(rows, WT_CHUNK).block_size(SQ_BLK);
#ifdef RMS_ABLATE_COMPUTE""",
)
c.sub(
    """                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(SQ_BLK),
                    ckl::CopyTile<X_IN_A>{hold_base},
                    ckl::PackTile<SQ_OUT>{});""",
    """                ckl::eltwise_chain(
                    sq_shape,
                    ckl::CopyTile<X_IN_A>{hold_base},
                    ckl::PackTile<SQ_OUT>{});""",
)
c.sub(
    """                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(SQ_BLK),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_A, X_IN_A, ckl::Dst::D0, SQ_OUT.dest_accumulation>{
                        hold_base, hold_base},
                    ckl::PackTile<SQ_OUT>{});""",
    """                ckl::eltwise_chain(
                    sq_shape,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_A, X_IN_A, ckl::Dst::D0, SQ_OUT.dest_accumulation>{
                        hold_base, hold_base},
                    ckl::PackTile<SQ_OUT>{});""",
)

c.save()
print(f"patch_kernels: {n[0]} edits -> {HERE/'k_grp'}")
