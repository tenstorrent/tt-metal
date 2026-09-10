"""Build pd_fold.py from the SHIPPED descriptor.

Re-runnable: it always starts from the shipped file, so the experiment's host edits are a
readable DIFF rather than a 4300-line fork that silently drifts from the op.

Two edits, one idea: `x_squared_wt` (== cb_x_squared's width per tile-row == the reduce's
per-call reduce-dim width) gets ONE definition, `_x_squared_wt(wt_chunk, partial_w)`, and
that definition can return a DIVISOR of wt_chunk instead of only 1 or wt_chunk.  With
`SQ_FOLD_GROUP = 0` the function returns exactly what the shipped two-line expression
returns, so the fork is byte-identical to the op at its default.
"""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[1] / "rms_norm_ttnn_program_descriptor.py"
DST = HERE / "pd_fold.py"

s = SRC.read_text()
n = [0]


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"expected {count} of:\n{old[:300]}\ngot {s.count(old)}"
    s = s.replace(old, new)
    n[0] += 1


# ---- 1. the knob + the ONE definition ------------------------------------
sub(
    "DEST_ACC_SQUARE_MAX_WT = 8",
    '''DEST_ACC_SQUARE_MAX_WT = 8

# ---- perf_experiments/square_fold_ceiling: THE GROUPED FOLD ------------------
# DEST_ACC_SQUARE_MAX_WT is a PRECISION ceiling: it bounds how many x^2 tiles the fold
# accumulates SERIALLY inside a 16-bit DEST register.  Because the shipped fold folds the
# WHOLE chunk or nothing, that precision bound is also a PERF bound -- every prefill
# profile (WT_CHUNK 32..80) is on the packed path, paying WT_CHUNK packs and WT_CHUNK
# unpacks per tile-row.
#
# The grouped fold decouples the two.  `SQ_FOLD_GROUP = G` folds in groups of at most G
# width tiles: the serial accumulation depth stays <= G (so it can be pinned at the
# already-precision-vetted 8) while cb_x_squared holds WT_CHUNK/G tiles per tile-row
# instead of WT_CHUNK, deleting (G-1)/G of the square's packs and of the reduce's unpacks.
#   0 (or 1)  the shipped behaviour exactly: fold the whole chunk when it is <= the
#             ceiling, else pack every tile.
#   G > 1     fold in groups of the LARGEST DIVISOR of WT_CHUNK that is <= G.  A divisor
#             is required because the group has to tile the chunk exactly -- a ragged last
#             group would need a second iteration shape.
SQ_FOLD_GROUP = 0


def _x_squared_wt(wt_chunk: int, partial_w: int) -> int:
    """cb_x_squared's width tiles per tile-row == the reduce's per-call reduce-dim width.

    ONE definition, read by the RESIDENT L1 solve, the CB table and the compute kernel's
    CT arg -- they must agree page-for-page or the ring is sized against a layout that
    does not exist.  Returns wt_chunk (no fold), 1 (the flat fold), or a divisor of
    wt_chunk (the grouped fold).
    """
    if partial_w != 0:
        # The fold folds the row's last width tile's PAD LANES in before the reduce runs,
        # so the reduce's partial scaler / 0-1 mask can no longer reach them.  Unchanged.
        return wt_chunk
    if wt_chunk <= DEST_ACC_SQUARE_MAX_WT:
        return 1
    g = max((d for d in range(2, min(int(SQ_FOLD_GROUP), wt_chunk) + 1) if wt_chunk % d == 0), default=1)
    return wt_chunk // g''',
)

# ---- 2. the RESIDENT L1 solve's price ------------------------------------
sub(
    "            sq_wt = 1 if (CB_SQ_EXACT and kernel_partial_w == 0 and wt_core <= DEST_ACC_SQUARE_MAX_WT) else wt_core",
    "            sq_wt = _x_squared_wt(wt_core, kernel_partial_w) if CB_SQ_EXACT else wt_core",
)

# ---- 3. the build's own decision -----------------------------------------
sub(
    """    square_dest_acc_per_row = kernel_partial_w == 0 and wt_chunk <= DEST_ACC_SQUARE_MAX_WT
    x_squared_wt = 1 if square_dest_acc_per_row else wt_chunk""",
    """    x_squared_wt = _x_squared_wt(wt_chunk, kernel_partial_w)
    square_dest_acc_per_row = x_squared_wt != wt_chunk""",
)

# ---- 4. the assert ------------------------------------------------------
sub(
    '    assert x_squared_wt in (1, wt_chunk), "rms_norm_ttnn: x_squared_wt must be 1 (DEST fold) or WT_CHUNK"',
    "    assert x_squared_wt >= 1 and wt_chunk % x_squared_wt == 0, (\n"
    '        "rms_norm_ttnn: x_squared_wt must divide WT_CHUNK"\n'
    "    )",
)

DST.write_text(s)
print(f"patch_pd: {n[0]} edits -> {DST}")
