# Builds pd_stream.py from the shipped descriptor.  Re-runnable: it always starts
# from the shipped file, so the experiment's host edits are a readable DIFF rather
# than a 3700-line fork that silently drifts from the op.
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parents[1] / "rms_norm_ttnn_program_descriptor.py"
DST = HERE / "pd_stream.py"

s = SRC.read_text()
n = [0]


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"expected {count} of:\n{old[:200]}\ngot {s.count(old)}"
    s = s.replace(old, new)
    n[0] += 1


# ---- 1. knobs -------------------------------------------------------------
sub(
    "PER_CHANNEL_TRIM_BIAS = TRIM_DERIVED",
    """PER_CHANNEL_TRIM_BIAS = TRIM_DERIVED

# ---- perf_experiments/stream_regime: THE COMPACT PER-CHANNEL HOLD -------------
# A TILE per-channel operand is a (1,1,1,W) vector: 31 of every 32 tile rows are
# PADDING, and D23's TRIM == 2 already fetches only the two FACE-ROWS that carry
# row 0.  Those `2 * TILE_DIM * elem` bytes per width tile are the WHOLE of the
# operand's information -- 1/16 of its tiled size -- so the reader can cache the
# entire row of BOTH operands in L1 once and re-materialize each chunk's tiles
# LOCALLY (an L1->L1 copy, no NoC read).  Two things fall out:
#   * STREAM stops re-reading the operands from DRAM per pass-B chunk of every
#     row-block (measured 230,305 ns of a 1,580,377 ns wall on the target case);
#   * ROW_RESIDENT can price its per-channel HOLD at the compact size, which is
#     what lets a shape that used to fall off the L1 cliff into STREAM hold a
#     tile-row instead -- deleting pass B's re-read of x AND the residual.
PC_COMPACT_HOLD = True
# WHEN the cache is filled: False = EAGER, the whole row at boot; True = LAZY,
# chunk c the first time chunk c is staged.  Lazy issues the identical reads into
# the hole where the reader is already blocked on pass B instead of as a
# device-wide burst in front of the first x read (measured 1.16x on the target).
PC_COMPACT_LAZY = True
# Whether the ROW_RESIDENT L1 solve is allowed to use the compact price (and thus
# chunk the per-channel tile ring).  False keeps the shipped hold exactly.
ROW_RESIDENT_COMPACT_PC = True
# Chunks of head-room in the per-channel TILE ring.  1 = the reader's local expand
# of chunk c sits on pass B's critical path (compute's Upfront wait on the ring);
# 2 lets the reader run one chunk ahead, at one extra chunk of L1 -- which the
# chunk solve then has to give back.  A measured trade, not an obvious one.
PC_RING_CHUNKS = 1
# 0 = take the L1 solve's own cap.  A positive value CAPS the ROW_RESIDENT chunk,
# so a sweep can ask "is a finer chunk better than the coarsest that fits?".
ROW_RESIDENT_CAP_OVERRIDE = 0
# ---- THE COMPACT BRANCH'S ONE PRECONDITION: a SATURATED grid -----------------
# The compact hold makes ROW_RESIDENT reachable for shapes the shipped solve sent
# to STREAM.  ROW_RESIDENT trades DRAM BYTES (it deletes pass B's re-read of x and
# of the residual -- 2 of the 5 tensor-crossings a residual STREAM pays) for
# READER/COMPUTE OVERLAP (its pass B issues no read at all, so the reader has
# nothing to do there but run ahead into the next row-block).  That trade pays
# exactly when DRAM is the constraint, and the shape-blind proxy for "DRAM is the
# constraint" is how much of the compute grid is pulling on it: every active core
# is an independent NoC client.
#
# MEASURED, blackhole p150b (110-core grid), base(STREAM) -> compact ROW_RESIDENT.
# The middle block holds W, the residual, the dtype, rows-per-core AND the solved
# chunk (5) FIXED and moves only the number of active cores:
#     32 / 110 cores   (1,1,1024,16384) gbr    501,524 ->   763,861   0.66x
#     64 / 110 cores   (1,1,2048,16384) gbr    947,448 ->   975,876   0.97x
#     96 / 110 cores   (1,1,3072,16384) gbr  1,401,062 -> 1,280,538   1.09x
#    110 / 110 cores   (1,1,3520,16384) gbr  1,583,861 -> 1,429,391   1.11x
# and the two shapes the op's perf group actually cares about, both on the full
# grid, are where the trade is worth most because the chunk stays coarse:
#    110 / 110 cores   (1,1,8192, 7168) gbr fp32d  1,584,324 -> 1,048,060  1.51x
#    110 / 110 cores   (1,1,8192,10240) gbr fp32d  2,263,244 -> 1,517,643  1.49x
#
# It is monotone in grid occupancy and BLIND to everything else the candidates
# differ in -- the 0.66x row and the 1.11x row solve to the SAME WT_CHUNK (5) and
# the SAME one row-block per core, and a 1.02x row at 110 cores / 2 blocks
# ((1,1,4096,16384) gbr, 1,876,550 -> 1,836,015) rules the block count out too.
# The crossover is bracketed at 58% (loses) / 87% (wins) of the grid; 3/4 sits
# between the two measured points.  That bracket is the whole justification for
# the constant -- it is not tighter than the two shapes that bracket it.
ROW_RESIDENT_COMPACT_MIN_GRID_FRACTION = 0.75
# Sweep-only: 0 == take the gate off, which is what the 0.66x / 0.97x rows above
# were measured with.
#
# Sweep-only: a chunk-WIDTH floor on the compact branch.  1 == off, and off is what
# the measurements support -- the win is 1.11x at WT_CHUNK 5 and 1.51x at 32 on a
# saturated grid, and 0.66x at WT_CHUNK 5 and 0.83x at 18 on a third of one, so the
# chunk does not carry the sign.
ROW_RESIDENT_COMPACT_MIN_CHUNK_WT = 1
# Sweep-only: try the COMPACT hold BEFORE the tiled one.  This is the ordering the
# experiment's first cut had, kept so the sweep can show what it costs.
COMPACT_FIRST = False""",
)

# ---- 2. CB slots ----------------------------------------------------------
sub(
    "CB_BIAS_TILES = 23  # bias tiles (row 0 valid)",
    """CB_BIAS_TILES = 23  # bias tiles (row 0 valid)
# --- stream_regime: the reader-private COMPACT per-channel caches --------------
# Producer AND consumer are the reader (it fills them at boot and copies out of
# them per chunk), so they are scratch, never pushed and never popped.
CB_GAMMA_COMPACT = 24  # gamma, 2 face-rows per width tile
CB_BIAS_COMPACT = 25  # bias, ditto""",
)

# ---- 3. CT-arg counts -----------------------------------------------------
sub("READER_CT_SCALARS = 30\nCOMPUTE_CT_SCALARS = 26", "READER_CT_SCALARS = 33\nCOMPUTE_CT_SCALARS = 27")
sub("    reader_ct += [0, 0, 0, 0, 0, 0, 0, 0, 0]", "    reader_ct += [0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0]")
sub("    compute_ct += [0, 0, 0, 0, 0, 0, 0]", "    compute_ct += [0, 0, 0, 0, 0, 0, 0] + [0]")

# ---- 4. eligibility, derived right after the trim policy ------------------
sub(
    "    gamma_trim = _trim_for(gt, has_gamma, PER_CHANNEL_TRIM_GAMMA)\n"
    "    bias_trim = _trim_for(bit, has_bias, PER_CHANNEL_TRIM_BIAS)",
    "    gamma_trim = _trim_for(gt, has_gamma, PER_CHANNEL_TRIM_GAMMA)\n"
    "    bias_trim = _trim_for(bit, has_bias, PER_CHANNEL_TRIM_BIAS)\n"
    "\n"
    "    # ---- stream_regime: is the COMPACT per-channel hold expressible here? ----\n"
    "    # It is exactly D23's TRIM == 2 question -- the face-row form has to be legal\n"
    "    # for EVERY per-channel operand present, because the cache stores what the\n"
    "    # trim fetches.  A ROW_MAJOR operand already arrives compact (it IS a stick),\n"
    "    # so it is out of scope; a block-float one demotes to TRIM 1 and is refused.\n"
    "    pc_compact_ok = (\n"
    "        bool(PC_COMPACT_HOLD)\n"
    "        and (has_gamma or has_bias)\n"
    "        and not per_channel_is_rm\n"
    "        and (gamma_trim == 2 if has_gamma else True)\n"
    "        and (bias_trim == 2 if has_bias else True)\n"
    "    )\n"
    "    # Compact bytes per WIDTH TILE, summed over the operands present.\n"
    "    pc_compact_bytes = ((2 * TILE_DIM * gamma_elem_bytes) if has_gamma else 0) + (\n"
    "        (2 * TILE_DIM * bias_elem_bytes) if has_bias else 0\n"
    "    )\n"
    "    pc_compact_rr = pc_compact_ok and bool(ROW_RESIDENT_COMPACT_PC)",
)

# ---- 5. ROW_RESIDENT's L1 price, now a function of the HOLD FORM ----------
sub(
    """        def _row_resident_chunk(depth_x, depth_out):""",
    """        def _row_resident_chunk(depth_x, depth_out, compact=False):""",
)
sub(
    """            def _fixed(hold_wt):
                held = (1 if has_residual else depth_x) * hold_wt * bt
                held += _per_channel_bytes(hold_wt, 0)""",
    """            def _fixed(hold_wt):
                held = (1 if has_residual else depth_x) * hold_wt * bt
                # stream_regime: the COMPACT hold prices the per-channel operands at
                # the 2 face-rows that carry row 0 instead of their whole tiles -- 1/16
                # -- and pays for it with a per-chunk LOCAL expand.  It is a FALLBACK,
                # tried only after the tiled hold has been refused (see the search).
                if compact:
                    held += hold_wt * pc_compact_bytes
                else:
                    held += _per_channel_bytes(hold_wt, 0)""",
)
sub(
    """                + _per_channel_bytes(0, 1)
                + (rm_stage_rings * CB_RM_STAGE_DEPTH * bt if not is_tile else 0)
            )
            room = (budget - _fixed(wt_core)) // per_chunk_tile""",
    """                + _per_channel_bytes(0, 1)
                + (rm_stage_rings * CB_RM_STAGE_DEPTH * bt if not is_tile else 0)
                # stream_regime: with a compact hold the per-channel TILE ring becomes
                # CHUNKED (one WT_CHUNK window, popped per chunk) instead of held.
                + (PC_RING_CHUNKS * ((gt if has_gamma else 0) + (bit if has_bias else 0)) if compact else 0)
            )
            room = (budget - _fixed(wt_core)) // per_chunk_tile""",
)

# ---- 6. STREAM keeps the shipped price EXACTLY ---------------------------
# The compact cache is scoped to ROW_RESIDENT-ENABLEMENT, not to STREAM.  Putting
# it in STREAM as well is expressible and CORRECT, and it does delete the
# per-chunk per-channel DRAM re-read that the ablation prices at 230,305 ns -- but
# the cache's own L1 shrinks the width chunk (56 -> 45 on the target), and the
# measurement says the chunk is worth more than the traffic:
#     STREAM + compact cache, target case:  1,611,124 ns vs 1,582,116 shipped (0.98x)
# so STREAM is left byte-identical to the op.

# ---- 7. post-solve derivation --------------------------------------------
sub(
    "    x_hold_wt = wt_chunk * num_w_chunks if x_resident else wt_chunk",
    """    x_hold_wt = wt_chunk * num_w_chunks if x_resident else wt_chunk
    # ---- stream_regime: the compact hold's three derived facts ---------------
    # `pc_hold_wt` is the CACHE's width -- always the core's whole (padded) row,
    # in every regime, because the cache's whole job is to be read once.
    # `pc_compact` comes back FROM the solve -- it is true exactly when the regime
    # search had to fall back to the compact hold to reach ROW_RESIDENT at all.
    # Every RESIDENT, every already-fitting ROW_RESIDENT and every STREAM build
    # therefore has it False and is BYTE-IDENTICAL to the op's.
    pc_hold_wt = wt_chunk * num_w_chunks
    assert not pc_compact or (x_resident and num_w_chunks > 1), "stream_regime: the compact hold is ROW_RESIDENT's"
    # The per-channel TILE ring is a CHUNKED window (WT_CHUNK pages, popped per
    # chunk) rather than a held row.  Already true in STREAM; the compact cache is
    # what makes it possible under ROW_RESIDENT.
    pc_chunked = (not x_resident) or pc_compact
    pc_tile_pages = (PC_RING_CHUNKS * wt_chunk) if pc_chunked else x_hold_wt""",
)
sub(
    '            f"rows_max={max((a.row_count for a in assignment), default=0)}",',
    '            f"rows_max={max((a.row_count for a in assignment), default=0)} "\n'
    '            f"PC_COMPACT={int(pc_compact)} PC_CHUNKED={int(pc_chunked)} PC_RING={pc_tile_pages}",',
)

# ---- 8. CB table ----------------------------------------------------------
sub(
    "        cbs.append(_cb(CB_GAMMA_TILES, gt, x_hold_wt, weight.dtype, all_cores))",
    "        cbs.append(_cb(CB_GAMMA_TILES, gt, pc_tile_pages, weight.dtype, all_cores))\n"
    "        if pc_compact:\n"
    "            cbs.append(\n"
    "                _cb(CB_GAMMA_COMPACT, 2 * TILE_DIM * gamma_elem_bytes, pc_hold_wt, weight.dtype, all_cores)\n"
    "            )",
)
sub(
    "        cbs.append(_cb(CB_BIAS_TILES, bit, x_hold_wt, bias.dtype, all_cores))",
    "        cbs.append(_cb(CB_BIAS_TILES, bit, pc_tile_pages, bias.dtype, all_cores))\n"
    "        if pc_compact:\n"
    "            cbs.append(_cb(CB_BIAS_COMPACT, 2 * TILE_DIM * bias_elem_bytes, pc_hold_wt, bias.dtype, all_cores))",
)

# ---- 9. reader CT args ----------------------------------------------------
sub(
    """        wt_pad,
    ]
    assert (
        len(reader_ct_args) == READER_CT_SCALARS""",
    """        wt_pad,
        # ---- stream_regime: the COMPACT per-channel hold, appended --------------
        (0 if not pc_compact else (2 if PC_COMPACT_LAZY else 1)),  # 30 PC_COMPACT 0/1 eager/2 lazy
        pc_hold_wt,  # 31 width tiles the cache spans (the core's padded row)
        1 if pc_chunked else 0,  # 32 PC_CHUNKED: the per-channel TILE ring is one chunk
    ]
    assert (
        len(reader_ct_args) == READER_CT_SCALARS""",
)

# ---- 10. compute CT args --------------------------------------------------
sub(
    """        RES_FUSE,  # 25 Lamp L-RES-FUSE: t = x + r and its square as ONE chain
    ]""",
    """        RES_FUSE,  # 25 Lamp L-RES-FUSE: t = x + r and its square as ONE chain
        # 26 stream_regime: the per-channel TILE CB is a CHUNKED window (WT_CHUNK
        # pages, popped after every chunk) rather than a held whole row.  Was
        # implicitly `!X_RESIDENT`; the compact hold decouples the two.
        1 if pc_chunked else 0,
    ]""",
)

# ---- 5b. the compact branch's granularity floor ---------------------------
sub(
    """            if wtc < ROW_RESIDENT_MIN_CHUNK_WT:""",
    """            if wtc < (ROW_RESIDENT_COMPACT_MIN_CHUNK_WT if compact else ROW_RESIDENT_MIN_CHUNK_WT):""",
)

# ---- 5c. the regime search: TILED hold first, COMPACT only as a fallback --
sub(
    """        stream_depth = depth_candidates[0]
        for depth in tuple(dict.fromkeys(depth_candidates + (1,))):
            if depth < stream_depth and max_rows < ROW_RESIDENT_MIN_ROWS_PER_CORE:
                continue
            fit = _row_resident_chunk(depth, depth)
            if fit:
                return 1, fit[0], fit[1], depth, depth, True, CB_RM_STAGE_DEPTH, False""",
    """        stream_depth = depth_candidates[0]
        # stream_regime -- ORDERED COARSEST-FIRST ON WHAT EACH STEP COSTS, the same
        # discipline the BAND search above uses.  The TILED per-channel hold is tried
        # at every depth FIRST: it stages the operands once per core and pass B reads
        # them with no producer handshake at all.  Only when no depth admits it does
        # the COMPACT hold get a turn -- it buys the regime (and with it the deletion
        # of pass B's re-read of x AND the residual) at the price of a per-chunk local
        # expand sitting on pass B's Upfront wait.  Ordering it second is what keeps
        # every shape that ALREADY fit ROW_RESIDENT byte-identical; measured, taking
        # the compact hold where the tiled one fits COSTS 0.84-0.94x
        # ((1,1,8192,5120) gamma_bias_residual 658,766 -> 782,650 ns,
        #  (1,1,8192,5120) gamma 419,090 -> 454,511, (1,1,8192,7168) gamma
        #  576,476 -> 614,070).
        _hold_forms = (True, False) if COMPACT_FIRST else (False, True)
        _grid = device.compute_with_storage_grid_size()
        _active = sum(1 for a in plan.assignment if a.row_count)
        _compact_ok = pc_compact_rr and _active >= ROW_RESIDENT_COMPACT_MIN_GRID_FRACTION * (_grid.x * _grid.y)
        for compact in (_hold_forms if _compact_ok else (False,)):
            for depth in tuple(dict.fromkeys(depth_candidates + (1,))):
                if depth < stream_depth and max_rows < ROW_RESIDENT_MIN_ROWS_PER_CORE:
                    continue
                fit = _row_resident_chunk(depth, depth, compact)
                if fit:
                    return 1, fit[0], fit[1], depth, depth, True, CB_RM_STAGE_DEPTH, False, compact""",
)

# ---- 5d. every other return grows the same (dead) field -------------------
sub(
    "                return min(max_rows, brmax), wt_core, 1, depth, depth, True, CB_RM_STAGE_DEPTH, False",
    "                return min(max_rows, brmax), wt_core, 1, depth, depth, True, CB_RM_STAGE_DEPTH, False, False",
)
sub(
    """                            True,
                            rm_depth,
                            narrow_pc,
                        )""",
    """                            True,
                            rm_depth,
                            narrow_pc,
                            False,
                        )""",
)
sub(
    "            return 1, wt_core, 1, depth_candidates[0], depth_candidates[0], True, band_depths[-1], True",
    "            return 1, wt_core, 1, depth_candidates[0], depth_candidates[0], True, band_depths[-1], True, False",
)
sub(
    "        return 1, wtc, n, depth, depth, False, CB_RM_STAGE_DEPTH, False",
    "        return 1, wtc, n, depth, depth, False, CB_RM_STAGE_DEPTH, False, False",
)
sub(
    """        rm_stage_depth,
        narrow_pc_stage,
    ) = solved""",
    """        rm_stage_depth,
        narrow_pc_stage,
        pc_compact,
    ) = solved""",
)

# ---- 11. the ROW_RESIDENT chunk-cap override (sweep knob) -----------------
sub(
    """            cap = min(room, wt_core - 1) if wt_core > 1 else 1""",
    """            cap = min(room, wt_core - 1) if wt_core > 1 else 1
            if ROW_RESIDENT_CAP_OVERRIDE:
                cap = min(cap, ROW_RESIDENT_CAP_OVERRIDE)""",
)

DST.write_text(s)
print(f"pd_stream.py written, {n[0]} edits", file=sys.stderr)
