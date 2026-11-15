# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for gated_delta_net (regimes R-STEP, R-VSPLIT and R-CHUNK, op_design.md).

Work unit: one piece of one (sequence, value head) pair: the pair's state columns split into
`pieces = Vt / v_block_tiles` column slices (R-STEP: one piece per pair). Units are split across the
grid with ttnn.split_work_to_cores(row_wise=True); each core walks a contiguous unit range, one
state slice resident at a time, token rows in blocks of one tile row, one token per recurrence step.
R-VSPLIT (pieces > 1) places one unit per core and all-gathers the per-piece partial sum o^2 of
each tile row over the pair's piece cores (semaphore-counted unicast) before the RMS-norm rsqrt.
R-CHUNK (multi-token calls) replaces the token steps of a tile row by one closed-form chunk step (F1): the
chunk's decay matrix and unit-lower-triangular system are built and solved in fp32 on the SFPU, the
state is carried in fp32 across chunks.
R-PREP (chunked R-VSPLIT) moves the per-tile-row preparation, everything that depends on the token rows alone, onto
the idle cores: prep cores per (sequence, key head) take its tile rows round robin and write prescaled head blocks into
the inboxes of the state cores, which run only the chunk step's state part and return o for the RMS norm.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
FP32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048

# ---- Blocking-model knobs (op_design.md, Blocking Model) ----
PAIRS_PER_BLOCK = 1  # pairs per work-split unit
TOKEN_TILE_ROWS_PER_BLOCK = 1  # prep / norm block along t (tile rows); only 1 is built
# State column block per piece. None = auto: the smallest block (most pieces, at most MAX_PIECES) whose
# units still fit one per core (R-VSPLIT), else Vt (R-STEP). An int forces it (must divide Vt).
V_BLOCK_TILES = None
MAX_PIECES = 4  # the partial-sum combine adds every piece's tile in one fp32 half-sync DEST window
MAX_CORES = None  # optional grid cap; None = full compute grid
# Conv taps on the FPU (row-broadcast ELWMUL accumulated in DEST) instead of the SFPU. Exact only for
# bf16 token operands (srcA/srcB hold 19 bits), so "auto" enables it for bf16 tokens only.
CONV_ON_FPU = "auto"
# R-CHUNK: calls with at least this many tokens per sequence take one chunk step (F1) per tile row instead
# of one recurrence step per token; decode (T = 1) keeps the token step. Needs the Blackhole SFPU solve.
CHUNK_MIN_TOKENS = 2
# Full-sync DEST (8 fp32 tiles per window, state update batches of 7). Measured slower on
# perf_decode_b1 (54.5 vs 53.2 us: it loses math/pack overlap), so off; the kernels adapt to either.
DST_FULL_SYNC = False
# R-BCAST (decode R-VSPLIT): of each key head's r * pieces cores, the first preps q (conv + L2 norm) and the
# second k, once each, and multicast qn / kn to the others, which read and prep only their v columns.
SHARE_QK_PREP = True
# R-PREP (chunked R-VSPLIT): the per-tile-row preparation, which depends on the token rows alone (gates, conv,
# L2 norm, [Kn Kn^T; Qn Kn^T], chunk vectors, L and QD), runs on the cores R-VSPLIT leaves idle: per (sequence,
# key head) up to MAX_PREP_CORES_PER_GROUP prep cores take its tile rows round robin and write each prepped
# tile row into the inboxes of the key head's r * pieces state cores, which run only the state part of the
# chunk step and the RMS norm.
PREP_OFFLOAD = True
# Prep cores per group, at most: the state cores' producer tables hold this many (more than about twice a prep tile
# row's time over a state step buys nothing).
MAX_PREP_CORES_PER_GROUP = 32
PREP_SEND_NOC_1 = True  # prep writer on NOC 1, reader on NOC 0 (see the kernel descriptors)
# Pieces per pair under R-PREP, at most. Fewer pieces leave more prep cores and halve the fan-out of a prepped row;
# model_prefill_4096: 4 pieces 1.34 ms (prep-bound, sends late), 2 pieces 1.18 ms (same run).
PREP_MAX_PIECES = 2
# R-PREP fills its pipeline once (about one prep core's tile row, ~120 us at the model geometry) and then runs a tile
# row per state step; below this many tile rows per sequence the local chunk step is faster (prefill T=96, 3 tile
# rows: 168 us local, 177 us R-PREP).
PREP_MIN_TILE_ROWS = 4

# ---- Buffer-depth knobs ----
WINDOW_DEPTH = 2  # window groups of tap tiles in flight
WINDOW_GROUP_MAX_COLS = 4  # column tiles per window hand-off (one DRAM flush each), upper bound
TOK_COEF_DEPTH = 2  # tokens of coefficient pages in flight
INBOX_DEPTH = 12  # R-PREP: prepped tile rows in flight per state core (slots of its inbox buffers), at most
O_IN_DEPTH = 2  # R-PREP: a prep core's tile rows of chunk outputs o in flight (slot k % O_IN_DEPTH for its k-th row)
O_ACC_DEPTH = 2  # R-PREP: a state core's tile rows of o awaiting its writer
PREP_SEND_HEADS = None  # R-PREP: head blocks a prep core holds for its writer; None = one tile row's (r)
ROW_DEPTH = 2  # tile rows of gate / z / output tiles in flight
STATE_IN_DEPTH = 1  # pair states (S0) in flight
STATE_OUT_DEPTH = 1  # finished pair states awaiting the writer
CONV_ACC_DEPTH = 1  # partial tap sum: compute-to-compute, one compute thread, so depth 1

HISTORY = 3  # spec constant: conv_state rows per sequence
CONV_TAPS = HISTORY + 1  # history rows + current token, not a knob
TOK_COEF_PAGES = 3  # alpha (all rows), beta at row p, alpha at row p
CHUNK_COEF_PAGES = 2  # R-CHUNK, per tile row: g and beta of the pair in column 0, 0 on rows that are no token
CHUNK_VEC_PAGES = 5  # R-CHUNK: D, c, exp(c), exp(c_last - c), beta exp(c)
CHUNK_MAT_PAGES = 2  # R-CHUNK: L (solve operand), (Qn Kn^T) * D (matmul operand)
# R-PREP head block, per value head and tile row, prescaled so the state core broadcasts nothing:
# [beta exp(c) (*) kn, exp(c) (*) qn] (2 Kt, its cb_in_kq slot), kn^T (*) exp(c_last - c)^T (Kt, cb_in_knt),
# M = (I + L)^-1 and QD (cb_in_mat), exp(c) (cb_in_vec: the chunk decay exp(c_last) is its last token row), then
# beta (*) v of the head's Vt columns (each piece's Vb of them go to its cb_in_v slot).
IN_MAT_PAGES = 2
IN_VEC_PAGES = 1
L1_BUDGET = 1460288  # usable L1 per Blackhole worker core (risks.md R11)

# ---- CB indices ----
CB_STATE_IN = 0
CB_STATE_IN_MM = 1
CB_STATE = 2
CB_STATE_MM = 3
CB_STATE_OUT = 4
CB_CONV_W = 5
CB_WINDOW = 6
CB_Y_QK = 7
CB_V = 8
CB_SQ = 9
CB_RINV = 10
CB_KQ_ROWS = 11
CB_KNT = 12
CB_D = 13
CB_GATE_IN = 14
CB_GATE_W = 15
CB_GATES = 16
CB_TOK_COEF = 17
CB_READS = 18
CB_U = 19
CB_O_ACC = 20
CB_Z = 21
CB_NORM_W = 22
CB_OUT_ROWS = 23
CB_REDUCE_SCALER = 24
CB_U_F32 = 25
CB_CONV_ACC = 26
CB_SCRATCH = 27
CB_SQ_O = 28
CB_RINV_O = 29
CB_ROW_STAGE = 30
CB_WINDOW_ZEROED = 31
CB_OSUM_PART = 32
CB_OSUM_GATHER = 33
CB_HIST_STAGE = 34
CB_GRAM = 35
CB_CHUNK_VEC = 36
CB_CHUNK_MAT = 37
CB_UW = 38
CB_TRI_MASK = 39
CB_QK_STAGE = 40
CB_KQ_ROWS_F32 = 41
CB_PREP_SEND = 42
CB_O_IN = 43
# R-PREP state core inboxes (written by the prep cores), then the prep core's beta-scaling input v.
CB_IN_KQ = 44
CB_IN_KNT = 45
CB_IN_MAT = 46
CB_IN_VEC = 47
CB_IN_V = 48
CB_V_PREP = 49
NUM_CB_SLOTS = 64
SIGNAL_PAGE_BYTES = 32  # one-page handshake CB
OSUM_SEM_ID = 0  # R-VSPLIT: partial sums o^2 that have landed from peer pieces (cumulative count)
QK_MCAST_FIRST_SEM_ID = 1  # R-BCAST channel semaphores follow
# R-PREP (chunked, never with R-BCAST): per inbox slot, the prepped tile rows that have landed on a state core
# (cumulative); then, on a prep core, the inbox slots its state cores have granted, then the o tiles that have landed
# (both cumulative).
LANDED_FIRST_SEM_ID = 1
# Kernel roles: local = prep and state on the same core (R-STEP / R-VSPLIT / R-CHUNK / R-BCAST); R-PREP splits
# them over prep and state cores.
ROLE_LOCAL, ROLE_PREP, ROLE_STATE = 0, 1, 2
# Per-core q/k prep role (R-BCAST): own = preps its own q and k (no sharing); q / k injector = preps qn / kn and
# multicasts it, receives the other; receiver = receives both.
QK_OWN, QK_Q_INJECTOR, QK_K_INJECTOR, QK_RECEIVER = 0, 1, 2, 3

# Rows staged per flush when a token face row is narrower than the DRAM read alignment (bf16 on
# Blackhole): each half-row is read as one aligned chunk into staging, then copied L1-to-L1.
ROW_STAGE_SLOTS = 32

# Buffers carrying token-tensor bytes (the tensor dtype); every other CB is a Float32 intermediate.
TOKEN_CBS = (
    CB_CONV_W,
    CB_WINDOW,
    CB_GATE_IN,
    CB_GATE_W,
    CB_Z,
    CB_NORM_W,
    CB_OUT_ROWS,
    CB_SCRATCH,
    CB_HIST_STAGE,
)


def _kernel_defines():
    """CB indices and spec constants for all three kernels (as GDN_<name>): defined here once, never restated."""
    names = [n for n in globals() if n.startswith("CB_")] + [
        "HISTORY",
        "CONV_TAPS",
        "TOK_COEF_PAGES",
        "CHUNK_COEF_PAGES",
        "CHUNK_VEC_PAGES",
        "CHUNK_MAT_PAGES",
        "QK_OWN",
        "QK_Q_INJECTOR",
        "QK_K_INJECTOR",
        "QK_RECEIVER",
        "IN_MAT_PAGES",
        "IN_VEC_PAGES",
        "LANDED_FIRST_SEM_ID",
        "O_IN_DEPTH",
        "MAX_PREP_CORES_PER_GROUP",
        "ROLE_LOCAL",
        "ROLE_PREP",
        "ROLE_STATE",
    ]
    return [(f"GDN_{n}", str(globals()[n])) for n in names]


# Pages per CB; each entry is (index list sharing one allocation, pages). Vt here is the piece's column
# block (v_block_tiles); pieces > 1 adds the partial-sum combine buffers, chunked the R-CHUNK buffers.
def _cb_plan(Kt, Vt, conv_on_fpu=False, pieces=1, chunked=False, tokens=0):
    n_state = Kt * Vt
    trb = TOKEN_TILE_ROWS_PER_BLOCK
    return [
        ((CB_STATE_IN, CB_STATE_IN_MM), n_state * STATE_IN_DEPTH),
        ((CB_STATE, CB_STATE_MM), n_state),
        ((CB_STATE_OUT,), n_state * STATE_OUT_DEPTH),
        ((CB_CONV_W,), (1 if conv_on_fpu else CONV_TAPS) * (2 * Kt + Vt)),
        ((CB_WINDOW,), CONV_TAPS * _window_group_cols(Kt, Vt, pieces) * WINDOW_DEPTH),
        ((CB_CONV_ACC,), 0 if conv_on_fpu else CONV_ACC_DEPTH),
        ((CB_Y_QK,), 2 * Kt * trb),
        ((CB_V,), Vt * trb),
        # L2-norm and RMS-norm squares / rsqrt columns get separate buffers: they are pushed in
        # different counts and read by index, so a shared FIFO would leave a phase's block wrapped.
        ((CB_SQ,), Kt * trb),
        ((CB_RINV,), 2 * trb),
        ((CB_SQ_O,), Vt * trb),
        ((CB_RINV_O,), trb),
        # R-BCAST lands kn / qn in place; the U2D view feeds the fp32 qn . kn.
        ((CB_KQ_ROWS, CB_KQ_ROWS_F32), 2 * Kt * trb),
        ((CB_KNT,), Kt * trb),
        ((CB_D,), trb),
        ((CB_GATE_IN,), 2 * ROW_DEPTH),
        ((CB_GATE_W,), 2),
        ((CB_GATES,), 2 * trb),
        ((CB_TOK_COEF,), (CHUNK_COEF_PAGES if chunked else TOK_COEF_PAGES) * TOK_COEF_DEPTH),
        ((CB_READS,), 2 * Vt),
        ((CB_U, CB_U_F32), Vt),
        ((CB_O_ACC,), Vt * trb),
        ((CB_Z,), Vt * ROW_DEPTH),
        ((CB_NORM_W,), Vt),
        ((CB_OUT_ROWS,), Vt * ROW_DEPTH),
        ((CB_SCRATCH,), 1),
        # conv_state history rows, then single token rows, of one window group: one aligned span per column each
        # (reader-local staging).
        ((CB_HIST_STAGE,), 2 * _window_group_cols(Kt, Vt, pieces)),
        # Own partial sum o^2 (compute -> writer); gathered partials of every piece, two tile rows in
        # flight so a peer one tile row ahead never lands on a slot still being summed.
        ((CB_OSUM_PART,), trb if pieces > 1 else 0),
        ((CB_OSUM_GATHER,), 2 * pieces * trb if pieces > 1 else 0),
        # R-CHUNK: [Kn Kn^T; Qn Kn^T]; chunk vectors / decay matrix; solve and output-matmul operands;
        # U scaled by exp(c_last - c) (update operand); the lower-triangular mask (built once).
        ((CB_GRAM,), 2 * trb if chunked else 0),
        ((CB_CHUNK_VEC,), CHUNK_VEC_PAGES if chunked else 0),
        ((CB_CHUNK_MAT,), CHUNK_MAT_PAGES if chunked else 0),
        ((CB_UW,), Vt if chunked else 0),
        ((CB_TRI_MASK,), 2 if chunked else 0),  # the mask, then the identity (R-PREP inverse of I + L)
        # R-BCAST injector: its qn or kn (compute -> writer, multicast source).
        ((CB_QK_STAGE,), Kt if _share_qk(pieces, tokens) else 0),
        # R-PREP prep core: prepped tile rows awaiting its writer; the state cores' chunk outputs o of its tile rows
        # (sized by _prep_cb_plan).
        ((CB_PREP_SEND,), 0),
        ((CB_O_IN,), 0),
        ((CB_IN_KQ,), 0),
        ((CB_IN_KNT,), 0),
        ((CB_IN_MAT,), 0),
        ((CB_IN_VEC,), 0),
        ((CB_IN_V,), 0),
        ((CB_V_PREP,), 0),
    ]


# R-PREP buffers by role. A state core reads only S0 and its inbox and sends o to the tile row's prep core, which
# runs the RMS norm and writes the output; a prep core never touches the state.
PREP_ONLY_CBS = (
    CB_KQ_ROWS,
    CB_KNT,
    CB_CHUNK_VEC,
    CB_CHUNK_MAT,
    CB_Z,
    CB_NORM_W,
    CB_OUT_ROWS,
    CB_SQ_O,
    CB_RINV_O,
    CB_CONV_W,
    CB_WINDOW,
    CB_CONV_ACC,
    CB_Y_QK,
    CB_SQ,
    CB_RINV,
    CB_GATE_IN,
    CB_GATE_W,
    CB_GATES,
    CB_TOK_COEF,
    CB_HIST_STAGE,
    CB_GRAM,
    CB_TRI_MASK,
)
STATE_ONLY_CBS = (
    CB_STATE_IN,
    CB_STATE,
    CB_STATE_OUT,
    CB_READS,
    CB_U,
    CB_O_ACC,
    CB_SCRATCH,
    CB_UW,
    CB_V,
)


def _head_block_pages(Kt, Vt):
    """Tiles of one value head's block of a prepped tile row (see IN_MAT_PAGES)."""
    return 3 * Kt + IN_MAT_PAGES + IN_VEC_PAGES + Vt


def _inbox_pages(Kt, Vb):
    """Pages per inbox slot of a state core; a prep core writes slot s at base + s * slot bytes."""
    return {CB_IN_KQ: 2 * Kt, CB_IN_KNT: Kt, CB_IN_MAT: IN_MAT_PAGES, CB_IN_VEC: IN_VEC_PAGES, CB_IN_V: Vb}


# Buffers written by other cores (R-PREP): allocated first and on every core, so each has one L1 address everywhere,
# which the writers take from their own allocation; every other buffer exists only on the role that uses it.
REMOTE_CBS = (CB_IN_KQ, CB_IN_KNT, CB_IN_MAT, CB_IN_VEC, CB_IN_V, CB_O_IN)


def _prep_cb_plan(Kt, Vt, ratio, conv_on_fpu, pieces, inbox):
    """R-PREP plans as (shared, state, prep) lists of (indices, pages): the remotely written buffers (on every core,
    sized for the role that receives them; the inbox buffers are exactly `inbox` slots, so tile row i lands in slot
    i % inbox), then each role's own buffers."""
    Vb = Vt // pieces
    state = {idx[0]: pages for idx, pages in _cb_plan(Kt, Vb, conv_on_fpu, pieces, True)}
    prep = {idx[0]: pages for idx, pages in _cb_plan(Kt, ratio * Vt, conv_on_fpu, 1, True)}
    for idx in PREP_ONLY_CBS:
        state[idx] = 0
    for idx in STATE_ONLY_CBS:
        prep[idx] = 0
    for idx in (CB_OSUM_PART, CB_OSUM_GATHER, CB_D):  # no piece combine (the prep core's RMS norm sees all), no d
        state[idx] = prep[idx] = 0
    state[CB_O_ACC] = O_ACC_DEPTH * Vb
    prep[CB_CHUNK_VEC] = CHUNK_VEC_PAGES + 1  # the chunk vectors, then exp(c_last - c)^T
    prep[CB_CHUNK_MAT] = 1  # L, the operand of the inverse
    prep[CB_V_PREP] = ratio * Vt
    prep[CB_PREP_SEND] = (PREP_SEND_HEADS or ratio) * _head_block_pages(Kt, Vt)
    prep[CB_O_IN] = O_IN_DEPTH * ratio * Vt
    prep[CB_NORM_W] = Vt
    prep[CB_RINV_O] = ratio * TOKEN_TILE_ROWS_PER_BLOCK
    for idx, pages in _inbox_pages(Kt, Vb).items():
        state[idx] = inbox * pages
    state[CB_V] = 0  # the state core reads v from its inbox
    order = [idx for idx, _ in _cb_plan(Kt, Vb, conv_on_fpu, pieces, True)]
    shared = [(idx, max(state[idx[0]], prep[idx[0]])) for idx in order if idx[0] in REMOTE_CBS]
    own = [[(idx, plan[idx[0]]) for idx in order if idx[0] not in REMOTE_CBS] for plan in (state, prep)]
    return shared, own[0], own[1]


def _prep_plan_bytes(plans, dtype):
    """Per-core footprint of the larger role."""
    shared, state, prep = plans
    return max(_plan_bytes(shared + own, dtype) for own in (state, prep))


def _share_qk(pieces, tokens):
    """R-BCAST predicate: one unit per core (R-VSPLIT) and one tile row per unit (decode), so each core takes
    one q/k block, into CB slots that are still at their base address on every core."""
    return SHARE_QK_PREP and pieces > 1 and tokens == 1


# Indices read by copy_tile / SFPU at full fp32 (UnpackToDestFp32) when Float32; the rest feed srcA/srcB
# or dataflow. A bf16 token buffer reaches DEST losslessly through srcA, so it is never tagged.
U2D_CBS = (
    CB_STATE_IN,
    CB_STATE,
    CB_CONV_W,
    CB_WINDOW,
    CB_CONV_ACC,
    CB_Y_QK,
    CB_V,
    CB_SQ,
    CB_RINV,
    CB_D,
    CB_KQ_ROWS_F32,
    CB_GATE_IN,
    CB_GATE_W,
    CB_TOK_COEF,
    CB_READS,
    CB_U_F32,
    CB_O_ACC,
    CB_Z,
    CB_NORM_W,
    CB_OSUM_GATHER,
    CB_GRAM,
    CB_CHUNK_VEC,
    CB_TRI_MASK,
    CB_O_IN,
    CB_IN_V,
    CB_V_PREP,
)


def _window_group_cols(Kt, Vt, pieces):
    """Column tiles per conv-window hand-off: a divisor of the unit's 2Kt + Vt columns, so every
    reservation stays contiguous in cb_window (compute consumes per column, so a group may straddle the
    q / k / v boundaries). R-STEP: the largest one up to WINDOW_GROUP_MAX_COLS. R-VSPLIT: all of them,
    since a piece's reader is bound by serial DRAM round trips and its state slice leaves L1 room."""
    cols = 2 * Kt + Vt
    return cols if pieces > 1 else _largest_divisor_at_most(cols, WINDOW_GROUP_MAX_COLS)


def _conv_on_fpu(dtype):
    return dtype == ttnn.bfloat16 if CONV_ON_FPU == "auto" else bool(CONV_ON_FPU)


def _tile_bytes(dtype):
    return FP32_TILE_BYTES if dtype == ttnn.float32 else BF16_TILE_BYTES


def _row_stage_bytes(token_tile_bytes):
    """Staging bytes for sub-alignment row reads; 0 when a face row already meets DRAM alignment."""
    seg_bytes = token_tile_bytes // (TILE * TILE) * (TILE // 2)
    align = ttnn.get_dram_alignment()
    return 0 if seg_bytes % align == 0 else (ROW_STAGE_SLOTS + 1) * align


def _plan_bytes(plan, dtype):
    token_bytes = _tile_bytes(dtype)
    cb_bytes = sum(pages * (token_bytes if indices[0] in TOKEN_CBS else FP32_TILE_BYTES) for indices, pages in plan)
    return cb_bytes + BF16_TILE_BYTES + SIGNAL_PAGE_BYTES * WINDOW_DEPTH + _row_stage_bytes(token_bytes)


def l1_footprint_bytes(Kt, Vt, dtype, pieces=1, chunked=True, tokens=0):
    """Per-core footprint of one unit (Vt = the unit's column block). validate() checks R-STEP (the
    whole Vt); the descriptor re-checks the regime it selects."""
    return _plan_bytes(_cb_plan(Kt, Vt, _conv_on_fpu(dtype), pieces, chunked, tokens), dtype)


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _largest_divisor_at_most(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _value_pieces(num_pairs, Vt, num_cores):
    """Pieces per pair (Vt / v_block_tiles). R-VSPLIT needs one unit per core: the pieces of a pair
    exchange partial sums once per tile row, so every core must run exactly one unit in lockstep."""
    if V_BLOCK_TILES is not None:
        assert Vt % V_BLOCK_TILES == 0, "V_BLOCK_TILES must divide Vt"
        pieces = Vt // V_BLOCK_TILES
        assert pieces <= MAX_PIECES and (pieces == 1 or num_pairs * pieces <= num_cores)
        return pieces
    fits = [d for d in range(1, min(Vt, MAX_PIECES) + 1) if Vt % d == 0 and num_pairs * d <= num_cores]
    return max(fits, default=1)


def _chunked(tokens, device):
    """R-CHUNK predicate: multi-token calls on Blackhole (the chunk solve is a Blackhole SFPU op)."""
    return tokens >= CHUNK_MIN_TOKENS and ttnn.device.is_blackhole(device)


def _seq_tile_rows(seq, tokens):
    return (seq * tokens + tokens - 1) // TILE - (seq * tokens) // TILE + 1


def _prep_layout(batch, value_heads, key_heads, tokens, Kt, Vt, pieces, chunked, dtype, spare_cores):
    """R-PREP plan, or None to keep prep and state on the same core: (prep cores per group cap, inbox depth).
    Needs one unit per core (R-VSPLIT), at least one idle core per (sequence, key head) group, the key head's
    value heads in one gate tile column (one gate tile per prepped tile row) and the merged plan inside L1."""
    ratio = value_heads // key_heads
    if not (PREP_OFFLOAD and chunked and pieces > 1):
        return None
    if any((hk * ratio) // TILE != (hk * ratio + ratio - 1) // TILE for hk in range(key_heads)):
        return None
    if min(_seq_tile_rows(seq, tokens) for seq in range(batch)) < PREP_MIN_TILE_ROWS:
        return None
    per_group = min(spare_cores // (batch * key_heads), MAX_PREP_CORES_PER_GROUP)
    if per_group < 1:
        return None
    # A prep core's next tile row is granted only after its current one has landed when its stride is at least
    # the inbox depth (the cumulative credit count relies on it).
    inbox = min([INBOX_DEPTH, per_group] + [_seq_tile_rows(seq, tokens) for seq in range(batch)])
    # The deepest inbox that fits L1 (a shallower one only costs slack between a grant and its landing).
    while inbox >= 1:
        if _prep_plan_bytes(_prep_cb_plan(Kt, Vt, ratio, _conv_on_fpu(dtype), pieces, inbox), dtype) <= L1_BUDGET:
            return per_group, inbox
        inbox -= 1
    return None


def _core_ranges(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _mcast_rects(device, cores):
    """Multicast rectangles covering exactly `cores` (logical, row-major order): one per run of cores consecutive in
    one row of virtual NoC coordinates (non-worker columns split a row), as [count, then (x_lo, y_lo, x_hi, y_hi,
    cores) per run]; the kernel orders the corners for its NoC."""
    runs = []
    for core in cores:
        virt = device.worker_core_from_logical_core(core)
        if runs and runs[-1][-1].y == virt.y and runs[-1][-1].x + 1 == virt.x:
            runs[-1].append(virt)
        else:
            runs.append([virt])
    args = [len(runs)]
    for run in runs:
        args += [run[0].x, run[0].y, run[-1].x, run[-1].y, len(run)]
    return args


def create_program_descriptor(inputs, outputs, *, epsilon, compute_kernel_config):
    mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight = inputs
    output, conv_state_out, recurrent_state_out = outputs

    assert TOKEN_TILE_ROWS_PER_BLOCK == 1, "only one tile row per prep block is built"
    batch, value_heads, key_dim, value_dim = (int(d) for d in recurrent_state.shape)
    rows, channels = int(mixed_qkv.shape[1]), int(mixed_qkv.shape[2])
    tokens = rows // batch
    Kt, Vt = key_dim // TILE, value_dim // TILE
    key_heads = (channels - value_heads * value_dim) // (2 * key_dim)
    Ct = channels // TILE
    Gt = (value_heads + TILE - 1) // TILE

    # ---- work split: units (pair, piece); R-STEP blocks of PAIRS_PER_BLOCK pairs ----
    device = mixed_qkv.device()
    grid = device.compute_with_storage_grid_size()
    if MAX_CORES is not None:
        grid = ttnn.CoreCoord(grid.x, max(1, min(grid.y, MAX_CORES // grid.x)))
    num_pairs = batch * value_heads
    num_grid_cores = grid.x * grid.y
    pieces = _value_pieces(num_pairs, Vt, num_grid_cores)
    chunked = _chunked(tokens, device)
    if V_BLOCK_TILES is None and pieces > PREP_MAX_PIECES:
        capped = _largest_divisor_at_most(Vt, PREP_MAX_PIECES)
        if capped > 1 and _prep_layout(
            batch,
            value_heads,
            key_heads,
            tokens,
            Kt,
            Vt,
            capped,
            chunked,
            mixed_qkv.dtype,
            num_grid_cores - num_pairs * capped,
        ):
            pieces = capped
    v_block_tiles = Vt // pieces
    num_units = num_pairs * pieces
    units_per_block = PAIRS_PER_BLOCK if pieces == 1 else 1  # R-VSPLIT: one unit per core
    num_blocks = (num_units + units_per_block - 1) // units_per_block
    (_, all_cores, group_1, group_2, blocks_1, blocks_2) = ttnn.split_work_to_cores(grid, num_blocks, row_wise=True)

    assignment = []  # (core, first_unit, unit_count)
    next_block = 0
    for group, per_core in ((group_1, blocks_1), (group_2, blocks_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            first_unit = next_block * units_per_block
            last_unit = min(num_units, (next_block + per_core) * units_per_block)
            assignment.append((core, first_unit, last_unit - first_unit))
            next_block += per_core
    assert pieces == 1 or all(count == 1 for _, _, count in assignment)
    ratio = value_heads // key_heads
    token_dtype = mixed_qkv.dtype
    conv_on_fpu = _conv_on_fpu(token_dtype)
    grid_cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    state_cores = [core for core, _, _ in assignment]
    state_set = {(c.x, c.y) for c in state_cores}
    spare = [c for c in grid_cores if (c.x, c.y) not in state_set]
    prep = _prep_layout(batch, value_heads, key_heads, tokens, Kt, Vt, pieces, chunked, token_dtype, len(spare))
    if prep is None:
        inbox = 0
        cb_plan = _cb_plan(Kt, v_block_tiles, conv_on_fpu, pieces, chunked, tokens)
        assert _plan_bytes(cb_plan, token_dtype) <= L1_BUDGET
    else:
        per_group, inbox = prep
        prep_plans = _prep_cb_plan(Kt, Vt, ratio, conv_on_fpu, pieces, inbox)
        assert _prep_plan_bytes(prep_plans, token_dtype) <= L1_BUDGET
    # NoC coordinates of the core running each unit (R-VSPLIT peers).
    unit_noc = {}
    unit_cores = []  # logical core of each unit when one unit per core (R-VSPLIT)
    for core, first_unit, _ in assignment:
        noc_core = device.worker_core_from_logical_core(core)
        unit_noc[first_unit] = (noc_core.x, noc_core.y)
        unit_cores.append(core)

    # R-BCAST: units of one (sequence, key head) are r * pieces consecutive units, so on one unit per core they
    # are consecutive cores in row-major order: one multicast group each, with the q injector its first core
    # and the k injector its second (two channels, one sender each).
    qk_group = (value_heads // key_heads) * pieces
    qk_channels = []
    if _share_qk(pieces, tokens):
        order = [(core.y, core.x) for core, _, _ in assignment]
        assert order == sorted(order), "units must be assigned in row-major core order"
        # One round per channel per call: no handshake (each receiver's landing CB is untouched before it) and a
        # flag signal linked behind the payload (a counter would wait a write barrier per rectangle). The
        # injectors' writers send on the writer's NoC (WriterConfigDescriptor: NOC 1, the DRAM-write NoC).
        config = ttnn.McastConfig(noc=ttnn.NOC.RISCV_1_default, handshake=False, data_ready=ttnn.McastDataReady.Flag)
        qk_channels = [
            ttnn.Mcast(
                device,
                config,
                all_cores,
                qk_group,
                ttnn.McastFixedSenderConfig(sender_index=sender),
                ttnn.McastCoreOrder.RowMajor,
            )
            for sender in (0, 1)
        ]

    def qk_role(unit):
        if not qk_channels:
            return QK_OWN
        return {0: QK_Q_INJECTOR, 1: QK_K_INJECTOR}.get(unit % qk_group, QK_RECEIVER)

    token_tile_bytes = _tile_bytes(token_dtype)

    def window_zeroing(unit_cols, group_cols, tile_rows):
        """(slots, bytes per slot) of cb_window the writer zeroes: only the window groups this core will use."""
        groups = tile_rows * (unit_cols // group_cols)
        return min(WINDOW_DEPTH, groups), CONV_TAPS * group_cols * token_tile_bytes

    def unit_zeroing(first_unit, count):
        groups, group_cols = 0, 0
        for unit in range(first_unit, first_unit + count):
            tile_rows = _seq_tile_rows(unit // pieces // value_heads, tokens)
            unit_cols = v_block_tiles + {QK_OWN: 2 * Kt, QK_RECEIVER: 0}.get(qk_role(unit), Kt)
            group_cols = unit_cols if pieces > 1 else _window_group_cols(Kt, v_block_tiles, pieces)
            groups += tile_rows * (unit_cols // group_cols)
        return window_zeroing(group_cols, group_cols, groups)

    # ---- R-PREP: prep cores per (sequence, key head) group, round robin over the group's tile rows ----
    prep_units = []  # (core, group first unit, stride position, stride)
    group_producers = {}  # group first unit -> NoC coordinates of its prep cores in stride order
    if prep is not None:
        groups = [(seq, hk) for seq in range(batch) for hk in range(key_heads)]
        strides = [min(per_group, _seq_tile_rows(seq, tokens)) for seq, _ in groups]
        # The groups take idle cores in turn, each the one nearest its own state cores (short multicast paths).
        centers = []
        for seq, hk in groups:
            first = (seq * value_heads + hk * ratio) * pieces
            members = unit_cores[first : first + ratio * pieces]
            centers.append((sum(c.x for c in members) / len(members), sum(c.y for c in members) / len(members)))
        free = list(spare)
        picks = [[] for _ in groups]
        for q in range(max(strides)):
            for g, (cx, cy) in enumerate(centers):
                if q < strides[g]:
                    core = min(free, key=lambda c: abs(c.x - cx) + abs(c.y - cy))
                    free.remove(core)
                    picks[g].append(core)
        for g, (seq, hk) in enumerate(groups):
            group_first = (seq * value_heads + hk * ratio) * pieces
            for q, core in enumerate(picks[g]):
                prep_units.append((core, group_first, q, strides[g]))
                noc_core = device.worker_core_from_logical_core(core)
                group_producers.setdefault(group_first, []).extend([noc_core.x, noc_core.y])
    prep_cores = [core for core, _, _, _ in prep_units]
    all_cores = _core_ranges(state_cores + prep_cores) if prep_units else all_cores

    # ---- circular buffers ----
    cbs = []

    def add_cbs(plan, core_ranges):
        for indices, pages in plan:
            if pages == 0:
                continue
            is_token = indices[0] in TOKEN_CBS
            dtype = token_dtype if is_token else ttnn.float32
            page_bytes = token_tile_bytes if is_token else FP32_TILE_BYTES
            formats = [
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page_bytes) for idx in indices
            ]
            cbs.append(
                ttnn.CBDescriptor(total_size=pages * page_bytes, core_ranges=core_ranges, format_descriptors=formats)
            )

    if prep is None:
        add_cbs(cb_plan, all_cores)
    else:
        # Allocation follows this order: the shared buffers first, so their addresses agree on every core.
        add_cbs(prep_plans[0], all_cores)
        add_cbs(prep_plans[1], _core_ranges(state_cores))
        add_cbs(prep_plans[2], _core_ranges(prep_cores))
    row_stage_bytes = _row_stage_bytes(token_tile_bytes)
    if row_stage_bytes:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=row_stage_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_ROW_STAGE, data_format=token_dtype, page_size=row_stage_bytes
                    )
                ],
            )
        )
    # Writer-to-reader tokens: one per cb_window slot, as each slot has been zeroed.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=SIGNAL_PAGE_BYTES * WINDOW_DEPTH,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_WINDOW_ZEROED, data_format=ttnn.bfloat16, page_size=SIGNAL_PAGE_BYTES
                )
            ],
        )
    )
    cbs.append(
        ttnn.CBDescriptor(
            total_size=BF16_TILE_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_REDUCE_SCALER, data_format=ttnn.bfloat16, page_size=BF16_TILE_BYTES
                )
            ],
        )
    )

    # ---- compile-time args per role (a prep core's unit is a key head: its r value heads' v columns) ----
    reader_tensors = (
        mixed_qkv,
        z,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        conv_state_out,
    )
    # Token tile bytes, state tile bytes.
    geometry_ct = [token_tile_bytes, FP32_TILE_BYTES]
    reader_accessor_ct = [arg for t in reader_tensors for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    writer_accessor_ct = [
        arg for t in (output, recurrent_state_out) for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()
    ]
    reader_addrs = [t.buffer_address() for t in reader_tensors]
    writer_addrs = [output.buffer_address(), recurrent_state_out.buffer_address()]

    def role_ct(role):
        unit_v = ratio * Vt if role == ROLE_PREP else v_block_tiles
        role_pieces = 1 if role == ROLE_PREP else pieces
        group_cols = _window_group_cols(Kt, unit_v, role_pieces)
        reader_ct = [Kt, Vt, value_heads, key_heads, Ct, Gt, *geometry_ct, ttnn.get_dram_alignment()]
        reader_ct += [ROW_STAGE_SLOTS if row_stage_bytes else 0, int(conv_on_fpu), group_cols, WINDOW_DEPTH]
        reader_ct += [pieces, int(chunked), role, inbox, *reader_accessor_ct]
        writer_ct = [Kt, Vt, value_heads, *geometry_ct, pieces, OSUM_SEM_ID, int(chunked), role, inbox, ratio]
        writer_ct += writer_accessor_ct
        compute_ct = [
            Kt,
            unit_v,
            value_heads,
            _largest_divisor_at_most(unit_v, 4),  # read-matmul subblock width (fp32 half-sync DEST: 4 tiles)
            _float_bits(value_dim**0.5),
            _float_bits(key_dim**-0.5),
            int(conv_on_fpu),
            pieces,
            group_cols,
            int(chunked),
            int(bool(qk_channels)),
            role,
            ratio,
        ]
        return reader_ct, writer_ct, compute_ct

    v_eps_bits = _float_bits(value_dim * epsilon)
    roles = {}  # role -> (cores, reader rt, writer rt, compute rt)
    state_role = ROLE_LOCAL if prep is None else ROLE_STATE
    rts = (ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs())
    for core, first_unit, count in assignment:
        role = qk_role(first_unit)  # R-BCAST runs one unit per core
        zero_slots, zero_slot_bytes = unit_zeroing(first_unit, count) if prep is None else (0, 0)
        rows = [0, 1]  # tile rows: every one of the unit's sequence
        producers = []
        if prep is not None:
            group_first = first_unit - first_unit % (ratio * pieces)
            producers = [len(group_producers[group_first]) // 2] + group_producers[group_first]
        rts[0][core.x][core.y] = reader_addrs + [first_unit, count, tokens, role, zero_slots] + rows + producers
        # R-VSPLIT: NoC coordinates of the pair's piece cores, in piece order.
        group_first = first_unit - first_unit % pieces
        peers = [c for s in range(pieces) for c in unit_noc[group_first + s]] if pieces > 1 else []
        rts[1][core.x][core.y] = (
            writer_addrs + [first_unit, count, tokens, role, zero_slots, zero_slot_bytes] + rows + peers + producers
        )
        rts[2][core.x][core.y] = [first_unit, count, tokens, v_eps_bits, role] + rows
    roles[state_role] = (state_cores, *rts)
    if prep_units:
        rts = (ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs())
        for core, group_first, q, stride in prep_units:
            seq = group_first // pieces // value_heads
            my_rows = len(range(q, _seq_tile_rows(seq, tokens), stride))
            group_cols = _window_group_cols(Kt, ratio * Vt, 1)
            zero_slots, zero_slot_bytes = window_zeroing(2 * Kt + ratio * Vt, group_cols, my_rows)
            rows = [q, stride]
            consumers = [c for u in range(ratio * pieces) for c in unit_noc[group_first + u]]
            # A head block goes to the head's pieces: one multicast per row run.
            for j in range(ratio):
                first = group_first + j * pieces
                consumers += _mcast_rects(device, unit_cores[first : first + pieces])
            rts[0][core.x][core.y] = reader_addrs + [group_first, 1, tokens, QK_OWN, zero_slots] + rows
            rts[1][core.x][core.y] = (
                writer_addrs + [group_first, 1, tokens, QK_OWN, zero_slots, zero_slot_bytes] + rows + consumers
            )
            rts[2][core.x][core.y] = [group_first, 1, tokens, v_eps_bits, QK_OWN] + rows
        roles[ROLE_PREP] = (prep_cores, *rts)

    modes = list(getattr(compute_kernel_config, "unpack_to_dest_mode", []) or [])
    if len(modes) < NUM_CB_SLOTS:
        modes = modes + [ttnn.UnpackToDestMode.Default] * (NUM_CB_SLOTS - len(modes))
    for idx in U2D_CBS:
        fp32_cb = idx not in TOKEN_CBS or token_dtype == ttnn.float32
        modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32 if fp32_cb else ttnn.UnpackToDestMode.Default
    compute_kernel_config.unpack_to_dest_mode = modes
    if DST_FULL_SYNC:
        compute_kernel_config.dst_full_sync_en = True

    defines = _kernel_defines()
    kernels = []
    writers = []
    for role, (cores, reader_rt, writer_rt, compute_rt) in roles.items():
        ranges = _core_ranges(cores) if prep_units else all_cores
        reader_ct, writer_ct, compute_ct = role_ct(role)
        reader_config, writer_config = ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor()
        if role == ROLE_PREP and PREP_SEND_NOC_1:
            # The prep cores sit below the state cores in row-major order: NOC 1 (which routes toward lower y) reaches
            # them directly, NOC 0 only around the torus. The reader takes the other NoC.
            writer_config = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
            )
            reader_config = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
            )
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_writer.cpp"),
            core_ranges=ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=defines,
            config=writer_config,
        )
        writers.append(writer)
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gated_delta_net_reader.cpp"),
                core_ranges=ranges,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                defines=defines,
                config=reader_config,
            ),
            writer,
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "gated_delta_net_compute.cpp"),
                core_ranges=ranges,
                compile_time_args=compute_ct,
                defines=defines + [("INP_FLOAT32", "1")],
                runtime_args=compute_rt,
                config=compute_kernel_config,
            ),
        ]
    semaphores = []
    if pieces > 1:
        semaphores.append(ttnn.SemaphoreDescriptor(id=OSUM_SEM_ID, core_ranges=all_cores, initial_value=0))
    if prep_units:
        # Landed counters per inbox slot (state cores), then the grant and o-landed counters (prep cores).
        for sem in range(LANDED_FIRST_SEM_ID, LANDED_FIRST_SEM_ID + inbox + 2):
            semaphores.append(ttnn.SemaphoreDescriptor(id=sem, core_ranges=all_cores, initial_value=0))
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    # The writer carries the q and k multicast channels (absent outside R-BCAST, so one kernel source).
    next_sem = QK_MCAST_FIRST_SEM_ID
    for prefix, channel in zip(("qkq", "qkk"), qk_channels or (None, None)):
        for writer in writers:
            if channel is None:
                ttnn.attach_absent_mcast(writer, prefix)
            else:
                channel.attach(descriptor, prefix, [writer], next_sem)
        if channel is not None:
            next_sem = channel.next_semaphore_id()
    descriptor.kernels = kernels
    return descriptor
