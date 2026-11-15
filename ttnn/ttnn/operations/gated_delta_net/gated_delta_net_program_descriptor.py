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
R-QKCORE (decode R-STEP, every sequence in one tile row) preps qn / kn once per key head for all sequences on a q and a
k core, which multicast them to the key head's pair cores and then write conv_state_out as whole tiles.
"""

import struct
from collections import OrderedDict
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
# R-BCAST: place the units on a grid narrowed to a width dividing the key-head group, so each group's q / k multicast
# covers one rectangle (a 12-core group on the 11-wide grid spans two rows). perf_decode_b1: send 1.28 -> 1.22 us, but the
# 6-wide region reads later: 22.95 vs 22.49 us (median of 10), so off.
BCAST_RECT_GROUPS = False
# R-QKCORE (decode R-STEP, one tile row of sequences): per key head a q core and a k core convolve and L2-normalize the
# key head's q (k) columns of every sequence of the tile row at once and multicast qn (kn) to the key head's pair cores,
# which read and convolve only their v columns. Pair cores run consecutive sequences of one value head.
SHARE_QK_CORES = True
# R-PREP (chunked R-VSPLIT): the per-tile-row preparation, which depends on the token rows alone (gates, conv,
# L2 norm, [Kn Kn^T; Qn Kn^T], chunk vectors, L and QD), runs on the cores R-VSPLIT leaves idle: per (sequence,
# key head) up to MAX_PREP_CORES_PER_GROUP prep cores take its tile rows round robin and write each prepped
# tile row into the inboxes of the key head's r * pieces state cores, which run only the state part of the
# chunk step and the RMS norm.
PREP_OFFLOAD = True
# Prep cores per group, at most: the state cores' producer tables hold this many (more than about twice a prep tile
# row's time over a state step buys nothing).
MAX_PREP_CORES_PER_GROUP = 32
# R-PREP: head blocks multicast to a head's pieces (one write per run of consecutive cores); False: one unicast each.
# Unicast and multicast measure the same on long calls (model_prefill_4096 920 us); unicast is steadier on short ones.
PREP_MCAST = False
# R-PREP: each group's state cores placed among its own prep cores (else the state cores fill the first grid rows and
# the prep cores take the idle cores nearest them). model_prefill_4096 1.08-1.15 ms -> 922 us, T=1024 390 -> 307 us.
PREP_CLUSTER = True
# R-PREP prep writer NoC (the reader takes the other): 1, 0, or "auto" (per core, the shorter paths to its state cores).
PREP_SEND_NOC = "auto"
NOC_GRID = (17, 12)  # Blackhole NoC torus (x, y), in NOC 0 coordinates (R-PREP is Blackhole-only)
# Pieces per pair under R-PREP, at most; _prep_estimate picks among them. Fewer pieces leave more prep cores and halve
# the fan-out of a prepped row, more shorten the state step: T=128 4 pieces 87 us, 2 pieces 92; T=512 255 vs 199;
# model_prefill_4096 4 pieces 1.34 ms (prep-bound), 2 pieces 1.18 ms (R6).
PREP_MAX_PIECES = 4
# R-PREP fills its pipeline once (one prep core's tile row) and then runs a tile row per state step; below this many tile
# rows per sequence the local chunk step is faster (model geometry, one value head per prep group: T=32 67 us local,
# 73 us R-PREP; T=64 118 vs 80 us).
PREP_MIN_TILE_ROWS = 2
# R-PREP: value heads per prep group (a divisor of the key head's r value heads). A group's prep cores take its tile
# rows round robin and prep its heads only, each redoing the key head's q / k prep: fewer heads spread one tile row's
# prep over more cores (short calls, whose time is the pipeline fill), all r share the q / k prep (long calls, where
# prep throughput binds). None: auto (_prep_heads).
PREP_HEADS = None
# R-PREP q / k cores (Refinement 14): per (sequence, key head, stride position) a q core and a k core convolve and
# L2-normalize the key head's q (k) columns of the position's tile rows once and write qn (kn) into cb_kq_rows of the
# position's prep cores of every value-head group of the key head, which then convolve only their v columns. True /
# False, or "auto": where _prep_estimate models it faster (short calls, whose time is the pipeline fill).
PREP_SHARE_QK = "auto"
PREP_KQ_DEPTH = 2  # tile rows of landed qn / kn a prep core holds (slots of its cb_kq_rows)
PREP_QK_GROUP_COLS = None  # q / k core columns per window hand-off (a divisor of Kt); None: as a prep core's
# _prep_estimate cost model, device us at the model geometry (Kt = Vt = 4, bf16): a prep core's tile row is the q / k
# part plus one part per value head, the state step one tile row (a base plus one part per state column tile of the
# piece). Fits T = 64..4096: h = 1 wins to 16 tile rows per sequence (T=512 199 vs 214 us), h = 3 from 32 (T=1024
# 335 vs 302 us); 4 pieces while every row has its own prep core (T=128 87 vs 92 us).
PREP_QK_ROW_US = 27.0
# PREP_SHARE_QK: what a prep core's tile row keeps of the q / k part (kn^T, the landing), and the q / k cores' row until
# qn / kn have landed (beyond the head part). Fit on the shared layout (2 pieces, 4 positions) at T = 128 / 256 / 512 /
# 1024: 69 / 103 / 182 / 341 us; unshared T=512 196 us, T=1024 299 us (h = 3).
PREP_KQ_ROW_US = 4.0
PREP_QK_FILL_US = 14.0
PREP_HEAD_ROW_US = 30.0
STATE_STEP_US = 2.9
STATE_STEP_COL_US = 1.2

# ---- Buffer-depth knobs ----
WINDOW_DEPTH = 2  # window groups of tap tiles in flight
WINDOW_GROUP_MAX_COLS = 4  # column tiles per window hand-off (one DRAM flush each), upper bound
# R-PREP prep cores with at most this many columns read a tile row's window as one group. Measured slower (T=128 82.7
# vs 87.8 us at 12): building the taps is bound by this RISC's issue of row copies, not DRAM round trips, and conv then
# waits for every column. 0: off.
PREP_WINDOW_MAX_COLS = 0
TOK_COEF_DEPTH = 2  # tokens of coefficient pages in flight
INBOX_DEPTH = 12  # R-PREP: prepped tile rows in flight per state core (slots of its inbox buffers), at most
O_IN_DEPTH = 2  # R-PREP: a prep core's tile rows of chunk outputs o in flight (slot k % O_IN_DEPTH for its k-th row)
O_ACC_DEPTH = 2  # R-PREP: a state core's tile rows of o awaiting its writer
PREP_SEND_HEADS = None  # R-PREP: head blocks a prep core holds for its writer; None = one tile row's (r)
ROW_DEPTH = 2  # tile rows of gate / z / output tiles in flight
# Per-unit inputs (S0, conv / gate / norm weights) in flight on a core that runs several units: the next unit's reads
# overlap this unit's compute instead of waiting for its last pop (decode B=16: 2 units on most cores).
UNIT_INPUT_DEPTH = 2
# Pair states (S0) in flight, at most unit_depth (decode B=16: 1 measured 95.5 us vs 93 us at 2).
STATE_IN_DEPTH = 2
# R-QKCORE: pair cores in logical rows [0, SWAP_NOC_ROWS) read on NOC 1 and write on NOC 0 (0: none). NOC 0 brings the
# top rows' read responses around the torus (their weights / gate rows landed at 25-31 us vs 4-5 us on the lower rows).
# Decode B=8 / B=16: 0 rows 48.8 / 86 us, 2 rows 42.3 / 73.9, 3 rows 42.6 / 74.1, 4 rows 51.6 / 78, 5 rows 56 / 90.
SWAP_NOC_ROWS = 3
# S0 tiles per DRAM read barrier on a core (None: the whole slice at once). Bounds the bulk a core queues at the banks
# ahead of other cores' small window reads.
S0_READ_TILES = None
# R-QKCORE pair cores running several units (sequences of one value head, so one key head): kn^T and d are computed
# once per core and held for every unit (SHARE_QK_FINISH), and a unit's RMS norm runs after the next unit's token step
# (UNIT_LAG), so the core's last state is final, and written, while the norms run.
SHARE_QK_FINISH = True
UNIT_LAG = True
# R-QKCORE pair cores: pair p updates and writes its final state starting at tile (p * STATE_STAGGER) mod Kt Vt, so the
# cores' simultaneous writes spread over the DRAM banks instead of all starting at the bank of tile 0 (0: in order).
STATE_STAGGER = 3
# Decode: a unit's S0 read issued with its window rows (own transaction id) instead of after the weights / gate rows.
# Measured with the NoC swap at B=8 / B=16: 44.7 / 75 us on vs 42.8 / 74.4 off, so off (0).
EARLY_S0 = 0
# Finished pair states awaiting the writer, at most unit_depth: the writer issues a core's non-last state after the next
# unit's coefficients, so compute's next update must not wait for it.
STATE_OUT_DEPTH = 2
CONV_ACC_DEPTH = 1  # partial tap sum: compute-to-compute, one compute thread, so depth 1

HISTORY = 3  # spec constant: conv_state rows per sequence
CONV_TAPS = HISTORY + 1  # history rows + current token, not a knob
TOK_COEF_PAGES = 3  # alpha, beta and 1 (row selector) at the token's row p, 0 elsewhere
D_PAGES = 2  # token step: [kn; qn] x kn^T, d = qn . kn of row p at (p, p) of the second tile
CHUNK_COEF_PAGES = 2  # R-CHUNK, per tile row: g and beta of the pair in column 0, 0 on rows that are no token
CHUNK_VEC_PAGES = 5  # R-CHUNK: D, c, exp(c), exp(c_last - c), beta exp(c)
CHUNK_MAT_PAGES = 2  # R-CHUNK: L (solve operand), (Qn Kn^T) * D (matmul operand)
# R-PREP head block, per value head and tile row, prescaled so the state core broadcasts nothing, in pack order (each
# part is sent as compute pushes it): exp(c) (cb_in_vec: the chunk decay exp(c_last) is its last token row), QD and
# M = (I + L)^-1 (cb_in_mat), [beta exp(c) (*) kn, exp(c) (*) qn] (2 Kt, its cb_in_kq slot),
# kn^T (*) exp(c_last - c)^T (Kt, cb_in_knt), then beta (*) v of the head's Vt columns (each piece's Vb of them go to
# its cb_in_v slot).
IN_MAT_PAGES = 2
IN_VEC_PAGES = 1
# The static CB region ends at or below this L1 address on every core (Refinement 12): the caller keeps L1 tensors
# alive across the call (the model's residual stream is L1 width-sharded), allocated downward from the top of L1.
L1_CB_REGION_END = 1_200_000

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
CB_ZW = 50  # R-PREP prep core: norm_weight (*)row silu(z) of its rows, ahead of the o that the RMS norm scales
NUM_CB_SLOTS = 64
SIGNAL_PAGE_BYTES = 32  # one-page handshake CB
OSUM_SEM_ID = 0  # R-VSPLIT: partial sums o^2 that have landed from peer pieces (cumulative count)
# In-place conv_state (conv_state_out is conv_state): reader cores whose conv_state reads have landed (cumulative), on
# the core that writes those rows / columns of conv_state_out.
CS_READ_SEM_ID = 1
MAX_CS_SIGNAL_TARGETS = 4  # writer cores a reader core signals, at most (the reader kernel's array bound)
QK_MCAST_FIRST_SEM_ID = 2  # R-BCAST / R-QKCORE channel semaphores follow
# R-PREP (chunked, never with R-BCAST): per inbox slot, the prepped tile rows that have landed on a state core
# (cumulative); then, on a prep core, the inbox slots its state cores have granted, then the o tiles that have landed
# (both cumulative).
LANDED_FIRST_SEM_ID = 1
# PREP_SHARE_QK, after those: on a prep core the qn and the kn tile rows landed (cumulative), on a q / k core one grant
# counter per destination prep core (the tile rows it has granted, cumulative).
MAX_SEMAPHORES = 16  # semaphore ids per core (tt_metal NUM_SEMAPHORES)
# Kernel roles: local = prep and state on the same core (R-STEP / R-VSPLIT / R-CHUNK / R-BCAST); R-PREP splits
# them over prep and state cores.
ROLE_LOCAL, ROLE_PREP, ROLE_STATE = 0, 1, 2
ROLE_QK = 3  # R-QKCORE q / k core (its qk role says which)
ROLE_PREP_QK = 4  # R-PREP q / k core (PREP_SHARE_QK; its qk role says which)
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
        "D_PAGES",
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
        "ROLE_QK",
        "ROLE_PREP_QK",
        "PREP_KQ_DEPTH",
        "EARLY_S0",
        "CS_READ_SEM_ID",
        "MAX_CS_SIGNAL_TARGETS",
    ]
    return [(f"GDN_{n}", str(globals()[n])) for n in names]


# Pages per CB; each entry is (index list sharing one allocation, pages). Vt here is the piece's column
# block (v_block_tiles); pieces > 1 adds the partial-sum combine buffers, chunked the R-CHUNK buffers.
def _cb_plan(Kt, Vt, conv_on_fpu=False, pieces=1, chunked=False, tokens=0, unit_depth=1, prep=False):
    n_state = Kt * Vt
    trb = TOKEN_TILE_ROWS_PER_BLOCK
    return [
        ((CB_STATE_IN, CB_STATE_IN_MM), n_state * min(STATE_IN_DEPTH, unit_depth)),
        ((CB_STATE, CB_STATE_MM), n_state),
        ((CB_STATE_OUT,), n_state * min(STATE_OUT_DEPTH, unit_depth)),
        ((CB_CONV_W,), (1 if conv_on_fpu else CONV_TAPS) * (2 * Kt + Vt) * unit_depth),
        ((CB_WINDOW,), CONV_TAPS * _window_group_cols(Kt, Vt, pieces, prep) * WINDOW_DEPTH),
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
        ((CB_D,), D_PAGES * trb),
        ((CB_GATE_IN,), 2 * ROW_DEPTH),
        ((CB_GATE_W,), 2 * unit_depth),
        ((CB_GATES,), 2 * trb),
        ((CB_TOK_COEF,), (CHUNK_COEF_PAGES if chunked else TOK_COEF_PAGES) * TOK_COEF_DEPTH),
        ((CB_READS,), 2 * Vt),
        ((CB_U, CB_U_F32), Vt),
        ((CB_O_ACC,), Vt * trb),
        ((CB_Z,), Vt * ROW_DEPTH),
        ((CB_NORM_W,), Vt * unit_depth),
        ((CB_OUT_ROWS,), Vt * ROW_DEPTH),
        ((CB_SCRATCH,), 1),
        # conv_state history rows, then single token rows, of one window group: one aligned span per column each
        # (reader-local staging).
        ((CB_HIST_STAGE,), 2 * _window_group_cols(Kt, Vt, pieces, prep)),
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
        ((CB_ZW,), 0),
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
    CB_ZW,
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


# Buffers written by cores of the other R-PREP role, per receiving role. Each role's plan starts with them, so they sit
# at the same offsets from the CB region base on every core of the role; a sender addresses them at its own region base
# (its plan's first buffer) plus the offsets of _remote_offsets. No core allocates another role's buffers.
STATE_REMOTE_CBS = (CB_IN_KQ, CB_IN_KNT, CB_IN_MAT, CB_IN_VEC, CB_IN_V)
PREP_REMOTE_CBS = (CB_O_IN,)
REMOTE_CBS = STATE_REMOTE_CBS + PREP_REMOTE_CBS
REMOTE_CB_NAMES = {
    name: globals()[f"CB_{name}"] for name in ("IN_KQ", "IN_KNT", "IN_MAT", "IN_VEC", "IN_V", "O_IN", "KQ_ROWS")
}


def _prep_unit_cols(Kt, Vt, heads, shared, role):
    """Window columns of an R-PREP prep core (its heads' v columns, and q / k unless PREP_SHARE_QK) or q / k core."""
    if role == ROLE_PREP_QK:
        return Kt
    return (0 if shared else 2 * Kt) + heads * Vt


def _prep_group_cols(cols, qk_core=False):
    """Columns per window hand-off of an R-PREP prep or q / k core with `cols` window columns."""
    if qk_core and PREP_QK_GROUP_COLS is not None:
        assert cols % PREP_QK_GROUP_COLS == 0, "PREP_QK_GROUP_COLS must divide Kt"
        return PREP_QK_GROUP_COLS
    return cols if cols <= PREP_WINDOW_MAX_COLS else _largest_divisor_at_most(cols, WINDOW_GROUP_MAX_COLS)


def _prep_qk_dests(ratio, heads):
    """Prep cores a q / k core writes each tile row to: one per value-head group of its key head."""
    return ratio // heads


def _prep_cb_plan(Kt, Vt, heads, conv_on_fpu, pieces, inbox, shared=False):
    """R-PREP plans as (state, prep, q / k) lists of (indices, pages): the role's remotely written buffers first (the
    inbox buffers are exactly `inbox` slots, so tile row i lands in slot i % inbox; cb_kq_rows exactly PREP_KQ_DEPTH
    rows), then its own buffers. The q / k plan (PREP_SHARE_QK, else empty) starts with cb_qk_stage, its region base."""
    Vb = Vt // pieces
    state = {idx[0]: pages for idx, pages in _cb_plan(Kt, Vb, conv_on_fpu, pieces, True)}
    prep = {idx[0]: pages for idx, pages in _cb_plan(Kt, heads * Vt, conv_on_fpu, 1, True, prep=True)}
    taps = 1 if conv_on_fpu else CONV_TAPS
    for idx in PREP_ONLY_CBS:
        state[idx] = 0
    for idx in STATE_ONLY_CBS:
        prep[idx] = 0
    for idx in (CB_OSUM_PART, CB_OSUM_GATHER, CB_D):  # no piece combine (the prep core's RMS norm sees all), no d
        state[idx] = prep[idx] = 0
    state[CB_O_ACC] = O_ACC_DEPTH * Vb
    prep[CB_CHUNK_VEC] = CHUNK_VEC_PAGES + 1  # the chunk vectors, then exp(c_last - c)^T
    prep[CB_CHUNK_MAT] = 1  # L, the operand of the inverse
    prep[CB_V_PREP] = heads * Vt
    prep[CB_PREP_SEND] = (PREP_SEND_HEADS or heads) * _head_block_pages(Kt, Vt)
    prep[CB_O_IN] = O_IN_DEPTH * heads * Vt
    prep[CB_NORM_W] = Vt
    prep[CB_RINV_O] = heads * TOKEN_TILE_ROWS_PER_BLOCK
    prep[CB_ZW] = 2 * heads * Vt  # row k's, made before row k - 1's RMS norm, which takes row k - 1's
    for idx, pages in _inbox_pages(Kt, Vb).items():
        state[idx] = inbox * pages
    state[CB_V] = 0  # the state core reads v from its inbox
    qk = {}
    if shared:
        # The prep core convolves only its v columns; qn / kn land in cb_kq_rows from the q / k cores.
        cols = _prep_unit_cols(Kt, Vt, heads, True, ROLE_PREP)
        group = _prep_group_cols(cols)
        prep.update({CB_CONV_W: taps * cols, CB_WINDOW: CONV_TAPS * group * WINDOW_DEPTH, CB_HIST_STAGE: 2 * group})
        prep.update({CB_Y_QK: 0, CB_SQ: 0, CB_RINV: 0, CB_KQ_ROWS: PREP_KQ_DEPTH * 2 * Kt})
        group = _prep_group_cols(Kt, True)
        qk = {
            CB_QK_STAGE: PREP_KQ_DEPTH * Kt,
            CB_CONV_W: taps * Kt,
            CB_WINDOW: CONV_TAPS * group * WINDOW_DEPTH,
            CB_CONV_ACC: 0 if conv_on_fpu else CONV_ACC_DEPTH,
            CB_Y_QK: Kt,
            CB_SQ: Kt,
            CB_RINV: 1,
            CB_HIST_STAGE: 2 * group,
        }
    prep_remote = PREP_REMOTE_CBS + ((CB_KQ_ROWS,) if shared else ())
    for idx in PREP_REMOTE_CBS:
        state[idx] = 0
    for idx in STATE_REMOTE_CBS:
        prep[idx] = 0
    order = [idx for idx, _ in _cb_plan(Kt, Vb, conv_on_fpu, pieces, True)]
    plans = tuple(
        [(idx, plan[idx[0]]) for idx in order if idx[0] in remote]
        + [(idx, plan[idx[0]]) for idx in order if idx[0] not in REMOTE_CBS + remote]
        for plan, remote in ((state, STATE_REMOTE_CBS), (prep, prep_remote))
    )
    qk_order = sorted(order, key=lambda idx: idx[0] != CB_QK_STAGE)
    return (*plans, [(idx, qk[idx[0]]) for idx in qk_order if qk.get(idx[0])])


def _remote_offsets(plan, remote):
    """Byte offset of each remotely written buffer from the region base of the plan's role (Float32 pages)."""
    offsets, offset = {}, 0
    for indices, pages in plan:
        if indices[0] not in remote:
            break
        offsets[indices[0]] = offset
        offset += pages * FP32_TILE_BYTES
    return offsets


def _prep_plan_bytes(plans, dtype):
    """Per-core footprint of the largest role."""
    return max(_plan_bytes(plan, dtype) for plan in plans if plan)


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
    CB_ZW,
)


def _window_group_cols(Kt, Vt, pieces, prep=False):
    """Column tiles per conv-window hand-off: a divisor of the unit's 2Kt + Vt columns, so every
    reservation stays contiguous in cb_window (compute consumes per column, so a group may straddle the
    q / k / v boundaries). R-STEP: the largest one up to WINDOW_GROUP_MAX_COLS. R-VSPLIT and an R-PREP prep core of
    at most PREP_WINDOW_MAX_COLS columns: all of them, since such a reader is bound by serial DRAM round trips."""
    cols = 2 * Kt + Vt
    if pieces > 1 or (prep and cols <= PREP_WINDOW_MAX_COLS):
        return cols
    return _largest_divisor_at_most(cols, WINDOW_GROUP_MAX_COLS)


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


def l1_cb_budget(device):
    """Bytes of static CBs per core that keep the CB region inside L1_CB_REGION_END."""
    return L1_CB_REGION_END - ttnn.device.get_allocator_base_address(device, ttnn.BufferType.L1)


def check_l1_footprint(role, footprint, budget):
    if footprint > budget:
        raise RuntimeError(
            f"gated_delta_net: {role} cores need {footprint} B of static circular buffers, above the {budget} B that "
            f"keep the CB region at or below L1 address {L1_CB_REGION_END}"
        )


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


def _prep_positions(batch, value_heads, key_heads, heads, shared, spare_cores):
    """Prep cores per R-PREP group (its stride positions): the spare cores over the groups, plus a q and a k core per
    (sequence, key head) and position under PREP_SHARE_QK."""
    ratio = value_heads // key_heads
    per_position = batch * key_heads * (ratio // heads + (2 if shared else 0))
    return min(spare_cores // per_position, MAX_PREP_CORES_PER_GROUP)


def _prep_estimate(batch, value_heads, key_heads, tokens, Vb, heads, spare_cores, shared=False):
    """Modelled R-PREP time (us): the pipeline fill (one prep row), then per tile row the state step, or the group's
    prep rate when its cores take several rows each. PREP_SHARE_QK moves the q / k part off the prep cores: their row
    is the head part alone, the fill waits for the q / k cores' row instead."""
    rows = max(_seq_tile_rows(seq, tokens) for seq in range(batch))
    per_group = _prep_positions(batch, value_heads, key_heads, heads, shared, spare_cores)
    if per_group < 1:
        return float("inf")
    core_row = heads * PREP_HEAD_ROW_US + (PREP_KQ_ROW_US if shared else PREP_QK_ROW_US)
    prep_row = core_row + (PREP_QK_FILL_US if shared else 0)
    state_step = STATE_STEP_US + Vb * STATE_STEP_COL_US
    rate = state_step if per_group >= rows else max(state_step, core_row / per_group)
    return prep_row + rows * rate


def _prep_heads(batch, value_heads, key_heads, tokens, Vb, spare_cores):
    """(value heads per R-PREP group, PREP_SHARE_QK): PREP_HEADS / PREP_SHARE_QK, or the modelled fastest."""
    ratio = value_heads // key_heads
    if PREP_HEADS is not None:
        assert ratio % PREP_HEADS == 0, "PREP_HEADS must divide the value heads per key head"
    options = [
        (h, shared)
        for h in ((PREP_HEADS,) if PREP_HEADS is not None else range(1, ratio + 1))
        if ratio % h == 0
        for shared in ((False, True) if PREP_SHARE_QK == "auto" else (bool(PREP_SHARE_QK),))
    ]
    return min(
        options,
        key=lambda opt: (
            _prep_estimate(batch, value_heads, key_heads, tokens, Vb, opt[0], spare_cores, opt[1]),
            opt[1],
            -opt[0],
        ),
    )


def _prep_layout(batch, value_heads, key_heads, tokens, Kt, Vt, pieces, chunked, dtype, spare_cores, budget):
    """R-PREP plan, or None to keep prep and state on the same core: (prep cores per group cap, inbox depth, value
    heads per group). A group is (sequence, key head, subset of `heads` of its value heads). Needs one unit per core
    (R-VSPLIT), at least one idle core per group, the key head's value heads in one gate tile column (one gate tile
    per prepped tile row) and the merged plan inside L1."""
    ratio = value_heads // key_heads
    if not (PREP_OFFLOAD and chunked and pieces > 1):
        return None
    if any((hk * ratio) // TILE != (hk * ratio + ratio - 1) // TILE for hk in range(key_heads)):
        return None
    if min(_seq_tile_rows(seq, tokens) for seq in range(batch)) < PREP_MIN_TILE_ROWS:
        return None
    heads, shared = _prep_heads(batch, value_heads, key_heads, tokens, Vt // pieces, spare_cores)
    per_group = _prep_positions(batch, value_heads, key_heads, heads, shared, spare_cores)
    if per_group < 1:
        return None
    # A prep core's next tile row is granted only after its current one has landed when its stride is at least
    # the inbox depth (the cumulative credit count relies on it).
    inbox = min([INBOX_DEPTH, per_group] + [_seq_tile_rows(seq, tokens) for seq in range(batch)])
    if shared:  # the q / k landing and grant counters follow the inbox's
        inbox = min(inbox, MAX_SEMAPHORES - LANDED_FIRST_SEM_ID - 4 - _prep_qk_dests(ratio, heads))
    # The deepest inbox that fits L1 (a shallower one only costs slack between a grant and its landing).
    while inbox >= 1:
        plans = _prep_cb_plan(Kt, Vt, heads, _conv_on_fpu(dtype), pieces, inbox, shared)
        if _prep_plan_bytes(plans, dtype) <= budget:
            return per_group, inbox, heads, shared
        inbox -= 1
    return None


def _qk_core_units(batch, tokens, value_heads, key_heads, Kt, Vt, pieces, num_cores):
    """R-QKCORE plan: units (sequences of one value head) per pair core, or None. Needs decode with every sequence in
    one tile row, one pair per unit (R-STEP), 2 Hk idle cores for the q / k cores, and a v-only window group whose FPU
    conv batch also divides the pair's columns (compute's CONV_BATCH)."""
    if not (SHARE_QK_CORES and tokens == 1 and pieces == 1 and batch <= TILE):
        return None
    batch_cols = _largest_divisor_at_most(Vt, 4)
    if (2 * Kt + Vt) % batch_cols:
        return None
    ratio = value_heads // key_heads
    pair_cores = num_cores - 2 * key_heads
    for per_core in range(1, batch + 1):
        if key_heads * ratio * -(-batch // per_core) <= pair_cores:
            return per_core
    return None


def _cs_read_signals(device, assignment, qk_cores, pieces, value_heads, ratio, qk_role):
    """In-place conv_state: every core that reads conv_state rows / columns which another core writes to
    conv_state_out signals that core once its reads have landed. Returns (x, y) -> (index of the core's unit after
    whose first tile row it signals, NoC x, y of each target) and (x, y) -> signals the core waits for before writing.
    R-QKCORE: a q / k core writes whole conv_state_out tiles of its key head's columns, read by the key head's pair
    cores. Otherwise: one unit per (sequence, key head) writes the q / k history rows, which every unit of the group
    reading q / k columns has read."""
    signals, waits = {}, {}

    def noc_xy(core):
        noc_core = device.worker_core_from_logical_core(core)
        return [noc_core.x, noc_core.y]

    if qk_cores:
        writers = {(hk, role): core for core, hk, role in qk_cores}
        for core, first_unit, count, _ in assignment:
            hk = (first_unit % value_heads) // ratio
            targets = [writers[hk, QK_Q_INJECTOR], writers[hk, QK_K_INJECTOR]]
            signals[core.x, core.y] = (count - 1, [xy for c in targets for xy in noc_xy(c)])
            for target in targets:
                waits[target.x, target.y] = waits.get((target.x, target.y), 0) + 1
        return signals, waits
    core_of = {}
    for core, first_unit, count, stride in assignment:
        for idx in range(count):
            core_of[first_unit + idx * stride] = (core, idx)
    targets_of = {}  # (x, y) -> (last signalling unit index, writer cores)
    for unit, (core, idx) in core_of.items():
        if qk_role(unit) == QK_RECEIVER:
            continue
        pair = unit // pieces
        writer_unit = (pair - pair % value_heads + (pair % value_heads) // ratio * ratio) * pieces
        writer = core_of[writer_unit][0]
        if (writer.x, writer.y) == (core.x, core.y):
            continue
        last, cores = targets_of.get((core.x, core.y), (0, {}))
        cores[writer.x, writer.y] = writer
        targets_of[core.x, core.y] = (max(last, idx), cores)
    for key, (last, cores) in targets_of.items():
        assert len(cores) <= MAX_CS_SIGNAL_TARGETS
        signals[key] = (last, [xy for c in cores.values() for xy in noc_xy(c)])
        for target in cores:
            waits[target] = waits.get(target, 0) + 1
    return signals, waits


def _core_ranges(cores):
    """One range per core, in the given order (the multicast groups enumerate cores in this order)."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _merged_ranges(cores):
    """The same core set as rectangles (row runs stacked over consecutive rows): fewer ranges to dispatch."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, set()).add(c.x)
    rects = []  # [x_lo, x_hi, y_lo, y_hi]
    for y in sorted(rows):
        xs = sorted(rows[y])
        runs, start = [], xs[0]
        for prev, x in zip(xs, xs[1:] + [None]):
            if x != prev + 1:
                runs.append((start, prev))
                start = x
        for lo, hi in runs:
            rect = next((r for r in rects if r[0] == lo and r[1] == hi and r[3] == y - 1), None)
            if rect:
                rect[3] = y
            else:
                rects.append([lo, hi, y, y])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(lo, y_lo), ttnn.CoreCoord(hi, y_hi)) for lo, hi, y_lo, y_hi in rects]
    )


def _copy_compute_config(config):
    copy = ttnn.ComputeConfigDescriptor(
        math_fidelity=config.math_fidelity,
        math_approx_mode=config.math_approx_mode,
        fp32_dest_acc_en=config.fp32_dest_acc_en,
        dst_full_sync_en=config.dst_full_sync_en,
        bfp8_pack_precise=config.bfp8_pack_precise,
    )
    copy.enable_trisc2_rvv = config.enable_trisc2_rvv
    copy.unpack_to_dest_mode = list(config.unpack_to_dest_mode)
    return copy


def compute_config_key(config):
    """Every field of a compute config that reaches the program (None: the op's default)."""
    if config is None:
        return None
    return (
        config.math_fidelity,
        config.math_approx_mode,
        config.fp32_dest_acc_en,
        config.dst_full_sync_en,
        config.bfp8_pack_precise,
        config.enable_trisc2_rvv,
        tuple(config.unpack_to_dest_mode),
    )


# Buffer addresses are the reader's and writer's common runtime args, in this order; every other argument is a
# function of the program cache key.
def reader_addresses(inputs, outputs):
    return [t.buffer_address() for t in inputs] + [outputs[1].buffer_address()]


def writer_addresses(outputs):
    return [outputs[0].buffer_address(), outputs[2].buffer_address()]


def _mcast_rects(device, cores):
    """Multicast rectangles covering exactly `cores` (logical, row-major order): one per run of cores consecutive in
    one row of virtual NoC coordinates (non-worker columns split a row), as [count, then (x_lo, y_lo, x_hi, y_hi,
    cores) per run]; the kernel orders the corners for its NoC."""
    runs = []
    for core in cores:
        virt = device.worker_core_from_logical_core(core)
        if PREP_MCAST and runs and runs[-1][-1].y == virt.y and runs[-1][-1].x + 1 == virt.x:
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
    budget = l1_cb_budget(device)
    if V_BLOCK_TILES is None and pieces > 1:
        # R-PREP: the modelled fastest of the piece counts it can run (else R-VSPLIT keeps its pieces).
        best = None
        for option in range(2, min(pieces, PREP_MAX_PIECES) + 1):
            spare_cores = num_grid_cores - num_pairs * option
            plan = Vt % option == 0 and _prep_layout(
                batch, value_heads, key_heads, tokens, Kt, Vt, option, chunked, mixed_qkv.dtype, spare_cores, budget
            )
            if plan:
                cost = _prep_estimate(
                    batch, value_heads, key_heads, tokens, Vt // option, plan[2], spare_cores, plan[3]
                )
                if best is None or cost < best[0]:
                    best = (cost, option)
        if best is not None:
            pieces = best[1]
    v_block_tiles = Vt // pieces
    num_units = num_pairs * pieces
    units_per_block = PAIRS_PER_BLOCK if pieces == 1 else 1  # R-VSPLIT: one unit per core
    num_blocks = (num_units + units_per_block - 1) // units_per_block
    work_grid = grid
    if BCAST_RECT_GROUPS and _share_qk(pieces, tokens):
        # One unit per core, in row-major order over a grid whose width divides the multicast group: each key head's
        # group is then one rectangle of whole rows (one multicast per channel instead of one per grid row it spans).
        group = (value_heads // key_heads) * pieces
        widths = [w for w in range(1, grid.x + 1) if group % w == 0 and -(-num_units // w) <= grid.y]
        if widths:
            work_grid = ttnn.CoreCoord(max(widths), grid.y)
    _, all_cores, group_1, group_2, blocks_1, blocks_2 = ttnn.split_work_to_cores(work_grid, num_blocks, row_wise=True)

    assignment = []  # (core, first_unit, unit_count)
    next_block = 0
    for group, per_core in ((group_1, blocks_1), (group_2, blocks_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            first_unit = next_block * units_per_block
            last_unit = min(num_units, (next_block + per_core) * units_per_block)
            assignment.append((core, first_unit, last_unit - first_unit, 1))
            next_block += per_core
    assert pieces == 1 or all(count == 1 for _, _, count, _ in assignment)
    ratio = value_heads // key_heads
    token_dtype = mixed_qkv.dtype
    conv_on_fpu = _conv_on_fpu(token_dtype)
    grid_cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    # R-QKCORE: pair cores grouped by key head (row-major, cores_per_kh each: value head j of the key head, sequences
    # s0, s0 + 1, ... at unit stride Hv), then the q and k core of each key head.
    qk_per_core = _qk_core_units(batch, tokens, value_heads, key_heads, Kt, Vt, pieces, num_grid_cores)
    qk_cores = []  # (core, key head, qk role)
    if qk_per_core:
        assignment = []
        for hk in range(key_heads):
            for j in range(ratio):
                for s0 in range(0, batch, qk_per_core):
                    first = s0 * value_heads + hk * ratio + j
                    assignment.append((grid_cores[len(assignment)], first, min(qk_per_core, batch - s0), value_heads))
        cores_per_kh = len(assignment) // key_heads
        for hk in range(key_heads):
            for role in (QK_Q_INJECTOR, QK_K_INJECTOR):
                qk_cores.append((grid_cores[len(assignment) + len(qk_cores)], hk, role))
        all_cores = _merged_ranges([core for core, _, _, _ in assignment] + [core for core, _, _ in qk_cores])
    state_cores = [core for core, _, _, _ in assignment]
    state_set = {(c.x, c.y) for c in state_cores}
    spare = [c for c in grid_cores if (c.x, c.y) not in state_set]
    prep = _prep_layout(batch, value_heads, key_heads, tokens, Kt, Vt, pieces, chunked, token_dtype, len(spare), budget)
    heads = ratio  # R-PREP: value heads per prep group
    shared = False  # R-PREP: PREP_SHARE_QK (q / k cores)
    if prep is None:
        inbox = 0
        # The deepest per-unit input depth that fits L1 (1 when every core runs one unit).
        unit_depth = min(UNIT_INPUT_DEPTH, max(count for _, _, count, _ in assignment))

        def unit_lag(depth):
            # The reader must run a unit ahead of the lagged norm: two units of inputs in flight.
            return bool(qk_cores) and SHARE_QK_FINISH and UNIT_LAG and depth > 1

        def local_plan(depth):
            plan = _cb_plan(Kt, v_block_tiles, conv_on_fpu, pieces, chunked, tokens, depth)
            if not qk_cores:
                return plan
            # UNIT_LAG holds a unit's o (and o^2) until after the next unit's token step.
            lag_units = 2 if unit_lag(depth) else 1
            # One plan on pair and q / k cores (the multicast lands at one cb_kq_rows address everywhere): window
            # groups of Vt (pair) or Kt (q / k core) columns, a q / k core's staged current-row and conv_state tiles,
            # its qn / kn stage.
            cs_tiles = -(-HISTORY * batch // TILE)
            sizes = {
                # Conv weights of a pair core's v columns per unit, or of the q / k core's Kt columns.
                CB_CONV_W: (1 if conv_on_fpu else CONV_TAPS) * max(Kt, v_block_tiles * depth),
                CB_WINDOW: CONV_TAPS * max(Kt, v_block_tiles) * WINDOW_DEPTH,
                CB_HIST_STAGE: max(2 * v_block_tiles, Kt * (1 + cs_tiles), 1 + 2 * cs_tiles),
                CB_QK_STAGE: Kt,
                CB_O_ACC: v_block_tiles * lag_units,
                CB_SQ_O: v_block_tiles * lag_units,
            }
            return [(idx, sizes.get(idx[0], pages)) for idx, pages in plan]

        cb_plan = local_plan(unit_depth)
        while unit_depth > 1 and _plan_bytes(cb_plan, token_dtype) > budget:
            unit_depth -= 1
            cb_plan = local_plan(unit_depth)
        check_l1_footprint("pair and q / k" if qk_cores else "pair", _plan_bytes(cb_plan, token_dtype), budget)
    else:
        per_group, inbox, heads, shared = prep
        prep_plans = _prep_cb_plan(Kt, Vt, heads, conv_on_fpu, pieces, inbox, shared)
        for role, plan in zip(("R-PREP state", "R-PREP prep", "R-PREP q / k"), prep_plans):
            if plan:
                check_l1_footprint(role, _plan_bytes(plan, token_dtype), budget)
        groups_per_kh = ratio // heads
        qk_extra = 2 if shared else 0  # q / k cores per (sequence, key head) and stride position
        # Group first units: heads value heads (each `pieces` units) from value head h0.
        groups = [(seq * value_heads + h0) * pieces for seq in range(batch) for h0 in range(0, value_heads, heads)]
        strides = [min(per_group, _seq_tile_rows(first // pieces // value_heads, tokens)) for first in groups]
        cluster_picks = None
        if PREP_CLUSTER:
            # Each group's state and prep cores as one blob around the first free core (row-major), its state cores at
            # the blob's center: a head block crosses a few links instead of every group's converging on the top rows.
            # PREP_SHARE_QK: the middle group of each key head's blob also takes the key head's q / k cores (after its
            # own prep cores), between the key head's other groups.
            free = list(grid_cores)
            assignment, cluster_picks, qk_pools = [], [], {}
            for g, (first, stride) in enumerate(zip(groups, strides)):
                takes_qk = shared and g % groups_per_kh == groups_per_kh // 2
                seed = free[0]
                blob = sorted(free, key=lambda c: (abs(c.x - seed.x) + abs(c.y - seed.y), c.y, c.x))
                blob = blob[: heads * pieces + stride * (1 + (qk_extra if takes_qk else 0))]
                cx, cy = sum(c.x for c in blob) / len(blob), sum(c.y for c in blob) / len(blob)
                blob.sort(key=lambda c: (abs(c.x - cx) + abs(c.y - cy), c.y, c.x))
                assignment += [(core, first + u, 1, 1) for u, core in enumerate(blob[: heads * pieces])]
                cluster_picks.append(blob[heads * pieces : heads * pieces + stride])
                if takes_qk:
                    qk_pools[g // groups_per_kh] = blob[heads * pieces + stride :]
                taken = {(c.x, c.y) for c in blob}
                free = [c for c in free if (c.x, c.y) not in taken]
            state_cores = [core for core, _, _, _ in assignment]
    # NoC coordinates of the core running each unit (R-VSPLIT peers).
    unit_noc = {}
    unit_cores = []  # logical core of each unit when one unit per core (R-VSPLIT)
    for core, first_unit, _, _ in assignment:
        noc_core = device.worker_core_from_logical_core(core)
        unit_noc[first_unit] = (noc_core.x, noc_core.y)
        unit_cores.append(core)

    # R-BCAST: units of one (sequence, key head) are r * pieces consecutive units, so on one unit per core they
    # are consecutive cores in row-major order: one multicast group each, with the q injector its first core
    # and the k injector its second (two channels, one sender each).
    qk_group = (value_heads // key_heads) * pieces
    qk_channels = []
    if _share_qk(pieces, tokens):
        order = [(core.y, core.x) for core, _, _, _ in assignment]
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

    if qk_cores:
        # Group kh = the key head's cores_per_kh pair cores (consecutive in row-major order); its senders are the
        # key head's q core (channel q) and k core (channel k), outside the group. One flag-signalled round per call.
        config = ttnn.McastConfig(noc=ttnn.NOC.RISCV_1_default, handshake=False, data_ready=ttnn.McastDataReady.Flag)
        qk_channels = [
            ttnn.Mcast(
                device,
                config,
                _core_ranges(state_cores),
                cores_per_kh,
                ttnn.McastExplicitFixedSenderConfig([core for core, _, r in qk_cores if r == role]),
                ttnn.McastCoreOrder.RowMajor,
            )
            for role in (QK_Q_INJECTOR, QK_K_INJECTOR)
        ]

    def qk_role(unit):
        if qk_cores:
            return QK_RECEIVER
        if not qk_channels:
            return QK_OWN
        return {0: QK_Q_INJECTOR, 1: QK_K_INJECTOR}.get(unit % qk_group, QK_RECEIVER)

    token_tile_bytes = _tile_bytes(token_dtype)

    def window_zeroing(unit_cols, group_cols, tile_rows):
        """(slots, bytes per slot) of cb_window the writer zeroes: only the window groups this core will use."""
        groups = tile_rows * (unit_cols // group_cols)
        return min(WINDOW_DEPTH, groups), CONV_TAPS * group_cols * token_tile_bytes

    def unit_zeroing(first_unit, count, stride):
        groups, group_cols = 0, 0
        for unit in range(first_unit, first_unit + count * stride, stride):
            tile_rows = _seq_tile_rows(unit // pieces // value_heads, tokens)
            unit_cols = v_block_tiles + {QK_OWN: 2 * Kt, QK_RECEIVER: 0}.get(qk_role(unit), Kt)
            group_cols = unit_cols if pieces > 1 or qk_cores else _window_group_cols(Kt, v_block_tiles, pieces)
            groups += tile_rows * (unit_cols // group_cols)
        return window_zeroing(group_cols, group_cols, groups)

    # ---- R-PREP: prep cores per (sequence, key head, head subset) group, round robin over the group's tile rows ----
    prep_units = []  # (core, group first unit, stride position, stride)
    group_producers = {}  # group first unit -> NoC coordinates of its prep cores in stride order
    prep_qk_units = (
        []
    )  # PREP_SHARE_QK: (core, key head's first unit, stride position, stride, qk role, dest prep cores)
    prep_qk_src = {}  # PREP_SHARE_QK: prep core (x, y) -> [q core NoC x, y, k core NoC x, y, its index at those]
    if prep is not None:
        picks = cluster_picks
        if picks is None:
            # The groups take idle cores in turn, each the one nearest its own state cores (short multicast paths).
            centers = []
            for first in groups:
                members = unit_cores[first : first + heads * pieces]
                centers.append((sum(c.x for c in members) / len(members), sum(c.y for c in members) / len(members)))
            free = list(spare)
            picks = [[] for _ in groups]
            for q in range(max(strides)):
                for g, (cx, cy) in enumerate(centers):
                    if q < strides[g]:
                        core = min(free, key=lambda c: abs(c.x - cx) + abs(c.y - cy))
                        free.remove(core)
                        picks[g].append(core)
        for g, group_first in enumerate(groups):
            for q, core in enumerate(picks[g]):
                prep_units.append((core, group_first, q, strides[g]))
                noc_core = device.worker_core_from_logical_core(core)
                group_producers.setdefault(group_first, []).extend([noc_core.x, noc_core.y])
        if shared:
            # Per (sequence, key head) and stride position, the q and k cores nearest the position's prep cores (one
            # per value-head group of the key head, which share the key head's stride).
            taken = {(c.x, c.y) for c in state_cores} | {(c.x, c.y) for c, _, _, _ in prep_units}
            free = [c for c in grid_cores if (c.x, c.y) not in taken]
            for kh_idx in range(len(groups) // groups_per_kh):
                kh_groups = range(kh_idx * groups_per_kh, (kh_idx + 1) * groups_per_kh)
                pool = qk_pools[kh_idx] if cluster_picks is not None else free
                for q in range(strides[kh_idx * groups_per_kh]):
                    dests = [picks[g][q] for g in kh_groups]
                    cx, cy = sum(c.x for c in dests) / len(dests), sum(c.y for c in dests) / len(dests)
                    srcs = []
                    for role in (QK_Q_INJECTOR, QK_K_INJECTOR):
                        core = min(pool, key=lambda c: (abs(c.x - cx) + abs(c.y - cy), c.y, c.x))
                        pool.remove(core)
                        prep_qk_units.append(
                            (core, groups[kh_idx * groups_per_kh], q, strides[kh_idx * groups_per_kh], role, dests)
                        )
                        noc_core = device.worker_core_from_logical_core(core)
                        srcs += [noc_core.x, noc_core.y]
                    for d, dest in enumerate(dests):
                        prep_qk_src[dest.x, dest.y] = srcs + [d]
    prep_cores = [core for core, _, _, _ in prep_units]
    prep_qk_cores = [unit[0] for unit in prep_qk_units]
    all_cores = _merged_ranges(state_cores + prep_cores + prep_qk_cores) if prep_units else all_cores

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
        # Allocation follows this order: each role's remotely written buffers start at the CB region base of its cores
        # (state and prep cores are disjoint), and the buffers on every core come after both roles'.
        add_cbs(prep_plans[0], _merged_ranges(state_cores))
        add_cbs(prep_plans[1], _merged_ranges(prep_cores))
        if prep_qk_cores:
            add_cbs(prep_plans[2], _merged_ranges(prep_qk_cores))
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
    reader_tensors = (*inputs, conv_state_out)
    # Token tile bytes, state tile bytes.
    geometry_ct = [token_tile_bytes, FP32_TILE_BYTES]
    reader_accessor_ct = [arg for t in reader_tensors for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    writer_accessor_ct = [
        arg for t in (output, recurrent_state_out) for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()
    ]

    def role_ct(role):
        # A q / k core's unit is nominally a prep core's (compute sizes nothing by it; its reader takes no v columns).
        unit_v = heads * Vt if role in (ROLE_PREP, ROLE_PREP_QK) else v_block_tiles
        role_pieces = 1 if role == ROLE_PREP else pieces
        group_cols = _window_group_cols(Kt, unit_v, role_pieces, role == ROLE_PREP)
        if role in (ROLE_PREP, ROLE_PREP_QK) and (shared or role == ROLE_PREP_QK):
            group_cols = _prep_group_cols(_prep_unit_cols(Kt, Vt, heads, shared, role), role == ROLE_PREP_QK)
        if qk_cores:
            group_cols = 1 if role == ROLE_QK else v_block_tiles  # q / k core: per column; pair core: per unit
        reader_ct = [Kt, Vt, value_heads, key_heads, Ct, Gt, *geometry_ct, ttnn.get_dram_alignment()]
        reader_ct += [ROW_STAGE_SLOTS if row_stage_bytes else 0, int(conv_on_fpu), group_cols, WINDOW_DEPTH]
        s0_tiles = Kt * v_block_tiles
        reader_ct += [pieces, int(chunked), role, inbox, min(S0_READ_TILES or s0_tiles, s0_tiles), int(bool(qk_cores))]
        reader_ct += [int(tokens == 1)]  # decode: window rows issued ahead of the other inputs
        reader_ct += [heads]  # R-PREP: value heads per prep group
        reader_ct += reader_accessor_ct
        # RATIO: a q / k core's destination prep cores.
        role_ratio = _prep_qk_dests(ratio, heads) if role == ROLE_PREP_QK else heads
        writer_ct = [Kt, Vt, value_heads, *geometry_ct, pieces, OSUM_SEM_ID, int(chunked), role, inbox, role_ratio]
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
            # R-BCAST receivers / injectors; PREP_SHARE_QK prep cores (qn / kn land from the q / k cores).
            int((bool(qk_channels) and role != ROLE_QK) or (shared and role == ROLE_PREP)),
            role,
            heads,
        ]
        return reader_ct, writer_ct, compute_ct

    v_eps_bits = _float_bits(value_dim * epsilon)
    # In-place conv_state: its reads must land before any core overwrites the same rows / columns. R-PREP needs no
    # signal: a state core copies conv_state_out after every tile row of its group has landed from the prep cores,
    # which read the history rows for the first one.
    cs_in_place = prep is None and conv_state_out.buffer_address() == conv_state.buffer_address()
    cs_signals, cs_waits = {}, {}
    if cs_in_place:
        cs_signals, cs_waits = _cs_read_signals(device, assignment, qk_cores, pieces, value_heads, ratio, qk_role)

    def cs_args(core):
        if not cs_in_place:
            return []
        last, targets = cs_signals.get((core.x, core.y), (0, []))
        return [last, len(targets) // 2, *targets, cs_waits.get((core.x, core.y), 0)]

    roles = {}  # role -> (cores, reader rt, writer rt, compute rt), rt: (x, y) -> args
    state_role = ROLE_LOCAL if prep is None else ROLE_STATE
    rts = ({}, {}, {})  # (x, y) -> args, per kernel
    for core, first_unit, count, stride in assignment:
        role = qk_role(first_unit)  # R-BCAST runs one unit per core
        zero_slots, zero_slot_bytes = unit_zeroing(first_unit, count, stride) if prep is None else (0, 0)
        rows = [0, 1, stride]  # tile rows: every one of the unit's sequence; then the unit stride
        producers = []
        if prep is not None:
            group_first = first_unit - first_unit % (heads * pieces)
            producers = [len(group_producers[group_first]) // 2] + group_producers[group_first]
        rts[0][core.x, core.y] = [first_unit, count, tokens, role, zero_slots] + rows + producers + cs_args(core)
        # R-VSPLIT: NoC coordinates of the pair's piece cores, in piece order.
        group_first = first_unit - first_unit % pieces
        peers = [c for s in range(pieces) for c in unit_noc[group_first + s]] if pieces > 1 else []
        rts[1][core.x, core.y] = (
            [first_unit, count, tokens, role, zero_slots, zero_slot_bytes] + rows + peers + producers
        )
        rts[2][core.x, core.y] = [first_unit, count, tokens, v_eps_bits, role] + rows
    roles[state_role] = (state_cores, *rts)
    prep_send_noc0 = {}  # R-PREP prep / q / k core (x, y) -> its writer sends on NOC 0

    def send_noc0(core, dsts):
        """NOC 0 has the shorter torus paths from `core` to the NoC (x, y) list `dsts` (NOC 0 routes toward higher x
        and y, NOC 1 toward lower)."""
        src = device.worker_core_from_logical_core(core)
        hops = [
            sum((dx - src.x) % NOC_GRID[0] + (dy - src.y) % NOC_GRID[1] for dx, dy in dsts),
            sum((src.x - dx) % NOC_GRID[0] + (src.y - dy) % NOC_GRID[1] for dx, dy in dsts),
        ]
        return hops[0] < hops[1]

    if prep_units:
        rts = ({}, {}, {})  # (x, y) -> args, per kernel
        prep_cols = _prep_unit_cols(Kt, Vt, heads, shared, ROLE_PREP)
        prep_qk_role = QK_RECEIVER if shared else QK_OWN  # PREP_SHARE_QK: qn / kn land, no q / k columns
        for core, group_first, q, stride in prep_units:
            seq = group_first // pieces // value_heads
            my_rows = len(range(q, _seq_tile_rows(seq, tokens), stride))
            group_cols = _prep_group_cols(prep_cols) if shared else _window_group_cols(Kt, heads * Vt, 1, True)
            zero_slots, zero_slot_bytes = window_zeroing(prep_cols, group_cols, my_rows)
            rows = [q, stride, 1]
            consumers = [c for u in range(heads * pieces) for c in unit_noc[group_first + u]]
            # A head block goes to the head's pieces: one multicast per row run.
            for j in range(heads):
                first = group_first + j * pieces
                consumers += _mcast_rects(device, unit_cores[first : first + pieces])
            qk_src = prep_qk_src.get((core.x, core.y), [])
            rts[0][core.x, core.y] = [group_first, 1, tokens, prep_qk_role, zero_slots] + rows + qk_src + cs_args(core)
            rts[1][core.x, core.y] = [group_first, 1, tokens, prep_qk_role, zero_slots, zero_slot_bytes] + rows
            rts[1][core.x, core.y] += consumers
            rts[2][core.x, core.y] = [group_first, 1, tokens, v_eps_bits, prep_qk_role] + rows
            dsts = [unit_noc[group_first + u] for u in range(heads * pieces)]
            prep_send_noc0[core.x, core.y] = send_noc0(core, dsts)
        roles[ROLE_PREP] = (prep_cores, *rts)
    if prep_qk_units:
        # A q / k core's unit is its key head's first group; it writes its rows' qn (kn) to the position's prep cores.
        rts = ({}, {}, {})
        for core, kh_first, q, stride, role, dests in prep_qk_units:
            seq = kh_first // pieces // value_heads
            my_rows = len(range(q, _seq_tile_rows(seq, tokens), stride))
            zero_slots, zero_slot_bytes = window_zeroing(Kt, _prep_group_cols(Kt, True), my_rows)
            rows = [q, stride, 1]
            dst_noc = [device.worker_core_from_logical_core(d) for d in dests]
            rts[0][core.x, core.y] = [kh_first, 1, tokens, role, zero_slots] + rows
            rts[1][core.x, core.y] = [kh_first, 1, tokens, role, zero_slots, zero_slot_bytes] + rows
            rts[1][core.x, core.y] += [c for d in dst_noc for c in (d.x, d.y)]
            rts[2][core.x, core.y] = [kh_first, 1, tokens, v_eps_bits, role] + rows
            prep_send_noc0[core.x, core.y] = send_noc0(core, [(d.x, d.y) for d in dst_noc])
        roles[ROLE_PREP_QK] = (prep_qk_cores, *rts)
    if qk_cores:
        # A q / k core's "unit" is its key head; its count the sequences (token rows 0..B-1 of the decode tile row).
        rts = ({}, {}, {})  # (x, y) -> args, per kernel
        zero_slots, zero_slot_bytes = 0, 0  # the writer zeroes each column's slot itself (with its history rows)
        # conv_state_out, whole tiles of the key head's columns (the ones only it and its pair cores read): a q / k core
        # writes its own Kt columns, then half of the key head's ratio * Vt v columns.
        v_half = -(-ratio * Vt // 2)
        for core, hk, role in qk_cores:
            rows = [0, 1, 1]
            own_lo = (hk if role == QK_Q_INJECTOR else key_heads + hk) * Kt
            v_lo = 2 * key_heads * Kt + hk * ratio * Vt + (0 if role == QK_Q_INJECTOR else v_half)
            v_hi = 2 * key_heads * Kt + hk * ratio * Vt + (v_half if role == QK_Q_INJECTOR else ratio * Vt)
            cs_cols = [own_lo, own_lo + Kt, v_lo, v_hi, cs_waits.get((core.x, core.y), 0)]
            rts[0][core.x, core.y] = [hk, batch, tokens, role, zero_slots] + rows + cs_cols
            rts[1][core.x, core.y] = [hk, batch, tokens, role, zero_slots, zero_slot_bytes] + rows
            rts[2][core.x, core.y] = [hk, batch, tokens, v_eps_bits, role] + rows
        roles[ROLE_QK] = ([core for core, _, _ in qk_cores], *rts)

    compute_kernel_config = _copy_compute_config(compute_kernel_config)  # never mutate the caller's
    modes = list(getattr(compute_kernel_config, "unpack_to_dest_mode", []) or [])
    if len(modes) < NUM_CB_SLOTS:
        modes = modes + [ttnn.UnpackToDestMode.Default] * (NUM_CB_SLOTS - len(modes))
    for idx in U2D_CBS:
        fp32_cb = idx not in TOKEN_CBS or token_dtype == ttnn.float32
        modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32 if fp32_cb else ttnn.UnpackToDestMode.Default
    compute_kernel_config.unpack_to_dest_mode = modes
    if DST_FULL_SYNC:
        compute_kernel_config.dst_full_sync_en = True

    defines = _kernel_defines() + [
        ("GDN_HOLD_QK", str(int(bool(qk_cores) and SHARE_QK_FINISH))),
        ("GDN_UNIT_LAG", str(int(prep is None and unit_lag(unit_depth)))),
        ("GDN_STATE_STAGGER", str(STATE_STAGGER if qk_cores else 0)),
        ("GDN_CS_IN_PLACE", str(int(cs_in_place))),
    ]
    # R-PREP: offsets of the other role's remotely written buffers from its CB region base (see STATE_REMOTE_CBS).
    remote = {}
    if prep is not None:
        prep_remote = PREP_REMOTE_CBS + ((CB_KQ_ROWS,) if shared else ())
        remote = _remote_offsets(prep_plans[0], STATE_REMOTE_CBS) | _remote_offsets(prep_plans[1], prep_remote)
    defines += [(f"GDN_OFFSET_{name}", str(remote.get(idx, 0))) for name, idx in REMOTE_CB_NAMES.items()]
    kernels = []
    writers = []

    def runtime_args(args, cores):
        rt = ttnn.RuntimeArgs()
        for core in cores:
            rt[core.x][core.y] = args[core.x, core.y]
        return rt

    kernel_groups = []  # (role, cores, swap the reader / writer NoCs)
    for role, (cores, *_) in roles.items():
        if role == ROLE_LOCAL and qk_cores and SWAP_NOC_ROWS:
            # Decode R-QKCORE: every pair core reads its S0 at once. NOC 0 brings read responses south, around the
            # torus for the top rows, which then lose to the lower rows' traffic; NOC 1 brings them north. The pair
            # cores in the top SWAP_NOC_ROWS rows read on NOC 1 and write on NOC 0.
            top = [c for c in cores if c.y < SWAP_NOC_ROWS]
            kernel_groups += [(role, top, True), (role, [c for c in cores if c.y >= SWAP_NOC_ROWS], False)]
        elif role in (ROLE_PREP, ROLE_PREP_QK) and PREP_SEND_NOC == "auto":
            # Each prep core writes on the NoC with the shorter paths to its state cores (NOC 0 routes toward higher x
            # and y, NOC 1 toward lower, both around the torus); the reader takes the other one.
            on_noc0 = [c for c in cores if prep_send_noc0[c.x, c.y]]
            kernel_groups += [(role, on_noc0, True), (role, [c for c in cores if not prep_send_noc0[c.x, c.y]], False)]
        else:
            kernel_groups.append((role, cores, False))
    for role, cores, swap_nocs in kernel_groups:
        if not cores:
            continue
        _, reader_rt, writer_rt, compute_rt = roles[role]
        ranges = _merged_ranges(cores)
        reader_rt, writer_rt, compute_rt = (runtime_args(rt, cores) for rt in (reader_rt, writer_rt, compute_rt))
        reader_ct, writer_ct, compute_ct = role_ct(role)
        reader_config, writer_config = ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor()
        if swap_nocs:
            writer_config = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
            )
            reader_config = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
            )
        if role in (ROLE_PREP, ROLE_PREP_QK) and PREP_SEND_NOC != 0 and not swap_nocs:
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
            common_runtime_args=writer_addresses(outputs),
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
                common_runtime_args=reader_addresses(inputs, outputs),
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
        # PREP_SHARE_QK: then the qn / kn landed counters (prep cores) and the grant counters (q / k cores).
        qk_sems = 2 + _prep_qk_dests(ratio, heads) if shared else 0
        for sem in range(LANDED_FIRST_SEM_ID, LANDED_FIRST_SEM_ID + inbox + 2 + qk_sems):
            semaphores.append(ttnn.SemaphoreDescriptor(id=sem, core_ranges=all_cores, initial_value=0))
    if cs_in_place:
        semaphores.append(ttnn.SemaphoreDescriptor(id=CS_READ_SEM_ID, core_ranges=all_cores, initial_value=0))
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    # The writer carries the q and k multicast channels (absent outside R-BCAST, so one kernel source).
    next_sem = QK_MCAST_FIRST_SEM_ID
    for prefix, channel in zip(("qkq", "qkk"), qk_channels or (None, None)):
        if channel is None:
            for writer in writers:
                ttnn.attach_absent_mcast(writer, prefix)
        else:
            channel.attach(descriptor, prefix, writers, next_sem)
            next_sem = channel.next_semaphore_id()
    descriptor.kernels = kernels
    return descriptor


# ---- Program cache (host cost per call) ----
# A descriptor depends on its inputs only through the cache key (gated_delta_net.py: the device, every tensor's spec,
# which state outputs are their inputs, epsilon, the compute config and the knobs above); a call with a known key
# re-sends the cached descriptor with its buffer addresses replaced. Entries hold no tensors.
PROGRAM_CACHE_SIZE = 64
_program_cache = OrderedDict()  # key -> (descriptor, reader kernels, writer kernels)
_KNOB_NAMES = tuple(
    name
    for name, value in globals().items()
    if name.isupper() and isinstance(value, (bool, int, float, str, tuple, type(None)))
)


def knob_values():
    module = globals()
    return tuple(module[name] for name in _KNOB_NAMES)


def clear_program_cache():
    _program_cache.clear()


def program_cache_size():
    return len(_program_cache)


def cached_program_descriptor(key, inputs, outputs, *, epsilon, compute_kernel_config):
    entry = _program_cache.get(key)
    if entry is None:
        descriptor = create_program_descriptor(
            inputs, outputs, epsilon=epsilon, compute_kernel_config=compute_kernel_config
        )
        # The structural hash leaves out per-core argument values and counts; the key covers them.
        descriptor.custom_program_hash = hash((ttnn.compute_program_descriptor_hash(descriptor), key)) % (1 << 64)
        kernels = descriptor.kernels  # references into the descriptor: reader, writer, compute per core group
        entry = (descriptor, kernels[0::3], kernels[1::3])
        _program_cache[key] = entry
        while len(_program_cache) > PROGRAM_CACHE_SIZE:
            _program_cache.popitem(last=False)
        return descriptor
    _program_cache.move_to_end(key)
    descriptor, readers, writers = entry
    reader_args, writer_args = reader_addresses(inputs, outputs), writer_addresses(outputs)
    for kernel in readers:
        kernel.common_runtime_args = reader_args
    for kernel in writers:
        kernel.common_runtime_args = writer_args
    return descriptor
