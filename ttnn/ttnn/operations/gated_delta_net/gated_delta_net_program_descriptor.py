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

import ast
import os
import struct
from collections import OrderedDict
from pathlib import Path

import ttnn

# The only Python binding of tt_metal::KernelBuildOptLevel (KernelDescriptor.opt_level).
from ttnn._experimental.fabric_mux import KernelBuildOptLevel

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
# R-BCAST: NOC 1 moves only toward lower x / y and starts a rectangle multicast at its highest corner (NOC 0 the
# opposite), so a sender at a group's first core sent every multicast around the torus, across every other group's.
# On: the q injector is the group's first core and sends on NOC 0 (its reader and writer swap NoCs), the k injector
# its last core on NOC 1: each path starts at the sender and q and k never share a NoC.
BCAST_SPLIT_NOCS = True
# R-QKCORE on R-VSPLIT pieces (decode, every sequence's pieces one per core, e.g. B = 1): the q / k part runs on 2 Hk
# idle cores instead of two pieces of each key head, so every state core is a receiver. Per key head its pieces fill a
# rectangle with the q core left of its first row (sends on NOC 0) and the k core right of its last (NOC 1).
QK_CORES_VSPLIT = True
# QK_CORES_VSPLIT: the q core left of the rectangle sending on NOC 0 (True), or both q / k cores right of its last row on
# NOC 1 (False: the two multicasts share a NoC, but no kernel group of q cores with swapped NoCs; every kernel group
# costs ~13 us of host time per call). perf_decode_b1 12.79 (True) vs 12.99 us (median of launches 2..12).
QK_VSPLIT_SPLIT_NOCS = False
# R-QKCORE (B=1 QK_CORES_VSPLIT, B=8 / B=16 pair cores): the q / k cores hand over in two stages, the conv output y as
# soon as it is made and its inverse norm (rsqrt(sum y^2 + eps), times K^-1/2 for q) when the norm is done, so the
# state cores transpose y_k, take y_q . y_k and read the state with the raw rows while the norm runs, and scale by the
# two scalars after (fp32 SFPU).
QK_RAW_HANDOFF = True
# Decode B=8 (median of launches 2..12): y lands on the state cores at 8.1-10.5 us instead of 12.3-14.9 (with
# QK_CORES_AFTER_GROUP), the inverse norms at 10-12; the wall stays at 39 us (state compute and write-back bind).
# R-QKCORE pair cores (decode R-STEP): each key head's q and k core placed right after its pair cores in row-major order
# (False: every q / k core after the last key head's). NOC 1 starts a multicast at the rectangle's highest corner, so a
# sender just past its group reaches it on a short path instead of from the grid's bottom rows across the other groups.
# y send 1.3-3 -> 0.3-0.6 us after its pack at B=8; wall 39.0 vs 38.9 (B=8), 69.4 vs 69.3 us (B=16): noise.
# Off since refinement 32: the q / k cores between the groups cut the kernel groups into 16 core ranges per RISC
# (B=8 / B=16), after the last group into 5, and the dispatcher writes binaries and launch messages once per range
# (start gap in the QB2 decode trace 28 / 40 us per layer). Kernel B=8 35.74 -> 36.22 us, B=16 69.71 -> 70.02.
QK_CORES_AFTER_GROUP = False
# R-QKCORE q cores: percent of the window's history rows their reader copies (the writer the rest, and it sends the
# small inputs; k cores split half and half). B=8: 70 / 85 make the reader the later RISC (last column 7.6 / 7.9 vs
# 7.4 us at 50); 50 emits no define.
QK_Q_READER_PCT = 50
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
# R-PREP placement in rectangles (see _prep_rect_place): the dispatcher writes kernel binaries and launch messages once
# per core range of a kernel group, after the previous program has ended when the binaries overlap its own (in the
# qwen36 prefill chunk: 48 ranges, 69 us). False: one blob per group (PREP_CLUSTER).
PREP_RECT = True
# Tile rows per sequence up to which PREP_RECT places (T=1024 233.0 us against 234.2 in blobs, T=2048 368.5 / 368.4,
# T=4096 808.4 / 808.0: 0.4 us of kernel against 59 us of start gap per layer in the QB2 4096-token prefill trace).
PREP_RECT_MAX_ROWS = 128
# The same for PREP_SHARE_QK plans of one sequence with one value head per group (see _prep_rect_shared; chip T=128 to
# 512: 69 -> 11 core ranges per RISC).
PREP_RECT_SHARED = True
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
# R-PREP: prepped tile rows in flight per state core (slots of its inbox buffers), at most. 13 is what the 16
# semaphore ids leave (one landed counter per slot, the grant and o counters); with the prep rows no longer paced
# by the norms the last prep cores of a round wait for their grant: T=2048 419 us at 12, 414 at 13.
INBOX_DEPTH = 13
O_IN_DEPTH = 2  # R-PREP: a prep core's tile rows of chunk outputs o in flight (slot k % O_IN_DEPTH for its k-th row)
O_ACC_DEPTH = 2  # R-PREP: a state core's tile rows of o awaiting its writer
PREP_SEND_HEADS = None  # R-PREP: head blocks a prep core holds for its writer; None = one tile row's (r)
# R-PREP norm split: a prep core preps all of its tile rows before any RMS norm (the norms of a row no longer sit
# between two rows of the state cores' supply), and the cores with one tile row more than the others (rows % stride of
# them) normalize nothing: the other cores of the group take the norm units (one value head of a tile row each) round
# robin. cb_o_in then holds every unit a core normalizes (no slot is reused, so o needs no flow control), each counted
# by its own semaphore (the inbox ids, unused on a prep core): only where a core's units are at most the inbox slots
# and PREP_NORM_MAX_ROWS tile rows' worth.
PREP_NORM_SPLIT = True
PREP_NORM_MAX_ROWS = 8
# PREP_NORM_SPLIT plans size the norm buffers (zw, o^2, inverse norm, z) for one unit instead of a tile row. With the
# landings counted per piece (one cumulative semaphore each, not one per unit) the split then also fits a chip
# T=4096 call (21 units per norm core, 1061 kB against 1217): 716.9 -> 682.4 us.
PREP_NORM_SLIM = True
# R-PREP prep writer: a head's state cores are told that their row landed after that head's block, not after the
# row's last one.
PREP_HEAD_UP = True
# The inverse norms (L2 norm of q / k, RMS norm) on whole tiles: rsqrt(sum + eps) * scale only on the vectors that hold
# column 0, where the row sum is and which every reader broadcasts (8 of a tile's 32 vectors).
NORM_RSQRT_COL0 = True
# R-PREP prep core, at most 16 value heads: softplus and sigmoid of the gate tiles on their left faces only (the gate
# columns are the value heads').
PREP_GATES_LEFT = True
# R-PREP prep core with STATE_FOLD: the chunk vectors exp(c), exp(c_last - c) and -exp(c) only in column 0 of their
# tiles; the products that read them broadcast it (FPU by column / by row, the M~ product by row).
PREP_VEC_COL0 = True
# PREP_NORM_SPLIT: a prep core whose stride position is past the inbox depth sends a row only when the state cores
# have freed its slot, and its next row's head blocks wait for that send. It normalizes this many of its norm units
# between the conv and the heads of each later row (the time it would wait), where their o precedes its own rows.
PREP_NORM_MID = 1
# R-PREP prep cores (bf16 tokens): silu through the bf16 arm of the SFPU sigmoid (exp to one bf16 ULP, one reciprocal
# iteration) instead of the fp32 one, in the fp32 DEST. A mask: 1 the gate factor silu(z) of the bf16 output, 2 the
# conv's v columns, 4 its q / k columns.
PREP_SILU_BF16 = 1
# The conv columns not under PREP_SILU_BF16 (prep and q / k cores): the bf16 arm's exp with two reciprocal iterations
# and no rounding of the result to bf16. T=4096 769.4 -> 717.0 us, state relative RMS 1.29e-3 -> 1.38e-3 (the rounding
# is what doubles it under PREP_SILU_BF16 6: 2.74e-3 at 722.1 us).
PREP_SILU_MID = 1
# R-PREP state step: the prep core sends -(beta exp(c)) (*) kn, so the state core accumulates the k read onto beta V in
# DEST (the rhs of M in one window: no packed reads, no subtraction window) and starts the output's window with the q
# read. Same operands and fp32 accumulation as the separate read.
STATE_FUSED_RHS = True
# R-PREP state core with two update batches per step (its piece of the state is the whole fp32 DEST, one batch per
# half): the upper batch stays in its DEST half from step to step, so only the lower batch is unpacked per step
# (4 of the 8 tile loads at Vb = 2). Needs STATE_FUSED_RHS (the step's other windows are counted) and half-sync DEST.
STATE_HOLD = True
# R-PREP head block: row scalings as FPU products (ELWMUL) instead of fp32 SFPU products; the scaling vector then
# passes the 19-bit operands like kn / qn themselves. A mask: 1 beta exp(c) (*) kn, 2 exp(c) (*) qn,
# 4 kn^T (*) exp(c_last - c)^T.
PREP_SCALE_FPU = 7
# A scaling vector of PREP_SCALE_FPU that reaches the state (masks 1 and 4) is scaled up by this factor (fp32 bits)
# first: an FPU operand is truncated to 10 mantissa bits, a mean loss of 0.36 units in the last place at each of the
# truncations on the way to the state (the vector, its product, the product as the state core's operand).
# 1 + 1.5 * 2^-10; state_carry_one_call_f32 relative RMS 1.45e-3 (SFPU) / 1.90e-3 (factor 1) / 0.90e-3.
PREP_FPU_UNBIAS_BITS = 0x3F803000
# R-PREP head block: beta folded into M (the inverse is solved on diag(beta) instead of the identity), so a head needs
# neither beta exp(c) (one exp tile and a broadcast product) nor beta (*) v (an SFPU window): the state core gets v as
# it is and exp(c) (*) kn. Same operand count through the FPU; beta reaches U through M's truncation instead of rhs's.
PREP_M_BETA = True
# M's unit diagonal passed the FPU exactly; beta on it is truncated to 10 mantissa bits like rhs, the other operand of
# U = M x rhs. The writer's diag(beta) page is scaled up by the mean loss of those two truncations (0.36 units in the
# last place each), in quarter units: 3 is a factor of 1 + 0.75 * 2^-10.
PREP_M_BETA_UNBIAS_Q = 3
# R-PREP state step on operands with M folded in by the prep core (needs PREP_M_BETA, STATE_FUSED_RHS and
# PREP_SCALE_FPU 1 | 2): U = M x V + Mk x S and O = (QD M) x V + q' x S, so the step has no rhs window and no pack /
# unpack round trip between rhs and U, and O does not wait for U. The prep core pays 2 + Kt matmul pairs per head for
# it (QD x M, QD x M~, M~ x kn in place of the kn scaling, + onto q): 1.3 us per head. T=1024 225.0 -> 209.4 us,
# T=4096 682.0 -> 672.4. "auto": off where the q / k cores' plan gives its prep cores more than one tile row each
# (prep-bound: T=512 134.3 -> 139.0 with it).
STATE_FOLD = "auto"
# STATE_FOLD: the factor on -exp(c) in M~ = M (*) -exp(c)^T (fp32 bits), the unbias of the truncations of M~ and of
# Mk as FPU operands (PREP_FPU_UNBIAS_BITS counted the vector, kn's product and rhs).
STATE_FOLD_UNBIAS_BITS = 0x3F801800
# STATE_FOLD: PREP_M_BETA_UNBIAS_Q of the folded step, where v passes the FPU as well (M x V). The smallest setting of
# the scan at which no gate case's state error is above the unfolded step's.
STATE_FOLD_M_UNBIAS_Q = 4
# R-PREP prefill of one sequence: the token axis in SEQ_SPLIT segments whose state chains run in parallel. Segment 0
# steps the state; segment s >= 1 steps X = [P | H] from [I | 0] (the P half on a second set of state cores, no v term)
# and, once segment s - 1 has its final state S_in, combines S_out = P x S_in + H and corrects each tile row's output
# O = Oh_P x S_in + Oh_H (Oh rows held on the state cores until then). Needs K = V, STATE_FOLD and a PREP_NORM_SPLIT
# plan; (2 S - 1) Hv pieces state cores, else the call runs unsplit. 1: off. 2 at Hv = 12: T=512 134.3 -> 261.4 us,
# T=1024 209.2 -> 475.5 (the prep cores bind and it leaves 32 of the 84); no plan fits at T >= 2048 or for 3 segments.
SEQ_SPLIT = 1
# R-PREP norm-split plans with the folded step: the last STATE_NORM_TAIL tile rows of a sequence are normalized by the
# state cores themselves after the chain (held o, partial sums gathered over the pieces as in R-VSPLIT, z and
# norm_weight read by their readers, output rows stored by their writers), so the prep cores' norm backlog after the
# chain is shorter. A row costs a state core 6.4 us after the chain. T=4096 (128 tile rows): 0 rows 672.4 us, 8 661.8,
# 12 654.7, 16 660.9, 24 711.0; T=2048: 4 rows 343.9 against 338.3, T=1024: 1 row 210.4 against 209.3 (the chain binds
# there). "auto": 3 rows per 32 from 128 tile rows on, else none.
STATE_NORM_TAIL = "auto"
SEG_ROW_DEPTH = 2  # SEQ_SPLIT: tile rows of Oh_P in flight from the P cores to a segment's state core
ROW_DEPTH = 2  # tile rows of gate / z / output tiles in flight
# Per-unit inputs (S0, conv / gate / norm weights) in flight on a core that runs several units: the next unit's reads
# overlap this unit's compute instead of waiting for its last pop (decode B=16: 2 units on most cores).
UNIT_INPUT_DEPTH = 2
# Pair states (S0) in flight, at most unit_depth (decode B=16: 1 measured 95.5 us vs 93 us at 2).
STATE_IN_DEPTH = 2
# R-QKCORE: pair cores in logical rows [0, SWAP_NOC_ROWS) read on NOC 1 and write on NOC 0 (0: none). NOC 0 brings the
# top rows' read responses around the torus (their weights / gate rows landed at 25-31 us vs 4-5 us on the lower rows).
# Decode B=8 / B=16: 0 rows 48.8 / 86 us, 2 rows 42.3 / 73.9, 3 rows 42.6 / 74.1, 4 rows 51.6 / 78, 5 rows 56 / 90.
# Decode B=1 (QK_CORES_VSPLIT state cores): 0 rows 15.3-15.4, 1 row 15.0-15.2, 2 rows 14.7-14.9, 3 rows 14.6-14.8,
# 4 rows 15.2-15.4 us.
SWAP_NOC_ROWS = 3
# The same when every pair core runs one whole-V unit (None: SWAP_NOC_ROWS). Decode B=8, 6 interleaved pairs on two
# chips: 3 rows 39.35 us, 2 rows 35.63 (row 2 leaves NOC 0: 22 instead of 31 cores write there). B=1 (pieces) keeps 3:
# 11.96 vs 12.44 at 2. B=16 / B=32 keep 3 (70.5 / 156.7 at 2 vs 69.2 / 156.1).
SWAP_NOC_ROWS_ONE_UNIT = 2
# Logical core rectangles (x_lo, y_lo, x_hi, y_hi), inclusive, whose pair cores take the opposite NoCs of the row rule.
SWAP_NOC_CORES = ()
# S0 tiles per DRAM read barrier on a core (None: the whole slice at once). Bounds the bulk a core queues at the banks
# ahead of other cores' small window reads. "auto": 8 tiles on R-QKCORE pair cores with one or two whole-V units
# (decode B=8 36.48 -> 35.81 us, B=16 70.25 -> 68.90, 3 interleaved pairs; 4 tiles 37.3 at B=8, 12 tiles 70.1 at
# B=16), the whole slice elsewhere (B=32 with 8: 152.7 against 150.4).
S0_READ_TILES = "auto"
# R-QKCORE pair cores running several units (sequences of one value head, so one key head): kn^T and d are computed
# once per core and held for every unit (SHARE_QK_FINISH), and a unit's RMS norm runs after the next unit's token step
# (UNIT_LAG), so the core's last state is final, and written, while the norms run.
SHARE_QK_FINISH = True
UNIT_LAG = True
# Units per pair core from which UNIT_LAG applies. Two units (decode B=16) run without it and with the fused token
# vector: kernel 69.8 against 70.0 us, pair role text 58.9 -> 48.6 kB (as B=8), so the trace allocator evicts older
# binaries for it and the QB2 decode trace starts it 6.5 us after the previous program instead of 19.3 (median over
# the layers). Four units (B=32) keep the lag: 150.8 against 156.1 us without.
UNIT_LAG_MIN_UNITS = 3
# R-QKCORE pair cores: pair p updates and writes its final state starting at tile (p * STATE_STAGGER) mod Kt Vt, so the
# cores' simultaneous writes spread over the DRAM banks instead of all starting at the bank of tile 0 (0: in order).
STATE_STAGGER = 3
# Decode: a unit's S0 read issued with its window rows (own transaction id) instead of after the weights / gate rows.
# Measured with the NoC swap at B=8 / B=16: 44.7 / 75 us on vs 42.8 / 74.4 off, so off (0).
# "auto" (refinement 37): on for R-QKCORE pair cores with two units (decode B=16 69.03 -> 67.15 us, 6 interleaved
# pairs: the second unit's S0 no longer waits for its small inputs behind the other cores' S0); off elsewhere (B=8
# 35.6 -> 37.9, B=32 151.5 -> 154.5). 2: the units after a core's first only (B=16 69.0: no gain).
EARLY_S0 = "auto"
# R-QKCORE pair cores whose every unit's S0 fits cb_state_in (STATE_IN_DEPTH): the writer RISC, idle until the first
# gates, reads S0 (every unit's at once, one transaction id each) on its NoC instead of the reader after each unit's
# window, weights and gate rows. Off: it moves S0 to the other NoC, which the SWAP_NOC_ROWS tuning picked against.
# Decode B=16 / B=8 (median of launches 2..12), off 73.4 / 40.6 us; on, issued after the window zeroing 79.3 / 45.0,
# after the reader's first inputs 85.0 / 47.0 (the late rows' weights / gates then land at 5-15 us instead of 17-30, but
# their S0 at 38-51 us), with KEEP_NOC 90.3 / 48.9, rows 3-4 only 84.4-84.8 / 45.1-45.7. B=1 12.95 -> 13.7-14.2.
S0_WRITER = False
# S0_WRITER: the writer issues S0 once the reader's first unit's weights and gate rows have landed (reader-to-writer
# token, CB_S0_GO) instead of right after the window zeroing.
S0_WRITER_AFTER_INPUTS = True
# S0_WRITER: on its cores the reader and writer swap NoCs relative to SWAP_NOC_ROWS, so S0 keeps the NoC picked for it.
S0_WRITER_KEEP_NOC = False
# S0_WRITER: the logical rows whose pair cores take it (None: every row).
S0_WRITER_ROWS = None
# R-QKCORE (decode): each key head's q core reads A_log, dt_bias, the b / a gate tile and norm_weight once per call and
# multicasts them to the key head's state cores, which copy them from L1 instead of reading DRAM: page 0 of every small
# tensor sits in one bank, where the state cores' reads queued behind the other cores' S0 (Refinement 19). Needs every
# value head of a key head in one gate column tile.
SMALL_INPUTS_MCAST = True
# Finished pair states awaiting the writer, at most unit_depth: the writer issues a core's non-last state after the next
# unit's coefficients, so compute's next update must not wait for it.
STATE_OUT_DEPTH = 2
# R-QKCORE pair cores: the NoC every core writes its final state on (None: its writer's). A core whose writer is on the
# other NoC (SWAP_NOC_ROWS) hands every state tile to its reader instead.
STATE_WRITE_NOC = None
# R-QKCORE pair cores writing on STATE_WRITE_NOC (or every core when it is None): the reader also writes every
# STATE_SPLIT-th final-state tile, on the other NoC (0: none).
STATE_SPLIT = 0
# R-QKCORE pair cores running several units: the reader writes the final states of all but the last unit, as compute
# packs them (the writer would issue them only at the next unit's store point), the writer the last unit's.
STATE_READER_EARLY_UNITS = False
# R-QKCORE pair cores running several units (UNIT_LAG): after unit 0 the writer polls instead of blocking, so a unit's
# coefficients never wait for an earlier state's write-back (decode B=32: the write-starved cores' compute idled 20 to
# 77 us per unit behind state k - 1's barrier). Up to two final states in flight, each freed on its own transaction id.
# 2: a state is also issued as soon as it is packed instead of after the next unit's coefficients.
WRITER_ASYNC = 1
# R-QKCORE pair cores: final-state tiles a core has in flight (issued, not acknowledged) at most; 0: no bound. At every
# router the passing packets and the local writer take turns, so the cores next to the DRAM columns keep half of a
# link and the cores upstream of them starve; a bound makes a core skip turns while its tiles are on the way.
STATE_WRITE_WINDOW = 0
# The bound applies from this many units per pair core on.
STATE_WRITE_WINDOW_UNITS = 3
# Decode token step (shared q / k builds): the token vector of a column tile in one SFPU pass over the token's band
# instead of ~14 scalar / binary / broadcast SFPU calls with their inits (2.0 -> 0.6 us per unit at Vb = 4). "auto":
# on for R-VSPLIT pieces (decode B=1: 12.5 -> 12.0 us), off for whole-V units (B=8 / B=16 are DRAM-bound there:
# the earlier state writes push back the late cores' S0 reads, 39.3 -> 40.1 / 69.1 -> 69.6 us).
TOKEN_VECTOR_FUSED = "auto"
# Token step state update: the decay of a DEST batch of state tiles in one SFPU pass instead of one call per tile.
DECAY_ONE_PASS = True
# R-BCAST: a kernel group per qk role instead of one for all its cores. Smaller reader / writer binaries (the q / k
# role compiled in), but the injector group stays the largest and decode B=1 host time per call rises 112 -> 166 us.
SPLIT_BCAST_GROUPS = False
# UNIT_LAG builds (decode pair cores with several units): the RMS norm as one function instead of inlined at its three
# call sites. Pair role text at B=16 58.9 -> 51.9 kB: the op's binaries overlap the previous program's less often in
# the kernel-config buffer, and a trace then need not wait for it before writing them.
RMS_ONE_COPY = True
# Per-block device profiler zones (GDN_ZONE in the kernels: compute blocks, reader / writer units and rows), on with
# env GDN_PROFILE_ZONES=1, for --profile runs (never compiled under NoC tracing). Each zone execution spends 2 of a
# RISC's ~250 markers per launch.
# Level 2 also compiles the point marks (GDN_MARK) of each block's end, for chain timelines (zone_timeline.py).
PROFILE_ZONES = int(os.environ.get("GDN_PROFILE_ZONES", "0") or 0)
# env GDN_ZONE_SKIP=n: each RISC leaves its first n zone executions unrecorded (captures of a later window).
PROFILE_ZONE_SKIP = int(os.environ.get("GDN_ZONE_SKIP", "0") or 0)
# Knob overrides for before / after runs in one build: GDN_KNOBS="NAME=literal,NAME=literal".
_knobs = os.environ.get("GDN_KNOBS", "")
for _item in filter(None, _knobs.split(";" if ";" in _knobs else ",")):  # ";" when a value holds a comma
    _name, _literal = _item.split("=", 1)
    assert _name.isupper() and _name in globals(), f"GDN_KNOBS: unknown knob {_name}"
    globals()[_name] = ast.literal_eval(_literal)
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
# Tensors of the call itself in L1 below that address (the packed projection rows at T=2048 take ~155 KB per core) lower
# the bound for the call to their lowest address, in steps of L1_END_STEP: the regimes shrink their buffers to it.
L1_END_STEP = 16384

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
CB_STATE_DONE = 51  # STATE_SPLIT: reader-to-writer token per final state, once the reader's tiles of it have left
CB_RINV_IN = 52  # QK_RAW_HANDOFF: the landed inverse norms of k (tile 0) and q (tile 1), column 0 per token row
CB_S0_GO = 53  # S0_WRITER_AFTER_INPUTS: reader-to-writer token, the first unit's weights and gate rows have landed
# SMALL_INPUTS_MCAST: the q core's reads of the small inputs, multicast to the same address on the state cores. Tiles:
# A_log, dt_bias, b, a (each the gate column tile of the key head), then norm_weight's Vt.
CB_SMALL_IN = 54
CB_CHUNK_VEC_MM = 55  # PREP_SCALE_FPU: cb_chunk_vec as an FPU operand (same pages)
CB_SEG_IN = 56  # SEQ_SPLIT state core of a segment 1..: P (Kt x Kt), then the entering state's piece (Kt x Vb)
CB_SEG_ROW = 57  # SEQ_SPLIT: Oh_P of a tile row (Kt tiles), SEG_ROW_DEPTH rows
SMALL_IN_GATE_PAGES = 4
SMALL_MCAST_NAMES = ("CB_SMALL_IN", "SMALL_IN_GATE_PAGES")
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
# Compute kernel source per role; the shared blocks are in gated_delta_net_compute_common.hpp and _chunk.hpp.
COMPUTE_SOURCES = {
    ROLE_LOCAL: "gated_delta_net_compute_local.cpp",
    ROLE_PREP: "gated_delta_net_compute_prep.cpp",
    ROLE_STATE: "gated_delta_net_compute_state.cpp",
    ROLE_QK: "gated_delta_net_compute_qk.cpp",
    ROLE_PREP_QK: "gated_delta_net_compute_qk.cpp",
}
# Per-core q/k prep role (R-BCAST): own = preps its own q and k (no sharing); q / k injector = preps qn / kn and
# multicasts it, receives the other; receiver = receives both.
QK_OWN, QK_Q_INJECTOR, QK_K_INJECTOR, QK_RECEIVER = 0, 1, 2, 3
# A kernel group whose cores share one qk role compiles it in (GDN_GROUP_QK_ROLE), so its binary drops the other
# roles' paths (e.g. an R-QKCORE pair core never preps q / k); QK_PER_CORE: the cores' runtime arg decides.
QK_PER_CORE = 4

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
    CB_SMALL_IN,
)


def _kernel_defines():
    """CB indices and spec constants for all three kernels (as GDN_<name>): defined here once, never restated."""
    names = [n for n in globals() if n.startswith("CB_") and n not in SMALL_MCAST_NAMES] + [
        "HISTORY",
        "CONV_TAPS",
        "TOK_COEF_PAGES",
        "D_PAGES",
        "CHUNK_COEF_PAGES",
        "TOK_COEF_DEPTH",
        "CHUNK_VEC_PAGES",
        "CHUNK_MAT_PAGES",
        "QK_OWN",
        "QK_Q_INJECTOR",
        "QK_K_INJECTOR",
        "QK_RECEIVER",
        "QK_PER_CORE",
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
        ((CB_CHUNK_VEC, CB_CHUNK_VEC_MM), CHUNK_VEC_PAGES if chunked else 0),
        ((CB_CHUNK_MAT,), CHUNK_MAT_PAGES if chunked else 0),
        ((CB_UW,), Vt if chunked else 0),
        ((CB_TRI_MASK,), 2 if chunked else 0),  # the mask, then the identity (R-PREP inverse of I + L)
        # R-BCAST injector: its qn or kn (compute -> writer, multicast source).
        ((CB_QK_STAGE,), Kt if _share_qk(pieces, tokens) else 0),
        ((CB_RINV_IN,), 0),  # QK_RAW_HANDOFF (sized by the R-QKCORE plan)
        ((CB_SMALL_IN,), 0),  # SMALL_INPUTS_MCAST (sized by the R-QKCORE plan)
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
        ((CB_SEG_IN,), 0),
        ((CB_SEG_ROW,), 0),
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


def _state_fold(shared, rows, per_group):
    """STATE_FOLD of an R-PREP plan (PREP_SHARE_QK, tile rows per sequence, prep cores per group)."""
    ok = PREP_M_BETA and STATE_FUSED_RHS and PREP_SCALE_FPU % 4 == 3
    if STATE_FOLD == "auto":
        return int(ok and not (shared and rows > per_group))
    return int(bool(STATE_FOLD and ok))


def _prep_cb_plan(Kt, Vt, heads, conv_on_fpu, pieces, inbox, shared=False, norm_units=0, fold=0, hold=0, tail=0):
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
    prep[CB_CHUNK_MAT] = 5 if fold else 1  # L, the operand of the inverse; STATE_FOLD: + QD, M, M~, QD M~
    prep[CB_V_PREP] = heads * Vt
    prep[CB_PREP_SEND] = (PREP_SEND_HEADS or heads) * _head_block_pages(Kt, Vt)
    prep[CB_O_IN] = norm_units * Vt if norm_units else O_IN_DEPTH * heads * Vt  # PREP_NORM_SPLIT: per norm unit
    # STATE_FUSED_RHS: -beta too; PREP_M_BETA: diag(beta) in its place
    prep[CB_TOK_COEF] = (CHUNK_COEF_PAGES + int(STATE_FUSED_RHS or PREP_M_BETA)) * TOK_COEF_DEPTH
    prep[CB_NORM_W] = Vt
    prep[CB_RINV_O] = heads * TOKEN_TILE_ROWS_PER_BLOCK
    prep[CB_ZW] = 2 * heads * Vt  # row k's, made before row k - 1's RMS norm, which takes row k - 1's
    if norm_units and PREP_NORM_SLIM:
        # PREP_NORM_SPLIT normalizes one unit (Vt tiles of one head) at a time: its buffers hold a unit, not a tile row.
        prep.update({CB_ZW: Vt, CB_SQ_O: Vt, CB_RINV_O: 1, CB_Z: 2 * Vt})
    for idx, pages in _inbox_pages(Kt, Vb).items():
        state[idx] = inbox * pages
    state[CB_V] = 0  # the state core reads v from its inbox
    if tail:  # STATE_NORM_TAIL: the held rows' o, then the RMS norm's buffers of one piece
        state.update({CB_READS: tail * Vb, CB_Z: Vb * ROW_DEPTH, CB_NORM_W: Vb, CB_OUT_ROWS: Vb * ROW_DEPTH})
        state.update({CB_SQ_O: Vb, CB_RINV_O: 1, CB_OSUM_PART: 1, CB_OSUM_GATHER: 2 * pieces})
    if hold:  # SEQ_SPLIT: Oh of every tile row of a segment, P and the entering state, Oh_P rows in flight
        state.update({CB_READS: hold * Vb, CB_SEG_IN: Kt * Kt + Kt * Vb, CB_SEG_ROW: SEG_ROW_DEPTH * Kt})
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
    signal_bytes = SIGNAL_PAGE_BYTES * (WINDOW_DEPTH + STATE_OUT_DEPTH)  # cb_window_zeroed, cb_state_done
    return cb_bytes + BF16_TILE_BYTES + signal_bytes + _row_stage_bytes(token_bytes)


def l1_cb_budget(device):
    """Bytes of static CBs per core that keep the CB region inside L1_CB_REGION_END."""
    return L1_CB_REGION_END - ttnn.device.get_allocator_base_address(device, ttnn.BufferType.L1)


def l1_region_end(tensors):
    """Where the static CB region must end when a tensor of the call (the packed rows, z, an L1 output) lives in L1
    below L1_CB_REGION_END: its lowest L1 buffer address, rounded down to L1_END_STEP (part of the program cache key);
    None when the default bound holds."""
    lowest = min(
        (t.buffer_address() for t in tensors if t.memory_config().buffer_type == ttnn.BufferType.L1),
        default=L1_CB_REGION_END,
    )
    return None if lowest >= L1_CB_REGION_END else lowest // L1_END_STEP * L1_END_STEP


def check_l1_footprint(role, footprint, budget, region_end=None):
    if footprint > budget:
        beside = "" if region_end is None else " (below the call's L1 tensors)"
        raise RuntimeError(
            f"gated_delta_net: {role} cores need {footprint} B of static circular buffers, above the {budget} B that "
            f"keep the CB region at or below L1 address {region_end or L1_CB_REGION_END}{beside}"
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
    return _prep_head_options(batch, value_heads, key_heads, tokens, Vb, spare_cores)[0]


def _prep_head_options(batch, value_heads, key_heads, tokens, Vb, spare_cores):
    """Every (value heads per R-PREP group, PREP_SHARE_QK) the knobs allow, the modelled fastest first."""
    ratio = value_heads // key_heads
    if PREP_HEADS is not None:
        assert ratio % PREP_HEADS == 0, "PREP_HEADS must divide the value heads per key head"
    options = [
        (h, shared)
        for h in ((PREP_HEADS,) if PREP_HEADS is not None else range(1, ratio + 1))
        if ratio % h == 0
        for shared in ((False, True) if PREP_SHARE_QK == "auto" else (bool(PREP_SHARE_QK),))
    ]
    return sorted(
        options,
        key=lambda opt: (
            _prep_estimate(batch, value_heads, key_heads, tokens, Vb, opt[0], spare_cores, opt[1]),
            opt[1],
            -opt[0],
        ),
    )


def _prep_layout(
    batch,
    value_heads,
    key_heads,
    tokens,
    Kt,
    Vt,
    pieces,
    chunked,
    dtype,
    spare_cores,
    budget,
    shrink=False,
    hold=0,
    tail=0,
):
    """R-PREP plan, or None to keep prep and state on the same core: (prep cores per group cap, inbox depth, value
    heads per group). A group is (sequence, key head, subset of `heads` of its value heads). Needs one unit per core
    (R-VSPLIT), at least one idle core per group, the key head's value heads in one gate tile column (one gate tile
    per prepped tile row) and the merged plan inside L1. shrink (a budget lowered by the call's L1 tensors): when the
    modelled fastest grouping does not fit, the next ones are tried (fewer heads per group: smaller prep cores)."""
    ratio = value_heads // key_heads
    if not (PREP_OFFLOAD and chunked and pieces > 1):
        return None
    if any((hk * ratio) // TILE != (hk * ratio + ratio - 1) // TILE for hk in range(key_heads)):
        return None
    if min(_seq_tile_rows(seq, tokens) for seq in range(batch)) < PREP_MIN_TILE_ROWS:
        return None
    options = _prep_head_options(batch, value_heads, key_heads, tokens, Vt // pieces, spare_cores)
    if hold:  # SEQ_SPLIT: no q / k cores; the first grouping that fits
        options = [option for option in options if not option[1]]
    for heads, shared in options if shrink or hold else options[:1]:
        per_group = _prep_positions(batch, value_heads, key_heads, heads, shared, spare_cores)
        if per_group < 1:
            continue
        # A prep core's next tile row is granted only after its current one has landed when its stride is at least
        # the inbox depth (the cumulative credit count relies on it).
        inbox = min([INBOX_DEPTH, per_group] + [_seq_tile_rows(seq, tokens) for seq in range(batch)])
        if shared:  # the q / k landing and grant counters follow the inbox's
            inbox = min(inbox, MAX_SEMAPHORES - LANDED_FIRST_SEM_ID - 4 - _prep_qk_dests(ratio, heads))
        # The deepest inbox that fits L1 (a shallower one only costs slack between a grant and its landing).
        if hold:
            # SEQ_SPLIT: the segment counters follow the inbox's (one, then one per piece). Its few prep cores per
            # group would leave fewer inbox ids than the norm units' landing counters need (heads x pieces): the
            # inbox is that deep all the same and the grants stop at the stride (GDN_GRANT_DEPTH).
            inbox = min(max(inbox, heads * pieces), MAX_SEMAPHORES - LANDED_FIRST_SEM_ID - 1 - pieces)
        fold = _state_fold(shared, max(_seq_tile_rows(seq, tokens) for seq in range(batch)), per_group)
        while inbox >= 1:
            plans = _prep_cb_plan(
                Kt, Vt, heads, _conv_on_fpu(dtype), pieces, inbox, shared, fold=fold, hold=hold, tail=tail
            )
            if _prep_plan_bytes(plans, dtype) <= budget:
                return per_group, inbox, heads, shared
            inbox -= 1
    return None


def _qk_vsplit_rect(grid, batch, value_heads, key_heads, pieces):
    """QK_CORES_VSPLIT: (width, rows) of each key head's rectangle of pieces, the widest whose rectangles stack in the
    grid with a column to spare on either side (the q and k cores), or None."""
    group = batch * (value_heads // key_heads) * pieces
    for width in range(min(group, grid.x - 2), 0, -1):
        if group % width == 0 and key_heads * (group // width) <= grid.y:
            return width, group // width
    return None


def _qk_core_units(batch, tokens, value_heads, key_heads, Kt, Vt, pieces, num_cores, grid=None):
    """R-QKCORE plan: units (sequences of one value head) per pair core, or None. Needs decode with every sequence in
    one tile row, one pair per unit (R-STEP; or one piece per core under QK_CORES_VSPLIT, placed by _qk_vsplit_rect),
    2 Hk idle cores for the q / k cores, and a v-only window group whose FPU conv batch also divides the pair's columns
    (compute's CONV_BATCH)."""
    if not (SHARE_QK_CORES and tokens == 1 and batch <= TILE):
        return None
    if pieces > 1:
        vsplit = QK_CORES_VSPLIT and grid is not None and _qk_vsplit_rect(grid, batch, value_heads, key_heads, pieces)
        return 1 if vsplit else None
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


def _noc_tracing():
    """NoC-event tracing inlines a recorder into every NoC call. A traced run measures traffic, never time, so under it
    the reader and writer are built for size and the zones (timing only) are left out: every role then fits the
    kernel-config text budget."""
    return os.environ.get("TT_METAL_DEVICE_PROFILER_NOC_EVENTS", "0") not in ("", "0")


def _dataflow_opt_level():
    return KernelBuildOptLevel.Os if _noc_tracing() else None


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
    return [t.buffer_address() for t in inputs] + [outputs[1].buffer_address(), outputs[2].buffer_address()]


# Inputs the writer reads: recurrent_state (S0_WRITER), then only in SMALL_INPUTS_MCAST programs A_log, dt_bias, b, a and
# norm_weight (other programs keep the shorter argument lists: host time per call grows with them).
WRITER_INPUTS = (5,)
SMALL_WRITER_INPUTS = (7, 8, 2, 3, 9)


def writer_inputs(small_mcast):
    return WRITER_INPUTS + (SMALL_WRITER_INPUTS if small_mcast else ())


def writer_addresses(inputs, outputs, small_mcast):
    return [outputs[0].buffer_address(), outputs[2].buffer_address()] + [
        inputs[i].buffer_address() for i in writer_inputs(small_mcast)
    ]


def _prep_rect_place(device, grid, n_groups, n_state, strides):
    """R-PREP cores as few rectangles per kernel group: the state cores of all groups as one block of 3 columns in
    the middle (a group's heads side by side, its pieces below one another), the prep cores that send on NOC 0 (which
    routes toward higher x and y) as the block of columns left of it, those on NOC 1 right of it, each group's rows
    of the left block at or above its state rows and of the right block at or below (the last group's wrap to the
    top rows: 3 hops on the torus). The rows above the state block hold the rest: its left column on NOC 0, its right
    column on NOC 1. Returns (state cores per group, prep cores per group in stride order, {(x, y): sends on NOC 0})
    or None when the plan does not fit this shape."""
    width = 3
    rows = n_state // width
    stride = strides[0]
    x0 = (grid.x - width) // 2
    y0 = grid.y - n_groups * rows
    if n_state % width or y0 < 0 or x0 < 1 or any(s != stride for s in strides):
        return None
    left = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(x0)]
    right = [ttnn.CoreCoord(x, (y0 + y) % grid.y) for y in range(grid.y) for x in range(x0 + width, grid.x)]
    n_left, n_right = len(left) // n_groups, len(right) // n_groups
    rest = stride - n_left - n_right
    top = [(ttnn.CoreCoord(x0, y), True) for y in reversed(range(y0))]
    top += [(ttnn.CoreCoord(x0 + width - 1, y), False) for y in range(y0)]
    if rest < 0 or rest * n_groups > len(top):
        return None
    # The NOC 0 column serves the first groups (nearest first), the NOC 1 column the last (through the wrap).
    order = sorted(range(n_groups), key=lambda g: (min(g, n_groups - 1 - g), g))
    extra = {g: [] for g in range(n_groups)}
    lo, hi = 0, len(top) - 1
    for _ in range(rest):
        for g in order:
            if g < (n_groups + 1) // 2:
                extra[g].append(top[lo])
                lo += 1
            else:
                extra[g].append(top[hi])
                hi -= 1
    state, picks, noc0 = [], [], {}
    for g in range(n_groups):
        ys = range(y0 + g * rows, y0 + (g + 1) * rows)
        state.append([ttnn.CoreCoord(x0 + j, y) for j in range(width) for y in ys])
        dst = [device.worker_core_from_logical_core(c) for c in state[g]]

        def hops(core_noc0):
            src = device.worker_core_from_logical_core(core_noc0[0])
            if core_noc0[1]:
                return sum((d.x - src.x) % NOC_GRID[0] + (d.y - src.y) % NOC_GRID[1] for d in dst)
            return sum((src.x - d.x) % NOC_GRID[0] + (src.y - d.y) % NOC_GRID[1] for d in dst)

        mine = [(c, True) for c in left[g * n_left : (g + 1) * n_left]]
        mine += [(c, False) for c in right[g * n_right : (g + 1) * n_right]] + extra[g]
        # Stride order (the first positions prep the most rows): the NoCs in turn, each nearest first (measured:
        # nearest first over both 236.4 us at T=1024, farthest first 244.4, in turn 233.0).
        on1 = sorted((m for m in mine if not m[1]), key=hops)
        on0 = sorted((m for m in mine if m[1]), key=hops)
        mine = [m for pair in zip(on1, on0) for m in pair] + on1[len(on0) :] + on0[len(on1) :]
        picks.append([c for c, _ in mine])
        noc0.update({(c.x, c.y): n for c, n in mine})
    return state, picks, noc0


def _prep_rect_shared(grid, n_kh, per_kh, pieces, stride):
    """PREP_SHARE_QK plan (one sequence, one value head per group) in column strips: a group is three columns, its
    state cores in the middle one (its pieces below one another), prep cores left of them sending on NOC 0 and right
    of them on NOC 1, so no two groups share a link on the way to their state cores; the key heads are bands of
    `pieces` rows below one another. Stride positions 0 and 1 take the two cores that reach both pieces without the
    torus wrap (NOC 0 routes east then south, NOC 1 north then west). The q / k cores: the columns right of the last
    group (NOC 1) and the rows above the bands (NOC 0). Returns (state cores per group, prep cores per group in stride
    order, {(x, y): sends on NOC 0}, q / k cores per key head) or None when the plan does not fit."""
    y0 = grid.y - n_kh * pieces
    side = range(3 * per_kh, grid.x)
    top = [ttnn.CoreCoord(x, y) for x in range(grid.x) for y in range(y0)]
    if y0 < 0 or pieces < 2 or stride > 2 * pieces or not side:
        return None
    state, picks, pools, noc0 = [], [], [], {}
    for kh in range(n_kh):
        ys = list(range(y0 + kh * pieces, y0 + (kh + 1) * pieces))
        pool = [ttnn.CoreCoord(x, y) for y in ys for x in side][: 2 * stride]
        noc0.update({(c.x, c.y): False for c in pool})
        need = 2 * stride - len(pool)
        if need > len(top):
            return None
        pool, top = pool + top[:need], top[need:]
        noc0.update({(c.x, c.y): True for c in pool[2 * stride - need :]})
        for j in range(per_kh):
            x = 3 * j + 1
            # Left column top down, right column bottom up, in turn: positions 0 and 1 need no wrap.
            cells = [c for pair in zip(((x - 1, y) for y in ys), ((x + 1, y) for y in reversed(ys))) for c in pair]
            if stride <= pieces:
                cells = [(x - 1, y) for y in ys]
            state.append([ttnn.CoreCoord(x, y) for y in ys])
            picks.append([ttnn.CoreCoord(*c) for c in cells[:stride]])
            noc0.update({c: c[0] < x for c in cells[:stride]})
        pools.append(pool)
    return state, picks, noc0, pools


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


def create_program_descriptor(
    inputs, outputs, *, epsilon, compute_kernel_config, packed=False, l1_end=None, seq_split=None, norm_tail=None
):
    """packed: z, b and a of `inputs` are mixed_qkv itself, whose rows hold [x | z | a | b]. l1_end: l1_region_end() of
    the call's tensors."""
    mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight = inputs
    output, conv_state_out, recurrent_state_out = outputs

    assert TOKEN_TILE_ROWS_PER_BLOCK == 1, "only one tile row per prep block is built"
    batch, value_heads, key_dim, value_dim = (int(d) for d in recurrent_state.shape)
    rows, channels = int(mixed_qkv.shape[1]), int(conv_state.shape[2])
    tokens = rows // batch
    Kt, Vt = key_dim // TILE, value_dim // TILE
    # SEQ_SPLIT: segments as sequences of the R-PREP plan (seqs of them with prep groups), then the P halves of
    # segments 1.. as further sequences of state cores only.
    segs = SEQ_SPLIT if seq_split is None else seq_split
    split = segs > 1 and batch == 1 and tokens % (TILE * segs) == 0 and Kt == Vt
    seqs = batch
    if split:
        seqs, tokens, batch = segs, tokens // segs, 2 * segs - 1
    hold = tokens // TILE if split else 0  # tile rows of Oh a state core of a segment 1.. holds
    # STATE_NORM_TAIL: tile rows per sequence the state cores normalize (every sequence keeps a row for the prep cores).
    tail = STATE_NORM_TAIL if norm_tail is None else norm_tail
    seq_rows_min = min(_seq_tile_rows(seq, tokens) for seq in range(batch))
    if tail == "auto":
        tail = seq_rows_min * 3 // 32 if seq_rows_min >= 128 else 0
    tail = 0 if split else max(0, min(int(tail), seq_rows_min - 1))
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
    if l1_end is not None:
        budget -= L1_CB_REGION_END - l1_end
    if V_BLOCK_TILES is None and pieces > 1:
        # R-PREP: the modelled fastest of the piece counts it can run (else R-VSPLIT keeps its pieces).
        best = None
        for option in range(2, min(pieces, PREP_MAX_PIECES) + 1):
            spare_cores = num_grid_cores - num_pairs * option
            plan = Vt % option == 0 and _prep_layout(
                seqs,
                value_heads,
                key_heads,
                tokens,
                Kt,
                Vt,
                option,
                chunked,
                mixed_qkv.dtype,
                spare_cores,
                budget,
                l1_end is not None,
                hold,
                tail,
            )
            if plan:
                cost = _prep_estimate(seqs, value_heads, key_heads, tokens, Vt // option, plan[2], spare_cores, plan[3])
                if best is None or cost < best[0]:
                    best = (cost, option)
        if best is not None:
            pieces = best[1]
    v_block_tiles = Vt // pieces
    num_units = num_pairs * pieces
    units_per_block = PAIRS_PER_BLOCK if pieces == 1 else 1  # R-VSPLIT: one unit per core
    num_blocks = (num_units + units_per_block - 1) // units_per_block
    work_grid = grid
    if (BCAST_RECT_GROUPS or BCAST_SPLIT_NOCS) and _share_qk(pieces, tokens):
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
    qk_per_core = _qk_core_units(batch, tokens, value_heads, key_heads, Kt, Vt, pieces, num_grid_cores, grid)
    qk_cores = []  # (core, key head, qk role)
    if qk_per_core and pieces > 1:
        width, rect_rows = _qk_vsplit_rect(grid, batch, value_heads, key_heads, pieces)
        assignment = []
        for hk in range(key_heads):
            y0 = hk * rect_rows
            units = [
                (seq * value_heads + hk * ratio + j) * pieces + piece
                for seq in range(batch)
                for j in range(ratio)
                for piece in range(pieces)
            ]
            x0 = 1 if QK_VSPLIT_SPLIT_NOCS else 0
            for idx, unit in enumerate(units):
                core = ttnn.CoreCoord(x0 + idx % width, y0 + idx // width)
                assignment.append((core, unit, 1, value_heads * pieces))
            y_last = y0 + rect_rows - 1
            q_core = ttnn.CoreCoord(0, y0) if QK_VSPLIT_SPLIT_NOCS else ttnn.CoreCoord(width, y_last)
            qk_cores.append((q_core, hk, QK_Q_INJECTOR))
            qk_cores.append(
                (ttnn.CoreCoord(x0 + width + (0 if QK_VSPLIT_SPLIT_NOCS else 1), y_last), hk, QK_K_INJECTOR)
            )
        cores_per_kh = len(assignment) // key_heads
        all_cores = _merged_ranges([core for core, _, _, _ in assignment] + [core for core, _, _ in qk_cores])
    elif qk_per_core:
        assignment = []
        free = iter(grid_cores)
        for hk in range(key_heads):
            for j in range(ratio):
                for s0 in range(0, batch, qk_per_core):
                    first = s0 * value_heads + hk * ratio + j
                    assignment.append((next(free), first, min(qk_per_core, batch - s0), value_heads))
            if QK_CORES_AFTER_GROUP:
                qk_cores += [(next(free), hk, role) for role in (QK_Q_INJECTOR, QK_K_INJECTOR)]
        cores_per_kh = len(assignment) // key_heads
        if not QK_CORES_AFTER_GROUP:
            qk_cores = [(next(free), hk, role) for hk in range(key_heads) for role in (QK_Q_INJECTOR, QK_K_INJECTOR)]
        all_cores = _merged_ranges([core for core, _, _, _ in assignment] + [core for core, _, _ in qk_cores])
    state_cores = [core for core, _, _, _ in assignment]
    state_set = {(c.x, c.y) for c in state_cores}
    qk_raw = bool(qk_cores) and QK_RAW_HANDOFF
    small_mcast = (
        bool(qk_cores)
        and SMALL_INPUTS_MCAST
        and all(hk * ratio // TILE == (hk * ratio + ratio - 1) // TILE for hk in range(key_heads))
    )
    # Who writes conv_state_out: 0 the pair cores (one unit per key head also the q / k columns), 1 the q / k cores
    # (whole tiles: every sequence's rows of a column at once), 2 (QK_CORES_VSPLIT) the q / k cores their own columns
    # and every state core its own v columns, each a column only that core reads, so in place needs no signal.
    cs_split = 0 if not qk_cores else (2 if pieces > 1 else 1)
    spare = [c for c in grid_cores if (c.x, c.y) not in state_set]
    prep = _prep_layout(
        seqs,
        value_heads,
        key_heads,
        tokens,
        Kt,
        Vt,
        pieces,
        chunked,
        token_dtype,
        len(spare),
        budget,
        l1_end is not None,
        hold,
        tail,
    )
    heads = ratio  # R-PREP: value heads per prep group

    def p_first(unit):
        """SEQ_SPLIT: the P-half unit beside `unit` of a segment 1.. (None for segment 0 and unsplit calls)."""
        seg = unit // pieces // value_heads
        return unit + (segs - 1) * value_heads * pieces if split and 1 <= seg < segs else None

    shared = False  # R-PREP: PREP_SHARE_QK (q / k cores)
    norm_split = False  # R-PREP: PREP_NORM_SPLIT
    fold = 0  # R-PREP: STATE_FOLD
    if prep is None:
        inbox = 0
        # The deepest per-unit input depth that fits L1 (1 when every core runs one unit).
        core_units = max(count for _, _, count, _ in assignment)
        unit_depth = min(UNIT_INPUT_DEPTH, core_units)

        def unit_lag(depth):
            # The reader must run a unit ahead of the lagged norm: two units of inputs in flight.
            return bool(qk_cores) and SHARE_QK_FINISH and UNIT_LAG and depth > 1 and core_units >= UNIT_LAG_MIN_UNITS

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
                CB_RINV_IN: 2 if qk_raw else 0,
                CB_SMALL_IN: SMALL_IN_GATE_PAGES + Vt if small_mcast else 0,
                CB_O_ACC: v_block_tiles * lag_units,
                CB_SQ_O: v_block_tiles * lag_units,
            }
            return [(idx, sizes.get(idx[0], pages)) for idx, pages in plan]

        cb_plan = local_plan(unit_depth)
        while unit_depth > 1 and _plan_bytes(cb_plan, token_dtype) > budget:
            unit_depth -= 1
            cb_plan = local_plan(unit_depth)
        check_l1_footprint("pair and q / k" if qk_cores else "pair", _plan_bytes(cb_plan, token_dtype), budget, l1_end)
    else:
        per_group, inbox, heads, shared = prep
        fold = _state_fold(shared, max(_seq_tile_rows(seq, tokens) for seq in range(batch)), per_group)
        prep_plans = _prep_cb_plan(Kt, Vt, heads, conv_on_fpu, pieces, inbox, shared, fold=fold, hold=hold, tail=tail)
        groups_per_kh = ratio // heads
        qk_extra = 2 if shared else 0  # q / k cores per (sequence, key head) and stride position
        # Group first units: heads value heads (each `pieces` units) from value head h0.
        groups = [(seq * value_heads + h0) * pieces for seq in range(seqs) for h0 in range(0, value_heads, heads)]
        strides = [min(per_group, _seq_tile_rows(first // pieces // value_heads, tokens)) for first in groups]
        if PREP_NORM_SPLIT:
            # Norm cores per group: its prep cores but the rows % stride that prep one tile row more; norm units per
            # group: its tile rows times its heads.
            group_rows = [_seq_tile_rows(first // pieces // value_heads, tokens) for first in groups]
            units = max(
                -(-(rows - tail) * heads // (stride - rows % stride)) for rows, stride in zip(group_rows, strides)
            )  # per norm core, at most
            split_plans = _prep_cb_plan(Kt, Vt, heads, conv_on_fpu, pieces, inbox, shared, units, fold, hold, tail)
            if (
                units <= PREP_NORM_MAX_ROWS * heads
                and heads * pieces <= inbox  # one landing counter per (head, piece), on the inbox ids
                and _prep_plan_bytes(split_plans, token_dtype) <= budget
            ):
                norm_split, prep_plans = True, split_plans
    if tail and not (prep is not None and norm_split):
        # The tail needs the norm units' routing: without it, no tail.
        return create_program_descriptor(
            inputs,
            outputs,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
            packed=packed,
            l1_end=l1_end,
            seq_split=seq_split,
            norm_tail=0,
        )
    if split and not (prep is not None and norm_split and fold and not shared and PREP_CLUSTER):
        # No plan the split can run on (its o arrive after the chain, its step is the folded one): the unsplit call.
        return create_program_descriptor(
            inputs,
            outputs,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
            packed=packed,
            l1_end=l1_end,
            seq_split=1,
        )
    if prep is not None:
        for role, plan in zip(("R-PREP state", "R-PREP prep", "R-PREP q / k"), prep_plans):
            if plan:
                check_l1_footprint(role, _plan_bytes(plan, token_dtype), budget, l1_end)
        cluster_picks, rect = None, None
        if PREP_CLUSTER:
            # Each group's state and prep cores as one blob around the first free core (row-major), its state cores at
            # the blob's center: a head block crosses a few links instead of every group's converging on the top rows.
            # PREP_SHARE_QK: the middle group of each key head's blob also takes the key head's q / k cores (after its
            # own prep cores), between the key head's other groups.
            free = list(grid_cores)
            assignment, cluster_picks, qk_pools = [], [], {}
            seq_rows = max(_seq_tile_rows(seq, tokens) for seq in range(batch))
            rect_ok = PREP_RECT and seq_rows <= PREP_RECT_MAX_ROWS and not split
            if rect_ok and not shared:
                rect = _prep_rect_place(device, grid, len(groups), heads * pieces, strides)
            elif rect_ok and PREP_RECT_SHARED and batch == 1 and heads == 1:
                rect = _prep_rect_shared(grid, key_heads, groups_per_kh, pieces, strides[0])
                qk_pools = dict(enumerate(rect[3])) if rect else qk_pools
            if rect is not None:
                cluster_picks = rect[1]
                for first, cores in zip(groups, rect[0]):
                    assignment += [(core, first + u, 1, 1) for u, core in enumerate(cores)]
            for g, (first, stride) in enumerate(() if rect else zip(groups, strides)):
                takes_qk = shared and g % groups_per_kh == groups_per_kh // 2
                seed = free[0]
                blob = sorted(free, key=lambda c: (abs(c.x - seed.x) + abs(c.y - seed.y), c.y, c.x))
                n_st = heads * pieces * (2 if p_first(first) is not None else 1)  # SEQ_SPLIT: + the P half's cores
                blob = blob[: n_st + stride * (1 + (qk_extra if takes_qk else 0))]
                cx, cy = sum(c.x for c in blob) / len(blob), sum(c.y for c in blob) / len(blob)
                blob.sort(key=lambda c: (abs(c.x - cx) + abs(c.y - cy), c.y, c.x))
                assignment += [(core, first + u, 1, 1) for u, core in enumerate(blob[: heads * pieces])]
                if n_st > heads * pieces:
                    assignment += [
                        (core, p_first(first) + u, 1, 1) for u, core in enumerate(blob[heads * pieces : n_st])
                    ]
                cluster_picks.append(blob[n_st : n_st + stride])
                if takes_qk:
                    qk_pools[g // groups_per_kh] = blob[n_st + stride :]
                taken = {(c.x, c.y) for c in blob}
                free = [c for c in free if (c.x, c.y) not in taken]
            assignment.sort(key=lambda entry: entry[1])
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
    k_sender = qk_group - 1 if BCAST_SPLIT_NOCS else 1  # the k injector's index in its group
    qk_channels = []
    if _share_qk(pieces, tokens) and not qk_cores:
        order = [(core.y, core.x) for core, _, _, _ in assignment]
        assert order == sorted(order), "units must be assigned in row-major core order"
        # One round per channel per call: no handshake (each receiver's landing CB is untouched before it) and a
        # flag signal linked behind the payload (a counter would wait a write barrier per rectangle). The
        # injectors' writers send on the writer's NoC (WriterConfigDescriptor: NOC 1, the DRAM-write NoC).
        q_noc = ttnn.NOC.NOC_0 if BCAST_SPLIT_NOCS else ttnn.NOC.NOC_1
        qk_channels = [
            ttnn.Mcast(
                device,
                ttnn.McastConfig(noc=noc, handshake=False, data_ready=ttnn.McastDataReady.Flag),
                all_cores,
                qk_group,
                ttnn.McastFixedSenderConfig(sender_index=sender),
                ttnn.McastCoreOrder.RowMajor,
            )
            for sender, noc in ((0, q_noc), (k_sender, ttnn.NOC.NOC_1))
        ]

    if qk_cores:
        # Group kh = the key head's cores_per_kh pair cores (consecutive in row-major order); its senders are the
        # key head's q core (channel q) and k core (channel k), outside the group. One flag-signalled round per call.
        # QK_CORES_VSPLIT: the q cores send on NOC 0 (their reader and writer swap NoCs). QK_RAW_HANDOFF: a second pair
        # of channels carries the inverse norms (one round per channel per call).
        q_noc = ttnn.NOC.NOC_0 if pieces > 1 and QK_VSPLIT_SPLIT_NOCS else ttnn.NOC.NOC_1
        qk_channels = [
            ttnn.Mcast(
                device,
                ttnn.McastConfig(noc=noc, handshake=False, data_ready=ttnn.McastDataReady.Flag),
                _core_ranges(state_cores),
                cores_per_kh,
                ttnn.McastExplicitFixedSenderConfig([core for core, _, r in qk_cores if r == role]),
                ttnn.McastCoreOrder.RowMajor,
            )
            for role, noc in ((QK_Q_INJECTOR, q_noc), (QK_K_INJECTOR, ttnn.NOC.NOC_1)) * (2 if qk_raw else 1)
        ]
    # SMALL_INPUTS_MCAST: the q core's writer sends the small inputs, the state cores' readers receive them.
    small_channel = None
    if small_mcast:
        small_channel = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=q_noc, handshake=False, data_ready=ttnn.McastDataReady.Flag),
            _core_ranges(state_cores),
            cores_per_kh,
            ttnn.McastExplicitFixedSenderConfig([core for core, _, r in qk_cores if r == QK_Q_INJECTOR]),
            ttnn.McastCoreOrder.RowMajor,
        )

    def qk_role(unit):
        if qk_cores:
            return QK_RECEIVER
        if not qk_channels:
            return QK_OWN
        return {0: QK_Q_INJECTOR, k_sender: QK_K_INJECTOR}.get(unit % qk_group, QK_RECEIVER)

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
    # env GDN_LAYOUT_DUMP=<file>: the core roles of this program, for timeline tools.
    if os.environ.get("GDN_LAYOUT_DUMP"):
        import json

        def _xy(core):
            noc = device.worker_core_from_logical_core(core)
            return [noc.x, noc.y, core.x, core.y]

        with open(os.environ["GDN_LAYOUT_DUMP"], "w") as out:
            json.dump(
                {
                    "batch": batch,
                    "tokens": tokens,
                    "value_heads": value_heads,
                    "key_heads": key_heads,
                    "pieces": pieces,
                    "state": [_xy(c) + [first, count, stride] for c, first, count, stride in assignment],
                    "qk": [_xy(c) + [hk, role] for c, hk, role in qk_cores],
                    "prep": [_xy(c) + [first, pos, stride] for c, first, pos, stride in prep_units],
                    "prep_qk": [_xy(u[0]) + [u[1], u[2], u[3], u[4]] for u in prep_qk_units],
                },
                out,
            )

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
    # The reader writes its share after all of its reads, so only on cores whose states all fit in cb_state_out.
    out_states = min(STATE_OUT_DEPTH, unit_depth) if prep is None else 0

    def reader_share(core, count):
        """Final-state tiles the reader writes (gdn::state_tile_on_reader): R-QKCORE pair cores only."""
        if not qk_cores or count > out_states:
            return 0
        writer_noc = 0 if _noc_swapped(core, unit_depth == 1 and pieces == 1) else 1
        return 1 if STATE_WRITE_NOC is not None and writer_noc != STATE_WRITE_NOC else STATE_SPLIT

    def reader_units(count):
        """Leading units whose final states the reader writes whole."""
        return count - 1 if qk_cores and STATE_READER_EARLY_UNITS and count <= out_states else 0

    state_split = any(reader_share(c, n) or reader_units(n) for c, _, n, _ in assignment)

    def s0_writer(core, count):
        """The writer reads the core's S0 (S0_WRITER): R-QKCORE pair cores whose units' S0 all fit cb_state_in."""
        row = S0_WRITER_ROWS is None or core.y in S0_WRITER_ROWS
        return int(bool(qk_cores) and S0_WRITER and row and count <= min(STATE_IN_DEPTH, unit_depth))

    s0_writer_any = prep is None and any(s0_writer(c, n) for c, _, n, _ in assignment)
    if s0_writer_any and S0_WRITER_AFTER_INPUTS:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=SIGNAL_PAGE_BYTES,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_S0_GO, data_format=ttnn.bfloat16, page_size=SIGNAL_PAGE_BYTES
                    )
                ],
            )
        )
    if state_split:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=SIGNAL_PAGE_BYTES * STATE_OUT_DEPTH,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_STATE_DONE, data_format=ttnn.bfloat16, page_size=SIGNAL_PAGE_BYTES
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
    reader_tensors = (*inputs, conv_state_out, recurrent_state_out)
    # Token tile bytes, state tile bytes.
    geometry_ct = [token_tile_bytes, FP32_TILE_BYTES]
    reader_accessor_ct = [arg for t in reader_tensors for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
    writer_accessor_ct = [
        arg
        for t in (output, recurrent_state_out, *(inputs[i] for i in writer_inputs(small_mcast)))
        for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()
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
        s0_read = S0_READ_TILES
        if s0_read == "auto":
            s0_read = 8 if qk_cores and pieces == 1 and prep is None and core_units <= 2 else None
        reader_ct += [pieces, int(chunked), role, inbox, min(s0_read or s0_tiles, s0_tiles), cs_split]
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
    if cs_in_place and cs_split != 2:
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
        norm_dst = []  # R-PREP: first norm core among the producers, norm cores (tile row i's o: first + i % cores)
        seg_args = []  # SEQ_SPLIT: NoC (x, y) of the other half's pieces of the head, then of the next segment's piece
        if prep is not None:
            is_p = split and first_unit // pieces // value_heads >= segs
            own = first_unit - (segs - 1) * value_heads * pieces if is_p else first_unit  # the unit its rows are of
            group_first = own - own % (heads * pieces)
            producers = [len(group_producers[group_first]) // 2] + group_producers[group_first]
            seq_rows = _seq_tile_rows(own // pieces // value_heads, tokens)
            extra = seq_rows % producers[0] if norm_split else 0
            norm_dst = [extra, producers[0] - extra]  # PREP_NORM_SPLIT: norm unit u goes to first + u % cores
            if split:
                other = own if is_p else p_first(first_unit)
                other = first_unit if other is None else other - other % pieces
                nxt = first_unit + value_heads * pieces
                nxt = nxt if not is_p and nxt < segs * value_heads * pieces else first_unit
                seg_args = [c for u in range(pieces) for c in unit_noc[other + u]] + list(unit_noc[nxt])
        share = [reader_units(count), reader_share(core, count)] if state_split else []
        share += [s0_writer(core, count)] if s0_writer_any else []
        rts[0][core.x, core.y] = (
            [first_unit, count, tokens, role, zero_slots] + rows + producers + seg_args + cs_args(core) + share
        )
        # R-VSPLIT: NoC coordinates of the pair's piece cores, in piece order.
        group_first = first_unit - first_unit % pieces
        peers = [c for s in range(pieces) for c in unit_noc[group_first + s]] if pieces > 1 else []
        rts[1][core.x, core.y] = (
            [first_unit, count, tokens, role, zero_slots, zero_slot_bytes]
            + rows
            + peers
            + producers
            + norm_dst
            + seg_args
            + share
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
            seq_rows = _seq_tile_rows(seq, tokens)
            my_rows = len(range(q, seq_rows, stride))
            group_cols = _prep_group_cols(prep_cols) if shared else _window_group_cols(Kt, heads * Vt, 1, True)
            zero_slots, zero_slot_bytes = window_zeroing(prep_cols, group_cols, my_rows)
            rows = [q, stride, 1]
            # PREP_NORM_SPLIT: the norm units (tile row * heads + head) it normalizes: offset, stride (none on the first
            # rows % stride cores), and their count.
            extra = seq_rows % stride if norm_split else 0
            norm_units = (seq_rows - tail) * heads  # the tail rows' units stay on the state cores
            norm = [q - extra if q >= extra else norm_units, stride - extra]
            norm_rows = len(range(norm[0], norm_units, norm[1]))
            # PREP_NORM_MID: unit m (before row m // mid + 1's heads) must not wait for a row this core still holds.
            mid = PREP_NORM_MID if norm_split and q >= inbox and not split else 0  # SEQ_SPLIT: o after the chain
            mid_rows = [((norm[0] + m * norm[1]) // heads, q + (m // mid) * stride) for m in range(mid * (my_rows - 1))]
            if any(m >= norm_rows or unit_row >= held_row for m, (unit_row, held_row) in enumerate(mid_rows)):
                mid = 0
            consumers = [c for u in range(heads * pieces) for c in unit_noc[group_first + u]]
            p_half = p_first(group_first)  # SEQ_SPLIT: the P half's cores take the head blocks too
            if p_half is not None:
                consumers += [c for u in range(heads * pieces) for c in unit_noc[p_half + u]]
            if split:
                consumers = [len(consumers) // 2] + consumers
            # A head block goes to the head's pieces: one multicast per row run.
            for j in range(heads):
                first = group_first + j * pieces
                cores = unit_cores[first : first + pieces]
                if p_half is not None:
                    cores = cores + unit_cores[p_half + j * pieces : p_half + (j + 1) * pieces]
                consumers += _mcast_rects(device, cores)
            qk_src = prep_qk_src.get((core.x, core.y), [])
            rts[0][core.x, core.y] = (
                [group_first, 1, tokens, prep_qk_role, zero_slots] + rows + norm + [mid] + qk_src + cs_args(core)
            )
            rts[1][core.x, core.y] = [group_first, 1, tokens, prep_qk_role, zero_slots, zero_slot_bytes] + rows + norm
            rts[1][core.x, core.y] += consumers
            rts[2][core.x, core.y] = [group_first, 1, tokens, v_eps_bits, prep_qk_role] + rows + [norm_rows, mid]
            dsts = [unit_noc[group_first + u] for u in range(heads * pieces)]
            prep_send_noc0[core.x, core.y] = rect[2][core.x, core.y] if rect else send_noc0(core, dsts)
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
            prep_send_noc0[core.x, core.y] = (
                rect[2][core.x, core.y] if rect else send_noc0(core, [(d.x, d.y) for d in dst_noc])
            )
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
            if cs_split == 2:
                v_hi = v_lo
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
        if idx == CB_IN_V and fold:
            continue  # v is a matmul operand of the folded step
        fp32_cb = idx not in TOKEN_CBS or token_dtype == ttnn.float32
        modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32 if fp32_cb else ttnn.UnpackToDestMode.Default
    compute_kernel_config.unpack_to_dest_mode = modes
    if DST_FULL_SYNC:
        compute_kernel_config.dst_full_sync_en = True

    # SMALL_INPUTS_MCAST's defines only where it runs (gated_delta_net_qk_rows.hpp defaults them off).
    small_defines = [("GDN_SMALL_MCAST", "1")] + [(f"GDN_{n}", str(globals()[n])) for n in SMALL_MCAST_NAMES]
    early_s0 = EARLY_S0
    if early_s0 == "auto":
        early_s0 = int(bool(qk_cores) and pieces == 1 and prep is None and core_units == 2)
    # "auto": R-VSPLIT pieces, and pair cores whose several units run without UNIT_LAG.
    fused_vector = pieces > 1 or (prep is None and bool(qk_cores) and unit_depth > 1 and not unit_lag(unit_depth))
    defines = (
        _kernel_defines()
        + (small_defines if small_mcast else [])
        + ([("GDN_QK_Q_READER_PCT", str(QK_Q_READER_PCT))] if qk_cores and QK_Q_READER_PCT != 50 else [])
        + [
            ("GDN_HOLD_QK", str(int(bool(qk_cores) and SHARE_QK_FINISH))),
            ("GDN_QK_RAW", str(int(qk_raw))),
            ("GDN_UNIT_LAG", str(int(prep is None and unit_lag(unit_depth)))),
            ("GDN_WRITER_ASYNC", str(int(WRITER_ASYNC) * int(prep is None and unit_lag(unit_depth)))),
            ("GDN_STATE_STAGGER", str(STATE_STAGGER if qk_cores else 0)),
            ("GDN_STATE_SPLIT", str(int(state_split))),
            ("GDN_S0_WRITER", str(int(s0_writer_any))),
            ("GDN_S0_WRITER_AFTER_INPUTS", str(int(S0_WRITER_AFTER_INPUTS))),
            ("GDN_CS_IN_PLACE", str(int(cs_in_place))),
            ("GDN_PROFILE_ZONES", str(0 if _noc_tracing() else int(PROFILE_ZONES))),
            *([("GDN_ZONE_SKIP", str(int(PROFILE_ZONE_SKIP)))] if PROFILE_ZONE_SKIP else []),
            ("GDN_TOKEN_VECTOR_FUSED", str(int(fused_vector if TOKEN_VECTOR_FUSED == "auto" else TOKEN_VECTOR_FUSED))),
            ("GDN_DECAY_ONE_PASS", str(int(DECAY_ONE_PASS))),
            ("GDN_EARLY_S0", str(int(early_s0))),
            ("GDN_RMS_ONE_COPY", str(int(RMS_ONE_COPY and prep is None and unit_lag(unit_depth)))),
            ("GDN_PREP_NORM_SPLIT", str(int(norm_split))),
            ("GDN_PREP_HEAD_UP", str(int(PREP_HEAD_UP))),
            ("GDN_RSQRT_COL0", str(int(NORM_RSQRT_COL0))),
            ("GDN_GATES_LEFT", str(int(PREP_GATES_LEFT and value_heads <= 16))),
            ("GDN_VEC_COL0", str(int(PREP_VEC_COL0))),
            ("GDN_SILU_BF16", str(int(PREP_SILU_BF16) if prep is not None and conv_on_fpu else 0)),
            ("GDN_SILU_MID", str(int(PREP_SILU_MID) if prep is not None and conv_on_fpu else 0)),
            ("GDN_STATE_FUSED_RHS", str(int(prep is not None and STATE_FUSED_RHS))),
            ("GDN_PREP_SCALE_FPU", str(int(PREP_SCALE_FPU) if prep is not None else 0)),
            ("GDN_FPU_UNBIAS_BITS", f"{PREP_FPU_UNBIAS_BITS:#x}u"),
            ("GDN_PREP_M_BETA", str(int(prep is not None and PREP_M_BETA))),
            ("GDN_STATE_HOLD", str(int(prep is not None and STATE_HOLD and STATE_FUSED_RHS and not DST_FULL_SYNC))),
            ("GDN_M_BETA_UNBIAS_Q", str(int(STATE_FOLD_M_UNBIAS_Q if fold else PREP_M_BETA_UNBIAS_Q))),
            ("GDN_STATE_FOLD", str(fold)),
            ("GDN_SEQ_SPLIT", str(segs if split else 1)),
            ("GDN_NORM_TAIL", str(tail)),
            ("GDN_SEG_ROW_DEPTH", str(SEG_ROW_DEPTH)),
            ("GDN_GRANT_DEPTH", str(min([inbox] + strides) if split else inbox)),
            ("GDN_FOLD_UNBIAS_BITS", f"{STATE_FOLD_UNBIAS_BITS:#x}u"),
        ]
    )
    if packed:
        packed_tiles = -(-int(mixed_qkv.shape[2]) // TILE)
        defines += [("GDN_PACKED_TILES", str(packed_tiles)), ("GDN_PACKED_GATE_TILE", str(Ct + value_heads * Vt))]
    program_values = {}
    if prep is not None and (STATE_FUSED_RHS or PREP_M_BETA):
        program_values["GDN_CHUNK_COEF_PAGES"] = CHUNK_COEF_PAGES + 1
    defines = [(name, str(program_values.get(name, value))) for name, value in defines]
    # R-PREP: offsets of the other role's remotely written buffers from its CB region base (see STATE_REMOTE_CBS).
    remote = {}
    if prep is not None:
        prep_remote = PREP_REMOTE_CBS + ((CB_KQ_ROWS,) if shared else ())
        remote = _remote_offsets(prep_plans[0], STATE_REMOTE_CBS) | _remote_offsets(prep_plans[1], prep_remote)
    defines += [(f"GDN_OFFSET_{name}", str(remote.get(idx, 0))) for name, idx in REMOTE_CB_NAMES.items()]
    kernels = []
    writers = []
    readers = []
    # SMALL_INPUTS_MCAST: the kernels on its channel (the q / k cores' writers send, the state cores' readers receive).
    small_kernels = []

    def runtime_args(args, cores):
        rt = ttnn.RuntimeArgs()
        for core in cores:
            rt[core.x][core.y] = args[core.x, core.y]
        return rt

    kernel_groups = []  # (role, cores, swap the reader / writer NoCs)
    for role, (cores, *_) in roles.items():
        if role == ROLE_QK and pieces > 1 and QK_VSPLIT_SPLIT_NOCS:
            # QK_CORES_VSPLIT: the q cores send on NOC 0.
            q_cores = {(core.x, core.y) for core, _, r in qk_cores if r == QK_Q_INJECTOR}
            kernel_groups += [
                (role, [c for c in cores if (c.x, c.y) in q_cores], True),
                (role, [c for c in cores if (c.x, c.y) not in q_cores], False),
            ]
        elif role == ROLE_LOCAL and qk_cores and (SWAP_NOC_ROWS or SWAP_NOC_ROWS_ONE_UNIT or SWAP_NOC_CORES):
            # Decode R-QKCORE: every pair core reads its S0 at once. NOC 0 brings read responses south, around the
            # torus for the top rows, which then lose to the lower rows' traffic; NOC 1 brings them north. The pair
            # cores in the top SWAP_NOC_ROWS rows read on NOC 1 and write on NOC 0.
            # S0_WRITER_KEEP_NOC: on the rows where S0 is the writer's, the writer takes the NoC the reader would have.
            def swapped(c):
                s0_by_writer = s0_writer_any and S0_WRITER_KEEP_NOC and roles[role][1][c.x, c.y][-1]
                return _noc_swapped(c, unit_depth == 1 and pieces == 1) != bool(s0_by_writer)

            kernel_groups += [(role, [c for c in cores if swapped(c)], True)]
            kernel_groups += [(role, [c for c in cores if not swapped(c)], False)]
        elif role in (ROLE_PREP, ROLE_PREP_QK) and PREP_SEND_NOC == "auto":
            # Each prep core writes on the NoC with the shorter paths to its state cores (NOC 0 routes toward higher x
            # and y, NOC 1 toward lower, both around the torus); the reader takes the other one.
            on_noc0 = [c for c in cores if prep_send_noc0[c.x, c.y]]
            kernel_groups += [(role, on_noc0, True), (role, [c for c in cores if not prep_send_noc0[c.x, c.y]], False)]
        elif role == ROLE_LOCAL and qk_channels and BCAST_SPLIT_NOCS:
            # The q injectors send on NOC 0 from the writer: their reader and writer swap NoCs.
            q_inj = [c for c in cores if roles[role][3][c.x, c.y][4] == QK_Q_INJECTOR]
            rest = [c for c in cores if roles[role][3][c.x, c.y][4] != QK_Q_INJECTOR]
            kernel_groups += [(role, q_inj, True), (role, rest, False)]
        elif role == ROLE_LOCAL and qk_channels and SPLIT_BCAST_GROUPS:
            # R-BCAST: one kernel group per qk role (q injectors, k injectors, receivers), each compiling its role in.
            core_role = {(c.x, c.y): roles[role][3][c.x, c.y][4] for c in cores}
            for qk in (QK_Q_INJECTOR, QK_K_INJECTOR, QK_RECEIVER):
                kernel_groups.append((role, [c for c in cores if core_role[c.x, c.y] == qk], False))
        else:
            kernel_groups.append((role, cores, False))
    for role, cores, swap_nocs in kernel_groups:
        if not cores:
            continue
        _, reader_rt, writer_rt, compute_rt = roles[role]
        ranges = _merged_ranges(cores)
        reader_rt, writer_rt, compute_rt = (runtime_args(rt, cores) for rt in (reader_rt, writer_rt, compute_rt))
        reader_ct, writer_ct, compute_ct = role_ct(role)
        # Names the role in the JIT cache (named_ct_arg_map_generated.h) for per-role binary size reports; unread.
        role_label = [("gdn_role", role)]
        group_qk_roles = {roles[role][3][c.x, c.y][4] for c in cores}  # compute rt arg 4: the qk role
        group_qk_role = group_qk_roles.pop() if len(group_qk_roles) == 1 else QK_PER_CORE
        window = STATE_WRITE_WINDOW if qk_cores and prep is None and core_units >= STATE_WRITE_WINDOW_UNITS else 0
        if isinstance(window, tuple):  # (writers on NOC 1, writers on NOC 0)
            window = window[int(swap_nocs)]
        group_defines = defines + [("GDN_GROUP_QK_ROLE", str(group_qk_role)), ("GDN_STATE_WRITE_WINDOW", str(window))]
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
            opt_level=_dataflow_opt_level(),
            core_ranges=ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            common_runtime_args=writer_addresses(inputs, outputs, small_mcast),
            named_compile_time_args=role_label,
            defines=group_defines,
            config=writer_config,
        )
        writers.append(writer)
        reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_reader.cpp"),
            opt_level=_dataflow_opt_level(),
            core_ranges=ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            common_runtime_args=reader_addresses(inputs, outputs),
            named_compile_time_args=role_label,
            defines=group_defines,
            config=reader_config,
        )
        readers.append(reader)
        small_kernels.append(writer if role == ROLE_QK else reader)
        kernels += [
            reader,
            writer,
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / COMPUTE_SOURCES[role]),
                core_ranges=ranges,
                compile_time_args=compute_ct,
                named_compile_time_args=role_label,
                defines=group_defines + [("INP_FLOAT32", "1")],
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
    channels = list(qk_channels) + [None] * (4 - len(qk_channels))
    for prefix, channel in zip(("qkq", "qkk", "qkqr", "qkkr"), channels):
        if channel is None:
            for writer in writers:
                ttnn.attach_absent_mcast(writer, prefix)
        else:
            channel.attach(descriptor, prefix, writers, next_sem)
            next_sem = channel.next_semaphore_id()
    if small_channel is not None:
        small_channel.attach(descriptor, "qks", small_kernels, next_sem)
        next_sem = small_channel.next_semaphore_id()
        for kernel in readers + writers:
            if all(kernel is not k for k in small_kernels):
                ttnn.attach_absent_mcast(kernel, "qks")
    descriptor.kernels = kernels
    return descriptor


# ---- Program cache (host cost per call) ----
def _noc_swapped(core, one_unit):
    """R-QKCORE pair core: reads on NOC 1 and writes on NOC 0 (SWAP_NOC_ROWS, SWAP_NOC_CORES)."""
    rows = SWAP_NOC_ROWS_ONE_UNIT if one_unit and SWAP_NOC_ROWS_ONE_UNIT is not None else SWAP_NOC_ROWS
    flip = any(x0 <= core.x <= x1 and y0 <= core.y <= y1 for x0, y0, x1, y1 in SWAP_NOC_CORES)
    return (core.y < rows) != flip


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


def cached_program_descriptor(key, inputs, outputs, *, epsilon, compute_kernel_config, packed=False, l1_end=None):
    entry = _program_cache.get(key)
    if entry is None:
        descriptor = create_program_descriptor(
            inputs,
            outputs,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
            packed=packed,
            l1_end=l1_end,
        )
        # The structural hash leaves out per-core argument values and counts; the key covers them.
        descriptor.custom_program_hash = hash((ttnn.compute_program_descriptor_hash(descriptor), key)) % (1 << 64)
        kernels = descriptor.kernels  # references into the descriptor: reader, writer, compute per core group
        small_mcast = len(kernels[1].common_runtime_args) > 2 + len(WRITER_INPUTS)
        entry = (descriptor, kernels[0::3], kernels[1::3], small_mcast)
        _program_cache[key] = entry
        while len(_program_cache) > PROGRAM_CACHE_SIZE:
            _program_cache.popitem(last=False)
        return descriptor
    _program_cache.move_to_end(key)
    descriptor, readers, writers, small_mcast = entry
    reader_args, writer_args = reader_addresses(inputs, outputs), writer_addresses(inputs, outputs, small_mcast)
    for kernel in readers:
        kernel.common_runtime_args = reader_args
    for kernel in writers:
        kernel.common_runtime_args = writer_args
    return descriptor
