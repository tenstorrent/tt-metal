import os

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side tunables for groupnorm_sc_N_1_HW_C (op_design.md "Parameters").

Every knob is read by attribute on this module at call time (``config.NAME``)
so tests and perf sweeps can ``monkeypatch.setattr(config, ...)``. Never
``from config import NAME`` — that snapshots the value.
"""

# Total per-core circular-buffer budget (bytes) the split search / regime
# selector must fit under. Blackhole has ~1.46 MB above the L1 allocator base (111 616 B) on an
# interleaved program; Refinement 5 measured 1_440_000 on the 1.2 MB/core VAE cell TILE (1,1,262144,256):
# it becomes resident at K = 4, Q = 1 (149 four-tile chunks x 3 passes over L1) and runs 1070 us against
# 1072 us for the two-pass stream at this default — a wash — so the default keeps the L1 headroom.
L1_CB_BUDGET_BYTES = 1_000_000

# Target tiles per compute chunk (Q = max(1, CHUNK_TILES_TARGET // K)).
CHUNK_TILES_TARGET = 32

# Cap on channel tiles a single core owns (bounds the per-core stat set).
MAX_CORE_C_TILES = 16

# Cap on group-slot tiles (Ng = ceil(G/32)); above it the dense membership
# matrix would grow with G (regime `sparse_membership`, deferred).
MAX_GROUP_TILES = 4

# Depth (in blocks) of every streaming CB: input tiles/sticks, output tiles/sticks.
STREAM_DEPTH = 2

# Split-search tiebreak: "hw_first" (maximize hw_splits) or "c_first".
SPLIT_ORDER = "hw_first"

# Split search: prefer a split whose per-core assignment is L1-resident (input
# crosses DRAM once) over a non-resident one with fewer tiles per core (input
# crosses DRAM three times in the streaming regime), shrinking the chunk
# (Q, down to 1 tile-row) when that is what makes the assignment fit.
# Measured (verifier, BH 110 cores, TILE (1,1,65536,512)): streaming K=16/Q=2
# 734 us -> resident K=4/Q=2 508 us. TILE only: on ROW_MAJOR the stick-slice
# width K dominates (same shape RM: K=16 streaming 771 us beat K=4 resident
# 916 us), so RM keeps the widest K and takes residency only when it comes free.
SPLIT_PREFER_RESIDENT = True
SPLIT_PREFER_RESIDENT_RM = False

# Split search: accept up to this fraction more max-tiles-per-core in exchange
# for a wider per-core column block K, per input layout. ROW_MAJOR reads
# K*64 B stick slices, so a narrow K is latency-bound (measured (1,1,16384,320)
# RM: K=1 408 us, K=2 245 us, K=5 150 us, K=10 140 us); TILE moves whole pages
# and the strict min-tiles pick measured best (K=1 88 us vs K=10 122 us).
SPLIT_COST_TOLERANCE_TILE = 0.0
SPLIT_COST_TOLERANCE_RM = 0.1

# ROW_MAJOR: the stick-width benefit saturates once slices reach K*64 B with
# K >= this target (640 B). Among admitted candidates, RM widens K up to the
# target; candidates at or above it compete on the TILE objective (fewest
# tiles, then narrower K). Measured: (1,1,16384,960) RM K=15 401 us vs K=10
# 318 us; (1,1,4096,1920) RM K=15 200 us vs K=12 185 us.
SPLIT_RM_STICK_K_TARGET = 10

# Test knob: pin c_splits (must divide Ct and keep K <= MAX_CORE_C_TILES);
# None = search.
FORCE_C_SPLITS = None

# Test knob: force the streaming regime even when the assignment is L1-resident.
FORCE_STREAMING = False

# Refinement 3 (perf): consume a ROW_MAJOR block shard IN PLACE as a row-major block whose width is
# lcm(shard_w, 32) elements — a [2048, 40] bf16 shard is byte-identical to a [512, 160] block, so the
# tilize reads the shard zero-copy (K' = 5 tiles, no pad lanes) and the untilize packs straight into
# the output shard; the per-stick L1 -> L1 staging / write-back loops disappear. Channel mapping
# becomes periodic (lane % shard_w) in the membership and gamma/beta rows. Applies only where the
# view is exact (host gate `_rm_direct_view`); every other RM shard keeps the staged path.
# False = the Refinement-1 staged path everywhere (the measurable baseline).
RM_SHARD_DIRECT_VIEW = True

# Refinement 3: a block-sharded program's CBs share L1 with the shard buffers themselves (allocated
# top-down), so its real CB ceiling is `lowest shard address - allocator base`, not the flat budget —
# (1,1,16384,960) RM [2048,120] measured 972 928 B of CBs against a 969 728 B ceiling and clashed.
# The sharded branch fits against min(L1_CB_BUDGET_BYTES, ceiling - this margin); the margin absorbs
# CB alignment rounding.
L1_CB_SAFETY_MARGIN_BYTES = 4096

# Refinement 5 (perf): two-pass streaming statistics. The Phase-0 streaming regime read the input once per
# pass (three passes: sum -> mean, centered squares -> variance, apply): 4 tensor volumes of DRAM traffic
# for a 2-volume problem on the DRAM-bound VAE shapes. With this knob a streaming program computes BOTH
# statistics from ONE read of each chunk (pass A: raw column sums S and centered-square column sums
# U = sum (x - s)^2 against a per-channel shift s = the column means of the chunk's first tile-row, taken
# while the chunk is resident in L1), combines them per channel once the group mean m is known —
# sum (x - m)^2 = U + n (s - m)(2 S / n - m - s), every large term a sum of squares of centered values (never
# E[x^2] - mean^2) — and applies in pass B: 3 volumes instead of 4. Compiled only into interleaved streaming
# programs without a ragged tile-row (hw_mask); resident and sharded programs are untouched.
# False = the Phase-0 three-pass streaming schedule (the measurable baseline).
STREAMING_TWO_PASS = True

# "Temporal" channel rounds for interleaved inputs that do not fit L1 as a whole: run the op once per
# contiguous slice of whole groups, each slice sized so every core's block is L1-resident (input read once,
# output written once) instead of one streaming pass (input read twice with STREAMING_TWO_PASS, three times
# on the hw_mask path). False = the streaming schedule. Slices are separate program launches.
TEMPORAL_ROUNDS = (
    os.environ.get("GN_TEMPORAL_ROUNDS", "0") == "1"
)  # measured slower on the VAE shapes (worklog); opt-in
# Extra L1 the planner reserves for the fixed CBs when sizing a round (bytes).
TEMPORAL_FIXED_RESERVE_BYTES = 300_000
