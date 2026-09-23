# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side tunables for groupnorm_sc_N_1_HW_C (op_design.md "Parameters").

Every knob is read by attribute on this module at call time (``config.NAME``)
so tests and perf sweeps can ``monkeypatch.setattr(config, ...)``. Never
``from config import NAME`` — that snapshots the value.
"""

# Total per-core circular-buffer budget (bytes) the split search / regime selector must fit under.
# Blackhole has ~1.46 MB of L1 above the allocator base; the rest is left as headroom.
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

# Split search: accept up to this fraction more max-tiles-per-core in exchange for a wider
# per-core column block K, per input layout. ROW_MAJOR reads K*64 B stick slices, so a narrow K
# is latency-bound; TILE moves whole pages and the strict min-tiles pick measured best.
SPLIT_COST_TOLERANCE_TILE = 0.0
SPLIT_COST_TOLERANCE_RM = 0.1

# ROW_MAJOR: the stick-width benefit saturates once slices reach K*64 B with K >= this target
# (640 B). Among admitted candidates, RM widens K up to the target; candidates at or above it
# compete on the TILE objective (fewest tiles, then narrower K).
SPLIT_RM_STICK_K_TARGET = 10

# Test knob: pin c_splits (must divide Ct and keep K <= MAX_CORE_C_TILES); None = search.
FORCE_C_SPLITS = None

# Test knob: force the streaming regime even when the assignment is L1-resident.
FORCE_STREAMING = False

# A block-sharded program's CBs share L1 with the shard buffers (allocated top-down), so its CB
# ceiling is `lowest shard address - allocator base`; this margin absorbs CB alignment rounding.
L1_CB_SAFETY_MARGIN_BYTES = 4096

# Two-pass streaming statistics: a streaming program computes the raw column sums S and the
# shift-centered square sums U from ONE read of each chunk, combines them into the centered
# variance once the group mean is known, and applies in a second pass (3 tensor volumes of DRAM
# traffic instead of 4). Compiled only into interleaved streaming programs without a ragged
# tile-row (hw_mask). False = the three-pass schedule (sum, centered squares, apply).
STREAMING_TWO_PASS = True
