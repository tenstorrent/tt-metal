# Operation Requirements: split

## Definition
- **Formula**: `chunk[c][..., h', ...] = input[..., c*split_size + h', ...]` along `dim`, for
  `0 <= h' < min(split_size, shape[dim] - c*split_size)`, `c = 0 .. ceil(shape[dim]/split_size) - 1`; every other
  index unchanged. Pure data movement: every element moved unchanged (spec R1), numeric requirement `exact`.
- **PyTorch Reference**: `list(torch.split(input_tensor, split_size, dim=dim))`
- **Import Path**: `from ttnn.operations.split import split`
- **Function Signature**: `split(input_tensor: ttnn.Tensor, split_size: int, dim: int = 0) -> list` (a list of
  `ttnn.Tensor`, one per chunk, each allocated interleaved in the input's buffer, spec R5)
- **Specification**: `eval/specs/split/spec.py` (governs). Target: Quasar on ttsim (Metal 2.0 `ttnn.ProgramSpec`
  only; perf gates are structural counts, `.claude/references/backends/ttsim.md` *Performance measurement*).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf on this backend**: ttsim has no clock. A `**Type**: perf` refinement is gated on the structural count it names (before → done value), read off the op by a test, with the time gain stated as unconfirmed until a timed backend measures it. Never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED memory_config / sharding**: DRAM, L1 / interleaved
- **SUPPORTED shape-derived axes**: dim = -2 only; chunks = 2; last_chunk = full; rank = 3; alignment = all three
- **Regimes built**: `page_remap` (every output page is one input page: RM cut along a page-row dim; TILE cut
  before H or at a tile multiple) and `tile_row_shift` (TILE cut along H inside a tile: scratchpad window +
  self-aimed row-run assembly). Deferred: `tile_col_shift`, `rm_substick` (both `dim = -1`).
- **Cores**: full grid, `split_work_to_cores(row_wise=True)` over a contiguous walk; thread plan (R, C, W) = (1, 0, 1)
  (data-movement bound, no compute kernel)
- **Compute config**: none (movement op; the spec gives it no compute config)
- **Golden baseline**: 36 / 36 in-phase cells pass (36 supported_pass, 228 xfail_expected, 0 loud; 3 / 3 rejections)

### [ ] Refinement 1 — Cut along W: the `tile_col_shift` and `rm_substick` regimes

**Goal**: add `-1` to `SUPPORTED["dim"]` by building the two regimes the design deferred, in-kernel:
- `rm_substick` (`layout == ROW_MAJOR and dim == -1`): an output stick is the byte range `[w0*elem, (w0+w)*elem)` of
  an input stick, unaligned in general. Reader reads whole input sticks (whole aligned pages; no sub-page DRAM
  reads) into a scratchpad window and realigns each chunk's segment into a `cb_output_pages` entry (explicit-sync
  producer, as `tile_row_shift`); writer unchanged (one implicit-sync write per entry, the entry being one chunk
  page). Note one input stick feeds `n_chunks` output pages: read it once, emit `n_chunks` entries.
- `tile_col_shift` (`layout == TILE and dim == -1 and split_size % 32 != 0`): the mirror of `tile_row_shift`: an
  output tile takes its columns from up to two horizontally adjacent input tiles; walk row-major `(o, th, c, j)`;
  window of the block's input tiles. Column shifts are element-granular (`w0 % 32` elements), which for bfloat16 is
  16-B aligned only when `w0 % 8 == 0`; assemble through RISC-V copies on the uncached L1 alias where a
  self-aimed NoC copy would be misaligned (design, Regimes, `tile_col_shift` row).

TILE `dim = -1` at a tile multiple already runs through `page_remap` (the verifier fixed its geometry branch in
`derive_geometry`; 32 widened-SUPPORTED golden cells pass), so it needs nothing here. With R1 alone, the golden
cells that reach these regimes inside SUPPORTED are few (rank 3, 2 chunks, bfloat16: `(1,32,128) s64 dim -1`), so
the regimes are proven below `validate()` (see Done when).

**Implementation skill**: /memory-layouts

**Verifier notes**: hardest generality work, so it goes first (scheme-change on the `within-page` axis per the
Blocking Model: a new reader topology, not a knob). The axis is `dim`, not `layout`: `/memory-layouts` is pointed
at for its in-kernel RM / tile data-access patterns and the non-aligned rule, not for a new `Layout` value.
Build both regimes for **both element sizes** (2 B bfloat16, 4 B float32: Refinement 2 claims float32 over every
regime) and for any rank and chunk count 1–4 (Refinement 2 claims those too); the walk helpers in
`kernels/split_walk.hpp` already take a general `[outer, cut, inner]` form. Build them performantly, not
correct-only (Refinement 3 optimizes the same reader pattern): spread over the full grid, a block of reads per DRAM
barrier (the Phase 0 `tile_row_shift` reader was fixed for exactly this: blocks span tile columns / sticks; carry
that over), each input page read once per core range. Keep every NoC transfer of an explicit-sync kernel tagged
(`{.trid = 0}`) and keep the sticky-tag hand-back; rerun `test_after_another_program_and_before_tt_metal` with a
`dim = -1` call in each new regime. No `ttnn.to_layout` / `ttnn.untilize` wrapper: one dispatch per call, and the
regimes are the kernel capability.

**Done when**: `-1` is in `SUPPORTED["dim"]`; a design test runs **every** `dim = -1` entry of the spec's INPUTS
(all ranks and chunk counts, both layouts, DRAM and L1, bfloat16 and float32) through `_dispatch` below
`validate()` and compares every chunk bit-exactly with `torch.split`; the golden suite has 0 `supported_fail`,
0 `xpass_drift`, 0 `xfail_wrong_mode`; the regime-selection test names both new regimes; no
`NotImplementedError("regime … is not built yet")` remains reachable for a TARGET cell.

### [ ] Refinement 2 — Claim the rest of TARGET: float32, ranks 2 and 4, 1/3/4 chunks, a smaller last chunk, dims -3 and -4

**Goal**: widen SUPPORTED to TARGET on the axes the built regimes already handle:
- add `ttnn.float32` to `SUPPORTED["dtype"]` (R1 gets its check: a float32 value does not survive a bfloat16
  buffer, so the ring and window keep the input's format; `data_format_metadata = input.dtype` already does);
- add `2` and `4` to `SUPPORTED["rank"]`;
- add `1`, `3` and `4` to `SUPPORTED["chunks"]`;
- add `smaller` to `SUPPORTED["last_chunk"]`;
- add `-3` and `-4` to `SUPPORTED["dim"]` (TILE and RM through `page_remap`: whole pages or whole images move).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: the verifier ran the golden suite with SUPPORTED temporarily widened to TARGET (Phase 0
reader, after the verification fixes): 211 / 264 cells pass. **Every** failure was `dim = -1` in a deferred
regime (44 `rm_substick`, 12 `tile_col_shift`); every float32, rank-2/4, 1/3/4-chunk, smaller-last-chunk and
dim -3/-4 cell outside those passed bit-exact. So this refinement is expected to need **no kernel change**: it is
the SUPPORTED edit plus the evidence. Only the dtype part of `/numeric-formats-metal` applies: split is a
movement op with no compute config (spec DEFINITION) — do **not** add a `compute_kernel_config` parameter, and no
compute kernel exists to carry intermediate-CB formats. Comes after Refinement 1 (hardest first); because R1
builds its regimes for float32 and every rank / chunk count already, this refinement then claims them across all
regimes at once. If R1 has shipped `[~]` with a regime missing, this refinement still claims everything it can and
the golden run shows the remaining `dim = -1` cells as `xfail_expected` only if `-1` is not yet in
`SUPPORTED["dim"]`. Keep the `MAX_CHUNKS = 4` writer cap (TARGET's `chunks` ends at 4).

**Done when**: SUPPORTED equals TARGET on `dtype`, `rank`, `chunks`, `last_chunk` and `dim` (with `-1` from
Refinement 1); the golden suite has 0 `supported_fail`, 0 `xpass_drift`, 0 `xfail_wrong_mode`, and
`xfail_expected` is 0; the precision baseline (`test_split_precision_baseline.py`) gains float32 cases, still
bit-exact.

### [ ] Refinement 3 — Read each input tile once and batch the assembly on the traced YOLO tile-layout calls

**Type**: perf

**Goal**: no LOOSE_CASE is perf-flagged, so the region is the model's own calls in tile layout, the
`tile_row_shift` regime: `[1, 4, 400] s2 dim 1` TILE in L1 (the traced small-grid call) and `[1, 4, 8400] s2 dim 1`
TILE in DRAM (the 640x640 call's shape in tile layout). Three levers, all knob-turns on the surface the design
exposed (catalog `ttnn/ttnn/operations/examples/master.md`: `double_buffer`, `width_split` / grid occupancy,
`tile_reorder`):
1. **Core assignment in whole tile columns** for `tile_row_shift` (deal `column_tiles`-sized units, or
   column-multiples, with `split_work_to_cores`), so no core range starts mid-column and no input tile is read
   twice. This is the ledger's deferred cheapest-traffic split; its deferral reason was occupancy only, and on the
   traced `[1,4,400]` the input crosses its tier twice (26 tile reads for 13 tiles). Engaging fewer cores (13
   instead of 26 there) is the accepted price; state it.
2. **Block-batched assembly**: today the reader reserves 1 ring entry, issues ≤ 8 self-aimed copies, barriers and
   pushes 1, per output tile. Reserve the block's entries, issue every copy of the block, barrier once, push the
   block. Needs `cb_output_pages` deep enough for a block on the explicit-producer side (turn `RING_DEPTH` for
   this regime; capacity stays derived from `BLOCK_H_TILES`, one source).
3. **Co-tune `BLOCK_H_TILES`** (8 → 16 or the core's range) against the window (`2 × BLOCK_H_TILES` slots) and the
   ring depth from lever 2, so `[1,4,8400]` (16–17 output tiles per core) takes one or two DRAM barriers per core.

No SUPPORTED change.

**Done when**: a test reads the structural counts off the op (its host split + a kernel-side counter or an
equivalent instrumented probe, not a re-derivation in Python alone) and they reach: `[1,4,400]` TILE input tile
reads 26 → 13 (= minimum); `[1,4,8400]` TILE input tile reads 270 → 263 (= minimum), DRAM read barriers 71 → ≤ 32
(≤ 1 per engaged core), local assembly barriers 526 → ≤ the DRAM-barrier count; the golden suite and every
existing test stay green and bit-exact; no regression in these counts across the guard set (one representative
per kernel path × layout × placement: `page_remap` RM DRAM `(1,4,8400) s2`, `page_remap` TILE L1
`(4,128,512) s64`, `tile_row_shift` TILE DRAM `(1,34,4096) s17`, and the R1 regimes on one `dim = -1` entry each).
The changelog records before → after and says the time gain is unconfirmed on ttsim.

### [ ] Refinement 4 — Hide the reader's DRAM wait behind assembly (stall shadow) in the shift regimes

**Type**: perf

**Goal**: in `tile_row_shift` (and `tile_col_shift` / `rm_substick` from Refinement 1) the reader issues a block's
window reads, waits at the barrier, then assembles: nothing overlaps the wait. Issue block k+1's reads under a
second transaction id before assembling block k, then barrier on k+1's id (catalog `trid_double_issue`; window
`2 ×` one block's input tiles; the design's *Stall shadow* perf lamp). Target the multi-block shapes:
`(1,34,4096) s17` TILE DRAM and `(2,40,8192) s20` TILE (8 blocks of 8 per core after Refinement 3's tuning), plus
the widest `dim = -1` entry, `(1,1,32,4096) s1024 dim -1` in RM. No SUPPORTED change.

**Verifier notes**: Comes after Refinement 3 (it reuses R3's batched assembly and block size). Every transfer of an
explicit-sync kernel must carry a user id, so both ids are user ids (`{.trid = 0}`, `{.trid = 1}`), never the
untagged default. Skip a regime where the counts show one block per core (nothing to overlap).

**Done when**: a test reads off the op the number of DRAM read barriers the reader waits on with no assembled
block pending; it drops from one per block to one per core range (the first block) on the target shapes; window
footprint stays a derived expression of `BLOCK_H_TILES` in `l1_ledger.md`; golden suite and every existing test
green and bit-exact; no regression in the Refinement 3 counts across its guard set. The changelog states the time
gain is unconfirmed on ttsim.

## Coverage of TARGET − SUPPORTED

Every missing value is named above: `SUPPORTED["dim"]` -1 (Refinement 1), -3, -4 (Refinement 2);
`SUPPORTED["dtype"]` ttnn.float32, `SUPPORTED["rank"]` 2, 4, `SUPPORTED["chunks"]` 1, 3, 4,
`SUPPORTED["last_chunk"]` smaller (Refinement 2). `layout`, `memory_config`, `sharding` and `alignment` already equal
TARGET. INVALID is empty; this part has no INVALID_FOR_ARCH value for split.
