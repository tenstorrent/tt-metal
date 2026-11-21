# Operation Requirements: expand

## Definition
- **Formula**: `out[i_0, …, i_{k-1}] = in[j_0, …, j_{r-1}]`, with the input's dims matched to the last `r` output dims
  and `j_d = 0` where the input's dim `d` is 1 and expanded, `j_d = i_{d+k-r}` otherwise; dims before the input's
  first dim are added and the whole tensor repeats along them. A copy, not a view (rule R1: every element moved
  unchanged; numeric requirement `exact` for every cell).
- **PyTorch Reference**: `input.double().expand(*output_shape).clone()` (`reference` in `eval/specs/expand/spec.py`)
- **Import Path**: `from ttnn.operations.expand import expand`
- **Function Signature**: `expand(input_tensor: ttnn.Tensor, output_shape) -> ttnn.Tensor` — `output_shape` a list or
  tuple of ints (R3), `-1` keeps a dim (R2); output in the input's dtype, layout and buffer, interleaved (R6).
- **Specification**: `eval/specs/expand/spec.py` (the only source of requirements; RULES R1–R8, REJECTIONS X1–X3).
- **Target / backend**: Quasar (`.claude/references/targets/quasar.md`) on ttsim (`.claude/references/backends/ttsim.md`):
  Metal 2.0 `ttnn.ProgramSpec` only; perf gates are **structural counts** (ttsim has no clock).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **One dispatch (mandatory, every phase)**: each call runs exactly one `ttnn.generic_op`. No `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` / `ttnn.repeat` / `ttnn.copy` / `ttnn.to_memory_config` before or after it — not even as a partial-tick fallback.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: DRAM / interleaved
- **SUPPORTED shape-derived axes**: changes = h; alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank = 2
- **Regime**: R-H (tile row-broadcast, source-major contiguous runs): read each source tile once per cluster, build
  the row broadcast in place in the ring entry (self-aimed NoC copies), write that entry to every output tile it
  feeds. Thread plan (R, C, W) = (1, 0, 1): no compute kernel.
- **Cores**: the whole queried grid (`split_work_to_cores(..., row_wise=True)`), knob `max_cores`
- **Compute config**: none (the op computes nothing; the specification gives it no compute config)
- **Golden baseline**: 9 supported_pass (8 PHASE0 cells + the loose case `output_shape_as_list`), 248
  xfail_expected, 3 rejections (X1–X3) pass; supported_fail / xpass_drift / xfail_wrong_mode all 0. Every cell bit-exact.

### [ ] Refinement 1 — General tile index map: leading and added dims, ranks 3 and 4

**Goal**: add `"leading"` and `"added"` to SUPPORTED["changes"], and add `3` and `4` to SUPPORTED["rank"], for the
tile layout (bf16, DRAM). This covers every TILE cell whose W does not change: whole planes repeated (`leading`,
including the no-op copy `(32,64)→(-1,-1)`), dims added in front (`added`, e.g. `(32,64)→(4,32,64)`,
`(1,128)→(2,1,32,128)`), and rows repeated at ranks 3/4 together with any expanded leading dim
(`(4,1,512)→(4,128,-1)`, `(2,1,1,128)→(2,4,50,128)`). 15 bf16/TILE/DRAM cells move from xfail to pass.

**Verifier notes**:
- This is the design's deferred regime **R-LEAD** plus the leading-dim part of **R-HW/combined**
  (`op_design.md`, Regimes). In Blocking-Model terms it is a knob-turn on the same topology: the loop nest,
  the ring, the source-major contiguous runs and the plan (1, 0, 1) all stay. What changes is the index map
  `L → (source tile, fan index)`: a source is any input tile `(input plane, input tile row, j)`; the fan-out is
  the product of every expanded dim (expanded leading dims × output tile rows when H expands from 1). A
  compile-time `build_mode` picks the build: identity (H not expanded: the page is written as read, no
  replication copies) or the row broadcast Phase 0 already has. Keep `output_h_tiles` per plane
  (`ceil(H/32)`, never `floor(planes·H/32)`), as Phase 0 does.
- Hold the unlock to the Phase 0 performance bar: whole grid engaged, each source read once per cluster
  (input page reads ≤ `num_sources + engaged_clusters − 1` — extend `test_input_reads_structural_count` to the new
  cells), one barrier per block on both sides. No new buffer is expected; if one is added, update `l1_ledger.md`.
- Must land before Refinement 2: the rank-3/4 `w` cells (e.g. `(1,1,17,1)→(2,1,17,50)`) need this index map.
- Re-run the one-session tests (`test_after_another_program`, `test_before_a_tt_metal_program`) on a rank-4 cell.

**Done when**: every bf16 / TILE / DRAM golden cell with `changes ∈ {h, leading, added}` at ranks 2–4 passes
(bit-exact), the golden CLI shows 0 supported_fail / xpass_drift / xfail_wrong_mode, and the input-read structural
count holds on the new cells.

### [ ] Refinement 2 — Column broadcast: W expanded (and H with it)

**Goal**: add `"w"` to SUPPORTED["changes"] for the tile layout (bf16, DRAM): rows made longer from one column
(`(32,1)→(32,64)`, `(128,1)→(-1,100)`, `(4,32,1)→(-1,-1,50)`), with H expanded as well (scalar broadcast:
`(1,1)→(64,128)`, `(1,1,1,1)→(1,1,64,128)`) and with leading dims expanded (`(1,1,17,1)→(2,1,17,50)`). 9
bf16/TILE/DRAM cells move from xfail to pass.

**Verifier notes**:
- The design's deferred regime **R-W** plus the scalar part of **R-HW/combined**. Same loop nest and ring as
  Refinement 1; two new `build_mode`s: column broadcast (column 0 copied to columns 1..31 of every face row —
  `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp` `copy_tile_column`, RISC stores, which need the
  data-cache flush before `push_back` — `tile_column_dataflow.hpp:23`), and scalar broadcast (column then row,
  reusing Phase 0's row replication). The fan-out axis now includes output tile columns
  (`output_w_tiles = ceil(W_out/32)`, the last one padded for `w_non_aligned` results).
- The column build's element stride is dtype-sized: derive it from the page size, as Phase 0 derives
  `face_row_bytes`, so Refinement 4's float32 needs no kernel change here.
- Depends on Refinement 1 (rank-3/4 cells with expanded leading dims use its index map).

**Done when**: every bf16 / TILE / DRAM golden cell with `changes = "w"` passes (bit-exact), all prior cells still
pass, golden CLI loud categories 0, and the one-session tests pass on a `w` cell.

### [ ] Refinement 3 — Writer throughput on the traced call

**Type**: perf

**Goal**: speed up the one call Qwen3-VL makes, `(1,2560)→(1330,-1)` bf16 / TILE / DRAM (no loose case is
perf-flagged, so the verifier selected this region: it is the reason the op exists). The op is bound by DRAM
writes: each cluster issues 105 output writes from one explicit-sync writer hart, against 3 source reads. Two
levers from the design's perf lamps and the catalog look applicable (`ttnn/ttnn/operations/examples/master.md`):
- **Writer threads** — the writer must stay explicit sync (it writes one entry many times);
  W = 2 writer harts is the lever: either a STRIDED ring dealing sources over 2 writer threads, or each
  writer thread writing alternate output rows of every entry. A DM→DM ring with 2 consumer threads is not in the
  examples: prove the protocol on one shape first.
- **DRAM bank order of writes** (`bank_stagger`) — a cluster writes one source's rows, pages `j, j+Wt, j+2Wt, …`;
  with `Wt = 80` and an even bank count every write of a source lands in the same bank. Writing row-major across
  the block's sources (`for r: for k`) alternates banks.
No SUPPORTED change.

**Done when** (structural counts — ttsim has no clock; the gain in time is unconfirmed until `ttsim-perf` or
hardware measures it, and the changelog says so):
- writer harts issuing output writes per cluster on the traced call: 1 → 2, with a test showing each hart wrote
  its share (the plan stated once on host, counts as CT args; a missing thread shows as missing output);
- most output writes issued by one hart on the busiest cluster: 105 → ≤ 53;
- share of a hart's consecutive write pairs that hit the same DRAM bank (bank count from the device): ≈ 100 % → ≤ 50 %;
- no regression on the guard set — one representative per kernel path: `(1,8192)→(17,-1)` (fan-out 1),
  `(1,17)→(32,17)` (one cluster), a Refinement-1 `leading` cell, a Refinement-2 `w` cell: output pages written once,
  input reads ≤ `num_sources + engaged_clusters − 1`, ring footprint per `l1_ledger.md` not grown beyond the
  added writer slice;
- golden suite green and both one-session tests pass.

### [ ] Refinement 4 — Row-major layout, float32 and L1 placement

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to SUPPORTED["layout"], `ttnn.float32` to SUPPORTED["dtype"], and `"L1"` to SUPPORTED["memory_config"],
across every `changes` value and rank the tile path supports by then. That is the
remaining 224 golden cells (7 of the 8 dtype × layout × placement combinations × 32 INPUTS entries).

**Implementation skill**: /memory-layouts, /numeric-formats-metal

**Verifier notes**:
- **Row-major** is the design's deferred regime **R-RM**: a pure-RM data-movement path (the op computes nothing,
  so no tilize/untilize stage): page = stick (`buffer_page_size()`), `h` writes the input's one stick to every
  output row, `w` builds each output stick from one element (element fill, dtype-sized), `leading`/`added` copy
  sticks as they are. Same source-major runs over pages, same ring discipline (explicit sync, `{.trid = 0}` on every
  transfer). Watch the RM stick alignment: L1 16 B, DRAM 64 B (target page); `W = 17` bf16 is a 34 B stick.
- **No wrapper fallback**: wrapping the tile path in `ttnn.to_layout` would be a second dispatch, which the run's
  one-dispatch rule forbids. A partial tick here means a named axis value left out of SUPPORTED, not a wrapper.
- **float32**: the tile path needs no kernel change (`page_size`, `face_row_bytes` and Refinement 2's column stride
  derive from the page). There is no compute kernel and the specification gives the op no compute config, so the
  skill's `compute_kernel_config` / DEST parts do not apply; R1 is the requirement that matters (no element passes a
  narrower format) — extend `test_expand_precision_baseline.py` to float32, special values included, bit-exact.
- **L1**: inputs are read through the same `TensorAccessor`; the output is allocated in L1 by R6
  (`input_tensor.memory_config()`), already wired. Check `output.memory_config()` in a test, as
  `test_output_placed_like_input` does for DRAM. The traced result is 6.8 MB bf16 / 13.6 MB float32 in L1 interleaved
  over 32 clusters; if any L1 cell fails with an `OOM`, it stays failing and is named in a follow-up
  (`Refinement 4b`), never put in EXCLUSIONS.
- Lands after Refinements 1–2 so the RM path is written once against the general index map and all build modes.

**Done when**: all 256 cells pass (bit-exact) apart from any named `OOM` follow-up, golden CLI loud categories 0,
the precision baseline passes in float32, and the one-session tests pass on an RM cell and an L1 cell.

### [ ] Refinement 5 — Read/write overlap on the fan-out-1 shapes

**Type**: perf

**Goal**: speed up the wide, short results where every source feeds one output tile (`fan_out_tiles = 1`):
`(1,8192)→(17,-1)` and `(1,4096)→(32,-1)` (bf16 / TILE / DRAM). There a cluster reads as many tiles as it writes
(8 and 4), all in one block: the writer starts only after every read and the build of the whole block, so reads
and writes never overlap, and the build pays 5 dependent barriers. Levers (design perf lamps *Overlap* and
*Replication schedule*; catalog `double_buffer`, `trid_double_issue` in `ttnn/ttnn/operations/examples/master.md`),
all knob-turns or local kernel changes (⭐/⭐⭐):
- `block_src_tiles ≈ ceil(max_src_tiles_per_core / 2)` when `fan_out_tiles` is small, so the depth-2 ring overlaps
  block `b+1`'s reads with block `b`'s writes (the ring depth then un-clamps to 2 by itself);
- the replication schedule: direct copies from row 0 instead of doubling steps, 5 → ≤ 2 barriers per block.
No SUPPORTED change.

**Done when** (structural counts; the gain in time is unconfirmed until `ttsim-perf` or hardware measures it):
- blocks per cluster on `(1,8192)→(17,-1)`: 1 → ≥ 2, the ring 2 blocks deep;
- replication barriers per block: 5 → ≤ 2;
- no regression on the guard set (the traced call, `(1,17)→(32,17)`, one `leading`, one `w`, one RM and one
  float32 cell): output written once, input reads ≤ `num_sources + engaged_clusters − 1`, blocks per cluster on the
  traced call unchanged (one), L1 footprint within the ledger;
- golden suite green and the one-session tests pass.
