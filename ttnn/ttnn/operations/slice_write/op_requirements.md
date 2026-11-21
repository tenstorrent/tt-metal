# Operation Requirements: slice_write

## Definition
- **Formula**: `output[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]] = input` (step 1 in every dimension), in place in the caller's destination, which is returned.
- **PyTorch Reference**: `reference(input_tensor, output_tensor, start, end)` in `eval/specs/slice_write/spec.py`. It clones the destination and copies the source into `narrow(...)` of the block.
- **Import Path**: `from ttnn.operations.slice_write import slice_write`
- **Function Signature**: `slice_write(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor, start, end, step) -> ttnn.Tensor`. `start`, `end` and `step` are lists or tuples of ints, one entry per dimension, and `step` is all ones. The specification fixes this signature, so no refinement adds a parameter (in particular no `compute_kernel_config`: the op computes nothing).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (ttsim, Quasar)**: no clock, so no device-ns. A perf refinement's gate is the **structural count** its entry names, read off the op by a test or a probe, and its time gain stays unconfirmed until a timed backend measures it (`.claude/references/backends/ttsim.md`, *Performance measurement*).

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [ROW_MAJOR]
- **SUPPORTED placement (source)**: memory_config=[L1], sharding=[height] (any shard grid, either orientation, ragged last shard). The destination is DRAM interleaved (rule R8).
- **SUPPORTED shape-derived axes**: written_region=[outer], alignment=[tile_aligned, h_non_aligned], rank=[4]
- **SUPPORTED op-specific axes**: step=[ones]
- **Regime**: `shard_owner_writes`. The ring is borrowed onto each core's resident shard (zero-copy, 0 B of L1 allocated), and the owning core writes its rows to their destination pages, one tagged write per row and one barrier per 32 rows.
- **Cores**: the source's shard grid (up to all 32 clusters on the golden shard configs). Thread plan `(R, C, W) = (1, 0, 1)`: the op is bound by data movement and has no compute.
- **Compute config**: none (the op computes nothing; spec DEFINITION)
- **Golden baseline**: 14 / 14 supported cells pass (17/244 tests incl. 3 rejections; 167 xfail_expected, 60 invalid_skipped; 0 in every loud category)

### [ ] Refinement 1 — Interleaved source (regime `interleaved_stream`)

**Goal**: add `"DRAM"` to `SUPPORTED["memory_config"]` and `"interleaved"` to `SUPPORTED["sharding"]`: a source interleaved in DRAM or in L1 (rule R7), served by the design's deferred regime `interleaved_stream` and spread over the whole compute grid. The sharded regime stays exactly as built. Regime selection is exact: `sharding == "interleaved"` selects the new regime, and `{L1, height}` keeps `shard_owner_writes`.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- This is a knob-turn in the Blocking Model: `src_row` is independent, with no combine and no reuse-shared operand. It still comes first because it is the only structural addition left: a second regime with a reader that moves data. Refinement 2's light axes must then extend both regimes once, not be rebuilt after this one.
- **Built performantly, not correct-only** (Refinement 3's guard set includes this regime):
  - Split `total_rows` over `device.compute_with_storage_grid_size()` with `ttnn.split_work_to_cores(..., row_wise=True)`.
  - A new reader fills an **allocated** ring `dfb_src_rows`, depth 2 (`l1_ledger.md` deferred row), with `rows_per_block` rows per entry through `TensorAccessor(tensor::src)`. Make the block the coarsest that fits `2 · rows_per_block · src_row_stride_bytes ≤` the queried budget (`ttnn.get_max_worker_l1_unreserved_size()`) and is ≤ the core's share. Issue the block's reads and then **one** barrier per block, never read-one/barrier (catalog `double_buffer`).
  - The reader uses explicit sync with every read tagged with a user transaction id, and hands the tag back (`noc_async_read_set_trid(0, …)`) as its last statement.
  - The writer kernel is reused unchanged: it already stores "a block of rows at a ring entry, starting at `first_row`".
- `rows_per_block`, the ring depth and `WRITE_BATCH_ROWS` each have one host source (DRY). Add the regime's rows to `l1_ledger.md` and its traffic to the data-movement budget: each source byte crosses once on read and is written once, with 0 cross-core traffic.
- The L1-interleaved source is read through its accessor; that is correct for an interleaved buffer. The rule against accessor reads applies only to a core's own *local shard*, which keeps the borrowed-ring path.
- Thread plan stays `(1, 0, 1)` per regime unless a structural count says otherwise. Extend the session test (one program before, a tt-metal program after, one device session) to the new regime.

**Done when**: every golden cell with `sharding = interleaved` (DRAM and L1) at the Phase 0 values of the other axes (bfloat16, `outer`, `tile_aligned` / `h_non_aligned`) passes. The regime engages the whole grid on shapes with ≥ 32 rows, a test reads that off the built `ProgramSpec`, and the session test passes for this regime.

### [ ] Refinement 2 — float32 / uint32, `h` blocks, non-aligned last dimension (both regimes)

**Goal**: add `ttnn.float32` and `ttnn.uint32` to `SUPPORTED["dtype"]`, `"h"` to `SUPPORTED["written_region"]`, and `"w_non_aligned"` to `SUPPORTED["alignment"]`, on both the sharded regime and Refinement 1's interleaved regime. Every value must travel bit-exactly (rule R1: a uint32 or float32 value must not pass through a bfloat16 buffer).

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**:
- Use only the dtype part of `/numeric-formats-metal` and only the non-aligned rule of `/memory-layouts`. Do **not** expose `compute_kernel_config` or add any parameter: the signature is the specification's, and the op has no compute and no DEST.
- **Probe evidence (verifier, sharded regime, SUPPORTED widened temporarily, then reverted):**
  - All 27 float32 / bfloat16 cells of `h` and `w_non_aligned` passed with **no code change**. That includes 34 B and 94 B rows, which retires the design's "sub-16 B-multiple NoC write" risk for the sharded path.
  - All 20 uint32 cells failed at program build: `TT_FATAL dfb.cpp:148 is_data_format_supported`, because `UInt32` is not a Quasar DFB format (`tt_metal/common/tt_backend_api_types.cpp:97-122`).
  - The same 20 cells passed once `dfb_src_shard`'s `data_format_metadata` was `ttnn.int32` for a uint32 source. The ring is a dataflow-only payload (bytes go L1 → NoC → DRAM), so the label carries no conversion.
  - Put that dtype → ring-label mapping in one helper used by both regimes' rings.
- Interleaved regime: the ring's row stride must be the L1-aligned page (`align(row_bytes, ttnn.get_l1_alignment())`), not `row_bytes`, so that 34 B / 94 B / 188 B rows land on aligned slots. Re-confirm the sub-16 B-multiple **reads** from DRAM / L1 pages there, since the probe covered writes only.
- Order: this comes after Refinement 1 so that each value is built once over both regimes (the difficulty ordering: structure first, light axes after).

**Done when**: every golden cell of `dtype ∈ {float32, uint32}`, `written_region = h` and `alignment = w_non_aligned` passes on all three source placements (L1 height-sharded, L1 interleaved, DRAM interleaved), uint32 and float32 bit-exactly.

### [ ] Refinement 3 — Writer issue rate on the traced single-shard call

**Type**: perf

**Goal**: no loose case is perf-flagged, so the region is the model's own traced call as ResNet-50 makes it:
- **The call**: `[1, 59, 115, 32]` bfloat16 in **one** height shard of 6785 rows on one cluster, written into `[1, 115, 115, 32]` in DRAM (`test_slice_write_plan.py::test_shard_configurations[traced_one_shard]`).
- **What limits it**: rows are 64 B, so cost per row is command issue, not bytes. A single writer hart issues all 6785 writes and waits on 213 barriers, while 4 of the cluster's 6 user DM harts sit idle.
- **What the knob does today**: `THREAD_PLAN`'s W knob exists, but it falls back to W = 1 here because 6785 = 5·23·59 has no divisor ≤ 4. The ring's entries must tile the borrowed shard exactly.

Levers from `ttnn/ttnn/operations/examples/master.md`:
- `split_reader`, applied to write issue: several writer harts, which needs a ragged split of the shard among them;
- `trid_double_issue`, applied to the write barrier: issue batch k+1 before retiring batch k;
- the batch depth `WRITE_BATCH_ROWS`;
- stateful writes (design lamp; TXN_ID is not allowed on stateful writes, so tag through `noc_async_write_set_trid`).

The implementer picks the levers. No SUPPORTED change.

**Done when**:
- **Structural count (ttsim: no device-ns).** On the traced single-shard call, the maximum number of rows any one writer hart issues drops from **6785 to ≤ 1697** (W = 4), or to ≤ 3393 if only W = 2 is reached, and the result stays exact. A test reads the count off the built program (thread counts × the rows per entry or per thread).
- The changelog records before and after and states that the time gain is unconfirmed until a timed backend measures it.
- Golden suite green, plan and session tests green.
- No regression of the same counts (rows per hart, barriers per hart) across the guard set, one per kernel path × placement:
  - the traced call on the golden 32-core shard config;
  - the 8 KiB-row `[1, 2, 32, 4096]` sharded cell;
  - one DRAM-interleaved and one L1-interleaved cell of Refinement 1's regime.

### [ ] Refinement 4 — Spread the traced single-shard call beyond its shard grid (regime `spread_beyond_shard_grid`)

**Type**: perf

**Goal**: build the design's deferred regime `spread_beyond_shard_grid` for sources whose shard grid under-fills the compute grid. On the traced call above, one cluster does all 6785 rows and 31 are idle.
- Helper cluster k takes a contiguous row range of an owner's shard with **one coalesced L1→L1 read per range** (catalog `read_coalescing`; `dataflow_kernel_lib::local_addr` / `async_read`, `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`) into a staging ring, then runs the built writer unchanged on it.
- This is a ⭐⭐⭐-class scheme-change (a new cross-core topology), so it stands alone.
- No SUPPORTED change.

**Verifier notes**:
- The regime **adds** up to `B` bytes of cross-core NoC traffic to a built regime that moves the minimum. The design kept it deferred because occupancy alone does not override the traffic ranking, so its value is the occupancy gain only. On ttsim that gain is structural and unconfirmed in time.
- Select it only when `clusters(shard grid with rows) < grid clusters`. The golden 32-core shard configs must keep `shard_owner_writes` unchanged (0 cross-core bytes).
- It composes with Refinement 3: helpers run the writer that Refinement 3 tuned.

**Done when**:
- **Structural count (ttsim).** On the traced single-shard call, clusters engaged rise from **1 to the full grid**, and the maximum rows written per cluster drop from **6785 to ≤ ceil(6785 / grid clusters)** (213 on this part's 32).
- Added cross-core traffic is ≤ `B` (434,240 B), at exactly one read per helper range.
- A test reads clusters and rows per cluster off the built program, and the result is exact.
- Golden suite green, session test green for the new regime.
- No change in the guard set's counts. In particular the golden shard configs still run `shard_owner_writes` with 0 cross-core bytes.
- The changelog says that the time gain is unconfirmed until a timed backend measures it.
