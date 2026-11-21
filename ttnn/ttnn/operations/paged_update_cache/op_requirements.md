# Operation Requirements: paged_update_cache

## Definition
- **Formula**: for every user `u` with `p = update_idxs_tensor[u] != -1`, every head `h < heads` (= `cache.shape[1]`) and every `d < D`:
  `cache[page_table[u, p // block_size], h, p % block_size, d] = input[0, u, h, d]`; every other cache element unchanged; written in place into the caller's `cache_tensor`, which is returned (R3).
- **PyTorch Reference**: `eval/specs/paged_update_cache/spec.py::reference` —
  `cache[page_table[u, p // block_size], :, p % block_size, :] = input[0, u, :heads, :]` for each user with `p != -1`.
- **Import Path**: `from ttnn.operations.paged_update_cache import paged_update_cache`
- **Function Signature**:
  ```python
  def paged_update_cache(
      cache_tensor: ttnn.Tensor,     # [blocks, heads, block_size, D], TILE, DRAM interleaved: written in place
      input_tensor: ttnn.Tensor,     # [1, users, H, D], TILE, L1 height-sharded
      *,
      update_idxs_tensor: ttnn.Tensor,   # [users], int32, ROW_MAJOR, DRAM interleaved
      page_table: ttnn.Tensor,           # [users, blocks_per_seq], int32, ROW_MAJOR, DRAM interleaved
  ) -> ttnn.Tensor: ...                  # cache_tensor itself
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement axes**: memory_config = L1, sharding = height (the input; the cache and the index tensors are fixed by the specification's CALL: DRAM interleaved, int32 row-major)
- **SUPPORTED shape-derived axes**: users = 1, alignment = h_non_aligned only (input rows per user < 32)
- **Regime**: `direct_row_write` — one data-movement kernel (`row_writer`, thread plan (R, C, W) = (0, 0, 1)) on each core holding a user's shard; 2·heads·dtiles face-row writes straight from the resident shard into the cache pages at byte offsets, one read barrier and one write barrier per core. No compute kernel, no rings.
- **Cores**: the input shard's cores (users = 1 → one cluster; the users axis is the only spread axis, see op_design.md Traffic ranking)
- **Compute config**: none (the specification gives the op none)
- **Golden baseline**: 18 / 18 supported cells pass (13 cells + 5 loose cases), 19 xfail_expected, 11 / 11 rejections pass, 0 in every loud category (per `verifier_report.json`)

### [ ] Refinement 1 — Complete TARGET: float32 and 32-row (tile-aligned) inputs

**Goal**: add the two values TARGET still lacks, on the built `direct_row_write` regime:
- `SUPPORTED["dtype"]` ← `ttnn.float32` (cache and input alike; `validate()` already refuses an input whose dtype is not the cache's, R8);
- `SUPPORTED["alignment"]` ← `"tile_aligned"` (an input of 32 rows per user; rows `heads..31` are ignored, R6).

This moves the 19 golden cells now in `xfail_expected` (16 float32 cells, 3 bfloat16 tile_aligned cells) to `supported_pass`, with the `exact` numeric requirement (rule R1: a float32 value must arrive bit-for-bit — it never passes through a 16-bit buffer).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Expected to need **zero kernel changes**. The verifier ran the program artifacts with `validate()` bypassed (`tests/ttnn/unit_tests/operations/paged_update_cache/probes/probe_001.py`): float32 at `8x4x32x512`, `6x12x96x96`, `2x1x64x128` (32 rows) and `8x8x32x64` (32 rows), and bfloat16 at `4x32x32x64` (32 rows), all bit-exact against the reference. `face_row_bytes` / `face_bytes` / page sizes already derive from `element_size()`, and the kernel never reads H.
- **Do not add `compute_kernel_config`** or any parameter, even though the skill's default bundle does. The signature must equal the specification's stub, and DECISIONS states the op has no compute config. The parts of the skill that apply are the dtype widening and the precision-matrix test contract.
- Extend the plan tests (`test_paged_update_cache_plan.py`), which hard-code bfloat16 today, to float32 for the session-order test and the off-origin-shard test, and extend the precision baseline to float32.
- A cheap generality refinement on purpose: the TARGET gap is only these two knob-turn values (op_design.md, "TARGET − PHASE0 as refinements"). Every other axis is already at TARGET.

**Done when**: `verify_supported` shows `supported_pass` = 37, `xfail_expected` = 0, every loud category 0; the 11 rejections still pass; the acceptance, plan and precision-baseline tests are green, including the new float32 variants.

### [ ] Refinement 2 — Spread the write issue of the widest cells over the writer-thread plan

**Type**: perf

**Goal**: after Refinement 1, the cells with the most writes per call are those with `heads·dtiles` = 64: `8x4x32x512` (float32 and bfloat16), `4x32x32x64` (32 heads, tile_aligned) and `3x16x32x128`. Each is **128 face-row NoC writes, all issued by one DM hart** while 5 user DM harts of the cluster sit idle. The op is latency-bound (index round trip, then write issue, then acknowledgements), and write issue is the only part that grows with the shape. Spread it over the writer threads the design already exposes: the `ROW_WRITER_THREADS` knob, with `(h, j)` units dealt round-robin, which the kernel already implements and `test_writer_thread_plan` exercises at W = 2, 3. This is the op_design.md *Writer threads* perf lamp, a knob-turn. The related catalog entries in `ttnn/ttnn/operations/examples/master.md` are `trid_double_issue` (keep one barrier per thread's block, never per unit) and `double_buffer` (keep bytes in flight). No SUPPORTED change.

The trade to settle: every extra thread re-reads the positions page and the page-table row (2 → 2·W DRAM index reads per core), and the small traced cells (Llama `128x8x32x64`: 32 writes; Qwen `8x8x32x128`: 64 writes) gain little. Choose W per shape from `heads·dtiles` (a host-side regime predicate, one source), **or** have one thread load the index and hand it to the others. Either way, the guard cells must not move more index bytes than today.

**Gate (structural — this run is on ttsim, which has no clock; see the backend page's *Performance measurement*)**: the count is **NoC writes issued per DM hart per call** on the widest cells. It is 128 today (W = 1); done at ≤ 64 (W ≥ 2). Read it off the op with a test, not from arithmetic: for example a debug CT arg that skips thread t, after which exactly that thread's units are unwritten and their number is counted on the host, as op_design.md Work Distribution test (5) asks for W > 1. The gain in time is **unconfirmed** until ttsim-perf or hardware measures it; state that in the changelog. If a timed backend later shows no win, the constant goes back to 1.

**Done when**: writes per hart on the widest cells ≤ 64, read off by a test; on the guard set — the Llama and Qwen traced cells, the smallest cell `4x1x32x32`, a float32 cell and a tile_aligned cell (one per dtype × alignment; there is only one kernel path, layout and placement) — DRAM index reads per core and writes per hart are no worse than today; the golden suite stays green (all 37 cells, 11 rejections) and the session-order test passes at the chosen W.
