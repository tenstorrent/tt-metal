# Operation Requirements: plus_one

## Definition
- **Formula**: `x <- x + 1` on every 32-bit element, in place, wrapping (INT32_MAX + 1 = INT32_MIN; 2^32 − 1 + 1 = 0). With `skip_negative_entries=True`, only elements whose bits read as int32 lie in `[0, INT32_MAX)` are incremented.
- **PyTorch Reference**: `input.add_(1)`; with the flag `input.add_(((input >= 0) & (input < 2**31 - 1)).to(input.dtype))` (the specification's `reference()` computes it in float64 with the dtype's wrap).
- **Import Path**: `from ttnn.operations.plus_one import plus_one`
- **Function Signature**: `plus_one(input_tensor: ttnn.Tensor, *, skip_negative_entries: bool = False) -> ttnn.Tensor` — returns `input_tensor` itself (rule R3: no output allocation).

Specification: `eval/specs/plus_one/spec.py`. Design: `op_design.md` (Regime A `segment_split_in_place`). Target: Quasar on ttsim (untimed — perf gates below are structural counts, per `.claude/references/backends/ttsim.md`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [int32, uint32]
- **SUPPORTED layout**: [ROW_MAJOR]
- **SUPPORTED memory_config / sharding**: DRAM / interleaved
- **SUPPORTED op-specific axes**: skip_negative_entries ∈ {False, True}; rank = 1
- **Cores**: multi-core × multi-thread: units (row segments) spread over `ceil(bytes / MIN_WORKER_BYTES)` workers (capped at grid × `WORKER_THREADS` = 32 × 4), one DM kernel of `WORKER_THREADS = 4` threads per engaged cluster; no NEO (no compute kernel)
- **Compute config**: none (spec DECISIONS: an exact integer increment has no compute config)
- **Golden baseline**: 28 / 28 supported cells passing; 148 xfail_expected (L1 × rank 1: 28; ranks 2–4 × DRAM/L1: 120); 0 loud (per `golden_results/verifier_report.json`)

### [ ] Refinement 1 — L1-interleaved placement and ranks 2–4

**Goal**: add `"L1"` to `SUPPORTED["memory_config"]` and add `2`, `3`, `4` to `SUPPORTED["rank"]` — i.e. widen SUPPORTED to the specification's full `TARGET`. Both are **knob-turns** of the built Regime A (op_design.md, "TARGET placement axes as regimes"): `rows = volume / W` already folds the leading dims into whole-row units, and `plan_for` already takes the segment granule from `ttnn.get_l1_alignment()` for an L1 buffer. The path must stay as performant as Phase 0's: whole-row units with `block_units` reads per barrier (the `(512, 8)` shape is 32 rows per thread, one barrier per block), the grid filled by the same `MIN_WORKER_BYTES` rule.

**Implementation skill**: /interleaved-parallel

**Verifier notes**: verifier probe (bypassing `validate()`, builder defaults) ran **all 148** currently-xfail cells — every INPUTS shape × {DRAM, L1} × {int32, uint32} × flag, minus PHASE0 — and every one was bit-exact; `test_block_path` already pins `(512, 8)` / `(1, 1, 17, 50)` through the builder. So the expected kernel delta is zero; the work is the gate widening, an L1 member in the plan tests (every-thread-ran and pinned splits on an L1 tensor: the L1 granule is 4 elements, not 16), and the one-device-session test (`untilize → plus_one → untilize → plus_one`) repeated on an **L1** tensor, since L1 pages share banks with other programs' buffers. `/interleaved-parallel` is the secondary pointer only for confirming the row split's `row_wise=True` wiring on the multi-row shapes — no new split is needed. This refinement lands the whole contract of Refinement 2's target (L1, ranks 2–4), which is why it comes first; it is the only generality refinement because `TARGET − SUPPORTED` holds just these two cheap axes (see verification_report.md).

**Done when**: all 148 golden cells now in `xfail_expected` pass (`supported_pass` = 176, `xfail_expected` = 0, loud categories 0); SUPPORTED equals TARGET; the plan tests and the session test pass on an L1 tensor as well as on DRAM.

### [ ] Refinement 2 — Bank-local in-place increment for L1-interleaved tensors (Regime E)

**Type**: perf

**Goal**: no loose case is perf-flagged, so the region is free-selected: the **L1-interleaved** cells Refinement 1 unlocks. Under Regime A every L1 page crosses the NoC twice (bank → worker scratch → bank) although the bank's own cluster could increment it where it lies. Build the design's deferred Regime E `l1_bank_local_in_place` (op_design.md, Regimes): units assigned by **bank owner** instead of contiguous ranges, each page incremented through a local `scoped_lock` on the page's own L1 address by the threads of the cluster that holds it — no scratchpad copy, no NoC transfer. Regime selection: one predicate on `memory_config == "L1"` (plus whatever the implementer measures for single-page tensors, where bank-local concentrates the whole tensor on one cluster's 4 threads instead of A's 16 workers — keep A there if that is the better structure, and record why). Catalog background: `ttnn/ttnn/operations/examples/master.md` (`width_split` for the occupancy trade-off on few-page tensors; `bank_stagger` for the bank ↔ core mapping). Lever tier ⭐⭐ T2 — one phase. No SUPPORTED change.

**Verifier notes**: ttsim is untimed, so the gate is a **structural count** (backend page): **NoC bytes moved per call on an L1-interleaved multi-page tensor**, now `2 × rows·W·4` (e.g. 32 KiB for `(512, 8)`, 16 KiB for `(32, 64)`, 8 KiB for `(1, 1, 32, 64)`), target **0**; read it off the op with a probe or test (e.g. counting the transfers the kernel issues via a probe CT arg, or asserting the selected regime and its zero-transfer unit list from the host plan). The gain in time is **unconfirmed** until a timed backend (ttsim-perf) or hardware measures it — say so in the changelog. Keep each page owned by exactly one thread (GUIDANCE: a page two cores increment is off by one), and keep the DM-cache discipline of Phase 0 (64-B line-exclusive regions per thread inside a page are not given by the allocator — the lock ranges of two threads must not share a line). The every-thread-ran probe (`probe_tag_threads`) must cover the new regime.

**Done when**: on the L1 multi-page target shapes `(512, 8)`, `(32, 64)`, `(4, 8, 32)`, `(1, 1, 32, 64)` the NoC bytes per call drop from `2 × tensor bytes` to 0, recorded before/after in the changelog; the golden suite stays green (176 supported_pass); and no structural regression across the config-spanning guard set — one representative per kernel path × placement: DRAM rank-1 `(4096,)` (16 units, 4 clusters × 4 threads, 2 transfers per unit), DRAM rank-2 `(512, 8)` (one barrier per 32-row block), L1 rank-1 `(100,)` — each keeps its Phase 0 counts (transfers per call, barriers per block, clusters and threads engaged) unless the entry records why the new count is better.
