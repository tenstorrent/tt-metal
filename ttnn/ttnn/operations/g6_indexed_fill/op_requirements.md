# Operation Requirements: g6_indexed_fill

## Definition
- **Formula**: `y[b, r, w] = s[i, r, w]` if `b == ids[i] < B` for some `i` (the largest such `i`), else `y[b, r, w] = x[b, r, w]`; `ids` read unsigned on the device in every call, an entry `>= B` writes nothing.
- **PyTorch Reference**: `pytorch_g6_indexed_fill` in `eval/golden_tests/g6_indexed_fill/helpers.py` (= `input_tensor.index_copy(0, batch_ids, value_tensor)` with out-of-range entries dropped; uint32 carried in int64).
- **Import Path**: `from ttnn.operations.g6_indexed_fill import g6_indexed_fill`
- **Function Signature**: `g6_indexed_fill(batch_ids: ttnn.Tensor, input_tensor: ttnn.Tensor, value_tensor: ttnn.Tensor, *, dim: int = 0) -> ttnn.Tensor` — `batch_ids` `[K]`, `input_tensor` `[B, d1, d2, W]`, `value_tensor` `[K, d1, d2, W]`, all uint32 ROW_MAJOR DRAM interleaved; result `[B, d1, d2, W]`, uint32 ROW_MAJOR DRAM interleaved, a new tensor.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [uint32]
- **SUPPORTED layout**: [ROW_MAJOR_LAYOUT]
- **SUPPORTED shape-derived axes**: alignment ∈ {h_non_aligned, w_non_aligned, tile_aligned} (verifier widened from {h_non_aligned}: the prompt's MUST rule forbids refusing a valid shape on the host, and the ROW_MAJOR kernel never branches on alignment)
- **SUPPORTED op-specific axes**: none (`dim` = 0 only, host-validated, not an axis)
- **Cores**: full grid; `split_work_to_cores(row_wise=True)` over x pages ++ value pages (production: 8 units on 8 cores, one rectangle)
- **Compute config**: none (data movement only; no compute kernel, no DEST)
- **Regimes built**: `ids_resident` (K ≤ 16384) and `ids_streamed`, one kernel, compile-time flag
- **Golden baseline**: 11 / 11 passing with `EVAL_CAPTURE_L1=0` (9 supported_pass + 2 regression); 9 / 11 under the default runner because of two L1-capture harness defects (`verification_report.md` § Verifier CLI Summary)

## Queue status: no refinements

The queue is empty on purpose. Every source of a refinement was checked:

| Source | Result |
|--------|--------|
| Generality: `TARGET − SUPPORTED` per axis | **∅**. `dtype` {uint32}, `layout` {ROW_MAJOR}, `alignment` {h_non_aligned} are all supported, and SUPPORTED exceeds TARGET on `alignment`. Phase 0 is TARGET, as the prompt states ("There is no refinement of scope"). |
| `INVALID` / `INVALID_FOR_ARCH` | Neither declares anything, so nothing is skipped. |
| Failing cells (OOM / precision / bug / hang) | None caused by the op. The two `supported_fail` cells under the default runner come from the harness (L1 capture disables the program cache, which breaks trace capture, and a refusal-only case allocates no L1). Stock `ttnn.add` reproduces the first (`probes/probe_006.py`). No change to this op can move them, so they are escalated in the report rather than queued. |
| Perf (2:1 cadence) | No measurable target. The flagged `LOOSE_CASES` `perf_focus` cell is 7 tiles and the largest golden cell is 10 tiles. `eval/prompts/perf_rules.txt` allows measuring only cells of more than 32 tiles and overrides every other instruction, so no perf refinement can carry a device-ns gate. That cell's stated goal (kernel time under the 2.10 µs stock incumbent) is already met: 1.09 µs on 8 cores, as reported by the implementer. |

Revisit if any of these happen:

- TARGET gains a value (e.g. `/golden-tests` adds a dtype or a memory placement).
- A production shape of more than 32 tiles enters INPUTS or LOOSE_CASES. The first perf lever to file would be the read/write overlap for multi-unit cores: co-tune `staging_budget_bytes` with `max_staging_depth`, catalog `double_buffer`. That is a knob-turn; both knobs already exist in `PLAN_CAPS`.
- A large-K caller appears. The `ids_mcast` deferred row (scheme-change, multicast helper) and a cheaper `select_block` scan become candidates.
