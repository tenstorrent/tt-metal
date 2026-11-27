# Operation Requirements: g6_clone

## Definition
- **Formula**: `bits(out[i]) = bits(x[i])` for every logical index `i`; `spec(out) = spec(x)`; `out` is a newly allocated buffer, and `x` is unchanged.
- **PyTorch Reference**: `def pytorch_g6_clone(x): return x.clone()` (in the input dtype, with no float32 round trip)
- **Import Path**: `from ttnn.operations.g6_clone import g6_clone`
- **Function Signature**: `g6_clone(input_tensor: ttnn.Tensor) -> ttnn.Tensor`. Rank is 1–4 and placement is DRAM interleaved. There is no `dtype`, `memory_config` or `compute_kernel_config` parameter; the spec forbids all three.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: none. The spec forbids alignment, rank and size axes; every rank 1–4 and every alignment is accepted.
- **SUPPORTED op-specific axes**: none
- **Regime**: `flat_page_range`. The flat page axis is split into contiguous ranges over the full grid (`split_work_to_cores(row_wise=True)`, 110 cores). Each core runs one block, behind one read barrier and one write flush. There is no compute kernel.
- **Cores**: full grid, `min(tensor_pages, 110)`.
- **Compute config**: none (no compute kernel)
- **Golden baseline**: 7/7 SUPPORTED cells pass, 28 xfail_expected, 3/3 regression cases pass. Every loud category is 0.
- **Perf**: p1 runs at 63.2 µs device kernel time (stock 68.32 µs, roofline 54.61 µs).

### [x] Refinement 1 — Full TARGET surface (bfloat16, int32, uint32; ROW_MAJOR) with the logical-rows partial-tile regime

**Goal**:
- Add `ttnn.bfloat16`, `ttnn.int32` and `ttnn.uint32` to `SUPPORTED["dtype"]`, and `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]`. This closes `TARGET − SUPPORTED` entirely and moves all 28 `xfail_expected` cells to passing:
  - the 14 bf16 TILE INPUTS cells;
  - the 3 bf16 ROW_MAJOR cells (Q3, Q7, Q14: rows of 5120 B);
  - the 8 int32/uint32 ROW_MAJOR cells (rows of 4–128 B on a 64 B aligned page);
  - the loose cases p3, p4 and `poisoned_padding_nan_bf16`.
  - A verifier probe (`probes/probe_006.py`) already ran all 28 bit-exact with SUPPORTED widened and **no kernel change**: the byte copy is dtype- and layout-agnostic, and the CB page is `buffer_aligned_page_size()`.
- In the same pass, build the design's deferred `logical_rows_partial_tile` regime (`op_design.md` → Regimes; `l1_ledger.md` → deferral expiry). For a TILE input with `H mod 32 ≠ 0` or `W mod 32 ≠ 0`, move only the logical face-row pieces of each partial page:
  - row `r` of the last tile row lies in faces {0,1} or {2,3} at face row `r mod 16`;
  - faces 1 and 3 are dropped when the last tile column holds ≤ 16 logical columns.
  - Full pages keep the whole-page transfer.
  - It uses the same split, the same CB page and the same block loop. Only the per-page transfer inside `load_block` / `store_block` changes, selected by a CT flag the host sets from the predicate.
- Why it is in scope now: adding bf16 makes it the cheapest-traffic split for Q8 `[1,1,1,62080]` (148 calls per Qwen pass) and F13 `[1,5120]`. Q8 now takes 19.4 µs moving 3.97 MB each way to deliver 124 KB of logical data.

**Implementation skill**: /memory-layouts, /numeric-formats-metal

**Verifier notes**:
- **Skill use.**
  - `/memory-layouts` covers the ROW_MAJOR data-movement path and the partial-tile rule.
  - `/numeric-formats-metal` applies **only** to the dtype list and the CB `data_format`. Do **not** expose `compute_kernel_config` or add a compute kernel. The spec forbids both: there is no arithmetic, and an engine path breaks the bit-exact rule.
- **Native kernel support.** The ROW_MAJOR and partial-tile handling is native in the reader and writer. `ttnn.to_layout`, `ttnn.tilize`, `ttnn.untilize`, `ttnn.pad` and `ttnn.slice` are forbidden outright by the spec's MUST rules. No wrapper fallback exists for this op.
- **Cache key (BB-07) must grow.** The logical-rows descriptor depends on the logical `H mod 32` and `W mod 32` and on which page ids are partial (`h_tiles`, `w_tiles`). Today's key cannot tell `[1,1,1,62080]` from `[1,1,32,62080]`: both have 1940 pages of 2048 B. Add the logical last-two dims, or the derived partial extents, to `_cache_key`. `g6_clone_plan` must report the selected regime (BB-08).
- **Alignment.** Pieces are 32 B (bf16) or 64 B (f32/int32/uint32). Blackhole DRAM reads align to 64 B (`op_design.md` → Mechanism caps). Land each piece at its own in-page offset in the CB slot, so source and destination share their offset mod 64. If the NoC rejects sub-64 B DRAM reads at odd face rows, widen each piece to its enclosing 64 B span. That is still ≤ 1/16 of a bf16 page for one logical row.
- **Poisoned padding.** The output's padding may hold anything. The input's padding must never reach a logical element. `poisoned_padding_nan_bf16` (Q8) and `poisoned_padding_nan_f32` (`[1,1]`) are the gate.
- **Ordering.** The perf-1 contract (p1: f32 TILE DRAM-interleaved, no compute config) has been in SUPPORTED since Phase 0, so this refinement is not a prerequisite for Refinement 2. It goes first because it is the only generality work, and because Refinement 2's no-regression guard set includes the RM and logical-rows paths this creates.
- **Blocking class.** Knob-turns throughout: the axis lists, plus a per-page transfer shape inside an unchanged block schedule. No new topology.

**Done when**:
- All 28 cells listed above pass in the golden suite, and `verify_supported` shows `supported_fail = xpass_drift = xfail_wrong_mode = 0`.
- The logical-rows regime runs on Q8 and F13. Device-ns on `[1,1,1,62080]` bf16 is measured below its 19.4 µs `flat_page_range` baseline, and on `[1,5120]` bf16 below 2.35 µs.
- p1, p2 and p4 (full-tile paths) do not regress beyond noise.

**Outcome**: Golden 38/38 bit-exact, and SUPPORTED equals TARGET. `logical_rows_partial_tile` runs Q8 `[1,1,1,62080]` bf16 at 7.06 µs (flat 19.44 µs) and F13 `[1,5120]` bf16 at 1.59 µs (flat 2.39 µs), with one-row bf16 ranges merged across the 448 B face gap. p1 63.7, p2 32.2 and p4 19.2 µs are unchanged within noise. Q8 is now bound by read requests: read issue + barrier take up to 2.3 + 4.6 µs on the slowest core, against 0.8–1.2 µs for write issue. 32 B ranges do not help, and merging wider gaps loses. What is left on Q8 is the DRAM bank and NoC route levers of Refinements 2–3, not a transfer-shape change.

### [ ] Refinement 2 — Speed up p1: DRAM bank locality

**Type**: perf

**Goal**:
- **Target.** `feature_spec.LOOSE_CASES` flags **p1** `(16, 12, 128, 128)` float32 TILE, DRAM-interleaved in and out, `gate: exact`, as the mandatory perf target (`attention: PERF FOCUS`). Its goal is a kernel time below the stock 68.32 µs and ≤ 1.35 × the 54.61 µs roofline. Phase 0 already meets both at **63.2 µs**. The remaining headroom is the roofline itself (1.16×): p1 moves 25.17 MB at about 396 GB/s, which is 77% of the p150b's 512 GB/s DRAM peak.
- **Levers already measured null.** The verifier's probes show the core count and the block size are not the bound. 44 to 110 cores run at 63.8–64.4 µs (`probes/grid_cap_probe.py`). 1 to 7 blocks per core at depth 2 are flat (`probes/block_knob_probe.py`).
- **Where the remaining time goes.** Every core walks all 8 DRAM banks (page `p` lives in bank `p mod 8`), and every 28-page range starts on bank 0 or 4. That is the interleaved round-robin, maximal-route-overlap regime.
- **Levers to try**, from `ttnn/ttnn/operations/examples/master.md`:
  - `bank_stagger` (⭐⭐): a CT switch that rotates each core's walk start by `core_idx mod num_banks`.
  - `read_coalescing` bank affinity (⭐⭐): the design's deferred `bank_affine_runs` regime. Each core takes the pages of one bank (`b, b+8, b+16, …`, contiguous in bank `b` at stride `aligned_page_bytes`) and moves them as multi-page transfers. `out` has `x`'s spec, so the write side lands in the same bank.
  - Placement from `tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md`: put each bank's cores next to that bank's DRAM column so routes do not overlap.
- No SUPPORTED change.

**Done when**:
- p1 device-ns improves on the 63.2 µs baseline beyond the 2–3% noise band, measured in one fresh-cache profiled run per variant.
- p1 stays bit-exact, and the golden suite is green.
- No regression across the config-spanning guard set, one representative per path × layout:
  - p2 f32 TILE;
  - p3 bf16 TILE;
  - Q8 `[1,1,1,62080]` bf16 TILE (the logical-rows path from Refinement 1);
  - Q3 `(16,3,2560)` bf16 ROW_MAJOR;
  - one single-page int32 ROW_MAJOR case.

**Verifier notes**:
- **Design update.** The design defers `bank_affine_runs` as "transaction count only; pays only when a DM RISC is issue-bound". The grid-cap sweep falsifies that premise: p1 is not issue-bound and still sits at 77% of DRAM peak. Update that Regimes row with the measured reason once the lever lands.
- **Contract rules.** Any new assignment must keep each kernel group to a few rectangular core ranges, choosing which core gets which bank rather than scattering groups (BB-17). It must be reported by `g6_clone_plan` (BB-08). A selected regime belongs in the cache key (BB-07). If bank affinity is gated to shapes with enough pages per bank, keep `flat_page_range` as the fallback and pin both with a plan assertion.
- **Ordering.** Runs after Refinement 1 only so that the guard set exists; it has no code dependency on it.

### [ ] Refinement 3 — Speed up p1: NoC route contention after bank locality

**Type**: perf

**Goal**:
- **Target.** The same flagged case, **p1** `(16, 12, 128, 128)` f32 TILE, `gate: exact`, measured from Refinement 2's landed time.
- **Lever.** Once each core touches one bank, the residual DRAM contention is request-route overlap. Two readers share a row toward the same DRAM column, and all reads ride NoC0 while all writes ride NoC1. Levers:
  - assign NoC virtual channels per core within a row (the tech report's fix for two readers per row);
  - split each direction's traffic across both NoCs where routes overlap. `/perf-ceiling-dm` Step 4b models this as roughly halving the per-link route count, capped at DRAM peak.
- Measure against the `noc_placement` (⭐⭐) finding: reads on NoC0 and writes on NoC1 is the right *default*. The question is whether a partial split beats it once bank locality removes the round-robin.
- No SUPPORTED change.

**Done when**:
- p1 device-ns improves on Refinement 2's landed time beyond noise, or the lever is measured null and recorded in the `**Outcome**` line, with the bottleneck named.
- p1 stays bit-exact, and the golden suite is green.
- No regression on Refinement 2's guard set.

**Verifier notes**:
- **Headroom gate.** If Refinement 2 lands p1 within about 1.05× of 54.61 µs (461 GB/s, 90% of the DRAM spec, close to the tech report's 92% practical ceiling), the residual is at the ceiling. Close this refinement as a measured null rather than forcing a lever.
- **Kept out of Refinement 2 on purpose.** NoC selection and VC assignment interact with the bank-locality assignment. They can only be measured meaningfully on top of it, and the two together are more than one perf phase.
