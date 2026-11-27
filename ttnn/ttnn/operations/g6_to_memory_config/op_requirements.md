# Operation Requirements: g6_to_memory_config

## Definition
- **Formula**: `y[i] = x[i]` for every logical index `i`; `shape(y) = shape(x)`, `dtype(y) = dtype(x)`, `layout(y) = TILE_LAYOUT`, `placement(y) = memory_config`. Per tile page: `out_page[p] = in_page[p]`, bit-identical in every dtype (block formats included).
- **PyTorch Reference**: `def pytorch_g6_to_memory_config(x, memory_config=None): return x.clone()`
- **Import Path**: `from ttnn.operations.g6_to_memory_config import g6_to_memory_config`
- **Function Signature**: `g6_to_memory_config(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor` (both positional; no dtype, layout or compute-config parameter). Companion: `plan_of(input_tensor, memory_config) -> dict` (BB-08 plan report).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: rank ∈ {2, 3, 4}; alignment ∈ {tile_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: in_scheme / out_scheme ∈ {interleaved, height_sharded, width_sharded, block_sharded}; in_buffer / out_buffer ∈ {dram, l1}
- **EXCLUSIONS**: `{"out_scheme": "width_sharded", "out_buffer": "dram"}` (spec-mandated Phase 0 entry)
- **Regimes**: `pull_shard`, `pull_bank`, `push_shard`, `push_bank`, `bounce` (owner moves; one NoC transfer per page when one endpoint is the mover's L1, two for DRAM→DRAM)
- **Cores**: bank and bounce regimes use the full 110-core grid; shard regimes use every core owning a shard of the moved side
- **Compute config**: none (no compute kernel, no arithmetic)
- **Golden baseline**: 91 / 117 registry cells pass (86 bf16 production cells + 5 Phase 0 loose cases); 26 xfail_expected; 0 loud. Plus 88/88 regression tests. 179 / 205 total.

### [x] Refinement 1 — Block formats and DRAM width-sharded targets

**Goal**: add `ttnn.bfloat8_b` and `ttnn.bfloat4_b` to `SUPPORTED["dtype"]` and remove the `{"out_scheme": "width_sharded", "out_buffer": "dram"}` entry from `EXCLUSIONS`. After this, `SUPPORTED == TARGET` on every axis. This moves all 26 current `xfail_expected` cells to passing:
- 12 bf8b production cells (Panoptic 0/2/5/7/9/11/15/25/26/28/34/36);
- 7 bf4b weight cells (Qwen 2178–2184, DRAM interleaved → DRAM width-sharded);
- 7 loose cases: PERF FOCUS 2183, p2 2184, p3 2182, p4 2179, Panoptic 7, and both outlier cases.

It lands the full config contract of the perf-flagged case (`LOOSE_CASES[0]`, Qwen 2183: bf4b, TILE, DRAM interleaved → DRAM width-sharded [5120, 2048] on 8 DRAM banks, exact gate) in SUPPORTED, ahead of Refinement 2.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Only part of the skill applies.** Use the dtype-axis expansion, the CB data-format derivation (`cb_bounce_pages` takes the input dtype; `page_size` from `buffer_aligned_page_size()` is the only dtype-dependent quantity) and the precision-matrix test contract. The spec forbids a `compute_kernel_config` (no arithmetic), so the skill's DEST/fidelity part does not apply. Grading is bit-equality in every dtype (`helpers.TOLERANCES` is `"exact"`), so the precision matrix reduces to `torch.equal` per dtype.
- **Kernel changes are expected to be zero.** The verifier lifted the gate in-process and ran all 19 block-format golden cells through the unchanged Phase 0 kernels: all bit-exact with exact placement (`probes/probe_013.py`). The bf8b cells land on `push_shard` / `pull_bank`; the bf4b cells land on `bounce` with 110 movers.
- **BB-05.** The bounce CB is 429696 B on 2183, under the 1347776 B budget. Re-check `plan_of(...)["l1_op_bytes"]` for every new cell.
- **Removing the EXCLUSION is a placement change** (`memory_layout` on DRAM). No skill covers placement. The existing `bounce` regime already moves DRAM width-sharded targets exactly through the target `TensorAccessor`, so no wrapper and no `to_memory_config` fallback is permitted or needed.
- **Acceptance test update (required).** `test_refused_cells` asserts that bf8b Panoptic 11 raises `UnsupportedAxisValue` and that bf16 Qwen 2178 raises `ExcludedCell`. Both stop raising here. Re-target the refusal checks to values outside TARGET (e.g. `float32` dtype → `UnsupportedAxisValue`; no `ExcludedCell` remains once EXCLUSIONS is empty). Add positive regime-pinned production cases: Panoptic 11 bf8b (`push_shard`) and Qwen 2178 bf4b (`bounce`). Extend `test_g6_to_memory_config_precision_baseline.py` with one bf8b case and one bf4b case (exact).
- **Performance bar.** The new path must keep the measured Phase 0 behaviour: full grid, block-batched reads and writes. The 2183 baseline is 307–318 µs device kernel time (`verification_report.md`, Device Performance Baseline); do not regress it. The coalesced-write regime (`bounce_dram_shard`) is **not** part of this refinement. It is Refinement 2's measured lever, so leave `bounce` as is here.

**Done when**: all 26 current `xfail_expected` cells (19 golden + 7 loose) pass bit-exact with exact placement, `verify_supported` reports `xpass_drift = xfail_wrong_mode = supported_fail = 0`, the updated acceptance test and the extended precision baseline are green, and Qwen 2183's device kernel duration is not regressed beyond noise (≤ ~318 µs).

### [x] Refinement 2 — Speed up the perf-flagged profile: coalesce writes into the DRAM width-sharded target

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags Qwen 2183 as the mandatory perf target: bf4b (5120, 16384), DRAM interleaved → DRAM width-sharded [5120, 2048] on 8 banks, 81920 pages of 576 B, gate `"exact"`, seed 1234. The lead's goal is device kernel duration ≤ 276.48 µs (1.35 × the 204.80 µs DRAM roofline) and below stock's 348.36 µs. Today it is 307–318 µs on `bounce`: 1.50× roofline, beating stock but ~11% over the bound.

The measured signature is transaction size. bf16 bounce with 2 KB pages runs at 1.11× roofline; bf4b bounce with 576 B pages runs at 1.48–1.55×. The writer issues one 576 B NoC write per page into a target whose shard rows are contiguous runs of 64 pages (36 KB) in one bank.

Relevant levers in `ttnn/ttnn/operations/examples/master.md`:
- transfer coalescing (`read_coalescing`, `tile_reorder`), applied to the write side. This is the design's deferred `bounce_dram_shard` row: assignment over the target's storage order, and one write per run contiguous inside a target shard row instead of one per page;
- `bank_stagger` for 110 writers sharing 8 DRAM banks;
- the bounce block/depth co-tune (`double_buffer`, `trid_double_issue`). Measured: `bounce_depth` 16 vs 2 is −4% on 2183, so depth alone is not enough.

Co-measure the other bf4b perf cases on the same path: p2 2184 (250 µs), p3 2182 (61 µs), p4 2179 (171 µs). No SUPPORTED change.

**Verifier notes**:
- **Blocking-model class: knob-turn on the `bounce` regime.** The design states the new row "changes only the host assignment order and the writer's transfer unit". It adds no cross-core communication. If it becomes a new regime, `plan_of` must report it (BB-08), the acceptance test gets a regime pin for it, and the cache key already covers the target placement.
- **Ragged shards.** Qwen 2184 has an uneven last DRAM shard (404 tile columns over 8 shards of 52), so run lengths must come from the accessor or shard spec, not from an assumed even split.
- **Hard constraints.** Pages stay whole (bit-exact bf4b), and the BB-05 clamp on the bounce CB stays in force.

**Done when**: Qwen 2183 measures ≤ 276.48 µs device kernel duration (and < 348.36 µs) with the exact gate holding; the golden suite is green; and there is no regression beyond noise across the config-spanning guard set. Guard set:
- `pull_shard`: Qwen 2067 (L1→L1) and 2074 (DRAM→L1);
- `pull_bank`: Qwen 2078 bf16 and Panoptic 34 bf8b;
- `push_shard`: Qwen 2071 (→ DRAM, ragged) and 2073 (→ L1 sharded);
- `push_bank`: L1 interleaved → DRAM interleaved on Qwen 2064's input;
- `bounce`: bf16 DRAM→DRAM interleaved (16, 128, 1536), plus bf4b Qwen 2178 / 2179 / 2182 / 2184.

**Outcome**: coalesced bounce writes (`BOUNCE_WRITE_RUN_PAGES = 0`: one NoC write per run contiguous in the target, up to a 36 KB shard row) take Qwen 2183 from 325.5 µs (per-page writes, same session, median of 3; Refinement 1 recorded 313.7 µs) to 294.8 µs (1.44× the 204.8 µs roofline), still above the 276.48 µs goal and well below stock's 348.36 µs. Also 2184 261.6 → 233.0, 2179 178.6 → 157.7, 2182 67.0 → 58.8, 2178 26.3 → 25.2 µs, bit-exact; the guard set is within noise. Measured and parked (default off, live knobs): storage-order assignment `BOUNCE_TARGET_ORDER` (each mover writes one bank) is a loss, 555 µs with row-clustered movers and 375 µs with movers interleaved over shards, because NoC1 paths to a bank saturate and both halves slow down. `BOUNCE_BANK_STAGGER` measures as noise (306.8 vs 302.2 µs), and so do `bounce_depth` 3/4/8 and write caps of 8/28 pages. Bottleneck now (2183 ablation): reads alone 165 µs, writes alone 184 µs, sync floor about 1 µs, full 295–302 µs. The aggregate read rate is flat at about 165 GB/s for the whole call, so the op is DRAM-efficiency-bound on mixed 576 B reads plus coalesced writes (bf16 with 2 KB pages reaches about 515 GB/s on the same path). Next: the read side, which Refinement 3 owns: coalesce reads from the interleaved source (pages `p, p+8, …` are contiguous in one bank), then reorder in L1 to the target order. Not done here because it is outside this heading's write-side scope.

### [x] Refinement 3 — Close the remaining gap on the flagged bf4b DRAM→DRAM profile

**Type**: perf

**Goal**: after Refinement 2, re-measure the flagged profile (Qwen 2183) and the lead's other bf4b perf cases against their 1.35× DRAM-roofline bounds (two crossings per page at 460.8 GB/s):

| Case | Bound | Phase 0 |
|---|---|---|
| p2 2184 | ≤ 218.2 µs | 250 µs, 1.55× |
| p4 2179 | ≤ 155.5 µs | 171 µs, 1.48× |
| p3 2182 | ≤ 60.5 µs | 61 µs, 1.35× |

Push every case still above its bound under it, and push 2183 further toward its 204.8 µs roofline. Once writes are coalesced, the remaining per-page cost is on the read side: 576 B reads from an interleaved source cannot coalesce, because consecutive pages sit on different banks. Relevant levers in `ttnn/ttnn/operations/examples/master.md`:
- `trid_double_issue`: reads kept on the wire across block waits;
- `double_buffer`: co-tune `bounce_depth` and `bounce_block_pages` (the depth knob measured −4% / −7% on 2183 / 2179 in Phase 0);
- `read_coalescing`: bank-affine mover assignment, which measured both gains and losses, so measure before keeping it;
- `noc_placement`: NoC selection per stream.

No SUPPORTED change.

**Verifier notes**:
- **Ordering.** This refinement depends on Refinement 2's write path. Measure on top of it, not on Phase 0 `bounce`.
- **Lever tier.** All levers are T1/T2 knob-turns on the existing reader/writer pair.
- **Headroom check.** If `/perf-ceiling-dm` shows that the remaining gap is the data-movement ceiling for 576 B interleaved reads on this board, record the measurement and close the refinement. Do not stack levers that measure as noise (≤ ~2–3%).

**Done when**: each of Qwen 2184, 2179 and 2182 measures within 1.35× of its DRAM roofline, and Qwen 2183 improves on its post-Refinement-2 number without losing the ≤ 276.48 µs bound; or a measured `/perf-ceiling-dm` result shows that the remaining gap is at the data-movement ceiling. In both cases the exact gate holds, the golden suite is green, and the Refinement 2 guard set shows no regression.

**Outcome**: staged coalesced reads (one DRAM read per bank run into `cb_bounce_stage`, then an L1 reorder to page order; gated to pages under 1024 B, i.e. bf4b) plus `BOUNCE_STAGED_DEPTH = 6` take Qwen 2183 from 294.8 to 260.2 µs (1.27× the 204.8 µs roofline), 2184 233.0 → 203.2 (≤ 218.2), 2179 157.7 → 141.3 (≤ 155.5), 2182 58.8 → 52.1 (≤ 60.5), 2178 25.2 → 22.4 µs. All are medians of 3, bit-exact; bf16 / bf8b bounce paths are unchanged within noise. Measured and parked: `BOUNCE_NOC_SWAP` (every grid subset is a loss, 278–385 µs) and stage chunks below a block (noise). Not built: trid overlap of the reorder, which is off the critical path (ablating it: 274 vs 269 µs). Bottleneck now: readers finish by 150 µs; writers bind at 259 µs max vs 189 µs median. The cause is NoC arbitration unfairness that grows with each mover's distance along shared paths (writes-only into half the DRAM banks: 15 → 103 µs per core), not DRAM bandwidth (365 of 460.8 GB/s). Next: a rate-balanced or dynamic (shared-counter) bounce assignment. Not done because it changes the work split, beyond this heading's T1/T2 knobs.
