# Verification Report: paged_scaled_dot_product_attention_decode

Target: Quasar (`quasar`) on the ttsim simulator (no clock: every perf statement below is structural). Specification: `eval/specs/paged_scaled_dot_product_attention_decode/spec.py`. Design: `op_design.md`. Ledger: `l1_ledger.md`.

## Code Review

Fixed in place:

1. **Writer stored partials one tile per barrier** (`kernels/..._writer.cpp`). The design budgets "one barrier + one semaphore inc per writer hart" for `store_partial_block`; the implementation did wait(1) → write → barrier → pop(1) per tile, in rotation over its NEOs, i.e. `C/W · (tensor_d_tiles + 2)` barriers per writer hart per user. On Quasar DM, `wait_front(n)` / `pop_front(n)` act on the current tile counter and `pop_front` advances to the next NEO's counter (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl`, `wait_front_impl` / `pop_front_impl`, `dfb_dm_advance_slot`), and `cb_partial_out` holds exactly one partial per NEO. So the writer now waits for one NEO's whole partial, issues its `tensor_d_tiles + 2` writes (offsets `i · get_stride_size()`, the counter's own entries), barriers once and pops the partial: **barriers per writer hart per user on the partials ring: 2·(Dt+2) → 2** (8 → 2 at D = 64). No deadlock risk: each NEO's ring holds its whole partial, so no NEO waits on the writer's rotation. Verified at plans (1,1,1), (2,2,1) (one hart serving two NEOs), (2,2,2), (4,4,2), on one and on eight clusters. The output ring stays one tile per barrier: `d_share_tiles` = 1 for every D ≤ 128 and a pop is what moves the hart to the next NEO, so there is nothing to batch.
2. **Rule R8 — the default was restated** (`_resolve_compute_config`): the `getattr` fallbacks hardcoded `HiFi4` / `True` / `False`, a second copy of `default_compute_kernel_config()`'s values. They now read the fields of `default_compute_kernel_config()`, the one place the default lives.
3. **Ledger** updated for (1), and the `B` row of its symbol table now states its (absent) bound instead of a dash.

Tests added:

- `test_..._precision_baseline.py` — 4 shapes × 2 accumulators, PCC / abs / relative RMS / got-true ratio spread (below).
- `test_..._callers.py` — rules R10 and R11, which no golden cell exercises (the runner builds a `ComputeConfigDescriptor` and passes no program config): Llama's per-head call with its device config (`ttnn.WormholeComputeKernelConfig`, HiFi2, fp32_dest_acc_en=False, packer_l1_acc=False) and an `SDPAProgramConfig`; the captured call with its 8×8-grid program config on this 32-core device; Qwen's two-head call with its program config. All 3 pass the specification's own check.

Deferred to refinements (architectural, each named in `op_requirements.md`): the wide-D regime (Refinement 1), K/V reads per barrier (Refinement 3), read-once q (Refinement 4), combine cost (Refinement 5).

Reviewed and accepted (no change):

- **Raw LLK where the design rejected helpers** (score/PV `matmul_tiles` with stride-multiplied indices, `reduce_tile<MAX>`, the stats DEST section, `read_tile_value`): each has its Quasar reason in the compute kernel's header and in the design's "Helpers considered and rejected" table; mechanism, not a finding (scope boundary). The rowsum, scale, exp, merge, β-product, normalize and emit steps are kernel-library helpers.
- **Implementation deviations from the design** (all in the implementer's changelog with file:line evidence): every reader ring explicit-sync (implicit sync's ISR batches across producer harts deadlock when per-NEO streams differ in length); 8 compute-owned rings instead of 10 (two Tensix-only tile counters per intra-NEO ring); neutral initial state instead of a first-chunk special case; DEST banks primed with a datacopy; ALU-format latch dropped before Float32 eltwise multiplies; `wait_min` + `set(0)` instead of `down()`; one semaphore per combiner reader hart; partial order [m, O share, l]; a local `ScaleByScalar` chain element. Each is forced by Quasar / ttsim and keeps the design's math, live sets, and synchronization frequency, except (1) above which this pass fixed.
- **Reader block shape**: K one tile-row (`tensor_d_tiles` reads) per barrier, V one d-sub-block per barrier, into rings of 2 tile-rows. The design plans K/V as *streams* (`kv_depth_tiles` = 2 rows, catalog `double_buffer`), so this is the planned schedule, not an expression failure; at D = 64 it is 2 reads in flight, below the catalog's 4–8 knee — Refinement 3.
- **Height-sharded q read through its `TensorAccessor`**: not the local-shard re-read anti-pattern. q is reuse-shared by every worker of a user (≥ 4 NEOs, up to 32), the workers are not the shard's core, and rule R12 / the spec's GUIDANCE require the same read for any shard grid and for interleaved q. The design's `R-q-in-place` row is deferred with that reason; the read-once broadcast is Refinement 4.

### Specification rules

| Rule | Status |
|------|--------|
| R1 mask ends at cur_pos | Last-tile column mask built by the reader from `cur_pos % 32`; later tiles never read. Loose cases at positions 0, 31, 32, 64, 255 pass. |
| R2 page table, blocks of several tiles | `block_of(t)` per tile, `kv_page` with `block_pos_tiles`; random permutations in every cell. |
| R3 head groups | Row masks per head segment (`row / group_size == head`). |
| R4 scale before mask | q' = q · scale before QK (any sign); 0.2 and 0.0625 entries pass. |
| R5 users independent | Per-user loop, slots `u · max_workers_per_user + w`; batched cells pass in the probe (Refinement 1). |
| R6 device values unchecked, no out-of-range read | cur_pos and every page-table entry clamped on the device. |
| R7 float32 + 16-bit DEST refused | `UnsupportedAxisValue` (a `NotImplementedError`). |
| R8 one default factory | Fixed (Code Review 2). |
| R9 no gate on fidelity / approx | Not gated; HiFi2 loose case passes. |
| R10 device config | Read by attribute; covered by the new callers test. |
| R11 program config is a hint | Not read; covered by the new callers test (8×8 grid on 32 cores). |
| R12 q placement via `memory_axes` | Gated by name; one accessor read for every placement. |
| R13 caches / table / positions placement | `_check_placements`, `NotImplementedError` naming the parameter. |
| R14 output in DRAM | Allocated from a DRAM `TensorSpec`; the check hook reads it in every cell. |
| R15 sliding window refused | X1 passes. |
| R16 max subtracted | Running row max; `exp(S − m_new)`, finite `NEG_BIG` / `NEG_EMPTY`. |
| R17 padding rows | Rows ≥ NH belong to no group (masked); probe cells with 8/16/20 heads pass. |
| R18 any number of cores | Grid queried; `_MAX_CLUSTERS_PER_USER = 1` runs every call on one cluster (debug and thread tests). |
| R19 refusals | Every cell outside SUPPORTED is `xfail_expected`, 0 wrong mode. |

GUIDANCE advisories: none unfollowed that apply (the DEST budget is sized against the fp32 limit; the output is allocated from a spec carrying the config; the row max is taken of scaled scores).

### Design conformance

- **Algorithm**: online softmax (flash-decode) over a virtual key sequence, partials merged by an associative combine — as designed; no full score matrix.
- **Work distribution**: √N worker rule over `G_u` clusters × C NEOs, users dealt row-wise; `LLAMA_CAPTURED` engages 8 clusters × 4 NEOs = 32 workers (thread test). The traced small calls engage one cluster by the same rule (8 units, the merge cost outweighs more workers) — a design decision with its perf lamp (κ), not idle silicon by accident.
- **Thread plan**: (R, C, W) = (4, 4, 2) from the one constant `THREAD_PLAN`, counts as CT args, `validate_thread_plan` enforces C ∈ {1,2,4}, R + W ≤ 6, R | C, W | C, R == C; every ring's `num_entries` = per-NEO capacity × C; explicit-sync rings carry `disable_dfb_implicit_sync_for_all` and every transfer trid 0; each DM kernel hands the tag back last. The every-thread-ran test (`test_every_worker_wrote_its_own_partial`: each slot holds that worker's own row max, at (4,4,2), with empty workers, at (1,1,1), and on 8 clusters) and the one-session test (another program, the op twice, then `ttnn.to_layout`) are present and pass. Plan alternatives were not measured: ttsim has no timing model and no reference figure exists for this phase (`pipeline_logs/model_cycles.jsonl` absent); the design's bound (compute: two matmuls, softmax and merges per unit) supports C = 4.
- **Blocking-model fidelity**: every knob is one host constant / function passed as a CT arg (`k_chunk_tiles`, `KV_DEPTH_ROWS`, `MASK_DEPTH_SEGMENTS`, `PARTIAL_DEPTH_PARTIALS`, `MERGE_COST_UNITS`, `THREAD_PLAN`, `PARTIALS_MEMORY`, `L1_RESERVE_BYTES`); ring capacities come from one list (`_rings`) that also drives the L1 predicate. `k_chunk_tiles` = the whole per-worker assignment at N_max (the coarsest block that fits), lowered only by the predicate. `clusters_for` exists on host and device by necessity (the host sizes for N_max, the device decides from cur_pos) and both read the same CT inputs; `DEST_TILES` mirrors `DEST_AUTO_LIMIT` and the kernel `static_assert`s the derived sub-block against it. No CB scales unconditionally with a whole-op dimension except the D-spanning resident state (q', O), which the L1 predicate guards and Refinement 1 blocks.

## Registry Conformance

- `INPUT_TAGGERS = AXES` imported from the specification; no op-local axis function.
- `SUPPORTED` lists every gated axis (dtype, fp32_dest_acc_en, layout, memory_config, sharding, output_memory, kv_heads_mode, batch, alignment) and equals `PHASE0` (`python3 -m eval.validate_contract … --phase0`: PASSED, 0 errors).
- `EXCLUSIONS = []`. `validate()` checks rejections (X1–X10), then SUPPORTED per axis, then EXCLUSIONS, then R13 placements; it is the entry point's first call. The op file declares no `INVALID`.
- No XPASS drift, so no SUPPORTED auto-fix. (The probe below shows most of TARGET already runs; SUPPORTED must equal PHASE0 at this stage, so the widening is the queue's.)
- **INVALID audit (specification)**: one entry, `{"memory_config": "DRAM", "sharding": "height"}` — a single tensor (q), structural (no DRAM height-sharded tensor is built). No cross-tensor coupling. The canonical bf8b + ROW_MAJOR entry does not apply: TARGET has neither (TILE only; Quasar has no bfloat8_b path). Not a norm op (no weight canonicalization). Well-formed.

## L1 Ledger Audit

- **Currency**: 17 rings in `_rings()` = 17 ledger rows; every capacity expression matches the code (q_tiles `Dt`, k/v `2·Dt`, masks 4, partials_in `2·(2+d_share)`, q_scaled `Dt`, scores/probs `k_chunk_tiles`, four 2-tile state rings, out_acc `Dt`, step `Dt+1`, partial_out `Dt+2`, output 2); the footprint formula matches `_ring_bytes`. Writer note updated (fix 1).
- **Capacity vs live set**: every gap is a stated window (double buffer, mask prefetch, partial prefetch, pop/push overlap). `cb_cur_pos` is a 4 KiB Int32 tile for one word — reasoned (TEN-4746 needs a real unpack before the pop). Under (spanning without scaling): none; the K/V windows correctly stream over the position axis.
- **Page format vs DEST**: I rings follow `fp32_dest_acc_en`. Two 16-bit pages under fp32 DEST — `cb_q_scaled` and `cb_probs` — carry a mechanism reason (matmul operands must match K/V's format, `matmul.h:99-102`); the precision baseline shows the cost is inside the tolerance (rel-RMS ≤ 0.0026 with fp32 DEST). Float32 operands are the design's perf lamp "fp32 matmul operands"; with float32 inputs (Refinement 2) they become Float32 anyway. No Float32 page under 16-bit DEST.
- **Disjoint lifetimes**: phase-1/phase-2 pairs share rings by role (`col_step`, `step`, `out_acc`); `partials_in` vs K/V justified (format and protocol differ). No silent pair.
- **Bounds**: every symbol has a bound and its predicate; B had none (now stated: only the positions page grows, 4 B per user).
- **Block-size defaults**: held — interleaved-style split spread over clusters by the √N rule, then the coarsest chunk that fits. Not departed from.
- **Per-core footprint**: `C·per_neo(Dt, d_share, k_chunk_tiles) + R·(pos + table pages) + W·pos page + partials_bank_share`; `Dt` scales q/K/V/q'/O/step/partial rings, `k_chunk_tiles` scales scores/probs, C multiplies all. 34 % of budget at `LLAMA_CAPTURED`; over budget at Dt = 128 (Refinement 1).
- **Data-movement budget**: present and consistent with the code: K and V cross DRAM once (attended tiles only), q W× from the shard, partials in L1. Cheapest-traffic split recorded (the built one); deferred rows (`R-tree`, `R-head-aligned`, `R-q-reuse`, `R-q-in-place`, `R-wide-D`) each carry a positive reason. `R-q-reuse` is filed as Refinement 4 (reuse-shared operand, deferred broadcast row).

## Precision Baseline

`test_paged_scaled_dot_product_attention_decode_precision_baseline.py`, bfloat16, HiFi4, a fixed position near the end of each user's range, seed 1234. Ratio = got/true over elements with |true| > 0.05·std.

| Shape | Position | fp32 DEST | PCC | Max Abs Err | Mean Abs Err | Relative RMS | Ratio p5 / median / p95 |
|-------|----------|-----------|-----|-------------|--------------|--------------|--------------------------|
| ONE_TILE_WIDE (D 32, 64 pos) | 63 | yes | 0.999995 | 0.0031 | 0.00046 | 0.0032 | 0.989 / 0.999 / 1.011 |
| ONE_TILE_WIDE | 63 | no | 0.999977 | 0.0086 | 0.00103 | 0.0075 | 0.974 / 0.996 / 1.017 |
| LLAMA_SPLIT (D 64, 256 pos) | 250 | yes | 0.999997 | 0.0012 | 0.00021 | 0.0025 | 0.991 / 1.000 / 1.009 |
| LLAMA_SPLIT | 250 | no | 0.999967 | 0.0057 | 0.00067 | 0.0081 | 0.971 / 0.999 / 1.028 |
| QWEN_TWO_KV_HEADS (D 128, 256 pos) | 250 | yes | 0.999996 | 0.0019 | 0.00024 | 0.0029 | 0.989 / 1.000 / 1.011 |
| QWEN_TWO_KV_HEADS | 250 | no | 0.999952 | 0.0067 | 0.00083 | 0.0104 | 0.959 / 0.996 / 1.031 |
| LLAMA_CAPTURED (D 64, 4096 pos, 8 clusters) | 4000 | yes | 0.999997 | 0.0003 | 0.00005 | 0.0026 | 0.990 / 0.999 / 1.008 |
| LLAMA_CAPTURED | 4000 | no | 0.999915 | 0.0019 | 0.00027 | 0.0132 | 0.950 / 0.997 / 1.053 |

**Assessment**: near the bfloat16 rounding floor with fp32 DEST (the spec notes 0.0017 rel-RMS for float64 rounded once to bf16); the 16-bit accumulator costs 3–5× in rel-RMS, still 10× under its limit, growing with the number of positions merged. Ratio medians sit at 1.000 ± 0.004 with symmetric spread: no scale/structural error. Probe (SUPPORTED widened in-process, golden checks): float32 cells PCC ≥ 0.999999, rel-RMS ≤ 0.0026 (limit 0.01); all bf16 cells rel-RMS ≤ 0.0122.
**Recommended tolerances**: defaults hold (`NUMERIC["legacy_tolerances"]`: bf16 (0.99, 0.05) / (0.99, 0.13), float32 (0.999, 0.01)).

## Verifier CLI Summary

Golden results: `/tmp/psdpa_golden2` (report copied to `verifier_report.json` in this directory).

- supported_pass: 23 (14 PHASE0 cells + 9 loose cases)
- xfail_expected: 178
- invalid_skipped: 64
- no_axes_found: 10 (the REJECTIONS X1–X10, all passing)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- queue_gaps: 0 (`--queue op_requirements.md`)

Local tests: acceptance 7/7, debug matrix 16/16, thread + session 5/5, precision baseline 8/8, callers 3/3.

## TARGET probe (verifier evidence for the queue)

With `SUPPORTED = TARGET` set in-process (not committed) and every cell graded by the specification's check, 135 / 144 cells pass on the Phase 0 kernels (16 INPUTS × {L1/height, DRAM/interleaved, L1/interleaved} × {bf16 fp32-DEST, bf16 16-bit DEST, float32}). The 9 failures are all the D = 4096 batched entry: `NotImplementedError` from the host L1 predicate (wide-D regime not built). So the `xfail_expected` bucket is almost entirely a SUPPORTED-widening gap, filed as Refinements 1–2; the wide-D regime is Refinement 1's real work.

## Recommendations

- Queue order: Refinement 1 (shape axes + wide-D, the only new structure) → 2 (interleaved q + float32) → 3 perf (K/V reads per barrier, 2 → ≥ 8 at `LLAMA_CAPTURED`) → 4 perf (q read once per cluster, C·Dt → Dt) → 5 perf (combine cost). No `LOOSE_CASES` entry is perf-flagged, so the perf target is free-selected: `LLAMA_CAPTURED`, the only supported call with real data volume (4 MiB K + 4 MiB V) and the one that engages 8 clusters.
- On a timed backend (`ttsim-perf`), measure the design's perf lamps that ttsim cannot: the thread plan (4,4,2) vs (1,1,1), κ of the √N worker rule (`MERGE_COST_UNITS` ∈ {0.5, 2, 4}, `_MAX_CLUSTERS_PER_USER = 1`), and a smaller `k_chunk_tiles` for overlap. Those are findings for a timed run, not refinements here.
- Compute binary size: the compute kernel is built at `-Os` to fit 24 KiB. Refinements 1 and 5 add compute code; watch the limit (and clear the JIT cache when `opt_level` changes).
- The positions page is read whole by every DM hart (4 B × B); fine up to large B, but a per-user element read would remove the scratchpad's dependence on B.
