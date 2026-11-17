# Verification Report: layer_norm_adaln

Box: Blackhole (12×10 compute grid), 2026-11-16. Golden results dir: `/tmp/lnadaln_golden` (copy of the CLI output:
`verifier_report.json` in this directory).

## Code Review

Every item below was **fixed in place**; all 41 acceptance tests, the 32-cell precision baseline and the golden
suite pass after the fixes.

| # | Finding | Fix |
|---|---------|-----|
| 1 | **Correctness (prompt MUST rule violated)** — with `fp32_dest_acc_en=False` (in `SUPPORTED`, explicitly allowed by the prompt) and a large common offset (mean 50, std 1) the output was wrong: rel-RMS 0.19 (256×1024) → 0.46 (256×3072) → **0.998** (2048×3072), PCC down to 0.84 with affine. Root cause: the first-pass mean `r` is accumulated in bf16 DEST (partial sums ≈ 5e4, ulp 256), so the residual shift `m = μ − p` was off by ~1 std. The got/true ratio spread was wide (p5..p95 ≈ −3.4..4.7), i.e. a per-row shift error, not a uniform scale bug. Neither the golden suite (TARGET has only `fp32_dest_acc_en=True`) nor the acceptance test (bf16 DEST only on zero-mean data) covered this cell. | Under 16-bit DEST only (`dev_sum_pass` CT flag): an extra `(x − p)` → L1-acc → row-reduce pass gives `d = Σ(x−p)/n` measured on O(std) deviations; the payload becomes `[p, d, M2]` (`stats_per_row(False) = 3`) and the combine forms `δ_j = (p_j − p_own) + d_j`, so no large value is ever rounded at bf16. fp32 DEST keeps the original `[r, M2]` scheme bit-for-bit (a unified 3-stat combine was tried first and measured a 32% regression at 256×3072 / S=12, so it was not kept). After: rel-RMS ≤ 0.019 on every bf16-DEST offset cell. Guarded by the new precision-baseline test (offset 50 × both DEST modes). |
| 2 | **Performance conformance** — the reader fetched the w/b row-0 slices with **one DRAM barrier per tile** (2·bw = 64 serialized round trips at bw=32), on the first block's critical path (compute's `prep_affine_block` waits on them). | Chunked: `AFFINE_STREAM_CHUNK = 8` tiles per reserve/barrier/push, CB capacity `AFFINE_STREAM_DEPTH` (2) chunks, both from one host source (`affine_stream_chunk(bw)` / `affine_stream_pages(bw)` feed the CB descriptor, the reader CT arg and `footprint_bytes`). Measured: 2304×3072 affine 122.9 → 110.8 µs, 2048×3072 affine 114.3 → 108.3 µs, 256×1024 affine 21.5 → 20.0 µs. |
| 3 | Writer waited for **DRAM write completion** (`noc_async_write_barrier`) per tile row before popping `cb_output_tiles`. | `noc_async_writes_flushed()` per row (only the L1 source must be consumed before `cb_pop_front`), one `noc_async_write_barrier()` at kernel end. |
| 4 | `GATHER_TILE_CAP` check hard-coded the payload height `2·S·bh`. | Uses `stats_per_row(fp32_dest_acc_en)·S·bh`; `stats_per_row` is the single source for the CB sizes, the writer payload CT arg, the compute payload layout (static_asserted) and the cap. No INPUTS selection changes in either mode (pinned test still passes). |

Checked and found conformant (no change needed):
- Kernel syntax `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, `TensorAccessor` everywhere (no `InterleavedAddrGen`).
- CB push/pop balance per block on every CB (reader↔compute: `bh·bw` pushed per row, popped `AtEnd` of normalize; stats CBs `spr·bh` / `spr·S·bh`; output `bw` per row on both sides; call-resident `cb_weight_full`/`cb_bias_full`/`cb_reduce_scaler` popped at kernel end).
- Helpers: `reduce` (mean, dev_sq / dev_sum row reduce), `eltwise_chain` (pivot, dev_sq / dev_sum L1-acc, normalize with DestReuse), `unary_bcast<Row/Col>`, `ttnn.Mcast` rotating sender + `McastArgs` for the all-gather (no raw semaphore code). The two raw-LLK custom block ops (`publish_stats_block`, `combine_block`) are justified in the kernel header (runtime-count fold over S peers with 4 live DEST slots; multi-slot pack into one CB) — mechanism, not flagged.
- Broadcast dims: `x − p` Col-bcast (p is a REDUCE_ROW col-0 tile), stats Col-bcast, w/b Row-bcast of row 0 — correct; no full-tile fills of repeated data (the Row/Col pre-broadcasts are required because `DestReuseBinary` has no broadcast field).

### Prompt rules
| Rule | Status |
|------|--------|
| MUST handle TILE natively; MUST NOT use to_layout/tilize/untilize | ✓ — entry point only validates, allocates, builds one descriptor, one `ttnn.generic_op`. |
| MUST be stable for large common offset (mean ≫ std) | ✓ **after fix #1** (was violated under `fp32_dest_acc_en=False`). fp32 DEST: offset-50 rel-RMS 0.0037. |
| MUST NOT cache / precompute from w, b across calls | ✓ — addresses are per-call RT args; Row-bcast is in-kernel per call; `test_weight_bias_change_every_call` passes. |
| One native dispatch per invocation | ✓ — single `generic_op`. |

## Design Conformance

- **Algorithm**: pivot-shifted two-pass + Chan combine, as designed. Deviations: `cb_dev_sq` (reduce helper forbids in==out; implementer, recorded in the ledger); `cb_row_stats` not `UnpackToDestFp32` (its consumer is `unary_bcast<Col>` on the FPU path — correct); the 16-bit-DEST payload change (fix #1, recorded as an amendment at the end of `op_design.md`).
- **Work distribution**: as designed; fills 120/120 cores at 2048/2304×3072, 96 at 256×3072, 12 at 32×3072. Both dataflow halves are batched per tile row (reader one barrier per row; writer flush per row).
- **Blocking-model fidelity**: `block_height`, `block_width`, `num_w_slices`, depths and the grid are parameters defined once (`select_blocking` + host constants) and passed as CT/RT args; every CB size and trip count derives from them. No CB scales with a whole-op dimension (`cb_input_tiles` = `INPUT_BLOCK_DEPTH·bh·bw` under the L1 predicate). The CB *slot indices* are restated as literals in host and kernels — those are buffer ids, not block values; left as is.
- **Expression**: one deviation from the planned schedule (not a defect, filed as a perf lever): `mean_block` uses `WaitUpfrontNoPop`, i.e. it waits for the **whole** block before reducing, where the design said "wait cumulative per row". With one block per core this serializes the core's entire DRAM read in front of its compute (perf lamp "Overlap (input)") — folded into Refinement 1.

## Registry Conformance

- `INPUT_TAGGERS` (`alignment`, `rank`, both `(inputs, axes)`), `SUPPORTED` (all 9 TARGET axes + `fp32_dest_acc_en=False` beyond TARGET), `EXCLUSIONS = []`, `validate()` (SUPPORTED per-axis → EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`, both `NotImplementedError` subclasses) — present and correctly wired. The weight-xor-bias check raises `ValueError` before the axis checks (it is outside the axis rectangle). The entry point calls `validate()` first.
- The op file does **not** declare `INVALID`.
- No XPASS evidence → no SUPPORTED auto-fixes.
- **INVALID audit** (`feature_spec.py`): 4 entries, all the `affine` ↔ sentinel canonicalization (presence ↔ `"none"` both ways) — multi-axis, but the canonicalization-only exception, and all three axes describe the same (weight/bias) tensor pair. No cross-tensor couplings, no "not yet supported" entries. The canonical bf8b+ROW_MAJOR entry is not applicable (TARGET dtype is bf16-only, layout TILE-only). Not norm-with-optional-gamma-only: weight and bias are coupled by the op contract, covered by the sentinel entries. Well-formed.

## L1 Ledger Audit

- **Currency**: updated `l1_ledger.md` — new rows `cb_dev_sum_acc` / `cb_dev_sum` (16-bit DEST only), `cb_local_stats` / `cb_gathered_stats` sized by `spr`, chunked affine streams; footprint expression now matches `footprint_bytes` exactly (it is the function `select_blocking` gates on).
- **Capacity vs live set**: over — `cb_weight_rows`/`cb_bias_rows` and `cb_output_tiles` exceed their live set by one double-buffer chunk/row, stated reason (overlap). Under — none: every row-spanning CB scales with `bh`, `cb_gathered_stats` with `S·bh`, the resident w/b with `bw`.
- **Page format vs DEST width**: acc CBs follow `TA` (Float32 only when `fp32_dest_acc_en`, Float16_b otherwise) — no fp32 page under 16-bit DEST. `cb_pivot` is bf16 under fp32 DEST by mechanism (exact srcB pivot). `cb_shift_full`/`cb_rstd_full` are fp32 pages read through FPU srcB (tf32) — still carries more than bf16, consumed by DestReuse; fine.
- **Disjoint lifetime / sharing**: every `Shares with / why not` cell is filled. Candidate alias `cb_weight_rows`↔`cb_bias_rows` (32 KB) recorded with reason.
- **Bounds / closed form**: all symbols bounded (`bh` by the footprint predicate, `S` by `S | grid_x`, `S | Wt`, `spr·S·bh ≤ 48`). Totals: 574 KB (2048/2304, fp32 DEST, affine), 280 KB (256×3072), 84 KB (32×64) — all well under the BH budget, no fallback regime exercised.
- **Block-size defaults**: interleaved → grid filled, coarsest block = whole per-core assignment (one block per core) ✓.
- **Traffic budget**: x/out 1× DRAM; w/b row-0 per core; stats all-gather row-local. Cheapest-traffic split is implemented (`rows_x_wslices`); the cheaper `affine_mcast` is a `deferred` regime with a positive reason (< 4% of DRAM bytes, needs a strided receiver set). Consistent with the code.

## Precision Baseline

`tests/ttnn/unit_tests/operations/layer_norm_adaln/test_layer_norm_adaln_precision_baseline.py` (32 cells: 4 shapes × offset {0, 50} × affine × DEST mode). Reference: fp32 torch `layer_norm` (+ `·w + b`). Rel-RMS = RMS error / std(ref). ULP = bf16 ULPs of the reference. Ratio = got/true over |ref| > 0.01.

**fp32 DEST (TARGET config, HiFi4)**

| Shape | offset | affine | PCC | Max Abs | Mean Abs | Rel RMS | ULP p99 | ratio median [p5, p95] |
|-------|--------|--------|-----|---------|----------|---------|---------|------------------------|
| 64×128 | 0 | no | 0.999997 | 0.024 | 0.0025 | 0.0038 | 1.8 | 0.997 [0.992, 1.001] |
| 64×128 | 0 | yes | 0.999994 | 0.078 | 0.0066 | 0.0087 | 13.3 | 0.993 [0.983, 1.001] |
| 256×1024 | 50 | no | 0.999997 | 0.041 | 0.0024 | 0.0037 | 1.4 | 0.998 [0.993, 1.001] |
| 256×3072 | 0 | no | 0.999995 | 0.052 | 0.0042 | 0.0061 | 2.5 | 0.995 [0.990, 1.000] |
| 256×3072 | 0 | yes | 0.999992 | 0.157 | 0.0083 | 0.0110 | 16.1 | 0.991 [0.980, 1.000] |
| 2048×3072 | 0 | no | 0.999995 | 0.068 | 0.0042 | 0.0061 | 2.2 | 0.995 [0.990, 1.000] |
| 2048×3072 | 0 | yes | 0.999991 | 0.210 | 0.0083 | 0.0111 | 16.6 | 0.991 [0.980, 1.000] |
| 2048×3072 | 50 | yes | 0.999994 | 0.174 | 0.0061 | 0.0083 | 9.2 | 0.993 [0.985, 1.001] |

**16-bit DEST (after fix #1)**

| Shape | offset | affine | PCC | Max Abs | Rel RMS | ratio median [p5, p95] | before fix (rel RMS) |
|-------|--------|--------|-----|---------|---------|------------------------|----------------------|
| 256×1024 | 50 | no | 0.999992 | 0.053 | 0.0080 | 1.007 [0.996, 1.019] | 0.192 |
| 256×3072 | 50 | yes | 0.999972 | 0.276 | 0.0178 | 1.015 [0.970, 1.059] | 0.459 |
| 2048×3072 | 0 | yes | 0.999988 | 0.176 | 0.0103 | 1.009 [0.999, 1.021] | 0.008 |
| 2048×3072 | 50 | no | 0.999988 | 0.125 | 0.0176 | 1.016 [0.975, 1.055] | 0.998 |
| 2048×3072 | 50 | yes | 0.999978 | 0.336 | 0.0191 | 1.016 [0.963, 1.067] | 0.995 |

**Assessment**: fp32 DEST is ~1–2 bf16 ULP (p99) without affine and ~10–17 ULP with affine (the error is dominated by the `·w + b` tail on |w| up to ~2.5 and the final bf16 rounding); offset 50 is as accurate as offset 0 — the pivot shift works. Scale triage: rel-RMS is ≤ 0.02 everywhere, far from the high-PCC/high-RMS scale-bug signature. There **is** a small systematic magnitude bias — ratio median 0.995 (no affine) / 0.991 (affine) under fp32 DEST, 1.007–1.016 under 16-bit DEST — consistent with truncating conversions (DEST→srcA moves and srcB tf32 truncation in the DestReuse normalize tail; bf16 L1 accumulation of `s2` under 16-bit DEST). Within tolerance; see Recommendations.

**Recommended tolerances**: defaults hold — `(bf16, True)`: PCC ≥ 0.995, rel-RMS ≤ 0.04 (measured worst 0.011); `(bf16, False)`: PCC ≥ 0.99, rel-RMS ≤ 0.13 (measured worst 0.019).

## Verifier CLI Summary

- supported_pass: 16
- xfail_expected: 0
- invalid_skipped: 48
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 37 (the `test_regression.py` input-distribution cells — all passed, no axis metadata)

`TARGET − SUPPORTED = ∅` on every axis (SUPPORTED additionally carries `fp32_dest_acc_en=False`). There are no generality refinements to file and `xfail_expected` is empty, so the refinement queue is perf-only.

## Performance Baseline (DEVICE KERNEL DURATION, BH, avg of 5)

Harness: `tests/ttnn/unit_tests/operations/layer_norm_adaln/test_layer_norm_adaln_perf.py` (measures, never asserts; fp32 DEST, HiFi4).

| Shape | affine | as handed over | after verifier fixes | tt_dit reference (incl. cross-chip exchange) |
|-------|--------|----------------|----------------------|------------------|
| 256×1024 | no / yes | 19.0 / 21.5 µs | 18.5 / 20.2 µs | — |
| 256×3072 | no / yes | 37.6 / 39.1 µs | 37.4 / 39.1 µs | 103.2 µs |
| 2048×3072 | no / yes | 82.3 / 114.3 µs | 81.4 / 108.3 µs | 191.6 µs |
| 2304×3072 | no / yes | 88.3 / 122.9 µs | 88.2 / 110.8 µs | 271.9 µs |

DRAM floor at 2048×3072 ≈ 25 MB / ~0.5 TB/s ≈ 50 µs; the op is ~2.2× off it with affine.

S sweep (forced `Blocking`, probe `tests/.../probes/probe_002.py`), µs no-affine / affine:

| Shape | S=1 | S=2 | S=3 | S=4 | S=6 | S=12 |
|-------|-----|-----|-----|-----|-----|------|
| 256×3072 | 50.6 / 85.4 | 34.0 / 50.1 | 31.0 / 39.6 | 28.2 / 36.0 | 29.6 / **33.0** | **37.5 / 38.9 (selected)** |
| 2048×3072 | 88.7 / 132.3 | 84.5 / 137.8 | **81.9 / 108.7 (selected)** | 89.3 / 114.2 | 104.6 / 123.9 | 182.7 / 192.8 |

Reading: `select_blocking` minimizes tiles per busiest core only, ignoring the per-(row, peer) cost of the SFPU combine and the S rotating mcast rounds. At 256×3072 the selected S=12 does 8 tiles/core yet takes 37.5 µs; S=6 is 15% (affine) / 21% (no affine) faster. At 2048, S=12 / bh=7 (56 tiles/core) costs 183 µs — ≈ 1 µs per (row, peer) of combine work. The affine tail adds ~27 µs at 2048 (2 DestReuse FPU ops on 64 tiles/core).

## Recommendations

- Refinement queue (`op_requirements.md`) is perf-only: (1) block-surface co-tune — S cost model + read/compute overlap; (2) compute-path cost — combine and normalize tail; (3) load-balanced split (scheme-change), gated on what remains after (1)–(2).
- **Precision bias** (no failing cell, no lever in scope): the ≈ −0.5% / −1% magnitude bias under fp32 DEST comes from the FPU DestReuse tail. Moving `− m`, `· rstd`, `· w + b` to SFPU ops on fp32 DEST (Refinement 2's normalize lever) should also remove it; Refinement 2 must not regress the precision baseline.
- `master.md` (perf-pattern catalog) is absent on this branch; perf refinements name regions and levers from the design's perf lamps and the measurements above instead.
- Test-infra friction (recorded, not op-related): `tests/plugins/up_front_collect.py` crashes at session start (`ttnn.graph.up_front_clear` missing), so every run here used `--no-precompile`; `EVAL_CAPTURE_PERF=1` produced no `device_kernel_ns` in the golden results without the three profiler env vars.
- L1 observation: largest footprint 574 KB of ~1.4 MB; the fallback multi-block regime is unreachable for INPUTS. Raising `INPUT_BLOCK_DEPTH` to 2 at the 2048 shape (bh=2 split into 2 blocks of 1 row) costs ≈ +128 KB and fits.
