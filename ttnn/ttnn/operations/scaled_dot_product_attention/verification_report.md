# Verification Report: scaled_dot_product_attention (Flash Attention)

## Code Review

### Fixes applied

1. **Missing `default_compute_kernel_config()` factory (registry-conformance bug — fixed).**
   The golden harness (`eval/golden_tests/.../axes.py`) imports
   `default_compute_kernel_config` from the op module as the single source of truth
   for tagging the `fp32_dest_acc_en` axis, and the design (§Parameters, l.99/103)
   specifies the default is resolved through it. The implementer inlined the default
   `ttnn.ComputeConfigDescriptor(...)` in the entry point and never exported the
   factory, so the entire golden suite failed to import (`ImportError:
   cannot import name 'default_compute_kernel_config'`). **Fix:** added
   `default_compute_kernel_config()` (HiFi2 + `fp32_dest_acc_en=True` +
   `math_approx_mode=False`), routed the entry point's `None` resolution through it,
   and exported it from `__init__.py`. Single source of truth restored.

2. **validate() ordering — support gate now precedes the shape contract (fixed).**
   validate() ran the detailed tensor-shape contract (ValueError for e.g. a
   batch-broadcast mask, `m_b != b_q`) *before* the SUPPORTED axis check. For a cell
   whose axes are outside SUPPORTED (`fp32_dest_acc_en=False`) but which *also*
   carries a shape issue, the ValueError fired first and the cell was rejected with
   the wrong exception type → 24 `xfail_wrong_mode`. The registry model requires the
   axis-level support refusal (NotImplementedError) to take precedence. **Fix:**
   reordered validate() — minimal rank guard (needed to build the axes dict) →
   SUPPORTED per-axis → EXCLUSIONS → detailed tensor-shape contract. All 24
   `xfail_wrong_mode` cleared; acceptance suite unchanged (36 passed).

### Reviewed, no change needed

- **Block-size / buffer-depth knobs are parameters with a single source of truth.**
  `Q_CHUNK_TILES`, `K_CHUNK_TILES`, `KV_BUFFER_FACTOR`, `Q_BUFFER_FACTOR` are host
  constants; effective chunk sizes are derived once (`_largest_divisor_leq`) and
  every CB page count / loop bound / kernel CT-arg is computed from them. No
  duplicate literals, no CB unconditionally sized to full S_q/S_kv (scores CB is
  `sq_chunk_t·sk_chunk_t`, never the full matrix — the flash-attention invariant
  holds). `k_num_chunks` is deliberately a runtime arg to keep the KV loop rolled
  (kernel-config-buffer safety on long sequences) — a sound, documented choice.
- **Multi-core distribution fills the grid.** `total_q_blocks = B·H_q·q_num_chunks`
  split via `ttnn.split_work_to_cores`; the independent (B, H_q, S_q-block) axes
  spread across the whole grid with no cross-core communication (Phase-1 design).
- **Documented deviations from the design that are correct alternatives:** the
  running `(m, l, O)` are held in fixed CBs updated in place (rather than the
  design's A/B ping-pong) to keep one straight-line kernel body — documented in the
  kernel header and CB-sync-balanced (verified below). PV uses `num_k_blocks=1` with
  the online-softmax rescale carrying the cross-KV-block accumulation instead of
  `packer_l1_acc` — equivalent and correct.
- **API hygiene:** `TensorAccessor` (not `InterleavedAddrGen`); `void kernel_main()`;
  `api/dataflow/dataflow_api.h`; reader batches a block of async reads then one
  barrier per CB (double_buffer pattern); broadcast dims correct (row-max/corr/1-l
  all `BroadcastDim::Col`). Compute phases all use kernel_lib helpers; the only raw
  API is interleaved tensor I/O (TensorAccessor's domain).

### Advisory (not blocking, no in-scope lever)

- The reader issues K → barrier → V → barrier → (mask → barrier) sequentially per
  KV-block (three barriers). Interleaving the three streams behind fewer barriers is
  a data-movement perf lever, folded into Refinement 5 (perf), not a correctness
  issue.

## Registry Conformance

- **Confirmed present & correctly wired:** `INPUT_TAGGERS` (three taggers, all with
  the `(inputs, axes)` signature), `SUPPORTED` (every gated axis incl.
  `fp32_dest_acc_en`), `EXCLUSIONS` (`[]`), `validate()` (SUPPORTED → EXCLUSIONS →
  shape contract, after the ordering fix), `PROPERTIES`. The public entry point calls
  `validate()` before any kernel work.
- **Confirmed the op file does NOT declare `INVALID`** — it is a test-suite concept
  living in `feature_spec.py`. ✓
- **No SUPPORTED auto-fix required** — `xpass_drift = 0`; the SUPPORTED block is
  honest about what passes.
- **INVALID audit (`feature_spec.py`):** `INVALID = []`. Well-formed — SDPA is
  TILE-only by design (no ROW_MAJOR in TARGET), so the canonical `bf8b + ROW_MAJOR`
  rule is vacuous (no such cell exists in the cartesian product). No cross-tensor-axis
  coupling, no "not-yet-implemented" masquerading as structural impossibility. SDPA
  is not norm-like (no weight axis), so the no-weight canonicalization rule does not
  apply. Audit passes.

## Precision Baseline

bf16, TILE, tile-aligned, self, MHA, no mask, auto scale, HiFi2 + fp32 DEST (the
Phase-0 supported corner), standard `randn` inputs, seed 42:

| Shape | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio (med, p5, p95) |
|-------|-------------|--------------|------------------|-------------------------------|
| (1,1,32,32)  | 0.01335 | 0.00169 | 0.00791 | 0.9951 / 0.9539 / 1.0170 |
| (1,4,128,64) | 0.00796 | 0.00084 | 0.00840 | 0.9956 / 0.9620 / 1.0287 |
| (1,8,256,64) | 0.01194 | 0.00067 | 0.00930 | 0.9950 / 0.9580 / 1.0309 |
| (1,4,512,64) | 0.00817 | 0.00051 | 0.00992 | 0.9947 / 0.9553 / 1.0340 |

All PCC ≥ 0.995. The got/true ratio is centered on ~0.995 with a **symmetric** spread
(p5≈0.95, p95≈1.03) — i.e. clustered on 1.0, not a tight non-1.0 constant, so this is
ordinary bf16 rounding noise, **not** a scale/structural bug. rel-RMS grows mildly
with sequence length (more KV-blocks → more bf16 accumulation of the running `O`),
which is the lever Refinement 1 (fp32 intermediate CBs / fp32 dtype) addresses.

**Assessment:** healthy for bf16. **Recommended tolerances:** PCC ≥ 0.995,
normalized rms ≤ 0.05 for `randn`-like data (matches the golden `(bf16, True)`
tolerance).

**Adversarial-distribution note (regression suite):** `test_regression.py`
`test_uniform_input` / `test_negative_input` / `test_large_magnitude_input` fail at
the default bf16 corner on near-uniform-attention distributions (uniform-positive /
negative-only inputs → low-variance outputs). Measured PCC 0.985–0.997 with rel-RMS
0.05–0.20, got/true ratio **centered on 1.0** (p5≈0.99, p95≈1.005) — genuine bf16
precision amplified by the rel-RMS normalization over a low-variance output, **not** a
scale bug. These tests import the op directly (not the `observed` wrapper) so they
land in `no_axes_found`, not a loud verifier category; they do not block. Refinement 1
(float32 support + fp32 intermediate CBs) is the concrete lever that clears them.

## Verifier CLI Summary

(`eval.verify_supported`, structured artifact `verifier_report.json` kept in the
results dir `/tmp/sdpa_results/` — 1.4 MB, exceeds the repo's 500 KB file limit so
not checked in)

- supported_pass: **206**
- xfail_expected: **2113**
- invalid_skipped: 0 (INVALID is empty)
- **supported_fail: 6** — all `OOM`, D∈{512,1024} at S=128 (see below). Per the
  registry-model routing table, OOM stays failing and becomes a refinement entry
  (Refinement 2); silencing via EXCLUSIONS would delete a real gap.
- **xpass_drift: 0** ✓
- **xfail_wrong_mode: 0** ✓ (was 24; fixed by the validate reorder)
- supported_marked_xfail: 0
- no_axes_found: 118 — `test_regression.py` + nightly `test_translated.py` rows that
  import the op directly / are skipped by default; 14 are the bf16 adversarial-
  distribution precision failures noted above. Not a loud category.

The two fixable loud categories (`xpass_drift`, `xfail_wrong_mode`) are clean. The 6
`supported_fail` are all OOM and are the subject of Refinement 2 (they legitimately
stay failing until that lands — not silenced).

### supported_fail detail (OOM)

```
Q1x1x128x512  → CBs grow to 1,581,824 B  (> 1,572,864 B L1 max)
Q1x1x128x1024 → CBs grow to 2,827,008 B  (> L1 max)
```

Root cause: the streaming CBs (`cb_q_in`, `cb_k_in`, `cb_v_in`, `cb_out`, and the
running-`O` accumulators) all scale linearly with `dht = D/32`. D is not blocked, so
per-core L1 blows past 1.5 MB once `dht ≥ 16`. This is a per-core L1-budget problem —
`/memory-budget-metal` (Refinement 2). Note the numeric refinement (float32, 2× tile
bytes) worsens this, hence the R1→R2 ordering dependency.

## Recommendations

- **Refinement order is anchored to the perf target.** The perf-flagged loose case
  `(1,10,9472,128)` runs bf16 @ `fp32_dest_acc_en=False` — a config not yet in
  SUPPORTED. Refinement 1 (numeric bundle) lands that contract; the trailing perf
  passes optimize the exact flagged config (never a supported proxy).
- **Dominant perf lever for the flagged shape is redundant K/V DRAM reads.** With
  740 Q-blocks over ~110 cores, every core owning a Q-block of a given (batch, head)
  re-reads the same ~2.4 MB K and V from DRAM. The `shared_input_reuse` mcast
  (design lamp #3, master.md T3) is the primary lever (Refinement 3); block-size /
  buffer-depth co-tunes (T2) are secondary (Refinement 5).
- **bf16 is at its precision ceiling** (HiFi2 is required for the multi-K matmuls;
  HiFi4 + fp32-DEST with bf16 inputs is the #38306 corruption trap and must be
  avoided). Higher precision requires the float32 path — Refinement 1.
- **L1 budget will be re-touched twice** — R1 adds fp32 (2× CB bytes) and R2 budgets
  it; the flagged perf shape (D=128) fits comfortably, so perf work is unaffected.
