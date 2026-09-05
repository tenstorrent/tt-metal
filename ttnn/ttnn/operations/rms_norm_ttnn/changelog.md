# Changelog: rms_norm_ttnn

## Phase 0 — Core Implementation

- **Date**: 2026-09-05
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier), as a **derivative of the designated seed**
  `ttnn/ttnn/operations/rms_norm/`. Every scheme, regime, knob, CB and kernel of the seed is
  preserved; capability is added at the seams (the A1–A13 delta table in the op file's
  docstring): `residual_input_tensor` before the statistics (A1), `bias` after the scale (A2),
  ranks 0/1/5 with no rank floor or ceiling (A3), zero-volume inputs as one degenerate device
  program (A4), `program_config` consumed with both variants validated and `subblock_w` /
  `inplace` honoured (A5), both compute-config object types accepted at the door (A6), the
  HiFi4 / approx / 16-bit-DEST default (A7), `{float32, fp32_dest_acc_en=False}` promoted from
  the seed's exclusion to a supported cell (A8), `epsilon = 0.0` accepted (A9), the per-channel
  shape rule as a logical floor plus the blocked `(Wt, 32)` ROW_MAJOR form (A10), a
  host-resident input refused at the entry point (A11), the renamed public surface taking
  `weight` (A12), and a torch reference consuming every operand (A13).

- **SUPPORTED at Phase 0** — equal to `feature_spec.TARGET` on **every** axis, with
  `EXCLUSIONS` empty:
  - `dtype` = [float32, bfloat16, bfloat8_b]
  - `fp32_dest_acc_en` = [True, False], at every dtype
  - `layout` = [TILE, ROW_MAJOR], both native (no host-side `to_layout` / `tilize` / `pad`)
  - `alignment` = [tile_aligned, w_non_aligned, h_non_aligned]
  - `rank` = [0, 1, 2, 3, 4, 5]
  - `gamma_mode` = [no_gamma, gamma, gamma_bias, bias, residual, gamma_bias_residual] — each a
    distinct compiled program, never one program with a buffer left unused
  - `gamma_dtype` = [float32, bfloat16, bfloat8_b, "none"]
  - `gamma_layout` = [TILE, ROW_MAJOR, "none"]
  - `memory_layout` = [INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED]

- **Accuracy achieved** (measured on 32 cells by
  `test_rms_norm_ttnn_precision_baseline.py`, over 4 shapes × 3 dtypes × both DEST modes,
  plus the 4 operand modes at the op's default config). Relative RMS is
  `||got - true||_2 / ||true||_2`; `r` is the got/true ratio spread — the scale-bug detector:

  | Cell | PCC | max abs | mean abs | rel RMS | `r` median | `r` std |
  |---|---|---|---|---|---|---|
  | float32, `fp32_dest_acc_en=True` | 0.99999+ | 6.1–7.5e-03 | 4.9–5.9e-04 | **7.2–8.7e-04** | 1.00020–1.00063 | 5.1–8.7e-04 |
  | float32, `fp32_dest_acc_en=False` | 0.99999+ | 2.7–5.5e-02 | 2.2–2.6e-03 | **3.4–4.1e-03** | 1.00030–1.00148 | 3.4–3.7e-03 |
  | bfloat16, `fp32_dest_acc_en=True` | 0.99999+ | 1.6e-02 | 5.0–6.6e-04 | **1.9–2.1e-03** | 1.000000 | 1.8–2.0e-03 |
  | bfloat16, `fp32_dest_acc_en=False` | 0.99999+ | 3.1e-02 | 1.5–2.1e-03 | **3.3–4.1e-03** | 1.000000 | 3.3–3.5e-03 |
  | bfloat8_b, either | 0.9999+ | 3.9–4.7e-02 | 7.6–8.9e-03 | **1.0–1.2e-02** | 1.000000 | 7.2–8.2e-02 |
  | bf16 + operands (default config) | — | 3.1e-02–1.3e-01 | 1.5–2.6e-03 | **3.7–4.6e-03** | 1.000000 | 3.7e-03–1.9e-02 |

  Precision is dtype-determined and **shape-independent** (flat across a 64× width range at
  every dtype), which is what a correctly blocked reduce with an fp32 cross-chunk accumulator
  should look like. **No scale bug anywhere**: `r` median is 1.000000–1.00063 on all 32 cells
  with `r` std of the same order as the rel-RMS — a broad spread centred on 1.0, never a tight
  cluster at a non-1.0 constant. Headroom against the shipped tolerance bands is 8.6–23×.

- **Golden suite at Phase 0** (per `generated/verifier_results/verifier_report.json`, complete
  121 438-cell collected set, run in 18 shards and merged):
  - `supported_pass`: **23 325**
  - `invalid_skipped`: 97 840
  - `infeasible_skipped`: 231 (uncharged — shard geometry vs. this device's L1)
  - `no_axes_found`: 32 (the numerics + validation tests, outside the registry grid by design)
  - `xfail_expected`: 0 — expected, since `SUPPORTED == TARGET` and `EXCLUSIONS` is empty
  - `xpass_drift`: **0** ✓
  - `xfail_wrong_mode`: **0** ✓
  - `supported_marked_xfail`: 0 ✓
  - `supported_fail`: **10 — all harness-attributed, zero op-attributed** (see below)

- **Perf at Phase 0**: **16 of 19** perf-group targets already meet their clock-scaled
  `achievable_ns` ceiling. Measured AICLK 1350 MHz = the reference clock, so the scale factor is
  exactly 1.0000. The hardest requirement in the spec — `(1,1,32,7168)` INTERLEAVED, the only
  case carrying `minimum_expected_speedup = 7.0`, i.e. a 14 894 ns ceiling — measures
  **9 118 ns** (0.612 of ceiling, an 11.4× speedup over the reference). The three misses are all
  the same regime: WIDTH-sharded decode at 28–32 cores, at ratios 1.060 / 1.050 / 1.014.

- **Occupancy at Phase 0** (measured `device_num_cores` over 23 340 cells): max **110 = the
  whole 11×10 Blackhole compute grid**; `1x1x2048x256` and `4x1x512x512` median 64 / max 110;
  `128x8192` median 44. The 20.4% of cells at one core are small shapes whose row axis is a
  single tile-row and whose width split is correctly gated off. This is what upgraded
  `PROPERTIES["multi_core"]` from `"declared"` to `"verified"`.

- **Issues encountered**:
  1. **Fixed — wrong refusal class.** `_check_per_channel` raised `UnsupportedAxisValue` (a
     `NotImplementedError`) for a per-channel operand at a dtype outside the accepted set, where
     the prompt's "## Validation" list and `test_validation.py::_REFUSED` both require
     `ValueError` / `RuntimeError`. Changed to `ValueError`, with the two refusals' different
     *kinds* documented at the raise, plus an import-time assertion that `PER_CHANNEL_DTYPES`
     stays a superset of `SUPPORTED["gamma_dtype"] - {"none"}` so a future narrowing of the axis
     cannot make the two swap (which would surface as `xfail_wrong_mode`).
  2. **Fixed — `l1_ledger.md` over-counted reuse-shared traffic by 16×.** Every per-channel
     figure was priced at whole tiles, but D23 trims a TILE-layout per-channel read to two
     face-rows (128 B of a 2048 B bf16 tile). Corrected in the traffic table. Consequence: the
     `GAMMA_MCAST` deferral is right for a stronger reason than it stated — gamma is ~1% of DRAM
     bytes, not ~18% — so it is explicitly **not** filed as a refinement.
  3. **Fixed — `D3`'s justification for `ReduceFp32Mode::Fast` was an argument, not a
     measurement.** A/B'd with each variant in its own isolated `TT_METAL_CACHE`:
     `Accurate` returns **inf/NaN** at `fp32_dest_acc_en=False` (the op's own default — it needs
     a 32-bit DEST) and is identical to `Fast` to five significant figures at `True`. Kernel
     reverted; the table and a silent-failure warning are recorded at `D3`.
  4. **Fixed — stray `</content>` end-tags** in `l1_ledger.md` and `op_design.md`.
  5. **Not the op — three harness defects**, each independently reproduced outside the op and
     accounting for all 19 red golden cells: `CoreRange.end_coord` / `.start_coord` in
     `program_config.py` on a build that exposes `.start` / `.end` (7 loose cells + 3 validation
     tests never reach the op); `torch.max()` on a zero-element readback in `eval/metrics.py`
     (3 zero-volume cells fail *after* the op returned a correct empty tensor); and
     `test_regression.py` calling `check_output` without `tolerance=`, so it scores the op's own
     declared default cell against `DEFAULT_TOLERANCES[float32]`'s 0.01 rather than the 0.04 the
     feature spec's `TOLERANCE_OVERRIDES[(float32, False)]` declares for exactly that cell
     (6 numerics cells; measured 0.0165–0.0189, and `fp32_dest_acc_en=True` reduces it 6.7–7.1×,
     confirming the 16-bit accumulator as the cause). Three one-line fixes requested in
     `verification_report.md`; not applied by the verifier because they are shared eval
     infrastructure.
  6. **Not the op — a poisoned JIT artifact cost real triage time.** This environment sets
     `TT_METAL_CACHE=<repo>/built`, `TT_METAL_CCACHE_KERNEL_SUPPORT=1` **and**
     `TT_METAL_JIT_SERVER_ENDPOINT`. After the `ReduceFp32Mode` A/B above was reverted, the
     reverted source kept being served the experiment's binary — 2/19 debug tests failed with
     non-finite output on exactly the config the experiment had compiled — and `rm -rf built`
     did **not** clear it. A fresh `TT_METAL_CACHE`, or `--no-jit-server`, restores correctness;
     verified clean afterwards at **455 passed / 1 skipped** across the whole unit directory.
     Every refinement in the queue is a kernel A/B, so this is called out at the top of
     `op_requirements.md`.
  7. **Not the op — five `feature_spec.INVALID` entries are misclassified.** `feature_spec.py`
     itself labels them "author-scoped exclusions ... NOT structural impossibility"; three
     additionally cross axes describing *different tensors* (activation `layout` /
     `memory_layout` × weight `gamma_layout`), which is the canonical INVALID authoring mistake.
     All five name capability that `SUPPORTED` claims, `validate()` accepts and the prompt
     requires — and because INVALID cells are skipped, the cartesian never tests any of it.
     `test_rms_norm_ttnn_invalid_audit.py` runs all five regions: **16/16 pass**. Removal
     requested in `verification_report.md`.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_precision_baseline.py`
    (32 cells — PCC / abs / relative-RMS **and the got/true ratio spread**, with an explicit
    assertion against a uniform scale error)
  - `tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_invalid_audit.py`
    (16 cells over the five capability regions the golden suite structurally cannot reach)
  - `ttnn/ttnn/operations/rms_norm_ttnn/perf_target_ranking.py` (ranks the `perf` loose group by
    measured device-ns / clock-scaled ceiling — the same ranking the trailing perf pass uses)
  - `scripts/verifier_golden_shards.sh` + `scripts/verifier_merge_golden_shards.py` (the
    sharded golden-run harness the 121 438-cell cartesian requires)

- **Pre-existing tests, all green**: `test_rms_norm_ttnn.py` (90, the immutable acceptance
  spec), `test_rms_norm_ttnn_debug.py` (19), `test_rms_norm_ttnn_matrix.py` (227),
  `test_rms_norm_ttnn_perf.py` (72, incl. the structural seed-parity gate). Whole directory:
  **455 passed, 1 skipped**.
