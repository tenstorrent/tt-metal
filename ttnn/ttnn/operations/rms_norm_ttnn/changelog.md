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

---

## Refinement 1 — The width-sharded decode combine round

- **Date**: 2026-09-05
- **Type**: perf. **No SUPPORTED / EXCLUSIONS change** — `verify_supported`'s categories are
  untouched by construction (nothing in the registry declarations was edited).

- **What was done** — all four named levers implemented and measured on device; nothing
  reverted. `Reused:` the existing combine dataflow, its CBs, its semaphores and all three
  kernels — **no kernel source changed except one stale comment**. `Added:` four host-side
  derived rules in `rms_norm_ttnn_program_descriptor.py`, each with one source of truth.

  1. **Lever 1 — combine-tree arity (`COMBINE_TREE_F0` → a band).** The level-0 fan-in was
     the constant `4`; it is now **derived**: the largest *divisor* of `GROUP_SIZE` in the
     measured band `[COMBINE_TREE_F0_MIN, _MAX] = [4, 10]` that both of D28's existing gates
     admit, with the cap as a ragged fallback for a group no divisor covers.
     `_combine_tree_candidates` / `_combine_tree_arity` are the only place it is decided;
     the two gates (`f1 >= 2`, `deleted >= 18`) are untouched. The sweep (whole-op device
     kernel ns, on top of lever 2, median of 5):

     | `G` | flat | f0=4 | f0=5 | f0=6 | f0=8 | f0=10 | f0=15/16/32 | rule picks |
     |---:|---:|---:|---:|---:|---:|---:|---:|---|
     | 28 | **5699** | 6004 | — | 6081 | 6018 | 5987 | — | flat (every tree loses) |
     | 30 | 5058 | 5051 | 4981 | 4981 | 5056 | *4787* | 5264 | `(6,5)` |
     | 32 | 5253 | 4979 | — | 4991 | **4736** | 5133 | 5201 | `(8,4)` |
     | 40 | 5402 | 5064 | — | 4927 | 4941 | **4789** | — | `(10,4)` |
     | 64 | 6695 | 5498 | — | 5362 | **5047** | 5363 | 5754 / 6436 | `(8,8)` |

     Two mechanisms, both visible: a **larger `f0` unloads the root** (its ingress is `f1-1`
     remote writes and its fold is `f1` pages, and the root is also the finalizer and the
     multicast sender), but **an exact divisor beats a bigger ragged `f0`** — at `G=32`,
     `f0=8` (8×4, exact) is 4736 while `f0=10` (10×4, ragged) is 5133, *same `f1`*, 8.4%
     apart. Past ~10 the level-0 gatherer becomes the new serial bottleneck.
     The derived shape is also never larger in L1 and at `G=64` is **32 KB/core smaller**
     (rings `4+16` → `8+8` pages).
  2. **Lever 2 — which NoC carries the combine.** New `COMBINE_NOC_RESIDENT` /
     `COMBINE_NOC_STREAMED`, chosen per plan by `_combine_noc(plan.native_in)` and read by
     BOTH the host mcast wire (five de-duplicated `McastConfig` call sites) and the two
     data-movement `KernelDescriptor` configs. On a `native_in` plan x is an aliased
     resident shard, so the reader carries no activation traffic and the combine takes
     **NOC_0**; the reader moves to NOC_1 in the same step. Gated **out** on streamed plans.
  3. **Lever 3 — `GATHER_FACES`.** Re-swept `{2, 3, 4}` on top of the new tree and the new
     NoC. 2 still wins and the loss is **monotone in the byte count** on every geometry
     (A 5709 / 5976 / 6325; C 4983 / 5142 / 5336; B 6573 / 6641 / 6827; the 8- and 9-core
     groups likewise). Unchanged — the D13/D27 scoping survives the re-measurement.
  4. **Lever 4 — Lamp L-RES-DEPTH (`CB_R_DEPTH`).** *The lamp's premise is false on its own
     target*, and that is a host-side fact rather than an argument: on `(1,1,32,5120)`
     WIDTH `[32,160]` `(8,4)` with `gamma_bias_residual` the residual carries the input's
     shard spec, so `native_residual` holds and `cb_residual_tiles` is
     `cb_descriptor_from_sharded_tensor` — dumped from the built descriptor as **5 pages,
     exactly the shard**, an alias that costs no arena L1 and to which no depth is applied.
     There is no second double-buffered stream to shrink. The knob was still **built and
     kept**, parked at its byte-identical default (`CB_R_DEPTH = 0` = follow `CB_X_DEPTH`),
     with `_residual_depth()` as the single source both L1 solves and the CB table read, and
     **measured** on the plans where it IS live (a streamed interleaved residual):
     `(1,1,32,5120)` 10281 / 10469 / 10378 and `(1,1,8192,5120)` 669877 / 667814 / 668641 ns
     at depth follow / 1 / 2 — a null inside ±0.3%. Parked, not reverted.

- **Perf achieved** — harness-native numbers, `run_safe_pytest.sh --profile` on the three
  target golden cells, blackhole p150b, AICLK 1350 MHz = the reference clock (scale 1.0000):

  | case | ceiling | Phase 0 | Refinement 1 | ratio |
  |---|---:|---:|---:|---:|
  | `(1,1,32,5120)` W `[32,160]` `(8,4)` **`gamma_bias_residual`**, `fp32_dest=True` | 6555 | 6882 (1.050) | **6420** | **0.979 ✅** |
  | `(1,1,32,5120)` W `[32,160]` `(8,4)` `gamma` | 5267 | 5339 (1.014) | **4807** | **0.913 ✅** |
  | `(1,1,32,7168)` W `[32,256]` `(7,4)` `gamma` | 5481 | 5812 (1.060) | 5766 | 1.052 ❌ |

  **2 of the 3 named misses now meet their ceilings; the perf-group miss count goes 3 → 1.**
  Whole-op A/B in one process (Phase-0 constants vs shipped, min of 2 reps × median of 5):

  | geometry | before | after | speedup |
  |---|---:|---:|---:|
  | `(1,1,32,8192)` W `[32,128]` `(8,8)` 64c | 5871 | 5075 | **1.157x** |
  | `(1,1,32,7040)` W `[32,128]` `(11,5)` 55c | 5624 | 5201 | **1.081x** |
  | `(1,1,32,5120)` W `[32,128]` `(10,4)` 40c | 5108 | 4764 | **1.072x** |
  | `(1,1,32,5120)` W `[32,160]` `(8,4)` 32c gamma | 5019 | 4765 | 1.053x |
  | `(1,1,32,5120)` W `[32,160]` `(8,4)` 32c gbr | 6632 | 6313 | 1.051x |
  | `(1,1,8192,1024)` BLOCK `[1024,128]` `(8,8)` 64c | 24418 | 23565 | 1.036x |
  | `(1,1,32,5632)` W `[32,128]` `(11,4)` 44c | 5218 | 5101 | 1.023x |
  | `(1,1,7168,1024)` BLOCK `[896,128]` 64c gbr | 33676 | 33013 | 1.020x |
  | `(1,1,32,1024)` W 8c / `(1,1,32,4800)` W 30c | 3719 / 5036 | 3681 / 5019 | 1.010x / 1.003x |
  | `(1,1,32,2304)` W 9c | 4399 | 4413 | 0.997x (noise; no rule fires) |
  | every INTERLEAVED case (row split + width split, incl. all three prefill gbr) | — | — | **0.999–1.006x — unchanged, as gated** |

- **Accuracy achieved**: PCC ≥ 0.999983 on every geometry above, at every variant swept —
  comfortably inside the perf cells' soft `pcc_threshold = 0.9995`. The two `fp32_dest_acc_en
  =True` operand cases measure 0.999988 and 0.999990. Nothing about the numerics moved: the
  combine sums the same partials in the same order, only the tree's *shape* and the NoC
  underneath it changed.

- **Golden test progress**: unchanged by construction (a perf refinement adds no axis value).
  Slices run: WIDTH_SHARDED loose **103 pass / 3 pre-existing harness `CoreRange.end_coord`
  failures**, BLOCK+HEIGHT_SHARDED loose **204 pass / 4 same**, cartesian
  WIDTH_SHARDED × `gamma_bias_residual` × BFLOAT8_B **340 pass**. The 7 failures are exactly
  the 7 loose cells Phase 0 already attributed to that harness defect.

- **Issues encountered**:
  1. **A one-sided NoC move HANGS.** Setting the writer to NOC_0 while leaving the reader
     there timed out `(1,1,32,4800)` WIDTH 30c with two physical cores never finishing:
     `DM_DEDICATED_NOC` gives each RISC its own engine, so two kernels on one engine is not
     sharing, it is a collision. The lever is a **swap** of both kernels; `_combine_noc_swapped`
     carries the finding.
  2. **The NoC lever must be gated, not global.** On the interleaved width split it measured
     **9044 → 13550 ns (0.667x)** — there the reader carries every activation byte and NOC_0
     is the arch's `preferred_noc_for_dram_read`. Hence the `native_in` predicate.
  3. **The nanobind `NOC` enum's aliases do not compare equal to themselves.**
     `ttnn.NOC.NOC_0 == ttnn.NOC.RISCV_0_default` is `False` despite both being value 0.
     Every comparison in the op and in the new test goes through `.value`.
  4. **Seed parity moved on exactly one geometry, with a measurement.** The derived arity
     reshapes the tree rings at `G = 64`, which the RM BAND width shard in
     `test_program_is_structurally_the_seeds` hits. Per the queue's own rule the test now
     carries a **one-directional allowance** (this op may never spend MORE L1 in those two
     rings than the seed) and masks `TREE_F0`/`TREE_F1` in the writer/compute CT lists;
     everything else is still asserted identical. The justifying measurement is in the test:
     `f0=4` 24035 / 24096 (no_gamma) and 24061 / 23992 (gamma) vs `f0=8` 23919 / 23836 and
     23839 / 23869 — 4/4 reps favour the derived shape, plus 32 KB/core of L1 returned.
  5. **`CB_R_DEPTH` at a non-default value perturbs the L1 solve on `native_residual` plans**
     (the solve prices a residual ring that the allocation then aliases away), which showed
     as a 33.0 → 36.2 µs regression on the 64-core BLOCK shard at `CB_R_DEPTH = 1`. Harmless
     at the shipped default; recorded at the constant so the next turner sees it first.

- **Where the last case's time actually goes** (permanent per-stage zones, `(1,1,32,7168)`
  WIDTH 28c, flat combine, NOC_0; medians, occupancy not payload): kernel 5381 ns, of which
  the **root's chain is 4130 ns (77%)** — `writer_gather_wait` 1040 + `compute_root_fused`
  1914 (fold 28 partials, then the rsqrt) + `writer_mcast_send` 1176. Every other core spends
  it waiting (`writer_gather_ship` 2426 + `writer_mcast_recv` 1861). Forcing the tree on there
  shortens the root chain to 3483 ns but adds ~1900 ns of level-0 gatherer work in series
  (`compute_tree_fold_l0` 1110 + `writer_tree_forward` 550 + `writer_gather_zero` 256), which
  is why the gate keeps it flat and why *no arity* wins at `G = 28`.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_combine_knobs.py`
    (132 cells) — the measured arity per group size, the two kernel-side tree invariants
    swept over every group 2..120, the `CB_R_DEPTH` byte-identical default, and that the
    combine's NoC choice is a swap of *both* kernels gated on a resident x. All host-side;
    nothing dispatches. These decisions are invisible to a numerical test, which is exactly
    why they need pinning.
  - `test_rms_norm_ttnn_perf.py::test_program_is_structurally_the_seeds` extended with the
    documented, measured tree-ring allowance (above).

## Refinement 2 — Spread the finalize (Lamp L-FIN)
- Date: 2026-09-05
- What was done:
  - **The named scheme change was BUILT, MEASURED, and PARKED.** `COMBINE_FIN_SPREAD`
    makes the last-level fold forward the **RAW** group sum (`combine_fold`'s existing
    `FINALIZE` template parameter, flipped) so the multicast carries that, and every core
    applies `rsqrt(sum/W + eps)` to its own copy in parallel — the literal Lamp L-FIN move,
    covering **both** branches with one predicate (the compact tile is finalized before the
    un-permute; the identity tile is finalized into `cb_row_stat`, which the combine path
    leaves dead). It is **correct** — pcc and rel-RMS are bit-identical to the root
    finalize on every case measured — and it **loses**: **0.953–1.004x** across nine
    combine geometries. Parked at its byte-identical default (`False`), kept live, and the
    reason is recorded next to the knob and in `op_design.md`'s stall-shadow table: the
    finalize is a **replicated** term, not a divisible one. Every core needs the *same*
    value, so moving the rsqrt does not delete it anywhere — it just relocates it along the
    identical serial chain (fold → [rsqrt] → send → recv → [rsqrt] → pass B). Two prior
    decisions had already taken what there was to take: **D22** fused the root's rsqrt into
    the fold's DEST window (so at ROOT it costs no pack at all, while a spread one needs its
    own copy + pack + unpack on every core), and **D27** collapsed the finalize from
    `BLOCK_ROWS` tile-ops to **one per round**, which is the O(`BLOCK_ROWS`) half the lamp
    was originally written against.
  - **The round's TRANSPORT is where the phase's win came from** — the half Refinement 1's
    outcome explicitly deposited here, and the half the verifier note anticipates ("changes
    what the multicast carries ... and therefore the receive side"). Two gates, both on the
    stat multicast, both inert on every non-combine build:
    1. `COMBINE_MCAST_FIRE_AND_FORGET` — elide `mcast_pipe`'s receiver-readiness
       **pre-handshake** (its own per-kernel `compile_time_args(pre_handshake=...)`
       override; no new CT arg, no kernel change) whenever the combine runs exactly
       **one round**. Safe by an exact ordering property, not a statistical one:
       `ReceiverPipe`'s ctor — which stores INVALID into the data-ready flag — runs before
       a core ships its partial, and the root cannot send until every partial has landed.
       The writer kernel carries that invariant as a comment at the receiver construction
       so a future edit cannot silently break it.
    2. `COMBINE_MCAST_FACES = 3` — the **identity-path** multicast carries faces 0..2
       (3 kB, ONE transaction covering both column-carrying faces) instead of the whole
       4 kB tile. Same measured licence as D26's gather trim applied in the other
       direction: the landing CB's only reader there is pass B's column broadcast. The
       COMPACT path is forced to whole tiles (its un-permute matmul sums 32 products).
       Packed into the **high byte of the existing `GATHER_FACES` CT word** so the writer's
       argument-list shape stays the seed's.
- Accuracy achieved: output is **bit-identical** across all four sweep variants — pcc and
  rel-RMS agree to every printed digit (pcc 0.999985–0.999990, rel-RMS 0.0045–0.0067 against
  the op's 0.04 bound) on the nine combine geometries plus three non-combine guards. The
  soft `pcc_threshold = 0.9995` on the `gamma_bias_residual` cell holds with four nines of
  margin.
- Golden test progress: `test_op_loose` **433/443** — the identical Phase-0 figure; all 10
  failures are the same harness-attributed ones (`CoreRange.end_coord`, `torch.max()` on a
  zero-element readback), none op-attributed. Plus a 2140-cell `test_op` slice over
  `1x1x32x8192 / 1x1x128x4096 / 1x1x2048x256` (every layout × placement), all green.
- Perf, measured (blackhole p150b 1350 MHz, in-process profiler, median of 5, min over 3
  reps; noise floor **±0.3%**, calibrated on the two BLOCK-shard cells whose program is
  byte-identical across the whole sweep):

  | case | before | after | x | ceiling | ratio |
  |---|---:|---:|---:|---:|---:|
  | `(1,1,32,7168)` WIDTH `[32,256]` (7,4) 28c | 5713 | **5520** | **1.035** | 5481 | 1.042 → **1.007** |
  | `(1,1,32,2304)` WIDTH `[32,256]` (9,1) 9c | 4422 | **4356** | **1.015** | 4617 | 0.943 |
  | `(1,1,256,512)` ROW_MAJOR BAND 64c | 23870 | **23556** | **1.013** | — | — |
  | `(1,1,32,5120)` WIDTH `[32,160]` (8,4) 32c | 4769 | **4724** | **1.010** | 5267 | 0.897 |
  | `(1,1,32,7168)` INTERLEAVED width-split 16c | 9026 | **8961** | 1.007 | 14894 | 0.602 |
  | `(1,1,32,1024)` WIDTH 8c | 3683 | **3658** | 1.007 | 4110 | 0.890 |
  | `(1,1,32,5120)` WIDTH gbr/fp32 32c | 6327 | **6295** | 1.005 | 6555 | 0.960 |
  | `(1,1,32,8192)` WIDTH 64c | 5081 | **5061** | 1.004 | 6000 | 0.844 |
  | `(1,1,8192,1024)` BLOCK 64c (gates OFF) | 23527 | 23596 | 0.997 | 28619 | 0.824 |
  | `(1,1,7168,1024)` BLOCK gbr 64c (gates OFF) | 32999 | 33021 | 0.999 | 34569 | 0.955 |
  | `(1,1,1024,512)` WIDTH compact 4c | 22834 | 22821 | 1.001 | — | — |
  | `(1,1,8192,1024)` INTERLEAVED (no combine) | 88282 | 87870 | 1.005 | 89992 | 0.976 |

  No cell below the noise floor. The two 64-core BLOCK shards the verifier flagged (one with
  only 1.6% of margin) are multi-round, so the pre-handshake gate keeps them at the seed's
  program and they measure flat, as designed.
- Issues encountered: None. The spread finalize worked first time and its result was a
  measured null, which is the finding rather than a failure — the lamp had already been
  closed by D22 and D27 without anyone noticing, and this phase is the measurement that says
  so. One structural constraint shaped the diff: the writer's CT-arg **list shape** is a
  checked seed property, so the multicast's face count is packed into an existing word
  rather than appended, and the two words that legitimately move (`15`, `22`) are added to
  `test_program_is_structurally_the_seeds`'s measured allowance alongside Refinement 1's
  tree arity.
- Tests added:
  - `test_rms_norm_ttnn_combine_knobs.py`: six new tests — the finalize site's default AND
    that it is **still a live knob** (flipping the module constant moves the CT arg), the
    pre-handshake's single-round gate, the packed face word's two bytes, and an end-to-end
    check that both gates reach the writer's CT list where the kernel reads them.
  - `test_rms_norm_ttnn_perf.py`: `_MCAST_WRITER_CT`, the measured allowance for the two
    combine-transport words, with the full sweep table inline.

## Refinement 3 — Strip the per-block fixed costs off the interleaved prefill
- Date: 2026-09-06
- Type: perf. **No SUPPORTED / EXCLUSIONS change** — nothing in the registry declarations
  was edited, so `verify_supported`'s categories are untouched by construction.
- What was done — **all four named levers built and measured; nothing deleted, nothing
  reverted; four of the five knobs are measured nulls parked at byte-identical defaults
  and the fifth is the phase's win.** `Reused:` every chain, CB, helper, kernel file and
  program-descriptor branch — no new kernel, no second descriptor path. `Added:` five
  knobs, each with exactly one source of truth, plus one permanent ablation switch.

  **The premise had to be corrected first, and that is the phase's most important
  finding.** The queue's Goal priced the two interleaved prefill cases against a
  "~450 GB/s roofline" and concluded "the gap is compute-side overhead rather than byte
  count". Measured against the *machine's* practical ceiling for the identical traffic —
  the same tensors through the cheapest possible kernels — that is not what is happening:

  | (1,1,8192,W) INTERLEAVED bf16 | rms_norm_ttnn (R2) | `ttnn.exp` | `ttnn.clone` |
  |---|---:|---:|---:|
  | W=1024, 33.5 MB | 88 128 ns / **381 GB/s** | 88 787 / 378 | 83 833 / **400** |
  | W=7168, 234.9 MB | 576 680 ns / **407 GB/s** | 624 674 / 376 | 590 613 / **398** |

  A full RMS norm *with a weight* already **beats a pure DRAM→DRAM copy** at W=7168 and
  sits 5% off it at W=1024; the no-operand build (83 087 ns) is AT `clone`'s number to
  0.9%. So the interleaved prefill is **DRAM-saturated, not overhead-bound**, ~400 GB/s
  is the achievable ceiling on this box, and the residual 5% at W=1024 is the per-channel
  operand's own DRAM traffic rather than any per-block fixed cost. Every lever below was
  still built and measured against that corrected picture.

  1. **Lever 1 — data-format reconfig elision. MEASURED NULL, bounded by an ablation
     rather than argued.** Instead of guessing at per-boundary predicates, the seven
     operand specs and the reduce's mode now route through one named `DFR` /
     `REDUCE_RECONFIG` pair, and `RMS_ABLATE_RECONFIG` strips **every** data-format
     reconfig in the compute kernel at once. That build is numerically destroyed
     (pcc ≈ 0 / NaN — proof the reconfigs are genuinely doing work) and it moves the
     clock **not at all**: 0.989–1.008x across all twelve cases, i.e. inside the noise
     band, with the largest single reading being a 1.1% *loss*. The mechanism is why:
     `eltwise_chain`'s reconfig fold is **boot-hoisted** — `emit_pre_element_transitions`
     runs in the chain's one-time setup, not per tile — so this kernel pays
     `stages × num_blocks × NUM_W_CHUNKS` reconfigs per core (five, in total, on
     `(1,1,8192,1024)`), never `stages × tiles`. `master.md`'s 1.19x for that lever is a
     per-tile-reconfig measurement and does not transfer. The elision is therefore **not
     implemented**: there is nothing to elide. The named constants and the ablation switch
     stay, so the next reader gets the bound for free.
  2. **Lever 2 — Lamp L-RES-FUSE. The lamp's own four-element form is STRUCTURALLY
     WRONG, and that is the finding.** `Add → PackTile(cb_x_sum) → Square → PackTile(
     cb_x_squared)` cannot work: in `eltwise_chain` **pack is its own cohort**, disjoint
     from math-MOP/SFPU (`chain.inl elem_pack_init`), so every pack in a chain runs after
     every compute element — the first pack therefore publishes the SQUARE into cb_x_sum
     and pass B normalizes `t²`. Built, measured: **pcc 0.260** on `(1,1,8192,5120)`
     ROW_RESIDENT `gamma_bias_residual`, and 0.947x, so it was not even fast. Publishing
     `t` and squaring it in one DEST window needs a DEST→DEST copy element the chain does
     not expose — recorded as a helper gap, not worked around with raw LLK. Re-gated to
     the lamp's *real* boundary: `t` need not survive **only in STREAM**, where pass B
     rebuilds it, so there the chain is the three-element `Add → Square → Pack`. Correct
     there (pcc 0.999980, bit-comparable to the unfused pair) and **0.989x** on
     `(1,1,1024,16384)` STREAM `gamma_bias_residual` — the SFPU `square_tile` costs more
     than the saved unpack and pack. **Parked at 0, kept live.**
  3. **Lever 3 — reader/writer transaction granularity. MEASURED FLAT; both NoC halves
     moved together, never one alone.** `DM_TXN_ROWS_MAX` groups `TXN_ROWS` tile-rows into
     ONE reserve / issue run / barrier / push in the reader **and** the symmetric ONE wait
     / issue run / barrier / pop in the writer — the writer twin is in the same commit, so
     the bottleneck cannot just move across the CB. Swept `{WT_CHUNK, 2·WT_CHUNK,
     BLOCK_ROWS·WT_CHUNK}` (the queue's set) on top of the pass-A block:
     `(1,1,8192,1024)` 1.024 / 1.018, `+bias` 1.005 / 1.005, `(1,1,8192,2048)` 1.012 /
     1.010, STREAM 1.002 / 0.994, every sharded guard within noise — no consistent
     direction, and mildly negative at the coarsest setting on two cases. That is the
     trade the queue predicted: a coarser handoff buys barriers and costs reader↔compute
     overlap *inside* a block. **Parked at 1 (byte-identical to the seed's per-tile-row
     barrier), kept live**, with `TXN_ROWS | BLOCK_ROWS` asserted host-side and
     `static_assert`ed in both kernels — that divisibility is what makes a multi-tile-row
     reserve straddle-free on a `depth × BLOCK_ROWS × WT_CHUNK` ring, so it is a
     correctness invariant rather than a preference.
  4. **Lever 4 — Lamp L-OPERAND-TRIM. D23's derived policy WINS the re-measurement, and
     the bias copying it is RIGHT.** Each operand now carries its own override
     (`PER_CHANNEL_TRIM_GAMMA` / `_BIAS`) filtered through the same legality rule, so a
     forced granularity can never produce a truncated block-float read. Speedup vs the
     derived default (two face-rows):

     | variant | prefill 1024 g | +bias | 2048 g | 5120 gbr | W28c | W32c gbr | BLK 64c |
     |---|---:|---:|---:|---:|---:|---:|---:|
     | trim 1 (half page) | 0.953 | 0.980 | 0.979 | 1.004 | 0.930 | 0.930 | 0.986 |
     | trim 0 (whole tile) | 0.865 | 0.838 | 0.880 | 0.895 | 0.790 | 0.758 | 0.959 |
     | gamma 2 / bias 0 | 1.000 | 0.955 | 0.979 | 0.957 | 0.992 | 0.852 | 1.000 |

     So the lamp's hypothesis is **refuted with a number**: at this granularity fewer,
     bigger per-channel transactions LOSE (up to 0.758x), the trim is a byte-count win and
     not a transaction-count one, and two trimmed reads per chunk instead of one does not
     flip it. Both operands stay derived; the lamp is closed.
  5. **Folded in — `_cb_block_mult` over-prices `cb_x_squared`.** Corrected behind
     `CB_SQ_EXACT`, in the RESIDENT solve where the chunk is already known to be
     `wt_core`. It does change the blocking where the D12 fold is on and L1 binds — the
     64-core BLOCK shard goes BLOCK_ROWS 20 → 25, `(1,1,7168,1024)` 11 → 12,
     `(1,1,1024,512)` WIDTH 21 → 25 — and it measures **0.987x on `(1,1,8192,1024)` BLOCK
     64c**, one of the two shards the verifier flagged. The coarser block costs more (more
     L1, bigger rings, less pipelining) than the fewer rounds buy. **Parked at 0**, which
     is also what keeps `test_program_is_structurally_the_seeds` byte-identical.
  6. **NEW — the phase's actual win, and it is squarely this heading's subject.** Pass A's
     `square` chain was the ONE chain in the kernel still running at **DEST block_size 1**:
     one `tile_regs` handshake, one per-element init, one format reconfig and one CB
     reserve/push per TILE, while every pass-B chain (and `residual_add_block`) had taken
     `PASS_B_BLK` since D21 measured 1.28–1.66x for exactly that. `PASS_A_SQ_BLOCK` gives
     it the same block — **derived from `PASS_B_BLK`, never a second literal** — paired
     with the `PerBlockSize` reserve/push the blocked pack lifecycle requires (a `PerTile`
     reserve under `block_size > 1` reserves one page and packs `PASS_A_SQ_BLK`: a
     corrupted ring, i.e. a hang, which is why the two are one change). Inert under the
     D12 fold, whose `DestAccumulation::PerRow` already owns D0 for the whole tile-row.

- Perf achieved — shipped Refinement 3 vs Refinement 2's program (the ONLY difference is
  `PASS_A_SQ_BLOCK`), blackhole p150b, AICLK 1350 MHz = the reference clock (scale
  1.0000), in-process profiler, **min over 3 reps of median-of-5**:

  | case | R2 | R3 | x |
  |---|---:|---:|---:|
  | `(1,1,8192,2048)` INTERLEAVED `gamma` | 172 433 | **167 994** | **1.026** |
  | `(1,1,8192,1024)` INTERLEAVED `no_gamma` (the seed-parity build) | 84 517 | **83 087** | **1.017** |
  | `(1,1,8192,1024)` INTERLEAVED `gamma_bias` | 96 727 | **95 361** | **1.014** |
  | `(1,1,32,7168)` INTERLEAVED width-split 16c | 8 964 | **8 852** | **1.013** |
  | `(1,1,8192,1024)` INTERLEAVED `gamma` | 88 128 | **87 372** | **1.009** |
  | `(1,1,1024,16384)` STREAM `residual` | 290 399 | **288 460** | 1.007 |
  | `(1,1,8192,7168)` INTERLEAVED `gamma` | 576 680 | **573 699** | 1.005 |
  | `(1,1,32,7168)` WIDTH `[32,256]` (7,4) 28c | 5 560 | **5 530** | 1.005 |
  | `(1,1,8192,5120)` INTERLEAVED `gamma_bias_residual` | 667 621 | **665 557** | 1.003 |
  | `(1,1,256,512)` ROW_MAJOR BAND 64c | 23 556 | 23 515 | 1.002 |
  | `(1,1,8192,1024)` BLOCK `[1024,128]` 64c | 23 570 | 23 568 | 1.000 |
  | `(1,1,32,5120)` WIDTH 32c `gbr` / `(1,1,7168,1024)` BLOCK 64c `gbr` | 6 299 / 33 019 | 6 308 / 33 036 | 0.999 |
  | `(1,1,1024,16384)` STREAM `gamma_bias_residual` | 522 549 | 524 550 | 0.996 |

  **Nothing regresses**; the win is 1.005–1.026x concentrated exactly on the interleaved
  prefill this heading names, and the operand-free build — the one
  `test_program_is_structurally_the_seeds` pins — gets **faster (1.017x) with a
  measurement attached**, which is the prompt's rule for touching it. Against the machine
  ceiling the prefill now runs at **384 GB/s** at W=1024 (`clone` 400) and **409 GB/s** at
  W=7168, i.e. **3% faster than a pure DRAM copy of the same bytes**.
- Accuracy achieved: PCC 0.999980–0.999992 and rel-RMS unchanged on every geometry in the
  sweep, at every variant — identical to R2's digits on all fourteen cases. The shipped
  change moves WHEN work is issued, never what: the square's operands, order and DEST
  slots are the same tiles in the same order. The one variant that DID move the numbers
  (the four-element L-RES-FUSE, pcc 0.260) is the one that is gated off.
- Golden test progress: `test_op_loose` **433 passed / 10 failed / 3 skipped** — the
  identical Phase-0 and Refinement-2 figure, all 10 the same harness-attributed failures
  (`CoreRange.end_coord`, `torch.max()` on a zero-element readback), none op-attributed.
  Plus a **2 196-cell** `test_op` cartesian slice over `1x1x2048x256 / 1x1x32x4096 /
  4x1x512x512` (every layout × placement × dtype × operand mode), all green.
- Issues encountered:
  1. **The lamp's fused chain is inexpressible, not just slow.** See lever 2: pack being
     its own cohort means a chain can publish only its FINAL DEST value, and the failure
     mode is a silent wrong answer (pcc 0.260), not a hang or a compile error. The gate
     comment in the compute kernel carries the mechanism so a later phase cannot re-derive
     the same broken chain.
  2. **A per-stage zone can invert the truth under DRAM contention.** The first breakdown
     of `(1,1,8192,1024)` showed `reader_read_gamma` at **60 000–72 000 cycles on the
     slowest cores** (54 µs of an 89 µs op) and made the gamma read look like the whole
     problem. It is not: the no-gamma build is only 4.9 µs faster. All 110 cores issue
     their gamma reads at t=0 against a DRAM the same 110 cores are saturating, so the
     zone measures *occupancy under contention*, not payload — exactly the trap
     `device-zone-scope-attribution.md` warns about. The end-time spread across cores
     (27 k → 120 k cycles at essentially zero start skew) is the honest signal.
  3. **`ttnn.NOC`-style enum aliasing bit nothing here**, but two knobs did have to be
     ordered inside the kernel: `PASS_B_BLK` is now defined ahead of the pass-A output
     spec that derives `SQ_BLK` from it (one source of truth, so the definition moves
     rather than being duplicated), and `RES_FUSE` after `X_RESIDENT`.
- Tests added:
  - `tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_dataflow_knobs.py`
    (**292 cells**) — the five knobs' shipped values, that each is still LIVE (flipping the
    module constant must move the CT arg), the packed transaction word's byte-identity at
    the default, the `TXN_ROWS | BLOCK_ROWS` straddle invariant swept over every block size
    1..40 × seven caps, and that a forced face-row trim on `bfloat8_b` falls back to the
    half page rather than issuing a truncated read. All host-side; nothing dispatches.
  - `tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes/bench_r3.py` — the reusable
    14-case A/B harness (6 interleaved-prefill targets, 2 STREAM, 6 guards) every number
    above came from.
  - Whole unit directory after the change: **601 passed, 1 skipped**.

## Refinement 4 — Remove the prime-`Wt` granularity cliff (ragged width chunk)

- Date: 2026-09-06
- What was done:

  **The cliff, restated from the code.** `WT_CHUNK` was constrained to a **divisor** of the
  per-core width (D1), so a prime `Wt` had nothing between 1 and the whole row: at
  `W = 4064` (`Wt = 127`) every chunked build came out at `WT_CHUNK = 1,
  NUM_W_CHUNKS = 127`, repaying one per-phase init / reconfig / dst-sync window /
  pipeline fill-and-drain **per width tile** instead of per chunk. Confirmed on device
  before touching anything (`RMS_TRACE_BLOCKING`, added as a permanent one-line dump of
  the solve): `(1,1,32,4064)` and `(1,1,3104,4064)`, TILE and ROW_MAJOR, all at 1×127.

  **What shipped, and what it deliberately is NOT.** The design filed this regime as "a
  runtime `wt_c` instead of a compile-time divisor", which would have meant changing three
  helper-side mechanisms (`tilize`/`untilize`'s compile-time `block_width_tiles`, the
  reduce's `num_pages % cols == 0` assert, and the no-straddle rule on a multi-page
  reserve). It shipped instead as the **coarsest BALANCED chunk with the last chunk
  PADDED** — `WT_CHUNK = ceil(wt_c / ceil(wt_c / cap))`, `NUM_W_CHUNKS = ceil(wt_c / WT_CHUNK)` —
  which keeps every chunk **uniform** and therefore satisfies all three mechanisms
  verbatim. `127` at a cap of 32 becomes 4 chunks of 32 with **one** pad tile. The pad is
  bounded by construction (`pad < NUM_W_CHUNKS`, i.e. under `1/cap` of the width), which is
  why the balanced form and not a greedy `cap`-wide one.

  **The compute kernel is byte-for-byte untouched.** `X_HOLD_WT` was already
  `WT_CHUNK * NUM_W_CHUNKS`, every chain already takes a runtime `IterationShape::grid`,
  and the reduce already takes a runtime block shape. Only the two dataflow halves moved,
  and both moves are the op's **existing ragged-width-SHARD machinery applied one axis
  over**: the reader zeroes the pad tiles with the device zero API (exactly
  `publish_native_shard`'s mechanism and exactly its reason — a pad tile that is not
  *exactly* zero inflates `sum(t²)`), and the writer skips them with the `wt < WT`
  predicate it already carried. The ROW_MAJOR tail needed one new nine-line pair
  (`stage_ragged_tail_sticks` + its writer mirror), because
  `read_sticks_for_tilize` / `write_sticks_after_untilize` derive the L1 **stride** from
  `row_bytes` and the padded tail needs the two to differ; that is recorded at both sites
  as a raw-API bypass with the helper gap that would close it.

  **Two knobs, one source of truth each.** `RAGGED_WIDTH_CHUNK` (ships at 1; `0` restores
  D1 exactly and is what the A/B below flips) and `ROW_RESIDENT_MIN_CHUNK_WT` (ships at 1,
  its byte-identical default — measured flat once the ragged chunk landed, kept live).
  `_width_chunk` is the single place the chunk-count decision is made; `wt_pad` is derived
  once and read by the CB sizes and both kernels. The reader carries `WT_PAD` as its own
  CT scalar; the **writer packs it into `WT_CHUNK`'s word** (index 2 high half), because
  the writer's CT-arg list LENGTH is a checked structural property
  (`test_program_is_structurally_the_seeds`) and index 4 / index 15 already establish that
  idiom.

  **The gate, and why it is not conservatism.** Ragged is taken only when
  `PARTIAL_W == 0`. With a non-tile-aligned width the reduce's partial scaler / 0-1 mask is
  aimed at the **last tile of the block**; padding would make that a pad tile and silently
  drop the mask off the real last tile. A non-aligned width keeps D1's divisor clamp.

  **One real bug found and fixed on the way.** The first ragged candidate is priced against
  a hold of `wt_core`, but the held CBs span the **padded** row — so at `Wt = 127` the
  19×7 candidate (6 pad tiles) missed by one tile of L1, and the fallback was
  "take the divisor", i.e. **1** — the cliff itself. The solve now re-caps against the
  candidate's own pad and retries (a strictly-shrinking fixed point), landing on 16×8.
  That is the difference between the RM `gamma_bias_residual` cells being 9.19× faster and
  being unchanged.

- Accuracy achieved: PCC **improves** everywhere the chunk coarsened —
  0.999870 → 0.999988 on `(1,1,3104,4064)` `gamma_bias_residual`, 0.999961 → 0.999987 on
  `gamma`, 0.999975 → 0.999991 operand-free — because a coarse chunk clears D7/D8's
  reduce-datapath floors and routes the reduce through `AccumulateViaAdd`, which is the
  precision lever Refinement 1 built. rel-RMS well inside the 0.04 bound on every prime-`Wt`
  cell in both layouts; the guard set is bit-for-bit unchanged (`WT_PAD == 0` ⇒ identical
  program).

  **Measured device-ns** (blackhole p150b 1350 MHz, in-process profiler, min over 2 reps of
  median-of-5; `RAGGED_WIDTH_CHUNK` 0 vs 1, one fresh-cache measurement per variant):

  | shape | layout | mode | divisor | ragged | speedup |
  |---|---|---|---:|---:|---:|
  | `(1,1,32,4064)` | RM | gamma_bias_residual | 590 592 | 56 486 | **10.46x** |
  | `(1,1,32,4064)` | RM | gamma | 376 319 | 37 950 | **9.92x** |
  | `(1,1,3104,4064)` | RM | gamma_bias_residual | 1 991 000 | 216 590 | **9.19x** |
  | `(1,1,3104,4064)` | RM | gamma | 1 145 067 | 138 309 | **8.28x** |
  | `(1,1,3104,2848)` | RM | gamma | 805 205 | 98 090 | **8.21x** |
  | `(1,1,3104,4064)` | RM | no_gamma | 992 721 | 133 170 | **7.46x** |
  | `(1,1,32,4064)` | TILE | no_gamma | 101 279 | 16 616 | **6.10x** |
  | `(1,1,32,4064)` | TILE | gamma | 148 493 | 33 278 | **4.46x** |
  | `(1,1,3104,4064)` | TILE | gamma | 213 058 | 150 576 | **1.42x** |
  | `(1,1,3104,4064)` | TILE | gamma_bias_residual | 329 830 | 246 944 | **1.34x** |
  | `(1,1,1024,16384)` | TILE | gamma_bias_residual | 525 549 | 498 670 | **1.05x** |

  The last row is a **non-prime** width and was not a target: `Wt = 512`'s coarsest divisor
  under the cap is 32, but its coarsest *fitting* chunk is 57. D33 pays wherever the two
  differ, not only at a prime.

  Guards (fourteen cases spanning the interleaved prefill, STREAM, the width-split combine,
  the 64-core BLOCK shard, and the RM BAND): **0.99–1.05x**, nothing outside noise. The two
  that looked like dips at 2 reps (`P4` 0.980, `P6` 0.995) both build `NUM_W_CHUNKS == 1` or
  a divisor chunk — i.e. a byte-identical program — and came back 1.014 / 0.991 at 3 reps.

- Golden test progress: `test_op_loose` **433 passed / 10 failed** — the identical prior
  figure, and the same 10 harness-attributed failures (`torch.max` on an empty tensor;
  `CoreRange.end_coord` in the harness's sharded program-config helper). The
  `resilience` + `perf` + `pad_poison` groups: **384 passed / 3 skipped, 0 failed**. A
  2 700-cell strict-cartesian slice (`1x1x32x8192`, `1x1x2048x256`, `2x1x64x4096`,
  `1x1x17x50` × the full axis product): green. Unit directory **1 061 passed / 7 skipped**.

- Issues encountered:
  1. **The pad re-pricing fallback was the cliff.** See above — fixed by re-capping instead
     of falling back to the divisor.
  2. **`read_sticks_for_tilize` cannot express a padded tail.** It derives
     `padded_row_bytes` (the destination stride) from `row_bytes` (the bytes read), so a
     tail chunk staged with the real `row_bytes` lands at the *real* stride while
     `tilize<WT_CHUNK>` reads it back at the *padded* one. Handled with a raw strided read
     shaped exactly like the BAND scheme's `stage_band`, with a single zero call covering
     every stick's pad lanes; the helper gap (separate "bytes read" and "stride written"
     parameters) is recorded at the call site.
  3. **The writer's CT-arg list length is load-bearing.** Appending a 19th scalar broke
     `test_program_is_structurally_the_seeds` by construction. Packed into index 2 instead,
     which is the file's own established idiom.

- Tests added: `tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_ragged_width.py`
  (174 cases) — `_width_chunk`'s covering/cap/no-worse-than-D1 contract over a
  width × cap grid, both knobs proven live, the cliff proven gone on every chunked
  prime-`Wt` build in both layouts, reader/writer pad-count agreement, the `PARTIAL_W != 0`
  gate, and on-device PCC/rel-RMS at the prime widths across every operand combination.
  Plus `probes/bench_r4*.py` via `bench_r3.py`'s harness (the A/B above).

## Refinement 4b — Remove the prime-`Wt` granularity cliff (debug: the golden run reported TOTAL=0)

- Date: 2026-09-06

- What was done: **the ragged width chunk was never the problem.** The harness's completion
  gate failed Refinement 4 with `REGRESSION — prior-passing golden cells no longer pass
  (responsible cells 0/0)`, and its `golden_results.txt` read
  `PASSED=0 FAILED=0 ERRORS=0 SKIPPED=0 HANGS=0 TOTAL=0` — nothing at all was recorded.
  Reading that run's `pytest_stdout.log` shows why: `test_golden.py` ran **to completion**
  with the same 10 harness-attributed failures as every prior round, and the process then
  died with `Fatal Python error: Aborted` at `profiler.cpp:373` on the
  `test_golden.py → test_regression.py` device re-open, before pytest could write
  `junit.xml`. No junit, no results — every cell scored as "never ran", i.e. a regression.

  **Root cause: a 16-bit device-zone-hash collision.** `populateZoneSrcLocations()` keys
  every device zone by `hash16CT("<zone>,<abs source path>,<line>,KERNEL_PROFILER")` and
  hard-`TT_THROW`s the moment two DISTINCT strings share a slot. The eval golden runner sets
  `TT_METAL_DEVICE_PROFILER=1` (to read the op-level `DEVICE KERNEL DURATION` off the
  *firmware* markers), which as a side effect compiles this op's **41** `MaybeDeviceZoneScope`
  sites in as well — 41 entries in a 65 536-slot table, i.e. a ~2–3% birthday collision every
  time a kernel edit moves a line. Refinement 4's edits moved `writer_tree_forward` onto
  `rms_norm_ttnn_writer.cpp:662`, which collides with `compute_scale`@`compute.cpp:1680` at
  **0x0773**. Reproduced locally: 3 throws on a 448-cell slice with the profiler env, 0
  without. The throw is caught at most sites (hence 47 089 of them in the log and the tests
  still passing) and escapes as `terminate` only at a device re-open.

  Three changes, in increasing order of durability:

  1. **`writer_tree_forward` moved off line 662** (a four-line comment above it), which
     breaks *this* collision.
  2. **D34 — the per-stage zones are now OPT-IN.** `MaybeDeviceZoneScope` compiles only under
     the `RMS_STAGE_ZONES` kernel define, which `STAGE_ZONES` / `_kernel_defines()` in the
     descriptor emits from the env var of the same name (ONE source of truth, plumbed to all
     six `KernelDescriptor` sites). A graded run now registers **zero** op zone locations, so
     the collision class cannot reach it at all; `RMS_STAGE_ZONES=1` restores every zone for a
     perf round. The zones themselves are untouched and still permanent — the durability
     contract in `perf_instrumentation.hpp` is amended, not weakened. Empty defines means the
     build key is unchanged, so a non-profiled build is byte-identical.
  3. **Purged the stale preprocessed artifacts that replay old zone strings.**
     `extract_zone_src_locations()` harvests zone pragmas out of each build dir's `.ii` /
     `*.o.log` **on the ELF-reuse path too**, and a build dir is keyed coarsely enough to be
     reused across source edits — so `.ii` files preprocessed from superseded kernel versions
     kept re-registering line numbers the tree no longer has (this is why the *source* fix
     alone did not clear it: the hash table is populated from the cache, not the tree).
     115 534 stale `.ii`/`*.o.log` under `built/*/kernels/rms_norm_ttnn_*/` older than the
     gating change were deleted (ELFs untouched, cache stays warm), and the op's rows dropped
     from `generated/profiler/.logs/*zone_src_locations.log`. Deleting them mid-flight during
     the verification run stopped the throws immediately — the last 100 k log lines went from
     ~7 000 collisions to 0.

  **No functional change to the op.** Not one line of the ragged-width-chunk work
  (`_width_chunk`, `WT_PAD`, the reader's pad zeroing, the writer's tail write) was altered.

- Accuracy achieved: unchanged — the op is bit-identical to Refinement 4 in every
  non-profiled build. Full golden suite: **PASSED=23348 FAILED=19 ERRORS=0 SKIPPED=98071
  REFUSED=0 HANGS=0 TOTAL=121438**, which is the byte-identical figure from
  `golden_phase0`, `golden_refinement_1`, `golden_refinement_2` and `golden_refinement_3`.
  Zero regression, zero hangs, suite ran to completion and wrote its junit.

- Golden test progress: **23348 / 23348** non-xfail cells passing (19 known
  harness-attributed failures, unchanged in count and identity from Phase 0).

- Issues encountered:
  1. **The source fix alone was not enough** — see (3) above. Diagnosing that took reading
     `build.cpp`'s `extract_zone_src_locations()`: the `.ii` harvest runs on cache hits, so a
     501 GB cache carrying every historical build of these kernels was re-injecting ~40
     distinct line numbers per zone. Worth remembering: after ANY profiling session run with
     `RMS_STAGE_ZONES=1`, purge those artifacts or a later graded run will harvest them.
  2. **`hash16CT` is only 16 bits and the table is global.** 41 zones is a lot to spend of
     65 536 slots for one op; the birthday budget is now asserted (`<= 128` including the
     firmware markers).

- Tests added: `tests/ttnn/unit_tests/operations/rms_norm_ttnn/test_rms_norm_ttnn_zone_hashes.py`
  (6 cases) — re-implements the profiler's `hash32CT`/`hash16CT` (pinned against two
  clone-path-independent fixtures), asserts no two of the op's CURRENT zone source locations
  collide in 16 bits, asserts the zone population stays under the birthday budget, guards that
  the macro is still discoverable, and adds a pre-flight check on the accumulated
  `zone_src_locations.log` that fails with the exact purge command when the machine is in the
  state that killed the Refinement 4 run.

## Perf 1 — The reader was holding the compute kernel hostage

- Date: 2026-09-06
- Type: perf tournament, round 1 of 3. **No `SUPPORTED` / `EXCLUSIONS` change** — nothing in
  the registry declarations was edited, so `verify_supported`'s categories are untouched by
  construction. The signal is device-ns.
- Box: blackhole p150b, measured AICLK **1350 MHz = the reference clock**, so every
  clock-scale factor below is exactly 1.0000.

### Focus shape — chosen by the mandated ranking, not by judgement

No `feature_spec.LOOSE_CASES` entry for this op carries an `attention:` note, so step 1 falls
through to its second branch. Every one of the **19 `perf`-group cases** was measured and
divided by its own clock-scaled `achievable_ns` (÷ `minimum_expected_speedup` where present).
All 19 were already under their ceiling; the largest ratio was

> **`(1,1,32,7168)` WIDTH_SHARDED, shard `[32,256]`, grid `(7,4)` = 28 cores, `gamma`
> (bf16 TILE weight), bf16 / HiFi2 / `fp32_dest_acc_en=False` / `math_approx_mode=False`
> — 5463 ns against a 5481 ns ceiling, ratio 0.997.**

Every knob that case declares is in `SUPPORTED`, so it was optimized at its exact config; no
proxy was substituted. Regime: SHARD_W native + cross-core width combine, `GROUP_SIZE = 28`,
exactly **one** combine round, `BLOCK_ROWS = 1` (so the COMPACT permute is the identity) and
the combine tree OFF (`_combine_tree_arity` refuses at G=28 — already measured, every arity
loses there).

### Measured breakdown — a critical-path timeline, not a stage histogram

`RMS_STAGE_ZONES=1`, one fresh-cache run, all 41 zones parsed out of
`profile_log_device.csv` and laid out as an absolute per-core timeline (which is what makes
the *dependency* stalls visible rather than merely the occupancies). Critical path on a plain
member core, ns from BRISC-FW start; whole op 5408 ns median:

| window | stage | ns | |
|---|---|---:|---|
| 0 – 556 | dispatch / firmware preamble | 556 | not ours |
| 607 – 914 | `reader_scaler_boot` | 307 | |
| 947 – 2007 | `reader_read_gamma` | **1060** | **blocks the x publish** |
| 2032 – 2110 | `reader_native_publish` | 78 | what compute is waiting on |
| 2110 – 2491 | `compute_square` + `compute_reduce` (pass A) | 381 | |
| 2491 – 2794 | `writer_gather_ship` | 303 | |
| → 3765 | root `writer_gather_wait` | 971 | bounded by the SLOWEST member |
| 3765 – 4333 | root `compute_root_fused` (fold 27 + fused rsqrt) | 568 | |
| 4333 – 4915 | root `writer_mcast_send` | 582 | |
| 4931 – 5126 | `compute_scale` real work after recv | 195 | |
| 5150 – 5559 | `compute_gamma_mul` | 409 | |
| 5559 – 5976 | kernel / firmware teardown | 417 | not ours |

`reader_read_x` and `writer_write` measure **~0 ns** — the shard is native/zero-copy on both
ends, so the sharding precondition holds and there is no escaped completeness bug to report.

**The ranked bottleneck was a serialization, not a cost.** `reader_read_gamma`'s cross-core
END spread is **541 ns**, and that spread propagates verbatim into `compute_reduce` END
(spread 541) and `writer_gather_ship` END (spread 827) — so the root's 971 ns arrival window
was bounded by the slowest member's *gamma read*, not by ingress bandwidth. Confirmed by
ablation: **removing gamma entirely (read + multiply) is 5408 → 3829 ns, 29% of the op.**

Roofline gate (`/perf-ceiling-dm`): the gather ships 2 faces × 1024 B per member
(`GATHER_FACES = 2`), i.e. 27 × 2048 B in 971 ns ≈ 57 GB/s — **not** byte-bound, which
reproduces D13's own finding, so no idea was spent on shrinking its bytes. The 1 KB gamma
read taking 1060 ns is pure DRAM *latency*, which is a thing to overlap, not to speed up.

### Portfolio floated (6 ideas, one `perf-part-optimizer` each, in parallel)

| # | idea | tier | verdict |
|---|---|---|---|
| 1 | `reader_boot_order` — hand the resident shard to compute FIRST | T1 | **WIN 1.295x** |
| 2 | `gather_transport` — cut the member→root gather | T2 | **WIN 1.038x** |
| 3 | `fold_gather_overlap` — fold partials as they land | T3 | **split: NULL + WIN 1.013x** |
| 4 | `mcast_transport` — cut the 582 ns stat broadcast | T2 | NULL |
| 5 | `passb_fusion` — one DEST window for normalize × scale × shift | T2 | REGRESSION (+ a WIN side finding) |
| 6 | `per_channel_reuse_mcast` — the standing reuse-shared broadcast | T3 | WIN in domain, **not graduated** |

Ideas 1 and 6 deliberately overlapped (both attack the per-channel read); 3 and 4 both attack
the combine's tail. Overlap was resolved at aggregation, never among subagents.

### Per-idea results

**1 — `reader_boot_order`. WIN, and the round's headline.** The `NATIVE_X`
`publish_native_shard` block moved from the bottom of the reader to the top. Four options
measured: (a) publish first **4121 ns**, (b) scaler → publish → per-channel 4247, (c) split
the per-channel NoC issue from its barrier 4687, (d) = (a)+(c) **4118**. Graduated **(a)**:
(d) is +3 ns on the focus shape for ~198 diff lines and a `capability` bypass of
`stage_per_channel_chunk`, and its real value (1.04x) is confined to the interleaved
prefill — **parked and recorded**, not lost. Structural floor for reference: the `no_gamma`
build is 3767 ns, and (a) reaches 4121 *with* gamma, so the read is essentially fully hidden.
Bit-identical output at every option.

**2 — `gather_transport`. WIN.** `Noc::async_writes_flushed()` ("departed") replaces
`noc_async_write_barrier()` ("acknowledged") before the gather semaphore on every REMOTE
ship. Isolated 28-core bench 1827 → 1680 ns (1.087x); whole op 5342 → 5145. Also priced, and
recorded so nobody re-derives them: `GATHER_FACES` 3 and 4 re-confirmed slower (0.839x /
0.730x); a dual-NoC gather is worth ~400 ns more but `get_noc_addr(..., noc=1)` from the
BRISC writer **hangs** under `DM_DEDICATED_NOC` and is a two-kernel shape that would also
break `COMBINE_MCAST_FIRE_AND_FORGET`'s ordering licence; `llk_math_transpose_dest` inside
the fold's DEST window **hangs** on a `TTI_STALLWAIT` for `SRCA_VLD` that `add_tiles` has
just cleared.

**3 — `fold_gather_overlap`. Half NULL, half WIN.** The idea proper — prefix/threshold
folding — is a **measured null with an exact explanation**: the overlap gain (fold 565 → 321
ns) equals the extra publish overhead (6 × ~39 ns = 234 ns) at every run count, and it cannot
do better because arrival *order* is uncorrelated with slot order, so every run of k slots
completes at the max of k arrivals. It stays null on a de-skewed build too. The WIN is
option (d), a different change found on the way: **delete the root's serial 3 kB L1→L1
publish** of the multicast landing page by having the fold pack straight into it. 1.013x–
1.025x on the focus shape, 1.006x–1.024x on every other combine geometry, bit-identical.

**4 — `mcast_transport`. NULL, and it re-attributed a stage.** Seven correct transports
measured (one-packet local publish, caller-managed flush, 4 faces, raw one-packet bypass,
`MCAST_INCL_SRC` loopback, …); every one landed at 0 ± noise or worse (loopback −2.1%,
faces4 −1.0%, raw one-packet −0.9%). The **ceiling** for any transport rewrite, from an
ablation that deletes the broadcast entirely, is 246 ns / 1.048x. More useful than the null:
only ~330 ns of the 582 ns `writer_mcast_send` stage is transport — the other ~250 ns is the
root waiting on its own fold, which is idea 3's territory, and the two are additive.

**5 — `passb_fusion`. REGRESSION.** Fusing `normalize × gamma [+ bias]` into one DEST window
is 0.972x on the focus shape and **0.818x** on the 64-core BLOCK shard. This independently
reproduces the 0.84x already in `op_design.md`, at a different geometry and by a different
mechanism, and the mechanism is the finding: pass B's three compute threads are **already
balanced** (unpack 8520 / math 8556 / pack 8110 ns per core), so the unpack and pack the
fusion deletes were free; and `eltwise_binary_run_with_dest_reuse` restarts the MOP **per
face** with a `move_d2a_fixed_face` + `TT_ZEROACC`, so the DEST-reuse mul is ~71 ns/tile
against the helper's one-MOP bcast mul at ~32 ns/tile. Not graduated. **Its side finding was
graduated** — see 5b.

**5b — `PASS_B_BLK` re-sweep on the *shipped* kernel. WIN, bit-identical.** Measured on the
baseline kernel with no fusion at all: the auto rule wants to be the *smallest* divisor ≥ 2 at
`BLOCK_ROWS == 1` and D21's largest divisor above it. Graduated as D38.

**6 — `per_channel_reuse_mcast`. WIN in its domain; NOT graduated.** The op's standing
reuse-shared broadcast idea, built against `op_design.md`'s deferred `GAMMA_MCAST` row. It
closes that deferral with numbers: **+2.2% to +7.4%** on the row-split interleaved prefill
against a same-build control (boot per-channel read 34.4 µs → 5.7 µs per core, 6×, converting
to only ~5% of the op because those shapes are DRAM-throughput-bound and the freed DRAM time
is re-absorbed by `reader_read_x`). It is **structurally inapplicable at the focus shape** —
one tile-row means all 28 cores own disjoint width slices, so there is no reuse to remove —
and a **measured regression on BLOCK shards** (−1.6% / −2.3% vs same-build: a 4-tile per-core
slice is latency-bound tiny reads, and the mcast adds a handshake plus an 8-of-64-core boot
straggler). **Blocked on integration, not on the idea**: the candidate carries an
unattributed same-build overhead that regresses *non-engaged* plans ~4–5% against the shipped
baseline (focus 5363 → 5577, BLOCK 22881 → 23975, BAND 23000 → 24172), i.e. its off path is
not yet byte-identical. Graduating that would have regressed supported cells, so it is
requeued for round 2 with "make the off build byte-identical" as the entry condition. Largest
remaining prize it identified: **STREAM, +20.1% ablation ceiling**, the biggest in the op.

### What graduated, and how widely

Four changes, all as **one unqualified path**, with the code they replace deleted. Three carry
no predicate at all; one carries a single carve-out earned by a measured regression.

| # | change | domain | carve-out |
|---|---|---|---|
| D35 | the reader publishes the resident shard FIRST | every plan, all layouts/placements. `if constexpr (NATIVE_X)` already scopes the *block*; the ordering is unconditional | none |
| D36 | the gather signals on *departed* | every remote gather ship: flat member, tree level-0 leaf, tree level-1 forward. Non-combine plans byte-identical | **local** ships keep the barrier — a gatherer writing its own L1 has no atomic to order against and its own fold reads those bytes back. That is *feasibility*, not a perf exception |
| D37 | the root's fold packs into the multicast landing page; the 3 kB L1→L1 copy is **deleted** | every combine geometry: flat and tree, identity and COMPACT, 1 / 2 / 3 rounds, both `fp32_dest_acc_en` | `static_assert(!FIN_SPREAD)` — under the spread finalize the last level forwards a RAW sum so its pack is not the stat. `COMBINE_FIN_SPREAD` is measured off, so this guards a **dead path**, not a live regime |
| D38 | pass B's auto DEST block is the smallest divisor ≥ 2 | every `BLOCK_ROWS == 1` block — WIDTH shards, the HEIGHT shard, interleaved decode, and every untested regime with one tile-row | **`BLOCK_ROWS > 1` keeps D21's largest-divisor rule**, earned by a MEASURED regression (0.958x on `(1,1,8192,1024)` BLOCK 64c, 0.976x on `(1,1,7168,1024)` BLOCK gbr). Written as `if (cannot) { legacy } else { new }` so the exception shrinks, not the win |

The tree's **level-1 forward** was outside the subagent's measured set and got D36 anyway: the
safety argument is identical and fencing it off would have frozen the win at the size of the
test matrix. It is covered by the G=32 tree cell below.

**A latent bug the round exposed, and fixed.** D35 deleted an *implicit* ordering nobody had
written down: `cb_scaler` and `cb_bank` are reader-synthesized constants the compute kernel
reads with **no wait of its own** — the reduce helper's contract puts that on the caller
(`reduce_helpers_compute.hpp:37`) and `matmul_tiles` has no CB lifecycle at all — so their
availability was guaranteed only by the publish running last. Hoisting it produced a
**non-deterministic** corruption on the COMPACT combine path: `1x1x2048x256` and
`4x1x512x512` BLOCK_SHARDED at pcc 0.08–0.12 / rel-RMS 1e5, with a *different* set of cells
failing on every run, because the permutation matmul was multiplying against a bank the reader
had not finished zeroing. The op now waits both fronts **explicitly, one-shot, at first use**
(`scaler_ready` / `bank_ready`). At first use and not in a joint prologue: a prologue makes
`compute_square` — which needs neither constant — block on both, and that cost 7–8% on the
COMPACT BLOCK shards (22001 vs 20453 ns on `(1,1,8192,1024)` BLOCK 64c). One-shot and not
per-chunk so the streaming regimes pay it once rather than hundreds of times per core. The
zones were extended to every changed path and `test_rms_norm_ttnn_zone_hashes.py` re-run
green after each edit.

### Whole-op result — all 19 `perf` cases, before → after

Median of 3, in-process profiler, `DEVICE KERNEL DURATION`. `ratio` = measured / clock-scaled
ceiling.

| # | case | before | after | x | ratio before → after |
|---|---|---:|---:|---:|---|
| 11 | **`(1,1,32,7168)` W `[32,256]` (7,4) 28c — FOCUS** | **5463** | **3801** | **1.437** | **0.997 → 0.693** |
| 16 | `(1,1,32,5120)` W `[32,160]` (8,4) gbr fp32_dest=True | 6076 | 4177 | 1.455 | 0.927 → 0.637 |
| 9 | `(1,1,32,2304)` W `[32,256]` (9,1) 9c | 4256 | 3002 | 1.418 | 0.922 → 0.650 |
| 8 | `(1,1,32,1024)` W `[32,128]` (8,1) 8c | 3504 | 2624 | 1.335 | 0.853 → 0.638 |
| 10 | `(1,1,32,5120)` W `[32,160]` (8,4) 32c (tree on) | 4601 | 3498 | 1.315 | 0.874 → 0.664 |
| 12 | `(1,1,8192,1024)` BLOCK `[1024,128]` (8,8) 64c | 22961 | 20448 | 1.123 | 0.802 → 0.714 |
| 17 | `(1,1,7168,1024)` BLOCK `[896,128]` (8,8) gbr | 32346 | 29027 | 1.114 | 0.936 → 0.840 |
| 0 | `(1,1,32,1024)` INTERLEAVED | 4651 | 4335 | 1.073 | 0.508 → 0.474 |
| 1 | `(1,1,32,2304)` INTERLEAVED | 5471 | 5183 | 1.056 | 0.322 → 0.305 |
| 2 | `(1,1,32,5120)` INTERLEAVED | 7419 | 7047 | 1.053 | 0.098 → 0.093 |
| 3 | `(1,1,32,7168)` INTERLEAVED (≥7x cell) | 8792 | 8494 | 1.035 | 0.590 → 0.570 |
| 13 | `(1,1,32,5120)` INTERLEAVED gbr fp32_dest=True | 9930 | 9525 | 1.043 | 0.063 → 0.060 |
| 14 | `(1,1,8192,5120)` INTERLEAVED gbr fp32_dest=True | 669301 | 651473 | 1.027 | 0.509 → 0.496 |
| 18 | `(1,1,128,4096)` INTERLEAVED, ROW_MAJOR weight | 11421 | 11276 | 1.013 | 0.175 → 0.173 |
| 4 | `(1,1,8192,1024)` INTERLEAVED prefill | 86992 | 86996 | 1.000 | 0.899 → 0.899 |
| 7 | `(1,1,8192,7168)` INTERLEAVED prefill | 578937 | 578296 | 1.001 | 0.561 → 0.560 |
| 15 | `(1,1,8192,7168)` INTERLEAVED gbr | 1582835 | 1584244 | 0.999 | 0.861 → 0.862 |
| 5 | `(1,1,8192,2304)` INTERLEAVED prefill | 190242 | 191531 | 0.993 | 0.900 → 0.906 |
| 6 | `(1,1,8192,5120)` INTERLEAVED prefill | 416449 | 419723 | 0.992 | 0.564 → 0.568 |

**Every case is under its ceiling and the worst ratio moves 0.997 → 0.906.** The two cells
below 1.00x (#5 at 0.993 and #6 at 0.992) are not regressions and the reason is structural,
not statistical: on a `(1,1,8192,W)` interleaved prefill the row split fills the grid, so
`CROSS_CORE == 0` and `NATIVE_X == 0`, which `if constexpr`-eliminates **every** graduated
hunk except a moved `x_tile_bytes` declaration and one one-shot `cb_wait_front`. Those two
programs are functionally the shipped ones. Across four measurement sessions of this build
#6 read 417281 / 421490 / 421949 / 419723 (spread 1.1%) and #5 read 191845 / 188984 / 191227
/ 191531 (spread 1.4%), which brackets both deltas; and Refinement 3 already established that
these shapes sit AT the box's ~400 GB/s DRAM roofline, where there is nothing left to move.

### Guard set — one representative per distinct kernel path × layout × placement

Measured against a true HEAD build of the same cases (not against the perf-case table).

| path | before | after | x |
|---|---:|---:|---:|
| HEIGHT shard, native, no combine — `(1,1,2048,256)` | 4876 | 3510 | **1.389** |
| ROW_MAJOR BAND, WIDTH-sharded — `(1,1,256,512)` 64c | 23141 | 22599 | **1.024** |
| interleaved decode + AUTO width split — `(1,1,32,7168)` | 8792 | 8411 | **1.045** |
| interleaved, TILE weight, fp32_dest=True — `(1,1,128,4096)` | 11951 | 11603 | **1.030** |
| STREAM, `gamma_bias_residual` — `(1,1,1024,16384)` | 501587 | 504124 | 0.995 (flat) |
| ragged-`Wt` interleaved — `(1,1,32,4064)` | 32358 | 32339 | 1.001 (flat) |

Plus the four combine topologies exercised above: flat root at G=8/9/28, the slot tree at
G=32, COMPACT at 2 and 3 rounds, both `fp32_dest_acc_en`. **No material regression anywhere.**

### Golden

`scripts/run_safe_pytest.sh --run-all eval/golden_tests/rms_norm_ttnn/`, run as 10
`pytest-split` shards (the suite exceeds a single foreground window):

> **PASSED = 23348, FAILED = 19, ERRORS = 0, HANGS = 0.**

That is **byte-identical to the Phase-0 / Refinement-1 / -2 / -3 / -4b figure**, in count and
in identity — the same three harness defects Phase 0 reproduced outside the op and recorded in
`verification_report.md`: `CoreRange.end_coord` on a build exposing `.start`/`.end` (7 loose
cells + 3 validation tests, which never reach the op), `torch.max()` on a zero-element readback
in `eval/metrics.py` (the zero-volume cells, which fail *after* the op returned a correct empty
tensor), and `test_regression.py` calling `check_output` without `tolerance=` so it scores the
op's own default cell against `DEFAULT_TOLERANCES[float32]` rather than the feature spec's
declared `TOLERANCE_OVERRIDES[(float32, False)]`. **Zero op-attributed failures, zero hangs.**
The op's own unit directory is **1067 passed / 7 skipped**.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `dataflow_kernel_lib::stage_per_channel_chunk` (and `read_sticks_for_tilize` beneath its RM-flat form) | capability | Owns `reserve → issue → noc_async_read_barrier() → cb_push_back` as one indivisible unit and exposes no parameter, overload or template knob that returns *after issue but before the barrier*. An issue/finish split — the only way to drain a per-channel operand's DRAM latency under a later stage — is inexpressible through it. | 4121 | 4118 | **not graduated** — variant (d), `perf_experiments/reader_boot_order/k_d/rms_norm_ttnn_reader.cpp`. Reported honestly: on the focus shape the pair is a **3 ns tie**; the raw split's whole value is 8676 → 8362 ns interleaved, so the gap is real but confined, and nothing was graduated on it |
| `compute_kernel_lib::reduce` (via `rms_norm_local::accumulate_reduce_block`) | ergonomics | Its scaler CB is a **caller-owned lifecycle**: `reduce_helpers_compute.hpp:37` states "the scaler CB must contain the scaling factor tile BEFORE calling reduce()", and the helper issues no `cb_wait_front` of its own. Nothing at the call site looks like a missing wait, so the requirement is satisfiable only by out-of-band reasoning about the *reader's* statement order — which is exactly what this round changed, and the failure mode is a silent non-deterministic wrong answer rather than a hang or an assert. | — | — | `rms_norm_ttnn_compute.cpp` — the `scaler_ready` one-shot front in `pass_a`. **No bypass was written**; the helper is still used. The gap is that the op must supply the ordering the helper documents but does not enforce, and the two ns are *equal by construction* — the cost here is author and maintenance cost, not device time |
| `compute_kernel_lib` — no helper covers a **one-hot column permutation** (the `matmul_block` / eltwise / bcast / reduce families all preserve or collapse the column axis, and `transpose_wh` is a different map) | capability | Pre-existing from D27 and unchanged by this round, but this round added a second symptom worth recording: because `matmul_tiles` carries **no CB lifecycle at all**, a bank operand has no wait, no pop and no policy — so a raw permutation's operand ordering is entirely the caller's to establish. Separately, `llk_math_transpose_dest` cannot be used to permute a value already in DEST: `_llk_math_transpose_dest_`'s `TTI_STALLWAIT` waits on `SRCA_VLD\|SRCB_VLD`, which `add_tiles` has just cleared, and `llk_unpack_set_srcb_dummy_valid()` covers SrcB only — there is no SrcA equivalent, so it **hangs**. That missing capability is what would make the root's back-permute free (223 ns) and flip the gather payload-shrink option into the lead. | — | 1368 (through a CB round trip) vs 1258 (dual-NoC, no permute) | `rms_norm_ttnn_compute.cpp` `member_pack` / `compute_recv_unpack`; negative recorded at the tail of `rms_norm_ttnn_writer.cpp` |
| `dataflow_kernel_lib::SenderPipe::send` (`mcast_pipe.inl`) | capability | `send_data_` hard-codes `max_page_size` at its default `NOC_MAX_BURST_SIZE + 1`, which forces `ncrisc_noc_fast_write_any_len`. A caller that knows its payload is a single packet (3072 B against Blackhole's 16 kB burst) has no way to ask for `noc_async_write_multicast_one_packet` — the parameter is not on `SenderPipe`'s API. | 5324 | 5380 | **not graduated** — the raw path **LOSES by ~56 ns** (`SenderPipe`'s ctor precomputes the rect/in-rect arithmetic the raw version redoes per round, and the one-packet path's `noc_cmd_buf_ready` spin costs more than the any-len loop saves). Recorded as a gap because it is one; not as a win, because it is not |
| `ckl::DestReuseBinary` (`chain.hpp:526`) and the metal `{mul,add}_reuse_dest_{init,tiles}` pair | capability | Takes a plain `InputSpec`, which carries **no `BroadcastDim`** — unlike `BinaryFpu`, whose B side takes a `BinaryFpuInputSpec` (`input(cb, BroadcastDim::Row)`). So `((x * stat<Col>) * gamma<Row>) [+ bias<Row>]` cannot be spelled as one chain. This is a *different* shape from the one Lamp L-RES-FUSE refuted (broadcast on the first element there, on the reuse element here) and the pack-cohort constraint does not bite. Missing piece is one template parameter, not a hardware path: `llk_unpack_A` is already templated on `BroadcastType` **and** `EltwiseBinaryReuseDestType`. | 8.5 µs/core (un-fused pair) | 13.5 µs/core (raw fused) | **not graduated** — `perf_experiments/passb_fusion/k_fuse/`. **Recommendation to the helper library: do NOT add the parameter on this evidence** — the capability it would expose is measurably slower than the two-chain form it would replace, at least for FPU binaries over 4-face tiles |

An LLK defect found on the way and worth an owner: on the **broadcast** dest-reuse path, at
`dst_index == MAX_TILES_IN_HALF_DEST - 1` the runner's per-face `TT_ZEROACC` does not clear
DEST face `(limit-1)*4+3`, so the accumulating ELWMUL MOP returns `y = n*(g+1)` there. Purely
additive signature; bit-identical at every smaller block. Not reachable from any shipped path
(no helper reaches that combination) and not reachable from this op, which caps its fused
block below the limit — recorded, not worked around.

### Issues encountered

1. **The only op-attributed correctness break of the round was one this round created and
   fixed in the same round: D35's deleted implicit ordering.** Written up under "a latent bug
   the round exposed" above. Worth restating as a rule, because it generalizes past this op:
   *a helper that documents a precondition but does not enforce it makes the caller's
   statement order load-bearing, and nothing at the call site says so.* Two of them
   (`reduce`'s scaler, `matmul_tiles`' operands) were satisfied here only as a side effect of
   where the reader's publish happened to sit. Both are now explicit, and both are rows in the
   helper-bypass table.
2. **`.ii` / `*.o.log` purging poisons a warm JIT cache — Refinement 4b's hygiene step needs
   amending.** R4b prescribes deleting the harvested preprocessed artifacts under
   `built/*/kernels/rms_norm_ttnn_*/` after a `RMS_STAGE_ZONES=1` session, and says "ELFs
   untouched, cache stays warm". In this environment (`TT_METAL_CACHE=<repo>/built` +
   `TT_METAL_CCACHE_KERNEL_SUPPORT=1` + `TT_METAL_JIT_SERVER_ENDPOINT`) that is **not safe
   mid-session**: it leaves build dirs the cache still considers reusable whose contents no
   longer match the tree, and the next narrow run gets a MIXED kernel set. Reproduced exactly —
   after such a purge the ROW_MAJOR BAND cell returned **all zeros** (pcc 0.000) and the HEIGHT
   shard silently reverted to its pre-round timing (4744 ns against the repaired 3437), while a
   run with a FRESH `TT_METAL_CACHE` on the identical tree was correct and fast (22612 ns /
   pcc 0.999985, and 3441 ns). **The correct repair is to delete the op's kernel build
   DIRECTORIES wholesale** (`find built -type d -name 'rms_norm_ttnn_*' -exec rm -rf {} +`),
   not just the harvest artifacts inside them; done, and re-verified green afterwards on the
   normal cache. Every number in this entry was taken BEFORE that purge, on a coherent cache,
   and the post-repair re-measurement reproduces them (focus 3716 ns, HEIGHT 3437, BAND 22587,
   BLOCK gbr 28877, G=32 tree 3354, interleaved decode 8369) with golden shard 2 green again.
3. **The golden suite exceeds a single foreground window.** It is ~20 minutes of device time,
   so it was run as 10 `pytest-split` shards (`--splits 10 --group k`), which
   `run_safe_pytest.sh` forwards to pytest unchanged. Counts are additive across shards and sum
   to the 23348 / 19 above.

### Round 2 entry state

Re-ranked, the worst `perf` cells are now `(1,1,8192,2304)` and `(1,1,8192,1024)`
INTERLEAVED prefill at 0.906 / 0.899 — the DRAM-saturated regime Refinement 3 measured at the
box's ~400 GB/s roofline, where `ttnn.clone` on the same tensors is no faster. The live
follow-ups, all measured this round: idea 6's STREAM variant (**+20.1% ablation ceiling**, the
largest single number in the op), idea 6's integration overhead, the dual-NoC gather as a
two-kernel shape, and idea 1's variant (d) for the interleaved prefill.

---

## Perf 2

**All six ideas measured; 3 graduated, 1 superseded, 2 regressions. The op is 1.069x on the
focus shape and 1.522x on the round's largest prize, with no regression on any supported
cell. The worst `perf`-group ratio moves 0.908 -> 0.871.**

Every number here is `DEVICE KERNEL DURATION [ns]` on **blackhole p150b, 11x10 = 110-core
compute grid**, at the reference 1350 MHz, under the case's own fixed precision contract.
No idea in this round touched `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode`,
`dst_full_sync_en` or a dtype; every graduated path is pcc-identical to six decimals.

### Focus shape

No `LOOSE_CASES` entry carries an `attention:` note, so the fallback applies: measure every
case in the `perf` group and divide by its own `achievable_ns`. Re-ranked at the start of
this round (`perf_target_ranking.py`'s rule, run as `probes/bench_perf1_rank.py`):

| rank | case | ns | ceiling | ratio |
|---|---|---:|---:|---:|
| **1** | **`(1,1,8192,2304)` INTERLEAVED, bf16 TILE, gamma (bf16 TILE weight), HiFi2, `fp32_dest_acc_en=False`** | 191,879 | 211,345 | **0.908** |
| 2 | `(1,1,8192,1024)` INTERLEAVED, same config | 85,966 | 96,744 | 0.889 |
| 3 | `(1,1,8192,7168)` INTERLEAVED `gamma_bias_residual`, `fp32_dest_acc_en=True` | 1,578,655 | 1,837,678 | 0.859 |
| 4 | `(1,1,7168,1024)` BLOCK `[896,128]` (8,8) gbr | 29,024 | 34,569 | 0.840 |

Perf 1's focus (`(1,1,32,7168)` WIDTH-sharded) has fallen from 0.997 to 0.692 and is now
6th. Every axis of case #05 is in `SUPPORTED`, so it was optimized at its full config and
never at a proxy.

### Measured breakdown — focus shape

The op's 41 `MaybeDeviceZoneScope` sites are permanent and opt-in (D34). This round added
the four cumulative-peel switches, in the same shipped-commented form as D-era
`RMS_ABLATE_RECONFIG`: `RMS_ABLATE_READ_X`, `RMS_ABLATE_PER_CHANNEL`, `RMS_ABLATE_WRITE`,
`RMS_ABLATE_COMPUTE`. Each keeps the loop, the reserve/push, the barrier and the trip
count and strips only the payload.

**Per-stage zones** (`RMS_STAGE_ZONES=1`, ns per core, wall 193,687):

| zone | RISC | ns/core | n/core |
|---|---|---:|---:|
| `writer_write` (occupancy) | BRISC | 103,159 | 2.3 |
| `compute_square` (occupancy) | TRISC | 83,373 | 2.3 |
| **`reader_read_gamma`** | NCRISC | **67,563** (max 158,451) | **1.0** |
| `reader_read_x` (occupancy) | NCRISC | 18,552 | 2.3 |
| `compute_scale` / `compute_gamma_mul` / `compute_reduce` / `compute_finalize` | TRISC | 6,348 / 5,614 / 2,740 / 2,100 | 2.3 |

`writer_write`, `compute_square` and `reader_read_x` wrap their own `cb_wait_front` /
`cb_reserve_back`, so those are **occupancy, not payload** — which is exactly why the
ranking below is taken from the ablation and not from the zones.

**Cumulative peel** (stages peeled together, never one at a time):

| configuration | ns | stage |
|---|---:|---|
| full op | 190,593 | |
| - per-channel NoC payload | 181,355 | per-channel = **9,238** |
| - compute payload | 179,144 | compute = **2,211** |
| - x read payload | 133,046 | x read = **46,098** |
| - write payload too (all stubbed) | 26,201 | write = **106,845**; scaffolding floor **26,201** |
| *write stubbed, read kept* (separate run) | 109,730 | x read ALONE = **83,529** |

So on the focus shape: read alone 452 GB/s, write alone 353 GB/s, the two together
494 GB/s, and the whole op stubbed at once is 26,201 ns (13.7%). Removing the operand
*entirely* (rather than just its NoC payload) costs 16,138 ns — 8.4% of the op.

**Two hypotheses tested and refuted before any idea was floated**, which is the reason the
round did not spend itself on the payload bytes:

* **Load imbalance is not the wall.** `Rt=256` over 110 cores is 36 cores x 3 tile-rows +
  74 x 2 — a 0.776 balance factor. Sweeping `Rt` at W=2304: balance 0.670 -> 373 GB/s,
  0.776 -> 392, 1.000 -> 392 (Rt=220) and 409 (Rt=330). **The wall tracks total BYTES, not
  `rowmax`.** Aggregate-DRAM bound, not imbalance bound.
* **The 1.35x "read/write overlap deficit" was an artefact of my own arithmetic, and idea 1
  retired it.** An unsynchronized duplex probe — the op's exact split, page ranges and
  transaction shape, with *no CB handshake at all between reader and writer* — puts the
  focus payload floor at **178,079 ns / 423 GB/s**, flat at 421-434 GB/s across
  `block in {8,24,72,144}` and `cores in {28,56,84,110}`. Reads and writes do **not** add.
  The op's DM runs **1.3% above a program with zero synchronization**. Idea 2's independent
  pure-copy instrument agrees: 166,086 ns / 452 GB/s best, 172,126 / 439 for the op's own
  transfer pattern, i.e. within 3.1% of the best pattern found.

**Ranked, roofline-gated:**

| rank | stage | ns | gate |
|---|---|---:|---|
| 1 | per-channel operand staging | 16,138 (8.4%) | **NOT gated** — 1.0 MB of real traffic at ~63 GB/s effective, ~7x off the op's payload rate |
| 2 | scaffolding / issue floor | 26,201 (13.7%) | partly irreducible (dispatch); the rest is per-transaction issue + CB handshake |
| 3 | payload DM bytes | 152,943 (80%) | **ROOFLINE-GATED at 494 GB/s** |
| 4 | read/write overlap | 0 | **ROOFLINE-GATED** — the deficit does not exist (above) |
| 5 | compute payload | 2,211 (1.2%) | gated / irrelevant |

### The portfolio, and every verdict

Six ideas, deliberately overlapping: three at the payload DM, two at the per-channel read
(transport vs granularity), one at the STREAM regime.

| # | idea | verdict | measured |
|---|---|---|---|
| 1 | `rw_overlap` — close the read/write overlap deficit | **NULL on focus, side WIN** | Retired the target (above). CB depth 3/4, sub-tile-row writer drain, coarse/fine barriers, reader run-ahead: all 0.969-1.011x on focus, inside a +/-1.4% noise band. Found that `(1,1,8192,1024)` runs **one row-block per core** — graduated as **D41**. |
| 2 | `dual_noc_write` — split the write across both DM RISCs / NoCs | **REGRESSION** | 1.031x in a pure-DM bench, **0.964x** in the real op (focus) and 0.933x on `(1,1,32,1024)`. With compute in the loop the writer waits on compute, NoC1 stops being the pole, and every byte moved to NoC0 is 1.75x more expensive plus an extra barrier. Not graduated. |
| 3 | `bank_coalesced_txn` — bank-contiguous multi-page transfers | **REGRESSION** | `NB=8`; 72 txn/tile-row -> 8 of 18 kB. Bitwise exact at every run length, and **0.959x** at 2 pages/txn, **0.867x** at 9, monotone. Not graduated. |
| 4 | `per_channel_mcast_v2` — broadcast the per-channel operand | **WIN — graduated as D40** | focus **192,455 -> 179,342 ns, 1.073x** |
| 5 | `per_channel_boot_overlap` — hide the boot storm / re-sweep the read granularity | **WIN in isolation — SUPERSEDED** | `TRIM=3` (one 576-B tile-prefix instead of two 64-B face-rows) is 186,604 -> 181,343 ns, 1.029x. Not graduated; see below. |
| 6 | `stream_regime` — cut STREAM's 5 tensor-crossings to 3 | **WIN — graduated as D39** | case #15 **1,578,655 -> 1,039,664 ns, 1.517x** |

**Three findings from the nulls that are worth more than most wins:**

* Idea 3 falsified the premise it was given. Halving the transaction count buys **exactly
  zero** in isolation (read half 91,755 -> 91,826 ns; write half 122,750 -> 123,172) — per-
  transaction issue cost is entirely hidden behind DRAM bandwidth. The 6% loss appears
  **only when read and write run concurrently**: a multi-page burst holds one DRAM bank
  longer, so the opposite-direction traffic to that bank can no longer interleave with it.
  The op's page-per-transaction round-robin is what keeps the two directions overlapped.
* Idea 2 established that **reader-NoC0 / writer-NoC1 is already the optimum of the four
  pairings** — the reverse is 0.736x — and that swapping which *RISC-V* does what while
  holding the NoCs fixed changes nothing (0.01% apart). The write's 1.28-1.35x premium over
  the read is the NoC direction, not the RISC-V. It also recorded a hardware fact worth
  keeping: **two DM RISC-Vs on the same NoC under `DM_DEDICATED_NOC` deadlock** (each
  tracks its issued count locally against a NoC-global counter), so `DM_DYNAMIC_NOC` is
  mandatory for that shape — and it is free (1.004x) on this op.
* Idea 4 found the cause of **Perf 1's unattributed 4-5% same-build overhead**, the thing
  that blocked its ancestor from graduating. It was never the idea:
  `ttnn/ttnn/operations/__init__.py` `walk_packages()`-executes every package under
  `operations/`, two `perf_experiments/` dirs set `RMS_STAGE_ZONES=1` at module scope, and
  the shipped descriptor is imported *earlier* in that walk than a forked one — so the
  baseline compiled clean kernels and the candidate compiled zone-instrumented ones. The
  `__init__.py` files are deleted and `perf_experiments/README.md` records why.

**Why idea 5 was not graduated, measured rather than assumed.** Its own decomposition on
the focus shape is: no operand 176,000 ns; operand loop/reserve/barrier/push with the NoC
payload stubbed **179,300**; `TRIM=2` (shipped) 189,100; `TRIM=3` 183,300. So 179,300 is
the floor granularity cannot go below — it is handshake, not bytes. **D40 measures 179,342
there**, i.e. at that floor, because the deferred wait hides the loop and the reserve too.
On the axes where `TRIM=3` also won (`gamma+bias` 1.060x, STREAM 1.027x) D40 and D39 reach
1.097x and 1.522x. And it carries two exceptions D40 does not: a measured **0.981x** on an
fp32 TILE operand (its 1152-byte prefix outruns the halved request count) and a reproducible
**0.990x** on `(1,1,8192,1024)`. A winning fusion supersedes its components; this is that
case, and the component is recorded here rather than shipped.

### What graduated, and how widely

Three changes. Each is the op's **one unqualified path** for every plan it is correct on,
and each replaced code is deleted — the compact hold replaced the tiled hold's monopoly on
the ROW_RESIDENT price, the broadcast replaced the per-core read on engaged lines, and D41
replaced the coarsest-block rule outright.

| # | change | domain | carve-outs, and what earned each |
|---|---|---|---|
| **D39** | the **compact per-channel hold**: cache the two face-rows D23 already fetches (1/16 of the tiled bytes), re-materialize each chunk with a local L1 copy, and let `_row_resident_chunk` price its hold at that. STREAM shapes lift into ROW_RESIDENT; 5 tensor-crossings become 3 | every interleaved chunked plan | (a) **tried as a FALLBACK after the tiled hold**, so a shape that already fits tiled keeps its byte-identical program — taking compact there measured **0.84-0.94x**; (b) **grid-occupancy gate** — ROW_RESIDENT trades DRAM bytes for reader/compute overlap and that only pays when DRAM is the constraint: holding W, dtype, residual, rows-per-core *and* the solved chunk fixed and moving only the active-core count gives **32/110 0.66x, 64/110 0.97x, 96/110 1.09x, 110/110 1.11x**, monotone; (c) **feasibility** — block-float has no face-row form (272-byte face, not 64-byte aligned) and a ROW_MAJOR operand already arrives compact |
| **D40** | the **per-channel broadcast** — one injector per grid ROW reads the block and multicasts it. Closes the Blocking Model's deferred `GAMMA_MCAST` row | every plan where a grid row is a real reuse group, **including ROW_MAJOR activations** | (a) **feasibility** — a ROW_MAJOR *operand* is tilized by compute from a stick ring, so there is no reader-side block to send; (b) **feasibility** — D39's compact hold stages by local L1 copy, so there is no DRAM read to broadcast (pinned by a `static_assert` in the reader and `not pc_compact` on the host); (c) **`PC_MCAST_MIN_GROUP = 3`**, earned by a MEASURED **0.932x** on a 2-core line — one receiver cannot pay a fixed handshake. Written as the narrow exception: a 3-core line is untested and therefore **included** |
| **D41** | RESIDENT picks its block on **row-blocks per core**, and may buy a deeper ring to get one more | every RESIDENT plan | (a) **tie-break is the SHALLOWEST**, so where a deeper ring cannot raise the block count it is refused and the program is byte-identical — that is every one-tile-row-per-core plan and the BLOCK shard; (b) the deeper candidate is offered **only to the RESIDENT search**, earned by a measured regression: in ROW_RESIDENT / STREAM a deeper ring can only be bought with a finer width chunk, and that is **0.980x** on `(1,1,8192,5120)` gbr and **0.984x** on `(1,1,1024,16384)` gbr |

**Not fenced off where it was merely untested.** D40 now covers ROW_MAJOR activations,
which no measurement had reached when it was authored — the guard that excluded them was a
**bug**, not a decision: it read `IS_TILE` (the *activation's* layout, CT arg 0) where it
meant `PER_CHANNEL_IS_RM` (CT arg 7). The two disagreed exactly where they must not, and it
cost all 48 of the golden suite's `1x1x64x128 layout=ROW_MAJOR gamma_layout=TILE` cells a
**jit_build failure**. Fixing the flag both restored them and widened the domain: the large
ROW_MAJOR plan `(1,1,8192,1024)` gains **1.052x**. This is the round's only op-attributed
break, and it was created and closed inside it.

### Whole-op result — all 19 `perf` cases

Median of 3, in-process profiler, same harness and method as Perf 1's table.
`ratio` = measured / clock-scaled ceiling.

| # | case | before | after | x | ratio before -> after |
|---|---|---:|---:|---:|---|
| 15 | `(1,1,8192,7168)` INT gbr `fp32_dest=True` | 1,578,655 | **1,037,141** | **1.522** | 0.859 -> 0.564 |
| 5 | **`(1,1,8192,2304)` INT gamma — FOCUS** | **191,879** | **179,515** | **1.069** | **0.908 -> 0.849** |
| 14 | `(1,1,8192,5120)` INT gbr `fp32_dest=True` | 647,863 | 613,762 | 1.056 | 0.493 -> 0.467 |
| 7 | `(1,1,8192,7168)` INT gamma | 584,534 | 562,696 | 1.039 | 0.566 -> 0.545 |
| 10 | `(1,1,32,5120)` W `[32,160]` (8,4) 32c | 3,539 | 3,459 | 1.023 | 0.672 -> 0.657 |
| 4 | `(1,1,8192,1024)` INT gamma | 85,966 | 84,275 | 1.020 | 0.889 -> 0.871 |
| 9 | `(1,1,32,2304)` W `[32,256]` (9,1) 9c | 2,994 | 2,947 | 1.016 | 0.648 -> 0.638 |
| 6 | `(1,1,8192,5120)` INT gamma | 412,498 | 408,983 | 1.009 | 0.559 -> 0.554 |
| 8 | `(1,1,32,1024)` W `[32,128]` (8,1) 8c | 2,668 | 2,647 | 1.008 | 0.649 -> 0.644 |
| 16 | `(1,1,32,5120)` W (8,4) gbr `fp32_dest=True` | 4,204 | 4,189 | 1.004 | 0.641 -> 0.639 |
| 12 | `(1,1,8192,1024)` BLOCK `[1024,128]` (8,8) | 20,458 | 20,430 | 1.001 | 0.715 -> 0.714 |
| 17 | `(1,1,7168,1024)` BLOCK `[896,128]` (8,8) gbr | 29,024 | 29,044 | 0.999 | 0.840 -> 0.840 |
| 11 | `(1,1,32,7168)` W `[32,256]` (7,4) 28c | 3,794 | 3,805 | 0.997 | 0.692 -> 0.694 |
| 2 | `(1,1,32,5120)` INT gamma | 7,047 | 7,084 | 0.995 | 0.093 -> 0.093 |
| 0 | `(1,1,32,1024)` INT gamma | 4,329 | 4,361 | 0.993 | 0.473 -> 0.477 |
| 18 | `(1,1,128,4096)` INT, ROW_MAJOR weight | 11,327 | 11,416 | 0.992 | 0.173 -> 0.175 |
| 1 | `(1,1,32,2304)` INT gamma | 5,153 | 5,192 | 0.992 | 0.303 -> 0.305 |
| 13 | `(1,1,32,5120)` INT gbr `fp32_dest=True` | 9,614 | 9,697 | 0.991 | 0.061 -> 0.062 |
| 3 | `(1,1,32,7168)` INT gamma (>=7x cell) | 8,378 | 8,460 | 0.990 | 0.563 -> 0.568 |

**Every case is under its ceiling and the worst ratio moves 0.908 -> 0.871.**

The six cells at 0.990-0.999 are the small interleaved-decode shapes, and they are not
regressions. Their programs are unchanged except for the extra compile-time args, and a
dedicated 7-read A/B against a true pre-round build put the same cells at
`(1,1,32,1024)` 1.003, `(1,1,32,5120)` 0.994, `(1,1,32,7168)` 1.000 and `(1,1,128,4096)`
1.030 — inside the 2.3-6.1% spread those sub-12-us cells carry. Reported as measured rather
than smoothed.

### Guard set — one representative per distinct kernel path x layout x placement

Both columns measured in the **same session** against a true pre-round-2 build of the op
(`git checkout HEAD~4 -- <op files>`), 3 reads each, min reported.

| path | before | after | x |
|---|---:|---:|---:|
| HEIGHT shard, native, no combine — `(1,1,2048,256)` 64c | 3,560 | 2,236 | **1.592** |
| ROW_RESIDENT + D39 compact — `(1,1,8192,7168)` gbr `fp32_dest=True` | 1,584,119 | 1,052,683 | **1.505** |
| interleaved RESIDENT, D40 engaged — `(1,1,8192,2304)` | 190,810 | 178,231 | **1.071** |
| ROW_MAJOR activation, TILE weight — `(1,1,8192,1024)` | 89,523 | 85,127 | **1.052** |
| interleaved RESIDENT, D41 engaged — `(1,1,8192,1024)` | 86,436 | 83,425 | **1.036** |
| ROW_RESIDENT tiled, no compact — `(1,1,8192,5120)` gamma | 418,386 | 407,116 | 1.028 |
| interleaved, TILE weight, `fp32_dest=True` — `(1,1,128,4096)` | 11,405 | 11,074 | 1.030 |
| STREAM, gated out of D39 and D40 — `(1,1,1024,16384)` gbr | 500,713 | 493,809 | 1.014 |
| WIDTH shard, native + flat combine — `(1,1,32,7168)` | 3,774 | 3,747 | 1.007 |
| interleaved, ROW_MAJOR weight — `(1,1,128,4096)` `fp32_dest=True` | 10,996 | 10,964 | 1.003 |
| BLOCK shard, native + slot tree — `(1,1,8192,1024)` | 20,420 | 20,375 | 1.002 |
| ROW_MAJOR BAND, WIDTH-sharded — `(1,1,256,512)` 64c | 22,283 | 22,246 | 1.002 |
| ragged-`Wt` interleaved — `(1,1,32,4064)` | 32,236 | 32,210 | 1.001 |
| ROW_MAJOR activation, TILE weight, 2-core line — `(1,1,64,128)` | 5,164 | 5,211 | 0.991 (carve-out; 0.932 without it) |
| W non-aligned (masked reduce) — `(1,1,224,1000)` | 6,595 | 6,487 | 1.017 |
| H non-aligned — `(1,1,333,544)` | 7,798 | 7,922 | 0.984 |

**No material regression anywhere.** `(1,1,333,544)` is the one cell that read below parity
in two sessions (0.984 on 7 reads, 0.969 on 3), a 124-250 ns delta on a 7.8 us kernel and
inside that cell's own 2.7-6.6% spread. It is **not** carved out, and deliberately so: two
structurally identical cells — same regime, same "few cores each owning the full width",
same engaged broadcast — gain instead (`(1,1,224,1000)` 1.017x, `(1,1,128,4096)` 1.030x),
so the effect is not systematic and a predicate for it would fence off exactly the shapes
that happened to be benchmarked. Recorded, not guarded.

### Golden

`scripts/run_safe_pytest.sh --run-all eval/golden_tests/rms_norm_ttnn/`, 10 `pytest-split`
shards (2648 + 2352 + 2120 + 1840 + 2376 + 2532 + 2571 + 2000 + 2571 + 2338):

> **PASSED = 23,348, FAILED = 19, ERRORS = 0, HANGS = 0.**

Byte-identical to the Phase-0 / Refinement-1..4b / Perf-1 figure in **count and identity** —
the same three harness defects Phase 0 reproduced outside the op (`CoreRange.end_coord` on
a build exposing `.start`/`.end`; `torch.max()` on a zero-element readback in
`eval/metrics.py`; `test_regression` calling `check_output` without `tolerance=`).
**Zero op-attributed failures, zero hangs.** The op's own unit directory: 90 + 226 + 292 +
146 + 168 + 32 + 6 passed.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `dataflow_kernel_lib::read_sticks_for_tilize` | capability | It maps a **page to a tile ROW** and covers no sub-page tiled read at all, so there is no helper form for "fetch just the two 64-byte face-rows of tile `j` that carry row 0". D39's compact FILL is byte-for-byte the reads `stage_per_channel_chunk`'s existing `TRIM == 2` branch already issues — it only needs them to land compacted — and there is still no helper that can express either. | — (inexpressible) | fill is 57 kB/core in place of a 917 kB tiled hold; the regime change it buys is 1,578,655 -> 1,039,664 ns | `rms_norm_ttnn_reader.cpp`, `pc_compact_fill` (the D39 "COMPACT PER-CHANNEL CACHE" block) |
| `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | ergonomics | It carries `zero_tile` / `prepare_zero_tile` and nothing that writes an arbitrary VECTOR into row 0 of a tile-row, so D39's compact EXPAND is a hand-rolled L1 word copy scattering `2 * TILE_DIM * elem` bytes into face-row slots `0` and `tile/4` of each ring page. The mechanism is ordinary L1 addressing; what is missing is a named "vector -> row 0 of a tile-row" primitive. `noc_async_read`-to-self was rejected because it puts a NoC transaction plus a barrier on a 64-byte move the RISC-V does in ~20 cycles. **The two ns are equal by construction** — the cost being claimed is author and maintenance cost, not device time. | — | — | `rms_norm_ttnn_reader.cpp`, `pc_compact_expand` |
| `dataflow_kernel_lib::stage_per_channel_chunk` (and `read_sticks_for_tilize` beneath its RM-flat form) | capability | **Re-confirmed from Perf 1, on this round's shape, and still unfixed.** Both own `reserve -> issue -> noc_async_read_barrier() -> cb_push_back` as one indivisible unit, with no parameter, overload or template knob that returns after the ISSUE and before the BARRIER — so an issue/finish split, the only way to drain a per-channel operand's DRAM latency under a later stage, is inexpressible through them. | 186,604 | 184,251 | **not graduated** — `perf_experiments/per_channel_boot_overlap/k_split_late/`. Reported honestly: worth 1.013x here, and its best PLACEMENT flips sign by shape (boot-issue is 1.011x on `(1,1,8192,1024)` and 0.990x on the focus shape; late-issue is the reverse), so there is no unified split path to graduate |
| `noc_async_write_tile` / every write helper in `kernel_lib` | capability | Each binds the transaction to the kernel's own `noc_index`; "issue this page on the **other** NoC" cannot be expressed through them. Needed to test whether the write's 1.28-1.35x premium over the read is the NoC or the RISC-V. | — | — | **not graduated** — `perf_experiments/dual_noc_write/k_alt/`. The idea LOST in the real op (0.964x), so the gap costs nothing here; recorded because it is one, and because the experiment that needed it settled the roofline question for the whole round |
| `TensorAccessor` / `noc_async_read_tile` / `noc_async_write_tile` | capability | All three are **page-at-a-time by construction**. There is no accessor API that answers "how many of the next pages of this tensor are contiguous in one bank, and where does that run start" — `pages()` / `PagesAddressIteratorInterleaved` iterates addresses, it does not expose the contiguous-run length. Idea 3 had to re-derive the run from `NUM_DRAM_BANKS` and take the first page's `get_noc_addr`. | 173,344 | 180,835 (2 pages/txn) … 199,899 (9) | **not graduated** — `perf_experiments/bank_coalesced_txn/bank_bench.py`. The raw path **LOSES**, monotonically in transaction size. Recorded as a gap, and with the explicit recommendation **not to close it on this evidence** |
| `ttnn.Mcast1D` | ergonomics | It engages or refuses a **whole grid**, but the reuse structure here is **per line**: on the STREAM geometry 9 of 10 grid rows are valid reuse groups and one is not (it mixes 2-block and 3-block cores). Making that work meant building the helper over the full grid and then overriding the role to an out-of-band `OPT_OUT` sentinel for the lines that do not qualify. A `Mcast1D` that accepted a per-line predicate — or merely exposed `line_index_` / `line_members` — would remove the sentinel. The alternative the helper does support, refusing the whole grid, costs the other nine lines their broadcast: measured, `col` geometry engages **0 of 11** lines on that shape and gets 0%, against `row`'s 1.089x. | — | — | `rms_norm_ttnn_program_descriptor.py`, the `pc_role` / `PC_OPT_OUT` block in the D40 regime row. **No LLK bypass was written** — the helper is still used; the gap is that the op must carry a sentinel the helper's model has no name for |

`SenderPipe`'s single-packet capability gap, logged in Perf 1, was **not** re-litigated: D40's
winning geometry broadcasts a whole `WT_CHUNK` block, so `send_data_`'s hard-coded
`max_page_size` never bites.

### Issues encountered

1. **A guard that reads the wrong compile-time arg is a domain bug, not a safety margin.**
   D40's `static_assert(IS_TILE != 0, "TILE per-channel operands only")` named the right
   rule and read the ACTIVATION's layout flag. Because the host predicate was right and the
   kernel's was wrong, the two disagreed on exactly the plans neither had measured, and the
   result was a **build failure** on 48 golden cells rather than a wrong answer. Two lessons
   worth carrying: a kernel-side guard that duplicates a host predicate should be spelled
   from the same quantity; and the golden suite is what caught it, not the guard set — no
   perf case has a ROW_MAJOR activation.
2. **An isolated bench cannot see a trap that another graduation opens.** D41's
   `CB_DEPTH_CANDIDATES = (3, 2)` was a clean, byte-identical-elsewhere win in its own fork.
   On the integrated tree D39 had freed the L1 that pays for depth 3, so two ROW_RESIDENT
   shapes could suddenly reach it — and bought it with a finer width chunk, at 0.980x and
   0.984x. The fix was to state what a deeper ring is actually worth (overlap it buys) and
   refuse it where it buys none. **Every graduation after the first must be re-measured on
   the integrated tree, not trusted from its own dir.**
3. **The bash tool caps at a 10-minute timeout**, which the 10-shard golden run exceeds. Run
   the shards one or two per invocation with an explicit `timeout 285`; a shorter budget
   kills them inside the precompile warm pass and returns no summary at all.
4. Perf 1's JIT-cache hazard (never selectively delete `.ii` / `*.o.log` under
   `built/*/kernels/`; purge whole kernel build DIRECTORIES or nothing) was observed
   throughout and did not recur.

### Round 3 entry state

Re-ranked, the worst `perf` cells are `(1,1,8192,1024)` INTERLEAVED at **0.871** and
`(1,1,8192,2304)` at **0.849** — the same DRAM-saturated family, now with both of this
round's per-channel wins already in. What is left there is small and well-bounded: the
focus shape sits **1.9% above its own no-operand ablation** (176,013 ns) and **0.8% above
the unsynchronized duplex floor** (178,079 ns). Third is `(1,1,7168,1024)` BLOCK `gbr` at
**0.840**, which neither D39 nor D40 touches — its reuse runs along grid COLUMNS, and column
geometry is this round's measured loser under the deferred wait. Live follow-ups, all
measured this round: broadcasting D39's compact-cache **fill** (left on the table by
construction, since D39 and D40 are mutually exclusive today); a column-geometry broadcast
that survives the deferred wait, which is what would reach the BLOCK shards; and the
`Mcast1D` per-line predicate in the bypass table above.

---

## Perf 3

Third and last perf round. Focus shape **re-ranked, and it moved**: with no `attention:`
note anywhere in `LOOSE_CASES`, step 1 falls through to *measure every `perf` case and
divide by its own `achievable_ns`*, which puts **case #4 `(1,1,8192,1024)` INTERLEAVED at
0.866** ahead of Perf 2's focus (#5 `(1,1,8192,2304)`, 0.849) and #17 (BLOCK `gbr`, 0.838).
Optimized at its full declared config — `bfloat16`, `TILE`, gamma-only with a **TILE**
weight, `math_fidelity = HiFi2`, `fp32_dest_acc_en = False`, `math_approx_mode = False`,
eps 1e-12, soft `pcc_threshold = 0.9995` — never a stand-in.

Solved plan: `scheme=rows cores=110 wt_per_core=32 BLOCK_ROWS=2 WT_CHUNK=32
NUM_W_CHUNKS=1 X_RESIDENT=1 depth=(3,3) rows_max=3`, D40's row broadcast engaged with 10
injectors on 10/10 lines.

### The measured breakdown

**Cumulative peel** (each stage's payload stubbed with its CB reserve/push/wait/pop, loop
trip counts and zone intact; stages peeled **together, never one at a time**):

| configuration | ns | stage |
|---|---:|---|
| full op | 84,586 | |
| − per-channel payload | 85,452 | per-channel = **~0** |
| − compute FPU payload too | 85,319 | compute payload = **~0** |
| − x read payload too | 63,513 | x read, marginal = **21,806** |
| − write payload too (**everything** stubbed) | **14,225** | write = **49,288**; floor = **14,225** |
| *write stubbed, read kept* (separate run) | 53,448 | read ALONE = **39,223** |

Rates: read alone **428 GB/s**, write alone **340 GB/s**, the two together **473 GB/s**,
whole op **394 GB/s**. A linear fit over an `Rt` sweep at this width gives
`ns = 5,450 + 309.6·Rt`, i.e. a marginal aggregate rate of **423 GB/s** and a 5,450 ns
fixed program cost — and it reconciles with the peel to within 0.5%
(277 ns/row payload + 55.6 ns/row floor = 332.6 against the fit's 330.9).

**The verdict needed the whole op stubbed at once, and got it.** The 14,225 ns residue is
measured with *every* payload gone in ONE run, and it is **additive, not hidden**: payload
70,923 + floor 14,225 = 85,148 = the wall. Zones on that same all-stubbed run say what it
is — the floor is **TRISC-bound**, and by a wide margin:

| RISC | marker span, max/core |
|---|---:|
| NCRISC (reader) | 2,493 |
| TRISC_0/1/2 (unpack/math/pack) | 14,670 / 14,528 / 14,734 |
| BRISC (writer) | 15,078, of which `writer_write` is 14,293 ns of **pure WAIT** |

Per-stage inside the floor (max/core): `compute_square` **4,967**, `compute_scale`
**3,681**, `compute_gamma_mul` **3,257**, `compute_reduce` **2,479**, `compute_finalize`
**1,433**. All three TRISCs carry ~14,300 ns each — a *balanced* pipeline, which is the
fact that decided the portfolio (see "why no pack-deleting idea was floated").

**Three hypotheses tested and refuted before an idea was spent:**

* **Load imbalance is NOT the wall — re-confirmed in a regime where it looked obvious.**
  The zones show a 1.59x span imbalance (BRISC mean 52,489 vs max 83,276) and `Rt=256`
  over 110 cores is 36 cores × 3 tile-rows + 74 × 2, a 0.776 balance factor. But sweeping
  `Rt` at this exact width, so balance moves independently of bytes, is **flat**:
  bal 0.727 → **404.4 GB/s**, 0.776 → 399.4, 0.833 → 398.1, 1.000 → **401.9** (Rt=330) and
  390.4 (Rt=220). Perf 2 found this at W=2304 with `BLOCK_ROWS=1`; it holds at W=1024 with
  `BLOCK_ROWS=2`. The span imbalance is a *consequence* of shared DRAM, not a cause — cores
  with fewer rows finish early and the tail cores inherit their bandwidth.
* **The per-channel operand is now free here.** Peeling its payload moves nothing, and
  disabling D40's broadcast outright (`PC_MCAST_MODE=None`) measures **84,133 vs 84,586** —
  neutral. The entire operand stage is ≤0.5% on this shape, against 9,238 ns at W=2304 in
  Perf 2. D40 collected that prize; there was no second helping. (`PC_MCAST_LATE=False`
  costs 2.6%, so the deferred receiver wait is still earning.)
* **Reconfig is 42 ns** of the floor (`RMS_ABLATE=RECONFIG`). Gated.

**Ranked, roofline-gated:**

| rank | stage | ns | gate |
|---|---|---:|---|
| 1 | the **additive TRISC floor** | **14,225 (16.8%)** | **NOT gated** — it adds to the payload instead of hiding behind it |
| 2 | payload DM | 70,923 (83%) | **ROOFLINE-GATED** — 473 GB/s against the 494 GB/s best ever measured on this op, 4.3% off |
| 3 | per-channel operand | ~0 | gated (D40) |
| 4 | load balance | 0 | **ROOFLINE-GATED** — refuted above |

So rank 1 *was* the tournament, and its own sub-ranking (square-pack round trip → scale →
gamma → reduce → finalize) is what the four ideas aimed at.

### Two instrument defects found and fixed before any number was trusted

1. **The `PER_CHANNEL` ablation switch had a hole D40 escaped through.** The broadcast
   injector reads via `pc_issue_slice_reads()`, which carried no `RMS_ABLATE_PER_CHANNEL`
   guard — so peeling the per-channel payload on a broadcast-engaged plan peeled *nothing*
   and the tell was that **pcc stayed at 0.999985**, i.e. the read was still running. Fixed
   in both read sites.
2. **The JIT kernel cache key does not hash the kernel source's CONTENT**, so ablating by
   uncommenting a `#define` at a kernel head is a **CACHE HIT on the previous binary**. An
   "unablated baseline" reproduced **twice at 56,090 ns with pcc=nan** against a true
   84,510 ns, because it was still the all-stubbed build. Every ablation switch is now a
   **host define** (`RMS_ABLATE=READ_X,WRITE,COMPUTE,PER_CHANNEL,…`, emitted by
   `_kernel_defines()`), which *is* part of the key — so no configuration can alias another
   and no cache purge is needed. Editing kernel source in place also re-rolls every zone's
   16-bit hash, which is a second reason not to.

Both are permanent; the peel is reproducible from `perf_experiments/r3_breakdown/`
(`peel.sh`, `knob.sh`, `zones_focus.py`, `rt_sweep.py`, `guard_set.py`, `ab.sh`).

### The portfolio, and every verdict

Four ideas, deliberately overlapping — two at the floor's *overlap* (A, C) by different
levers, two at its *size* (B, D) on different passes.

**Why no pack-deleting idea was floated.** The all-stubbed zones show the three TRISCs
balanced to within 1.5% of each other, and `op_design.md` already records the same for pass
B (unpack 8520 / math 8556 / pack 8110). A change that deletes only packs cannot move a
balanced pipeline — which is exactly why Perf 1's `passb_fusion` LOST (13.5 vs 8.5 µs/core:
`eltwise_binary_run_with_dest_reuse` restarts the MOP *per face* with a
`move_d2a_fixed_face` + `TT_ZEROACC`, ~71 ns/tile against the helper's one-MOP bcast mul at
~32). That idea was **not re-floated**, and idea D was briefed to find a route that is *not*
DEST reuse.

| # | idea | verdict | measured |
|---|---|---|---|
| A | `resident_block_count` — buy more row-blocks by taking a SMALLER block at a given depth (Lamp **L-OVERLAP**'s open half) | **WIN — graduated as D42** | focus **83,997 → 82,889 ns (1.013x, ~6.7σ over 17–20 reads)**; BLOCK shard 20,372 → 19,493 (1.045x) |
| B | `square_fold_ceiling` — raise D12's fold ceiling by folding in GROUPS, decoupling the pack saving from the 16-bit serial depth | **WIN, regime-scoped — graduated as D43**; honest **NULL on the focus shape** | 1.033x–1.061x on the TRISC-bound geometries; focus 1.005x |
| C | `block_stream_granularity` — stream a row-block at sub-block granularity, reader lever + writer twin | **WIN in isolation — SUPERSEDED by A** | focus 1.008–1.022x; reader half a **NULL** (0.999) and its finer forms a **REGRESSION** (0.996) |
| D | `passb_op_count` — cut pass B's math op count by any non-DEST-reuse route | **idea-as-briefed NULL — and the null retired the whole class**; a reorder fell out as a **WIN — graduated as D44** | reorder 1.010x–1.151x on all 7 `combine=True` plans, bit-exact on all 6 `combine=False` |

**Four findings from the nulls, each worth more than a percent:**

* **D's null retired an entire idea class with one ablation.** Replacing the gamma
  broadcast-mul with a bare `CopyTile` over the same tiles — traversal, unpack, pack and CB
  lifecycle all kept, **only the multiply deleted** — is `0.982 / 1.002 / 0.982 / 1.004` on
  perf cases 8 / 11 / 12 / 17, while deleting the whole **traversal** is
  `1.096 / 1.113 / 1.244 / 1.168`. **Pass B's second pass costs its traversal; the multiply
  is free.** No arithmetic argument is needed after that: fewer muls, pre-combined
  broadcasts, `mul_tiles_bcast` variants and matmul-by-diagonal cannot pay at any geometry.
  It also re-explains Perf 1's fusion loss from the other side, and it says where the real
  prize is (`ceil` = **1.244x** on the BLOCK shard) and what shape a helper would need to
  claim it.
* **C falsified two premises of its own brief.** `DM_TXN_ROWS_MAX = 1`, so the reader
  **already** pushes `cb_input_tiles` one tile-row at a time — the block-scale handover was
  entirely in *compute* (`X_IN_A`'s `WaitPolicy::Upfront`), not in the reader. And the
  tile-row is the *right* granularity: a half-row push is 0.999 and a quarter-row 0.996,
  because a finer push needs a finer consumer and the row's reduce needs the row's whole
  width. `noc_async_read_barrier()` also fences **all** outstanding reads on the RISC, so
  sub-group *k+1* cannot be in flight during *k*'s barrier.
* **A found that depth is irrelevant once the block is picked** (82,570–83,153 ns across
  depths 2/3/4/5/6/8, a 0.7% band), which is what let D41's second ladder be *deleted*
  rather than merely left alone — and hands back the L1 it was spending.
* **A established the round's measurement discipline, and it is load-bearing.** The **first
  variant measured in a device session reads 1.5–2.2% slow**, proven on cases whose program
  does not change at all (4,381 vs 4,281 ns for the *identical* program). At this op's
  effect sizes that bias *is* the signal, so every headline number below is either
  drift-cancelled (BEFORE/AFTER alternated, `ab.sh`) or a min over ≥2 independent sessions.

**Why C was superseded rather than shipped alongside A.** C gates on `BLOCK_ROWS > 1`; D42
sets `br = 1` on every plan whose x is read over the NoC. After D42 that gate is **false
everywhere C would have fired** — ROW_RESIDENT, STREAM and BAND are already
`BLOCK_ROWS == 1`, and native shards are excluded from both. Same mechanism, same
magnitude, but D42 is host-only, costs no kernel code and no L1, and *also* wins 1.073x on
the BLOCK shard where C is gated off. A winning lever supersedes its component; this is
that case, and the component is recorded here rather than shipped as a second path.

### What graduated, and how widely

Three changes. Each is the op's **one unqualified path** for every plan it is correct on,
and each replaced its predecessor rather than sitting beside it: D42 **deleted**
`CB_DEPTH_CANDIDATES_RESIDENT`, and D43 **deleted** the all-or-nothing fold predicate in
favour of one `_x_squared_wt()` definition read by the L1 solve, the CB table and the CT arg
alike.

| # | change | domain | carve-outs, and what earned each |
|---|---|---|---|
| **D42** | **the block is PICKED, not inherited from what fits.** D41 got the objective right (row-blocks per core are the only thing to pipeline over) but raised the count only by deepening the ring, and still took the *largest* block that fits at each depth — a smaller block always fits, so the search never offered itself the finest split | every RESIDENT plan whose x is read from DRAM **as tiles** | Three, and all three are the *same* principle — a finer block is only worth the per-block fixed cost it multiplies, so it pays exactly where a block boundary is what overlaps a DRAM read of tiles and nothing else. (a) **measured regression** — a **zero-copy resident shard** (`native_in`) has no read to overlap: `br 16 → 1` is **0.50x** on the `(1,1,8192,1024)` BLOCK shard and **0.42x** on `(1,1,7168,1024)` gbr. (b) **measured regression** — a **ROW_MAJOR activation** (`not is_tile`) is *tilized by compute* out of a stick ring, one `ckl::tilize<WT_CHUNK>(rows)` call per block, and that fixed cost dwarfs the overlap: `BLOCK_ROWS 8 → 1` is **0.711x** on `(1,1,256,512)` ROW_MAJOR WIDTH-sharded and **0.647x** on `(1,1,512,1024)`. (An RM plan is also pinned to depth 1, so it has no ring to pipeline the finer blocks against.) Both exceptions may still **level out** their blocks at the same block count — 20+12 → 16+16 is **1.074x** — and (c) **measured regression** limits that to an **exact** divisor, because an inexact rebalance (11 → 10 on `(1,1,7168,1024)`) is **0.993x**. Depth is *not* a carve-out: it is provably unable to raise the block count, so the second ladder is gone rather than guarded |
| **D43** | **the GROUPED square fold.** `DEST_ACC_SQUARE_MAX_WT` is a *precision* ceiling on the fold's serial 16-bit accumulation depth, and because the fold was all-or-nothing that ceiling was also a *perf* ceiling — every prefill profile paid `WT_CHUNK` packs plus `WT_CHUNK` unpacks per tile-row. `X_SQUARED_WT` may now be any **divisor** of `WT_CHUNK`, so depth and pack-saving are independent: a chunk of 32 folds in groups of 16 and deletes 15 of every 16 packs. Expressed purely by reshaping the chain's iteration grid — **no helper bypassed** | every plan with `PARTIAL_W == 0` | (a) **correctness (pre-existing, unchanged)** — `PARTIAL_W != 0` gets no fold at all: the fold folds the row's last width tile *including its pad lanes* before the reduce runs, so the reduce's partial scaler / 0-1 mask can no longer reach them; (b) **inexpressible** — a `WT_CHUNK` with no divisor in `[2, G]` gets no grouped fold, because a ragged last group needs a second iteration shape and one `eltwise_chain` call cannot carry two. Nothing in the sweep hit it (chunk 57 = 3×19 folds at depth 3). **Not a carve-out:** the roofline-gated prefill band, where the change is flat, keeps the unified path |
| **D44** | **pass B does gamma FIRST on a cross-core plan.** Pass B's first op is the one that needs the finalized stat, and on a `combine` plan that stat arrives by gather → root fold → multicast. The gamma mul depends on x and gamma only, so doing it first fills that wait with the traversal instead of idling through it. On the small-block combine plans it measures **as fast as deleting the traversal outright** (1.095 vs 1.096; 1.129 vs 1.113; 1.036 vs 1.034; 1.074 vs 1.075), which only latency-hiding explains | every `combine`-engaged plan carrying a gamma | (a) **infeasible** — `!HAS_G`: with no gamma there is no second mul to move; (b) **measured regression** — `!CROSS_CORE`: with the stat computed locally there is no arrival to hide behind, and the reorder cost a reproducible **0.983x and 0.989x** in two independent sessions on `(1,1,8192,2304)`. Every other `combine=False` case was flat, so the carve-out is the **regime**, not that one shape. Written as `if constexpr (!HAS_G || !CROSS_CORE) { legacy } else { new }` — the narrow exception, so `combine=False` plans stay **bit-exact** |

**Guard polarity was corrected on graduation.** The subagent returned D44 as
`if constexpr (HAS_G && CROSS_CORE) { new } else { legacy }` — an allow-list around what it
measured. It ships inverted, so each exception names the reason that earned it and shrinks
as understanding grows instead of having to be widened by hand.

**D42's ROW_MAJOR carve-out was found by a STRUCTURAL PIN, not by the perf harness — and
that is the round's most useful process finding.** `tests/.../test_rms_norm_ttnn_perf.py::`
`test_program_is_structurally_the_seeds` diffs the op's program against the seed's; after
D42 it reported *three dropped staging CBs* on the ROW_MAJOR WIDTH-sharded plan. Chasing
that flag showed `BLOCK_ROWS 8 → 1` there and a measured **0.711x / 0.647x**. The guard set
could not have caught it: the row it inherited from Perf 2 was a **pinned 64×1 grid that no
longer fits this part's live 11×10 compute grid**, so it had been substituted with a
HEIGHT-sharded shape that D42's rule does not reach. A guard set whose rows silently stop
being constructible is a guard set with holes, and a program-structure diff is what found
this one. Both ROW_MAJOR WIDTH-sharded cells are now in the guard set explicitly, built
with `auto_shard_config` so they cannot go stale the same way.

**That pin was itself stale, and the round left it in better shape than it found it.** It
asserted `mine's CB set == the seed's` — an *identity* the op has deliberately and
measurably outgrown across six rounds (D39's compact hold, D41/D42's block, D43's
`cb_x_squared` width), and it was already red on 3 cells before Perf 3. Two changes, both
measured rather than asserted:
* the CB assertion is now a **budget plus a buffer-set identity**: re-blocking may change a
  CB's *size* but may never cost MORE L1 than the seed's, and may never add or drop a
  buffer. Every divergence is in fact strictly **cheaper** — `(1,1,8192,1024)` INTERLEAVED
  1,009,664 → 276,480 bytes (**−733,184**), the BLOCK shard 1,103,872 → 878,592
  (−225,280) — so the identity form was pointing the wrong way.
* the three **blocking** CT-arg indices are masked **by name** with the decision that owns
  each (`BLOCK_ROWS` at reader 4 / writer 4 / compute 3, its per-core row count at reader
  20, `X_SQUARED_WT` at compute 14). What the test still asserts — that a configuration
  supplying **no operand pays nothing for operands** — is intact and unweakened.

Rescoped, the pin reads **2 failed / 70 passed against the pre-round op** (better than the
original form's 3/70) and **7 failed / 65 passed** after. The five it still flags are
earlier rounds' buffer-set and reader-arg drift plus one allocator fatal in the test's own
two-descriptor setup; they are recorded here rather than masked, because unlike the
blocking indices nobody has yet established what owns them.

**D44 carries a real, recorded cost — the round's one semantic price.** The reordered
intermediate is `x · gamma`, which is **un-normalized**, so it can saturate the intermediate
CB's dtype where the shipped intermediate (≈1) cannot. The boundary is exactly
`|x · gamma| > dtype_max`: measured, `x=1e10, gamma=1e29` gives **9.965e28** shipped against
**3.373e28** reordered. This **narrows the op's dynamic range on combine-engaged plans**.
It is graduated because nothing in the op's tested universe reaches that band (all 19 perf
cases and 31 device-reachable structural `LOOSE_CASES` match the shipped order to 6 decimal
places of pcc, and the op's own `sum(x²)` already saturates by `|x| ≈ 1.8e19`) — but it is a
*price*, not a free win, it is stated at the bypass site as well as here, and reverting it is
one predicate.

**Precision was never a lever.** `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode`,
`dst_full_sync_en` and every dtype are untouched. D43 came back as an option menu and the
**fastest option meeting the contract** was graduated, not the fastest option: against a
float64 reference on adversarial inputs (every summand identical and non-dyadic — the
textbook serial-sum worst case), `G = 16` is rel-RMS **0.00717** at chunk 72 against the
shipped path's **0.01849**, i.e. **2.6× more accurate than what ships**, while the
*unbounded* fold — the obvious way to raise the ceiling, and ~1.001x on the focus shape
anyway — is **2.5× worse (0.01862 vs 0.00737)** and was **refused on precision, not on
speed**. `G = 8`, which pins the depth at exactly today's vetted 8, is recorded as the
conservative alternative at a cost of 0.5–2 points of the win.

### Whole-op result — all 19 `perf` cases

`min` over **two independent BEFORE sessions and two AFTER sessions**, same harness and
method as Perf 1/2 (`probe_262`). `ratio` = measured / clock-scaled ceiling; lower is
better, `> 1.0` MISSES.

| # | case | before | after | x | ratio before → after |
|---|---|---:|---:|---:|---|
| 9 | `(1,1,32,2304)` W `[32,256]` (9,1) | 2,936 | **2,551** | **1.151** | 0.636 → 0.553 |
| 3 | `(1,1,32,7168)` INT gamma (≥7× cell) | 8,440 | **7,541** | **1.119** | 0.567 → 0.506 |
| 11 | `(1,1,32,7168)` W `[32,256]` (7,4) | 3,784 | **3,412** | **1.109** | 0.693 → 0.623 |
| 8 | `(1,1,32,1024)` W `[32,128]` (8,1) | 2,621 | **2,370** | **1.106** | 0.640 → 0.577 |
| 2 | `(1,1,32,5120)` INT gamma | 7,004 | **6,350** | **1.103** | 0.092 → 0.084 |
| 10 | `(1,1,32,5120)` W `[32,160]` (8,4) | 3,451 | **3,147** | **1.097** | 0.657 → 0.597 |
| 18 | `(1,1,128,4096)` INT, ROW_MAJOR weight | 11,176 | **10,217** | **1.094** | 0.172 → 0.156 |
| 12 | `(1,1,8192,1024)` BLOCK `[1024,128]` (8,8) | 20,362 | **19,053** | **1.069** | 0.711 → 0.666 |
| 13 | `(1,1,32,5120)` INT gbr `fp32_dest=True` | 9,612 | **8,991** | **1.069** | 0.062 → 0.057 |
| 16 | `(1,1,32,5120)` W (8,4) gbr `fp32_dest=True` | 4,156 | **3,919** | **1.060** | 0.634 → 0.598 |
| 1 | `(1,1,32,2304)` INT gamma | 5,136 | **4,857** | **1.057** | 0.304 → 0.286 |
| 0 | `(1,1,32,1024)` INT gamma | 4,333 | **4,123** | **1.051** | 0.474 → 0.451 |
| 17 | `(1,1,7168,1024)` BLOCK `[896,128]` (8,8) gbr | 28,984 | 28,679 | 1.011 | 0.838 → 0.830 |
| 4 | **`(1,1,8192,1024)` INT gamma — FOCUS** | **83,777** | **82,912** | **1.010** | **0.866 → 0.857** |
| 14 | `(1,1,8192,5120)` INT gbr `fp32_dest=True` | 611,992 | 608,898 | 1.005 | 0.466 → 0.463 |
| 6 | `(1,1,8192,5120)` INT gamma | 406,697 | 405,861 | 1.002 | 0.551 → 0.550 |
| 7 | `(1,1,8192,7168)` INT gamma | 560,072 | 559,427 | 1.001 | 0.544 → 0.542 |
| 15 | `(1,1,8192,7168)` INT gbr `fp32_dest=True` | 1,043,276 | 1,044,230 | 0.999 | 0.568 → 0.568 |
| 5 | `(1,1,8192,2304)` INT gamma | 179,391 | 180,536 | 0.994 | 0.849 → 0.854 |

**Every case is under its ceiling, 12 of 19 gain ≥ 1.05x, and the worst ratio moves
0.866 → 0.857.** Perf 1 → Perf 2 → Perf 3 on that worst cell: 0.908 → 0.871 → 0.857.

The focus shape's own headline, measured **drift-cancelled** (BEFORE/AFTER alternated over
**four** pairs, 9 reads each, `ab.sh`): per-pair **1.005x / 1.008x / 1.011x / 1.005x**,
pooled **82,594 → 81,939 ns = 1.008x**. The paired table above reads 1.010x, so the two
methods agree to within 0.2 points — and the spread across pairs is itself the honest
caveat, because it is the same size as the effect. That is a modest number and it is the
right one: the
focus shape's payload is roofline-gated at 473 of 494 GB/s, and D43's real 1.019x saving
there (visible the moment the write payload is stubbed) is *hidden* by that roofline. The
round's value landed on the twelve cells where TRISC, not DRAM, holds the wall.

**Case 5 is the one sub-parity read, and it is not a regression.** `(1,1,8192,2304)` is
`combine=False`, so D44 is carved out and its program differs from pre-round only in D43's
CT arg. The dedicated drift-cancelled A/B on that exact cell reads **178,401 / 179,126
BEFORE against 179,154 / 178,420 AFTER — flat (0.999)**, and D43's own bench put it at
0.998. The 0.994 above is single-session noise on a shape whose spread is ~1%. Reported as
measured rather than smoothed, and deliberately **not** carved out: a predicate there would
fence off the shapes that happened to land badly in one session.

### Guard set — one representative per distinct kernel path × layout × placement

Perf 2's 16-path set, so the rounds compare, **plus the two ROW_MAJOR WIDTH-sharded rows
this round had to add** (see below). `min` of 3–9 reads; BEFORE is the Perf-2 tip op files
(`git checkout 79ab065a7f -- kernels rms_norm_ttnn_program_descriptor.py`); the headline and
ROW_MAJOR cells were measured **drift-cancelled** with BEFORE and AFTER alternated (`ab.sh`).
One inherited row had to change shape: Perf 2's "ROW_MAJOR BAND, WIDTH-sharded `(1,1,256,512)`
64c" pins a 64×1 grid that **no longer fits this part's live 11×10 compute grid** and now
raises rather than measuring — which is exactly how D42's ROW_MAJOR regression stayed hidden
from the perf harness. It is replaced by a HEIGHT-sharded band plus two `auto_shard_config`
ROW_MAJOR WIDTH-sharded cells that cannot go stale the same way.

| path | before | after | x |
|---|---:|---:|---:|
| WIDTH shard, native + flat combine, D44 — `(1,1,32,7168)` | 3,684 | **3,330** | **1.106** |
| BLOCK shard, native + slot tree, D42 balance + D44 — `(1,1,8192,1024)` | 20,272 | **18,876** | **1.074** |
| interleaved, TILE weight, `fp32_dest=True` — `(1,1,128,4096)` | 11,589 | **10,707** | **1.082** |
| ROW_MAJOR BAND, HEIGHT-sharded — `(1,1,256,512)` | 8,627 | **8,187** | **1.054** |
| interleaved, ROW_MAJOR weight — `(1,1,128,4096)` `fp32_dest=True` | 10,670 | **10,248** | **1.041** |
| W non-aligned (masked reduce, fold gated OFF) — `(1,1,224,1000)` | 6,546 | 6,427 | 1.019 |
| interleaved RESIDENT, D42 engaged — **`(1,1,8192,1024)` FOCUS** | 82,310 | 81,912 | **1.005–1.011** (4 pairs) |
| ROW_RESIDENT + D39 compact — `(1,1,8192,7168)` gbr `fp32_dest=True` | 1,042,844 | 1,037,836 | 1.005 |
| STREAM — `(1,1,1024,16384)` gbr | 495,095 | 493,320 | 1.004 |
| ragged-`Wt` interleaved — `(1,1,32,4064)` | 32,307 | 32,224 | 1.003 |
| ROW_RESIDENT tiled, no compact — `(1,1,8192,5120)` gamma | 404,127 | 403,659 | 1.001 |
| ROW_MAJOR activation, TILE weight — `(1,1,8192,1024)` | 84,389 | 84,293 | 1.001 |
| **ROW_MAJOR WIDTH-sharded — `(1,1,256,512)`** (D42 carve-out) | 22,318 | 22,290 | 1.001 (**0.711 without it**) |
| **ROW_MAJOR WIDTH-sharded — `(1,1,512,1024)`** (D42 carve-out) | 39,027 | 39,019 | 1.000 (**0.647 without it**) |
| interleaved RESIDENT, `combine=False` — `(1,1,8192,2304)` | 178,401 | 179,154 | 0.999 |
| HEIGHT shard, native, no combine — `(1,1,2048,256)` 64c | 2,243 | 2,259 | 0.993 |
| ROW_MAJOR activation, 2-core line (D40 carve-out) — `(1,1,64,128)` | 5,267 | 5,309 | 0.992 |
| H non-aligned — `(1,1,333,544)` | 8,055 | 8,126 | 0.991 |

**No material regression anywhere — after one was found and carved out.** The two ROW_MAJOR
WIDTH-sharded rows are the round's one real regression: D42 reached them and cost
**0.711x / 0.647x** until the `not is_tile` exception was added, and they now read at
parity. Every other path is flat or better. The three cells at 0.991–0.993 are sub-9-µs kernels
whose own session spread is 2.7–6.6%; two of them (`(1,1,2048,256)` HEIGHT native and
`(1,1,64,128)`) build programs D42/D43/D44 cannot reach at all — `rows_max == 1` on a native
shard, and no combine — so their delta is measurement, not code. `(1,1,333,544)` read 0.984
in Perf 2 and 0.991 here on the same reasoning, and is again **not** carved out: two
structurally identical cells gain instead (`(1,1,224,1000)` 1.019x, `(1,1,128,4096)`
1.082x), so a predicate would fence off exactly the benchmark set rather than a real effect.

### Golden

`scripts/run_safe_pytest.sh --run-all eval/golden_tests/rms_norm_ttnn/`, 10 `pytest-split`
shards (2648 + 2352 + 2120 + 1840 + 2376 + 2532 + 2571 + 2000 + 2571 + 2338):

> **PASSED = 23,348, FAILED = 19, ERRORS = 0, HANGS = 0.**

**Byte-identical to the Phase-0 / Refinement-1..4b / Perf-1 / Perf-2 figure in count and in
identity** — the same three harness defects Phase 0 reproduced outside the op
(`CoreRange.end_coord` on a build exposing `.start`/`.end`, 10 cells; `torch.max()` on a
zero-element readback in `eval/metrics.py`, 3 cells; `test_regression` calling
`check_output` without `tolerance=`, 6 cells). **Zero op-attributed failures, zero hangs.**

Because D43 changes an accumulation, the 6 precision-shaped harness failures were re-run
against the pre-round tree and are **bit-identical** before and after (rms 0.016671 /
0.017828 / 0.018925 / 0.016491 / 0.017824 / 0.018904 in both) — those widths are
`WT_CHUNK ≤ 8`, where the fold already shipped and D43 is byte-identical by construction.
`test_rms_norm_ttnn_zone_hashes.py` green after every kernel edit (6 passed).

### Helper bypasses

**None.** All three graduations are helper-native: D42 is a host blocking rule, D43 is a
`ckl::eltwise_chain` **iteration-shape reshape** with the shipped
`ckl::output(..., DestAccumulation::PerRow)` spec, and D44 is a re-spelling of two existing
chains in the other order. No raw LLK was admitted this round and no kernel-head bypass
justification was needed.

Three helper **gaps** were nevertheless measured and are reported as feedback, in the same
schema, because a gap is worth recording whether or not it was worked around:

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `ckl::WaitPolicy::Cumulative` | capability | It emits `cb_wait_front(cb, i_flat + inner_count)` (`chain.inl:2643`) **without adding the operand's `TileBase`**, so it is only correct at base 0 — it cannot express an incremental wait on a **held** (non-popping) CB, which is every RESIDENT plan. Idea C had to express the same thing as a span loop of `Upfront` + `TileOffset::Set` chains, relying on `emit_wait_upfront` adding `tile_base` (`chain.inl:2784`). Adding the base to `Cumulative` would let pass A carry the split in one chain call. | 83,589 (Upfront, whole block) | 82,636 (span loop) | **not graduated** — superseded by D42; `perf_experiments/block_stream_granularity/` |
| `ckl::IterationShape::grid(H,W).block_size(blk)` | capability | A chain block is **within a row** — it cannot span the row axis. On the BLOCK-shard plans (`WT_CHUNK = 4`, `BLOCK_ROWS = 20/11`) that caps the DEST-lane block at **4 of the 8 available lanes**, and the curve is still climbing at the cap: case 12 at `blk 1 / 2 / 4` is `1.000 / 1.182 / 1.243` and case 17 `1.000 / 1.200 / 1.245`. Missing piece is a 2-D block (`block_h × block_w ≤ the DEST limit`) so a narrow-and-tall block can fill DEST. **This is the single largest lever any of the four ideas pointed at and did not reach.** | 20,413 @ blk 4 | not built | pass B's two chains |
| `ckl::DestReuseBinary` (`chain.hpp:526`) | capability | Perf 1 recorded the missing `BroadcastDim`; this round adds the other half and **sharpens the recommendation rather than repeating it**. The prize a fusion would claim is real and large — deleting pass B's second **traversal** is **1.244x** on case 12 and 1.168x on case 17 — but idea D's `nomul` ablation proves the cost is the traversal and **not the math**, so it is only claimable by a fusion that adds *no* math. Dest-reuse adds ~39 ns/tile (71 vs 32) by restarting the MOP per face. So: still **do not** add the broadcast parameter on this evidence; a **one-MOP, non-face-restarting DEST-as-srcA broadcast mul** would be the thing worth building. | 20,404 (un-fused) | 0.818x equivalent (Perf 1) | `perf_experiments/passb_fusion/k_fuse` |

`CopyTile` also carries **no `DestAccumulation` parameter**, which is why
`RMS_ABLATE=COMPUTE`'s payload-free stub for the square cannot compile on a *folding*
geometry (`chain.inl:2906` static_asserts accumulation on both the math element and the
output). That is a pre-existing hole D43 merely widens; the stub is now compile-time gated
on `!SQ_FOLD` and the limit is documented at the site rather than left to fail a build.

### Issues encountered

1. **An ablation switch that a later feature routes around is a silently broken
   instrument.** D40's broadcast reads through a *different* function than the one the
   `PER_CHANNEL` guard sat in, so the peel reported "per-channel costs 1,771 ns" when the
   read was still running — and the only tell was that **pcc did not move**. Check that an
   ablated stage actually breaks the answer before believing its number.
2. **The JIT cache does not hash kernel source content**, so ablating by editing a `.cpp`
   in place is a cache hit on the previous build. It cost two measurements (a "baseline"
   reproduced twice at 56,090 ns / pcc=nan against a true 84,510) and 377 GB of cache to
   purge. Every switch is a host **define** now. Corollary for anyone measuring here: drive
   a variant through a define, a CT arg, or a different file path — never an in-place edit.
3. **`git checkout --` as an experiment-restore reverts uncommitted work in the same file.**
   `knob.sh` used it and silently deleted this round's `RMS_ABLATE` plumbing mid-session; it
   restores from a byte copy now.
4. **The first variant in a device session reads 1.5–2.2% slow** on an *identical* program.
   Any A-then-B comparison without a burn read or ABBA alternation favours B by roughly this
   round's entire effect size. Both `ab.sh` and the two-session `min` in the tables above
   exist for that reason.
5. Perf 1's JIT-cache hygiene rule (never selectively delete `.ii` / `*.o.log` under
   `built/*/kernels/`; purge whole kernel build DIRECTORIES or nothing) held, and issue 2
   above is its sibling: the cache is content-blind, so it must be keyed or purged, never
   trimmed.
6. **`git checkout HEAD --` as an experiment-restore bit TWICE**, and the second time it
   cost a wrong conclusion, not just work: `ab.sh` reverted D42's ROW_MAJOR carve-out
   between the BEFORE and AFTER legs, so both columns measured the *same* code and the fix
   appeared not to work. The rule that came out of it: an A/B harness must **snapshot the
   working tree** and restore from that copy, and any graduation must be **committed before
   it is measured**. Both `ab.sh` and `knob.sh` do it that way now.
7. **A guard-set row that has silently stopped being constructible is a hole, not a row.**
   Perf 2's ROW_MAJOR WIDTH-sharded entry pins a 64×1 grid; on this part it now raises
   `shard grid 64x1 exceeds live compute grid 11x10`, and substituting a nearby shape moved
   it off the very code path D42 changed. The regression was caught by a **program-structure
   diff** instead. Worth carrying: a perf guard set and a structural pin fail in different
   directions, and this round needed both.

### Where the op stands after three rounds

The worst `perf` cell is `(1,1,8192,1024)` INTERLEAVED at **0.857** (0.908 → 0.871 → 0.857
across the three rounds), then `(1,1,8192,2304)` at 0.854 and the BLOCK `gbr`
`(1,1,7168,1024)` at 0.830. Across the three rounds the *shape* of the remaining gap has
changed: rounds 1 and 2 were spent on DRAM traffic and per-channel staging, and round 3
found both of those closed on this family and the residue sitting on the TRISCs. All three are now **payload-bound at 473 of the 494 GB/s best
this op has ever measured**, with an additive TRISC floor that this round cut from 16.8% by
the amount the roofline lets show. The measured, unclaimed levers, in the order the evidence
ranks them:

* **A 2-D chain block** (`block_h × block_w`), worth a measured **1.24x** on the BLOCK
  shards — the largest single number any experiment produced this round, and it is a helper
  change, not an op change.
* **A one-MOP DEST-as-srcA broadcast mul**, which would finally make pass B's two traversals
  one; the prize is the same 1.24x and the reason it is unclaimed is documented above.
* **`WaitPolicy::Cumulative` + `TileBase`**, which would let a held CB be consumed
  incrementally in one chain call instead of a span loop.

---

## Reference-suite iteration 1 — three defects the blind upstream suite found

- **Date**: 2026-09-08
- **Trigger**: `eval/golden_tests/rms_norm_ttnn/reference_suite/` (unmodified upstream
  `ttnn.rms_norm` tests, aliased onto this op) reported 93 of 416 failing.  All 93 are now
  fixed and the suite is **416/416**.  Three independent root causes, all host-side in
  `rms_norm_ttnn_program_descriptor.py`; no kernel source changed.

### 1. The shard spec's ORIENTATION was never read (76 cases)

`_plan_placement` and `_plan_band` mapped shard index → core row-major unconditionally, so
every `ShardOrientation::COL_MAJOR` input was handed to the wrong cores:

* a **BLOCK** shard has its two grid axes TRANSPOSED under COL_MAJOR — grid column `x`
  carries the row block and grid row `y` the width slice — so the width group is a grid
  COLUMN and the stat multicast is `Mcast1DShape::PerColumn`, not `PerRow`;
* a **HEIGHT / WIDTH** shard enumerates its cores y-fastest, so the linear shard index
  walks down a column.

Added `_shard_row_wise()` / `_shard_ordered_cores()` (the orientation-aware counterpart of
the deliberately row-wise `_cores_in`), plus a plan-level `group_axis` so the combine's slot
tree keys its parent lookup on the group's constant grid coordinate instead of always `y`.
Every ROW_MAJOR build is byte-identical.

### 2. Compute intermediates were held in the INPUT's block-float format (8 cases)

`cb_x_sum` (`t = x + r`), `cb_x_squared` and `cb_normalized` took `input_tensor.dtype`, so a
`bfloat8_b` input paid three extra block-float roundings that the target op never pays — its
intermediate CBs are `fp32_dest_acc_en ? Float32 : Float16_b` whatever the input dtype.  New
`_intermediate_dtype()`: the input dtype, except a block float, which becomes **bfloat16**.
Modelled on the failing cell (`probes/probe_421.py`), relative Frobenius error is 1.390e-2
with bf8b intermediates, 1.013e-2 with bf16 and 9.98e-3 with fp32 — i.e. bf16 is already at
the input/output-quantization floor, for half of fp32's L1.  `_cb_block_mult` became
`_cb_block_bytes` so the L1 solve prices tensor-format and intermediate-format CBs apart
instead of assuming one tile size.

**Measured perf** (10 representative bfloat8_b golden cells, Tracy device-kernel ns, fresh
cache, run-to-run spread < 0.2%): total 644 187 → 646 885 ns, **0.996x — flat**.  It is not
uniform: `128x8192` HEIGHT-sharded `gamma_bias` 0.878x and `residual` 0.865x (the bigger
intermediates cost the first a clean `WT_CHUNK` 32 for a ragged 29, and the second the
ROW_RESIDENT regime outright), against `gamma_bias_residual` at **1.153x**.  Every bfloat16
and float32 build is byte-identical (`it == bt` there, and the byte formula collapses to the
old tile formula).

### 3. The reduce MASK tile was written in a format the reduce does not read it in (6 cases)

`reduce_accumulate_via_add` (`kernel_lib/reduce_helpers_compute.inl`) programs BOTH unpack
operands from the reduce INPUT CB — "both add operands = the input CB" — and never points
SrcB at the scaler CB.  The partial (non-tile-aligned) fold's masked broadcast-mul therefore
unpacks the 0/1 mask in `cb_x_squared`'s format.  With a bf16 mask and an fp32 input that
halves the mask's lane pitch: the 1.0s land in tile columns `{0..3, 16..19}` instead of
`{0..PARTIAL_W-1}`, dropping 4 real lanes and admitting 4 padded ones.  Measured exactly
that way (`probes/probe_425.py`: all-ones input, pad poison swept, implied leak 3.90–3.94
lanes at every poison value).  Only visible at `Wt >= REDUCE_ACC_VIA_ADD_MIN_WT == 4` (below
it the ReduceTile path, whose reconfig DOES take the scaler's format, is chosen) and only at
float32.

`cb_scaler` now takes the INTERMEDIATE format whenever `PARTIAL_W != 0`, so the mask is
written in the format it will be read in.  **The underlying defect is in the shared helper**,
not in this op, and fixing it there (reconfig SrcB to `scaler_dfb_id` inside
`fold_partial_last`, restore after) would protect every caller.

### 4. `subblock_w` crashed instead of clamping on the plans that re-chunk the width

`program_config.subblock_w` is validated against `block_w` — the shard's width in tiles, the
only extent the caller can see — but a ragged shard over a non-tile-aligned W cannot keep the
shard as the per-core reduce dim (the last REAL width tile lands in the middle of the last
core's block, where no partial scaler reaches it), so the plan falls back to `SCHEME_ROWS`
and `WT_CHUNK` becomes the whole `Wt`.  `w=72` over 2 cores is `block_w=2` vs `WT_CHUNK=3`.
The descriptor asserted, turning a call `validate()` had already accepted into an internal
`AssertionError`.  It now takes the largest divisor of `WT_CHUNK` not exceeding the caller's
value — `subblock_w` bounds the pass-B DEST sub-block, so "no larger than this" is the honest
reading.

**Capability gap left open**: that geometry should keep the NATIVE shard plan and route the
partial mask to the last REAL tile of the core's block rather than to the block's last page.
That is a kernel change (a per-core partial-tile index), not a host one.

### Regression evidence

* Reference suite: **416 passed, 0 failed** (was 93 failed).
* Golden suite: all **23 325** previously-passing `test_golden.py` cells re-run and passing,
  in two disjoint halves (13 806 covering 100% of the bfloat8_b and 100% of the
  non-tile-aligned cells plus one of every distinct axis combination, then the 9 519
  complement).  `test_regression.py` + `test_validation.py`: 7 failed / 25 passed, **identical
  at `0e0d184c52` (pre-change) and at HEAD** — all seven pre-existing.
