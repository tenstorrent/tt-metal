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
