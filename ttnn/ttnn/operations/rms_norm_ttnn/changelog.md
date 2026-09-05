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
