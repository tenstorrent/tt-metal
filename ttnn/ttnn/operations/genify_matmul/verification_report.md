# Verification Report: genify_matmul

Phase 0 = regime R1 `interleaved_out_split` (op_design.md, Blocking Model): bf16 × bf16 → bf16, TILE,
fp32 DEST, interleaved DRAM/L1, epilogue none. Device: Blackhole p300c, 11×10 compute grid.

## Code Review

### Fixed in this pass

1. **Grid factorization chose pathological splits (perf-conformance bug).** `choose_grid()` ranked
   per-core output tiles W strictly before read bytes. R1 has no operand reuse across cores and
   is DRAM-bound on every measured shape: HiFi2 and HiFi4 device times are within ~10% of each other.
   For M=9472, K=5120, N=3456 the old rule took a flat 1×108 N-only split (W 296 vs 324 for a 10×9
   rectangle). That gave 1×1 subblocks, every core reading all of A, and 10.5 GB of DRAM reads
   instead of 1.3 GB. The same ordering re-read a 113 MB B 8× on the M=256 DiT shapes.
   - Replaced by `choose_split()`. It keeps the candidates within `(1 + GRID_W_SLACK)·W_min`
     (`GRID_W_SLACK = 0.25`, a named knob defined once), solves `choose_blocking()` for each, and
     takes the min of (aggregate A+B read bytes **including the block re-reads the L1 solve adds**,
     W, cores).
   - A blocking-unaware version was tried first and measured. It regressed 2048×9216×6144 by 1.2×,
     because the S3 shrink doubled the A re-reads, so the blocking is part of the score.
   - Host cost is ≤ 1 ms per call.
   - Measured (device ns, `tests/.../genify_matmul/perf/`, HiFi4 and HiFi2): 9472-5120-3456
     28.2 → 7.2 ms (0.25×); 256-6144-9216 2.44 → 1.12 ms (0.46×); 2048-128-3072 158 → 64 µs (0.40×);
     4096³ 3.37 → 3.03 ms (0.90×); 9472-3456-5120 6.92 → 6.32 ms (0.91×). Every other guard shape
     is within ±1% (noise).
   - Golden suite geomean device time is 0.95×. The worst regressions are +7%
     (`test_matmul_with_core_grid[7]`) and +4–5% (`test_padded_2d_matmul[*-width]`): small-K shapes
     whose chosen W sits exactly at the slack cap. These are a tuning point for the perf refinements.
2. **L1 CB budget ignored L1 fragmentation (correctness bug: TT_THROW).** `l1_cb_budget()` used
   `bank − allocated bytes`. L1 tensors are placed top-down, so a freed tensor above a live one
   leaves a hole, and the real CB room ends at the lowest allocated block. The deep-K walk grows
   `block_k` right up to the budget, so the CBs overlapped live L1 tensors and program.cpp threw
   "circular buffers … clash with L1 buffers".
   - I hit this in a probe (fp32 / bf8b output and L1 weight on 64×16384×512). It reproduces
     deterministically with the old formula; the new one passes.
   - Fixed: the budget is now `min(lowest allocated block offset from the L1 memory view
     block_table, bank − this op's L1 tensors per core) − reserve`.
   - Regression test: `test_genify_matmul_extended.py::test_fragmented_l1`.
3. **DRY:** `TILE = 32` was defined in both `genify_matmul.py` and the descriptor. The op file now
   imports it from the descriptor.
4. **Docs kept current:** `op_design.md` (Work-split rule, L1 budget) and `l1_ledger.md` (the
   `l1_cb_budget` symbol) now describe what the code does.

### Reviewed, no change needed
- **Kernel syntax and API usage.** All three kernels use `void kernel_main()`, the
  `api/dataflow/dataflow_api.h` / `api/compute/...` includes and `TensorAccessor` (no
  `InterleavedAddrGen`).
- **CB sync balance.**
  - Reader: one reserve/push per operand per K-block, always the nominal page count.
  - Compute: the helper waits/pops per K-block (`WaitAndPopPerKBlock`).
  - Output: the helper pushes `sb_h·block_n` per row group; the writer waits/pops the same.
  - `cb_partials`: helper-internal.
- **Helper usage.** Compute is a single `compute_kernel_lib::matmul_block` call per block, with
  packer L1 accumulation, the `LastKBlockInnerDim` ragged-K hook and the `cb_output_tiles`
  placeholder when `num_k_blocks == 1`. There is no raw compute API, no mcast yet, and no broadcast
  in Phase 0.
- **Blocking-model fidelity.**
  - Every knob is a host value handed to the kernels as one CT/RT arg: `block_*`, `num_k_blocks`,
    `last_block_k_tiles`, `sb_*`, the `DEPTH_*` constants and `BLOCK_K_FLOOR`. CB page counts and
    loop trip counts derive from those same variables.
  - No CB scales with a whole-op dimension: `tensor_*` never appears in a capacity.
  - The block defaults to the whole per-core region (S1/S2), so it is not half-turned.
- **Expression.**
  - Reader: one barrier per K-block covering both operands, one handshake per operand.
  - Writer: one `noc_async_writes_flushed` per row group, one final barrier. This is better than
    the design's barrier per row group.
  - Compute: one helper call over the whole block.
  - No per-tile handshakes anywhere.
- **Work distribution.** The grid comes from `device.compute_with_storage_grid_size()` and fills the
  grid on every large shape (88–110 cores). The 1-D N split covers M ≤ 32. No coordinates are
  hardcoded.
- **Prompt `## Rules`.** None violated. The op picks its own split and blocking; it derives the grid
  from the device; interleaved DRAM and L1 both go through the accessor; per-core L1 stays bounded
  (the L1 solve plus the fixed budget); fp32+acc False is an EXCLUSION; subblocks respect
  DEST_AUTO_LIMIT (4 under fp32 DEST); and there is no K split across cores, so results are
  bit-identical (tested). The rules on sharding, dram_sharded and epilogues re-arm when those axes
  are added.

### Deferred (tracked in refinements; architectural)
- **R1 is the stepping stone, not the destination.** On the passing model cases R1 is 2.2–5.5×
  stock `achievable_ns`, because every core re-reads its rows/columns of A and B from DRAM. The
  mcast successors (R1a / R1b) are the queued perf refinements 3 and 9.
- Perf lamps from the design, not exercised in this pass:
  - `InitMode::None` after the first block;
  - a `block_k` cap with deeper `depth_weight` (no compute–read overlap on S1 single-K-block
    regions);
  - `q_n`;
  - split reader;
  - K-walk stagger.
  These are knob co-tunes for the perf refinements.

### Advisories (soft rules)
- *prefer* worker cores next to their DRAM bank for dram_sharded: re-arms with Refinement 4.

## Registry Conformance
- **Op file declarations.** `INPUT_TAGGERS` (both `(inputs, axes)`), `SUPPORTED` (all 8 TARGET
  axes), `EXCLUSIONS` and `validate()` are all present.
  - `validate()` checks SUPPORTED per axis, then EXCLUSIONS, and raises `UnsupportedAxisValue` /
    `ExcludedCell`.
  - Extra placement guards follow: an interleaved weight, a TILE weight and an interleaved output
    `memory_config`.
  - The entry point runs the contract checks (`ValueError`) first, then `validate()`, then the
    dispatch. That order is intentional: the design and the golden contract tests need
    `ValueError` before the support gate.
  - Exactly one `ttnn.generic_op` dispatch per call.
- **No `INVALID` in the op file.** Confirmed.
- **SUPPORTED differs from the prompt's Phase 0 list.** `alignment` holds `m_non_aligned` in
  addition to `tile_aligned`. This is honest: the M=1 decode cells and loose cases pass, because
  each image is tile-padded and ceil-tiled.
- **Drift.** No XPASS, so no SUPPORTED auto-fixes were needed.
- **INVALID audit (feature_spec.py).** `INVALID = []`, which is well-formed: no cross-tensor
  couplings and no "not yet" entries.
  - No bf8b+ROW_MAJOR entry is needed, because TARGET `layout` is TILE only.
  - Not norm-like, so there are no weight canonicalization cells.
  - The two "not used by any model / precision convention" combinations are correctly op-side
    EXCLUSIONS, not INVALID.
  - No `INVALID_FOR_ARCH`.
- **Output `dtype` kwarg.** It is not an axis. Verified working for bf16 → fp32 and bf16 → bf8b,
  with both one-K-block and K-walk (`test_output_dtype_override`).

## L1 Ledger Audit
- **Ledger currency.** The four Phase 0 CBs match the code exactly in index, capacity expression
  and format:
  - `cb_act_tiles` 0 = `DEPTH_ACT·bm·bk`;
  - `cb_weight_tiles` 1 = `DEPTH_WEIGHT·bk·bn`;
  - `cb_output_tiles` 16 = `DEPTH_OUT·sb_h·out_bn`;
  - `cb_partials` 24 = `bm·bn`, only when `num_k_blocks > 1`.

  The later rows are clearly marked as reserved. The `l1_cb_budget` symbol row was updated for the
  fragmentation fix.
- **Size vs live set.**
  - *Over:* when a core's whole walk is **one** K-block (S1 with `nb_m = nb_n = 1`, the common
    small-shape case), the second slot of `cb_act_tiles` / `cb_weight_tiles` (`depth = 2`) never
    holds anything. That doubles the A/B footprint and makes S1 reject earlier than it needs to.
    Not fixed in place: depth 1 there also removes any read/compute overlap, which is the lamp the
    design names (`block_k` cap + `depth_weight` co-tune). **Folded into Refinement 3** (perf), the
    block-size / depth co-tune.
  - *Under:* none. Each spanned axis has capacity that scales with it.
- **Page format vs DEST.** `cb_partials` is Float32 only when `fp32_dest_acc_en` (the K-accumulator
  carries the fp32 DEST value across K-blocks), and Float16_b when it is off. The output CB is the
  output dtype (ABI). No findings.
- **Disjoint lifetimes.** None; all four CBs are concurrently live, and every row has a sharing
  justification.
- **Bounds and closed form.** The non-block symbols are bounded (`sb_h ≤ dest_limit`, tile bytes
  ≤ 4096, `p_tile_bytes ∈ {2048, 4096}`). No `tensor_*` dimension appears in a capacity.
  `F = 2·bm·bk·a + 2·bk·bn·b + [nkb>1]·bm·bn·p + 2·sb_h·bn·o ≤ l1_cb_budget`:
  - the act term scales with m·k;
  - the weight term with k·n;
  - partials with m·n;
  - output with n.
- **Block-size defaults.** Held. Interleaved spreads the output split over the full grid and takes
  the coarsest block that fits (S1 whole region; S2 K-walk; S3 shrink only under L1 pressure). The
  new grid rule departs from "W first" only on **measured** DRAM-traffic grounds (above).
- **Data-movement budget.** Present and consistent with R1: A crosses DRAM `grid_n'·nb_n` times
  and B `grid_m'·nb_m` times; the output once. The cheapest-traffic split, R1a 2-D mcast (and R1b
  for 1-D grids), is a `deferred` regime row with a positive reason (R1 is its stepping stone; same
  CBs and compute). It is now queued as Refinements 3 and 9.

## Precision Baseline

`test_genify_matmul_precision_baseline.py`: bf16 × bf16 → bf16, HiFi4, fp32 DEST, A ~ N(0,1), B ~ N(0,1/K).

| Shape (A × B) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 (bf16) | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|---|
| (1,1,32,32) × (32,32) | 0.9999985 | 0.0082 | 0.00126 | 0.00176 | 3 | 1.00012 [0.9972, 1.0030] |
| (1,2048) × (2048,1536) | 0.9999986 | 0.0082 | 0.00124 | 0.00168 | 3 | 1.00004 [0.9972, 1.0031] |
| (1,384,1536) × (1536,1024) | 0.9999986 | 0.0159 | 0.00117 | 0.00169 | 4 | 1.00004 [0.9971, 1.0030] |
| (1,64,16384) × (16384,512) (5 K-blocks) | 0.9999985 | 0.0115 | 0.00120 | 0.00172 | 4 | 0.99974 [0.9967, 1.0027] |
| (1,2048,4096) × (4096,2048) (K-walk) | 0.9999985 | 0.0182 | 0.00120 | 0.00173 | 4 | 0.99973 [0.9967, 1.0027] |

**Assessment.**
- Errors sit at the bf16 output-rounding floor: relative RMS ≈ 0.0017 ≈ bf16 half-ULP noise,
  against a gate of 0.01.
- The ratio spread is symmetric noise centred on 1. There is no scale bug.
- The multi-K-block shapes show a small, consistent −2.6e-4 median bias. That matches truncating the
  Float32 partials to tf32 (~2^-11) when the matmul helper reloads them through SrcA on the last
  K-block (design Key Risk "fp32 partial reload via SrcA").
  - It is invisible at bf16 output (ULP 3.9e-3).
  - It matters for `dtype=float32` outputs. Measured with fp32 out: rel-RMS 0.00047, inside
    (fp32, True) = 0.005.
  - It is the reason Refinement 1 carries the `UnpackToDestFp32` note.
- Golden-suite passing cells: PCC 0.99998–1.0 and rel-RMS ≤ 0.0061 (HiFi2 model cases ≈ 0.0046),
  against gates of 0.9999/0.01 (HiFi4) and 0.9995/0.03 (low fidelity).

**Recommended tolerances:** the defaults hold (`helpers.TOLERANCES` /
`LOW_FIDELITY_TOLERANCES` keyed by output dtype and `fp32_dest_acc_en`).

## Verifier CLI Summary (`verifier_report.json`, final run)
- **total:** 2213
- **supported_pass:** 94
- **xfail_expected:** 2088. Every cell maps to a TARGET−SUPPORTED value queued in `op_requirements.md`:

  | Axis = value | Cells | Refinement |
  |---|---|---|
  | dtype = bf8b | 652 | R1 |
  | dtype = fp32 | 623 | R1 |
  | weight_dtype = bf8b | 718 | R1 |
  | weight_dtype = bf4b | 645 | R1 |
  | fp32_dest_acc_en = False | 937 | R1 |
  | epilogue = bias | 382 | R2 |
  | epilogue = bias_relu | 374 | R2 |
  | epilogue = swiglu | 377 | R2 |
  | epilogue = addcmul | 375 | R2 |
  | placement = dram_sharded | 517 | R4 |
  | placement = block_sharded | 461 | R5 |
  | placement = height_sharded | 462 | R7 |
  | weight_batch = batched | 364 | R7 |

  Cells blocked by axes outside TARGET (translated reference tests) are **documented omissions**:
  `alignment=k_non_aligned` (37), `n_non_aligned` (28) and `weight_dtype=float32` (18).
- **invalid_skipped:** 0
- **no_axes_found:** 30. These are `test_regression.py` (all pass) and translated tests skipped by
  their own `skipif` on BH. They are not registry cells.
- **supported_fail:** 1, a **harness false-negative, not an op failure.**
  `test_translated.py::test_matmul_same_shape_but_invalid` expects `ValueError` for
  A(1,1,1,32) @ B(1,1,1,32), where K=32 ≠ B[-2]=1. The op raises it from the contract check before
  allocating anything. The graded run's L1 capture (`eval/l1_profiling.py`) then fails the passing
  test with `peak_total=0`. The test passes under `run_safe_pytest.sh`, where L1 capture is off.
  Fix belongs in the harness: skip the L1 peak for a test whose graded op raised the expected error.
  I did not silence it with an EXCLUSION, because the cell is correct.
- **xpass_drift:** 0
- **xfail_wrong_mode:** 0

## Model-case performance today (passing loose cases, device ns after the grid fix vs stock `achievable_ns`)

| Case | ours | stock | × |
|---|---|---|---|
| FLUX x_embedder 2048×128×3072 | 63.4 µs | 62.9 µs | 1.01 |
| guidance linear_1 1×256×6144 | 13.9 µs | 10.9 µs | 1.28 |
| guidance linear_2 1×6144×6144 | 311 µs | 192 µs | 1.62 |
| QKV txt 256×6144×9216 | 989 µs | 453 µs | 2.18 (was 2394 µs) |
| QKV img 2048×6144×9216 | 4.29 ms | 1.22 ms | 3.53 |
| FF2 img 2048×9216×6144 | 4.39 ms | 1.26 ms | 3.49 |
| single out 2304×12288×6144 | 7.46 ms | 1.69 ms | 4.42 |
| context_embedder 256×15360×3072 | 2.23 ms | 405 µs | 5.51 |

All are DRAM-traffic-bound under R1. The A/B re-reads are what the mcast regimes remove.

## Recommendations
- **Queue order and reasons.** See `op_requirements.md`.
  - Refinements 1 and 2 land the full PERF FOCUS 1 contract: bf16 × **bf4b**, LoFi, fp32 DEST,
    interleaved, **swiglu**. Refinement 3 can then optimize that exact config.
  - The 2-D mcast (R1a) is the first perf lever. Every large-M interleaved case is 3.5–4.4× stock
    under R1.
- **`GRID_W_SLACK`.** The slack is a tuning knob. A cost model of the form `max(W·tk·t_tile(fidelity),
  traffic/BW)` would replace it once the mcast regimes change the traffic term; reconsider it then.
- **Opt-in perf guard.**
  `GENIFY_MATMUL_PERF=1 eval/eval_test_runner.sh tests/ttnn/unit_tests/operations/genify_matmul/perf <out>`
  covers 14 shapes × {HiFi4, HiFi2} spanning 2-D, 1-D, tall-M and small-M / deep-K. Use it as the
  base of every perf refinement's no-regression set; extend it per new placement or dtype.
- **Harness.** Fix the L1-capture false-negative on expected-error tests (above).
- **L1 / memory-pressure observation (no OOM today).** Large interleaved shapes run at
  F ≈ 1.34–1.45 MB of the 1.46 MB budget. The L1 solve keeps them legal (S2/S3), but any epilogue
  CB (bias row, residual) or a foreign L1 tensor shrinks `block_n`, which adds re-reads. Refinement
  2 must price its CBs into `_footprint` (the ledger's `E_bias` / `E_addcmul` terms), not add them
  after the solve.
