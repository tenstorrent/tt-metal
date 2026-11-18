# Verification Report: rms_norm

Verifier pass, 2026-11-18, Blackhole p150b (110-core compute grid).
The final golden results directory is `/tmp/rms_golden_r2`. A compact copy of the CLI report is in `verifier_artifacts/verifier_report.json`; the full 25 MB file was not committed.

## Code Review

Files reviewed: `rms_norm.py`, `rms_norm_program_descriptor.py`, the three kernels in `kernels/`, `l1_ledger.md`, and `op_design.md`.

**Fixed in place**
- **Unused compile-time args removed.**
  - Reader: the `t_f` arg (index 15) was never used. The kernel only `(void)`-ed it. Removed; `TensorAccessorArgs<16>` became `<15>`.
  - Writer: `OUT_RM_DEPTH` (index 8) was never used; the writer waits and pops `core_w_tiles` per tile-row and does not need the ring depth. Removed, and indices 9–20 renumbered down by one.
  - The host-side `OUT_RM_DEPTH` constant stays: it still sizes `cb_out_rm` and the L1 budget, its only legitimate uses.
  - Re-verified: acceptance 110/110, precision baseline 16/16, full golden suite rerun with identical counts.

**Checked, no change needed**
- **Kernel hygiene.** All three kernels use `void kernel_main()`, `api/dataflow/...` include paths, and `TensorAccessor` (no `InterleavedAddrGen`).
- **CB sync.** Push and wait counts balance per CB per block, as tabulated in `op_design.md` → CB sync. Checked:
  - `cb_x`: reader pushes in `read_chunk_tiles` chunks that never straddle a block; sumsq waits cumulatively; normalize pops the block at end.
  - `cb_y`: the writer clamps each chunk to the ring end.
  - `cb_gathered` and `cb_rinv`: one block, constant landing address.
- **Helpers.** Every compute phase is a kernel_lib helper call over one block: `tilize`, `eltwise_chain` (sumsq with `DestAccumulation::PerRow`, finalize, fused normalize), `reduce` with AccumulateViaAdd for collapse and combine, and `untilize`.
  - The rinv broadcast uses the multicast helper (`ttnn.Mcast` + `McastArgs` sender/receiver).
  - The gather (many-to-one fan-in) is raw `noc_async_write` plus a semaphore. That is justified: `ttnn.Mcast` models only fan-out (`op_design.md` → "Helpers considered and rejected").
  - The raw RM stick loops are justified by the nominal-stride mismatch with `read_sticks_for_tilize`.
- **Broadcasts.** γ uses `BroadcastDim::Row` and reads only row 0. rinv is finalized once into a full tile (`UnaryBcast<Col>`), then used as a plain operand or DEST-reuse operand. No redundant fills.
- **CB index table.** Each kernel restates its CB index constants, mirroring the host `CB_*` constants. These are not block knobs, and any mismatch fails every test immediately. Noted, not changed.

## Registry Conformance
- **Op file declarations.** `INPUT_TAGGERS` (`tag_alignment`, `tag_rank`, both `(inputs, axes)`), `SUPPORTED` (all 9 TARGET axes), `EXCLUSIONS`, and `validate()` are all present and wired. `validate()` checks SUPPORTED per axis, then EXCLUSIONS. It raises `UnsupportedAxisValue` / `ExcludedCell`, and `ValueError` for rank < 2 or a γ width mismatch. The public `rms_norm()` calls `validate()` first.
- **INVALID.** The op file does **not** declare `INVALID`.
- **γ-format axes.** `tag_gamma_dtype` / `tag_gamma_layout` return `"none"` for an absent γ, and `"none"` is in SUPPORTED. They are deliberately not in `INPUT_TAGGERS`: registering them would make the harness stop iterating the two TARGET axes. `validate()` calls them directly, so the harness-built axes and the op-built axes agree (`test_axes_consistency.json` is empty).
- **Duplicated TARGET.** The op file also exports `TARGET` / `INPUT_SHAPES`, which duplicate `feature_spec.py`. They came from the stub and the harness doesn't read them.
- **XPASS auto-fixes.** None needed (`xpass_drift = 0`).
- **INVALID audit** (`eval/golden_tests/rms_norm/feature_spec.py`):
  - **OK:** the canonical `{bfloat8_b, ROW_MAJOR}` activation entry is present, as is the γ equivalent `{gamma_dtype: bf8b, gamma_layout: RM}`. The no-γ canonicalization via the `"none"` sentinel is present, in both directions.
  - **Flagged, cross-tensor:** `{layout: RM, memory_layout: *_SHARDED, gamma_layout: TILE}` (3 entries) couples the activation's layout and placement with the γ tensor's layout. This is two different tensors, and the file itself labels it "out of scope for now". Under the three sanity rules it belongs in op-side `EXCLUSIONS` if anywhere, not in INVALID. Recommend removing it.
  - **Flagged, "for now":** `{dtype: bf8b, alignment: w_non_aligned | h_non_aligned}` is also labelled "COULD eventually be supported". This is EXCLUSIONS material, not structural impossibility. Recommend moving it.
  - Neither is edited here. Please update via `/golden-tests`.
  - Consequence: those cells sit in `invalid_skipped` and are not in the refinement queue. If they are reclassified, they bundle into Refinement 1 (bf8b non-aligned: in-kernel masking of block-float pad lanes) and Refinement 2 (RM sharded + TILE γ).
- **`invalid_unexpected` = 2.** `test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[w72_c2_nonaligned-bfloat8_b]` and `[w200_c3_nonaligned-bfloat8_b]` are INVALID cells (bf8b × w_non_aligned) that the translated suite runs as xfail instead of skipping. This is a harness wiring gap in `test_translated.py`, not op drift; the op refuses them correctly.
- **`infeasible_skipped` = 2.** The `(99991, 64)` resilience shape under WIDTH_SHARDED cannot build a legal input shard. The harness skips it; nothing is charged to the op.
- **`no_axes_found` = 13.** The non-registry `test_regression.py` numerics tests (small/large magnitude, uniform, positive-only, negative-only). All 13 pass.

## Design Conformance
- **Algorithm.** Matches `op_design.md`: Σx² is folded in DEST (no x² materialized); within-tile collapse; logical `1/W` divisor; `(x·γ)·rinv` fused normalize with rinv as the DEST-reuse operand.
- **Pipeline and RISC ownership.** Reader on NoC0 and writer on NoC1. Phases and ownership match the Block schedule.
- **Work distribution.** It fills the machine. LOCAL spreads tile-rows over `min(R, C)` cores. W_SPLIT spreads W when R under-fills the grid or a row does not fit. Measured core counts: 110 cores for the 7168/8192-row prefill, 100 for 640×{6144, 7168}, 20–28 for 32×{5120, 7168}.
  - **Both halves batched.** The reader issues 8 tile reads per barrier (TILE) or one tile-row of sticks per barrier (RM). The writer writes 8 tiles per barrier or one tile-row per barrier.
  - **Overlap.** `cb_x` is two blocks deep, so the reader looks one block ahead, covering the normalize and the W_SPLIT combine stall.
- **Inter-core communication.** Gather by unicast plus a semaphore increment to the root; rinv broadcast via `ttnn.Mcast` (fixed sender = line root). The root's own copy comes from the loopback send, signalled by the local `SEM_RINV_LOCAL`.
- **Blocking-model fidelity.**
  - **Single source.** `block_rows`, `core_w_tiles`, `w_split` and `num_lines` come from `select_blocking()`. Every CB page count, trip count, RT arg and the Mcast group size is derived from them. The depth and chunk constants are defined once (host) and passed as CT args. No duplicate block literals.
  - **No CB sized to a whole-op dimension.** `cb_x` scales with `br·cw`, where `cw` is the W-slice bounded by the `k_fit` residency predicate.
  - **Per-core loop is coarse.** It is the coarsest `block_rows` that fits.
- **Expression.** At the scheduling boundary:
  - Compute phases are block-scoped helper calls.
  - The TILE reader and writer handshake per 8-tile chunk, not per tile.
  - The RM reader and writer handshake per tile-row. This is the design-sanctioned `cb_x_rm` / `cb_out_rm` streaming window over R: `tilize` / `untilize` take a CT tile-row width.
- **Axis accounting.** Ledger rows match the live sets (`cb_y` and `cb_x_rm` stream; `cb_x` spans R and W).

## L1 Ledger Audit
- **Ledger currency.** Every CB the descriptor allocates has a row, with matching page counts and formats: `cb_x_rm` [rm], `cb_x`, `cb_gamma_rm` [g_rm], `cb_gamma` [g], `cb_sumsq_raw`, `cb_partial` / `cb_gathered` / `cb_rinv_bcast` [ws], `cb_rinv_col`, `cb_rinv`, `cb_y` [tile], `cb_out_rm` [rm]. `_per_row_bytes()` / `_fixed_bytes()` in the descriptor are exactly the ledger's closed form. The non-CB semaphore `SEM_RINV_LOCAL` is recorded in the implementation notes.
- **Capacity vs live set, over.**
  - `cb_gathered` (`k·br` Float32 tiles) and the root-only `cb_rinv_col` / `cb_rinv_bcast` are allocated on every core of a W_SPLIT program, though only roots use them. This is justified (uniform remote-write addresses) and counted in the budget. But at large `w_split` it is the dominant per-core term: about 412 KB per core at `w_split = 103` for (1,1,32,32768). Disposition: folded into **Refinement 3** (lamp L4, compact column-0 gather), which shrinks the payload and the CB together.
  - No other over-capacity found. `cb_y` is a 16-tile window, not a block.
- **Capacity vs live set, under.** None. Every spanned axis scales its capacity (`cb_x` with `br·cw`, `cb_gathered` with `br·k`, `cb_gamma` with `cw`).
- **Page format vs DEST width.** Phase 0 runs only `fp32_dest_acc_en=True`, so the Float32 intermediates are correct. The *over* direction becomes a defect as soon as `fp32_dest_acc_en=False` is admitted, because the Float32 pages are hard-coded. Folded into **Refinement 1**: derive the intermediate format and `t_f` from the resolved config, in one place. No under-direction hits.
- **Disjoint lifetime.** `cb_gamma_rm` (prologue only) is the only CB disjoint from the per-block set, and its row gives a concrete reason not to alias it (different producers, at most 1/32 of a tile-row of x). Every other pair overlaps or is written in the same helper call, and each row justifies it.
- **Bounds and closed form.** `br`, `cw` and `k` each have a bound and an establishing predicate (`select_blocking()`). The total is closed-form. `k_fit` existence is enforced, and the host raises `UnsupportedAxisValue("row too wide")` otherwise.
- **Data-movement budget.** Present and consistent with the code: x and y cross DRAM once; γ is read once per core, row-0 lanes only; W_SPLIT cross-core traffic is `R·(k−1)` partial tiles plus `R` rinv tiles multicast. The cheapest-traffic alternative (GAMMA_MCAST) is a `deferred` regime row with a positive reason: γ is at most 1/16 of x's DRAM bytes. STREAM_TWO_PASS (x read twice) is rejected.
- **Block-size defaults.** Interleaved: full grid first, then the coarsest `block_rows` that fits (`select_blocking()`), which holds. The design records the departure risk as perf lamp L1: one block per core means no read/compute overlap on the last block. It has not been measured; it is a candidate lever in Refinement 5. Sharded: not built yet. Refinement 2 must use one block = the whole shard.
- **Per-core footprint.** `footprint(br, cw, k) = br·[2·cw·T_in (tile) | cw·T_in (rm) + 3·T_f + ws·(2+k)·T_f] + [g]·cw·T_g + [g_rm]·cw·32·es_γ + [tile]·16·T_in + [rm]·4·cw·T_in`. The terms scale as follows:
  - `br·cw`: `cb_x`, the dominant term.
  - `br`: the Float32 per-row tiles.
  - `br·k`: `cb_gathered`.
  - `cw`: the γ CBs, `cb_x_rm` and `cb_out_rm`.
  - Constant: `cb_y`.
  - Budget: `L1_CB_BUDGET_BYTES = 1 MiB` (lamp L8: conservative on Blackhole).

## Precision Baseline

From `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`. Settings: HiFi4, fp32 DEST, γ ROW_MAJOR in the input dtype, ε = 1e-6, randn input. Errors are measured against an fp32 PyTorch reference. ULP is measured in the output dtype against the reference rounded to that dtype. The ratio is got/true over |true| > 1e-3.

| Shape | dtype | Layout | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | Max / Mean ULP | Ratio p5 / median / p95 |
|-------|-------|--------|-----|-------------|--------------|-------------|----------------|-------------------------|
| (1,1,32,64) | bf16 | TILE / RM | 0.999996 | 0.0289 | 0.00174 | 0.00339 | 2 / 0.48 | 0.9933 / 0.9975 / 1.0013 |
| (2,4,128,512) | bf16 | TILE / RM | 0.999996 | 0.0819 | 0.00183 | 0.00359 | 2 / 0.47 | 0.9932 / 0.9975 / 1.0013 |
| (1024,1024) | bf16 | TILE / RM | 0.999996 | 0.0877 | 0.00181 | 0.00350 | 2 / 0.46 | 0.9933 / 0.9976 / 1.0013 |
| (1,1,32,8192) W_SPLIT | bf16 | TILE / RM | 0.999996 | 0.0824 | 0.00178 | 0.00344 | 2 / 0.44 | 0.9935 / 0.9977 / 1.0015 |
| (1,1,32,64) | fp32 | TILE | 0.9999998 | 0.0100 | 0.00075 | 0.00137 | 4.8e4 / 1.5e4 | 0.9976 / 0.9988 / 0.9999 |
| (1,1,32,64) | fp32 | RM | 0.9999997 | 0.0109 | 0.00095 | 0.00167 | 5.5e4 / 1.9e4 | 0.9972 / 0.9985 / 0.9997 |
| (2,4,128,512) | fp32 | TILE | 0.9999997 | 0.0187 | 0.00073 | 0.00133 | 5.7e4 / 1.4e4 | 0.9976 / 0.9989 / 1.0000 |
| (2,4,128,512) | fp32 | RM | 0.9999997 | 0.0242 | 0.00093 | 0.00165 | 6.3e4 / 1.8e4 | 0.9972 / 0.9985 / 0.9997 |
| (1024,1024) | fp32 | TILE | 0.9999997 | 0.0246 | 0.00071 | 0.00132 | 5.7e4 / 1.3e4 | 0.9977 / 0.9989 / 1.0001 |
| (1024,1024) | fp32 | RM | 0.9999997 | 0.0280 | 0.00092 | 0.00163 | 6.4e4 / 1.7e4 | 0.9973 / 0.9986 / 0.9998 |
| (1,1,32,8192) W_SPLIT | fp32 | TILE | 0.9999997 | 0.0219 | 0.00067 | 0.00124 | 5.0e4 / 1.2e4 | 0.9977 / 0.9990 / 1.0002 |
| (1,1,32,8192) W_SPLIT | fp32 | RM | 0.9999997 | 0.0235 | 0.00086 | 0.00155 | 5.8e4 / 1.6e4 | 0.9973 / 0.9987 / 0.9999 |

bf16 TILE and RM give bit-identical results. LOCAL and W_SPLIT agree to the same error profile.

**Assessment**
- **No scale bug.** Relative RMS is low (≤ 0.0036), and the got/true spread is broad and brackets 1.0.
- **Small downward bias.** Both dtypes show a small, uniform bias: bf16 median 0.9975 (about half a bf16 ULP), fp32 median 0.999. This is consistent with truncation, not rounding:
  - FPU operands are unpacked to TF32 (x, γ, rinv and the DEST-reuse operand, each up to −2⁻¹¹ relative).
  - bf16 output additionally goes through the fp32→bf16 pack step.
- **fp32 is FPU-limited.** Its ULP numbers (about 1.5e4 fp32 ULP mean, about 1e-3 relative) are the TF32 FPU precision, not a kernel defect.
- **Lever, recorded but not queued.** Moving the two multiplies to an SFPU fp32 path (`UnpackToDestFp32`) would remove the fp32 bias, at a measured cost (catalog `compute_fusion`: SFPU multiply 0.58×). There are no failing cells to move, so it is not queued.

**Recommended tolerances**:
- bf16: PCC ≥ 0.999, relative RMS ≤ 0.02, rtol = 1.6e-2, atol = 0.1.
- fp32: PCC ≥ 0.9999, relative RMS ≤ 0.002, rtol = 5e-3, atol = 3e-2.
- The golden TOLERANCES (0.995 / 0.04 for bf16, 0.999 / 0.02 for fp32) have ≥ 10× margin.

## Verifier CLI Summary

Final run, after the cleanup: `/tmp/rms_golden_r2`. The pre-cleanup run `/tmp/rms_golden_r1` was identical.

| Category | Count | Note |
|---|---|---|
| supported_pass | 659 | |
| xfail_expected | 5566 | |
| invalid_skipped | 33900 | |
| infeasible_skipped | 2 | |
| invalid_unexpected | 2 | translated-suite harness gap, see above |
| no_axes_found | 13 | non-registry numerics tests, all passed |
| **supported_fail** | **0** | |
| **xpass_drift** | **0** | |
| **xfail_wrong_mode** | **0** | |

Pytest runner: PASSED = 672, FAILED = 0, ERRORS = 0, HANGS = 0, TOTAL = 40142.

**`xfail_expected` breakdown by missing axis value → refinement:**

| Missing axis value(s) | Cells | Refinement |
|---|---|---|
| `fp32_dest_acc_en=False` only | 753 | R1 |
| `gamma_dtype=bfloat8_b` (± fp32 off) | 259 | R1 |
| `dtype=bfloat8_b` (± γ bf8b, ± fp32 off) | 209 | R1 |
| `memory_layout ∈ {HEIGHT, WIDTH, BLOCK}_SHARDED`, fp32 on, no bf8b | 1550 | R2 |
| sharded × (`fp32_dest_acc_en=False` and/or bf8b) | 2795 | R2, after R1 |

Every `(axis, missing value)` in TARGET − SUPPORTED is covered:
- `dtype`: bfloat8_b → R1
- `fp32_dest_acc_en`: False → R1
- `gamma_dtype`: bfloat8_b → R1
- `memory_layout`: HEIGHT / WIDTH / BLOCK_SHARDED → R2

No values are omitted.

## Interleaved perf snapshot (headroom check for the perf queue)

Device-kernel ns on the interleaved `_perf_case` shapes. Each value is the best of 3 runs, measured with `tests/ttnn/unit_tests/operations/rms_norm/probes/perf_loose_probe.py` under the profiler.

These were measured at the nearest **supported** config: bf16, TILE γ, HiFi2, **fp32_dest_acc_en=True**. The cases' real config (fp32 off) is not supported yet, so this is a headroom estimate, not an acceptance number.

| rows × W | ns (cores) | achievable_ns (@1350 MHz) |
|---|---|---|
| 32×1024 | 5817 (4) | 9149 |
| 32×2304 | 6465 (9) | 17003 |
| 32×5120 | 7584 (20) | 75825 |
| 32×7168 | 9146 (28) | 14894 |
| 640×6144 | 48579 (100) | 101968 |
| 640×7168 | 55041 (100) | 119307 |
| 7168×1024 | 76522 (110) | 89392 |
| 7168×2048 | 155598 (110) | 172892 |
| 8192×1024 | 86819 (110) | 96744 |
| 8192×2304 | 198521 (110) | 211345 |
| 8192×5120 | 435822 (110) | 738307 |
| 8192×7168 | 555741 (110) | 1032281 |

The interleaved cells are already at or under reference, and prefill moves about 380–420 GB/s of x+y. The tight targets are the sharded ones (4.1–5.5 µs decode, 19–26 µs block-sharded prefill). That is why the perf queue targets them first: Refinements 3 and 4, after sharding lands in Refinement 2. The interleaved refinement (5) is gated on a `/perf-ceiling-dm` headroom check.

## Recommendations
- **Queue order** (`op_requirements.md`):
  1. R1: numerics (fp32 DEST off + bf8b input/γ).
  2. R2: all three sharded placements, zero-copy.
  3. R3, perf: width-sharded decode combine latency.
  4. R4, perf: block-sharded compute throughput.
  5. R5, perf: interleaved, headroom-gated.
- **Perf targets.** No loose case carries an `attention:` perf flag, so the perf targets were chosen from the `_perf_case` entries (explicit `achievable_ns`). Every perf phase must measure at the cases' exact config: bf16, HiFi2, fp32_dest_acc_en=False.
- **Cross-cutting, R2 sharded padding.** For non-aligned W, a backed input shard's pad lanes belong to the caller's tensor. The interleaved reader's "zero the pad lanes in L1" trick would mutate the input there. Mask in compute instead.
- **Cross-cutting, R1 formats.** Refinement 1's intermediate-format derivation also changes `_per_row_bytes()`, which changes `block_rows` and the LOCAL/W_SPLIT boundary. Re-run the regime-pinned tests (`test_rms_norm.py` → `REGIME_SHAPES`) after it lands.
- **L1 pressure (no OOM today).** The W_SPLIT uniform-address rule puts `cb_gathered` (`k·br·T_f`) on every core. At `w_split ≈ 100` that is about 400 KB per core, which bounds `block_rows` and is why the 32×32768 loose case runs with `br = 1`. Addressed by R3 / lamp L4.
- **Precision without a queued lever.** fp32 output is limited by TF32 FPU operands (about 1e-3 relative, with a −0.1% bias). The SFPU fp32 multiply path would fix it at a throughput cost. Revisit only if an fp32 precision requirement appears.
- **INVALID hygiene.** Ask `/golden-tests` to move the two "author-scoped, for now" INVALID groups (RM × sharded × TILE γ; bf8b × non-aligned) to op-side EXCLUSIONS or drop them. If moved, they bundle into R2 and R1 respectively. Also ask it to wire INVALID into `test_translated.py` (2 `invalid_unexpected`).
