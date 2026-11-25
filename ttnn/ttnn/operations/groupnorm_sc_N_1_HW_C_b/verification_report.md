# Verification Report: groupnorm_sc_N_1_HW_C_b

Verifier pass over Phase 0. Phase 0 covers DRAM-interleaved bf16 input in TILE or ROW_MAJOR layout, γ/β as bf16 ROW_MAJOR, and tile-aligned HW and C. Partial channels per group are supported, with no `(C/G) % 32` gate.

Board: Blackhole, 11×10 worker grid. Date: 2026-11-25.

## Code Review

### Fixed

1. **`_shard_placement` raised `AttributeError`.** This hit every BLOCK_SHARDED call. It produced **300 `xfail_wrong_mode`** cells and broke the 12 sharded loose cases.
   - `ttnn.Tensor` has no `shard_spec()` method; the spec lives on `input_tensor.memory_config().shard_spec`.
   - Because `validate()` crashed before reaching the SUPPORTED check, sharded calls never raised `NotImplementedError`.
   - After the fix, every sharded cell is rejected through `UnsupportedAxisValue` as intended.
2. **L1 budget source: 6 of the 9 DRAM PERF FOCUS cases ran out of L1.**
   - Affected cases:
     - `vae_dram_1048576x128`
     - `vae_dram_262144x256`
     - `vae_dram_262144x512`
     - `vae_dram_1048576x256`
     - `flux1_dram_1048576x64`
     - `mochi_dram_8x407040x128`
   - Error: `Statically allocated circular buffers ... grow to 1581248 B which is beyond max L1 size of 1572864 B`.
   - Cause: the planner sized residency against `ttnn.get_max_worker_l1_unreserved_size()`. That value is measured from the kernel-config base, so it includes the 70 688 B kernel-config ring that sits below the allocator base.
   - Effect: every shape whose solve came within 70 KB of the limit overflowed, by 9–54 KB.
   - Fix: a new `l1_cb_budget(device)` in the program descriptor. It returns L1 end − allocator base (1 460 288 B on BH), read from `ttnn.get_memory_view(device, L1).total_bytes_per_bank`.
   - It also caps the budget at the lowest live L1 buffer offset in the view's block table. This automatically leaves room for any L1 tensor the caller owns, such as an R1 input shard.
   - Result: all 9 DRAM perf cases now pass at `PERF_TOLERANCE`.
   - The `L1_budget` symbol in `l1_ledger.md` and the Key Risks row in `op_design.md` were updated to match.
3. **One size expression was written in three places (DRY).**
   - The alias group B allocation is `max(4 KB·T, D_OUT·T·2 KB)`. It was computed separately in `footprint_bytes`, in the group-B CB `total_size`, and in the output-view page count.
   - All three now go through one helper, `group_b_bytes(T)`. Turning `D_OUT` or the chunk knob now changes exactly one place.
4. **`l1_ledger.md` worked examples** were replaced with the actual plans measured on BH, one per perf case: `K/B_max`, `ρ`, footprint and regime.
5. **Ledger note on `cb_sum_acc`:** records the accumulator-alias lever and where it is filed (see the L1 Ledger Audit below).

### Reviewed, no change needed

**Kernels**
- **Block-expressed throughout.** On every block axis with extent > 1:
  - The reader issues all `T` reads of a chunk under one barrier. In RM it issues `32·h` stick-segment reads per chunk.
  - Compute calls each helper once per chunk or per block: square chain, reduce with Accumulate, `matmul_block`, and the chains.
  - The writer flushes once per chunk.
  - No kernel acquires, synchronizes and retires one tile at a time.
- `kernel_main()`, `api/...` includes, and `TensorAccessor` are used throughout. No `InterleavedAddrGen`.
- **CB push and pop counts balance.** Every producer pushes the nominal `T` per chunk, including the ragged last chunk, and every consumer pops `T`. The acceptance residency pins (`max_resident_chunks` 0, 1 and default, in both layouts) exercise all three ring states.
- **Helpers are used where they apply.** The statistics broadcast uses `ttnn.Mcast` + `McastArgs`. The gather is N:1 (four 64 B row writes plus one semaphore increment per sender), which the multicast helper does not express, so the raw NoC writes are correct there.
- **Broadcasts.**
  - γ/β are expanded into row-0 tiles and consumed with `BroadcastDim::Row`. No full tile is filled with repeated data.
  - `a` and `b'` come out of the expand matmul with rows already replicated, so pass 2 is a plain elementwise op.

**Design conformance** (`op_design.md`)
- **Algorithm.** One pass computes Σx and Σx² per channel. The result is collapsed to group lanes, combined at the unit root, and folded into the normalize as `y = x·a − b'`. This matches the design.
- **Topology and work split** match: `n × xs × ht`, gather to root, mcast back.
- **The op fills the machine.** Every perf case plans 104–110 of 110 cores; mochi uses 104, due to the uniform-`cpi` lamp.
- **Block knobs are parameters, not constants.** `CHUNK_TILES_TARGET`, `D_IN`, `D_RM`, `D_OUT`, `K` (via `TUNING_OVERRIDES["max_resident_chunks"]`), `gx` and `cpi` are host-derived and passed as CT/RT args.
- **No CB scales with an unbounded op dimension.** `B` appears only through `K_alloc = min(B_max, K_max)`, which is the sanctioned predicate-guarded residency.
- **Implementer deviations** are recorded in the ledger's deviation table, and all are L1-neutral:
  - one input ring with two lockstep views instead of separate stream and resident CBs;
  - accumulate every chunk, then one publish copy;
  - writer split by role.

**Prompt `## Rules` (MUST / MUST NOT)** — all hold:
- The op file calls only `ttnn.generic_op`: exactly one dispatch.
- No `ttnn.group_norm` and no chained TTNN ops. There is no `to_layout`, `to_memory_config`, `tilize` or `untilize` op call; RM input is tilized inside the compute kernel.
- No existing group-norm implementation was read.
- The perf-claim rule (before/after device-ns per PERF FOCUS case) is followed in `changelog.md`.
- The "sharded input is read where it is" rule applies from R1 onward; it is written into R1's Done-when.

## Registry Conformance

- `INPUT_TAGGERS` match the prompt verbatim, all with the `(inputs, axes)` signature. `SUPPORTED`, `EXCLUSIONS = []` and `validate()` are present.
- `validate()` checks run in this order:
  1. Structural errors raise `ValueError`.
  2. Axes are derived from the actual call. `memory_layout` comes from `memory_config()`, `in_place` from the kwarg, and the affine axes from γ/β, including the `"none"` sentinel.
  3. SUPPORTED, per axis.
  4. EXCLUSIONS, per cell.
  5. `plan_or_raise`, the L1-feasibility shape gate, which raises `NotImplementedError`.
- The public entry point calls `validate()` first.
- The op file does **not** declare `INVALID`.
- No XPASS evidence, so SUPPORTED was not auto-widened.

**INVALID audit** (`eval/golden_tests/groupnorm_sc_N_1_HW_C_b/feature_spec.py`; report only, the file was not edited):
- **Structural entries are well-formed:**
  - bf8b+ROW_MAJOR, for both the activation and the affine tensors;
  - no-weight canonicalization, covering `gamma_beta`/`gamma_only` with a `"none"` dtype or layout, and `no_affine` with a concrete dtype or layout;
  - `{memory_layout: INTERLEAVED, in_place: True}` — single-tensor and structural, since a DRAM input has no L1 shard;
  - `{shard_fit: illegal}` — structural, since the granule leaves a core empty.
- **Scope-parked entries** (`{BS, dtype fp32/bf8b}`, `{BS, affine_dtype fp32/bf8b}`, `{BS, affine_layout TILE}`) are **not structural** and fail the sanity rules:
  - The first two describe "not built yet", not impossibility.
  - The last three couple the *input's* placement with the *γ/β* tensor's format, which are different tensors. γ/β always live in DRAM, so no kernel-level coupling forces that.
  - The spec documents them as intentional scope-parking with no refinement queued, so they are not in this queue. Removing them would turn those cells into xfail TARGET cells that R4/R7 would then need to cover on the sharded path. That decision belongs to the suite owner (`/golden-tests`).

## L1 Ledger Audit

1. **Ledger currency.**
   - Every CB the descriptor declares has a row, from `cb_input_rm` (0) through `cb_affine_staging` (21), including alias groups A {2,3}, B {4,5}, C {10,11} and D {15,18}.
   - The size expressions match `footprint_bytes()`, which is the same function the planner solves residency against.
   - The `L1_budget` row was wrong (see Code Review §2) and is fixed.
2. **Capacity vs live set.**
   - No buffer is sized above its live set without a stated reason. Double-buffered rings state the overlap they buy. Group B's x² chunk is a sequential helper intermediate that must hold one full reduce block.
   - No collapsed extents. Every span-`W` row scales with `W`, and the ring scales with `K·T`.
   - **Finding (lever, not defect):** `cb_sum_acc` / `cb_sumsq_acc` (2·W Float32 pages) are held apart from `cb_chan_scale` / group C, because the publish copy reads one while packing the other.
     - A lockstep two-view alias would make the publish copy unnecessary. The bytes are already Float32; only the `UnpackToDestFp32` tag differs.
     - That would free `2·W·4 KB`. For W = 8 this is one 64 KB resident chunk.
     - It is noted on the ledger row and **folded into Refinement 10** (DRAM streaming perf), since it is a residency lever.
3. **Page format vs DEST width.** `fp32_dest_acc_en = True`.
   - Every page that carries a DEST result between phases is Float32: x², accumulators, channel vectors, partials, sums and stats.
   - The 16-bit pages fall into three kinds, none of which carries a DEST accumulation across a phase boundary:
     - input and output bytes (the output dtype is ABI);
     - exact constants (map, ones, scaler);
     - γ/β input bytes.
   - No over- or under-width finding.
4. **Disjoint lifetimes.** Every non-shared pair carries a concrete reason: concurrent, remote-written landing zone, or stall-shadow production. The single weak spot is the accumulator pair from item 2.
5. **Bounds and closed form.**
   - Every symbol is bounded with a predicate. `B` enters only through `K_alloc`.
   - `F` is closed-form.
   - `L1_budget` now has a device-derived bound that is correct.

**Data-movement budget.**
- The budget is present and consistent with the implemented split.
  - Input DRAM crossings: `1 + ρ`. The ρ re-reads are counted.
  - γ/β reads are counted per core.
  - Cross-core traffic: `P_used·256 B` gather plus `U·8 KB` mcast.
- The cheapest-traffic statement holds. For the streaming rows, `X` exceeds the aggregate free L1, so the measured `ρ` (0.11–0.87) is the residency the budget allows, not a split choice.
- Deferred regimes and their reasons:
  - `affine_mcast` has a positive reason (γ/β ≤ 0.2% of a core's input bytes).
  - `two_pass_centered_variance` was deferred because the one-pass form meets tolerance on zero-mean data. **That reason does not survive the suite:** the golden `test_sharded_offset_heavy[mean10]` fails one-pass precision (Precision Baseline below). The regime is pulled into Refinement 1.

**Block-size defaults.**
- Interleaved: the work units (`n × xs`, then a `cpi`-way row split) fill the grid. Each core's block is its whole row range, with `K` of `B` chunks resident, which is the coarsest the L1 allows.
- `CHUNK_TILES_TARGET = 32` is the streaming quantum. It comes from the catalog plateau and was not measured on this op; it stays a perf lamp and is filed in Refinement 10.
- The sharded default (one block = the whole shard) starts with R1.

**Per-core footprint** (from `l1_ledger.md`):

```
F = 2 KB·[RM·D_RM·T + STREAM·(TILE·D_IN + RM)·T + K_alloc·T + 1 + R + W·(1 + HAS_GAMMA + HAS_BETA)]
  + 4 KB·[T + 4·W + 2·R + 6] + align_up((HAS_GAMMA + HAS_BETA)·W·64, 64)
```

Which knob each term scales with:

| Term | Scales with |
|---|---|
| `K_alloc·T` | residency × chunk |
| `D·T` and group B | chunk and depth knobs |
| `W` terms | `gx` |
| `R` terms | `cpi` |

The streaming perf cases solve to 1.40–1.45 MB of the 1.46 MB budget.

## Precision Baseline

Source: `tests/ttnn/unit_tests/operations/groupnorm_sc_N_1_HW_C_b/test_groupnorm_sc_N_1_HW_C_b_precision_baseline.py`.

All cases use bf16 with γ+β, DRAM. PCC is from `comp_pcc` and max-abs from `comp_allclose`. Relative RMS is RMS(err)/std(ref). The ratio is actual/expected over elements where |ref| > 0.1·std.

| Shape (G, layout) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ratio p5 / median / p95 |
|---|---|---|---|---|---|
| (1,1,32,32) G1 TILE | 0.999996 | 0.0312 | 0.00207 | 0.00307 | 0.9926 / 1.0000 / 1.0060 |
| (2,1,512,256) G8 TILE | 0.999996 | 0.0625 | 0.00183 | 0.00303 | 0.9928 / 1.0000 / 1.0060 |
| (1,1,1024,640) G32 RM, Cg = 20 | 0.999996 | 0.0625 | 0.00195 | 0.00319 | 0.9926 / 1.0000 / 1.0059 |
| (1,1,16384,1152) G32 TILE, Cg = 36 | 0.999996 | 0.0625 | 0.00173 | 0.00297 | 0.9929 / 1.0000 / 1.0059 |
| (1,1,65536,128) G32 TILE, eps 1e-6 | 0.999996 | 0.0938 | 0.00185 | 0.00308 | 0.9929 / 1.0000 / 1.0056 |
| (1,1,1024,1280) G32, per-group mean 5 | 0.999959 | 0.1875 | 0.00851 | 0.00982 | 0.9700 / 1.0000 / 1.0209 |
| (1,1,1024,1280) G32, per-group mean 10 | 0.999717 | 0.5625 | 0.02374 | **0.02734** (> 0.02) | 0.9178 / 0.9948 / 1.0495 |

**Assessment.**
- On zero-mean data the error is at bf16 output rounding: rel-RMS ≈ 0.003, ratio centred on 1.0. This is 3× inside `PERF_TOLERANCE` (0.999 / 0.01) on every perf-scale shape.
- Error grows with the mean/std ratio.
  - At mean 5 the rel-RMS is 0.0098, which still meets bf16 `TOLERANCES` but only just meets the perf bar.
  - At mean 10 it fails both (rel-RMS 0.027).
- **This is not a scale bug.** The ratio spread is broad (p5–p95 ≈ 0.92–1.05) and centred near 1.0. A scale bug would show high PCC with a tight cluster off 1.0. The broad spread is the signature of one-pass `E[x²] − mean²` cancellation, with the TF32 truncation of x² and channel sums in SrcA amplified by `(mean/std)²`.
- The mean-10 row is a strict `xfail` pinned to Refinement 1. The same math is what golden `test_sharded_offset_heavy[mean5/mean10]` checks once BLOCK_SHARDED cells activate. Lever: `numerical_stability_analysis_reference.md` §3.3/§4.2 and the design's `two_pass_centered_variance` regime.

**Recommended tolerances:** the defaults hold. bf16 is `TOLERANCES[bf16]` (0.995, 0.02); perf cases are `PERF_TOLERANCE` (0.999, 0.01). Do not relax either for offset-heavy data; the formula is what needs fixing.

## Verifier CLI Summary

Golden suite `eval/eval_test_runner.sh eval/golden_tests/groupnorm_sc_N_1_HW_C_b/ /tmp/gn_verify/golden_r1`: `PASSED=434 FAILED=0 ERRORS=0 SKIPPED=42345 HANGS=0 TOTAL=47152`.

`python3 -m eval.verify_supported`: a trimmed copy is in `verifier_report.json` in this directory. The full 30 MB report stays in the results dir.

| Category | Count |
|---|---|
| supported_pass | 399 |
| supported_fail | 0 |
| xpass_drift | 0 |
| xfail_wrong_mode | 0 (was 300 before fix §1) |
| supported_marked_xfail | 0 |
| xfail_expected | 4365 |
| invalid_skipped | 42345 |
| no_axes_found | 43 |

The 43 `no_axes_found` are the regression tests:
- 35 pass.
- 8 xfail on SUPPORTED:
  - `test_sharded_offset_heavy[oop/inplace × mean5/mean10]` → R1 / R2.
  - `test_sharded_program_cache_fresh_address[rm/tile × oop/inplace]` → R1 / R2.

Pre-fix run (`/tmp/gn_verify/golden_r0`): `xfail_wrong_mode` 300, and 6 DRAM loose cases ran out of L1.

All 9 DRAM loose cases pass. The 12 sharded loose cases xfail on SUPPORTED, waiting on R1/R2.

### Phase 0 device-kernel time

Measured with `_op_bench.py --profile`, "DEVICE KERNEL DURATION [ns]". The native time is `MEASURED_NS_BLACKHOLE`.

| Case | Phase 0 (ns) | native (ns) | speedup | effective DRAM GB/s |
|---|---|---|---|---|
| vae_dram_1048576x128 | 1 972 728 | 9 425 472 | 4.78× | 351 |
| vae_dram_65536x512 | 436 297 | 1 007 233 | 2.31× | 308 |
| vae_dram_262144x256 | 914 726 | 2 860 516 | 3.13× | 317 |
| vae_dram_262144x512 | 2 006 876 | 3 924 191 | 1.96× | 352 |
| vae_dram_1048576x256 | 4 213 458 | 11 404 370 | 2.71× | 355 |
| refiner_dram_16384x1152 | 242 230 | 2 261 146 | 9.33× | 312 |
| flux1_dram_16384x256 | 61 871 | 330 455 | 5.34× | 271 |
| flux1_dram_1048576x64 | 863 864 | 21 712 690 | 25.1× | 327 |
| mochi_dram_8x407040x128 | 6 382 888 | 117 736 535 | 18.4× | 375 |

Effective bandwidth counts `(2 + ρ)·X` bytes. That is roughly 53–73% of the board's nominal DRAM bandwidth, so there is headroom left, which Refinement 10 targets.

Non-perf comparisons, TILE vs RM DRAM:

| Shape | TILE (ns) | RM (ns) |
|---|---|---|
| 4096x640 | 46 212 | 64 886 |
| 16384x320 | 84 450 | 113 178 |

RM is 1.34–1.40× slower.

## Recommendations

- **Refinement 1 carries the variance precision fix.** One-pass `E[x²] − mean²` is a design-deferred regime that the suite now exercises (`test_sharded_offset_heavy`).
  - The DRAM path shares the kernel, so the fix lands on both paths.
  - The strict xfail on `offset_mean10` in the precision baseline will then XPASS, so the mark must be removed.
  - A full extra DRAM pass would cost `ρ·X` on the streaming VAE/Mochi cases (ρ up to 0.87), so the formulation choice must be measured against them (see the R1 verifier notes).
- **`--dev` kernel-config ring.** The implementer kept kernel text just under the 70 688 B config ring under `--dev` by halving helper instantiations; see the ledger deviation table.
  - Every refinement that adds kernel variants (RM untilize for sharded output in R1, partial scalers in R5, bf8b packs in R7) risks overflowing it.
  - Each refinement should re-run the acceptance suite with `--dev`. If text grows, split the compute kernel by CT-specialized role, as the writer already is, rather than dropping `--dev` coverage.
- **RM DRAM reads are 1.34–1.4× slower than TILE.** Each stick-segment read is `W·64 B`, which is latency-bound at small `W_core`.
  - No PERF FOCUS case is DRAM-RM, so this is not queued.
  - If an RM DRAM model case is added, the candidate patterns are `read_coalescing` / `trid_double_issue` (master.md).
- **Uniform `cpi` idles cores when `P % U ≠ 0`** (mochi uses 104/110). This is folded into Refinement 10.
- **Phase 0 already beats native on every DRAM perf case** (1.96×–25×). The sharded perf cases (R8/R9) have no Phase 0 number yet; R1/R2 must record a first measurement when they land.
- **Queue shape vs. the 2:1 cadence.** The op prompt fixes Refinements 1–7, in order and unmerged, with measured perf refinements after R7. The runner adds its own trailing perf rounds after R2. So R8–R10 are the perf entries, rather than every third slot.
  - The anchor rule still holds: R1 + R2 land the full contract of the first perf target (sharded RM in-place, 8×8, bf16 γ/β RM).
- **Scope-parked INVALID entries** couple input placement with γ/β format (see the INVALID audit). The suite owner should decide whether they stay parked.
