# Verification Report: groupnorm_sc_N_1_HW_C

Verifier pass over the Phase-0 implementation (commits after `7cfd55570f`). Device: Blackhole p150a, 11×10 compute grid (110 cores). All device runs through `scripts/run_safe_pytest.sh --dev` / `eval/eval_test_runner.sh`; perf through `--profile` (Tracy per-op CSV) and the golden harness's `device_kernel_ns` (the two agree within ~1 %).

## Code Review

Everything below was **fixed in code** during this pass unless marked *deferred*.

### Prompt rules (`eval/prompts/groupnorm_sc_N_1_HW_C.txt`)
- **Centered variance (MUST)** — verified: pass 2 forms `Σ(x − mean_row)²` from the pass-1 mean (`centered_sq_chunk`); no `E[x²]` anywhere. `test_large_mean_stable_variance` (`mean = 10σ`) green; precision baseline at `mean = 10σ` PCC 0.9999 (see below).
- **Partial channels from Phase 0 (MUST)** — verified: no `(C/G) % 32` gate; `groups_alignment` lists both values; every SD/SDXL straddling shape in INPUTS passes. Group aggregation is a membership matmul, never a lane reduce.
- **No external mask tensor / no host-side prepare (MUST)** — verified: the 0/1 membership (and its transpose) are built on-device by the writer from runtime args.
- **Exact signature / import path / ValueError set** — verified by the acceptance test.
- **Multi-core (MUST)** — verified: 2-D `(hw_splits × c_splits)` split fills all 110 cores on every SDXL shape; ragged shapes use `min(HWt·Ct, 110)`.

### Registry conformance
Confirmed present and correctly wired: `INPUT_TAGGERS` (`alignment`, `groups_alignment`; both `(inputs, axes)` signatures), `SUPPORTED` (covers every TARGET axis: dtype, layout, memory_layout, in_place, alignment, groups_alignment, affine, affine_dtype, affine_layout), `EXCLUSIONS = []`, `validate()` (SUPPORTED per-axis then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`), `validate()` is the first statement of the entry point. The op file declares **no** `INVALID`. No drift auto-fixes were needed (`xpass_drift = 0`).

**INVALID audit** (`feature_spec.py`): all entries pass the three sanity rules — `INTERLEAVED × in_place=True` (same tensor's placement), the canonical `bf8b × ROW_MAJOR` for the activation **and** for the affine tensors (single-tensor couplings), and the full `no_affine ↔ "none"` canonicalization set (both directions). No cross-tensor couplings, no "not yet supported" entries disguised as INVALID. Nothing to report to the spec author.

### Design conformance (`op_design.md`)
- **Algorithm**: matches — per-channel column sums (`reduce<SUM, REDUCE_COL, AccumulateViaAdd>`), membership matmul for the group aggregation, centered second pass, affine apply. Membership handles straddling groups and the trailing-C padding.
- **Combine topology (documented deviation, planner's own lamp)**: the design's flat all-gather (every core multicasts to every core, `num_active²` atomics) measured 1.15 ms on `(1,1,4096,320)` at 110 cores; the implementer built the design's named alternative — root-reduce + `mcast_pipe` broadcast (34 µs). This pass kept that topology and replaced the root's *reduce* with the aggregation matmul (below). `l1_ledger.md` now describes the implemented protocol (its "all-gather data path" note and `num_active²` traffic row were stale — fixed).
- **Work distribution / grid fill (performance-conformance)**: full-grid 2-D split ✓; reader batches a whole chunk of tile reads per barrier, writer a whole chunk of writes per barrier ✓; `STREAM_DEPTH = 2` double-buffering on the streaming CBs ✓; resident pass 1 overlaps the DRAM fill (cumulative `cb_wait_front`) ✓. **Defect found and fixed**: the split-search objective (min max-tiles-per-core, `hw_first` tiebreak) picked `K = 1` for the 16384-row SDXL shapes on this grid — 64 B ROW_MAJOR stick slices — making RM 2.2–4.6× slower than TILE on the same shape, and picked a `K = 16` *streaming* split for the VAE shape where an equal-tile-count split was L1-resident. See "Split search" below; measured, encoded as host knobs.
- **Blocking-model fidelity**: `K`, `Q`, `D`, `Ng`, `GT`, the split and now the residency-seeking `Q` are all derived once on the host (`config.py` → `create_program_descriptor`) and passed as CT/RT args; CB page counts, loop trip counts and the `fixed_footprint()` closed form are all computed from them (DRY holds). No CB scales unconditionally with a whole-op dimension: the resident `cb_input_tiles = core_hw_tiles·K` is the sanctioned predicate-guarded fast path with a streaming fallback. Chain block size `b = largest divisor of K ≤ DEST_AUTO_LIMIT` is a compile-time derivation from `K` (it collapses to 1 for `K ∈ {1, 5}` — a perf lamp filed in Refinement 3, not a fidelity defect: the quantum contract is what forces `b | K` on the RM path).
- **Expression**: reader, compute and writer all operate at chunk granularity (`Q·K` tiles) with one synchronization per chunk; the block schedule reads as blocks. Uniform `K` per program (tilize/untilize template width) is a documented deviation from the design's remainder column split.
- **Helper usage**: reduce / matmul_block / eltwise chains / tilize / untilize / `mcast_pipe` / `prepare_reduce_scaler` / `read_sticks_for_tilize` / `write_sticks_after_untilize` throughout; `TensorAccessor` for interleaved I/O; `void kernel_main()`; `api/dataflow/dataflow_api.h` includes ✓. **Fixed**: the compute kernel's raw `binop_with_scalar.h` / `rsqrt.h` post-op is gone — the finalize is a helper chain (`CopyTile → AddUnary → Rsqrt → PackTile`).
- **Broadcasts**: pass-2 `Sub` and pass-3 `Mul` use `BroadcastDim::Row` on row-0-valid per-channel tiles indexed by column (`InputTileMapping::Row`) ✓; `cb_shift_full` is the one pre-expanded full-tile operand (chain `DestReuseBinary` cannot broadcast — design-documented).
- **CB sync**: push/pop counts audited per CB across reader/compute/writer, including the nominal `Q·K` quantum pads on ragged tails, the `Hmax·K` resident pad, the root's `Ng·GT` gather push/pop, the per-round `GT` re-push of `cb_inv_rows`, and the pass-3 pops (`cb_group_var` by the finalize chain, `cb_group_rstd` in release) ✓. Idle rectangle cores never touch the landing CBs and run only the mcast handshake ✓.

### Fixes made in this pass
1. **Split search (host, measured)** — `choose_split` now (a) prefers an L1-resident split for TILE inputs and **shrinks the chunk `Q`** (down to 1 tile-row) when that makes the assignment fit (`SPLIT_PREFER_RESIDENT`; RM excluded: `SPLIT_PREFER_RESIDENT_RM = False` because stick width dominates there); (b) for ROW_MAJOR widens the per-core column block up to `SPLIT_RM_STICK_K_TARGET = 10` (640 B sticks) within `SPLIT_COST_TOLERANCE_RM = 10 %` extra tiles, then falls back to fewest tiles; (c) for TILE keeps the strict min-tiles objective and breaks ties toward the **narrower** `K` (`SPLIT_COST_TOLERANCE_TILE = 0`). `FORCE_C_SPLITS` test pin added. Every choice is backed by a measured pair in `config.py` comments (sweep: `tests/.../probes/probe_ksweep.py`, 60 profiled cells).
2. **Kernel-config ring (dev build) — correctness on the RM `K = 10` cells.** The layout-aware split made the 16384-row RM shapes pick `K = 10`, whose `--dev` (watcher) binaries overran the 70 656 B kernel-config ring by 1.7 KB (`Program size (72336) too large`). Fixed without algorithmic change, compute binary 57.2 KB → 42–50 KB, program ≈ 65 KB:
   - root combine = the **aggregation matmul** (`(1×GT) @ (GT×Ng)` against `cb_inv_rows`, row 0 = `1/n`) instead of a second `reduce` instantiation + raw SFPU post-op; `cb_gather` is a single landing region (the root-reduce protocol makes the second half unnecessary — argument in `l1_ledger.md`);
   - the writer builds the membership **and its transpose** (`cb_membership_t`), so aggregation, combine and both expansions are ONE non-transposed `matmul_block` body (`row_matmul_block`, runtime CB ids, retain + manual pop); the transposed instantiation and two templated copies (11.3 KB) are gone;
   - `rstd = rsqrt(var + eps)` is formed on the `Ng` group tiles into a compute-owned `cb_group_rstd`, then expanded and multiplied by gamma (FPU `mul`); forming it on the `K` expanded rows first measured ~1 µs per tile of exact SFPU rsqrt (4–9 % on the 1024-row shapes — reverted);
   - grid-wide constants moved to **common runtime args** (reader 12 → 4, writer 18 → 6, compute 9 → 3 per-core args); membership waits deferred to first use so the writer's build overlaps the pass-1 fill.
3. **`-Os` for TRISCs verified broken** (`ckernel_addrmod.h:192: impossible constraint in 'asm'`), confirming the implementer's note — and `opt_level` is **not** part of the JIT hash, so a stale binary was silently reused until evicted. Not used.
4. **Ledger** brought in line with the implementation (combine protocol, traffic row, `cb_fp32_scratch` lifetime, new CBs, closed form, worked values, cheapest-traffic line).

### Deferred (architectural, filed)
- Block-sharded placement, `in_place`, non-tile-aligned HW/C, the dtype set, TILE affine weights → Refinements 1, 2, 4.
- Two-pass streaming statistics for the DRAM-bound VAE shapes; fixed-cost floor of the combine → Refinements 5, 6.

## Code size (the constraint every refinement must respect)
The per-core kernel-config ring on Blackhole is **70 656 B** and holds the reader + writer + compute binaries (plus a few hundred bytes of config). The `--dev` build adds ~28 KB of watcher/LLK-assert code. Phase-0 dev sizes now: compute 42–50 KB (per config), reader 3.8–4.8 KB, writer 7.3–10 KB (RM) → program 60–66 KB. Every helper instantiation costs 1–7 KB (`reduce` ≈ 5–6 KB, `matmul_block` 3–7 KB, a chain 1–3 KB, `tilize<10>` 2.5 KB, `untilize<10>` 3.3 KB). Rules that held up: one body per helper family (runtime CB ids, `noinline, noclone`), share instantiations across phases, prefer `if constexpr` variants inside a body over new bodies. `-Os`/`-Oz` are not available on TRISC.

## L1 Ledger Audit
- **Ledger currency**: after this pass every declared CB has a row and every row a live CB: 19 CBs on the RM leg (`cb_input_tiles`, `cb_input_sticks`, `cb_scaler`, `cb_membership`, `cb_membership_t`, `cb_gamma_rows`, `cb_beta_rows`, `cb_inv_rows`, `cb_colsum_rows`, `cb_partial_rows`, `cb_gather`, `cb_group_mean`, `cb_group_var`, `cb_group_rstd`, `cb_stats_bcast`, `cb_mean_rows`, `cb_scale_rows`, `cb_shift_full`, `cb_fp32_scratch`, `cb_output_tiles`, `cb_output_sticks`); size expressions match `fixed_footprint()` term by term. Stale entries fixed: `cb_gather` capacity/producer/consumer, `cb_fp32_scratch` lifetime (pass 1 **and** 2), `cb_group_rstd` semantics, the combine traffic row.
- **Capacity vs live set**: over — only the stated `D = 2` double buffers and nothing else; under — no CB spans an axis without scaling with it (`cb_colsum_rows` streams the raw running sum, `cb_fp32_scratch` streams `Q`). No finding.
- **Page format vs DEST width**: `fp32_dest_acc_en = True`; every compute-produced CB is `Float32`; the 16-bit pages are consumed-only (input, weights, scaler) or the tensor-dtype output. No finding.
- **Disjoint lifetimes**: `cb_colsum_rows` (pass 1 ↔ 2, also the Accumulate accumulator) and `cb_mean_rows` (pass 2 ↔ 3) are reused; `cb_scale_rows` in place; the `cb_shift_full ↔ cb_colsum_rows` alias is recorded, not taken (folded into Refinement 5's L1 budget work, not its own entry); `cb_group_rstd ↔ cb_partial_rows` cannot alias (dataflow consumer). Recorded.
- **Bounds and closed form**: `K ≤ min(16, Ct)`, `Ng ≤ 4`, `GT ≤ ceil(110/32) = 4`, `Q` default `max(1, 32 // K)` and now halved toward 1 by the residency search, `D = 2` — all in the symbol table with predicates; the total is closed-form and is exactly the `fixed_footprint()` the split search evaluates. Per-core total = `fixed(K, Ng, GT, Q, layout, gamma, beta) + input` where input is `TB·core_hw_tiles·K` (resident) or `TB·D·Q·K` (streaming); the `K`-proportional terms dominate (`membership ×2`, four `K`-row CBs, weights), the `Q·K` terms (`scratch`, `output ×D`) are the residency lever the search now turns.
- **Data-movement budget**: present and consistent — input 1× (resident) / 3× (streaming), output 1×, weights `hw_splits` re-reads (≤ 140 KB vs 10.5 MB, multicast not warranted), stats never touch DRAM; cross-core per image = `num_active·Ng·128 B` unicast + one `Ng·4 KB` multicast + ~440 atomics at 110 cores (was `num_active²` atomics). **Cheapest-traffic split**: the split search now ranks residency first for TILE (implemented; `(1,1,65536,512)` TILE 733 → 499 µs) and documents why RM does not (measured). `block_sharded_resident` (0 DRAM crossings) remains the deferred row → Refinement 1.
- **Block-size defaults**: interleaved — full grid, then the coarsest block that fits: held (the residency-seeking `Q` shrink is exactly "coarsest that fits"). The RM departure (wider `K` over fewer tiles) is justified by measurement (below), not by the solve settling.

## Split search — measurements that set the knobs (device kernel µs, 110 cores)

| Shape | Layout | K=1 | K=2 | K=4/5 | K=8 | K=10 | K=12 | K=15/16 |
|---|---|---|---|---|---|---|---|---|
| `(1,1,16384,320)` | TILE | **87.7** | 90.3 | 101.6 (K=5) | — | 122.5 | — | — |
| `(1,1,16384,320)` | RM | 408.5 | 245.2 | 150.4 (K=5) | — | **140.5** | — | — |
| `(1,1,4096,640)` | TILE | — | **54.4** | 60.2 / 64.5 | — | 85.6 | — | — |
| `(1,1,4096,640)` | RM | — | 129.5 | 93.3 / **90.2** | — | 95.0 | — | — |
| `(1,1,16384,960)` | TILE | — | — | — | — | 265.7 | — | 282.3 (K=15) |
| `(1,1,16384,960)` | RM | — | — | — | — | **318.3** | — | 401.1 (K=15) |
| `(1,1,4096,1920)` | RM | — | — | — | — | — | **185.0** | 200.0 (K=15) |
| `(1,1,65536,512)` | TILE | — | — | **497 (K=4,Q=4 resident)** | 691 (Q=2 stream) | — | — | 734 (stream) |
| `(1,1,65536,512)` | RM | — | — | 916 (K=4 resident) | 985 (stream) | — | — | **771 (stream)** |

Reading: TILE moves whole pages — the strict min-tiles pick (narrower `K`) wins; RM moves `K·64 B` stick slices — widen `K` until ~640 B, beyond which it stops paying (K=15 lost to K=10/12 twice); residency is worth 1.47× on TILE but does not beat stick width on RM. All single runs (run-to-run spread ~2–4 %); Refinement 6 re-measures the marginal calls with repeats.

## Precision Baseline (bf16 in/out, bf16 RM gamma+beta, HiFi4 + fp32 DEST; `test_groupnorm_sc_N_1_HW_C_precision_baseline.py`)

| Shape | G | Layout | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|---|---|
| (1,1,32,32) | 1 | TILE / RM | 0.999999 | 0.031 | 0.00058 | 0.0016 | 1.0000 [0.9954, 1.0042] |
| (1,1,128,128) | 4 | TILE / RM | 0.999999 | 0.031 | 0.00048 | 0.0015 | 1.0000 [1.0000, 1.0048] |
| (1,1,1024,640) | 32 (straddling) | TILE / RM | 0.999999 | 0.063 | 0.00050 | 0.0015 | 1.0000 [1.0000, 1.0048] |
| (2,1,256,1280) | 32 (straddling, N=2) | TILE / RM | 0.999999 | 0.063 | 0.00050 | 0.0015 | 1.0000 [1.0000, 1.0049] |
| (1,1,256,320), mean = 10σ | 32 | TILE / RM | 0.999895 | 0.125 | 0.0154 | 0.0145 | 1.0000 [0.939, 1.060] |

**Assessment**: zero-mean data is at the bf16 output quantization floor (rel-RMS 0.15 %, max-abs = 1 bf16 ulp at |y| ≈ 4). The ratio spread is centred on 1.0000 with a broad, symmetric tail — ordinary rounding, **no scale/structural signature** (a scale bug would show a tight cluster off 1.0). The `mean = 10σ` case shows the documented tf32 operand rounding of the fp32 statistics on the FPU (design "Key Risks"): the group mean is rounded ~3× at `2⁻¹¹` relative (colsum → gather sum, the `1/n` fold in the combine matmul added one), i.e. ≈ `3·2⁻¹¹·|mean|/σ ≈ 1.5 %` per-group offset of σ; still 4× below the bf16 *input* quantization of the same mean (`2⁻⁸·|mean|`). PCC 0.99989 ≫ the 0.995 gate. Lever if fp32 inputs need it (Refinement 4 notes): move `1/n` back to the SFPU.
**Recommended tolerances**: bf16 — PCC ≥ 0.995 (harness), rel-RMS ≤ 0.02; for `|mean| ≳ 10σ` inputs expect rel-RMS ≈ 0.015.

## Verifier CLI Summary (`verifier_report.json`, final kernels)
- supported_pass: **346**
- xfail_expected: 10 784
- invalid_skipped: 0 (INVALID cells are pruned at collection — they appear as `no_axes_found: 46 868`, not as a signal)
- infeasible_skipped: 0
- supported_fail: **0**
- xpass_drift: **0**
- xfail_wrong_mode: **0**
- Acceptance / debug / perf / precision unit tests: 119 / 119 in `--dev` (watcher + LLK asserts) — including the three RM `K = 10` SDXL cells that overflowed the kernel-config ring before the code-size work.

## Perf baseline — SDXL / VAE loose cases (interleaved, bf16, gamma+beta), device kernel µs

Baseline = implementer's Phase 0 as handed over; final = this pass. Reference (`perf_cases.py`, production `ttnn.group_norm`, interleaved TILE at 1350 MHz): 16384×320 600 µs, 4096×640 225, 1024×1280 97, 1024×2560 120, 4096×1920 336, 16384×960 818, 16384×640 474, 65536×512 1015, 262144×256 2931 — all beaten 2–7× by the TILE cells below. RM interleaved has no measured reference; the RM *sharded* model bars (337 / 105 / 44 / 139 / 390 / 382 µs) are the Refinement 3 target.

| Shape | Layout | baseline µs | final µs | speedup | PCC |
|---|---|---|---|---|---|
| `(1,1,1024,640)` | RM | 43.7 | 44.7 | 0.98x | 0.999999 |
| `(1,1,1024,640)` | TILE | 25.6 | 26.3 | 0.97x | 0.999999 |
| `(1,1,1024,1280)` | RM | 59.0 | 60.2 | 0.98x | 0.999999 |
| `(1,1,1024,1280)` | TILE | 43.3 | 43.9 | 0.99x | 0.999999 |
| `(1,1,1024,1920)` | RM | 74.3 | 75.9 | 0.98x | 0.999999 |
| `(1,1,1024,1920)` | TILE | 59.8 | 61.0 | 0.98x | 0.999999 |
| `(1,1,1024,2560)` | RM | 90.4 | 91.7 | 0.99x | 0.999999 |
| `(1,1,1024,2560)` | TILE | 78.1 | 79.0 | 0.99x | 0.999999 |
| `(1,1,4096,320)` | RM | 70.8 | 71.3 | 0.99x | 0.999999 |
| `(1,1,4096,320)` | TILE | 35.1 | 32.1 | 1.09x | 0.999999 |
| `(1,1,4096,640)` | RM | 93.4 | 91.1 | 1.03x | 0.999999 |
| `(1,1,4096,640)` | TILE | 60.5 | 54.7 | 1.11x | 0.999999 |
| `(1,1,4096,960)` | RM | 118.6 | 120.1 | 0.99x | 0.999999 |
| `(1,1,4096,960)` | TILE | 90.2 | 76.5 | 1.18x | 0.999999 |
| `(1,1,4096,1280)` | RM | 139.5 | 142.3 | 0.98x | 0.999999 |
| `(1,1,4096,1280)` | TILE | 113.8 | 100.2 | 1.14x | 0.999999 |
| `(1,1,4096,1920)` | RM | 185.2 | 188.2 | 0.98x | 0.999999 |
| `(1,1,4096,1920)` | TILE | 165.4 | 143.9 | 1.15x | 0.999999 |
| `(1,1,16384,320)` | RM | 404.9 | 143.9 | **2.81x** | 0.999999 |
| `(1,1,16384,320)` | TILE | 87.6 | 88.6 | 0.99x | 0.999999 |
| `(1,1,16384,640)` | RM | 452.3 | 235.3 | **1.92x** | 0.999999 |
| `(1,1,16384,640)` | TILE | 164.0 | 164.0 | 1.00x | 0.999999 |
| `(1,1,16384,960)` | RM | 511.8 | 321.2 | **1.59x** | 0.999999 |
| `(1,1,16384,960)` | TILE | 233.9 | 235.5 | 0.99x | 0.999999 |
| `(1,1,65536,512)` | RM | 787.2 | 773.1 | 1.02x | 0.999999 |
| `(1,1,65536,512)` | TILE | 732.7 | 498.5 | **1.47x** | 0.999998 |
| `(1,1,262144,256)` | RM | 3357.4 | 1947.2 | **1.72x** | 0.999998 |
| `(1,1,262144,256)` | TILE | 1456.0 | 1402.3 | 1.04x | 0.999998 |
| `(1,1,262144,512)` | RM | 3848.0 | 2844.3 | **1.35x** | 0.999996 |
| `(1,1,262144,512)` | TILE | 2686.5 | 2686.2 | 1.00x | 0.999996 |
| `(1,1,1048576,128)` | RM | 6773.8 | 6827.1 | 0.99x | 0.999996 |
| `(1,1,1048576,128)` | TILE | 2665.8 | 2711.2 | 0.98x | 0.999996 |
| `(1,1,1048576,256)` | RM | 7664.9 | 7661.5 | 1.00x | 0.999994 |
| `(1,1,1048576,256)` | TILE | 5333.9 | 5541.7 | 0.96x | 0.999994 |

The ±2–4 % entries are within run-to-run spread (the VAE 1048576-row cells vary ~4 % between identical runs). The streaming VAE cells sit at 340–375 GB/s aggregate over 4 tensor volumes — DRAM-bound; only fewer passes moves them (Refinement 5).

## Recommendations
- **Queue** (`op_requirements.md`): R1 block-sharded + in-place (native shard-placed CBs; the anchor for the perf-competitive SDXL target), R2 non-tile-aligned HW/C, R3 perf on the sharded RM model configs, R4 numerics + TILE affine, R5 two-pass streaming stats (VAE), R6 fixed-cost floor.
- **Code size is a first-class constraint** for every refinement: budget ≈ 5 KB of dev-build headroom today; add variants with `if constexpr` inside existing bodies rather than new helper instantiations; re-check `Program size` on the RM `K = 10` and `K = 12` cells after every kernel change.
- **Precision without a lever in scope**: the `1/n`-in-matmul fold costs one extra tf32 rounding of the mean (measured above). Harmless for bf16; revisit when fp32 inputs land (Refinement 4 notes).
- **Chain block size** `b = largest divisor of K ≤ 4` is 1 for `K ∈ {1, 5}` (per-tile push/pop on `(1,1,32,·)`, `(1,1,64,160)` and the `K = 5` RM picks). The TILE quantum pads make `b | Q·K` sufficient there; the RM path's ragged untilize tails are why `b | K` was chosen. A perf lamp, filed in Refinement 3.
- **Measurement hygiene**: the K/Q sweep in `tests/.../probes/probe_ksweep.py` is single-run; two calls (K=15 vs K=12 on 4096×1920 RM, K=6 on 16384×960 RM) are within 10 % and should be repeated before the knobs are treated as final.
- **Friction worth forwarding** (also in the implementer's breadcrumbs): the kernel-config ring and the dev-build inflation are undocumented in the references; `opt_level` is not hashed into the JIT cache key (a changed level silently reuses a stale binary); `-Os` is unusable on TRISC.
