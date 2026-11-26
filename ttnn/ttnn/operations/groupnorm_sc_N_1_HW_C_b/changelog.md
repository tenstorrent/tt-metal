# Changelog: groupnorm_sc_N_1_HW_C_b

## Phase 0 — Core Implementation
- **Date**: 2026-11-25
- **What was done**: initial implementation via the incremental pipeline (planner → implementer → verifier).
  - The op is one `ttnn.generic_op` dispatch with an `n × xs × ht` multi-core split.
  - Statistics are one pass of Σx and Σx² per channel, using fp32 `UnpackToDestFp32` accumulators.
  - These are collapsed to group lanes by a one-hot map matmul, gathered to a per-unit root, finalized, and returned to the unit with `ttnn.Mcast`.
  - The normalize step is folded into `y = x·a − b'`.
  - Input chunks are kept resident in L1 as far as the budget allows (`dram_resident` / `dram_partial_resident`).
- **SUPPORTED at Phase 0**:
  - dtype = [bfloat16]
  - layout = [TILE, ROW_MAJOR] (RM is tilized inside the kernel; the output is TILE)
  - alignment = [tile_aligned]
  - affine = [gamma_beta, gamma_only, no_affine]
  - affine_dtype = [bfloat16, "none"]
  - affine_layout = [ROW_MAJOR, "none"]
  - memory_layout = [INTERLEAVED]
  - shard_fit = ["none"]
  - in_place = [False]
  - Partial channels per group (`(C/G) % 32 ≠ 0`) are supported, with no gate.
  - Compute config is fixed: HiFi3 + fp32 DEST.
- **Accuracy achieved** (`test_groupnorm_sc_N_1_HW_C_b_precision_baseline.py`, 5 zero-mean shapes from (1,1,32,32) to (1,1,65536,128)):
  - PCC = 0.999996
  - max_abs_err ≤ 0.094
  - relative RMS ≈ 0.0030–0.0032
  - got/true ratio median 1.0000 (p5/p95 ≈ 0.993/1.006)
  - Offset-heavy data (per-group mean 10, std 1) gives rel-RMS 0.027. That is one-pass variance cancellation; it is a strict xfail and assigned to Refinement 1.
- **Golden suite at Phase 0**: 399 `supported_pass`; 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`. That is 434 of 47 152 collected cells passed; the rest are 4 365 `xfail_expected` and 42 345 `invalid_skipped` (`verifier_report.json`).
- **Issues encountered** (verifier fixes):
  - `_shard_placement` called the non-existent `Tensor.shard_spec()`. Every BLOCK_SHARDED cell crashed in `validate()` with `AttributeError`, giving 300 `xfail_wrong_mode`. It now reads `memory_config().shard_spec`.
  - The L1 budget was taken from `ttnn.get_max_worker_l1_unreserved_size()`. That value includes the 70 688 B kernel-config ring, so 6 DRAM perf cases ran out of L1 (9–54 KB over). It is replaced by `l1_cb_budget(device)`: allocator-base-relative, 1 460 288 B on BH, and capped at the lowest live L1 buffer.
  - The alias group B allocation expression was duplicated in three places. It is now the single source `group_b_bytes(T)`.
  - The ledger and design docs were updated to match.
- **Tests added**:
  - `test_groupnorm_sc_N_1_HW_C_b.py` (planner / implementer)
  - `test_groupnorm_sc_N_1_HW_C_b_precision_baseline.py` (verifier)
  - `_op_bench.py`: device-time bench, run with `scripts/run_safe_pytest.sh --profile --no-py-zones`; not collected by default.

### Device kernel duration, PERF FOCUS cases (Blackhole)

Measurements are DEVICE KERNEL DURATION [ns], median of the trial loop, bf16 with γ+β.
- **"Before"** is the implementer's build with the planner's L1 budget. It ran out of L1 at program creation on 6 of 9 cases, so it has no time for them.
- **"Phase 0"** is after the verifier's budget fix.
- **Native** is `MEASURED_NS_BLACKHOLE`, the existing `ttnn.group_norm` with the model's args.

| Case | Before (ns) | Phase 0 (ns) | Native (ns) | Phase 0 vs native |
|---|---|---|---|---|
| vae_dram_1048576x128 | OOM | 1 972 728 | 9 425 472 | 4.78× faster |
| vae_dram_65536x512 | = Phase 0 † | 436 297 | 1 007 233 | 2.31× faster |
| vae_dram_262144x256 | OOM | 914 726 | 2 860 516 | 3.13× faster |
| vae_dram_262144x512 | OOM | 2 006 876 | 3 924 191 | 1.96× faster |
| vae_dram_1048576x256 | OOM | 4 213 458 | 11 404 370 | 2.71× faster |
| refiner_dram_16384x1152 | = Phase 0 † | 242 230 | 2 261 146 | 9.33× faster |
| flux1_dram_16384x256 | = Phase 0 † | 61 871 | 330 455 | 5.34× faster |
| flux1_dram_1048576x64 | OOM | 863 864 | 21 712 690 | 25.1× faster |
| mochi_dram_8x407040x128 | OOM | 6 382 888 | 117 736 535 | 18.4× faster |

† These three cases are `dram_resident` (`K = B`), so they already fit before the fix. With the old and new budgets, their plans differ only in the unused `K_max` headroom (14/15, 18/19 and 17/18 vs `B_max` = 10, 7 and 2), as compared by `probes/probe_008.py`. The CBs, RT args and program are identical, so the "before" time is the Phase 0 time; it was not separately re-timed. No perf win is claimed for them.

The 12 block-sharded PERF FOCUS cases (`unet_rm_ip_*`, `refiner_rm_ip_*`, `unet_tile_*`, `vae_tile_16384x512`) are outside the Phase 0 SUPPORTED set. They xfail until Refinements 1–2.

## Refinement 1 — L1 BLOCK_SHARDED input, out of place (even shards, ROW_MAJOR + TILE, bf16), plus the variance precision fix
- Date: 2026-11-25
- What was done:
  - **Registry.** `SUPPORTED["memory_layout"]` gains `BLOCK_SHARDED` and `SUPPORTED["shard_fit"]` gains `"even"`; `in_place = True` and `ragged` stay unsupported (186 cells xfail cleanly in `validate()`).
  - **Native zero-copy sharding** (`ttnn.cb_descriptor_from_sharded_tensor`; no `TensorAccessor` touches a core's own shard):
    - TILE: the input shard backs the input ring (both views, `K = B`); pass 2 packs straight into the output shard (`cb_output_tiles`).
    - RM: the input shard backs `cb_input_rm` (stick pages viewed as tile-row pages, `k = lcm(sw, 32)/sw` sticks per virtual row, so 40-/80-channel slices tilize to `Wt_core` tiles with the pad lanes masked by the group map); pass 2 untilizes into the output shard (`cb_output_rm`). Partial residency (`K < B`, head chunks re-tilized from the shard, output written tail-first) only when no fully-resident chunk fits.
    - Chunk = the per-core block: `h` is the coarsest divisor of the shard's tile rows whose fully-resident footprint fits, i.e. the whole shard on every model geometry.
    - Combine per image grid column: gather to the column root, stats back down in `ColumnMajor` mcast groups, `gy / N` shards per image when `N > 1`.
    - Program cache: the shard-backed CB addresses are patched on a cache hit (fresh-address regression tests pass).
  - **Variance precision fix** (shifted moments, no extra DRAM traffic; the DRAM path shares it):
    - Per-group pilot `k = x[image row 0, first channel of the group]`, read by the reader in chunk 0's barrier (bf16, exact).
    - Pass 1 accumulates `S1' = Σ(x − k)` from an exact `−n·k` seed (CB 23, `UnpackToDestFp32`) and `T2 = Σ x·(x − k)`.
    - The root finalize adds `k` back per group lane (CB 24): `mean = mean' + k`, `var = T2/n − mean'·mean`. Cancellation now scales with `|mean|·std`, not `mean²`.
  - **Perf work to keep the 9 DRAM cases flat after the precision fix:**
    - The two-op per-chunk chains (pass-1 `x·(x−k)`, pass-2 `a·x − b'`) are blocked over the DEST window (`IterationShape::…block_size(DEST_AUTO_LIMIT)`, `PerBlockSize` pack): one FPU init per element per window, not per tile.
    - Narrow slices fill that window too: the pilot rows are replicated `m = pilot_repeat(W, h)` times (pass 1 walks `m` tile rows as one row), and at `W = 1` pass 2 walks the chunk as `tiles(rows)` with `Scalar` a / b'.
    - Accumulators alias the channel CBs (groups E, C) to pay for the pilot / seed / shift CBs.
  - **Kernel-config ring under `--dev`.** The compute kernel is built per role (`IS_ROOT` CT arg: the member binary carries no combine/finalize), and `no-ipa-cp-clone` applies only in watcher / LLK-assert builds, so production keeps the clones.
  - `l1_ledger.md` was updated: new CB rows (1, 22, 23, 24), alias groups, the block-sharded footprint section, the `m·W` pilot, recomputed worked examples, and DM budget (0 DRAM crossings for sharded input/output).
- Accuracy achieved:
  - 114 block-sharded golden cells: PCC ≥ 0.999996, relative RMS ≤ 0.00300 (median 0.00245).
  - 7 DRAM golden picks: PCC ≥ 0.999996, relative RMS ≤ 0.00268.
  - Sharded TILE loose cases: `unet_tile_1024x1280` PCC 0.999998 / RMS 0.00203; `unet_tile_4096x640_4x8` 0.999998 / 0.00198; `vae_tile_16384x512` 0.999998 / 0.00192.
  - `test_sharded_offset_heavy`: mean 5 PCC 0.999991 / RMS 0.00436; mean 10 PCC 0.999971 / RMS 0.00800 (limit 0.02).
  - Precision baseline (DRAM): zero-mean shapes RMS 0.00177–0.00225 (Phase 0: 0.0030–0.0032); `offset_mean10` RMS 0.00770 (Phase 0: 0.0273), now passing without its xfail mark.
- Golden test progress: targeted slice (`--dev` and plain): 125 passed, 335 invalid-skipped, 186 xfail-expected, 0 failed. That covers all 114 `BLOCK_SHARDED, shard_fit=even, in_place=False` cells (including the 3 sharded TILE loose cases), the 4 R1 regression tests, and the DRAM picks. `verify_supported` on that run: 121 `supported_pass`, 186 `xfail_expected`, 335 `invalid_skipped`, 0 in every loud category.
- Issues encountered:
  - Under `--dev`, the RM root compute binary overflowed the 70 688 B kernel-config ring by 960–3 504 B. Fixed by the per-role compute split plus the debug-only `no-ipa-cp-clone`.
  - The first precision-fix version cost +1–10% on the DRAM perf cases. Chain blocking plus pilot replication brought all 9 back to −3.1%…+0.9% of Phase 0.
  - `flux1_dram_1048576x64` (W = 1) is pass-1 compute-bound on the extra `Sub` + DEST→srcB move. Before the `W = 1` normalize walk it was +3.8%; now it is +0.9% (+1.2% vs the same-session Phase 0 rerun), inside the noise band.
  - The non-focus `mid_*` DRAM bench shapes (latency-bound, ~50–120 µs) are +2.9…+10.6% vs Phase 0 (`mid_4096x640_tile` 46 105 → 50 996 ns). This is the fixed cost of the pilot fetch, seed, and root-side shift on short ops. It is recorded, not addressed: these shapes are not in the DRAM perf contract.
- Tests added:
  - `test_groupnorm_sc_N_1_HW_C_b_sharded.py`: 11 sharded out-of-place cases (TILE / RM, k = 1 / 2 / 4 narrow RM slices, N > 1, the 8×8 / 4×8 / 4×4 grids) plus 2 RM partial-residency cases.
  - `_op_bench.py`: `test_bench_sharded` (3 sharded TILE perf cases + 7 RM out-of-place U-Net / refiner shapes) and the `GN_BENCH_CHUNK` chunk-size knob.
  - `test_groupnorm_sc_N_1_HW_C_b_precision_baseline.py`: the `offset_mean10` xfail mark was removed.
  - Probes `probe_009`–`probe_013`.

### Device kernel duration — Refinement 1 (Blackhole, DEVICE KERNEL DURATION [ns], one fresh-cache profiled run per case, bf16 γ+β)

Sharded TILE PERF FOCUS cases (out of place), against native `ttnn.group_norm` (`MEASURED_NS_BLACKHOLE`):

| Case | R1 (ns) | Native (ns) | vs native |
|---|---|---|---|
| unet_tile_1024x1280 (8×8) | 24 618 | 39 669 | 1.61× faster |
| unet_tile_4096x640_4x8 (4×8) | 45 990 | 136 060 | 2.96× faster |
| vae_tile_16384x512 (8×8) | 56 141 | 143 179 | 2.55× faster |

Sharded RM out of place (8×8; the RM PERF FOCUS cases are in place and land with Refinement 2):
unet 1024x1280 26 453 · unet 16384x320 52 861 · unet 4096x320 26 472 · unet 4096x640 35 241 · unet 1024x640 22 444 · refiner 256x1536 21 413 · refiner 16384x384 51 641.

DRAM PERF FOCUS cases. "Phase 0" is the changelog value above; "Phase 0 rerun" is the Phase 0 kernel re-profiled in this session:

| Case | Phase 0 (ns) | Phase 0 rerun (ns) | R1 (ns) | vs Phase 0 | vs rerun |
|---|---|---|---|---|---|
| vae_dram_1048576x128 | 1 972 728 | 1 977 324 | 1 950 353 | −1.1% | −1.4% |
| vae_dram_65536x512 | 436 297 | 436 915 | 434 069 | −0.5% | −0.7% |
| vae_dram_262144x256 | 914 726 | 907 628 | 900 630 | −1.5% | −0.8% |
| vae_dram_262144x512 | 2 006 876 | 2 010 157 | 1 945 500 | −3.1% | −3.2% |
| vae_dram_1048576x256 | 4 213 458 | 4 220 485 | 4 141 145 | −1.7% | −1.9% |
| refiner_dram_16384x1152 | 242 230 | 241 850 | 240 481 | −0.7% | −0.6% |
| flux1_dram_16384x256 | 61 871 | 61 491 | 60 809 | −1.7% | −1.1% |
| flux1_dram_1048576x64 | 863 864 | 861 453 | 871 479 | +0.9% | +1.2% |
| mochi_dram_8x407040x128 | 6 382 888 | 6 363 399 | 6 344 183 | −0.6% | −0.3% |

## Refinement 2 — in_place=True on block-sharded input (both layouts), including the 16384x640 / 16384x960 L1 fit
- Date: 2026-11-25
- What was done:
  - **Registry.** `SUPPORTED["in_place"]` = `[False, True]`. `True` on an interleaved input stays a `ValueError` in `validate()` (INVALID in `feature_spec`); ragged shards stay unsupported (R3).
  - **Aliasing (knob-turn, no new CB, no kernel change).** In place, the entry point returns the input tensor itself. `cb_output_tiles` (TILE) / `cb_output_rm` (RM) are `cb_descriptor_from_sharded_tensor(input)`, the same bytes as the input ring / `cb_input_rm`. `io_tensors` lists the input once, as the output (`[γ, β, input]`), so the cache-hit address patching sees the standard in-place alias. Write ordering needs nothing extra:
    - TILE pass 2 packs tile i over the tile i it just unpacked.
    - RM pass 2 untilizes a chunk only over rows already tilized: resident chunks in pass 1, re-read chunks by the `tilize_chunk` just before.
    - The column members read the pilot from the root's shard at step 0, before their gather, so before the root's pass 2.
  - **L1 fit.** `_plan_sharded` no longer subtracts an output shard from the budget in place (`+sh·sw·2 B`). RM residency is now solved as `K = B` against the stream-free footprint (`K_full`), and against the footprint with the stream slot (`K_part`) only when `K_full < B`. R1 solved both against the streaming footprint, so a shard whose whole tilized copy fit with less than one chunk to spare was planned one chunk short and re-tilized it in pass 2 for the same ring bytes.
    - 16384×640: `h = 32, K = B = 1` (was `K = 0`).
    - 16384×960: `h = 2, K = B = 8` (was `K = 7`).
    - Both have about 5.6 KB of margin (`l1_ledger.md` "In place"). Of the 154 sharded RM plans over INPUTS and the perf shapes, only 11 change, none of them an R1 golden cell.
  - `l1_ledger.md` (in-place section, worked fits, deviations, DM budget) and `op_design.md` (the regime row is built, with the realization) were updated, and a one-line in-place note was added to the compute kernel.
- Accuracy achieved:
  - 9 RM in-place perf cases at `PERF_TOLERANCE`: PCC 0.999997–0.999998, relative RMS 0.00186–0.00224 (`unet_rm_ip_16384x640` 0.999998 / 0.00199; `unet_rm_ip_16384x960` 0.999998 / 0.00207).
  - TILE perf shapes run in place: PCC 0.999998, RMS 0.00183–0.00196.
  - `test_sharded_offset_heavy[inplace]`: mean 5 PCC 0.999991 / RMS 0.00425; mean 10 PCC 0.999968 / RMS 0.00802 (limit 0.02).
- Golden test progress: targeted slice (plain mode): **250 passed, 0 failed**, made up of:
  - all 120 `BLOCK_SHARDED, shard_fit=even, in_place=True` cells (111 `test_op` + the 9 RM in-place loose cases);
  - all 114 R1 out-of-place sharded cells;
  - 8 DRAM picks;
  - the 8 sharded regression tests, `[oop|inplace]` × `offset_heavy` / `program_cache_fresh_address`.
  The unit directory passes 77/77 both plain and under `--dev`.
- Issues encountered:
  - None in the kernels: the in-place path passed on its first run.
  - The planner's off-by-one-chunk RM residency (above) was found from the in-place plans and fixed.
  - Probe pitfall: the returned tensor aliases the input, so a plain `x.deallocate()` while `y` is alive frees nothing, and later plans see a shrunken budget. The bench frees the result with `ttnn.deallocate(out, force=True)`.
  - Finding for R10 (DRAM, out of scope here): the DRAM planner has the same accounting. `K_max` is solved with the `D_IN·T` stream depth even when `K = B` would need none. `flux1_dram_1048576x64` (K = 17, B = 19, D_IN = 2) would be fully resident (ρ 0.11 → 0) for the same ring bytes.
- Tests added:
  - `test_groupnorm_sc_N_1_HW_C_b_sharded.py`: `test_sharded_inplace` (the 11 R1 geometries in place) and `test_sharded_rm_partial_residency_inplace` (K = 0 / 1: tail written over pass-1 rows, head re-tilized just before overwrite).
  - `_op_bench.py`: `test_bench_sharded_inplace` (9 RM perf cases + 3 TILE perf shapes in place).

### Device kernel duration — Refinement 2 (Blackhole, DEVICE KERNEL DURATION [ns], one fresh-cache profiled run per case, bf16 γ+β, RM, 8×8, in place)

| Case | R2 (ns) | Native (ns) | vs native |
|---|---|---|---|
| unet_rm_ip_1024x1280 | 25 791 | 41 026 | 1.59× faster |
| unet_rm_ip_16384x320 | 52 627 | 256 041 | 4.87× faster |
| unet_rm_ip_4096x320 | 26 405 | 81 969 | 3.10× faster |
| unet_rm_ip_4096x640 | 34 813 | 88 043 | 2.53× faster |
| unet_rm_ip_1024x640 | 22 048 | 40 638 | 1.84× faster |
| refiner_rm_ip_256x1536 | 21 371 | 30 251 | 1.42× faster |
| refiner_rm_ip_16384x384 | 52 566 | 253 781 | 4.83× faster |
| unet_rm_ip_16384x640 | 86 893 (91 450 with R1's `K = 0`, −5.0%) | 320 169 | 3.68× faster |
| unet_rm_ip_16384x960 | 166 190 (166 449 with `K = 7`) | 345 006 | 2.08× faster |

These are R8's "before" numbers. The TILE perf shapes run in place track their R1 out-of-place times: 24 199 / 45 961 / 56 065 ns vs 24 618 / 45 990 / 56 141.

16384×960 costs about 0.69 µs per tilized tile per core (166 190 ns / 240 tiles), against 0.54 µs for 16384×640 (86 893 ns / 160 tiles). It is the widest slice (`W = 15`, `k = 4` sticks per virtual row) and the smallest chunk (`h = 2`, eight chunks), because the Float32 x·(x−k) intermediate (`4 KB·T`) is what limits `h`. That is a lead for R8, not measured further here.

## Perf 1 — perf tournament round 1: six ideas fanned out, six graduated (single unified paths; no measurement-earned carve-outs)
- Date: 2026-11-26
- Contract (fixed, never a lever): bf16, HiFi3, `fp32_dest_acc_en=True`, `math_approx_mode=False`, `dst_full_sync_en=False`. `SUPPORTED` untouched.
- Focus: the 21 `attention: PERF FOCUS` LOOSE_CASES (9 RM sharded in place, 3 TILE sharded out of place, 9 DRAM TILE), γ+β bf16 RM, `PERF_TOLERANCE` (0.999, 0.01). Only cells larger than 32 tiles were measured.
- Measurement: Blackhole, `DEVICE KERNEL DURATION [ns]`, one fresh-cache profiled run per case (`_op_bench.py` via `run_safe_pytest.sh --profile --no-py-zones`); noise band ±2–3%.

### Step 1 — measured breakdown

**Instrumentation (permanent).** `MaybeDeviceZoneScope` stage zones on all three kernels (reader `r_block`, `r_map_setup`, `r_shift_rows`, `r_ones_shift_affine`; compute `c_wait_map`, `c_wait_pilot`, `c_pass1`, `c_collapse`, `c_wait_gather`, `c_finalize`, `c_wait_stats`, `c_wait_affine`, `c_expand`, `c_pass2`; writer `w_wait_partials`, `w_gather_send`, `w_wait_gather`, `w_wait_finalize`, `w_mcast_send`, `w_mcast_recv`, `w_output`, `w_final_barrier`). Waits have their own zones. Per-chunk zones (`c_p1_moments`, `c_p2_normalize`, `r_issue`, `r_barrier`, ...) are opt-in via `TUNING_OVERRIDES["chunk_zones"]`, and are compiled in only when every core's chunk count fits the per-RISC marker budget. Tools: `perf_experiments/_tools/zone_table.py`, `zone_breakdown.py`. Ablation kit: `perf_experiments/_ablation/` (pre-Perf-1 kernels, now stale; see its README).

**Sharded focus — cumulative peel (ns).** Compute stubbed via the helper switches (`CKL_ELTWISE_CHAIN_SKIP_COMPUTE`, matmul `SKIP_COMPUTE`) plus hand stubs. The prologue is the reader's step 0; the peer is the gather + semaphore + stats mcast.

| case | full | −compute | −compute −prologue | −compute −prologue −peer | −prologue only | −peer only |
|---|---:|---:|---:|---:|---:|---:|
| rm_ip 1024x1280 | 27,088 | 12,393 | 6,629 | 6,645 | 21,509 | 26,259 |
| rm_ip 16384x320 | 54,035 | 14,297 | 7,821 | 7,801 | 47,671 | 53,338 |
| rm_ip 4096x320 | 27,843 | 13,719 | 7,284 | 7,300 | 21,689 | 27,159 |
| rm_ip 4096x640 | 36,091 | 13,388 | 7,415 | 7,390 | 30,180 | 35,397 |
| rm_ip 1024x640 | 23,317 | 13,241 | 7,228 | 7,264 | 17,764 | 22,713 |
| rf_ip 256x1536 | 22,690 | 13,786 | 7,096 | 7,107 | 16,182 | 21,962 |
| rf_ip 16384x384 | 53,179 | 11,038 | 7,377 | 6,767 | 48,882 | 52,676 |
| rm_ip 16384x640 | 88,473 | 15,677 | 10,947 | 10,976 | 82,301 | 87,546 |
| rm_ip 16384x960 | 168,318 | 35,429 | 18,231 | 18,424 | 151,534 | 168,115 |
| tile_ip 1024x1280 | 25,474 | 12,527 | 6,504 | 6,518 | 19,575 | 24,789 |
| tile_ip 4096x640 4x8 | 46,279 | 13,027 | 7,093 | 7,089 | 39,960 | 45,528 |
| tile_ip 16384x512 | 55,170 | 9,413 | 6,716 | 6,261 | 52,047 | 54,066 |

**DRAM focus — cumulative peel (ns).**

| case | full | −compute | −compute −DM | floor (−compute −DM −prologue −peer) | −DM only | −peer only | effective GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| vae 1048576x128 | 1,945,748 | 1,874,920 | 43,951 | 37,636 | 461,896 | 1,821,661 | 356 |
| vae 65536x512 | 435,142 | 410,081 | 24,549 | 11,764 | 147,477 | 405,302 | 309 |
| vae 262144x256 | 901,421 | 865,437 | 24,819 | 16,556 | 242,630 | 841,147 | 321 |
| vae 262144x512 | 1,945,706 | 1,901,666 | 46,164 | 33,166 | 506,193 | 1,827,245 | 363 |
| vae 1048576x256 | 4,169,097 | 4,001,159 | 61,961 | 55,047 | 919,648 | 3,892,562 | 359 |
| refiner 16384x1152 | 243,059 | 223,846 | 25,005 | 11,742 | 107,781 | 238,052 | 311 |
| flux1 16384x256 | 61,536 | 51,890 | 9,770 | 5,736 | 28,708 | 54,491 | 273 |
| flux1 1048576x64 | 871,665 | 831,144 | 20,618 | 15,858 | 218,123 | 791,719 | 324 |
| mochi 8x407040x128 | 6,349,439 | 6,157,242 | 99,459 | 88,487 | 1,507,506 | 6,188,514 | 377 |

**Ranked bottlenecks (roofline-gated).**
1. **Sharded: compute-bound.** With compute stubbed the op runs at 17–51% of full. The cross-core peer costs ≤ 1.1 µs (−peer only), so gather and mcast schemes were not floated. The small shapes (1024x1280-class, ~26 µs) carried a fixed serial chain:
   - reader prologue ≈ 5.5 µs, serial ahead of pass 1 on every case (16.8 µs on 16384x960);
   - collapse ≈ 2.4 µs;
   - combine + finalize ≈ 3.7 µs;
   - expand ≈ 6.2 µs, including the stats wait.

   On the large shapes it was the pass-1 reduce (16384x960: 58.6 µs of 168) and the per-tile chains in both passes. No NoC payload moves per chunk, so the DM ceiling does not bind.
2. **DRAM: data-movement-bound.** Stubbing compute leaves 92–97% of full; compute alone is 22–44%. The big cases run at 70–74% of the BH 512 GB/s DRAM spec. Per-core pass-1 and pass-2 times spread up to 3× by grid position (NoC0 reads slow the top rows, NoC1 writes the right columns). The peer costs 3–11%, because units are phase-locked behind their slowest member. The prologue is hidden (≤ 1.8%).
3. **Ceiling gate (`/perf-ceiling-dm`).** The `noc_estimate` CLI is not built in this checkout, so the bracket was read from `noc_latencies.yaml`:
   - BH DRAM_INTERLEAVED 2 KB: `ONE_FROM_ALL` 77.7 GB/s per core; `ALL_FROM_ALL` 3.6 GB/s per core (NOC0); `ALL_TO_ALL` 2.8 GB/s (NOC1) and 1.6 GB/s (NOC0).
   - DRAM floor at 92% of 512 GB/s, before Perf 1: vae 1048576x256 3.18 ms against 4.17 ms measured (1.31×); mochi 5.08 against 6.35 ms (1.25×); vae 262144x256 0.615 against 0.901 ms (1.47×). That headroom justified a DRAM stream idea.

### Step 2 — portfolio

| idea | stage targeted | why (measured) | overlaps |
|---|---|---|---|
| `prologue_off_critical_path` | reader step-0 prologue (pilot read, zero-fills, map one-hots, shift rows) + the compute map wait | −prologue only: −3.1 to −16.8 µs sharded | the pilot-row mcast (reuse-shared broadcast) folded in as an option |
| `expand_fused` | expand: 2 transposed map matmuls + 3 W-tile chains (a, m, b') | ≈ 6 µs fixed on small sharded shapes | stats layout shared with `combine_critical_path` |
| `combine_critical_path` | collapse + root finalize | ≈ 6 µs fixed | ↑ |
| `pass1_moments_accumulate` | pass-1 reduce: accumulate first, reduce once; drop the per-chunk reload and the x² CB | 16384x960 pass-1 reduce 58.6 µs | the pass-1 chain is shared with `per_tile_pass_chains` |
| `per_tile_pass_chains` | pass-1 x·(x−k) chain, pass-2 normalize, tilize / untilize | large sharded shapes | ↑ |
| `dram_stream_balance` | DRAM reader/writer streams, residency, NoC position imbalance | 1.25–1.47× above the DRAM floor; 3× per-core spread | — |

Not floated:
- **Cross-core transfer schemes** (tree all-reduce, mcast topology): the no-peer ablation bounds them at ≤ 1.1 µs sharded.
- **Reuse-shared γ/β mcast:** γ/β is read in the combine's stall shadow. The one shared operand on the critical path, the pilot row, was floated as an option inside the prologue idea (cand3).

### Step 3 — verdicts (isolated `perf-part-optimizer` benches; ns before → after on that idea's focus shapes)

| idea | verdict | graduated option | focus shapes (base → option) | domain sweep | precision |
|---|---|---|---|---|---|
| `prologue_off_critical_path` | **WIN** | cand4: the prologue runs after the last pass-1 credit push (sharded) or with chunk 0 (DRAM); the map is built after the pass-1 pushes; the compute map wait moves after pass 1 | rm_ip 1024x1280 27,093 → 21,673; rm_ip 16384x960 167,853 → 151,061; vae_tile 16384x512 54,567 → 51,875 | sharded −4.9 to −28.4%; DRAM flat to −3%; Cg = 3 −8 to −13% | PCC 0.999998, unchanged |
| `expand_fused` | **WIN** | v3s: raw-LLK `expand_matmul`, a = rstd × gmap and b' = [mean·rstd, e] × [gmap; bmap] | rm_ip 1024x1280 27,088 → 25,433; vae_tile 54,562 → 52,940; rm_ip 16384x960 167,853 → 162,793. Expand stage alone: 2.76 / 1.46 / 5.86 → 1.29 / 0.68 / 2.44 µs | −1 to −8% on all sharded, DRAM and affine modes | PCC 0.999998; rel-RMS equal or lower |
| `combine_critical_path` | **WIN** | vpub + fin: the publish becomes a credit hand-off; the finalize runs inside the sender-sum matmul's DEST window | rm_ip 1024x1280 27,125 → 24,267; tile 1024x1280 25,202 → 21,895; rm_ip 16384x960 167,876 → 164,696. Root finalize 3.1 → 1.27 µs; collapse 2.4 → 0.79 µs | −1.2 to −15.8%; 16384x640 +1.8% (noise band; root-binary code layout, see below) | bit-identical |
| `pass1_moments_accumulate` | **WIN** | C2: S += x−k and Q += x·(x−k) elementwise in fp32 DEST, then one SFPU column collapse | rm_ip 16384x960 167,844 → 107,009; vae_tile 54,580 → 43,221; rm_ip 16384x320 53,753 → 42,811 | sharded −14 to −36%; DRAM flat (−1.7 / −2.1%) | equal or better, including offset-heavy data (mean 50 rms 0.048 vs 0.050) |
| `per_tile_pass_chains` | **WIN** (pass 2) / NULL (pass 1) | n1u: normalize as DEST = (−1)·b' + x·a (two accumulating ELWMULs); sharded RM uses `pack_untilize_dest` straight into the output shard | vae_tile 54,564 → 49,704; rm_ip 16384x320 53,756 → 49,695; rm_ip 16384x960 167,860 → 153,853 | −2.6 to −11.5%; DRAM flat | rel-RMS lower everywhere |
| `dram_stream_balance` | **WIN** | NoC pairing split by grid column (odd-column non-root DRAM cores read on NOC1, write on NOC0) + `K = B` solved against the stream-free footprint | vae_dram 262144x256 904,632 → 794,458; vae_dram 1048576x256 4,132,801 → 3,817,369; flux1 1048576x64 868,416 → 729,769 | DRAM −6.5 to −16%; refiner / flux1 16k flat; sharded program unchanged | identical |

**Null, rejected and blocked options (completed investigations, not graduated):**
- **cand3**, pilot-row multicast from the unit root (the reuse-shared-broadcast idea). FLAT: −154 / −18 / +124 ns. It evens out the farthest member's ~450 ns remote-read lag but delays the root, and it adds a semaphore protocol.
- **v3a**, expand with helpers only. 25,973 / 53,556 / 164,320: a win, but slower than v3s; it would have been the fallback.
- **v3b** (helper 4+tail split) and **v3d** (out-of-line helper). BLOCKED: the `--dev` kernel-config ring overflowed by 496 B and 48 B (71,184 > 70,688).
- **v3c**, one runtime-shaped helper call. 25,607 / 53,726 / 163,154: slower than v3s, and +1.0–5.5 KB of root text.
- **"all"**, both collapse tiles in one raw-LLK matmul. NULL: the collapse is unpack-bound (≈ 680 ns for 2W tile pairs). The combine window moves only ±0.4 µs, tile 4x8 is 0.87 µs worse, and it adds +2.4 KB of `--dev` text.
- **C1**, Q += x·x + x·(−k) with stock `mul_tiles`. Faster than C2 (16384x960 103,687) but REJECTED on precision. At HiFi3, x·x drops the srcA_lo × srcB_lo partial product, a one-sided ~1e-4 bias that cancels catastrophically at |mean| ≫ std (mean 50 rms 0.085 vs 0.050 baseline). Trading precision for speed is outside the contract.
- **p1l**, pass-1 x−k through a bf16 L1 ring. NULL / REJECTED: 16384x960 +29%, and less precise (rel-RMS 0.00239). Tilize was not attempted.
- **dyn_split**, `DM_DYNAMIC_NOC`, alternating NoC per tile. BLOCKED on two counts:
  - PCC 0.999995 where bit-identical output is expected (a correctness failure);
  - the Mcast sender does not compile under `DM_DYNAMIC_NOC` (TXN_ID `async_writes_flushed` hits a `static_assert` at `noc.h:789`: a helper capability gap).

### Step 4 — aggregation
All six ideas had a winning option, and none conflicted once ordered.
- `combine_critical_path`'s **vpub** became moot: C2 had already deleted the publish copy, so only **fin** graduated.
- `per_tile_pass_chains`' pass-1 option was null and is superseded by C2. Only its pass-2 **n1u** graduated.
- `expand_fused` and **fin** both touch the root's stats. They were merged: FinalizeStats feeds `[rstd, mean]`, and the expand consumes it.
- CB-slot collisions in the subagents' drafts were resolved at merge:
  - n1u's −1 tile moved from CB 25 to the freed CB 6;
  - v3s's `[mean·rstd, e]` moved from a group-B view (group B was deleted by C2) to CB 18, a view on the local partials (group D).

### Step 5 — graduations (in commit order; each one a single unified path that deletes the code it replaced)

| # | commit | graduation | code deleted | domain / carve-outs |
|---|---|---|---|---|
| 0 | `c3dd69fa8f2` | permanent stage zones, opt-in per-chunk zones, ablation kit | — | — |
| 1 | `4f506ea5b0e` | DRAM NoC pairing split across grid columns + `K = B` against the stream-free footprint | the single reader / writer NoC configuration | Every interleaved cell (TILE + RM). **Feasibility carve-outs**, neither measurement-earned: sharded paths stream no DRAM, so the pairing has nothing to balance there; unit roots keep (read NOC0, write NOC1), because the one `ttnn.Mcast` channel spans every core with a single configured NoC and the root is its sender. |
| 2 | `11569808800` | prologue after the last pass-1 push / chunk 0; map built after the pass-1 pushes | the step-0 prologue | everywhere, no carve-outs |
| 3 | `768918c7730` | pass-1 moments as fp32 DEST accumulators + one SFPU column collapse per window; sharded `K == B` in one window | `cb_x_squared` (4), reduce scaler (6) + its reader prep, `−n·k` seed (23) + its generation, pilot replication, the publish copy, the group-B drain reserve, 2 CT args | everywhere, no carve-outs |
| 4 | `2c04f8e9ddb` | root finalize as a raw-sfpi PostComputeFn in the sender-sum matmul's DEST window | `cb_group_sums` (18) round trip, the 3-copy chain finalize | everywhere (`static_assert(DST_ACCUM_MODE)`; fp32 DEST is the fixed contract) |
| 5 | `cd3fa42e9b2` | normalize as two accumulating multiplies; sharded RM pack-untilizes into the shard | the Mul + DestReuse-Sub chain, the `untilize<>` pass, the sharded-RM output tile ring | everywhere, no carve-outs |
| 6 | `f5942fd11a1` | expand as two matmuls against reader-built γ / β maps | 2 map matmuls + 3 W-tile chains, `cb_chan_shift` (11), `cb_beta_rows` (14), the `EXP_SBW` CT arg | everywhere: γ uses the non-transposed `[group lane × channel]` gamma map; no affine uses the plain map, transposed. No carve-outs. |

Graduation 6 changed the subagent's design. The first merge of v3s regressed the short shapes: `refiner_rm_ip_256x1536` 10,083 → 12,812 ns and `unet_rm_ip_1024x640` 11,681 → 12,828. That path builds a transposed `[channel × group lane]` gamma map with W·32 halfword load/store pairs, about 2 µs more reader time (`r_ones_shift_affine` 5.2 → 7.2 µs). With one tile row per core, the ~2 µs combine round trip no longer hid it (`c_wait_affine` 3.2 µs). Re-laying the gamma map as `[group lane × channel]` fixed it:
- the matmul runs without the in1 transpose;
- adjacent channels of a group become adjacent halfwords, so both maps are word copies (the beta map becomes exactly the old row-0 β rows);
- the affine zone fell to 3.7–4.4 µs, below the pre-v3s 5.2 µs;
- 256x1536 → 9,830 and 1024x640 → 10,270. Every sharded case is faster than graduation 5.

There is a halfword fallback for an odd channels-per-group count; the guard set covers it (Cg = 3).

Every graduation was gated on golden loose + regression (64/64) and re-benched on the 12 in-place, 10 out-of-place and 13 DRAM cases. No measured regression appeared at any step, so none earned a carve-out. Each experiment dir was committed with its graduation.

Cumulative progression (ns; sharded cases are unchanged by graduation 1):

| case | base | +1 DSB | +2 prologue | +3 moments | +4 finalize | +5 normalize | +6 expand |
|---|---:|---:|---:|---:|---:|---:|---:|
| unet_rm_ip_1024x1280 | 27,088 | — | 21,652 | 17,050 | 15,056 | 14,567 | 13,195 |
| refiner_rm_ip_256x1536 | 22,690 | — | 16,426 | 12,761 | 10,599 | 10,083 | 9,830 |
| unet_rm_ip_16384x960 | 168,318 | — | 152,170 | 90,126 | 87,476 | 71,427 | 69,001 |
| vae_tile_16384x512 (oop) | 54,979 | — | 52,270 | 40,320 | 38,769 | 33,944 | 33,150 |
| vae_dram_1048576x256 | 4,151,711 | 3,810,690 | 3,799,577 | 3,772,348 | 3,765,855 | 3,763,520 | 3,753,476 |
| flux1_dram_1048576x64 | 867,622 | 732,831 | 733,278 | 727,327 | 725,955 | 727,524 | 726,388 |

### Whole op, before (pre-Perf 1, `737d534db7c`) → after (`f5942fd11a1`)

**Sharded in place (9 RM PERF FOCUS + 3 TILE)**

| case | before ns | after ns | Δ | speedup |
|---|---:|---:|---:|---:|
| `unet_rm_ip_1024x1280` | 27,088 | 13,195 | -51.3% | 2.05× |
| `unet_rm_ip_16384x320` | 54,035 | 30,521 | -43.5% | 1.77× |
| `unet_rm_ip_4096x320` | 27,843 | 13,204 | -52.6% | 2.11× |
| `unet_rm_ip_4096x640` | 36,091 | 18,850 | -47.8% | 1.91× |
| `unet_rm_ip_1024x640` | 23,317 | 10,270 | -56.0% | 2.27× |
| `refiner_rm_ip_256x1536` | 22,690 | 9,830 | -56.7% | 2.31× |
| `refiner_rm_ip_16384x384` | 53,179 | 29,944 | -43.7% | 1.78× |
| `unet_rm_ip_16384x640` | 88,473 | 53,750 | -39.2% | 1.65× |
| `unet_rm_ip_16384x960` | 168,318 | 69,001 | -59.0% | 2.44× |
| `unet_tile_ip_1024x1280` | 25,474 | 11,683 | -54.1% | 2.18× |
| `unet_tile_ip_4096x640_4x8` | 46,279 | 24,694 | -46.6% | 1.87× |
| `vae_tile_ip_16384x512` | 55,170 | 33,224 | -39.8% | 1.66× |

**Sharded out of place (3 TILE PERF FOCUS + 7 RM)**

| case | before ns | after ns | Δ | speedup |
|---|---:|---:|---:|---:|
| `unet_tile_1024x1280` | 25,214 | 11,556 | -54.2% | 2.18× |
| `unet_tile_4096x640_4x8` | 46,298 | 24,661 | -46.7% | 1.88× |
| `vae_tile_16384x512` | 54,979 | 33,150 | -39.7% | 1.66× |
| `unet_rm_oop_1024x1280` | 27,441 | 13,262 | -51.7% | 2.07× |
| `unet_rm_oop_16384x320` | 54,004 | 30,533 | -43.5% | 1.77× |
| `unet_rm_oop_4096x320` | 27,842 | 13,189 | -52.6% | 2.11× |
| `unet_rm_oop_4096x640` | 36,152 | 18,844 | -47.9% | 1.92× |
| `unet_rm_oop_1024x640` | 23,312 | 10,447 | -55.2% | 2.23× |
| `refiner_rm_oop_256x1536` | 22,721 | 9,879 | -56.5% | 2.30× |
| `refiner_rm_oop_16384x384` | 53,173 | 29,873 | -43.8% | 1.78× |

**DRAM interleaved (9 PERF FOCUS + 4 mid)**

| case | before ns | after ns | Δ | speedup |
|---|---:|---:|---:|---:|
| `vae_dram_1048576x128` | 1,942,128 | 1,760,855 | -9.3% | 1.10× |
| `vae_dram_65536x512` | 436,907 | 373,184 | -14.6% | 1.17× |
| `vae_dram_262144x256` | 905,545 | 780,927 | -13.8% | 1.16× |
| `vae_dram_262144x512` | 1,951,504 | 1,760,133 | -9.8% | 1.11× |
| `vae_dram_1048576x256` | 4,151,711 | 3,753,476 | -9.6% | 1.11× |
| `refiner_dram_16384x1152` | 242,528 | 215,514 | -11.1% | 1.13× |
| `flux1_dram_16384x256` | 61,565 | 54,310 | -11.8% | 1.13× |
| `flux1_dram_1048576x64` | 867,622 | 726,388 | -16.3% | 1.19× |
| `mochi_dram_8x407040x128` | 6,339,149 | 5,904,184 | -6.9% | 1.07× |
| `mid_4096x640_tile` | 51,404 | 38,202 | -25.7% | 1.35× |
| `mid_4096x640_rm` | 67,197 | 45,467 | -32.3% | 1.48× |
| `mid_16384x320_tile` | 86,514 | 67,646 | -21.8% | 1.28× |
| `mid_16384x320_rm` | 121,624 | 81,987 | -32.6% | 1.48× |
Against native `ttnn.group_norm` (R2's native column), the 9 RM in-place cases now run 3.11× (1024x1280), 8.39× (16384x320), 6.21× (4096x320), 4.67× (4096x640), 3.96× (1024x640), 3.08× (256x1536), 8.48× (16384x384), 5.96× (16384x640) and 5.00× (16384x960) faster.

DRAM against the 92% floor: vae 1048576x256 is 1.18× (was 1.31×); mochi is 1.16× (was 1.25×); vae 262144x256 is 1.27× (was 1.47×).

### Guard set (no regression)
One representative per kernel path × layout × placement for the affine paths the focus benches do not run, plus an odd channels-per-group case, which exercises the reader's halfword map path (`_op_bench.py::test_bench_guard`). The pre-Perf-1 op files were swapped in for the base column.

| guard case | base ns | after ns | Δ |
|---|---:|---:|---:|
| rm in place 1024x1280, γ only | 25,229 | 12,704 | −49.6% |
| rm in place 1024x1280, no affine | 25,244 | 12,676 | −49.8% |
| tile oop 1024x1280, γ only | 23,745 | 11,246 | −52.6% |
| tile oop 1024x1280, no affine | 23,113 | 10,977 | −52.5% |
| DRAM tile 4096x640, γ only | 49,547 | 37,645 | −24.0% |
| DRAM tile 4096x640, no affine | 49,487 | 37,706 | −23.8% |
| DRAM RM 4096x640, γ only | 65,276 | 44,903 | −31.2% |
| DRAM RM 4096x640, no affine | 63,969 | 45,047 | −29.6% |
| DRAM tile 16384x96, γ+β, Cg = 3 | 40,141 | 26,912 | −33.0% |

Every guard cell and every focus / mid cell is faster. No cell regressed, so no carve-out was earned by measurement.

### Correctness
- Full golden directory (`eval/golden_tests/groupnorm_sc_N_1_HW_C_b/`): **676 passed, 0 failed**. The 4,131 xfailed are the unsupported cells and the skips are invalid cells.
- Loose + regression: **64/64**, both plain and under `--dev` (watcher, LLK asserts). Every kernel binary fits the 70,688 B kernel-config ring with the asserts built in.
- Precision is equal or better throughout:
  - C2 keeps x·(x−k) on the FPU, exactly as before (C1's x·x bias was rejected);
  - n1u forms a·x − b' in fp32 DEST rather than truncating a·x to tf32 on a DEST → srcA move;
  - fin is bit-identical.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `reduce<SUM, REDUCE_COL, ReduceTile>` + `Accumulate`, `eltwise_chain` (BinaryFpu Sub → DestReuse Mul → PackTile), `mul_reuse_dest_tiles` | capability | No helper keeps several elementwise FPU accumulators resident in one DEST window over a strided walk and then collapses each column once with the SFPU. The reduce helper reloads and re-packs its accumulator on every call; the chain's DestAccumulation sums along a walk row, not down a column. `mul_reuse_dest_tiles` writes the product back to the same tile it moved into srcB, so it cannot MOVD2B from a scratch tile while the MOP's destination walks a different accumulator. | 167,844 (rm_ip 16384x960) / 54,580 (vae_tile 16384x512) | 107,009 / 43,221 | `groupnorm_sc_N_1_HW_C_b_compute.cpp:169` (`moments_rows` :236) |
| `eltwise_chain` finalize (CopyTile ×3, 8 SFPU ops, PackTile ×2) | capability | The chain has no vector scope: every SFPU op runs over all 32 vectors of a tile, although only the row-0 group needs computing. It also cannot run inside `matmul_block`'s DEST window as a PostComputeFn, so the sums had to round-trip through a CB (18). | 27,125 (rm_ip 1024x1280; root finalize zone 3.1 µs) | 25,603 (fin alone; zone 1.27 µs) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:465` (`FinalizeStats` :469) |
| `eltwise_chain` (BinaryFpu Mul + DestReuseBinary Sub) normalize | capability | No element accumulates a second FPU product into the DEST tile the first one wrote (ELWMUL accumulates natively). The DestReuse path moves DEST → srcA once per face (stalls), and truncates a·x to tf32 on the move. | 54,564 (vae_tile 16384x512) | 49,707 (n1b; same with n1u) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:290` (`normalize_chunk` :300) |
| `untilize<>` (helper) after PackTile → `cb_output_tiles` | capability | No untilize helper takes a block already in DEST (`pack_untilize_dest`). The helper re-unpacks the packed tiles from a CB, then datacopies and packs them again, and needs a tile ring for the round trip. | 161,601 (rm_ip 16384x960, n1b) | 153,853 (n1u) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:341` (`normalize_untilize_chunk` :358) |
| `matmul_block` (expand products only) | capability | It packs every subblock it computes, so a row of W = 5 / 7 / 10 / 15 tiles can only be tiled by the largest divisor of W that fits DEST (1 / 1 / 2 / 3). There is no partial-subblock pack, so an overlapped last subblock cannot publish only its new tiles. Under `fp32_dest_acc_en` it also re-issues `pack_reconfig_data_format` on every subblock. Splitting into helper calls (v3b, v3d) overflowed the `--dev` kernel-config ring. | 25,973 (rm_ip 1024x1280, v3a helpers only; expand 2.02 µs) | 25,433 (v3s; expand 1.29 µs) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:484` (`expand_matmul` :492) |
| no helper covers this (`l1_helpers` / `tile_column` family) | capability | No helper scatters a staged row segment into tile-face layout at a computed row: the gamma map's `gamma[c]` at row `g(c)`, the beta map's row 0. `fill_tile_column` writes all 32 rows of a column; `generate_index_tile` writes a ramp. Raw word stores, as before Perf 1 (the row-0 copies), now in the expand's map layout. | — (pre-Perf-1 code was raw too) | — | `groupnorm_sc_N_1_HW_C_b_reader.cpp:212` |

Every compute bypass site carries a kernel-head comment with the measured win that authorizes it.

### Notes for round 2 (re-run Step 1 first)
- **Where the time sits now** (`zone_table.py`, in place, p50 over cores):
  - `unet_rm_ip_1024x1280`, span 11.9 µs: pass 1 5.5, stats round trip (`c_wait_stats`) 1.9, root finalize 1.5, expand 2.7 (the math-thread zone; it still includes waiting on the unpacker), pass 2 2.6.
  - `unet_rm_ip_16384x960`, span 68.2 µs: pass 1 40.8, pass 2 21.8. The moments and normalize loops are now the targets on large shapes.
- **The ablation kit is stale** (pre-Perf-1 kernels). Rebuild it with hand stubs for `moments_rows`, `FinalizeStats`, `expand_matmul` and `normalize_*`: the chain and matmul helper switches no longer reach those sites.
- **Root-binary code layout.** The combine subagent saw root cores' pass 2 move by up to 1.2 µs with nops inserted after the finalize: an RM in-place root-vs-member `c_pass2` gap of ~18%. Compare root and non-root `c_pass2` before attributing a ±2–5% RM delta to a lever.
- **γ/β reads.** Every core of every unit reads its slice from the one RM γ/β page, a single DRAM bank. That read sits inside `r_ones_shift_affine` (3.7–4.4 µs) and is hidden by the combine today, but would surface if the combine shrinks further.
- **Helper gap to file.** The Mcast sender does not compile under `DM_DYNAMIC_NOC` (`noc.h:789` `static_assert`), which blocks dynamic-NoC DRAM splitting.

**Summary:** six ideas floated and measured; six graduated (each with its losing options recorded); 9 options null, rejected or blocked. On the focus shapes the op is 1.65–2.44× faster on sharded in place, 1.66–2.30× on sharded out of place, and 1.07–1.19× on DRAM (mid shapes 1.28–1.48×). No regression on the focus, mid or guard cells; golden green.
