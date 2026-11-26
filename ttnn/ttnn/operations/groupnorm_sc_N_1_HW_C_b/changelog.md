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

## Perf 2 — perf tournament round 2: six ideas fanned out, five graduated, one null (single unified paths; no measurement-earned carve-outs)
- Date: 2026-11-26
- Contract (fixed, never a lever): bf16, HiFi3, `fp32_dest_acc_en=True`, `math_approx_mode=False`, `dst_full_sync_en=False`. `SUPPORTED` untouched.
- Focus: the same 21 `attention: PERF FOCUS` LOOSE_CASES as Perf 1 (9 RM sharded in place, 3 TILE sharded out of place, 9 DRAM TILE), γ+β bf16 RM, `PERF_TOLERANCE`. Only cells larger than 32 tiles were measured; every measured cell is ≥ 384 tiles.
- Measurement: Blackhole, `DEVICE KERNEL DURATION [ns]`, one fresh-cache profiled run per case (`perf_experiments/_round2/bench.py` via `run_safe_pytest.sh --profile --no-py-zones`). Noise band ±2–3%: a delta inside it is reported as flat, never as a win.
- Baseline: `02a76783017` (the Perf 1 record; kernels as of `f5942fd11a1`).

### Step 1 — measured breakdown (re-run on the Perf-1 op)

**Instrumentation.** The Perf 1 permanent zones were reused. The ablation kit was rebuilt for the Perf-1 kernels with hand stubs (`perf_experiments/_round2/ablation/make_variants.py`):
- `P1` stubs `moments_rows`;
- `P2` stubs the normalize;
- `TZ` stubs the RM tilize;
- `CMB` stubs the collapse, finalize and expand;
- `PEER` stubs the gather, semaphores and stats mcast;
- `DM` stubs the DRAM payload reads and writes.

Every stub keeps its CB and semaphore scaffolding. The stages were peeled cumulatively, and each was also removed on its own.

**Sharded focus — cumulative peel (ns).**

| case | full | −P2 | −P2 −P1 | … −TZ | … −CMB | … −PEER (floor) | −P1 only | −P2 only | −TZ only | −CMB only | −PEER only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unet_rm_ip 1024x1280 | 13,188 | 10,075 | 8,513 | 8,432 | 7,244 | 7,270 | 11,581 | 10,087 | 12,239 | 11,010 | 12,139 |
| unet_rm_ip 16384x960 | 68,876 | 48,198 | 19,486 | 19,630 | 16,468 | 16,689 | 40,174 | 48,202 | 63,980 | 64,953 | 67,708 |
| unet_rm_ip 16384x320 | 30,469 | 19,529 | 9,459 | 9,492 | 8,037 | 8,035 | 20,401 | 19,529 | 28,404 | 28,281 | 29,344 |
| refiner_rm_ip 256x1536 | 9,936 | 9,220 | 9,160 | 9,024 | 7,880 | 7,913 | 9,899 | 9,090 | 9,964 | 8,708 | 9,981 |
| vae_tile 16384x512 (oop) | 33,166 | 22,869 | 6,690 | — | 4,951 | 4,969 | 16,923 | 22,882 | — | 31,872 | 31,942 |
| unet_tile 1024x1280 (oop) | 11,556 | 9,757 | 8,953 | — | 7,775 | 7,807 | 10,783 | 9,725 | — | 9,644 | 10,759 |

**DRAM focus — cumulative peel (ns).**

| case | full | −P1 −P2 | −DM | −DM −P1 −P2 | −DM −P1 −P2 −TZ −CMB −PEER |
|---|---:|---:|---:|---:|---:|
| vae 1048576x256 | 3,760,067 | 3,749,536 | 620,143 | 119,413 | 116,256 |
| vae 65536x512 | 373,796 | 360,063 | 98,481 | 32,361 | 28,716 |
| refiner 16384x1152 | 214,584 | 204,012 | 67,810 | 26,892 | 22,964 |
| flux1 16384x256 | 53,130 | 48,795 | 16,186 | 7,905 | 5,295 |
| mochi 8x407040x128 | 5,909,156 | 5,938,276 | 1,013,992 | 192,796 | 189,013 |

**Ranked bottlenecks (roofline-gated).**
1. **Large sharded: pass-1 moments, then the pass-2 normalize.** These are compute-bound; no NoC payload moves per chunk, so the DM ceiling does not bind.
   - P1, measured after P2 was peeled: 28.7 µs (42%) on 16384x960, 16.2 µs (49%) on vae_tile 16384x512, 10.1 µs (33%) on 16384x320.
   - P2: 20.7 µs (30%), 10.3 µs (31%) and 10.9 µs (36%) on the same three.
2. **Small sharded: the combine plus the peer.** CMB costs a fixed ~1.1–1.7 µs once the passes are peeled (3.2 µs on 16384x960), and 1.2–2.2 µs (12–17%) removed on its own. PEER costs ~0.8–1.0 µs on its own on the 1024x1280 cells (7–8%) and is hidden once the passes are gone.
3. **The reader is the hidden floor.** With every compute stage and the peer stubbed, the op still runs 5.0–16.7 µs. That is the reader: `r_block` 5.8–6.5 µs and `r_ones_shift_affine` 3.7–4.2 µs on the small shapes. The TILE `c_wait_pilot` of 1.2–1.4 µs sits on the critical path.
4. **TZ (RM tilize) is overlap-hidden, not free.** Removed on its own it costs 4.9 µs (−7.1%) on 16384x960. After the P1 + P2 peel it is ~0, because the reader floor then holds the wall.
5. **DRAM is DM-bound.** Stubbing DM leaves 16–32% of full (vae 1048576x256: 3.76 → 0.62 ms). Stubbing compute alone leaves 92–100%.
   - Throughput is 312–405 GB/s, against a ~396–448 GB/s ceiling (`/perf-ceiling-dm`, `noc_latencies.yaml`).
   - The headroom left is per-core imbalance: the slowest core runs 1.34× the p50 core's time, and the unit waits on its slowest member. It is not bandwidth.
   - DRAM compute ideas were therefore not floated; only a DM balance idea was.

### Step 2 — portfolio

| idea | stage targeted | why (measured) | overlaps |
|---|---|---|---|
| I1 `moments_fpu_stream` | pass-1 `moments_rows` (C2: ELWADD S, MOVD2B + ELWMUL Q per row block) | rank 1 | the RM tilize fusion would also rewrite pass 1 |
| I2 `normalize_bcast_srcb` | pass-2 normalize (n1u: (−1)·b′ + x·a, two HiFi3 ELWMULs per tile) | rank 1 | `cb_neg_one` is shared with the combine's finalize |
| I3 `rm_tilize_fusion` | RM tilize folded into pass 1 / pass 2 | rank 4 (overlap-hidden) | I1, I2 |
| I4 `combine_tail` | collapse → gather → finalize → stats mcast → expand, including an all-reduce option | rank 2 | I2 (`cb_neg_one`), I5 (ones tiles and the group-shift fill) |
| I5 `reader_prologue` | γ/β read issue, map and affine build, zero fills, pilot → shift rows | rank 3 | I4 |
| I6 `dram_dynamic_balance` | DRAM per-core imbalance: dynamic chunk stealing and residency, a reader/writer output split | rank 5 | the I2 normalize (output CB choice) |

Not floated:
- **Round-1 ideas** (re-floating them is barred).
- **Reuse-shared broadcast:** the pilot-row mcast was measured flat in Perf 1 (cand3). The γ/β single-bank read was folded into I5's early issue.
- **Precision knobs.**

### Step 3 — verdicts (isolated `perf-part-optimizer` benches; ns base → option)

**I1 `moments_fpu_stream` — WIN.** Graduated option: `fused_tile`.
- **Mechanism:**
  - one whole-tile srcA unpack per x tile feeds ELWADD D, ELWADD S and MOVD2B + ELWMUL Q;
  - the HiFi3 fidelity loop runs outside the 8 row blocks (an accumulating ELWMUL needs ≥ 5 distinct 8-row blocks, or it stalls on DEST every phase);
  - the sequence is replayed from the replay buffer (entries 0–15, addr-mod slots 0–5).

| case | base | `fused` | `fused_b1` | `fused_tile` |
|---|---:|---:|---:|---:|
| unet_rm_ip 16384x960 | 68,905 | 61,353 (−11.0%) | 61,383 | 50,556 (−26.6%) |
| vae_tile 16384x512 | 33,162 | 28,863 (−13.0%) | 28,839 | 23,014 (−30.5%) |
| unet_rm_ip 16384x320 | 30,498 | 27,854 (−8.7%) | 27,876 | 24,281 (−20.4%) |
| unet_rm_ip 1024x1280 | 13,193 | 12,491 (−5.3%) | 12,524 | 11,679 (−11.5%) |

`fused_tile` comes from a separate run, whose bases agree with this one within 0.1%.

- **Options:**
  - `fused` shares one srcA unpack across the three ops per 8-row block, with the fidelity loop inside each block.
  - `fused_b1` is `fused` with one block per MOP; it is the same as `fused`.
  - Probes on `fused`: `NOSTALL` (DEST stall removed, wrong output) 28,091 / 59,965 and `NOMATH` 19,758 / 43,323. They showed the per-phase DEST stall and the math floor, which motivated `fused_tile`.
- **Sweep, graduated path vs base:**
  - unet_tile 4096x640 4x8 24,598 → 18,473 (−24.9%); guards −1.9% (DRAM RM, flat) to −13.0%;
  - DRAM flat (vae 65536x512 −0.3%, flux1 16k −2.6%), as expected for a DM-bound plan.
- **Precision:** bit-identical. **Domain:** everywhere, no exceptions.

**I2 `normalize_bcast_srcb` — WIN.** Graduated option: `zsub_raw_wseed`.
- **Mechanism:**
  - DEST is seeded with ELWSUB(0, b′ row broadcast); srcA is zeroed by UNPACR_NOP ZEROSRC, so there is no L1 read and no −1 tile.
  - One unpacker context covers each DEST window, with the b′ tiles addressed by the ch0 Z counter.
  - Then DEST += x·a (ELWMUL, ROW broadcast, HiFi3).

| case | base | `rowb` | `zsub` | `zsub_raw` | `zsub_raw_wseed` |
|---|---:|---:|---:|---:|---:|
| unet_rm_ip 16384x960 | 68,868 | 68,879 | 66,636 | 66,566 | 66,486 (−3.5%) |
| vae_tile 16384x512 | 33,150 | 33,183 | 31,917 | 31,759 | 29,498 (−11.0%) |
| unet_rm_ip 16384x320 | 30,490 | 30,387 | 30,351 | 30,270 | 30,119 (−1.2%, flat) |
| unet_rm_ip 1024x1280 | 13,179 | 13,116 | 13,046 | 12,951 | 12,870 (−2.3%, flat) |
| unet_tile 1024x1280 | 11,530 | 11,679 | 11,416 | 11,389 | 11,066 (−4.0%) |

- **`rowb`** (stock row-broadcast srcB for a and b′): flat.
- **`zsub_raw_seed2`** (two tiles per seed call): 31,259 on vae_tile; superseded.
- **`zsub_raw_wseed_wmul`** (windowed multiply as well): equal to wseed (29,499 vs 29,536). It is not graduated: more raw code for no gain.
- **Ablation on vae_tile:** `zsub_raw` 31,746; with no seed 27,695; with no seed and no multiply 24,827. The seed's cost was the per-call unpacker context, not its data, and that pointed to the window seed.
- **Rejected:**
  - MOVB2D writes only the high 16 bits under fp32 DEST, so it cannot seed;
  - an SFPU MAD was analysed as slower and not built;
  - more tiles per window is impossible without `dst_full_sync_en`, which is part of the contract.
- **Sweep:** unet_tile 4096x640 4x8 24,623 → 22,404 (−9.0%). DRAM and the guards are flat to −4.6%, e.g. g_dram_rm γ-only +1.3% (noise).
- **Precision:** bit-identical. 0 − tf32(b′) equals HiFi3 (−1)·tf32(b′) bit for bit. **Domain:** everywhere, no exceptions.

**I3 `rm_tilize_fusion` — NULL.** Benched on the Perf-1 kernels.

| case | base | `tz` (bound) | `tzraw` | `tznopack` | `tzslow` | `tzcopy` | `ux2m` | `ux2n` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unet_rm_ip 16384x960 | 68,894 | −7.1% | +0.0% | −3.2% | +7.8% | +4.2% | +5.4% | +0.0% |
| unet_rm_ip 16384x320 | 30,491 | −6.8% | +0.0% | −1.3% | +4.6% | +1.8% | +4.1% | +0.0% |
| unet_rm_ip 1024x1280 | 13,181 | −7.2% | +0.1% | −0.8% | +0.6% | −0.8% | +2.6% | +0.0% |
| refiner_rm_ip 16384x384 | 29,720 | −9.7% | +0.0% | −1.8% | +5.9% | +2.5% | +4.9% | +0.0% |

- **`tz`** stubs the tilize entirely, so its output is wrong. It is the upper bound on what any tilize fusion could win.
- **`tzraw`**, the helper's fast-tilize loop inlined, is flat: the helper already is the fast path.
- **`tznopack`** drops the packer's row-chunk pack. The tilize is not pack-bound.
- **`tzslow`** (the non-fast tilize) and **`tzcopy`** (a datacopy tilize) are slower.
- **`ux2m` / `ux2n`** double the srcA unpack in pass 1 / pass 2:
  - pass 1 is partly srcA-unpack-bound (+2.6 to +5.4%), so a tilize-mode srcA premium would not hide there;
  - pass 2 has slack (flat).
- **Fused unpack-tilize** (estimated ≈ −6%) was not built. It needs the raw-LLK moments and normalize, which I1 and I2 were rewriting in parallel. It is a round-3 candidate on top of them.

**I4 `combine_tail` — WIN.** Graduated option: `mr+fraw`.
- **`fraw`:** a raw finalize stages the matmul init and −k = (−1 tile) × [k row; zeros] into DEST 2 before the gather wait.
- **`mr`:** the finalize emits mean·rstd. The group stats are 3 pages, [rstd, mean·rstd, e], with e = 1.0 written once by the writer. This removes:
  - the ones tiles (CB 17);
  - the `cb_mr_e` (18) chain;
  - the group-shift row replication and its UnpackToDestFp32.

| case | base | `mr` | `fraw` | `mr+fraw` | `au+mr+fraw` |
|---|---:|---:|---:|---:|---:|
| unet_rm_ip 1024x1280 | 13,176 | 12,787 | 12,827 | 12,558 (−4.7%) | 12,944 |
| unet_tile 1024x1280 | 11,576 | 11,288 | 11,221 | 11,016 (−4.8%) | 11,547 |
| refiner_rm_ip 256x1536 | 9,947 | 9,708 | 9,781 | 9,719 (−2.3%) | 10,133 |
| unet_rm_ip 1024x640 | 10,359 | 10,186 | 10,264 | 10,053 (−3.0%) | 10,507 |
| unet_tile 4096x640 4x8 | 24,590 | 24,339 | 24,311 | 24,129 (−1.9%) | 24,678 |

- **Nulls:**
  - `ar`, an mcast all-gather with every core finalizing: 13,202 → 32,144 (+143%); `ar+mr` +116%; `ar+mr+fraw` +187%.
  - `au`, a unicast all-gather, is slower than `mr+fraw` everywhere.
- **Graduation re-bench:** 13,156 → 12,550, 11,546 → 11,025, 9,919 → 9,669, 10,391 → 10,055, 24,583 → 24,137.
- **Large and DRAM cells:** flat (vae_dram 65536x512 +0.5%).
- **Cg = 3 guard:** a +3.4% in the first sweep was re-run twice, giving −0.4% and +1.2%. That is noise, not a regression.
- **Precision:**
  - `fraw` is bit-identical;
  - `mr` gives 1-ulp differences, with rel-RMS equal or lower against the reference on every case.
- **Domain:** everywhere, no exceptions.

**I5 `reader_prologue` — WIN.** Graduated options: OB + OZ + OE + OS + OP.
- **OB:** 8-word load-then-store runs for the affine maps.
- **OZ:** MEM_ZEROS stateful NoC writes plus whole-tile self-copies for the zero fills. They replace `Noc::async_write_zeros`, whose loopback reads shared the read barrier with the in-flight γ/β read.
- **OE:** the γ/β DRAM read is issued at step `num_chunks − 1`, before `build_map`.
- **OS:** straight-line shift-row stores.
- **OP:** stateful pilot reads.

OR (the ones-tile and group-shift fill) was superseded by I4, which deleted both. OM (straight-line map one-hot stores) was flat and dropped.

| case | base | OB alone | +OZ +OE | +OS | +OP | floor (P1 P2 TZ CMB PEER stubbed) base → graduated |
|---|---:|---:|---:|---:|---:|---:|
| refiner_rm_ip 256x1536 | 10,043 | 8,904 (−11.3%) | 8,893 | 8,886 | 8,874 | 7,913 → 4,968 |
| unet_tile 1024x1280 | 11,594 | 11,579 | 11,678 | 11,430 | 11,421 | 7,696 → 5,132 |
| unet_rm_ip 1024x1280 | 13,191 | 13,183 | 13,114 | 13,203 | 13,194 | 7,247 → 4,979 |
| unet_tile 4096x640 4x8 | 24,644 | — | — | 24,579 | 24,604 | 7,597 → 6,096 |

- **What it buys:** the reader floor drops 1.5–2.9 µs (−20 to −37%). On the Perf-1 compute this showed on the whole op only where the reader was critical: refiner 256x1536 9,939 → 8,890 (−10.6%), unet_tile 1024x1280 −1.2%, the rest flat.
- **Why it still graduated:** a flat cell keeps the unified path, and I1 and I4 shrink the compute in front of the floor. The final whole-op numbers below show the floor reduction surfacing.
- **Zones:**
  - the zero fills went from ~1.5 µs to ~0.59 µs;
  - the map / affine build went from 2.55 µs to 1.12 µs;
  - OP was flat (kept: same code size, one NoC state).
- **Sweep:** DRAM and guards flat (within ±1.1%). **Precision:** bit-identical. **Domain:** everywhere, no exceptions.

**I6 `dram_dynamic_balance` — WIN.** Graduated option: `split:d+da:auto`.
- On DRAM plans the packer puts each pass-2 chunk in whichever output CB has a free chunk slot: `cb_output_tiles` (5, the writer's share) or `cb_output_b` (27, the reader's share). It announces the chunk index on `cb_out_seq_a` (28) / `cb_out_seq_b` (29), and `0xFFFFFFFF` closes the block.
- The reader drains its share while it waits for input-ring room.
- `out_writer_chunks` is 2 when L1 has a chunk to spare.

| case | base | D6 `dyn` (tickets) | `split` 0.5 static | `split:d` | `split:d+da:2` | `split:d+da:auto` (grad.) |
|---|---:|---:|---:|---:|---:|---:|
| vae_dram 65536x512 | 374,024 | 389,871 (+4.2%) | 389,651 | 335,610 | 339,293 | 340,548 (−9.0%) |
| vae_dram 262144x256 | 783,727 | 766,781 | 802,788 | 706,127 | — | 711,263 (−8.8%) |
| refiner_dram 16384x1152 | 215,685 | 213,224 | 227,344 | 200,092 | 201,698 | 203,003 (−5.7%) |
| flux1_dram 16384x256 | 53,194 | 59,179 (+11.3%) | 55,130 | 55,415 (+4.2%) | 53,354 | 53,205 (flat) |
| vae_dram 1048576x256 | 3,754,261 | 3,947,893 (+5.2%) | — | — | — | 3,687,368 (−1.7%, flat) |

Each option's % is against its own run's base. The graduated column is the graduation re-bench (bases 374,204 / 780,065 / 215,289 / 53,097 / 3,750,676, within 0.5% of the bases shown).

- **D6 dynamic tickets** (NoC atomic fetch-add work stealing across a unit): a REGRESSION. vae_dram 1048576x128 +6.0% and mochi +1.6% as well.
- **Static splits are unstable:**
  - the oracle-tuned static shares win one shape and lose another (`static:p12`: refiner −8.1% but vae 262144x256 +4.4%, mochi +3.0%);
  - `split:1` (the reader writes everything) is +10 to +15%.
- **`split:d`** (the writer keeps one chunk) regressed flux1 16k by 4.2%. `da:auto` fixes that, and the `tau` thresholds added nothing.
- **Sweep:** flux1 1048576x64 −12.9%; vae 1048576x128 −4.1%; vae 262144x512 −3.3%; mochi +0.3% and the mid / guard cells within ±1.4% (flat).
- **Sharded:** unchanged. Its output is the zero-copy shard, so there is no stream to split.
- **Precision:** bit-identical. **Domain:** every DRAM plan.

### Step 4 — aggregation
Five WINs, and no win superseded another. Three conflicts were resolved at merge:
- **I2 vs I4 (`cb_neg_one`).** I2 deleted the −1 tile (CB 6), but I4's finalize uses it as in0, for −k and for the −S sender sum. The CB was kept, and its fill now runs only on unit roots (`if constexpr (IS_ROOT)`); the normalize no longer reads it.
- **I5 OR vs I4.** OR sped up the ones-tile and group-shift fill, which I4 deletes. I4 wins, and the group shift is now one NoC MEM_ZEROS fill (block 0) plus row 0 per block.
- **I6 vs I2.** I6's output-CB choice was merged onto I2's seeded normalize (`normalize_chunk(rows_valid, q)`).

The merge order was I1 → I5 → I4 → I2 → I6, each re-gated on golden before the next.

### Step 5 — graduations (in commit order; each one a single unified path that deletes the code it replaced)

| # | commit | graduation | code deleted | domain / carve-outs |
|---|---|---|---|---|
| 1 | `d9d9df8e02a` | I1: pass-1 moments as one whole-tile FPU program per x tile | C2's `add_bcast_rows_acc_init`, `math_mul_acc_from_dest_tile`, `mul_x_by_dest_tile_acc` and the per-row-block stock eltwise loop | everywhere, no carve-outs |
| 2 | `499c233744f` | I5: early γ/β issue, NoC MEM_ZEROS fills, run-copy map build, straight-line shift rows, stateful pilot reads | the `Noc::async_write_zeros` fills, the per-word map / affine store loops, the late γ/β issue | everywhere, no carve-outs |
| 3 | `5b79ab6d6cb` | I4: raw finalize staged before the gather wait; mean·rstd in the finalize; 3-page group stats; shared-init expand | the `matmul_block` + PostComputeFn sender sum, the ones tiles (CB 17) and their reader fill, the `cb_mr_e` (18) chain, the group-shift row replication and its UnpackToDestFp32 copy | everywhere, no carve-outs |
| 4 | `8ab37ff4bda` | I2: normalize seed 0 − b′ in one unpacker context per DEST window, x·a row broadcast | the (−1)·b′ `mul_tiles` seed (per-tile unpacker context, 3 fidelity phases); the normalize's read of `cb_neg_one` | everywhere, no carve-outs |
| 5 | `9416ccba748` | I6: DRAM pass-2 output split across both DM RISCs by free slot | the writer-only, single-consumer output stream | Every DRAM interleaved plan. **Feasibility carve-out** (not measurement-earned): sharded plans pack into their own output shard (`if SHARDED {shard} else {split}`), with no stream to split. |

Instrumentation follows the new paths:
- `r_ones_shift_affine` became `r_affine` (the ones and shift fills are gone; `zone_table.py` reads both names);
- inside `finalize_raw`, `c_wait_gather` times only the gather wait (the init and the −k product are staged before it), and `c_finalize` times the sums, the SFPU finalize and the pack after it;
- the reader's output drain runs under the opt-in `r_reserve` chunk zone;
- the writer's output loop runs under `w_wait_out` / `w_issue`.

All six experiment dirs, including the null `rm_tilize_fusion`, are committed as artifacts.

### Whole op, before (`02a76783017`) → after (`9416ccba748`)

**Sharded in place (9 RM PERF FOCUS + 3 TILE)**

| case | before ns | after ns | Δ | speedup |
|---|---:|---:|---:|---:|
| `unet_rm_ip_1024x1280` | 13,207 | 10,851 | -17.8% | 1.22× |
| `unet_rm_ip_16384x320` | 30,705 | 23,544 | -23.3% | 1.30× |
| `unet_rm_ip_4096x320` | 13,260 | 10,809 | -18.5% | 1.23× |
| `unet_rm_ip_4096x640` | 18,934 | 15,077 | -20.4% | 1.26× |
| `unet_rm_ip_1024x640` | 10,572 | 8,599 | -18.7% | 1.23× |
| `refiner_rm_ip_256x1536` | 9,840 | 7,658 | -22.2% | 1.28× |
| `refiner_rm_ip_16384x384` | 30,124 | 20,646 | -31.5% | 1.46× |
| `unet_rm_ip_16384x640` | 54,040 | 40,431 | -25.2% | 1.34× |
| `unet_rm_ip_16384x960` | 69,026 | 47,573 | -31.1% | 1.45× |
| `unet_tile_ip_1024x1280` | 11,683 | 8,971 | -23.2% | 1.30× |
| `unet_tile_ip_4096x640_4x8` | 24,754 | 15,696 | -36.6% | 1.58× |
| `vae_tile_ip_16384x512` | 33,257 | 18,747 | -43.6% | 1.77× |

**Sharded out of place (3 TILE PERF FOCUS + 7 RM)**

| case | before ns | after ns | Δ | speedup |
|---|---:|---:|---:|---:|
| `unet_tile_1024x1280` | 11,636 | 9,020 | -22.5% | 1.29× |
| `unet_tile_4096x640_4x8` | 24,847 | 15,690 | -36.9% | 1.58× |
| `vae_tile_16384x512` | 33,216 | 18,755 | -43.5% | 1.77× |
| `unet_rm_oop_1024x1280` | 13,360 | 10,916 | -18.3% | 1.22× |
| `unet_rm_oop_16384x320` | 30,681 | 23,478 | -23.5% | 1.31× |
| `unet_rm_oop_4096x320` | 13,244 | 10,818 | -18.3% | 1.22× |
| `unet_rm_oop_4096x640` | 18,887 | 15,087 | -20.1% | 1.25× |
| `unet_rm_oop_1024x640` | 10,489 | 8,601 | -18.0% | 1.22× |
| `refiner_rm_oop_256x1536` | 9,858 | 7,675 | -22.1% | 1.28× |
| `refiner_rm_oop_16384x384` | 30,109 | 20,629 | -31.5% | 1.46× |

**DRAM interleaved (9 PERF FOCUS + 4 mid)**

| case | before ns | after ns | Δ | speedup |
|---|---:|---:|---:|---:|
| `vae_dram_1048576x128` | 1,759,737 | 1,677,164 | -4.7% | 1.05× |
| `vae_dram_65536x512` | 374,656 | 333,369 | -11.0% | 1.12× |
| `vae_dram_262144x256` | 780,447 | 703,718 | -9.8% | 1.11× |
| `vae_dram_262144x512` | 1,765,779 | 1,695,393 | -4.0% | 1.04× |
| `vae_dram_1048576x256` | 3,762,345 | 3,675,407 | -2.3% | 1.02× |
| `refiner_dram_16384x1152` | 214,630 | 197,230 | -8.1% | 1.09× |
| `flux1_dram_16384x256` | 53,757 | 51,525 | -4.2% | 1.04× |
| `flux1_dram_1048576x64` | 725,887 | 630,667 | -13.1% | 1.15× |
| `mochi_dram_8x407040x128` | 5,908,341 | 5,879,519 | -0.5% | 1.00× |
| `mid_4096x640_tile` | 37,865 | 36,623 | -3.3% | 1.03× |
| `mid_4096x640_rm` | 46,261 | 44,779 | -3.2% | 1.03× |
| `mid_16384x320_tile` | 68,240 | 65,719 | -3.7% | 1.04× |
| `mid_16384x320_rm` | 82,461 | 77,749 | -5.7% | 1.06× |

Results by placement:
- **Sharded:** every focus cell is 1.22–1.77× faster.
- **DRAM:**
  - The real wins are on the mid-size plans, where the per-core imbalance dominated: vae 65536x512 −11.0%, vae 262144x256 −9.8%, refiner −8.1%, flux1 1048576x64 −13.1%.
  - The giant plans move by less: vae 1048576x256 −2.3% is inside the noise band; mochi −0.5% is flat.
  - Against the 92%-of-512 GB/s floor: vae 1048576x256 is now 1.16× (Perf 1: 1.18×); mochi 1.16× (unchanged); vae 262144x256 1.14× (was 1.27×).
- **Mid cells:** −3.2 to −5.7%. The −3.2% / −3.3% pair is at the noise edge.

### Guard set (no regression)
One representative per kernel path × layout × placement for the affine modes the focus benches do not run, plus the odd channels-per-group case (Cg = 3; the reader's halfword map path). Same base and after builds as above.

| guard case | before ns | after ns | Δ |
|---|---:|---:|---:|
| rm in place 1024x1280, γ only | 12,872 | 10,569 | -17.9% |
| rm in place 1024x1280, no affine | 12,673 | 10,499 | -17.2% |
| tile oop 1024x1280, γ only | 11,272 | 8,756 | -22.3% |
| tile oop 1024x1280, no affine | 11,011 | 8,607 | -21.8% |
| DRAM tile 4096x640, γ only | 37,698 | 36,023 | -4.4% |
| DRAM tile 4096x640, no affine | 37,684 | 35,939 | -4.6% |
| DRAM RM 4096x640, γ only | 45,310 | 44,121 | -2.6% |
| DRAM RM 4096x640, no affine | 45,099 | 43,740 | -3.0% |
| DRAM tile 16384x96, γ+β, Cg = 3 | 27,000 | 26,113 | -3.3% |

No cell is slower: every focus, mid and guard cell is faster or flat. The two DRAM RM guards (−2.6%, −3.0%) are noise-band flat, so no carve-out was earned by measurement.

### Correctness
- Full golden directory (`eval/golden_tests/groupnorm_sc_N_1_HW_C_b/`): **676 passed, 0 failed** after every graduation. The rest are xfail (unsupported cells) or skip (invalid cells).
- Loose + regression: **64/64 under `--dev`** (watcher, LLK asserts) on the final tree, and after graduations 3–5. The I1 and I5 subagents ran `--dev` on their graduation candidates. Every kernel binary fits the 70,688 B `--dev` kernel-config ring.
- Precision:
  - I1, I2, I5 and I6 are bit-identical on every benched case;
  - I4's `mr` (mean·rstd computed once in the finalize, in fp32) differs by 1 ulp in places, with rel-RMS equal or lower against the reference;
  - no precision-trading option was graduated.

### Helper bypasses
Every raw-LLK path in the op today, with the round that measured it. Each site carries a kernel-head comment stating the measured win.

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `reduce<SUM, REDUCE_COL>` + `Accumulate`, `eltwise_chain`, `mul_reuse_dest_tiles`, stock `add_tiles_bcast<ROW>` + `mul_reuse_dest_init` (Perf 2, I1) | capability | No API keeps one unpacked srcA bank live across ELWADD, MOVD2B and ELWMUL; every helper re-unpacks x per op. The stock eltwise MOP runs the fidelity phases inside a 2-block face, so an accumulating ELWMUL stalls on DEST every phase. Hiding it needs the fidelity loop outside ≥ 5 distinct 8-row blocks, and no helper exposes that loop order. | 68,881 (rm_ip 16384x960) / 33,130 (vae_tile 16384x512), C2 | 50,567 / 23,033 | `groupnorm_sc_N_1_HW_C_b_compute.cpp:175` (`moments_rows` :342) |
| `mul_tiles` / `mul_init` seed and `llk_unpack_A` / the binary MOP (Perf 2, I2) | capability | No op seeds DEST with 0 − B (row broadcast) with srcA zeroed by the unpacker (no L1 read). No helper runs several tiles in one unpacker context, addressing them by the Z counter: each call pays a base-address write and a semaphore round trip, which was the seed's whole cost. | 33,136 (vae_tile 16384x512) / 68,868 (rm_ip 16384x960) | 29,505 / 66,506 | `groupnorm_sc_N_1_HW_C_b_compute.cpp:384` (`p2_seed_window` :476) |
| `compute_kernel_lib::matmul_block` + a finalize `PostComputeFn` (Perf 2, I4) | capability | The helper acquires DEST only after its input waits, so the init and the −k product cannot be staged before the gather arrives. There is no "pre-wait" hook. Carrying k in its PostComputeFn forced a row-replicated UnpackToDestFp32 group-shift tile. | 13,176 (rm_ip 1024x1280) / 11,576 (tile 1024x1280) | 12,827 / 11,221 (fraw alone) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:694` (`finalize_raw` :699) |
| `cb_reserve_back` / `cb_push_back` single-consumer output pattern (Perf 2, I6) | capability | There is no producer-side non-blocking "free pages" probe on the packer, and no way for one producer to choose between two consumers' CBs per chunk. Raw `tiles_received` / `tiles_acked` reads and a sequence word stored into the reserved page. | 374,204 (dram 65536x512) / 780,065 (dram 262144x256) | 340,548 / 711,263 | `groupnorm_sc_N_1_HW_C_b_compute.cpp:495` (`pack_free_pages`), `:510` (`output_cb_for`) |
| `Noc::async_write_zeros` + `write_zeros_l1_barrier` (Perf 2, I5) | capability | It zeroes by loopback *reads*, so its barrier also waits for the γ/β DRAM read in flight. Overlapping the two needs stateful NoC writes from `MEM_ZEROS`, which retire on the write counter. | zero-fill zone ~1,500 | ~590 | `groupnorm_sc_N_1_HW_C_b_reader.cpp:209` (`build_map`), `:497` (group shift) |
| no helper covers this (`l1_helpers` strided word-run copy; supersedes Perf 1's map-scatter row) (Perf 2, I5) | capability | No helper copies a strided L1 word run into tile-face layout at a computed row (the gamma map's `gamma[c]` at row `g(c)`, the beta map's row 0). `fill_tile_column` writes all 32 rows, and `generate_index_tile` writes a ramp. 8-word load-then-store runs let the in-order core overlap the L1 load latencies. | map / affine zone 2,550 | 1,120 | `groupnorm_sc_N_1_HW_C_b_reader.cpp:281` (`r_affine` :279) |
| `eltwise_chain` finalize (CopyTile ×3, 8 SFPU ops, PackTile ×2) (Perf 1, now inside `finalize_raw`) | capability | The chain has no vector scope: every SFPU op covers all 32 vectors though only row 0 is needed. It cannot run inside a matmul's DEST window. | 27,125 (rm_ip 1024x1280, Perf 1) | 25,603 (Perf 1) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:686` (`gn_finalize_sfpu` :675) |
| `matmul_block` (expand products only) (Perf 1) | capability | It has no partial-subblock pack, and under fp32 DEST it re-issues `pack_reconfig_data_format` per subblock. In Perf 2 the two products also share one init (the helper issues `matmul_block_init` on every call), and in0 is read at a page offset of the 3-page group stats (the helper has an in1 base-offset hook but none for in0). | 25,973 (rm_ip 1024x1280, Perf 1) | 25,433 (Perf 1) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:730` (`expand_matmul` :740) |
| `untilize<>` after PackTile → CB (Perf 1) | capability | No untilize helper takes a block already in DEST (`pack_untilize_dest`); the helper round-trips through a tile ring. | 161,601 (rm_ip 16384x960, Perf 1) | 153,853 (Perf 1) | `groupnorm_sc_N_1_HW_C_b_compute.cpp:581` (`normalize_untilize_chunk`) |

Perf 1's `reduce`/`Accumulate` moments row and its `eltwise_chain` normalize row are superseded by the I1 and I2 rows above: the same sites, now with Perf 2's code and numbers.

### Notes for round 3 (re-run Step 1 first)
- **Where the time sits now** (`zone_table.py` on `_round2/logs/final_full.device.csv`, TRISC_1 p50 over cores; the math-thread zones include their own unpacker waits):
  - unet_rm_ip_1024x1280, span 9.7 µs: pass 1 3.9; `c_wait_stats` 1.6; finalize 1.0; expand 2.3; pass 2 2.5.
  - unet_rm_ip_16384x960, span 46.9 µs: pass 1 22.3, pass 2 19.6, `r_affine` 4.6.
  - vae_tile_16384x512, span 18.3 µs: pass 1 7.7, pass 2 9.5. Pass 2 now leads on TILE.
- **Exposed on small TILE:** the pilot wait, `c_wait_pilot` ≈ 1.06 µs on unet_tile 1024x1280.
- **Fused unpack-tilize (I3)** is now buildable on top of the raw-LLK moments and normalize. `ux2m` showed pass 1 is partly srcA-unpack-bound, so the tilize-mode unpack premium has to be measured there, not assumed hidden.
- **DRAM** is at 1.14–1.16× the 92% floor. Pass-2 writes are now balanced across both DM RISCs, but pass-1 reads are still reader-only: a writer-side share of the pass-1 reads is the remaining DM lever.
- **`--dev` kernel-config ring:** every graduation still fits, but check the margin before adding root-side code.

**Summary:** six ideas floated and measured; five graduated (each as one unified path with the replaced code deleted, and the losing options recorded) and one null (`rm_tilize_fusion`). On the focus shapes the op is now:
- 1.22–1.77× faster on sharded in place;
- 1.22–1.77× on sharded out of place;
- 1.04–1.15× on DRAM wherever the change is outside noise (vae 1048576x256 −2.3% and mochi −0.5% are flat).

Mid cells are 3–6% faster. Sharded guards are 17–22% faster and DRAM guards 3–5% (the DRAM RM pair sits at the noise edge). No cell regressed, and golden is green.

## Refinement 3 — ragged shard geometries (padded last row/column, shards straddling images, groups straddling core columns)
- Date: 2026-11-26
- What was done:
  - **Registry.** `SUPPORTED["shard_fit"]` = `["none", "even", "ragged"]`, both `in_place` values. The remaining sharded refusals are geometric and raised by the planner (`NotImplementedError`): a core without data, image bands of unequal height, more than 32 group lanes in one statistic unit, a non-rectangular or non-ROW_MAJOR shard grid.
  - **Statistic unit = column cluster × image band** (`_plan_sharded`). Columns cluster into the whole grid width when a group straddles a core-column boundary (`sw % Cg ≠ 0`), else single columns. Grid rows split into bands only where a core-row boundary is an image boundary. A unit holds `NU` images, a core's shard `NSEG` image segments. The unit root gathers `R = ceil(unit cores / 32)` slot rows per image, finalizes `NU` images and multicasts `[rstd_i, mean_i·rstd_i]·NU + e`. Whole-width units multicast to the RowMajor band rectangle.
  - **Padded edges (knob-turn).**
    - TILE contributes whole valid tiles only (`vrows`, `vc/32`).
    - RM masks the padded lanes after the column collapse (`col_mask`: an SFPU lane mask, `(32c + col) mod P ≥ vc`). It stages a partial last tile row k-filled: every stick that is not the row's own is the pilot `k` stick, so `x − k = 0` and `x·(x − k) = 0` add nothing to either moment.
  - **Straddled images (scheme-change).** Each segment of a core collapses to its own partial pair, sent to its image's slot block at the unit root. Pass 2 expands each image's stats pair and applies it to that segment's rows. RM tile rows mixing two images are staged like partial rows; pass 2 untilizes them into `cb_rm_stage_out` and the writer copies the valid sticks back into the shard.
    - Deviation from the verifier sketch (two combine rounds with different roots): one root per unit with per-image slots is one gather / mcast round, not two serialized ones.
  - **Straddled groups (`cross_column_group_combine`, scheme-change).** Each core's collapse already emits partials in the unit's lane space (from lane `l0` of its first channel, padded lanes zero). The unit root's existing sum over gathered slot rows therefore combines column neighbours with no extra exchange: a one-level 2-D fan-in, not the sketched row exchange + column gather. The pilot `k` of a lane is read from the core column holding the group's first channel, in the band's first core row (an image's row 0).
  - **Sharded RM ops path (single-source chunking).** Every fully resident sharded RM core, even shards included, walks its shard as op words: whole tile-row runs of up to `h` rows, `OP_STAGED` rows, `OP_SEG_END` segment ends, and `OP_SKIP` over a staged position. Even and ragged shards share one path; the legacy chunked path remains for DRAM and partial residency.
  - **Perf repairs on the even-shard path**, so R1/R2 cells don't regress (measured: `perf_experiments/r3_ragged/`):
    - `normalize_untilize_rows` takes its output CB as a template argument. A runtime CB id makes the packer's per-CB descriptor tables (4 × 64 B) live `.data`, which `do_crt1` copies on every launch: +234 ns on the pack thread's kernel start.
    - `NU == 1` folds the writer's and compute's per-image loops to constant trips (−~150 ns).
    - The RM pilot issues one span read of stick 0 per source column instead of one 16 B read per group lane: ~200 ns of NCRISC issue on a 4-lane core, now hidden by pass 1.
  - **Docs.** `l1_ledger.md`: rows for CBs 15/16/19/20 (`NSEG` / `NU`), 21 (`staging_bytes()` incl. the per-column pilot spans), new 25/26 (alias group F), the ring and shard-CB `total_size` notes, deviations, footprint formula, a ragged-geometry table from the planner, and the DM budget. `op_design.md`: the `block_sharded_shard_resident`, `cross_column_group_combine` and `shard_fit=ragged` rows are built, with their realizations.
- Reused: the R1/R2 shard-backed CBs, collapse / gather / finalize / mcast / expand chain, map channel bound, the pilot shift, the RM tilize / untilize helpers and the `K = B` residency plan.
- Added: per-image / per-segment slot blocks and stats (`NU`, `NSEG`), the column-cluster × band unit planner, RM staged rows (alias group F, CBs 25/26), the RM padded-lane mask, op words for sharded RM, and the per-column pilot.
- Accuracy achieved (bf16, γ+β, bench `perf_experiments/r3_ragged/bench_ab.py`): every ragged case PCC 0.999998, relative RMS 0.00180–0.00198; the even sharded A/B cases PCC 0.999998, RMS 0.00176–0.00193 (unchanged from base).
- Golden test progress:
  - **66/66** `BLOCK_SHARDED, shard_fit=ragged` cells (33 out of place + 33 in place), under `--dev` and plain;
  - plus 234 even sharded cells and a 60-cell interleaved sample (360/360 `--dev`, 360/360 plain);
  - `test_regression.py` 43/43 (`--dev`);
  - unit directory (`test_groupnorm_sc_N_1_HW_C_b.py` 44, `_sharded.py` + `_ragged.py` 48) green `--dev` and plain (452/452 plain, golden slice included).
- Issues encountered:
  - **"Runtime args size mismatch: cached 11 vs new 12"** (`max_resident_chunks = 1`). The generic-op program hash covers CT args, CBs, core ranges and the core count, but not per-core RT lengths. The override branch recomputed `h` with the global chunk target: the same CT args as the default plan, a different op-word count. Fixed: the override uses `h_l` (the largest divisor of `rows_core` ≤ cap) and takes the legacy path only when `override < rows_core / h_l`.
  - **Hang:** `test_sharded_rm_partial_residency[max_resident=0]`. The writer waited for more pages than the shard CB holds. Fixed with per-chunk draining (`IS_RM && !OPS`).
  - **Watcher NoC error under `--dev`** ("unicast write ... overflows a circular buffer", RM 4096×320 8×7). The shard CB's `total_size` is floored to whole 2 KB pages (ragged RM shards are not page multiples), and the staged-row copy sourced bytes past it from local L1. The sanitizer checks the local address of a NoC transaction against the CBs containing it. Fixed by issuing the copy as a NoC read of the core's own L1 (91dbc589f6b).
  - **A `--dev` "device timeout"** with every TRISC at init: stale device state after a watcher-aborted run. A clean re-run passed, with no code change.
  - **The `.data` / runtime-CB-id regression** above, found by zone timelines (`perf_experiments/_tools/zone_timeline.py`) when the even-shard A/B showed +300–400 ns.
- Findings (not deferred work; recorded for the perf refinements):
  - **Residual on small even RM shards:** +0.1–1.4% (≤ ~110 ns) vs base, inside the noise band. The zones place it in the ops path's pass-2 prologue / op-word walk on the pack thread (~40–60 ns, part of it zone overhead) plus ~20 ns of per-row normalize. That is the price of one chunking path for even and ragged RM shards. The pilot's remaining +70 ns on NCRISC is absorbed by pass 1 (`c_p1_moments` ends at the same time).
  - **RM whole-width units with `sw % 32 ≠ 0` are W-bound.** 7×8 RM 1024×1280: `k = 4` sticks per virtual row make `W = 23` in one tile row, and moments (7.9 µs) and expand (6.5 µs) scale with `W` amortized over a single tile row: 21.8 µs vs the TILE twin's 10.9 µs. The reader's map / affine build (~10 µs) stays off the critical path.
  - **RM image-straddle 8×3:** the root waits 4.2 µs on the unit's slowest pass-1 member (the image-mixed core row, with staged rows and two segments), and pass 2 costs ~170 ns/tile of untilize vs ~65 for TILE.
- Tests added:
  - `tests/.../test_groupnorm_sc_N_1_HW_C_b_ragged.py`: 11 geometries (padded column TILE × 3; padded row TILE 8×7, RM 8×7, RM 8×7 `k = 4`; image straddle TILE / RM 8×3; group straddle TILE 7×8, RM 7×8, RM 6×8) × out of place / in place.
  - `perf_experiments/_round2/bench.py`: the `r3_*` sharded cases and their DRAM equivalents (groups `r3`, `sh_all`, `dram_small`).
  - `perf_experiments/r3_ragged/`: `bench_ab.py` (same-session A/B against a frozen pre-R3 copy, `base_pkg/` = 8db3a5fa93d), `run.sh`, logs.
  - `perf_experiments/_tools/zone_timeline.py`: first start / last end per (RISC, zone), p50 over cores.

### Device kernel duration — Refinement 3 (Blackhole, DEVICE KERNEL DURATION [ns], one fresh-cache profiled run per case, bf16 γ+β)

Even shards and DRAM, same-session A/B against the pre-R3 op (`base_pkg` = 8db3a5fa93d, Perf 2 final). R1/R2 cells do not regress beyond noise:

| Case | Base (ns) | R3 (ns) | Δ |
|---|---|---|---|
| unet_rm_ip_1024x1280 | 10 827 | 10 841 | +0.1% |
| unet_rm_ip_16384x320 | 23 499 | 23 588 | +0.4% |
| unet_rm_ip_4096x320 | 10 741 | 10 853 | +1.0% |
| unet_rm_ip_4096x640 | 15 051 | 15 171 | +0.8% |
| unet_rm_ip_1024x640 | 8 560 | 8 676 | +1.4% |
| refiner_rm_ip_256x1536 | 7 687 | 7 793 | +1.4% |
| refiner_rm_ip_16384x384 | 20 587 | 20 637 | +0.2% |
| unet_rm_ip_16384x640 | 40 407 | 40 509 | +0.3% |
| unet_rm_ip_16384x960 | 47 468 | 47 623 | +0.3% |
| unet_rm_oop_1024x1280 | 10 826 | 10 846 | +0.2% |
| unet_tile_1024x1280 | 9 001 | 8 813 | −2.1% |
| unet_tile_4096x640_4x8 | 15 719 | 15 744 | +0.2% |
| vae_tile_16384x512 | 18 759 | 18 867 | +0.6% |
| unet_tile_ip_1024x1280 | 9 000 | 8 791 | −2.3% |
| flux1_dram_16384x256 | 50 338 | 50 908 | +1.1% |
| refiner_dram_16384x1152 | 198 021 | 198 704 | +0.3% |
| vae_dram_65536x512 | 334 484 | 333 905 | −0.2% |
| mid_4096x640_rm | 44 613 | 44 112 | −1.1% |
| mid_16384x320_tile | 65 443 | 65 067 | −0.6% |

Ragged shards (newly supported; the base op refuses them) vs the same tensor through this op's DRAM-interleaved path:

| Case | Ragged sharded (ns) | DRAM path (ns) | vs DRAM |
|---|---|---|---|
| TILE 4096×960 8×8 (padded column, groups straddle) | 14 249 | 62 798 | 4.41× faster |
| TILE 1024×1920 8×8 (padded column, groups straddle) | 12 550 | 40 215 | 3.20× faster |
| TILE 4096×1152 8×8 (padded column, groups straddle) | 16 676 | 63 441 | 3.80× faster |
| RM in place 4096×320 8×7 (padded row, `k = 4`) | 13 813 | 26 862 | 1.94× faster |
| TILE 1024×1280 8×7 (padded row) | 9 279 | — | — |
| TILE 2×256×1280 8×3 (images straddle) | 13 356 | 15 021 | 1.12× faster |
| RM in place 2×256×1280 8×3 (images straddle) | 21 590 | — | — |
| TILE 1024×1280 7×8 (groups straddle) | 10 819 | — | — |
| RM in place 1024×1280 7×8 (groups straddle, `W = 23`) | 21 802 | 25 699 | 1.18× faster |
| RM in place 4096×640 6×8 (padded column, groups straddle) | 20 056 | 45 206 | 2.25× faster |

## Refinement 4 — fp32 input (DRAM interleaved), fp32 γ/β, and TILE-layout γ/β
- Date: 2026-11-26
- What was done:
  - **Registry.**
    - `SUPPORTED["dtype"]` gains `float32`, `["affine_dtype"]` gains `float32`, and `["affine_layout"]` gains `TILE_LAYOUT`.
    - `EXCLUSIONS` refuses `BLOCK_SHARDED` combined with any of the three. Those cells are scope-parked INVALID in `feature_spec`; for a direct caller the op raises `ExcludedCell`. The sharded pilot, staging and pack-untilize paths are bf16 byte math, and a reader `static_assert` pins that.
    - `validate()` passes `formats_of(input, gamma, beta)` to `plan_or_raise`. The new `fmt=` argument defaults to the input dtype with bf16 RM γ/β, so the old probe call sites still work.
  - **One page-size source (descriptor).** A frozen `Formats` record (`in_dtype`, `aff_dtype`, `aff_tile`) derives `in_tile`, `in_elem`, `amap_tile` and `aff_elem` from `ttnn.tile_size`. It is stored on the plan and threaded through `footprint_bytes`, `output_ring_bytes`, `staging_bytes` / `affine_stage_bytes`, `_evaluate_gx` / `_choose_gx`, the `K_max` / `f_resident` / `chunk_bytes` solve and the sharded planner.
    - CB formats follow the tensors. The input / output side (0, 1, 2/3, 5, 25/26, 27) uses the input dtype at `in_tile` pages; the output dtype is the input dtype. The γ/β maps (13) use `aff_dtype` at `amap_tile` pages. The map (12), pilot rows (22) and −1 tile (6) stay bf16; the statistics stay Float32.
    - The `BF16_TILE_BYTES` literals that remain size only those bf16-by-design CBs.
  - **Reader (same file).**
    - Four new CT args: 16 `IN_TILE_BYTES`, 17 `AFF_ELEM`, 18 `AFF_TILE`, 19 `AFF_PAGE_BYTES`. The `TensorAccessorArgs` chain now starts at 20.
    - The stick segment, chunk tile stride, face offsets and the RM pilot length all derive from `IN_TILE_BYTES`.
    - The pilot `k` is the bf16 upper halfword of a Float32 sample, a valid shift. The kernel's `kv` stays bf16 bits, so `cb_pilot_rows` and `cb_group_shift` are unchanged.
    - γ/β staging:
      - RM sticks: the 64 B-aligned slice is now in γ/β's element size.
      - TILE: one 64 B row-0 read of faces 0/1 per channel tile, 128 B per tile staged. This replaces the RM stick read for that layout.
    - `aff_idx()` maps a channel to its staged element.
    - `build_affine` word-copies runs into the maps in γ/β's own format: a Float32 run is one word per lane, so odd `C/G` works too. `copy_run` now takes up to 16 words. bf16 with odd `C/G` keeps the halfword path.
    - The γ/β ring zero-fill copies the 2 KB zero tile across `AMAP_TILES·amap_tile` bytes.
  - **Compute and writer: no change.** Formats come from the CB configs via the existing `reconfig_data_format` / matmul init / `pack_reconfig` calls. The tilize helper drops fast tilize for a Float32 output by itself. The interleaved writer already used `OUT_PAGE_BYTES`.
  - **No `UnpackToDestFp32` on the new CBs.** Per /numeric-formats-metal §1.5, every consumer of the ring (2/3) is an FPU op: pass-1 ELWADD / ELWMUL and the pass-2 multiplies. The tilize input (0) feeds only a tilize whose output is FPU-consumed. A lossless tilize would therefore buy nothing: Float32 x reaches srcA as TF32 on both layouts, and TILE and RM give bit-identical outputs (measured). The fp32 accumulators (7, 8) keep their existing tags.
  - **Grid fill and knobs unchanged.** Same `_choose_gx` full-grid split (104–110 cores on the perf shapes) and the same chunk `h` / `T`. The residency `K_max = floor((L1 − F_fixed)/(in_tile·T))` roughly halves for Float32.
- Reused: the whole two-pass blocking scheme, every compute / writer kernel path, the reader's chunk / pilot / map / affine sites (parameterized, not forked), and the golden helper `run_groupnorm_sc_N_1_HW_C` for the new tests.
- Added: the descriptor's `Formats` record / `formats_of` / `affine_stage_bytes`, reader CT args 16–19, the TILE γ/β row-0 read loop and `aff_idx`, and `EXCLUSIONS` for sharded × the new values. No new kernel file.
- Accuracy achieved (Blackhole, HiFi3, fp32 DEST, γ+β Float32 TILE; probe_023):
  - Float32 input, `randn`: PCC ≥ 0.999998, relative RMS 1.1e-3 – 1.3e-3 (limit 0.01, 0.999) on 256×256 / 4096×320 / 64×1280 / 16384×512, TILE = RM.
  - With a +10σ mean offset (`randn + 10`): PCC 0.99997, relative RMS 6.4e-3 – 7.3e-3. This is still inside the limit, but the margin shrinks. The error is the TF32 quantization of x, a and b' at srcA / srcB in pass 2's `x·a − b'`. Its absolute size scales with `|x|`, and the cancellation against `b' ≈ mean·a` exposes it relative to the unit-variance output. An FPU pass 2 cannot avoid it: only an SFPU pass 2 from `UnpackToDestFp32` x would. Recorded as a finding; the golden distribution passes with ~8× margin.
  - bf16 input with Float32 or TILE γ/β: within the bf16 limits (golden slice below).
- Golden test progress:
  - **1950/1950** newly supported cells pass. That is every cell whose only previously-unsupported axes are among `{dtype=FLOAT32, affine_dtype=FLOAT32, affine_layout=TILE}`, all DRAM interleaved and tile-aligned. Run as node ids, `--precompile`, 9 min.
  - The legacy bf16 cells and the perf loose cases: **678/678 + 21/21**.
  - `test_regression.py` **43/43**.
  - Unit tests: spec + `_sharded` + `_ragged` + `_precision_baseline` 99/99 plain. With `--dev`, spec + `_sharded` + `_ragged` + `_dtypes` gave 192/192, so the kernel text still fits the kernel-config ring under the watcher build.
- Perf (Blackhole, DEVICE KERNEL DURATION [ns], one fresh-cache profiled run per case, `_op_bench.py`):
  - **bf16 DRAM cases, no regression** vs R3 / Perf 2:

    | Case | ns | Change |
    |---|---|---|
    | vae_dram_1048576x128 | 1 681 538 | +0.3% |
    | vae_dram_65536x512 | 331 393 | −0.6% |
    | vae_dram_262144x256 | 705 114 | +0.2% |
    | vae_dram_262144x512 | 1 697 099 | +0.1% |
    | vae_dram_1048576x256 | 3 665 704 | −0.3% |
    | refiner_dram_16384x1152 | 197 807 | −0.5% |
    | flux1_dram_16384x256 | 51 255 | +0.7% |
    | flux1_dram_1048576x64 | 632 182 | +0.2% |
    | mochi_dram_8x407040x128 | 5 843 043 | −0.6% |
    | mid_4096x640_rm | 44 228 | +0.3% |
    | mid_16384x320_tile | 66 182 | +1.7%, noise band |

    The bf16 reader code is constexpr-equivalent to R3: same reads, same stores, same loop trips.
  - **New formats:**

    | Case | ns | vs bf16 |
    |---|---|---|
    | Float32 vae 65536×512 (TILE γ/β) | 845 314 | 2.55× |
    | Float32 refiner 16384×1152 | 392 307 | 1.98× |
    | Float32 flux1 16384×256 | 96 179 | 1.88× |
    | Float32 4096×640 RM | 67 000 | 1.51× |
    | bf16 x + TILE bf16 γ/β, vae 65536×512 | 333 831 | +0.7% |
    | bf16 x + Float32 RM γ/β, vae 65536×512 | 334 576 | +1.0% |

  - The Float32 ratios track bytes moved. vae 65536×512 drops from fully resident (`K = B = 10`) to `K = 5`: a Float32 per-core slice is about 1.2 MB against a 1.46 MB budget. Its traffic is (1 + 0.5 + 1)·2 = 2.5× the bf16 bytes, and the effective DRAM bandwidth is the same, about 397 vs 405 GB/s.
- Issues encountered:
  - The profiled bench run first "failed" because the tracy wrapper splits a `-k "a and not b"` expression on spaces. Passing explicit node ids fixed it; there was no op issue. Otherwise no failures: the first full run of the new test file passed 100/100.
- Tests added:
  - `tests/.../test_groupnorm_sc_N_1_HW_C_b_dtypes.py`, 100 cases:
    - Float32 input × TILE / RM × {γβ f32 RM, γβ f32 TILE, γβ bf16 TILE, γ f32 RM, none} on 4 shapes, including `C/G = 4`.
    - bf16 input × {γβ bf16 TILE, γβ f32 RM, γβ f32 TILE, γ bf16 TILE} on 6 shapes, including odd `C/G = 3` (halfword path) and `C/G = 56`.
    - Float32 odd `C/G`.
    - Float32 stream-only / partial residency.
    - Float32 multi-unit per core (`max_cores = 4`).
    - Float32 wide (64×4096) and deep (16384×512).
  - `_op_bench.py`: `test_bench_formats` (Float32 / TILE γ/β / Float32 γ/β cases). `test_bench` now shares a `_bench` helper.

## Refinement 5 — hw_non_aligned (HW % 32 ≠ 0)
- Date: 2026-11-26
- What was done:
  - **Registry.**
    - `SUPPORTED["alignment"]` gains `"hw_non_aligned"`.
    - A new `EXCLUSION` refuses `{memory_layout: BLOCK_SHARDED, alignment: hw_non_aligned}`. The refinement's scope is DRAM interleaved, feature_spec has no block-sharded `HW % 32 ≠ 0` entry, and the sharded planner already refuses TILE images that are not whole tile rows.
  - **Mechanism: reader k-fill instead of `ReducePartialScaler::with_partial()`. This deviates from the verifier note.**
    - The design's partial-scaler plan predates Perf 1. Pass 1's moments are now a raw FPU DEST accumulation of `S1' = Σ(x − k)` and `T2 = Σ x·(x − k)`, with `k` the per-group pilot. No reduce helper is left to take a partial scaler.
    - Instead, the reader overwrites the pad rows `HR = HW % 32 .. 31` of the pass-1 copy of each image's last tile row with `k`. Then `x − k = 0` and `x·0 = 0`, so each pad row adds exactly 0 to both moments.
    - This is the mechanism R3 already uses for the sharded-RM staged rows. Compute does not change, so there is no new reduce variant and no `--dev` kernel-text growth on the compute side.
  - **Descriptor.**
    - `plan_or_raise` uses `Ht = ceil(HW/32)` per image. It is the row unit of the `n × xs × ht` split, so N > 1 keeps per-image padding. The TILE page id stays `(n·HT + ht)·WT + wt`.
    - `inv_n = 1/(HW·Cg)` already used the true `HW`.
    - New reader CT arg 20 is `HW`; the `TensorAccessorArgs` chain now starts at 21.
    - The writer walks `HT = ceil` tile rows per image, writing the output's pad rows too: they are the output tensor's TILE padding.
  - **Reader (same file).**
    - `HR = HW % 32` is a constexpr.
    - The k extraction moved into an `extract_kv` lambda. For `HR ≠ 0` it runs before chunk 0's push, because chunk 0 may already hold an image's last tile row.
    - `k_fill_last_row(chunk_base, lr)` builds the k row in `cb_affine_staging`, in the input's format. That buffer is free from the pilot extraction until the affine reads, which are issued after the last pass-1 push. The lambda then issues core-local NoC writes and a write barrier before the push:
      - RM: one write per pad stick.
      - TILE: one `16·in_elem` B face row per pad row, per face pair, per tile, via `noc_async_write_one_packet_with_state`.
    - The fill runs on pass-1 steps only (`step < num_chunks`), and only on the chunk where `row0 + rows_valid == HT`. A pass-2 re-read of that chunk needs no fill: its pad rows become output pad rows only.
    - The RM issue loop reads `rows_valid·32 − (32 − HR)` sticks on that chunk. Stick paging is `n·HW + row`, so reading past `HW` would read the next image. The pad sticks are never read from DRAM.
    - With `HR = 0` every new site is `if constexpr`-dead, so the aligned reader is the R4 reader apart from the CT index shift.
  - **Compute and writer: no change.**
- Reused: R3's k-fill idea (pad rows ← pilot `k`) and `cb_affine_staging` as the k-row source; the pilot `kv` and `for_each_run` lane walk; the existing chunk / residency / pass-2 re-read structure; and the golden helper `run_groupnorm_sc_N_1_HW_C` / `check_output` for the tests.
- Added: reader CT 20 (`HW`), `extract_kv` (a moved block) and `k_fill_last_row`, the RM short-last-row stick count, `Ht = ceil` in the plan, and the `BLOCK_SHARDED × hw_non_aligned` EXCLUSION. No new CB, no new kernel file.
- Accuracy achieved (Blackhole, HiFi3, fp32 DEST, γ+β RM in the input dtype; probe_025):

  | Shape | Groups | bf16 PCC | bf16 relative RMS | Float32 PCC | Float32 relative RMS |
  |---|---|---|---|---|---|
  | (1,1,17,64) | g1 | 0.999997 | 2.42e-3 | 0.999999 | 1.77e-3 |
  | (1,1,47,256) | g1 | 0.999998 | 2.29e-3 | 0.999999 | 1.76e-3 |
  | (2,1,100,128) | g1 | 0.999998 | 1.88e-3 | 0.999999 | 1.43e-3 |
  | Mochi (4,1,6360,768) | g32 | 0.999998 | 1.94e-3 | 0.999999 | 1.35e-3 |

  TILE and RM give identical metrics on every shape.
- Golden test progress:
  - **180/180** `alignment=hw_non_aligned` cells pass. That is every valid cell: DRAM interleaved, bf16 / Float32 input, TILE / RM, and every γ/β form, including Mochi.
  - Plus **32/32** tile-aligned DRAM representatives: 4×2048×256 and 4096×320, gamma_beta, all formats.
  - Run as node ids with `--precompile --run-all`.
  - Unit tests:
    - New file: 62/62 plain and `--dev`.
    - The whole unit directory: 261/261 plain.
    - Spec, `_dtypes`, `_sharded` and `_ragged`: 192/192 under `--dev`, so the reader text still fits the watcher build's kernel-config ring.
- Sensitivity check: with the fill disabled (`if (false)`), 16/16 targeted cases fail. PCC drops to 0.10–0.98, and output is Inf with NaN padding. The fill is what carries correctness.
- Perf (Blackhole, DEVICE KERNEL DURATION [ns], one fresh-cache profiled run per case, `_op_bench.py`):

  | Case | ns | Comparison |
  |---|---|---|
  | mochi_hwna_4x6360x768_tile | 223 098 | aligned twin 4×6368×768: 224 482 (−0.6%) |
  | mochi_hwna_4x6360x768_rm | 228 659 | aligned twin: 230 395 (−0.8%) |
  | mid_4096x640_tile | 36 633 | R4/Perf 2: 36 623 (flat) |
  | mid_4096x640_rm | 44 502 | R4: 44 228 (+0.6%, noise) |
  | mid_16384x320_tile | 65 466 | R4: 66 182 (−1.1%) |
  | mid_16384x320_rm | 76 750 | Perf 2: 77 749 (−1.3%) |

  - The k-fill costs nothing measurable: the non-aligned Mochi shape tracks its aligned twin by bytes moved, on 108 cores.
  - The fill is bounded by `(32 − HR)·W·2` face-row writes per block. It sits behind the chunk's read barrier and is a small fraction of one chunk's NoC time.
- Issues encountered:
  - None in the kernel: the first run of the new tests passed 62/62.
  - The first golden-id extraction found 0 ids, because `--collect-only -q` prints `<Function test_op[...]>` lines rather than node ids. Parsing those lines fixed it.
- Tests added:
  - `tests/.../test_groupnorm_sc_N_1_HW_C_b_hw_non_aligned.py`, 62 cases:
    - The 5 golden hw_non_aligned shapes plus HR edges (HR = 1, 16, 31, and HW = 5 < 16), Cg = 10 / 20, and N > 1. Each runs on TILE / RM × bf16 / Float32.
    - γ-only and no-affine.
    - TILE inputs whose pad rows hold 1e4 or NaN, injected through a `ttnn.reshape(logical, padded)` view.
    - An offset-heavy input (`randn + 3`).
    - Stream-only and partial residency (`K < B`, with the k-filled chunk re-read in pass 2).
    - Several units per core (`max_cores = 4`).
  - `_op_bench.py`: `test_bench_hw_non_aligned`, covering Mochi 6360 and its 6368 twin on TILE and RM.

## Refinement 6 — c_non_aligned (C % 32 ≠ 0)
- Date: 2026-11-26
- What was done:
  - **Registry.**
    - `SUPPORTED["alignment"]` gains `"c_non_aligned"`. `tag_alignment` already returns it for any `C % 32 ≠ 0`, including when HW is non-aligned too.
    - A new `EXCLUSION` refuses `{memory_layout: BLOCK_SHARDED, alignment: c_non_aligned}`. The refinement's scope is DRAM interleaved, and feature_spec has no block-sharded `C % 32 ≠ 0` INPUTS entry.
  - **Planner (`plan_or_raise` / `_evaluate_gx`).**
    - `S = ceil(C / lcm(Cg, 32))` spans. The old `C % L ≠ 0` refusal is gone.
    - `_choose_gx` still cuts S into gx equal runs of `S/gx` spans. Every slice therefore starts on a span boundary, which is a tile boundary and a group boundary: the verifier's `_choose_gx` constraint holds by construction.
    - `W = min((S/gx)·span_tiles, Wt)` (`_evaluate_gx` gains `Wt`). The slice of a single-slice row is clamped to the row's real tile count.
    - Only a row's last slice is partial: `vc_last = C − (gx−1)·W·32` valid channels in `ceil(vc_last/32)` tiles. It is a new plan field; `col_mask` becomes `vc_last < 32·W`.
    - For tile-aligned C, `C % lcm(Cg, 32) = 0`, so `S`, `W` and every buffer size are unchanged.
  - **Mechanism: compute's lane select, not the map's channel bound. This deviates from the verifier note.**
    - TILE pad channels are unspecified and may be NaN, and the group map is a matmul: `NaN·0 = NaN`. The map's channel bound therefore cannot keep a pad channel out of a group.
    - Instead compute reuses R3's `COL_MASK`, an SFPU `v_if` select. After pass 1's column collapse it zeroes the S1′ and T2 lanes at positions `≥ vc`, before the collapse matmul.
    - The map is still built over all `P` lanes. Its pad lanes now multiply exact zeros.
    - Compute decides per block whether this is a last slice: `(unit_start + blk) % GX == GX − 1`. There is a new compute RT arg (`unit_start`) and CT args 14/15 (`GX`, `VC_LAST`). The streaming `moments_rows` takes the block's `w_blk` / `vc_blk`.
  - **Reader.** New CT arg 21 is `VC_LAST`; `TensorAccessorArgs` now start at 22, and the unused `G_CORE` is removed. Per block, `vc`, `w_blk = ceil(vc/32)` and `g_blk = vc/Cg` bound the following:
    - TILE reads: `w_blk` tiles per row. RM reads: the stick's `vc·in_elem` valid bytes, from the accessor's aligned page start, so the source alignment is unchanged.
    - The pilot read (RM `vc·in_elem` B, TILE `w_blk` tiles).
    - `extract_kv(g_blk)`, `build_shift_rows(..., vc, g_blk, ...)`, and R5's `k_fill_last_row` (`vc` lanes, `w_blk` tiles).
    - `build_affine` / `issue_affine_reads(c_start, vc)`. The RM γ/β staging reads `align_up(vc·aff_elem, 64)` B inside the tensor's aligned page.
    - `write_b_chunk` writes `w_blk` tiles per row of the reader's pass-2 output share.
    - The ring layout keeps the `W`-tile row pitch, so the partial slice fills a prefix of each row.
  - **Writer.** CT 15 is `VC_LAST` (`TensorAccessorArgs` at 16). It writes `W_LAST = ceil(VC_LAST/32)` tiles per row on a last slice.
  - **No new CB, and no CB resized.**
- Reused: R3's `COL_MASK` lane select and its `vc` RT plumbing; `moments_rows`' `w_valid` / `vc` parameters; the `for_each_run` lane walk bounded by `vc`; R5's per-block `extract_kv` / `k_fill_last_row` structure; the existing chunk / residency / pass-2 re-read structure; and the golden helpers `run_groupnorm_sc_N_1_HW_C` / `check_output` for the tests.
- Added: `S = ceil` and `W = min(…, Wt)` in the planner; the `vc_last` plan field; the `VC_LAST` CT arg in all three kernels (reader 21, compute 15, writer 15) and compute's `GX` CT 14; compute's `unit_start` RT arg; the per-block `vc` / `w_blk` / `g_blk` bounds; and the `BLOCK_SHARDED × c_non_aligned` EXCLUSION.
- Accuracy achieved (Blackhole, HiFi3, fp32 DEST, γ+β in the input dtype; golden c_non_aligned shapes × TILE / RM):

  | dtype | min PCC | max relative RMS | worst shape |
  |---|---|---|---|
  | bf16 | 0.999991 | 4.58e-3 | — |
  | Float32 | 0.999991 | 4.80e-3 | (2,1,64,47) g1 |

  - Shapes: (1,1,64,17) g1, (1,1,64,50) g1, (1,1,128,100) g1, (2,1,64,47) g1, (1,1,64,48) g2, (1,1,128,48) g3, (1,1,64,80) g4, (1,1,128,144) g4, (1,1,64,200) g8, (1,1,256,330) g33 with gx = 3, and (2,1,96,6) g3.
  - Float32 TILE is 1.1e-3 – 2.3e-3 per shape, the same band as the aligned twins (1.1e-3 – 2.1e-3).
  - TILE (zero pad) and RM (stale ring pad) outputs are bit-identical. Pad columns filled with 1e4 or NaN pass. Nothing leaks from the pad.
- Golden test progress:
  - **396/396** in-scope `alignment=c_non_aligned` cells pass: DRAM interleaved, bf16 / Float32 input, TILE / RM, every γ/β form, `in_place=False`. The other 792 c_non_aligned cells are INVALID-skipped.
  - Plus **64/64** DRAM representatives: 64×64 g2, 50×128 g1, 2×100×128 g1 and 4096×320 g32, gamma_beta, all non-bf8 formats.
  - The slice ran as one `-k` selection with `--precompile --run-all`: 460 passed, 872 skipped, 0 failed, 0 XPASS.
  - Unit tests:
    - New file: 80/80 plain.
    - The whole unit directory: 341/341 plain.
    - The c_non_aligned, sharded, ragged and hw_non_aligned files: 190/190 under `--dev`. The kernel text still fits the watcher build's 70 688 B kernel-config ring, and the NoC sanitizer is clean on the short RM reads.
- Perf (Blackhole, DEVICE KERNEL DURATION [ns], `_op_bench.py`). The non-aligned shape and its aligned twin share Cg and Wt, so they have the same spans, slices and grid. Median of 3 fresh-cache profiled runs, on 110 cores:

  | Case | non-aligned | aligned twin | Δ |
  |---|---|---|---|
  | 16384×300 g30 vs 16384×320 g32, TILE | 66 040 | 65 377 | +1.0% |
  | 16384×300 g30 vs 16384×320 g32, RM | 78 330 | 77 439 | +1.2% |
  | 65536×100 g1 vs 65536×128 g1, TILE | 102 775 | 102 557 | +0.2% |
  | 65536×100 g1 vs 65536×128 g1, RM | 151 299 | 150 957 | +0.2% |

  - Every delta is inside the run-to-run spread (about 1%). The partial slice's trimmed reads and writes, and the one `COL_MASK` select per block, cost nothing measurable.
  - Aligned DRAM cases are within noise of R5: `mid_4096x640` 35 811 / 44 431 on TILE / RM, and `mid_16384x320` 65 046 / 77 519.
  - Finding: RM 65536×100 moves 22% fewer bytes than its 128-channel twin but takes the same time. At that shape RM is bound by the per-stick transaction count (65 536 sticks) and the tilize/untilize, not by bytes. This is a pointer for Refinement 10 (DRAM streaming), not an R6 cost.
- Issues encountered:
  - None in the kernels: every new test passed on its first run.
  - Float32 precision depends on the pilot. Float32 error tracks `|k − mean|`, where the pilot `k` is one sample per group: image row 0, the group's first channel. With `k` forced to +2.84, the per-seed relative RMS is 3.7e-3 – 5.6e-3 for aligned C = 32 / 64 / 96 / 128, and 5.7e-3 – 5.9e-3 for non-aligned C = 47 / 50 / 100, with a maximum of 9.5e-3. This is the pre-existing shifted-moment sensitivity: Float32 x reaches srcA as TF32. It is not specific to R6, and it stays within the 0.01 limit. Golden data is deterministic, because the root conftest's autouse `reset_seeds` sets `torch.manual_seed(213919)`, so those cells are not flaky. Possible lever: a pilot taken over several samples.
  - Tooling:
    - A pytest `-k` with `=` in it is a parse error, so the golden slice selects on id fragments instead: `not True` for `in_place=False`, plus `INTERLEAVED` and `gamma_beta`.
    - `run_safe_pytest.sh --profile` passes a spaced `-k` through Tracy split into separate words, so the run collects nothing and still reports PASS. Profile with node ids instead.
- Tests added:
  - `tests/.../test_groupnorm_sc_N_1_HW_C_b_c_non_aligned.py`, 80 cases:
    - The 9 golden c_non_aligned shapes, plus 256×330 g33, where gx = 3 and the last slice is 1 of 5 tiles, plus 2×96×6 g3, where C < 16. Each runs on TILE / RM × bf16 / Float32.
    - Three affine forms on 256×330: bf16 TILE γ/β, Float32 RM γ-only, and no affine.
    - HW and C both non-aligned: 2×50×100 g4.
    - TILE inputs whose pad columns hold 1e4 or NaN, injected through a `ttnn.reshape(logical, padded)` view.
    - An offset-heavy input (`randn + 3`).
    - Stream-only and partial residency (`K < B`, with the partial slice's chunks re-read in pass 2).
    - Several units per core (`max_cores = 4`), with full and partial slices alternating on one core.
  - `_op_bench.py`: `test_bench_c_non_aligned`, covering 16384×300 g30 and 65536×100 g1 against their aligned twins on TILE and RM.
  - Probes 026–032: the R6 bring-up and precision investigation.

## Refinement 7 — bfloat8_b input / gamma / beta (TILE)
- Date: 2026-11-26
- What was done:
  - **SUPPORTED.** `ttnn.bfloat8_b` was added to `SUPPORTED["dtype"]` (TILE, DRAM interleaved; output dtype = bfloat8_b) and to `SUPPORTED["affine_dtype"]` (TILE γ/β, under bf16 / Float32 / bfloat8_b input).
    - Two new EXCLUSIONS refuse a direct caller's BLOCK_SHARDED bfloat8_b input or γ/β. These cells are scope-parked INVALID in feature_spec, and the sharded pilot / staging paths do bf16 byte math.
    - bfloat8_b + ROW_MAJOR is structural INVALID: a bfloat8_b tensor is TILE.
    - No `bf8b × non-aligned` exclusion was needed, contrary to the skill's usual expectation. TILE pad rows and lanes reach the op as zeros (from_torch padding) or are k-filled, so they never raise a valid face row's shared exponent. The output's pad lanes are `x·0 − 0 = 0`.
  - **Descriptor (`Formats`, still the single size source).**
    - `in_bfp8` / `aff_bfp8` flags.
    - `amap_dtype`: bf16 for bfloat8_b γ/β, else γ/β's own dtype. The maps are decoded to bf16, because a map tile carries one value per column in a different row per group, and a shared-exponent face row cannot hold those exactly. A bf8b value has ≤ 7 significant bits, so the decode is exact.
    - `pilot_tile_stage` / `aff_tile_stage`: 192 B per tile for bfloat8_b (faces 0/1 row-0 runs + the 64 B exponent section), 128 B otherwise.
    - `staging_bytes` / `affine_stage_bytes` / `cb_gamma_rows` derive from those properties.
    - `in_tile = ttnn.tile_size(bfloat8_b) = 1 088 B` flows into the ring, output shares, `K_max` and the footprint unchanged.
    - New reader CT args 22–23: `IN_BFP8`, `AFF_BFP8`. `TensorAccessorArgs` now start at 24.
  - **Reader.** Two helpers: `bfp8_to_bf16` (tt-metal's host decode) and `bf16_to_bfp8` (truncating encode at a given shared exponent). Bfp8_b tile layout: a 64 B exponent section, one byte per face row, face-major, then 1 024 sign|7-bit-magnitude bytes, face-major.
    - **Pilot (TILE).** Per tile it reads the faces 0/1 row-0 runs (64 B at `64 + f·256`) and the exponent section (64 B at 0). `extract_kv` decodes each group's first-channel sample with its face row's exponent.
    - **k-fill (`hw_non_aligned`).** A pad face row's 16 lanes may span several groups (C/G < 16) but share one exponent. So `extract_kv` truncates every `k` to one common exponent `k_exp` (the largest). The quantized `k'` is then used everywhere (pilot rows, group shift, pad rows). `k` is only a shift, so the moments stay exact.
    - Each pad face row is written as 16 mantissa bytes, a core-local NoC write as before, at the bf8b face offsets. Its shared-exponent byte is a reader byte store: two per pad row per tile, a run no NoC write can align to.
    - **γ/β (TILE, bfloat8_b).** Three 64 B reads per channel tile, decoded in place to bf16 after the read barrier: each face's 16 mantissas are loaded, then stored as 16 bf16 values over the same run. The existing word-copy / halfword map build then runs unchanged with `AFF_ELEM = 2`.
  - **Compute and writer: unchanged.** The moments' whole-tile srcA `UNPACR` and pass 2's `llk_unpack_AB<ROW>` unpack Bfp8_b x into bf16 srcA. The output pack forms Bfp8_b face rows from fp32 DEST. Every format reconfig is CB-driven.
  - **No new CB, no CB resized** except by the format-derived page sizes.
- Reused:
  - R4's `Formats` record and dtype-derived page sizes.
  - R5's `extract_kv` / `k_fill_last_row` structure and its face-row NoC writes.
  - R6's `vc` / `w_blk` bounds.
  - The `build_affine` word-copy / halfword paths (fed decoded bf16).
  - The whole compute and writer.
  - The golden helpers for the tests.
- Added:
  - The bfloat8_b decode/encode helpers.
  - The `IN_BFP8` / `AFF_BFP8` CT args and branches: pilot exponent read, staged-γ/β decode, common-exponent k quantization, k-fill exponent stores.
  - The `Formats` bf8b properties, the SUPPORTED values and the two EXCLUSIONS.
- Accuracy achieved (Blackhole, HiFi3, fp32 DEST; golden-style `randn` input, γ+β bfloat8_b TILE):
  - PCC ≥ 0.99991 and relative RMS 0.0125–0.0130 on (1,1,256,256) g8, (2,1,512,128) g4, (1,1,64,1280) g32, (2,1,100,128) g4 [hw_non_aligned], (1,1,64,100) g2 [c_non_aligned], Mochi (4,1,6360,768) g32 and (1,1,16384,512) g32. The tolerance is `TOLERANCES[bf8b]` = (0.99, 0.10).
  - The remaining error is the input's and output's own bfloat8_b rounding: 7-bit magnitudes under a shared exponent.
  - **The op's own error**: the reference is computed from the device's round-tripped bfloat8_b input, with group offsets from −12 to +12, so the k's have very different exponents. Relative RMS is 0.0086 – 0.0088 on hw_non_aligned shapes (2,1,40,256) g32 and (1,1,47,320) g32, against 0.0093 on the aligned (2,1,64,256) g16. That is the same band, so the common-exponent k-fill adds nothing.
- Golden test progress:
  - **1539/1539** runnable bfloat8_b cells pass:
    - 891 with bfloat8_b input: 715 tile_aligned, 55 hw_non_aligned, 121 c_non_aligned.
    - 648 with bfloat8_b γ/β under bf16 / Float32 input, TILE + RM.
    - The rest of the 23 544 collected bf8b ids are INVALID-skipped: RM bf8b, BLOCK_SHARDED bf8b, `in_place=True` on interleaved input, and affine/none mismatches.
    - The slice ran as node ids in three `--precompile --run-all` chunks: 515 + 512 + 512 passed, 0 failed, 0 XPASS.
  - Prior cells: a random sample of 48 non-bf8b golden cells (seed 7) passes 48/48.
  - Unit tests:
    - The whole unit directory passes 445/445 plain: 341 prior + 104 new.
    - The new bf8b file also passes 104/104 under `--dev`: the watcher NoC sanitizer is clean on the 1 088 B pages and the exponent-byte stores, and the kernel text fits.
- Perf (Blackhole, DEVICE KERNEL DURATION [ns], `_op_bench_bf8b.py`). One fresh-cache profiled run; γ+β TILE in the input dtype; 108–110 cores both ways:

  | Case | bf16 | bfloat8_b | ratio |
  |---|---|---|---|
  | vae_dram_65536x512 | 334 621 | 188 272 | 0.56 |
  | vae_dram_262144x256 | 703 555 | 349 070 | 0.50 |
  | refiner_dram_16384x1152 | 198 833 | 116 897 | 0.59 |
  | mid_16384x320 | 65 888 | 44 466 | 0.67 |
  | mochi 4×6360×768 (hw_non_aligned) | 223 705 | 155 722 | 0.70 |

  - bfloat8_b moves 0.53× the input / output bytes. Its larger `K_max` also removes pass-2 re-reads: `vae_dram_262144x256` goes from K 16 / B 19 to fully resident, hence 0.50×.
  - bf16 / Float32 paths are compile-time identical apart from the CT-arg shift, so the DRAM PERF FOCUS cases (bf16) are untouched.
- Issues encountered: None. Every bf8b case passed on its first run, and the pilot/k-fill decode matched the host format on the first try.
- Tests added:
  - `tests/.../test_groupnorm_sc_N_1_HW_C_b_bf8b.py`, 104 cases:
    - bfloat8_b input × 12 shapes (aligned, odd C/G, hw_non_aligned HR edges with C/G < 16, c_non_aligned) × 5 γ/β forms.
    - bfloat8_b γ/β under bf16 / Float32 input × TILE / RM × gamma_beta / gamma_only.
    - A tight op-error check against the quantized-input reference with offset groups: (0.9995, 0.03).
    - Stream-only and partial residency, aligned and hw_non_aligned.
    - Several units per core.
    - Large shapes: 64×4096, 16384×512 and Mochi.
  - `_op_bench_bf8b.py`: bfloat8_b vs bf16 device time on five DRAM shapes.
  - Probes: the R7 bring-up (first-cut six-case smoke, accuracy metrics, residency plans).
