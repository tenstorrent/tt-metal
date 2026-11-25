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
