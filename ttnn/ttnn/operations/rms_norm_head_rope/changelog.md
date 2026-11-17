# Changelog: rms_norm_head_rope

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], alignment=[tile_aligned], rank=[4],
  affine=[no_affine, weight] (bf16 TILE), rope=[no_rope, rope_bcast] (bf16), memory_layout=[INTERLEAVED],
  fp32_dest_acc_en=[True, False]. SUPPORTED ⊇ TARGET (no generality gap).
- **Accuracy achieved** (default config, after verifier fix): weight+rope PCC=0.999999, max_abs_err=0.091,
  rel_rms_err=0.0019, ULP mean 0.33; plain PCC=0.999999, max_abs_err=0.017, rel_rms_err=0.0017, ULP mean 0.25
  (measured on 4 shapes up to 2304×3072 via test_rms_norm_head_rope_precision_baseline.py)
- **Golden suite at Phase 0**: 32 / 32 supported cells passing, 224 INVALID skipped, 0 loud (per `verifier_report.json`)
- **Perf (weight+rope, BH device kernel)**: 256×3072 20.0 µs, 2048×3072 ~115 µs, 2304×3072 ~119 µs
- **Issues encountered / verifier fixes**:
  - FPU operand placement: the weight DEST-reuse (`DEST_TO_SRCA` → `DEST_TO_SRCB`) and both RoPE multiplies
    (operands swapped) now keep bf16 on srcA and DEST-width values on srcB. Removed a −0.36% output bias
    (fp32 DEST, weight+rope: rel RMS 0.0044 → 0.0019); perf-neutral.
  - `select_grid_split` raises `ValueError` instead of returning None (H > 24·cores).
  - Removed the unused `device` param of `_check_aux`; documented `cb_sumsq` wrap-safety; ledger gained the
    `cb_sumsq` ↔ `cb_normed`/`cb_rotated` sharing justification and the page-format audit notes.
  - Collapsed knob `BLOCK_ROW_TILES` (host-only, asserted 1) → filed inside Refinement 1.
- **Tests added**: test_rms_norm_head_rope.py (planner), test_rms_norm_head_rope_perf.py (implementer),
  test_rms_norm_head_rope_precision_baseline.py, test_rms_norm_head_rope_tail_blocks.py (verifier)

## Phase 0 — implementer notes

Built per op_design.md: R1 regime (row-group x head-group split via `select_grid_split`), reader / compute / writer
as in the Block Operation Realization table. Acceptance 48/48 (plain + `--dev`), golden 50 passed / 225 skipped (INVALID).

Deviations (advisory items):
- Tail-block CB handling: full-`BT` push/pop on reader/writer CBs + realign of compute-internal CBs (wrap safety).
- Optional tensors absent → input's TensorAccessorArgs used as the CT placeholder (no no-arg binding on this branch).
- `rotate_pairs_block` is raw matmul (no matmul helper on this branch), init once per block.

### Perf (Blackhole, device kernel duration, weight + rope, HiFi4 fp32-acc)

| shape | Phase 0 | reference op | roofline |
|---|---|---|---|
| 256x3072  | 19.8 us | 99.0 | 8.3 |
| 2048x3072 | ~114 us | 129.8 | 58.1 |
| 2304x3072 | ~120 us | 137.5 | 65.2 |

Attribution at 2304x3072 (ablations, all with CB handshakes kept):
- compute only (NoC payload stubbed): plain 24 / +w 40 / +rope 59 / w+rope 76 us — HiFi4 FPU muls (~6 per tile) dominate.
- DM only (compute stubbed): plain 76, w+rope 89 us; reads alone 55 us (~290 GB/s), writes alone 56 us (~250 GB/s).
- reads+compute 106 us, writes+compute 77 us; full 120 us. The binding resource is aggregate DRAM/NoC efficiency
  (well below the 460 GB/s roofline), not block fill/drain: `block_heads` in {1,2,4,8} measured flat (119–122 us).

Levers kept:
- Chain DEST blocking (normalize, rope): compute-only 78.8 → 76.2 us.
- Writer chunk streaming (`STORE_CHUNK_HEADS = 1`): flat at 2304 (120.1 vs 119.7), 256: 20.2 → 19.8 us. Kept as a knob.

Findings for the next refinement: the DRAM read/write efficiency (~290/250 GB/s isolated) is what has to move — NoC
placement / core ordering (`noc_placement`), more reads in flight across block boundaries (no per-block barrier
drain), and per-core balance (max 16 vs avg 14.4 (row, head) units at G=3, Rg=40). Compute next: RoPE reformulation
perf lamp (fold rot*sin into a matmul-accumulate) and the weight DestReuse (+16 us compute).

## Perf 1

- **Date**: 2026-11-17. Blackhole, 12x10 worker grid, device kernel duration (`DEVICE KERNEL DURATION [ns]`).
- **Focus shape**: `LOOSE_CASES` is empty (no perf-flagged entry), so I picked 2304x3072 weight+rope (FLUX joint stream,
  96 calls/step, the dominant shape) at the default config (bf16, TILE, DRAM interleaved, HiFi4, fp32_dest_acc_en=True).
  Only cells over 32 tiles were measured. Noise band is ±2–3%.
- **Instrumentation (permanent)**: `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`, an alias of
  `DeviceZoneScopedN`, free when the profiler is off). Each stage has separate wait / issue / barrier / work zones:
  reader `reader_constants|reader_weight|reader_reserve_rope|reader_issue_rope|reader_reserve_x|reader_claim|reader_issue_x|reader_barrier`;
  compute `compute_weight_expand|compute_wait_rope|compute_wait_x|compute_sumsq|compute_rsqrt|compute_normalize|compute_rope`;
  writer `writer_constants|writer_wait|writer_issue|writer_flush|writer_barrier`.
  Tooling: `perf_experiments/zone_report.py` (per-op, per-zone max/p50 across cores). Bench files:
  `test_rms_norm_head_rope_bench.py` (guard set: 6 shapes x 4 variants), `_ablate.py`, `_small.py`.

### Measured breakdown (round start, focus shape)

Cumulative ablation (payload stubbed, sync kept):

| variant | ns |
|---|---|
| full | 120,424 |
| compute stubbed (DM only) | 90,092 |
| compute + reads stubbed (writes only) | 52,690 |
| compute + writes stubbed (reads only) | 57,153 |
| everything stubbed (sync floor) | 7,564 |
| DM stubbed (compute only) | 56,247 |
| reads stubbed (compute + writes) | 73,208 |
| writes stubbed (compute + reads) | 104,652 |

- **Reads and compute serialize.** reads + compute = 104.7 µs, against 57 µs and 56 µs separately. Writes overlap with
  compute fine (73 µs).
- **Why**: kernel duration per core is max 118 µs against p50 80 µs, a 1.47× imbalance. The per-core zone map shows
  start-up DRAM starvation:
  - `reader_constants` (the weight prologue) takes 4.5 µs on the bottom grid rows but 60.7 µs on the top rows (y=2,3).
  - In the rope-only variant the same starvation shows up in `reader_issue_x` / `reader_barrier` (max 52/63 µs).
  - NoC0 serves DRAM reads in core-y order, so the top rows get their first block at 50–60 µs and only then run their
    full ~56 µs of compute.
- **Compute (TRISC1 math, per core, 2 blocks of 32 tiles)**: rope 18.6 µs, rsqrt 15.5 µs (about 1 µs/tile, although
  only column 0 of each rstd tile is consumed), normalize 11.4 µs, sum-of-squares ~7 µs (includes stall),
  rotate 3.3 µs.
- **Ceiling**: the DM roofline is 65.2 µs at 460 GB/s (`op_design.md`). DM-only measured 90.1 µs, so DM is not at its
  roofline. Compute is not at its roofline either (rsqrt runs SFPU on 32× more elements than needed).
- **Ranked bottleneck**:
  1. Read/compute serialization caused by start-up starvation (~30 µs between full and max(DM, compute)).
  2. Compute payload (rope, rsqrt, normalize).
  3. DM efficiency (90 vs 65 µs).

### Portfolio and per-idea verdicts (subagent measurements, before → after ns)

| idea | verdict | focus-shape result | domain / notes |
|---|---|---|---|
| `rsqrt_stage`: one packed-column eps+rsqrt per block | **WIN** | stage 15,545 → ~3,200 math; whole op about −3% (noise edge, because the op was DM-bound) | Applies everywhere `block_heads ≤ 16`. Op output bit-identical on 8 shapes. Menu: v1 (helper + `VectorMode::C`) 8.2 µs; v3 (`add_rsqrt_tile`, BH-only) 8.3 µs; v7 packed 3.2 µs. The `fast_and_approx` variants were not taken. |
| `rope_single_dest`: matmul y@T into DEST, DEST-reuse ×sin, ELWMUL-accumulate cos·y | **WIN** | rotate+rope 21,842 → 15,000 math; whole op 120,319 → 115,585 | Applies everywhere rope applies. With `fp32_dest_acc_en=False`, rel RMS goes 0.0050 → 0.0060, still far inside tolerance. **Option recorded, not graduated:** running the rotation matmul at HiFi3 (bit-identical, since T = ±1) touches `math_fidelity`, which is part of the precision contract. |
| `normalize_fusion` (L4: all-FPU tail using linearity) | WIN, borderline → **superseded** | 120,927 → 117,494 (median of 3) | Same tail-math saving as `rope_single_dest`, but worse precision (rel RMS 0.00222 → 0.00261), a weight-only regression (+3.2% at 256x3072), and more CBs. `rope_single_dest` graduated instead. Its finding that a DEST→srcB reuse takes srcB's current format is honored in the graduated rope path (srcB is reconfigured from the fp32 CB first). |
| `read_pipeline`: per-head-group dynamic row claiming + NoC swap for the starved top rows (writer twin: cadence) | **WIN** | 119.6k → 93.3k on the old compute | Nulls inside it: `block_heads` 1/2/4 (flat to worse), weight-after-x alone (NULL), writer end-only barrier (NULL), column-wise core order (NULL), G=4/6/8 (NULL/worse). |
| `reuse_mcast`: cos/sin mcast over the G head-group cores + weight column mcast | **WIN** (not graduated this round) | 120,367 → 104,281 | cos/sin mcast **conflicts** with dynamic row claiming: the G cores of a row-group no longer process the same rows. Weight column mcast is compatible but needs G to divide grid.x and full grid rows. Whole-grid weight mcast measured a regression on small shapes (1.08–1.67×). |

### What graduated

1. **Fused single-DEST RoPE**, as the only rope path. Deleted: the `apply_rope` eltwise_chain, the separate rotate
   pass, and `cb_rotated` (frees `block_tiles` × 4 KB of L1 per core).
2. **Packed-column rsqrt**, as the only rsqrt path, for every variant. Deleted: the `reduce<SUM, REDUCE_ROW>` call,
   the `prepare_reduce_scaler` reader code, and the scaler CB.
   - New: `cb_colsel` (5) and `cb_colext` (10), `block_heads` bf16 pages each, generated by the writer (idle at start),
     plus `cb_rstd_packed` (11), 1 DEST-width page. `cb_sumsq` is now realigned on tail blocks.
   - The SFPU pass uses the portable `add_unary_tile` + `rsqrt_tile<Default>` with `VectorMode::C` rather than the
     BH-only experimental `add_rsqrt_tile`. It runs on 1 tile per block, so the difference is negligible and the
     kernel stays arch-portable.
   - Rope + rsqrt commit: focus 119,736 → 111,670; 2048 w+rope 114,154 → 101,917; no guard-set regression.
3. **Dynamic row claiming + NoC swap**.
   - Each core keeps a static first row-tile. Further rows of its head group come from a per-group fetch-and-add
     counter (program semaphore, initial value Rg) and are published to compute/writer through `cb_ctrl_compute` (12)
     and `cb_ctrl_writer` (13).
   - The top 2 grid rows read on NOC_1 and write on NOC_0 (two DM kernel groups, still one dispatch).
   - The reader barriers every 8 x tiles.
   - Corrected against the subagent drop-in, by measurement on the new compute:
     - **INPUT_DEPTH stays 2**. Depth 1, medians of 3: 2048 plain 80.1 vs 74.2 µs and 2048 weight-only 84.5 vs
       76.5 µs regressed, while the focus difference was in noise (91.5 vs 93.3 µs).
     - **Weight-first prologue kept**. The drop-in's lazy weight read/expand made 256x1024 w+rope +9%, and with the
       swap on it bought nothing at the focus shape.
   - **Carve-outs (each earned by a measured regression):**
     - No NoC swap when fewer than 8 grid rows are used. On such grids every core sits in the swapped rows, so all
       reads move to NoC1. Measured: 128x512 +27%, 32x3072 w+rope +47%, 256x1024 w+rope about +10%. Measured wins at
       8 rows (256x3072) and 10 rows (focus, −6%).
     - The claim and the ctrl-CB handshake are compiled out (`dynamic_rows = 0`) when Rg ≥ Rt. Every row is then some
       core's static row, so claiming has no meaning. Measured +3–4% on such grids before the carve-out, +1.3–3.1%
       (noise band) after.
   - Read-pipeline commit (medians of 3, interleaved A/B against the previous commit): focus 110,412 → 93,799
     (−15%); 2304 plain 77,449 → 74,899.
   - Verified: correct over 3 back-to-back calls (program-cache hit) at 2304x3072 and 1024x2048, and the guard set
     passes under `--dev` (watcher).

### Whole-op before → after (guard set: every measured shape × {plain, weight, rope, weight+rope}; single runs)

| shape | plain | weight | rope | weight+rope |
|---|---|---|---|---|
| 128x512 | 4,201 → 3,869 | 5,248 → 4,855 | 7,269 → 6,318 | 8,337 → 7,455 |
| 256x1024 | 5,511 → 5,240 | 6,597 → 6,274 | 11,231 → 10,327 | 11,475 → 10,591 |
| 32x3072 | 4,556 → 4,249 | 5,512 → 5,255 | 9,201 → 8,039 | 9,721 → 8,739 |
| 256x3072 | 13,133 → 11,047 | 13,903 → 11,477 | 20,804 → 17,085 | 20,156 → 18,057 |
| 2048x3072 | 79,484 → 75,121 | 81,547 → 77,078 | 102,218 → 73,910 | 114,154 → 81,239 |
| **2304x3072** | 77,150 → 75,127 | 87,350 → 84,210 | 108,824 → 96,966 | **119,736 → 94,762 (−20.9%)** |

- **No-regression result**: no cell is slower than at round start.
- **Against the intermediate commit**: the tiny grids (128x512, 32x3072, 256x1024) sit +1.3–3.1% (≤ 220 ns, medians of
  3) above the rope+rsqrt commit. That is inside the noise band but not zero. They are still faster than at round
  start.
- **Correctness**: golden 50 passed / 225 skipped (INVALID); op unit + tail-block + precision-baseline tests 68/68;
  guard bench 24/24 with `check_output`. `SUPPORTED` is unchanged.

### Leads for Perf 2 (measured, not built)

- **Weight row reads are now the hottest item.** `reader_weight` reaches 65 µs on the worst core (p50 12.7 µs), and
  `compute_weight_expand` waits on it. Ablating the weight DRAM reads gives focus 94.8k → ~85.3k (−9%) and 2304
  weight-only 84.2k → ~75.3k. Ways to remove the 40× redundant 64 B reads: weight column multicast (from `reuse_mcast`;
  needs G | grid.x) or a column-banded core enumeration so that each head group is a rectangle.
- **Remaining DM inefficiency**: `writer_issue` reaches 55 µs on some cores (swapped rows write on NoC0), against the
  65 µs DM roofline.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::reduce<SUM, REDUCE_ROW>` (+ post-op) | capability | No mode that writes each input's row-reduction into a chosen **column of a shared output tile**. The helper emits one output tile per reduced row-block, so the eps+rsqrt post-op runs once per head on whole tiles even though only column 0 is consumed. A `VectorMode` knob on the post-op alone recovers only about 1.9×. | 15,545 | ~3,200 | `rms_norm_head_rope_compute.cpp:200` (`compute_rsqrt`) |
| `compute_kernel_lib::eltwise_chain` (BinaryFpu / DestReuseBinary / AddBinary) + no matmul helper | capability | The chain has no matmul element, so `rot = y @ T` cannot seed a DEST slot inside the chain's acquire. It had to be packed to an fp32 CB and re-unpacked, which forces lane width 2 and an SFPU add. `BinaryFpu<Mul>` also cannot accumulate into an occupied slot (ELWMUL's native `D += a*b`; today only `DestAccumulation::PerRow` exposes it). Fix: a `MatmulTile<in0 Block, in1 fixed tile, Dst>` element plus an accumulate-into-slot flag on `BinaryFpu`. | 21,842 | 15,000 | `rms_norm_head_rope_compute.cpp:291` (`compute_rope`) |
| no helper covers this (dataflow work-distribution family) | capability | No dataflow helper for a dynamic work queue: a per-group fetch-and-add counter (`noc_semaphore_inc` non-posted + `noc_async_atomic_barrier` + `invalidate_l1_cache` + read of `MEM_NOC_ATOMIC_RET_VAL_ADDR`) and fan-out of the claimed index to compute (`read_tile_value`) and writer (raw L1 word). Each caller has to hand-roll the atomic-return protocol and the ctrl-CB publish. | n/a (no helper; static split 110,412) | 93,799 | `rms_norm_head_rope_reader.cpp:149` (`reader_claim`) |

**Summary**: all 5 ideas measured; 3 graduated (rope fusion, packed-column rsqrt, dynamic rows + NoC swap with 2
measured carve-outs). 1 was superseded (normalize_fusion L4) and 1 conflicts with dynamic rows and is deferred
(reuse_mcast: cos/sin mcast conflicts, weight column mcast left for Perf 2). The focus shape got 20.9% faster
(119.7 → 94.8 µs) and the 2048x3072 FLUX shape 28.8% faster (114.2 → 81.2 µs). No regression against round start.

## Perf 2

- **Date**: 2026-11-17. Blackhole, 12x10 worker grid, `DEVICE KERNEL DURATION [ns]`.
- **Focus shape**: `LOOSE_CASES` is empty again (no perf-flagged entry), so the focus stays 2304x3072 weight+rope at the
  default config (bf16, TILE, DRAM interleaved, HiFi4, fp32_dest_acc_en=True). Only cells over 32 tiles were measured.
- **Noise**: the bench host draws from a pool of cards, and identical code moved up to ±3% between runs. Every A/B below
  is a median of 3 runs, with baseline and candidate runs alternated.
- **Tooling note**: the precompile pytest plugin currently crashes (`ttnn.graph.up_front_clear` is missing), so every run
  used `--no-precompile`.

### Measured breakdown (round start, focus shape, single runs)

Reused the Perf 1 zones and ablated by stubbing the payload behind temporary defines, with sync kept. Removed again after.

| variant | ns |
|---|---|
| full | 94,747 |
| compute stubbed (DM only) | 100,121 (noise; compute is hidden) |
| compute + weight reads stubbed | 91,030 |
| compute + weight + cos/sin reads stubbed (x + out only) | 76,741 |
| x reads stubbed too (writes only) | 48,594 |
| writes stubbed instead (x reads only) | 36,631 |
| everything stubbed (sync floor) | 9,557 |
| all DM stubbed (compute only) | 40,951 |
| only weight reads stubbed | 90,027 |
| only cos/sin reads stubbed | 84,154 |
| weight + cos/sin reads stubbed | 75,584 (about the plain op's 76k) |

- **Head-group sweep, rope-only 2304x3072**: G=1 82.3k, G=2 85.6k, G=3 (chosen) 95.3k. At G=1 each row's cos/sin is read
  once. G=1 w+rope regresses to 120k because only 72 cores are busy and each holds the weight for 24 heads.
- **Ranked bottleneck** (compute, at 41 µs, is hidden under DM, so no compute ideas were floated):
  1. cos/sin DRAM reads, 10–14 µs. Each row's 4+4 tiles are read once per head group (G=3), which adds 25% to the x read
     traffic.
  2. Weight row-0 reads, about 5 µs. They are Rg-fold redundant 64 B reads issued at kernel start.
  3. x/out DM efficiency. Writes alone run at 363 GB/s, reads alone at 525 GB/s, and the two together at 422 GB/s.
- **Ceiling**: plain 2304 moves 28.3 MB, about 55 µs at a ~512 GB/s DRAM peak; it measures about 75 µs. The DM ceiling
  CLI (`noc_estimate`) is not built on this box.

### Portfolio and per-idea verdicts (subagent measurements, medians of 3 unless noted)

| idea | verdict | focus-shape result | domain / notes |
|---|---|---|---|
| `cossin_share`: team of ≤4 head-group cores; the leader claims the row, reads cos/sin once and forwards them | **WIN** | 92,791 → 85,625 | Applies everywhere rope is on with team size > 1, for any G (including teams that wrap grid rows, which use unicast) and tail rows. Rope-only 2304: −7%; 256x3072 w+rope: −10.5%; 512x4096: −17%. An uncapped team (G=24 on 32x3072) measured +32%: the leader's fan-out serializes, so the team is capped at 4 (cap 3: −8.6%, cap 4 better on G=4 shapes). |
| `weight_prologue`: per-head-group leader(s) on the writer RISC read row 0 once and forward a compact 128 B/tile copy | **WIN** | 93,876 → 91,460 | 2304 weight-only −10.5%, 2048 w+rope −9.6%. **Measured regression on static grids** (Rg ≥ Rt): 256x3072 weight-only +27%, 512x3072 w+rope +7.6%, 1024x1024 weight-only +14%. Bit-identical output. Nulls inside it: the writer reading its own weight (95–97k, regression), hardware mcast per column (leader time 45–73 µs), one leader per 4 cores (2048 +4.6%), delaying the expand (flat). Found and fixed a race: a writer-side `noc_semaphore_inc` lands its atomic return in `MEM_NOC_ATOMIC_RET_VAL_ADDR`, the word the reader's claim reads. Flags are now plain remote writes. |
| `dm_write_path`: posted writes, bank-staggered order, VC choice, both NoCs, output CB depth 1/3, chunk sizes, read chunk, port aiming | **NULL** | best correct trade `id1od1_top` 92.4k → 89.9k, but plain +2.2%, 2048 w+rope +4.1%, 256 +6.2% | In the main phase x+out run near the DRAM peak. The gap is tail imbalance (NoC0 first-byte starvation, swapped rows draining queued writes), not the writer's issue path. Recorded hazards: writes aimed at the other NoC's DRAM port gave wrong output; different CB sizes on one index over disjoint core ranges gave NaN; a reader and writer sharing a NoC hangs in dedicated mode. |

### What graduated

1. **cos/sin team share** (commit `82ba0fe7dbe`). This is the only rope path whenever the team size `min(G, 4)` is > 1;
   G=1 and no-rope compile to the old code, where the old code is the same thing.
   - Up to `COSSIN_TEAM_MAX = 4` consecutive head groups of a row-group form a team. One counter per team index sits on
     the team leader in row-group 0.
   - The leader waits on `PEER_READY` (peers +1 after reserving x and cos/sin space), claims, and writes the row into
     each peer's `ROW_MAILBOX`.
   - It then reads cos/sin and forwards them at its first x-chunk barrier into the same CB slot on every peer (mcast if
     the peers are the next cores in its grid row, otherwise unicast), then sets `ROPE_VALID`.
   - Deleted: the peers' cos/sin DRAM reads.
   - No carve-out.
   - Zone renamed `reader_issue_rope_lead`: its 16-bit source-location hash collided with `compute_rsqrt`, and the profiler
     aborted.
2. **Weight forwarding** (commit `33db6068537`). Under `WEIGHT_FWD` the writer owns `cb_weight_row`, and the reader's
   `load_weight_rows` is compiled out.
   - Leaders are the last `ceil((Rg-1)/20)` cores of each head group (bottom rows, so they read on NOC_1).
   - Peers wait on `SEM_W_READY`, renumbered to 4 so it does not collide with the team semaphores 1–3.
   - **Carve-out, earned by measured regressions**: `weight_fwd = has_weight and dynamic_rows`. Static grids keep the
     reader prologue: 256x3072 weight-only +27%, 512x3072 w+rope +7.6%, 1024x1024 weight-only +14%.
   - **Checked and not carved out**: 2048x3072 weight-only measured +1.6% over 4 alternated pairs (noise), so it keeps
     the new path.
   - Correctness: 75 repeated calls (program-cache hits) were bit-identical, and the bench and tail-block tests are clean
     under `--dev` (watcher).

Instrumentation extended: `reader_rope_forward`, `reader_rope_wait`, `reader_issue_rope_lead`, `writer_weight`.

### Whole-op before → after (guard set, medians of 3, alternated)

| shape | plain | weight | rope | weight+rope |
|---|---|---|---|---|
| 128x512 | 3,884 → 3,861 | 4,793 → 4,793 | 6,351 → 6,429 | 7,424 → 7,636 (+2.9%) |
| 256x1024 | 5,116 → 5,212 | 6,307 → 6,328 | 10,258 → 8,864 (−13.6%) | 10,487 → 9,636 (−8.1%) |
| 32x3072 | 4,132 → 4,195 | 5,204 → 5,230 | 8,122 → 7,169 (−11.7%) | 8,852 → 8,315 (−6.1%) |
| 256x3072 | 10,767 → 10,884 | 11,329 → 11,334 | 17,119 → 14,670 (−14.3%) | 17,680 → 16,364 (−7.4%) |
| 2048x3072 | 73,741 → 75,941 | 78,888 → 79,423 | 74,010 → 68,996 (−6.8%) | 79,994 → 67,030 (−16.2%) |
| **2304x3072** | 75,672 → 75,614 | 82,764 → 76,339 (−7.8%) | 96,972 → 79,672 (−17.8%) | **93,742 → 83,059 (−11.4%)** |

- **No-regression result**: no cell regressed beyond the noise band.
- **Plain** and the static-grid weight cells run unchanged code. Their +1–3% moves are run-to-run card noise.
- **128x512 w+rope** at +2.9% is the largest move. It is in the noise band; the subagent's 4 runs gave +1.3–2.3%.
- **Correctness**: golden 50 passed / 225 skipped (INVALID) after each graduation; op unit + tail-block +
  precision-baseline + guard bench 92/92. `SUPPORTED` is unchanged.

### Leads for Perf 3 (measured, not built)

- **The writer is now the tail on the focus shape.** BRISC-KERNEL max is 77 µs against 65 µs for TRISC.
- **`writer_weight` costs about 10 µs on every core.** It is the leader's DRAM read plus the forward, and the peers wait
  on it. It overlaps the first block, but it delays the start of output writes.
- **`dm_write_path` points at scheduling**: NoC0 first-byte starvation of the middle rows, and swapped rows draining
  queued writes after their last claim. One lead is to cap claims by how far the writes have drained.

### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| no helper covers this (dataflow work-distribution / team-forward family) | capability | No helper for a team work queue. Only the leader claims from a fetch-and-add counter, then fans the claimed index plus a payload (cos/sin) out to a small peer set: mcast when the peers are contiguous, unicast when they are not, with a same-VC data → valid-flag ordering. The peers' "slot reserved" credit must flow back before the next claim, and every member's CB write pointer has to stay in lockstep so a single destination address is valid on all of them. | n/a (no helper; per-group claim 93,742) | 88,583 | `rms_norm_head_rope_reader.cpp:190` (`reader_claim`, team path) and `:247` (`forward_rope`) |
| `ttnn.Mcast` / `McastArgs` multicast helper | capability | The receivers are a strided, non-rectangular core set (head group g = cores k ≡ g mod G). The helper only partitions consecutive receiver groups, and its hardware-multicast transport measured 45–73 µs of leader time under the x-read load. A unicast forward to an explicit peer list, with a plain remote-write flag, was needed. A writer-side `noc_semaphore_inc` is unsafe here because it shares `MEM_NOC_ATOMIC_RET_VAL_ADDR` with the reader's claim. | ~90.3–93k (per-column hardware mcast, focus) | 91,460 (subagent bench); op 88,665 → 83,059 | `rms_norm_head_rope_writer.cpp:88` (`writer_weight`) |

**Summary**: all 3 ideas measured; 2 graduated (cos/sin team share, weight forwarding with 1 measured carve-out) and 1 was
null (write path). The focus shape got 11.4% faster (93.7 → 83.1 µs), rope-only 2304 17.8% faster, and 2048x3072
w+rope 16.2% faster. No regression beyond noise. Since Phase 0, the focus shape has gone from 119.7 to 83.1 µs.

## Perf 3

- **Date**: 2026-11-17. Blackhole, 12x10 worker grid, `DEVICE KERNEL DURATION [ns]`.
- **Focus shape**: `LOOSE_CASES` is empty (no perf-flagged entry). The focus stays 2304x3072 weight+rope at the default
  config (bf16, TILE, DRAM interleaved, HiFi4, fp32_dest_acc_en=True), but the breakdown showed the most headroom in the
  plain and weight-only cells, so the portfolio targeted those as well. Only cells over 32 tiles were measured.
- **Noise**: ±2–3%. The focus cell is **bimodal** between identical programs: about 75 µs or about 84–86 µs, depending
  on whether a starved top-row team claims a late second row. A ≤10% move on that cell is not attributable from 3 runs.
- **Tooling**: the precompile plugin still crashes, so every run used `--no-precompile`. The ablation stubs were
  temporary defines and were removed again. Existing zones were reused; no kernel changed, so no new zones were needed.

### Measured breakdown (round start)

Baseline: 2304 plain / w / rope / w+rope = 74,973 / 77,068 / 78,330 / 76,632; 2048 = 76,674 / 77,979 / 70,407 /
67,198.

Cumulative ablation (payload stubbed, sync kept; x reads/writes only, cos/sin + weight reads kept):

| variant | 2304 plain | 2304 w | 2304 rope | 2304 w+rope | 2048 w+rope | 2048 plain | 256x3072 w+rope |
|---|---|---|---|---|---|---|---|
| full | 76,453 | 76,613 | 79,892 | 76,784 | 68,236 | 76,291 | 15,729 |
| compute stubbed (DM only) | 72,682 | 74,562 | 82,600 | 77,750 | 68,462 | 71,556 | 13,276 |
| + x reads stubbed (writes only) | 46,915 | 48,007 | 48,913 | 53,947 | 49,410 | 35,582 | 8,310 |
| + writes stubbed instead (reads only) | 34,924 | 34,908 | 39,909 | 39,465 | 37,181 | 31,203 | 9,147 |
| everything stubbed (sync floor) | 5,740 | 9,860 | 14,145 | 13,627 | 13,334 | 6,687 | 5,901 |
| DM stubbed (compute only) | 16,036 | 26,627 | 32,122 | 45,783 | 46,015 | 22,481 | 12,063 |

- **Compute is hidden on every cell.** DM-only is at or above full, so no compute ideas were floated.
- **2048 plain is slower than 2048 w+rope**, which does strictly more work. Per-core end times show why.
  - The old rule picked G=24, Rg=5 (1 head per core).
  - Each head group therefore sat in one grid column, and end times ramped 38 → 76 µs from the left columns to the
    right.
  - Claiming balances only within a group.
- **Focus per-core ends are 50–77 µs.** That is claim quantization (1.8 row-units of 32 tiles per core) plus starved
  top rows.
- **Ranked**:
  1. Cross-core tail imbalance and work-split choice.
  2. Read/write non-overlap: 2048 plain DM-only 71.6 ≈ writes 35.6 + reads 31.2.
  3. Compute (hidden).
- **Ceiling**: ~512 GB/s gives 49–55 µs. The `rw_overlap` copy bench later measured this card's real mixed-copy ceiling
  with 2 KB pages at about 390 GB/s, i.e. 72.5 µs for the focus x/out bytes.

### Portfolio and per-idea verdicts

| idea | verdict | result (ns, medians) | domain / notes |
|---|---|---|---|
| `global_queue`: one op-wide (row, head) work pool, guided chunk claims, two-phase all-heads weight | WIN → **superseded** | 2048 plain 75,121 → 66,387; 2048 w 79,503 → 71,944; focus flat (74,747 → 75,427) | Plain/weight on dynamic grids only. Measured regressions for rope rows split over the pool: focus rope 81.0 → 100.5k, and smallest-claim-8 and cos/sin "homes" variants also regressed. So rope kept team claiming. Large kernel rewrite; its gain comes from the same cells the host-only split fix wins on. Not graduated. |
| `core_placement`: snake-draft placement so every claim pool spans all columns | WIN → **superseded** | 2048 plain 75,840 → 67,703; 2048 w 78,892 → 68,700; focus identity | The coordinator's 3-way A/B on top of the new split was **flat** everywhere: 2048 plain 67,466 → 66,713; 512x3072 w 19,110 → 19,771; 4096 w 132,643 → 131,276. Not graduated. |
| `split_granularity`: measured cost model for `select_grid_split` | **WIN, graduated** | 2048 plain 74,981 → 67,982; 2048 w 78,393 → 68,938; 256x3072 plain 10,592 → 9,116 | Everywhere; rope splits unchanged. Key finding: the 2048 plain loss was the 4-tile blocks, not column placement. Forcing (3, 40) with 1-head blocks measured 75.2 µs, the same as (24, 5). Every focus split was slower than (3, 40); the best alternative, (2, 60, 12), measured 81.1k. |
| `rw_overlap`: make x reads and out writes overlap / run nearer peak | **NULL** | copy bench: reads 34.3 + writes 41.4 → mixed 72.5 µs; globally phased 74.8 | Reads and writes share DRAM bandwidth; there is no overlap to recover. The op's DM is within ~3% of a pure copy of the same bytes. Flat: swap3 (also +9–17% single runs at 256x3072), rc4, deep32 CBs, bank rotation. Regression: 4 heads per block, focus 80.4 → 89.5k. |

### What graduated

**Work-split cost model** (commit `0fdae4833ec`), as the only `select_grid_split`. It replaces the busiest-core-units
rule, and the old rule is kept only as the static-split candidate generator `_static_split`.
- `cost = u_avg·(1 + 0.15·blocks_per_row/hpc) + λ·max(0, u_max − u_avg) + (0.5 if weight and Rg < Rt)`.
  - Units are head-rows.
  - λ = 0.25 without rope and 1.0 with rope.
  - Each constant's measurement is cited in the descriptor.
- **No carve-out.** No kernel changed.
- Instrumentation is unchanged; the change is host-only.

Coordinator 3-way A/B (base / split / split + placement, medians of 3, alternated, every cell `check_output`):

| cell | base | split (graduated) | split + placement |
|---|---|---|---|
| 256x3072 plain | 10,401 | 9,236 (−11.2%) | 9,296 |
| 512x3072 plain | 19,146 | 18,310 (−4.4%) | 18,188 |
| 512x3072 w | 20,603 | 19,110 (−7.2%) | 19,771 |
| 1024x3072 plain | 37,660 | 35,575 (−5.5%) | 35,742 |
| 1024x3072 w | 39,896 | 34,288 (−14.1%) | 34,685 |
| 2048x3072 plain | 74,012 | 67,466 (−8.8%) | 66,713 |
| 2048x3072 w | 78,719 | 68,231 (−13.3%) | 69,238 |
| 2304x3072 plain | 75,484 | 75,471 | 74,575 |
| 2304x3072 w | 76,421 | 76,757 | 77,350 |
| 2304x3072 w+rope (focus, same program) | 84,580 | 77,040 | 85,510 |
| 4096x3072 w | 141,353 | 132,643 (−6.2%) | 131,276 |
| 1024x2048 w | 26,724 | 23,293 (−12.8%) | 23,261 |
| 1952x1024 w | 26,018 | 22,699 (−12.8%) | 22,284 |

The focus row runs the same program in all three columns. Its 77 vs 85 µs is the bimodal noise, not a win.

### Guard set (whole op, medians of 3, base → new, alternated)

| shape | plain | weight | rope | weight+rope |
|---|---|---|---|---|
| 128x512 | 3,861 → 3,869 | 4,818 → 4,804 | 6,408 → 6,444 | 7,606 → 7,584 |
| 256x1024 | 5,253 → 5,307 | 6,300 → 6,267 | 8,881 → 8,924 | 9,651 → 9,561 |
| 32x3072 | 4,089 → 4,237 (=) | 5,241 → 5,193 | 7,201 → 7,121 | 8,186 → 8,338 |
| 256x3072 | **10,859 → 9,316 (−14.2%)** | 11,085 → 11,210 | 14,715 → 14,686 | 16,254 → 16,422 |
| 2048x3072 | **76,096 → 67,987 (−10.7%)** | **79,359 → 68,722 (−13.4%)** | 67,756 → 68,275 | 68,066 → 66,352 |
| 2304x3072 | 74,666 → 76,172 | 77,007 → 76,557 | 79,085 → 79,533 | 85,804 → 76,187 (bimodal, same program) |

- **No-regression result**: only the 3 bold cells change split, and all of them got faster.
- **The other 21 cells run the identical program.** I checked this on the host by comparing old vs new splits on all
  24 cells, so their moves are run-to-run noise. The largest, 32x3072 plain at +3.6%, is a single-row grid where only
  the static split exists.
- **Correctness**:
  - Golden: 50 passed / 225 skipped (INVALID).
  - Op unit + tail-block + precision-baseline tests: 68/68.
  - Guard bench and Perf 3 bench: all pass `check_output`.
  - `SUPPORTED` is unchanged.

### Leads for a later round (measured, not built)

- **The focus shape's slow mode is claim-policy.** A starved top-row team claims a late second row and becomes the
  tail. A late-claim cutoff, or finer units for the last rows only, could remove the ~8 µs mode. Splitting every rope
  row finer measured a regression (`global_queue`).
- **DM is at the measured copy ceiling.** The ceiling is about 390 GB/s mixed with 2 KB pages, so the remaining focus
  headroom is only the tail imbalance.

### Helper bypasses — none

The graduated change is host-side only (work-split selection); no kernel or helper usage changed.

**Summary**: all 4 ideas measured.
- **Graduated (1)**: the split cost model.
- **Superseded (2)**: `global_queue` and `core_placement`. Both won in isolation, but the split fix covers their wins,
  and placement measured flat on top of it.
- **Null (1)**: `rw_overlap`.
- **Results**: 2048x3072 plain −10.7%, weight −13.4%, 256x3072 plain −14.2%. Off-guard shapes: weight −6% to −14%
  (1024x3072, 4096x3072, 1024x2048, 1952x1024).
- **Focus shape**: unchanged (same program; DM already at the copy ceiling).
- No regression.
