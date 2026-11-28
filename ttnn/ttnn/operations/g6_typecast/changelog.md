# Changelog: g6_typecast

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime
  R1 `tile_stream`: interleaved TILE in → eltwise-chain typecast → interleaved TILE out, contiguous tile ranges
  over the full compute grid, one CB handshake + one reader barrier per block.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], output_dtype=[float32, bfloat8_b, int32], layout=[TILE],
  buffer=[dram, l1], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[3, 4]
- **Accuracy achieved**: float32 / int32 destinations bit-exact (PCC=1.0, max_abs_err=0, 0 ULP); bfloat8_b
  PCC=0.999975, max_abs_err=0.03125, rel_rms_err=0.0076, ≤ 1 quantization step vs host rendering (measured on
  6 production shapes via test_g6_typecast_precision_baseline.py)
- **Golden suite at Phase 0**: 42 supported_pass / 27 xfail_expected / 0 supported_fail / 0 xpass_drift /
  0 xfail_wrong_mode (44/71 incl. 2 untagged regression tests; per `verifier_report.json`)
- **Issues encountered**: verifier fix — program-cache key now includes the block knobs (`BUFFER_DEPTH`,
  `BLOCK_TILES_CAP`, `L1_OP_BUDGET_BYTES`, `L1_SLACK_BYTES`) so a turned knob never replays a stale program (BB-07).
- **Tests added**: test_g6_typecast.py, test_g6_typecast_blocking.py, test_g6_typecast_precision_baseline.py

## Refinement 1 — ROW_MAJOR regime (R2 `rm_pseudo_tile_stream`) with its conversion arms
- Date: 2026-11-27
- What was done: Regime R2 built natively (no to_layout / tilize wrapper; one program). Selected by
  `input_tensor.layout` and reported in `last_plan["regime"]`. Contiguous row ranges over the full grid
  (`split_work_to_cores(row_wise=True)`); per block of `rows_per_block` rows the reader issues ONE NoC read
  per bank run (rows r, r+nb, ... are adjacent in a bank at the aligned-page stride; nb =
  `TensorAccessor::contiguous_page_stride()`) into a 64 B-aligned staging CB, one barrier, then compacts the
  rows densely into `cb_input_tiles` as 1024-value pseudo-tiles; compute runs the unchanged R1 chain; the
  writer expands dense rows back into bank-run order and issues ONE write per bank run (flush before
  staging reuse). `block_tiles` stays the single block knob (`plan_rm_blocks` derives rows_per_block from
  it, `BLOCK_TILES_CAP` caps it). New arms: bf16→uint8 (SFPU truncation), uint32→int32 and int32→uint32
  (unpack-to-DEST, no SFPU, all 32 bits). 32-bit sources tag `cb_input_tiles` `UnpackToDestFp32`;
  `fp32_dest_acc_en` also for uint8 destinations and 32-bit sources (Phase 0 pairs unchanged).
  SUPPORTED: dtype += uint32, int32; output_dtype += uint8, uint32; layout += ROW_MAJOR; rank += 1, 2.
  `validate()` refuses pairs with no arm (`CONVERSION_ARMS`) and bf8b + ROW_MAJOR with ValueError;
  EXCLUSIONS stays empty.
  Reused: compute kernel (unchanged), reader/writer files (rm_mode `if constexpr` branch), split/plan/
  cache. Added: `_create_rm_program_descriptor`, `plan_rm_blocks`, 2 staging CBs, `_compute_config`.
- Accuracy achieved: bit-exact (0 differing values) on [1,3844], [1,31], [1], [3,77,33], [2,50,3] (L1),
  [1,1024,1024,3], [1,2048,2048,3], TILE [1,1,32,128] u32→i32 and [1,1,64,96] i32→u32.
- Golden test progress: targeted slice (all 14 ROW_MAJOR cells + loose F322/F317 + Q2230 + TILE
  representatives Q2185/P39/Q2229): 23 passed, 1 xfailed (Q2228, int32→bf16 = Refinement 2). Unit dir:
  test_g6_typecast.py + test_g6_typecast_blocking.py 30/30.
- Perf baseline (device kernel ns, one fresh run): F322 2 330 848 ns (stock 14 563 310; target 110 590),
  F317 952 224 ns; BRISC (writer) is the longest RISC (F322: writer 2.33 ms, reader 1.71 ms) — the
  per-row RISC-V expansion/compaction (byte copies of 3 B / 6 B rows) binds, not the NoC. Refinement 3
  levers: wider copy units / fewer per-row ops, overlap staging reuse (double staging), bank-local core
  assignment. TILE shapes unchanged (Q2185 36.1 µs, P39 74.7 µs).
- Issues encountered: None (first run green).
- Tests added: tests/ttnn/unit_tests/operations/g6_typecast/test_g6_typecast_rm.py (regime pins for TILE and
  ROW_MAJOR, new arms, forced multi-block / ragged RM via BLOCK_TILES_CAP, L1 RM, host refusals).

## Refinement 2 — 32-bit and 16-bit sources on TILE: float32 / int32 / uint16 sources, bfloat16 destination
- Date: 2026-11-27
- What was done: Reused the R1 `tile_stream` reader/writer/CBs unchanged (page size + format already come from
  the tensors), plus the existing chain and the `fp32_dest_acc_en` / `UnpackToDestFp32` compute-config path.
  Added: `float32`, `uint16` to SUPPORTED["dtype"]; `bfloat16` to SUPPORTED["output_dtype"]; three
  CONVERSION_ARMS (fp32->bf16, int32->bf16, uint16->int32); `UnpackToDestFp32` also for the uint16 source
  (integer source, so it must not go through SRCA). Compute kernel: an arm switch. int32->bf16 runs as
  `Typecast<Int32,Float32> -> Typecast<Float32,Float16_b>` in ONE chain. The direct Int32->Float16_b LLK arm
  rounds its fp32->bf16 step half-away (SFP_STOCH_RND), so it is wrong on ties: 257->258, 33685505->33816576,
  2^24+2^16+1->16908288. The fp32->bf16 LLK arm is a software RNE. The two-step chain has two distinct SFPU
  inits, so it cannot be boot-hoisted. It keeps `InitReconfigOwner::Chain` every block and re-inits per tile.
  That costs a little on int32->bf16 only (decode-size cells). Every other arm is still hoisted.
- Accuracy achieved: bit-exact (gate `exact`) on F315, F316, F318, F319, F320, F323, F324 (fp32->bf16, incl.
  w_non_aligned), Q2228 (int32->bf16, two-rounding witnesses), Q2231 (uint16->int32).
- Golden test progress: targeted slice 12/12. All golden (src,dst) pairs now have an arm and every TARGET axis
  value is in SUPPORTED, so xfail_expected should be 0. The harness full run is the confirmation of record.
- Issues encountered: hardware finding — fp32 subnormals and -0.0 come out of the fp32->bf16 path as +0
  (Tensix flush-to-zero). No golden input contains them (randn + exact ties). Pinned as a non-strict xfail.
- Tests added: tests/ttnn/unit_tests/operations/g6_typecast/test_g6_typecast_sources.py (fp32->bf16 with RNE
  tie witnesses, DRAM/L1, non-aligned; int32->bf16 with the two-rounding witnesses; uint16->int32; FTZ xfail).

## Refinement 3 — Speed up the perf-flagged profile (F322, with F317)
- Date: 2026-11-27
- What was done: R2 (`rm_pseudo_tile_stream`) perf, both dataflow halves.
  (1) Row movers rewritten as a shared header `kernels/g6_typecast_rm_rows.hpp`. They walk the dense order and
  funnel groups of G rows (G·row_bytes ≡ 0 mod 4) through a 64-bit accumulator with compile-time shifts.
  Only whole-word loads and stores are used: the reader compacts 2 bf16 rows into 3 words, the writer expands
  3 words into 4 uint8 slots. Before, each byte was a separate load and store. Word-sized rows are plain word
  copies; large odd rows and the leftover rows of a block keep the byte/halfword copy.
  (2) `STAGING_DEPTH` knob (default 2): reader issues block b+1's bank-run reads into the other staging slot
  before compacting block b; writer expands block b while block b−1's writes drain, flushes after the
  expansion. Only one block is in flight per direction, so the plain barrier/flush retire exactly one block
  (no trids).
  (3) `RM_BLOCK_TILES` knob (default 2, sweep-chosen) caps the R2 block for a short pipeline fill/drain.
  All knobs are in the program-cache key. Reused: reader/writer rm_mode branches, `plan_rm_blocks`,
  staging CBs. Added: the shared row-mover header, two knobs, one CT arg per dataflow kernel.
- Perf (device kernel ns, BH 110 cores): F322 2 320 µs → 1 562 µs; F317 958 µs → 361 µs. Q2185 35.9 µs and
  P39 68.9 µs unchanged. F322 ablation: row moves only 782 µs, NoC only 1 476 µs, both stubbed 38 µs. The op
  is now NoC/DRAM-bound on 537 MB of physical traffic (64 B-aligned page per 6 B / 3 B row; floor ≈ 1.05 ms
  at the DRAM peak). The 110.59 µs target assumes logical bytes and is unreachable with this page geometry.
  See the Outcome line in op_requirements.md.
- Accuracy achieved: bit-exact (gate `exact`) on F322, F317, [1,7,13,3], [5,3], [1,333,77,3], [2,50,6], plus
  every existing RM/TILE unit case.
- Golden test progress: targeted slice (F322, F317, Q2189, F321, Q2185, P39, Q2228, F315) 12/12; unit dir 61
  passed + 1 expected xfail.
- Issues encountered: None (`--profile` on the golden harness emits no CSV; guard timings came from unit-test
  equivalents of Q2185/P39).
- Tests added: tests/ttnn/unit_tests/operations/g6_typecast/test_g6_typecast_perf.py (F322/F317 profile pins,
  odd / tiny / ragged / 6 B-row shapes for the funnel and tail paths).

## Refinement 4 — Speed up the flagged TILE profiles (Q2185, P39): read/convert/write overlap
- Date: 2026-11-27
- What was done:
  - New R1 knob `TILE_BLOCK_TILES = 4`, min'd into `plan_blocks` together with `BLOCK_TILES_CAP` (single source).
    Each core now walks its range in 4-tile blocks at depth 2, so read, convert and write overlap. Before, the
    whole range was one block, so the three stages ran serially.
  - Sweep (cap None/16/8/6/4/3/2 × depth 2/3/4): P39 69.5/38.6/20.4/18.1/17.4/17.2/20.3; Q2185 is flat around 35 µs.
    Depth 3 is within noise of depth 2, so `BUFFER_DEPTH` stays 2 and R2 is unchanged.
  - New knob `READ_INFLIGHT` (reader TILE path): a transaction-id sliding window that keeps N blocks of reads on the
    wire (catalog `trid_double_issue`), clamped to the effective depth. It does not win (P39 slows with more reads in
    flight), so it ships at 1, which issues the same reads and barriers as before.
  - Writer twin: already in place. The writer flushes per block and does a single final barrier, so its writes stay
    on the wire.
  - Also measured and not kept: NoC swap (Q2185 53 µs, much worse) and `UnpackToDestFp32` on the bf16 source (flat).
  - `bank_stagger` analysis: core starts are already spread over banks (see the Outcome line).
  - Both knobs are in the program-cache key.
  - Reused: reader block loop (refactored into issue/finish lambdas), `plan_blocks`, CBs, compute, writer.
  - Added: 2 knobs and 2 reader CT args (inert on R2).
- Perf (device kernel ns, BH 110 cores):
  - P39 69.5 → 17.3 µs; Q2185 36.6 → 35.2 µs.
  - Guards: Q2187 7.6 → 7.5, i32 DRAM [8,1024,256] 38.8 → 35.3, f32 L1 [4,512,512] 9.3 → 8.4, Q2191 1.28 → 1.25 µs.
  - R2 F322 1.55 ms and F317 0.36–0.37 ms are unchanged.
- Remaining headroom (finding):
  - Q2185 is DRAM-bound. Ablation: writes only 28.9 µs, reads only 11.9 µs, compute only 1.5 µs; about 366 GB/s on
    12.6 MB of physical traffic.
  - P39 is limited by NoC/L1-bank congestion.
- Accuracy achieved: Q2185 bit-exact (gate `exact`); P39 within the `bfp8_group` gate. Unit cases bit-exact or bfp8-group.
- Golden test progress: targeted slice (Q2185, P39, P46, P50, Q2187, Q2191, Q2227, Q2228, Q2229, F315, F317, F322)
  16/16. Unit dir: 79 passed, 1 expected xfail. Forced `READ_INFLIGHT=3, BUFFER_DEPTH=3` on the blocking and main
  unit tests: 30/30.
- Issues encountered:
  - `tracy-capture` occasionally crashes (exit 3); a re-run works.
  - A `-k` filter with spaces is mangled under `--profile`, so the sweep test filters by the `G6_CASES` environment
    variable instead.
- Tests added: tests/ttnn/unit_tests/operations/g6_typecast/test_g6_typecast_tile_perf.py. It covers the flagged
  TILE shapes and guards at the shipped knobs, plus a forced trid window (cap 3, depth 3, 2 or 3 in flight, ragged
  tails). `G6_SWEEP` / `G6_CASES` drive knob sweeps for profiling.

## Perf 1 — perf tournament: every production case faster than stock `ttnn.typecast`
- Date: 2026-11-28. Board: p150b (Blackhole), chip 1, 1350 MHz, 110 worker cores. Times are `DEVICE KERNEL
  DURATION` in ns: the median of launches 2..12 of the case bench, one profiler session per case
  (`perf_experiments/_coord/bench.sh`, which wraps `generated/meta/g6_typecast/bench/bench_cases.py`). Stock is timed
  as a black box (`BENCH_SIDE=stock`). Raw runs: `perf_experiments/_coord/{baseline_op,baseline_stock,final2_op}.txt`.
- **Outcome:** all 7 ideas measured. 7 graduated (counting zone gating) and 0 were null as whole ideas; the
  nulls were sub-options (listed below). Before, the op lost to stock on 49 of the 65 production cases. Now it is
  faster than stock on 65 of 65; the smallest margin is Q2218 at 1.05× (a guard). SUPPORTED, EXCLUSIONS and the
  precision contract are unchanged. Golden 71/71 after every commit. Unit dir: 111 passed, 1 xfail.

### Measured breakdown (Step 1) and ranked bottleneck
- **Instrumentation:** this tree has no `perf_instrumentation.hpp`. `DeviceZoneScopedN` costs about 51 cycles per
  execution whenever the profiler is on, and that is exactly when the A/B is timed. The per-block zones (READ_BLOCK,
  READ_WAIT_SPACE, CONVERT, WRITE_BLOCK, WRITE_WAIT_DATA) are therefore now `G6_ZONE(...)`
  (`kernels/g6_typecast_zones.hpp`). They compile in only with `G6_TYPECAST_ZONES=1`; they stay in the source
  permanently, and every new path has its own zones.
  - Measured cost of the zones, on → off: Q2198 1047→946, Q2221 1212→1101, F318 1627→1536, Q2228 1504→1413.
    The operator's inference is confirmed.
- **Cumulative ablation, Q2198** (bf16→fp32, 1 core, 1 tile, zones off): full 946; compute stubbed 502; + reads
  stubbed 455; + writes stubbed 132. Removing one stage alone: no write 662, no read 707, compute only 421.
  - The lean-compute subagent's per-RISC timeline later corrected the reading of this ablation. The compute
    startup is mostly hidden behind the DRAM read. Stubbing compute also let the reads and writes overlap, so
    the 444 ns overstates the cost of compute.
- **Ablation, Q2185** (110 cores): full 35293; write only 28370; read only 11788; compute only 1280. The write
  side binds, as the operator found.
- **Ranked bottleneck:**
  1. G1 (RM u32↔i32): per-value row moves on 1 core.
  2. G4: DRAM writes of whole fp32 tiles of which 2 of 4 faces are padding (W=12).
  3. G3: fixed per-core latency (DRAM read placement, NoC call overhead, redundant compute reconfig, zones).
  4. G2: the int32→bf16 SFPU chain re-inits per tile; math thread 1082 vs 818.

### Portfolio and verdicts (per idea; experiment dirs under `perf_experiments/`)
| idea | verdict | focus before → after (stock) | domain / exceptions |
|---|---|---|---|
| zone gating (coordinator) | WIN | Q2198 1047→946 | all kernels |
| `pure_copy_arms`: u32↔i32 as a byte copy, no compute | WIN | Q2220 57563→739 (2285), F321 918→491 (879), Q2230 1052→636 (1086) | both arms, every layout/rank/buffer; none |
| `tile_face_skip`: read/write only live faces | WIN | Q2185 34926→16745 (33859), Q2188 2870→1778 (2653) | every TILE dtype incl. bf8b, DRAM/L1; none (flat where no face is dead) |
| `lean_compute_startup`: inits once + single-packet NoC | WIN (small) | Q2198 946→935 (compute), →880 with face skip + one-packet | every arm; none |
| `int32_bf16_fused_sfpu`: one-init RNE element | WIN (partial) | Q2228 1413→1246 (1130) | int32→bf16 only |
| `bank_adjacent_placement` | WIN | Q2198 880→801, Q2221 1064→993, F320 1681→1532 | DRAM, fewer tiles than cores; carve-outs below |
| `subtile_split`: NF faces of one tile per core | WIN | Q2228 1150→1015 (1130), Q2229 1264→1009 (1397), F318 1439→1287 | SFPU arms whose unit count fits the grid; carve-out below |

Nulls and options measured but not graduated:
- writer in-flight: deeper output CB (d4) flat; flush every 2 blocks flat to +5% (tile_face_skip).
- raw-LLK compute (`lcs_compute_raw.hpp`): equal to the helper chain owned by the caller, so the helper was kept.
- skipping the SFPU init on the packer-only arms: 0 ns.
- issuing the reads before `cb_reserve_back`: 0 ns.
- int32→bf16 options:
  - Precision-trading options, recorded and NOT graduated: `packer_round`, 1217 ns, 8/4096 wrong; `direct_llk`
    (stock's arm), 1141–1219 ns, inexact. On Blackhole both the packer and the SFPU round fp32→bf16 ties away
    from zero.
  - Exact variants slower than the one graduated: fused_plain 1430; twopass 1435; veltkamp_plain 1412;
    veltkamp_fused_x2 1356.
- pure copy:
  - Reader and writer pair in chunk mode: slower (F321 614 vs 499).
  - Both RISCs on NoC0: NOT bit-exact.
  - One kernel reading on NoC0 and writing on NoC1: hangs (dedicated-NoC mode).
- placement: `noc0` piling at 2–16 tiles was slower (Q2216 1113 vs 1047); a column-major control was flat or slower.
- subtile split:
  - A 16x32 half tile packs bf8b wrong; incorrect, so 32x16 is used.
  - Packer-only arms regress (see the carve-out).

### What graduated, how widely, carve-outs (one commit each, "Perf 1: ...")
1. **Zone gating.** Applies to all kernels.
2. **Regime R0 `pure_copy`** (`_create_copy_program_descriptor`, `g6_typecast_copy_{mover,reader,writer}.cpp`):
   - Applies to every (uint32, int32) and (int32, uint32) cell in either layout. Those cells no longer touch
     R1/R2 at all; no carve-out.
   - Page mode (pages ≥ cores): reader on NoC0 and writer on NoC1 share a 2-slot CB.
   - Chunk mode (fewer pages than cores): one BRISC/NoC0 mover per core, with ~1 KB chunks aligned to 64 B
     (16 B in L1), on the DRAM bank's row.
3. **Int32ToBf16Rne** (`g6_typecast_sfpu_elements.hpp`): replaces the two-element chain. The old code is deleted.
4. **Compute inits once**, Caller-owned from block 0, for every arm. The chain-owned first block is deleted.
5. **Live-face TILE reader/writer + single-packet NoC** (`g6_typecast_tile_faces.hpp`):
   - The whole-tile per-page reads/writes are replaced on every TILE cell.
   - The bf8b whole-face zeroing is deleted: dead faces are never read.
   - The null `READ_INFLIGHT` trid window (it shipped at 1) is deleted, together with its knob, CT args and
     forced tests.
6. **Bank-adjacent placement** (`_tile_core_order`). Two carve-outs, written as exceptions:
   - The split fills the whole grid: measured +3..10% (Q2188 +8–10%, Q2219 +8–10%, Q2187 +3–4.5%, Q2209 +3–3.5%).
     Reads got faster but writes got slower.
   - L1 buffers: a DRAM bank has no meaning there. (0,0) was measured fast; (5,5) was ~850 vs 635 ns.
7. **Sub-tile split** (`_create_split_program_descriptor`, `g6_typecast_split_{reader,writer}.cpp`, compute CT arg
   `NF`):
   - Applies to the SFPU arms (int32→bf16, bf16→int32, uint16→int32, fp32→bf16, bf16→uint8) when the quarter or
     half units fit the grid.
   - bf16→uint8 is untested for speed (no production TILE cell) and rides the rule. It is correct
     (`test_g6_typecast_split.py`).
   - Carve-out, measured regression: the packer-only arms bf16→fp32 and bf16→bf8b (Q2221 half +3.7%, Q2194
     quarter +6%, Q2227/Q2195 half +5..7%).
   - Cells whose quarter and half unit counts both exceed the grid run whole tiles. One unit per core is the only
     form built.

### Whole-op result: all 65 production cases (op before → after, stock; full table: perf_experiments/_coord/final_table.md)
- **operator group G1: graduated** (R0 pure_copy). Bench op before → after (stock):
  - Q2220 57563→746 (2285)
  - Q2189 54130→735 (2268)
  - Q2226 15057→567 (1000)
  - Q2211 14737→567 (996)
  - Q2205 4510→530 (909)
  - Q2214 2785→519 (899)
  - Q2234 2751→518 (891)
  - Q2223 2384→518 (892)
  - Q2208 2085→512 (893)
  - Q2196 1890→512 (896)
  - Q2202 1466→498 (890)
  - Q2199 1443→498 (878)
  - F321 1001→492 (879)
- **operator group G2: graduated** (Int32ToBf16Rne + placement + sub-tile split). Q2228 1504→1010 (1130).
  Stock stays inexact (3/4096); the op is bit-exact.
- **operator group G3: graduated** (zone gating + inits once + live faces + single packet + placement + split).
  - Q2198 1047→810 (919)
  - Q2227 707→516 (595)
  - Q2221 1212→981 (1072)
  - Q2194 1139→932 (1002)
  - Q2203 1327→1119 (1240)
  - F318 1627→1277 (1579)
  - F320 1851→1409 (1779)
  - F323 1653→1330 (1599)
  - F324 1849→1416 (1758)
  - Q2229 1453→1010 (1397)
  - Q2230 1130→626 (1086)
  - Q2231 1150→930 (1089)
  - All 29 G3 cells: 1.07–1.75×.
- **operator group G4: graduated** (live-face transfers).
  - Q2185 35371→16841 (33859)
  - Q2188 2870→1756 (2653)
  - Q2186 18170→8745 (17401)
  - Q2215 5336→3051 (5014)
  - Q2219 2871→1793 (2779)
- **Guards** (the 20 cases already won), all still faster than stock:
  - F322 1564309→1557320 (14602223)
  - F317 366625→365347
  - P39 17331→16951
  - Q2187 7387→7389
  - Q2218 7957→7771 (8174; smallest margin, 1.05)
  - P22 2462→2214
  - Q2219, Q2215, Q2186, Q2192: see above or the run files
  - Nothing regressed beyond noise.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `Typecast` (kernel_lib eltwise, int32→bf16) | capability | (a) No LLK/kernel_lib Typecast rounds int32→bf16 to nearest-even: Int32→Float16_b (and the Blackhole packer) round fp32→bf16 ties away from zero. (b) The exact two-element composition cannot keep its inits hoisted: both inits write vConstIntPrgm0 (LREG12: −31 vs 1), so the chain re-inits both every tile, and the chain cannot tell that one element's constant clobbers the other's. | 1449–1484 (two-element chain) | 1316–1320 | `kernels/g6_typecast_sfpu_elements.hpp:128` (`Int32ToBf16Rne`) |
| `Typecast` / `typecast_tile` | capability | The SFPU vector mode is hard-coded to `VectorMode::RC`. There is no way to run the SFPU over only the 1 or 2 faces a tiny-tile CB holds (R / None), so a 16x16 or 32x16 tile pays 4 faces of SFPU work. | 1262 (Q2229 whole tiles, 4 cores) | ~1006 (16 cores, 1 face each, NF-restricted) | `kernels/g6_typecast_sfpu_elements.hpp:149` (`TypecastNF`) |
| `eltwise_chain` (InitReconfigOwner::Chain) | ergonomics | Not bypassed; worked around. The obvious "let the chain own init on the first block" re-emits `reconfig_data_format_srca` + `pack_reconfig_data_format` right after `compute_kernel_hw_startup` programmed the same CBs. Getting "init once" without that cost means calling each element's init yourself and running Caller-owned from block 0, a contract only the `.inl` reveals. | 903 (Q2198, Chain first block) | 884 (Caller-owned, or raw LLK: equal) | `kernels/g6_typecast_compute.cpp:77` |
| no helper (DM page/chunk copy) | capability | No kernel_lib data-movement helper copies an untransformed page range or byte chunk src→L1→dst. The raw TensorAccessor loops in the copy kernels and the live-face reader/writer bypass nothing. | n/a | n/a | `kernels/g6_typecast_copy_mover.cpp`, `kernels/g6_typecast_reader.cpp` |
