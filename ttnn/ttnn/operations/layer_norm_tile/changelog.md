# Changelog: layer_norm_tile

## Phase 0 - Core Implementation
- **Date**: 2026-11-01
- **What was done**: Initial implementation via the incremental pipeline (planner -> implementer -> verifier). Regime R1 `row_resident`: tile-rows split over the full compute grid, whole 768-wide row resident in L1, optional in-kernel residual add, two-pass centered variance, gamma/beta row-0 staging for TILE or ROW_MAJOR affine in L1 or DRAM. One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True, False], layout=[TILE], alignment=[tile_aligned, h_non_aligned], residual=[residual, no_residual], affine_dtype=[bfloat16], affine_layout=[ROW_MAJOR, TILE], memory_layout=[INTERLEAVED], buffer_type=[DRAM, L1], affine_buffer_type=[DRAM, L1]. SUPPORTED == TARGET.
- **Accuracy achieved**: fp32 DEST: PCC=0.999998, max_abs_err=0.052, rel_rms_err=0.0019. bf16 DEST: PCC=0.999991, max_abs_err=0.19, rel_rms_err=0.0053 (measured on 4 shapes x residual x DEST mode via test_layer_norm_tile_precision_baseline.py).
- **Golden suite at Phase 0**: 545 / 545 passing; verifier CLI 524 supported_pass, 21 untagged regression tests passing, 0 loud categories (`verifier_results/verifier_report.json`).
- **Performance at Phase 0** (device us vs stock): 8x512 in-L1/res-DRAM/out-DRAM 55.0-55.8 vs 55.0; all-DRAM 66.6 vs 70.0; DRAM no-res 40.1-41.3 vs 48.3; all-L1 47.5-49.1 vs 43.3 (loses); L1 no-res 32.5-33.2 vs 34.5; 8x384 all-DRAM 48.0-48.4 vs 47.0 (loses); 32x32 22.3 vs 24.8; affine TILE DRAM 89.2-90.2 vs 125.4.
- **Issues encountered**: no drift. Fixed the ledger symbol-table name for the output-depth constant (`OUT_DEPTH`).
- **Tests added**: test_layer_norm_tile.py (planner), test_layer_norm_tile_precision_baseline.py, probes/probe_tail.py.

## Refinement 1 - DRAM stream efficiency on the flagged headline rows
- Date: 2026-11-01
- What was done: reader/writer rebuilt on a shared stream header (`kernels/layer_norm_tile_stream.hpp`): whole-block reserve, per-block trid ring over C-tile chunks, optional in-chunk bank rotation, grouped writer flush, and split_reader (residual read by the writer RISC on NoC1, `cb_residual_tiles` producer moves to the writer). Affine reads now issue untagged behind the block's last chunk, so the RM affine fix-up no longer delays chunk 0's push. Knobs are host constants in `layer_norm_tile_program_descriptor.py` carried on `Blocking`; split_reader is path-gated (same-tier input/residual, >= half the grid active). Measured (device kernel us, before -> after): flagged #1 54.0 -> 48.6, #2 66.2 -> 52.2, #3 39.8 -> 40.4 (flat), 8x384 all DRAM 48.8 -> 44.5, all L1 47.5 -> 31.4, L1 no-residual 32.3 -> 29.1, 32x32 22.6 -> 23.1, affine TILE DRAM 90.8 -> 77.5. Trid depth > 1, writer grouping, bank stagger and chunk 4/12/24 were flat or worse and stay parked at trivial settings.
- Accuracy achieved: unchanged; precision baseline rel-RMS ~0.0019 (fp32 DEST), PCC >= 0.99999 on [(8,1,512,768), (3,1,500,768)]; golden `(0.999, 0.02)` gate unchanged.
- Golden test progress: 75/75 on the targeted slice (8x512, 3x500, 32x32 ROW_MAJOR affine, 8x512 TILE affine, large-magnitude/mean); full suite left to the harness (545/545 at Phase 0).
- Issues encountered: deeper trid rings slow L1-interleaved sources (all-L1 39 -> 60 us at depth 3); split_reader loses ~2 us when input is in L1 and residual in DRAM, and ~1.5 us on a 32-core grid, hence the gate. Remaining headroom is the lockstep read -> compute -> write structure (see the Outcome line in op_requirements.md).
- Tests added: `tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_stream_knobs.py` (20 cases: every knob off its shipped setting on multi-block / ragged / h_non_aligned shapes).

## Refinement 2 - Compute critical path on the L1-resident rows
- Date: 2026-11-01
- What was done: compute kernel only. The streaming passes (residual add, center, normalize, gamma scale, beta shift) now take `COMPUTE_DEST_LANES` (= 4, clamped to DEST capacity) tiles per DEST acquire through PerBlockSize lifecycles instead of one tile per acquire. The inv_std SFPU post-op (scale, +eps, rsqrt) runs on `VectorMode::C` (faces 0 and 2, where the REDUCE_ROW result lives). Data-format reconfig is dropped at work_a -> work_b and gamma -> beta boundaries (same page format by construction). A fused normalize/affine pass (raw LLK dest-reuse with ROW-broadcast gamma/beta; the chain's DestReuseBinary has no broadcast slot) is implemented, correct, and parked behind `FUSE_NORMALIZE_AFFINE = False` because it measured slower (~7.4k vs 6.2k math cycles per 24-tile row). `MIN_BLOCKS_PER_CORE` = 1 re-measured and still loses (flagged #1 52.0 us), stays 2. Measured (device kernel us, baseline -> after): flagged #1 47.3-48.3 -> 46.0, all L1 30.6-31.8 -> 27.6-28.3, L1 no-residual 27.7-29.9 -> 27.8-28.8 (flat), 8x384 all DRAM ~43.9 -> 42.8, 32x32 23.0 -> 22.0, affine TILE DRAM ~77.7 -> 76.0, DRAM rows flat. Per-row math cycles: center 1.6k -> 0.9k, inv_std 2.2k -> 1.5k, normalize+affine 6.1k -> 5.0k.
- Accuracy achieved: unchanged; fp32 DEST rel-RMS 0.00188-0.00190, PCC >= 0.999998 on [(32,1,32,768), (64,1,17,768), (3,1,500,768), (8,1,512,768)]; bf16 DEST rel-RMS ~0.0052.
- Golden test progress: 117/117 on the targeted slice (8x512, 8x511, 32x32 across all axes, residual-cancel, lofi, large-magnitude); full suite left to the harness.
- Issues encountered: dest-reuse fusion loses under fp32 DEST (math-bound, not unpack/pack-bound). The L1 no-residual row stays flat: its critical path is the first block's whole-row wait plus two serial compute blocks on the 18 two-row cores.
- Tests added: `test_compute_knobs` in `test_layer_norm_tile_stream_knobs.py` (fused affine, 1 and 3 DEST lanes on multi-block / ragged / h_non_aligned shapes).

## Perf 1
- Date: 2026-11-01. Device: Blackhole p100a, 110 cores, AICLK 1350 MHz. Focus (flagged, LOOSE_CASES #1): (8,1,512,768) bf16 TILE, input L1, residual DRAM, output DRAM, gamma/beta ROW_MAJOR in L1, fp32 DEST, HiFi4. All numbers are DEVICE KERNEL DURATION from `tests/ttnn/unit_tests/operations/layer_norm_tile/test_perf_bench.py` under `run_safe_pytest.sh --profile`.
- Instrumentation (permanent): `MaybeDeviceZoneScope` on every stage. Reader: `reader_reserve`, `reader_block`, `reader_affine` / `reader_affine_send`. Writer: `writer_residual_block`, `writer_store_block`, `writer_final_barrier`. Compute: `compute_residual_add`, `compute_wait_z`, `compute_row_mean`, `compute_center`, `compute_row_sumsq`, `compute_inv_std`, `compute_normalize`, `compute_wait_affine`, `compute_affine_scale`, `compute_affine_shift`. The header `kernel_lib/perf_instrumentation.hpp` is carried from 9518ba45263. At most 44 markers per RISC, well under the 250 cap. Tools are in `perf_experiments/_coordinator/`.

### Measured breakdown (focus, before)
- Whole op: 45.3-47.1 us.
- Peel, each run removing stages cumulatively (us):

  | full | no compute | no read | no write | compute only | read only | write only | sync only |
  |---|---|---|---|---|---|---|---|
  | 47.1 | 36.8 | 33.5 | 41.5 | 21.5 | 27.1 | 23.7 | 4.7 |

  The flagged all-DRAM rows have the same shape: focus2 52.3 / 52.5 / 33.9 / 44.5 / 20.6 / 32.9 / 23.8 / 4.5, focus3 40.8 / 37.7 / 33.0 / 34.2 / 19.7 / 21.5 / 23.6 / 4.3.
- Per core: the work split was the problem. Slowest-core kernel time was 46 us against a p50 of 34 us. The 18 two-row cores sat at y=2..3 (`split_work_to_cores(row_wise=True)` places the larger group first). On NoC0 the first-row read takes ~25 us at y=2 and ~7.5 us at y=11, falling steadily with y. Those cores then ran two rows of compute back to back at ~9 us per row and finished at ~46 us. A typical one-row core finished computing at ~24 us but its writer ran until 37-42 us; the east side drains its DRAM writes last.
- Compute per row on the math thread (us, unstarved, 24 tiles): residual_add 0.88, row_mean 1.04, center 0.73, row_sumsq 1.63, inv_std 1.17 (one tile of real work), normalize 1.45, affine_scale 1.44, affine_shift 0.86.
- gamma/beta staging costs ~3.5 us of NCRISC time when run alone. On the focus it is off the critical path (compute_wait_affine is ~0).
- Ranked headroom:
  1. Read placement and NoC balance (the y-gradient, with the two-row cores on the slowest rows).
  2. Serial compute on the two-row cores.
  3. The east-side write tail.
- DRAM floor for the focus is ~27 us (12.6 MB at ~470 GB/s). The read-only peel reaches ~370-450 GB/s on DRAM sources.

### Ideas floated and their verdicts
Every subagent artifact is under `perf_experiments/<slug>/`. ns are device kernel time on the focus unless noted.

| idea | verdict | before -> after | domain / notes |
|---|---|---|---|
| `noc_balance`: heavy cores read on NoC1, bottom-half cores write on NoC0 | WIN, graduated | focus 46.4 -> 36.9 us | Host only. Every option was bit-identical to baseline. Carve-out measured (see below). |
| `affine_mcast`: one injector reads compact gamma/beta and multicasts it; every core scatters locally | WIN, graduated | focus flat (46.4 -> 46.2); affine TILE DRAM 76.1 -> 52.9; L1 no-residual 27.5 -> 23.6 | Correct and faster or flat in every placement and layout tested, at 1/2/32/48/110 active cores, under `--dev`. |
| `core_placement`: two-row cores moved to fast-read cores | WIN alone, superseded | alone 46.3 -> 43.8; on top of noc_balance 36.9 -> 39.2, and 32x32 22.3 -> 26.9 | noc_balance fixes the same gradient better. The two together measured slower, so reverted. |
| `stats_fusion`: center sumsq, collapse and rsqrt in one DEST window; rsqrt on column-0 loads only | isolated WIN, not graduated | stage 2704 -> 2235 ns/row; whole op on top of the graduated DM: focus flat, 8x384 -1.4 us, 32x32 -1 us, but focus2 median ~49.8 -> ~51.4 us and focus3 +~1 us | Faster compute moves the DRAM write phase onto the new NoC schedule. Both regressed rows are flagged, so neither compute change graduates in this round. Re-test in round 2 against the re-measured DM schedule. Kernel: `perf_experiments/stats_fusion/graduate/`. |
| `normalize_affine`: seed DEST with the beta row-broadcast, then ELWMUL accumulates n*gamma (helper chain only) | isolated WIN, not graduated | stage flat (3966 -> 3976 ns/row); whole op same pattern as stats_fusion (seed alone: focus2 51.8-52.3 vs 46.8-50.2) | Also lowers rel-RMS 0.00190 -> 0.00181. Kernel: `perf_experiments/normalize_affine/graduate/`. |
| `fuse_add_mean`: accumulate the row sum during residual_add or as chunks arrive | NULL | 46159 -> 46686 to 47468 | The accumulation is only ~500 ns/row; residual_add is packer-bound. Streaming the mean regressed L1 no-residual 26.3 -> 28.1 us. |
| `three_pass_restructure`: sumsq without materializing c, plus one fused output pass | REGRESSION | 46653 -> 51212 | Under fp32 DEST each dest-reuse op costs ~86 ns/tile, more than an fp32 L1 round trip (41-81 ns). The parked fused path (mode 1) measured 47790. |

Precision options in the menus: none graduated. stats_fusion and normalize_affine stay inside the contract (rel-RMS 0.0019 / 0.0018) and change no precision knob.

### What graduated
1. **noc_balance**, in `_core_nocs` in `layer_norm_tile_program_descriptor.py`.
   - Reader and writer are emitted per (reader NoC, writer NoC) group through `DataMovementConfigDescriptor`. The kernels address through `noc_index` and are otherwise unchanged.
   - When the residual is read on the writer RISC (split reader), both RISCs read, so they must use different NoCs; the writer's choice wins.
   - Carve-out, earned by a measured regression: when most cores own two or more rows, every core keeps reader NoC0 / writer NoC1. With the writer-half rule alone, a 25x256 case measured 70.6-72.3 vs 63.9-68.6 us. No SUPPORTED shape on 110 cores reaches that regime (at most 128 tile-rows), but smaller grids do.
   - The `NOC_BALANCE` knob was deleted, leaving one path.
2. **affine_mcast**: new `CB_AFFINE_SCRATCH` (id 9; 6 KB TILE / 3 KB ROW_MAJOR), counted in `select_blocking`'s budget and in `l1_ledger.md`.
   - One `ttnn.Mcast(handshake=False)` from the last core is attached to every reader group, replacing the per-core affine reads. The old `issue_affine_reads` / `finish_affine` path was deleted.
   - The injector always keeps the default reader NoC0 / writer NoC1 pair. When noc_balance gave it the same NoC on both RISCs, the golden suite hung in the writer's final `noc_async_write_barrier` on that core. One hang also hit the swapped pair (reader NoC1 / writer NoC0), though that pattern is not understood. Pinning to the pair the multicast was validated on fixed both: three full golden runs green.
3. **Cleanup**: the parked DEST-reuse `FUSE_NORMALIZE_AFFINE` path was deleted (kernel, host knob, CT arg 16, knob test). It measured slower again this round (47.8 vs 46.7 and 48.7 vs 46.4 us). The default path is unchanged; A/B on focus2/focus3 was equal.

### Whole op, before (c6c759736aa) -> after (554c17896db)
Three alternating runs each, us; stock ttnn.layer_norm in parentheses.

| row | before | after |
|---|---|---|
| focus1 in L1 / res DRAM / out DRAM (55.0) | 46.0-46.4 | 36.6-37.8 (-20%) |
| focus2 all DRAM + res (70.0) | 51.4-52.2 | 50.2-51.0 (-2%); other runs in this round ranged 46.6-52.1, depending on what ran before it |
| focus3 all DRAM no res (48.3) | 40.4-41.6 | 33.4-34.5 (-18%) |
| all L1 + res (43.3) | 27.9-29.2 | 26.0 |
| L1 no res (34.5) | 25.1-28.0 | 22.6-23.3 |
| 8x384 all DRAM (47.0) | 43.0-43.5 | 38.8-39.3 (-10%) |
| 32x32 all DRAM (24.8) | 22.1-22.5 | 22.2-22.4 (flat) |
| affine TILE in DRAM (125.4) | 75.2-77.3 | 48.1-50.4 (-36%) |

Guard set (one representative per remaining path):

| row | before | after |
|---|---|---|
| focus1, bf16 DEST | 45.1-45.9 | 36.6-38.2 |
| 3x500 (h_non_aligned, 48 cores, mixed tiers) | 24.9-25.1 | 25.0-25.6 (flat) |
| 64x17 all DRAM no res, TILE affine | 25.7-25.8 | 24.3-25.0 |
| 2x512 all L1 + res | 15.5-15.8 | 15.3-15.4 |

No regression beyond noise. Golden: 545/545 after every graduation. Op unit tests: 110/110.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| local_copy_helpers (stateful local NoC read) | capability | The multicast scatter is a strided gather. For each tile column it reads 32 B face rows from the compact block to slot+0 and slot+FACE_BYTES, so both source and destination move every tile. The stateful read pins one source address, so it cannot express a per-transfer source. | n/a (not expressible) | part of the graduated path (affine TILE DRAM 76.1 -> 52.9 us whole op) | kernels/layer_norm_tile_reader.cpp:104 |

The multicast itself uses `ttnn.Mcast` / `McastArgs` / `SenderPipe::send()` unchanged.
