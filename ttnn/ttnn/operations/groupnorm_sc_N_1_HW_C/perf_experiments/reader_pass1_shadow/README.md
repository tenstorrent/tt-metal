# reader_pass1_shadow — hide the reader's constant generation in the DRAM shadow of chunk 0

Isolated bake-off (perf-lab style, multi-core, the op's exact pass-1 reader geometry) of the reader's pass-1
schedule. Baseline = the op's current reader (`kernels/groupnorm_sc_N_1_HW_C_reader.cpp`: scaler helper, then per
column group chunks {reserve, issue, barrier, push}, then E^T {reserve, zero-fill, barrier, lanes, push}),
reproduced verbatim for the TILE-input path. A BRISC consumer stub waits/pops exactly like the op's compute; in
`perf` mode it writes nothing, so DEVICE KERNEL DURATION = the slowest core's "pass-1 reader done" — the quantity
that sets the op's 110-core rendezvous.

Files: `reader_pass1_shadow_bench.py` (inline reader with 7 schedules via CT arg + consumer stub + `ttnn.ProgramDescriptor`),
`test_reader_pass1_shadow.py` (bit-exact gate: x tiles == input, E^T lanes == Python reference of
`write_membership_lanes(transposed=true)`, scaler/E^T pages == baseline dump; in-process device profiler; zone
readout from `generated/profiler/.logs/profile_log_device.csv`), `report_perf_sweep.md` (raw table written by the test).

Run: `scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/perf_experiments/reader_pass1_shadow/test_reader_pass1_shadow.py`
(env `RPS_CASES`, `RPS_REPEATS` (default 3), `RPS_REPORT`).

Precision: pure dataflow reorder — every variant lands bit-identical bytes in cb_x_pass1 / cb_scaler / cb_membership
(gated before every timing). No ComputeKernelConfig knob is involved.

## Mechanism facts (Blackhole, verified in `tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl`)

* `Noc::async_write_zeros` on an L1 destination is a NoC loopback **read** from `MEM_ZEROS_BASE` (512 B per
  command, on the READ cmd buffer); `write_zeros_l1_barrier()` is the **full** `noc_async_read_barrier()`. A zero-fill
  barrier issued while x reads are in flight therefore waits for the x reads too — the naive "issue x, then
  zero-fill + barrier" order destroys the shadow. The scaler helper (`calculate_and_prepare_reduce_scaler`) contains
  exactly that zero-fill + full barrier, which is why the candidates fill the scaler tile raw.
* Transaction ids: `noc_async_read_set_trid(t)` writes `NOC_PACKET_TAG` on the read cmd buffer and the tag persists
  for every later plain / stateful read (neither `ncrisc_noc_fast_read` nor `ncrisc_noc_read_set_state` touch it), so
  x reads tagged 2 and zero reads tagged 1 can be waited on separately with `noc_async_read_barrier_with_trid(1)`.
  This is the same mechanism the DRAM-sharded matmul reader uses.

## Variants

| variant | schedule |
|---|---|
| `baseline` | op today |
| `shadow` | per cg: reserve chunk 0 -> reserve E^T (+scaler once) -> zero-fill E^T (+scaler) -> full barrier (nothing else in flight) -> issue chunk 0 -> RISC fills scaler row-0 + E^T lanes, pushes both -> read barrier -> push chunk 0 -> other chunks as today |
| `shadow_trid` | issue chunk 0 FIRST (trid 2) -> reserve E^T -> zero-fill tagged trid 1 -> `barrier_with_trid(1)` -> RISC fills + pushes -> read barrier -> push chunk 0. The zero-fill is now inside the shadow too |
| `shadow_fastlanes` | `shadow_trid` + incremental in-group counter instead of `ch / Cg` per lane. **Measured regression**: `Cg` is a compile-time constant, so the divide was already a mul-shift; the per-lane compare/branch is slower (r_memb_fill 0.94 -> 1.30 us) |
| `shadow_tightlanes` | `shadow_trid` + run-based lane writer: one divide per column group, then per (tile, group) run a stride-16-word pointer loop (r_memb_fill 0.94 -> 0.53 us for 64 lanes) |
| `shadow_lookahead` | `shadow_trid` + chunk rc+1's reads issued (trid 2/3 alternating) before chunk rc's `barrier_with_trid`; resident regime (L1 address = write_ptr + chunk bytes, the credits ring never wraps inside an image block) |
| `shadow_all` | trid + tight lanes + lookahead — the graduation candidate |

## Measured (box=bh-qb-11-special-mstaletovic-for-reservation-93463, arch=BLACKHOLE, median of 3 fresh launches per cell, perf mode)

Focus (1,1,1024,640) G=32 bf16, 110 cores (pr=10 x pc=11, Ht_core 3|4, Ct_core 1|2, cols 2, chunk_rows 2, resident):

| variant | ns | speedup | r_pass1 end p50 / max (us, zoned run) | r_x_barrier p50 / max | r_memb_fill p50 | r_zero_fill p50 |
|---|---:|---:|---|---|---|---|
| baseline | 5537 | 1.000x | 4.59 / 5.75 | 1.02 / 1.95 | 1.22 (zero-fill+barrier+lanes) | - |
| shadow | 4449 | 1.245x | 3.46 / 4.66 | 0.47 / 1.21 | 0.94 (lanes) | 0.41 |
| shadow_trid | 4076 | 1.358x | 3.43 / 4.35 | 0.29 / 1.09 | 0.94 | 0.42 |
| shadow_fastlanes | 4198 | 1.319x | 3.69 / 4.64 | 0.27 / 1.34 | 1.30 | 0.41 |
| shadow_tightlanes | 4040 | 1.371x | 3.00 / 4.12 | 0.44 / 0.99 | 0.52 | 0.41 |
| shadow_lookahead | 3873 | 1.430x | 3.26 / 4.19 | 0.20 / 1.16 | 0.94 | 0.42 |
| shadow_all | 3833 | 1.445x | 3.02 / 4.00 | 0.26 / 1.37 | 0.53 | 0.41 |

The baseline bench reproduces the op's own reader timeline (op profile: r_memb_fill end max 5.84 us; bench r_pass1
end max 5.75 us), so the ~1.7 us taken off the slowest core here is what the op's 110-core rendezvous should see.

Domain sweep (speedup of `shadow_trid` / `shadow_tightlanes` / `shadow_all` vs baseline, medians of 3; full table in
`report_perf_sweep.md`):

| case | cores | block | baseline ns | shadow_trid | shadow_tightlanes | shadow_all |
|---|---:|---|---:|---:|---:|---:|
| floor (1,1,32,32) G=1 | 1 | 1x1, 1 chunk | 1400 | 1.31x | 1.47x | 1.42x |
| manychunk (1,1,256,160) resident | 1 | 8x5, cols 5, 3 chunks | 5834 | 1.03x | 1.12x | 1.19x |
| manychunk streaming (X_DEPTH=2) | 1 | 8x5, cols 5, 3 chunks | 5710 | 1.03x | 1.11x | 1.12x (no lookahead) |
| rect4 (1,1,256,256) | 4 | 4x4, cols 4, 2 chunks | 3997 | 1.06x | 1.22x | 1.23x |
| twocg (1,1,1024,1280) | 110 | 4x4, cols 2, 2 col groups | 9498 | 1.36x | 1.33x | 1.34x |
| threecg (1,1,64,768) | 1 | 2x24, cols 8, 3 col groups | 16853 | 1.05x | 1.61x | 1.60x |

`shadow_fastlanes` is 0.83-0.97x everywhere (recorded null; not a candidate). `shadow_lookahead` alone is flat on
rect4 / twocg / streaming (within noise) and +4..10% on focus / manychunk.

## Not covered here

* RM input (`load_x_chunk_sticks`): `read_sticks_for_tilize` reserves, issues, **barriers** and pushes internally per
  tile-row block, so there is no seam to put the RISC work between issue and barrier — the shadow is inexpressible on
  that path without a split helper (or raw stick reads). Untouched by this idea.
* `HW % 32 != 0` (partial scaler pair): the raw row-0 fill here covers the full tile only; the pair is the same fill
  with tile 1 limited to `hw_tail` valid positions (the helper's `fill_each_face_row0_partial`), or keep the helper
  call BEFORE the first read issue (its internal full barrier is harmless when nothing is in flight; only the 0.3 us
  scaler shadow is lost).
* Float16_b membership pages (16-bit DEST): the tight lane writer is written for 4-byte elements; generalise the
  column offset / row stride by `e_elem_size` (`tile_elem_offset` already does).
