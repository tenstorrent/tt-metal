# Verification Report: all_gather_rms_norm

Box: Blackhole QuietBox 2 (4× p300c). Golden `test_op` cells run on the 2×2 system mesh (Linear; Ring needs wrap
links and is pruned there). Ring is exercised on the 1×4 snake-ring mock by `test_ring_mock.py`, the LOOSE perf
cases and the acceptance test. All under FABRIC_2D_TORUS_XY with the 14 KiB + 64 B router payload.

## Code Review

Fixed:

1. **Race in the in-chip partial combine (correctness bug, fixed).** The golden suite showed 5 intermittent failures
   on the 2048-row shapes (cluster_axis 0, L = 1): two with `inf` outputs (bf16 / bf8b, fp16 DEST) and three fp32
   cells at rel-RMS ≈ 0.007 (target 0.005). Reruns passed or failed at random. Triage:
   - A sequence probe replaying the golden cells reproduced it in 3 of 20 calls (`seq_probe`, verifier scratch).
   - The bad elements were whole member ranges (tile columns 70–79 / 60–69 = members 6–7 of k = 8) on a few
     tile-rows, on every device, including where that slice is the device's own. So the transport is not involved.
   - got/true was **constant per row** (≈ 1.07 on some rows, 0.00078 on others): a wrong inv_rms, i.e. the
     high-PCC / scale signature, not rounding.
   - Cause: every member waited on **one cumulative** partials semaphore (`≥ k·(b+1)`). A fast peer's block-(b+1)
     increment could complete the count while a slow peer's block-b increment was still in flight, so combine(b)
     read a stale slot: block b−2's partial (ratio ≈ 1) or the previous program's bytes (inf). NoC congestion right
     after a program switch widens the window, which is why repeating an identical call never showed it.
   - Fix: one partials semaphore **per block parity**, for both the symmetric and the owner combine. A peer can lead
     by at most one block (its partial(b+2) needs this member's partial(b+1), which exists only after combine(b)),
     so a per-parity count completes only with all k partials of block b. After the fix the probe shows 0 of 40 bad
     calls, and the full golden suite is green. Recorded as a Mechanism-caps row in `op_design.md`. The base op has
     the same single-counter pattern (not fixed there: out of scope).
2. **Per-stage device zones gated** behind `AGRMS_PROFILE_ZONES=1`. `DeviceZoneScopedN` per block on every member
   compiled in whenever the profiler was on. Under the eval CCL perf harness (500 traced calls) that filled the
   profiler DRAM buffers ("markers were dropped"), added marker cost to the headline number, and drove host memory
   up on the 640×1792 case.
3. **DRY**: `GAMMA_READ_ALIGN` was a literal in both the host (cb_gamma page stride) and the writer (gamma reads). It
   is now passed as a writer CT arg from the one host constant.
4. **Program-cache key**: `dst_full_sync_en` added to `custom_program_hash`. Before, two calls differing only in it
   aliased the same cached program.
5. Stale comment (`cb_inv_owned` "reserved") corrected. `l1_ledger.md` owner-combine CB sizes and semaphore list
   brought current.

Checked, no change needed:
- **Blocking-model fidelity**: `BLOCK_ROW_TILES`, `OUT_UNIT_CAP`, `OUT_DEPTH`, `COMPOSE_DEPTH`, `ITEM_MIN_CHUNKS`,
  `PORT_CB_BYTES` / `group`, `RELAY_LAG` and `K_SYM_MAX` are each defined once on the host. Every CB size, loop trip
  count and kernel CT arg derives from them. No inlined block count, no duplicate literal (after fix 3).
  `cb_x_landing` scales with `num_blocks_max · w_pitch`. That is the design's mechanism cap (fabric writes without
  credit) and `select_split()` guards it against the L1 budget (`RuntimeError` otherwise, `dram_landing` deferred),
  so it is not a collapsed knob.
- **Expression**: the reader waits / pushes once per block, compute runs one chain call per block, and the writer
  drains per `out_unit` with one flush. The port reader batches `group` chunks per barrier and the sender flushes
  per batch. No per-unit handshake inside a planned block.
- **Design conformance**: algorithm (raw-slice relay gather into member L1, every chip normalizes the full row, no
  partial crosses the fabric), topology (balanced ring / line, one port per direction × link, NoC1 senders next to
  their Ethernet cores), round-robin row groups of slice-nested members, and the symmetric (k ≤ 8) / owner (k > 8)
  combine all match `op_design.md`. The grid is filled: 96–104 member cores on QB2, both dataflow halves batched.
- **Helpers / API**: `reduce<>` for the collapse and the symmetric combine, `eltwise_chain` for the normalize,
  `calculate_and_prepare_reduce_scaler`, `fill_l1_range`, `TensorAccessor`. The raw pieces (pass-1 batching,
  column-0 finalize, gamma ROW-broadcast dest-reuse, owner combine, fabric sender) are documented helper bypasses
  inherited from the base op / example. Kernel syntax and include paths are current.
- **Prompt rules**: one fused native `generic_op` per call (one MeshProgramDescriptor). No other norm / CCL op and
  no host round trip. Every member returns the full row. The op never changes the fabric config.
  `default_compute_kernel_config()` is the single default.

Deferred to refinements (architectural): native shard-resident placement (Refinement 1).

Advisory (recorded, not a refinement):
- **Linear line ends**: a receive-only port (no outgoing entries) notifies its members only after all its items
  arrived (`has_out == false` skips the per-step notify loop), so a Linear group's end chips get no overlap on that
  stream. Ring, the production topology and every perf case, is unaffected. Noted under Refinement 4.

## Registry Conformance

- `INPUT_TAGGERS` (`alignment`, `rank`, both `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS`, `validate()`: present and
  wired. validate() is the entry point's first line and checks SUPPORTED per axis, then EXCLUSIONS
  (`UnsupportedAxisValue` / `ExcludedCell`). Caller errors raise `ValueError`.
- The op file does not declare `INVALID`.
- No XPASS drift: SUPPORTED needed no auto-fix. SUPPORTED equals TARGET on every axis.
- **INVALID audit** (`feature_spec.py`): only the 6 gamma "none" canonicalization entries. Each couples axes of the
  same tensor (gamma), and each describes a cell that cannot exist. Well-formed. The bf8b + ROW_MAJOR activation entry
  is correctly absent: the input layout is TILE-only and gamma_dtype has no bfloat8_b, so that cell cannot arise
  (the spec says so).

## L1 Ledger Audit

- **Currency**: all 11 CBs (indices 0–10) have rows. Capacities and formats match `cb_plan()` / the port CB. Fixed:
  the owner-combine `cb_gathered_sumsq` / `cb_inv_owned` sizes (now the closed forms `ceil(b/k)·ceil(k/32)` /
  `ceil(b/k)`), and the semaphore list (+ `sem_partials_odd`).
- **Capacity vs live set**: `cb_x_landing` capacity = live set (every block resident: remote writers have no credit,
  mechanism cap). All other CBs are block- or window-sized, matching their live sets. No over- or under-span findings.
- **Page format vs DEST**: T_acc CBs follow `fp32_dest_acc_en` (Float32 / Float16_b). Float32 input-dtype pages
  occur only with fp32 input, which requires fp32 DEST (EXCLUSION). No findings.
- **Disjoint lifetimes**: every row carries a concrete reason. No findings.
- **Bounds**: `num_blocks_max = ceil(R_t / n_groups)` is the one op-dimension term. It is guarded by the
  `select_split()` fit check (fallback: `RuntimeError`, `dram_landing` deferred), and every INPUTS shape fits (the
  ledger's worst cases at G = 8 ≤ 1.31 MiB).
- **Block-size defaults**: interleaved spreads rows round-robin across every worker group. `block_row_tiles = 1` was
  chosen by design reasoning (rows land progressively), not measured. Measuring 2 is folded into Refinement 4.
- **Footprint**: `worker = num_blocks_max·w_pitch·T_in (landing, scales with row blocks × member width) +
  gamma (w_pitch) + 3·T_acc + [k>1]·depth_g·k·T_acc + windows (2·out_unit·T)`. `port = landing + 2·group·run_pages·T_in
  (≈ 112 KiB)`. Focus: ≈ 0.04 MiB worker / 0.14 MiB port.
- **Data-movement budget**: present and consistent with the implemented split. x crosses DRAM once, out once, and
  the busiest link carries (G−1)/2·S. The cheapest-traffic split is the implemented one.

## Precision Baseline

`tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm_precision_baseline.py`. 1×4 ring, 2
links, gamma bf16 ROW_MAJOR sticks, HiFi4. Reference: torch rms_norm of the device-quantized gathered row in fp32.
Worst device shown. ULP is in the output dtype. r = got/true over |ref| > 1e-3.

| Shape (per chip) | Corner | PCC | Max Abs Err | Mean Abs Err | Rel RMS | ULP p99 / max | r median (p5, p95) |
|---|---|---|---|---|---|---|---|
| (1,1,32,64) | bf16, fp32 DEST | 0.999997 | 0.041 | 0.0020 | 0.0037 | 1.66 / 2.15 | 0.9973 (0.9930, 1.0012) |
| (1,1,32,64) | bf16, fp16 DEST | 0.999993 | 0.057 | 0.0022 | 0.0042 | 2.02 / 3.0 | 1.0020 (0.9959, 1.0083) |
| (1,1,32,64) | fp32 | 1.000000 | 0.011 | 0.00065 | 0.0012 | 3.6e4 / 5.5e4 | 0.9991 (0.9978, 1.0001) |
| (1,1,32,64) | bf8b, fp32 DEST | 0.999956 | 0.051 | 0.0074 | 0.0094 | 215 / 255 | 1.0001 (0.823, 1.136) |
| (1,1,640,384) FOCUS | bf16, fp32 DEST | 0.999997 | 0.083 | 0.0019 | 0.0037 | 1.61 / 2.34 | 0.9973 (0.9932, 1.0012) |
| (1,1,640,384) | bf16, fp16 DEST | 0.999993 | 0.106 | 0.0020 | 0.0040 | 1.88 / 4.36 | 1.0012 (0.9952, 1.0076) |
| (1,1,640,384) | fp32 | 1.000000 | 0.025 | 0.00059 | 0.0011 | 3.5e4 / 5.6e4 | 0.9992 (0.9979, 1.0002) |
| (1,1,640,384) | bf8b, fp32 DEST | 0.999951 | 0.115 | 0.0077 | 0.0099 | 225 / 257 | 1.0002 (0.744, 1.139) |
| (1,1,640,1792) | bf16, fp32 DEST | 0.999997 | 0.086 | 0.0017 | 0.0033 | 1.51 / 2.26 | 0.9978 (0.9936, 1.0016) |
| (1,1,640,1792) | bf16, fp16 DEST | 0.999994 | 0.115 | 0.0019 | 0.0036 | 1.68 / 3.27 | 1.0006 (0.9951, 1.0068) |
| (1,1,640,1792) | fp32 | 1.000000 | 0.023 | 0.00044 | 0.0009 | 3.0e4 / 5.3e4 | 0.9995 (0.9983, 1.0006) |
| (1,1,640,1792) | bf8b, fp32 DEST | 0.999953 | 0.110 | 0.0075 | 0.0098 | 222 / 257 | 1.0003 (0.791, 1.138) |
| (1,1,64,1000) W non-aligned | bf16, fp32 DEST | 0.999997 | 0.063 | 0.0021 | 0.0038 | 1.62 / 2.31 | 0.9971 (0.9931, 1.0009) |
| (1,1,64,1000) | bf16, fp16 DEST | 0.999992 | 0.136 | 0.0033 | 0.0062 | 2.69 / 4.53 | 1.0048 (0.9986, 1.0115) |
| (1,1,64,1000) | fp32 | 1.000000 | 0.021 | 0.00066 | 0.0012 | 3.7e4 / 5.9e4 | 0.9990 (0.9978, 1.0001) |

**Assessment**: no structural or scale bug. Errors are flat across shapes and regimes (k = 1, symmetric k = 8, owner
k = 12, regime N). bf16 stays within ~2 bf16 ULP at p99. There is a small systematic bias: r ≈ 0.997 for bf16 with
fp32 DEST and ≈ 0.999 for fp32. The p5–p95 spread is wider than the offset, so this is rounding, not a scale bug. It
is consistent with mantissa truncation: FPU source registers carry tf32 (10-bit mantissa) for x and inv_rms, and the
fp32 → bf16 pack adds roughly half an ulp. With fp16 DEST (round-to-nearest SFPU convert) the bias flips to
+0.1…+0.5 %. fp32 output is therefore ~1e-3 accurate, not ~1e-6: it is bounded by the FPU's tf32 operands. bf8b
shows the expected block-float noise (broad r spread centred on 1.0).

**Recommended tolerances**: bf16 PCC ≥ 0.999, rel-RMS ≤ 0.01 (fp16 DEST ≤ 0.015). fp32 PCC ≥ 0.9999, rel-RMS ≤ 0.003.
bf8b PCC ≥ 0.99, rel-RMS ≤ 0.02. These match the golden gates with margin (golden fp32 rel-RMS target 0.005 vs
measured ≤ 0.0012).

## Verifier CLI Summary

Merged over 13 shape chunks + 1 chunk for `test_ring_mock` / `test_regression` (`eval/eval_test_runner.sh … -k`), on
the code after the race fix. Compact report: `verifier_report.json` (this directory).

- supported_pass: 10,503
- xfail_expected: 3,000 (`{float32, fp32_dest_acc_en=False}` 2,280; `{bfloat8_b, w_non_aligned}` 720)
- invalid_skipped: 35,568 (gamma "none" canonicalization)
- infeasible_skipped: 180 (harness HEIGHT_SHARDED geometries of W_local ≥ 4096 whose shards do not fit L1; uncharged)
- no_axes_found: 66 (ring-mock + regression tests, not registry cells; all passed)
- supported_fail: **0**
- xpass_drift: **0**
- xfail_wrong_mode: **0**

**TARGET − SUPPORTED** is empty on every axis. Queue coverage of the xfail bucket:
- `{bfloat8_b, w_non_aligned}` (720 cells) → Refinement 2.
- `{float32, fp32_dest_acc_en=False}` (2,280 cells): **documented omission**. It is the prompt-mandated precision
  convention ("float32 input with fp32_dest_acc_en=False is refused, an op-side EXCLUSION"), permanent by
  specification, so no refinement.

**Harness / infrastructure findings** (not op defects; nothing for the queue):
1. **L1 profiling cannot measure this op.** With the runner's default `EVAL_CAPTURE_L1=1`, every DRAM-interleaved
   cell fails with `L1ProfilingError: peak_total=0`. A graph-capture probe around one call shows
   `circular_buffer_allocate` nodes are never emitted for this mesh `generic_op` (only `circular_buffer_deallocate_all`
   and buffer nodes). The CB allocation of a MeshWorkload program never reaches the graph tracker, so the peak is 0
   (interleaved) or only the L1 output buffer (sharded). The suite above was therefore run with `EVAL_CAPTURE_L1=0`.
   This needs a fix in tt-metal's graph tracking of mesh workloads or in `eval/l1_profiling.py`, not in the op.
2. **Host memory**: the box had ~17–19 GB available (≈ 220 GB held outside userland). The perf harness's profiler
   capture of the 640×1792 case at 5 experiments was OOM-killed twice. It was measured at 2 experiments instead
   (stdev 0.02 µs).
3. Under `--dev`, multi-test runs that reopen fabric meshes are unreliable on Blackhole (ERISC halted at teardown;
   the implementer's note). Every `--dev` check ran one test per process (8 regime-pinned cases, all PASS). Multi-test
   runs used production mode.

## Performance (eval harness, fastest chip, QB2 1×4 ring, 2 links, bf16, gamma RM sticks)

| Case | Fastest chip | Slowest chip | Spread | Roofline / target | Dispatch |
|---|---|---|---|---|---|
| 640×384 [FOCUS] | **31.36 µs** | 32.25 µs | 0.89 µs | 7.6 / 10.1 µs | not bound (gap 0.57 µs) |
| 640×512 | **41.51 µs** | 42.88 µs | 1.37 µs | 10.1 / 13.5 µs | not bound |
| 640×1792 | **99.89 µs** | 101.95 µs | 2.06 µs | 35.5 / 47.3 µs | not bound |

Focus breakdown (`AGRMS_ABLATE`, same harness protocol): full 31.36 · TRANSPORT-only 26.22 · LOCAL-only 14.73 ·
empty program 0.75 µs.
- **TRANSPORT** binds: 26.2 µs vs the link roofline's 7.6 µs, and ~8 µs slower than the `fabric_all_gather` example
  at the same tiles/chip (~18 µs on 2 links). The ports are packet-rate bound on 6- and 3-tile packets
  (implementer zones).
- **OVERLAP**: 5.1 µs exposed beyond the transport.
- **Local work**: 14.7 µs. Hidden today, but it becomes the binder once the transport is under ~15 µs.

Queue order follows this breakdown: TRANSPORT (R3) → OVERLAP (R4) → local (R5), after the two generality refinements
the 2:1 cadence puts first.

## Recommendations

- **Refinement order**: R1 (native sharding, a scheme-change) and R2 (bf8b × W-non-aligned) are the only generality
  left. The perf-1 contract (the FOCUS case's full config) is already fully in SUPPORTED, so the perf refinements
  need no unlock. R3 / R4 / R5 attack 31.4 → 10.1 µs in breakdown order.
- **Precision lever, not filed** (no failing cells): fp32 accuracy is capped at ~1e-3 by the FPU's tf32 operands in
  the normalize. If fp32 ever needs ~1e-6, the lever is an SFPU fp32 multiply for x·inv_rms·γ with `UnpackToDestFp32`
  on the landing tiles (`/numeric-formats-metal`).
- **The same single-counter race exists in the base op** (`base_ops/rms_norm` writer exchange, ported from there).
  It is out of scope here (do not edit the base op), but worth passing on to whoever owns it.
- **L1 pressure**: the resident landing grows with `R_t · land_w_tiles / busy cores`. At G = 8 the worst INPUTS shape
  sits at ≈ 1.31 MiB per worker (fp32 `(1,1,64,1000)`), close to the ~1.4 MiB budget. A longer prefill chunk or a
  larger G will need the deferred `dram_landing` regime (or row-chunked calls) before it raises `RuntimeError`.
