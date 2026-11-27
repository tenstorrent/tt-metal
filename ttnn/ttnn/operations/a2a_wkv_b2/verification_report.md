# Verification Report: a2a_wkv_b2

Board: Blackhole LoudBox `bh-lb-120-a08u32`, mesh 2×4, `FABRIC_2D_TORUS_X`, `cluster_axis=1` (2 rings of 4).
Verified at the implementer's R1 build (`a66ed26be01`).

## Summary

- **Correct and contract-clean**: the acceptance suite passes 10/10 and the golden suite 7/7. That includes back-to-back bit-identity and trace capture/replay. The verifier CLI shows 0 in every loud category.
- **Precision**: PCC 0.99991, relative RMS 0.0177 on every input distribution tried. The gate is PCC ≥ 0.9995.
- **Performance is the open item**: **131.6 µs per call** under trace replay, against a SOL of 43.8 µs (3.0×) and a roofline of about 49.5 µs (2.7×). The device profile puts the critical path on the **fabric_tx sender path**, not on DRAM.
- **TARGET − SUPPORTED is empty** (one cell), so the refinement queue is all perf.

## Code Review

### Prompt rules (`eval/prompts/a2a_wkv_b2.txt` `## Rules`) — all MUST rules hold

| Rule | Status |
|---|---|
| Derive ring position / neighbours from the mesh; no hard-coded chip ids, ring order, ring count | ✓ `_ring_streams(n, hpc)`, `get_fabric_node_id`, `_coord_with`. `_RING_SIZE = 4` appears only in `validate()`, as the shape contract the prompt itself requires ("`cluster_axis` not 4 chips long" → error). |
| Each ring only talks within itself; short way round; antipode split over both directions | ✓ The plan iterates only coordinates that share every index except `cluster_axis`. The antipode chunk is split into two half-streams. |
| Grid / core coordinates from the device | ✓ `compute_with_storage_grid_size()`, `worker_core_from_logical_core`. |
| attn tilize in-kernel | ✓ `read_sticks_for_tilize` + `compute_kernel_lib::tilize`. |
| Persistent state left as the next call expects | ✓ The `GlobalSemaphore`s are reset right after each wait (ready, done). `test_back_to_back_calls_bit_identical` passes. |
| Result independent of timing | ✓ Each output tile is one core's fixed-order K accumulation. |
| Trace-capturable (no host reads / syncs / non-replayable allocs during a call) | ✓ Semaphores are created on the first eager call and cached. `test_trace_replay_matches_eager` passes (see the harness note below). |
| Output L1 interleaved, bf8_b, TILE | ✓ |
| One native dispatch per call | ✓ One `ttnn.generic_op` over one `MeshProgramDescriptor`. `device_num_programs = 1` for the golden call. |

### Fixes applied

1. **DRY — scatter cap restated in the kernel.** `a2a_wkv_b2_tx_writer.cpp` declared `uint64_t dst_addrs[4]`, a literal copy of the host's `SCATTER_MAX_CHUNKS`. The array is now sized by the CT arg `tiles_per_packet`, and `static_assert(tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS)` guards it. Turning the `tiles_per_packet` knob now changes one place.
2. **DRY — tile height restated in the kernel.** `a2a_wkv_b2_reader.cpp` hard-coded `tile_h = 32`, while the host derives `Mt` from its own `TILE`. The host now passes `TILE` as CT arg 10, and the accessor args move to offset 11.

Both fixes were re-verified: acceptance 10/10, golden 7/7, precision and timing unchanged.

### Blocking-model fidelity

- **Knobs are single-source host constants**, and every CB size and loop count is derived from them. The knobs are `MT_BLOCK`, `ATTN_RM_DEPTH`, `TX_DEPTH`, `tiles_per_packet` (derived) and `heads_per_core` (from `split_work_to_cores`).
- **Ragged handling exists.** `mt_rows` is computed per block in all five kernels, so raising `MT_BLOCK` past an exact divisor of `Mt` stays legal.
- **`cb_out` spans `Mt`.** It is sized `heads_per_core·Mt·Nt`, which is a whole-op dimension, but this is the design's deliberate no-back-pressure residency. It is bounded: `Mt = 5` by the shape contract and `heads_per_core ≤ 2` on any grid of 52 cores or more. The ledger records it. Not a collapsed knob.
- **Expression check.** The worker reader issues one barrier per W head and one per 32-stick tile row. Compute is block-expressed. The worker writer issues one barrier per published block. The fabric_tx reader synchronizes per packet, but block = packet = `Nt` tiles by design, so this is not a half-turned knob. **It is, however, the measured perf bottleneck** (see Performance) and is the subject of Refinement 1.

### Design conformance

The build matches `op_design.md` R1 on every binding dimension:

- **Algorithm:** matmul-first.
- **Pipeline topology:** worker reader → tilize → matmul_block with W retained → publish; fabric_tx pull → scatter send.
- **Work split:** 96 heads over 96 workers plus 4 fabric_tx cores. The 20 idle cores are the design's choice: R3 is rejected because it would re-read W.
- **Handshake protocol:** ready/done `GlobalSemaphore`s.
- **CB placement:** uniform across all program cores.

`pipeline_spec.json` matches the build in its dtypes, page sizes, bank layout and stage names.

### Helper usage

The build uses the stock helpers where they fit:

- `read_sticks_for_tilize`, `tilize`, and `matmul_block` (`WaitAndRetainOnLastBlock` for W, K in one block);
- the `FabricStreamSender` / `arm_scatter_write` / `arm_inc` fabric helpers.

The raw paths each have a design-documented helper mismatch:

- W tile reads;
- the remote pull;
- the local publish.

No multicast is hand-rolled. Nothing to change.

### Correctness checks

- CB push/wait counts balance. `cb_out` uses a cumulative `cb_wait_front` and a single final pop.
- TensorAccessor is used everywhere.
- Every kernel uses `void kernel_main()`.
- Includes are `api/dataflow/dataflow_api.h`.

### Harness finding (not an op defect)

**A default `eval/eval_test_runner.sh` run reports `test_trace_replay_matches_eager` FAILED**, with `TT_FATAL: Cannot load new binaries during trace capture`. That leaves the device hung and errors the 3 tests after it.

- **Cause:** with `EVAL_CAPTURE_L1=1` (the runner's default on hardware), `eval/l1_profiling.py::start_for_item` calls `device.disable_and_clear_program_cache()` for the whole test body. Any trace-capture test then fails whatever the op.
- **Evidence:** the same file passes 5/5 under `run_safe_pytest.sh`, and the whole golden suite passes 7/7 with `EVAL_CAPTURE_L1=0`. Those are the numbers reported here.
- **Action:** the harness owner should exempt trace tests from L1 capture, or re-enable the cache around capture. Tests are read-only for agents.

## Registry Conformance

- `INPUT_TAGGERS = {}` (one shape, as the prompt specifies).
- `SUPPORTED = {dtype: [bf16], layout: [ROW_MAJOR], weight_dtype: [bf8_b], fp32_dest_acc_en: [False]}`.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the shape contract (`ValueError`). It is the first call in `a2a_wkv_b2()`.
- The op file does **not** declare `INVALID`.
- `num_links != streams per direction` raises `NotImplementedError` (regime R6, deferred). `num_links` is not a TARGET axis, so this is not a refinement candidate.
- `default_compute_kernel_config()` is exported and shared with `axes.py`.
- No auto-fixes to SUPPORTED were needed (no XPASS).

**INVALID audit** (`feature_spec.py`): `INVALID = []`. That is correct: one meaningful cell, and attn is ROW_MAJOR bf16, so the canonical bf8b+ROW_MAJOR entry would not be in the universe anyway. No cross-tensor couplings. Nothing to report.

## L1 Ledger Audit

- **Currency:** all 5 CBs have rows, and the size expressions match `create_mesh_program_descriptor`. The total is 198,400 B per core.
  - `cb_weight` 64×1088
  - `cb_attn_rm` 32×2048
  - `cb_attn_tiled` 16×2048
  - `cb_out` 20×1088
  - `cb_tx_stage` 8×1088
- **Capacity vs live set:**
  - `cb_attn_rm` and `cb_tx_stage` exceed their live sets by a stated double-buffer.
  - `cb_out` spans `m` and `n` because fabric_tx reads it in place with no back-pressure; justified and bounded.
  - Nothing under-spans: `cb_weight` spans k and n, and both are live across the head.
- **Page format vs DEST width:** `fp32_dest_acc_en = False`, and no CB uses a Float32 page. bf16 / bf8_b pages are right in both directions.
- **Disjoint lifetimes:** every pair is either concurrent (pipelining) or justified in the "Shares with" column. The worker-side `cb_tx_stage` (8.7 KB) exists only for address uniformity, and the ledger says so.
- **Bounds:** `Kt`, `Nt`, `Mt` come from the shape contract. `heads_per_core = ceil(H / num_workers)` is ≤ 2 on grids of 52+ cores. `mt_block · Nt ≤ 8` (DEST). The total is closed-form.
- **Data-movement budget:** present and consistent with the code.
  - DRAM: attn ×1, W ×1 (all 96 heads, as matmul-first needs).
  - NoC: 1.57 MB pulled, 0.52 MB written locally.
  - Fabric: 522,240 B per link direction.
  - Cheapest-traffic split: implemented (matmul-first). No deferred row is needed.
- **Block-size defaults:** interleaved, spread over 96 cores. `mt_block = 1` departs from "coarsest block that fits"; the opgraph ranking (51.7 vs 70.2 µs for `mt_block = 5`) supports that, but it is a calculator ranking, not a measurement. It is folded into Refinement 3's measured knob co-tune.
- **Footprint terms:**
  - `mt_block` scales `cb_attn_rm` and `cb_attn_tiled` (98 KB);
  - `tx_depth · tiles_per_packet` scales `cb_tx_stage`;
  - `heads_per_core` scales `cb_out`.

  There is about 1.26 MB of L1 headroom for deeper tx staging and larger blocks.

No ledger finding needs a refinement of its own.

## Precision Baseline

`tests/ttnn/unit_tests/operations/a2a_wkv_b2/test_a2a_wkv_b2_precision_baseline.py`. The op has **one legal shape**, so the baseline sweeps the input distribution instead of shapes. All 8 chips are gated together against an fp64 reference built from the device-quantized W. Config is HiFi2 with `fp32_dest_acc_en = False`.

| Input (per chip attn [1,96,160,512]) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median [p5, p95] |
|---|---|---|---|---|---|
| N(0,1), seed 0 | 0.999907 | 0.149 | 0.0155 | 0.01772 | 1.0128 [0.949, 1.086] |
| N(0,1), seed 1 | 0.999907 | 0.157 | 0.0155 | 0.01772 | 1.0128 [0.949, 1.086] |
| N(0,1), seed 2 | 0.999907 | 0.160 | 0.0155 | 0.01773 | 1.0128 [0.949, 1.086] |
| N(0,1)·0.01 | 0.999909 | 0.00144 | 0.000153 | 0.01739 | 1.0127 [0.950, 1.085] |
| N(0,1)·8 | 0.999907 | 1.390 | 0.124 | 0.01773 | 1.0128 [0.949, 1.086] |

The golden suite reports the same: PCC 0.999907, RMS 0.0177, ULP p99 ≈ 2.5e7 (an fp32 ULP measure; not meaningful for a bf8_b output).

**Scale-vs-precision triage.** PCC is high and the median ratio sits about 1.3% above 1, so I checked for a structural scale bug.

| Path | Least-squares slope (got vs true) |
|---|---|
| Host bf8_b quantization of the reference | 0.99999 — the output format alone is unbiased |
| Device, LoFi | 0.997 |
| Device, HiFi2 | 1.011 |
| Device, HiFi3 / HiFi4 | 1.012 |
| Stock `ttnn.matmul` comparator (probe only, same dtypes and config, chip-local) | 1.04 (LoFi) – 1.05 (HiFi2 / HiFi4) |

Conclusion: the bias is **Blackhole's 16-bit DEST accumulate / pack rounding**, not an op defect.

- It grows with the number of fidelity phases accumulated into DEST.
- The stock matmul, which spills K-block partials, is about 4× worse.
- The ratio spread is broad (p5–p95 ≈ 0.95–1.09). A structural bug would show a tight cluster.

The lever that would remove it, `fp32_dest_acc_en = True`, is outside TARGET (`fp32_dest_acc_en: [False]`). The probes are in `tests/.../a2a_wkv_b2/probes/test_probe_scale.py`.

**Assessment:** precision is comfortably inside the 0.9995 PCC gate (margin about 4e-4), with no outliers and no non-finite values.

**Recommended tolerances:** the defaults hold — `helpers.TOLERANCES` / PCC 0.9995.

## Verifier CLI Summary

Golden run `EVAL_CAPTURE_L1=0 eval/eval_test_runner.sh eval/golden_tests/a2a_wkv_b2/`: 7/7 passed. `verifier_report.json` is copied next to this file.

- supported_pass: 2 (`test_op`, `test_op_loose`)
- xfail_expected: 0 (TARGET == SUPPORTED: there is no outside cell)
- invalid_skipped: 0
- no_axes_found: 5 (regression tests: back-to-back, trace replay, 3× shape contract; all passed)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

**TARGET − SUPPORTED = ∅** on every axis (dtype, layout, weight_dtype, fp32_dest_acc_en). No generality refinement exists to file, and that is documented here rather than left as a gap.

## Performance (baseline for the perf queue)

**Wall time.** `test_a2a_wkv_b2_trace_timing.py` (trace capture of one call, 50 replays): **131.6 µs per call**.

**Golden `device_kernel_ns`.** This is summed over the 8 chips' records, so divide by 8: ≈ 126 µs per chip for the golden call, about 240 µs per chip-call in the eager back-to-back test (launch skew included).

**Device profile** (`TT_METAL_DEVICE_PROFILER=1`, steady-state calls 3–4, all 8 chips; times from program start):

| Role | End time (µs) |
|---|---|
| Worker reader (DRAM W + attn) | 59.1 – 60.5 |
| Worker compute (tilize + matmul) | 64.5 – 66.0 |
| Worker writer (publish) | 64.7 – 66.1 |
| fabric_tx neighbour streams (link 0, 24 heads → 120 packets) | **104 – 120.6** |
| fabric_tx antipode half-streams (link 1, 12 heads → 60 packets) | 75 – 92 |

**Reading the profile:**

1. **The critical path is the fabric_tx sender path.** The neighbour streams finish about 55 µs after the last worker block is ready. That is about 1 µs per 4,352 B packet, an effective ≈ 4.3 GB/s per sender against a 50 GB/s channel rating. Per packet, each stream runs a serial chain:
   - fabric_tx reader: remote 4 B semaphore poll, then barrier;
   - fabric_tx reader: 4,352 B pull, then barrier;
   - fabric_tx writer: scatter write, then `flush`;

   with only `tx_depth = 2` slots in flight.

   The planner's ring model agrees. `opgraph.py … ring_builder.py --opt mt_block=1 --opt cores=96 --opt antipode_split=True` gives 51.7 µs at 50 GB/s per channel and 148.7 µs with `line_rate_gbs` set to the measured 4.4, against 118–131 µs measured.
2. **Per-sender load is 2:1 unbalanced.** Neighbour senders send 120 packets and antipode senders 60, although link *bytes* are balanced: link 1 also carries the relayed antipode half at router level.
3. **The worker DRAM stream (≈ 59.6 µs) is the next bound.** It is 36% above the 43.8 µs DRAM SOL and slightly above the planner's NoC 0 delivery estimate of 54.2 µs. This is the split-NoC lamp in `op_design.md`.
4. **Compute adds ≈ 5 µs after the reader ends.** That is the last-round tail: the design's drain estimate is about 2.6 µs.

## Recommendations

- **Perf queue order:** fabric_tx path first (Refinement 1). It alone moves the per-call time from about 120 to about 65 µs. Then the worker read stream (Refinement 2). Then the scheme-change / tail work (Refinement 3).
- **No L1 pressure:** 198 KB of about 1.46 MB is used, so deeper `tx_depth`, larger packets or a larger `mt_block` are all affordable.
- **Harness:** fix the L1-capture / trace-test interaction (see Code Review). Until then, run the golden suite with `EVAL_CAPTURE_L1=0` to get an honest result for this op.
- **Precision:** no lever is in scope. `fp32_dest_acc_en = True` would remove the ~1% DEST rounding bias but is not in TARGET.
