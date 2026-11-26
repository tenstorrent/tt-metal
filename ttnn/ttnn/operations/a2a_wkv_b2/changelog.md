# Changelog: a2a_wkv_b2

## Phase 0 — Core Implementation
- **Date**: 2026-11-26
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). One `ttnn.generic_op` over a `MeshProgramDescriptor`.
  - Matmul-first: 96 heads over 96 worker cores per chip, with an in-kernel tilize of the ROW_MAJOR attn and `matmul_block` with W resident.
  - 4 fabric_tx streams per chip (neighbours plus an antipode half each way) scatter-write bf8_b results straight into each destination's L1-interleaved output.
  - Cross-call ready/done handshakes on cached `GlobalSemaphore`s.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[ROW_MAJOR_LAYOUT], weight_dtype=[bfloat8_b], fp32_dest_acc_en=[False]. No shape-derived axes. This equals TARGET.
- **Accuracy achieved**: PCC=0.99991, max_abs_err=0.149–0.160 (N(0,1) inputs), rms_err (relative)=0.0177, measured on 5 input distributions via `test_a2a_wkv_b2_precision_baseline.py`.
  - The op has one legal shape, so the baseline varies seeds and magnitude ×0.01 / ×8.
  - Median got/true ratio is 1.013: Blackhole 16-bit-DEST rounding bias, not a scale bug. Stock `ttnn.matmul` shows 1.05.
- **Performance**: 131.6 µs per call (trace replay) against a SOL of 43.8 µs. The critical path is the fabric_tx sender path (about 1 µs per 4.3 KB packet). Workers finish at about 65 µs.
- **Golden suite at Phase 0**: 7 / 7 passing, with `EVAL_CAPTURE_L1=0`.
  - Per `verifier_report.json`: supported_pass 2, no_axes_found 5 (all passed), 0 in every loud category.
- **Issues encountered**:
  - Verifier DRY fixes:
    - the tx writer's `dst_addrs[4]` is now sized by `tiles_per_packet`, with a `static_assert` against `NOC_SCATTER_WRITE_MAX_CHUNKS`;
    - the worker reader's hard-coded `tile_h = 32` now comes from the host `TILE` CT arg.
  - Harness: the eval runner's default L1 capture (`EVAL_CAPTURE_L1=1`) disables the program cache for each test. That makes `test_trace_replay_matches_eager` fail at capture and hangs the device for the tests after it. The op passes the test outside that mode. Reported in `verification_report.md`.
- **Tests added**:
  - `test_a2a_wkv_b2_precision_baseline.py`;
  - `test_a2a_wkv_b2_trace_timing.py` (perf baseline);
  - probes `probes/test_probe_scale.py` (scale-bias triage) and `probes/test_probe_profile.py` (device-profiler run).
  - `test_a2a_wkv_b2.py` (acceptance) came from the planner.

## Refinement 1 — Speed up the fabric_tx sender path (perf-flagged profile)
- **Date**: 2026-11-26
- **What was done**:
  - **Reader (`a2a_wkv_b2_tx_reader.cpp`)**: the stream's packets are pulled in batches of `TX_BATCH` packets (aligned to mt rounds). Each batch has one pull barrier, not one barrier per packet.
    - Publish polls are cached per head: `sem_blocks_done` is monotonic, so a head whose last-seen value covers the block is not polled again.
    - All polls of a batch sit behind one barrier. A head that is still short is re-polled on its own.
  - **Writer twin (`a2a_wkv_b2_tx_writer.cpp`)**: waits for a whole batch, issues every packet, then one `flush` and one pop per batch (was a flush + pop per packet).
  - **Host**: `TX_DEPTH` is replaced by `TX_BATCH = 8` and `TX_BUFFERS = 2`. `cb_tx_stage = TX_BUFFERS·TX_BATCH·tiles_per_packet`, derived in one place.
  - Reused: the stream / loop structure, the `FabricStreamSender` channels, and the handshakes (ready before data, done armed with flush after the last packet, resets right after each wait), all unchanged. Added: the batch loop on both halves, the poll cache, and 2 CT args.
- **Calculator prediction (before measuring)**:
  - `estimate.py` refuses L1→L1 pulls (`l1_read`) and does not model fabric sends, so the lever was ranked with the opgraph ring model by per-sender rate: `opgraph.py perf/ring_problem.json perf/ring_builder.py --opt mt_block=1 --opt cores=96 --opt antipode_split=True` with `line_rate_gbs` edited.
  - Current ≈ 4.4 GB/s → 148.7 µs (device: 131.5).
  - At the writer-alone ablation rate ≈ 11 GB/s → 74.4 µs.
  - At ≥ 20 GB/s → 55 µs (DRAM-bound).
- **Measured** (trace replay, µs per call; one fresh run per variant):

  | Variant | µs per call |
  |---|---|
  | Baseline | **131.5** |
  | Batched pull/flush + poll cache, `TX_BATCH = 8` (shipped) | **102.5** (102.4 on the first run) |
  | `TX_BATCH = 4` | 103.2 |
  | `TX_BATCH = 12` | 106.2 |
  | `TX_BATCH = 24` | 111.3 |
  | `TX_BATCH = 8`, `TX_BUFFERS = 3` | 102.7 |

  - `TX_BATCH` 4–8 tie, and depth beyond 2 does not move anything.
  - The calculator over-predicts the win (74 vs 102.5). Its ring model streams the sender at a constant rate, but the device's per-stream send is gated by when the workers publish (below).
- **Profile** (`TT_METAL_DEVICE_PROFILER=1`, eager steady-state calls, end times from program start; parsers `perf/parse_profile.py`, `perf/parse_zones.py`):
  - The last fabric_tx BRISC end went from 120.6 to 90.8 µs, with workers ending at about 65.5 µs. The **tx tail past the worker stream shrank from about 54.5 to about 25 µs**.
  - Ablations (tx reader payload stubbed: no poll, no pull, same CB sync):
    - Baseline writer alone: 120 packets in about 48 µs (0.40 µs per packet).
    - Batched writer alone: about 38 µs (0.32 µs per packet, ≈ 13.6 GB/s).
    - So the old reader chain (poll RTT + pull RTT per packet, about 1 µs) was the binding stage, and the writer's per-packet flush was the next one.
  - Zones on the shipped kernel (neighbour stream, 120 packets):
    - writer SEND 47.5 µs (about 0.4 µs per 4,352 B packet under full load), writer WAIT 45 µs;
    - reader POLL (waiting for publishes) 32 µs, pulls only 4.9 µs.
- **Bottleneck now**: the fabric send rate, about 0.32–0.4 µs per 4,352 B packet (≈ 11–13 GB/s per link-0 sender). The per-stream data arrival is shaped by the worker DRAM stream (Refinement 2's lever), and the last data published at about 60 µs still leaves about 25 µs of sends.
- **Levers measured and not kept**:
  - **Ready-order send.** The reader fills each batch with whichever heads have published, round-robin, and passes `(head, mb, tile offset)` per packet to the writer through a `cb_tx_meta` CB. It is correct (acceptance 11/11) but measured 108.8 / 108.9 / 109.1 µs. The same metadata path in strict order measured 113.2–113.9 µs, so that machinery costs about 11 µs and its default is not byte-identical to the shipped path. It was therefore not parked as a knob. The kernels are archived in `perf/experiment_ready_order/`. The cost is not root-caused; suspects are the denser poll traffic and the per-packet meta reads.
  - **tx pull on NoC 1 / tx writer on NoC 0.** NoC 0 carries the workers' DRAM reads. With this swap the output was **wrong** (PCC 0.957, about 10% of packets corrupted, mostly on + direction streams; `probes/test_probe_errmap.py`). The cause is not diagnosed. The zones then showed the pull is only about 5% of the stream time, so the lever has little headroom. Removed (broken, not parked).
  - **Bigger packets.** Not available: the fabric's packet payload is `default_packet_payload_size_bytes` = 4,352 B (`erisc_datamover_builder.hpp`), which is set by the fabric config and not by the op. `tiles_per_packet = 4` is already the full payload.
  - **Per-sender rebalance.** Not built. The link routers are already packet-balanced: link 1 carries its own 60 antipode packets plus 60 relayed. `FabricStream` binds one route per stream, so one tx core sending to two destinations would need raw fabric code or the R6 mux (Refinement 3).
- **Accuracy achieved**: PCC = 0.999907, relative RMS = 0.0177 (5 input distributions, `test_a2a_wkv_b2_precision_baseline.py`), on shape attn [1,96,160,512] per chip → out [1,24,640,128]. Unchanged.
- **Golden test progress**: 7 / 7 passing (`EVAL_CAPTURE_L1=0`), including `test_regression.py` (back-to-back bit-identity and trace replay). Acceptance 10 / 10.
- **Issues encountered**: the NoC-swap corruption above (left undiagnosed, lever removed).
- **Tests added**:
  - probe `probes/test_probe_errmap.py` (per chip / source / mt error map);
  - parsers `perf/parse_profile.py` (per-role end times, all chips) and `perf/parse_zones.py` (TX zone totals).

## Refinement 2 — Speed up the worker DRAM read stream (perf-flagged profile)
- **Date**: 2026-11-26
- **What was done**:
  - **Split-NoC reads, by core (`READ_NOC_SPLIT`, host knob).** One worker set runs its reader on NoC 1 and its writer on NoC 0. The rest keep reader NoC 0 / writer NoC 1. Shipped value: `"halves"` (logical x ≥ grid.x/2 reads on NoC 1); `"none"` is the Phase 0 layout.
    - Each kernel still uses one NoC. In dedicated-NoC mode a RISC initialises only its own NoC's barrier counters, so per-request alternation on one RISC is not safe without `DM_DYNAMIC_NOC`.
    - Reused: the reader and writer sources and args, unchanged. Added: a second reader/writer descriptor pair per NoC set.
  - **Core sets packed into rectangles (`_core_ranges`).** Worker, tx and per-NoC sets were lists of single-core ranges. That cost nothing for `none`, but a checkerboard split (48 single-core ranges per kernel) took 183 µs per call under trace replay: dispatch could not multicast.
  - **W overlaps the first attn block (worker reader).** W's reads stay in flight and are covered by the first attn block's barrier (one NoC). `cb_weight` is pushed right after that block, because compute tilizes block 0 before it needs W.
  - **Race fix 1, fabric_tx writer.** A fabric send issues the payload, then the packet header, with non-blocking NoC writes from L1. R1's batched flush re-programmed the single scatter header for packet i+1 before packet i's header had left L1. Packet i's payload then went to packet i+1's addresses: rows of head h held head h-1's data (`probes/test_probe_errmap_heads.py::test_probe_badrow_content`).
    - The writer now rotates a ring of independently armed scatter headers. The ring size comes from the per-RISC header pool (`NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType` minus the inc and unicast headers), capped at `TX_BATCH`.
    - A header is reused only after a flush: per batch, or mid-batch if the batch outruns the ring. The unicast tail header is flushed before reuse.
    - The race was latent under R1 timing and showed up as soon as the reads got faster. It is very likely the "tx pull on NoC 1: wrong output" that R1 left undiagnosed.
  - **Race fix 2, block counters.** `sem_blocks_done` was a program semaphore, which is re-initialised only when the worker core launches. Measured launch skew: fabric_tx cores start 0.25–0.6 µs in, workers up to 0.5 µs. So a tx reader could see the previous call's final count and pull unproduced blocks (mt 0/1 rows).
    - The counters are now per-head-slot `GlobalSemaphore`s (`get_block_counter_semaphores`, created on the first eager call and cached; persistent, zero at creation).
    - The worker writer bumps a counter only for a remote head.
    - The consuming tx reader waits for `>= mb + 1` and resets each head's counter to 0 (`noc_inline_dw_write` + write barrier) after the head's last block. Every call starts from 0 whatever the launch order. `op_design.md`'s synchronization contract is updated.
- **Calculator prediction (before measuring)**: `estimate.py sweep perf/sweep_attn.json` / `sweep_w.json`, `--phase "Refinement 2"`, recorded in `dm_estimates.json` (ids c92d491d / 396f7029).
  - Current (NoC 0, attn batch 32, W 64): 38.0 + 16.2 = **54.2 µs**.
  - Rank 1 (split NoC, attn batch 16): 32.3 + 13.9 = **46.2 µs**. It carries the tool's caveat that a split-NoC read on 96 cores measured 17–26% slower than estimated.
  - Batch is flat in the model (38.0 µs for batch 8/16/32).
- **Measured** (eager profile `probes/test_probe_profile.py` + `perf/parse_profile.py`: worker NCRISC end, max / mean over 8 chips; trace replay `test_a2a_wkv_b2_trace_timing.py`, µs per call):

  | Variant | Reader end, max / mean (µs) | Last tx BRISC (µs) | µs per call |
  |---|---|---|---|
  | `none` (R1 baseline + race fixes) | 60.5 / 59.8 | 92.2 | 103.1–103.6 (R1: 102.5) |
  | `checker` (x+y odd on NoC 1) | 56.2 / 54.3 | ≤ 91 | **183** (dispatch to 48 single-core ranges) |
  | `all` (every reader on NoC 1) | 107.4 / 105.4 | — | 143.4 |
  | `rows` (odd rows on NoC 1) | 58.9 / 57.1 | 86.0 | 103.1–103.5 |
  | `cols` (odd columns on NoC 1) | 57.5 / 55.1 | 87.0 | 107.0–107.2 |
  | `halves_left` (left half on NoC 1) | 53.8 / 53.5 | 86.7–87.7 | 95.6–96.3 (3 runs) |
  | `halves` (right half on NoC 1) | 56.5–58.9 / 54.6 | 84.1–86.5 | 93.5–94.4 (3 runs) |
  | **`halves` + W overlap (shipped)** | **55.8 / 54.4** | 86.0 | **93.9–94.2** |
  | shipped + `ATTN_RM_DEPTH = 3` | 55.8 / 53.9 | 84.9 | 93.9–94.9 (flat, not kept) |

  - **Calculator vs device.**
    - NoC 0: the model predicts 54.2 µs; the device measured 59.8 mean, the model being 10% fast.
    - The model's split-NoC schedule (per-request alternation, 46.2 µs) was not built as such, because it needs dynamic NoC mode.
    - The core-split equivalents measured 54.4–55.1 mean, about 18% slower than the estimate, which matches the tool's caveat.
    - The order (split faster than NoC 0) held for halves, columns and checker, but not for `all`. Running every read on NoC 1 alone is about 1.8× slower on this placement.
    - `estimate.py measure` was not recorded: the reader end is the attn and W streams together, while the estimates are per tensor.
- **Bottleneck now**: the fabric_tx send path. The last tx BRISC ends at about 84–86 µs, against about 56 µs for the reader and 59 µs for compute. Per call is 93.9 µs, of which about 8 µs is dispatch and launch skew beyond the device end. The reader is 27% above the DRAM SOL (43.8 µs).
- **Accuracy achieved**: PCC = 0.999907, relative RMS = 0.0177 (5 distributions, `test_a2a_wkv_b2_precision_baseline.py`), unchanged, on attn [1,96,160,512] per chip → out [1,24,640,128].
- **Golden test progress**: 7 / 7 (`EVAL_CAPTURE_L1=0`, including `test_regression.py`: back-to-back bit-identity and trace replay). Acceptance 10 / 10.
- **Issues encountered**: the two latent races above. Both passed every test under R1 timing and corrupted 1–3% of rows once the read stream changed timing. Under `--dev` (watcher) they did not reproduce.
- **Tests added**: `probes/test_probe_errmap_heads.py` (error map by producing head / tx stream position, and bad-row content matching).

## Refinement 3 — Remaining tail: sender topology and last-round drain (perf-flagged profile)
- **Date**: 2026-11-26
- **Headroom gate (verifier note)**: not met at the start. Per call was 94.4 µs (trace replay), against a reader end of 55.8 µs plus the 2.6 µs design drain. So the mux and the tail levers were built.
- **Diagnosis (before building)**, from the eager device profile with temporary zones (`perf/timeline.py`, `perf/pub_times.py`, `perf/parse_zones.py`):
  - **Send path.** The link-0 tx writer spent 36–42 µs in send for its 120 packets (about 0.32 µs each). Two ablations split that cost:
    - fabric send stubbed: 10.7 µs;
    - per-packet destination-address computation skipped: 27–33 µs, with the tx end 7–10 µs earlier.
    So the 4 TensorAccessor lookups per packet sat on the serial send path.
  - **Worker progress is very uneven** (`WPUB` zone per block): by core position, a worker's last block lands anywhere from ~24 to ~60 µs.
    - The strict mt-major send order gates every round of a stream on its slowest head. The link-0 sender was idle from ~30 to ~46 µs, waiting for round 1.
    - So the tail is the fabric send of late remote heads, not a last-round drain (the design's ~2.6 µs).
- **What was done** (all levers are host knobs with one source of truth in `a2a_wkv_b2_program_descriptor.py`):
  1. **Destination address table** (`cb_tx_addrs`, the tx writer). Every packet's tile NoC addresses are computed once, right after `announce_ready` (about 9 µs on the link-0 stream, while the destination starts and the first blocks are produced). The send loop only indexes the table.
     - Measured 94.4 → **89.3 µs** per call.
     - A lazy variant (fill while waiting for batches, catch up inline) measured 86 µs plus one 193 µs outlier run, so it was not kept.
  2. **Ready-order send** (`TX_ORDER = "ready"`; `"strict"` keeps the old order and passes the acceptance tests, measured 93.3 µs with the R3 placement).
     - The tx reader fills each batch with whatever packets are published, round-robin over the stream's heads. It sends a partial batch when a poll finds nothing more ready.
     - Each batch carries a `cb_tx_meta` page `[k, code…]`, code = (mt block, head, packet); the writer indexes the address table with it.
     - R1's ready-order attempt (108.8 µs, archived in `perf/experiment_ready_order/`) paid for per-packet address computation in the writer and waited for full batches; neither applies here.
     - Measured 89.3 → **84.2 µs**.
  3. **Head placement** (`HEAD_PLACEMENT = "by_stream"`; `"interleaved"` and `"rowmajor"` also available). Per chip, the local heads (no fabric send) go on the first core rows, which publish slowest. The remote destinations follow by ascending per-stream head count, so the neighbour chunks sit on the fastest rows. A core's heads stay consecutive (row-major fallback otherwise).
     - Measured 84.2 → **82.1–82.6 µs** (4 runs; `interleaved` 82.7–82.8).
  4. **R6 mux, built and parked** (`MUX_OVERFLOW_HEADS`, default 0 = the previous topology; `MUX_NUM_BUFFERS = 8`).
     - The last x heads of each neighbour stream become a second stream on link 1. Link 1 is shared with the antipode stream through one fabric mux core per direction.
     - Host side: `ttnn.experimental.fabric_mux` Config / client args / mux kernel `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp`, fixed-address map on mux cores outside every CB range. The two mux cores share one kernel descriptor.
     - Kernel side: the tx writer only swaps `DirectConn` for `MuxConn<N>` under `A2A_TX_MUX`.
     - Peers are matched by (direction, link, slot).
     - Correct on the first run (acceptance 10/10 at x = 4 and 6).
     - Trace per call: x = 0 / 2 / 4 / 6 / 8 → 82.7 / 84.8 / 83.6 / 83.1 / 84.2 µs (separate mux kernels), then 82.0 µs at x = 6 with the mux kernels merged.
     - Trace-replay device profile (`probes/probe_trace_profile.py` + `perf/trace_gaps.py`): device span per replay 71 → **63.5 µs** at x = 6, but the idle gap between back-to-back programs grows 11.5 → **17.5 µs**. The device win is spent on launch. Mux opt level O2 / Os / O3 did not change the gap.
     - Parked at 0 because per call does not improve.
- **Calculator prediction (before measuring)**: `/perf-calculator` loaded. No tool models the measured costs:
  - `estimate.py` refuses fabric and L1-to-L1 schedules;
  - `opgraph.py`'s ring builder has one sender per link and prices links at a rate, not per packet;
  - none model the launch gap.
  The predictions therefore came from the ablations: address table −7…−10 µs on the tx end (measured −5.1 µs per call); ready order, link 0 busy from first publish (~12 µs), expected tx end ≈ 60 µs (measured: busy from 23 µs at 0.36 µs per packet, end 65–75 µs, −5.1 µs per call); mux at x = 6, link 0 −30 packets ≈ −7 µs of device span (measured −7.5 µs of device span, but +6 µs launch gap). No `dm_estimates.json` record: no estimate exists for these schedules.
- **Measured summary** (trace replay, µs per call): R2 94.4 → address table 89.3 → ready order 84.2 → placement **82.1–82.3** (final, 3 runs). Eager device profile, last tx BRISC: 80–89 → 70–75 µs. Worker reader end unchanged (~53–57 µs).
- **Bottleneck now**:
  - **Device span, ~71 µs.** The link-0 senders are busy from ~23 µs to the end at ~0.36 µs per 4,352 B packet (~12 GB/s). They are slower while the DRAM read stream is on (0.22 µs per packet without it): NoC contention on the source and landing sides.
  - **The link-1 router.** It carries 120 packets (60 own plus 60 relayed) and still finishes about 10 µs earlier. So a single worker channel, not the Ethernet link, limits link 0, which is why the mux moves the device span.
  - **Launch gap, 11.5 µs per program** in back-to-back trace replay (14% of per call), and 17.5 µs with the mux.
- **Next** (not done here): find what the launch gap is made of (per fabric connection? the mux's router connection?) before turning the mux on. With that gap cut, `MUX_OVERFLOW_HEADS = 6` is worth about −7 µs. Also: an earlier first publish for the neighbour heads (the link-0 send starts at ~23 µs).
- **Accuracy achieved**: PCC = 0.999907, relative RMS = 0.0177 (5 distributions, `test_a2a_wkv_b2_precision_baseline.py`), on attn [1,96,160,512] per chip → out [1,24,640,128]. Unchanged.
- **Golden test progress**: 7 / 7 (`EVAL_CAPTURE_L1=0`, including `test_regression.py`). Acceptance 10 / 10.
- **Issues encountered**: running the whole unit directory gives one setup ERROR, "Resource temporarily unavailable" in mesh discovery. It moves to a different test each run, and the R2 commit hits it too (27 passed, 1 error), so it is a node process/thread-limit flake, not this change. The new trace-profile probe is named `probes/probe_trace_profile.py` so the directory run does not collect it (one less mesh open).
- **Tests added**:
  - `probes/probe_trace_profile.py`, run explicitly: device profile of 6 back-to-back trace replays;
  - parsers `perf/trace_gaps.py` (per-replay span and launch gap), `perf/timeline.py` (tx zone timeline) and `perf/pub_times.py` (worker publish times per block and core). They need temporary `DeviceZoneScopedN` zones: `TX_*` in the tx kernels, `WPUB` in the worker writer.

## Perf 1 — perf tournament round 1 (perf-flagged focus case)
- **Date**: 2026-11-26
- **Focus case** (the only cell, LOOSE_CASES perf flag): per chip attn [1,96,160,512] bf16 ROW_MAJOR DRAM, W [1,96,512,128] bf8_b TILE DRAM → out [1,24,640,128] bf8_b L1; HiFi2, fp32_dest_acc_en=False, 2x4 LoudBox (FABRIC_2D_TORUS_X). Every case in INPUTS is in SUPPORTED, so there is no generality gap. SOL 43.8 µs; roofline (op_design Bounds) ≈ 49.5 µs.
- **Result**: trace replay, `test_a2a_wkv_b2_trace_timing.py`, µs per call:
  - **82.1 → 72.8 µs (−11.3%)**: 4 ideas WIN in isolation, 3 graduated, 1 NULL, 1 win that regressed once combined.
  - Golden 7/7 after every step. Acceptance 10/10, also under `--dev` (watcher, NoC sanitizer).

### Instrumentation (permanent)
- `MaybeDeviceZoneScope` zones were added in every kernel, from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (the header was brought onto this branch). None sits in a per-tile or per-packet loop: at most 20 markers per RISC, against a cap of 250.
  - worker reader: `reader_w_issue`, `reader_attn_block` (per tile row);
  - compute: `compute_tilize`, `compute_matmul`;
  - worker writer: `writer_wait_block`, `writer_local_write`;
  - tx reader: `tx_reader_stream`, `tx_reader_reset`;
  - tx writer: `tx_writer_addr_table`, `tx_writer_await_ready`, `tx_writer_stream`, `tx_writer_close`, `tx_writer_await_done`.
- **Ablation hooks:** the env `A2A_ABLATE` = comma list of NO_READ, NO_COMPUTE, NO_PULL, NO_SEND sets `A2A_ABL_*` defines. Each stubs a payload and keeps every CB, barrier and handshake. Not for production.
- **Parsers:**
  - `perf/stage_zones.py`: per-zone occupancy and end time for one program instance of an eager or trace-replay profile, plus a marker-budget check;
  - `perf/trace_gaps.py`: per-replay span and launch gap.

### Measured breakdown (before)
Trace replay with `probes/probe_trace_profile.py`, per-replay device span; launch gap = last kernel end → next replay's first kernel start.

| Part (cumulative peel) | Measured µs | Part SOL µs | Part roofline µs | Gap |
|---|---|---|---|---|
| Full op | span 71, gap 11.5, per call 82.1 | 43.8 | 49.5 | — |
| − fabric sends | 60 | | | |
| − sends − compute = **worker DRAM reads alone** | **52.4** | 43.8 (DRAM, 512 GB/s) | 46.2 (estimate.py rank 1 = current schedule) to 49.5 | (52.4−43.8)/49.5 = 17% |
| − all payloads = **fixed** (handshakes, address table) | **18.3** | — | ≈ 3.1 (launch + open + entry handshake) | 490%. Address table: 9.4 µs serial on the tx BRISC |
| reads + compute stubbed = **tx path alone** | **49.7** (start-up ~10.6, 120 link-0 packets ~37.7 = 0.31 µs/pkt ≈ 14 GB/s, done ~1.6) | 10.4 (fabric, 50 GB/s/ch) | 13.5 | 290% |
| compute (matmul zone) | 8.9 (hidden) | 7.6 | — | hidden under the reads |
| **launch gap** (trace-replay boundary) | **11.5**, identical in every ablation | — | ≈ 1.4 | 10 µs |
| Overlap loss | 71 − 52.4 = 18.6 | | | |

- **Ranked bottlenecks:**
  1. the tx send path's per-packet rate;
  2. the fixed start-up;
  3. the launch gap;
  4. the worker read stream, 17% over its floor.
- **Data-movement search (`estimate.py sweep`, `--phase "Perf 1"`):**
  - attn sweep id ebb7ef37, W sweep id ceec378f;
  - the current schedule (split NoC, attn batch 32, W batch 64) is tied for rank 1, 32.3 + 13.9 = 46.2 µs;
  - so no schedule idea was fanned out, and the read lever had to be at the implementation level.

### Portfolio and verdicts
Each idea was run by its own `perf-part-optimizer` in `perf_experiments/<slug>/`. The isolation numbers were measured against the op as it stood after the breakdown (82.1–82.6 µs per call).

| Idea | Stage | Verdict | Isolation (µs per call) | Domain |
|---|---|---|---|---|
| `launch_gap`: decompose the 11.5 µs trace-replay gap | dispatch / program structure | **WIN** | 82.2 → 77.9, gap 11.56 → 7.67 | everywhere |
| `reader_issue`: read issue path, order, per-bank NoC | worker reader | **WIN** for the op, NULL for the read itself | 82.4 → 78.6; reader alone 52.4 → 52.4 | this board's placement; correct on any grid |
| `tx_send_rate`: per-packet cost of the worker → router channel | tx writer | **WIN** | 82.6 → 79.8; tx loop alone 35.8 → 28.1 | everywhere |
| `tx_rebalance_direct`: neighbour overflow heads on link 1 over the existing direct connection, no mux | tx topology | **WIN in isolation, REGRESSION combined** | x = 3: 82.3 → 79.0; combined: x = 0 73.0 / 72.7 / 73.2 vs x = 3 75.3 / 74.7 / 75.0 | — |
| `tx_startup`: address table off the critical start-up, cheaper generation | tx writer / tx reader start | **NULL** | 82.5 → 82.4; all-stubbed span 18.3 → 12.3; tx alone 50.1 → 44.3 | — |

**`launch_gap`.**
- The 11.5 µs is a cost of the **trace-replay boundary**, not of each call. Within one trace holding 4 calls, the gap between calls is 0.45 µs. At each replay start the dispatcher waits for the workers, resets the launch ring, then writes the first program's config.
- So the one-call-per-trace metric charges the op about 11 µs that a multi-op model trace mostly hides. With 4 calls per trace: 72.8 → 71.7 µs per call.
- Decomposition (empty-kernel stub with the same structure, gap in µs):

  | Change | Gap µs | Effect |
  |---|---|---|
  | Full structure | 11.53 | binary size does not matter |
  | Worker per-core runtime args removed | 8.51 | −3.0 |
  | Worker per-core runtime args as a common table | 9.5 | −2.0 |
  | Fabric-connection args removed (with their 8 single-core semaphore descriptors) | 9.60 | −1.9 |
  | Those semaphores merged into 2 rectangles | 9.61 | −1.9 |
  | NoC-split kernel pairs merged | 10.6 | −0.9 |
  | CBs removed | 10.5 | −1.0 |
  | Compute kernel removed | 11.33 | −0.2 |

- Cost scales with the number of dispatch destinations, about 0.25–0.45 µs each, not with bytes. The co-located-config variant measured −0.85 µs, inside the noise, so it was not kept.

**`reader_issue`.**
- The read stream is at a DRAM ceiling of about 430 GB/s: reader alone stays at 51.6–53 µs whatever the request size, the in-flight depth, trid pipelining or NoC relief.
- R2's "halves" split leaves the NoC 1 northbound links of the two DRAM columns saturated, according to a link-load model (`ri_linkmodel.py`) that matched measurement within ~10%.
- A searched per-core, per-bank NoC mask keeps the read time but frees those links for the fabric_tx pulls and sends. Full-run effects:
  - the tx ready wait ends at 15.5 µs instead of 25.7;
  - `tx_writer_stream` ends at 62 µs instead of 73;
  - the slowest worker read lands at 51 µs instead of 59.
- Searched masks s1 / u4 / u6 measured 78.2–79.4 µs, so the result does not depend on one seed.
- Measured regressions (masks not to use): uniform masks, "geo", parity, and a throttled local-head window (92.9).

**`tx_send_rate`.**
- Per-step cycle timers on the link-0 sender, in cycles per packet:

  | Step | Cycles per packet |
  |---|---|
  | Header fill (`populate_unicast_scatter_write_fields` re-encodes each chunk address) | 208 |
  | Waiting for an EDM slot | 27 (the 17-slot channel never filled) |
  | Payload write issue | 47 |
  | Header write + credit | 79 |
  | Batch flush | 2 |

- Pre-encoding the addresses into the table, filled once, gives 79.8 µs per call. A raw mirror of the helper's steps gives 81.0, so ~1.6 µs of the win is helper overhead.
- With the header cost gone, the router channel binds at about 0.17–0.21 µs per packet.
- Measured not to help: per-batch flush removal (2 cycles), deeper buffering, header co-located with the payload (`pullhdr` 81.4, `colo` no better), and swapping the tx NoCs (REGRESSION, 82.9–87.8).

**`tx_rebalance_direct`.**
- **Why it helped in isolation.** The link-1 router also relays the 60 packets of the other side's antipode stream, at ~0.245 µs per packet, so the two links balance at x ≈ 3, not 6. In isolation (link-0 tail-bound op):

  | x | 0 | 1 | 2 | 3 | 4 | 6 | 8 |
  |---|---|---|---|---|---|---|---|
  | µs per call | 83.4 | 81.0 | 79.6 | **79.0** | 80.9 | 84.3 | 88.1 |

- **Why it did not graduate.** Built on top of the three graduated changes, with the multi-route writer merged with the pre-encoded headers (`perf_experiments/tx_rebalance_direct/kernels/a2a_wkv_b2_tx_writer_mr_prehdr_combined.cpp`):

  | x | µs per call |
  |---|---|
  | 0 | 72.6–74.1 (alternating runs 73.0 / 72.7 / 73.2) |
  | 1 | 74.5 |
  | 2 | 74.6 |
  | 3 | 75.3 / 74.7 / 75.0 |

  Once the reads stopped crowding NoC 1 and the send rate improved, link 0 no longer sets the tail, and link 1 plus the 2-route handshake add ~2 µs. That is a measured regression, so it was not graduated.
- **Helper gap it exposed** (no graduated bypass): `FabricStream` binds one route at `open()`, so one connection cannot serve two destinations. A fix would add route-taking overloads of `arm_*` (like the multicast `arm_inc(route, val)`).

**`tx_startup`.**
- Recommended variant: the tx reader fills half the table at its idle start, the writer the other half, with incremental L1-bank arithmetic instead of an accessor call per tile.
- In the full op the link-0 stream starts ~8 µs earlier but ends at the same time, because it is publish- and send-bound.
- Sending before the ready wait is not allowed by the synchronization contract: `sem_ready` is the only proof that the destination's previous call has ended.

### What graduated (each one the op's only path; the code it replaced is deleted)

| Step | Commit | µs per call (trace, runs) | Change |
|---|---|---|---|
| 0 | — | 82.1 | baseline with the zones (zones cost nothing: same as R3's 82.1–82.3) |
| 1 | `launch_gap` | **78.1 / 77.9** | worker per-core runtime args removed → one common-arg table per kernel, one word per logical core (`head_start \| num_heads << 8`), indexed with `get_relative_logical_x/y`. `_merge_semaphores` re-packs the fabric connection's single-core semaphore descriptors into rectangles. |
| 2 | `reader_issue` | **74.6 / 75.0 / 74.6 / 74.3 / 74.1** | worker reader issues each DRAM page on the NoC chosen by its per-bank mask (`NOC1_BANK_MASKS`, packed into the reader's table word bits 16–31), with transaction-id barriers on both NoCs. Worker reader and writer run in `DM_DYNAMIC_NOC` mode (see below). |
| 3 | `tx_send_rate` | **72.9 / 72.7 / 72.9** | tx writer: the address table holds addresses pre-encoded for the receiving router, headers armed once per ring slot, per packet copy 8 words + count + size; connection still the helper's `DirectConn` / `MuxConn`. |

- **Carve-out (one, feasibility).** `NOC1_BANK_MASKS` is meaningful only on the layout it was searched for: a 12-wide worker grid of 8 rows on this Blackhole p150b placement. On any other grid `_noc1_bank_mask` falls back to the "halves" equivalent (0xFFFF on the NoC 1 half, else 0). Any mask is correct on any grid: it only picks a NoC per page. Written as `if layout matches: table else: halves`. The table has meaning only on that layout, so this is a feasibility carve-out, not an allow-list of measured shapes.
- **Correctness finding during graduation.** The `reader_issue` kernel as delivered (dedicated NoC mode) passed the normal acceptance test but **hung under `--dev`**:
  - The worker BRISC's end-of-kernel NoC-idle ASSERT tripped. The NCRISC reads on BRISC's NoC advance that NoC's shared hardware counter, which BRISC's own software count does not include.
  - Release builds compile the check out, and every kernel re-syncs its counters at launch.
  - Fixed by running the worker reader and writer in `DM_DYNAMIC_NOC` mode, the supported mode for two RISCs sharing a NoC. Each kernel's default NoC is unchanged. Per call is unchanged (74.1–74.6), and `--dev` acceptance passes 10/10.
- **Whole op:** 82.1 → **72.8 µs per call** (−9.3 µs, −11.3%), with 4 calls per trace not re-measured after graduation.

### After (trace-replay profile of the graduated op)

| Part | Before µs | After µs |
|---|---|---|
| Device span per replay | ~71 | 61–72 (median ~65; varies more between replays) |
| Launch gap | 11.5 | **7.5** |
| Reader alone | 52.4 | 51.5 |
| Tx path alone | 49.7 | 43.8–46.5 |
| Fixed (all stubbed) | 18.3 | 18.9 |

- Full-run zones:
  - the last worker read lands at ~53 µs and compute ends at ~56.5;
  - `tx_writer_stream` ends at ~60;
  - `tx_writer_await_done` waits up to 19 µs (ends at ~69) for the **peer chips'** data.
- **Bottleneck now (round 2):**
  - cross-chip skew in the done handshake, where the slowest chip's last sends set every chip's end;
  - the fixed start-up (18.9 µs all-stubbed; `tx_startup`'s split fill cut it to 12.3 µs and becomes worth re-measuring now that the stream end is less send-bound);
  - the read stream at ~430 GB/s against 512.

### Guard set
- The op has one cell (layout × placement × kernel path), and it is the focus case.
- The per-step whole-op measurement above is the no-regression check: every step was flat or faster.
- R3's parked knobs: the `tx_send_rate` agent ran `TX_ORDER="strict"` (94.3 → 89.7) and the mux (`MUX_OVERFLOW_HEADS=6`, 86.0 → 85.2, flat) with the new tx writer. Both are correct.
- Golden 7/7 (`EVAL_CAPTURE_L1=0`) after each graduation. Acceptance 10/10, twice, plus `--dev` 10/10 on the final op.

### Data-movement estimates (`estimate.py report`)
- Perf 1 sweeps ebb7ef37 (attn) and ceec378f (W): 12 candidates each, 6 shortlisted, 0 built. The current schedule is rank 1–2 (tied), so nothing was fanned out from them.
- No `estimate.py measure` was recorded. The estimates are per tensor, but the reader-alone measurement (52.4 µs before, 51.5 after) is the attn and W streams together.
- Measured vs estimate: 52.4 µs vs 46.2 (the attn + W rank-1 sum), 13% slow. This matches the tool's split-NoC caveat; `reader_issue` found the stream DRAM-bound at ~430 GB/s.
- Fabric and L1-to-L1 schedules are refused by the tool, so the tx ideas went out unranked.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `dataflow_kernel_lib::read_sticks_for_tilize` | capability | issues every stick on the kernel's one NoC and ends each tile row with `noc_async_read_barrier()`, which counts only this RISC's own-NoC reads. There is no way to spread one core's reads over both NoCs per page, and no transaction-id barrier option. (On one NoC the raw path ties the helper: 52.4 vs 52.3 µs reader alone.) | 82400 (op per call, helper, halves) | 78600 (op per call, per-bank dual NoC) | `kernels/a2a_wkv_b2_reader.cpp:111` (attn), `:96` (W) |
| `dataflow_kernel_lib::ccl::FabricStream::arm_scatter_write` / `ScatterWriteChannel::write_scatter` (+ `FabricStreamSender::open`, `arm_inc`, `arm_unicast_write`, `flush`, `close`; the connection stays `DirectConn` / `MuxConn`) | capability | `write_scatter` runs `populate_unicast_scatter_write_fields` on every packet, which re-encodes each chunk address for the EDM-local NoC and re-stores the chunk sizes (208 of ~370 cycles per packet). There is no entry point for addresses encoded once ahead of time. Suggested: `write_scatter_encoded(const uint64_t* encoded, chunks, src)` plus an `encode_dst(noc_addr)`. | 82600 (op per call, helper) | 79800 (op per call, pre-encoded) | `kernels/a2a_wkv_b2_tx_writer.cpp:283` (send loop), `:240` (`send_scatter`) |
| same `FabricStream` layers | ergonomics | a raw call sequence identical to the helper's (no pre-encoding) measured ~25 cycles per packet faster. Likely the command-header copies through the helper's channel layers and the pointer hop to the sender; not confirmed. Reported only as a measured overhead: it is not why the bypass was taken (the capability row is). | 82600 | 81000 | `kernels/a2a_wkv_b2_tx_writer.cpp:122` |

- **Not graduated, so not in the table.** `tx_rebalance_direct` needs two routes on one connection, which `FabricStream` cannot express (it binds one route at `open()`). `tx_startup`'s address walk needs a step iterator over interleaved pages that `TensorAccessor` lacks (17.9 vs 12.9 ns per entry).

### Tests and artifacts
- `perf_experiments/{launch_gap,reader_issue,tx_send_rate,tx_rebalance_direct,tx_startup}/` hold each idea's variant op, kernels, tests, probes and parsers. Profile CSVs were deleted.
- The experiment tests need `-o addopts= --import-mode=prepend`. Their `import torch` lines are function-local, as the ttnn torch-import hook requires.
