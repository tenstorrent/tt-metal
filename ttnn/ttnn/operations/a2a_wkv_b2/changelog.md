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
