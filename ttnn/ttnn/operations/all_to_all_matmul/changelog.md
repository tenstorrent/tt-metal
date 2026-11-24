# Changelog: all_to_all_matmul

## Phase 0 — Core Implementation
- **Date**: 2026-11-24
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). Regime R1: heads are split across the compute grid (grid minus every chip's 2·num_links port cores). Remote row quarters go over Fabric 2D straight into the consuming core's L1 (zero-copy landing scratch, one arrival counter per head slot, per-port ready fence). Weights are read from DRAM while the rows are in flight. `matmul_block` per head with A retained across n-blocks. Transport T1: the far quarter goes direct on the routing table's 2-hop route. One `generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], weight_dtype=[bfloat16], layout=[TILE], fp32_dest_acc_en=[True], alignment=[tile_aligned], output_buffer=[DRAM], memory_layout=[INTERLEAVED], cluster_axis=[1], topology=[Ring], num_links=[2]
- **Accuracy achieved**: PCC=0.9999986, max_abs_err=0.0163, mean_abs_err=0.00117, rel_rms_err=0.00169, ULP p99 ≤ 4 (measured on 4 shapes via test_all_to_all_matmul_precision_baseline.py). The exchange is bit-exact.
- **Golden suite at Phase 0**: 16 / 16 supported cells passing; 132 xfail_expected; 0 supported_fail / xpass_drift / xfail_wrong_mode (per `verifier_report.json`). Regression suite 7 / 8 pass. `test_small_packet_size` cannot open a 4160 B fabric on this board (ERISC kernel config buffer overflow in `open_mesh_device`); this is an environment issue.
- **Perf (Phase 0 loose case, Kimi, 7232 B, harness)**: fastest chip 94.4 us, slowest 104.2 us, not dispatch-bound (roofline 63.0, target 84.0). Exchange + fence alone 49.3 us.
- **Issues encountered (verifier fixes)**:
  - `cb_weight` was allocated `WEIGHT_DEPTH` blocks on cores that process one block per call. It is now `min(WEIGHT_DEPTH, heads_per_core_max * num_n_blocks)`. Perf is neutral (within noise); L1 peak fell 368,672 → 270,368 B (Kimi) and 458,752 → 262,144 B (GLM).
  - `validate()` raised two caller errors (weight layout, input buffer) after the support refusal; they now run before it.
  - The acceptance-test modules each imported the golden mesh fixtures, which created a second session cache when the directory ran. The fixtures now come from the directory `conftest.py`.
  - Stale NoC note in the program-descriptor docstring corrected.
- **Tests added**: test_all_to_all_matmul.py (planner), test_all_to_all_matmul_precision_baseline.py (verifier)

## Refinement 1 — Model configuration: 16-bit DEST, bfloat8_b weight, L1 output
- Date: 2026-11-24
- What was done: `SUPPORTED["fp32_dest_acc_en"]` += False, `SUPPORTED["weight_dtype"]` += bfloat8_b, `SUPPORTED["output_buffer"]` += L1. No kernel change was needed: `cb_weight`'s format follows `weight.dtype` (Bfp8_b pages, 1088 B), `block_n` derives from the byte knob `WEIGHT_BLOCK_BYTES`, the host subblock rule picks the 8-tile limit under 16-bit DEST, `matmul_block` reconfigures the unpacker between the bf16 A and bfp8 weight, and the writer's `TensorAccessor` follows the L1-interleaved output. Added a live knob `COMPUTE_WRITER_NOC_L1_OUT` (writer NoC for an L1 output), parked at NoC0 = the prior choice: NoC1 measured fastest 81.01 -> 78.53 us but slowest 84.92 -> 86.15 us on the Kimi model case, rejected by `eval.ccl_perf compare` (slowest regressed beyond noise).
- Accuracy achieved (precision baseline, 4 shapes small / medium / Kimi / deep K): fp32 DEST HiFi4 (bf16 or bf8b weight, reference on the quantized weight): PCC 0.9999986, rel RMS 0.00169. 16-bit DEST HiFi2 L1 out: PCC 0.99998, rel RMS 0.0051-0.0062 (K <= 128); deep K 4096: PCC 0.9990, rel RMS 0.070 (gate 0.13) with a median got/true ratio of 1.058 — `ttnn.matmul` with 16-bit DEST on the same data gives 1.0584 / 0.0701 (probes/probe_013.py), so this is the hardware's 16-bit accumulation over 128 K-tiles, not an op bug.
- Golden test progress: tile_aligned 120/120 pass (every DEST x weight dtype x output buffer cell), LOOSE_CASES 4/4 pass; the 24 m_non_aligned cells remain xfail (Refinement 2). 124 supported pass / 24 xfail_expected of 148.
- Perf (harness, 5 experiments, CCL_PERF_CASES=all, none DISPATCH-BOUND): Kimi model (L1 out, HiFi2, 16-bit DEST) fastest 81.01 / slowest 84.92 us; Kimi Phase 0 94.43 / 104.30 us (unchanged); DeepSeek model 99.12 / 104.94 us; GLM model 79.10 / 83.86 us.
- Issues encountered: the deep-K 16-bit-DEST scale drift above (test's ratio check scoped to fp32 DEST or K <= 512, with the probe as evidence). None otherwise.
- Tests added: `test_all_to_all_matmul_precision_baseline.py` gains a CONFIGS axis (phase0, model 16-bit DEST HiFi2 L1, 16-bit DEST HiFi4 L1, bf8b weight fp32 DEST, bf8b weight 16-bit DEST HiFi2 L1) — 20 cases; `probes/probe_013.py` (ttnn.matmul 16-bit DEST deep-K reference).

## Refinement 2 — Non-tile-aligned rows per device (regime R1')
- Date: 2026-11-24
- What was done: `SUPPORTED["alignment"]` += `m_non_aligned` (regime R1'). When `Mq = S/G` is not a multiple of 32, destination t's quarter starts at row `shift = (t*Mq) % 32` of source tile row `t0 = (t*Mq) / 32`. The port reader (remote quarters) and the compute-core reader (local quarter) re-tile it through a new shared header `kernels/all_to_all_matmul_retile.hpp`: per image, one batched DRAM read of the quarter's `nsrc <= Mq_t + 1` source tile rows x Kt into `cb_retile_stage` (one barrier), then per destination tile core-local NoC copies of contiguous face-row runs (<= 6 per tile) plus a zero fill of the padding rows from `MEM_ZEROS_BASE`. The port reader keeps its single barrier per packet batch before the push. Landing, arrival counters, fence, sender, compute and writer are unchanged (`Mq_t = ceil(Mq/32)`); the output's padded rows are written as computed (zeros). The aligned path is byte-identical in behavior (compile-time `retile = Mq % 32 != 0`; the base tile row is `t*Mq/32`, which equals `t*Mq_t` when aligned) and allocates no staging CB.
  - Reused: both readers' entry / tile cursors and batching, all CBs, landing and transport. Added: the retile header, `cb_retile_stage` (`(Mq_t+1)*Kt` tiles, R1' only), CT args `Mq`, `CB_RETILE_STAGE`.
- Accuracy achieved: exchanged rows bit-exact (identity-weight closed-form provenance test, 40 rows/device and B=2 x 12 rows/device); the four R1' shapes ((1,24,160,128), (1,8,96,128), (1,8,48,64), (2,4,200,64)) pass the golden bf16 tolerance (PCC >= 0.99 / rel RMS <= limit from helpers.TOLERANCES) under fp32 DEST HiFi4.
- Golden test progress: m_non_aligned 24/24 pass (every DEST x weight dtype x output buffer cell); + 8 tile-aligned Kimi-2048 representatives pass in the same slice; acceptance suite 10/10. Expected whole suite: 148/148 supported.
- Issues encountered: the first build gave stale / foreign rows on 2 heads per chip, deterministic. DEVICE_PRINT of the L1 addresses showed `cb_retile_stage` placed inside `cb_port_chunks` on the port cores of chips 1 and 3 when it was declared with one CBDescriptor over the (non-rectangular) union of compute and port cores. Fixed with one descriptor per core set. Worth knowing for any future CB shared by both core sets.
- Tests added: `test_all_to_all_matmul_non_aligned.py` (4 shapes vs reference incl. a multi-batch quarter spanning 3 source tile rows; 2 bit-exact row-provenance cases).

## Refinement 3 — OVERLAP: hide the local work behind the exchange on the perf-flagged case
- Date: 2026-11-24
- What was done: two OVERLAP levers, each a single-source host knob.
  - **Weight-read throttle** `WEIGHT_READ_CHUNK_BYTES = 16 KiB` → CT arg `weight_chunk_tiles`: the compute reader issues one read barrier per chunk instead of one per 128 KiB block. Diagnosis: exchange + weight reads alone (`nowrite,nocompute`) took 68.9 us vs 49.3 us for the exchange alone. `noportread` stayed at 77 us, so this is NoC contention with the send / landing path, not DRAM contention with the port readers; the weight reads on NoC0 were far worse (93.5 us). Measured chunk sweep (Kimi model case, fastest chip): whole block 80.8, 64 KiB 76.4, 32 KiB 64.1, 16 KiB 64.2, 8 KiB 64.3, 2 KiB 63.6 us. Weight-after-landing ("wlate") measured 65.9 us.
  - **Tail split** `TAIL_SPLIT_MAX_HEADS = 8` (0 = off). Per chip, the remote heads that land last (estimated from their entry's position in the source port's send list) are halved over N. The primary keeps n-blocks `[0, nb/2)`, and a helper (idle compute cores first, then cores whose heads are all local on that chip, fewest heads first) takes `[nb/2, nb)` in an extra slot after its own. After landing, the primary's reader pushes the slot to its compute, then NoC-writes the rows into the helper's landing slot and increments that slot's arrival counter. When `num_n_blocks == 1`, `block_n` halves for every head, so all perf cases now run 2 n-blocks per whole head. All compute cores run the kernels when the split is on (idle cores with no slot do nothing). Split-count sweep: 0 / 4 / 8 / 16 / all → Kimi model 64.1 / – / 60.5 / 60.3 / 60.5; Phase 0 91.0 / 87.2 / 87.6 / 92.0 / 91.5; DeepSeek 81.0 / – / 77.0 / 77.2 / 77.3; GLM 65.0 / – / 60.1 / 60.2 / 62.2 us.
  - Reused: the remote-slot wait path (helper slots are remote-style slots with an arrival counter), the persistent landing scratch and global arrival semaphores (cap = own + helper slots), the multi-n-block path of all three kernels (now a per-slot runtime `[nb0, nb1)`). Added: per-slot runtime words (reader 7, compute 1, writer 2), the forward step in the reader, the host tail plan.
  - Rejected (measured, not kept): writer on NoC1 for L1 output, 66.0 vs 64.1 us (after the throttle).
- Perf (harness `perf_measure.py`, 5 experiments, `CCL_PERF_CASES=all`, fastest / slowest chip; `eval.ccl_perf compare` verdict win on every case vs both the baseline and the throttle-only step; none dispatch-bound):
  - Kimi model (focus, LOOSE_CASES[0]) 81.20 / 84.56 → **60.53 / 64.64 us** (roofline 32.3, target 43.0)
  - Kimi Phase 0 94.53 / 103.52 → 87.66 / 91.72 us
  - DeepSeek 99.54 / 104.29 → 77.13 / 83.32 us
  - GLM 79.12 / 83.78 → 59.88 / 64.74 us
  - Breakdown, focus case, before → after (fastest chip): full 80.8 → 60.5; exchange alone (`nowrite,noweight,nocompute`) 49.3 → 50.5; `noweight` 62.9 → 59.6; `nocompute` 69.0 → 55.4; `nowrite` 76.7 → 56.6. Exposed local time per call: 31.5 → ~10 us.
  - Remaining headroom (finding): the compute tail (~5 us) and the L1-output writes (~4 us) after the last landing, and above all the exchange itself (50 us vs the 20.3 us links roofline; Refinement 4). Untried next: a writer in-flight bound (the writer twin of the throttle), a finer split of the very last landings, per-row-tile arrival increments.
- Accuracy achieved: unchanged math (same matmul_block per n-block; the split only partitions output columns). Unit directory 36/36, including the precision baseline (bf16 fp32 DEST HiFi4: PCC 0.9999986, rel RMS 0.00169) and the bit-exact / provenance / back-to-back / determinism tests.
- Golden test progress: targeted slice 68/68 (N=4096 and N=8192 multi-n-block, Nt=1 (no split possible), m_non_aligned (1,24,160,128) and (1,8,48,64), B=2, deep K 4096, GLM-256 shape, plus all 4 LOOSE_CASES), across every dtype / DEST / output-buffer cell of those shapes.
- Issues encountered: none (no hang, no mismatch). One measurement artefact: piping `tt-probe.sh` into `head` killed the probe midway (SIGPIPE). Send probe output to a file.
- Tests added: none (behaviour is covered by the existing suites). Probe `probes/r3_variants.py` (A2AMM_ABLATE variants on any LOOSE case via the eval harness) and saved perf JSONs under `perf/r3/`.

## Refinement 4 — TRANSPORT: shrink the exchange itself on the perf-flagged case
- Date: 2026-11-24
- What was done: three port-side TRANSPORT levers, each a single-source host knob (plus a path gate), all in the port reader / sender and the host send plan.
  - **Finding first:** the routing table's far-quarter routes on this board already alternate (chip 0 BWD, 1 FWD, 2 BWD, 3 FWD: 2D NS-first routing on the 2x2 snake), so every directed link carries 2 quarters (balanced). T2 relaying would buy nothing here. The exchange is bound by the router's per-packet cost and the fence, not by link balance.
  - **Destination runs** (`SEND_GROUPED_BY_DEST`, `SEND_RUN_ENTRIES` = 3, `SEND_DEST_ORDER` = "near"): a port sends runs of 3 consecutive entries per destination chip, the destinations' runs alternating, nearest destination first (was: one entry per destination alternating; the old sort key also put the far destination first on BWD ports). Exchange alone: all-grouped near-first 47.3, far-first 54.4, legacy mix 52.6 us.
  - **Per-destination ready fence** (`FENCE_PER_DEST`): one ready counter per destination chip of a port (`res.ready` is now a list; readies carry the counter address); the sender waits for a destination's ready only before its first run, so the 1-hop run starts after a 1-hop round trip and the 2-hop ready is awaited behind it. Exchange alone (grouped): 53.7 → 52.6 us.
  - **Packing across entries** (`PACK_ACROSS_ENTRIES`, new shared `kernels/all_to_all_matmul_stream.hpp`): within a destination run the tiles are one stream of full packets; a packet straddling two entries is a 2-chunk `to_noc_fused_unicast_scatter_write_atomic_inc` (end of entry e + start of entry e+1, increment of e's arrival counter). Kimi 7232 B: 3 entries = 60 tiles = 20 full packets instead of 21 (168 → 160 packets per link). Off when `packet_tiles > slot_tiles`. Exchange alone (grouped): 53.7 → 52.6 us.
  - **Batch depth in bytes** (`PORT_BATCH_BYTES` = 87 KiB, was `PORT_BATCH_CHUNKS` = 8): 12 packets at 7232 B (exchange 47.3 → 46.2, whole op ~-1 us), 6 at 14 KiB (the port CB shrinks 224 → 168 KiB there).
  - **Path gate** `SEND_RUN_MAX_LOCAL_RATIO` = 6: runs change which heads land last. Where the chip's own DRAM traffic is large next to its input (Kimi Phase 0: DRAM output, ratio 8.2), the interleaved landing hides more local work and every run length cost 1.3-1.5 us (run 1 / 2 / 3 / 6 / all: 87.6 / 89.0 / 89.0 / 90.7 / 91.2 us); packing, fence and batch made no difference there. So runs apply only up to ratio 6 (model configs: 4.2); above it the Phase 0 order is kept (fence and packing still on, null effect).
  - The tail-split landing estimate now uses the entry's share of its source port's list (`TAIL_SPLIT_ESTIMATE` = "share"; ports carry 12 or 24 entries). Measured null vs the raw index (56.2 vs 56.1); kept as the right estimate for unequal ports. Tail size 8 / 12 / 16 under the new order: null.
  - Reused: the port reader/sender batching, header ring, CB and route headers, the landing/arrival protocol. Added: the stream header, per-destination ready counters, 6 host knobs (+ `A2AMM_XP` measurement override, off by default).
- Perf (harness `perf_measure.py`, 5 experiments, `CCL_PERF_CASES=all`, baseline measured in the same session with the R3 code; fastest / slowest chip; none dispatch-bound):
  - Kimi model (focus, LOOSE_CASES[0]) 60.36 / 64.82 → **55.84 / 61.36 us** (`compare`: win)
  - Exchange alone (`nowrite,noweight,nocompute`), focus case: 50.45 / 52.07 → **45.77 / 48.35 us** (bytes per busiest link 0.98 MB → 21.4 GB/s, 44% of 48.5 GB/s, was 40%)
  - Kimi Phase 0 87.62 / 91.52 → 87.62 / 91.53 (null; gated)
  - DeepSeek 77.31 / 83.55 → 72.16 / 80.93 (win); GLM 59.87 / 64.88 → 58.82 / 63.88 (win)
  - Exposed local time on the focus case: 55.8 − 45.8 = ~10 us (unchanged): the whole-op gain tracks the exchange gain.
- Accuracy achieved: unchanged math; the exchange is still bit-exact (row / head provenance tests pass). Unit directory 36/36 (incl. precision baseline: bf16 fp32 DEST HiFi4 PCC 0.9999986, rel RMS 0.00169).
- Golden test progress: targeted slice 29/29 — one cell per INPUTS shape (18: 16-bit DEST, L1 out, bf16 weight; covers packing on and off (slot < packet), R1' retile with packing, B=2, multi n-block, deep K), the 4 LOOSE_CASES, regression suite except `test_small_packet_size` (cannot open a 4160 B fabric on this board; environment, as before).
- Issues encountered: the first per-group ready counters assumed one run per destination and asserted on the interleaved order; made them per destination chip (bitmask of consumed readies). Grouped order made the whole op 1.2-1.5 us worse than the legacy mix on Kimi model and 3.7 us worse on Kimi Phase 0 although the exchange alone improved; runs of 3 recovered the model cases and the ratio gate the Phase 0 case.
- Tests added: none (covered by the unit and golden suites). Perf JSONs under `perf/r4/`; `probes/r3_variants.py` takes `OUTDIR`.

## Refinement 5 — Local work: weight-read and matmul efficiency on the perf-flagged case
- Date: 2026-11-24
- What was done: measured the local-only breakdown on LOOSE_CASES[0], then built and measured every named local lever. Each is a single-source knob parked at a byte-identical default (no lever won).
  - Breakdown (harness, fastest / slowest / chip average): full 55.76 / 61.39 / 58.69; exchange alone 45.81 / 48.31 / 47.05; `noweight` 54.20 / 61.94 / 58.21; `nocompute` 52.99 / 58.05 / 55.64; `nowrite` 54.44 / 58.58 / 56.65; pairs (weight / write / compute only on top of the exchange) 50.73 / 53.52 / 49.89 fastest. Zones: a whole head that lands late needs 7.8 us of matmul (32.9 cycles per tile, HiFi2 SOL 32) plus ~1 us of drain; a split half head needs forward 0.9 + 3.7 + 1.1 us.
  - **Matmul subblock** (`A2AMM_XP sb=hxw` hook): 1x8 55.70 (kept), 1x4 55.80, 1x2 55.92, 5x1 56.89 us.
  - **Weight reads**: coarser transactions are impossible because consecutive interleaved pages sit in different DRAM banks. Chunk 8 / 4 KiB: 55.65 / 58.11 (16 KiB kept). New `WEIGHT_PREFETCH_BLOCKS` (all of a slot's blocks before its landing wait, cb_weight depth already 2): 2 → 57.00 (worse: the extra 64 KiB per core during the exchange costs 1.3 us), parked at 1. New `WEIGHT_PACE_CYCLES` (a core's first weight read waits pace x its head's estimated landing share; per-slot runtime word, `riscv_wait`): 10 / 20 / 30 us → 55.79 / 55.59 / 55.45 fastest, slowest 61.25 / 62.04 / 61.77 (null), parked at 0. Compute reader on NoC0 / writer on NoC1: 82.2 us.
  - **Writer twin of the throttle** `OUT_WRITE_CHUNK_BYTES` (one write barrier per chunk): 4 / 8 / 16 KiB → 60.80 / 56.47 / 55.79, parked at 0 (flush per row group).
  - **Tail split co-tune**: split count 12 / 16 / 24 / 40 → fastest 54.93 / 54.32 / 54.14 / 54.15 but slowest 62.05 / 62.64 / 62.97 / 62.70 (`compare` rejects: slowest regressed), chip average flat at 58.4-58.6. New `TAIL_SPLIT_HELPER_PICK = "nearest"` (shortest forward path): null at 8, same trade at 16 / 24. Kept "pool", 8.
  - Finding: readies go out at program start, so every chip's next call is gated by the last chip to finish. The steady-state period (calls 2-4 of each trace: 57.6 us) is the worst chip's send span + post-landing tail, and which heads land last varies per call. A static split set therefore reshuffles time between the chips without lowering the worst case.
  - Reused: the reader's per-slot record, weight loader and tail plan; the writer loop. Added: 4 host knobs, reader CT arg `w_prefetch_blocks`, a per-slot pace word, writer CT arg `out_chunk_tiles`, XP hooks `wchunk`, `ochunk`, `wpre`, `wpace`, `pick`, `sb`, `cnoc=swap`. Removed: a writer/reader same-NoC override, which hung (the two RISCs share one NoC's transaction counters; there is now an assert against it).
- Perf (harness, 5 experiments, `CCL_PERF_CASES=all`; `compare` vs R4 final, all null): Kimi model 55.84 / 61.36 → 55.70 / 61.55; Kimi Phase 0 87.62 → 87.59; DeepSeek 72.16 → 72.27; GLM 58.82 → 58.79 us. Not dispatch-bound. JSONs under `perf/r5/`.
- Accuracy achieved: unchanged math (default path identical). Unit directory 36/36, incl. precision baseline (bf16 fp32 DEST HiFi4 PCC 0.9999986, rel RMS 0.00169).
- Golden test progress: targeted slice 20/20 (4 LOOSE_CASES, (1,24,160,128) bf8b R1' x 4, (1,8,48,64) R1' x 8, N=8192 L1 x 4).
- Issues encountered: no lever beat the noise; the remaining headroom is cross-chip (see the Outcome line). Untried next: per-row-tile arrival increments (sender-side OVERLAP), which shorten every remote head's post-landing path whatever the landing order.
- Tests added: none (defaults byte-identical). Probes 042-053 (variant sweeps via `r3_variants.py`).

## Perf 1
- Date: 2026-11-24. Focus: LOOSE_CASES[0] (Kimi K3.1 model config: input (1,24,640,128) bf16, weight (1,96,128,512) bf16, HiFi2, 16-bit DEST, L1 output; 1x4 ring, FABRIC_2D_TORUS_XY, 2 links, 7232 B packets). Roofline 32.3 us (DRAM), links 20.3 us, matmul 6.6 us. The cell is SUPPORTED.
- Instrumentation: `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) does not exist in this tree. The op's existing equivalent, `A2AMM_ZONE(name)` (compiled out unless `A2AMM_ABLATE=zones`), stays as the permanent instrumentation. Added zones: `psd_hdr_setup`, `psd_open` and `psd_wait_ready` (the ready wait, zoned apart from the send). Added ablation stub `nosend`: the sender sends only each entry's arrival increment, as a header-only packet with no payload.

### Measured breakdown (harness, 5 experiments, fastest chip, us)
| run (cumulative peel) | fastest | slowest | average |
|---|---|---|---|
| full | 55.81 | 61.20 | 58.64 |
| nowrite | 54.43 | 58.66 | 56.72 |
| nowrite,nocompute | 50.53 | 52.30 | 51.49 |
| exchange alone (nowrite,nocompute,noweight) | 45.91 | 48.22 | 47.06 |
| exchange alone, port DRAM reads also stubbed | 45.74 | 47.18 | 46.46 |
| local alone (nosend,noportread) | 41.07 | 43.78 | 42.42 |
| nosend,noportread,nowrite | 40.42 | 42.83 | 41.80 |
| nosend,noportread,nowrite,nocompute (weight reads + floor) | 35.96 | 36.74 | 36.41 |
| everything stubbed | 17.95 | 18.29 | 18.12 |
The 17.95 us all-stubbed floor is inflated by the `nosend` stub itself: each header-only increment blocks until it is flushed.

Per-part speed of light:
- Exchange: 0.98 MB on the busiest link. At the link rate that is 20.3 us. At the measured per-packet rate for 7232 B packets it is ~44 us (from the `fabric_all_gather` fit, 0.17 us + bytes / 72 GB/s per packet = 0.273 us; this op's send phase runs at 0.25 us per packet).
- Weight reads: 12.6 MB / DRAM peak = 24.6 us.
- Matmul: 6.6 us.

Overlap loss = full − slowest part alone = 55.8 − 45.9 = 9.9 us.

Zones (one untraced call, every chip and port):
- **Sender startup.** `psd_hdr_setup` takes 4.3 us on every port: `fabric_set_unicast_route` costs ~0.33 us per L1 header, and there are 12 headers. `conn.open()` adds 0.4 us and the fence 0.3 us. So no ready and no data packet leaves any chip before ~5 us.
- **Post-exchange chain.** The last remote slots land at 46–54 us and the last output write ends at 56–59 us: 5–8 us after the last landing.

Ranked bottlenecks:
1. TRANSPORT start latency, 4.3 us on the exchange's critical path.
2. OVERLAP: the post-exchange chain (5–8 us), plus the weight reads' contention with the exchange (+4.6 us).
3. The TRANSPORT data phase. It is already at the example's per-packet rate for this payload size, which the case fixes, so it has no headroom here.

### Portfolio (cap 2) and verdicts
- **sender_startup (TRANSPORT, handshake cost)**, `perf_experiments/sender_startup/`. Route each destination once into a stack image, store only the route words into that destination's L1 headers, and overlap `open_start()` / `open_finish()` with the setup.
  - Effect on the sender: header setup 4.28 → 1.79 us, readies leave at 2.35 instead of 4.9 us, data starts at 2.70 instead of 5.17 us.
  - Focus case alone: fastest −1.0 us (−1.8%), slowest +0.7 us, average flat. That is inside the noise band, so on its own this idea is NULL on the focus case. The saving turns into skew between chips: the ring period is set elsewhere.
  - Other cases, stacked on the tail-chain change (same session): GLM fastest 55.79 → 53.57, DeepSeek 71.63 → 70.03, Kimi Phase 0 87.64 → 87.30.
  - Domain: applies everywhere, no exceptions. 148/148 golden cells pass on the experiment's copy.
  - Menu of other variants: `sslazy` (headers built on first use) lowered the fastest chip as much, but `compare` rejected it twice because the slowest chip rose 1.1–1.45 us. Copying L1 to L1, or splitting the open alone, gained less.
- **tail_chain (OVERLAP, post-exchange chain)**, `perf_experiments/tail_chain/`. Tail M-split: the 4 remote heads per chip estimated to land last are cut by output rows into 3 pieces (2, 2 and 1 tile-rows).
  - The source port sends each piece's rows straight into its own core's slot, with its own arrival counter. There is no forward hop, and the first pieces start their matmul while later rows are still in flight.
  - Piece slots prefetch both weight n-blocks.
  - With the M-split in place, the N-halving tail split can cover 12 heads on the grouped path instead of 8.
  - Focus case alone: 55.61 / 61.80 → 54.30 / 60.35 (fastest / slowest), average 58.93 → 57.40. That is −2.4% on the fastest chip, at the edge of the band. GLM: −5.7% fastest.
  - Zones: the chain after the last landing shrank by ~1 us per chip. What is left is whole heads that land later than estimated (the landing order varies per call).
  - Menu (fastest / slowest / average): M8 P3 N8 55.03 / 60.62 / 57.90; M4 P3 N8 55.08 / 60.09 / 57.66; M4 P3 N12 54.30 / 60.35 / 57.40 (chosen). Variants with more pieces or no N-split were worse.
  - Precision: identical. The split only divides output rows, and fidelity, DEST width and dtypes are unchanged.

### What graduated: both, combined
The two changes touch different stages (sender startup vs. the tail plan and the slot record), so they do not conflict. Together they are faster than either alone.

The sender startup change is now the only sender path, everywhere; the old 12× L1 routing loop and `conn.open()` are deleted.

The tail M-split and the grouped path's 12-head N-split are the default. They are carved out of one path only: the interleaved send path, which is the existing `SEND_RUN_MAX_LOCAL_RATIO` gate (local ratio > 6, Kimi Phase 0). The carve-out was earned by a measured regression: on Phase 0, ungated, the change gave 87.59 / 91.35 → 92.82 / 98.57. The cause is extra helper slots on local-head cores under HiFi4 (12 N-split heads alone: 91.21; 4 M-split heads alone: 88.30). Gated, Phase 0 is within noise.

The host also skips the M-split where it cannot be expressed: without packing, when a piece would be smaller than one packet, and when the tail split would be off anyway.

Whole op (harness, 5 experiments, `CCL_PERF_CASES=all`, all measured in this session; fastest / slowest / average, us):
| case | baseline | tail_chain only | graduated (both) | `compare` vs baseline |
|---|---|---|---|---|
| Kimi model (focus) | 55.79 / 61.27 / 58.63 | 54.41 / 59.96 / 57.20 | **53.45 / 60.49 / 57.07** | win (−4.2% fastest) |
| Kimi Phase 0 | 87.67 / 91.41 / 89.42 | 87.64 / 91.47 / 89.45 | 87.30 / 90.49 / 88.79 | win (−0.4%: flat) |
| DeepSeek model | 72.38 / 80.58 / 76.61 | 71.63 / 77.98 / 74.96 | 70.03 / 76.70 / 73.55 | win (−3.2%) |
| GLM model | 58.78 / 63.98 / 61.55 | 55.79 / 60.61 / 58.37 | 53.57 / 60.75 / 57.28 | win (−8.9%) |

- No case is DISPATCH-BOUND. JSONs are under `perf/p1/{base,B,AB}/`, and the ablations under `perf/p1/abl/`.
- No regression on any case. The guard set covers one representative per path: the grouped and interleaved send paths, K = 128 and K = 192, and 24 and 32 heads per chip. The golden suite covers the rest: m_non_aligned (retile), Nt = 1 (no split possible), B = 2, deep K, wide N, bf8b weight, and DRAM and L1 output.
- Golden: `eval/golden_tests/all_to_all_matmul/` 156/156 pass. Unit directory 36/36 pass. The tail_chain subagent also ran 20 back-to-back calls and a stress variant (every head split, 5 pieces).
- Lesson from tail_chain: adding per-core runtime arguments made back-to-back calls hang. The new fields are now packed into the existing words, so each kernel keeps its argument count. Test any future change to the runtime arguments back-to-back.
- Remaining headroom on the focus case: 53.45 us vs a 32.3 us roofline.
  - The data phase sits at the per-packet rate for 7232 B packets (~41 us for the busiest link's 160 packets).
  - Whole heads that land later than the static estimate still cost 7.8 us of matmul after landing.
  - The weight reads still contend with the exchange (+4.6 us).
  - Not tried yet: a dynamic tail, where the piece and helper assignment follows the actual landing order.

### Helper bypasses — none
No raw LLK. Compute still calls `matmul_block`, now with a runtime row range. The sender uses the fabric adapter's public `open_start()` / `open_finish()`.
