# Routed expert NoC study, unified GU_IL, and flat expert + combine overlap: work log

2026-10-08 .. 2026-10-10, Blackhole LoudBox (8 x p150, 11 x 10 worker grid per chip). Three trees:

| Tree | Branch (local, not pushed) | What |
|---|---|---|
| `/localdev/mstaletovic/tt-metal-main` | `mstaletovic/unified-noc-study` | unified_routed_expert_ffn NoC clone, GU_IL, flags |
| `/localdev/mstaletovic/tt-metal` | `dnijemcevic/model_bringup` | flat planner: restricted rows / 12-column grids |
| `/localdev/mstaletovic/tt-metal-pr58093` | `mstaletovic/flat-combine-overlap` | PR #58093 + flat expert + flat/combine overlap |

Units: us per expert for the single-op sweeps, us per layer (all local experts) for the overlap cases, traced replay
unless noted. DRAM peak 512 GB/s.

---

## 1. unified_routed_expert_ffn (main): NoC pattern study

The op is two chained 2D-multicast matmuls (gate+up fused, then down) on a fixed 11 x 8 grid. h passes core to core
along the rows (down K-block kb is column slice kb of h), all experts run in one launch with device-side counts, and M
is chunked at <= 1024 tokens, each chunk re-reading all weights. Senders are diagonal: weight senders at row gx % 8,
x senders at column (gy + 1) % 11. Reader (NCRISC, NoC 0): x, gate, half of down, the down multicast, the activated
multicast, counts. Writer (BRISC, NoC 1): up, the other half of down, the gate/up multicast (IN1_WRITER_MCAST), output.

Python clone of the data movement: `bench_moe/noc_clone/` (`clone.py`, `kernels/clone_dm.cpp`, `sets/*.py`,
results `generated/noc_clone/s*.jsonl`), ~200 variants, mostly GLM 5.3. Results at M 1024, weights only:

| Question | Result |
|---|---|
| 1. Max weight rate, diagonal, ND-sharded | reads alone 45.6 us, 465 GB/s (91%); reads + column mcast as today 62.3 us, 341 GB/s (67%); issuing the next block's reads before mcasting the current one 49.0 us, 433 GB/s (85%) |
| 2. All weight readers in row 0 | ~half: 213 GB/s reads alone, 207 with mcast. NoC 1 routes y then x, so every NoC 1 response ends on that row's x-links (60 of 64 B/cycle). Reader-only (NoC 0) row-0 reads match the diagonal |
| 3. Rows 0 + 1 reading | 258-270 GB/s, only partial recovery. No extra sync needed; the ND shard need not change (half/half and alternating rows measure the same) |
| 4. VCs | no meaningful effect (+-1% weights only, up to ~7% at M 32): mcast VC 5 vs 4, read-request VCs, acks / output on other VCs |
| 5. Moving traffic between NoCs | activated mcast on the writer -16% at M 32 (clone); all reads on writer + all mcasts on reader much worse (92.8 us); gate/up mcast back on reader -5% weights only but slower in full pattern |

Best clone combination (prefetch + activated mcast on writer): M 32 78.9 -> 51.7 (-35%), M 256 86.5 -> 69.1 (-20%),
holds on K2.7, K3, M3, DSv3 (-30..-33% at M 32).

**Real op, what carried over:** `DS_ACT_WRITER=1` (activated mcast on the writer), commit `79743183f61`, us per forward:

| Model | M 32 | M 128 | M 256 | M 512 |
|---|---|---|---|---|
| GLM 5.3 | 682.7 -> 641.9 | 691.6 -> 649.8 | 752.2 -> 714.6 | 1059 -> 1045 |
| K2.7 | 1141 -> 1077 | 1167 -> 1096 | 1282 -> 1209 | 1805 -> 1786 |
| MiniMax M3 | 474.0 -> 445.1 | 470.7 -> 450.6 | 510.4 -> 481.7 | 771.9 -> 759.3 |
| Kimi K3 | 2563 -> 2397 | 2524 -> 2398 | 2630 -> 2492 | 4056 -> 3927 |

Off by default: it adds a worker multicast on NoC 1, the fabric-CCL hazard the op's comments describe; needs a run
beside concurrent CCL first.

**Did not carry over:** writer-side prefetch (two variants, both correct, both +3%): in the real op the writer cannot
start block s+1 until the reader grants the slot, and the reader also waits on the writer's read of block s. Needs
the reader restructured too. Precomputed ND-shard addresses: only 4-7%. At M >= 256 the real op is compute bound and
the NoC variants stop mattering (clone with 16-20 cycles per tile-matmul tracks the real op within ~10%).

## 2. unified: interleaved gate/up (GU_IL) + SFPU on pack

v1 (`02c5cd2d892`): one weight tensor [K, 2 hidden] (gate tile j at column 2j, up at 2j+1), detected from the shape;
one matmul per subblock writes (gate, up) pairs into DST; activation on the pack thread's SFPU, packed straight into
`activated`. Correct (PCC 0.9985) but no gain: GLM M 32 / 256 / 1024 -1.5% / +1.4% / +1.2%. Profile (`DS_ZONES=1`):
the silu pass (9%) just moves to the pack SFPU, and down starts later (632k -> 801k cycles) waiting on activated.

v2, pipelined (`56f347dc565`): items run gate/up of item i, then down of item i-1 (`adaptive_chunk::PipeOrder`); item
i-1's activation rides in two spare DST tiles of a matmul subblock of item i; fused one-pass SiLU-GLU
(`g * sigmoid(g) * u`); `VALID_ROWS` skips padding rows. PCC >= 0.9986 vs the separate path.

| Model | GU_IL alone M 32 / 256 / 1024 | + DS_ACT_WRITER + DS_VALID_ROWS vs today's op |
|---|---|---|
| GLM 5.3 | -2.8 / -5.2 / -2.0% | -9.9 / -10.5 / -2.7% |
| Kimi K2.7 | -2.4 / -5.4 / -2.1% | -9.7 / -11.2 / -2.5% |
| Kimi K3 | -8.7 / -6.9 / -9.4% | -15.9 / -14.9 / -10.2% |
| DSv4 Flash | -2.3 / -6.3 / -6.5% | -10.0 / -12.3 / -7.8% |
| MiniMax M3 | -3.3 / -4.9 / -3.6% | -10.0 / -10.3 / -4.3% |

Uneven routing (`GU_IL_COUNTS`): GLM -1.3%, K3 -7.1% (K3 with the two flags only -3.0%, not understood).

What did not work: a noinline lambda for the pipelined loop (default path -5..6%; replaced by an inline while loop
over PipeOrder, default path keeps its original for loop); activation on the math thread or other DST placements
(no change).

**Why flat hides its activation and unified does not** (measured with flat's `MIMO_FL_NO_ACT`: 0% / 2.4% / 3.9% at
M 128 / 512 / 2048, unified still ~7-8% at M 1024): flat accumulates a sub-block in DST straight through all of K, so
the pack thread is idle for a whole K-loop; unified re-packs every subblock into L1 partials with packer accumulation
every K-block, so the pack thread is near saturated and the activation lands on the busiest thread. Earlier claim
"SFPU and FPU do not run in parallel on this hardware" was wrong. Ideas (not done): wider K-blocks (32 instead of 16,
L1-limited), two resident K-blocks in DST per pack, cheaper (LUT) sigmoid (numerics change).

Before main: `TtRoutedExpert` must build the interleaved weights (see `bench_moe/test_gu_il.py`), no fused biases in
GU_IL, DS_ACT_WRITER needs a CCL-concurrency check.

## 3. flat vs unified (each in its own default setup), us per expert

| Shape | M | flat | unified stock | unified best | best / flat |
|---|---|---|---|---|---|
| GLM 5.3 | 32 / 128 / 256 | 52.5 / 60.9 / 73.7 | 85.2 / 86.5 / 94.0 | 76.5 / 80.0 / 84.3 | 1.46 / 1.31 / 1.14x |
| | 512 / 1024 / 2048 | 114.3 / 221.7 / 436.1 | 132.5 / 250.5 / 499.6 | 130.7 / 243.2 / 484.9 | 1.14 / 1.10 / 1.11x |
| Kimi K2.7 | 32 / 128 / 256 | 61.3 / 69.9 / 83.7 | 95.2 / 97.2 / 106.6 | 85.9 / 89.5 / 94.5 | 1.40 / 1.28 / 1.13x |
| | 512 / 1024 / 2048 | 118.7 / 224.3 / 437.8 | 150.5 / 285.5 / 571.0 | 149.1 / 279.3 / 557.8 | 1.26 / 1.25 / 1.27x |
| Kimi K3 | 32 / 128 / 256 | 49.0 / 56.0 / 68.3 | 91.6 / 90.0 / 93.9 | 77.0 / 79.2 / 79.9 | 1.57 / 1.41 / 1.17x |

Small M: flat streams weights at 76-82% DRAM, unified has 11 weight-reading cores plus per-block handshakes. Large
M: compute; flat has 110 cores vs unified's 88, and unified pays the row-major x tilize (~15%) and a partly hidden
activation. Not format-matched (flat: fp32 DEST down, RM y; unified: bf16 RM x, bf8 out, bf16 DEST).

## 4. PR #58093 (combine_fabric2d overlapped with the hybrid routed expert)

pmilojevicTT, "Overlap combine fabric2D (newest combine) with hybrid routed expert OP". Routed expert + combine in
one program per chip; every RE writer bumps a per-step (and per-chunk) counter on combine's collector after a write
barrier; the collector releases combine's untilizers; a global `go` semaphore (multicast over the RE rectangle) holds
the writers until the collector has zeroed its counts. Combine on rows 0-1 (senders next to the eth cores,
untilizers, collector), RE on the 11 x 8 below.

Reproduced on the LoudBox 8 x 1 (models scaled to 8 chips: Kimi 96 experts / 12 per chip, GLM 64 / 8 per chip,
seq 640/chip):

| Case | RE | combine | sequential | overlap | combine hidden |
|---|---|---|---|---|---|
| Kimi K2.7 balanced | 1122.1 | 555.7 | 1677.8 | 1350.4 | 59% |
| Kimi K2.7 hot | 1699.0 | 831.4 | 2530.4 | 1898.7 | 76% |
| GLM 5.3 balanced | 777.2 | 420.4 | 1197.6 | 947.9 | 59% |
| GLM 5.3 hot | 1404.3 | 690.9 | 2095.2 | 1585.6 | 74% |

The ~15% end-to-end claim needs the Galaxy and the deepseek_v3_d_p model; not tested.

**Ring on the LoudBox:** the box is a 2 x 4 grid (2 links per neighbour pair); auto-discovery tries mesh shapes
outer / fabric types inner (`topology_mapper.cpp` ~1700) and returns a 2 x 4 MESH before ever trying 8 x 1 as a ring,
so FABRIC_2D_TORUS_Y silently falls back to a mesh. `TT_MESH_GRAPH_DESC_PATH=p150_x8_ring_8x1.textproto` (8 x 1, ring
on the long axis) maps the perimeter (1, 5, 4, 6, 7, 3, 2, 0) with direct hops. Combine 1-2% faster, overlap results
unchanged. Worth an upstream issue (not filed).

## 5. flat on fewer cores / other grids (our branch, `d389bca0656`)

Knobs: `MIMO_FL_ROWS=y0,y1`, `MIMO_FL_COLS`, `MIMO_FL_ND` (cap down cores), DRAM-optimal fallback list for the
row-dispatch grid, `bw_max` 4 on 12-column grids, rectangles reshaped to the exact gate/up core count.

| Shape | M | 110 cores | 99 (rows 1-9) | 88 (rows 2-9) |
|---|---|---|---|---|
| GLM 5.3 | 32 / 128 | 52.5 / 60.9 | 51.8 / 61.0 | 56.4 / 68.4 |
| | 512 / 2048 | 114.3 / 436.2 | 165.1 / 650.9 | 165.0 / 650.8 |
| Kimi K2.7 | 32 / 128 | 61.3 / 69.9 | 59.8 / 70.4 | 63.5 / 78.1 |
| | 512 / 2048 | 118.7 / 437.9 | 211.7 / 835.1 | 193.0 / 762.4 |
| Kimi K3 | 32 / 128 | 49.0 / 56.0 | 43.6 / 50.3 | 46.6 / 53.6 |
| | 512 / 2048 | 103.4 / 404.2 | 97.0 / 380.0 | 122.6 / 482.6 |

64-column models (GLM, K2.7, It = 64) need 64 gate/up cores at np 1; 9 rows of the gate/up columns hold 60-63
cells, so the planner drops to np 2 (32 cores): +49..+91% at M 2048. K3 (already np 2, 48 cores) gains when idle
rectangle cells become down cores (full grid: 49.0 -> 43.9 at M 32), but is sensitive to the down-core count (34-38
best at M 2048; 40-42 is +22%).

**12 x 9 (dispatch on a row):** needs the fabric Tensix MUX passed into `DispatchCoreConfig(...,
fabric_tensix_config=MUX)` on a fabric 2 x 4 mesh (global setting is not enough, single chip without fabric does
nothing). The driver's DRAM-optimal worker query throws there (picks a dispatch-row core), hence the fallback list.
No real-time profiler records: wall time. GLM 32 / 256 / 1024 / 2048: 52.3 / 70.9 / 207.8 / 405.9 (0 / -4 / -6 /
-7%); K2.7 flat (436.9 at 2048); K3 (ND 36) 51.3 / 98.9 / 381.0 at 128 / 512 / 2048 (-8 / -4 / -6%).
**12 x 10 (slow dispatch):** 0.85-1.4 ms launch skew per call lands in kernel time; last-start timing removes most;
M 2048 GLM -8%, K2.7 +4%, K3 -6% (estimates). The Galaxy's east DRAM readers are in column 7 (not 6): its layout
(east-shifted path in the planner) is not measured yet.

## 6. flat expert overlapped with combine (PR tree)

Port: `ec7316c7a94` copies the flat bringup into the PR tree (`ttnn/ttnn/bringup/`, hooks in `ttnn/CMakeLists.txt`,
nanobind, `ttnn/__init__.py`). Commits use `--no-verify` (the global-torch-import check rejects `flat_expert.py`).

### 6.1 Initial version (`1f772644208`)
- `Program::append(const ProgramDescriptor&)` in tt-metal (factored out of the descriptor constructor): combine's
  per-chip program is appended to flat's program.
- `flat_combine_overlap` device op (mesh workload): builds combine with `create_combine_workload` (threshold 0: one
  pass, experts in slot order; its L1 in flat's arena, which spans the whole grid), then per chip flat's program +
  combine's descriptor, then fills each y writer's 4 report args (collector x / y, count array, go address).
- `se_cmb_done.hpp` `SeCmbDone`: each y writer (down cores, reader tails) counts the blocks it still owes per local
  expert and reports the longest finished prefix of slots (step s = slot s) to the collector, behind a write
  barrier; waits for `go` before the first report. Hooks in `se6_drecv.cpp`, `se9_rdown.cpp`.
- bfp8 y tiles, combine's untilizers unchanged, combine on rows 0-1, flat on rows 2-9 (88 cores, np 2).

Test: `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_flat_combine_overlap.py`, the PR's four 8 x 1 cases.
Accuracy: overlap vs flat then combine bit-exact (first call and cache hit), 10240/10240 slots vs the PyTorch
reference. Perf: traced flat, combine and overlap.

| Case | flat 110 + combine (baseline) | overlap initial (88 cores) |
|---|---|---|
| Kimi bal | 1457.4 | 1312.6 (-9.9%) |
| Kimi hot | 1829.9 | 2129.2 (+16.4%) |
| GLM bal | 1032.9 | 969.5 (-6.1%) |
| GLM hot | 1735.0 | 1837.1 (+5.9%) |

### 6.2 Row-major y, combine without untilizers (`e2ba47da16a`)
- Flat writes bf16 rows (`y_row_major`, now the overlap's default); combine's readers read them straight from DRAM,
  gated per step on the collector's `ready` (`CMBF2D_READY_SEM`, first read of a step's own rows waits; forwarded
  chunks never wait). Readers become the collector's waiting cores; zero untilizers when the dispatched buffer is
  ROW_MAJOR; the collector prefers the sender row.
- Combine then needs only row 0 (4 senders + collector) -> flat on rows 1-9 (99 cores, still np 2 for 64-col models).
- Row-major writers report as their row tiles' transaction ids flush (`SeYRmWriter::done_a`, `report(false)`); with
  `SE_Y_NC` the down cores' NCRISC (`se6_dw.cpp`) is the writer that reports.

Overlap: Kimi bal 1295.5, hot 2198.1, GLM bal 933.0, hot 1798.8 (3 of 4 better than 6.1). Bit-identical.

### 6.3 Diagnosis (device profiler zones, kept in the kernels)
Zones: `CMB-RELEASE` (collector, ends when step s is released), `CMB-STEP` / `CMB-FABRIC` / `CMB-LOCAL` / `CMB-GATE`
(reader). Script: `models/demos/deepseek_v3_d_p/tests/op_unit_tests/flat_combine_tools/` (`cmb_timeline.py`, `cmb_split.py`).
- `TT_CMBF2D_IDLE=1` (combine kernels exit at once): overlap = flat solo + ~9 us. Flat is not slowed by the reports.
- pin 1: the largest expert (often slot 0) is chunked through the whole schedule, slot-prefix reporting releases
  step 0 only at ~190-550 us; ring coupling: step s on every chip waits for the slowest chip's step s.
- pin 0 (schedule in slot order): releases come early and evenly, but combine's per-reader fabric phase is ~2.2x
  slower beside flat (~895 us vs 405 us standalone, plus ~137 us gate wait). Contention, not late release, is the
  main loss. Standalone combine's step costs are uneven too (late steps 100-140 us).

### 6.4 Combine reader on its own VC (`e1fe9fec0ac`)
Flat's VCs: 0 h exchange, 1 default writes + all read requests (weights), 2 row-major y, 3 weight forwarding.
Combine had everything on VC 1. The read-request VC is fixed at init in dedicated-NoC mode (`read_req_vc` only applies
in dynamic mode), so the reader reprograms the read cmd buffer's NOC_CTRL at start and restores VC 1 at exit; local
writes take a VC argument. Default now VC 0 for both (`CMBF2D_RD_VC` / `CMBF2D_WR_VC` probes; VC 2 and 3 measure like
0 - what matters is leaving VC 1).

| overlap us | VC 1 | VC 0 |
|---|---|---|
| pin 1: Kimi bal / hot / GLM bal / hot | 1295.5 / 2198.1 / 933.0 / 1798.8 | 1222.4 / 2182.3 / 897.2 / 1763.8 |
| pin 0: Kimi bal / GLM bal | 1221.7 / 881.3 | 1147.4 / 824.0 |

### 6.5 64 gate/up cores next to combine (`94f108ef9b9`)
- `MIMO_FL_RD_SAMECOL` (planner probe): a bank's second reader in its first reader's column, so columns 1 and 7 join
  the gate/up rectangles: 64 gate/up cores on rows 1-9, but only 15 down cores.
- `MIMO_FL_XDOWN="x,y;.."`: extra down cores outside the capped rows. Combine cells on any chip (FLAT_CMB_LOG):
  (0,0) (2,0) (3,0) (5,0) (6,0) (7,0) (8,0) -> (1,0) (4,0) (9,0) (10,0) free on every chip -> 19 down cores. `go`
  stays a multicast over the flat rectangle; cores outside it get it unicast
  (`CombineFabric2dParams::routed_expert_extra_cores`).
- Flat solo, down-core sensitivity on the full layout (MIMO_FL_ND): Kimi bal / hot 26 down 932 / 1055, 21 down
  1028 / 1222.

### 6.6 Current best (overlap, us; all bit-identical, 10240/10240)

| Case | np 2, rows 1-9 | 64 gu + 19 down | flat 110 then combine | best vs baseline |
|---|---|---|---|---|
| Kimi bal | **1147.4** (pin 0) | 1564.8 | 1457.4 | -21% |
| Kimi hot | 2182.3 | **1803.6** | 1829.9 | -1% |
| GLM bal | **824.0** (pin 0) | 883.8 | 1032.9 | -20% |
| GLM hot | 1763.8 | **1549.0** | 1735.0 | -11% |

Balanced routing wants np 2 (more down cores), a hot expert wants np 1 (full math rate); routing is only known on
device, so one fixed program cannot pick. pin 1 is better for hot, pin 0 for balanced. Combine is 25-64% hidden.

## 7. What did not work / dead ends
- Unified: writer-side weight prefetch (+3%), GU_IL v1 without pipelining (no gain), noinline pipelined loop.
- Flat on 88 / 99 cores with the default planner for 64-column models: np 2 fallback, hot expert +45..60% flat time.
- `MIMO_FL_RD_SAMECOL` alone (64 gu, 15 down): Kimi bal 1211.6 / hot 1336.6 flat solo, down-starved.
- Row-major y overlap with pin 1: largest-expert chunking delays slot-prefix releases.
- `MIMO_FL_ND=15` on the full layout: 2329 / 4138 us (uneven column split; not representative).
- A merge of PR #58093 into our branch (conflicts in core files and combine): based the work on the PR tree instead.

## 8. Next ideas (rough order)
1. **Combine's forwarding buffer in L1** (row-0 reader/sender cores) as a ring with credits to the upstream sender,
   instead of DRAM sized for the whole launch. Removes ~40 MB of DRAM write + read per chip (Kimi: ~1.1 intermediate
   hops per token on the 8-ring), the bulk of combine's avoidable DRAM traffic. Combine-only change.
2. **More combine reads in flight** (`NUM_L1_SLOTS` 8 -> 16, batch 4 -> 8; arena room on row 0 is free now): cheap
   probe of how latency-bound the readers are beside flat.
3. **Pipelined local-phase copies** (today one read, barrier, write, barrier per token through one slot).
4. **y never through DRAM:** combine's readers read y rows from the down cores' L1 (or down cores write into the
   senders' staging), the down core frees its output buffer once combine took the expert. Removes ~36 MB of DRAM per
   chip but couples flat's down pipeline to combine's pace.
5. **Release order = combine walk order:** have combine (and the collector's steps) walk experts in flat's completion
   order (pinned schedule: s1 .. s(K-1), B, sK ..), computable from the replicated counts, so pin 1 keeps its flat
   gain without holding back combine. Or order both by descending count so the combine tail is the smallest expert.
6. **One layout for both regimes:** a third small gate/up rectangle (64th gu core fed by an existing relay) so np 1
   keeps more down cores; better down-core rule (K3's sensitivity); try the readers' down columns (`MIMO_FL_PCD_R`)
   to offload the down cores in the 19-down layout.
7. **Galaxy (12 x 10 with dispatch, east readers in column 7):** measure the east-shifted layout; the 12th column
   should ease the down-core shortage that forces the np 1 / np 2 trade.
8. Combine sender's fabric-side VC (inside the fabric API, not tried); collector placement off the sender row if it
   ever blocks a free cell.

## 9. How to run (PR tree)
```
cd /localdev/mstaletovic/tt-metal-pr58093
source /localdev/mstaletovic/tt-metal/python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD:$PWD/tools
export TT_MESH_GRAPH_DESC_PATH=models/demos/deepseek_v3_d_p/tests/op_unit_tests/flat_combine_tools/p150_x8_ring_8x1.textproto TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
ninja -C build_Release install
scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_flat_combine_overlap.py
```
Knobs: `FLAT_CMB_YRM` (default 1), `FLAT_CMB_PIN` (default 1), `FLAT_CMB_NO_OVERLAP=1 MIMO_FL_ROWS=0,9` (110-core
baseline), `FLAT_CMB_ITERS`, `FLAT_CMB_LOG`, `TT_CMBF2D_IDLE`, `CMBF2D_RD_VC` / `CMBF2D_WR_VC`, `MIMO_FL_ROWS`,
`MIMO_FL_RD_SAMECOL`, `MIMO_FL_XDOWN`, `MIMO_FL_ND`. Profile: `TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1`, then `cmb_timeline.py <csv> <run group>` / `cmb_split.py`.
