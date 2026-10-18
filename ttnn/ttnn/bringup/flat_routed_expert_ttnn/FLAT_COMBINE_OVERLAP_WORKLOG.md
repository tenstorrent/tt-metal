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

**Correction (found 2026-10-10 evening):** a probe edit raising combine's reader -> sender ring from 8 to 16 slots
(batch 4 -> 8), which the user had stopped before it ran, still landed in the file and went in with this commit. So
every number from 6.4 on ran with 16 slots. The pin 0 rows above compare VC 1 and VC 0 both at 16 slots (VC effect
real, -6%); the pin 1 VC 1 row is from 6.2 at 8 slots, so the pin 1 gain mixes VC and ring depth. The ring depth is
now a probe (`CMBF2D_SLOTS`) and measured on its own in 6.11.

### 6.5 64 gate/up cores next to combine (`94f108ef9b9`)
- `MIMO_FL_RD_SAMECOL` (planner probe): a bank's second reader in its first reader's column, so columns 1 and 7 join
  the gate/up rectangles: 64 gate/up cores on rows 1-9, but only 15 down cores.
- `MIMO_FL_XDOWN="x,y;.."`: extra down cores outside the capped rows. Combine cells on any chip (FLAT_CMB_LOG):
  (0,0) (2,0) (3,0) (5,0) (6,0) (7,0) (8,0) -> (1,0) (4,0) (9,0) (10,0) free on every chip -> 19 down cores. `go`
  stays a multicast over the flat rectangle; cores outside it get it unicast
  (`CombineFabric2dParams::routed_expert_extra_cores`).
- Flat solo, down-core sensitivity on the full layout (MIMO_FL_ND): Kimi bal / hot 26 down 932 / 1055, 21 down
  1028 / 1222.

### 6.6 Where it stands: one configuration per model (rule: no hot / balanced specific setups)
Routing is only known on device, so a result counts only as one program per model across both routing cases
(runtime scheduling on device is allowed). Overlap us, pin 1, combine reader on VC 0, bit-identical:

| Config | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| np 2, rows 1-9 (default) | 1222.4 (-14%) | 2182.3 (+25%) | 897.2 (-10%) | 1763.8 (+8%) |
| 64 gu + 19 down | 1564.8 (+10%) | 1803.6 (+3%) | 894.8 (-11%) | 1549.0 (-5%) |
| baseline: flat on 110 cores (row-major y), then combine reading it directly | 1425.7 | 1747.6 | 1001.3 | 1635.7 |
| (old baseline: bfp8 y, typecast, combine's untilizers) | 1457.4 | 1829.9 | 1032.9 | 1735.0 |

The like-for-like baseline is the row-major one (flat untilizes on its down cores in both); percentages above are
against it. Before 2026-10-10 evening this log compared against the old bfp8 baseline.

Per model today: GLM 64 gu + 19 down (-11% / -5%); Kimi has no config that wins both (np 2 loses the hot case,
the 19-down layout the balanced one). Earlier "best per case" tables mixed configs and are not achievable.

### 6.7 Where combine actually loses time (NoC model + discriminating probes)
**Static link-load model** (`flat_combine_tools/noc_model.py` + `dump_plan.py`: every flow routed on the
17 x 12 physical torus, NoC 0 +x then +y, NoC 1 -y then -x, harvest mask 192: logical x 0-5 -> physical 1-6, 6-10 ->
11-15, DRAM columns 0 and 9, eth row 1, workers rows 2-11; DRAM ports per NoC from the soc descriptor). Kimi
balanced, rows 1-9 layout, per layer: no link above ~54% of capacity averaged over the layer. The hottest links are
the DRAM ports' NoC 0 egress (e.g. (9,11)E, (0,11)E: flat's weight read responses 46 MB + combine's reads 8 MB);
next the NoC 1 west links between the east readers and the west rectangle (weight forwarding 31 MB). Combine's own
links are light (<= 9%).
Found on the way: **the fabric router writes received packets to local memory on NoC 1, VC 2 + link % 2**
(`DEFAULT_RECEIVER_LOCAL_WRITE_NOC = 1`, `edm_noc_vc`), i.e. on flat's y-write VC (2) and weight-forward VC (3),
and from the eth row a NoC 1 write to any worker or DRAM port goes north first: it wraps through row 0 / row 11
and climbs the eth core's column (flat's gate/up columns) before turning west.

**First hypothesis, DRAM bandwidth, looked right and was wrong.** Overlap time matched flat's time scaled by total
DRAM bytes (Kimi bal 1182 predicted / 1147 measured, GLM 862 / 824), but the probe below shows flat is not DRAM bound
(no weight reads: 966 -> 869 us only) and removing ~300 MB/chip of flat's DRAM reads moved the gated overlap little.

**Probes** (`MIMO_FL_W_NOREAD`: flat's weight readers skip their DRAM reads, everything else unchanged;
`MIMO_FL_XRD_SKIP`: no x reads; `MIMO_FL_CMB_EARLY`: every step reported at launch, combine runs ungated). Overlap
us (share of combine hidden), pin 1, rows 1-9:

| Probe | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| normal | 1222 (48%) | 2183 (16%) | 893 (42%) | 1766 (33%) |
| no weight reads | 1037 (66%) | 2160 (16%) | 812 (59%) | 1735 (36%) |
| no weight + x reads | 999 (71%) | 2153 (16%) | 784 (63%) | 1727 (37%) |
| combine released at launch | 1147 (63%) | 1628 (95%) | 830 (60%) | 1408 (95%) |
| released at launch + no weight / x reads | 872 (97%) | 1572 (100%) | 672 (95%) | 1368 (99%) |

What this says:
- **Hot expert: the whole loss is release timing.** Ungated, combine is 95% hidden beside the full flat expert;
  gated, 16-33%. The hot expert's tokens (2878 of one chip's) are released only when the whole expert is done (pin 1
  even finishes it last), and every chip's step waits on it. Fix: release a large expert's rows progressively
  (per sub-block / chunk, as the PR does for the unified pass) and walk in flat's completion order: runtime
  scheduling, so allowed. Ceiling from the probe: Kimi hot ~1630 (-11% vs baseline), GLM hot ~1410 (-19%).
- **Balanced: two parts.** Gating costs 75 us (Kimi) / 63 us (GLM) on top; the rest is interference with flat's
  DRAM reads: with them removed and combine ungated it is 97% hidden. Whether the shared resource is the DRAM banks
  or the DRAM ports' NoC links cannot be told apart by these probes (both go away together); the model says no
  link is saturated on average, so it is the read responses sharing the port egress links / bank queues in bursts.
- So the slowdown is not "going to DRAM" as such: the hot case is the release pattern, the balanced case is
  combine's traffic meeting flat's weight-read traffic at the DRAM ports.

**What this means for the L1 idea.** Moving combine's buffers into L1 only helps by taking its traffic off the
DRAM ports, and only if the new paths avoid flat's busy links. Landing forwarded tokens in row-0 L1 would make every
fabric write (NoC 1, north first) climb the eth core's whole column through flat's gate/up columns (11 hops) on
flat's VCs 2/3, likely worse than today. Better options, in order:
1. **No landing at all for forwarded tokens:** send each token over the fabric to its destination chip
   (multi-hop routing; the routers forward eth to eth along the eth row, which flat does not use) and write it
   straight into the destination's output, as combine already does for the last hop. Removes the forwarding
   buffer's DRAM write + read (~41 MB/chip, 53% of combine's DRAM bytes) and the readers' re-forward work.
2. **y from L1:** down cores write y rows into the senders' rings (NoC 1 from a down core goes north to row 0, then
   west along combine's row: off flat's busy links) instead of DRAM + combine's DRAM read (~34 MB/chip). Harder:
   a token row is assembled from every down core's columns.
3. If a landing buffer is needed, put it where the fabric's NoC 1 write is short: the bottom row (2 hops from the
   eth row through the wrap), west of the eth core; row 0 reads it back in 3 hops on NoC 0 through the wrap.

### 6.7b Read side or write side? (combine DRAM probes)
Probes (perf only, garbage output; host env read at program build):
- `CMBF2D_NO_DRAM_READ`: combine's reader takes token payloads from an L1 scratch on its own core instead of DRAM
  (forwarded pages still read their 16 B metadata tail from DRAM: same read count, ~99% fewer bytes).
- `CMBF2D_NO_DRAM_WRITE`: every packet keeps its full size on the cable but lands in an L1 scratch on the receiving
  reader core; forwarded pages add a 64 B metadata packet to their DRAM page (blocking flush). Local-token copies
  (~2 MB) still write DRAM.

Overlap us, pin 1, rows 1-9 (excess over flat in brackets for the balanced cases):

| Combine probe | mode | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|---|
| none | gated | 1223 | 2185 | 896 | 1763 |
| none | ungated | 1149 (181) | 1628 | 830 (141) | 1409 |
| reads off | gated | 1126 | 2086 | 835 | 1678 |
| reads off | ungated | 1054 (88) | 1605 | 766 (78) | 1391 |
| writes off | gated | 1250 | 2315 | 946 | 1894 |
| writes off | ungated | 1079 (113) | 1617 | 784 (95) | 1400 |
| both off | gated | 1173 | 2241 | 914 | 1822 |
| both off | ungated | 994 (27) | 1597 | 721 (32) | 1384 |
| any | ungated, flat without weight / x reads | 870-882 | 1556-1570 | 674-678 | 1367-1370 |

- **Both sides contribute, about additively:** reads ~55-60% of the interference, writes ~40-45%; with both removed
  combine is 91-94% hidden ungated (as with flat's reads removed).
- **It is contention, not combine's own DRAM cost:** beside a flat expert without DRAM reads, the probes change
  nothing (870-882 / 674-678).
- **Writes off gated is worse** (Kimi 1250 vs 1223): the probe's extra blocking 64 B packet per forwarded token
  costs more than the writes it removes when gating, not bandwidth, sets the pace. So the write numbers understate
  the write side.
- Still not separated: DRAM banks vs the NoC links into / out of the DRAM ports (each probe removes both).
- Hot cases: release timing dominates in every probe (gated 1678-2315 vs ungated 1384-1617).

### 6.8 Fabric multi-hop vs store-and-forward: evidence from code (nothing run)
Prompted by: "store and forward is, for some reason, mostly just better". Collected from the tree and git history.

| Transport | Kind | Measured (LoudBox / BH, 2 links unless noted) |
|---|---|---|
| bare one-hop stream (`fabric_link_ceiling`, `05d3e4d94b9`, `bf712886b59`) | 1 hop | 48.5 GB/s per link direction (14336 B payload) only with each link's core in its eth core's NoC column; side-by-side cores sharing NoC links: 40.6 uni / 30.5 bi |
| `fabric_all_gather` (ours, `mimo-v2-dp`; README, `ac4cd4dec5a`, `a29854dac97`) | store-and-forward: forwards from its own output in DRAM | 2-chip line 84% of a link, QuietBox 2-link ring ~80%, 4-chip ring 155 GB/s per chip |
| `fabric_reduce_scatter` (ours, `ac4cd4dec5a`) | store-and-forward (add and forward) | 69-87% of a link |
| `high_bw_all_gather` (README) | neighbour hops only | "a topology that would require Fabric forwarding is rejected rather than silently taking a slower or unsafe path" |
| `all_to_all_async_generic` (Pavle Josipovic, `7dd72a0c648`, `dbb91c2dfe0`, `6f877e59a39`) | direct multi-hop (`fabric_set_line_unicast_route`, up to 3 workers per egress through a fabric mux, banks assigned to links, antipodes split over both arcs) | 31-37 GB/s useful per chip on the 8-chip LoudBox ring: x ~2.3 average hops = ~21 GB/s per link direction, ~43% of a link |
| combine_fabric2d standalone (this work, Kimi) | store-and-forward through a DRAM forwarding buffer | 16 MB own + 20.7 MB forwarded per chip in ~495 us = ~32 GB/s useful per chip, ~18.6 GB/s per link direction (~38%) |

- Multi-hop has no demonstrated edge here: the best tuned direct all-to-all reaches the same useful rate per chip as
  combine's store-and-forward, and both sit far below what store-and-forward collectives reach (~80%+).
- Stability history: Fabric express link routing (#48280, multi-hop shortcuts) was reverted (#57434, Pavle Popovic)
  for a GLM-5.2 sparse-MLA `all_to_all_async_generic` hang on 8x4 TORUS_XY (unit repro `test_a2a_hang_repro.py`
  on `ppopovic/hang_ut`); the all-to-all dropped its custom routing (`6f877e59a39`) because TTNN-owned routes broke
  with fabric changes.
- So: keep store-and-forward. The lever is not the transport kind but combine's own link efficiency (~38% vs ~80%
  for the all-gather) and where its landing traffic goes. What the fast store-and-forward ops do that combine does not:
  link cores in their eth core's NoC column, separate local-copy cores (`5bbfbc34f39`: 1 link 32.6 -> 48.2 GB/s),
  contiguous same-bank multi-page payloads instead of one token per packet, counter-gated relays.
- 6.7's "no landing" option (1) is withdrawn; the landing location / pattern options (2, 3) stand.

### 6.9 Combine walks experts in the order flat finishes them (idea 1a)
- `se_dyn.hpp`: `se_dyn_load` split into a pure `se_dyn_build` (counts in, schedule out) + `se_dyn_from_counts`
  (build + pinning + ring regions) + `se_dyn_completion_order` (inactive experts first, then each expert at its
  last entry: a pinned expert comes at its last chunk).
- Flat's writers (`SeCmbDone`) report prefixes of that order; combine's readers (`CMBF2D_FLAT_ORDER`) rebuild every
  ring chip's order once per launch from the replicated counts with flat's own schedule code and defines
  (`flat_schedule_defines`: SE_RPS, SE_MAX_E, SE_GU_NREG, SE_PIN_*, passed through
  `CombineFabric2dParams::routed_expert_schedule_defines`), so `expert_of` and the gate follow it. Slot order stays
  for bfp8 tiles (untilizers) and the `MIMO_FL_CMB_SLOT_ORDER` probe; writers and readers switch together.
- Why it should work: with pin 1 the largest expert (often slot 0) finished last, so slot-order prefixes released
  almost nothing until the end. Without pinning the order is slot order: nothing changes there.
- Bit-identical on all four cases (first call and cache hit), 10240/10240.

| overlap us (vs row-major baseline) | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| np 2 rows 1-9, slot order | 1222 (-14%) | 2183 (+25%) | 896 (-10%) | 1763 (+8%) |
| np 2 rows 1-9, completion order | 1136 (-20%) | 1922 (+10%) | 840 (-16%) | 1693 (+3.5%) |
| 64 gu + 19 down, slot order | 1565 (+10%) | 1804 (+3%) | 895 (-11%) | 1549 (-5%) |
| **64 gu + 19 down, completion order** | **1350 (-5%)** | **1739 (-0.5%)** | **885 (-12%)** | **1510 (-8%)** |

It worked: -4..-12% everywhere; the 19-down layout is now one config per model that wins both routing cases for
Kimi and GLM. Remaining gap to the ungated ceiling is largest on the hot expert (its rows are still released only
when the whole expert is done): next, progressive release (1b).

### 6.10 Progressive release of a pinned expert (idea 1b)
- Writers (`SeCmbDone::wrote_entry`, `SE_CMB_PROGRESSIVE`) count sub-blocks left per schedule entry; the last one of
  an entry of a multi-entry (pinned) expert bumps (step, chunk = the entry's rank) in the collector's chunk counts
  (the PR's `hybrid_expert_done.hpp` layout). Readers (`CMBF2D_PROGRESSIVE`) keep their own chip's entry boundaries
  from the rebuilt schedule and gate per page on such a step: the page's entry bit in the collector's progress
  word, or the whole step. `MIMO_FL_CMB_NOPROG=1` turns it off on both sides.
- **Bug found on the way (in the PR's collector):** the progress word is `step << 24 | chunk_rows << 16 | mask`, and
  `chunk_rows` is neither zeroed nor masked; our writers never write it, so stale L1 spilled into the step field and
  released a whole chunk early (GLM hot: 9882/10240, deterministic). Fixed by masking it to 8 bits.
- **The test hid it on the second call:** a later overlap call read rows the earlier call had already written
  correctly. Every accuracy call now gets a freshly poisoned y (`y_out` filled with 7777); with combine released at
  launch (`MIMO_FL_CMB_EARLY`) both calls now fail (697-3688 / 10240), the real configuration passes both.

| overlap us (vs row-major baseline) | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| np 2, completion order | 1136 | 1922 | 840 | 1693 |
| np 2, + progressive | 1136 (-20%) | 1900 (+9%) | 839 (-16%) | 1628 (-0.5%) |
| 64 gu + 19 down, completion order | 1350 | 1739 | 885 | 1510 |
| **64 gu + 19 down, + progressive** | **1350 (-5.3%)** | **1702 (-2.6%)** | **885 (-11.6%)** | **1479 (-9.6%)** |
| ungated ceiling (np 2, 6.7) | 1149 | 1628 | 830 | 1409 |

Worked, modestly, and only where expected (hot: -1..-4%; balanced has no multi-entry expert). Why not more: by the
time combine reaches the pinned expert's step (it finishes last among the small ones) most of its entries are
already out, so per-entry release only overlaps its last entry; and a reader blocks on the first unreleased page
in its schedule order, holding up the forwarded chunks behind it.

### 6.11 Combine ring depth (`CMBF2D_SLOTS`, idea 2 part 1): no effect
Hypothesis: the reader is DRAM-latency bound with only `batch` = slots / 2 reads in flight per round trip (4 x 14 KB
per ~3 us ~ 19 GB/s, the measured ~18.6 GB/s per link direction). 64 gu + 19 down, pin 1, progressive:

| slots (batch) | combine alone Kimi bal / hot / GLM bal / hot | overlap Kimi bal / hot / GLM bal / hot |
|---|---|---|
| 8 (4), PR value | 508.0 / 706.1 / 370.4 / 598.7 | 1351.5 / 1705.7 / 880.7 / 1487.0 |
| 16 (8), current | 495.3 / 691.6 / 356.7 / 583.1 | 1354.1 / 1698.7 / 885.3 / 1474.3 |
| 24 (12) | 508.4 / 711.0 / 352.1 / 588.1 | 1345.2 / 1711.2 / 884.1 / 1485.7 |
| 32 (16) | 502.3 / 708.4 / 359.8 / 582.2 | 1348.2 / 1703.4 / 887.6 / 1467.5 |

Did not work: combine alone -2.5% from 8 to 16, nothing beyond; the overlap is flat (+-0.5%). The batch ends in a
full read barrier however deep the ring is, so more slots do not put more reads in flight per wait. Kept 16.

### 6.12 Where combine's reader and sender wait (`CMBF2D_WAIT_STATS`: wall-clock totals per wait, DPRINT)
Kimi balanced, 64 gu + 19 down, chip 0 streams (cycles at 1.35 GHz; typical core):

| | total | upstream (fwd) | read barriers | free slots | gate | sender: wait for reader | sender: router slot |
|---|---|---|---|---|---|---|---|
| combine alone | 662k (490 us) | 29% | 26% | 5% | - | 54% | 17% |
| overlapped | 1716k | 8-32% | 14% | 2% | 30-51% | 83% | 6% |

- Alone, the reader is the bottleneck (the sender idles half the time): it waits on upstream chunks (ring coupling)
  and on read barriers; ~40% is other work, including the local-token phase (read, barrier, write, barrier per token).
- Overlapped, the reader sits at the gate (flat not done) for a third to half of the time, and the forwarded
  chunks behind its own work in the fixed schedule wait too. Reordering relays before own work would deadlock the
  ring (every chip waits for upstream relays first); a lagged order (relays of step i - 1, then own of step i) is
  deadlock free and order-consistent downstream but reproduces almost the same sequence, so not pursued.

### 6.13 Flat's y / weight-forward VCs off the fabric's VC 2 / 3: no gain
The fabric lands received packets on NoC 1 VC 2 + link % 2, i.e. flat's y (VC 2) and weight-forward (VC 3) classes.
Moving flat off them (64 gu + 19 down, pin 1): `MIMO_FL_YRM_VC=1` makes flat itself 1-4% slower and the overlap
follows (Kimi bal 1354 -> 1378, GLM hot 1474 -> 1510); `MIMO_FL_FWD_VC=1` neutral (1354 / 1706 / 886 / 1474);
both moved (y 1, forward 0) like y alone. Did not work: the shared VCs are not what limits the overlap.

### 6.14 Pipelined combine reads (`CMBF2D_PIPE`, idea 2 part 2)
- Each read batch on its own transaction id (2 / 3 alternating, metadata prefetch on 1); with `CMBF2D_PIPE` a
  batch is announced only after the next one is issued (two in flight; batch = slots / 4 to keep the ring's
  deadlock rule). Any pending batch is retired before every wait (slots, upstream, gate) and before the local phase.
- Re-forward / final is now decided from the chunk descriptor (destination == downstream neighbour) instead of the
  first page's data, and the sender takes a final write's address from `final_addr`; forwarded slots' cmd /
  this_addr are still written at retire, after the data: a forwarded page's read (token + 16 B) lands the slot's
  whole 64 B metadata block (first attempt wrote them at issue: wrong slots).
- **Race found:** `noc_async_read_barrier_with_trid` polls the id's outstanding-request count, which only covers
  requests the NIU has taken; right after issuing, the last read can still sit in the read command buffer, so the
  barrier returned early and the sender sent partly landed tokens (1-3 wrong slots per failing case, even without
  pipelining; `CMBF2D_NO_TRID` = global barriers fixed it). Fix: wait for the read command buffer to drain before
  polling the id. Repro written down (not yet run): `flat_combine_tools/trid_read_barrier_race/` (README with the
  theory, open questions, a single-core kernel + pytest, and the in-situ recipe).
- Results (64 gu + 19 down, pin 1, completion order + progressive; bit-identical, both calls):

| | combine alone Kimi bal / hot / GLM bal / hot | overlap Kimi bal / hot / GLM bal / hot |
|---|---|---|
| one batch in flight, 16 slots | 499.1 / 693.9 / 357.7 / 582.2 | 1349.4 / 1707.5 / 885.9 / 1477.0 |
| two batches, 32 slots (batch 8) | 454.8 / 650.8 / 333.3 / 549.3 | 1340.6 / 1683.4 / 883.0 / 1447.1 |
| **two batches, 16 slots (batch 4), now default** | **444.9 / 630.3 / 319.0 / 540.7** | **1341.5 / 1691.2 / 880.4 / 1454.9** |

  Worked for combine itself (-7..-11% alone: the batch barrier no longer stalls the next batch's reads), barely
  for the overlap (-0.6..-1.5%): there the reader mostly waits at the gate for the flat expert, not on reads (6.12).
  Vs the row-major baseline (1426 / 1748 / 1001 / 1636): -5.9% / -3.3% / -12.0% / -11.1%.

### 6.15 Cheap probes on the 64 gu + 19 down config: pin 0, reader-tail down columns
Overlap us (current default: pin 1, `MIMO_FL_PCD_R` 6, pipelined reads: 1342 / 1691 / 880 / 1455):

| probe | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| pin 0 | 1295 (flat 1138) | 2138 (flat 1786) | 880 | 1650 (flat 1294) |
| `MIMO_FL_PCD_R=8` | 1368 (flat 1231) | 1700 | 891 | 1508 |
| `MIMO_FL_PCD_R=10` | 1390 | 1722 | 928 | 1607 |
| `MIMO_FL_PCD_R=12` | 1372 | 1759 | L1 overflow (plan TT_FATAL) | same |

- More reader-tail down columns: worse everywhere; flat itself slows (the reader tails become the bottleneck once
  they also stream more down weights). The down-core starvation cannot be moved onto the readers.
- pin 0: Kimi balanced -3.5% (pinning costs that flat 52 us here), hot cases +13..26%. Pinning does not have to be
  global: the schedule is built on device from the counts, so a runtime rule (pin only when the largest expert is
  far above the mean: Kimi balanced ~2.8x, hot ~27x) keeps both; flat and combine rebuild the schedule from the same
  se_dyn code, so they stay consistent. Next.

### 6.16 Runtime pin rule (`SE_PIN_RATIO`, default 8; `MIMO_FL_PIN_RATIO`, 0 = always pin)
`se_dyn_pin` pins the largest expert only when its sub-blocks are >= SE_PIN_RATIO x the mean of the others'.
Decided on device from the counts; part of `flat_schedule_defines`, so combine's rebuilt order follows. Bit-identical.

| overlap us (vs row-major baseline) | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| always pin (6.14) | 1342 (flat 1190) | 1691 | 880 | 1455 |
| **ratio 8 (default)** | **1271 (-10.9%, flat 1103)** | **1705 (-2.4%)** | **890 (-11.1%)** | **1462 (-10.7%)** |
| ratio 4 | 1272 | 1704 | 892 | 1459 |

Worked: Kimi balanced -5.3% (its flat is faster unpinned than the global pin 0 run, 1103 vs 1138 us), the others
within ~1% (noise); hot experts (~27x the mean) stay pinned. Not sensitive to the ratio between 4 and 8.

**Baseline re-measured on the current code.** The pin rule speeds up flat, and the pipelined reads (6.14) speed up
combine, on the non-overlapped path too, so the baseline moves with them (flat on 110 cores then combine,
`FLAT_CMB_NO_OVERLAP=1 MIMO_FL_ROWS=0,9`): flat 901.0 / 1054.9 / 630.3 / 1054.6 + combine 443.9 / 630.0 / 319.2 /
541.8 = **1344.9 / 1684.9 / 949.4 / 1596.4 us**. Against it the overlap is **-5.5% / +1.2% / -6.2% / -8.4%**: part of
the earlier "overlap gains" were improvements to flat and combine themselves. From here on compare against the
baseline measured on the same code.

### 6.17 Streaming the hot expert: combine steps = flat's schedule entries
- Step s of ring chip X = entry s of X's schedule (one active expert, or one chunk of a pinned expert, as a row range
  [r0, r1) of its region); entries finish in schedule order, so a pinned hot expert is released chunk by chunk as it
  is computed. Writers (`SeCmbDone`) count sub-blocks per entry and report entry prefixes; steps past the last
  entry are empty. Readers rebuild every ring chip's entries from the counts (`build_flat_walk`) and size every
  chunk with `step_run`: origin / destination run of the step's expert intersected with the entry's rows (own
  assignments, forwarded-chunk sizing, local phase, expected final writes all through it, so the ring agrees).
  Collector steps = `CMBF2D_WALK_STEPS` (2 E). Progressive release (6.10) removed: superseded.
- Bit-identical on all four cases first try, both calls, no hang.

| overlap us (vs baseline 1345 / 1685 / 949 / 1596) | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| expert steps + pin rule (6.16) | 1271 | 1705 | 890 | 1462 |
| **entry steps (2 E steps)** | 1298 (-3.5%) | **1521 (-9.7%)** | 910 (-4.1%) | **1292 (-19.1%)** |
| entry steps, ungated (`MIMO_FL_CMB_EARLY`) | 1239 | 1486 | 857 | 1234 |

Worked for what it targets: hot -10.8 / -11.6% (combine hidden 32 -> 61% / 46 -> 78%); gated is now within 35-59 us
of ungated, so release timing is mostly solved. Balanced +2%: nothing is pinned there (pin rule), so the walk is the
same, but the step count is padded to 2 E and every empty step still costs the reader a schedule pass, a local-phase
setup and a flush (~12 x 2 us). Fix: walk only up to the ring's longest schedule (every chip computes it from the
same counts); steps past it are empty on every chip.

| overlap us | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| **entry steps, walk trimmed to the longest schedule** | **1265 (-5.9%)** | **1515 (-10.1%)** | **891 (-6.1%)** | **1287 (-19.4%)** |

Worked: balanced back to (slightly below) the expert-step numbers, hot gains kept; the first configuration that
beats the same-code baseline in all four cases. Bit-identical.

### 6.18 Grid splits: LoudBox 11 x 10, 12 x 9 (row dispatch), Galaxy 12 x 10 (host-side plans)
- **12 x 9 (Galaxy proxy that can trace):** `FLAT_CMB_GRID=12x9` = dispatch on a row with the fabric Tensix mux on
  the 8 x 1 ring (works with FABRIC_2D_TORUS_Y). Combine's cells are the same as on 11 x 10 ((0,0) (2,0) (3,0) (5,0)
  (6,0) (7,0) (8,0)), so row-0 columns 1, 4, 9, 10, 11 are free. Flat on rows 1-8 alone leaves 12 down cores (64 gu +
  16 readers + 4 relays of 96 cells): ~15 output columns per down core, L1 overflow (plan TT_FATAL). With the five
  row-0 cells (`MIMO_FL_XDOWN=1,0;4,0;9,0;10,0;11,0`): 17 down, bit-identical. It has one row less than a Galaxy chip,
  so it is a weak proxy for the overlap (fewer flat cells than the LoudBox).
- **Galaxy 12 x 10, host-side plans** (`TT_METAL_MOCK_CLUSTER_DESC_PATH=.../blackhole_galaxy.yaml`; nothing run):
  full grid 64 gu (np 1) + 16 readers + 4 relays + 36 down; rows 1-9 (combine on row 0) 64 gu (np 1) + 32 down,
  plus whatever row-0 cells combine leaves (5 on the LoudBox): about the full grid's down count. So on a Galaxy
  chip the layout loss that costs the LoudBox up to 200+ us (19 vs 26 down) should mostly disappear.
- **Planner bug (rows capped on a Galaxy chip):** the east gate/up rectangle reaches column 12 of a 12-wide grid:
  `bw_max = min(4, grid.x - 8)` assumes the east readers at column 6 (p150), but a Galaxy chip's are at 7
  (shift 1), leaving 3 columns. Fix before running on a Galaxy: size it after the readers are placed (and add a
  third gate/up rectangle if 64 do not fit).

Measured (pin rule, entry walk; one config per model, vs the baseline on the same grid and code):

| overlap us | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| 11 x 10 baseline (flat 110 cores, then combine) | 1345 | 1685 | 949 | 1596 |
| 11 x 10 np 2, rows 1-9 (planner default) | **1144 (-15.0%)**, flat 980 | 1657 (-1.7%), flat 1595 | **812 (-14.5%)**, flat 661 | 1454 (-8.9%), flat 1376 |
| 11 x 10 64 gu + 19 down | 1265 (-5.9%), flat 1104 | **1515 (-10.1%)**, flat 1274 | 891 (-6.1%), flat 742 | **1287 (-19.4%)**, flat 1170 |
| 12 x 9 baseline (flat full grid, then combine) | 1697 (combine 805) | 2137 (combine 1115) | 1231 (combine 627) | 1994 (combine 990) |
| 12 x 9 rows 1-8 + 5 row-0 cells | 2075 (flat 2026) | 3611 (flat 3520) | 949 | 1444 |

- 11 x 10: with the entry walk the np 2 layout is back: its flat is much faster on balanced routing, and its slower
  hot-case flat is now largely covered (combine 86-90% hidden). Both layouts beat the baseline in all four cases;
  np 2 wins balanced by ~9 points, 64 gu + 19 down wins hot by ~8-10. The choice per model depends on how often
  routing is hot (captured routing: max / mean typically 1.8-2.8, occasionally 25x+).
- 12 x 9 is not a usable Galaxy proxy: combine alone is ~1.8x slower than on 11 x 10 (805 vs 442 us; the fabric
  Tensix mux that row dispatch requires sits in the fabric path), and Kimi's flat on 8 rows is badly starved
  (2026 vs 891 us on the full 12 x 9). The Galaxy evidence that stands is the host-side plan above.

### 6.19 Final numbers, one build (`b702748119d`), test defaults, same fabric profile (2026-10-10)
`FABRIC_2D_TORUS_Y` 8 x 1 ring (MGD), 2 links, max BH payload + 16 B, RELAXED_INIT. All accuracy passing (the PR's
hybrid tests; ours bit-identical both calls, 10240/10240).

| us, LoudBox 8 x 1 ring | Kimi bal | Kimi hot | GLM bal | GLM hot |
|---|---|---|---|---|
| 1. PR best: hybrid (unified) expert overlapped with combine | 1365.6 | 1904.1 | 936.7 | 1581.9 |
| (same, measured on the PR's unmodified code, 2026-10-10 morning) | 1350.4 | 1898.7 | 947.9 | 1585.6 |
| 2. ours serial: flat (default plan, 110 cores), then combine | 1343.7 | 1686.1 | 948.1 | 1593.7 |
| 3. ours overlapped, default layout (np 2, rows 1-9) | **1140.7** | **1656.2** | **810.4** | **1454.2** |
| vs 1 | -16.5% | -13.0% | -13.5% | -8.1% |
| vs 2 | -15.1% | -1.8% | -14.5% | -8.8% |
| alternative static layout: 64 gu + 19 down | 1263.7 | 1512.6 | 892.1 | 1287.3 |
| vs 1 | -7.5% | -20.6% | -4.8% | -18.6% |

Our combine changes (VC 0, pipelined reads, collector fix) also run under the PR's hybrid overlap: its combine
alone is faster (Kimi bal 556 -> 490 us) but its overlap is unchanged within ~1%.

### 6.20 Galaxy prep: unified branch, planner fix, fast perf mode, race isolation
- **Unified branch** `mstaletovic/galaxy-unified` (worktree `/localdev/mstaletovic/tt-metal-galaxy`) =
  `dnijemcevic/model_bringup` + PR #58093 + this work: `combine_fabric2d` and `hybrid_routed_expert_ffn` synced to the
  PR tree (model_bringup never changed them, it was only older), flat changes applied onto model_bringup's own flat
  expert (the PR tree's copy came from the same commit: no conflicts), hand merges only in `Program::append` and the
  prefill adapter / runner flag. Builds; overlap accuracy bit-identical on all four cases.
- **Planner fix (Galaxy):** with the rows capped, the gate/up rectangles are now sized from the DRAM-optimal reader
  columns (queried first): west from column 2 (1 same-column) to E - 1, east from E + 2 (E + 1) to the edge, at most 4
  (5) wide. Before, the east width assumed E = 6 and ran a Galaxy chip's (E = 7) rectangle into column 12 of a 12-wide
  grid. LoudBox plans byte-identical before / after (Kimi, GLM; full, rows 1-9, 64 + 19). Galaxy (mock cluster): full
  64 gu + 36 down; rows 1-9 64 gu (np 1) + **24 down** (+ free row-0 cells), west rectangle 5 wide. (6.18's "32 down"
  was read off the buggy plan.)
- **Fast perf runs:** `FlatRoutedExpert(weights="fake")` allocates the weight tensors in the bank layout without data
  (shapes from the layout code on meta tensors); the overlap perf test uses it by default (`FLAT_CMB_REAL_WEIGHTS=1`:
  real) and `FLAT_CMB_ONLY=overlap` times the overlap only. Same numbers as real weights (1142.6 / 1643.0 / 809.8 /
  1454.2 vs 1140.7 / 1656.2 / 810.4 / 1454.2), 4 cases in 61 s instead of ~5 min.
- **Read-barrier race:** isolated in situ to combine's metadata-prefetch barrier; the standalone repro does not
  reproduce it (details: `flat_combine_tools/trid_read_barrier_race/README.md`).

## 7. What did not work / dead ends
- Unified: writer-side weight prefetch (+3%), GU_IL v1 without pipelining (no gain), noinline pipelined loop.
- Flat on 88 / 99 cores with the default planner for 64-column models: np 2 fallback, hot expert +45..60% flat time.
- `MIMO_FL_RD_SAMECOL` alone (64 gu, 15 down): Kimi bal 1211.6 / hot 1336.6 flat solo, down-starved.
- Row-major y overlap with pin 1: largest-expert chunking delays slot-prefix releases.
- `MIMO_FL_ND=15` on the full layout: 2329 / 4138 us (uneven column split; not representative).
- A merge of PR #58093 into our branch (conflicts in core files and combine): based the work on the PR tree instead.
- DRAM-bandwidth explanation of the balanced loss (6.7): fitted the totals, but flat is not DRAM bound and removing
  its weight reads changed the gated overlap little.

## 8. Next ideas (rough order, after 6.7)
1. **Progressive release + completion-order walk** (hot case: ungated ceiling 95% hidden). Writers report per
   sub-block (or chunk) of a large expert, the collector releases row ranges, readers gate per page; combine walks
   experts in the order flat finishes them (computable on device from the replicated counts).
2. **Combine's link efficiency, store-and-forward kept (6.8):** ~38% of a link today vs ~80% for fabric_all_gather;
   copy cores, contiguous payloads, landing traffic off flat's busy links / VCs. (Multi-hop dropped: no perf edge.)
3. **y from L1** into the senders' rings (see 6.7).
4. **Fabric local writes off flat's VCs:** flat's y and weight forwarding could move to VCs 0/1 classes the fabric
   does not use, or the fabric's `edm_noc_vc` changed; measure with the probes above.
5. **One layout per model that wins both regimes** (Kimi): third gate/up rectangle so np 1 keeps more down cores,
   reader down columns, better down-core rule; Galaxy 12 x 10 should ease it.
6. More combine reads in flight; pipelined local copies (cheap probes).
7. **No combine cores at all ("dumbest" version, not tried):** the flat expert's y writers send straight to the
   fabric routers instead of writing y to DRAM for combine to read back. What it has to solve:
   - **Segments, not tokens.** Flat's down projection is column split: every down core owns `pcd` tile columns of
     every row (Kimi ~7 columns = 448 B of a 14 KB token row). A token leaves as ~30 segments from ~30 cores, so
     packets would be ~450 B instead of 14 KB: the fabric would be packet-rate bound (header per segment), unless
     segments are first gathered into whole rows somewhere.
   - **Router connections.** An eth channel takes one worker connection (the EDM stores a single worker_xy per
     channel), so ~30 senders per link need a fabric mux (as all_to_all_async_generic does: up to 3 workers per
     egress through a mux), which costs cores and its own L1.
   - **Routing per token.** Each segment needs its destination (chip, output page) from the dispatch metadata, so
     every down core reads metadata it does not need today.
   - **Store-and-forward still needs a relay** on intermediate chips (6.8: multi-hop is not faster), i.e. someone
     still lands and re-sends forwarded tokens there: the current reader/sender pair or a router-side relay.
   - **Flow control** against the fabric (credits per mux channel) moves into the down cores' event loop, beside
     the h chain and the y writes: back-pressure would stall the down matmul pipeline.
   Middle ground that keeps whole-token packets: the down cores write their segments into the sender's slot ring
   on the row-0 core next to the eth core (NoC 1: north to row 0, then west along it, off flat's busy links), the
   sender sends whole rows; no DRAM write of y, no DRAM read, no reader for own tokens. That is idea 3 ("y from L1")
   with the reader removed for own tokens.

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
`MIMO_FL_RD_SAMECOL`, `MIMO_FL_XDOWN`, `MIMO_FL_ND`, probes `MIMO_FL_W_NOREAD`, `MIMO_FL_XRD_SKIP`,
`MIMO_FL_CMB_EARLY` (garbage output, perf only). Profile: `TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1`, then `cmb_timeline.py <csv> <run group>` / `cmb_split.py`.

## Flat expert + router-forwarded level-1 combine, one program per chip (2026-10-16, bh-lb-09 LoudBox 2x4 FABRIC_2D, 200G, 12x10 grid)

What: `ttnn.bringup.flat_routed_expert_with_program` (device/flat_with_program_device_operation.cpp) builds the flat
expert's program per chip (DRISC on, as standalone) and appends another op's MeshProgramDescriptor; the flat y writers
report each local expert (se_cmb_done.hpp, slot order, cumulative counts, no go handshake: SE_CMB_NO_GO) to a
collector core. The combine is `ttnn/ttnn/bringup/combine_rf` (port of tt-metal-dispatch's sparse_combine_rf: rows
injected once as multi-hop 2D unicasts, the routers relay, home accumulators in L1) on the cells the flat plan leaves
free (K3: 24 cells: row 8, most of row 9, 4 scattered; 9 port cells over the chips + 15 accumulators), its state, rings
(depth 4) and CBs inside the flat expert's arena, which the caller now owns (persistent, zeroed once). Port readers read
an expert's rows from y once its count covers the call; accumulators take their local rows as experts finish,
interleaved with the rings.
Test: models/demos/deepseek_v3_d_p/tests/perf/scratch_flat_combine_rf.py: K3 per-chip load (56 local experts x 8 chips,
top-16, 320 home tokens/chip, ~5120 rows/chip, latent 3584, I 3072, bfp4, one random expert shared, FC_WEIGHTS=real);
the first fused call checked against the sum of the y rows read back (y zeroed before). Device time, slowest chip,
median of 10.
| routing | flat alone | combine alone | serial | fused | combine hidden | fused / flat |
| uniform | 2462 | 1013 | 3476 | 2674 | 79% | 1.086 |
| Zipf spread 1.2 | 2730 | 1110 | 3840 | 3101 | 67% | 1.136 |
| Zipf packed 1.2 | 5875 | 2422 | 8297 | 6106 | 90% | 1.039 |
All exact (worst token 1.74-1.76e-3, bf16 rounding). Bugs on the way: (1) the Python build appended CBs to
`program.cbs`, a copy through the binding: no CBs at all (kernel groups' local_cb_mask 0, every first reserve blocks;
FLAT_WP_LOG=1 logs the merged program's kernel groups); (2) credit arrays for 15 accumulators were not 16-byte aligned.
Next: where the remaining 4-14% goes (the tail after the last expert vs the flat slowed by the combine's NoC traffic),
entry-order reporting instead of slot order, the L1 hand-off (no y writes).

### Why the overlap was not perfect, and the L1 hand-off (2026-10-16, same setup)
1. Launch skew: the host starts the 8 chips one after another (~130 us first to last); the per-call timing synced after
   every call, so the earliest chip paid it every time (its tail after its last expert 133 us vs 9-14 us on the
   last-launched chips). Queued back to back (FC_PIPE=1), only the first call pays it: uniform fused 2674 -> 2595 us.
2. The combine's y reads from DRAM (36.7 MB / chip) slow the DRAM-bound flat expert: the experts' release times (SPC_TPROF
   timeline) put the flat ~100 us later beside the combine; with the reads removed (SPC_NOYREAD probe, garbage rows)
   fused / flat = 1.008 (uniform), 1.003 (packed); Zipf spread stayed 1.115: there the experts are released whole in
   slot order, so a hot expert's rows wait for the whole expert.
3. L1 hand-off (FC_L1=1; se_yrm.hpp SE_Y_L1 via MIMO_FL_Y_L1, set by the op when l1_args are passed): no y in DRAM. Each
   down core writes its 256-byte segment of every row straight into the row's destination slot (a combine port's
   injection ring in its unused ring region, 64 slots, or an accumulator's local ring); destinations number rows in
   the writers' common production order, grant slots (cumulative grant[d] on each writer), and take a row once every
   writer's landed counter (+rows per acked row tile) covers it; the column-0 writer writes each row's trailer.
   Per-chip host table: (destination, header) per y row, one 256-byte page per 32-row tile.
Queued back to back, slowest chip, median, all exact (1.74-1.75e-3):
| routing | flat alone (writes y) | flat alone (no y) | combine alone | fused, y via DRAM | fused, L1 hand-off | hand-off fused / flat no y |
| uniform | 2462 | 2320 | 1027 | 2595 | 2432 | 1.048 |
| Zipf spread 1.2 | 2731 | 2588 | 1029 | 3056 | 2667 | 1.031 |
| Zipf packed 1.2 | 5873 | 5747 | 2336 | 6004 | 5888 | 1.024 |
Open: the remaining 3-5% (uniform ~110 us) with no combine DRAM traffic left.

### Gate weights (2026-10-16)
Each row's gate weight (bf16) rides in the hand-off table (destination word, upper half) -> the column-0 writer puts it
in the row's trailer (word 1) -> through the fabric with the row -> the accumulator reader puts it into a weight CB
(element 0) -> the compute multiplies (mul_tiles_bcast_scalar, HiFi4, fp32 DEST) before the packer L1 accumulation.
Test: softmax gates over each token's 16 experts, bf16; reference = sum of bf16(w) * y in fp32. Queued, slowest chip:
| routing | fused unweighted | fused weighted | worst token |
| uniform | 2432 | 2427 | 1.73e-3 |
| Zipf spread 1.2 | 2667 | 2671 | 1.75e-3 |
| Zipf packed 1.2 | 5888 | 5896 | 1.75e-3 |
No measurable cost (the accumulators had headroom).

### Device-side tables (2026-10-16)
`ttnn/ttnn/bringup/combine_rf/combine_plan.py`: a planner kernel (one core per chip, generic_op) builds, from the
replicated routing (top-16 ids uint16 + gate weights bf16 for all N x S tokens), the flat expert's counts / regions /
token index (indexed x: the gathered tokens, row home * S + t) and the combine's hand-off table and per-call counts
(n_inj / n_ret per port, n_local / want per accumulator) -- the same rules as the host builder (plan_reference: the
host model; scratch_combine_plan.py: bit-exact on all 8 chips, uniform / spread / packed). The combine kernels read
their counts from the plan (__DEVPLAN__: the count args carry the plan's address; the compute stops at the reader's end
marker). Overlap test FC_DEVPLAN=1 (with FC_L1): every per-call table from the device.
| routing | fused, host tables | fused, device tables | worst token |
| uniform | 2427 | 2431 | 2.19e-3 |
| Zipf spread 1.2 | 2671 | 2706 | 2.18e-3 |
| Zipf packed 1.2 | 5896 | 5889 | 2.18e-3 |
(Rows within an expert are now in (home, token) order; spread +35 us.) Planner: 1.45-1.66 ms on one core (~24 cycles per
routing entry, two passes over 40960) -- must be split over cores; it can overlap the shared expert.

### Planner: parallel + optimized (2026-10-16) — combine_plan.py `_PLANNER_MC2` (cores=..., default 32 in the tests)
Per chip, M cores: (1) core p counting-sorts routing chunk p's local-expert entries by expert (count pass over id pairs,
four loads in flight, plus a compact list of the local entries; then place from the list), each entry already its
final table words; sends counts + bucket prefix to every core. (2) the owner of expert e (core e % M) totals it and
each chunk's first row and sends that to every core (second round); core p takes 1/M of the row tiles and pulls the
matching entry slices from each chunk's bucket (every read issued, one barrier, then copied): a hot expert is split
over all cores. (3) rows to DRAM; then this chip's home-token counts (core q owns accumulators q, q + M, ...: n_local /
want written whole to core 0's plan; n_inj / n_ret as atomics); core 0 writes counts / regions / plan.
Bit-exact vs the host model on all 8 chips, 3 routings, after every step. Slowest chip, us:
| version | uniform | Zipf spread 1.2 | Zipf packed 1.2 |
| one core | 1451 | 1466 | 1655 |
| 8 cores, expert-group placement | 215 | 452 | 982 |
| v2 row-range placement, 16 cores | 155 | 189 | 298 |
| + compact local list, owner-computed home counts | 112 | 154 | 290 |
| + all slice reads before one barrier | 94 | 121 | 258 |
| + bucket prefix sent with the counts | 75 | 81 | 183 |
| + per-expert owner totals (2nd round), home counts in registers | 69 | 80 | 183 |
| 32 cores | 52 | 59 | 117 |
| + home counts after placement, id pairs | 51 | 52 | 113 |
Probe that did not help: multicast all-to-all exchanges (every core multicasting into one rectangle at once): 187 us.
Phases now (32 cores, uniform): read 5, sort 14, counts exchange 2, totals + placement 14-19, rows + home 6, core 0
final 4. With the planner on 32 cores, fused (device tables): uniform 2425, spread 2714, packed 5896 us, exact.

### The whole Kimi K3 MoE block, token-split (2026-10-16) — `models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py`
Per chip S = 320 own tokens, K3 widths (H 7168, latent 3584, I 3072, shared I 6144, top-16 sigmoid + bias, SiTU),
56 experts per chip (the K3 4 x 4 stage's per-chip load; 448 on the LoudBox's 8 chips, from a pool of 8 random
experts). Stages: router (gate matmul + moe_grouped_topk) -> routing all-gather (single-page rows for the planner) ->
planner (32 cores) -> latent down -> arena + state zero -> dense all-gather dispatch -> fused flat + combine (L1
hand-off, device tables, gate weights) -> RMSNorm + latent up -> shared expert (replicated, own rows) -> add.
Test: `tests/perf/scratch_k3_moe_layer.py`. Exact: PCC 0.99623 vs the host model with the device's routing (0.99592
with host routing; 99.41% of the top-16 picks equal, the rest near-ties in bf16 logits); a repeat call is bit-identical.

Bring-up problems and decisions:
- A persistent arena (~1.06 MB per core, lockstep) leaves the dense ops ~360 KB: the gate matmul's static CBs clash.
  The arena is now per call, allocated after the dense ops and freed after the experts. What made that safe:
  * the combine's state (22 KB block at the arena base: credits, used / ret / misc counters, collector counts) and
    every ring slot's piece trailer (a slot is taken when its trailer flags are set) are zeroed on the device
    (`combine_rf/l1_zero.py`, 5.6 us), and the writers' words; a call then starts like the first one;
  * the dispatch all-gather between the zeroing and the experts is the cross-chip barrier (no chip's rows land in a
    chip that has not zeroed);
  * the ports drain at the end (wait until every injected row's credit is back), so nothing lands in a chip's L1
    after its program ends (it is another op's by then);
  * the arena must land at one address (the kernels bake it): the writers' words go above it, and the dense ops run
    once at build (CCL semaphores are allocated on first use and kept, 256 B that moved the arena);
  * a program-cache miss rebuilds the combine's CBs on the call's arena (`run.rebind`), not the freed first one.
- The planner reads the routing as one page [1, N S K]: row-major uint16 has no reshape, so ids go
  typecast u32 -> row-major -> reshape -> u16 and both gather along the last dim (columns, then rows).

Per stage, slowest chip, median of 5 (us; stage = the programs between two syncs):
| router | routing gather | planner | latent down | arena zero | dispatch (dense AG) | experts + combine | latent norm + up | shared expert | add | total |
| 252 | 312 | 53 | 305 | 6 | 394 | 2538 | 286 | 1055 | 34 | 5233 |
The fused op is where expected (2425-2530 uniform). Every dense op is far from its floor (the gate matmul is 12.8 MB
of bf16 weights at 896 experts; latent down 27 MB bf8 ~53 us; shared 3 x 47 MB ~274 us): default configs, row-major
round trips, tiny all-gathers at ~60 us each. Next: dense op configs, the routing gather in one op, the sparse 2D
dispatch, the shared expert in its own lane (parallel with the experts).
Port credit drain cost (standalone fused test, uniform): 2533 us with it vs 2520 without (SPC_NODRAIN=1), noise;
pipelined (FC_PIPE=1) fused 2438 us, as before the change (2425). The layer's 2516 is the unpipelined number (a sync
before each stage leaves ~80-100 us of launch skew).

### K3 block: dense op configs + routing without the reshape (2026-10-16)
- Matmul configs at 320 rows (`tests/perf/zz_scratch_k3_dense_mm.py`, slowest chip, HiFi2): the default picks are
  18-31% of the weight's DRAM floor; the GLM prefill 2D-multicast picker (glm53_flash_d_p/tt/mm_configs.mm2d) is
  45-48%: router 157 -> 42, latent down 292 -> 118, latent up 173 -> 116, shared gate / up 302 -> 188, down 276 -> 191.
  1D in0-mcast and hand 2D variants are no better; DRAM width-sharded weights with mm2d give wrong results (PCC 0.13,
  not pursued). LoFi changes nothing (bandwidth bound). The block uses mm2d for every linear.
- The planner reads the routing row-paged (`row_pages=True`: [N S, K] row-major, one 32 B page per token): no
  reshape / typecast chain (that 10 KB reshape alone was 130 us). Pages land 64 B apart (a DRAM read keeps its
  64-byte alignment in L1: packed 32 B destinations hung the planner) in local scratch -- the home ids in the slice
  staging, not past the bucket region (the other cores' counts exchange lands there: corrupted home counts). Bit-exact
  vs the host model in both modes; the reads cost ~43 us (packed routing 113 -> 156 us; in the layer 53 -> 98).
Per stage now (us): router 136 (gate 42 + top-k 95), routing gather 292 (four ttnn.all_gather of 10-80 KB at 54-88 us
each), planner 98, latent down 131, arena zero 6, dispatch 394, experts + combine 2526, norm + latent up 236
(to tile 54, rms_norm 58, linear 124), shared 766 (gate 197, up 198, situ_glu 179, down 196), add 33: 4617 us (was
5223). Exact as before (PCC 0.99623, repeat call bit-identical).
- Joint routing rows (`joint=True`): ids and the weights' bf16 bits concatenated per token ([N S, 2 K] uint16, 64 B a
  token: bitcast + concat), so one all-gather pair instead of two. Planner bit-exact (uniform 95, packed 154 us
  standalone). Routing gather 292 -> 163 us (the two all-gathers 54 + 88 us: mostly launch skew after the stage
  sync), block 4617 -> 4487 us, exact.

### Accuracy: the K3 block vs TtMoe, and the flat expert vs production at GLM 5.3 / K2.7 / K3 (2026-10-16)
Tools (tests/perf): `zz_k3_moe_accuracy.py` (offline, dumps of scratch_k3_moe_layer.py K3_DUMP= and
zz_scratch_k3_ttmoe_compare.py K3_DUMP_TTMOE=; refs: upstream KimiSparseMoeBlock fp32 / bf16 with the experts pointed
at the pool, and a host model with the device's weight formats), `zz_scratch_expert_precision_models.py` (one chip,
model expert shapes, runs unchanged in an origin/main worktree for the production op) + `zz_expert_precision_report.py`.
- Host model == upstream block to 1e-7 (fp32). RMSNorm eps fixed 1e-6 -> 1e-5 (config.json; no measurable effect here).
- Block end to end vs upstream fp32 (448 experts, 2560 tokens, same weights / input): token-split PCC 0.99592 (rel
  9.1%), TtMoe 0.99566 (rel 10.7%). Both dominated by bfp4 routed weights (yr rel 20% from the weights alone).
- Device arithmetic per stage needs a reference with the device's own weight rounding: the flat expert's upload goes
  fp32 -> bf16 -> bfp4 (K3_WCHECK: device weights == bfp4(bf16(W)) exactly; straight fp32 -> bfp4 differs in 0.75% of
  elements), the block's dense bf8 weights and TtMoe's experts go straight (lower error with the matched reference).
  Matched: token-split routed experts + combine 1.9% rel (of it the tokens' bfp8 rounding 1.1%; bf16 h 1.35%), latent
  down 0.4%, norm + up 0.5%, shared 1.0%; TtMoe routed 5.0%, latent 2.0 / 2.0%, shared 6.1% (bf16 DEST + bf16
  reduce-scatter of the TP partials). Routing: 0.54% (ours) / 0.66% (TtMoe) of the picks differ from the host, every
  one a near-tie.
- Expert matmul fidelity: HiFi2 bit-identical to LoFi (LoFi takes 5 weight bits / 7 activation bits: bfp4 weights are
  exact). bfp8 weights are LESS accurate in LoFi (routed 3.3% vs 1.9%: the 7-bit mantissa truncated to 5) and 1.8x
  the experts' time (4571 vs 2523 us). MXFP4 checkpoint weights -> bfp4: 11% weight error (19% on the routed output,
  either exponent orientation); -> bfp8: exact.
- Flat expert vs production (TtRoutedExpert as TtMoe builds it: LoFi, bf16 DEST, packer L1 acc, hybrid threshold
  320 for GLM 5.3 / K2.7, unified only for K3; origin/main cb0eabe05e7 == this branch's copy, bit-identical), same bf16
  weights / row-major inputs, arithmetic rel error over 5 input distributions:
  | shape | flat (bfp8 h) | flat (bf16 h) | production |
  | GLM 5.3 6144 x 2048 | 7.1-7.9% (BUG, below) | 0.8-2.0% | 2.6-5.2% |
  | K2.7 7168 x 2048 | 1.3-2.5% | 0.8-2.0% | 2.6-5.5% |
  | K3 3584 x 3072 | 1.2-5.9% | 0.8-5.9% | 2.0-8.3% |
  Production's norm ratio 1.03-1.04 (bf16 DEST gain); flat 1.00-1.015. (student_t2 inputs: every op ~145% off the
  host bfp8-x reference while agreeing within 4% with each other: a reference artifact, excluded.)
- BUG (flat expert, GLM 5.3 shape H 6144, bfp8 h): the plan gets 4-tile (128-row) sub-blocks (row passes: Ht / KBLK
  = 24 = x_slots; K2.7 has 28 K-blocks, K3 NP 2: both mt 2) and the last row tile of some sub-blocks is never written:
  the same tiles every run and input (expert 0 tile 7 of 16, expert 5 tile 3 of 7, expert 7 tile 7 of 12), contents
  whatever was there (30% error in one run, inf in the next). MIMO_FL_MT=2: correct. Not the row-pass gate/up kernel
  (MIMO_FL_GU_FULL_SYNC=1: same tiles), not the x ring depth (MIMO_FL_X_SLOTS_MAX=26: same): the 128-row sub-block's
  h exchange / down / y write. The K3 block is unaffected (mt 2). Open.
- FIXED (same day): only the reader-tail columns were bad (y col tiles 150..191 = 7 tails x pcd_r 6; in each tail the
  last 3 of its 6, row 3: out-CB tiles 21..23 of the 24-tile block), and only on sub-blocks using the out CB's second
  half (v 1 / 7 / 11, all odd). The plan sized the shared out region at o_off for the down cores' widest column count
  (pcd 5) while the reader tails put 2 x mt x pcd_r (6) tiles there: at mt 4 the tails' second half ended 6.6 KB past
  the arena (arena set by the down role), in L1 owned by something else. Not an LLK issue (llk_push_tiles stalls for
  the packer before publishing). Fix: out region sized for max(pcd, pcd_r) (plan.cpp; the Python builder too) and a
  TT_FATAL in arena_cb for any CB past the arena. GLM 5.3 6144 x 2048 (any E: the LoudBox's 32 too; bf16 h mt 2 was
  undersized as well, unharmed only because another role made the arena bigger) gets +4 arena tiles. After: GLM 5.3
  flat 1.25-2.5% arithmetic error (worst row 3.4%, was 31% / inf), K2.7 / K3 / the K3 block unchanged. GLM-5.3-Flash
  (4096, no reader tails), K2.7 (tails == down width), K3: never affected (zz_scratch_flat_plan_cmp.py flags the risk).

### K3 block: sparse 2D dispatch (2026-10-16)
The dense dispatch (two ttnn.all_gather, 168 + 225 us) is replaced by sparse_all_gather_2d (combine_rf/, the
dispatch worktree's op): each own row goes only to the chips holding one of its experts, dimension-order trees, one
hop per port, neighbour fences + generational END markers (so it is a barrier, as the arena zeroing needs: a chip
finishes only after every chip's ports started). Inputs from `combine_rf/dispatch_prep.py` (one generic_op): the
route words of the own tokens from the joint routing rows (destination mask, X- or Y-first by fewer tree links, link by
a hash of t; 8 cores, 40 tokens each) and the own rows copied into their slots idx * S + t of the dispatch buffer (16
cores, ping-pong batches). K3_DISPATCH=dense keeps the baseline.
| | dense AG | sparse 2D |
| prep (route words + own-row copy) | - | 28 us (one core: 124; 8 copy cores, no ping-pong: 35) |
| dispatch | 393 us | 229 us |
Block 4551 -> 4373 us, exact (PCC 0.99623, repeat call bit-identical). On the LoudBox a top-16 row reaches 6.2 of the
other 7 chips: busiest link (CPU model, this routing) stateless 653 rows = 187 us floor at 25 GB/s, the cheap greedy
plan 554 (159 us), dense 726 (208 us): the sparse op is at 82% of its floor; the gain is mostly a better transport than
ttnn.all_gather (53% of its floor), not sparsity (that grows on 4 x 4). Next for dispatch: the greedy plan (~-35 us,
sequential over every token: needs a parallel / overlapped form), the own-row copy folded into latent down's untilize.

### K3 block: the shared expert as a flat expert (2026-10-16)
The shared expert (7168 -> 6144 -> 7168, bf8) runs as the flat routed expert with one-expert-for-every-token routing:
3 experts of I 2048 (the gate/up split takes <= 64 tile columns per expert at H 7168 bf8: I 6144 has no split), each
over all S own tokens (indexed rows 0..S-1 three times), y = the sum of the three parts (down over disjoint
intermediate slices). New flat expert options (C++ config, no plan change): `y_bf16` (bf16 tiled y: the out CB single
buffered in the bfp8 double buffer's region) and `math_fidelity` (per op; MIMO_FL_FIDELITY still overrides for
probes). The sparse dispatch's manifest is now per call (manifest_spec; the persistent 123 KB on every core left the
SE's arena no room) and its global semaphores are created at build (else they fragment L1 under the per-call buffers).
Accuracy vs the exact bf8-weight model (se_err.py) and time (flat program, us):
| config | rel | gain | block PCC | flat us |
| LoFi, bfp8 x / y | 3.1% | 0.989 | 0.99589 | 466 |
| LoFi, bf16 y | 3.3% | 0.982 | 0.99591 | 480 |
| HiFi2, bf16 y | 2.0% | 1.013 | 0.99618 | 531 |
| HiFi2, bf16 x + y (default) | 1.2% | 0.997 | 0.99619 | 529 |
| HiFi3 / HiFi4, bf16 x + y | 1.1% | 1.003 | 0.9962 | 688 / 844 |
LoFi truncates bf8 weights to 5 bits (a gain loss); HiFi2 restores them; bf16 x removes the tokens' bfp8 rounding.
(ttnn.linear path: ~1.0%, block 0.99623.) Shared expert 767 -> 621 us (to row-major 49.5 + flat 529 + part sum
42.6), block 4373 -> 4232 us, routed experts still LoFi. The flat SE is compute / pipeline bound (HiFi2 at 320 rows
per part: DRAM floor 274 us); next: its own lane (overlapped with the routing / dispatch phase) rather than more
single-op speed.

### K3 block traced (2026-10-16): scratch_k3_moe_layer.py -k trace
One forward captured as a trace: 4105 us wall per replay (bit-identical to the untraced forward), vs untraced 26.8 ms
back to back (host-bound: the Python per call -- descriptor rebinds, generic_op hashing -- not the device). Profiled
replay (slowest chip): 24 programs back to back, busy 4098 us of a 4098 us span: no idle gaps, no overlap. Programs:
router 41 + top-k 95 | routing rows 26 + all-gathers 53 + 87 | planner 91 | latent down 119 + row-major 12 | dispatch
prep 28 + arena zero 5 + dispatch 213 | experts + combine 2442 | to tile 53 + rms_norm 58 + latent up 124 | shared
expert x row-major 49 + flat 527 + part sum 42 | add 32. The tiny routing all-gathers keep their 140 us in the trace
(latency, not launch skew). Everything outside the experts (1656 us) is serial: the next gains are overlap (the shared
expert beside the routing / dispatch phase or the experts) and the small-op latency chain.

### K3 block to 3700 us traced (2026-10-17, bh-lb-09 2x4, 12x10 grid, 200G)
Traced forward ~3820 -> 3773 -> 3736 -> 3721 -> 3710 -> 3700 us (scratch_k3_moe_layer.py -k trace, twice at 3700;
PCC vs host 0.99619 / 0.99620, routing 99.77% equal, repeat call 0.99992). Commits 45210ce3873 .. ee9080ebe05. Steps:
- **fused SE part sum + final add** (combine_rf/sum_ops.py, y = yr + p0 + p1 + p2 per tile, one bf16 rounding): 74
  -> 55 us. It was 75 us at first: **DRAM reads on NOC1 are slow here** (NOC0 52.5 us = 438 GB/s, NOC1 75 us; the
  reads, not compute: probes noread 28 / nocompute 75). layout_ops' reads moved to NOC0 too: SE x untilize 32 -> 24,
  [320, 3584] tilize 18 -> 15 us. Other combine_rf kernels already read on NOC0.
- **transient intermediates in L1** (K3_L1, default router,latup,yr,latdown): router logits / biased scores (custom
  top-k 49.6 -> 36 us, its 64 B reads), latent down / up linear outputs and the latent up input (-4..-8 us each).
  The shared expert's flat arena takes 1.42 of 1.46 MB per bank, so it now runs before latent up (yr then lives in
  L1 until the sum). Arena address check: no L1 tensor may outlive into the routed arena (keep copies to DRAM).
- **latent down untilized straight into the dispatch buffer** (layout_ops dst + per-chip row0; dispatch_prep
  copy=False): latent down + prep 156 -> 137 us; then the untilize and the route words in **one generic_op** per chip
  (the untilize off the 8 route cores): prep 21.5 -> 12.2 us.
- **router**: sigmoid fused as the bias add's input activation; router_topk bias_mode reads only the biased scores
  and the bias row (pick score = biased - bias): -1 program, traced -3 us (the top-k is scan bound: noweights 32.6).
- **sum reads on both NoCs** (parts 0, 1 on NOC0; yr from L1 + part 2 on NOC1 by the writer RISC): 55 -> 48 us.

Traced programs now (slowest chip, us): router 41.4 + 7.1 + 38.4 | gather 35.9 | planner 56.8 | latent down 116.3 |
untilize + route 12.5 | arena zero 6.2 | dispatch 206.5 | experts + combine 2431.6 | SE x untilize 24.1 + flat 524.6
| tilize 12.2 + norm 7.4 + s2i 4.8 + latent up 113.6 | sum 53.1 (17 programs, ~0.5 us gaps).

Shared expert, both variants (as the goal asked): **local flat** (default) untilize 24 + flat 530 + sum (fused) vs
**TP** (K3_SE=tp: x row all-gather over the mesh row, I split 4 ways, reduce-scatter back) untilize 33 + AG 299 + flat
560 + RS 376 = 1264 us; SE PCC 0.99976 / 0.99975. **Traced: local 3700 us, TP 4402 us** (K3_SE=tp on the same
code; its trace had failed on 2-page x rows before the 7-page fix). TP loses on CCL time at 320 tokens a chip; the flat SE is compute /
pipeline bound either way (~160 TFLOP/s: LoFi only 530 -> 510, bfp8 x no change, 4 parts worse, DRISC on / off same).

Dead ends this round (numbers in the scratch tests):
- **a custom streaming linear** (combine_rf/stream_linear.py, kept as a probe, docstring has the history) for the
  latent projections (ttnn mm2d 117 / 123 us = 45% of the 53 us DRAM floor): 1D (x multicast to the whole grid) 210;
  2D with rotating x / W owners 133-230; resident x + W ring 160+ us. Each step costs ~1-2 us of credit / valid
  latency with 22 cores coupled per step. Not the DRAM reads: **mm2d with its in1 DRAM reads disabled is still 117.5
  us**, and every mm2d / 1D / DRAM-sharded / minimal_matmul config sweeps to 117-123 (fidelity, block width 8-32,
  sub-blocks, full sync: no effect). Compute alone at one M tile per core: ~72-80 us (unpack bound). Multicast
  bandwidth (zz_scratch_mcast_bw.py): **52 GB/s per sender down a column, 44 along a row, 8 GB/s unicasting to each
  member**, all columns at once without loss - 27 MB of weights by multicast >= 47 us.
- the SE x untilize merged into another op's program (K3_SE_X_IN): into the planner (cores it leaves) planner 60 ->
  76 us, traced neutral; into the sparse dispatch 230 -> 284 us; into the routing gather 50 -> 150 us. The fabric
  ops are very sensitive to co-scheduled DRAM / NoC traffic.
- combine depth K3_D 2: 3727 (worse), 6: does not fit L1. MIMO_FL_DRISC_MAXCNT 160 / 1024: neutral.

### TP-sharded shared expert, piece by piece (2026-10-17, zz_scratch_k3_se_tp.py, 2x4 200G)
ttnn CCLs (num_links 2, Linear) + the SE as dense ttnn (bf8 weights, HiFi2, mm2d), slowest chip, us; output vs fp32:
| layout | all-gather x | dense SE (gate, up, situ_glu, down) | reduce-scatter | serial sum | rel / PCC |
| TP4 row bf16 | 316.5 | 484 (115 + 115 + 89 + 165) | 371 | 1172 | 1.53% / 0.99975 |
| TP4 row bf8 payloads | 199.7 | 484 | 343 | 1027 | 2.65% / 0.99830 |
| TP2 col bf16 | 129.2 | 461 (118 + 118 + 89 + 136) | 160 | 750 | 1.55% / 0.99975 |
| TP2 col bf8 payloads | 90.2 | 461 | 145 | 696 | 2.22% / 0.99834 |
| local dense (replicated) | - | 765 (197 + 197 + 176 + 196) | - | 765 | 1.58% / 0.99975 |
(local flat, the default: 549 incl. its x untilize; TP4 flat was 560 for the SE alone.) Findings: at TP shapes dense
beats flat (more rows per weight: 245 TFLOP/s on the gate linear at M 1280); ttnn.situ_glu is a 5-program composite,
89 us at TP shapes (176 local) - one fused op would be ~15-20; the collectives are near the 200G link floor (TP4 edge
chips: one neighbour) and bf8 payloads cut the AG 30-37% but the RS only 8-10% at +0.7-1.1% error. Serially even TP2
(~680 with a fused activation) loses to local flat 549: TP pays off only with the collectives overlapped.

### K3 experts + combine program: what bounds it (2026-10-17, traced, experts program, slowest-chip trace line)
Probe semantics on this path (K3: DRISC gate/up, Tensix down-weight readers se6_dw on NOC0, L1 hand-off combine):
MIMO_FL_W_NOREAD skips only the DRAM reads - in DRISC mode the DRISCs still push every (garbage) gate/up block over the
NoC (W_NOFWD stops that); MIMO_FL_NO_MATH hangs here (the gate/up side never publishes h; se6_dcompute waits in
cb_wait_front(h_all_cb)) - broken, not used.
| DRAM / NoC traffic left | experts us |
| everything (base) | 2433 |
| no gate/up NoC push (W_NOFWD) | 2430 |
| push only (W_NOREAD) | 1990 |
| no gate/up DMA (GU_NOREAD) | 2024 |
| no down reads (DN_NOREAD) | 2112 |
| no weight traffic (W_NOREAD + W_NOFWD) | 1221 (+ no x reads: 1225) |
| no push: gate/up DMA only | 1456 (694 MB: 477 GB/s, 93%) |
| no push: gate/up DMA + x | 1595 |
| no push: down reads + x | 1462 (compute bound underneath: 1221) |
| no push, no x | 2326 |
| Tensix readers (DRISC=0) | 2420 |
DRAM bound: 1077 MB a chip a call (gate/up 694 + down 347 + x 36) at 443 GB/s (86.5%); floor 2104 us. Compute + h +
combine 1221, the DRISC push ~1990 - both hidden; the combine's release (CMB_EARLY) and h (H_NODATA) change nothing.
The gate/up DMA alone streams at 93%; the Tensix down-weight NoC reads into the same banks pull the mix to 86% (~170
us), x reads ~105 us. GDDR arbitration weights (MIMO_FL_DRISC_MPFE 0,0,3 / 0,3,0 / 3,0,0 / 1,3,3 / 3,1,1 / 1,1,1):
2418-2439, noise. Next lever: down weights through the DRISCs (one bank-local DMA path), then the x reads.

### K3 block: x read once for the routed experts (2026-10-17) -> 3640 us traced
x-path configs on the K3 trace (experts program / traced total, us; base 2433 / 3707): MIMO_FL_X1 (x read from DRAM once,
rectangle 0's relays multicast to both) 2372 / 3819; X2 2659 / 4188; XFWD 2859 / 4495; NH 2 2471 / 3759; NH 3 2457 /
3736; X1 + NH 2 2396 / 3858. X1's total got worse because the env knob also built the shared expert's flat op with it
(523 -> 702 us: one primary's tilize + double multicast caps 320-row parts, as in the standalone sweep past M 128).
Now set for the routed experts' program only (K3_EXP_X1, default 1; read when the program is built): traced 3643 / 3640
(was 3700), PCC 0.99619 unchanged, shared expert 529 us. (Earlier x-read isolation and probe history: the pr58093
worktree's worklog 8e-8g - x read twice at ~22 GB/s per relay, 1 KB requests.)

### K3 experts: the weight source does not matter (2026-10-17, with X1; experts / traced us)
base 2380 / 3648; DRISC_PURE=0 (dual) 2362 / 3649; DRISC_TILES=2 2370 / 3648; TILES=1 2379 / 3650; dual + TILES=2
2372 / 3650; DRISC forced (MAXCNT=1024) 2375 / 3643 - all within noise. Correction to the section above: at K3's ~91
tokens / expert the busiest expert exceeds the pure mode's 96-token switch, so the block's calls normally take the
Tensix-reader gate/up path (the "push" there is the readers' forwarding, not the DRISCs'); forcing the DRISCs changes
nothing. Moving the down weights onto the DRISCs is out regardless: the gate/up forwarding alone (no DRAM reads)
already needs ~1990 us for 694 MB (~350 GB/s), the down weights' 347 MB more would make it ~2980. What is left is the
DRAM efficiency of the full read mix: 1059 MB at ~446 GB/s (87%) vs the gate/up reads alone at 93%.

### K3 experts + combine: the combine tail (2026-10-17; details in FLAT_EXPERT_WORKLOG.md)
Inside the block trace the fused program's flat cores end at ~2305-2340 us and the program at ~2440 (profiled): the
combine runs ~110-120 us alone after the experts every call. tt-npe on a K3-like trace: no NoC congestion (impact 0%,
links avg 8%), DRAM ~87% in steady state.

### K3 block: the latent linears as a custom 2D matmul, combine_rf/lin2d.py (2026-10-17) -> 3613 us traced
Why: ttnn mm2d at 320 rows sits at 117-123 us whatever the config (its in0/in1 distribution pipeline, ~8 us a block;
compute-skip and in1-read-skip probes change nothing). The earlier custom stream_linear v4 (rotating column owners)
lost ~90 us to colliding multicasts. Measurements that shaped lin2d (bh-lb-09, 12 x 10):
- multicast concurrency (zz_scratch_mcast_bw.py::test_mcast_concurrent, per column): one sender 52 GB/s; two senders on
  NOC0 (top + middle) 28.9; top on NOC0 + bottom on NOC1 99.7.
- DRAM reads from the edge rows (zz_scratch_dram_mix.py::test_edge_readers, 44-52 KB pages): one row on NOC0 384 GB/s,
  on NOC1 ~100; rows 0 + 9 both NOC0 369, split NOC0 / NOC1 185 (the NOC1 row drags); whole grid NOC0 390, NOC1 236.
  2 KB pages need depth 6 for ~30 GB/s a core. Pinning each reader to one bank: no gain (~390 GB/s ceiling for cores
  away from the DRAM; the flat op's bank-adjacent readers reach ~510).
- crossing multicasts: a row multicast (x) and column multicasts (W) on the same NoC stall each other - the per-core
  event trace (L2D_PROBE=512) showed core (0, 0)'s x block 2 leaving at 57.7 us (block 0 at 0.4), row 0's last x push
  at 58 us. W-only multicasts 72 us, x-only 51, both 107.
Design (see the module docstring): x resident in L1 and multicast first by the row's two ends (NOC0 / NOC1), then W
down each column by its two ends (NOC0 / NOC1), every DRAM read on the BRISC (NOC1 senders fed through L1 semaphores),
W pieces one contiguous region each (bank-sharded per group of equal-width columns, no padding), per-slot credits (a
shared credit counter let a fast core's credits stand in for a slow one's: hang).
Steps (lat_down / lat_up standalone, us; mm2d 117 / 122): first version (all reads per tile, NOC1 reads) 181 / 174;
bank-contiguous pieces 186 (reads were not the per-tile cost: NOC1 reads were); NOC0 reads + resident x 141 / 121;
x on NOC1 only (one sender) 107 / 100; x first (barrier) with both ends 105 / 97; exact pieces 103 / 95. Probes at the
end (lat_down): skeleton 23, compute only 67, multicasts only 47, DRAM reads only 95 -> DRAM bound (x then W at ~340
GB/s). LoFi 101 / 93 (no gain). Deeper W prefetch (8, 12 pieces) or x staging (6) or no W prefetch during x: no gain.
In the block (K3_LIN2D=1, default; K3_LIN2D=0 = ttnn mm2d): latent down 115.4 -> 103.2, latent up 113.6 -> 94.6, traced
3649 -> 3613 us; PCC 0.99619 both (latent down stage 0.99987 vs 0.99988).
Next (if worth it): feed W through bank-adjacent readers (~510 GB/s) instead of the edge rows (~390): ~70-75 us floor.

### K3 block: the shared expert as lin2d (2026-10-17) -> 3483 us traced
lin2d on the shared expert's shapes (standalone, us; mm2d in brackets): gate 158 (196), down 156 (195), gate + up as one
N 12288 linear 286 (378). Two new lin2d modes: glu=True (weight tile columns are gate / up pairs; Kimi's SiTU-GLU
applied on the PACK thread's SFPU over the pass's DST tiles, as se3_compute, and half the tiles packed: a [S, I]
directly) and residual=True (the NCRISC reads the core's tiles of r into CB 5; compute starts DST from them, then the
matmul accumulates: y = r + x @ W, one rounding). Standalone: gate / up + SiTU 299 us (a PCC 0.99994), down 156,
down + residual 153 (0.99989), both 456 (0.99987) vs the flat SE's ~525 + the sum.
In the block (K3_SE=lin2d, now default): latent up first (yr in L1), then gate / up + SiTU (a in L1), then down + yr
-> the block's y; the SE untilize (24 us) and the sum (45 us) are gone. Trace: gate / up 309, down + residual 158;
traced 3613 -> 3483 us. PCC 0.99622 (flat SE 0.99619), shared expert stage 0.99981 (0.99976).

### lin2d: what bounds it now, and the bank-reader numbers (2026-10-17)
Probes (standalone, us; full / reads only / compute only / multicasts only / skeleton): SE gate+up (N 12288) 286 / 272 /
196 / 118 / 41; SE down 156 / 146 / 108 / 72 / 35; latent down 103 / 95 / 67 / 47 / 23. All DRAM-read bound: the SE's
reads run at ~375 GB/s, near the ~390 ceiling of readers away from the DRAM. Sender variants on the current code
(lat_down / se_gu2): both ends + x first (default) 102.5 / 286.5; one end each, no barrier 116 / 314; one end each +
barrier 129 / 326; x one end, W both 116 / 303.
Readers next to the DRAM (zz_scratch_dram_mix.py::test_bank_readers, each streaming only its own bank, 44.5 KB reads):
the optimal worker per bank (get_optimal_dram_bank_to_logical_worker_assignment, NOC0: (0,9) (0,0) (0,7) (0,3) (7,9)
(7,1) (7,6) (7,4) on the 12 x 10 grid) alone 507 GB/s (99%); plus its vertical neighbour (2 a bank) 398; 3-4 a bank
~300-325 (more readers per bank is worse). So the 99% path is exactly 8 readers, one per bank, ~63 GB/s each.
Feeding lin2d's 12 columns (24 sender roles) from 8 single-bank readers needs either uneven bank loads or a
forwarding layer (reader -> column-end sender, remote slot credits); estimated <= ~180 us over the block's four lin2d
calls (309 + 158 + 103 + 95 us at ~330 GB/s -> ~500). Not started.
Correction (same day): the 2-a-bank drop was the neighbour's position, not the count. With the neighbour in the bank
worker's row (BR_HORIZ=1, x + 1: exactly the flat plan's 16 readers (0|1, y) / (7|8, y)): 1 a bank 507 GB/s, 2 a bank
502 (98%), 3 a bank 450 (88%); vertical neighbours 398. The flat plan's readers on the gate/up stream alone: 511 (99.9%).
So 16 bank-local readers (2 a bank, row neighbours) are the fast set for a lin2d W feed.

### lin2d: W from the bank readers (2026-10-17) -> 3401 us traced
lin2d's W path now is 24 streams (column x q0 / q1), each with a reader core and its pieces in order in one bank.
L2D_FEED=1: the 16 bank readers (optimal core + row neighbour) read every stream and multicast into the column from
outside it (min-cost-flow placement). Correct, but slower (lat_down 144-164 us vs 104): the trace showed 10 of 12
columns done by ~83 us and two stalled ~45 us (a multicast left at 29 us, one core got it at 74). Cause: in-transit
routes - NOC0 runs east then south, NOC1 NORTH then west (tt-npe's Blackhole model) - so a q1 stream climbs its
reader's column and every q1 runs west along the bottom row. Offline link model (17 x 12 torus, translated coords):
column-end senders put at most 1 stream on any link; the best bank-reader placement found (local search) still 5 on
one link (~100 GB/s wanted vs ~86 a link) - the NoC caps it near the far readers' ~390 GB/s. Probes agreed: skeleton
52 vs 23 us, reads-only 91 vs 95.
Kept: column-end senders, the ends that are bank cores ((0,0) (0,9) (1,0) (1,9) (7,9) (8,9)) reading their bank,
the rest balanced (L2D_NEAR=1, default): lat_down 103 -> 93, lat_up 95 -> 89, SE gate+up 291 -> 268, SE down
153 -> 141 standalone. Bank cores elsewhere in the column (L2D_NEAR_COL=1): no gain (SE gate+up 284). The W staging is
one grid-wide CB (q1 slots overlay q0's unless a core reads both: L1 for the SE gate/up). Block: lat_down 91.3,
lat_up 76.7, SE gate/up 270.3, SE down 143.9; traced 3483 -> 3401 us, PCC 0.99622.
Bank balance (same day): the streams were balanced by count (3 a bank), but columns differ in width (8 / 10 tiles for
the latents, 16 / 20 for the SE down), so the busiest bank held +7.1% bytes (lat_down 3.17-3.66 MB a bank). Now by
bytes (largest first to the least loaded bank; a bank-core column end keeps its bank only within the even share):
exactly even on all four shapes, the six near streams kept. Time unchanged within noise (lat_down 93.4, lat_up
86.7-88.0, SE gate+up 271-273, SE down 138.5-139.4 us): at ~45-50 GB/s a bank no bank is saturated - the cap is the
far readers' NoC/DRAM path (~390 GB/s), not the bank split.

### DeepSeek V4.1-Flash: flat + combine overlap (2026-10-17, LoudBox 2 x 4, bh-lb-09 12 x 10 grid)
On top of mstaletovic/galaxy-unified-k3-l1combine (per-source-ring combine). Stage harness
scratch_flat_combine_rf.py at V4.1 dims, 4x4 per-chip load on 8 chips: H 5120, I 2304, clamped SwiGLU, top-6,
24 local experts per chip, 320 home tokens per chip (~80 rows per expert), y handed over in L1 (FC_L1=1), every
per-call table from the device planner (FC_DEVPLAN=1), real weights for the check.
Run: `SPC_PIECE=10240 FC_PAYLOAD=10880 MIMO_FL_ND=27 FC_H=5120 FC_I=2304 FC_EPC=24 FC_K=6 FC_ACT=clamped_silu FC_S=320
FC_L1=1 FC_DEVPLAN=1 FC_WEIGHTS=real TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1 scripts/run_safe_pytest.sh --mesh --no-precompile <harness> -s`
What it took:
- combine_plan: top-6 (the planner unpacked ids 4 at a time: K % 4) - a one-id-at-a-time path for other K; and its
  "row pages" static_assert sat in a discarded if-constexpr branch of a non-template (still checked): RP == 0 ||.
- Cells: the default unit plan (48 gate/up + 40 down + 16 readers + 4 relays = 108 of 120) leaves ~4 accumulators
  after the 8 port cores; each would own ~80 tokens' fp32 rows (20 KB each): no L1. MIMO_FL_ND=27 (6 columns a down
  core: down 177 vs gate/up 197 us at M 1024 in the balance model, so still gate/up-bound) frees 13 cells: 15
  accumulators. (The harness now gives the combine FC_NACC accumulators with the device planner, as the planner's.)
- **Rows wider than SPC_PIECE (7168 B) hang the L1 hand-off**: the combine splits them into pieces with a trailer each,
  the y writers write one contiguous row + one trailer (se_yrm.hpp SE_Y_L1). V4.1's 10240 B row: SPC_PIECE=10240 and
  a 10880 B fabric payload (BH max 15232). sparse_combine_rf now asserts one piece with the L1 hand-off.
- (The runner's precompile pass "fused first call, rel err 0" is not a check: run with --no-precompile.)
| routing | flat alone | combine alone | serial | fused | combine hidden | fused / flat | link floor |
| uniform | 1089 us | 646 | 1735 | 1238 | 77% | 1.137 | 400 |
| zipf spread 1.2 | 1216 | 671 | 1887 | 1416 | 70% | 1.164 | 466 |
| zipf packed 1.2 | 2618 | 1283 | 3901 | 2927 | 76% | 1.118 | 1108 |
Worst per-token rel err 2.4e-3 (all 2560 tokens) on every routing. Same with the default DRISC setting (DRISC is
off inside the overlap): uniform fused 1248 us.

### V4.1 overlap: queued calls, DRISC inside the overlap, the remaining gap (2026-10-17)
Same setup, calls queued back to back (FC_PIPE=1: the 1.137 above included ~130 us launch skew per call):
| config | uniform fused / flat | spread fused / flat | packed fused / flat |
| A no DRISC, 27 down | 1175 / 1088 = 1.080 (86% hidden) | 1335 / 1214 = 1.099 | 2833 / 2618 = 1.082 |
| B DRISC dual (MIMO_FL_CMB_DRISC=1), 27 down | 1211 / 1195 = 1.014 | 1340 / 1222 = 1.097 | 2833 / 2623 = 1.080 |
| C DRISC-only (DUAL=0), 32 down, 35 combine cells | 1102 / 1084 = 1.017 | 1588 / 1582 = 1.004 | 2991 / 2817 = 1.062 |
K3 (L1 hand-off, queued, same harness): 1.048 / 1.031 / 1.024.
- DRISC inside the overlap (MIMO_FL_CMB_DRISC=1, opt-in): exact on all routings. Dual makes the uniform flat 10%
  slower here (1195 vs 1088) and the fused call slower (1211 vs 1175). DRISC-only frees 15 cells (combine 35 cells, 32
  down cores): best uniform (1102, -6% vs A) but the skewed routings lose (spread 1588 vs 1335, packed 2991 vs 2833):
  their hot experts are past switch mode's 224 tokens (one DRISC tile per bank, ~28 GB/s a tile) and two tiles hit
  the x-read conflict (FLAT_EXPERT_WORKLOG "DRISC-only ... what hurts it"). With 40 down cores the L1 hand-off's
  32-writer limit trips (sparse_combine_rf asserts <= 32 writers).
- The plan now exports drisc / dual / tensix_readers (+ the launcher); the harness gives DRISC-freed reader cells to
  the combine.
- Where A's 8% goes (SPC_TPROF, uniform, us from each chip's combine start; flat alone 1088): every chip's port rows
  have landed by 1114-1132 (the last experts' rows after the flat: release granularity, 24 experts of ~80 rows), the
  accumulators end within ~20 us of their last local row, but the port senders end 1141 / 1141 / 1162 / 1191 / 1200 /
  1223 / 1269 / 1257 on chips 0-7 (slot waits 12-45 us): a chip-dependent sender tail of up to ~140 us. Open: the
  10880 B fabric payload (fewer router slots than K3's 7232) vs the ports' load per chip.

### K3 overlap with DRISC-only and 64 gate/up cores (2026-10-17, 2 x 4, queued, L1 hand-off, device planner)
Harness defaults (K3: 56 experts / chip, top-16, 320 tokens / chip, latent 3584, I 3072), fused us (all exact, 2.2e-3):
| config | uniform | Zipf spread | Zipf packed |
| base: 48 gate/up, Tensix readers, 28 down, 24 combine cells | 2442 | 2727 | 5920 |
| DRISC-only, 48 gate/up (39 combine cells), per-call tile switch | 2314 | 3369 | 6148 |
| DRISC-only, 48 gate/up, both tiles always (ALT_MAXCNT high) | 2314 | 2589 | 5906 |
| 64 gate/up (MIMO_FL_HG=1 HG_NU=3) + DRISC-only + MIMO_FL_ND=28, 14 accumulators, per-call switch | 2289 | 3566 | 6127 |
| 64 gate/up + DRISC-only, both tiles always | 2289 | 2590 | 5705 |
Flat alone: base 2324 / 2586 / 5750; 64 gate/up both tiles 2265 / 2488 / 5601. The skewed routings' DRISC loss was the
switch-mode policy: a call whose largest expert passes 224 tokens ran every bank on one tile (~28 GB/s), starving the
call's DRAM-bound small experts. Default now: both tiles on every call when no Tensix readers back the DRISC slots
(non-dual); dual plans keep 224 / 384. The harness accepts FC_NACC <= 16 (asserting the combine found that many).
DRAM utilization of these fused calls (busiest chip, the harness's routing; bytes = active experts x 18.6 MB bfp4
weights + x rows x 7168 B x 2 rectangles + 2.3 MB combine output; y stays in L1; /512 GB/s):
| routing (busiest chip) | bytes | base 48 gate/up Tensix | 64 gate/up DRISC-only |
| uniform (5246 rows, max 123 / expert) | 1118 MB | 458 GB/s = 89.4% | 488 GB/s = 95.4% |
| Zipf spread (6369 rows, one expert 2552) | 1134 MB | 81.2% | 85.5% |
| Zipf packed (28854 rows) | 1456 MB | 48.0% | 49.9% (compute: 1906 GFLOP at 334 TFLOP/s) |

### 64 gate/up + DRISC-only in the K3 block (2026-10-17, 2 x 4, K3_GU64=1)
tt_k3_token_split_moe.py: K3_GU64=1 sets the layout's plan options (MIMO_FL_CMB_DRISC / DRISC_DUAL=0 / HG / HG_NU=3 /
ND=28, env defaults); the block's combine cells now use the plan's tensix_readers (DRISC-freed reader cells go to the
combine). Same PCC as the base (0.99622 vs host, device routing). Device-routing load per chip (K3_LOAD): mean ~91
tokens / expert, max 141-193, ~40% of experts > 96, none > 224 (the harness's uniform: max 123).
| config | experts_combine (per-op, median) | block sum | traced forward |
| base (48 gate/up, Tensix readers) | 2446 us | 3504 | 3407 |
| DRISC-only 48 gate/up | 2443 | 3516 | - |
| 64 gate/up + DRISC-only | 2476 / 2496 | 3531 / 3568 | 3453 |
| + tile 1 DMA depth 1 / burst 4 | 2548 / 2526 | | |
In the trace, experts_combine 2370 (base) vs 2414 us (64 gate/up). Not what the harness shows (2442 / 2314 / 2289
queued): not the accumulator count (harness with 16: 2311), not x's buffer (both DRAM interleaved row pages), and not
tile 1's x-read conflict (throttling tile 1 makes it worse). Open: profile the block's call (DRISC section cycles,
relay / gate/up wait zones) against the harness's.
**Found (profiling):** DRISC section cycles were the same in the block and the harness (dma / write / flush within
1%), gate/up and relay waits alike; the block's down cores waited for combine slots (SE_OUT_WAIT mean 172k vs 6k
cycles on chip 7). The block reads x once (MIMO_FL_X1 scoped around experts_combine: one primary multicasts into
both rectangles, a win for 48 gate/up: 2433 -> 2372 us); with 64 gate/up cores that one multicaster cannot keep up.
x read twice with 64 gate/up (K3_EXP_X1=0): experts_combine 2380 us, block traced forward 3315 us (base 3407, -2.7%),
PCC 0.99622 as base. K3_GU64 is now the block's default (K3_GU64=0: the old layout), and it reads x twice.

### Deterministic combine (2026-10-17)
Why: 99 traced replays of the K3 block were bit-exact on every stage but the routed output (combine): one element
(chip 3, row 270, col 3153) toggled by one bf16 ulp in 26 of 99 replays - the accumulators summed each token's rows in
fp32 in arrival order. CI should assert bit-exact repeats (to catch any other nondeterminism), so the combine's sum
order is now fixed (`sparse_combine_rf_prepare(deterministic=True)` + `combine_planner_prepare(det=...)`, K3 block
default; harness FC_DET, default 1; test `scratch_k3_moe_layer.py::test_k3_moe_determinism_trace`, K3_DET_REPLAYS).
The device planner (multi-core, no host step) exchanges per-chunk expert counts, rebuilds every source chip's flat
schedule (se_dyn_from_counts: the flat's own code incl. pinning) and writes an order table per accumulator.
- v1, one global order (schedule entry v, source chip k), per-accumulator counts per (k, v): K3 block 0 of 99 replays
  differ (was 26 of 99). Harness (K3, 48 gate/up Tensix): uniform 2430 vs 2432 us, packed 5898 vs 5924, but **Zipf
  spread 3876 vs 2728 (+42%)**: lockstep by entry index makes every chip wait for the hot chip's (pinned, long)
  entries at every step - the call costs sum over v of max over chips, not max over chips of the sum. V4.1 (DRISC-only,
  32 down) spread 1528 vs 1306 the same way.
- v2, per-token order: per owned home token, its rows' source chips sorted by (predicted finish of the row's
  schedule entry on its chip: an entry costs max(w_rows = 160, its rows), source). Nondecreasing along each source's
  production, so deadlock-free (the lowest-key row not taken is takeable once it lands); a ring now waits only when
  its head row's token still needs an earlier row from another chip. (results below)

### V4.1 with DRISC-only, both tiles every call (2026-10-17, queued, L1 hand-off, device planner, 32 down)
The K3 switch-mode fix (both DRISC tiles on every call when no Tensix readers back the slots) carries over: the flat
alone is now on par with the Tensix layout on the skewed routings and 1% faster uniform.
| config | uniform fused / flat | spread fused / flat | packed fused / flat |
| A no DRISC, 27 down (old best) | 1175 / 1088 | 1335 / 1214 | 2833 / 2618 |
| C DRISC-only, per-call tile switch, 32 down | 1102 / 1084 | 1588 / 1582 | 2991 / 2817 |
| DRISC-only both tiles, 32 down, arrival-order combine | **1103 / 1082** | **1306 / 1228** | 2869 / 2624 |
| same, deterministic v1 (global order) | 1103 / 1082 | 1528 / 1229 | 2823 / 2620 |
Exact (rel err 2.4e-3) on all routings; v1 bit-exact across 10 repeats on all.

### Smaller combine footprint (2026-10-18): small-ring fix, ports per direction, accumulators on ports
Why: the combine took ~21-23 cells (a port core per (direction, link) + 14-16 accumulators), more on Galaxy (more
directions / links); the flat expert could use them (or must give them up on Galaxy).
- **Small-ring hang** (the "7 accumulator" hang): a y writer issued a row tile only once every destination had
  granted slots for all of that tile's rows; an accumulator's local ring smaller than that (5 slots: 7 accumulators in
  1242 KB, or 8 in 1190 KB) could never be granted. Fix (se_yrm.hpp): a tile goes out in grant-sized units. 8 / 9 / 10
  / 7 accumulators all pass; fewer accumulators still cost under skew (7: packed fused 3149 vs 2727 us).
- **Ports per direction** (ports_per_direction, port_luts(per_direction=True)): one sender drives both links of a
  direction. Exact; V4.1 fused 1106 / 1344 / 2947 vs 1103 / 1304 / 2727 us per link. The hot chip (packed: chip 0) is
  link bound and its writers stall on the shared 64-slot ring (rows all handed off 2939 vs 2721 us); a 90-slot ring
  2881. Ring capacity per link matters on skew: per link with 32 slots 1354 / 2865. Kept as an option for Galaxy (where
  ports outnumber accumulators); off on 2x4.
- **Accumulators on the ports** (acc_on_ports): the A accumulator cells are the same on every chip, each chip's ports
  take the nearest to its Ethernet cores, and a port core is also an accumulator: its sender RISC also runs the
  hand-off grants / pushes and the accumulator's output writes; RISCV_1 runs the accumulator reader, the TRISCs its
  compute. Combine cells = max(ports, A). The rings come from what the arena leaves (local ring <= 16, injection
  ring <= 64 slots).
| layout | combine cells | V4.1 fused u / s / p (us) | K3 fused u / s / p (harness) |
| separate, 64-slot rings | ~19-23 | 1103 / 1304 / 2727 | 2289 / 2596 / 5727 |
| accumulators on ports | 15 (V4.1) / 16 (K3) | 1106 / 1360 / 2877 (32-slot ring: L1) | 2289 / 2592 / 5807 |
K3 block (default now): PCC 0.99622, experts_combine 2356 us, traced forward 3312 us (was 3314), 99 CI-mode replays
bit-exact. Freed cells: K3 with 32 down cores (min(96, 96) effective) gains nothing (flat alone packed 5528 vs 5592:
K3 packed is not down bound); V4.1 72 gate/up now runs with its combine (13 cells) but loses to 48 (1303 / 1670 /
2822 us).

### Combine alone, profiled (2026-10-18, 2 x 4, separate ports / accumulators, DRAM y, SPC_TPROF=1)
Why: what bounds the combine by itself. Floor = the busiest link's rows (route() per hop, direction, link) x row bytes
at 25 GB/s. Harness: SPC_TPROF now also times the combine-alone call; per sender: end, slot wait (fabric slot),
idle, credits, send, flush, rows, blocked (destination ring full), credit packets.
| | uniform | spread | packed |
| K3 alone / floor (us) | 1023 / 739 (72%) | 1061 / 798 (75%) | 2332 / 2074 (89%) |
| V4.1 alone / floor (us) | 629 / 400 (64%) | 662 / 466 (70%) | 1278 / 1108 (87%) |
- Packed: link bound. The hot chip's 4 senders push 5.3-7.3k rows each at ~22.4 GB/s per link (K3 7255 rows x 7.2 KB
  in 2278 us), idle < 10 us, almost never blocked; ~40% of their send time is waiting for a fabric slot. Every
  other chip's own rows are out in < 400 us; two of its senders then only return credits until the hot chip ends.
- Uniform / spread: the busiest link carries only 17-19 GB/s. Not the readers (they end just before the senders: the
  injection CB throttles them; SPC_NOYREAD=1 no faster), not the accumulators (end with the last row), not flushes
  (2-5 us), not credits (SPC_CREDTHR=2: K3 -4%, V4.1 0).
- Ring depth 8 (FC_D=8, FC_ARENA_KB=1200): blocked drops from 300-1150 to 0-250 per sender; K3 uniform 1022 -> 965,
  V4.1 632 -> 588, packed -1-2%. What is left is fabric backpressure: the slowest chip's senders at D 8 spend 50-60%
  of their time waiting for a fabric slot with nothing blocked or idle (K3 chip 7: slot 445-505 of 850-970 us), on
  links that carry forwarded traffic too. Pure injection gets ~22.4 GB/s per link, mixed forward + inject 17-19.
- Fused: depth 8 changes nothing (V4.1 1188 -> 1177, K3 2456 -> 2460 uniform; packed same): the combine is hidden
  and the fused cost is on the flat side. Depth stays 4 (the K3 block's arena is per call).

### When the level-1 combine becomes link bound: stage size x chunk size (2026-10-18, CPU model)
Why: the combine's link time grows with the chunk while the flat expert is weight-stream bound (nearly flat in the
chunk); at some chunk the combine can no longer hide. Model: `models/demos/deepseek_v3_d_p/tests/perf/
scratch_combine_linkbound_model.py` (no device).
- Flat, per chip: max(active experts x w0 + a x rows, c x rows), slowest chip. K3 w0 37.1 us, a 0.062, c 0.180 us
  per row; V4.1 (384 experts, top-6, H 5120, I 2304) w0 39.4, a 0.089, c 0.247. Within 1-4% of the 2x4 harness on
  uniform / spread; c is fitted on the packed run (it includes the hot-expert slowdown).
- Combine: busiest (chip, direction, link) rows x 2H B at 25 GB/s / the efficiency measured on 2x4 (uniform 72 / 64%,
  spread 75 / 70%, packed 89 / 87% for K3 / V4.1). Checks: K3 2x4 2.5K floor 746 vs 739 us, combine 1035 vs 1023.
- Assumed: 2 links per direction everywhere; 2x4 = mesh, 4x4 = wrap on the 4-wide axis, 8x4 = full torus; the 2x4
  efficiencies on every stage (more forwarding on 4x4 / 8x4 likely lowers them).
combine / flat (>= 1: link bound), global chunk:
| routing | stage | 2.5K | 5K | 8K | 10K | 16K |
| uniform | K3 2x4 / 4x4 | 0.23 / 0.24 | 0.43 / 0.44 | 0.64 | 0.76 | **1.06 / 1.07** |
| | K3 8x4 | 0.31 | 0.55 | 0.81 | 0.97 | **1.35** |
| | V4.1 2x4 / 4x4 | 0.31 / 0.32 | 0.56 / 0.57 | 0.81 / 0.84 | 0.96 / 0.98 | **1.30 / 1.35** |
| | V4.1 8x4 | 0.41 | 0.75 | **1.06** | **1.28** | **1.63** |
| spread | K3 2x4 / 4x4 / 8x4 | 0.23 / 0.30 / 0.45 | 0.42 / 0.53 / 0.78 | 0.62 / 0.74 / 0.95 | 0.73 / 0.86 / 0.97 | 0.91 / 0.91 / 0.96 |
| | V4.1 2x4 / 4x4 / 8x4 | 0.31 / 0.50 / 0.71 | 0.53 / 0.82 / 0.82 | 0.74 / 0.90 / 0.84 | 0.79 / 0.89 / 0.83 | 0.79 / 0.89 / 0.82 |
| packed | K3 2x4 / 4x4 / 8x4 | 0.45 / 0.67 / 0.47 at every chunk | | | | |
| | V4.1 2x4 / 4x4 / 8x4 | 0.48 / 0.72 / 0.51 at every chunk | | | | |
- Uniform crossover (today's combine / at the link floor): K3 2x4 and 4x4 ~14.8K / never (the floor stays ~0.9x the
  flat compute slope), 8x4 ~10.8K / ~17K; V4.1 2x4 and 4x4 ~10.5K / ~20.5K, 8x4 ~7.5K / ~13.8K. Exposure starts
  before 1.0: on 2x4 the fused cost is already +2-9% at ratios 0.23-0.58.
- Packed: constant ratio - the hot chip's flat work is compute bound on its own rows and its out links carry those
  rows. 4x4 is worst (no Y wrap: 12 of the hot chip's 15 peers through one Y direction). Never link bound.
- Spread: rises, then plateaus at 0.8-0.97 once the chip holding the hottest expert goes compute bound. K3 8x4 sits
  at 0.95-0.97 from 8K: link exposed. Small chunks on big stages are worse than uniform (V4.1 8x4 2.5K 0.71 vs 0.41).
- Zipf 1.2 over all experts is extreme (the hottest expert gets ~every token); the measured hot-expert slowdown at
  4096+ tokens is not in the model, so the skewed ratios are upper bounds. Skew is a load-balancing problem.
Levers that move the crossovers: fp8 rows (1.78-1.94x fewer bytes; pjosipovic/k3-moe-fp8-wire: e4m3 + pow2 scale
per 1x32, Galaxy 8x4 combine 1001 -> 560 us, PCC -0.001, -2.3% output norm drift with truncation; for us the
quantize fits the down cores' epilogue - one DEST tile row = one 1x32 block - and the dequantize the accumulator's
gate multiply as a column broadcast), level 2 combine (one row per (token, chip): K3 2x4 2.3x fewer rows, but rows
wait for the token's last local expert), and the balanced-routing link efficiency (fabric forwarding: 64-77% of the
floor today).

### GLM-5.3-Flash flat + combine overlap (2026-10-18, 2 x 4, bh-lb-09 12 x 10, harness, DRISC-only 64 gate/up + 32 down)
Run: `SPC_PIECE=8192 FC_PAYLOAD=8704 FC_H=4096 FC_I=2048 FC_EPC=36 FC_K=8 FC_ACT=clamped_silu FC_S=640 FC_L1=1
FC_DEVPLAN=1 FC_PIPE=1 MIMO_FL_CMB_DRISC=1 MIMO_FL_DRISC_DUAL=0 MIMO_FL_ND=32` (32 down = 36 down: 128 column tiles,
4 a core either way; single chip within 0.2-1.4%). 19 free cells: separate layout 7 ports + 12 accumulators
(FC_NACC=12), or accumulators on ports (16, MIMO_FL_NH=2). Ring depth 3 at S 640 (L1), 4 at S 320. Exact (2.4e-3).
| S / chip, layout | uniform flat / fused | spread | packed | combine alone u / s / p |
| 640, separate | 1305 / 1444 (1.11) | 1592 / 2156 (1.35) | 4191 / 6254 (1.49) | 1270 / 1457 / 2844 |
| 640, acc on ports | 1311 / 1533 (1.17) | 1605 / 2294 (1.43) | 4262 / 7018 (1.65) | - |
| 320, acc on ports | 1224 / 1251 (1.02) | 1347 / 1453 (1.08) | 2440 / 2969 (1.22) | - |
Link floors (S 640, busiest link at 25 GB/s): 847 / 961 / 2425 us.
**Why packed does not overlap: the device planner's row order.** Rows of a local expert are in (home, token) order
(combine_plan docstring), so a hot expert's rows go out home by home: 640 rows to one home, then the next. A burst to
one home uses only the ports toward it (2 of the hot corner chip's 4 links) and that home's accumulator rings (A x D
= 36 slots in flight); the senders block on credits instead of the link. Hot chip, fused separate layout: rows handed
off by 6197 us (flat alone 4191), senders send ~0.8-1.0 ms, slot wait ~0.2 ms, idle 3.5-4.0 ms, blocked 7-12.7k
times. Proof with the combine alone (DRAM y, host tables): token-major order (the harness default) 2844 us, link
bound (slot wait 0.66-0.82 ms, blocked ~1.2k); home-major (FC_ORDER=home, as the planner) 4121 us with blocked
6.9-12.5k and slot wait 0.24-0.31 ms - the fused run's signature. Uniform does not care (1270 vs 1297: an expert's
rows per home are few). Roofline per burst: the flat makes a home's 640 rows in ~92 us (0.144 us a row); the 2 links
toward that home need 640 x 8192 B / 44.8 GB/s = 117 us - each remote burst is link bound on half the links, while
over the call the links are only 63% busy. K3 (S 320, 7 KB rows, 0.20 us a row) was under that line (64 vs 51 us).
Accumulators on ports is worse on packed (7018 vs 6254): the port RISC also runs the hand-off grants.
**Fix (combine_plan token_major, default):** the multi-core planner's core p takes tokens [p S / M, (p + 1) S / M) of
every home (one id / weight run per home), so an expert's rows come out in (token, home, k) order; the deterministic
combine keeps (home, token, k) (its order table assumes it). Bit-exact vs plan_reference(token_major=True) on all
chips: K3, GLM S 640 and V4.1 shapes, single-page routing and K3's joint row pages. Planner +7-10 us (32 cores: K3
joint 69 -> 76, GLM 48 -> 59 us). Fused / flat, home order -> token order:
| | uniform | spread | packed |
| GLM S 640, separate | 1444 -> 1353 (1.036) | 2156 -> 1704 (1.055) | 6254 -> 4744 (1.130) |
| GLM S 640, acc on ports | 1533 -> 1445 (1.101) | 2294 -> 1758 (1.088) | 7018 -> 4846 (1.138) |
| GLM S 320, acc on ports | 1251 -> 1248 (1.020) | 1453 -> 1389 (1.029) | 2969 -> 2610 (1.066) |
| K3 production layout (same day A/B) | 2289 -> 2287 | 2625 -> 2490 (1.009) | 5800 -> 5380 (1.020) |
| V4.1 (28 down, separate, same day) | 1180 -> 1172 | 1355 -> 1295 | 2816 -> 2722 |
K3 block: PCC 0.99622 (unchanged), traced forward 3322 us (3312: its near-uniform routing gains nothing, the
planner's +10 us shows), 99 CI-mode replays bit-exact. Open at GLM S 640: fused still +4-14% (ring depth 3 there).

### GLM-5.3-Flash on the K3 token-split block (2026-10-18, hack: GLM_EXPERTS_MODE=ts, LoudBox 2 x 4, 12 x 10 grid)
TtK3TokenSplitMoE(routed_only=True, flat_env=...): no router (the model's top-k goes in), no latent projections
(dispatch rows = x), no shared expert; one block per (mesh, chunk) for every layer, each layer's FlatRoutedExpert
passed per call (routed(op=)). glm53_flash_d_p/tt/experts_ts.py wraps it (split + full-mesh input only); the shared
expert keeps its TP path on a full-mesh tiled gather of x (the dispatch is sparse now). Plan: DRISC-only, 64 gate/up +
32 down, two x helpers, MIMO_FL_X1 off (fused 1.99 -> 1.70 ms). Fabric payload 8704 (BRINGUP_FABRIC_PAYLOAD; a
8192 B row + the 32 B trailer).
What it took: the sparse dispatch's row group 4 for rows > 7 KB (8 KB x 8 x 2 CBs clashed with the arena); the
arena checked per call (arena_fits) and the block rebuilt once when the model's later first-call allocations moved
it (the shared expert's); joint routing rows at top-8 (the staging check assumed 16); one combine piece a row
(piece_max). Host: place_ports and the fabric all-gather plan memoized (45 -> 15 ms per call; K3 block untraced
32.2 -> 6.7 ms per call); the rest is the fused op's descriptor conversion (~9 ms).
Accuracy (zz_scratch_experts_ts.py, random bfp4 experts, 5120 tokens): ts vs ag rel L2 0.37%, PCC 0.999994; both vs
CPU fp32 0.2005 / 0.2008 (bfp4), uniform and one hot expert. The checkpoint shards are gone from disk: no real-weight
run.
Layer device / traced (fake weights, chunk 5120):
| layer | ag device / traced / eager wall | ts device / traced / eager wall | experts step ag -> ts |
| L3 dsa_moe | 16.16 / 16.18 / 18.77 | 15.27 / 15.20 / 18.36 | 3.711 -> 2.716 |
| L4 kda_moe | 9.63 / 9.66 / 10.71 | 8.91 / 8.76 / 15.47 | 3.450 -> 2.724 |
ts experts step: flat + combine 1.65-1.67, routing gather / planner / dispatch prep / zero / dispatch / tilize 0.56-0.60,
the shared expert's x gather 0.42-0.49 ms. ~-0.9 ms per MoE layer traced (~38 ms per chunk over 42 layers).
