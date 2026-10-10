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
