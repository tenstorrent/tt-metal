# matmul_reduce_scatter — post-run perf worklog

Box: Blackhole LoudBox 2x4, TP line of 4 along cluster_axis=1, 2 links, FABRIC_2D. Times: device kernel time,
slowest chip, median (test_mmrs_ab.py, in-process profiler). Default payload 14400 B unless stated.

## Starting point (working tree on top of run commit 9e3a564)

Already applied before this log:
- Perf 2 hand-off depth (G slots when L1 fits) — from the eval run, uncommitted.
- Public-API transport add (`xport_add.cpp`) replacing the raw-LLK one (A/B: within ~1-3 us; raw kept as
  `xport_add_raw.cpp`, `MMRS_XADD_RAW=1`).
- Diagonal line injectors (`MMRS_INJ=diag`, default) + NoC rule: block-invariant operand reads on NoC0, streamed on
  NoC1 (`MMRS_A_NOC/MMRS_W_NOC=auto`). MiMo 329 -> 271 us, FOCUS/GLM flat.

| case | honest unfused (mm + fabric_reduce_scatter) | fused, start of this log |
|---|---|---|
| FOCUS 640x2048x7168 | 225.5 | ~156 |
| GLM 640x4096x6144 | 256.6 | ~202 |
| MiMo 2048x2048x4096 | 391.9 | ~271-275 |

## Log

### 1. Where the FOCUS time goes (zones, chip 0 = interior, chip 1 = line end)
- Matmul: block ends 32.5 / 57.2 / 81.5 / 106.5 us. Steady block = ~24.5 us; the math alone would be ~21 us/block.
  The matmul is paced by the 8 W line injectors: 16 W K-blocks x ~5.8 us (read ~3.9 + multicast ~1.9). Block 0
  pays ~8 us extra (first W multicast only at ~5 us, A resident load until ~14 us).
- Ready fence: chip 0's ports wait in `snd_fence` until 16-28 us (cross-chip launch skew), but that ends before
  block 0 is ready (32.5), so it is NOT on the critical path here.
- Transport: ports send from ~33 us (block 0 ready) to ~120-139 us, i.e. ~95 us = 3.4 MB/link at ~36 GB/s.
- Tail: interior-chip finals wait for the own block (computed last) until ~107 us, then read+add+store until
  ~161 us (~54 us). Line-end finals: ~25 us. Interior final = 3 input streams (~1.7 MB) on ONE NCRISC per final core.

### 2. Operand buffer depth 3 / injector read-ahead — NO WIN
| variant | FOCUS | GLM | MiMo |
|---|---|---|---|
| base | 160.0 | 203.3 | 273.9 |
| read-ahead (depth 2) | 167.0 | 207.2 | 280.5 |
| depth 3 | 165.4 | 201.7 | 275.0 |
| depth 3 + read-ahead | 166.3 | 203.1 | 280.4 |
Zones with depth 3 + read-ahead: reads and multicasts do overlap, but a W K-block still takes ~5.8 us. The W
injectors are per-core DRAM-read-bandwidth bound (~30-38 GB/s per injector, 8 injectors), not serialization bound.
Next lever for the matmul phase would be MORE W readers per line (rotating senders), not deeper buffers.

### 3. Three final cores per link (FINALS_PER_LINK=3, was hard-coded 2) — SMALL WIN on FOCUS
Generalized the finals count (placement, compute-reader consumer counts, port-sender per-final counters, final
segment stride F L). Unit suite passes with F=2 and F=3.
| | F=2 | F=3 |
|---|---|---|
| FOCUS | 156.0 | 153.2 |
| GLM | 201.9 | 200.9 |
| MiMo | 271.9 | 273.0 |
Timeline (F=3): interior finals still start at ~109 us (own block computed last) and end ~20 us after their last
arrival (~130-134); line-end ports send 3 blocks link-bound from ~33 to ~134 us (~34 GB/s/link) -> that is the floor
for every chip with the current fill. Finals' processing went 54 -> 44 us but stays the interior-chip tail.

### 4. Transport knobs — NO WIN (keep defaults)
| variant (all F=3) | FOCUS | GLM | MiMo |
|---|---|---|---|
| base | 153.8 / 157.2 (two runs) | 202.7 / 198.3 | 269.5 / 273.4 |
| transport CB 224 KB (group 8 segs) | 153.6 | 201.6 | 271.6 |
| INC_EVERY=4 | 161.5 | 200.4 | 280.3 |
| INC_EVERY=2 | 171.5 | 202.0 | 298.1 |
| INC_EVERY=16 | 161.8 | 199.3 | 276.5 |
| INC_EVERY=32 | 180.1 | 209.1 | 283.8 |
INC_EVERY=8 is the sweet spot both ways (more increments cost link throughput; fewer delay the relays/finals).
Run-to-run noise of the medians is ~±3-4 us.

### 5. Rotating W senders (MMRS_W_ROT=1, diagonal rotation) — NO WIN
Every core of an n-line reads + multicasts every 10th W K-block (rounds offset by line index -> diagonal senders).
Correct (unit suite passes). FOCUS 153.2 -> 154.4, GLM 201.1 -> 203.2, MiMo 268.2 -> 270.3 (noise).
Zones: block ends unchanged (33.4 / 57.8 / 82.5 / 107.1). Reason: one K-block of math is ~5.3-6 us, about the same as
the old ~5.8 us W injection per K-block, so the matmul phase is ~compute-bound already; faster W delivery cannot show.
The fill is ~one block of compute (~24.5 us) + ~6 us to land the first W K-block (read 3.9 + multicast 1.9).
Kept as an off-by-default knob.

### 6. Where the interior-chip tail comes from (diagnosis)
- Only 2 forwarding links per hop exist under FABRIC_2D on this box (probed), so the line-end floor (3 blocks through
  2 links, ~95-100 us after block 0) stands.
- Device map: mesh rows [7,6,4,5] / [3,2,0,1]; TP lines are the rows, so device 0 is position 2 of line 3-2-0-1.
- Ablations (FOCUS, F=3): payload-free link -5 us; no transport reads -8 us; both -28 us (they co-limit).
- Line-end ports forward ~34 GB/s/link; RELAY ports only ~20-28 GB/s/link (dev 2 relays block 3 from 33 to 89 us while
  its upstream sent it 33-66). Interior finals then wait ~63 us on that relayed arrival stream (new zones
  xr_arr_wait_a/b, xr_barrier), not on their own processing.
- New ablation MMRS_ABLATE=ARRREADS (skip only the arrival DRAM reads): FOCUS 161.8 -> 147.2, GLM 200.0 -> 192.8,
  MiMo 274.6 -> 260.5. Relay/final readers pull 2-3 x 14 KB per segment on ONE NCRISC (NoC0); a line-end port pulls 1.
  Bigger transport groups (more reads in flight) did not help (#4) -> per-RISC read bandwidth, not latency.
Next: move the arrival reads of relay ports onto the port's BRISC (NoC1), which mostly waits to send.

### 7. Relay arrival reads on the port's BRISC (MMRS_PORT_ARR=1) — MUCH WORSE, reverted to off
Non-blocking "arrival pump" in the port sender: reads arrival segments into cb_arrival_a under the same counter
gating; relay reader then only gathers. Correct (unit suite passes).
| | F=3 | F=3 + PORT_ARR |
|---|---|---|
| FOCUS | 154.4 | 216.6 |
| GLM | 202.0 | 234.9 |
| MiMo | 270.0 | 355.3 |
With the arrival reads ablated the pump structure itself costs only ~5-9 us (FOCUS 145.3 -> 154.6), so the loss is
the DRAM reads travelling on NoC1 (BRISC): up the DRAM columns and along the transport row, where the fabric sends
also go. Kept in the code as an off-by-default knob (MMRS_PORT_ARR).
