# Flat routed expert: GLM-5.3, 16 experts a chip (4x4 stage), M ~ 320 — perf worklog

Branch `mstaletovic/flat-expert-perf-glm16`, from `origin/mstaletovic/moe-flat-combine-lab-wip` (`0f15f622221`).
Goal: speed up the flat routed expert alone (no quantization) for GLM-5.3 (H 6144, I 2048, E 256, K 8) at 16 experts
a chip, M ~ 320 (chunk 10240), without more experts a chip. Single Blackhole p150b (11 x 10 workers), 1.35 GHz.
Numbers: `test_flat_routed_expert_checks.py` perf medians (5 runs), run through `scripts/run_safe_pytest.sh`.

## Where the time goes (probes)

Measurement probes: `FLAT_ZZ_PROBES=1` + `TT_METAL_DEVICE_PROFILER=1` (kernels/zz_probes.hpp; summed waits a kernel
emits at its end). Report script: per-probe mean / min / max over cores; a grid view per core.

GLM 4x4 M 320, after the commits below (1094 us):
- gate/up (64 cores, the critical path): span ~1050 us; math (TRISC1 acquire -> commit) 19.4 cycles a tile vs 18.1
  for the same 2 x 2 matmul_block_k alone (single-core microbench, kernels/mm_shape_bench_*); x wait ~90 us, almost
  all in the first two sub-blocks (supply ramp + snake fill; the snake tail starts ~50 us after the head and, through
  the h buffers, every core runs at the tail's pace); weight wait ~15 us.
- down (32 cores): busy ~750 us, waits on h ~260 us; finishes ~20 us after its last h (~55 us after gate/up's mean).
- Block shape / per-call overhead: 2x2 half sync 18.15 cycles, K 8 vs 32 a call only 2% - not the gap.

## Committed

| commit | change | GLM 4x4 10240 uniform |
|---|---|---|
| `15116bf869b` | x ring grown into the arena's leftover L1 (pair splits; off in the combine, whose landing pool uses that L1) | 1258 -> 1217 (-3.3%) |
| `f3fcf1d0ed0` | x forwarder: 6 pieces in flight for experts of 6+ row tiles, 3 below (6 everywhere slowed short experts / the Galaxy MoE block) | -> 1157 (-5.1%) |
| `e0f2d6148c2` | x snake + head writes on unicast VC 3 (was VC 1 with the weight read requests; h 0, y 2) | -> 1108 (-4.1%) |
| `789fcaa2888` | gate/up weight grants capped at 8 until a core's first sub-block x is in (loads of 24+ blocks) | -> 1094 (-1.4%) |
| `ef26bcabe09` | 1 gate/up weight read in flight at 16+ experts a chip, whole-tile gate / up (was 2); determinism test compares token rows only | 1096 -> 1093 (-0.3%); M 80 / 160 / 240 / 480 -2.9 / -2.7 / -4.7 / -0.8%, zipf -1.6% |

Zipf 1381 -> 1288. Every commit: full perf matrix (21 cases, all models) + accuracy / determinism (52 passed).
Side effects elsewhere: 8x4 10240 -10%, DSV4.1-Flash hot -15%, K3 hot -3%; worst +1% (short experts).

## Tried, not kept (measurement switches, all default off)

- `FLAT_ISO` bits (1 no gate/up weight reads, 2 no down weight reads, 4 no x DRAM reads): M 320 1156 -> 1060 / 1151 /
  1163. Only the gate/up weight reads slow x: NoC contention (their NOC0 responses run down the snake's columns), not
  DRAM bandwidth.
- `FLAT_GU_W_NOC1=1` (gate/up weights read by the BRISC on NOC1): 1768 us - the helpers' x DRAM reads (NOC1) 160 -> 645.
- `FLAT_X_MCAST` (x multicast over each rectangle, no snake; a freed word a core on the primary): plain 1836, linked
  within a super-block 1417, + posted no change, VC 5 no change. Issue rate on the primary and the slowest-of-40 slot
  freeing bound it. First version hit a watcher NOC alignment error (semaphore source word not 16-byte aligned) - fixed.
- `FLAT_X_STATE=1` (stateful forwarder writes): no change (and lost the VC gain).
- `FLAT_X_PIECES` 8 / 2 pieces a block: +14% / +3% (4 best). Batched XARR increments (kept, neutral at 4).
- `FLAT_EVEN_RECTS=1` (two 4 x 8 rectangles, east relays in the anchor column): +1.8% - weight waits up (down cores
  off the bank anchors). The p150 grid only fits 4x10 + 3x8 as full rectangles otherwise.
- `FLAT_W_CAP` (weight ring capped for the whole run): 36 of 48 +2.5%, worse below.
- `FLAT_GU_FULL_SYNC`, x ring 29+ (falls back to MT 2 / 1), `FLAT_NO_ROW_PASSES`, relay landing 3/4, down ring 2.5/3,
  `FLAT_X_NOC1`, `FLAT_RELAY_HN`, matmul_block_k (kept, bit-identical, neutral), skip activation (free): no gain.
- `FLAT_DW_BATCH` 1 / 4 (down weight reads a barrier): no gain.
- `FLAT_GU_RIF=1` (one gate/up weight read in flight, not 2): GLM 4x4 M 80 / 160 / 240 -3.2 / -3.4 / -4.0%, the 8 cores
  of a bank share it more evenly (at 2, some cores of a bank waited 300-450 us at M 80, their slot-mates 6-30).
  8 experts a chip and unit splits slower (DSV4.1-Flash up to +7%). Enabled at 16+ experts a chip, whole-tile gate / up
  only (A/B over the 23-case matrix: GLM 4x4 -0.3 .. -4.7%, every other case within +-0.8%).

## custom_mm for the gate/up matmul (full [32, 32] in0 tiles)

Gate/up matmul per core and expert: x [M, 192 K tiles] @ W [192, 2] (one gate/up tile pair a core); K blocks of 8
tiles (24 an expert), MT 4 row tiles a sub-block, DEST passes of rt 2 x ct 2 (fp32 half sync), so matmul_block_k
calls of 2 x 2 x 8. Plain custom_mm extended to full [32, 32] in0 tiles + rt_dim (the commit before this one in
tt-llk / the compute API; bfp in0 needed the SrcB base stepped per tile, not the face counters). Single core,
cycles a tile matmul: 2x2x8 18.14 -> 16.65, 2x2x24 17.85 -> 16.34, bit-identical. In the op the gate/up matmul
runs at ~16.8 (was ~19). Default on (FLAT_GU_CMM=0: off).

| GLM 4x4 | matmul_block_k | custom_mm |
|---|---|---|
| M 80 | 800.0 | 802.1 |
| M 160 | 904.2 | 901.7 |
| M 240 | 987.7 | 979.1 |
| M 320 | 1093.9 | 1066.8 (-2.5%) |
| M 320 zipf | 1265.7 | 1215.2 (-4.0%) |
| M 480 | 1497.6 | 1389.4 (-7.2%) |

GLM 4x4 accuracy + determinism pass. M 320 after it (per-expert probes ZZ-GU-EX/EW/ET): expert 0 107 us (x wait
35, weight wait 27), experts 1-15 55-64 us each with 4-10 us x wait (ideal math ~51). x wait is uniform over the
rectangle (head cores too) and every wait is in pass 0: with row passes an x block is freed only in the last pass,
so the next sub-block's x can only come in behind it and then has to arrive at pass speed (~87 GB/s, the link).
FLAT_GU_FULL_SYNC=1 / FLAT_NO_ROW_PASSES=1 (one pass): 1164.7 / 1202.6 (worse). Next: x blocks of a pass's rows
(x_row_tiles), each pass reading and freeing its own (WIP commit after this one, untested).

## x path split per row pass (x_row_tiles)

With row passes the x of a sub-block now travels as one x sub-block a pass (2 row tiles), and each gate/up pass
frees its own x blocks, so the next x streams in through the whole sub-block. First device runs took the host down
twice: the helper relay measured the x schedule in 128-row sub-blocks (a ~4 GB landing write for an expert's last x
sub-block) - fixed, and the schedule now carries its rows a sub-block; then the host placed helper 1's landing ring in
MT-row super-blocks (every odd super-block read from the wrong place) - fixed. Pieces: a row tile a piece (4 pieces a
2-row block halved the bytes in flight: M 320 1192 vs 1055).

GLM 4x4 with custom_mm, split vs not: M 80 796 / 802, 160 888 / 902, 240 965 / 979, 320 1055 / 1067, 480 1374 / 1389
(about -1%). x wait barely moved (122 vs 129 us at M 320): the 40-core west snake is now the x limit (its primary
waits 213 us on slots; the east primary 304 us on data, its helpers held by the west's landing credits). Tried, not
kept: FLAT_X_FWD_DEPTH=7, FLAT_EVEN_RECTS=1 (M 320 1059 / 1075). FLAT_ISO=1 (no gate/up weight reads) 943 at M 320:
the weight traffic (DRAM + its NoC contention with the snake) is what is left there. FLAT_X_NOC1=1: 1357.

Full matrix (watcher + NoC sanitizer: 28 accuracy + 28 determinism pass), vs before custom_mm: GLM 4x4 M 320 1093 ->
1055 (-3.5%), M 480 1498 -> 1376 (-8.2%), zipf -4.5%; GLM 8x4 hot -10.1%, DSV4.1-Flash hot -10.6%; worst +1.6%
(DSV4.1-Flash 5120 uniform), +0.7% GLM-5.3-Flash.

## DRAM timeline (zz_hist.hpp: bytes landed a reader in 12 us bins, FLAT_ZZ_PROBES=1)

Per bin: gate/up weights by bank, down weights, x reads (helpers), y writes (bfp8 tiles, down_core_writer). Bytes a
call: gate/up 226.5 MB + down 113.2 MB (42.5 MB on every bank: even) + x 16 / 32 / 48 / 64 MB + y 10 / 18 / 27 / 35 MB
(M 80 / 160 / 240 / 320). (The status table's "w+x+y" counted y as bf16: it is bfp8 here, ~half.)

| M | reads end | last y | steady DRAM | gate/up MM / span | x wait | floor (bytes / 510) |
|---|---|---|---|---|---|---|
| 80 | ~750 us | ~800 | 490-520 GB/s | 744 / 751 | 353 | 717 |
| 160 | ~800 | ~898 | ~486 | 835 / 842 | 284 | 766 |
| 240 | ~886 | ~995 | ~459 | 922 / 930 | 170 | 813 |
| 320 | ~970 | ~1117 | ~420 (350-480) | 1036 / 1046 | 70 | 861 |

- M 80: DRAM saturated in steady state; the loss is the end: banks finish their gate/up reads 704-740 us (service, not
  bytes), the last expert's down weights land in a late burst, then its gate/up + down + y (~50 us).
- M 320: gate/up compute-bound (the math busy 1036 of its 1046 us; 66048 tiles a core = 885 us at 18.1 cycles a
  tile), weight ring full, DRAM idle ~15%. Loss: in-op math (~19 vs 18.1 cycles), x wait 70 (startup), weight wait
  32, down after the last h ~50.
- Every gate/up / down weight byte is read once and the rings run full: the DRAM side has nothing left to fix at
  M >= 240; at M 80-160 only the end-of-call drain (~50-80 us).
- `FLAT_DOWN_RING` 3 / 4 (experts in the down ring, default 2): M 80 817.9 / 795.4 (default 801.0), M 160 914.6 /
  903.8 (902.7). Not kept.

## Status (GLM-5.3, 16 experts a chip, uniform, at `789fcaa2888`)

Math peak 4096 FLOP / cycle / core (LoFi tile matmul in 16 cycles), DRAM 512 GB/s nominal; weights bfp4 340 MB a
call, x / y bf16 rows.

| M | us | math util (96) | math util (110) | DRAM weights | DRAM w+x+y |
|---|---|---|---|---|---|
| 80 | 823.8 | 22.7% | 19.8% | 80.5% | 88.2% |
| 160 | 928.7 | 40.2% | 35.1% | 71.4% | 85.0% |
| 240 | 1031.8 | 54.1% | 47.2% | 64.3% | 82.6% |
| 320 | 1095.7 | 67.7% | 59.1% | 60.6% | 83.4% |
| 480 | 1509.7 | 73.5% | 64.1% | 44.0% | 68.7% |
| 640 | 1980.2 | 74.5% | 65.0% | 33.5% | 58.7% |

M <= 320 runs at ~430-465 GB/s of DRAM, above that gate/up-math-bound. CORRECTION: the 465 GB/s "pure-read ceiling"
used earlier was wrong. Measured on this p150b (perf_microbenchmark/8_dram_adjacent_core_read, one reader a bank at
its optimal core, 53 / 107 MB, device-profiler kernel span): 508.5 / 510.4 GB/s, ~99.7% of 512. The op's ~465 GB/s
at M 80 is 9% under the real ceiling. M 480: ideal math
1292 + in-op slowdown ~60 + x wait 71 + weight wait 32 + down tail ~45 = 1510.

## Resolved: GLM 4x4 M 80 "determinism" failure (a test bug, not the op)

`test_flat_expert_determinism[glm53-4x4-2560-uniform]` failed after the other glm53-4x4 cases (and with
`FLAT_GU_RIF=1` alone). A per-row dump (a scratch test running 10240 first, then 2560 six times) showed every
differing row at or past its expert's token count, only on call 0, never a token row. Those padding rows of the
last row tile are written (y is bfp8 tiles in the standalone op), but computed from x rows the x reader never loads
(it skips rows past the count), i.e. from whatever the x ring held: earlier experts' rows (the same every call - the
diffs start at the running max of the earlier counts) or the previous program's (call 0 only). Rows stay independent
through the matmuls and bfp8's per-row shared exponents, so token rows are unaffected. The test compared whole row
tiles on the stale assumption "padding rows are zero" (the first commit's message says the writer skips them: that is
the row-major y path, not this one). Fix: the determinism test compares the token rows only. With it,
glm53-4x4 determinism (incl. M 80 / M 480) passes at defaults and with `FLAT_GU_RIF=1`.

## Next

- gate/up in-op math 19.4 -> 18.1 cycles a tile (~7% at large M)
- x startup ramp (~50 us at the snake tail)
- remove measurement switches before any PR
