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
| (next) | 1 gate/up weight read in flight at 16+ experts a chip, whole-tile gate / up (was 2); determinism test compares token rows only | 1096 -> 1093 (-0.3%); M 80 / 160 / 240 / 480 -2.9 / -2.7 / -4.7 / -0.8%, zipf -1.6% |

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

M <= 320 DRAM-bound (~430-450 GB/s: the lab's pure-read figure is 465), above gate/up-math-bound. M 480: ideal math
1292 + in-op slowdown ~60 + x wait 71 + weight wait 32 + down tail ~45 = 1510.

## Resolved: GLM 4x4 M 80 "determinism" failure (a test bug, not the op)

`test_flat_expert_determinism[glm53-4x4-2560-uniform]` failed after the other glm53-4x4 cases (and with
`FLAT_GU_RIF=1` alone). A per-row dump (a scratch test running 10240 first, then 2560 six times) showed every
differing row at or past its expert's token count, only on call 0, never a token row. y is row major and its
writer (y_row_major.hpp) skips rows past an expert's count, even inside the last row tile, so those rows hold whatever
the y allocation held. The test compared whole row tiles on the stale assumption "padding rows are zero"; the 10240
cases just happened to find matching DRAM there. Fix: the determinism test compares the token rows only. With it,
glm53-4x4 determinism (incl. M 80 / M 480) passes at defaults and with `FLAT_GU_RIF=1`.

## Next

- gate/up in-op math 19.4 -> 18.1 cycles a tile (~7% at large M)
- x startup ramp (~50 us at the snake tail)
- remove measurement switches before any PR
