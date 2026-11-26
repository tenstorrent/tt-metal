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

## Data movement ablation (FLAT_ISO bits: the transfer skipped, every count / semaphore kept)

Bits: 1 gate/up weight reads, 2 down weight reads, 4 x DRAM reads, 8 x helper -> primary landing, 16 x primary ->
snake head, 32 x snake forwarding, 64 h gate/up -> chain heads, 128 h along the chains, 256 y writes. All bits
together and the on-chip ones (504) run hang-free under the watcher + NoC sanitizer. GLM 4x4 uniform, us:

| removed | M 80 | M 320 | M 480 |
|---|---|---|---|
| nothing | 796.7 | 1054.0 | 1380.2 |
| gate/up weights (1) | 357.9 | 942.6 | 1334.4 |
| down weights (2) | 601.4 | 959.8 | 1345.1 |
| x DRAM (4) | 770.3 | 1000.4 | 1352.8 |
| x landing (8) | 793.6 | 1035.0 | 1362.3 |
| x head (16) | 796.5 | 1038.2 | 1361.1 |
| x snake (32) | 789.4 | 1024.6 | 1354.1 |
| h send (64) | 797.1 | 1055.3 | 1380.3 |
| h chain (128) | 791.2 | 1045.5 | 1351.7 |
| y (256) | 769.2 | 1005.7 | 1357.1 |
| all x (60) | 755.9 | 954.3 | 1342.6 |
| both weights (3) | 334.2 | 935.9 | 1328.1 |
| all h (192) | 789.9 | 1044.7 | 1353.0 |
| all reads (7) | 333.3 | 934.2 | 1325.7 |
| all on-chip (504) | 750.1 | 949.0 | 1337.6 |
| everything (511) | 287.3 | 893.2 | 1280.8 |

Only one movement left (511 minus its bit): gate/up weights 490.9 / 906.5 / 1293.6, x snake 317.1 / 914.4 / 1301.0,
h chain 295.0 / 901.3 / 1287.9, every other one within ~6 us of the floor. Alone almost nothing costs at M 320 (the
snake +21, gate/up weights +13): the 161 us over the floor there is the movements' interaction (gate/up + down weight
reads together, ~110, and the x path beside them). M 80 is the weight bytes (DRAM).

FLAT_DN_NOC_SWAP=1 (down weight reads on NOC1, the h chain / y on NOC0; each RISC still one NOC): M 80 / 320 / 480
857.5 / 1521.9 / 2014.6 - the down weight wait does not move (96.5 vs 97.4 us), the h chain on NOC0 (in NOC1 order)
stalls everything (gate/up h wait 818 us). With the chains ordered by NOC0 hops: 821.7 / 1084.3 / 1469.3, still worse
than 796.6 / 1055.9 / 1364.2. Not kept.

## Preload upper bounds (emulated: FLAT_PRE_W / FLAT_PRE_X, timing only)

Idea: the block's arena persists from moe_head to routed_experts_combine and sparse_dispatch_2d does not use it, so
something running during dispatch could fill the flat expert's rings. Emulation: FLAT_PRE_W=n - the first n loads'
gate/up chunks and down blocks count as landed (not read; still granted / published); FLAT_PRE_X=n - the first n x
sub-blocks count as in every gate/up ring (the relays skip them entirely - not read, tilized, landed or sent - the
forwarders count them arrived and forwarded, the primaries' ring slots start past them). GLM 4x4 uniform, us:

| | M 80 | M 160 | M 320 | M 480 |
|---|---|---|---|---|
| baseline | 797.3 | 885.1 | 1057.5 | 1374.7 |
| PRE_W 2 (the rings' 2 loads, ~42 MB) | 730.8 | 836.1 | 1017.8 | 1329.3 |
| PRE_X 1 / 2 | 795.1 / 793.9 | 895.0 / 895.9 | 1053.2 / 1055.9 | 1378.8 / 1366.2 |
| PRE_X 2 + PRE_W 2 | 792.0 | 894.3 | 1052.6 | 1386.8 |

Weights: -8.3 / -5.5 / -3.8 / -3.3% - worth a dispatch-time prefetch (needs a fixed first-loads order in the
schedule and an arena contract with whatever runs during dispatch). x: nothing, and with the weights preloaded it
takes their gain back (expert 0 then waits ~46 us on weights through the grant / publish handshakes - an emulation
artifact or not, the x path's start is not just the snake's fill: even preloaded, expert 0 waits ~15-18 us for x).

## K split (WIP): gate/up K over partner cores

Prior art: moe_fused_swiglu (main: K over KGROUPS grid rows, gate / up partials reduce-scattered per column on both
NOCs, distributed SwiGLU) and ws_expert (mstaletovic/ep4-glx WORKLOG, 2026-11-16/17: 10 x 2 groups, x only to the
partner, bf16 partials reduce-scattered down the column; fold + SwiGLU ~540 cyc a row, compute ceiling ~76%).
Bench (one core, cyc a 32-row tile of gate/up work, 384 tile matmuls; kernels in the session scratch): today 1 pair x
K 192 (custom_mm 2 x 2) 6173; K split 2 pairs x K 96 (custom_mm 1 x 4) 6164; + the fold of the partner's 2 partial
tiles (dest-reuse add) + re-init + packing 2 partials and 1 h tile 6378 (+3.3%, ~+27 us at M 320). x a core halves
(~33 -> ~16.5 MB a call at M 320), the partial exchange ~0.65 MB a core each way, one hop.

Step 1 (plan + weight layout, FLAT_KS=2; kernels refuse it yet): gate_up_k_split; GLM-5.3 4x4: 32 column groups of 2
pairs x 2 K parts = 64 cores, 4 groups / 8 cores a bank slot, 12 K-blocks a core (blocks of 8 K x 4 tiles), MT 4 in 4
one-row passes. Partners: horizontal neighbours over a rectangle's column pairs, vertical ones in an odd last column
(all 32 one hop apart), placed in a slot together (nearest-first + min-max on pair units). Core j of slot r: group
r * groups a slot + j / 2, K part j % 2. Host check: every core's blocks rebuild its Wg / Wu slice exactly, every (K
block, column) once; KS 1 unchanged (layout identical, GLM 4x4 accuracy / determinism pass).

Step 2 (kernels, FLAT_KS=2, off by default): a snake per K part (the primary sends super-blocks 0-2 of an x
sub-block to part 0's head, 3-5 to part 1's; per head its freed word, ring slots, XARR), 12 blocks an x sub-block a
core; the compute packs the partner's pair (2 bf16 tiles) to partial_out_cb and its own to partial_own_cb after
each one-row pass, the receiver (BRISC) writes the former into the partner's partial_in_cb ring (arrival semaphore
11, slot credits on 13), the compute folds own + partner (add_tiles into DST), SwiGLU, packs the row's h tile; h at
K-tile group x 2 + part. x blocks hold a group of passes' rows (x_row_tiles a multiple of the pass rows: the first
pass of the group waits for them, the last frees them). Every GLM 4x4 case passes accuracy / determinism under the
watcher (PCC unchanged: bf16 partials). Bugs found on the way: a static_assert in an if constexpr of a non-template
function (fires for KS 1 too); a two-row fold reading 4 tiles across the partial ring's end (the unpacker's tile
indices do not wrap): an LLK assert under the watcher (looked like a hang), wrong results without - folds take 2 rows
only from consecutive slots.

GLM 4x4 us, KS 1 vs KS 2 (x blocks of 4 rows, fold in row pairs): M 80 797 / 833, 160 889 / 916, 240 967 / 997, 320
1054 / 1084, 320 zipf 1209 / 1266, 480 1366 / 1461 - worse everywhere. M 320 KS 2: x wait 73 us (KS 1 117), weight
wait 21, the matmul at its ideal rate (821 us), the fold 159 us on MATH (~1.6 us a two-row fold), gate/up span 1017
(KS 1 ~997). Not the fold's reconfigs (-12 us without), not the activation (-11 us skipped), not the exchange's
latency (folding a pass later: no change). Variants on the way: x blocks of 1 row (the x split's default): 1203 us
(the relays move 4x the super-blocks: primaries 478 us issuing + 172 flushing, helpers 415 waiting credits); 2 rows
1100; 4 rows 1088; + folds in row pairs 1084.
Verdict: the K split buys back at most the x wait it removes (~45 us at M 320) and its fold costs more than that;
even a fold at the bench's cost (~25 us a call) would net ~4%, and that needs the own partial kept in DST (folding
before the commit deadlocks the partners, or costs ~6% on the matmul through 1 x 2 calls). Parked behind FLAT_KS=2.

## NoC placement experiments (after the ablation)

GLM 4x4 uniform, us (baseline 796.6 / 1055.9 / 1364.2 at M 80 / 320 / 480):
- FLAT_ISO=508 (only the two weight streams): 721.2 / 939.4 / 1326.8 - the pair alone is +46 over the floor at M 320;
  the rest (~115) is the x path beside the weights (x alone ~+20).
- FLAT_GU_RIF=2: M 320 1112.0 / M 480 1375.5 (more weight reads in flight slow x). FLAT_W_START=4 / 12: noise.
- FLAT_GU_W_NOC1=1 (gate/up weights on the receiver BRISC, NOC1): 1200.6 / 1576.5 / 1899.5 - the helpers' x DRAM reads
  (NOC1) collapse.
- FLAT_XH_SWAP=1 (helper x DRAM reads on NOC0, landing writes on NOC1): 804.0 / 1109.0 / 1457.6; + FLAT_GU_W_NOC1=1:
  805.7 / 1111.6 / 1457.9. The helpers' x reads on NOC0 wait 507 us (165 on NOC1): x supply-limited (primaries wait
  336-370 us for data), whichever NOC the gate/up weights use. Not kept.
The x source (helper DRAM reads) and the x snake each lose to whatever weight stream shares their NOC; no placement
of the existing streams separates them.

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
