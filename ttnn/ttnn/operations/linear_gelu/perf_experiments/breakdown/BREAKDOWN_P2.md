# Perf 2 measured breakdown: linear_gelu 8x512 (focus), L1 x / L1 out, W/bias DRAM, bf16, fp32_dest_acc_en=True, HiFi2, approx=False

Device: Blackhole p100a, 110 cores, 1350 MHz. Op at Perf-1 HEAD (load-macro GELU, BLOCK_H_MAX 4).
Busiest core: 13 m tiles x 9 n tiles = 117 tiles; row-blocks 4,4,4,1 (subblock 4x1, tail 1x1), block_w 1, K_ext 25.

Whole op: 94.76 us. Ablation (us): nogelu 89.2, gelu_only (no matmul) 80.3, DM only 46.4, + no W 38.4,
+ no x 38.4, sync floor 23.0, no writes only 94.8 (writer off the critical path).

Critical-core timeline (us from kernel start; TRISC0 = unpack, TRISC1 = math, TRISC2 = pack):
  compute_fill          0.7 ->  8.9   (x rb0: sender reads 96 tiles 3.1 us, x mcast 3.0 us, then W chunk 0)
  rb0 matmul_gelu       8.9 -> 32.5   23.6
  compute_wait_x       32.5 -> 36.8    4.3   <- x row-block 1 not there yet
  rb1                  36.8 -> 60.2   23.4
  rb2                  60.3 -> 84.2   23.9
  rb3 (1 row, 1x1 sb)  84.2 -> 93.3    9.0   (9 tiles: ~1.0 us/tile vs 0.66 us/tile in 4-row blocks)
  pack tail            -> 94.4
Reader (NCRISC) on the same core: w_deliver 7.6 -> 30.1 (22.5 us, 9 chunks ~2.5 us each, serial read+mcast per
round; only round 0 prefetched), THEN x rb1 issue+mcast 30.1 -> 37.0 (6.8 us). The reader is serialized: x rb1
cannot start until all W chunks are delivered, although cb_activation_block (ACT_DEPTH 2) has its slot free.

nogelu timeline: steady rb 22.9 us (vs 23.9 with GELU), rb1 still +4.4 us stall, tail 6.0 us (vs 9.0).
=> steady state: math (~2.54 us / 4x1 subblock) and pack thread (GELU 4 x ~0.6 us + pack) co-bind within ~4%.
   Pure matmul ~34 cycles per 32x32x32 tile-op (HiFi2). Bias is the 25th K tile (4% of matmul work).

Ranked headroom (busiest core): (1) x rb1 starvation 4.3 us, (2) 1-row tail block ~3 us over 4-row rate,
(3) fill 8.2 us (Perf 1 fill_first_block was NULL on many variants), (4) steady math/pack co-bind (~4%).
Zones: max 26 markers/RISC of 250. New permanent zone `compute_wait_x` (TRISC0, rb >= 1).
Timeline script: /tmp/tl.py copy at perf_experiments/breakdown/timeline_p2.py (busiest core per-zone timeline).
