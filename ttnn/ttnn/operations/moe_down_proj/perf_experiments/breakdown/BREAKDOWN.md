# Perf 1 measured breakdown: moe_down_proj t4096_layer0, model L1 occupancy (853,568 B free), L1 x / L1 y,
bfloat8_b, fp32_dest_acc_en=True, LoFi, math_approx_mode=False

Device: blackhole p100a, AICLK 1350 MHz, L layout (12 n-lines x 9 m-groups = 108 cores, core_n_tiles 2).
Blocking at this occupancy: 2 waves, k_block_tiles 6 (16 K rounds per wave, 32 x steps + 32 w steps),
core_row_tiles_max 15. x step = 15 rows x 6 k x 1088 B = 97.9 KB, sent twice by the step's rotating sender
(xa row multicast to the 11-core rect row + xb copy to the group's strip core). w step = in-wave experts x
6 k x 2 n tiles from the line's rotating sender (staged DRAM prefetch), one multicast down the line.

## Measurement mechanics
- Bench: `tests/ttnn/unit_tests/operations/moe_down_proj/probes/test_perf_bench.py` (cases: focus, guard set).
- `bench.sh` (this dir): `MOE_CASES=a,b [MOE_KERNEL_DIR=<dir with a full kernels/ copy>] [MOE_DEFINES=A,B]
  [MOE_CHECK=0] ./bench.sh` -> `<case> <DEVICE KERNEL DURATION ns>`. One fresh-cache run per case.
- `ablate.sh` + `kernels_ablate/`: ABL_NO_COMPUTE (matmul SKIP_COMPUTE), ABL_NO_X (no x L1 reads, 1-tile x
  sends), ABL_NO_STRIP (1-tile xb send), ABL_NO_W (no w DRAM reads, 1-tile w sends), ABL_NO_STORE (no y writes).
  Sync scaffolding (CB ops, handshakes, trip counts) kept.
- `zones.py <profile_log_device.csv>`: per (zone, RISC) per-core totals (max / p50) and exec counts.
- Profiler output goes under $TT_METAL_HOME (/localdev/kgrujcic/tt-metal/generated/profiler/...).
- Instrumentation cost under the profiler: the permanent zones add ~4-6 us to the focus number (orig kernels
  172.1 us, instrumented 178.1 us). Compare instrumented to instrumented only.

## Whole-op (instrumented baseline, DEVICE KERNEL DURATION us)
focus 178.1 | t4096_l3_occ 177.0 | t4096_l2_occ 177.1 | t4096_l0 (unoccupied, 2 waves k=8) 170.5 |
t3072 138.2 | t2048 111.1 | t1024 86.5 | t4096_dram_x 210.4 | t4096_dram_out 209.5 | t1024_dram_both 114.0 |
t37 85.0 | t1 41.4

## Zones (focus, per-core totals us, max / p50; max markers 158 < 250, zones cover the kernel span)
BRISC (w channel + L1 y store) kernel 177.0 / 173.6: w_receive 114 / 61 (29 exec), w_reserve 67 / 50
  (back-pressure from compute), w_send 14 / 9.4 (4 exec), w_send_barrier 5.2 / 1.2, y_store 40 / 36.8 (9 exec)
NCRISC (x channel) kernel 156.9 / 155.0: x_receive 142 / 120 (32 exec), x_reserve 57.7 / 2.0, x_send 17 / 15.3
  (3 exec, ~5 us per send), x_send_barrier 0.9 / 0.4 (x L1 prefetch is ready in time)
TRISC0 (unpack) kernel 162.9 / 160.7: compute_accumulate 156 / 153, of which compute_wait_inputs 87.6 / 60.0
  (32 exec) -> compute is starved ~60 us; payload ~93 us; compute_untilize 5.5 / 5.5 (2 exec)
TRISC1/2: compute_accumulate ~153-156, untilize 5-6.6.
BRISC ends ~14 us after TRISC: the last wave's store is the exposed tail.

## Cumulative ablation (us)
full 178.3
noC (no matmul)                  168.0   compute hidden behind DM (-10)
noC + noW                        165.4   w data costs little on top (-2.6)
noC + noW + noX(+strip)           82.9   x bytes + x L1 reads: -82.5 (the x chain)
sync (+ no store)                 70.5   store -12.4; sync floor 70.5 us = 2.2 us per step of handshakes
solo: noX 149.9 (-28), noW 174.5 (-4), noStrip 160.3 (-18), noStore 163.2 (-15), noXnoW 127.9, noC_noX 130.2

## Ranking (headroom)
1. x chain (NCRISC): ~82 us of DM above the sync floor, on the critical path (compute waits 60 us). Strip copy
   alone -18 us solo. Ceiling: per m-group 3.1 MB per channel over the run; at ~80 GB/s per row link ~39 us,
   so far from the link roofline -> latency / serialization bound (5 us per 98 KB send, rotating senders but
   steps serialized by the ack-then-wait handshake).
2. Sync floor 70.5 us: 64 channel steps whose rendezvous are serialized (receiver acks step s only after
   receiving s - 1). Not a roofline; a protocol latency.
3. Compute payload ~93 us vs ~68 us ideal (5760 tile products per core at 16 cycles, LoFi). The floor once DM
   is fixed; target 90.8 needs it.
4. Store tail ~12-15 us (last wave's untilize + store exposed).
5. w channel: ~3-4 us solo, not binding. DRAM w roofline 22.5 MB / 448 GB/s = 50 us, well hidden.

## Correction (from the early_ack_handshake subagent)
Item 2 is not handshake latency. In the fully stubbed run the matmul scaffolding (SKIP_COMPUTE keeps unpack /
pack, about 57 us), the untilize and the store's wait on compute fill the 70.5 us, and NCRISC ends at about 50 us.
Pipelining the acks measured flat (see changelog Perf 1).
