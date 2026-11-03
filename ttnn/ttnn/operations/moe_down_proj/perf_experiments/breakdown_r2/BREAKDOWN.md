# Perf 2 measured breakdown: moe_down_proj t4096_layer0 (focus), model L1 occupancy (853,568 B free), L1 x / L1 y,
bfloat8_b, fp32_dest_acc_en=True, LoFi, math_approx_mode=False. HEAD = Perf 1 graduations.

Device: blackhole p100a, 1350 MHz. L layout, 108 cores (11x9 rect + 9 strip), core_n_tiles 2. 2 waves, k_block 6,
15 rows/core, 32 x steps + 32 w steps. x step = 15 x 6 tiles = 97.9 KB; w step <= 8 x 6 x 2 tiles = 104 KB.

Mechanics: `bench.sh` / `ablate.sh` / `zones.py` (copied from ../breakdown), ablation kernels in `kernels_ablate/`
(= HEAD kernels + ABL_NO_COMPUTE / ABL_NO_X / ABL_NO_STRIP / ABL_NO_W / ABL_NO_STORE, sync scaffolding kept).
New permanent zone: compute_wait_x (nested in compute_wait_inputs; wait_w = inputs - x). Max 198 markers/RISC.

## Whole op (instrumented, one fresh-cache run): focus 154.0-155.0 us

## Cumulative ablation (us)
full 154.0 -> noC 145.3 -> +noW 136.9 -> +noX 89.8 -> +noStore 83.4 (all stubbed: compute scaffold floor)
solo: noX 137.6 | noStrip 151.5 | noW 145.0 | noStore 141.2
pairs: noX+noW 106.3 | noX+noW+noStore 100.5 (compute alone) | noC+noStrip 143.6 | noC+noW+noStrip 136.8
       | noW+noStrip 142.8

## Zones (us per core, max / p50)
full:  TRISC0 accumulate 137.9/135.3, wait_inputs 78/49.7 (x 46.7/35.5, w ~14), untilize 5.4.
       NCRISC x_receive 131/106, x_forward (9 cores) 48.7/45.9, x_send 9.8/7.3 (4 sends).
       BRISC w_receive 100/57, w_reserve 48/37.5, y_store 16/15.1 (paced, wave 0), y_store_tail 15/14.1;
       NCRISC y_store_tail_helper 11.8. Kernel ends: TRISC ~146, DM ~152.
noC:   TRISC0 waits 120 (x 91, w 29): the DM chain alone runs ~137-145 us.
compute alone (noX noW noStore): accumulate 89.7 p50 (wait 3.9) -> payload ~86 us; op 100.5 us.

## Reading
1. x channel step chain binds: with compute stubbed and w stubbed, x at 98 KB/step = 136.9 vs 1 tile 89.8 ->
   +47 us = 1.47 us/step SERIAL on top of the per-step scaffold; ~4.2 us/step total. The strip forward is NOT on
   the critical path (noC+noW+noStrip 136.8 = noC+noW 136.9). One xa multicast is in flight per row at a time
   (receiver acks s+1 only after data s lands), so the transfer does not overlap the step.
2. x and w are balanced: removing either alone leaves the other holding the wall (noX 137.6, noW 145.0; both
   106.3). w costs ~31 us once x is gone.
3. Compute payload floor ~86 us accumulate + 5 untilize (compute alone 100.5 us); scaffold without math 83 us.
   Ideal 68 us at 16 cycles/tile product. Target 90.8 needs both DM and compute to move.
4. y store: noStore solo -12.8 us (paced store 15 us on BRISC during wave 1 + tail 14 us on two RISCs; TRISC ends
   ~146, DM ~152 -> ~6-7 us exposed tail).
