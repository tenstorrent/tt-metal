# rm_ring_bank_run (Perf 17): ROW_MAJOR ring slab + bank-run fill, overlapped with compute

Blackhole p150b, 110 cores, `l1_small_size=65536`, bf16 / HiFi4 / fp32 DEST / no approx / bias, padding (0, 0, 1, 1),
DRAM-interleaved ROW_MAJOR input. Device kernel time (`DEVICE KERNEL DURATION`), one run per variant unless noted.
Every ring output is bit-identical to `op_base` (sha1 of the output bytes; PCC vs torch 0.9999985 on every case).

## What it does
On the K-blocked tap-direct walk over a DRAM-interleaved ROW_MAJOR input with one pixel per page, `cb_halo_slab` becomes
a ring of consecutive input pixels across a core's M-blocks (`kernels/conv2d_3x3_rm_ring.hpp`):
- each input pixel is read from DRAM once per core (the per-M-block slab re-reads every band, ~3x where bm * 32 < W);
- the fill is bank-run: chunks of `RM_RING_CHUNK_BYTES` (8 KB) consecutive pixels, one DRAM read per bank per chunk,
  into a dedicated 2-chunk staging CB (`CB_RM_STAGE` = 16, counted in the L1 solve), de-interleaved into the ring with
  self-aimed local NoC reads (the graduated bank-run fill's mechanism);
- the next block's pixels are filled while compute works (the reader runs fill steps while the tap-direct unit FIFO
  is full: `TapDirectFifo::reserve(group, units_left, idle)`), and ring slots are freed K-block by K-block from the FIFO's
  progress (no end-of-block wait);
- the host solve (`select_blocking`) keeps the slab solve's bk (bit-identity) and weight residency and takes the
  largest bm whose ring + staging fit; `rm_ring_plan` mirrors the reader's release / band-wait sequence to size the
  ring (`sim_ring.py` checks it on the focus cases and on odd shapes: 0 violations). Not taken where the slab solve runs
  one M-block per core (nothing is re-read).

## Results (base -> ring, us)
| idx | case | calls | base | ring | change | bm base -> ring |
|---|---|---|---|---|---|---|
| 41 | P1 1026x2048 32->128 | 5 | 2607.6 | 1882.3 | -27.8% | 46 -> 46 |
| 35 | P3 1026x2048 128->3 | 1 | 5291.5 | 2459.4 | -53.5% | 32 -> 7 |
| 44 | P5 1026x2048 64->128 | 1 | 3886.2 | 2946.1 | -24.2% | 34 -> 24 |
| 46 | P6 1026x2048 64->256 | 1 | 5883.7 | 5347.9 | -9.1% | 19 -> 10 |
| 33 | P9 514x1024 128->3 | 1 | 1408.6 | 802.3 | -43.0% | 30 -> 30 |
| 36 | P4 514x1024 32->128 | 5 | 729.6 | 609.2 | -16.5% | 50 -> 50 |
| 42 | P2 514x1024 64->256 | 6 | 1618.2 | 1400.2 | -13.5% | 19 -> 15 |
| 39 | P11 514x1024 64->128 | 1 | 1077.4 | 826.7 | -23.3% | 30 -> 30 |
| 40 | P14 258x512 128->256 | 1 | 837.3 | 753.9 | -10.0% | 10 -> 7 |
| 37 | P8 258x512 64->256 | 5 | 512.3 | 464.1 | -9.4% | 19 -> 19 |
| 45 | P12 514x1024 128->256 | 1 | 2806.5 | 2817.5 | +0.4% (median of 3) | 10 -> 2 |
| 47 | P13 514x1024 128->512 (streamed weights) | 1 | 6391.6 | 6490.8 | +1.55% (median of 3) | 9 -> 4 |
| 38 | P10 130x256 128->512 (streamed) | 17 | 444.1 | 441.4 | flat | 5 -> 5 |
| 43 | P7 258x512 128->512 (streamed) | 8 | 1677.8 | 1684.3 | flat | 8 -> 8 |
| 34 | P15 66x128 128->512 | 10 | 139.8 | 140.2 | flat (no ring: 1 M-block/core) | 3 |

Call-weighted over the FLUX.2 bh2x4 decode: about -10.9 ms. Raw logs: `runs/fin1..fin4`, `runs/rc1` (3-run rechecks).

## Menu (same precision contract, all bit-identical)
| variant | P3 (idx35) | P1 (idx41) | note |
|---|---|---|---|
| base (op_base) | 5226 / 5291 | 2609 | slab per M-block, serialized bank-run fill |
| **ring** (2 x 8 KB staging, overlapped) | 2468 / 2459 (bm 7) | 1876 / 1882 | recommended |
| serial (ring, fill only at band waits) | 2790 | 2028 | `RM_RING_OVERLAP=False` |
| 3 x 8 KB staging | 2492 (bm 6) | 1896 | |
| 4 x 8 KB staging | 2621 (bm 5) | 1877 | |
| 2 x 16 KB staging | 2547 (bm 5) | 1885 | |
| 4 x 16 KB staging | 2811 (bm 3) | 1891 | |
| band-stream fill order (each block's 3 bands first) | 2446 | 1866 | flat (P9 819 vs 817, P5 2915 vs 2933); code removed |
| ring with streamed weights (bigger bm) | 2432 (bm 9) | 3293 | P1 +75%; code removed |
| guard `RM_RING_MIN_BM_FRAC=0.5` (unless C_out <= 64) | = ring | = ring | only P12 / P13 fall back to the slab (flat there); off by default |
| diagnostic: ring, no DRAM reads | 2123 | 1854 | wrong output |
| diagnostic: ring, no fill at all | 2031 | 1813 | compute floor at that bm (base fill-stubbed P1: 1805) |

P1 now sits on its compute floor. P3's remainder is the bm 32 -> 7 compute cost (~300 us, the ring holds 2 image rows of
256 B pixels: 1.08 MB) plus ~440 us of exposed fill (DRAM ~350, de-interleave ~90); a band-ordered first fill did not
move it, so it is bandwidth, not startup.

## Domain
Applies to every DRAM-interleaved ROW_MAJOR input with one pixel per page (C_in a multiple of 32) on the K-blocked
tap-direct walk with more than one M-block per core, resident or streamed weights. Measured exceptions beyond the noise
band: none. P13 (streamed weights, bm 9 -> 4) is +1.55% in 3 tight runs (inside the band, but consistent): the ring's L1
halves bm and doubles the weight stream; the `RM_RING_MIN_BM_FRAC` guard is the option if that is to be avoided.
Not reached (feasibility, by construction, unchanged programs): TILE input, sharded input (Panoptic), L1-interleaved input,
C_in not a multiple of 32, K split, slab-direct / face-direct / row-gather walks, one M-block per core (idx 31, 32, 34).
Untested: dilation > 1, stride 2, asymmetric padding on device (host plan `sim_ring.py` covers them: 0 violations).

## Files
- `op_base/` untouched copy of the working-tree op; `op_copy/` the candidate; `integration.patch` (`make_patch.sh`:
  op_base -> op_copy, profiler zone-hash pad lines stripped; applies to the real op with `git apply`).
- `test_rmb.py` + `run.sh` / `run_wait.sh` (`run_wait.sh <budget_s> <label> <cases> <variants>`; signposts
  `cmp|<case>|<variant>|<i>`, parsed by `asym_padding/cmp_windows.py`).
- `l1_plan.py` (host-only blocking / ring / fill pixels per core), `sim_ring.py` (ring-plan safety), `zone_pad.py`
  (16-bit profiler zone-hash collisions between op_base and op_copy: pads op_copy files).
- Raw NoC: the de-interleave uses `ncrisc_noc_read_set_state` / `ncrisc_noc_read_with_state` (capability:
  `dataflow_kernel_lib::read_with_state` fixes the source address, the de-interleave changes both), as the graduated
  bank-run fill. The FIFO's idle hook polls `cb_pages_reservable_at_back` (no "reserve or do other work" in the CB API).
- `runs/op_copy_menu_snapshot/`: op_copy as measured for the menu (band-stream order, streamed-weight ring, fill ablation knobs), before those were removed.
