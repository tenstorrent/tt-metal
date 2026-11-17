# conv2d_3x3 changelog

## Phase 0 — regime R0 (interleaved-slab)

Built as designed: M split row-major over the live grid (11x10 on the measured p150b), one M-block per
core on every acceptance shape, K walked per kh block (`bk = tensor_kkh_tiles`, solved down to 24 of 48
tiles for the deep-K case), per-core weight stream, fp32 packer-L1 accumulation, row-broadcast bias.
CB set, sizes and data-movement budget match `l1_ledger.md` unchanged.

Departures from the design text (advisory items only):
- The slab fill's page runs use `num_contiguous_pages` only when `contiguous_page_stride() == 1`. On an
  interleaved accessor a "run" is pages `p, p + num_banks, ...` (same bank, not consecutive pixels), so
  a contiguous slab destination can take only one page per read there.
- Self-aimed gather/zero reads use `noc_async_read(get_noc_addr(l1), ...)` with the local_copy helpers'
  trid calls for fencing; per-trid outstanding reads are throttled below `NOC_MAX_TRANSACTION_ID_COUNT`.

### Measured (Blackhole p150b, HiFi4, bf16, with bias; DEVICE KERNEL DURATION)

| Shape | cores | kernel | weights stubbed | binding stage |
|-------|-------|--------|-----------------|---------------|
| 32x64x128 -> 128 | 64 | 97.7 us | 24.8 us | weight stream: 18.4 MB at ~190 GB/s (DRAM peak) |
| 16x16x64 -> 128 | 8 | 16.2 us | 10.8 us | mixed |
| 8x16x512 -> 256 | 4 | 126 us | 68.8 us | weights + compute (4 cores) |
| 8x16x2048 -> 256 | 4 | 468 us | 258 us | compute (4608 tile-MACs/core at HiFi4) + weights |
| 64x128x128 -> 128 | 86 | 134 us | 80 us | slab fill + weights |
| 128x256x64 -> 64 | 103 | 225 us | 212 us | slab fill: 835 x 128 B DRAM reads/core (halo 2.6x) |

Slab-fill classification: sourcing the same reads from local L1 takes 128x256x64 from 212 to 88 us
(weights stubbed), so the fill is DRAM-transaction-bound in aggregate, not RISC-issue-bound.

Lamps measured (kept at design defaults):
- P9 bank stagger (`SLAB_READ_STAGGER_PX_PER_CORE`, parked at 0): flat on every shape.
- P2 `im2col_rm_depth_rows = block_m_tiles`: flat (225 -> 218 us at best).
- P3 `core_m_tiles x2`: 32x64x128 97 -> 66 us, 64x128x128 134 -> 119 us, but 128x256x64 225 -> 242 us
  and the 4-core shapes slower. Weight-bound shapes want R1 (weight multicast), not fewer cores.

Next levers by measured payoff: R1 weight multicast (weight-bound shapes), R2/R3 sharded input (page
runs coalesce; removes the per-pixel DRAM read), P4/R6 for the few-core deep-K shapes.
