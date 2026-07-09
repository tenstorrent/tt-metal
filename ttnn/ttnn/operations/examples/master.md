# Performance examples — catalog

Short, self-contained, **runnable** ops that each isolate one or two kernel-level performance
concepts and **measure** them on device (real `ns`, never a claimed speedup). Use them to learn a
pattern, then re-measure on your own shapes with each example's CLI / test.

**Reading order:** this file → the example's `README.md` → read the code and run the test only if
you need to. For the ⭐ Starter examples, the *gist* below is often enough to act immediately.

**Difficulty:**
- **⭐ T1 Starter** — one knob/placement decision, no kernel restructure. Actionable from the gist.
- **⭐⭐ T2 Intermediate** — a CB-sizing / transfer-shape / kernel change. Read the README.
- **⭐⭐⭐ T3 Advanced** — kernel restructure, overlap scheduling, mcast / semaphores. Read the code.

Every number below is stamped in that example's `report.md` with the box + arch it was measured
on. They are illustrative of the *effect*, not CI bounds.

---

## ⭐ T1 — [`reader_placement`](reader_placement/README.md)
**Concept:** core placement → DRAM-traffic (read+write) NoC contention.
**Situation:** you spread a *line* of reader/worker cores over an interleaved-DRAM tensor (grid-
filling copies, or the reader line of an mcast) and it is slower than it should be.
**Measured win:** placing the line as a **row** or **diagonal** instead of a **column** is
**~2.2–2.8× faster** (Wormhole B0, 4–8 cores); the gap grows with core count. A column line
saturates its shared NoC links and stops scaling.
**Gist:** a column line is what `split_work_to_cores(..., row_wise=False)` (the **default**) gives
you — pass **`row_wise=True`** to spread across the DRAM-facing axis. (Diagonal only beats row on
asymmetric grids like Blackhole.)

## ⭐⭐ T2 — [`tile_reorder`](tile_reorder/README.md)
**Concept:** transfer coalescing on a DRAM-bandwidth-bound move.
**Situation:** a whole-tile relocation (permute / transpose-of-tiles) written the generic way —
as many small sub-tile (face) writes with a barrier each.
**Measured win:** relocating each **whole 2 KB tile in one NoC write** is at least as fast as, and
on this move faster than, writing it as 4 × 512 B faces — bigger coalesced transactions hit higher
achieved DRAM bandwidth. Reader on NoC0, writer on NoC1 to overlap.
**Gist:** on a DRAM-bound move, move whole pages and batch barriers; don't scatter sub-tile faces.
