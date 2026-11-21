# to_memory_config changelog

## Phase 0 (quasar, ttsim)

Built as op_design.md says: one reader + one writer kernel per engaged cluster, one staging ring
`dfb_pages` with implicit sync on both ends, regimes R-in / R-out / R-bal as core tables over the
same kernels, `block_pages` derived once in `ring_geometry` (whole per-core assignment when it fits
the ring budget and the 255-entries-per-txn-id cap, else `BUFFER_DEPTH = 2` blocks).

- Thread plan `(1, 0, 1)`: no compute kernel (the op computes nothing; R1 forbids unpacking bytes).
  `kernels/to_memory_config_compute.cpp` is therefore not written. The `(2, 0, 2)` plan is exercised by
  `test_thread_plan_knob` and passes (DM->DM STRIDED ring with 2 harts per side, 16 pages per core).
- The design's risk "DM->DM ring with implicit sync on both ends is unexercised" did not bite: no
  explicit-sync fallback was needed.
- Results: acceptance 55/55; own regime tests 47/47 (reshard 2->1 and 1->2 in both layouts, COL_MAJOR
  grid with a partial last shard, depth-2 multi-block path, one device session followed by
  `ttnn.to_layout`); golden suite 560 passed, 0 failed, 0 hangs.
- Deferred as in the design: R-borrow (ring backed on the shard, removing the core-local staging copy
  on the shard side) and R-bank (L1-interleaved bank-owner split; no Python binding maps an L1 bank to
  its core). Structural counts today: DRAM crossings 1 per tensor side in DRAM; mesh hops per page
  1 (R-in / R-out), 2 when an L1-interleaved side is in R-bal; one core-local ring copy per page on
  the shard side. No timing is claimed on this backend.
