# add: changelog

## Phase 0 (ttnn-implementer)

Built regime `stream_flat` of op_design.md as designed: Metal 2.0 ProgramSpec, one `ttnn.generic_op`.

- Work distribution: flat page ranges over the full grid (`split_work_to_cores`, row_wise, in rounds of C
  pages); clusters engaged = min(num_tiles, 32) (tested: 13, 52, 263, 1050 tiles).
- Thread plan (1, 1, 1) from host constant `THREAD_PLAN`; plans (2,2,2), (4,4,2), (1,4,1) tested correct
  with a NaN-prefilled output (a missing thread leaves NaN pages).
- Knobs `BLOCK_TILES`=4, `RING_DEPTH_BLOCKS`=2, `MIN_TILES_PER_CORE`=1 are live and test-turned
  (block 1/2/4 with ragged tails, depth 2/3/4, min tiles 1/4/8).
- Reader/writer: implicit sync, one tagged transfer per page, tag handed back; session test with tt-metal's
  `to_layout` before and after passes.
- Compute: one `compute_kernel_lib::add` call per NEO (PerBlockSize wait/pop/reserve/push, Block mapping).
- L1 inventory and data-movement budget unchanged from `l1_ledger.md` (3 rings, 49,152 B; a, b, out cross
  their tier once each).

Advisory deviation: the host also clamps `block_tiles` to the DEST section size of the call's compute config
(mirrors `dest_helpers.hpp`), so a turned knob cannot trip the kernel's static_assert.

Results: acceptance 7/7; plan tests 22/22; golden suite 87 passed, 0 failed, 0 hangs.
No performance figure: the ttsim backend has no clock (perf is structural here).
