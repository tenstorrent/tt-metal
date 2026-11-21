# clone changelog

## Phase 0 (quasar, ttsim)

Built as designed (`op_design.md`, regime `interleaved_page_stream`); the L1 ledger's inventory holds
unchanged (one ring, `staged_pages`, `RING_ENTRIES_PER_THREAD · R` entries of `entry_bytes`).

- Host: `clone_program_artifacts.py` (Metal 2.0 `ProgramSpec`, one `ttnn.generic_op`). Knobs single-sourced:
  `THREAD_PLAN = (1, 1)`, `RING_ENTRIES_PER_THREAD = 4`, grid queried; `entry_bytes` derived; pages dealt
  to clusters in whole rounds of R (`split_pages`).
- Kernels: `clone_reader.cpp`, `clone_writer.cpp`, implicit sync at both ends, trid handed back last.
  No compute kernel (design: `compute_passthrough` rejected), so no `clone_compute.cpp`.
- Structural counts (the perf measure on this backend; time gains unconfirmed until a timed backend):
  1 tagged read + 1 tagged write per page, 0 barriers / reserve / push in either kernel, each tensor
  crosses its tier once, clusters engaged = min(32, tensor_pages) (e.g. 32 for every traced YOLOv8 call).
- Tests: acceptance 25/25; golden suite 202 passed, 1000 strict xfails (float32, output_memory None/DRAM),
  0 failures; `tests/.../clone/test_clone_plan.py` 23/23 (every hart ran at R = 1, 2, 3 on one cluster and
  the full grid with a sentinel-prefilled output; tt-metal program before and after clone in one session;
  R3 placement, R5 fresh buffer, X1/X2/R7/R8 refusals).
