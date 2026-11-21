# softmax changelog

## Phase 0 (ttnn-implementer)

Built: `dim=-2`, bfloat16, TILE, rank 4, DRAM or L1 interleaved, every alignment (SUPPORTED = PHASE0),
regime `column_resident`, thread plan `(R, C, W) = (2, 4, 2)` (`THREAD_PLAN`, one host constant).

- Host: `softmax_program_artifacts.py` (Metal 2.0 `ProgramSpec`, one `ttnn.generic_op`). Column-units
  `nc·Wt + wt` dealt in whole rounds of C to clusters (`split_work_to_cores`), padded last round recomputes
  NEO 0's column. `select_blocking()` picks `block_cols` / `in_depth` from the ring cap and the L1 budget.
- Reader: implicit-sync tagged reads, explicit-sync H-mask ring from local stores (`fill_l1_range`), tag handed back.
- Compute: three block operations in raw SFPU (justification at the kernel head; same helper rejections as the
  design), stats in Float32 rings read through SrcA (`UnpackMode.UnpackToSrc`, required explicitly by the host
  API for an FP32 ring under 32-bit DEST). SFPU inits once per block operation (LLK evidence in the kernel).
- Writer: implicit-sync tagged writes, tag handed back.

Deviations from the design (advisory parts only):
- `cb_h_mask` is always allocated (8 KB per cluster) and only filled/read when H is ragged (ledger updated).
- The design's thread-plan session test used `ttnn.add` as the preceding program; `ttnn.add` is Metal 1 and does
  not run on Quasar, so the test uses a device-side `ttnn.to_layout` tilize instead.

Structural counts (ttsim has no clock; no timing is claimed):
- input DRAM crossings 1, output 1 (+ at most C−1 recomputed columns per op);
- clusters engaged on `[1,4,16,8400]`: 32 of 32, 4 NEOs each, 8–9 columns per NEO, one block per NEO;
- SFPU inits per input tile: 0 (5 per block operation per block).

Tests: `test_softmax.py` (acceptance, 7/7), `test_softmax_regimes.py` (24: multi-block ragged, block_cols knob,
tall column, non-monotonic max fold, refusal, LoFi+approx masked, R1/R2/R8/R10, X1, every-NEO-ran,
plans (1,1,1)…(4,4,2), device session before/after tt-metal programs). Golden first-phase slice: 48 passed,
6 expected refusals (float32 `dim=-1` numeric cases).
