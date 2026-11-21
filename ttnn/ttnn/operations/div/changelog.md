# div changelog

## Phase 0 (ttnn-implementer)

- Built per op_design.md: `ttnn.ProgramSpec`, thread plan `THREAD_PLAN = (4, 4, 2)`, `BLOCK_TILES = 4`,
  `BUFFER_DEPTH = 2` (one host constant each; kernels take the counts as CT args).
- Work split: whole rounds of C tiles over the full grid (`split_work_to_cores(..., row_wise=True)`).
- Reader / writer: tagged implicit-sync NoC transfers per ring entry; tag handed back as the last statement.
- Compute: one `eltwise_chain` per block — `CopyTile` a → D0, `CopyTile` b or `FillBitcast` (number) → D1,
  `DivBinary`, `Floor` / `Trunc` (CT `rounding_mode`), `PackTile` into the output dtype. Regime chosen by the
  define `DIVISOR_IS_NUMBER` (a binary holds one chain and names only its bound rings).
- bf16 outputs at 0 ulp: the pack of the fp32 quotient to bf16 met the 0-ulp grade on every golden cell;
  no in-DEST RNE narrowing was needed.
- Results: acceptance 55/55; golden suite 437 passed / 0 failed / 0 hangs (rest refused as out of phase);
  `test_div_thread_plan.py` 26/26 (per-NEO tagging, block sizes 1/3/4/8 × plans (1,1,1)…(4,4,2) with ragged
  blocks, buffer depth 1/3, session ordering incl. `ttnn.to_layout` after div).
- Performance: not measurable on ttsim (no clock). Structural counts: each tensor crosses DRAM once, the number
  0 times; up to 32 clusters × 4 NEOs engaged; one chain call per `BLOCK_TILES` tiles per NEO.
