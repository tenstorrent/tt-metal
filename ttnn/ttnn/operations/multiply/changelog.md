# multiply changelog

## Phase 0 (ttnn-implementer)

Built as designed (op_design.md), no deviation from the binding parts:

- One `ttnn.ProgramSpec` per call via `ttnn.generic_op`; regime chosen on host by `select_regime`
  (`flat_stream` for broadcast `none`, `row_bcast_strips` for `over_h`), passed as CT arg `regime`.
- Knobs, each defined once in `multiply_program_artifacts.py`: `THREAD_PLAN = (1, 1, 1)`, `STREAM_DEPTH = 4`,
  `B_WINDOW_DEPTH = 2`, `L1_RESERVE_BYTES = 0`; per call `row_groups` (G) and `block_cols`
  (`min(Wt, max_units_per_core, block_cols_cap)`). Ring capacities derive from them; the block decomposition
  lives in one header (`kernels/multiply_block_walker.hpp`) used by all three kernels.
- Reader/writer implicit sync (tagged transfers, tag handed back at kernel end); the `row_bcast_strips` b
  window is explicit sync with user trid 0, nominal `block_cols` windows, listed in the reader's
  `disable_dfb_implicit_sync_for`.
- Compute: `compute_kernel_lib::mul` (Scalar/PerTile streaming; `BroadcastDim::Row` + `InputTileMapping::Row`
  on the held window, caller wait/pop of `block_cols`). No raw LLK.
- Shape-dependent values (G, Ht, Wt, b's image strides) are common runtime args, so cells of one regime share
  compiled kernels; `block_cols` and the plan are CT args as the design says.

Structural counts (ttsim: no timings are performance results): every tensor crosses the DRAM/L1 boundary
once, except b in `row_bcast_strips`, which crosses G times (G = 2 only for `[128,512]x[1,512]` among INPUTS);
one helper call per block; blocks per core = 1 on the YOLO shapes (`[1,4,8400]`: 9 or 8 columns per
cluster, 32 clusters).

Tests: acceptance 13/13; `test_multiply_regimes.py` 17/17 (regime pinning incl. G = 2, multi-block and
short-block runs, leading-dim broadcast of b, distinct-column broadcast, flat_stream at plans (2,2,2),
(4,4,2), (1,4,1), one device session with a tilize before and an untilize after, R9 placement, X1/R2
refusals); golden PHASE0 slice 99/99 (incl. numeric regression and X1), 40 sampled out-of-phase cells xfail.

## Phase 0 — Core Implementation (verified)
- **Date**: 2026-11-21
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier).
- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[DRAM, L1],
  sharding=[interleaved], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  broadcast=[none, over_h], output_dtype=[None, bfloat16]
- **Accuracy achieved**: bit-exact (0 ulp, 100% of elements) on 6 shapes; vs the float64 product
  PCC ≥ 0.9999986, max_abs_err ≤ 0.0312, rel_rms_err ≤ 1.66e-3 (the one bf16 rounding), got/true ratio
  median 1.0000 (measured via test_multiply_precision_baseline.py)
- **Golden suite at Phase 0**: 98 / 98 supported cells passing (92 PHASE0 + 6 numeric regression), X1 passing,
  1540 refused as expected; 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`)
- **Issues encountered** (verifier fixes, no drift):
  - the compute config now reaches the kernel whole: `math_approx_mode` → `sfpu_precision_mode`,
    `dst_full_sync_en` → `double_buffer_dest` (it was dropped);
  - `block_cols_cap` sizes each ring with its own page size (it used the output's for all three, wrong once
    dtypes differ); `l1_ledger.md` updated.
- **Tests added**: test_multiply_precision_baseline.py; `test_compute_config_passed_through` in
  test_multiply_regimes.py
- **Queue**: op_requirements.md — R1 ROW_MAJOR, R2 broadcast completeness, R3 perf (YOLO first-output latency),
  R4 numeric (float32, fp32_dest_acc_en=False, float32 output), R5 perf (leading-dim b reuse), R6 perf
  (per-block fixed costs); perf gates are structural counts on ttsim.

## Refinement 0 — A fused SiLU on the first operand, `memory_config`, width-sharded operands (Qwen)
- Date: 2026-11-21
- What was done:
  - Signature equals the specification's stub: keyword-only `memory_config=None` and
    `input_tensor_a_activations=None` (between `dtype` and `compute_kernel_config`). validate() derives the
    `memory_argument` and `a_activation` axes with the specification's own `canonical_memory` /
    `canonical_activation` (anything else reads `"other"` and is refused by name). SUPPORTED gained
    `memory_argument: [None, "input"]`, `a_activation: [None, "silu"]`, `sharding: [..., "width"]`.
  - SiLU: a compile-time switch (`a_activation`) of the one compute kernel in every regime. The block operation
    `multiply::product` is the FPU multiply chain without it (unchanged, still 0 ulp), and with it
    `CopyTile a→D0`, `b→D1` (`CopyTile`, or `UnaryBcast<Row>` for over_h), `silu_tile(D0)`, SFPU
    `MulBinary(D0,D1)→D0`, pack — one pass over the tile. Chose the LLK's `silu_tile` (one SFPU pass, x kept in
    a register: no second DEST slot or second unpack of a) over `sigmoid_tile` + multiply; the SFPU product
    over the FPU's DEST-reuse multiply (no srcB broadcast there, and it would round silu(a) to 19 bits). The
    chain has no SiLU element, so `Silu` is written on the library's `UnaryOp` base (no raw LLK around the
    chain).
  - Width sharding: new regime `sharded_local` (native, zero copy). a, b and out are DFBs borrowed from their
    resident shards (`dfb_spec_from_sharded_tensor(..., borrowed_from=...)`); the reader only credits a and b
    (one reserve/push each, explicit sync), the writer only retires out (one wait/pop); no NoC transfer. The
    shard is the per-core block: compute walks it whole (one chain call for `none`; one per image for
    `over_h`, b's row picked by `TileAddressing::Offset` = `img_b(n)·shard_cols`, so a leading-dim broadcast of
    b works with no re-read). `width_shards()` checks the two operands share the shard grid, orientation and
    shard width (refused by name otherwise) and that each shard fits one DFB ring.
  - Output allocated with a's memory config (R9's interleaved placement, or a's width sharding);
    `memory_config=` can only name that same config.
  - Reused: the regime structure, the reader/writer/compute files, the walker, the `a_stream` / out specs, the
    host plan. Added: one regime branch per kernel, the `product` dispatcher, `width_shards()`, three CT/RT args
    (`a_activation`, `b_bcast_row`, `b_tiles`).
- Accuracy achieved: without SiLU 0 ulp (bit-exact) in every regime incl. sharded; with SiLU PCC ≥ 0.999996,
  relative RMS ≈ 0.0029 on [1,1,128,512], [1,1,32,256] (2-core [32,128] shards and 8-core), [1,4,400]×[1,1,400],
  [2,4,128,512]×[2,4,1,512] (limit 0.995 / 0.04); ~62% of SiLU elements bit-equal to the float64 reference
  rounded to bf16.
- Golden test progress: all 500 of this refinement's in-SUPPORTED cells pass (every cell with `a_activation=silu`,
  `memory_argument=input` or `sharding=width` at bf16 / fp32 DEST / TILE / broadcast none|over_h / output
  None|bf16; run as node-id slices), plus 12 sampled prior-phase cells; over_hw cells of the slice correctly
  xfail. Full suite left to the harness.
- Issues encountered: the runner's `width_shards` puts [1,1,32,256] on 8 cores; the model's 2-core [32,128]
  config is built explicitly in the model-call test. Thread plan stays (1, 1, 1) as designed; with the SiLU a
  tile carries two SFPU passes, so a compute-bound plan (C = 4 on flat_stream) is a perf candidate for a
  later round (finding, not measured here).
- Tests added: `test_multiply_qwen.py` — the two model calls (values vs silu(a)·b in float64, placement),
  `sharded_local` pinning with/without SiLU on flat, over_h, leading-dim-broadcast b and the 263-column YOLO
  shape, borrowed-ring check, SiLU in both interleaved regimes, `memory_config=` equivalence, refusals
  (activation, memory config, mismatched shards), one device session (program before, to_layout after).
