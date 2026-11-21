# Verification Report: rotary_embedding_llama

Target: Quasar (`quasar`), backend: ttsim simulator. Specification: `eval/specs/rotary_embedding_llama/spec.py`, sha256 `92e48fc9c621`. Phase 0 = PHASE0: bfloat16, `fp32_dest_acc_en=True`, TILE, L1 height-sharded, decode mode, one user, both alignments.

## Code Review

**Fixed in place**
- **R1 refusal encoded ahead of the axis.** Added `EXCLUSIONS = [{"dtype": float32, "fp32_dest_acc_en": False}]`, the place the conventions give for the native refusal (`eval/CONVENTIONS.md`: `fp32 + acc=False` is refused op-side through `EXCLUSIONS`, not `INVALID`). It does nothing while float32 is outside `SUPPORTED["dtype"]` and starts refusing the moment Refinement 1 adds float32, so R1 cannot be forgotten. The golden categories did not change (re-run below).
- **Dead block constant removed.** `CONTRACTION_TILES = 1` was declared in the artifacts module and read nowhere. T is one tile, so no loop runs over the contraction. A comment replaces it, so the constant does not read as a knob.

**Checked, no change needed**
- **Specification rules.**
  - R2: a real `matmul_tiles(x, T)`, with no assumption about the matrix.
  - R3: `BroadcastDim::Row` on cos and sin in the chain.
  - R4: the `is_decode_mode` axis comes from the argument, and False is refused.
  - R5: `default_compute_kernel_config()` is the only place the default lives.
  - R6: `validate()` reads neither fidelity nor approx mode.
  - R7: the three fields are read as attributes and rebuilt into a `ComputeConfigDescriptor`, and the Metal 2.0 `ComputeHardwareConfig` is built from them with the fidelity always set.
  - R8: `memory_axes` gating, plus a companion check on dtype, layout, placement and shard grid (`NotImplementedError`).
  - R9: the output is allocated from `input_tensor.spec`, so its memory config compares `==`.
  - R10: covered.
  - REJECTIONS X1–X10: all 10 pass.
- **One native dispatch.** `_dispatch` issues exactly one `ttnn.generic_op` with one `ProgramSpec`. There is no host-side tensor manipulation.
- **Kernel hygiene.** `void kernel_main()`, `api/...` include paths, no `InterleavedAddrGen` (no NoC access at all).
- **Ring credit balance.**
  - Reader: pushes Wt on each STRIDED ring and 1 on T.
  - Each NEO: waits for and pops `neo_dim_tiles` (C × share = Wt), and pops T once at the end.
  - `rotated`: push and wait of `neo_dim_tiles` per NEO.
  - Output: Wt pushed by the NEOs and drained by the writer.
- **Helper usage.** The combine phase is one `eltwise_chain`: two row-broadcast FPU multiplies into D0/D1, an SFPU add and a pack, with block-scoped Upfront/AtEnd policies. The matmul is raw LLK. The block is 1×1×`neo_dim_tiles`, so per-tile `matmul_tiles` inside one DEST section is the block's matmul. This is inside the scope boundary and not flagged.
- **Broadcast efficiency.** cos and sin are consumed as row-0 broadcasts straight from their shards. Nothing is replicated or filled.

## Design Conformance

- **Algorithm and pipeline.** These match the design: `publish_shards` → `rotate_block` → `rotate_combine_block` → `release_output`. The reader and writer only post and drain credits on borrowed, zero-copy rings, and they issue no NoC transfer.
- **Work distribution.** `WorkUnitSpec(target_nodes=shard_spec.grid)` engages the shard's cores. It does not assume (0,0) or a device-wide grid.
- **Thread plan, read off the host program.**
  - `THREAD_PLAN = (1, 4, 1)` is a single named constant. The counts reach every kernel as compile-time arguments (`reader_threads`, `compute_threads`, `writer_threads`) and as `KernelSpec(num_threads=…)`.
  - The effective C is clamped so that it divides Wt: 128 → 4, 64 → 2, 32 and 96 → 1.
  - `rotated` is sized per NEO × C. Every DM ring is explicit-sync, credits only, and listed in `disable_dfb_implicit_sync_for`. There are no transfers, so there is no transaction tag to hand back.
  - The design states the bound: compute.
  - All three required tests are present and pass:
    - (a) `test_each_neo_computes_its_own_tiles` at plans (1,4,1) and (1,2,1);
    - (b) `test_thread_clamp`, one case per effective C, plus (1,2,1) with share 2 (strided matmul index k = 1) and (1,1,1) with 4 tiles in one DEST section;
    - (c) `test_session_order`: after another call and `ttnn.experimental.quasar.add`, before `ttnn.to_layout(..., ROW_MAJOR)`, all in one session.
  - Plan alternatives were **not** timed, because ttsim has no timing model. The design's lamp, (1,1,1) against (1,C,1) on a ≤ 4-tile op, stays open for `ttsim-perf`. No `pipeline_logs/model_cycles.jsonl` exists for this phase.
- **Blocking-model fidelity.**
  - `HEAD_ROW_TILES`, `ROTATED_DEPTH_BLOCKS` and `THREAD_PLAN` are defined once in the artifacts module.
  - `tensor_dim_tiles`, `users_per_core`, `neo_dim_tiles` and `rotated`'s `num_entries` are all derived from the shard and those constants.
  - The kernels size their loops from compile-time arguments, and `section_tiles = min(neo_dim_tiles, DEST_AUTO_LIMIT)` is derived in the kernel.
  - No literal block value is duplicated across host and kernel.
  - The per-entry credit loops in the reader and writer follow the DM STRIDED share rule (1 entry per call; design, Mechanism caps). They move no bytes, so they are not an expression failure.
  - Compute waits, reserves, pushes and pops at block scope. The tile loop inside `rotate_block` sits within one DEST section, with no per-tile completion boundary.
- **Sharded path.** It is native: the shards are consumed in place through `dfb_spec_from_sharded_tensor(..., borrowed_from=…)`, never re-read through a `TensorAccessor`.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification. No axis function is defined in the op file.
- `SUPPORTED` equals PHASE0 (contract lint: PASSED, 0 errors).
- `EXCLUSIONS` holds the R1 entry (above).
- `validate()` checks, in order: the shape rejections (ValueError), then the per-axis SUPPORTED check (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the R8 companion placement (`NotImplementedError`).
- The public entry point calls `validate()` first.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed: xpass_drift = 0.
- **INVALID audit (specification).**
  - `INVALID = []` is consistent: TARGET has one layout (TILE), one placement (L1 height) and no block-float dtype, so no cell is structurally impossible.
  - The canonical bf8b + ROW_MAJOR entry is not applicable: neither value is in TARGET, and Quasar has no bfloat8_b path at all (target page).
  - Not a norm-like op, so there are no weight canonicalization cells.
  - `python3 -m eval.validate_contract rotary_embedding_llama --phase0 --text`: PASSED, 0 errors. The 7 items it reports are all `UNRESOLVED` markers in the draft specification, not defects.

## L1 Ledger Audit

1. **Currency.** Every ring the code declares has a ledger row, and every row is a live ring. `rotated` = `neo_dim_tiles × C × ROTATED_DEPTH_BLOCKS` pages in the input dtype, which matches the code.
2. **Capacity against live set.**
   - The borrowed rings *are* the shards, and their live set is the whole shard, resident for the whole program.
   - `rotated` holds exactly one block per NEO, which is the live set between two sequential phases. No over-capacity.
   - Axis accounting spans `head_dim_tile` with capacity that scales with `neo_dim_tiles`. No under-capacity.
3. **Page format against DEST width.** Finding, folded into Refinement 1. `rotated` is a 16-bit page while `fp32_dest_acc_en=True`, the *under* direction, which truncates the accumulation at the phase boundary. The precision baseline puts a number on the cost: rel RMS 0.0024 against the exact reference versus 0.0017 against a "rotated rounded to bf16" model, so about 30 % of the error. PCC stays at 0.999997, and the cell passes its 0.9997 entry. Refinement 1 must retry a Float32 `rotated`.
4. **Disjoint lifetimes.** `rotated` is the only ring the op allocates, so there is no pair to justify.
5. **Bounds.** Wt ≤ 4 (X4 and X7), `head_row_tiles` = 1 (X6), users = 1, C ∈ {1, 2, 4}. All symbols are bounded and the total is in closed form.

- **Data-movement budget.** Present and consistent with the code: 0 DRAM crossings and 0 cross-core bytes for every tensor. Core-local unpacks: x twice, cos, sin and T once per tile, plus the `rotated` round trip.
- **Cheapest-traffic split.** j dealt over the core's NEOs, in place, which is what the code implements.
- **Block-size defaults.** These hold: sharded means one block = the whole resident shard, and each NEO's block is its whole share.
- **Per-core footprint (own allocation).** `own_L1 = tensor_dim_tiles × tile_bytes(input dtype)` ≤ 8 KiB (bf16, Wt = 4), or 16 KiB once float32 is supported. It scales with Wt only, not with C.

## Precision Baseline

`tests/ttnn/unit_tests/operations/rotary_embedding_llama/test_rotary_embedding_llama_precision_baseline.py`. Settings: bf16, default config (HiFi4, fp32 DEST), standard-normal x, cos, sin and a dense random T, seed 1234. Raw rows: `precision_baseline.json` in this directory. The reference is the specification's float64 reference on the bf16-stored inputs.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max / mean ULP (bf16) | got/true ratio median [p5, p95] |
|-------|-----|-------------|--------------|------------------|-----------------------|----------------------------------|
| (1,1,1,32) | 0.999998 | 0.024 | 0.0043 | 0.0021 | 1.6 / 0.44 | 1.0006 [0.9962, 1.0027] |
| (1,1,8,64) | 0.999998 | 0.071 | 0.0069 | 0.0021 | 34.8 / 0.52 | 0.9999 [0.9962, 1.0045] |
| (1,1,12,96) | 0.999997 | 0.081 | 0.0070 | 0.0023 | 45.3 / 0.57 | 1.0000 [0.9962, 1.0037] |
| (1,1,8,128) | 0.999997 | 0.204 | 0.0076 | 0.0025 | 152.7 / 0.73 | 1.0000 [0.9962, 1.0035] |
| (1,1,32,128) | 0.999997 | 0.226 | 0.0076 | 0.0024 | 369.7 / 0.66 | 1.0001 [0.9963, 1.0039] |

**Assessment.** This is ordinary rounding.
- The ratio is centred on 1.0 with a ±0.4 % spread, so there is no scale or structural signature.
- The large max-ULP values are at cancellation points, where `x·cos ≈ −rotated·sin` and the reference is near zero. Mean ULP stays below 1.
- Against a reference that rounds `x @ T` to bf16 before the multiply, the error falls to rel RMS 0.0016–0.0017 and max ULP ≤ 51. The bf16 `rotated` page is therefore the largest single contributor (ledger finding 3).
- The error grows slightly with head_dim, which follows the larger values of 32-term dense sums.

**Recommended tolerances.** The defaults hold. `(bfloat16, True)` is graded by the resolved entry, PCC ≥ 0.9997, with about 30× margin on 1 − PCC. The `(bfloat16, False)` (0.999, 0.05) and float32 (0.999, 0.01) cells are not measurable until Refinement 1.

## Verifier CLI Summary

Golden suite: `eval/eval_test_runner.sh eval/golden_tests/rotary_embedding_llama/`, 8 sim workers. Result: 28/65 passed, 0 failed, 0 hangs, 1 skipped (the empty `LOOSE_CASES` placeholder). It was run before and after the fixes with identical categories. Report: `verifier_report.json` in this directory.

- supported_pass: 18 (12 cells + 6 numeric regression cases: small/large magnitude and uniform, on the two traced shapes)
- xfail_expected: 36 (12 × `(bf16, False)`, 12 × `(float32, True)`, 12 × `(float32, False)`)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 11 (10 rejection tests, all passed, and the loose-case placeholder)
- Queue coverage (`--queue op_requirements.md`): 2 values in TARGET − SUPPORTED, 0 gaps.

Unit tests: `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rotary_embedding_llama/`, 24/24 passed. That is 4 acceptance tests, 15 thread-plan / session-order tests and 5 precision-baseline tests.

## Recommendations

- **Queue.** Refinement 1 is the only generality refinement: float32 and a 16-bit DEST, bundled, plus a retry of the Float32 `rotated` page. Refinements 2 and 3 are perf, gated on structural counts because ttsim has no clock: redundant format reconfigs per NEO per call, and NEOs at work for head_dim 96. There is no perf-flagged `LOOSE_CASE`, so Refinement 2 targets the two traced calls.
- **Not filed, for a timed backend.**
  - The design's thread-plan lamp, (1,1,1) against (1,4,1) on a 4-tile op. Turning it is a one-constant change.
  - The DEST-fused combine as an alternative to the `rotated` L1 round trip. The catalog predicts a loss for an FPU consumer.

  Both need model cycles (`ttsim-perf`). No structural count settles them on this backend.
- **Memory pressure.** None. The op's own L1 is ≤ 16 KiB at the largest TARGET shape, so no OOM is reachable inside X4, X6 and X7.
