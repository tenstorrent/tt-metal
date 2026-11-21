# Verification Report: linear

Target: Quasar (`quasar`), 8×4 clusters; backend: ttsim (functional, untimed). Specification:
`eval/specs/linear/spec.py` (sha256 daf6eedc…). Verified on the Phase 0 build `c23c48263c5`.

## Code Review

Fixed in place:
- **R2: the default compute config was restated outside its factory.** `resolve_compute_config` used
  hard-coded `getattr` fallbacks: `MathFidelity.HiFi4`, `fp32_dest_acc_en=False`, `math_approx_mode=False`.
  The `fp32_dest_acc_en` fallback even disagreed with the factory's `True`. Every fallback now reads
  `default_compute_kernel_config()`, so the default is defined once. (Behaviour is unchanged for every caller
  the suite has: both config types carry all three fields.)
- **A no-op indirection was removed.** `_dm_config(make)` only called `make()`. The data-movement kernels now
  call `E.create_{reader,writer}_datamovement_config()` directly. The implicit-sync note moved to the ring
  declarations, where it applies.

Reviewed, no change needed:
- **Specification rules.**
  - R1: refused before the axis gate.
  - R2: one factory.
  - R3: no gate on fidelity or approximation.
  - R4: fields read as attributes, and a fresh `ComputeConfigDescriptor` is built from them.
  - R5 / R8 / R9 / R12 / R13: axes read with the specification's `memory_axes` and `canonical_*`, then gated.
  - R6: weight and bias must be DRAM-interleaved, TILE, and of `a`'s dtype, with `NotImplementedError`
    naming the parameter.
  - R7: DRAM, the exact passed config, or `memory.place(...)` for `L1/width/auto`.
  - R10: `program_config` and `core_grid` are read only for `fused_activation`.
  - R11: refused.
  - R14: every placement pair goes through the accessors.
  - R15: holds.
  - X1–X7 all raise.
- **One dispatch.** One `ttnn.generic_op` per call; validation, planning and output allocation are host-side.
- **Kernels.**
  - `void kernel_main()`, `api/...` include paths, and `TensorAccessor` throughout.
  - Ring sync balances: compute waits and pops exactly what each reader posts per chunk and per round, and
    pushes `block_h*block_w` per round, which the writers drain in rotation.
  - Each data-movement kernel hands its NoC tag back (`noc_async_{read,write}_set_trid(0, …)`) as its last
    statement. `test_one_device_session` covers this: the op runs after a first call and before `to_layout`.
- **Broadcast.** The bias row is broadcast down DEST by `unary_bcast<ROW>`, with no replicated fill. The reader
  fetches only `block_w` bias tiles per NEO per round.
- **Helpers.** Raw compute API, justified per block operation in op_design.md's "Helpers considered and
  rejected" table and at the head of `linear_compute.cpp`:
  - `matmul_block_helpers.hpp` and `bias_add_helpers.hpp` are not used;
  - the eltwise chain cannot act on a DEST accumulator left by a raw matmul.

  No multicast exists in the built regime, so the `McastArgs` rule does not apply.

## Registry Conformance

- **The four declarations are wired correctly.**
  - `INPUT_TAGGERS = AXES`, imported from the specification; no local axis function.
  - `SUPPORTED` equals `PHASE0` (lint).
  - `EXCLUSIONS = []`.
  - `validate()` runs the malformed-call checks first, then R11 / R1 / R6, then SUPPORTED per axis
    (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`).
  - `linear()` calls `validate()` before any kernel work.
- **INVALID.** The op file declares no `INVALID`.
- **No drift.** No auto-fix to SUPPORTED was needed (0 XPASS).
- **INVALID audit (specification).** The one entry, `{"memory_config": "DRAM", "sharding": "width"}`, couples
  two placement axes of the same tensor (the activation). It is structural: the runner builds no DRAM-sharded
  tensor (CONVENTIONS §2.3). It is well formed.
  - The canonical bf8b + ROW_MAJOR entry is not applicable: TARGET has neither `bfloat8_b` nor ROW_MAJOR, and
    Quasar has no block floats.
  - Not a norm-like op, so no weight canonicalization cells apply.
  - float32 + `fp32_dest_acc_en=False` is correctly a rule (R1, refused) rather than INVALID: it is legal but
    lossy.
- **Spec lint.** `python3 -m eval.validate_contract linear --phase0 --text` passes: 0 errors, and 6 UNRESOLVED
  items that belong to the draft specification, not to the op.

## Design Conformance

- **Algorithm.** The built algorithm matches R-2D-stream: DEST-resident accumulation over the whole of K, the
  bias folded into DEST first, one pack per block, and no product or partial ring.
- **Pipeline, ownership and rings.** As designed. Activation reader: 1 hart → ALL ring. Weight reader: 1 hart →
  BLOCKED weight and bias rings. Compute: C = 4 NEOs. Writer: 2 harts → STRIDED output ring.
- **Work distribution.**
  - Whole rounds are dealt with `split_work_to_cores(row_wise=True)` over the full 32-cluster grid.
  - `select_blocking` puts occupancy first and then the coarsest block. On 32 clusters every Phase 0 shape picks
    1×1 blocks, so each NEO owns exactly one block (the per-core loop runs once), and occupancy is as high as an
    independent-axis split allows. This is not a half-turned knob.
  - The grid is under-filled only where `M_t × N_t` is small: ResNet uses 8/32 clusters (32 output tiles = 32
    NEOs), Llama 1/32. That is the dependent-axis (split-K) case, filed as Refinement 5.
- **Thread plan.** Read off the host program:
  - `THREAD_PLAN = ThreadPlan(1, 1, 4, 2)` is one constant. Its counts reach the kernels as CT args
    (`compute_threads`, `writer_threads`) and `num_threads`.
  - Rings are sized per NEO × C.
  - The tests show every thread ran: `test_every_neo_computes_its_column`, and `test_thread_plans_agree`
    (C = 1 / 2 / 4, bit-identical).
  - The one-session test is present and passes.
  - The design states the op is compute-bound and gives C = 4. R = 2 (rather than the page's 4) has a mechanism
    reason: one producer per matmul operand ring. A second weight hart is a perf lamp → Refinement 4.
  - ttsim is untimed, so no plan-vs-alternative measurement can be taken on this backend, and no
    `model_cycles.jsonl` reference exists for this run.
- **Blocking-model fidelity.** `block_h`, `block_w`, `block_k`, the four `*_DEPTH` constants and C are each
  defined once (`select_blocking`, the constants, `THREAD_PLAN`). Every ring capacity and loop bound derives
  from them as CT args. No ring scales with a whole-op dimension. `test_forced_blocks` exercises non-1×1 blocks
  (4×1, 1×4, 2×2, 4×2, 2×4, with and without bias, both DEST widths).
- **Expression.** Compute waits and pops once per K chunk per ring and packs once per block. The readers and the
  writer use implicit sync: one tagged transfer per entry, no per-unit barrier or handshake. No per-unit
  completion boundary sits inside a planned block.

## L1 Ledger Audit

- **Ledger currency.** The four rows (`cb_act_tiles`, `cb_weight_tiles`, `cb_bias_tiles` when there is a bias,
  `cb_output_tiles`) match the four `DataflowBufferSpec`s. Their capacities match `BlockPlan.*_entries`
  exactly.
- **Size vs live set.** Each capacity is `depth × live set` with depth 2 and a stated double-buffering reason.
  Axis accounting holds:
  - act spans `block_h × block_k` and is not × C (ALL);
  - weight spans C × `block_k × block_w`;
  - output spans C × `block_h × block_w`, with K folded in DEST.

  Nothing is over- or under-sized.
- **Page format vs DEST width.** All pages are bf16. Under fp32 DEST, the bf16 output page is the output ABI
  (R9). Act, weight and bias are dataflow payloads that never pass through DEST. No finding.
- **Disjoint lifetimes.** None: all four rings are live together in the round pipeline, and the ledger records
  why none can share.
- **Bounds.** Every symbol has its bound and the predicate behind it, and the total is closed-form.
- **Block-size defaults.** Held: spread over the full grid first, then the coarsest block. `block_k` is the
  coarsest that fits the ring cap and the budget.
- **Per-core footprint.**
  `tile_bytes × (ACT_DEPTH·block_h·block_k + WEIGHT_DEPTH·C·block_k·block_w + has_bias·BIAS_DEPTH·C·block_w + OUT_DEPTH·C·block_h·block_w)`.
  It scales with `block_k` (act, weight), C (weight, bias, out), and `block_h·block_w ≤ dest_tiles`. ResNet:
  688,128 B of ~4 MiB.
- **Data-movement budget.** Present and consistent with the code: weight × `num_row_blocks`, activation ×
  `rounds_per_row_block`, bias × `num_row_blocks`, output once.
- **Finding:** the cheapest-traffic split (R-mcast-weight + R-mcast-act) is deferred as "unexercised on this
  part". That is a risk statement, not one of the two positive reasons (unreachable shapes, or covered by a
  built regime). The shapes it serves are in Phase 0 (Qwen re-reads the weight 4×; ResNet re-reads the
  activation 8×). **Folded into Refinement 3**, which builds it.

## Precision Baseline

`test_linear_precision_baseline.py`: bf16 standard-normal inputs, the specification's `bias_values`, and the
reference in float64 on the stored inputs. Gate: PCC ≥ 0.99 and relative RMS ≤ 0.15 (`legacy_default`), plus
a got/true median within 1 % of 1. All 9 cases pass.

| Shape (bias) | DEST | PCC | Max Abs Err | Mean Abs Err | Relative RMS | ULP p99 | got/true p5 / median / p95 |
|---|---|---|---|---|---|---|---|
| 1x1x32x64 @ 1x1x64x128 | fp32 | 0.999999 | 0.099 | 0.0089 | 0.0016 | 1 | 0.9973 / 1.0000 / 1.0026 |
| 1x1x32x64 @ 1x1x64x128 | bf16 | 0.999997 | 0.151 | 0.0146 | 0.0026 | 16 | 0.9922 / 0.9999 / 1.0076 |
| 1x1x64x96 @ 96x160 | fp32 | 0.999999 | 0.110 | 0.0111 | 0.0016 | 1 | 0.9974 / 1.0000 / 1.0026 |
| 1x1x64x96 @ 96x160 | bf16 | 0.999995 | 0.200 | 0.0216 | 0.0031 | 16 | 0.9896 / 0.9999 / 1.0100 |
| 1x1x128x256 @ 256x512 (vector) | fp32 | 0.999999 | 0.249 | 0.0253 | 0.0017 | 1 | 0.9974 / 1.0000 / 1.0026 |
| 1x1x128x256 @ 256x512 (vector) | bf16 | 0.999983 | 1.606 | 0.0876 | 0.0058 | 29 | 0.9835 / 1.0000 / 1.0169 |
| 1x1x32x2048 @ 1x1x2048x1024 (row) | fp32 | 0.999999 | 0.500 | 0.0729 | 0.0017 | 1 | 0.9973 / 1.0000 / 1.0027 |
| 1x1x32x2048 @ 1x1x2048x1024 (row) | bf16 | 0.999869 | 10.47 | 0.7016 | 0.0162 | 82 | 0.9509 / 0.9998 / 1.0494 |
| same, LoFi + approx (ResNet's config) | bf16 | 0.999869 | 10.47 | 0.7016 | 0.0162 | 82 | — |

**Assessment.**
- With fp32 DEST the result is the float product rounded once to bf16: ULP p99 = 1 and relative RMS ≈ 0.0017
  at every K.
- With 16-bit DEST the error grows with K (the running sum is rounded per K step). At K = 2048 it is still far
  inside the callers' PCC 0.99 and the `dot` bound (the golden suite passes every such cell).
- The got/true ratio is centred on 1.000 with a symmetric spread, so there is no scale bug.

**Recommended tolerances**: the defaults hold. Every cell's grading entry (`NUMERIC_RESOLVED` `dot`, c = 41,
plus PCC 0.99) is met with large margin.

## Verifier CLI Summary

Golden suite, re-run on the final code (`/tmp/linear_golden_final`, 8 sim workers; copy at
`ttnn/ttnn/operations/linear/verifier_report.json`):
- supported_pass: 331 (324 cells + resnet50_fc_lofi + 6 numeric regression cases)
- xfail_expected: 2916
- invalid_skipped: 1080
- rejections (no axes sidecar): 7 passed
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- Queue check (`--queue op_requirements.md`): 6 TARGET − SUPPORTED values, 0 gaps

Acceptance + plan + precision tests: `tests/ttnn/unit_tests/operations/linear/`, 197 passed.

## Probes (findings that shape the queue)

- `probes/probe_002.py`: SUPPORTED widened to ranks 2 / 3, batch broadcast and both non-aligned values. All 48
  out-of-phase bf16 cells pass (two placements each).
- The same probe filled the tile padding non-zero (`fill_implicit_tile_padding`, 7.0 on `a`, 3.0 on `b`) for
  `1x1x32x50 @ 1x1x50x64`. Max error was 295 against a reference max of 28: **K padding reaches the sum**. Safe
  only for zero-padded inputs → Refinement 1, item 1.
- `probes/probe_003.py`: float32 with SUPPORTED widened. Every call fails at build: an FP32 ring under
  `enable_32_bit_dest` needs an explicit `unpack_modes` entry (`hardware_config.cpp:193`) → Refinement 2.

## Recommendations

- **Queue order.** Shapes (R1) → float32 (R2) → perf R3 (multicast) → R4 (reader knobs) → R5 (split-K).
  - The specification has no perf-flagged loose case, so the perf regions are free-selected on supported bf16
    shapes. Neither generality refinement is a prerequisite for them.
  - R1 precedes R2 so float32 extends the final shape path once.
  - R3 precedes R4 because R4 tunes readers that R3 rewrites.
- **R-out-aligned** (output borrowed from its width shard; deferred in the design): not queued. The output is
  the smallest tensor. Aligning the split to the caller's shard grid would cut ResNet from 8 clusters to 2.
  R7 / R14 require any grid pair, which the accessor path already serves.
- **Activation consumed in place from its width shard**: not queued separately. The shard cuts K, so in-place
  use *is* split-K (Refinement 5). The specification's GUIDANCE sanctions the accessor path for every placement.
- **No L1 pressure.** The worst Phase 0 footprint is 688 KiB of ~4 MiB. float32 doubles `tile_bytes`, and
  `select_blocking` absorbs that through `block_k`.
- **Hardware-only checks** (another backend, not this board): LoFi accuracy at K = 2048 (the specification's
  DECISIONS says it is unmeasured on Quasar), and every time figure.
