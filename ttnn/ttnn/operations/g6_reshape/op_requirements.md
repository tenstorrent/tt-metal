# Operation Requirements: g6_reshape

## Definition
- **Formula**: `out[j_0..j_{m-1}] = x[i_0..i_{n-1}]` whenever `lin(j; e) = lin(i; d)`, `lin(i; d) = Σ_k i_k · Π_{l>k} d_l`; one `-1` in `shape` resolves to `N / Π(other entries)`. Bit-identical, no arithmetic; output in the input's dtype, layout and memory configuration.
- **PyTorch Reference**: `torch.reshape(input_tensor, tuple(shape))` (`eval/golden_tests/g6_reshape/helpers.py:pytorch_g6_reshape`)
- **Import Path**: `from ttnn.operations.g6_reshape import g6_reshape`
- **Function Signature**: `g6_reshape(input_tensor: ttnn.Tensor, shape: list[int] | tuple[int, ...]) -> ttnn.Tensor` (rank 1..4 on both sides; no `memory_config`, `dtype` or `compute_kernel_config` parameter, per spec)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Golden-run caveat (harness, not op)**: under the graded default `EVAL_CAPTURE_L1=1`, the bf16 trace loose case and `test_regression.py::test_validation_errors[*]` fail for any op (program cache disabled per test; no L1 peak for a test that raises before dispatch). See `verification_report.md` H1/H2. Judge those cells with `EVAL_CAPTURE_L1=0`; every other cell must pass in the graded run.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED shape-derived axes**: none tagged. Every rank 1..4 on both sides and any last two dims; regime chosen on the host: `tile_face_row` when both W are multiples of 16, else `tile_element_run`.
- **SUPPORTED op-specific axes**: `shape` as list or tuple, one `-1` allowed; DRAM interleaved only (placement is not a TARGET axis)
- **Cores**: full grid, `P = min(110, G)` output atoms, 2 movers per core (mover 0: RISCV_1/NoC0, mover 1: RISCV_0/NoC1); one block per mover on every golden shape
- **Compute config**: none (no compute kernel; the spec forbids `compute_kernel_config`)
- **Golden baseline**: 21 / 21 supported cells pass with `EVAL_CAPTURE_L1=0`. Graded default run: 20 pass, plus the trace case (harness H1). 32 `xfail_expected`, 0 drift, 0 wrong-mode.

### [ ] Refinement 1 — Integer ROW_MAJOR regime and 4-byte dtypes (int32, uint32, float32; ROW_MAJOR)

**Goal**: add `ttnn.int32`, `ttnn.uint32`, `ttnn.float32` to `SUPPORTED["dtype"]` and `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]`. Build the design's deferred `rm_row_gather` regime (`op_design.md` → Regimes) natively in the mover skeleton: page = one logical row; atom = one output page slot (`align(W_out·e, A)` bytes) or one `A`-byte chunk of a wide row; atoms folded **bank-affine** so a mover's block is consecutive slots of one DRAM bank and stores as one write per (block, bank). It must handle both directions the golden set calls: row → column (`[1,W]→[W,1]`, one input page fans out to `W` output pages) and column → row (`[8,1]→[8]`, `[8,1]→[1,1,1,8]`). It must also handle rank-1 inputs (`[16]→[16,1]`, `[8]→[8,1]`). Cells moved xfail → pass (32): the 20 int32/uint32 ROW_MAJOR golden cells, the 5 fp32 TILE golden cells (FLUX `[1,W]→[1,1,W,1]`), and the 7 loose cases. The loose cases are the flagged `[1,4096]→[4096,1]` int32 (plain and `int32_extremes`), its `[-1,1]` variant, the int32 `[16]` sequence, uint32 high bit, fp32 special values, and fp32 NaN-poisoned padding.

**Implementation skill**: /memory-layouts, /interleaved-parallel, /numeric-formats-metal

**Verifier notes**:
- This refinement lands the whole perf-1 contract (int32, ROW_MAJOR, DRAM interleaved, exact gate). It is also the whole TARGET − SUPPORTED gap, so it is the only generality refinement.
- The 4-byte TILE work is SUPPORTED + tests only. `probes/probe_007.py` (`validate()` bypassed) ran 10/10 bit-exact on the current tile kernel: all FLUX fp32 cells, fp32 special values, fp32 NaN poison, int32 extremes, uint32 high bit (`e = 4` → `F = 64`, `atom_rows = 1`). Don't touch the tile regimes for fp32.
- Adding int32/uint32 and ROW_MAJOR per axis also claims `(bfloat16, ROW_MAJOR)` and `(int32|uint32, TILE)`, which no golden cell covers. int32/uint32 TILE already work (same probe). Make the RM regime `e`-generic so bf16 RM falls out, and cover each of those cells with one acceptance case. Use `EXCLUSIONS` only for a concrete kernel gap, named in the changelog.
- Spec rule: a dtype lands with its production layout in the same change (int32/uint32 with ROW_MAJOR, float32 with TILE), and golden cells of each new value must run, not be refused.
- `/numeric-formats-metal`: use only its dtype-widening and precision-matrix test contract. Do **not** add `compute_kernel_config`: the spec forbids it, and there is no compute kernel or DEST.
- Native only. The spec bans `ttnn.to_layout`/`tilize`/`untilize`/`pad`/`slice` at the entry point, and any wrapper is a second dispatch (BB-01). This refinement has no partial-tick wrapper escape.
- Build it performantly. Refinements 2–3 tune exactly this path. Use the full grid by the blocking decision: the flagged input is one 16 KB page, and the spec says the core count must not follow the shape. Per block: one covering read, one write per (block, bank), one read barrier, one write barrier at the end. The Phase 0 profile (`verification_report.md`) shows the mover's per-unit RISC-V work dominates, about 250 ns per output face-row. Keep per-element work in the RM permute to a strided copy, without per-element 64-bit math or per-element plan entries.
- Knobs are module-wide. If the RM regime wants different `MAX_CORES`/`MOVERS_PER_CORE` than the tile regimes, make them per-regime. Keep them in `_knobs()` (the cache key).

**Done when**:
- All 32 currently-`xfail_expected` cells pass bit-exactly in the graded golden run. Phase 0's 21 supported cells are unchanged (the trace cell judged per the caveat above).
- `verify_supported` shows 0 `xpass_drift` and 0 `xfail_wrong_mode`.
- `describe_plan()` reports `regime == "rm_row_gather"` with `num_cores > 1` and `l1_bytes_per_core ≤ 1347776` for `[1,4096]→[4096,1]` int32. The acceptance test pins this.
- The acceptance suite passes in plain and `--dev` mode.
- The flagged shape's device-ns is recorded in `changelog.md` as the baseline for Refinement 2.

### [ ] Refinement 2 — Speed up the perf-flagged int32 row-to-column profile (knob co-tune)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags int32 ROW_MAJOR DRAM interleaved `[1, 4096] → [4096, 1]` (Qwen #529) as the mandatory perf target, with gate `exact`. The lead's goal: faster than the incumbent (362.12 µs) and toward 1.35× of the hardware bound. Optimize this shape with the cheap T1/T2 levers in `ttnn/ttnn/operations/examples/master.md`. Co-tune the knobs the planner exposed on the RM regime: `MAX_CORES`, `MOVERS_PER_CORE`, `MAX_BLOCK_ATOMS`, `STAGE_DEPTH`. This is the design's grid-synchronization lamp: at full grid a mover owns about 19 output pages, so launch, RT-arg delivery and one DRAM round trip per mover may dominate. Also co-tune per-mover NoC (`noc_placement`; mover 1 currently reads on NoC1), bank start order (`bank_stagger`), and covering-read granule (`read_coalescing`). Several T1/T2 levers fit in one phase. No SUPPORTED change.

**Done when**:
- Measured device-ns on `[1,4096]→[4096,1]` int32 improves beyond the 2–3% noise band over Refinement 1's recorded baseline.
- The exact gate holds, including the `int32_extremes` loose variant, and the golden suite is green.
- No regression across the config-spanning guard set (measurable cells > 32 tiles): bf16 `tile_face_row` `(16,1,5120)→(1,1,16,5120)` and `(1,1,16,4608)→(16,1,4608)`, int32 RM `(1,3844)→(3844,1)`. `tile_element_run` (every cell ≤ 32 tiles) is guarded by the golden suite only.

### [ ] Refinement 3 — Speed up the perf-flagged profile, phase 2 (per-unit mover cost)

**Type**: perf

**Goal**: same flagged shape, int32 ROW_MAJOR `[1, 4096] → [4096, 1]`, exact gate, incumbent 362.12 µs. Attack the mover's per-unit RISC-V cost: the walk, the plan entries written to L1 and read back, the CPU copy, the image staging. Phase 0 measured about 250 ns per unit, which dominated DRAM await on large cells. Candidate levers, one T3 per phase, chosen by which zone Refinement 2's profile leaves on the critical path:
- the design's "store realization" lamp: write from staging, skipping the permute/image where congruence allows;
- the "copy engine" lamp: NoC self-copies for 16-B-congruent moves, or the idle TRISCs as copiers inside the same program;
- the "mover roles" lamp: one RISC gathers and awaits, the other permutes and stores, handing off through a CB pair;
- common-RT-args-only, with movers deriving ranges from the logical core index;
- trid sub-batched overlap (`trid_double_issue`).

Catalog: `ttnn/ttnn/operations/examples/master.md` (`split_reader`, `trid_double_issue`, `tile_reorder`). No SUPPORTED change.

**Done when**:
- Measured device-ns on the flagged shape improves beyond noise over Refinement 2's result.
- The exact gate holds and the golden suite is green.
- No regression across the guard set named in Refinement 2.
- If Refinement 2's result is already within 3% of the floor measured on this board (an empty-kernel program on the same core set plus one DRAM round trip), record that measurement in `changelog.md` and close `[x]` with no code change.
