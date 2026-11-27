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
> **Golden-run caveat (harness, not op)**: under the graded default `EVAL_CAPTURE_L1=1`, the bf16 trace loose case and `test_regression.py::test_validation_errors[*]` fail for any op. The program cache is disabled per test, and a test that raises before dispatch has no L1 peak. See `verification_report.md` H1/H2. Judge those cells with `EVAL_CAPTURE_L1=0`; every other cell must pass in the graded run.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]. float32 was added by the verifier: the tile kernel moves 4-byte elements bit-exactly, and float32's production layout, TILE, is already supported.
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED shape-derived axes**: none tagged. Ranks 1..4 on both sides, any last two dims. The host picks the regime: `tile_face_row` when both W are multiples of 16, else `tile_element_run`.
- **SUPPORTED op-specific axes**: `shape` as a list or tuple, with one `-1` allowed. DRAM interleaved only; placement is not a TARGET axis.
- **Cores**: full grid, `P = min(110, G)` output atoms, 2 movers per core (mover 0: RISCV_1/NoC0, mover 1: RISCV_0/NoC1). One block per mover on every golden shape.
- **Compute config**: none. There is no compute kernel, and the spec forbids `compute_kernel_config`.
- **Golden baseline**: with `EVAL_CAPTURE_L1=0`, 28 / 28 supported cells pass. The graded default run passes 27, plus the trace case (harness H1). 25 `xfail_expected`, 0 drift, 0 wrong-mode.

### [x] Refinement 1 — Integer ROW_MAJOR regime (int32, uint32; ROW_MAJOR)

**Goal**: add `ttnn.int32` and `ttnn.uint32` to `SUPPORTED["dtype"]` and `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]`, in one change. Build the design's deferred `rm_row_gather` regime (`op_design.md` → Regimes) natively in the mover skeleton:
- page = one logical row;
- atom = one output page slot (`align(W_out·e, A)` bytes), or one `A`-byte chunk of a wide row;
- atoms folded **bank-affine**, so a mover's block is consecutive slots of one DRAM bank and stores as one write per (block, bank).

The regime must handle both directions the golden set calls:
- row → column: `[1,W]→[W,1]` (one input page fans out to `W` output pages), `[8]→[8,1]`, `[16]→[16,1]`, `[1,1,1,32]→[32,1]`;
- column → row and reshape-within-RM: `[8,1]→[8]`, `[8,1]→[1,1,1,8]`, the `[16]→[4,4]` sequence target.

Rank-1 inputs are included. Cells moved xfail → pass (25): 15 int32 and 5 uint32 ROW_MAJOR golden cells, and 5 loose cases. The loose cases are the flagged `[1,4096]→[4096,1]` int32 (plain and `int32_extremes`), its `[-1,1]` variant, the int32 `[16]` sequence, and uint32 high bit.

**Implementation skill**: /memory-layouts, /interleaved-parallel, /numeric-formats-metal

**Verifier notes**:
- This refinement lands the whole perf-1 contract: int32, ROW_MAJOR, DRAM interleaved, exact gate. It is also the whole remaining TARGET − SUPPORTED gap, so it is the only generality refinement left, and Refinements 2–3 are perf.
- Per-axis SUPPORTED makes this also claim `(bfloat16, ROW_MAJOR)`, `(float32, ROW_MAJOR)` and `(int32|uint32, TILE)`. No golden cell covers them.
  - int32/uint32 TILE already work on the current tile kernel: `probes/probe_007.py` with `validate()` bypassed runs 10/10 bit-exact, re-run in the second verifier pass.
  - Make the RM regime `e`-generic (2 and 4 bytes) so bf16 and fp32 RM fall out.
  - Cover each of those cells with one acceptance case. Use `EXCLUSIONS` only for a concrete kernel gap, named in the changelog.
- Spec rule: int32/uint32 land with ROW_MAJOR in the same change, and their golden cells must run, not be refused. Don't split this refinement by dtype.
- `/numeric-formats-metal`: use only its dtype-widening and precision-matrix test contract. Do **not** add `compute_kernel_config`: the spec forbids it, and there is no compute kernel or DEST.
- Native only. The spec bans `ttnn.to_layout`/`tilize`/`untilize`/`pad`/`slice` at the entry point, and any wrapper is a second dispatch (BB-01). This refinement has no partial-tick wrapper escape. The RM regime is one more descriptor chosen on the host. Each call still dispatches exactly one `generic_op`.
- Build it performantly; Refinements 2–3 tune exactly this path.
  - Use the full grid by the blocking decision: the flagged input is one 16 KB page, and the spec says the core count must not follow the shape.
  - Per block: one covering read, one write per (block, bank), one read barrier, and one write barrier at the end.
  - Phase 0's profile (`verification_report.md`) shows the mover's per-unit RISC-V work dominates, about 250 ns per output face-row. Keep per-element work in the RM permute to a strided copy, without per-element 64-bit math or per-element plan entries.
- Knobs are module-wide. If the RM regime wants different `MAX_CORES`/`MOVERS_PER_CORE` than the tile regimes, make them per-regime, and keep them in `_knobs()` (the cache key). Size the RM CBs through the same single-source helper pattern as `_slot_bytes`, and add rows to `l1_ledger.md`.

**Done when**:
- All 25 currently-`xfail_expected` cells pass bit-exactly in the graded golden run. Phase 0's 28 supported cells are unchanged (judge the trace cell per the caveat above).
- `verify_supported` shows 0 `xpass_drift` and 0 `xfail_wrong_mode`.
- `describe_plan()` reports `regime == "rm_row_gather"` with `num_cores > 1` and `l1_bytes_per_core ≤ 1347776` for `[1,4096]→[4096,1]` int32. The acceptance test pins this.
- The acceptance suite passes in plain and `--dev` mode.
- The flagged shape's device-ns is recorded in `changelog.md` as the baseline for Refinement 2.
- **Outcome**: landed. Flagged `[1,4096]→[4096,1]` int32 = 4610 device-ns on 110 cores (`[1,3844]→[3844,1]` 4280 ns), incumbent 362.12 µs. Harness gate run (`eval_test_runner.sh`, graded default): 54/59, 0 hangs; the 5 failures are H1 (trace) + H2 (4 validation tests), as at Phase 0.

### [x] Refinement 2 — Speed up the perf-flagged int32 row-to-column profile (knob co-tune)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags int32 ROW_MAJOR DRAM interleaved `[1, 4096] → [4096, 1]` (Qwen #529) as the mandatory perf target, with gate `exact`. The lead's goal is faster than the incumbent (362.12 µs) and toward 1.35× of the hardware bound. Optimize this exact config with the cheap T1/T2 levers in `ttnn/ttnn/operations/examples/master.md`.
- Co-tune the knobs the planner exposed on the RM regime: `MAX_CORES`, `MOVERS_PER_CORE`, `MAX_BLOCK_ATOMS`. This is the design's grid-synchronization lamp: at full grid a mover owns about 19 output pages, so launch, RT-arg delivery and one DRAM round trip per mover may dominate.
- Co-tune per-mover NoC direction (`noc_placement`; mover 1 currently reads on NoC1), bank start order (`bank_stagger`), and the covering-read granule (`read_coalescing`).

Several T1/T2 levers fit in one phase. No SUPPORTED change.

**Done when**:
- Measured device-ns on `[1,4096]→[4096,1]` int32 improves beyond the 2–3% noise band over Refinement 1's recorded baseline.
- The exact gate holds, including the `int32_extremes` loose variant, and the golden suite is green.
- No regression across the config-spanning guard set (measurable cells > 32 tiles): bf16 `tile_face_row` `(16,1,5120)→(1,1,16,5120)` and `(1,1,16,4608)→(16,1,4608)`, and int32 RM `(1,3844)→(3844,1)`. `tile_element_run` (every cell ≤ 32 tiles, fp32 FLUX included) is guarded by the golden suite only.
- **Outcome**: landed.
  - Flagged `[1,4096]→[4096,1]` int32: ≈ 4300 ns (bank fold, same build; 4610 recorded) → **≈ 2900 ns** (−33%), 110 cores, exact.
  - `[1,3844]→[3844,1]`: ≈ 4250 → ≈ 2900 ns. Tile guards unchanged (8.5–8.7 µs, 6.85 µs).
  - Levers: linear fold with a bank-grouped image (`RM_FOLD`; removes the 8× read amplification on the single hot input bank, still ≤ NB writes per block) and covering reads pre-issued before the walk (await 1.2 µs → 0.25 µs).
  - Co-tune: full grid, 2 movers and one block per mover remain best. Native `MOVER_NOCS` beats swapped by ~6%; it is now a cache-keyed knob.
  - Now bound by the mover's per-unit RISC-V work (walk ≈ 500 ns + permute ≈ 620 ns for 19 atoms) on top of ≈ 500 ns launch. That is Refinement 3's scope, so it was not attempted here.

### [x] Refinement 3 — Speed up the perf-flagged profile, phase 2 (per-unit mover cost)

**Type**: perf

**Goal**: same flagged shape and exact config: int32 ROW_MAJOR `[1, 4096] → [4096, 1]`, exact gate, incumbent 362.12 µs. Attack the mover's per-unit RISC-V cost: the walk, the plan entries written to L1 and read back, the CPU copy, and the image staging. Phase 0 measured about 250 ns per unit, which dominated DRAM await on large cells. Candidate levers, one T3 per phase, chosen by which zone Refinement 2's profile leaves on the critical path:
- the design's "store realization" lamp: write from staging, skipping the permute/image where congruence allows;
- the "copy engine" lamp: NoC self-copies for 16-B-congruent moves, or the idle TRISCs as copiers inside the same program;
- the "mover roles" lamp: one RISC gathers and awaits, the other permutes and stores, handing off through a CB pair;
- common-RT-args-only, with movers deriving ranges from the logical core index;
- overlap: trid sub-batches inside one block (`trid_double_issue`). Alternatively, if a tuned config gives a mover > 1 block, software-pipeline gather(k+1) ahead of permute(k). That is what would make the currently inert `STAGE_DEPTH`/`IMAGE_DEPTH` knobs pay, and it needs a depth-2 plan CB.

Catalog: `ttnn/ttnn/operations/examples/master.md` (`split_reader`, `trid_double_issue`, `tile_reorder`). No SUPPORTED change.

**Done when**:
- Measured device-ns on the flagged shape improves beyond noise over Refinement 2's result.
- The exact gate holds and the golden suite is green.
- No regression across the guard set named in Refinement 2.
- If Refinement 2's result is already within 3% of the floor measured on this board (an empty-kernel program on the same core set plus one DRAM round trip), record that measurement in `changelog.md` and close `[x]` with no code change.
- **Outcome**: landed. Flagged `[1,4096]→[4096,1]` int32 **2915 → ≈ 1880 ns (−36%)**, `[1,3844]→[3844,1]` 2897 → ≈ 1845 ns (−36%), same-session device kernel ns, 110 cores, exact; tile guards unchanged (8.5 µs / 6.8 µs). Not near the floor (empty kernel ≈ 290 ns; R2-structure reads-only ≈ 1600 ns), so code changed. Levers: (1) plan-free linear-fold permute — closed-form copy sources, gather walk and plan CB deleted, whole-page fast path with compile-time copy size, write addresses computed in the read-latency shadow (flagged ≈ −5%, `[1,3844]` −7%: the walk was already overlapped with DRAM, so this mostly cut post-barrier work); (2) shared per-core covering read (`RM_SHARED_READ`): the per-mover zone map showed the await tail is a per-NoC request queue at the single hot input bank (await 90 ns → 1.4 µs by distance; removing NoC1's reads left NoC0's unchanged), so one mover per core reads for both, readers alternating NoCs by checkerboard (55 per NoC instead of 110) — −25%; (3) bank split (`RM_BANK_SPLIT`): the movers take alternate bank groups of the shared range, 4 writes each instead of 8 — −8%. Now bound by: the hot-bank read tail (NoC0 await max ≈ 0.9 µs) + ≈ 0.5 µs of post-barrier copy/fence/write issue + ≈ 0.19 µs write drain. Next would be cutting readers per NoC below 55 (one DRAM read per core pair forwarded L1→L1, or spreading requests across the bank's alternate NoC endpoints) — not tried: it needs a cross-core handoff (remote semaphore + extra L1 hop latency) whose net win against a ~0.3 µs tail reduction is unclear, and it is a separate scheme change.
