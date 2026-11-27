# Operation Requirements: g6_fill

## Definition
- **Formula**: `e = cast_T(v)`; `out[i] = e` for every logical position `i` of `input_tensor` (rank 1, `[N]`). float32: rounded once from the exact value, ties to even. int32: exact. No input value is read.
- **PyTorch Reference**: `torch.full_like(x, v)` (in place: `x.fill_(v)`); oracle `eval/golden_tests/g6_fill/helpers.py:pytorch_g6_fill`
- **Import Path**: `from ttnn.operations.g6_fill import g6_fill`
- **Function Signature**: `g6_fill(input_tensor: ttnn.Tensor, fill_value: float | int, *, output_tensor: ttnn.Tensor) -> ttnn.Tensor` (returns `output_tensor`)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {w_non_aligned, h_non_aligned} (both values a rank-1 tensor can produce)
- **SUPPORTED op-specific axes**: none
- **Regime**: R1 tile row-0 segment scatter. One writer kernel (BRISC), two 64-B writes per tile, one barrier per core. The fill row is built in `cb_fill_row` (64 B), the default because the graded L1 capture refuses a zero peak.
- **Cores**: `w_tiles` split across `min(ceil(N/32), grid)` cores, row-wise (16 / 8 / 7 / 7 / 6 cores on the production shapes)
- **Compute config**: none (no compute kernel; the contract forbids `compute_kernel_config`)
- **Golden baseline**: graded run 15 supported_pass + 3 xfail_expected + 1 supported_fail. The fail is the trace case `loose6`, caused by the harness: L1 capture disables the program cache (verification_report.md, H1). With `EVAL_CAPTURE_L1=0`: 16 / 16 supported pass.

### [ ] Refinement 1 — int32 and ROW_MAJOR (regime R2, row-major chunk scatter)

**Goal**: add `ttnn.int32` to `SUPPORTED["dtype"]` and `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` by building regime R2 of `op_design.md` natively in the existing writer kernel.
- A ROW_MAJOR `[N]` tensor is one page of `4N` bytes.
- The scatter unit is a chunk of `rm_chunk_bytes = min(align_up(4N, 16), RM_CHUNK_CAP_BYTES = 2048)` bytes at offset `k * rm_chunk_bytes` of page 0. The last chunk's length is ragged.
- The chunks are split row-wise across `min(num_rm_chunks, max_fill_cores)` cores by the same `split_work_to_cores` plumbing R1 uses.
- `cb_fill_row` is one page of `rm_chunk_bytes`, in the output dtype. The regime predicate is `layout == ROW_MAJOR`, exact and host-side. `g6_fill_plan()` must report `R2_rm_chunks` with `chunks_per_core` (BB-08).
- R1 already moves 32-bit patterns, so `int32 + TILE` runs on R1 unchanged.
- R2 writes the RM page directly: no tilize stage, no `ttnn.to_layout`, `tilize`, `untilize`, `pad` or `slice` anywhere (the prompt rule forbids them).

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- This is the only TARGET − SUPPORTED gap, so it is the whole generality queue.
- Do **not** pull in `/numeric-formats-metal`. Here int32 is a 32-bit dataflow payload, and the contract forbids exposing `compute_kernel_config`.
- SUPPORTED is per-axis, so this also admits `int32 + TILE` (R1) and `float32 + ROW_MAJOR` (R2), which are off-production pairs that never become golden cells. Check each once with a `tt-probe.sh` probe (bit-exact, 1/3, -0.0 and NaN for float32), not with new test files (BB-22).
- Keep `FILL_ROW_SOURCE = "cb"` as the default. With `"common_args"` the op allocates no L1, and every graded cell fails `peak_total=0` (verification_report.md, H2). Do not offer `"common_args"` for R2 unless its common-args length (`rm_chunk_bytes / 4` words) is capped against the kernel-config region (BB-04).
- Route `RM_CHUNK_CAP_BYTES` through the existing `Knobs` / plan path as one host constant. Derive the CT arg `rm_chunk_bytes`, the CB size and the chunk split from it once (DRY), and assert `rm_chunk_bytes <= NOC_MAX_BURST_SIZE` on the host.
- The production `[1]` cell is a 4-B DRAM write from a 16-B-aligned CB base to a page start, which satisfies the Blackhole mod-16 congruence rule. Only the length is ragged. Run the acceptance test once under `--dev` (NoC sanitizer) to confirm.

**Done when**:
- The three graded `xfail_expected` cells pass: `test_op[1-…-INT32-ROW_MAJOR]`, `loose5` (`16777217`, `-2^31` bit-exact) and `loose10` (`1.5`, `2^31` refused with ValueError after one admitted call).
- `test_fill_int32_row_major` and `test_fill_int32_value_bits` in `test_g6_fill.py` pass instead of xfail.
- All Phase 0 cells and `test_g6_fill_knobs.py` stay green.
- `verify_supported` shows `xpass_drift = 0`, `xfail_wrong_mode = 0`, and no `supported_fail` other than the harness-caused `loose6` (H1).

## Perf refinements — none filed

The 2:1 cadence would put a perf refinement after the generality refinements. None is filed, because none can be measured:

- `eval/prompts/perf_rules.txt` says to "measure only cells of more than 32 tiles … this overrides any other instruction". Every g6_fill cell is at most 16 tiles, the PERF FOCUS `[512]` (`LOOSE_CASES[0]`) included.
- A perf refinement without a measurable device-ns gate is not a refinement.
- verification_report.md (H3, Recommendations) records the conflict with the PERF FOCUS flag and the levers that are ready to file if the eval owners exempt this op from the threshold.
