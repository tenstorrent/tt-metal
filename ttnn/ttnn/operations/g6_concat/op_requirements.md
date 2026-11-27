# Operation Requirements: g6_concat

## Definition
- **Formula**: `d = dim mod r`; `o_k = n_0 + … + n_{k-1}`; `out[…, o_k + j, …] = x_k[…, j, …]` for `0 ≤ j < n_k`, every logical element bit-identical to its source (NaN payloads, ±0, subnormals, ±inf included)
- **PyTorch Reference**: `torch.cat(tensors, dim)` on the inputs' own dtype (no float32 round trip: it canonicalizes bf16 NaN payloads)
- **Import Path**: `from ttnn.operations.g6_concat import g6_concat`
- **Function Signature**: `g6_concat(tensors: list[ttnn.Tensor], dim: int = 0, *, memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Known harness failures (do not fix in the op)**: the golden bf16 special-values cells (`test_op_loose[1x6x31x64+1x6x31x192_d-1 …]` with `special_values`, `test_regression.py::test_special_values`) fail because `helpers.pytorch_g6_concat` canonicalizes bf16 NaN payloads. The op returns `torch.cat`'s bits. `test_trace_replay` fails only under `eval_test_runner.sh` (an `eval/profiling.py` unpack bug, reproduced with stock ops). See `verification_report.md`. Never canonicalize NaN or EXCLUDE that cell to turn it green.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED shape-derived axes**: concat_dim ∈ {outer, h, w}; alignment ∈ {tile_aligned, h_non_aligned}
- **SUPPORTED placement axes**: input_buffer = dram; output_buffer = dram
- **Cores**: full grid. `out_page` split by `split_work_to_cores(row_wise=True)` gives `min(P_out, 110)` cores. Blocks of `block_pages = min(n1, BLOCK_BYTES_CAP / page_bytes)` pages, double-buffered.
- **Regimes**: `page_copy` (one input page per output page, rows `[0, v)` only when `v ≤ 16`) and `row_gather` (h joins with unaligned offsets, row-range reads plus a core-local hop for odd shifts)
- **Compute config**: none. No compute kernel, and the spec forbids `compute_kernel_config`.
- **Golden baseline**: 79 / 80 supported cells pass. The 1 failure is the harness NaN-oracle defect above. 86 cells are xfail_expected. Drift and wrong-mode counts are 0.

### [x] Refinement 1 — dtypes float32 + bfloat8_b, interleaved L1 / mixed placement

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, `"l1"` and `"mixed"` to `SUPPORTED["input_buffer"]`, and `"l1"` to `SUPPORTED["output_buffer"]`. Each input is read in its own buffer and the output is written where `memory_config` names. This moves 49 xfail cells to passing:
- 40 fp32 INPUTS cells;
- LOOSE p1 (Qwen 18, the perf-focus contract) and the fp32 special-values case;
- Qwen 157 and 158, plus the L1-input poisoned-padding loose case;
- Panoptic 23 (bf8_b, mixed inputs → L1 output) and Panoptic 32/40/51 (bf8_b, L1 inputs → L1 output).

`test_regression.py::test_cache_key_placement` steps 2 and 3 start running and must pass. `EXCLUSIONS` gains `{"dtype": ttnn.bfloat8_b, "concat_dim": "h", "alignment": "h_non_aligned"}`. That is the design's deferred `row_gather_bfp8` regime: the 1-byte face-row exponents cannot be placed by the NoC. No INPUTS cell is in it.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Measured starting point.** With SUPPORTED widened and no other change, all 40 fp32 INPUTS cells, p1, and the all-L1 bf16 cells Qwen 157/158 already pass bit-exact. fp32 and all-L1 inputs are therefore a SUPPORTED change plus acceptance pins. The real work is the remaining items below.
- **Mixed inputs** (Panoptic 23, and Qwen 159 in R2). `build_program` asserts that all inputs share one accessor. Replace that with the design's *Interleaved accessor buffer type* mechanism: a per-input buffer-type bit (CT mask, already in the cache key via the buffer types), plus one DRAM and one L1 accessor rebased only when the walk crosses into the next input. No per-page branching in the hot `page_copy` loop: R3 optimizes this exact path.
- **L1 output.** Remove `build_plan`'s DRAM-output assert. Add `out_l1_per_bank = ceil(P_out / num_l1_banks) · page_bytes` to `l1_total` for the BB-05 check (ledger symbol table). The writer already takes its accessor from the output tensor.
- **`row_gather` with L1 inputs.** `input_read_alignment` is the compile-time `NOC_DRAM_READ_ALIGNMENT_BYTES`. With all-L1 bf16 inputs `has_staging` is false, so the `static_assert` fails to compile. With mixed inputs, L1 pieces would be staged needlessly. Make the alignment per input (L1 = 16, so every L1 piece is direct). Pin it with an acceptance test (h join, odd shifts, L1 and mixed inputs), because SUPPORTED will claim `l1 × h × h_non_aligned` and no golden cell covers it.
- **fp32 `row_gather`.** `rb = 64`, so every piece is direct and no staging CB is allocated. Pin one fp32 h join with odd shifts; no golden fp32 cell is `row_gather`.
- **bf8_b.** `page_copy` with whole pages (`TRIM_ROWS_MAX = 0` already). Every Panoptic cell is a tile-aligned w join, so the grade is bit-exact.
- **Which parts of the skill apply.** Only the dtype-surface part. No compute kernel exists and the spec forbids `compute_kernel_config`, so skip the `compute_kernel_config` / `fp32_dest_acc_en` / intermediate-CB / `UnpackToDestFp32` parts. CB formats already follow the tensor dtype.
- **Why placement is here.** It is a knob-turn with no skill (interleaved buffer type, not sharding, so not `/memory-layouts`). It rides in this refinement because the bf8_b production cells cannot pass without it. R2 depends on the per-input mask for Qwen 159.

**Done when**: the 49 cells above pass and the golden suite is otherwise unchanged. Acceptance pins pass for fp32 `row_gather`, L1 and mixed `row_gather`, bf8_b on DRAM, and an L1 output. `test_cache_key_placement` runs all three steps. `g6_concat_plan` reports the L1 total including the output's per-bank bytes.

### [x] Refinement 2 — ROW_MAJOR layout (native stick copy)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` for all inputs and the output, through the design's deferred `page_copy_row_major` regime, natively in the kernels. Page = stick, using the same readers, writer, `out_page` split and block knob. The `(O, C_k)` map is in stick units with H untiled, and there is no trimming. This moves the 36 RM DRAM cells plus Qwen 159 (RM, mixed inputs) to passing: 37 cells. `EXCLUSIONS` gains `{"layout": ttnn.ROW_MAJOR_LAYOUT, "concat_dim": "w"}`, the design's deferred `stick_gather_row_major` regime: no INPUTS cell joins RM tensors along the last dim, and BB-22 forbids testing it.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- **Pure-RM data movement.** The op converts nothing, so there is no tilize/untilize stage and no `ttnn.to_layout` wrapper (spec MUST NOT, BB-02). There is no partial-tick wrapper escape hatch for this op.
- **RM stick map.** For `d ≤ r−2`: `O = Π s[0:d]` and `C_k = n_k · Π s[d+1:r−1]`. Covers rank-3 dim 0 (Qwen 28/29/33…) and rank-3 dim 1, which is an RM `h` join (Qwen 77/78/141/142/159). Stride by the buffer's aligned page size. Trimming and `valid_rows` do not apply to RM, so disable them on that path.
- **Dtypes.** SUPPORTED will claim fp32 × RM. Pin one fp32 RM case; no golden RM cell is fp32, and an fp32 stick is `W·4` bytes.
- **Placement.** Carry R1's per-input buffer-type mask onto the RM path; Qwen 159 is RM with DRAM and L1 inputs.
- **Work split.** Keep the `out_page` split (here: output sticks) over `min(P_out, grid)` cores. Every RM production cell has at most 48 sticks, i.e. ≤ 32 tiles, so `perf_rules.txt` exempts them from measurement. Correctness and the split are the bar.
- **Ordering.** After R1 (mask). Before R3, so R3's guard set covers the RM path.

**Done when**: the 37 cells pass and the golden suite is otherwise unchanged. Acceptance pins pass for an RM outer join, an RM `h` join, fp32 RM and RM mixed placement. `g6_concat_plan` reports the RM regime.

### [ ] Refinement 3 — Speed up the perf-focus profile (p1, Qwen 18)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags p1 as the mandatory perf target: `(108, 4096, 128) + (84, 4096, 128) → (192, 4096, 128)`, float32, TILE, DRAM interleaved in and out, dim 0, `extras: {"math_fidelity": None, "gate": "exact"}`. Optimize that exact config; R1 puts it in SUPPORTED.
- **Measured during verification** (device kernel ns, 110 cores, `validate()` bypassed in a measurement-only test, output bit-exact): **1889 µs**, against stock `ttnn.concat` 1950 µs and the lead's DRAM roofline 1747.6 µs (805.3 MB at 460.8 GB/s). That is **1.081×** the roofline. The case's goals (< 1948.52 µs, ≤ 1.35× roofline) are already met; this refinement targets the residual ~8%. bf16 p2 on the same path reaches 1.025× the roofline, so the fp32 configuration is what is losing.
- **Levers** (all ⭐⭐ T2 in `ttnn/ttnn/operations/examples/master.md`, so one phase):
  - `BLOCK_BYTES_CAP` ∈ {128, 256, 512} KiB. Today p1 gets `block_pages = 32` fp32 pages, while ~164 would fit the budget at depth 2. This is the L1-ledger finding folded in: the cap was justified by a bf16 catalog plateau, never measured on this op.
  - `OUTPUT_PAGES_DEPTH` 3–4 and `trid_double_issue` (the design's overlap lamp: more reads in flight across the per-block barrier).
  - `bank_stagger`: p1 gives ~894 pages per core; rotate each core's in-block issue order by `core_idx mod num_banks`.
  - `noc_placement` (row-wise cores, reads on NOC0, writes on NOC1) is already in place.

  No SUPPORTED change. The knobs are in the program-cache key (`blocking_knobs()`), so a sweep in one process is safe.

**Done when**: p1 device-ns improves beyond the ~3% noise band (≤ ~1830 µs). The bit-exact gate holds and the golden suite is green. Nothing regresses across the config-spanning guard set: one representative per kernel path × layout × placement, at least p2 (bf16 `page_copy`), Qwen 79 (`row_gather`), Qwen 80 (trimmed w join), one Panoptic cell (bf8_b, L1 output) and one RM cell, the last two only where they exceed 32 tiles. If the full lever sweep lands within noise, the measured sweep table in `changelog.md` is the deliverable and the entry ticks with the residual gap to the roofline stated.
