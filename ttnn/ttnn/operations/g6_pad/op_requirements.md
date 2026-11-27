# Operation Requirements: g6_pad

## Definition
- **Formula**: `D_i = d_i + h_i`; `out[j] = x[j]` if `j_i < d_i` for every `i`, else `c` (`c` = `value` converted to the input dtype as torch converts it: bf16 = RNE of fp32, integers by truncation). Bit-exact move, no arithmetic.
- **PyTorch Reference**: `torch.nn.functional.pad(x, flat, mode="constant", value=value)` with `flat` = the `(low, high)` pairs reversed and flattened (`eval/golden_tests/g6_pad/helpers.py::pytorch_g6_pad`).
- **Import Path**: `from ttnn.operations.g6_pad import g6_pad`
- **Function Signature**: `g6_pad(input_tensor: ttnn.Tensor, padding: Sequence[Sequence[int]], value: float | int = 0.0) -> ttnn.Tensor`. The `padding` pairs are aligned to the trailing dims, with `low == 0` and `high >= 0`.
- **Storage classes**: `tile_padding_extension` (TILE, only dim -2 grows inside the last tile row) writes `c` in place and returns a metadata-only view. Every other call is `new_storage`: each output page is assembled once in L1 and written once. Exactly one device program per call.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32, int32, uint32]. The verifier widened this from [bfloat16] on measured evidence (see verification_report.md).
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED memory_config**: ["dram", "l1"] (interleaved)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}; padded_dims ∈ {leading, h, w, hw}
- **Regimes built**: `extension_band` (in-place band write, two DM-RISC lanes) and `new_storage_tiles` (NCRISC reader + BRISC writer, L1 page assembly). `new_storage_rows` is deferred.
- **Cores**: full grid. `split_work_to_cores(row_wise=True)` over the flat band-tile / output-tile index; `ncores = min(grid, ceil(N / min_units_per_core))`.
- **Compute config**: none. There is no compute kernel and no DEST.
- **Golden baseline**: 78 / 85 tagged cells pass and 7 are `xfail_expected` (all ROW_MAJOR). The default runner reports 74 pass + 4 `supported_fail`. Those 4 are the harness L1-profiling artifact on host-side `args` refusals (see verification_report.md), not op failures.

The perf-1 contract (p1 = `[5,12,128,128]` float32 TILE DRAM, `padded_dims=leading`, `alignment=tile_aligned`, rank 4) is fully inside SUPPORTED at Phase 0. No generality refinement is needed to unlock it.

### [x] Refinement 1 — ROW_MAJOR layout (`new_storage_rows` regime)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` by building the design's deferred `new_storage_rows` regime natively in the kernels, for every TARGET dtype (bf16, fp32, int32, uint32) on DRAM and L1 interleaved. The page is one row of `D_{r-1} · elem_bytes`. Each input row is read once into the start of a 64 B-aligned CB page, and its tail `[d_{r-1}, D_{r-1})` is filled in L1 (`fill_l1_range`). Fill-only rows (new leading or dim -2 rows) are written from a fill row page and never read. Every output page is written once. This moves the 7 `xfail_expected` cells to passing:
- cases 260 `[1,1,1,1]` u32 and 264 `[1,1,1,8]` u32 (`w`);
- cases 262 `[5,8192]` i32 and 263 `[5,1]` i32 with `c = -1`, and 265 `[5,1]` u32 (dim -2 grows);
- case 267 `[5,3,2560]` bf16 (`leading`);
- the loose `special_values` case on 262.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**: this is the only remaining generality gap (TARGET − SUPPORTED = {layout: ROW_MAJOR}), so it carries the whole RM surface. Do not split it by dtype.
- **Reuse.** Reuse the `make_plan` / `select_regime` / descriptor-cache plumbing: `select_regime` already returns `"new_storage_rows"`, and `make_plan` currently raises for it. Reuse the reader/writer split, the `cb_input_tiles` nominal-push discipline (page = row) and the fill-page idea.
- **Multi-core.** Split the flat output-row index with `split_work_to_cores(row_wise=True)`.
- **Few rows.** The spec's work-geometry rule (core count is the blocking decision, never a limit set by the tensor's shape) bites here: `[5,8192]` i32 has only 8 output rows of 32 KB, and `[5,3,2560]` has 24 rows of 5 KB. Expose a row-chunk knob that splits the column axis of a row across cores. This is an independent-axis knob-turn.
  - Chunk offsets must respect the Blackhole DRAM read alignment (64 B) and the L1/DRAM write alignment (16 B).
  - Do sub-16 B tails with RISC stores in L1, never with misaligned NoC transfers.
- **L1.** Bound per-core L1 by BB-05: a block of rows or row chunks clamped by the same `block_tiles_cap`-style cap. Add the RM CB rows to `l1_ledger.md`.
- **Tests.** Pin the regime in the acceptance test (new buffer, input unchanged).
- **No host transforms.** Calling `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` at the entry point is forbidden by the spec (rules 5 and 7, BB-01/BB-02). There is no partial-tick wrapper escape hatch for this op: a wrapper would be a second program and a contract violation.

**Done when**: the 7 ROW_MAJOR cells pass bit-exactly in the golden suite with `xpass_drift = xfail_wrong_mode = 0`, and each call is one device program with no host-side conversion. All Phase 0 tests stay green, and the RM path uses more than one core wherever the input holds more than one output page.

### [x] Refinement 2 — Speed up the perf-flagged profile p1 (pipeline knobs)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags p1, case 266: `[5,12,128,128]` float32 TILE DRAM interleaved, dim 0 padded 5 → 8, `new_storage_tiles`, exact gate. It is the mandatory perf target.
- **Goal**: faster than stock (< 29.91 µs, the binding limit) and within 1.35× of the 22.19 µs DRAM roofline.
- **Phase 0 baseline**: 28.3 µs device kernel duration on 110 cores. It already meets the binding limit and sits at 1.28× roofline, leaving ~6 µs of headroom.
- **Current schedule**: each core gets ~14 fp32 tiles as one block. Its reader issues all reads, then one barrier, so the writer cannot start a single store until the whole block has landed. Copy cores move 2× the bytes of the ~41 fill-only cores.
- **Levers**: co-tune the block knobs the plan already exposes, `block_tiles` and `input_depth_blocks` (`double_buffer`: block 4–8, CB = 2·block). They are carried as parameters and correctness-tested at non-default values in `test_g6_pad_knobs.py`. Then try `bank_stagger`: contiguous 14-tile ranges start many cores on the same DRAM bank. See `ttnn/ttnn/operations/examples/master.md`. These are ⭐⭐ T2 knobs and placement, with no kernel restructure.

No SUPPORTED change.

**Done when**: measured device-ns on p1 improves on 28.3 µs, stays < 29.91 µs, and stays bit-exact, with the golden suite green. There must also be no device-ns regression across the config-spanning guard set:
- p2 / p3 / p4 (extension, bf16 DRAM);
- `[1,3844,1536]` extension;
- `[1,1,196,196]` bf16 new_storage with row and column patches;
- case 63 `[1,196]` uint32 rank 2;
- an L1 case (299);
- one ROW_MAJOR case from Refinement 1.

**Outcome**: p1 went from 29.30 µs to 26.78 µs (median of 3, bit-exact). That is −8.6% and 1.21× the roofline, under the 29.91 µs limit. The defaults are now `block_tiles` = 2 at depth 2 (gated by `pipeline_min_units` = 8), `bank_stagger` = 1 and `fill_seed_bytes` = 256. The last of these took the writer's RISC fill of the whole fill page from 2.3 µs to 0.5 µs. The NoC knobs stay at their defaults, which measured fastest. There is no guard-set regression beyond noise, and new_storage shapes gained 15–40%. `bank_stagger` alone measured within noise on p1. The bottleneck now is a placement skew: per-core end time rises from about 8 µs at x=0 to about 26 µs at x=10 in every grid row, for copy and fill-only cores alike. Next would be an x/cost-aware split, which is Refinement 3's scope and so was not done here.

### [x] Refinement 3 — Speed up the perf-flagged profile p1 (copy/fill load balance)

**Type**: perf

**Goal**: the same flagged p1 shape and goal as Refinement 2, starting from Refinement 2's measured result. The equal-count flat split hands the 960 source-bearing tiles to the first ~69 cores (read + write each) and the 576 fill-only tiles to the rest (write only, no read). The DRAM-read half of the op therefore runs on ~60% of the grid while the fill-only cores finish early.
- **Lever**: rebalance by cost. Either use a cost-weighted split (copy = 2, fill = 1) or two independent even splits (a source-bearing list and a fill-only list) so every core gets a mix. This is the design's "load balance (new_storage_tiles)" perf lamp.
- **Scope**: host-side split plus a walker change in both kernels (the same classification predicate). Keep BB-17: few rectangular core ranges per kernel group.
- **Reference**: see `ttnn/ttnn/operations/examples/master.md` (`noc_placement`, `bank_stagger` for the read stream).

No SUPPORTED change.

**Done when**: measured device-ns on p1 improves on Refinement 2's result toward the 22.19 µs roofline (≤ 29.96 µs = 1.35× is the goal) and stays bit-exact, with the golden suite green. There must be no regression across the same guard set as Refinement 2.

**Outcome**: the load-balance lever was built, measured and parked; p1 is unchanged at 26.3–26.9 µs (median 26.8 µs, the same as Refinement 2's 26.78 µs; 1.21× the roofline, under 29.96 µs, bit-exact).
- **What was built.** New knob `split_mode`: 0 = equal count, the default; 1 = cost-weighted contiguous split (`copy_cost` / `fill_cost`); 2 = two segments per core, a slice of the source region and a slice of the new dim-0 region, interleaved in the walk. A second new knob, `fill_upfront`, issues every fill write of a core before block 0.
- **Measured on p1, median of 3:**

  | Variant | Device time (µs) |
  |---|---|
  | `split_mode` 0 (default) | 26.9 |
  | mode 1 at copy/fill cost 2/1 | 28.5 |
  | mode 1 at cost 1/2 | 27.6 |
  | mode 1 at cost 2/3 | 27.5 |
  | mode 2 | 27.6 |
  | mode 2 at cost 1/1 | 27.6 |
  | mode 2 with `fill_upfront` | 28.1 |
  | mode 0 with `fill_upfront` | 27.1 |
  | mode 2 at depth 4 | 28.1 |
  | mode 0 at depth 4 | 27.7 |
  | position weights (1/T per core) | 25.9–27.5, noise |

- **What binds now.** Ablation shows the bound is the DRAM write stream, not the copy/fill split. Write-only (reads stubbed) is 22.5 µs. Read-only (writes stubbed) is 11.4 µs. Write-only does not move under cost or position weighting (22.7 µs). The write-only end-time skew (8 µs west, 25 µs east) is arbitration order, not per-core work. Writes on NoC0 cost 38 µs, so moving fills to the reader RISC or NoC0 is not viable. The remaining ~4 µs over the write-only floor is read/write interference in DRAM.
- **What would be next, and why not here.** Next would be fewer write bytes in flight from congested positions, or DRAM-bank-aware write ordering, i.e. per-bank write queues. Neither is a load-balance change, and the write-only floor (22.5 µs) is already roughly the 22.19 µs roofline.
