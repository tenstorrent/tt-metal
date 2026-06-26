# Operation Requirements: softmax

## Definition
- **Formula**: `output[n,c,h,w] = exp(x[n,c,h,w] - max(x[n,c,row_or_col])) / sum(exp(x[n,c,h,w] - max(x[n,c,row_or_col])))` where max and sum are along `dim` (-1 = W, -2 = H)
- **PyTorch Reference**: `torch.softmax(x, dim)`
- **Import Path**: `from ttnn.operations.softmax import softmax`
- **Function Signature**: `softmax(input_tensor: ttnn.Tensor, dim: int = -1, *, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED alignment**: [tile_aligned]
- **SUPPORTED rank**: [4]
- **SUPPORTED dim**: [-1, -2]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **EXCLUSIONS**: {fp32_dest_acc_en=False} — fp32-dest-only op, rejected for all dtypes
- **Cores**: multi-core (split_work_to_cores over NC slabs)
- **Compute config**: hard-coded HiFi4 + fp32_dest_acc_en=True + math_approx_mode=False
- **Golden baseline**: 37 / 1250 cells passing (per verifier CLI); 1053 xfail_expected, 140 invalid_skipped, 12 supported_fail (all OOM on wide shapes)

### [x] Refinement 1 — Numerical configurability (dtypes + fp32-dest-only policy)

**Goal**: add `ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, and correct intermediate-CB precision (incl. `UnpackToDestFp32` tagging where applicable). The op is fp32-dest-only — `fp32_dest_acc_en=False` is already rejected via EXCLUSIONS for all dtypes and stays there. Cells that fail out of the box (typically `bfloat8_b + non_tile_aligned`) land in `EXCLUSIONS`, not in their own refinement. Pass condition: zero kernel changes when helpers are wired correctly.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: should land first — Refinement 2 (layout) and Refinement 3 (non-tile-aligned) will reuse the dtype-driven CB format derivation introduced here. The `bf8b + ROW_MAJOR` cells are already in `INVALID` (structural impossibility). The `bf8b + w_non_aligned` / `bf8b + h_non_aligned` cells should go to `EXCLUSIONS` when this refinement lands, since bf8b is a block format and non-aligned shapes need the masking from Refinement 3 first.

**Done when**: all `dtype=BFLOAT16` and `dtype=BFLOAT8_B` cells currently in `xfail_expected` with `layout=TILE, alignment=tile_aligned, rank=4, fp32_dest_acc_en=True` pass.

### [x] Refinement 2 — Layout support + multi-core distribution

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` via an in-kernel tilize-wrapped reader (math always stays on tiles). Distribute the per-row work across the available core grid (interleaved DRAM, embarrassingly parallel — already multi-core from Phase 0, but the layout path needs its own reader validation).

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**: the layout work and the work-split share the same reader/writer rewrite — bundle them. The multi-core stamp does not address the L1 fit on the wide-W shapes (Refinement 5 below) — that stays separate. The `bf8b + ROW_MAJOR` cells are in `INVALID` and won't be touched. `fp32 + ROW_MAJOR` and `bf16 + ROW_MAJOR` are the target cells.

**Done when**: all `layout=ROW_MAJOR` cells currently in `xfail_expected` with `dtype ∈ {float32, bfloat16}` (after Refinement 1), `alignment=tile_aligned`, `rank=4`, `fp32_dest_acc_en=True` pass.

### [x] Refinement 3 — Non-tile-aligned H/W (reduction-axis masking)

**Goal**: add `w_non_aligned` and `h_non_aligned` to `SUPPORTED["alignment"]`. The kernel must mask padded lanes on the **reduction axis** before max and exp — for `dim=-1` with `W % 32 != 0`, the `(32 - W % 32)` trailing lanes in the tail tile of each row must be treated as `-inf` before the max reduction (so `exp(-inf) = 0` and they contribute nothing to `sum_exp`); for `dim=-2` with `H % 32 != 0`, same rule along the column direction. Padded lanes on the non-reduction axis are don't-care (golden checks compare on logical shape via `ttnn.to_torch`).

**Verifier notes**: this is the "tricky non-tile-alignment" standalone case — the reduction unit changes shape because padded lanes on the reduction axis would contaminate max and sum. The mask must be applied per-tile-row / per-tile-column, not per-tensor, because only the last tile along the reduction axis has tail lanes. No skill in the current inventory covers this precisely — the `/memory-layouts` skill's "non-aligned rule" section (last-tile H/W zero-pad/mask in the reader or compute) is the closest reference. `bf8b + non_tile_aligned` cells should go to `EXCLUSIONS` if they fail (bf8b is a block format; non-aligned shapes may need special handling).

**Done when**: all `alignment ∈ {w_non_aligned, h_non_aligned}` cells currently in `xfail_expected` with `dtype=float32`, `layout=TILE`, `rank=4`, `fp32_dest_acc_en=True` pass.

### [x] Refinement 4 — Rank expansion (2D, 3D tensors)

**Goal**: add `rank=2` and `rank=3` to `SUPPORTED["rank"]`. The program descriptor currently assumes rank-4 (`N, C = shape[0], shape[1]; H, W = shape[2], shape[3]`). For rank-2 `(B, H)` and rank-3 `(B, S, H)`, the host-side code needs to reinterpret the shape as 4D internally (e.g., rank-2 → `(1, 1, B, H)`, rank-3 → `(1, B, S, H)`) before building the program descriptor. The kernel itself is rank-agnostic — it only sees Ht, Wt, and num_slabs as compile-time/runtime args.

**Verifier notes**: this is a host-side change, not a kernel change. The `validate()` function already canonicalizes `dim` based on `ndim`, so positive dim aliases for rank-2/3 already work. The program descriptor needs to collapse leading dims into `NC` and extract `H, W` from the last two dims. No skill pointer — this is a straightforward host-side shape reinterpretation.

**Done when**: all `rank ∈ {2, 3}` cells currently in `xfail_expected` with `dtype=float32`, `layout=TILE`, `alignment=tile_aligned`, `fp32_dest_acc_en=True` pass.

### [~] Refinement 5 — L1 budget fit for wide/tall reduce dim

**Goal**: rewrite the reduction phase so the per-core L1 CB footprint is bounded by a constant (chunking on the reduce dim), so the op stops OOMing on the wide-hidden shapes in `feature_spec.INPUTS` (W ∈ {4096, 8192}, H ∈ {2048, 4096, 512}). Phase 0 leaves these cells failing with `OOM`; this refinement moves them to passing. No SUPPORTED axis is added — `shape_size` is not a kernel-level branch, just a resource boundary, and bucketing it would hide the gap.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: this is the per-core L1 fit refinement — Phase 0 sizes `cb_input_tiles` and `cb_exp` as `Ht × Wt × tile_size`, which exceeds the 1.5 MB budget around `Wt = 128` (W = 4096) for fp32. The skill's streaming-reduce wrapper (`accumulate_reduce_block`) is the natural fit; the online-softmax algorithm (single-pass running max + running sum-of-exp) would allow sub-slab processing with constant-size CBs. The `softmax_full.txt` prompt describes a two-variation dispatch (V1 fast path ≤ 256 KiB, V2 conservative streaming path > 256 KiB) — the implementer should follow that host-side selection rule. This refinement goes last — by the time it runs, the rest of the SUPPORTED rectangle is stable.

**Done when**: every Phase 0 cell currently in the `OOM` category passes.

### [ ] Refinement 5a — V2 RM layout streaming path

**Goal**: extend the V2 streaming path to support ROW_MAJOR layout. The V2 TILE path (3-pass chunk_along_reduce + V1-style chunk_along_non_reduce) is working and passes all TILE-layout wide/tall shapes. The V2 RM path (chunked tilize/untilize with `byte_offset_within_page` per chunk) is not yet implemented — RM shapes that exceed the V1 CB budget (256 KiB) still OOM. The specific cells that need this are `layout=ROW_MAJOR` × wide/tall shapes (W∈{4096,8192}, H∈{2048,4096}, 1024×1024).

**Verifier notes**: the V2 RM path requires the reader to use `read_sticks_for_tilize` with `byte_offset_within_page` to read W-chunks of each stick, and the writer to use `write_sticks_after_untilize` with the same offset. The compute kernel's tilize/untilize helpers need to be called per-chunk (InitAndUninit lifecycle). The `chunk_along_non_reduce` path for dim=-2 RM is the hardest case — the output tiles are in column-major order, which doesn't map to contiguous RM sticks. The next implementer should start with `chunk_along_reduce` dim=-1 RM (the attention use case) and handle `chunk_along_non_reduce` dim=-2 RM as a stretch goal.

**Done when**: all `layout=ROW_MAJOR` cells with wide/tall shapes that currently OOM pass.
