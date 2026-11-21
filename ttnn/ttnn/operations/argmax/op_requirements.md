# Operation Requirements: argmax

## Definition
- **Formula**: `out[..., r] = min{ i : x[..., r, i] = max_j x[..., r, j] }` along `dim = -1` (lowest index on ties; `+0.0 == -0.0`), the searched dimension removed (`keepdim=False`) or kept with size 1 (`keepdim=True`). Output uint32, ROW_MAJOR, in the input's buffer (R2, R8).
- **PyTorch Reference**: `torch.argmax(input_tensor.double(), dim=dim, keepdim=keepdim)` (specification `reference`)
- **Import Path**: `from ttnn.operations.argmax import argmax`
- **Function Signature**: `argmax(input_tensor: ttnn.Tensor, dim: int = None, keepdim: bool = False, *, sub_core_grids: ttnn.CoreRangeSet = None, output_tensor: ttnn.Tensor = None) -> ttnn.Tensor`
- **Specification**: `eval/specs/argmax/spec.py` (RULES R1–R10, REJECTIONS X1–X2; graded `exact`)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (ttsim, untimed)**: a perf refinement is gated on the **structural count** its entry names (value before → value that makes it done), read off the op by a test or probe; the gain in time stays unconfirmed until a timed backend or hardware measures it. Never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [ROW_MAJOR_LAYOUT]
- **SUPPORTED memory_config / sharding**: [DRAM] / [interleaved]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank = 4
- **SUPPORTED op-specific axes**: dim = -1; keepdim = False
- **Cores**: full grid, every user DM hart: one DM kernel `argmax_scan`, thread plan (R, C, W) = (6, 0, 0) (`SCAN_THREADS`), two regimes — ROWS (rows dealt to all 192 scanners) and SEGMENTED (long rows cut into segments across scanners, merged by an in-program L1→L1 gather to one finalizer)
- **Compute config**: none (the op compares; DECISIONS of the specification)
- **Golden baseline**: 12 / 12 PHASE0 cells + 3 / 3 numeric regression cases + 2 / 2 rejections pass; 308 cells xfail as expected; 0 loud (per verifier CLI)

### [ ] Refinement 1 — Tile-layout input (native tile walk)

**Goal**: add `ttnn.TILE_LAYOUT` to SUPPORTED["layout"] with the scan reading tiles natively in the kernel: `load_block` fetches tile-row (or tile-segment) pieces, `scan_block` walks face order (face = 16×16; rows 0–15 in faces 0–1, rows 16–31 in faces 2–3) and never lets a padded element win (columns ≥ W, rows ≥ H — GUIDANCE "Tile layout"). The output stays uint32 ROW_MAJOR (R2). All three alignment buckets must pass on TILE, including `h_non_aligned` and `w_non_aligned`, where the padding holds arbitrary values.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Hardest generality axis (a new load path), so it goes first and lands on the small surface (bf16 / DRAM / rank 4 / keepdim=False). Refinement 2 then extends both the RM and the TILE path for float32 / L1 / keepdim / rank 2–3.
- The unit changes shape: a tile-row covers up to 32 logical rows. **Keep the grid filled.** A plain tile-row split strands scanners: `(4,8,32,256)` has 32 tile-rows, so 32 of 192 scanners, against 192 on the RM path. Deal sub-tile-row units instead (16-row face halves, or SEGMENTED tile-column ranges). The bar is the RM path's figure on each shape, for both scanners engaged and the most elements any scanner reads, through the planner's `select_regime()` / `split_units()` and not a parallel copy.
- **Do not read padding across DRAM when it can be skipped.** `(1,1,2,151936)` in TILE is one tile-row of 4,748 tiles with only 2 real rows. Its rows 0–1 sit in faces 0–1 (the first 1 KiB of each bf16 tile), and their bytes are 2 × 2 face-rows of 32 B per tile. The design counts DRAM crossings per input element (budget: once); add a TILE line to `l1_ledger.md`'s data-movement budget with the bytes actually read per real element.
- SEGMENTED records become per (row, segment), up to 32 rows per tile-row unit: re-derive the landing-zone bound (`U ≤ S_total` no longer bounds the records) and update the ledger's `sp_partials` row and symbol table.
- No wrapper: `ttnn.to_layout` / `ttnn.untilize` before the scan would be a second dispatch, which the one-dispatch rule forbids. It is not even a partial-tick escape here.

**Done when**: every `layout=TILE` golden cell whose other axes are in SUPPORTED passes `exact`; the thread-plan, session and sub-grid tests have TILE counterparts (one ROWS shape and one SEGMENTED shape); `l1_ledger.md` is current; no PHASE0 cell regresses.

### [ ] Refinement 2 — float32, L1 placement, keepdim=True, ranks 2 and 3

**Goal**: add `ttnn.float32` to SUPPORTED["dtype"], `"L1"` to SUPPORTED["memory_config"], `True` to SUPPORTED["keepdim"], and `2`, `3` to SUPPORTED["rank"], on **both** the ROW_MAJOR and the TILE path. All four are knob-turns on the existing scheme (op_design.md, "TARGET axes against the scheme"):
- float32: `elem_bytes = 4` and a 32-bit order-preserving key (sign-magnitude → signed, `-0.0 → +0.0`), one template parameter of `scan_piece`. R3: the full 32-bit word is compared, and nothing is narrowed to bf16. `MIN_SEGMENT_BYTES` and `block_bytes` are in bytes already, so the segment and block geometry follow.
- L1: `TensorAccessor` addresses L1 banks unchanged; the output follows the input (R8). Add a test asserting `output.memory_config()` equals the input's buffer type, for DRAM and for L1.
- keepdim=True: `out_rows_per_page = 1`. Every row is its own output page; keep the writes batched per block (one barrier per block, not per row).
- Ranks 2 and 3: `tensor_rows = prod(shape[:-1])` is already general. Rank 2 with keepdim=False gives a 1-D output: one page of `tensor_rows` indices.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Use the skill for the dtype part only: adding `float32` to SUPPORTED, the dtype-derived buffer sizes and the precision test contract. **Do not add a `compute_kernel_config`.** The specification gives the op none (DECISIONS), and its signature has no such parameter. `bfloat8_b` / `bfloat4_b` are not in TARGET and have no path on Quasar.
- Must follow Refinement 1, so that float32, keepdim and rank land on the TILE walk as well. A float32 tile is 4 KiB; a face row is 16 × 4 B.
- L1 inputs share each cluster's L1 with this op's scratchpads. The scratchpads are tens of KiB at these shapes and the allocator places both, but run the largest L1 cells, `(1,1,2,151936)` in TILE float32 at 19.4 MB, before calling it done.
- Rank 2 brings in `(128, 8192)`, the shape Refinement 4 targets. Leave its split as it is here (ROWS, 128 of 192 scanners).

**Done when**: every golden cell with dtype ∈ {bfloat16, float32}, memory_config ∈ {DRAM, L1}, keepdim ∈ {False, True}, rank ∈ {2, 3, 4} and layout ∈ SUPPORTED passes `exact`; the R8 placement test passes; `verify_supported` shows 0 xfail_expected cells left in TARGET; no prior cell regresses.

### [ ] Refinement 3 — Occupancy and scan knobs on the model's 32-row call

**Type**: perf

**Goal**: speed up the already-supported shapes with three ⭐ T1 knob-turns on the block surface the planner exposed (`ttnn/ttnn/operations/examples/master.md`: `width_split` for occupancy, `double_buffer` for overlap). No SUPPORTED change.
1. **Segment floor** (`MIN_SEGMENT_BYTES`, op_design.md perf lamp "Grid synchronization"). On the model's `(1,1,32,4096)` call, and on rank-3 `(1,32,4096)`, SEGMENTED stops at `spr = 4`, because 8192 B / 2048 B = 4 caps it below `S_total // rows = 6`. Lowering the floor to 1 KiB, or deriving it from `S_total // rows` with a byte minimum, gives `spr = 6`.
2. **Overlap** (perf lamp "Overlap"). Every PHASE0 scanner walks its whole assignment as one block at `in_depth = 1`, so the read never overlaps the scan. Where a scanner's assignment is large, `(1,1,2048,256)` with 11 rows / 5.6 KB and `(4,8,32,256)` with 6 rows, split it into ≥ 2 blocks at `IN_DEPTH = 2`. The block stays whole rows (catalog floor: no sub-row chunking when rows fit).
3. **Scan granularity** (perf lamp "Scan granularity"). The DM harts are 64-bit (`-mcpu=tt-qsr64-rocc`, `qa_hal.cpp:420`), so fold 4 bf16 (or 2 fp32) per 64-bit load instead of 2 per 32-bit load. The column order inside a word must stay low-to-high, so ties are unchanged.

**Done when** (structural counts, read by a test or probe off `select_regime` / `split_units` / `derive_blocks` and the kernel's CT args; gain in time unconfirmed on ttsim):
- `(1,1,32,4096)`: scanners engaged 128 → 192 and the most elements one scanner reads 1024 → ≤ 683;
- `(1,1,2048,256)` and `(4,8,32,256)`: blocks per scanner 1 → ≥ 2 with `in_depth = 2`;
- loads per bf16 element in `scan_piece` 0.5 → 0.25;
- the golden suite is green, including the `ties` / `signed_zeros` cases;
- no regression across the config-spanning guard set: one representative per kernel path × layout × placement × dtype, at least ROWS-RM, SEGMENTED-RM, ROWS-TILE, SEGMENTED-TILE, each at bf16-DRAM and fp32-L1. On each, scanners engaged must not fall and the most elements per scanner must not rise.

### [ ] Refinement 4 — BALANCED-RANGES: fill the grid when the row count is between S_total/2 and S_total

**Type**: perf

**Goal**: build the design's `deferred` regime BALANCED-RANGES (op_design.md, Regimes). Each scanner takes a contiguous range of `ceil(rows·W / S_total)` elements, which may start and end mid-row. It emits a head and a tail record for each partial row, merged by the finalizer in column order with strict `>`; whole rows inside the range are written directly as in ROWS. This is a ⭐⭐⭐ T3 scheme change: the topology of the cross-core combine changes, `sp_partials` grows to 2 slots per scanner, and the selection predicate gains a third branch. Use master.md's `width_split` / `tensix_all_reduce` entries for the shape of the split and the gather. No SUPPORTED change.

**Verifier notes**: depends on Refinement 2, since its target `(128, 8192)` is rank 2. The merge order is the lexicographic maximum of `(key, -index)` (design, "Tie rule"), so a row split across two scanners keeps the lowest index of a tie. Add a tie case that straddles a range boundary. Update `l1_ledger.md`: the landing-zone bound, the symbol table and the data-movement budget (records = U + 2·S_total at most).

**Done when** (structural counts; gain in time unconfirmed on ttsim):
- `(128, 8192)`: scanners engaged 128 → 192 and the most elements per scanner 8192 → ≤ 5,462;
- the same counts do not get worse on any shape of the Refinement 3 guard set (ROWS and SEGMENTED, RM and TILE, bf16-DRAM and fp32-L1);
- the golden suite and the numeric regression cases stay green;
- a boundary-straddling tie test passes.

## Coverage of TARGET − SUPPORTED

| Axis | Missing value | Refinement |
|------|---------------|------------|
| layout | TILE_LAYOUT | 1 |
| dtype | float32 | 2 |
| memory_config | L1 | 2 |
| keepdim | True | 2 |
| rank | 2, 3 | 2 |

`bfloat8_b` / `bfloat4_b` are not in TARGET (and have no path on Quasar). `dim` other than -1 and `dim=None` are not in TARGET (specification DECISIONS). `sharding` has one TARGET value, `interleaved`, already supported.
