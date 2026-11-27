# Operation Requirements: g6_to_layout

## Definition
- **Formula**: `y[n, h, w] = x[n, h, w]` for `n < N, h < H, w < W` (`N` = product of the leading dims). Only the storage order changes: TILE (32×32 tiles, faces 0–3, padded to whole tiles per image) ↔ ROW_MAJOR (page `n·H + h` holds row `h` of image `n`). A bfloat16 output is bit-identical to a bfloat16 input, or exactly equal to the value of a bfloat8_b input. A bfloat8_b output is within one quantization step of the host's bfloat8_b rendering.
- **PyTorch Reference**:
  ```python
  def pytorch_g6_to_layout(input_tensor, layout, *, dtype=None, memory_config=None):
      return input_tensor.clone()   # layout / dtype / memory_config select storage only
  ```
- **Import Path**: `from ttnn.operations.g6_to_layout import g6_to_layout`
- **Function Signature** (fixed by the spec: no `compute_kernel_config`, ever):
  ```python
  g6_to_layout(input_tensor: ttnn.Tensor, layout: ttnn.Layout, *,
               dtype: ttnn.DataType = None, memory_config: ttnn.MemoryConfig = None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape**: the mandatory perf target (LOOSE_CASES[0], case 308: bf16 TILE DRAM-interleaved → RM bf16 DRAM) is fully inside Phase 0 SUPPORTED already. So Refinements 1–2 carry no perf-anchor duty. They are ordered by difficulty and true dependency, and they cover every `TARGET − SUPPORTED` value. Generality is exhausted after Refinement 2, so every later slot is perf. Every perf slot targets case 308, the only `attention:`-flagged case.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE] (input; output is ROW_MAJOR)
- **SUPPORTED output_dtype**: [bfloat16]
- **SUPPORTED memory_layout**: [INTERLEAVED] (input in DRAM or L1; output interleaved DRAM or L1)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **Regimes**: R1 `staged_reorder`, R2 `face_direct` (`tensor_cols_tiles == 1 ∧ W ≤ 16`). Dataflow-only, so bit-exact by construction.
- **Cores**: full grid, `split_work_to_cores(grid, tensor_tiles, row_wise=True)` over the flattened tile space (110 cores on every large shape)
- **Compute config**: none (no compute kernel)
- **Golden baseline**: 58 / 89 registry cells passing (49/73 INPUTS + 9/16 LOOSE_CASES), 31 `xfail_expected`, 0 loud; 2/2 regression tests

### [x] Refinement 1 — bfloat8_b TILE input and sharded inputs (HEIGHT / BLOCK)

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`, `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to `SUPPORTED["memory_layout"]`. Use the two deferred regimes of `op_design.md`:

- **`packer_untilize`**: a compute stage between the tile window and the row block. It unpacks the bfp8 tiles and pack-untilizes them to bf16 rows, a real value conversion, not a byte copy. `store_block` stays as it is.
- **`sharded_in_place`**: the input is consumed in place from a CB backed on the resident shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy). That CB replaces `load_block`. The block is the shard's live tile-rows. Core assignment is the caller's shard grid.

Cells moved from `xfail_expected` to passing (8):
- c29 `[1,32,64,256]` and c43 `[1,128,256,32]`: bf8b TILE, L1-interleaved input, L1 output.
- loose c29 `bf16_specials` (exact gate).
- c20 `[1,1,1,256]`, BLOCK_SHARDED, 8×1 grid, one tile per core.
- c37 `[1,64,128,256]` and c48 `[1,64,128,128]`, HEIGHT_SHARDED on grid 11×7 + 9×1, shard 96 rows.
- loose c37 / c48 with the 1347776-B L1 guard.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Order: first.** Sharding is the hardest structural change in the queue. The HEIGHT shards sit on an irregular two-range grid, and their last shard holds 32 live rows of 96. The BLOCK shard has 8 cores, each writing a 64-B segment of the one shared output page. Every production sharded input is also bf8b, a true dependency, so the conversion and the placement land together. The sharding itself is local: no axis it cuts is dependent, so there is no cross-core combine.
- **Fixed signature.** Do **not** add `compute_kernel_config`; the spec's signature is fixed. Use the skill for the dtype-driven CB formats (bfp8 tile pages are 1088 B, so the reader takes whole tiles) and the compute wiring only. Hard-code the compute config.
- **Bit identity.** The output must equal the host's rendering of the bf8b input bit-for-bit, and the specials loose case grades exactly that. The design records that the unpack → SrcA → MOVA2D path zero-substitutes exponent-0 datums under the default Src zero flag (-0.0 → +0, subnormals → +0). Probe loose c29 first. If any bit differs, route the datacopy through a preserve-zero path (`copy_tile_init` sets it) or unpack to DEST. Never EXCLUDE the cell.
- **Native sharding.** Native means the CB is placed on the shard. A `TensorAccessor` re-read of a core's own local shard does not count as implemented. "Touches no L1 beyond live shard rows" (spec rule): the last shard runs only its live tile-row. The L1-interleaved output's per-core share stays inside 1347776 B (`out_l1_share`).
- **Cache key and plan report.** Add the input's shard spec (grid, shard shape, orientation) to the program-cache key (BB-07) and to `select_plan` (BB-08).
- **bf16 × sharded.** These combinations are not production cells and cannot get a production test (BB-22). List them in `EXCLUSIONS` unless the resident-CB path demonstrably covers them; the EXCLUSIONS rule forbids only production cells.
- **Interleaved split.** bf8b × interleaved keeps Phase 0's full-grid tile-range split with batched reads. Ship it at the performance-conformance bar, not as a single-core stub.

**Done when**: the 8 cells above pass (loose c29 specials bit-exact), the golden suite shows no Phase-0 regression, and `l1_ledger.md` / `op_design.md` carry the new regime rows and CBs.

### [ ] Refinement 2 — ROW_MAJOR input → TILE output (bfloat16 and bfloat8_b)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` and `ttnn.bfloat8_b` to `SUPPORTED["output_dtype"]` via the design's `row_major_to_tile` regime, natively in-kernel, on the full grid:

- **bf16 output, dataflow-only** (bit-exact by construction). The reader reads each live row segment of a block into a row stage, the reorder does the inverse 32-B face-row permutation into tile slots, and the writer writes whole tiles.
- **bf8b output.** A compute tilize stage packs bfp8, with zero-filled padding inside any 16-value group that holds a live value.

Cells moved from `xfail_expected` to passing (23):
- 18 golden RM bf16 → TILE bf16 cells: c1989, c1991, c2022, c2024, c270, c272, c277, c289, c291, c296, c301, c310, c312, and the L1-input cases c21, c30, c38, c45 (`dtype=bf16`), c49.
- Loose c1989 `bf16_specials`, and loose c38 / c49 with the L1 guard.
- c16 `[1,1,1,256]` and loose Q3 `[1,1,32,256]` (RM bf16 → TILE bf8b, `block_step` gate).

**Implementation skill**: /memory-layouts, /numeric-formats-metal, /interleaved-parallel

**Verifier notes**:
- **No wrapper fallback.** In-kernel only. The prompt forbids `ttnn.tilize` / `ttnn.to_layout` / `ttnn.pad` at the entry point outright, so this refinement has **no** partial-tick wrapper escape hatch.
- **bf16 stays dataflow-only.** A compute tilize path carries the same MOVA2D hazard as Refinement 1 for bf16 values, and loose c1989 specials grades it bit-for-bit. The compute stage is for bf8b output only. After Refinement 1, reuse its dtype-driven CB-format derivation for bf8b.
- **H = 1 multi-image shapes** (c1989 `[16,1,1536]`, c1991, c2022, c2024): every image is its own tile-row, with one live row and 31 padding rows. Output tile padding is free for bf16. Each image's tile-row is padded independently, so pages are `n·H + h`, never `floor(N·H/32)`.
- **L1 placement.** L1-input cases with `memory_config=L1_INTERLEAVED` put the TILE output's per-core pages inside the 1347776-B budget (loose c38/c49 guards).
- **Perf bar.** Ship at the performance-conformance bar: full-grid split, reads batched per block, whole-tile writes. Catalog `tile_reorder`: move whole pages; don't scatter faces.
- **Order: second.** "Different layouts" ranks below the sharding of Refinement 1. Neither refinement depends on the other.

**Done when**: the 23 cells above pass, with the specials bit-exact and Q3 within `block_step`. No regression on Phase 0 or Refinement 1. Ledger and design carry the new regime.

### [ ] Refinement 3 — Speed up the perf-flagged profile: DRAM write transactions on case 308

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags case 308 `[1,2048,2048,3]` (bf16 TILE, DRAM interleaved → RM bf16 DRAM, regime R2 `face_direct`) as the mandatory perf target. The lead's goal is ≤ 147.46 µs; today it runs at **3331–3432 µs**.

Measured (verification_report.md § Performance):
- Its 4.19 M 6-B DRAM writes cost 2167 µs on their own.
- Its reads alone take 517 µs.
- Together, the write storm congests the reads: per-core reader time grades from 0.40 ms (core row y = 11) to 2.57 ms (y = 2).

Cut the DRAM write transactions. The design's deferred **`same_bank_slot_writes`** regime does it: write each bank's run of consecutive 64-B page slots in one NoC write. The run is pages `p, p+B, p+2B, …`, which `TensorAccessor::contiguous_page_stride()` / `num_contiguous_pages()` give. That turns ~38 k writes per core into one write per DRAM bank per block. The slot image is built with 32-B local face-row copies. If those copies make one RISC issue-bound, split them across both DM RISCs (catalog `split_reader`). Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `tile_reorder` (move whole pages), `split_reader`, `noc_placement`. No SUPPORTED change.

**Verifier notes**:
- **Floor, not target.** The 147 µs goal is below this case's physical floor. Reads are at least 128 MiB (a 32-B face row is the read quantum for 3 live values). Writes are either the measured 2.17 ms of sub-slot transactions or 256 MiB of slot runs. That puts the floor near 0.8–1.0 ms. Measure progress toward the floor.
- **Bound regime.** Writing a slot's alignment gap is writing the output buffer's own memory, which is allowed. The regime is selected by shape (`2·W < out_aligned_page_bytes`).
- **Plan report.** Report it from `select_plan`, and update `test_regime_selection`'s pinned regime for 308/286 if the name changes (BB-08).
- **Ledger.** `cb_left_faces` gains a slot-image CB or changes consumer. Update its ledger row and the data-movement budget, since output bytes rise from 24 MiB to 256 MiB.

**Done when**:
- Device-ns improves on 308, measured.
- The loose 308 specials and poisoned-padding cases stay bit-exact, and the golden suite is green.
- No regression across the config-spanning guard set, one representative per kernel path × layout × placement:
  - R2: c286 (DRAM).
  - R1: c304 (full-row blocks), c294 (column-chunked), c2038 (L1).
  - One cell per path Refinements 1–2 added.

### [ ] Refinement 4 — Speed up the perf-flagged profile: read overlap and block co-tune on case 308

**Type**: perf

**Goal**: same flagged case 308 and the same 147 µs lead goal, on the topology Refinement 3 leaves. Today the R2 reader drains every block with one full `noc_async_read_barrier()`, so no read stays in flight across blocks. Reads alone run at 517 µs for 128 MiB (~260 GB/s with no writes competing).

Several cheap levers from `ttnn/ttnn/operations/examples/master.md` fit in one phase:
- trid-pipelined read batches across blocks (`trid_double_issue`; R1's reader already does this);
- `bank_stagger`, since per-core read time grades by core row;
- a co-tune of `pipeline_blocks_per_core` / `ROW_DEPTH` against the new writer (block size trades L1 for fewer handshakes; depth trades L1 for overlap). Whole-tile blocks minimum; coarser amortizes.

No SUPPORTED change.

**Verifier notes**: first re-profile 308 on Refinement 3's result. If the profile shows the reader off the critical path (writer/DRAM-bound with the reader idle), that measurement is the outcome. Record it in the changelog, and say which stage now bounds 308.

**Done when**: device-ns improves on 308 (measured), golden suite green, and no regression across the same guard set as Refinement 3. Alternatively, the post-Refinement-3 profile shows the read side off 308's critical path, recorded with numbers in the changelog.
