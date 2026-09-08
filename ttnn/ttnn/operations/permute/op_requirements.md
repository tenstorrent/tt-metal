# Operation Requirements: permute

## Definition
- **Formula**: `output[i_{dims[0]}, ..., i_{dims[r-1]}] = input[i_0, ..., i_{r-1]}]` — pure relabeling of axes, no arithmetic, dtype preserved. Invariant: `permute(permute(x, dims), inverse(dims)) == x`.
- **PyTorch Reference**:
  ```python
  def permute_reference(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
      return torch.permute(x, dims).contiguous()
  ```
- **Import Path**: `from ttnn.operations.permute import permute`
- **Function Signature**:
  ```python
  permute(
      input_tensor: ttnn.Tensor,                 # bfloat16 | float32 | bfloat8_b
      dims: tuple[int, ...],                     # a permutation of range(rank)
      *,
      memory_config: ttnn.MemoryConfig = None,   # output placement (interleaved DRAM | L1 sharded)
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. A follow-up to a `[~]` partial appends a lowercase letter to the parent's number (`Refinement 1b`, `Refinement 1c`), ordered immediately after its parent. The parser matches exactly `Refinement \d+[a-z]?`.
> **Perf-claim rule (from `eval/prompts/permute.txt`)**: every perf refinement records the ceiling **target**, the tt-npe **pin** (cycles + DRAM util + congestion) and the **measured** device duration (median of the `/perf-measure` trial loop) in `changelog.md`.

### [x] Phase 0 — Core Implementation (`whole_tile_relocation`)

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment=tile_aligned, rank=4
- **SUPPORTED op-specific axes**: swap_hw=False, mem=dram_interleaved (+ validate-only gate: inner pair must be preserved)
- **Cores**: multi-core — `split_work_to_cores(grid, tensor_tiles, row_wise=True)`, linear output-tile range per core
- **Block knobs**: `BLOCK_TILES = 8`, `BUFFER_DEPTH = 2` (single source: `permute_program_descriptor.py`)
- **Compute config**: none — no compute kernel (pure relocation, bit-exact)
- **Golden baseline**: 7 / 532 cells passing (supported_pass=7, xfail_expected=433, invalid_skipped=88, all loud categories 0)

### [ ] Refinement 1 — `mem=l1_sharded` output placement + rank 2/3

**Goal**: add `"l1_sharded"` to `SUPPORTED["mem"]` and `2, 3` to `SUPPORTED["rank"]`.
*Sharding*: the design's `sharded_output` regime — the output CB becomes a zero-copy view of the
resident L1 shard via `ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor)`, the reader
writes straight into it and the DRAM writer stage disappears. Each core's shard is local (its own
linear output-tile range), so the work stays per-core: **no cross-core combine**. The core grid is
then dictated by the output shard grid, not by `split_work_to_cores`.
*Rank*: the index math in `permute_program_descriptor.py` and the reader is already written over a
generic `rank`; ranks 2 and 3 need the CT arrays and `tiles_per_plane` generalized off the hard-coded
`[4]` array width and the entry point's rank-4 assumptions dropped. Rank 2 with `swap_hw=False` is
the identity permute — keep it on the same path (it must still honour `memory_config`).

**Verifier notes**: knob-turn + placement change, not a new topology — hence bundled. Sharding leads
the queue because `eval/prompts/permute.txt` measures this op as DRAM-bandwidth-bound at ~0% NoC
congestion: removing the output DRAM crossing is the single largest lever (ledger: DRAM `2B → B`).
There is **no `/memory-layouts` pointer here** — that skill is RM/TILE layout, not placement; the
CB-placement pattern named above is the whole mechanism. Do NOT read the local shard back through a
`TensorAccessor`: an accessor read of a core's own shard means the axis was never implemented.
Update `l1_ledger.md` with the sharded regime's CB row — shard capacity is tensor-sized, not
`BLOCK_TILES`-sized, so state the L1 bound and the predicate that establishes it.

**Done when**: `mem=l1_sharded` and `rank ∈ {2,3}` cells move from `xfail_expected` to
`supported_pass`, the golden suite is green with all loud categories 0, and the sharded path is
verified zero-copy (no DRAM store in the writer for its own shard).

### [ ] Refinement 2 — `swap_hw=True` (within-tile transpose)

**Goal**: add `True` to `SUPPORTED["swap_hw"]` — the design's `within_tile_transpose` regime. A
compute kernel enters between the two existing CBs (`kernels/permute_compute.cpp`, currently a
placeholder): `compute_kernel_hw_startup()` → `transpose_init` / `transpose_tile`
(`tt_metal/hw/inc/api/compute/transpose.h`), with the destination tile index transposing
`(ht,wt) → (wt,ht)`. Keep the linear block schedule and the one-barrier-per-block batching; hoist
the transpose LLK init to once per block, not per tile.

**Verifier notes**: **scheme-change — stands alone.** A new pipeline stage (reader → cb_in →
compute → cb_out → writer) plus a strided output-tile order is the work; do not bundle it. No skill
in the inventory covers a raw-LLK transpose stage; `kernel_lib` has no transpose wrapper, so the raw
compute API is the intended mechanism (design §Mechanism table). Land this before the dtype and
layout refinements so those extend one final structure instead of two. Extend `l1_ledger.md` with
the second CB (two CBs, overlapping lifetimes — justify the non-sharing explicitly).

**Done when**: all `swap_hw=True` cells inside the otherwise-supported rectangle pass, the
round-trip invariant holds for transposing `dims`, and all loud categories stay 0.

### [ ] Refinement 3 — Perf: block/depth co-tune on the DRAM-bound interleaved path

**Type**: perf

**Goal**: no loose case is perf-flagged in `feature_spec.py`, so the region is the prompt's own
measured baseline profile: **`(2,4,512,512)`, `dims=(1,0,2,3)`, fp32/bf16, TILE, interleaved DRAM**
(stock lowering: 47.5 us, tt-npe DRAM util ~125%, congestion ~0%), plus the wide-inner shapes
`(1,2,128,4096)` and `(2,1,64,8192)`. Levers from
`ttnn/ttnn/operations/examples/master.md`: transactions-per-barrier / block granularity
(`BLOCK_TILES ∈ {4,8,16,32}`, whole tiles minimum — coarser amortizes barrier + CB
reserve/push/wait/pop) co-tuned with buffer depth (`BUFFER_DEPTH ∈ {2,3,4}`), grid occupancy for the
small-tensor tail (`MIN_TILES_PER_CORE`), and NoC assignment (reader NoC0 / writer NoC1 vs swapped;
`split_reader` only after confirming issue-bound rather than bandwidth-bound). Both are knob-turns
on the surface already exposed — keep the single source of truth in
`permute_program_descriptor.py`. If the plane clamp fires often, advancing the CB by the actual
`run` instead of a full `BLOCK_TILES` is the co-change to measure. No SUPPORTED change.

**Done when**: measured device-ns improves on the target shapes (`/perf-measure` median, with the
`/perf-ceiling-dm` target and the tt-npe pin — cycles, DRAM util, congestion — recorded in
`changelog.md`), congestion stays ~0 and DRAM utilization moves toward `dram_peak`, the golden suite
is green, and there is no regression across the config-spanning guard set (one representative per
kernel path × layout × placement: interleaved-TILE, sharded, transpose).

### [ ] Refinement 4 — dtype expansion (bfloat16 + bfloat8_b)

**Goal**: add `ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. Page size becomes
dtype-derived end-to-end (2048 B / 1088 B vs 4096 B) — it already flows from
`buffer_aligned_page_size()`, so the work is the CB format derivation, the transpose path's DEST
handling for the narrower formats, and exposing `ttnn.ComputeKernelConfig` on the transpose regime
only (the relocation regime has no DEST and must stay bit-exact). Cells that fail out of the box
(expect `bfloat8_b` + non-tile-aligned once Refinement 5 lands) go to `EXCLUSIONS`, not their own
refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: cheapest tier, so it lands after the structural work (Refinements 1 and 2) and
after the block knobs are tuned — the smaller pages change the transactions-per-barrier optimum, so
re-check `BLOCK_TILES` against Refinement 3's sweep rather than re-deriving it. The relocation path
must remain bit-exact for every dtype (it is a copy); only the transpose path has a precision story.

**Done when**: bf16 and bf8b cells pass across the then-supported rectangle (minus documented
`EXCLUSIONS`), and all loud categories stay 0.

### [ ] Refinement 5 — `ROW_MAJOR` layout + non-tile-aligned shapes

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` (the design's `row_major_stick`
regime — the **stick** is the transfer unit, extents computed from `W * element_size`) and
`"w_non_aligned"`, `"h_non_aligned"` to `SUPPORTED["alignment"]` (last-tile H/W zero-pad / mask done
in the reader or compute). Both must be **native in-kernel** data access.

**Implementation skill**: /memory-layouts

**Verifier notes**: prompt rule (MUST): kernels handle row-major natively — inserting
`ttnn.to_layout` / `ttnn.tilize` in the Python entry point is forbidden, not merely discouraged, so
the usual manipulation-op escape hatch is **not** available here; if the stick reader overruns a
focused pass, partial-tick on the alignment half and file `Refinement 5b` for the RM half rather
than wrapping. Bundled because the stick path and the ragged-edge masking touch the same reader /
writer index math. Note the `{bfloat8_b, ROW_MAJOR}` cells are INVALID and stay skipped. Stick size
may fall below the DRAM-efficient transfer size — measure it, it is the input to Refinement 6.

**Done when**: RM and non-aligned cells pass (minus INVALID and documented `EXCLUSIONS`), and all
loud categories stay 0.

### [ ] Refinement 6 — Perf: the sharded and row-major regimes

**Type**: perf

**Goal**: re-target the two regimes that Refinements 1 and 5 introduced and that Refinement 3 could
not measure. (a) **Sharded** `(2,4,512,512)` / `(4,8,128,256)`: confirm the DRAM-traffic halving is
realized (`/perf-ceiling-dm` Step 4b + local-L1 `--loopback`), and tune one-block-per-shard against
any mechanism cap. (b) **Row-major** `(128,8192)` / `(2,512,1024)`: sticks are often below the
DRAM-efficient transfer size, so coalesce contiguous sticks into one transaction and re-tune
transactions-per-barrier — same `master.md` granularity levers, different unit. No SUPPORTED change.

**Done when**: measured device-ns improves on both target groups with target / tt-npe pin / measured
recorded in `changelog.md`, congestion stays ~0, the golden suite is green, and no regression across
the config-spanning guard set.
