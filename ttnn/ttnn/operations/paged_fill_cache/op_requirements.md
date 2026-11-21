# Operation Requirements: paged_fill_cache

## Definition
- **Formula**: for every position `p < seq` whose block `b = page_table[batch_idx, p // block_size]` is not `-1`, every head `h` and every `d < D`: `cache[b, h, p % block_size, d] = input[0, h, p, d]`, in place; every other cache element is unchanged.
- **PyTorch Reference**: `reference(cache_tensor, input_tensor, page_table, batch_idx=0)` in `eval/specs/paged_fill_cache/spec.py` (float64; `cache[page_table[batch_idx, k], :, :n, :] = input[0, :, k*block_size : k*block_size + n, :]` for each block `k` of the sequence whose entry is not `-1`).
- **Import Path**: `from ttnn.operations.paged_fill_cache import paged_fill_cache`
- **Function Signature**: `paged_fill_cache(cache_tensor: ttnn.Tensor, input_tensor: ttnn.Tensor, page_table: ttnn.Tensor, *, batch_idx: int = 0) -> ttnn.Tensor` (returns `cache_tensor` itself)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding** (the input's, `eval.memory.memory_axes`): DRAM / interleaved
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only
- **Cores**: full grid — the input's tiles split tile-granular over `compute_with_storage_grid_size()` (32 clusters on Quasar), row-wise, each cluster's contiguous range streamed through an 8-entry implicit-sync ring
- **Thread plan**: (R, C, W) = (1, 0, 1): one reader and one writer DM thread, no compute kernel (data-movement bound bit-exact copy)
- **Compute config**: none (no compute kernel; the spec gives the op no compute config)
- **Golden baseline**: 15 / 15 supported cells pass (12 matrix + 3 loose), 36 xfail_expected, 11 / 11 rejections, 0 loud (`verifier_report.json`)

### [ ] Refinement 1 — float32 and an L1-interleaved input

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` (the cache and the input alike: the cell's dtype) and `"L1"` to `SUPPORTED["memory_config"]` (the input interleaved in L1; `SUPPORTED["sharding"]` stays `interleaved`, the only TARGET value). This takes SUPPORTED to TARGET on every axis and moves the 36 `xfail_expected` cells (float32 × {DRAM, L1}, bfloat16 × L1) to passing.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Both values are **validate()-only knob-turns** of the one built regime (op_design.md: R-filter, rows R-L1-input and the `dtype: float32` note). The verifier ran them with validate() bypassed (`create_program_artifacts` + `generic_op`): float32 and bfloat16, input in DRAM and in L1, on `[8,2,64,1024]←[1,2,160,1024]` with a `-1` entry, the Llama call and the 8192-wide row: **12 / 12 bit-exact, no kernel change**. The tile ring's entry size and format already come from `input_tensor.buffer_page_size()` / `input_tensor.dtype`, and the `TensorAccessor` resolves interleaved L1 banks like DRAM ones. Expected diff: SUPPORTED, plus a test.
- The two values are bundled because each alone is far below a tenth of the Phase 0 build. Of `/numeric-formats-metal`, only the dtype half applies: the op has **no compute kernel and no `compute_kernel_config`** (spec: "It has no compute config"; R1: no element passes through another dtype). Do not add a config parameter or an fp32 intermediate. The tile ring is a raw dataflow payload whose format must equal the tensors' (R1, `exact`).
- An L1-interleaved input is **not** a local shard: reading it through the `TensorAccessor` is the native path here (`/memory-layouts` does not apply; no placement skill exists). Sharded inputs stay refused by `memory_axes` (R7); TARGET has no sharded value.
- Keep R8: the cache stays DRAM interleaved, and an input whose dtype differs from the cache's is refused with `NotImplementedError`.
- Footprint at float32: `dfb_input_tiles` = 8 × 4096 B = 32 KB per cluster (ledger), far under the ring cap and L1.
- Lands first so that Refinement 2's no-regression guard set can span float32 and the L1 input.

**Done when**: `SUPPORTED["dtype"] == [ttnn.bfloat16, ttnn.float32]` and `SUPPORTED["memory_config"] == ["DRAM", "L1"]`; the golden suite has 0 `xfail_expected` matrix cells and 0 loud categories; `test_paged_fill_cache_plan.py` also runs a float32 case and an L1-input case (including the one-device-session test with `to_layout` after the op).

### [ ] Refinement 2 — Cut the per-cluster fixed cost on small calls (page-table row loads)

**Type**: perf

**Goal**: no loose case is perf-flagged, so this region was chosen by the verifier: the **small calls**. Most INPUTS entries, the traced Qwen3-VL call among them (16 tiles → 16 clusters × 1 tile), give each engaged cluster one tile. Every cluster first makes **two DRAM reads of the same 64-B page-table page**: both copies of row `batch_idx`, from the one bank that holds it, so 32 reads converge on one bank in the Qwen call and 64 in the Llama and 8192-wide calls. Only then is the tile read and written. Reduce that fixed cost, with levers from `ttnn/ttnn/operations/examples/master.md` (`double_buffer` on bytes in flight vs. fixed per-batch cost; `noc_placement` on the split staying row-wise) and the design's grid-synchronization perf lamp:
- (a) **one DRAM read of the row per cluster.** Take the second copy from the first in the cluster's own L1: a NoC L1→L1 self-read tagged `{.trid = 0}`, or a RISC copy with the DM L1 data-cache flush/invalidate the dataflow API requires before another hart reads it. The ring protocol stays as it is: ROW_COPIES pushed and ROW_COPIES popped, entry 0 for the writer before the first tile read.
- (b) optionally, **the split unit**: deal tiles to clusters in units of one txn-id half (`SPLIT_UNIT_TILES = TILE_RING_ENTRIES / 2`) when the input has fewer tiles than `grid × unit`. That engages fewer clusters, each with a full half-ring in flight, and fewer clusters paying the row load. It must leave the full grid engaged on inputs of at least `grid × unit` tiles (Llama, 4096/8192-wide), and stay row-wise. Unit and ring depth are co-tuned knobs; both stay single-source host constants.

No SUPPORTED change.

**Done when** (structural counts: the ttsim backend has no clock):
- the page-table DRAM reads per call, read off the op by a test or a probe, go from `2 × engaged clusters` to `≤ 1 × engaged clusters` on every INPUTS entry (Qwen traced call: 32 → ≤ 16, or ≤ 4 if lever (b) is taken);
- the guard set's other structural counts do not regress: each mapped input tile read once, each written cache tile written once, the cache never read, implicit sync on the tile ring, full grid engaged for inputs ≥ `grid × unit` tiles. The guard set is one representative per path × placement: bfloat16 and float32, input in DRAM and in L1 (after Refinement 1), a `-1` entry, the 8192-wide row and the Llama call;
- the golden suite is green, the one-device-session test passes, and the changelog records the counts before and after. Say there and in the outcome that the gain in time is **unconfirmed** until a timed backend (ttsim-perf) or hardware measures it.

**Verifier notes**: waits on Refinement 1, so the guard set covers the float32 and L1 paths. This is also where the ledger's deferred traffic item lands (`l1_ledger.md`, data-movement budget). Do not reopen R-mcast-pt (a broadcast of the row across clusters, rejected in the design) or turn the thread plan.
