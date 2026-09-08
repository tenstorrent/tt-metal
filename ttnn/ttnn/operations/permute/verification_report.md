# Verification Report: permute

Phase 0 regime: `whole_tile_relocation` (fp32 / TILE / tile-aligned / rank 4 /
interleaved DRAM / inner-pair-preserving `dims`). Reader (NoC0) → `cb_tiles` → writer (NoC1),
one `ttnn.generic_op` dispatch.

## Code Review

**Fixed**

1. **Registry conformance bug (the only loud finding).** `SUPPORTED` carried a validate-only axis
   `inner_pair`, which the golden harness never generates. `eval.feature_matrix.unsupported_reason`
   treats a SUPPORTED axis missing from the cell as *unsupported*, so **every** generated cell —
   including the seven Phase 0 cells — was classified `unsupported`, producing
   `supported_pass = 0` and `xpass_drift = 7`. Fixed by moving the gate out of `SUPPORTED` into a
   module-level `SUPPORTED_INNER_PAIR` list checked explicitly in `validate()` (after the per-axis
   SUPPORTED loop, before EXCLUSIONS), still raising `UnsupportedAxisValue`. Semantics for external
   callers are unchanged: `permute(t, (0,2,1,3))` is still refused, never silently re-tiled.
   Post-fix: `supported_pass = 7`, `xpass_drift = 0`.

**Checked, no change needed**

- `INPUT_TAGGERS` are `(inputs, axes)`-signature, shape-derived only (`alignment`, `rank`), exactly
  as the prompt requires; `swap_hw` / `mem` / `inner_pair` are validate-derived, not taggers.
- `validate()` is the first statement of the public entry point; order is per-axis SUPPORTED →
  inner-pair gate → EXCLUSIONS (`EXCLUSIONS = []` at Phase 0).
- The op file does **not** declare `INVALID` (correct — it lives in `feature_spec.py`).
- One native dispatch: `ttnn.generic_op([input, output], descriptor)`; no `to_layout`/`tilize`/
  `to_memory_config` wrapper anywhere (prompt rule: MUST NOT — satisfied).
- Kernels use `#include "api/dataflow/dataflow_api.h"`, `void kernel_main()`, and `TensorAccessor`
  (no deprecated `InterleavedAddrGen`, no namespace pattern).
- CB sync balances exactly: reader `cb_reserve_back(block_tiles)`/`cb_push_back(block_tiles)` vs
  writer `cb_wait_front(block_tiles)`/`cb_pop_front(block_tiles)` — always whole-`BLOCK_TILES`
  quanta on both sides, including the ragged tail block (only `run` pages carry data), so quanta
  can never desync and the CB wrap can never split a block.
- No compute kernel is dispatched (pure relocation); `kernels/permute_compute.cpp` is an unused
  placeholder for the deferred transpose regime and is not in the `ProgramDescriptor`. No kernel_lib
  compute helper is bypassed. `mcast_pipe.hpp` is correctly *not* used — the op has no shared operand
  and no cross-core dependency at all.

**Design conformance** (`op_design.md`): algorithm (whole-tile relocation, no compute), pipeline
topology (reader NoC0 → 1 CB → writer NoC1), and work distribution
(`ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)`, linear output-tile range per core,
full compute-with-storage grid) all match. Both dataflow halves are batched — `BLOCK_TILES`
transactions per **single** barrier on read and on write, whole tile pages, never sub-tile faces.
Depth-2 CB gives the read/write overlap the design's stall-shadow argument relies on.

**Blocking-model fidelity**: `BLOCK_TILES` and `BUFFER_DEPTH` are module-level constants in
`permute_program_descriptor.py` and are each defined **once**; CB `total_size`, the block clamp and
both kernels' loop trip counts all derive from them via CT args (single source of truth — no
duplicated literal). `BLOCK_N`/`BLOCK_C` are likewise named constants and feed `tiles_per_plane`.
No CB is sized by a whole-op dimension: capacity is `BUFFER_DEPTH * BLOCK_TILES * page_bytes`,
independent of N/C/H/W. Expression check: reader, compute (absent) and writer all schedule at the
block boundary — one reserve/barrier/push and one wait/barrier/pop per block, per-tile loops sit
*inside* one block-scoped phase and introduce no per-unit completion boundary. The split's *count*
(cores) and *size* (`BLOCK_TILES = 8` tiles, whole tiles ≥ master.md's granularity floor) are both
turned. No collapsed or half-turned knob found.

## Registry Conformance

- `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()` present and correctly wired; entry point
  calls `validate()` first. No `INVALID` in the op file.
- One auto-fix applied (the `inner_pair` SUPPORTED-axis removal above). No SUPPORTED value was
  added or removed as a capability claim.
- **INVALID audit** (`eval/golden_tests/permute/feature_spec.py`): one entry,
  `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — the canonical bf8b+RM activation entry. Single
  tensor, no cross-tensor axis coupling, genuinely representational (not "not built yet"), and it
  does change the universe (both values are in TARGET). No norm-like weight axes here, so no
  no-weight canonicalization cells are expected. **No issues; no change requested.**

## L1 Ledger Audit

- **Currency**: the single row (`cb_tiles`) matches the one declared CB; the size expression
  `BUFFER_DEPTH * BLOCK_TILES * page_bytes` matches `permute_program_descriptor.py` exactly. (The
  ledger writes the page size as `tile_size(fp32)` = 4096 B while the code uses
  `buffer_aligned_page_size()`; identical for the Phase 0 fp32/TILE rectangle, and the code form is
  the more correct one to keep. Noted, no change.)
- **Capacity vs live set**: capacity exceeds the live set by exactly `BUFFER_DEPTH = 2` — that gap
  is the double-buffering mechanism and is stated as such. No axis is spanned whose capacity fails
  to scale with it: `ht`×`wt` is spanned jointly by `BLOCK_TILES`, `n`/`c` are streamed (extent 1)
  and correctly not spanned.
- **Page format vs DEST width**: `Float32` page with no compute kernel and therefore no DEST — the
  page format must equal the tensor format for a bit-preserving copy. Correct in both directions.
- **Disjoint lifetime**: only one CB, nothing to share with; justified in the row.
- **Bounds / closed form**: every symbol (`BLOCK_TILES`, `BUFFER_DEPTH`, `tile_bytes`) has a bound
  and an establishing predicate; the total is closed-form and contains **no** tensor dimension.
  64 KB/core (~4% of L1).
- **Data-movement budget**: input 1× / output 1× DRAM, 0 B cross-core — the interleaved minimum,
  consistent with the implemented split. The strictly cheaper *scheme* (sharded output, DRAM
  `2B → B`) is present as a `deferred` regime row with a positive reason and is queued as
  Refinement 1.
- **Block-size defaults**: interleaved default held — work units spread across the full grid, then
  the coarsest block that fits (`BLOCK_TILES = 8`, the measured 4–8 outstanding-reads sweet spot;
  L1 permits up to 32 at depth 4). No departure to justify.
- **Filing**: no ledger finding required a fix or a fold-in.

## Precision Baseline

`tests/ttnn/unit_tests/operations/permute/test_permute_precision_baseline.py`, fp32,
`dims=(1,0,2,3)`. permute is pure relocation, so the expectation is bit-exactness and the test
asserts it.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio (med / p5 / p95) |
|-------|-----|-------------|--------------|------------------|----------------------------------|
| (1,1,32,64)   | 1.0 | 0.0 | 0.0 | 0.0 | 1.000000 / 1.000000 / 1.000000 |
| (2,4,64,128)  | 1.0 | 0.0 | 0.0 | 0.0 | 1.000000 / 1.000000 / 1.000000 |
| (4,8,128,256) | 1.0 | 0.0 | 0.0 | 0.0 | 1.000000 / 1.000000 / 1.000000 |
| (2,4,512,512) | 1.0 | 0.0 | 0.0 | 0.0 | 1.000000 / 1.000000 / 1.000000 |

**Assessment**: bit-exact on every shape (`Max ATOL Delta: 0.0`), ratio spread degenerate at 1.0 —
no scale or structural error signature. Correct for a whole-tile relocation kernel.
**Recommended tolerances**: PCC ≥ 0.9999 with `rtol = atol = 0` for fp32/TILE; the golden-suite
per-dtype thresholds stay as-is for the dtype refinement.

## Verifier CLI Summary (post-fix, `/tmp/permute_results2`)

- supported_pass: 7
- xfail_expected: 433
- invalid_skipped: 88
- supported_fail: 0 ✓
- xpass_drift: 0 ✓
- xfail_wrong_mode: 0 ✓
- no_axes_found: 4 (the `test_regression.py` pins — untagged by design, all passed)

Acceptance suite: 10/10 passed. Precision baseline: 4/4 passed.

## Recommendations

- **The `xfail_expected = 433` bucket is the whole job**, not a success signal: every TARGET axis
  value beyond the Phase 0 rectangle (dtype bf16/bf8b, layout ROW_MAJOR, alignment w_/h_non_aligned,
  rank 2/3, swap_hw True, mem l1_sharded) is queued in `op_requirements.md`. Nothing from
  `TARGET − SUPPORTED` is undocumented.
- **Perf framing (from `eval/prompts/permute.txt`)**: this op is DRAM-bandwidth-bound with ~0% NoC
  congestion; the dominant lever is *reducing DRAM traffic* (sharding), not contention. That is why
  Refinement 1 leads with `mem=l1_sharded` even though it is a light structural change: it halves
  DRAM bytes. Barrier batching and block granularity (Refinement 3) are the second-order lever.
- **Memory pressure**: none. The footprint is dimension-independent at 64 KB/core; the bf8b/bf16
  dtype refinement *reduces* it, and the sharded regime replaces the CB with a zero-copy view of
  the resident shard — check the shard *tiles* against L1 there, since a shard is sized by the
  tensor, not by `BLOCK_TILES`.
- **Ragged-tail efficiency (not a refinement — no failing cell, no measured number)**: on the
  last block of a core, and on any block clipped by a plane boundary, the CB still advances a full
  `BLOCK_TILES` pages while only `run` carry data. This costs L1 turnover, not correctness or DRAM
  bytes, and keeps reader/writer quanta trivially in lockstep. If Refinement 3's sweep shows the
  clip firing often (small `tiles_per_plane`), advancing the CB by `run` instead is the natural
  co-change — measure before touching it.
- **`kernels/permute_compute.cpp`** is currently dead (not dispatched). Left in place deliberately
  as the seat for Refinement 2; if Refinement 2 is ever dropped, delete the file.
