# Operation Requirements: g6_untilize

## Definition
- **Formula**: `out[b, r, w] = x[b, r, w]`, bit-identical for every uint32 pattern. `x` is TILE (32×32 tiles,
  faces TL, TR, BL, BR, row-major inside a face); `out` is a new ROW_MAJOR tensor of the same shape, dtype
  and placement (DRAM interleaved), with one page per row `(b, r)`.
- **PyTorch Reference**:
  ```python
  def pytorch_g6_untilize(input_tensor, *, use_multicore=True, sub_core_grids=None):
      return input_tensor.clone()
  ```
- **Import Path**: `from ttnn.operations.g6_untilize import g6_untilize`
- **Function Signature**:
  `g6_untilize(input_tensor: ttnn.Tensor, *, use_multicore: bool = True, sub_core_grids: ttnn.CoreRangeSet | None = None) -> ttnn.Tensor`.
  `use_multicore` must be `True` and `sub_core_grids` must be `None`; any other value raises `ValueError`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. A partial's follow-up appends a lowercase letter to the parent's number (`Refinement 1b`) and is ordered immediately after its parent. The runner's parser matches exactly `Refinement \d+[a-z]?` in a `### [ ] … — …` heading.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [uint32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: none. The spec forbids alignment, rank and size axes. Every rank-4
  uint32 TILE (32×32) DRAM-interleaved shape is handled; a probe of 15 ragged, large and multi-block
  configurations was bit-exact.
- **SUPPORTED op-specific axes**: none (`use_multicore=True` and `sub_core_grids=None` are fixed by the spec)
- **Cores**: one rectangle from `choose_core_rectangle(B·H, grid, CORE_LIMIT)`; production 8×4 = 32 cores,
  1 row each
- **Compute config**: none (data movement only, no compute kernel)
- **Golden baseline**: 6 / 6 passing. Verifier CLI: 2 supported_pass, 0 supported_fail, 0 xpass_drift,
  0 xfail_wrong_mode, 0 xfail_expected, 4 untagged regression tests (passed).

## Queue

No refinement entries are filed. Both halves of the queue are empty for a structural reason, not for
lack of review:

- **Generality.** `TARGET − SUPPORTED = ∅`: dtype {uint32} and layout {TILE} are both supported, and the
  golden suite has no `xfail_expected` cell.
- **Perf.** The only golden cell is the perf-flagged `LOOSE_CASES[0]` `(1,1,32,128)`, which is 4 tiles.
  `eval/prompts/perf_rules.txt` (injected into every perf refinement) limits measurement to cells of more
  than 32 tiles and overrides any instruction to target a smaller one. BB-22 forbids adding test shapes,
  and `feature_spec.py` is read-only. A perf entry on the flagged case would therefore have an unmeetable
  device-ns gate. A perf entry on any other shape would break both the flagged-case rule and BB-22.

The perf position and the unexplored levers from `op_design.md`'s perf lamps are recorded in
`verification_report.md` (Recommendations and Findings). They become queue material only if the tile
threshold is lifted for this op or the model trace becomes the measurement:

- `CORE_LIMIT` grid-sync sweep (knob-turn)
- `rows × tile_cols` split (knob-turn)
- fused single-kernel gather+store
- rectangle aspect / origin (`noc_placement`)
