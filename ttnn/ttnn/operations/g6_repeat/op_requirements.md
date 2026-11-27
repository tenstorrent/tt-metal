# Operation Requirements: g6_repeat

## Definition
- **Formula**: `out[i0, i1, i2, i3] = x[i0 mod d0, i1 mod d1, i2 mod d2, i3 mod d3]`, out shape `(d_k * reps[k])`. Phase 0: `out[0, 0, h, w] = x[0, 0, 0, w]`, `h < R`, `w < W`, every logical element bit-identical.
- **PyTorch Reference**: `def pytorch_g6_repeat(x, reps): return x.repeat(*[int(r) for r in reps])`
- **Import Path**: `from ttnn.operations.g6_repeat import g6_repeat`
- **Function Signature**: `g6_repeat(input_tensor: ttnn.Tensor, reps: list | tuple) -> ttnn.Tensor` (reps positional, four positive ints; out takes the dtype, layout and memory config of `input_tensor`)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. A `[~]` partial's follow-up appends a lowercase letter to the parent's number (`Refinement 1b`), ordered immediately after its parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = h_non_aligned (out is one partial tile row)
- **Contract** (`validate()`, `ValueError`): x `(1, 1, 1, W)`, `W % 32 == 0`; reps `[1, 1, R, 1]`, `1 <= R < 32`; DRAM interleaved, on device
- **Cores**: `out_tile_col` split over `min(Wt, grid)` cores, whole assignment per block (clamped by the L1 budget); p1 uses 8 cores, 1 tile each
- **Compute config**: none; the op does no arithmetic (one data-movement kernel, NoC byte moves only)
- **Golden baseline**: 5 / 6 registry cells passing. The one `supported_fail`, `poisoned_padding_nan`, is a golden-harness precondition defect (see `verification_report.md`); the op passes that property in the acceptance test.

## Refinement queue: empty

No `Refinement` entry is filed. This is the full accounting, not an omission.

**Generality: TARGET − SUPPORTED is empty on every axis.**

| Axis | TARGET (`feature_spec.py`) | SUPPORTED (`g6_repeat.py`) | Gap |
|---|---|---|---|
| dtype | [bfloat16] | [bfloat16] | none |
| layout | [TILE] | [TILE] | none |
| alignment | ["h_non_aligned"] | ["h_non_aligned"] | none |

`xfail_expected` is 0 and `INVALID` is empty, so no out-of-SUPPORTED cell exists to unlock.

**Perf: no golden cell is measurable.**

- `eval/prompts/perf_rules.txt` permits measuring only cells of more than 32 tiles, and says that
  rule overrides any instruction to target a smaller one.
- The only perf-flagged case, `LOOSE_CASES` p1 (x `(1,1,1,256)` → out `(1,1,16,256)`), is 8 tiles.
- Every other golden INPUT and LOOSE_CASE is the same 8-tile shape.
- So the 2:1 cadence has no slot it can legally fill.

**Failure-category moves: none available.**

- The only failing cell's category is `other`, a HARNESS PRECONDITION that fails before `g6_repeat`
  is dispatched.
- No change to the op can move it. Its owner is the golden suite (`verification_report.md`, options (a)/(b)).

**Re-open conditions.** The queue reopens if any of these happens:

- **`TARGET` gains a value.**
  - A placement axis: per the design's Blocking Model, a WIDTH_SHARDED out is a knob-turn on
    `out_tile_col` (a local shard; the slots back onto the shard via
    `cb_descriptor_from_sharded_tensor`, and the store becomes a no-op). An `out_tile_row` cut is a
    scheme-change: regime 3 with a `ttnn.Mcast` of the built tile.
  - Multi-tile-row out, or `reps[0..1] > 1`: regime 2, reachable without restructuring.
- **A production case above 32 tiles enters `LOOSE_CASES`.** The first perf refinement would then
  target it with the design's lamps L4 (writes on NOC_1, or a reader/writer split) and L6
  (write only faces 0/1 when R ≤ 16), plus a co-tune of `REPEAT_BUFFER_DEPTH` / `block_width_tiles`.
  Measured headroom at a non-golden 300-tile shape: 5.3 µs against a ~1.8 µs DRAM-byte bound.
