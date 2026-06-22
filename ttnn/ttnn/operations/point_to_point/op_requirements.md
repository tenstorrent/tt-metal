# Operation Requirements: point_to_point

## Definition
- **Formula**: `output[receiver_coord] = input[sender_coord]` (identity copy over the
  Tenstorrent fabric); `output[d]` for `d != receiver_coord` is preserved (supplied
  `output_tensor`) or uninitialized (fresh allocation). No arithmetic.
- **PyTorch Reference** (single sender→receiver shard copy across a row/column-adjacent mesh):
  ```python
  def point_to_point_ref(global_input, sender_lin, receiver_lin, S, output=None):
      # global_input: (num_devices*S, ...) sharded along dim 0; S = per-device shard size
      out = global_input.clone() if output is None else output.clone()
      out[receiver_lin * S : (receiver_lin + 1) * S] = global_input[sender_lin * S : (sender_lin + 1) * S]
      return out
  ```
- **Import Path**: `from ttnn.operations.point_to_point import point_to_point`
- **Function Signature**:
  ```python
  point_to_point(
      input_tensor: ttnn.Tensor,
      sender_coord: ttnn.MeshCoordinate,
      receiver_coord: ttnn.MeshCoordinate,
      topology: ttnn.Topology = ttnn.Topology.Linear,
      output_tensor: ttnn.Tensor | None = None,
      intermediate_tensor: ttnn.Tensor | None = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to
> update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete + tests pass, `[~]` real work landed but an axis value
> deferred, `[ ]` nothing usable produced yet.

### [x] Phase 0 — Core Implementation (full TARGET rectangle claimed)

- **SUPPORTED dtype**: `[bfloat16, float32, bfloat8_b, uint16, int32, uint32]` — the entire
  TARGET dtype universe.
- **SUPPORTED layout**: `[TILE, ROW_MAJOR]` — the entire TARGET layout universe.
- **SUPPORTED topology**: `[Linear, Ring]` — the entire TARGET topology universe.
- **SUPPORTED alignment** (INPUT_TAGGERS-derived): `[tile_aligned, non_tile_aligned]` — both.
- **EXCLUSIONS**: `[]` (empty).
- **Memory**: interleaved DRAM or L1 (sharded rejected by `validate()`; not a TARGET axis).
- **Cores / links**: single worker core `{0,0}`, single fabric link (`link_idx = 0`).
- **Coordination**: one fresh `GlobalSemaphore` per call; 2-party handshake; cache-reuse reset.
- **Golden baseline**: **0 / 0 cells** executed — the golden suite could not run in this
  environment (see "Why the queue is empty" below). Single-device Python-path tests: 22 / 22
  pass. Cross-device transfer: **unverified on hardware** (1 device available, op needs ≥ 2).

---

## Why the refinement queue is empty

This op is a **pure cross-chip byte copy** (no compute / unpack / math / pack). The Phase 0
implementer therefore claimed the **entire** TARGET rectangle on every axis:

| Axis | TARGET | SUPPORTED | `TARGET − SUPPORTED` |
|------|--------|-----------|----------------------|
| dtype     | bf16, f32, bf8b, uint16, int32, uint32 | (same) | **∅** |
| layout    | TILE, ROW_MAJOR | (same) | **∅** |
| topology  | Linear, Ring | (same) | **∅** |
| alignment | tile_aligned, non_tile_aligned | (same) | **∅** |

`TARGET − SUPPORTED = ∅` on **every** axis. Per the refinement-queue hard rule, a valid entry
must either *(a)* add a value to `SUPPORTED[axis]`, or *(b)* move named failing cells out of a
failure category (`OOM` / `numerical-precision` / `numerical-bug` / `hang`) into passing.

- **(a) is empty** — there is no `(axis, missing_value)` pair to add. The one structurally
  impossible cell (bf8b + ROW_MAJOR) is correctly in `feature_spec.INVALID`, not a candidate.
- **(b) is empty** — the golden suite produced **zero** cells (verifier `total: 0`), so there
  are no named failing cells to move. Every loud category (`supported_fail`, `xpass_drift`,
  `xfail_wrong_mode`) is `0` **because nothing ran**, not because the op was proven clean.

Everything else that surfaced in review is either a **report recommendation** (multi-link
throughput; the redundant first host barrier) or a **verification/infrastructure blocker**
(below) — none of which add a SUPPORTED value or move a tracked failing cell, so by the hard
rule they do **not** belong in this queue.

There are therefore **no Phase-1+ refinements to file.** The honest next step is *verification*,
not refinement.

---

## Verification blockers (NOT refinements — close these to confirm Phase 0)

These do not unlock golden cells; they unlock the ability to *measure* the cells Phase 0 already
claims. Listed here so the queue is a complete picture of what stands between "claimed" and
"confirmed."

1. **≥ 2-device mesh (or a configured simulator).** The transfer needs a sender and a receiver
   device. This pass had 1 Blackhole device and no simulator; the global safety rule forbids
   changing the IRD reservation. All 59 multi-device acceptance/extended/precision cells skipped.
2. **`ttnn.operations._op_contract` module is missing in this branch.** `eval/golden_tests/
   conftest.py` imports `SupportRefusal` from it; the import fails (`ModuleNotFoundError`),
   aborting golden collection for *every* op. Must exist before any golden suite runs.
3. **`point_to_point` golden harness wiring is missing.** The golden dir has only
   `feature_spec.py` — no `test_golden.py` / `helpers.py` / `conftest.py` / `axes.py`, and the
   stock single-`device`-fixture harness does not fit a multi-device op. Generate it via
   `/golden-tests`, adapted to a `mesh_device` fixture + sender/receiver coordinates.

Once 1–3 are resolved, re-run `eval/eval_test_runner.sh` + `python3 -m eval.verify_supported`.
If a SUPPORTED cell then fails: a *structural* gap → `EXCLUSIONS`; an `OOM` / `numerical`
failure → a real refinement filed at that point. Until then, do not narrow SUPPORTED — the
pure-byte-copy reasoning makes the full-rectangle claim defensible (bf8b + non_tile_aligned
copies full padded tile pages bit-exactly; no compute path, so no `/numeric-formats-metal`
bf8b+non-aligned exclusion is warranted).
