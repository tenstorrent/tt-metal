# Port Report — nlp_concat_heads

Metal 2.0 port of `nlp_concat_heads` (single `NLPConcatHeadsProgramFactory`, both runtime-selected paths).
**Verified on Blackhole p150b: `test_nlp_concat_heads.py` → 217 passed, 0 failed.**

> Supersedes the earlier grounded-stop conclusion: the interleaved path's cross-op writer blocker was resolved by forking the writer (patterns catalog: *Fork with `_metal2` suffix*), so the whole factory ported.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`. Single-program, no op-owned device resources. The one factory builds two spec shapes by `input.is_sharded()` (both paths now Metal 2.0, so the atomic-unit rule is satisfied — the factory never selects an unconverted kernel).

### Device-op-class edits
- Custom `compute_program_hash` deleted: none.
- Pybind entry points removed: none.

## Handoff points

- **Cross-op writer fork (eltwise/unary).** The interleaved path bound the shared `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, which reads a positional CTA + `TensorAccessorArgs<1>()` + a buffer-address RTA — none emittable by a Metal 2.0 factory. That writer has ~31 co-borrowers, so it cannot be edited in place. Per the patterns catalog, forked it as `device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (in this op's own dir), converted the fork to named bindings (`dfb::in` / `ta::output` / named RTAs), and pointed the interleaved writer `KernelSpec::source` at the fork. The legacy copy is untouched.
  - **Sunset:** delete the fork when the shared eltwise writer is itself Metal-2.0-prepped (then all co-borrowers, including `nlp_concat_heads_boltz`, can use it directly).
  - **Drift discipline:** the fork only implements the forward, non-sharded-output path nlp_concat_heads uses (the legacy `OUT_SHARDED` / `BACKWARDS` branches were dropped); bug fixes to the legacy writer affecting that path should be mirrored until sunset.

## Successes

- **Sharded path = nlp_concat_heads_decode shape.** Input (c_0) and output (c_16) are borrowed-memory CBs used as local address source/sink for a self-NoC head-rearrange (no FIFO). Bound `reader=PRODUCER` / `writer=CONSUMER` of each (the two kernels split nheads across the RISCs on the same nodes) — the same validator-satisfying split proven on `nlp_concat_heads_decode`.
- **Interleaved path = clean Case-1.** Input read page-by-page via `TensorAccessor(ta::input)` into the src CB (c_0, `reader=PRODUCER`); the forked writer consumes it (`writer=CONSUMER`) and writes pages to `TensorAccessor(ta::output)`. The legacy `Buffer*` RTAs + `TensorAccessorArgs` CTAs all collapse to `TensorParameter`/`TensorBinding`.

## Friction

- **Confusion — fork placement.** The catalog says to place the `_metal2` fork "alongside the original" (in eltwise/unary's dir), but the recipe's scope boundary says stay in the op's own dir. Resolved in favor of the op's own dir (`nlp_concat_heads/device/kernels/dataflow/`) — self-contained and scope-respecting. Worth a catalog clarification on which location is preferred.

## Open items for downstream

- **Cross-op kernel fork (coordination signal).** `writer_unary_interleaved_start_id_metal2.cpp` is a fork of the eltwise/unary writer. Remaining unmigrated co-borrowers of the legacy writer: ~31 factories (incl. `nlp_concat_heads_boltz`, which shares the same shape and could reuse this fork or a shared Metal-2.0 writer once one exists). Sunset the fork when the shared writer is Metal-2.0-prepped.
- **Fake-CB self-loops (sharded path).** `in`/`out` borrowed DFBs are validator-satisfying address source/sink bindings, not real FIFOs (tensor-local-view) — replace when the local-`TensorAccessor` mechanism lands.
- **Test coverage:** `test_nlp_concat_heads.py` exercises both paths across shape/sharding sweeps — 217 passed on Blackhole p150b.
