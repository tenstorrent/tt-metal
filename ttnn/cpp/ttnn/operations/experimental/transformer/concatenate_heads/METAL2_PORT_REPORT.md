# Metal 2.0 Port Report — experimental/transformer/concatenate_heads

## Outcome

**PORTED, runtime-unverified.** The single factory (`ConcatenateHeadsProgramFactory`, new) is on
`ProgramSpecFactoryConcept`. It builds cleanly and passes the static self-audit. **No test has exercised it**: the
op needs a ≥ 12 × 9 compute grid, and the porting machine has an 8 × 8 Wormhole. So the baseline (pre-port) and the
post-port run both report `25 skipped` ("Grid size 8-8 is not supported"). Spec validation, the kernel JIT compile
and numerics are therefore **still owed on a Blackhole**. The invoker chose to commit with this caveat (2026-10-06).

To verify on a Blackhole with the legality-check scaffolding applied (see Open items):

```bash
export TT_METAL_WATCHER=10
pytest models/experimental/bert_large_performant/unit_tests/test_bert_large_concatenate_heads.py -v
# expect 25 passed, and both METAL2_CHECKS_FORCED markers in the log
```

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
  (from the `Port_Recipe` checkout; `git log` over the docs path prints nothing in `Metal_Ports`)
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, as the audit chose. There is no `override_runtime_arguments`. The two tensor bindings
(`input`, `output`) are the whole cache-hit refresh. That matches the legacy behaviour, where the two `Buffer*` RTAs
were the only per-dispatch state.

### Device-op-class edits
- **Exception 3 (direct-descriptor shape).** The op arrived with `create_descriptor` as a static member of
  `ConcatenateHeadsDeviceOperation`, and no `program_factory_t` (still true at port time). Changes in
  `device/concatenate_heads_device_operation.hpp`:
  - added nested `struct ConcatenateHeadsProgramFactory` with `create_program_artifacts`
  - added `using program_factory_t = std::variant<ConcatenateHeadsProgramFactory>;`
  - removed the device-op-level `create_descriptor`

  Its cache-refresh comment moved onto the new method, with "runtime-arg bindings" reworded to "bindings". The
  includes swapped `program_descriptors.hpp` for `<variant>` and `ttnn/metal_v2_artifacts.hpp`. The body stays in
  `device/concatenate_heads_program_factory.cpp`.
- Pybind entry points removed: none. The only binding is the user function.
- Custom `compute_program_hash`: none. The default hash is untouched.

### Open items
- Relaxation candidates: none considered (strict matching kept).
- No concept-fit friction.

## Handoff points

- **Direct-descriptor op converted to a conventional factory** (exception 3, above). Owner: TTNN. This is
  informational: the shape that PR #57409 introduced is undone for this op.
- **Readiness sheet is stale** (carried from the audit waiver). The sheet row still says `legacy device-op` /
  `ConcatenateHeadsProgramFactory` at a deleted `.hpp`. After this port the factory name exists again, now in
  `concatenate_heads_device_operation.hpp`, on `ProgramSpecFactoryConcept`. Owner: readiness-sheet owner.
- **Runtime verification owed on Blackhole** (see Outcome). Owner: invoker.

## Successes

- **Kernel-side whitelist rule 7** (DFB metadata via the object). Both kernels' `get_tile_size(cb_id)` were
  non-`constexpr` locals. So they became `dfb_in0.get_tile_size()` / `dfb_out0.get_tile_size()` (reader `:26`,
  writer `:27`), with no `.id` extraction. The `constexpr` carve-out correctly did not apply.
- **Self-audit "Buffer* not only `->address()`" note.** The legacy factory delivered addresses as
  `emplace_runtime_args(core, {in0_buffer, ...})`, with no `->address()` anywhere. The recipe's warning to grep
  `emplace_runtime_args` / `Buffer*` as well is exactly what would catch a missed binding on this op.
- **TT_FATAL census** kept the four debug-only `TT_ASSERT`s in the factory (`factory.cpp:41,61,62,68`) on their
  original `ttnn::Tensor::buffer()` subjects, rather than rewriting them against `MeshTensor`. The counts are unchanged
  (8 in the device op, 4 in the factory).
- **Endpoint census.** Re-derived from the kernels: one DFB, reader PRODUCER and writer CONSUMER, plain 1:1. This
  agrees with the brief.

## Friction

- **Gap — DM helper signature.** Recipe §Hardware configuration shows `create_reader_datamovement_config(device->arch())`.
  In this tree the helper (`ttnn/cpp/ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp:24,32`)
  takes no arch argument, only `bool disable_dfb_implicit_sync_for_all = false`. I used the no-arg form, as the
  earlier `bcast_to` port did.
- **Gap — hardware the test needs vs hardware the porter has.** Neither the audit nor the recipe treats "the op's
  only tests can't run on the available card" as a pre-port check. The audit brief did note the 12 × 9 requirement,
  but only under Verification. It would save a round-trip to surface it at audit time as a verification-environment
  prerequisite, e.g. "requires Blackhole".
- **Confusion — accessor naming for one buffer seen by two kernels.** The legacy writer names the same CB `cb_out0`
  ("same as cb_id_in0"). The recipe doesn't say whether accessor names should be unified per DFB or kept per kernel.
  I kept the per-kernel legacy names:
  - spec `in0`
  - reader accessor `in0`
  - writer accessor `out0`

  This is the minimal kernel diff.
- **Environment.** The workspace's Python bindings were stale (`import ttnn` failed on a missing
  `ChunkGdnWyInverse`), so a full rebuild was needed before a baseline could run. The recipe docs also live in a
  separate checkout, so provenance comes from there.

## Open items for downstream

- **Verification on Blackhole** is the only thing between this and a confirmed port. Apply the `skip_validation`
  force and the `METAL2_CHECKS_FORCED` markers in `tt_metal/impl/metal2_host_api/{program_spec,program_run_args}.cpp`
  (not committed), then run the command under Outcome.
  - Watch `test_bert_large_concatenate_heads_with_program_cache`. It is the cache-hit check for the new tensor
    bindings, and expects 2 cache entries.
  - Watcher matters here because of the writer's over-pop (next bullet).
- **Writer pops 34 tiles against 32 pushed** (`writer_tm_tile_layout_concat_heads.cpp:46`, carried verbatim; audit
  Misc anomalies). On Gen1 the DFB lowers to the same CB, so behaviour should match legacy. If Watcher or a DFB
  check objects on Blackhole, that's the first suspect. It is not a port defect.
- **Shared kernel touches:** none. Both kernels are private to this op and were converted in place.
- **Carried audit anomalies** (not acted on):
  - The 64-tile "double buffer" DFB is never double-buffered (`factory.cpp:95`).
  - The grid checks are debug-only `TT_ASSERT`s.
  - `compute_with_storage_grid_size` is hashed but only feeds asserts.
  - Input layout isn't validated.
  - The nanobind doc understates dtype support.
- **Per-op carry-over:** `experimental/transformer/create_qkv_heads_from_separate_tensors` has the same #57409
  direct-descriptor shape (`create_descriptor` on the device op, `..._device_operation.hpp:26`), so it will need the
  same exception-3 edit. Check its test's grid requirement against the available card before starting.
