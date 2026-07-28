# Metal 2.0 Port Report — `pool/upsample`

## Outcome

**PORTED** — three factories of the `UpsampleOperation` device operation converted to `MetalV2FactoryConcept`:
- `UpsampleMultiCoreInterleavedProgramFactory` (integer-scale interleaved, row-major and tiled)
- `UpsampleMultiCoreShardedProgramFactory` (integer-scale sharded, `WorkloadDescriptor` → single-program with an op-owned config tensor)
- `UpsampleNearestFloatProgramFactory` (float-scale general path)

`UpsampleBilinearProgramFactory` is **not** ported (audit-blocked: Device 2.0 gate — see `METAL2_PREPORT_AUDIT.md`). It stays on the legacy `ProgramDescriptor` API, untouched; `UpsampleOperation::select_program_factory` still dispatches to it for `mode == "bilinear"`.

Verification (on `blackhole`, this session's actual hardware — see Friction below):
- `tests/ttnn/unit_tests/operations/pool/test_upsample.py` — **323 passed, 96 skipped, 0 failed** (`SAFE_PYTEST_RESULT: PASS`). Coverage spans all three ported factories plus the untouched bilinear path.
- `tests/ttnn/nightly/unit_tests/operations/pool/test_upsample.py` — **231 passed, 20 skipped, 0 failed** (`SAFE_PYTEST_RESULT: PASS`).

## Provenance

- **Recipe docs (this port):** `a21c8f3f324 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`
- **Audit docs (inherited):** `a21c8f3f324 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for all three factories, as the audit decided. No deviation. `UpsampleMultiCoreShardedProgramFactory` collapses its legacy `WorkloadDescriptor` (secretly single-program — one structurally-identical `ProgramDescriptor` copied verbatim across every `MeshCoordinateRange`) onto the single-program concept, carrying its config lookup tensor in `op_owned_tensors`.

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (none of the three factories had one; matches the audit).
- Pybind entry points removed: **none** (`upsample_nanobind.cpp` binds only `mode`/`memory_config`/`compute_kernel_config`; no `create_descriptor` or `create_workload_descriptor` pybind existed).
- Header changes in `upsample_device_operation.hpp`: the three factory structs' methods renamed `create_descriptor`/`create_workload_descriptor` → `create_program_artifacts` (return type `ttnn::device_operation::ProgramArtifacts`); `UpsampleMultiCoreShardedProgramFactory` additionally drops its `tensor_coords` parameter (a `WorkloadDescriptorFactoryConcept`-specific parameter with no equivalent on the fixed `create_program_artifacts` signature — not a pybind-hook parameter, so not device-op-class-edit exception #3; simply the natural consequence of the concept change). Added `#include "ttnn/metal_v2_artifacts.hpp"`; removed the now-dead `#include <tt-metalium/workload_descriptor.hpp>` (its only referenced type, `WorkloadDescriptor`, no longer appears anywhere in the header). `UpsampleBilinearProgramFactory`'s declaration and the pre-existing `#include <tt-metalium/program_descriptors.hpp>` / `<tt-metalium/global_circular_buffer.hpp>` are untouched (the first still needed by Bilinear; the second is a pre-existing dead include unrelated to this port — flagged in the audit's Misc anomalies, left alone here per scope discipline).

### Open items
- None affecting the factory layer. No relaxation candidates — none of the three factories declare `dynamic_tensor_shape`/`match_padded_shape_only`, and none seem like they'd need one (fixed shapes, strict matching is correct).

## Handoff points

None. No capitulation; no boundary-rule assumption violations; no kernel-lib gaps encountered; no removed pybind surface.

## Successes

- **`borrowed_from` satisfies the `TensorParameter` binding requirement, no spurious `TensorBinding`s.** `UpsampleMultiCoreShardedProgramFactory`'s input, output, and op-owned config tensor are all delivered to their kernels purely through borrowed-memory DFBs (`SHARD_IN`/`SHARD_OUT`/`SHARD_CONFIG`, each `borrowed_from` its `TensorParameter`) — no kernel constructs a `TensorAccessor` for any of the three. Declaring the `TensorParameter`s with no matching `TensorBinding` compiled and validated cleanly, confirming the migration guide's statement that a borrowed DFB alone satisfies the parameter.
- **[Two-toucher DFB → assign 1P+1C](../shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split) fired exactly as documented** on the sharded factory's dual-instance work-split (`writer_upsample_multi_core_sharded.cpp` instantiated twice, Writer-config `is_reader=0` + Reader-config `is_reader=1`, over the same `cores_with_work`). All three of its DFBs (`SHARD_IN`, `SHARD_OUT`, `SHARD_CONFIG`) are role-free raw touches on both instances; assigning writer=PRODUCER/reader=CONSUMER on all three satisfied the SPSC validator with zero use of `allow_instance_multi_binding`.
- **Reusing the existing `untilize_metal2.cpp` fork (rung 1) worked exactly as the shared-kernel Caution describes.** The interleaved factory's tiled-path compute kernel pointed straight at `data_movement/untilize/device/kernels/compute/untilize_metal2.cpp` (created earlier by the `data_movement/fold` port) and adopted its binding vocabulary verbatim (`dfb::src`/`dfb::out`, `args::per_core_block_cnt`/`args::per_core_block_tile_cnt`) — no new fork, no edits to either copy, first-try compile and test pass on that path.

## Friction

### Gaps
- **WorkUnitSpecs with overlapping `target_nodes` are a hard runtime error (`program_spec.cpp:1704`), and no doc I read stated this as an explicit invariant — it only appears implicitly in one worked example.** My first draft of the interleaved factory's tiled path put `{INTLV_READER, INTLV_WRITER}` in one `WorkUnitSpec` over `all_cores`, and `{INTLV_COMPUTE_G1}` in a second `WorkUnitSpec` over `core_group_1` (a subset of `all_cores`) — reasoning that "a kernel may be included in multiple WorkUnitSpecs" (migration guide, WorkUnitSpec section) meant nodes could be too. It compiled, and 32 of the `test_upsample_nearest_interleaved[...Layout.TILE...]` parametrizations failed at runtime with `TT_FATAL: WorkUnitSpecs 'main' and 'compute_g1' overlap in target nodes`. The fix — matching [Anti-pattern: Demoting per-group CTA to RTA](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)'s own correct-port code block, which I had read but under-weighted — is to list READER and WRITER in **each** per-group `WorkUnitSpec` alongside that group's compute instance (`wu_g1 = {READER, WRITER, COMPUTE_G1}` over `core_group_1`; `wu_g2 = {READER, WRITER, COMPUTE_G2}` over `core_group_2`), so every `WorkUnitSpec`'s `target_nodes` are pairwise disjoint and a kernel's *effective* placement is the union across the WUs it's listed in. **Suggested doc fix:** state explicitly, as a load-bearing rule (not just implicit in an example), that `WorkUnitSpec::target_nodes` must be pairwise disjoint across a `ProgramSpec`'s `work_units`, and that a kernel needing to run across several disjoint groups must be **listed in each group's `WorkUnitSpec`** rather than placed in one `WorkUnitSpec` spanning their union while a narrower one covers a sub-range.

### Confusion
- **Stale environment memory (this session, not the docs) — arch mismatch.** A prior-session memory note claimed this host's target arch was `wormhole_b0`; `tt-smi` showed the actual hardware is Blackhole (`board_type: p100a`), and a second, more detailed memory file (correctly) already said `blackhole`. The one-line memory *index* summary was stale relative to the underlying memory file it pointed to. Caught by asking the user before burning a full build cycle on the wrong arch; fixed the index afterward. Not a recipe/doc issue — noted here only because it cost a rebuild-and-rediscover cycle before the port's own testing could start.
- **Backgrounding a device-test process incorrectly caused an external termination.** Combining the harness's own `run_in_background: true` with a manual `nohup ... &` inside the same command caused the harness to consider the shell "done" as soon as the `&` detached it, and something in that cleanup path later sent the actual (still-running) `pytest` process a `SIGTERM` mid-run (exit 143) rather than letting it finish. Per the on-device-test safety rule, treated this as a corruption risk and ran `tt-smi -r` before continuing, then re-ran the test using the harness's background mechanism directly (no manual `&`/`nohup`), which completed normally. Not a recipe issue; noted so a future porter in this same harness doesn't double-background a long device-test run.

## Open items for downstream

- **Shared kernel touch:** `UpsampleMultiCoreInterleavedProgramFactory`'s tiled path binds `data_movement/untilize/device/kernels/compute/untilize_metal2.cpp` — **reused an existing fork**, no new file created, no edits to it or to the legacy `untilize.cpp` beside it. Remaining unmigrated consumers (per the audit): `data_movement/untilize`'s own four factories and `data_movement/untilize_with_unpadding`'s three factories, still on the legacy API.
- **Per-op carry-over:** `UpsampleBilinearProgramFactory` is the natural next candidate once the Device 2.0 team clears the finding in `device/kernels/compute/bilinear.cpp` (see `METAL2_PREPORT_AUDIT.md`) — a re-audit of that factory alone should need little beyond confirming the fix didn't change CB shapes.
- **CB core-range widening is now automatic and slightly different from legacy for the sharded factory.** Legacy declared `in_cb`/`out_cb`'s `CBDescriptor::core_ranges` as `all_cores` (the full shard grid) even though only `cores_with_work` (a possibly-smaller subset, when the last shard is uneven) ever runs a kernel touching them — a pre-existing legacy discrepancy. Metal 2.0 derives DFB placement from kernel bindings, so the ported `SHARD_IN`/`SHARD_OUT` are placed on exactly `cores_with_work`, narrower than legacy's nominal (but functionally unused) `all_cores` declaration. This is strictly conservative (fewer L1 allocations on cores that never touched the buffer anyway) and not a behavior change; noting it since it's a case where the port's mechanical translation is *more* correct than the legacy code it replaced, not merely equivalent to it.
- Doc-evolution suggestion: see the WorkUnitSpec-overlap Gap entry above — candidate for a new patterns-catalog entry or a stronger callout in the recipe's Construct section.
