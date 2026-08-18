# Port Report — `data_movement/copy` (`CopyDeviceOperation`)

## Outcome

**PORTED** — 2 of 3 factories ported to Metal 2.0 (`ProgramSpecFactoryConcept`):
- **`DefaultRowMajor`** — ported, tests pass.
- **`DefaultTilized`** — ported, tests pass.
- **`SameMemoryConfig`** — **CAPITULATED** (left on the legacy `descriptor` concept). It has a cross-op
  consumer the audit did not catch: the peer op `data_movement/move` reuses its `create_descriptor`
  and depends on both the `ProgramDescriptor` return type and the positional runtime-arg layout.
  Porting it would break an op outside this port's writeable surface. See Handoff points.

The op builds and runs with the two ported factories on Metal 2.0 and `SameMemoryConfig` on legacy;
the framework dispatches per factory at runtime.

## Provenance

- **Recipe docs (this port):** `c16f21b8cb6 2026-08-18 docs(metal_2.0): unpack_modes -- the trigger is the buffer format, not the dtypes`
- **Audit docs (inherited):** `c16f21b8cb6 2026-08-18 docs(metal_2.0): unpack_modes -- the trigger is the buffer format, not the dtypes`

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept` for `DefaultRowMajor` and `DefaultTilized` (as the audit chose).
`SameMemoryConfig` remains `ProgramDescriptorFactoryConcept` (capitulated — see Handoff points).
Mixed-concept variant is valid; the framework dispatches per factory. Both ported factories are the
base concept (no `override_runtime_arguments`; framework refreshes tensor bindings on cache hit).

### Device-op-class edits
- Pybind entry points removed: none (nanobind binds only `copy`/`assign` free functions).
- Custom `compute_program_hash`: none — nothing to preserve.
- Device-op `.hpp`: `create_descriptor` → `create_program_artifacts` for the two ported factories;
  `SameMemoryConfig::create_descriptor` left intact (a comment records why). Added
  `#include "ttnn/metal_v2_artifacts.hpp"`.

### Open items
- **Relaxation candidates:** none applied (audit said none; kept strict tensor matching).
- `SameMemoryConfig` is portable in principle once `data_movement/move` is addressed (see Handoff).

### Verification (silent-perf settings — before/after diff)
- **`hw_config`:** DefaultRowMajor + DefaultTilized reader/writer are legacy `ReaderConfigDescriptor{}` /
  `WriterConfigDescriptor{}` (default triples) → ported to `create_reader_datamovement_config(arch)` /
  `create_writer_datamovement_config(arch)` (identical Gen1 triples). DefaultTilized compute is legacy
  `ComputeConfigDescriptor{}` (all defaults) → `ComputeGen1Config{}` (defaults coincide). No custom DM
  triple, no non-default compute knob, no `unpack_modes` required (`enable_32_bit_dest` stays false, so
  the Float32-consumer rule does not fire). `bfp_pack_precision_mode` left default.
- **`opt_level`:** reader/writer are DM (legacy default O2 == Metal 2.0 default O2 — left unset).
  DefaultTilized compute set explicitly to `O3` (legacy compute defaults to O3; Metal 2.0
  `CompilerOptions` defaults to O2). No DM kernel set an explicit level to carry over.
- **Anti-pattern self-audit:** clean over the shipped files (2 ported factories + 2 ported redistribute
  kernels + 1 created compute fork): 0 buffer-address/`Buffer*`/`emplace_runtime_args`, 0 CB
  types/magic indices, 0 `TensorAccessorArgs`, 0 `cb`-shaped names (variables renamed, comments
  aligned), 0 `.id` extraction, 0 varargs, 0 positional CTAs, 0 `.md` citations from code. TT_FATAL
  count in the op dir unchanged (13, all in the untouched device-op class).

## Handoff points

- **`SameMemoryConfig` factory capitulation — cross-op consumer missed by the audit.**
  - Op / factory: `ttnn/cpp/ttnn/operations/data_movement/copy` → `SameMemoryConfig`.
  - The blocker: `ttnn/cpp/ttnn/operations/data_movement/move/device/move_program_factory.cpp:27`
    calls `CopyDeviceOperation::SameMemoryConfig::create_descriptor(...)` and returns its
    `ProgramDescriptor` from `MoveProgramFactory::create_descriptor`. Its
    `override_runtime_arguments` (same file, ~`:36-56`) then patches the buffer addresses at
    positional runtime-arg slot 0 of kernels 0 (reader) and 1 (writer) via
    `GetRuntimeArgs(program, 0/1)[...][0]`.
  - Why mechanical conversion fails: porting `SameMemoryConfig` replaces `create_descriptor`
    (returns `ProgramDescriptor`) with `create_program_artifacts` (returns `ProgramArtifacts`),
    which `MoveProgramFactory` cannot consume from its own `ProgramDescriptor`-based factory; and the
    buffer address stops being a positional slot-0 runtime arg (it becomes a `TensorBinding`), so
    `move`'s address-patching override would silently target the wrong slot. A factory cannot expose
    *both* `create_descriptor` and `create_program_artifacts` (that trips the `AllFactoriesValid`
    dual-concept `static_assert`). `move` is a separate op with its own tests and override, outside
    this port's writeable surface.
  - What the fix would look like: port `data_movement/move` alongside `copy` (a bundled multi-op
    change), or give `move` its own program-construction path so it no longer reuses copy's factory,
    then port `SameMemoryConfig`. Both are out of scope for this single-op port.
  - Audit gap: the audit's TTNN factory analysis recorded "primary cross-check against the code is
    clean" but did not scan for *external* consumers of the factory entry points. A future audit step
    should `grep -rn "<Op>DeviceOperation::<Factory>::create_descriptor"` across the tree.
  - Note: `ttnn/cpp/ttnn/operations/experimental/quasar/move/...` also calls this — ignored per the
    quasar-tree rule; not a consideration for scope.

## Successes

- **Self-loop DFB pattern (DefaultRowMajor c_0).** The [Sync-free / single-ended → self-loop]
  pattern fit exactly: the redistribute reader uses c_0 as a private L1 scratchpad
  (`reserve_back`/`get_write_ptr`/`push_back`/`wait_front`/`pop_front`, all in the one kernel).
  Bound the reader as both PRODUCER and CONSUMER of `input_pages` with one accessor name — clean,
  matches the brief's disposition.
- **Existing `_metal2` forks reused (DefaultTilized reader/writer).** The eltwise/unary interleaved
  reader/writer forks already existed with a clear binding vocabulary (`dfb::in`/`dfb::out`,
  `tensor::src`/`tensor::dst`, args `num_pages`/`start_id`); binding them was drop-in, exactly as the
  shared-kernel Caution's rung 1 describes.
- **Preserved-multiplicity guidance held (DefaultTilized).** The audit/brief correctly flagged the
  compute path as a single KernelSpec (num_tiles is an RTA over all_cores), not a per-group CTA — no
  spurious multiplicity introduced.

## Friction

### Gaps
- **`get_tile_size()` member getter is compute-only, but the tilized reader/writer are DM kernels.**
  The CB→DFB whitelist §A / rule 7 prescribes the member getter `dfb.get_tile_size()` for a
  non-`constexpr` legacy `get_tile_size(cb_id)`. But `DataflowBuffer::get_tile_size()` is gated on
  `DFB_DESCRIPTORS_DEFINED` (`chlkc_descriptors.h`, present only in compute builds), so on a
  data-movement kernel it does not compile. (This surfaced while planning `SameMemoryConfig`'s
  tilized own kernels, which use `get_tile_size(cb_id)`.) The correct DM form is the free-function
  shim with the binding token — `get_tile_size(dfb::in0)` — which is byte-identical to legacy but
  isn't the constexpr-only case the whitelist carves out for the free-function form. **Doc
  suggestion:** the whitelist should note that the DM-kernel case (no `DFB_DESCRIPTORS_DEFINED`)
  keeps the free-function-with-token form for `get_tile_size` regardless of `const`/`constexpr`.
  *(This friction was found but the SameMemoryConfig factory it applied to ultimately capitulated;
  it still holds for any DM kernel that reads tile size.)*

### Confusion
- **The audit's "primary cross-check clean" gave false confidence on scope.** The `move` cross-op
  dependency (see Handoff) is exactly the sort of thing a clean audit implies is absent. Discovering
  it only at build time (a `no member named 'create_descriptor'` error from a peer op) cost a build
  cycle. Cheap to have caught in the audit with a tree-wide grep for external factory-method callers.

## Open items for downstream

### Shared kernel touches
- **Created fork:** `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy_metal2.cpp`
  (bound by `DefaultTilized`). Pointer comment added to the legacy original
  (`.../sharded/device/kernels/compute/eltwise_copy.cpp`). Remaining unmigrated consumers of the
  legacy original (sunset list): `data_movement/sharded/interleaved_to_sharded`,
  `data_movement/sharded_partial/interleaved_to_sharded_partial`. The legacy copy is retired when the
  last of them migrates.
- **Reused existing forks (no new file):** `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp`
  and `writer_unary_interleaved_start_id_metal2.cpp` (bound by `DefaultTilized`). Broadly shared;
  binding vocabulary inherited unchanged.
- **Forks NOT created (SameMemoryConfig capitulated):** the three shared-pool kernels the brief listed
  for `SameMemoryConfig` — `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp`,
  `.../writer_unary_stick_layout_interleaved_start_id.cpp`, `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` —
  still have **no** `_metal2` fork. The next port to reach `SameMemoryConfig` (after `move` is
  addressed) creates them.

### Other
- **Dead CTA (ops-team prune candidate, not porter work):** `redistribute_pages_row_major_reader.cpp`
  declares `num_output_pages_in_row` but never uses it; the factory still emits it. Carried forward as
  a named CTA unchanged (zero behavior change).
- **RTA→CRTA candidates (not converted — would change dispatch semantics):** `SameMemoryConfig`'s
  row-major `stick_size` / `num_shards` are the same on every node (really CRTAs). Not relevant now
  (that factory stayed legacy), but a future porter of it should note them for a separate cleanup pass.
