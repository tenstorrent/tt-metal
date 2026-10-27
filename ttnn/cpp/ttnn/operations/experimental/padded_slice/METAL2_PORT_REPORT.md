# Port Report — `experimental/padded_slice`

## Outcome

**PORTED** — both factories. `PaddedSliceRMProgramFactory` on `CustomProgramSpecFactoryConcept`,
`PaddedSliceTileProgramFactory` on `ProgramSpecFactoryConcept`; every kernel entry point either factory
can select converted with it. Tests pass; outputs are bit-exact against the legacy factories.

Verification (Blackhole p150b, Watcher on, Metal 2.0 legality checks forced in the working tree and
proven live by the `METAL2_CHECKS_FORCED` markers in every log):

| run | legacy (pre-port tree) | port |
|---|---|---|
| `tests/ttnn/unit_tests/operations/data_movement/test_padded_slice.py -k 'not quasar'` | 13 passed | 13 passed |
| 53-case capture (every unit-test shape plus the first two slices of 18 nightly conv configurations across RM/Tile × height/block/width sharding, bf16/f32/bf8_b/uint8, pad 8/16/32), each dispatched twice (cache miss, then cache hit on freshly allocated buffers) | reference | 106/106 phases bit-exact, identical program-cache deltas |
| `tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py -k for_conv2d` (height / block / width sharded × RM / Tile, plus the dtype × pad coverage) | — | 163 passed, 4 skipped (the test's own grid-size skip for `[2, 1024, 1024, 256]` block sharding), 0 failed |

## Provenance

- **Recipe docs (this port):** `aab653ac9b9 2026-10-27 docs(metal_2.0): import the op-porting recipe docs for the padded_slice audit`
  (verbatim import of `akertesz/op-porting-recipe` @ `4bd4bf42bfe`; the import commit is dropped from the
  PR branch at the invoker's request, so the recipe revision is pinned by the recipe-branch hash).
- **Audit docs (inherited):** the same import, `aab653ac9b9`, per `METAL2_PORT_BRIEF.md`.
- **Code ported from:** `origin/main` @ `65b565e8f8b` (carries pre-port fix #62499 as `0226df45c4a`).

## TTNN ProgramFactory

### Concept realized
- RM: `CustomProgramSpecFactoryConcept`, as the audit chose. `override_runtime_arguments` now returns a
  `ProgramRunArgs` whose `tensor_args` carry **both** io `TensorParameter`s (`input`, `output`) — the two
  things the ported-from override refreshed (reader base-address RTA and the output CB's backing
  buffer). Nothing else is refreshed, as before. Its three guards moved with it.
- Tile: `ProgramSpecFactoryConcept`, as the audit chose.

### Device-op-class edits
- Pybind entry points removed: none (`padded_slice_nanobind.cpp` never exposed `create_descriptor`).
- Custom `compute_program_hash`: none.
- `padded_slice_device_operation.{hpp,cpp}`, `padded_slice.cpp`, `padded_slice_nanobind.cpp`,
  `padded_slice_utils.{hpp,cpp}`, `slice_cb_descriptor.hpp`: untouched.

### Open items
- Relaxation candidates: none noticed; both factories key the cache on the full tensor specs.
- No friction with the concept fit.

## Handoff points

none.

## Successes

- **Two-toucher census → 1P+1C** (patterns catalog): RM `out` has a locked producer (reader) and either a
  locked consumer (plain-writer fork) or a role-free toucher (pad writer); the census gave 1P+1C in both
  configs without a multi-binding flag (`padded_slice_rm_program_factory.cpp`, writer `KernelSpec`).
- **Conditional / optional resource bindings — "promote a CTA gate to a define"**: the RM reader's
  `if constexpr (is_non_aligned)` would have name-looked-up `dfb::non_aligned` in the discarded branch;
  the entry's prescription (`#ifdef NON_ALIGNED` on the host define, conditional binding) was exactly the
  shape needed (`padded_slice_reader_rm_interleaved_start_id.cpp`).
- **Caution: Porting a shared kernel, rung 1**: the existing `writer_unary_sharded_metal2.cpp` fork fit
  as-is; its vocabulary (`dfb::out`, `args::num_units`) became the plain-writer `KernelSpec`'s interface.
- **Compiler options rule 2**: the Tile compute spec gets an explicit `O3`; the legacy
  `ComputeConfigDescriptor` set nothing, so this would have silently dropped to `O2`.
- **`constexpr` metadata rule** (whitelist §A): Tile writer keeps `get_tile_size(dfb::out)` for its
  `constexpr tile_size`; Tile reader's runtime `const` moved to `dfb_in.get_tile_size()`.
- **Noc overload check before porting**: the DFB-managed `async_read<TXN_ID>(src, DataflowBuffer&, …)`
  overload (no size, no trid) is `#ifdef ARCH_QUASAR` only, so the RM reader's explicit-size + trid calls
  keep resolving on Gen1 with a `DataflowBuffer` destination — confirmed from `noc.h` before the swap.

## Friction

### Gaps
- **Shared compute kernel whose consumers disagree on topology.** halo feeds `pack_untilize.cpp` two
  distinct output CBs (`untilize_with_halo_program_factory.cpp:185-197`); padded_slice feeds it one CB
  under both names. The brief's "one DFB, two kernel-side names" is true only for padded_slice. A fork
  with a single `dfb::out` would be unusable by halo, and a later porter may not edit a fork with a
  consumer, so the fork gates the second output on a `SEPARATE_OUT1` define (Same-FIFO aliasing,
  path-dependent variant). The shared-kernel Caution could say what to do when the first fork's consumer
  is the degenerate case of the kernel's real shape.
- **Mutable vararg memory** (also audit note 3): the three DM kernels advance a per-dimension index counter that the host
  seeds; the `Scratchpad` seeded from the vararg block (slice's shape) is still not a catalog entry.
- **Debug prints of CB identity.** The `#ifdef DEBUG` blocks printed CB indices (`cb_id_non_aligned`,
  `cb_temp_pad`, `cb_untilized_id`, `cb_out_id`) and the dropped RTAs (`src_addr`, `padded_stick_size`).
  No whitelist row covers a debug print of a DFB's identity; those fields were removed from the prints,
  the rest of each block kept. Worth a sentence in rule 8 / the whitelist.
- **Superfluous legacy RTAs.** The legacy RM factory sent the pad writer's four RTAs to the plain writer
  too (`padded_slice_rm_program_factory.cpp:251-252, :394` pre-port); the fork reads one. Metal 2.0's
  schema rejects superfluous args, so the plain-writer path now passes `num_units` only — behaviour
  preserving, but the recipe's Dropped Plumbing list has no row for "dead RTA on one selected source".

### Confusion
- `experimental_device_api.hpp` (pool) both aliases `CircularBuffer` and supplies `experimental::local_addr`;
  the brief says keep the include, the whitelist says the `CircularBuffer` include drops with rule 1. Kept
  the include (needed), so `api/dataflow/circular_buffer.h` still arrives transitively in four kernels
  although no `CircularBuffer` object survives.
- The recipe's `UpdateTensorArgs` note: the grep found five `skip_validation` sites in
  `program_run_args.cpp` (incl. `UpdateTensorArgs` and a `KernelRunArgs` merge helper) and four in
  `program_spec.cpp`; all nine forced, two markers (`BuildProgramFromSpec`, `SetProgramRunArgs`).

## Open items for downstream

- **Shared kernel touches**
  - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` —
    **reused the existing fork** `writer_unary_sharded_metal2.cpp` (no new file). Remaining legacy
    consumers on `main`: `data_movement/sharded/interleaved_to_sharded`,
    `data_movement/sharded_partial/interleaved_to_sharded_partial`, `data_movement/tilize` (2 factories),
    `data_movement/tilize_with_val_padding`, `data_movement/transpose` (wh sharded),
    `data_movement/untilize` (2 factories), `experimental/transformer/nlp_kv_cache_load_slice`,
    `reduction/generic` (2 factories). Sunset tracked in #52228.
  - `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp` — **created the
    fork** `pack_untilize_metal2.cpp` beside it; pointer comment added to the original. Vocabulary:
    `dfb::src` CONSUMER, `dfb::out0` PRODUCER, optional `dfb::out1` PRODUCER under `SEPARATE_OUT1`,
    CTAs `tiles_per_row` / `block_size`, RTA `total_blocks`. Remaining legacy consumer:
    `sliding_window/halo/device/untilize_with_halo_program_factory.cpp` (define `SEPARATE_OUT1` and bind
    `out1` when it ports).
  - Lent host headers left in place for `experimental/slice_write` (3 factories):
    `device/slice_cb_descriptor.hpp` (no longer included by this op; the "no `CBDescriptor` in the op
    directory" sweep hits it — 5 lines, all in that lent header) and `device/padded_slice_utils.hpp`.
- **Retained varargs (per recipe rule 4):** three `num_dims`-long blocks on each DM kernel — RM reader and
  Tile reader: `num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim` seed; Tile writer: `output_coord`
  seed, `output_start_in_input`, `output_end`. All are read in `j < num_dims` loops or by computed index.
- **Token-form `get_tile_size(dfb::out)`** (Gen1-only conversion, per whitelist §A):
  `writer_unary_sharded_padded_tiled.cpp`, `constexpr tile_size`. Also the compute fork's
  `constexpr uint32_t src_dfb = dfb::src` etc. for the kernel-lib NTTPs.
- **DM self-loops (Quasar-uplift debt):** RM `non_aligned` on the reader, RM `temp_pad` on the pad writer,
  Tile `out` and `padding` on the writer.
- **Legitimate `TT_FATAL` loss:** `padded_slice_rm_program_factory.cpp:403-407` (pre-port) asserted the
  reader's positional kernel index for the override's `GetRuntimeArgs`; kernels are addressed by name
  now, so the subject is gone (RM factory count 11 → 10; every other guard is in place).
- **Findings routed, not fixed** (all pre-existing, unchanged by the port):
  - `padded_slice_tile_program_factory.cpp` — two `get_tiled_start_offset` calls whose results are
    discarded, and several `[[maybe_unused]]` values computed and never used (also in the audit).
  - `writer_unary_sharded_padded_tiled.cpp` includes the internal `dataflow_api_addrgen.h` and uses
    nothing from it; defines a local `round_down`.
  - `pack_untilize{,_metal2}.cpp` include `untilize_helpers.hpp` twice through two paths (copied as is).
  - RM factory: `slice_row_size_bytes` means the per-core row in the factory and the whole sliced row in
    the run-args builder.
- **Test coverage note:** no C++ gtest exercises this op; the Python unit file covers the cache-hit path
  of each factory and the RM alignment/padding paths, the nightly conv sweep covers the sharding × layout ×
  dtype grid. `test_padded_slice.py`'s two `quasar` cases bind the out-of-bounds quasar copy and were
  deselected.
- **Sibling carry-over:** `experimental/slice_write` (three factories) shares this op's host helpers and
  the same counter-in-varargs shape; the readiness sheet rows 294–296 for it were still stale at audit
  time.
