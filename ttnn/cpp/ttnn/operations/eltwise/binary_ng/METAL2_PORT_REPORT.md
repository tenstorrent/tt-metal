# binary_ng — Metal 2.0 Port Report

**STATUS: PARTIAL** — the simplest path of the op is ported to a new Metal 2.0
`ProgramSpecFactory`; every other path remains on the legacy `ProgramFactory::create_descriptor`.
The op builds and runs as a mixed-concept device operation (`program_factory_t` holds one
legacy `ProgramDescriptorFactoryConcept` factory and one Metal 2.0 `ProgramSpecFactoryConcept`
factory), dispatched per-call by a new `select_program_factory`.

## Op / factory

- Op dir: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/`
- Device op: `BinaryNgDeviceOperation` (`device/binary_ng_device_operation.hpp`)
- Legacy factory: `ProgramFactory::create_descriptor` (`device/binary_ng_program_factory.cpp:384`),
  a single `ProgramDescriptorFactoryConcept` factory that runtime-selects its kernel *source
  files* across ~6 multiplying axes (layout × broadcast type × compute flavor × b-present ×
  LLK-vs-software-bcast × sharded), binding ~24 op-local kernel sources.

## Why PARTIAL (not full), and why this shape is buildable

A Metal 2.0 factory's conversion is **atomic within the factory body**: `create_program_spec`
emits Metal 2.0 bindings (`dfb::`/`ta::`/`args::`), so *every* kernel source it can select at
runtime must read those bindings. The legacy `create_descriptor` selects among ~24 sources across
multiplying axes inside one body — converting it in place would require converting all 24 kernels
(each with `#ifdef`-driven sub-variants) plus the per-axis host logic (3 RTA layouts, borrowed-
memory sharded CBs, software/LLK bcast forks, per-variant dummy-arg padding) as one unit. Not
faithfully completable in one pass.

Instead, the framework supports a `program_factory_t` variant whose alternatives satisfy
*different* concepts, dispatched per-call by `select_program_factory` (operation_concepts.hpp:90,
129–145; recipe "atomic unit" note). So this port **adds a second, narrow Metal 2.0 factory** for
one path and leaves the legacy factory untouched for all others. The three kernels the narrow
factory binds are **forked** to `*_m2.cpp` copies (the legacy factory still binds the originals),
so the originals are not disturbed.

## Path ported

`SubtileBroadcastType::NONE × tile layout × FPU (not SFPU/where/quant) × tensor-b-present ×
interleaved (not sharded) × no activations × no typecast × plain ADD/SUB/MUL.`

`select_program_factory` (`device/binary_ng_device_operation.cpp`) routes only that exact case to
`ProgramSpecFactory`; the gate is deliberately conservative — anything the narrow factory does not
model falls through to the legacy factory.

### Files added

- `device/binary_ng_program_factory_m2.cpp` — `ProgramSpecFactory::create_program_spec`.
- `device/kernels_ng/dataflow/reader_interleaved_no_bcast_m2.cpp` — fork+port of the reader.
- `device/kernels_ng/dataflow/writer_interleaved_no_bcast_m2.cpp` — fork+port of the writer.
- `device/kernels_ng/compute/eltwise_binary_no_bcast_m2.cpp` — fork+port of the compute kernel.

### Files changed

- `device/binary_ng_device_operation.hpp` — added `ProgramSpecFactory` to `program_factory_t`,
  declared `select_program_factory`, `#include "ttnn/metal2_artifacts.hpp"`.
- `device/binary_ng_device_operation.cpp` — added `select_program_factory`.
- `sources.cmake` — added the new factory `.cpp`.

### Kernel conversion (logic unchanged; access mechanism only)

- Reader/writer: CB ids → `dfb::src`/`dfb::src_b`/`dfb::dst`; `TensorAccessorArgs<N>()` +
  address RTA → `TensorAccessor(ta::...)` (address auto-injected, so the `src_addr`/`dst_addr`
  RTA slots drop); positional args → `get_arg(args::...)`; the trailing positional `has_sharding`
  CTA → named CTA. `#if SRC_SHARDED`/`DST_SHARDED` branches preserved verbatim (compiled out on
  this interleaved path, kept for faithfulness).
- Compute: CB ids → `dfb::pre_lhs`/`dfb::pre_rhs`/`dfb::out`; the activation-intermediate CBs
  (c_3/c_4) become conditionally-bound `dfb::post_lhs`/`dfb::post_rhs`, and the legacy
  `HAS_ACTIVATIONS(LHS) ? c_3 : c_0` C++ ternary is rewritten as a preprocessor `#if` so the
  conditionally-bound DFB name never enters name lookup when its activation is absent (recipe
  kernel-side rule 6). `dfb::` handles flow into the `PREPROCESS`/`BINARY_OP`/`pack_tile`/`cb_*`
  helpers via the accessor's implicit `uint32_t` conversion (no `.id` extraction). The op-local
  helper headers `eltwise_utils*.hpp` are left untouched.

## TensorParameter relaxation (mirrored, not invented)

The legacy factory declares `ArgConfig::RuntimeTensorShape` on its I/O `TensorAccessorArgs`
(eltwise family). Mirrored here as `TensorParameter::advanced_options.dynamic_tensor_shape = true`
on a/b/c — faithful to the relaxation the legacy op already had. Interleaved ⇒ TensorAccessor
config is unchanged by it (safe).

## Custom compute_program_hash — KEPT (deliberate deviation from the recipe default)

The recipe normally deletes a custom `compute_program_hash` on a Metal 2.0 port. Here it is
**kept** (`device/binary_ng_device_operation.cpp:487`): the op is mixed-concept — the legacy
`ProgramFactory` still serves every non-narrow path and depends on the custom hash (it folds in
`shard_volumes`). Deleting it would change the legacy factory's caching behavior. The hash is
shared across both factories. (Per the task instruction: keep the custom hash.) When the FULL
port lands and the legacy factory is retired, the custom hash should be revisited per the recipe.

## Remaining entry points (for the full port)

The legacy factory still binds all of these; the full port must convert the factory body + all of
them as one atomic unit (or peel further narrow paths off into the Metal 2.0 factory, each with
its own forked kernels):

- Readers (`device/kernels_ng/dataflow/`): `reader_interleaved_row_bcast.cpp`,
  `reader_interleaved_col_bcast.cpp`, `reader_interleaved_row_col_mixed_bcast.cpp`,
  `reader_interleaved_scalar_bcast.cpp`, and the `*_rm_*` row-major family
  (`reader_interleaved_rm_no_bcast.cpp`, `_rm_row_bcast`, `_rm_col_bcast`,
  `_rm_row_col_mixed_bcast`, `_rm_scalar_bcast`, `_rm_scalar_op`).
- Writers: `device/kernels_ng/dataflow/writer_interleaved_rm_no_bcast.cpp`; legacy scalar writer
  `device/kernels/dataflow/writer_interleaved_scalar.cpp` (`WriterScalar`).
- Compute (LLK-bcast tree, `device/kernels_ng/compute/`): `eltwise_binary_row_bcast.cpp`,
  `eltwise_binary_col_bcast.cpp`, `eltwise_binary_scalar_bcast.cpp`,
  `eltwise_binary_row_col_bcast.cpp` and their `*_sfpu_*` + `eltwise_where_sfpu_*` siblings.
- Compute (software-bcast tree, `device/kernels/compute/`): `eltwise_binary.cpp`,
  `eltwise_binary_scalar.cpp`, `eltwise_binary_sfpu*.cpp`, `eltwise_where_*.cpp`, plus the FPU
  `eltwise_binary_no_bcast.cpp` for the scalar/SFPU sub-cases of the no-bcast axis.
- Axes not yet modeled by the narrow factory: row-major layout; SCALAR/ROW/COL/mixed broadcasts;
  SFPU and where-op compute flavors; scalar-b (no tensor b); sharded (borrowed-memory CBs +
  shard work-split); activations / typecast / quant; ISCLOSE (5-arg compute RTA); the
  software-vs-LLK bcast fork and its arch/dtype hang-workaround fallbacks.

## Handoff points / open items

- **Custom hash kept by design** (above) — revisit when the full port retires the legacy factory.
- **Untested at runtime** — no build dir in this worktree; this is a static/faithful conversion.
  Per recipe, the narrow path's pytests (`tests/ttnn/.../eltwise`, the binary-ng add/sub/mul tile
  interleaved bf16/fp32 cases) should be run once built; exclude not-yet-converted paths are
  unaffected since they stay on the legacy factory.
- **No pybind hook removed** — the legacy `create_descriptor` is still live (the legacy factory
  remains), so no pybind surface changed.

## Anti-pattern self-audit

- No `tensor.buffer()->address()` in the new factory (addresses via `TensorBinding`). ✓
- No magic CB indices in CTAs (only `has_sharding`/`num_tiles_per_cycle` named CTAs). ✓
- No `TensorAccessorArgs<N>()` in the `_m2` kernels (replaced by `TensorAccessor(ta::...)`). ✓
- Conditional compute DFBs (`post_lhs`/`post_rhs`) follow the `#ifdef`-gated-alias pattern. ✓
- All CTAs named; compute work-split multiplicity preserved (one KernelSpec per core group). ✓
