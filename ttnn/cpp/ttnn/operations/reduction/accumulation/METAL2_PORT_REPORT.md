# Port Report — accumulation (`AccumulationDeviceOperation`)

Friction record for the Metal 2.0 port of the accumulation device-op (cumsum / cumprod).
Captured during the port; cites recipe sections, file:line, and the resolved answer.

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`. `create_descriptor` → `create_program_spec` returning
`ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. No deviation from the audit's choice.

### Device-op-class edits
- Custom `compute_program_hash` deleted: `accumulation_device_operation.hpp:64` (decl),
  `accumulation_device_operation.cpp:105-119` (defn). Reverted to default reflection hash.
- Pybind entry points removed: none. No consumer nanobind file (cumsum/cumprod/ema) pybound
  `create_descriptor` on the factory; the rename is invisible to Python.

### Open items
- Relaxation candidates: none assessed (kept strict per recipe).
- `flip` RTA is per-dispatch-constant; a CRTA (or CTA) would dispatch more efficiently. Left as a
  per-node RTA to preserve legacy behavior exactly — flagged for owner, not changed in-port.

## Handoff points

- **None requiring an external team.** No out-of-op kernel touches; no `sem::`/`ta::` boundary
  violations; no kernel-lib gaps; no pybind surface removed. All three kernels are accumulation-owned
  and converted in-place.

## Successes

- **Work-split multiplicity (recipe "Plan the spec" / Anti-pattern: Demoting per-group CTA to RTA).**
  The recipe's insistence on one KernelSpec per legacy compute descriptor + one WorkUnitSpec per core
  group mapped cleanly onto the legacy `compute_desc_1` / `compute_desc_2` shape. Reader/writer as
  single KernelSpecs that are members of *both* WorkUnitSpecs worked exactly as
  `program_spec.hpp` (WorkUnitSpec) advertises ("A kernel may be included in multiple WorkUnitSpecs").
  See `accumulation_program_factory.cpp` `make_compute_spec` + `wu_g1`/`wu_g2`.
- **Self-loop DFB binding (patterns catalog: Self-loop DFB binding).** ACC bound as PRODUCER+CONSUMER
  with a shared accessor name "acc" on each compute KernelSpec; kernel constructs a single
  `DataflowBuffer dfb_acc_obj(dfb::acc)`. Compiled and ran first try — the shared-name form is on this
  branch. `accumulation_compute.cpp:29`.
- **Pass DFB handles directly to LLKs (patterns catalog).** `unary_op_init_common(dfb::src, dfb::dst)`,
  `reconfig_data_format(dfb::acc, dfb::acc)`, `copy_tile(dfb::src, 0, DST_IN)`, `pack_tile(DST_ACC,
  dfb::dst)` all took `dfb::name` via implicit conversion with no `.id` or wrapper. Zero friction.
- **RoleHint hardware config (recipe "Hardware-config shortcuts").** One-line
  `DataMovementHardwareConfig{.role = RoleHint::READER/WRITER}` replaced the legacy implicit
  reader/writer config. Clean.

## Friction

### Gaps

- **`unpack_to_dest_mode` for non-FP32 DFBs — legacy set it unconditionally; Metal 2.0 validator
  rejects it. This is the one real friction of the port and the most valuable finding.**
  Legacy (`accumulation_program_factory.cpp` pre-port, lines ~122-129) set
  `unpack_to_dst[ACC] = UnpackToDestFp32` *unconditionally* and `unpack_to_dst[SRC] = UnpackToDestFp32`
  whenever `input != Float16_b`. For **int32** cumsum/cumprod the ACC and SRC DFBs are *integer*
  format, so this set `UnpackToDestFp32` on a non-FP32 buffer. The legacy CB path never validated
  this and the LLK ignored it for integer formats. The Metal 2.0 spec validator
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:819`) **does** validate it and `TT_FATAL`s:
  *"unpack_to_dest_mode entry for DFB 'acc' specifies UnpackToDestFp32, but the DFB data format is
  not Float32."* Surfaced as a failing test: `test_cumsum.py::test_cumsum[int32-size=[1]-dim=0]`.
  - **Resolution (behavior-preserving):** gate each `UnpackToDestFp32` entry on the DFB's *actual*
    data format being `Float32` — exactly the cases where the legacy setting was meaningful. For
    bf16/fp32 inputs the resulting modes are bit-identical to legacy (verified: 27 bf16 + fp32 cases
    pass); for integer inputs the entry is now omitted (Default), which the LLK semantics confirm is
    equivalent. See the comment block at `accumulation_program_factory.cpp` ("The Metal 2.0 spec
    validator ... rejects UnpackToDestFp32 ...").
  - **Why this is a doc gap:** the recipe's "Hardware-config shortcuts" note says
    `unpack_to_dest_mode` is "not part of the [`to_compute_hardware_config`] helper — the factory must
    configure it separately." It does **not** warn that legacy ops frequently set `UnpackToDestFp32`
    on non-FP32 buffers and that the Metal 2.0 validator is stricter than the legacy CB path here.
    A porter following the recipe and faithfully copying the legacy unconditional set will hit a
    runtime `TT_FATAL` on the first integer-dtype test. **Suggested doc addition:** a one-line caution
    under "Hardware-config shortcuts" — "When copying a legacy `unpack_to_dest_mode` vector, drop any
    `UnpackToDestFp32` entry whose DFB is not Float32 format; the Metal 2.0 validator enforces the
    meaningfulness triple (CONSUMER endpoint + Float32 format + fp32_dest_acc_en) that the legacy CB
    path did not." This is arguably a latent legacy bug the migration usefully exposes.
- **`Table` has no `push_back` / no iterator-pair constructor (recipe "Construct paired spec +
  run-args", `utility/table.hpp`).** First build attempt failed: I used `.push_back` on the
  `UnpackToDestModes` Table and on `run_args.tensor_args` (also a Table), and constructed the
  `CompilerOptions::Defines` Table from a `std::map`'s `begin()/end()` iterator pair (the legacy
  `KernelDescriptor::Defines` was iterator-pair-constructible). `Table` (a custom map type, not a
  vector) supports only `{{k,v},...}` brace-init, `insert`, `emplace`, `operator[]`, and a *single*-
  arg range constructor (`Table(const R&)`). Resolved: brace-init / `insert` for the Tables, and
  `Defines{defines_kernel_args}` (range ctor) for the map. **Doc/example gap:** the `test_pack_relu.cpp`
  reference only ever brace-inits Tables inline, so the "how do I build a Table conditionally / from an
  existing std::map" idiom isn't demonstrated anywhere a porter would look. Worth a sentence in the
  Construct step or a comment in `table.hpp` pointing at `insert`/range-ctor for non-literal builds.

### Confusion

- **Brief says cumsum / cumprod / **ema** share `AccumulationDeviceOperation`; ema does not.**
  `ema/` is a fully separate device-op (`EmaDeviceOperation`) with its own `create_descriptor`,
  program factory, and kernels (`ema/device/ema_program_factory.cpp`). The accumulation port affects
  cumsum + cumprod only. ema's `create_descriptor` is *not* in scope and was left untouched (its 3
  tests still pass). Minor, but a porter could waste time hunting for ema's use of the shared op.
- **No reference C++ port exists (as the invoker warned).** The layernorm / pool `METAL2_PORT_*.md`
  files present in the tree are doc-only artifacts — no `.cpp` defines `create_program_spec` anywhere
  in `ttnn/`/`tt_metal/` (confirmed by grep). `test_pack_relu.cpp` was the only concrete construction
  example, and it is a hand-written host test, not a TTNN factory: it shows `ProgramSpec` /
  `KernelSpec` / `ProgramRunArgs` brace-init and `MakeProgramFromSpec`/`SetProgramRunArgs`, but is
  silent on the adapter path (`create_program_spec` signature, `ProgramArtifacts` return, MeshTensor
  identity matching). The TTNN-integration doc + `mesh_device_operation_adapter.hpp` filled that gap
  adequately, but a single end-to-end TTNN factory example would have saved ~20 min of reading the
  adapter to confirm the tensor-arg `std::cref(input_tensor.mesh_tensor())` reference shape.

## Open items for downstream

- **Cross-op kernel touches:** none. All three kernels are accumulation-owned.
- **`flip` could be a CRTA, not a per-node RTA.** `flip` is constant across all nodes per dispatch
  (`static_cast<uint32_t>(operation_attributes.flip)`), but the legacy code emitted it per-core and the
  port preserves that. A `common_runtime_arg` (CRTA) would dispatch more efficiently. Left as a named
  RTA to keep behavior identical; flagged for the op owner as a perf micro-opt, not a port change.
- **`input_tile_offset` / `tiles_per_row` are also per-node-constant RTAs** in the legacy reader/writer
  schema — same CRTA candidate as `flip`. Bundled into the same owner note.
- **Relaxation candidates:** none assessed; kept strict per the recipe and TTNN-integration doc.
- **`FIRST_TILE` / `WORKING_REG` constants in `kernels/accumulation_common.hpp` are unused** (were
  unused pre-port too). Left in place to stay scope-tight; a future cleanup could drop them.
