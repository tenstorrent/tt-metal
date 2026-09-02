# Metal 2.0 Port Report — `matmul` / `MatmulMultiCoreProgramFactory`

## Outcome

**`PORTED`** — `MatmulMultiCoreProgramFactory` converted from `ProgramDescriptorFactoryConcept` to
Metal 2.0 `ProgramSpecFactoryConcept`. The op's other seven factories are untouched and stay on
their legacy concepts; each needs its own audit and port pass (see `METAL2_PORTING_PLAN.md`).

**Verification is build-only, by the invoker's decision.** This workspace has no attached device
(`/dev/tenstorrent` absent, `lspci` reports no Tenstorrent card, `tt-smi -ls` says "No Tenstorrent
devices detected", and `ird list` is blocked inside the container), so the confirmed test set was
not run. What *was* verified, and the exact commands to finish it, are in
[Verification status](#verification-status) below. **The three converted kernel sources have never
been JIT-compiled** — kernels compile at runtime, from disk, on a device — so they are the least
verified part of this port.

## Provenance

- **Recipe docs (this port):** `git log -1 --format='%h %cs %s' -- docs/…/metal_2.0/` prints
  **nothing** in this working tree — the recipe docs are not merged to `main` and were not present
  in the checkout. They were read from `origin/akertesz/op-porting-recipe` at
  `b419a49b934 2026-09-01 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`.
- **Audit docs (inherited):** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`
- The port therefore ran against a doc revision one commit newer than the audit's. The single
  intervening commit concerns conditional bindings for tensors and semaphores, which this port does
  not use, so nothing the audit decided is affected.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (base) — as the audit chose. The factory implements exactly one method,
`create_program_artifacts`, returning `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`.
**No `override_runtime_arguments` was added**, so the framework refreshes the three tensor bindings
on every cache hit and nothing else, which matches the ported-from factory (it had no override, so
the descriptor adapter's own address inference was doing the same job). `op_owned_tensors` is left
default-empty — the factory allocates no tensors of its own.

### Device-op-class edits

- **Pybind entry point removed:** `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp` — the whole
  `nb::class_<ttnn::prim::MatmulMultiCoreProgramFactory>` block (was lines 1260-1274), whose only
  member was `create_descriptor`. Sanctioned exception 1; details under
  [Handoff points](#handoff-points).
- **Pybind-hook-only parameter dropped:** `create_descriptor`'s fourth argument
  `const std::optional<CoreRangeSet>& core_range_set`, which the factory body ignored (it was
  spelled `/*core_range_set*/`). Sanctioned exception 2. There was no production default to inline,
  because nothing read it.
- **Exception 3 not applicable** — the op has a proper `program_factory_t` variant, so this was a
  method swap inside the existing struct.
- **Custom `compute_program_hash`: none, and nothing was touched.** The op uses the default
  reflection hash. The deliberately-renamed `compute_descriptor_program_hash` helper
  (`device/matmul_device_operation.hpp:50`) and the pybind that exposes it under the name
  `compute_program_hash` (`matmul_nanobind.cpp:1233-1237`, inside the
  `nb::class_<MatmulDeviceOperation>` block) are byte-identical. That whole block was left alone.

### Open items

- **Relaxation candidates: none applied, none obviously available.** No kernel uses
  `ArgConfig::Runtime*`, and the readers/writer bake in shapes through RTAs rather than reading the
  tensor spec, so the strict `TensorSpec` match costs nothing here.
- **The concept fit was clean.** The one place the base concept's cache-hit contract mattered is
  that the three io tensors are the only mutable surface, which is exactly what the framework
  refreshes for free.

---

## Handoff points

### 1. Removed pybind surface — `MatmulMultiCoreProgramFactory.create_descriptor`

*Tagged: API surface — removed entry point. Owner: TTNN + the experimental descriptor framework.*

`matmul_nanobind.cpp` no longer exposes `MatmulMultiCoreProgramFactory`. The block bound one static
method, `create_descriptor(operation_attributes, tensor_args, tensor_return_value, core_range_set)`,
which built a `ProgramDescriptor` for the Python descriptor/fusion framework to run through
`ttnn.generic_op`.

**This breaks a live Python consumer, on one path.**
`models/experimental/ops/descriptors/matmul.py:97` calls
`ttnn.matmul_select_program_factory(operation_params, tensor_args)` and then, at `:120`,
`factory.create_descriptor(operation_params, tensor_args, [out], core_range_set)`. When the program
config is a `MatmulMultiCoreProgramConfig` — including via
`create_simple_matmul_program_config`'s final fallback — `select_program_factory` returns the
`MatmulMultiCoreProgramFactory` alternative, and converting that alternative to Python now raises a
`TypeError` because the class is unregistered. This is the *same* failure the file already
documents for the gather_in0 factory at its lines 66-72 ("has no Python binding, so
`matmul_select_program_factory` would raise a TypeError"), which is why the file carries an
`_UNSUPPORTED_FACTORY` guard.

Suggested resolution (not the porter's to make): extend that `_UNSUPPORTED_FACTORY` mechanism to
cover ported factories, or give the framework a Metal 2.0 path. Note this is the surface that
`METAL2_PORTING_PLAN.md` raised as decision **D3** and gated Port 4 on; it arrives one port early,
because the MultiCore factory is also reachable from that framework.

### 2. Dropped a pybind-hook-only factory parameter

*Tagged: API surface — removed entry point. Owner: same as above.*

`create_descriptor`'s fourth argument existed only so the deleted pybind hook could drive the
factory; the body ignored it. The fixed `create_program_artifacts` signature cannot carry it. A
Python caller that passed a `core_range_set` to this factory was already having it silently
discarded (also recorded in the audit's Misc anomalies), so dropping it removes a no-op rather than
a capability.

### 3. Audit gap — the shared-kernel census cannot see binders outside `ttnn/cpp/ttnn/operations/`

*Tagged: doc / audit-recipe defect. Owner: the Metal 2.0 doc maintainers.*

**This is the finding with the widest reach in this report, because it applies to every port, not
just this one.** The audit recipe's census is
`grep -rl <kernel-filename> ttnn/cpp/ttnn/operations/`. The `tests/` tree is outside that path, and
a test can bind an op's kernel source through `ttnn::generic_op` + a hand-built `ProgramDescriptor`.
For this op:

| kernel | out-of-operations-tree binder |
|---|---|
| `device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp` | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:623` (`TTNNFixtureWithDevice.TestGenericOpMatmul`) |
| `device/kernels/compute/bmm.cpp` | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:658` and `:668` (same test, one descriptor per core group) |

The audit and the brief both concluded "Cross-op / shared kernels: NONE. Convert all three kernels
in place. Create no `_metal2` fork" — and the brief added "This is worth reading twice, because the
naive census says the opposite", framing the *filename* decoy so firmly that it discourages
re-running the census at all. The verdict is right for the writer and wrong for the other two.

**Why it matters rather than being cosmetic:** the Metal 2.0 generated headers are emitted and
auto-included only when `JitBuildSettings::is_metal2_kernel()` is true
(`tt_metal/jit_build/genfiles.cpp:126-129`, `:294-296`, `:550-559`), and that flag is set true only
on the `ProgramSpec` path (`tt_metal/impl/metal2_host_api/program_spec.cpp:3125`), defaulting to
false everywhere else (`tt_metal/impl/kernels/kernel.hpp:326`). So a converted kernel source has no
`args::`, `dfb::` or `tensor::` names when a legacy binder JIT-compiles it, and
`TestGenericOpMatmul` would have failed at runtime — not at build time, which is what makes it easy
to ship.

Concrete doc fix: widen the census to `grep -rl <kernel-filename> ttnn/ tests/` (or add a second
sweep over `tests/`), and note that `generic_op` tests are legitimate binders that cannot migrate.
The invoker was consulted and confirmed rung 2 (create the forks); see
[Open items](#open-items-for-downstream) for the resulting consumer/sunset list.

**Could the test bind the forks instead, retiring the legacy copies?** Not today, and not by editing
the test — the sunset needs a framework change. Three blockers, in increasing depth:

1. **`generic_op` has no `ProgramSpec` entry point.** Both public overloads take a
   `ProgramDescriptor` or a `MeshProgramDescriptor` (itself
   `vector<pair<MeshCoordinateRange, ProgramDescriptor>>`), and its only factory,
   `GenericMeshDescriptorFactory`, is a `ProgramDescriptorFactoryConcept` returning the caller's
   descriptor verbatim. `grep -rn ProgramSpec ttnn/cpp/ttnn/operations/generic/` returns nothing.
2. **`KernelDescriptor` cannot express the bindings the generated headers are built from.** The
   generator reads `process_dataflow_buffer_binding_handles`, `process_tensor_binding_handles`,
   `get_runtime_arg_names` and `get_crta_layout` off `JitBuildSettings`;
   `KernelDescriptor` (`tt_metal/api/tt-metalium/program_descriptors.hpp:145-200`) carries
   `named_compile_time_args` (name → `uint32_t`) but no DFB accessor-name/role pairs, no tensor
   bindings, and only *positional* `runtime_args`. `dfb::`, `tensor::` and `args::` have no source
   on that path, so `is_metal2_kernel()` is the symptom rather than the cause.
3. **Even hand-feeding the handles, the tensor binding would be inert.** A `TensorBindingHandle`
   carries `cta_offset` + `addr_crta_offset`, which `program_spec.cpp` computes while packing
   accessor layout metadata into the CTA list and reserving a CRTA slot; `SetProgramRunArgs` /
   `UpdateTensorArgs` write the base address into that slot at enqueue. The descriptor path has no
   such machinery — `GenericOpDeviceOperation`'s own comment states "This op never derives an
   address from a tensor: `create_descriptor` returns the caller's `ProgramDescriptor` verbatim, so
   resolving per-core addresses is the caller's job." A declared `tensor::in0` with nothing writing
   its address compiles, runs, and reads garbage.

So the prerequisite is a Metal 2.0 path through `generic_op`: a `ProgramSpec`-accepting entry point
plus a `ProgramSpecFactoryConcept` factory that passes the caller's spec through the way the present
one passes the descriptor through. Owned by TTNN.

**There is already a precedent for this exact standoff in the same test file.** An earlier port
forked `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` to
`…_metal2.cpp` and left `test_generic_op.cpp` binding the legacy copy (its lines 407, 633, 772, 855,
974). Rung 2 is what the tree already does here.

Rejected alternative: give `TestGenericOpMatmul` its own kernel copies under `tests/`, decoupling it
from matmul. It is a pure test change and needs no framework work, but it relocates the duplication
rather than removing it and costs the test its value as a check on the real op's kernels.

### 4. Audit finding wrong — `unpack_modes` *is* required here

*Tagged: doc / audit-recipe defect. Owner: the Metal 2.0 doc maintainers and the auditors.*

Both the audit and the brief state: "`unpack_modes` needs no entry: no compute kernel here consumes
a Float32 DFB." That cannot be established statically for this factory. The input DFB formats are
`tt_metal::datatype_to_dataformat_converter(tensor.dtype())`, so they *are*
`tt::DataFormat::Float32` whenever the input tensors are fp32 — and `enable_32_bit_dest` is
`fp32_dest_acc_en`, a caller-supplied knob.

`MatmulSmoke.MultiCoreExplicit` (`tests/ttnn/unit_tests/gtests/test_matmul.cpp:237`) — one of the
three cases that *pin this very factory* — runs `DataType::FLOAT32` inputs with
`fp32_dest_acc_en = true`. Both compute `KernelSpec`s consume two Float32 DFBs with a 32-bit Dest,
which the validator's required-entry rule
(`tt_metal/impl/metal2_host_api/program_spec.cpp:1095-1123`) rejects without an explicit entry. Had
the brief been followed, the port would have `TT_FATAL`'d on its own pinning test.

The port adds `{in0, UnpackMode::UnpackToSrc}` and `{in1, UnpackMode::UnpackToSrc}` on both compute
specs. This is a **zero-behavior-change** addition: the legacy `ComputeConfigDescriptor` set no
`unpack_to_dest_mode` vector, so every buffer took `UnpackToDestMode::Default`, and
`BuildUnpackToDestModeVector` (`program_spec.cpp:2740-2761`) maps `UnpackToSrc` back to exactly
`UnpackToDestMode::Default`. `UnpackToSrc` is also unconditionally legal
(`program_spec.cpp:1050-1052`), so no config-dependent gating is needed and no dtype can make the
entry itself illegal.

The recipe's own text is right and the audit misapplied it — the recipe says "**The trigger is the
DFB's format, not the op's tensor dtypes**". What the audit missed is that here the DFB's format
*is derived from* a tensor dtype, so the trigger is a property of the call, not of the op. Suggested
doc fix: state that a DFB whose `data_format_metadata` comes from an input tensor's dtype must be
treated as possibly-Float32, so the entry is unconditional; the auditor cannot clear it by reading
the factory.

### 5. No device-less verification path reachable through the sanctioned build

*Tagged: infra / doc gap. Owner: infra + the Metal 2.0 doc maintainers.*

A device-less functional run **does** exist: `tests/emule/CMakeLists.txt` registers CTest entries
that run the ordinary `unit_tests_ttnn` binary under four environment variables
(`TT_METAL_EMULE_MODE=1`, `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_RUNTIME_ROOT`,
`TT_METAL_MOCK_CLUSTER_DESC_PATH=…/wormhole_N150.yaml`), and its ttnn tier already lists matmul
filters. Attempting it here fails with

```
TT_FATAL @ tt_metal/llrt/tt_cluster.cpp:477: TargetDevice::Emule requires building with TT_METAL_USE_EMULE=ON
```

and `build_metal.sh --help` exposes no flag for that option and no generic cmake-options
passthrough. Reaching it would mean hand-targeting cmake, which the recipe forbids. So a porter on a
device-less box has no sanctioned way to run anything, even though the capability exists. Either
`build_metal.sh` should grow the flag or `workspace_setup.md` should say emule is out of reach.

---

## Successes

- **[Compiler options](#) — the `opt_level` mechanical check earned its emphasis.**
  `grep -n opt_level` on the legacy factory returns nothing, which reads naturally as "nothing to
  carry across." The recipe's insistence that an *absent* `KernelDescriptor::opt_level` still
  resolves to the legacy per-kernel-type default — `O3` for a `ComputeConfigDescriptor`
  (`tt_metal/impl/program/program.cpp:465` — the `ComputeConfigDescriptor` arm resolves
  `kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O3)`, against `O2` on the reader /
  writer / DM arms at `:428`, `:436`, `:448`) — is the only reason both compute specs carry an explicit
  `KernelBuildOptLevel::O3` at `matmul_multicore_program_factory.cpp:293`. Left alone, both would
  have quietly dropped to `O2`, and nothing in the build or the tests would have said so. The
  "do not eyeball this one" framing is correct.

- **[CB→DFB whitelist §A/§B] — the `constexpr` test resolved two rewrites that go opposite ways.**
  `writer_unary_interleaved_start_id.cpp:19`'s `get_local_cb_interface(dfb_id_out).fifo_page_size`
  is `const uint32_t`, so it became the member getter `dfb_out.get_entry_size()`
  (`writer_unary_interleaved_start_id.cpp:19` post-port). The reader's `get_dataformat(dfb_id_in0)`
  is declared `constexpr` and feeds a non-type template parameter of `pad_last_ktile`, so it kept
  the free-function form with the binding token, `get_dataformat(dfb::in0)`
  (`reader_bmm_8bank_output_tiles_partitioned_metal2.cpp:70` and `:76`). Reading "the legacy
  declaration is the entire test" made this mechanical instead of a judgment call, and the
  no-demotion-to-`const` warning is exactly right — the value is a template argument, so demoting
  it would not have compiled.

- **[Caution: Porting a shared kernel] — the *lent* case, and one sentence in it, caught the audit
  gap.** "Nothing about the path warns you: the file sits inside your writeable surface, so
  converting it in place feels safe, and it breaks every borrower the moment you do." That is a
  precise description of what almost happened to
  `device/kernels/compute/bmm.cpp` and
  `device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp`. It is also what motivated
  widening the census past the brief's confident NONE. Keep that sentence.

- **[Anti-pattern: Demoting per-group CTA to RTA] — the worked example matches the validator
  exactly.** The catalog's shape (two same-source `KernelSpec`s in two `WorkUnitSpec`s over disjoint
  node sets, both binding the same DFBs in the same roles) is precisely what the spec validator's
  per-node census accepts (`program_spec.cpp:1292-1420`: "every node hosting the DFB runs exactly
  one producer instance and exactly one consumer instance"). Worth noting because the migration
  guide's troubleshooting list says "A local DFB's producer and consumer kernels must share
  *identical* `WorkUnitSpec` membership", which reads as forbidding this shape — the reader
  (`READER`) is in both work units while each compute spec is in one. The validator is per-node, not
  per-membership-set, so the catalog is right and that troubleshooting line is misleading; see
  Friction.

- **[Construct — `Table`s are maps, not vectors]** saved a compile error outright: the legacy
  `mm_kernel_defines` is a `std::map<std::string, std::string>` and
  `KernelSpec::CompilerOptions::Defines` is a `Table` whose range constructor is `explicit`. The
  note that a legacy `std::map` of defines goes through the single-argument range constructor is
  exactly what the code needed (`matmul_multicore_program_factory.cpp:292`).

---

## Friction

### Gaps

- **The TT_FATAL census command is asymmetric and always reports a difference.** As written,
  ```bash
  diff <(git grep -cE '…' "$BASE" -- "$OP" | cut -d: -f2,3) <(git grep -cE '…' -- "$OP")
  ```
  `git grep -c <rev>` prefixes each line with `<rev>:`, so `cut -d: -f2,3` yields `path:count` on
  the left, while the working-tree side yields `path:count` already — cutting fields 2,3 there gives
  just the count. The two sides are never comparable and the check "fails" on a clean port. The
  invariant it wants held here (identical counts, in order), which took a second normalized run to
  see. Fix: drop the `cut` and strip the rev instead —
  `git grep -cE '…' "$BASE" -- "$OP" | sed "s|^$BASE:||"` against a bare
  `git grep -cE '…' -- "$OP"`.

- **The `cb`-name sweep's "expect zero hits over `<op-dir>`" is structurally unattainable for a
  single-factory port.** `matmul` has eight factories; seven stay on the legacy CB API, and the op
  directory carries **1781** `cb`-pattern hits before the port. Worse, rung 2 deliberately *keeps*
  legacy kernel copies inside the op directory, each with its `get_named_compile_time_arg_val("cb_in0")`
  lines. The check is still valuable — it just needs scoping to the port's converted/created set
  (5 files here, 0 hits) rather than the whole directory, and the doc should say so, since "zero over
  the op dir" is the stated pass condition and a porter who hits 1781 has no guidance.

- **Two self-audit checks fire on this report, because they scan the artifacts they sit beside.**
  Both are the same shape and both need scoping to code files:
  - `git diff "$BASE" | grep -nE 'METAL2_CHECKS_FORCED|DO NOT COMMIT'` — "expect no output" — hits 3
    times here, in the `METAL2_PORT_REPORT.md` prose. It has to: the recipe *instructs* the porter to
    grep the test log for `METAL2_CHECKS_FORCED`, so a report that hands the invoker that command
    contains the string. Scope it: `git diff "$BASE" -- '*.cpp' '*.hpp' '*.h'`.
  - The `TT_FATAL` census over `-- "$OP"` picks up the four `METAL2_*.md` files, and any report
    discussing a guard adds a row that has no pre-port counterpart (`METAL2_PORT_REPORT.md:4` here),
    so the diff reports a change on a clean port. Scope it the same way:
    `-- "$OP/*.cpp" "$OP/*.hpp"`.

  Neither is dangerous — both fail *loud*, in the direction of a false alarm rather than a false
  pass — but a porter following the text literally has to stop and work out that the check is
  reading its own output, on every port.

- **`unpack_modes`: the recipe shows `std::get<ComputeGen1Config>(compute_hw).unpack_modes = …`, but
  the header prefers the common-field accessor and the recipe's own TTNN helper makes `std::get`
  unsafe.** `to_compute_hardware_config(arch, cfg)` returns a `ComputeGen2Config` on Quasar, so
  `std::get<ComputeGen1Config>` would throw there — while
  `compute_hardware_config.hpp:229-240` supplies `unpack_modes(config)` precisely so callers need
  not know the alternative ("For common fields, prefer this syntax over e.g.
  `std::get<ComputeGen1Config>(config).field`, which throws if the wrong architecture is
  targeted"). This port used the accessor. The recipe's Style-A example should too, otherwise it
  hands the porter a Gen1-only line right after telling them to use an arch-agnostic helper.

- **The recipe's build/test workflow assumes subagents are available.** "When it exits, hand the log
  to a subagent to read and report… Use a Sonnet subagent" is the prescribed way to keep build noise
  out of context. This session was instructed not to spawn subagents, so the logs were read with
  targeted `grep` for `error:`/`FAILED` plus a 3-line tail, which worked fine and cost almost
  nothing. Worth offering as the no-subagent fallback rather than leaving the porter to invent it.

- **Provenance can't be recorded as specified when the docs aren't in the tree.** The recipe's
  command returns nothing, and it says to "record that fact instead" — but the more useful thing to
  record is the doc branch and commit actually read, which the recipe doesn't ask for. Suggest
  adding: if the command prints nothing, record `git log -1 --format='%h %cs %s' <doc-branch> --
  docs/…/metal_2.0/`.

### Confusion

- **"A local DFB's producer and consumer kernels must share *identical* `WorkUnitSpec` membership"**
  (migration guide, Troubleshooting) contradicts the work-split shape the patterns catalog
  prescribes and the validator accepts. In this port the reader belongs to both work units and each
  compute spec to one, so membership is *not* identical, yet the per-node census passes and this is
  the documented correct port. The sentence cost a detour into
  `program_spec.cpp` to confirm which one to trust. Suggest restating it per-node: "every node the
  DFB lives on must run exactly one producer instance and exactly one consumer instance."

- **The brief's shared-kernel section reads as a trap-warning, which suppresses re-derivation.**
  "Cross-op / shared kernels: NONE… This is worth reading twice, because the naive census says the
  opposite" is written to stop a porter from over-reading a filename grep. It works — and it also
  discourages the porter from running any census at all, which is where the two real binders were
  hiding. The recipe already tells the porter to *verify* endpoint dispositions rather than
  transcribe them; the same instruction should apply explicitly to the shared-kernel list, and a
  brief asserting NONE should still be re-derived.

---

## Open items for downstream

### Shared kernel touches

Coordination signal for the next matmul porter and the eventual sunset checklist.

| kernel | rung taken | remaining unmigrated consumers |
|---|---|---|
| `device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp` | **2 — created the fork**: `reader_bmm_8bank_output_tiles_partitioned_metal2.cpp` beside the original; pointer comment landed in the original (its lines 5-8) | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:623` |
| `device/kernels/compute/bmm.cpp` | **2 — created the fork**: `bmm_metal2.cpp` beside the original; pointer comment landed in the original (its lines 5-7) | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:658`, `:668`; and `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1433` (`TestCrossOpCompilation` reads the file's *source text*, extracting only the block above `void kernel_main(` — so it is a consumer of the file, though the conversion leaves that block unchanged apart from one added `#include`) |
| `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **converted in place** — sole binder is this factory (the 24-to-1 filename decoy the brief describes is real; the other 23 bind the `eltwise/unary` or `data_movement/slice` copies) | none |

**Fork binding vocabulary**, which later consumers inherit and cannot rename: `dfb::in0`,
`dfb::in1`, `dfb::out`; `tensor::in0`, `tensor::in1`. Taken from the kernels' own named-CTA keys
(`cb_in0` / `cb_in1` / `cb_out`, `cb_` dropped), which is also what the brief specified. Named args:
reader CTAs `in0_last_ktile_w`, `in0_last_ktile_h`; reader RTAs `Mt`, `Kt`, `Nt`, `MtKt`, `KtNt`,
`batch`, `bcast_B`, `output_tile_start_id`, `num_output_tiles`, `MtNt`; compute CTAs `batch`, `Mt`,
`Kt`, `Nt`. No `#ifdef`s are gated on by either fork.

**Sunset:** the two legacy copies can be deleted, and the forks renamed onto their names, once
`test_generic_op.cpp` stops binding matmul's kernels. That is gated on `generic_op` growing a
`ProgramSpec` entry point — the test cannot be repointed at the forks before then, for the three
reasons set out in handoff point 3. Once it lands the cleanup is mechanical: repoint the test at the
forks, delete the legacy copies, rename the forks. Nobody is tracking that dependency today.

### Findings for the op owner (preserved, not fixed)

None of these were changed; all are pre-existing behavior the port reproduces.

- **Three reader RTAs are dispatched and never used.** `Mt` (slot 2), `MtKt` (slot 5) and `batch`
  (slot 7) are read into locals and never referenced —
  `reader_bmm_8bank_output_tiles_partitioned_metal2.cpp:22`, `:25`, `:27`. The host still pays to
  set them on every node, on every dispatch. Dropping them would shrink the RTA payload; that is a
  behavior change (dispatch cost) and so not port work.
- **Eight of the ten reader RTAs are node-invariant and are really CRTAs.** `Mt`, `Kt`, `Nt`,
  `MtKt`, `KtNt`, `batch`, `bcast_B`, `MtNt` take the same value on every node
  (`matmul_multicore_program_factory.cpp:242-253`); only `output_tile_start_id` and
  `num_output_tiles` vary. `common_runtime_arg_values` would dispatch these once instead of
  per-node. RTA→CRTA changes dispatch semantics, so the recipe routes it to a later pass; recorded
  here for that pass.
- **`in0_last_ktile_h` is hardcoded to 0** (`matmul_multicore_program_factory.cpp:149`), so the
  reader's `if constexpr (in0_last_ktile_h > 0) { … pad_last_transposed_ktile … }` block
  (`…_metal2.cpp:74-79`) is unreachable from this factory. Live for sibling factories' transposed
  paths. Carried across unchanged, as the brief directs. Whether the hardcoded 0 is a silent
  `transpose_a` limitation is the owner's question (also in the audit's Misc anomalies).
- **Two dead preprocessor branches in matmul's private writer copy.** `OUT_SHARDED`
  (`writer_unary_interleaved_start_id.cpp:21`) and `BACKWARDS` (`:30`) are never defined — this
  factory emits no `defines` on the writer at all, and now that the copy has exactly one binder both
  branches are unreachable. Carried across; a candidate cleanup.
- **Two unused locals in the compute kernel**, `dst_tile_index` and `in0_block_tile_index`
  (`bmm_metal2.cpp:25-26`), preserved from the original.
- **A commented-out `DPRINT` in the reader fork names deleted locals.**
  `…_metal2.cpp:37` still reads `// DPRINT("src0={} src1={}\n", src0_addr, src1_addr);`, but
  `src0_addr` / `src1_addr` are gone (they were the buffer-address RTAs the `TensorBinding`
  replaced). Kept verbatim under the comment-preservation rule — deleting a comment is what that
  rule forbids — but uncommenting it will not compile. Flagging so the next person who reaches for
  it is not surprised.
- **`all_cores` is now unused in the factory.** Metal 2.0 derives DFB and kernel placement from
  `WorkUnitSpec::target_nodes`, so the third element of the `split_work_to_cores` structured binding
  has no consumer. It cannot be omitted (structured bindings take every element) and no warning
  fired.

### Test coverage notes

- **No test deterministically pins the two-compute-`KernelSpec` path on every architecture.**
  `MatmulSmoke.MultiCoreExplicit`'s `{2080, 64, 64}` case exists to force a non-empty
  `core_group_2`, and its own comment records the caveat: "a 130-core grid, e.g. Blackhole 13x10,
  divides evenly." On such a grid the `COMPUTE_G2` spec and its work unit are never built, so the
  preserved-multiplicity shape — the part of this port most likely to break — goes unexercised.
  A case whose output-tile count is coprime with every supported grid would close this.
- **`TestGenericOpMatmul` is now a regression test for the fork decision**, not just for
  `generic_op`: it is the only thing that would notice if a future port converted matmul's legacy
  reader or `bmm.cpp` in place. Worth a comment there, which is outside this port's writeable
  surface.

### Verification status

**Done in this workspace:**

- Clean build through `./build_metal.sh --build-tests` (three times: cold, then after each edit
  round). No errors and no warnings naming any ported file.
- Full anti-pattern self-audit, each sweep reported as *hits / files scanned* over the port's
  converted/created set (**5** files: the factory `.cpp`/`.hpp`, the two kernel forks, the
  in-place writer):
  - buffer address / `emplace_runtime_args` / bare `Buffer*` in run-args — **0 / 5**
  - `CBIndex` / `CBDescriptor` / `CBFormatDescriptor` / `CircularBuffer` — **0 / 5**
  - `TensorAccessorArgs` — **0 / 5**
  - `cb`-shaped names (`[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]`) — **0 / 5** (22 before the
    port)
  - `.id` extraction / `get_id()` at LLK call sites — **0 / 5**
  - `allow_instance_multi_binding` — **0 / 5**
  - varargs of any kind — **0 / 5**
  - positional `compile_time_args` — **0**; both sites are named-pair form
  - `opt_level` — one line, in the shared `make_compute` helper, so **both** compute `KernelSpec`s
    carry `O3`; the two DM specs correctly carry none (legacy DM default `O2` == Metal 2.0 default)
  - ephemeral `.md` citation from code — **0 hits / 10 files scanned** (the diff-scoped list, which
    included both untracked kernel forks)
  - `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census across the op directory — **identical**, file for
    file, pre-port vs post-port (5 in the ported factory, before and after)
  - `hw_config` before/after, by resolved value:

    | kernel | legacy resolved | ported | equal |
    |---|---|---|---|
    | reader | `ReaderConfigDescriptor{}` → `ReaderDataMovementConfig` → RISCV_1, NOC_0, DM_DEDICATED_NOC (`kernel_types.cpp:19-22`, `preferred_noc_for_dram_read` returns NOC_0 on every arch) | `create_reader_datamovement_config(arch)` → `CreateReaderGen1DataMovementConfig()` → RISCV_1, NOC_0, DM_DEDICATED_NOC | ✓ |
    | writer | `WriterConfigDescriptor{}` → RISCV_0, NOC_1, DM_DEDICATED_NOC | `create_writer_datamovement_config(arch)` → RISCV_0, NOC_1, DM_DEDICATED_NOC | ✓ |
    | compute ×2 | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` from the resolved TTNN config; `bfp8_pack_precise` and `unpack_to_dest_mode` unset | `to_compute_hardware_config(arch, config)` → `fpu_math_fidelity`, `enable_32_bit_dest`, `double_buffer_dest = !dst_full_sync_en`, `sfpu_precision_mode` from `math_approx_mode`; `bfp_pack_precision_mode` left default (`Approximate` == legacy `false`); `unpack_modes` set to `UnpackToSrc` on the two consumed input DFBs, which lowers to `UnpackToDestMode::Default` — the legacy value | ✓ |

    No dropped field: the factory resolves five knobs and set four; the fifth, `packer_l1_acc`, has
    no Metal 2.0 counterpart (it was explicitly discarded in the legacy factory too).
- Test-set collection confirmed without a device: **1182** tests in the three
  `unit_tests/operations/matmul/` files, **2702** in `nightly/unit_tests/operations/matmul/` (one
  file, `test_rs_matmul_1d_gather_in0.py`, errors at collection with "No chips detected in the
  cluster" — a device artifact, unrelated to the port), **26** in the `tt_eager` legacy sweep,
  **10** `cross_op_compilation` cases, and the gtest names
  `MatmulSmoke.{MultiCoreExplicit,MultiCoreUnaligned,MultiCorePostProcessedBias}` plus
  `TestGenericOpMatmul` all present in `unit_tests_ttnn`.

**Not done, and needs a device:**

- Every runtime check: the spec validator, the JIT compile of the three converted kernel sources,
  numerics, and program-cache behavior.
- The `METAL2_CHECKS_FORCED` proof. The forcing *is* in place and *is* in the built binary —
  `skip_validation = false` was set as the first statement of all nine sites named by
  `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp`, with a one-shot
  `log_warning` marker in each of the two files, and
  `strings build_Release/lib/libtt_metal.so | grep METAL2_CHECKS_FORCED` finds it. But grepping a
  *test log* for two markers requires running a test.

**The forcing scaffolding is deliberately left uncommitted in the working tree**, so the
already-built binary can be used the moment a device appears. It must never be committed; the
committed diff contains no `tt_metal/` file (verified). To drop it:

```bash
git checkout -- tt_metal/impl/metal2_host_api/program_run_args.cpp tt_metal/impl/metal2_host_api/program_spec.cpp
```

**Commands to finish verification** (from the checkout root, venv active):

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
source python_env/bin/activate
export TT_METAL_WATCHER=10          # Watcher on for every run; unset is the only way off

# 1. gtests first -- the three cases that pin this factory, plus the generic-op binder
#    of the two kernels this port forked.
./build/test/ttnn/unit_tests_ttnn \
  --gtest_filter='MatmulSmoke.*:*GenericOpMatmul*' 2>&1 | tee /tmp/mm_gtest.log
grep -c METAL2_CHECKS_FORCED /tmp/mm_gtest.log   # expect >= 2 before trusting any green

# 2. pytests
pytest tests/ttnn/unit_tests/operations/matmul/test_matmul.py \
       tests/ttnn/unit_tests/operations/matmul/test_matmul_program_cache.py \
       tests/ttnn/unit_tests/operations/matmul/test_matmul_batch_mismatch.py -x
pytest tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py \
       -k cross_op_compilation -x
pytest tests/ttnn/nightly/unit_tests/operations/matmul/ -x
pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/tt_dnn/test_matmul.py -x
```

The three sharpest cases are `MatmulSmoke.MultiCoreExplicit` (fp32 + `fp32_dest_acc_en`, which is
what exercises the `unpack_modes` requirement of handoff point 4, and whose `{2080,64,64}` shape is
what may exercise the second compute `KernelSpec`), `MatmulSmoke.MultiCoreUnaligned` (the
`in0_last_ktile_w` padding path in the reader fork), and `TestGenericOpMatmul` (proves the fork
decision of handoff point 3 kept the legacy binder working).
