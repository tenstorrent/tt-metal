# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/reduction/manual_seed`

## Outcome

**`PORTED`** — all four factories of `ManualSeedDeviceOperation` converted to `ProgramSpecFactoryConcept`
in one change, together with all five kernel sources the op owns. The confirmed test set (9 tests) passes
post-port, matching the pre-port baseline exactly; the final run used an empty JIT kernel cache, so every
kernel was recompiled from the ported sources.

Nothing is left for a later pass: no factory was skipped, no capitulation, no kernel fork.

## Provenance

- **Recipe docs (this port):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `4d6e9518bf5 2026-07-30 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, on all four factories — as the audit decided. Each factory's
`static ProgramDescriptor create_descriptor(...)` became
`static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`, so the whole
`program_factory_t` variant is on the new concept and no factory is left on the legacy one. No
disagreement with the audit arose.

### Device-op-class edits

- Custom `compute_program_hash` deleted: **none** — the op never defined one, so it was already on the
  default reflection-based hash.
- Pybind entry points removed: **none** — `manual_seed_nanobind.cpp` exposes only the top-level
  `manual_seed` function, never `create_descriptor`. No factory parameter existed solely for a pybind hook.

The device-operation class (`manual_seed_operation.cpp` / `.hpp`) and the nanobind file are **unmodified**;
the diff is confined to the two program-factory files and the five kernel sources.

### Open items

- **Relaxation candidates: none, and deliberately so.** The readers only ever read `page_id = 0`, so at a
  glance the `user_ids` / `seeds` `TensorParameter`s look like they would tolerate
  `advanced_options.dynamic_tensor_shape`. They would not buy anything: `number_of_ids` is a **compile-time**
  arg derived from the tensor's volume, so a shape change has to rebuild the program regardless. Strict
  matching is the right setting here and no relaxation should be added later either.
- **Raw `MeshDevice*` in the hashed attributes.** `ManualSeedParams::device`
  ([manual_seed_device_operation_types.hpp:13](device/manual_seed_device_operation_types.hpp#L13)) is a raw
  pointer inside `operation_attributes_t`, so it feeds the default program hash by pointer value. The port
  neither needed nor changed this, and it is used host-side only, but it is unusual enough in a cache key to
  be worth an owner's eye. (Also recorded as audit Misc anomaly 5.)

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation (no call site needed a `sem::` or
`tensor::` handle outside the op), no kernel-lib gap, no framework gap that bit, no removed pybind surface,
and no edit anywhere outside the op's own directory.

## Successes

- **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  is what made three of the five buffers expressible at all.** `user_ids` (factories 3 and 4) and `seeds`
  (factory 4) are NoC read landing areas the reader fills and reads straight back
  ([reader_manual_seed_read_all_data.cpp:37-48](device/kernels/dataflow/reader_manual_seed_read_all_data.cpp#L37-L48)).
  Nothing pushes or pops them and no second kernel touches them, so a naive 1:1 port would have produced a
  DFB with one endpoint and been rejected by the "≥1 PRODUCER and ≥1 CONSUMER" validator. The pattern names
  the shape and prescribes the fix, including the explicit statement that a **DM** self-loop is legal on
  Gen1 — without that sentence the natural reading of the self-loop entry (whose examples lead with compute)
  is that it does not apply to a reader.

- **"Re-derive, don't transcribe" on endpoint dispositions
  ([endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split))
  paid off as a confirmation rather than a correction.** Running the census independently produced the same
  five rows as the brief, and it surfaced the structural reason the landing buffers are one-toucher: the
  compute kernels bind *only* `kernel_communication`, so they cannot reach the other buffers even in
  principle. That is a stronger argument than "the brief said so", and it is the argument that justifies not
  reaching for `allow_instance_multi_binding`.

- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  rule 2 caught a silent perf regression that greps as "nothing to do".** `grep -n opt_level` over the
  legacy factory returns **zero** hits, which reads as "the op set no optimization level, so there is nothing
  to carry over." The section's insistence that an absent `KernelDescriptor::opt_level` is *not* "no setting"
  — it resolves to `O3` on a `ComputeConfigDescriptor` while Metal 2.0 defaults to `O2` — is the only reason
  all four compute `KernelSpec`s carry an explicit `KernelBuildOptLevel::O3`. Nothing in the build or the
  tests would have flagged the drop.

- **[Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)
  Style A / Style B fork steered away from the wrong helper.** This op builds its compute config as a bare
  `ComputeConfigDescriptor{}` with no TTNN `ComputeKernelConfig` behind it — Style B. The instruction "don't
  reroute these through the TTNN helper to save typing" is what stopped me reaching for
  `to_compute_hardware_config`, which needs a resolved `ComputeKernelConfig` this op does not have and
  defaults the opposite way. A field-by-field diff of the legacy `ComputeConfigDescriptor` defaults against
  `ComputeGen1Config` defaults confirmed all six knobs coincide, so a bare `ComputeGen1Config{}` reproduces
  the legacy config exactly.

## Friction

### Gaps

- **The brief issued an instruction that the port cannot follow, and the recipe had to override it.**
  The brief ("Kernel-side metadata lookups (whitelist rule 7)") says to leave three
  `constexpr DataFormat … = get_dataformat(<dfb_index>);` lines untouched — pre-port at
  `reader_manual_seed_read_user_id.cpp:29` and `reader_manual_seed_read_all_data.cpp:33,36`, now at
  [reader_manual_seed_read_user_id.cpp:30](device/kernels/dataflow/reader_manual_seed_read_user_id.cpp#L30)
  and
  [reader_manual_seed_read_all_data.cpp:31-34](device/kernels/dataflow/reader_manual_seed_read_all_data.cpp#L31-L34)
  — reasoning that the results are dead and that moving the call onto the `DataflowBuffer` object
  would cost a declaration reorder. But the argument those calls take *is* the CB-index CTA the port deletes,
  so "leave as-is" was never an available option — the lines cannot compile unchanged. The port followed
  [whitelist rule 7](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)
  instead (`user_ids_dfb.get_dataformat()`), paying the reorder: the `Noc` / `DataflowBuffer` declarations
  moved above the tensor-config block in both readers.
  **Right answer:** a `get_*(cb_id)` call site can only be "left alone" when its argument survives the port.
  Since a CB-index CTA never survives, the audit should not offer a leave-as-is disposition for one at all —
  the choice is only ever between rule 7's object getter and the `dfb::name → uint32_t` shim, and rule 2's
  note already settles that in rule 7's favour.

- **Rule 7 doesn't mention that the `constexpr` on the receiving variable has to weaken to `const`.**
  The legacy free functions are `constexpr inline`, so kernels routinely write
  `constexpr DataFormat f = get_dataformat(cb_id);`. The DFB member getters are themselves marked
  `constexpr`, but `DataflowBuffer`'s constructor is not
  ([dataflow_buffer.h:72-75](../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L72-L75)), so the
  object is not usable in a constant expression and `constexpr DataFormat f = dfb.get_dataformat();` cannot
  work. Rule 7 and the [whitelist §A table](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md)
  present the change as a pure name mapping, so this is invisible until the porter tries it. The same applies
  to `get_tile_size`, where the legacy call folded to a literal and the DFB call is an array index the
  optimizer has to fold. **Right answer:** one sentence in the whitelist's metadata table — "the DFB object is
  not constexpr-constructible, so a `constexpr` local reading one of these getters becomes `const`."

- **The recipe never states that a whole `ProgramRunArgs` may be empty.** Factories 1 and 2 build a program
  whose only kernel has no runtime args at all, so their `ProgramArtifacts` returns `.spec` alone and lets
  `run_params` default-construct
  ([manual_seed_program_factory.cpp:89-90](device/manual_seed_program_factory.cpp#L89-L90),
  [:125-126](device/manual_seed_program_factory.cpp#L125-L126)). The
  [Construct](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#construct-paired-spec--run-args)
  step always shows both fields being moved in, and only `program_run_args.hpp` implies the empty case is
  legal, via the parenthetical "except for kernels that have no runtime or common runtime arguments". It does
  work — verified on device — but the recipe leaves the porter guessing whether an argument-free program is a
  supported shape or a sign of a missed step. The brief's "that is a legitimately minimal `ProgramSpec`, not
  a sign you missed something" note covers the *spec* side of exactly this doubt; the recipe should say the
  same for the run-args side.

### Confusion

- **Rung 3 of the shared-kernel Caution reads as unavailable for an intra-op kernel, when it is the obvious
  answer.** `manual_seed_set_seed.cpp` is bound by two of this op's own factories.
  [Rung 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  gates in-place conversion on the invoker "explicitly assigning you the bundled port — every binding factory
  named", and warns that a consumer list is "a sunset and coordination list, not authorization". Both
  sentences are written for the cross-op case, where the binders belong to other people's ops. Here the
  invoker assigned the whole device-operation, both binders are inside it, and forking would leave a
  permanent duplicate serving nobody — so rung 3 is plainly right, but the wording made me re-read the rung
  three times to be sure I was not talking myself past a guard rail. **Right answer:** state it directly —
  for an *intra-op* shared kernel, an assignment covering the whole device-operation **is** the bundled
  assignment, provided every binding factory converts in the change.

- **A "circular buffer" mention in a kernel comment is caught by the sweep rule but not addressed by the
  comment rule.** Rule 1 requires that no CB references survive, naming "stale comments referencing CB";
  rule 8 forbids deleting comments and permits only "slight tweaks to align an existing comment with the line
  you're forced to change." The two meet at `// Read user_id from circular buffer`, which is both a CB
  reference and a comment on a line the port changes. I applied the API rename in place
  (`… from dataflow buffer`) and changed nothing else about it, which seems to be what both rules want, but
  neither says so. A one-line example in rule 1's sweep bullet would settle it.

### Not friction, noted for accuracy

The recipe prescribes `./build_metal.sh --build-tests`. This op has no C++ gtest and no sweep — its entire
baseline is 9 Python tests — so the TTNN test binaries were never needed and the workspace's lighter build
sufficed. Not a doc defect (the recipe's "no target list to maintain" rationale is sound), just a case where
the heavier build buys nothing.

## Open items for downstream

- **Shared kernel touches.**
  `device/kernels/compute/manual_seed_set_seed.cpp` — **intra-op** share, **rung 3: converted in place**.
  Bundled set (both converted in this change): `ManualSeedSingleSeedToAllCoresProgramFactory` and
  `ManualSeedSingleSeedSingleCoreProgramFactory`. **Remaining unmigrated consumers: none** — the filename
  census over `ttnn/cpp/ttnn/operations/` finds no other binder of this or any of the op's other four
  kernels. No `_metal2` fork was created and none is needed, so there is no sunset list and no pointer
  comment to place. A future porter of a *new* factory binding this kernel inherits a Metal 2.0 kernel, not a
  legacy one.

- **The three dead `DataFormat` locals are still there, and are now cheaper to delete.**
  ([reader_manual_seed_read_user_id.cpp:30](device/kernels/dataflow/reader_manual_seed_read_user_id.cpp#L30),
  [reader_manual_seed_read_all_data.cpp:31-34](device/kernels/dataflow/reader_manual_seed_read_all_data.cpp#L31-L34))
  The port had to rewrite these lines (rule 7) but deliberately did not delete the variables — that is the
  ops team's call, per audit Misc anomaly 1. They are now `const` locals rather than `constexpr`; the JIT
  compiles kernels with `-Wno-unused-variable`, so they stay silent, but they are pure dead weight and
  deleting them would also retire the last `get_dataformat` question in this op.

- **Audit Misc anomalies 2–6 are untouched and still open** for the ops team. Anomaly 2 in particular — the
  NoC read transfers a *tile* size (4096 B for UINT32) from `page_id = 0` of a rank-1 row-major tensor — was
  preserved semantically: `get_tile_size(<cb_index>)` became `<dfb>.get_tile_size()`, the same value from the
  same descriptor table. The port changed nothing about the over-read or the >1-page correctness hazard.

- **Test coverage is thin for two of the four factories.** `SingleSeedToAllCores` and `SingleSeedSingleCore`
  are reached by tests that assert only "does not throw" — the reproducibility assertions in
  `test_manual_seed.py` all run through `ttnn.sampling` on the all-cores path, and the scalar-`user_ids`
  path (`test_manual_seed_skip_with_uint32_max_user_ids_scalar`) makes no assertion at all. A test that seeds
  a single core by index and shows that *only* that core's PRNG changed would give the per-core factories the
  same grade of coverage the tensor-mapping factories get. Not port work; worth someone's time.

- **Carry-over for sibling ops.** Any op whose reader borrows a CB purely as a NoC landing area and then
  reads it back through `CoreLocalMem` has the same one-toucher shape as this op's `user_ids` / `seeds`
  buffers, and takes the same DM self-loop. That idiom is common in small config-reading readers, so the
  next porter who hits "my DFB has only a producer" on a reader should check for it before assuming a missing
  partner kernel.
