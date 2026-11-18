# Port Report — slice_reshard_async

## Outcome

**PORTED** — `SliceReshardAsyncProgramFactory` (the only factory) is on `MeshWorkloadSpecFactoryConcept` with a translated
`override_runtime_arguments`. Both kernels are converted in place. Multi-chip runtime behavior is not verified on this host
(see Verification).

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
  (from `git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`; the
  recipe docs are not in this checkout)
- **Audit docs (inherited):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
- **Worked example:** `point_to_point` on this branch (`2e5f720ff76..c210a7093f6`), incl. its branch-specific overrides.

## TTNN ProgramFactory

- **Concept realized:** `MeshWorkloadSpecFactoryConcept`. `create_mesh_workload_artifacts` returns one `PerCoordProgram` per
  coordinate of `tensor_coords` (same granularity as the `WorkloadDescriptor`). `override_runtime_arguments(attrs,
  tensor_args, tensor_return_value, const MeshCoordinateRange&)` returns `ProgramRunArgs`.
- **Override translation.** The ported-from override wrote the final semaphore address (reader RTA 9, writer RTA 14) and the
  barrier semaphore address (writer RTA 18) on every core of every program. The Metal 2.0 override returns
  `global_semaphore_args` for `final_semaphore` and `barrier_semaphore`, plus `tensor_args` for `input` and `output` (from a
  `WorkloadDescriptor` source the framework patched the bindings, so the override now owns them). No `kernel_run_args`. The
  cache-miss run args are built by the same helper (`make_dispatch_run_args`) plus the per-core RTAs, so miss and hit cannot
  drift apart.
- **`MeshWorkloadArtifacts::global_semaphores`:** empty. Both semaphores are caller-owned (`operation_attributes_t`).
- **Custom hash left intact:** `device/slice_reshard_async_device_operation.cpp:50`.
- **Device-op-class edits:** none. The factory header changed (it is the factory); the device-op files are byte-identical.
- **Pybind entry points removed:** none.

## Handoff points

none — no capitulation, no boundary violation, no removed pybind surface.

## Successes

- **`ttnn_factory.md` — "The override owns the tensor bindings too", `WorkloadDescriptor` exception.** Without it the override
  would have carried only the two semaphores (what the ported-from override wrote), and tensor bindings would have frozen at
  the cache-miss addresses. `slice_reshard_async_program_factory.cpp` `make_dispatch_run_args`.
- **`metal2_port.md` — Preserved Multiplicity.** The legacy factory has one reader and one writer descriptor per core with
  per-direction CTAs. Kept as `reader_<i>` / `writer_<i>` KernelSpecs with one work unit per core, CTAs unchanged.
- **Self-audit — no `Buffer*` and `cb` sweeps.** Caught a leftover `Buffer* input_buffer` local (kept only for
  `page_size()`) and a stale "into the CB" comment in the reader.

## Friction

- **Gap — caller-supplied GlobalSemaphores.** The recipe (and the branch overrides) only describe op-owned semaphores. A
  caller-supplied semaphore that the custom hash leaves out and the override refreshes maps cleanly onto
  `global_semaphore_args` in both the miss run args and the override, but nothing says so. `UpdateProgramRunArgs` accepts a
  semaphore-only partial update; `ValidateGlobalSemaphoreArgs` checks that the semaphore spans every binder node.
- **Gap — a binder node without a fabric target.** Kernel side, `build_from_binding` has the precondition
  `has_fabric_connection`. The P2P example always had a connection; here the first/last device's writer in one direction has
  none, so the sender is default-constructed and built only under the flag, and `open()` / `close()` are guarded the same way
  `FabricConnectionManager` guarded them. A short note in the fabric-connection docs would help the next porter.
- **Confusion — collapsing the forward/backward pick.** Override 4b says to replace the manager "incl. the forward/backward
  flags". With one connection per writer, the `if (direction) { forward } else { backward }` blocks became one block on the
  single sender (`minimal_default_writer.cpp`). This is the direction pick that 4b removes, but it is a structural change in
  the kernel; the alternative was two identical branches.
- **Confusion — 3rd-arg triage says Class 1.** The triage lists `slice_reshard_async` as Class 1, but the custom hash keys on
  the input `TensorSpec`, so the page size is fixed per cache entry → Class 2, no `dynamic_tensor_shape`.

## Open items for downstream

- **Not verified at runtime.** Only the mock-fabric sentinel ran here (spec validation, run-args validation, JIT compile).
  The fabric connection, the GlobalSemaphore CRTAs and the cache-hit override run only in CI:
  `tests/nightly/t3000/ccl/test_slice_reshard_async.py` (has a multi-iteration cache-hit case with fresh semaphores each
  iteration) and the `models/tt_dit` VAE path (`models/tt_dit/parallel/config.py:263`).
- **Dropped `TT_FATAL`s (subject deleted), `slice_reshard_async_program_factory.cpp`:** 4 → 1.
  - `output_buffer != nullptr` (old `:69`) — guarded a raw `Buffer*` that the output `TensorBinding` replaced.
  - Two `desc.kernels.size() == reader/writer_kernel_idx(core_idx)` (old `:185`, `:268`) — guarded the kernel push order
    that the index-based legacy override relied on. Kernels are now addressed by name; the order has no meaning.
  - `input_tensor.buffer() != nullptr` is kept: the factory still reads the page size from the buffer.
- **Dead writer RTA dropped:** writer RTA0 `input_tensor_address` (old `minimal_default_writer.cpp:40`) was never used; the
  writer has no input binding. Zero behavior change.
- **DFB placement edge case.** Legacy put the CB only on `worker_core_ranges` but created kernels on every core of
  `CoreCoord(num_links * 2, 1)`. When `num_sticks_per_outer_dim < num_links` some cores had kernels and no CB (undefined in
  legacy, see `METAL2_PREPORT_AUDIT.md` Misc anomalies). Metal 2.0 places the DFB on every node of its binding kernels, so
  those cores now get the DFB. Same L1 use and behavior in every other config.
- **Quasar uplift.** No DM self-loop DFB; the one DFB is a proper 1P+1C. The writer still uses `PacketHeaderPool` for
  packet headers (fabric firmware pool, not a DFB). The hw configs use the arch-agnostic TTNN helpers.
- **Sibling ops.** Other `experimental/ccl` ops with the same shape (per-direction writers, one fabric neighbor per core,
  caller-supplied GlobalSemaphores refreshed by an override) can follow this port: e.g. `neighbor_pad_async`.

## Verification

Host: 4x Blackhole p300b; multi-device meshes cannot open here, and the op `TT_FATAL`s on a 1-device ring, so no pytest runs
on this host.

- **Build:** `./build_metal.sh -b RelWithDebInfo --build-tests -e --enable-ccache` — SUCCESS.
- **Legality checks:** not forced in `tt_metal/impl`. The only sentinel that exercises specs calls `MakeProgramFromSpec`,
  `SetProgramRunArgs` and `UpdateProgramRunArgs` directly with `skip_validation = false` (the default), so the checks ran.
  The TTNN adapter path (where TTNN sets `skip_validation`) cannot run here.
- **Sentinel, before the port** (`c210a7093f6`, throw-away gtest, not committed, on `ProgramSpecFabricTestGen1`): the legacy
  `ProgramDescriptor` for the first, middle and last device of a ring (two cores, legacy CTAs/RTAs,
  `append_fabric_connection_rt_args`) built into a `Program` and JIT-compiled — 3 passed, plus the existing
  `CPU_FabricConnectionBindingJITSmoke`.
- **Sentinel, after the port** (same throw-away file): the `ProgramSpec` the ported factory builds (same KernelSpecs, DFB,
  tensor parameters, GlobalSemaphore parameters, fabric connection, named args) for the first, middle and last device ran
  through `MakeProgramFromSpec` and JIT-compiled — 3 passed. A fourth case ran `SetProgramRunArgs` with every named RTA, both
  tensors and both GlobalSemaphores, then `UpdateProgramRunArgs` with the override's run args (fresh tensors and semaphores) —
  passed. This checks spec and run-args legality and kernel compilation only; it does not run the host factory or move data.
- **Self-audit** (11 `.cpp`/`.hpp` files in the op directory): no `Buffer*` / `address()` / `emplace_runtime_args` in the
  factory; no `TensorAccessorArgs`, `get_compile_time_arg_val`, `get_arg_val`, `CircularBuffer` or `CBDescriptor`; `cb`
  name grep — 0 hits; no `.md` citation in changed code; no `tt_metal/` file in the commit; `TT_FATAL` counts as above; no
  `opt_level` (DM-only, legacy unset → O2 both sides); `hw_config` = reader/writer defaults as legacy
  (`ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`); no varargs; no multi-binding flag.
