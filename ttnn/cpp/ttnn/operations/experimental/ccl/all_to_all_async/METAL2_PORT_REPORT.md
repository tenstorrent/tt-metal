# Port Report — all_to_all_async

## Outcome

**PORTED** — `AllToAllAsyncProgram` (the only factory) is on `MeshWorkloadSpecFactoryConcept` with a translated
`override_runtime_arguments`. All four kernels are converted in place. Multi-chip runtime behavior is not verified on this
host (see Verification).

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
  (from `git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`; the
  recipe docs are not in this checkout)
- **Audit docs (inherited):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
- **Worked examples:** `slice_reshard_async` (`37ea412ca55..96a29057d61`), `all_to_all_dispatch`
  (`dbf8fd3bd30..c116d1cba2f`) and `all_to_all_combine` (`377d1c00652..5f28f745c0c`) on this branch, incl. the branch
  overrides.

## TTNN ProgramFactory

- **Concept realized:** `MeshWorkloadSpecFactoryConcept`. `create_mesh_workload_artifacts` returns one `PerCoordProgram`
  per coordinate of `tensor_coords`, range = that coordinate — the same granularity as the per-coordinate
  `create_descriptor`. `override_runtime_arguments(attrs, tensor_args, tensor_return_value, const MeshCoordinateRange&)`
  returns `ProgramRunArgs`.
- **Override translation.** The ported-from override wrote, on every core of every program, the input address (sender
  reader RTA0, receiver reader RTA1), the intermediate address (sender writer RTA0, receiver reader RTA0), the output
  address (sender writer RTA1, receiver writer RTA0) and the semaphore address (sender writer RTA2, receiver reader RTA2).
  The Metal 2.0 override returns `tensor_args` for `input`, `intermediate` and `output` (every io-tensor `TensorParameter`)
  and `global_semaphore_args` for `semaphore`. No `kernel_run_args`: the ported-from override wrote no other RTA. The
  cache-miss run args come from the same helper (`make_dispatch_run_args`) plus the per-core RTAs, so miss and hit cannot
  drift apart. The ported-from override read no coordinate; the new one ignores `range`.
- **`MeshWorkloadArtifacts::global_semaphores`:** empty. The semaphore is caller-owned (`operation_attributes_t`).
- **Custom hash left intact:** `device/all_to_all_async_device_operation.cpp:246`.
- **Device-op-class edits:** none. The factory header changed (it is the factory); the device-op files are
  byte-identical.
- **Pybind entry points removed:** none (`all_to_all_async_nanobind.cpp` binds only the op).

## Handoff points

none — no capitulation, no boundary violation, no removed pybind surface.

## Successes

- **`ttnn_factory.md` — "The cache-hit contract" and "The override owns the tensor bindings too".** The ported-from
  override already refreshed every tensor address, so the translated override names every io `TensorParameter`, plus the
  semaphore, and nothing else. `all_to_all_async_program_factory.cpp` `make_dispatch_run_args`.
- **`port_patterns.md` — Self-loop DFB binding.** The packet-header CB has only the sender writer as a toucher; it is
  bound PRODUCER and CONSUMER with one accessor name (`packet_header`), and the kernel keeps its
  `reserve_back` / `push_back` calls unchanged.
- **Branch override 7 (one spec per destination slot).** The two `append_fabric_connection_rt_args` calls of the sender
  core became two `FabricConnectionSpec`s (`forward`, `backward`) bound by the one writer. No `(src, dst, link)` repeats,
  because the factory rejects every config where the two neighbors are the same device (`TT_FATAL(forward != backward)`).

## Friction

- **Gap — `FabricConnectionManager` open/close order with two bindings.** The legacy manager ran
  `BUILD_AND_OPEN_CONNECTION_START_ONLY` (build + `open_start` forward, then backward), `open_finish` for both later, and
  `close()` = `close_start` both, then `close_finish` both. With two `fabric::` bindings there is no manager, so the writer
  repeats that order by hand with one flag per sender (`interleaved_all_to_all_writer.cpp`). A small kernel-side helper
  that takes several binding tokens and keeps the manager's two-phase open/close would remove the repetition; the
  overrides say how to build one sender, not how to keep a manager's phase order for several.
- **Confusion — a CTA and an RTA with the same name.** Both receiver kernels declare a file-scope `constexpr
  num_pages_per_packet` (CTA) that a local `num_pages_per_packet` (RTA, same value) shadows. Named args must be unique per
  kernel, so the CTA got the arg name `packet_size_in_pages` (the name the sender kernels use for the same value) and both
  variables stay. The recipe's "names match the variables" rule has no case for a shadowed name.
- **Confusion — retained Device 1.0 read.** The receiver reader keeps `noc_async_read(<accessor>.get_noc_addr(first_id,
  0, 0), …)`, which the Device 2.0 migration retained on purpose. The port changes only the accessor construction
  (`TensorAccessor(tensor::intermediate)`); the recipe's "transfers stay as-is" note covers it.
- **Confusion — `cb`-named locals.** The kernel locals `cb` (receivers) become `dfb`, which is also the binding-token
  namespace (`dfb::receiver_data`). It compiles (a name before `::` looks up only namespaces and types), and the
  migration guide uses the same form, but it reads oddly.

## Open items for downstream

- **Not verified at runtime.** Only the mock-fabric sentinel ran here (spec validation, run-args validation, JIT compile).
  The fabric connections, the GlobalSemaphore CRTAs and the cache-hit refresh run only in CI:
  `tests/nightly/t3000/ccl/test_all_to_all.py` (has a multi-iteration case with a fresh semaphore per iteration),
  `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/{all_post_commit,nightly}/test_all_to_all.py`,
  `tests/nightly/t3000/ccl/test_ccl_topology_labels.py`, `tests/ttnn/unit_tests/benchmarks/test_ccl_bandwidth.py`, the
  sweep `tests/sweep_framework/sweeps/ccl/generality/all_to_all.py`, and the DeepSeek MLA paths
  (`models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_alltoall_deepseek.py`).
- **`TT_FATAL` census, `all_to_all_async_program_factory.cpp`:** 12 → 11. Subject deleted:
  `TT_FATAL(mesh_dispatch_coordinate.has_value(), …)` (old `:155`) — it guarded the `std::optional<MeshCoordinate>`
  parameter of `create_descriptor`; `create_mesh_workload_artifacts` passes a concrete coordinate from `tensor_coords`.
  The other 11 are carried over verbatim. The ported-from override had no guard.
- **Dropped:** the dead sender writer output accessor (`output_tensor_addrgen`, old `interleaved_all_to_all_writer.cpp:94-95`)
  and its RTA; the debug-only `ASSERT(has_*_connection())` inside `FabricConnectionManager::get_*_connection()`, which went
  with the manager.
- **Kept as named args (dead in the kernels; see the audit's Misc anomalies):** sender writer `out_row_tiles`,
  `out_col_tiles`, `wait_output_semaphore`, `reset_global_semaphore`; sender reader `in_row_tiles`, `out_row_offset`,
  `out_col_offset`; receiver `in_row_tiles`, `out_row_tiles` and others; receiver CTAs `chunk_num_tiles`,
  `num_chunks_per_shard`, `packet_size_in_pages`. A later cleanup can drop them.
- **Behavior notes carried forward, not changed:** the receiver reader's hardcoded `2 * N_DRAM_BANKS`
  (`interleaved_all_to_all_receiver_reader.cpp`), and its NoC-0 accessor address used on the reader NoC.
- **Shared kernel touches:** none (all four kernels op-owned, converted in place).
- **Quasar uplift.** One DM self-loop DFB (`packet_header` on `sender_writer`) — target of the DM self-loop post-port
  pass. The hw configs use the arch-agnostic TTNN DM helpers.
- **Sibling ops.** `experimental/ccl` ops with a forward+backward `FabricConnectionManager` writer and a caller-supplied
  GlobalSemaphore refreshed by an override can follow this port.

## Verification

Host: 4x Blackhole p300b; multi-device meshes cannot open here, and the op needs a ring of at least 3 devices, so no
pytest runs on this host.

- **Build:** `bd` (`./build_metal.sh -b RelWithDebInfo --build-tests -e --enable-ccache`) — SUCCESS.
- **Legality checks:** not forced in `tt_metal/impl`. The only test that exercises the specs calls `MakeProgramFromSpec`,
  `SetProgramRunArgs` and `UpdateProgramRunArgs` directly with `skip_validation = false` (the default), so the checks ran.
  The TTNN adapter path (where TTNN sets `skip_validation`) cannot run here.
- **Sentinel, before the port** (`c116d1cba2f`, throw-away gtest on `ProgramSpecFabricTestGen1`, not committed): the legacy
  `ProgramDescriptor` (four kernels, legacy CTAs/RTAs, three CBs, `append_fabric_connection_rt_args` toward the other mock
  chip) for a ring of 4 — first device / in_dim 2 / both connections, second device / in_dim 3 / both, last device /
  forward only, third device / backward only — built into a `Program` and JIT-compiled: 4 passed, plus
  `CPU_FabricConnectionBindingJITSmoke`.
- **Sentinel, after the port** (same file name): the `ProgramSpec` the ported factory builds (same four
  KernelSpecs, three DFBs incl. the self-looped `packet_header`, three tensor parameters, the GlobalSemaphore parameter,
  the `forward` / `backward` fabric connection specs, named args) for the same four configs ran through
  `MakeProgramFromSpec` (validation on) and JIT-compiled — 4 passed, re-run in an empty JIT cache to confirm that all four
  kernels compile. A fifth case ran `SetProgramRunArgs` with every named RTA, the three tensors and the GlobalSemaphore,
  then `UpdateProgramRunArgs` with the override's run args (fresh tensors and semaphore) — passed. Plus
  `CPU_FabricConnectionBindingJITSmoke`: 6/6. This checks spec and run-args legality and kernel compilation only; it does
  not run the host factory or move data.
- **Self-audit** (op directory): 13 `.cpp`/`.hpp` files. No `Buffer*` / `emplace_runtime_args` in the
  factory; the one `address()` hit is `global_semaphore.address()` in the sender writer (override 4a). No
  `TensorAccessorArgs`, `get_compile_time_arg_val`, `get_arg_val`, `CircularBuffer` or `CBDescriptor`; `cb` name grep — 0
  hits; no `.md` citation in code; no `tt_metal/` file in the diff; `TT_FATAL` counts as above (device-op files
  unchanged); no `opt_level` (DM-only, legacy unset → O2 both sides); `hw_config` = reader/writer defaults as legacy
  (`ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` → `create_reader/writer_datamovement_config()`); no varargs;
  no multi-binding flag.
