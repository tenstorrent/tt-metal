# Port Report — all_to_all_dispatch

## Outcome

**PORTED** — `AllToAllDispatchSparse` (the only factory) is on `MeshWorkloadSpecFactoryConcept`, in both modes
(`FullPacket`, `PageByPage`). Both kernels are op-owned and converted in place. Multi-chip runtime behavior is not verified
on this host (see Verification).

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
  (from `git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`; the
  recipe docs are not in this checkout)
- **Audit docs (inherited):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
- **Worked examples:** `point_to_point` (`2e5f720ff76..c210a7093f6`), `slice_reshard_async` (`37ea412ca55..96a29057d61`)
  and the sibling `all_to_all_combine` (`377d1c00652..5f28f745c0c`) on this branch, incl. the branch-specific overrides.

## TTNN ProgramFactory

- **Concept realized:** `MeshWorkloadSpecFactoryConcept`. `create_mesh_workload_artifacts`
  (`device/all_to_all_dispatch_program_factory.cpp:687`) returns one `PerCoordProgram` per coordinate of `tensor_coords`,
  range = that coordinate (same granularity as the `WorkloadDescriptor`). No override, as in the ported-from factory: on a
  cache hit the framework refreshes the five tensor bindings per range.
- **Op-owned GlobalSemaphores:** both are created and `Synchronize` runs once per cache miss, as before; they now live in
  `MeshWorkloadArtifacts::global_semaphores` (two slots reserved before the pushes, so the references the run args hold stay
  valid). Without an override they keep their cache-miss value on a hit, which matches legacy (their raw addresses sat in
  the cached RTAs, and nothing refreshed them).
- **Custom hash:** none.
- **Device-op-class edits:** the factory struct nested in `all_to_all_dispatch_device_operation.hpp` changed its method
  (`create_workload_descriptor` → `create_mesh_workload_artifacts`), its comment and its includes. Nothing else in the
  device-op files.
- **Pybind entry points removed:** none.

## Handoff points

none — no capitulation, no boundary violation, no removed pybind surface.

## Successes

- **`port_patterns.md` — Conditional / optional resource bindings, "Promote a CTA gate to a define".** The intermediate
  metadata CB exists only in `FullPacket`, and both kernels selected its code with the CTA `write_page_by_page` and
  `if constexpr`. The entry predicted the failure (name lookup of `dfb::metadata_buffer` in the discarded branch) and the
  fix: the `metadata_buffer` DFB, its two bindings and a `FULL_PACKET` define on **both** kernels share one condition
  (`.cpp:320`, `:350`, `:364`, `:424`); the kernels use `#ifndef FULL_PACKET … #else … #endif` in place of the
  `if constexpr`, in the same order as before, with the comments kept.
- **`metal2_audit.md` — CB endpoints census.** Two CBs had no FIFO call on one side or at all (`c_3` packet header: only a
  `get_read_ptr`; `c_5`: only `get_write_ptr` in the reader and `get_read_ptr` in the writer). The census gave self-loop
  for the one-toucher `c_3` and 1P+1C for the two role-free touchers of `c_5`, with no multi-binding flag.
- **`metal2_port.md` — Hardware configuration, "Custom".** Both legacy DM configs are custom (`RISCV_1`/`NOC_1` reader,
  `RISCV_0`/`NOC_0` writer) and are copied verbatim into `DataMovement1XXConfig` (`.cpp:483`, `:551`).
- **`ttnn_factory.md` — Extracting the tensor.** The legacy factory copied the three input tensors
  (`auto input_tensor = tensor_args.input_tensor;`). A `TensorArgument` must reference the io tensor, so the three locals
  are now `const auto&` (`.cpp:155-157`).

## Friction

- **Gap — several fabric connections per node.** As in `all_to_all_combine`: the moe_utils helper
  `open_direction_connections_async` reads RTAs (`build_from_args`) and `moe_utils.hpp` is shared. The writer builds and
  opens the four senders from an array of `FabricConnectionBindingToken`s. Unlike combine, the writer keeps calling the
  moe_utils `open_direction_connections_barrier` and `close_direction_connections`: they take a runtime
  `std::array<bool, 4>` (now built from `has_fabric_connection`) and read no RTA, so only the open loop changes.
- **Gap — GlobalSemaphore increment.** The kernel `experimental::GlobalSemaphore` has no increment. The writer's local
  self-increments stay `noc_semaphore_inc(get_noc_addr(gs.address()), 1)` (three sites).
- **Gap — `DataMovementHardwareConfig` shape.** The recipe names `DataMovementGen1Config{…}`; on this branch the type is
  `DataMovementHardwareConfig{.config_1xx = DataMovement1XXConfig{…}}` with an optional `config_2xx`.
- **Gap — `noc_async_read_page` in the Device 2.0 prereq.** `noc_async_read_page` takes the transfer size from the accessor
  (`get_aligned_page_size()`); `Noc::async_read` needs it explicitly. The prereq passes
  `<accessor>.get_aligned_page_size()` so the size is the same by construction. The migration guide does not say where the
  size of a page read comes from.
- **Gap — `Noc::src_args_t` is private.** A helper that forwards `Noc::async_write` arguments for any endpoint type has to
  spell `typename noc_traits_t<T>::src_args_type`; the `Noc` aliases `src_args_t` / `dst_args_t` are private.
- **Confusion — shared positional CTA vector.** The legacy factory passes one 37-entry CTA vector to both kernels. The
  recipe's per-`KernelDescriptor` inventory assumes one CTA list per kernel; the port gives each kernel exactly the CTAs it
  declares (reader 15, writer 30), including declared-but-unused ones, so the kernels need no further edits.
- **Confusion — `cb` names that stay.** The self-audit `cb` grep hits `detail::get_cb_sizes` and its locals
  (`.cpp:49-93`) and the caller in `all_to_all_dispatch.cpp:39`. The function sizes the buffers for the op's L1 budget,
  is declared in the device-op header and is called outside the factory, so a rename is out of scope. The factory's own
  locals are `dfb_sizes` / `dfb_page_sizes`.
- **Confusion — Device 2.0 prereq in the queue.** As for `point_to_point` and `all_to_all_combine`: the audit recipe routes
  a Device 2.0 RED to the Device 2.0 team; the launching session asked for the fix as a `[Prereq]` commit (`94bf191ce8f`).

## Open items for downstream

- **Not verified at runtime.** Only the mock-fabric sentinel ran here (spec validation, run-args validation, JIT compile).
  The fabric connections, the GlobalSemaphore CRTAs, the cross-device metadata writes and the cache-hit refresh run only in
  CI: `tests/nightly/t3000/ccl/test_all_to_all_dispatch.py`, `tests/nightly/t3000/ccl/test_moe_ccl_end_to_end.py`,
  `tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py`, the Blackhole box / galaxy / exabox suites under
  `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/` (`box/all_post_commit/test_all_to_all_dispatch_bh.py` among them),
  `tests/ttnn/multidevice_perf_tests/test_all_to_all_dispatch_perf.py`, `tests/ttnn/docs_examples/test_ccl_examples.py`
  and the sweep `tests/sweep_framework/sweeps/ccl/generality/all_to_all_dispatch.py`.
- **Cross-device `metadata_buffer` address.** In `FullPacket` every writer writes into the `metadata_buffer` DFB of the
  other devices at its own local DFB address (`writer_all_to_all_dispatch.cpp`, `intermediate_metadata_write_addr`). This
  needs the same L1 placement on every coordinate. The port keeps one DFB set for every coordinate (as the legacy CB set),
  but Metal 2.0 does not state that equal specs place DFBs at equal addresses across devices. Worth a CI check.
- **Shared kernel touches:** none (both kernels op-owned, converted in place).
- **2-device Ring along the dispatch axis.** East and West are the same neighbor, so the `east` and `west` specs both carry
  `(src, neighbor, link i)` on core `i`, and `ValidateFabricConnectionUsage` rejects the program. Legacy built two senders on
  one core to one router (suspect in legacy too). Config edge case; not changed.
- **Unused tensor addresses and CTAs** (legacy plumbing, see the audit's Misc anomalies): the reader no longer receives the
  output address and the writer no longer receives the input / indices / mapping addresses; both still receive CTAs they
  declare but never use. A later cleanup can drop the unused declarations.
- **`TT_FATAL` census:** factory count unchanged (2 → 2). No guard lost.
- **Quasar uplift.** Two DM self-loop DFBs (`packet_header`, `send_preparation_buffer` on the writer) — targets of the DM
  self-loop post-port pass; `metadata_buffer` has no credit calls — target of the sync-free pass. Both DM configs are
  Gen1-only (`config_2xx` not set) — target of the Gen2 post-port pass.

## Verification

Host: 4x Blackhole p300b; multi-device meshes cannot open here, and on one device `get_neighbors` `TT_FATAL`s ("No
neighbors found"), so no pytest runs on this host.

- **Build:** `./build_metal.sh -b RelWithDebInfo --build-tests -e --enable-ccache` (`bd`) — SUCCESS.
- **Legality checks:** not forced in `tt_metal/impl`. The sentinel calls `MakeProgramFromSpec`, `SetProgramRunArgs` and
  `UpdateProgramRunArgs` directly with `skip_validation = false` (the default), so the checks ran. The TTNN adapter path
  cannot run here.
- **Sentinel, before the port** (throw-away gtest on `ProgramSpecFabricTestGen1`, not committed): the legacy
  `ProgramDescriptor` (same kernels, CTAs, RTAs, defines, CBs, `append_fabric_connection_rt_args` toward the fixture's other
  chip) for five configs — Linear 1x2 axis 1 first device FullPacket (East, 2 links), Linear 2x1 axis 0 second device
  PageByPage (North), Ring 1x4 axis 1 second device FullPacket and PageByPage (East + West), Linear 1x2 no axis second
  device PageByPage (West) — built into a `Program` and JIT-compiled: 5 passed on `dbf8fd3bd30` (= `5f28f745c0c` code) and
  again on `94bf191ce8f`, plus `CPU_FabricConnectionBindingJITSmoke`.
- **Sentinel, after the port:** the `ProgramSpec` the ported factory builds for the same five configs (the first slot whose
  neighbor is the fixture's other chip gets the targets; the others have none), through `MakeProgramFromSpec` and JIT
  compile: 5 passed (FullPacket variants compiled with the `metadata_buffer` binding and `FULL_PACKET`). Two more cases ran
  `SetProgramRunArgs` (named RTAs, five tensors, two GlobalSemaphores) and then `UpdateProgramRunArgs` with fresh tensors
  only (the framework's no-override cache-hit refresh), one per mode: passed. Plus `CPU_FabricConnectionBindingJITSmoke`.
  8 passed. This checks spec and run-args legality and kernel compilation only; it does not run the host factory or move
  data.
- **Self-audit** (9 `.cpp`/`.hpp` files in the op directory): no `Buffer*` / `buffer()` / `emplace_runtime_args` in the
  factory (the three `mesh_buffer()` reads are page-size / page-count queries in `detail::`); no `TensorAccessorArgs`,
  `get_compile_time_arg_val`, `get_arg_val`, `CircularBuffer`, `CBDescriptor`, `ProgramDescriptor`,
  `WorkloadDescriptor` (0 hits / 9 files); `cb` name grep — only `get_cb_sizes` and its locals (see Friction); no `.md`
  citation in code; no `tt_metal/` file in the diff; `TT_FATAL` counts unchanged; no `opt_level` (DM-only, legacy unset →
  O2 both sides); `hw_config` = legacy custom triples; no varargs; no multi-binding flag; the conditional `metadata_buffer`
  follows the conditional-binding pattern (host binding + define on both kernels + `#ifdef` in both kernels).
