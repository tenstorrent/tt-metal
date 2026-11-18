# Port Report — all_gather_async

## Outcome

**PORTED (subset)** — `LlamaShardedMeshWorkloadFactory` is on `MeshWorkloadSpecFactoryConcept` with a translated
`override_runtime_arguments`. Its two kernels are converted in place. `DefaultMeshWorkloadFactory` and
`AllGatherViaBroadcastFactory` stay on the `WorkloadDescriptor` path (audit: fabric mux and Device 2.0 for the default
factory; `RoutingPlaneConnectionManager` for via-broadcast). Multi-chip runtime behavior is not verified on this host (see
Verification).

| Factory | Status |
|---|---|
| `LlamaShardedMeshWorkloadFactory` | PORTED |
| `DefaultMeshWorkloadFactory` | BLOCKED — fabric mux has no Metal 2.0 form (multi-worker configs); Device 2.0 RED on its shared kernels; the one-worker remainder is not split out |
| `AllGatherViaBroadcastFactory` | not ported (out of queue scope); would be blocked on `RoutingPlaneConnectionManager` |

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
  (from the recipe branch ref `origin/akertesz/op-porting-recipe`; the recipe docs are not in this checkout)
- **Audit docs (inherited):** same line.
- **Worked examples:** `all_to_all_async` (`f83d662ac65..3e1814617f9`), `slice_reshard_async`
  (`37ea412ca55..96a29057d61`), `point_to_point` (`2e5f720ff76..c210a7093f6`) on this branch, incl. the branch overrides.

## TTNN ProgramFactory

- **Concept realized:** `MeshWorkloadSpecFactoryConcept`. `create_mesh_workload_artifacts` returns one `PerCoordProgram`
  per coordinate of `tensor_coords`, range = that coordinate — the same granularity as the legacy `WorkloadDescriptor`.
  `override_runtime_arguments(attrs, tensor_args, output_tensor, const MeshCoordinateRange&)` returns `ProgramRunArgs`.
- **Mixed variant.** `program_factory_t` keeps its three alternatives in the same order; only the llama alternative changed
  concept. The framework visits each alternative on its own (`dispatch_to_mesh_workload_factory`,
  `ttnn/api/ttnn/device_operation.hpp`), so no device-op change was needed.
- **Override translation.** The ported-from override wrote the writer's semaphore CRTA and barrier CRTA (0 when absent);
  the `Buffer*` CRTAs (input, output) were patched by the descriptor framework. The Metal 2.0 override returns
  `tensor_args` for `input` and `output` and `global_semaphore_args` for `semaphore` and, when the writer binds it,
  `barrier_semaphore`. No `kernel_run_args`. The miss and the hit use one helper (`make_dispatch_run_args`). The
  ported-from override read no coordinate; the new one ignores `range`.
- **`MeshWorkloadArtifacts::global_semaphores`:** empty (both semaphores are caller-owned).
- **Custom hash left intact:** `device/all_gather_async_device_operation.cpp:243`.
- **Device-op-class edits:** none. The factory header changed (it is the factory).
- **Pybind entry points removed:** none.

## Handoff points

- **Default factory (not ported).** Multi-worker configs add the `tt_metal` fabric mux (`tt_fabric_mux.cpp`), its
  program-local L1 placement and its client semaphores — none expressible in Metal 2.0. See the audit for the split
  decision and the recommended path.
- **Via-broadcast factory (not ported).** Its writer needs a `RoutingPlaneConnectionManager` built from `fabric::`
  bindings.

## Successes

- **`ttnn_factory.md` — per-coordinate programs and the cache-hit contract.** From a `WorkloadDescriptor` source the
  override names every io `TensorParameter` plus the two GlobalSemaphores.
- **`port_patterns.md` — Conditional / optional resource bindings.** The optional barrier semaphore became a conditional
  `GlobalSemaphoreParameter` + binding + `USE_BARRIER_SEM` define; the define replaced the `use_barrier_sem` CTA. The
  hash already keys on the condition, so it is stable per cache entry.
- **`port_patterns.md` — Self-loop DFB binding** for the packet-header DFB (writer only).
- **Branch overrides 4b/6/7** — `forward` / `backward` specs, senders built under `has_fabric_connection`, the
  manager's open/close phase order written by hand (as in `all_to_all_async`).

## Friction

- **Gap — no `TensorAccessor::get_bank_base_address` on this branch.** Both bindings are Case 2 (raw sharded base +
  explicit NoC coordinates). The recipe's bridge does not exist; the L1 bridge is `LocalTensorAccessor<uint8_t>` plus
  `local_mem::detail::AddressAccess::address(x.local_mem())` from the internal header `internal/local_mem/address_access.h`
  (precedent: the ported matmul DRAM-sharded readers). A public accessor for the base address would avoid the internal
  include.
- **Gap — per-node vararg count.** The core lists differ in length per worker. Metal 2.0 wants one
  `num_runtime_varargs` per kernel (the per-node override is deprecated), so every node's list is padded with zeros to the
  longest one in the program (`pad_varargs_to_common_length`). The kernels read only `2 * num_cores`, so behavior does
  not change. Named runtime array args (`RuntimeArgSchema::runtime_array_args`) also exist but have a fixed size too; the
  recipe does not mention them, so they were not used.
- **Friction — positional CTAs read by a shared helper.** `ccl_routing_utils::get_line_multicast_route_info_from_args<10>()`
  reads positional CTAs. The writer now builds the two `line_multicast_route_info_t` from twelve named CTAs
  (`forward_mcast_*`, `backward_mcast_*`). A named-arg overload in the shared header would serve every ported CCL writer.
- **Friction — donor helper takes `FabricConnectionManager&`.** `write_and_advance_local_read_address_for_fabric_write`
  (`ccl/common/kernels/minimal_ccl_common.hpp`) cannot be called once the manager is gone (and is Device 1.0). The
  `[Prereq]` replaced the call with a local copy that uses `Noc`; the port then swapped the manager for two senders.
- **Confusion — the recipe's override table vs. `WorkloadDescriptor`.** The recipe's Step 1 inventories what the
  ported-from override writes; for a descriptor source the tensor addresses are not in the override, they are framework
  `Buffer*` bindings. `ttnn_factory.md` covers it ("name them all"); the recipe's Step 3 example reads as if they must
  appear in the override.

## Open items for downstream

- **Not verified at runtime.** Only the mock-fabric sentinel ran here. The fabric connections, the GlobalSemaphore
  bindings and the cache-hit refresh run only in CI on a Galaxy: `tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py`
  (llama-sharded shapes, linear and ring, persistent output), and the llama TG model paths that call
  `ttnn.experimental.all_gather_async` with these shapes.
- **`TT_FATAL` census, `all_gather_async_llama_sharded_program_factory.cpp`:** 5 → 4. Subject deleted: the kernel-index
  check (`reader_kernel_idx == kLlamaShardedReaderKernelIdx && …`, old `:199-205`). It guarded the index-based legacy
  override, which no longer exists. The other four are carried over verbatim. The ported-from override had no guard.
- **Dropped with the positional plumbing:** the `log_trace` loops over the positional CTA/RTA vectors (old `:153-156,
  182-185, 281-284, 308-311`); the kernel-index constants; the legacy `semaphore` / `barrier_semaphore` /
  `using_persistent_buffers` locals (now read in `uses_barrier_semaphore` and `make_dispatch_run_args`); the writer's
  unused `arg_for_fab` / `connect_to_fabric_when_creating` locals; the debug-only `ASSERT`s inside
  `FabricConnectionManager::get_*_connection()`.
- **Kept as named args (unused in the kernels):** `my_chip_id` (reader and writer CTAs).
- **Behavior notes carried forward, not changed:** the `cores_per_device` slice of `output_cores_vec` reads past its end
  for some ring indices (old `:212-221`, kept verbatim); the barrier and drain-core semaphores live on the worker cores,
  so the caller's GlobalSemaphores must span them (`use_optimal_ccl_for_llama` uses a fixed placement).
- **Shared kernel touches:** none. `ccl_runtime_args.hpp` keeps the now-unused `LlamaGather*CommonArgs` structs.
- **2-device ring:** the validator rejects the repeated `(src, dst, link)` when the forward and backward neighbors are the
  same device. Not handled (audit edge case).
- **Quasar uplift.** One DM self-loop DFB (`packet_header` on `writer`) — target of the DM self-loop post-port pass. The hw
  configs use the arch-agnostic TTNN DM helpers.

## Verification

Host: 4x Blackhole p300b; multi-device meshes cannot open here, so no pytest runs on this host.

- **Build:** `bd` (`./build_metal.sh -b RelWithDebInfo --build-tests -e --enable-ccache`) — SUCCESS.
- **Legality checks:** not forced in `tt_metal/impl`. The sentinel calls `MakeProgramFromSpec`, `SetProgramRunArgs` and
  `UpdateProgramRunArgs` directly with `skip_validation = false` (the default), so the checks ran. The TTNN adapter path
  cannot run here.
- **Sentinel, before the port** (`6420f30222a`, throw-away gtest on `ProgramSpecFabricTestGen1`, not committed): the
  legacy llama-sharded `ProgramDescriptor` (two kernels, legacy CTAs/RTAs/CRTAs, two CBs,
  `append_fabric_connection_rt_args` toward the other mock chip) for a line of 4 — first device / forward only / barrier /
  1 link / mult+silu shape; second device / both / no barrier / 2 links / post-SDPA shape; last device / backward only /
  barrier / 2 links / mult+silu shape — built into a `Program` and JIT-compiled: 3 passed, plus
  `CPU_FabricConnectionBindingJITSmoke` (4/4).
- **Sentinel, after the `[Prereq]`:** same legacy cases in an empty JIT cache: 4/4.
- **Sentinel, after the port:** the `ProgramSpec` the ported factory builds (same two KernelSpecs, two DFBs incl. the
  self-looped `packet_header`, two tensor parameters, the GlobalSemaphore parameters incl. the conditional barrier, the
  `forward` / `backward` specs, named args, padded varargs) for the same three configs ran through `MakeProgramFromSpec`
  (validation on) and JIT-compiled in an empty JIT cache — 3 passed. A fourth case ran `SetProgramRunArgs` with every named
  RTA, the varargs, both tensors and both GlobalSemaphores, then `UpdateProgramRunArgs` with the override's run args
  (fresh tensors and semaphores) — passed. Plus `CPU_FabricConnectionBindingJITSmoke`: 5/5. This checks spec and run-args
  legality and kernel compilation only; it does not run the host factory or move data.
- **Self-audit** (op directory, 20 `.cpp`/`.hpp` files; llama factory + its two kernels): no `Buffer*` /
  `emplace_runtime_args` / `->address()` in the llama factory; the three `address()` hits in the writer are
  `GlobalSemaphore::address()` (override 4a). No `TensorAccessorArgs`, `get_compile_time_arg_val`, `get_arg_val`,
  `get_common_arg_val`, `CircularBuffer` or `CBDescriptor` in the llama files; no `cb`-named DFB locals or specs; no `.md`
  citation in code; no `tt_metal/` file in the diff; `TT_FATAL` counts as above; no `opt_level` (DM-only, legacy unset →
  O2 both sides); `hw_config` = reader/writer defaults as legacy (`ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`
  → `create_reader/writer_datamovement_config()`); varargs only for the core lists; no multi-binding flag. The default
  and via-broadcast factories and their kernels are unchanged.
