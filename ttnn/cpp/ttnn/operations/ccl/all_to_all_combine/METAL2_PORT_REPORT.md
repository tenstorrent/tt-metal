# Port Report — all_to_all_combine

## Outcome

**PORTED** — `AllToAllCombineFromSparse` (the only factory) is on `MeshWorkloadSpecFactoryConcept`. The reader is converted
through a new `_metal2` fork (it is lent to `moe_expert_token_remap`); the writer is converted in place. Multi-chip runtime
behavior is not verified on this host (see Verification).

## Provenance

- **Recipe docs (this port):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
  (from `git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`; the
  recipe docs are not in this checkout)
- **Audit docs (inherited):** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`
- **Worked examples:** `point_to_point` (`2e5f720ff76..c210a7093f6`) and `slice_reshard_async` (`37ea412ca55..96a29057d61`)
  on this branch, incl. the branch-specific overrides.

## TTNN ProgramFactory

- **Concept realized:** `MeshWorkloadSpecFactoryConcept`. `create_mesh_workload_artifacts` returns one `PerCoordProgram` per
  coordinate of `tensor_coords`, range = that coordinate (same granularity as the `WorkloadDescriptor`). No override, as in
  the ported-from factory: on a cache hit the framework refreshes the four tensor bindings per range.
- **Op-owned GlobalSemaphores:** both are created and `Synchronize` runs once per cache miss, as before; they now live in
  `MeshWorkloadArtifacts::global_semaphores` (two slots reserved before the pushes, so the references the run args hold stay
  valid). Each program's run args bind them through `global_semaphore_args`. Without an override they keep their cache-miss
  value on a hit, which matches legacy (their raw addresses sat in the cached RTAs, and nothing refreshed them).
- **Custom hash:** none.
- **Device-op-class edits:** the factory struct nested in `all_to_all_combine_device_operation.hpp` changed its method
  (`create_workload_descriptor` → `create_mesh_workload_artifacts`) and its includes. Nothing else in the device-op files.
- **Pybind entry points removed:** none.

## Handoff points

none — no capitulation, no boundary violation, no removed pybind surface.

## Successes

- **`port_patterns.md` — Caution: Porting a shared kernel, "Lent".** The reader sits in this op's directory, so in-place
  conversion looked safe; the census (`grep -rl reader_all_to_all_combine ttnn/cpp/ttnn/operations/`) found
  `moe_expert_token_remap_program_factory.cpp:176`. Rung 2: fork beside the original plus the pointer comment.
- **`metal2_port.md` — Hardware configuration, "Custom".** Both legacy DM configs are custom (`RISCV_1`/`NOC_1` reader,
  `RISCV_0`/`NOC_0` writer) and differ from both TTNN helpers. They are copied verbatim into `DataMovement1XXConfig`.
- **Self-audit `cb` sweep.** Caught a "scratch CB" comment in the writer (now "scratch DFB").

## Friction

- **Gap — several fabric connections per node.** Branch override 7 settles the spec side (one `FabricConnectionSpec` per
  direction). Kernel side, the moe_utils helpers `open_direction_connections_async` / `_barrier` /
  `close_direction_connections` take RTA indices (`build_from_args`), and `moe_utils.hpp` is shared and may not be edited.
  The writer now opens and closes the four senders itself from an array of `FabricConnectionBindingToken`s (three short
  loops). A binding-token overload of those helpers in a shared place would remove that duplication for every MoE CCL port.
- **Gap — GlobalSemaphore increment.** The kernel `experimental::GlobalSemaphore` has `wait`, `wait_min`, `set`, `value`,
  `address`, but no increment. The writer's local self-increment stays `noc_semaphore_inc(get_noc_addr(gs.address()), 1)`.
- **Gap — `DataMovementHardwareConfig` shape.** The recipe names `DataMovementGen1Config{…}`; on this branch the type is
  `DataMovementHardwareConfig{.config_1xx = DataMovement1XXConfig{…}}` with an optional `config_2xx`.
- **Confusion — `DIRECTIONS` define.** Override 4b names "the per-direction flags" as part of the replaced fabric RTAs; here
  the flags were a define, not RTAs. The define is dropped: each flag is now `has_fabric_connection` of its binding, which is
  true on exactly the same nodes (a target exists iff `get_neighbors` found a neighbor in that direction).
- **Confusion — Device 2.0 prereq in the queue.** The audit recipe routes a Device 2.0 RED to the Device 2.0 team; the
  launching session asked for the fix as a `[Prereq]` commit (`66b645994d5`), as `point_to_point` did.

## Open items for downstream

- **Not verified at runtime.** Only the mock-fabric sentinel ran here (spec validation, run-args validation, JIT compile).
  The fabric connections, the GlobalSemaphore CRTAs and the cache-hit refresh run only in CI: the MoE CCL tests that drive
  `ttnn.all_to_all_combine`: `tests/nightly/t3000/ccl/test_all_to_all_combine.py`,
  `tests/nightly/tg/ccl/test_all_to_all_combine_6U.py`, the Blackhole box / galaxy / exabox suites under
  `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/` (`all_post_commit/test_all_to_all_combine_bh.py` among them),
  `tests/ttnn/multidevice_perf_tests/test_all_to_all_combine_perf.py` and the sweep
  `tests/sweep_framework/sweeps/ccl/generality/all_to_all_combine.py`.
- **Shared kernel touch — `device/kernels/dataflow/reader_all_to_all_combine.cpp`:** rung 2, **created the fork**
  `device/kernels/dataflow/reader_all_to_all_combine_metal2.cpp`; pointer comment added to the original. Remaining
  unmigrated consumer: `ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap`. Its factory passes
  `ReaderConfigDescriptor{}` and its own CB set; when it ports, it must adopt the fork's binding names (`dfb::mapping`,
  `dfb::local_experts`, `dfb::metadata`, `dfb::data`, `tensor::data` / `mapping` / `metadata`) and named args.
- **Lint allow-list entry now stale:** `scripts/check_kernel_cb_ptrs.py:52` lists the writer as a reviewed exception to its
  `get_read_ptr(cb)` rule; the writer has no free CB calls any more. Harmless; outside the op directory.
- **2-device Ring along the dispatch axis.** East and West are the same neighbor, so the `east` and `west` specs both carry
  `(src, neighbor, link i)` on core `i`, and `ValidateFabricConnectionUsage` rejects the program. Legacy built two senders on
  one core to one router (suspect in legacy too). Config edge case; not changed.
- **`TT_FATAL` census:** factory count unchanged (2 → 2). No guard lost.
- **Quasar uplift.** Two DM self-loop DFBs (`mapping` on the reader, `packet_header` on the writer) — targets of the DM
  self-loop post-port pass. Both DM configs are Gen1-only (`config_2xx` not set) — target of the Gen2 post-port pass.

## Verification

Host: 4x Blackhole p300b; multi-device meshes cannot open here, and on one device `get_neighbors` `TT_FATAL`s ("No
neighbors found"), so no pytest runs on this host.

- **Build:** `./build_metal.sh -b RelWithDebInfo --build-tests -e --enable-ccache` (`bd`) — SUCCESS.
- **Legality checks:** not forced in `tt_metal/impl`. The sentinel calls `MakeProgramFromSpec`, `SetProgramRunArgs` and
  `UpdateProgramRunArgs` directly with `skip_validation = false` (the default), so the checks ran. The TTNN adapter path
  cannot run here.
- **Sentinel, before the port** (throw-away gtest on `ProgramSpecFabricTestGen1`, not committed): the legacy
  `ProgramDescriptor` (same kernels, CTAs, RTAs, defines, CBs, `append_fabric_connection_rt_args` toward the fixture's other
  chip) for three configs — Linear 1x2 axis 1 first device (East, 2 links), Linear 2x1 axis 0 second device (North, locally
  reduced), Ring 1x4 axis 1 second device (East + West) — built into a `Program` and JIT-compiled: 3 passed on `377d1c00652`
  and again on `66b645994d5`, plus `CPU_FabricConnectionBindingJITSmoke`.
- **Sentinel, after the port:** the `ProgramSpec` the ported factory builds for the same three configs (the slot whose
  neighbor is the fixture's other chip gets the target; the others have none) through `MakeProgramFromSpec` and JIT compile:
  3 passed. A fourth case ran `SetProgramRunArgs` (named RTAs, four tensors, two GlobalSemaphores) and then
  `UpdateProgramRunArgs` with fresh tensors only (the framework's no-override cache-hit refresh): passed. Plus
  `CPU_FabricConnectionBindingJITSmoke`. This checks spec and run-args legality and kernel compilation only; it does not run
  the host factory or move data.
- **Self-audit** (10 `.cpp`/`.hpp` files in the op directory; the legacy reader original is excluded where noted): no
  `Buffer*` / `buffer()` / `emplace_runtime_args` in the factory (one `address()` left: the legacy `log_debug` of the
  semaphore address); no `TensorAccessorArgs`, `get_compile_time_arg_val`, `get_arg_val`, `CircularBuffer`, `CBDescriptor`,
  `ProgramDescriptor` outside the legacy reader; `cb` name grep — 0 hits outside the legacy reader; no `.md` citation in
  code; no `tt_metal/` file in the diff; `TT_FATAL` counts unchanged; no `opt_level` (DM-only, legacy unset → O2 both
  sides); `hw_config` = legacy custom triples; no varargs; no multi-binding flag.
