# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async`

**Audited at `6420f30222a`.** One `[Prereq]` commit (Device 2.0, llama-sharded writer) clears the only red gate of the
llama-sharded factory. The default factory and the via-broadcast factory are not ported (see Result).

- **`AllGatherAsyncDeviceOperation`** (`device/all_gather_async_device_operation.{hpp,cpp}`). One device operation, three
  factories, one porting unit (shared attributes, shared hash, shared `select_version`):
  - `program_factory_t = std::variant<DefaultMeshWorkloadFactory, LlamaShardedMeshWorkloadFactory,
    AllGatherViaBroadcastFactory>` (`all_gather_async_device_operation.hpp:22-23`). `select_program_factory`
    (`all_gather_async_device_operation.cpp:111-127`) maps `select_version` (`:94-109`): `use_all_gather_async_llama_sharded`
    → llama-sharded; else `use_all_gather_async_via_broadcast` → via-broadcast; else default (`MINIMAL_DEFAULT`).
  - `DefaultMeshWorkloadFactory` (`device/all_gather_async_default_program_factory.{hpp,cpp}`) —
    `create_workload_descriptor` (`.cpp:759`) builds one `ProgramDescriptor` per coordinate with the public builder
    `ttnn::build_all_gather_async_minimal_default_program_descriptor` (`.cpp:97`); `override_runtime_arguments(Program&,
    …)` (`.cpp:817`).
  - `LlamaShardedMeshWorkloadFactory` (`device/all_gather_async_llama_sharded_program_factory.{hpp,cpp}`) —
    `create_workload_descriptor` (`.cpp:346`), one `ProgramDescriptor` per coordinate (`build_llama_sharded_program_descriptor`,
    `.cpp:31`); `override_runtime_arguments(Program&, …)` (`.cpp:361`).
  - `AllGatherViaBroadcastFactory` (`device/all_gather_via_broadcast_factory.{hpp,cpp}`) — `create_workload_descriptor`
    (`.cpp:282`), one `ProgramDescriptor` per coordinate (`build_descriptor_at`, `.cpp:52`); no override.
- Kernels (all in `device/kernels/`, all referenced):
  - `minimal_default_reader.cpp`, `minimal_default_writer.cpp` (+ `all_gather_worker_split.hpp`) — default factory. Also
    instantiated by `experimental/ccl/all_gather_matmul_async` (through the public builder,
    `all_gather_matmul_async_program_factory.cpp:118`) and by `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:1493`.
  - `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp` — fabric mux kernel, default factory with more than one worker per
    direction (`all_gather_async_default_program_factory.cpp:537`). Owned by `tt_metal/fabric`; used by many CCL ops.
  - `llama_shapes_sharded_reader.cpp`, `llama_shapes_sharded_writer.cpp` — llama-sharded factory. Census:
    `grep -rn llama_shapes_sharded ttnn/ tests/ models/ tt_metal/` hits only this factory.
  - `broadcast_rm_reader.cpp`, `broadcast_rm_writer.cpp` — via-broadcast factory; only this factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`. The queue asked for the default and
the llama-sharded factories; the via-broadcast factory is audited because it shares the device operation.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` — from
the recipe branch ref `origin/akertesz/op-porting-recipe`. The recipe docs are not in this checkout.

**Branch-specific overrides applied** (this branch is newer than the recipe; given by the launching session):

1. GlobalSemaphore is supported. The Appendix A "GlobalSemaphore — UNSUPPORTED" entry is not applied.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores`. Llama-sharded: none. Via-broadcast owns
   two (`all_gather_via_broadcast_factory.cpp:298-305`), not ported.
3. Fabric connections replace `append_fabric_connection_rt_args`.
4. Sanctioned kernel-side exception: (a) raw GlobalSemaphore address RTAs and their raw-pointer uses → `global_sem::`
   bindings; (b) the fabric-connection RTAs read by `FabricConnectionManager::build_from_args` (incl. the per-direction
   flags) → `fabric::` bindings.
5. Caller-supplied GlobalSemaphores: **applies to llama-sharded and default.** `semaphore` (vector) and the optional
   `barrier_semaphore` are in `AllGatherAsyncParams` (`all_gather_async_device_operation_types.hpp:27, 32`), left out of
   the hash (`attribute_names` has only `barrier_semaphore_present`, `:83-100`), and re-applied by the legacy overrides.
6. A binder node with no fabric target: **applies** (an end device of a linear line has one neighbor).
7. One node that talks to more than one destination: **applies to llama-sharded** (each worker opens a forward and a
   backward connection on its link → `forward` and `backward` specs).
8. (Queue rule 8, router shared by several cores of one program) — checked per factory below; it fires for no factory.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async` |
| **Overall** | **RED at op level; subset `LlamaShardedMeshWorkloadFactory` is clear** (after one `[Prereq]`). |
| **DOps / Factories** | `AllGatherAsyncDeviceOperation` → `DefaultMeshWorkloadFactory` · `LlamaShardedMeshWorkloadFactory` · `AllGatherViaBroadcastFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Llama-sharded: **No → Yes after `[Prereq]`** (isolated: the writer calls the Device 1.0 donor helper `write_and_advance_local_read_address_for_fabric_write`). Default: **No** (raw `noc_async_read/write` in the common path, `ShardedAddrGen` in sharded configs, `get_semaphore` in the mux path; kernels shared with `all_gather_matmul_async`). Via-broadcast: Yes except the override-4a sites. |
| *Prereqs* — Cross-op escapes | Llama-sharded: `⚠ workable` (the donor helper takes `FabricConnectionManager&`; replaced by a local copy in the `[Prereq]`; `worker_routing_utils.hpp` reads positional CTAs). |
| *Feature Support* — overall | Llama-sharded: GREEN. Default: **RED (config-scoped)** — fabric mux (more than one worker per direction) has no Metal 2.0 form. Via-broadcast: **RED (kernel API gap)** — `RoutingPlaneConnectionManager` cannot be built from `fabric::` bindings. |
| *Feature Support* — Variadic-CTA | Llama-sharded: none (the route-info CTA reads are at constexpr offsets) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Not read — sheet not reachable.** Derived from code: no blocking signal for any factory. |
| *TTNN Readiness* — Concept (current) | All three: `WorkloadDescriptor` (`create_workload_descriptor`); default and llama-sharded add the per-program `override_runtime_arguments(Program&, …, optional<MeshCoordinate>)` |
| *TTNN Readiness* — Execution Model | `MPMD` for all three — the ring index (CTA of every kernel), the forward/backward target counts and the route CTAs depend on the coordinate. `Secretly SPMD Workload?` = no. |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate): `all_gather_async_device_operation.cpp:243-268`. Leaves out `semaphore` and the barrier address; for `MINIMAL_DEFAULT` it hashes specs without the row count and adds the resolved worker count. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | Default: yes (`all_gather_async_default_program_factory.cpp:817`). Llama-sharded: yes (`all_gather_async_llama_sharded_program_factory.cpp:361`). Via-broadcast: no. |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`all_gather_async_nanobind.cpp` binds only the op) |
| *TTNN Readiness* — Op-owned tensors | No (no factory sets `WorkloadDescriptor::buffers`). Op-owned GlobalSemaphores: via-broadcast only (`WorkloadDescriptor::semaphores`, `all_gather_via_broadcast_factory.cpp:304-305`). |
| *TTNN Readiness* — Target concept | Llama-sharded: `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments` (code-derived). |
| *Port work* — Offset base pointer | **none** (GREEN), all three factories |
| *Port work* — Tensor bindings (per binding) | Llama-sharded: `input` (reader) and `output` (writer) are **Case 2** (raw base + shard offset with explicit NoC coordinates). |
| *TTNN Readiness* — TensorParameter relaxation | Not read — sheet not reachable. Code shows no need for llama-sharded (the hash keys on both full specs). |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in any kernel passes a 3rd argument |
| *Port work* — CB endpoints | Llama-sharded: data CB `c_in0` 1P+1C (reader → writer); packet-header CB `c_in1` self-loop (writer, DM). |

## Result

**RED at op level; subset `LlamaShardedMeshWorkloadFactory` is clear.** Per factory:

| Factory | Status | Reason |
|---|---|---|
| `LlamaShardedMeshWorkloadFactory` | **PORT** (after the Device 2.0 `[Prereq]`) | Every gate clears. Rule 8 does not fire: one worker per link, each with at most one forward and one backward target on its own link. |
| `DefaultMeshWorkloadFactory` | **BLOCKED, stays on the `WorkloadDescriptor` path** | (1) Missing M2 feature, config-scoped: every config with more than one worker per direction runs a fabric mux kernel, which Metal 2.0 cannot express. That is the default for any tensor that moves more than 4 KB per link. (2) Device 2.0 RED on both kernels, which `all_gather_matmul_async` also runs. (3) The clean remainder (one worker per direction, interleaved, not fused) is not ported: see "Why the default factory is not split" below. |
| `AllGatherViaBroadcastFactory` | **Not ported (out of queue scope), stays on the `WorkloadDescriptor` path** | Would be blocked: its writer drives the linear fabric API through `RoutingPlaneConnectionManager`, which has no way to be built from `fabric::` bindings. |

The framework dispatches each `program_factory_t` alternative on its own (`dispatch_to_mesh_workload_factory`,
`ttnn/api/ttnn/device_operation.hpp:266-303`; `AllFactoriesValid` checks each alternative on its own,
`ttnn/api/ttnn/operation_concepts.hpp:193-209`), so a `MeshWorkloadSpecFactoryConcept` alternative next to two
`ProgramDescriptorFactoryConcept` alternatives needs no framework change. The cache stores the factory index with the
program, and the hash already depends on `use_all_gather_async_llama_sharded`, so a cache hit never crosses factories.

### Why the default factory is not split

A config-scoped port of the default factory (one worker per direction, no mux, interleaved input and output) is possible
in principle: it passes rule 8 and needs no missing feature. It is not done here, for these reasons:

- **Narrow config.** `default_workers` (`all_gather_async_default_program_factory.cpp:42-93`) picks one worker per
  direction only when the data per link is at most 4 KB (`SINGLE_PACKET_THRESHOLD`, `:69, 72-73`) or when the caller
  passes `num_workers_per_link = 1`. Any larger tensor gets 2 or 4 workers and a mux (`:70-76`, `:141-162`).
- **Two implementations of one factory.** The split needs a second variant alternative chosen by the resolved worker
  count, a second (Metal 2.0) program builder, and `_metal2` forks of `minimal_default_reader.cpp` (426 lines) and
  `minimal_default_writer.cpp` (726 lines). The descriptor builder and the original kernels must stay for the mux configs
  and for `all_gather_matmul_async`, which appends the all-gather to its own matmul `ProgramDescriptor` with an
  `AllGatherFusedOpSignaler` (semaphore ids in RTAs). Two divergent copies of the same algorithm, selected by a
  hyperparameter, is a maintenance cost that the narrow config does not pay for.
- **A Device 2.0 prerequisite on shared kernels first.** Both default kernels keep raw `noc_async_read/write` in the
  common path (below). The fix touches kernels that `all_gather_matmul_async` and `test_generic_op` run.

Recommended path: port the whole default factory at once when Metal 2.0 can express a fabric mux, after a Device 2.0
migration of the two kernels (incl. the sharded path off `ShardedAddrGen`).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Not read.** No tool in this session can fetch the readiness sheet.
  Values from the code: all three factories define `create_workload_descriptor` → `WorkloadDescriptor` concept
  (`ProgramDescriptorFactoryConcept`); default and llama-sharded add the per-program `override_runtime_arguments`
  that the descriptor adapter probes. `Custom hash`: yes. `get_dynamic_runtime_args`: none. `Pybind descriptor`: none.
  `Op-owned tensors?`: no. Smuggled pointers: none — every tensor address is a `Buffer*` RTA/CRTA; the GlobalSemaphore
  addresses are raw CRTAs marked `smuggled-rta-ok` and re-applied by the overrides
  (`all_gather_async_llama_sharded_program_factory.cpp:335-340, 367-372`;
  `all_gather_async_default_program_factory.cpp:699-711, 719-733`). Via-broadcast embeds the addresses of its own
  op-owned semaphores in RTAs (`all_gather_via_broadcast_factory.cpp:233-234`); they live as long as the cached workload.
- **Device 2.0 (every kernel used):**

  *Llama-sharded* — **RED, isolated, cleared by the `[Prereq]`.** Both kernels are structurally Device 2.0 (`Noc`,
  `CircularBuffer`, `CoreLocalMem`, `UnicastEndpoint`). The writer calls a shared donor that is Device 1.0:

  | File | Line | Call | Disposition |
  |---|---|---|---|
  | `llama_shapes_sharded_writer.cpp` | 145-151 | `write_and_advance_local_read_address_for_fabric_write(…, fabric_connection, …)` → `ccl/common/kernels/minimal_ccl_common.hpp:76` raw `noc_async_write(…, safe_get_noc_addr(…))`, `:91` `noc_async_writes_flushed()` | `[Prereq]`: a local copy of the helper in the writer that uses `noc_obj` (in scope, `:77`). The donor also takes `FabricConnectionManager&`, which override 4b removes, so the port could not keep the call. |

  Sites left on purpose (override 4, replaced by the port, not holdovers):

  | File | Line | Call | Override |
  |---|---|---|---|
  | `llama_shapes_sharded_writer.cpp` | 53-54, 64 | GlobalSemaphore address CRTAs (`semaphore`, `barrier`) | 4a |
  | `llama_shapes_sharded_writer.cpp` | 126-127, 189-195 | `noc_semaphore_wait_min` / `noc_semaphore_set` on the raw semaphore pointers | 4a (`wait_min`, `set`) |
  | `llama_shapes_sharded_writer.cpp` | 106-107, 163-164 | `safe_get_noc_addr(x, y, sem, 0)` for the fabric atomic-inc headers | 4a (`.address()`) |
  | `llama_shapes_sharded_writer.cpp` | 183-185 | `noc_semaphore_inc(safe_get_noc_addr(drain_x, drain_y, sem), 1)` (increment on the drain core) | 4a; no `GlobalSemaphore` increment exists → keep `noc_semaphore_inc` on `gs.address()` (queue answer) |
  | `llama_shapes_sharded_writer.cpp` | 73-75, 103, 181 | `FabricConnectionManager::build_from_args<BUILD_AND_OPEN_CONNECTION_START_ONLY>`, `open_finish`, `has_*_connection`, `close` | 4b |

  Also kept: `get_noc_addr(core_x, core_y, output_base, 0)` (`:140-141`) builds the NoC address for the fabric packet
  header (no Device 2.0 replacement; same shape as the `safe_get_noc_addr` header sites that earlier ports kept).
  The reader is compliant (`Noc`, `CircularBuffer`, `CoreLocalMem`, `UnicastEndpoint`). Donors:
  `minimal_ccl_common.hpp` `perform_payload_send` (fabric API only, used after the `[Prereq]`),
  `kernel_common/worker_routing_utils.hpp` (constexpr CTA reads and packet-header setters).

  *Default* — **RED.** Shared with `all_gather_matmul_async` and `test_generic_op`.

  | File | Line | Call | Scope |
  |---|---|---|---|
  | `minimal_default_reader.cpp` | 146, 336 | raw `noc_async_read(addrgen.get_noc_addr(tile), l1, page_size)` with `noc_obj` in scope (`:122`) | every config |
  | `minimal_default_writer.cpp` | 427 (and the forward twin) | raw `noc_async_write(l1, local_noc_addrs[i], page_size)` | every config |
  | `minimal_default_reader.cpp` | 89, 113 · `minimal_default_writer.cpp` 180 | `experimental::ShardedAddrGen` from `shard_builder` CTAs/RTAs | sharded input or output (broad Device 1.0) |
  | `minimal_default_writer.cpp` | 155-159 | `get_semaphore(id)` for the mux client semaphores | mux configs |
  | `minimal_default_writer.cpp` | 231, 502-503 · `minimal_default_reader.cpp` 418 | `get_semaphore` (fused self-write-done semaphore); `noc_semaphore_inc(get_noc_addr(sem), -target)` (local atomic decrement of a GlobalSemaphore, no binding method) | fused / every config |

  *Via-broadcast* — compliant except the override-4a GlobalSemaphore sites (`broadcast_rm_writer.cpp:183, 203-204,
  238-257`). The reader is compliant.

  *Fabric mux kernel* (`tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp`) — a `tt_metal/fabric` kernel, outside the op;
  not audited for Device 2.0 (it is the subject of the missing feature below).
- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | none |
  | CBDescriptor `address_offset` (non-zero) | N/A | none |
  | GlobalSemaphore | **Supported on this branch (override 1)** — not RED | caller-supplied (llama-sharded, default); op-owned (via-broadcast) |
  | MeshSocket | N/A | none |
  | Fabric mux (default factory, more than one worker per direction) | **RED (config-scoped) — missing M2 feature** | see below |
  | Router connection through `RoutingPlaneConnectionManager` (via-broadcast) | **RED — kernel-side binding gap** | see below |

  #### Rule 8 check (router shared by several cores of one program)

  For each `append_fabric_connection_rt_args` site, (core, dst, link) per program:
  - **Llama-sharded** (`all_gather_async_llama_sharded_program_factory.cpp:313-326`): one worker per link
    (`num_workers_per_link = 1`, `:86-90`). Worker `link` opens `(forward neighbor, link)` and `(backward neighbor, link)`.
    No `(src, dst, link)` repeats. Edge case: on a 2-device ring the forward and backward neighbors are the same device,
    so one worker would hold `(src, N, link)` twice → rejected by the validator (override 7 edge case, not a block). The
    llama tests use a 4-device cluster axis (`tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py`).
  - **Default, no mux** (`all_gather_async_default_program_factory.cpp:660-680`): reached only when
    `num_workers_per_direction == 1` (`:143-145, 160-162`), so one worker per (direction, link): the forward worker
    opens `(backward neighbor, link)`, the backward worker opens `(forward neighbor, link)`. No repeats; same 2-device
    ring edge case.
  - **Default, mux** (`:561-573`): the router connection belongs to the mux core, one mux per valid direction per link
    (`:238-264`); the workers connect to the mux, not to the router. No repeats.
  - **Via-broadcast** (`all_gather_via_broadcast_factory.cpp:263-264`): one worker per link, forward and backward
    neighbor on `link`. No repeats.

  **Rule 8 fires for no factory.**

  #### Fabric mux (missing M2 feature, default factory)

  - **Signal.** With `num_workers_per_direction > 1`, `num_mux_cores_per_direction_per_link = 1`
    (`all_gather_async_default_program_factory.cpp:141-162`): the program adds the `tt_metal` fabric mux kernel
    `tt_fabric_mux.cpp` (`:533-548`), places its memory map in program-local L1 with
    `FabricMuxConfig::place_in_program_local_l1(desc, …)` (`:432-434`), gives it its router connection with
    `FabricMuxConfig::get_fabric_mux_run_time_args<ProgramDescriptor>` (`:565-570`), and connects each worker to it with
    `ccl::fabric_mux_connection_rt_args(…, desc, …)` (`:643-655`) under `USE_WORKER_MUX` (`:331-333`).
  - **Why Metal 2.0 cannot express it.** There is no mux in the M2 host API: `grep -rni mux
    tt_metal/api/tt-metalium/experimental/metal2_host_api/ tt_metal/impl/metal2_host_api/` finds nothing.
    - The mux kernel reads its router connection with `FabricMuxToEdmSender::build_from_args`
      (`tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp:164`), fed by `append_fabric_connection_rt_args` inside
      `get_fabric_mux_run_time_args` (`tt_metal/fabric/fabric_mux_config.cpp:311-340`). A `FabricConnectionSpec` binding
      could replace those RTAs only in a `_metal2` fork of a `tt_metal/fabric` kernel, which is outside the op.
    - The mux reads its program-local L1 placement as positional CTAs: a CB index and a semaphore id
      (`tt_fabric_mux.cpp:46-48, 149-150`), reserved by `reserve_fabric_mux_program_local_l1`, which has only
      `ProgramDescriptor` and `Program` overloads (`tt_metal/fabric/fabric_mux_program_local_l1.cpp:45, 90`;
      `fabric_mux_config.cpp:467-483`). No `ProgramSpec` form exists.
    - Each worker gets four or five local semaphores by id from the mux helper (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp`,
      `fabric_mux_connection_rt_args(…, ProgramDescriptor&, …)`, it pushes `SemaphoreDescriptor`s into `desc`) and reads
      them with `get_semaphore(id)` (`minimal_default_writer.cpp:155-159`).
  - **Scope.** Config-scoped: every default-factory config with more than one worker per direction. That is the default
    heuristic's choice for more than 4 KB per link (`:68-76`), and what `tests/ttnn/unit_tests/operations/ccl/test_ccl_mux_l1.py`
    and the blackhole nightly (`num_workers_per_link=2`) run.
  - **Expected resolution.** A Metal 2.0 fabric-mux feature (a mux node bound to a `FabricConnectionSpec`, its L1 map as
    a program resource, client bindings for the workers). Routed to the Metal 2.0 host API team.

  #### `RoutingPlaneConnectionManager` (kernel-side binding gap, via-broadcast)

  - `broadcast_rm_writer.cpp:12, 162-167, 260` holds a `tt::tt_fabric::RoutingPlaneConnectionManager`, filled by
    `RoutingPlaneConnectionManager::build_from_args` (`tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp:62-98`)
    from `append_routing_plane_connection_manager_rt_args` (`all_gather_via_broadcast_factory.cpp:263-264`), and passes
    it to the linear fabric API (`fabric_multicast_noc_unicast_write_set_state(RoutingPlaneConnectionManager&, …)`,
    `tt_metal/fabric/hw/inc/linear/api.h:996`, and the scatter and atomic-inc twins).
  - The manager has no `build_from_binding` and its slots are private (`routing_plane_connection_manager.hpp:155-157`),
    so `fabric::` bindings cannot fill it. Under `FABRIC_2D` it also carries per-target `dst_dev_id` / `dst_mesh_id`
    from RTAs (`:86-94`). Override 4b covers the RTAs, not a rewrite of the linear-API calls onto raw senders.
  - Not ported in this queue (the queue asked for the other two factories). Expected resolution: a
    `RoutingPlaneConnectionManager` built from `fabric::` bindings (kernel API, `tt_metal/fabric`).
- **CB endpoints:** llama-sharded only (below). Default and via-broadcast: skipped (no port ahead).
- **Offset base pointers:** **GREEN.** Every address is a clean `Buffer*` base:
  default reader/writer CRTAs input, output (`all_gather_async_default_program_factory.cpp:703-704`); llama-sharded
  reader CRTA input (`all_gather_async_llama_sharded_program_factory.cpp:333`), writer CRTA output (`:337-338`);
  via-broadcast reader RTA input (`all_gather_via_broadcast_factory.cpp:209`), writer RTA output (`:270`). The shard and
  page offsets are added in the kernels (`llama_shapes_sharded_reader.cpp:60`, `llama_shapes_sharded_writer.cpp:142`).
- **TensorAccessor 3rd argument:** **N/A.** All constructions take two arguments (`minimal_default_reader.cpp:96, 119`,
  `minimal_default_writer.cpp:186`, `broadcast_rm_reader.cpp:36`, `broadcast_rm_writer.cpp:170`); the llama kernels build
  no accessor.

## Port-work summary  *(mirrors the brief; llama-sharded subset)*

- **Tensor bindings:** `input` → reader, **Case 2** (raw shard base + `shard_tile_id * page_size`, read from listed
  cores, `llama_shapes_sharded_reader.cpp:37, 60-67`); `output` → writer, **Case 2** (raw base for the local write and
  the fabric header, `llama_shapes_sharded_writer.cpp:52, 140-142`). Both tensors are L1-sharded by the selection rule
  (`composite_common.cpp:206-262`), so the bridge is `LocalTensorAccessor<uint8_t>(tensor::x)` and its local-memory
  address (this branch has no `TensorAccessor::get_bank_base_address`; precedent:
  `matmul/.../reader_bmm_tile_layout_in0_sender_dram_sharded.cpp:145, 161`).
- **TensorParameter relaxation:** none needed (code-derived).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** `c_in0` (`.cpp:123-131`) — reader FIFO-produces, writer FIFO-consumes → 1P+1C. `c_in1` packet
  header (`:136-144`) — only the writer touches it (`reserve_back`/`push_back` + `get_write_ptr`) → **self-loop** (DM).
  No multi-binding, no dead CB.
- **GlobalSemaphores (override 5):** `semaphore` (= `semaphore.at(0)`) and `barrier_semaphore` (optional; used only
  when present and `!using_persistent_buffers`, `:177-178`), both bound by the writer only.
- **Fabric (override 4b, 6, 7):** `forward` and `backward` specs; target on worker `link` = `{neighbor, link}` when the
  neighbor exists.

## Heads-ups  *(mirrors the brief)*

- **RTA varargs (both llama kernels).** The core lists are a variable-count block: `num_cores` then `num_cores` x
  coordinates then `num_cores` y coordinates, read through `get_arg_addr` (`llama_shapes_sharded_reader.cpp:41-45`,
  `llama_shapes_sharded_writer.cpp:58, 67-70`). They become RTA varargs; the scalars before them become named RTAs.
- **Positional route CTAs.** The writer reads its forward/backward line-multicast route as positional CTAs 10-21
  through `ccl_routing_utils::get_line_multicast_route_info_from_args<10>()` (`llama_shapes_sharded_writer.cpp:36-39`;
  shared header `kernel_common/worker_routing_utils.hpp:47-57`). Name the twelve values and build the two
  `line_multicast_route_info_t` locally; do not edit the shared header.
- **Conditional barrier semaphore.** `barrier_semaphore` is optional. Bind it only when the writer uses it; the kernel
  code that touches it is behind the `use_barrier_sem` CTA, so a define must gate the binding token.
- **Semaphore span.** Both caller-supplied GlobalSemaphores must span the worker cores. With
  `use_optimal_ccl_for_llama` the workers are a fixed placement (`llama_common.cpp:28-32`); callers create the
  semaphores on the sub-device grid. CI check, not a blocker.

## Team-only

- **Out-of-directory coupling:**
  - Llama-sharded writer: `ccl/common/kernels/minimal_ccl_common.hpp` (`perform_payload_send`, templated sender ✓;
    `write_and_advance_local_read_address_for_fabric_write`, `FabricConnectionManager&` + Device 1.0 → replaced locally by
    the `[Prereq]`), `ccl/kernel_common/worker_routing_utils.hpp` (positional CTA reader → named CTAs at the call site;
    `fabric_set_line_multicast_route` ✓), `ccl/shared_with_host/ccl_runtime_args.hpp` (`LlamaGather*CommonArgs` CRTA
    indices; unused after the port, left in place because the header is shared).
  - Default factory: its builder is public API used by `all_gather_matmul_async`; its kernels are also in
    `test_generic_op.cpp`; `neighbor_pad_halo/.../np_w_mux_writer.cpp:12` says it copies the `USE_WORKER_MUX` path.
  - Borrowed kernel files: none for llama-sharded (both kernels op-owned, no `_metal2` fork needed). Default borrows
    `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp`.
- **Relaxation candidates:** none noted.

## Misc anomalies  *(team-only, non-gating)*

- `build_llama_sharded_program_descriptor` re-reads `input_tensor.device()` when the first read is null
  (`all_gather_async_llama_sharded_program_factory.cpp:60-63`); dead branch.
- Llama-sharded `cores_per_device = output_cores_vec.size() + ring_size - (1 / ring_size)` (`:212`) is larger than the
  device slice it names; the loop only indexes the needed prefix.
- Llama-sharded writer `arg_for_fab`, `connect_to_fabric_when_creating`, `my_chip_id` and the reader `my_chip_id`,
  `arg_idx` locals are unused (`llama_shapes_sharded_writer.cpp:26, 71-72`; `llama_shapes_sharded_reader.cpp:22`).
- `AllGatherAsyncParams::use_all_gather_async_via_broadcast` defaults to `true` in the struct
  (`all_gather_async_device_operation_types.hpp:35`); every builder passes it explicitly.

## Per-DeviceOperation attribution

One device operation. Per-factory status is in the Result table.

## Questions for the user

1. **Readiness sheet.** Please confirm the `experimental/ccl/all_gather_async` rows (three factories): `Concept`,
   `Is able to port?`, `TensorParameter relaxation`, `Known op issues`, `Porting Target`. This audit could not read it.
2. **Default factory split.** Is a config-scoped Metal 2.0 port of the default factory (one worker per direction,
   interleaved, not fused) wanted before a fabric-mux feature exists? This audit chose no (see Result).

## Recipe notes

- **Fabric mux.** Appendix A has no fabric-mux entry. The queue brief's rule ("a missing feature is something the M2
  host API cannot express at all") applies: no M2 construct names a mux, its L1 placement or its client connections.
- **Case 2 bridge.** The recipe names `TensorAccessor::get_bank_base_address`; this branch has no such method. For L1
  tensors the bridge is `LocalTensorAccessor` and `local_mem::detail::AddressAccess::address(...local_mem())`.
- **Mixed factory variants.** The recipe does not say whether one device operation may hold Metal 2.0 and descriptor
  factories at once. The framework allows it per alternative.
- **Unreachable readiness sheet** and **provenance command**: as for the earlier ports on this branch.
