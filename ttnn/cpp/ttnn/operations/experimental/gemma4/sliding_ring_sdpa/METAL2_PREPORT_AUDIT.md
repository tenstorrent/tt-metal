# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa`

**Audited at `3e1814617f9`.** Audit only: no `[Prereq]` commit and no port were made.

- **`Gemma4SlidingRingSDPADeviceOperation`** (`device/sliding_ring_sdpa_device_operation.{hpp,cpp}`)
  - `Gemma4SlidingRingSDPAMeshWorkloadFactory` (`device/sliding_ring_sdpa_program_factory.hpp:44`) — the only
    entry of `program_factory_t` (`sliding_ring_sdpa_device_operation.hpp:25`). It is a hand-written
    `MeshWorkloadFactoryConcept` (`create_mesh_workload` + `override_runtime_arguments(cached_mesh_workload_t&, …)`)
    that forwards to `DescriptorMeshWorkloadAdapter<Gemma4SlidingRingSDPAProgramFactory>`
    (`sliding_ring_sdpa_program_factory.cpp:2372-2397`).
  - `Gemma4SlidingRingSDPAProgramFactory::create_workload_descriptor` (`sliding_ring_sdpa_program_factory.cpp:2357`)
    — builds one `ProgramDescriptor` per mesh coordinate (`build_ring_joint_sdpa_program_descriptor`, `:692`).
- Kernels (all five are referenced; all are op-owned):
  - `device/kernels/dataflow/sliding_ring_sdpa_reader.cpp` — SDPA reader, one instance per core of the SDPA grid
  - `device/kernels/dataflow/sliding_ring_sdpa_writer.cpp` — SDPA writer, same grid
  - `device/kernels/compute/sliding_ring_sdpa.cpp` (+ `sliding_ring_sdpa_streaming.hpp`) — SDPA compute, same grid
  - `device/kernels/dataflow/sliding_halo_reader.cpp` — halo reader, one instance per link worker of each exchange
  - `device/kernels/dataflow/sliding_halo_writer.cpp` — halo writer (fabric sender), same workers

**The sliding halo exchange is not a separate program.** `sliding_halo_exchange_helper`
(`device/sliding_halo_exchange.cpp:53`) takes the SDPA `ProgramDescriptor& desc` and appends to it: its CBs
(`:129-155`), its worker-sync semaphore (`:233-240`), its fabric connection args (`:440-447`) and its reader/writer
kernels (`:462-463`). The factory pushes the three SDPA kernels first (`sliding_ring_sdpa_program_factory.cpp:2085-2087`)
and then calls the helper once per exchange on the same `desc` (`:2325-2343`). The two parts are coupled inside the
program: the halo reader signals the SDPA cores' local semaphores (`fused_sem0/1`, `:1028-1041`, handed to the halo
through `AllGatherFusedOpSignaler::init_fused_op`, `:2089-2095`), and the SDPA reader waits on them before it reads the
halo (`sliding_ring_sdpa_reader.cpp:296, 519-532`). The cache-hit override patches the halo kernels by kernel index
`3 + 2 * exchange` (`sliding_ring_sdpa_program_factory.cpp:147-148, 474, 485`). A port of the halo exchange is
therefore a port of the whole SDPA program.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` — from
the recipe branch ref `origin/akertesz/op-porting-recipe`. The recipe docs are not in this checkout.

**Branch-specific overrides applied** (this branch is newer than the recipe):

1. GlobalSemaphore is supported (`ProgramSpec::advanced_options.global_semaphore_parameters`,
   `KernelSpec::advanced_options.global_semaphore_bindings`, `ProgramRunArgs::advanced_options.global_semaphore_args`;
   kernel `experimental::GlobalSemaphore`). The Appendix A "GlobalSemaphore — UNSUPPORTED" entry is not applied.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores`. This op owns none.
3. Fabric connections (`ProgramAdvancedOptions::fabric_connections`, `FabricConnectionSpec`,
   `fabric_connection_bindings`) replace `tt::tt_fabric::append_fabric_connection_rt_args`.
4. Sanctioned kernel-side exception: replace (a) the raw GlobalSemaphore address RTA and its raw-pointer uses with a
   `global_sem::` binding and (b) the fabric-connection RTAs read by `build_from_args` with `fabric::` bindings.
5. Caller-supplied GlobalSemaphores (left out of the hash, re-applied by the override): one `GlobalSemaphoreParameter`
   per semaphore, passed in both the cache-miss run args and the M2 override from one helper.
6. A binder node with no fabric target: guard the sender with `has_fabric_connection`.
7. One node that talks to several destinations: one `FabricConnectionSpec` per destination slot. Validator rule: each
   `(src_fabric_node_id, dst_fabric_node_id, link_idx)` appears at most once per program. A missing feature is only
   something that the M2 host API cannot express at all.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa` |
| **Overall** | **RED** — three gates fail. One needs a **new M2 host feature** (config-scoped); two are op-side prerequisites (whole-op). |
| **DOps / Factories** | `Gemma4SlidingRingSDPADeviceOperation` → `Gemma4SlidingRingSDPAMeshWorkloadFactory` (wraps `Gemma4SlidingRingSDPAProgramFactory`) |
| *Prereqs* — Device 2.0 (every kernel used) | **No** (RED, op-owned halo kernels, isolated holdovers: a Device 1.0 local semaphore in the halo writer and two free-function NoC barriers with a `Noc` in scope). SDPA reader / writer / compute and all donors: compliant. |
| *Prereqs* — Cross-op escapes | Not inventoried in full (skipped, see below). Partial notes in Team-only. |
| *Feature Support* — overall | **RED** — sequential sharing of one fabric router by several workers of one program (time-shared halo links). No Appendix A entry fires after override 1. |
| *Feature Support* — Variadic-CTA | Not run (skipped). |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Not read — sheet not reachable.** Derived from code: **RED** — the registered factory is a `MeshWorkloadFactoryConcept` shell (`legacy (MeshWorkload)` shape) around a `WorkloadDescriptor` factory. |
| *TTNN Readiness* — Concept (current) | `legacy (MeshWorkload)` shell (`sliding_ring_sdpa_program_factory.hpp:44-62`) over `WorkloadDescriptor` (`:36-41`) |
| *TTNN Readiness* — Execution Model | `MPMD` — the kernel count, the halo CTAs (`ring_index`, route, `send_backward`, `multicast`) and the link hand-off semaphore depend on the coordinate. `Secretly SPMD Workload?` = no. |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate): `sliding_ring_sdpa_device_operation.cpp:1134`. It leaves out the GlobalSemaphores (through `RingAttentionAllGatherAsyncParams::attribute_names`, `ring_attention_all_gather_async_device_operation_types.hpp:78-82`), and `logical_n` under KV-pad rotation. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate): `sliding_ring_sdpa_program_factory.cpp:2381` → `apply_ring_joint_scalar_runtime_args` (`:422-678`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`sliding_ring_sdpa_nanobind.cpp:132` binds only the op) |
| *TTNN Readiness* — Op-owned tensors | No (`WorkloadDescriptor::buffers` and `::semaphores` are not set, `:2357-2370`). Op-owned GlobalSemaphores: no. |
| *TTNN Readiness* — Target concept | After the TTNN prereq: `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments` (code-derived; sheet not read) |
| *Port work* — Offset base pointer | **none** (GREEN) |
| *Port work* — Tensor bindings (per binding) | Case 1 for every live binding (collected with the offset scan; see Team-only). Not a full TensorParameter census. |
| *TTNN Readiness* — TensorParameter relaxation | Not read — sheet not reachable. |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor that this op runs passes a 3rd argument |
| *Port work* — CB endpoints | Not run (skipped). |

**Skipped subjects** (`skipped — whole-op RED, no portable subset; two of the three blockers are cleared on the
op-code side, so re-audit reads changed code`): TTNN porting shape (only the facts above), TensorParameter
relaxations, TensorParameter analysis (only the per-binding classes collected for the offset scan), CB endpoints, RTA
varargs, Out-of-directory coupling (only partial notes), Incidental anomalies (only what was seen).

## Result

**RED → blocked on three gates.** RED at op level; **no portable subset today.**

1. **Feature compatibility — new M2 host feature needed (config-scoped).** A halo with more unicast hops in one
   direction than fabric links "time-shares a link": several halo writers on different worker cores of the **same
   program** open the same `(src, dst, link)` router connection one after another, chained by a local semaphore. The
   M2 validator rejects any `(src_fabric_node_id, dst_fabric_node_id, link_idx)` that appears more than once in a
   program, across all specs. M2 cannot express this. Routed to the Metal 2.0 host API team.
2. **Device 2.0 (whole-op).** Both op-owned halo kernels have isolated Device 1.0 holdovers (5 sites). Routed to the
   Device 2.0 track (or a `[Prereq]` in this queue: the fix is local to two op-owned files).
3. **TTNN factory concept (whole-op).** The registered factory is a hand-written `MeshWorkloadFactoryConcept` shell. The
   descriptor concept already has the per-program override this shell adds, so the fix is local to the op: register
   `Gemma4SlidingRingSDPAProgramFactory` and move `apply_ring_joint_scalar_runtime_args` into its
   `override_runtime_arguments(Program&, …, const std::optional<MeshCoordinate>&)`. Routed to TTNN / the op owner.

**After the two op-side prerequisites, a subset is clear:** every config in which no direction has more unicast halo
exchanges than its link span — one-hop halos, multicast halos, and multi-hop unicast halos with at most
`min(num_links, links_available)` hops per direction. The time-shared configs stay blocked on item 1.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED (code-derived; sheet not read).** The readiness sheet needs a
  Google Drive download that this session does not have. From the code:
  - `program_factory_t = std::variant<Gemma4SlidingRingSDPAMeshWorkloadFactory>`
    (`sliding_ring_sdpa_device_operation.hpp:25`). That struct defines `create_mesh_workload` and
    `override_runtime_arguments(cached_mesh_workload_t&, …)` (`sliding_ring_sdpa_program_factory.hpp:44-60`) and
    `static_assert`s `MeshWorkloadFactoryConcept` (`:62`). By the recipe's concept rule that is `legacy (MeshWorkload)`,
    which the sheet would show as `yes (with PD step)`.
  - The shell is thin. `create_mesh_workload` calls `descriptor_adapter_t::create_mesh_workload`
    (`sliding_ring_sdpa_program_factory.cpp:2378`). `override_runtime_arguments` calls
    `descriptor_adapter_t::apply_descriptor`, then `apply_ring_joint_scalar_runtime_args(program, …, coord)` for each
    per-coordinate program (`:2386-2396`). The descriptor concept already probes a factory-level
    `override_runtime_arguments(Program&, attrs, tensor_args, ret, const std::optional<MeshCoordinate>&)`
    (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:507-516`); `slice_reshard_async` uses that hook with a
    `WorkloadDescriptor`. **Path forward:** an op-side `[Prereq]` that deletes the shell, registers
    `Gemma4SlidingRingSDPAProgramFactory`, and moves the scalar patch into that hook. Then the concept is
    `WorkloadDescriptor` + override and the target is `MeshWorkloadSpecFactoryConcept`.
  - `Custom hash`: yes (`sliding_ring_sdpa_device_operation.cpp:1134`). `get_dynamic_runtime_args`: none.
    `Pybind descriptor`: none. `Op-owned tensors?`: no. Factory set: one registered factory.
  - `Smuggled pointer`: no tensor address is smuggled; every tensor address is a `Buffer*` RTA/CRTA. The
    GlobalSemaphore addresses are raw RTAs marked `smuggled-rta-ok` (`sliding_halo_exchange.cpp:267, 423-424`) and
    re-patched by the override (`sliding_ring_sdpa_program_factory.cpp:462-501`) — override 5.
- **Device 2.0 (every kernel used):** **RED — isolated holdovers in the two op-owned halo kernels.** Both kernels are
  otherwise structurally Device 2.0 (`Noc`, `CircularBuffer`, `CoreLocalMem`, `TensorAccessor`). The same sites exist
  in the upstream copies (`experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_neighbor_halo_{reader,writer}.cpp`),
  from which these kernels were forked.

  | File | Line | Call | Wrapper in scope / replacement |
  |---|---|---|---|
  | `sliding_halo_writer.cpp` | 191 | `get_semaphore(chain_semaphore_id)` (local program semaphore, link hand-off) | `Semaphore<> chain(chain_semaphore_id)` |
  | `sliding_halo_writer.cpp` | 194 | `noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chain_semaphore_addr), 1)` | `chain.wait_min(1)` |
  | `sliding_halo_writer.cpp` | 324 | `noc_semaphore_inc(get_noc_addr(chain_successor_noc_x, chain_successor_noc_y, chain_semaphore_addr), 1)` | `chain.up(noc_obj, x, y, 1)` |
  | `sliding_halo_writer.cpp` | 267 | `noc_async_writes_flushed()` | `noc_obj` (line 197) → `noc_obj.async_writes_flushed()` |
  | `sliding_halo_reader.cpp` | 256 | `noc_async_atomic_barrier()` | `noc` (line 207) → `noc.async_atomic_barrier()` |

  Sites left on purpose (override 4, replaced by the port, not holdovers):

  | File | Line | Call | Override |
  |---|---|---|---|
  | `sliding_halo_reader.cpp` | 200-205, 248, 254 | arrival GlobalSemaphore address RTAs, `noc_semaphore_wait_min`, local atomic decrement `noc_semaphore_inc(get_noc_addr(sem), -1)` | 4a (see API gap 1: the kernel `GlobalSemaphore` has no atomic `up`/`down`) |
  | `sliding_halo_writer.cpp` | 61-63, 218-219 | arrival GlobalSemaphore address RTA and `safe_get_noc_addr(x, y, sem, 0)` for the fused atomic-inc header | 4a |
  | `sliding_halo_writer.cpp` | 179-180, 214, 216-217, 319 | `FabricConnectionManager::build_from_args`, `open`, forward/backward pick, `close` | 4b |

  Compliant: `sliding_ring_sdpa_reader.cpp` and `sliding_ring_sdpa_writer.cpp` (all CB access through `CircularBuffer`,
  all transfers through `Noc`, local semaphores through `Semaphore<>`; sanctioned `get_tile_size(cb)` at reader
  441-443 and writer 241, 445, 457; sanctioned `get_local_cb_interface` at halo reader 209-210). The compute kernel is
  outside the Device 2.0 data-movement scope; it uses `CircularBuffer` objects and compute CB calls only, and its
  donor `compute_streaming.hpp:179` uses the sanctioned `get_local_cb_interface`. Donors called by the five kernels are
  compliant: `transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp` (`fetch_block`, `read_block`,
  `write_block_row_grouped_trid`, `PaddedAddrGenerator`, `generate_lightweight_mask_tiles`; its raw L1 tile fills at
  `:338-339, 382-383, 513-514` were kept by the transformer Device 2.0 migration #47107 / #47424, same as
  `ttnn/kernel/dataflow/generate_bcast_scalar.hpp`), `fused_op_receiver.hpp` (`Semaphore<>`),
  `metadata_scalar_read.hpp`, `ccl/kernel_common/worker_sync_utils.hpp` (`OpSignaler::synchronize_workers_and_signal_op`,
  `Semaphore<>`), `ring_attention_prefetch_utils.hpp` (`Noc`, `CircularBuffer`), `ring_attention_all_gather_metadata.hpp`
  (math only), `ccl/common/kernels/minimal_ccl_common.hpp` (`perform_payload_send`, fabric API),
  `kernel_lib/reduce_helpers_dataflow.inl`.
- **Feature compatibility:** **RED — one new M2 host feature needed.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type, no `.global_circular_buffer` field, no remote CB |
  | CBDescriptor `address_offset` (non-zero) | N/A | no CB sets `address_offset`; no CB is buffer-backed |
  | GlobalSemaphore | **Supported on this branch (override 1)** — not RED | caller-supplied `std::vector<GlobalSemaphore> semaphore` in `RingAttentionAllGatherAsyncParams` (`ring_attention_all_gather_async_device_operation_types.hpp:34`), used through `args.all_gather_operation_attributes.semaphore` (`sliding_ring_sdpa_program_factory.cpp:464, 2240, 2336`). Arrival semaphores start at index 2 (`sliding_halo_exchange.hpp:90`). Under the recipe as written this row is RED. |
  | MeshSocket | N/A | none |
  | Fabric connection, sequential sharing of one router by several workers in one program | **RED (config-scoped) — new M2 host feature** | see below |

  #### Time-shared halo links (new M2 host feature)

  - **Signal.** `SlidingHaloConfig` (`sliding_halo_exchange.hpp:55-64`): "A halo with more hops than fabric links
    time-shares a link … a later hop waits on this local semaphore, which its predecessor on the same link increments
    after closing its connection." The factory splits the exchanges by direction and gives each direction a link span
    `min(exchanges in that direction, num_links, links_available)` (`sliding_ring_sdpa_program_factory.cpp:2173-2202`).
    Exchange `k` of a direction runs on link `k % span` (`link_base`, `:2287`; used at
    `sliding_halo_exchange.cpp:444`), waits when `k >= span` and signals when `k + span` exists (`:2259-2265`). When
    `halo_links_shared` (`:2203-2204`), a local `SemaphoreDescriptor` on every exchange's lead core carries the
    hand-off (`:2207-2238`). Each exchange has its own worker cores (`halo_exchange_core_grid_offset`, `:2135-2140`), so
    two exchanges on the same link are two different nodes, each with its own kernel instance, in one program, with the
    same `(src = this device, dst = the forward or backward neighbour, link_idx)` (`append_fabric_connection_rt_args`,
    `sliding_halo_exchange.cpp:440-447`).
  - **Why M2 cannot express it.** `ProgramAdvancedOptions::fabric_connections` invariant: "Each
    (src_fabric_node_id, dst_fabric_node_id, link_idx) appears at most once in the program: a router accepts one
    connected worker at a time" (`tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:119-122`).
    `ValidateFabricConnectionUsage` keeps one `used_links` set for the whole program, across every spec and node, and
    `TT_FATAL`s on a repeat (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/fabric_connection.cpp:96-107`).
    Splitting the targets into more specs does not help. There is no way to declare "these workers open this router one
    after another".
  - **Why this is not the override-7 edge case.** Override 7 covers one node whose slots reach the same neighbour on
    the same link by accident (for example a 2-device ring), which is not a valid use anyway. Here the reuse is the
    designed protocol of the op: different nodes, sequential use, ordered by a semaphore that the op allocates. It is
    exercised by the shared multi-hop test that the gemma4 suite reuses
    (`tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py:7022-7031`, "On a BH Galaxy's two fabric links the four-hop
    case time-shares them, two hops per link"; `tests/ttnn/unit_tests/operations/experimental/gemma4/test_sliding_ring_sdpa.py:199-200`).
  - **Scope (config-scoped, not whole-op).** The planner `plan_chunked_sliding_halo_exchanges`
    (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sliding_halo_layout.cpp:36-71`) decides:
    - **Multicast halos** (1D fabric, source-keyed layout, one halo slot): at most one exchange per direction, each on
      `num_links` workers with links `0 … num_links-1` (`halo_links_per_exchange`, `sliding_ring_sdpa_program_factory.cpp:209-212`).
      **Clear.** `hop_origin_rows` and the line multicast route are kernel-side packet-header data
      (`sliding_halo_writer.cpp:272-281`); the connection still targets the neighbour router.
    - **One-hop halos**: one exchange on `num_links` distinct links. **Clear.**
    - **Unicast multi-hop halos** (2D fabric, or not source-keyed, or more than one halo slot): one exchange per remote
      hop, one worker each. **Blocked when, in some direction, the number of exchanges is greater than
      `min(num_links, links_available)`.** On a ring topology every hop goes forward, so the condition is
      `remote_hop_count > min(num_links, links_available)`. On a linear topology, device `r` sends
      `min(hops, ring_size - 1 - r)` hops forward and the rest backward, so the condition is per coordinate. The op
      allows up to `sliding_max_halo_hops = 8` hops (`sliding_ring_sdpa_device_operation.cpp:728-744`,
      `transformer/sdpa/device/kernels/sliding_window_work_plan.hpp:167`).
  - **Expected resolution.** A Metal 2.0 host feature that lets one program declare that several worker nodes open
    the same `(src, dst, link)` one after another (the program orders them; the next `open()` adopts the cursor that
    the previous `close()` persisted, `edm_fabric_worker_adapters.hpp`). An alternative that needs no new feature is an
    op redesign in which one worker per link carries every exchange that shares the link; that is a semantic change on
    the op side, not port work.
- **CB endpoints:** not run (skipped). Nothing here gates.
- **Offset base pointers:** **GREEN.** Every address RTA/CRTA is a clean `Buffer*` base:
  SDPA reader RTAs Q, K, V, gathered K, gathered V, joint Q/K/V, attention sink, gathered joint K/V
  (`sliding_ring_sdpa_program_factory.cpp:1975-1990`); reader CRTAs `slot_id`, `kv_actual_isl`, `logical_n`/`logical_l`,
  `kv_valid_end` (`:1884-1895`); writer RTAs output, joint output, stats (`:2015-2017`); writer CRTAs (`:1914-1918`);
  halo reader input buffers, `slot_id`, `kv_actual_isl` (`sliding_halo_exchange.cpp:390-394, 413-415`); halo writer
  output buffers, `kv_actual_isl` (`:394, 429-431`). The slot-derived and layer-derived starts are **page indices**,
  never folded into an address: `kv_cache_layer_idx` and `kv_cache_num_layers` go to
  `trace_metadata::bounded_cache_batch_idx` (`metadata_scalar_read.hpp:44-53`, returns `slot * num_layers + layer`),
  `input_batch_base` is a page count added to page ids (`sliding_halo_exchange.cpp:335-342`,
  `sliding_halo_reader.cpp:156-157, 237`), and the halo origin rows are page offsets. The op is not in the
  offset-base-pointer triage tables.
- **TensorAccessor 3rd argument:** **N/A** — no accessor that this op runs passes a 3rd argument. All constructions
  are two-argument: SDPA reader `:459-461, 492, 494`, SDPA writer `:304`, `sliding_ring_sdpa_lanes.hpp:24`,
  `metadata_scalar_read.hpp:37`, and the halo kernels' `make_tensor_accessor_tuple` (`tensor_accessor.h:782-788`).
  The only 3-argument site in a donor, `dataflow_common.hpp:83` (`read_page_table_for_batch`), is not called by this
  op.

## Port-work summary  *(mirrors the brief)*

No brief is issued (RED). The items below were collected while the gates ran; they are not a full census.

- **Tensor bindings:** every live binding is Case 1 (`Buffer*` base fed to a `TensorAccessor`). Dead address RTAs are
  listed in Misc anomalies.
- **TensorAccessor 3rd arg:** none.

## Heads-ups  *(mirrors the brief)*

- **Caller-supplied GlobalSemaphores (override 5).** Arrival semaphores `semaphore[2 + key]`, one key per predecessor
  (`halo_arrival_key`, `sliding_ring_sdpa_program_factory.cpp:166-174`). Writer RTA 2 (`kNeighborWriterSemaphoreArg`)
  and the collector's trailing RTAs are re-patched on every dispatch (`:462-501`). `semaphore[0]` and `[1]` are not used
  by this op.
- **Runtime-arg override is large.** Besides the semaphores, `apply_ring_joint_scalar_runtime_args` re-writes the halo
  page ranges, origins and batch base and the SDPA scalars (`logical_nt`, Q mapping, `kv_cache_batch_idx`) on every
  dispatch (`:422-678`), indexed by fixed RTA offsets (`halo_rt::*`, `sliding_halo_exchange.hpp:88-109`). In M2 these
  become named run args.
- **Local semaphores by id through RTAs (donor coupling).** The halo reader's `OpSignaler(arg_idx)`
  (`ccl/kernel_common/worker_sync_utils.hpp:42-55`) and the SDPA reader/writer's `RingSDPAOpReceiver`
  (`transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp`) read semaphore ids from the RTA stream
  (pushed by `AllGatherFusedOpSignaler::push_all_gather_fused_op_rt_args` and
  `RingSDPAFusedOpSignaler::push_ring_sdpa_fused_op_rt_args`). M2 delivers semaphores as `sem::` binding tokens, so a
  port needs `_metal2` forks of these donors (shape `uint32_t sem_id`, "not workable today").

## Team-only

- **Out-of-directory coupling (partial; the full inventory was skipped):**
  - `sliding_halo_exchange.hpp:19` includes
    `experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp`
    (for `RingAttentionAllGatherAsyncParams` and `AllGatherFusedOpSignaler`), and `sliding_halo_exchange.cpp:6` includes
    its program factory header (for `input_batch_base_pages`). The op's attributes carry
    `all_gather_operation_attributes` of that type.
  - The factory uses `transformer/sdpa/device/sliding_halo_layout.hpp` (`build_chunked_sliding_halo_layout`,
    `plan_chunked_sliding_halo_exchanges`), shared with `transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`,
    which has the same time-shared link design (`:3650-3729`).
  - Kernel donors with shape notes: `ring_attention_prefetch_utils.hpp` `prefetch_batch_read_tiles(const Noc&,
    CircularBuffer&, …)` (⭐ `CircularBuffer&` — op-by-op friction); `OpSignaler` and `RingSDPAOpReceiver` (semaphore
    ids from RTAs, see Heads-ups); `dataflow_common.hpp` helpers take `uint32_t cb_id` (✓).
  - Borrowed kernel files: none. All five kernels are op-owned; no `_metal2` fork exists.
- **CB endpoint shapes seen (not a census):** the SDPA writer reads three compute CBs of a remote K-split sender
  (`cb_max_A`, `cb_sum_A`, `cb_out_im_A`) by address (`sliding_ring_sdpa_writer.cpp:452-466`); the SDPA reader and
  writer use `cb_q_in` / `cb_out` write pointers as metadata scratch (`sliding_ring_sdpa_reader.cpp:341-342`,
  `sliding_ring_sdpa_writer.cpp:263-265`); `cb_v_in` aliases `cb_k_in` for latent V
  (`sliding_ring_sdpa_program_factory.cpp:1642`); every halo exchange re-uses CB indices `c_in1/c_in2/c_in4/c_in5` on
  its own disjoint worker cores (`sliding_halo_exchange.cpp:122-125`).
- **Relaxation candidates:** none noted.
- **TTNN factory analysis:** current concept `legacy (MeshWorkload)` shell over `WorkloadDescriptor`; after the
  prereq, target `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments`. Ported-from entry
  point: `create_workload_descriptor` (`sliding_ring_sdpa_program_factory.cpp:2357`), one program per coordinate in
  `tensor_coords`. Per coordinate: transport rank, forward/backward neighbours, the exchange list and directions, the
  halo CTAs, the fabric targets, and the link hand-off semaphore. No op-owned tensors or semaphores.

## Misc anomalies  *(team-only, non-gating)*

- **Attention sink is never loaded (suspected hang).** Validation accepts `attention_sink`
  (`sliding_ring_sdpa_device_operation.cpp:529-557`) and the factory allocates `cb_attention_sink`
  (`sliding_ring_sdpa_program_factory.cpp:1653-1659`), but the SDPA reader reads `attention_sink_addr`
  (`sliding_ring_sdpa_reader.cpp:275`) and never fills the CB, while compute waits on it
  (`transformer/sdpa/device/kernels/compute/compute_streaming.hpp:834`). Not verified on hardware.
- **Dead address RTAs.** SDPA reader: joint Q/K/V, gathered joint K/V and attention sink addresses
  (`sliding_ring_sdpa_reader.cpp:267-283`; the joint helpers ignore them, `:57-115`). SDPA writer: `joint_out_addr`,
  `stats_addr` (`sliding_ring_sdpa_writer.cpp:201-202`).
- **Dead CTAs.** The halo kernels get one `page_size` CTA per input (`sliding_halo_exchange.cpp:177-179, 218-220`);
  the kernels only skip over them (`sliding_halo_reader.cpp:57`, `sliding_halo_writer.cpp:40, 45`).
- **Unused GlobalSemaphores.** `semaphore[0]` and `[1]` ("the all-gather directions",
  `sliding_halo_exchange.hpp:89`) are required by the caller contract but not used by this op.
- **Unused includes (probable).** `sliding_halo_exchange.cpp:22-26` includes CCL command-stream headers; the
  helper's CCL symbols come from `ccl_common.hpp` and `ccl_host_datastructures.hpp` (`:15-16`).

## Questions for the user

1. **Readiness sheet.** Please confirm the `experimental/gemma4/sliding_ring_sdpa` row: `Concept`, `Is able to port?`,
   `TensorParameter relaxation`, `Known op issues`, `Porting Target`. This audit could not read it.
2. **Time-shared links.** Is a new M2 feature for sequential router sharing wanted, or should the op move to one
   worker per link that carries every exchange on that link?

## Recipe notes

- **Hybrid factory shape.** The recipe's concept rule does not cover a `MeshWorkloadFactoryConcept` shell around a
  `DescriptorMeshWorkloadAdapter`. By the rule it is `legacy (MeshWorkload)`, but the "PD step" is only to remove the
  shell. The same shape exists in other ops (`transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.cpp` and
  several `experimental/ccl/*_matmul*` factories use `descriptor_adapter_t::apply_descriptor`).
- **Missing-feature boundary.** Appendix A says a construct that is not listed is supported. The queue brief adds that a
  missing feature is "something the M2 host API cannot express at all". Sequential sharing of a fabric router is not in
  Appendix A, but the fabric-connection validator rejects it in every form, so this audit records it as a missing
  feature. Override 7 (a repeated `(src, dst, link)` is a config edge case) was written for one node with two slots,
  not for a designed multi-node hand-off.
- **Kernel GlobalSemaphore API.** Override 4a assumes every raw-pointer use of a GlobalSemaphore maps onto the binding.
  The collector's local atomic decrement (`sliding_halo_reader.cpp:254`) has no `experimental::GlobalSemaphore` method
  (only `wait`, `wait_min`, `set`, `value`, `address`; `tt_metal/hw/inc/api/dataflow/global_semaphore.h:37-58`).
- **Compute kernels and the Device 2.0 gate.** The gate says "every kernel", but the Device 2.0 guide is data movement
  only. This audit checked the compute kernel for data-movement idioms only.
- **Unreachable readiness sheet** and **provenance command**: as for the earlier ports on this branch.
