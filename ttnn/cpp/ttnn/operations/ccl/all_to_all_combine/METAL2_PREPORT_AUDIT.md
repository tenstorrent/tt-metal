# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine`

**Audited at `66b645994d5`.** The audit on `377d1c00652` was RED on Device 2.0 (the writer used the Device 1.0 CB and NoC
calls). One prerequisite commit cleared that gate:

- `66b645994d5` — the writer uses `CircularBuffer` objects and the `Noc` object for the local output write, the write flush
  and barrier, and the atomic barriers. The reader was already on the Device 2.0 API.

- **`AllToAllCombineDeviceOperation`** (`device/all_to_all_combine_device_operation.{hpp,cpp}`)
  - `AllToAllCombineFromSparse` (`device/all_to_all_combine_program_factory.cpp`) — one factory with
    `create_workload_descriptor` (`:334`). It builds one program per mesh coordinate (`build_combine_program_descriptor`,
    `:28`) and two op-owned GlobalSemaphores per cache miss (`:347-350`). No `override_runtime_arguments`.
- Kernels (both in `device/kernels/dataflow/`, both referenced):
  - `reader_all_to_all_combine.cpp` — reads the expert mapping, the metadata and the input tokens into CBs. **Also bound by
    `data_movement/moe_expert_token_remap`** (`moe_expert_token_remap_program_factory.cpp:176`).
  - `writer_all_to_all_combine.cpp` — writes each token to the local output or sends it over the fabric, then runs the
    completion barrier. Op-owned; no other factory binds it.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` — from
`git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`. The recipe
docs are not in this checkout; the line was taken from the recipe branch ref.

**Branch-specific overrides applied** (this branch is newer than the recipe; given by the launching session):

1. GlobalSemaphore is supported. The Appendix A "GlobalSemaphore — UNSUPPORTED" entry is not applied.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`).
   This op owns two. The `ttnn_factory.md` "Feasibility gate" line "Op-owned `GlobalSemaphore`s → RED" is superseded.
3. Fabric connections replace `append_fabric_connection_rt_args`.
4. Sanctioned kernel-side exception: (a) the raw GlobalSemaphore address RTAs and their raw-pointer uses become
   `global_sem::` bindings; (b) the fabric-connection RTAs read by `build_from_args` (here through the moe_utils helper
   `open_direction_connections_async`, incl. the per-direction flags) become `fabric::` bindings.
6. A binder node with no fabric target: build the sender only under `has_fabric_connection`, guard `open*()` / `close()`.
7. One node that talks to more than one destination: one `FabricConnectionSpec` per destination slot, all bound by the kernel.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine` |
| **Overall** | **GREEN** after `66b645994d5` (TTNN readiness derived from code; sheet not read — see Questions) |
| **DOps / Factories** | `AllToAllCombineDeviceOperation` → `AllToAllCombineFromSparse` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** after `66b645994d5` (was broad Device 1.0 in the writer; fixed by the prereq). The only Device 1.0 sites left are the GlobalSemaphore and fabric-connection sites that override 4 replaces. |
| *Prereqs* — Cross-op escapes | Ok — `⚠ workable` (moe_utils helpers take templated senders and NoC addresses; see Team-only) |
| *Feature Support* — overall | GREEN — no Appendix A entry fires after override 1 |
| *Feature Support* — Variadic-CTA | Ok — no variable-index CTA reads |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Not read — sheet not reachable.** Derived from code: no blocking signal. |
| *TTNN Readiness* — Concept (current) | `WorkloadDescriptor` (`all_to_all_combine_device_operation.hpp:66`) |
| *TTNN Readiness* — Execution Model | `SPMD (+ per-device args)` — every coordinate has the same kernels and CBs; the writer CTAs `src_chip_id` / `linearized_mesh_coord`, the reader CTA `linearized_mesh_coord`, the `DIRECTIONS` define and the fabric targets depend on the coordinate |
| *TTNN Readiness* — Custom hash | No (default reflection hash over `attribute_values()` and `tensor_args_t`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`all_to_all_combine_nanobind.cpp` binds only the op) |
| *TTNN Readiness* — Op-owned tensors | No. Op-owned **GlobalSemaphores**: yes, two (`all_to_all_combine_program_factory.cpp:347-350`) — supported by override 2 |
| *TTNN Readiness* — Target concept | `MeshWorkloadSpecFactoryConcept`, no override (derived from code; sheet not read) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Case 1 for all four bindings |
| *TTNN Readiness* — TensorParameter relaxation | Not read — sheet not reachable. Code shows no need (every `TensorSpec` is in the default hash). |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | 1P+1C on `c_0`, `c_2`, `c_3`; self-loop on `c_1` (reader) and `c_4` (writer); no multi-binding; no dead CB |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`).

Every gate-bearing subject clears: Device 2.0 (after `66b645994d5`, with override 4), feature compatibility (with override
1), offset base pointers, and the TensorAccessor 3rd argument (no sites). The TTNN factory concept gate is cleared **from the
code only**: this session could not read the readiness sheet (Question 1). If the sheet says `no`, that cell overrides this
verdict.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Not read.** No tool in this session can fetch the readiness sheet. Values
  from the code:
  - `Concept` = `WorkloadDescriptor`: `AllToAllCombineFromSparse::create_workload_descriptor`
    (`all_to_all_combine_device_operation.hpp:66`, `all_to_all_combine_program_factory.cpp:334`). No `create_descriptor`.
  - `Custom hash`: none. The default hash covers `attribute_values()` (`output_mem_config`, `axis`, `num_links`, `topology`,
    `locally_reduced`, `worker_core_range_set`, `output_shard_dim`) and `tensor_args_t` (each `Tensor` with its `TensorSpec`).
  - `get_dynamic_runtime_args`: none. `override_runtime_arguments`: none. `Pybind descriptor`: none.
  - `Op-owned tensors?`: no (`WorkloadDescriptor::buffers` not set). `WorkloadDescriptor::semaphores` holds the two
    GlobalSemaphores (`.cpp:355-356`), created with `create_global_semaphore(mesh_device, worker_core_range_set, 0)` and
    followed by `distributed::Synchronize` (`.cpp:351`).
  - `Secretly SPMD Workload?`: no — coordinate-derived CTAs (`.cpp:174,185,212,218`) and the `DIRECTIONS` define (`.cpp:238`).
    Why mesh-shaped: per-coordinate CTAs, define and fabric targets.
  - `Smuggled pointer`: none for tensors (all four are `Buffer*` RTAs). The two GlobalSemaphore addresses are raw writer RTAs
    (`.cpp:300-301`), stable for the cached workload because the semaphores are op-owned; override 4a replaces them.
  - Factory set: one factory.
- **Device 2.0 (every kernel used):** **GREEN after `66b645994d5`** (with override 4). Before that commit the writer was
  **broad Device 1.0**: free-function `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` /
  `get_read_ptr(cb_id)` on all four of its CBs, raw `noc_async_write`, `noc_async_writes_flushed`, `noc_async_write_barrier`
  and `noc_async_atomic_barrier`. The prereq moved them to `CircularBuffer` and `Noc` (no arg change). The reader was already
  Device 2.0 (`CircularBuffer`, `Noc::async_read`, `CoreLocalMem`). Sites left on purpose, all replaced by the port:

  | File | Line | Call | Override |
  |---|---|---|---|
  | `writer_all_to_all_combine.cpp` | 160-161, 187, 202-203 | raw init/final GlobalSemaphore address RTAs, `get_noc_addr(init_semaphore_addr)`, `noc_semaphore_wait` / `noc_semaphore_set` | 4a |
  | `writer_all_to_all_combine.cpp` | 288, 86, 313-316 | `get_noc_addr(global_semaphore_addr)`, the local `noc_semaphore_inc` on that address (2D / no-axis branch), `noc_semaphore_wait` / `noc_semaphore_set` | 4a |
  | `writer_all_to_all_combine.cpp` | 156, 158, 166, 189, 308 | `DIRECTIONS` flags, `open_direction_connections_async` (→ `build_from_args`), `_barrier`, `close_direction_connections` | 4b |

  No sanctioned free functions are in use. Packet-header pointers taken from the `c_4` CB are fabric API usage, not a
  Device 2.0 holdover. The moe_utils helpers that the writer calls use the fabric API (`tt_metal/fabric/…`), not raw NoC
  calls, on the paths this op compiles (`noc_async_writes_flushed` in `fabric_send_noc_unicast` is only on the
  `FabricMuxV2Sender` branch).
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type, no `.global_circular_buffer` field, no remote CB |
  | CBDescriptor `address_offset` (non-zero) | N/A | no CB sets `address_offset` |
  | GlobalSemaphore | **Supported on this branch (override 1)** — not RED | `create_global_semaphore` at `all_to_all_combine_program_factory.cpp:347-350`; `const GlobalSemaphore&` parameters at `:33-34`; include at `all_to_all_combine_device_operation.hpp:15`. Under the recipe as written this entry would be RED. |

  Not in Appendix A, so supported: `append_fabric_connection_rt_args` (`.cpp:315`) → fabric connections (override 3), one
  per neighbor direction per sender core (override 7).
- **CB endpoints (GATE-free):** see Port-work summary. Nothing blocks.
- **Offset base pointers:** GREEN. Every address RTA is a clean base passed as `Buffer*`: reader RTA0 mapping, RTA1
  metadata, RTA2 input (`.cpp:286-288`); writer RTA0 output (`.cpp:321`; the `mesh_buffer().address()` at `:299` is a
  placeholder it replaces). The op is not in the offset-base-pointer triage tables.
- **TensorAccessor 3rd argument:** N/A — no accessor in the op passes a 3rd argument (`reader_all_to_all_combine.cpp:81-83`,
  `writer_all_to_all_combine.cpp:168` are all two-argument). No kernel reads `get_aligned_page_size()`; every transfer size is
  an explicit CTA (`data_size_bytes`, `mapping_page_size_bytes`, `metadata_page_size_bytes`).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; clean bases, all fed to a `TensorAccessor`):
  - mapping → reader (`reader_all_to_all_combine.cpp:82`, RTA0) — **Case 1**
  - metadata → reader (`:81`, RTA1) — **Case 1**
  - input → reader (`:83`, RTA2) — **Case 1**
  - output → writer (`writer_all_to_all_combine.cpp:168`, RTA0; used by `Noc::async_write` and by the moe_utils fabric
    senders through `to_noc_unicast_write(…, addrgen, …)`) — **Case 1**
- **TensorParameter relaxation:** none (code-derived; sheet not read).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (every sender core has one reader and one writer instance; same census on every core and in every config):

  | CB | Touchers on a node | Disposition |
  |---|---|---|
  | `c_0` data (`.cpp:110`) | reader FIFO-produces (`reserve_back` / `push_back`, NoC read target); writer FIFO-consumes (`wait_front` / `get_read_ptr` / `pop_front`) | 1P+1C (legal) |
  | `c_1` mapping (`.cpp:122`) | reader only (`reserve_back`, `get_write_ptr`, `push_back`; then NoC read target and raw reads) | **self-loop** (DM) |
  | `c_2` local experts (`.cpp:138`) | reader FIFO-produces (raw writes through `get_write_ptr`, then `push_back`); writer FIFO-consumes | 1P+1C (legal) |
  | `c_3` metadata (`.cpp:150`) | reader FIFO-produces (and raw-reads its own entry — one toucher); writer FIFO-consumes | 1P+1C (legal) |
  | `c_4` packet header (`.cpp:164`) | writer only (`reserve_back(3)`, `get_read_ptr`, `push_back(3)`, never popped) | **self-loop** (DM) |

  No hidden second writer: no kernel writes a CB it does not FIFO-produce. No dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes):** none.
- **Quasar debt (not a Gen1 blocker):** the two self-loop CBs are DM self-loops (`dm_self_loop_dfbs.md`).
- **Cross-op / shared kernels:** the reader is *lent*: `moe_expert_token_remap` binds it too. No `_metal2` fork exists beside
  it → the port creates `reader_all_to_all_combine_metal2.cpp` beside the original (rung 2) and adds the pointer comment.
  Sunset list (not authorization to convert in place): `data_movement/moe_expert_token_remap`. The writer is op-owned →
  convert in place. `scripts/check_kernel_cb_ptrs.py:52` lists the writer as a reviewed-safe exception; after the port the
  writer has no free `get_read_ptr`, so that entry is stale but harmless (outside the op directory; not edited).
- **RTA varargs:** none after override 4b (the per-direction fabric-connection tail read through `rt_arg_count` is replaced
  by four `fabric::` bindings).
- **Multi-destination fabric node (override 7):** each sender core opens one connection per neighbor direction, link index =
  core index. Four `FabricConnectionSpec`s (`east`, `west`, `north`, `south`, the `{E, W, N, S}` order of
  `ccl/common/host/moe_utils.cpp` `get_neighbors` and of the writer's `std::array<WorkerToFabricEdmSender, 4>`), all bound
  by the writer.
- **GlobalSemaphore, fabric connection:** see the brief.

## Team-only

- **Out-of-directory coupling & donor shape:** `⚠ workable`.

  | Op kernel | Donor | Class | Shape |
  |---|---|---|---|
  | reader | `ccl/common/kernels/moe_utils.hpp` (`find_if`) | cross-family (ccl) | L1 pointer + value — ✓ |
  | reader | `data_movement/common/kernels/common.hpp`, `api/debug/dprint_pages.h` | cross-family / `tt_metal/*` | included, no function called |
  | writer | `moe_utils.hpp` — `fabric_send_chip_unicast_noc_unicast(_1d)`, `fabric_send_chip_unicast_noc_unicast_semaphore_only`, `send_init_semaphore_to_configured_targets`, `fabric_multicast_bidirectional_atomic_inc_1d` | cross-family (ccl) | `TensorAccessor` (Shape 1) ✓; `std::array<SenderType, 4>&` templated senders ✓; `uint64_t` NoC GlobalSemaphore address ⚠ — workable on this branch: the kernel builds it from the binding as `get_noc_addr(gs.address())`, as `point_to_point` does |
  | writer | `moe_utils.hpp` — `open_direction_connections_async` / `_barrier`, `close_direction_connections` | cross-family (ccl) | `build_from_args` on RTAs — replaced by override 4b; the writer opens and closes the senders from the bindings itself (`moe_utils.hpp` is shared and not edited) |
  | writer | `moe_utils.hpp` — `find_if`, `is_configured_target`, `is_1d_topology`, `ReplicateGroup` | cross-family (ccl) | pure — ✓ |
  | writer | `tt_metal/fabric/hw/inc/…` | `tt_metal/*` | no concern |

  Borrowed kernel files: none. Lent kernel file: the reader (above).
- **Relaxation candidates:** none.
- **TTNN factory analysis:** target `MeshWorkloadSpecFactoryConcept`, no override. Ported-from entry point:
  `create_workload_descriptor` (`all_to_all_combine_program_factory.cpp:334`), one program per coordinate of
  `tensor_coords`. Per coordinate: `src_chip_id`, the linearized mesh index (reader CTA 8, writer CTA 15), the neighbors /
  `DIRECTIONS` define and the fabric targets. Op-owned GlobalSemaphores (two). No custom hash, no `get_dynamic_runtime_args`,
  no pybound `create_descriptor`.

## Misc anomalies  *(team-only, non-gating)*

- **Recomputed index.** The factory computes the linearized mesh index twice (`flat_mesh_idx` at `.cpp:174` for the
  reader, `get_linearized_index(...)` again at `:218` for the writer); harmless.
- **Packet-header CB page size.** `c_4` uses `CLIENT_INTERFACE_SIZE` (3280 B) pages (`.cpp:165-170`), while the writer
  places three `sizeof(PACKET_HEADER_TYPE)` headers back to back from the CB base (`writer_all_to_all_combine.cpp:180-186`).
  The CB is far larger than it needs to be.
- **Debug include.** `reader_all_to_all_combine.cpp:14` includes `api/debug/dprint_pages.h`, unused.
- **2-device Ring along the axis.** With `Topology::Ring` and two devices on the dispatch axis, `get_neighbors` returns the
  same neighbor for both directions, and the legacy factory appends two fabric connections from one core to the same
  `(src, dst, link)`. A fabric router accepts one connected worker at a time, so that config is suspect in legacy too. In
  Metal 2.0 the spec validator rejects it (`fabric_connection.cpp` `ValidateFabricConnectionUsage`). Config edge case, not a
  blocker (override 7).

## Questions for the user

1. **Readiness sheet.** Please confirm the `ccl/all_to_all_combine` row: `Is able to port?`, `TensorParameter relaxation`,
   `Known op issues`, `Porting Target`. This audit could not read them; the GREEN verdict assumes they agree with the code.

## Recipe notes

- **Op-owned GlobalSemaphores.** `ttnn_factory.md` "Feasibility gate" still says op-owned GlobalSemaphores are RED; on this
  branch they are supported (override 2).
- **Device 2.0 prereq done in the queue.** The recipe routes a Device 2.0 RED to another team; here the launching session
  asked for the fix as a `[Prereq]` commit in the op's own directory.
- **Unreachable readiness sheet.** As for `point_to_point`: the primary columns were derived from code and the derived
  verdict is unconfirmed.
- **Provenance command** assumes the recipe docs are in the checkout; here the line came from the recipe branch ref.
