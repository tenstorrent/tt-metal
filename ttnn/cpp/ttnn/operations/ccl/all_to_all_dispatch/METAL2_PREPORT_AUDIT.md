# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch`

**Audited at `94bf191ce8f`.** The audit on `dbf8fd3bd30` (the `[Plan]` commit, same code as `5f28f745c0c`) was RED on
Device 2.0: both kernels used the Device 1.0 CB and NoC calls. One prerequisite commit cleared that gate:

- `94bf191ce8f` — the reader and the writer use `CircularBuffer` objects and the `Noc` object (`async_read` /
  `async_write` with tensor accessor, circular buffer and `CoreLocalMem` endpoints, the read / write / atomic barriers,
  the write flush). Transfer sizes, addresses and order are unchanged; CTAs and RTAs are unchanged.

- **`AllToAllDispatchDeviceOperation`** (`device/all_to_all_dispatch_device_operation.{hpp,cpp}`)
  - `AllToAllDispatchSparse` (`device/all_to_all_dispatch_program_factory.cpp`) — one factory with
    `create_workload_descriptor` (`:501`). It builds one program per mesh coordinate (`build_dispatch_program_descriptor`,
    `:99`) and two op-owned GlobalSemaphores per cache miss (`:514-517`). No `override_runtime_arguments`. Two modes
    selected by `operation_attributes.impl`: `FullPacket` (intermediate metadata CB `c_5`) and `PageByPage` (no `c_5`).
- Kernels (both in `device/kernels/dataflow/`, both referenced, both op-owned):
  - `reader_all_to_all_dispatch.cpp` — reads the mapping table, the indices and the input tokens into CBs, waits for the
    completion credits, and in `FullPacket` copies the intermediate metadata buffer to the metadata tensor.
  - `writer_all_to_all_dispatch.cpp` — sends each token to the local output or over the fabric, then sends the metadata
    (to the metadata tensor in `PageByPage`, to the intermediate buffer of every device in `FullPacket`) with the
    completion credits.
  - Census: `grep -rn "reader_all_to_all_dispatch\|writer_all_to_all_dispatch" ttnn/ tests/ models/` hits only this
    factory (`experimental/ccl/all_to_all_dispatch_metadata` has its own, differently named kernels).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` — from
`git log -1 … origin/akertesz/op-porting-recipe -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`. The recipe
docs are not in this checkout; the line was taken from the recipe branch ref.

**Branch-specific overrides applied** (this branch is newer than the recipe; given by the launching session):

1. GlobalSemaphore is supported. The Appendix A "GlobalSemaphore — UNSUPPORTED" entry is not applied.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`).
   This op owns two. The `ttnn_factory.md` "Feasibility gate" line "Op-owned `GlobalSemaphore`s → RED" is superseded.
3. Fabric connections replace `append_fabric_connection_rt_args`.
4. Sanctioned kernel-side exception: (a) the raw GlobalSemaphore address RTAs (reader RTA 5, writer RTAs 5 and 6) and
   their raw-pointer uses become `global_sem::` bindings; (b) the fabric-connection RTAs read by `build_from_args` (here
   through the moe_utils helper `open_direction_connections_async`, incl. the per-direction flags) become `fabric::`
   bindings.
5. Caller-supplied GlobalSemaphores: **not applicable** — both semaphores are created inside the factory, and there is no
   legacy `override_runtime_arguments`.
6. A binder node with no fabric target: build the sender only under `has_fabric_connection`, guard `open*()` / `close()`.
7. One node that talks to more than one destination: one `FabricConnectionSpec` per destination slot, all bound by the
   writer.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch` |
| **Overall** | **GREEN** after `94bf191ce8f` (TTNN readiness derived from code; sheet not read — see Questions) |
| **DOps / Factories** | `AllToAllDispatchDeviceOperation` → `AllToAllDispatchSparse` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** after `94bf191ce8f` (was broad Device 1.0 in both kernels; fixed by the prereq). The only Device 1.0 sites left are the GlobalSemaphore and fabric-connection sites that override 4 replaces, and `get_noc_addr(<L1 addr>)` for fabric payload / semaphore destinations. |
| *Prereqs* — Cross-op escapes | Ok — `⚠ workable` (moe_utils helpers take templated senders, `TensorAccessor`s and NoC addresses; see Team-only) |
| *Feature Support* — overall | GREEN — no Appendix A entry fires after override 1 |
| *Feature Support* — Variadic-CTA | Ok — no variable-index CTA reads |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Not read — sheet not reachable.** Derived from code: no blocking signal. |
| *TTNN Readiness* — Concept (current) | `WorkloadDescriptor` (`all_to_all_dispatch_device_operation.hpp:77`) |
| *TTNN Readiness* — Execution Model | `SPMD (+ per-device args)` — every coordinate has the same kernels and CBs; the CTAs `src_mesh_id`, `src_chip_id`, `linearized_mesh_coord`, the `DIRECTIONS` define and the fabric targets depend on the coordinate |
| *TTNN Readiness* — Custom hash | No (default reflection hash over `attribute_values()` and `tensor_args_t`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`all_to_all_dispatch_nanobind.cpp` binds only the op) |
| *TTNN Readiness* — Op-owned tensors | No. Op-owned **GlobalSemaphores**: yes, two (`all_to_all_dispatch_program_factory.cpp:514-517`) — supported by override 2 |
| *TTNN Readiness* — Target concept | `MeshWorkloadSpecFactoryConcept`, no override (derived from code; sheet not read) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Case 1 for all six bindings (input, indices, mapping, metadata on the reader; output, metadata on the writer) |
| *TTNN Readiness* — TensorParameter relaxation | Not read — sheet not reachable. Code shows no need (every `TensorSpec` is in the default hash). |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | 1P+1C on `c_0`, `c_1`, `c_2`; self-loop on `c_3` and `c_4` (writer); **conditional DFB** `c_5` (live in `FullPacket`, absent in `PageByPage`), 1P+1C there; no multi-binding; no dead CB |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`).

Every gate-bearing subject clears: Device 2.0 (after `94bf191ce8f`, with override 4), feature compatibility (with override
1), offset base pointers, and the TensorAccessor 3rd argument (no sites). The TTNN factory concept gate is cleared **from the
code only**: this session could not read the readiness sheet (Question 1). If the sheet says `no`, that cell overrides this
verdict.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Not read.** No tool in this session can fetch the readiness sheet. Values
  from the code:
  - `Concept` = `WorkloadDescriptor`: `AllToAllDispatchSparse::create_workload_descriptor`
    (`all_to_all_dispatch_device_operation.hpp:77`, `all_to_all_dispatch_program_factory.cpp:501`). No `create_descriptor`.
  - `Custom hash`: none. The default hash covers `attribute_values()` (`worker_core_range_set`, `output_mem_config`,
    `axis`, `num_links`, `topology`, `impl`, `output_concat_dim`) and `tensor_args_t` (each `Tensor` with its
    `TensorSpec`, plus the optional preallocated outputs).
  - `get_dynamic_runtime_args`: none. `override_runtime_arguments`: none. `Pybind descriptor`: none.
  - `Op-owned tensors?`: no (`WorkloadDescriptor::buffers` not set). `WorkloadDescriptor::semaphores` holds the two
    GlobalSemaphores (`.cpp:522-523`), created with `create_global_semaphore(mesh_device, worker_core_range_set, 0)` and
    followed by `distributed::Synchronize` (`.cpp:518`).
  - `Secretly SPMD Workload?`: no — coordinate-derived CTAs (`.cpp:351-352,367`) and the `DIRECTIONS` define (`.cpp:388`).
    Why mesh-shaped: per-coordinate CTAs, define and fabric targets.
  - `Smuggled pointer`: none for tensors (all five are `Buffer*` RTAs). The GlobalSemaphore addresses are raw RTAs (reader
    `.cpp:439`, writer `.cpp:454-455`), stable for the cached workload because the semaphores are op-owned; override 4a
    replaces them.
  - Factory set: one factory.
- **Device 2.0 (every kernel used):** **GREEN after `94bf191ce8f`** (with override 4). Before that commit **both kernels were
  broad Device 1.0**: reader — free `cb_reserve_back` / `cb_push_back` / `get_write_ptr(cb_id)` on three CBs and on `c_5`,
  `noc_async_read_page`, `noc_async_read_barrier`, raw `noc_async_write` + `noc_async_write_barrier`; writer — free
  `cb_wait_front` / `cb_pop_front` / `get_read_ptr` / `get_write_ptr` on all six CBs, raw `noc_async_write`,
  `noc_async_writes_flushed`, `noc_async_write_barrier`, `noc_async_atomic_barrier`. The prereq moved them to
  `CircularBuffer`, `Noc` and `CoreLocalMem` (no arg change). Sites left on purpose, all replaced by the port:

  | File | Line (at `94bf191ce8f`) | Call | Override |
  |---|---|---|---|
  | `reader_all_to_all_dispatch.cpp` | 67, 123-132 | raw final GlobalSemaphore address RTA, `noc_semaphore_wait` / `noc_semaphore_set` on it | 4a |
  | `writer_all_to_all_dispatch.cpp` | 155-156, 223, 235-236, 320 | raw final / init GlobalSemaphore address RTAs, `get_noc_addr(<sem addr>)`, `noc_semaphore_wait` / `noc_semaphore_set` | 4a |
  | `writer_all_to_all_dispatch.cpp` | 49, 395, 458 | local `noc_semaphore_inc` on the NoC address of the final semaphore | 4a (kept: the kernel `GlobalSemaphore` has no increment) |
  | `writer_all_to_all_dispatch.cpp` | 164, 167, 220, 467 | `DIRECTIONS`, `open_direction_connections_async` (→ `build_from_args`), `_barrier`, `close_direction_connections` | 4b |

  Not a holdover: packet-header pointers taken from `c_3` (fabric API usage), `get_noc_addr(<metadata buffer L1 addr>)`
  (`writer…:405`, a fabric payload destination that the moe_utils helpers take as `uint64_t`), and the raw `uint8_t*` /
  `uint16_t*` reads of CB memory (`writer…:234,249,255`). No sanctioned free function is in use. The moe_utils helpers
  that the writer calls use the fabric API (`tt_metal/fabric/…`), not raw NoC calls, on the paths this op compiles
  (`noc_async_writes_flushed` in `fabric_send_noc_unicast` is only on the `FabricMuxV2Sender` branch).
- **Feature compatibility:** GREEN — no gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no GCB type, no `.global_circular_buffer` field, no remote CB |
  | CBDescriptor `address_offset` (non-zero) | N/A | no CB sets `address_offset` |
  | GlobalSemaphore | **Supported on this branch (override 1)** — not RED | `create_global_semaphore` at `all_to_all_dispatch_program_factory.cpp:514-517`; `const GlobalSemaphore&` parameters at `:104-105`; include at `all_to_all_dispatch_device_operation.hpp:15-16`. Under the recipe as written this entry would be RED. |

  Not in Appendix A, so supported: `append_fabric_connection_rt_args` (`.cpp:472`) → fabric connections (override 3), one
  per neighbor direction per sender core (override 7).
- **CB endpoints (GATE-free):** see Port-work summary. Nothing blocks.
- **Offset base pointers:** GREEN. Every tensor address RTA is a clean base passed as `Buffer*`: reader RTA 0-4
  (`.cpp:434-438`), writer RTA 0-4 (`.cpp:484-488`; the `mesh_buffer().address()` values at `:449-453` are placeholders it
  replaces). The op is not in the offset-base-pointer triage tables. The `base + offset` NoC addresses in the writer
  (`noc_core_offset_md_write_addr`, `writer…:405-406`) are on the **intermediate CB**, not on a tensor, and are computed in
  the kernel — not a host fold.
- **TensorAccessor 3rd argument:** N/A — no accessor in the op passes a 3rd argument
  (`reader_all_to_all_dispatch.cpp:71-74`, `writer_all_to_all_dispatch.cpp:207-208` are two-argument). The reader reads
  pages with the accessor's own `get_aligned_page_size()` (the prereq made that explicit where `noc_async_read_page`
  used it implicitly).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; clean bases, all fed to a `TensorAccessor`):
  - input → reader (`reader…:71`, RTA0) — **Case 1**
  - indices → reader (`:72`, RTA1) — **Case 1**
  - mapping → reader (`:73`, RTA2) — **Case 1**
  - metadata → reader (`:74`, RTA4; `FullPacket` copy to the metadata tensor) — **Case 1**
  - output → writer (`writer…:207`, RTA3; local `Noc::async_write` and moe_utils fabric senders) — **Case 1**
  - metadata → writer (`:208`, RTA4; `PageByPage` local write and fabric senders) — **Case 1**
  - Not bound (address read but unused): output on the reader (RTA3), input / indices / mapping on the writer (RTA0-2).
    Their `TensorAccessorArgs` are parsed but no accessor is built.
- **TensorParameter relaxation:** none (code-derived; sheet not read).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (every sender core has one reader and one writer instance; same census on every core):

  | CB | Touchers on a node | Disposition |
  |---|---|---|
  | `c_0` input (`.cpp:250`) | reader FIFO-produces; writer FIFO-consumes (`wait_front` / `get_read_ptr` / `pop_front`) | 1P+1C (legal) |
  | `c_1` indices (`.cpp:259`) | reader FIFO-produces; writer FIFO-consumes, and peeks the base once (`get_read_ptr` before the first wait) for the metadata sends | 1P+1C (legal) |
  | `c_2` mapping (`.cpp:268`) | reader FIFO-produces (`reserve_back(mapping_pages)` / `push_back`); writer FIFO-consumes | 1P+1C (legal) |
  | `c_3` packet header (`.cpp:277`) | writer only (`get_read_ptr` once; no FIFO call at all) | **self-loop** (DM) |
  | `c_4` send preparation (`.cpp:286`) | writer only (`get_write_ptr`, `async_write_zeros`, raw byte reads/writes) | **self-loop** (DM) |
  | `c_5` metadata buffer (`.cpp:295-305`, `FullPacket` only) | writer peeks `get_read_ptr` (local loopback write; NoC destination of the remote sends); reader peeks `get_write_ptr` (copy to the metadata tensor). Both role-free. | **conditional DFB**: live under `FullPacket` → 1P+1C (writer PRODUCER, reader CONSUMER); absent under `PageByPage` (no CB in legacy either) |

  No hidden second writer in the multi-binding sense: `c_5` is also written **by the writers of the other devices**
  through the fabric (cross-device NoC writes to the same L1 address), which is not a binding on this node. No dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes):** none.
- **Conditional `c_5`.** The kernels select the `c_5` code through the CTA `write_page_by_page` and `if constexpr`; the
  port promotes it to a define (`FULL_PACKET`) on both kernels, per "Conditional / optional resource bindings".
- **Cross-device CB address.** In `FullPacket` the writer sends metadata into the `c_5` buffer of every other device at
  this device's local `c_5` address (`writer…:405-406`). This relies on every per-coordinate program placing `c_5` at the
  same L1 address. Legacy relied on the same thing (same CB set on every coordinate). The port keeps the same DFB set on
  every coordinate.
- **Quasar debt (not a Gen1 blocker):** the two self-loop CBs are DM self-loops (`dm_self_loop_dfbs.md`); `c_5` has no
  credit calls (`sync_free_dfbs.md`).
- **Cross-op / shared kernels:** none. Both kernels are op-owned → convert in place. No `_metal2` fork.
- **RTA varargs:** none after override 4b (the per-direction fabric-connection tail read through `rt_args_idx` is replaced
  by four `fabric::` bindings).
- **Multi-destination fabric node (override 7):** each sender core opens one connection per neighbor direction, link index
  = core index. Four `FabricConnectionSpec`s (`east`, `west`, `north`, `south`), all bound by the writer.
- **Final GlobalSemaphore bound by two kernels.** The final (cross-device) semaphore is read by the reader (wait / set) and
  written by the writer (local increment and remote atomic increments); the init semaphore only by the writer.

## Team-only

- **Out-of-directory coupling & donor shape:** `⚠ workable`.

  | Op kernel | Donor | Class | Shape |
  |---|---|---|---|
  | reader | `ccl/common/kernels/moe_utils.hpp` (`ReplicateGroup`) | cross-family (ccl) | enum only — ✓ |
  | writer | `moe_utils.hpp` — `fabric_send_chip_unicast_noc_unicast(_1d)`, `fabric_send_chip_unicast_noc_unicast_with_semaphore(_1d)`, `l1_only_fabric_send_chip_unicast_noc_unicast_with_semaphore(_1d)`, `send_init_semaphore_to_configured_targets`, `fabric_multicast_bidirectional_atomic_inc_1d` | cross-family (ccl) | `TensorAccessor` (Shape 1) ✓; `std::array<WorkerToFabricEdmSender, 4>&` ✓; `uint64_t` NoC semaphore / L1 address ⚠ — workable on this branch: the kernel builds it from the binding as `get_noc_addr(gs.address())`, as `point_to_point` does |
  | writer | `moe_utils.hpp` — `open_direction_connections_async` | cross-family (ccl) | `build_from_args` on RTAs — replaced by override 4b (`moe_utils.hpp` is shared and not edited) |
  | writer | `moe_utils.hpp` — `open_direction_connections_barrier`, `close_direction_connections` | cross-family (ccl) | take a runtime `std::array<bool, 4>` and the senders; no RTA — ✓, usable with a direction array built from `has_fabric_connection` |
  | writer | `moe_utils.hpp` — `is_configured_target`, `is_1d_topology` | cross-family (ccl) | pure — ✓ |
  | writer | `data_movement/common/kernels/common.hpp` | cross-family | included, no function called |
  | writer | `tt_metal/fabric/hw/inc/…` | `tt_metal/*` | no concern |

  Borrowed kernel files: none. Lent kernel files: none.
- **Relaxation candidates:** none.
- **TTNN factory analysis:** target `MeshWorkloadSpecFactoryConcept`, no override. Ported-from entry point:
  `create_workload_descriptor` (`all_to_all_dispatch_program_factory.cpp:501`), one program per coordinate of
  `tensor_coords`. Per coordinate: `src_mesh_id`, `src_chip_id`, the linearized mesh index, the neighbors / `DIRECTIONS`
  define and the fabric targets. Op-owned GlobalSemaphores (two). No custom hash, no `get_dynamic_runtime_args`, no
  pybound `create_descriptor`.

## Misc anomalies  *(team-only, non-gating)*

- **Shared CTA vector.** The reader and the writer receive the same 37 positional CTAs (`.cpp:322-375`,
  `writer_compile_time_args = reader_compile_time_args`); each kernel reads a subset, and several CTAs that a kernel
  declares are never used (reader: `src_mesh_id`, `src_chip_id`, `aligned_metadata_page_size`; writer: the five
  `*_pages`, most page sizes, `hidden_size`, `batch_size`, `experts`, `num_links`, `src_mesh_id`). Harmless.
- **Unused buffer addresses.** Both kernels read all five tensor addresses; the reader never uses the output, the writer
  never uses input / indices / mapping (`reader…:64`, `writer…:150-152`).
- **Dead locals.** Reader `my_device_offset` (`reader…:117`) and, before the prereq, `base_indices_addr` (removed by the
  prereq; it was unused).
- **`AXIS` on the reader is always defined.** The factory always emits `AXIS` for the reader, `-1` without an axis
  (`.cpp:378`), so the reader's `#ifdef AXIS` is always true and `ReplicateGroup(-1)` = `NONE` covers the no-axis case;
  the `#else` branch is dead. The writer gets `AXIS` only with an axis.
- **Packet-header CB.** `c_3` is never reserved or pushed; the writer takes `get_read_ptr()` once and uses three headers
  from that base.
- **2-device Ring along the axis.** With `Topology::Ring` and two devices on the dispatch axis, `get_neighbors` returns the
  same neighbor for both directions, and the legacy factory appends two fabric connections from one core to the same
  `(src, dst, link)`. A fabric router accepts one connected worker at a time, so that config is suspect in legacy too. In
  Metal 2.0 the spec validator rejects it (`ValidateFabricConnectionUsage`). Config edge case, not a blocker (override 7).

## Questions for the user

1. **Readiness sheet.** Please confirm the `ccl/all_to_all_dispatch` row: `Is able to port?`, `TensorParameter relaxation`,
   `Known op issues`, `Porting Target`. This audit could not read them; the GREEN verdict assumes they agree with the code.

## Recipe notes

- **Op-owned GlobalSemaphores.** `ttnn_factory.md` "Feasibility gate" still says op-owned GlobalSemaphores are RED; on this
  branch they are supported (override 2).
- **Device 2.0 prereq done in the queue.** The recipe routes a Device 2.0 RED to another team; here the launching session
  asked for the fix as a `[Prereq]` commit in the op's own directory.
- **`noc_async_read_page` has no direct `Noc` twin.** It takes its size from the accessor (`get_aligned_page_size()`);
  the `Noc::async_read` form needs the size explicitly. The migration guide's examples all pass a known tile size; a
  line on the page-size source would help.
- **Unreachable readiness sheet.** As for `point_to_point` and `all_to_all_combine`: the primary columns were derived from
  code and the derived verdict is unconfirmed.
- **Provenance command** assumes the recipe docs are in the checkout; here the line came from the recipe branch ref.
