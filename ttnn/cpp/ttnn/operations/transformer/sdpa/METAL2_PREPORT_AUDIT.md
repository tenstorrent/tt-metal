# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/transformer/sdpa/`

This op directory bundles **five** independent `DeviceOperation`s, each with a single program factory. All five are on the `ProgramDescriptor` API. They share a large pool of kernel files (`device/kernels/`). Audited together because they share kernels and the `device/` directory; per-DeviceOperation attribution is retained throughout because findings differ sharply.

- **`SDPAOperation`** (`sdpa_device_operation.{hpp,cpp}`)
  - `SDPAProgramFactory` (`sdpa_program_factory.cpp`) — single `ProgramDescriptor`
- **`JointSDPADeviceOperation`** (`joint_sdpa_device_operation.{hpp,cpp}`)
  - `JointSDPAProgramFactory` (`joint_sdpa_program_factory.cpp`) — single `ProgramDescriptor`
- **`RingDistributedSdpaDeviceOperation`** (`ring_distributed_sdpa_device_operation.{hpp,cpp}`)
  - `RingDistributedSdpaProgramFactory` (`ring_distributed_sdpa_program_factory.cpp`) — `WorkloadDescriptor` (one program per mesh coord)
- **`RingJointSDPADeviceOperation`** (`ring_joint_sdpa_device_operation.{hpp,cpp}`)
  - `RingJointSDPAProgramFactory` / `RingJointSDPAMeshWorkloadFactory` (`ring_joint_sdpa_program_factory.cpp`) — `WorkloadDescriptor` (one program per mesh coord); embeds a fused CCL all-gather
- **`ExpRingJointSDPADeviceOperation`** (`exp_ring_joint_sdpa_device_operation.{hpp,cpp}`)
  - `ExpRingJointSDPAProgramFactory` (`exp_ring_joint_sdpa_program_factory.cpp`) — `WorkloadDescriptor` (one program per mesh coord); embeds a fused fabric-mux all-gather

> **Note on the original scope.** The task brief named "~6 program factories (… SparseSDPA …)". There is **no `SparseSDPA` factory** in this directory — the only `sparse` token is an identifier in `device/kernels/compute/compute_streaming.hpp`. The directory has exactly **five** factories. There are ~29 kernel/header files under `device/kernels/`; each was audited against the factory that references it (kernel references followed, not directory boundaries).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

**Recipe docs:** `776151aeca8 2026-06-24 docs(metal2): clarify SPSC face-(b) producer-as-consumer, aliased-vs-same-FIFO, no-portable-subset`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/transformer/sdpa/` |
| **Overall** | **RED** (config-scoped) — clean subset exists |
| **DOps / Factories** | `SDPAOperation`→`SDPAProgramFactory` · `JointSDPADeviceOperation`→`JointSDPAProgramFactory` · `RingDistributedSdpaDeviceOperation`→`RingDistributedSdpaProgramFactory` · `RingJointSDPADeviceOperation`→`RingJointSDPAProgramFactory` · `ExpRingJointSDPADeviceOperation`→`ExpRingJointSDPAProgramFactory` |
| *Prereqs* — ProgramDescriptor | **Yes** (all five factories) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** for SDPA / Joint / RingDistributed / RingJoint kernels (incl. CCL all-gather donor). **Open question** for ExpRingJoint (fabric-mux raw-L1 sem idiom) |
| *Prereqs* — Cross-op escapes | Ok — CCL all-gather donor is Device-2.0 clean; coupled only to the GlobalSemaphore gate |
| *Feature Support* — overall | **RED** (GlobalSemaphore) — confined to RingJoint + ExpRingJoint |
| *Feature Support* — Variadic-CTA | Ok (no CTA varargs anywhere) |
| *TTNN Readiness* — Op-owned tensors | **No** (all `create_device_tensor` calls produce *declared output* tensors inside `create_output_tensors`; no scratch/intermediate) |
| *TTNN Readiness* — MeshWorkload needed | **No** for SDPA & Joint (single-program). **Yes (genuine)** for RingDistributed / RingJoint / ExpRingJoint — per-mesh-coord forward/backward/device-index for the ring; genuine cross-device coordination, not an op-owned-tensor artifact |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `sdpa_nanobind.cpp` binds only the user-facing op functions (`bind_function<…>`); no `ProgramFactory`/`DeviceOperation` class binding |
| *TTNN Readiness* — Other risky pybind | **None** |
| *TTNN Readiness* — Custom hash | **Yes → delete** (3 of 5): `SDPAOperation`, `RingJointSDPADeviceOperation`, `ExpRingJointSDPADeviceOperation` (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | **Yes**: `RingJointSDPAMeshWorkloadFactory::override_runtime_arguments` (`ring_joint_sdpa_program_factory.cpp:2279`) |
| *Ops readiness* — Sync-free CBs (address-only) | **None** found — every CB read by raw pointer is also FIFO-synchronized (`wait_front`/`pop_front` or `reserve_back`/`push_back`) |

**Sync-free CBs** = CBs used purely as an address source (kernel grabs base pointer, no FIFO ops). None present in this op — all raw-pointer CB reads/writes pair with FIFO ops.

## Result

**RED at op level**, on two distinct blockers, each config-/factory-scoped — so a substantial clean subset exists:

1. **GlobalSemaphore (UNSUPPORTED in Metal 2.0)** — `RingJointSDPAProgramFactory` and `ExpRingJointSDPAProgramFactory` each embed a `std::vector<GlobalSemaphore>` *into their own program descriptor* (the fused ring all-gather). Routed to the **wait-for-feature** team; unblocks when GlobalSemaphore bindings land on `KernelSpec` (tracked by the `TODO -- GlobalSemaphore bindings` in `kernel_spec.hpp`).
2. **SPSC violation — hidden second writer** in `SDPAProgramFactory`, confined to the **non-causal KV-chain multicast-forwarding** path (`cb_k_in` / `cb_v_in` co-filled by a remote core's `noc.async_write_multicast`). Op-owner pre-port functional fix; route to the **SDPA op owner**. The **causal path and the non-causal-no-forwarding path are clean**.

Additionally, `ExpRingJointSDPAProgramFactory` raises an **open question** (Device 2.0): its fabric-mux / cross-device sync uses raw-L1 semaphore addresses (`get_semaphore(...)` + raw `noc_semaphore_wait/inc/set`) for which no Metal-2.0 `Semaphore<>` bridge exists today — see Questions for the user. (Moot in practice for the port decision: ExpRingJoint is already RED on GlobalSemaphore.)

**Clean / portable subset (`RED at op level; subset clear`):**
- **`JointSDPAProgramFactory`** — fully GREEN on every gate.
- **`RingDistributedSdpaProgramFactory`** — fully GREEN on every gate.
- **`SDPAProgramFactory`** — GREEN on the **causal + no-KV-forwarding** code paths (the non-causal mcast-forwarding path is the only SPSC violation).

These three (Joint, RingDistributed, and SDPA-base on its clean subset) can port once the audit clears with user go-ahead; RingJoint and ExpRingJoint wait for GlobalSemaphore support.

> No `METAL2_PORT_BRIEF.md` is emitted — the op is RED. (Were the GlobalSemaphore-free subset carved into its own port, the clean factories would qualify for a brief; that scoping is a user decision.)

## Gate detail

### ProgramDescriptor — GREEN (all five factories)

Every factory populates a `ProgramDescriptor` with `KernelDescriptor` / `CBDescriptor` / `SemaphoreDescriptor`; none uses the imperative `host_api.hpp` builder for the program build. `sdpa_program_factory.cpp:154`, `joint_sdpa_program_factory.cpp:28`, `ring_distributed_sdpa_program_factory.cpp:74/566`, `ring_joint_sdpa_program_factory.cpp:635/2256`, `exp_ring_joint_sdpa_program_factory.cpp:108/1739`.

One incidental imperative-API leak exists but is **dead code** (see Misc anomalies): `ring_fusion.cpp:20-60` `RingSDPAFusedOpSignaler::init_fused_op(Program&, …)` calls `CreateSemaphore(program,…)` but has no caller — the live fused-op semaphores are created via `SemaphoreDescriptor` (`ring_joint_sdpa_program_factory.cpp:908-924`).

### Device 2.0 (every kernel used)

| Factory | Verdict | Notes |
|---|---|---|
| `SDPAProgramFactory` | **GREEN** | `reader_interleaved.cpp`, `writer_interleaved.cpp`, `compute/sdpa.cpp` + headers — all `noc.async_*`, `CircularBuffer`, `Semaphore<>`, `TensorAccessor` |
| `JointSDPAProgramFactory` | **GREEN** | `joint_reader.cpp`, `joint_writer.cpp`, `compute/joint_sdpa.cpp` |
| `RingDistributedSdpaProgramFactory` | **GREEN** | `reader_interleaved.cpp`, `writer_interleaved.cpp`, `compute/sdpa.cpp` (mcast-forward path disabled, CTA = 0 at `ring_distributed_sdpa_program_factory.cpp:301`) |
| `RingJointSDPAProgramFactory` | **GREEN** | `ring_joint_reader.cpp`, `ring_joint_writer.cpp`, `compute/ring_joint_sdpa.cpp` + `chain_link.hpp`; CCL all-gather donor (`ring_attention_all_gather_reader.cpp` / `_writer.cpp`) is Device-2.0 clean |
| `ExpRingJointSDPAProgramFactory` | **OPEN QUESTION** (YELLOW→RED candidate) | structurally Device 2.0 *except* the fabric-mux / out-ready cross-device semaphores — raw-L1-address idiom (table below) |

No legacy addr-gen (`InterleavedAddrGen` / `ShardedAddrGen` / `*Fast` / `Pow2`), no free-function `noc_async_read/write`, anywhere in the op. The only CB-index free functions in use are `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` — both **sanctioned** by Device 2.0, **not** holdovers. **No isolated YELLOW holdovers** of the member-form-exists kind were found.

**ExpRingJoint fabric-mux / out-ready raw-L1 semaphore idiom** (the open question):

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `device/kernels/dataflow/exp_ring_joint_reader.cpp` | 80-81 | `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(argidx++))` (raw out-ready sem ptr; explicit comment :76 "passed as L1 addresses via RT args") | No `Semaphore<>` overload for fabric out-ready sem |
| `device/kernels/dataflow/exp_ring_joint_reader.cpp` | 252, 465 | `noc_semaphore_wait_min(per_link_sem_ptrs[lnk], …)` / `noc_semaphore_set(per_link_sem_ptrs[lnk], 0)` | — |
| `device/kernels/dataflow/exp_ring_joint_writer.cpp` | 145-149 | `get_semaphore(get_arg_val<uint32_t>())` ×5 (fabric-mux connection state) | — |
| `device/kernels/dataflow/exp_ring_joint_writer.cpp` | 487-493 | raw `noc_semaphore_wait(termination_sync_ptr, …)` / `noc_semaphore_inc(dest_addr, 1)` (mux termination handshake) | — |

These are the canonical `tt::tt_fabric` mux-client idiom (`build_connection_to_fabric_endpoint`, `fabric_endpoint_terminate`), matching shipping CCL ops. The fabric API mandates raw L1 addresses; per the recipe's donor-shape row, `uint32_t sem_addr` (L1) is "✗ not OK — no clean Metal-2.0 → donor bridge today." Non-fabric semaphores in the same kernels correctly use `Semaphore<>` (e.g. `exp_ring_joint_reader.cpp:96,265-287`). See Questions for the user. (Note: the RingJoint CCL all-gather donor uses the *same* sanctioned raw-L1 pattern for its GlobalSemaphore out-ready sem at `ring_attention_all_gather_reader.cpp:213,284` — but that is documented as an intentionally-retained primitive within the donor's own completed Device 2.0 migration, so it is not a donor gate.)

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `remote_index`, `.global_circular_buffer` field anywhere |
| Dynamic CircularBuffer (borrowed memory) | N/A | no `CBDescriptor::buffer` set; no `set_globally_allocated_address` |
| CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` used |
| Aliased Circular Buffers | **GREEN** | ExpRingJoint declares aliased pairs `c_1`+`c_14` (K) and `c_2`+`c_15` (V) — one `CBDescriptor`, two `CBFormatDescriptor`s over one allocation (`exp_ring_joint_sdpa_program_factory.cpp:643-672`). Port maps to `DataflowBufferSpec::advanced_options.alias_with`. Other four factories: single-element `format_descriptors` (no aliasing) |
| GlobalSemaphore | **RED** | RingJoint + ExpRingJoint (detail below) |
| Non-zero semaphore initial value | **GREEN** (heads-up) | `.initial_value = VALID` (=1) at `ring_joint_sdpa_program_factory.cpp:1206` and `exp_ring_joint_sdpa_program_factory.cpp:514`. `[[deprecated]]`, Gen2-unsupported. (The `INVALID`=0 inits are zero — no flag.) |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | no `ArgConfig::Runtime` token in op host code |
| `UpdateCircularBuffer*` | N/A | none used |
| Variable-count compile-time arguments (CTA varargs) | N/A | no kernel loops `get_compile_time_arg_val(i)` with runtime `i`; no `tensor_args_t` variable-count container. (Variable input *counts* — optional joint Q/K/V — are handled by presence flags producing distinct cache keys / fixed-shape branches, not CTA varargs.) |

#### RED: GlobalSemaphore in use (RingJoint + ExpRingJoint)

**Signal:** `std::vector<GlobalSemaphore>` carried in the device-op attributes and consumed by `.address()` into kernel RTAs *within the factory's own program descriptor*.

- **RingJointSDPAProgramFactory** — `exp_ring_joint`-distinct; here the all-gather is fused by appending CCL kernels into the SDPA `desc`: `ring_attention_all_gather_async_multi_core_with_workers_helper(desc, …, args.all_gather_operation_attributes.semaphore, …)` at `ring_joint_sdpa_program_factory.cpp:2222,2234`. The helper consumes `semaphore.at(0/1).address()` into kernel RTAs (`ring_attention_all_gather_async_multi_core_with_workers_program_factory.cpp:500,520,548,584`). Device-op attribute: `ring_joint_sdpa_device_operation.cpp:690,742`. Type/include: `ring_joint_sdpa_device_operation.hpp:48`.
- **ExpRingJointSDPAProgramFactory** — the exp factory embeds the GlobalSemaphore directly (fabric path): `args.semaphore[lnk].address()` → reader RTA at `exp_ring_joint_sdpa_program_factory.cpp:1549`; `args.semaphore[link].address()` → `out_ready_sem_addr` at `exp_ring_joint_sdpa_program_factory.cpp:1626`. Type/include: `exp_ring_joint_sdpa_device_operation_types.hpp:15,34`.

**Expected resolution:** Not yet supported in Metal 2.0 — there is no `GlobalSemaphore` binding on `KernelSpec` (`tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp`, `TODO -- GlobalSemaphore bindings`). The port of these two factories becomes possible once that binding lands. **Code-path scope:** confined to these two factories; SDPA / Joint / RingDistributed use no GlobalSemaphore and are unaffected.

### DFB endpoint legality (SPSC)

| Factory | Verdict |
|---|---|
| `SDPAProgramFactory` | **⛔ RED (config-scoped)** — hidden second writer on `cb_k_in` / `cb_v_in` |
| `JointSDPAProgramFactory` | **✓ legal** — every CB (1 producer, 1 consumer); single reader/compute/writer, no split reader |
| `RingDistributedSdpaProgramFactory` | **✓ legal** — uniform interleaved CBs; mcast-forward path disabled (CTA = 0) |
| `RingJointSDPAProgramFactory` | **✓ legal** — reader/compute/writer (1,1) FIFO pairs; chain forwarding writes to *remote* cores' CBs (no local multi-endpoint) |
| `ExpRingJointSDPAProgramFactory` | **✓ legal** (best-effort; formally deferred pending the Device 2.0 fabric-sem ruling) — aliased `c_1`/`c_14`, `c_2`/`c_15` are two distinct DFBs over one allocation, each (1,1); the `ASSERT(cb_k.get_write_ptr()==cb_k_writer.get_write_ptr())` is the aliasing invariant, **not** a hidden second writer |

#### ⛔ SPSC violation — SDPAProgramFactory, non-causal KV-chain mcast-forwarding (config-scoped GATE → op owner)

On a **receiving** core in a K/V-forwarding chain, `cb_k_in` (and `cb_v_in`) is filled by a **remote core's `noc.async_write_multicast`** targeting this node's `cb_k.get_write_ptr()` slot, coordinated by dedicated `sender`/`receiver`/`valid` semaphores rather than CB FIFO sync — while the local receiver does `reserve_back`/`get_write_ptr`/`push_back` without writing the data itself. That is two producers on one CB instance: the **hidden-second-writer** shape, invisible to a FIFO-sync trace.

- Sender (remote write): `device/kernels/dataflow/reader_interleaved.cpp:467-499`
- Receiver: `device/kernels/dataflow/reader_interleaved.cpp:409-416`
- Host gate: `device/sdpa_program_factory.cpp:719-760` (semaphores under `if (!is_causal)`); `mcast_enabled` CTA index 32.

Binding the remote multicast write through a `dfb::` handle makes `cb_k_in` a two-producer node → the Metal 2.0 spec validator rejects it at port time (cryptic, late). **No port-time workaround** (self-loop cannot absorb a second cross-core writer).

- **Op-owner pre-port fix:** forward into a *dedicated* receive CB rather than co-filling `cb_k_in`/`cb_v_in` (or a Gen2-conditional single DM fill) — a functional change, out of port scope.
- **Config scope:** the violation engages only when `!is_causal && is_chain_participant && mcast`. The **causal** path and the **non-causal-but-non-forwarding** path keep `cb_k_in`/`cb_v_in` at (1,1) → legal. ⇒ `RED at op level (SDPAProgramFactory); subset = causal + no-KV-forwarding is clear`.

No dead CBs found in any factory.

## Port-work summary  *(would mirror the brief, were one issued)*

Applies to the clean subset (Joint, RingDistributed, SDPA-base on its clean paths). RingJoint / ExpRingJoint port work is moot until GlobalSemaphore lands.

- **Tensor bindings** (per binding, all factories): **all Case 1** (`TensorAccessor`). Every kernel feeds an RTA-sourced base address into a `TensorAccessor(args, addr)` constructor and accesses memory through the accessor. Host side uses the framework's interim **`Buffer*`-binding form** (`emplace_runtime_args` with `Buffer*` objects — auto-registered `BufferBinding`, patched on cache hits; *not* the silent-stale hazard). Port: express each as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::name)`; the address RTA + `TensorAccessorArgs` plumbing disappear. Mechanical, low-risk. **No Case 2 (raw pointer) bindings; no compute-kernel tensor access** (compute kernels are CB-only — out of scope). Representative sites:
  - SDPA: host `sdpa_program_factory.cpp:302-316,739,1310-1348`; kernel `reader_interleaved.cpp:108-216`, `writer_interleaved.cpp:43-72`. Bindings: `q_in,k_in,v_in,mask_in,page_table,attention_sink,chunk_start_idx,out`.
  - Joint: host `joint_sdpa_program_factory.cpp:630-659`; kernel `joint_reader.cpp:35-61`, `joint_writer.cpp:39-53`. Bindings: `q,k,v,joint_q,joint_k,joint_v,out,joint_out`.
  - RingDistributed: host `ring_distributed_sdpa_program_factory.cpp:513-542`; kernel `reader_interleaved.cpp`, `writer_interleaved.cpp`. Bindings: `q,k,v,page_table,out` (+ nullptr placeholders for absent mask/attention_sink/chunk_start_idx).
  - RingJoint / ExpRingJoint: host `ring_joint_sdpa_program_factory.cpp:2089-2097` / `exp_ring_joint_sdpa_program_factory.cpp:1483-1490`; bindings `q,k,v,gathered_k,gathered_v,joint_q,joint_k,joint_v,out,joint_out,stats`.
- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception) for `SDPAOperation` (`sdpa_device_operation.cpp:391`), `RingJointSDPADeviceOperation` (`ring_joint_sdpa_device_operation.cpp:581`), `ExpRingJointSDPADeviceOperation` (`exp_ring_joint_sdpa_device_operation.cpp:326`). `JointSDPADeviceOperation` and `RingDistributedSdpaDeviceOperation` have **no** custom hash (nothing to delete). See Custom program hash for relaxation candidates mined before deletion.

## Heads-ups  *(would mirror the brief)*

- **Aliased CBs (LANDED, FYI-P):** ExpRingJoint `c_1`+`c_14` and `c_2`+`c_15` at `exp_ring_joint_sdpa_program_factory.cpp:643-672` → port to `DataflowBufferSpec::advanced_options.alias_with` (two DFBs per allocation, mutually `alias_with`; verify matching size/format — they match). Do not split into independent DFBs.
- **Non-zero semaphore initial value (LANDED-deprecated, FYI-P):** `.initial_value = VALID` (=1) at `ring_joint_sdpa_program_factory.cpp:1206` and `exp_ring_joint_sdpa_program_factory.cpp:514` → `SemaphoreSpec::advanced_options.initial_value` (`[[deprecated]]`, Gen2-unsupported). Expected, not a blocker. (Both factories are RED on GlobalSemaphore anyway.)
- **Sync-free CBs:** none.
- **Dead CBs (zero endpoints):** none.
- **Cross-op / shared kernels:** see Team-only — the SDPA-family kernels are shared across factories (port-the-family-together), and RingJoint file-path-instantiates the CCL all-gather donor kernels; ExpRingJoint file-path-instantiates the fabric-mux LLK kernel.
- **RTA varargs:** none — every kernel pulls RTAs positionally via an `argidx++` cursor (statically-bounded sequence with compile-time-gated sections), not a runtime-varying-count loop.
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other risky pybind. Custom `override_runtime_arguments`: `RingJointSDPAMeshWorkloadFactory::override_runtime_arguments` (`ring_joint_sdpa_program_factory.cpp:2279`) — re-patches indexed-kv-cache / kv-pad-rotation scalar RTAs per dispatch (RingJoint is RED on GlobalSemaphore, so moot until that clears).

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean for SDPA/Joint/RingDistributed (only `tt_metal/*` LLK + `ttnn/cpp/ttnn/kernel_lib/` shared-lib includes). ⚠ RingJoint pulls in a cross-family CCL donor (Device-2.0 clean; coupled to the GlobalSemaphore gate). ⚠ ExpRingJoint file-path-instantiates a fabric-mux LLK kernel and `#include`s `tt::tt_fabric` + CCL headers.

**Borrowed / file-path kernel instantiation:**

| Factory | Kernel file instantiated | Owning family / pool | Shared? |
|---|---|---|---|
| RingJoint | `ring_attention_all_gather_reader.cpp`, `ring_attention_all_gather_writer.cpp` | `experimental/ccl/ring_attention_all_gather_async/` (cross-family donor) | shared with the standalone ring-attention all-gather op |
| ExpRingJoint | `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp` | `tt_metal/*` LLK/firmware (category 1) | broadly shared (fabric infra) |

**Per-call donor shape (CCL all-gather, RingJoint):** the donor kernels take `TensorAccessor`-based addrgens (Shape 1 ✓), `Semaphore<>` for local sync (✓), and one raw-L1 GlobalSemaphore out-ready address (`uint32_t sem_addr`, the ✗ "not OK today" shape — but this is the GlobalSemaphore gate itself, not a separable donor concern). Helper appends kernels via `KernelDescriptor` (ProgramDescriptor API), not imperative `CreateKernel`. **No donor Device-2.0 gate.**

**SDPA-family shared-kernel set (port-together):** the five factories share `device/kernels/dataflow/dataflow_common.hpp`, `compute/compute_common.hpp`, `compute/compute_streaming.hpp`, `reader_interleaved.cpp`/`writer_interleaved.cpp` (SDPA + RingDistributed), and the ring/joint variants. A Metal 2.0 CB→DFB / named-token rewrite of any shared header must land for all co-using factories in the same change.

### Relaxation candidates (mined from custom hashes — FALLIBLE, candidates to verify; default strict)

- **`SDPAOperation`** (`sdpa_device_operation.cpp:391-419`): when `flexible_chunked` (chunk-start-idx supplied as a *tensor*), the hash deliberately drops `page_table` and `chunk_start_idx` from the key (`page_table_for_hash = std::nullopt`, `chunk_start_idx_for_hash = std::nullopt`) so one program serves varying chunk positions, re-patched at runtime. ⇒ candidate: `page_table` / `chunk_start_idx_tensor` bindings may tolerate a `dynamic_tensor_shape` (or `match_padded_shape_only`) relaxation. Verify — the default strict hash would force a re-hash per chunk position; the relaxation is the mechanism that preserves today's single-program behavior.
- **`RingJointSDPADeviceOperation`** (`ring_joint_sdpa_device_operation.cpp:581-621`): uses `cache_key_logical_n = 0` when `kv_pad_rotation_enabled`, reusing one program across varying `logical_n` (re-patched per dispatch by `override_runtime_arguments`). ⇒ candidate: the indexed-kv-cache / kv-pad-rotation scalars are runtime-dynamic by design; a relaxation on the gathered-K/V bindings may be needed to keep single-program caching post-default-hash. Verify.
- **`ExpRingJointSDPADeviceOperation`** (`exp_ring_joint_sdpa_device_operation.cpp:326-352`): straightforward tensor+attribute hash; no obvious relaxation clue. Note it does **not** hash `multi_device_global_semaphore` — fine (semaphores aren't tensor args; the default hash would also exclude it).

### TTNN factory analysis — six-question answers

1. **Op-owned tensors? — No (all five).** Every `create_device_tensor` is inside `create_output_tensors` and produces a *declared output* (`sdpa_device_operation.cpp:386-388`; `joint_sdpa_device_operation.cpp:169-174`; `ring_distributed_sdpa_device_operation.cpp:279-281`; `ring_joint_sdpa_device_operation.cpp:571-577`; `exp_ring_joint_sdpa_device_operation.cpp:316-322`). No scratch/intermediate device tensor is constructed and threaded into a program. (RingJoint/ExpRingJoint consume *passed-in* persistent gather buffers — caller-owned, not op-owned.)
2. **MeshWorkload needed?** — **No** for `SDPAProgramFactory` and `JointSDPAProgramFactory` (single `ProgramDescriptor` via `create_descriptor`). **Yes (genuine)** for `RingDistributedSdpaProgramFactory` (`create_workload_descriptor`, `ring_distributed_sdpa_program_factory.cpp:566`), `RingJointSDPAProgramFactory` (`create_workload_descriptor`/`create_mesh_workload`, `ring_joint_sdpa_program_factory.cpp:2256,2271`), and `ExpRingJointSDPAProgramFactory` (`create_workload_descriptor`, `exp_ring_joint_sdpa_program_factory.cpp:1739`). Per their own comments, the per-coord descriptors differ because forward/backward neighbor coords and device-index depend on the mesh coordinate — genuine cross-device ring coordination, **not** an op-owned-tensor plumbing artifact (Q1 = No rules that out).
3. **Pybind `create_descriptor`? — No.** `sdpa_nanobind.cpp` binds only the nine user-facing op functions via `ttnn::bind_function<…>` (`bind_sdpa`, line 277+); no `nb::class_<…ProgramFactory>` / `def_static("create_descriptor", …)`.
4. **Other migration-risky pybind? — None.** No `DeviceOperation`/factory/param class binding; only the normal op-function surface.
5. **Custom hash? — Yes (3 of 5):** `SDPAOperation` (`sdpa_device_operation.cpp:391`), `RingJointSDPADeviceOperation` (`ring_joint_sdpa_device_operation.cpp:581`), `ExpRingJointSDPADeviceOperation` (`exp_ring_joint_sdpa_device_operation.cpp:326`). Treatment = delete → default (see Custom program hash / Port-work summary). `JointSDPADeviceOperation` and `RingDistributedSdpaDeviceOperation`: No.
6. **Custom override-runtime-args? — Yes (1 of 5):** `RingJointSDPAMeshWorkloadFactory::override_runtime_arguments` (`ring_joint_sdpa_program_factory.hpp:55`, def `ring_joint_sdpa_program_factory.cpp:2279`). The other four factories: No.

## Misc anomalies  *(team-only, non-gating)*

- **Dead imperative helper:** `device/ring_fusion.cpp:20-60` `RingSDPAFusedOpSignaler::init_fused_op(Program&, …)` (imperative `CreateSemaphore`) has **no caller** in the ring factory — the factory uses `init_all_gather` + `SemaphoreDescriptor` and a *different* class (`AllGatherFusedOpSignaler::init_fused_op`) at `ring_joint_sdpa_program_factory.cpp:2207`. Op-owner cleanup; not port work. (If left, it does not affect the port — it is never on the program-build path.)

## Per-DeviceOperation attribution

| DeviceOperation / Factory | ProgramDescriptor | Device 2.0 | Features | SPSC | Verdict |
|---|---|---|---|---|---|
| `SDPAOperation` / `SDPAProgramFactory` | Yes | GREEN | GREEN (VALID heads-up n/a) | ⛔ config-scoped (non-causal mcast-forward) | **RED (subset clear: causal + no-forward)** |
| `JointSDPADeviceOperation` / `JointSDPAProgramFactory` | Yes | GREEN | GREEN | ✓ legal | **GREEN** |
| `RingDistributedSdpaDeviceOperation` / `RingDistributedSdpaProgramFactory` | Yes | GREEN | GREEN | ✓ legal | **GREEN** (genuine MeshWorkload — carried natively) |
| `RingJointSDPADeviceOperation` / `RingJointSDPAProgramFactory` | Yes | GREEN | **RED (GlobalSemaphore)** + VALID heads-up | ✓ legal | **RED (wait-for-feature: GlobalSemaphore)** |
| `ExpRingJointSDPADeviceOperation` / `ExpRingJointSDPAProgramFactory` | Yes | **OPEN (fabric-mux raw-L1 sem)** | **RED (GlobalSemaphore)** + aliased-CB + VALID heads-up | ✓ legal (best-effort) | **RED (wait-for-feature: GlobalSemaphore; + Device-2.0 fabric question)** |

## Questions for the user

1. **ExpRingJoint fabric-mux / out-ready raw-L1 semaphore idiom — Device 2.0 sanctioned or a holdover?** `exp_ring_joint_reader.cpp:80-81,252,465` and `exp_ring_joint_writer.cpp:145-149,487-493` use raw L1 semaphore addresses (`get_semaphore(get_arg_val<…>())`, `noc_semaphore_wait/inc/set`, `reinterpret_cast<volatile tt_l1_ptr uint32_t*>`) for fabric-mux connection state and cross-device out-ready/termination sync. This is the canonical `tt::tt_fabric` mux-client interface (`build_connection_to_fabric_endpoint`, `fabric_endpoint_terminate`) — it takes raw L1 addresses and has no Metal-2.0 `Semaphore<>` overload today. Is this a sanctioned fabric idiom (→ Device 2.0 GREEN for ExpRingJoint), or a holdover with no Metal-2.0 bridge (→ Device 2.0 RED / route to wait-for-feature)? *(In practice moot for the immediate port decision: ExpRingJoint is independently RED on GlobalSemaphore.)*
2. **Scoped-subset port?** Do you want a separate brief carved for the clean subset — `JointSDPAProgramFactory` + `RingDistributedSdpaProgramFactory` (fully GREEN), and optionally `SDPAProgramFactory` on its causal + no-KV-forwarding paths? If so, the SDPA-base subset port must scope out the non-causal mcast-forwarding path (the SPSC violation) and the SDPA op owner should be asked to land the dedicated-receive-CB fix before that path can be ported.

## Recipe notes

- **Fused-CCL-into-own-descriptor is a GlobalSemaphore-gate shape the recipe's Appendix-A GlobalSemaphore entry doesn't explicitly call out.** RingJoint embeds an entire all-gather (and its GlobalSemaphore) by calling a cross-family CCL *helper that appends kernels into the SDPA `desc`*, rather than by holding a `GlobalSemaphore` parameter on its own factory signature. The recognition signal (`std::vector<GlobalSemaphore>` device-op attribute → `.address()` into RTAs) still fires, but the trace runs through a donor helper one level removed from the factory. Worth a one-line note in the GlobalSemaphore entry that "construction-by-consumption" can occur via a fused-op helper, parallel to the existing GlobalCircularBuffer "construction-by-consumption" bullet.
- **The fabric-mux raw-L1 semaphore case sits awkwardly between the Device 2.0 gate and the donor-shape table.** The recipe's donor-shape row "`uint32_t sem_addr` (L1) — ✗ not OK" describes it, but the idiom appears in the op's *own* kernels (not only donor includes) because the fabric API surface forces it. It's neither a clean YELLOW holdover (no member-form replacement exists) nor an obvious Device-1.0 RED (it's the current sanctioned fabric interface). The recipe could use an explicit note on how to tier a raw-L1-sem idiom that is structurally forced by the `tt::tt_fabric` API rather than being a legacy holdover.

---

## ✅/⚠️ Post-port update (2026-06-25)

**Ported successfully (PR #48175):** `JointSDPAProgramFactory` — 160 passed / 32 skipped / 0 failed + program-cache 8/0. Fully GREEN grade held; joint-only kernels, no custom hash, mixed-variant `std::visit` dispatch leaves the other four factories legacy.

**`RingDistributedSdpaProgramFactory` — framework-blocked (audit miss):** graded GREEN-portable above with "MeshWorkload needed: Yes (genuine)", but the port gated. RingDistributed needs **one program per mesh coord** (`create_mesh_workload`). In Metal 2.0, `MetalV2FactoryConcept` (the `create_program_artifacts`/ProgramSpec/`dfb::` codegen path) and `MeshWorkloadFactoryConcept` are **mutually exclusive** (`operation_concepts.hpp:91`), and **every** existing `create_mesh_workload` factory (incl. the only "metal2" matmul one) builds each per-coord program via **legacy imperative `tt_metal::Program{}`**, never ProgramSpec. So per-coord-mesh dispatch + `dfb::` kernels cannot coexist. Wait-for-feature: a ProgramSpec-backed mesh-workload path. See [[metal2-mesh-workload-vs-programspec-block]]. **Audit rule:** treat "MeshWorkload needed: Yes (genuine, per-coord)" as RED framework-blocked for a `dfb::` port, not GREEN.

`RingJointSDPAProgramFactory` / `ExpRingJointSDPAProgramFactory` correctly RED (GlobalSemaphore). `SDPAProgramFactory` correctly needs the op-owner non-causal-mcast-forwarding SPSC fix before its (otherwise clean) port.
