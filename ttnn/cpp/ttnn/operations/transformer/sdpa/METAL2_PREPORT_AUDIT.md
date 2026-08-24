# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/transformer/sdpa`

This directory bundles **seven** independent `DeviceOperation`s that share the kernel utility layer
(`device/kernels/dataflow/dataflow_common.hpp`, `device/kernels/compute/compute_common.hpp`,
`device/kernels/compute/compute_streaming.hpp`). They are audited together and reported here with
per-DeviceOperation attribution.

- **`SDPADeviceOperation`** (`sdpa_device_operation.*`)
  - `SDPAProgramFactory` (`sdpa_program_factory.cpp`) — kernels `reader_interleaved.cpp`, `writer_interleaved.cpp`, `compute/sdpa.cpp`
- **`JointSDPADeviceOperation`** (`joint_sdpa_device_operation.*`)
  - `JointSDPAProgramFactory` (`joint_sdpa_program_factory.cpp`) — kernels `joint_reader.cpp`, `joint_writer.cpp`, `compute/joint_sdpa.cpp`
- **`SparseSDPADeviceOperation`** (`sparse_sdpa_device_operation.*`)
  - `SparseSDPAProgramFactory` (`sparse_sdpa_program_factory.cpp`) — kernels `sparse_sdpa_reader.cpp`, `sparse_sdpa_writer.cpp`, `compute/sparse_sdpa_compute.cpp`
- **`SparseSDPAMSADeviceOperation`** (`sparse_sdpa_msa_device_operation.*`)
  - `SparseSDPAMsaProgramFactory` (`sparse_sdpa_msa_program_factory.cpp`) — kernels `sparse_sdpa_msa_reader.cpp`, `sparse_sdpa_msa_writer.cpp`, `compute/sparse_sdpa_msa_compute.cpp`
- **`RingDistributedSDPADeviceOperation`** (`ring_distributed_sdpa_device_operation.*`)
  - `RingDistributedSdpaProgramFactory` (`ring_distributed_sdpa_program_factory.cpp`)
- **`RingJointSDPADeviceOperation`** (`ring_joint_sdpa_device_operation.*`)
  - `RingJointSDPAMeshWorkloadFactory` (`ring_joint_sdpa_program_factory.cpp`)
- **`ExpRingJointSDPADeviceOperation`** (`exp_ring_joint_sdpa_device_operation.*`)
  - `ExpRingJointSDPAProgramFactory` (`exp_ring_joint_sdpa_program_factory.cpp`)

Host-side helpers (`sdpa_perf_model.*`, `ring_fusion.*`, `sliding_halo_layout.*`, `block_cyclic_layout.hpp`) and
numerous kernel `*.hpp` helpers are all referenced through the factories/kernels above; no unreferenced (dead) kernel
files were found in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `d6087d9353f 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/transformer/sdpa` |
| **Overall** | **RED** (whole op) — **clean subset survives: `SDPADeviceOperation` + `JointSDPADeviceOperation`** |
| **DOps / Factories** | 7 DeviceOperations, one factory each (see identifying section) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** for the 5 non-`exp_ring` DOps' kernels; the `exp_ring_joint` kernels carry raw-semaphore Device-1.0 holdovers (part of the GlobalSemaphore machinery). The op's kernels use `Noc`/`CircularBuffer`/`DataflowBuffer` wrappers + `TensorAccessor` throughout; no legacy addr-gen. |
| *Prereqs* — Cross-op escapes | **Ok / ⚠ workable** — two shared-pool donor headers (`reduce_helpers_dataflow.hpp` ✓ Device 2.0-native; `generate_bcast_scalar.hpp` legacy `CircularBuffer` sig, but a `_metal2` fork already exists beside it). No file-path kernel borrows. |
| *Feature Support* — overall | **RED** |
| *Feature Support* — GlobalSemaphore | **RED** — `RingJointSDPADeviceOperation`, `ExpRingJointSDPADeviceOperation` |
| *Feature Support* — GlobalCircularBuffer | N/A (absent) |
| *Feature Support* — `address_offset` (non-zero) | N/A (absent) |
| *TTNN Readiness* — `Is able to port?` (the gate, per factory) | SDPA **yes** · JointSDPA **yes** · SparseSDPA **no** · SparseSDPAMSA **no** · RingDistributed **no** · RingJoint **no** · ExpRingJoint **no** |
| *TTNN Readiness* — Concept (current) | `descriptor` (SDPA, JointSDPA, SparseSDPA, SparseSDPAMSA) · `WorkloadDescriptor` (RingDistributed, RingJoint, ExpRingJoint) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | RingDistributed **yes** ("Per-coord RTAs") · RingJoint **no** (genuine multi-program) · ExpRingJoint **no** (genuine multi-program) |
| *TTNN Readiness* — Custom hash | SDPA/JointSDPA **no** · SparseSDPA/SparseSDPAMSA **yes** · RingJoint **yes** · ExpRingJoint **no** · RingDistributed **no** (port leaves any custom hash intact — not a gate) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** (all factories) |
| *TTNN Readiness* — `override_runtime_arguments` | SparseSDPA/SparseSDPAMSA **yes** (→ CustomProgramSpecFactoryConcept) · RingJoint **yes** · others **no** |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** (`sdpa_nanobind.cpp` binds no `create_descriptor`) |
| *TTNN Readiness* — Op-owned tensors | **No** (all) |
| *TTNN Readiness* — Target concept (cleared DOps) | SDPA → `ProgramSpecFactoryConcept` · JointSDPA → `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none** — no address RTA folds a host-side offset into its base (whole op) |
| *Port work* — Tensor bindings (subset, per binding) | all **Case 1** (via `TensorAccessor`; `Buffer*`-binding delivery) |
| *TTNN Readiness* — TensorParameter relaxation | SDPA/JointSDPA/RingDistributed `none` · **SparseSDPA `dynamic` → GATE** · **SparseSDPAMSA `dynamic` → GATE** · RingJoint/ExpRingJoint `(pending analysis)` |
| *Port work* — TensorAccessor 3rd arg | one site (SDPA page-table), **Class 2 → drop** (non-gating); JointSDPA none |
| *Port work* — CB endpoints (subset) | all **legal / self-loop / 1P+1C** — no dead CB, no multi-binding |

---

## Result

**RED at op level** — two independent blockers land on the three ring/distributed DeviceOperations, and a
relaxation blocker lands on the two sparse DeviceOperations:

- **GlobalSemaphore (Appendix A UNSUPPORTED)** blocks `RingJointSDPADeviceOperation` and `ExpRingJointSDPADeviceOperation`.
- **Genuine multi-program `WorkloadDescriptor`** (not secretly-SPMD) blocks `RingJointSDPADeviceOperation` and
  `ExpRingJointSDPADeviceOperation`; both are also flagged **"Broken Op"** on the readiness sheet (smuggled RTA pointer).
- **`RingDistributedSDPADeviceOperation`** is secretly-SPMD but blocked by the single-program adapter's inability to
  express **per-coord runtime args** (`Is able to port? = no`, target "TBD (SPMD + coord args issue)") → framework work.
- **`SparseSDPADeviceOperation`** and **`SparseSDPAMSADeviceOperation`** are blocked by **`TensorParameter relaxation = dynamic`** → ops team.

**A clean subset is clear and portable now: `SDPADeviceOperation` + `JointSDPADeviceOperation`.** Both are
`descriptor`-concept, `Is able to port? = yes`, and pass every code-based gate (Device 2.0 ✓, no Appendix-A features,
no offset base pointers, TensorAccessor 3rd arg Class 2). A porter brief is issued for this subset
(`METAL2_PORT_BRIEF.md`).

`RED at op level; subset {SDPADeviceOperation, JointSDPADeviceOperation} is clear.`

**Path forward for the blocked DeviceOperations** (none is a permanent block):
- Sparse / SparseMSA: the `dynamic` `TensorParameter relaxation` is under ops-team analysis; unblocks when that lands.
- RingDistributed: unblocks when the single-program (SPMD) adapter grows per-coord runtime-arg support (framework).
- RingJoint / ExpRingJoint: unblock when GlobalSemaphore support lands in Metal 2.0 **and** the "Broken Op" (smuggled
  pointer) is fixed by TTNN **and** genuine multi-program `WorkloadDescriptor` support arrives (framework) — a
  multi-team, multi-gate effort.

---

## Gate detail

### TTNN factory concept (`Is able to port?`) — readiness-sheet lookup + code cross-check

Sheet fetched fresh this session (Diego's "Operations analysis"). Per-factory verdict:

| DeviceOperation | Concept | `Is able to port?` | Attribution / route |
|---|---|---|---|
| `SDPADeviceOperation` | descriptor | **yes** | cleared |
| `JointSDPADeviceOperation` | descriptor | **yes** | cleared |
| `SparseSDPADeviceOperation` | descriptor | **no** | `TensorParameter relaxation = dynamic` → **ops team** |
| `SparseSDPAMSADeviceOperation` | descriptor | **no** | `TensorParameter relaxation = dynamic` → **ops team** |
| `RingDistributedSDPADeviceOperation` | WorkloadDescriptor (secretly-SPMD) | **no** | per-coord RTAs / "SPMD + coord args issue" → **framework** (single-program adapter gap) |
| `RingJointSDPADeviceOperation` | WorkloadDescriptor (genuine multi-program) | **no** | "Broken Op" (smuggled RTA pointer) → **TTNN**; also GlobalSemaphore + genuine multi-program |
| `ExpRingJointSDPADeviceOperation` | WorkloadDescriptor (genuine multi-program) | **no** | "Broken Op" (smuggled RTA pointer) → **TTNN**; also GlobalSemaphore + genuine multi-program |

**Lightweight cross-check vs. code — CLEAN.** Every cheaply-checkable primary column agrees with the code:
- `Concept`: confirmed from factory methods — `create_descriptor` returning `ProgramDescriptor` for the four
  `descriptor` ops (`sdpa_device_operation.hpp:25`, `joint_sdpa_device_operation.hpp:26`,
  `sparse_sdpa_device_operation.hpp:27`, `sparse_sdpa_msa_device_operation.hpp:30`); `create_workload_descriptor`
  returning `WorkloadDescriptor` for `RingDistributedSdpaProgramFactory` (`ring_distributed_sdpa_program_factory.cpp:590`)
  and the two mesh-workload factories.
- `Custom hash`: `compute_program_hash` present for `ring_joint` (`ring_joint_sdpa_device_operation.cpp:879`),
  `sparse_sdpa`, `sparse_sdpa_msa`; **absent** for `sdpa`/`joint` (grep clean). **`exp_ring_joint`: sheet says `no`,
  and the code agrees** — the grep hit at `exp_ring_joint_sdpa_device_operation.cpp:329` is a *comment*
  ("Order mirrors compute_program_hash…"), not an override. Not a conflict.
- `Override runtime args method?`: `override_runtime_arguments` present for SparseSDPA / SparseSDPAMSA
  (`sparse_sdpa_device_operation.hpp:32`, `sparse_sdpa_msa_device_operation.hpp:38`) and RingJoint — matches sheet.
- `Pybind descriptor`: `sdpa_nanobind.cpp` binds no `create_descriptor` — matches sheet `no`.
- `get_dynamic_runtime_args`: none in any device-op — matches sheet `no`.
- **Factory-set match:** 7 code factories ↔ 7 sheet rows, 1:1; no phantom or missing rows. (Sheet DeviceOperation
  names differ cosmetically from code — e.g. sheet `SDPAOperation` vs code `SDPADeviceOperation` — a naming-convention
  difference, not a phantom row.)
- **RingDistributed `Secretly SPMD` — recorded discrepancy, not a conflict.** The code builds **N distinct per-coord
  `ProgramDescriptor`s** in a loop (`ring_distributed_sdpa_program_factory.cpp:598-601`, "ring_id is inferred from the
  coord, so each coord builds a distinct descriptor"), which the recipe's *single-entry* heuristic would read as
  **not** secretly-SPMD. The sheet marks it secretly-SPMD = `yes` with reason "Per-coord RTAs" — a finer judgment
  (the per-coord descriptors differ *only by RTAs*, hence morally single-program). Deferred to the sheet's judgment;
  logged in **Recipe notes**. The gate verdict (`no`) is unaffected.

### Device 2.0 (every kernel used)

**GREEN for the portable subset and for SparseSDPA / SparseSDPAMSA / RingDistributed.** The op's kernels are
structurally Device 2.0: `Noc` object (`noc.async_read` / `async_write` / `async_read_barrier`), `CircularBuffer` /
`DataflowBuffer` wrappers, `TensorAccessor`. No legacy address generators (`InterleavedAddrGen` / `ShardedAddrGen` /
`InterleavedAddrGenFast` / pow2) anywhere; no CB-index free-function holdovers (`get_read_ptr(cb_id)` etc.); the
`get_tile_size(cb_id)` / `get_local_cb_interface(cb_id)` free functions in use are **sanctioned**. The raw
`noc_async_read/write` grep hits in `reader_interleaved.cpp:50,582` and `ring_joint_reader.cpp:1023` are all in
**comments**.

**`ExpRingJointSDPADeviceOperation` — raw-semaphore Device-1.0 holdovers (recorded; the op is RED on other gates).**

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/exp_ring_joint_reader.cpp` | 465 | `noc_semaphore_set(per_link_sem_ptrs[lnk], 0)` | raw L1 sem pointer (per-link GlobalSemaphore) |
  | `device/kernels/dataflow/exp_ring_joint_reader.cpp` | 252 | `noc_semaphore_wait_min(per_link_sem_ptrs[lnk], …)` | raw L1 sem pointer (per-link GlobalSemaphore) |

  These are intertwined with the GlobalSemaphore machinery this op is already RED on; the Device 2.0 completeness of
  the ring_joint / exp_ring_joint kernels should be re-sized by the Device 2.0 team alongside the GlobalSemaphore work.

### Feature compatibility (Appendix A)

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | absent — no `GlobalCircularBuffer` type, `remote_index`, or `.global_circular_buffer` field anywhere |
| CBDescriptor `address_offset` (non-zero) | N/A | absent — no `address_offset` / `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` |
| GlobalSemaphore | **RED** | in use by `RingJointSDPADeviceOperation` + `ExpRingJointSDPADeviceOperation` (detail below) |

#### GlobalSemaphore — UNSUPPORTED (RED)

- **Signal:** the type `tt::tt_metal::GlobalSemaphore` in device-op state and factory signatures.
- **Sites:**
  - `device/exp_ring_joint_sdpa_device_operation_types.hpp:15` (`#include <tt-metalium/global_semaphore.hpp>`), `:34` (`std::vector<GlobalSemaphore> semaphore;` op state), `:51` (ctor param)
  - `device/exp_ring_joint_sdpa_device_operation.hpp:45` and `device/exp_ring_joint_sdpa_device_operation.cpp:393` (`const std::vector<GlobalSemaphore>&` param)
  - `device/ring_joint_sdpa_device_operation.hpp:51`, `device/ring_joint_sdpa_device_operation.cpp:1035` (`const std::vector<GlobalSemaphore>&` param)
  - `device/exp_ring_joint_sdpa_program_factory.cpp:1646,1825,1840` (`args.semaphore[...].address()` written into RTAs)
- **Expected resolution:** not yet supported in Metal 2.0 (`GlobalSemaphore bindings` listed under "coming soon" in
  `kernel_spec.hpp`); the port becomes possible once GlobalSemaphore support lands on `KernelSpec`. Confined to the two
  ring-joint DeviceOperations; the other five DeviceOperations use only plain `Semaphore` or none.

### CB endpoints (GATE-free) — clean for the portable subset

Assessed per `(CB, node, config)` for the two cleared DeviceOperations. **No CB gates anything** (the subject is
GATE-free), and for this subset there is **no dead CB and no multi-binding** — every out-of-window CB resolves to a
self-loop or a plain 1P+1C. See the Port-work summary for the disposition list. (Not assessed in depth for the five
RED DeviceOperations — those are re-audited when their gates clear; the exp_ring_joint kernels are additionally
deferred per the Device-2.0-holdover note above.)

### Offset base pointers

**GREEN (whole op).** No address RTA folds a host-side offset into its base. `sdpa` is not listed in the offset-base
triage (`2026-07-19_offset_base_pointers.md`), and the scan confirms clean:
- SDPA / JointSDPA deliver tensor bases as **`Buffer*`-bindings** (e.g. `sdpa_program_factory.cpp:1407-1413`,
  `joint_sdpa_program_factory.cpp:592-599`) — clean bases, framework-patched, no arithmetic.
- SparseSDPA pushes bare `->address()` values (`sparse_sdpa_program_factory.cpp:361-362`) — clean bases, no offset.
- The ring/joint ops smuggle GlobalSemaphore addresses and *on-device metadata-tensor* addresses
  (`ring_joint_sdpa_program_factory.cpp:2565-2581`) — clean bases, not offset folds.

The `+offset` grep hits (`exp_ring_joint_sdpa_program_factory.cpp:1447`, `ring_joint_sdpa_program_factory.cpp:588,638`)
are loop/index variables, not address arithmetic.

### TensorAccessor 3rd argument

**GREEN (non-gating).** Exactly **one** 3-arg `TensorAccessor` site in the whole op:
`device/kernels/dataflow/dataflow_common.hpp:83`, inside `read_page_table_for_batch`
(`TensorAccessor(page_table_args, page_table_addr, page_table_stick_size)`).

- **Specialization:** interleaved (DRAM page table, strided by `.page_id = batch_idx`).
- **Magnitude:** `page_table_stick_size` resolves to `page_table_tensor.buffer()->aligned_page_size()`
  (`sdpa_program_factory.cpp:165`) → correct magnitude, and equal to the aligned page.
- **Class 2 — Redundant / inert** (interleaved + correct-magnitude ⇒ realigned; `sdpa` page-table matches the triage
  doc `2026-07-06_tensor_accessor_3rd_arg_triage.md`, and `relaxation = none` on the sheet confirms it is *not* a
  Class-1 `dynamic_tensor_shape` customer). **Port action: drop the 3rd arg (pure no-op).**
- **Reachability:** the page-table path is used only by SDPA's chunked/paged branch; JointSDPA does not call it.

---

## Port-work summary  *(mirrors the brief — for the cleared subset `SDPADeviceOperation` + `JointSDPADeviceOperation`)*

- **Tensor bindings** (per binding, all **Case 1** — via `TensorAccessor`; addresses arrive as `Buffer*`-binding
  `uint32_t` RTAs and are immediately wrapped in `TensorAccessor(args, addr)`, so express each as a
  `TensorParameter` / `TensorBinding` and build `TensorAccessor(tensor::name)`):
  - **SDPA** (`reader_interleaved.cpp:211-216`, `writer_interleaved.cpp:84,114`): `q_in`, `k_in`, `v_in`, `mask` (optional),
    `page_table` (chunked), `attention_sink` (optional), `chunk_start_idx` (flexible-chunked); `out`, `cu_window_seqlens` (windowed).
  - **JointSDPA** (`joint_reader.cpp:56-61`, `joint_writer.cpp:52-53`): `q`, `k`, `v`, `joint_q`, `joint_k`, `joint_v`; `out`, `joint_out`.
  - No Case 2 (raw-pointer) bindings; no borrowed-memory (`borrowed_from`) DFB reads.
- **TensorParameter relaxation:** `none` (both) — clears.
- **TensorAccessor 3rd arg:** SDPA — drop the redundant page-size arg @ `dataflow_common.hpp:83` (Class 2, pure no-op;
  do **not** set `dynamic_tensor_shape`). JointSDPA — none.
- **CB endpoints** (per `(CB, config)`; no dead CB, no multi-binding):
  - **SDPA** — *self-loop* (single toucher): compute intermediates `qk_im`, `out_im_A`, `out_im_B`, `max_A`, `max_B`,
    `sum_A`, `sum_B`, `exp_max_diff` (always), plus `page_table` (chunked), `cu_window_seqlens` (windowed),
    `recip_scratch` (streaming). *Plain 1P+1C* (legal, no action): `q_in`, `k_in`, `v_in`, `mask_in`,
    `identity_scale_in`, `col_identity`, `chunk_start_idx_compute`, `chunk_start_idx_writer`, `attention_sink`, `out`.
    *Conditional DFB* (created only in some configs — make the spec conditional): `mask_in`, `page_table`,
    `attention_sink`, `chunk_start_idx_*`, `recip_scratch`, `cu_window_seqlens`.
  - **JointSDPA** — *self-loop*: `cb_qk_im` (c_24), `cb_out_im_A` (c_25), `cb_out_im_B` (c_26), `cb_max_A` (c_27),
    `cb_max_B` (c_28), `cb_sum_A` (c_29), `cb_sum_B` (c_30), `cb_exp_max_diff` (c_31). *Plain 1P+1C*: `cb_q_in` (c_0),
    `cb_k_in` (c_1), `cb_v_in` (c_2), `cb_mask_in` (c_3, conditional), `cb_identity_scale_in` (c_5), `cb_col_identity`
    (c_7), `cb_out` (c_16, compute→writer). *Conditional DFB*: `cb_mask_in` (only if `use_joint_mask`).

---

## Heads-ups  *(mirrors the brief — cleared subset)*

- **CB endpoints (no multi-binding to watch), but two SDPA shapes to keep straight:**
  - **KV-chain cross-core semaphore forwarding** (SDPA, **non-causal only**): `k_in`/`v_in` are FIFO-produced locally by
    the reader, and a *peer core's* reader instance writes the tiles into this core's CB via `noc.async_write` +
    `sender`/`receiver`/`valid` semaphores (`reader_interleaved.cpp:412-420,474-517,604-612,665-709`; semaphores
    declared `sdpa_program_factory.cpp:837-856`). This is the **same reader source on a different node**, not a second
    on-node kernel — per node the census is still 1 producer (reader) + 1 consumer (compute), so it is **plain 1P+1C,
    not multi-binding**. The three semaphores port as ordinary `SemaphoreSpec`s; the cross-core writes are faithful.
  - **`mask_in` producer flips by config** (SDPA): the **reader** produces it when `use_provided_mask`, the **writer**
    produces it otherwise (generated / lightweight / windowed) — mutually exclusive, so 1P+1C per config, but the
    producer *kernel* differs between configs.
  - **`cu_window_seqlens` CTA index aliases `q_in`** when not windowed (`writer_interleaved.cpp:778`) — a benign alias
    so `get_tile_size` stays well-formed; carry the conditional DFB accordingly.
- **Cross-op / shared kernels** (function-call escapes; no file-path kernel borrows — SDPA/Joint own all kernels):
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` — `calculate_and_prepare_reduce_scaler<uint32_t dfb_id, …>()`
    is **Device-2.0-native** (`uint32_t` cb/dfb-id template param → handled by `dfb::name`'s constexpr cast).
    **No donor-side change.** (Called `writer_interleaved.cpp:90`, `joint_writer.cpp:63`.)
  - `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` — `generate_bcast_col_scalar(CircularBuffer cb, …)` takes
    the **legacy `CircularBuffer`** handle, **but a real `_metal2` fork `generate_bcast_scalar_metal2.hpp` (taking
    `DataflowBuffer`) already exists beside it** (in `ttnn/cpp/ttnn/kernel/dataflow/`, **not** quasar). The port swaps
    the include to the fork and passes a named `DataflowBuffer` built from the token. (Called `writer_interleaved.cpp:95`,
    `joint_writer.cpp:68`.)
- **RTA varargs:** **none** — every kernel reads a fixed, enumerable set of args (the `if (num_phases == 2)` and
  `if constexpr (!is_causal)` blocks are fixed-width optional-field reads, not variable-count loops). Name them all.
- **Compute kernels** (`compute/sdpa.cpp`, `compute/joint_sdpa.cpp`) touch only CBs — out of scope for tensor-binding
  work.

---

## Team-only

### Out-of-directory coupling & donor shape (full inventory)

**Roll-up: ⚠ workable** (no ⭐ scheduling blocker for the subset — the one legacy-CB donor already has a `_metal2` fork).

| Op kernel | Donor file | Class | Shape | Status |
|---|---|---|---|---|
| `writer_interleaved.cpp`, `joint_writer.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | shared kernel_lib | `uint32_t dfb_id` (template) | ✓ Device-2.0-native |
| `writer_interleaved.cpp`, `joint_writer.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | shared `kernel/` pool | `CircularBuffer` (by value) | ⚠ legacy-CB signature; `_metal2` fork exists → bind the fork |
| all subset kernels | `tt_metal/*` (`api/dataflow/*`, `api/compute/*`, `api/tensor/*`, `api/core_local_mem.h`) | LLK / HAL | — | ✓ no concern |
| all subset kernels | op-local headers (`dataflow_common.hpp`, `compute_common.hpp`, `compute_streaming.hpp`, `sdpa_interleaved_cb_ids.hpp`, …) | in-directory | — | in scope, audited |

`reduce_helpers_dataflow.hpp` body uses `DataflowBuffer dfb(dfb_id)`, `noc.async_write_zeros(dfb, …)`,
`get_dataformat(dfb_id)` — fully Device 2.0. `generate_bcast_scalar.hpp` uses the `CircularBuffer` *wrapper* (Device
2.0-compliant idioms — `cb.reserve_back` / `cb.get_write_ptr` / `cb.push_back` + L1 fills), so it is **not** a Device
2.0 *gate* violation; the concern is only the Metal 2.0 *syntax* (its parameter is the legacy CB type), and its header
self-documents the fork. **Sunset note:** other consumers of the legacy `generate_bcast_scalar.hpp` are the standard
kernel-library sunset list — not an authorization to convert the header in place.

### Relaxation candidates (FYI-U — fallible; the ops team owns the real analysis)

SparseSDPA / SparseSDPAMSA carry a custom `compute_program_hash` and are held on `TensorParameter relaxation = dynamic`.
Whatever tensor properties those custom hashes key on are candidate inputs for the ops team's relaxation analysis — not
verified here (custom hashes are frequently wrong). Do not mine them into the port.

### TTNN factory analysis (sheet-derived facts + code evidence)

- **Op-owned tensors:** none (all seven).
- **MeshWorkload need:** RingJoint / ExpRingJoint are *genuine* multi-program (per-coord distinct descriptors) — not an
  op-owned-tensor artifact. RingDistributed is per-coord-RTA SPMD.
- **Pybind `create_descriptor`:** none.
- **Custom hash:** RingJoint (`ring_joint_sdpa_device_operation.cpp:879`), SparseSDPA, SparseSDPAMSA — left intact by
  any future port.
- **`override_runtime_arguments`:** SparseSDPA (`sparse_sdpa_device_operation.hpp:32`), SparseSDPAMSA
  (`sparse_sdpa_msa_device_operation.hpp:38`), RingJoint — selects `CustomProgramSpecFactoryConcept` for those ops.
- **`get_dynamic_runtime_args`:** none.

---

## Misc anomalies  *(team-only, non-gating)*

- The ring/joint factories carry explicit `// smuggled-rta-ok: …` annotations at their address-RTA sites
  (`exp_ring_joint_sdpa_program_factory.cpp:1562`, `ring_joint_sdpa_program_factory.cpp:2565-2581`). These correspond to
  the readiness sheet's `Smuggled pointer = yes` / "Broken Op" classification for those DeviceOperations — flagged there
  for the TTNN fix; recorded here only so the annotations are not mistaken for an all-clear.

## Per-DeviceOperation attribution

| DeviceOperation | Concept → target | `Is able to port?` | Blocker(s) → route |
|---|---|---|---|
| `SDPADeviceOperation` | descriptor → `ProgramSpecFactoryConcept` | **yes** | — (portable; brief issued) |
| `JointSDPADeviceOperation` | descriptor → `ProgramSpecFactoryConcept` | **yes** | — (portable; brief issued) |
| `SparseSDPADeviceOperation` | descriptor → `CustomProgramSpecFactoryConcept` | **no** | `TensorParameter relaxation = dynamic` → ops team |
| `SparseSDPAMSADeviceOperation` | descriptor → `CustomProgramSpecFactoryConcept` | **no** | `TensorParameter relaxation = dynamic` → ops team |
| `RingDistributedSDPADeviceOperation` | WorkloadDescriptor (secretly-SPMD) | **no** | per-coord-RTA / SPMD-adapter gap → framework |
| `RingJointSDPADeviceOperation` | WorkloadDescriptor (genuine MP) | **no** | GlobalSemaphore (feature) + "Broken Op"/smuggled ptr (TTNN) + genuine multi-program (framework) |
| `ExpRingJointSDPADeviceOperation` | WorkloadDescriptor (genuine MP) | **no** | GlobalSemaphore (feature) + "Broken Op"/smuggled ptr (TTNN) + genuine multi-program (framework) |

## Questions for the user  *(none)*

## Recipe notes

- **`Secretly SPMD` cross-check heuristic is coarse for N-entry per-coord-RTA workloads.** The recipe's cross-check
  says "a single entry in its `programs` vector ⇒ SPMD." `RingDistributedSDPADeviceOperation` builds **N** per-coord
  `ProgramDescriptor`s in a loop (`ring_distributed_sdpa_program_factory.cpp:598-601`), which a literal reading of the
  heuristic would classify as **not** secretly-SPMD — yet the sheet marks it secretly-SPMD = `yes` ("Per-coord RTAs"),
  because the per-coord descriptors differ *only* by runtime args (morally single-program). The single-entry test is a
  sufficient condition, not a necessary one; a note that "multiple entries differing only in RTAs are still SPMD" would
  prevent an auditor from wrongly reporting a spreadsheet conflict here. I deferred to the sheet's finer judgment
  (as instructed for a `Secretly SPMD` value on a `WorkloadDescriptor` op) and did not flag it as broken.
- **Bundled multi-DeviceOperation op with a partial subset.** This directory holds 7 independent DeviceOperations
  sharing only the kernel utility layer. The "clean factory subset survives" rule worked cleanly, but note the subset
  here is *per-DeviceOperation* (2 of 7 whole DeviceOperations clear), not the more common *per-factory-within-one-DOp*
  scope the Code-path-scope examples describe. The brief is scoped to the two cleared DeviceOperations.
