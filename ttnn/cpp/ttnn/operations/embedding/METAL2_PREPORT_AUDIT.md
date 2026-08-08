# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/embedding/`

Single device operation, three program factories sharing a common helper and a common
kernel-side header:

- **`EmbeddingsDeviceOperation`** (`device/embedding_device_operation.{hpp,cpp}`)
  - `EmbeddingsFusedProgramFactory` (`device/embeddings_fused_program_factory.cpp`) — selected when `tilized && input layout != TILE`. Reader `embeddings_tilize.cpp`; compute `tilize_chunked.cpp` (chunked) or `data_movement/tilize/.../tilize.cpp` (non-chunked); writer (interleaved only) `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`.
  - `EmbeddingsRMProgramFactory` (`device/embeddings_rm_program_factory.cpp`) — selected when input is ROW_MAJOR and not tilized. Reader `embeddings.cpp`; writer `kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` (non-chunked) or op-owned `embeddings_rm_writer_chunked.cpp` (chunked).
  - `EmbeddingsTilizedIndicesProgramFactory` (`device/embeddings_tilized_indices_program_factory.cpp`) — selected when input layout == TILE. Reader `embedding_ind_tilized.cpp`; writer `kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp`.
- Shared host helper: `device/embedding_program_factory_common.{hpp,cpp}` (`split_work_to_cores_aligned`).
- Shared kernel header: `device/kernels/dataflow/embeddings_common.hpp` (`prepare_local_cache`, `read_token_async`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/embedding/` |
| **Overall** | RED |
| **DOps / Factories** | `EmbeddingsDeviceOperation` → `EmbeddingsFusedProgramFactory`, `EmbeddingsRMProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No — donor writer `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` is wholesale Device 1.0 (used by RM + tilized-indices factories) |
| *Prereqs* — Cross-op escapes | Ok (workable; one ⭐ blocker = the Device 2.0 donor above) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (all CTAs fixed count) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

**Fake CBs** = CBs used purely as an address source (none here). Every CB has a producer and a consumer.

## Result

**RED at op level — blocked on the Device 2.0 prerequisite**, routed to the Device 2.0 team.

The single blocker is the shared interleaved row-major writer
`ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp`
(`ttnn/cpp/ttnn/kernel/` shared pool), which is still entirely on Device 1.0 data-movement
idioms (`get_read_ptr(cb_id)`, raw `noc_async_write`, `cb_wait_front`/`cb_pop_front` against a
bare CB index — no `CircularBuffer` / `Noc` wrapper in scope). This is *not* an isolated
CB-index holdover (the YELLOW carve-out): the whole kernel is Device 1.0, so the Metal 2.0
binding tokens have nothing to attach to. It blocks the **RM** and **tilized-indices**
factories. The port unblocks once that donor's Device 2.0 migration lands.

**Clean subset:** `EmbeddingsFusedProgramFactory` is fully Device 2.0 today (its interleaved
writer is `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`, already Device 2.0; its
sharded path uses no writer). A scoped subset port of the fused factory alone is feasible now —
but note the device operation's `select_program_factory` dispatches to all three from one
`program_factory_t` variant, so a subset port would land one factory's spec while the other two
remain on `ProgramDescriptor`; that is a port-planning decision for the user, not a structural
blocker.

This RED gates *this* port attempt only; it is the expected outcome when a borrowed donor kernel
has not yet completed Device 2.0 migration, and is not a permanent blocker.

## Gate detail

- **ProgramDescriptor:** GREEN. `EmbeddingsDeviceOperation` is a `ttnn::prim` device operation with a
  `program_factory_t = std::variant<...>` and three factories that each populate a
  `tt::tt_metal::ProgramDescriptor` via `CBDescriptor`, `KernelDescriptor`, `TensorAccessorArgs`
  (`embeddings_fused_program_factory.cpp:18`, `embeddings_rm_program_factory.cpp:18`,
  `embeddings_tilized_indices_program_factory.cpp:19`). No imperative `host_api.hpp` builder calls
  (`CreateProgram`/`CreateKernel`/`CreateCircularBuffer`/`SetRuntimeArgs`) appear in factory bodies.
  Note: `embedding_program_factory_common.hpp:11` `#include <tt-metalium/host_api.hpp>` is a header
  include only (pulls in `split_work_to_cores` / `CoreRangeSet`); no imperative builder API is used.

- **Device 2.0 (every kernel used):** RED. Inventory of every kernel the op exercises:

  | Kernel | Owner | Device 2.0? |
  |---|---|---|
  | `embeddings_tilize.cpp` (reader, fused) | embedding (own) | Yes — `Noc`, `CircularBuffer`, `CoreLocalMem`, `TensorAccessor`, `noc.async_read` |
  | `embeddings.cpp` (reader, RM) | embedding (own) | Yes — same idioms |
  | `embedding_ind_tilized.cpp` (reader, tilized-idx) | embedding (own) | Yes — same idioms |
  | `embeddings_common.hpp` (shared reader helper) | embedding (own) | Yes — `Noc`, `CircularBuffer`, `CoreLocalMem`, `UnicastEndpoint` |
  | `embeddings_rm_writer_chunked.cpp` (writer, RM chunked) | embedding (own) | Yes — `Noc`, `CircularBuffer`, `TensorAccessor`, `noc.async_write` |
  | `tilize_chunked.cpp` (compute, fused chunked) | embedding (own) | Yes — `compute_kernel_lib::tilize`, CB-index CTAs (sanctioned) |
  | `data_movement/tilize/.../tilize.cpp` (compute, fused non-chunked) | data_movement (donor, shared) | Yes — `compute_kernel_lib::tilize` |
  | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (writer, fused interleaved) | eltwise/unary (donor) | Yes — `Noc`, `CircularBuffer`, `TensorAccessor`; `get_local_cb_interface(cb_id)` is sanctioned |
  | **`kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp`** (writer, RM non-chunked + tilized-idx) | **`ttnn/cpp/ttnn/kernel/` shared pool** | **NO — Device 1.0** |

  Device 1.0 violations in the donor stick writer:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 26 | `cb_wait_front(cb_id_out0, 1)` | none |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 27 | `get_read_ptr(cb_id_out0)` | none |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 29 | `noc_async_write(l1_read_addr, dst_noc_addr, stick_size)` | none (no `Noc`) |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 30 | `noc_async_write_barrier()` | none |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 31 | `cb_pop_front(cb_id_out0, 1)` | none |

  These are the canonical legacy idioms the [Device 2.0 migration guide](../../../../../docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md) replaces with `Noc` / `CircularBuffer` wrappers. No wrapper object is in scope at any call site, so this is the RED tier (wholesale Device 1.0), not the isolated-holdover YELLOW. The kernel does already construct a `TensorAccessor` (line 17), so the addr-gen side is modern, but the CB + NoC side is not. **Route to the Device 2.0 team naming this kernel file and the `ttnn/cpp/ttnn/kernel/` shared pool; co-borrowers are listed under Team-only, so the migration is one shared rewrite.**

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `.global_circular_buffer` field anywhere. |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | Output CB placed on the sharded output buffer in fused + RM factories — `out_cb_desc.buffer = out_buffer` (`embeddings_fused_program_factory.cpp:173`, `embeddings_rm_program_factory.cpp:134`). Port uses `DataflowBufferSpec::borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set; `out_cb_desc.buffer` set without offset. |
  | Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element. |
  | GlobalSemaphore | N/A | Op uses no semaphores. |
  | Non-zero semaphore initial value | N/A | Op uses no semaphores. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | All `TensorAccessorArgs(*buffer)` are the single-arg static form. |
  | `UpdateCircularBuffer*` | N/A | No update calls; no `override_runtime_arguments`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed struct (input + weight + optional output); all CTAs are fixed-count vectors; no `get_compile_time_arg_val(i)` runtime-index loop. |

## Port-work summary  *(mirrors the brief — not actionable until the gate clears)*

- **Tensor bindings** (per binding, all factories): all **Case 1** (via `TensorAccessor`). Each
  factory smuggles `input`, `weight`, and `output` base addresses through the
  `Buffer*`-binding RTA form (`reader_args.push_back(a_buffer)` / `weights_buffer` /
  `output_buffer`; e.g. `embeddings_fused_program_factory.cpp:321-336`,
  `embeddings_rm_program_factory.cpp:264-283`,
  `embeddings_tilized_indices_program_factory.cpp:209-224`). On the device side every base feeds a
  `TensorAccessor(args, addr)` constructor and all access goes through the accessor — Case 1, no raw
  pointer arithmetic. (The `Buffer*` push form is the framework's interim BufferBinding hack; it is
  patched on cache hits today, so it is routine port work, not a silent-wrong hazard.) Port: express
  each as a `TensorParameter`/`TensorBinding`; kernels build `TensorAccessor(ta::name)` and the
  address-via-RTA plus the `TensorAccessorArgs` CTAs disappear.
  - Sharded-output edge (fused + RM): the output binding is a **clean borrowed-memory DFB** (causal-link gate) — producer is the compute (fused) / reader (RM) pushing into the output CB; consumer is the L1-resident sharded output. Handle via `DataflowBufferSpec::borrowed_from`, not Case 1.
- **Custom hash:** none.

## Heads-ups  *(mirrors the brief — surfaced for when the gate clears)*

- **Notable LANDED constructs:** borrowed-memory DFB on the sharded-output path of the fused and
  RM factories (`embeddings_fused_program_factory.cpp:173`, `embeddings_rm_program_factory.cpp:134`)
  → `DataflowBufferSpec::borrowed_from` naming the output `TensorParameter`. No aliased CBs, no
  dynamic TA, no non-zero sem init.
- **Fake CBs (address-only):** none.
- **Cross-op / shared kernels:** the op file-path-instantiates four kernels it does not own —
  `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (fused interleaved writer),
  `data_movement/tilize/.../tilize.cpp` (fused non-chunked compute),
  `kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` (RM + tilized-idx writer,
  **the Device 2.0 blocker**), and `kernel_lib/tilize_helpers.hpp` (via the compute kernels). The
  stick writer is broadly shared → **port-together set** (see Team-only). The fused interleaved
  writer is also shared but already Device 2.0.
- **RTA varargs:** none. (The reader RTA count varies by one element for the PADDED embeddings type —
  `reader_args.push_back(pad_token.value())` — but that is host-side conditional arg construction,
  not a kernel-side runtime-varying loop; the kernel reads `pad_token` at a fixed arg index. Not an
  RTA-vararg signal.)
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other risky pybind,
  no custom `override_runtime_arguments`.

## Team-only

- **Out-of-directory coupling & donor shape.**
  - **Op-level roll-up:** ⭐ blocked — one donor kernel is pre-Device-2.0 (the stick writer);
    everything else is ✓ clean or workable.
  - **Function-call escapes (kernel `#include`s outside the op dir):** all op reader/compute kernels
    include only `api/...` (Device 2.0 / `tt_metal/*` — no concern) plus the op's own
    `embeddings_common.hpp`. The compute kernels include `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
    (official shared kernel library — lib team handles internally). No cross-family helper *function*
    calls with risky resource-handle shapes. Roll-up: ✓ clean on function-call escapes.
  - **File-path kernel instantiation (borrowed `.cpp` the factories `CreateKernel`):**

    | Borrowed kernel file | Owning family / pool | Shared? |
    |---|---|---|
    | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary | broadly shared (used by fused interleaved writer; many unary/eltwise ops) — already Device 2.0 |
    | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | data_movement/tilize | broadly shared — already Device 2.0 |
    | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | `ttnn/cpp/ttnn/kernel/` shared pool | **broadly shared — port-together set** (below); **Device 2.0 blocker** |

    **Port-together set for the stick writer** (every op that file-path-instantiates it must adopt
    the same Metal 2.0 rewrite in one change, and all share the Device 2.0 migration dependency):
    - `ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp`
    - `ttnn/cpp/ttnn/operations/embedding/device/embeddings_tilized_indices_program_factory.cpp`
    - `ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/indexed_fill_program_factory.cpp`
    - `ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_same_memory_config_program_factory.cpp`
    - `ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp`

- **Relaxation candidates:** none mined (no custom hash to mine).

- **TTNN factory analysis (six questions):**
  1. **Op-owned tensors?** No. No factory creates intermediate/scratch device tensors;
     `create_output_tensors` (`embedding_device_operation.cpp:129`) only returns the optional output
     or one `create_device_tensor` for the declared output. CBs are L1 buffers, not device tensors.
  2. **MeshWorkload needed?** No. Each factory builds a single `ProgramDescriptor`; no
     `create_mesh_workload` / `cached_mesh_workload_t`. Not on the MeshWorkload path.
  3. **Pybind `create_descriptor`?** No. `embedding_nanobind.cpp` binds only the user-facing
     `embedding` function (`bind_function<"embedding">`, line 40) and `export_enum<EmbeddingsType>`
     (line 19) — the normal op surface, not a finding.
  4. **Other migration-risky pybind?** None. No `nb::class_<>` wrapping a `DeviceOperation` or
     factory/param struct.
  5. **Custom hash?** No (`compute_program_hash` absent — cross-ref Custom program hash subject).
  6. **Custom override-runtime-args?** No (`override_runtime_arguments` absent in all three factories).

## Misc anomalies  *(team-only, non-gating)*

- `embedding_ind_tilized.cpp:11` includes `api/debug/dprint.h` but the kernel issues no `DPRINT`
  calls — a leftover debug include. Harmless; route to the op owner, not the port.
- `embedding_ind_tilized.cpp` reads `pad_token` at arg index 6 via `prepare_local_cache(..., /*pad_token_arg_idx=*/6)`, but the reader RTA layout for this factory pushes `pad_token` at index 7 (after `col_offset` at 5 and `starting_index` at 6 — `embeddings_tilized_indices_program_factory.cpp:214-217`). Worth the op owner confirming the PADDED-embeddings path for tilized indices is correct; it does not affect the Metal 2.0 port mechanics. (Flagged as an observation only — not verified against a failing test.)

## Recipe notes

- The audit's `Buffer*`-binding-form detection bullet (Detection — host side) says to enumerate
  `Buffer*` pushes as Case 2 ("the kernel consumes a raw `uint32_t` base"). Here every such pushed
  base is immediately consumed by a `TensorAccessor(args, addr)` constructor on the kernel side, which
  the two-cases section classifies as **Case 1**. The two bullets are individually consistent (the
  `Buffer*` bullet is about the host smuggling shape; the Case-1/2 split is about kernel use), but a
  reader could mistake "push_back(buffer)" for an automatic Case 2. Resolved here in favor of the
  kernel-side rule (Case 1, since the address feeds a `TensorAccessor`). Noting in case the wording
  could call this out explicitly.

---

## ⚠️ Post-port-attempt correction (2026-06-25) — second, framework-level blocker

The Device-2.0 donor prerequisite called out above was resolved (`writer_unary_stick_layout_interleaved_start_id.cpp` migrated to D2.0, **PR #48147**, validated). A re-audit *for the actual port* then surfaced a **second, independent blocker that this audit missed**, putting the op back at **RED — framework-blocked** (not portable):

- The RM reader (`device/kernels/dataflow/embeddings.cpp`) self-stages the input indices in **`cb_in1`** (`reserve_back`/`push_back` + `get_write_ptr`, then raw self-read) with **no cross-kernel consumer** — a **DM-kernel sync-free CB**. (Check the other factories' readers for the same shape.)

Metal 2.0 requires one producer + one consumer per local DFB, has **no scratch/sync-free DFB** (`DataflowBufferSpec` = `borrowed_from`/`alias_with` only), and the single-ended **self-loop workaround is compute-kernel-only** — binding it on a data-movement kernel FATALs (`"self-looped by data-movement kernel"`). A DM-kernel sync-free CB is therefore **framework-inexpressible today**. See [[metal2-port-portability-predictor]].

**Corrected status: RED, framework-blocked** (wait-for-feature: a sync-free/scratch DFB or DM-kernel self-loop). The #48147 prereq is necessary but **not sufficient**; do not attempt the port until the sync-free-CB feature lands.

---

## 🔄 Revision (2026-06-25, supersedes the correction above) — workaround found; NOT framework-blocked

The "framework-blocked / wait-for-feature" verdict above is **overstated**. A workaround exists with **no framework change**: the **cross-kernel DFB bridge**. Only a DM-kernel *self*-loop FATALs; a DM kernel paired *cross-kernel* with a different co-located kernel (DM↔DM or DM↔compute) is fully legal. **Proven in shipped code:** the landed JointSDPA port (PR #48175, 160 passed/0 failed) binds `mask`/`scale`/`col_identity` as PRODUCER on the **writer (DM)** → CONSUMER on **compute** (`joint_sdpa_program_factory.cpp:359-451`); the SPSC validator accepts and runs them.

**This op:** the RM reader's `cb_in1` is **read-staging** (reads a DRAM chunk, indexes via `CoreLocalMem`). Relocate the read to a producer endpoint paired with the consumer — a real data dependency, directly analogous to the proven JointSDPA reader→compute bindings. **PORTABLE via cross-kernel bridge** (proven-class); no framework feature needed. The #48147 D2.0 prereq remains a real prerequisite.
