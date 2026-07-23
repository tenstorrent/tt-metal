# Metal 2.0 Audit Findings — `data_movement/moe_expert_token_remap`

- **`MoeExpertTokenRemapDeviceOperation`**
  - `Multicore` (`device/moe_expert_token_remap_program_factory.cpp`)

Single device operation, single program factory. The factory instantiates **two** kernels:

- **Writer (op-owned):** `device/kernels/dataflow/writer_moe_expert_token_remap.cpp`.
- **Reader (cross-family donor):** `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp` — file-path-instantiated from the `ccl/all_to_all_combine` op.

No unreferenced kernel files in the op directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `05554b94288 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MoeExpertTokenRemapDeviceOperation` → `Multicore` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — both kernels are structurally Device 2.0 (`Noc`, `CircularBuffer`/`DataflowBuffer` wrappers, `TensorAccessor`, `CoreLocalMem`); no CB-index free-function holdovers; no legacy addr-gen |
| *Prereqs* — Cross-op escapes | Ok — one cross-family borrowed reader kernel + two shared header helpers, all clean shapes (port-together coupling noted) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (cleared) |
| *Port work* — Tensor bindings (per binding) | 5 bindings, all Case 1 (`TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd arg) |
| *Port work* — CB endpoints | 3 self-loop, 3 legal 1:1 |

**CB endpoints** are dispositions, not gates: every CB here is either a legal 1:1 FIFO or a single-ended scratch CB resolved by a self-loop. No multi-binding, no dead CB. Single config (no sharding variants).

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Features ✓ · TTNN factory concept (`Is able to port?` = yes) ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. The op is on the `descriptor` `ProgramDescriptor` concept and ports to `MetalV2FactoryConcept`. Port work is the ordinary tensor-binding translation (5 Case-1 bindings) plus CB self-loops on the three single-ended scratch buffers. `METAL2_PORT_BRIEF.md` is written alongside this file.

One coupling to carry into planning (does not gate): the reader kernel is borrowed by file path from `ccl/all_to_all_combine`, so its Metal 2.0 rewrite is a single change shared with that op — see [Team-only](#team-only).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. The readiness sheet's row for `data_movement/moe_expert_token_remap / MoeExpertTokenRemapDeviceOperation / Multicore` reads `Is able to port? = yes`, and every cheaply-checkable column matches the code:
  - `Concept = descriptor` — confirmed: `Multicore::create_descriptor(...)` returns a `tt::tt_metal::ProgramDescriptor` ([moe_expert_token_remap_device_operation.hpp:42-48](device/moe_expert_token_remap_device_operation.hpp#L42-L48), [moe_expert_token_remap_program_factory.cpp:18-22](device/moe_expert_token_remap_program_factory.cpp#L18-L22)). The `mesh_dispatch_coordinate` parameter is the per-coord descriptor-adapter path — the return type is a plain `ProgramDescriptor`, not a `WorkloadDescriptor`, so the concept is `descriptor`.
  - `Custom hash = no` — confirmed: no `compute_program_hash` override; `validate_on_program_cache_hit` is empty ([moe_expert_token_remap_device_operation.hpp:58-59](device/moe_expert_token_remap_device_operation.hpp#L58-L59)).
  - `Runtime-args update = no` — confirmed: no `override_runtime_arguments` / `get_dynamic_runtime_args`; args are set once inside `create_descriptor` via `emplace_runtime_args` ([moe_expert_token_remap_program_factory.cpp:230-237](device/moe_expert_token_remap_program_factory.cpp#L230-L237)).
  - `Pybind descriptor = no` — confirmed: the nanobind module binds only the free function `moe_expert_token_remap`, no `create_descriptor` binding ([moe_expert_token_remap_nanobind.cpp:45-56](moe_expert_token_remap_nanobind.cpp#L45-L56)).
  - Cross-column invariants hold: `Op-owned tensors?` blank/no is consistent with the `descriptor` concept; `Runtime-args update = no` is consistent with any concept.
  - `Is safe to port? = yes` (the readiness-sheet owner's correctness axis — not re-derived here).

- **Device 2.0 (every kernel used):** GREEN. Both kernels the factory instantiates are structurally Device 2.0:
  - **Writer** ([writer_moe_expert_token_remap.cpp](device/kernels/dataflow/writer_moe_expert_token_remap.cpp)): `Noc noc;` with `noc.async_write` / `noc.async_write_barrier`; `DataflowBuffer` wrapper objects (`local_experts_dfb`, `metadata_dfb`, `data_dfb`, `output_mapping_dfb`, `output_reduced_dfb`) with `reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_read_ptr`/`get_write_ptr` **method** calls; `TensorAccessor(args, base_addr)`; `CoreLocalMem<uint32_t>`. No CB-index free functions, no legacy addr-gen.
  - **Reader** ([reader_all_to_all_combine.cpp](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp)): `Noc noc;` with `noc.async_read` / `noc.async_read_barrier`; `CircularBuffer` wrapper objects (`mapping_cb`, `local_experts_cb`, `metadata_cb`, `data_cb`) with method calls; `TensorAccessor`; `CoreLocalMem`. The reader's executed code path uses no fabric / no `GlobalSemaphore`, despite the `moe_utils.hpp` include (only `find_if` is called from it).
  - Helper functions called from shared headers are Device 2.0 native: `tt::data_movement::common::tt_memmove` is called via the **`Noc`-parameter overload** ([writer_moe_expert_token_remap.cpp:83-84](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L83-L84)), not the deprecated no-`Noc` form; `find_if` and `fill_with_val` touch only raw L1 pointers.

  No violations.

- **Feature compatibility:** every Appendix A entry scanned against host code, factory, and both kernels. All absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | All CBs are plain `CBDescriptor`s ([moe_expert_token_remap_program_factory.cpp:66-145](device/moe_expert_token_remap_program_factory.cpp#L66-L145)); no `global_circular_buffer` field, no `remote_cb`/`.remote_index`, no `CreateGlobalCircularBuffer`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset`; no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp`. The op uses no semaphores at all; reader/writer coordinate purely via CB FIFO. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed set of named tensors ([moe_expert_token_remap_device_operation.hpp:30-36](device/moe_expert_token_remap_device_operation.hpp#L30-L36)), not a `std::vector<Tensor>`. Both kernels read CTAs at **constexpr** indices only (`get_compile_time_arg_val(0..13)` plus `TensorAccessorArgs<N>` at constexpr offsets); no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** every CB carries a port-time disposition; nothing blocks. Census is per node — both kernels are placed over the same `total_cores`, so each node hosts reader + writer.

  | CB (index) | Definition | Touchers on a node | Disposition |
  |---|---|---|---|
  | mapping (`c_0`) | [factory:66-74](device/moe_expert_token_remap_program_factory.cpp#L66-L74) | reader only — producer + raw self-read ([reader:96-101](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L96-L101)) | **self-loop** (1 toucher) |
  | local_experts (`c_1`) | [factory:83-91](device/moe_expert_token_remap_program_factory.cpp#L83-L91) | reader produces ([reader:92-102](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L92-L102)), writer consumes ([writer:60-117](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L60-L117)) | **legal 1:1** |
  | metadata (`c_2`) | [factory:97-105](device/moe_expert_token_remap_program_factory.cpp#L97-L105) | reader produces ([reader:104-138](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L104-L138)), writer consumes ([writer:65-115](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L65-L115)) | **legal 1:1** |
  | topk/data (`c_3`) | [factory:111-119](device/moe_expert_token_remap_program_factory.cpp#L111-L119) | reader produces ([reader:121-131](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L121-L131)), writer consumes ([writer:76-98](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L76-L98)) | **legal 1:1** |
  | output_mapping (`c_4`) | [factory:124-132](device/moe_expert_token_remap_program_factory.cpp#L124-L132) | writer only — staging scratch ([writer:49-51](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L49-L51)) | **self-loop** (1 toucher) |
  | output_reduced (`c_5`) | [factory:137-145](device/moe_expert_token_remap_program_factory.cpp#L137-L145) | writer only — staging scratch ([writer:54-56](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L54-L56)) | **self-loop** (1 toucher) |

  No hidden second writer: for `c_1`/`c_2`/`c_3` the writer only ever reads (no raw `get_write_ptr` co-fill by a non-producer); the three producer CBs each have exactly one FIFO producer and one FIFO consumer. No dead CB.

- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into its base. Every tensor is delivered to a kernel via the **`Buffer*`-binding form** (a `Buffer*` pushed into `emplace_runtime_args`, not `buffer()->address()`), so there is no host-side `base + offset` arithmetic anywhere:
  - reader RTAs `{mapping_buffer, metadata_buffer, topk_buffer, page_idx_start, page_idx_end}` ([factory:230-231](device/moe_expert_token_remap_program_factory.cpp#L230-L231)) — slots 0-2 are bare `Buffer*`; slots 3-4 are page indices.
  - writer RTAs `{output_mapping_buffer, page_idx_start, page_idx_end, output_reduced_buffer, reduction_idx_start}` ([factory:236-237](device/moe_expert_token_remap_program_factory.cpp#L236-L237)) — slots 0 and 3 are bare `Buffer*`; the rest are indices.

  Not in the offset-base-pointer triage doc (a dated prior) — consistent with a clean scan. The kernel-side `data_dfb.get_read_ptr() + expert_idx * datum_size_bytes` ([writer:81](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L81)) is CB-internal L1 addressing, not a host-folded tensor base — not this gate.

- **TensorAccessor 3rd argument:** GREEN — no accessor in either kernel passes a 3rd (page-size) argument. All five `TensorAccessor(args, base_addr)` constructions are 2-arg ([reader:81-83](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L81-L83), [writer:38-39](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L38-L39)). Not in the 3rd-arg triage doc (a dated prior) — consistent with a clean scan.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — all **Case 1** (base fed into a `TensorAccessor`, all memory access through the accessor). Delivered today via the `Buffer*`-binding form (correct-on-cache-hit; the Metal 2.0 typed binding supersedes it). Port work: express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` and the RTA `Buffer*` + `TensorAccessorArgs` plumbing disappear.

  | Binding | Delivery (host) | Kernel use |
  |---|---|---|
  | `mapping_tensor` | reader RTA slot 0 (`Buffer*`) ([factory:230-231](device/moe_expert_token_remap_program_factory.cpp#L230-L231)) | `TensorAccessor(mapping_args, mapping_tensor_addr)` ([reader:82](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L82)) |
  | `metadata_tensor` | reader RTA slot 1 (`Buffer*`) | `TensorAccessor(metadata_args, metadata_tensor_addr)` ([reader:81](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L81)) |
  | `topk_tensor` (data) | reader RTA slot 2 (`Buffer*`) | `TensorAccessor(data_args, data_tensor_addr)` ([reader:83](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L83)) |
  | `output_mapping_tensor` | writer RTA slot 0 (`Buffer*`) ([factory:236-237](device/moe_expert_token_remap_program_factory.cpp#L236-L237)) | `TensorAccessor(output_mapping_args, output_mapping_base_addr)` ([writer:38](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L38)) |
  | `output_reduced_tensor` | writer RTA slot 3 (`Buffer*`) | `TensorAccessor(output_reduced_args, output_reduced_base_addr)` ([writer:39](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L39)) |

- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_0` (mapping), `c_4` (output_mapping staging), `c_5` (output_reduced staging); legal 1:1 for `c_1` (local_experts), `c_2` (metadata), `c_3` (topk/data). Single config.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader, no ≥3-toucher CB.
- **Cross-op / shared kernels:** the reader is `ccl/all_to_all_combine`'s `reader_all_to_all_combine.cpp`, borrowed by file path. Its Metal 2.0 rewrite (CB→DFB, named-token bindings) is one change shared with `all_to_all_combine` — port the two ops' use of this kernel as one unit. See [Team-only](#team-only).
- **RTA varargs:** none — both kernels read RTAs at fixed constant indices (0-4), no loop-indexed or data-selected reads.

## Team-only

- **Out-of-directory coupling & donor shape.**

  **Op-level roll-up:** ✓ clean on call shapes; one file-path port-together coupling to sequence.

  **Summary table** (op kernel → donor):

  | Op kernel | Donor / include | Class | Status |
  |---|---|---|---|
  | writer (own) | `ttnn/operations/ccl/common/kernels/moe_utils.hpp` | cross-family shared header | ✓ (only `find_if` called) |
  | writer (own) | `ttnn/operations/data_movement/common/kernels/common.hpp` | in-family shared pool | ✓ |
  | reader | `ttnn/operations/ccl/common/kernels/moe_utils.hpp` | cross-family shared header | ✓ (only `find_if` called) |
  | reader | `ttnn/operations/data_movement/common/kernels/common.hpp` | in-family shared pool | ✓ |
  | reader (file) | `ttnn/operations/ccl/all_to_all_combine/.../reader_all_to_all_combine.cpp` | cross-family borrowed **kernel file** | ⭐ port-together coupling |

  **Per-call detail:**
  - `find_if<T, Size, ReturnIdx>(volatile tt_l1_ptr T*, uint32_t)` from `moe_utils.hpp` — no resource handles in the signature; returns `bool` / `std::tuple`. Crosses cleanly (nothing to translate). `moe_utils.hpp` also defines a namespace-scope global `routing_state::polar_state` and a large fabric-helper library, but none of it is referenced by either kernel's executed path — only `find_if` is used.
  - `tt::data_movement::common::tt_memmove<...>(Noc, uint32_t, uint32_t, uint32_t)`, `fill_with_val<T>(uint32_t, uint32_t, T)`, `ByteSizeAddressType<N>` from `common.hpp` — Device 2.0 native (`Noc`-leading `tt_memmove` overload) or pure L1 pointer/type utilities. All clean.

  **Borrowed kernel files (file-path instantiation):**
  - `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp` — owned by op family `ccl/all_to_all_combine`. Also instantiated by `all_to_all_combine` itself ([all_to_all_combine_program_factory.cpp:250](../../ccl/all_to_all_combine/device/all_to_all_combine_program_factory.cpp#L250)). **Metal 2.0 port-together set: {`ccl/all_to_all_combine`, `data_movement/moe_expert_token_remap`}** — the shared kernel's CB→DFB/named-token rewrite must land in both ops in the same change, or the co-borrower breaks the instant one migrates in isolation. (Independent of the Device 2.0 gate, which the reader already clears.)

- **Relaxation candidates** (mined from a custom hash): none — the op has no custom hash.

- **TTNN factory analysis (sheet-derived facts + `file:line` evidence):**
  - Current concept: `descriptor` (`create_descriptor` returns `ProgramDescriptor`).
  - Op-owned tensors: none (sheet blank/no; consistent with the `descriptor` concept — outputs are ordinary `create_output_tensors` device tensors, [device_operation.cpp:94-105](device/moe_expert_token_remap_device_operation.cpp#L94-L105)).
  - MeshWorkload need: none — single-program `ProgramDescriptor`; per-device specialization is via the `mesh_dispatch_coordinate` adapter (`flat_mesh_idx` baked into a reader CTA, [factory:150-164](device/moe_expert_token_remap_program_factory.cpp#L150-L164)), not a `WorkloadDescriptor`.
  - Custom hash: no. Custom `override_runtime_arguments`: no. Pybind `create_descriptor`: no. Other risky pybind: no.
  - Target concept: `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **`std::ceil` over integer division is a no-op** ([device_operation.cpp:78-80](device/moe_expert_token_remap_device_operation.cpp#L78-L80)): `std::ceil(batch_seq / reduction_size)` computes `batch_seq / reduction_size` in **integer** arithmetic (both operands `uint32_t`) *before* `std::ceil` sees it, so the ceiling never rounds up. If the output-reduced height was meant to be a ceiling division (`(batch_seq + reduction_size - 1) / reduction_size`), this truncates when `batch_seq` is not a multiple of `reduction_size`. Not a port concern; routes to the ops team.
- **Consumer reads via the write pointer** ([writer_moe_expert_token_remap.cpp:65-67](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L65-L67)): the writer `wait_front`s `metadata_dfb` (consumer role) but then reads its contents through `get_write_ptr()` rather than `get_read_ptr()`. Works because the CB is effectively single-page here, but it is semantically the wrong accessor for a consumer. Cosmetic; note for the ops team.
- **Kernel placement vs. runtime-arg scope** ([factory:205-241](device/moe_expert_token_remap_program_factory.cpp#L205-L241)): both kernels are placed over the full `total_cores` grid, but `emplace_runtime_args` is called only on the `utilized_cores` subset returned by `split_work_to_cores_even_multiples`. If that subset is ever smaller than `total_cores`, the un-utilized cores run the kernel with unset RTAs. This is pre-existing behavior (not introduced or altered by a port) and may well be benign if the split always returns the full grid; flagged only because the auditor is well-placed to notice it. Routes to the ops team.

## Recipe notes

- The op is delivered its tensor bases entirely through the **`Buffer*`-binding form** (`Buffer*` pushed into `emplace_runtime_args`), never through `buffer()->address()`. The [Offset base pointers](#offset-base-pointers) and [TensorParameter analysis] subjects are both framed primarily around `->address()` RTAs and only secondarily mention the `Buffer*` form. For a `descriptor`-concept op that has fully adopted `Buffer*` bindings, the entire "resolve every address RTA" scan resolves to "there are no `->address()` RTAs; classify the `Buffer*` slots by kernel use." That worked cleanly here, but a reader expecting `->address()` expressions could momentarily think the scan found nothing to do. A one-line pointer in the Offset-base-pointers recognition section noting that a fully-`Buffer*`-migrated op is trivially clean (no fold possible) would save a beat.
