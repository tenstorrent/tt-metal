# Metal 2.0 Audit Findings — `data_movement/indexed_fill`

**Device operations and factories in this directory:**

- **`IndexedFillDeviceOperation`** (`ttnn::prim`)
  - `IndexedFillProgramFactory` (`device/indexed_fill_program_factory.cpp`)

Single DeviceOperation, single ProgramFactory. The factory selects among four internal code paths inside one `create_descriptor` (chosen by tensor geometry, not by separate factory variants):

| Path | `kernel_mode` | When |
|---|---|---|
| **Generic 2D-stride** | `MODE_GENERIC` (0) | interleaved input_a, or any non-`dim==0` geometry |
| **Native** | `MODE_NATIVE` (1) | `dim==0`, HEIGHT_SHARDED L1 input_a/output, one batch per core |
| **Shard-local (interleaved B)** | `MODE_SHARD_LOCAL_INTERLEAVED_B` (2) | `dim==0`, WIDTH/BLOCK_SHARDED L1 input_a/output, interleaved input_b |
| **Shard-local (sharded B)** | `MODE_SHARD_LOCAL_SHARDED_B` (3) | as above, input_b same WIDTH_SHARDED grid |

Kernels (all owned by this op, all file-path-instantiated from this directory):

- `device/kernels/dataflow/indexed_fill_reader.cpp` — unified reader, path chosen via the `mode` compile-time arg.
- `device/kernels/dataflow/indexed_fill_writer.cpp` — native / shard-local writer (CB aliased to output; wait/pop stub).
- `device/kernels/dataflow/indexed_fill_writer_strided.cpp` — generic-path scatter writer.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `ca0b78e9ad7 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/indexed_fill` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `IndexedFillDeviceOperation` → `IndexedFillProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all three kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`, endpoints); no free-function CB-index holdovers |
| *Prereqs* — Cross-op escapes | Ok — kernels include only `api/*`; no donor kernels, no borrowed kernel files |
| *Feature Support* — overall | GREEN — no Appendix A feature in use |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore / CTA-varargs | N/A / N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not a WorkloadDescriptor) |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none — no host-folded offset on any address |
| *Port work* — Tensor bindings (per binding) | see below — batch_ids Case 1; input_a/input_b Case 1 + Case 2 (config-dependent); output Case 1 + clean borrowed-DFB (config-dependent) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2 — redundant, `page_id ≡ 0`) @ `indexed_fill_reader.cpp:49` |
| *Port work* — CB endpoints | data CB legal 1:1 (borrowed-DFB in native/shard-local) · batch CB self-loop |

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear:

- **Device 2.0** — every kernel the op uses is structurally Device 2.0. No holdovers.
- **Feature compatibility** — no Appendix A feature present (all N/A).
- **TTNN factory concept** — readiness sheet `Is able to port? == yes`; every cross-checkable column matches the code.
- **Offset base pointers** — no address RTA folds a host-side offset into its base.
- **TensorAccessor 3rd argument** — the one site is Class 2 (redundant, `page_id ≡ 0`); a mechanical drop, not a gate.

The remaining subjects surface only port work and heads-ups (tensor bindings, CB endpoints), none gating.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (`data_movement/indexed_fill` / `IndexedFillDeviceOperation` / `IndexedFillProgramFactory`): `Concept = descriptor`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Runtime-args update (PD override_runtime_args) = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Is able to port? = yes`, `TensorParameter relaxation = none`, `Op-owned tensors? = (blank)`. Cross-check against the code, all consistent:
  - `Concept = descriptor` — `IndexedFillProgramFactory::create_descriptor(...)` returns a `ProgramDescriptor` @ [`indexed_fill_program_factory.cpp:123`](device/indexed_fill_program_factory.cpp#L123); the device op declares `program_factory_t = std::variant<IndexedFillProgramFactory>` @ [`indexed_fill_device_operation.hpp:19`](device/indexed_fill_device_operation.hpp#L19).
  - `Custom hash = no` — no `compute_program_hash` override in [`indexed_fill_device_operation.hpp`](device/indexed_fill_device_operation.hpp) / `.cpp`.
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` in the factory.
  - `Pybind descriptor = no` — [`indexed_fill_nanobind.cpp:35`](indexed_fill_nanobind.cpp#L35) binds only the free function `ttnn::indexed_fill`; no `create_descriptor` binding, no `nb::class_` of the device op.
  - Cross-column invariants hold (no op-owned tensors on a `descriptor` row; `Runtime-args update` absent as required).
- **Device 2.0 (every kernel used):** **GREEN.** All three kernels use Device 2.0 idioms exclusively:
  - Data movement via `Noc noc;` + `noc.async_read(...)` / `noc.async_write(...)` / `noc.async_read_barrier()` / `noc.async_writes_flushed()` / `noc.async_write_barrier()`.
  - Buffers via `DataflowBuffer` objects with method-form FIFO ops (`reserve_back` / `push_back` / `wait_front` / `pop_front` / `get_write_ptr`).
  - Addressing via `TensorAccessor` / `TensorAccessorArgs` and `UnicastEndpoint{}`.
  - The lone `get_write_ptr()` @ [`indexed_fill_reader.cpp:59`](device/kernels/dataflow/indexed_fill_reader.cpp#L59) is the **method** on `batch_dfb` (a `DataflowBuffer`), not the free-function `get_write_ptr(cb_id)` holdover.
  - No `noc_async_read`/`noc_async_write`, no `InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedAddrGenFast`, no raw semaphore addresses, no CB-index free functions.

  No violations table — the gate is clean.

- **Feature compatibility:** every Appendix A entry, in order. Each is UNSUPPORTED, so `N/A` means absent (not a vacuous pass).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `remote_*` idiom. The two CBs are plain `CBDescriptor`s with no `.global_circular_buffer` field set. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The native / shard-local CBs set `.buffer = output.buffer()` @ [`indexed_fill_program_factory.cpp:261`](device/indexed_fill_program_factory.cpp#L261), [`:273`](device/indexed_fill_program_factory.cpp#L273) (borrowed-memory pattern — a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, *not* this entry), but **no `.address_offset` is set** (defaults to 0). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op uses no semaphores at all (no `Semaphore`, no `GlobalSemaphore`). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed 3-tensor struct ([`indexed_fill_device_operation_types.hpp:21`](device/indexed_fill_device_operation_types.hpp#L21)); the reader reads CTAs only at constexpr offsets (`get_compile_time_arg_val(0..3)` plus fixed-count `TensorAccessorArgs<N>()`), no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** two CBs; both carry a clean port-time disposition. Classified per `(CB, config)`:
  - **Data CB (`cb_index = 0`)** — reader FIFO-produces (`dfb_in0.reserve_back` / `push_back`), a writer FIFO-consumes (`wait_front` / `pop_front`). Two touchers, one locked producer + one locked consumer → **plain 1:1, legal** in every path.
    - *Generic:* local double-buffered CB @ [`indexed_fill_program_factory.cpp:276`](device/indexed_fill_program_factory.cpp#L276); consumer is `indexed_fill_writer_strided.cpp`.
    - *Native / shard-local:* CB is **globally allocated to `output.buffer()`** @ [`:261`](device/indexed_fill_program_factory.cpp#L261) / [`:273`](device/indexed_fill_program_factory.cpp#L273) → borrowed-memory DFB (`borrowed_from`); consumer is the wait/pop stub `indexed_fill_writer.cpp`. Still 1P+1C.
  - **Batch CB (`batch_cb_index = 1`)** @ [`indexed_fill_program_factory.cpp:288`](device/indexed_fill_program_factory.cpp#L288) — touched **only** by the reader: `batch_dfb.reserve_back(1)` / `get_write_ptr()` / `push_back(1)`, then reads the staged indices back through `addr_ptr` ([`indexed_fill_reader.cpp:57-63`](device/kernels/dataflow/indexed_fill_reader.cpp#L57-L63)). One toucher (fills and self-reads) → **self-loop** (bind the reader both PRODUCER and CONSUMER) in every path. No writer or compute kernel touches it.

  Nothing here blocks a Gen1 port.

- **Offset base pointers:** **GREEN.** The factory delivers every buffer to the kernels as a raw `Buffer*` pushed into `emplace_runtime_args` (`batch_ids_buffer`, `input_a_buffer`, `input_b_buffer` to the reader; `output_buffer` to the strided writer) — the framework auto-registers these as `BufferBinding`s. **No site folds a host-side offset into an address** (`buffer()->address() + <offset>` never appears). Where the kernel needs an interior address (shard-local paths), it receives the **clean base** plus **separate scalar offset args** (`col_byte_offset`, `batch_offset_a`, and page-index arithmetic) and does the arithmetic device-side — exactly the split-out shape the gate wants, not a host fold. Not listed in the offset-base-pointer triage (`2026-07-19_offset_base_pointers.md`), consistent with this scan. No Type 1 / Type 2; no `ttnn::narrow` (Type 4).

- **TensorAccessor 3rd argument:** **GREEN — Class 2 (redundant), drop.** One site: the batch-ids accessor @ [`indexed_fill_reader.cpp:49`](device/kernels/dataflow/indexed_fill_reader.cpp#L49) — `TensorAccessor(batch_ids_args, batch_ids_addr, batch_id_size << 2)`. The reader only ever reads this accessor at **`page_id = 0`** ([`:60`](device/kernels/dataflow/indexed_fill_reader.cpp#L60)), so the page-size stride never multiplies a nonzero index → **inert** regardless of sharding. Matches the 3rd-arg triage (`2026-07-06_tensor_accessor_3rd_arg_triage.md`): `indexed_fill` → **Class 2 — Redundant**, additionally **(c)-type — `page_id ≡ 0`**. Port action: drop the arg. (See Heads-ups for the misleading in-code comment.) The two data accessors — `s0` @ [`:44`](device/kernels/dataflow/indexed_fill_reader.cpp#L44) and `s1` @ [`:45`](device/kernels/dataflow/indexed_fill_reader.cpp#L45) — and the writer's `dst` @ [`indexed_fill_writer_strided.cpp:34`](device/kernels/dataflow/indexed_fill_writer_strided.cpp#L34) pass **no** 3rd arg.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, classified by what the kernel does with the base pointer). All four buffers arrive today as `Buffer*` RTAs (correct-on-cache-hit `BufferBinding`s); the port replaces each with a typed `TensorParameter` / `TensorBinding`. Classification is config-dependent within the single factory:

  | Binding | Delivery today | Generic path | Native path | Shard-local path |
  |---|---|---|---|---|
  | `batch_ids` | `Buffer*` RTA (reader arg 0) | **Case 1** (accessor `batchAddr`) | Case 1 | Case 1 |
  | `input_a` | `Buffer*` RTA (reader arg 2) | **Case 1** (accessor `s0`, [`:44`](device/kernels/dataflow/indexed_fill_reader.cpp#L44)/[`:228`](device/kernels/dataflow/indexed_fill_reader.cpp#L228)) | **Case 1** (bulk read via `s0`, [`:183`](device/kernels/dataflow/indexed_fill_reader.cpp#L183)) | **Case 2** — raw base + offset, [`:130-138`](device/kernels/dataflow/indexed_fill_reader.cpp#L130-L138) |
  | `input_b` | `Buffer*` RTA (reader arg 3) | **Case 1** (accessor `s1`, [`:225`](device/kernels/dataflow/indexed_fill_reader.cpp#L225)) | **Case 1** (`s1`, [`:176`](device/kernels/dataflow/indexed_fill_reader.cpp#L176)) | **Case 1** for INTERLEAVED_B ([`:122`](device/kernels/dataflow/indexed_fill_reader.cpp#L122)); **Case 2** for SAME_SHARDED_B — raw base + offset, [`:105-113`](device/kernels/dataflow/indexed_fill_reader.cpp#L105-L113) |
  | `output` | `Buffer*` RTA (strided-writer arg 0) | **Case 1** (accessor `dst`, [`indexed_fill_writer_strided.cpp:34`](device/kernels/dataflow/indexed_fill_writer_strided.cpp#L34)/[`:45`](device/kernels/dataflow/indexed_fill_writer_strided.cpp#L45)) | **clean** — borrowed-memory DFB (CB aliased to `output.buffer()`); writer is wait/pop stub | **clean** — borrowed-memory DFB |

  - **Case 1** bindings → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the address-via-RTA and its `TensorAccessorArgs` CTAs both disappear.
  - **Case 2** bindings (`input_a` shard-local; `input_b` SAME_SHARDED_B) → bind as `TensorParameter`, pull the base via the `get_bank_base_address` bridge, **keep the existing raw offset arithmetic unchanged**.
  - **clean** (`output` in native/shard-local) → the CB is `set_globally_allocated_address(output.buffer())`; port via `DataflowBufferSpec::borrowed_from`. The framework's re-point on cache hit (noted in-code as `UpdateDynamicCircularBufferAddress`) is what `borrowed_from` expresses.
  - Op-level roll-up: **⚠ port work** (Case 1 + Case 2 bindings present).

- **TensorParameter relaxation:** none (sheet `= none`; no custom hash).
- **TensorAccessor 3rd arg:** drop the redundant page-size arg @ [`indexed_fill_reader.cpp:49`](device/kernels/dataflow/indexed_fill_reader.cpp#L49) (Class 2; no `dynamic_tensor_shape`).
- **CB endpoints:** self-loop the batch CB (`batch_cb_index=1`, all configs); the data CB (`cb_index=0`) is legal 1:1 (bind reader PRODUCER + writer CONSUMER), with `borrowed_from(output.buffer())` in the native / shard-local configs.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader CB. The data CB has exactly one producer (reader) and one consumer (writer); the batch CB has a single toucher. No semaphore-gated raw co-fill anywhere (the op uses no semaphores).
- **Misleading 3rd-arg comment:** the reader comment @ [`indexed_fill_reader.cpp:47-48`](device/kernels/dataflow/indexed_fill_reader.cpp#L47-L48) says the runtime page-size 3rd arg "overrides `TensorAccessorArgs::AlignedPageSize`, which may be stale on program cache hits." Do not let this deter the Class-2 drop: because the accessor is only read at `page_id = 0`, the stride is inert; and the staleness the comment worries about is exactly what a Metal 2.0 `TensorBinding` refresh removes (the accessor args are rebuilt from the bound base on each enqueue). Drop the arg.
- **Cross-op / shared kernels:** none. All three kernels are owned by this op and file-path-instantiated from its own directory; kernel `#include`s are all `api/*` (Device 2.0 API surface). No donor functions, no borrowed kernel files, no port-together set.
- **RTA varargs:** none. The reader and both writers read runtime args at fixed constexpr indices (reader `0..12`, generic writer `0..6`); no `arg_index++`-in-loop and no data-selected index. All args are nameable — ordinary named-arg port work.

## Team-only

- **Out-of-directory coupling & donor shape:** ✓ clean. Inventory: every kernel `#include` resolves to `tt_metal/hw/inc/api/*` (LLK/HAL/firmware, class 1 — no concern). No `ttnn/cpp/ttnn/operations/**` cross-op includes, no shared-pool includes. Borrowed kernel files: none (the factory instantiates only this op's own three kernel `.cpp` files). No summary table or per-call detail needed (all rolls ✓).
- **TTNN factory analysis (sheet-derived facts + evidence):**
  - Current concept: `descriptor` → target `MetalV2FactoryConcept` (no op-owned tensors).
  - Op-owned tensors: none (sheet blank; the `descriptor` concept cannot carry them, and the factory declares no `buffers` vector).
  - MeshWorkload: not a `WorkloadDescriptor` — no SPMD question.
  - Custom hash / custom `override_runtime_arguments` / pybind `create_descriptor` / other risky pybind: all absent (confirmed against code).
  - Smuggled pointer: `no` (Diego's correctness call); consistent with the audit finding that no `->address()` reaches an RTA — all buffers ride `Buffer*` `BufferBinding`s or the borrowed-memory CB.

## Misc anomalies  *(team-only, non-gating)*

- **Batch CB declared with input_a's data format though it stores `uint32` indices.** [`indexed_fill_program_factory.cpp:288-296`](device/indexed_fill_program_factory.cpp#L288-L296) sets the batch CB's `data_format = cb_data_format` (= `datatype_to_dataformat_converter(input_a.dtype())`, e.g. bfloat16), but the CB holds `b` `uint32` batch ids and its `page_size` is computed in bytes (`round_up_to_mul32(b * sizeof(uint32))`). The format field is inert here (the CB is filled/read by raw byte access at an explicit byte page size, never as tiles), so this is cosmetic — but a fresh reader may find the mismatch confusing. Ops team may wish to set an integer/`UInt32` format for clarity. Not porter work.

## Recipe notes  *(none)*

No friction with the audit recipe itself on this op. The subjects mapped cleanly; the two dated triage docs and the readiness sheet all agreed with the first-principles code read.
