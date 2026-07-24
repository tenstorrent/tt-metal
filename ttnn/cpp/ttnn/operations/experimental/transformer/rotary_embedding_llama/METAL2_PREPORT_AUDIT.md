# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama`

Single device operation in the directory:

- **`RotaryEmbeddingLlamaDeviceOperation`** (`device/rotary_embedding_llama_device_operation.{hpp,cpp}`)
  - `RotaryEmbeddingLlamaMultiCore` — `device/rotary_embedding_llama_multi_core_program_factory.cpp` (prefill, interleaved)
  - `RotaryEmbeddingLlamaMultiCorePrefillSharded` — `device/rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp` (prefill, cos/sin and/or trans_mat HEIGHT_SHARDED)
  - `RotaryEmbeddingLlamaMultiCoreSharded` — `device/rotary_embedding_llama_sharded_program_factory.cpp` (decode, fully HEIGHT_SHARDED)

Referenced kernels (all in-directory; none borrowed):

- `device/kernels/dataflow/reader_rotary_embedding_llama_interleaved_start_id.cpp` (MultiCore reader)
- `device/kernels/dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp` (MultiCore + PrefillSharded writer)
- `device/kernels/dataflow/reader_rotary_embedding_llama_prefill_sharded.cpp` (PrefillSharded reader)
- `device/kernels/compute/rotary_embedding_llama.cpp` (MultiCore + PrefillSharded compute)
- `device/kernels/compute/rotary_embedding_llama_sharded.cpp` (decode compute — the only kernel for the Sharded factory)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

**Recipe docs:** `776151aeca8 2026-06-24 docs(metal2): clarify SPSC face-(b) producer-as-consumer, aliased-vs-same-FIFO, no-portable-subset`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama` |
| **Overall** | GREEN |
| **DOps / Factories** | `RotaryEmbeddingLlamaDeviceOperation` → `RotaryEmbeddingLlamaMultiCore`, `RotaryEmbeddingLlamaMultiCorePrefillSharded`, `RotaryEmbeddingLlamaMultiCoreSharded` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (no out-of-directory `#include`s; no borrowed kernels) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (all CTAs fixed-count) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No (factories expose only `create_descriptor`; no `override_runtime_arguments`) |
| *Ops readiness* — Sync-free CBs (address-only) | None |

**Sync-free CBs** = CBs used purely as an address source. None found — every CB that a kernel touches is driven through the FIFO machinery (`reserve_back`/`push_back` + `wait_front`/`pop_front`). The decode output CB `c_16` is single-ended (producer only), which is a separate case (see DFB endpoint legality), not a sync-free CB.

## Result

**GREEN → brief issued.** All gates clear: the op is on the `ProgramDescriptor` API, every kernel it uses is Device 2.0 compliant, and no UNSUPPORTED feature is in use. The only LANDED feature requiring a non-obvious construct is the **borrowed-memory DFB** (the `.buffer = <buffer>` CBs in the decode and prefill-sharded factories) → `DataflowBufferSpec::borrowed_from`. Port work is the routine tensor-binding conversion plus deleting the (default-equivalent) custom `compute_program_hash`. One heads-up: the decode factory's output CB `c_16` is single-ended (producer-only) → self-loop workaround.

## Gate detail

- **ProgramDescriptor:** GREEN. All three factories define `static ProgramDescriptor <Factory>::create_descriptor(...)` and populate `desc.cbs` (`CBDescriptor`), `desc.kernels` (`KernelDescriptor`), `compile_time_args`, and `emplace_runtime_args`. The device-op uses the decentralized `ttnn::device_operation::launch<OperationType>` path (`device/rotary_embedding_llama_device_operation.cpp:265`). No imperative `host_api.hpp` builder calls (`CreateProgram`/`CreateKernel`/`CreateCircularBuffer`/`SetRuntimeArgs`) anywhere.

- **Device 2.0 (every kernel used):** GREEN. All five kernels use Device-2.0 idioms exclusively:
  - Data-movement: `Noc noc;` object with `noc.async_read(...)` / `noc.async_write(...)` / `noc.async_read_barrier()` / `noc.async_write_barrier()` (e.g. `reader_rotary_embedding_llama_interleaved_start_id.cpp:13,67,124`; `writer_..._interleaved_start_id.cpp:19,65,70`).
  - CB access via `CircularBuffer cb_*(cb_id);` wrapper objects with member-form `reserve_back` / `push_back` / `wait_front` / `pop_front` / `get_write_ptr()` / `get_read_ptr()` (e.g. `reader_...interleaved...:57-60,66`; `writer...:40-44,61`; both compute kernels).
  - Tensor access via `TensorAccessor(args, addr)` + `TensorAccessorArgs<N>()` (`reader_...interleaved...:36-39,47-55`; `writer...:34,38`; `reader_...prefill_sharded...:38-41`).
  - The only CB-index free function used is `get_tile_size(cb_id)` (e.g. `reader_...interleaved...:46`), which the recipe lists as **sanctioned** (Device 2.0 keeps it as a free function; not a holdover). No raw `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw CB-index management, no raw semaphore addresses. No isolated holdovers.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_*`/`.remote_index(`. |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer` set (`.buffer = src_buffer`/`cos_buffer`/`sin_buffer`/`trans_mat_buffer`/`dst_buffer`) — Sharded (decode) factory `rotary_embedding_llama_sharded_program_factory.cpp:87,99,111,125,171`; PrefillSharded fast-path branches `rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp:175,186,260`. Port uses `borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set anywhere; no `set_address_offset`; no 4-arg `UpdateDynamicCircularBufferAddress(...,offset)`. |
  | Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element (`{{CBFormatDescriptor{...}}}`); no CB carries 2+ format descriptors. |
  | GlobalSemaphore | N/A | No semaphores of any kind in this op. |
  | Non-zero semaphore initial value | N/A | No `CreateSemaphore` / `SemaphoreDescriptor` usage. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | All `TensorAccessorArgs(*buffer)` are the single-argument (static) form; no `ArgConfig::Runtime*` token. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed 4-tensor struct (`RotaryEmbeddingLlamaInputs`); no `std::vector<Tensor>`. Every kernel reads CTAs with constant indices `get_compile_time_arg_val(0..N)`; no runtime-varying CTA index. |

- **DFB endpoint legality (SPSC):** GREEN (no violations). Per-`(CB, factory)` census below. The decode-factory output CB `c_16` is single-ended `(1, 0)` → handled by the self-loop bridge (FYI-P, not a gate). All other CBs are `(1 producer, 1 consumer)` per node. No dead CBs, no hidden second writer, no multi-reader CB. There are **no semaphores and no split-reader / mcast paths** in this op, so the two SPSC faces (hidden raw co-fill; multi-reader borrowed CB) have no surface to appear on.

  Endpoint census (per node):

  | CB | Factory | Producer | Consumer | Verdict |
  |---|---|---|---|---|
  | `c_0` input | MultiCore | reader `push_back` | compute `wait_front`/`pop_front` | (1,1) legal |
  | `c_1`/`c_2` cos/sin | MultiCore | reader `push_back` | compute `wait_front`/`pop_front` | (1,1) legal |
  | `c_3` trans_mat | MultiCore | reader `push_back` | compute `wait_front`/`pop_front` | (1,1) legal |
  | `c_16` output | MultiCore | compute `push_back` | writer `wait_front`/`pop_front` | (1,1) legal |
  | `c_24/25/26` interm | MultiCore | compute `push_back` | compute `wait_front`/`pop_front` (same kernel) | (1,1) legal |
  | `c_27` zero | MultiCore | writer `push_back` | writer `wait_front`/`pop_front` (same kernel) | (1,1) legal |
  | `c_0/1/2/3/16/24/25/26/27` | PrefillSharded | same producer/consumer shape as MultiCore | | (1,1) legal |
  | `c_0/1/2/3` (borrowed) | Sharded (decode) | compute `reserve_back`/`push_back` | compute `wait_front`/`pop_front` (same kernel) | (1,1) legal |
  | `c_16` output (borrowed) | Sharded (decode) | compute `reserve_back`/`push_back` | **none** (no `wait_front`/`pop_front` on `out_cb`) | **(1,0) single-ended → self-loop** |
  | `c_24/25/26` interm | Sharded (decode) | compute `push_back` | compute `wait_front`/`pop_front` (same kernel) | (1,1) legal |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory — classification varies by factory; record the split):
  - **MultiCore (interleaved):** `input` / `cos` / `sin` / `trans_mat` / `output` — all **Case 1**. The factory smuggles the base address as a `Buffer*` runtime arg via `emplace_runtime_args` (`...multi_core_program_factory.cpp:337-338`), and the kernel feeds that address into `TensorAccessor(args, addr)` (`reader_...interleaved...:47-55`; `writer...:38`). Port: express each as a `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(ta::name)`; the RTA address and `TensorAccessorArgs` CTA plumbing disappear. All five are dataflow-kernel bindings (no compute-kernel raw-pointer case).
  - **PrefillSharded:** `input` and `output` — always **Case 1** (TensorAccessor in reader/writer). `cos` / `sin` / `trans_mat` — **per-config split**: **clean (borrowed-memory DFB)** on the sharded fast path (`.buffer = cos_buffer/sin_buffer/trans_mat_buffer`, kernel reads via CB FIFO with no TensorAccessor — `reader_...prefill_sharded...:60-67` for trans_mat, `:82-89` fast path for cos/sin), and **Case 1** on the reload path (`COS_SIN_SHARDED_RELOAD==1` / non-global trans_mat → `TensorAccessor(args, addr)`, `reader_...prefill_sharded...:69,87-88,143,146`). Port handles the clean branch via `borrowed_from` and the reload branch via `TensorParameter`.
  - **Sharded (decode):** `input` / `cos` / `sin` / `trans_mat` / `output` — all **clean (borrowed-memory DFB)**. Every CB is `.buffer = <buffer>` and the (compute-only) kernel reads/writes through CB FIFO ops, never via a TensorAccessor or RTA address. Port handles all via `borrowed_from`.
- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). See Custom program hash below.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Borrowed-memory DFB** — `CBDescriptor::buffer` set in the Sharded (decode) factory (`rotary_embedding_llama_sharded_program_factory.cpp:87,99,111,125,171`) and the PrefillSharded fast paths (`rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp:175,186,260`). Port via `DataflowBufferSpec::borrowed_from` naming the `TensorParameter` whose buffer backs each DFB.
- **Sync-free CBs (address-only):** none.
- **Single-ended CB (`(1,0)`):** decode-factory output CB `c_16` — the compute kernel `reserve_back`/`push_back`es into the borrowed output CB but nothing `wait_front`/`pop_front`s it (`rotary_embedding_llama_sharded.cpp:67,118`; CB declared at `..._sharded_program_factory.cpp:163-172` with `.buffer = dst_buffer`). A Metal 2.0 DFB requires ≥1 producer + ≥1 consumer, so the port fabricates the missing consumer via the sanctioned **self-loop workaround**. Does **not** gate.
- **Dead CBs (zero endpoints):** none.
- **Cross-op / shared kernels:** none — the op owns all five kernel files; no factory `#include`s or file-path-instantiates a kernel outside the directory. (`rotary_embedding_llama_fused_qk` and `rotary_embedding_indexed` have their *own* same-named kernels; they do not borrow this op's kernels.)
- **RTA varargs:** none — every kernel reads RTAs with a fixed, statically-known count via `argrt++` on consecutive `get_arg_val<uint32_t>` calls.
- **TTNN factory analysis (porter-relevant):** no pybind `create_descriptor`, no other migration-risky pybind, no custom `override_runtime_arguments`. (Custom `compute_program_hash` present → carried as PORT WORK above.)

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**. No kernel file in this op `#include`s a header outside its own directory (all includes are framework `api/...` headers). No program factory file-path-instantiates a kernel it does not own. No function-call escapes, no borrowed kernel files → no port-together coupling set. (Per-call detail omitted — all rolls ✓.)

- **Relaxation candidates** (mined from the custom hash before deletion): **none** — the custom `compute_program_hash` (`device/rotary_embedding_llama_device_operation.cpp:228-232`) is a thin wrapper around the default `tt::tt_metal::operation::hash_operation<RotaryEmbeddingLlamaDeviceOperation>(operation_attributes, tensor_args)`; it encodes no op-specific tensor-property dependence, so there is nothing to mine. **FALLIBLE — candidates to verify; default remains strict.**

- **TTNN factory analysis (six questions):**
  1. **Op-owned tensors? No.** The only `create_device_tensor` is for the declared output in `create_output_tensors` (`device/rotary_embedding_llama_device_operation.cpp:224-225`); no factory allocates an intermediate/scratch device tensor. Intermediate CBs (`c_24/25/26/27`) are circular buffers, not device tensors.
  2. **MeshWorkload concept needed? No.** Each factory returns a single `ProgramDescriptor` from `create_descriptor`; no `create_mesh_workload` / `cached_mesh_workload_t`. No cross-program or cross-device coordination. (Not even a Q1 op-owned-tensor artifact — there are no op-owned tensors.)
  3. **Pybind `create_descriptor`? No.** `rotary_embedding_llama_nanobind.cpp` binds only the user-facing op via `ttnn::bind_function<"rotary_embedding_llama", ...>` (line 18). No `nb::class_<...ProgramFactory>` and no `.def_static("create_descriptor", ...)`.
  4. **Other migration-risky pybind? None.** No `DeviceOperation`/factory/param `nb::class_<>`; no device-op methods (`compute_program_hash`, `create_output_tensors`, `compute_output_specs`, `select_program_factory`) exposed to Python.
  5. **Custom hash? Yes** — `device/rotary_embedding_llama_device_operation.cpp:228-232`. Treatment is in Custom program hash below (delete → default). Note it is *already* a default-equivalent wrapper, so deletion is behavior-preserving.
  6. **Custom override-runtime-args? No.** The factory `.hpp` files declare only `create_descriptor`; no `static void <Factory>::override_runtime_arguments(...)`.

### Custom program hash

**Recognition.** `RotaryEmbeddingLlamaDeviceOperation::compute_program_hash` is declared (`device/rotary_embedding_llama_device_operation.hpp:29`) and defined (`device/rotary_embedding_llama_device_operation.cpp:228-232`).

**Finding role: PORT WORK — delete it.** The body is `return tt::tt_metal::operation::hash_operation<RotaryEmbeddingLlamaDeviceOperation>(operation_attributes, tensor_args);` — i.e. it already reproduces the default reflection-based hash (no `TensorSpec` omission, no custom keying). The port deletes this member (and its declaration) and reverts to the default TTNN hash. No Metal 2.0 factory concept reads it; the default is correct-by-construction. Because the wrapper is already default-equivalent, deletion is behavior-preserving.

## Misc anomalies  *(team-only, non-gating)*

- **`output_single_tile_size` used for cos/sin interm page sizes in decode factory.** In `rotary_embedding_llama_sharded_program_factory.cpp:142,153`, the `cos_interm_cb` (`c_25`) and `sin_interm_cb` (`c_26`) `total_size` is computed with `num_interm_tiles * input_single_tile_size` but the `page_size` is set from `cos_single_tile_size`/`sin_single_tile_size`. Since all inputs are asserted bfloat16 (validate, line 71-74) these tile sizes are identical, so it is harmless today; noted only because the size/page-size operands are mismatched expressions. Routes to op owner; not port work.
- **`compute_kernel_config` packer_l1_acc / dst_full_sync_en unused.** The compute factories destructure `packer_l1_acc` and `dst_full_sync_en` from `get_compute_kernel_config_args` but only `math_fidelity` and `fp32_dest_acc_en` are passed to `ComputeConfigDescriptor`. Pre-existing; not port-relevant.

## Recipe notes

- **`Buffer*`-binding form that the kernel then feeds into a `TensorAccessor` — Case 1 or Case 2?** The factories use the `Buffer*`-binding shape (`emplace_runtime_args(core, {src_buffer, ...})`), which the Detection§ "`Buffer*`-binding form" bullet labels **Case 2** ("the kernel consumes a raw `uint32_t` base, so it is Case 2"). But the *kernels here* feed that base straight into `TensorAccessor(args, addr)` — the textbook **Case 1** shape. I classified by the dominant rule "classify by what the kernel does with the base pointer" → Case 1, treating the `Buffer*`-binding bullet's "Case 2" as describing the *common* case for that host shape rather than a hard override. A one-line clarification in the `Buffer*`-binding bullet ("…unless the kernel constructs a `TensorAccessor` from the base, in which case it is Case 1 per the kernel-side rule") would remove the apparent contradiction for the next auditor.

---

## ✅/⚠️ Post-port update (2026-06-25)

**Ported successfully (PR #48172):** the decode (`RotaryEmbeddingLlamaMultiCoreSharded`) and prefill-interleaved (`RotaryEmbeddingLlamaMultiCore`) factories — 423 passed / 5 skipped / 0 failed across the full suite (incl. the cache-hit borrowed-DFB path). The GREEN grade held for these two.

**`RotaryEmbeddingLlamaMultiCorePrefillSharded` — framework-blocked (audit miss):** when the cos/sin/trans_mat **shard grid is smaller than the device grid** (`partial_cos_sin`/`partial_tm`), the legacy factory defines the *same* CB index (`c_1`/`c_2`/`c_3`) **twice** — borrowed (`.buffer=`) on the shard-grid cores and plain scratch on the remaining cores. A Metal 2.0 `DataflowBufferSpec` is **program-wide** (one `borrowed_from` per id), so a CB that is borrowed on some nodes and scratch on others has no expression today. Left on legacy `create_descriptor` (decoupled via forked `_metal2` kernels; regression-clean). Wait-for-feature: per-node borrowed-DFB binding. See [[metal2-port-portability-predictor]].

**Reusable constraint discovered during the MultiCore port:** Gen1 **data-movement kernels cannot self-loop a DFB** (only compute kernels can). The legacy writer's producer-only zero-fill CB was fixed by hoisting it to a reader (PRODUCER) → writer (CONSUMER) cross-kernel DFB (output-identical).
