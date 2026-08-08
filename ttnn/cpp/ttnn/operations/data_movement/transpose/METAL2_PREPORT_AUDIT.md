# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/transpose`

**Identifying section:**

- **`TransposeDeviceOperation`** (`ttnn::prim::TransposeDeviceOperation`)
  - `TransposeWHProgramFactory` (`transpose_wh_program_factory.cpp`) — two sub-paths: tiled and RM
  - `TransposeWHShardedProgramFactory` (`transpose_wh_sharded_program_factory.cpp`) — tiled sharded
  - `TransposeWHShardedRMProgramFactory` (`transpose_wh_sharded_rm_program_factory.cpp`) — RM sharded
  - `TransposeHCTiledInterleavedProgramFactory` (`transpose_hc_tiled_interleaved_program_factory.cpp`) — active HC tiled path
  - `TransposeHCTiledProgramFactory` (`transpose_hc_tiled_program_factory.cpp`) — **DEAD CODE: in variant type but never returned from `select_program_factory`**
  - `TransposeHCRMProgramFactory` (`transpose_hc_rm_program_factory.cpp`)
  - `TransposeHCShardedProgramFactory` (`transpose_hc_sharded_program_factory.cpp`)
  - `TransposeCNProgramFactory` (`transpose_cn_program_factory.cpp`)

**Note on dead factory:** `TransposeHCTiledProgramFactory` appears in `program_factory_t` (the variant type in `transpose_device_operation.hpp`) but is never selected by `select_program_factory` (`transpose_device_operation.cpp:135–158`). Its kernels (`reader_unary_transpose_hc_interleaved_partitioned.cpp`) are therefore unreferenced and out of audit scope. Noted as a dead-code anomaly in §Misc anomalies.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/transpose` |
| **Overall** | GREEN |
| **DOps / Factories** | `TransposeDeviceOperation` → WH·WHSharded·WHShardedRM·HCTiledInterleaved·HCTiled(dead)·HCRM·HCSharded·CN |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

---

## Result

**GREEN → brief issued.** All gates cleared. The port can proceed once the user gives go-ahead.

Port work to prepare:

- **Tensor bindings:** Four active factories use `Buffer*`-binding or `->address()`-as-RTA RTAs (all Case 1 re-express). Two factories use borrowed-memory DFB (clean). Two factories use borrowed-memory DFB (clean). See §Port-work summary.
- **Dynamic TensorAccessor:** Six `ArgConfig::RuntimeTensorShape` sites across four active factories — the UNSAFE relaxation opt-in is the Metal 2.0 translation; see §Heads-ups.
- **Borrowed shared kernels:** `TransposeWHProgramFactory` borrows `writer_unary_interleaved_start_id.cpp` (eltwise/unary) and `TransposeWHShardedProgramFactory` borrows `reader_unary_sharded.cpp` (eltwise/unary) and `writer_unary_sharded.cpp` (data_movement/sharded). These form port-together sets — see §Heads-ups.

---

## Gate detail

### ProgramDescriptor

GREEN — all eight factories define a `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` signature, populate a `ProgramDescriptor desc`, and use `KernelDescriptor`, `CBDescriptor`, `CBFormatDescriptor`. No imperative `host_api.hpp` builder calls (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`). The factories include `<tt-metalium/program_descriptors.hpp>`, confirming PD-API status.

### Device 2.0 (every kernel used)

GREEN — every kernel the op instantiates is Device 2.0 compliant. Specifically:

- All dataflow kernels include `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h` and use `Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem` wrapper objects.
- No legacy address-generator types (`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`) appear anywhere in the op's kernel files.
- `get_read_ptr()`, `get_write_ptr()`, `get_tile_size()` are called exclusively in member form on `CircularBuffer` objects (not the free-function-with-cb-id form).
- `get_local_cb_interface(cb_id_out)` in `writer_unary_interleaved_start_id.cpp` (line 19) is sanctioned in Device 2.0 (no member-form replacement; explicitly kept as a free function in the D2.0 migration guide).
- `tt::data_movement::common::noc_async_read_sharded` / `noc_async_write_sharded` in `common/kernels/common.hpp` internally call `noc_async_read` / `noc_async_write`, but only as an implementation detail of a `TensorAccessor`-typed helper — the kernels themselves never call the raw functions directly. The helper is dispatched through Device 2.0 wrappers.
- Compute kernels (`transpose_wh.cpp`, `transpose_wh_rm.cpp`, `transpose_wh_sharded.cpp`) use only `CircularBuffer` wrapper objects and LLK compute APIs; `tilize_helpers.hpp` (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`) uses only CB wrappers.
- Shared/donor kernels (`reader_unary_sharded.cpp`, `writer_unary_sharded.cpp`, `writer_unary_interleaved_start_id.cpp`) are all Device 2.0 compliant.

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, or `.global_circular_buffer` field in any factory |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `.buffer` field set on CBDescriptors in WHSharded, WHShardedRM, HCSharded factories — port uses `borrowed_from` |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set to non-zero anywhere |
| Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element; no multi-index aliasing |
| GlobalSemaphore | N/A | No semaphores of any kind in this op |
| Non-zero semaphore initial value | N/A | No semaphores in this op |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | GREEN | `ArgConfig::RuntimeTensorShape` used in four factories (6 sites) — see §Heads-ups for porter warning |
| `UpdateCircularBuffer*` | N/A | No calls to `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = TransposeInputs{Tensor input}` — fixed single-input op; no variable-count CTA loops in kernels |

**Dynamic TensorAccessor detail** — fires as LANDED / FYI-P heads-up:

Sites:
- `transpose_wh_program_factory.cpp:229` — `TensorAccessorArgs(*src0_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (reader)
- `transpose_wh_program_factory.cpp:245` — `TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (writer)
- `transpose_hc_tiled_interleaved_program_factory.cpp:192` — `TensorAccessorArgs(*src_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (reader)
- `transpose_hc_tiled_interleaved_program_factory.cpp:219` — `TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (writer)
- `transpose_hc_rm_program_factory.cpp:154` — `TensorAccessorArgs(*src0_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (reader)
- `transpose_hc_rm_program_factory.cpp:161` — `TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (writer)
- `transpose_cn_program_factory.cpp:72` — `TensorAccessorArgs(*src0_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (reader)
- `transpose_cn_program_factory.cpp:78` — `TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)` (writer)

Metal 2.0 supports the runtime-shape capability via `TensorParameterAdvancedOptions::dynamic_tensor_shape` (full relaxation) or `match_padded_shape_only` (padded-shape-only relaxation), but both options are marked **UNSAFE** in the framework header and adopting them has structural implications for the factory's interaction with per-dispatch caching. The default remains strict; applying a relaxation is an explicit user-OK decision (see `port_op_to_metal2_ttnn_factory.md`), not an automatic port step.

Note: `TransposeHCTiledProgramFactory` (dead code) also uses `TensorAccessorArgs` but without `RuntimeTensorShape`, so no additional FYI-P sites from it.

---

## Port-work summary *(mirrors the brief)*

**Tensor bindings** (per binding, per active factory):

### TransposeWHProgramFactory (tiled sub-path)

- `input` (reader) — **Case 1** (re-express). Factory at `transpose_wh_program_factory.cpp:63`: `{input_tensor.buffer(), ...}` (`Buffer*` form). Kernel `reader_unary_transpose_wh_interleaved_start_id.cpp:12`: `src_addr = get_arg_val<uint32_t>(0)`; builds `TensorAccessor(src_args, src_addr)`. Port: bind as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::input)`, RTA slot 0 drops.
- `output` (writer) — **Case 1** (re-express). Factory at `transpose_wh_program_factory.cpp:74`: `{output_tensor.buffer(), ...}` (`Buffer*` form). Writer kernel `writer_unary_interleaved_start_id.cpp:11`: `dst_addr = get_arg_val<uint32_t>(0)`. Port: re-express via `TensorParameter`.

### TransposeWHProgramFactory (RM sub-path)

- `input` (reader) — **Case 1**. Factory `transpose_wh_program_factory.cpp:112`: `{input_tensor.buffer(), ...}` (`Buffer*` form). Kernel `reader_unary_transpose_wh_interleaved_start_id_rm.cpp:13`: `src_addr = get_arg_val<uint32_t>(0)`. Port: re-express via `TensorParameter`.
- `output` (writer) — **Case 1**. Factory `transpose_wh_program_factory.cpp:116`: `{output_tensor.buffer(), ...}` (`Buffer*` form). Kernel `writer_unary_transpose_wh_interleaved_start_id_rm.cpp:12`: `dst_addr = get_arg_val<uint32_t>(0)`. Port: re-express via `TensorParameter`.

### TransposeWHShardedProgramFactory

- `input` — **clean** (borrowed-memory DFB). `transpose_wh_sharded_program_factory.cpp:63`: `.buffer = input_tensor.buffer()` on `CBDescriptor`. Port uses `DataflowBufferSpec::borrowed_from` naming the input `TensorParameter`.
- `output` — **clean** (borrowed-memory DFB). `transpose_wh_sharded_program_factory.cpp:76`: `.buffer = output_tensor.buffer()`. Port uses `DataflowBufferSpec::borrowed_from`.

### TransposeWHShardedRMProgramFactory

- `input` — **clean** (borrowed-memory DFB). `transpose_wh_sharded_rm_program_factory.cpp:98`: `.buffer = input_tensor.buffer()`. Port uses `DataflowBufferSpec::borrowed_from`.
- `output` — **clean** (borrowed-memory DFB). `transpose_wh_sharded_rm_program_factory.cpp:112`: `.buffer = output_tensor.buffer()`. Port uses `DataflowBufferSpec::borrowed_from`.

### TransposeHCTiledInterleavedProgramFactory

- `input` (reader) — **Case 1** (re-express). Factory `transpose_hc_tiled_interleaved_program_factory.cpp:88`: `{input_buffer->address(), ...}` (**`->address()` form — correctness hazard on cache hits**). Kernel `reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp:13`: `src_addr = get_arg_val<uint32_t>(0)`. Port: route through `TensorParameter` / `TensorBinding`; drop raw-address RTA. Note: `TensorAccessorArgs(*src_buffer, RuntimeTensorShape)` already feeds CTA/CRTA for the accessor config — the raw-address RTA at slot 0 is the redundant and hazardous part.
- `output` (writer) — **Case 1**. Factory `transpose_hc_tiled_interleaved_program_factory.cpp:91`: `{output_buffer->address(), ...}` (**`->address()` form — correctness hazard**). Kernel `writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp:14`: `dst_addr = get_arg_val<uint32_t>(0)`. Port: re-express via `TensorParameter`.

### TransposeHCRMProgramFactory

- `input` (reader) — **Case 1**. Factory `transpose_hc_rm_program_factory.cpp:67`: `{input_buffer, ...}` (`Buffer*` form). Kernel `reader_unary_transpose_hc_interleaved_partitioned_rm.cpp:13`: `src_addr = get_arg_val<uint32_t>(0)`. Port: re-express.
- `output` (writer) — **Case 1**. Factory `transpose_hc_rm_program_factory.cpp:71`: `{output_buffer, ...}` (`Buffer*` form). Kernel `writer_unary_transpose_hc_interleaved_start_id_rm.cpp:12`: `dst_addr = get_arg_val<uint32_t>(0)`. Port: re-express.

### TransposeHCShardedProgramFactory

- `input` — **clean** (borrowed-memory DFB). `transpose_hc_sharded_program_factory.cpp:326`: `.buffer = input_tensor.buffer()`. Port uses `DataflowBufferSpec::borrowed_from`.
- `output` — **clean** (borrowed-memory DFB). `transpose_hc_sharded_program_factory.cpp:341`: `.buffer = output_tensor.buffer()`. Port uses `DataflowBufferSpec::borrowed_from`.

### TransposeCNProgramFactory

- `input` (reader) — **Case 1**. Factory `transpose_cn_program_factory.cpp:134–135`: `{src0_buffer, N, C, HtWt, ...}` (`Buffer*` form). Kernel `reader_unary_transpose_cn_interleaved_start_id.cpp:13`: `src_addr = get_arg_val<uint32_t>(0)`. Port: re-express.
- `output` (writer) — **Case 1**. Factory `transpose_cn_program_factory.cpp:136`: `{dst_buffer, ...}` (`Buffer*` form). Kernel `writer_unary_transpose_cn_interleaved_start_id.cpp:11`: `dst_addr = get_arg_val<uint32_t>(0)`. Port: re-express.

**Custom hash:** none — no `compute_program_hash` override anywhere in the op.

---

## Heads-ups *(mirrors the brief)*

### Notable LANDED constructs

- **Borrowed-memory DFB (Dynamic CB):** Three factories use `.buffer` on `CBDescriptor` — `TransposeWHShardedProgramFactory` (`transpose_wh_sharded_program_factory.cpp:63, 76`), `TransposeWHShardedRMProgramFactory` (`transpose_wh_sharded_rm_program_factory.cpp:98, 112`), `TransposeHCShardedProgramFactory` (`transpose_hc_sharded_program_factory.cpp:326, 341`). Port uses `DataflowBufferSpec::borrowed_from` naming the relevant `TensorParameter`.

- **Dynamic TensorAccessor** (`ArgConfig::RuntimeTensorShape`): Eight sites across four active factories (listed in §Feature compatibility above). Metal 2.0 translation is `TensorParameterAdvancedOptions::dynamic_tensor_shape = true`, but the option is marked **UNSAFE** in the framework header. Adopting it has structural implications for per-dispatch caching. The default remains strict; the porter applies the relaxation only on explicit user-OK (see `port_op_to_metal2_ttnn_factory.md`).

### Cross-op / shared kernels

Three borrowed kernel files from external families — all Device 2.0 compliant:

1. **`writer_unary_interleaved_start_id.cpp`** — borrowed by `TransposeWHProgramFactory` (tiled sub-path). Source: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/`. Broadly shared across eltwise/unary and other families. A shared kernel's Metal 2.0 rewrite must land as a single change touching all co-borrowers. The porter must coordinate with eltwise/unary to rewrite this kernel once, with all borrowers switching in the same PR. Function signature shape: the kernel takes `dst_addr` as RTA[0] — Case 1 (`TensorAccessor(dst_args, dst_addr)`). Constexpr cast from `TensorAccessorArgs` handles this cleanly in Metal 2.0.

2. **`reader_unary_sharded.cpp`** — borrowed by `TransposeWHShardedProgramFactory`. Source: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/`. No tensor address RTAs (sharded, all CB); port-together coupling exists but is trivial for this kernel (no TensorAccessor at all — it's a pure sharded CB reader).

3. **`writer_unary_sharded.cpp`** — borrowed by `TransposeWHShardedProgramFactory`. Source: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/`. Same: no tensor address RTAs. Port-together coupling is trivial.

### RTA varargs

None — no `num_runtime_varargs` or counted-loop RTA retrieval anywhere in the op's kernels.

### TTNN factory analysis (porter-relevant)

- **Pybind `create_descriptor`:** None — `transpose_nanobind.cpp` binds only the `ttnn::transpose` function (via `bind_function<"transpose">`). No ProgramFactory class or `create_descriptor` method is exposed.
- **Other risky pybind:** None.
- **Custom `override_runtime_arguments`:** None across all factories.

---

## Team-only

### TensorAccessor convertibility (per Case-1 binding)

All twelve active Case-1 bindings are straightforward re-express (standard page-by-page or stick-by-stick iteration via `TensorAccessor`). No binding is genuinely exotic; all are Case 1.

Note on dead factory `TransposeHCTiledProgramFactory` (never selected by `select_program_factory`): its reader kernel `reader_unary_transpose_hc_interleaved_partitioned.cpp` performs exotic sub-tile face-line reads with address arithmetic at `batch_itile` granularity (`s0.get_noc_addr(batch_itile, rem)` at line 111, using `TADDR_FLOAT32`/`TADDR_BFLOAT16` helpers for sub-tile offsets). If this factory is ever reactivated, the reader binding would require per-user judgment on Case 1 vs. Case 2 (the pattern navigates tile faces at sub-tile byte offsets — potentially genuinely exotic). Since the factory is dead code today, no port action is needed now, but the anomaly is flagged here for the team record.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean — all in-scope donors are Device 2.0 compliant; coupling is manageable.

**Summary table** (active factory kernel, donor file, status):

| Op kernel | Donor file | Donor class | Shape status |
|---|---|---|---|
| `TransposeWHProgramFactory` (tiled) | `writer_unary_interleaved_start_id.cpp` | cross-family (eltwise/unary) | ✓ Case 1: RTA[0] = `dst_addr`; `TensorAccessor(dst_args, dst_addr)` — standard Metal 2.0 re-express |
| `TransposeWHShardedProgramFactory` | `reader_unary_sharded.cpp` | cross-family (eltwise/unary) | ✓ no tensor address: sharded CB reader, trivial port |
| `TransposeWHShardedProgramFactory` | `writer_unary_sharded.cpp` | cross-family (data_movement/sharded) | ✓ no tensor address: sharded CB writer, trivial port |

**Borrowed kernel files (file-path instantiation):**

Active factories only:
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — borrowed by `TransposeWHProgramFactory` (tiled path) and by the dead `TransposeHCTiledProgramFactory`. Broadly shared (eltwise/unary and many other families). Port-together set includes all ops instantiating this kernel.
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — borrowed by `TransposeWHShardedProgramFactory`. Shared across sharded ops.
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — borrowed by `TransposeWHShardedProgramFactory`. Shared.

All other kernel files are owned by this op directory.

**Per-call detail for `writer_unary_interleaved_start_id.cpp`:**

Functions called:
- `kernel_main()` — takes `dst_addr` (RTA[0], uint32), `num_pages` (RTA[1]), `start_id` (RTA[2]); uses `TensorAccessorArgs<1>()` (CTA-encoded accessor config); constructs `TensorAccessor(dst_args, dst_addr)`. Shape: receives address as `uint32_t RTA`. Status: ✓ Case 1 — re-express via `TensorParameter` / `TensorBinding` in Metal 2.0. The `uint32_t cb_id_out` compile-time arg → `dfb::name`'s constexpr cast handles it.

### Relaxation candidates

None — no custom `compute_program_hash` to mine.

### TTNN factory analysis (six questions)

1. **Op-owned tensors?** No. `create_output_tensors` (`transpose_device_operation.cpp:223`) calls `create_device_tensor(...)` for the output tensor only — that is the standard output allocation, not an op-owned intermediate. No factory allocates additional device tensors.

2. **MeshWorkload needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` anywhere. Q1 answer was No (no op-owned tensors), so there is no MeshWorkload-path artifact either.

3. **Pybind `create_descriptor`?** No. `transpose_nanobind.cpp` exposes only `ttnn::transpose` via `bind_function<"transpose">`. No `nb::class_<...ProgramFactory>` or `create_descriptor` binding.

4. **Other migration-risky pybind?** No. The nanobind file does not expose any `DeviceOperation` methods, factory-parameter structs, or internal factory hooks.

5. **Custom hash?** No. No `compute_program_hash` override in `transpose_device_operation.cpp` or any factory file.

6. **Custom override-RTA?** No. No `static void ::override_runtime_arguments(...)` definition in any factory.

---

## Misc anomalies *(team-only, non-gating)*

- **Dead factory `TransposeHCTiledProgramFactory`** (`transpose_hc_tiled_program_factory.cpp`): listed in `program_factory_t` variant (`transpose_device_operation.hpp:34`) but never returned by `select_program_factory`. The `MULTI_CORE_HC` tiled path always returns `TransposeHCTiledInterleavedProgramFactory` instead. Recommend either removing `TransposeHCTiledProgramFactory` from the variant and deleting the factory file, or documenting why it is retained. Owner: `transpose` op maintainer. Neither the port nor the Device 2.0 team needs to act on this.

- **`reader_unary_transpose_wh_interleaved.cpp`** (in `device/kernels/dataflow/`): this kernel file exists in the directory but is not instantiated by any active factory (verified by grep). It is dead code. Noted for the op owner; no audit action.

- **`TransposeHCTiledProgramFactory` double-binding pattern** (moot while dead): in `transpose_hc_tiled_program_factory.cpp:163–171`, `TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args)` feeds the accessor config via CTA only (no `RuntimeTensorShape`), while `input_buffer->address()` is ALSO passed as `reader_compile_time_args` via RTA (line 68). If this factory is ever reactivated, the double-binding should be untangled before porting.

---

## Questions for the user

*(none — all audit findings were deterministic)*

---

## Recipe notes

*(no recipe friction encountered during this audit)*
