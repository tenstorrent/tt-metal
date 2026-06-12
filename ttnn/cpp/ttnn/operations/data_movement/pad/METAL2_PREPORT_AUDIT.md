# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/pad`

Single device-operation in this directory: **`PadDeviceOperation`** (`ttnn::prim` namespace), with seven program factories:

- **`PadDeviceOperation`**
  - `PadRmReaderWriterMultiCoreProgramFactory` (`pad_rm_reader_writer_multi_core_program_factory.cpp`) — `create_workload_descriptor`
  - `PadRmReaderWriterMultiCoreDefaultProgramFactory` (`pad_rm_reader_writer_multi_core_default_program_factory.cpp`) — `create_descriptor`
  - `PadRmReaderWriterProgramFactory` (`pad_rm_reader_writer_program_factory.cpp`) — `create_workload_descriptor`
  - `PadRmShardedHeightOnlyProgramFactory` (`pad_rm_sharded_height_only_program_factory.cpp`) — `create_descriptor`
  - `PadRmShardedWidthOnlyProgramFactory` (`pad_rm_sharded_width_only_program_factory.cpp`) — `create_descriptor`
  - `PadTileMulticoreProgramFactory` (`pad_tile_multicore_program_factory.cpp`) — `create_descriptor`
  - `PadTileCoreProgramFactory` (`pad_tile_program_factory.cpp`) — `create_descriptor`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/pad` |
| **Overall** | GREEN |
| **DOps / Factories** | `PadDeviceOperation` → `PadRmReaderWriterMultiCoreProgramFactory`, `PadRmReaderWriterMultiCoreDefaultProgramFactory`, `PadRmReaderWriterProgramFactory`, `PadRmShardedHeightOnlyProgramFactory`, `PadRmShardedWidthOnlyProgramFactory`, `PadTileMulticoreProgramFactory`, `PadTileCoreProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN (no UNSUPPORTED features in use) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | Yes: `PadRmReaderWriterMultiCoreProgramFactory::create_workload_descriptor` (`pad_rm_reader_writer_multi_core_program_factory.cpp:416`) and `PadRmReaderWriterProgramFactory::create_workload_descriptor` (`pad_rm_reader_writer_program_factory.cpp:197`) each allocate a pad-value const tensor via `build_pad_value_const_tensor_*()`, park it on `WorkloadDescriptor::buffers` |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned-tensor artifact only): the two `create_workload_descriptor` factories are on the MeshWorkload path solely because they need to keep the pad-value const tensor alive across cache hits via `WorkloadDescriptor::buffers`. Not a genuine multi-program or cross-device coordination need. |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: (a) `PadRmShardedHeightOnlyProgramFactory`: `cb_in0` (c_0, `.buffer = src_buffer`, `pad_rm_sharded_height_only_program_factory.cpp:289`) — no FIFO pair; `cb_out0` (c_16, `.buffer = dst_buffer`, line 304) — no FIFO pair. (b) `PadRmShardedWidthOnlyProgramFactory`: `cb_input` (c_0, `.buffer = input_buffer`, `pad_rm_sharded_width_only_program_factory.cpp:75`) — no FIFO pair. (workaround: sanctioned fake-CB workaround in porting recipe) |

**Fake CBs** = CBs used purely as an address source. **Litmus: does the CB have a producer *and* a consumer?** (Same core may be both.) No producer–consumer pair → fake: a Metal 2.0 DFB needs ≥1 of each, so a fake CB can't be expressed as a DFB — the port resolves it with the sanctioned fake-CB workaround (see the porting recipe), so it's an **FYI-P heads-up, not a gate**. Granularity is the **(CB, endpoint) edge** — the same CB can be a real LLK operand on one binding and address-only on another; record each address-only edge.

## Result

**GREEN → brief issued.** All gates clear. Port work consists of re-expressing `Buffer*`-form tensor bindings via `TensorParameter`/`TensorBinding` (Case 1) across all factories, and applying the fake-CB workaround in three sharded-factory CBs. Two factories are on the MeshWorkload path as a plumbing artifact of their op-owned const tensor — that path is selected downstream by the factory concept selection doc.

## Gate detail

- **ProgramDescriptor:** GREEN. Five factories expose `create_descriptor` (returning `ProgramDescriptor`). Two factories (`PadRmReaderWriterMultiCoreProgramFactory`, `PadRmReaderWriterProgramFactory`) expose `create_workload_descriptor` (returning `WorkloadDescriptor`). `WorkloadDescriptorConcept` is a subset of `ProgramDescriptorFactoryConcept` per `ttnn/api/ttnn/operation_concepts.hpp:69-72`; both populate `ProgramDescriptor` internally. ProgramDescriptor prerequisite is met across all factories.

- **Device 2.0 (every kernel used):** GREEN. All kernels use Device 2.0 idioms — `Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint` from `api/dataflow/` and `api/tensor/` headers. No `InterleavedAddrGen`, `ShardedAddrGen`, raw `noc_async_read`, or manual CB index management detected. Two sanctioned free functions are present and are not holdovers:
  - `get_local_cb_interface(cb_id_in0)` at `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:20` — sanctioned per audit recipe (no member form exists).
  - `get_tile_size(cb_id_out0)` at `pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp:28` — sanctioned per audit recipe (no member form exists).

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer` or `global_circular_buffer` field on any `CBDescriptor` |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `PadRmShardedHeightOnlyProgramFactory` (`pad_rm_sharded_height_only_program_factory.cpp:289,304`) and `PadRmShardedWidthOnlyProgramFactory` (`pad_rm_sharded_width_only_program_factory.cpp:75,91`) set `CBDescriptor::buffer` to non-null buffer pointers (LANDED, port uses `borrowed_from`) |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set in any `CBDescriptor` across all factories |
  | Aliased Circular Buffers | N/A | All `CBDescriptor::format_descriptors` are single-element throughout |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type or `CreateGlobalSemaphore` calls anywhere |
  | Non-zero semaphore initial value | N/A | No `SemaphoreDescriptor` or `CreateSemaphore` calls anywhere in the op |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` enumerators in any factory |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` calls |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is `PadInputs` — fixed-count (`Tensor input` + `std::optional<Tensor> preallocated_output`). No `std::vector<Tensor>`. No kernel-side `get_compile_time_arg_val(i)` in a loop with runtime-varying `i`. |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input` (src tensor) — **Case 1** across all interleaved and tile factories. All factories push `src0_buffer` / `input_buffer` as `Buffer*` into `RTArgList`; kernels extract via `get_arg_val<uint32_t>(0)` and build `TensorAccessor(src_args, src_addr)`. Re-express via `TensorParameter`/`TensorBinding`.
  - `output` (dst tensor) — **Case 1** across all interleaved and tile factories. Same pattern: `dst_buffer` / `output_buffer` pushed as `Buffer*`; kernels build `TensorAccessor(dst_args, dst_addr)`. Re-express via `TensorParameter`/`TensorBinding`.
  - `pad_value_const_tensor` (op-owned const tensor, two WorkloadDescriptor factories only) — **Case 1**. `pad_value_const_buffer` pushed as `Buffer*` at slot 13; kernel extracts and builds `TensorAccessor(pad_tensor_args, pad_value_const_buffer_addr)`. Port re-expresses via `TensorParameter`/`TensorBinding` (also requires resolving the op-owned-tensor factory concept — see TTNN factory analysis).
  - `input shard` / `output shard` (sharded factories) — **clean** (borrowed-memory DFBs via `CBDescriptor::buffer`). Port uses `DataflowBufferSpec::borrowed_from`. Three CBs are fake (see Fake CBs section).
- **Custom hash:** None.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Dynamic CircularBuffer (borrowed memory)** — `PadRmShardedHeightOnlyProgramFactory` uses two borrowed-memory CBs: `cb_src0` at `pad_rm_sharded_height_only_program_factory.cpp:289` (`.buffer = src_buffer`) and `cb_output` at line 304 (`.buffer = dst_buffer`). `PadRmShardedWidthOnlyProgramFactory` uses two: `cb_input` at `pad_rm_sharded_width_only_program_factory.cpp:75` (`.buffer = input_buffer`) and `cb_output` at line 91 (`.buffer = output_buffer`). Port uses `DataflowBufferSpec::borrowed_from` for each.

- **Fake CBs (address-only):**
  - `(cb_in0/c_0, reader kernel)` in `PadRmShardedHeightOnlyProgramFactory`: kernel reads `l1_read_addr = cb_in0_exp.get_write_ptr()` as a raw L1 address — no `reserve_back`, `push_back`, `wait_front`, or `pop_front`. No producer-consumer pair. Port: sanctioned fake-CB workaround.
  - `(cb_out0/c_16, reader+writer kernels)` in `PadRmShardedHeightOnlyProgramFactory`: the reader calls `reserve_back(num_sticks_padded)` / `push_back(num_sticks_padded)` as a single bulk reservation covering the entire shard; the writer calls `get_write_ptr()` and accesses it as a raw L1 buffer, also calling `reserve_back(padded_shard_height)` / `push_back(1)` in a loop. These do not form a proper producer-consumer FIFO (both reader and writer act as producers); the CB is functionally address-only. Port: sanctioned fake-CB workaround.
  - `(cb_input/c_0, reader kernel)` in `PadRmShardedWidthOnlyProgramFactory`: kernel reads `input_shard_base_addr = cb_input_shard.get_write_ptr()` as a raw L1 address — no FIFO operations. Port: sanctioned fake-CB workaround.
  - `(cb_output/c_16, writer+reader kernels)` in `PadRmShardedWidthOnlyProgramFactory`: the writer (`writer_pad_dims_rm_sharded_stickwise.cpp`) calls `reserve_back(padded_shard_height)` and `push_back(1)` per stick; the reader (`reader_pad_dims_rm_sharded_stickwise.cpp`) calls `wait_front(1)` and `pop_front(1)` per stick. **Real DFB** — not fake. Port uses `DataflowBufferSpec::borrowed_from` normally.

- **Cross-op / shared kernels:**
  - `PadTileCoreProgramFactory` instantiates `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` — owned by `eltwise/unary`. This kernel is Device 2.0 compliant. Because it is shared, any Metal 2.0 rewrite of this kernel (CB→DFB, named-token bindings) must be coordinated with all ops that instantiate it — this forms a **port-together set** including at least `eltwise/unary` and this `data_movement/pad` factory. Surface to planners before porting.

- **RTA varargs:** None. All RT-arg counts are statically known at factory build time. The `start_dim_offset` pointer-to-args pattern in `reader_pad_dims_rm_interleaved_v2.cpp` and `reader_pad_dims_rm_sharded.cpp` is a fixed-slot pointer read, not a counted vararg loop.

- **TTNN factory analysis (porter-relevant):**
  - **Pybind `create_descriptor`:** None. `pad_nanobind.cpp` binds only the op-level functions (`ttnn::pad` overloads) via `bind_function<"pad">` — no factory-innards bindings.
  - **Other risky pybind:** None.
  - **Custom `override_runtime_arguments`:** None. No `override_runtime_arguments` static method in any factory.

## Team-only

### TensorAccessor convertibility (Case-2 bindings)

No Case-2 bindings — all non-sharded tensor bindings are Case 1 (standard page-by-page iteration patterns). No exotic NoC walks.

### Out-of-directory coupling & donor shape analysis

**Op-level roll-up:** `✓ clean` — no cross-family function-call escapes from the op's own kernels. One file-path kernel instantiation creates a port-together coupling (see below).

**Summary table:**

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `pad_tile_program_factory.cpp` | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Cross-family (eltwise/unary) | ✓ Device 2.0 compliant |
| All other factories | `api/dataflow/`, `api/tensor/`, `api/core_local_mem.h` | LLK/HAL | ✓ No concern |
| All other factories | `pad/device/kernels/dataflow/common.hpp` | Local | ✓ No concern |

**Per-call detail:** Omitted — all donor entries are ✓.

**Borrowed kernel files (file-path instantiation):**

`PadTileCoreProgramFactory::create_descriptor` at `pad_tile_program_factory.cpp:105` instantiates `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` — owned by `eltwise/unary`. This kernel is broadly shared; at minimum `eltwise/unary` ops and this `data_movement/pad` factory must coordinate the Metal 2.0 rewrite. Any CTA-slot renaming or binding-token additions to that kernel must land in a single change that updates all consumers simultaneously. Route to planners so the eltwise/unary audit and this port are sequenced together.

### Relaxation candidates

No custom `compute_program_hash` — no relaxation candidates to mine.

### TTNN factory analysis

1. **Op-owned tensors?** Yes. `PadRmReaderWriterMultiCoreProgramFactory::create_workload_descriptor` (line 416 of `pad_rm_reader_writer_multi_core_program_factory.cpp`) and `PadRmReaderWriterProgramFactory::create_workload_descriptor` (line 197 of `pad_rm_reader_writer_program_factory.cpp`) each call `build_pad_value_const_tensor_*()` to allocate a 32-element bfloat16 L1 tensor filled with the pad value, then park it on `WorkloadDescriptor::buffers` as a `WorkloadBuffer{owner, buffer}`. This tensor is not in `tensor_args` or `tensor_return_value`; it is created and owned by the factory. The `WorkloadDescriptor::buffers` mechanism keeps it alive across cache hits (the comment in the code explains this was added to fix issue #44565 — `~Tensor` force-deallocates the device memory regardless of `shared_ptr<MeshBuffer>` owners).

2. **MeshWorkload concept needed?** No (op-owned-tensor artifact only). The two `create_workload_descriptor` factories return a `WorkloadDescriptor` — which materializes to a `MeshWorkload` — solely because they need the `WorkloadDescriptor::buffers` mechanism to keep the pad-value const tensor alive across cache hits. There is no cross-device coordination, no multi-program genuine need. The other five factories use simple `ProgramDescriptor`. Downstream concept selection (`port_op_to_metal2_ttnn_factory.md`) should treat this op as not genuinely requiring MeshWorkload.

3. **Pybind `create_descriptor`?** No. `pad_nanobind.cpp` contains only `bind_function<"pad">` — normal op-surface binding. No `nb::class_<…ProgramFactory>(...).def_static("create_descriptor", …)`.

4. **Other migration-risky pybind?** None. No `DeviceOperation`, `ProgramDescriptor`, or factory-class bindings in `pad_nanobind.cpp`.

5. **Custom hash?** No. No `compute_program_hash` override in `PadDeviceOperation` or any factory.

6. **Custom override-runtime-args?** No. No `override_runtime_arguments` static method in any factory.

## Misc anomalies  *(team-only, non-gating)*

- `PadRmReaderWriterMultiCoreProgramFactory` (`pad_rm_reader_writer_multi_core_program_factory.cpp`) is a very narrow factory: it only supports `nbatch` values of 1, 2, and 8 via hardcoded `switch` cases (see `split_across_cores`). All other values immediately `TT_THROW`. A comment in the code says "generic case -- TODO". The factory is only selected when `use_multicore = true` and layout is ROW_MAJOR, but its actual selection in `select_program_factory` shows it as unreachable (the `use_multicore = true` ROW_MAJOR path selects `PadRmReaderWriterMultiCoreDefaultProgramFactory`, not `PadRmReaderWriterMultiCoreProgramFactory`). This factory may be dead code or only reachable via a legacy path not reflected in `select_program_factory`. Route to op owner for clarification.

- `reader_pad_dims_rm_interleaved.cpp` hardcodes `pad_value_const_buffer_nbytes = 64` at line 52 (`// assumed to be 64 bytes, fails on BH when > 64. TODO: generalize?`), overriding the RTA value at slot 14. The factory computes and passes `pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size()` but the kernel ignores slot 14 entirely. The comment references issue #21978.

## Questions for the user

1. **`PadRmReaderWriterMultiCoreProgramFactory` reachability:** This factory exposes `create_workload_descriptor` but `select_program_factory` (`pad_device_operation.cpp:100`) routes `use_multicore=true, ROW_MAJOR, non-sharded` to `PadRmReaderWriterMultiCoreDefaultProgramFactory`, not this factory. Is `PadRmReaderWriterMultiCoreProgramFactory` dead code, or reachable via another call path? This affects whether it needs to be included in the port scope.

## Recipe notes

- The `WorkloadDescriptor`-based factories (`create_workload_descriptor`) are classified as `ProgramDescriptorFactoryConcept` by the framework (`operation_concepts.hpp:72`), which makes Check 1 GREEN. However, the recipe's Check 1 description doesn't explicitly address `WorkloadDescriptor` as a valid ProgramDescriptor-API form — it describes `ProgramDescriptor` / `KernelDescriptor` / `CBDescriptor` etc., which `WorkloadDescriptor` wraps. This could confuse future auditors. Suggest adding a note to Check 1 that `WorkloadDescriptorConcept ⊂ ProgramDescriptorFactoryConcept` and factories exposing `create_workload_descriptor` (returning `tt::tt_metal::WorkloadDescriptor`) also satisfy the ProgramDescriptor prerequisite.

- The "op-owned tensors" question in TTNN factory analysis (Q1) specifically asks about `create_device_tensor` / `allocate_tensor_on_device` in the factory. The pad factories use `.to_device(device, ...)` on a host tensor (i.e., `Tensor(...).to_device(...)`) rather than `create_device_tensor`. Both result in device-allocated tensors owned by the factory. The recipe's recognition signals could be extended to include the `.to_device()` pattern.
