# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/conv/conv2d`

Single device operation in this directory:

- **`Conv2dDeviceOperation`** (`ttnn::prim`) — device op in `device/conv2d_device_operation.{hpp,cpp}`; types in `device/conv2d_device_operation_types.hpp`.
  - `Conv2dShardedProgramFactory` — `device/conv2d_op_sharded_program_factory.{hpp,cpp}` (height-sharded, block-sharded, and 1D-depthwise paths)
  - `Conv2dWidthShardedProgramFactory` — `device/conv2d_op_width_sharded_program_factory.{hpp,cpp}` (width-sharded path)
  - Shared factory helpers (CB emission, L1-usage checks): `conv2d_op_program_factory_common.{hpp,cpp}` (op root).

`select_program_factory` (`device/conv2d_device_operation.cpp:33`) routes `WIDTH_SHARDED` → `Conv2dWidthShardedProgramFactory`, everything else → `Conv2dShardedProgramFactory`. The two factories share kernels (`conv_bmm_tilize.cpp` compute) and helper code, so they are audited together as one porting unit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`. `prepare_conv2d_weights.cpp`, `conv2d_utils.cpp`, `conv2d.cpp` are host-side weight-prep / config helpers (no program factory) and are out of audit scope except where they touch the device-op surface.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/conv/conv2d` |
| **Overall** | GREEN |
| **DOps / Factories** | `Conv2dDeviceOperation` → `Conv2dShardedProgramFactory`, `Conv2dWidthShardedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | Yes: both factories — `conv_reader_indices_tensor` (`conv2d_op_sharded_program_factory.cpp:1563-1572`, `conv2d_op_width_sharded_program_factory.cpp:727-736`) |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned tensors — carried natively, single-program) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `ACT_SHARDED` (CB, reader-read-ptr endpoint) — workaround |

**Fake CBs** = CBs used purely as an address source. Litmus: producer *and* consumer. `ACT_SHARDED` (the resident input shard) is read by base pointer in the readers and has no kernel producer — fake-CB workaround at the port; does not gate.

## Result

**GREEN → brief issued.** All gates clear:

- **ProgramDescriptor** — both factories populate `ProgramDescriptor` / `WorkloadDescriptor` via `KernelDescriptor` / `CBDescriptor` / `SemaphoreDescriptor`; no imperative `host_api.hpp` builder calls.
- **Device 2.0** — every kernel the op instantiates (10 referenced kernels + the shared `conv_reader_common.hpp` + the pool donor header) is fully on the Device 2.0 object-oriented data-movement API. No legacy idioms; only the sanctioned free functions `get_tile_size` / `get_local_cb_interface` appear.
- **Feature compatibility** — no UNSUPPORTED feature fires. Borrowed-memory DFBs are in use (LANDED).

Port work is mechanical: per-binding `TensorParameter` conversion (all Case 1), borrowed-memory DFB wiring (`borrowed_from`), the `ACT_SHARDED` fake-CB workaround, and deletion of the custom `compute_program_hash`.

## Gate detail

- **ProgramDescriptor:** GREEN. `build_program_descriptor_sharded` (`conv2d_op_sharded_program_factory.cpp:181`) and `build_program_descriptor` (`conv2d_op_width_sharded_program_factory.cpp:56`) construct a `tt::tt_metal::ProgramDescriptor desc;` and push `KernelDescriptor` / `CBDescriptor` / `SemaphoreDescriptor` onto it; each factory's `create_workload_descriptor` (`...sharded...:1461`, `...width_sharded...:692`) wraps the descriptor into a `WorkloadDescriptor`. No `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` anywhere in the device factories.

- **Device 2.0 (every kernel used):** GREEN. All referenced kernels are written end-to-end in the Device 2.0 idiom — `Noc` object for all NoC traffic (reads, writes, multicasts, barriers, zero-fills), `experimental::CB` for all CB access (member-form `reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_read_ptr`/`get_write_ptr`), `Semaphore<>` objects for all sync (no raw `get_semaphore` / `noc_semaphore_*`), `TensorAccessor` / `TensorAccessorArgs` for addr-gen, and typed `MulticastEndpoint`/`UnicastEndpoint`/`McastRect`/`McastDst` endpoints. No `noc_async_read/write` free functions, no `InterleavedAddrGen*` / `ShardedAddrGen`, no `get_noc_addr_from_bank_id`, no CB-index-keyed free functions. The only free functions present are the sanctioned `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`.

  Per-kernel verdicts (all GREEN):

  | Kernel file (`device/kernels/`) | Role | Verdict |
  |---|---|---|
  | `reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp` | reader (height-sharded) | GREEN |
  | `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` | reader (block-sharded) | GREEN |
  | `conv_reader_common.hpp` | shared reader header | GREEN |
  | `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | writer/weights sender (1D) | GREEN |
  | `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | writer/weights receiver (1D) | GREEN |
  | `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | writer/weights sender (2D) | GREEN |
  | `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | writer/weights receiver (2D) | GREEN |
  | `activation_reader_width_sharded.cpp` | reader (width-sharded) | GREEN |
  | `weights_reader_width_sharded.cpp` | weights reader (width-sharded) | GREEN |
  | `reader_depthwise_conv1d.cpp` | reader (1D depthwise) | GREEN |
  | `conv_bmm_tilize.cpp` | compute (all paths except depthwise) | GREEN |
  | `compute_depthwise_conv1d.cpp` | compute (1D depthwise) | GREEN |

  *Note (not a holdover):* `conv_bmm_tilize.cpp` does manual CB FIFO-pointer manipulation via `get_local_cb_interface(cb_id).fifo_rd_ptr/fifo_wr_ptr` (e.g. `:71`, `:77`, `:369`, and the partials spill/reload saves/restores around `:296-297`, `:519-565`) to implement activation-reuse / split-reader pointer juggling and matmul-partials spill/reload. `get_local_cb_interface` is the *sanctioned* Device 2.0 free function (no member-form replacement today), so this is **not** a holdover. Flagged only so the porter knows these `fifo_*_ptr` writes are load-bearing and have no OO wrapper equivalent.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | `global_circular_buffer.hpp` is `#include`d at `conv2d_device_operation.hpp:12` but no `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, `.global_circular_buffer` field, `remote_cb`/`remote_index`, or 4-arg `CreateCircularBuffer(..., global_cb)` is used anywhere. Header-only presence; feature absent. |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `emit_cb_descriptors` (`conv2d_op_program_factory_common.cpp:756`) sets `CBDescriptor::buffer` non-null for `ACT_SHARDED` (input buffer), `OUT`/`MATMUL_PARTIALS` (output buffer), `READER_INDICES` (indices buffer) — `:770-795`. Port uses `borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress`. Default zero. |
  | Aliased Circular Buffers | N/A | Every `CBDescriptor::format_descriptors` initializer is single-element (`conv2d_op_program_factory_common.cpp:789`). No multi-`buffer_index` config. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, `CreateGlobalSemaphore`, or `global_semaphore.hpp`. All semaphores are plain `SemaphoreDescriptor`. |
  | Non-zero semaphore initial value | N/A | All `SemaphoreDescriptor::initial_value` are explicit literal `0` (`conv2d_op_sharded_program_factory.cpp:728`, `conv2d_op_width_sharded_program_factory.cpp:370,376`). The `// 0 == INVALID` comment refers to the value 0, not a non-zero sentinel. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` token anywhere. All `TensorAccessorArgs(buffer)` are the single-argument standard form (`conv2d_op_sharded_program_factory.cpp:875,1035,1036`; `conv2d_op_width_sharded_program_factory.cpp:571,578,579`). |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize` calls. The two `UpdateDynamicCircularBufferAddress` mentions (`...sharded...:686`, `...width_sharded...:465,472`) are comments documenting that the call is **not** needed (framework patches addresses on cache hit). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` (`Conv2dInputs`) is fixed: `a`, `b`, `std::optional<Tensor> bias` — no `std::vector<Tensor>`. CT-arg vectors are built with fixed (branch-determined) cardinality; no kernel reads `get_compile_time_arg_val(i)` with a runtime-varying `i`. |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, both factories):
  - **`b` (weights)** — Case 1. Host pushes `Buffer* weights_buffer` as a runtime-arg `Buffer*` binding (`conv2d_op_sharded_program_factory.cpp:1302`, `conv2d_op_width_sharded_program_factory.cpp:667`); the kernel reads the base via `get_arg_val<uint32_t>` and feeds it to `TensorAccessor(s_weight_args, weight_addr_dram_base)` (sender writers: `reader_writer_tiled_out_1d_mcast_sender_...:135`, `writer_tiled_out_2d_mcast_sender_...:158`, `weights_reader_width_sharded.cpp:42`). Express as `TensorParameter`; kernel builds `TensorAccessor(ta::weights)`. The `Buffer*`-binding shape is framework-patched on cache hits today, so this is routine — not a silent-wrong hazard.
  - **`bias` (optional)** — Case 1, same shape. `Buffer* bias_buffer` binding (`...sharded...:1304`, `...width_sharded...:669`) consumed via `TensorAccessor(s_bias_args, bias_addr)` (`...1d_mcast_sender...:131`, `...2d_mcast_sender...:153`, `weights_reader_width_sharded.cpp:45`). When `bias` is absent the host pushes a literal `0` and `has_bias` gates the read kernel-side; the port keeps that gating.
  - **`conv_reader_indices` (op-owned config tensor)** — Case 1 in the DRAM path. Address baked into a CT arg plus `TensorAccessorArgs(conv_reader_indices_buffer).append_to(...)` (`...sharded...:873-875`, `...width_sharded...:569-571`); the kernel rebuilds a `TensorAccessor` over it (`conv_reader_common.hpp:361-364`). In the L1-small path it is a borrowed-memory DFB (`READER_INDICES` CB, see below). Bind as `TensorParameter`. Per-factory note: only fires when `config_tensors_in_dram == true`.
  - **`a` (activation, input shard)** — clean borrowed-memory DFB, *except* see Fake CBs below. Read via `cb_sharded_act.get_read_ptr()`; the `ACT_SHARDED` CB is on the input buffer.
  - **Output (`OUT` / `MATMUL_PARTIALS`)** — clean borrowed-memory DFB on the output buffer; produced by compute, consumed by the writer.
- **Custom hash:** delete custom `compute_program_hash` (`device/conv2d_device_operation.cpp:145-166`) → default (sanctioned exception). See Custom program hash.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Borrowed-memory DFB** — `ACT_SHARDED` (input buffer), `OUT` / `MATMUL_PARTIALS` (output buffer), `READER_INDICES` (config buffer, L1-small path). Emitted by `emit_cb_descriptors` setting `CBDescriptor::buffer` (`conv2d_op_program_factory_common.cpp:770-795`). Port uses `DataflowBufferSpec::borrowed_from = <TensorParameter>`.
- **Fake CBs (address-only):** **`ACT_SHARDED`** at the reader read-ptr endpoint. It is the resident input shard read purely by base pointer (`reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp:77`, `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp:216`, `activation_reader_width_sharded.cpp:138,149`, sender split-reader `reader_writer_tiled_out_1d_mcast_sender_...:115`) — there is a consumer (the reader/tilize path) but no kernel producer pushing into it. This is the canonical sharded-input "resident input, read by raw pointer, no producer" shape. The port resolves it with the sanctioned fake-CB workaround; it does **not** gate. (Contrast `READER_INDICES`, which *is* a genuine borrowed-DFB: the reader writes config into it via `get_write_ptr` and a consumer `wait_front(1)`s on it — `reader_writer_tiled_out_1d_mcast_sender_...:103`.)
- **Cross-op / shared kernels:** every conv2d kernel `#include`s `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` — the **pool** op's Device-2.0 convenience header (aliases `experimental::CB`, `Noc`, `Semaphore<>`, `read_with_state`, endpoint types). It is itself a thin Device-2.0 wrapper (clean). `conv_bmm_tilize.cpp` additionally includes `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` / `untilize_helpers.hpp` (official shared kernel-lib); `compute_depthwise_conv1d.cpp` includes `tilize_helpers.hpp`. These induce a port-the-family-together coupling on the shared header — see Team-only § coupling.
- **RTA varargs:** none. No kernel reads `num_runtime_varargs` or loop-retrieves RTAs with a runtime-varying index; the reader/writer mcast-NoC lookup tables are fixed-cardinality CT/RT vectors.
- **TTNN factory analysis (porter-relevant):** pybind `create_descriptor` — none. Other risky pybind — none (`conv2d_nanobind.cpp` binds only `bind_function<"conv2d">`, `mod.def("prepare_conv_weights"/"prepare_conv_bias", ...)`, and the `Conv2dConfig` config class — all normal op surface). Custom `override_runtime_arguments` — none.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean (function-call escapes) · file-path coupling present (shared Device-2.0 header + kernel-lib helpers).

All cross-directory `#include`s resolve to framework or shared-pool donors that are already Device 2.0:

| Op kernel | Donor include | Donor class | Status |
|---|---|---|---|
| all kernels | `api/dataflow/dataflow_api.h`, `api/dataflow/*`, `api/compute/*`, `api/core_local_mem.h`, `noc/noc_parameters.h`, `api/debug/dprint*.h` | `tt_metal/*` LLK/HAL/firmware | ✓ no concern |
| all kernels | `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | cross-family (pool) — shared Device-2.0 convenience header | ✓ Device 2.0 (it *is* the wrapper-aliasing layer) |
| `conv_bmm_tilize.cpp`, `compute_depthwise_conv1d.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp` | `ttnn/cpp/ttnn/kernel_lib/` official shared kernel lib | ✓ lib team handles internally |
| `conv_reader_common.hpp` | `tt-metalium/constants.hpp` | Metalium public | ✓ no concern |

No function-call escape lands on Shape 3/4 or `CircularBuffer&` / raw-sem shapes — there are no ⚠/✗/⭐ donor rows, so per-call detail is omitted. The only donor "function-call" surface (the pool `experimental_device_api.hpp` helpers — `read_with_state`, `set_read_state`, `local_addr`, `set_read_trid`, `async_read_barrier_with_trid`) take `Noc` / `experimental::CB` / typed-endpoint arguments → Shape 1 / `uint32_t cb_id` equivalents (✓ excellent).

**Borrowed kernel files (file-path instantiation).** The factories instantiate kernels by file path; conv2d **owns** all of them (they live in `conv2d/device/kernels/`), so there is no cross-family file-path borrow. The coupling that *does* exist is the **shared header** `pool/.../experimental_device_api.hpp`, included by every conv2d kernel and (per its own comment) intended for "conv/pool operations." Its Metal 2.0 rewrite, if any, is a single change that both conv2d and pool must adopt together → conv2d + pool form a port-together set for that header. (This is independent of the Device 2.0 gate, which the header already passes.)

### Relaxation candidates (mined from the custom hash before deletion) — FALLIBLE, candidates to verify; default strict

The custom `compute_program_hash` (`device/conv2d_device_operation.cpp:145`) hashes `hashable_operation_attributes_t` (a `Conv2dHashableParams`) **plus** `tensor_args` via `hash_objects_with_default_seed`. Notably it **omits** `groups`, `full_inner_dim`, and `pre_op_l1_allocation_size_bytes` from the full `Conv2dParams` (they are absent from `Conv2dHashableParams`, `conv2d_device_operation_types.hpp:221-238`). It still hashes `tensor_args` (so `TensorSpec` is keyed — this hash is *not* obviously of the broken `TensorSpec`-omitting class). No safe relaxation candidate is evident: the program shape depends on the full sharding/parallelization config and the input `TensorSpec`. **Verify before relaxing**; the deletion to the default hash (which keys on op type, attributes, and each tensor's `TensorSpec`) is correct-by-construction and is what the port should land. The omitted `groups` is also a Misc anomaly (below).

### TTNN factory analysis (six questions)

1. **Op-owned tensors? — Yes.** Both factories allocate the `conv_reader_indices` config tensor in `create_workload_descriptor` and park it on `workload_descriptor.buffers`: `conv2d_op_sharded_program_factory.cpp:1563-1572`, `conv2d_op_width_sharded_program_factory.cpp:727-736` (`construct_on_host_config_tensor` → `move_config_tensor_to_device`, wrapped in a `shared_ptr<Tensor>`). It is neither an input nor the output — a genuine op-owned intermediate.
2. **MeshWorkload concept needed? — No (op-owned-tensor artifact).** Both factories provide `create_workload_descriptor` and return a `WorkloadDescriptor` — the MeshWorkload path — but the per-coord program is explicitly "structurally identical for every coord … conv2d doesn't depend on cluster position" (`...sharded...:1574-1587`, `...width_sharded...:738-751`): the same `ProgramDescriptor` is copied into every range entry. The op is on this path to carry the op-owned `conv_reader_indices` tensor, not for cross-program/cross-device coordination. `MetalV2FactoryConcept` carries op-owned tensors natively (`op_owned_tensors`), so this ports cleanly as morally single-program. Cause: Q1.
3. **Pybind `create_descriptor`? — No.** `conv2d_nanobind.cpp` contains no `nb::class_<…ProgramFactory>` / `def_static("create_descriptor", …)`. Only `bind_function<"conv2d">` (`:154`), `mod.def("prepare_conv_weights", …)` (`:181`), `mod.def("prepare_conv_bias", …)` (`:207`), and `nb::class_<Conv2dConfig>` (`:233`) — all normal op surface.
4. **Other migration-risky pybind? — None.** No `DeviceOperation` / factory / param-struct class is wrapped; no device-op method (`compute_program_hash`, `create_output_tensors`, `compute_output_specs`, `select_program_factory`) is exposed to Python.
5. **Custom hash? — Yes.** `Conv2dDeviceOperation::compute_program_hash` at `device/conv2d_device_operation.cpp:145`. Treatment: delete → default (see Custom program hash subject / Port-work summary).
6. **Custom override-runtime-args? — No.** Neither factory defines `override_runtime_arguments`.

## Misc anomalies  *(team-only, non-gating)*

- **`groups` omitted from the hash.** `Conv2dParams::groups` is consumed by both factories (drives `is_1d_depthwise_conv`, e.g. `conv2d_op_sharded_program_factory.cpp:332-333`) and so affects which kernels/CT-args are generated, but it is **absent** from `Conv2dHashableParams` (`conv2d_device_operation_types.hpp:221-238`) and therefore from the custom `compute_program_hash`. Two invocations differing only in `groups` (e.g. depthwise vs. grouped vs. dense at the same shapes) could collide on the custom hash. The port's switch to the default hash (which keys on the full attribute set) resolves this incidentally; flagged for the op owner. `full_inner_dim` is likewise omitted but is folded into `block_config`/parallelization in practice.
- **Commented-out perf-model logging** (`device/conv2d_device_operation.cpp:213-222`, `#if 0`) references `pad_h`/`pad_w`/`this->output_channels` that no longer exist in scope — dead, would not compile if re-enabled. Op owner.

## Recipe notes

- The Buffer*-binding shape (recipe *Detection — host side*, `Buffer*`-binding form) and the kernel-side `TensorAccessor(args, addr)` form (Case 1) co-occur here: the host pushes a `Buffer*` into the RTA list (framework-patched), and the *kernel* turns that base into a `TensorAccessor`. The recipe says the `Buffer*` form is "Case 2 (raw `uint32_t` base)", but the kernel here actually constructs a `TensorAccessor` from it, which the recipe elsewhere defines as Case 1. I classified by the *kernel's* use (Case 1, per "classify by what the kernel does with the tensor's base pointer"), since the binding's destination construct is what the port rewrites. Worth a one-line clarification in the recipe that the `Buffer*` host shape does not by itself force Case 2 — the kernel's downstream use still decides Case 1 vs 2.
