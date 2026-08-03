# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/moe`

- **`MoeDeviceOperation`** (`device/moe_device_operation.hpp` / `.cpp`)
  - `MoeProgramFactory` (`device/moe_program_factory.cpp`) — the only factory

Kernels (all three owned by this op, all three referenced by the factory; no unreferenced kernel files in the directory):

- `device/kernels/dataflow/reader_create_index_tensor.cpp` (reader)
- `device/kernels/dataflow/writer_unary_interleaved.cpp` (writer)
- `device/kernels/compute/moe.cpp` (compute)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/moe` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MoeDeviceOperation` → `MoeProgramFactory` (single factory, single config) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all three kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `CoreLocalMem`, `TensorAccessor`); both `kernel_lib` donors likewise. See the judgment note on `get_dataformat(cb_id)` below. |
| *Prereqs* — Cross-op escapes | Ok — two `ttnn/cpp/ttnn/kernel_lib/` includes, both with `uint32_t cb_id` NTTP signatures (✓ shape). No borrowed kernel *files*. |
| *Feature Support* — overall | GREEN — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a literal constant |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor` |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (confirmed: no `compute_program_hash` / `attribute_values` / `to_hash` anywhere in the op) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (confirmed by grep of the device-op) |
| *TTNN Readiness* — `override_runtime_arguments` | No (confirmed by grep of the device-op and factory) |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `moe_nanobind.cpp` binds only the user-facing `ttnn::moe` function |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none — the op contains **no** `->address()` expression at all; tensor bases reach the kernels as `MeshTensor` bindings |
| *Port work* — Tensor bindings (per binding) | 4 bindings, all **Case 1** (`TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | N/A — no accessor in this op passes a 3rd argument |
| *Port work* — CB endpoints | 13 CBs: 6 legal 1:1 · 7 **self-loop** · 0 multi-binding · 0 dead |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. This op has a single instantiation (one core, interleaved-only — sharding is rejected in `validate_on_program_cache_miss`), so each CB has exactly one disposition; there is no per-config split to record.

## Result

**GREEN → brief issued.** `METAL2_PORT_BRIEF.md` is written alongside this file.

Every gate cleared. The op is a good port candidate: one factory, one core, one config, no semaphores, no sharded path, no `->address()` smuggling, no `TensorAccessor` page-size overrides, and kernels that are already fully on Device 2.0 idioms with `DataflowBuffer` objects throughout. The port work is a small, mechanical set: four Case-1 tensor bindings, seven self-loop DFBs, and named compile-time args.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet's row for `reduction/moe` / `MoeDeviceOperation` / `MoeProgramFactory` reads `Is able to port? = yes`, with every conjunct clean: `Concept = descriptor`, `Custom hash = no`, `Backdoor custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`. Cross-check clean on every cheaply-checkable column:
  - `Concept` — [`moe_program_factory.cpp:18`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L18) defines `MoeProgramFactory::create_descriptor(...)` returning a `tt::tt_metal::ProgramDescriptor`; [`moe_device_operation.hpp:24`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_device_operation.hpp#L24) declares `program_factory_t = std::variant<MoeProgramFactory>`. Vanilla single-program `descriptor`. ✓
  - `Custom hash` — a grep of the whole op directory for `compute_program_hash`, `attribute_values`, and `to_hash` returns nothing. ✓
  - `get_dynamic_runtime_args` — absent from [`moe_device_operation.hpp:19-28`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_device_operation.hpp#L19-L28); the device-op declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`. ✓
  - `override_runtime_arguments` — absent from both the device-op and the factory. ✓
  - `Pybind descriptor` — [`moe_nanobind.cpp:73-83`](ttnn/cpp/ttnn/operations/reduction/moe/moe_nanobind.cpp#L73-L83) binds only `ttnn::moe` via `ttnn::bind_function`; no `nb::class_` of the device op, no `create_descriptor` binding. ✓
  - `Op-owned tensors` — the factory returns a bare `ProgramDescriptor` (no `buffers` vector exists on that type), so op-owned tensors are structurally impossible here. Sheet cell is blank, consistent. ✓
  - **Factory-set match** — the sheet has exactly one row for this op and the code has exactly one factory (`MoeProgramFactory`). No phantom row, no missing row. ✓
  - **Cross-column invariants** — no violation: `get_dynamic_runtime_args = no` on a `descriptor` row is legal, and `Op-owned tensors` is blank (not `yes`) on a `descriptor` row. ✓

  Sheet columns also read `Porting Target = ProgramSpecFactoryConcept`, `Execution Model = SPMD`, `TensorParameter relaxation = none`, `Pointer patching perf issue? = OK`, `Formerly custom hashed? = no` — all consistent with the code.

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels are structurally Device 2.0, and so are both `kernel_lib` donors they call into. Evidence:

  | Kernel | Device 2.0 idioms observed |
  |---|---|
  | [`reader_create_index_tensor.cpp`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp) | `Noc noc` + `noc.async_read(...)` / `noc.async_read_barrier()` ([:72](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L72), [:83-86](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L83-L86)); `DataflowBuffer` objects for every FIFO op ([:73-75](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L73-L75)); `CoreLocalMem<volatile uint32_t>` for the index-tile fill ([:22](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L22)); `TensorAccessor` for all three inputs ([:62](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L62), [:66](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L66), [:70](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L70)) |
  | [`writer_unary_interleaved.cpp`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp) | `Noc noc` + `noc.async_write(...)` / `noc.async_write_barrier()` ([:33](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L33), [:40-49](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L40-L49)); `DataflowBuffer dfb_out` for the FIFO ops ([:34](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L34), [:37](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L37), [:50](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L50)); `TensorAccessor` for the output ([:31](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L31)) |
  | [`compute/moe.cpp`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp) | `DataflowBuffer` objects for **every** FIFO operation across all helpers ([:31-32](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L31-L32), [:69-70](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L69-L70), [:104-105](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L104-L105), [:133-134](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L133-L134), [:163](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L163), [:190](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L190), [:259-266](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L259-L266)). No NoC operations at all — it is a pure CB-to-CB compute kernel. |
  | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl` (donor) | `DataflowBuffer` + `Noc` (`:161-168`, `:203`) |
  | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl` (donor) | `DataflowBuffer` objects throughout (`:337-340` and all FIFO sites) |

  A targeted scan for Device 1.0 idioms across all three kernels found **zero** hits for `noc_async_read` / `noc_async_write` / `noc_async_read_tile` / `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` / `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*` / `get_noc_addr_from_bank_id` / free-function `get_read_ptr(` / free-function `get_write_ptr(` / `noc_semaphore_*` / `get_semaphore`.

  **CB-index free functions — one judgment call, resolved GREEN.** The kernels make four `get_tile_size(<dfb_id>)` calls ([reader :60](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L60), [:64](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L64), [:68](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L68); [writer :23](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L23)) — explicitly **sanctioned** by the audit recipe's Green bullet, not flagged. There is one further CB-index-keyed free function the recipe's sanction list does not name:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/writer_unary_interleaved.cpp` | 24 | `get_dataformat(out_dfb_index)` | `DataflowBuffer dfb_out` declared 10 lines later (`:34`); `DataflowBuffer::get_dataformat()` exists |

  **Not treated as a holdover — GREEN.** The recipe's controlling rule is *"if Device 2.0 allows the free function, so do we,"* and instructs checking the current Device 2.0 surface rather than judging by shape. Three pieces of evidence say Device 2.0 keeps this one: (1) `CircularBuffer::get_dataformat()` at `tt_metal/hw/inc/api/dataflow/circular_buffer.h:115` is a pure forward to `::get_dataformat(cb_id)`, structurally identical to `CircularBuffer::get_tile_size()` at `:113` forwarding to the sanctioned `::get_tile_size(cb_id)`; (2) the free function is used at 94 sites across TTNN kernels outside `experimental/quasar`, including many kernels already migrated to `Noc` / `CircularBuffer` / `DataflowBuffer` idioms (`reduction/argmax`, `experimental/paged_cache`, the `nlp_create_qkv_heads` family, `experimental/ccl/rms_allgather`); (3) it is a tile/format *metadata accessor*, the same family the recipe's breadcrumb describes as moving onto the object **at Metal 2.0 port time** (kernel-side whitelist rule 7), not at the Device 2.0 boundary. Logged as a recipe note below so the sanction list can be made explicit either way. Separately, this particular call's result is **dead** (see Misc anomalies), so the port drops the line regardless.

- **Feature compatibility:** every Appendix A entry, in order. A directory-wide grep for all four entries' recognition signals returned no hits.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any of the 13 `CBDescriptor`s ([`moe_program_factory.cpp:73-224`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L73-L224) — each sets only `total_size`, `core_ranges`, `format_descriptors`), no `remote_index` / `remote_cb_*` / `remote_circular_buffer.h`, no `<tt-metalium/global_circular_buffer.hpp>` include. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The field is never mentioned. No `set_address_offset`, no `UpdateDynamicCircularBufferAddress` (either arity), no `cb_descriptor_from_sharded_tensor`, no `set_globally_allocated_address`. No borrowed-memory CBs at all — all 13 CBs are plain SRAM allocations. |
  | GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — a grep for `semaphore` / `Semaphore` across the whole op directory returns nothing, and `desc.semaphores` is never populated. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Both signals absent. Op-level: `MoeInputs` ([`moe_device_operation_types.hpp:17-22`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_device_operation_types.hpp#L17-L22)) carries three named `Tensor` fields plus one `std::optional<Tensor>` — a fixed-count input set, no variable-count container. Kernel-level (the decider): all 29 `get_compile_time_arg_val` calls across the three kernels use **literal constant** indices (reader 0-6, writer 0-2, compute 0-18); no loop, no computed index, no runtime-varying count. |

- **CB endpoints (GATE-free):** classified below. Single instantiation (single core `CoreRange({0,0},{0,0})` at [`moe_program_factory.cpp:27`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L27); interleaved-only, since `validate_on_program_cache_miss` rejects sharded output at [`moe_device_operation.cpp:46`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_device_operation.cpp#L46)), so there is one node and one config. Census run per CB over all three kernels, including a hunt for raw-pointer touchers.

  | CB | Name | Producer(s) | Consumer(s) | Census | Disposition |
  |---|---|---|---|---|---|
  | `c_0` | input | reader (FIFO) | compute (FIFO) | 1 locked P + 1 locked C | legal 1:1 |
  | `c_1` | expert_mask | reader (FIFO) | compute (FIFO) | 1 locked P + 1 locked C | legal 1:1 |
  | `c_2` | topk_mask | reader (FIFO) | compute (FIFO) | 1 locked P + 1 locked C | legal 1:1 |
  | `c_3` | scale | writer (FIFO, via `kernel_lib`) | compute (FIFO `wait_front`, never pops) | 1 locked P + 1 locked C | legal 1:1 |
  | `c_4` | index | reader (FIFO) | compute (FIFO) | 1 locked P + 1 locked C | legal 1:1 |
  | `c_5` | input_transposed | compute | compute | **1 toucher** | **self-loop** |
  | `c_6` | index_transposed | compute | compute | **1 toucher** | **self-loop** |
  | `c_7` | values | compute | compute | **1 toucher** | **self-loop** |
  | `c_8` | output_ind | compute | compute | **1 toucher** | **self-loop** |
  | `c_9` | cur_max | compute | compute | **1 toucher** | **self-loop** |
  | `c_10` | cur_sum | compute | compute | **1 toucher** | **self-loop** |
  | `c_11` | out | compute (FIFO, via `kernel_lib` reduce) | writer (FIFO) | 1 locked P + 1 locked C | legal 1:1 |
  | `c_12` | masked_input | compute | compute | **1 toucher** | **self-loop** |

  Supporting detail for the non-obvious rows:

  - **`c_4` index — the one raw-pointer write, and it is not a second toucher.** [`generate_index_tile`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L18-L38) fills the tile through `dfb.get_write_ptr()` ([:22](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L22)), but the *same* kernel brackets it with `dfb.reserve_back(1)` ([:21](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L21)) and `dfb.push_back(1)` ([:37](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L37)). A FIFO producer that also raw-writes its own buffer is **one** toucher — the PRODUCER binding covers the peek. Compute consumes at [`compute/moe.cpp:283`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L283) / [:320](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L320). So 2 touchers, one locked producer + one locked consumer → legal 1:1.
  - **Hidden-second-writer hunt: negative, and structurally so.** The recipe's face (a) needs a raw co-fill coordinated by dedicated semaphores; this op allocates no semaphores at all, and the only `get_write_ptr` in the whole op is the reader's own (above). Faces (b) and (c) also cannot apply: there are no borrowed-memory / tensor-view CBs (no CB is backed by a device buffer), and no kernel source is instantiated twice — the factory pushes exactly three `KernelDescriptor`s, one per source ([`moe_program_factory.cpp:291-293`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L291-L293)).
  - **`c_3` scale is written by the *writer* kernel.** `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<...>()` at [`writer_unary_interleaved.cpp:28-29`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L28-L29) does the `reserve_back(1)` / `push_back(1)`; compute reads it via `compute_kernel_lib::reduce` (`reduce_helpers_compute.inl:364`, `scaler_dfb.wait_front(1)`, never popped). Both endpoints are real and locked, so this is a genuine cross-kernel 1:1 — worth flagging only because the producer is the *writer*, which is not where a reader-produces/compute-consumes reading of the op would look.
  - **The seven self-loops are all compute-internal scratch.** Each is produced and consumed inside `compute/moe.cpp` only, and neither dataflow kernel names its index: `c_5`/`c_6` are the single-buffered transpose/merge working set ([:277-278](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L277-L278), [:380-387](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L380-L387)); `c_7`/`c_8` hold the top-k values and indices across the whole softmax chain ([:402-408](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L402-L408), [:484-495](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L484-L495)); `c_9`/`c_10` are the reduce max/sum intermediates ([:488-492](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L488-L492)); `c_12` is the two-tile masked-input staging buffer ([:291-308](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L291-L308)).
  - **No dead CB.** All 13 `buffer_index` values are referenced. Twelve arrive as compile-time args (reader 0-3, writer 0, compute 0-9 + 15-16 + 18); `c_3` reaches the writer as a hardcoded literal rather than a CTA (see Misc anomalies) but is genuinely referenced at [`writer_unary_interleaved.cpp:27`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L27).

- **Offset base pointers:** **GREEN.** There is no `->address()` expression anywhere in the op — a directory-wide grep for `address()` returns zero hits, as does one for `narrow(`. The four tensor bases reach the kernels through the framework's `MeshTensor` binding channel: [`moe_program_factory.cpp:239-245`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L239-L245) and [:257-261](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L257-L261) call `emplace_runtime_args(core, {<mesh tensors>})`, which the framework auto-registers as `BufferBinding`s (see the `KernelDescriptor::emplace_runtime_args` overload taking `std::reference_wrapper<const MeshTensor>` and the `buffer_bindings` field in `tt_metal/api/tt-metalium/program_descriptors.hpp:164-201`). No host arithmetic is applied to any base — there is none to fold an offset into. Neither Type 1 nor Type 2 applies; Type 3 (`address_offset`) is absent per Appendix A above; Type 4 (`narrow`) is absent.

  Reconciled against the dated triage prior `2026-07-19_offset_base_pointers.md`: `reduction/moe` is **not** in its tables, and my own scan finds no fold — the *"no fold, op not in the tables"* outcome. Clean base; handed to TensorParameter analysis. (The doc's `moe`-named entries are unrelated ops: `fused_ops/moe/op.py`, `fused_ops/moe_routed_expert/op.py`, and the C++ `deepseek_moe_gate` / `generalized_moe_gate` gate builders.)

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** All four `TensorAccessor` constructions in the op pass exactly two arguments: [`reader_create_index_tensor.cpp:62`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L62) `TensorAccessor(s0_args, src_addr)`, [:66](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L66) `TensorAccessor(s1_args, topk_addr)`, [:70](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L70) `TensorAccessor(s2_args, expert_addr)`, and [`writer_unary_interleaved.cpp:31`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L31) `TensorAccessor(out_args, dst_addr0)`. No explicit page size is supplied at any site, so there is nothing to classify and nothing for the port to drop.

  Reconciled against the dated triage prior `2026-07-06_tensor_accessor_3rd_arg_triage.md`: `reduction/moe` is not in its table, consistent with having no 3rd-arg site. (Its `moe`-named rows are other ops: `moe_grouped_topk`, `deepseek_moe_fast_reduce_nc_fused`, `deepseek_moe_post_combine_reduce`.) Worth a note for the porter: the reader *does* compute `tile_bytes_*` values via `get_tile_size` ([:60](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L60), [:64](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L64), [:68](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L68)) and passes them to `noc.async_read` as the *transfer size* — that is a NoC argument, not an accessor page-size override, and it stays.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — 4 bindings, **all Case 1** (base fed into a `TensorAccessor`, all memory access through the accessor). The legacy delivery mechanism is the `MeshTensor` form of `emplace_runtime_args`, which the framework already patches on cache hits, so none of these is the silent-wrong RTA hazard — but all four are still enumerated as routine port work, and the classification is by what the kernel does with the base:

  | Binding | Legacy delivery | Kernel consumption | Case |
  |---|---|---|---|
  | `input` | [`moe_program_factory.cpp:242`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L242) → reader RTA 0 | `TensorAccessor(s0_args, src_addr)` [reader :62](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L62), read via `noc.async_read(s0, …)` [:96](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L96), [:99](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L99) | **Case 1** |
  | `topk_mask` | [`moe_program_factory.cpp:243`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L243) → reader RTA 1 | `TensorAccessor(s1_args, topk_addr)` [reader :66](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L66), read at [:112](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L112) | **Case 1** |
  | `expert_mask` | [`moe_program_factory.cpp:244`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L244) → reader RTA 2 | `TensorAccessor(s2_args, expert_addr)` [reader :70](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L70), read at [:83](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L83) | **Case 1** |
  | `output` | [`moe_program_factory.cpp:260`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L260) → writer RTA 0 | `TensorAccessor(out_args, dst_addr0)` [writer :31](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L31), written at [:40-46](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L40-L46) | **Case 1** |

  No binding is `clean` (there are no borrowed-memory DFBs), and no binding is Case 2 (no kernel does raw address arithmetic on a base). The compute kernel is out of scope for this subject — it only consumes from and produces to CBs, and touches no tensor memory.

  **Op-level roll-up:** `⚠ port work` — four Case-1 bindings, all mechanical.

  Note the knock-on: since all four RTAs *are* the tensor bases, expressing them as `TensorParameter` / `TensorBinding` leaves the reader and writer with **zero** runtime args. The three `TensorAccessorArgs` blocks appended at [`moe_program_factory.cpp:228-230`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L228-L230) and the one at [:248](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L248), plus the kernel-side `TensorAccessorArgs<7>` / chained-offset declarations at [reader :55-57](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L55-L57) and [writer :19](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L19), all disappear with them.

- **TensorParameter relaxation:** none. Sheet says `none`, and the op has no custom hash for a relaxation to be consistent with.

- **TensorAccessor 3rd arg:** none — no site passes one.

- **CB endpoints:** self-loop `c_5`, `c_6`, `c_7`, `c_8`, `c_9`, `c_10`, `c_12` (each a compute-internal scratch buffer with a single toucher — bind compute both PRODUCER and CONSUMER). The other six are already legal 1:1 and need no action. No 1P+1C assignment needed, no multi-binding advanced option, no dead-CB drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The hidden-second-writer hunt came back negative and structurally so — the op allocates no semaphores, has no borrowed-memory CB, and instantiates no kernel source twice.
- **Cross-op / shared kernels:** none to fork or coordinate. The op owns all three kernel `.cpp` files, and a repo-wide grep confirms no other op or test instantiates any of them. The two out-of-directory `#include`s both land in `ttnn/cpp/ttnn/kernel_lib/`, which the shared-kernel caution explicitly excludes from its scope (never forked, out of porter scope), and neither has a `_metal2` sibling.
- **`kernel_lib` call shapes:** both donor entry points take the DFB id as a `uint32_t` non-type template parameter, the ✓ shape — `dfb::name`'s constexpr cast covers template-parameter position, so both call sites port by substituting the token:
  - `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<scale_dfb_index, PoolType::SUM, ReduceDim::REDUCE_ROW>()` — [`writer_unary_interleaved.cpp:28-29`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L28-L29)
  - `compute_kernel_lib::reduce<pool_type, reduce_dim, in_dfb, scale_dfb, out_dfb, ReduceInputPolicy::WaitUpfrontNoPop>(...)` — [`compute/moe.cpp:222-229`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L222-L229), called four times from `kernel_main`
- **Donor internal that looks like a stray binding:** `reduce_helpers_compute.inl:340` constructs `DataflowBuffer accum_dfb(...)` with a literal `0` when accumulation is disabled (which is the case for all four `reduce` calls here — no `Accumulate` argument is passed). No FIFO operation is ever issued on it; the `accum_dfb.wait_front` / `pop_front` sites (`:196`, `:217`) sit behind an `enable_accumulation` guard. So it is inert, and in this op `c_0` is bound to compute anyway (compute consumes the input CB). Flagged only so a `DataflowBuffer(0)` sighting inside the donor doesn't read as an unbound touch.
- **RTA varargs:** none. Every runtime-arg read uses a literal constant index and is a distinct field: reader [:41-43](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L41-L43) reads args 0, 1, 2; writer [:12](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L12) reads arg 0. No counted loop over args, no data-selected index, no running `arg_index++`. All four become tensor bindings anyway, so the kernels end up with no runtime args to name.

## Team-only

- **Out-of-directory coupling & donor shape.**

  **Op-level roll-up: ✓ clean.** Two donor files, both in the official shared kernel library, both with the ✓ `uint32_t cb_id` handle shape. No Shape-4 (pre-Device-2.0 addr-gen) donor, no `CircularBuffer&` parameter, no `uint32_t sem_id` / `sem_addr`, no `TensorAccessorArgs<N>` or CTA-offset-NTTP donor signature. No scheduling blocker (no ⭐ entry).

  **Summary table** — one row per (op kernel, donor file):

  | Op kernel | Donor file | Donor class | Status |
  |---|---|---|---|
  | `reader_create_index_tensor.cpp` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` LLK/HAL | ✓ no concern |
  | `writer_unary_interleaved.cpp` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ no concern |
  | `writer_unary_interleaved.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — official shared kernel library | ✓ `uint32_t cb_id` (NTTP) |
  | `compute/moe.cpp` | `tt_metal/hw/inc/api/compute/*` (11 headers), `api/debug/dprint.h`, `ckernel_sfpu.h`, `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` | ✓ no concern |
  | `compute/moe.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | 2 — official shared kernel library | ✓ `uint32_t cb_id` (NTTP) |

  **Per-call detail:** omitted — all rolls are ✓.

  **Borrowed kernel files (file-path kernel instantiation): none.** All three `KernelDescriptor::kernel_source` paths point inside this op's own directory ([`moe_program_factory.cpp:233-234`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L233-L234), [:251-252](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L251-L252), [:285](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L285)). A repo-wide grep for `reduction/moe/device/kernels` across `ttnn/` and `tests/` finds no consumer outside this op, so none of the three files is lent either. There is no `experimental/quasar` copy of this op — a `find` under that tree for any `moe`-matching path returns nothing — so the porter has no out-of-bounds lookalike to trip over here.

- **Relaxation candidates** (mined from a custom hash on a gated op): N/A — no custom hash, and the op is not gated.

- **TTNN factory analysis:** all sheet-derived facts confirmed against the code in the Gate detail section above. Summary of the non-gating facts that inform the port's TTNN ProgramFactory wiring: current concept `descriptor`; **no** op-owned tensors (structurally impossible on a bare `ProgramDescriptor`); no MeshWorkload need (`Execution Model = SPMD`, and the factory returns a single `ProgramDescriptor`); target concept `ProgramSpecFactoryConcept`. The gate conjuncts — custom hash, pybind `create_descriptor`, other risky pybind, `get_dynamic_runtime_args`, `override_runtime_arguments`, genuine multi-program — are all confirmed absent.

## Misc anomalies  *(team-only, non-gating)*

These route to the ops team; the port does not act on them (except where the dead line simply disappears with the code around it).

1. **`c_5` input_transposed CB sizes its allocation with a different tile size than its page size.** [`moe_program_factory.cpp:133-141`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L133-L141) sets `.total_size = Wt * value_tile_size` (`value_tile_size` is `tile_size(Float16_b)` = 2048) while `.page_size = input_tile_size` (`tile_size(input_cb_data_format)`). For a BFLOAT16 input the two are equal and the CB holds the `Wt` pages the compute kernel reserves at [`compute/moe.cpp:277`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L277). For a FLOAT32 input, `input_tile_size` is 4096, so the allocation covers only `Wt/2` pages while compute still reserves `Wt` — a latent mismatch. Reachable only via an undocumented dtype: the nanobind docs restrict inputs to BFLOAT16, but `validate_on_program_cache_miss` never checks dtype, and the factory *does* contemplate FLOAT32 at [:35-36](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L35-L36) (`scalar_df`). Either the validation should enforce the documented BFLOAT16-only contract, or `total_size` should use `input_tile_size` to match its own `page_size`. Sibling CBs are internally consistent (`c_7` uses `value_cb_data_format` with `value_tile_size`).
2. **The scale CB index is hardcoded in the writer instead of arriving as a compile-time arg.** [`writer_unary_interleaved.cpp:27`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L27) declares `constexpr uint32_t scale_dfb_index = tt::CBIndex::c_3;`, duplicating the host's choice at [`moe_program_factory.cpp:106`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L106). The compute kernel receives the same index properly, as CTA 3. If the host ever renumbered the CB, the writer would silently fill the wrong buffer. (The port fixes this incidentally — a named DFB binding removes the duplicated literal — but the current state is worth recording.)
3. **Dead local in the writer.** [`writer_unary_interleaved.cpp:24`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L24) computes `const DataFormat data_format = get_dataformat(out_dfb_index);` and never uses it. This is the same line discussed under the Device 2.0 gate; it disappears with the port.
4. **Dead `onetile` constants.** [`reader_create_index_tensor.cpp:59`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp#L59) and [`writer_unary_interleaved.cpp:22`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L22) each declare `constexpr uint32_t onetile = 1;` and never reference it.
5. **Unused include and commented-out call in the compute kernel.** `#include "ckernel_sfpu.h"` at [`compute/moe.cpp:19`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L19) appears to exist only for the commented-out `// sfpu::_init_sfpu_config_reg();` at [:435](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L435). `#include "api/debug/dprint.h"` at [:18](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp#L18) is likewise unused — no `DPRINT` appears in the file.
6. **Stale comment on the `c_5` allocation.** The comment at [`moe_program_factory.cpp:118-119`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L118-L119) ("Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four tiles of space This CB carries the indices that are created in the reader kernel") sits above the `index_cb` block but its first sentence duplicates the `input_cb` comment at [:70-71](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp#L70-L71), and the two sentences are run together without punctuation. Cosmetic.
7. **`k` is validated to a single value yet remains a hashed attribute.** [`moe_device_operation.cpp:31`](ttnn/cpp/ttnn/operations/reduction/moe/device/moe_device_operation.cpp#L31) enforces `args.k == 32` unconditionally, so `MoeParams::k` can only ever take one value while still participating in the default program hash. Harmless today (it is genuinely consumed as a compile-time arg), but it means the `k` parameter exposed through the pybind is effectively fixed.

## Questions for the user

None. Every finding resolved from the code; no site was left ambiguous.

## Recipe notes

1. **The Device 2.0 sanctioned-free-function list reads as exhaustive but probably isn't meant to be.** The Device 2.0 gate's Green bullet names exactly two sanctioned CB-index free functions (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`) and, two bullets later, defines an isolated holdover by a *shape* test that `get_dataformat(cb_id)` also satisfies. This op hits that exact case ([`writer_unary_interleaved.cpp:24`](ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp#L24)), and the two readings give opposite verdicts on a one-line dead statement: "list is exhaustive" → RED the whole op and route it to the Device 2.0 team; "check the current Device 2.0 surface" → GREEN. I took the second, because `CircularBuffer::get_dataformat()` forwards to the free function exactly as the sanctioned `get_tile_size()` does, and 94 sites across already-migrated TTNN kernels use it. **Suggestion:** either name the sanctioned set as a *family* ("tile/format metadata accessors keyed by CB index — `get_tile_size`, `get_dataformat`, and siblings — stay sanctioned; they move onto the object at Metal 2.0 port time under kernel-side whitelist rule 7") or state explicitly that the two-entry list is closed and everything else gates. As written, the boundary is decided by an auditor's judgment on a case the recipe says elsewhere should be mechanical.
2. **A sheet column name in the recipe doesn't match the live sheet.** The the *TTNN factory concept prerequisite* subject of `audit/metal2_audit.md` refers to `Override runtime args method? (PD and legacy)` throughout; the sheet's actual header today is `Override runtime args method?\n(PD only)`. Reference-by-name still resolved it unambiguously, and the recipe's own standing rule says existing column *names* never change — so this looks like recipe drift rather than a sheet rename. Worth correcting in the recipe (and in `ttnn_op_porting_readiness.md`, which carries the same `(PD and legacy)` spelling) so a future auditor doesn't read the mismatch as a broken sheet.
3. **The sheet carries columns the recipe doesn't mention, two of which look audit-relevant.** `Backdoor custom hash (attribute_values / to_hash)`, `Op Classification`, `Execution Model`, `Porting Target`, `Known op issues`, `Pointer patching perf issue?`, and `Formerly custom hashed?` are all present. `Backdoor custom hash` in particular reads like a second custom-hash signal the `Is able to port?` derivation doesn't list, and `Porting Target` duplicates what the the *TTNN porting shape* subject of `audit/metal2_audit.md` subject derives by hand from `Concept` + `Op-owned tensors?`. For this op they all agreed with my own derivation, so nothing turned on it — but the recipe could say whether `Backdoor custom hash` is a conjunct the auditor should also read, and whether `Porting Target` may be used directly instead of re-deriving the target concept. (Curiosity, not friction: this op's `Backdoor custom hash` cell reads `(complete)`, not `yes`/`no`, which the column legend doesn't cover.)
4. **A "no `->address()` anywhere" op sits slightly outside how two subjects are written.** the *Offset base pointers* subject of `audit/metal2_audit.md` and the *TensorParameter analysis* subject of `audit/metal2_audit.md` are both framed around inventorying address RTAs ("you are already scanning address RTAs"). This op has none: all four tensor bases ride the `MeshTensor` overload of `emplace_runtime_args`, which the recipe covers under the `Buffer*`-binding form but only as a *variant* of the address-RTA shape. The classification rule ("classify by what the kernel does with the base") applied cleanly, so this was easy — but a reader working the subjects in order may look for an `->address()` site, find none, and hesitate over whether the subject fires at all. One sentence in the `Buffer*`-binding bullet noting that the `MeshTensor` overload is the same shape (and is now the common form for a recently-PD-migrated op) would settle it.
