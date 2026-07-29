# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/accumulation/ema`

**Device operations and program factories in this directory:**

- **`EmaDeviceOperation`** (`device/ema_device_operation.hpp`, `device/ema_device_operation.cpp`)
  - `EmaProgramFactory` (`device/ema_program_factory.cpp`) — the op's only factory

**Kernels** (all three owned by this op, all three referenced by the factory; no unreferenced kernel files in the directory):

- `kernels/dataflow/ema_reader.cpp` (`ema_program_factory.cpp:143`)
- `kernels/dataflow/ema_writer.cpp` (`ema_program_factory.cpp:153`)
- `kernels/compute/ema_compute.cpp` (`ema_program_factory.cpp:166`)

*Directory-layout note for the porter:* this op's kernels live at `ema/kernels/`, i.e. at the **op root**, not under `ema/device/kernels/` as the sibling `accumulation` op does. Nothing depends on this; it is only a place where a path guess goes wrong.

**Bundling decision — audited alone, not bundled with `AccumulationDeviceOperation`.** The parent directory `reduction/accumulation/` also holds `AccumulationDeviceOperation` (serving `cumsum` / `cumprod`) with its own factory and its own kernels. The two device operations share **no** program factory and **no** kernel `.cpp`; their only common code is the kernel constants header `accumulation/device/kernels/accumulation_common.hpp`, from which the EMA kernels use exactly one constant (`ONE_TILE`). Under the audit's shared-code test that is not a shared porting unit, and the readiness sheet likewise carries them as two separate ops. So this report covers `EmaDeviceOperation` only.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/accumulation/ema` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `EmaDeviceOperation` → `EmaProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three kernels structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`); the only CB-index free function is the sanctioned `get_tile_size(cb_id)` |
| *Prereqs* — Cross-op escapes | Ok — `✓ clean` (framework `api/*` headers + one in-family constants header) |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a literal constant |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none — no address arithmetic reaches a kernel arg (the factory never calls `->address()` at all) |
| *Port work* — Tensor bindings (per binding) | `input` → **Case 1** · `output` → **Case 1** |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — both accessor constructions are two-argument |
| *Port work* — CB endpoints | `c_0` legal 1:1 · `c_1` legal 1:1 · `c_2` **self-loop** |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution — a **self-loop** (one toucher), a **1P+1C assignment** (two touchers), the **multi-binding advanced-option flag** (a census that cannot fit 1P+1C), or a **dead-CB drop** (zero endpoints). This op needs one self-loop and nothing else.

**Config space — a single instantiation.** `EmaProgramFactory::create_descriptor` has **no configuration branch**: one core-range set, one kernel triple, three CBs, no sharded/interleaved fork, no split reader, no multicast. Core count and per-core tile counts vary with input shape and requested grid, but they change only CTA/RTA *values*, never which kernel touches which CB. So every per-`(CB, config)` disposition below is a per-CB disposition, and there is only one instantiation shape to classify.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear:

- Device 2.0 migration is complete for all three kernels this op uses. They are already on `Noc`, `DataflowBuffer`, and `TensorAccessor`, with no legacy addr-gen, no raw `noc_async_*`, and no raw CB index management.
- No Appendix A feature is in use (no `GlobalCircularBuffer`, no non-zero `address_offset`, no `GlobalSemaphore` — this op uses no semaphores at all — no CTA varargs).
- The readiness sheet's `Is able to port?` is `yes` for the op's single factory, and the cheaply-checkable columns all match the code.
- No offset is folded into any device pointer.
- No `TensorAccessor` passes a third (page-size) argument.

Port work is small and mechanical: two Case-1 tensor bindings, one self-looped CB, and the routine spec translation. `METAL2_PORT_BRIEF.md` is written alongside this file.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet (*"Operations analysis"*, fetched fresh this run) carries exactly one row for this op — `Op = reduction/accumulation/ema`, `Device operation = EmaDeviceOperation`, `Factory (variant) = EmaProgramFactory` — with `Is able to port? = yes`. Every conjunct of that verdict is `no`/`yes` as required: `Is safe to port? = yes`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? (PD and legacy) = no`, `Pybind descriptor = no`, `Concept = descriptor`.

  Cross-check against the code — every cheaply-checkable column agrees, no conflict:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` — `device/ema_device_operation.hpp:24-27`, defined at `device/ema_program_factory.cpp:21-196` |
  | `Custom hash` | `no` | no `compute_program_hash` anywhere in the op directory (grep clean); `EmaDeviceOperation` declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` — `device/ema_device_operation.hpp:32-36` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on the device-op (grep clean over the directory) |
  | `Override runtime args method?` | `no` | no `override_runtime_arguments` method (grep clean) |
  | `Pybind descriptor` | `no` | `ema_nanobind.cpp:70-80` binds only the user-facing `ttnn::ema` free function via `ttnn::bind_function<"ema">`; no `nb::class_` of the device op, no `create_descriptor` binding |
  | `Op-owned tensors?` | *(blank = no)* | the factory returns a plain `ProgramDescriptor`, which structurally cannot carry op-owned tensors — `device/ema_program_factory.cpp:74`, `195` |
  | `Factory definition path` / `Declared in` | `…/ema/device/ema_device_operation.hpp` | matches |
  | **Factory-set match** | 1 row | 1 factory in code: `program_factory_t = std::variant<EmaProgramFactory>` — `device/ema_device_operation.hpp:30`. One-to-one, no phantom row, no missing row. |

  Cross-column invariants hold: `get_dynamic_runtime_args = no` on a `descriptor` concept is consistent, and a `descriptor` row with no op-owned tensors is the normal state.

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels the factory instantiates are owned by this op (no donor kernels, no shared-pool instantiation), and all three are structurally Device 2.0:

  | Kernel | Device 2.0 evidence |
  |---|---|
  | `kernels/dataflow/ema_reader.cpp` | `Noc noc;` (`:36`), `DataflowBuffer dfb_src(src_cb_idx)` (`:37`), `TensorAccessor(src_args, src_base_addr)` (`:34`), object-form transfer `noc.async_read(src_accessor, dfb_src, …)` (`:43`), `noc.async_read_barrier()` (`:44`), FIFO via methods `dfb_src.reserve_back` / `push_back` (`:42`, `:45`) |
  | `kernels/dataflow/ema_writer.cpp` | `Noc noc;` (`:36`), `DataflowBuffer dfb_dst(dst_cb_idx)` (`:37`), `TensorAccessor(dst_args, dst_base_addr)` (`:34`), `noc.async_write(dfb_dst, dst_accessor, …)` (`:43`), `noc.async_write_barrier()` (`:44`), `dfb_dst.wait_front` / `pop_front` (`:42`, `:45`) |
  | `kernels/compute/ema_compute.cpp` | three `DataflowBuffer` objects (`:82-84`), all FIFO traffic through their methods (`:102`, `:107`, `:109`, `:113`, `:116`, `:120`, `:122`, `:126`) |

  A negative sweep over `kernels/` returns **zero** hits for any of: `noc_async_read`, `noc_async_write`, `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, `get_write_ptr(` / `get_read_ptr(`, `get_local_cb_interface`, `get_noc_addr`, `get_semaphore`, `noc_semaphore*`, `evil_set_*`, the `CircularBuffer` wrapper, or the older `api/dataflow/circular_buffer.h` include. The includes are the current set: `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` (plus `api/compute/transpose.h`, `api/compute/ema.h` on the compute kernel).

  **Two CB-index free-function shapes appear; neither is a holdover.** Recorded here so nobody re-litigates them:

  | Site | Call | Why it is not a violation |
  |---|---|---|
  | `kernels/dataflow/ema_reader.cpp:30`, `kernels/dataflow/ema_writer.cpp:30` | `get_tile_size(cb_id)` | Explicitly **sanctioned** by the Device 2.0 gate's Green bullet; Device 2.0's own migrated examples use it. (It *is* port work — `DataflowBuffer::get_tile_size()` exists at `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167`, so the port moves the lookup onto the object.) |
  | `kernels/compute/ema_compute.cpp:93`, `:95`, `:104`, `:111`, `:118`, `:124` | `compute_kernel_hw_startup(cb, cb)`, `transpose_init(cb)`, `transpose_tile(cb, …)`, `pack_tile(dst, cb)` | These are **compute-side** LLK APIs, outside the Device 2.0 *data-movement* surface, and `DataflowBuffer` exposes **no** method replacement for any of them (its accessor set is buffer/tile metadata and pointers only — `dataflow_buffer.h:80-316`). The gate's holdover test requires a wrapper-method replacement to exist; none does. |

- **Feature compatibility:** every Appendix A entry, in order. Every entry is UNSUPPORTED, so a per-row status is `N/A` (feature absent) or `RED` (feature in use). This scan is all-`N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include, no `.global_circular_buffer` field on any of the three `CBDescriptor` literals (`ema_program_factory.cpp:92-120`), no `experimental::CreateCircularBuffer(…, global_cb)`, no `remote_index` / `remote_cb_*` / `remote_circular_buffer.h`. Grep over the op directory returns zero hits for all of these. |
  | CBDescriptor `address_offset` (non-zero) | N/A | All three `CBDescriptor` literals set only `total_size`, `core_ranges`, `format_descriptors` (`ema_program_factory.cpp:92-120`) — `address_offset` is left at its default `0`. No `set_address_offset`, no four-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op uses **no semaphores of any kind**: `desc.semaphores` is never populated, and a case-insensitive grep for `semaphore` over the whole op directory (host and kernel) returns zero hits. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `EmaInputs` is a fixed pair — `Tensor input` plus `std::optional<Tensor> optional_output_tensor` (`device/ema_device_operation_types.hpp:20-23`) — no variable-count container. Kernel-level decider absent: every `get_compile_time_arg_val` call uses a **literal** index — `(0)` and the `TensorAccessorArgs<1>()` NTTP base in both dataflow kernels (`ema_reader.cpp:16-17`, `ema_writer.cpp:16-17`), and `(0)`–`(3)` in the compute kernel (`ema_compute.cpp:71-74`). No loop reads a CTA. |

- **CB endpoints (GATE-free):** three CBs, all declared over the single `all_cores` range, so each has one device-side instance per node and the census is identical on every node.

  | CB | Factory name | Kernel name | Touchers on a node | Verdict | Port-time resolution |
  |---|---|---|---|---|---|
  | `c_0` | `src_cb_index` (`ema_program_factory.cpp:78`, CB at `:92-100`) | `src_cb_idx` | 2 — reader is a **locked producer** (`ema_reader.cpp:42`, `:45`); compute is a **locked consumer** (`ema_compute.cpp:102`, `:107`) | **plain 1:1** | bind 1 PRODUCER + 1 CONSUMER; the roles are already fixed by the FIFO ops. No flag, no assignment question. |
  | `c_1` | `dst_cb_index` (`:79`, CB at `:102-110`) | `dst_cb_idx` | 2 — compute is a **locked producer** (`ema_compute.cpp:122`, `:126`); writer is a **locked consumer** (`ema_writer.cpp:42`, `:45`) | **plain 1:1** | same — bind 1 PRODUCER + 1 CONSUMER. |
  | `c_2` | `prev_cb_index` (`:80`, CB at `:112-120`) | `trp_cb_idx` | **1** — only `ema_compute.cpp` touches it, and it drives **both** FIFO ends: `reserve_back`/`push_back` at `:109`, `:113` and `wait_front`/`pop_front` at `:116`, `:120` | **single toucher → self-loop** | bind the compute kernel **both PRODUCER and CONSUMER**. Legal on Gen1 (a DFB lowers to a plain circular buffer one RISC can fill and drain). Kernel code untouched; runtime behavior identical. |

  Zero-endpoint check: no CB is dead — every one of the three `buffer_index` values is referenced by at least one kernel, at the `file:line` sites tabulated above.

  Hidden-second-writer hunt (Appendix face (a)): **negative, and positively so.** No kernel in this op calls `get_write_ptr()`, `get_read_ptr()`, or `get_local_cb_interface(...).fifo_*_ptr` on any CB — the negative sweep above covers exactly those names. There is also no semaphore anywhere in the op, so the coordination mechanism a raw co-fill depends on does not exist here. Face (b) (multiple readers) and face (c) (dual-instance work-split) are likewise absent: each of the three `KernelDescriptor`s has a distinct `kernel_source` (`ema_program_factory.cpp:143`, `:153`, `:166`), so no kernel source is instantiated twice, and no CB is borrowed-memory (nothing calls `set_globally_allocated_address`).

  Why `c_2` looks unusual but is not: the compute kernel round-trips a tile through SRAM to get a second transpose — it packs the EMA result into `c_2`, then re-unpacks it from `c_2` to transpose back before packing into `c_1` (`ema_compute.cpp:109-120`). One kernel on both ends of a real FIFO. That is a one-toucher census, so the self-loop applies.

- **Offset base pointers:** **GREEN — no fold exists to split out, and none ever did.** The factory contains **no `->address()` call at all** (grep over the op directory returns zero hits). Both tensor base addresses reach their kernels through the descriptor API's typed binding path instead: `reader_desc.emplace_runtime_args(core, {input, src_start_tile})` and `writer_desc.emplace_runtime_args(core, {output, dst_start_tile})` (`ema_program_factory.cpp:184-185`), where `input` / `output` are `MeshTensor` references (`:23-24`). That resolves to the `std::reference_wrapper<const MeshTensor>` overload of `emplace_runtime_args` (`tt_metal/api/tt-metalium/program_descriptors.hpp:192-194`), which auto-registers a buffer binding the framework patches on cache hits — there is no host-side arithmetic anywhere near the address, so no offset can be folded into it.

  The per-core work offset that *would* be the natural place for such a fold is instead passed as its own scalar in the very next arg position: `src_start_tile` / `dst_start_tile`, accumulated in units of tiles (`ema_program_factory.cpp:186-187`) and used by the kernels purely as a **page index** into the accessor (`ema_reader.cpp:41-43`, `ema_writer.cpp:41-43`). That is the already-split shape the offset-base-pointer gate wants, not a Type 1 or Type 2 fold.

  Reconciliation against the dated triage prior `analyses/2026-07-19_offset_base_pointers.md`: this op is **not** in its tables (a grep for `accumulation` and `reduction/` over that doc returns nothing), and my own scan finds no fold — the *no fold, op not in the tables* outcome. Clean; both address RTAs hand off to TensorParameter analysis below. Type 3 (`address_offset`) is N/A per Appendix A above; Type 4 (`narrow`) does not appear.

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** Both `TensorAccessor` constructions pass exactly **two** arguments: `TensorAccessor(src_args, src_base_addr)` (`ema_reader.cpp:34`) and `TensorAccessor(dst_args, dst_base_addr)` (`ema_writer.cpp:34`). No explicit page size is supplied anywhere, so there is no override to classify, drop, or gate — the accessors already take the implicitly-supplied aligned page size that Metal 2.0 provides. Consistent with the dated triage prior `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, which does not list this op; here the syntactic scan (no third argument present at either site) is conclusive on its own rather than relying on that silence.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **`input`** — **Case 1** (via `TensorAccessor`). Delivered today as arg 0 of the reader's per-core RTAs via the `MeshTensor` binding overload (`ema_program_factory.cpp:184`), read as `src_base_addr` (`ema_reader.cpp:21`) and fed straight into `TensorAccessor(src_args, src_base_addr)` (`:34`); every memory access goes through that accessor (`:43`). Port work: express it as a `TensorParameter` / `TensorBinding`, build `TensorAccessor(tensor::…)` in the kernel, and drop both the address arg and the `TensorAccessorArgs(input).append_to(reader_compile_args)` CTA plumbing (`ema_program_factory.cpp:125`, `ema_reader.cpp:17`).
  - **`output`** — **Case 1** (via `TensorAccessor`). Same shape on the writer side: `ema_program_factory.cpp:185` → `ema_writer.cpp:21` → `:34` → `:43`, with the CTA args at `ema_program_factory.cpp:128` / `ema_writer.cpp:17`.
  - Neither binding is Case 2: no kernel does hand-rolled address arithmetic on a base pointer, so no `get_bank_base_address` bridge is needed. Neither is the silent-wrong RTA-smuggled-address hazard either — the `MeshTensor` binding overload is patched on cache hits — so this is routine port work, not a correctness fix.
  - Op-level roll-up: **⚠ port work** (two Case-1 bindings; no clean-by-borrowed-DFB bindings, since the op has no borrowed-memory CB).
- **TensorParameter relaxation:** none. The sheet's `TensorParameter relaxation` column reads `none`, consistent with the op having no custom hash to relax against.
- **TensorAccessor 3rd arg:** none — neither accessor passes one.
- **CB endpoints:** self-loop `c_2` (`prev_cb_index`, the transpose round-trip buffer; single instantiation shape). `c_0` and `c_1` are already legal 1:1. No multi-binding flag anywhere, no dead-CB drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The hidden-second-writer, multiple-reader, and dual-instance-work-split faces were all hunted and all came back negative, with the negative evidence recorded in Gate detail above (no raw pointer access, no semaphores, three distinct kernel sources).
- **Cross-op / shared kernels:** none to coordinate. The op owns all three kernel files and is their **only** binder — a filename census (`grep -rl ema_reader.cpp ttnn/cpp/ttnn/operations/`, same for `ema_writer.cpp` and `ema_compute.cpp`) returns exactly one hit each, `ema_program_factory.cpp`. No `_metal2` fork exists beside any of them, and none is needed: with a single binder the port converts the kernels **in place**, and the fork convention does not apply. There is also no `_metal2` file anywhere under `ttnn/cpp/ttnn/operations/reduction/`.
- **RTA varargs:** none. Each dataflow kernel reads exactly two runtime args at **literal** indices — `get_arg_val<uint32_t>(0)` and `(1)` (`ema_reader.cpp:21-22`, `ema_writer.cpp:21-22`) — and the compute kernel reads none. No counted loop over args, no data-selected index. Both args in each kernel are nameable distinct fields (`src_base_addr` / `src_start_tile`, `dst_base_addr` / `dst_start_tile`), and the address one disappears entirely into the tensor binding.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: `✓ clean`.** No donor needs work, nothing sequence-blocks, and the port crosses no op boundary.

- Every escape but one resolves into `tt_metal/hw/inc/api/*` — framework LLK / HAL surface (donor class 1, "no concern"), and explicitly excluded from the shared-kernel convention.
- The single non-framework escape is an **in-family constants header** carrying no resource handles in any signature.
- **No file-path kernel instantiation from outside the op**, and no other op instantiates this op's kernels — the file-path coupling that normally survives a clean function-call roll-up is absent here too.

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `ema_reader.cpp` | `api/dataflow/dataflow_api.h` | 1 — `tt_metal/*` LLK/HAL | ✓ |
| `ema_reader.cpp` | `api/dataflow/noc.h` | 1 — `tt_metal/*` | ✓ |
| `ema_reader.cpp` | `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` | ✓ |
| `ema_reader.cpp` | `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ |
| `ema_reader.cpp` | `../../../device/kernels/accumulation_common.hpp` | 5 — in-family shared | ✓ |
| `ema_writer.cpp` | `api/dataflow/dataflow_api.h` | 1 — `tt_metal/*` | ✓ |
| `ema_writer.cpp` | `api/dataflow/noc.h` | 1 — `tt_metal/*` | ✓ |
| `ema_writer.cpp` | `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` | ✓ |
| `ema_writer.cpp` | `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ |
| `ema_writer.cpp` | `../../../device/kernels/accumulation_common.hpp` | 5 — in-family shared | ✓ |
| `ema_compute.cpp` | `api/compute/transpose.h` | 1 — `tt_metal/*` | ✓ |
| `ema_compute.cpp` | `api/compute/ema.h` | 1 — `tt_metal/*` | ✓ |
| `ema_compute.cpp` | `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` | ✓ |
| `ema_compute.cpp` | `../../../device/kernels/accumulation_common.hpp` | 5 — in-family shared | ✓ |

**Per-call detail:** omitted — every roll-up is `✓`. The one entry worth a sentence anyway, since a reader may wonder what an in-family escape costs here: `accumulation/device/kernels/accumulation_common.hpp` declares only `constexpr uint32_t` constants (`ONE_TILE`, `FIRST_TILE`, `WORKING_REG`, `CB_IN`, `CB_OUT`, `CB_ACC`) and one `FORCE_INLINE uint32_t get_tile_id(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)`. No `Semaphore`, no `CircularBuffer&`, no `TensorAccessor` / `TensorAccessorArgs` / addr-gen, no CB id in any signature — so no named Metal 2.0 handle ever has to bridge into it. The EMA kernels call **no** function from it; they use only `ONE_TILE`.

**Borrowed kernel files (file-path kernel instantiation):** none. All three `KernelDescriptor::kernel_source` paths point inside this op's own directory (`ema_program_factory.cpp:143`, `:153`, `:166`).

### Relaxation candidates

None. There is no custom hash to mine, and the sheet proposes no relaxation.

### TTNN factory analysis

Sheet-derived facts with `file:line` evidence, for the port's TTNN ProgramFactory wiring:

- **Current concept:** `descriptor` — `EmaProgramFactory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor` (`device/ema_device_operation.hpp:24-27`; body `device/ema_program_factory.cpp:21-196`, returning the local `ProgramDescriptor desc` built at `:74` and returned at `:195`).
- **Op-owned tensors:** none — a `ProgramDescriptor` has no `buffers` vector to carry them, and the op allocates its output through the ordinary TTNN path (`create_output_tensors` → `create_device_tensor`, `device/ema_device_operation.cpp:91-97`).
- **MeshWorkload need:** none — single-program `descriptor` concept, no `WorkloadDescriptor`, so the *secretly SPMD* question does not arise.
- **Pybind `create_descriptor`:** absent. `ema_nanobind.cpp:70-80` binds only `ttnn::ema` (the user-facing composite at `ema.cpp:11-29`); no device-op or factory internals are exposed.
- **Other risky pybind:** none — the binding surface is the documented user API only (`input_tensor`, `alpha`, `out`, `core_grid`, `memory_config`, `compute_kernel_config`).
- **Custom hash:** absent — no `compute_program_hash` override, so the framework default hash over `EmaParams` + tensor specs applies. `EmaParams` is `{alpha, grid_size, output_mem_config, compute_kernel_config}` (`device/ema_device_operation_types.hpp:13-18`); all four genuinely affect the program (`alpha` → the `alpha_bits`/`beta_bits` CTAs at `ema_program_factory.cpp:69-70`, `grid_size` → the core split at `:30-52`, `compute_kernel_config` → the compute config at `:162-175`), so the default hash is correct here.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent.
- **Target concept:** `MetalV2FactoryConcept`, without op-owned tensors.

## Misc anomalies  *(team-only, non-gating)*

Noticed while auditing; these route to the ops team and the port does **not** act on them.

1. **`c_2`'s host-side name describes a role it does not have.** The factory calls it `prev_cb_index` with size `prev_cb_size` (`ema_program_factory.cpp:80`, `:90`), implying it holds the previous EMA output. It does not: the previous output lives in an SFPU register (LREG4), reset by `ema_clear_previous_output()` (`ema_compute.cpp:99` → `tt_metal/hw/inc/api/compute/ema.h:18` → `_clear_previous_output_` in `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_ema.h`). The CB's actual job is staging one tile through SRAM for the second transpose, which is what the kernel's own name for it — `trp_cb_idx` (`ema_compute.cpp:80`) — says. The host and kernel names for the same buffer disagree, and the host one is the misleading of the two.
2. **Compute CTA 0 is named for batches but counts batch×channel-tile rows.** The kernel reads it as `total_batches_per_core` (`ema_compute.cpp:71`), but the host passes `total_batch_channel_tiles_per_core` (`ema_program_factory.cpp:131`), which comes from splitting `num_batches * num_channel_tiles` across cores (`:45`, `:48-49`). The loop is correct; only the name is off by the channel-tile factor, which makes the kernel harder to read than it needs to be.
3. **The docstring's stated initial condition does not match what the SFPU computes.** `ema_nanobind.cpp:27-29` documents `output_t = α·output_{t−1} + (1−α)·input_t` "with output_0 = input_0". The kernel starts each sequence by zeroing the previous-output register (`ema_compute.cpp:99`, resolving to `_clear_previous_output_` → `TTI_SFPLOADI(LREG4, …, 0)` in `ckernel_sfpu_ema.h`), and the recurrence is evaluated unconditionally (`_compute_ema_math_` in the same file), so the first output is `(1−α)·input_0`, not `input_0`. Either the docstring's initial condition or the first-sample handling is wrong; deciding which is the op owner's call.
4. **The docstring restricts memory support to interleaved, but nothing enforces it.** `ema_nanobind.cpp:60-61` says "Memory Support: — Interleaved: DRAM and L1", yet `validate_on_program_cache_miss` only checks that a sharded output's shard grid fits the device grid (`device/ema_device_operation.cpp:62-68`, via `ReduceOpDeviceGridValidationOptions::shard_grid_contained_in_device_grid`). A sharded input or output would be accepted and would run — the kernels go entirely through `TensorAccessor`, so they would address it correctly — which makes the documented limitation either stale or an unenforced intent. *(This does not change any finding above: the factory has no config branch, so a sharded tensor alters only the `TensorAccessorArgs` the accessor is built from, never the CB census.)*
5. **`alpha` is validated for NaN only.** `device/ema_device_operation.cpp:60` rejects NaN but accepts infinities and any finite value outside `[0, 1]`, while the docstring calls α "the smoothing factor, typically between 0 and 1" (`ema_nanobind.cpp:33`). Plausibly deliberate ("typically"), noted in case it is not.
6. **Unused constants pulled in by the shared kernel header.** `accumulation/device/kernels/accumulation_common.hpp` defines `CB_IN` / `CB_OUT` / `CB_ACC` (= `c_0` / `c_1` / `c_2`), `FIRST_TILE`, `WORKING_REG`, and `get_tile_id(...)`; the EMA kernels include it (`ema_reader.cpp:11`, `ema_writer.cpp:11`, `ema_compute.cpp:9`) but use only `ONE_TILE`, declaring their own CB indices instead (`ema_reader.cpp:26`, `ema_writer.cpp:26`, `ema_compute.cpp:78-80`). So each EMA kernel has two sets of CB-index constants in scope naming the same three buffers — harmless today, an easy way to introduce a mix-up later.

## Questions for the user

None. Every check resolved on code evidence; nothing needed a conservative default.

## Recipe notes

Four places where the recipe left me to make a call it could make for the next auditor. All four resolved cleanly here, but each cost a detour.

1. **The Device 2.0 gate says nothing about *compute-side* CB-index free functions.** the audit recipe's *Device 2.0 prerequisite* subject (`audit/metal2_audit.md`) sanctions `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` by name, and defines a holdover as "free functions taking a `uint32_t` CB index where the corresponding Device-2.0 wrapper object is already in scope at the call site *and* a wrapper-method replacement exists." This op's compute kernel has a `DataflowBuffer` in scope and still calls `compute_kernel_hw_startup(cb, cb)`, `transpose_init(cb)`, `transpose_tile(cb, …)`, and `pack_tile(dst, cb)` (`ema_compute.cpp:93-124`) — shape-identical to the holdover cue. I cleared them on the second half of the test (no wrapper-method replacement exists) plus the guide's own data-movement scoping, but that reasoning took reading `dataflow_buffer.h`'s full method list to confirm a negative. One sentence in the Green bullet — *compute-side LLK APIs taking a CB index (`pack_tile`, `transpose_tile`, `compute_kernel_hw_startup`, the `*_init` family) are outside the Device 2.0 data-movement surface and are never holdovers* — would settle it up front. Every compute kernel in the fleet will hit this.

2. **The recipe's kernel-side Device 2.0 vocabulary is `CircularBuffer`; current kernels use `DataflowBuffer`.** The gate's own wording ("kernel-side `CircularBuffer` wrappers"), the donor-shape table's `CircularBuffer&` row, and the linked Device 2.0 migration guide (`kernel_apis/data_movement/device_api_migration_guide.md`) (which documents `api/dataflow/circular_buffer.h` and `CircularBuffer cb(cb_id)`) all name `CircularBuffer`. These kernels use `DataflowBuffer` from `api/dataflow/dataflow_buffer.h` — they were converted by `bed70038e18 [Cleanup] Migrate MM/Fused/Reduce Kernels from CircularBuffer to DataflowBuffer (#49173)`. Since the audit *also* teaches that `DataflowBuffer` is the Metal 2.0 spec-layer replacement for the legacy CB, a first-time auditor can read a kernel-side `DataflowBuffer` as "not the Device 2.0 object" and go looking for a gate. It is the opposite — a strictly more migrated state. A line in the the audit recipe's *Read this first* orientation bullets noting that the kernel-side wrapper now ships under both names, `DataflowBuffer` being the current one, would prevent that.

3. **The audit recipe's *TensorParameter analysis* subject's "`Buffer*`-binding form" bullet does not mention the `MeshTensor` overload it also covers.** The bullet is written entirely in terms of "the factory pushes a `Buffer*` … into `KernelDescriptor::RTArgList` / `emplace_runtime_args`". This factory pushes a **`MeshTensor` reference** instead (`ema_program_factory.cpp:184-185`), which lands on a different overload — `std::reference_wrapper<const MeshTensor>` (`program_descriptors.hpp:192-194`, `:199`, `:207`) — with the same auto-registered, patched-on-cache-hit semantics (the header's own comment at `:161-163` says "Buffer/MeshTensor args"). I matched it to the bullet by reading the overload set, but the bullet as written describes a token this factory never uses, and `->address()`-oriented greps do not surface it at all: **there is no `address()` call anywhere in this op**, so an auditor grepping only for the documented shapes could conclude "no address RTA exists" and skip both bindings. Naming the `MeshTensor` overload in the bullet would close that.

4. **The one-toucher row's label doesn't fit a one-kernel FIFO round-trip.** The The audit recipe's *CB endpoints* table calls the 1-toucher case "**single-ended / sync-free**", and the prose elaborates it as "one real endpoint, or pointer-only access by that one kernel". This op's `c_2` is neither single-ended nor sync-free: the compute kernel drives a **complete, genuinely synchronizing** FIFO cycle on it — `reserve_back` → `push_back` → `wait_front` → `pop_front` (`ema_compute.cpp:109-120`) — to round-trip a tile through SRAM for a second transpose. It is a locked producer *and* a locked consumer, but only **one** toucher, so the census rule and the self-loop resolution both apply exactly as written; only the label misdescribes it. Worth widening to something like "one toucher (single-ended, sync-free, or one kernel on both FIFO ends)", since a hurried reader could otherwise mis-slot a pack/unpack staging buffer as two touchers and reach for a 1P+1C assignment that has no second kernel to assign.
