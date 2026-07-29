# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/accumulation`

Two device operations share this directory tree and are audited together (they share the kernel
header `device/kernels/accumulation_common.hpp`, which all six kernel files include):

- **`AccumulationDeviceOperation`** (`device/`) — backs the public `cumsum` and `cumprod`
  - `AccumulationProgramFactory` (`device/accumulation_program_factory.cpp`)
- **`EmaDeviceOperation`** (`ema/device/`) — backs the public `ema`
  - `EmaProgramFactory` (`ema/device/ema_program_factory.cpp`)

Each device operation declares exactly one factory (`program_factory_t` is a single-alternative
`std::variant` in both cases), so there are two factories in total.

Kernel files (all six are referenced by a factory; none are unreferenced/dead files):

| Kernel | Bound by |
|---|---|
| `device/kernels/dataflow/accumulation_reader.cpp` | `AccumulationProgramFactory` |
| `device/kernels/compute/accumulation_compute.cpp` | `AccumulationProgramFactory` (two descriptors, disjoint core groups) |
| `device/kernels/dataflow/accumulation_writer.cpp` | `AccumulationProgramFactory` |
| `ema/kernels/dataflow/ema_reader.cpp` | `EmaProgramFactory` |
| `ema/kernels/compute/ema_compute.cpp` | `EmaProgramFactory` |
| `ema/kernels/dataflow/ema_writer.cpp` | `EmaProgramFactory` |
| `device/kernels/accumulation_common.hpp` | header, included by all six above |

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/accumulation` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `AccumulationDeviceOperation` → `AccumulationProgramFactory` · `EmaDeviceOperation` → `EmaProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all six kernels structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`); the only CB-index free function in use is the sanctioned `get_tile_size(cb_id)` |
| *Prereqs* — Cross-op escapes | **Ok** — every out-of-directory kernel include resolves under `tt_metal/hw/inc/api/` (donor class 1); no borrowed kernel files, no donor function calls |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a literal constant; no variable-count tensor container |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (both) |
| *TTNN Readiness* — Custom hash | No (both) — no `compute_program_hash` anywhere in the directory |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (both) |
| *TTNN Readiness* — `override_runtime_arguments` | No (both) |
| *TTNN Readiness* — Pybind `create_descriptor` | No — the three `*_nanobind.cpp` files bind only the public op functions |
| *TTNN Readiness* — Op-owned tensors | No (blank on both rows; consistent with the `descriptor` concept) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none** — no `->address()` expression exists in the directory; every tensor base reaches its kernel as a `MeshTensor` entry in `emplace_runtime_args`, which the framework resolves to a clean `buffer->address()` |
| *Port work* — Tensor bindings (per binding) | 4 bindings, all **Case 1** (`TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none (sheet: `none` on both rows; no custom hash to reconcile) |
| *Port work* — TensorAccessor 3rd arg | none — all four `TensorAccessor` constructions use the two-argument form |
| *Port work* — CB endpoints | 4 × legal 1:1 · 2 × **self-loop** (`CB_ACC` on accumulation compute; `c_2` on EMA compute) |

**CB endpoints** are dispositions, not gates. Both self-loops here are single-*compute*-kernel CBs, so
neither carries Quasar debt (a DM self-loop would; a compute self-loop is legal on Gen1 and Gen2).
Both factories have exactly one configuration path each, so no disposition flips with config — see
[CB endpoints](#cb-endpoints-gate-free) for why.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, beside this file).

All five gates cleared for both factories. This is an unusually clean op for a Metal 2.0 port: its
kernels are already written against the Device 2.0 / `DataflowBuffer` surface, every tensor base
already arrives through the framework's buffer-binding channel rather than a hand-written
`->address()` runtime arg, there are no semaphores, no shared or borrowed kernel files, no sharded
paths, and no Appendix A features. The port work is four `TensorParameter` bindings (all Case 1),
two self-loop CB bindings, and named-argument conversion.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet was fetched fresh
  this run. Both factories appear, one row each, and both read `Is able to port? = yes`:

  | `Op` | `Device operation` | `Factory (variant)` | `Concept` | `Custom hash` | `get_dynamic_runtime_args` | `override_runtime_args` | `Pybind descriptor` | `Is safe to port?` | `Is able to port?` | `TensorParameter relaxation` |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `reduction/accumulation` | `AccumulationDeviceOperation` | `AccumulationProgramFactory` | `descriptor` | no | no | no | no | yes | **yes** | none |
  | `reduction/accumulation/ema` | `EmaDeviceOperation` | `EmaProgramFactory` | `descriptor` | no | no | no | no | yes | **yes** | none |

  Both rows also carry `Smuggled pointer = no` and `Op Classification = PD (pointer-patching)`.

  Cross-check against the code — **no disagreement found**:

  | Column | Code evidence | Verdict |
  |---|---|---|
  | `Concept` = `descriptor` | `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor` at [accumulation_program_factory.cpp:42-45](device/accumulation_program_factory.cpp#L42-L45) and [ema_program_factory.cpp:21-22](ema/device/ema_program_factory.cpp#L21-L22); no mesh-workload return, no `create()`/`override_runtime_arguments()` pair | ✓ |
  | `Custom hash` = no | no `compute_program_hash` (or renamed variant) anywhere under the op directory | ✓ |
  | `get_dynamic_runtime_args` = no | no such hook on either device-op ([accumulation_device_operation.hpp:26-63](device/accumulation_device_operation.hpp#L26-L63), [ema_device_operation.hpp:17-37](ema/device/ema_device_operation.hpp#L17-L37)) | ✓ |
  | `override_runtime_arguments` = no | absent from both device-ops and both factories | ✓ |
  | `Pybind descriptor` = no | `cumsum_nanobind.cpp`, `cumprod_nanobind.cpp`, `ema_nanobind.cpp` bind only the public `ttnn::` op functions via `ttnn::bind_function<...>` — no `nb::class_` of a device-op, no `create_descriptor` binding | ✓ |
  | `Secretly SPMD Workload?` (blank) | N/A — neither factory returns a `WorkloadDescriptor` | ✓ |
  | Factory-set match | two sheet rows ↔ two factories in code; each `program_factory_t` is a single-alternative `std::variant`; neither device-op defines a `select_program_factory` | ✓ no phantom rows, no missing rows |
  | Cross-column invariants | `Op-owned tensors?` blank on both `descriptor` rows (a `descriptor` row with op-owned tensors would be a broken sheet); `get_dynamic_runtime_args = no` is consistent with `descriptor` | ✓ |

  Target concept for both factories: **`MetalV2FactoryConcept`**, no op-owned tensors.

- **Device 2.0 (every kernel used):** **GREEN — no violations.** All six kernels are structurally
  Device 2.0 and in fact already use the Metal 2.0 kernel-side buffer wrapper: they include
  `api/dataflow/noc.h` + `api/dataflow/dataflow_buffer.h` and drive I/O through `Noc` and
  `DataflowBuffer` objects. There are **no** legacy Device 1.0 idioms anywhere in the directory — a
  grep for `noc_async_read`, `noc_async_write`, `noc_semaphore*`, `cb_reserve_back`, `cb_push_back`,
  `cb_wait_front`, `cb_pop_front`, `get_read_ptr`, `get_write_ptr`, `get_local_cb_interface`,
  `evil_set_*`, `get_semaphore`, `get_noc_addr`, `InterleavedAddrGen*`, `ShardedAddrGen`, and
  `CircularBuffer` returns zero hits across all six kernel files.

  The only CB-index-keyed free function in use is `get_tile_size(cb_id)`:

  | File | Line | Call | Wrapper in scope | Disposition |
  |---|---|---|---|---|
  | `device/kernels/dataflow/accumulation_reader.cpp` | 33 | `get_tile_size(CB_IN)` | `dfb_in_obj` | **sanctioned — not a violation** |
  | `device/kernels/dataflow/accumulation_writer.cpp` | 27 | `get_tile_size(CB_OUT)` | `dfb_out_obj` | **sanctioned — not a violation** |
  | `ema/kernels/dataflow/ema_reader.cpp` | 30 | `get_tile_size(src_cb_idx)` | `dfb_src` | **sanctioned — not a violation** |
  | `ema/kernels/dataflow/ema_writer.cpp` | 30 | `get_tile_size(dst_cb_idx)` | `dfb_dst` | **sanctioned — not a violation** |

  `get_tile_size(cb_id)` is kept as a free function by Device 2.0 itself — its migration guide's own
  migrated example still calls it, at
  [device_api_migration_guide.md:630](../../../../../../docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md#L630) —
  so per the Green bullet these are not holdovers and the gate is clean. The Metal 2.0 *port* moves
  them onto the object (`DataflowBuffer::get_tile_size()` exists at
  [dataflow_buffer.h:167](../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L167)); that is
  port work under the kernel-side whitelist, recorded in the brief, not a Device 2.0 finding.

  The compute kernels pass CB indices to compute-side LLK calls (`unary_op_init_common`,
  `copy_tile`, `pack_tile`, `reconfig_data_format`, `pack_reconfig_data_format`, `transpose_tile`,
  `compute_kernel_hw_startup`). These are the normal signatures of the compute API, not
  data-movement free functions with a wrapper-method replacement, so they are outside the Device 2.0
  data-movement boundary and are not flagged.

- **Feature compatibility:** all four Appendix A entries scanned against both factories' host code,
  descriptors, and all six kernels. Nothing fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include, no `.global_circular_buffer` field on either factory's `CBDescriptor` literals ([accumulation_program_factory.cpp:107-115](device/accumulation_program_factory.cpp#L107-L115), [ema_program_factory.cpp:92-120](ema/device/ema_program_factory.cpp#L92-L120)), no `remote_index` / `remote_cb_*` idiom, no `experimental::CreateCircularBuffer` 4-arg form. All six CBs are plain `CBDescriptor`s with `total_size`, `core_ranges`, and one `CBFormatDescriptor` each. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The token `address_offset` does not appear anywhere in the directory; neither factory sets the field, so it defaults to zero. No `set_address_offset`, no `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. No runtime-team consultation needed. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include. Neither factory pushes any `SemaphoreDescriptor` at all — `desc.semaphores` is left empty in both, and no kernel performs any semaphore operation. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | **Op-level signal absent:** neither `tensor_args_t` carries a variable-count container — `AccumulationInputs` is `{const Tensor&, std::optional<Tensor>}` ([accumulation_device_operation_types.hpp:26-29](device/accumulation_device_operation_types.hpp#L26-L29)) and `EmaInputs` is `{Tensor, std::optional<Tensor>}` ([ema_device_operation_types.hpp:20-23](ema/device/ema_device_operation_types.hpp#L20-L23)). **Kernel-level decider absent:** every `get_compile_time_arg_val` call in the directory uses a literal index (`(0)` in `accumulation_compute.cpp`; `(0)` in `ema_reader.cpp` / `ema_writer.cpp`; `(0)`–`(3)` in `ema_compute.cpp`), and the two dataflow readers/writers of the accumulation factory take only `TensorAccessorArgs<0>()`. No CTA is read at a runtime-varying index. |

- **CB endpoints (GATE-free):** every CB either is a legal 1:1 FIFO or takes a **self-loop**. Full
  per-CB, per-node census below. No dead CBs, no 1P+1C assignments needed, no multi-binding flags.

  **`AccumulationProgramFactory`** — CBs declared at
  [accumulation_program_factory.cpp:102-120](device/accumulation_program_factory.cpp#L102-L120), all over
  `all_cores`. The reader and writer are instantiated over `all_cores`; the compute kernel is
  instantiated as **two descriptors over disjoint core ranges** (`core_group_1` at
  [line 190](device/accumulation_program_factory.cpp#L190) and `core_group_2` at
  [line 206](device/accumulation_program_factory.cpp#L206), whose union is `all_cores`), so **each node
  carries exactly one compute instance**. This is the disjoint-node-set shape, not the
  dual-instance work-split — nothing is co-touched.

  | CB | Index | Touchers on a node | Roles | Verdict | Resolution |
  |---|---|---|---|---|---|
  | `SRC` / `CB_IN` | `c_0` (4 tiles) | reader ([accumulation_reader.cpp:43,46](device/kernels/dataflow/accumulation_reader.cpp#L43-L46)), compute ([accumulation_compute.cpp:70,83](device/kernels/compute/accumulation_compute.cpp#L70-L83)) | 1 locked producer + 1 locked consumer | **plain 1:1** | none — legal as-is |
  | `DST` / `CB_OUT` | `c_1` (4 tiles) | compute ([accumulation_compute.cpp:89,92](device/kernels/compute/accumulation_compute.cpp#L89-L92)), writer ([accumulation_writer.cpp:40,44](device/kernels/dataflow/accumulation_writer.cpp#L40-L44)) | 1 locked producer + 1 locked consumer | **plain 1:1** | none — legal as-is |
  | `ACC` / `CB_ACC` | `c_2` (1 tile) | compute only ([accumulation_compute.cpp:37-38,43-44,57,63,81,94,101,106-107](device/kernels/compute/accumulation_compute.cpp#L37-L107)) | same kernel FIFO-produces **and** FIFO-consumes | **single-ended → self-loop** | bind the compute kernel PRODUCER **and** CONSUMER |

  `CB_ACC` is a *genuine-FIFO* self-loop, not a sync-free scratchpad: the compute kernel uses the
  CB's FIFO to sequence its own unpacker against its own packer (the kernel says so at
  [accumulation_compute.cpp:41-42](device/kernels/compute/accumulation_compute.cpp#L41-L42) — *"Synchronize
  unpacker-packer between iterations / This is necessary to avoid data-races on cb_acc"*), with the
  1-tile depth deliberately making every `reserve_back` land on the same address
  ([line 56](device/kernels/compute/accumulation_compute.cpp#L56)). Both FIFO paths must stay
  functional, which is exactly what a PRODUCER+CONSUMER self-loop preserves.

  **`EmaProgramFactory`** — CBs declared at
  [ema_program_factory.cpp:92-120](ema/device/ema_program_factory.cpp#L92-L120), all over `all_cores`;
  reader, writer, and compute are each one instance over `all_cores`.

  | CB | Index | Touchers on a node | Roles | Verdict | Resolution |
  |---|---|---|---|---|---|
  | `src_cb` | `c_0` (2 tiles) | reader ([ema_reader.cpp:42,45](ema/kernels/dataflow/ema_reader.cpp#L42-L45)), compute ([ema_compute.cpp:102,107](ema/kernels/compute/ema_compute.cpp#L102-L107)) | 1 locked producer + 1 locked consumer | **plain 1:1** | none — legal as-is |
  | `dst_cb` | `c_1` (2 tiles) | compute ([ema_compute.cpp:122,126](ema/kernels/compute/ema_compute.cpp#L122-L126)), writer ([ema_writer.cpp:42,45](ema/kernels/dataflow/ema_writer.cpp#L42-L45)) | 1 locked producer + 1 locked consumer | **plain 1:1** | none — legal as-is |
  | `prev_cb` (kernel: `trp_cb_idx`) | `c_2` (1 tile) | compute only ([ema_compute.cpp:109,113,116,120](ema/kernels/compute/ema_compute.cpp#L109-L120)) | same kernel FIFO-produces **and** FIFO-consumes | **single-ended → self-loop** | bind the compute kernel PRODUCER **and** CONSUMER |

  `c_2` in EMA is also a genuine-FIFO self-loop: the compute kernel packs the transposed tile out to
  it and immediately reads it back to transpose again, a packer→unpacker round trip inside one
  kernel.

  **Why no disposition flips with config.** Each factory has a single code path:
  - `AccumulationProgramFactory` — the only branching is (a) `core_group_2` empty or not, which adds
    or removes a *disjoint* compute descriptor without changing any node's toucher count, and (b)
    integer vs floating-point data format, which only swaps `defines` strings
    ([lines 133-147](device/accumulation_program_factory.cpp#L133-L147)). Sharded and non-tile inputs are
    rejected in validation ([accumulation_device_operation.cpp:44-57](device/accumulation_device_operation.cpp#L44-L57)),
    so there is no sharded variant to census.
  - `EmaProgramFactory` — the only branching is the default-vs-supplied core grid
    ([lines 30-33](ema/device/ema_program_factory.cpp#L30-L33)), which changes the core count, not the
    per-node census.

  No `(CB, config)` pair therefore has a second disposition.

- **Offset base pointers:** **GREEN — nothing to split out.** There is **no `->address()` expression
  anywhere in the op directory** (host or kernel), so there is no site at which a host-side offset
  could be folded into a device pointer. Every tensor base instead reaches its kernel by pushing the
  `MeshTensor` itself into `KernelDescriptor::emplace_runtime_args`:

  - [accumulation_program_factory.cpp:231-251](device/accumulation_program_factory.cpp#L231-L251) — `input_tensor` at reader arg 0, `output_tensor` at writer arg 0.
  - [ema_program_factory.cpp:184-185](ema/device/ema_program_factory.cpp#L184-L185) — `input` at reader arg 0, `output` at writer arg 0.

  The framework resolves that entry to `ref.get().mesh_buffer().get_reference_buffer()->address()`
  and registers a `BufferBinding` at that arg index
  ([program_descriptors.cpp:232-236](../../../../../../tt_metal/impl/program/program_descriptors.cpp#L232-L236)) —
  a **clean base, with no arithmetic applied**. Type 1 and Type 2 are structurally impossible here.
  Type 3 is N/A (no `address_offset`; see the Appendix A row above). Type 4 is N/A (no
  `ttnn::narrow`, no `MeshBuffer::create` interior-base view). Neither device operation appears in
  the dated offset-base-pointer triage analysis, and my own scan agrees with that silence.

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** All four `TensorAccessor`
  constructions in the directory use the **two-argument** form, with no page-size override to
  classify or drop:

  | Site | Construction |
  |---|---|
  | [accumulation_reader.cpp:36](device/kernels/dataflow/accumulation_reader.cpp#L36) | `TensorAccessor(input_addrg_args, input_base_addr)` |
  | [accumulation_writer.cpp:30](device/kernels/dataflow/accumulation_writer.cpp#L30) | `TensorAccessor(output_addrg_args, output_base_addr)` |
  | [ema_reader.cpp:34](ema/kernels/dataflow/ema_reader.cpp#L34) | `TensorAccessor(src_args, src_base_addr)` |
  | [ema_writer.cpp:34](ema/kernels/dataflow/ema_writer.cpp#L34) | `TensorAccessor(dst_args, dst_base_addr)` |

  Neither device operation appears in the dated 3rd-arg triage analysis, and my own read agrees —
  there is no site to classify, so no Class 1/2 drop and no Class 3/4/Special gate. (Note the
  kernels *do* compute a tile size via `get_tile_size(cb_id)` and use it as the NoC transfer size —
  `input_tile_bytes` / `output_tile_bytes` / `src_tile_size` / `dst_tile_size` — but that value is
  passed to `noc.async_read` / `noc.async_write` as the byte count, **not** to the accessor
  constructor. It is not a page-size override.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — 4 total, all **Case 1** (`TensorAccessor`):
  - `AccumulationProgramFactory` / reader input — **Case 1**. Base at reader RTA index 0, fed to `TensorAccessor(input_addrg_args, input_base_addr)` at [accumulation_reader.cpp:36](device/kernels/dataflow/accumulation_reader.cpp#L36); all addressing goes through `noc.async_read(input_addrg, …, {.page_id = read_tile_id}, …)` at [line 44](device/kernels/dataflow/accumulation_reader.cpp#L44). The accessor CTAs come from `TensorAccessorArgs(input_tensor).append_to(reader_compile_time_args)` at [accumulation_program_factory.cpp:165-166](device/accumulation_program_factory.cpp#L165-L166) and disappear with the port.
  - `AccumulationProgramFactory` / writer output — **Case 1**. Base at writer RTA index 0 → `TensorAccessor` at [accumulation_writer.cpp:30](device/kernels/dataflow/accumulation_writer.cpp#L30); accessor CTAs from [accumulation_program_factory.cpp:168-169](device/accumulation_program_factory.cpp#L168-L169).
  - `EmaProgramFactory` / reader input — **Case 1**. Base at reader RTA index 0 → `TensorAccessor` at [ema_reader.cpp:34](ema/kernels/dataflow/ema_reader.cpp#L34); accessor CTAs from [ema_program_factory.cpp:124-125](ema/device/ema_program_factory.cpp#L124-L125) (note they start at CTA slot 1, behind `total_tiles_per_core`, hence the kernel's `TensorAccessorArgs<1>()`).
  - `EmaProgramFactory` / writer output — **Case 1**. Base at writer RTA index 0 → `TensorAccessor` at [ema_writer.cpp:34](ema/kernels/dataflow/ema_writer.cpp#L34); accessor CTAs from [ema_program_factory.cpp:127-128](ema/device/ema_program_factory.cpp#L127-L128).

  All four arrive by the **`MeshTensor`-binding delivery form** (`emplace_runtime_args` with the
  tensor object, which the framework turns into a `BufferBinding` it patches on cache hits) rather
  than a hand-written `buffer()->address()` RTA. That means **none of them is the silent-wrong
  stale-pointer hazard** — they are correct on cache hits today. They still all need expressing as
  `TensorParameter` / `TensorBinding` in the port, which is what supersedes the interim binding
  mechanism. No Case 2 (raw-pointer) binding exists; no `get_bank_base_address` bridge is needed.
  No borrowed-memory DFB reads exist either (no `set_globally_allocated_address` anywhere), so no
  binding is *clean* via the causal-link gate.
- **TensorParameter relaxation:** none. The sheet proposes `none` for both factories, and there is
  no custom hash to reconcile a relaxation against.
- **TensorAccessor 3rd arg:** none — no site passes one.
- **CB endpoints:** self-loop `(ACC/CB_ACC c_2, AccumulationProgramFactory — all configs)` and
  `(prev_cb c_2, EmaProgramFactory — all configs)`; the remaining four CBs are legal 1:1 as-is. No
  1P+1C assignment, no multi-binding flag, no dead-CB drop.
- **`get_tile_size(cb_id)` → DFB method:** four sites (table in the Device 2.0 bullet above) move
  onto the `DataflowBuffer` object under kernel-side whitelist rule 7.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** No CB in either factory has a hidden
  second writer or a second reader. Specifically ruled out: no kernel in the directory calls
  `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface` / `evil_set_*` at all (grep is empty),
  and no semaphores exist to coordinate a raw co-fill — so the hidden-second-writer face cannot be
  present. Every CB access in the op is a FIFO operation on a `DataflowBuffer` object.
- **Two same-source compute descriptors over disjoint core ranges (accumulation).** The factory
  pushes `accumulation_compute.cpp` into two `KernelDescriptor`s
  ([accumulation_program_factory.cpp:187-217](device/accumulation_program_factory.cpp#L187-L217)) that differ
  **only** by `core_ranges` (`core_group_1` vs `core_group_2`). Two things follow: the pair is the
  *disjoint-node-set* shape, so each node still has one compute instance and the CB census stays
  1P+1C (it is **not** the dual-instance work-split, which would co-touch CBs on every node); and
  the two descriptors are otherwise byte-identical, so merging them would be behaviour-preserving —
  but the port makes no functional changes, so keep two `KernelSpec`s mirroring the legacy shape.
- **Cross-op / shared kernels:** no borrowed kernel files and no donor function calls — see the
  Team-only inventory. The one intra-directory sharing point is the header
  `device/kernels/accumulation_common.hpp`, included by all six kernels *across both device
  operations*; the EMA kernels consume only `ONE_TILE` from it. Because both device operations are
  in the same port unit, edits to that header are in scope, but don't leave one device operation's
  kernels half-converted against it. No `_metal2` fork exists anywhere under
  `ttnn/cpp/ttnn/operations/reduction/`, and there is no `experimental/quasar` copy of this op — so
  the porter has no fork to reuse and nothing to be tempted by.
- **RTA varargs:** **none.** Every runtime-arg read in every kernel is at a literal constant index
  (reader/writer args 0–7 in the accumulation factory, 0–1 in EMA, 0–1 in accumulation compute);
  there is no counted loop over `get_arg_val` / `get_common_arg_val` and no data-selected index. Every
  runtime arg is nameable. No `common_runtime_args` are used by either factory.

## Team-only

- **Out-of-directory coupling & donor shape:** **op-level roll-up `✓ clean`** — nothing to
  coordinate with another team.
  - **Function-call escape:** every `#include` in every kernel file resolves either inside the op
    directory or under `tt_metal/hw/inc/api/` (donor class 1 — LLK / HAL / firmware, no concern):
    `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`,
    `api/core_local_mem.h`, `api/tensor/noc_traits.h`, and the `api/compute/*` set. There are **no**
    includes from `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`,
    `ttnn/cpp/ttnn/operations/kernel_helper_functions/`, another op family, or another op in the
    `reduction` family. The only non-`api/` include is the op's own
    `device/kernels/accumulation_common.hpp`, whose single function `get_tile_id(uint32_t × 5)`
    takes plain scalars — no CB handle, no semaphore, no accessor — so no boundary shape needs
    bridging. The summary table and per-call detail are omitted: all rolls are `✓`.
  - **Borrowed kernel files (file-path instantiation):** **none.** The op owns all six kernel
    sources it instantiates. A repo-wide grep for the six filenames and for
    `accumulation_common.hpp` finds no consumer outside this directory, so nothing here is *lent*
    either — no sunset list, no cross-op coordination cost, and the shared-kernel fork rungs do not
    apply to this port.
- **Relaxation candidates** (mined from a custom hash on a gated op): **N/A** — the op has no custom
  hash and is not gated.
- **TTNN factory analysis:** sheet-derived facts with code evidence are in the
  [TTNN factory concept gate detail](#gate-detail) above. Summary of the non-gating facts that feed
  the port's TTNN ProgramFactory wiring: no op-owned tensors (both factories are plain `descriptor`;
  neither returns a `WorkloadDescriptor`, so there is no MeshWorkload need — genuine or artifact);
  target concept `MetalV2FactoryConcept` for both. Gate conjuncts confirmed absent in code: custom
  hash, `get_dynamic_runtime_args`, `override_runtime_arguments`, pybind `create_descriptor`. No
  other migration-risky pybind: the three nanobind files expose only the public op functions.
  - One wiring detail the porter needs: the output tensor a factory binds is always
    `tensor_return_value` (`.mesh_tensor()`), never `tensor_args`' optional output directly. Both
    device operations funnel the preallocated-output case through `create_output_tensors`, which
    returns the caller's tensor when present ([accumulation_device_operation.cpp:95-103](device/accumulation_device_operation.cpp#L95-L103),
    [ema_device_operation.cpp:91-97](ema/device/ema_device_operation.cpp#L91-L97)). So each factory has
    exactly **one input and one output** `TensorParameter`, with no optional-tensor binding.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

1. **Dead runtime arg — `start_id` in both accumulation dataflow kernels.** The reader takes
   `start_id` at RTA index 4 ([accumulation_reader.cpp:22](device/kernels/dataflow/accumulation_reader.cpp#L22))
   and uses it only as the initial value of a loop counter whose variable `i` is never referenced in
   the body ([lines 38-53](device/kernels/dataflow/accumulation_reader.cpp#L38-L53)) — the body addresses
   tiles purely from `low_rank_offset` / `high_rank_offset`. The loop is therefore equivalent to
   `for (i = 0; i < num_rows_per_core; ++i)`, and the *value* the host sends (`tile_offset`, at
   [accumulation_program_factory.cpp:237](device/accumulation_program_factory.cpp#L237)) is dead. Identical
   in the writer ([accumulation_writer.cpp:19,35-51](device/kernels/dataflow/accumulation_writer.cpp#L19-L51),
   fed from [accumulation_program_factory.cpp:248](device/accumulation_program_factory.cpp#L248)). The two
   arguments that *do* carry the per-core start position are indices 5 and 6
   (`tile_offset / input_tile_offset` and `tile_offset % input_tile_offset`).
2. **Unused constants in a shared kernel header.** `FIRST_TILE` and `WORKING_REG` at
   [accumulation_common.hpp:8-9](device/kernels/accumulation_common.hpp#L8-L9) are referenced by no
   kernel in the directory. (`ONE_TILE` and the three `CB_*` constants are live.)
3. **Unused includes in the accumulation reader.** `<cstring>`
   ([line 5](device/kernels/dataflow/accumulation_reader.cpp#L5)) and `api/core_local_mem.h`
   ([line 10](device/kernels/dataflow/accumulation_reader.cpp#L10)) are not used — no `memcpy`, no
   `CoreLocalMem`. The sibling writer includes neither.
4. **`AccumulationInputs::opt_output` is unreachable in practice.** The only caller of
   `ttnn::prim::accumulation` passes `std::nullopt` for it
   ([accumulation_common.cpp:127-134](accumulation_common.cpp#L127-L134)); the preallocated-output case is
   instead handled after the fact by grafting storage onto the caller's tensor
   ([accumulation_common.cpp:136-141](accumulation_common.cpp#L136-L141), which carries its own TODO
   #37807). So the three `opt_output` branches in the device operation — validation
   ([accumulation_device_operation.cpp:22-31](device/accumulation_device_operation.cpp#L22-L31)),
   `compute_output_specs` ([lines 77-79](device/accumulation_device_operation.cpp#L77-L79)), and
   `create_output_tensors` ([lines 97-100](device/accumulation_device_operation.cpp#L97-L100)) — are dead
   today. Contrast EMA, which does thread its optional output through
   ([ema.cpp:27-28](ema/ema.cpp#L27-L28)).
5. **Redundant core-group split of the accumulation compute kernel.** The two compute
   `KernelDescriptor`s at
   [accumulation_program_factory.cpp:187-217](device/accumulation_program_factory.cpp#L187-L217) carry
   identical `kernel_source`, `compile_time_args`, `defines`, and `ComputeConfigDescriptor`, and
   differ only in `core_ranges`. Since the per-core work count `num_tiles_per_core` is a runtime arg
   ([lines 254,257](device/accumulation_program_factory.cpp#L254-L257)), not a per-group compile-time arg,
   the split serves no purpose and one descriptor over `all_cores` would be equivalent. (Flagged for
   the ops team; the port keeps the legacy shape.)
6. **Duplicate data-format computation.** `dst_cb_data_format`
   ([accumulation_program_factory.cpp:57](device/accumulation_program_factory.cpp#L57)) and
   `output_dataformat` ([line 100](device/accumulation_program_factory.cpp#L100)) are the same expression
   on the same tensor; both are used.
7. **Dead branch in `compute_output_specs`.** At
   [accumulation_device_operation.cpp:82-84](device/accumulation_device_operation.cpp#L82-L84), a sharded
   output memory config sets `output_layout = tensor_args.input_tensor.layout()` — but validation
   already pins the input to `Layout::TILE`
   ([lines 48-51](device/accumulation_device_operation.cpp#L48-L51)), which is the default the branch
   overwrites. The branch cannot change the result.
8. **Stale CB name in the EMA factory.** The host calls `c_2` `prev_cb_index`
   ([ema_program_factory.cpp:80](ema/device/ema_program_factory.cpp#L80)), but the kernel uses it as
   `trp_cb_idx` — a scratchpad for a transpose round trip through the packer
   ([ema_compute.cpp:80,109-120](ema/kernels/compute/ema_compute.cpp#L80-L120)). The actual "previous
   output" state lives in DST registers, managed by `ema_clear_previous_output()` /
   `ema_tile()`. The host-side name describes a mechanism the CB does not implement.

## Per-DeviceOperation attribution

Findings are identical across the two device operations on every gate. The per-DeviceOperation
differences that do exist:

| Field | `AccumulationDeviceOperation` | `EmaDeviceOperation` |
|---|---|---|
| Sheet `Op` key | `reduction/accumulation` | `reduction/accumulation/ema` |
| `Is able to port?` | yes | yes |
| Overall | GREEN | GREEN |
| Tensor bindings | 2, both Case 1 | 2, both Case 1 |
| CB dispositions | `c_0` 1:1 · `c_1` 1:1 · `c_2` self-loop | `c_0` 1:1 · `c_1` 1:1 · `c_2` self-loop |
| Kernel-instance shape | reader + writer over `all_cores`; compute split into **two** descriptors over disjoint core groups | reader + writer + compute, one descriptor each over `all_cores` |
| RTAs per dataflow kernel | 8 (base + 7 scalars) | 2 (base + 1 scalar) |
| CTAs | reader/writer: accessor args only · compute: 1 (`default_acc_value`) + `defines` for the op/dtype variant | reader/writer: 1 + accessor args · compute: 4 |
| Misc anomalies attributed | 1, 2, 3, 4, 5, 6, 7 | 8 |

## Questions for the user

None. Every check resolved from the code and the readiness sheet without residual ambiguity.

## Recipe notes

1. **A kernel already on `DataflowBuffer` rather than the Device 2.0 `CircularBuffer` wrapper.** The
   [Device 2.0 prerequisite](#device-20-prerequisite) subject and the Device 2.0 migration guide both
   frame compliance in terms of `Noc` + the kernel-side **`CircularBuffer`** wrapper (the guide's
   migrated example includes `api/dataflow/circular_buffer.h` and constructs `CircularBuffer cb(cb_id)`).
   All six kernels here instead include `api/dataflow/dataflow_buffer.h` and construct
   `DataflowBuffer` objects. I read that as clearing the gate — it is strictly ahead of the wrapper
   the guide targets, and it is where the Metal 2.0 port lands anyway — but the recipe never says
   so, and an auditor who keyed literally on `CircularBuffer` could talk themselves into a RED here.
   A one-line ruling in the Green bullet ("a kernel already using `DataflowBuffer` in place of the
   Device 2.0 `CircularBuffer` wrapper clears this gate") would remove the judgment call. Note this
   is *not* the quasar-tree caution in reverse: these kernels are on `main`, under production model
   code, and use the current header, not the stale `circular_buffer.h` include the quasar warning
   describes.
2. **The one-toucher row reads as if a self-loop implies sync-free.** The
   [CB endpoints](#cb-endpoints) table labels the 1-toucher case *"single-ended / sync-free"*, and
   the prose describes it as *"one real endpoint, or pointer-only access by that one kernel"*. Both
   self-loops in this op are a third thing: a single **compute** kernel driving **real FIFO traffic
   in both directions** on the CB, deliberately, to sequence its own unpacker against its own packer
   (`CB_ACC` here even documents the data race it prevents). The disposition is unchanged, but
   "sync-free" reads like a precondition rather than one of several one-toucher flavours, and it
   cost me a second pass to be confident. Suggest adding the genuine-FIFO self-loop explicitly to
   that row's vocabulary — it also matters to the porter, who must keep *both* FIFO paths live
   rather than treating the second binding as cosmetic.
3. **No ruling on two *identical* same-source KernelDescriptors split only by core range.** The
   dual-instance work-split face (both instances on every node) and the demoting-per-group-CTA
   anti-pattern (disjoint node sets, differing per-group CTAs) between them cover the neighbouring
   shapes, but not this one: disjoint node sets with *byte-identical* CTAs, defines, and config, so
   merging into a single `KernelSpec` over the union would be behaviour-preserving. I ruled that the
   porter keeps two specs (the port makes no functional changes) and logged the redundancy as a
   team-only anomaly. A one-line statement in the CB-endpoints or port recipe would settle it
   without each porter re-deciding.
4. **Bundling test when two device operations share only a *header*.** The multi-device-op rule
   bundles when the device operations share *"factories or kernels"*. Here they share neither a
   factory nor a bound kernel source — only the kernel header
   `device/kernels/accumulation_common.hpp`, and the child device operation uses exactly one
   constant from it. Two further wrinkles push the other way: the two live in a parent/child
   directory relation (`accumulation/` and `accumulation/ema/`, the child having its own `device/`
   and its own `kernels/`), and the readiness sheet lists them under **separate `Op` keys**
   (`reduction/accumulation` and `reduction/accumulation/ema`), i.e. as separate ops. I bundled them
   — one audit request, one directory, a genuinely shared header that a port would touch — and
   carried per-DeviceOperation attribution throughout. Worth an explicit line on whether a shared
   *header* counts as "sharing kernels", and on how to weigh a sheet that splits what the directory
   joins.
