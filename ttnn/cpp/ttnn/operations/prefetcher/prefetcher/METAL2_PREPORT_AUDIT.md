# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/prefetcher/prefetcher`

One device operation lives in this directory, with a single program factory:

- **`ttnn::prim::DramPrefetcherOperation`** (`device/dram_prefetcher_device_operation.hpp`)
  - `DramPrefetcherOperation::create_descriptor` — the op's only factory (`device/dram_prefetcher_program_factory.cpp`); the readiness sheet names it `DramPrefetcherOperation (single-descriptor)`.

Kernels (both referenced by the factory, both in scope; no unreferenced kernel files in the directory):

- `device/kernels/reader_dram.cpp` — instantiated at `dram_prefetcher_program_factory.cpp:244`
- `device/kernels/writer_l1.cpp` — instantiated at `dram_prefetcher_program_factory.cpp:255`

Not in scope: `ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/` is a **separate** op pair (`DramPrefetcherConsumerDeviceOperation`, `DramPrefetcherValidatorDeviceOperation`) that consumes this op's GlobalCircularBuffer. It shares no code with this directory and carries its own readiness rows; it is not bundled here.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/prefetcher/prefetcher` |
| **Overall** | **RED** — at op level; **no portable subset** |
| **DOps / Factories** | `DramPrefetcherOperation` → `create_descriptor` (single factory) |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — both own kernels are **broad Device 1.0** (full migration, not a cleanup) → Device 2.0 track |
| *Prereqs* — Cross-op escapes | *Skipped* — whole-op RED, no portable subset; re-audit on unblock. (The one donor include is named under Gate detail so the Device 2.0 gate is well-scoped.) |
| *Feature Support* — overall | **RED** — `GlobalCircularBuffer` in use, unconditionally |
| *Feature Support* — Variadic-CTA | Ok (N/A) — variable-count tensor list, but all CTAs are read at constexpr indices |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — sheet says `yes`; cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No (blank on the sheet; `descriptor` concept cannot carry them) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` *(recorded for the eventual re-audit; not carried into a brief — no brief is issued)* |
| *Port work* — Offset base pointer | **none** — GREEN; the op has no `->address()` runtime arg at all |
| *Port work* — Tensor bindings (per binding) | *Skipped* — whole-op RED, no portable subset; re-audit on unblock |
| *Port work* — TensorParameter relaxation | *Skipped* — whole-op RED, no portable subset; re-audit on unblock (sheet lists `none`) |
| *Port work* — TensorAccessor 3rd arg | **none** — GREEN; the op constructs no `TensorAccessor` |
| *Port work* — CB endpoints | *Skipped* — whole-op RED, no portable subset; re-audit on unblock (also deferred by the broad-Device-1.0 rule) |

**CB endpoints** are dispositions, not gates. This subject was **not run** — see the skip note above and under *Skipped subjects*.

## Result

**RED at op level; no portable subset.** Two independent gates fire, and both must clear before this op can be re-audited:

1. **`GlobalCircularBuffer` is not supported in Metal 2.0** (Appendix A). This is not a branch of one factory that a subset port could route around — the GlobalCircularBuffer *is* the op. `validate_on_program_cache_miss` hard-requires it (`dram_prefetcher_device_operation.cpp:19`), the factory hard-requires it (`dram_prefetcher_program_factory.cpp:41`), the op's core grid, buffer, and size all come from it, and the writer kernel's entire output path is the remote-CB sender API. Route: **wait-for-feature** — the port becomes possible when a `GlobalDataflowBuffer` (the user-managed-lifetime analog) lands on `KernelSpec` / `DataflowBufferSpec`.

2. **Neither kernel is Device 2.0 migrated**, and the incompleteness is **broad**, not isolated holdovers — a full migration, not a cleanup. Route: **the Device 2.0 migration team**.

The two are independent and can be worked in parallel. Note the ordering constraint for planning: the Device 2.0 migration of `writer_l1.cpp` is tractable today — a `RemoteCircularBuffer` wrapper class already exists at `tt_metal/hw/inc/api/remote_circular_buffer.h:457` with `reserve_back` / `push_back` / resize methods that wrap exactly the free functions the kernel calls — so the Device 2.0 work does **not** have to wait on the Metal 2.0 GlobalCircularBuffer feature.

Everything else the audit checks is **clean**: the TTNN factory-concept gate passes, there are no offset base pointers, and there is no `TensorAccessor` 3rd-argument site. So once GlobalCircularBuffer support and the Device 2.0 migration land, this op has no other known blocker.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The sheet's row for `prefetcher/prefetcher` → `DramPrefetcherOperation` → `DramPrefetcherOperation (single-descriptor)` reads `Is able to port? = yes`. Cross-check against the code is clean on every cheaply-checkable column:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` returns `ProgramDescriptor` — `dram_prefetcher_device_operation.hpp:26-29`, defined at `dram_prefetcher_program_factory.cpp:35` |
  | `Custom hash (compute_program_hash)` | `no` | no `compute_program_hash` anywhere under `operations/prefetcher/` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on the device-op |
  | `Override runtime args method? (PD and legacy)` | `no` | no `override_runtime_arguments` anywhere under `operations/prefetcher/` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | `dram_prefetcher_nanobind.cpp:17-38` binds only the free function `ttnn::dram_prefetcher`; no `create_descriptor` binding, no `nb::class_` of the device op |
  | `Op-owned tensors?` | *(blank)* | consistent — a `descriptor`-concept row cannot carry op-owned tensors |
  | `Secretly SPMD Workload?` | *(blank)* | N/A — not a `WorkloadDescriptor` |
  | Factory-set match | 1 row | 1 factory in code — one-to-one, no phantom or missing row |
  | `Is safe to port?` | `yes` | *(not verified — expert-judgment axis, by design)* |

  No cross-column invariant is violated. Nothing to route here.

- **Device 2.0 (every kernel used):** **RED — broad Device 1.0** on both of the op's own kernels. Routed to the **Device 2.0 migration team**. This is a full migration, not a holdover cleanup: neither kernel constructs a `Noc`, a `CircularBuffer`, a `RemoteCircularBuffer`, or a `CoreLocalMem` anywhere — every data-movement and circular-buffer operation goes through the legacy free-function surface, and both include `api/dataflow/dataflow_api.h` rather than the Device 2.0 headers (`api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`). Because no wrapper object is ever in scope, none of these is an "isolated CB-index holdover" — the fix is to introduce the objects, not to swap a call.

  Representative violations (not exhaustive within each repeated idiom; every call listed below is a distinct site):

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/reader_dram.cpp` | 7 | `#include "api/dataflow/dataflow_api.h"` | — (no Device 2.0 headers included) |
  | `device/kernels/reader_dram.cpp` | 34, 35, 60 | `get_write_ptr(cb_id)` | none — no `CircularBuffer` object exists |
  | `device/kernels/reader_dram.cpp` | 38 | `get_read_ptr(addrs_cb_id)` | none |
  | `device/kernels/reader_dram.cpp` | 48 | `get_noc_addr_from_bank_id<true>(bank_id, tensor_base_address)` | none — Device 2.0 shape is `AllocatorBank<AllocatorBankType>` |
  | `device/kernels/reader_dram.cpp` | 49 | `noc_async_read_one_packet_set_state<true>(...)` | none — no `Noc` object |
  | `device/kernels/reader_dram.cpp` | 57, 100 | `cb_reserve_back(cb_id, ...)` | none |
  | `device/kernels/reader_dram.cpp` | 69 | `noc_async_read_set_trid(...)` | none |
  | `device/kernels/reader_dram.cpp` | 74 | `noc_async_read_one_packet_with_state_with_trid<...>(...)` | none |
  | `device/kernels/reader_dram.cpp` | 83, 108 | `noc_async_read_barrier_with_trid(...)` | none |
  | `device/kernels/reader_dram.cpp` | 84, 109 | `cb_push_back(cb_id, ...)` | none |
  | `device/kernels/reader_dram.cpp` | 114, 115 | `cb_wait_front(sync_cb_id, 1)` / `cb_pop_front(sync_cb_id, 1)` | none |
  | `device/kernels/reader_dram.cpp` | 118-122 | raw `noc_mode` test + `ncrisc_noc_counters_init()` / `dynamic_noc_local_state_init()` | none |
  | `device/kernels/writer_l1.cpp` | 7 | `#include "api/dataflow/dataflow_api.h"` | — (no Device 2.0 headers included) |
  | `device/kernels/writer_l1.cpp` | 39 | raw `noc_index` read into a local `noc` **uint32_t** (threaded into every call below) | none — Device 2.0 shape is a `Noc` object |
  | `device/kernels/writer_l1.cpp` | 50 | `experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, ...)` | none — `RemoteCircularBuffer::resize_page_size` exists |
  | `device/kernels/writer_l1.cpp` | 54, 79 | `cb_wait_front(local_cb_id, ...)` / `cb_pop_front(local_cb_id, ...)` | none |
  | `device/kernels/writer_l1.cpp` | 55 | `experimental::remote_cb_reserve_back(remote_cb_id, 1)` | none — `RemoteCircularBuffer::reserve_back` exists |
  | `device/kernels/writer_l1.cpp` | 56 | `get_read_ptr(local_cb_id)` | none |
  | `device/kernels/writer_l1.cpp` | 57-64 | `experimental::remote_cb_push_back_and_write_pages<...>(...)` | none — `RemoteCircularBuffer::push_back` exists |
  | `device/kernels/writer_l1.cpp` | 75, 77 | `noc_async_posted_writes_flushed(noc)` / `noc_async_writes_flushed(noc)` | none — no `Noc` object |
  | `device/kernels/writer_l1.cpp` | 84 | `experimental::remote_cb_sender_barrier(remote_cb_id)` | none |
  | `device/kernels/writer_l1.cpp` | 89 | `experimental::update_remote_cb_config_in_l1(remote_cb_id)` | none |
  | `device/kernels/writer_l1.cpp` | 90 | `noc_async_atomic_barrier()` | none |
  | `device/kernels/writer_l1.cpp` | 92-96 | raw `noc_mode` test + noc-counter re-init | none |
  | `device/kernels/writer_l1.cpp` | 98, 99 | `cb_reserve_back(sync_cb_id, 1)` / `cb_push_back(sync_cb_id, 1)` | none |

  **Donor kernels:** the op instantiates no borrowed kernel `.cpp` — it owns both. `reader_dram.cpp:8` includes one out-of-directory header, `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`, but the **only** thing it uses from it is `increment_arg_idx` (`worker_sync_utils.hpp:18`), a pure runtime-arg-index counter that touches no resource handle. So there is **no donor-side Device 2.0 dependency** — the gate is scoped entirely to this op's own two kernels, and the Device 2.0 team can schedule it without a cross-family prerequisite. (`writer_l1.cpp:11-15` defines its own local copy of the same helper.)

- **Feature compatibility:** all four Appendix A entries scanned.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **RED** | Unconditional in the op's only factory — no clean subset. Detail below. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` / `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor` anywhere under `operations/prefetcher/`. The two Buffer-backed CBs (`dram_prefetcher_program_factory.cpp:156`, `:177`) leave `address_offset` at its default zero — the ordinary borrowed-memory pattern, not this rule. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include. The op creates no semaphores at all — cross-core sync rides the GlobalCircularBuffer's own credit mechanism, and reader/writer sync rides the local `sync_cb`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Detail below. |

#### GlobalCircularBuffer — UNSUPPORTED (RED)

**Signals that fired** (several, all definitive):

- **Descriptor-API attachment** — the arcane signal, and the decisive one: `dram_prefetcher_program_factory.cpp:188`, `.global_circular_buffer = std::addressof(global_cb)` on a `CBDescriptor` (the remote CB, `buffer_index = tt::CBIndex::c_31`, built at `:180-189`).
- **Type reference** — `std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb` as an operation attribute, `device/dram_prefetcher_device_operation_types.hpp:17`.
- **Factory-function signature** — `const std::optional<const GlobalCircularBuffer>&` at `device/dram_prefetcher_device_operation.hpp:39`, `device/dram_prefetcher_device_operation.cpp:102`, `dram_prefetcher.hpp:22`, `dram_prefetcher.cpp:14`.
- **Include** — `<tt-metalium/global_circular_buffer.hpp>` at `device/dram_prefetcher_device_operation.hpp:13`, `device/dram_prefetcher_device_operation_types.hpp:10`, `device/dram_prefetcher_program_factory.cpp:12`, `dram_prefetcher.hpp:11`; the `experimental/` spelling at `device/dram_prefetcher_device_operation.cpp:9`.
- **Remote-CB idiom** — `remote_cb_size` / `remote_cb_index` / `remote_format_descriptors` at `dram_prefetcher_program_factory.cpp:141,144,183-187`; kernel-side `#include "api/remote_circular_buffer.h"` at `device/kernels/writer_l1.cpp:8` with `experimental::remote_cb_*` calls at `:50,55,57,84,89`.
- **GlobalCircularBuffer accessors drive the whole factory** — `global_cb.cb_buffer()` (`:47`), `.sender_receiver_core_mapping()` (`:77`), `.size()` (`:108,111,128,141`), `.sender_cores()` (`:114`). The op's core grid, its reader CB backing buffer, and its size checks are all derived from the GCB.

**Code-path scope: no clean subset.** The op has exactly one factory, and the GlobalCircularBuffer is not behind a branch in it. `validate_on_program_cache_miss` fails hard without one (`device/dram_prefetcher_device_operation.cpp:19`), as does the factory (`dram_prefetcher_program_factory.cpp:41`). There is no interleaved/sharded, tiled/row-major, or single-/multi-core sibling path to carve out. Per the Appendix A action ("RED the offending factory and name the clean factories as a subset"), the offending factory here **is** the only factory → `RED at op level; no portable subset`.

**Expected resolution:** not yet supported in Metal 2.0; the port will be possible once GlobalCircularBuffer support lands. Per the entry's Status field, the eventual analog is the (unimplemented) user-managed `GlobalDataflowBuffer` — mapped by *lifetime*. **Do not** map this op's remote CB onto the local `DataflowBuffer` or onto the cross-node DFB stub in `dataflow_buffer_spec.hpp`; despite the "remote CB" nickname, that stub is a separate ephemeral construct with no legacy analog, and it is not this op's destination.

#### Variable-count compile-time arguments (CTA varargs) — N/A

The **op-level signal fires**: `tensor_args_t` is `DramPrefetcherInputs { std::vector<Tensor> input_tensors; }` (`device/dram_prefetcher_device_operation_types.hpp:20-22`) — a genuinely runtime-varying list (N data tensors plus a trailing address tensor), and the nanobind arg is a Python `List[ttnn.Tensor]` (`dram_prefetcher_nanobind.cpp:24,34`).

Per the entry, that signal is a prompt to read the kernel, not a verdict. **The kernel-level signal does not fire**, so the rule is N/A:

- `reader_dram.cpp:14-23` reads exactly ten compile-time args, every one at a **literal constant index** `0`–`9`. There is no loop over `get_compile_time_arg_val`, no runtime-varying index, and no template instantiated over a variable count.
- `writer_l1.cpp:19-27` likewise reads exactly nine, all at literal indices `0`–`8`.
- The varying quantity — `num_tensors` — arrives *as* CTA index 1 (a single scalar), and **all per-tensor metadata rides runtime args**, not CTAs: the factory appends `page_sizes`, `block_num_pages`, `tensor_block_num_tiles` per core at `dram_prefetcher_program_factory.cpp:285-287` and `coalesced_page_sizes`, `coalesced_num_pages`, `tensor_block_num_tiles`, `tensor_tile_sizes`, per-tensor block heights at `:293-299`; the kernels pick them up as pointers into the RTA region via `get_arg_addr(increment_arg_idx(rt_args_idx, num_tensors))` (`reader_dram.cpp:30-32`, `writer_l1.cpp:32-37`) and index them with the loop variable `t`.

That is exactly the entry's third false-positive guard — a variable-count input list whose per-input data rides RTAs, so the compile-time-arg *count* is fixed. The CTA-vararg gate does not fire. *(The RTA side of this shape is a genuine RTA-vararg pattern, which Metal 2.0 supports; it is inventoried by the RTA-varargs subject, skipped here per the RED scoping rule.)*

- **CB endpoints (GATE-free):** **Not run.** *Skipped — whole-op RED, no portable subset; re-audit on unblock.* Independently, the subject's own precondition also defers it: the Device 2.0 gate is RED **broadly** (not isolated holdovers), and the subject directs `(deferred — re-evaluate after Device 2.0 migration)` in that case, because the Device 2.0 rewrite changes the very idioms the endpoint scan keys on. Nothing here gates a port regardless.

- **Offset base pointers:** **GREEN.** The op has **no** `->address()` expression anywhere — not in a runtime-arg list, not in a common-runtime-arg list, not in a compile-time-arg list, and not via a helper (verified by grepping `address()` across the whole `operations/prefetcher/` tree: zero hits). With no address arg, there is no base into which an offset could be folded, so no Type 1 (raw offset arg) and no Type 2 (accessor-fed offset arg) site exists. Type 3 (`address_offset`) is absent — see the Appendix A row. Type 4 (`ttnn::narrow`) does not appear.

  Consistent with the checked-in triage prior: `analyses/2026-07-19_offset_base_pointers.md` does not list this op. That agreement is a cross-check only — the verdict above rests on the scan, not on the doc's silence.

  **How the op addresses DRAM instead, recorded so the eventual re-audit doesn't re-derive it.** The data tensors' DRAM base addresses never pass through the host arg channel at all. They live in a **device-side address table**: the trailing input tensor (`tensor_addrs`, an L1 height-sharded row-major `UInt32` tensor — validated at `device/dram_prefetcher_device_operation.cpp:73-80`) is attached as the backing buffer of a borrowed-memory CB at `dram_prefetcher_program_factory.cpp:169-178`, and the reader indexes it on-device as `tensor_addrs_l1[layer * num_tensors + t]` (`device/kernels/reader_dram.cpp:37-38,47`), feeding the result to `get_noc_addr_from_bank_id<true>(bank_id, tensor_base_address)` (`:48`). The per-page walk then advances a device-side `src_read_addr` offset (`:51,75-76`) that is passed as a *separate* argument to `noc_async_read_one_packet_with_state_with_trid` — it is never added into the base on the host. So the base is clean by construction, and this is the reverse of the offset-fold hazard: the offsets are all computed on-device.

- **TensorAccessor 3rd argument:** **GREEN — N/A.** The op constructs no `TensorAccessor` at all (zero hits for `TensorAccessor` across `operations/prefetcher/`), so there is no 3rd-argument site to classify. Consistent with the checked-in prior: `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md` does not list this op — again a cross-check, not the basis of the verdict.

## Skipped subjects

Per the **Red** outcome scoping rule (whole-op RED, no portable subset), the seven purely-informational subjects were **not run**. Each is recorded here so the omission is never mistaken for a clean result:

| Subject | Note |
|---|---|
| TTNN porting shape | skipped — whole-op RED, no portable subset; re-audit on unblock *(target concept `MetalV2FactoryConcept` is recorded in the status summary from the sheet's `descriptor` + no-op-owned-tensors, for planning only)* |
| TensorParameter relaxations | skipped — whole-op RED, no portable subset; re-audit on unblock *(sheet lists `none`, and the op has no custom hash, so no relaxation candidate to mine)* |
| TensorParameter analysis | skipped — whole-op RED, no portable subset; re-audit on unblock |
| CB endpoints | skipped — whole-op RED, no portable subset; re-audit on unblock *(also deferred by the broad-Device-1.0 precondition)* |
| Out-of-directory coupling | skipped — whole-op RED, no portable subset; re-audit on unblock *(the one donor include is named under the Device 2.0 gate detail, so the gate is well-scoped)* |
| RTA varargs | skipped — whole-op RED, no portable subset; re-audit on unblock |
| Incidental anomalies | skipped — whole-op RED, no portable subset; re-audit on unblock |

## Port-work summary

None issued. The op is RED with no portable subset, so no port work is scoped and **no `METAL2_PORT_BRIEF.md` is written**.

## Heads-ups

None issued — see above.

## Team-only

- **Out-of-directory coupling & donor shape:** the full by-shape inventory was skipped (see *Skipped subjects*). The gate-relevant fact is recorded under the Device 2.0 gate detail: the op owns both its kernels, borrows no kernel `.cpp`, and its single out-of-directory include (`ccl/kernel_common/worker_sync_utils.hpp`) contributes only `increment_arg_idx`, a resource-handle-free arg-index counter. No donor blocks the Device 2.0 gate.
- **Relaxation candidates:** none — the op has no custom hash to mine.
- **TTNN factory analysis:** all sheet-derived facts and their `file:line` evidence are in the *Gate detail* cross-check table above. Every gate conjunct is `no`/`yes`-clearing; the non-gating facts are: no op-owned tensors, no MeshWorkload need (plain `descriptor` concept, single program), no pybind of factory or device-op internals (`dram_prefetcher_nanobind.cpp:17-38` binds the free function only).

## Questions for the user

1. **Is a `GlobalDataflowBuffer` timeline known?** This op is blocked on a feature, not on op-side work, and its entire design is the GlobalCircularBuffer. `device/dram_prefetcher_device_operation.cpp:19-25` also shows the op deliberately *rejects* DRAM-sender GCBs and points users at `ttnn.experimental.start_tensor_prefetcher` / `stop_tensor_prefetcher` — worth confirming with the ops team whether the worker-sender prefetcher is still the strategic path before the Device 2.0 team invests in migrating these two kernels, or whether that investment should follow the other prefetcher entry point instead.

## Recipe notes

1. **Appendix A's GlobalCircularBuffer "Examples in the wild" points at the wrong file for this op.** It lists `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_device_operation.cpp`. That file carries the include and a `sender_core_type` validate call, but the **descriptor-API attachment** — the signal the entry itself calls "the arcane signal ... an AI scanning a `CBDescriptor` setup can easily miss it" — is in `dram_prefetcher_program_factory.cpp:188`. An auditor who ground-truths the match against the named file finds only the weak signals and could under-call it. Suggest repointing the example at the program factory.

2. **"Incidental anomalies" is listed among the seven skippable subjects, but it is not a scan — it is opportunistic capture that has already happened by the time you know the op is RED.** The subject itself says "don't go hunting — just note what you happen to see," and the rationale for skipping the other six (unread, stale, expensive detail) doesn't transfer: noting a thing you already noticed costs one line and can't go stale in a way that matters, since it routes to the ops team rather than to the porter. As written I recorded it as skipped. Suggest either exempting it from the RED scoping rule, or saying explicitly that already-noticed items should still be written down.

3. **The op's addressing shape — a device-side address table — is not one the TensorParameter-analysis model anticipates.** That subject's cases are: fed to a `TensorAccessor` (Case 1), used raw from an RTA-delivered base (Case 2), or read through a borrowed-memory DFB (clean). Here the *data* tensors are not bound at all: a separate L1 tensor holds their DRAM base addresses, is itself attached as a borrowed-memory CB, and the kernel reads a base out of it and hands that to `get_noc_addr_from_bank_id` (`device/kernels/reader_dram.cpp:47-48`). The borrowed-DFB "causal-link gate" covers the *address-table* tensor cleanly, but says nothing about the N data tensors whose bases arrive through it — they have no host-side arg site at all, so no case applies. This op is RED for other reasons so it didn't need resolving, but a future auditor hitting the same shape on a GREEN op would have to improvise. Worth a sentence in the subject, if the shape recurs.

4. **The offset-base-pointer subject frames its scan as "you are already resolving every address RTA for TensorParameter analysis; the only added question is whether an offset is folded in."** On this op that framing inverts: TensorParameter analysis is a skipped informational subject, while Offset base pointers is gate-bearing and always runs — so on a RED op the "nearly free" scan is the *only* one being run, not a rider on another. It worked out fine (the answer is "zero address RTAs"), but the stated dependency runs backwards for exactly the RED case where the gate-bearing subject stands alone. A word acknowledging that the gate scan is self-contained would help.
