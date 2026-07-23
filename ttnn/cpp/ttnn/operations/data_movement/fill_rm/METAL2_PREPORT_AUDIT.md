# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/fill_rm`

- **`FillRMDeviceOperation`** (`device/fill_rm_device_operation.hpp` / `.cpp`)
  - `FillRMProgramFactory` (`device/fill_rm_program_factory.hpp` / `.cpp`)
    - kernel: `device/kernels/dataflow/fill_rm_interleaved.cpp` (op-owned; the only kernel)

Single DeviceOperation, single ProgramFactory, single kernel — no bundling needed. `fill_rm` and `fill_ones_rm` are two host entry points into the *same* device op (`fill_ones_rm` just hardwires `val_hi=1, val_lo=0`), so there is one porting unit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `7ca84865be5 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/fill_rm` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `FillRMDeviceOperation` → `FillRMProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — kernel already on Device 2.0 idioms (`Noc`, `DataflowBuffer`, wrapper methods); only sanctioned free fn `get_tile_size(cb_id)` |
| *Prereqs* — Cross-op escapes | Ok — no donor kernels; all includes are `tt_metal/hw/inc/api/*` (LLK/HAL) |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore / CTA-varargs | N/A (all absent) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean base) |
| *Port work* — Tensor bindings (per binding) | output → **Case 1** (via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | **drop** (Class 2 — redundant on interleaved) |
| *Port work* — CB endpoints | self-loop ×2 (CB `buffer_index` 0 and 1) |

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (Class 2 drop). The op is a genuinely clean, single-factory port. Port work is minimal: one Case-1 tensor binding, drop one redundant page-size arg, self-loop two single-toucher CBs.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (one row, exact match on `data_movement/fill_rm`) reads `Is able to port? = yes`, derived from: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no` (both the `get_dynamic_runtime_args` and PD `override_runtime_args` columns), `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`. **Cross-check clean** against the code:
  - `Concept = descriptor` — `FillRMProgramFactory::create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/fill_rm_program_factory.hpp:14`). ✓
  - `Custom hash = no` — no `compute_program_hash` override anywhere in the op (grep clean). Default TMP hash is used. ✓
  - `Runtime-args update = no` — no `override_runtime_arguments` / `get_dynamic_runtime_args` (grep clean). ✓
  - `Pybind descriptor = no` — `fill_rm_nanobind.cpp` binds the plain function `&ttnn::fill_rm` / `&ttnn::fill_ones_rm`; no `create_descriptor` binding, no `nb::class_` of the device op. ✓
  - Cross-column invariants hold: `Op-owned tensors?` is blank (consistent — a `descriptor`-concept op cannot carry op-owned tensors); `Secretly SPMD` blank (N/A — not `WorkloadDescriptor`).
- **Device 2.0 (every kernel used):** **GREEN.** The op's single kernel `device/kernels/dataflow/fill_rm_interleaved.cpp` is already on Device 2.0 idioms — `Noc noc; noc.async_write(...) / noc.async_write_barrier()` (`:64,70-76`), `DataflowBuffer dfb_in0/dfb_in1` with `reserve_back` / `push_back` / `get_write_ptr()` **wrapper methods** (`:36-37,47-50,61-62`), `TensorAccessor` / `TensorAccessorArgs` (`:28,31`). No Device 1.0 addr-gen (`InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_read/write`), no manual CB index management, no raw sem addresses. The only CB-index free function is `get_tile_size(cb_id_in0)` (`:39`), which is **sanctioned** by the Device 2.0 Green bullet (Device 2.0 keeps it as a free function). No holdovers, no violations.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | *(none — no Device 2.0 violations)* | — | — | — |

- **Feature compatibility (Appendix A):** every entry `N/A` — no `GREEN`/`RED` row fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Two plain `CBDescriptor`s (`fill_rm_program_factory.cpp:49-67`); no `.global_circular_buffer` field, no `remote_index` / `remote_cb_*` idiom, no `CreateGlobalCircularBuffer`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Neither `CBDescriptor` sets `address_offset` (default 0). No `cb_descriptor_from_sharded_tensor`, no 4-arg `UpdateDynamicCircularBufferAddress`. |
  | GlobalSemaphore | N/A | Op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | CTAs are a fixed `TensorAccessorArgs(*dst_buffer)` block (`fill_rm_program_factory.cpp:69-70`); kernel reads `TensorAccessorArgs<0>()` at a constexpr offset (`fill_rm_interleaved.cpp:28`). `tensor_args_t` is a single fixed `Tensor input` (`fill_rm_device_operation_types.hpp:23-25`) — no `std::vector<Tensor>`, no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** two CBs, both **self-loop** (single config — the factory hardwires a 1-core `CoreRange{{0,0},{0,0}}` and interleaved-only). Each CB is touched by exactly one kernel (the lone reader): `buffer_index 0` (`dfb_in0`) — `reserve_back(16)`/`push_back(16)` (`:47,61`), `get_write_ptr()` (`:49`), and read as the `noc.async_write` source (`:70`); `buffer_index 1` (`dfb_in1`) — the same shape (`:48,62,50,73`). One toucher per CB → bind each kernel PRODUCER **and** CONSUMER (self-loop, legal on Gen1 for a DM kernel). No hidden second writer (single kernel), no multi-reader, no dead CB. Disposition per `(CB, config)`: `CB0` → self-loop (all configs); `CB1` → self-loop (all configs).
- **Offset base pointers:** **GREEN — clean base.** The only address RTA is the output buffer, delivered as a `Buffer*` (`dst_buffer`) in slot 0 of `emplace_runtime_args` (`fill_rm_program_factory.cpp:82-91`) — the framework's `Buffer*`-binding form, not a `->address()` expression, and with **no** host-folded offset. The kernel reads it as `dst_addr = get_arg_val<uint32_t>(0)` (`fill_rm_interleaved.cpp:19`) and uses it only as a clean `TensorAccessor` base (`:31`); the per-page position is a page index (`page_id = nch_dst`, `:71,74`), computed on-device, not a host-folded pointer offset. No Type 1/2/3/4 fold. Hands a clean base to TensorParameter analysis. (fill_rm is not in the `2026-07-19_offset_base_pointers.md` triage tables; scan confirms clean, consistent with its absence.)
- **TensorAccessor 3rd argument:** **GREEN — Class 2 (redundant, drop).** One site: `TensorAccessor(dst_args, dst_addr, W << 1)` (`fill_rm_interleaved.cpp:31`). Classified by the two questions:
  1. **Interleaved** — the op hard-rejects sharding (`validate_on_program_cache_miss` `TT_FATAL`s both input and output `TensorMemoryLayout::INTERLEAVED`, `fill_rm_device_operation.cpp:36-41`). So realignment is in play.
  2. **Correct magnitude** — the true logical page is one row-major row = `W` bf16 elements = `2·W` bytes; the arg passes exactly `W << 1` (= `2·W`), matching the per-page write size `noc.async_write(..., (W << 1), ...)` (`:70,73`). Correct magnitude; on an interleaved accessor it is realigned up to allocator alignment (harmless).
  → **Class 2**: drop the arg at port time (Metal 2.0 supplies `aligned_page_size` implicitly). Not Class 1 (dynamic): `W` is part of `operation_attributes` and thus of the default program hash, so a different `W` is a cache **miss** (rebuild), never a stale cache hit — the kernel comment at `:29-30` ("may be stale on program cache hits") describes a hazard that cannot arise for this op, so the override is defensive/redundant, not load-bearing. (fill_rm is not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table; classified here from first principles.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): **output** — **Case 1** (via `TensorAccessor`). Delivered today as a `Buffer*` RTA in slot 0 (`fill_rm_program_factory.cpp:82-91`), fed into `TensorAccessor(dst_args, dst_addr, …)` in the kernel (`fill_rm_interleaved.cpp:31`). Express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA + its `TensorAccessorArgs` CTA plumbing (`fill_rm_program_factory.cpp:69-70`) both disappear. The **input** tensor (`tensor_args.input`) is *not* a kernel binding — it is consumed host-side only, for `dtype()` and `device()` (`fill_rm_device_operation.cpp:33-34,57`; it is the doc'd "any" metadata tensor); no on-device access, no work item.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash to reconcile).
- **TensorAccessor 3rd arg:** drop the redundant page-size arg `W << 1` @ `fill_rm_interleaved.cpp:31` (Class 2 — no `dynamic_tensor_shape` needed).
- **CB endpoints:** self-loop `CB buffer_index 0` (all configs) and `CB buffer_index 1` (all configs) — bind the single reader kernel as both PRODUCER and CONSUMER on each.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — both CBs are single-toucher self-loops; no hidden co-fill, no multi-reader.
- **Cross-op / shared kernels:** none — the op owns its only kernel and instantiates no borrowed/donor kernel. Kernel `#include`s are all `tt_metal/hw/inc/api/*` (LLK/HAL, Class 1); no function-call escape.
- **RTA varargs:** none. The 8 RTAs (`fill_rm_program_factory.cpp:82-91`) are read at fixed constexpr indices 0–7 (`fill_rm_interleaved.cpp:19-26`) — all nameable (`dst_addr`→tensor binding CRTA; then `NC`, `H`, `W`, `fillH`, `fillW`, `val_hi`, `val_lo`). No loop-indexed or data-selected reads → no vararg block.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** Function-call escapes: none outside `tt_metal/*`. Kernel includes — `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` — all resolve under `tt_metal/hw/inc/api/` (donor class 1 — LLK/HAL/firmware, no concern). Borrowed kernel files (file-path instantiation): none — the factory's single `KernelDescriptor::kernel_source` points at the op's own `fill_rm_interleaved.cpp` (`fill_rm_program_factory.cpp:73-74`). No port-together set.
- **Relaxation candidates** (mined from a custom hash): N/A — no custom hash.
- **TTNN factory analysis:** `Concept = descriptor`; op-owned tensors = none; MeshWorkload = not needed; pybind `create_descriptor` = no; other risky pybind = none; custom hash = no; custom `override_runtime_arguments` = no. Target concept `MetalV2FactoryConcept`. All gate conjuncts absent — see Gate detail.

## Misc anomalies  *(team-only, non-gating; not porter work)*

Dead kernel locals in `device/kernels/dataflow/fill_rm_interleaved.cpp` — declared and never used (the kernel's own comment at `:17` notes it is written "with maximum simplicity in mind"):

- `num_bytes_per_tile = get_tile_size(cb_id_in0)` @ `:39` — computed, never read.
- `num_bytes_per_tile_row = 64` @ `:40` — never read.
- `Wt = (W >> 5)` @ `:41` — assigned once, referenced only in comments thereafter, never in code.
- `replicate_dest_addr` @ `:44` — declared, never assigned or read.
- `start_dram_addr_offset_for_tensor_row = 0` @ `:45` — never read.

These route to the ops team as an optional cleanup; the Metal 2.0 port does **not** act on them (kernel-body cleanup is out of port scope). Note that dropping `num_bytes_per_tile` would remove the op's only `get_tile_size(cb_id)` call — irrelevant to the Device 2.0 gate (it is sanctioned either way).

## Recipe notes  *(none)*

The recipe covered every case cleanly; no friction to report.
