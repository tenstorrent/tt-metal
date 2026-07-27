# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_arange`

Single device operation, single program factory (one `create_descriptor`) that selects one of two op-owned writer kernels by the `untilize_out` attribute:

- **`MorehArangeOperation`**
  - `MorehArangeOperation (single-descriptor)` — `device/moreh_arange_program_factory.cpp`
    - `untilize_out == false` → `device/kernels/writer_moreh_arange.cpp` (TILE output)
    - `untilize_out == true`  → `device/kernels/writer_moreh_arange_rm.cpp` (ROW_MAJOR output)

Both kernel variants are exercised by the one factory (config-selected), so they are audited together as one porting unit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_arange` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehArangeOperation` → single-descriptor factory (tile + RM kernel variants) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — both writer kernels are Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok — kernels include only `api/*` (tt_metal HAL/LLK) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (fixed-count CTAs) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `output` → Case 1 (via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd arg passed) |
| *Port work* — CB endpoints | self-loop (`c_16`, both configs) |

**CB endpoints** are dispositions, not gates: the op's single CB is single-toucher and resolves via a **self-loop** (bind the writer PRODUCER **and** CONSUMER), the same under both `untilize_out` configs.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (absent). This is a small, single-factory, single-output writer op with no sharding, no semaphores, no cross-op coupling. `METAL2_PORT_BRIEF.md` is emitted alongside this file.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (`moreh/moreh_arange`, `MorehArangeOperation`, single-descriptor) reads `Is able to port? = yes`. Cross-check against the code, all consistent:
  - `Concept = descriptor` — confirmed: `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/moreh_arange_device_operation.hpp:30`, `device/moreh_arange_program_factory.cpp:21`).
  - `Custom hash = no` — confirmed: no `compute_program_hash` override in the device op.
  - `Runtime-args update = no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments` in the factory.
  - `Pybind descriptor = no` — confirmed: `moreh_arange_nanobind.cpp:17` binds the host function `&ttnn::moreh_arange`, not `create_descriptor`.
  - `Is safe to port? = yes`, `Smuggled pointer = no` (readiness-sheet owner's correctness call — trusted, not re-derived).
- **Device 2.0 (every kernel used):** **GREEN.** Both writer kernels are structurally Device 2.0: `Noc` object + `noc.async_write` / `noc.async_write_barrier`, `CircularBuffer` wrapper (`reserve_back`, `get_write_ptr`), `TensorAccessor`, `CoreLocalMem`, and the `use<CircularBuffer::AddrSelector::WRITE_PTR>(...)` object form. No legacy Device 1.0 idioms (no raw `noc_async_read/write`, no `InterleavedAddrGen`/`ShardedAddrGen`, no manual CB-index management). The only CB-index free function is the **sanctioned** `get_tile_size(cb_out)` (`device/kernels/writer_moreh_arange.cpp:24`) — not a holdover. *(Breadcrumb, port-time only: the Metal 2.0 port moves that `get_tile_size` lookup onto the DFB metadata accessor per kernel-side whitelist rule 7; this does not affect the Device 2.0 boundary.)*

- **Feature compatibility:** all Appendix A entries scanned; none fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Plain `CBDescriptor` (`device/moreh_arange_program_factory.cpp:44`); no `.global_circular_buffer` field, no `remote_index`, no experimental include. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` never set (default 0). |
  | GlobalSemaphore | N/A | Op uses no semaphores of any kind. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | CTAs are a single fixed-shape `TensorAccessorArgs(*output.buffer())` (`program_factory.cpp:64`); kernels read `TensorAccessorArgs<0>()` at a constexpr offset. `tensor_args_t` carries a single optional output, not a variable-count container. |

- **CB endpoints (GATE-free):** one CB, `c_16` (`buffer_index = tt::CBIndex::c_16`, `program_factory.cpp:42`/`44`). **Single toucher** — only the writer kernel touches it (`reserve_back(1)` + `get_write_ptr()`; **no** `push_back`/`wait_front`/`pop_front`). It is used as a per-tile scratch staging buffer: the kernel reserves space, fills it via `CoreLocalMem`, then `noc.async_write`s the tile to DRAM. Single-ended → **self-loop** (bind the writer PRODUCER **and** CONSUMER; legal on Gen1 for a DM kernel). Disposition is identical under both configs (`untilize_out` true → RM kernel, false → tile kernel).
- **Offset base pointers:** **GREEN — no host-folded offset.** The only address argument is `output.buffer()` (a `Buffer*`) pushed as the first runtime arg (`program_factory.cpp:90`) — a clean base, no `base + offset` arithmetic on the host. The RM kernel's per-tile DRAM offset (`noc_offfset = tile_idx * TILE_WIDTH * element_size`, `writer_moreh_arange_rm.cpp:80`) is computed **on-device** and applied through the `TensorAccessor`'s `offset_bytes` field, not folded into the base host-side. Not present in the offset-base-pointer triage doc; own scan is clean.
- **TensorAccessor 3rd argument:** **GREEN — N/A.** Both kernels construct `TensorAccessor(dst_args, dst_addr)` with only two arguments (`writer_moreh_arange.cpp:27`, `writer_moreh_arange_rm.cpp:28`). No explicit page-size third argument anywhere.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `output` — **Case 1** (via `TensorAccessor`). The output buffer is delivered today via the `Buffer*`-binding form (`emplace_runtime_args({output.buffer(), …})`, `program_factory.cpp:88-90`); the framework auto-registers it as a `BufferBinding` and patches it on cache hits, so this is *not* the silent-wrong stale-address hazard — routine port work. The kernel receives the raw base as `dst_addr = get_arg_val<uint32_t>(0)` and feeds it straight into `TensorAccessor(dst_args, dst_addr)`. Port: express `output` as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`, and both the CTA `TensorAccessorArgs` plumbing (`program_factory.cpp:64`) and the RTA address argument disappear. Applies to both kernel variants.
- **TensorParameter relaxation:** none (sheet: `none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_16` (both `untilize_out` configs).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — the single CB is a one-toucher self-loop.
- **Cross-op / shared kernels:** none — both kernels are op-owned and instantiated by file path; kernel `#include`s are limited to `api/*` (tt_metal HAL/LLK).
- **RTA varargs:** none — both kernels read a fixed set of distinct scalar fields (tile kernel args 0–4; RM kernel args 0–5), each read once at a constexpr index. No loop-indexed or data-selected reads.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** No function-call escapes (kernel `#include`s resolve only to `tt_metal/*` HAL/LLK: `api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`). No file-path kernel borrows — the factory instantiates only the op's own `writer_moreh_arange.cpp` / `writer_moreh_arange_rm.cpp` (`program_factory.cpp:16-19`, `67`). The host-side `device/moreh_arange_device_operation.cpp:8` includes the in-family `moreh_helper_functions.hpp`, but that is host code, not a kernel escape, and does not affect the port.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** `descriptor` concept, no op-owned tensors, no custom hash, no custom `override_runtime_arguments`, no pybind `create_descriptor`, no MeshWorkload. Target concept `MetalV2FactoryConcept`. `Op Classification` on the sheet is `PD (pointer-patching)`, consistent with the `Buffer*`-binding delivery of the output address.

## Misc anomalies  *(team-only, non-gating)*

- **Dead RTA on the tile path.** The factory unconditionally pushes `output.element_size()` as runtime arg index 5 for **both** kernel variants (`program_factory.cpp:95`). The RM kernel reads it (`writer_moreh_arange_rm.cpp:21`, used to compute the byte stride), but the **tile** kernel (`writer_moreh_arange.cpp`) reads only args 0–4 and never consumes arg 5 — it derives its byte count from `get_tile_size(cb_out)` instead. So for `untilize_out == false`, arg 5 is a dead runtime arg. Harmless; not porter work. (During the port the RTA set is reworked into named args regardless, so this naturally drops out — noted for the ops team.)

## Recipe notes  *(none)*
