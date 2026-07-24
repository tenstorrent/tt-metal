# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/plusone`

Single device operation, single program factory:

- **`PlusOneDeviceOperation`** (`device/plusone_device_operation.hpp` / `.cpp`)
  - `PlusOneProgramFactory` (`device/plusone_program_factory.cpp`) — `create_descriptor()` → `ProgramDescriptor`
    - kernel: `device/kernels/reader_plusone_interleaved.cpp` (op-owned; the only kernel, file-path-instantiated)

No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/plusone` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `PlusOneDeviceOperation` → `PlusOneProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — reader kernel is structurally Device 2.0 (`Noc`, `CircularBuffer`, `TensorAccessor`); no holdovers |
| *Prereqs* — Cross-op escapes | Ok — no donors; only kernel is op-owned; includes are `tt_metal/*` LLK/HAL |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (fixed 6 scalars + `TensorAccessorArgs`) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean base — `Buffer*` delivered, no offset fold) |
| *Port work* — Tensor bindings (per binding) | input: Case 1 (interleaved/DRAM) · clean borrowed-DFB (sharded) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (accessor is 2-arg) |
| *Port work* — CB endpoints | self-loop (`c_0`, both configs — one toucher) |

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean), Offset base pointers ✓ (no fold), TensorAccessor 3rd arg ✓ (no 3rd arg present). The port is a small, mechanical single-kernel `descriptor` → `MetalV2FactoryConcept` conversion. Port work is limited to one tensor binding (Case 1 / borrowed-DFB per config) and a self-loop on the single CB.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet row (`experimental/plusone`, `PlusOneDeviceOperation`, `PlusOneProgramFactory`): `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Is able to port? = yes`. Cross-check against the code all agree:
  - `Concept = descriptor` — `PlusOneProgramFactory::create_descriptor()` returns `ProgramDescriptor` (`device/plusone_program_factory.cpp:22`).
  - `Custom hash = no` — no `compute_program_hash` override in the device op (`device/plusone_device_operation.cpp`).
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` anywhere.
  - `Pybind descriptor = no` — `plusone_nanobind.cpp` binds the free function `plus_one` only; no `nb::class_` of the device op and no `create_descriptor` binding.
  - `Op-owned tensors = no` (blank) — consistent with the `descriptor` concept (cross-column invariant holds).
- **Device 2.0 (every kernel used):** GREEN. The op uses exactly one kernel, `device/kernels/reader_plusone_interleaved.cpp` (op-owned). It is structurally Device 2.0: `Noc noc;` with `noc.async_read` / `noc.async_read_barrier` / `noc.async_write` / `noc.async_write_barrier` (lines 14, 36–37, 52–53), `TensorAccessor s0(s0_args, src_addr)` (line 26), `CircularBuffer cb_in0(cb_id_in0)` (line 28), and the wrapper method `cb_in0.get_write_ptr()` (line 31, not the free-function holdover). Includes are all `api/dataflow/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` — `tt_metal/*` LLK/HAL. No CB-index free-function holdovers, no Device 1.0 idioms (no raw `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw sem addresses). No donor kernels.

- **Feature compatibility:** every Appendix A entry scanned; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | The sharded-path CB sets `.buffer = src_buffer` (`device/plusone_program_factory.cpp:65`) — a borrowed-memory DFB (`set_globally_allocated_address`-equivalent, a mechanical `borrowed_from` translation), **not** a GlobalCircularBuffer. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` set on the `CBDescriptor`; default 0. |
  | GlobalSemaphore | N/A | No semaphores of any kind. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `compile_time_args` is a fixed list of 6 scalars (`device/plusone_program_factory.cpp:68-69`) plus `TensorAccessorArgs` (fixed-count). Kernel reads CTAs 0–5 as distinct `constexpr` (lines 18–23) then `TensorAccessorArgs<6>()` — no runtime-varying `get_compile_time_arg_val(i)` loop. |

- **CB endpoints (GATE-free):** one CB, `src0_cb_index = c_0` (`device/plusone_program_factory.cpp:45,57`). The single reader kernel is the only toucher on the single core, via a raw peek `cb_in0.get_write_ptr()` (`reader...cpp:31`) used as L1 scratch (interleaved/DRAM) or as the in-place borrowed buffer (sharded). **One toucher → self-loop** (bind the reader PRODUCER **and** CONSUMER; legal on Gen1 for a DM kernel). Applies to both configs. Not dead (index referenced), no multi-binding.

- **Offset base pointers:** GREEN. The factory delivers the input via the `Buffer*`-binding form — `reader_desc.emplace_runtime_args(core, {src_buffer})` (`device/plusone_program_factory.cpp:85`), passing the `Buffer*` object, **not** `->address()` and with **no** host-folded offset. The kernel reads `src_addr = get_arg_val<uint32_t>(0)` (`reader...cpp:16`) and hands it to the accessor as a clean base. No Type 1/2 fold. (Op not listed in the offset-base-pointer triage doc; my scan confirms clean.)

- **TensorAccessor 3rd argument:** GREEN — N/A. The only accessor is `TensorAccessor(s0_args, src_addr)` (`reader...cpp:26`) — two arguments, no page-size override. No site to classify. (Op not listed in the 3rd-arg triage doc; consistent.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **input** — delivered as a `Buffer*` binding (`emplace_runtime_args(core, {src_buffer})`, `device/plusone_program_factory.cpp:85`). Classified by kernel use, per config:
    - **interleaved / DRAM** config: the base (`src_addr`) is fed into `TensorAccessor(s0_args, src_addr)` and used via `noc.async_read(s0, …)` / `noc.async_write(…, s0, …)` → **Case 1** (via `TensorAccessor`). Express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA and `TensorAccessorArgs` plumbing disappear.
    - **sharded (L1)** config: the CB is `borrowed_from` the input buffer (`.buffer = src_buffer`, line 65); `src0_is_dram` is false so the accessor path is skipped and the kernel reads/writes the borrowed CB memory in place (`reader...cpp:31-32,39-50`) → **clean** (borrowed-memory DFB; causal-link gate). Port via `DataflowBufferSpec::borrowed_from` on the same `TensorParameter`.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_0` (both configs — single toucher).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — single kernel, single toucher.
- **Cross-op / shared kernels:** none — the sole kernel is op-owned and file-path-instantiated from the op's own `device/kernels/`; no borrowed/donor kernels, no function-call escapes.
- **RTA varargs:** none — the only RTA is the single fixed scalar `src_addr` (`get_arg_val<uint32_t>(0)`); no loop-indexed or data-selected reads.

## Team-only

- **Out-of-directory coupling & donor shape:** clean. No `#include` resolves outside the op directory except `tt_metal/*` LLK/HAL headers (donor class 1 — no concern). The program factory file-path-instantiates only its own kernel. No port-together coupling.
- **TTNN factory analysis:** current concept `descriptor`; no op-owned tensors; no custom hash; no custom `override_runtime_arguments`; no pybind `create_descriptor`; no smuggled pointer (delivery is the framework-patched `Buffer*` form). Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Interleaved-in-L1 input appears unhandled.** The reader only DMAs the input into/out of the scratch CB when `src0_is_dram` is true (`reader...cpp:35-38,51-54`). For a non-sharded input the CB is plain scratch (`.buffer = nullptr`, `device/plusone_program_factory.cpp:65`). So an L1-*interleaved* input (not DRAM, not sharded) would neither be read into the CB nor written back — the kernel would increment uninitialized scratch. `validate_on_program_cache_miss` (`device/plusone_device_operation.cpp:9-19`) constrains dtype/layout/rank but not memory config, and the docstring only describes DRAM-interleaved and L1-sharded. Pre-existing behavior, unrelated to the port; flagged for the ops team. The port must preserve this behavior exactly (zero functional change).
- **`src_addr` RTA is dead in the sharded path** — constructed into `s0` but never used when `src0_is_dram` is false. Harmless; noted only because the auditor reads every line.

## Recipe notes

- None. The op is small and every subject resolved cleanly against the recipe.
