# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow`

Single DeviceOperation, single program factory:

- **`MorehAbsPowOperation`**
  - `MorehAbsPowOperation (single-descriptor)` — `device/moreh_abs_pow_program_factory.cpp`

Kernels (all owned by the op directory, all factory-referenced):
- `device/kernels/reader_moreh_abs_pow.cpp`
- `device/kernels/writer_moreh_abs_pow.cpp`
- `device/kernels/moreh_abs_pow_kernel.cpp` (compute)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehAbsPowOperation` → `MorehAbsPowOperation (single-descriptor)` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (reader/writer/compute all on `Noc`/`DataflowBuffer`/`TensorAccessor`; donor `moreh_common.hpp` takes `DataflowBuffer`) |
| *Prereqs* — Cross-op escapes | Ok (shared-pool `moreh_common.hpp`, `DataflowBuffer` signatures — ✓ excellent) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (fixed CTAs) |
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
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `output` Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd arg) |
| *Port work* — CB endpoints | 1P+1C (c_0/c_1/c_2/c_3/c_16) · self-loop (c_24–c_27 intermediates) |
| *Safety* — Tensorless-at-dispatch (empty `tensor_args`) | **No** — `input` is a required `const Tensor&`; MeshDevice always sourceable. No framework block. |

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓ (all Appendix A entries N/A), TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean), Offset base pointers ✓, TensorAccessor 3rd arg ✓. Single `descriptor`-concept factory → target `MetalV2FactoryConcept`. No portable-subset scoping needed — the whole op is clear.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet ("Operations analysis", fetched fresh this run) row `moreh/moreh_abs_pow` / `MorehAbsPowOperation (single-descriptor)`: `Concept = descriptor`, `Is able to port? = yes`, `Is safe to port? = yes`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `TensorParameter relaxation = none`, `Op-owned tensors?` blank. Cross-check against code confirms every cheaply-checkable column:
  - `Concept = descriptor` — `create_descriptor()` returns `ProgramDescriptor` (`device/moreh_abs_pow_device_operation.hpp:34`, `device/moreh_abs_pow_program_factory.cpp:23`).
  - `Custom hash = no` — no `compute_program_hash` override anywhere in the op (grep clean).
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean).
  - `Pybind descriptor = no` — `moreh_abs_pow_nanobind.cpp:18` binds only the op entry point `ttnn::moreh_abs_pow` via `bind_function`; no `create_descriptor` / device-op `nb::class_`.
  - `Smuggled pointer = no` — the factory passes `input.buffer()` / `output.buffer()` (the `Buffer*`-binding form, framework auto-patched on cache hit — `device/moreh_abs_pow_program_factory.cpp:250,260`), not a raw `->address()` RTA.
  No cross-column invariant violated. Gate cleared for the single factory.

- **Device 2.0 (every kernel used):** GREEN. All three kernels are structurally Device 2.0:
  - `reader_moreh_abs_pow.cpp` — `Noc noc`, `DataflowBuffer dfb_*`, `TensorAccessor`, `noc.async_read(...)`. No legacy addr-gen (`InterleavedAddrGen`, raw `noc_async_read`, etc.).
  - `writer_moreh_abs_pow.cpp` — `Noc`, `DataflowBuffer`, `TensorAccessor`, `noc.async_write(...)`. No legacy idioms.
  - `moreh_abs_pow_kernel.cpp` (compute) — `DataflowBuffer` objects for all FIFO ops (`wait_front`/`pop_front`/`reserve_back`/`push_back`) and object-taking `*_with_dt` init helpers. The remaining raw-CB-index calls (`copy_tile(cb_x,…)`, `copy_tile(cb_mask_w,…)`, `mask_tile`, `abs_tile`) are standard compute-engine LLK primitives, not Device-2.0-DM idioms; the DFB token binds to them at Metal 2.0 port time (kernel-side whitelist).
  - `get_tile_size(cb_id)` (reader:49, writer:30) — sanctioned CB-index free function (Green bullet); not a holdover.
  - Donor `moreh_common.hpp` functions actually called — `fill_cb_with_value`, `generate_mask_w` (dataflow), `power_tile_to_cb`, `copy_tile_init_with_dt`, `pack_tile_with_dt` (compute) — all take `DataflowBuffer` and use `.get_write_ptr()` methods (Device 2.0 native). No pre-Device-2.0 donor shape.

  No violations table — nothing to route.

- **Feature compatibility:** every Appendix A entry scanned; all absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | plain `CBDescriptor`s only; no `.global_circular_buffer` field, no `remote_*`/`CreateGlobalCircularBuffer` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` set on any of the 9 `CBDescriptor`s (default 0) |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | CTAs are fixed: `TensorAccessorArgs` (reader/writer) + a single `num_units_per_core_group_*` value (compute). `tensor_args_t` is fixed-shape (`input` + optional `output`), not `std::vector<Tensor>`; no runtime-varying `get_compile_time_arg_val(i)` loop |

- **CB endpoints (GATE-free):** all resolvable at port time; nothing blocks. Nine CBs, one factory, no sharding branches:
  - `c_0` input — reader produces (`dfb_input` reserve/push), compute consumes (`dfb_x_obj` wait/pop) → **1P+1C**.
  - `c_1` one — reader produces (`fill_cb_with_value(dfb_one)`), compute consumes (`dfb_one_obj`) → **1P+1C**.
  - `c_2` decimal — reader produces (`fill_cb_with_value(dfb_decimal)`), compute consumes (`dfb_decimal_obj`) → **1P+1C**.
  - `c_3` mask_w — reader produces (`generate_mask_w`), compute consumes — **both gated on `do_mask_w` (= `origin_w % 32 != 0`)**. `(c_3, do_mask_w)` → **1P+1C**; `(c_3, !do_mask_w)` → neither kernel touches it (allocated-but-unused this config). Not a dead-CB drop — it is live under the mask config; bind it 1P+1C. See Misc anomalies re: the unconditional allocation.
  - `c_16` output — compute produces (`power_tile_to_cb` → `dfb_y_obj` pack/push), writer consumes (`dfb_output` wait/pop) → **1P+1C**.
  - `c_24`/`c_25`/`c_26`/`c_27` intermediates (`xabs`/`xpow`/`logx`/`exp_lxmd`) — touched **only** by the compute kernel (produced and consumed within it) → single toucher → **self-loop** (bind compute PRODUCER+CONSUMER; legal on Gen1).

- **Offset base pointers:** GREEN. Both address RTAs are clean bases — `input.buffer()` (reader RTA arg 0) and `output.buffer()` (writer RTA arg 0), delivered as `Buffer*` (no host-side `+ offset` fold). `tile_offset` (reader/writer RTA) is a **separate scalar tile index** used on-device only to compute a `page_id` for the `TensorAccessor` (`reader:53,55`, `writer:34,36`), never folded into a NoC base pointer. No Type 1/2/3/4 site. (Op is not in the `2026-07-19_offset_base_pointers.md` tables; scan confirms clean.)

- **TensorAccessor 3rd argument:** GREEN. Both accessor constructions pass **two** args only — `TensorAccessor(input_args, input_addr)` (`reader:27`) and `TensorAccessor(output_args, output_addr)` (`writer:24`). No explicit page-size 3rd argument anywhere. N/A. (Op not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table; nothing to classify.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input` — **Case 1** (via `TensorAccessor`). Factory pushes `input.buffer()` as reader RTA arg 0 (`Buffer*`-binding form); kernel reads `input_addr` and feeds `TensorAccessor(input_args, input_addr)` (`reader:12,27`). CT args carry `TensorAccessorArgs(*input.buffer())` (`program_factory.cpp:183`). Express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`.
  - `output` — **Case 1** (via `TensorAccessor`). Factory pushes `output.buffer()` as writer RTA arg 0; kernel feeds `TensorAccessor(output_args, output_addr)` (`writer:14,24`). CT args carry `TensorAccessorArgs(*output.buffer())` (`program_factory.cpp:194`). Express as `TensorParameter`/`TensorBinding`.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; op has no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24`,`c_25`,`c_26`,`c_27` (compute-only intermediates); 1P+1C `c_0`,`c_1`,`c_2`,`c_3`,`c_16` (`c_3` only under `do_mask_w`). All legal.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden-second-writer, no multi-reader, no dual-instance work-split. (Compute intermediates are ordinary self-loops.)
- **Cross-op / shared kernels:** the three `.cpp` kernels are all owned in-directory (file-path instantiated from the op's own `device/kernels/`). Function-call escapes: `#include "ttnn/kernel/dataflow/moreh_common.hpp"` (reader) and `#include "ttnn/kernel/compute/moreh_common.hpp"` (compute) — shared-pool headers under `ttnn/cpp/ttnn/kernel/`. Those helpers are broadly shared across the `moreh` family, so their Metal 2.0 rewrite (if any is needed) is a family-wide unit — but the functions this op calls already take `DataflowBuffer`, so no rewrite is forced here.
- **RTA varargs:** none — all kernels read a **fixed** run of args via a sequential `i++`/`input_id++` counter over a known set (reader 7, writer 5, compute 5); each is nameable. Not the loop-indexed vararg pattern.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean.
  - **Summary table (function-call escapes):**

    | Op kernel | Donor file | Donor class |
    |---|---|---|
    | `reader_moreh_abs_pow.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared kernel pool (`ttnn/cpp/ttnn/kernel/`, class 3) |
    | `moreh_abs_pow_kernel.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared kernel pool (class 3) |

    Also included: `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`, `api/dataflow/dataflow_api.h`, `api/debug/dprint.h` — all `tt_metal/*` (LLK/HAL, class 1, no concern).
  - **Per-call detail:** functions called — `fill_cb_with_value(DataflowBuffer,…)`, `generate_mask_w(DataflowBuffer,…)`, `power_tile_to_cb(DataflowBuffer…)`, `copy_tile_init_with_dt(DataflowBuffer)`, `pack_tile_with_dt(uint32_t, DataflowBuffer)`. All handle shapes are `DataflowBuffer` (✓ excellent — Device 2.0 native). No `CircularBuffer&`, no `uint32_t cb_id`/`sem_id`, no old-style addr-gen. No ⚠/✗/⭐ entries.
  - **Borrowed kernel files (file-path instantiation):** none — the factory instantiates only its own three kernels (`READER_KERNEL_PATH`/`WRITER_KERNEL_PATH`/`COMPUTE_KERNEL_PATH`, `program_factory.cpp:16-21`).
- **TTNN factory analysis:** current concept `descriptor`; op-owned tensors none; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **`c_3` (mask_w) CB is allocated unconditionally but only used when `do_mask_w` (`origin_w % 32 != 0`)** — `program_factory.cpp:124-132` always pushes the `c_3` `CBDescriptor`, yet both the reader's `generate_mask_w` (`reader:40-43`) and compute's `wait_front`/`pop_front` (`compute:59-61,104-106`) are gated on `do_mask_w`. Under a tile-aligned `origin_w` the CB's 1-tile L1 is allocated and never touched. Harmless (correct numerics), minor L1 waste. Not porter work — the CB is genuinely live under the mask config, so it ports as an ordinary 1P+1C binding, not a dead-CB drop.
- **`writer_moreh_abs_pow.cpp:15` `output_is_dram`** and **`reader_moreh_abs_pow.cpp:13` `input_is_dram`** are unpacked from RTAs (arg 1 in each) but never used in the kernel body (addressing goes entirely through the `TensorAccessor`). Dead RTAs — the factory still emits `static_cast<uint32_t>(is_dram(...))` for both (`program_factory.cpp:251,260`). Team-only; not porter work.

## Recipe notes

- The readiness-sheet fetch (`download_file_content` via the Google Drive MCP connector) **succeeded from within this spawned audit agent** — the doc `ttnn_op_porting_readiness.md` and the audit both warn that the connector "authorizes only in the main interactive session" and a subagent "hits the OAuth wall." That did not happen here; the CSV downloaded and decoded normally. Either the warning is stale, or this agent inherited the parent session's authorization. Worth reconciling the doc's wording with actual behavior. (The decoded CSV was used for the lookup only and then deleted — not committed, per the doc's standing rule, and it lives outside the op directory.)
