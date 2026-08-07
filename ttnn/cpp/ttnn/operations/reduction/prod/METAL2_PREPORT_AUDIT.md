# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/prod`

This directory bundles **two** device operations that implement `ttnn.prod`. They share the op directory *and* share a donor kernel (both use `eltwise/unary`'s `writer_unary_interleaved_start_id.cpp`), so they are audited together as one porting unit. Per-DeviceOperation attribution is retained where findings differ (see the *Per-DeviceOperation attribution* section).

- **`ProdAllDeviceOperation`** (`ttnn::prim`) — full-tensor product on a single core.
  - `ProdAllProgramFactory` (`device/prod_all_program_factory.cpp`)
- **`ProdNcDeviceOperation`** (`ttnn::prim`) — reduction over dim 0 or 1, multi-core.
  - `ProdNcProgramFactory` (`device/prod_nc_program_factory.cpp`)

Host wrappers `tt::operations::primary::prod_all` / `prod_nc` (`device/prod_op_all.*`, `device/prod_nc_op.*`) are thin shims that call `ttnn::prim::prod_all` / `ttnn::prim::prod_nc`; the real device ops are the two `*_device_operation` classes above.

**Unreferenced files (out of scope):** `device/kernels/dataflow/utils.hpp` is included by no bound kernel (grep confirms only `ttnn.egg-info/SOURCES.txt` references it) — dead code, not audited. It *does* contain a raw `cb.get_write_ptr()` write, but since no kernel includes it, it is not an endpoint anywhere.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `37f03926088 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/prod` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `ProdAllDeviceOperation` → `ProdAllProgramFactory`; `ProdNcDeviceOperation` → `ProdNcProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 5 referenced kernels are Device 2.0 (own + donor); only sanctioned free functions in use |
| *Prereqs* — Cross-op escapes | Ok — no kernel-side function-call escapes; file-path borrow of 2 broadly-shared `eltwise/unary` kernels (port-together coupling, non-gating) |
| *Feature Support* — overall | **GREEN** — all Appendix A entries N/A |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes (both) |
| *TTNN Readiness* — Custom hash | No (both) — cross-checked: no `compute_program_hash` override |
| *TTNN Readiness* — Runtime-args update | No (both) — cross-checked: no `get_dynamic_runtime_args`/`override_runtime_arguments` |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `prod_nanobind.cpp` binds only `ttnn::prod` overloads |
| *TTNN Readiness* — Op-owned tensors | No (both) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none — no `->address()` fold anywhere; tensors bound directly |
| *Port work* — Tensor bindings (per binding) | all **Case 1** (`TensorAccessor`) — 2 bindings per factory |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — every `TensorAccessor` is 2-arg |
| *Port work* — CB endpoints | all **legal 1:1** — `c_0`, `c_3` in both factories |

**CB endpoints** are dispositions, not gates. Here every CB is a plain 1 producer + 1 consumer on every node — nothing needs a self-loop, 1P+1C assignment, multi-binding flag, or dead-CB drop.

## Result

**GREEN → brief issued.** Both `ProdAllDeviceOperation`/`ProdAllProgramFactory` and `ProdNcDeviceOperation`/`ProdNcProgramFactory` clear all five gates:

- **Device 2.0** ✓ — every referenced kernel (own reader/compute + the two borrowed `eltwise/unary` dataflow kernels) is Device 2.0-native.
- **Feature compatibility** ✓ — no `GlobalCircularBuffer`, no `address_offset`, no `GlobalSemaphore`, no CTA varargs.
- **TTNN factory concept** ✓ — sheet says `Is able to port? == yes` for both rows; code cross-check clean.
- **Offset base pointers** ✓ — no host-folded offset; addresses arrive via tensor bindings, tile offsets are separate page-index scalars.
- **TensorAccessor 3rd argument** ✓ — no site passes a page-size argument.

Port work is entirely mechanical: express the 2 tensor bindings per factory as `TensorParameter`/`TensorBinding` (all Case 1), no relaxations, no CB endpoint fixes. See `METAL2_PORT_BRIEF.md`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both factories. The readiness sheet ("Operations analysis", fetched fresh this run) rows for `reduction/prod`:
  - `ProdAllDeviceOperation` / `ProdAllProgramFactory`: `Concept=descriptor`, `Custom hash=no`, `Runtime-args update=no`, `Override runtime args method?=no`, `Pybind descriptor=no`, `Smuggled pointer=no`, `Is safe to port?=yes`, **`Is able to port?=yes`**, `TensorParameter relaxation=none`, `Op-owned tensors?=` (empty).
  - `ProdNcDeviceOperation` / `ProdNcProgramFactory`: identical column values; **`Is able to port?=yes`**.

  Cross-check against code (cheaply-checkable columns):
  - `Concept=descriptor` ✓ — both factories expose `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor` (`prod_all_device_operation.hpp:22`, `prod_nc_device_operation.hpp:24`).
  - `Custom hash=no` ✓ — no `compute_program_hash` override in the op dir.
  - `Runtime-args update=no` ✓ — no `get_dynamic_runtime_args`/`override_runtime_arguments` in the op dir.
  - `Pybind descriptor=no` ✓ — `prod_nanobind.cpp:88-109` binds only the two `ttnn::prod` function overloads; no `create_descriptor` binding.
  - `Op-owned tensors?=`(empty)/no ✓ — cross-column invariant holds (a `descriptor`-concept row cannot carry op-owned tensors).

  No conflict between sheet and code; no missing row. Gate cleared.

- **Device 2.0 (every kernel used):** **GREEN.** Five kernels are referenced across the two factories; all are Device 2.0-native (object-oriented `Noc` / `CircularBuffer` / `DataflowBuffer`, `TensorAccessor`), with only *sanctioned* CB-index free functions (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`).

  | Kernel file | Used by | Owner | Device 2.0? | Notes |
  |---|---|---|---|---|
  | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | prod_all reader | eltwise/unary (donor) | ✓ | `Noc`, `DataflowBuffer`, `TensorAccessor`; `get_local_cb_interface(cb_id_in0).fifo_page_size` (sanctioned) |
  | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | prod_all + prod_nc writer | eltwise/unary (donor) | ✓ | `Noc`, `DataflowBuffer`, `TensorAccessor`; `get_local_cb_interface(cb_id_out)` (sanctioned) |
  | `reduction/prod/device/kernels/dataflow/reader_prod_nc.cpp` | prod_nc reader | prod (own) | ✓ | `Noc noc;`, `CircularBuffer cb_in0`, `TensorAccessor`; `get_tile_size(cb_in0.get_cb_id())` (sanctioned) |
  | `reduction/prod/device/kernels/compute/prod_all.cpp` | prod_all compute | prod (own) | ✓ | Compute kernel — `CircularBuffer` objects for FIFO; CB-index args to compute LLK APIs are standard |
  | `reduction/prod/device/kernels/compute/prod_nc.cpp` | prod_nc compute | prod (own) | ✓ | Compute kernel — `CircularBuffer` objects for FIFO |

- **Feature compatibility:** every Appendix A entry, in order. All absent → all N/A. Clean scan.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, `.global_circular_buffer` field, `remote_cb`/`remote_index`, or `CreateCircularBuffer(..., global_cb)` anywhere. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` / `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress`. CBs are plain `CBDescriptor` literals with default (zero) offset. |
  | GlobalSemaphore | N/A | No semaphores of any kind; no `GlobalSemaphore`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` carries fixed named tensors (no `std::vector<Tensor>`). Kernels read CTAs only at constexpr offsets (`get_compile_time_arg_val(0)` in `reader_prod_nc.cpp`/`writer_unary_*`; none in a runtime-varying index loop). |

- **CB endpoints (GATE-free):** **all legal 1:1.** Both factories declare exactly two CBs — `c_0` (input) and `c_3` (output). Census per node:
  - **prod_all** (single core `{0,0}`): `c_0` — donor reader produces (`dfb.reserve_back`/`push_back`), compute consumes (`input_cb_obj.wait_front`/`pop_front`) → 1P+1C legal. `c_3` — compute produces (`final_output_cb_obj.reserve_back`/`push_back`), donor writer consumes (`dfb.wait_front`/`pop_front`) → 1P+1C legal.
  - **prod_nc** (multi-core, split across `core_group_1`/`core_group_2`): `c_0` — own reader produces (`cb_in0.reserve_back`/`push_back`), compute consumes → 1P+1C legal. `c_3` — compute produces (`cb_out0_obj.reserve_back`/`push_back`), donor writer consumes → 1P+1C legal. The compute kernel is instantiated as `compute_desc_1`/`compute_desc_2` over **disjoint** core groups (each node sees exactly one compute instance — the per-group split, *not* a dual-instance work-split), so per-node census is unchanged.
  - No hidden second writer (no raw `get_write_ptr`/`fifo_wr_ptr` co-fill on any bound kernel), no multi-reader, no dead CB. Nothing blocks a Gen1 port and nothing needs a special disposition.

- **Offset base pointers:** **GREEN.** No `->address()` expression exists anywhere in the op; addresses reach kernels as tensor bindings (`emplace_runtime_args(core, {input.mesh_tensor(), ...})` / `{output, ...}`, `prod_all_program_factory.cpp:103-104`, `prod_nc_program_factory.cpp:189,200`). The reader's tile-navigation scalars (`input_tile_offset`, `start_id`, `HtWt`, `CHtWt`) are **page indices** consumed as `{.page_id = ...}` by the `TensorAccessor` (`reader_prod_nc.cpp:41`), never folded into the base pointer — so there is no offset base. prod is **not** in the `2026-07-19_offset_base_pointers.md` tables (confirmed: the single "prod" grep hit is the word "produces"), consistent with a clean scan.

- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` construction is 2-arg: `TensorAccessor(dram_input_addrg_args, input_addr)` (`reader_prod_nc.cpp:29`), `TensorAccessor(src_args, src_addr)` (donor reader:25), `TensorAccessor(dst_args, dst_addr)` (donor writer:31). No page-size override site exists, so the subject cannot fire. (The dated `2026-07-06_tensor_accessor_3rd_arg_triage.md` was not present at the analyses path this run; immaterial — a 3rd-arg site is a syntactic signal and none is present.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, both factories):
  - **prod_all** `input` — **Case 1** (fed to `TensorAccessor` in donor reader). `output` — **Case 1** (fed to `TensorAccessor` in donor writer).
  - **prod_nc** `input` — **Case 1** (fed to `TensorAccessor` in `reader_prod_nc.cpp`). `output` — **Case 1** (fed to `TensorAccessor` in donor writer).
  - All four are the low-risk mechanical case: express as `TensorParameter`/`TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` and the address-via-RTA + its `TensorAccessorArgs` plumbing disappear. No Case 2 (raw-pointer) bindings.
- **TensorParameter relaxation:** none (sheet `none`; no custom hash to reconcile).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal — no disposition needed.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no multi-binding CBs.
- **Cross-op / shared kernels:** prod borrows 2 kernels from `eltwise/unary` by file path — `writer_unary_interleaved_start_id.cpp` (used by both factories) and `reader_unary_interleaved_start_id.cpp` (prod_all only). Both are **broadly shared** (writer instantiated by ~29 op families, reader by ~12), so their Metal 2.0 rewrite is a single change the whole co-borrower set must adopt together (port-together set). Both are already Device 2.0-native, so no Device 2.0 blocker rides on them. See *Team-only → Out-of-directory coupling*.
- **RTA varargs:** none — every kernel reads its runtime args as a fixed set of distinct named fields at constexpr offsets (no counted `arg_index++` loop, no data-selected index).

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Function-call escape (kernel `#include`s outside the op dir):** **✓ clean.** The only non-`api/` includes in prod's kernels are `<cstdint>` / `<stdint.h>`. No cross-op helper functions are called; no per-call shape analysis is needed.
  - **File-path kernel instantiation (borrowed kernels):** two, both from cross-family `eltwise/unary`:
    | Kernel file | Owner | Borrowed by prod factory | Broadly shared? |
    |---|---|---|---|
    | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary | prod_all + prod_nc writer | Yes — ~29 op families (typecast, bcast, concat, copy, permute, reshape_on_device, slice, tilize, tilize_with_val_padding, transpose, embedding, …) |
    | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | eltwise/unary | prod_all reader | Yes — ~12 op families |
    Coupling implication: these two kernels' Metal 2.0 rewrite (CB→DFB, named-token bindings) is a single shared rewrite; every co-borrower must migrate in lockstep. Not a gate (both are Device 2.0-clean today), but a sequencing fact for planners.
- **Relaxation candidates (mined from a custom hash):** N/A — neither factory has a custom hash.
- **TTNN factory analysis:** both factories are vanilla single-program `descriptor` concept. No op-owned tensors, no MeshWorkload, no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept` for both. Evidence: `prod_all_device_operation.hpp:15-40`, `prod_nc_device_operation.hpp:17-42`, `prod_nanobind.cpp:88-109`.

## Misc anomalies  *(team-only, non-gating)*

Latent issues noticed while auditing — routed to the ops team, not porter work. The port does not act on these:

- **Dead reader RTA `dim` in prod_nc.** `prod_nc_program_factory.cpp:197-198` appends `static_cast<uint32_t>(dim)` as reader runtime arg index 7, but `reader_prod_nc.cpp` reads `dim` from a **compile-time** arg (`get_compile_time_arg_val(0)`, line 20) and only consumes runtime args 0–6. Runtime arg 7 is never read — dead.
- **Dead writer RTA `is_dram` in prod_nc.** `prod_nc_program_factory.cpp:205` appends `is_dram(output)` as writer runtime arg index 3, but the donor `writer_unary_interleaved_start_id.cpp` reads only args 0–2 (`dst_addr`, `num_pages`, `start_id`). Arg 3 is never read — dead.
- **Dead compute CTA in prod_nc.** `prod_nc_program_factory.cpp:136,151` (and `:160,165` for group 2) pass `num_cols_per_core_group_*` as the compute kernel's `compile_time_args`, but `prod_nc.cpp` reads no compile-time arg — it gets its tile count from runtime arg 1. The CTA is unread (and, with the default hash, still contributes to the cache key). Redundant with the RTA.
- **Output CB uses index `c_3`, not the `c_16`+ output convention.** Both factories put the output CB at `CBIndex::c_3` (`prod_all_program_factory.cpp:44`, `prod_nc_program_factory.cpp:101`). Functionally valid, but deviates from the inputs-`c_0..15` / outputs-`c_16..31` convention. Cosmetic; no action required for the port.

## Per-DeviceOperation attribution

Findings are largely identical across the two bundled device ops; differences:

| Field | `ProdAllDeviceOperation` / `ProdAllProgramFactory` | `ProdNcDeviceOperation` / `ProdNcProgramFactory` |
|---|---|---|
| Overall gate verdict | GREEN | GREEN |
| Reader kernel | donor `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | own `reduction/prod/.../reader_prod_nc.cpp` |
| Writer kernel | donor `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | donor `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` |
| Compute kernel | own `prod_all.cpp` | own `prod_nc.cpp` (per-core-group split into 2 instances over disjoint ranges) |
| Cores | single `{0,0}` | multi-core via `split_work_to_cores` |
| Tensor bindings | input Case 1, output Case 1 | input Case 1, output Case 1 |
| CB endpoints | `c_0`/`c_3` legal 1:1 | `c_0`/`c_3` legal 1:1 |
| Misc anomalies | none | dead reader RTA `dim`, dead writer RTA `is_dram`, dead compute CTA |

## Recipe notes

- The prod directory contains both legacy-named wrapper files (`prod_op_all.*`, `prod_nc_op.*`, namespace `tt::operations::primary`) and the real modern device ops (`*_device_operation.*`, namespace `ttnn::prim`). This could momentarily read as a legacy-imperative op. The recipe's "confirm the resolved directory really is an op" guidance held up — following the call chain (`prod.cpp` → `prod_op_all.cpp` → `ttnn::prim::prod_all`) resolves it cleanly. No recipe change needed; noting in case other reduction/moreh-derived ops carry the same dual-file shape and trip a future auditor.
