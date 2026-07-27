# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward`

- **`MorehSumBackwardOperation`** (single DeviceOperation)
  - `MorehSumBackwardOperation (single-descriptor)` — `device/moreh_sum_backward_program_factory.cpp` (`create_descriptor`)

Kernels referenced by the factory (all owned by the op, under `device/kernels/`):
- `reader_moreh_sum_backward.cpp` (reader / `ReaderConfigDescriptor`)
- `writer_moreh_sum_backward.cpp` (writer / `WriterConfigDescriptor`)
- `moreh_sum_backward.cpp` (compute; instantiated once per core group)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehSumBackwardOperation` → single `descriptor` factory (`create_descriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (own kernels already on Metal-2.0 kernel idioms; donor `moreh_common.hpp` clean) |
| *Prereqs* — Cross-op escapes | Ok (shared-lib header only; no borrowed kernel files) |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (fixed-count CTAs; no runtime-varying CTA index) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases; `Buffer*` + separate scalar offset) |
| *Port work* — Tensor bindings (per binding) | `output_grad` Case 1 · `input_grad` (output) Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (all `TensorAccessor` sites are 2-arg) |
| *Port work* — CB endpoints | all legal 1:1 (`c_0`, `c_1`, `c_16`) |

## Tensorless-dispatch check (orchestrator-requested)

**PASS — not a framework block.** `tensor_args_t` (`device/moreh_sum_backward_device_operation.hpp:20-24`) carries `const Tensor& output_grad` as a **mandatory, non-optional** field (only `input` and `input_grad` are `std::optional`). Dispatch therefore always presents at least `output_grad` in `tensor_args`, and the device is sourced from `output_grad.device()` (`device_operation.cpp:120`, `program_factory.cpp:80`). The MetalV2 factory adapter's MeshDevice-from-tensor lookup will always find `output_grad`. There is no optional-only / tensorless-at-dispatch path.

## Result

**GREEN → brief issued.** `METAL2_PORT_BRIEF.md` written alongside this file. Every gate cleared: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd arg ✓. Single `descriptor` factory → target `MetalV2FactoryConcept`. No portable-subset scoping needed (whole op is clear).

Notable context: the op's own kernels are already written against the Metal-2.0 kernel-side API (`api/dataflow/dataflow_buffer.h`, `DataflowBuffer`, `Noc`, `TensorAccessor`), so the port is a host-side factory rewrite (`ProgramDescriptor`/`CBDescriptor`/`KernelDescriptor` → `MetalV2FactoryConcept` spec) with minimal kernel-token rebinding.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Fresh "Operations analysis" sheet fetched this run; single row `moreh/moreh_sum_backward` → `MorehSumBackwardOperation (single-descriptor)`: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, **`Is able to port? = yes`**. Cross-check clean:
  - `Concept = descriptor` — confirmed: `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device_operation.hpp:29`, `program_factory.cpp:66`).
  - `Custom hash = no` — confirmed: no `compute_program_hash` override in the op.
  - `Runtime-args update = no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments`.
  - `Pybind descriptor = no` — confirmed: `moreh_sum_backward_nanobind.cpp` binds only the `moreh_sum_backward` function, no `create_descriptor`.
  - No cross-column invariant violations (no op-owned tensors on a `descriptor` row; no runtime-args-update).
- **Device 2.0 (every kernel used):** GREEN. All three op kernels and the donor library functions they call use Device-2.0 (indeed Metal-2.0) idioms — `Noc` objects (`noc.async_read`/`async_write`/`*_barrier`), `DataflowBuffer` objects (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr()`), and `TensorAccessor`. The only CB-index free functions in use are sanctioned metadata lookups:
  - `get_tile_size(cb_id)` — `reader_moreh_sum_backward.cpp:88`, `writer_moreh_sum_backward.cpp:27` — explicitly sanctioned by the recipe.
  - `get_dataformat(cb.get_id())` — donor `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:100` (reached via `fill_cb_with_value`). Not in the recipe's explicitly-named sanctioned pair, but verified de-facto sanctioned: the free-function form is used pervasively across already-migrated kernels — including Metal-2.0 `*_metal2.cpp` kernels, one passing a Metal-2.0 binding token directly (`reader_unary_transpose_wh_universal_input_cols_partitioned_metal2.cpp:32` → `get_dataformat(dfb::in)`). Per the recipe's "if Device 2.0 allows the free function, so do we," this is not a holdover. (See Recipe notes.)
- **Feature compatibility:** all Appendix A entries N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `global_circular_buffer` field, no `remote_index`/`remote_cb` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | Three `CBDescriptor`s (`c_0`,`c_1`,`c_16`); none set `address_offset` (default 0) |
  | GlobalSemaphore | N/A | Op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Reader CTAs = `{input_grad_rank}` + `TensorAccessorArgs`, read at constexpr offsets; variable-count loops read **RTAs**, not CTAs. `tensor_args_t` is fixed 3-field, no `std::vector<Tensor>` |

- **CB endpoints (GATE-free):** all three CBs are legal 1-producer/1-consumer; no self-loop, 1P+1C-assign, multi-binding, or dead-CB disposition needed. Precondition met (kernels are structurally Device 2.0).
  - `c_0` (input, `program_factory.cpp:144`): reader FIFO-produces (`reserve_back`/`push_back`, `reader...cpp:94,97`); compute FIFO-consumes (`wait_front`/`pop_front`, `moreh_sum_backward.cpp:28,46`). 1 locked P + 1 locked C → legal.
  - `c_1` (zero tile, `program_factory.cpp:153`): reader produces via `fill_cb_with_value` (`reserve_back`+`push_back`, `moreh_common.hpp:99,108`); compute consumes (`wait_front`/`pop_front`, `moreh_sum_backward.cpp:23,48`). 1 P + 1 C → legal.
  - `c_16` (output, `program_factory.cpp:162`): compute produces (`reserve_back`/`push_back`, `moreh_sum_backward.cpp:41,45`); writer consumes (`wait_front`/`pop_front`, `writer...cpp:31,35`). 1 P + 1 C → legal.
  - Config note: the op has a single code path (interleaved tile I/O; two compute core groups differing only in tile count) — no config-dependent endpoint flips.
- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into its base. Both address args are delivered as clean `Buffer*` bindings with the tile offset passed as a **separate** scalar:
  - reader: `reader_rt_args.push_back(output_grad.buffer())` then separate `num_tiles_per_core`, `tile_offset` (`program_factory.cpp:252-254`). Kernel consumes `start_id`/`num_output_tiles` as independent scalars; tile index is recomputed on-device from the dim/stride args (`reader...cpp:90-92`).
  - writer: `writer_desc.emplace_runtime_args(core, {input_grad.buffer(), num_tiles_per_core, tile_offset})` (`program_factory.cpp:261`), clean base + separate offset.
  - Not in the dated offset-base triage tables, and the scan confirms no fold → clean.
- **TensorAccessor 3rd argument:** GREEN — both `TensorAccessor` construction sites are 2-arg (`args`, `base_addr`) with no explicit page-size: `reader...cpp:84`, `writer...cpp:23`. Subject does not fire.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `output_grad` — **Case 1** (via `TensorAccessor`). Host delivers `output_grad.buffer()` as a `Buffer*` RTA (`program_factory.cpp:252`); kernel receives the base as `uint32_t` and feeds it into `TensorAccessor(output_grad_args, output_grad_addr)` (`reader...cpp:84`), doing all reads through the accessor (`noc.async_read(output_grad_addrg, ...)`, `reader...cpp:95`). Express as a `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA and the `TensorAccessorArgs<1>` CTA plumbing (`program_factory.cpp:176`, `reader...cpp:36`) both disappear.
  - `input_grad` (the output tensor) — **Case 1** (via `TensorAccessor`). Host delivers `input_grad.buffer()` as a `Buffer*` RTA (`program_factory.cpp:261`); kernel feeds it into `TensorAccessor(input_grad_args, input_grad_addr)` (`writer...cpp:23`) and writes through it. Same treatment; the `TensorAccessorArgs<0>` CTA (`program_factory.cpp:187`, `writer...cpp:12`) disappears.
  - Note: both are delivered via the `Buffer*`-binding form (framework auto-registers a `BufferBinding`, patched on cache hits) — correct-on-cache-hit today, **not** the silent-wrong `->address()`-RTA hazard. Recorded as routine port work.
- **TensorParameter relaxation:** none (sheet: `none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal — no disposition.

## Heads-ups  *(mirrors the brief)*

- **RTA varargs (FYI-P):** `reader_moreh_sum_backward.cpp:44-57` reads **three variable-count RTA blocks** — `output_grad_dim`, `input_grad_dim`, `need_bcast_dim`, each of length `input_grad_rank` — in `for (i < input_grad_rank)` loops via `arg_fetcher.get_next_arg_val<uint32_t>()`. The count is bounded by a CTA (`input_grad_rank`) but still varies per instantiation, so these are genuine RTA-vararg blocks (recipe RTA-varargs shape (a)); port them via the kernel-side vararg mechanism, not by naming each element. The three leading reader RTAs (`output_grad_addr`, `num_output_tiles`, `start_id`, lines 40-42) and all three writer RTAs (`writer...cpp:16-18`) are fixed distinct fields → name those normally.
- **Cross-op / shared kernels:** the op instantiates only its own three kernel files (no borrowed/file-path kernels). Kernels `#include` the shared moreh helper library `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (reader/writer: `ArgFetcher`, `fill_cb_with_value`) and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (compute). These are shared-kernel-pool (`ttnn/cpp/ttnn/kernel/`) headers; already on Metal-2.0 idioms; any rewrite of them is a lib-team concern shared with the rest of the moreh family — see Team-only.

## Team-only

- **Out-of-directory coupling & donor shape (FYI-U):** roll-up **✓ clean**.

  | Op kernel | Donor include | Donor class | Functions used | Shape |
  |---|---|---|---|---|
  | reader, writer | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared pool (`ttnn/cpp/ttnn/kernel/`, class 3) | `ArgFetcher::get_next_arg_val`, `fill_cb_with_value(DataflowBuffer, uint32_t)` | ✓ excellent — `DataflowBuffer` by value (Device 2.0 native); `get_next_arg_val` wraps `get_arg_val` |
  | reader, writer | `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | `tt_metal/*` (class 1) | LLK/HAL | ✓ no concern |
  | compute | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared pool (class 3) | compute helpers / LLK includes | ✓ excellent (compute side; no data-movement idioms) |
  | compute | `api/dataflow/dataflow_buffer.h` | `tt_metal/*` (class 1) | `DataflowBuffer` | ✓ no concern |

  No function-call escape has a `CircularBuffer&`, `uint32_t sem_*`, `TensorAccessorArgs<N>`, or old-style addr-gen signature. No file-path kernel instantiation of foreign kernels. The moreh_common headers form a moreh-family (and broader) shared-kernel port-together set — coordinate any Metal-2.0 rewrite of them with co-borrowers, though they are already Metal-2.0-flavored.
- **TTNN factory analysis (sheet-derived, cross-checked):** Concept `descriptor`; custom hash `no`; runtime-args update `no`; override-runtime-args method `no`; pybind `create_descriptor` `no`; smuggled pointer `no`; op-owned tensors none; `Is safe to port? = yes`. Target concept `MetalV2FactoryConcept` (no op-owned tensors).

## Misc anomalies

- None observed. RTAs and CTAs are all consumed; `compute_defines` (`FP32_DEST_ACC_EN`), the `need_bcast_dim[0]`/`[1]` compute CTAs, and the dim/stride/bcast RTA vectors are all read by the kernels. No dead CBs, no dead-but-hashed attributes, no suspicious constants noted.

## Recipe notes

- **Sanctioned-free-function list appears non-exhaustive.** The Device 2.0 Green bullet names only `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` as sanctioned, and instructs auditors to "check the current Device 2.0 surface rather than assuming the shape alone makes it a holdover." I hit `get_dataformat(cb_id)` (donor `moreh_common.hpp:100`), which has the exact isolated-holdover *shape* (CB-index free fn, `DataflowBuffer` wrapper `cb.get_dataformat()` in scope, 1-line replacement). Resolving it required a repo-wide scan, which showed the free-function form is used pervasively in migrated code — including Metal-2.0 `*_metal2.cpp` kernels passing `dfb::name` binding tokens to it — so it is de-facto sanctioned. Consider adding `get_dataformat(cb_id)` (and any other retained metadata free functions) to the explicit sanctioned list, or stating the list is illustrative, to spare the next auditor the codebase sweep on a common pattern.
