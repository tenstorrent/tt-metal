# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/reduction/integral_image`

One DeviceOperation, one program factory:

- **`IntImgDeviceOperation`** (`ttnn::experimental::prim`)
  - `IntImgDeviceOperation` single-descriptor factory — `create_descriptor()` in `device/intimg_program_factory.cpp`

Kernels (all owned by this op, all referenced by the single factory):
- `device/kernels/intimg_reader.cpp` (ReaderConfig / dataflow)
- `device/kernels/intimg_compute.cpp` (ComputeConfig)
- `device/kernels/intimg_writer.cpp` (WriterConfig / dataflow)
- `device/kernels/common.hpp`, `device/kernels/common_dataflow.hpp` (in-directory shared headers)

No unreferenced kernel files. The factory contains one commented-out CB (`AXIS_3_BUFFER_1`, `intimg_program_factory.cpp:111`) — dead comment, not a CB.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/reduction/integral_image` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `IntImgDeviceOperation` → single-descriptor factory (`intimg_program_factory.cpp`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (GREEN) — all kernels on Device 2.0 idioms (`Noc`, `CircularBuffer`, `TensorAccessor`) |
| *Prereqs* — Cross-op escapes | Ok — no out-of-directory kernel includes; op owns all 3 kernels |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (fixed 18 CTAs + 2 `TensorAccessorArgs` blocks) |
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
| *Port work* — Tensor bindings (per binding) | input → Case 1 · output → Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (both accessors are 2-arg) |
| *Port work* — CB endpoints | 4× legal 1:1 · 5× self-loop |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Recorded per `(CB, config)`; this op has a single config (interleaved, fixed 2×4 core grid), so no per-config flips.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 (all kernels compliant), Feature compatibility (no Appendix A feature used), TTNN factory concept (`Is able to port? == yes`, cross-check clean), Offset base pointers (no fold), TensorAccessor 3rd arg (no 3rd-arg site). Port work is light: two Case-1 tensor bindings and five self-loop CBs. No portable-subset scoping needed — the op is a single unconditional descriptor factory.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Live "Operations analysis" sheet (fetched fresh this run) has one row for this op — `IntImgDeviceOperation (single-descriptor)` — with `Is able to port? == yes`. Derivation all clear: `Concept == descriptor`, `Custom hash == no`, `Runtime-args update == no`, `Override runtime args == no`, `Pybind descriptor == no`, `Smuggled pointer == no`, `Is safe to port? == yes`. Cross-check against code confirms every cheaply-checkable column:
  - `Concept == descriptor` — `create_descriptor()` returns a `ProgramDescriptor` (`intimg_device_operation.hpp:39`, `intimg_program_factory.cpp:67`). ✓
  - `Custom hash == no` — no `compute_program_hash` override in `intimg_device_operation.cpp`. ✓
  - `Runtime-args update == no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` in the factory. ✓
  - `Pybind descriptor == no` — `intimg_nanobind.cpp` binds only the free function `intimg` via `bind_function`; no `nb::class_` of the device op / `create_descriptor` binding. ✓
  - `Op-owned tensors` (blank) — `descriptor` concept, no `buffers` vector. ✓ (cross-column invariant satisfied.)
  No conflict; sheet trusted.

- **Device 2.0 (every kernel used):** GREEN. All three kernels are structurally Device 2.0:
  - NoC access exclusively through the `Noc` object — `noc.async_read` / `async_read_barrier` / `async_write` / `async_write_barrier` / `async_write_zeros` / `write_zeros_l1_barrier` (`common_dataflow.hpp:17-41`, `intimg_reader.cpp:13-14`).
  - CB access exclusively through the `CircularBuffer` wrapper and the RAII `ReadCBGuard`/`WriteCBGuard` (`common.hpp:77-131`) — `wait_front`/`pop_front`/`reserve_back`/`push_back` are wrapper methods, not free functions.
  - Address generation through `TensorAccessor(args, base_addr)` (`intimg_reader.cpp:53`, `intimg_writer.cpp:64`) — no `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_read`.
  - The only bare-idiom grep hits are inside doc-comments of the RAII guards (`common.hpp:57-102`), not live calls.
  - One CB-index free-function metadata lookup: `get_dataformat(ctas.input_cb)` at `intimg_reader.cpp:52`, used in a `constexpr` type-selection (`std_type_t<...>`). This is **not** a Device 2.0 holdover and does not gate: it is structurally identical to the explicitly-sanctioned `get_tile_size(cb_id)` — both are `constexpr` free functions in `dataflow_api.h` (`get_dataformat` at line 300; `get_tile_size`), and the `CircularBuffer` wrapper forwards to both (`circular_buffer.h:113`, `:115`); Device 2.0's own migration guide keeps `get_tile_size(cb_id)` as a free function in its migrated examples (`device_api_migration_guide.md:605,630`). It also fails the isolated-holdover shape (no `CircularBuffer` for `input_cb` is in scope, and the wrapper method is non-`constexpr` so could not substitute in this template-argument context). It is a **port-time** move onto the DFB object (`DataflowBuffer::get_dataformat()`, `dataflow_buffer.h:241`, is `constexpr`) per kernel-side whitelist rule 7 — carried to the brief as an FYI, not a gate. (See Recipe notes — the sanctioned list names only `get_tile_size` / `get_local_cb_interface` explicitly, forcing this judgment.)

- **Feature compatibility:** every Appendix A entry scanned; none fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` in factory or kernels |
  | CBDescriptor `address_offset` (non-zero) | N/A | CBs built via `make_cb`; no `address_offset` set (`intimg_program_factory.cpp:35-49`) |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | fixed 18 CTAs + 2 fixed `TensorAccessorArgs` blocks; no runtime-varying CTA loop; `tensor_args_t` is a single `Tensor`, no variable-count container |

- **CB endpoints (GATE-free):** all nine CBs resolve cleanly under the single config. Census per node (the factory places all kernels over one 2×4 `core_range_set`, so every node sees reader+compute+writer):

  | CB (index) | Reader | Compute | Writer | Verdict | Disposition |
  |---|---|---|---|---|---|
  | `START` (0) | producer (WriteCBGuard + zero-fill, `intimg_reader.cpp:18-22,70`) | consumer (`ReadCBGuard`, `intimg_compute.cpp:87`) | — | 1P+1C | legal, no action |
  | `INPUT` (1) | producer (`load_from_dram`, `intimg_reader.cpp:43`) | consumer (`input_cb.wait_front/pop_front`, `intimg_compute.cpp:96,101`) | — | 1P+1C | legal, no action |
  | `ACC` (2) | — | producer+consumer (`intimg_compute.cpp:109-139`) | — | single toucher | **self-loop** |
  | `CUMSUM_STAGE_0` (3) | — | producer+consumer (`intimg_compute.cpp`) | — | single toucher | **self-loop** |
  | `CUMSUM_STAGE_1` (4) | — | producer+consumer | — | single toucher | **self-loop** |
  | `CUMSUM_STAGE_2` (5) | — | producer+consumer | — | single toucher | **self-loop** |
  | `OUTPUT` (6) | — | producer (`WriteCBGuard cb_output_write_guard`, `intimg_compute.cpp:206`) | consumer (`write_to_dram`, `intimg_writer.cpp:55`) | 1P+1C | legal, no action |
  | `AXIS_2_BUFFER` (7) | — | producer+consumer (`intimg_compute.cpp:129,170,187,189`) | — | single toucher | **self-loop** |
  | `AXIS_3_BUFFER` (8) | — | consumer (`get_and_propagate_adder_cube`, `intimg_compute.cpp:205`) | producer (readback `load_from_dram`, `intimg_writer.cpp:31`) | 1P+1C | legal, no action |

  No dead CBs (every index is a CTA into the compute/dataflow kernels and touched). No multi-binding (no CB has ≥3 touchers or ≥2 of one FIFO role; the writer's readback of `OUTPUT`-tensor data into `AXIS_3_BUFFER` uses the output `TensorAccessor`, not a second producer on any CB).

- **Offset base pointers:** GREEN. The two address RTAs are delivered as raw `Buffer*` (`reader_desc.emplace_runtime_args(core, {src_buffer})` at `intimg_program_factory.cpp:145`; `{dst_buffer}` at `:169`) — the Buffer*-binding form, no `->address()` and no host-folded offset. Kernels read them as a clean base (`get_arg_val<uint32_t>(0)` → `TensorAccessor(args, base_addr)`, `intimg_reader.cpp:50-53`, `intimg_writer.cpp:62-64`). Not present in the offset-base-pointer triage doc (`2026-07-19_offset_base_pointers.md`), consistent with a clean scan.

- **TensorAccessor 3rd argument:** GREEN — no site fires. Both `TensorAccessor` constructions are 2-arg (`args, base_addr`): `intimg_reader.cpp:53`, `intimg_writer.cpp:64`. No explicit page-size override anywhere. Not present in the 3rd-arg triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md`), consistent.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **input** — **Case 1** (via `TensorAccessor`). Delivered today as a `Buffer*`-binding RTA (`{src_buffer}`, `intimg_program_factory.cpp:145`); the reader feeds the base into `TensorAccessor(ctas.input_args, input_base_addr)` and does all reads through the accessor (`intimg_reader.cpp:50-53,43`). Port: express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the address RTA and `TensorAccessorArgs(src_buffer).append_to(...)` (`intimg_program_factory.cpp:133`) both disappear.
  - **output** — **Case 1** (via `TensorAccessor`). Delivered as a `Buffer*`-binding RTA (`{dst_buffer}`, `intimg_program_factory.cpp:169`); the writer uses `TensorAccessor(ctas.output_args, output_base_addr)` for **both** writing output (`write_to_dram`, `intimg_writer.cpp:64,55`) and reading back the upper block for cross-row propagation (`receive_upper_block` → `load_from_dram`, `intimg_writer.cpp:31`). One binding, both directions through the accessor. Port: `TensorParameter`/`TensorBinding` + `TensorAccessor(tensor::name)`; drop the address RTA and `TensorAccessorArgs(dst_buffer).append_to(...)` (`intimg_program_factory.cpp:134`).
  - Both `Buffer*`-binding deliveries are correct-on-cache-hit today (framework auto-registers and patches `BufferBinding`s) — not the silent-wrong RTA-address hazard. Routine port work.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation == none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `ACC`, `CUMSUM_STAGE_0`, `CUMSUM_STAGE_1`, `CUMSUM_STAGE_2`, `AXIS_2_BUFFER` (each a single compute-kernel toucher). The other four (`START`, `INPUT`, `OUTPUT`, `AXIS_3_BUFFER`) are legal 1P+1C — no action.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader. The writer reads output-tensor memory back through the `TensorAccessor` (a tensor read, into `AXIS_3_BUFFER`), not via a raw co-fill of any CB, so no census surprise.
- **`get_dataformat(ctas.input_cb)`** (`intimg_reader.cpp:52`) — port-time move onto the DFB object (`dfb::input.get_dataformat()`) per kernel-side whitelist rule 7; not a Device 2.0 change.
- **Cross-op / shared kernels:** none — the op owns all three kernels; no borrowed kernel files, no out-of-directory function-call escapes.
- **RTA varargs:** none — each dataflow kernel reads exactly one RTA (`get_arg_val<uint32_t>(0)`, the base address). No counted RTA loop, no data-selected index. Ordinary named-arg port work.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`. Every `#include` in the op's kernels resolves to either `tt_metal/*` (`api/dataflow/*.h`, `api/compute/*.h` — LLK/HAL, no concern) or an in-directory header (`common.hpp`, `common_dataflow.hpp`). No `ttnn/cpp/ttnn/kernel*`, `kernel_helper_functions`, in-family, or cross-family donor includes. The factory instantiates only its own three kernel files by path (`intimg_program_factory.cpp:51-56`). No port-together coupling set.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** `descriptor` concept, no op-owned tensors, no custom hash, no custom `override_runtime_arguments`, no pybind `create_descriptor`, no smuggled pointer, `Is safe to port? == yes`. Target concept `MetalV2FactoryConcept`. `Op Classification == PD (pointer-patching)` (the `Buffer*`-binding delivery of the two base addresses).

## Misc anomalies  *(team-only, non-gating)*

- **Reader uses `tile_width` where writer/compute use `tile_height` for the same quantity.** `intimg_reader.cpp:56`: `num_blocks_in_column = ceil(ctas.input_height, ctas.tile_width)`, whereas `intimg_writer.cpp:67` and `intimg_compute.cpp:277` use `ceil(ctas.input_height, ctas.tile_height)`. Harmless today because tiles are square (32×32 ⇒ `tile_width == tile_height`), but it is an inconsistency that would break if a non-square tile were ever used. Ops team.
- **`num_batches` loops are dead-bounded to 1.** Reader/writer/compute all loop `for (batch_i < ctas.num_batches)` (e.g. `intimg_reader.cpp:65`), with in-code comments that only one batch is expected and multi-batch is not fully implemented; `validate_on_program_cache_miss` hard-fails `input_shape[0] != 1` (`intimg_device_operation.cpp:21`). The `num_batches` CTA (`input_shape[0]`, `intimg_program_factory.cpp:129`) is therefore always 1. Latent/unfinished, not a port concern.
- **Commented-out CB in the factory:** `intimg_program_factory.cpp:111` carries a dead `// create_cb(... AXIS_3_BUFFER_1 ...)` line referencing a non-existent enum value. Cosmetic dead comment.
- **`operation_attributes_t` (`IntImgParams`) is an empty struct** (`intimg_device_operation_types.hpp:11`) — no attributes to hash; consistent with `Custom hash == no`.

## Recipe notes

- **Sanctioned-free-function list vs. `get_dataformat`.** The Device 2.0 Green bullet names only `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` as sanctioned, but `get_dataformat(cb_id)` is the same shape (a `constexpr` free function in `dataflow_api.h` that the `CircularBuffer` wrapper forwards to, `circular_buffer.h:115`) and appears in a genuine op (`intimg_reader.cpp:52`, in a `constexpr` type-selection where the wrapper method — non-`constexpr` — cannot substitute). The recipe's principle ("if Device 2.0 allows the free function, so do we") and the whitelist-rule-7 breadcrumb resolve it as non-gating, but the *explicit* sanctioned list forced a judgment call to get there. Consider adding `get_dataformat` to the sanctioned examples, or generalizing the bullet to "any `constexpr` CB-metadata free function the DataflowBuffer object also exposes."
