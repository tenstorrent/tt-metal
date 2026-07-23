# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/fill_pad`

Single device operation, two program factories (shared kernels):

- **`FillPadDeviceOperation`**
  - `FillPadProgramFactory` (`device/fill_pad_program_factory.cpp`) — DRAM interleaved + DRAM-sharded
  - `FillPadL1ShardedProgramFactory` (`device/fill_pad_program_factory.cpp`) — all L1-sharded (HEIGHT/WIDTH/BLOCK)

Bundled into one report: both factories live in one file, share the compute kernel (`fill_pad_compute.cpp`) and the mask helpers (`fill_pad_dataflow_common.hpp`), and differ only in their dataflow kernels. Per-factory attribution is retained where findings differ (tensor-binding case, TensorAccessor 3rd arg).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c1349c0d941 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/fill_pad` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `FillPadDeviceOperation` → `FillPadProgramFactory`, `FillPadL1ShardedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all kernels on Device 2.0 idioms (`Noc`, `DataflowBuffer`, `TensorAccessor`, `UnicastEndpoint`); only sanctioned free function is `get_tile_size(cb_id)` |
| *Prereqs* — Cross-op escapes | Ok — no cross-op / donor kernels; all kernels owned by the op |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (N/A) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (both factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases; no host-folded offset) |
| *Port work* — Tensor bindings (per binding) | `input`: Case 1 (DRAM factory) / Case 2 (sharded factory) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2, both DRAM-factory sites) |
| *Port work* — CB endpoints | all legal (1P+1C) |

**CB endpoints** are dispositions, not gates. Every CB in both factories is a plain 1-producer/1-consumer FIFO — no self-loops, no dead CBs, no multi-binding. Recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gates clear: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (both factories, `Is able to port? = yes`, cross-check clean), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (Class 2 drop). No portable-subset scoping needed — the whole op is portable. Port work is routine: two tensor bindings (Case 1 / Case 2), two redundant 3rd-arg drops.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN for both factory rows. Readiness sheet (fetched fresh this run) rows 39–40:
  - `FillPadProgramFactory` — Concept `descriptor`, Custom hash `no`, Runtime-args update `no` (both columns), Pybind descriptor `no`, Smuggled pointer `no`, Is safe to port? `yes`, **Is able to port? `yes`**, TensorParameter relaxation `none`, Op-owned tensors blank.
  - `FillPadL1ShardedProgramFactory` — identical column values.
  - **Cross-check (clean):** `Concept=descriptor` ↔ both factories expose `create_descriptor()` returning `ProgramDescriptor` (`fill_pad_program_factory.hpp:97,105`). `Custom hash=no` ↔ no `compute_program_hash` override in `fill_pad_device_operation.cpp` (grep clean). `Runtime-args update=no` ↔ no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean). `Pybind descriptor=no` ↔ `fill_pad_nanobind.cpp` binds only the host function `fill_implicit_tile_padding`, no `create_descriptor` binding. Cross-column invariants hold (Runtime-args update `no` and Op-owned tensors blank are consistent with the `descriptor` concept). `Smuggled pointer=no` is consistent with the `Buffer*`-binding form used here (auto-registered `BufferBinding`, patched on cache hit — not the silent-wrong raw-address hazard; see Tensor bindings below).
- **Device 2.0 (every kernel used):** GREEN. Every kernel the op instantiates is structurally Device 2.0:
  - `fill_pad_reader.cpp`, `fill_pad_writer.cpp`, `fill_pad_sharded_reader.cpp`, `fill_pad_sharded_writer.cpp`, `fill_pad_compute.cpp`, and the shared header `fill_pad_dataflow_common.hpp`.
  - Idioms: `Noc noc;` + `noc.async_read/async_write/async_read_barrier/async_writes_flushed/async_write_barrier`; `DataflowBuffer dfb(...)` + `reserve_back/push_back/wait_front/pop_front/get_id/get_read_ptr/get_write_ptr` (methods on the wrapper, e.g. `fill_pad_dataflow_common.hpp:41,52`); `TensorAccessor(args, addr, page)` / `TensorAccessorArgs<N>()`; `UnicastEndpoint{}` for local-L1 self reads/writes (`fill_pad_sharded_reader.cpp:74`, `fill_pad_sharded_writer.cpp:94`). Includes are `api/dataflow/*`, `api/tensor/*`, `api/compute/*` (Device 2.0 / LLK) plus the in-op common header only.
  - Only CB-index free function present is `get_tile_size(cb_id)` (e.g. `fill_pad_reader.cpp:76`) — **sanctioned** per the Device 2.0 Green bullet; not a holdover.

  No violations. No table needed.

- **Feature compatibility:** all Appendix A entries scanned against both factories and all kernels; none fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | plain `CBDescriptor` only; no `.global_circular_buffer` field, no `remote_index`/`remote_cb`, no `CreateCircularBuffer(..., global_cb)` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` set on any `CBDescriptor` (grep clean) |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = FillPadInputs { Tensor input; }` — single fixed tensor, no `std::vector<Tensor>`; kernels read CTAs only at constexpr offsets (`get_compile_time_arg_val(0..9)`), no runtime-varying CTA index |

- **CB endpoints (GATE-free):** all CBs legal (1 producer, 1 consumer per node). Device 2.0 gate is GREEN, so the census ran on intact idioms. Four CB indices, identical roles in both factories:

  | CB (index) | Config | Producer | Consumer | Verdict |
  |---|---|---|---|---|
  | `cb_data_in` (c_0) | all | reader (`reserve_back`/`push_back`) | compute (`wait_front`/`pop_front`) | 1P+1C legal |
  | `cb_right_mask` (c_1) | `has_right_pad` | writer (`push_right_mask_tile`) | compute (`wait_front`/`pop_front`) | 1P+1C legal |
  | `cb_bot_mask` (c_2) | `has_bottom_pad` | writer (`push_bottom_mask_tile`) | compute (`wait_front`/`pop_front`) | 1P+1C legal |
  | `cb_data_out` (c_16) | all | compute (`reserve_back`/`push_back`) | writer (`wait_front`/`pop_front`) | 1P+1C legal |

  Mask CBs (c_1/c_2) exist only under their respective pad configs; when absent, the CB is not created (no dead CB). No raw co-fill / hidden second writer, no multi-reader — each CB is touched by exactly one locked producer and one locked consumer on every node. Nothing to flag.
- **Offset base pointers:** GREEN. Every address-bearing RTA was resolved; no host-side offset fold in either factory.
  - Both factories pass the whole-buffer pointer `tens_buffer` (a `Buffer*`, **not** `->address()`, no `+ offset`) into reader/writer RTA slot 0: `fill_pad_program_factory.cpp:293,295` (DRAM) and `:619-622` (sharded). The framework recovers the clean base.
  - Kernel-side arithmetic (sharded: `addr = shard_l1_base + local_right_col * tile_bytes`, `fill_pad_sharded_reader.cpp:70`) is computed **on-device from the clean base**, not folded on the host — so it is not a Type-1/2 fold. `fill_pad` is (correctly) **not listed** in `2026-07-19_offset_base_pointers.md`; scan confirms clean. No Type 3 (`address_offset`) / Type 4 (`narrow`).
- **TensorAccessor 3rd argument:** GREEN — both sites Class 2 (redundant), drop.
  - `fill_pad_reader.cpp:87` — `TensorAccessor(src_args, buf_addr, tile_bytes)`, 3rd arg `tile_bytes = get_tile_size(cb_tile_in_idx)`.
  - `fill_pad_writer.cpp:81` — `TensorAccessor(dst_args, buf_addr, tile_bytes)`, 3rd arg `tile_bytes = get_tile_size(cb_data_out_idx)`.
  - **Classification.** (1) Sharded-or-interleaved: the DRAM factory serves interleaved and (rarely) DRAM-sharded. (2) Magnitude: `get_tile_size(cb)` = `tt::tile_size(data_format)` = the true logical tile page size, and equals `buffer->page_size()` for these tile-layout tensors (supported dtypes are BFLOAT16/FLOAT32/UINT16/UINT32/INT32 — none block-float, so no exponent-drop trap). **Correct magnitude → inert on interleaved (realigned) and equal to `aligned_page_size` on sharded → Class 2.** Matches `2026-07-06_tensor_accessor_3rd_arg_triage.md:52` (`fill_pad` = 2 — Redundant, no fix) and its note at line 112 (previously mis-flagged; `tt::tile_size` is correct). The sharded factory's dataflow kernels use no `TensorAccessor` (they self-read local L1 via `UnicastEndpoint`), so they have no 3rd-arg site.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding — the single binding `input`, split by factory):
  - `input` @ `FillPadProgramFactory` — **Case 1** (via `TensorAccessor`). Host smuggles `tens_buffer` (`Buffer*`) into reader/writer RTA slot 0 (`fill_pad_program_factory.cpp:293,295`); the kernel feeds `buf_addr` into `TensorAccessor(src_args, buf_addr, tile_bytes)` and does all NoC access through the accessor (`fill_pad_reader.cpp:87,101`, `fill_pad_writer.cpp:81,109`). Express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the RTA address + `TensorAccessorArgs` plumbing disappear.
  - `input` @ `FillPadL1ShardedProgramFactory` — **Case 2** (raw pointer). Host smuggles the same `tens_buffer` (`fill_pad_program_factory.cpp:619-622`); the kernel uses the delivered base **raw** in hand-rolled `UnicastEndpoint` address arithmetic, never through a `TensorAccessor` (`fill_pad_sharded_reader.cpp:46,70,90`; `fill_pad_sharded_writer.cpp:54,92`). Bind as `TensorParameter`, pull the base via the `get_bank_base_address` bridge, keep the raw arithmetic unchanged.
- **TensorParameter relaxation:** none (sheet `none`, both factories).
- **TensorAccessor 3rd arg:** drop the redundant page-size arg @ `fill_pad_reader.cpp:87` and `fill_pad_writer.cpp:81` (Class 2; no `dynamic_tensor_shape` needed).
- **CB endpoints:** all legal — no dispositions to apply.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden co-fill, no multi-reader.
- **Cross-op / shared kernels:** none borrowed. Note the *in-op* sharing: `fill_pad_compute.cpp` and `fill_pad_dataflow_common.hpp` are shared by both factories — port them once and keep both factories' bindings consistent (single rewrite, in-op only).
- **RTA varargs:** none. All kernels read a fixed set of RTAs at constexpr indices (reader/writer `get_arg_val(0..6)`, compute `get_arg_val(0..2)`, sharded `get_arg_val(0..4)`) — every arg names cleanly.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** No function-call escapes to other ops: kernel `#include`s resolve to `tt_metal/*`-class LLK/HAL headers (`api/dataflow/*`, `api/tensor/*`, `api/compute/*` — donor class 1, no concern) plus the in-op `fill_pad_dataflow_common.hpp`. No file-path kernel instantiation of borrowed sources — all five kernels are owned by the op. No summary table or per-call detail (all rolls ✓).
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** Concept `descriptor` (both factories) → target `MetalV2FactoryConcept`. Confirmed absent (each a gate conjunct): custom hash, custom `override_runtime_arguments`, pybind `create_descriptor`, op-owned tensors, genuine multi-program. `Is safe to port? = yes` (Smuggled pointer `no`). Both factories use the `Buffer*`-binding form (`emplace_runtime_args(core, {tens_buffer, ...})`) — the framework's interim BufferBinding hack, correct-on-cache-hit today, superseded by the typed `TensorParameter` binding at port time.

## Misc anomalies  *(team-only, non-gating)*

- **Dead compile-time args in the reader.** `fill_pad_reader.cpp` reads `elem_size = get_compile_time_arg_val(7)` (`:64`) but never uses it; CT slots `[2] N_slices` and `[8] fill_bits` are passed by the host (`fill_pad_program_factory.cpp:161-172`) and self-documented "unused here" (`fill_pad_reader.cpp:37,43`) — dead plumbing. Harmless; routes to the ops team, not the port diff.
- **Unused `elem_size` reads elsewhere.** `fill_pad_compute.cpp:94` (CT[4]) and `fill_pad_sharded_reader.cpp:41` (CT[2]) also read `elem_size` into a `constexpr` that is never referenced. Same disposition.

## Per-DeviceOperation attribution

Single `FillPadDeviceOperation`; per-factory differences are the tensor-binding case (Case 1 DRAM / Case 2 sharded) and the TensorAccessor 3rd-arg sites (DRAM factory only), both recorded above. All gate verdicts are identical across the two factories.
