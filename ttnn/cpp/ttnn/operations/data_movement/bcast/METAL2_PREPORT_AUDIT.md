# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/bcast`

- **`BcastDeviceOperation`** (single DeviceOperation; all 5 factories share `bcast_h.cpp` compute and the same tensor bindings)
  - `BcastMultiCoreHProgramFactory` (`bcast_multi_core_h_program_factory.cpp`)
  - `BcastMultiCoreWProgramFactory` (`bcast_multi_core_w_program_factory.cpp`)
  - `BcastMultiCoreHWProgramFactory` (`bcast_multi_core_hw_program_factory.cpp`)
  - `BcastShardedHProgramFactory` (`bcast_sharded_h_program_factory.cpp`)
  - `BcastShardedHOptimisedProgramFactory` (`bcast_sharded_h_optimised_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `e9e376712e5 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

**Unreferenced kernel files (out of scope — no factory instantiates them):** `reader_bcast_h_interleaved.cpp`, `reader_bcast_hw_interleaved.cpp`, `reader_bcast_scalar_interleaved_partitioned.cpp`, `reader_bcast_w_interleaved.cpp` (all under `device/kernels/dataflow/`). Their contents were not audited. Note: `reader_bcast_scalar_interleaved_partitioned.cpp` is the only file that `#include`s an out-of-directory header (`ttnn/kernel/dataflow/generate_bcast_scalar.hpp`); since it is unreferenced, that coupling is not in scope.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/bcast` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `BcastDeviceOperation` → H / W / HW / ShardedH / ShardedHOptimised (5 factories) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (all own kernels + the one cross-family donor writer are Device 2.0) |
| *Prereqs* — Cross-op escapes | Ok (1 file-path donor writer, Device 2.0-clean; no function-call escapes in referenced kernels) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A (fixed 2-input op; CTAs read at constexpr offsets only) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (all 5 factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 5) |
| *TTNN Readiness* — Secretly SPMD | N/A (`descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes (all 5) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (plain `bind_function<"bcast">`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (no address fold on any RTA) |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved) / clean borrowed-DFB (sharded) — see Port-work |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (every `TensorAccessor` is 2-arg) |
| *Port work* — CB endpoints | legal, plus self-loop on the output CB in the two sharded-H factories |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear for all five factories: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd argument ✓. This op is a `descriptor`-concept, fixed-shape, `bfloat16` elementwise-broadcast op with no semaphores, no global buffers, no custom hash, and no offset-folded pointers — a clean mechanical port to `MetalV2FactoryConcept`. `METAL2_PORT_BRIEF.md` is issued alongside this report.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. The readiness sheet ("Operations analysis", fetched this run) has one row per factory; all five are `Concept = descriptor`, `Is able to port? = yes`, with `Custom hash = no`, `Runtime-args update = no` (both columns), `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `TensorParameter relaxation = none`, `Op-owned tensors?` blank. Cross-check against code confirms every cheaply-checkable column:
  - `Concept = descriptor` — each factory defines `create_descriptor(...)` returning `tt::tt_metal::ProgramDescriptor` (e.g. `bcast_multi_core_h_program_factory.cpp:17`). ✓
  - `Custom hash = no` — no `compute_program_hash` override in `bcast_device_operation.cpp`/`.hpp`. ✓
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` in any factory. ✓
  - `Pybind descriptor = no` — `bcast_nanobind.cpp:57` binds via `ttnn::bind_function<"bcast">`; no `create_descriptor` / `nb::class_` of the device op. ✓
  - No cross-column-invariant violations (no op-owned tensors on a `descriptor` row; `Runtime-args update = no`). ✓
- **Device 2.0 (every kernel used):** GREEN. Every referenced kernel — 5 readers, 1 own writer, 4 compute, and the 1 cross-family donor writer — is structurally Device 2.0: `Noc noc;`, `DataflowBuffer dfb(cb_id)`, `TensorAccessor(args, addr)`, `noc.async_read/async_write(...)`, `noc.async_read_barrier/async_write_barrier`, `CoreLocalMem<>`, and the compute API (`init_bcast`, `BCAST_OP`, `tile_regs_acquire/commit/wait/release`, `pack_tile`). Only sanctioned CB-index free functions appear — `get_tile_size(cb_id)` (readers/writers) and `get_local_cb_interface(cb_id).fifo_page_size` (donor writer) — both explicitly kept by Device 2.0. No `InterleavedAddrGen`/`ShardedAddrGen`, no raw `noc_async_read`, no `get_read_ptr(cb_id)`/`get_write_ptr(cb_id)` free-function holdovers. (The one `get_write_ptr()` — `reader_bcast_h_sharded_optimised.cpp:39` — is the DFB **object method** `dfb_in1.get_write_ptr()`, Device 2.0-native, not a free-function holdover.) No violations to route.
- **Feature compatibility:** GREEN — clean scan, all `N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `.global_circular_buffer` field, `remote_cb`/`.remote_index`, or `CreateGlobalCircularBuffer`. The buffer-backed CBs (`.buffer = src0_buffer`/`dst_buffer` in the HW and sharded factories) are the ordinary borrowed-memory pattern, not GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `.address_offset`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress`. Borrowed-memory CBs are bound at base (offset 0). |
  | GlobalSemaphore | N/A | The op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = {input_a, input_b, preallocated_output}` (fixed count; no `std::vector<Tensor>`). Kernels read compile-time args only at constexpr offsets (`TensorAccessorArgs<0>()`, `get_compile_time_arg_val(0)`); no runtime-varying CTA index. |

- **Offset base pointers:** GREEN — cleared. No address RTA folds a host-side offset into a base pointer. Every factory passes tensor buffers to kernels via the **`Buffer*`-binding form** (`src0_buffer`, `src1_buffer`, `dst_buffer`, `b.buffer()` pushed directly into `emplace_runtime_args` / `runtime_args`), never `buffer()->address()` — there is no `->address()` expression anywhere in the op. The sharded factories pass a separate scalar `offset` runtime arg (`bcast_sharded_h_program_factory.cpp:190-199`, `bcast_sharded_h_optimised_program_factory.cpp:193-203`), but it is a **tile/page-index** consumed as `.page_id` by the `TensorAccessor` (`reader_bcast_h_sharded.cpp:36`), i.e. a clean tile-index scalar — not a `base + offset` fold. bcast is not in the offset-base-pointer triage doc (dated 2026-07-19); the recognition scan on every RTA confirms no fold (neither Type 1 nor Type 2), so nothing is missed.
- **TensorAccessor 3rd argument:** GREEN — no site fires. Every `TensorAccessor` construction across all referenced kernels is the 2-arg form `TensorAccessor(args, addr)` (verified: `reader_bcast_h_interleaved_input_rows_partitioned.cpp:30-31`, `reader_bcast_w_interleaved_input_cols_partitioned.cpp:35-36`, `reader_bcast_hw_interleaved_partitioned.cpp:38,44`, `reader_bcast_h_sharded.cpp:25`, `reader_bcast_h_sharded_optimised.cpp:29`, `writer_unary_interleaved_input_cols_batched.cpp:25`, donor `writer_unary_interleaved_start_id.cpp:31`). No explicit page-size 3rd argument anywhere. bcast is not in the 3rd-arg triage doc (dated 2026-07-06); nothing to classify.

- **CB endpoints (GATE-free):** Three CBs per factory — `c_0` (src0/input_a), `c_1` (src1/input_b), `c_16` (output). Dispositions per `(CB, config)`:
  - **H, W factories** (`reader` + own `writer` + `compute`): `c_0` reader-produces / compute-consumes → legal 1:1; `c_1` reader-produces / compute-consumes → legal 1:1; `c_16` compute-produces / writer-consumes → legal 1:1. All legal.
  - **HW factory, interleaved config** (src0 & output not sharded): same three-kernel 1:1 as H/W (writer is the eltwise/unary donor) → all legal.
  - **HW factory, `IN0_SHARDED` config**: `c_0` is borrowed-memory (`.buffer = src0_buffer`); reader `dfb_in0.reserve_back/push_back(num_tiles)` (producer) + compute consumes → **1P+1C, legal**. `c_1`, `c_16` as interleaved.
  - **HW factory, `OUT_SHARDED` config**: `c_16` is borrowed-memory (`.buffer = dst_buffer`); compute produces + donor writer `dfb.wait_front(num_pages)` consumes (`writer_unary_interleaved_start_id.cpp:25`) → **1P+1C, legal**.
  - **ShardedH & ShardedHOptimised factories** (`reader` + `compute`, **no writer**): `c_0` borrowed (`.buffer = src0_buffer`) — reader `dfb_in0.push_back(Ht*Wt)` (producer) + compute consumes → 1P+1C legal. `c_1` — reader produces / compute consumes → legal. `c_16` borrowed (`.buffer = dst_buffer`) — compute produces (`dfb_out.reserve_back/push_back`) and **nothing drains it** (output is resident; no writer kernel) → **single toucher → self-loop** (bind compute both PRODUCER and CONSUMER; legal on Gen1 for compute). This is the only out-of-window disposition in the op.

  Nothing here gates a Gen1 port.
- **Offset base pointers** and **TensorAccessor 3rd argument** are cross-listed under Gate detail above (both cleared).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding; classification varies per factory/config — see Per-DeviceOperation notes):
  - **`input_b` (src1):** **Case 1** in *all* factories — always read through `TensorAccessor(src1_args, src1_addr)`; `src1_addr` arrives via the `Buffer*`-binding form (`src1_buffer` / `b.buffer()` in the reader RTA). Express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`.
  - **`input_a` (src0):** **Case 1** in H, W, and HW-interleaved (read via `TensorAccessor(src0_args, src0_addr)`); **clean** (borrowed-memory DFB, `.buffer = src0_buffer`) in ShardedH, ShardedHOptimised, and HW-`IN0_SHARDED` — port via `DataflowBufferSpec::borrowed_from`.
  - **`output` (dst):** **Case 1** in H, W, and HW-interleaved (writer `TensorAccessor(dst_args, dst_addr)`); **clean** (borrowed-memory DFB, `.buffer = dst_buffer`) in ShardedH, ShardedHOptimised, and HW-`OUT_SHARDED`.
  - No Case 2 (raw-pointer) bindings anywhere — no `get_noc_addr_from_bank_id`, no RTA-sourced base + hand-rolled NoC arithmetic.
- **TensorParameter relaxation:** none (sheet `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop the output CB `c_16` in `BcastShardedHProgramFactory` and `BcastShardedHOptimisedProgramFactory` (single toucher: compute produces, nothing drains). All other CBs are legal 1:1 (including the HW borrowed-memory configs).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer (no semaphore-gated raw co-fill), no multi-reader, no dual-instance work-split. The only non-1:1 CB is the sharded output self-loop above.
- **Cross-op / shared kernels:** the HW factory instantiates the cross-family donor writer `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (owned by `eltwise/unary`). It is **broadly shared** — 46 program factories across the tree instantiate it — so its Metal 2.0 (CB→DFB, named-token) rewrite is a single change that every borrower must adopt together (a large port-together set). Note: a `writer_unary_interleaved_start_id_metal2.cpp` variant already exists in the `experimental/quasar/` tree, suggesting the shared-writer rewrite may land as a parallel file rather than in-place.
- **RTA varargs:** none. Every kernel reads runtime args at fixed constexpr indices as distinct named fields; no runtime-count loop and no data-selected element.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: ✓ clean.** No function-call escapes in any referenced kernel — they `#include` only `api/*` (tt_metal LLK/HAL/firmware, "no concern" class). One file-path kernel instantiation crosses a family boundary (donor writer, below), Device 2.0-clean.
  - **Borrowed kernel files (file-path instantiation):**

    | Kernel file | Owning family | Sharing | Used by bcast factory |
    |---|---|---|---|
    | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly shared (46 factories) | HW |

    All other referenced kernels (5 readers, `writer_unary_interleaved_input_cols_batched.cpp`, 4 compute) are bcast-owned.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** all 5 factories `descriptor`, no op-owned tensors, no MeshWorkload, no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept` for all.

## Misc anomalies  *(team-only, non-gating)*

- **Dead runtime args in the interleaved readers.** `bcast_multi_core_h_program_factory.cpp:172-190` emplaces 15 reader args, but `reader_bcast_h_interleaved_input_rows_partitioned.cpp` reads only indices 0,3,4,8,9,10,11,12,13,14 — indices 1,2,5,6,7 are set to `0u` and never read. Same shape in the W factory (`bcast_multi_core_w_program_factory.cpp:168-187` → `reader_bcast_w_interleaved_input_cols_partitioned.cpp`). Harmless; ops team may trim.
- **Dead local reads in readers.** `reader_bcast_h_interleaved_input_rows_partitioned.cpp:15` and `reader_bcast_w_interleaved_input_cols_partitioned.cpp:15` read `NCHtWt = get_arg_val<uint32_t>(8)` into a local that is never used; both also declare `uint32_t num_tiles = src0_num_tiles;` that is unused. Cosmetic.
- **Dead writer compile-time args in the sharded factories.** `bcast_sharded_h_program_factory.cpp:137-140` and `bcast_sharded_h_optimised_program_factory.cpp:139-142` build `writer_compile_time_args = {dst_is_dram}` then immediately `(void)`-discard both — and these factories have **no writer kernel** at all. Dead computation.

## Recipe notes

- The audit's tensor-binding subjects assume `->address()`-in-RTA as the canonical smuggled-pointer shape, but this op is fully on the framework's newer **`Buffer*`-binding form** (pushing the `Buffer*` object, not its address, into `emplace_runtime_args`). The recipe does cover this (the "`Buffer*`-binding form" bullet under TensorParameter analysis — "classify by what the kernel does with the base"), and it applied cleanly (all → Case 1). Noting only that for this op the entire offset-base / tensor-binding scan reduces to that one bullet, since there is not a single `->address()` in the op.
