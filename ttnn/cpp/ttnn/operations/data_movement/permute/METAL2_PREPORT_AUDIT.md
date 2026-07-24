# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/permute`

Single device operation with five program factories, sharing kernels and helpers; audited together as one porting unit.

- **`PermuteDeviceOperation`**
  - `MultiCoreRowInvariant` (`permute_rm_program_factory.cpp`) — RM, last dim unchanged
  - `MultiCoreBlockedGeneric` (`permute_rm_program_factory.cpp`) — RM, last dim moved (blocked transpose)
  - `MultiCoreTileInvariant` (`permute_tiled_program_factory.cpp`) — tiled, tile dims stay in last two positions (identity or WH swap)
  - `MultiCoreTileRowInvariant` (`permute_tiled_program_factory.cpp`) — tiled, one tile dim moved out
  - `MultiCoreTiledGeneric` (`permute_tiled_program_factory.cpp`) — tiled, both tile dims moved

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `2a53d817976 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

All 10 kernel files in the op's `device/kernels/` are referenced by a factory (no unreferenced/dead kernel files). Three additional kernels are instantiated by file-path from outside the op directory (donors — see Team-only → Out-of-directory coupling).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/permute` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `PermuteDeviceOperation` → {`MultiCoreRowInvariant`, `MultiCoreBlockedGeneric`, `MultiCoreTileInvariant`, `MultiCoreTileRowInvariant`, `MultiCoreTiledGeneric`} |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 13 referenced kernels (10 owned + 3 donor) are structurally Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok (all donor kernels Device 2.0; file-path coupling reported, non-gating) |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore / CTA-varargs | Ok / Ok / Ok / Ok (all N/A) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (all 5 factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 5 factories) |
| *TTNN Readiness* — Secretly SPMD | N/A (concept is `descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind uses `bind_function<"permute">`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (clean bases; `Buffer*`-binding form, no host-folded offset) |
| *Port work* — Tensor bindings (per binding) | all **Case 1** (input + output, every factory) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (all accessor sites are 2-arg) |
| *Port work* — CB endpoints | legal, except two **self-loop** intermediates (`c_1` tilize CB in `MultiCoreBlockedGeneric` and `MultiCoreTiledGeneric`) |

## Result

**GREEN → brief issued.** Every gate clears:

- **Device 2.0** — all 13 kernels the op exercises use Device 2.0 idioms (`Noc`, `DataflowBuffer`, `TensorAccessor`, object methods); only sanctioned CB-index free functions appear (`get_tile_size`, `get_local_cb_interface`).
- **Feature compatibility** — no Appendix A feature is present (no GlobalCircularBuffer, no non-zero `address_offset`, no GlobalSemaphore, no CTA varargs). The op uses no semaphores at all.
- **TTNN factory concept** — the readiness sheet reports `Is able to port? = yes` for all five factory rows; the cheaply-checkable columns cross-check clean against the code.
- **Offset base pointers** — no `->address()` anywhere; buffers reach kernels via the `Buffer*`-binding form and all offset arithmetic is on-device. Clean bases.
- **TensorAccessor 3rd argument** — every `TensorAccessor(...)` construction (8 owned + 2 donor sites) uses the 2-arg form; no manual page-size override exists.

No blocking findings and no code-path caveats — the port covers all five factories.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Sheet rows 60–64 (one per factory) all read `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Is able to port? = yes`, `TensorParameter relaxation = none`, `Op-owned tensors? = (blank)`, `Secretly SPMD = (blank)`. Cross-check against code:
  - `Concept = descriptor` ✓ — each factory is a `create_descriptor()` returning `ProgramDescriptor` ([permute_device_operation.hpp:37-75](device/permute_device_operation.hpp#L37-L75)).
  - `Custom hash = no` ✓ — no `compute_program_hash` override in the op (grep clean).
  - `Runtime-args update = no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean).
  - `Pybind descriptor = no` ✓ — [permute_nanobind.cpp:35](permute_nanobind.cpp#L35) binds via `ttnn::bind_function<"permute">`; no `create_descriptor` / `nb::class_` of device-op internals.
  - Cross-column invariants hold (a `descriptor` row with no op-owned tensors and no runtime-args-update is internally consistent).

- **Device 2.0 (every kernel used):** GREEN. No violations. Every referenced kernel is structurally Device 2.0; the only CB-index free functions present are sanctioned (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`), which do not knock the op out of Green. Kernels audited:

  | Kernel file | Owner | Device 2.0 idioms observed |
  |---|---|---|
  | `device/kernels/dataflow/reader_permute_interleaved_rm_row_invariant.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, `noc_async_read_sharded(noc,…)` |
  | `device/kernels/dataflow/writer_permute_interleaved_rm_row_invariant.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, `noc_async_write_sharded(noc,…)` |
  | `device/kernels/dataflow/reader_permute_interleaved_rm_blocked_generic.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, sharded read helper |
  | `device/kernels/dataflow/writer_permute_interleaved_rm_blocked_generic.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, sharded write helper |
  | `device/kernels/compute/transpose_xw_rm_single_tile_size.cpp` | permute | `DataflowBuffer`, `tile_regs_*`, compute API |
  | `device/kernels/dataflow/reader_permute_interleaved_tiled_invariant.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, `noc.async_read(s,dfb,…)`, `get_tile_size(cb)` (sanctioned) |
  | `device/kernels/dataflow/writer_permute_interleaved_tiled_row_invariant.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, `get_tile_size(cb)` (sanctioned) |
  | `device/kernels/dataflow/reader_permute_interleaved_tiled_generic.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, `s.get_noc_addr(…)` |
  | `device/kernels/dataflow/writer_permute_interleaved_tiled_generic.cpp` | permute | `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem` |
  | `device/kernels/compute/transpose_xw_tiled.cpp` | permute | `DataflowBuffer`, `tile_regs_*`, compute API |
  | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary (donor) | `Noc`, `DataflowBuffer`, `TensorAccessor`, `get_local_cb_interface(cb).fifo_page_size` (sanctioned) |
  | `data_movement/transpose/device/kernels/compute/transpose_wh.cpp` | data_movement/transpose (donor) | `DataflowBuffer`, `tile_regs_*`, compute API |
  | `data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` | data_movement/transpose (donor) | `Noc`, `DataflowBuffer`, `TensorAccessor`, `dfb.get_entry_size()` |

- **Feature compatibility:** every Appendix A entry, in order. No entry fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `remote_cb`/`.remote_index(`, `num_global_cb_receivers`, or `.global_circular_buffer` field anywhere |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset`, `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor`; all CBs are plain `CBDescriptor` literals |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` / `CreateGlobalSemaphore`; the op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is fixed (`const Tensor&` + `std::optional<Tensor>`), not `std::vector<Tensor>`; the only live `get_compile_time_arg_val(i)` is `cb_id_out = get_compile_time_arg_val(0)` in the donor unary writer — a fixed constexpr index, not a runtime-varying CTA loop |

- **CB endpoints (GATE-free):** every CB is either a legal 1P+1C FIFO or carries a **self-loop** disposition. No dead CBs, no multi-binding. Census per `(CB, config)` in Port-work below.

- **Offset base pointers:** GREEN. No address RTA folds a host-side offset into a base — the factories never call `->address()`; each buffer is pushed as a `Buffer*` (BufferBinding form) into `emplace_runtime_args`, and all page/row/tile-index arithmetic happens on-device from a clean base. permute is not in the offset-base-pointer triage table (`2026-07-19_offset_base_pointers.md`); the scan confirms no fold — clean, consistent with its absence from the table.

- **TensorAccessor 3rd argument:** GREEN. All accessor constructions use the 2-arg `TensorAccessor(args, addr)` form — [reader_permute_interleaved_rm_row_invariant.cpp:22](device/kernels/dataflow/reader_permute_interleaved_rm_row_invariant.cpp#L22), [writer_permute_interleaved_rm_row_invariant.cpp:22](device/kernels/dataflow/writer_permute_interleaved_rm_row_invariant.cpp#L22), [reader_permute_interleaved_rm_blocked_generic.cpp:57](device/kernels/dataflow/reader_permute_interleaved_rm_blocked_generic.cpp#L57), [writer_permute_interleaved_rm_blocked_generic.cpp:55](device/kernels/dataflow/writer_permute_interleaved_rm_blocked_generic.cpp#L55), [reader_permute_interleaved_tiled_invariant.cpp:26](device/kernels/dataflow/reader_permute_interleaved_tiled_invariant.cpp#L26), [writer_permute_interleaved_tiled_row_invariant.cpp:115](device/kernels/dataflow/writer_permute_interleaved_tiled_row_invariant.cpp#L115), [reader_permute_interleaved_tiled_generic.cpp:117](device/kernels/dataflow/reader_permute_interleaved_tiled_generic.cpp#L117), [writer_permute_interleaved_tiled_generic.cpp:131](device/kernels/dataflow/writer_permute_interleaved_tiled_generic.cpp#L131), plus the two donor sites (`writer_unary_interleaved_start_id.cpp:31`, `reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp:40`). The `*_page_size`/`aligned_page_size()` named CTAs the factories emit are consumed for CB sizing and address math, **not** passed to a `TensorAccessor` constructor. permute is not in the 3rd-arg triage table (`2026-07-06_tensor_accessor_3rd_arg_triage.md`); consistent — no site to classify.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, all factories): both bindings are **Case 1** (via `TensorAccessor`).
  - `input_tensor` (`src_buffer`) — **Case 1**. Delivered today via the `Buffer*`-binding form (`emplace_runtime_args(core, {src_buffer, …})`); the kernel receives a `uint32_t` base and feeds it straight into `TensorAccessor(src_args, src_addr)`. Port: express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA slot and the `TensorAccessorArgs(*src_buffer).append_to(...)` CTA plumbing both disappear.
  - `output_tensor` (`dst_buffer`) — **Case 1**, symmetric to input.
  - No Case 2 (no raw-pointer NoC walk off a bound base), no borrowed-memory DFB (no `set_globally_allocated_address`).
- **TensorParameter relaxation:** none (sheet says `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per `(CB, config)`):
  - `MultiCoreRowInvariant`: `c_0` legal (reader P, writer C). — all legal.
  - `MultiCoreBlockedGeneric`: `c_0` legal (reader P, compute C); `c_2` legal (compute P, writer C); **`c_1` self-loop** — the tilize intermediate is produced *and* consumed by the compute kernel alone ([transpose_xw_rm_single_tile_size.cpp:27-63](device/kernels/compute/transpose_xw_rm_single_tile_size.cpp#L27-L63)) → bind it PRODUCER **and** CONSUMER.
  - `MultiCoreTileInvariant`: non-swap config — `c_0` legal (reader P, writer C). swap-HW config — `c_0` legal (reader P, compute C), `c_16` legal (compute P, writer C). — all legal, both configs.
  - `MultiCoreTileRowInvariant`: `c_0` legal; `c_1` (padding, only when `needs_padding`) legal (reader P, writer C); `c_16` (only when `swap_hw`) legal (compute P, writer C). — all legal across all `swap_hw × needs_padding` configs.
  - `MultiCoreTiledGeneric`: `c_0` legal (reader P, compute C); `c_2` legal (compute P, writer C); `c_3` (padding, only when `needs_y_padding`) legal (reader P, writer C); **`c_1` self-loop** — tilize intermediate produced and consumed by the compute kernel alone ([transpose_xw_tiled.cpp:33-69](device/kernels/compute/transpose_xw_tiled.cpp#L33-L69)) → bind PRODUCER **and** CONSUMER.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader, no ≥3-toucher CB in any factory/config. The only out-of-window dispositions are the two single-toucher tilize intermediates (self-loops, above).
- **Cross-op / shared kernels:** three donor kernels instantiated by file-path (below). All are Device 2.0 compliant, so no gate — but they induce a **port-the-family-together** coupling: the shared kernel's Metal 2.0 rewrite (CB→DFB, named-token bindings) is one change that every co-borrower must adopt together.
- **RTA varargs:** the RM and tiled reader/writer kernels read rank-length shape / permutation / stride arrays in a count-bounded loop (`for (i…; i < N + 3)` / `for (i…; i < RANK)`). Because the count is the tensor rank (`N`/`rank`, a CTA that varies across instantiations), these are genuine RTA varargs — port them via the kernel-side vararg mechanism, not by naming each element. Sites: `writer_permute_interleaved_rm_row_invariant.cpp:28`, `reader_permute_interleaved_rm_blocked_generic.cpp:37`, `writer_permute_interleaved_rm_blocked_generic.cpp:60`, `reader_permute_interleaved_tiled_invariant.cpp:32`, `writer_permute_interleaved_tiled_row_invariant.cpp:84`, `reader_permute_interleaved_tiled_generic.cpp:93`, `writer_permute_interleaved_tiled_generic.cpp:94`. The per-core scalar prefixes (`src_addr`/`dst_addr`, `start`, `end`, and the padding-tile scalars) remain ordinary named RTAs. (The two donor DM kernels and all compute kernels read only fixed scalar RTAs — no varargs.)

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean (no ⭐/✗). All function-call escapes resolve to LLK/HAL headers, the official shared kernel library, or in-family shared helpers, and all donor kernels are Device 2.0.
  - **Function-call escapes (`#include` outside the op dir):**

    | Include | Class | Notes |
    |---|---|---|
    | `api/dataflow/*`, `api/tensor/noc_traits.h`, `api/core_local_mem.h`, `api/compute/*` | `tt_metal/*` LLK/HAL | No concern. |
    | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | shared kernel library (`kernel_lib/`) | Lib team handles internally. Used by both compute kernels (`compute_kernel_lib::tilize<…>`). |
    | `ttnn/operations/data_movement/common/kernels/common.hpp` | in-family shared (data_movement/common) | Functions used take `Noc&` / raw addrs / `TensorAccessor` — Device 2.0 native shapes (✓). Used: `noc_async_read_sharded`, `noc_async_write_sharded`, `swap_elements`, `tt_memmove`, `fill_with_val`, `round_up`, `div_up`. Ports within the family. |

  - **Borrowed kernel files (file-path instantiation):**

    | Kernel file | Owning family | Shared? |
    |---|---|---|
    | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary (cross-family) | Broadly shared — a general-purpose interleaved unary writer instantiated by many eltwise/data-movement ops. Instantiated here by `MultiCoreTileInvariant`. Its Metal 2.0 rewrite must be coordinated across all co-borrowers. |
    | `data_movement/transpose/device/kernels/compute/transpose_wh.cpp` | data_movement/transpose (in-family) | Shared with the transpose op. Instantiated by `MultiCoreTileInvariant` and `MultiCoreTileRowInvariant` (swap-HW path). |
    | `data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` | data_movement/transpose (in-family) | Shared with the transpose op. Instantiated by `MultiCoreTileRowInvariant` reader. |

- **Relaxation candidates (mined from a custom hash):** none — the op has no custom hash.

- **TTNN factory analysis:** current concept `descriptor` (all factories); no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; no runtime-args-update hook. Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead compute RTA slots.** [permute_rm_program_factory.cpp:364](device/permute_rm_program_factory.cpp#L364) emits `{num_blocks_per_core, 0u, 0u}` to the `MultiCoreBlockedGeneric` compute kernel, but the kernel reads only slot 0 (`num_blocks = get_arg_val<uint32_t>(0)`, [transpose_xw_rm_single_tile_size.cpp:21](device/kernels/compute/transpose_xw_rm_single_tile_size.cpp#L21)). The two trailing `0u` slots are unused — the code comment says they "preserve historical layout." Routes to the ops team; the port does not act on it.
- **Dead local `curr_addr`.** `uint32_t curr_addr = …;` is assigned but never used in [writer_permute_interleaved_rm_row_invariant.cpp:34](device/kernels/dataflow/writer_permute_interleaved_rm_row_invariant.cpp#L34) and [reader_permute_interleaved_tiled_invariant.cpp:38](device/kernels/dataflow/reader_permute_interleaved_tiled_invariant.cpp#L38). Cosmetic dead code.
- **Unused CTAs in the RM row-invariant reader.** [reader_permute_interleaved_rm_row_invariant.cpp:13-15](device/kernels/dataflow/reader_permute_interleaved_rm_row_invariant.cpp#L13-L15) reads named CTAs `N` and `num_rows` that the kernel body never uses (only `page_size` is used). Harmless.

## Recipe notes

- The `Buffer*`-binding form is now the dominant delivery mechanism in this op (no `->address()` at all), which makes both the *Offset base pointers* and *TensorParameter analysis* scans quick: there is no `->address()` RTA to resolve, only `Buffer*` pushes to classify by kernel usage. The recipe's *TensorParameter analysis* subject already covers this shape explicitly (the "`Buffer*`-binding form" bullet), and its "classify by what the kernel does with the base" instruction resolved cleanly (all Case 1). No friction — noting only that an op fully on `Buffer*` binding is a clean, easy case for these two subjects.
