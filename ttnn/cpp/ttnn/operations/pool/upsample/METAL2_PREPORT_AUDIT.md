# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/pool/upsample`

- **`UpsampleOperation`** (`ttnn::prim`, in `device/upsample_device_operation.hpp`)
  - `UpsampleBilinearProgramFactory` (`device/upsample_bilinear_program_factory_multicore.cpp`)
  - `UpsampleMultiCoreInterleavedProgramFactory` (`device/upsample_program_factory_multicore_interleaved.cpp`)
  - `UpsampleMultiCoreShardedProgramFactory` (`device/upsample_program_factory_multicore_sharded.cpp`) — `WorkloadDescriptor`, op-owned config tensor
  - `UpsampleNearestFloatProgramFactory` (`device/upsample_nearest_float_program_factory.cpp`)

Single `DeviceOperation`, four `ProgramFactory` variants selected by `UpsampleOperation::select_program_factory` (`device/upsample_device_operation.cpp:24`) on `mode` / integer-vs-float scale / sharded-vs-interleaved.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `a21c8f3f324 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/pool/upsample` |
| **Overall** | **RED at op level; subset is clear** — `UpsampleBilinearProgramFactory` blocked, the other three factories clear every gate |
| **DOps / Factories** | `UpsampleOperation` → Bilinear, MultiCoreInterleaved, MultiCoreSharded, NearestFloat |
| *Prereqs* — Device 2.0 (every kernel used) | **No** for `UpsampleBilinearProgramFactory` only (isolated holdover, `device/kernels/compute/bilinear.cpp:16-26`) — **Yes** for the other three factories |
| *Prereqs* — Cross-op escapes | Ok — one cross-family kernel borrow (`untilize.cpp`), a `_metal2` fork already exists |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** for all four factories (readiness sheet, cross-checked) |
| *TTNN Readiness* — Concept (current) | `descriptor` (Bilinear, MultiCoreInterleaved, NearestFloat) / `WorkloadDescriptor`, secretly SPMD (MultiCoreSharded) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | Yes — single structurally-identical program copied across mesh coords (`device/upsample_program_factory_multicore_sharded.cpp:463-476`) |
| *TTNN Readiness* — Is safe to port? | Yes (all four rows) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`upsample_nanobind.cpp` binds only `mode`/`memory_config`/`compute_kernel_config`) |
| *TTNN Readiness* — Op-owned tensors | Yes, `UpsampleMultiCoreShardedProgramFactory` only — the per-core halo/replication config tensor (`device/upsample_program_factory_multicore_sharded.cpp:461`) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (+ op-owned tensors on the sharded factory) |
| *Port work* — Offset base pointer | none — no address RTA folds a host-side offset into its base |
| *Port work* — Tensor bindings (per binding) | Interleaved & NearestFloat: Case 1 (input, output). Bilinear & MultiCoreSharded: clean (borrowed-memory DFB for input/output/config) |
| *Port work* — TensorParameter relaxation | none (sheet: `none` on all four rows) |
| *Port work* — TensorAccessor 3rd arg | none — no site anywhere in the op passes a 3rd argument |
| *Port work* — CB endpoints | self-loop (1) / 1P+1C (7) / all legal (2) — full inventory below, no multi-binding, no dead CBs |

**CB endpoints** are dispositions, not gates: every CB in this op resolves to a self-loop, a 1P+1C assignment, or an already-legal (1,1) FIFO. No multi-binding advanced option needed anywhere; no dead CBs found.

## Result

**RED at op level; subset is clear.** `UpsampleBilinearProgramFactory` is blocked on the **Device 2.0 prerequisite** — its compute kernel (`device/kernels/compute/bilinear.cpp`) manually reimplements the CB push-back pointer arithmetic via raw `get_local_cb_interface(...).fifo_wr_ptr` mutation instead of the DataflowBuffer wrapper's `push_back()` / `evil_set_write_ptr()`. This is a Device 2.0 Data Movement migration gap, routed to the **Device 2.0 team**; it reads as an **isolated holdover** (one small helper function, wrapper already in scope, a mechanical replacement is available), not a broad Device 1.0 rewrite.

The other three factories — `UpsampleMultiCoreInterleavedProgramFactory`, `UpsampleMultiCoreShardedProgramFactory`, `UpsampleNearestFloatProgramFactory` — clear every gate (Device 2.0, Feature compatibility, TTNN factory concept, Offset base pointers, TensorAccessor 3rd argument). A `METAL2_PORT_BRIEF.md` is issued for that clean subset.

Once the Device 2.0 team lands the one-line-ish fix in `bilinear.cpp`, re-audit `UpsampleBilinearProgramFactory` alone — everything else in this report (TTNN readiness, CB endpoints, tensor bindings) was already computed for it below and is unlikely to change from a pointer-arithmetic fix that touches no CB shape or binding.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN for all four factories. Readiness-sheet rows (`pool/upsample`) all show `Is able to port? = yes`; cross-check against code confirms `Concept` (descriptor ×3, WorkloadDescriptor ×1), `Custom hash = no`, `get_dynamic_runtime_args = no`, `override_runtime_arguments = no`, `Pybind descriptor = no`, `Op-owned tensors` (yes only on the sharded factory), `Secretly SPMD Workload? = yes` on the sharded factory. Factory-set match: the sheet's 4 rows map 1:1 to the code's 4 `program_factory_t` variants, no phantom/missing rows.

- **Device 2.0 (every kernel used):**

  | Factory | Status |
  |---|---|
  | `UpsampleMultiCoreInterleavedProgramFactory` | GREEN — own kernels (`reader_upsample_unary_stick_layout_interleaved_start_id.cpp`, `writer_upsample_interleaved.cpp`) use `Noc`/`DataflowBuffer` throughout; the borrowed compute kernel (`.../data_movement/untilize/device/kernels/compute/untilize.cpp`, tiled path only) calls `compute_kernel_lib::untilize<...>()` from the official shared `kernel_lib`, itself DFB-index-native (`compute_kernel_hw_startup` + DFB NTTPs) |
  | `UpsampleMultiCoreShardedProgramFactory` | GREEN — `writer_upsample_multi_core_sharded.cpp` (both reader and writer instances) uses `Noc`/`DataflowBuffer`/`UnicastEndpoint` exclusively |
  | `UpsampleNearestFloatProgramFactory` | GREEN — `reader_upsample_nearest_float.cpp` / `writer_upsample_nearest_float.cpp` use `Noc`/`DataflowBuffer`/`TensorAccessor` exclusively |
  | `UpsampleBilinearProgramFactory` | **RED — isolated holdover** (table below) |

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/compute/bilinear.cpp` | 16–26 | `llk_push_pages_bilinear()`: `get_local_cb_interface(output).fifo_wr_ptr += num_words; get_local_cb_interface(output).fifo_wr_tile_ptr = 0; get_local_cb_interface(output).fifo_wr_ptr -= get_local_cb_interface(output).fifo_size;` — reimplements `push_back`'s pointer-advance/wraparound by hand, called via `PACK(llk_push_pages_bilinear(out_cb_id, tiles_per_reduction));` at line 59 | `DataflowBuffer out_dfb(out_cb_id)` is in scope in the caller (`kernel_main`, passed into `reduce_h_fused`); `evil_set_write_ptr(uint32_t)` exists on `DataflowBuffer` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:300`) for exactly this raw-cursor-set need |

  Why this isn't the sanctioned `get_local_cb_interface(cb_id)` carve-out: the Green bullet sanctions `get_local_cb_interface(cb_id)` for **metadata reads** (e.g. `.fifo_page_size`, mirrored in Device 2.0's own migrated examples) — not for **mutating** `.fifo_wr_ptr`/`.fifo_wr_tile_ptr`. Cross-checked against `llk_push_tiles` (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h:70-92`, what `DataflowBuffer::push_back()` calls on TRISC): the arithmetic in `llk_push_pages_bilinear` is line-for-line the same pointer-advance/wraparound `llk_push_tiles` does — the one thing it omits is `llk_push_to_brisc(...)` (the downstream consumer credit signal), which is presumably intentional here since `out_cb_id` is a borrowed-memory output CB with no on-device consumer. That omission doesn't change the classification: this is still hand-rolled Device 1.0-style manual CB pointer management, not a call through the Device 2.0 wrapper. **Isolated holdover** — confined to one small helper, in one kernel, in one factory; the fix is a mechanical swap to `out_dfb.evil_set_write_ptr(...)` (computing the same wrapped address) or, if semantically acceptable without the credit signal difference, `out_dfb.push_back(...)`. Route to the **Device 2.0 team** — this is out of the port's own scope regardless of how small the fix is.

  No other Device 1.0 idioms (raw `noc_async_read`, `InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedAddrGenFast`, raw semaphore addresses, manual CB index management elsewhere) were found in any of the op's kernels, own or borrowed.

- **Feature compatibility:** every Appendix A entry is `N/A` — none in use.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | `device/upsample_device_operation.hpp:17` includes `<tt-metalium/global_circular_buffer.hpp>`, but no `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, `.global_circular_buffer` field, or `remote_cb_*`/`.remote_index()` idiom appears anywhere in the directory — this is a **dead include** (see Misc anomalies), not feature use. Header-presence-alone is explicitly a non-signal per the recognition rule. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The only match is a comment (`device/upsample_program_factory_multicore_sharded.cpp:349-351`) describing the framework's internal `UpdateDynamicCircularBufferAddress` mechanism for borrowed-memory CBs — no `.address_offset` field is ever set to non-zero, and no direct call to `set_address_offset`/`UpdateDynamicCircularBufferAddress`/`cb_descriptor_from_sharded_tensor` appears with a non-zero offset. |
  | GlobalSemaphore | N/A | No reference anywhere. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single tensor, no `std::vector<Tensor>`); all kernels read CTAs at fixed constexpr indices. |

- **CB endpoints (GATE-free):** full inventory below, per factory. Everything resolves to self-loop / 1P+1C / already-legal (1,1) — nothing needs the multi-binding advanced option, and no dead CBs were found. `UpsampleBilinearProgramFactory`'s census is reported despite the Device 2.0 RED being an isolated holdover (idioms structurally intact elsewhere), per the precondition; re-confirm after the Device 2.0 fix lands since it changes how `out_cb` is produced (not its endpoint count).

- **Offset base pointers:** GREEN. Every address RTA in the op is a clean base:
  - `UpsampleMultiCoreInterleavedProgramFactory`: `src_buffer`/`dst_buffer` (`Buffer*`, Buffer\*-binding form) passed directly with no host-side offset folded in (`device/upsample_program_factory_multicore_interleaved.cpp:250,257`).
  - `UpsampleNearestFloatProgramFactory`: `input.buffer()`/`output_tensor.buffer()` (`Buffer*`) passed directly, same form (`device/upsample_nearest_float_program_factory.cpp:139,146`).
  - `UpsampleBilinearProgramFactory` / `UpsampleMultiCoreShardedProgramFactory`: no address RTA at all — every tensor (input, output, and the sharded factory's op-owned config tensor) is delivered via a borrowed-memory `CBDescriptor.buffer` binding, not an RTA.

- **TensorAccessor 3rd argument:** GREEN — no site anywhere in the op passes a 3rd (page-size) argument to `TensorAccessor`. Only two kernels use `TensorAccessor` at all (`reader_upsample_unary_stick_layout_interleaved_start_id.cpp`, `writer_upsample_interleaved.cpp`, `reader_upsample_nearest_float.cpp`, `writer_upsample_nearest_float.cpp`), all via the 2-argument form (`TensorAccessor(args, addr)`). The Bilinear and MultiCoreSharded factories don't use `TensorAccessor` at all (raw NoC addressing via `local_addr`/explicit core coordinates instead).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `UpsampleMultiCoreInterleavedProgramFactory`: input **Case 1** (`TensorAccessor`, `reader_upsample_unary_stick_layout_interleaved_start_id.cpp:19`), output **Case 1** (`writer_upsample_interleaved.cpp:23`).
  - `UpsampleNearestFloatProgramFactory`: input **Case 1** (`reader_upsample_nearest_float.cpp:31`), output **Case 1** (`writer_upsample_nearest_float.cpp:23`).
  - `UpsampleMultiCoreShardedProgramFactory`: input, output, and the op-owned config tensor — all **clean** (borrowed-memory DFB reads via `.buffer` on `in_cb`/`out_cb`/`config_cb`; causal-link gate applies to all three).
  - `UpsampleBilinearProgramFactory`: input (`halo_in`) and output — both **clean** (borrowed-memory DFB on `halo_cb`/`out_cb`).
- **TensorParameter relaxation:** none (sheet: `none` on all four rows).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**
  - `UpsampleNearestFloatProgramFactory`: 1 CB (`output_cb_index`) — legal (1,1): reader produces, writer consumes.
  - `UpsampleMultiCoreInterleavedProgramFactory`, row-major path: 1 CB (`src0_cb_index` reused as output) — legal (1,1): reader produces, writer consumes.
  - `UpsampleMultiCoreInterleavedProgramFactory`, tiled path: 2 CBs — `src0_cb_index` legal (1,1) (reader produces / borrowed `untilize.cpp` compute consumes), `output_cb_index` legal (1,1) (compute produces / writer consumes).
  - `UpsampleMultiCoreShardedProgramFactory`: 3 CBs, all dual-instance work-split (reader+writer instances of the same kernel source, `writer_upsample_multi_core_sharded.cpp`, over the same `cores_with_work`) — `in_cb` **1P+1C** (both instances raw-peek via `get_read_ptr()`), `config_cb` **1P+1C** (both instances raw-peek), `out_cb` **1P+1C** (both instances raw-write disjoint offsets via `noc.async_read(..., out_dfb, ..., {.offset_bytes=...})`, output resident — nothing drains it).
  - `UpsampleBilinearProgramFactory`: 5 CBs — `halo_cb` **1P+1C** (reader+writer instances of `reader_bilinear_multi_core_sharded.cpp`, both raw-peek via `get_read_ptr()`), `tilize_reduce_cb_0` legal (1,1) (reader instance produces / compute consumes), `tilize_reduce_cb_1` legal (1,1) (writer instance produces / compute consumes), `in_scalar_cb_id1` legal (1,1) (reader instance produces / compute consumes), `in_scalar_cb_id2` legal (1,1) (writer instance produces / compute consumes), `out_cb` **self-loop** (single toucher: the compute kernel, producer-only via the raw pointer mutation flagged above — no downstream on-device consumer).
  - No dead CBs found anywhere.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB in this op needed the multi-binding advanced option.
- **Cross-op / shared kernels:**
  - `UpsampleMultiCoreInterleavedProgramFactory` (tiled path only) instantiates `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp` by file path — a **cross-family donor** (owning family: `data_movement/untilize`). A `_metal2` fork **already exists** beside it at the same path with a `_metal2` suffix (`.../untilize/device/kernels/compute/untilize_metal2.cpp`) — bind that fork, don't create a new one. Other consumers of the legacy file (the sunset list, not an authorization to bundle-port): `data_movement/untilize`'s own four factories (`untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp`, `untilize_single_core_program_factory.cpp`, `untilize_multi_core_sub_core_grids_program_factory.cpp`, `untilize_multi_core_parallelize_column_program_factory.cpp`) and `data_movement/untilize_with_unpadding`'s three factories.
  - All four factories `#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>` (in-family shared, category "pool") — only `experimental::local_addr(...)` is actually called (in `reader_bilinear_multi_core_sharded.cpp`), a pure NoC-address-args helper over Device-2.0-native types; ✓ excellent, no concern.
  - Three kernels (`reader_bilinear_multi_core_sharded.cpp`, `reader_upsample_nearest_float.cpp`, `writer_upsample_nearest_float.cpp`) also `#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>` (in-family shared) — pure compile-time/runtime fixed-point math, no resource-handle types at all; ✓ clean.
- **RTA varargs:** none — every kernel reads a fixed, small set of RTAs at fixed positions; no counted loop over `get_arg_val`/`get_common_arg_val` and no data-selected index read.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - Op-level roll-up: ✓ clean (function-call escapes) with one ⚠-worth-noting file-path borrow (untilize compute kernel, fork already exists — see Heads-ups).
  - Summary table:

    | Op kernel | Donor file | Category | Shape used | Status |
    |---|---|---|---|---|
    | all 6 own kernels | `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | in-family shared (`pool`) | `experimental::local_addr(uint32_t, uint8_t)` — plain helper over Device-2.0-native `Noc`/`UnicastEndpoint` types, no Semaphore/TensorAccessor/CB handle in the shape table | ✓ excellent |
    | `reader_bilinear_multi_core_sharded.cpp`, `reader_upsample_nearest_float.cpp`, `writer_upsample_nearest_float.cpp` | `ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp` | in-family shared (`pool`) | pure math, no resource handles | ✓ clean |
    | `writer_upsample_interleaved.cpp` (factory-instantiated, tiled path) | `ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp` | cross-family donor (file-path instantiation, not `#include`) | calls `compute_kernel_lib::untilize<...>()` from the official shared `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (category: official shared kernel library) | ✓ Device-2.0-native; `_metal2` fork already exists |

  - Per-call detail: nothing below ✓/excellent to detail — no ⚠/✗/⭐ entries.

- **Relaxation candidates** (mined from a custom hash on a gated op): none — no factory has a custom hash to mine (this op has none, gated or otherwise).

- **TTNN factory analysis:** (sheet-derived, cross-checked against code)

  | Factory | Concept | Custom hash | `get_dynamic_runtime_args` | `override_runtime_arguments` | Pybind `create_descriptor` | Op-owned tensors | Secretly SPMD |
  |---|---|---|---|---|---|---|---|
  | Bilinear | descriptor | no | no | no | no | no | n/a |
  | MultiCoreInterleaved | descriptor | no | no | no | no | no | n/a |
  | MultiCoreSharded | WorkloadDescriptor | no | no | no | no | **yes** | **yes** |
  | NearestFloat | descriptor | no | no | no | no | no | n/a |

## Misc anomalies

- **Dead include:** `device/upsample_device_operation.hpp:17` includes `<tt-metalium/global_circular_buffer.hpp>` but no `GlobalCircularBuffer` construct is used anywhere in the op directory. Harmless today (confirmed N/A for the Appendix A feature scan), but worth trimming — a future reader could mistake it for a signal that GCB is in use here.
- **Missing `reserve_back` before a `push_back`:** `device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp:400-406` calls `scalar_dfb.push_back(1)` (writing weights via `fill_four_val(scalar_dfb.get_write_ptr(), ...)`) with no preceding `scalar_dfb.reserve_back(1)` anywhere in the kernel. The CB is double-buffered host-side (`in_scalar_cb_npages = 1 * buffering_factor` = 2 pages) so this likely never overflows in practice, but it's a latent correctness risk (rule 3: every `reserve_back` should be paired, and implicitly every `push_back` should be preceded by a matching `reserve_back`) independent of the Metal 2.0 port. Route to the ops team; the port does not act on it.

## Questions for the user

1. **Bilinear Device 2.0 fix — semantic equivalence of `push_back` vs. the hand-rolled pointer arithmetic:** `llk_push_pages_bilinear` (`device/kernels/compute/bilinear.cpp:16-26`) omits the `llk_push_to_brisc(...)` consumer-credit signal that `DataflowBuffer::push_back()` (via `llk_push_tiles`) would perform. Since `out_cb` has no on-device consumer (borrowed-memory output CB), this looks intentional/harmless, but the Device 2.0 team should confirm whether the correct replacement is `out_dfb.push_back(tiles_per_reduction)` (simpler, but adds the credit signal) or `out_dfb.evil_set_write_ptr(...)` (preserves the current no-signal behavior exactly) before making the change.

## Recipe notes

None — the recipe's guidance mapped cleanly onto this op. The one subtlety worth flagging back to the recipe maintainer: the "sanctioned free function" carve-out for `get_local_cb_interface(cb_id)` (Device 2.0 prerequisite, Green bullet) reads as covering *any* call to that free function, but in practice it only covers **read-only metadata access** — this op's compute kernel calls the same free function to **mutate** `.fifo_wr_ptr`/`.fifo_wr_tile_ptr`, which is a materially different (and non-sanctioned) usage of the identical function name. A future auditor skimming for "is `get_local_cb_interface` used → sanctioned, skip it" would false-negative this exact case. Worth a one-line addendum to the Green bullet clarifying that the carve-out is for reads, not writes, to the interface struct.
