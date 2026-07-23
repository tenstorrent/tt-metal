# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/gather`

One device operation, four program factories (all `create_descriptor` → `ProgramDescriptor`):

- **`GatherDeviceOperation`** (`device/gather_device_operation.hpp`)
  - `SingleRowSingleCore` — TILE layout, single-core (`gather_program_factory.cpp:50`)
  - `SingleRowMultiCore` — TILE layout, multi-core (`gather_program_factory.cpp:206`)
  - `RmSingleRowSingleCore` — ROW_MAJOR layout, row-distributed single-core (`gather_program_factory.cpp:366`)
  - `RmSingleRowMultiCore` — ROW_MAJOR layout, column-distributed multi-core (`gather_program_factory.cpp:523`)

The op owns all eight of its kernels (`device/kernels/dataflow/`). The RM kernels also `#include` one cross-op donor header (`data_movement/common/kernels/common.hpp`) for `noc_async_read_sharded` / `noc_async_write_sharded`. The `tosa/` subtree is a separate host-side entry wrapper and instantiates no kernels; it is out of audit scope.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `1e584828bbf 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/gather` |
| **Overall** | **RED at op level; subset {`SingleRowSingleCore`, `SingleRowMultiCore`} (the TILE factories) is clear** |
| **DOps / Factories** | `GatherDeviceOperation` → `SingleRowSingleCore`, `SingleRowMultiCore`, `RmSingleRowSingleCore`, `RmSingleRowMultiCore` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all eight kernels + the donor are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok (RM factories only: one shared-pool donor, Device 2.0 native `TensorAccessor` shape) |
| *Feature Support* — overall | GREEN (every Appendix A entry N/A) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **TILE factories: Yes** · **RM factories: No** — no sheet row (spreadsheet-broken) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all four factories, confirmed from code) |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | TILE: Yes (sheet) · RM: **unknown — no sheet row** (→ readiness-sheet owner) |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (TILE factories) |
| *Port work* — Offset base pointer | none (no address fold anywhere) |
| *Port work* — Tensor bindings (per binding) | TILE: all **Case 1** (via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | TILE: none · RM: **Class 2 drop** (redundant page-size), verify note below |
| *Port work* — CB endpoints | c_0 legal 1:1 · c_1 self-loop · c_2 legal 1:1 (both TILE and RM) |

## Result

**RED at op level; subset {`SingleRowSingleCore`, `SingleRowMultiCore`} (the TILE factories) is clear.**

The two TILE factories clear every gate — Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (sheet `Is able to port? == yes`), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (no 3rd arg) — so a **TILE-only subset port can proceed**. A brief is issued for that subset (`METAL2_PORT_BRIEF.md`).

The two ROW_MAJOR factories (`RmSingleRowSingleCore`, `RmSingleRowMultiCore`) are **blocked**: they have **no row in the readiness sheet** ("Operations analysis"), which per the recipe is a *spreadsheet-broken* condition. The gate that decides portability (`Is able to port?`) and the correctness axis it depends on (`Is safe to port?`) do not exist for these factories, and the recipe forbids re-deriving the correctness call from code. **Routed to the readiness-sheet owner (Diego) to add and classify the two RM factory rows.** These factories are recent additions (2026 copyright); the sheet simply has not caught up. Their code shape is otherwise identical in kind to the TILE factories (same `descriptor` concept, no custom hash, no runtime-args update, no pybind descriptor), so once the sheet rows land they are expected to clear — see the RM-specific port work already gathered below (it is not lost; it feeds the re-audit).

## Gate detail

- **TTNN factory concept (`Is able to port?`):**
  - **TILE factories — GREEN.** Sheet rows (`gather_device_operation.hpp`, rows for `SingleRowSingleCore` and `SingleRowMultiCore`): `Concept = descriptor`, `Custom hash = no`, both `Runtime-args update` columns `no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Is able to port? = yes`, `TensorParameter relaxation = none`. Cross-check against code confirms every cheaply-checkable column: `create_descriptor` returns a `ProgramDescriptor` (descriptor concept); no `compute_program_hash` override in the device op; no `get_dynamic_runtime_args` / `override_runtime_arguments` in the factory; no `create_descriptor` binding in `gather_nanobind.cpp`. No cross-column invariant is violated. Gate clear.
  - **RM factories — RED (spreadsheet-broken).** `RmSingleRowSingleCore` and `RmSingleRowMultiCore` have **no row** in the sheet. Per the prerequisite subject's routing, a missing row means the sheet is broken for those factories → GATE → **readiness-sheet owner** to reconcile (add the rows and supply `Is safe to port?`). This is the primary — and, pending the sheet, the only known — blocker for the RM factories.

- **Device 2.0 (every kernel used): GREEN.** All eight kernels use Device 2.0 idioms exclusively — `Noc`, `noc.async_read` / `noc.async_write` / `noc.async_read_barrier`, `TensorAccessor` / `TensorAccessorArgs`, and the `DataflowBuffer` (TILE kernels) or `CircularBuffer` (RM kernels) wrapper objects with method-form FIFO calls (`.reserve_back()`, `.push_back()`, `.wait_front()`, `.pop_front()`, `.get_read_ptr()`, `.get_write_ptr()`). No legacy addr-gen (`InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`), no raw `noc_async_read`, no manual CB-index management.

  The one shape worth calling out (and clearing): the TILE kernels use the CB-index free functions `get_tile_size(cb)`, `get_dataformat(cb)`, and `get_tile_hw(cb)` (e.g. `gather_reader_single_row_single_core.cpp:136-146`). These are **not** Device 2.0 holdovers. `get_tile_size(cb)` is explicitly sanctioned by the recipe, and a survey of the current Device 2.0 kernel surface shows `get_dataformat(cb)` and `get_tile_hw(cb)` are used as free functions in **134** Device-2.0 kernel files — including already-Metal-2.0-ported kernels that call them with a binding token (`get_dataformat(dfb::in)` in `experimental/quasar/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned_metal2.cpp:32`). They are part of the sanctioned tile/format metadata accessor set; the Metal 2.0 port relocates them onto the bound object as a whitelisted syntax swap, but they do not move the Device 2.0 boundary. No violation, no gate.

  The RM kernels' donor (`noc_async_read_sharded` / `noc_async_write_sharded` in `data_movement/common/kernels/common.hpp`) is Device 2.0 native — templated on a `TensorAccessor` address generator, using `Noc`, `CoreLocalMem`, `tensor.dspec()`, `tensor.get_aligned_page_size()`. No donor-side Device 2.0 gate.

- **Feature compatibility:** every Appendix A entry N/A. Clean scan.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateCircularBuffer(..., global_cb)`, no `.remote_index(`, no `.global_circular_buffer` field. Plain `CBDescriptor`s only. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. (The kernel-side `offset` params to `noc_async_*_sharded` are the unrelated kernel-side feature — false-positive guard.) |
  | GlobalSemaphore | N/A | No semaphores of any kind; no `GlobalSemaphore`, no `CreateGlobalSemaphore`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` (`GatherInputs`) carries a fixed set (input, index, optional output) — no `std::vector<Tensor>`. Kernels read every CTA at a constexpr offset; no runtime-varying `get_compile_time_arg_val(i)` index. |

- **CB endpoints (GATE-free):** three CBs per factory, identical census on each active node for all four factories. All resolvable at port time; nothing blocks.
  - **`c_0` input-tensor CB** — writer FIFO-produces (`reserve_back`/`push_back`), reader FIFO-consumes (`wait_front`/`pop_front` + `get_read_ptr` peek). One locked producer + one locked consumer → **plain 1:1**, no action.
  - **`c_1` input-index CB** — the reader both FIFO-produces and FIFO-consumes it; no other kernel touches it. One toucher → **self-loop** (bind the reader PRODUCER *and* CONSUMER). Legal on Gen1.
  - **`c_2` output CB** — reader FIFO-produces (`reserve_back`/`push_back` + `get_write_ptr`), writer FIFO-consumes (`wait_front`/`pop_front`). One locked producer + one locked consumer → **plain 1:1**, no action.

  No dead CBs, no hidden second writer, no multi-reader — a FIFO trace and a raw-pointer scan agree. Census is config-independent here (no split-reader / mcast / dual-instance work-split; the RM multi-core factory instantiates *distinct* reader and writer sources, not one source twice).

- **Offset base pointers: GREEN.** No address RTA folds a host-side offset into a base. The factory never calls `->address()` at all — it passes `Buffer*` objects into `emplace_runtime_args` (the framework's `BufferBinding` path) and the kernel receives a clean base via `get_arg_val<uint32_t>(0)`. Every offset the kernels need is passed as a **separate** scalar arg and applied on-device: the RM slice offset rides its own arg (`w_start` → `input_index_byte_offset` at `gather_reader_rm_single_row_multi_core.cpp:43`, fed to the donor's `offset` parameter at `:60`), and the TILE page indices are computed in-kernel (`h * Wt_index + w`). No Type 1, Type 2, Type 3, or Type 4 pattern. Gather is in neither table of the (dated) offset-base-pointer triage; the scan above is the source of truth.

- **TensorAccessor 3rd argument:**
  - **TILE factories — GREEN (no 3rd arg).** Every `TensorAccessor` in the four TILE kernels is 2-arg (`TensorAccessor(args, addr)`) — e.g. `gather_reader_single_row_single_core.cpp:138`, `gather_writer_single_row_single_core.cpp:51,54`. Nothing to classify.
  - **RM factories — Class 2 (redundant → drop), with a verification note.** All four RM kernels pass an explicit per-shard page size as the `TensorAccessor` 3rd argument:
    - `gather_reader_rm_single_row_single_core.cpp:42` (`input_index_per_shard_page_size_bytes`)
    - `gather_writer_rm_single_row_single_core.cpp:41` (input), `:43` (output)
    - `gather_reader_rm_single_row_multi_core.cpp:47` (`input_index`)
    - `gather_writer_rm_single_row_multi_core.cpp:46` (input), `:48` (output)

    The value comes from `per_shard_page_size_bytes()` (`gather_program_factory.cpp:38-46`): `shard_W * element_size` for BLOCK/WIDTH-sharded, and the full row (`W * element_size`) otherwise. In every case this **is the tensor's true logical page size** — for a B/W-sharded buffer the shard-row page, for an interleaved / height-sharded buffer the full-row page — i.e. exactly what `TensorAccessorArgs::AlignedPageSize` supplies implicitly when the 3rd arg is omitted (the constructor default at `tensor_accessor.h:81-85`). So it is **correct magnitude** (equals `buffer->page_size()`), not a wrong-magnitude value → Class 2, not Class 3/4. It is not a raw-pack page nor a sub-page fragment → not Special. There is no custom hash, so the program recompiles per shape and the page size is fixed within a cache entry → not the Class-1 dynamic case. **Port action: drop the arg; Metal 2.0 supplies `aligned_page_size` from the binding.**

    *Verification note for the ops team / porter (does not gate; it is a drop-time check):* the manual value is passed **unaligned** (`W * element_size`), whereas the implicit default is the allocator-**aligned** page. On an *interleaved* accessor this is inert (the interleaved path realigns), so the drop is a pure no-op there. On a *sharded* accessor the stride is used verbatim (`tensor_accessor.h:310`, via `get_aligned_page_size()` in the donor), so before dropping, confirm the implicit aligned page addresses identically to the manual unaligned value for B/W-sharded RM — which holds when shard rows are allocator-aligned (the normal case). This is the same value either way for aligned shard rows; the note is to make the porter check rather than assume.

## Port-work summary  *(mirrors the brief — TILE subset)*

- **Tensor bindings** (per binding, TILE factories): all **Case 1** (via `TensorAccessor`).
  - `input_index_tensor` — reader; `Buffer*` bound via `emplace_runtime_args`, kernel feeds the base into `TensorAccessor(input_index_tensor_args, input_index_tensor_buffer_addr)` (`gather_reader_single_row_single_core.cpp:138`). Express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`.
  - `input_tensor` — writer; `TensorAccessor(input_tensor_args, input_tensor_buffer_addr)` (`gather_writer_single_row_single_core.cpp:51`). Case 1.
  - `output_tensor` — writer; `TensorAccessor(output_tensor_args, output_tensor_buffer_addr)` (`gather_writer_single_row_single_core.cpp:54`). Case 1.

  Delivery today is the `Buffer*`-binding form (`BufferBinding`, patched on cache hits) — correct today, not the silent-wrong hazard; the typed `TensorParameter` binding supersedes it. All three bindings are mechanical Case-1 conversions; the address-via-RTA plumbing and the `TensorAccessorArgs` CTAs disappear.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash).
- **TensorAccessor 3rd arg:** none for the TILE subset.
- **CB endpoints:** self-loop `c_1` (input-index CB); `c_0` and `c_2` are plain 1:1 (no action). All configs.

## Heads-ups  *(mirrors the brief — TILE subset)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader, no dual-instance work-split.
- **Cross-op / shared kernels:** none for the TILE factories — the four TILE kernels `#include` only `gather_common.hpp` (in-directory) and `tt_metal/*` HAL headers. No borrowed kernel files (gather owns every kernel it instantiates).
- **RTA varargs:** none. The TILE kernels read a fixed set of RTAs at distinct constant indices (`get_arg_val<uint32_t>(0..4)` in the reader, `(0..3)` in the writer) — ordinary named-arg port work, no loop-indexed or data-selected reads.

## Team-only

- **Out-of-directory coupling & donor shape (RM factories only):**
  - **Op-level roll-up: ✓ clean.** One donor header, one donor class, Device 2.0 native shape.
  - **Summary table**

    | Op kernel | Donor file | Donor class | Functions used | Shape | Status |
    |---|---|---|---|---|---|
    | `gather_reader_rm_single_row_single_core.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared pool (`data_movement/common`) | `noc_async_read_sharded` | `TensorAccessor` templated (Shape 1) | ✓ excellent |
    | `gather_writer_rm_single_row_single_core.cpp` | same | same | `noc_async_read_sharded`, `noc_async_write_sharded` | Shape 1 | ✓ excellent |
    | `gather_reader_rm_single_row_multi_core.cpp` | same | same | `noc_async_read_sharded` | Shape 1 | ✓ excellent |
    | `gather_writer_rm_single_row_multi_core.cpp` | same | same | `noc_async_read_sharded`, `noc_async_write_sharded` | Shape 1 | ✓ excellent |

    Both donor functions are templated on the address-generator type and are called with a `TensorAccessor` — Shape 1 (Device 2.0 native, the porter passes `TensorAccessor(tensor::name)`). No `uint32_t sem_*`, no `CircularBuffer&`, no old-style addr-gen. This coupling does not gate and does not sequence-block; it induces the ordinary port-the-family-together coupling if `data_movement/common` is rewritten, but the shape crosses cleanly. No file-path kernel instantiation of borrowed sources (gather owns all its kernel files).
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** all four factories are `descriptor` concept (code-confirmed). No op-owned tensors, no MeshWorkload, no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept for the cleared (TILE) factories: `MetalV2FactoryConcept`.
- **CB endpoints (RM factories):** identical census to the TILE factories (c_0 1:1, c_1 self-loop, c_2 1:1). Recorded here for completeness; the RM factories are gated on the sheet issue, so this is not carried into a brief until they unblock.

## Misc anomalies  *(team-only, non-gating)*

- **Unused `dprint.h` include.** `gather_writer_single_row_single_core.cpp:10` and `gather_writer_single_row_multi_core.cpp:9-10` include `api/debug/dprint.h`, but no `DPRINT` statement appears in either kernel. Dead include.
- **Deprecated donor overload.** The RM kernels call the 5-arg `noc_async_read_sharded(l1_addr, tensor, id, offset, size)` / `noc_async_write_sharded(...)` overloads, which are marked `[[deprecated]]` in `data_movement/common/kernels/common.hpp:300,345` in favor of the leading-`Noc` overloads. Functionally identical (the deprecated form constructs `Noc noc;` internally and forwards), so it is not a Device 2.0 concern — just a deprecation the RM kernels should eventually track. Donor-side; not porter work.
- **Silent default in the format-size dispatch.** `get_value_from_tile` / `write_value_to_tile` in `gather_common.hpp:32-34,63-66` fall back to a `uint16_t` read/write for any `data_format_size` not in {1,2,4,8}, rather than asserting. A wrong element size would silently mis-read rather than fail loudly. Latent, pre-existing; not porter work.

## Questions for the user  *(for routing, not blocking)*

1. **RM factory sheet rows:** `RmSingleRowSingleCore` and `RmSingleRowMultiCore` are absent from the "Operations analysis" sheet. Confirm the routing to the sheet owner (Diego) to add both rows — the correctness axis (`Is safe to port?`) is theirs to supply and I am not permitted to re-derive it. Everything else about these two factories looks port-ready by code shape.

## Recipe notes

- **Partial sheet coverage within one op.** The recipe's "op has no row → spreadsheet-broken → GATE" routing is written for a *whole* op being absent. Here the op is *half* present: two factories have rows and clear, two have no row. I applied the missing-row gate per-factory (the sheet is "one row per factory") and offered the covered factories as a clean subset via Code-path scope. This composition of "spreadsheet-broken (per factory)" with "Code-path scope (clean subset)" worked, but the recipe does not explicitly walk through a *partial-coverage* case — worth a sentence in the prerequisite subject so the next auditor does not read "the op has no row" as all-or-nothing.
- **`get_dataformat(cb)` / `get_tile_hw(cb)` sanctioning.** The Green-bullet sanctioned list names only `get_tile_size` and `get_local_cb_interface`, but the breadcrumb ("check the current Device 2.0 surface") is what actually resolved these two — they are pervasive in Device 2.0 and even Metal-2.0-ported kernels. Consider adding `get_dataformat(cb)` and `get_tile_hw(cb)` to the explicit sanctioned list (they are the rest of the metadata-accessor set alongside `get_tile_size`), so an auditor need not run the 134-file survey to reach the same conclusion.
