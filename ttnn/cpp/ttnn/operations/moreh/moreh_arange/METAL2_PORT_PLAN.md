# Port Plan — moreh/moreh_arange

> **SUPERSEDED — Outcome: CAPITULATED.** This plan was executed and the port built and ran
> 8/9 on-device, but the create-output path hit a **framework** tensorless-dispatch block
> (the MetalV2 adapter sources the MeshDevice only from `tensor_args`, which is empty for this
> input-tensor-less, output-creating op). All code was reverted to the legacy
> `create_descriptor` form with no regression. See `METAL2_PORT_REPORT.md` for the full
> root-cause and framework handoff. The spec shape below remains an accurate re-attempt
> blueprint for once the adapter gains a tensorless device-sourcing path.

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_arange`, ported from the
`descriptor` (`ProgramDescriptor`) API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — realized as a **direct-descriptor** op
  (`HasDirectDescriptor`): `create_descriptor` is a static member of
  `MorehArangeOperation`; there is **no** `program_factory_t` variant.
  (`device/moreh_arange_device_operation.hpp:30`, `device/moreh_arange_program_factory.cpp:21`.)
- Variants: single factory. One `create_descriptor` selects one of two op-owned
  writer kernels at runtime by the `untilize_out` attribute:
  - `untilize_out == false` → `device/kernels/writer_moreh_arange.cpp`   (TILE output)
  - `untilize_out == true`  → `device/kernels/writer_moreh_arange_rm.cpp` (ROW_MAJOR output)
- Custom `compute_program_hash`: none — already default reflection-based hash
  (audit confirmed).

*(Target concept `MetalV2FactoryConcept` inherited from the audit; carried forward
in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels
Runtime kernel-source selection on one axis (`untilize_out`). Two selectable
sources, one `KernelSpec` (source + arg schema chosen by `untilize_out`).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| writer (tile) | `device/kernels/writer_moreh_arange.cpp` | `all_cores` | `TensorAccessorArgs(*output.buffer())` (→ drops) | none | `dst_addr`(→drop), `tile_offset`, `num_tiles`, `start`, `step` | none | `OUTPUT_DTYPE_{BFLOAT16,INT32,FLOAT32}` | `WriterConfigDescriptor{}` |
| writer (rm) | `device/kernels/writer_moreh_arange_rm.cpp` | `all_cores` | `TensorAccessorArgs(*output.buffer())` (→ drops) | none | `dst_addr`(→drop), `tile_offset`, `num_tiles`, `start`, `step`, `element_size` | none | `OUTPUT_DTYPE_{BFLOAT16,INT32,FLOAT32}` | `WriterConfigDescriptor{}` |

Note: the factory pushes 6 RTAs for **both** kernels, but the tile kernel reads only
args 0–4 (never `element_size`, arg 5) — see audit "Misc anomalies". Post-port the
tile path does not carry an `element_size` named RTA (schema would reject a
superfluous arg); the RM path does.

### CBs
| buffer_index | total_size | page_size | data_format | tile | core_ranges | touch census |
|---|---|---|---|---|---|---|
| `tt::CBIndex::c_16` | `tile_size(out_data_format)` | `tile_size(out_data_format)` | `out_data_format` | default (unset) | `all_cores` | **one toucher** — writer only: `reserve_back(1)` + `get_write_ptr()`; no `push_back`/`wait_front`/`pop_front` |

Scratch staging buffer: the writer reserves a tile, fills it via `CoreLocalMem`, then
`noc.async_write`s the tile to DRAM. `num_entries` = total_size/page_size = 1.

### Semaphores
None.

### Tensor accessors
| tensor | host RTA slot (legacy) | kernel construction (legacy) | case |
|---|---|---|---|
| `output` (output) | arg 0 = `output.buffer()` (`program_factory.cpp:90`) | `TensorAccessor(dst_args, dst_addr)` w/ `dst_args = TensorAccessorArgs<0>()` | Case 1 (via `TensorAccessor`) |

### Work split
`split_work_to_cores(grid, Wt)` where `Wt = div_up(W, TILE_WIDTH)`, `W = output.padded_shape()[-1]`.
Yields `(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2)`.
Core-index mapping is **column-major**: `core = {i / grid.y, i % grid.y}` over `i ∈ [0, num_cores)`.
No multi-`KernelDescriptor` work-split multiplicity (a single writer `KernelDescriptor` over `all_cores`).

### Cross-op kernels
None — both writer kernels are op-owned, file-path-instantiated; `#include`s resolve
only to `api/*` (tt_metal HAL/LLK). (Audit "Team-only" confirmed no escapes.)

### Flags
- No unreferenced kernel files in the op directory.
- No descriptor type outside the audit's scan.
- Dead RTA (`element_size`, arg 5) on the tile path — naturally drops when RTAs are named.

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none.
- Implementation notes: the legacy op is **direct-descriptor** (no `program_factory_t`).
  The framework's default `resolve_program_factory` only synthesizes a
  `DirectDescriptorFactory` around `create_descriptor` — there is **no** direct
  `create_program_artifacts` path. So the port introduces a nested factory struct
  `MorehArangeOperation::MorehArangeProgramFactory` carrying `create_program_artifacts`,
  plus `using program_factory_t = std::variant<MorehArangeProgramFactory>;`, and removes
  the direct `create_descriptor`. Single-element variant ⇒ no `select_program_factory`
  needed. This is the TTNN-wiring shape used by ported single-factory ops (e.g.
  `experimental/quasar/fold`).

## Planned Spec Shape
- **KernelSpecs**: 1 — the writer. Source + `runtime_arg_schema` chosen by `untilize_out`.
- **DataflowBufferSpecs**: 1 — `arange_out` (legacy `c_16`). `entry_size = out_tile_size`,
  `num_entries = 1`, `data_format_metadata = out_data_format` (feeds the tile kernel's
  `get_tile_size()`; tile default 32×32). **Self-loop** (see Applied Patterns).
- **SemaphoreSpecs**: none.
- **TensorParameters**: 1 — `arange_output` (from `output.tensor_spec()`), one `TensorBinding`.
- **WorkUnitSpecs**: 1 — the writer over `all_cores`.
- **Op-owned tensors**: none.

## Preserved Multiplicity
None — no work-split multiplicity in legacy (single writer `KernelDescriptor` over `all_cores`).

## Dropped Plumbing
- **`TensorAccessorArgs` plumbing** → replaced by `TensorBinding` end-to-end.
  Host: `TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args)` (`program_factory.cpp:64`).
  Kernel: `constexpr auto dst_args = TensorAccessorArgs<0>();` +
  `TensorAccessor(dst_args, dst_addr)` (`writer_moreh_arange.cpp:26-27`,
  `writer_moreh_arange_rm.cpp:27-28`) → `TensorAccessor(tensor::dst)`.
- **Buffer-address RTA** → replaced by `TensorBinding`. Host RTA slot 0 =
  `output.buffer()` (`program_factory.cpp:90`); kernel `dst_addr = get_arg_val<uint32_t>(0)`
  (both kernels, line 16). Dropped.
- **Magic CB index** → replaced by `DFBBinding`. Kernel
  `constexpr uint32_t cb_out = tt::CBIndex::c_16;` (`writer_moreh_arange.cpp:22`,
  `writer_moreh_arange_rm.cpp:23`) → `DataflowBuffer dfb_out(dfb::out)`.
- **Positional RTAs** → named. `tile_offset`, `num_tiles`, `start`, `step`
  (+ `element_size` on the RM path).
- **Page-size 3rd CTA/RTA**: none (both kernels pass a 2-arg `TensorAccessor`; audit GREEN).
- **Semaphore-ID RTAs**: none.

## Applied Patterns
- [Sync-free / single-ended CB → self-loop DFB](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `arange_out` is a one-toucher scratch buffer (writer only, `reserve_back` +
  `get_write_ptr`, no FIFO drain). Bind the single writer kernel as **both** PRODUCER and
  CONSUMER (shared accessor `out`). Legal on Gen1 for a DM kernel. Same disposition under
  both `untilize_out` configs. DM self-loop is a Gen1-only shape (Quasar-uplift's concern).
- [DFB metadata via the object (whitelist rule 7)](../port/metal2_port.md#kernel-side-whitelist):
  tile kernel's `get_tile_size(cb_out)` (`writer_moreh_arange.cpp:24`) → `dfb_out.get_tile_size()`.

## Deferred / Flagged
- None new. RTA→CRTA opportunity noted for a later cleanup (not port work): `start`, `step`,
  and `element_size` are identical on every node and are morally CRTAs; kept as per-node
  RTAs to mirror legacy exactly (recipe forbids RTA→CRTA during a port). See report.
