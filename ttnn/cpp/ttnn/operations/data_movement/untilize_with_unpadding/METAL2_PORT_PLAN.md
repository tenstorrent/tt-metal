# Port Plan — `data_movement/untilize_with_unpadding`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`, ported from
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to Metal 2.0 (`MetalV2FactoryConcept`,
`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

> ## ⚠ SCOPED PORT — 4 of 5 factories planned, **3 landed**
>
> The audit is **RED at op level** (`METAL2_PREPORT_AUDIT.md`); the brief
> (`METAL2_PORT_BRIEF.md`) is a **config-scoped subset brief** covering the four factories that
> cleared. This plan was written for those four:
>
> - `UntilizeWithUnpaddingSingleCoreProgramFactory` — **ported**
> - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` — **ported**
> - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` — **attempted, CAPITULATED**
> - `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` — **ported**
>
> **`MultiCoreBlockInterleaved` stays on `create_descriptor`.** Its faithful Metal 2.0 shape
> (per-sub-region DFB pairs, below) trips a **Metal 2.0 framework defect**: the per-node DFB
> config region is *sized* by DFB count but *addressed* by DFB id, so any program that partitions
> its DFBs across disjoint node sets overruns that region and silently corrupts L1. A second,
> unresolved wrong-numerics failure survives even when the DFBs are collapsed to a single shared
> pair. Both are written up with repro and evidence in `METAL2_PORT_REPORT.md` → Handoff points.
> The plan text for this factory is retained below **as the design record for whoever re-attempts
> it once the framework defect is fixed** — it is a description of what was built and why, not of
> what shipped.
>
> `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` is **out of scope** — blocked on the
> Device 2.0 gate (two donor kernels in the `ttnn/cpp/ttnn/kernel/` shared pool are still on
> Device 1.0 free-function idioms). It **stays on `create_descriptor`**. The `program_factory_t`
> variant therefore holds a mixed-concept set post-port (three Metal 2.0, two `ProgramDescriptor`),
> which the framework dispatches per-factory (recipe: *the atomic unit of a port is one
> ProgramFactory*).

**Invoker-resolved open items carried into this plan** (both were left open by the audit):

1. **Readiness sheet confirmed.** All four in-scope factory rows read
   `Concept = descriptor`, `Is safe to port? = yes`, `Is able to port? = yes`.
   The sheet is slightly stale: it also lists a sixth factory, `MultiCoreColInterleaved`,
   which PR #50383 deleted on 2026-07-24. **Five factories in the code is correct**, and the
   audit's own factory-set enumeration (five) is the authoritative one.
2. **`writer_unary_stick_layout_wh_multicore.cpp` is forked, not modified in place.** It lives in
   *this* op's kernel directory but is also bound **by file path** from `data_movement/untilize`'s
   block factory (`untilize_multi_core_block_program_factory.cpp:196`), which stays on
   `create_descriptor`. `untilize` is not co-migrated.

---

## Legacy Inventory

Legacy source of truth: the five `*_program_factory.cpp` files under `device/factories/` and the
kernel sources they bind. Only the four in-scope factories are inventoried.

### Legacy factory shape

- **Concept**: `ProgramDescriptorFactoryConcept` — each factory exposes
  `static tt::tt_metal::ProgramDescriptor create_descriptor(const UntilizeWithUnpaddingParams&, const Tensor&, Tensor&)`
  (`*_program_factory.hpp:14` in all five).
- **Variants**: one `DeviceOperation` (`ttnn::prim::UntilizeWithUnpaddingDeviceOperation`) with a
  5-alternative `program_factory_t` (`device/untilize_with_unpadding_device_operation.hpp:24-29`).
  `tensor_args_t = Tensor`, `tensor_return_value_t = Tensor` (one input, one output).
  Each factory is inventoried separately below.
- **Custom `compute_program_hash`**: **none** — no `compute_program_hash` anywhere under `device/`.
  The op already uses the default reflection-based hash; nothing to delete.
- **`get_dynamic_runtime_args` / `override_runtime_arguments` / pybind `create_descriptor`**: all
  absent (audit *Gate detail*). No device-op-class edit is forced by this port.
- **Op-owned tensors**: none.
- **Semaphores**: none anywhere in the op.

---

### Variant: SingleCore (`untilize_with_unpadding_single_core_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` **(cross-op)** | `core` (1 node: `{0,0}`, or first node of `sub_core_grids`) | `TensorAccessorArgs(*src0_buffer)` only (`:130`) | — | `{src0_buffer(Buffer*), num_tiles, 0u}` (`:187`) | — | — | `ReaderConfigDescriptor{}` |
| writer | `untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp` **(op-owned)** | `core` | `{FLOAT32_DTYPE, unpadded_stick_size}` + `TensorAccessorArgs(*dst_buffer)` (`:132-137`) | — | 15 values (`:188-204`), slot 0 = `dst_buffer(Buffer*)` | — | — | `WriterConfigDescriptor{}` |
| compute | `data_movement/untilize/device/kernels/compute/untilize.cpp` **(cross-op)** | `core` | `{num_tiles/num_tiles_per_block, num_tiles_per_block, src0_cb_index, output_cb_index}` (`:158-162`) | — | — | — | `DST_ACCUM_MODE=1` iff input format ∈ {Int32, UInt32, Float32} | `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode}` |

Writer RTA slots (`:188-204`), in order:
`dst_buffer`, `output_w`, `padded_W_diff_blocks`, `output_z`, `padded_Z_diff_blocks`, `output_y`,
`padded_Y_diff_blocks`, `num_leftover_Y`, `output_x`, `padded_stick_size`, `num_blocks_w_input`,
`num_blocks_w_output`, `num_blocks_w_diff`, `block_row_size`, `block_row_leftover_size`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`src0_cb_index=0`) | `num_tiles_per_block * input_single_tile_size` | `core` | `input_cb_data_format` | `input_single_tile_size` | not set |
| `c_16` (`output_cb_index`) | `num_tiles_per_block * output_single_tile_size` | `core` | `output_cb_data_format` | `output_single_tile_size` | not set |

#### Semaphores

none

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._single_core_program_factory.cpp:130` (`TensorAccessorArgs(*src0_buffer)`) | input `a` | reader RTA 0 (`:187`) |
| `..._single_core_program_factory.cpp:137` (`TensorAccessorArgs(*dst_buffer)`) | output | writer RTA 0 (`:190`) |

Kernel-side: `reader_unary_interleaved_start_id.cpp:15,25`; `writer_unary_unpad_dims_split_rows.cpp:38,44`.

#### Work split

n/a — single core. `core` is a 1-node `CoreRange`.

---

### Variant: MultiCoreInterleaved (`untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` **(cross-op)** | `all_cores` | `TensorAccessorArgs(*src0_buffer)` (`:93`) | — | per node `{src0_buffer, num_tiles_per_core, tile_start_id}` (`:231`) | — | — | `ReaderConfigDescriptor{}` |
| writer | `untilize_with_unpadding/.../writer_unary_stick_layout_split_rows_multicore.cpp` **(op-owned)** | `all_cores` | `{FLOAT32_DTYPE, unpadded_row_size_bytes, writer_page_size}` + `TensorAccessorArgs(*dst_buffer)` (`:114-119`) | — | per node: 4 fixed + a **variable-length** 5-tuple block (`:195-232`) | — | — | `WriterConfigDescriptor{}` |
| compute (full) | `data_movement/untilize/.../compute/untilize.cpp` **(cross-op)** | `core_range` | `{nblocks_per_core, num_tiles_per_row, c_0, c_16}` (`:154`) | — | — | — | `DST_ACCUM_MODE=1` (as above) | `ComputeConfigDescriptor{...}` |
| compute (cliff) | same source | `core_range_cliff` | `{nblocks_per_core_cliff, num_tiles_per_row, c_0, c_16}` (`:169`) | — | — | — | same | same |

Writer RTA layout per node: `dst_buffer`, `padded_row_size_bytes`, `row_start_id`,
`assignment.size()` (`n_block_reps`), then **N × `{n_data, n_mixed, n_pads, times, repeat_count}`**,
where N (the number of *distinct-run* groups) varies per node with the block assignment.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `num_tiles_per_row * input_single_tile_size` | `all_cores` | `input_cb_data_format` | `input_single_tile_size` | not set |
| `c_16` | `num_tiles_per_row * output_single_tile_size` | `all_cores` | `output_cb_data_format` | `output_single_tile_size` | not set |

#### Semaphores

none

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:93` | input `a` | reader RTA 0 (`:231`) |
| `:119` | output | writer RTA 0 (`:196`) |

Kernel-side: `reader_unary_interleaved_start_id.cpp:15,25`;
`writer_unary_stick_layout_split_rows_multicore.cpp:30,34` (**3-arg accessor**, see *Dropped Plumbing*).

#### Work split

- Driver: `ttnn::split_blocks_for_tilize(available_grid, num_blocks)` (`:48-49`)
- `ncores`, `all_cores` = union
- `core_range`: the first `ncores_no_cliff` nodes of `available_grid`, `nblocks_per_core` each
- `core_range_cliff`: at most one node, `nblocks_per_core_cliff` blocks (present iff
  `nblocks_per_core_cliff > 0`)
- Both sub-ranges are **disjoint by construction** (`work_split_tilize.hpp:334-364` walks one
  enumerated core list with a monotonically advancing index).

---

### Variant: MultiCoreBlockInterleaved (`untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` **(cross-op)** | `all_cores` | `{num_tiles_2d, third_dim, total_tiles_per_row}` + `TensorAccessorArgs(*src0_buffer)` (`:184-185`) | — | per node `{src0_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg}` (`:294`) | — | — | `ReaderConfigDescriptor{}` |
| writer | `untilize_with_unpadding/.../writer_unary_stick_layout_wh_multicore.cpp` **(op-owned but externally co-borrowed → FORK)** | `all_cores` | `{total_num_rows, third_dim, TILE_HEIGHT, unpadded_row_size_bytes}` + `TensorAccessorArgs(*dst_buffer)` (`:196-197`) | — | per node, 8 values (`:298-307`), slot 0 = `dst_buffer` | — | — | `WriterConfigDescriptor{}` |
| compute ×≤4 | `data_movement/untilize/.../compute/untilize_wh.cpp` **(cross-op)** | one per non-empty sub-region (`:239-250`) | `{block_size_col, block_size_row, third_dim}` (per region) | — | — | — | `DST_ACCUM_MODE=1` (as above) | `ComputeConfigDescriptor{...}` |

Per-region compute CTAs (`:239-250`):

| region | `core_ranges` | `{block_size_col, block_size_row, third_dim}` |
|---|---|---|
| full | `core_range` | `{single_sub_block_size_wh, single_sub_block_size, third_dim}` |
| cliff_col_row | `cliff_col_row_core_range` | `{single_block_size_cliff_col, single_block_size_cliff_row, third_dim}` |
| cliff_row | `cliff_row_core_range` | `{single_block_size, single_block_size_cliff_row, third_dim}` |
| cliff_col | `cliff_col_core_range` | `{single_sub_block_size_cliff_col_wh, single_sub_block_size, third_dim}` |

Writer RTA slots: `dst_buffer`, `width_size`, `start_row_id`, `start_column_id`,
`single_block_size_row_arg`, `single_block_size_col_arg`, `sub_block_width_size`,
`single_sub_block_size_row_arg`.

#### CBs — **this factory emits up to 4 CB *pairs*, with region-dependent sizes**

`push_cb_pair(...)` (`:26-52`) is called once per non-empty sub-region (`:127-166`); each call
pushes a `c_0` and a `c_16` `CBDescriptor` **on that region's core range only**:

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` / `c_16` | `single_sub_block_size × tile_size` | `core_range` | in/out format | in/out tile size | not set |
| `c_0` / `c_16` | `single_block_size_cliff_row × tile_size` | `cliff_col_row_core_range` | " | " | not set |
| `c_0` / `c_16` | `single_block_size_cliff_row × tile_size` | `cliff_row_core_range` | " | " | not set |
| `c_0` / `c_16` | `single_sub_block_size × tile_size` | `cliff_col_core_range` | " | " | not set |

**This is the structurally interesting part of the port** — see *Planned Spec Shape* below.

#### Semaphores

none

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:185` | input `a` | reader RTA 0 (`:295`) |
| `:197` | output | writer RTA 0 (`:300`) |

Kernel-side: `reader_unary_interleaved_wh_multicore.cpp:21,27`;
`writer_unary_stick_layout_wh_multicore.cpp:21,25`.

#### Work split

- Driver: `ttnn::split_blocks_for_tilize_wh(available_grid, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit)` (`:101-102`)
- Produces `all_cores` plus four **pairwise-disjoint** sub-ranges (`core_range`,
  `cliff_row_core_range`, `cliff_col_core_range`, `cliff_col_row_core_range`) — the
  `addCore` lambda (`work_split_tilize.hpp:204-210`) advances one `core_index` over one enumerated
  core list and inserts each node into exactly one target set; `all_cores` is their union.
- A sub-range can be **empty even when its `has_cliff_*` flag is set** (e.g. `cliff_row` is empty
  when `full_cores_per_col == 0`). Legacy tolerates an empty `core_ranges` on a CB/kernel
  descriptor; the port must instead **skip the region entirely** (see *Deferred / Flagged*).

---

### Variant: MultiCoreNDSharded (`untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` **(cross-op)** | `compute_core_range` | `{src0_cb_index, num_tiles_per_input_block, num_shards, num_compute_cores}` + `TensorAccessorArgs(*src0_buffer)` (`:121-126`) | — | per node `{src0_buffer, start_shard_id}` (`:272`) | — | — | `ReaderConfigDescriptor{}` |
| writer | `untilize_with_unpadding/.../writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` **(op-owned)** | `compute_core_range` | 17 values (`:155-174`) + `TensorAccessorArgs(*dst_buffer)` (`:185`) + `TensorAccessorArgs(*src0_buffer)` (`:187`) | — | per node `{dst_buffer, src0_buffer, start_shard_id}` (`:275`) | `output.padded_shape()` dims then `input.padded_shape()` dims (`:175-183`, `:199`) | — | `WriterConfigDescriptor{}` |
| compute | `data_movement/untilize/.../compute/untilize_variable_num_blocks.cpp` **(cross-op)** | `compute_core_range` | `{num_tiles_per_input_block, src0_cb_index, output_cb_index}` (`:220-221`) | — | per node `{num_input_blocks_to_process}` (`:280`) | — | `DST_ACCUM_MODE=1` iff dtype ∈ {INT32, UINT32, FLOAT32} | `ComputeConfigDescriptor{...}` |

Writer CTA slots 0-16: `output_cb_index`, `output_stick_size`, `tile_height`,
`num_tiles_per_input_block`, `output_num_blocks_across_width`, `output_element_size`,
`num_cols_per_input_block`, `num_cols_per_output_block`, `input_single_tile_size`, `num_shards`,
`num_cores`, `num_tiles_per_input_row`, `num_tiles_per_output_row`, `tile_width`,
`output_tensor_width`, `output_tensor_height`, `tensor_rank`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `input_cb_num_tiles * input_single_tile_size` | `compute_core_range` | `input_cb_data_format` | `input_single_tile_size` | not set |
| `c_16` | `output_cb_num_tiles * output_single_tile_size` | `compute_core_range` | `output_cb_data_format` | `output_single_tile_size` | not set |

(`*_cb_num_tiles` = `num_tiles_per_input_block`, doubled when a core processes 2+ blocks.)

#### Semaphores

none

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:126` (reader) | input | reader RTA 0 (`:272`) |
| `:185` (writer, dst) | output | writer RTA 0 (`:275`) |
| `:187` (writer, src) | input | writer RTA 1 (`:275`) |

Kernel-side: `reader_unary_nd_sharded_blocks.cpp:26,27`;
`writer_..._nd_sharded.cpp:40-43` (two accessors, second at a **chained constexpr offset**).

#### Work split

- Driver: not `split_work_to_cores` — the split comes from the input buffer's
  `BufferDistributionSpec` (`:56-63`): `compute_core_range = CoreRangeSet(distribution_spec.cores_with_data())`,
  one shard-strided `start_shard_id` per node (`:241`, `:276`).
- All three kernels share `compute_core_range`; there is **no** per-group CTA specialization.

---

### Cross-op kernels

Kernel sources bound by these factories that live **outside** this op's directory
(each a [Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel) case), with their non-Quasar co-borrowers:

| kernel source | owner | non-Quasar co-borrowers (excluding this op) | decision |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `eltwise/unary` | typecast, `data_movement/copy`, `data_movement/pad`, `untilize` ×2, `examples/*` ×3, `nlp_create_qkv_heads_falcon7b`, `reduction/prod`, 2 test suites (≈15) | **FORK** |
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | `eltwise/unary` | `untilize` block factory | **FORK** |
| `data_movement/untilize/device/kernels/compute/untilize.cpp` | `data_movement/untilize` | fold, `untilize` ×4, upsample, halo, padded_slice, deepseek_prefill/combine, 1 llk test (≈10) | **FORK** |
| `data_movement/untilize/device/kernels/compute/untilize_wh.cpp` | `data_movement/untilize` | `untilize` block factory | **FORK** |
| `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | `data_movement/untilize` | `untilize` ×3 | **FORK** |
| `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | `data_movement/sharded` | `untilize` ND factory | **FORK** |

**No co-borrower can co-migrate in this PR** (all sit in ops outside the scoped subset), so every
one takes the fork path: a `_metal2`-suffixed copy alongside the original, in the **owning op's**
kernel directory (the sanctioned cross-op-kernel exception to the scope boundary). The legacy copy
is untouched.

**Forks actually landed (3 of 6).** The three forks that served only the capitulated
`MultiCoreBlockInterleaved` factory — `reader_unary_interleaved_wh_multicore_metal2.cpp`,
`untilize_wh_metal2.cpp`, and `writer_unary_stick_layout_wh_multicore_metal2.cpp` — were deleted
rather than shipped as dead files. Recreate them from their legacy originals when that factory is
re-attempted; the conversion itself was mechanical and is not what blocked it.

**In-directory kernel with an outside legacy consumer — the same fork decision, reached the other way:**

| kernel source | owner | non-Quasar co-borrowers | decision |
|---|---|---|---|
| `untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` | **this op** | `data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp:196` | **FORK** *(fork deleted with the capitulation — this kernel belongs to `MultiCoreBlockInterleaved`; the finding stands and is reported)* |

This kernel is *inside* the porter's writeable surface, so the legacy-inventory "cross-op kernel"
signal never fires on it — but it is bound by file path from `data_movement/untilize`, which stays
on `create_descriptor`. Modifying it in place would break that op. Forked to
`writer_unary_stick_layout_wh_multicore_metal2.cpp` in this op's own kernel directory. (Recorded as
a recognition-signal gap in `METAL2_PORT_REPORT.md` → Friction → Gaps.)

Kernels modified **in place** (this op is the sole non-Quasar consumer; the
`experimental/quasar/untilize_with_unpadding` clone carries its own copies under its own
`kernels/dataflow/` directory and does **not** borrow these by path):

- `writer_unary_unpad_dims_split_rows.cpp` (SingleCore)
- `writer_unary_stick_layout_split_rows_multicore.cpp` (MultiCoreInterleaved)
- `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` (MultiCoreNDSharded)

### Flags

- **Unreferenced file in the op directory**: `device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp`
  — a pre-`ProgramDescriptor` `KernelHandle` shared-variables struct, included by nothing, listed
  only in `ttnn/cpp/ttnn/operations/data_movement/CMakeLists.txt:324`. **Not audited, not touched.**
- **Dead CTAs** (values emitted by the host that no kernel reads). Carried forward as *named* CTAs
  so the port stays a pure syntax swap; each is reported for the ops team:
  - SingleCore writer CTA 1 `unpadded_stick_size` (`..._single_core_program_factory.cpp:136`) —
    `writer_unary_unpad_dims_split_rows.cpp` reads only CTA 0 and `TensorAccessorArgs<2>()`.
    *(Not in the audit's Misc-anomaly list; new finding.)*
  - ND writer CTA 1 `output_stick_size` and CTA 8 `input_single_tile_size` (audit Misc anomaly 2).
- **Dead RTAs** (read into a local the kernel never uses) in
  `writer_unary_unpad_dims_split_rows.cpp`: `num_unpadded_X` (slot 8), `padded_X_size` (slot 9),
  `num_blocks_w_input` (slot 10). Carried forward as named RTAs; reported. *(New finding.)*
- **Dead locals** `full_compute_idx` / `cliff_compute_idx`
  (`..._multi_core_interleaved_program_factory.cpp:147-148,160-161,175-176`) — computed, `(void)`-cast,
  never read (audit Misc anomaly 4). They exist only to reason about descriptor ordering, a concept
  that has no Metal 2.0 counterpart (kernels are named, not positional), so they **evaporate with
  the `desc.kernels.insert(begin, …)` ordering trick they annotated**. This is a deletion the port
  forces, not a bundled cleanup.
- **Unused debug include** `#include "api/debug/dprint.h"` in
  `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:12` (audit Misc anomaly 7) —
  **left alone**, out of the kernel-side whitelist.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` — realized by replacing
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` with
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const UntilizeWithUnpaddingParams&, const Tensor&, Tensor&)`
  on each of the four in-scope factory structs.
- **Custom `compute_program_hash`**: **none** — the op already uses the default reflection-based
  hash. Nothing to delete.
- **Implementation notes**:
  - The `program_factory_t` variant is intentionally **mixed-concept** post-port: four
    `MetalV2FactoryConcept` factories + `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` still
    on `ProgramDescriptorFactoryConcept`. `AllFactoriesValid` accepts this; the adapter dispatches
    per-factory.
  - `select_program_factory` and every `TT_FATAL` in `validate_on_program_cache_miss`
    (`device/untilize_with_unpadding_device_operation.cpp:40-95` and below) are **untouched**.
  - `untilize_with_unpadding_nanobind.cpp` binds only the public `ttnn::untilize_with_unpadding`
    free function — **no pybind surface is removed** by this port.
  - Each factory extracts `input.mesh_tensor()` / `output.mesh_tensor()` at entry for the
    `TensorParameter` specs and `TensorArgument`s; the rest of each factory body keeps using the
    `ttnn::Tensor` accessors it already used (minimal diff — a wholesale `MeshTensor` rewrite of
    shape/dtype/shard-spec reads is not port work).

---

## Planned Spec Shape

Shared across all four factories:

- **TensorParameters**: exactly 2 — `INPUT` (from `input.mesh_tensor().tensor_spec()`) and
  `OUTPUT` (from `output.mesh_tensor().tensor_spec()`). No relaxation
  (`advanced_options` left default → strict `TensorSpec` match). The op has no custom hash, so no
  relaxation can be active; the confirmed readiness sheet names none.
- **SemaphoreSpecs**: none — the op has no semaphores.
- **Op-owned tensors**: none.
- **Kernel-side accessor names** (uniform, so one forked source serves every `KernelSpec` that
  binds it): DFBs `"in"` / `"out"`; tensors `"src"` / `"dst"`.

### SingleCore

- **KernelSpecs**: 3 — `reader`, `writer`, `compute`.
- **DataflowBufferSpecs**: 2 — `IN` (`entry_size = input_single_tile_size`,
  `num_entries = num_tiles_per_block`, `data_format_metadata = input_cb_data_format`) and
  `OUT` (output tile size / same count / output format). `tile_format_metadata` left `nullopt`
  (legacy `CBFormatDescriptor::tile` unset).
- **TensorParameters**: `INPUT` (reader), `OUTPUT` (writer).
- **WorkUnitSpecs**: 1 — `{reader, writer, compute}` on the single node.
- Endpoints: `IN` = reader PRODUCER + compute CONSUMER; `OUT` = compute PRODUCER + writer CONSUMER.

### MultiCoreInterleaved

- **KernelSpecs**: 3 or 4 — `reader`, `writer`, `compute_full` (iff `core_range` non-empty),
  `compute_cliff` (iff `has_cliff`).
- **DataflowBufferSpecs**: 2 — `IN` / `OUT`, `num_entries = num_tiles_per_row`. Derived placement
  = `all_cores` (union of the bound kernels' WUs), matching the legacy `core_ranges = all_cores`.
- **TensorParameters**: `INPUT` (reader), `OUTPUT` (writer).
- **WorkUnitSpecs**: up to 2 — `{reader, writer, compute_full}` on `core_range`;
  `{reader, writer, compute_cliff}` on `core_range_cliff`.
- Endpoints per node: reader PRODUCER of `IN`, exactly one compute CONSUMER of `IN`; that same
  compute PRODUCER of `OUT`, writer CONSUMER of `OUT`. The two computes are node-disjoint, so the
  per-node census is 1P+1C on both DFBs (this is the *disjoint-node* work split, **not** the
  same-grid two-toucher case and **not** the multi-binding flag).

### MultiCoreBlockInterleaved — *design record only; this factory CAPITULATED*

> **Not shipped.** Everything below is what was built and measured before the port stopped on this
> factory. Keep it as the starting point for the re-attempt; see `METAL2_PORT_REPORT.md` →
> Handoff points for the two defects that block it.

The one factory whose spec shape is **not** 1:1 with the legacy kernel set. The legacy CBs are
**per-region with region-dependent sizes**, but a Metal 2.0 DFB's placement is *derived* from the
union of its bound kernels' WorkUnitSpecs — so a single `all_cores` reader binding one DFB would
spread that DFB (and one size) across every node. Preserving the legacy per-node L1 footprint
therefore requires the reader and writer to be **specialized per region** alongside the compute
kernel that already was:

- Per non-empty region `R ∈ {full, cliff_col_row, cliff_row, cliff_col}`:
  - **DataflowBufferSpecs** `IN_R`, `OUT_R` — `num_entries` = that region's legacy CB tile count.
  - **KernelSpecs** `reader_R`, `writer_R`, `compute_R` — reader/writer CTAs are **identical**
    across regions (they differ only in which DFB pair they bind); compute CTAs are the legacy
    per-region triple.
  - **WorkUnitSpec** `wu_R` = `{reader_R, writer_R, compute_R}` on that region's node set.
- Totals: up to 12 `KernelSpec`s, 8 `DataflowBufferSpec`s, 4 `WorkUnitSpec`s, 2 `TensorParameter`s.
- Per node this reproduces the legacy allocation exactly: one `IN`/`OUT` pair of the legacy size,
  one reader, one writer, one compute.
- **Rejected alternative**: one DFB pair sized to the region maximum. It would change the per-node
  L1 footprint (the whole point of `cb_block_size_limit` at `:83`) and could push an
  L1-tight configuration over budget — a behavior change, not a syntax swap.
- **Outcome: this shape does not work on the current framework.** Building it produces >2
  `DataflowBufferSpec`s partitioned across disjoint node sets, which trips the DFB config-region
  sizing defect (report → Handoff point 1): silent wrong numerics plus L1 corruption. Collapsing
  to the rejected single-pair alternative *does* fix every 2-region shape, confirming the
  mechanism — but a 4-region shape (`[1, 1, 1280, 1280]`) still returns wrong data even then, so a
  second defect is in play (report → Handoff point 2). The legacy `create_descriptor` factory
  passes all of these shapes.

### MultiCoreNDSharded

- **KernelSpecs**: 2 or 3 — `reader`, `writer`, `compute` (iff `compute_core_range` is non-empty,
  mirroring the legacy `has_compute` guard).
- **DataflowBufferSpecs**: 2 — `IN` (`num_entries = input_cb_num_tiles`), `OUT`
  (`num_entries = output_cb_num_tiles`).
- **TensorParameters**: `INPUT` — bound by **two** kernels (reader and writer; the writer walks its
  own shard's pages via `accessor_src.shard_pages(shard_id)`), i.e. one `TensorParameter` with two
  `TensorBinding`s — and `OUTPUT` (writer).
- **WorkUnitSpecs**: 1 — all kernels on `compute_core_range`.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| MultiCoreInterleaved: 2 × `untilize.cpp` (`:150-163`, `:164-178`) over `core_range` / `core_range_cliff` | `compute_full`, `compute_cliff` — per-group CTA `per_core_block_cnt` = `nblocks_per_core` / `nblocks_per_core_cliff` | `wu_full`, `wu_cliff` | `IN` (CONSUMER, one each), `OUT` (PRODUCER, one each) — **disjoint node sets**, so each is a legal single-role binding; **no** `allow_instance_multi_binding` |
| MultiCoreBlockInterleaved: ≤4 × `untilize_wh.cpp` (`:239-250`) over the four disjoint sub-regions | `compute_full`, `compute_cliff_col_row`, `compute_cliff_row`, `compute_cliff_col` — per-region CTA triple | `wu_full`, `wu_cliff_col_row`, `wu_cliff_row`, `wu_cliff_col` | each binds **its own** region DFB pair (`IN_R` CONSUMER, `OUT_R` PRODUCER) — no DFB is shared across regions |
| MultiCoreBlockInterleaved: 1 × reader / 1 × writer over `all_cores` | **split** into `reader_R` / `writer_R` per region (identical CTAs) | one per region | `IN_R` PRODUCER / `OUT_R` CONSUMER |
| SingleCore, MultiCoreNDSharded | none — one `KernelDescriptor` per source | — | — |

The reader/writer split in MultiCoreBlockInterleaved is *not* CTA specialization (their CTAs are
identical); it is **placement** specialization forced by derived DFB placement. It is the inverse
of the [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
anti-pattern's shape: same source, multiple `KernelSpec`s, disjoint node sets.

---

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._single_core_program_factory.cpp:187` slot 0 | `src0_buffer` (`Buffer*` RTA) | `TensorBinding{INPUT, "src"}` on `reader` |
| `..._single_core_program_factory.cpp:190` slot 0 | `dst_buffer` | `TensorBinding{OUTPUT, "dst"}` on `writer` |
| `..._multi_core_interleaved_program_factory.cpp:231` slot 0 | `src0_buffer` | `TensorBinding{INPUT, "src"}` on `reader` |
| `..._multi_core_interleaved_program_factory.cpp:196` slot 0 | `dst_buffer` | `TensorBinding{OUTPUT, "dst"}` on `writer` |
| `..._multi_core_block_interleaved_program_factory.cpp:295` slot 0 | `src0_buffer` | `TensorBinding{INPUT, "src"}` on each `reader_R` |
| `..._multi_core_block_interleaved_program_factory.cpp:300` slot 0 | `dst_buffer` | `TensorBinding{OUTPUT, "dst"}` on each `writer_R` |
| `..._multi_core_nd_sharded_program_factory.cpp:272` slot 0 | `src0_buffer` | `TensorBinding{INPUT, "src"}` on `reader` |
| `..._multi_core_nd_sharded_program_factory.cpp:275` slot 0 | `dst_buffer` | `TensorBinding{OUTPUT, "dst"}` on `writer` |
| `..._multi_core_nd_sharded_program_factory.cpp:275` slot 1 | `src0_buffer` (second consumer) | `TensorBinding{INPUT, "src"}` on `writer` — same `TensorParameter`, second binding |

All are **Case 1** (every one is consumed through a `TensorAccessor`); no Case 2, no
`get_bank_base_address` bridge, no borrowed-memory DFB anywhere in the subset.

Kernel-side counterparts removed: `get_arg_val<uint32_t>(0)` address reads at
`reader_unary_interleaved_start_id.cpp:11`, `writer_unary_unpad_dims_split_rows.cpp:19`,
`reader_unary_interleaved_wh_multicore.cpp:12`, `writer_unary_stick_layout_wh_multicore.cpp:23`,
`writer_unary_stick_layout_split_rows_multicore.cpp:19`,
`reader_unary_nd_sharded_blocks.cpp:13`, `writer_..._nd_sharded.cpp:20-21`.

### `TensorAccessorArgs` plumbing → binding mechanism

| host site | kernel-side chain removed |
|---|---|
| `..._single_core:130`, `:137` | `TensorAccessorArgs<0>()` (`reader_unary_interleaved_start_id.cpp:15`), `TensorAccessorArgs<2>()` (`writer_unary_unpad_dims_split_rows.cpp:38`) |
| `..._multi_core_interleaved:93`, `:119` | `TensorAccessorArgs<0>()`, `TensorAccessorArgs<3>()` (`writer_unary_stick_layout_split_rows_multicore.cpp:30`) |
| `..._multi_core_block_interleaved:185`, `:197` | `TensorAccessorArgs<3>()` (`reader_unary_interleaved_wh_multicore.cpp:21`), `TensorAccessorArgs<4>()` (`writer_unary_stick_layout_wh_multicore.cpp:21`) |
| `..._multi_core_nd_sharded:126`, `:185`, `:187` | `TensorAccessorArgs<4>()` (`reader_unary_nd_sharded_blocks.cpp:26`), `TensorAccessorArgs<17>()` **and** the chained `TensorAccessorArgs<dst_args.next_compile_time_args_offset()>()` (`writer_..._nd_sharded.cpp:40,42`) |

### Magic CB indices in CTAs → `DFBBinding`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._single_core:161-162` (compute CTA 2, 3) | `src0_cb_index`, `output_cb_index` | `DFBBinding{IN, "in", CONSUMER}` / `DFBBinding{OUT, "out", PRODUCER}` |
| `..._multi_core_interleaved:154`, `:169` (compute CTA 2, 3) | `tt::CBIndex::c_0`, `tt::CBIndex::c_16` | same |
| `..._multi_core_nd_sharded:121-122` (reader CTA 0) | `src0_cb_index` | `DFBBinding{IN, "in", PRODUCER}` |
| `..._multi_core_nd_sharded:155-156` (writer CTA 0) | `output_cb_index` | `DFBBinding{OUT, "out", CONSUMER}` |
| `..._multi_core_nd_sharded:220-221` (compute CTA 1, 2) | `src0_cb_index`, `output_cb_index` | `DFBBinding{IN, "in", CONSUMER}` / `DFBBinding{OUT, "out", PRODUCER}` |

Kernel-side hardcoded CB-index constants also removed (**not** replaced with a named CTA — rule 2):
`reader_unary_interleaved_start_id.cpp:17` (`cb_id_in0 = 0`),
`writer_unary_unpad_dims_split_rows.cpp:16` (`dfb_id_out0 = 16`),
`writer_unary_stick_layout_split_rows_multicore.cpp:16` (`dfb_id_out0 = 16`),
`reader_unary_interleaved_wh_multicore.cpp:17` (`cb_id_in0 = 0`),
`writer_unary_stick_layout_wh_multicore.cpp:15` (`dfb_id_out0 = 16`),
`untilize_wh.cpp:13,16-17` (`tt::CBIndex::c_0` / `c_16` passed straight to the LLK + kernel-lib).

### Page-size 3rd-argument CTA → dropped (binding supplies the aligned page size)

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `writer_unary_stick_layout_split_rows_multicore.cpp:34` | `TensorAccessor(dst_args, dst_addr, writer_page_size)` | `TensorAccessor(tensor::dst)` — 2-arg form |
| `writer_unary_stick_layout_split_rows_multicore.cpp:29` | `constexpr uint32_t writer_page_size = get_compile_time_arg_val(2);` | deleted |
| `..._multi_core_interleaved_program_factory.cpp:118` | CTA slot 2 `writer_page_size` | deleted |
| `..._multi_core_interleaved_program_factory.cpp:105-113` | `writer_page_size` local + its `if (out_mem_config.is_sharded() && …)` computation, and the comment that explains it | deleted (dead once the CTA goes) |

Audit classification: **Class 2 (redundant / inert)** — the value resolves to the output buffer's
logical page in every reachable config, and the sharded sub-case is exact because
`compute_output_specs` rounds every shard width up to a tile multiple. **Not** Class 1, so
**no** `dynamic_tensor_shape` relaxation. (The second Class-2 site,
`writer_unary_unpad_cross_sharded.cpp:35`, belongs to the out-of-scope `MultiCoreSharded` factory
and is left alone.)

### Semaphore-ID RTAs

none — the op has no semaphores.

### Positional CTAs → named CTAs

| kernel | legacy positional CTA list | names assigned |
|---|---|---|
| `reader_unary_interleaved_start_id.cpp` | (accessor args only) | — (no scalar CTAs) |
| `writer_unary_unpad_dims_split_rows.cpp` | `{FLOAT32_DTYPE, unpadded_stick_size}` | `float32_dtype`, `unpadded_stick_size` (the second is **dead** — kept for a pure syntax swap) |
| `writer_unary_stick_layout_split_rows_multicore.cpp` | `{FLOAT32_DTYPE, unpadded_X_size, writer_page_size}` | `float32_dtype`, `unpadded_X_size`; `writer_page_size` **dropped** (see above) |
| `untilize.cpp` | `{per_core_block_cnt, per_core_block_tile_cnt, src_cb, out_cb}` | `per_core_block_cnt`, `per_core_block_tile_cnt`; CB slots → bindings |
| `reader_unary_interleaved_wh_multicore.cpp` | `{num_tiles_per_2d, third_dim, total_tiles_per_row}` | same three names |
| `writer_unary_stick_layout_wh_multicore.cpp` | `{total_num_rows, third_dim, tile_height, unpadded_X_size}` | same four names |
| `untilize_wh.cpp` | `{block_size_col, block_size_row, third_dim}` | same three names |
| `reader_unary_nd_sharded_blocks.cpp` | `{cb_id_in0, num_tiles_per_input_block, num_shards, num_cores}` | `num_tiles_per_input_block`, `num_shards`, `num_cores`; CB slot → binding |
| `writer_..._nd_sharded.cpp` | 17 slots (list above) | slot 0 → binding; the other 16 keep their kernel-side names (slots 1 and 8 are **dead**, kept) |
| `untilize_variable_num_blocks.cpp` | `{per_core_block_tile_cnt, src_cb, out_cb}` | `per_core_block_tile_cnt`; CB slots → bindings |

### Positional RTAs / CRTAs → named (or vararg)

Every RTA becomes a named runtime arg with the kernel's own local name, **except** the two genuine
vararg blocks below. Full name lists are in the *Legacy Inventory* RTA rows.

---

## Applied Patterns

- **[Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) — avoided.**
  MultiCoreInterleaved's 2 compute descriptors and MultiCoreBlockInterleaved's ≤4 stay as
  multiple `KernelSpec`s of one source, each in its own `WorkUnitSpec` over disjoint nodes, with
  their per-group CTAs intact. Nothing is demoted to an RTA.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).**
  `compute_kernel_hw_startup(dfb::in, dfb::out)` and
  `compute_kernel_lib::untilize<block_w, dfb::in, dfb::out, …>` — the latter in
  **non-type-template-parameter** position, which the `constexpr operator uint32_t()` covers.
  No `.id` extraction, no temp `DataflowBuffer`.
- **[Modifying a shared dataflow kernel → fork](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel).**
  Seven forks with the `_metal2` suffix (six cross-op + one in-directory-but-externally-borrowed);
  see *Cross-op kernels*. No consumer can co-migrate in this PR.
- **[Avoid varargs unless absolutely necessary](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  — two retained, both genuine indexed-collection blocks:**
  1. `writer_unary_stick_layout_split_rows_multicore.cpp:73-86` (**RTA vararg**, MultiCoreInterleaved).
     `n_block_reps` bounds a loop pulling a 5-tuple per group through `rt_arg_idx`, advanced
     *inside* the loop at `:82`; the group count varies per node with the block assignment
     (`..._multi_core_interleaved_program_factory.cpp:195-226`). The **four leading args**
     (`padded_X_size`, `start_stick_id`, `n_block_reps`, plus the dropped `dst_addr`) are distinct
     fixed fields and are **named**. Because the vararg count differs per node, the schema uses
     `KernelAdvancedOptions::num_runtime_varargs_per_node` (the `[[deprecated]]` per-node override)
     rather than the scalar `num_runtime_varargs`.
  2. `writer_..._nd_sharded.cpp:100-105` (**CRTA vararg**, MultiCoreNDSharded). Two
     `get_common_arg_val` loops bounded by the `tensor_rank` CTA read the output shape dims then
     the input shape dims. A CTA-bounded count still varies across instantiations →
     `num_common_runtime_varargs = 2 × rank`, values broadcast to every node.

  **Non-signal, deliberately named:** `writer_unary_stick_layout_wh_multicore.cpp:65-70` re-reads
  args 2-7 inside the `third_dim` loop at **constant** indices — a fixed set of distinct fields
  re-read (so the loop body gets fresh copies after the previous iteration mutated its locals),
  not a loop-indexed block. Each gets a name, and the reads stay inside the loop.
- **[Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).**
  `ttnn_op_data_movement` is a unity-build target with dozens of factory `.cpp`s, so the
  `KernelSpecName` / `DFBSpecName` / `TensorParamName` constants are declared **function-local**
  inside each `create_program_artifacts` rather than at anonymous-namespace scope. This
  sidesteps cross-`.cpp` symbol merging entirely (no factory-name prefixing needed).
- **Self-loop / two-toucher / multi-binding / aliased / borrowed / conditional DFB**: **none apply.**
  Re-derived from the kernel-touch census rather than transcribed: every DFB in the subset is
  touched by exactly two distinct kernels on any given node, one FIFO producer and one FIFO
  consumer (`reserve_back`/`push_back` on one side, `wait_front`/`pop_front` on the other) — a
  plain 1P+1C. This matches the brief. There are no semaphores (so no hidden co-filler), no
  `.buffer`-backed CBs in the subset, no multi-element `format_descriptors`, and no conditionally
  bound resource.

---

## Deferred / Flagged

- **BLOCKER (found during construction, not planning) — the framework's per-node DFB config region
  is sized by DFB *count* but addressed by DFB *id*.** Any `ProgramSpec` that partitions its
  `DataflowBufferSpec`s across disjoint node sets — which is exactly the shape the
  MultiCoreBlockInterleaved per-region plan produces — overruns that region and silently corrupts
  L1. Nothing in the audit's Appendix A, the recipe, or the patterns catalog anticipates this; the
  planning step could not have caught it because it is a framework defect, not a spec-shape
  question. Full mechanism, repro and the confirming experiment are in `METAL2_PORT_REPORT.md` →
  Handoff points 1. The three shipped factories are **immune by construction** (each declares
  exactly two DFBs, both resident on every node the program touches).
- **New finding — empty sub-regions in MultiCoreBlockInterleaved.** Legacy pushes a CB pair and a
  compute `KernelDescriptor` for a region whose `has_cliff_*` flag is set even when that region's
  `CoreRangeSet` is **empty** (`cliff_row` is empty when `full_cores_per_col == 0`; `cliff_col` is
  empty when `full_cores_per_row == 0`). On the legacy path these are harmless no-ops. Metal 2.0
  would reject them (a `WorkUnitSpec` with no nodes, and a DFB whose derived placement is empty),
  so the port gates every region on `!range.empty()` instead of on the `has_cliff_*` flag. The two
  conditions coincide wherever the region is non-empty, so this is behavior-identical.
- **New finding — dead CTAs / dead RTAs.** Listed under *Flags* above. Carried forward verbatim as
  named args (a pure syntax swap); routed to the ops team via the port report rather than deleted
  here.
- **New finding (recognition-signal gap) — in-directory kernel with an outside legacy consumer.**
  `writer_unary_stick_layout_wh_multicore.cpp`. The legacy-inventory step's cross-op-kernel signal
  only fires on kernel sources *outside* the op directory, so this one is invisible to it even
  though it carries exactly the same co-migration hazard. Recorded in the port report under
  Friction → Gaps.
- **Not a port finding, carried from the audit and left alone**: the `unpad_tensor_w_16` +
  interleaved-output latent bug (audit Misc anomaly 1) sits in the out-of-scope `MultiCoreSharded`
  factory; the unreferenced `..._multi_core_shared_variables.hpp`; the author-flagged shard-spec
  uncertainty at `..._multi_core_sharded_program_factory.cpp:60-61`.
