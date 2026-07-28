# Port Plan — `pool/upsample`

Port plan for `ttnn/cpp/ttnn/operations/pool/upsample`, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Recipe docs:** `a21c8f3f324 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`
**Audit docs (inherited):** `a21c8f3f324 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

Single `DeviceOperation` (`UpsampleOperation`) with four `ProgramFactory` variants. Per `METAL2_PREPORT_AUDIT.md`, **three factories are ported in this change**:

- **`UpsampleMultiCoreInterleavedProgramFactory`** — integer-scale interleaved path (row-major and tiled)
- **`UpsampleMultiCoreShardedProgramFactory`** — integer-scale sharded path (`WorkloadDescriptor`, op-owned config tensor)
- **`UpsampleNearestFloatProgramFactory`** — float-scale (general) path

**`UpsampleBilinearProgramFactory` is NOT ported** — blocked on the Device 2.0 prerequisite (isolated holdover in `device/kernels/compute/bilinear.cpp`; see the audit). It stays on the legacy `ProgramDescriptor` API; `UpsampleOperation::select_program_factory` continues to dispatch to it for `mode == "bilinear"`, and the framework runs a half-ported `program_factory_t` variant correctly (per-factory dispatch).

## Legacy Inventory

### Legacy factory shape
- Concept: **`descriptor`** (`UpsampleMultiCoreInterleavedProgramFactory`, `UpsampleNearestFloatProgramFactory`) / **`WorkloadDescriptor`, secretly SPMD** (`UpsampleMultiCoreShardedProgramFactory`) — all three expose `create_descriptor` / `create_workload_descriptor` returning `ProgramDescriptor` / `WorkloadDescriptor`.
- Variants: three (of the op's four factories; Bilinear excluded — stays `descriptor`/legacy).
- Custom `compute_program_hash`: **none** on any of the three (confirmed: no override anywhere in `upsample_device_operation.{hpp,cpp}`; matches the readiness sheet's `Custom hash = no` on all four rows).

*(Target Metal 2.0 concept `MetalV2FactoryConcept` for all three, inherited from the audit — see TTNN ProgramFactory below.)*

### Variant: UpsampleNearestFloatProgramFactory (`upsample_nearest_float_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_upsample_nearest_float.cpp` | `all_cores` | `output_cb_index`(c_0), `aligned_input_page_size`, `input_height`, `input_width`, `output_height`, `output_width`, `num_pages_across_width`, `reciprocal_scale_h_fixed`, `reciprocal_scale_w_fixed` + `TensorAccessorArgs(input.buffer())` | `input.buffer()`(Buffer\*), `num_sticks`, `start_stick_id` | ReaderConfigDescriptor |
| writer | `kernels/dataflow/writer_upsample_nearest_float.cpp` | `all_cores` | `output_cb_index`(c_0), `aligned_output_page_size` + `TensorAccessorArgs(output.buffer())` | `output.buffer()`(Buffer\*), `num_sticks`, `start_stick_id` | WriterConfigDescriptor |

No compute kernel.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 (`output_cb_index`) | `output_cb_page_size * num_cb_pages * BUFFERING_FACTOR` (= `aligned_output_page_size * 2 * 2`) | all_cores | output dtype | `aligned_output_page_size` | none (default 32×32) |

Not borrowed — a plain L1-allocated CB, not backed by a tensor buffer.

#### Semaphores
none

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `upsample_nearest_float_program_factory.cpp:105` (`TensorAccessorArgs(*input.buffer())`) | input | reader RTA[0] (`input.buffer()`, Buffer\* form) |
| `upsample_nearest_float_program_factory.cpp:111` (`TensorAccessorArgs(*output_tensor.buffer())`) | output | writer RTA[0] (`output_tensor.buffer()`, Buffer\* form) |

#### Work split
- Driver: `split_work_to_cores(compute_grid_size, total_pages_in_output)`
- num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2 — but **not used for CTA specialization**: every core (whichever group) gets the *same* reader/writer kernel, just a different `num_sticks` RTA value. No per-group `KernelDescriptor` split (unlike the anti-pattern's premise) — one reader + one writer over `all_cores`, RTA `num_sticks` varies per core between the two group counts.

#### Shared kernels
none — both kernels live in the op's own directory and aren't referenced elsewhere.

#### Flags
none.

### Variant: UpsampleMultiCoreInterleavedProgramFactory (`upsample_program_factory_multicore_interleaved.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_upsample_unary_stick_layout_interleaved_start_id.cpp` | `all_cores` | `src0_cb_index`(c_0), `aligned_input_unit_size` + `TensorAccessorArgs(*src_buffer)` | `src_buffer`(Buffer\*), `reader_units`, `reader_start` | ReaderConfigDescriptor |
| writer | `kernels/dataflow/writer_upsample_interleaved.cpp` | `all_cores` | `output_cb_index`, `writer_unit_size`, `scale_factor_h`, `scale_factor_w`, `output_shape[1]`, `output_shape[2]`, `block_height`, `num_units_per_output_stick` + `TensorAccessorArgs(*dst_buffer)` | `dst_buffer`(Buffer\*), `blocks_per_core`, `blocks_processed` | WriterConfigDescriptor |
| compute (tiled only, ≤2 instances) | **donor** `data_movement/untilize/device/kernels/compute/untilize.cpp` | `core_group_1` / `core_group_2` | `work_per_core_group_{1,2}`(per_core_block_cnt), `num_input_tiles_in_row`(per_block_ntiles), `src0_cb_index`, `output_cb_index` | none | ComputeConfigDescriptor{} (all default) |

`output_cb_index == src0_cb_index` on the row-major path (one CB reused for both reader-output and writer-input — no compute kernel at all); a **second**, distinct `output_cb_index` exists only on the tiled path, where the compute kernel is also present.

#### CBs
| index | total_size | core_ranges | data_format | page_size | condition |
|---|---|---|---|---|---|
| c_0 (`src0_cb_index`) | `aligned_input_unit_size * num_pages_in_input_cb` | all_cores | input dtype | `aligned_input_unit_size` | always |
| c_1 (`output_cb_index`) | `output_unit_size * num_pages_in_output_cb` | all_cores | output dtype | `output_unit_size` | **tiled only** (`is_tiled_layout`); row-major reuses `src0_cb_index` |

Neither is borrowed-memory (no `.buffer` set on either `CBDescriptor`).

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `upsample_program_factory_multicore_interleaved.cpp:139` | input | reader RTA[0] (`src_buffer`, Buffer\* form) |
| `upsample_program_factory_multicore_interleaved.cpp:178` | output | writer RTA[0] (`dst_buffer`, Buffer\* form) |

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, work_units_to_split)`
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `work_per_core_group_1`, `work_per_core_group_2`
- Reader/writer: one `KernelDescriptor` each over `all_cores` (RTA-only per-core specialization, not CTA — no work-split multiplicity to preserve for reader/writer).
- Compute (tiled only): **up to two** `KernelDescriptor`s of the *same* donor source, one per core group (`core_group_1` / `core_group_2`), each with its own `work_per_core_group_N` baked in as a CTA. `core_group_2` may be absent (`std::optional<KernelDescriptor>`, pushed only `if (core_group_2.num_cores() > 0)`).

#### Shared kernels
- `data_movement/untilize/device/kernels/compute/untilize.cpp` — **cross-family donor** (borrowed), file-path-instantiated on the tiled path only. `grep -rl "untilize.cpp" ttnn/cpp/ttnn/operations/` hits: `data_movement/untilize`'s own 4 factories, `data_movement/untilize_with_unpadding`'s 3 factories, and this op's interleaved factory. **A `_metal2` fork already exists beside the original**: `data_movement/untilize/device/kernels/compute/untilize_metal2.cpp` (created by the `data_movement/fold` Metal 2.0 port; comment confirms "Its binding names (`dfb::src` / `dfb::out`) and named args are the shared interface — do not rename"). **Decision: REUSE (rung 1)** — point this factory's compute `KernelSpec::source` at the existing fork; adopt its binding vocabulary verbatim: `dfb::src` (consumer), `dfb::out` (producer), named args `args::per_core_block_cnt`, `args::per_core_block_tile_cnt`. No new fork, no edits to either the legacy `untilize.cpp` or the `_metal2` fork.

#### Flags
none — both own kernels + the untilize donor are Device 2.0 (audit GREEN for this factory). No unreferenced kernel files.

### Variant: UpsampleMultiCoreShardedProgramFactory (`upsample_program_factory_multicore_sharded.cpp`)

#### Kernels
Both instances share **one** source, differing only by config descriptor + the `is_reader` CTA:

| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| writer-inst | `kernels/dataflow/writer_upsample_multi_core_sharded.cpp` | `cores_with_work` | `in_cb_id`, `out_cb_id`, `is_reader`=0, `config_cb_id`, `input_stick_nbytes`, `input_nsticks_per_core`, `scale_factor_h`, `scale_factor_w`, `config_tensor_width/4` | none | WriterConfigDescriptor |
| reader-inst | same source | `cores_with_work` | same 8, `is_reader`=1 | none | ReaderConfigDescriptor |

No RTAs on either instance (the config lookup table travels via the `config_cb`, read at compile-time-known offsets computed from `elem_per_core_reader` — a CTA, not a per-core RTA).

#### CBs
| index | total_size | core_ranges | data_format | page_size | buffer (borrowed) |
|---|---|---|---|---|---|
| `in_cb_id` (c_0) | `in_cb_pagesize * in_cb_npages` | `all_cores` | input dtype | `aligned_input_stick_nbytes` | `input.buffer()` |
| `out_cb_id` (c_1) | `out_cb_pagesize * out_cb_npages` | `all_cores` | output dtype | (aligned) `output_stick_nbytes` | `output.buffer()` |
| `config_cb_id` (c_2) | `config_buffer_page_size` | `cores_with_work` | `RawUInt16` | `config_buffer_page_size` | `config_buffer` (op-owned) |

All three CBs are borrowed-memory. `config_cb_id`'s core range (`cores_with_work`) is a subset of `in_cb_id`/`out_cb_id`'s (`all_cores`) when the last shard is uneven — but both `in_cb`/`out_cb` KernelSpecs are only ever bound on `cores_with_work` too (the reader/writer instances themselves are scoped to `cores_with_work`, not `all_cores`; the CBs' own `core_ranges` field in legacy is directly `all_cores` for in/out — a pre-existing legacy discrepancy where the CB is nominally declared over a larger range than any kernel touches. Metal 2.0 derives DFB placement from kernel bindings, so this discrepancy evaporates automatically: the ported DFBs are placed wherever `writer-inst`/`reader-inst` run, i.e. `cores_with_work` for all three).

#### Semaphores
none

#### Tensor accessors
none — input, output, and the op-owned config tensor are all delivered via borrowed-memory CBs; no `TensorAccessor` anywhere in either kernel instance.

#### Op-owned tensors
`config_tensor` — the per-core halo/replication lookup table built host-side (`create_config_tensor`), moved to device (`config_tensor.to_device(...)`), and parked in the legacy `WorkloadDescriptor::buffers` via a `shared_ptr<Tensor>` owner (`upsample_program_factory_multicore_sharded.cpp:450-461`). One op-owned tensor.

#### Work split
- `all_cores = shard_spec.grid`; `cores_with_work = get_cores_with_work(...)` (drops cores with no work when the shard grid is larger than needed).
- No `split_work_to_cores` — dual-instance work-split (below) is the whole story.

#### Shared kernels
none — `writer_upsample_multi_core_sharded.cpp` lives in the op's own directory and is bound only by this factory (in two instances).

#### Flags
none.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept` for all three factories.
- **Custom `compute_program_hash`:** none.
- **Implementation notes:** `UpsampleMultiCoreShardedProgramFactory` is the WorkloadDescriptor→`MetalV2FactoryConcept` case with a single op-owned tensor (the config lookup table); per the audit's TTNN factory analysis, its legacy `WorkloadDescriptor` was secretly single-program (one structurally-identical program copied across mesh coords), so it collapses cleanly onto the single-program concept, carrying the config tensor in `op_owned_tensors`.

## Planned Spec Shape

### Variant: UpsampleNearestFloatProgramFactory
- **KernelSpecs (2):** `NEARFLOAT_READER`, `NEARFLOAT_WRITER`.
- **DataflowBufferSpecs (1):** `NEARFLOAT_OUT` (plain L1, not borrowed; `entry_size = aligned_output_page_size`, `num_entries = BUFFERING_FACTOR * BUFFERING_FACTOR = 4`, `data_format_metadata = output dtype`).
- **SemaphoreSpecs:** none.
- **TensorParameters (2):** `NEARFLOAT_INPUT` (bound by reader, `tensor::input`), `NEARFLOAT_OUTPUT` (bound by writer, `tensor::output`). Both Case 1.
- **WorkUnitSpecs (1):** `{NEARFLOAT_READER, NEARFLOAT_WRITER}` over `all_cores`.
- **Op-owned tensors:** none.

### Variant: UpsampleMultiCoreInterleavedProgramFactory
- **KernelSpecs (2–4):** `INTLV_READER`, `INTLV_WRITER` (always); `INTLV_COMPUTE_G1` and, when `core_group_2` is non-empty, `INTLV_COMPUTE_G2` (tiled path only; both bind the reused `untilize_metal2.cpp` fork with per-group CTAs — preserving the legacy per-group `KernelDescriptor` multiplicity per [Anti-pattern: Demoting per-group CTA to RTA]).
- **DataflowBufferSpecs (1–2):** `INTLV_SRC0` (plain L1; entry `aligned_input_unit_size`, num `num_pages_in_input_cb`, fmt input dtype) always; `INTLV_OUT` (plain L1; entry `output_unit_size`, num `num_pages_in_output_cb`, fmt output dtype) **tiled path only** — row-major reuses `INTLV_SRC0` as both reader-output and writer-input (no separate spec).
- **SemaphoreSpecs:** none.
- **TensorParameters (2):** `INTLV_INPUT` (reader `tensor::input`, Case 1), `INTLV_OUTPUT` (writer `tensor::output`, Case 1).
- **WorkUnitSpecs:** `WU_MAIN` `{INTLV_READER, INTLV_WRITER}` over `all_cores` always; row-major path stops there (no compute). Tiled path additionally: `WU_COMPUTE_G1` `{INTLV_COMPUTE_G1}` over `core_group_1`, and — only if `core_group_2.num_cores() > 0` — `WU_COMPUTE_G2` `{INTLV_COMPUTE_G2}` over `core_group_2`.
- **Op-owned tensors:** none.

### Variant: UpsampleMultiCoreShardedProgramFactory
- **KernelSpecs (2):** `SHARD_WRITER` (is_reader=0 CTA baked in), `SHARD_READER` (is_reader=1 CTA baked in) — same source (the existing kernel file stays `writer_upsample_multi_core_sharded.cpp`; both instances point at it).
- **DataflowBufferSpecs (3):** `SHARD_IN` (`borrowed_from = SHARD_INPUT`), `SHARD_OUT` (`borrowed_from = SHARD_OUTPUT`), `SHARD_CONFIG` (`borrowed_from = SHARD_CONFIG_TENSOR`, the op-owned tensor).
- **SemaphoreSpecs:** none.
- **TensorParameters (3):** `SHARD_INPUT`, `SHARD_OUTPUT` (io tensors; no `TensorBinding` on either kernel — satisfied entirely via `borrowed_from`, per the `borrowed-memory DFB` pattern and matching the `data_movement/fold` `MultiCore` precedent), `SHARD_CONFIG_TENSOR` (op-owned; likewise no `TensorBinding`, only `borrowed_from` on `SHARD_CONFIG`).
- **WorkUnitSpecs (1):** `{SHARD_WRITER, SHARD_READER}` over `cores_with_work`.
- **Op-owned tensors (1):** the config lookup tensor — construction unchanged (`create_config_tensor` → `to_device`), tail changed to `release_mesh_tensor()` into `op_owned_tensors`, bound as `SHARD_CONFIG_TENSOR`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| Interleaved (tiled): `compute` over `core_group_1` + optional `compute` over `core_group_2`, both instantiating the untilize donor, **disjoint** node sets | `INTLV_COMPUTE_G1`, `INTLV_COMPUTE_G2` | `WU_COMPUTE_G1` / `WU_COMPUTE_G2` | `INTLV_SRC0`: each compute instance CONSUMER on its own disjoint node set — ordinary 1:1, no assignment question. `INTLV_OUT`: each compute instance PRODUCER, same disjointness. |
| Sharded: 2× `writer_upsample_multi_core_sharded.cpp` over the **same** `cores_with_work` grid (Writer-config `is_reader=0` + Reader-config `is_reader=1`) | `SHARD_WRITER`, `SHARD_READER` | one WU (both) | `SHARD_IN`: both instances raw-peek via `get_read_ptr()` → **1P+1C** (roles cosmetic). `SHARD_CONFIG`: both instances raw-peek via `get_read_ptr()` → **1P+1C**. `SHARD_OUT`: both instances raw-write disjoint offset ranges via `noc.async_read(..., out_dfb, ..., {.offset_bytes=...})` → **1P+1C**, output resident (nothing drains). |

The interleaved compute pair is the **disjoint-node** work-split (ordinary 1:1 per DFB, no endpoint-assignment question — ​matches [Anti-pattern: Demoting per-group CTA to RTA]'s "correct port" shape, not the two-toucher pattern). The sharded pair is the **dual-instance work-split** (two-toucher 1P+1C on every shared DFB, *not* multi-binding) — verified against the `data_movement/fold` `MultiCore` factory, the closest in-tree precedent for this exact shape.

Reader/writer kernels in all three factories have no work-split multiplicity of their own (one `KernelDescriptor` each over the full grid; per-core variation rides RTAs only) — this section otherwise reads "none" for them.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| nearest_float reader RTA[0] (`upsample_nearest_float_program_factory.cpp:139`) | `input.buffer()` (Buffer\*) | `TensorBinding(NEARFLOAT_INPUT, "input")` → `TensorAccessor(tensor::input)` |
| nearest_float writer RTA[0] (`:146`) | `output_tensor.buffer()` (Buffer\*) | `TensorBinding(NEARFLOAT_OUTPUT, "output")` → `TensorAccessor(tensor::output)` |
| nearest_float reader/writer CTA[0] | `output_cb_index` magic index (same CB on both — reader PRODUCER, writer CONSUMER) | `DFBBinding(NEARFLOAT_OUT, "out", PRODUCER)` on reader / `DFBBinding(NEARFLOAT_OUT, "out", CONSUMER)` on writer |
| nearest_float reader CTA + kernel `TensorAccessorArgs<9>()` | `TensorAccessorArgs(*input.buffer()).append_to(...)` | dropped entirely; `TensorAccessor(tensor::input)` |
| nearest_float writer CTA + kernel `TensorAccessorArgs<2>()` | `TensorAccessorArgs(*output_tensor.buffer()).append_to(...)` | dropped entirely; `TensorAccessor(tensor::output)` |
| interleaved reader RTA[0] (`upsample_program_factory_multicore_interleaved.cpp:250`) | `src_buffer` (Buffer\*) | `TensorBinding(INTLV_INPUT, "input")` |
| interleaved writer RTA[0] (`:257`) | `dst_buffer` (Buffer\*) | `TensorBinding(INTLV_OUTPUT, "output")` |
| interleaved reader/writer CTA + kernel `TensorAccessorArgs<N>()` | `TensorAccessorArgs(*src_buffer)/(*dst_buffer).append_to(...)` | dropped entirely |
| interleaved reader/writer/compute CTA[0]/[1] (`src0_cb_index`, `output_cb_index`) | magic CB indices | `DFBBinding(INTLV_SRC0/INTLV_OUT, ...)` |
| interleaved compute (untilize donor) CTA[2]/[3] | `src0_cb_id`/`out_cb_id` | already-named args on the fork (`args::per_core_block_cnt`, `args::per_core_block_tile_cnt`) + `dfb::src`/`dfb::out` bindings — inherited from the existing `_metal2` fork verbatim |
| sharded writer/reader-inst CTA[0]/[1]/[3] (`in_cb_id`, `out_cb_id`, `config_cb_id`) | magic CB indices | `DFBBinding(SHARD_IN/SHARD_OUT/SHARD_CONFIG, ...)` |
| all kernels | positional `get_compile_time_arg_val(N)` / `get_arg_val<uint32_t>(N)` | `get_arg(args::name)` |

No page-size 3rd-argument sites anywhere (audit confirmed). No semaphore-ID RTAs. No offset-folded base pointers.

## Applied Patterns

- **[Two-toucher DFB → assign 1P+1C](../shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split):** Sharded `SHARD_IN`/`SHARD_OUT`/`SHARD_CONFIG` — dual-instance work-split, both instances raw-touch each borrowed DFB; assign one PRODUCER + one CONSUMER per DFB. Not multi-binding.
- **[Borrowed-memory DFB](../shared/migration_guide.md#dataflowbufferspec):** all three sharded DFBs (`SHARD_IN`←`SHARD_INPUT`, `SHARD_OUT`←`SHARD_OUTPUT`, `SHARD_CONFIG`←`SHARD_CONFIG_TENSOR`).
- **[Caution: Porting a shared kernel](../shared/port_patterns.md#caution-porting-a-shared-kernel) — reuse rung:** interleaved factory's tiled-path compute kernel reuses the existing `untilize_metal2.cpp` fork beside the legacy donor; no new fork created, no edits to either copy.
- **[Anti-pattern: Demoting per-group CTA to RTA](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) (avoided):** interleaved factory's tiled-path compute kernels preserve the legacy per-core-group `KernelDescriptor` multiplicity as two `KernelSpec`s (`INTLV_COMPUTE_G1`/`G2`) over disjoint `WorkUnitSpec`s, each with its own per-group CTA baked in — not demoted to a shared `KernelSpec` with the block count moved to an RTA.
- **Pass-through op-owned tensor (sharded):** `config_tensor`'s host construction (`create_config_tensor` → `to_device`) is kept verbatim; only the tail changes (`release_mesh_tensor()` into `op_owned_tensors`, bound as `SHARD_CONFIG_TENSOR`), per [Construct — op-owned tensors].

## Deferred / Flagged

- None — planning surfaced no new structural issues beyond what the audit already recorded (the Device 2.0 bilinear finding, out of scope for this port).
