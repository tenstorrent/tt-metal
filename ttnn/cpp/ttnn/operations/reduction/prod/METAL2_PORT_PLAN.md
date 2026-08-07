# Port Plan — `reduction/prod`

Port plan for `ttnn/cpp/ttnn/operations/reduction/prod`, ported from the legacy
`ProgramDescriptor` API (`ProgramDescriptorFactoryConcept`) to Metal 2.0
(`MetalV2FactoryConcept`). Written during the inventory and planning steps;
committed alongside the port for review.

**Porting unit:** two bundled device operations sharing this directory, ported
together (they share a donor writer kernel):
- `ProdAllDeviceOperation` / `ProdAllProgramFactory` — full-tensor product, single core.
- `ProdNcDeviceOperation` / `ProdNcProgramFactory` — reduction over dim 0 or 1, multi-core.

Audit cleared **GREEN** (see `METAL2_PREPORT_AUDIT.md` / `METAL2_PORT_BRIEF.md`).

---

## Legacy Inventory

### Legacy factory shape
- Concept: **`ProgramDescriptorFactoryConcept`** (both factories expose
  `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor`).
- Variants: single factory each (`program_factory_t = std::variant<ProdAllProgramFactory>` /
  `std::variant<ProdNcProgramFactory>`). No multi-variant switch inside a factory.
- Custom `compute_program_hash`: **none** (both) — already default reflection-based hash.
  Nothing to delete.

*(Metal 2.0 target concept chosen during audit: `MetalV2FactoryConcept` for both.
 Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Variant: prod_all (`prod_all_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `eltwise/unary/.../dataflow/reader_unary_interleaved_start_id.cpp` (**donor, cross-op**) | `{0,0}` | *(none, only `TensorAccessorArgs<0>` appended)* | `{src_addr=input, num_pages=num_tiles, start_id=0}` | none | `ReaderConfigDescriptor{}` |
| writer | `eltwise/unary/.../dataflow/writer_unary_interleaved_start_id.cpp` (**donor, cross-op**) | `{0,0}` | `[0]=output_cb_index(c_3)` + `TensorAccessorArgs<1>` | `{dst_addr=output, num_pages=1, start_id=0}` | none | `WriterConfigDescriptor{}` |
| compute | `reduction/prod/.../compute/prod_all.cpp` (own) | `{0,0}` | `[0]=num_tiles`, `[1]=1` **(dead — kernel reads only CTA[0])** | none | none | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en=true, dst_full_sync_en=false, math_approx_mode=true}` |

- Compute config detail: `fp32_dest_acc_en = true` **always**. `math_fidelity = HiFi3`
  when (`fp32_dest_acc_en && arch==WORMHOLE_B0`) else `HiFi4`. `math_approx_mode=true`.

#### CBs
| buffer_index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` (input) | `2 * in_single_tile_size` | `{0,0}` | `datatype_to_dataformat_converter(input.dtype())` | `in_single_tile_size` | default |
| `c_3` (output) | `2 * out_single_tile_size` | `{0,0}` | `datatype_to_dataformat_converter(output.dtype())` | `out_single_tile_size` | default |

- Note: output uses `c_3`, not the `c_16+` output convention (cosmetic; audit
  "Misc anomalies"). **Preserved as-is** — index is invisible in Metal 2.0 (DFB name).

#### Semaphores
None.

#### Tensor accessors
| Tensor | site | RTA slot |
|---|---|---|
| input (Case 1) | donor reader `TensorAccessor(src_args, src_addr)` | reader RTA 0 |
| output (Case 1) | donor writer `TensorAccessor(dst_args, dst_addr)` | writer RTA 0 |

#### Work split
Single core `{0,0}`. `num_tiles = input.physical_volume() / tile_hw`. No `split_work_to_cores`.

### Variant: prod_nc (`prod_nc_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `reduction/prod/.../dataflow/reader_prod_nc.cpp` (own) | `all_cores` | `[0]=dim` | `{input_addr=input, num_input_tiles=num_reduce_input_tile, num_output_tiles=num_tiles_per_core, input_tile_offset, start_id=tile_offset, HtWt, CHtWt, dim` **(RTA 7 dead — reader reads dim from CTA)**`}` | `ReaderConfigDescriptor{}` |
| writer | `eltwise/unary/.../dataflow/writer_unary_interleaved_start_id.cpp` (**donor, cross-op; shared with prod_all**) | `all_cores` | `[0]=cb_id_out(c_3)` + `TensorAccessorArgs<1>` | `{dst_addr=output, num_pages=num_tiles_per_core, start_id=tile_offset, is_dram` **(RTA 3 dead — donor reads only 0–2)**`}` | `WriterConfigDescriptor{}` |
| compute_1 | `reduction/prod/.../compute/prod_nc.cpp` (own) | `core_group_1` | `[0]=num_cols_per_core_group_1` **(dead — kernel reads no CTA)** | `{num_input_tiles=num_reduce_input_tile, num_output_tiles=num_tiles_per_core}` | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en=false}` |
| compute_2 | `reduction/prod/.../compute/prod_nc.cpp` (own) | `core_group_2` (only if non-empty) | `[0]=num_cols_per_core_group_2` **(dead)** | same shape as compute_1 | same |

- Compute config detail: `fp32_dest_acc_en = (output.dtype() != BFLOAT16)`.
  `math_fidelity = HiFi3` when (`fp32_dest_acc_en && arch==WORMHOLE_B0`) else `HiFi4`.
  `math_approx_mode` **unset** → legacy `ComputeConfigDescriptor` default `false`.
  `bfp8_pack_precise` unset → default `false`.

#### CBs
| buffer_index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| `c_0` (input) | `2 * single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(output.dtype())` | `single_tile_size` |
| `c_3` (output) | `2 * single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(output.dtype())` | `single_tile_size` |

- Both CBs' data_format derives from **output** dtype (inplace op; input == output dtype in practice).

#### Semaphores
None.

#### Tensor accessors
| Tensor | site | RTA slot |
|---|---|---|
| input (Case 1) | `reader_prod_nc.cpp` `TensorAccessor(dram_input_addrg_args, input_addr)` | reader RTA 0 |
| output (Case 1) | donor writer `TensorAccessor(dst_args, dst_addr)` | writer RTA 0 |

#### Work split
`split_work_to_cores(grid, num_output_tiles)` →
`(num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2)`.
Per-core `num_tiles_per_core` and running `tile_offset` drive the per-node RTAs.
Compute is split into `compute_desc_1`/`compute_desc_2` over **disjoint** core groups.

### Cross-op kernels (top-level flag)
Two **broadly-shared** `eltwise/unary` donor kernels, referenced by file path:
- `writer_unary_interleaved_start_id.cpp` — used by **both** factories; ~29 co-borrowers.
- `reader_unary_interleaved_start_id.cpp` — used by **prod_all only**; ~12 co-borrowers.

Both already Device 2.0-native. Their Metal 2.0 rewrite is a *shared* change every
co-borrower must adopt together, so migrating them in-place would break the co-borrowers
the instant prod moves. **Resolution: fork with `_metal2` suffix** (see
[Cross-op kernel handling](#cross-op-kernel-handling)).

### Flags / not-audited
- `device/kernels/dataflow/utils.hpp` — unreferenced by any bound kernel (dead code). Not touched.
- Legacy wrapper files `prod_op_all.*`, `prod_nc_op.*` (namespace `tt::operations::primary`)
  are thin shims to `ttnn::prim::prod_all` / `prod_nc`. Not part of the factory port; untouched.

---

## Plan the spec

### TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept` — realized as a static
  `create_program_artifacts(...)` on each `ProgramFactory`, returning
  `ttnn::device_operation::ProgramArtifacts`. This **replaces** the `create_descriptor`
  declaration in each device-op header (the concept requires `create_program_artifacts`
  and `!ProgramDescriptorFactoryConcept`, so `create_descriptor` must go).
- **Custom `compute_program_hash`:** none — nothing to delete.
- **Pybind:** `prod_nanobind.cpp` binds only the two `ttnn::prod` overloads; no
  `create_descriptor` / `create_program_descriptor` pybind hook. **No pybind edit needed.**
- **Implementation notes:** the two device-op `.hpp` headers change the factory-method
  signature only; the surrounding device-op class (`validate_*`, `compute_output_specs`,
  `create_output_tensors`) is untouched.

### Planned Spec Shape (both factories, 1:1 with legacy)

**prod_all** — one `ProgramSpec`, single node `{0,0}`, one `WorkUnitSpec`:
- KernelSpecs: `reader`, `writer`, `compute` (1:1 with the 3 legacy KernelDescriptors).
- DataflowBufferSpecs: `INPUT_DFB` (c_0), `OUTPUT_DFB` (c_3).
- TensorParameters: `INPUT`, `OUTPUT`.
- WorkUnitSpecs: one (`main`, all three kernels, `{0,0}`).

**prod_nc** — one `ProgramSpec`, multi-core, **two WorkUnitSpecs** (per core group):
- KernelSpecs: `reader`, `writer`, `compute_g1`, and `compute_g2` (present only when
  `core_group_2` non-empty). This **preserves the work-split multiplicity** — one
  KernelSpec per legacy compute KernelDescriptor. (Reader/writer are single instances,
  members of both WorkUnitSpecs.)
- DataflowBufferSpecs: `INPUT_DFB` (c_0), `OUTPUT_DFB` (c_3).
- TensorParameters: `INPUT`, `OUTPUT`.
- WorkUnitSpecs: `wu_g1` (reader, writer, compute_g1 → core_group_1); `wu_g2`
  (reader, writer, compute_g2 → core_group_2), added only when group 2 present.

### Preserved Multiplicity
- **prod_all:** none — no work-split multiplicity (single core).
- **prod_nc:**
  ```
  Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source prod_nc.cpp
    → KernelSpecs [compute_g1, compute_g2] of same source
    → in WorkUnitSpecs [wu_g1, wu_g2]  (DISJOINT node sets: core_group_1 / core_group_2)
    → sharing DFBs: INPUT_DFB (each CONSUMER), OUTPUT_DFB (each PRODUCER)
  ```
  Disjoint node sets → each node sees exactly one compute instance → each DFB is an
  ordinary per-node 1:1. **No `allow_instance_multi_binding` flag.** (This is the
  Demoting-CTA anti-pattern's *correct* form, not the same-grid two-toucher case.)

### CB endpoints (dispositions — re-derived from the kernel-touch census)
Both factories: `INPUT_DFB` (c_0) and `OUTPUT_DFB` (c_3), each a plain per-node
**1 producer + 1 consumer** FIFO:
- **prod_all** `c_0`: donor reader PRODUCER, compute CONSUMER. `c_3`: compute PRODUCER,
  donor writer CONSUMER.
- **prod_nc** `c_0`: own reader PRODUCER, compute CONSUMER. `c_3`: compute PRODUCER,
  donor writer CONSUMER. (compute_g1/g2 bind on disjoint groups → still 1:1 per node.)

No self-loop, no 1P+1C dual-instance assignment, no multi-binding flag, no dead-CB drop.
Census agrees with the brief.

### Dropped Plumbing

**Buffer-address RTAs → `TensorBinding`:**
- prod_all reader RTA 0 (`src_addr=input`) → `INPUT` TensorBinding on reader.
- prod_all writer RTA 0 (`dst_addr=output`) → `OUTPUT` TensorBinding on writer.
- prod_nc reader RTA 0 (`input_addr=input`) → `INPUT` TensorBinding on reader.
- prod_nc writer RTA 0 (`dst_addr=output`) → `OUTPUT` TensorBinding on writer.
All Case 1 (fed to `TensorAccessor`); no `get_bank_base_address` bridge.

**`TensorAccessorArgs` plumbing → binding mechanism:**
- prod_all reader `TensorAccessorArgs(input).append_to(...)` + kernel `TensorAccessorArgs<0>()`.
- prod_all/prod_nc writer `TensorAccessorArgs(output).append_to(...)` + kernel `TensorAccessorArgs<1>()`.
- prod_nc reader `TensorAccessorArgs(input.mesh_tensor()).append_to(...)` + kernel `TensorAccessorArgs<1>()`.

**Magic CB indices in CTAs → `DFBBinding`:**
- prod_all/prod_nc writer CTA `[0]=cb_id_out(c_3)` → OUTPUT_DFB binding (name gone from CTA).

**Positional CTAs → named CTAs:**
- prod_all compute CTA `[0]=num_tiles` → named `{"num_tiles", num_tiles}`. CTA `[1]=1`
  (`per_core_block_size`) is **dead** (kernel reads only CTA[0]) → not carried (no kernel read
  to name; structural consequence of the named-arg model, not a cleanup — see report).
- prod_nc reader CTA `[0]=dim` → named `{"dim", dim}`.

**Dead legacy args (audit "Misc anomalies" — NOT cleaned up, simply absent because no kernel reads them):**
- prod_nc reader RTA 7 (`dim`) — reader reads `dim` from its CTA, never RTA. No named RTA for it.
- prod_nc writer RTA 3 (`is_dram`) — donor writer reads only args 0–2. No named RTA for it.
- prod_nc compute CTA 0 (`num_cols_per_core_group_*`) — compute reads no CTA. No named CTA for it.
- prod_all compute CTA 1 (`per_core_block_size=1`) — compute reads only CTA[0]. No named CTA for it.
These are **not** port improvements: the named-arg model binds exactly the kernel's reads, and
these values were never read. Behavior unchanged. The underlying dead-arg anomalies remain the
ops team's to remove (flagged in the report).

**Named RTAs (kept; renamed to what the kernel reads):**
- prod_all reader: `num_pages`, `start_id`. writer: `num_pages`, `start_id`.
- prod_nc reader: `num_input_tiles`, `num_output_tiles`, `input_tile_offset`, `start_id`, `HtWt`, `CHtWt`.
  writer: `num_pages`, `start_id`. compute: `num_input_tiles`, `num_output_tiles`.

No semaphore-ID RTAs (no semaphores). No page-size 3rd-arg CTAs (every accessor is 2-arg).

### Applied Patterns
- **[Multi-variant / per-group work-split](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta):**
  prod_nc compute → two KernelSpecs of one source over two WorkUnitSpecs (disjoint node sets).
  Preserve per-group as separate KernelSpecs; do **not** demote to RTA. (prod_nc compute happens
  to read its per-group count from an RTA already, and has *no* per-group CTA — so there is
  nothing to demote; the multiplicity is preserved purely via two KernelSpecs + two WorkUnitSpecs.)
- **[Pass DFB handles directly to LLKs](../shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):**
  compute kernels pass `dfb::in` / `dfb::out` to `binary_op_init_common`, `pack_reconfig_data_format`,
  `copy_tile*`, `binary_dest_reuse_tiles*`, `pack_tile` (all take `uint32_t` cb id).
- **[Modifying a shared dataflow kernel → fork](../shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel):**
  the two donor kernels (see below).
- No self-loop, no conditional/optional bindings, no aliased DFBs, no varargs.

### Cross-op kernel handling
Fork each donor kernel with a `_metal2` suffix, **alongside the original** in
`eltwise/unary/device/kernels/dataflow/`, and point prod's KernelSpecs at the forks:
- `writer_unary_interleaved_start_id.cpp` → `writer_unary_interleaved_start_id_metal2.cpp`
  (used by **both** prod factories' writer KernelSpec).
- `reader_unary_interleaved_start_id.cpp` → `reader_unary_interleaved_start_id_metal2.cpp`
  (used by **prod_all** reader KernelSpec).

The legacy originals stay untouched for the ~29 / ~12 unmigrated co-borrowers. `eltwise/unary`
CMake globs `device/kernels/*.cpp`, so the forks are picked up automatically. Recorded in the
port report under "Open items for downstream" (fork path + remaining consumer set + sunset note).
Shared-accessor-name discipline: both prod factories bind the writer fork's output DFB with the
same `accessor_name` the fork source references (`out`), and its tensor with `output`.

### Own kernels (in-directory, Metal-2.0-ified in place)
- `reduction/prod/device/kernels/dataflow/reader_prod_nc.cpp` — `CircularBuffer`→`DataflowBuffer`,
  `TensorAccessor(tensor::input)`, named CTA `dim`, named RTAs, `get_tile_size(cb.get_cb_id())`
  → `dfb.get_tile_size()`.
- `reduction/prod/device/kernels/compute/prod_all.cpp` — CB→DFB, named CTA `num_tiles`, LLK cb-id
  args → `dfb::in`/`dfb::out`.
- `reduction/prod/device/kernels/compute/prod_nc.cpp` — CB→DFB, named RTAs, LLK cb-id args → dfb handles.

### Hardware configuration (Style B — direct Metal `ComputeConfigDescriptor`; build `ComputeGen1Config` directly)

**DM kernels (reader/writer, both factories):** all resolve to the reader/writer defaults
(`ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`). Use the arch-agnostic TTNN helpers
`create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.

**prod_all compute** — `ComputeGen1Config`:
- `fpu_math_fidelity = math_fidelity` (HiFi3 on WH when fp32_acc, else HiFi4).
- `enable_32_bit_dest = fp32_dest_acc_en` (= **true** always).
- `double_buffer_dest = !dst_full_sync_en` = `!false` = **true**.
- `sfpu_precision_mode` = (`math_approx_mode=true`) → **`Approximate`**.
- `bfp_pack_precision_mode` — legacy unset → default `Approximate`. Leave default.
- `unpack_modes` — compute CONSUMES INPUT_DFB. When INPUT_DFB format is `Float32`
  (input dtype FLOAT32) and `enable_32_bit_dest==true`, an explicit entry is **required**;
  legacy set no `unpack_to_dest_mode` (default `Default` → `UnpackMode::UnpackToSrc`).
  So add `{INPUT_DFB, UnpackMode::UnpackToSrc}` **iff** INPUT_DFB format == Float32.

**prod_nc compute** — `ComputeGen1Config`:
- `fpu_math_fidelity = math_fidelity` (HiFi3 on WH when fp32_acc, else HiFi4).
- `enable_32_bit_dest = fp32_dest_acc_en` (= `output.dtype()!=BFLOAT16`).
- `double_buffer_dest = !dst_full_sync_en` = **true**.
- `sfpu_precision_mode` — legacy `math_approx_mode` unset (default `false`) → **`Precise`** (leave default).
- `bfp_pack_precision_mode` — default `Approximate` (leave default).
- `unpack_modes` — compute CONSUMES INPUT_DFB (format = output-dtype format). Add
  `{INPUT_DFB, UnpackMode::UnpackToSrc}` **iff** INPUT_DFB format == Float32 (i.e. output FLOAT32).

Gen2 configs: not authored (port targets Gen1; helpers supply the Gen2 DM branch for free).

### Deferred / Flagged
- No new structural findings beyond the audit. The four dead legacy args and the `c_3`
  output-index deviation are pre-existing (audit "Misc anomalies"); left for the ops team.
- prod_nc's DFB data format keys off `output.dtype()` for **both** CBs (legacy behavior);
  the op's `validate` enforces input/output shape parity but the dtype relationship is the
  legacy factory's. Preserved verbatim.
