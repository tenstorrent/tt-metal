# Port Plan — `data_movement/tilize`

Port plan for `tilize`, ported from the legacy `ProgramDescriptor` (`create_descriptor`) API to Metal 2.0 (`create_program_artifacts` / `MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

**Scope (config-scoped, from the brief):** the brief cleared the subset — `TilizeMultiCoreDefaultProgramFactory`,
`TilizeSingleCoreProgramFactory`, `TilizeMultiCoreBlockProgramFactory`. **This pass ported Default + SingleCore;
`TilizeMultiCoreBlockProgramFactory` was DEFERRED** — see [Deferred/Flagged] Finding 2 and the port report (its per-group
DFB fan-out is materially larger and the confirmed tests barely exercise it). **`TilizeMultiCoreShardedProgramFactory`
stays on the legacy `create_descriptor` path** (RED-gated: `Runtime-args update == yes` + `Is safe to port? == no`).
The device-op `program_factory_t` variant runs mixed-concept (2 Metal-2.0 factories + 2 legacy descriptor factories);
the framework dispatches per-factory. The Block/Sharded plan sections below are retained as planning groundwork.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (each factory defines `create_descriptor()` → `ProgramDescriptor`).
- Variants (device-op `program_factory_t`): Default, SingleCore, Block, **Sharded (out of scope — stays legacy)**.
- Custom `compute_program_hash`: **none** — device op does not override it (default reflection-based hash). No deletion needed.
- `get_dynamic_runtime_args` exists (`tilize_device_operation.cpp:270`) but early-returns `{}` for the three ported
  factories (live only for sharded). It is device-op-class code, **off-limits** — leave untouched.

*(Target Metal 2.0 concept `MetalV2FactoryConcept`, chosen in the audit — carried forward under [TTNN ProgramFactory].)*

### Kernels — per factory

**Ownership legend:** OWN = in `tilize/device/kernels/`, edit in place. FORK = shared/cross-op kernel; co-borrowers
cannot co-migrate, so fork with `_metal2` suffix per [Caution: Modifying a shared dataflow kernel].

#### Default (`tilize_multi_core_default_program_factory.cpp`)
| unique_id | source | ownership | core_ranges | CTAs (positional) | RTAs (live slots) | config |
|---|---|---|---|---|---|---|
| reader | `.../tilize/.../reader_unary_stick_layout_split_rows_multicore.cpp` | OWN | all_cores | `[0]aligned_page_size(DEAD), [1]num_pages_in_row, [2]size_of_valid_data_in_last_page_in_row, TensorAccessorArgs<3>` | `[0]src0_buffer(Buffer*), [1]num_rows, [3]num_tiles_per_block, [4]block_width_size, [5]num_full_blocks_in_row, [8]start_page_id` (dead: 2,6,7) | ReaderConfigDescriptor{} |
| writer | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | **FORK** (42 consumers) | all_cores | `[0]output_cb_index(→DFB), TensorAccessorArgs<1>` | `[0]dst_buffer(Buffer*), [1]ntiles_per_core, [2]tile_start_id` | WriterConfigDescriptor{} |
| compute (full) | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | **FORK** (6 consumers) | core_range | `[0]nblocks_per_core, [1]ntiles_per_block` | — | ComputeConfigDescriptor{fp32_dest_acc_en, unpack_to_dest_mode} |
| compute (cliff) | same fork | FORK | core_range_cliff | `[0]nblocks_per_core_cliff, [1]ntiles_per_block` | — | same |

#### SingleCore (`tilize_single_core_program_factory.cpp`)
| unique_id | source | ownership | core_ranges | CTAs (positional) | RTAs (live slots) | config |
|---|---|---|---|---|---|---|
| reader | `.../tilize/.../reader_unary_stick_layout_split_rows_singlecore.cpp` | OWN | 1 core | `[0]stick_size(DEAD), TensorAccessorArgs<1>` | `[0]src0_buffer(Buffer*), [1]num_sticks, [3]num_tiles_per_block, [4]block_width_size, [5]num_full_blocks_in_row, [8]start_stick_id` (dead: 2,6,7) | ReaderConfigDescriptor{} |
| writer | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | **FORK** (same as Default) | 1 core | `[0]output_cb_index(→DFB), TensorAccessorArgs<1>` | `[0]dst_buffer(Buffer*), [1]num_tiles, [2]0` | WriterConfigDescriptor{} |
| compute | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | **FORK** (same as Default) | 1 core | `[0]num_tiles/num_tiles_per_block, [1]num_tiles_per_block` | — | ComputeConfigDescriptor{fp32_dest_acc_en, unpack_to_dest_mode} |

#### Block (`tilize_multi_core_block_program_factory.cpp`)
| unique_id | source | ownership | core_ranges | CTAs (positional) | RTAs (live slots) | config |
|---|---|---|---|---|---|---|
| reader | `tilize_with_val_padding/.../reader_unary_pad_multicore_both_dims.cpp` | **FORK** (in-family) | all_cores | `[0]total_num_rows,[1]third_dim,[2]tile_height,[3]element_size,[4]unpadded_X_size,[5]dram_alignment, TensorAccessorArgs<6>` | `[0]src0_buffer, [1]pad_value(=0), [2]width_size, [3]start_row_id, [4]start_column_id, [5]single_block_size_row_arg, [6]single_block_size_col_arg, [7]sub_block_width_size, [8]single_sub_block_size_row_arg` (all live) | ReaderConfigDescriptor{} |
| writer | `eltwise/unary/.../writer_unary_interleaved_start_id_wh.cpp` | **FORK** (2 consumers) | all_cores | `[0]cb_id_out(→DFB),[1]num_tiles_2d,[2]third_dim,[3]total_tiles_per_row, TensorAccessorArgs<4>` | `[0]dst_buffer, [1]tile_start_id, [2]single_block_size_row_arg, [3]single_block_size_col_arg` | WriterConfigDescriptor{} |
| compute ×(1–4 groups) | `.../tilize/device/kernels/compute/tilize_wh.cpp` | OWN | per group (core_range / cliff_col_row / cliff_row / cliff_col) | `[0]block_size_col, [1]block_size_row, [2]third_dim` (per-group values) | — | ComputeConfigDescriptor{fp32_dest_acc_en, unpack_to_dest_mode} |

### CBs — per factory
| factory | index | role | total_size | data_format | notes |
|---|---|---|---|---|---|
| Default | c_0 (in) | reader→compute | ntiles_per_block·in_tile | input dtype fmt | legal 1:1, one size on all_cores |
| Default | c_16 (out) | compute→writer | ntiles_per_block·out_tile | output dtype fmt | legal 1:1 |
| SingleCore | c_0 (in) | reader→compute | num_tiles_per_block·in_tile | input fmt | legal 1:1 |
| SingleCore | c_16 (out) | compute→writer | num_tiles_per_block·out_tile | output fmt | legal 1:1 |
| Block | c_1 (temp) | **reader-only staging** | input_row_bytes·N + 2·dram_align | input fmt | **self-loop** (reader P+C). Sized **per core-range group**. |
| Block | c_0 (in) | reader→compute | N·in_tile | input fmt | legal 1:1. Sized **per group**. |
| Block | c_16 (out) | compute→writer | N·out_tile | output fmt | legal 1:1. Sized **per group**. |

`tile_format_metadata`: legacy `CBFormatDescriptor.tile` unset in every CB → omit (`nullopt`).

### Semaphores
none — the op uses no semaphores.

### Tensor accessors
| host site | originating Tensor | RTA slot (host) | kernel accessor |
|---|---|---|---|
| all readers: `TensorAccessor(src_tensor_args, src_addr)` | input (`src0_buffer`) | slot 0 (Buffer*) | Case 1 |
| all writers: `TensorAccessor(dst_args, dst_addr)` | output (`dst_buffer`) | slot 0 (Buffer*) | Case 1 |

Both Case 1 (accessor iteration; no raw-pointer/`get_bank_base_address` bridge). No 3-arg `TensorAccessor`.
No `TensorParameter` relaxation. Pre-flight grep for `ArgConfig::Runtime*` in kernels: none present (not an eltwise
shape-agnostic case).

### Work split
- **Default:** `split_blocks_for_tilize(available_grid, nblocks)` → `(ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff)`. Compute splits full/cliff by CTA.
- **SingleCore:** n/a — single core.
- **Block:** `split_blocks_for_tilize_wh(available_grid, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit)` → up to **4 core-range groups** (core_range, cliff_col_row, cliff_row, cliff_col), each with its own block sizes → per-group CBs and compute CTAs.

### Cross-op / shared kernels (all FORK targets — see [Deferred/Flagged])
- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — 42 consumers.
- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` — 2 consumers.
- `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (shared pool) — 6 consumers.
- `data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` — in-family.

### Flags
- **Dead CTA/RTA slots** (audit "Misc anomalies"): Default/SingleCore reader CTA slot 0 (`aligned_page_size`/`stick_size`)
  never read; Default/SingleCore reader RTA slots 2,6,7 never read. **Not fixed** — they fall away naturally under the
  positional→named conversion (zero behavior change, since the kernel never read them). Recorded, not "cleaned up."
- **Unreferenced kernel** `device/kernels/compute/tilize.cpp` (op-local, distinct from the shared-pool
  `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`) — not referenced by any tilize factory; not audited, not port work. Untouched.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`:** none.
- **Implementation notes:** each ported factory's `.hpp` changes signature from
  `static ProgramDescriptor create_descriptor(...)` → `static ttnn::device_operation::ProgramArtifacts
  create_program_artifacts(const TilizeParams&, const TilizeInputs&, Tensor&)`. The sharded factory `.hpp`/`.cpp`
  are untouched. No pybind `create_descriptor` to remove (nanobind binds via `bind_function`). No factory-hook parameter to drop.

## Planned Spec Shape

Names shared across factories: `TensorParamName INPUT{"input"}, OUTPUT{"output"}`; `DFBSpecName` per factory below.

### Default
- **KernelSpecs:** `READER` (reader source), `WRITER` (writer fork), `COMPUTE_FULL` + `COMPUTE_CLIFF` (compute fork, same source, two specs — preserve CTA multiplicity). Cliff specs built only when the cliff core range is non-empty.
- **DataflowBufferSpecs:** `IN{"in"}` (entry_size=input_tile, num_entries=ntiles_per_block), `OUT{"out"}` (output_tile, ntiles_per_block). One size across all_cores.
- **TensorParameters:** INPUT, OUTPUT.
- **WorkUnitSpecs:** `WU_FULL{reader,writer,compute_full}`→core_range; `WU_CLIFF{reader,writer,compute_cliff}`→core_range_cliff (if present). reader/writer effective nodes = union = all_cores.
- **Bindings:** reader → IN PRODUCER (`dfb::in0`), tensor INPUT (`tensor::input`). writer → OUT CONSUMER (`dfb::out`), tensor OUTPUT. compute(both) → IN CONSUMER (`dfb::in`) + OUT PRODUCER (`dfb::out`).

### SingleCore
- **KernelSpecs:** `READER`, `WRITER`, `COMPUTE` (one each).
- **DataflowBufferSpecs:** `IN`, `OUT` (entry=tile size, num_entries=num_tiles_per_block).
- **TensorParameters:** INPUT, OUTPUT. **WorkUnitSpecs:** one `WU{reader,writer,compute}` on the single core.
- **Bindings:** same role pattern as Default.

### Block  *(per-group fan-out — see [Deferred/Flagged] Finding 2)*
- For each non-empty core-range group `G ∈ {full, cliff_col_row, cliff_row, cliff_col}`:
  - **KernelSpecs:** `READER_G`, `WRITER_G`, `COMPUTE_G` (reader/writer forks + own compute; one triple per group).
  - **DataflowBufferSpecs:** `IN_G`, `TEMP_G` (c_1 self-loop), `OUT_G` — sized from that group's block sizes.
  - **WorkUnitSpec:** `WU_G{reader_G, writer_G, compute_G}` → `G`.
  - **Bindings:** reader_G → IN_G PRODUCER (`dfb::in0`) + TEMP_G PRODUCER&CONSUMER (`dfb::in1`, self-loop), tensor INPUT. writer_G → OUT_G CONSUMER (`dfb::out`), tensor OUTPUT. compute_G → IN_G CONSUMER (`dfb::in`) + OUT_G PRODUCER (`dfb::out`).
- **TensorParameters:** INPUT, OUTPUT (singular across groups).
- Per-core RTAs are computed in the existing single loop over `available_grid`; each core's reader/writer RTAs are routed to its group's `READER_G`/`WRITER_G` via `AddRuntimeArgsForNode`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| Default: compute full + compute cliff (source `tilize.cpp`, different CTAs) | COMPUTE_FULL, COMPUTE_CLIFF | WU_FULL, WU_CLIFF | IN (both CONSUMER, disjoint grids), OUT (both PRODUCER, disjoint grids) — **no flag** |
| Block: up to 4 compute descriptors (source `tilize_wh.cpp`, per-group CTAs) — **plus** reader/writer fanned per group (Finding 2) | COMPUTE_G, READER_G, WRITER_G per group | WU_G per group | per-group IN_G/TEMP_G/OUT_G — each bound within one group; no cross-group sharing → **no flag** |

Endpoint census re-derived: no CB needs `allow_instance_multi_binding`. Default's two compute consumers of IN sit on
disjoint node sets (1P + [1C ⊎ 1C]); Block's per-group DFBs are single-group. Neither is a ≥3-toucher or same-role-lock.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| all readers RTA slot 0 | `src0_buffer` (Buffer*) | `TensorBinding(INPUT)` + `tensor::input` |
| all writers RTA slot 0 | `dst_buffer` (Buffer*) | `TensorBinding(OUTPUT)` + `tensor::output` |
| all readers CTA tail | `TensorAccessorArgs(*src0_buffer).append_to(...)` | binding mechanism (dropped) |
| all writers CTA tail | `TensorAccessorArgs(*dst_buffer).append_to(...)` | binding mechanism (dropped) |
| writer CTA slot 0 | `output_cb_index` / `cb_id_out` (magic CB index) | `DFBBinding(OUT, "out", CONSUMER)` + `dfb::out` |
| Default/SingleCore reader CTA slot 0 | `aligned_page_size` / `stick_size` (DEAD) | dropped (unread; named conversion omits it) |
| Default/SingleCore reader RTA slots 2,6,7 | duplicate `page_size` + two hardcoded `0` (DEAD) | dropped (unread) |
| all positional CTAs/RTAs | positional `uint32_t` lists | named `{{name,value},...}` / `runtime_arg_schema.runtime_arg_names` |

No semaphore-ID RTAs (no semaphores). No page-size 3rd-arg CTA (accessors are 2-arg). No Case 2 raw-pointer bindings.

## Applied Patterns
- [Self-loop DFB binding] — Block `c_1` (TEMP_G): reader bound both PRODUCER and CONSUMER (single-toucher staging buffer).
- [Two-toucher/disjoint work-split → no flag] — Default compute full+cliff share IN/OUT over disjoint grids; per-group compute in Block. **1P+1C-style, not `allow_instance_multi_binding`.**
- [Pass DFB handles directly to LLKs] — compute kernels pass `dfb::in`/`dfb::out` as `uint32_t` NTTPs to `compute_kernel_lib::tilize<...>`, `is_fp32_input_format<...>()`, `compute_kernel_hw_startup(...)`. (Verified: `DFBAccessor::operator uint32_t()` is `constexpr` → valid NTTP.)
- [Modifying a shared dataflow kernel → FORK] — 4 forks (see below).
- [Multi-variant/work-split placement] — WorkUnitSpec per (kernel-set, node-set) grouping.
- [DFB metadata via the object] — writer `get_local_cb_interface(cb).fifo_page_size` → `dfb.get_entry_size()`; writer_wh `get_tile_size(cb)` → `dfb.get_tile_size()`.

## Deferred / Flagged

- **Finding 1 — four kernel forks required.** Every clean-subset kernel except the two own readers and the own block
  compute (`tilize_wh.cpp`) is a shared/cross-op kernel whose co-borrowers are not co-migrating. Per [Caution: Modifying
  a shared dataflow kernel] the sanctioned bulk-port answer is fork-with-`_metal2`. Forks (alongside originals):
  `writer_unary_interleaved_start_id_metal2.cpp`, `writer_unary_interleaved_start_id_wh_metal2.cpp`,
  `ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp`, `reader_unary_pad_multicore_both_dims_metal2.cpp`. Legacy copies stay
  for unmigrated consumers; recorded in the report under Open items (sunset checklist). Not a blocker.
- **Finding 2 — Block per-group DFB fan-out.** Legacy Block allocates c_0/c_1/c_16 **per core-range group** with
  different sizes (one `push_cb_pair` per group), relying on legacy CBs being index-based per-core-range. Metal 2.0 DFBs
  are named specs with a **scalar** `entry_size`/`num_entries`, so per-group sizes require per-group DFB specs — which
  require per-group reader/writer/compute KernelSpecs (a kernel binds one DFB name). This is "preserved multiplicity"
  taken to its conclusion (mechanical, within Metal 2.0's expressive power), but it makes the Block factory materially
  larger/riskier than Default/SingleCore. **Sequencing:** port Default + SingleCore first (clean, shared forks), verify,
  then Block.
- **`unpack_modes` precision watch.** Legacy sets `unpack_to_dest_mode[c_0] = UnpackToDestFp32` whenever `fp32_llk_acc`
  (input FLOAT32 **or** FP8_E4M3, **or** output FP8_E4M3/BFLOAT8_B). Faithful map → `unpack_modes = {{IN, UnpackToDest}}`
  when `fp32_llk_acc`. **Risk:** the Metal 2.0 validator reportedly rejects `UnpackToDest` on a ≤16-bit-format DFB on
  Gen1; in the mixed cases (fp8 input, or bf16-in/bf8-out) the input DFB is ≤16-bit. Port faithfully; if the validator
  rejects, investigate at verification (misunderstanding vs genuine finding) — do not silently drop the entry.
