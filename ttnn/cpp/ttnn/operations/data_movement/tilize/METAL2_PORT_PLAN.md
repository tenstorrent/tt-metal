# Port Plan — tilize

Port plan for `data_movement/tilize`, ported from the TTNN `ProgramDescriptor` (descriptor) API to
Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.

**Scope**: the GREEN 5-factory subset {Default, SingleCore, Sharded, ShardedRetile, Retile}.
**Excluded (stays legacy)**: `TilizeMultiCoreBlockProgramFactory` (readiness-sheet `Known op issues =
"Per-node CB size"`, ops-team fix) and the dead `device/kernels/compute/tilize.cpp`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`descriptor`; `create_descriptor` returning `ProgramDescriptor`), all 6 factories.
- Variants: single device-op `TilizeDeviceOperation`, `program_factory_t` = variant of 6 factory structs (has `program_factory_t` already → **no direct-descriptor exception-3 edit**).
- Custom `compute_program_hash`: none — default reflection hash.
- Every factory defines `override_runtime_arguments` → **target `CustomProgramSpecFactoryConcept`** (inherited from audit).
- Shared cache-hit helper `patch_tilize_kernel_slot0` (tilize_device_operation.cpp:372) re-points slot 0 (buffer address) of a kernel's per-core args. Legacy also auto-registers `Buffer*`-form RTA slot-0 `BufferBinding`s. Both replaced by `TensorParameter`/`TensorArgument`.

### Kernels (per factory)

**Default** (`tilize_multi_core_default_program_factory.cpp`)
| id | source | ranges | CTAs (positional) | RTAs (slot→meaning) | config | opt_level(resolved) |
|---|---|---|---|---|---|---|
| reader | tilize/.../reader_unary_stick_layout_split_rows_multicore.cpp (op-owned, port in place) | all_cores | {tile_height, num_pages_in_row, size_of_valid_data_in_last_page_in_row} + TAArgs(src) | 0=src_buf*,1=nblocks*tile_h,2=page_size(dead),3=ntiles_per_block,4=page_size,5=1,6=0(dead),7=0(dead),8=page_start_id | Reader | O2 |
| writer | eltwise/unary/.../writer_unary_interleaved_start_id.cpp (**reuse `_metal2` fork**) | all_cores | {c_16} + TAArgs(dst) | 0=dst_buf*,1=ntiles,2=tile_start_id | Writer | O2 |
| compute (full) | ttnn/cpp/ttnn/kernel/compute/tilize.cpp (**reuse `tilize_metal2` fork**) | core_range | {nblocks_per_core, ntiles_per_block} | — | Compute fp32_dest_acc_en=fp32_llk_acc, unpack[c_0]=UnpackToDestFp32 if fp32_llk_acc&&!=UINT8 | O3 |
| compute (cliff) | same source | core_range_cliff | {nblocks_per_core_cliff, ntiles_per_block} | — | same | O3 |

`fp32_llk_acc = in∈{FLOAT32,FP8_E4M3,UINT8} ‖ out∈{FP8_E4M3,BFLOAT8_B}`.
Reader slots 2,6,7 are emitted but never read kernel-side → **dead RTAs, dropped** (multicore reader reads 0,1,3,4,5,8).

**SingleCore** (`tilize_single_core_program_factory.cpp`)
| id | source | ranges | CTAs | RTAs | config | opt |
|---|---|---|---|---|---|---|
| reader | tilize/.../reader_unary_stick_layout_split_rows_singlecore.cpp (op-owned, in place) | 1 core | {stick_size(dead), tile_height} + TAArgs(src) | 0=src*,1=num_sticks,2=stick_size(dead),3=ntpb,4=block_width_size,5=num_full_blocks,6=0(dead),7=0(dead),8=0(row_start) | Reader | O2 |
| writer | writer_unary_interleaved_start_id.cpp (**reuse fork**) | 1 core | {c_16}+TAArgs(dst) | 0=dst*,1=num_tiles,2=0 | Writer | O2 |
| compute | tilize.cpp (**reuse `tilize_metal2` fork**) | 1 core | {num_tiles/ntpb, ntpb} | — | Compute (as Default) | O3 |

Singlecore reader CTA slot 0 (stick_size) dead; reads only CTA(1)=tile_height. RTA slots 2,6,7 dead.

**Sharded** (`tilize_multi_core_sharded_program_factory.cpp`) — `output_is_interleaved` selects writer/output shape (structural, distinct cache entry).
| id | source | ranges | CTAs | RTAs | config | opt |
|---|---|---|---|---|---|---|
| reader | eltwise/unary/.../reader_unary_sharded.cpp (**reuse `_metal2` fork**) | shard grid | {c_0} | per-core {num_tiles_per_shard} | Reader | O2 |
| writer (interleaved out) | writer_unary_interleaved_start_id.cpp (**reuse fork**) | shard grid | {c_16}+TAArgs(dst) | 0=dst*,1=num_tiles_per_shard,2=tile_start_id | Writer | O2 |
| writer (sharded out) | sharded/.../writer_unary_sharded.cpp (**reuse `_metal2` fork**) | shard grid | {c_16} | per-core {num_tiles_per_shard} | Writer | O2 |
| compute | tilize.cpp (**reuse `tilize_metal2` fork**) | shard grid | {num_tiles_per_shard/num_tiles_per_row, num_tiles_per_row} | — | Compute (as Default) | O3 |
- c_0 borrowed from input shard buffer (`cb_src0.buffer = src_buffer`). c_16: interleaved → local (num_tiles_per_row); sharded → borrowed from dst_buffer (num_tiles_per_shard).

**ShardedRetile** (`tilize_multi_core_sharded_retile_program_factory.cpp`) — combines borrowed c_0 + aliased mid c_1/c_2 + borrowed/local c_16.
| id | source | ranges | CTAs | RTAs | config | opt |
|---|---|---|---|---|---|---|
| reader | reader_unary_sharded.cpp (**reuse fork**) | shard grid | {c_0} | per-core {num_tiles_per_shard_in} | Reader | O2 |
| writer (int/shard) | writer_unary_interleaved_start_id.cpp / writer_unary_sharded.cpp (**reuse forks**) | shard grid | as Sharded | as Sharded (num_tiles_per_shard_out) | Writer | O2 |
| compute | tilize/.../retile.cpp (op-owned, in place) | shard grid | {tiles_per_block,c_0,c_1,c_2,c_16,in_tile_h,out_tile_h,out_tile_size_input_fmt,mid_page_size} | per-core {num_input_tile_rows, num_input_tile_rows} | Compute fp32(3 CBs) | O3 |

**Retile** (`tilize_multi_core_retile_program_factory.cpp`) — interleaved in/out; aliased mid c_1/c_2 + double-buffered c_0/c_16.
| id | source | ranges | CTAs | RTAs | config | opt |
|---|---|---|---|---|---|---|
| reader | untilize/.../reader_unary_start_id.cpp (**reuse `_metal2` fork**) | all_cores | {c_0}+TAArgs(src) | 0=src*,1=num_input_tiles,2=input_tile_start_id | Reader | O2 |
| writer | writer_unary_interleaved_start_id.cpp (**reuse fork**) | all_cores | {c_16}+TAArgs(dst) | 0=dst*,1=num_output_tiles,2=output_tile_start_id | Writer | O2 |
| compute (full/cliff) | retile.cpp (op-owned, in place) | core_range / core_range_cliff | {tiles_per_block,c_0,c_1,c_2,c_16,in_tile_h,out_tile_h,out_tile_size_input_fmt,mid_page_size} | per-core {num_input_blocks, real_rows, real_output_rows} | Compute fp32(3 CBs) | O3 |

`fp32_llk_acc(retile) = in∈{FLOAT32,FP8_E4M3} ‖ out∈{FP8_E4M3,BFLOAT8_B}` (no UINT8 clause).

### CBs → DataflowBufferSpecs (per factory)
- **Default/SingleCore**: c_0 (input, ntiles_per_block or num_tiles_per_block × input_tile_size), c_16 (output, × output_tile_size). Local.
- **Sharded**: c_0 borrowed_from INPUT (num_tiles_per_shard × in_tile_size); c_16 interleaved→local(num_tiles_per_row) / sharded→borrowed_from OUTPUT(num_tiles_per_shard).
- **Retile/ShardedRetile**: c_0 (input; retile local double-buffered / shardedretile borrowed_from INPUT); **c_1 mid + c_2 mid_view = aliased pair** (single L1 region, mutual `alias_with`, equal total size = 2·mid_pages_per_out_block·mid_page_size); c_16 output (retile local / shardedretile borrowed_from OUTPUT or local interleaved).

### Semaphores: none (all factories).

### Tensor accessors
| factory | host site | tensor | note |
|---|---|---|---|
| Default | reader TAArgs(src), writer TAArgs(dst) | input, output | Case 1 |
| SingleCore | reader/writer TAArgs | input, output | Case 1 |
| Retile | reader TAArgs(src), writer TAArgs(dst) | input, output | Case 1 |
| Sharded | writer TAArgs(dst) (interleaved out only) | output | Case 1 (interleaved); input borrowed (clean); sharded output borrowed (clean) |
| ShardedRetile | writer TAArgs(dst) (interleaved out only) | output | as Sharded |

No accessor passes a 3rd (page-size) argument. TensorParameter relaxation: none.

### Work split
- Default/Retile: `ttnn::split_blocks_for_tilize(available_grid, nblocks|num_split_units)` → (ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff). Two same-source compute descriptors over disjoint full/cliff ranges.
- SingleCore: single core.
- Sharded/ShardedRetile: per-core over shard grid; one compute instance.

### Shared kernels — all reuse existing `_metal2` forks (do NOT re-fork, do NOT edit legacy original)
| legacy source | fork to bind | vocabulary |
|---|---|---|
| ttnn/cpp/ttnn/kernel/compute/tilize.cpp | tilize_metal2.cpp | dfb::in(CONSUMER), dfb::out(PRODUCER); args::per_core_block_cnt, per_core_block_tile_cnt |
| eltwise/unary/.../writer_unary_interleaved_start_id.cpp | ..._metal2.cpp | dfb::out(CONSUMER), tensor::dst; args::num_pages, start_id; defines OUT_SHARDED/BACKWARDS (unused here) |
| untilize/.../reader_unary_start_id.cpp | ..._metal2.cpp | dfb::in(PRODUCER), tensor::src; args::num_tiles, start_id |
| eltwise/unary/.../reader_unary_sharded.cpp | ..._metal2.cpp | dfb::in(PRODUCER, borrowed); args::num_tiles_per_core |
| sharded/.../writer_unary_sharded.cpp | ..._metal2.cpp (#52228) | dfb::out(CONSUMER); args::num_units |

Op-owned kernels ported **in place**: reader_unary_stick_layout_split_rows_{multicore,singlecore}.cpp, compute/retile.cpp.

### Flags
- Dead file `device/kernels/compute/tilize.cpp` — unreferenced, not ported (route to ops team).
- Reader dead RTA/CTA slots (multicore/singlecore) dropped as noted above.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` (every factory has `override_runtime_arguments`).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: op already has `program_factory_t` (no exception-3 restructure). Block factory + dead file left on legacy concept; the variant stays mixed (framework dispatches per-factory). No device-op-class edits, no nanobind edits (no pybound `create_descriptor`).

## Planned Spec Shape (per factory)
- **Default**: KernelSpecs {READER, WRITER, COMPUTE_FULL, COMPUTE_CLIFF (each present only if its range non-empty)}; DFBs {INPUT(c_0 local), OUTPUT(c_16 local)}; TensorParameters {INPUT, OUTPUT}; WorkUnitSpecs {full: READER+WRITER+COMPUTE_FULL @core_range ∪ cliff}, but note reader/writer span all_cores → see Preserved Multiplicity. No semaphores.
- **SingleCore**: {READER, WRITER, COMPUTE} on 1 core; DFBs {INPUT, OUTPUT}; TP {INPUT, OUTPUT}; 1 WorkUnit.
- **Sharded**: {READER, WRITER, COMPUTE}; DFBs {INPUT(borrowed), OUTPUT(local or borrowed)}; TP {INPUT, OUTPUT}; 1 WorkUnit @shard grid.
- **ShardedRetile**: {READER, WRITER, COMPUTE}; DFBs {INPUT(borrowed), MID(c_1)+MID_VIEW(c_2) aliased self-loop, OUTPUT(local/borrowed)}; TP {INPUT, OUTPUT}; 1 WorkUnit.
- **Retile**: {READER, WRITER, COMPUTE_FULL, COMPUTE_CLIFF}; DFBs {INPUT(local), MID+MID_VIEW aliased self-loop, OUTPUT(local)}; TP {INPUT, OUTPUT}.

## Preserved Multiplicity (Default, Retile)
```
Legacy compute KernelDescriptors [core_range, core_range_cliff] of tilize.cpp / retile.cpp
  → KernelSpecs [COMPUTE_FULL, COMPUTE_CLIFF] of the same fork source
  → in WorkUnitSpecs [WU_FULL @core_range, WU_CLIFF @core_range_cliff]
  → reader/writer span all_cores: bound in BOTH WorkUnits (their KernelSpec is shared across groups)
  → shared DFBs: INPUT (reader PRODUCER, compute CONSUMER — per node one compute instance → ordinary 1:1),
                 OUTPUT (compute PRODUCER, writer CONSUMER — 1:1).
```
Two same-source compute KernelSpecs over disjoint node sets each bind one role → ordinary 1:1, NOT multi-binding. Per-group CTAs (nblocks_per_core vs _cliff) preserved as distinct KernelSpec CTAs.
Sharded/SingleCore: none — single compute instance.

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (all factories) | `src_buffer` (Buffer*-form BufferBinding) + `patch_tilize_kernel_slot0(prog,0,addr)` | `TensorParameter INPUT` + `tensor::src`/borrowed_from + override tensor_args |
| writer RTA slot 0 (all factories) | `dst_buffer` (Buffer*-form) + `patch_tilize_kernel_slot0(prog,1,addr)` | `TensorParameter OUTPUT` + `tensor::dst`/borrowed_from + override tensor_args |
| reader/writer CTA {c_0}/{c_16} magic index | CB index CTA | `DFBBinding` (accessor dfb::in/dfb::out) |
| reader/writer `TensorAccessorArgs(buf).append_to(cta)` | host TA plumbing + kernel `TensorAccessorArgs<N>()` | `TensorBinding` + kernel `TensorAccessor(tensor::name)` |
| sharded `cb_addr_only` ProgramDescriptor + `apply_descriptor_runtime_args` | borrowed-CB address refresh | `DataflowBufferSpec::borrowed_from` + override tensor_args |
| reader RTA slots 2,6,7 (multi/single); reader CTA slot 0 (single, stick_size) | emitted-but-unread positional args | dropped (not emitted) |
| all positional CTAs/RTAs | positional | named (`args::name`) |

## Applied Patterns
- Aliased DFBs (retile/shardedretile mid_cb + mid_view_cb): mutual `advanced_options.alias_with`, equal total size, same node set.
- Self-loop DFB: MID + MID_VIEW (compute both PRODUCER and CONSUMER; MID_VIEW has hand-driven read cursor, no FIFO producer).
- Borrowed-memory DFB: sharded input (INPUT), sharded output (OUTPUT) via `borrowed_from`.
- Two same-source compute KernelSpecs over disjoint node sets (Default/Retile full+cliff) → ordinary 1:1 (NOT multi-binding, NOT demoting CTA→RTA).
- CustomProgramSpecFactoryConcept override: refreshes only tensor_args {INPUT, OUTPUT} (mirrors legacy slot-0/CB-addr refresh; all shape-derived args baked).

## Deferred / Flagged
- **unpack_modes (resolved, not a blocker)**: legacy `UnpackToDestFp32` on c_0 (and c_1/c_2 for retile) maps to `UnpackMode::UnpackToDest`. Validator (program_spec.cpp:1065) permits UnpackToDest on ANY format when `enable_32_bit_dest` is true — and the same `fp32_llk_acc` flag drives both here, so every legacy-set CB is permitted regardless of width. Float32 required-entry rule satisfied (legacy always sets c_0 when input FLOAT32). Faithful mechanical mapping.
- **Aliased mid num_entries**: MID entry_size=mid_page_size, num_entries=2·mid_pages_per_out_block; MID_VIEW entry_size=out_tile_size_input_fmt, num_entries=(total)/entry_size — division exact (input-format tiles, height-proportional). Both totals equal by construction.
