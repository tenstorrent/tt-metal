# Metal 2.0 Port Plan — `experimental/quasar/tilize_with_val_padding`

Audit: GREEN. Recipe @ `origin/akertesz/metal2-documentation`.
**Porting unit this pass: `TilizeWithValPaddingSingleCoreFactory`.** Other factories (multi_core_default / multi_core_block_interleaved / multi_core_sharded) stay on `ProgramDescriptorFactoryConcept`; per-factory dispatch keeps the op building/running. Enumerated as remaining work.

## Legacy Inventory (single-core factory)
- **Factory shape:** `ProgramDescriptorFactoryConcept` — `create_descriptor()` → `ProgramDescriptor`.
- **Custom `compute_program_hash`:** NONE on `TilizeWithValPaddingDeviceOperation` (audit confirmed). No deletion.
- **Kernels (3):** reader `reader_unary_pad_dims_split_rows.cpp` (single_core-only → port in place); writer `writer_unary_interleaved_start_id.cpp` (**shared** with multi_core_default → FORK); compute `compute/tilize.cpp` (**shared** with default+sharded → FORK).
- **CBs:** `src0`(idx0, in, `num_input_tiles`×`input_single_tile_size`) reader→compute; `output`(idx c_16, out, `num_output_tiles`×`output_single_tile_size`) compute→writer. Proper SPSC (1 producer/1 consumer each). No GlobalCB/aliasing/borrowed/tile-meta beyond format.
- **Semaphores:** none.
- **Tensors:** `input`(src0) Case 1 via `TensorAccessor`; `output`(dst) Case 1 via `TensorAccessor`. Buffer* passed in RTA slot 0 (BufferBinding form).
- **Work split:** single core (`sub_core_grids[0]` or `{0,0}`).
- **Cross-op kernels:** none (all op-owned, copied in earlier).
- **Flags:** all 3 kernels Device-2.0 clean. Reader CTA[1] (`unpadded_row_size_bytes`) is **dead** (kernel reads CTA[0] then `TensorAccessorArgs<2>`, never CTA[1]) → dropped.

## Planned Spec Shape (single-core → MetalV2FactoryConcept)
- **DFBs:** `IN`(entry=`input_single_tile_size`, num_entries=`num_input_tiles`, fmt=input), `OUT`(entry=`output_single_tile_size`, num_entries=`num_output_tiles`, fmt=output).
- **TensorParameters:** `INPUT`, `OUTPUT`.
- **KernelSpecs:** `READER`(reader src, IN producer, INPUT, CTA `bytes_per_tile_row`, 15 named RTAs, RoleHint READER); `WRITER`(writer fork, OUT consumer, OUTPUT, RTAs `num_pages`/`start_id`, RoleHint WRITER); `COMPUTE`(compute fork, IN consumer + OUT producer, CTAs `per_core_block_cnt`/`per_core_block_tile_cnt`, ComputeHardwareConfig{fp32_dest_acc_en=`fp32_llk_acc`; unpack_to_dest_mode[IN]=Fp32 when fp32}).
- **WorkUnitSpec:** `wu` = {READER,WRITER,COMPUTE} on the single core.
- **Op-owned tensors:** none.

## Forks (shared-kernel handling — recipe "fork the source" + `untilize_metal2.cpp` precedent)
- `compute/tilize.cpp` → `compute/tilize_metal2.cpp` (single_core binds the fork; legacy `tilize.cpp` retained for default/sharded).
- `writer_unary_interleaved_start_id.cpp` → `writer_unary_interleaved_start_id_metal2.cpp` (single_core binds fork; legacy retained for default).

## Dropped Plumbing
- Buffer-address RTAs (`src0_buffer` slot0 reader; `dst_buffer` slot0 writer) → `TensorBinding` (INPUT/OUTPUT), Case 1.
- `TensorAccessorArgs` plumbing (reader `<2>`, writer `<1>`) → binding mechanism.
- Magic CB indices (writer CTA `output_cb_index`; compute `c_0`/`c_16`; reader hardcoded `cb_id_in0=0`) → `DFBBinding`.
- Positional CTAs → named (`bytes_per_tile_row`; `per_core_block_cnt`/`per_core_block_tile_cnt`). Dead reader CTA `unpadded_row_size_bytes` dropped.
- Per-core RTAs → named.

## Applied Patterns
- Fork shared dataflow/compute kernel (Caution: Modifying a shared dataflow kernel).
- DM RoleHint READER/WRITER; compute `to_compute`-style hw via `ComputeHardwareConfig` + `unpack_to_dest_mode` for IN when fp32.

## Deferred / Flagged
- multi_core_default / block_interleaved / sharded factories — remaining work (later passes).

---
## Pass 2: multi_core_default factory (PORTED, build pending)
- Work-split (`split_blocks_for_tilize` → full `core_range` + cliff `core_range_cliff`); **full + cliff compute KernelSpecs** (preserved multiplicity). Reader/writer span all cores; both work units reference them.
- **Reuses the pass-1 forks** (`tilize_metal2.cpp`, `writer_unary_interleaved_start_id_metal2.cpp`). Reader `reader_unary_pad_dims_split_rows_multicore.cpp` (default-only) ported in place.
- **RTA varargs on the reader**: per-BlockRep run-length-encoded 5-tuples (n_data,n_mixed,n_pads,times,count_repeated) via `get_vararg`; named prefix `padded_X_size`/`pad_value`/`start_page_id`/`n_block_reps`; `advanced_options.num_runtime_varargs` = max over cores (padded with zeros). Mirrors the ported `untilize_with_unpadding_multi_core_interleaved` writer-vararg machinery.
- Dead reader CTA `aligned_page_size` (legacy CTA[5], unread) dropped.

---
## Remaining factories — analysis for next passes (NOT yet ported)

### multi_core_block_interleaved (per-group multiplicity)
Kernels (all block_interleaved-only → port in place, NO forks): compute `tilize_wh.cpp`, reader `reader_unary_pad_multicore_both_dims.cpp`, writer `writer_unary_interleaved_start_id_wh.cpp`.
3 CBs: `c_1` (per-row DRAM-alignment staging), `c_0` (input), `c_16` (output) — created **per core-group** via `push_padded_cb_pair` with **group-specific sizes** (full `core_range` uses `single_sub_block_size`; cliff groups use `single_block_size_cliff_row` / `single_sub_block_size`). Up to 4 groups: core_range, cliff_col_row, cliff_row, cliff_col. Compute is already per-group (4 CTA sets).
**Port shape:** a `DataflowBufferSpec` is uniform-size, so per-group sizes ⇒ **per-group DFB triples (IN_g/STAGING_g/OUT_g) + per-group reader/writer/compute KernelSpecs + per-group WorkUnitSpec** (the legacy *unified* reader/writer on all_cores must split per group). Route each core's per-core RTAs to its group's reader/writer by `CoreRangeSet::contains(core)`. No varargs. `split_blocks_for_tilize_wh` drives groups. Kernel ports: `tilize_wh` compute → dfb::in/out + named CTAs (block_size_col/row/third_dim); writer_wh → dfb::out/ta::output, `get_tile_size(cb_id)` → `cb.get_tile_size()` (rule 7), named args, keep BACKWARDS #ifdef; reader_both_dims → dfb::in (c_0) + dfb::staging (c_1), ta::input, named CTAs/RTAs.

### multi_core_sharded (borrowed-memory DFBs)
Kernels: reader `reader_unary_pad_height_width_sharded.cpp` + writer `writer_unary_sharded.cpp` (both sharded-only → port in place); compute `tilize.cpp` → **reuse the existing `tilize_metal2.cpp` fork** (CTAs per_core_block_cnt=nblocks_per_core, per_core_block_tile_cnt=ntiles_per_block match the fork).
4 CBs, uniform on all_cores: `c_1`(src0, **borrowed_from INPUT** when src_sharded — `cb.buffer=a.buffer()`), `c_0`(src1, intermediate tilize input, normal), `c_2`(src2, staging, likely **sync-free/single-ended** — verify), `c_16`(output, **borrowed_from OUTPUT** when out_sharded).
**Port shape:** map borrowed CBs via `DataflowBufferSpec::borrowed_from = INPUT/OUTPUT` (causal-link gate; resolves backing addr at runtime). Determine c_2's endpoints (sync-free → DM fabricate-consumer, or single-ended → STOP per catalog — **must verify before binding**). compute `in→c_0`, `out→c_16`. Pre-port: read both sharded kernels to map exact producer/consumer per CB (reader: reads borrowed c_1 + staging c_2 → produces c_0; compute c_0→c_16; writer consumes borrowed c_16). Set `unpack_to_dest_mode[IN]` when fp32. Recommend doing sharded BEFORE block_interleaved (more self-contained; reuses the fork).
