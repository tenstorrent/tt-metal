# Port Plan — fill_pad

Port plan for `ttnn/cpp/ttnn/operations/data_movement/fill_pad`, ported from
`ProgramDescriptor` (`create_descriptor`) to Metal 2.0 (`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this port**: BOTH factories are ported together in one change, because they
**share the compute kernel** `device/kernels/compute/fill_pad_compute.cpp` (an intra-op
shared kernel). Metal-2.0-ifying that source for one factory would break the other's
legacy binding of it, so co-porting is the clean move and avoids a `_metal2` fork. The
audit also framed the two factories as "one porting unit."

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories define `create_descriptor()` returning `tt::tt_metal::ProgramDescriptor`).
- Variants: two program factories under one `FillPadDeviceOperation`:
  - `FillPadProgramFactory` — DRAM interleaved + DRAM-sharded. Single reader/writer/compute over `all_cores`.
  - `FillPadL1ShardedProgramFactory` — L1 HEIGHT/WIDTH/BLOCK sharded. Per-shard local self-reads/writes.
- Custom `compute_program_hash`: **none** — already default reflection-based hash (audit confirmed).

### Kernels (FillPadProgramFactory — DRAM)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/fill_pad_reader.cpp` | all_cores | [0]W_tiles [1]H_tiles [2]N_slices [3]has_right_pad [4]has_bottom_pad [5]W_mod32 [6]H_mod32 [7]elem_size [8]fill_bits [9]cb_data_in=0 [10+]TensorAccessorArgs | [0]buf_addr [1]start_right [2]num_right [3]start_bottom [4]num_bottom [5]start_corner [6]num_corner | kernel_defines (MASK_ELEM_UINT/MASK_VALUE/FILL_PAD_DATA_FMT/FILL_PAD_FILL_FN/FILL_PAD_FILL_ARG…) | absent → O2 | ReaderConfigDescriptor{} |
| writer | `device/kernels/dataflow/fill_pad_writer.cpp` | all_cores | [0]W_tiles [1]H_tiles [2]N_slices [3]has_right_pad [4]has_bottom_pad [5]W_mod32 [6]H_mod32 [7]cb_right_mask=1 [8]cb_bot_mask=2 [9]cb_data_out=16 [10+]TensorAccessorArgs | [0]buf_addr [1]start_right [2]num_right [3]start_bottom [4]num_bottom [5]start_corner [6]num_corner | same kernel_defines | absent → O2 | WriterConfigDescriptor{} |
| compute | `device/kernels/compute/fill_pad_compute.cpp` | all_cores | [0]W_tiles [1]H_tiles [2]has_right_pad [3]has_bottom_pad [4]elem_size [5]fill_bits [6]cb_data_in=0 [7]cb_right_mask=1 [8]cb_bot_mask=2 [9]cb_data_out=16 | [0]num_right [1]num_bottom [2]num_corner | same kernel_defines | absent → **O3** (compute default) | ComputeConfigDescriptor{.fp32_dest_acc_en=need_fp32_dest_acc, .unpack_to_dest_mode=…} |

Reads actually consumed (per source; the rest are positional-layout padding shared across reader/writer):
- reader reads CTA 0,1,3,4,7 and the CB-id at 9; **does not** read 2,5,6,8.
- writer reads CTA 0,1,3,4,5,6 and the CB-ids at 7,8,9; **does not** read 2.
- compute reads all of 0..9; **W_tiles(0), H_tiles(1), elem_size(4) are read into a `constexpr` and never used** (dead). Only elem_size flagged by the audit.

### Kernels (FillPadL1ShardedProgramFactory — L1-sharded)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader (per has_right_pad group) | `device/kernels/dataflow/fill_pad_sharded_reader.cpp` | rw_ranges[rp] | [0]W_tiles(=pages_per_shard_x) [1]has_right_pad(=rp) [2]elem_size [3]cb_data_in=0 | [0]shard_l1_base [1]shard_H_tiles [2]has_bottom_pad_core [3]num_work [4]local_right_col | absent → O2 | ReaderConfigDescriptor{} |
| writer (per has_right_pad group) | `device/kernels/dataflow/fill_pad_sharded_writer.cpp` | rw_ranges[rp] | [0]W_tiles [1]has_right_pad(=rp) [2]W_mod32 [3]H_mod32 [4]cb_right_mask=1 [5]cb_bot_mask=2 [6]cb_data_out=16 | [0]shard_l1_base [1]shard_H_tiles [2]has_bottom_pad_core [3]num_work [4]local_right_col | absent → O2 | WriterConfigDescriptor{} |
| compute (per (rp,bp,H,eff_W) group) | `device/kernels/compute/fill_pad_compute.cpp` | compute_ranges[key] | [0]W_tiles(=eff_W) [1]H_tiles(=key.H) [2]has_right_pad [3]has_bottom_pad [4]elem_size [5]fill_bits [6]cb_data_in=0 [7]cb_right_mask=1 [8]cb_bot_mask=2 [9]cb_data_out=16 | [0]num_right [1]num_bottom [2]num_corner | absent → **O3** | ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode} |

Sharded reader/writer bottom-pad is **runtime** (`has_bottom_pad_core` RTA drives Mode A/B); compute bottom-pad is **compile-time** (`key.has_bottom_pad`). `num_work` RTA is a "has any work" guard: used by the writer's early-return (`fill_pad_sharded_writer.cpp:60`), **inert in the reader** (dead RTA — keep it, audit says so). `elem_size` CTA dead in the reader.

### CBs (both factories)
| index | name | total_size | num_entries | data_format | page_size | tile | condition |
|---|---|---|---|---|---|---|---|
| c_0 | data-in | tile_bytes*2 | 2 | cb_data_format | tile_bytes | (unset) | always |
| c_1 | right-mask | tile_bytes | 1 | cb_data_format | tile_bytes | (unset) | has_right_pad |
| c_2 | bottom-mask | tile_bytes | 1 | cb_data_format | tile_bytes | (unset) | has_bottom_pad |
| c_16 | data-out | tile_bytes*2 | 2 | cb_data_format | tile_bytes | (unset) | always |

`tile_bytes = tt::tile_size(cb_data_format)`. No `.tile` field set → default 32×32. DRAM factory places CBs on `all_cores`; sharded places on `all_active_set`. **In Metal 2.0 the placement is derived from bindings**, so the explicit `core_ranges` on the CBs is dropped (not replicated).

### Semaphores
none — the op uses no semaphores.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) | binding case |
|---|---|---|---|
| `fill_pad_reader.cpp:86-87` `TensorAccessor(src_args, buf_addr, tile_bytes)` | `input` (in-place; also the output) | reader RTA[0] `buf_addr` | **Case 1** (DRAM) |
| `fill_pad_writer.cpp:80-81` `TensorAccessor(dst_args, buf_addr, tile_bytes)` | `input` | writer RTA[0] `buf_addr` | **Case 1** (DRAM) |
| sharded reader/writer — **no `TensorAccessor`**; raw `shard_l1_base` RTA[0] → `UnicastEndpoint` self-reads/writes | `input` | reader/writer RTA[0] `shard_l1_base` | **Case 2** (raw pointer) |

The op is **in-place**: `create_output_tensors` returns the input tensor (`fill_pad_device_operation.cpp:39-43`), so reader and writer bind the *same* tensor. Both factories deliver the base today via the framework `Buffer*`-binding form (`emplace_runtime_args({tens_buffer, …})`).

### Work split
- **DRAM**: `split_work_to_cores(compute_with_storage_grid_size, total_work)` → `(num_cores, all_cores, core_group_1, core_group_2, num_work_per_core_group_1, num_work_per_core_group_2)`. **Per-group difference is RTA-only** (num_work → per-core num_right/num_bottom/num_corner); no per-group CTA. One compute binary over all cores. `total_work = T_right + T_bottom + T_corner` (unified border-tile split).
- **Sharded**: no `split_work_to_cores`; per-shard-core enumeration. reader/writer grouped by `has_right_pad`; compute grouped by `ComputeKey{has_right_pad, has_bottom_pad, H, effective_W}`.

### Shared kernels
- **Intra-op**: `device/kernels/compute/fill_pad_compute.cpp` bound by **both** factories. Ported in place (both factories convert in the same change → no `_metal2` fork needed). `grep -rl fill_pad_compute.cpp ttnn/cpp/ttnn/operations/` → only this op's two factories bind it; no external consumer.
- `device/kernels/dataflow/fill_pad_dataflow_common.hpp` — shared header, bound by DRAM writer + sharded writer. Contains only templated `push_right_mask_tile` / `push_bottom_mask_tile` over `CB_T&` using FIFO methods + `get_write_ptr()`. **No `cb_id`/`dfb::` references inside → unchanged, no fork.**
- No borrowed (out-of-directory) kernels; op owns all five sources (audit "Team-only" clean).

### Flags
- Dead CTAs: `elem_size` (reader, compute, sharded_reader), plus `W_tiles`/`H_tiles` read-but-unused in compute. Preserved as named CTAs (faithful — not cleaned up).
- Dead RTA: `num_work` in the sharded reader — kept inert (audit: keeping it is the safe behavior).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both factories).
- **Custom `compute_program_hash`**: none — already default.
- **Implementation notes**:
  - Both `create_descriptor` methods become `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`. Factory struct signatures in the header change accordingly.
  - No pybind `create_descriptor` binding exists (`fill_pad_nanobind.cpp` binds only the `fill_implicit_tile_padding` free function) → **no pybind edit forced**.
  - No custom-hash deletion, no pybind removal, no pybind-hook parameter to drop. Clean.

## Planned Spec Shape

### FillPadProgramFactory (DRAM) — single program
- **KernelSpecs** (3): `READER` (fill_pad_reader.cpp), `WRITER` (fill_pad_writer.cpp), `COMPUTE` (fill_pad_compute.cpp). One WorkUnitSpec over `all_cores`.
- **DataflowBufferSpecs**: `DATA_IN` (c_0, 2 entries), `DATA_OUT` (c_16, 2 entries) always; `RIGHT_MASK` (c_1, 1 entry) iff has_right_pad; `BOT_MASK` (c_2, 1 entry) iff has_bottom_pad. entry_size = tile_bytes; data_format_metadata = cb_data_format.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (in-place input; also output). Bound by READER and WRITER (Case 1 → `TensorAccessor(tensor::input)`). Not bound by compute (CB-only).
- **WorkUnitSpecs** (1): `{READER, WRITER, COMPUTE}` over `all_cores`.
- **DFB bindings**:
  - DATA_IN: READER PRODUCER, COMPUTE CONSUMER.
  - DATA_OUT: COMPUTE PRODUCER, WRITER CONSUMER.
  - RIGHT_MASK (iff has_right_pad): WRITER PRODUCER, COMPUTE CONSUMER.
  - BOT_MASK (iff has_bottom_pad): WRITER PRODUCER, COMPUTE CONSUMER.

### FillPadL1ShardedProgramFactory — single program, grouped by ComputeKey
- **Grouping decision**: group **reader, writer, AND compute all by the full `ComputeKey`** `(has_right_pad, has_bottom_pad, H, effective_W)`, one **WorkUnitSpec per key group** containing `{reader_k, writer_k, compute_k}`. This is a superset of legacy's reader/writer-by-`has_right_pad` grouping (reader/writer binaries are identical for same-`has_right_pad` keys, since their CTAs depend only on `has_right_pad`; splitting them per-key produces identical binaries over disjoint node sets — legal, behavior-identical). Rationale: it makes every DFB's producer and consumer share one WorkUnitSpec, satisfying the per-node 1P+1C invariant **and** aligning the conditionally-bound masks (see below) without the writer's mask binding spilling onto cores that don't consume it.
- **DataflowBufferSpecs**: `DATA_IN` (2), `DATA_OUT` (2) always; `RIGHT_MASK` (1) iff global has_right_pad; `BOT_MASK` (1) iff global has_bottom_pad. entry_size = tile_bytes; data_format = cb_data_format.
- **TensorParameters**: `INPUT` — **Case 2** binding. Bound by every reader_k and writer_k (DM); base pulled via `TensorAccessor(tensor::input).get_bank_base_address()`; raw `shard_l1_base + geometry` arithmetic unchanged. Not bound by compute.
- **DFB bindings per key group**:
  - DATA_IN: reader_k PRODUCER, compute_k CONSUMER.
  - DATA_OUT: compute_k PRODUCER, writer_k CONSUMER.
  - RIGHT_MASK (iff key.has_right_pad): writer_k PRODUCER, compute_k CONSUMER.
  - BOT_MASK (iff key.has_bottom_pad): writer_k PRODUCER, compute_k CONSUMER.

  Note: because `key.has_bottom_pad == core_has_bottom_pad == writer RTA has_bottom_pad_core`, per group the writer's runtime `if (has_bottom_pad_core)` mask-push is uniformly taken (bp=1 group) or skipped (bp=0 group), so the compile-time `BOT_MASK` binding matches the runtime production exactly.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| DRAM: 1 reader, 1 writer, 1 compute (all over all_cores; per-core work split is RTA-only) | 1 READER, 1 WRITER, 1 COMPUTE | 1 (all_cores) | DATA_IN (READER P / COMPUTE C), DATA_OUT (COMPUTE P / WRITER C), RIGHT_MASK/BOT_MASK (WRITER P / COMPUTE C) |
| Sharded: readers/writers per `has_right_pad` (≤2 each), compute per ComputeKey (≤~4) | reader_k, writer_k, compute_k **per ComputeKey** | 1 per ComputeKey | per group: DATA_IN (reader_k P / compute_k C), DATA_OUT (compute_k P / writer_k C), RIGHT_MASK/BOT_MASK (writer_k P / compute_k C) |

No same-grid dual-instance work-split (reader and writer are distinct sources). Each node sees exactly one reader, one writer, one compute → ordinary 1P+1C per DFB per node; no `allow_instance_multi_binding`, no self-loop.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `fill_pad_reader.cpp` RTA[0] `buf_addr` | `emplace_runtime_args({tens_buffer,…})` + kernel `get_arg_val<uint32_t>(0)` | `TensorBinding(INPUT)` → `TensorAccessor(tensor::input)` |
| `fill_pad_reader.cpp:86` host `TensorAccessorArgs(*tens_buffer).append_to(reader_ct)` + kernel `TensorAccessorArgs<10>()` | manual accessor-args plumbing | binding mechanism (auto CTAs) |
| `fill_pad_reader.cpp:87` `TensorAccessor(src_args, buf_addr, tile_bytes)` 3rd arg | `tile_bytes` page-size 3rd arg | dropped (Class 2 no-op); accessor supplies aligned page size |
| `fill_pad_reader.cpp:65,90` `cb_tile_in_idx = get_compile_time_arg_val(9)` | magic CB index CTA | `DFBBinding` DATA_IN → `DataflowBuffer(dfb::data_in)` |
| `fill_pad_reader.cpp:76` `get_tile_size(cb_tile_in_idx)` | free `get_tile_size(cb_id)` | `dfb_tile_in.get_entry_size()` (rule 7 / whitelist §B) |
| reader CTAs 2/5/6/8 (N_slices/W_mod32/H_mod32/fill_bits) | positional-layout padding never read by reader | dropped (named CTAs are per-kernel) |
| `fill_pad_writer.cpp` RTA[0] `buf_addr`, `TensorAccessorArgs` (`:80,192`), 3rd arg (`:81`), cb CTAs 7/8/9 | as above | `TensorBinding(INPUT)`, `DFBBinding`s, `get_entry_size()` |
| writer CTA 2 (N_slices) | positional-layout padding | dropped |
| all reader/writer/compute positional CTAs | positional | **named** CTAs |
| compute CB CTAs 6/7/8/9 | magic CB indices | `DFBBinding`s → `dfb::data_in/right_mask/bot_mask/data_out` |
| compute `has_right_pad`(2)/`has_bottom_pad`(3) CTAs | gate `if constexpr` around conditional mask DFB refs | promoted to `#define FILL_PAD_HAS_RIGHT_PAD` / `FILL_PAD_HAS_BOTTOM_PAD` (`compiler_options.defines`) + `#ifdef` (Conditional-DFB pattern) |
| writer `has_right_pad`(3)/`has_bottom_pad`(4) CTAs | gate `if constexpr` around conditional mask DFB refs | promoted to `#define` + `#ifdef` |
| sharded reader/writer RTA[0] `shard_l1_base` | `emplace_runtime_args({tens_buffer,…})` | **Case 2**: `TensorBinding(INPUT)` → `TensorAccessor(tensor::input).get_bank_base_address()`, raw arithmetic unchanged |
| sharded reader/writer/compute CB-id CTAs | magic CB indices | `DFBBinding`s |
| sharded writer `has_right_pad`(1) CTA + compute `has_right_pad`(2)/`has_bottom_pad`(3) | gate conditional mask DFBs | `#define` + `#ifdef` |
| sharded reader/writer `get_tile_size(cb_id)` | free helper | `dfb.get_entry_size()` |

Kept (named, not dropped): reader `elem_size`; compute `W_tiles`/`H_tiles`/`elem_size` (dead-but-preserved per audit); sharded reader `elem_size` CTA + `num_work` RTA (inert, preserved per audit).

## Applied Patterns

- [Conditional / optional DFB bindings](../shared/port_patterns.md): `RIGHT_MASK` gated on `has_right_pad`, `BOT_MASK` on `has_bottom_pad`, in the writer and compute kernels of both factories — host binds conditionally + emits `FILL_PAD_HAS_RIGHT_PAD`/`FILL_PAD_HAS_BOTTOM_PAD` defines; kernels `#ifdef`-gate the `DataflowBuffer` construction and every mask reference. **Promote-a-CTA-gate-to-a-define** sub-case (legacy used `if constexpr (has_right_pad)`).
- [Pass DFB handles directly to LLKs](../shared/port_patterns.md): compute kernel passes `dfb::data_in`/`dfb::data_out` to `unary_op_init_common`; the shared `push_*_mask_tile` helpers keep taking `DataflowBuffer&`.
- Case 2 raw-pointer tensor binding: sharded reader/writer via `get_bank_base_address()`.
- TensorAccessor 3rd-arg drop (Class 2): DRAM reader/writer.

## Deferred / Flagged

- **New finding (structural, resolved in-plan, not a stop):** the sharded factory's legacy reader/writer group by `has_right_pad` only while compute groups by the full `ComputeKey`, and the bottom mask is produced at *runtime* (`has_bottom_pad_core`) but consumed at *compile-time* (`key.has_bottom_pad`). Under Metal 2.0's derived DFB placement + per-node 1P+1C invariant, a writer bound to `BOT_MASK` over a whole `has_right_pad` group would place the mask producer on bottom-pad=0 nodes that have no consumer → invalid. **Resolution:** group reader/writer by the full `ComputeKey` too (behavior-identical multiplicity), so each WorkUnitSpec carries a matched {reader,writer,compute} and the mask bindings line up per node. No kernel-logic change; the sharded writer still reads `has_bottom_pad_core` as a runtime arg (also drives Mode A/B write-back geometry).
- No feature gate missed; no capitulation.
