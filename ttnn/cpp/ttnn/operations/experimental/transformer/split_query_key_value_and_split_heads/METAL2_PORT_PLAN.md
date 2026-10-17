# Port Plan — split_query_key_value_and_split_heads

Port plan for `experimental/transformer/split_query_key_value_and_split_heads`, ported from the `ProgramDescriptor` API (`create_descriptor`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Both factories of `SplitFusedQKVAndSplitHeadsDeviceOperation` are ported in this change. They share no kernel source, so each is an independent atomic unit.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor`) for both factories, declared as structs inside `program_factory_t` (`device/split_query_key_value_and_split_heads_device_operation.hpp:22,31,44-45`). Not the direct-descriptor shape, so exception 3 does not apply. The port is a method swap inside each existing struct.
- Variants: two factories, selected by `input_tensor.is_sharded()` (`device_operation.cpp:12-18`):
  - `SplitFusedQKVAndSplitHeadsProgramFactory`: interleaved (`device/split_query_key_value_and_split_heads_program_factory.cpp`)
  - `SplitFusedQKVAndSplitHeadsShardedProgramFactory`: block-sharded (`device/split_query_key_value_and_split_heads_sharded_program_factory.cpp`)
- Custom `compute_program_hash`: none, so the default reflection-based hash applies. There is no `attribute_values` / `to_hash` either.
- `override_runtime_arguments`: none, on either factory.

### Variant: Interleaved

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_tm_tile_layout_create_qkv_heads.cpp` | `all_cores` = `{(0,0)-(num_cores_c-1, num_cores_r-1)}` | 0 `block_size`, 1 `num_blocks_per_tensor` (`out_num_blocks_per_tensor`), 2.. `TensorAccessorArgs(in0_buffer)` | — | 0 `in0_buffer` (`Buffer*`), 1 `in0_tensor_tile_id` | — | — | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer | `kernels/dataflow/writer_tm_tile_layout_create_qkv_heads.cpp` | `all_cores` | 0 `block_size_is_one`, 1 `block_size`, 2 `writer_num_blocks_per_tensor`, 3 `num_c_per_block`, 4 `out_w_tiles`, 5 `out_h_tiles`, 6 `out_HtWt`, 7.. `TensorAccessorArgs(q/k/v_buffer)` chained | — | 0 `q_buffer`, 1 `k_buffer`, 2 `v_buffer` (`Buffer*`), 3 `out_tensor_tile_id`, 4 `out_tensor_tile_id_with_transpose` | — | — | O2 (unset, DM) | `WriterConfigDescriptor{}` |
| compute | `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (shared pool) | `all_cores` | 0 `num_tiles_per_tensor` | — | — | — | — | O3 (unset, compute) | `ComputeConfigDescriptor{}` (all defaults: HiFi4, no fp32 dest, `dst_full_sync_en = false`, `math_approx_mode = false`, `bfp8_pack_precise = false`, empty `unpack_to_dest_mode`) |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | role |
|---|---|---|---|---|---|---|
| 0 | `2 * num_tiles_per_tensor * tile` | `all_cores` | input dtype | `single_tile_size` | — | K heads pre-transpose (reader → compute) |
| 1 | `4 * num_tiles_per_tensor * tile` | `all_cores` | input dtype | `single_tile_size` | — | Q and V heads (reader → writer) |
| 16 | `num_tiles_per_tensor * tile` | `all_cores` | input dtype | `single_tile_size` | — | K heads transposed (compute → writer) |

#### Semaphores
none

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:99,187` | input `a` | reader RTA 0 |
| `program_factory.cpp:111,193` | `output[0]` (q) | writer RTA 0 |
| `program_factory.cpp:112,194` | `output[1]` (k) | writer RTA 1 |
| `program_factory.cpp:113,195` | `output[2]` (v) | writer RTA 2 |

#### Work split
n/a. There is no `split_work_to_cores`: one core per (seq-tile column, batch row), a rectangle of `ashape[2]/32` × `ashape[0]` cores starting at (0,0). Every core runs the same kernels with the same CTAs, so there is no group multiplicity.

### Variant: Sharded

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_tm_tile_layout_create_qkv_heads_sharded.cpp` | shard grid `all_cores` | 0 `num_heads_per_tensor`, 1 `block_ht`, 2 `block_wt`, 3 `out_block_wt`, 4 `block_wt * tile`, 5 `out_block_wt * tile`, 6 `num_tiles_per_tensor`, 7 `block_wt * tile / 3` | — | — | — | — | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| writer | `kernels/dataflow/writer_tm_tile_layout_create_qkv_heads_sharded.cpp` | shard grid | same 8 as reader | — | — | — | — | O2 (unset, DM) | `WriterConfigDescriptor{}` |
| compute | `kernels/compute/transpose_wh_sharded.cpp` (lent: also bound by `create_qkv_heads`) | shard grid | 0 `num_tiles_per_tensor` | — | — | — | — | O3 (unset, compute) | `ComputeConfigDescriptor{}` (all defaults) |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile | borrowed `.buffer` | role |
|---|---|---|---|---|---|---|---|
| 0 | `block_wt * block_ht * tile` | shard grid | input dtype | `single_tile_size` | — | `a.buffer()` | input shard |
| 24 | `2 * block_ht * tile` | shard grid | input dtype | `single_tile_size` | — | — | K intermediate (writer → compute) |
| 16 | `out_block_wt * out_block_ht * tile` | shard grid | input dtype | `single_tile_size` | — | `output[0].buffer()` | Q shard |
| 17 | same | shard grid | input dtype | `single_tile_size` | — | `output[1].buffer()` | K shard |
| 18 | same | shard grid | input dtype | `single_tile_size` | — | `output[2].buffer()` | V shard |

#### Semaphores
none

#### Tensor accessors
none. All four tensors reach the kernels only through borrowed CBs, and there are no RTAs at all.

#### Work split
n/a. One shard per core over the shard grid. All cores run identical kernels and CTAs.

### Shared kernels
- **`ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`** (interleaved compute; borrowed from the shared pool). A `_metal2` fork exists beside it: `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`. **Rung 1, reuse.** Fork vocabulary: `dfb::in` (consumed), `dfb::out` (produced), named CTA `NHtWt`; no `#ifdef`s, no tensors. Legacy binders left on the original: `nlp_create_qkv_heads_boltz`. (`nlp_create_qkv_heads` and `nlp_create_qkv_heads_vit` already bind the fork. The brief's sunset list still names `vit`, but it was ported on this branch after the audit's source survey.)
- **`kernels/compute/transpose_wh_sharded.cpp`** (sharded compute; lent, since `experimental/transformer/create_qkv_heads` also binds it). A `_metal2` fork exists beside it: `kernels/compute/transpose_wh_sharded_metal2.cpp`. **Rung 1, reuse.** Fork vocabulary: `dfb::in` (consumed), `dfb::out` (produced, packer target), named CTA `num_tiles`; no `#ifdef`s. It is already bound by `create_qkv_heads_from_separate_tensors`. Legacy binder left on the original: `create_qkv_heads`.
- The four dataflow kernels are each bound by exactly one factory of this op and by no other op (census: `grep -rl <filename> ttnn/cpp/ttnn/operations/`, where the only hits are this op's two factories). They are converted **in place**.

### Flags
- `kernels/compute/transpose_wh_sharded_metal2.cpp` exists in this directory but was not bound by this op before the port. This port becomes its second consumer.
- The sharded kernels declare unused `constexpr` CB indices (reader: `cb_im0` = c_24, `cb_out1` = c_17; writer: `cb_out0..2` = c_16..18) that name CBs those kernels never touch. Metal 2.0 has no handle for an unbound DFB, so these dead declarations are dropped along with the CB-index lines they sit beside (see Dropped Plumbing). They carry no behavior.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both factories).
- **Custom `compute_program_hash`**: none (default reflection hash), so nothing to leave intact.
- **Implementation notes**:
  - Method swap inside the two existing structs: `create_descriptor` becomes `create_program_artifacts`, with the same parameter list (`tensor_return_value_t` is `std::vector<Tensor>`). There is no pybound `create_descriptor` to remove.
  - The spec-name constants are function-local, as in the sibling ports. This target is a unity build, and many `ttnn::experimental::prim` factories in it already define `READER` / `WRITER` / `INPUT`. File-scope anonymous-namespace constants could collide within a unity batch.
  - The `Buffer*` locals and the `TT_ASSERT`s on them stay verbatim. The `MeshTensor` is extracted only for `TensorParameter` / `TensorArgument`, as `nlp_create_qkv_heads` did.

## Planned Spec Shape

### Variant: Interleaved
- KernelSpecs: `reader`, `writer`, `compute` (one per legacy descriptor), all over `all_cores`.
- DataflowBufferSpecs (1:1 with the legacy CBs):
  - `k_pre_transpose` ← c_0. Reader PRODUCER (`dfb::k`), compute CONSUMER (`dfb::in`).
  - `qv` ← c_1. Reader PRODUCER (`dfb::qv`), writer CONSUMER (`dfb::qv`).
  - `k_transposed` ← c_16. Compute PRODUCER (`dfb::out`), writer CONSUMER (`dfb::k`).
  - The entry is one tile of the input data format; `num_entries` is copied from the legacy `total_size / single_tile_size`. No `tile` field was set.
- SemaphoreSpecs: none.
- TensorParameters: `input` (bound to the reader as `tensor::input`); `q`, `k`, `v` (bound to the writer as `tensor::q`, `tensor::k`, `tensor::v`).
- WorkUnitSpecs: one, `{reader, writer, compute}` over `all_cores`.
- Run args: per core (legacy loop kept, via `AddRuntimeArgsForNode`): reader `in0_tensor_tile_id`; writer `out_tensor_tile_id`, `out_tensor_tile_id_with_transpose`. `tensor_args` gets all four tensors.

### Variant: Sharded
- KernelSpecs: `reader`, `writer`, `compute`, all over the shard grid.
- DataflowBufferSpecs:
  - `in` ← c_0, `borrowed_from = input`. Two role-free touchers (reader and writer both `get_read_ptr` only), so **1P+1C**: reader PRODUCER, writer CONSUMER, both accessor `in`.
  - `k_pre_transpose` ← c_24. Writer PRODUCER (`dfb::k_pre_transpose`), compute CONSUMER (`dfb::in`).
  - `out_q` ← c_16, `borrowed_from = q`. One toucher (reader), so **self-loop** (reader PRODUCER + CONSUMER, `dfb::out_q`).
  - `out_k` ← c_17, `borrowed_from = k`. One toucher (compute packer), so **self-loop** (compute PRODUCER + CONSUMER, `dfb::out`).
  - `out_v` ← c_18, `borrowed_from = v`. One toucher (reader), so **self-loop** (reader PRODUCER + CONSUMER, `dfb::out_v`).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `q`, `k`, `v`. None is bound to a kernel; they back the borrowed DFBs.
- WorkUnitSpecs: one, `{reader, writer, compute}` over the shard grid.
- Run args: no kernel run args (legacy had no RTAs). `tensor_args` gets all four tensors, which the framework refreshes on a cache hit (it refreshed the borrowed CB addresses before).

## Preserved Multiplicity

none. Neither factory has a work-split multiplicity: one `KernelDescriptor` per role, each over the factory's single core range.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `program_factory.cpp:187`, reader RTA 0 | `in0_buffer` (`Buffer*`) | `TensorBinding{input → "input"}` on reader |
| `program_factory.cpp:99`, reader CTA 2.. | `TensorAccessorArgs(in0_buffer).append_to(...)` / kernel `TensorAccessorArgs<2>()` | binding token `tensor::input` |
| `program_factory.cpp:193-195`, writer RTAs 0-2 | `q_buffer`, `k_buffer`, `v_buffer` (`Buffer*`) | `TensorBinding{q/k/v → "q"/"k"/"v"}` on writer |
| `program_factory.cpp:111-113`, writer CTA 7.. | chained `TensorAccessorArgs(q/k/v_buffer)` / kernel `TensorAccessorArgs<7>()` + `next_compile_time_args_offset()` | binding tokens `tensor::q`, `tensor::k`, `tensor::v` |
| reader kernel `:25-26` | hard-coded `cb_id_in0 = 0`, `cb_id_in1 = 1` | `dfb::k`, `dfb::qv` |
| writer kernel `:36-37` | hard-coded `cb_id_out0 = 16`, `cb_id_out1 = 1` | `dfb::k`, `dfb::qv` |
| `transpose_wh.cpp` (legacy) | hard-coded `c_0` / `c_16` | fork's `dfb::in` / `dfb::out` |
| sharded reader `:24-28`, writer `:24-28` | `constexpr uint32_t cb_* = tt::CBIndex::c_*` (five each; the reader's `cb_im0`/`cb_out1` and the writer's `cb_out0..2` were already unused) | used ones become `dfb::in` / `dfb::out_q` / `dfb::out_v` / `dfb::k_pre_transpose`; unused ones dropped |
| `transpose_wh_sharded.cpp` (legacy) | hard-coded `c_24` / `c_17` | fork's `dfb::in` / `dfb::out` |
| `sharded_program_factory.cpp:342,363,374,385` | `CBDescriptor::buffer = <tensor>.buffer()` | `DataflowBufferSpec::borrowed_from` + `TensorArgument` |
| all positional CTAs (interleaved reader 2, writer 7, compute 1; sharded reader 8, writer 8, compute 1) | `get_compile_time_arg_val(i)` | named CTAs with the kernel's existing local names (`block_size`, `out_num_blocks_per_tensor`, `block_size_is_one`, `out_num_c_per_block`, `out_w_tiles`, `out_h_tiles`, `out_HtWt`, `num_heads_per_tensor`, `block_ht`, `block_wt`, `out_block_wt`, `block_wt_size_bytes`, `out_block_wt_size_bytes`, `num_tiles_per_tensor`, `tensor_stride_size_bytes`; compute forks `NHtWt` / `num_tiles`) |
| remaining positional RTAs | `get_arg_val<uint32_t>(i)` | named RTAs `in0_tensor_tile_id`, `out_tensor_tile_id`, `out_tensor_tile_id_with_transpose` |

No page-size third arguments, semaphores, or varargs.

## Applied Patterns

- [Two-toucher DFB → assign 1P+1C](../shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split): sharded `in` (input shard), reader PRODUCER and writer CONSUMER. Both are role-free raw `get_read_ptr` touchers, so the labels are cosmetic.
- [Sync-free and single-ended CBs → self-loop DFB](../shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb): sharded `out_q` / `out_v` (DM reader) and `out_k` (compute packer).
- Borrowed-memory DFBs (`borrowed_from`): sharded `in`, `out_q`, `out_k`, `out_v`.
- [Porting a shared kernel](../shared/port_patterns.md#caution-porting-a-shared-kernel), rung 1 (reuse) for both compute kernels.
- [Pass DFB handles directly to LLKs](../shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): already inside the reused forks.
- Runtime DFB selection in the interleaved reader: legacy rebuilt `CircularBuffer cb_cur(cb_id)` from a runtime id each iteration. The port keeps one `DataflowBuffer` object per DFB (`dfb_in0` ← `dfb::k`, `dfb_in1` ← `dfb::qv`) and selects through a `DataflowBuffer*` in the same `if` / `else if`. No id extraction, and no second object per DFB (see the Same-FIFO aliasing "Don't" list).

## Deferred / Flagged

- Hardware config: DM descriptors were `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`, so they map to `ttnn::create_reader_datamovement_config()` / `create_writer_datamovement_config()`. The compute descriptors were default `ComputeConfigDescriptor{}` (Style B), so they map to `ComputeHardwareConfig{}`, whose defaults match field by field. There are no `unpack_modes`: inputs are BF16 / BFP8_B only (`validate_on_program_cache_miss`), never Float32, and `enable_32_bit_dest` is off. Both compute KernelSpecs get an explicit `O3`.
- Verification on this 8×8 Wormhole: `tests/tt_eager/.../test_split_query_key_value_and_split_heads.py` covers interleaved. No committed sharded test fits the card: the BERT-large sharded test hard-codes a (12, 8) grid. A scratch copy at grid (8, 8) is used instead (still 16 heads / 8 M-cores = 2 heads per core, matching the factory's hard-coded 2), plus scratch cache-hit loops for both factories.
- New findings during planning: none that block. The brief's sunset list for `transpose_wh.cpp` is stale (`vit` already ported).
