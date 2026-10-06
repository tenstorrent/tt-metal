# Port Plan — create_qkv_heads_from_separate_tensors

Port plan for `experimental/transformer/create_qkv_heads_from_separate_tensors`, ported from the `ProgramDescriptor` API (direct-descriptor shape) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, reached through the **direct-descriptor shim**: `create_descriptor` is a static member of `CreateQKVHeadsSeparateTensorsDeviceOperation` itself, with no `program_factory_t` (`device/create_qkv_heads_from_separate_tensors_device_operation.hpp:26-29`). Body in `device/create_qkv_heads_from_separate_tensors_program_factory.cpp:16-183`. Re-checked at port time (branch tip `b77c495b871`): still no `program_factory_t`, so ttnn_factory exception 3 applies.
- Variants: single factory, one config switch `transpose_k_heads` (Python default `true`).
- Custom `compute_program_hash`: none — default reflection-based hash. No `attribute_values` / `to_hash` either.
- `override_runtime_arguments`: none.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `create_qkv_heads_from_separate_tensors/device/kernels/reader_create_qkv_heads_sharded_separate.cpp` (own) | `all_cores` (Q shard grid) | 0 `q_shard_ht`, 1 `q_shard_wt`, 2 `k_shard_ht`, 3 `k_shard_wt`, 4 `q_heads_per_core`, 5 `k_heads_per_core`, 6 `head_dim / TILE_WIDTH` (tiles per head) — host emission order, `program_factory.cpp:46-54` | none | none | none | `TRANSPOSE_K_HEADS=1` iff `transpose_k` | O2 (unset, DM default) | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / dedicated |
| compute (only if `transpose_k`) | `split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp` (**borrowed**) | `all_cores` | 0 `per_core_k_tiles` | none | none | none | none | O3 (unset, compute default) | `ComputeConfigDescriptor{.fp32_dest_acc_en = kv dtype == FLOAT32}`, everything else default (HiFi4, approx false, dst_full_sync false, bfp8_pack_precise false, unpack_to_dest_mode empty) |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile | backing |
|---|---|---|---|---|---|---|
| c_0 | `q_size` = per_core_q_tiles · tile | all_cores | q fmt | tile size (of **q** fmt) | unset | `input_tensor_q.buffer()` |
| c_1 | `kv_size` = 2 · k_size | all_cores | kv fmt | tile size (q fmt) | unset | `input_tensor_kv.buffer()` |
| c_16 | `q_size` | all_cores | q fmt | tile size (q fmt) | unset | `output_q.buffer()` |
| c_17 | `k_size` = per_core_k_tiles · tile | all_cores | kv fmt | tile size (q fmt) | unset | `output_k.buffer()` |
| c_18 | `v_size` = k_size | all_cores | kv fmt | tile size (q fmt) | unset | `output_v.buffer()` |
| c_24 (only if `transpose_k`) | `k_size` | all_cores | kv fmt | tile size (q fmt) | unset | none (op-allocated L1) |

`single_tile_size` is computed once from the **Q** data format and used as the page size for every CB, including the KV-format ones. Carried verbatim.

### Semaphores
none

### Tensor accessors
none — no `TensorAccessor` on either side. Every tensor reaches the kernels through a borrowed CB (`.buffer = <tensor>.buffer()`). No address RTAs.

### Work split
n/a — no `split_work_to_cores`; every core in the Q shard grid runs the same kernels with the same CTAs.

### Shared kernels
- **`split_query_key_value_and_split_heads/device/kernels/compute/transpose_wh_sharded.cpp`** — *borrowed*. Census (`grep -rl transpose_wh_sharded ttnn/cpp/ttnn/operations/`, disambiguated): bound by this op, `split_query_key_value_and_split_heads_sharded_program_factory.cpp`, and `create_qkv_heads_program_factory.cpp`. The `data_movement/transpose` hits bind that op's own same-stem kernels (`data_movement/transpose/.../transpose_wh_sharded_metal2.cpp`), a different kernel — not a consumer and not a fork of this one. No `_metal2` sibling in the original's directory → **rung 2: create `transpose_wh_sharded_metal2.cpp` beside it, plus the pointer comment in the original.**
- `reader_create_qkv_heads_sharded_separate.cpp` — only bound by this op's factory. Converted in place.

### Flags
none — no unreferenced kernel files in the op directory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: direct-descriptor op → ttnn_factory exception 3. Nest `CreateQKVHeadsSeparateTensorsProgramFactory` (the pre-#57409 name) with `create_program_artifacts` in the device-op struct, add `using program_factory_t = std::variant<CreateQKVHeadsSeparateTensorsProgramFactory>;`, remove `create_descriptor`. Body stays in `device/create_qkv_heads_from_separate_tensors_program_factory.cpp`. No pybind change (`create_descriptor` was never pybound).

## Planned Spec Shape

- **TensorParameters** (5): `input_q`, `input_kv`, `output_q`, `output_k`, `output_v` — each from its tensor's `tensor_spec()`, strict matching (relaxation `none`). No `TensorBinding`s: no kernel constructs a `TensorAccessor`; the parameters exist to back the borrowed DFBs.
- **DataflowBufferSpecs** (5 + 1 conditional), `entry_size = single_tile_size`, `num_entries = total_size / single_tile_size` (i.e. per-core tile counts), `data_format_metadata` = legacy format, `tile_format_metadata` unset (legacy unset):
  - `in_q` (borrowed_from `input_q`, q_tiles entries, q fmt)
  - `in_kv` (borrowed_from `input_kv`, 2·k_tiles, kv fmt)
  - `out_q` (borrowed_from `output_q`, q_tiles, q fmt)
  - `out_k` (borrowed_from `output_k`, k_tiles, kv fmt)
  - `out_v` (borrowed_from `output_v`, k_tiles, kv fmt)
  - `k_pre_transpose` (only if `transpose_k`; not borrowed, k_tiles, kv fmt)
- **KernelSpecs**:
  - `reader`: source unchanged path; named CTAs (below); `TRANSPOSE_K_HEADS=1` define iff `transpose_k`; `hw_config = ttnn::create_reader_datamovement_config()`; no `opt_level` (O2). DFB bindings:
    - `in_q`, `in_kv`, `out_q`, `out_v`: self-loop (PRODUCER + CONSUMER, shared accessor name).
    - `!transpose_k`: `out_k` self-loop (accessor `out_k`).
    - `transpose_k`: `k_pre_transpose` PRODUCER only (accessor `k_pre_transpose`).
  - `compute` (only if `transpose_k`): source = new fork `transpose_wh_sharded_metal2.cpp`; CTA `num_tiles = per_core_k_tiles`; `hw_config = ComputeHardwareConfig{.enable_32_bit_dest = kv dtype == FLOAT32}` (Style B), with `unpack_modes = {{k_pre_transpose, UnpackToSrc}}` only when that flag is on (legacy `unpack_to_dest_mode` empty ⇒ Default ⇒ UnpackToSrc; required by the validator for a Float32 consumed DFB under 32-bit Dest — the kv format is Float32 exactly when the flag is on); `compiler_options.opt_level = O3`. DFB bindings: `k_pre_transpose` CONSUMER (accessor `in`), `out_k` self-loop (accessor `out`).
- **SemaphoreSpecs**: none.
- **WorkUnitSpecs**: one, `create_qkv_heads_from_separate_tensors`, target `all_cores`, kernels `{reader}` or `{reader, compute}`.
- **ProgramRunArgs**: `tensor_args` for all five parameters; no `kernel_run_args` (no RTAs/CRTAs on either kernel).
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one reader descriptor, one optional compute descriptor, one node set).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader.cpp:15-21 CTA slots 0–6 | positional `get_compile_time_arg_val(0..6)` | named CTAs `q_shard_ht`, `q_shard_wt`, `k_shard_ht`, `k_shard_wt`, `q_num_heads_per_core`, `k_num_heads_per_core`, `tiles_per_head` (kernel local names) |
| reader.cpp:23-31 | hardcoded `tt::CBIndex::c_0/c_1/c_16/c_17/c_24/c_18` | `dfb::in_q`, `dfb::in_kv`, `dfb::out_q`, `dfb::out_k` / `dfb::k_pre_transpose` (by `#ifdef`), `dfb::out_v` |
| transpose_wh_sharded.cpp:12 CTA slot 0 | positional `get_compile_time_arg_val(0)` | named CTA `num_tiles` (in the fork) |
| transpose_wh_sharded.cpp:14-18 | hardcoded `c_24` / `c_17` | `dfb::in` / `dfb::out` (in the fork) |
| program_factory.cpp:108-168 | `CBDescriptor.buffer = <tensor>.buffer()` (5×) | `TensorParameter` + `DataflowBufferSpec.borrowed_from` + `TensorArgument` |

No buffer-address RTAs, no `TensorAccessorArgs`, no page-size args, no semaphore ids exist to drop.

## Applied Patterns

- Sync-free and single-ended CBs → self-loop DFB: `in_q`, `in_kv` (reader, sync-free raw read-ptr peeks), `out_q`, `out_v` (reader, single-ended producer into resident output shard), `out_k` (reader when `!transpose_k`; compute when `transpose_k`).
- Plain 1:1: `k_pre_transpose` — reader PRODUCER, compute CONSUMER.
- Conditional / optional resource bindings: the reader's K destination is chosen by the existing `TRANSPOSE_K_HEADS` define; each `#ifdef` branch names its own token, and the host binds `out_k` or `k_pre_transpose` on the reader under the same condition. `k_pre_transpose` spec and the compute kernel are conditional exactly as the legacy CB/descriptor were.
- Caution: Porting a shared kernel — rung 2 fork of `transpose_wh_sharded.cpp` with kernel-role names (`dfb::in`, `dfb::out`, `num_tiles`).
- Whitelist rule 7 `constexpr` exception: `constexpr uint32_t single_tile_size_bytes = get_tile_size(dfb_inq);` keeps the free-function token form.
- ttnn_factory exception 3: direct-descriptor → nested program factory.

## Deferred / Flagged

- Recipe/header drift: the recipe's `DataMovementGen1Config` / `ComputeGen1Config` / `create_reader_datamovement_config(device->arch())` spellings don't match the tree. The tree's `ComputeHardwareConfig` is a flat struct (common fields + `config_1xx`), and the TTNN DM helper takes no arch. Values map identically; noted for the report.
- No new structural findings.
