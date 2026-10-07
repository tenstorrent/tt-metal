# Port Plan — nlp_create_qkv_heads_vit

Port plan for `experimental/transformer/nlp_create_qkv_heads_vit`, ported from the ProgramDescriptor API (direct `create_descriptor`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `NlpCreateHeadsVitDeviceOperation` (`device/nlp_create_qkv_heads_vit_device_operation.hpp:21-27`), with the body in `device/nlp_create_qkv_heads_vit_program_factory.cpp:19-236`. There is no `program_factory_t`, so `ttnn_factory.md` exception 3 applies. No `program_factory_t` has appeared since the audit.
- Variants: single factory. It contains one compile-time-dead configuration, gated by `const bool transpose_k_heads = false;` (`program_factory.cpp:98`).
- Custom `compute_program_hash`: none. The default reflection-based hash is used, with no `attribute_values` / `to_hash` backdoor.

### Kernels

Live configuration (`transpose_k_heads == false`):

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs (per node) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `…/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads.cpp` | `all_cores` | 0 `q_num_tiles`, 1 `kv_num_tiles`, 2.. `TensorAccessorArgs(in0_buffer)`, then an empty `TensorAccessorArgs()` (`:80-85`) | none | 0 `in0_buffer` (`Buffer*`), 1 `in1_buffer_addr` (=0), 2 `num_blocks`, 3 `in0_tensor_tile_id` = `num_blocks_written * per_tensor_tiles`, 4 `in1_tensor_tile_id` (=0) (`:198-206`) | none | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / DEDICATED |
| writer | `…/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads.cpp` | `all_cores` | 0 `q_out_h_tiles`, 1 `q_out_w_tiles` (=2), 2 `q_out_HtWt`, 3 `q_out_c` (=`num_q_heads`=12), 4 `kv_out_c` (=`num_kv_heads`=12), then `TensorAccessorArgs` for q, k, v (`:86-95`) | none | 0 `q_buffer`, 1 `k_buffer`, 2 `v_buffer` (`Buffer*`), 3 `num_blocks`, 4 `q_out_h_dim`, 5 `q_out_tensor_tile_id`, 6 `k_out_tensor_tile_id`, 7 `v_out_tensor_tile_id` (`:217-228`) | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / DEDICATED |

The dead configuration (`transpose_k_heads == true`, unreachable) adds:

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| compute group 1 | `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` | `core_group_1` | 0 `num_blocks_per_core_group_1 * kv_num_tiles` | none | none | O3 (unset, compute) | `ComputeConfigDescriptor{}` (all defaults) |
| compute group 2 (only if `core_group_2` is non-empty) | same | `core_group_2` | 0 `num_blocks_per_core_group_2 * kv_num_tiles` | none | none | O3 (unset, compute) | `ComputeConfigDescriptor{}` |

It also adds `TRANSPOSE_K_HEADS=1` to the reader and writer defines, and uses the alternate `k_out_tensor_tile_id` formula (`program_factory.cpp:213-215`).

`grep -n opt_level` on the factory finds nothing, so every level is the resolved default.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile | config |
|---|---|---|---|---|---|---|
| 1 | `2 * per_tensor_tiles * single_tile_size` (144 tiles) | `all_cores` | `datatype_to_dataformat_converter(a.dtype())` | `single_tile_size` | not set | always |
| 0 | same | `all_cores` | same | `single_tile_size` | not set | dead (`transpose_k_heads`) |
| 16 | same | `all_cores` | same | `single_tile_size` | not set | dead (`transpose_k_heads`) |

No GlobalCircularBuffer, no `.buffer`, no `address_offset`.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:84`; reader `:26,39` | `input_tensor` | reader RTA 0 (`in0_buffer`) |
| `program_factory.cpp:85`; reader `:27-29,41-43` | none: an empty placeholder whose kernel side compiles only under the never-defined `READ_FROM_INPUT_TENSOR_KV` | reader RTA 1 (dummy `0`) |
| `program_factory.cpp:93`; writer `:32,42` | `output[0]` (q) | writer RTA 0 |
| `program_factory.cpp:94`; writer `:33,43` | `output[1]` (k) | writer RTA 1 |
| `program_factory.cpp:95`; writer `:34,44` | `output[2]` (v) | writer RTA 2 |

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)`, with `num_blocks = B * 1 * S / TILE_HEIGHT`.
- num_cores / all_cores / core_group_1 (`num_blocks_per_core_group_1`) / core_group_2 (`num_blocks_per_core_group_2`).
- The per-core RTA loop runs `i < num_cores` over column-major `CoreCoord{i / num_cores_y, i % num_cores_y}` with a running `num_blocks_written`. Kept verbatim.

### Shared kernels
- Reader and writer: op-private. Other ops have files with the same names, but they are different files. Converted in place.
- `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (dead path only): a shared compute-pool kernel. A `_metal2` fork already exists at `ttnn/cpp/ttnn/kernel/compute/transpose_wh_metal2.cpp`. **Rung: reuse.** Fork vocabulary: `dfb::in` (CONSUMER), `dfb::out` (PRODUCER), named CTA `NHtWt`. It has no `#ifdef`s and no tensor bindings.

### Flags
- `transpose_k_heads` is a hard-coded `const false`, so the whole branch is unreachable and untestable.
- This factory never defines `READ_FROM_INPUT_TENSOR_KV`, so the in1 accessor and RTAs are dead plumbing.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: this is a direct-descriptor op, so exception 3 applies.
  - Nest `NlpCreateHeadsVitDeviceOperation::NlpCreateQkvHeadsVitProgramFactory` with `create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`.
  - Add `using program_factory_t = std::variant<NlpCreateQkvHeadsVitProgramFactory>;`.
  - Remove the device-op-level `create_descriptor`. Its cache-hit comment moves onto the factory struct.
  - There is no pybind entry point to remove.

## Planned Spec Shape

- **KernelSpecs**:
  - `reader`: reader source, `ttnn::create_reader_datamovement_config()`, O2 default. Defines `{TRANSPOSE_K_HEADS=1}` only when `transpose_k_heads`.
  - `writer`: writer source, `ttnn::create_writer_datamovement_config()`, O2 default, with the same conditional define.
  - (dead) `compute_g1`, and `compute_g2` only when `core_group_2` is non-empty:
    - Source `transpose_wh_metal2.cpp`, explicit `opt_level = O3`, named CTA `NHtWt`.
    - `ComputeHardwareConfig{}` (Style B). The legacy `ComputeConfigDescriptor{}` defaults are HiFi4, Precise, no fp32 dest and `dst_full_sync_en = false`; the Metal 2.0 defaults match, including `double_buffer_dest = true`.
    - No `unpack_modes`: `enable_32_bit_dest` is false, so the Float32 entry rule doesn't fire.
- **DataflowBufferSpecs**:
  - `qv` (legacy CB 1): `entry_size = single_tile_size`, `num_entries = 2 * per_tensor_tiles`, `data_format_metadata = data_format`. Always present. Reader PRODUCER (accessor `qv`), writer CONSUMER (accessor `qv`).
  - (dead) `k_in` (legacy CB 0): same sizing, `transpose_k_heads` only. Reader PRODUCER (`k`), compute CONSUMER (`in`).
  - (dead) `k_out` (legacy CB 16): same sizing, `transpose_k_heads` only. Compute PRODUCER (`out`), writer CONSUMER (`k`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `q`, `k`, `v`, each from its `MeshTensor::tensor_spec()`. The reader binds `input` (accessor `input`); the writer binds `q`, `k` and `v` (accessors `q`, `k`, `v`).
- **WorkUnitSpecs**: one per work-split core group, `core_group_1` and (if non-empty) `core_group_2`. Each runs the reader and writer, plus that group's compute instance on the dead path. Together they cover exactly `all_cores`, the legacy reader/writer core ranges, and the split lets each per-group compute spec target only its own group, as the legacy descriptor did.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_group_1`, `compute_desc_group_2` (`transpose_wh.cpp`, dead path) | `compute_g1`, `compute_g2` (`transpose_wh_metal2.cpp`), per-group CTA `NHtWt` | `core_group_1`, `core_group_2` | `k_in` CONSUMER and `k_out` PRODUCER on each. The node sets are disjoint, so each binds its role legally and no flag is needed. |

The reader and writer have no work-split multiplicity (one descriptor each, over `all_cores`).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:201`, reader RTA 0 (`:16`) | `in0_buffer` (`Buffer*`) | `TensorBinding{input, "input"}`; reader `TensorAccessor(tensor::input)` |
| factory `:84`, reader `:26` | `TensorAccessorArgs(in0_buffer).append_to` / `TensorAccessorArgs<2>()` | binding mechanism |
| factory `:202`, reader RTA 1 (`:17`) | `in1_buffer_addr`, a dummy address (`0`) | **dropped**. It is the base address of the never-compiled in1 accessor, which becomes `TensorAccessor(tensor::input_kv)` under `READ_FROM_INPUT_TENSOR_KV`. There is no host binding, since the macro is never defined. |
| factory `:85`, reader `:28` | empty `TensorAccessorArgs()` placeholder / `TensorAccessorArgs<in0_args.next…>()` | dropped (as above) |
| factory `:220-222`, writer RTAs 0-2 (`:17-19`) | `q_buffer`, `k_buffer`, `v_buffer` (`Buffer*`) | `TensorBinding`s `q`, `k`, `v`; writer `TensorAccessor(tensor::q/k/v)` |
| factory `:93-95`, writer `:32-34` | three `TensorAccessorArgs(...).append_to` / `TensorAccessorArgs<5>()` chain | binding mechanism |
| reader `:31-36` | `cb_id_qv = 1`, `cb_id_k = 1` or `0` (hard-coded CB indices) | `dfb::qv`; `dfb::k` under `TRANSPOSE_K_HEADS` |
| writer `:36-41` | `cb_id_qv = 1`, `cb_id_k = 1` or `16` | `dfb::qv`; `dfb::k` under `TRANSPOSE_K_HEADS` |
| reader CTAs 0-1 | positional `q_num_tiles`, `kv_num_tiles` | named CTAs, same names |
| writer CTAs 0-4 | positional | named CTAs `q_out_h_tiles`, `q_out_w_tiles`, `q_out_HtWt`, `q_out_c`, `kv_out_c` |
| reader RTAs 2-4 | positional | named RTAs `num_blocks`, `in0_tensor_tile_id`, `in1_tensor_tile_id` (=0; carried because the kernel reads it unconditionally) |
| writer RTAs 3-7 | positional | named RTAs `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id`, `k_out_tensor_tile_id`, `v_out_tensor_tile_id` |
| dead compute CTA 0 | positional `NHtWt` | named CTA `NHtWt` (fork interface) |

## Applied Patterns

- **Conditional / optional resource bindings**: the `k_in` / `k_out` DFBs and the compute specs exist only when `transpose_k_heads`. The reader and writer bind `dfb::k` only under `TRANSPOSE_K_HEADS`, which is carried via `compiler_options.defines`. The kernel-side `#ifdef` structure is kept.
- **Caution: Porting a shared kernel**: reuse the existing `transpose_wh_metal2.cpp` fork.
- **Pass DFB handles to the fork as-is**: `dfb::in` / `dfb::out` are the fork's accessor names.
- **`AddRuntimeArgsForNode`** keeps the legacy node-first RTA loop.
- **Rule 7**: the non-`constexpr` `get_tile_size(cb_id_*)` locals become `dfb_*.get_tile_size()`.

## Deferred / Flagged

- **Dead-branch decision (audit Question 1):** carried faithfully. The user decided this on 2026-10-07 ("Keep the behaviour unchanged, stick with the default"). Recorded in the report.
- **Brief deviation (in1 dummy address):** the brief says to carry `in1_tensor_addr` as a named `0` arg. This plan drops it instead, as a buffer-address RTA (recipe rule 5 / Dropped Plumbing). That matches how the already-ported `nlp_create_qkv_heads` reader treats the same lineage. Zero functional change: no compiled configuration ever reads the value.
- **Two wrappers, one DFB:** on the live path both kernels construct `dfb_qv` and `dfb_k` as two `DataflowBuffer` objects from the same `dfb::qv` token. This mirrors the legacy pair of `CircularBuffer`s on index 1, per the brief. A `DataflowBuffer` holds only its id and a reference to the shared local interface (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h`), so two wrappers behave exactly like one. The sibling `nlp_create_qkv_heads` port used a `DataflowBuffer&` alias instead; both forms are equivalent on WH/BH.
