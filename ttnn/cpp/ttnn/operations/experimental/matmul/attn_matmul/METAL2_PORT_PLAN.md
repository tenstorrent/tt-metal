# Port Plan — experimental/matmul/attn_matmul

Port plan for `ttnn/cpp/ttnn/operations/experimental/matmul/attn_matmul`, ported from the `ProgramDescriptor`
API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.
Line references are against `origin/main` @ `fd25f0f4aac` (the pre-port revision).

## Legacy Inventory

*Filled in during the inventory step.*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `AttnMatmulProgramFactory::create_descriptor` returns a
  `ProgramDescriptor` (`device/attn_matmul_program_factory.cpp:19`); no `override_runtime_arguments`.
  The factory lives in the `program_factory_t` variant (`device/attn_matmul_device_operation.hpp:23`),
  so no direct-descriptor conversion is needed.
- Variants: single.
- Custom `compute_program_hash`: none — default reflection-based hash (the former custom hash was removed by
  PR #47232).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory
analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_transformer_attn_matmul.cpp` (own) | `all_device_cores` = full device compute grid (`:75-76`) | `[0] transpose_hw_bool`, `[1] fp32_dest_acc_en && in0 == Float32`, then `TensorAccessorArgs(*src0_buffer)` tail, then `TensorAccessorArgs(*src1_buffer)` tail (`:172-176`) | none | per core (`:234-248`): `[0] src0_buffer (Buffer*)`, `[1] src1_buffer (Buffer*)`, `[2] Mt`, `[3] Kt`, `[4] Nt`, `[5] MtKt`, `[6] in1_KtNt_skip`, `[7] in1_KtNt_stride * 32`, `[8] num_output_blocks_per_core`, `[9] num_blocks_written * MtKt` (itileA_start), `[10] 0` (itileB_start); idle cores: eleven zeros (`:228`) | none | none | O2 (resolved; field absent) | `ReaderConfigDescriptor{}` (`:188`) → RISCV_1 / NOC_0 / DM_DEDICATED_NOC |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed**) | `all_device_cores` | `[0] output_cb_index (c_5)`, then `TensorAccessorArgs(*dst_buffer)` tail (`:178-179`) | none | per core (`:257-263`): `[0] dst_buffer (Buffer*)`, `[1] num_output_blocks_per_core * MtNt` (num_pages), `[2] num_blocks_written * MtNt` (start_id); idle cores: three zeros (`:230`) | none | none | O2 (resolved; field absent) | `WriterConfigDescriptor{}` (`:196`) → RISCV_0 / NOC_1 / DM_DEDICATED_NOC |
| compute | `device/kernels/compute/transformer_attn_matmul.cpp` (own) | `all_device_cores` | `[0] transpose_hw_bool` (`:198-200`) | none | per core (`:249-256`): `[0] 1` (batch), `[1] 1` (Mt), `[2] Kt`, `[3] num_output_blocks_per_core * MtNt` (Nt); idle cores: four zeros (`:229`) | none | none | **O3** (resolved; field absent on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en}` (`:208-211`); `math_approx_mode` / `dst_full_sync_en` resolved at `:32-33` but **not** set (descriptor defaults `false` / `false`); `packer_l1_acc` resolved, no counterpart; no `unpack_to_dest_mode`, no `bfp8_pack_precise` |

*(`opt_level` is the legacy kernel's **resolved** level, not what the source literally says — `grep -n opt_level`
on the factory returns nothing, so the DM kernels resolve to `O2` and the compute kernel to `O3`.)*

### CBs
All six are on `all_device_cores` (full device compute grid). No `.buffer` (no borrowed memory), no
`.address_offset`, no `.global_circular_buffer`, no `.tile` on any format descriptor.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (src0 / in0) — `:103-112` | `Kt * in0_single_tile_size` | all | `in0_data_format` (= dtype of `input_tensor_a`) | `in0_single_tile_size` | unset |
| `c_1` (src1 / in1) — `:114-124` | `2 * in1_single_tile_size` | all | `in1_data_format` (= dtype of `input_tensor_b`) | `in1_single_tile_size` | unset |
| `c_2` (intermed0) — `:126-135` | `interm_single_tile_size` | all | `interm_data_format` = `fp32_dest_acc_en && in0 == Float32 ? Float32 : Float16_b` (`:36-39`) | `interm_single_tile_size` | unset |
| `c_3` (intermed1) — `:137-146` | `interm_single_tile_size` | all | `interm_data_format` | `interm_single_tile_size` | unset |
| `c_4` (intermed2) — `:148-157` | `interm_single_tile_size` | all | `interm_data_format` | `interm_single_tile_size` | unset |
| `c_5` (output) — `:159-169` | `2 * output_single_tile_size` | all | `output_data_format` (= dtype of `output`) | `output_single_tile_size` | unset |

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(in0_args, src0_addr)` (`reader_transformer_attn_matmul.cpp:48`; CTA tail at factory `:175`, kernel `TensorAccessorArgs<2>()` `:46`) | `input_tensor_a` (`a`) | reader RTA 0 = `src0_buffer` (`Buffer*`, `:237`) |
| reader `TensorAccessor(in1_args, src1_addr)` (`reader:49`; CTA tail at factory `:176`, kernel `:47`) | `input_tensor_b` (`b`) | reader RTA 1 = `src1_buffer` (`Buffer*`, `:238`) |
| legacy writer `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:42`; CTA tail at factory `:179`, kernel `TensorAccessorArgs<1>()` `:24`) | `output` (`tensor_return_value`) | writer RTA 0 = `dst_buffer` (`Buffer*`, `:260`) |

No accessor passes a 3rd (page-size) argument.

### Work split
- Driver: `split_work_to_cores(operation_attributes.compute_with_storage_grid_size, num_output_blocks_total)`
  (`:64-73`), where `num_output_blocks_total = ashape[1]` (q_heads). Note: the split runs over the **op's**
  grid attribute, while the kernels / CBs / runtime args are placed on the **device** grid.
- num_cores: `num_cores` (unused), `all_cores` (unused)
- core_group_1: `core_group_1`, count_per_core: `num_output_blocks_per_core_group_1`
- core_group_2: `core_group_2`, count_per_core: `num_output_blocks_per_core_group_2`
- Every other device core: `num_output_blocks_per_core = 0`, all-zero runtime args (`:223-232`).
- The per-core loop (`:220-265`) walks `i < total_num_cores` with `core = {i / device_num_cores_y, i % device_num_cores_y}`
  (PR #59789) and accumulates `num_blocks_written` over the working cores in that order.

### Shared kernels
- **Borrowed:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`.
  Census by bound path on `origin/main` (quasar copies and the same-named private copies under
  `data_movement/slice` and `matmul` excluded): `data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp`,
  `eltwise/unary_backward/device/unary_backward_program_factory.cpp`,
  `examples/example/device/multi_core_program_factory.cpp`, `examples/example/device/single_core_program_factory.cpp`,
  and this op. A `_metal2` fork exists beside it: `writer_unary_interleaved_start_id_metal2.cpp` → **rung 1 (reuse)**.
  **Fork vocabulary (the constraint this port's writer `KernelSpec` is built against):** `dfb::out` bound
  CONSUMER (`:29`), `tensor::dst` (`:48`), named RTAs `num_pages`, `start_id` (`:23-24`); optional defines
  `ENTRIES_PER_PAGE` / `PAGE_BYTES` (`:31-39`), `OUT_SHARDED` (`:41`), `BACKWARDS` (`:50`) — **none** of them
  set here (the legacy factory sets no defines; one page per DFB entry, `page_bytes = dfb.get_entry_size()`).
- Lent: none (no other op binds this op's kernels).
- Intra-op: none (single factory).

### Flags
- `attn_matmul_program_factory.hpp:14-18` and `.cpp:99-102` carry comments claiming `padded_shape` is folded into
  `compute_program_hash()`; that custom hash was removed by PR #47232. The comments are not carried into the
  ported factory (the `in0` DFB capacity is keyed per shape by the default reflection hash; the comment's
  substance is restated in the port's own words where the DFB is declared).
- No unreferenced kernel files in the directory.
- No descriptor type outside the audit's scan: only `KernelDescriptor` × 3 and `CBDescriptor` × 6.

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: the factory keeps its two-public-entry-point shape (`attn_matmul` and
  `attn_matmul_from_cache` differ only in attributes) — nothing outside the factory body changes. The
  `tensor_args_t` carries an optional `preallocated_output`; the `OUTPUT` `TensorParameter` / `TensorArgument`
  is built from `tensor_return_value`, which is that tensor when supplied, exactly as the legacy `dst_buffer` was.

## Planned Spec Shape

*Filled in during the planning step.* 1:1 with legacy.

- KernelSpecs: three — `READER` (`reader`), `WRITER` (`writer`, bound to the `_metal2` fork), `COMPUTE`
  (`compute`), one per legacy `KernelDescriptor`.
- DataflowBufferSpecs: six — `IN0` (`in0`), `IN1` (`in1`), `INTERMED0` (`intermed0`), `INTERMED1`
  (`intermed1`), `INTERMED2` (`intermed2`), `OUT` (`out`), one per legacy `CBDescriptor`; `entry_size` /
  `num_entries` / `data_format_metadata` copied from the legacy `page_size` / `total_size ÷ page_size` /
  `data_format`; no `tile_format_metadata` (legacy `.tile` unset), no `borrowed_from`, no advanced options.
- SemaphoreSpecs: none.
- TensorParameters: three — `INPUT_A` (`input_a`, `input_tensor_a.tensor_spec()`), `INPUT_B` (`input_b`,
  `input_tensor_b.tensor_spec()`), `OUTPUT` (`output`, `tensor_return_value.tensor_spec()`); strict matching.
- WorkUnitSpecs: one — `main`, kernels `{READER, WRITER, COMPUTE}`, `target_nodes = all_device_cores` (the full
  device compute grid, as legacy). Idle cores keep running all three kernels with zero-valued run-args, as
  legacy.
- Op-owned tensors: none.

### DFB bindings (endpoint census re-derived from the kernels)

| DFB | reader | compute | writer | disposition |
|---|---|---|---|---|
| `IN0` | PRODUCER (`reserve_back(Kt)` `reader:82`, `push_back(1)` `:87`), accessor `in0` | CONSUMER (`wait_front(kt + 1)` `compute:54`, `pop_front(Kt)` `:82`), accessor `in0` | — | legal 1:1 |
| `IN1` | PRODUCER (`reader:96-99`), accessor `in1` | CONSUMER (`compute:56`, `:60`), accessor `in1` | — | legal 1:1 |
| `INTERMED0` | — | PRODUCER (`reserve_back` / `pack_tile` / `push_back` `compute:64-68`) **and** CONSUMER (`untilize` helper drains it, `:71-77`), accessor `intermed0` | — | **self-loop** on compute |
| `INTERMED1` | CONSUMER (`wait_front(1)` `reader:109`, `pop_front(1)` `:118`, `get_read_ptr()` peek `:58`), accessor `intermed1` | PRODUCER (`untilize` helper packs into it, `:71-77`), accessor `intermed1` | — | legal 1:1 |
| `INTERMED2` | PRODUCER (`reserve_back(1)` `reader:92`, `get_write_ptr()` peek `:59` + NoC loopback write `:110-116`, `push_back(1)` `:124`), accessor `intermed2` | CONSUMER (`tilize` helper drains it, `compute:86`), accessor `intermed2` | — | legal 1:1 |
| `OUT` | — | PRODUCER (`tilize` helper packs into it, `compute:86`), accessor `out` | CONSUMER (fork `wait_front` / `pop_front` `:57`, `:60`), accessor `out` (fork vocabulary) | legal 1:1 |

### Tensor bindings
| TensorParameter | kernel | accessor | kernel-side construction |
|---|---|---|---|
| `INPUT_A` | reader | `src0` | `TensorAccessor(tensor::src0)` (replaces `reader:46-48`) |
| `INPUT_B` | reader | `src1` | `TensorAccessor(tensor::src1)` (replaces `reader:47-49`) |
| `OUTPUT` | writer (fork) | `dst` | `TensorAccessor(tensor::dst)` (fork `:48`) |

### Named arguments
| kernel | CTAs (named) | RTAs (named) |
|---|---|---|
| reader | `transpose_hw`, `fp32_acc_en` | `Mt`, `Kt`, `Nt`, `MtKt`, `in1_KtNt_skip`, `in1_KtNt_mul_32`, `blocks`, `itileA_start`, `itileB_start` (the kernel's own local names, `reader:17-25`) |
| compute | `transpose_hw` | `batch`, `Mt`, `Kt`, `Nt` (`compute:22-25`) |
| writer (fork) | none | `num_pages`, `start_id` (fork `:23-24`) |

### Hardware configuration
- reader: `ttnn::create_reader_datamovement_config()` (resolved legacy triple = reader default).
- writer: `ttnn::create_writer_datamovement_config()` (resolved legacy triple = writer default).
- compute (Style A with dropped fields): `to_compute_hardware_config(compute_kernel_config)` then, on the returned
  config, force `sfpu_precision_mode = Precision::Precise` and `double_buffer_dest = true` (the legacy descriptor
  never set `math_approx_mode` / `dst_full_sync_en`, so the op ran at the descriptor defaults regardless of the
  caller's config); `config_1xx.bfp_pack_precision_mode` stays default; `unpack_modes` gets a
  `{<dfb>, UnpackMode::UnpackToSrc}` entry (legacy `Default`) for each DFB the compute kernel consumes whose
  format is `Float32` — `IN0` / `IN1` when the input dtype is Float32, `INTERMED0` / `INTERMED2` when
  `interm_data_format == Float32` — because the validator requires an explicit entry for every Float32 DFB
  consumed under `enable_32_bit_dest`. `INTERMED1` and `OUT` are produced, not consumed, by the compute kernel.
- `compiler_options.opt_level`: compute `O3` explicitly; DM kernels default (`O2`).

## Preserved Multiplicity

none — no work-split multiplicity in legacy (three distinct sources, one `KernelDescriptor` each).

## Dropped Plumbing

For each legacy RTA / CTA that disappears in the port:

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:237` → reader RTA slot 0 | `src0_buffer` (`Buffer*` binding) | `TensorBinding{INPUT_A, "src0"}`; kernel `TensorAccessor(tensor::src0)` |
| factory `:238` → reader RTA slot 1 | `src1_buffer` (`Buffer*` binding) | `TensorBinding{INPUT_B, "src1"}`; kernel `TensorAccessor(tensor::src1)` |
| factory `:175-176` → reader CTA tails; `reader:46-47` | `TensorAccessorArgs(*src0_buffer).append_to(...)`, `TensorAccessorArgs(*src1_buffer).append_to(...)`; `TensorAccessorArgs<2>()`, `TensorAccessorArgs<in0_args.next_compile_time_args_offset()>()` | the binding mechanism (host packs layout metadata; kernel builds from the token) |
| factory `:172-174` → reader CTA slots 0-1; `reader:27-28` | positional `transpose_hw_bool`, `fp32_acc_en` | named CTAs `transpose_hw`, `fp32_acc_en` (`get_arg(args::…)`) |
| factory `:234-248` → reader RTA slots 2-10; `reader:17-25` | positional scalars | named RTAs `Mt`, `Kt`, `Nt`, `MtKt`, `in1_KtNt_skip`, `in1_KtNt_mul_32`, `blocks`, `itileA_start`, `itileB_start` |
| `reader:30-34` | hardcoded CB ids `c_0`, `c_1`, `c_2`, `c_3`, `c_4` | `dfb::in0`, `dfb::in1`, `dfb::intermed1`, `dfb::intermed2` (the reader never touches `c_2`; the dead constant drops) |
| factory `:260` → writer RTA slot 0 | `dst_buffer` (`Buffer*` binding) | `TensorBinding{OUTPUT, "dst"}` (fork: `TensorAccessor(tensor::dst)`) |
| factory `:178` → writer CTA slot 0 | `output_cb_index` (magic CB index) | `DFBBinding{OUT, "out", CONSUMER}` (fork: `dfb::out`) |
| factory `:179` → writer CTA tail | `TensorAccessorArgs(*dst_buffer).append_to(...)` | the binding mechanism |
| factory `:257-263` → writer RTA slots 1-2 | positional `num_pages`, `start_id` | the fork's named RTAs `num_pages`, `start_id` |
| factory `:198-200` → compute CTA slot 0; `compute:21` | positional `transpose_hw_bool` | named CTA `transpose_hw` |
| factory `:249-256` → compute RTA slots 0-3; `compute:22-25` | positional scalars | named RTAs `batch`, `Mt`, `Kt`, `Nt` |
| `compute:27-32` | hardcoded CB ids `c_0`-`c_5` | `dfb::in0`, `dfb::in1`, `dfb::intermed0`, `dfb::intermed1`, `dfb::intermed2`, `dfb::out` |
| factory `:228-230` idle-core slot 0 zeros | `0` placeholders in the `Buffer*` slots | nothing — bindings are per-kernel; idle cores carry only the zero-valued named RTAs |

## Applied Patterns

- Self-loop DFB binding: `INTERMED0` on the compute `KernelSpec` (both PRODUCER and CONSUMER) — the kernel
  packs into it and the `untilize` helper drains it (`compute:64-77`).
- Pass DFB handles directly to LLKs and kernel-lib helpers: every `compute_kernel_hw_startup` / `matmul_*` /
  `pack_tile` / `reconfig_data_format*` call and the id-only `untilize<…>` / `tilize<…>` NTTP positions take
  `dfb::name` tokens (`compute:43-90`).
- Caution: Porting a shared kernel — rung 1 (reuse the existing `writer_unary_interleaved_start_id_metal2.cpp`
  fork; adopt `dfb::out`, `tensor::dst`, `num_pages`, `start_id`; the original is untouched).
- Kernel-side whitelist rule 7 (metadata via the object): `reader:43-44` `const uint32_t in0_tile_bytes =
  get_tile_size(cb_id_in0)` → `dfb_in0.get_tile_size()` (runtime `const`, so the getter, not the token form).
- Transfers stay as-is (recipe rule 1): the reader's loopback L1→L1 row copy keeps its `UnicastEndpoint` from
  `my_x[noc.get_noc_id()]` / `my_y[…]` and its `CoreLocalMem` destination; only the two public peeks that feed
  it change spelling.

## Deferred / Flagged

- New findings during planning: none beyond the audit. The idle-core all-zero run-args shape maps directly:
  the work unit covers the full device grid and every node gets a value for every named RTA (zeros where the
  legacy vector was zeros).
