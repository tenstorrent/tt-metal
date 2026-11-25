# Port Plan — `data_movement/reshape_on_device`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device`, ported from `ProgramDescriptor` factories to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Line numbers in the Legacy Inventory are those of the legacy files at `4571c481b8f` (the commit before the port). They can be checked with `git show 4571c481b8f:<path>`.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept`. Both factories are structs holding `static create_descriptor(const ReshapeOnDeviceParams&, const ReshapeOnDeviceInputs&, ttnn::Tensor& output_tensor)` that returns a `ProgramDescriptor`. They live in a `program_factory_t` variant (`device/reshape_op.hpp:20`). That means the "direct descriptor" device-op-class edit does not apply.
- Variants: two, chosen by `select_program_factory` on the input layout (`device/reshape_op.cpp:21-27`).
  - `ReshapeRMProgramFactory` for a ROW_MAJOR input.
  - `ReshapeTileProgramFactory` for a TILE input.
- Custom `compute_program_hash`: present at `device/reshape_op.cpp:111-127`. It keys the input with `hash_tensorspec_with_relaxation(spec, {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true})` and the rest of the key is `logical_output_shape`, `padded_output_shape`, `output_mem_config` and `program_factory.index()`. **Left intact.** No backdoor `attribute_values` / `to_hash`.
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no `validate_on_program_cache_hit`, no pybound `create_descriptor` (`reshape_nanobind.cpp` binds only the user-facing function).

*(The factory concept the port lands on, `ProgramSpecFactoryConcept`, was chosen in the audit. It is carried forward in the TTNN ProgramFactory section below.)*

### Variant: RM (`ReshapeRMProgramFactory`, `device/reshape_rm_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/kernels/dataflow/reader_unary_reshape_stick_layout_interleaved_multi_core.cpp` (`:96-98`) | `total_core_ranges`: the whole compute grid `{0,0}`-`{num_cores_x-1,num_cores_y-1}` (`:58-59`, `:100`) | `[0]` `old_stick_size` (`:88`); `TensorAccessorArgs(*src0_buffer)` appended from offset 1 (`:89`) | none | 5 per core, set on every core of the grid (`:116-216`): `[0]` `src0_buffer` as `Buffer*`, or literal `0u` when `num_old_sticks_per_core == 0` (`:176-193`); `[1]` `num_old_sticks_per_core_read`; `[2]` `num_old_sticks_read_per_barrier`; `[3]` `num_old_sticks_per_cb_push`; `[4]` `curr_sticks_read` | none | none | O2 (resolved: no `opt_level` set, DM kernel) | `ReaderConfigDescriptor{}` (`:102`) |
| writer | `.../kernels/dataflow/writer_unary_reshape_stick_layout_interleaved_multi_core.cpp` (`:105-107`) | `total_core_ranges` (`:109`) | `[0]` `src0_cb_index` = 0 (the CB index); `[1]` `new_stick_size` (`:92`); `TensorAccessorArgs(*dst_buffer)` from offset 2 (`:93`) | none | 5 per core: `[0]` `dst_buffer` as `Buffer*`, or `0u` when `num_new_sticks_per_core == 0` (`:195-212`); `[1]` `num_new_sticks_per_core_read`; `[2]` `num_new_sticks_read_per_barrier`; `[3]` `num_new_sticks_per_cb_push`; `[4]` `curr_sticks_write` | none | none | O2 (resolved) | `WriterConfigDescriptor{}` (`:111`) |

Kernel-side names of the same values (the kernels' own locals): reader `src_addr`, `num_sticks_per_core_read`, `num_read_per_barrier`, `num_sticks_per_cb_push`, `start_id`, `old_stick_size`; writer `dst_addr`, the same four counters, `dfb_out0` (the CB index), `new_stick_size`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 | `num_pages * max_page_size` (`:78`) | `total_core_ranges` | `datatype_to_dataformat_converter(input dtype)` | `new_stick_size` (`:71`, `:83`) | not set |

`num_pages` is the larger of the two per-core-group stick counts. `max_page_size` is the larger of the two stick sizes. `TT_FATAL` at `:47-52` enforces `max % min == 0`. That makes `total_size / page_size` an exact integer.

#### Semaphores

none

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `reshape_rm_program_factory.cpp:89`, kernel `reader_unary_reshape_stick_layout_interleaved_multi_core.cpp:19,23` | input (`tensor_args.input_tensor`) | reader slot 0 (`:188`; `0u` on idle cores `:180`) |
| `reshape_rm_program_factory.cpp:93`, kernel `writer_unary_reshape_stick_layout_interleaved_multi_core.cpp:19,21` | output (`output_tensor`) | writer slot 0 (`:207`; `0u` on idle cores `:199`) |

All Case 1 (the kernels use the accessor's page methods; neither does arithmetic on the raw base). No 3rd-argument accessor.

#### Work split

- Driver: `split_work_to_cores(compute_with_storage_grid_size, split_work_by_old_sticks ? num_old_sticks : num_new_sticks)` (`:63-65`). It splits by whichever stick is larger.
- The RTAs are set for every core `i` in `[0, num_cores_total)` with `core = {i / num_cores_y, i % num_cores_y}` (`:116-117`). A core that is in neither group gets zero counts. The reader and writer run on all of them.

### Variant: Tile (`ReshapeTileProgramFactory`, `device/reshape_tile_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `.../kernels/dataflow/reader_unary_reshape_interleaved.cpp` (`:85-87`) | core `(0,0)` only (`:27`) | `[0]` `alignment`, `[1]` `input_tensor.element_size()` (`:63`); `TensorAccessorArgs(*src0_buffer)` from offset 2 (`:64`) | none | 6, one core (`:95-102`): `[0]` `src0_buffer` as `Buffer*`; `[1]` `input_Wt` = input padded W / 32; `[2]` `output_N`; `[3]` `output_C`; `[4]` `output_Ht` = output padded H / 32; `[5]` `output_Wt` = output padded W / 32 | none | none | O2 (resolved) | `ReaderConfigDescriptor{}` (`:91`) |
| writer | **borrowed**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (`:107-108`) | core `(0,0)` | `[0]` `src0_cb_index` = 0; `TensorAccessorArgs(*dst_buffer)` from offset 1 (`:66-67`) | none | 3 (`:116`): `[0]` `dst_buffer` as `Buffer*`; `[1]` `num_tiles`; `[2]` `0u` (`start_id`) | none | none | O2 (resolved) | `WriterConfigDescriptor{}` (`:112`) |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 | `num_input_tiles * single_tile_size`, with `num_input_tiles = 2` (`:51`) | core `(0,0)` | `datatype_to_dataformat_converter(input dtype)` | `single_tile_size` | not set |
| 1, **conditional** | `alignment` (`:74`) | core `(0,0)` | same | `alignment` | not set |

CB 1 is built only when `alignment > FACE_WIDTH * element_size` (`:71`). `alignment` is `hal::get_dram_alignment()` for a DRAM input and `hal::get_l1_alignment()` for an L1 input (`:60-61`). That holds on Blackhole for a BFLOAT16 input in DRAM, and in no other configuration (Wormhole never builds it).

#### Semaphores

none

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `reshape_tile_program_factory.cpp:64`, kernel `reader_unary_reshape_interleaved.cpp:25,42` | input | reader slot 0 (`:97`) |
| `reshape_tile_program_factory.cpp:67`, borrowed writer `writer_unary_interleaved_start_id.cpp:24,42` | output | writer slot 0 (`:116`) |

All Case 1. No 3rd-argument accessor.

#### Work split

n/a — single core `(0,0)`.

### Shared kernels

Census by filename over `ttnn/cpp/ttnn/operations` (excluding `experimental/quasar/`), each hit checked by its bound path:

| kernel | relation | other binders | `_metal2` fork | rung |
|---|---|---|---|---|
| `reader_unary_reshape_stick_layout_interleaved_multi_core.cpp` | op's own | none: the only hit is this op's RM factory | n/a | convert in place |
| `writer_unary_reshape_stick_layout_interleaved_multi_core.cpp` | op's own | none | n/a | convert in place |
| `reader_unary_reshape_interleaved.cpp` | op's own | none | n/a | convert in place |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** (tile writer) | still on the legacy copy: `eltwise/unary_backward/.../unary_backward_program_factory.cpp:343`, `examples/example/.../multi_core_program_factory.cpp:81`, `.../single_core_program_factory.cpp:83`. The same-named files under `matmul/` and `slice/` are private copies and are not consumers. | **yes**: `writer_unary_interleaved_start_id_metal2.cpp`, beside it (`ls` of its directory) | rung 1: reuse the fork, adopt its names |

The fork's binding vocabulary is now the constraint on the writer `KernelSpec`: `dfb::out` (CONSUMER, one page per entry; the fork reads `page_bytes = dfb.get_entry_size()`), `tensor::dst`, named RTAs `num_pages` and `start_id`, and the optional defines `ENTRIES_PER_PAGE` / `PAGE_BYTES`, `OUT_SHARDED`, `BACKWARDS`. The legacy factory sets no defines, and none are needed. The fork is read-only for this port, and so is the legacy file with its existing pointer comment.

### Flags

- No unreferenced kernel file in the op directory; there is no compute kernel.
- The reader kernels already construct `DataflowBuffer` and `TensorAccessor`, and use `Noc` / `CoreLocalMem`. The conversion changes only how the handles are built (tokens), how arguments are read, and how the scratch handle is gated.
- Tile reader, `reader_unary_reshape_interleaved.cpp:37-39`: it builds the scratch DFB handle in every configuration, while the factory builds the scratch CB in only one family of configurations. Under Metal 2.0 the token for an unbound DFB does not exist.
- Local `volatile tt_l1_ptr` byte copies in the same kernel (`:40`, `:87-91`) are compliant by the user's ruling of 2026-10-22; they stay as they are.
- Test set (see the last section) is not the audit's 17-case baseline: the audit's `-k` filter missed the RM-path tests.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` for both factories (the brief's *Target concept*; the sheet's `Porting Target`).
- **Custom `compute_program_hash`**: present at `device/reshape_op.cpp:111-127` — **leave intact**. It is equal to the declared input relaxation by construction.
- **Implementation notes**:
  - Each factory's `create_descriptor` becomes `create_program_artifacts` inside the existing struct. `program_factory_t`, `select_program_factory`, validation, `compute_output_specs` and `compute_program_hash` are not touched. None of the three forced device-op-class edits applies (no pybound `create_descriptor`, no hook-only parameter, `program_factory_t` already exists).
  - The relaxation declaration is transcribed from the relaxation analysis for this op: **input** `TensorParameter` of both factories gets `{.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}`; the **output** stays strict. Never `dynamic_tensor_shape`. The six validity checks were run against this tree and all pass (recorded in the report).
  - The buffer type for the tile factory's alignment choice is read from `input_tensor.memory_config().buffer_type()` instead of the raw `Buffer*`. The validator requires an interleaved input, for which the two are the same value.

## Planned Spec Shape

### Variant: RM

- **KernelSpecs**: `reader`, `writer` (one per legacy `KernelDescriptor`), both `DataMovementHardwareConfig` from `ttnn::create_reader_datamovement_config()` / `create_writer_datamovement_config()` (the legacy descriptors used the default `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`, which resolve to the reader and writer defaults). No `opt_level` line is needed: the legacy DM kernels resolve to O2, and so does Metal 2.0.
  - reader: named CTA `old_stick_size`; named RTAs `num_sticks_per_core_read`, `num_read_per_barrier`, `num_sticks_per_dfb_push`, `start_id`; DFB binding `src0` as PRODUCER, accessor `in0`; tensor binding `input`, accessor `src`.
  - writer: named CTA `new_stick_size`; the same four RTA names; DFB binding `src0` as CONSUMER, accessor `out0`; tensor binding `output`, accessor `dst`.
- **DataflowBufferSpecs**: `src0`: `entry_size = new_stick_size`, `num_entries = num_pages * max_page_size / new_stick_size`, `data_format_metadata = dfb_data_format`; no `tile_format_metadata` (the legacy CB set none). Placement is derived: the whole compute grid.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input` with the relaxation above; `output` strict. Each has one `TensorBinding`.
- **WorkUnitSpecs**: one, `main`: kernels `{reader, writer}`, nodes `total_core_ranges`.

### Variant: Tile

- **KernelSpecs**: `reader` (own kernel) and `writer` (the borrowed kernel's fork).
  - reader: named CTAs `ALIGNMENT`, `ELEMENT_SIZE`; named RTAs `input_Wt`, `output_N`, `output_C`, `output_Ht`, `output_Wt`; DFB binding `in0` as PRODUCER (accessor `in0`); tensor binding `input` (accessor `src0`); **conditionally** the scratch DFB (below) and the define `WIDE_ALIGNMENT_SCRATCH`.
  - writer (fork): DFB binding `in0` as CONSUMER (accessor `out`); tensor binding `output` (accessor `dst`); named RTAs `num_pages`, `start_id`; no CTAs, no defines.
- **DataflowBufferSpecs**: `in0`: `entry_size = single_tile_size`, `num_entries = 2`. `scratch`, **conditional**: `entry_size = alignment`, `num_entries = 1`. Both with `data_format_metadata = dfb_data_format`.
  - Endpoint census for `scratch`: one toucher (the reader). It only takes a raw peek (`dfb_intermed.get_write_ptr()`); the NoC read into it and the byte copy out of it are both in the same kernel, and no kernel calls a FIFO method on it. One toucher means a self-loop: the reader is bound PRODUCER and CONSUMER, both under accessor `intermed`.
  - Endpoint census for `in0`: reader `reserve_back` / `push_back` (locked producer), writer fork `wait_front` / `pop_front` (locked consumer). 1P+1C, as the brief says.
- **SemaphoreSpecs**: none.
- **TensorParameters**: as in the RM variant.
- **WorkUnitSpecs**: one, `main`: kernels `{reader, writer}`, node `(0,0)`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Each kernel has exactly one `KernelDescriptor`. The RM work split varies runtime arguments per core, not compile-time arguments.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| RM reader RTA slot 0 (`reshape_rm_program_factory.cpp:178-193`, kernel `:12`) | `src0_buffer` as `Buffer*`, or `0u` on idle cores | `TensorBinding` `input` / accessor `src` |
| RM writer RTA slot 0 (`:197-212`, kernel `:11`) | `dst_buffer` as `Buffer*`, or `0u` on idle cores | `TensorBinding` `output` / accessor `dst` |
| Tile reader RTA slot 0 (`reshape_tile_program_factory.cpp:97`, kernel `:16`) | `src0_buffer` as `Buffer*` | `TensorBinding` `input` / accessor `src0` |
| Tile writer RTA slot 0 (`:116`, borrowed writer `:19`) | `dst_buffer` as `Buffer*` | `TensorBinding` `output` / accessor `dst` (fork) |
| RM reader `TensorAccessorArgs(*src0_buffer).append_to` (`:89`) and kernel `TensorAccessorArgs<1>()` (`:19`) | accessor layout plumbed through CTA slots from offset 1 | the binding token end to end |
| RM writer `TensorAccessorArgs(*dst_buffer).append_to` (`:93`) and kernel `TensorAccessorArgs<2>()` (`:19`) | same, offset 2 | same |
| Tile reader `TensorAccessorArgs(*src0_buffer).append_to` (`:64`) and kernel `TensorAccessorArgs<2>()` (`:25`) | same, offset 2 | same |
| Tile writer `TensorAccessorArgs(*dst_buffer).append_to` (`:67`) | same, offset 1 | the fork's `TensorAccessor(tensor::dst)` |
| RM writer CTA `[0]` (`:92`, kernel `:17`) | `src0_cb_index` = 0, a magic CB index | `DFBBinding` accessor `out0` |
| Tile writer CTA `[0]` (`:66`, borrowed writer `:23`) | `src0_cb_index` = 0 | `DFBBinding` accessor `out` (fork) |
| RM reader kernel `:21` | `constexpr auto dfb_in0 = tt::CBIndex::c_0` | `dfb::in0` |
| Tile reader kernel `:34`, `:37` | `dfb_id_in0 = 0`, `dfb_id_intermed = 1` | `dfb::in0`, `dfb::intermed` (the second one `#ifdef`-gated) |
| RM reader CTA `[0]`, writer CTA `[1]`, tile reader CTAs `[0]`, `[1]` | positional | named CTAs `old_stick_size`, `new_stick_size`, `ALIGNMENT`, `ELEMENT_SIZE` |
| All RTA slots after slot 0 (every kernel) | positional `get_arg_val<uint32_t>(N)` | named RTAs (listed above), none re-indexed by hand |
| the `reader_idle` / `writer_idle` branches (`:176-212`) | exist only to pick `0u` over a `Buffer*` | gone: one `AddRuntimeArgsForNode` per kernel per core |
| `TT_ASSERT(dst_buffer != nullptr, ...)` (`reshape_tile_program_factory.cpp:42`) | guard on the raw `Buffer*` | deleted with its subject (see the report) |

Names: every `cb` in a name is renamed to `dfb` (`num_*_per_cb_push` → `num_*_per_dfb_push`, `cb_write_offset` → `dfb_write_offset`, `cb_read_offset` → `dfb_read_offset`, `cb_data_format` → `dfb_data_format`). The brief wrote the RM RTA name as `num_sticks_per_cb_push`; the recipe's name sweep expects zero `cb` hits in the op directory. The Metal 2.0 name is therefore `num_sticks_per_dfb_push`.

## Applied Patterns

- Sync-free and single-ended CBs → self-loop DFB: the tile reader's scratch (one toucher, raw peek). A DM self-loop is legal on Gen1 and is Quasar-uplift's concern.
- Conditional / optional resource bindings, including *Promote a CTA gate to a define*: `scratch` is declared, bound and given its define under the one host condition; the kernel `#ifdef`s the handle and the one ternary that reads it. `MISALIGNED` stays a `constexpr` built from the `ALIGNMENT` and `ELEMENT_SIZE` CTAs.
- Porting a shared kernel, rung 1: the tile writer binds the existing `_metal2` fork.
- Relaxation analysis transcription (input only, both flags) — `ttnn_factory.md`, *Tensor-arg matching*, source 1.
- Not applicable: aliased DFBs, same-FIFO aliasing, multi-binding, multi-variant helper (the two variants are separate factory structs already).

## Deferred / Flagged

- **Audit baseline gap.** The audit's baseline used `-k "reshape_on_device or on_device"` on `test_reshape.py`: 17 cases, of which 16 (`test_reshape_in_4D_on_device`) call the `ttnn.reshape` view and never reach this op, and one is the tile cache test. The five RM-path tests that do call `ttnn.reshape_on_device` (`test_reshape_cw_div2_rm`, `test_reshape_cw_mul2_rm`, `test_reshape_hw_div2_rm`, `test_reshape_hw_mul2_rm`, `test_reshape_hw_rm_with_program_cache`) are not selected by that filter. The test set below includes them.
- **Blackhole-only configuration.** The scratch DFB and the volatile byte-copy loop run only for Blackhole with a BFLOAT16 DRAM input. This machine is a Wormhole n150. Plan: build and compile the gated variant, and try a temporary switch that forces the variant on Wormhole. Neither is a substitute for a Blackhole run; the report says what was and was not covered.
- **Legality-check forcing.** The recipe's `skip_validation` step does not apply on this tree (see the report): the Metal 2.0 miss path always validates, and the hit path is switched by `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`. Every verification run uses that knob.
- Nothing from the audit's *Misc anomalies* is touched by the port: they are routed to the report.

## Test set (to confirm)

The recipe asks the invoker to confirm this set; the session could not ask. The set is recorded here for review. Verification runs every test that reaches the op's kernels, with Watcher on and `validate_program_args` on, each group on a fresh JIT cache. Reachability is established statically (the callers below) and by which of the three op-owned kernels a fresh-cache run compiles.

Callers of `ttnn::reshape_on_device`: the Python binding `ttnn.reshape_on_device`; `fold.cpp:80,93` (only the `use_transpose_as_fold=True` path); `split.cpp:43,52` (only the batched last-dim two-chunk tiled split). No gtest names the op.

| group | tree | files / selection |
|---|---|---|
| op cases | `tests/ttnn/unit_tests/base_functionality/` | `test_reshape.py`: the RM-path tests above, `test_reshape_on_device_tile_same_padded_shape_different_alignment`, and the audit's `test_reshape_in_4D_on_device` |
| legacy coverage | `tests/tt_eager/python_api_testing/unit_testing/misc/` | `test_reshape.py`, in full (13) |
| fold | `tests/ttnn/unit_tests/operations/conv/data_movement/` | `test_fold_op.py`, in full |
| fold, nightly | `tests/ttnn/nightly/unit_tests/operations/data_movement/` | `test_universal_input_tm_fold.py`, in full |
| split, nightly | same directory | `test_split.py`, `test_universal_input_tm_split.py`, in full |
| grid sample, nightly | `tests/ttnn/nightly/unit_tests/operations/pool/` | `test_grid_sample.py -k test_grid_sample_oft` (the only function calling `ttnn.split`) |
| guards and others | `tests/ttnn/unit_tests/` | `operations/data_movement/test_zero_divisor_guards.py`, `operations/pool/test_maxpool2d_shard_grid.py`, `base_functionality/test_comparison_mode_context.py -k test_fold_legacy_transpose_with_asymmetric_padding_in_comparison_mode` (needs `enable_fast_runtime_mode` off) |
| end to end | `tests/end_to_end_tests/` | `test_unit_ops.py` |
| sweeps | `tests/sweep_framework/sweeps/` | `data_movement/split/split_pytorch2.py`, `model_traced/split_model_traced.py`, `model_traced/fold_model_traced.py` |

Considered and not run: `ops_for_profiling.py` (calls `ttnn.reshape`, the view op), `tests/tt_eager/.../sweep_tests/pytests/tt_dnn/test_reshape.py` (every line is commented out; its helper `tt_lib_ops.reshape` is reachable only from that dead file), eleven model files under `models/experimental/` (need model weights and hardware not present here), the multi-device `ccl/composite_common.cpp` split caller (needs a mesh), and the other 444 cases of `test_reshape.py` (they call `ttnn.reshape`, which routes to `reshape_view`).
