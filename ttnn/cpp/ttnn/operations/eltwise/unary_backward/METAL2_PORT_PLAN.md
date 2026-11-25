# Port Plan — eltwise/unary_backward

Port plan for `eltwise/unary_backward`, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Line numbers in the *Legacy Inventory* refer to the legacy factory, `device/unary_backward_program_factory.cpp`
as of `5340d88ced5` (an `origin/main` commit, the file before this port). The brief and the audit cite the same lines.

## Legacy Inventory

*Filled in during the inventory step.*

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept`. `UnaryBackwardProgramFactory::create_descriptor(args, tensor_args, output)`
  takes no coordinate (`.cpp:26-27`, declared in `.hpp`). The factory methods live in the factory struct of the
  `program_factory_t` variant, not on the device operation (`device/unary_backward_device_operation.hpp:28`), so
  [exception 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md) of the
  device-op edits does not apply.
- Variants: single. One factory serves 40 op types. `get_kernel_spec(op_type)` picks the compute kernel. Four op types are
  registered by hand (`SIGMOID_BW`, `TANH_BW`, `GELU_BW`, `GELU_TANH_BW`; `device/unary_backward_op_utils.cpp:15-61`) and 36
  by `device/ops/*.def` files, which CMake globs into a generated `unary_bw_ops.def.inc` (`CMakeLists.txt:39-46`).
  There are three *configurations* inside the one factory (see *Kernels* and *CBs*): ALIAS (`can_alias_shards`, `.cpp:82-129`),
  ADDRESS-TILE (not aliased, TILE), and ADDRESS-RM (`rm_blocked`, `.cpp:138`).
- Custom `compute_program_hash`: present at `device/unary_backward_device_operation.cpp:273-300` (declared `.hpp:35`) — left
  intact. No backdoor `attribute_values` / `to_hash`. No `validate_on_program_cache_hit`, so the framework runs the miss
  validator on every hit.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory analysis section.
Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels

`opt_level` is the resolved level: the legacy factory sets none (`grep -n opt_level` over the factory prints nothing), so
every DM kernel resolves to `O2` and the compute kernel to `O3`.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs (positional, per core) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (ALIAS, ADDRESS-TILE) | `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` (borrowed, no `_metal2` fork) | `all_cores` (`.cpp:295`); ALIAS: the shard grid | `{0}` + (ADDRESS only) `TensorAccessorArgs(grad_output)`, `TensorAccessorArgs(input)` (`.cpp:324,330-331`) | none | `{grad_output Buffer*, input Buffer*, units, units_done, 0, 0, num_cores_y}` (`.cpp:468`) | none | ALIAS: `IN0_SHARDED=1`, `IN1_SHARDED=1` (`.cpp:326-328`) | O2 | `ReaderConfigDescriptor{}` |
| writer (ALIAS, ADDRESS-TILE) | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (borrowed, `_metal2` fork exists) | `all_cores` (`.cpp:300`) | `{grad_input_cb_index (2), TensorAccessorArgs(output)}`, the accessor always appended (`.cpp:340-341`) | none | `{grad_input Buffer*, units, units_done}` (`.cpp:470`) | none | ALIAS: `OUT_SHARDED=1` (`.cpp:345-347`) | O2 | `WriterConfigDescriptor{}` |
| reader (ADDRESS-RM) | `unary_backward/device/kernels/dataflow/reader_unary_backward_row_major.cpp` (own) | `all_cores` | `TensorAccessorArgs(grad_output)`, `TensorAccessorArgs(input)` (`.cpp:306-308`) | none | 17: `units, units_done, row_elements, TILE_HW, chunks_per_row, rows_per_tile, total_rows, grad Buffer*, grad {page_elements, pages_per_row, element_bytes, alignment}, input Buffer*, input {same four}` (`.cpp:431-449`) | none | none | O2 | `ReaderConfigDescriptor{}` |
| writer (ADDRESS-RM) | `unary_backward/device/kernels/dataflow/writer_unary_backward_row_major.cpp` (own) | `all_cores` | `TensorAccessorArgs(output)` (`.cpp:314-315`) | none | 12: the first 7 as the reader, `output Buffer*`, `output {four}` (`.cpp:450-463`) | none | none | O2 | `WriterConfigDescriptor{}` |
| compute | one of 41 files in `unary_backward/device/kernels/compute/` (own), chosen by `get_kernel_spec`; `GELU_TANH_BW` takes `eltwise_bw_gelu_tanh_fp32.cpp` when `fp32_dest_acc_en` (`.cpp:387-390`) | `all_cores` (`.cpp:392`) | none | none | `{units}`, or `{units * chunks_per_row}` for ADDRESS-RM (`.cpp:465,469`) | none | `MIXED_OPERAND_DATA_FORMATS` (only when the two operand formats differ), `COPY_DEST_DATA_FORMAT`, `BF16_ROUNDING_MODE` (`.cpp:371-384`) | O3 | `ComputeConfigDescriptor{.math_fidelity = spec.math_fidelity, .fp32_dest_acc_en, .unpack_to_dest_mode}` (`.cpp:393-397`); `dst_full_sync_en`, `math_approx_mode`, `bfp8_pack_precise` unset (all `false`) |

The 41 compute kernels are the 5 hand-registered files (`eltwise_bw_sigmoid.cpp`, `_tanh.cpp`, `_gelu_poly.cpp`,
`_gelu_tanh.cpp`, `_gelu_tanh_fp32.cpp`) and the 36 files named by the `.def` files. At the start of the port:
`ls device/ops/*.def | wc -l` = 36 and `ls device/kernels/compute/eltwise_bw_*.cpp | wc -l` = 41. The 36 are clones of
`eltwise_bw_abs.cpp`: after mapping each file's own op name (header, struct, `*_bw_tile_init`, `*_bw_tile`, comment) to
`abs`, all 36 are identical, with the sole difference that `sigmoid` and `tanh` name their struct `…BwBf16`.
Each compute kernel has one positional RTA (`get_arg_val<uint32_t>(0)`) and no CTAs.

### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 grad_output | `cb_pages * grad_output tile size` (`.cpp:245`) | `all_cores` | `datatype_to_dataformat_converter(grad_output.dtype())` | grad_output tile size | unset. `.buffer` = the grad_output shard under ALIAS (`.cpp:256-260`) |
| 1 input | `cb_pages * input tile size` | `all_cores` | from `input.dtype()` | input tile size | unset. `.buffer` = the input shard under ALIAS (`.cpp:261`) |
| 2 grad_input | `cb_pages * output tile size` | `all_cores` | from `output.dtype()` | output tile size | unset. `.buffer` = the output shard under ALIAS (`.cpp:262-266`) |
| 3 reader scratch | `scratch_bytes` = max tile size + 3 * max alignment (`.cpp:275-277`) | `all_cores`, **only under `rm_blocked`** (`.cpp:272-289`) | output format | `scratch_bytes` | unset |
| 4 writer scratch | `scratch_bytes` | `all_cores`, only under `rm_blocked` | output format | `scratch_bytes` | unset |

`cb_pages` is `shard_tiles` under ALIAS and 2 otherwise (`.cpp:229-230`). There is no `GlobalCircularBuffer`, no
`address_offset`, and no aliased multi-element `format_descriptors`.

### Semaphores

none — the op has no semaphores (zero `semaphore` hits under `device/`).

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `.cpp:330` (reader CTAs, ADDRESS-TILE) | grad_output | RTA 0 (`.cpp:468`) |
| `.cpp:331` | input | RTA 1 (`.cpp:468`) |
| `.cpp:341` (writer CTAs, ADDRESS-TILE; appended in every config) | output (`grad_input_buffer`) | RTA 0 (`.cpp:470`) |
| `.cpp:307` (RM reader CTAs) | grad_output | RTA 7 (`.cpp:440`) |
| `.cpp:308` | input | RTA 12 (`.cpp:445`) |
| `.cpp:315` (RM writer CTAs) | output | RTA 7 (`.cpp:459`) |

Under ALIAS no kernel builds an accessor (`IN0_SHARDED`, `IN1_SHARDED`, `OUT_SHARDED`). The legacy factory still pushes the
`Buffer*` RTA slots there (`.cpp:468,470`), and they are dead. Kernel side, six `TensorAccessor(args, addr)` constructions,
all two-argument: `reader_binary_interleaved_start_id.cpp:46,53`, `writer_unary_interleaved_start_id.cpp:42`,
`reader_unary_backward_row_major.cpp:55-56`, `writer_unary_backward_row_major.cpp:38`.

### Work split

- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_units)` for ADDRESS (`.cpp:214`). For ALIAS the cores are the
  shard grid and each core owns exactly its own shard (`.cpp:207-212`).
- `num_units` = `num_tiles` (tile) or `ceil(total_rows / rows_per_tile)` blocks (RM) (`.cpp:190`).
- ADDRESS: `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, and the two per-core counts come straight from the
  split. ALIAS: `num_cores = shard_grid.num_cores()`, `core_group_1 = all_cores = shard grid`, `core_group_2` empty,
  `num_units_per_core_group_1 = shard_tiles`.
- The two core groups differ only in their per-core RTA values (`.cpp:419-474`); each kernel is one descriptor over `all_cores`.

### Shared kernels

| kernel | kind | census | `_metal2` fork beside it | rung |
|---|---|---|---|---|
| `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | borrowed | `grep -rl reader_binary_interleaved_start_id ttnn/cpp/ttnn/operations/`: this factory only. `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437` binds it through the legacy generic-op API | none | **2: create the fork** `reader_binary_interleaved_start_id_metal2.cpp` in that directory, plus the pointer comment in the original |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | borrowed | 26 op factories bind the fork; legacy original also by `test_generic_op.cpp:894,1013`, `test_exp_approx_disable_sfploadmacro.py:86`, `test_generic_op_host_time.py:68`. `data_movement/slice` binds its own private copy of the same name, which is not a binder | `writer_unary_interleaved_start_id_metal2.cpp` exists | **1: reuse** |

Fork vocabulary of the existing writer (the constraint this port is built against): `dfb::out` (consumer), `tensor::dst`,
named RTAs `num_pages` and `start_id`, defines `OUT_SHARDED`, `BACKWARDS`, `ENTRIES_PER_PAGE` / `PAGE_BYTES`. This factory
supplies `dfb::out`, `tensor::dst` (not under ALIAS), `num_pages`, `start_id` and `OUT_SHARDED` (under ALIAS). It uses neither
`BACKWARDS` nor the `ENTRIES_PER_PAGE` pair. The fit is exact; the fork is not edited.

This op's own kernels (41 compute kernels, the RM reader and writer, `row_major_pages.hpp`, `operand_reconfig.hpp`) are not
lent: no file outside the op directory names them (the `binary_backward` op binds its own private `eltwise_bw_*` kernels in
its own directory). They are converted in place.

### Flags

- **The 36 `.def` kernels are clones, and the op-type set is open.** Open `.def` PRs add one file and one kernel each (#59374
  logiteps, #59369 acosh, #57589 softsign, #59336 erfc). Invoker decision (audit Question 1, option b): a `.def` type that
  arrives mid-port, for example at a rebase, is converted with the same edit when its kernel is a clone, and is listed in the
  report. Stop and ask only for a kernel that is not a clone. None had landed when the port started: 36 `.def` files, 41
  kernels, and the checkout was `origin/main` `5340d88ced5`.
- **Dead RTAs in the borrowed reader, for this op.** `block_height = 0`, `block_width = 0`, `num_cores_y` and CTA
  `block_or_width_sharded = 0` are read only under `if constexpr (block_or_width_sharded)`. They are dead for this op and live
  for the kernel's other binder, so the fork keeps the full interface.
- **Two compute kernels name the RTA `num_tiles`** (`eltwise_bw_gelu_tanh.cpp:19`, `eltwise_bw_gelu_tanh_fp32.cpp:21`); the other 39
  name it `per_core_tile_cnt`. The schema name is `per_core_tile_cnt` for all 41, so one `runtime_arg_schema` serves every
  op type. The local variable in those two kernels keeps its name.
- **Stale kernel-contract comment** at `device/unary_backward_op_utils.hpp:19-21` (it describes CB indices and positional RTA 0).
  It is outside the factory body. Recorded in the report, not edited.
- **Per-config tensor bindings.** The three `TensorParameter`s exist in every config. They are bound by kernels in the two
  ADDRESS configs and only borrowed by DFBs in ALIAS.
- No unreferenced kernel file in the op directory (all 41 `eltwise_bw_*.cpp` are reached through `get_kernel_spec` or a `.def`).

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments` to translate,
  no op-owned tensors, one program per mesh. A cache hit refreshes the tensor bindings and nothing else, which is safe because
  the custom hash pins everything the baked values depend on.
- **Custom `compute_program_hash`**: present at `device/unary_backward_device_operation.cpp:273-300` — leave intact.
- **Implementation notes**:
  - `create_program_artifacts(args, tensor_args, Tensor& output)` replaces `create_descriptor`. The geometry computations
    (ALIAS test, row-major paging, work split) stay on the `ttnn::Tensor` accessors exactly as they were, because rewriting
    them onto `MeshTensor` would be a refactor beyond the port. The three `MeshTensor`s are taken once at the top of the
    factory and used for the `TensorParameter` specs and the `TensorArgument`s.
  - Relaxation (analysis doc, `legality (see analysis)`): every `TensorParameter` declares
    `.relaxations = {.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}`, unconditionally. The audit ran the
    doc's six validity checks and all passed; none of this op's files changed since (the checkout is the audited one).
  - Verification runs set `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` (invoker instruction, see the report's
    *Friction*): the cache-hit `TensorParameter` check is off by default.
  - Implicit-sync opt-out (audit Question 2, decided by the invoker): left out. Every DM kernel uses the default
    `ttnn::create_reader_datamovement_config()` / `ttnn::create_writer_datamovement_config()`.

## Planned Spec Shape

*Filled in during the planning step.* One `ProgramSpec`, built per cache miss, in one of three configurations of the same
shape.

- **KernelSpecs** (one per legacy `KernelDescriptor`, 1:1):
  - `reader`: ALIAS and ADDRESS-TILE bind the new `reader_binary_interleaved_start_id_metal2.cpp`; ADDRESS-RM binds
    `reader_unary_backward_row_major.cpp`. `hw_config = ttnn::create_reader_datamovement_config()`.
  - `writer`: ALIAS and ADDRESS-TILE bind the existing `writer_unary_interleaved_start_id_metal2.cpp`; ADDRESS-RM binds
    `writer_unary_backward_row_major.cpp`. `hw_config = ttnn::create_writer_datamovement_config()`.
  - `compute`: the file `get_kernel_spec` selects. `hw_config` is Style B, built field by field (the legacy op sets a
    `ComputeConfigDescriptor` directly): `fpu_math_fidelity = spec.math_fidelity`, `enable_32_bit_dest = fp32_dest_acc_en`,
    and `unpack_modes = {grad_output: UnpackToDest, input: UnpackToDest}` only when `fp32_dest_acc_en` (the legacy vector set
    `UnpackToDestFp32` on c_0 and c_1 and nothing else, and only then). Everything else stays at the Metal default, which equals
    the legacy default (`double_buffer_dest = true` for `dst_full_sync_en = false`, `sfpu_precision_mode = Precise` for
    `math_approx_mode = false`, `bfp_pack_precision_mode` unset). `compiler_options.opt_level = O3`, explicit.
- **DataflowBufferSpecs** (one per legacy CB, 1:1):
  - `grad_output`, `input`, `grad_input`: `entry_size` = tile size, `num_entries` = `dfb_pages` (2, or `shard_tiles` under ALIAS),
    `data_format_metadata` from the tensor dtype. `borrowed_from` = the matching `TensorParameter` under ALIAS.
    `tile_format_metadata` stays unset (the legacy `tile` field was unset).
  - `reader_scratch`, `writer_scratch` (only under `rm_blocked`): `entry_size = scratch_bytes`, `num_entries = 1`.
- **Endpoints** (re-derived from the kernel-touch census, matches the brief):
  - `grad_output`, `input`: reader PRODUCER, compute CONSUMER, in every config (1P+1C).
  - `grad_input`: compute PRODUCER, writer CONSUMER, in every config (1P+1C).
  - `reader_scratch`: the RM reader is the only toucher, through the raw pointer (`row_major_pages.hpp:140-143`) → **self-loop**
    (PRODUCER + CONSUMER on the RM reader, one shared accessor name). `writer_scratch`: the RM writer alone → self-loop.
  - No dead CB, no multi-binding flag, no other conditional DFB.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `grad_output`, `input`, `output`, from each tensor's `tensor_spec()`, with the relaxation above.
  `TensorBinding`s: ADDRESS-TILE reader `src0`/`src1`, writer `dst`; ADDRESS-RM reader `grad`/`input`, writer `dst`; none
  under ALIAS (the parameters are then used by `borrowed_from` alone).
- **WorkUnitSpecs**: one, `main` = `{reader, writer, compute}` on `all_cores`.
- **Binding names**: reader fork `dfb::in0`, `dfb::in1`, `tensor::src0`, `tensor::src1`; writer fork `dfb::out`, `tensor::dst`;
  RM reader `dfb::grad/input/scratch`, `tensor::grad/input`; RM writer `dfb::dst/scratch`, `tensor::dst`; compute
  `dfb::grad_out/input/grad_in`.
- **Run args**: per-node RTAs through `AddRuntimeArgsForNode` inside the legacy per-core loop (unchanged loop); one
  `KernelRunArgs` per kernel; `tensor_args` for all three parameters.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. `core_group_1` and `core_group_2` differ only in per-core RTA values; each kernel is
one `KernelDescriptor` on `all_cores`, so each stays one `KernelSpec` in one `WorkUnitSpec`.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `.cpp:468` RTA 0,1 (reader) | `grad_output Buffer*`, `input Buffer*` | `TensorBinding` `src0`, `src1` (ADDRESS-TILE); dead under ALIAS, dropped |
| factory `.cpp:470` RTA 0 (writer) | `grad_input Buffer*` | `TensorBinding` `dst` (ADDRESS-TILE); dead under ALIAS, dropped |
| factory `.cpp:440,445` (RM reader RTA 7, 12) | `grad_output Buffer*`, `input Buffer*` | `TensorBinding` `grad`, `input` |
| factory `.cpp:459` (RM writer RTA 7) | `grad_input Buffer*` | `TensorBinding` `dst` |
| factory `.cpp:307-308,315,330-331,341` | `TensorAccessorArgs(*buffer).append_to(cta)` (6 sites) | the binding mechanism |
| `reader_binary_interleaved_start_id.cpp:28-35` | `TensorAccessorArgs<N>()` chain | `TensorAccessor(tensor::src0 / src1)` in the fork |
| `reader_binary_interleaved_start_id.cpp:46,53`; `writer_unary_interleaved_start_id.cpp:42` | `TensorAccessor(args, addr)` | `TensorAccessor(tensor::…)` (the writer fork already has it) |
| `reader_unary_backward_row_major.cpp:52-56`; `writer_unary_backward_row_major.cpp:37-38` | `TensorAccessorArgs<…>()` + `TensorAccessor(args, addr)` | `TensorAccessor(tensor::grad / input / dst)` |
| `reader_binary_interleaved_start_id.cpp:17-18` RTA 0,1 | `src0_addr`, `src1_addr` | the two tensor bindings |
| `reader_binary_interleaved_start_id.cpp:19-23` RTA 2-6 | `num_tiles, start_id, block_height, block_width, num_cores_y` | named RTAs, same names |
| `reader_binary_interleaved_start_id.cpp:27` CTA 0 | `get_compile_time_arg_val(0) == 1` | named CTA `block_or_width_sharded` |
| `reader_binary_interleaved_start_id.cpp:25-26` | `cb_id_in0 = c_0`, `cb_id_in1 = c_1` | `dfb::in0`, `dfb::in1` |
| `reader_binary_interleaved_start_id.cpp:45,52` | `get_tile_size(cb_id)` (non-constexpr) | `dfb.get_tile_size()` (whitelist §A) |
| `writer_unary_interleaved_start_id.cpp:23-24` CTA 0 and `TensorAccessorArgs<1>()` | `cb_id_out`, `dst_args` | already `dfb::out` in the fork |
| RM reader RTAs 0-6, 8-11, 13-16 | positional `get_arg_val(arg++)` | 15 named RTAs (`num_blocks`, `start_block`, `row_elements`, `chunk_elements`, `chunks_per_row`, `rows_per_tile`, `total_rows`, `grad_page_elements`, `grad_pages_per_row`, `grad_element_bytes`, `grad_alignment`, `input_page_elements`, `input_pages_per_row`, `input_element_bytes`, `input_alignment`) |
| RM writer RTAs 0-6, 8-11 | positional `get_arg_val(arg++)` | 11 named RTAs (the first 7 as the reader, `dst_page_elements`, `dst_pages_per_row`, `dst_element_bytes`, `dst_alignment`) |
| RM reader `cb_id_grad/input/scratch` (c_0, c_1, c_3); RM writer `cb_id_dst/scratch` (c_2, c_4) | constants fed to `DataflowBuffer(...)` | `dfb::grad`, `dfb::input`, `dfb::scratch`; `dfb::dst`, `dfb::scratch` |
| 41 compute kernels: `compute_kernel_hw_startup(c_0, c_2)` and `dfb_*_id = c_0/c_1/c_2` | CB numbers | `dfb::grad_out`, `dfb::input`, `dfb::grad_in` |
| 41 compute kernels: `get_arg_val<uint32_t>(0)` | positional RTA 0 | named RTA `per_core_tile_cnt` |
| factory `unpack_to_dest_mode` vector indexed by CB number (`.cpp:362-366`) | `NUM_CIRCULAR_BUFFERS`-sized vector | `ComputeHardwareConfig::unpack_modes`, keyed by DFB name |
| factory `CBIndex::c_0..c_4` constants (`.cpp:223-225`, `:278`) | CB numbers | DFB names |
| reader CTA 0 `{0}` (`.cpp:324`) | positional CTA | named CTA `block_or_width_sharded = 0` |

Same-value-on-every-node RTAs (`row_elements`, `chunk_elements`, `chunks_per_row`, `rows_per_tile`, `total_rows`, the paging
fields, `block_height`, `block_width`, `num_cores_y`) are really CRTAs. They stay RTAs here, as the recipe requires; they are
listed in the report for the later cleanup.

## Applied Patterns

- [Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings):
  the reader's and writer's tensor bindings are present only where the kernel builds the accessor. The kernels already
  `#ifdef`-gate the accessor construction on `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED`, and the host emits the same defines
  and omits the same bindings, so the gate and the binding share one condition.
- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  `reader_scratch` on the RM reader and `writer_scratch` on the RM writer, one toucher each (a DM self-loop, legal on Gen1).
- Borrowed-memory DFBs (`borrowed_from`): the three data DFBs under ALIAS.
- [Multi-variant factory](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories):
  the three configurations and the per-op / per-dtype compute kernel choice are branches inside `create_program_artifacts`.
- [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  rung 2 for the binary reader (new fork, pointer comment in the original), rung 1 for the unary writer (reuse).
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `dfb::name` into `compute_kernel_hw_startup` and into `ckl::input(...)` / `ckl::output(...)`.

## Deferred / Flagged

- New findings during planning: none that block the port. The brief suggests `get_entry_size()` for the reader's
  `get_tile_size(cb_id)` calls; the whitelist (§A) and the recipe (kernel-side rule 7) map `get_tile_size(cb_id)` to the member
  `get_tile_size()`, which reads the same descriptor array, so the fork uses that (see the report's *Friction*).
