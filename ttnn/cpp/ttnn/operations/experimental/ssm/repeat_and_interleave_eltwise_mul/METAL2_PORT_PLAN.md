# Port Plan — `experimental/ssm/repeat_and_interleave_eltwise_mul`

Port plan for `RepeatAndInterleaveEltwiseMulProgramFactory`, ported from the legacy
`ProgramDescriptor` API (`ProgramDescriptorFactoryConcept`) to Metal 2.0
(`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

**The op has one factory and three kernel-source configurations**, selected per cache miss by
input width. Everything config-scoped below uses the audit's labels:

| Label | Defines | Trigger (`a` width × `b` width) |
|---|---|---|
| **Config A** | `REPEAT_IN0` + `REPEAT_INTERLEAVE_IN1` | `a[-1] == 32`, `b[-1] == 5120` |
| **Config B** | `REPEAT_INTERLEAVE_IN1` only | `a[-1] == 32·5120`, `b[-1] == 5120` |
| **Config C** | `REPEAT_IN0` only | `a[-1] == 32`, `b[-1] == 32·5120` |

The fourth combination is unreachable (`TT_FATAL((ashape[3] != bshape[3]))`,
`..._device_operation.cpp:72`). All three are live CI paths.

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `static tt::tt_metal::ProgramDescriptor
  create_descriptor(...)` (`device/repeat_and_interleave_eltwise_mul_program_factory.hpp:15-16`,
  definition `..._program_factory.cpp:24-25`, returns at `:259`).
- Variants: single. `program_factory_t =
  std::variant<RepeatAndInterleaveEltwiseMulProgramFactory>`
  (`..._device_operation.hpp:26`). Variation is by *kernel-source configuration* (A/B/C via
  `defines`), not by factory variant.
- Custom `compute_program_hash`: **none** — the device-op declares only
  `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`
  (`..._device_operation.hpp:28-32`). Nothing to delete.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's
TTNN factory analysis section. Carried forward in the TTNN ProgramFactory section below.)*

### Kernels

All three `KernelDescriptor`s use `core_ranges = all_cores` and
`source_type = FILE_PATH`. All three are op-owned; none is shared (see *Shared kernels*).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_ssm_eltwise_mul.cpp` | `all_cores` | `[0]=src0_cb_index(c_0)`, `[1]=src1_cb_index(c_1)`, `[2]=cb_intermed1_index(c_25)`, `[3]=cb_intermed2_index(c_26)`, then `TensorAccessorArgs(src0_buffer)` and `TensorAccessorArgs(src1_buffer)` appended (`:84-85`) | none | per core (`:222-230`): `src0_buffer` (`Buffer*`), `src1_buffer` (`Buffer*`), `num_blocks_per_core`, `num_blocks_written`, `bshape[2]/TILE_HEIGHT`, `bshape[-1]/TILE_WIDTH`, `ashape[-1]/TILE_WIDTH` | none | `REPEAT_IN0` / `REPEAT_INTERLEAVE_IN1` per config (`:185`) | absent → resolves **O2** | `ReaderConfigDescriptor{}` (`:186`) |
| writer | `device/kernels/writer_ssm_eltwise_mul.cpp` | `all_cores` | `[0]=output_cb_index(16)`, then `TensorAccessorArgs(out_buffer)` appended (`:89`) | none | per core (`:240-246`): `out_buffer` (`Buffer*`), `writer_num_tiles`, `writer_start_id`, `bshape[2]/TILE_HEIGHT`, `HIDDEN_SIZE` | none | none (writer gets no `defines`) | absent → resolves **O2** | `WriterConfigDescriptor{}` (`:196`) |
| compute | `device/kernels/ssm_eltwise_mul.cpp` | `all_cores` | `[0..6]` = `c_0`, `c_1`, `16`, `c_24`, `c_25`, `c_26`, `c_27` (`:90-98`) | none | per core (`:249-250`): `num_blocks_per_core`, `bshape[2]/TILE_HEIGHT` | none | `REPEAT_IN0` / `REPEAT_INTERLEAVE_IN1` per config (`:206`) | absent → resolves **O3** (`ComputeConfigDescriptor`) | `ComputeConfigDescriptor{.math_fidelity = operation_attributes.math_fidelity, .fp32_dest_acc_en = false, .math_approx_mode = false}` (`:207-208`) |

`grep -n opt_level ..._program_factory.cpp` → **no hits**. So every level is the resolved default:
`O2` on the two DM descriptors, **`O3`** on the `ComputeConfigDescriptor`.

Resolved compute config (all six `ComputeConfigDescriptor` fields, defaults from
`tt_metal/api/tt-metalium/program_descriptors.hpp:99-108`):

| field | resolved value | source |
|---|---|---|
| `math_fidelity` | `operation_attributes.math_fidelity` | set explicitly |
| `fp32_dest_acc_en` | `false` | set explicitly (also the default) |
| `math_approx_mode` | `false` | set explicitly (also the default) |
| `dst_full_sync_en` | `false` | struct default |
| `unpack_to_dest_mode` | empty | struct default |
| `bfp8_pack_precise` | `false` | struct default |

This is **Style B** (a Metal `ComputeConfigDescriptor` with literal / computed field values; no
TTNN `ComputeKernelConfig` anywhere in the op).

### CBs

Seven `CBDescriptor`s (`..._program_factory.cpp:121-175`), each with a single
`format_descriptor` (no aliasing), `core_ranges = all_cores`, `.buffer` unset (no
borrowed memory), `.address_offset` unset, `.global_circular_buffer` unset, and `.tile` unset.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`src0`) | `cb0_tiles(2) * in0_single_tile_size` | `all_cores` | `in0_data_format` = dataformat(`a.dtype()`) | `in0_single_tile_size` | not set |
| `c_1` (`src1`) | `cb1_tiles(2) * in1_single_tile_size` | `all_cores` | `in1_data_format` = dataformat(`b.dtype()`) | `in1_single_tile_size` | not set |
| `16` (`output`) | `output_cb_tiles(2) * output_single_tile_size` | `all_cores` | `output_data_format` = dataformat(`output.dtype()`) | `output_single_tile_size` | not set |
| `c_24` (`in0_transposed`) | `interm_cb_size` = `2 * interm_single_tile_size` | `all_cores` | `Float16_b` | `interm_single_tile_size` | not set |
| `c_25` (`in1_transposed`) | `interm_cb_size` | `all_cores` | `Float16_b` | `interm_single_tile_size` | not set |
| `c_26` (`in1_bcast_row`) | `interm_cb_size` | `all_cores` | `Float16_b` | `interm_single_tile_size` | not set |
| `c_27` (`out_transposed`) | `interm_cb_size` | `all_cores` | `Float16_b` | `interm_single_tile_size` | not set |

No GlobalCircularBuffer anywhere in the op (confirmed by re-scan, matching the audit).

### Semaphores

none — `desc.semaphores` is never populated and no kernel constructs a `Semaphore`.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._program_factory.cpp:84` (`TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args)`) | `tensor_args.a` | reader RTA slot 0 (`src0_buffer`, `:224`) |
| `..._program_factory.cpp:85` (`TensorAccessorArgs(src1_buffer).append_to(reader_compile_time_args)`) | `tensor_args.b` | reader RTA slot 1 (`src1_buffer`, `:225`) |
| `..._program_factory.cpp:89` (`TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args)`) | `tensor_return_value` (output) | writer RTA slot 0 (`out_buffer`, `:242`) |

Kernel side: `reader:28-32` (`TensorAccessorArgs<4>()` / chained
`TensorAccessorArgs<src0_args.next_compile_time_args_offset()>()`) → `TensorAccessor s0`, `s1`;
`writer:24-25` (`TensorAccessorArgs<1>()`) → `TensorAccessor s`. All three are **Case 1**
(accessor-mediated); none passes a page-size 3rd argument; no raw-pointer arithmetic anywhere.

### Work split

- Driver: `tt::tt_metal::split_work_to_cores(device_compute_with_storage_grid_size,
  num_output_blocks_total, /*row_major=*/false)` (`..._program_factory.cpp:53-54`), where
  `num_output_blocks_total = bshape[-1] / TILE_WIDTH`.
- Returns `(num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1,
  num_blocks_per_core_group_2)`.
- `g1_numcores = core_group_1.num_cores()` (`:56`); `cores = grid_to_cores(num_cores, grid.x,
  grid.y, /*row_major=*/false)` (`:57-58`) is the per-core RTA iteration order.
- **The per-group difference is carried entirely by RTAs**, not by CTAs: the single loop at
  `:213-253` picks `num_blocks_per_core` from the group and emits it as an RTA. There is **one**
  `KernelDescriptor` per kernel, not one per core group.

### Shared kernels

none. All three kernel sources live in this op's own directory and no other op or test binds
them:

```
grep -rl reader_ssm_eltwise_mul.cpp ttnn/cpp/ttnn/operations/   # only this op's factory
grep -rl writer_ssm_eltwise_mul.cpp ttnn/cpp/ttnn/operations/   # only this op's factory
grep -rl ssm_eltwise_mul.cpp        ttnn/cpp/ttnn/operations/   # only this op's factory
```

No `_metal2` fork exists beside any of them. The port creates no fork, adds no pointer comment,
and carries no sunset list. (The sibling ssm ops — `prefix_scan`, `ssm_1d_sum_reduce` — have
their own private kernel files.)

### Flags

- No unreferenced kernel file in the op directory: all three are bound by the factory
  (`..._program_factory.cpp:179-203`).
- Every legacy descriptor field used maps onto an audit-Appendix-A entry. No out-of-scan
  descriptor type.
- `KernelDescriptor::emplace_runtime_args` with a `Buffer*` in the value list is the
  framework's interim pointer-patching channel (`BufferBinding`,
  `tt_metal/api/tt-metalium/program_descriptors.hpp:114-118`). Metal 2.0's `TensorBinding`
  supersedes it; all three sites are Case 1 and convert cleanly.
- Two known-inert reader oddities, deliberately **not** port work (audit *Misc anomalies*): the
  hardcoded `5120` at `reader:130` where the sibling loop at `reader:91` uses the
  `in0_num_blocks_w` RTA, and reader RTA index 6 (`in0_num_blocks_w`) being dead in Configs A
  and C. Both carried forward verbatim.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none — already the default reflection-based hash.
- **Implementation notes**:
  - `create_descriptor` → `create_program_artifacts(const RepeatMulParams&, const
    RepeatMulInputs&, Tensor&)` returning `ttnn::device_operation::ProgramArtifacts`.
  - No op-owned tensors (no `CBDescriptor` sets `.buffer`; nothing beyond the op's io is
    allocated), so `ProgramArtifacts::op_owned_tensors` is left defaulted.
  - No pybind touches `create_descriptor` (`..._nanobind.cpp:23-32` binds the plain host
    function via `ttnn::bind_function`), so no pybind deletion is forced.
  - `MeshTensor` references are extracted once at the top of the factory
    (`a.mesh_tensor()`, `b.mesh_tensor()`, `output.mesh_tensor()`) and used for
    `tensor_spec()` and the `TensorArgument`s.
  - Spec-name constants are declared **function-local** rather than in a file-scope anonymous
    namespace, so they cannot collide with a sibling ssm factory's constants under a unity
    build.

---

## Planned Spec Shape

Default: 1:1 with legacy. One `ProgramSpec` per cache miss, built by a single
`create_program_artifacts`; the A/B/C configuration affects only `compiler_options.defines`,
one conditional DFB binding, and one `advanced_options` flag (below).

- **KernelSpecs** (3, one per legacy `KernelDescriptor`):
  - `READER{"reader"}` → `reader_ssm_eltwise_mul.cpp`, `hw_config =
    create_reader_datamovement_config(device->arch())`, `compile_time_args` **empty**
    (all four legacy CTAs were CB indices), `runtime_arg_schema.runtime_arg_names =
    {in1_num_blocks, in1_start_id, in1_num_blocks_h, in1_num_blocks_w, in0_num_blocks_w}`,
    `compiler_options.defines` = the config's `REPEAT_*` set, `opt_level` left at Metal 2.0's
    `O2` (matches the resolved legacy DM default).
  - `WRITER{"writer"}` → `writer_ssm_eltwise_mul.cpp`, `hw_config =
    create_writer_datamovement_config(device->arch())`, `compile_time_args` **empty**,
    `runtime_arg_schema.runtime_arg_names = {out_num_blocks_w_per_core, start_id,
    out_num_blocks_h, out_total_blocks_w}`, no defines (matching legacy), `opt_level` `O2`.
  - `COMPUTE{"compute"}` → `ssm_eltwise_mul.cpp`, `hw_config = ComputeGen1Config{...}` (see
    below), `compile_time_args` **empty** (all seven legacy CTAs were CB indices),
    `runtime_arg_schema.runtime_arg_names = {in1_num_blocks, in1_num_blocks_h}`,
    `compiler_options.defines` = the config's `REPEAT_*` set, **`opt_level =
    KernelBuildOptLevel::O3`** — set explicitly, because the legacy `ComputeConfigDescriptor`
    resolved to `O3` while Metal 2.0's `CompilerOptions` defaults to `O2`.

  Compute `hw_config` (Style B — build `ComputeGen1Config` directly, per-field from the legacy
  descriptor; **not** through `to_compute_hardware_config`, whose TTNN-side defaults are the
  high-performance ones):

  | legacy `ComputeConfigDescriptor` | Metal 2.0 `ComputeGen1Config` | value |
  |---|---|---|
  | `math_fidelity` (set) | `fpu_math_fidelity` | `operation_attributes.math_fidelity` |
  | `math_approx_mode = false` (set) | `sfpu_precision_mode` | `Precision::Precise` |
  | `fp32_dest_acc_en = false` (set) | `enable_32_bit_dest` | `false` |
  | `dst_full_sync_en = false` (default) | `double_buffer_dest` | `true` (`= !false`) — Metal 2.0 default, left unset |
  | `bfp8_pack_precise = false` (default) | `bfp_pack_precision_mode` | `Precision::Approximate` — Metal 2.0 default, left unset |
  | `unpack_to_dest_mode` empty (default) | `unpack_modes` | empty, left unset |

  The three explicitly-set legacy fields are written explicitly; the three defaulted ones are
  left at Metal 2.0 defaults, which coincide with the legacy defaults. No `unpack_modes` entry
  is *required*: the rule fires only when `enable_32_bit_dest = true`, and it is `false` here.

- **DataflowBufferSpecs** (7, one per legacy `CBDescriptor`; no aliasing, no borrowed memory,
  `tile_format_metadata` left `nullopt` because the legacy `.tile` was unset):

  | DFBSpecName | legacy CB | `entry_size` | `num_entries` | `data_format_metadata` |
  |---|---|---|---|---|
  | `IN0{"in0"}` | `c_0` | `in0_single_tile_size` | `2` | `in0_data_format` |
  | `IN1{"in1"}` | `c_1` | `in1_single_tile_size` | `2` | `in1_data_format` |
  | `OUT{"out"}` | `16` | `output_single_tile_size` | `2` | `output_data_format` |
  | `IN0_TRANSPOSED{"in0_transposed"}` | `c_24` | `interm_single_tile_size` | `2` | `Float16_b` |
  | `IN1_TRANSPOSED{"in1_transposed"}` | `c_25` | `interm_single_tile_size` | `2` | `Float16_b` |
  | `IN1_BCAST_ROW{"in1_bcast_row"}` | `c_26` | `interm_single_tile_size` | `2` | `Float16_b` |
  | `OUT_TRANSPOSED{"out_transposed"}` | `c_27` | `interm_single_tile_size` | `2` | `Float16_b` |

  Legacy `total_size == num_entries * entry_size` in every row, so the split is exact.

- **SemaphoreSpecs**: none — legacy has no `SemaphoreDescriptor`.

- **TensorParameters** (3, one per distinct originating tensor). Names taken from the kernels'
  own vocabulary (`src0_addr`, `src1_addr`, `dst_addr`):
  - `SRC0{"src0"}` ← `a`, bound on `READER` as `tensor::src0`.
  - `SRC1{"src1"}` ← `b`, bound on `READER` as `tensor::src1`.
  - `DST{"dst"}` ← output, bound on `WRITER` as `tensor::dst`.

  Each has exactly one `TensorBinding`. `relaxations` left default (strict) — the audit records
  `TensorParameter relaxation = none`, and no kernel uses `ArgConfig::Runtime*`.

- **WorkUnitSpecs**: one — `{.name = "main", .kernels = {READER, WRITER, COMPUTE},
  .target_nodes = all_cores}`. All three legacy `KernelDescriptor`s share `all_cores`, so a
  single work unit reproduces placement exactly.

- **Op-owned tensors**: none.

### DFB endpoint dispositions — re-derived from the kernel-touch census

Re-derived per `(DFB, config)` from the kernel sources rather than transcribed from the brief.
A **toucher** is a kernel with a FIFO op or a raw-pointer access; a kernel that merely *names*
the DFB (constructs the wrapper, or passes the id to a format-metadata LLK) still needs its
`dfb::` token to exist, so it must be bound even with no access.

| DFB | reader | writer | compute | Config A | Config B | Config C |
|---|---|---|---|---|---|---|
| `in0` | P (`reader:58,61` / `:86,112,125,150`) | — | C (`compute:45,62` / `:106,121` / `:45,172`) | 1P+1C | 1P+1C | 1P+1C |
| `in1` | P (`reader:65,74`) | — | C (`compute:70,100` / `:70,87`) | 1P+1C | 1P+1C | 1P+1C |
| `out` | — | C (`writer:32,40`) | P (`compute:156,161` / `:81,86`) | 1P+1C | 1P+1C | 1P+1C |
| `in0_transposed` | — (never named) | — | P+C (`:56,61` / `:115,120` produce; `:64,170` / `:123,142` consume) | **self-loop** | **self-loop** | **self-loop** (names only: wrapper `:37`, `pack_reconfig_data_format` `:78`) |
| `in1_transposed` | C (`reader:77` wait, `:78` `get_read_ptr`, `:156` pop) | — | P (`:94,99`) **and** C (`:165` pop) | **1P + 2C → multi-binding flag** | same | reader C + compute P → **1P+1C, no flag** (names only: `reader:36`, `compute:38`) |
| `in1_bcast_row` | P (`reader:83,114,122,152`) | — | C (`compute:126,144`) | 1P+1C | 1P+1C | 1P+1C (names only: `reader:37`, `compute:39`) |
| `out_transposed` | — (never named) | — | P+C (`:135,140` produce; `:147,162` consume) | **self-loop** | **self-loop** | **self-loop** (names only: wrapper `:40`) |

Consequences for construction:

1. **The set of DFBs each kernel binds is the same in all three configs**, because every
   kernel-side `DataflowBuffer` construction is unconditional (`reader:34-37`,
   `compute:34-40`, `writer:27`). So the port adds **no new `#ifdef`s and no new `defines`** —
   the existing `REPEAT_IN0` / `REPEAT_INTERLEAVE_IN1` pair is all that is emitted, exactly as
   legacy emitted it.
2. **Only one binding is config-conditional**: compute's **CONSUMER** endpoint on
   `in1_transposed`, present only under `REPEAT_INTERLEAVE_IN1` — the config in which
   `compute:165` actually pops. It shares the accessor name `in1_transposed` with compute's
   PRODUCER binding, so the kernel still sees exactly one `dfb::in1_transposed` token in every
   config and no kernel-side gating is needed.
3. **`in1_transposed`'s `advanced_options.allow_instance_multi_binding` is gated on the same
   condition** (`REPEAT_INTERLEAVE_IN1`). Under A/B the census is 1 producer + **2 locked
   consumers** (compute's `:165` pop *and* the reader's `:77`/`:156` wait+pop) — no relabelling
   fits 1P+1C, so the flag is required. Under C nobody touches it and the two naming kernels
   split the roles cleanly, so the flag stays off.

**On the audit's *Questions* item 1** (Config-C zero-endpoint CBs: self-loop or drop?). The
audit recommended self-loop and asked the user to confirm; the invoker's go-ahead did not
answer it explicitly, so the plan takes the recommended route — with one census-driven
refinement, and the reasoning recorded here:

- **Route taken: bind, don't drop.** Dropping the four DFBs under Config C would shrink
  Config C's L1 footprint and force `#ifdef` guards onto four kernel-side wrapper
  constructions plus the `compute:78` metadata reference — i.e. a *behavior* change (footprint)
  plus kernel edits the port would otherwise not make. Binding them keeps runtime behavior and
  L1 footprint byte-identical to legacy and needs zero kernel edits, which is what the port's
  no-functional-change premise asks for.
- **Refinement vs. the brief.** The brief says "self-loop each from the kernel that constructs
  its wrapper." That is right for `in0_transposed` and `out_transposed` (compute is the only
  kernel that names them). But `in1_transposed` and `in1_bcast_row` are named by **two**
  kernels under Config C, and the endpoint-assignment procedure is explicit that a self-loop is
  a *one-toucher* resolution: with two candidates you assign **1P+1C**. So under Config C those
  two get one role each — reader CONSUMER + compute PRODUCER for `in1_transposed`, reader
  PRODUCER + compute CONSUMER for `in1_bcast_row` — which also happens to keep their role
  labels identical to Configs A and B. Recorded as a disagreement-with-brief in the port
  report.
- Consequence: `in1_bcast_row`'s bindings are entirely config-independent, and only compute's
  extra CONSUMER endpoint on `in1_transposed` varies.

---

## Preserved Multiplicity

none — no work-split multiplicity in legacy. `split_work_to_cores` yields two core groups, but
the legacy factory emits **one** `KernelDescriptor` per kernel over `all_cores` and carries the
per-group `num_blocks_per_core` difference as a **runtime arg** (`..._program_factory.cpp:212-250`).
So there is nothing to preserve: one `KernelSpec` per kernel, one `WorkUnitSpec`, and the
per-core values stay RTAs. (No CTA is demoted — none of the legacy CTAs were per-group; all
seven/four/one were CB indices.)

---

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._program_factory.cpp:78-83`, reader CTA slots 0–3 | `src0_cb_index`, `src1_cb_index`, `cb_intermed1_index`, `cb_intermed2_index` (magic CB indices) | `DFBBinding`s on `READER` for `IN0`, `IN1`, `IN1_TRANSPOSED`, `IN1_BCAST_ROW` |
| `..._program_factory.cpp:84` | `TensorAccessorArgs(src0_buffer).append_to(reader_compile_time_args)` | `TensorParameter SRC0` + `TensorBinding` on `READER` |
| `..._program_factory.cpp:85` | `TensorAccessorArgs(src1_buffer).append_to(reader_compile_time_args)` | `TensorParameter SRC1` + `TensorBinding` on `READER` |
| `..._program_factory.cpp:86-88`, writer CTA slot 0 | `output_cb_index` (magic CB index) | `DFBBinding` on `WRITER` for `OUT` |
| `..._program_factory.cpp:89` | `TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args)` | `TensorParameter DST` + `TensorBinding` on `WRITER` |
| `..._program_factory.cpp:90-98`, compute CTA slots 0–6 | seven magic CB indices | seven `DFBBinding`s on `COMPUTE` (nine bindings — `IN0_TRANSPOSED` and `OUT_TRANSPOSED` self-loop, plus the conditional `IN1_TRANSPOSED` CONSUMER) |
| `..._program_factory.cpp:224`, reader RTA slot 0 | `src0_buffer` (`Buffer*` — interim pointer-patching channel) | `TensorBinding(SRC0)`; the base address rides the binding's implicit CRTA |
| `..._program_factory.cpp:225`, reader RTA slot 1 | `src1_buffer` (`Buffer*`) | `TensorBinding(SRC1)` |
| `..._program_factory.cpp:242`, writer RTA slot 0 | `out_buffer` (`Buffer*`) | `TensorBinding(DST)` |
| `reader:23-26` | `constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);` … `(3)` | `dfb::in0`, `dfb::in1`, `dfb::in1_transposed`, `dfb::in1_bcast_row` |
| `reader:28-32` | `TensorAccessorArgs<4>()` + chained `next_compile_time_args_offset()`; `src0_addr`/`src1_addr` from `get_arg_val<uint32_t>(0)`/`(1)` | `TensorAccessor(tensor::src0)`, `TensorAccessor(tensor::src1)` |
| `reader:46-47` | `get_tile_size(cb_id_in0)` / `get_tile_size(cb_id_in1)` (cb-id free helper) | `dfb_in0.get_tile_size()` / `dfb_in1.get_tile_size()` |
| `writer:19` | `constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);` | `dfb::out` |
| `writer:23` | `get_tile_size(cb_id_out)` | `dfb_out.get_tile_size()` |
| `writer:24-25` | `TensorAccessorArgs<1>()`; `dst_addr` from `get_arg_val<uint32_t>(0)` | `TensorAccessor(tensor::dst)` |
| `compute:16-22` | `constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);` … `(6)` | `dfb::in0`, `dfb::in1`, `dfb::out`, `dfb::in0_transposed`, `dfb::in1_transposed`, `dfb::in1_bcast_row`, `dfb::out_transposed` |
| `reader:15-21`, `writer:13-17`, `compute:13-14` | positional `get_arg_val<uint32_t>(N)` RTAs | named RTAs via `get_arg(args::<name>)` (schema on `KernelSpec::runtime_arg_schema`, values on `KernelRunArgs::runtime_arg_values`) |

Post-port CTA count: **zero on all three kernels** — every legacy CTA was a CB index or
`TensorAccessorArgs` plumbing.

RTA name mapping (positional → named), with the two buffer-address slots dropped:

| kernel | legacy slot | named RTA |
|---|---|---|
| reader | 0, 1 | *dropped* (`TensorBinding`) |
| reader | 2 | `in1_num_blocks` |
| reader | 3 | `in1_start_id` |
| reader | 4 | `in1_num_blocks_h` |
| reader | 5 | `in1_num_blocks_w` |
| reader | 6 | `in0_num_blocks_w` |
| writer | 0 | *dropped* (`TensorBinding`) |
| writer | 1 | `out_num_blocks_w_per_core` |
| writer | 2 | `start_id` |
| writer | 3 | `out_num_blocks_h` |
| writer | 4 | `out_total_blocks_w` |
| compute | 0 | `in1_num_blocks` |
| compute | 1 | `in1_num_blocks_h` |

Names are the kernels' own existing local-variable names, so the kernel diff is a retrieval-syntax
swap with no renaming. No vararg is used: every RTA is a distinct field read once at a literal
index.

---

## Applied Patterns

- **Self-loop DFB binding** / **Sync-free and single-ended CBs → self-loop DFB** —
  `IN0_TRANSPOSED` and `OUT_TRANSPOSED` on `COMPUTE`, bound PRODUCER **and** CONSUMER under one
  accessor name. Genuine accumulator-shaped self-loops in Configs A/B (compute really does
  `reserve_back`/`push_back` then `wait_front`/`pop_front`); name-only in Config C.
- **Two-toucher DFB → assign 1P+1C** — used as the *procedure* to re-derive every disposition,
  and decisive for `IN1_TRANSPOSED` / `IN1_BCAST_ROW` under Config C, where two kernels name a
  DFB that neither touches: 1P+1C, not two self-loops.
- **Multi-binding advanced option** — `IN1_TRANSPOSED` under Configs A/B only
  (`advanced_options.allow_instance_multi_binding = true`): a self-popping producer
  (`compute:94,99` push + `compute:165` pop) alongside the reader's genuine consume
  (`reader:77,156`) is 1P + 2 locked consumers, which no relabelling reduces to 1P+1C. The flag
  self-documents the Quasar debt.
- **Pass DFB handles directly to LLKs** — every compute LLK call site
  (`compute_kernel_hw_startup`, `transpose_init`, `transpose_tile`, `mul_init`, `mul_tiles`,
  `mul_bcast_rows_init`, `mul_tiles_bcast_rows`, `pack_tile`, `reconfig_data_format*`,
  `pack_reconfig_data_format`) takes `dfb::<name>` directly via
  `DFBBindingToken::operator uint32_t()`. All eleven callees take plain `uint32_t` operands
  (verified in `tt_metal/hw/inc/api/compute/`), so no `.id` extraction and no temporary
  `DataflowBuffer` is needed anywhere. (The first, fourth and sixth of those names arrived with
  the rebase — see the port report's *Rebase onto main* section; upstream renamed the init APIs
  while this port renamed their operands.)
- **DFB metadata via the object** — `get_tile_size(cb_id)` → `dfb.get_tile_size()` at
  `reader:46-47` and `writer:23`. The member getter is gated on `DFB_DESCRIPTORS_DEFINED`,
  which keys on the same generated `chlkc_descriptors.h` as the legacy free helper's
  `DATA_FORMATS_DEFINED`, so it is available on the DM path exactly where the free helper was.
- **Conditional / optional DFB bindings** — invoked only in its host-side half: compute's
  conditional CONSUMER endpoint on `IN1_TRANSPOSED`. No kernel-side `#ifdef` is needed because
  the accessor name is shared with the unconditional PRODUCER binding, so `dfb::in1_transposed`
  exists in every config.
- **Unity-build hygiene** — spec-name constants are function-local (not in a file-scope
  anonymous namespace), so they cannot collide with a sibling ssm factory's under a unity build.

Not applied, and why: *Aliased DFBs* (no legacy multi-`format_descriptor` CB); *Same-FIFO
aliasing* (no kernel-side `uint32_t` CB alias, no host-side index mirroring); *Multi-variant
factories* (single variant — the A/B/C fork is a `defines` fork inside one spec build, not a
per-variant spec); *Demoting per-group CTA to RTA* (no per-group CTA exists); *Removing pybound
legacy factory entry points* (no pybind touches `create_descriptor`); *Porting a shared kernel*
(no shared kernel).

---

## Deferred / Flagged

- **New findings during planning: none structural.** No feature gate fired that the audit
  missed; no construct resisted a binding-token replacement; nothing reached outside the op
  directory.
- One **disagreement with the brief**, resolved by the census and recorded above: the brief's
  blanket "self-loop each from the constructing kernel" for the four Config-C zero-endpoint CBs
  over-applies the one-toucher resolution to two DFBs that are named by *two* kernels. Those
  two are 1P+1C under Config C. Carried to the port report.
- The audit's *Questions* item 1 was answered by the plan (bind rather than drop) because the
  invoker's go-ahead did not address it; the choice is the audit's own recommendation and is
  the zero-functional-change option. Flagged in the report so the op owner can still choose the
  host-side conditional-allocation end state on their own track.
- The audit's *Questions* item 2 (removing the redundant `compute:165` pop, which would collapse
  `IN1_TRANSPOSED` to a plain 1:1 and retire the multi-binding flag) is a behavior change and
  explicitly out of port scope. The flag is set as-is.
- Node-invariant RTAs that are morally CRTAs — reader `in1_num_blocks_h`, `in1_num_blocks_w`,
  `in0_num_blocks_w`; writer `out_num_blocks_h`, `out_total_blocks_w`; compute
  `in1_num_blocks_h` — are kept as RTAs. RTA→CRTA changes dispatch semantics and is a separate
  cleanup, not port work. Noted in the report.
