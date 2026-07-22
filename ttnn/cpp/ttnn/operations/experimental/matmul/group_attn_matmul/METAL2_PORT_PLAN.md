# Port Plan — `experimental/matmul/group_attn_matmul`

Port plan for `group_attn_matmul`, ported from the legacy `ProgramDescriptor` (`descriptor`
concept) API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Audit: **GREEN** (see `METAL2_PREPORT_AUDIT.md`). Brief: `METAL2_PORT_BRIEF.md`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `GroupAttnMatmulProgramFactory::create_descriptor`
  returns `tt::tt_metal::ProgramDescriptor` (`group_attn_matmul_program_factory.hpp:21`).
- Variants: single (one `GroupAttnMatmulProgramFactory` in the `program_factory_t` variant).
- Custom `compute_program_hash`: **none** — the device-op declares only
  `validate_on_program_cache_miss` / `compute_output_specs` / `create_output_tensors`
  (`group_attn_matmul_device_operation.hpp`). The factory header/body comments *mention*
  `compute_program_hash()` (`...factory.hpp:17`, `...factory.cpp:158`) but no override exists;
  the op uses the default reflection hash. (Audit "Misc anomalies": misleading comment.)

*(Metal 2.0 target concept `MetalV2FactoryConcept` chosen during the audit; carried forward below.)*

**Config axes (single source per kernel — NOT runtime source selection).** One reader, one
writer, one compute source. The four configs (interleaved / IN0_SHARDED / IN1_SHARDED /
OUT_SHARDED) are selected by `defines` (+ CB `.buffer` borrowing), all within the same three
sources. So the atomic port unit is **1 factory + 3 kernel sources**; there is no per-config
source fan-out.

### Kernels
All three sources are op-owned, in-directory, and already on the Device 2.0 object API
(`Noc`, `CircularBuffer`, `Semaphore<>`, `TensorAccessor`). All on `all_device_cores`.

| unique_id | source | core_ranges | CTAs (positional) | RTAs (positional; key ones) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp` | `all_device_cores` | `[0]transpose_hw` `[1]row_major` `[2]out_subblock_w` `[3+]TensorAccessorArgs(src1_buffer)` (`#ifndef IN1_SHARDED`) | `has_work_for_mcast_kv_heads, has_work_for_q_heads, `**`src1_buffer(addr)`**`, Mt, Nt, num_kv_heads, in1_CKtNt, in1_CKtNt*32, blocks, in1_start_id, in0_block_w, out_block_w, in1_num_subblocks, in1_num_blocks, in1_block_num_tiles, Nt_bytes, in1_block_w_tile_bytes, out_last_subblock_w, in1_last_block_w_tile_read_bytes, in1_last_block_addr_skip, mcast_dest_noc_{start,end}_{x,y}, in1_mcast_num_dests, in1_mcast_num_cores, in1_mcast_grid_size, `**`sender_sem_id`**`, `**`receiver_sem_id`**`, in1_mcast_sender_size_bytes, in1_mcast_sender_id, num_x, num_y, `**`[noc_x tail][noc_y tail]`** | `IN1_SHARDED` (if in1 sharded) | `DataMovementConfigDescriptor{RISCV_1, reader_noc}` |
| writer | `device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp` | `all_device_cores` | `[0]`**`output_cb_index (c_5, MAGIC)`** `[1]out_subblock_w` `[2]intermediate_num_tiles` `[3+]TensorAccessorArgs(src0_buffer)` (`#ifndef IN0_SHARDED`) `[..]TensorAccessorArgs(dst_buffer)` (`#ifndef OUT_SHARDED`) | `has_work_for_q_heads, `**`src0_buffer(addr)`**`, `**`dst_buffer(addr)`**`, Mt, Kt, Nt, MtKt, blocks, in0_start_id, out_start_id, in0_block_w, in1_num_subblocks, in1_num_blocks, out_num_tiles, bfloat16_row_bytes, bfloat16_Nt_bytes, bfloat16_last_row_bytes_read` | `IN0_SHARDED`, `OUT_SHARDED` (conditional) | `DataMovementConfigDescriptor{RISCV_0, writer_noc}` |
| compute | `device/kernels/compute/transformer_group_attn_matmul.cpp` | `all_device_cores` | `[0]transpose_hw` `[1]out_subblock_w` `[2]out_subblock_num_tiles` `[3]intermediate_num_tiles` | `has_work_for_q_heads, batch(=num_output_blocks_per_core), Mt, num_kv_heads_skip, num_kv_heads_remaining, in0_block_w, out_subblock_h, in1_num_subblocks, in1_num_blocks, in0_block_num_tiles, in1_block_num_tiles, out_num_tiles(=MtNt), in0_subblock_num_tiles, in1_per_core_w` | none | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |

**DM config resolution.** `reader_noc = detail::preferred_noc_for_dram_read(arch)`, which is
`NOC_0` on Gen1 (WH/BH; `kernel_types.hpp:134`). `writer_noc` is the complement (`NOC_1`).
`noc_mode` defaults to `DM_DEDICATED_NOC`. So on Gen1 the resolved triples are exactly the
reader/writer defaults → use the arch-agnostic TTNN helpers.

**Bold RTAs above evaporate** (buffer-address RTAs → `TensorBinding`; semaphore-id RTAs →
`SemaphoreBinding`; noc x/y tail → runtime varargs). See Dropped Plumbing.

### CBs
Legacy `CBDescriptor`s (`...factory.cpp:163-245`). `entry_size` = page_size; `num_entries` =
total_size / page_size. No `.tile` set on any (→ `tile_format_metadata` unset, standard 32×32).

| CB | buffer_index | entry_size | num_entries | data_format | borrowed (`.buffer`) |
|---|---|---|---|---|---|
| c_0 in0 | 0 | `in0_single_tile_size` | `in0_is_sharded ? shardnumel/TILE_HW : in0_block_w` | `in0_data_format` | `src0_buffer` iff in0 sharded |
| c_1 in1 | 1 | `in1_single_tile_size` | `2*in1_block_num_tiles` | `in1_data_format` | none |
| c_2 in1-sharded | 2 | `in1_single_tile_size` | `shardnumel/TILE_HW` | `in1_data_format` | `src1_buffer` (**only created if in1 sharded**) |
| c_3 intermed0 | 3 | `interm_single_tile_size` | `2*intermediate_num_tiles` | `interm_data_format` | none |
| c_4 intermed1 | 4 | `interm_single_tile_size` | `MtNt` | `interm_data_format` | none |
| c_5 out | 5 | `output_single_tile_size` | `output_is_sharded ? shardnumel/TILE_HW : MtNt` | `output_data_format` | `dst_buffer` iff out sharded |

`interm_data_format = (fp32_dest_acc_en && in0==Float32) ? Float32 : Float16_b`.
No `GlobalCircularBuffer`, no aliased CBs (all single-element `format_descriptors`), no
`address_offset`. Confirmed by audit.

### Semaphores
| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (sender) | WORKER | `all_device_cores` | `INVALID` |
| 1 (receiver) | WORKER | `all_device_cores` | `INVALID` |

`INVALID == 0` (`hostdevcommon/common_values.hpp:13`), so initial value = 0 = Metal 2.0
`SemaphoreSpec` default. **No non-zero init needed** (would have required the deprecated
`SemaphoreAdvancedOptions.initial_value`). Both semaphores are used only by the reader.

### Tensor accessors
| host site (file:line) | originating Tensor | kernel | RTA slot (addr) |
|---|---|---|---|
| `reader...cpp:83` `TensorAccessor(in1_args, src1_addr)` | `input_tensor_b` (in1) | reader | reader RTA `src1_addr` |
| `writer...cpp:65` `TensorAccessor(in0_args, src0_addr)` | `input_tensor_a` (in0) | writer | writer RTA `src0_addr` |
| `writer...cpp:71` `TensorAccessor(out_args, dst_addr)` | `output` | writer | writer RTA `dst_addr` |

All 2-arg (no 3rd page-size arg). Each accessor is built only on the tensor's **interleaved**
code path (`#ifndef *_SHARDED`); on the sharded path the tensor is reached via a borrowed DFB.

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_active_cores, row_major)`,
  `num_active_cores = max(Q_HEADS, TILE_HEIGHT)` (`...factory.cpp:97-106`).
- `TT_FATAL(num_output_blocks_per_core_group_1 == 1 && group_2 == 0)` — group 1 does 1 block/core,
  group 2 empty. **Single `KernelDescriptor` per kernel** (no per-group multiplicity); per-core
  values vary via RTAs over `grid_to_cores_with_noop(...)` for the whole grid. Inactive cores
  early-return via the `has_work_*` RTAs.

### Cross-op kernels
None — all three sources are op-owned and in-directory.

### Flags
- **Writer dead metadata read.** `writer...cpp:59` `const uint32_t in1_tile_bytes =
  get_tile_size(cb_id_in1);` is never used, and the writer does not otherwise touch `c_1`. It
  cannot be ported (`c_1` has no writer binding; rule 7 needs `dfb::in1.get_tile_size()` which
  needs a binding — one that would be wrong to add). Dropping the dead line is behavior-preserving
  and forced. → Dropped Plumbing / report.
- **Unused `onetile` constant** in writer (`:57`) and compute (`:60`) — pre-existing dead
  constant (audit Misc anomalies). Left as-is (not touched by the port).
- **`c_2` read is unconditional in legacy but `c_2` only exists in IN1_SHARDED.** `reader...cpp:123`
  reads `cb_in2_obj.get_read_ptr()` unconditionally; the value is dead in interleaved (only used
  under `#ifdef IN1_SHARDED` and inside a runtime branch that is false in interleaved). The port
  gates the `dfb::in2` construction+read under `#ifdef IN1_SHARDED` (conditional-binding pattern),
  initializing the address to 0 in interleaved (behavior-preserving; the value was dead there).

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept` — `create_program_artifacts`.
- **Custom `compute_program_hash`**: none (nothing to delete).
- **Forced device-op-class edits**: **none.** No custom hash, no pybind `create_descriptor`
  (`group_attn_matmul_nanobind.cpp` binds only the op function), no `override_runtime_arguments`,
  no pybind-hook-only factory parameter. The device-operation class is untouched; only the factory
  `.hpp`/`.cpp` and the three kernels change.
- **Implementation notes**: `create_program_artifacts` extracts `MeshTensor`s from
  `tensor_args`/`tensor_return_value`, builds the spec + run-args, returns `ProgramArtifacts`
  (no `op_owned_tensors`). Per-config branching (`in0_is_sharded` / `in1_is_sharded` /
  `output_is_sharded`) selects `borrowed_from` vs `TensorBinding` and the matching `defines`.

## Planned Spec Shape

- **KernelSpecs**: 3 — `reader`, `writer`, `compute` (1:1 with legacy `KernelDescriptor`s).
- **DataflowBufferSpecs**: 6 — `IN0`(c_0), `IN1`(c_1), `IN2`(c_2, **conditional** on in1 sharded),
  `INTERMED0`(c_3), `INTERMED1`(c_4), `OUT`(c_5). `IN0`/`IN2`/`OUT` set `borrowed_from` in their
  sharded configs; `IN2` is only declared+bound when in1 sharded.
- **SemaphoreSpecs**: 2 — `SENDER_SEM`(id0), `RECEIVER_SEM`(id1), both `target_nodes =
  all_device_cores`, both bound to reader only.
- **TensorParameters**: 3 — `T_IN0`(input_tensor_a), `T_IN1`(input_tensor_b), `T_OUT`(output).
  Always declared + always supplied as `TensorArgument`s. Bound via `TensorBinding` on the
  interleaved path; referenced via DFB `borrowed_from` on the sharded path (both count as a use —
  `program_spec.cpp:533-543`).
- **WorkUnitSpecs**: 1 — `{reader, writer, compute}` over `all_device_cores`.

**DFB endpoint roles (per-node census; matches audit/brief):**

| DFB | producer | consumer | notes |
|---|---|---|---|
| IN0 (c_0) | writer | compute | writer reserve/push (fills in interleaved; sync-only in IN0_SHARDED) |
| IN1 (c_1) | reader | compute | writer's dead `get_tile_size(c_1)` dropped → not a toucher |
| IN2 (c_2) | reader (P+C **self-loop**) | — | single toucher (raw `get_read_ptr`); IN1_SHARDED only; borrowed_from(T_IN1) |
| INTERMED0 (c_3) | compute | writer | |
| INTERMED1 (c_4) | writer | compute | |
| OUT (c_5) | compute | writer | writer consumes (`wait_front`/`pop_front`) |

**Kernel↔DFB accessor names**: dfb::in0, in1, in2, intermed0, intermed1, out.
**Tensor accessor names**: tensor::src0 (T_IN0, writer), src1 (T_IN1, reader), dst (T_OUT, writer).
**Semaphore accessor names**: sem::sender_sem, sem::receiver_sem (reader).

**Compute `hw_config` (ComputeGen1Config).** Legacy `ComputeConfigDescriptor` sets ONLY
`math_fidelity` + `fp32_dest_acc_en`, leaving `dst_full_sync_en`/`math_approx_mode`/`bfp8_pack_precise`
at their `false` defaults (`program_descriptors.hpp:101-106`). The legacy destructures
`math_approx_mode`/`dst_full_sync_en` from `get_compute_kernel_config_args` but does **not** wire
them into the descriptor. So the faithful Gen1 config sets only:
- `fpu_math_fidelity = math_fidelity`
- `enable_32_bit_dest = fp32_dest_acc_en`
- (`sfpu_precision_mode = Precise`, `double_buffer_dest = true`, `bfp_pack_precision_mode =
  Approximate` — all ComputeGen1Config defaults, matching the legacy descriptor defaults.)

⚠ Do **not** use `to_compute_hardware_config(arch, config)` unqualified: it maps the config's
`math_approx_mode`/`dst_full_sync_en`, which the legacy op ignored — honoring them would silently
change precision/perf for a caller who sets them. See report Friction.

**Compute `unpack_modes` (FP32 required entries).** The Metal 2.0 validator requires an explicit
`unpack_modes` entry for every Float32-formatted DFB a compute kernel *consumes* when
`enable_32_bit_dest = true` (legacy defaulted silently). Compute consumes IN0/IN1/INTERMED1. Legacy
`unpack_to_dest_mode` is empty → all `Default` → `UnpackToSrc`. So, at spec-construction time, for
each of IN0/IN1/INTERMED1 whose resolved data_format is Float32 (and fp32_dest_acc_en true), add
`{DFB, UnpackMode::UnpackToSrc}`. Needed for `test_group_attn_matmul_fp32`.

## Preserved Multiplicity
None — no work-split `KernelDescriptor` multiplicity in legacy (single descriptor per kernel;
per-core variation is RTA-only; `TT_FATAL` pins 1 block/core in group 1, group 2 empty).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA `src1_addr` (`reader:24`) + CTA `TensorAccessorArgs(src1_buffer)` (`factory:253`) | buffer-addr RTA + TA-args plumbing | `TensorBinding` T_IN1 (reader, `tensor::src1`), interleaved path |
| writer RTA `src0_addr` (`writer:20`) + CTA `TensorAccessorArgs(src0_buffer)` (`factory:260`) | buffer-addr RTA + TA-args | `TensorBinding` T_IN0 (writer, `tensor::src0`), interleaved path |
| writer RTA `dst_addr` (`writer:21`) + CTA `TensorAccessorArgs(dst_buffer)` (`factory:261`) | buffer-addr RTA + TA-args | `TensorBinding` T_OUT (writer, `tensor::dst`), interleaved path |
| writer CTA `[0] output_cb_index` (`factory:256`, `writer:41`) | magic CB index in CTA | `DFBBinding` OUT (writer CONSUMER, `dfb::out`) |
| reader `cb_id_in1=1`, `cb_id_in2=2` (`reader:70-71`); compute/writer `tt::CBIndex::c_*` | hardcoded magic CB indices | `dfb::name` handles |
| reader RTA `sender_sem_id`, `receiver_sem_id` (`reader:54-55`, `factory:389-390`) | semaphore-id RTAs | `SemaphoreBinding` (reader `sem::sender_sem`/`sem::receiver_sem`) |
| reader RTA tail `in1_mcast_sender_noc_x[]`,`_y[]` (`reader:61-64`, `factory:396-399`) | `get_arg_addr`+pointer array | runtime **varargs** (`num_runtime_varargs = num_x+num_y`); `get_vararg(x)` / `get_vararg(num_x+y)` |
| all positional CTAs (reader/writer/compute) | positional `compile_time_args` | named CTAs (`get_arg(args::name)`) |
| all positional RTAs (reader/writer/compute) | positional `get_arg_val<uint32_t>(i++)` | named RTAs (`get_arg(args::name)`) |
| writer `get_tile_size(cb_id_in1)` (`writer:59`) | dead metadata read of unbound CB | **dropped** (unused; can't bind c_1 to writer) |

## Applied Patterns
- **[Sync-free / single-ended → self-loop DFB]**: `IN2` (c_2), IN1_SHARDED only — single raw
  `get_read_ptr` toucher (reader); bind reader PRODUCER+CONSUMER.
- **[Conditional / optional DFB bindings]**: `IN2` — declared, bound, and referenced only under
  `#ifdef IN1_SHARDED` (the define already emitted by the legacy factory; reused). Kernel gates the
  `dfb::in2` alias + read.
- **[Borrowed-memory DFBs]**: `IN0`←T_IN0 (IN0_SHARDED), `IN2`←T_IN1 (IN1_SHARDED), `OUT`←T_OUT
  (OUT_SHARDED). Conditional `borrowed_from`; backing address auto-resolves from the TensorArgument.
- **[Conditional tensor bindings]**: T_IN0/T_IN1/T_OUT `TensorBinding`s emitted only on the
  interleaved path of their kernel (`#ifndef *_SHARDED`), matching where the kernel builds the
  `TensorAccessor`. On the sharded path the tensor is used via `borrowed_from`.
- **[Pass DFB handles directly to LLKs]**: compute passes `dfb::in0/in1/intermed0/intermed1/out`
  directly to `matmul_tiles`, `matmul_init`, `pack_untilize_*`, `tilize_*`, `reconfig_*`,
  `compute_kernel_hw_startup` via the implicit `DFBAccessor→uint32_t` conversion.
- **[Runtime varargs]**: reader mcast sender noc x/y tail (loop-read with runtime index).

## Deferred / Flagged
- **FP32 `unpack_modes`** required entries — new Metal 2.0 validator requirement not present in
  legacy; expected port work per the recipe's compute section (not a stop). Implemented as above.
- **Compute config knobs dropped by legacy** (`math_approx_mode`, `dst_full_sync_en`) — the recipe's
  "use `to_compute_hardware_config`" happy path doesn't fit; building `ComputeGen1Config` directly
  to stay faithful. Report Friction.
- **Writer dead `get_tile_size(c_1)`** — forced drop (see Flags). Report.
- **RTA→CRTA-vararg opportunity**: the reader's noc x/y tail is identical on every node (a mcast
  sender grid constant), so it is really a *common* vararg. Left as per-node runtime varargs to
  avoid changing dispatch semantics during the port. Report Open items.
- No new structural blockers surfaced during planning.
