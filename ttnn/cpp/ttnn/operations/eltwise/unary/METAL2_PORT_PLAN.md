# Port Plan — `ttnn/cpp/ttnn/operations/eltwise/unary`

Port plan for `eltwise/unary`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Inputs consumed: `METAL2_PORT_BRIEF.md` (actionable), `METAL2_PREPORT_AUDIT.md` (detail),
`analyses/relaxations/eltwise_unary.md` (relaxation declaration).

---

## Legacy Inventory

### Legacy factory shape

- **Concept**: `ProgramDescriptorFactoryConcept` — `ProgramFactory::create_descriptor` returns
  `tt::tt_metal::ProgramDescriptor` (`device/unary_device_operation.hpp:44`,
  `device/unary_program_factory.cpp:337`). A `void`-returning
  `ProgramFactory::override_runtime_arguments` sits beside it (`hpp:51`, `cpp:570`).
- **Variants**: single — `using program_factory_t = std::variant<ProgramFactory>;`
  (`hpp:59`), no `select_program_factory`.
- **Where the factory methods live**: in a nested `ProgramFactory` struct, **not** directly on
  the device-operation struct. So this is *not* the direct-descriptor shape and
  `ttnn_factory.md` exception 3 does not apply.
- **Custom `compute_program_hash`**: present at `device/unary_device_operation.cpp:179`, plus
  the backdoor `operation_attributes_t::to_hash()` at `:16` — **both left intact.** Recorded so
  the port knows not to touch them, and so a `TensorSpec` legality failure has a named suspect.
  The comment block at `:197-213` was written about the Metal 2.0 relaxation contract.

### Kernels

Three `KernelDescriptor`s, pushed in this order (the indices `override_runtime_arguments`
addresses positionally today: reader 0, writer 1, compute 2).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary.cpp` | `operation_attributes.worker_grid` | `TensorAccessorArgs(*src_buffer, ArgConfig::RuntimeTensorShape)` payload only (`:462-463`) | none | per-core; **3** slots when `has_sharding` else **8**: `0` = `input.buffer()` (`Buffer*`), `1` = `in_units`, `2` = `start_id`, `3..7` = RM chunk constants (`rm_interleaved`) or literal zeros (`:531,536,555`) | accessor payload (`:472`) | `SRC_SHARDED`, `RM_INTERLEAVED` | absent → **O2** (DM default) | `ReaderConfigDescriptor{}` → resolved `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` = reader default |
| writer | `device/kernels/dataflow/writer_unary.cpp` | same | `TensorAccessorArgs(*dst_buffer, …RuntimeTensorShape)` payload only (`:481-482`) | none | same shape; `0` = `output.buffer()`, `1` = `out_units`, `2` = `start_id`, `3..7` = output-side chunk constants or zeros (`:532,546,557`) | accessor payload (`:491`) | `DST_SHARDED`, `RM_INTERLEAVED` | absent → **O2** | `WriterConfigDescriptor{}` → resolved `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` = writer default |
| compute | runtime-selected, 9 sources (below) | same | HARDSWISH → `{is_float32, is_int}` (`:499-502`); LOGIT → `{logit_clamp_enabled}` (`:504`); **then always** `static_cast<uint32_t>(cb_data_format)` (`:506`) | none | per-core, **3** slots: `{compute_units, packed_scalar1, packed_scalar2}` (`:559-560`) | none | `get_block_defines(ops_chain, "0", "0", input.dtype())` + `add_input_dtype_defines(...)` (`:407-408`) | absent → **O3** (`ComputeConfigDescriptor` default) | `ComputeConfigDescriptor{HiFi4, fp32_dest_acc_en, unpack_to_dest_mode, bfp8_pack_precise, math_approx_mode=false}` (`:508-514`) |

`opt_level` is confirmed absent op-wide: `grep -rn 'opt_level' <op-dir>` returns nothing.

### Runtime kernel-source selection

The compute `KernelDescriptor`'s `kernel_source` is chosen at runtime by
`get_compute_kernel_path(ops_chain[0].type(), input.dtype())`
(`common/unary_op_utils.cpp:1190-1209`). **One axis, nine reachable sources** — so the port's
atomic unit is this one factory plus all nine:

| op_chain[0].type() | input dtype | source |
|---|---|---|
| `LGAMMA` | `BFLOAT16` | `lgamma_fast_kernel.cpp` |
| `LGAMMA` | other | `lgamma_kernel.cpp` |
| `IDENTITY` | — | `eltwise_identity_kernel.cpp` |
| `WHERE_TSS` | — | `where_tss_kernel.cpp` |
| `MAC_TSS` | — | `mac_tss_kernel.cpp` |
| `LOGIT` | — | `logit_kernel.cpp` |
| `HARDSWISH` | — | `hardswish_kernel.cpp` |
| `LOGSIGMOID` | — | `logsigmoid_kernel.cpp` |
| default | — | `eltwise_sfpu.cpp` |

Per-DFB producer/consumer roles are **identical across all nine** selected sources — every one
consumes `c_0` and produces `c_2`; only `logit_kernel.cpp` additionally touches `c_1`. No role
moves between kernels across paths (the reader always fills `c_0`; there is no row-major
`tilize` path where compute would become `c_0`'s producer).

### CBs

Three `CBDescriptor`s. None sets `format_descriptors[i].tile`, so `tile_format_metadata` stays
`nullopt` on all three — this is the non-32×32 mis-sizing defect, preserved deliberately.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` src0 (`:420`) | `input_cb_page_size * src_num_tiles_per_shard.value_or(2)` | `worker_grid` | `cb_data_format_for_input` (= output format for `BITCAST`, else input format) | `input_cb_page_size` = `tile_size(input_df)` | not set |
| `c_1` tmp0 (`:432`, **LOGIT only**, gated by `needs_tmp0_cb`) | `input_cb_page_size * 2` | `worker_grid` | `cb_data_format` (input format) | `input_cb_page_size` | not set |
| `c_2` out (`:444`) | `output_cb_page_size * dst_num_tiles_per_shard.value_or(2)` | `worker_grid` | `cb_data_format_output` | `output_cb_page_size` = `tile_size(output_df)` | not set |

`.buffer` is set on two of them — `c_0` ← `src_buffer` when `src_sharded` (`:428`), `c_2` ←
`dst_buffer` when `dst_sharded` (`:452`) — the borrowed-memory idiom. No
`GlobalCircularBuffer`, no `address_offset`, no multi-element `format_descriptors` (so no
aliasing).

### Semaphores

**none** — the op uses no semaphores of any kind.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/unary_program_factory.cpp:462-463` | `tensor_args.input` (via `*src_buffer`) | reader slot 0 (`Buffer*` on miss, raw `uint32_t` on hit at `:613`) |
| `device/unary_program_factory.cpp:481-482` | `output` (via `*dst_buffer`) | writer slot 0 (`Buffer*` on miss, raw at `:616`) |

Device sites: `reader_unary.cpp:25-26` (`TensorAccessorArgs<0,0>()` + `TensorAccessor(src_args,
src_addr)`, inside the `#else` of `#if SRC_SHARDED`), `writer_unary.cpp:27-28` (mirror, under
`DST_SHARDED`). Both are two-argument constructions — **no third (page-size) argument anywhere
in the op.** Both are Case 1 (all access through the accessor; no raw base pointer).

### Work split

Driven by `enumerate_core_rt_args` (`device/unary_program_factory.cpp:127-333`), shared by
`create_descriptor` and `override_runtime_arguments` so the miss and hit paths cannot drift.
Two regimes:

- **`has_sharding`**: no `split_work_to_cores`. `core_group_1` *is* the input shard grid;
  per-core tile counts come from `compute_shard_pages`, and the core vector from
  `grid_to_cores_with_noop(...)` (`:265-273`).
- **interleaved**: `split_work_to_cores(compute_with_storage_grid | all_device_cores,
  out_num_tiles, row_major)` (`:303`, `:308`) →
  `(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1,
  num_tiles_per_core_group_2)`; core vector from `grid_to_cores(...)` / `corerange_to_cores(...)`.

Either way the callback is invoked for **every** core in `worker_grid` — `num_cores_total`
equals `worker_grid.num_cores()` on both branches — with `noop = true` for cores outside the
work set. That is what makes the uniform per-core arg write possible.

### Shared kernels

Census run per the shared-kernel Caution: `grep -rl <filename> ttnn/cpp/ttnn/operations/`, then
each hit disambiguated by the **bound path**.

| kernel | direction | consumers that will not convert here | `_metal2` fork beside it? | rung |
|---|---|---|---|---|
| `device/kernels/compute/eltwise_sfpu.cpp` | **lent** | 3 external C++ factories — `operations/examples/example/device/single_core_program_factory.cpp:91`, `.../multi_core_program_factory.cpp:89`, `operations/examples/example_multiple_return/device/single_core_program_factory.cpp:80` — plus 2 tests: `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246`, `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436` | **no** | **2 — create the fork** (`eltwise_sfpu_metal2.cpp`, beside the original in this op's own directory) + pointer comment in the original |

**Everything else is unary-exclusive** and converts in place:
`reader_unary.cpp`, `writer_unary.cpp`, and the other eight compute kernels
(`eltwise_identity_kernel`, `hardswish_kernel`, `lgamma_kernel`, `lgamma_fast_kernel`,
`logit_kernel`, `logsigmoid_kernel`, `mac_tss_kernel`, `where_tss_kernel`).

Disambiguation that mattered: `reader_unary.cpp` / `writer_unary.cpp` produce 17 / 43 hits
outside the op directory, and **every one of them binds
`tests/tt_metal/tt_metal/test_kernels/dataflow/{reader,writer}_unary.cpp`** — same-named private
copies, not this op's files. The op's own paths appear nowhere outside the op directory. The
eight other compute kernels' only external hit is `ttnn/ttnn.egg-info/SOURCES.txt`, a build
artifact.

### Factory variants

Single variant — no per-variant inventory needed.

### Flags

- **Unreferenced kernel files in this directory (not audited, not this op's to touch).** Nine
  dataflow kernels in `device/kernels/dataflow/` are never instantiated by
  `UnaryDeviceOperation`; they are lent to other families:
  `reader_unary_interleaved_start_id.cpp`, `reader_unary_interleaved_start_id_metal2.cpp`,
  `reader_unary_interleaved_col_multicore.cpp`, `reader_unary_interleaved_wh_multicore.cpp`,
  `reader_unary_sharded.cpp`, `reader_unary_sharded_metal2.cpp`,
  `writer_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id_metal2.cpp`,
  `writer_unary_interleaved_start_id_wh.cpp`.
- **`_metal2` name-adjacency trap, confirmed.** Three of those are `*_metal2.cpp` files in the
  directory the port edits, and **none forks `reader_unary.cpp` or `writer_unary.cpp`** — the
  fork test is per-stem. They are read as idiom precedent only.
- **Two sanctioned Device 2.0 free-function sites** that become port work under kernel-side
  whitelist rule 7: `get_local_cb_interface(cb_id_src).fifo_page_size` (`reader_unary.cpp:57`)
  and `get_local_cb_interface(cb_id_dst).fifo_page_size` (`writer_unary.cpp:60`).
- **No descriptor type outside the audit's scan** appears in the factory.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: **`CustomProgramSpecFactoryConcept`** — selected because
  the ported-from factory has an `override_runtime_arguments`
  (`device/unary_program_factory.cpp:570`). The port *translates* that method into one returning
  a `ProgramRunArgs`; it is not deleted. No disagreement with the audit's choice.
- **Custom `compute_program_hash`**: present at `device/unary_device_operation.cpp:179` (plus
  `to_hash()` at `:16`) — **leave intact.**
- **Implementation notes**:
  - The override owns the **entire** cache-hit refresh, tensor bindings included — the custom
    adapter replaces `UpdateTensorArgs` rather than adding to it
    (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1010-1031`). So the translated override
    returns a `TensorArgument` for **both** `TensorParameter`s on every hit, mirroring the
    ported-from override, which rewrote both addresses on every hit (`:585-586,613,616`) and
    re-pointed both tensor-backed CBs (`:657-665`).
  - `enumerate_core_rt_args` stays **byte-identical** and stays shared between the two methods.
    It is the mechanism that guarantees the hit path and the miss path cannot disagree; the port
    keeps that property rather than re-deriving the split in the override.
  - The op-level `MeshTensor` extraction is done only where Metal 2.0 needs it (the
    `TensorArgument`s). `enumerate_core_rt_args` keeps its `ttnn::Tensor` interface — rewriting
    that shared work-split helper to `MeshTensor` would be an out-of-scope refactor of the one
    piece of code both paths depend on.

---

## Planned Spec Shape

1:1 with legacy throughout.

- **KernelSpecs** (3): `READER`, `WRITER`, `COMPUTE` — one per legacy `KernelDescriptor`.
  `COMPUTE`'s `source` is the same runtime selection as legacy.
- **DataflowBufferSpecs** (2, or 3 under LOGIT): `SRC` ← `c_0`, `DST` ← `c_2`, `TMP0` ← `c_1`
  (declared only when `needs_tmp0_cb(ops_chain[0].type())`, mirroring the existing host-side
  conditional at `:431`). `entry_size` / `num_entries` are the legacy `page_size` and
  `total_size / page_size`. `tile_format_metadata` left unset on all three (legacy `.tile` unset).
  `borrowed_from = SRC_TENSOR` on `SRC` when `src_sharded`; `= DST_TENSOR` on `DST` when
  `dst_sharded`.
- **SemaphoreSpecs**: none — legacy has no `SemaphoreDescriptor`.
- **TensorParameters** (2): `SRC_TENSOR` (input), `DST_TENSOR` (output). Both declared
  unconditionally, both carrying the relaxation below.
- **WorkUnitSpecs** (1): all three kernels over `operation_attributes.worker_grid`. Legacy gives
  all three `KernelDescriptor`s the same `core_ranges`, so one work unit reproduces it, and the
  local-DFB invariant (producer and consumer share identical `WorkUnitSpec` membership) holds
  trivially.
- **Op-owned tensors**: none — structurally impossible on the ported-from `descriptor` concept.

### RTA schema

Named, per kernel. The reader/writer schema mirrors the legacy slot count exactly, minus slot 0
which becomes the tensor binding:

| kernel | `has_sharding` | `runtime_arg_names` |
|---|---|---|
| reader / writer | true | `num_pages`, `start_id` (legacy 3 slots − address) |
| reader / writer | false | `num_pages`, `start_id`, `chunks_per_row`, `chunk_size`, `last_chunk_size`, `rows_per_tile`, `total_rows` (legacy 8 slots − address) |
| compute | either | `num_tiles`, `packed_scalar1`, `packed_scalar2` |

No varargs anywhere: every argument is a distinct field read a fixed number of times.

The tail five are declared in **both** interleaved configs and written as literal zeros when
`!rm_interleaved`, exactly as legacy does (`:555-557`) — they are dead-on-purpose args
(audit *Misc anomalies* #3), and dropping an arg is a functional change the port is not entitled
to make. In the `RM_INTERLEAVED == 0` build the generated header declares them and the kernel
never reads them, which is a harmless unused `constexpr`.

Every declared name is written for **every** node on **every** dispatch (active cores with their
values, no-op cores with zeros), which is what preserves the legacy guarantee that a core flipped
between active and no-op cannot retain stale args (`:590-591`).

### CRTAs

**None declared.** The legacy common runtime args are the `TensorAccessorArgs` payload and
nothing else (`:472`, `:491`); the binding mechanism auto-builds that, so both the schema entry
and its bespoke cache-hit refresh (`:638-653`) disappear.

### Hardware configuration

- **reader** → `ttnn::create_reader_datamovement_config(device->arch())`. Legacy
  `ReaderConfigDescriptor{}` resolves to `(RISCV_1, NOC_0, DM_DEDICATED_NOC)`, which is the
  reader default byte-for-byte.
- **writer** → `ttnn::create_writer_datamovement_config(device->arch())`. Legacy
  `WriterConfigDescriptor{}` resolves to `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` = writer default.
- **compute** → **Style B**: the op sets a Metal `ComputeConfigDescriptor` directly (literals
  plus `operation_attributes` fields), with no TTNN `ComputeKernelConfig` feeding it. So build a
  `ComputeGen1Config` by hand and copy each field the op set. Do **not** route through
  `to_compute_hardware_config` — its defaults lean high-performance and would flip anything not
  explicitly copied.

  | legacy `ComputeConfigDescriptor` field | value | `ComputeGen1Config` |
  |---|---|---|
  | `math_fidelity` | `HiFi4` (explicit) | `fpu_math_fidelity = HiFi4` |
  | `math_approx_mode` | `false` (explicit, via the local `math_approx_mode`) | `sfpu_precision_mode = Precision::Precise` |
  | `fp32_dest_acc_en` | `operation_attributes.fp32_dest_acc_en` | `enable_32_bit_dest` (1:1) |
  | `bfp8_pack_precise` | `operation_attributes.bfp8_pack_precise` | `bfp_pack_precision_mode = Precise : Approximate` |
  | `dst_full_sync_en` | **not set** → `false` | `double_buffer_dest = !false = true` = the Metal 2.0 default → **no explicit setting needed** |
  | `enable_trisc2_rvv` | not set → `false` | no Metal 2.0 counterpart; no action |
  | `unpack_to_dest_mode` | see below | `unpack_modes` |

  **`unpack_modes` — the reindex + translate + required-entry case.** Legacy builds a
  `vector<UnpackToDestMode>` of `NUM_CIRCULAR_BUFFERS` entries, all `Default`, then sets indices
  `c_0` **and** `c_1` to `UnpackToDestFp32` when `preserve_fp32_precision` (`:397-404`).
  Reindexed to DFB names and translated (`UnpackToDestFp32` → `UnpackToDest`, `Default` →
  `UnpackToSrc`), that is: `SRC` and — when it exists — `TMP0` get `UnpackToDest` iff
  `preserve_fp32_precision`, else `UnpackToSrc`. `DST` gets nothing: the compute kernel is its
  **producer** only, so no entry is required and legacy's `Default` for it is the no-entry case.

  The port emits an **explicit entry for every DFB the compute kernel consumes**, carrying the
  legacy-derived value, rather than omitting the `UnpackToSrc` ones. Three reasons, all
  behaviour-preserving (`UnpackToSrc` lowers to `UnpackToDestMode::Default`, which is what
  omission also yields):
  1. The validator *requires* an explicit entry for a consumed Float32 DFB when
     `enable_32_bit_dest` is true (`program_spec.cpp:1114-1143`), and that combination is
     reachable with legacy `Default`: `BITCAST` to `FLOAT32` from a non-FLOAT32 input makes
     `cb_data_format_for_input == Float32` and `fp32_dest_acc_en == true` while
     `preserve_fp32_precision` is `false`.
  2. The required-entry rule is documented as an intentional intermediate gap that will extend
     to Int32/UInt32 (issue #49936); emitting entries now is inert until it does.
  3. It keeps one rule for all nine compute kernels instead of a per-dtype case analysis.

  `TMP0`'s entry is gated on the DFB existing — the validator rejects an `unpack_modes` key
  naming a DFB the kernel does not bind (`program_spec.cpp:1062-1067`), and legacy's index-`c_1`
  entry was simply ignored when `c_1` was not allocated (audit *Misc anomalies* #5). Gating it is
  required and zero-functional-change.

  `UnpackToDest` is only ever emitted when `preserve_fp32_precision` is true, which (per
  `unary.cpp:56-60`) implies `fp32_dest_acc_en` is true, so `enable_32_bit_dest` is always set
  where it is needed — the "32-bit format into a 16-bit Dest" and "≤16-bit format on Gen1"
  rejections cannot fire.

### Compiler options

- **compute**: `compiler_options.opt_level = KernelBuildOptLevel::O3` — **explicit and
  mandatory.** The legacy `KernelDescriptor::opt_level` is absent, which on a
  `ComputeConfigDescriptor` resolves to `O3`, while Metal 2.0's type-agnostic default is `O2`.
  Omitting it would be a silent one-level perf loss.
- **reader / writer**: nothing to set — absent legacy `opt_level` on a DM descriptor resolves to
  `O2`, which is already the Metal 2.0 default.
- **defines**: `compiler_options.defines` carries what legacy put on `KernelDescriptor::defines`
  — `SRC_SHARDED`/`RM_INTERLEAVED` (reader), `DST_SHARDED`/`RM_INTERLEAVED` (writer), and the
  `unary_defines` map (compute), converted with `Table`'s explicit range constructor.

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Each of the three `KernelDescriptor`s is pushed
exactly once, over the same `core_ranges`. Unary's per-core tile count is an **RTA** (compute
slot 0), not a per-group CTA, so there is no two-`KernelSpec`-per-source case and no CTA→RTA
demotion hazard to guard against. (This is where unary differs from the already-ported
`copy/typecast`, which does carry per-group `per_core_block_cnt` CTAs and therefore two compute
`KernelSpec`s.)

---

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `unary_program_factory.cpp:531,536,555` | reader RTA slot 0 = `input.buffer()` (`Buffer*`) | `TensorBinding{SRC_TENSOR, "src"}` on `READER` + `TensorArgument{SRC_TENSOR, input.mesh_tensor()}` |
| `unary_program_factory.cpp:532,546,557` | writer RTA slot 0 = `output.buffer()` (`Buffer*`) | `TensorBinding{DST_TENSOR, "dst"}` on `WRITER` + `TensorArgument{DST_TENSOR, output.mesh_tensor()}` |
| `unary_program_factory.cpp:585,613` | `r[0] = input.buffer()->address()` on the cache-hit path | the same `TensorArgument`, returned from the translated override |
| `unary_program_factory.cpp:586,616` | `wr[0] = output.buffer()->address()` on the cache-hit path | the same `TensorArgument` |
| `unary_program_factory.cpp:462-463` | `TensorAccessorArgs(*src_buffer, …).append_to(reader_compile_time_args, reader_common_runtime_args)` | the binding mechanism, end-to-end (host CTA+CRTA emission and the kernel's `TensorAccessorArgs<0,0>()` both vanish) |
| `unary_program_factory.cpp:481-482` | `TensorAccessorArgs(*dst_buffer, …).append_to(writer_compile_time_args, writer_common_runtime_args)` | same |
| `reader_unary.cpp:25` | `constexpr auto src_args = TensorAccessorArgs<0, 0>();` | dropped — `TensorAccessor(tensor::src)` |
| `writer_unary.cpp:27` | `constexpr auto dst_args = TensorAccessorArgs<0, 0>();` | dropped — `TensorAccessor(tensor::dst)` |
| `unary_program_factory.cpp:638-653` | the two accessor common-arg refresh loops (`RuntimeTensorShape` payload re-copied on every hit, bounded by `i < common_args.size() && i < reader_common.size()`) | the binding mechanism; `dynamic_tensor_shape` makes the varying shape an implicit CRTA the framework owns |
| `unary_program_factory.cpp:657-665` | `apply_descriptor_runtime_args(program, cb_addr_only)` — re-points the two tensor-backed CBs by **positional** CB matching (hazard described in its own comment at `:655-656`) | `DataflowBufferSpec::borrowed_from` + the `TensorArgument`s; the framework re-resolves the backing L1 address, and the positional-matching hazard goes with the block |
| `reader_unary.cpp:15` | `constexpr auto cb_id_src = tt::CBIndex::c_0;` (magic CB index) | `dfb::src` via `DFBBinding` |
| `writer_unary.cpp:15` | `constexpr auto cb_id_dst = tt::CBIndex::c_2;` | `dfb::dst` |
| compute kernels ×9 | `constexpr auto dfb_input_id = tt::CBIndex::c_0;` / `dfb_output_id = c_2` / `dfb_tmp0_id = c_1` (and `cb_input`/`cb_output` in `eltwise_sfpu.cpp`, `mac_tss_kernel.cpp`) | `dfb::input` / `dfb::output` / `dfb::tmp0` via `DFBBinding` |
| `unary_program_factory.cpp:499-506` | positional compute CTA list | named CTAs: `is_float32`, `is_int` (HARDSWISH); `do_clamp` (LOGIT); `data_format` (the always-appended, never-read one) |
| reader/writer RTA slots 1-7, compute RTA slots 0-2 | positional `get_arg_val<uint32_t>(N)` | named `get_arg(args::<name>)` |
| `reader_unary.cpp:57`, `writer_unary.cpp:60` | `get_local_cb_interface(cb_id).fifo_page_size` | `dfb.get_entry_size()` (whitelist §B; same idiom the two neighbouring `_metal2` forks use) |

**Not dropped, deliberately** — every dead RTA/CTA slot the audit itemised
(*Misc anomalies* #1-#4) is carried across: the always-appended compute `data_format` CTA, the
sharded reader/writer slots the kernel ignores, the INT-TILE zero tail, and the compute scalar
slots that only three of the nine kernels read. Dropping an argument is a functional change.

---

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding) — `TMP0` on the LOGIT compute `KernelSpec`, bound **PRODUCER and CONSUMER** under one accessor name. Census re-derived, not transcribed: `logit_kernel.cpp` is the **only** kernel that touches `c_1` — it `PackTile`s into it (`:41-45`) and `CopyTile`s out of it (`:49-56`). One toucher locked to both FIFO roles → self-loop, which is the one-toucher resolution. Legal on Gen1 for a compute kernel; kernel logic untouched.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb) — the hard gate that says the above is a self-loop rather than 1P+1C or multi-binding.
- [Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings) — `TMP0` is bound only on the LOGIT path, so `dfb::tmp0` exists only in that build. No retrofit is needed: `logit_kernel.cpp` is a *separate source file* selected by the same predicate that allocates `c_1`, so the name is never referenced in a build that lacks the binding. No new `#ifdef` and no new define.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers) — every compute kernel. `dfb::name` flows into `compute_kernel_hw_startup` / `copy_init` / `copy_tile` / `pack_tile`, and into `ckl::input(...)` / `ckl::output(...)` in NTTP position, on the token's `constexpr operator uint32_t()`.
- [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel) — rung 2 for `eltwise_sfpu.cpp`: create `eltwise_sfpu_metal2.cpp` beside the original, point `COMPUTE`'s source at it for the default op path, leave the original untouched apart from the pointer comment. Bindings named for the *kernel* (`dfb::input` / `dfb::output`, `args::num_tiles`), not for unary's locals.
- **Not applied, checked:** multi-binding flag (no CB has ≥3 touchers or two same-role touchers; no raw `get_write_ptr`/`get_read_ptr`/`fifo_*_ptr` anywhere in scope; no semaphores; no dual-instance work-split — all three `KernelDescriptor`s carry distinct sources), aliased DFBs (no multi-element `format_descriptors`), same-FIFO aliasing (no CB-index alias in any kernel), dead-CB drop (none), op-owned tensors (none), varargs (none), `Demoting per-group CTA to RTA` (no per-group CTA exists).

---

## TensorParameter relaxation

Source: `analyses/relaxations/eltwise_unary.md` §2. Verdict **`dynamic` — CONFIRMED**; all five
of that document's validity checks re-run and pass, check 1 having been fixed in the op's code by
the route §1 prescribes (`unary_device_operation.cpp:217-218` hashes `tensor_layout()` for both
slots). The doc's §1 "report UNCONFIRMED" paragraph is stale and its owner has confirmed so.

Applied to **both** `TensorParameter`s, unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

- `dynamic_tensor_shape` is **mandatory**: the TILE-path cache key omits `padded_shape`, so one
  entry legitimately serves many shapes and the *first* hit at a new shape would otherwise throw.
- `relax_logical_rank` for the same reason — rank lives in `logical_shape`, which the TILE key
  also omits, and hashing `tensor_layout` does not reach it.
- **Not set**: `match_page_size` (declined on precedent grounds — no shipped factory sets it) and
  `match_padded_shape_only` (strictly weaker than `dynamic_tensor_shape`).

---

## Deferred / Flagged

- **~~New finding — the regression test the invoker named as the relaxation's safety net does not
  exist.~~ RESOLVED during the port.** `test_unary_sharded_input_on_interleaved_path_cache_reuse`
  had been removed by mistake; on being reported it was restored and it passes. It now sits at
  `tests/ttnn/unit_tests/operations/eltwise/test_unary_program_cache.py:558` and covers analysis §3
  row 5's cache-hit axis directly: a DRAM height-sharded input on the interleaved path, two shapes
  through one asserted cache entry, parameterised over both of the things the TILE branch of the key
  omits (logical rank, and even-vs-uneven padded shape) in both orders. The two pre-existing sharded
  tests — `test_unary_nd_sharded_fallback` (`test_unary_sharding.py:205`) and
  `test_unary_uneven_sharding_fallback` (`:163`) — are single-dispatch, and a wrong relaxation is
  accepted on the first dispatch, so this new test is the only coverage of that axis in the tree.
  One caveat carried to the report: it proves the reused program's numerics, not that the framework
  *accepts* the declaration — the check that reads `TensorParameter::relaxations` runs only on the
  cache-hit path gated by `validate_program_args`, which is off by default.
- **New finding — the borrowed-DFB size check makes the non-32×32 tile defect loud under
  sharding.** `program_spec.cpp:1628-1641` rejects a borrowed DFB larger than its backing
  tensor's per-bank allocation. Unary sizes `entry_size` from `tile_size(DataFormat)` (32×32) while
  `num_entries` comes from the real tile, so a **sharded** non-32×32-tile tensor computes roughly
  double the true shard size and would now `TT_FATAL` where legacy silently produced wrong data.
  Not reachable from the sentinels (the only non-default-tile tests are interleaved). Recorded, not
  acted on — fixing the sizing is the family-wide defect the brief puts out of scope.
- No structural issue that the audit missed. No feature outside the audit's Appendix A. No
  host-computed base-pointer offset. No `sem::`/`tensor::` handle demanded by an out-of-op call
  site. No `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`.
