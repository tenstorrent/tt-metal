# Port Plan — `ttnn/cpp/ttnn/operations/matmul`

Port plan for **`MatmulMultiCoreReuseOptimizedProgramFactory`** (one factory of eight), ported from
the `ProgramDescriptor` API to Metal 2.0. Written during the inventory and planning steps; committed
alongside the port for review.

**Scope:** this factory only. The other seven factories in the op keep their current concept; the
`program_factory_t` variant dispatches per factory, so the op builds and runs half-ported.

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — `create_descriptor` returning a
  `ProgramDescriptor` (`factory/matmul_multicore_reuse_optimized_program_factory.hpp:13-17`).
- **Where the methods live:** in a proper factory struct listed in
  `MatmulDeviceOperation::program_factory_t` (`device/matmul_device_operation.hpp:26`), selected at
  `device/matmul_device_operation.cpp:2196-2197` for `MatmulMultiCoreReuseProgramConfig`.
  **Exception 3 does not apply** — this is not a direct-descriptor op.
- **Variants:** single. The `.cpp` (637 lines) holds exactly two function bodies:
  `default_core_range` (`:31-34`) and `create_descriptor` (`:36-635`). No legacy sibling builder, no
  `override_runtime_arguments`, no helper exported to the CCL fused ops.
- **Custom `compute_program_hash`:** **none.** The device-op declares
  `compute_descriptor_program_hash` (`device/matmul_device_operation.hpp:50`), deliberately *not*
  named `compute_program_hash`, so the framework uses its default reflection hash. Reached only
  through a pybind alias. **Left untouched.**
- **Selection is explicit-config only.** `select_program_factory` returns this factory solely for
  `MatmulMultiCoreReuseProgramConfig`; no auto-selected path reaches it.

### Kernels

Four `KernelDescriptor`s from three sources. `opt_level` is **absent** from every descriptor
(`grep -n opt_level` on the factory returns nothing), so each resolves to its legacy per-kernel-type
default — recorded below as the *resolved* level.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `dataflow/reader_bmm_tile_layout_in0.cpp` | `all_cores` | 11 (`:275-287`) + `TensorAccessorArgs(in0)` | `cb_named_args` (6 CB indices) | per core: `{in0_buffer, in0_start_tile_id, num_output_blocks_per_core}` (`:459`) | none | `IN0_SHARDED` | **O2** | `ReaderConfigDescriptor{}` |
| reader_writer | `dataflow/reader_writer_bmm_tile_layout_in1.cpp` | `all_cores` | 19 (`:291-311`) + accessors for in1, output, (bias) | `cb_named_args` | per core: `{in1_buffer, in1_start_tile_id, num_output_blocks_per_core, output, out_start_tile_id}` + `{*bias, 0u}` when bias (`:463-475`) | none | `IN1_SHARDED`, `OUT_SHARDED`, `FUSE_BIAS` | **O2** | `WriterConfigDescriptor{}` |
| compute_g1 | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | `core_group_1` | 18 (`:349-368`) + 1 when bias | `compute_named_args` = `cb_named_args` + `bias_ntiles` when bias | **empty vectors** per core (`:479`) | none | `compute_defines` | **O3** | `ComputeConfigDescriptor{…}` (`:500-504`) |
| compute_g2 | same source | `core_group_2` (only if non-empty) | 18, slot 13 = `num_blocks_per_core_group_2` (`:509-531`) | same | **empty vectors** per core (`:481`) | none | same | **O3** | same (`:543-547`) |

`compute_defines` (`:375-389`): `PACKER_L1_ACC` when `packer_l1_acc_en`, `FP32_DEST_ACC_EN` when
`fp32_dest_acc_en`, `IN1_TRANSPOSE_TILE` when `in1_transpose_tile`, `FUSE_BIAS` +
`BIAS_FULL_BLOCK` when bias, plus whatever `add_stagger_defines_if_needed` and `throttle_mm_perf`
inject for the arch / core count / throttle level.

The compute kernels' `runtime_args` are **empty `std::vector<uint32_t>{}` per core** — no RTA values
at all. In Metal 2.0 that means no `runtime_arg_schema` and no `KernelRunArgs` entry.

### CBs

`make_cb_descriptor` (`:554-569`) builds all but the aliased pair; every CB is over `all_cores`.

| index | total_size | core_ranges | data_format | page_size | tile | tensor (borrowed) |
|---|---|---|---|---|---|---|
| `c_0` in0 | `in0_CB_size` | all_cores | `in0_data_format` | `in0_aligned_tile_size` | `in0_tile` | `&in0_buffer` when `in0_is_sharded` |
| `c_1` in1 | `in1_CB_size` | all_cores | `in1_data_format` | `in1_aligned_tile_size` | `in1_tile` | `&in1_buffer` when `in1_is_sharded` |
| `c_3` bias | `in3_CB_size` | all_cores | `bias_data_format` | `bias_single_tile_size` | `bias_tile` | — (conditional on `bias.has_value()`) |
| `c_4` out | `out_CB_size` | all_cores | `output_data_format` | `output_single_tile_size` | `output_tile` | `&output` when `output_is_sharded` |
| `c_5` interm0 | `interm0_CB_size` | all_cores | `interm0_data_format` | `interm0_single_tile_size` | `output_tile` | — |
| `c_10` in0 transposed | `in0_CB_size` | all_cores | `in0_data_format` | `in0_aligned_tile_size` | `in0_tile` | — (conditional on `in0_transpose_tile`) |

**Two branches for `c_4`/`c_5`** (`:596-626`):

- **Separate** (`:598-607`) when `interm0_data_format != output_data_format` **or**
  (`untilize_out && in1_num_subblocks > 1`) — two independent `CBDescriptor`s, sizes
  `out_CB_size` / `interm0_CB_size`.
- **Aliased** (`:608-626`) otherwise — **one** `CBDescriptor` of `total_size = out_CB_size` carrying
  *two* `format_descriptors` (`c_4` output format, `c_5` interm0 format), with
  `tensor = output_is_sharded ? &output : nullptr`.

`interm0_data_format` derives from `packer_l1_acc_en` and `fp32_dest_acc_en` (`:114-116`), and
`packer_l1_acc_en` is `packer_l1_acc && (num_blocks > 2)` (`:112`) — so the **`> 2` threshold decides
which of the two branches is taken.**

No `GlobalCircularBuffer` anywhere; no `address_offset`.

### Semaphores

**none** — this factory does no multicast; every core reads its own operand blocks.

### Tensor accessors

| host site | originating Tensor | RTA slot (host) | kernel-side |
|---|---|---|---|
| `TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args)` (`:288`) | input A | reader RTA 0 (`in0_buffer`, `:459`) | `TensorAccessorArgs<11>()` + `in0_tensor_addr` → `TensorAccessor(in0_args, in0_tensor_addr)` (in0 kernel `:39`, `:17`, `:60`) |
| `TensorAccessorArgs(in1_buffer)` (`:312`) | input B | rw RTA 0 (`in1_buffer`, `:464`) | `TensorAccessorArgs<19>()` + `in1_tensor_addr` (in1 kernel `:67`, `:17`, `:115`) |
| `TensorAccessorArgs(output)` (`:313`) | output | rw RTA 3 (`output`, `:467`) | chained `out_args` + `out_tensor_addr` (in1 kernel `:68`, `:24`, `:120`) |
| `TensorAccessorArgs(bias.value())` (`:316`, conditional) | bias | rw RTA 5 (`*bias`, `:470`) | chained `bias_args` + `in3_tensor_addr` (in1 kernel `:71`, `:29`, `:85`) |

All four are **Case 1** (consumed through `TensorAccessor`, never as a raw base pointer). Every
construction is **2-arg** — no third page-size argument anywhere. The factory contains **no
`.address()` / `->address()` expression at all**: tensors are pushed into `emplace_runtime_args` as
objects, so there is no offset-folding to worry about.

### Work split

Three-way (`:210-262`), in priority order:

1. `shard_spec.has_value()` → `all_cores = shard_spec->grid`, `core_group_1 = all_cores`,
   `num_blocks_per_core_group_1 = num_output_blocks_total / num_cores * batch_scale_factor`.
   **Single group.**
2. `core_range_set.has_value()` → `split_work_to_cores(core_range_set.value(), …)`.
   **Pybind-only branch — deleted by this port** (see Dropped Plumbing).
3. else → `split_work_to_cores(program_config.allowed_worker_cores.value(), …)` when set, otherwise
   `split_work_to_cores(program_config.compute_with_storage_grid_size, …)`, with a `log_warning`
   when `allowed_worker_cores` is unset. **This is the production path and it produces both
   groups.**

Both groups then scale by `batch_scale_factor`. `g1_numcores = core_group_1.num_cores()`;
`cores = corerange_to_cores(all_cores, num_cores, row_major)` where `row_major` comes from the shard
orientation when a shard spec exists (`:391-395`).

### Shared kernels

| source | binders | rung |
|---|---|---|
| `dataflow/reader_bmm_tile_layout_in0.cpp` | 1 (this factory) | **convert in place** |
| `dataflow/reader_writer_bmm_tile_layout_in1.cpp` | 1 (this factory) | **convert in place** |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | 4 remaining legacy binders | **rung 1 — reuse the existing `_metal2` fork** |

The compute fork `compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp` already exists beside
the original (created by #55961, reused by #56114). **Rung 1: point `KernelSpec::source` at it, adopt
its names, change nothing in it.** The audit brief predates it and says rung 2 — superseded.

**The fork's binding vocabulary, which this port inherits and cannot rename:**

- DFB accessor names on the compute spec: `in0`, `in1`, `bias`, `out`, `intermed0`,
  `in0_transposed` (plus `intermed0_reload_alias`, which this factory does not use).
- 17 ungated named args: `in0_block_w`, `in0_num_subblocks`, `in0_block_num_tiles`,
  `in0_subblock_num_tiles`, `in1_num_subblocks`, `in1_block_num_tiles`, `in1_block_w`,
  `num_blocks_inner_dim`, `num_blocks_w_dim`, `num_blocks_h_dim`, `out_subblock_h`,
  `out_subblock_w`, `out_subblock_num_tiles`, `batch`, `out_block_num_tiles`, `untilize_out`,
  `get_batch_from_reader`.
- Under `FUSE_BIAS`: `bias_ntiles`, `row_broadcast_bias`, and `dfb::bias`.
- Gates this factory does **not** emit, so their args are not required: `MATMUL_DRAM_SHARDED`
  (`is_worker_core`, `last_subblock_w_valid`), `SFPU_ACTIVATION` (activation args),
  `MM_PARTIALS_RELOAD_ALIAS` (`dfb::intermed0_reload_alias`), `PACK_RELU`, `SKIP_COMPUTE`.

Remaining legacy consumers of the un-forked original after this port: `mcast_2d`, `mcast_1d` (one
file, two factories), and the sparse device-op's factory — **3**. Recorded for the fork's sunset.

### Flags

- The reader and reader_writer descriptors are both handed the **full** `cb_named_args` set
  (`:426`, `:437`) including CB indices they never read. Legacy plumbing with no effect; the port
  binds each kernel only the DFBs it actually touches.
- The two compute descriptors carry per-core **empty** RTA vectors (`:479`, `:481`).
- `default_core_range` (`:31-34`) has no production C++ caller — only its declaration, its
  definition, and the pybind. Removal authorized by the invoker.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` (base). The factory has no
  `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and the port
  writes one method, `create_program_artifacts`. **No override is added.**
- **Custom `compute_program_hash`:** none — default reflection hash. Nothing to touch.
- **`TensorParameter` relaxations:** `none` (audit). All four parameters stay strict.
- **Implementation notes:** the three device-op-class edits the port forces are all authorized by
  the invoker and enumerated under Dropped Plumbing.

---

## Planned Spec Shape

- **KernelSpecs — 3 or 4:** `READER`, `READER_WRITER`, `COMPUTE_G1`, and `COMPUTE_G2` when
  `core_group_2` is non-empty. 1:1 with the legacy descriptors.
- **DataflowBufferSpecs — 4 to 6:** `IN0`, `IN1`, `OUT`, `INTERM0` always; `BIAS` when
  `bias.has_value()`; `IN0_TRANSPOSED` when `in0_transpose_tile`. `entry_size` / `num_entries` taken
  from the legacy `total_size` ÷ `page_size` pairing, `data_format_metadata` from the legacy
  `data_format`, and `tile_format_metadata` copied from each legacy `CBFormatDescriptor::tile`
  (every CB here sets one).
- **SemaphoreSpecs — none.**
- **TensorParameters — 3 or 4:** `IN0`, `IN1`, `OUT`, and `BIAS` when present, each from
  `<tensor>.tensor_spec()`.
- **WorkUnitSpecs — 1 or 2:** `WU_G1 = {READER, READER_WRITER, COMPUTE_G1}` over `core_group_1`;
  `WU_G2 = {READER, READER_WRITER, COMPUTE_G2}` over `core_group_2` when non-empty. The two DM
  kernels belong to both, so their effective node set is the union — which reproduces the legacy
  `all_cores` placement.
- **Op-owned tensors — none.**

### DFB dispositions

| DFB | endpoints | disposition |
|---|---|---|
| `IN0` | reader PRODUCER (`reserve_back`/`push_back`, in0 kernel `:48-49`, `:65`/`:106`), compute CONSUMER | plain 1:1; `borrowed_from = IN0` when `in0_is_sharded` |
| `IN1` | reader_writer PRODUCER (`:105-106`, `:128`/`:151`), compute CONSUMER | plain 1:1; `borrowed_from = IN1` when `in1_is_sharded` |
| `BIAS` | reader_writer PRODUCER (`:86`/`:100`), compute CONSUMER | plain 1:1, **conditionally bound** |
| `OUT` | compute PRODUCER (packs), reader_writer CONSUMER (`:166`/`:187`/`:198`) | plain 1:1; `borrowed_from = OUT` when `output_is_sharded` |
| `INTERM0` | compute only — writes partials and reads them back | **compute self-loop** (PRODUCER + CONSUMER, one accessor name) |
| `IN0_TRANSPOSED` | compute only — transpose target, then matmul operand | **compute self-loop**, conditionally bound |

Both self-loops are genuine FIFO users, not sync-free — neither becomes a scratchpad.

### Aliased DFBs — `OUT` + `INTERM0`, config-dependent

Taken **only** in the aliased branch (`interm0_data_format == output_data_format` and not
(`untilize_out && in1_num_subblocks > 1`)). Both members declare `advanced_options.alias_with`
naming the other (strict clique of two — there is no third index).

Legality, checked against the branch's own values:

- **Same total backing size** ✓ — the branch is only reachable when the two formats are equal, so
  `interm0_single_tile_size == output_single_tile_size` and both DFBs are
  `num_entries = out_CB_tiles`, `entry_size = output_single_tile_size`. Legacy's single descriptor
  sized the region `out_CB_size`, which matches.
- **Same node set** ✓ — `advanced_options.hpp:170-171` states the rule as *"All members must target
  the same node set (derived from their bound kernels' WorkUnitSpecs)"*, **not** "same bound
  kernels" as the patterns catalog paraphrases it. `OUT`'s binders (reader_writer in both WUs,
  compute_g1 in WU_G1, compute_g2 in WU_G2) and `INTERM0`'s (compute_g1, compute_g2) both resolve to
  `core_group_1 ∪ core_group_2 = all_cores`. So the alias is legal with `INTERM0` bound only on the
  compute specs — no invented endpoint, no stop. (Catalog-vs-header discrepancy noted for the
  report.)
- **Borrowed-memory consistency** ⚠ — in the aliased branch `tensor = output_is_sharded ? &output
  : nullptr` applies to the *whole* descriptor, i.e. to both `c_4` and `c_5`. So when the alias
  branch is taken **and** the output is sharded, **`INTERM0` is borrowed from `OUT`'s tensor too**,
  not just `OUT`. Setting `borrowed_from` on only one member both diverges from legacy placement and
  trips the all-or-none rule. In the *separate* branch `INTERM0` is never borrowed.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_kernel_desc` (`core_group_1`, `:491-505`) and `compute_kernel_desc_g2` (`core_group_2`, `:533-548`), same source, differing only at CTA slot 13 | `COMPUTE_G1`, `COMPUTE_G2` | `WU_G1` (`core_group_1`), `WU_G2` (`core_group_2`) — **disjoint** | `IN0` CONSUMER · `IN1` CONSUMER · `BIAS` CONSUMER (cond.) · `OUT` PRODUCER · `INTERM0` PRODUCER+CONSUMER · `IN0_TRANSPOSED` PRODUCER+CONSUMER (cond.) — identical roles on both specs |

Each node sees exactly one compute instance, so both specs binding the same DFB in the same role is
legal with **no** `allow_instance_multi_binding`. The per-group value is the fork's **`args::batch`**
(legacy slot 13 = `num_blocks_per_core_group_{1,2}`) and it stays a **named CTA on each spec** —
demoting it to an RTA to collapse the two specs is the documented anti-pattern and would cost the
compile-time unrolling on that dimension.

---

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`:459`) | `in0_buffer` pushed as an RTA object | `TensorBinding` → `TensorParameter IN0` |
| reader CTAs 11+ (`:288`) | `TensorAccessorArgs(in0_buffer).append_to(...)` | binding mechanism end-to-end |
| in0 kernel `:17`, `:39`, `:60` | `in0_tensor_addr` RTA + `TensorAccessorArgs<11>()` + 2-arg ctor | `TensorAccessor(tensor::in0)` |
| rw RTA slot 0 (`:464`) | `in1_buffer` as RTA object | `TensorParameter IN1` |
| rw RTA slot 3 (`:467`) | `output` as RTA object | `TensorParameter OUT` |
| rw RTA slot 5 (`:470`, cond.) | `*bias` as RTA object | `TensorParameter BIAS` |
| rw CTAs 19+ (`:312-317`) | three chained `TensorAccessorArgs` | binding mechanism |
| in1 kernel `:17`, `:24`, `:29`, `:67`, `:68`, `:71` | three address RTAs + chained accessor args | `TensorAccessor(tensor::{in1,out,bias})` |
| `cb_named_args` (`:336-343`) | `{"cb_in0", CBIndex::c_0}` … `{"cb_in0_transposed", CBIndex::c_10}` | `DFBBinding`s; every `get_named_compile_time_arg_val("cb_*")` in all three kernels becomes `dfb::<name>` |
| reader CTAs 0-10, rw CTAs 0-18, compute CTAs 0-16 + 18 | positional | named CTAs (positional CTAs are not in the Metal 2.0 API) |
| **compute CTA slot 17** (`in0_transpose_tile`, `:367`/`:527`) | positional CTA read as `constexpr bool` | **no argument** — becomes the `IN0_TRANSPOSE_TILE` define, matching the fork, which derives `in0_transpose_tile` from the macro and gates `dfb::in0_transposed` on it |
| compute `runtime_args` (`:479`, `:481`) | per-core empty `std::vector<uint32_t>{}` | omitted entirely — no schema, no `KernelRunArgs` entry |
| `create_descriptor`'s `core_range_set` parameter (`:40`) and its work-split branch (`:219-227`) | pybind-only entry point | dropped; production default is `std::nullopt`, so the branch is unreachable from C++ and the `else` path at `:230-261` is the production behavior — **preserved unchanged** |
| `matmul_nanobind.cpp:1239-1258` | `nb::class_` binding `create_descriptor` + `default_core_range` | deleted (invoker-authorized) |
| `default_core_range` (`hpp:19`, `cpp:31-34`) | pybind-only factory method | deleted (invoker-authorized) |
| `ttnn/ttnn/operations/matmul.py:25`, `ttnn/ttnn/__init__.py:512` | Python re-export of the factory class | deleted — without this, removing the `nb::class_` makes `import ttnn` raise `AttributeError` at module scope (invoker-authorized) |

Surviving named RTAs, per kernel:

- **reader:** `in0_tensor_start_tile_id`, `batch` (the legacy third RTA, receiving
  `num_output_blocks_per_core`). Both are distinct fields read once at the top of the kernel through
  a running `rt_args_idx++` — **named, not varargs.**
- **reader_writer:** `in1_tensor_start_tile_id`, `batch`, `out_tensor_start_tile_id`, plus
  `in3_tensor_start_tile_id` under `FUSE_BIAS`. Same reasoning.
- **compute:** none.

No varargs anywhere: neither DM kernel reads in a loop or at a data-computed index, and there is no
scan-terminating sentinel.

---

## Applied Patterns

- [Porting a shared kernel](#) — **rung 1**, reusing `…_metal2.cpp` for the compute kernel; the two
  DM kernels are private and convert in place.
- [Self-loop DFB binding](#) — `INTERM0` (always) and `IN0_TRANSPOSED` (conditional) on the compute
  specs, PRODUCER + CONSUMER under one accessor name each.
- [Conditional / optional resource bindings](#) — `BIAS` gated on `bias.has_value()` with
  `FUSE_BIAS`; `IN0_TRANSPOSED` gated on `in0_transpose_tile` with `IN0_TRANSPOSE_TILE`. The fork
  already `#ifdef`-gates both on the kernel side.
- [Aliased DFBs](#) — `OUT` + `INTERM0` mutual `alias_with` in the shared-descriptor branch only,
  derived per instantiation.
- [Demoting per-group CTA to RTA](#) — the anti-pattern this port must avoid on `args::batch`.
- [Removing pybound legacy factory entry points](#) — exceptions 1 and 2, both authorized.
- [Pass DFB handles directly to LLKs](#) — the DM kernels' `get_tile_size` / `get_dataformat` calls.

Kernel-side metadata calls, keyed on the legacy declaration:

- in0 kernel `:52` `constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(dfb_id_in0)` and
  `:84`/`:91` `constexpr DataFormat in0_data_format = get_dataformat(dfb_id_in0)` are **`constexpr`**
  → keep the **free-function form with the binding token**: `get_tile_size(dfb::in0)`,
  `get_dataformat(dfb::in0)`. Not demoted to `const`, not converted to member getters. Recorded for
  the report as Gen1-only token-form sites.
- in1 kernel `:84`, `:108`, `:119` `dfb_*.get_tile_size()` are already **member getters** on
  non-`constexpr` locals → unchanged.

---

## Deferred / Flagged

1. **Alias group legality — resolved, no action.** The third rule is *same node set*, not *same
   bound kernels* (`advanced_options.hpp:170-171`); both members resolve to `all_cores`. The same
   header also confirms the disjoint-node work-split shape explicitly: *"a DFBSpec (spanning
   multiple nodes) can have more than one KernelSpec producer or more consumer bindings, as long as
   every node's DFB instance has one producer and one consumer"* (`:182-186`) — so the two compute
   specs binding one DFB in one role need no advanced option.
2. **Borrowed `INTERM0` in the aliased + sharded-output case** (see Planned Spec Shape). Derived,
   not inherited from the brief — worth a reviewer's eye.
3. **`packer_l1_acc_en` uses `> 2`, not `> 1`** (`:112`). Carried verbatim. It selects
   `interm0_data_format`, which selects the aliased-vs-separate branch, so "fixing" it would change
   CB topology. Reported, not repaired.
4. **The bias `TT_FATAL`** (`:175-182`) lives inside the factory body and must survive the rewrite
   verbatim; the census will check. Eight `TT_FATAL`s total in the file (`:41`, `:45`, `:46`, `:47`,
   `:82`, `:102`, `:175`, `:264`).
5. **`unpack_modes`.** No legacy `unpack_to_dest_mode` is set, so there is no table to reindex. An
   entry is required only if a compute kernel consumes a **Float32** DFB while
   `enable_32_bit_dest` is true — reachable here, since `interm0_data_format` is `Float32` whenever
   `fp32_dest_acc_en` (`:114-116`). Check the resolved config during construction and add the entry
   for `INTERM0` (and `OUT` in the aliased branch, same format) when it fires; derive the value from
   the absent legacy table, i.e. `UnpackToSrc`.
6. **Descriptor-framework casualties.** Removing `create_descriptor` disables ~15 tests across
   `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py` and
   `.../demo/test_fused_demo.py`, both of which reach matmul *only* through
   `models/experimental/ops/descriptors/matmul.py`. Per the invoker: leave them failing, record in
   the report. Enumerated in `METAL2_PORT_REPORT.md` under Handoff points.
