# Port Plan — `ttnn/cpp/ttnn/operations/matmul`

Port plan for **one factory**: `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory`, ported from
the legacy `ProgramDescriptor` API to Metal 2.0. Written during the inventory and planning steps;
committed alongside the port for review.

The op directory holds two DeviceOperations and eight ProgramFactories. Only this one is in scope —
the audit cleared only this one. Nothing here is a statement about the other seven.

**Recipe docs (this port):** `b419a49b934 2026-09-01 docs(metal_2.0): the conditional-binding pattern
covers tensors and semaphores too`
**Audit docs (inherited):** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program
ops, and bound what the port covers`

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — `create_descriptor` returning a
  `ProgramDescriptor` is the struct's only member
  (`device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp:14-18`).
- **Where the factory methods live:** in a `program_factory_t` variant alternative
  (`device/matmul_device_operation.hpp:30`), selected at
  `device/matmul_device_operation.cpp:2211`. Not the direct-descriptor shape, so
  ttnn_factory exception 3 does not apply.
- **Variants:** single. One program builder, `create_program_dram_sharded_descriptor`
  (`.cpp:44-933`), reached from `create_descriptor` (`.cpp:937-1090`). No second builder in the
  file, no sibling factory sharing it, no `override_runtime_arguments` anywhere.
- **Custom `compute_program_hash`:** none — the op uses the default reflection hash. A
  `compute_descriptor_program_hash` helper exists at `device/matmul_device_operation.hpp:50`,
  deliberately *not* named `compute_program_hash` so the framework does not treat it as the cache
  hook; it is reached only through a pybind alias. **Left untouched.**

*(Target Metal 2.0 concept was chosen during the audit; carried forward under
[TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels

Three `KernelDescriptor`s, all on `all_cores_in_rect_grid` (the bounding box of mcast senders ∪
worker cores, `.cpp:299-302`). CTA columns are read from the **host's emission order**.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| in0 sender | `device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` (`:429`) | `all_cores_in_rect_grid` | 16 (`.cpp:323-344`) — see mapping below | `cb_in0`→`c_0`, `cb_in0_sharded`→`c_2` (`:434-437`) | per-core, **variable length**: 3 + 2·S on sender/receiver cores, 1 on idle cores (`:684-736`) | none | `mm_kernel_in0_sender_define` — **empty** (`skip_in0_mcast` is hardcoded `false`) | absent → resolves **O2** | `DataMovementConfigDescriptor{RISCV_1, in0_noc}` (`:438-439`) |
| in1 sender/writer | `device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` (`:445`) | `all_cores_in_rect_grid` | 12, +3 when bias (`.cpp:346-365`) | `cb_in1`→`c_1`, `cb_bias`→`c_3`, `cb_out`→`c_4`, `cb_out_reshard`→`c_6` (`:450-455`) | per-core, **variable length**: 1 on non-worker cores, 11 + 3·(k−1) on worker cores with k write-back shards (`:766-917`) | none | `OUT_SHARDED`, `SKIP_MCAST` (both unconditional, `:389-390`), `FUSE_BIAS` (bias), `SPLIT_DRAM_BANK` (`workers_per_bank > 1`), `SKIP_WRITE_BACK` (never emitted) | absent → resolves **O2** | `DataMovementConfigDescriptor{RISCV_0, in1_noc}` (`:456-457`) |
| compute | `device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` (`:495`) | `all_cores_in_rect_grid` | 18, +1 when bias (`.cpp:465-491`) | 10, +4 when SFPU activation (`:501-521`) | 1 per core: `is_worker_core` (`:766-782`) | none | `mm_kernel_defines`: `MATMUL_DRAM_SHARDED` always; `FUSE_BIAS`, `PACK_RELU`/`SFPU_ACTIVATION`, `PACKER_L1_ACC`, `FP32_DEST_ACC_EN`, `IN1_TRANSPOSE_TILE` conditionally; `SKIP_COMPUTE` never; plus stagger/throttle defines (`:407-410`) | absent → resolves **O3** (`ComputeConfigDescriptor` default) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` (`:523-528`) |

`grep -n opt_level` over the factory returns **nothing** — no kernel sets it. The resolved levels
above are the legacy per-kernel-type defaults.

S = `input_all_storage_cores_vec.size()` = number of in0 mcast sender (storage) cores.

#### in0 sender — positional CTA slots

| slot | host value | kernel reads as | port disposition |
|---|---|---|---|
| 0 | `in0_block_num_tiles` | `in0_block_num_tiles` | named CTA |
| 1 | `in0_block_num_tiles * in0_single_tile_size` | `in0_block_size_bytes` | named CTA |
| 2 | `in0_last_ktile_w` | `in0_last_ktile_w` | named CTA |
| 3 | `0` (transpose unsupported here) | `in0_last_ktile_h` | named CTA |
| 4 | `in0_mcast_sender_semaphore_id` | `Semaphore<> sender_sem(...)` (`:61`) | **SemaphoreBinding** `sem::in0_mcast_sender` |
| 5 | `in0_mcast_receiver_semaphore_id` | `Semaphore<> receiver_sem(...)` (`:62`) | **SemaphoreBinding** `sem::in0_mcast_receiver` |
| 6 | `num_worker_cores` | `in0_mcast_num_dests` | named CTA |
| 7 | `num_mcast_cores` | `in0_mcast_num_cores` | named CTA |
| 8 | `num_blocks` | `num_blocks` | named CTA |
| 9-12 | `start_core_noc.{x,y}`, `end_core_noc.{x,y}` | `in0_mcast_dest_noc_{start,end}_{x,y}` | named CTAs |
| **13** | `in0_mcast_sender_valid_semaphore_id` | **never read** | **drop** — see Flags |
| 14 | `num_blocks_per_shard` | `num_blocks_per_shard` | named CTA |
| 15 | `in0_block_w` | `in0_block_w` | named CTA |

#### in1 sender/writer — positional CTA slots

| slot | kernel reads as | port disposition |
|---|---|---|
| 0-11 | `in1_page_size`, `in1_num_pages`, `in1_block_w`, `in1_block_num_tiles`, `num_blocks`, `out_block_num_tiles`, `out_tensor_stride_w_bytes`, `out_reshard_tensor_stride_w_bytes`, `per_core_M`, `workers_per_bank`, `bank_row_stride_tiles`, `reader_width_tiles` | named CTAs |
| 12-13 (bias) | `in3_page_size`, `in3_num_pages` | named CTAs, emitted only when bias |
| **14** (bias) | literal `1` pushed at `.cpp:364` — **never read** | **drop** — see Flags |

#### compute — positional CTA slots

| slot | kernel reads as | port disposition |
|---|---|---|
| 0-16 | `in0_block_w`, `in0_num_subblocks`, `in0_block_num_tiles`, `in0_subblock_num_tiles`, `in1_num_subblocks`, `in1_block_num_tiles`, `in1_block_w`, `num_blocks_inner_dim`, `num_blocks_w_dim`, `num_blocks_h_dim`, `out_subblock_h`, `out_subblock_w`, `out_subblock_num_tiles`, `batch`, `out_block_num_tiles`, `untilize_out`, `get_batch_from_reader` | named CTAs (kernel-side names; the host's slot-8/9 comments say `out_num_blocks_x`/`_y`) |
| **17** | `in0_transpose_tile` — selects a DFB in a parse-time ternary (kernel `:200`) | **moves to a define**, `IN0_TRANSPOSE_TILE` — see Applied Patterns |
| 18 (bias) | `row_broadcast_bias` | named CTA, emitted only when bias |

Named CTAs `cb_in0`, `cb_in1`, `cb_bias`, `cb_out`, `cb_intermed0`, `cb_in0_transposed` become
DFB bindings; `bias_ntiles`, `last_subblock_w_valid`, `activation_type`, `activation_param0..2`
stay named CTAs; `cb_in0_intermediate` (`c_8`) and `cb_in1_intermediate` (`c_9`) are dead and drop.

### CBs

Seven `CBDescriptor`s, all on `all_cores_in_rect_grid`. `c_4`/`c_5` are emitted as **either** two
descriptors **or** one two-format descriptor, depending on config.

| index | total_size | core_ranges | data_format | page_size | tile | backing |
|---|---|---|---|---|---|---|
| `c_0` in0 | `in0_CB_size` (`:543`) | rect grid | `in0_data_format` | `in0_single_tile_size` | `in0_tile` | regular |
| `c_1` in1 | `in1_CB_size` (`:556`) | rect grid | `in1_data_format` | `in1_aligned_tile_size` | `in1_tile` | regular |
| `c_2` in0 sharded | `in2_CB_size` (`:569`) | rect grid | `in0_data_format` | `in0_single_tile_size` | `in0_tile` | **borrowed** — `cb_desc.tensor = &in0_tensor` (`:576`) |
| `c_4` out | `out_CB_size` (`:585`/`:608`) | rect grid | `output_data_format` | `output_single_tile_size` | `output_tile` | regular |
| `c_5` interm0 | `interm0_CB_size` (`:596`) / shares `c_4`'s descriptor (`:615-619`) | rect grid | `interm0_data_format` | `interm0_single_tile_size` | `output_tile` | regular |
| `c_6` out reshard | `out_reshard_CB_size` (`:626`) | rect grid | `output_data_format` | `output_single_tile_size` | `output_tile` | **borrowed** — `cb_desc.tensor = &out_tensor` (`:633`) |
| `c_3` bias | `in3_CB_size` (`:640`) | rect grid | `bias_data_format` | `bias_aligned_tile_size` | `bias_tile` | regular, **conditional** (bias present) |

No `GlobalCircularBuffer`, no `address_offset`, no `cb_descriptor_from_sharded_tensor`.

**Kernel-touch census** (single code path — the three `skip_*` flags are hardcoded `false` at
`.cpp:1036-1038`, so there is no per-config census flip):

| CB | touchers | verdict |
|---|---|---|
| `c_0` | in0 kernel producer (`:145`/`:214`/`:224`/`:233`), compute consumer (`:321`/`:477`) | plain 1P+1C |
| `c_1` | in1 kernel producer (`:92`/`:108`/`:122`/`:158`/`:162`/`:178`), compute consumer (`:322`/`:478`) | plain 1P+1C |
| `c_3` | in1 kernel producer (`:183`/`:206`), compute consumer (`:501`/`:576`/`:621`) | plain 1P+1C, conditional |
| `c_4` | compute producer (all four bias × untilize configs — see below), in1 kernel consumer (`:210`/`:246`) | plain 1P+1C |
| `c_5` | compute only — packs into it (`:434`) and reads it back (`:453`/`:513`/`:543`) | **1 toucher → compute self-loop** |
| `c_2` | in0 kernel only, `get_read_ptr()` at `:71`; **no FIFO ops anywhere** | **1 toucher, sync-free → self-loop** |
| `c_6` | in1 kernel only, `get_write_ptr()` at `:218`; **no FIFO ops anywhere** | **1 toucher, sync-free → self-loop** |

`c_4` is produced by compute on every path, verified per config: `!bias`/`!untilize` packs to
`out` directly (`mm_out_dfb_id == out_dfb_id`); `!bias`/`untilize` and `bias`/`untilize` reach it
through `reblock_and_untilize` (`:595`); `bias`/`!untilize` packs to it as
`untilize_mode_out_dfb_id` (`:567`). No config leaves `c_4` unproduced.

### Semaphores

| id | core_type | core_ranges | initial_value | used by |
|---|---|---|---|---|
| 0 | worker (default) | `all_cores_in_rect_grid` (`:653-654`) | `INVALID` = 0 | in0 kernel, CTA 4 |
| 1 | worker (default) | `all_cores_in_rect_grid` (`:655-656`) | `INVALID` = 0 | in0 kernel, CTA 5 |
| 2 | worker (default) | `all_cores_in_rect_grid` (`:657-658`) | `VALID` = 1 | **nothing** — its id rides CTA 13, which no kernel reads |

### Tensor accessors

**No kernel constructs a `TensorAccessor` today** — the DM kernels address DRAM through
`AllocatorBank<DRAM>` bank/address endpoints, and in0/output arrive through borrowed CBs. So there
is no legacy accessor site and no third-argument page size to drop. The four tensors and how each
surfaces:

| tensor | host site | how the kernel consumes it | classification |
|---|---|---|---|
| in1 | RTA slot [1], rebound to the tensor at `.cpp:912` | raw base address — `{.bank_id, .addr = in1_tensor_addr + …}` (in1 kernel `:87`, `:100`, `:132`, `:137`, `:148`, `:255`) | **Case 2** |
| bias | RTA slot [2], rebound at `.cpp:914` (bias only) | raw base address — `{.bank_id, .addr = in3_tensor_addr}` (in1 kernel `:191`, `:198`) | **Case 2** |
| in0 | borrowed CB `c_2` (`.cpp:576`) | `dfb_in2.get_read_ptr()` (in0 kernel `:71`) | clean — borrowed DFB |
| output | borrowed CB `c_6` (`.cpp:633`) | `dfb_out_reshard.get_write_ptr()` (in1 kernel `:218`) | clean — borrowed DFB |

`.cpp:796-797` compute `.address()` values that the variant rebinding at `.cpp:912-915` overwrites.
Both carry a `smuggled-rta-ok` marker. They are placeholders, not smuggled pointers.

### Work split

**n/a** — no `split_work_to_cores`. All three kernels are placed on the single
`all_cores_in_rect_grid` set; per-core behaviour is selected by runtime args
(`worker_core_type` / `is_worker_core`), not by descriptor multiplicity.

### Shared kernels

| kernel | binders | `_metal2` fork exists? | rung |
|---|---|---|---|
| `device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` | **1** — this factory | no | **convert in place** |
| `device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` | **1** — this factory | no | **convert in place** |
| `device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | **6** | **no** | **rung 2 — create the fork** |

Rung-1 check re-run against this branch's tree: `find ttnn/cpp/ttnn/operations/matmul/device/kernels/
-name '*_metal2*'` returns **zero** files, so this port creates the first fork.

The compute kernel's five remaining binders, all in this directory:
`matmul_multicore_reuse_optimized_program_factory.cpp`,
`matmul_multicore_reuse_mcast_1d_program_factory.cpp` (hosts **two** factories),
`matmul_multicore_reuse_mcast_2d_program_factory.cpp`,
`matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp`,
`device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`.

Filename hits that are **not** binders of this file, disambiguated:
`tests/tt_metal/.../1_compute_mm/kernels/` and `.../old/matmul/kernels/` hold their own private
copies (`test_compute_mm.cpp:1086-1088`, `:1234-1236`, `matmul_global_l1.cpp:512`,
`test_remote_cb_sync_matmul.cpp:316` — the latter binds `…_copy.cpp`);
`llama_1d_mm_fusion.cpp:477` binds `…_gathered.cpp`; two deepseek `kernel_utils.hpp` and
`fused_swiglu.cpp` mention the name in comments only.

**Fork binding vocabulary** (ratified with the invoker before naming; taken from the kernel's own
named-CTA keys with `cb_` dropped, matching the vocabulary
`bmm_metal2.cpp` established for `MatmulMultiCoreProgramFactory`):
`dfb::in0`, `dfb::in1`, `dfb::bias`, `dfb::out`, `dfb::intermed0`, `dfb::in0_transposed`,
`dfb::intermed0_reload_alias`. Named args as listed in the compute CTA table above.

### Flags

- **Three dead legacy CTA slots.** in0 sender slot 13 (`in0_mcast_sender_valid_semaphore_id`) and
  in1 writer slot 14 (the literal `1` at `.cpp:364`) are emitted and never read; the named CTAs
  `cb_in0_intermediate` (`c_8`) and `cb_in1_intermediate` (`c_9`) name CB indices that no
  `CBDescriptor` allocates and the compute kernel never references. The brief flagged the two named
  ones; slots 13 and 14 are new findings from this inventory.
- **One dead semaphore.** Semaphore id 2 (`in0_mcast_sender_valid`, initial `VALID`) is allocated
  but no kernel reads its id — the only path to it was CTA 13. The "sender valid" handshake is in
  fact implemented on semaphore 1 (`receiver_sem.set(VALID)`, in0 kernel `:67`).
- **Runtime args are read as arrays in both DM kernels** — via `get_arg_addr`, which the audit's
  scan did not cover, so the brief's "RTA varargs: none" is wrong. See Dropped Plumbing.
- **`SKIP_MCAST` is two different defines** twelve lines apart: unconditional for the in1 writer
  (`.cpp:390`), conditional on `skip_in0_mcast` for the in0 sender (`.cpp:396`). Only the first is
  ever emitted.
- **`in0_mcast_sender_noc_y`'s base offset is fragile.** The kernel computes the y-array base as
  `get_arg_addr(3 + num_storage_cores)` where `num_storage_cores = num_blocks /
  num_blocks_per_shard` (`:36`), while the host emits exactly S x-values then S y-values. The two
  agree only when `num_blocks % S == 0`. Preserved verbatim; reported, not fixed.
- **No unreferenced kernel files** in the op directory attributable to this factory.
- The kernel file's header comment (`:24-29`) records a maintenance coupling to
  `tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/…_copy.cpp`. That copy tracks
  the *legacy* file, which this port leaves functionally untouched.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` (base). The factory has no
  `override_runtime_arguments`, so the framework refreshes tensor bindings on cache hit and the port
  writes one method, `create_program_artifacts`. **No override is added** — adding one would move
  the op to the custom concept and make the port responsible for the whole cache-hit refresh.
- **Custom `compute_program_hash`:** none. Left intact (nothing to do); see Legacy Inventory.
- **Implementation notes:**
  - Spec-scope resource-name constants are declared **function-local**, not at file scope: the
    matmul factory `.cpp` files share one unity-build target, so file-scope constants would collide
    as sibling factories are ported. (Convention set by `matmul_multicore_program_factory.cpp`.)
  - Device-op-class edits this port forces: delete the `nb::class_<…DRAMShardedProgramFactory>`
    block at `matmul_nanobind.cpp:1308-1323` (ttnn_factory exception 1) and drop
    `create_descriptor`'s pybind-hook-only `core_range_set` parameter (exception 2). The
    `nb::class_<MatmulDeviceOperation>` block at `:1222-1237` stays untouched. No Python caller of
    the removed entry point exists in-tree.

---

## Planned Spec Shape

- **KernelSpecs: 3** — `IN0_SENDER`, `IN1_SENDER_WRITER`, `COMPUTE`. 1:1 with the legacy
  `KernelDescriptor`s; no multiplicity to preserve.
- **DataflowBufferSpecs: 6, or 7 with bias** — `in0` (`c_0`), `in1` (`c_1`), `in0_sharded` (`c_2`,
  borrowed), `out` (`c_4`), `intermed0` (`c_5`), `out_reshard` (`c_6`, borrowed), and `bias` (`c_3`)
  when bias is present. `entry_size` / `num_entries` are the legacy page size and tile count, so
  `entry_size * num_entries` reproduces each legacy `total_size` exactly.
- **SemaphoreSpecs: 3** — 1:1 with the legacy descriptors, including the dead one, so the program's
  semaphore allocation is unchanged. Two are bound to `IN0_SENDER`; the third is declared and
  unbound, mirroring a legacy semaphore whose id no kernel reads.
- **TensorParameters: 3, or 4 with bias** — `in0`, `in1`, `output`, and `bias`.
- **WorkUnitSpecs: 1** — all three kernels on `all_cores_in_rect_grid`.
- **Op-owned tensors: none.**

DFB endpoint bindings:

| DFB | producer | consumer |
|---|---|---|
| `in0` | `IN0_SENDER` | `COMPUTE` |
| `in1` | `IN1_SENDER_WRITER` | `COMPUTE` |
| `bias` (cond.) | `IN1_SENDER_WRITER` | `COMPUTE` |
| `out` | `COMPUTE` | `IN1_SENDER_WRITER` |
| `intermed0` | `COMPUTE` | `COMPUTE` (self-loop) |
| `in0_sharded` | `IN0_SENDER` | `IN0_SENDER` (self-loop) |
| `out_reshard` | `IN1_SENDER_WRITER` | `IN1_SENDER_WRITER` (self-loop) |

`borrowed_from`: `in0_sharded` ← `in0`, `out_reshard` ← `output`. Neither needs a `TensorBinding`
on any kernel — the borrowed DFB *is* the tensor access, and its backing L1 address resolves from
the corresponding `TensorArgument` on every cache hit. Neither needs a `dfb_run_overrides` entry.

`alias_with`: `out` ↔ `intermed0`, **only** in the config where the legacy factory emitted them as
one two-format `CBDescriptor` (`interm0_data_format == output_data_format` and not (`untilize_out`
&& `in1_num_subblocks > 1`), `.cpp:581`). Derived per instantiation. In that branch the two formats
have equal tile size, so the alias-group equal-total-size rule holds by construction. The group is
exactly two members — there is no third alias index in this factory.

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Each legacy `KernelDescriptor` maps to exactly one
`KernelSpec`, and all three sit in the single `WorkUnitSpec`. No CTA is per-core, so there is no
CTA→RTA demotion pressure.

---

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| in1 writer RTA slot [1] (`.cpp:796`, rebound `:912`) | `in1_tensor.address()` → tensor variant | `TensorParameter`/`TensorBinding` `in1`; kernel pulls the base via `TensorAccessor(tensor::in1).get_bank_base_address()` (**Case 2** bridge). Raw arithmetic left unchanged. |
| in1 writer RTA slot [2] (`.cpp:797`, rebound `:914`) | `bias->address()` → tensor variant | `TensorParameter`/`TensorBinding` `bias`, conditional on bias presence; same Case 2 bridge. |
| `c_2` `cb_desc.tensor = &in0_tensor` (`.cpp:576`) | borrowed CB backed by a raw tensor pointer | `DataflowBufferSpec::borrowed_from = in0` |
| `c_6` `cb_desc.tensor = &out_tensor` (`.cpp:633`) | borrowed CB backed by a raw tensor pointer | `DataflowBufferSpec::borrowed_from = output` |
| in0 sender named CTAs (`.cpp:434-437`) | `{"cb_in0", c_0}`, `{"cb_in0_sharded", c_2}` | `DFBBinding`s `in0`, `in0_sharded` |
| in1 writer named CTAs (`.cpp:450-455`) | `{"cb_in1", c_1}`, `{"cb_bias", c_3}`, `{"cb_out", c_4}`, `{"cb_out_reshard", c_6}` | `DFBBinding`s `in1`, `bias`, `out`, `out_reshard` |
| compute named CTAs (`.cpp:502-509`) | `cb_in0`, `cb_in1`, `cb_bias`, `cb_out`, `cb_intermed0`, `cb_in0_transposed` | `DFBBinding`s `in0`, `in1`, `bias`, `out`, `intermed0`, `in0_transposed` |
| compute named CTAs (`.cpp:507-508`) | `cb_in0_intermediate` (`c_8`), `cb_in1_intermediate` (`c_9`) | **dropped** — dead; no descriptor allocates them, no kernel reads them |
| in0 sender CTA slots 4, 5 (`.cpp:329-330`) | semaphore ids as positional CTAs | `SemaphoreBinding`s `in0_mcast_sender`, `in0_mcast_receiver` |
| in0 sender CTA slot 13 (`.cpp:341`) | `in0_mcast_sender_valid_semaphore_id` | **dropped** — never read |
| in1 writer CTA slot 14 (`.cpp:364`) | literal `1` | **dropped** — never read |
| compute CTA slot 17 (`.cpp:487`) | `in0_transpose_tile`, selecting a DFB in a parse-time ternary | `compiler_options.defines["IN0_TRANSPOSE_TILE"]` + `#ifdef`-gated binding |
| compute define `MM_PARTIALS_RELOAD_ALIAS_CB` (kernel `:207-213`; emitted by mcast_1d/2d, never by this factory) | a **CB index carried in a preprocessor define** | `#ifdef MM_PARTIALS_RELOAD_ALIAS` + `DFBBinding` `intermed0_reload_alias` |
| all remaining positional CTAs on all three kernels | `get_compile_time_arg_val(N)` | named CTAs, per the slot tables above |
| in0 sender RTA slots 3 … 3+2S−1 (`.cpp:699-702`, `:717-720`) | `get_arg_addr(3)` / `get_arg_addr(3 + num_storage_cores)` cast to `tt_l1_ptr uint32_t*`, indexed `[block_id]` (kernel `:47-48`, `:210`, `:229`) | **runtime varargs** — `advanced_options.num_runtime_varargs = 2·S`, kernel reads `get_vararg(block_id)` / `get_vararg(num_storage_cores + block_id)` |
| in1 writer RTA slots 8 … (`.cpp:828-841`, `:857-888`) | `get_arg_addr(8/9/10)` cast to `tt_l1_ptr uint32_t*`, walked as interleaved triples with `index_offset += 3` (kernel `:31-33`, `:231-241`) | **runtime varargs** — `num_runtime_varargs = 3·max_k` over worker cores, zero-padded per node; kernel reads `get_vararg(index_offset{,+1,+2})` |

No page-size third-argument CTA/RTA exists (no `TensorAccessor` is constructed today), and no
`TensorAccessorArgs` plumbing exists to remove.

**Named-RTA schema consequence.** Metal 2.0 requires every name in a kernel's
`runtime_arg_schema` to have a value on **every** node the kernel runs on
(`program_run_args.hpp:57-60`), while the legacy descriptor emitted 1 arg on cores that return
early and the full list on the rest. The port therefore supplies the full named set on every node,
with zero for the fields an early-returning core never reads. That mirrors what the legacy factory
already does within the worker set (`.cpp:905-907` resizes short worker arg lists to
`fixed_writer_arg_count` with zeros), and no kernel reads past its early return, so device
behaviour is unchanged; only the dispatched arg payload for idle/non-worker cores grows.

---

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `in0_sharded` (`c_2`) on `IN0_SENDER` and `out_reshard` (`c_6`) on `IN1_SENDER_WRITER` — each a
  borrowed, sync-free one-toucher; and `intermed0` (`c_5`) on `COMPUTE`, a genuine-FIFO one-toucher.
  All three bind their single touching kernel as both PRODUCER and CONSUMER.
- [Aliased DFBs](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs):
  `out` ↔ `intermed0`, mutual `advanced_options.alias_with`, only in the shared-descriptor branch.
- [Conditional / optional resource bindings](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings):
  three instances — `bias` (existing `FUSE_BIAS` define, DFB + `TensorParameter` +
  `TensorBinding` + named CTAs all gated), `in0_transposed` (new `IN0_TRANSPOSE_TILE` define
  replacing positional CTA 17), and `intermed0_reload_alias` (new `MM_PARTIALS_RELOAD_ALIAS`
  define replacing the raw-index define `MM_PARTIALS_RELOAD_ALIAS_CB`). The last two are never
  emitted by this factory; both branches exist so the fork's inherited interface is complete for
  the five later consumers.
- [Avoid varargs unless absolutely necessary](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary):
  retained deliberately in both DM kernels — genuine indexed collections, one data-selected by a
  runtime index, the other a runtime-counted loop. Every other argument is named.
- [Porting a shared kernel](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  rung 2 for the compute kernel — fork beside the original plus a pointer comment in the original;
  rung "convert in place" for both DM kernels.
- Cursor surgery per [CB→DFB whitelist §D](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#d-cursor-surgery-evil_set_-only):
  the compute kernel's `get_local_cb_interface(...).fifo_rd_ptr = …` (`:114-117`) becomes
  `evil_set_read_ptr(...)`, inside the `MM_PARTIALS_RELOAD_ALIAS` gate.
- `constexpr` metadata exception per [whitelist §A](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#tile--format-metadata-jit-descriptors):
  the in0 kernel's `constexpr` `get_tile_size(dfb_id_in0)` and `get_dataformat(dfb_id_in0)`
  (`:55-56`) keep the free-function form with the binding token.

---

## Deferred / Flagged

New findings from planning, all carried to `METAL2_PORT_REPORT.md`:

1. **The brief's "RTA varargs: none" is wrong** — both DM kernels read arg arrays through
   `get_arg_addr`. Audit scan gap, not a port blocker.
2. **Two dead positional CTA slots** (in0 sender 13, in1 writer 14) and **one dead semaphore**
   (id 2) that the audit did not identify.
3. **`num_storage_cores` vs. S** — the in0 y-array base offset is only correct when
   `num_blocks % S == 0`. Preserved verbatim.
4. **Non-zero semaphore initial value** requires the `[[deprecated]]`
   `SemaphoreAdvancedOptions::initial_value` for the dead semaphore id 2. It is dead, so the value
   is unobservable, but dropping the semaphore would change the program's semaphore allocation and
   is out of a syntax-swap's remit.
5. **`c_2`/`c_6` are `LocalTensorAccessor` candidates** for the post-port `sync_free_dfbs` style
   pass — sync-free *and* borrowed. Not done here.
