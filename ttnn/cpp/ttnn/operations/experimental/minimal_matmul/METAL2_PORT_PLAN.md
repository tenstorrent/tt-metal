# Port Plan — `experimental/minimal_matmul`

Port plan for `MinimalMatmulDeviceOperation::ProgramFactory`, ported from the `ProgramDescriptor`
API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port.

Inputs: `METAL2_PORT_BRIEF.md` (provisional — see `METAL2_PORT_REPORT.md` → *Precondition*) and
`METAL2_PREPORT_AUDIT.md`.

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — `create_descriptor` returning a
  `ProgramDescriptor`.
- **Where the methods live:** in a **nested `ProgramFactory` struct** with
  `using program_factory_t = std::variant<ProgramFactory>`
  (`device/minimal_matmul_device_operation.hpp:28-42`); bodies in
  `device/minimal_matmul_program_descriptor.cpp`. **Not** the direct-descriptor shape — the
  `ttnn_factory.md` §3 exception does not apply.
- **Variants:** single.
- **Custom `compute_program_hash`:** none — default reflection-based hash, no `attribute_values` /
  `to_hash` backdoor. Nothing to preserve.
- **`override_runtime_arguments`:** present (`device/minimal_matmul_program_descriptor.cpp:951`) →
  target concept is the **custom** one.

**Not part of this unit** (same directory, different owners):
`device/minimal_matmul_program_factory.cpp` — the legacy `Program&` emitter for the CCL composite;
`device/minimal_matmul_fabric_bound_program_factory.cpp` + its four `fabric_bound_*` kernels —
a free-standing factory audited with `strided_all_gather_minimal_matmul_async`.

### Kernels

Five `KernelDescriptor`s from three sources. Core ranges below are the **non-transposed**
(`M ≤ N`) orientation; `transpose_core_grid` swaps the row/column roles *and* the
processor/NOC assignment (`:296-304, :377-380`).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `in0_sender` | `device/kernels/dm_in0_sender.cpp` | `CoreRange({0,0},{0,gy-1})` (column x=0) | 22 (`:540-563`) + accessor args | none | 14 + 3(ternary) + N_chunks, per core | none | base, or `+IN0_VIRTUAL_CONCAT`/`IN0_K_SPLIT_TILES` | resolved **O2** (unset) | `DataMovementConfigDescriptor{in0_risc, in0_noc}` |
| `in0_receiver` | `device/kernels/dm_in0_sender.cpp` | `CoreRange({1,0},{gx-1,gy-1})` (x≥1) | 22 (`:586-609`), `is_injector_core=false` | none | same | none | base | resolved **O2** | same |
| `in1_sender` | `device/kernels/dm_in1_sender_out.cpp` | `CoreRange({0,0},{gx-1,0})` (row y=0) | 21 (`:627-649`) + accessor args | none | 13 + 3(ternary) + N_chunks, per core | none | base | resolved **O2** | `DataMovementConfigDescriptor{in1_risc, in1_noc}` |
| `in1_receiver` | `device/kernels/dm_in1_sender_out.cpp` | `CoreRange({0,1},{gx-1,gy-1})` (y≥1) | 21 (`:667-689`), `is_injector_core=false` | none | same | none | base | resolved **O2** | same |
| `compute` | `device/kernels/compute.cpp` | full `core_grid` | 8 (`:707-715`) | none | 4 + 2(ternary), per core | none | base + activation + throttle | resolved **O3** (unset on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, math_approx_mode}` |

`grep -n opt_level` over the whole op directory returns **nothing** — so the compute spec needs an
explicit `KernelBuildOptLevel::O3` and the four DM specs need nothing.

**Resolved DM triples** (`preferred_noc_for_dram_read = NOC_0`, `..._write = NOC_1` on every arch;
`noc_mode` left at the descriptor default `DM_DEDICATED_NOC`):

| | `transpose_core_grid == false` | `transpose_core_grid == true` |
|---|---|---|
| in0 sender/receiver | `RISCV_1`, `NOC_1` | `RISCV_0`, `NOC_0` |
| in1 sender/receiver | `RISCV_0`, `NOC_0` | `RISCV_1`, `NOC_1` |

**None matches the reader (`RISCV_1`/`NOC_0`) or writer (`RISCV_0`/`NOC_1`) default** — every DM
spec is a custom `DataMovementGen1Config`.

### CBs

All seven use `CoreRangeSet(core_grid)` (the full grid) and a single `CBFormatDescriptor`; none sets
`tile`, none is buffer-backed, none sets `address_offset`. Built by the `push_cb` lambda at `:414-423`.

| index | total_size | data_format | page_size | num pages | present when |
|---|---|---|---|---|---|
| `c_0` in0 | `pages × page_size` | `in0_data_format` (activation dtype) | `in0_tile_size` | `M_block_tiles × K_block_tiles × 2` | always |
| `c_1` in1 | ″ | `in1_data_format` (weight dtype) | `in1_tile_size` | `K_block_tiles × N_block_tiles × 2` | always |
| `c_2` out | ″ | `output_data_format` | `out_tile_size` | `out_block_num_tiles_written × 2` (halved under SwiGLU) | always |
| `c_3` intermediate | ″ | `fp32_dest_acc_en ? Float32 : Float16_b` | `intermediate_tile_size` | `M_block_tiles × N_block_tiles` (**not** double-buffered) | always |
| `c_4` in2/bias | ″ | `in2_data_format` (bias dtype) | `in2_tile_size` | `N_block_tiles` | `use_bias` |
| `c_5` ternary_a | ″ | `ternary_a_data_format` (== `in1_data_format`, asserted `:452-453`) | `ternary_a_tile_size` | `M_block_tiles × N_block_tiles` | `use_fused_ternary` |
| `c_6` ternary_b | ″ | `ternary_c_data_format` | `ternary_c_tile_size` | `N_block_tiles` | `use_fused_ternary` |

### Semaphores

Six, all `CoreType::WORKER` over `CoreRangeSet(core_grid)`, ids hard-coded 0-5 as literals and
passed to kernels as CTAs 14-16 (`:388-408, :555-557`).

| id | name | initial_value | bound by |
|---|---|---|---|
| 0 | `in0_sender` | `INVALID` | in0 kernels |
| 1 | `in0_receiver` | `INVALID` | in0 kernels |
| 2 | `in0_valid` | `VALID` | in0 kernels |
| 3 | `in1_sender` | `INVALID` | in1 kernels |
| 4 | `in1_receiver` | `INVALID` | in1 kernels |
| 5 | `in1_valid` | `VALID` | in1 kernels |

### Tensor accessors

Host-side `TensorAccessorArgs(...).append_to(cta)` all flow through the `append_accessors` helper
(`:107-132`); addresses are delivered as **`Buffer*` in the RTA list**, not `->address()`.

| host site | originating Tensor | RTA slot (host) | kernel site |
|---|---|---|---|
| `:115` (`main_tensor`) | `input_tensor` (in0 kernels) / `weight_tensor` (in1 kernels) | in0 idx 0 / in1 idx 0 | `dm_in0_sender.cpp:68` / `dm_in1_sender_out.cpp:67` |
| `:116-118` (loop) | each of `tensor_return_value[0..N)` | tail, `out_addr_rt_arg_idx + i` | `matmul_dataflow_common.hpp:33-34` (3-arg — page size drops) |
| `:119-121` | `bias_tensor` | in0/in1 idx 1 | `dm_in0_sender.cpp:79`, `dm_in1_sender_out.cpp:78` |
| `:123-125` | `optional_input_tensor` (in3) | in0 idx 2 | `dm_in0_sender.cpp:195` |
| `:126-128` | `fused_ternary_input_a` | in0 idx 14 / in1 idx 13 | `dm_in0_sender.cpp:117`, `dm_in1_sender_out.cpp:97` |
| `:129-131` | `fused_ternary_input_b` | in0 idx 15 / in1 idx 14 | `dm_in0_sender.cpp:118`, `dm_in1_sender_out.cpp:98` |

### Work split

**Not** `split_work_to_cores`. A custom 2-D grid split (`:299-335, :781-828`):

- `in0_parallel_axis_cores` = `grid.x` if transposed else `grid.y`; `in1_parallel_axis_cores` the other.
- `M_tiles_per_core = round_up(M_tiles, in0_parallel_axis_cores) / in0_parallel_axis_cores`
- `N_tiles_per_core = round_up(N_tiles, in1_parallel_axis_cores) / in1_parallel_axis_cores`
  (SwiGLU partitions on gate/up **pairs**, `:318-328`)
- Per-core `[M_start,M_end) × [N_start,N_end)` from `in0_idx`/`in1_idx` = the core's coordinate on
  each axis. **Uniform** per-core ranges are load-bearing — the comment at `:777-779` warns that
  non-uniform counts deadlock the forwarding handshake.
- `num_cores = core_grid.size()`; every core participates.

### Shared kernels

**Lent / intra-op.** All four sources below are bound by a *second* emitter that will not convert in
this change: `minimal_matmul_factory_helper_common` in `device/minimal_matmul_program_factory.cpp`
(`:551, 596, 635, 674, 704` — identical file paths), reached from
`experimental/ccl/minimal_matmul_strided_reduce_scatter_async`, whose sheet row is
`Concept = legacy (MeshWorkload)`, `Is able to port? = no`.

| kernel | `_metal2` fork exists? | rung |
|---|---|---|
| `device/kernels/dm_in0_sender.cpp` | no | **2 — create** |
| `device/kernels/dm_in1_sender_out.cpp` | no | **2 — create** |
| `device/kernels/compute.cpp` | no | **2 — create** |
| `device/kernels/matmul_dataflow_common.hpp` | no | **2 — create** (header, forked with its includers) |

`ls device/kernels/` confirms no `*_metal2.*` sibling. No tree-wide `_metal2` grep was used (the
recipe's locational check); the `experimental/quasar/` tree is out of bounds and was not consulted.

**Fork lands in this op's own directory** (the *lent* case), so no peer-directory carve-out is
needed. The legacy originals get the pointer comment; nothing else in them changes.

### Flags

- `device/kernels/fabric_bound_*` (4 files) are **unreferenced by this factory** — they belong to
  `minimal_matmul_fabric_bound_program_factory.cpp`. Not audited, not ported.
- The `#ifdef MM_WINDOW_BLOCKS` / `FUSE_AG` / `SRS_FUSE_OP_SIGNALER` regions inside the three
  kernels are dead for every configuration this factory emits (those defines come only from the
  legacy emitter). They are copied into the fork verbatim and left alone — converting them is
  neither required nor permitted.
- Dead legacy args to carry across unchanged: `in3_tile_size` (in0 CTA index 21, never read) and
  `max_defer_write_k_block` (RTA idx 13, used only under `SRS_FUSE_OP_SIGNALER`).

## TTNN ProgramFactory

- **Concept:** **`ProgramSpecFactoryConcept`** (base) — **deviates from the audit/recipe rule, on the
  invoker's explicit decision.** The rule maps the factory's existing `override_runtime_arguments`
  to `CustomProgramSpecFactoryConcept`; the invoker chose to delete the override and rely on the
  framework's automatic tensor-binding refresh instead, because the override refreshes *only* tensor
  addresses and nothing else. Full rationale, what is given up, and the safety check are in
  `METAL2_PORT_REPORT.md` → *Concept deviation*.
- **Custom `compute_program_hash`:** none.
- **Implementation notes:**
  - `create_descriptor` → `create_program_artifacts` returning
    `ttnn::device_operation::ProgramArtifacts{.spec = …, .run_params = …}`; no `op_owned_tensors`.
  - `override_runtime_arguments` is **removed** from `ProgramFactory` — both the declaration
    (`device/minimal_matmul_device_operation.hpp:34-39`) and the definition
    (`device/minimal_matmul_program_descriptor.cpp:951-1030`). Leaving a `void`-returning one behind
    would be inert (`HasSpecRuntimeArgsOverride` is keyed on the return type) but misleading.
  - With it go the RTA-layout constants it indexed with (`:151-166`) and the cache-miss `TT_FATAL`
    guard that protected them (`:921-941`) — the guard's whole subject is the arg layout
    `override_runtime_arguments` walked, so it is a *subject-deleted* loss, the one legitimate kind.
    Recorded in the report's TT_FATAL census.
  - The struct edits live in `device/minimal_matmul_device_operation.hpp:28-42`; that is the factory,
    not the device-op class, so it is in scope.

## Planned Spec Shape

- **KernelSpecs (5):** `in0_sender`, `in0_receiver`, `in1_sender`, `in1_receiver`, `compute` — 1:1
  with the legacy descriptors. The two same-source pairs cover **disjoint** node sets, so this is
  ordinary 1:1 multiplicity, not the demoting anti-pattern and not a two-toucher assignment.
- **DataflowBufferSpecs (4 unconditional + 3 conditional):** `in0`, `in1`, `out`, `intermediate`;
  `in2` when `use_bias`; `ternary_a` / `ternary_b` when `use_fused_ternary`. `entry_size` =
  the legacy `page_size`, `num_entries` = the legacy page count. No `borrowed_from`, no
  `alias_with`, no `allow_instance_multi_binding`.
- **SemaphoreSpecs (6):** `in0_sender`, `in0_receiver`, `in0_valid`, `in1_sender`, `in1_receiver`,
  `in1_valid`, each with `target_nodes` = the full grid.
- **TensorParameters (2 + N + 4 conditional):** `in0`, `in1`, `out0 … out{N_chunks-1}`; plus `in2`,
  `in3`, `ternary_a`, `ternary_b` under their conditions.
- **WorkUnitSpecs (4):** the four distinct (kernel-set, node-set) pairings the sender/receiver
  ranges produce. Non-transposed:

  | nodes | kernels |
  |---|---|
  | `x=0, y=0` | `in0_sender`, `in1_sender`, `compute` |
  | `x≥1, y=0` | `in0_receiver`, `in1_sender`, `compute` |
  | `x=0, y≥1` | `in0_sender`, `in1_receiver`, `compute` |
  | `x≥1, y≥1` | `in0_receiver`, `in1_receiver`, `compute` |

  (Transposed swaps the axis roles; the four-way shape is unchanged.)
- **Op-owned tensors:** none.

### DFB endpoint bindings

| DFB | PRODUCER | CONSUMER | note |
|---|---|---|---|
| `in0` | the node's in0 kernel | `compute` | plain 1:1 |
| `in1` | the node's in1 kernel | `compute` | plain 1:1 |
| `out` | `compute` | the node's **output-writer** DM kernel | conditional binding (§ below) |
| `intermediate` | `compute` | `compute` | **self-loop** — one toucher |
| `in2` | the node's **non-writer** DM kernel | `compute` | conditional ×2 |
| `ternary_a` | the node's non-writer DM kernel | `compute` | conditional |
| `ternary_b` | the node's non-writer DM kernel | `compute` | conditional |

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `in0_sender` + `in0_receiver` (`dm_in0_sender.cpp`) | `in0_sender`, `in0_receiver` | the four above | `in0` PRODUCER (both); `out` CONSUMER *or* `in2`/`ternary_*` PRODUCER, depending on `is_output_writer` |
| `in1_sender` + `in1_receiver` (`dm_in1_sender_out.cpp`) | `in1_sender`, `in1_receiver` | ″ | same, with the roles swapped by `transpose_core_grid` |

Disjoint node sets per pair ⇒ each node sees exactly one instance of each source ⇒ ordinary 1:1.
**No `allow_instance_multi_binding` anywhere.**

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `program_descriptor.cpp:834-836, 867-868` | `Buffer*` pushed into `RTArgList` (in0/in1/in2/in3) | `TensorBinding` on the KernelSpec |
| `program_descriptor.cpp:850-851, 882-883` | `Buffer*` (ternary_a/b) | `TensorBinding` (conditional) |
| `program_descriptor.cpp:856, 888` | `Buffer*` per output, in a loop | `TensorParameter` `out{i}` + `TensorBindingSequence{"outputs"}` |
| `program_descriptor.cpp:107-132` (`append_accessors`) | `TensorAccessorArgs(buf).append_to(cta)` ×6 | binding mechanism end-to-end — helper deleted |
| `dm_in0_sender.cpp:67, 71-72, 76-78, 189-193`; `dm_in1_sender_out.cpp:66, 70-71, 75-77` | `TensorAccessorArgs<N>()` + `next_compile_time_args_offset()` chains | `TensorAccessor(tensor::name)` |
| `matmul_dataflow_common.hpp:25-47` | `make_tensor_accessor_tuple_uniform_page_size` (address RTAs + 3rd-arg page size) | `make_tensor_accessors(tensor::outputs)` |
| `program_descriptor.cpp:553, 599, 640, 681` | `out_tile_size` CTA (idx 12) | **NOT dropped** — becomes named `args::out_tile_size`. It feeds the 3rd accessor argument *and* is the `tile_size_bytes` L1 stride at 8 further sites per DM kernel. Only the accessor argument goes. |
| `program_descriptor.cpp:555-557` etc. | semaphore-id CTAs 14-16 | `SemaphoreBinding` → `sem::name` |
| all five CTA lists (`:540-563, 586-609, 627-649, 667-689, 707-715`) | positional `compile_time_args` | named `compile_time_args` Table |
| `dm_in0_sender.cpp:148-152`, `dm_in1_sender_out.cpp:129-133`, `compute.cpp:427-434` | `constexpr uint32_t cb_*_id = tt::CBIndex::c_N` | `dfb::name` |
| in0/in1 RTA idx 0-2 (+14/15, 13/14), tail | address RTAs read via `get_arg_val` | bindings; remaining scalars become named args |

## Applied Patterns

- **Self-loop DFB binding** — `intermediate` (`c_3`) on the compute KernelSpec, bound PRODUCER *and*
  CONSUMER. One toucher in every configuration.
- **Conditional / optional resource bindings** — four sites, the bulk of the CB-side work:
  1. `out` on the non-output-writer DM instance (promote the `is_output_writer` CTA to a define).
  2. `in2` on the output-writer DM instance (same define, inverted).
  3. `ternary_a` / `ternary_b` metadata lookups on the writer instance
     (`dm_in0_sender.cpp:111-112`, `dm_in1_sender_out.cpp:92-93`) — sink into the `!is_output_writer`
     scope so the writer never binds them. Both are legacy `constexpr`, so whitelist rule 7's
     carve-out applies: keep the free-function form, `get_tile_size(dfb::ternary_a)`.
  4. `in2` / `ternary_*` in `compute.cpp` (`:433-434, :440, :535, :557`, and `:157` inside
     `add_bias_and_addcmul_block`) — `#ifdef FUSE_BIAS` / `FUSE_TERNARY` gate them; the host
     allocates no such CB in those configurations.
- **Caution: Porting a shared kernel** — rung 2 on all four sources (see *Shared kernels*).
- **Tensor binding sequence** (`KernelAdvancedOptions::tensor_binding_sequences`) — the N outputs.
  Not yet a catalog entry; this op appears to be its first production user.

## Deferred / Flagged

- **New finding (planning):** the `TensorBindingSequence` mechanism has **no in-tree production
  user** — `grep` finds it only in `kernel.cpp` / `kernel.hpp` / `advanced_options.hpp` and the
  kernel-side helpers in `tensor_accessor.h`. It is the documented answer for a variadic tensor-binding
  count and maps cleanly onto this op's `outputs_tuple`, but expect to shake it out. Also absent
  from the patterns catalog; a catalog entry is a candidate deliverable of this port.
- **Not a port task, recorded for the ops team** (already in the audit): the dead `in3_tile_size`
  CTA, the dead `max_defer_write_k_block` RTA, the inaccurate page-size comment at
  `matmul_dataflow_common.hpp:31-32`, and the two hand-synchronized emitters.
- **Verification gap:** the CCL consumers of the legacy kernels are covered only by t3000 / TG /
  galaxy tests; this bench has a single Blackhole p150b. The fork strategy makes them
  unaffected *by construction*, but that is unmeasured here. Recorded as a handoff point.
