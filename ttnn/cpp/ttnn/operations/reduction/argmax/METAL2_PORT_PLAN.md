# Port Plan — `reduction/argmax` (`ArgMaxDeviceOperation`)

Port plan for `ttnn/cpp/ttnn/operations/reduction/argmax`, ported from the legacy
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope: `ArgMaxDeviceOperation` only** — `ArgMaxSingleCoreProgramFactory` +
`ArgMaxMultiCoreProgramFactory` and the four kernels they bind.
`ArgMaxNCDeviceOperation` / `ArgMaxNCProgramFactory` (and its three kernels) are **out of
scope and untouched** — still on the legacy imperative `host_api.hpp` builder, blocked pending
its `ProgramDescriptor` migration (audit gate).

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — `create_descriptor()` returning
  `tt::tt_metal::ProgramDescriptor`, declared at `device/argmax_device_operation.hpp:17`
  (single-core) and `:22` (multi-core).
- **Variants:** two factories in a `program_factory_t` variant
  (`std::variant<ArgMaxSingleCoreProgramFactory, ArgMaxMultiCoreProgramFactory>` @
  `device/argmax_device_operation.hpp:31`), selected at runtime by `select_program_factory`
  (`argmax_device_operation.cpp:76`) via `uses_multicore_path`. **Both port together** — see
  *Flags*.
- **Factory methods live in factory structs**, not on the device-operation. So
  [`ttnn_factory.md` exception 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md)
  (direct-descriptor shape) does **not** apply — the port is a method swap inside the
  existing structs.
- **Custom `compute_program_hash`:** none — default reflection-based hash. No backdoor
  `attribute_values` / `to_hash` either. Nothing to leave alone.
- **`override_runtime_arguments`:** absent on both factories → base
  `ProgramSpecFactoryConcept`.

---

### Variant: `ArgMaxSingleCoreProgramFactory`

The factory selects **one of three kernel sources at runtime**
(`argmax_single_core_program_factory.cpp:185-193`), and the CTA list differs per source
(`get_ctime_args_single_core`, `:48`). All three convert together.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (RM) | `kernels/reader_argmax_interleaved.cpp` | `all_cores` (1 node) | 0 `src_cb_index`, 1 `dst_cb_index`, 2 `src_page_size`, 3 `dst_page_size`, 4 `outer_dim_units`, 5 `inner_dim_units`, 6 `red_dim_units`, 7 `reduce_all`, then `TensorAccessorArgs(input)` @ 8, `TensorAccessorArgs(output)` chained | none | per node: `{input, output}` (MeshTensor bindings) @ `:205` | none | none | unset → **O2** (DM) | `ReaderConfigDescriptor{}` |
| reader (TILE-W) | `kernels/reader_argmax_tile_layout.cpp` | `all_cores` (1 node) | 0 `src_cb_index`, 1 `dst_cb_index`, 2 `src_page_size`, 3 `dst_page_size` *(dead — kernel skips 3)*, 4 `tile_height`, 5 `tile_width`, 6 `h_tiles`, 7 `w_tiles`, 8 `h_logical`, 9 `w_logical`, 10 `outer_dim_units`, 11 `reduce_all`, 12 `keepdim`, then TA args @ 13 | none | same | none | none | unset → **O2** (DM) | `ReaderConfigDescriptor{}` |
| reader (TILE-H) | `kernels/reader_argmax_tile_layout_h.cpp` | `all_cores` (1 node) | as TILE-W but **without** 11/12; TA args @ 11 | none | same | none | none | unset → **O2** (DM) | `ReaderConfigDescriptor{}` |

`opt_level` confirmed mechanically: `grep -n opt_level` over both factory `.cpp`s returns
nothing → `std::nullopt` → resolves to `O2` on a reader/DM descriptor. Both factories build
**only DM kernels**, so [Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
rule 2 (compute → explicit `O3`) does not fire anywhere in this port; Metal 2.0's `O2` default
already reproduces legacy.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (src) | `src_page_size` | `all_cores` | `input_data_format` | `src_page_size` | not set |
| `c_1` (dst) | `dst_page_size` | `all_cores` | `output_data_format` | `dst_page_size` | not set |

`total_size == page_size` on both → **one entry each**. No `GlobalCircularBuffer`, no
`address_offset`, no multi-element `format_descriptors` (no aliasing).

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `argmax_single_core_program_factory.cpp:182` (`TensorAccessorArgs(input).append_to`) | `tensor_args.input` | RTA 0 (`input`) @ `:205` |
| `argmax_single_core_program_factory.cpp:183` (`TensorAccessorArgs(output).append_to`) | `tensor_return_value` | RTA 1 (`output`) @ `:205` |

Kernel-side pairs: `reader_argmax_interleaved.cpp:41,42`;
`reader_argmax_tile_layout.cpp:49,50`; `reader_argmax_tile_layout_h.cpp:44,45`.
Both bindings are **Case 1** (consumed via `TensorAccessor`, never as a raw base pointer).

#### Work split

- Driver: `split_work_to_cores(grid_size, /*num_units=*/1)` @ `:133`.
- num_cores: 1 · all_cores: the single resulting node · groups 2-6 discarded (`unused_1..4`).
- `cores = grid_to_cores(num_cores, grid_size.x, grid_size.y, false)` @ `:203` → one node.

---

### Variant: `ArgMaxMultiCoreProgramFactory`

One kernel source, instantiated **once or twice** over disjoint core groups.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader_desc0 | `kernels/reader_argmax_interleaved_multicore.cpp` | `cores0` | 0-27 (below) + TA(input) @ 28 + TA(output) | none | per node: `{input, output, core_id, src_offset, red_dim_offset, src_read_size, red_dim_units_this_core}` @ `:394` | none | none | unset → **O2** (DM) | `DataMovementConfigDescriptor{RISCV_1, NOC::RISCV_1_default, DM_DEDICATED_NOC}` |
| reader_desc1 *(only if `num_cores1 > 0`)* | same source | `cores1` | identical CTA vector | none | same shape, different offsets @ `:424` | none | none | unset → **O2** (DM) | identical |

Positional CTA list (`argmax_multi_core_program_factory.cpp:342-372`), with the **kernel's**
name for each slot (`reader_argmax_interleaved_multicore.cpp:237-297`):

| # | host value | kernel name |
|---|---|---|
| 0 | `src_cb_idx` | `src_cb_idx` |
| 1 | `dst_cb_idx` | `dst_cb_idx` |
| 2 | `red_idxs_cb_idx` | `red_idxs_cb_idx` |
| 3 | `red_vals_cb_idx` | `red_vals_cb_idx` |
| 4 | `src_page_size` | *(never read — dead CTA)* |
| 5 | `dst_page_size` | `dst_page_size` |
| 6 | `red_idxs_page_size / num_total_cores` | `red_idx_size_per_core` |
| 7 | `red_vals_page_size / num_total_cores` | `red_val_size_per_core` |
| 8 | `outer_dim_units` | `outer_dim_units` |
| 9 | `inner_dim_units` | `inner_dim_units` |
| 10 | `red_dim_units` | `red_dim_units` |
| 11 | `reduce_all` | `reduce_all` |
| 12 | `num_total_cores` | `num_cores` |
| 13 | `reduce_core_id` | `reduce_core_id` |
| 14 | `reduce_core.x` | `reduce_core_x` |
| 15 | `reduce_core.y` | `reduce_core_y` |
| 16 | **`end_core0.x`** | `start_core_x0` |
| 17 | **`end_core0.y`** | `start_core_y0` |
| 18 | **`start_core0.x`** | `end_core_x0` |
| 19 | **`start_core0.y`** | `end_core_y0` |
| 20 | **`end_core1.x`** | `start_core_x1` |
| 21 | **`end_core1.y`** | `start_core_y1` |
| 22 | **`start_core1.x`** | `end_core_x1` |
| 23 | **`start_core1.y`** | `end_core_y1` |
| 24 | `num_cores_range0` | `num_cores0` |
| 25 | `num_cores_range1` | `num_cores1` |
| 26 | `start_sem_idx` | `start_sem_idx` |
| 27 | `done_sem_idx` | `done_sem_idx` |

> **Slots 16-23 carry a deliberate start/end swap** — the host comments it
> `// end comes before start for NOC1` (`:359`). The kernel's `start_core_*0` receives the
> group's **end** coordinate and its `end_core_*0` receives the **start**, because the NOC_1
> multicast rectangle is addressed end-corner-first. The named port **preserves the value
> mapping, not the label**: `{"start_core_x0", end_core0.x}`. Renaming the kernel's variables
> instead would be kernel-logic surgery outside the whitelist.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (src, group 0) @ `:218` | `round_up_to_mul32(red_dim_units0 * input_unit_size)` | `cores0` | `input_cb_data_format` | same | not set |
| `c_0` (src, group 1) @ `:231`, **`if (num_cores1 > 0)`** | `round_up_to_mul32(red_dim_units1 * input_unit_size)` | `cores1` | `input_cb_data_format` | same | not set |
| `c_1` (dst) @ `:244` | `dst_page_size` | `all_cores` | `output_cb_data_format` | same | not set |
| `c_2` (red_idxs) @ `:257` | `round_up_to_mul32(output_last_dim*output_unit_size) * num_total_cores` | `all_cores` | `output_cb_data_format` | same | not set |
| `c_3` (red_vals) @ `:270` | `round_up_to_mul32(output_last_dim*input_unit_size) * num_total_cores` | `all_cores` | `input_cb_data_format` | same | not set |

All five have `total_size == page_size` → **one entry each**. No GlobalCircularBuffer, no
`address_offset`, no aliasing.

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (`start_sem_idx`) @ `:303` | `WORKER` | `all_cores` | 0 |
| 1 (`done_sem_idx`) @ `:309` | `WORKER` | `all_cores` | 0 |

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `argmax_multi_core_program_factory.cpp:373` | `tensor_args.input` | RTA 0 @ `:394` / `:424` |
| `argmax_multi_core_program_factory.cpp:374` | `tensor_return_value` | RTA 1 @ `:394` / `:424` |

Kernel-side pair: `reader_argmax_interleaved_multicore.cpp:299,300`. Both **Case 1**.

#### Work split

- Driver: `distribute_work_to_cores(...)` @ `:195`, which either honours `sub_core_grids`
  or calls `tt::tt_metal::split_work_to_cores(core_grid, div_up(red_dim_units, min_red_dim_units_per_core))`.
- Returns `(all_cores, cores0, cores1, red_dim_units0, red_dim_units1)`;
  `num_cores0 = cores0.num_cores()`, `num_cores1 = cores1.num_cores()`,
  `num_total_cores = num_cores0 + num_cores1`.
- Per-node coords: `corerange_to_cores(cores0, num_cores0, true)` / `(cores1, num_cores1, true)`.

---

### Shared kernels

| kernel | class | other consumers | `_metal2` fork present? | rung |
|---|---|---|---|---|
| `kernels/reader_argmax_interleaved.cpp` | **lent** (out-of-directory *test* consumer) | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:126` (`TestGenericOpArgmaxSingleCore`) file-path-instantiates it from a hand-built `ProgramDescriptor`, CTA/RTA contract hardcoded at `:105-121` | **no** | **rung 2 — create the fork** |
| `kernels/reader_argmax_interleaved_multicore.cpp` | private | none | n/a | convert in place |
| `kernels/reader_argmax_tile_layout.cpp` | private | none | n/a | convert in place |
| `kernels/reader_argmax_tile_layout_h.cpp` | private | none | n/a | convert in place |

Census run as `grep -rl <filename> ttnn/cpp/ttnn/operations/ tests/`; the only non-factory,
non-`METAL2_*.md` hit in the whole set is the gtest above.

**Resolution (confirmed with the invoker; audit *Questions* #1, option (a)):** create
`kernels/reader_argmax_interleaved_metal2.cpp` beside the original, point the single-core
factory's RM branch at it, add the pointer comment to the legacy original, and leave
`test_generic_op.cpp` untouched on the legacy copy. `generic_op` / `ProgramDescriptor` cannot
supply Metal 2.0 named bindings, so the test could not be re-pointed even in principle. The
legacy copy is a **sunset** item recorded in `METAL2_PORT_REPORT.md`, not authorization to
convert it in place.

### In-directory kernel headers (not shared outside the op)

`kernels/argmax_common.hpp`, `kernels/argmax_tile_layout.hpp`, `kernels/argmax_tile_h_col.hpp`
are included only by the four in-scope readers (and, for `argmax_common.hpp`, by the retained
legacy fork). They take **raw `uint32_t` L1 addresses and `DataFormat` NTTPs** — no CB object,
no CB id, no FIFO call — so they need no API change. They do carry `cb`-flavoured *identifier*
names for the buffer whose address they hold (`InputContext::cb_addr`,
`OutputContext::output_cb_addr`, `compare_values(src_cb_addr, …)`, `dst_cb_mem`); those follow
the `cb_* → dfb_*` rename with the rest of the port. Renames are positional-parameter /
member renames, invisible to the retained legacy fork.

### Flags

- **Both factories must convert in the same change.** They live in one `program_factory_t`
  variant selected at runtime; the variant itself tolerates mixed concepts, but leaving one on
  `create_descriptor` while the header declares only `create_program_artifacts` for the other
  is not the shape here — the brief instructs both, and both are tractable in one pass.
- **Dead CTAs (known, *not* the port's to fix — routed to the ops team via the report):**
  index 4 (`src_page_size`) in the multi-core reader; index 3 (`dst_page_size`) in both TILE
  readers. They become **named CTAs with no kernel-side reader**. Kept and named, per the
  brief — not silently deleted.
- **`reduce_core_id` narrowed through `(bool)`** @ `reader_argmax_interleaved_multicore.cpp:273`.
  Latent, inert today (value 0). Left exactly as-is; reported.
- **Dummy `(0,0)` core-range CTAs for the single-group case** @
  `argmax_multi_core_program_factory.cpp:290`. Left as-is.
- No unreferenced kernel file in the in-scope set. No descriptor type outside the audit's
  Appendix A scan.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` (base) — neither in-scope
  factory has an `override_runtime_arguments`, so the framework refreshes tensor bindings on a
  cache hit and each factory writes exactly one method.
- **Custom `compute_program_hash`:** none — default reflection-based hash. Untouched.
- **Pybind:** `argmax_nanobind.cpp` binds only the user-facing `ttnn::argmax`; there is no
  pybound `create_descriptor`, so **no pybind deletion and no user-visible API change**
  (`ttnn_factory.md` exceptions 1 and 2 do not fire).
- **Implementation notes:** resource-name constants (`KernelSpecName` / `DFBSpecName` /
  `SemaphoreSpecName` / `TensorParamName`) are declared **function-locally inside
  `create_program_artifacts`**, not at namespace scope. Namespace-scope `const` objects have
  internal linkage, and the two factory `.cpp`s share a unity-build translation unit, so
  identically-named constants in both would collide
  ([Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)).
  Function-local declarations sidestep it without prefixing.

---

## Planned Spec Shape

### Variant: `ArgMaxSingleCoreProgramFactory`

- **KernelSpecs:** 1 — `READER{"reader"}`, source selected by the same runtime branch as
  legacy (RM → the new `_metal2` fork; TILE-H → `reader_argmax_tile_layout_h.cpp`;
  TILE-W → `reader_argmax_tile_layout.cpp`). `hw_config =
  ttnn::create_reader_datamovement_config(device->arch())` — the legacy
  `ReaderConfigDescriptor{}` resolves to exactly the reader default triple
  (`RISCV_1`, `NOC_0`, `DM_DEDICATED_NOC`).
- **DataflowBufferSpecs:** 2 — `SRC{"src"}`, `DST{"dst"}`, each `entry_size = <page_size>`,
  `num_entries = 1`, `data_format_metadata` copied from the legacy CB.
  `tile_format_metadata` left `nullopt` (legacy `.tile` unset).
- **SemaphoreSpecs:** none.
- **TensorParameters:** 2 — `INPUT{"input"}`, `OUTPUT{"output"}`, from
  `input.tensor_spec()` / `output.tensor_spec()`. `relaxations` left at the strict default
  (audit: relaxation `none`).
- **WorkUnitSpecs:** 1 — `{"main", {READER}, all_cores}`.
- **Op-owned tensors:** none.
- **KernelRunArgs:** **none** — after the two address RTAs become `TensorBinding`s the reader
  has zero RTAs and zero CRTAs, so its `KernelRunArgs` entry is omitted entirely (permitted:
  "except for kernels that have no runtime or common runtime arguments").
  `ProgramRunArgs` carries only `tensor_args`.

Kernel-side accessor names (all three sources): `dfb::src`, `dfb::dst`, `tensor::src`,
`tensor::dst` — taken from the kernels' own vocabulary (`src_cb_idx`/`src_dfb_idx`,
`src_base_addr`, `s_src`).

### Variant: `ArgMaxMultiCoreProgramFactory`

- **KernelSpecs:** 1 or 2 — `READER0{"reader0"}` over `cores0`, and `READER1{"reader1"}` over
  `cores1` **only when `num_cores1 > 0`**. Same source. `hw_config` is a **custom** Gen1
  triple, replicated verbatim (see *Applied Patterns*).
- **DataflowBufferSpecs:** 4 or 5 — `SRC0{"src0"}`, `SRC1{"src1"}` *(conditional)*,
  `DST{"dst"}`, `RED_IDXS{"red_idxs"}`, `RED_VALS{"red_vals"}`; each `num_entries = 1` with
  `entry_size` = the legacy per-CB page size, `data_format_metadata` copied.
  **Declaration order is `SRC0, SRC1, DST, RED_IDXS, RED_VALS`** — matching the legacy
  `desc.cbs.push_back` order, because the DFB allocator walks
  `ProgramSpec::dataflow_buffers` in user order.
- **SemaphoreSpecs:** 2 — `START{"start"}`, `DONE{"done"}`, `target_nodes = all_cores`.
  (Metal 2.0 semaphores are zero-initialised; legacy `initial_value` was 0 on both.)
- **TensorParameters:** 2 — `INPUT{"input"}`, `OUTPUT{"output"}`; strict matching.
- **WorkUnitSpecs:** 1 or 2 — `{"group0", {READER0}, cores0}` and, conditionally,
  `{"group1", {READER1}, cores1}`.
- **Op-owned tensors:** none.
- **KernelRunArgs:** one per instantiated reader, built with `AddRuntimeArgsForNode` from the
  existing node-first loop (loop nesting left as-is; no name-first restructure).

Kernel-side accessor names: `dfb::src` (READER0→SRC0, READER1→SRC1 — one name, two specs,
one per KernelSpec), `dfb::dst`, `dfb::red_idxs`, `dfb::red_vals`, `sem::start`, `sem::done`,
`tensor::src`, `tensor::dst`.

#### `c_0` declared twice at one buffer index, over disjoint core ranges

Legacy declares `c_0` twice with **genuinely different sizes** on the default (no
`sub_core_grids`) path. Metal 2.0 has no buffer index, so this becomes **two
`DataflowBufferSpec`s** (`SRC0`, `SRC1`) with the two legacy sizes, each bound by exactly one
reader, each placed (derived) on that reader's disjoint node set. The second spec stays
conditional on `num_cores1 > 0`, matching legacy.

#### Address-uniformity requirement — **checked, holds**

The kernel computes the reducer's destination from its **own** `c_2`/`c_3` base pointer
(`red_idx_cb.get_write_ptr() + core_id * red_idx_size_per_core`, then writes to
`{reduce_core_x, reduce_core_y, that address}` — `reader_argmax_interleaved_multicore.cpp:326-339,
416-427, 467-478`). That is only correct if `c_2`/`c_3` land at the **same L1 address on every
node**, even though `SRC0`/`SRC1` differ in size between the two groups.

Verified against the Metal 2.0 allocator rather than assumed:
`ProgramImpl::allocate_dataflow_buffers` (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2505-2571`)
computes **one** `alloc_addr` per DFB as the max `get_cb_region_end()` across every core range
the DFB spans, then writes that single address into every core's entry — the code says so
explicitly: *"All cores of a DFB get the same alloc_addr so the L1 buffer is at a uniform
absolute address on every physical core."* This is a byte-for-byte analog of the legacy
`ProgramImpl::allocate_circular_buffers` behaviour the audit cited
(`tt_metal/impl/program/program.cpp:1719-1751`). Since `RED_IDXS` / `RED_VALS` span
`all_cores`, their address clears the larger of the two `SRC` sizes on every node. Preserving
the legacy declaration order keeps the resulting addresses identical to legacy as well.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `reader_desc0` @ `argmax_multi_core_program_factory.cpp:376` (over `cores0`) and `reader_desc1` @ `:408` (over `cores1`, `if num_cores1 > 0`), both of `reader_argmax_interleaved_multicore.cpp` | `READER0`, `READER1` | `group0` (`cores0`), `group1` (`cores1`) | `DST`, `RED_IDXS`, `RED_VALS`: **each reader binds PRODUCER + CONSUMER (self-loop)**. `SRC0` bound only by `READER0`; `SRC1` only by `READER1`. |

This is the **disjoint-node** shape, *not* a dual-instance work-split: `cores0` and `cores1`
never overlap, so every node sees exactly one reader instance and every DFB's per-node
toucher census is **1**. Multiple `KernelSpec`s on one endpoint of `DST` / `RED_IDXS` /
`RED_VALS` are legal precisely because their node coverage is non-overlapping, the kernel kind
is the same (DM) and the binding-site parameters are identical (`STRIDED`, `num_threads = 1`)
— the `dataflow_buffer_spec.hpp` invariant. **No `allow_instance_multi_binding` anywhere.**

The single-core factory has no work-split multiplicity: one `KernelDescriptor` on one node.

---

## Dropped Plumbing

### Single-core factory

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `argmax_single_core_program_factory.cpp:205` RTA slot 0 | `emplace_runtime_args(core, {input, …})` — MeshTensor `BufferBinding` | `TensorParameter INPUT` + `TensorBinding{INPUT, "src"}` + `TensorArgument` |
| `argmax_single_core_program_factory.cpp:205` RTA slot 1 | `… {…, output})` | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT, "dst"}` + `TensorArgument` |
| `get_ctime_args_single_core` CTA slot 0 (`:68`, `:89`) | `src_cb_index` (`tt::CBIndex::c_0`) | `DFBBinding{SRC, "src", …}` |
| `get_ctime_args_single_core` CTA slot 1 (`:69`, `:90`) | `dst_cb_index` (`tt::CBIndex::c_1`) | `DFBBinding{DST, "dst", …}` |
| `argmax_single_core_program_factory.cpp:182` | `TensorAccessorArgs(input).append_to(ctime_args)` | binding mechanism (host-side CTA payload emitted by the framework) |
| `argmax_single_core_program_factory.cpp:183` | `TensorAccessorArgs(output).append_to(ctime_args)` | same |
| `reader_argmax_interleaved.cpp:41,42` | `TensorAccessorArgs<8>()` / `next_compile_time_args_offset()` | `TensorAccessor(tensor::src)` / `(tensor::dst)` |
| `reader_argmax_tile_layout.cpp:40,49,50` | `num_c_time_args = 13` + the `TensorAccessorArgs<…>` chain | same — and the `num_c_time_args` constant disappears with it |
| `reader_argmax_tile_layout_h.cpp:39,44,45` | `num_c_time_args = 11` + the chain | same |
| `reader_argmax_interleaved.cpp:15,16` RTA reads | `get_arg_val<uint32_t>(0/1)` for the two base addresses | gone — binding auto-injects the address |
| `reader_argmax_tile_layout.cpp:44,45`; `_h.cpp:41,42` | same | gone |
| all remaining positional CTAs | `get_compile_time_arg_val(N)` | named: `get_arg(args::<name>)` (names in *Planned Spec Shape*) |

**Page-size 3rd argument:** none — all accessor constructions are the 2-arg form (audit).
Note `src_page_size` / `dst_page_size` are **not** third-argument page sizes: they are the
kernel's own NoC transfer sizes (`noc.async_read(..., src_page_size, ...)`), so they stay as
named CTAs.

### Multi-core factory

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `argmax_multi_core_program_factory.cpp:394,424` RTA slots 0,1 | `{input, output, …}` MeshTensor bindings | `TensorParameter`/`TensorBinding`/`TensorArgument` ×2 |
| CTA slot 0 (`:343`) | `src_cb_idx` | `DFBBinding{SRC0 or SRC1, "src", …}` |
| CTA slot 1 (`:344`) | `dst_cb_idx` | `DFBBinding{DST, "dst", …}` |
| CTA slot 2 (`:345`) | `red_idxs_cb_idx` | `DFBBinding{RED_IDXS, "red_idxs", …}` |
| CTA slot 3 (`:346`) | `red_vals_cb_idx` | `DFBBinding{RED_VALS, "red_vals", …}` |
| CTA slot 26 (`:370`) | `start_sem_idx` | `SemaphoreBinding{START, "start"}` |
| CTA slot 27 (`:371`) | `done_sem_idx` | `SemaphoreBinding{DONE, "done"}` |
| `argmax_multi_core_program_factory.cpp:373,374` | the two `TensorAccessorArgs(...).append_to` calls | binding mechanism |
| `reader_argmax_interleaved_multicore.cpp:299,300` | `TensorAccessorArgs<28>()` / chain | `TensorAccessor(tensor::src)` / `(tensor::dst)` |
| `reader_argmax_interleaved_multicore.cpp:219,220` | `get_arg_val<uint32_t>(0/1)` base addresses | gone — binding auto-injects |
| RTA slots 2-6 | `get_arg_val<uint32_t>(2..6)` | named RTAs `core_id`, `src_offset`, `red_dim_offset`, `src_read_size`, `red_dim_units_this_core` |
| CTA slots 4-25 | `get_compile_time_arg_val(N)` | named CTAs (kernel-side names; slots 16-23 keep the legacy value↔name swap) |

**Retained varargs:** none. Every argument in every in-scope kernel is a distinct field read a
fixed number of times at a literal index — no counted loop, no data-selected index, no
sentinel scan. All become named.

---

## Applied Patterns

- **[Sync-free CB → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md):**
  all six single-core `(CB, config)` pairs and all five multi-core DFBs. Re-derived from the
  kernel-touch census rather than transcribed from the brief: every DFB in scope is touched by
  **exactly one** kernel per node, and that touch is a bare `get_write_ptr()` peek — there is
  no `reserve_back` / `push_back` / `wait_front` / `pop_front` and no `evil_set_*` cursor
  surgery anywhere in the four kernels. One toucher ⇒ self-loop (bind PRODUCER **and**
  CONSUMER under one accessor name). Census agrees with the brief on all ten.
  The multi-core cross-core NoC writes do not add touchers: the destination is a bare
  `{noc_x, noc_y, addr}` on the reducer, not a DFB binding.
- **[Multi-variant factory](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md):**
  the single-core factory's runtime source selection (RM / TILE-W / TILE-H) stays a branch
  inside `create_program_artifacts`; all three sources convert together.
- **[Porting a shared kernel — rung 2](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md):**
  `reader_argmax_interleaved_metal2.cpp` created beside the original; pointer comment added to
  the legacy file; nothing else in it changed.
- **[Pass DFB handles directly](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
  / CB→DFB whitelist §A `constexpr` carve-out:** the five `get_dataformat(<cb_idx>)` sites keep
  the **free-function** form with the binding token — `get_dataformat(dfb::src)` — because each
  is a `constexpr DataFormat` feeding non-type template parameters, and a `DataflowBuffer`
  object can never be a constant expression (`DataflowBuffer(uint16_t)` is a non-`constexpr`
  out-of-line constructor, `dataflow_buffer.h:113`). **This contradicts the brief**, which
  directs these onto the member getter; see the report's Friction section.
- **Custom DM hardware config (multi-core):** the legacy triple is
  `(RISCV_1, NOC::RISCV_1_default, DM_DEDICATED_NOC)`, and `NOC::RISCV_1_default == NOC_1`.
  That matches **neither** default — reader is `(RISCV_1, NOC_0)`, writer is `(RISCV_0, NOC_1)`
  — so it is replicated field-by-field as
  `DataMovementGen1Config{.processor = RISCV_1, .noc = NOC_1, .noc_mode = DM_DEDICATED_NOC}`,
  **not** routed through a "close" helper.
- **Not applied, deliberately:** no `allow_instance_multi_binding`, no `alias_with`, no
  conditional DFB `#ifdef` gating (the conditional `SRC1` / `READER1` pair is a whole extra
  `KernelSpec` with its own generated bindings, so no kernel-side `#ifdef` is needed — the
  same source compiles twice, and both builds bind exactly one `dfb::src`), no borrowed-memory
  DFB, no varargs, no op-owned tensors.

---

## Deferred / Flagged

- **Brief vs. CB→DFB whitelist disagreement on `get_dataformat`** — planning-step finding.
  The brief instructs "move onto the DFB member, which is `constexpr` and so survives the NTTP
  uses"; the whitelist §A and recipe rule 7 say the opposite for a legacy-`constexpr` site.
  The whitelist is right (the *getter* is `constexpr`, the *object* can never be), and it is
  authoritative. Following the whitelist. → report.
- **Conditional `KernelRunArgs` for `READER1`** — the multi-core `ProgramRunArgs` must not
  carry a `KernelRunArgs` entry for a `READER1` that was never added to the spec.
- **`argmax_common.hpp` is shared with the retained legacy fork** — only positional-parameter
  renames there; no signature or behaviour change.
- **Anti-pattern sweep denominator caveat:** the `cb`-name sweep over the op directory will
  legitimately report hits from the two retained legacy files
  (`reader_argmax_interleaved.cpp` and the whole out-of-scope NC half). The sweep is therefore
  reported both op-wide *and* scoped to the ported file set, with the residue enumerated.
- Nothing found during planning that the audit missed structurally. No stop signal.
