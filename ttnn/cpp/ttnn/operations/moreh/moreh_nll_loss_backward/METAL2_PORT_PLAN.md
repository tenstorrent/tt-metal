# Port Plan — `moreh_nll_loss_backward`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss_backward`, ported from the
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope:** one DeviceOperation, one factory, three rank-dispatched code paths (2d / 3d / 4d), five
kernels. Target concept: the base `ProgramSpecFactoryConcept` (inherited from the audit).

**Config axes carried through every section:** rank ∈ {2d, 3d, 4d} × `WEIGHT` (optional
`weight_tensor`) × `DIVISOR` (optional `divisor_tensor`) × `fp32_dest_acc_en` (formats only).

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — `static ProgramDescriptor create_descriptor(...)`
  on `Factory` (`device/moreh_nll_loss_backward_device_operation.hpp:38-43`, defined at
  `device/moreh_nll_loss_backward_program_factory.cpp:691`).
- **Variants:** single — `program_factory_t = std::variant<Factory>`
  (`...device_operation.hpp:45`). `create_descriptor` branches on
  `input_grad.logical_shape().rank()` into three free functions in the same file:
  `moreh_nll_loss_backward_impl_2d` (`:46`), `_impl_3d` (`:259`), `_impl_4d` (`:474`). These are
  **configs of one factory**, not three factories.
- **Custom `compute_program_hash`:** none — default reflection-based hash. No backdoor
  `attribute_values` / `to_hash` either. Nothing for the port to preserve or touch.

*(Metal 2.0 target concept chosen during the audit — see the brief's TTNN factory analysis. Carried
forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels

Identical structure across the three rank paths; the only differences are the reader **source file**
and the reader **RTA count** (2d has 10, 3d/4d have 11 — 3d/4d insert `num_inner_tile`). Rows below
give the 2d values with the 3d/4d deltas called out.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_nll_loss_backward_2d.cpp` (3d/4d: `_3d.cpp` / `_4d.cpp`) | `all_cores` | 4 × `TensorAccessorArgs` blocks appended in order: target, weight, divisor, output_grad (`:111-114`; 3d `:325-328`; 4d `:542-545`) — no scalar CTAs | none | 10 per core: `target_buf`, `output_grad_buf`, `weight_buf`, `divisor_buf`, `ignore_index`, `units_per_core`, `tile_offset`, `channel_size`, `weight_num_tile`, `element_size` (`:218-231`). **3d/4d: 11** — `num_inner_tile` inserted after `channel_size` (`:432-446`, `:649-663`) | none | `WEIGHT=1` if weight present; `DIVISOR=1` if divisor present; `FP32_DEST_ACC_EN=1` if `fp32_dest_acc_en` (`:123-135`) | field absent → resolves **O2** | `ReaderConfigDescriptor{}` (`:153`) |
| writer | `device/kernels/writer_moreh_nll_loss_backward.cpp` | `all_cores` | 1 × `TensorAccessorArgs` block: input_grad (`:117`) | none | 3 per core: `input_grad_buf`, `units_per_core`, `tile_offset` (`:233`) | none | none (`writer_defines` built but never populated, `:120`) | field absent → resolves **O2** | `WriterConfigDescriptor{}` (`:161`) |
| compute_1 | `device/kernels/moreh_nll_loss_backward_kernel.cpp` | `core_group_1` | `{units_per_core_group_1, static_cast<uint32_t>(divisor_has_value)}` (`:169`) | none | 2 per core: `{units_per_core, tile_offset}` (`:236`, `:239`) | none | same three as reader (`compute_defines`, `:125-135`) | field absent → resolves **O3** | `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en, .dst_full_sync_en, .unpack_to_dest_mode, .math_approx_mode}` (`:171-177`) |
| compute_2 | *(same source)* | `core_group_2`, only when `!core_group_2.ranges().empty()` (`:180-194`) | `{units_per_core_group_2, static_cast<uint32_t>(divisor_has_value)}` (`:185`) | none | same shape, `:241` | none | same | **O3** | same |

`opt_level` verified by `grep -n opt_level` over the factory: **zero hits**, so every
`KernelDescriptor::opt_level` is `std::nullopt` and resolves per kernel type — `O2` for the two DM
descriptors, **`O3` for both `ComputeConfigDescriptor`s**.

`unpack_to_dest_mode` is `std::vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default)`
(`:163`, `:377`, `:594`) — i.e. **`Default` for every CB**, on all three rank paths.

3d/4d line references for the compute descriptors: `:383`/`:399` (3d), `:600`/`:616` (4d).

### CBs

All built through the local `push_cb` helper (`:23-42`), which **skips allocation entirely when
`num_tiles == 0`** (`:29-31`) — this is how `c_2` / `c_3` come out absent rather than zero-sized.
`total_size = num_tiles * tt::tile_size(data_format)`, `page_size = tt::tile_size(data_format)`,
`core_ranges = all_cores`, and `CBFormatDescriptor::tile` is **never set** on any CB (so every DFB's
`tile_format_metadata` stays `nullopt`).

Formats: `data_format = datatype_to_dataformat_converter(input_grad.dtype())`;
`fp32_dest_acc_en_data_format = fp32_dest_acc_en ? Float32 : data_format`.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | present when |
|---|---|---|---|---|---|---|
| `c_0` output_grad | 1 tile | `all_cores` | `data_format` | `tile_size(data_format)` | unset | always |
| `c_1` target | 1 tile | `all_cores` | `Int32` | `tile_size(Int32)` | unset | always |
| `c_2` weight | `weight_num_tile` tiles | `all_cores` | `data_format` | `tile_size(data_format)` | unset | **`WEIGHT`** |
| `c_3` divisor | 1 tile | `all_cores` | `data_format` | `tile_size(data_format)` | unset | **`DIVISOR`** |
| `c_24` tmp_weight | 1 tile | `all_cores` | `fp32_dest_acc_en_data_format` | matching | unset | always |
| `c_25` tmp1 | 1 tile | `all_cores` | `fp32_dest_acc_en_data_format` | matching | unset | always *(allocated unconditionally — see Flags)* |
| `c_26` tmp2 | 1 tile | `all_cores` | `fp32_dest_acc_en_data_format` | matching | unset | always *(same)* |
| `c_16` input_grad | 1 tile | `all_cores` | `data_format` | `tile_size(data_format)` | unset | always |
| `c_7` weight_scratch | 1 tile | `all_cores` | `data_format` | `tile_size(data_format)` | unset | **`WEIGHT`** |
| `c_8` *(intended output_grad scratch)* | 1 tile | `all_cores` | `data_format` | `tile_size(data_format)` | unset | **2d only** — `:107`; **DEAD** |

`weight_num_tile = tt::div_up(channel_size, TILE_WIDTH)`.

No `GlobalCircularBuffer` anywhere (no `.global_circular_buffer` field, no `remote_cb` idiom, no
`Buffer`-backed CB). Confirmed by the audit's Appendix A sweep and re-checked here.

### Semaphores

none — the op declares no semaphores of any kind (no `.semaphores` on any `ProgramDescriptor`, no
`Semaphore` / `semaphore` token in the directory).

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `:111` / `:325` / `:542` — `TensorAccessorArgs(*target.buffer())` → reader CTAs | `tensor_args.target_tensor` | reader RTA 0 (`target_buf`) |
| `:112` / `:326` / `:543` — `TensorAccessorArgs(weight ? … : nullptr)` → reader CTAs | `tensor_args.weight_tensor` (optional) | reader RTA 2 (`weight_buf`, `nullptr` when absent) |
| `:113` / `:327` / `:544` — `TensorAccessorArgs(divisor ? … : nullptr)` → reader CTAs | `tensor_args.divisor_tensor` (optional) | reader RTA 3 (`divisor_buf`, `nullptr` when absent) |
| `:114` / `:328` / `:545` — `TensorAccessorArgs(*output_grad.buffer())` → reader CTAs | `tensor_args.output_grad_tensor` | reader RTA 1 (`output_grad_buf`) |
| `:117` / `:331` / `:548` — `TensorAccessorArgs(*input_grad.buffer())` → writer CTAs | `tensor_return_value` (`input_grad`) | writer RTA 0 (`input_grad_buf`) |

Device-side construction sites (13 total, **all 2-argument** — no page-size third argument anywhere):
each reader builds `target` / `output_grad` unconditionally and `weight` / `divisor` inside their
`#if` guards (4 × 3 readers = 12); the writer builds `input_grad` (1).

Note the delivery mechanism: the pushed RTA values are `Buffer*` objects, not `->address()` results
(the factory comments the intent at `:197-198`, `:411-412`, `:628-629`). There is **no `->address()`
call anywhere in the op**, hence no host-computed `base + offset` fold to split out.

### Work split

- Driver: `split_work_to_cores(grid, units_to_divide)` (`:70-71`, `:286-287`, `:503-504`), where
  `grid = device->compute_with_storage_grid_size()`.
- `units_to_divide`:
  - 2d / 3d: `input_grad.physical_volume() / TILE_HEIGHT / TILE_WIDTH`
  - 4d: `input_grad.physical_volume() / H / W * Ht * Wt`
- Returns `(num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1,
  units_per_core_group_2)`; `all_cores == core_group_1 ∪ core_group_2`, the two groups disjoint.
- Per-core iteration: `CoreCoord core = {i / core_h, i % core_h}` for `i ∈ [0, num_cores)`, with
  `core_h = grid.y`; `tile_offset` accumulates `units_per_core` across the loop.

### Shared kernels

**none.** All five kernel sources live in this op's own `device/kernels/` and are bound only by this
op's single factory. `grep -rl <filename> ttnn/cpp/ttnn/operations/` returns only this directory plus
the family CMake glob (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:42`), which is a build file and
not a consumer. No op borrows these files, this op borrows none, and no `_metal2` fork exists beside
any of them — so no rung of the shared-kernel Caution applies and no fork question arises.

The writer and the compute kernel are each bound at three call sites, but all three are rank branches
of the *same* factory, so one port converts every binder at once. The practical consequence is
narrower than sharing: **any edit to `writer_moreh_nll_loss_backward.cpp` or
`moreh_nll_loss_backward_kernel.cpp` must satisfy all three rank paths.**

### Flags

- **Dead CB `c_8`** (`:106-107`, 2d only) — allocated with a stale comment claiming a scratch is
  needed for the output_grad read; no kernel references it. The 2d reader's output_grad read is the
  3-argument `read_tile` overload, which takes no scratch. Confirmed dead → dropped by this port.
- **`c_25` / `c_26` allocated unconditionally** (`:96-97`, `:312-313`, `:529-530`) while every *use*
  sits inside `#if defined(DIVISOR)` in the compute kernel. Not dead in general — dead only in the
  non-`DIVISOR` config. Handled as a conditional allocation, not a drop.
- **Dead args and one dead CTA** (names deliberately not invented for these):
  - reader `element_size` — host-computed at `:205` / `:419` / `:636`, read into an unused local
    (`_2d.cpp:22`, `_3d.cpp:23`, `_4d.cpp:23`).
  - compute RTA index 0 — never read by the kernel (its tile count comes from CTA 0).
  - compute RTA index 1 (`tile_offset`) — read into an unused local
    (`moreh_nll_loss_backward_kernel.cpp:14`).
  - compute CTA index 1 (`divisor_has_value`) — never read; the kernel branches on the `DIVISOR`
    define, which the factory also supplies (`:129`).
- **Nine dead `get_dataformat(cb_id)` locals** (`_2d.cpp:34,36,38`; `_3d.cpp:35,37,39`;
  `_4d.cpp:35,37,39`) — assigned once, never read. Two of the three per reader query CBs that do not
  exist in the non-`WEIGHT` / non-`DIVISOR` configs. **Ops team confirmed deletion** (invoker,
  answering audit Question 2).
- **`reduction_mean` is accepted, hashed, and unused** by all three impls (`:52`, `:265`, `:480`).
  **Invoker instruction: leave it completely alone** — it is public API with an external consumer
  (`tt-train/sources/ttml/ops/losses.cpp`); removal is a separate decision on a separate track.
- **Inconsistent assertion macro** for the same unreachable condition: `TT_FATAL` in 2d (`:243`) vs
  `TT_ASSERT` in 3d (`:458`) and 4d (`:675`). Reported, not fixed.
- **Unreferenced kernel files:** none — all five kernels in `device/kernels/` are bound.
- **Descriptor types outside the audit's scan:** none.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` — the **base** concept.
  `override_runtime_arguments` is absent from both the device-op and the factory, so this is not a
  `CustomProgramSpecFactoryConcept` port and there is no override to translate.
- **Custom `compute_program_hash`:** none — default reflection-based hash. Nothing to preserve.
- **Implementation notes:**
  - `Factory::create_descriptor` becomes `Factory::create_program_artifacts`, returning
    `ttnn::device_operation::ProgramArtifacts`. The rank dispatch stays host-side in that method,
    exactly as today; each of the three impls returns its own `ProgramArtifacts`
    ([Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)).
  - `op_owned_tensors` is left defaulted — the op allocates no device tensors beyond its io.
  - Device-op-class edit forced by the port: the `create_descriptor` declaration in
    `...device_operation.hpp:38-43` is replaced by `create_program_artifacts`. **No pybind line
    references `create_descriptor`** (`moreh_nll_loss_backward_nanobind.cpp:23-36` binds only the
    user-facing op), so there is nothing to delete there and this port carries **no user-visible API
    change**.
  - Anonymous-namespace constants introduced by the port sit inside
    `namespace ttnn::operations::moreh::moreh_nll_loss_backward { namespace { … } }`. `ttnn_op_moreh`
    is a **unity-build** target (`ttnn/cpp/ttnn/operations/moreh/CMakeLists.txt:7`) and sibling moreh
    factories declare their own same-named anon-namespace helpers (three define `push_cb`), but the
    enclosing op namespace differs per file, so the merged TU keeps them distinct. No name prefixing
    needed.

---

## Planned Spec Shape

Default is 1:1 with legacy. The shape below is identical across the three rank paths — each impl
builds its own `ProgramSpec` from it, differing only in the reader source, the reader RTA set, and
(2d only) the `c_8` drop.

### KernelSpecs — 4 (3 when `core_group_2` is empty)

| unique_id | source | notes |
|---|---|---|
| `reader` | rank-specific reader | one spec over both work units |
| `writer` | `writer_moreh_nll_loss_backward.cpp` | one spec over both work units |
| `compute_group_1` | `moreh_nll_loss_backward_kernel.cpp` | CTA `per_core_tile_cnt = units_per_core_group_1` |
| `compute_group_2` | *(same source)* | CTA `per_core_tile_cnt = units_per_core_group_2`; emitted only when `has_core_group_2` |

### DataflowBufferSpecs — 9 max, 6–9 depending on config

One per surviving legacy `CBDescriptor`. No legacy CB had multi-element `format_descriptors`, so
there are **no aliased DFBs** and no `advanced_options.alias_with`. No borrowed-memory DFBs
(`borrowed_from` unset everywhere). `tile_format_metadata` left `nullopt` on all — the legacy
`CBFormatDescriptor::tile` was never set.

| unique_id | legacy CB | entry_size | num_entries | data_format_metadata | declared when |
|---|---|---|---|---|---|
| `output_grad` | `c_0` | `tile_size(data_format)` | 1 | `data_format` | always |
| `target` | `c_1` | `tile_size(Int32)` | 1 | `Int32` | always |
| `weight` | `c_2` | `tile_size(data_format)` | `weight_num_tile` | `data_format` | `weight_has_value` |
| `divisor` | `c_3` | `tile_size(data_format)` | 1 | `data_format` | `divisor_has_value` |
| `weight_scratch` | `c_7` | `tile_size(data_format)` | 1 | `data_format` | `weight_has_value` |
| `input_grad` | `c_16` | `tile_size(data_format)` | 1 | `data_format` | always |
| `tmp_weight` | `c_24` | `tile_size(fp32_acc_df)` | 1 | `fp32_dest_acc_en_data_format` | always |
| `tmp1` | `c_25` | `tile_size(fp32_acc_df)` | 1 | `fp32_dest_acc_en_data_format` | **`divisor_has_value`** |
| `tmp2` | `c_26` | `tile_size(fp32_acc_df)` | 1 | `fp32_dest_acc_en_data_format` | **`divisor_has_value`** |
| — | `c_8` | — | — | — | **dropped — dead CB** |

**Endpoint dispositions — re-derived from the kernel-touch census, not transcribed.** Per node each
`WorkUnitSpec` places exactly one reader, one writer and one compute instance (the two compute specs
cover **disjoint** core groups), so every census below is a per-node count.

| DFB | touchers on a node | census | disposition | agrees with brief |
|---|---|---|---|---|
| `output_grad` | reader produces (`read_tile` → `reserve_back`/`push_back`); compute `wait_front`s and holds (never pops) | 1P + 1C | plain 1:1 — reader PRODUCER, compute CONSUMER | ✓ |
| `target` | reader only — produces, `wait_front`/`pop_front`s, and `get_read_ptr` peeks | 1 toucher | **self-loop** on reader | ✓ |
| `weight` | reader only — `read_line` produces, then `wait_front` + `get_read_ptr` (never pops) | 1 toucher | **self-loop** on reader | ✓ |
| `divisor` | reader produces (`read_tile`); compute `wait_front`/`pop_front`s | 1P + 1C | plain 1:1 | ✓ |
| `weight_scratch` | reader only, inside the donor's `read_line` — NoC-written, `get_write_ptr()`-read, **no FIFO ops at all** | 1 toucher, sync-free | **self-loop** on reader | ✓ |
| `input_grad` | compute produces; writer `wait_front`/`pop_front`s | 1P + 1C | plain 1:1 | ✓ |
| `tmp_weight` | reader produces (`reserve_back` + raw `get_write_ptr` write + `push_back`); compute consumes | 1P + 1C | plain 1:1 | ✓ |
| `tmp1` | compute only (`DIVISOR`) — produces and consumes | 1 toucher | **self-loop** on each compute spec | ✓ |
| `tmp2` | compute only (`DIVISOR`) — produces and consumes | 1 toucher | **self-loop** on each compute spec | ✓ |

**No multi-binding anywhere.** The reader's raw `get_write_ptr()` write into `tmp_weight` is bracketed
by its own `reserve_back` … `push_back` — the producer's own peek, not a hidden second writer. No CB
has ≥3 distinct touchers, and no two kernels are locked to the same FIFO role on one node. No DFB is
both self-looped and multi-bound. `advanced_options.allow_instance_multi_binding` is set nowhere.

Self-loops use a **single shared `accessor_name`** for the PRODUCER/CONSUMER pair, which the validator
explicitly permits (`tt_metal/impl/metal2_host_api/program_spec.cpp:298`) — one kernel-side
`dfb::name` handle drives both directions, so the kernel body is unchanged.

The three DM self-loops (`target`, `weight`, `weight_scratch` on the reader) are legal on Gen1 and
are Quasar-uplift's concern, not this port's.

### SemaphoreSpecs

none — the legacy op declares no `SemaphoreDescriptor`.

### TensorParameters — 5 max, 3–5 depending on config

One per distinct originating tensor. `weight` and `divisor` are declared **only when present**: the
validator rejects a `TensorParameter` with zero `TensorBinding`s, and an absent optional has no
tensor to bind. `relaxations` left default (strict) on all five — the audit's
`TensorParameter relaxation` cell reads `none`.

| unique_id | tensor | bound by | declared when |
|---|---|---|---|
| `target` | `tensor_args.target_tensor` | reader | always |
| `output_grad` | `tensor_args.output_grad_tensor` | reader | always |
| `weight` | `tensor_args.weight_tensor` | reader | `weight_has_value` |
| `divisor` | `tensor_args.divisor_tensor` | reader | `divisor_has_value` |
| `input_grad` | `tensor_return_value` | writer | always |

(`DFBSpecName` and `TensorParamName` are distinct strong types emitting distinct kernel-side
namespaces, so reusing a name across the two tables — e.g. `dfb::target` alongside `tensor::target` —
is unambiguous and mirrors the legacy `cb_target` / `addrg_target` pairing.)

### WorkUnitSpecs — 2 (1 when `core_group_2` is empty)

| name | kernels | target_nodes |
|---|---|---|
| `group_1` | `reader`, `writer`, `compute_group_1` | `core_group_1` |
| `group_2` | `reader`, `writer`, `compute_group_2` | `core_group_2` (omitted when empty) |

The reader's and writer's effective node sets are the union `core_group_1 ∪ core_group_2 == all_cores`,
matching their legacy `core_ranges`. Each compute spec's node set is its own group. WU names are
distinct — the validator's overlap check skips self by comparing `work_unit.name`
(`program_spec.cpp:1712`), so duplicate names would silently disable it.

### Op-owned tensors

none. The `descriptor` concept carries no `WorkloadBuffer`s and the factory allocates no device
tensors beyond the op's io; `ProgramArtifacts::op_owned_tensors` is left defaulted.

---

## Preserved Multiplicity

```
Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source
  device/kernels/moreh_nll_loss_backward_kernel.cpp
  → KernelSpecs [compute_group_1, compute_group_2] of same source
  → in WorkUnitSpecs [group_1, group_2]
  → sharing upstream/downstream DFBs (endpoint role each KernelSpec binds):
      output_grad (CONSUMER), tmp_weight (CONSUMER), input_grad (PRODUCER),
      divisor (CONSUMER, DIVISOR only),
      tmp1 (PRODUCER + CONSUMER, DIVISOR only), tmp2 (PRODUCER + CONSUMER, DIVISOR only)
```

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_1` (`core_group_1`), `compute_desc_2` (`core_group_2`) | `compute_group_1`, `compute_group_2` | `group_1`, `group_2` | `output_grad` C · `tmp_weight` C · `input_grad` P · `divisor` C · `tmp1` P+C · `tmp2` P+C |

The two node sets are **disjoint**, so each node sees exactly one compute instance and every shared
DFB is an ordinary single-role binding per node. This is the disjoint-node work-split, **not** the
same-grid two-toucher case: no `allow_instance_multi_binding`, no 1P+1C reassignment.

The per-group CTA (`per_core_tile_cnt`) stays a **CTA on two specs**. Collapsing the pair into one
spec by demoting it to an RTA is the *Demoting per-group CTA to RTA* anti-pattern — it would cost
compile-time loop unrolling on `per_core_tile_cnt` for no gain.

---

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`:221`, `:435`, `:652`) | `target_buf` (`Buffer*`) | `TensorBinding{TENSOR_TARGET, "target"}` |
| reader RTA slot 1 (`:222`, `:436`, `:653`) | `output_grad_buf` | `TensorBinding{TENSOR_OUTPUT_GRAD, "output_grad"}` |
| reader RTA slot 2 (`:223`, `:437`, `:654`) | `weight_buf` or `nullptr` | `TensorBinding{TENSOR_WEIGHT, "weight"}`, bound only when `weight_has_value` |
| reader RTA slot 3 (`:224`, `:438`, `:655`) | `divisor_buf` or `nullptr` | `TensorBinding{TENSOR_DIVISOR, "divisor"}`, bound only when `divisor_has_value` |
| writer RTA slot 0 (`:233`, `:448`, `:665`) | `input_grad_buf` | `TensorBinding{TENSOR_INPUT_GRAD, "input_grad"}` |

All five are **Case 1** — every kernel-side use goes through a `TensorAccessor`. No kernel does
address arithmetic on a tensor base, so no binding needs the `get_bank_base_address` bridge. (The
readers' `CoreLocalMem<...>(dfb.get_read_ptr())` pointers are DFB/L1 pointers obtained from a DFB
method, not tensor bases — they stay exactly as they are.) The host-side `target_buf` /
`weight_buf` / `divisor_buf` / `output_grad_buf` / `input_grad_buf` locals and their explanatory
comment (`:196-202`, `:410-416`, `:627-633`) are dropped with the RTAs.

### `TensorAccessorArgs` plumbing → binding mechanism

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `:111-114` / `:325-328` / `:542-545` | four `TensorAccessorArgs(...).append_to(reader_compile_time_args)` calls, the 2nd and 3rd passing `nullptr` for an absent optional | framework-built accessor args from the reader's `TensorBinding`s |
| `:117` / `:331` / `:548` | `TensorAccessorArgs(*input_grad.buffer()).append_to(writer_compile_time_args)` | framework-built from the writer's `TensorBinding` |
| readers `:40-43` | `TensorAccessorArgs<0>()` and the `next_compile_time_args_offset()` chain (target → weight → divisor → output_grad) | deleted — `TensorAccessor(tensor::name)` |
| writer `:18` | `constexpr auto input_grad_args = TensorAccessorArgs<0>();` | deleted |

The null placeholder blocks existed solely to keep that offset chain aligned across configs. Metal 2.0
has no chain to preserve, so **no placeholder binding is carried forward** — `weight` and `divisor`
become genuinely conditional bindings.

### Page-size 3rd-argument CTAs/RTAs

none — all 13 `TensorAccessor(` construction sites are 2-argument. Nothing to drop.

### Magic CB indices → `DFBBinding`

No CB index was ever carried in a CTA on this op (the readers' only CTAs were the four accessor
blocks; compute's two CTAs are `per_core_tile_cnt` and the dead `divisor_has_value`). The magic
indices live as kernel-side `constexpr uint32_t cb_x = tt::CBIndex::c_N` literals instead, which the
port replaces with `dfb::` handles:

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| readers `:24-31` (2d), `:25-32` (3d), `:25-32` (4d) | `cb_output_grad`/`cb_target`/`cb_weight`/`cb_divisor`/`cb_tmp_weight`/`cb_weight_scratch` = `tt::CBIndex::c_*` | `dfb::output_grad`, `dfb::target`, `dfb::weight`, `dfb::divisor`, `dfb::tmp_weight`, `dfb::weight_scratch` |
| writer `:16` | `cb_input_grad = tt::CBIndex::c_16` | `dfb::input_grad` |
| compute `:16-27` | six `cb_* = tt::CBIndex::c_*` constants | `dfb::divisor`, `dfb::output_grad`, `dfb::tmp_weight`, `dfb::tmp1`, `dfb::tmp2`, `dfb::input_grad` |

### Semaphore-ID RTAs

none — the op has no semaphores.

### Positional CTAs → named CTAs

| legacy location (file:line) | legacy positional CTA list | named CTAs assigned |
|---|---|---|
| compute `:169` / `:185` / `:383` / `:399` / `:600` / `:616` | `{units_per_core_group_N, static_cast<uint32_t>(divisor_has_value)}` | `{{"per_core_tile_cnt", units_per_core_group_N}}` — **slot 1 dropped, not named** |
| reader / writer | no scalar CTAs (accessor blocks only) | none |

Slot 1 (`divisor_has_value`) is **never read** by the compute kernel, which branches on the `DIVISOR`
define the factory also supplies. It has no meaning to name, so it is dropped rather than carried
across. Reported.

### Dead RTAs dropped (not renamed)

| legacy location (file:line) | legacy form | disposition |
|---|---|---|
| reader RTA slot 9 / 10 (`:230`, `:445`, `:662`) | `element_size = weight ? weight->element_size() : 0` (`:205`, `:419`, `:636`) | **dropped** — read into an unused local in all three readers (`_2d.cpp:22`, `_3d.cpp:23`, `_4d.cpp:23`). Host computation and kernel read line both deleted. |
| compute RTA slot 0 (`:236`, `:451`, `:668`) | `units_per_core` | **dropped** — never read by the kernel |
| compute RTA slot 1 (`:236`, `:451`, `:668`) | `tile_offset` | **dropped** — read into an unused local (`moreh_nll_loss_backward_kernel.cpp:14`); that read line is deleted |

With both compute RTAs gone the compute kernels have **no runtime args at all**, so they get no
`runtime_arg_schema` and no `KernelRunArgs` entry. That also removes the per-group compute RTA
assignment block, which carried the only `TT_FATAL` (2d) / `TT_ASSERT` (3d, 4d) in the loop — a
**subject-deleted** guard loss, recorded in the port report. The `TT_THROW` guarding the same
"core in neither group" condition earlier in the same loop (`:215`, `:429`, `:646`) **survives
unchanged**, so the condition is still checked.

### Surviving RTAs → named

| kernel | legacy positional RTAs (after address drops) | named RTAs |
|---|---|---|
| reader 2d | slots 4–8 | `ignore_index`, `num_tiles_per_core`, `start_id`, `C`, `weight_num_tile` |
| reader 3d / 4d | slots 4–9 | `ignore_index`, `num_tiles_per_core`, `start_id`, `C`, `num_inner_tile`, `weight_num_tile` |
| writer | slots 1–2 | `num_tiles_per_core`, `start_id` |
| compute | — | none |

Names are taken from the kernel-side locals they already land in, per the brief. None of these is a
vararg: each is a **distinct field read exactly once** at the top of `kernel_main`. The legacy
`get_arg_val<uint32_t>(i++)` running counter is a fixed run over a fixed set, not an indexed
collection — `advanced_options.num_runtime_varargs` stays 0 on every kernel.

---

## Applied Patterns

- **[Sync-free and single-ended CBs → self-loop DFB]** — `weight_scratch` (`c_7`) on the reader: the
  donor's `read_line` NoC-writes it and reads it back via `get_write_ptr()` with **no FIFO ops at
  all**. One toucher, sync-free → reader bound PRODUCER + CONSUMER under one accessor name.
- **[Self-loop DFB binding]** — four more one-toucher DFBs: `target` and `weight` on the reader
  (genuine FIFO traffic, but only one kernel touches them), and `tmp1` / `tmp2` on each compute spec
  (compute produces and consumes both within its own loop).
- **[Conditional / optional DFB bindings]** — `weight` + `weight_scratch` gated on `WEIGHT`;
  `divisor` + `tmp1` + `tmp2` gated on `DIVISOR`. The host conditionally declares the
  `DataflowBufferSpec`s and the matching `DFBBinding`s, and the factory **already emits** the
  `WEIGHT` / `DIVISOR` defines the kernels gate on — so this reuses the op's own existing
  preprocessor gates rather than inventing new ones. Also applies to the conditional
  `TensorParameter`s / `TensorBinding`s for `weight` and `divisor`.
- **[Two-toucher DFB → assign 1P+1C]** — *considered and rejected.* Re-ran the endpoint census
  independently; no DFB has two co-resident touchers needing role assignment. Recorded because the
  reader's raw `get_write_ptr()` write into `tmp_weight` is the shape that invites the misread.
- **[Multi-variant factories]** — rank dispatch (2d / 3d / 4d) stays host-side inside
  `create_program_artifacts`, each branch returning its own `ProgramArtifacts`.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers]** — `dfb::name` flows straight into
  the donor helpers (`read_tile`, `read_line`, `copy_tile_init_with_dt`, `pack_tile_with_dt`,
  `mul_bcast_scalar_init_with_dt`), which take `DataflowBuffer` **by value** via the implicit
  `DataflowBuffer(DFBBindingToken)` constructor, and into the raw-`uint32_t` LLKs (`init_sfpu`,
  `copy_tile`, `mul_tiles_bcast_scalar`) via the token's `constexpr operator uint32_t()`. No `.id`
  extraction, no temporary wrappers.
- **[Unity-build hygiene for anonymous-namespace symbols]** — checked, no action needed: the
  constants sit inside the op's own namespace, so the merged unity TU keeps them distinct from the
  sibling moreh factories' same-named helpers.
- **[Anti-pattern: Demoting per-group CTA to RTA]** — avoided; see *Preserved Multiplicity*.

### Kernel-side whitelist rule 7 (DFB metadata via the object)

- **Applied:** the writer's `get_tile_size(cb_input_grad)` (`writer_...cpp:26`) becomes
  `dfb_input_grad_obj.get_tile_size()`.
- **Not applied — deleted instead:** the nine dead `get_dataformat(cb_id)` locals. Rewriting them as
  `DataflowBuffer(dfb::weight).get_dataformat()` would name DFBs that are **not bound** in the
  non-`WEIGHT` / non-`DIVISOR` configs, and all three sit outside the guards that wrap every real use
  of those CBs. They are provably dead, so deletion is behaviour-preserving. **Ops team confirmed the
  deletion** (invoker, answering audit Question 2).

### Hardware configuration

- **DM kernels.** Legacy resolved triples: reader `ReaderConfigDescriptor{}` → the reader default
  (`RISCV_1` / `NOC_0` / `DM_DEDICATED_NOC`); writer `WriterConfigDescriptor{}` → the writer default
  (`RISCV_0` / `NOC_1` / `DM_DEDICATED_NOC`). Both match a default exactly, so both take the
  arch-agnostic TTNN helpers: `create_reader_datamovement_config(device->arch())` and
  `create_writer_datamovement_config(device->arch())`. No custom triple, no `DM_DYNAMIC_NOC`, no
  Gen2 branch authored by hand.
- **Compute kernels — Style A.** The op resolves a TTNN `ComputeKernelConfig` and destructures it via
  `get_compute_kernel_config_args` (`:73-74`), so the port translates the same config with
  `to_compute_hardware_config(device->arch(), compute_kernel_config)`. Verified that
  `get_compute_kernel_config_args` is a pure field passthrough
  (`compute_kernel_config.cpp:99-107`), so the two read the identical values:

  | legacy `ComputeConfigDescriptor` field | value | Metal 2.0 `ComputeGen1Config` field | transform |
  |---|---|---|---|
  | `math_fidelity` | from config | `fpu_math_fidelity` | 1:1 |
  | `math_approx_mode` | from config | `sfpu_precision_mode` | `true`→`Approximate`, `false`→`Precise` |
  | `fp32_dest_acc_en` | from config | `enable_32_bit_dest` | 1:1 |
  | `dst_full_sync_en` | from config | `double_buffer_dest` | **inverted** |
  | `unpack_to_dest_mode` | all `Default` | `unpack_modes` | see below |
  | *(not set)* | — | `bfp_pack_precision_mode` | left default (`Approximate`) — legacy `bfp8_pack_precise` was never set |

- **`unpack_modes` — the newly-required entries.** Legacy passed `Default` for every CB, which maps
  to `UnpackMode::UnpackToSrc` and is normally expressed by *omitting* the entry. But the Metal 2.0
  validator (`program_spec.cpp:1049-1076`) **requires** an explicit entry whenever a compute kernel
  **consumes** a **`Float32`** DFB with `enable_32_bit_dest = true`. With `fp32_dest_acc_en` on,
  `fp32_dest_acc_en_data_format` is `Float32`, so the compute kernel's consumed DFBs
  `tmp_weight`, `tmp1` and `tmp2` all trip it (and `output_grad` / `divisor` do too whenever
  `input_grad.dtype()` is float32). The port emits `UnpackMode::UnpackToSrc` — derived from the
  legacy `Default`, not guessed — for exactly those DFBs, keyed by `DFBSpecName`. This path is
  covered by `test_moreh_nll_loss_backward_compute_kernel_options[...fp32_dest_acc_en=True...]`.

### Compiler options

| KernelSpec | legacy resolved `opt_level` | Metal 2.0 `compiler_options.opt_level` |
|---|---|---|
| `reader` | `O2` (DM default; field absent) | left at the `CompilerOptions` default `O2` — no action |
| `writer` | `O2` (DM default; field absent) | left at `O2` — no action |
| `compute_group_1` | **`O3`** (`ComputeConfigDescriptor` default; field absent) | **explicit `KernelBuildOptLevel::O3`** |
| `compute_group_2` | **`O3`** (same) | **explicit `KernelBuildOptLevel::O3`** |

`grep -n opt_level` over the legacy factory returns zero hits, so no kernel set an explicit level.
Metal 2.0's single `CompilerOptions` defaults to `O2` for both kernel kinds, so **both** compute
specs need `O3` set by hand or the port silently drops a level on the compile and the link.

`compiler_options.defines` carries the legacy `KernelDescriptor::defines` unchanged: `WEIGHT` /
`DIVISOR` / `FP32_DEST_ACC_EN` on the reader and compute specs, nothing on the writer.

---

## Deferred / Flagged

- **New findings during planning:** none that change the port's shape. The audit's census, endpoint
  dispositions and dead-arg list all reproduced independently and agreed.
- **Surfaced during planning, routed to the port report (not fixed here):**
  - The compute RTA vector is entirely dead, so the port removes the assignment block that carried
    the 2d `TT_FATAL` / 3d–4d `TT_ASSERT`. Subject-deleted guard loss; the `TT_THROW` on the same
    condition survives.
  - The 2d reader's `Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT` tiles a *width* dimension with
    `TILE_HEIGHT` while the host computes the same quantity with `TILE_WIDTH`
    (`program_factory.cpp:83`). Numerically identical today (both 32). Left exactly as-is.
  - Dead local `n` in the 2d reader (`reader_..._2d.cpp:90`). Left as-is — it is not port-forced.
  - `create_output_tensors`' `create_device_tensor` call is unreachable (`compute_output_specs`
    fatals first). Device-op-class code, off-limits.
  - `reduction_mean` unused by every impl — **invoker instruction: leave completely alone.**
- **Stop signals encountered:** none. No GlobalCircularBuffer, no `get_cb_tiles_acked_ptr` /
  `get_cb_tiles_received_ptr`, no Case 2 binding, no host-computed base+offset fold, no out-of-op
  kernel edit required, no `sem::` / `tensor::` handle demanded by an out-of-op call site.
