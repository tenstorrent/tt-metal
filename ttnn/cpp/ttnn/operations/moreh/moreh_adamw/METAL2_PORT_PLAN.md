# Port Plan — `moreh_adamw`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_adamw`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape

- Concept: **`ProgramDescriptorFactoryConcept`**, in its *direct* form — `create_descriptor` is a static
  member of the device-operation struct itself (`device/moreh_adamw_device_operation.hpp:60-63`,
  defined `device/multi_core_program_factory.cpp:58`). There is **no `program_factory_t`** and no separate
  factory class; the framework reaches it through `MeshDeviceOperationAdapter::DirectDescriptorFactory`
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170-187`).
- Variants: single.
- Custom `compute_program_hash`: **none**. Backdoor hash **present** — hand-written `attribute_names` /
  `attribute_values` @ `device/moreh_adamw_device_operation.hpp:35-40`, deliberately excluding `lr` and
  `step` (rationale in the comment @ `:31-34`). **Left intact.**
- `override_runtime_arguments` present @ `device/multi_core_program_factory.cpp:353-428`
  (declared `device/moreh_adamw_device_operation.hpp:76-81`), returning `void` and mutating the cached
  `Program` in place → the port **translates** it (see [TTNN ProgramFactory](#ttnn-programfactory)).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory
analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels

Four `KernelDescriptor`s: reader, writer, and the compute kernel instantiated **twice** over the two
**disjoint** work-split core groups (`compute_desc_1` @ `:247-253`, `compute_desc_2` @ `:255-264`, the
latter only when `core_group_2` is non-empty). Disjoint node sets ⇒ every node sees exactly one compute
instance; this is the per-group-CTA shape, *not* a dual-instance work-split.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_adamw.cpp` | `all_cores` | 4× (5× if `amsgrad`) `TensorAccessorArgs` blocks, appended @ `:193-199` — **no other CTA** | none | 16/core @ `:305-322`: `param_addr`, `grad_addr`, `exp_avg_addr`, `exp_avg_sq_addr`, `max_exp_avg_sq_addr`, `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `beta1_exponent`, `beta2_exponent`, `step`, `amsgrad`, `num_tiles_per_core`, `start_id` | none | `AMSGRAD=1` iff `amsgrad`; `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` (`:211-218`) | absent → **O2** (DM default) | `ReaderConfigDescriptor{}` @ `:226` |
| writer | `device/kernels/writer_moreh_adamw.cpp` | `all_cores` | 3× (4× if `amsgrad`) `TensorAccessorArgs` blocks @ `:202-207` — **no other CTA** | none | 6/core @ `:324-331`: `param_addr`, `exp_avg_addr`, `exp_avg_sq_addr`, `max_exp_avg_sq_addr`, `num_tiles_per_core`, `start_id` | none | same as reader | absent → **O2** (DM default) | `WriterConfigDescriptor{}` @ `:235` |
| compute_g1 | `device/kernels/moreh_adamw.cpp` | `core_group_1` | `{num_units_per_core_group_1}` @ `:251` | none | 1/core: `step` @ `:335` | none | `AMSGRAD` / `FP32_DEST_ACC_EN` as above | absent → **O3** (compute default) | `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en, .math_approx_mode}` @ `:241-245` |
| compute_g2 | `device/kernels/moreh_adamw.cpp` | `core_group_2` (only if non-empty) | `{num_units_per_core_group_2}` @ `:261` | none | 1/core: `step` @ `:337` | none | same | absent → **O3** | same `compute_config` object |

`grep -n opt_level` over the factory returns nothing — the field is absent on every descriptor, so the
resolved levels above are the lowering defaults (O2 for the DM pair, **O3** for both compute descriptors).

**Compute config provenance (matters for the port — see [Planned Spec Shape](#planned-spec-shape)).** The op
resolves a TTNN `DeviceComputeKernelConfig` via `get_compute_kernel_config_args(arch, compute_kernel_config)`
@ `:98-99`, destructuring **five** fields — then hand-builds a Metal `ComputeConfigDescriptor` that sets only
**three** of them. `packer_l1_acc` and `dst_full_sync_en` are destructured and **dropped**, so both keep the
`ComputeConfigDescriptor` defaults (`dst_full_sync_en = false`, i.e. double-buffered Dest) regardless of what
the user requested. `unpack_to_dest_mode` and `bfp8_pack_precise` are never set → legacy defaults.

### CBs

19 `CBDescriptor`s, every one over `all_cores`, every one single-element `format_descriptors` (no aliasing),
none with `.tile` set, none a `GlobalCircularBuffer`, none with `address_offset`.
`data_format = datatype_to_dataformat_converter(param_in.dtype())` (BFLOAT16 or BFLOAT8_B — the only dtypes
`validate_inputs` admits); `intermed_cb_format = fp32_dest_acc_en ? Float32 : data_format` (`:104-107`).

| index | total_size | core_ranges | data_format | page_size | tile (if set) | factory site |
|---|---|---|---|---|---|---|
| `c_0` param_in | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:111-116` |
| `c_1` grad | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:117-122` |
| `c_2` exp_avg_in | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:123-128` |
| `c_3` exp_avg_sq_in | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:129-134` |
| `c_4` max_exp_avg_sq_in | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:135-140` |
| `c_5` scalar_args | `5 * intermed_tile_size` | `all_cores` | `intermed_cb_format` | `intermed_tile_size` | — | `:141-146` |
| `c_6` one | `1 * intermed_tile_size` | `all_cores` | `intermed_cb_format` | `intermed_tile_size` | — | `:147-152` |
| `c_24`…`c_31` (8 CBs) | `1 * intermed_tile_size` each | `all_cores` | `intermed_cb_format` | `intermed_tile_size` | — | loop @ `:155-162` |
| `c_16` param_out | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:164-169` |
| `c_17` exp_avg_out | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:170-175` |
| `c_18` exp_avg_sq_out | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:176-181` |
| `c_19` max_exp_avg_sq_out | `1 * data_tile_size` | `all_cores` | `data_format` | `data_tile_size` | — | `:182-187` |

The `c_24..c_31` loop covers `tmp_param` (`c_24`), `tmp_exp_avg` (`c_25`), `tmp_exp_avg_sq` (`c_26`),
`tmp_max_exp_avg_sq` (`c_27`), `beta1_exponent` (`c_28`), `beta2_exponent` (`c_29`), `tmp1` (`c_30`),
`tmp2` (`c_31`) — all identically shaped.

**Endpoint census (re-derived from the kernels, not transcribed from the brief).** Agrees with the brief and
the audit in every row. All three kernels are `#ifdef AMSGRAD`-gated on `c_4` / `c_19` / `c_27` and nowhere
else, so `amsgrad` is the only census axis (`fp32_dest_acc_en` moves the intermediates' *format*, not any
endpoint). No kernel calls `get_write_ptr` / `get_read_ptr` / `fifo_*_ptr` / `evil_set_*` on any buffer, so
every toucher is a FIFO toucher and the census is exhaustive. Max touchers per node per CB is **2**.

| CB | touchers | disposition (`amsgrad == true`) | disposition (`amsgrad == false`) |
|---|---|---|---|
| `c_0` `c_1` `c_2` `c_3` | reader (P), compute (C) | 1:1 | 1:1 |
| `c_5` `c_6` `c_28` `c_29` | reader (P, via `fill_cb_with_value`), compute (C) | 1:1 | 1:1 |
| `c_16` `c_17` `c_18` | compute (P), writer (C) | 1:1 | 1:1 |
| `c_4` | reader (P), compute (C) | 1:1 | **0 touchers — config-dead** |
| `c_19` | compute (P), writer (C) | 1:1 | **0 touchers — config-dead** |
| `c_24` `c_25` `c_26` `c_30` `c_31` | compute only | **self-loop** (1 toucher → P+C) | **self-loop** |
| `c_27` | compute only | **self-loop** | **0 touchers — config-dead** |

No CB reaches 3 touchers and no two kernels are locked to the same FIFO role ⇒ **no multi-binding anywhere**;
`allow_instance_multi_binding` is never set. No CB has a single toucher *and* two touchers across configs, so
no self-loop/multi-binding stacking arises.

### Semaphores

none — `grep -i semaphore` over the op directory returns nothing.

### Tensor accessors

Nine, all Case 1 (constructed and used only through `TensorAccessor`; no raw base-address arithmetic
anywhere, so no `get_bank_base_address` bridge is needed). All nine are **2-argument** — no page-size third
argument to drop.

| host site (file:line) | originating Tensor | RTA slot (host) | kernel accessor |
|---|---|---|---|
| `multi_core_program_factory.cpp:307` (miss), `:375` (hit) | `tensor_args.param_in` | reader 0 | `reader_moreh_adamw.cpp:51` |
| `:308` / `:376` | `tensor_args.grad` | reader 1 | `reader_moreh_adamw.cpp:52` |
| `:309` / `:377` | `tensor_args.exp_avg_in` | reader 2 | `reader_moreh_adamw.cpp:53` |
| `:310` / `:378` | `tensor_args.exp_avg_sq_in` | reader 3 | `reader_moreh_adamw.cpp:54` |
| `:311` / `:379` | `tensor_args.max_exp_avg_sq_in` *(amsgrad only)* | reader 4 | `reader_moreh_adamw.cpp:59` |
| `:326` / `:381` | `tensor_return_value.at(0)` (`param_out`) | writer 0 | `writer_moreh_adamw.cpp:28` |
| `:327` / `:382` | `tensor_return_value.at(1)` (`exp_avg_out`) | writer 1 | `writer_moreh_adamw.cpp:29` |
| `:328` / `:383` | `tensor_return_value.at(2)` (`exp_avg_sq_out`) | writer 2 | `writer_moreh_adamw.cpp:30` |
| `:329` / `:384` | `tensor_return_value.at(3)` (`max_exp_avg_sq_out`) *(amsgrad only)* | writer 3 | `writer_moreh_adamw.cpp:35` |

The miss path delivers each as an annotated `Buffer*` through `emplace_runtime_args` (auto-registered as a
`BufferBinding`); the hit path re-writes `->address()` into the same slot from `override_runtime_arguments`.

### Work split

- Driver: `split_work_to_cores(grid, num_units)` inside `compute_adamw_work_split` (`:41-54`), with
  `grid = param_in.device()->compute_with_storage_grid_size()` and
  `num_units = param_in.physical_volume() / tt::constants::TILE_HW`.
- num_cores: `num_cores`; core enumeration is `CoreCoord{i / num_cores_y, i % num_cores_y}` for
  `i ∈ [0, num_cores)` (`:294`), i.e. **column-major** over `grid.y`.
- core_group_1: `core_group_1`, count_per_core: `num_units_per_core_group_1`
- core_group_2: `core_group_2` (possibly empty), count_per_core: `num_units_per_core_group_2`
- `tile_offset` accumulates `num_tiles_per_core` across the loop and feeds `start_id` on both DM kernels.

### Shared kernels

**none.** All three kernel sources live in this op's own `device/kernels/` and no other factory binds any of
them. Census re-run per the Caution entry: `grep -rl reader_moreh_adamw.cpp ttnn/cpp/ttnn/operations/` and
the writer equivalent return only this op's factory. `moreh_adamw.cpp` additionally hits
`ttnn/cpp/ttnn/operations/moreh/sources.cmake` — that entry is the **host** wrapper
`moreh_adamw/moreh_adamw.cpp`, which shares the filename with the compute kernel; not a second binder.
No `_metal2` fork exists beside any of them and none is needed.

### Flags

1. **No direct-`create_program_artifacts` path in the framework — the port must add a `program_factory_t`.**
   The adapter's `DirectDescriptorFactory` shim covers only `create_descriptor`
   (`mesh_device_operation_adapter.hpp:170-187`). `DeviceOperationConcept`
   (`ttnn/api/ttnn/operation_concepts.hpp:206-208`) is satisfied by
   `HasDirectDescriptor || (HasProgramFactoryType && AllFactoriesValid)`, and `HasDirectDescriptor`
   (`:139`) keys on `create_descriptor`. Removing `create_descriptor` without introducing a
   `program_factory_t` therefore fails the concept. The port adds a nested factory struct plus
   `using program_factory_t = std::variant<...>` to the device-op header. Confirmed with the invoker before
   proceeding; recorded as a forced device-op-class edit in the port report.
2. **Two unused includes** in the port's blast radius, left untouched:
   `<tt-metalium/experimental/program_descriptor_patching.hpp>` @ `moreh_adamw_device_operation.hpp:15` and
   `"ttnn/operations/moreh/moreh_helper_functions.hpp"` @ `multi_core_program_factory.cpp:13`. The first
   becomes *more* clearly dead once the descriptor path goes; still not the port's to remove (audit
   *Misc anomalies*). Routed to the report.
3. **`amsgrad == true` with `max_exp_avg_sq_in` absent is already broken in legacy** and becomes a hard error
   under Metal 2.0. Legacy appends the 5th `TensorAccessorArgs` block only `if
   (max_exp_avg_sq_in.has_value())` (`:197-199`) while the kernel's `#ifdef AMSGRAD` block unconditionally
   reads a 5th accessor (`reader_moreh_adamw.cpp:58`) — so that combination already mismatches CTA supply
   against kernel expectation. Nothing enforces the coupling (`validate_inputs` only dtype-checks when
   present); the only caller, the pytest, always passes it iff `amsgrad`. The port gates the tensor binding
   on `amsgrad` (the same condition that emits the `AMSGRAD` define), so the combination raises rather than
   reading garbage. Routed to the report; no validation added.
4. No unreferenced kernel files in the op directory. No descriptor type outside the audit's Appendix A scan.

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` — selected because the ported-from
  factory has an `override_runtime_arguments`. Not re-derived; no disagreement with the audit's choice.
- **Custom `compute_program_hash`**: none. Backdoor hash present @
  `device/moreh_adamw_device_operation.hpp:35-40` — **leave intact**.
- **Implementation notes**:
  - `MorehAdamWDeviceOperation::MultiCoreProgramFactory` is introduced as a nested struct holding the two
    static methods (`create_program_artifacts`, `override_runtime_arguments`), with
    `using program_factory_t = std::variant<MultiCoreProgramFactory>;` on the device-op — forced by Flag 1.
    Both are defined in `device/multi_core_program_factory.cpp`, so the file layout is unchanged.
  - `override_runtime_arguments` changes shape as well as body: it loses the `Program&` parameter and returns
    `ProgramRunArgs` instead of `void` (only the return type is concept-enforced —
    `operation_concepts.hpp:109-114`).
  - **The override owns the tensor bindings on this concept.** The custom adapter *replaces* the framework's
    cache-hit tensor refresh rather than adding to it, so the returned `tensor_args` must carry **every**
    `TensorParameter` — which mirrors the ported-from override, whose address writes @ `:374-384, 398-400,
    409-411` cover all nine bindings. Verified against the brief's warning that the backdoor hash's `lr` /
    `step` exclusion is only safe *because* the override re-applies them.
  - The `TT_THROW("Core not in specified core ranges.")` @ `:302` moves into the new factory body verbatim.

## Planned Spec Shape

Default is 1:1 with legacy. Name vocabulary drops the `cb_` / `CBIndex` spelling entirely: DFB spec names are
role names (`param_in`, `tmp1`, …), matching the legacy trailing comments rather than the indices.

- **KernelSpecs** (4, or 3 when `core_group_2` is empty) — one per legacy `KernelDescriptor`:
  - `reader` — source `reader_moreh_adamw.cpp`; `compile_time_args` **empty** (every legacy reader CTA was
    `TensorAccessorArgs` plumbing and drops); 11 named RTAs; `hw_config =
    create_reader_datamovement_config(arch)`; `compiler_options.opt_level` left at Metal's `O2` default
    (matches the legacy DM resolution); `compiler_options.defines` carry `AMSGRAD` / `FP32_DEST_ACC_EN` on
    the same conditions as legacy.
  - `writer` — source `writer_moreh_adamw.cpp`; `compile_time_args` **empty**; 2 named RTAs; `hw_config =
    create_writer_datamovement_config(arch)`; `opt_level` default `O2`; same defines.
  - `compute_g1`, `compute_g2` — source `moreh_adamw.cpp`; one named CTA `per_core_tile_cnt` carrying that
    group's `num_units_per_core_group_N`; 1 named RTA `step`; `hw_config` a hand-built `ComputeGen1Config`
    (see below); **`opt_level = O3` set explicitly on each** (legacy `ComputeConfigDescriptor` resolves to
    O3, Metal 2.0's `CompilerOptions` defaults to O2 — a silent level drop otherwise); same defines.
    `compute_g2` exists only when `core_group_2` is non-empty, exactly as legacy.

  **`hw_config` for the compute kernels — hand-built `ComputeGen1Config`, *not* `to_compute_hardware_config`.**
  The op reaches its values through `get_compute_kernel_config_args`, which points at the recipe's Style A;
  but it then hand-assembles a Metal `ComputeConfigDescriptor` that **drops** `dst_full_sync_en` and
  `packer_l1_acc` (inventory note above). Routing through the TTNN helper would set
  `double_buffer_dest = !config.dst_full_sync_en`, whereas legacy always lands on the descriptor default
  `dst_full_sync_en = false` ⇒ `double_buffer_dest = true` — so for any caller passing
  `dst_full_sync_en = true` the helper would silently flip a perf setting the legacy op ignored. Fidelity
  wins over the style label: copy the three fields the legacy descriptor actually set, leave the rest at
  Metal defaults (which coincide with the legacy descriptor defaults).

  | legacy `ComputeConfigDescriptor` field | value | Metal 2.0 `ComputeGen1Config` |
  |---|---|---|
  | `math_fidelity` | resolved | `fpu_math_fidelity` — copied |
  | `fp32_dest_acc_en` | resolved | `enable_32_bit_dest` — copied |
  | `math_approx_mode` | resolved | `sfpu_precision_mode` = `Approximate` if true else `Precise` |
  | `dst_full_sync_en` | **not set** → `false` | `double_buffer_dest` left default `true` (= `!false`) ✓ |
  | `bfp8_pack_precise` | **not set** → `false` | `bfp_pack_precision_mode` left default `Approximate` ✓ |
  | `unpack_to_dest_mode` | **not set** → all `Default` | `unpack_modes` — see below |

  **`unpack_modes` is a *forced addition*.** When `fp32_dest_acc_en` is true, `intermed_cb_format` is
  `Float32`, and the validator requires an explicit entry for every Float32 DFB a compute kernel **consumes**
  with `enable_32_bit_dest = true` (`tt_metal/impl/metal2_host_api/program_spec.cpp:1049-1078`). Legacy set
  no `unpack_to_dest_mode` at all ⇒ every entry was `Default` ⇒ **`UnpackMode::UnpackToSrc`**. So under
  `fp32_dest_acc_en` each compute `KernelSpec` gets `UnpackToSrc` for the intermediate DFBs it consumes:
  `scalar_args`, `one`, `beta1_exponent`, `beta2_exponent`, and the self-loops `tmp_param`, `tmp_exp_avg`,
  `tmp_exp_avg_sq`, `tmp1`, `tmp2` (+ `tmp_max_exp_avg_sq` when `amsgrad`) — 9 entries, 10 with amsgrad. The
  data-format DFBs (`c_0`–`c_4`, `c_16`–`c_19`) never need one: `validate_inputs` admits only BFLOAT16 /
  BFLOAT8_B, so `data_format` is never Float32. No entry is added when `fp32_dest_acc_en` is false (the
  requirement doesn't fire, and `UnpackToSrc` is the assumed default either way).

- **DataflowBufferSpecs** (19 when `amsgrad`, **16** otherwise) — one per legacy `CBDescriptor`, with
  `entry_size = page_size` and `num_entries = total_size / page_size`, `data_format_metadata` copied, and
  `tile_format_metadata` left unset (no legacy CB set `.tile`). No aliasing, no borrowed memory, no
  `advanced_options` anywhere.

  | DFB spec name | from | entry_size | num_entries | data_format | declared |
  |---|---|---|---|---|---|
  | `param_in` | `c_0` | `data_tile_size` | 1 | `data_format` | always |
  | `grad` | `c_1` | `data_tile_size` | 1 | `data_format` | always |
  | `exp_avg_in` | `c_2` | `data_tile_size` | 1 | `data_format` | always |
  | `exp_avg_sq_in` | `c_3` | `data_tile_size` | 1 | `data_format` | always |
  | `max_exp_avg_sq_in` | `c_4` | `data_tile_size` | 1 | `data_format` | **`amsgrad` only** |
  | `scalar_args` | `c_5` | `intermed_tile_size` | **5** | `intermed_cb_format` | always |
  | `one` | `c_6` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |
  | `param_out` | `c_16` | `data_tile_size` | 1 | `data_format` | always |
  | `exp_avg_out` | `c_17` | `data_tile_size` | 1 | `data_format` | always |
  | `exp_avg_sq_out` | `c_18` | `data_tile_size` | 1 | `data_format` | always |
  | `max_exp_avg_sq_out` | `c_19` | `data_tile_size` | 1 | `data_format` | **`amsgrad` only** |
  | `tmp_param` | `c_24` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |
  | `tmp_exp_avg` | `c_25` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |
  | `tmp_exp_avg_sq` | `c_26` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |
  | `tmp_max_exp_avg_sq` | `c_27` | `intermed_tile_size` | 1 | `intermed_cb_format` | **`amsgrad` only** |
  | `beta1_exponent` | `c_28` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |
  | `beta2_exponent` | `c_29` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |
  | `tmp1` | `c_30` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |
  | `tmp2` | `c_31` | `intermed_tile_size` | 1 | `intermed_cb_format` | always |

  The legacy `c_24..c_31` loop stays a loop only where it still reads as one: the four unconditional
  same-shaped intermediates plus the two exponent buffers now have distinct names, so they are declared
  individually. `intermed_tile_size` and the format are shared by construction.

  **The three `amsgrad`-only specs are the port's one piece of new structure.** Legacy allocates `c_4`,
  `c_19` and `c_27` unconditionally, but under `amsgrad == false` no kernel touches them (every reference is
  inside `#ifdef AMSGRAD`) — and a DFB with no PRODUCER *and* no CONSUMER binding is rejected by the spec
  validator. They are **not dropped**: `c_4` / `c_19` are ordinary 1:1 and `c_27` is a live self-loop when
  `amsgrad` is on. There is no dead CTA to remove alongside them (none of the three indices was ever threaded
  to a kernel as a compile-time arg). Net effect under `amsgrad == false`: three tiles of L1 per core that
  legacy burned are no longer allocated — the only observable change the port makes, and it is forced.

- **SemaphoreSpecs**: none — the legacy op declares no semaphores.

- **TensorParameters** (9 when `amsgrad`, **7** otherwise) — one per distinct originating tensor, each with
  `.spec = <tensor>.tensor_spec()` off the `MeshTensor` and `relaxations` left default (strict):
  `param_in`, `grad`, `exp_avg_in`, `exp_avg_sq_in`, `max_exp_avg_sq_in`*, `param_out`, `exp_avg_out`,
  `exp_avg_sq_out`, `max_exp_avg_sq_out`* (*`amsgrad` only). Each is bound by exactly one kernel (the five
  inputs by `reader`, the four outputs by `writer`), so there is no multi-kernel-same-tensor case here.

- **WorkUnitSpecs** (2, or 1 when `core_group_2` is empty) — one per compute core group, since that is the
  only axis on which the kernel sets differ:
  - `wu_g1`: `{reader, writer, compute_g1}` on `core_group_1`
  - `wu_g2`: `{reader, writer, compute_g2}` on `core_group_2` (only when non-empty)

  `reader` and `writer` belong to both, so their effective node set is `core_group_1 ∪ core_group_2` =
  `all_cores`, reproducing the legacy `.core_ranges = all_cores`. This also satisfies the local-DFB invariant
  that a DFB's producer and consumer share identical `WorkUnitSpec` membership — every DFB here is bound by
  kernels present in both work units, or (the self-loops) by a compute kernel binding both ends.

- **Op-owned tensors**: none. The optional outputs built by `create_output_tensors`
  (`device/moreh_adamw_device_operation.cpp:99-133`) are ordinary op outputs reachable from
  `tensor_return_value`, not factory-allocated scratch.

## Preserved Multiplicity

```
Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source device/kernels/moreh_adamw.cpp
  → KernelSpecs [compute_g1, compute_g2] of same source
  → in WorkUnitSpecs [wu_g1, wu_g2]
  → sharing upstream/downstream DFBs (endpoint role each KernelSpec binds):
      param_in, grad, exp_avg_in, exp_avg_sq_in, scalar_args, one,
      beta1_exponent, beta2_exponent            → CONSUMER (each spec)
      param_out, exp_avg_out, exp_avg_sq_out    → PRODUCER (each spec)
      tmp_param, tmp_exp_avg, tmp_exp_avg_sq,
      tmp1, tmp2                                → PRODUCER + CONSUMER (self-loop, each spec)
      max_exp_avg_sq_in                         → CONSUMER   (amsgrad only)
      max_exp_avg_sq_out                        → PRODUCER   (amsgrad only)
      tmp_max_exp_avg_sq                        → PRODUCER + CONSUMER (self-loop, amsgrad only)
```

The per-group tile count stays a **CTA** (`per_core_tile_cnt`) on each spec — it is not demoted to an RTA.
`core_group_1` and `core_group_2` are **disjoint**, so each node sees exactly one compute instance and each
shared DFB is an ordinary single-role binding per node: no `allow_instance_multi_binding`, and this is *not*
the same-grid two-toucher case.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `multi_core_program_factory.cpp:193-199` | 4–5× `TensorAccessorArgs(*t.buffer()).append_to(reader_ct_args)` | `TensorBinding`s on `reader` — the reader's whole CTA list disappears |
| `multi_core_program_factory.cpp:202-207` | 3–4× `TensorAccessorArgs(*t.buffer()).append_to(writer_ct_args)` | `TensorBinding`s on `writer` — the writer's whole CTA list disappears |
| `:269-276` + reader RTA slots 0-4 (`:307-311`) | five `Buffer*` captures pushed through `emplace_runtime_args` | `TensorParameter`/`TensorBinding` `param_in`, `grad`, `exp_avg_in`, `exp_avg_sq_in`, `max_exp_avg_sq_in` |
| `:278-281` + writer RTA slots 0-3 (`:326-329`) | four `Buffer*` captures pushed through `emplace_runtime_args` | `TensorParameter`/`TensorBinding` `param_out`, `exp_avg_out`, `exp_avg_sq_out`, `max_exp_avg_sq_out` |
| `:374-379` (override) | `std::array<uint32_t,5> reader_addrs{…->address()…}` + write loop `:398-400` | `tensor_args` entries in the returned `ProgramRunArgs` |
| `:380-384` (override) | `std::array<uint32_t,4> writer_addrs{…->address()…}` + write loop `:409-411` | `tensor_args` entries in the returned `ProgramRunArgs` |
| `reader_moreh_adamw.cpp:15-19` | `get_arg_val<uint32_t>(i++)` ×5 address reads | gone — addresses ride the binding |
| `reader_moreh_adamw.cpp:46-49, 58` | `TensorAccessorArgs<0>()` + `next_compile_time_args_offset()` chain | `TensorAccessor(tensor::<name>)` |
| `writer_moreh_adamw.cpp:12-15` | `get_arg_val<uint32_t>(i++)` ×4 address reads | gone — addresses ride the binding |
| `writer_moreh_adamw.cpp:24-26, 34` | `TensorAccessorArgs<0>()` + chain | `TensorAccessor(tensor::<name>)` |
| `reader_moreh_adamw.cpp:34-44, 57` | 8 `constexpr uint32_t cb_id_* = tt::CBIndex::c_N` magic indices | `DFBBinding`s → `dfb::<name>` |
| `writer_moreh_adamw.cpp:20-22, 33` | 4 `constexpr uint32_t cb_id_* = tt::CBIndex::c_N` | `DFBBinding`s → `dfb::<name>` |
| `moreh_adamw.cpp:20-65` | 19 `constexpr auto cb_* = tt::CBIndex::c_N` (each paired with a `DataflowBuffer` object) | `DFBBinding`s → `dfb::<name>`; the object is constructed from the token and the raw-index LLK call sites take the token directly |
| `moreh_adamw.cpp:18` | `get_compile_time_arg_val(0)` | named CTA `get_arg(args::per_core_tile_cnt)` |
| `reader_moreh_adamw.cpp:21-32`, `writer_moreh_adamw.cpp:17-18`, `moreh_adamw.cpp:17` | positional `get_arg_val<uint32_t>(i++)` / `(0)` | named RTAs `get_arg(args::<name>)` |

**Page-size 3rd-argument CTAs/RTAs**: none — all nine accessors are 2-arg (audit GREEN).
**Semaphore-ID RTAs**: none — the op has no semaphores.
**Positional CTAs**: the reader/writer lists were *entirely* `TensorAccessorArgs` and vanish; the compute
list's single slot becomes the named `per_core_tile_cnt`.

**Retained RTAs, and one thing deliberately *not* changed.** Reader keeps `lr`, `beta1`, `beta2`, `eps`,
`weight_decay`, `beta1_exponent`, `beta2_exponent`, `step`, `amsgrad`, `num_tiles_per_core`, `start_id`;
writer keeps `num_tiles_per_core`, `start_id`; compute keeps `step`. Of these, only `num_tiles_per_core` and
`start_id` actually vary per node — every other one is set to the same value on every core and is really a
**CRTA**. That conversion changes dispatch semantics, so the port does **not** make it; noted for the
later name-first / CRTA cleanup pass. The three dead RTAs the audit flagged (reader `step`, reader `amsgrad`,
compute `step`) are **ported as-is**: removing a dead argument is an ops-team functional change, and the
ported-from override writes reader `step` on every cache hit, so dropping it would also change what the
translated override has to carry.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  / [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  `tmp_param`, `tmp_exp_avg`, `tmp_exp_avg_sq`, `tmp1`, `tmp2` (and `tmp_max_exp_avg_sq` under `amsgrad`) —
  compute-kernel scratch with a single toucher, bound PRODUCER **and** CONSUMER under one accessor name.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings):
  `max_exp_avg_sq_in`, `max_exp_avg_sq_out`, `tmp_max_exp_avg_sq` DFBs and the `max_exp_avg_sq_in` /
  `max_exp_avg_sq_out` tensors, all gated on `amsgrad`. The legacy op **already** gates the kernel side on an
  `#ifdef AMSGRAD` fed by `defines`, so no `if constexpr`→`#ifdef` promotion is needed — the existing
  preprocessor structure is exactly what the pattern requires, and the port's job is to make the *host*
  bindings conditional on the same flag.
- [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  (anti-pattern, avoided): the two compute `KernelSpec`s keep their per-group tile count as a CTA across
  disjoint `WorkUnitSpec`s.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  the compute kernel's raw-index LLK calls (`binary_op_init_common`, `sub_tiles`, `mul_tiles`, `add_tiles`,
  `copy_tile`) take `dfb::<name>` directly via the token's `uint32_t` conversion; the object-taking donor
  helpers in `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` keep their existing local `DataflowBuffer`
  objects, now constructed from tokens.
- Not applied, and why: **Aliased DFBs** (no legacy CB carried multiple `buffer_index`es);
  **Same-FIFO aliasing** (no CB index is aliased under a second name); **Two-toucher 1P+1C** (no CB has two
  touchers on one node needing assignment); **Multi-variant factories** (single variant);
  **Removing pybound legacy factory entry points** (`moreh_adamw_nanobind.cpp` binds only
  `ttnn::bind_function<"moreh_adamw">` @ `:43` — no `create_descriptor` exposure, so nothing to delete);
  **Unity-build hygiene** (one factory `.cpp` in this op; its anonymous-namespace names are prefixed by the
  op's own vocabulary).

## Deferred / Flagged

- **New finding (structural, resolved with the invoker):** the direct-`create_descriptor` device-op shape has
  no Metal 2.0 counterpart in the framework, so the port must introduce `program_factory_t` — an edit to the
  device-op header outside the recipe's two documented exceptions. Detail in [Flags](#flags) item 1.
- **New finding (fidelity, resolved in-plan):** the recipe's compute-`hw_config` Style A / Style B split does
  not cover this op's hybrid — TTNN-resolved values hand-assembled into a Metal descriptor that drops two of
  them. Following Style A would silently flip `double_buffer_dest`. Resolved as Style B; see
  [Planned Spec Shape](#planned-spec-shape).
- **New finding (validator-forced addition):** `unpack_modes` entries must be added where legacy had none,
  under `fp32_dest_acc_en`. Legacy value is `Default` ⇒ `UnpackToSrc`; nine or ten entries per compute spec.
- **Latent legacy inconsistency, not fixed:** `amsgrad == true` with `max_exp_avg_sq_in` absent
  ([Flags](#flags) item 3).
- The three dead RTAs and the two unused includes the audit recorded are carried forward untouched and
  re-reported.
