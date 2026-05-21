# Port Plan — `ttnn::operations::data_movement::clone::CloneOperation`

Port plan for the clone op, ported from imperative `host_api.hpp` builder style directly to Metal 2.0 (`ProgramSpecFactoryConcept`), per user override on the audit's Q1.

Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Factory shape

- **Concept:** `ProgramFactoryConcept` (oldest tier) — `cached_program_t create(...)` returning `Program` + shared variables, with `override_runtime_arguments(...)` hook. **Not** `ProgramDescriptorFactoryConcept`.
- **Variants:** single `ProgramFactory`, branching internally on two dimensions:
  - `tilized = output.layout() == Layout::TILE` (CTA-level branch — selects kernel sources)
  - `is_sharded = output_memory_layout != INTERLEAVED` (CTA-level branch — also affects work-split)
  - Total four kernel-pair combinations: T+I, RM+I, T+S, RM+S. Plus optional compute kernel when `convert_dtype = true` (requires tilized; can co-occur with sharded).

### Kernels

Per the legacy factory's branching, the kernel set instantiated for a single Program execution depends on the variant. The full list of kernel source files referenced by the factory:

| unique_id (in plan) | source (legacy file path) | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader_t_i | `device/kernels/read_kernel.cpp` | `all_cores` | `{src_cb_id}` + `TensorAccessorArgs(input_buffer)` | `{input.buffer()->address(), num_units_per_core, start_id}` | none | `ReaderDataMovementConfig` |
| reader_rm_i | `device/kernels/read_kernel_rm.cpp` | `all_cores` | `{src_cb_id, input_unit_size}` + `TensorAccessorArgs(input_buffer)` | `{input.buffer()->address(), input_unit_size, num_units_per_core, start_id}` | none | `ReaderDataMovementConfig` |
| reader_t_s | `device/kernels/read_kernel_sharded.cpp` | `all_cores` | `{src_cb_id}` (no `TensorAccessorArgs`) | `{input.buffer()->address(), num_units_per_core}` | none | `ReaderDataMovementConfig` |
| reader_rm_s | `device/kernels/read_kernel_rm_sharded.cpp` | `all_cores` | `{src_cb_id, input_unit_size}` (no `TensorAccessorArgs`) | `{input.buffer()->address(), input_unit_size, num_units_per_core}` | none | `ReaderDataMovementConfig` |
| writer_t_i | `device/kernels/write_kernel.cpp` | `all_cores` | `{dst_cb_id}` + `TensorAccessorArgs(output_buffer)` | `{output.buffer()->address(), num_units_per_core, start_id}` | none | `WriterDataMovementConfig` |
| writer_rm_i | `device/kernels/write_kernel_rm.cpp` | `all_cores` | `{dst_cb_id, output_unit_size}` + `TensorAccessorArgs(output_buffer)` | `{output.buffer()->address(), output_unit_size, num_units_per_core, start_id}` | none | `WriterDataMovementConfig` |
| writer_t_s | `device/kernels/write_kernel_sharded.cpp` | `all_cores` | `{dst_cb_id}` | `{output.buffer()->address(), num_units_per_core}` | none | `WriterDataMovementConfig` |
| writer_rm_s | `device/kernels/write_kernel_rm_sharded.cpp` | `all_cores` | `{dst_cb_id, output_unit_size}` | `{output.buffer()->address(), output_unit_size, num_units_per_core}` | none | `WriterDataMovementConfig` |
| compute | `device/kernels/compute_kernel.cpp` | per-group (`core_group_1` or `core_group_2`) | `{src_cb_id, dst_cb_id, num_units_per_core}` — note `num_units_per_core` is here a per-group CTA | none | none | `ComputeConfig{math_fidelity, fp32_dest_acc_en, ...}` |

Notes:
- For the data-movement kernels, `num_units_per_core` and `start_id` are RTAs that vary per core (not per group), so a single `KernelSpec` per role suffices in the port (no Anti-pattern: Demoting per-group CTA to RTA risk).
- For the compute kernel, `num_units_per_core` is a per-group **CTA**. If both `core_group_1` and `core_group_2` are populated and have different `num_units_per_core`, the legacy code creates two compute `KernelDescriptor`s with the same source but different CTAs. The Metal 2.0 port must preserve this multiplicity — two `KernelSpec`s of the same source, one per group — per [Anti-pattern: Demoting per-group CTA to RTA](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta).

### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `src_cb_id` = `CBIndex::c_4` | `2 * aligned_input_unit_size` | `all_cores` | `input_data_format` | `aligned_input_unit_size` | not set (default 32x32) |
| `dst_cb_id` = `CBIndex::c_20` (only if `convert_dtype`) | `2 * aligned_output_unit_size` | `all_cores` | `output_data_format` | `aligned_output_unit_size` | not set |

When `!convert_dtype`, the factory sets `dst_cb_id = src_cb_id` and creates only one CB; the writer reads from the reader's CB.

### Semaphores

none.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| `clone_program_factory.cpp:121` | input | RTA slot 0 (`input_buffer->address()`) | CTA offset 1 (after `src_cb_id`) for tilized; 2 (after `src_cb_id, input_unit_size`) for RM |
| `clone_program_factory.cpp:123` | output | RTA slot 0 | CTA offset 1 (tilized) or 2 (RM) |
| `clone_program_factory.cpp:126` | input (RM variant) | as above | offset 2 |
| `clone_program_factory.cpp:128` | output (RM variant) | as above | offset 2 |

The sharded kernels do **not** use `TensorAccessor`; they pass the buffer address as an RTA and call `get_noc_addr(buffer_address)` directly. The port replaces this with a `TensorBinding` + kernel-side `TensorAccessor(ta::name)` whose `.bank_base_address` substitutes for the legacy RTA. See [Applied Patterns](#applied-patterns).

### Work split

- For `is_sharded`: `all_cores = shard_spec.grid(); num_cores = all_cores.num_cores(); core_group_1 = all_cores; core_group_2 = empty;` — all cores in one group, `num_units_per_core_group_1 = shard_height` (RM) or `shard_height * shard_width / TILE_HW` (tilized).
- For interleaved: `split_work_to_cores(grid_size, num_units)` yields `(num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2)` — standard work-split pattern.

### Cross-op kernels

none — all kernels are op-local under `device/kernels/`.

### Flags

- The legacy factory's `override_runtime_arguments` updates RTA slot 0 (buffer addresses) on cache hit. In Metal 2.0 this becomes implicit via `TensorBinding` auto-injection.
- Five Device 2.0 DM holdovers (`get_tile_size(cb_id)` free function) listed in the audit; will be folded into the port-time cleanup per the audit doc's default recommendation.

## Planned Spec Shape

The port preserves the four-way kernel branching inside `create_program_spec`. Resource declarations vary across branches; the framework adapter receives a fully-populated spec per execution.

- **KernelSpecs**:
  - **Always one reader, one writer.** Per execution, the factory selects one reader source (from `read_kernel.cpp` / `read_kernel_rm.cpp` / `read_kernel_sharded.cpp` / `read_kernel_rm_sharded.cpp`) and one writer source (symmetric).
  - **Compute when `convert_dtype = true`.** One or two `KernelSpec`s of `compute_kernel.cpp`, one per populated core group, preserving the per-group `num_units_per_core` CTA. Variable count: 0, 1, or 2.
- **DataflowBufferSpecs**:
  - `INPUT_DFB` always present (size `2 * aligned_input_unit_size`, num_entries=2, entry_size=`aligned_input_unit_size`, format=`input_data_format`).
  - `OUTPUT_DFB` only when `convert_dtype = true` (similar shape on output side).
  - When `!convert_dtype`, the writer binds INPUT_DFB as its CONSUMER (legacy aliased `dst_cb_id = src_cb_id`).
- **SemaphoreSpecs**: none.
- **TensorParameters**:
  - `INPUT_TENSOR` and `OUTPUT_TENSOR` always declared.
  - Bound on reader (`INPUT_TENSOR` → `ta::input`) and writer (`OUTPUT_TENSOR` → `ta::output`).
- **WorkUnitSpecs**:
  - **Interleaved**: when `core_group_2` is non-empty, two WUs (`wu_g1`, `wu_g2`) splitting reader/writer (and compute if convert_dtype) per group, even when the data-movement kernels share the same `KernelSpec` (a `KernelSpec` placed in two WUs is the supported shape — see migration guide on `WorkUnitSpec`). When `core_group_2` is empty, one WU `wu_main`.
  - **Sharded**: one WU `wu_main` on the shard grid.
  - For the compute case, the per-group compute `KernelSpec`s each belong to one WU.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| 2× `compute_kernel.cpp` (per-group, when `convert_dtype` and `core_group_2` non-empty) | 2× `KernelSpec` named `compute_g1`, `compute_g2` with different `num_units_per_core` CTA | `wu_g1`, `wu_g2` | INPUT_DFB (multi-CONSUMER from both compute KernelSpecs); OUTPUT_DFB (multi-PRODUCER from both compute KernelSpecs) |

The data-movement kernels are single-KernelSpec; per-core variation is fully captured in per-node RTAs. The compute kernel is the only multiplicity site, and only on the `convert_dtype + interleaved-two-group` code path.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `clone_program_factory.cpp:120` reader CTA slot 0 | `(uint32_t)src_cb_id` (magic CB index) | `DFBBinding(INPUT_DFB, "input", PRODUCER)` |
| `clone_program_factory.cpp:121,126` reader CTA tail | `TensorAccessorArgs(*input_buffer).append_to(...)` | `TensorBinding(INPUT_TENSOR, "input")` |
| `clone_program_factory.cpp:122,127` writer CTA slot 0 | `(uint32_t)dst_cb_id` (magic CB index — equals src when !convert_dtype) | `DFBBinding(INPUT_DFB or OUTPUT_DFB, "output", CONSUMER)` |
| `clone_program_factory.cpp:123,128` writer CTA tail | `TensorAccessorArgs(*output_buffer).append_to(...)` | `TensorBinding(OUTPUT_TENSOR, "output")` |
| `clone_program_factory.cpp:160-162` compute CTA slot 0/1 | `(uint32_t)src_cb_id, (uint32_t)dst_cb_id` (magic CB indices) | `DFBBinding(INPUT_DFB, "input", CONSUMER)` + `DFBBinding(OUTPUT_DFB, "output", PRODUCER)` |
| `clone_program_factory.cpp:160-162` compute CTA slot 2 | `num_units_per_core` positional | named CTA `"num_units"` (per-group, preserved multiplicity) |
| `clone_program_factory.cpp:196,213,234,253` reader RTA slot 0 | `input_buffer->address()` (positional) | `TensorBinding` auto-injection (consumed via `TensorAccessor(ta::input)` constructor on device) |
| `clone_program_factory.cpp:204,222,243,262` writer RTA slot 0 | `output_buffer->address()` (positional) | same on output side |
| `clone_program_factory.cpp:197/214/235/254` reader RTA slot 1 or 2 | `num_units_per_core` positional | named RTA `"num_units"` |
| `clone_program_factory.cpp:214,236,255` reader RTA slot for stick / start_id | `input_unit_size` (RM) / `start_id` (tilized interleaved / RM interleaved) | named RTAs `"stick_size"` and `"start_id"` |
| writer side analogous to reader |  |  |
| `clone_device_operation.hpp:30-33` `shared_variables_t` | `read_kernel_id`, `write_kernel_id`, `cores` cached in `cached_program_t` | gone — Metal 2.0 framework caches the `Program` itself; `create_program_spec` returns paired `ProgramArtifacts` per execution |
| `clone_program_factory.cpp:275-291` `override_runtime_arguments` | per-core RTA slot[0] update with new buffer addresses | gone — buffer addresses auto-inject via `TensorBinding` (cache-hit path uses `UpdateTensorArgs`) |

The CTA positional `src_cb_id` token, the `TensorAccessorArgs` plumbing, the buffer-address RTA, and the `override_runtime_arguments` hook all disappear. The named-RTA values (per-core slice info) survive but as named instead of positional.

## Applied Patterns

- **Multi-variant factory** ([catalog entry](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories)): four kernel-pair branches on (tilized × is_sharded), plus optional compute branch. Each branch's spec is built within the same `create_program_spec` body (no class hierarchy needed).
- **Preserved multiplicity** for compute when `convert_dtype` + `core_group_2` non-empty: two `KernelSpec`s of the same `compute_kernel.cpp` source, one per group; both bind INPUT_DFB / OUTPUT_DFB. Avoids the [Demoting per-group CTA to RTA](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) anti-pattern.
- **TensorAccessor injection on sharded kernels** (a Metal 2.0 idiom that subsumes the legacy buffer-address-RTA + `get_noc_addr(addr)` pattern even when the kernel's "real" access is direct L1 striding rather than per-page iteration). Kernel construct: `auto input_a = TensorAccessor(ta::input); local_l1_read_addr = get_noc_addr(input_a.bank_base_address);`. This goes one step beyond the strict reading of the recipe's "sanctioned kernel changes" list (which contemplates `TensorAccessorArgs<N>()` → `TensorAccessor(ta::name)` substitution but doesn't list "introduce a TensorAccessor where there was none"); it's the natural Metal 2.0 idiom for the case where a buffer-address RTA was the only legacy plumbing. Per the audit's Q2, this is the chosen resolution.
- **Unity-build hygiene**: clone has a single program-factory `.cpp`, so the standard prefix-with-factory-name dance isn't required at the same scale as multi-factory ops. Anon-namespace constants get a `C_` prefix anyway for forward-compatibility if sibling factories later land in the same unity TU.

## Deferred / Flagged

- **Audit Q1 (Check 1 framing, direct port from `host_api.hpp`)**: proceeding under user override.
- **Audit Q2 (sharded kernels Check 3 resolution)**: proceeding with `TensorAccessor.bank_base_address` substitution per Option A in the audit. Marked as a doc-evolution finding in [METAL2_PORT_REPORT.md → Friction] when the port report is written.
- **Audit Q3 (Device 2.0 DM holdovers)**: folding `get_tile_size(cb_id)` → `cb_obj.get_tile_size()` as port-time cleanup per the audit doc default.
- **New finding during planning — the sharded kernel access pattern.** The sharded kernels iterate by raw stride (`local_l1_read_addr += stick_size`) rather than per-page. The legacy code conflates two distinct things: (i) "get me the local L1 base address of this tensor's shard" (which `TensorAccessor.bank_base_address` answers cleanly), and (ii) "iterate me through that L1 region by stride" (which the kernel does manually with no abstraction). The Metal 2.0 port keeps (ii) as-is and replaces (i) with the TensorAccessor accessor. The cleaner long-term shape is a borrowed-memory DFB on the input/output shards, but per audit Q2 we're not taking that path here.
- **New finding during planning — `Program` vs `CachedProgram` model.** The legacy `cached_program_t` caches the `Program` plus three shared variables (`read_kernel_id`, `write_kernel_id`, `cores`); the override hook iterates `cores` and updates RTA slot 0. In Metal 2.0 the framework owns the `Program` cache; `create_program_spec` is re-invoked each execution with fresh artifacts, and `UpdateTensorArgs` handles the address-update fast path. None of the three shared variables survives the port. This is the cleanest case of "structural debt evaporates" the migration guide's Principle 1 advertises.
