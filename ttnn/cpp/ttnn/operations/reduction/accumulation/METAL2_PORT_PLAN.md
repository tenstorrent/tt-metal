# Port Plan — `ttnn/cpp/ttnn/operations/reduction/accumulation`

Port plan for the `accumulation` op directory, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

The directory holds **two device operations**, each with exactly one factory. They share the kernel
header `device/kernels/accumulation_common.hpp`, so they are one port unit (per the brief's *Scope*
section):

- `AccumulationDeviceOperation` → `AccumulationProgramFactory` (backs `cumsum` / `cumprod`)
- `EmaDeviceOperation` → `EmaProgramFactory` (backs `ema`)

Both factories are ported in this change. The inventory and planning sections below are therefore
split per factory, using the multi-variant nesting convention from the recipe's Appendix A.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — **both** factories (each exposes
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`;
  [accumulation_device_operation.hpp:46-49](device/accumulation_device_operation.hpp#L46-L49),
  [ema_device_operation.hpp:24-27](ema/device/ema_device_operation.hpp#L24-L27)).
- Variants: single factory per device operation. `program_factory_t` is a one-alternative
  `std::variant` in both cases
  ([accumulation_device_operation.hpp:55](device/accumulation_device_operation.hpp#L55),
  [ema_device_operation.hpp:30](ema/device/ema_device_operation.hpp#L30)).
  Neither factory selects its kernel *source* at runtime — each `KernelDescriptor` has one fixed
  source path, so there is no runtime kernel-source fan-out.
- Custom `compute_program_hash`: **none** on either device operation — already the default
  reflection-based hash. Confirmed by grep over the op directory: no `compute_program_hash`,
  no `get_dynamic_runtime_args`, no `override_runtime_arguments`.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section
below.)*

---

### Variant: `AccumulationProgramFactory`

Source: [device/accumulation_program_factory.cpp](device/accumulation_program_factory.cpp).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/accumulation_reader.cpp` | `all_cores` | `TensorAccessorArgs(input_tensor)` block only ([:165-166](device/accumulation_program_factory.cpp#L165-L166)) | none | per core, 8 values: `{input_tensor, num_tiles_per_core, tiles_per_row, input_tile_offset, tile_offset, tile_offset / input_tile_offset, tile_offset % input_tile_offset, flip}` ([:231-240](device/accumulation_program_factory.cpp#L231-L240)) | none | none | `ReaderConfigDescriptor{}` ([:176](device/accumulation_program_factory.cpp#L176)) |
| writer | `device/kernels/dataflow/accumulation_writer.cpp` | `all_cores` | `TensorAccessorArgs(output_tensor)` block only ([:168-169](device/accumulation_program_factory.cpp#L168-L169)) | none | per core, 8 values: same shape, `output_tensor` in slot 0 ([:242-251](device/accumulation_program_factory.cpp#L242-L251)) | none | none | `WriterConfigDescriptor{}` ([:183](device/accumulation_program_factory.cpp#L183)) |
| compute (group 1) | `device/kernels/compute/accumulation_compute.cpp` | `core_group_1` | `{bit_cast<uint32_t>(default_acc_value)}` ([:191](device/accumulation_program_factory.cpp#L191)) | none | per core, 2 values: `{num_tiles_per_core, tiles_per_row}` ([:254](device/accumulation_program_factory.cpp#L254)) | none | `BINARY_OP_INIT`, `BINARY_OP`, `FILL_TILE` ([:131-147](device/accumulation_program_factory.cpp#L131-L147)) | `ComputeConfigDescriptor{math_fidelity = default_math_fidelity, fp32_dest_acc_en = true, dst_full_sync_en = false, unpack_to_dest_mode = unpack_to_dst, math_approx_mode = false}` ([:193-199](device/accumulation_program_factory.cpp#L193-L199)) |
| compute (group 2) | same source | `core_group_2` (descriptor pushed only when non-empty) | identical to group 1 | none | per core, 2 values ([:257](device/accumulation_program_factory.cpp#L257)) | none | identical | identical ([:209-215](device/accumulation_program_factory.cpp#L209-L215)) |

`default_math_fidelity` is `HiFi3` on Wormhole B0 with a FLOAT32 output, else `HiFi4`
([:159-163](device/accumulation_program_factory.cpp#L159-L163)) — a hardware-bug workaround (#38306)
the port carries over verbatim.

`default_acc_value` is `0.0f` for CUMSUM; for CUMPROD it is `1.0f`, or the bit pattern `0x00000001`
when the output format is an integer format ([:149-157](device/accumulation_program_factory.cpp#L149-L157)).

#### CBs

Built through the `push_cb` lambda at
[accumulation_program_factory.cpp:102-116](device/accumulation_program_factory.cpp#L102-L116);
`total_size = num_tiles * tt::tile_size(data_format)`, `page_size = tt::tile_size(data_format)`,
one `CBFormatDescriptor` each (no aliasing), `tile` unset.

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`SRC`) | `4 * tile_size(input_dataformat)` | `all_cores` | `input_dataformat` = input dtype | `tile_size(input_dataformat)` | unset |
| `c_2` (`ACC`) | `1 * tile_size(acc_dataformat)` | `all_cores` | `acc_dataformat` — output dtype if integer, else `Float32` ([:93-96](device/accumulation_program_factory.cpp#L93-L96)) | `tile_size(acc_dataformat)` | unset |
| `c_1` (`DST`) | `4 * tile_size(output_dataformat)` | `all_cores` | `output_dataformat` = output dtype | `tile_size(output_dataformat)` | unset |

No `GlobalCircularBuffer`: no `.global_circular_buffer` field, no `remote_cb_config`, no
`global_cb` parameter anywhere in the factory.

#### Semaphores

none — the factory pushes no `SemaphoreDescriptor`, and no kernel performs a semaphore operation.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [accumulation_program_factory.cpp:165-166](device/accumulation_program_factory.cpp#L165-L166) → kernel [accumulation_reader.cpp:16,36](device/kernels/dataflow/accumulation_reader.cpp#L16-L36) | `tensor_args.input_tensor.mesh_tensor()` | reader slot 0 (the `MeshTensor` itself, via `emplace_runtime_args`) |
| [accumulation_program_factory.cpp:168-169](device/accumulation_program_factory.cpp#L168-L169) → kernel [accumulation_writer.cpp:13,30](device/kernels/dataflow/accumulation_writer.cpp#L13-L30) | `tensor_return_value.mesh_tensor()` | writer slot 0 (same mechanism) |

Both are **Case 1** (consumed through `TensorAccessor`; no raw base-address arithmetic). Both use the
two-argument `TensorAccessor(args, addr)` form — no page-size third argument to drop.

#### Work split

- Driver: `tt::tt_metal::split_work_to_cores(grid, num_rows_total)`
  ([accumulation_program_factory.cpp:78-80](device/accumulation_program_factory.cpp#L78-L80))
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `num_cols_per_core_group_1`,
  `num_cols_per_core_group_2` — the standard 6-tuple. `all_cores` is the union of the two groups;
  the groups are disjoint and `core_group_2` may be empty.
- Per-core RTA values are assigned in a single loop over `i ∈ [0, num_cores)` mapping to
  `CoreCoord{i / num_cores_y, i % num_cores_y}` ([:219-263](device/accumulation_program_factory.cpp#L219-L263)),
  accumulating a running `tile_offset`.

---

### Variant: `EmaProgramFactory`

Source: [ema/device/ema_program_factory.cpp](ema/device/ema_program_factory.cpp).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `ema/kernels/dataflow/ema_reader.cpp` | `all_cores` | `{total_tiles_per_core}` then `TensorAccessorArgs(input)` ([:124-125](ema/device/ema_program_factory.cpp#L124-L125)) | none | per core, 2 values: `{input, src_start_tile}` ([:184](ema/device/ema_program_factory.cpp#L184)) | none | none | `DataMovementConfigDescriptor{processor = RISCV_0, noc = preferred_noc_for_dram_read(arch)}` ([:147-150](ema/device/ema_program_factory.cpp#L147-L150)) |
| writer | `ema/kernels/dataflow/ema_writer.cpp` | `all_cores` | `{total_tiles_per_core}` then `TensorAccessorArgs(output)` ([:127-128](ema/device/ema_program_factory.cpp#L127-L128)) | none | per core, 2 values: `{output, dst_start_tile}` ([:185](ema/device/ema_program_factory.cpp#L185)) | none | none | `DataMovementConfigDescriptor{processor = RISCV_1, noc = preferred_noc_for_dram_write(arch)}` ([:157-160](ema/device/ema_program_factory.cpp#L157-L160)) |
| compute | `ema/kernels/compute/ema_compute.cpp` | `all_cores` | `{total_batch_channel_tiles_per_core, tiles_per_channel, alpha_bits, beta_bits}` ([:130-135](ema/device/ema_program_factory.cpp#L130-L135)) | none | none | none | none | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` from `get_compute_kernel_config_args(arch, attributes.compute_kernel_config)` ([:162-175](ema/device/ema_program_factory.cpp#L162-L175)) |

`preferred_noc_for_dram_read` returns `NOC_0` and `preferred_noc_for_dram_write` returns `NOC_1` on
every architecture ([kernel_types.hpp:134-147](../../../../../../../tt_metal/api/tt-metalium/kernel_types.hpp#L134-L147)),
so the resolved DM triples are reader `(RISCV_0, NOC_0, DM_DEDICATED_NOC)` and writer
`(RISCV_1, NOC_1, DM_DEDICATED_NOC)`. **Neither matches the reader or writer default** — the
processors are swapped relative to the conventional assignment. See
[Hardware configuration](#hardware-configuration-notes) below.

`packer_l1_acc` is destructured out of `get_compute_kernel_config_args` but never used by the
descriptor ([:162-175](ema/device/ema_program_factory.cpp#L162-L175)).

#### CBs

Three `CBDescriptor`s at
[ema_program_factory.cpp:92-120](ema/device/ema_program_factory.cpp#L92-L120), one
`CBFormatDescriptor` each (no aliasing), `tile` unset. `ema_buffer_depth = 2`
([:19](ema/device/ema_program_factory.cpp#L19)).

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`src_cb`) | `src_tile_size * 2` | `all_cores` | `src_data_format` (input dtype) | `src_tile_size` | unset |
| `c_1` (`dst_cb`) | `dst_tile_size * 2` | `all_cores` | `dst_data_format` (output dtype) | `dst_tile_size` | unset |
| `c_2` (`prev_cb`) | `src_tile_size` | `all_cores` | `src_data_format` | `src_tile_size` | unset |

`src_tile_size` / `dst_tile_size` come from `tensor_spec().tile().get_tile_size(format)`
([:85-86](ema/device/ema_program_factory.cpp#L85-L86)). No `GlobalCircularBuffer`.

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [ema_program_factory.cpp:124-125](ema/device/ema_program_factory.cpp#L124-L125) → kernel [ema_reader.cpp:17,34](ema/kernels/dataflow/ema_reader.cpp#L17-L34) | `tensor_args.input.mesh_tensor()` | reader slot 0 |
| [ema_program_factory.cpp:127-128](ema/device/ema_program_factory.cpp#L127-L128) → kernel [ema_writer.cpp:17,34](ema/kernels/dataflow/ema_writer.cpp#L17-L34) | `tensor_return_value.mesh_tensor()` | writer slot 0 |

Both **Case 1**, both the two-argument form. The accessor CTA block starts at slot **1** in both
kernels (behind `total_tiles_per_core`), hence `TensorAccessorArgs<1>()`.

#### Work split

- Driver: `get_max_cores_divisible_by_tiles_per_core_tiles(total_batch_channel_tiles, num_cores_available, /*request_even=*/false)`
  ([ema_program_factory.cpp:48-49](ema/device/ema_program_factory.cpp#L48-L49)) — **not**
  `split_work_to_cores`. It returns `(num_cores, total_batch_channel_tiles_per_core)`; every core
  gets the *same* per-core count, so there is **no** core-group split and no per-group CTA.
- `all_cores = CoreRangeSet(grid_to_cores(num_cores, grid_size.x, grid_size.y, false))`
  ([:52](ema/device/ema_program_factory.cpp#L52)).
- Per-core RTA values are assigned by iterating `all_cores.ranges()` and each range's cores
  ([:182-189](ema/device/ema_program_factory.cpp#L182-L189)), advancing `src_start_tile` /
  `dst_start_tile` by `total_tiles_per_core` per core.

---

### Shared kernels

**none.**

- Nothing is *borrowed*: all six kernel sources live inside this op directory.
- Nothing is *lent*: `grep -rl` for each of the six kernel filenames across
  `ttnn/cpp/ttnn/operations/` finds no consumer outside this directory (confirmed independently of
  the audit).
- The one *intra-directory* sharing point is the header
  [device/kernels/accumulation_common.hpp](device/kernels/accumulation_common.hpp), included by all
  six kernels across **both** device operations. Because both factories convert in this same change,
  the shared-kernel fork rungs do not apply — the header is edited in place and every consumer is
  converted with it.
- No `_metal2` fork exists anywhere under `ttnn/cpp/ttnn/operations/reduction/`.

### Flags

- **Unreferenced kernel files:** none. All six kernel sources in the directory are bound by a factory.
- **Unused constants in the shared kernel header:** `FIRST_TILE` and `WORKING_REG`
  ([accumulation_common.hpp:8-9](device/kernels/accumulation_common.hpp#L8-L9)) are referenced by no
  kernel. Left untouched (audit anomaly 2; out of port scope).
- **Unused includes in the accumulation reader:** `<cstring>` and `api/core_local_mem.h`
  ([accumulation_reader.cpp:5,10](device/kernels/dataflow/accumulation_reader.cpp#L5-L10)). Left
  untouched (audit anomaly 3; out of port scope).
- **Dead RTA value:** reader/writer RTA slot 4 (`start_id`) is a dead *value* — it seeds a loop
  counter the body never reads (audit anomaly 1). **Ported as-is** as a named RTA; removing it is a
  functional change and out of scope.
- **Host/kernel name disagreement on EMA `c_2`:** the host calls it `prev_cb_index`, the kernel uses
  it as `trp_cb_idx` — a transpose round-trip scratchpad, not a "previous output" store (audit
  anomaly 8). The port preserves *both* existing names rather than picking a side: the host-side
  `DFBSpecName` keeps the host's word (`prev`), the kernel-side `accessor_name` keeps the kernel's
  word (`trp`).
- **Descriptor types used:** `KernelDescriptor`, `CBDescriptor`, `CBFormatDescriptor`,
  `ReaderConfigDescriptor`, `WriterConfigDescriptor`, `DataMovementConfigDescriptor`,
  `ComputeConfigDescriptor`. All are in the audit's scan scope; nothing unmapped.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` — for **both** factories.
  (In code on this branch the concept is spelled `ProgramSpecFactoryConcept`; see the port report's
  Friction section. Same concept: one `create_program_artifacts` returning
  `ttnn::device_operation::ProgramArtifacts`.)
- **Custom `compute_program_hash`**: none — already the default reflection-based hash on both device
  operations. Nothing to delete.
- **Pybind entry points to remove**: none. The three nanobind files
  (`cumsum/cumsum_nanobind.cpp`, `cumprod/cumprod_nanobind.cpp`, `ema/ema_nanobind.cpp`) expose only
  the public op functions; neither `create_descriptor` nor any other factory entry point is bound. A
  repo-wide grep confirms no caller of either `create_descriptor` outside its own factory, and no
  external caller of `AccumulationProgramFactory::calc_input_tile_offset`.
- **Implementation notes**:
  - Each factory's device-operation class needs exactly two mechanical edits: the
    `create_descriptor` declaration becomes `create_program_artifacts` with the
    `ProgramArtifacts` return type, and the `<tt-metalium/program_descriptors.hpp>` include is
    replaced by `"ttnn/metal_v2_artifacts.hpp"`. No other device-op-class member changes.
  - `calc_input_tile_offset` stays a `static` member of `AccumulationProgramFactory` (it is part of
    the factory, not the legacy concept surface).
  - The output tensor bound by each factory is always `tensor_return_value.mesh_tensor()`; the
    preallocated-output case is resolved upstream in `create_output_tensors`, so **no** optional-output
    `TensorParameter` is modelled.
  - Both factory `.cpp` files may land in the same unity-build translation unit, so all file-local
    spec-name constants are prefixed per factory (`ACCUM_*` / `EMA_*`) per
    [Pattern: Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).

## Planned Spec Shape

### Variant: `AccumulationProgramFactory`

- **KernelSpecs** (4, or 3 when `core_group_2` is empty) — 1:1 with the legacy `KernelDescriptor`s:
  - `ACCUM_READER` ("reader") — source `accumulation_reader.cpp`
  - `ACCUM_WRITER` ("writer") — source `accumulation_writer.cpp`
  - `ACCUM_COMPUTE_G1` ("compute_g1") — source `accumulation_compute.cpp`
  - `ACCUM_COMPUTE_G2` ("compute_g2") — same source, built only when `core_group_2` is non-empty
- **DataflowBufferSpecs** (3) — 1:1 with the legacy `CBDescriptor`s, no aliasing, no borrowed memory:
  - `ACCUM_SRC` ("src") — `entry_size = tile_size(input_dataformat)`, `num_entries = 4`,
    `data_format_metadata = input_dataformat`
  - `ACCUM_DST` ("dst") — `entry_size = tile_size(output_dataformat)`, `num_entries = 4`,
    `data_format_metadata = output_dataformat`
  - `ACCUM_ACC` ("acc") — `entry_size = tile_size(acc_dataformat)`, `num_entries = 1`,
    `data_format_metadata = acc_dataformat`
  - `tile_format_metadata` left unset on all three: the legacy `CBFormatDescriptor::tile` field was
    never set.
- **SemaphoreSpecs**: none — no legacy `SemaphoreDescriptor`.
- **TensorParameters** (2): `ACCUM_INPUT` ("input") from `input.tensor_spec()`, `ACCUM_OUTPUT`
  ("output") from `output.tensor_spec()`. Strict matching (no relaxation).
- **WorkUnitSpecs** (2, or 1 when `core_group_2` is empty):
  - `wu_g1` — kernels `{ACCUM_READER, ACCUM_WRITER, ACCUM_COMPUTE_G1}`, `target_nodes = core_group_1`
  - `wu_g2` — kernels `{ACCUM_READER, ACCUM_WRITER, ACCUM_COMPUTE_G2}`, `target_nodes = core_group_2`

  Reader and writer belong to both work units, so their derived node set is
  `core_group_1 ∪ core_group_2 = all_cores` — matching the legacy `core_ranges`. The two work units
  have disjoint `target_nodes` (required) and one compute kernel each (required).
- **Op-owned tensors**: none.

**DFB endpoint census** (re-derived from the kernel bodies, per node):

| DFB | touchers on a node | FIFO roles | disposition |
|---|---|---|---|
| `ACCUM_SRC` | reader ([accumulation_reader.cpp:43,46](device/kernels/dataflow/accumulation_reader.cpp#L43-L46)) + exactly one compute instance ([accumulation_compute.cpp:70,83](device/kernels/compute/accumulation_compute.cpp#L70-L83)) | reader locked PRODUCER (`reserve_back`/`push_back`), compute locked CONSUMER (`wait_front`/`pop_front`) | plain 1:1 |
| `ACCUM_DST` | one compute instance ([accumulation_compute.cpp:89,92](device/kernels/compute/accumulation_compute.cpp#L89-L92)) + writer ([accumulation_writer.cpp:40,44](device/kernels/dataflow/accumulation_writer.cpp#L40-L44)) | compute locked PRODUCER, writer locked CONSUMER | plain 1:1 |
| `ACCUM_ACC` | one compute instance only | locked **both** ways — `reserve_back`/`push_back` at [:37-38,57-63,94-101](device/kernels/compute/accumulation_compute.cpp#L37-L101) and `wait_front`/`pop_front` at [:43-44,67,81,106-107](device/kernels/compute/accumulation_compute.cpp#L43-L107) | **self-loop** — one toucher, so the compute `KernelSpec` binds it PRODUCER *and* CONSUMER under one accessor name |

This agrees with the brief. Two notes on why the two compute `KernelSpec`s do not disturb the
census: their node sets are **disjoint** (`core_group_1` / `core_group_2`), so each node hosts exactly
one compute instance; and binding two same-source `KernelSpec`s to one endpoint role over
non-overlapping node coverage is explicitly legal
([dataflow_buffer_spec.hpp:40-50](../../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp#L40-L50)).
This is the disjoint-node work-split, **not** the same-grid dual-instance shape — no 1P+1C
assignment question, no multi-binding flag.

Not a single kernel in the directory calls `get_write_ptr` / `get_read_ptr` /
`get_local_cb_interface` / `evil_set_*`, and there are no semaphores, so there is no hidden raw
co-filler anywhere: every DFB access in this op is a FIFO operation.

### Variant: `EmaProgramFactory`

- **KernelSpecs** (3) — 1:1 with legacy:
  - `EMA_READER` ("reader") — `ema_reader.cpp`
  - `EMA_WRITER` ("writer") — `ema_writer.cpp`
  - `EMA_COMPUTE` ("compute") — `ema_compute.cpp`
- **DataflowBufferSpecs** (3) — 1:1, no aliasing, no borrowed memory, `tile_format_metadata` unset:
  - `EMA_SRC` ("src") — `entry_size = src_tile_size`, `num_entries = ema_buffer_depth` (2),
    `data_format_metadata = src_data_format`
  - `EMA_DST` ("dst") — `entry_size = dst_tile_size`, `num_entries = ema_buffer_depth` (2),
    `data_format_metadata = dst_data_format`
  - `EMA_PREV` ("prev") — `entry_size = src_tile_size`, `num_entries = 1`,
    `data_format_metadata = src_data_format`
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `EMA_INPUT` ("input"), `EMA_OUTPUT` ("output"). Strict matching.
- **WorkUnitSpecs** (1): `wu` — kernels `{EMA_READER, EMA_WRITER, EMA_COMPUTE}`,
  `target_nodes = all_cores`.
- **Op-owned tensors**: none.

**DFB endpoint census** (re-derived, per node):

| DFB | touchers on a node | FIFO roles | disposition |
|---|---|---|---|
| `EMA_SRC` | reader ([ema_reader.cpp:42,45](ema/kernels/dataflow/ema_reader.cpp#L42-L45)) + compute ([ema_compute.cpp:102,107](ema/kernels/compute/ema_compute.cpp#L102-L107)) | reader locked PRODUCER, compute locked CONSUMER | plain 1:1 |
| `EMA_DST` | compute ([ema_compute.cpp:122,126](ema/kernels/compute/ema_compute.cpp#L122-L126)) + writer ([ema_writer.cpp:42,45](ema/kernels/dataflow/ema_writer.cpp#L42-L45)) | compute locked PRODUCER, writer locked CONSUMER | plain 1:1 |
| `EMA_PREV` | compute only | locked both ways — `reserve_back`/`push_back` at [ema_compute.cpp:109,113](ema/kernels/compute/ema_compute.cpp#L109-L113), `wait_front`/`pop_front` at [:116,120](ema/kernels/compute/ema_compute.cpp#L116-L120) | **self-loop** — one toucher; compute binds PRODUCER *and* CONSUMER under accessor name `trp` |

Agrees with the brief. Both self-loops in this port carry real FIFO traffic in both directions (the
accumulation one sequences the compute kernel's own unpacker against its own packer; the EMA one is a
packer→unpacker transpose round trip), so both bindings are functional, not cosmetic. Both are
**compute**-kernel self-loops, so neither records Quasar debt.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_1` (`core_group_1`) + `compute_desc_2` (`core_group_2`) of `accumulation_compute.cpp` ([accumulation_program_factory.cpp:187-217](device/accumulation_program_factory.cpp#L187-L217)) | `ACCUM_COMPUTE_G1`, `ACCUM_COMPUTE_G2` | `wu_g1`, `wu_g2` (disjoint node sets) | `ACCUM_SRC` (CONSUMER each), `ACCUM_DST` (PRODUCER each), `ACCUM_ACC` (PRODUCER **and** CONSUMER each — self-loop) |

The two legacy compute descriptors are byte-identical except for `core_ranges`: same source, same
CTAs, same defines, same `ComputeConfigDescriptor` (the per-core work count is a runtime arg, not a
per-group CTA). Merging them into one `KernelSpec` over `all_cores` would be behaviour-preserving,
but **the port keeps two `KernelSpec`s**, mirroring the legacy shape — the redundancy is recorded for
the ops team in the audit's Misc anomalies, not resolved here.

`EmaProgramFactory` has no work-split multiplicity: `get_max_cores_divisible_by_tiles_per_core_tiles`
gives every core the same per-core count, so there is one `KernelDescriptor` per kernel.

## Dropped Plumbing

### Variant: `AccumulationProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| [accumulation_program_factory.cpp:165-166](device/accumulation_program_factory.cpp#L165-L166) | `TensorAccessorArgs(input_tensor).append_to(reader_compile_time_args)` | `TensorParameter{ACCUM_INPUT}` + `TensorBinding{ACCUM_INPUT, "input"}` on the reader |
| [accumulation_reader.cpp:16](device/kernels/dataflow/accumulation_reader.cpp#L16) | `constexpr auto input_addrg_args = TensorAccessorArgs<0>();` | (line deleted) |
| [accumulation_reader.cpp:18,36](device/kernels/dataflow/accumulation_reader.cpp#L18-L36) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(input_addrg_args, input_base_addr)` | `TensorAccessor(tensor::input)` |
| [accumulation_program_factory.cpp:168-169](device/accumulation_program_factory.cpp#L168-L169) | `TensorAccessorArgs(output_tensor).append_to(writer_compile_time_args)` | `TensorParameter{ACCUM_OUTPUT}` + `TensorBinding{ACCUM_OUTPUT, "output"}` on the writer |
| [accumulation_writer.cpp:13](device/kernels/dataflow/accumulation_writer.cpp#L13) | `constexpr auto output_addrg_args = TensorAccessorArgs<0>();` | (line deleted) |
| [accumulation_writer.cpp:15,30](device/kernels/dataflow/accumulation_writer.cpp#L15-L30) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(output_addrg_args, output_base_addr)` | `TensorAccessor(tensor::output)` |
| [accumulation_program_factory.cpp:233,244](device/accumulation_program_factory.cpp#L233-L244) | `input_tensor` / `output_tensor` pushed as RTA slot 0 via `emplace_runtime_args` (framework `BufferBinding`) | `TensorArgument` in `ProgramRunArgs::tensor_args` |
| [accumulation_common.hpp:11-13](device/kernels/accumulation_common.hpp#L11-L13) | `constexpr uint32_t CB_IN = tt::CBIndex::c_0;` (and `CB_OUT`, `CB_ACC`) | `DFBBinding`s → `dfb::in`, `dfb::out`, `dfb::acc` |
| [accumulation_program_factory.cpp:105,111](device/accumulation_program_factory.cpp#L105-L111) | `cb_id = static_cast<uint32_t>(accumulation_cb)`, `.buffer_index = cb_id` | `DataflowBufferSpec::unique_id` (the `AccumulationCB` enum and the magic `c_0`/`c_1`/`c_2` indices are deleted) |
| [accumulation_program_factory.cpp:122-129](device/accumulation_program_factory.cpp#L122-L129) | `std::vector<UnpackToDestMode> unpack_to_dst(NUM_CIRCULAR_BUFFERS, …)` indexed by CB id | `ComputeGen1Config::unpack_modes`, a `Table<DFBSpecName, UnpackMode>` keyed by DFB name |
| [accumulation_program_factory.cpp:191,207](device/accumulation_program_factory.cpp#L191-L207) | positional CTA `{bit_cast<uint32_t>(default_acc_value)}` | named CTA `{"default_acc_value", …}` |
| [accumulation_reader.cpp:19-28](device/kernels/dataflow/accumulation_reader.cpp#L19-L28), [accumulation_writer.cpp:16-25](device/kernels/dataflow/accumulation_writer.cpp#L16-L25) | positional `get_arg_val<uint32_t>(1..7)` | named RTAs: `num_rows_per_core`, `tiles_per_row`, `input_tile_offset`, `start_id`, `low_rank_offset`, `high_rank_offset`, `flip` |
| [accumulation_compute.cpp:21,23-24](device/kernels/compute/accumulation_compute.cpp#L21-L24) | `get_compile_time_arg_val(0)`, `get_arg_val<uint32_t>(0..1)` | `get_arg(args::default_acc_value)`, `get_arg(args::num_rows)`, `get_arg(args::tiles_per_row)` |
| [accumulation_reader.cpp:33](device/kernels/dataflow/accumulation_reader.cpp#L33), [accumulation_writer.cpp:27](device/kernels/dataflow/accumulation_writer.cpp#L27) | `get_tile_size(CB_IN)` / `get_tile_size(CB_OUT)` (cb-id free helper) | `dfb_in_obj.get_tile_size()` / `dfb_out_obj.get_tile_size()` (whitelist rule 7) |

No page-size third-argument CTA/RTA (both accessors already use the two-argument form), no
semaphore-ID RTA, no `tensor.buffer()->address()` anywhere in the legacy factory.

### Variant: `EmaProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| [ema_program_factory.cpp:125](ema/device/ema_program_factory.cpp#L125) | `TensorAccessorArgs(input).append_to(reader_compile_args)` | `TensorParameter{EMA_INPUT}` + `TensorBinding{EMA_INPUT, "src"}` on the reader |
| [ema_reader.cpp:17](ema/kernels/dataflow/ema_reader.cpp#L17) | `constexpr auto src_args = TensorAccessorArgs<1>();` | (line deleted) |
| [ema_reader.cpp:21,34](ema/kernels/dataflow/ema_reader.cpp#L21-L34) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(src_args, src_base_addr)` | `TensorAccessor(tensor::src)` |
| [ema_program_factory.cpp:128](ema/device/ema_program_factory.cpp#L128) | `TensorAccessorArgs(output).append_to(writer_compile_args)` | `TensorParameter{EMA_OUTPUT}` + `TensorBinding{EMA_OUTPUT, "dst"}` on the writer |
| [ema_writer.cpp:17](ema/kernels/dataflow/ema_writer.cpp#L17) | `constexpr auto dst_args = TensorAccessorArgs<1>();` | (line deleted) |
| [ema_writer.cpp:21,34](ema/kernels/dataflow/ema_writer.cpp#L21-L34) | `get_arg_val<uint32_t>(0)` → `TensorAccessor(dst_args, dst_base_addr)` | `TensorAccessor(tensor::dst)` |
| [ema_program_factory.cpp:184-185](ema/device/ema_program_factory.cpp#L184-L185) | `input` / `output` pushed as RTA slot 0 via `emplace_runtime_args` | `TensorArgument` in `ProgramRunArgs::tensor_args` |
| [ema_program_factory.cpp:78-80](ema/device/ema_program_factory.cpp#L78-L80) | `constexpr auto src_cb_index = tt::CBIndex::c_0;` (and `dst_cb_index`, `prev_cb_index`) | `DataflowBufferSpec::unique_id`s (`EMA_SRC` / `EMA_DST` / `EMA_PREV`) |
| [ema_reader.cpp:26](ema/kernels/dataflow/ema_reader.cpp#L26), [ema_writer.cpp:26](ema/kernels/dataflow/ema_writer.cpp#L26), [ema_compute.cpp:78-80](ema/kernels/compute/ema_compute.cpp#L78-L80) | kernel-side `constexpr auto src_cb_idx = tt::CBIndex::c_0;` etc. | `dfb::src`, `dfb::dst`, `dfb::trp` |
| [ema_program_factory.cpp:124,127](ema/device/ema_program_factory.cpp#L124-L127) | positional CTA `{total_tiles_per_core}` | named CTA `{"total_tiles_per_core", …}` |
| [ema_program_factory.cpp:130-135](ema/device/ema_program_factory.cpp#L130-L135) | positional CTAs `{total_batch_channel_tiles_per_core, tiles_per_channel, alpha_bits, beta_bits}` | named CTAs `total_batches_per_core`, `tiles_per_channel`, `alpha_bits`, `beta_bits` (names taken from the kernel's own locals) |
| [ema_reader.cpp:22](ema/kernels/dataflow/ema_reader.cpp#L22), [ema_writer.cpp:22](ema/kernels/dataflow/ema_writer.cpp#L22) | `get_arg_val<uint32_t>(1)` | named RTAs `src_start_tile` / `dst_start_tile` |
| [ema_reader.cpp:30](ema/kernels/dataflow/ema_reader.cpp#L30), [ema_writer.cpp:30](ema/kernels/dataflow/ema_writer.cpp#L30) | `get_tile_size(src_cb_idx)` / `get_tile_size(dst_cb_idx)` | `dfb_src.get_tile_size()` / `dfb_dst.get_tile_size()` (whitelist rule 7) |

No page-size third argument, no semaphore-ID RTA, no `tensor.buffer()->address()`.

## Hardware configuration notes

| kernel | legacy resolved values | Metal 2.0 |
|---|---|---|
| accumulation reader | `ReaderConfigDescriptor{}` → `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` = the reader default | `create_reader_datamovement_config(device->arch())` |
| accumulation writer | `WriterConfigDescriptor{}` → `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` = the writer default | `create_writer_datamovement_config(device->arch())` |
| accumulation compute ×2 | Style **B** — Metal `ComputeConfigDescriptor` set directly: `math_fidelity = default_math_fidelity`, `fp32_dest_acc_en = true`, `dst_full_sync_en = false`, `math_approx_mode = false`, `unpack_to_dest_mode = unpack_to_dst`; `bfp8_pack_precise` left default | `ComputeGen1Config{fpu_math_fidelity = default_math_fidelity, sfpu_precision_mode = Precision::Precise, enable_32_bit_dest = true, double_buffer_dest = true, unpack_modes = …}`; `bfp_pack_precision_mode` left default |
| EMA reader | `(RISCV_0, NOC_0, DM_DEDICATED_NOC)` — **custom**, matches neither default | `DataMovementGen1Config{.processor = RISCV_0, .noc = reader_noc, .noc_mode = DM_DEDICATED_NOC}` |
| EMA writer | `(RISCV_1, NOC_1, DM_DEDICATED_NOC)` — **custom**, matches neither default | `DataMovementGen1Config{.processor = RISCV_1, .noc = writer_noc, .noc_mode = DM_DEDICATED_NOC}` |
| EMA compute | Style **A** — resolved from a TTNN `ComputeKernelConfig` via `get_compute_kernel_config_args` | `to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config)`; no `unpack_modes` entries (see below) |

**Compiler optimization level — carried over explicitly on every compute kernel.** Legacy
`KernelDescriptor::opt_level` is a `std::optional` resolved per kernel kind by the descriptor→program
path: `O2` for data movement, **`O3` for compute**
([program.cpp:439,455](../../../../../../tt_metal/impl/program/program.cpp#L439-L455)). Metal 2.0's
`KernelSpec::CompilerOptions::opt_level` instead defaults to `O2` for *both* kinds, and the lowering
passes it straight through. So all three compute `KernelSpec`s set
`.compiler_options.opt_level = KernelBuildOptLevel::O3` explicitly; the data-movement kernels need no
action, since `O2` already matches. Without this the port would silently drop compute-kernel
optimization from `O3` to `O2` — no build error, no test signal. Routed to the port report as a
handoff point.

`double_buffer_dest = !dst_full_sync_en` — the inverted knob. For accumulation
`dst_full_sync_en = false` → `double_buffer_dest = true`. `math_approx_mode = false` →
`Precision::Precise`. Both happen to coincide with the `ComputeGen1Config` defaults; the port sets
them explicitly because the legacy descriptor set them explicitly.

The two EMA DM kernels are on distinct RISC cores and distinct NOCs under
`DM_DEDICATED_NOC`, so the Gen1 node invariants the spec validator enforces are satisfied. Note the
processors are the *reverse* of the conventional assignment (reader on RISCV_0, writer on RISCV_1);
that is the legacy op's own choice and is reproduced verbatim, not normalized.

**`unpack_modes` translation (accumulation compute).** The legacy
`std::vector<UnpackToDestMode>` was `Default` everywhere except:

- index `ACC` (`c_2`) — always `UnpackToDestFp32` → `{ACCUM_ACC, UnpackMode::UnpackToDest}`
- index `SRC` (`c_0`) — `UnpackToDestFp32` **iff** `input_dataformat != DataFormat::Float16_b` →
  `{ACCUM_SRC, UnpackMode::UnpackToDest}` under the same condition
- index `DST` (`c_1`) — `Default` → **entry omitted** (omission *is* `UnpackToSrc`), and the compute
  kernel only produces into `ACCUM_DST` anyway

`enable_32_bit_dest = true` on this kernel, so both `UnpackToDest` entries are unconditionally
accepted by the validator; and the newly-required-entry rule (a consumed Float32 DFB with
`enable_32_bit_dest = true`) is satisfied for `ACCUM_ACC` in the non-integer path and for `ACCUM_SRC`
when the input is FLOAT32 — both get an explicit entry.

**`unpack_modes` (EMA compute): none.** The legacy `ComputeConfigDescriptor` left
`unpack_to_dest_mode` empty, i.e. `Default` for every CB. EMA validation pins both input and output
to `BFLOAT16` ([ema_device_operation.cpp:20-21,39-42](ema/device/ema_device_operation.cpp#L20-L42)),
so no DFB it consumes can be Float32 and the required-entry rule cannot fire regardless of the
user's `fp32_dest_acc_en`.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  `ACCUM_ACC` on both accumulation compute `KernelSpec`s (PRODUCER **and** CONSUMER, one shared
  accessor name `acc`), and `EMA_PREV` on the EMA compute `KernelSpec` (accessor name `trp`). Both
  are the genuine accumulator / round-trip case — real FIFO traffic in both directions — not the
  sync-free resolution.
- [Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  (**avoided**): the accumulation compute work split stays two `KernelSpec`s in two `WorkUnitSpec`s
  over disjoint node sets. Nothing is demoted; the per-core count was already an RTA in legacy.
- [Pass DFB handles directly to LLKs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `dfb::in` / `dfb::out` / `dfb::acc` passed straight into `unary_op_init_common`,
  `reconfig_data_format`, `pack_reconfig_data_format`, `copy_tile_to_dst_init_short`, `copy_tile`,
  `pack_tile`; `dfb::src` / `dfb::dst` / `dfb::trp` into `compute_kernel_hw_startup`,
  `transpose_init`, `transpose_tile`, `pack_tile`. No `.id` extraction, no temporary wrappers.
- [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  file-local spec-name constants are prefixed `ACCUM_*` / `EMA_*` so the two factory `.cpp` files can
  share a unity-build translation unit.

Not applied, and why: **Conditional / optional DFB bindings** — no DFB, tensor, or semaphore is
conditionally used in either factory (the accumulation `default_acc_value` / `defines` variation
selects an *operation*, not a binding). **Aliased DFBs** — every legacy `CBDescriptor` has exactly
one `CBFormatDescriptor`. **Same-FIFO aliasing** — no CB index is mirrored onto another, on either
side. **Multi-variant factories** — one spec shape per factory. **Two-toucher 1P+1C** and the
**multi-binding advanced option** — no DFB has more than two touchers or two same-role touchers.
**Porting a shared kernel** — nothing borrowed or lent; the one shared header's consumers all convert
in this change. **Removing pybound legacy factory entry points** — no factory entry point is pybound.

## Deferred / Flagged

- **New finding (naming drift, docs vs code).** The recipe and the TTNN integration doc name the
  target concept `MetalV2FactoryConcept`; on this branch
  [`ttnn/api/ttnn/operation_concepts.hpp:119`](../../../../../api/ttnn/operation_concepts.hpp#L119)
  spells it `ProgramSpecFactoryConcept` (with a `CustomProgramSpecFactoryConcept` sibling for
  factories that also define a spec-runtime-args override). No effect on the port — the factory
  satisfies the concept structurally by declaring `create_program_artifacts` — but the doc name is
  not greppable. Routed to the port report's Friction section.
- **New finding (`get_tile_size()` cannot stay `constexpr`).** The brief states that
  `DataflowBuffer::get_tile_size()` being `constexpr` lets the two EMA sites keep binding the result
  to a `constexpr uint32_t`. It does not: `DataflowBuffer`'s constructor is declared out-of-line
  and is not `constexpr` ([dataflow_buffer.h:72-75](../../../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L72-L75)),
  so a member call on a non-`constexpr` object is not a constant expression. The two EMA
  declarations change from `constexpr uint32_t` to `const uint32_t`. The value is used only as a NoC
  transfer byte count, so this is behaviour-neutral (and the getter still folds at `-O2`). Routed to
  the port report's Friction section.
- **New finding during construction (`opt_level` default divergence).** Metal 2.0's
  `CompilerOptions::opt_level` defaults to `O2` for compute kernels where legacy resolved `O3`. The
  port sets it explicitly (see [Hardware configuration notes](#hardware-configuration-notes)); the
  framework-side fix and the recipe amendment it calls for are the port report's first handoff point.
  Worth noting here because the field sits **outside** `hw_config`, so the recipe's diff-the-config
  discipline does not direct a porter to check it.
- Nothing else surfaced. No feature gate fired that the audit's Appendix A does not cover, and no
  construct required a legacy workaround.
