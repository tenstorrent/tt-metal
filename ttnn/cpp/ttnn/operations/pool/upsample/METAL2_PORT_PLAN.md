# Port Plan — upsample `UpsampleBilinearProgramFactory`

Port plan for `ttnn/cpp/ttnn/operations/pool/upsample` factory `UpsampleBilinearProgramFactory`,
ported from the legacy `descriptor` API (`create_descriptor` → `ProgramDescriptor`) to Metal 2.0
(`create_program_artifacts` → `ProgramArtifacts`, `ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Scope: **`UpsampleBilinearProgramFactory` only**. The three sibling factories
(`UpsampleMultiCoreInterleaved`, `UpsampleMultiCoreSharded`, `UpsampleNearestFloat`) are already
Metal 2.0 and are untouched.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` returns `ProgramDescriptor`,
  `upsample_bilinear_program_factory_multicore.cpp:30`). Method lives on the `UpsampleBilinearProgramFactory`
  struct, which is one alternative of the `UpsampleOperation::program_factory_t` variant.
- Variants: single (this factory is one path; the op's other 3 factories are out of scope).
- Custom `compute_program_hash`: none — default reflection-based hash. (Confirmed in the device-op.)

Target concept (inherited from audit): `ProgramSpecFactoryConcept`.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named→dfb) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_bilinear_multi_core_sharded.cpp` | all_cores (=input shard grid) | [0]stick_nbytes [1]scale_h [2]scale_w [3]in_w [4]out_w [5]in_h [6]halo_cb [7]tilize_reduce_cb_0 [8]in_scalar_cb_id1 [9]scale_h_inv [10]scale_w_inv [11]y_start [12]x_start [13]is_reader(=1) [14]blocks [15]input_block_size_bytes | 6,7,8 are CB ids → DFB tokens | {start_output_idx, min_input_offset, out_sticks_this_core} | none | none | O2 (DM default) | `ReaderConfigDescriptor{}` |
| writer | same source as reader | all_cores | same as reader except [7]=tilize_reduce_cb_1 [8]=in_scalar_cb_id2 [13]is_reader=0 | 6,7,8 → DFB tokens | {start_output_idx, min_input_offset, out_sticks_this_core} | none | none | O2 (DM default) | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/bilinear.cpp` | all_cores | [0]tilize_reduce_cb_0 [1]tilize_reduce_cb_1 [2]in_scalar_cb_id1 [3]in_scalar_cb_id2 [4]out_cb [5]in_ntiles_c [6]in_ntiles_hwc [7]window_size_hw [8]out_ntiles_c [9]blocks [10]input_block_size_bytes | 0–4 are CB ids → DFB tokens | {nsticks_per_core} | none | reduce_op SUM / dim H (`reduce_op_utils::get_defines`) | **O3** (compute default, must set explicitly) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en=false, math_approx_mode}` |

Reader and writer share **one** source file (`reader_bilinear_multi_core_sharded.cpp`), branched on the
`is_reader` CTA (dual-instance work-split over the same grid). Both are DM kernels.

### CBs
| index | name | total_size | page_size (entry_size) | num_entries | data_format | face_geometry | borrowed |
|---|---|---|---|---|---|---|---|
| c_0 | halo_cb | in_cb_pagesize*in_cb_npages | in_cb_pagesize=input_stick_nbytes | in_cb_npages=halo_shard_shape[0] | input_cb_data_format | — | `.buffer = halo_in.buffer()` (input) |
| c_1 | tilize_reduce_cb_0 | in1_cb_pagesize*4*2 | in1_cb_pagesize | 8 (=4*buffering_factor) | input_cb_data_format | {face_r_dim=4, num_faces=2} | no |
| c_2 | tilize_reduce_cb_1 | in_cb_pagesize*4*2 | in_cb_pagesize | 8 | input_cb_data_format | {face_r_dim=4, num_faces=2} | no |
| c_3 | in_scalar_cb_id1 | in_scalar_cb_pagesize*2 | tile_size(input_df) | 2 (=1*buffering_factor) | input_cb_data_format | — | no |
| c_4 | in_scalar_cb_id2 | in_scalar_cb_pagesize*2 | tile_size(input_df) | 2 | input_cb_data_format | — | no |
| c_5 | out_cb | out_cb_pagesize*out_cb_npages | out_cb_pagesize=TILE_WIDTH*out.element_size() | out_cb_npages=out_shard[0]*in_ntiles_c | output_cb_data_format | {face_r_dim=1, num_faces=2} | `.buffer = output.buffer()` (output) |

No `GlobalCircularBuffer`, no `address_offset`. `buffering_factor = 2`.

### Semaphores
none.

### Tensor accessors
none in the bilinear kernels (borrowed-memory CBs carry input/output; raw L1 offset arithmetic). The
input/output tensors surface only as borrowed-CB backing buffers, not via `TensorAccessor`.

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `c_0` `.buffer = halo_in.buffer()` | input (halo) | — (no address RTA) |
| `c_5` `.buffer = output.buffer()` | output | — (no address RTA) |

### Work split
- Driver: manual per-core loop over `logical_cores = corerange_to_cores(shard_spec.grid, ...)`.
- Distribution: `max_out_sticks_per_core = div_up(total_output_sticks, ncores_nhw)`; each core gets
  `min(max_out_sticks_per_core, remaining)` output sticks; trailing cores get 0 (no-work path with
  `{start_output_idx, 0, 0}` / `{0}`).
- num_cores = ncores = shard_spec.num_cores(); all on one grid (all_cores).

### Shared kernels
none — both bilinear kernels are op-owned and bound by no other op (audit + `grep -rl` confirms). Header
helpers (`experimental_device_api.hpp`, `fixed_point_arithmetic.hpp`, `bilinear_weights_lut.hpp`) are
in-family / local headers, not file-path kernel instantiations; Device 2.0 native, unchanged.

### Flags
- Dead compute CTAs `in_ntiles_hwc` [6] and `window_size_hw` [7] (and local `num_output_tiles`) are unused
  in `bilinear.cpp` — carried across unchanged (named CTAs), pruning is an ops-team cleanup (report).
- `MAX_TILES_PER_REDUCTION = 8` duplicated literal in factory + compute kernel — left as-is (report).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (`Override runtime args method? = no`;
  no op-owned tensors — the framework refreshes tensor bindings on cache hit; one `create_program_artifacts`).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: forces the device-op-class **exception 1** edit — the struct declares
  `create_descriptor` (which vanishes) and the header must switch to `create_program_artifacts`. No pybind
  hook exists for `create_descriptor` (nanobind binds only the top-level `upsample` function), so no pybind
  deletion is required. Header includes `<tt-metalium/global_circular_buffer.hpp>` (presence-only, unused
  by this factory) — left untouched (out-of-scope, shared header).

## Planned Spec Shape

- KernelSpecs (3): READER, WRITER (same source, dual-instance work-split → preserved multiplicity),
  COMPUTE.
- DataflowBufferSpecs (6): HALO (c_0, borrowed input), TILIZE_REDUCE_0 (c_1), TILIZE_REDUCE_1 (c_2),
  IN_SCALAR_0 (c_3), IN_SCALAR_1 (c_4), OUT (c_5, borrowed output, self-loop).
- SemaphoreSpecs: none.
- TensorParameters (2): INPUT (backs HALO), OUTPUT (backs OUT).
- WorkUnitSpecs (1): {READER, WRITER, COMPUTE} over all_cores.

### DFB binding map (endpoint role per KernelSpec)
| DFB (cb) | reader | writer | compute | disposition |
|---|---|---|---|---|
| HALO (c_0), borrowed INPUT | PRODUCER `halo` | CONSUMER `halo` | — | 1P+1C (two role-free raw-readers, cosmetic on Gen1) |
| TILIZE_REDUCE_0 (c_1) | PRODUCER `tilize_reduce` | — | CONSUMER `tilize_reduce_0` | legal 1:1 |
| TILIZE_REDUCE_1 (c_2) | — | PRODUCER `tilize_reduce` | CONSUMER `tilize_reduce_1` | legal 1:1 |
| IN_SCALAR_0 (c_3) | PRODUCER `in_scalar` | — | CONSUMER `in_scalar_1` | legal 1:1 |
| IN_SCALAR_1 (c_4) | — | PRODUCER `in_scalar` | CONSUMER `in_scalar_2` | legal 1:1 |
| OUT (c_5), borrowed OUTPUT | — | — | PRODUCER `out` + CONSUMER `out` | self-loop (sole toucher: compute) |

Reader and writer bind DIFFERENT specs behind the SAME accessor names (`tilize_reduce`, `in_scalar`) —
the same shared source reads `dfb::tilize_reduce` / `dfb::in_scalar`, the KernelSpec routes each instance
to its own DFB (c_1/c_3 for reader, c_2/c_4 for writer). This is the accessor-name-vs-spec-name split
(cf. clone `writer_dfb`).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| reader_desc, writer_desc (both `reader_bilinear_multi_core_sharded.cpp`) | READER, WRITER | main (both, disjoint roles per DFB) | HALO (reader P / writer C), TILIZE_REDUCE_0 (reader P), TILIZE_REDUCE_1 (writer P), IN_SCALAR_0 (reader P), IN_SCALAR_1 (writer P) — no `allow_instance_multi_binding` |

Reader/writer are two KernelSpecs of one source over the **same** grid, but each binds a **distinct** DFB
for the tilize_reduce / in_scalar buffers, and disjoint P/C roles on the shared HALO — no multi-binding.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer CTA [6] halo_cb_id | CB index in CTA | `DFBBinding` HALO, accessor `halo` |
| reader CTA [7] / writer CTA [7] | CB index (tilize_reduce_cb_0 / _1) | `DFBBinding` TILIZE_REDUCE_0/_1, accessor `tilize_reduce` |
| reader CTA [8] / writer CTA [8] | CB index (in_scalar_cb_id1 / 2) | `DFBBinding` IN_SCALAR_0/_1, accessor `in_scalar` |
| compute CTA [0..4] | CB indices | `DFBBinding` TILIZE_REDUCE_0/_1, IN_SCALAR_0/_1, OUT |
| all reader/writer/compute positional CTAs | positional `compile_time_args` | named CTAs (`{name, value}`) |
| reader/writer/compute positional RTAs | positional `runtime_args` | `runtime_arg_schema` + `AddRuntimeArgsForNode` |
| CB `.buffer = halo_in.buffer()` / `output.buffer()` | borrowed-memory CB | DFB `borrowed_from = INPUT / OUTPUT` + `TensorParameter` |

No buffer-address RTA, no `TensorAccessorArgs`, no page-size 3rd arg, no semaphore-ID RTA.

## Applied Patterns

- [Self-loop DFB binding](../shared/port_patterns.md): OUT on compute KernelSpec (PRODUCER + CONSUMER,
  shared accessor `out`) — single toucher (packer + raw `fifo_wr_ptr` advance).
- [Two-toucher DFB → assign 1P+1C](../shared/port_patterns.md): HALO — reader PRODUCER, writer CONSUMER
  (both raw-read; cosmetic on Gen1). Borrowed-memory DFB.
- [Preserved multiplicity](../shared/port_patterns.md): reader+writer dual-instance work-split → two
  KernelSpecs of one source.
- Accessor-name-vs-spec-name split: reader/writer bind different DFB specs behind the same accessor names.

## Deferred / Flagged

- Compute config: legacy resolves `dst_full_sync_en` via `get_compute_kernel_config_args` but **drops** it
  (never set on `ComputeConfigDescriptor`), so the descriptor default (false) → `double_buffer_dest = true`.
  Force `double_buffer_dest = true` after `to_compute_hardware_config`. `fp32_dest_acc_en` is asserted
  false (`enable_32_bit_dest = false`) so no `unpack_modes` entry is required. `packer_l1_acc` dropped —
  no Metal 2.0 counterpart, no action.
- No structural surprises beyond the audit.
