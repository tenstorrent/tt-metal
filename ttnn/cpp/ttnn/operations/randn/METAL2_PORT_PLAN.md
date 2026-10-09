# Port Plan — randn

Port plan for `randn` (`RandnDeviceOperation::ProgramFactory`), ported from the `ProgramDescriptor` API (`create_descriptor` + PD-style `override_runtime_arguments`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- **Concept:** `ProgramDescriptorFactoryConcept` with an `override_runtime_arguments(Program&, …)`. The methods live on a conventional nested `RandnDeviceOperation::ProgramFactory`, with `program_factory_t = std::variant<ProgramFactory>` (`device/randn_device_operation.hpp:30-46`). This is not the direct-descriptor shape, so exception 3 does not apply.
- **Variants:** single.
- **Custom `compute_program_hash`:** present at `device/randn_device_operation.cpp:47-52`. It hashes all attributes except `seed`. Left intact.
- **Inputs:** none. `tensor_args_t {}`; the only tensor is the output.

### Kernels
Both kernels run on `core_ranges = all_cores` from `split_work_to_cores(grid, tiles)`.

| unique_id | source | CTAs (positional) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| writer | `device/kernels/writer_standard_normal.cpp` | `dst_cb_id` (= `c_0`), then the `TensorAccessorArgs(output)` block at offset 1 | none | `output Buffer*`, `start_id`, `num_tiles` | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` (writer default: RISCV_0 / NOC_1 / dedicated) |
| compute | `device/kernels/compute_standard_normal.cpp` | `dst_cb_id` | none | `seed`, `num_tiles` | none | `OUTPUT_DTYPE_BFLOAT16=1` when the output is BF16 | O3 (unset, compute) | see below |

The compute config is built from `get_compute_kernel_config_args(arch, attrs.compute_kernel_config)`:

| field | value |
|---|---|
| `math_fidelity` | from the config |
| `fp32_dest_acc_en` | **forced `true`** |
| `dst_full_sync_en` | from the config |
| `math_approx_mode` | from the config |
| `packer_l1_acc` | resolved but unused (no Metal 2.0 counterpart) |
| `bfp8_pack_precise`, `unpack_to_dest_mode` | default |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| 0 (`c_0`) | 2 × tile_size(out fmt) | all_cores | output dtype's format | tile_size(out fmt) | unset |

Not borrowed, no aliasing, no GlobalCB.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `randn_program_factory.cpp:80` (CTA block), `:135` (RTA) | `output` | writer 0 |

The kernel side is `TensorAccessor(dst_args, dst_addr)`, with no 3rd argument.

### Work split
- **Driver:** `split_work_to_cores(grid, units_to_divide = output.physical_volume() / TILE_HW)`.
- **Cores:** `num_cores = min(tiles, grid.x × grid.y)`, walked column-major via `randn_core(i, grid)`. This mirror is guarded by a `TT_FATAL` against `split_work_to_cores`.
- **Groups:** `core_group_1` / `core_group_2` differ only in `units_per_core`, which is already an RTA. So no per-group CTA and no multiplicity is needed.

### Legacy `override_runtime_arguments` (`randn_program_factory.cpp:146-172`)
On every cache hit, for cores `i = 0..num_cores-1` in `randn_core` order:
1. **compute RTA 0 (`seed`)** ← one draw from `seeded_rng(seed)` if `seed` is set, else from the persistent file-scope `default_rng`. The draw order is identical to `create_descriptor`.
2. **writer RTA 0** ← `output.buffer()->address()`.

Nothing else is refreshed: `start_id` and `num_tiles` stay as built.

### Shared kernels
none. Both kernels are bound only by this factory (`grep -rl` for each kernel filename).

### Flags
- `writer_kernel_idx` / `compute_kernel_idx` exist only so the legacy override can index the cached `Program`. They have no Metal 2.0 role.

## TTNN ProgramFactory

- **Concept (inherited from audit):** `CustomProgramSpecFactoryConcept`
- **Custom `compute_program_hash`:** present at `device/randn_device_operation.cpp:47-52`. Leave intact.
- **Implementation notes:** `create_descriptor` → `create_program_artifacts`. `override_runtime_arguments` becomes the declarative form, returning `ProgramRunArgs` and taking no `Program&`. The existing `ProgramFactory` struct is kept, so no device-op-class edits beyond the method signatures. Spec names (`writer`, `compute`, `dst`, `output`) are shared file-scope constants, replacing the kernel-index constants.

## Planned Spec Shape

- **KernelSpecs:** `writer` and `compute` (same sources, in place).
- **DataflowBufferSpecs:** `dst`, with `entry_size = tile_size(out fmt)`, `num_entries = 2`, `data_format_metadata = out fmt`.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `output` (`output.mesh_tensor().tensor_spec()`).
- **WorkUnitSpecs:** one, `{writer, compute}` on `all_cores`.

## Preserved Multiplicity

none. There is no work-split multiplicity in legacy: the two core groups share one kernel each and differ only in an RTA.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| writer RTA 0 (`:135`) | `output.buffer()` (`Buffer*`) | `TensorBinding` `tensor::output` |
| writer CTA 1.. (`:80`) + kernel `TensorAccessorArgs<1>()` (writer `:16`) | `TensorAccessorArgs` plumbing | binding mechanism |
| writer and compute CTA 0 | `dst_cb_id = c_0` | `DFBBinding` `dfb::dst` (writer CONSUMER, compute PRODUCER) |
| writer RTAs 1–2 | positional `start_id`, `num_tiles` | named RTAs `start_id`, `num_tiles` |
| compute RTAs 0–1 | positional `seed`, `num_tiles` | named RTAs `seed`, `num_tiles` |
| override: `GetRuntimeArgs(program, writer_kernel_idx, core)[0] = output_addr` | raw address patch | `params.tensor_args = {{OUTPUT, output.mesh_tensor()}}` |
| override: `GetRuntimeArgs(program, compute_kernel_idx, core)[0] = seed` | positional patch | `KernelRunArgs{COMPUTE}.runtime_arg_values["seed"][core]` (partial update) |
| compute `cb_dst.get_cb_id()` / `dst_cb_id` at LLK calls | CB id | `dfb::dst` passed directly |

## Applied Patterns

- **[Translating `override_runtime_arguments`](metal2_port.md#translating-override_runtime_arguments-custom-concept-only):** step 1's inventory is the list above. Every refresh maps to an entry: the address becomes a `TensorArgument`, and the per-node `seed` becomes a `runtime_arg_values` entry. Nothing is added (no `num_tiles` or `start_id` refresh). The guard does not live in the override, so no TT_FATAL moves.
- **[Pass DFB handles directly to LLKs](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):** `pack_reconfig_data_format`, `pack_tile`, `compute_kernel_hw_startup`, and `copy_init` take `dfb::dst`.
- **Whitelist §A:** the writer's `dst_tile_bytes` was a non-`constexpr` `cb_dst.get_tile_size()`, so it stays a member getter (`dfb_dst.get_tile_size()`).
- **Hardware config:**
  - Writer: the default writer triple, so `ttnn::create_writer_datamovement_config()`.
  - Compute (Style A): `ttnn::to_compute_hardware_config(attrs.compute_kernel_config)`, then `enable_32_bit_dest = true` to keep legacy's force. The helper's other fields are exactly the legacy-copied subset (fidelity, approx → precision, `!dst_full_sync_en`). `packer_l1_acc` has no counterpart.
  - No `unpack_modes`: compute consumes no DFB.
  - Explicit O3 on compute.

## Deferred / Flagged

- none
