# Port Plan — `sharded_to_interleaved`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved`, ported from
`ProgramDescriptorFactoryConcept` (`create_descriptor` → `tt::tt_metal::ProgramDescriptor`) to Metal 2.0
(`ProgramSpecFactoryConcept` → `ttnn::device_operation::ProgramArtifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Code baseline:** `origin/main` @ `2b7bf3396eb` — byte-identical to the baseline the audit recorded, so
no rebase was needed. This base already carries both gate-clearing merges the brief calls out
(`0fb47949a27` Device 2.0 on `eltwise_copy.cpp`; `6abdf94214d` / PR #51747 offset base pointer on the RM writer).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `ShardedToInterleavedProgramFactory::create_descriptor()`
  returns `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_program_factory.hpp:15`).
- Variants: **single** factory. Not multi-variant in the `program_factory_t` sense — but it selects its
  kernel *sources* at runtime on two axes (input layout; dtype-conversion need), giving three reachable
  configs. All three convert together (atomic unit = one factory + every source it can bind):
  - **C1** — `TILE`, `!convert_df`: reader + tiled writer.
  - **C2** — `TILE`, `convert_df`: reader + tiled writer + compute.
  - **C3** — `ROW_MAJOR` (never converts — a dtype mismatch requires TILE per
    `sharded_to_interleaved_device_operation.cpp` `validate_inputs:67-71`): reader + RM writer.
- Custom `compute_program_hash`: **none** — already the default reflection-based hash, so the port leaves
  the cache key untouched.
  (Also confirmed absent: `get_dynamic_runtime_args`, `override_runtime_arguments`, pybind
  `create_descriptor`. No device-op-class edit is forced by this port.)

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory
analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels

All `core_ranges` are `used_cores` (`program_factory.cpp:122-124`) — `all_cores` from the input shard spec,
narrowed to `num_cores_unpadded` when the shard grid is larger than the data.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (`:167`) | `used_cores` | `[0] = src0_cb_index` (`:168`) | none | per core: `[0] = num_units_per_shard` (`:201`) | none | none | absent → resolves **O2** (DM) | `ReaderConfigDescriptor{}` (`:166`) |
| writer (C1/C2, TILE) | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (`:178-180`) | `used_cores` | `[0] = out_cb_index`, then `TensorAccessorArgs(*dst_buffer)` appended (`:175-176`) | none | per core, 9 slots (`:241-251`) — see below | none | none | absent → resolves **O2** (DM) | `WriterConfigDescriptor{}` (`:174`) |
| writer (C3, ROW_MAJOR) | `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` (`:182-184`) | `used_cores` | same as tiled (`:175-176`) | none | per core, 7 slots (`:292-300`) — see below | none | none | absent → resolves **O2** (DM) | `WriterConfigDescriptor{}` (`:174`) |
| compute (C2 only) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (`:191`) | `used_cores` | `[0] = num_units_per_shard` (`:195`) | none | none | none | none | absent → resolves **O3** (compute) | `ComputeConfigDescriptor{}` (`:194`) — every field left at its default |

`ComputeConfigDescriptor{}` resolved values (`program_descriptors.hpp:99-108`): `math_fidelity=HiFi4`,
`fp32_dest_acc_en=false`, `dst_full_sync_en=false`, `unpack_to_dest_mode={}`, `bfp8_pack_precise=false`,
`math_approx_mode=false`.

**Tiled writer RTA slots** (`program_factory.cpp:241-251` → kernel `:11-19`):

| slot | host value | kernel local |
|---|---|---|
| 0 | `dst_buffer` (a `Buffer*` binding) | `dst_addr` |
| 1 | `num_units_per_shard_height` | `block_height_tiles` |
| 2 | `num_units_per_shard_width` | `block_width_tiles` |
| 3 | `shard_height` | `unpadded_block_height_tiles` |
| 4 | `shard_width` | `unpadded_block_width_tiles` |
| 5 | `num_units_offset` | `output_width_tiles` |
| 6 | `num_units_per_shard` | `block_num_tiles` |
| 7 | `curr_idx_h + curr_idx_w` | `start_id_offset` |
| 8 | `starting_idx_h` | `start_id_base` |

`start_id = start_id_base + start_id_offset` (kernel `:20`) is a kernel-local sum of slots 8 and 7 — not a
ninth arg.

**RM writer RTA slots** (`program_factory.cpp:292-300` → kernel `:12-17`):

| slot | host value | kernel local |
|---|---|---|
| 0 | `dst_buffer` (a `Buffer*` binding) | `dst_addr` |
| 1 | `num_units_per_row` (`:294`) | **never read** — dead slot |
| 2 | `shard_height` | `block_height` |
| 3 | `shard_width` | `block_width_bytes` |
| 4 | `padded_shard_width` | `padded_block_width_bytes` |
| 5 | `curr_idx_w` | `input_width_offset_bytes` |
| 6 | `curr_idx_h` | `start_id` |

### CBs

Both built through the local `push_s2i_cb_pair` helper (`:25-43`); one `CBFormatDescriptor` each (no
aliasing in the legacy multi-`format_descriptors` sense), no `tile` field set, no `GlobalCircularBuffer`.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| `c_0` (`src0_cb_index`, `:128`) | `num_input_units * input_page_size` (`:144`) | `used_cores` | `input_cb_data_format` | `input_page_size = align(input_unit_size, src_buffer->alignment())` (`:133`) | not set | **`src_buffer`** (`:147`) → borrowed memory |
| `c_16` (`out_cb_index`, C2 only, `:149-160`) | `num_input_units * output_page_size` (`:156`) | `used_cores` | `output_cb_data_format` | `output_page_size = align(output_unit_size, dst_buffer->alignment())` (`:151`) | not set | `nullptr` (`:159`) |

**Aliasing to preserve:** when `!convert_df`, `out_cb_index == src0_cb_index == c_0` (`:129`) — the writer
drains the *same* borrowed CB the reader fills. This is *index* aliasing across configs, **not** a legacy
aliased CB (two `format_descriptors` on one `CBDescriptor`), so it needs no `advanced_options.alias_with`;
it is expressed by pointing the writer's DFB binding at `IN_DFB` in C1/C3 and at `OUT_DFB` in C2.

### Semaphores

none — the op creates no semaphores in any config, and no kernel constructs one.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:176` — `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` | `output_tensor` | writer RTA slot 0 (`:242` tiled / `:293` RM), pushed as a `Buffer*` |
| tiled writer `:23` `TensorAccessorArgs<1>()` → `:28` `TensorAccessor(dst_args, dst_addr)` | `output_tensor` | reads RTA 0 |
| RM writer `:20` `TensorAccessorArgs<1>()` → `:25` `TensorAccessor(dst_args, dst_addr)` | `output_tensor` | reads RTA 0 |

The **input** tensor has no accessor anywhere: the reader only `push_back`s already-resident pages and
builds no `TensorAccessor`. Its buffer reaches the device as the `c_0` CB's bound buffer, not as an address.

Both accessor constructions are already **2-argument** — no page-size third argument to drop.

### Work split

- Driver: **none** — no `split_work_to_cores`. Parallelism is fixed by the input's shard grid: one shard per
  core, `num_cores = all_cores.num_cores()` (`:76`), narrowed to `num_cores_unpadded` (`:110-119`) for
  HEIGHT/WIDTH-sharded inputs whose grid exceeds the data.
- num_cores: `num_cores_unpadded`, all in one group (`used_cores`).
- core_group_1 / core_group_2: n/a — a single group. Every kernel is instantiated **once** over
  `used_cores`; the per-core RTA loop (`:210-308`) varies *values*, not kernel multiplicity.

### Shared kernels

**All four kernels are borrowed — the op owns none.** Verified by filename grep over
`ttnn/cpp/ttnn/operations/`, hits filtered to factory bindings, `experimental/quasar/**` copies excluded
(those are whole-op pre-port copies, not forks to reuse).

| kernel | class | `_metal2` fork beside the original? | rung taken | remaining consumers |
|---|---|---|---|---|
| `eltwise/unary/…/dataflow/reader_unary_sharded.cpp` | borrowed, cross-family | **No** beside the original — but a real non-quasar fork exists at `copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (PR #51397) | **Rung 1 — reuse typecast's fork** (invoker decision; audit Questions #1 option (a)). No new file. | `sharded_to_interleaved_partial`, `tilize` (×2), `transpose_wh_sharded`, `untilize` (×3), `untilize_with_unpadding`, `slice_write` (×2) |
| `data_movement/sharded/…/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | borrowed, in-family | No | **Rung 2 — create** `…_metal2.cpp` beside the original + pointer comment in the original | `sharded_to_interleaved_partial` |
| `data_movement/sharded/…/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | borrowed, in-family | No | **Rung 2 — create** `…_metal2.cpp` beside the original + pointer comment in the original | `sharded_to_interleaved_partial` |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | borrowed, shared pool | No | **Rung 2 — create** `eltwise_copy_metal2.cpp` beside the original + pointer comment in the original | `copy` (×2), `interleaved_to_sharded`, `sharded_to_interleaved_partial`, `interleaved_to_sharded_partial`, `untilize_with_unpadding` |

**Reused fork's binding vocabulary is a constraint, not a choice.** `reader_unary_sharded_metal2.cpp`
fixes the reader's interface: DFB accessor `in` (PRODUCER), one RTA named `num_tiles_per_core`. The reader
`KernelSpec` is built against that, not against names of this port's choosing.

`ttnn/cpp/ttnn/kernel/compute/` is a shared *kernel pool*, not `ttnn/cpp/ttnn/kernel_lib/` — whitelist
rule 9 (no edits to framework kernel code) does not apply; the shared-kernel Caution does, hence the fork.

### Flags

- **No unreferenced kernel files** in the op directory — it holds no kernels at all.
- **Dead RM writer RTA slot 1** (`num_units_per_row`, `:294`) is pushed and never read. The named-RTA
  conversion makes this conspicuous, because the schema names only the args the kernel reads. **Left
  alone** per the brief: dropping the host-side push is a behaviour-neutral cleanup owned by the ops team.
  Reported, not fixed.
- **`is_l1_aligned` is a hardcoded `true`** (`:55`), making the RM guard at `:286-289` unconditionally
  taken and `is_blackhole` / `dst_is_dram` effectively dead there. Pre-existing; carried over verbatim.
  Reported, not fixed.
- **`num_slices` / `slice_index` are vestigial** for this op (launch site hardcodes `1` / `0`), so
  `starting_idx_h` — tiled writer slot 8 — is always `0`. Carried over verbatim (the value still comes from
  `calculate_starting_idx_h`); it is real generality for the sibling `_partial` op.
- **No descriptor type outside the audit's scan** appeared: the factory uses only `CBDescriptor`,
  `KernelDescriptor`, and their configs. No semaphores, no `WorkloadDescriptor`, no `GlobalCircularBuffer`.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — `create_program_artifacts()` returning
  `ttnn::device_operation::ProgramArtifacts`, replacing `create_descriptor()`.
- **Custom `compute_program_hash`**: none — the op is already on the default reflection-based hash, so
  there is nothing to touch either way.
- **Implementation notes**: the device-operation class needs **no** edit. `select_program_factory`
  (`sharded_to_interleaved_device_operation.cpp`) returns the same
  `ShardedToInterleavedProgramFactory{}` and the framework dispatches on the concept the factory
  satisfies, so only the factory's `.hpp`/`.cpp` change. The header swaps
  `<tt-metalium/program_descriptors.hpp>` for `"ttnn/metal_v2_artifacts.hpp"`. No pybind surface is
  removed (`sharded_to_interleaved_nanobind.cpp` exposes only the free function).

## Planned Spec Shape

One `ProgramSpec` per invocation, shaped by the config (C1 / C2 / C3). Resource names below are the
`unique_id`s; kernel-facing accessor names are given in quotes.

- **KernelSpecs** (2 in C1/C3, 3 in C2):
  - `READER` — always. Source: typecast's `reader_unary_sharded_metal2.cpp` (rung 1 reuse).
    `hw_config = ttnn::create_reader_datamovement_config(device->arch())` (legacy `ReaderConfigDescriptor{}`
    = the reader default triple). `opt_level` left at Metal 2.0's `O2` = legacy DM default.
  - `WRITER` — always; source selected by `input.layout()` between the two new `_metal2` writer forks.
    `hw_config = ttnn::create_writer_datamovement_config(device->arch())` (legacy `WriterConfigDescriptor{}`).
    `opt_level` left at `O2` = legacy DM default.
  - `COMPUTE` — C2 only. Source: new `eltwise_copy_metal2.cpp`.
    `hw_config = ComputeHardwareConfig{ComputeGen1Config{}}` — **every field left at its default**, because
    the legacy `ComputeConfigDescriptor{}` set none and the two structs' defaults coincide field-for-field
    (`HiFi4`; `math_approx_mode=false` → `sfpu_precision_mode=Precise`; `bfp8_pack_precise=false` →
    `bfp_pack_precision_mode=Approximate`; `fp32_dest_acc_en=false` → `enable_32_bit_dest=false`;
    `dst_full_sync_en=false` → `double_buffer_dest=!false=true`). Style B, per the recipe — **not** routed
    through `to_compute_hardware_config`, whose defaults are the high-performance ones and would flip
    fields the legacy op never set.
    `opt_level` set **explicitly to `O3`** — legacy `ComputeConfig` defaults to `O3`, Metal 2.0 to `O2`.
  - No same-source multiplicity in any config.
- **DataflowBufferSpecs** (1 in C1/C3, 2 in C2):
  - `IN_DFB` (`"in"`) — always. `entry_size = input_page_size`, `num_entries = num_input_units`
    (legacy `total_size = num_input_units * input_page_size`), `data_format_metadata = input_cb_data_format`,
    **`borrowed_from = INPUT`** (legacy `cb.buffer = src_buffer`).
  - `OUT_DFB` (`"out"`) — **C2 only**, mirroring the legacy `if (convert_df)` allocation.
    `entry_size = output_page_size`, `num_entries = num_input_units`,
    `data_format_metadata = output_cb_data_format`, **not** borrowed (legacy `bound_buffer = nullptr`).
  - No `advanced_options` on either: no aliasing, no multi-binding, no self-loop.
- **SemaphoreSpecs**: none — legacy creates none.
- **TensorParameters** (2, in every config):
  - `INPUT` — `input.tensor_spec()`. Declared because `IN_DFB` borrows its memory. **No `TensorBinding`
    on any kernel**: no kernel builds an accessor over the input.
  - `OUTPUT` — `output.tensor_spec()`. Bound as a `TensorBinding` on the writer only.
  - No relaxations on either.
- **WorkUnitSpecs**: **one** — all of the config's kernels over `target_nodes = used_cores`. Legacy gives
  every `KernelDescriptor` the identical `core_ranges`, so there is no second (kernels, nodes) pairing.
- **Op-owned tensors**: none — the legacy factory allocates no device tensors beyond the op's io.

### DFB endpoint census (re-derived from the kernel-touch counts, not transcribed)

Touches counted from the kernel sources, per config. Every DFB is a clean two-toucher → **1P+1C**;
nothing to self-loop, nothing to flag, nothing to drop.

| DFB | config | PRODUCER | CONSUMER | touch evidence |
|---|---|---|---|---|
| `IN_DFB` | C1 | `READER` | `WRITER` (tiled) | reader `dfb.push_back` (`:26`); tiled writer `dfb_out.wait_front` / `pop_front` (`:36`, `:49`) — binds `IN_DFB` here because `out_cb_index == src0_cb_index` |
| `IN_DFB` | C2 | `READER` | `COMPUTE` | reader `push_back`; compute `cb_in.wait_front` / `pop_front` (`:26`, `:34`) |
| `IN_DFB` | C3 | `READER` | `WRITER` (RM) | reader `push_back`; RM writer `dfb_out.wait_front` / `pop_front` (`:31`, `:44`) |
| `OUT_DFB` | C2 | `COMPUTE` | `WRITER` (tiled) | compute `cb_out.reserve_back` / `push_back` (`:27`, `:35`); tiled writer `wait_front` / `pop_front` |

Positive evidence for **no hidden co-filler**, checked in all four kernels: no `get_write_ptr`,
`get_read_ptr`, `get_local_cb_interface`, `fifo_*_ptr`, or `evil_set_*` anywhere; no semaphores; and each
kernel source goes into exactly one `KernelDescriptor` (`:310-314`) — no dual-instance work split. This
matches the brief's census.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Each of the three (C1/C2/C3) kernel sets instantiates every
kernel exactly once over one node set, so each legacy `KernelDescriptor` maps to exactly one `KernelSpec`.
No per-group CTAs exist to preserve, and nothing is at risk of CTA→RTA demotion.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `program_factory.cpp:168` | reader CTA slot 0 = `src0_cb_index` (a magic CB index) | `DFBBinding{IN_DFB, "in", PRODUCER}` |
| `program_factory.cpp:175` | writer CTA slot 0 = `out_cb_index` (a magic CB index) | `DFBBinding{IN_DFB or OUT_DFB, "out", CONSUMER}` |
| `program_factory.cpp:176` | `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT, "dst"}` on the writer |
| `program_factory.cpp:242` (tiled), `:293` (RM) | writer RTA slot 0 = `dst_buffer` (`Buffer*` → base address) | the same `TensorBinding{OUTPUT, "dst"}`; the base address is auto-injected per enqueue |
| tiled writer `:23`, RM writer `:20` | `constexpr auto dst_args = TensorAccessorArgs<1>()` + the `<1>` CTA-offset arithmetic | gone; `TensorAccessor(tensor::dst)` |
| tiled writer `:11`, RM writer `:12` | `get_arg_val<uint32_t>(0)` → `dst_addr` fed to the accessor | gone; the binding token carries the base |
| tiled writer `:22`, RM writer `:19` | `constexpr uint32_t cb_id_out / dfb_id_out0 = get_compile_time_arg_val(0)` | `dfb::out` |
| tiled writer `:26` | `get_tile_size(cb_id_out)` — free function keyed by cb id | `dfb_out.get_tile_size()` (whitelist rule 7 / §A) |
| tiled writer RTAs 1–8; RM writer RTAs 2–6 | positional `get_arg_val<uint32_t>(N)` | named RTAs via `get_arg(args::<name>)` (names = the existing kernel locals) |
| reader `:12` | `get_arg_val<uint32_t>(0)` → `num_tiles_per_core` | `get_arg(args::num_tiles_per_core)` (already so in the reused fork) |
| reader `:13` | `constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0)` | `dfb::in` (already so in the reused fork) |
| compute `:13` | `get_compile_time_arg_val(0)` → `per_core_tile_cnt` | named CTA `get_arg(args::per_core_tile_cnt)` |
| compute `:16-17`, `:28`, `:32` | hardcoded `tt::CBIndex::c_0` / `c_16` at LLK call sites | `dfb::in` / `dfb::out` passed directly (implicit `DFBBindingToken → uint32_t`) |
| compute `:19-20` | `CircularBuffer cb_in(tt::CBIndex::c_0)` / `cb_out(c_16)` | `DataflowBuffer dfb_in(dfb::in)` / `dfb_out(dfb::out)` |
| RM writer RTA slot 1 (`:294`) | `num_units_per_row`, pushed and never read | **not** dropped host-side (ops-team cleanup, not port work); simply has no name in the schema, so it stops being emitted as part of this kernel's arg set |

**No positional CTA survives.** Post-port the reader and both writers carry **no** compile-time args at
all (their only legacy CTAs were the CB index and the accessor plumbing); the compute kernel carries one
named CTA, `per_core_tile_cnt`.

## Applied Patterns

- **Borrowed-memory DFB** — `IN_DFB.borrowed_from = INPUT`, replacing the legacy `CBDescriptor::buffer =
  src_buffer` dynamic-rebinding idiom. The backing L1 address re-resolves from the `INPUT` `TensorArgument`
  on every enqueue, including program-cache hits, which is exactly what `cb.buffer` bought legacy.
- **Runtime kernel-source selection inside one factory** — the `input.layout()` branch picks between the
  two writer forks. Both forks are converted; the branch survives as a `.source` selection on one
  `KernelSpec`, and both present the *same* binding interface (`dfb::out`, `tensor::dst`), so the spec
  around it is layout-independent apart from its RTA schema.
- **Conditional resource, host-side only** — `OUT_DFB` and the `COMPUTE` `KernelSpec` exist only under
  `convert_df`, mirroring the legacy `if`. This needs **no** `#ifdef` / `defines` coordination: the
  conditional DFB is bound by a kernel that itself only exists in that config, and the writer reaches its
  DFB through one accessor name (`dfb::out`) whose *spec* differs by config. No kernel source references a
  binding token that its own config doesn't bind.
- **Shared kernel — rung 1 (reuse)** for the reader, **rung 2 (create the fork)** for the two writers and
  the compute kernel. See the Shared kernels table.
- **`dfb::name` crossing to LLKs** — the compute kernel passes `dfb::in` / `dfb::out` straight into
  `unary_op_init_common`, `copy_tile_init`, `copy_tile`, `pack_tile` via the implicit `uint32_t` conversion.
  No `.id` extraction, no temporary `DataflowBuffer` wrappers.

## Deferred / Flagged

- **New findings during planning: none structural.** The brief's census, binding cases, and RTA-naming
  gotchas all re-derived identically from the sources. No feature gate fired that the audit missed, no
  offset-folded base pointer, no `GlobalCircularBuffer`, no vararg-shaped argument, no `sem::` / `tensor::`
  handle needed at an out-of-op call site.
- **One naming decision worth flagging for reviewers:** the output tensor's accessor is named `dst`, not
  `out`, in both writer forks — `dfb::out` and `tensor::out` would be legal (different namespaces) but read
  as the same thing at a glance in a file that touches both. `dst` matches the kernels' existing
  `dst_addr` / `dst_args` vocabulary. These are shared kernels, so the name is every later consumer's
  interface.
- **Carried over verbatim, reported not fixed:** the dead RM writer RTA slot 1, the hardcoded
  `is_l1_aligned = true` and its unreachable branch, and the vestigial `num_slices` / `slice_index`. See
  `METAL2_PORT_REPORT.md` → Open items.
