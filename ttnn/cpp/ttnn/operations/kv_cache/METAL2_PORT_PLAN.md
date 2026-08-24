# Port Plan — kv_cache (UpdateKVCacheOperation)

Port plan for `ttnn/cpp/ttnn/operations/kv_cache/`, ported from the `ProgramDescriptor`
(`create_descriptor`) API to Metal 2.0. Two factories on one device operation, ported together.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Device op: `UpdateKVCacheOperation` (`device/update_cache_device_operation.{hpp,cpp}`).
  `program_factory_t = std::variant<UpdateCacheMultiCoreProgramFactory, FillCacheMultiCoreProgramFactory>`.
  `select_program_factory` picks on `op_type` (FILL → fill factory, UPDATE → update factory).
- Both factories: **`ProgramDescriptorFactoryConcept`** — a `create_descriptor()` returning
  `tt::tt_metal::ProgramDescriptor`, plus a **void** `override_runtime_arguments()` cache-hit hook that
  mutates the cached `Program` in place. Methods live on factory structs (NOT direct-descriptor —
  `program_factory_t` exists, so ttnn_factory exception 3 does not apply).
- Custom `compute_program_hash`: **present** at `update_cache_device_operation.cpp:160` — hashes
  `op_type` + `{cache, input}` tensors; deliberately excludes `batch_idx`/`update_idx`/`batch_offset`
  and `compute_kernel_config`. **Left intact** (recorded so I know not to touch it; it is the reason
  the port lands on the custom concept).

### Kernels

**UpdateCacheMultiCoreProgramFactory** (`update_cache_multi_core_program_factory.cpp`)

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `kv_cache/device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp` | all_cores | src0_cb(c_0), src1_cb(c_1), granularity, u_count, TA(cache=dst), TA(input=src) | cache_addr, input_addr, Wt, B, num_batched_heads, cache_total_num_tiles, cache_batch_num_tiles, cache_head_num_tiles, cache_start_id, input_start_id, batch_start_id | `INPUT_SHARDED` (if sharded) | none→O2 | ReaderConfigDescriptor{} (reader default) |
| writer | `kv_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp` | all_cores | out_cb(c_16), interm0(c_24), interm1(c_25), interm2(c_26), granularity, u_count, TA(cache=dst) | cache_addr, Wt, B, num_batched_heads, cache_total_num_tiles, cache_batch_num_tiles, cache_head_num_tiles, cache_start_id, batch_start_id, Wbytes, offset(=tile_update_offset), batch_read_offset | none | none→O2 | WriterConfigDescriptor{} (writer default) |
| compute_g1 | `kv_cache/device/kernels/compute/update_cache.cpp` | core_group_1 | c_0, c_1, c_24, c_25, c_26, c_16, num_batched_heads_per_core_group_1, Wt, granularity, u_count | (none) | none | none→**O3** | ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode} |
| compute_g2 (optional) | same compute source | core_group_2 | …same, CTA[6]=num_batched_heads_per_core_group_2 | (none) | none | none→**O3** | same ComputeConfigDescriptor |

**FillCacheMultiCoreProgramFactory** (`fill_cache_multi_core_program_factory.cpp`)

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp` | all_cores | TA(input=src) | src_addr, num_tiles(=num_blocks_per_core*Wt), start_id(=num_blocks_written*Wt) | `INPUT_SHARDED` (if sharded) | none→O2 | ReaderConfigDescriptor{} |
| writer | **donor** `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | all_cores | output_cb(c_0), TA(cache=dst) | dst_addr, num_pages(=num_blocks_per_core*Wt), start_id(=cache_start_id) | none | none→O2 | WriterConfigDescriptor{} |

### CBs

**UpdateCacheMultiCoreProgramFactory** (one `CBDescriptor` per row except the aliased pair)

| index | total_size | page_size | data_format | notes |
|---|---|---|---|---|
| c_0 (cache/src0) | num_cache_tiles·cache_tile · (num_cache_tiles=2·granularity·Wt) | cache_single_tile_size | cache dtype | |
| c_1 (input/src1) | num_input_tiles·input_tile | input_single_tile_size | input dtype | `.buffer = src_buffer` when sharded (borrowed memory) |
| **c_24 (interm0) + c_25 (interm1)** | num_interm_tiles·interm_tile (num_interm_tiles=2·granularity·Wt) | interm_single_tile_size | interm (fp32_dest_acc_en ? Float32 : Float16_b) | **ONE `CBDescriptor` carrying TWO `CBFormatDescriptor`s** → aliased L1 region |
| c_26 (interm2/untilized_input) | num_interm_tiles·interm_tile | interm_single_tile_size | interm | |
| c_16 (output) | num_output_tiles·cache_tile (num_output_tiles=B·Wt) | cache_single_tile_size | cache dtype | |

**FillCacheMultiCoreProgramFactory**

| index | total_size | page_size | data_format | notes |
|---|---|---|---|---|
| c_0 (src0, reused as output pass-through) | num_input_tiles·single_tile | single_tile_size | input dtype | `.buffer = src_buffer` when sharded (borrowed memory) |

### Semaphores
none (no semaphores of any kind in the op).

### Tensor accessors
| host site | originating Tensor | RTA slot (host, legacy) |
|---|---|---|
| reader_update:38 `TensorAccessor(cache_args, cache_addr)` | cache (Case 1) | reader arg 0 |
| reader_update:43 `TensorAccessor(input_args, input_addr)` | input (Case 1, interleaved only) | reader arg 1 |
| writer_update:44 `TensorAccessor(cache_args, cache_addr)` | cache (Case 1) | writer arg 0 |
| reader_fill:29 `TensorAccessor(src_args, src_addr)` | input (Case 1, interleaved only) | fill reader arg 0 |
| donor writer_unary:39 `TensorAccessor(tensor::dst)` (fork) | cache (Case 1) | fill writer arg 0 (legacy) |

All Case 1 (via `TensorAccessor` page access; no raw base-pointer arithmetic). No 3rd (page-size) arg.

### Work split
- update: `split_work_to_cores(compute_with_storage_grid_size, num_batched_heads, row_major=true)`
  (interleaved) OR shard-grid single group (sharded). Two core groups possible → **two compute
  KernelSpecs** (preserved multiplicity). reader/writer identical across groups (per-core counts ride
  RTAs, not CTAs) → single KernelSpec each.
- fill: `split_work_to_cores(..., num_blocks_of_work, row_major=true)` (interleaved) OR shard-grid.
  reader & writer read per-core counts from RTAs (no per-group CTA) → single KernelSpec each, one WU.
- Per-core arg order preserved via the shared helpers `compute_update_cache_dynamic_args` /
  `compute_fill_cache_start_ids` (kept as the single source of truth for create + override), which
  iterate `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major)`.

### Shared kernels
Census run: `grep -rl <filename> ttnn/cpp/ttnn/operations/`, then disambiguated by bound path.
- `reader_update_cache_interleaved_start_id.cpp` — hits `experimental/paged_cache/.../kernels/dataflow/`
  copy, but that factory binds **its own private copy** (path
  `experimental/paged_cache/device/kernels/dataflow/…`), not this file. **NOT lent — convert in place.**
- `writer_update_cache_interleaved_start_id.cpp` — same: paged_cache has its own private copy;
  the other two hits are *comments* in paged_cache kernels. **NOT lent — convert in place.**
- `compute/update_cache.cpp` — paged_cache has its own private copy. **NOT lent — convert in place.**
  (`#include`s shared-lib `kernel_lib/untilize_helpers.hpp` + `tilize_helpers.hpp` — lib team owns,
  no fork; function-call escape bridged by `dfb::name → uint32_t` implicit conversion.)
- `reader_fill_cache_interleaved_start_id.cpp` — deepseek_prefill has its own D2.0 fork (comment
  reference). **NOT lent — convert in place.**
- **fill writer**: **borrowed** cross-family donor
  `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. A `_metal2` fork
  **already exists** beside it (`writer_unary_interleaved_start_id_metal2.cpp`). **Rung 1 — reuse the
  existing fork.** Its interface (read-only to this port): `dfb::out` (CONSUMER), `tensor::dst`,
  named RTAs `num_pages` + `start_id`, `#ifdef`s `OUT_SHARDED`/`BACKWARDS` (this port sets neither).

### Flags
- Dead-ish includes in `update_cache_device_operation.hpp:13-14` (`global_circular_buffer.hpp`,
  `program_descriptor_patching.hpp`) — audit Misc; not port work, leave unless they block the build.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `CustomProgramSpecFactoryConcept` (both factories — each carries
  an `override_runtime_arguments`; the override re-applies the hash-excluded, attribute-derived
  per-dispatch scalars + tensor addresses).
- **Custom `compute_program_hash`:** present at `update_cache_device_operation.cpp:160` — **leave intact.**
- **Implementation notes:** kv_cache is (as of this port) the **first** op to realize
  `CustomProgramSpecFactoryConcept` — no prior in-tree precedent for the `ProgramRunArgs`-returning
  override; signature taken from `operation_concepts.hpp:129-131`. Device-op class needs **no** edits
  (no pybound `create_descriptor`; `program_factory_t` already present; no pybind-hook-only param).

## Planned Spec Shape

### UpdateCacheMultiCoreProgramFactory
- **KernelSpecs:** `reader`, `writer`, `compute_group_1`, [`compute_group_2` if group 2 present].
- **DataflowBufferSpecs:** `cache`(c_0), `input`(c_1), `interm0`(c_24), `interm1`(c_25),
  `interm2`(c_26), `output`(c_16). `interm0`+`interm1` are **aliased** (`advanced_options.alias_with`).
  `input` is `borrowed_from(input)` when sharded.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `cache`, `input`.
- **WorkUnitSpecs:** `wu_g1{reader, writer, compute_group_1} @ core_group_1`;
  [`wu_g2{reader, writer, compute_group_2} @ core_group_2` if group 2]. (reader/writer in both WUs —
  the sanctioned per-group-compute shape; see Applied Patterns.)

### FillCacheMultiCoreProgramFactory
- **KernelSpecs:** `reader`, `writer` (writer source = donor `_metal2` fork).
- **DataflowBufferSpecs:** `src0`(c_0) — reader PRODUCER (accessor "in0"), donor writer CONSUMER
  (accessor "out"). `borrowed_from(input)` when sharded.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `input` (fill reader, interleaved / borrow when sharded), `dst`=cache (donor writer).
- **WorkUnitSpecs:** `wu{reader, writer} @ all_cores`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| update compute ×{group_1[, group_2]} of `compute/update_cache.cpp` | `compute_group_1`[, `compute_group_2`] | `wu_g1`[, `wu_g2`] over **disjoint** node sets | c_0 CONSUMER, c_1 CONSUMER, c_24 PRODUCER, c_25 CONSUMER, c_26 PRODUCER, c_16 PRODUCER — each group binds one role; disjoint nodes ⇒ legal single-role, **no** multi-binding flag |

reader/writer: single KernelSpec each, listed in both WUs (per-core counts via RTA, no per-group CTA).
fill: none — no work-split multiplicity (reader/writer counts ride RTAs).

## DFB endpoint census (re-derived — GATE-free port work)

**Update** (per node, exactly one compute instance + reader + writer):
| DFB | producer | consumer | verdict |
|---|---|---|---|
| c_0 cache | reader | compute | 1P+1C |
| c_1 input | reader | compute | 1P+1C (borrowed_from input when sharded) |
| c_24 interm0 | compute (untilize out) | writer (wait/pop + raw in-place L1 poke via own consumer binding) | 1P+1C |
| c_25 interm1 | writer (reserve/push) | compute (tilize in) | 1P+1C |
| c_26 interm2 | compute (untilize out) | writer (wait/pop) | 1P+1C |
| c_16 output | compute (tilize out) | writer (wait/pop) | 1P+1C |

**c_24/c_25 disposition = Aliased DFBs, NOT multi-binding.** Two **distinct** `buffer_index`es
(c_24, c_25) on one `CBDescriptor`, each with **independent** FIFO cursors (compute produces c_24 &
consumes c_25; writer consumes c_24 & produces c_25) over one **shared** L1 region. Per
`buffer_index` the count is a clean 1P+1C. That is exactly the **Aliased DFBs** pattern
(`advanced_options.alias_with`), not multi-binding (which is ≥3 touchers or ≥2 same-role on one
`buffer_index`). The writer's raw L1 poke is a `get_read_ptr()` peek on its own c_24 consumer binding,
not a separate endpoint. The (updated) audit + brief both prescribe exactly this — one
`DataflowBufferSpec` per `buffer_index` with mutual `alias_with`, each a clean 1P+1C — and the port
matches it. Legality: identical `num_entries * entry_size`; same kernel set `{compute, writer}`; same
node set (all_cores); neither borrowed.

**Fill:** c_0 — reader PRODUCER, donor writer CONSUMER → 1P+1C (borrowed_from input when sharded).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader_update RTA 0 (cache_addr) | `dst_buffer` / `cache.buffer()->address()` | `TensorBinding(cache)` → `tensor::cache` |
| reader_update RTA 1 (input_addr) | `src_buffer` | `TensorBinding(input)` → `tensor::input` (interleaved); borrowed DFB when sharded |
| reader_update CTA 0/1 (src0/src1 cb idx) | `c_0`/`c_1` | `DFBBinding(cache/input)` |
| reader_update CTA TA(cache),TA(input) | `TensorAccessorArgs(...).append_to` | binding mechanism |
| writer_update RTA 0 (cache_addr) | `dst_buffer` | `TensorBinding(cache)` → `tensor::cache` |
| writer_update CTA 0..3 (cb idxs) | c_16/c_24/c_25/c_26 | `DFBBinding`s |
| writer_update CTA TA(cache) | `TensorAccessorArgs` | binding mechanism |
| compute CTA 0..5 (cb idxs) | c_0/c_1/c_24/c_25/c_26/c_16 | `DFBBinding`s (`dfb::name` in template args) |
| fill reader RTA 0 (src_addr) | `src_buffer` | `TensorBinding(input)` → `tensor::input` |
| fill reader CTA (cb 0 + TA) | `cb_id_in0=0` + `TensorAccessorArgs` | `DFBBinding(src0)` + `TensorBinding(input)` |
| fill writer RTA 0 (dst_addr) | `dst_buffer` | `TensorBinding(dst=cache)` → `tensor::dst` |
| fill writer CTA 0 + TA | `output_cb_index` + `TensorAccessorArgs` | `DFBBinding(src0, "out", CONSUMER)` + `TensorBinding(dst)` |
| all remaining positional CTAs (granularity, u_count) | positional | named CTAs |
| all remaining positional RTAs | positional | named RTAs (names = kernel local vars) |
| both `override_runtime_arguments` (void, mutating Program) | in-place `GetRuntimeArgs`/`UpdateDynamicCircularBufferAddress` | translated to `ProgramRunArgs`-returning override (addresses → `tensor_args`; scalars → named RTA values; borrowed-DFB backing auto-refreshes from `tensor_args`) |

No semaphore-ID RTAs (no semaphores). No page-size 3rd-arg CTAs/RTAs. No Case-2 raw-pointer bindings.

## Applied Patterns
- [Aliased DFBs](port_patterns.md) — interm0(c_24)/interm1(c_25) share one L1 region; each declares the
  other via `advanced_options.alias_with`. Same total size, same node set (all_cores), neither borrowed.
- [Multi-variant / per-group compute work-split](port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) —
  two compute KernelSpecs over disjoint core groups; reader/writer listed in both WUs; per-group count
  stays a **CTA** (not demoted to RTA).
- [Conditional / optional tensor binding](port_patterns.md#pattern-conditional--optional-dfb-bindings) —
  `tensor::input` on the update reader and fill reader is bound only when **not** `INPUT_SHARDED`; the
  existing `#ifdef INPUT_SHARDED` already gates the `TensorAccessor(tensor::input)` use. Sharded path
  uses the borrowed DFB; the `input` TensorParameter stays "used" via `borrowed_from`.
- [Borrowed-memory DFB](migration_guide.md#dataflowbufferspec) — sharded input CB → `borrowed_from(input)`
  (legacy `.buffer = src_buffer` + `UpdateDynamicCircularBufferAddress`).
- [Reuse existing `_metal2` fork (shared kernel rung 1)](port_patterns.md#caution-porting-a-shared-kernel) —
  fill writer binds the existing donor fork; adopt its `dfb::out`/`tensor::dst`/named-arg interface.
- [Pass DFB handles directly to LLK/kernel-lib helpers](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers) —
  compute passes `dfb::name` into `compute_kernel_lib::untilize/tilize<...>` template args + `compute_kernel_hw_startup`/`reconfig_data_format_srca`.

## Deferred / Flagged
- **First `CustomProgramSpecFactoryConcept` port in the tree** — no reference for the override shape;
  worth surfacing any friction (report).
- **c_24/c_25 aliased CB** — audit + brief (updated mid-port) and the re-derived census all agree:
  Aliased DFBs (`alias_with`), each a clean 1P+1C, no multi-binding flag. Port matches.
- **`num_threads`/Gen2** out of scope — Gen1 only.
