# Port Plan — `data_movement/sharded/interleaved_to_sharded`

Port plan for `InterleavedToShardedProgramFactory`, ported from the `ProgramDescriptor` API to
Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.

All `factory:N` references are to `device/interleaved_to_sharded_program_factory.cpp` at the
pre-port revision.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor()` returning a `ProgramDescriptor`
  (factory:58; declared `device/interleaved_to_sharded_program_factory.hpp:14`).
- Variants: single. `program_factory_t` is a one-alternative variant
  (`device/interleaved_to_sharded_op.hpp:23`), and the factory methods live **in a factory struct**,
  not directly on the device operation — so exception 3 (direct-descriptor shape) does not apply.
- Custom `compute_program_hash`: **present**, declared `device/interleaved_to_sharded_op.hpp:35`,
  defined `device/interleaved_to_sharded_op.cpp:144-162` — **left intact**.
- `override_runtime_arguments`: absent anywhere in the op directory.

### Config axes

The factory has no variant attribute, but three axes gate its structure, and the inventory below is
per-config where it differs:

| axis | values | decided by |
|---|---|---|
| layout | `TILE` / `RM` | `input.layout()` — factory:98 / :120 |
| output buffer | `dst-L1` / `dst-DRAM` | `dst_buffer->buffer_type()` — factory:94 |
| format conversion | `convert_df` / plain | `input_cb_data_format != output_cb_data_format` — factory:90; **TILE-only** (`device/interleaved_to_sharded_op.cpp:92-96`) |

Six reachable combinations: `TILE·{plain,convert_df}·{dst-L1,dst-DRAM}`, `RM·{dst-L1,dst-DRAM}`.

### Kernels

All six sources are **borrowed** from the in-family pool
`ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/`; the op directory holds no kernel
files. `core_ranges` is `all_cores` for every descriptor (factory:197, :218, :244) — the
`CoreRangeSet` over `get_optimal_worker_cores_for_sharded_tensor(output)` (factory:86-87).

| unique_id | source (under `…/sharded/device/kernels/`) | reached in | CTAs (positional) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | `TILE` | `{input_cb_index, all_cores.num_cores()}` + `TensorAccessorArgs(*src_buffer)` (factory:200-201) | none | `src_buffer`, `shard_height`, `shard_width`, `padded_offset`, `num_units_offset`, `curr_num_units_per_shard`, `curr_idx_h + curr_idx_w`, `starting_idx_h` (factory:290-299) | none | none | `O2` | `ReaderConfigDescriptor{}` |
| reader | `dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `RM` | `{input_cb_index, scratch_cb_index, num_trids}` + `TensorAccessorArgs(*src_buffer)` (factory:207-208) | none | `src_buffer`, `num_units_per_row` **(dead — never read)**, `shard_height`, `shard_width`, `padded_offset_bytes`, `aligned`, `aligned_width_offset`, `aligned_shard_width`, `aligned_offset`, `curr_idx_h` (factory:387-398) | none | none | `O2` | `ReaderConfigDescriptor{}` |
| writer | `dataflow/writer_unary_sharded_blocks_start_id.cpp` | `TILE·dst-DRAM` | `{out_cb_index}` + `TensorAccessorArgs(*dst_buffer)` (factory:220, :231) | none | `dst_buffer`, `shard_height`, `shard_width`, `pad_offset`, `curr_num_units_per_shard`, `num_units_offset`, `curr_idx_h + curr_idx_w`, `starting_idx_h` (factory:304-313) | none | none | `O2` | `WriterConfigDescriptor{}` |
| writer | `dataflow/writer_unary_sharded_stick_layout_start_id.cpp` | `RM·dst-DRAM` | `{out_cb_index}` + `TensorAccessorArgs(*dst_buffer)` | none | `dst_buffer`, `shard_height`, `shard_width`, `padded_offset_bytes`, `start_id`, `output_width_in_pages` (factory:405-412) | none | none | `O2` | `WriterConfigDescriptor{}` |
| writer | `dataflow/writer_unary_sharded.cpp` | `dst-L1` | `{out_cb_index}` (factory:220) | none | `{curr_num_units_per_shard}` (factory:315, :414) | none | none | `O2` | `WriterConfigDescriptor{}` |
| compute | `compute/eltwise_copy.cpp` | `convert_df` | none — the kernel hardcodes `c_0` / `c_16` (:20-21) | none | `{curr_num_units_per_shard}` (factory:425) | none | none | **`O3`** (absent field on a `ComputeConfigDescriptor` resolves to O3) | `ComputeConfigDescriptor{}` |

`grep -n opt_level` over the factory returns nothing, so every level above is the resolved default,
not a stated one.

### CBs

All three are built through the one anonymous-namespace helper `push_i2s_cb_pair` (factory:30-48),
which sets only `total_size`, `core_ranges`, a single-element `format_descriptors`, and `buffer`.
`format_descriptors[i].tile` is **never set** → `tile_format_metadata` stays `nullopt`. No
`address_offset`, no `global_circular_buffer`, no multi-element `format_descriptors` (so no aliasing).

| index | allocated when | total_size | page_size | data_format | `buffer` |
|---|---|---|---|---|---|
| `c_0` (input) | `convert_df` (factory:151-163) | `num_input_units * input_page_size` | `input_page_size = align(input_unit_size, src_buffer->alignment())` | `input_cb_data_format` | `nullptr` |
| `c_16` if `convert_df` else `c_0` (output; `out_cb_index = input_cb_index` at factory:145) | always (factory:167-174) | `num_input_units * output_page_size` | `output_page_size = align(output_unit_size, dst_buffer->alignment())` | `output_cb_data_format` | `dst_is_dram ? nullptr : dst_buffer` |
| `c_1` (alignment scratchpad) | factory:179 — layout-independent, and its last disjunct is the hardcoded `keep_l1_aligned = true` (factory:65), so **always** | `num_trids * scratch_cb_page_size` | `scratch_cb_page_size = align(input_unit_size + dram_alignment, dram_alignment)` | `input_cb_data_format` | `nullptr` |

### Semaphores

none — the op declares none, and `grep -i semaphore` over the op directory is clean.

### Tensor accessors

| host site | originating Tensor | RTA slot (host) | kernel site |
|---|---|---|---|
| factory:201 (`TILE`), :208 (`RM`) | `input` | reader slot 0 (`src_buffer`, factory:291 / :388) | `reader_unary_sharded_blocks_interleaved_start_id.cpp:40`; `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:36` |
| factory:231 | `output` (`dst-DRAM` only) | writer slot 0 (`dst_buffer`, factory:306 / :406) | `writer_unary_sharded_blocks_start_id.cpp:29`; `writer_unary_sharded_stick_layout_start_id.cpp:24` |

Both reach the kernel as a `Buffer*` pushed into an `RTArgList`, which `emplace_runtime_args`
auto-registers as a framework `BufferBinding` — no `->address()` expression exists in the factory.

### Work split

n/a — no `split_work_to_cores`. The core set is
`get_optimal_worker_cores_for_sharded_tensor(output)` (factory:86), and every kernel descriptor
covers all of it. Per-core variation rides runtime args only, so there is **no multi-`KernelDescriptor`
multiplicity to preserve**.

### Shared kernels

All six sources are **borrowed** (they live outside this op's directory). Census run as
`grep -rl <filename> ttnn/cpp/ttnn/operations/`, hits disambiguated by the bound path:

| kernel | `_metal2` sibling exists? | rung | other binders (remaining consumers) |
|---|---|---|---|
| `dataflow/writer_unary_sharded.cpp` | **yes** — `writer_unary_sharded_metal2.cpp` | **1 — reuse** | `interleaved_to_sharded_partial`, `tilize_multi_core_sharded*`, `untilize_*_nd_shard_type_*`, `experimental/padded_slice`, `experimental/transformer/nlp_kv_cache_load_slice` (tracked in #52228) |
| `compute/eltwise_copy.cpp` | **yes** — `compute/eltwise_copy_metal2.cpp` (sibling; **runtime**-arg `per_core_tile_cnt`) | **1 — reuse** | `interleaved_to_sharded_partial` |
| `dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp` | no | **2 — create** | `interleaved_to_sharded_partial`; tt-metal DM microbenchmark `tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/` |
| `dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | no | **2 — create** | same two |
| `dataflow/writer_unary_sharded_blocks_start_id.cpp` | no | **2 — create** | same two |
| `dataflow/writer_unary_sharded_stick_layout_start_id.cpp` | no | **2 — create** | same two |

**Reuse vocabulary (now a constraint on this port's `KernelSpec`s, not a free choice):**
- `writer_unary_sharded_metal2.cpp` — `dfb::out` bound CONSUMER; `args::num_units` (runtime).
- `compute/eltwise_copy_metal2.cpp` — `dfb::in` CONSUMER, `dfb::out` PRODUCER;
  `args::per_core_tile_cnt` (**runtime**). The *other* fork of this stem,
  `ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp`, reads the same name as a **`constexpr`**
  compile-time arg and therefore does **not** fit: i2s emits the count per core (factory:425 — it
  differs on the end core).

`interleaved_to_sharded_partial` binds all six and is blocked on its own gate
(`Is able to port? = no`, `TensorParameter relaxation = (legality - pending analysis)`), so the two
ops cannot co-migrate and rung 3 (convert-in-place) is unavailable.

### Flags

- **Dead RTA**: the `RM` reader's arg 1 (`num_units_per_row`, factory:389) is never read by the kernel.
- **Dead CB index under `TILE`**: `c_1` is allocated in every config but its index reaches a kernel
  only through the `RM` reader's CTA list.
- **Inert public attribute**: `keep_l1_aligned` is plumbed into `InterleavedToShardedParams` but the
  factory hardcodes `true` and never reads it (factory:64-65).
- No unreferenced kernel files in the op directory (there are none at all).
- No descriptor type outside the audit's Appendix A scan.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: present at `device/interleaved_to_sharded_op.hpp:35` /
  `device/interleaved_to_sharded_op.cpp:144-162` — **left intact**. It keys on the whole input
  `TensorSpec` (and the output's when pre-allocated), so it is at least as strict as the strict
  `TensorParameter` match the port introduces; the containment test passes under the `none`
  relaxation.
- **Implementation notes**: the factory `.hpp` swaps `create_descriptor` for
  `create_program_artifacts` and its include of `<tt-metalium/program_descriptors.hpp>` for
  `"ttnn/metal_v2_artifacts.hpp"`. No device-op-class edit is forced: there is no pybound
  `create_descriptor`, no pybind-hook-only parameter, and the op already has a `program_factory_t`.

## Planned Spec Shape

Resource names (typed constants): `IN_DFB{"in"}`, `OUT_DFB{"out"}`, `SCRATCH_DFB{"scratch"}`,
`INPUT{"input"}`, `OUTPUT{"output"}`, `READER{"reader"}`, `WRITER{"writer"}`, `COMPUTE{"compute"}`.

- **KernelSpecs**: 2 or 3 — `READER`, `WRITER`, plus `COMPUTE` when `convert_df`. One per legacy
  `KernelDescriptor`; the source path is selected per config exactly as legacy selected it.
- **DataflowBufferSpecs**: 1–3.
  - `OUT_DFB` — always. `entry_size = output_page_size`, `num_entries = num_input_units`,
    `data_format_metadata = output_cb_data_format`, `borrowed_from = OUTPUT` iff `!dst_is_dram`.
    This is the legacy "output CB", which is `c_0` itself when `!convert_df`.
  - `IN_DFB` — `convert_df` only. `entry_size = input_page_size`,
    `num_entries = num_input_units`, `data_format_metadata = input_cb_data_format`, not borrowed.
  - `SCRATCH_DFB` — `RM` only (see *Applied Patterns*). `entry_size = scratch_cb_page_size`,
    `num_entries = num_trids`, `data_format_metadata = input_cb_data_format`, not borrowed.
  - `tile_format_metadata` is `nullopt` on all three — the legacy CBs set no `.tile`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 2 — `INPUT` (`input.tensor_spec()`), `OUTPUT` (`output.tensor_spec()`), both
  strict (`relaxations` untouched). `OUTPUT` is bound by the writer under `dst-DRAM` and is the
  `borrowed_from` target under `dst-L1`; a borrow-only parameter counts as used by the validator, so
  it is declared in every config.
- **WorkUnitSpecs**: 1 — `{READER, WRITER}` (+ `COMPUTE`) over `all_cores`.
- **Op-owned tensors**: none.

### DFB endpoint bindings (from the census, re-derived)

| DFB | config | PRODUCER | CONSUMER |
|---|---|---|---|
| `OUT_DFB` | `!convert_df` (all four) | `READER` as `"in"` | `WRITER` as `"out"` |
| `OUT_DFB` | `convert_df` | `COMPUTE` as `"out"` | `WRITER` as `"out"` |
| `IN_DFB` | `convert_df` | `READER` as `"in"` | `COMPUTE` as `"in"` |
| `SCRATCH_DFB` | `RM` | `READER` as `"scratch"` | `READER` as `"scratch"` (self-loop) |

Each is exactly 1P + 1C per node. **No `allow_instance_multi_binding` anywhere**, and the one
self-loop is not stacked with it.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Every kernel descriptor covers `all_cores`, and per-core
variation is carried entirely by runtime args.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (factory:291 `TILE`, :388 `RM`) | `reader_rt.push_back(src_buffer)` → framework `BufferBinding` | `TensorParameter INPUT` + `TensorBinding{INPUT, "src"}` |
| writer RTA slot 0 (factory:306, :406) | `writer_rt.push_back(dst_buffer)` | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT, "dst"}` |
| reader CTA slot 0 (factory:200, :207) | `input_cb_index` (`tt::CBIndex::c_0`) | `DFBBinding{IN_DFB or OUT_DFB, "in", PRODUCER}` |
| reader CTA slot 1 (factory:207, `RM` only) | `scratch_cb_index` (`tt::CBIndex::c_1`) | `DFBBinding{SCRATCH_DFB, "scratch", PRODUCER}` + `{…, CONSUMER}` |
| writer CTA slot 0 (factory:220) | `out_cb_index` | `DFBBinding{OUT_DFB, "out", CONSUMER}` |
| compute kernel body (`eltwise_copy.cpp:20-21, 23-24, 30, 38`) | hardcoded `tt::CBIndex::c_0` / `c_16` | `dfb::in` / `dfb::out` on the reused fork |
| reader CTA tail (factory:201, :208) | `TensorAccessorArgs(*src_buffer).append_to(cta)` + kernel `TensorAccessorArgs<2>()` / `<3>()` | binding mechanism end-to-end; `TensorAccessor(tensor::src)` |
| writer CTA tail (factory:231) | `TensorAccessorArgs(*dst_buffer).append_to(cta)` + kernel `TensorAccessorArgs<1>()` | `TensorAccessor(tensor::dst)` |
| reader CTA slot 1 (factory:200, `TILE`) | positional `all_cores.num_cores()` | named CTA `{"num_readers", …}` |
| reader CTA slot 2 (factory:207, `RM`) | positional `num_trids` | named CTA `{"num_trids", …}` |
| every positional RTA on every kernel | `get_arg_val<uint32_t>(N)` | named RTAs — see the schemas below |
| `RM` reader RTA slot 1 (factory:389) | `num_units_per_row`, **never read by the kernel** | nothing — no named slot; the dead value stops being emitted |
| `c_1` allocation under `TILE` (factory:176-192) | a `CBDescriptor` no kernel receives the index of | nothing — no `DataflowBufferSpec` is built under `TILE` |

No page-size 3rd-argument CTAs/RTAs (no accessor passes a third argument), no semaphore-ID RTAs, no
Case 2 raw-pointer bindings.

### Runtime-arg schemas (named)

| kernel | config | `runtime_arg_names` |
|---|---|---|
| `READER` | `TILE` | `block_height_tiles`, `block_width_tiles`, `padded_offset_bytes`, `input_width_offset_tiles`, `block_num_tiles`, `start_id_offset`, `start_id_base` |
| `READER` | `RM` | `block_height`, `block_width_bytes`, `padded_block_width_bytes`, `aligned`, `aligned_input_width_offset_bytes`, `aligned_block_width_bytes`, `aligned_offset`, `start_id` |
| `WRITER` | `TILE·dst-DRAM` | `block_height_tiles`, `block_width_tiles`, `padded_offset`, `block_width_padded_num_tiles`, `output_width_tiles`, `start_id_offset`, `start_id_base` |
| `WRITER` | `RM·dst-DRAM` | `block_height`, `block_width_bytes`, `padded_block_width_bytes`, `start_id`, `output_width_in_pages` |
| `WRITER` | `dst-L1` | `num_units` (the reused fork's name) |
| `COMPUTE` | `convert_df` | `per_core_tile_cnt` (the reused fork's name) |

Names are taken from the kernels' own vocabulary — the existing parameter names in each legacy
kernel's `get_arg_val` block — not from the host's locals, because four of these become the fork's
permanent interface.

## Applied Patterns

- [Conditional / optional resource bindings](port_patterns.md#pattern-conditional--optional-resource-bindings):
  `SCRATCH_DFB` is declared and bound only under `ROW_MAJOR`. **No kernel-side `#ifdef` is needed**:
  the two readers are *separate sources*, and only the `RM` one names `dfb::scratch`, so the token is
  absent exactly where no kernel references it. The `TILE` reader never mentions it. This is the
  cheaper form of the pattern — conditional binding without a preprocessor gate — and it is available
  only because the layout axis already selects the source file.
- [Sync-free and single-ended CBs → self-loop DFB](port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `SCRATCH_DFB` has exactly one toucher (the `RM` reader), which both `reserve_back`s/`push_back`s it
  and peeks it with `get_write_ptr()`. Bound PRODUCER **and** CONSUMER on that one kernel, sharing the
  accessor name `"scratch"`. A DM self-loop is legal on Gen1; it is Quasar-uplift debt, not a Gen1
  blocker.
- [Same-FIFO-style one-DFB-two-names](port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  (the host-side shape `sharded_to_interleaved` also uses): when `!convert_df` the legacy
  `out_cb_index = input_cb_index` makes one CB serve as both the reader's output and the writer's
  input. One `DataflowBufferSpec` (`OUT_DFB`), bound by the reader as `"in"` and the writer as
  `"out"` — two accessor names on two *different* kernels, which is ordinary 1P+1C, not aliasing.
- [Porting a shared kernel](port_patterns.md#caution-porting-a-shared-kernel): rung 1 for two
  kernels, rung 2 (create the fork beside the original, plus a pointer comment in the original) for
  four. See the *Shared kernels* table.
- [Pass DFB handles directly to LLKs](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  the reused compute fork already does this (`compute_kernel_hw_startup(dfb::in, dfb::out)` etc.);
  the new reader fork uses the same conversion for the `constexpr` `get_tile_size(dfb::in)`.

## Deferred / Flagged

- **New finding (planning):** the `TILE` reader's `constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0)`
  feeds a **non-type template argument** (`get_barrier_read_threshold<tile_bytes, num_readers>()`,
  :42). The whitelist's `constexpr` carve-out covers it — keep the free-function form with the binding
  token — and this is the *loud* failure case if it were demoted to a member getter, so it would not
  ship silently. Confirmed the `DFBAccessor → uint32_t` conversion is `constexpr` before relying on
  it.
- **New finding (planning):** the two `dst-DRAM` writers and the `TILE` reader read a tile size the
  legacy kernel declared **`const`**, not `constexpr` (`writer_unary_sharded_blocks_start_id.cpp:27`),
  so those take the member getter. The declaration is the whole test; the three sites split two ways
  and each was checked individually rather than as a group.
- Nothing else the audit missed surfaced during planning. No structural issue, no feature gate outside
  Appendix A, no construct that cannot be expressed.
