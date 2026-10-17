# Port Plan — experimental/transformer/nlp_concat_heads_boltz

Port plan for `ttnn.experimental.nlp_concat_heads_boltz` (`NLPConcatHeadsBoltzDeviceOperation`), ported from the
`ProgramDescriptor` API (direct `create_descriptor`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Line numbers below are pre-port: `factory.cpp` = `device/nlp_concat_heads_boltz_program_factory.cpp`,
`reader` = `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz.cpp`,
`sharded` = `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, **direct-descriptor shape**. `create_descriptor` is a static member
  of `NLPConcatHeadsBoltzDeviceOperation` itself (`device/nlp_concat_heads_boltz_device_operation.hpp:27-28`), body
  at `factory.cpp:18-219`. There is no `program_factory_t` (re-checked at port time: still none). This forces
  `ttnn_factory.md` exception 3.
- Variants: one factory body, two paths chosen by `in_sharded = a.is_sharded()` (`factory.cpp:29`). The path picks
  the kernel sources at runtime, so both paths and all three kernel sources convert together.
  - **Interleaved:** reader `reader_tm_tile_layout_nlp_concat_heads_boltz.cpp` + borrowed writer
    `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`.
  - **Sharded:** `reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp`, instantiated twice (reader config and
    writer config) over the same grid.
- Custom `compute_program_hash`: none. Default reflection-based hash over `NLPConcatHeadsBoltzParams`
  (`output_mem_config`) + `NLPConcatHeadsBoltzInputs` (`input`, `preallocated_output`). No `attribute_values` /
  `to_hash` backdoor.

*(The target concept was chosen in the audit; it is carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Variant: interleaved (`!in_sharded`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz.cpp` | `all_cores` (from `split_work_to_cores`) | `[0] in0_h_tiles`, `[1] in0_w_tiles`, `[2] in0_c`, `[3] in0_HtWt`, `[4..] TensorAccessorArgs(*in0_buffer)` (`factory.cpp:107-113`) | none | per core: `[0] in0_buffer` (`Buffer*`), `[1] num_blocks`, `[2] in0_h_dim`, `[3] in0_tensor_tile_id` (`:194-201`) | none | none | unset → **O2** | `ReaderConfigDescriptor{}` (RISCV_1 / NOC_0 / dedicated) |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed**) | `all_cores` | `[0] src0_cb_index` (=0), `[1..] TensorAccessorArgs(*out_buffer)` (`:114-115`) | none | per core: `[0] out_buffer` (`Buffer*`), `[1] num_blocks_per_core * per_tensor_tiles`, `[2] num_blocks_written * per_tensor_tiles` (`:203-209`) | none | none (`OUT_SHARDED`, `BACKWARDS` unset) | unset → **O2** | `WriterConfigDescriptor{}` (RISCV_0 / NOC_1 / dedicated) |

RTA node order: `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major=false)` (`:186`); group 1 cores get
`num_blocks_per_core_group_1`, the rest `num_blocks_per_core_group_2` (`:189`).

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 (`src0_cb_index`) | `2 * per_tensor_tiles * single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(a.dtype())` | `single_tile_size` | not set |
| 16 (`out_cb_index`) — **only if `out_sharded`** (a sharded preallocated output) | `per_tensor_tiles * single_tile_size`, `.buffer = out_buffer` | `all_cores` | same | same | not set |

On this path CB 16 is **dead**: neither interleaved kernel touches index 16.

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `factory.cpp:113` `TensorAccessorArgs(*in0_buffer)` → reader CTA 4..; kernel `TensorAccessorArgs<4>()` (reader `:26`), `TensorAccessor(in0_args, in0_tensor_addr)` (reader `:30`) | `tensor_args.input` | reader RTA 0 (`Buffer*`, `:197`) |
| `factory.cpp:115` `TensorAccessorArgs(*out_buffer)` → writer CTA 1..; kernel side in the borrowed writer | output (`tensor_return_value`) | writer RTA 0 (`Buffer*`, `:206`) |

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` (`:66`), `num_blocks = ashape[1] * ashape[2] / 32`.
- `(num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2)`. One
  `KernelDescriptor` per kernel covers both groups; the per-group count is an RTA (not a CTA), so there is no
  per-group KernelSpec multiplicity to preserve.

### Variant: sharded (`in_sharded`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_boltz_sharded.cpp` | `all_cores` = input shard grid | `[0] src0_cb_index` (=0), `[1] out_cb_index` (=16), `[2] in0_h_tiles`, `[3] in0_w_tiles * single_tile_size`, `[4] num_blocks_per_core_group_1 * in0_w_tiles * single_tile_size`, `[5] num_blocks_per_core_group_1 * in0_HtWt` (`:84-91`) | none | per core: `[0] nheads_first_risc`, `[1] 0`, `[2] 0` (`:170-176`) | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | same source | `all_cores` | same CTA vector (`:104`) | none | per core: `[0] nheads_second_risc`, `[1] nheads_first_risc * in0_HtWt * single_tile_size`, `[2] nheads_first_risc * in0_w_tiles * single_tile_size` (`:177-183`) | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

RTA node order: `corerange_to_cores(all_cores, std::nullopt, row_major)` (`:169`); every node gets the same values.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 | `per_tensor_tiles * single_tile_size` (`per_tensor_tiles` = shard tiles), `.buffer = in0_buffer` | `all_cores` | same | `single_tile_size` | not set |
| 16 — only if `out_sharded` | `per_tensor_tiles * single_tile_size`, `.buffer = out_buffer` | `all_cores` | same | same | not set |

#### Tensor accessors
none. Both tensors are reached only through borrowed CBs (`.buffer` set).

#### Work split
n/a. `all_cores` is the input shard grid; the two DM RISCs split each core's heads
(`nheads_first_risc = div_up(n, 2)`, `nheads_second_risc = n - nheads_first_risc`).

### Semaphores
none (both variants).

### Shared kernels
- **Borrowed:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`.
  Located the fork by listing that directory: `writer_unary_interleaved_start_id_metal2.cpp` exists beside it →
  **rung 1, reuse**. The fork's binding vocabulary is now this factory's constraint:
  - `dfb::out` (the kernel waits on / pops it → CONSUMER)
  - `tensor::dst` (output tensor, via `TensorAccessor`)
  - named RTAs `num_pages`, `start_id`; no CTAs
  - `#ifdef OUT_SHARDED`, `#ifdef BACKWARDS`: neither defined here (the legacy op set no defines)
  The legacy original already carries the pointer comment; nothing to add.
- **Own kernels:** `grep -rl reader_tm_tile_layout_nlp_concat_heads_boltz ttnn/cpp/ttnn/operations/` hits only this
  factory (plus this op's METAL2 docs). Outside `experimental/quasar/` (not consulted), no other binder exists.
  Both convert in place.

### Flags
- No unreferenced kernel files in the op directory (two kernels, both bound).
- Sharded input + interleaved output (accepted by validation, `nlp_concat_heads_boltz_device_operation.cpp:47-50`)
  runs the sharded kernel against an unallocated CB 16. This is a legacy bug; see Deferred / Flagged.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (no `override_runtime_arguments`).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: exception 3. In `NLPConcatHeadsBoltzDeviceOperation`:
  - nest `struct NLPConcatHeadsBoltzProgramFactory` with `create_program_artifacts`
  - add `using program_factory_t = std::variant<NLPConcatHeadsBoltzProgramFactory>;`
  - remove the device-op-level `create_descriptor`

  Its explanatory comment (the cache-hit rationale) moves onto the new factory method, reworded from
  "runtime-arg bindings / CB `.buffer` bindings" to "tensor bindings / borrowed-DFB bindings". The factory body stays
  in `device/nlp_concat_heads_boltz_program_factory.cpp`. There is no pybind `create_descriptor` to delete.

## Planned Spec Shape

### Variant: interleaved
- KernelSpecs:
  - `reader` (own reader, converted in place)
  - `writer` (the `_metal2` fork)
- DataflowBufferSpecs: one, `in0` (legacy CB 0).
  - `entry_size = single_tile_size`, `num_entries = 2 * per_tensor_tiles`, `data_format_metadata = data_format`
  - reader binds it as PRODUCER, accessor `in0`
  - writer binds it as CONSUMER, accessor `out` (the fork's name)

  Legacy CB 16 (dead here) gets **no** spec.
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`.
  - reader `TensorBinding{input, "in0_tensor"}` (the kernel's word is `in0_tensor_addr`)
  - writer `TensorBinding{output, "dst"}` (the fork's name)
- WorkUnitSpecs: one, `nlp_concat_heads_boltz` = {reader, writer} on `all_cores`.

### Variant: sharded
- KernelSpecs: `reader`, `writer`, both of the sharded source and over `all_cores`, with the same named CTAs.
- DataflowBufferSpecs:
  - `in0` (legacy CB 0): `entry_size = single_tile_size`, `num_entries = per_tensor_tiles`,
    `borrowed_from = input`.
  - `out0` (legacy CB 16), **only when `out_sharded`**: `num_entries = per_tensor_tiles`, `borrowed_from = output`.
- SemaphoreSpecs: none.
- TensorParameters: `input`, `output`. They are referenced only through `borrowed_from`; there are no
  `TensorBinding`s.
- WorkUnitSpecs: one, as above.

Both TensorParameters are declared on both paths (same as the sibling `nlp_concat_heads`).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `reader_desc` + `writer_desc` of `..._sharded.cpp`, Reader- / Writer-config, **same** `all_cores` (`factory.cpp:93-105`) | `reader`, `writer` (both the sharded source) | `nlp_concat_heads_boltz` (one, same grid) | `in0`: reader PRODUCER / writer CONSUMER; `out0`: reader PRODUCER / writer CONSUMER |

**Census, re-derived from the kernel** (sharded `:31-35,42-43`): each instance calls `reserve_back(block_size)` on
both CBs once at entry, then raw-peeks `get_read_ptr` / `get_write_ptr` + offset. Nothing is ever pushed, waited on or
popped (the `push_back` is commented out). So each CB has exactly two touchers on every node. Strictly, `reserve_back`
locks both as producers, which would put them in the multi-binding row. But the flag can't express "two producers,
zero consumers": the validator still requires a CONSUMER per node. And the `reserve_back` calls are dead, since they
reserve once on an empty CB and nothing ever consumes. That makes them **1P+1C**, which matches the brief. The kernel
stays verbatim, including both `reserve_back` calls and the commented-out `push_back`. The Gen1
`DataflowBuffer::reserve_back` doesn't role-check (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:175`).

The interleaved path has no same-source multiplicity.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:197` / reader `:17` RTA 0 | `in0_buffer` (`Buffer*`) → `in0_tensor_addr` | `TensorBinding{input, "in0_tensor"}` |
| factory `:113` / reader `:26` | `TensorAccessorArgs(*in0_buffer).append_to(...)` / `TensorAccessorArgs<4>()` | `TensorAccessor(tensor::in0_tensor)` |
| factory `:206` writer RTA 0 | `out_buffer` (`Buffer*`) | `TensorBinding{output, "dst"}` (fork `tensor::dst`) |
| factory `:115` writer CTA 1.. | `TensorAccessorArgs(*out_buffer).append_to(...)` | fork's `TensorAccessor(tensor::dst)` |
| factory `:114` writer CTA 0 | `src0_cb_index` (=0) | `DFBBinding{in0, "out", CONSUMER}` |
| reader `:28` | hardcoded `cb_id_in0 = 0` | `DFBBinding{in0, "in0", PRODUCER}`; `DataflowBuffer dfb_in0(dfb::in0)` |
| reader `:29`, sharded `:29` | `get_tile_size(cb_id_in0)` (non-`constexpr`) | `dfb_in0.get_tile_size()` (whitelist rule 7) |
| sharded CTAs 0/1 (factory `:85-86`, kernel `:20-21`) | `src0_cb_index` / `out_cb_index` | `DFBBinding`s `in0` / `out0` (reader PRODUCER, writer CONSUMER) |
| sharded CB 0 `.buffer = in0_buffer` (`:147`) | buffer-backed CB | `DataflowBufferSpec{in0, borrowed_from = input}` |
| sharded CB 16 `.buffer = out_buffer` (`:160`) | buffer-backed CB | `DataflowBufferSpec{out0, borrowed_from = output}` (iff `in_sharded && out_sharded`) |
| interleaved CB 16 (`:150-162`, iff `out_sharded`) | dead CB (no kernel touches index 16) | **dropped**, no spec |
| reader CTAs 0..3 | positional `in0_h_tiles`, `in0_w_tiles`, `in0_c`, `in0_HtWt` | named CTAs, same names |
| reader RTAs 1..3 | positional `num_blocks`, `in0_h_dim`, `in0_tensor_tile_id` | named RTAs, same names |
| writer RTAs 1..2 | positional page count / start page | fork's named RTAs `num_pages`, `start_id` |
| sharded CTAs 2..5 | positional `in0_h_tiles`, `head_dim_size_bytes`, `out_row_size_bytes`, `block_size` | named CTAs, same names |
| sharded RTAs 0..2 | positional `nheads`, `start_read_offset_bytes`, `start_write_offset_bytes` | named RTAs, same names |

There are no page-size third-argument sites (both accessors are 2-arg), no semaphore RTAs and no varargs.

## Applied Patterns

- Two-toucher DFB → assign 1P+1C (dual-instance work-split): sharded `in0` and `out0`.
- Caution: Porting a shared kernel, rung 1 (reuse `writer_unary_interleaved_start_id_metal2.cpp`).
- Dead CB → no spec: interleaved-path CB 16.
- Borrowed-memory DFBs: sharded `in0` / `out0`.
- `ttnn_factory.md` §3: give a direct-descriptor op a conventional program factory.
- DM hw_config: every legacy config is the reader or writer default, so it maps to
  `ttnn::create_reader_datamovement_config()` / `ttnn::create_writer_datamovement_config()`.
- `AddRuntimeArgsForNode` inside the legacy per-core loops (loop shape and node order kept as they are).

## Deferred / Flagged

- **`out0` is conditional on the host but referenced unconditionally in the kernel.** That is fine for every
  configuration that reaches the sharded kernel legally: sharded output ⇔ `out0` bound. The one other configuration,
  sharded input + interleaved output, is the legacy bug. It is accepted by validation, and legacy runs the kernel
  against an unallocated CB 16. After the port, `dfb::out0` is undeclared there, so the kernel fails to JIT
  instead of hanging or writing stray L1. Per the brief, no `TT_FATAL` and no `#ifdef` gate is added: gating
  would keep the bug running, and a guard is an ops-team change. Reported prominently.
- The sharded path can't be exercised on this 8×8 Wormhole. The output BLOCK_SHARDED spec needs ≥ S ≥ 32 shard
  rows, so it is compile- and runtime-unverified here.
- `reserve_back(uint16_t)` narrowing on the sharded `block_size` CTA (dead call; report only).
- Audit anomalies carried verbatim: sharded over-read via `in0_h_tiles = S·S/32`, unused `single_tile_size_bytes`
  in the sharded kernel, stale comments ("WRITER RUNTIME ARGS" in a reader, "interleaved accessor args" in the
  sharded kernel, "Grayskull Device Setup").
