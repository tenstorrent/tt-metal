# Port Plan — `experimental/transformer/nlp_concat_heads`

Port plan for `NLPConcatHeadsProgramFactory`, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

The op has **one** program factory whose `create_descriptor` branches on `in_sharded` into two
configs that share no kernel, no CB layout and no binding classification. Because a factory is the
atomic unit of a port, **both configs convert in this change**. Almost every table below is therefore
scoped per *config*, not per factory.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `NLPConcatHeadsProgramFactory::create_descriptor(const NlpConcatHeadsParams&, const Tensor& input, Tensor& output) -> ProgramDescriptor` (`device/nlp_concat_heads_program_factory.hpp:15-16`)
- Variants: single (`program_factory_t = std::variant<NLPConcatHeadsProgramFactory>`, `device/nlp_concat_heads_device_operation.hpp:24`); two internal configs, INTERLEAVED (`in_sharded == false`) and SHARDED (`in_sharded == true`)
- Custom `compute_program_hash`: none — already the default reflection-based hash (confirmed: no `compute_program_hash` anywhere under the op directory)
- Pybound factory entry point: none — `nlp_concat_heads_nanobind.cpp` binds only the user-facing `ttnn.experimental.nlp_concat_heads`
- `get_dynamic_runtime_args` / `override_runtime_arguments`: absent

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `ProgramSpecFactoryConcept`. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Config: INTERLEAVED (`in_sharded == false`)

Note: `validate_on_program_cache_miss` (`device/nlp_concat_heads_device_operation.cpp:52-57`) forces an
INTERLEAVED output whenever the input is not sharded, so `out_sharded` is always false here and CB
index 16 is never allocated.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` (op-private) | `all_cores` (from `split_work_to_cores`) | `[0]=in0_h_tiles`, `[1]=in0_w_tiles`, `[2]=in0_c`, `[3]=in0_HtWt`, `[4..]=TensorAccessorArgs(*in0_buffer)` (`:117`) | none | per core: `[0]=in0_buffer` (bare `Buffer*`), `[1]=num_blocks_per_core`, `[2]=in0_h_dim`, `[3]=in0_tensor_tile_id` (`:198-205`) | none | none | unset → resolves `O2` (DM) | `ReaderConfigDescriptor{}` → `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed** — `eltwise/unary`) | `all_cores` | `[0]=src0_cb_index (=0)`, `[1..]=TensorAccessorArgs(*out_buffer)` (`:118-119`) | none | per core: `[0]=out_buffer` (bare `Buffer*`), `[1]=num_blocks_per_core * per_tensor_tiles`, `[2]=num_blocks_written * per_tensor_tiles` (`:207-213`) | none | none (`OUT_SHARDED` / `BACKWARDS` both undefined) | unset → resolves `O2` (DM) | `WriterConfigDescriptor{}` → `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| 0 (`src0_cb_index`) | `per_tensor_tiles * 2 * single_tile_size` (double-buffered, `:138-143`) | `all_cores` | `cb_data_format` = `datatype_to_dataformat_converter(a.dtype())` | `single_tile_size` | not set | `nullptr` (own L1) |

Index 16 (`out_cb_index`) is **not** allocated in this config.

#### Semaphores
none — the op declares no semaphore of any kind.

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `nlp_concat_heads_program_factory.cpp:117` (`TensorAccessorArgs(*in0_buffer).append_to`) → kernel `reader_tm_tile_layout_nlp_concat_heads.cpp:27,31` | `input` | reader RTA[0] (`:201`) |
| `nlp_concat_heads_program_factory.cpp:119` (`TensorAccessorArgs(*out_buffer).append_to`) → kernel `writer_unary_interleaved_start_id.cpp:16,31` | `output` | writer RTA[0] (`:210`) |

Both are two-argument `TensorAccessor` constructions — no page-size third argument anywhere in the op.

#### Work split
- Driver: `tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks)` (`:61-68`), with `num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT`
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `num_blocks_per_core_group_1`, `num_blocks_per_core_group_2`
- The per-group difference (`num_blocks_per_core`) is carried in an **RTA**, not a CTA — so there is no per-group `KernelDescriptor` multiplicity to preserve.
- Core iteration order: `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major)` with `row_major == false` in this config (`:167`, `:191`).

### Config: SHARDED (`in_sharded == true`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` (op-private) | `all_cores` = input shard grid | `[0]=src0_cb_index (=0)`, `[1]=out_cb_index (=16)`, `[2]=in0_h_tiles`, `[3]=in0_w_tiles*single_tile_size`, `[4]=num_blocks_per_core_group_1*in0_w_tiles*single_tile_size`, `[5]=num_blocks_per_core_group_1*in0_HtWt` (`:87-94`) | none | per core (identical on every core): `[0]=nheads_first_risc`, `[1]=0`, `[2]=0` (`:174-180`) | none | none | unset → resolves `O2` (DM) | `ReaderConfigDescriptor{}` → `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` |
| writer | **same source** as reader | `all_cores` | same six values (`:108`) | none | per core (identical on every core): `[0]=nheads_second_risc`, `[1]=nheads_first_risc*in0_HtWt*single_tile_size`, `[2]=nheads_first_risc*in0_w_tiles*single_tile_size` (`:181-187`) | none | none | unset → resolves `O2` (DM) | `WriterConfigDescriptor{}` → `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` |

One kernel source, two `KernelDescriptor`s over the **same** `all_cores`, differing only in DM config
and in the three-element RTA that splits the head range between the two RISCs.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| 0 (`src0_cb_index`) | `per_tensor_tiles * single_tile_size` where `per_tensor_tiles` is recomputed for the shard at `:58` | `all_cores` | `cb_data_format` | `single_tile_size` | not set | `in0_buffer` (**borrowed memory**, `:150`) |
| 16 (`out_cb_index`) | `per_tensor_tiles * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | not set | `out_buffer` (**borrowed memory**, `:163`) — allocated **only when `out_sharded`** (`:153`) |

#### Semaphores
none.

#### Tensor accessors
none — this config has no `TensorAccessor` and no address argument at all. Both tensors are reached
through borrowed-memory CBs (`cb_in0.get_read_ptr()` / `cb_out0.get_write_ptr()`,
`...sharded.cpp:43-44`).

#### Work split
- No `split_work_to_cores`. `all_cores = a.shard_spec().value().grid`, `core_group_1 = all_cores`,
  `num_blocks_per_core_group_1 = shard_shape[0] / padded_shape[-2]` (heads per shard), `:53-59`.
- The work split is *within* each core, between the two RISC instances:
  `nheads_first_risc = div_up(num_blocks_per_core_group_1, 2)`,
  `nheads_second_risc = num_blocks_per_core_group_1 - nheads_first_risc` (`:169-170`).
- Core iteration for RTA emission: `corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)` (`:173`).

### Shared kernels
| kernel path | kind | census (non-quasar binders) | `_metal2` fork beside it? | rung |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** (owned by `eltwise/unary`) | 36 factory/example/test files bind this filename (`grep -rl`, quasar excluded) | **No** — re-verified at port time: `ls ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/` shows no `*_metal2*` sibling | **rung 2 — create the fork** beside the original, add the pointer comment to the original |
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` | op-private | only this op's factory (`:123`) | n/a | convert in place |
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` | op-private | only this op's factory (`:97`, `:105`) | n/a | convert in place |

A converted fork of the donor already exists **in the wrong directory** at
`ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
(from `cbde3d44ff3`). It is not a rung-1 target (the check is locational) and is not this port's to
edit or relocate; it was read as a reference for what the converted body should look like, and its
binding vocabulary (`dfb::out`, `tensor::output`, `args::num_pages`, `args::start_id`) is adopted so
the two forks stay interchangeable when the misplacement is eventually resolved.

### Flags
- No unreferenced kernel files in the op directory.
- No descriptor type outside the audit's scan set. No `GlobalCircularBuffer`, no `address_offset`, no
  semaphores, no CTA varargs, no compute kernel.
- **The `in_sharded && !out_sharded` hole.** `validate_on_program_cache_miss` (`nlp_concat_heads_device_operation.cpp:48-51`) only forbids a
  `HEIGHT_SHARDED` output when the input is sharded, so an INTERLEAVED output on a sharded input
  passes validation and `compute_output_specs` builds a well-formed spec for it. But the factory
  allocates `cb_out0` only when `out_sharded` (`:153`) while the SHARDED kernel constructs and writes
  through it unconditionally (`...sharded.cpp:33,36,44`). Legacy tolerates this silently (a CB that
  was never created — undefined behaviour). No test parametrizes the mixed combination. See
  [Deferred / Flagged](#deferred--flagged) for how the port handles it.
- Dead code carried through unchanged (noted, not fixed — see the port report): the sharded kernel's
  unused `single_tile_size_bytes` (`...sharded.cpp:30`); the factory's `grid_to_cores` result unused
  in the SHARDED branch and the resulting no-op `row_major` variable (`:167`); the stale `// 142`
  comment (`:36`); the commented-out `cb_out0.push_back` (`...sharded.cpp:62`).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`, plain (no op-owned tensors).
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**:
  - `create_descriptor` → `create_program_artifacts(const NlpConcatHeadsParams&, const Tensor& input, Tensor& output)`. The legacy signature already matches the concept's `(attributes, tensor_args, tensor_return_value)` shape because `tensor_args_t = Tensor` and `tensor_return_value_t = Tensor`, so no parameter unwinding is needed.
  - The single `create_program_artifacts` keeps the legacy `if (in_sharded)` branch shape: the shared shape/format computation stays at the top exactly as legacy had it, then each branch builds and returns its own `ProgramArtifacts`. Returning from inside the branch (rather than filling shared, default-constructed spec variables) is what lets every spec be written as a designated-initializer literal.
  - No pybind line references `create_descriptor`, so exception 2 (pybind removal) does not apply.

## Planned Spec Shape

### Config: INTERLEAVED

- **KernelSpecs** (2):
  - `READER` — `reader_tm_tile_layout_nlp_concat_heads.cpp`; named CTAs `{in0_h_tiles, in0_w_tiles, in0_c, in0_HtWt}`; RTA schema `{num_blocks, in0_h_dim, in0_tensor_tile_id}`; `dfb_bindings = {SRC0 / "in0" / PRODUCER}`; `tensor_bindings = {INPUT / "input"}`; `hw_config = create_reader_datamovement_config(arch)`; `opt_level` left at `O2`.
  - `WRITER` — `writer_unary_interleaved_start_id_metal2.cpp` (**the new fork**); no CTAs; RTA schema `{num_pages, start_id}`; `dfb_bindings = {SRC0 / "out" / CONSUMER}`; `tensor_bindings = {OUTPUT / "output"}`; `hw_config = create_writer_datamovement_config(arch)`; `opt_level` left at `O2`.
- **DataflowBufferSpecs** (1): `SRC0` — `entry_size = single_tile_size`, `num_entries = per_tensor_tiles * 2`, `data_format_metadata = cb_data_format`; no `tile_format_metadata` (legacy `.tile` unset); not borrowed.
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `INPUT` = `input.tensor_spec()`, `OUTPUT` = `output.tensor_spec()`.
- **WorkUnitSpecs** (1): `{READER, WRITER}` over `all_cores`.
- **Op-owned tensors**: none.

### Config: SHARDED

- **KernelSpecs** (2, same source): `READER` and `WRITER`, both `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`; identical named CTAs `{in0_h_tiles, head_dim_size_bytes, out_row_size_bytes, block_size}`; identical RTA schema `{nheads, start_read_offset_bytes, start_write_offset_bytes}`; `hw_config` reader vs writer default; `opt_level` left at `O2`. DFB bindings per the endpoint census below.
- **DataflowBufferSpecs** (2):
  - `SRC0` — `entry_size = single_tile_size`, `num_entries = per_tensor_tiles` (shard-recomputed), `data_format_metadata = cb_data_format`, `borrowed_from = INPUT`.
  - `OUT` — same sizing, `borrowed_from = OUTPUT`.
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `INPUT`, `OUTPUT` (both needed as `borrowed_from` targets even though no kernel builds a `TensorAccessor` in this config; the validator registers a `borrowed_from` as a parameter use, so neither is an unbound parameter).
- **WorkUnitSpecs** (1): `{READER, WRITER}` over `all_cores`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| SHARDED: `reader_desc` + `writer_desc`, both of `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` over the same `all_cores` | `READER` + `WRITER` of the same source, differing only in `hw_config` and RTA values | one — `{READER, WRITER}` over `all_cores` | `SRC0`: `READER` PRODUCER, `WRITER` CONSUMER · `OUT`: `READER` PRODUCER, `WRITER` CONSUMER |

INTERLEAVED has **no** work-split multiplicity: the two core groups differ only in the `num_blocks`
**RTA**, never in a CTA, so one `KernelSpec` per kernel covers both groups (nothing is demoted — the
value was already an RTA in legacy).

### Endpoint census — re-derived, and it disagrees with the brief

The brief and audit prescribe `advanced_options.allow_instance_multi_binding = true` on both SHARDED
DFBs, on the grounds that both same-source instances call `reserve_back` and are therefore two locked
PRODUCERs. Re-running the census per the endpoint-assignment procedure:

| DFB / config | touchers on a node | how each touches | role lock |
|---|---|---|---|
| `SRC0` / INTERLEAVED | reader; borrowed writer | `reserve_back`+`push_back`+`get_write_ptr` (`...nlp_concat_heads.cpp:47,59,41`); `wait_front`+`pop_front`+`async_write` (`writer_unary_interleaved_start_id.cpp:40,43,41`) | 1 locked producer + 1 locked consumer |
| `SRC0` / SHARDED | instance A (reader cfg); instance B (writer cfg) | both `reserve_back(block_size)` (`...sharded.cpp:35`) + `get_read_ptr()` (`:43`) | both locked **producer** |
| `OUT` / SHARDED | instance A; instance B | both `reserve_back(block_size)` (`...sharded.cpp:36`) + `get_write_ptr()` (`:44`) | both locked **producer** |

So the SHARDED census does read "2 kernels locked to the same FIFO role", which the classification
table routes to multi-binding. **But that disposition is not constructible.** The spec validator's
per-node census (`tt_metal/impl/metal2_host_api/program_spec.cpp:1352-1389`) relaxes its bound from
"exactly one" to "**at least** one" under `allow_instance_multi_binding` — it does *not* waive the
requirement that every node host at least one PRODUCER **and** at least one CONSUMER. With exactly two
touchers and no third kernel available to be the consumer, the only binding assignment that passes
validation is **1 PRODUCER + 1 CONSUMER** — which is precisely the two-toucher work-split assignment,
and it needs no flag. (Self-looping one instance *and* multi-binding the other is both forbidden by the
recipe and rejected by the validator's self-loop rule, which requires the producer and consumer kernel
sets to be equal.)

The 1P+1C assignment is behaviour-preserving on Gen1: the DFB lowers to a plain circular buffer whose
`risc_mask` is `producer_risc_mask | consumer_risc_mask`
(`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1714`), so both RISCs get the same CB interface
state that legacy `CreateCircularBuffer` gave them, and the role label drives no device-side
machinery. The CONSUMER-bound instance's `reserve_back` and `get_write_ptr()` behave exactly as before
— `reserve_back(block_size)` where `block_size` equals the DFB's full entry count is unconditionally
satisfiable and advances no cursor. **No kernel change is made to accommodate this**: both
`reserve_back` calls stay exactly where legacy left them.

Recorded as a disagreement with the brief in the port report.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:117` + reader kernel `:27` | `TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args)` / `TensorAccessorArgs<4>()` | `TensorParameter INPUT` + `TensorBinding{INPUT, "input"}`; kernel builds `TensorAccessor(tensor::input)` |
| factory `:119` + writer kernel `:16` | `TensorAccessorArgs(*out_buffer).append_to(writer_compile_time_args)` / `TensorAccessorArgs<1>()` | `TensorParameter OUTPUT` + `TensorBinding{OUTPUT, "output"}`; kernel builds `TensorAccessor(tensor::output)` |
| factory `:201` (reader RTA slot 0) | `in0_buffer` — bare `Buffer*` pushed into the RTA list; kernel reads `get_arg_val<uint32_t>(0)` | `TensorBinding` (address auto-injected); RTA slot disappears |
| factory `:210` (writer RTA slot 0) | `out_buffer` — bare `Buffer*`; kernel reads `get_arg_val<uint32_t>(0)` | `TensorBinding`; RTA slot disappears |
| factory `:118` (writer CTA slot 0) | `src0_cb_index` (= 0) — magic CB index | `DFBBinding{SRC0, "out", CONSUMER}`; kernel uses `dfb::out` |
| reader kernel `:29` | `constexpr uint32_t cb_id_in0 = 0;` — hardcoded CB index | `DFBBinding{SRC0, "in0", PRODUCER}`; kernel uses `dfb::in0` |
| factory `:88` (sharded CTA slot 0) | `src0_cb_index` (= 0) — magic CB index | `DFBBinding{SRC0, "in0", …}`; kernel uses `dfb::in0` |
| factory `:89` (sharded CTA slot 1) | `out_cb_index` (= 16) — magic CB index | `DFBBinding{OUT, "out", …}`; kernel uses `dfb::out` |
| factory `:111-115` (reader CTA slots 0-3) | positional `{in0_h_tiles, in0_w_tiles, in0_c, in0_HtWt}` | named CTAs `in0_h_tiles`, `in0_w_tiles`, `in0_c`, `in0_HtWt` |
| factory `:90-93` (sharded CTA slots 2-5) | positional | named CTAs `in0_h_tiles`, `head_dim_size_bytes`, `out_row_size_bytes`, `block_size` (the kernel's own variable names) |
| reader RTAs `:202-204` | positional slots 1-3 | named RTAs `num_blocks`, `in0_h_dim`, `in0_tensor_tile_id` |
| writer RTAs `:211-212` | positional slots 1-2 | named RTAs `num_pages`, `start_id` (the donor kernel's own names) |
| sharded RTAs `:176-187` | positional slots 0-2 | named RTAs `nheads`, `start_read_offset_bytes`, `start_write_offset_bytes` |
| writer kernel `:19` | `get_local_cb_interface(cb_id_out).fifo_page_size` | `dfb.get_entry_size()` (CB→DFB whitelist §B) |
| reader kernel `:30`, sharded kernel `:30` | `get_tile_size(cb_id_in0)` | `dfb_in0.get_tile_size()` (whitelist §A metadata getters) |
| `desc.cbs` / `CBDescriptor` (`:142-165`) | legacy CB API | `DataflowBufferSpec` + `borrowed_from` |

Page-size third-argument CTAs/RTAs: **none** — no accessor in this op passes a third argument.
Semaphore-ID RTAs: **none** — the op has no semaphores.
Retained varargs: **none** — every RTA in all three kernels is a distinct field read once at a
literal index, so all become named args.

## Applied Patterns

- **Two-toucher DFB → assign 1P+1C (dual-instance work-split)** (`port_patterns.md`): both SHARDED DFBs (`SRC0`, `OUT`), each bound PRODUCER by the reader-config instance and CONSUMER by the writer-config instance, over one grid. See the census above for why this supersedes the brief's multi-binding disposition.
- **Caution: Porting a shared kernel** (`port_patterns.md`), rung 2: create `writer_unary_interleaved_start_id_metal2.cpp` beside the `eltwise/unary` donor, add the pointer comment to the original, bind the fork.
- **Multi-variant factories** (`port_patterns.md`) (shape only): `create_program_artifacts` branches on `in_sharded` and each branch returns its own artifact. The two configs are not device-op variants — they are one factory's internal branch.
- Borrowed-memory DFBs (`DataflowBufferSpec::borrowed_from`) for both SHARDED DFBs.

Not applied: self-loop (no one-toucher DFB), aliased DFBs (no legacy multi-`buffer_index` CB),
conditional/optional DFB bindings (see Deferred below for why the one candidate site does not take
this shape), varargs, op-owned tensors.

## Deferred / Flagged

1. **The `in_sharded && !out_sharded` hole — one forced construction decision, flagged for the ops team.**
   The brief instructs the porter not to resolve this alone. The port therefore invents *no* semantic:
   it adds no `TT_FATAL`, changes no validation, and touches no device-op code. What it cannot avoid is
   choosing whether the `OUT` `DataflowBufferSpec` and its bindings are emitted conditionally on
   `out_sharded` (mirroring legacy `:153`) or unconditionally within the SHARDED branch.

   **Decision: unconditional within the SHARDED branch**, because the kernel binds `dfb::out`
   unconditionally and a Metal 2.0 kernel cannot name a binding the spec does not declare. The
   conditional-binding pattern is not an option here: gating the kernel's output path behind an
   `#ifdef` would require inventing a "what does the kernel write when there is no output buffer"
   semantic, which is exactly the ops-team question.

   Consequence, stated plainly: for the mixed config, `borrowed_from = OUTPUT` on a non-L1 output trips
   the framework's own borrowed-DFB invariant
   (`tt_metal/impl/metal2_host_api/program_spec.cpp:1528-1533`, "must be L1-resident"), so an
   INTERLEAVED-DRAM output on a sharded input now fails loudly at program build where legacy silently
   ran a kernel against a CB that was never created. That is the framework check firing, not a
   validation the port added — but it *is* an observable change for a reachable, untested config, and
   it needs an ops-team ruling (the audit's own suggestion was a `TT_FATAL` requiring a sharded output
   whenever the input is sharded). Recorded as a Handoff point in the port report.

2. **Endpoint disposition disagreement** with the brief on both SHARDED DFBs — see the census above.
   Recorded in the port report under Friction.

3. No other new findings surfaced during planning. No feature gate fired that the audit's Appendix A
   does not cover.
