# Port Plan — nlp_concat_heads_decode

Port plan for `nlp_concat_heads_decode`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories return `tt::tt_metal::ProgramDescriptor`)
- Variants: two — `NLPConcatHeadsDecodeProgramFactory`, `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (selected by `on_subcoregrids`)
- Custom `compute_program_hash`: none — already default reflection-based hash

Both factories share an identical structure. Differences: the subcoregrid variant uses tile-face geometry from the tensor spec and a flat single-axis input-core list (`in_num_cores`) rather than a bounding-box `(num_x, num_y)`.

### Kernels (per factory; reader and writer are the SAME source)
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `reader_tm_tile_layout_nlp_concat_heads_decode.cpp` | `q_cores` | element_size, sub_tile_line_bytes, **cb_id_q_out**, head_size, batch, head_tiles, **PHASES_TO_READ=1**, num_x, num_y | in_tile_offset_by_head, **in_buffer (Buffer\*)**, noc_x[num_x], noc_y[num_y] | ReaderConfig |
| writer | (same source) | `q_cores` | (same, **PHASES_TO_READ=2**) | (same RTAs) | WriterConfig |

Subcoregrid factory: same shape; CTA tail is `in_num_cores, face_h, face_hw`; RTA arrays are `noc_x[in_num_cores], noc_y[in_num_cores]`.

### CBs
| index | total_size | core_ranges | data_format | page_size | borrowed |
|---|---|---|---|---|---|
| c_16 | q_num_tiles · single_tile_size | q_cores | input dtype | single_tile_size | **`.buffer = output.buffer()`** |

### Semaphores
none

### Tensor accessors
| binding | originating Tensor | legacy form | Metal 2.0 |
|---|---|---|---|
| input | tensor_args.input (HEIGHT_SHARDED) | `Buffer*` RTA → `q_start_addr`; exotic per-core NoC walk | TensorParameter + `get_bank_base_address()` (Case 2 bridge) |
| output | tensor_return_value (WIDTH_SHARDED) | borrowed CB `.buffer` | TensorParameter, backs borrowed DFB |

### Work split
n/a — one kernel instance per output core, no `split_work_to_cores` core-group multiplicity. Per-core RTA `in_tile_offset_by_head`.

### Cross-op kernels
none.

### Flags
Both kernels write *only* to the output CB via `get_write_ptr()` (no FIFO ops) → fake CB. The two kernels split the per-tile read into two phases (RISC0/RISC1) via `PHASES_TO_READ`.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: convert BOTH factories' `create_descriptor` → `create_program_spec` (they satisfy the same concept; framework dispatches per `select_program_factory`).

## Planned Spec Shape (per factory)
- **KernelSpecs**: `reader` (PHASES_TO_READ=1, `DataMovementRoleHint::READER`), `writer` (PHASES_TO_READ=2, `DataMovementRoleHint::WRITER`) — same source.
- **DataflowBufferSpecs**: `OUT` (entry_size = single_tile_size, num_entries = q_num_tiles, data_format, `borrowed_from = OUTPUT`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (input spec), `OUTPUT` (output spec, backs OUT DFB).
- **WorkUnitSpecs**: one — `{reader, writer}` on `q_cores`.

### DFB endpoint bindings
- `OUT`: bound `reader=PRODUCER`, `writer=CONSUMER` (fake-CB validator-satisfying split; both kernels run on the same nodes, so one producer + one consumer per node).

### Tensor bindings
- `INPUT` (`ta::input`) on both reader and writer (base via `get_bank_base_address()`).
- `OUTPUT` not bound as a TensorBinding — used only via `borrowed_from`.

## Preserved Multiplicity
none — no work-split core-group multiplicity in legacy.

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer CTA slot 2 | `cb_id_q_out` magic CB index | `DFBBinding(OUT, "q_out", …)` |
| reader/writer RTA slot 1 | `in_buffer` (Buffer\*) → `q_start_addr` | `TensorBinding(INPUT)` + `get_bank_base_address()` |
| reader/writer RTA slots 2.. | `get_arg_addr` NoC-coord arrays | common runtime varargs (`get_common_vararg`) |
| reader/writer CTAs (positional) | `get_compile_time_arg_val(N)` | named CTAs `get_arg(args::…)` |
| reader/writer RTA slot 0 | `get_arg_val<uint32_t>(0)` | named RTA `get_arg(args::in_tile_offset_by_head)` |

## Applied Patterns
- [Fake CB → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md): output CB is write-only address source — bound reader=PRODUCER / writer=CONSUMER.
- Borrowed-memory DFB: `borrowed_from = OUTPUT`.
- Case-2 base-pointer bridge: `get_bank_base_address()` for the input.
- Common runtime varargs for the NoC coordinate arrays.

## Deferred / Flagged
- `input` Case-2 classification flagged in audit (user-directed port). TensorAccessor enhancement candidate recorded team-side.
