# Port Plan — nlp_concat_heads_decode

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode`, ported from the
`ProgramDescriptor` API (`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

*Filled in during the inventory step.*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — both factories define `static ProgramDescriptor create_descriptor(...)`
  inside a `program_factory_t` variant (`device/nlp_concat_heads_decode_device_operation.hpp:20-21`), so exception 3
  (direct-descriptor shape) does **not** apply; the port is a method swap inside the existing structs.
- Variants: two factories, selected by `operation_attributes_t::on_subcoregrids`:
  - `NLPConcatHeadsDecodeProgramFactory` (`device/nlp_concat_heads_decode_program_factory.cpp`) — default full-grid path
  - `NLPConcatHeadsDecodeSubcoregridsProgramFactory` (`device/nlp_concat_heads_decode_subcoregrids_program_factory.cpp`) — sub-core-grid path
- Custom `compute_program_hash`: none — default reflection-based hash (audit-confirmed; grep re-confirmed:
  no `compute_program_hash` / `attribute_values` / `to_hash` in the op directory).
- `override_runtime_arguments`: none (base concept target).
- Pybound `create_descriptor`: none (`nlp_concat_heads_decode_nanobind.cpp` binds only the public op function).

> Both factories have the same structural shape (dual-instance work-split of one kernel source over the output
> shard grid, one borrowed-memory CB, no semaphores, no TensorAccessor). Inventories are given per variant below.

### Variant: default (`NLPConcatHeadsDecodeProgramFactory`)

#### Kernels
Both descriptors bind the **same source**:
`device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp`

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (desc 0) | reader_tm_tile_layout_nlp_concat_heads_decode.cpp | `q_cores` | [0] element_size, [1] sub_tile_line_bytes, [2] q_output_cb_index (=c_16), [3] head_size, [4] batch, [5] head_tiles, [6] 1 (phase), [7] in_num_cores_x, [8] in_num_cores_y | none | per core `cores[i]`: [0] in_tile_offset_by_batch, [1] `Buffer* in_buffer`, [2..2+num_x) noc_x_coords, [2+num_x..2+num_x+num_y) noc_y_coords | none | none | resolved **O2** (field unset; `opt_level.value_or(O2)` on DM lowering, `program.cpp:424`) | `ReaderConfigDescriptor{}` → resolved (RISCV_1, NOC_0, DM_DEDICATED_NOC) (verified: `kernel_types.cpp:13-27`, `preferred_noc_for_dram_read` = NOC_0 all Gen1 arches) |
| writer (desc 1) | same source | `q_cores` | identical except [6] = 2 (phase) | none | **byte-identical** RTA list per core (same `rt_args` emplaced into both descriptors) | none | none | resolved **O2** | `WriterConfigDescriptor{}` → resolved (RISCV_0, NOC_1, DM_DEDICATED_NOC) (verified: `kernel_types.cpp:29-43`, `preferred_noc_for_dram_write` = NOC_1) |

RTA slot [0]'s host-side name is `in_tile_offset_by_batch`; the kernel reads it as `in_tile_offset_by_head` (arg 0).
Slot [1] is the input `Buffer*` (framework `BufferBinding` pointer-patching form), consumed kernel-side raw as
`q_start_addr` (arg 1).

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| c_16 (`q_output_cb_index`) | `q_num_tiles * single_tile_size` | `q_cores` | `cb_data_format` (from input dtype) | `single_tile_size` | not set | **borrowed**: `.buffer = output.buffer()` (`nlp_concat_heads_decode_program_factory.cpp:54`) |

#### Semaphores
none

#### Tensor accessors
none — no `TensorAccessor` anywhere in the op (host or kernel). The input address travels as a raw base
(`Buffer*` RTA slot 1); the kernel assembles remote NoC addresses by hand (`{.noc_x, .noc_y, .addr}`).

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `nlp_concat_heads_decode_program_factory.cpp:130` (`rt_args.push_back(in_buffer)`) | input | slot 1, both kernels |

#### Work split
- n/a — no `split_work_to_cores`. One head per output core: `cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, /*row_wise=*/true)`
  over the bounding box of `q_cores` (output shard grid); core `i` handles head `i` (offset computed at
  `nlp_concat_heads_decode_program_factory.cpp:119-124`).
- The *intra-tile* work split is between the two kernel instances: phase CTA selects left (phase 1, reader
  instance) vs right (phase 2, writer instance) half-tile lines. Both instances run on every core of `q_cores`.

### Variant: subcoregrids (`NLPConcatHeadsDecodeSubcoregridsProgramFactory`)

#### Kernels
Both descriptors bind the **same source**:
`device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp`

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (desc 0) | reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp | `q_cores` | [0] element_size, [1] sub_tile_line_bytes, [2] q_output_cb_index (=c_16), [3] head_size, [4] batch, [5] head_tiles, [6] 1 (phase), [7] in_num_cores, [8] face_h, [9] face_hw | none | per core `cores[i]`: [0] in_tile_offset_by_batch, [1] `Buffer* in_buffer`, [2..2+in_num_cores) noc_x_coords, [2+in_num_cores..2+2*in_num_cores) noc_y_coords | none | none | resolved **O2** | `ReaderConfigDescriptor{}` → (RISCV_1, NOC_0, DM_DEDICATED_NOC) |
| writer (desc 1) | same source | `q_cores` | identical except [6] = 2 | none | byte-identical RTA list per core | none | none | resolved **O2** | `WriterConfigDescriptor{}` → (RISCV_0, NOC_1, DM_DEDICATED_NOC) |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| c_16 | `q_num_tiles * single_tile_size` | `q_cores` | `cb_data_format` | `single_tile_size` | not set | **borrowed**: `.buffer = output.buffer()` (`..._subcoregrids_program_factory.cpp:64`) |

#### Semaphores
none

#### Tensor accessors
none (same as default variant).

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._subcoregrids_program_factory.cpp:137` (`rt_args.push_back(in_buffer)`) | input | slot 1, both kernels |

#### Work split
- n/a — one head per output core: `cores = corerange_to_cores(q_cores, num_cores, /*row_wise=*/true)`;
  offset at `..._subcoregrids_program_factory.cpp:127-131`. Input NoC coordinate tables come from
  `corerange_to_cores(in_cores, ...)` (per-core pairs rather than the default variant's per-axis vectors).
- Intra-tile split between the two instances via the phase CTA, as in the default variant.

### Shared kernels
none — repo-wide grep of both kernel filenames (`grep -rl` over `ttnn/cpp/ttnn/operations/`) hits only this op's
two factories (audit re-confirmed). No `_metal2` fork exists beside either; neither is bound by any other op or by
the *other* factory of this op (each factory has its own kernel file). No fork is needed.

### Flags
- No unreferenced kernel files in the op directory (both files are bound, one per factory).
- All descriptor types used (`CBDescriptor`, `KernelDescriptor` + Reader/Writer config descriptors) are within the
  audit's scan; no GlobalCircularBuffer, no `address_offset`, no semaphores.
- Kernels are fully Device 2.0 already (`Noc`, `CircularBuffer` wrapper, `UnicastEndpoint`, `CoreLocalMem`) — the
  kernel-side port is a binding-layer swap, not an idiom rewrite. NoC transfer idiom stays as-is.

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — both factories.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: `create_program_artifacts` replaces `create_descriptor` inside each existing factory
  struct (the `program_factory_t` variant stays as-is). Signature matches the legacy one
  (`Params`, `Inputs`, `Tensor& output`), so no device-op-class edits are expected. No pybind edits (nothing binds
  `create_descriptor`).

## Planned Spec Shape

Default: 1:1 with legacy. Both variants share the same shape; per-variant differences are the kernel source, the
CTA set, and the vararg count.

### Variant: default
- **KernelSpecs**: 2 — `READER{"reader"}` (phase CTA = 1) and `WRITER{"writer"}` (phase CTA = 2), both of source
  `reader_tm_tile_layout_nlp_concat_heads_decode.cpp` (preserved multiplicity of the two legacy `KernelDescriptor`s).
  - `hw_config`: `create_reader_datamovement_config(device->arch())` / `create_writer_datamovement_config(device->arch())`
    (TTNN arch-agnostic helpers; resolved legacy triples match the reader/writer defaults exactly — see inventory).
  - `opt_level`: nothing to set — legacy resolved O2 on both DM kernels; Metal 2.0 `CompilerOptions` default is O2.
  - CTAs (named): `element_size`, `subtile_line_bytes`, `head_size`, `batch`, `head_size_num_tiles`,
    `phases_to_read` (1 vs 2 — the one CTA differing between the two specs), `num_x`, `num_y`.
    The CB-index CTA (legacy slot 2) is dropped — replaced by the DFB binding.
  - RTA schema: `runtime_arg_names = {"in_tile_offset_by_head"}` (named after the kernel-side variable).
  - Varargs: `advanced_options.num_runtime_varargs = in_num_cores_x + in_num_cores_y` — the NoC coordinate
    tables, a genuine indexed collection (CTA-driven counts, indexed by a data-driven cursor). Layout:
    x coords at vararg indices `[0, num_x)`, y coords at `[num_x, num_x + num_y)` (legacy arg-buffer order preserved).
  - Tensor bindings: `{INPUT, "input"}` on **both** specs (Case 2 raw-pointer consumption via the
    `TensorAccessor::get_bank_base_address` bridge kernel-side).
  - DFB bindings: `Q_OUT` bound with accessor `"q_out"` on both specs — READER as PRODUCER, WRITER as CONSUMER (1P+1C).
- **DataflowBufferSpecs**: 1 — `Q_OUT{"q_out"}`: `entry_size = single_tile_size`, `num_entries = q_num_tiles`,
  `data_format_metadata = data_format`, `tile_format_metadata` unset (legacy `format_descriptors[0].tile` unset),
  `borrowed_from = OUTPUT` (replaces `CBDescriptor::buffer = output.buffer()`).
- **SemaphoreSpecs**: none — legacy has no semaphores.
- **TensorParameters**: 2 —
  - `INPUT{"input"}` (spec = `input_tensor.tensor_spec()`), bound by both kernels;
  - `OUTPUT{"output"}` (spec = `output.tensor_spec()`), **borrow-only** (no kernel `TensorBinding`; named by
    `Q_OUT.borrowed_from`, which the validator counts as a use).
- **WorkUnitSpecs**: 1 — `"main"`, kernels `{READER, WRITER}`, `target_nodes = q_cores`.
- **Op-owned tensors**: none (audit-confirmed).
- **ProgramRunArgs**: `KernelRunArgs` per kernel — per-node `in_tile_offset_by_head` (same value into both kernels'
  tables, mirroring the legacy shared `rt_args`), per-node `advanced_options.runtime_varargs` = the concatenated
  x‖y coordinate vector (identical for every node — host builds it once, as legacy did);
  `tensor_args = {{INPUT, input mesh tensor}, {OUTPUT, output mesh tensor}}`.

### Variant: subcoregrids
Identical shape, with:
- source `reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp`;
- CTAs (named): `element_size`, `subtile_line_bytes`, `head_size`, `batch`, `head_size_num_tiles`,
  `phases_to_read` (1 vs 2), `in_num_cores`, `face_h`, `face_hw`;
- varargs: `num_runtime_varargs = 2 * in_num_cores` — x coords at `[0, in_num_cores)`, y at
  `[in_num_cores, 2*in_num_cores)`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| default: reader_desc + writer_desc of `reader_tm_tile_layout_nlp_concat_heads_decode.cpp`, both over `q_cores` | READER + WRITER, same source | one ("main", both kernels, `q_cores`) | `Q_OUT`: READER = PRODUCER, WRITER = CONSUMER (cosmetic 1P+1C; both are sync-free raw writers) |
| subcoregrids: reader_desc + writer_desc of `..._subcoregrid.cpp`, both over `q_cores` | READER + WRITER, same source | one ("main", both kernels, `q_cores`) | `Q_OUT`: READER = PRODUCER, WRITER = CONSUMER |

Endpoint census (re-derived per the endpoint-assignment procedure, not transcribed): per node, exactly **two**
distinct kernel instances touch `Q_OUT` (the Reader-config and Writer-config instances of the one source); both are
**role-free** (only `get_write_ptr()` raw peeks — grep of both kernels confirms zero
`reserve_back`/`push_back`/`wait_front`/`pop_front` and no `evil_set_*`). Two touchers, none role-locked →
**1P+1C**, matching the brief. No self-loop, no `allow_instance_multi_binding`, no dead CBs.

## Dropped Plumbing

For each legacy RTA / CTA that disappears in the port:

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| default factory RTA slot 1 (`nlp_concat_heads_decode_program_factory.cpp:130`); kernel arg 1 (`reader_tm_tile_layout_nlp_concat_heads_decode.cpp:18`) | `rt_args.push_back(in_buffer)` (`Buffer*` → base address); kernel `q_start_addr = get_arg_val<uint32_t>(1)` | `TensorBinding{INPUT, "input"}` on both KernelSpecs; kernel-side **Case 2 bridge**: `TensorAccessor(tensor::input).get_bank_base_address()` — raw walk unchanged |
| subcoregrids factory RTA slot 1 (`..._subcoregrids_program_factory.cpp:137`); kernel arg 1 (`..._subcoregrid.cpp:16`) | same | same |
| default factory CTA slot 2 (`nlp_concat_heads_decode_program_factory.cpp:85`); kernel CTA 2 (kernel line 22) | `q_output_cb_index` (= `CBIndex::c_16`) magic CB index | `DFBBinding` (`Q_OUT`, accessor `"q_out"`); kernel constructs `DataflowBuffer dfb_q_out(dfb::q_out)` |
| subcoregrids factory CTA slot 2 (`..._subcoregrids_program_factory.cpp:91`); kernel CTA 2 (kernel line 20) | same | same |
| default factory CTA slots 0,1,3–8 (positional) | positional `compile_time_args` vector | named CTAs: `element_size`, `subtile_line_bytes`, `head_size`, `batch`, `head_size_num_tiles`, `phases_to_read`, `num_x`, `num_y` (slot 6's copied-vector overwrite `writer_compile_time_args[6] = 2` becomes the per-spec named value) |
| subcoregrids factory CTA slots 0,1,3–9 (positional) | positional vector | named CTAs: `element_size`, `subtile_line_bytes`, `head_size`, `batch`, `head_size_num_tiles`, `phases_to_read`, `in_num_cores`, `face_h`, `face_hw` |
| default factory RTA slot 0; kernel arg 0 | positional RTA `in_tile_offset_by_batch` | named RTA `in_tile_offset_by_head` (per-node) |
| default factory RTA slots 2.. (`rt_args.append(noc_x_coords)` / `.append(noc_y_coords)`); kernel `get_arg_addr(2)` pointer walk (lines 31-32) | positional RTA tail read via raw `tt_l1_ptr uint32_t*` | runtime **varargs** (`num_runtime_varargs`, per-node values); kernel indexes `get_vararg(i)` / `get_vararg(num_x + i)` |
| subcoregrids RTA slots 0, 2.. | same shapes | same (vararg count `2 * in_num_cores`; y block at `get_vararg(in_num_cores + i)`) |

No `TensorAccessorArgs` plumbing, no page-size 3rd-arg CTAs/RTAs, no semaphore-ID RTAs exist in this op.

## Applied Patterns

- **Two-toucher DFB → assign 1P+1C (dual-instance work-split)**: `Q_OUT` in both factories — the same source
  instantiated Reader-config + Writer-config over one grid, both sync-free raw writers; READER bound PRODUCER,
  WRITER bound CONSUMER (cosmetic on Gen1).
- **Borrowed-memory DFB**: `Q_OUT.borrowed_from = OUTPUT`, replacing `CBDescriptor::buffer = output.buffer()`;
  backing L1 address resolves from the `OUTPUT` `TensorArgument` at runtime. OUTPUT is a borrow-only
  `TensorParameter` (validator-sanctioned use without a kernel binding).
- **Case 2 (raw pointer) tensor binding**: input base pulled kernel-side via the sanctioned
  `TensorAccessor::get_bank_base_address` bridge; the hand-rolled remote-NoC walk stays byte-identical.
- **Caution: Avoid varargs — genuine vararg case**: the NoC coordinate blocks are indexed collections with
  CTA-driven counts and data-driven cursors (`in0_mcast_noc_x[qkv_x]`), so they ride `num_runtime_varargs`;
  the two leading scalars are named/bound instead (arg 0 → named RTA, arg 1 → tensor binding). Retained vararg
  use is reported in the port report.
- **Multi-variant factories**: two factories ported in one pass, each with its own `create_program_artifacts`;
  spec-name constants are declared function-locally in each factory (unity-build hygiene: this op's two factory
  `.cpp`s share a translation unit under unity builds, so no anonymous-namespace name constants).

## Deferred / Flagged

- **RTA→CRTA candidates (not converted — dispatch-semantics change)**: the vararg coordinate blocks are identical
  on every node (host builds them once), so `num_common_runtime_varargs` would be the more efficient form; kept as
  per-node runtime varargs to mirror legacy per-core RTA dispatch. Noted for a later cleanup pass, with the same
  candidate status for `in_tile_offset_by_head`'s *schema* (it is genuinely per-core, so it stays a per-node RTA).
- No new audit-missed findings surfaced during planning.
