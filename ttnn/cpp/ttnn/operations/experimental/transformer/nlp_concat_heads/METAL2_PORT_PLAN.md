# Port Plan — nlp_concat_heads

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`, ported from the
legacy `ProgramDescriptor` API (`ProgramDescriptorFactoryConcept`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `NLPConcatHeadsProgramFactory::create_descriptor`
  (`device/nlp_concat_heads_program_factory.cpp:19`), returning `ProgramDescriptor`.
- Factory methods live in a proper `program_factory_t` variant
  (`program_factory_t = std::variant<NLPConcatHeadsProgramFactory>`,
  `device/nlp_concat_heads_device_operation.hpp:24`) — **not** the direct-descriptor shape; no
  device-op restructure needed.
- Variants: single factory, **two internal config branches** selected by `input.is_sharded()`:
  **interleaved** and **sharded**. Both branches convert together in this port (they share the one
  `create_descriptor` body and the atomic unit is the factory).
- Custom `compute_program_hash`: **none** — default reflection-based hash (verified by audit and by
  grep of the device-op).
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no pybound `create_descriptor`
  (`nlp_concat_heads_nanobind.cpp` binds only the public op function).

### Kernels

#### Config branch: interleaved (`!in_sharded`)

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` (own; sole consumer, verified by audit) | `all_cores` (from `split_work_to_cores`) | 0:`in0_h_tiles`, 1:`in0_w_tiles`, 2:`in0_c`, 3:`in0_HtWt`, 4+:`TensorAccessorArgs(*in0_buffer)` (factory:111–117) | none | per-core (factory:198–205): 0:`in0_buffer` (**Buffer\***), 1:`num_blocks_per_core`, 2:`in0_h_dim`, 3:`in0_tensor_tile_id` | none | none | O2 (resolved default; no `opt_level` set anywhere in the factory) | `ReaderConfigDescriptor{}` → reader default (RISCV_1 / NOC_0 / DM_DEDICATED_NOC) |
| writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed**, eltwise/unary) | `all_cores` | 0:`src0_cb_index` (=0, magic CB index), 1+:`TensorAccessorArgs(*out_buffer)` (factory:118–119) | none | per-core (factory:207–213): 0:`out_buffer` (**Buffer\***), 1:`num_blocks_per_core * per_tensor_tiles` (=num_pages), 2:`num_blocks_written * per_tensor_tiles` (=start_id) | none | none (neither `OUT_SHARDED` nor `BACKWARDS`) | O2 (resolved default) | `WriterConfigDescriptor{}` → writer default (RISCV_0 / NOC_1 / DM_DEDICATED_NOC) |

#### Config branch: sharded (`in_sharded`)

One kernel source, `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`
(own; sole consumer, verified by audit), instantiated **twice** (dual-instance work-split over the
same grid; factory:86–109 — reader instance copies the CTA vector, writer instance takes it by
move).

| unique_id | source | core_ranges | CTAs (positional, shared vector) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader instance | `..._sharded.cpp` | `all_cores` (= input shard grid) | 0:`src0_cb_index`(=0), 1:`out_cb_index`(=16), 2:`in0_h_tiles`, 3:`in0_w_tiles*single_tile_size` (kernel: `head_dim_size_bytes`), 4:`num_blocks_per_core_group_1*in0_w_tiles*single_tile_size` (kernel: `out_row_size_bytes`), 5:`num_blocks_per_core_group_1*in0_HtWt` (kernel: `block_size`) | none | same on every core (factory:173–180): 0:`nheads_first_risc`, 1:`0` (`start_read_offset_bytes`), 2:`0` (`start_write_offset_bytes`) | none | none | O2 (resolved default) | `ReaderConfigDescriptor{}` → reader default |
| writer instance | same source | `all_cores` | identical CTA vector | none | same on every core (factory:181–187): 0:`nheads_second_risc`, 1:`nheads_first_risc*in0_HtWt*single_tile_size`, 2:`nheads_first_risc*in0_w_tiles*single_tile_size` | none | none | O2 (resolved default) | `WriterConfigDescriptor{}` → writer default |

No compute kernel in either branch (pure data-movement op).

### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer (borrowed) |
|---|---|---|---|---|---|---|
| 0 (interleaved) | `2 * per_tensor_tiles * single_tile_size` (double-buffered) | `all_cores` | `cb_data_format` (from input dtype) | `single_tile_size` | not set | `nullptr` (own allocation) |
| 0 (sharded) | `per_tensor_tiles * single_tile_size` (per-shard tiles; no double buffer) | `all_cores` | `cb_data_format` | `single_tile_size` | not set | **`in0_buffer`** (borrowed-memory CB, factory:150) |
| 16 (sharded, only if `out_sharded`) | `per_tensor_tiles * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | not set | **`out_buffer`** (borrowed-memory CB, factory:153–165) |

No GlobalCircularBuffer anywhere (audit-verified). `address_offset` unset on both descriptors.

### Semaphores
none

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `TensorAccessorArgs(*in0_buffer)` appended to reader CTAs (factory:117); kernel `TensorAccessorArgs<4>()` + `TensorAccessor(in0_args, in0_tensor_addr)` (reader kernel:27,31) | input | reader RTA 0 (`in0_buffer` `Buffer*`) |
| `TensorAccessorArgs(*out_buffer)` appended to writer CTAs (factory:119); kernel `TensorAccessorArgs<1>()`-equivalent in borrowed writer (fork consumes `tensor::dst`) | output | writer RTA 0 (`out_buffer` `Buffer*`) |

Neither accessor passes a 3rd (page-size) constructor argument (audit-verified).
Sharded branch: no tensor accessors, no tensor-address RTAs.

### Work split
- Interleaved: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` (factory:61–68) →
  `(num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1,
  num_blocks_per_core_group_2)`. Per-group variation flows **only through RTAs**
  (`num_blocks_per_core`, tile ids) — CTAs identical across groups, so a single `KernelSpec` per
  kernel remains faithful (no per-group CTA multiplicity to preserve). RTA-assignment core order:
  `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major=false)`.
- Sharded: no `split_work_to_cores` — `all_cores` = input shard grid, `core_group_1 = all_cores`,
  `num_blocks_per_core_group_1 = shard_height / padded_height`; the *intra-core* work split between
  the two kernel instances is `nheads_first_risc = div_up(nheads, 2)` / `nheads_second_risc = rest`
  (factory:169–170). RTA emission order: `corerange_to_cores(all_cores, num_cores, row_wise=true)`.

### Shared kernels
- `writer_unary_interleaved_start_id.cpp` (eltwise/unary) — **borrowed**, broadly shared.
  A checked-in `_metal2` fork exists beside it:
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  → **rung 1: reuse the fork, read-only.** Fork's binding vocabulary (the constraint this factory
  conforms to): DFB accessor **`dfb::out`** (CONSUMER; the kernel drains it), tensor accessor
  **`tensor::dst`**, named RTAs **`num_pages`**, **`start_id`**; optional gates `OUT_SHARDED` /
  `BACKWARDS` — neither needed on this op's interleaved path. Fit confirmed: this op's writer RTAs
  map 1:1. The duplicate fork in `copy/typecast/` is ignored per the fork's own header note.
- Both own kernels (`reader_tm_tile_layout_nlp_concat_heads*.cpp`) are bound **only** by this op's
  factory (audit-verified) — convert in place, no fork.

### Flags
- **Dead FIFO sync in the sharded kernel** (`reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:35–36`):
  `cb_in0.reserve_back(block_size)` (self-annotated `// Redundant`) and
  `cb_out0.reserve_back(block_size)`; the paired `cb_out0.push_back(block_size)` is already
  commented out (line 62). Full-capacity reserve on an empty borrowed CB — a no-op at runtime, but
  by the strict census it locks **both** instances to the PRODUCER role on **both** sharded CBs.
  Audit open question #1; disposition below depends on the invoker's answer.
- `single_tile_size_bytes` in the sharded kernel (line 30) is computed but never used — ported
  as-is (dead local preserved).
- Latent broken config (audit "Misc anomalies"): validation permits sharded-in + interleaved-out,
  but the factory then creates no `cb16` while the sharded kernel touches CB index 16
  unconditionally. Pre-existing bug, not a port target; Metal 2.0 consequence recorded under
  Deferred / Flagged.
- Stale comments in the factory (`// 142`, `// Output shape is: [B, 1, s, 4544]`,
  `Grayskull Device Setup` banner) — preserved per comment rules; noted in the report.
- Both reader kernels label their RTAs `// WRITER RUNTIME ARGS` — cosmetic copy-paste, preserved.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: `tensor_args_t = Tensor` (the input tensor directly), so
  `create_program_artifacts(const NlpConcatHeadsParams&, const Tensor& input, Tensor& output)`
  keeps the legacy `create_descriptor` parameter shape. Extract `mesh_tensor()` from both tensors
  at entry. The two config branches stay an if/else inside `create_program_artifacts`.

## Planned Spec Shape

### Branch: interleaved

- **KernelSpecs** (2):
  - `READER{"reader"}` — source: own reader kernel (converted in place).
    - `dfb_bindings`: `{IN0, "in0", PRODUCER}` (kernel: `dfb::in0`).
    - `tensor_bindings`: `{INPUT, "src"}` (kernel: `TensorAccessor(tensor::src)`).
    - `compile_time_args` (named): `in0_h_tiles`, `in0_w_tiles`, `in0_c`, `in0_HtWt`.
    - `runtime_arg_schema`: `num_blocks`, `in0_h_dim`, `in0_tensor_tile_id`.
    - `hw_config = create_reader_datamovement_config(arch)` (legacy resolved triple == reader default).
    - `opt_level`: leave default (legacy resolved O2 == Metal 2.0 default O2 for DM).
  - `WRITER{"writer"}` — source: existing `_metal2` fork (reused, read-only).
    - `dfb_bindings`: `{IN0, "out", CONSUMER}` (fork-owned name `dfb::out`).
    - `tensor_bindings`: `{OUTPUT, "dst"}` (fork-owned name `tensor::dst`).
    - `compile_time_args`: none (legacy slot 0 was the magic CB index; the accessor args become the
      tensor binding).
    - `runtime_arg_schema`: `num_pages`, `start_id` (fork-owned names).
    - `hw_config = create_writer_datamovement_config(arch)`.
- **DataflowBufferSpecs** (1): `IN0{"in0"}` — `entry_size = single_tile_size`,
  `num_entries = 2 * per_tensor_tiles`, `data_format_metadata = cb_data_format`,
  no `tile_format_metadata` (legacy `.tile` unset), own allocation.
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `INPUT{"input"}` (spec from input mesh tensor), `OUTPUT{"output"}`
  (spec from output mesh tensor). Paired `tensor_args` entries for both.
- **WorkUnitSpecs** (1): `{reader, writer}` over `all_cores` (single WU: per-group variation is
  RTA-only; both kernels run on every active core).

### Branch: sharded

- **KernelSpecs** (2, same source — preserved dual-instance multiplicity):
  - `READER{"reader"}` — Reader-config instance.
    - `dfb_bindings`: see CB-endpoint disposition below.
    - `compile_time_args` (named, shared values): `in0_h_tiles`, `head_dim_size_bytes`,
      `out_row_size_bytes`, `block_size` (legacy slots 2–5; slots 0–1 were CB indices → dropped).
    - `runtime_arg_schema`: `nheads`, `start_read_offset_bytes`, `start_write_offset_bytes`.
    - `hw_config = create_reader_datamovement_config(arch)`.
  - `WRITER{"writer"}` — Writer-config instance; identical CTAs and schema, own RTA values;
    `hw_config = create_writer_datamovement_config(arch)`.
- **DataflowBufferSpecs**:
  - `IN0{"in0"}` — `entry_size = single_tile_size`, `num_entries = per_tensor_tiles` (shard
    tiles), `data_format_metadata = cb_data_format`, **`borrowed_from = INPUT`**.
  - `OUT0{"out0"}` — same sizes, **`borrowed_from = OUTPUT`**, declared **only when
    `out_sharded`** (translation of the existing host conditional at factory:153; not dropped).
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `INPUT`, `OUTPUT` — no kernel `TensorBinding`s in this branch; both are
  referenced via `borrowed_from`, which counts as use per the validator's borrow-only exception
  (`migration_guide.md` — TensorParameter validator note). Paired `tensor_args` entries for both.
- **WorkUnitSpecs** (1): `{reader, writer}` over `all_cores`.

Kernel accessor names in the sharded kernel: `dfb::in0`, `dfb::out0` (both instances bind the same
accessor names — same compiled source, shared CTA vector preserved as identical named CTA tables).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| sharded: reader_desc + writer_desc, one source (`..._sharded.cpp`), same `all_cores` grid, shared CTA vector, per-instance RTAs | `READER` + `WRITER`, same source, identical named CTAs | one WU over `all_cores` (both instances on every node — a **two-toucher same-grid** work split, not the disjoint-node case) | `IN0`: one instance PRODUCER, other CONSUMER; `OUT0`: likewise (see disposition below) |

Interleaved branch: none — no same-source multiplicity in legacy.

### CB-endpoint disposition (re-derived census)

- `IN0` / interleaved: reader FIFO-produces (`reserve_back`/`push_back`, reader kernel:47,59; its
  `get_write_ptr` peek at line 41 rides the same binding), writer fork FIFO-consumes
  (`wait_front`/`pop_front`). **Legal 1:1** — 1 locked producer + 1 locked consumer. Matches brief.
- `IN0` + `OUT0` / sharded: exactly two touchers each — the two same-source instances. Both
  raw-peek (`get_read_ptr`/`get_write_ptr` + offset, kernel:43–44) and both issue the **dead**
  `reserve_back` (kernel:35–36; never paired with a push/pop).
  - **If the invoker approves stripping the dead `reserve_back` pair** (audit question #1 —
    ops-team-scoped 3-line cleanup, off the porter's whitelist without approval): both touchers are
    role-free → **1P+1C** (reader-instance PRODUCER, writer-instance CONSUMER; labels cosmetic on
    Gen1). This is the audit's recommended path and the disposition this plan assumes.
  - **If the lines must stay**: strictly-per-census both instances are locked producers →
    2-producers/0-consumers, which the validator cannot accept even with the multi-binding flag
    (≥1 CONSUMER is required); the only workable shapes are (a) 1P+1C with the CONSUMER-bound
    instance keeping its `reserve_back` (runtime-identical on Gen1 — the FIFO ops lower to plain
    CB pointer ops any RISC may issue; the host label drives nothing the kernel invokes), noted in
    the report, or (b) capitulation on the sharded branch. Decision recorded after the invoker
    answers; the port does not silently delete the lines.
  - **RESOLVED (invoker, 2026-08-27): strip approved.** Lines 35–36 (and the already-commented
    `push_back` at line 62) removed; both sharded DFBs bind clean 1P+1C. Recorded in
    `METAL2_PORT_REPORT.md` → Findings.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory:201 / reader kernel:17 | reader RTA 0 = `in0_buffer` (`Buffer*`); kernel `get_arg_val<uint32_t>(0)` → accessor addr | `TensorBinding{INPUT, "src"}`; kernel `TensorAccessor(tensor::src)` |
| factory:210 / fork | writer RTA 0 = `out_buffer` (`Buffer*`) | `TensorBinding{OUTPUT, "dst"}` (fork already consumes `tensor::dst`) |
| factory:117 / reader kernel:27 | `TensorAccessorArgs(*in0_buffer).append_to(cta)` / `TensorAccessorArgs<4>()` | binding mechanism end-to-end |
| factory:119 | `TensorAccessorArgs(*out_buffer).append_to(cta)` | binding mechanism end-to-end |
| factory:118 / legacy writer CTA 0 | `src0_cb_index` (=0) magic CB index | `DFBBinding{IN0, "out", CONSUMER}` |
| factory:88–89 / sharded kernel:22–23 | sharded CTA slots 0–1 = `src0_cb_index`, `out_cb_index` (magic CB indices) | `DFBBinding`s (`dfb::in0`, `dfb::out0`) |
| reader kernel:29 | hardcoded `constexpr uint32_t cb_id_in0 = 0;` | `dfb::in0` binding token |
| factory:111–116 / reader kernel:23–26 | positional CTA slots 0–3 | named CTAs `in0_h_tiles`, `in0_w_tiles`, `in0_c`, `in0_HtWt` |
| factory:90–93 / sharded kernel:24–28 | positional CTA slots 2–5 | named CTAs `in0_h_tiles`, `head_dim_size_bytes`, `out_row_size_bytes`, `block_size` |
| reader kernels:16–20 / 15–18 | positional RTAs via `get_arg_val<uint32_t>(N)` | named RTAs (schemas above) |

No page-size 3rd-arg CTAs/RTAs (none exist). No semaphore-ID RTAs (no semaphores). No varargs
anywhere (every arg is a distinct nameable scalar; the sharded reader instance's two zero RTAs are
real named fields that happen to be 0).

## Applied Patterns

- [Two-toucher DFB → assign 1P+1C (dual-instance work-split)] (port_patterns.md): both sharded
  DFBs, pending the dead-sync answer.
- [Borrowed-memory DFBs] (migration_guide.md — DataflowBufferSpec): `IN0`/`OUT0` sharded via
  `borrowed_from`; borrow-only `TensorParameter`s count as used.
- [Conditional DFB spec] — `OUT0` declared only under `out_sharded`, translating the existing host
  conditional (no kernel-side `#ifdef` added: within every *intended* config the sharded kernel
  always binds both DFBs; see Deferred/Flagged for the broken-config consequence).
- [Caution: Porting a shared kernel — rung 1 reuse]: interleaved writer binds the existing
  `writer_unary_interleaved_start_id_metal2.cpp` fork and conforms to its vocabulary.
- [Multi-variant factories]: interleaved/sharded branch inside `create_program_artifacts`.
- [Unity-build hygiene]: spec-name constants declared function-locally in the factory body.

## Deferred / Flagged

- **Dead `reserve_back` pair** (sharded kernel:35–36): resolution awaits the invoker's answer to
  audit question #1 (strip with approval → clean 1P+1C; keep → 1P+1C with a CONSUMER-bound
  `reserve_back`, reported).
- **Latent broken config** (sharded-in + interleaved-out): legacy silently ran the sharded kernel
  against an unconfigured CB 16 and never wrote the output. Post-port this config cannot generate a
  `dfb::out0` token (no DFB is declared, faithfully mirroring the legacy conditional), so it fails
  loudly at JIT instead of silently producing garbage. Behavior difference confined to a config the
  audit already classified as broken; routed to the report for the ops team.
- **`get_tile_size(cb_id)` mapping**: both own kernels read the CB page size via the DM-side free
  helper into a `const` (not `constexpr`) local; ported to the DFB member getter form. Since these
  are DM kernels (no `chlkc_descriptors.h` guarantee), the §B mapping `get_entry_size()` is used —
  byte-identical here (page size == tile size), and the same call the writer fork already uses.
