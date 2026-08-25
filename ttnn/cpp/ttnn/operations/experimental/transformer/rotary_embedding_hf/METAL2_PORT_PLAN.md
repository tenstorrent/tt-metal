# Port Plan — rotary_embedding_hf

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf`, ported from the
descriptor API (`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

**Port order (coordinated with the sibling `rotary_embedding` port running in the same tree):**
1. `RotaryEmbeddingHfMultiCoreSharded` (decode) — first; binds no shared kernel.
2. `RotaryEmbeddingHfMultiCore` (prefill) — second; its single-tile path binds the sibling op's
   `rotary_embedding_single_tile.cpp`, whose `_metal2` fork is created by the sibling port. (This order
   was followed; the fork existed by the time the prefill factory was ported and was reused — rung 1.)

## Legacy Inventory

*Filled in during the inventory step.*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — both factories define
  `static ProgramDescriptor create_descriptor(params, inputs, output)` inside a proper
  `program_factory_t` variant (`rotary_embedding_hf_device_operation.hpp:19`), NOT the
  direct-descriptor shape (no exception-3 restructure needed).
- Variants: `RotaryEmbeddingHfMultiCore` (prefill; selected when `!is_decode_mode`) and
  `RotaryEmbeddingHfMultiCoreSharded` (decode; `is_decode_mode`). Each factory internally selects one of
  two descriptor-builder functions on `input.padded_shape()[-1] / TILE_WIDTH == 1`
  (single-tile vs multi-tile). This is a *host-side* branch: each builder emits fixed kernel sources, so
  the "runtime kernel-source selection" axis is exactly the single-tile/multi-tile × (`in_sharded` for the
  prefill reader) fan-out enumerated below — all sources convert with their factory.
- Custom `compute_program_hash`: **none** — default reflection-based hash (audit confirmed: no
  `compute_program_hash`, `attribute_values`, or `to_hash` in the op directory).
- `override_runtime_arguments` / `get_dynamic_runtime_args` / pybound `create_descriptor`: none.

> Both factories are multi-config. Per-config blocks below; Shared kernels and Flags are top-level.

### Variant: RotaryEmbeddingHfMultiCoreSharded — single-tile decode (`head_dim == TILE_WIDTH`)

Source: `create_single_tile_decode_descriptor` (sharded factory `:21-212`). No runtime args at all; no
defines; no semaphores. `all_cores = shard_spec->grid.bounding_box()` (single CoreRange).

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs/CRTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_rotary_embedding_hf_single_tile_sharded.cpp` | all_cores | `[0]=trans_mat_cb_index(c_3)` | none | none | O2 (unset DM) | `ReaderConfigDescriptor{}` |
| compute | `device/kernels/compute/rotary_embedding_hf_single_tile_sharded.cpp` | all_cores | `[0..7]=cb idx c_0,c_1,c_2,c_3,c_24,c_25,c_26,c_16; [8]=n_heads_per_batch_t; [9]=batch_per_core` | none | none | **O3** (unset compute) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` (only those two fields set — see Flags) |

#### CBs
| index | total_size | core_ranges | data_format | page_size | borrowed buffer |
|---|---|---|---|---|---|
| c_0 input | `n_heads_t * input_tile_size` | all_cores | input fmt | input tile size | **`input.buffer()`** (always) |
| c_1 cos | `batch_per_core * cos_tile_size` | all_cores | cos fmt | cos tile size | **`cos.buffer()`** |
| c_2 sin | `batch_per_core * sin_tile_size` | all_cores | sin fmt | sin tile size | **`sin.buffer()`** |
| c_3 trans_mat | 1 tile | all_cores | Bfp8_b/Float32/Float16_b derived from input fmt | its tile size | — |
| c_24 rotated_interm | 1 tile (`head_dim_t=1`) | all_cores | input fmt | input tile size | — |
| c_25 cos_interm | 1 tile | all_cores | input fmt | input tile size | — |
| c_26 sin_interm | 1 tile | all_cores | input fmt | input tile size | — |
| c_16 output | `n_heads_t * output_tile_size` | all_cores | output fmt | output tile size | **`output.buffer()`** |

#### Semaphores
none

#### Tensor accessors
none — the sharded factory has zero address RTAs and zero `TensorAccessorArgs`; all four io tensors are
delivered as borrowed CB backings only.

#### Work split
n/a — every core in `shard_spec->grid.bounding_box()` runs both kernels identically
(`batch_per_core = ceil(batch / min(batch, num_cores))` is a CTA, same on all cores).

#### CB endpoint census (re-derived from kernel reads; agrees with audit)
- c_0/c_1/c_2/c_16, c_24/c_25/c_26: **compute-only** (compute reserve/push/wait/pops the borrowed
  inputs itself; output tiles are produced into resident memory and never drained) → **self-loop** on compute.
- c_3 trans_mat: reader FIFO-P (raw fill inside reserve/push), compute FIFO-C (`wait_front(1)` at
  `rotary_embedding_hf_single_tile_sharded.cpp:37`, never pops — locked consumer) → 1P+1C.

### Variant: RotaryEmbeddingHfMultiCoreSharded — multi-tile decode

Source: `create_multi_tile_decode_descriptor` (sharded factory `:214-408`). Same structure; differences only:

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs/CRTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_rotary_embedding_hf_sharded.cpp` | all_cores | `[0]=src_scalar_cb_index(c_3); [1]=bfloat16(-1.0f) bit pattern` | none | none | O2 | `ReaderConfigDescriptor{}` |
| compute | `device/kernels/compute/rotary_embedding_hf_sharded.cpp` | all_cores | `[0..7]=cb idx c_0,c_1,c_2,c_3,c_24,c_25,c_26,c_16; [8]=head_dim_t (kernel: Wt); [9]=n_heads_t (kernel: Ht — read then `(void)`-discarded, DEAD); [10]=n_heads_per_batch_t; [11]=batch_per_core` | none | none | **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |

#### CBs
| index | total_size | data_format | borrowed buffer |
|---|---|---|---|
| c_0 input | `n_heads_t*head_dim_t * input_tile_size` | input fmt | **`input.buffer()`** |
| c_1 cos | `head_dim_t*batch_per_core * cos_tile_size` | cos fmt | **`cos.buffer()`** |
| c_2 sin | `head_dim_t*batch_per_core * sin_tile_size` | sin fmt | **`sin.buffer()`** |
| c_3 scalar | 1 tile | **Float16_b always** | — |
| c_24 rotated_interm | `head_dim_t` tiles, input fmt | input fmt | — |
| c_25 cos_interm | `head_dim_t` tiles | **cos fmt** (differs from single-tile decode, which uses input fmt) | — |
| c_26 sin_interm | `head_dim_t` tiles | **sin fmt** | — |
| c_16 output | `n_heads_t*head_dim_t * output_tile_size` | output fmt | **`output.buffer()`** |

(all on all_cores; semaphores none; tensor accessors none; work split n/a — same as single-tile decode)

#### CB endpoint census
- c_0/c_1/c_2/c_16 + interm c_24/25/26: compute-only → **self-loop** on compute.
- c_3 scalar: reader FIFO-P (writes -1.0 bf16, its only job), compute FIFO-C (waits at start
  `rotary_embedding_hf_sharded.cpp:44`, pops at end `:146`) → 1P+1C.

### Variant: RotaryEmbeddingHfMultiCore — multi-tile prefill

Source: `create_multi_tile_descriptor` (multi_core factory `:326-630`).

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per core) | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_rotary_embedding_hf_interleaved.cpp` | all_cores | `[0..4]=cb idx c_0,c_1,c_2,c_3,c_4; [5]=bfloat16(-1.0f); [6]=Ht; [7]=Wt; [8]=HtWt; [9]=half_Wt;` then `TensorAccessorArgs(src)`, `(cos)`, `(sin)` appended | `{src_buffer*, cos_buffer*, sin_buffer*, num_rows_per_core, num_tiles_written, num_tiles_written/Wt%Ht, cos_sin_start_id}` (slots 0–2 are `Buffer*` address bindings) | none | O2 | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_rotary_embedding_hf_interleaved.cpp` | all_cores | `[0]=output_cb_index(c_16);` then `TensorAccessorArgs(dst)` | `{dst_buffer*, num_rows_per_core*Wt, num_tiles_written}` | `OUT_SHARDED=1` iff `out_sharded` | O2 | `WriterConfigDescriptor{}` |
| compute_g1 | `device/kernels/compute/rotary_embedding_hf.cpp` | core_group_1 | `[0..8]=cb idx c_0,c_1,c_2,c_3,c_4,c_24,c_25,c_26,c_16; [9]=num_rows_per_core_group_1; [10]=Wt; [11]=half_Wt` | none | none | **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |
| compute_g2 (iff `core_group_2` nonempty) | same source | core_group_2 | same, with `[9]=num_rows_per_core_group_2` | none | none | **O3** | same |

#### CBs
| index | total_size | data_format | borrowed buffer |
|---|---|---|---|
| c_0 input | `num_input_tiles * input_tile_size` (`shard shape/TILE_HW` if sharded path else `2*Wt`) | input fmt | `input.buffer()` **iff `in_sharded`** |
| c_1 rotated_input | `2*Wt * input_tile_size` | input fmt | — |
| c_2 cos | `2*Wt * cos_tile_size` | cos fmt | — |
| c_3 sin | `2*Wt * sin_tile_size` | sin fmt | — |
| c_4 scalar | 1 tile | Float16_b | — |
| c_24/25/26 interm | 1 tile each | input/cos/sin fmt respectively | — |
| c_16 output | `num_output_tiles * output_tile_size` | output fmt | `output.buffer()` **iff `out_sharded`** |

(all on all_cores)

#### Semaphores
none

#### Tensor accessors
| host site | originating Tensor | kernel consumption |
|---|---|---|
| multi_core factory `:521` `TensorAccessorArgs(*src_buffer)` → reader CTA tail; RTA slot 0 `src_buffer` | input | reader `:33,39` `TensorAccessor(src_args, src_addr, get_tile_size(c_0))` |
| `:522` cos → CTA tail; RTA slot 1 | cos | reader `:34,42` |
| `:523` sin → CTA tail; RTA slot 2 | sin | reader `:35,45` |
| `:526` dst → writer CTA tail; RTA slot 0 | output | writer `:20,23` (whole accessor path `#ifdef`-dead under `OUT_SHARDED`) |

#### Work split
- Interleaved (no shard_spec): `split_work_to_cores(compute_with_storage_grid_size, num_rows, row_major=true)`
  → `(num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2)`.
- Sharded path (`in_sharded || out_sharded`): `all_cores = shard_spec.grid`, `core_group_1 = all_cores`,
  **`core_group_2 = empty`**, `num_rows_per_core_group_1 = shard_shape[0]/TILE_HEIGHT`.
  ⇒ **Borrowed CBs and two work units never coexist**: borrowing happens only on the sharded path, where
  group 2 is empty (single compute spec). This structurally avoids the known borrowed-DFB ≥2-work-unit bug.
- RTA loop: `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major)`; `num_tiles_written` accumulates
  `num_rows_per_core * Wt`; `cos_sin_start_id = num_tiles_written % HtWt`.

#### CB endpoint census
- c_0 input: reader FIFO-P (NoC read even when borrowed/in_sharded — the intentional self-aliasing copy,
  reader `:89-95`), compute FIFO-C → 1P+1C.
- c_1 rotated_input, c_2 cos, c_3 sin: reader FIFO-P, compute FIFO-C → 1P+1C.
- c_4 scalar: reader FIFO-P (raw fill), compute FIFO-C (`wait_front` at `rotary_embedding_hf.cpp:60`, never
  pops — locked consumer) → 1P+1C.
- c_24/25/26 interm: compute-only → self-loop.
- c_16 output: compute FIFO-P, writer FIFO-C (under `OUT_SHARDED` the writer only `wait_front`s — still the
  consumer) → 1P+1C.

### Variant: RotaryEmbeddingHfMultiCore — single-tile prefill (`Wt == 1`)

Source: `create_single_tile_prefill_descriptor` (multi_core factory `:20-324`). Reader source selected on
`in_sharded`; compute borrows the sibling op's kernel.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per core) | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|
| reader (interleaved, `!in_sharded`) | `device/kernels/dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id.cpp` | all_cores | `[0..3]=cb idx c_0,c_2,c_3,c_1(trans_mat); [4]=Ht; [5]=HtWt;` + `TensorAccessorArgs(src),(cos),(sin)` | `{src*, cos*, sin*, num_rows_per_core, num_tiles_written, num_tiles_written/Wt%Ht, cos_sin_start_id}` | none | O2 | `ReaderConfigDescriptor{}` |
| reader (`in_sharded`) | `device/kernels/dataflow/reader_rotary_embedding_hf_single_tile_interleaved_start_id_sharded.cpp` | all_cores | same `[0..5]` + `TensorAccessorArgs(cos),(sin)` (no src accessor) | `{cos*, sin*, num_rows_per_core, num_tiles_written/Wt%Ht, cos_sin_start_id}` | none | O2 | `ReaderConfigDescriptor{}` |
| writer | `writer_rotary_embedding_hf_interleaved.cpp` (shared with multi-tile prefill) | all_cores | `[0]=c_16` + `TensorAccessorArgs(dst)` | `{dst*, num_rows_per_core*Wt, num_tiles_written}` | `OUT_SHARDED=1` iff `out_sharded` | O2 | `WriterConfigDescriptor{}` |
| compute_g1 / compute_g2 | **BORROWED** `../rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp` | core_group_1 / core_group_2 | `[0..7]=cb idx c_0,c_2,c_3,c_1,c_24,c_25,c_26,c_16; [8]=num_rows_per_core_group_N` (this op supplies no `DECODE_MODE` define and only CTAs 0–8) | none | none | **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |

#### CBs
| index | total_size | data_format | borrowed buffer |
|---|---|---|---|
| c_0 input | `num_input_tiles * input_tile_size` | input fmt | `input.buffer()` iff `in_sharded` |
| c_1 trans_mat | 1 tile | derived (Bfp8_b/Float32/Float16_b) | — |
| c_2 cos | 1 tile | cos fmt | — |
| c_3 sin | 1 tile | sin fmt | — |
| c_24/25/26 interm | 1 tile each | input fmt (all three) | — |
| c_16 output | `num_output_tiles * output_tile_size` | output fmt | `output.buffer()` iff `out_sharded` |

#### Semaphores
none

#### Tensor accessors
Interleaved reader: src/cos/sin (reader `:87-99`); in_sharded reader: cos/sin only (`:85-86,99,102`);
writer: dst (`:20,23`). Same `Buffer*`-in-RTA delivery as multi-tile prefill.

#### Work split
Identical machinery to multi-tile prefill (same `split_work_to_cores` / shard_spec branch, `Wt=1`).
Same structural exclusion: borrowed CBs only on the sharded path, where core_group_2 is empty.

#### CB endpoint census
- c_0 input: reader FIFO-P (interleaved: NoC fill; in_sharded: bare cursor-advance
  `reserve_back(num_rows)/push_back(num_rows)` at `..._start_id_sharded.cpp:95-96` — still the producer),
  compute FIFO-C → 1P+1C.
- c_1 trans_mat: reader FIFO-P (raw fill), compute FIFO-C (`wait_front` only) → 1P+1C.
- c_2 cos / c_3 sin: reader FIFO-P, compute FIFO-C → 1P+1C.
- c_24/25/26: compute self-loop. c_16: compute FIFO-P, writer FIFO-C → 1P+1C.

### Shared kernels
- **Borrowed (cross-op):** `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding_single_tile.cpp`
  — bound by this op's single-tile-prefill compute (multi_core factory `:257-259`, `:273-275`) and by the
  sibling `rotary_embedding` op's own factory (`rotary_embedding_program_factory.cpp:399,416`, which also
  exercises a `DECODE_MODE` CTA/define path this op never enables).
  `grep -rln rotary_embedding_single_tile ttnn/cpp/ttnn/operations/` → real consumers: {`rotary_embedding`,
  `rotary_embedding_hf`} (other hits are METAL2_*.md artifacts).
  **Rung taken: rung 1 — reused the existing `_metal2` fork.** At inventory time no fork existed; by
  construction time the sibling `rotary_embedding` port (running in parallel in this tree) had created
  `rotary_embedding/device/kernels/compute/rotary_embedding_single_tile_metal2.cpp`, per the batch
  coordination (this port creates/edits nothing under `rotary_embedding/`). The sharded factory ported
  first; the prefill single-tile compute KernelSpec binds the fork and conforms to ITS vocabulary:
  - DFB accessor names: `in`, `cos`, `sin`, `trans_mat`, `rotated_in_interm`, `cos_interm`, `sin_interm`, `out`
  - Named args: `num_rows` (read `constexpr` → must be a CTA)
  - `DECODE_MODE`-gated names (`untilized_*`, `retilized_*`): not bound — this op emits no defines, so those
    tokens never enter name lookup.
- **Lent:** none — `grep -rl` over each of this op's 9 kernel filenames shows no other op binding them.
- **Intra-op:** `writer_rotary_embedding_hf_interleaved.cpp` is bound by both *configs* of the ONE
  `RotaryEmbeddingHfMultiCore` factory (single-tile + multi-tile prefill). Both configs convert together
  with that factory, so this is not a fork case. The sharded factory shares no kernel with the prefill
  factory — the two factories are independently portable (which the sharded-first ordering relies on).

### Flags
- **Dropped compute-config fields (both factories, all four configs):** the factories destructure all five
  resolved `get_compute_kernel_config_args` knobs but set only `math_fidelity` and `fp32_dest_acc_en` on
  `ComputeConfigDescriptor`; `math_approx_mode`, `dst_full_sync_en` (and the counterpart-less
  `packer_l1_acc`) are resolved-but-dropped. Legacy therefore always runs descriptor defaults
  `math_approx_mode=false`, `dst_full_sync_en=false`. Port must reproduce those *defaults*, not the resolved
  values (recipe: Hardware configuration → "Check for a dropped field before using the helper"):
  `sfpu_precision_mode = Precise`, `double_buffer_dest = true (= !false)`, regardless of what the caller's
  `compute_kernel_config` says.
- **Dead CTA:** sharded multi-tile compute `rotary_embedding_hf_sharded.cpp:25,29` reads CTA[9] (`Ht`, fed
  `n_heads_t`) then `(void)`-discards it. Named-arg conversion names it and keeps emitting it (faithful
  translation; no functional change). Recorded for the ops team in the report.
- **Inert writer plumbing under `OUT_SHARDED`:** writer RTAs 0/2 and its accessor are unused when
  `OUT_SHARDED` — preserved as-is (the `#ifdef` structure already handles it).
- **`(void)Ht`-style unused-value patterns aside, no unreferenced kernel files** — all 9 in-dir kernels are
  bound by some config (audit confirmed).
- **opt_level:** neither factory ever sets `opt_level` (grep: zero hits) → every DM kernel resolved O2,
  every compute kernel resolved **O3**. The Metal 2.0 spec must set `opt_level = O3` explicitly on every
  compute KernelSpec (Metal 2.0 default is O2).

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` (both factories).
- **Custom `compute_program_hash`:** none — default reflection hash; nothing to preserve.
- **Implementation notes:**
  - Each factory's `create_descriptor` becomes `create_program_artifacts` returning
    `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`; the internal single-tile/multi-tile
    host-side branch is kept as two builder functions returning the artifacts pair.
  - The sharded factory has zero runtime args → its `ProgramRunArgs` carries only the four
    `TensorArgument`s backing the borrowed DFBs; no `KernelRunArgs` entries at all.
  - Anonymous-namespace name constants (`DFBSpecName`, kernel unique-id strings) must be **function-local**
    (unity-build collision between the two factory TUs).

## Planned Spec Shape

Default 1:1 with legacy. DFB names below are the spec-name strings shared by both configs of a factory
(each config builds only the DFBs its legacy counterpart allocated).

### Variant: RotaryEmbeddingHfMultiCoreSharded (port first)

- **KernelSpecs:** 2 per config — `reader` (DM, reader-default hw_config via
  `create_reader_datamovement_config(device->arch())`, opt O2 default) and `compute`
  (ComputeGen1Config per the dropped-field rule above; `opt_level = O3` explicit).
  - single-tile decode compute named CTAs: `n_heads_per_batch_t`, `batch_per_core`.
  - multi-tile decode compute named CTAs: `head_dim_t` (kernel local `Wt`), `n_heads_t` (dead `Ht` — kept),
    `n_heads_per_batch_t`, `batch_per_core`.
  - reader named CTAs: single-tile — none (its only CTA was the trans_mat CB index → DFBBinding);
    multi-tile — `scalar_value` (the bf16 −1.0 bit pattern; the CB index CTA → DFBBinding).
  - No KernelRunArgs (legacy sets no RTAs).
- **DataflowBufferSpecs:** 8 per config:
  `input`/`cos`/`sin`/`output` with `borrowed_from = INPUT/COS/SIN/OUTPUT`;
  `trans_mat` (single-tile) or `scalar` (multi-tile); `rotated_interm`, `cos_interm`, `sin_interm`.
  `entry_size`/`num_entries` copied from legacy `page_size`/`total_size÷page_size`; formats verbatim
  (note multi-tile interm formats follow cos/sin fmt, single-tile interm all input fmt). No `tile` fields
  set in legacy → none set here.
- **SemaphoreSpecs:** none.
- **TensorParameters:** 4 — `INPUT`, `COS`, `SIN`, `OUTPUT` (needed as borrow anchors even though no kernel
  binds an accessor; `TensorArgument`s reference the live tensors so the framework refreshes borrowed
  addresses on cache hits).
- **WorkUnitSpecs:** 1 — {reader, compute} × all_cores. (Single WU + borrowed DFBs = safe re the known
  multi-WU borrowed-DFB bug.)

### Variant: RotaryEmbeddingHfMultiCore (port second; blocked on the sibling's `_metal2` fork for the single-tile path)

- **KernelSpecs:** 3–4 per config — `reader` (source per `in_sharded` in the single-tile config; reader
  default hw_config), `writer` (writer default hw_config; `defines = {{"OUT_SHARDED","1"}}` iff
  `out_sharded`), `compute_g1`, and `compute_g2` iff `core_group_2` nonempty (same source, per-group
  `num_rows_per_core` **named CTA** — preserved multiplicity, see below). All compute specs `opt_level = O3`.
  - reader named CTAs: multi-tile — `scalar_value`, `Ht`, `Wt`, `HtWt`, `half_Wt`; single-tile — `Ht`, `HtWt`.
  - reader named RTAs (per node, via `AddRuntimeArgsForNode`): multi-tile & single-tile-interleaved —
    `num_rows`, `start_id`, `start_row_id`, `cos_sin_start_id`; single-tile-in_sharded — `num_rows`,
    `start_row_id`, `cos_sin_start_id`. (Address slots become TensorBindings.)
  - writer named RTAs: `num_tiles`, `start_id`.
  - compute named CTAs: multi-tile — `num_rows`, `Wt`, `half_Wt`; single-tile — `num_rows` (final names on
    the borrowed-kernel path are dictated by the sibling's fork vocabulary).
- **DataflowBufferSpecs:** 9 (multi-tile: `input`, `rotated_input`, `cos`, `sin`, `scalar`, 3 interm,
  `output`) / 8 (single-tile: `input`, `trans_mat`, `cos`, `sin`, 3 interm, `output`).
  `borrowed_from = INPUT` on `input` iff `in_sharded`; `borrowed_from = OUTPUT` on `output` iff
  `out_sharded` (borrowing is conditional in legacy — carried as a conditional field, bindings unchanged).
- **SemaphoreSpecs:** none.
- **TensorParameters:** 4 (`INPUT`, `COS`, `SIN`, `OUTPUT`). TensorBindings: reader binds INPUT/COS/SIN
  (single-tile-in_sharded reader binds COS/SIN only); writer binds OUTPUT — including under `OUT_SHARDED`
  (the kernel's accessor is `#ifdef`-compiled out; see Applied Patterns for how the binding is handled).
- **WorkUnitSpecs:** 1 when `core_group_2` empty ({reader, writer, compute_g1} × all_cores split as
  {reader,writer} on all_cores + compute_g1 on group_1... — concretely: WU1 = {reader, writer, compute_g1}
  × core_group_1, WU2 = {reader, writer, compute_g2} × core_group_2 when nonempty. Borrowing only occurs
  when the shard path forces core_group_2 empty → never 2 WUs with borrowed DFBs.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| MultiCore prefill (both configs): compute_g1 (`:256`/`:566`) + compute_g2 (`:272`/`:583`) of the same compute source, disjoint core groups, differing only in the `num_rows_per_core` CTA (single-tile CTA[8], multi-tile CTA[9]) | `compute_g1`, `compute_g2` (same source, per-group named CTA value) | WU(group_1), WU(group_2) | each DFB binds one role per instance over disjoint node sets — ordinary 1:1, no flag (input/rotated/cos/sin/scalar CONSUMER; interm self-loop P+C; output PRODUCER) |

Sharded factory: none — no work-split multiplicity in legacy.

**Do not** demote `num_rows_per_core` to an RTA (anti-pattern: Demoting per-group CTA to RTA).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| multi_core `:298-312`, `:608-618` reader RTA slots 0–2 (0–1 in in_sharded single-tile) | `Buffer*` (src/cos/sin) via `emplace_runtime_args` (descriptor `BufferBinding`) | `TensorBinding` INPUT/COS/SIN on reader; kernel `TensorAccessor(tensor::…)` |
| multi_core `:312`, `:618` writer RTA slot 0 | `Buffer*` (dst) | `TensorBinding` OUTPUT on writer |
| reader kernels `TensorAccessorArgs<6|10>()` chains + `:33-35` addr RTAs; writer `TensorAccessorArgs<1>()` | host `TensorAccessorArgs(*buf).append_to(cta)` + kernel offset chain + addr RTA | binding mechanism end-to-end; kernel one-line `TensorAccessor(tensor::name)` |
| reader `:39,42,45` / writer `:23` / single-tile readers `:93,96,99` & `:99,102` | `TensorAccessor(args, addr, get_tile_size(cb))` 3rd arg | dropped (Class 2 — binding token supplies aligned page size; audit-cleared, 9 sites) |
| every CB-index CTA: sharded reader CTA[0], sharded compute CTAs[0..7] (both configs), prefill reader CTAs[0..4]/[0..3], writer CTA[0], prefill compute CTAs[0..8]/[0..7] | magic CB index as positional CTA | `DFBBinding` with named DFB; kernel `dfb::name` |
| all remaining positional CTAs | positional `get_compile_time_arg_val(N)` | named CTAs (`scalar_value`, `Ht`, `Wt`, `HtWt`, `half_Wt`, `head_dim_t`, `n_heads_t`, `n_heads_per_batch_t`, `batch_per_core`, `num_rows`) |
| all remaining positional RTAs (prefill readers slots 3–6 / 2–4; writer slots 1–2) | positional `get_arg_val<uint32_t>(N)` | named RTAs via `get_arg(args::…)` |

No semaphore-ID RTAs (op has no semaphores). No page-size CTA/RTA slots beyond the accessor 3rd args above.

## Applied Patterns

- **Self-loop DFB binding** (`port_patterns.md` — Sync-free and single-ended CBs → self-loop DFB):
  interm `c_24/c_25/c_26` in every config (compute produces+consumes); additionally `input`/`cos`/`sin`
  (compute cursor-advances the borrowed residents) and `output` (compute produces, nothing drains) in both
  sharded-decode configs. All self-loops are on the compute kernel.
- **Two-toucher 1P+1C** everywhere else, including wait-only consumers (trans_mat/scalar CBs: consumer
  `wait_front`s without popping — still a locked CONSUMER binding, and deliberate fill-once/read-many
  reuse, not a FIFO to "balance").
- **Borrowed-memory DFB** (`borrowed_from = <TensorParameter>`): all four io DFBs in the sharded factory;
  `input`/`output` conditionally in the prefill factory. The in-sharded multi-tile prefill keeps the
  intentional self-aliasing NoC read (reader reads `input` via accessor into the DFB borrowed from
  `input.buffer()`) byte-for-byte.
- **Multi-variant factory** (host-side config selection inside `create_program_artifacts`): single-tile vs
  multi-tile builder per factory; prefill reader source additionally selected on `in_sharded`.
- **Conditional define, unconditional binding** — `OUT_SHARDED` on the writer: the define stays
  conditionally emitted exactly as legacy (`out_sharded` only). The writer's OUTPUT TensorBinding and its
  named RTAs are emitted unconditionally, mirroring legacy exactly: the legacy kernel constructs its
  `TensorAccessor` *outside* the `#ifdef` (only its NoC use is `#else`-gated), so the ported kernel keeps
  `TensorAccessor(tensor::dst)` unconditional too — no new `#ifdef` structure was needed at all. (Resolved
  at construction; the conditional-binding fallback in the earlier draft of this entry was not needed.)
- **Preserved per-group CTA multiplicity** (anti-pattern guard): compute_g1/compute_g2, above.
- **Pass DFB handles directly to LLKs** (`matmul_tiles`, `mul_tiles_bcast*`, `pack_tile`,
  `reconfig_data_format`, `compute_kernel_hw_startup`, `copy_tile_init_with_dt` from
  `ttnn/kernel/compute/dest_format_helpers.hpp` — `uint32_t cb_id` shapes bridged by the implicit
  conversion).
- **`constexpr` metadata keeps free-function form**: none needed — every `get_tile_size(cb)` in these
  kernels is a non-constexpr local, so all become `dfb.get_tile_size()` member getters (whitelist §A/§B).
- **unpack_modes required-entry rule**: legacy sets no `unpack_to_dest_mode` (vector empty → `Default` →
  `UnpackToSrc`). Under `enable_32_bit_dest = true` (i.e. resolved `fp32_dest_acc_en`), Metal 2.0 requires
  an explicit `unpack_modes` entry for every **Float32-format DFB the compute kernel consumes** — emit
  `UnpackMode::UnpackToSrc` (the legacy-default translation) for each such DFB, conditionally on
  `fp32_dest_acc_en && format == Float32` (formats are runtime values here; fp32 tests exercise this).
  No entries otherwise; never `UnpackToDest`.

## Deferred / Flagged

- **Sibling-fork dependency (ordering constraint) — RESOLVED:** the prefill single-tile compute needed
  `rotary_embedding_single_tile_metal2.cpp` from the sibling port. The sharded factory was ported first;
  at prefill time the fork existed and was reused (rung 1), and this factory's accessor names conform to
  the fork's vocabulary (see Shared kernels).
- **Dead CTA `n_heads_t`/`Ht`** in `rotary_embedding_hf_sharded.cpp` — kept as a named CTA, still dead in
  the kernel; ops-team cleanup candidate (report).
- **Borrowed-DFB multi-WU bug** — structurally excluded here (borrowing ⇒ single work unit in every config);
  re-verify at construction that no config emits ≥2 WUs with a borrowed output DFB.
- New findings during planning: none beyond the above — the audit's census matched the re-derivation
  exactly (all endpoint dispositions re-derived independently and agreed).
