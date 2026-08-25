# Port Plan — rotary_embedding

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding`, ported from the
`ProgramDescriptor` API (`create_descriptor` + `override_runtime_arguments`) to Metal 2.0
(`create_program_artifacts` + `ProgramRunArgs`-returning `override_runtime_arguments`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` (a `create_descriptor` returning `ProgramDescriptor`), with an
  imperative `override_runtime_arguments(Program&, ...)` cache-hit hook.
- Factory methods live in a proper `program_factory_t` variant
  (`program_factory_t = std::variant<RotaryEmbeddingProgramFactory>` @ `device/rotary_embedding_device_operation.hpp:18`)
  — NOT the direct-descriptor shape; no ttnn_factory exception 3 needed. The port is a method swap inside the
  existing struct.
- Variants: **one factory, two internal descriptor variants** selected by shape at `create_descriptor`
  (`device/rotary_embedding_program_factory.cpp:893-901`):
  - **single-tile** (`Wt == 1`, `padded_shape[-1] == TILE_WIDTH`) — `create_single_tile_descriptor` (`:91`)
  - **multi-tile** (`Wt >= 2`) — `create_multi_tile_descriptor` (`:477`)
- Orthogonal config axes (within each variant): **decode** (`token_idx.has_value()`, emits `DECODE_MODE` to all
  three kernel classes) vs **prefill**; **in-sharded** vs interleaved input (selects the reader source and the
  `c_0` borrowed buffer); **out-sharded** (`OUT_SHARDED` define on the writer, `c_16` borrowed) vs interleaved output.
- Custom `compute_program_hash`: **present** @ `device/rotary_embedding_device_operation.cpp:146-162` — left
  intact. Deliberately keys `token_idx.has_value()` but not the value, so decode positions cache-hit one program;
  that is exactly why the translated override must re-emit the two token-derived decode scalars.
- Runtime kernel-source selection: the reader source is chosen at runtime by `in_sharded` (per variant), so
  **all four readers + writer + both computes (7 kernel sources) convert together** — the factory is the atomic unit.

### Variant: single-tile (Wt == 1)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|
| reader | `reader_rotary_embedding_single_tile_interleaved_start_id.cpp` (interleaved-in) / `..._sharded.cpp` (in-sharded) | all_cores | interleaved: `[c_0, c_2, c_3, c_1(trans_mat), Ht, HtWt] + TensorAccessorArgs(src,cos,sin)`; sharded: `[c_0, c_2, c_3, c_1, Ht, HtWt] + TensorAccessorArgs(cos,sin)` | interleaved: `{src*, cos*, sin*, num_rows, start_id, start_row_id, cos_sin_start_id}` (7); sharded: `{cos*, sin*, num_rows, start_row_id, cos_sin_start_id}` (5) | `DECODE_MODE` (decode) | O2 (DM default; no explicit set) | `ReaderConfigDescriptor{}` |
| writer | `writer_rotary_embedding_interleaved_start_id.cpp` | all_cores | `[c_16] + TensorAccessorArgs(dst)` + decode: `[c_27, c_5, c_28, c_6]` | `{dst*, num_tiles(=num_rows*Wt), start_id, cos_sin_offset, Wt, Wbytes}` (6) | `DECODE_MODE` (decode), `OUT_SHARDED` (out-sharded) | O2 | `WriterConfigDescriptor{}` |
| compute_g1 | `rotary_embedding_single_tile.cpp` (**lent** — see Shared kernels) | core_group_1 | `[c_0, c_2, c_3, c_1, c_24, c_25, c_26, c_16, num_rows_g1]` + decode: `[c_27, c_5, c_28, c_6, c_29, c_30]` | none | `DECODE_MODE` (decode) | **O3** (compute default; no explicit set) | `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en}` (`:404-407`) |
| compute_g2 | same source | core_group_2 (only if non-empty) | same, `num_rows_g2` at slot 8 (`:412`) | none | same | **O3** | same config as g1 (`:421-424`) |

`*` = `Buffer*` runtime arg (the descriptor `emplace_runtime_args` Buffer-pointer form) — becomes a `TensorBinding`.

CTAs read from host emission order (`:314-348`, `:375-394`); no CRTAs anywhere in this op.

#### CBs

All `core_ranges = all_cores`. Formats: `input_fmt = dataformat(input.dtype)`, `cos_fmt`/`sin_fmt` likewise,
`out_fmt = dataformat(output.dtype)`, `trans_mat_fmt = (input_fmt==Bfp8_b) ? Bfp8_b : Float16_b`,
`scalar_fmt = Float16_b`. `num_cos_sin_tiles = decode ? Wt : 2*Wt` (Wt = 1 here).

| index | total_size | data_format | page_size | notes |
|---|---|---|---|---|
| c_0 input | `num_input_tiles * input_tile_size` (sharded: shard volume; interleaved: 2*Wt) | input_fmt | input_tile_size | `.buffer = input.buffer()` when in-sharded → **borrowed** |
| c_1 trans_mat | 1 * trans_mat_tile_size | trans_mat_fmt | trans_mat_tile_size | reader fills once; compute waits, never pops |
| c_2 cos | num_cos_sin_tiles * cos_tile_size | cos_fmt | cos_tile_size | |
| c_3 sin | num_cos_sin_tiles * sin_tile_size | sin_fmt | sin_tile_size | |
| c_24 rotated_in_interm | 1 * input_tile_size | input_fmt | input_tile_size | compute-internal |
| c_25 cos_interm | 1 * input_tile_size | **input_fmt** | input_tile_size | deliberately input format (comment @ `:204-206`) |
| c_26 sin_interm | 1 * input_tile_size | **input_fmt** | input_tile_size | ditto |
| c_16 out | `num_output_tiles * out_tile_size` (out-sharded: shard volume; else 2*Wt) | out_fmt | out_tile_size | `.buffer = output.buffer()` when out-sharded → **borrowed** |
| c_29 retilized_cos (decode only) | Wt * cos_tile_size | cos_fmt | cos_tile_size | |
| c_30 retilized_sin (decode only) | Wt * sin_tile_size | sin_fmt | sin_tile_size | |
| c_27 + c_5 (decode only) | **one CBDescriptor**, Wt * scalar_tile_size, **two** format descriptors (`:271-286`) | scalar_fmt | scalar_tile_size | **aliased pair**: c_27 untilized-cos data, c_5 untilized-cos sync |
| c_28 + c_6 (decode only) | one CBDescriptor, Wt * scalar_tile_size (`:288-303`) | scalar_fmt | scalar_tile_size | aliased pair: c_28 untilized-sin data, c_6 sync |

### Variant: multi-tile (Wt >= 2)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|
| reader | `reader_rotary_embedding_interleaved_start_id.cpp` / `..._sharded.cpp` | all_cores | interleaved: `[c_0, c_1(rotated), c_2, c_3, c_4(scalar), bfloat16_scalar(-1.0f), Ht, Wt, HtWt, half_Wt] + TA(src,cos,sin)`; sharded: same first 9 but slot 9 = `half_Wt_size = half_Wt*input_tile_size`, `+ TA(cos,sin)` | same shapes as single-tile (7 interleaved / 5 sharded) | `DECODE_MODE` | O2 | `ReaderConfigDescriptor{}` |
| writer | `writer_rotary_embedding_interleaved_start_id.cpp` (same file as single-tile) | all_cores | same as single-tile | same as single-tile | `DECODE_MODE`, `OUT_SHARDED` | O2 | `WriterConfigDescriptor{}` |
| compute_g1 | `rotary_embedding.cpp` | core_group_1 | `[c_0, c_1, c_2, c_3, c_4, c_24, c_25, c_26, c_16, num_rows_g1, Wt, half_Wt]` + decode: `[c_27, c_5, c_28, c_6, c_29, c_30]` | none | `DECODE_MODE` | **O3** | **`ComputeConfigDescriptor{}`** — deliberate legacy-parity asymmetry (`:812-814`); preserve, don't "fix" |
| compute_g2 | same source | core_group_2 (if non-empty) | same, `num_rows_g2` at slot 9 (`:819`) | none | same | **O3** | `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en}` (`:828-831`) |

#### CBs

Differences from single-tile: **no trans_mat**; `c_1` is **rotated_input** (`2*Wt * input_tile_size`, input_fmt);
adds `c_4` scalar (1 * Float16_b tile); `c_25`/`c_26` use **cos_fmt/sin_fmt** here (not input_fmt — `:602-622`).
c_0/c_2/c_3/c_16/c_24 and the decode CBs (c_29/c_30, aliased c_27+c_5 @ `:664-679`, c_28+c_6 @ `:681-696`)
have the same shape as single-tile.

### Semaphores

none

### Tensor accessors

| host site | originating Tensor | RTA slot (host) | kernel accessor |
|---|---|---|---|
| `:453`/`:867` (interleaved readers) | input (`src`) | reader RTA 0 (`src_buffer`) | `TensorAccessor(src_args, src_addr[, input_tile_bytes]*)` |
| `:449`/`:453`/`:863`/`:867` (all four readers) | cos | reader RTA 1 (0 sharded) | `TensorAccessor(cos_args, cos_addr[, cos_tile_bytes]*)` |
| same | sin | reader RTA 2 (1 sharded) | `TensorAccessor(sin_args, sin_addr[, sin_tile_bytes]*)` |
| `:462-463`/`:876-877` (writer) | output (`dst`) | writer RTA 0 | `TensorAccessor(dst_args, dst_addr)` — compiled out under `OUT_SHARDED` |

`*` The 3rd (page-size) argument appears only in the two **single-tile** readers — audit classified all 5 sites
Class 2 (redundant) → **drop**: `reader_..._single_tile_interleaved_start_id.cpp:103,106,109`,
`reader_..._single_tile_interleaved_start_id_sharded.cpp:95,98`.

### Work split

- Driver: `compute_rotary_work_split(input, output, Wt)` (`:45-85`), shared by `create_descriptor` and
  `override_runtime_arguments` — keep it the single source of truth in the port too.
- Sharded (in or out): `all_cores = shard grid`, `core_group_1 = all_cores`, **`core_group_2 = empty`**,
  `num_rows_per_core_group_1 = shard_shape[0]/TILE_HEIGHT`. → every borrowed-DFB config has exactly one
  compute KernelSpec / one WorkUnitSpec (relevant to the borrowed-DFB multi-work-unit bug — see Flags).
- Interleaved: `split_work_to_cores(grid, num_rows, row_major=true)` → `(num_cores, all_cores, core_group_1,
  core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2)`.
- Core iteration: `grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major)`; per-core
  `num_tiles_written` accumulates `num_rows_per_core * Wt`.

### Shared kernels

- `device/kernels/compute/rotary_embedding_single_tile.cpp` — **lent**: `rotary_embedding_hf`'s
  `RotaryEmbeddingHfMultiCore` binds it (`rotary_embedding_hf/device/rotary_embedding_hf_multi_core_program_factory.cpp:257-260, 273-276`).
  **No `_metal2` fork exists beside it** (checked locationally 2026-08-25; the hf port runs in parallel and has
  only audit artifacts so far) → **rung 2: create `rotary_embedding_single_tile_metal2.cpp` beside the original**,
  leave the pointer comment in the original, bind the fork from this factory. Binding vocabulary chosen for the
  kernel's roles (hf will consume it) — see Applied Patterns. hf binds it **without** `DECODE_MODE` and without
  the decode CTAs, so the entire decode surface stays behind `#ifdef DECODE_MODE` exactly as in the legacy file.
- The other six kernel sources have no external binders (audit filename census; `sources.cmake` hit discarded).
- Intra-op sharing: the writer and both compute sources are each bound by multiple KernelSpecs *of this same
  factory* (both variants / both core groups) — all convert together in this one change; not a Caution case.

### Flags

- **Borrowed-DFB multi-work-unit bug (known framework bug, orchestration heads-up):** specs with >= 2 work units
  and borrowed output DFBs corrupt the borrowed DFB's device base. Structurally avoided here: borrowed DFBs occur
  only in sharded configs, and `compute_rotary_work_split` forces `core_group_2` empty whenever any shard spec is
  present → borrowed configs always build exactly **one** WorkUnitSpec. Interleaved configs may build two WUs but
  borrow nothing. No capitulation needed; assert nothing, just preserve the structure.
- `tests/tt_metal/tt_metal/test_kernels/compute/rotary_embedding.cpp` is an old test kernel under `tt_metal`,
  referenced by nothing (grep of `tests/` finds no user); NOT one of this op's sources — not audited, not touched.
- Config-dead legacy args (audit Misc): `start_row_id` unread under `DECODE_MODE` (all four readers);
  writer's `dst_addr` + dst accessor CTAs + `start_id` unread under `OUT_SHARDED`; writer's
  `cos_sin_offset`/`Wt`/`Wbytes` read only under `DECODE_MODE`. Named schemas are built per config and omit
  args a config never reads; the (currently unconditional) kernel-side reads of dead args get `#ifdef`-gated in
  step (`start_row_id` → `#ifndef DECODE_MODE`; writer `start_id` → `#ifndef OUT_SHARDED`).
- Known-anomaly preservation (from audit; zero-functional-change): the g1/g2 compute-config asymmetry
  (multi-tile), and the absent single-tile dtype `TT_FATAL` promised by the comment @
  `rotary_embedding_device_operation.cpp:44-49` — neither is "fixed".

## TTNN ProgramFactory

- **Concept (inherited from audit): `CustomProgramSpecFactoryConcept`** — the op declares
  `override_runtime_arguments`; the port translates it to the `ProgramRunArgs`-returning shape (signature change
  in `device/rotary_embedding_program_factory.hpp`), never deletes it.
- **Custom `compute_program_hash`**: present @ `device/rotary_embedding_device_operation.cpp:146-162` — leave intact.
- **Implementation notes**:
  - Existing `program_factory_t` variant → method swap inside `RotaryEmbeddingProgramFactory`
    (`create_descriptor` → `create_program_artifacts`; `override_runtime_arguments` re-shaped). No pybound
    `create_descriptor` exists (audit) → no pybind edits.
  - Reference shape (skeptically held; recipe outranks): `ttnn/cpp/ttnn/operations/kv_cache/device/
    update_cache_multi_core_program_factory.{hpp,cpp}` — a landed custom-concept port with the same
    hash-excluded-decode-scalar structure.
  - Keep `compute_rotary_work_split` (anonymous namespace) shared by both methods, exactly as legacy.

## Planned Spec Shape

Default 1:1 with legacy. One `create_program_artifacts` branching to two builder helpers (single-tile /
multi-tile), mirroring `create_single_tile_descriptor` / `create_multi_tile_descriptor`.

> Names below are the `DFBSpecName`/`KernelSpecName`/`TensorParamName` strings; all declared **function-locally**
> (unity-build hygiene).

### Common to both variants

- **TensorParameters** (declared from `<tensor>.mesh_tensor().tensor_spec()`): `src` (input), `cos`, `sin`,
  `dst` (output). All four exist in **every** config: in interleaved configs they carry `TensorBinding`s; in
  sharded configs `src`/`dst` instead back the borrowed DFBs (`borrowed_from`) and have **no** kernel binding.
- **SemaphoreSpecs**: none.
- **WorkUnitSpecs**: `wu_g1 = {reader, writer, compute_g1} @ core_group_1`; plus
  `wu_g2 = {reader, writer, compute_g2} @ core_group_2` only when group 2 is non-empty (never in sharded configs).
  Reader/writer membership in both WUs reproduces their legacy `all_cores` coverage.
- **KernelSpecs**: `reader` (source runtime-selected by `in_sharded`, as legacy), `writer`, `compute_g1`
  [, `compute_g2`] — multiplicity preserved (see Preserved Multiplicity).
- **hw_config**:
  - reader → `create_reader_datamovement_config(device->arch())`; writer → `create_writer_datamovement_config(...)`
    (both legacy configs are the exact reader/writer defaults).
  - compute → build `ComputeGen1Config` **directly** (Style B-equivalent: the legacy factory hand-copies a
    subset of the resolved TTNN config onto `ComputeConfigDescriptor`, so `to_compute_hardware_config` would
    resurrect the dropped `math_approx_mode`/`dst_full_sync_en` — do NOT use the helper):
    - single-tile g1 & g2: `ComputeGen1Config{.fpu_math_fidelity = math_fidelity, .enable_32_bit_dest = fp32_dest_acc_en}`
      (defaults for `sfpu_precision_mode`/`double_buffer_dest`/`bfp_pack_precision_mode` match the legacy
      descriptor defaults exactly).
    - multi-tile g1: `ComputeGen1Config{}` (the preserved asymmetry); multi-tile g2: as single-tile.
  - **unpack_modes (newly-required entries)**: for each compute KernelSpec with `enable_32_bit_dest == true`,
    add `{DFB, UnpackMode::UnpackToSrc}` for **every Float32-format DFB that kernel consumes** (legacy
    `unpack_to_dest_mode` was unset = `Default` → `UnpackToSrc`). Candidate consumed DFBs whose format can be
    Float32: `input`, `cos`, `sin`, `rotated_input` (multi-tile), `rotated_in_interm`, `cos_interm`, `sin_interm`,
    and decode `retilized_cos`/`retilized_sin` (gated on the same decode condition as their bindings). The
    Float16_b DFBs (`scalar`, `trans_mat` non-fp32, `untilized_*`) and produced-only `out` get no entry. Multi-tile
    g1 (`enable_32_bit_dest = false` by the asymmetry) gets none.
- **opt_level**: explicit `KernelBuildOptLevel::O3` on **every compute KernelSpec** (legacy resolved O3; Metal 2.0
  defaults O2). DM kernels: nothing (legacy O2 == Metal 2.0 O2).
- **defines** (`compiler_options.defines`): `DECODE_MODE=1` → reader/writer/compute when `token_idx.has_value()`;
  `OUT_SHARDED=1` → writer when out-sharded. Decode-only DFBs, bindings, CTAs, RTAs and unpack_modes entries are
  conditional in step with `DECODE_MODE`; dst binding/schema conditional in step with `OUT_SHARDED`.

### Variant: single-tile — DataflowBufferSpecs & bindings

| DFBSpecName | entry_size / num_entries | data_format_metadata | borrowed_from | bindings (endpoint) |
|---|---|---|---|---|
| `input` | input_tile_size / num_input_tiles | input_fmt | `src` iff in-sharded | reader P; compute_g1[,g2] C |
| `trans_mat` | trans_mat_tile_size / 1 | trans_mat_fmt | — | reader P; compute C |
| `cos` | cos_tile_size / num_cos_sin_tiles | cos_fmt | — | reader P; compute C |
| `sin` | sin_tile_size / num_cos_sin_tiles | sin_fmt | — | reader P; compute C |
| `rotated_in_interm` | input_tile_size / 1 | input_fmt | — | compute **self-loop** (P+C) |
| `cos_interm` | input_tile_size / 1 | input_fmt | — | compute self-loop |
| `sin_interm` | input_tile_size / 1 | input_fmt | — | compute self-loop |
| `out` | out_tile_size / num_output_tiles | out_fmt | `dst` iff out-sharded | compute P; writer C |
| `retilized_cos` (decode) | cos_tile_size / Wt | cos_fmt | — | compute self-loop |
| `retilized_sin` (decode) | sin_tile_size / Wt | sin_fmt | — | compute self-loop |
| `untilized_cos` (decode) | scalar_tile_size / Wt | scalar_fmt | — | compute P **and** C, writer P **and** C — **`allow_instance_multi_binding = true`**; `alias_with = {untilized_cos_sync}`. *(Revised during verification: the validator requires producer set == consumer set once any kernel self-loops a DFB, so the writer's role-free raw in-place write is bound as its PRODUCER side — see census note below.)* |
| `untilized_cos_sync` (decode) | scalar_tile_size / Wt | scalar_fmt | — | writer P; compute C; `alias_with = {untilized_cos}` |
| `untilized_sin` (decode) | scalar_tile_size / Wt | scalar_fmt | — | as untilized_cos; `alias_with = {untilized_sin_sync}` |
| `untilized_sin_sync` (decode) | scalar_tile_size / Wt | scalar_fmt | — | writer P; compute C; `alias_with = {untilized_sin}` |

Endpoint census re-derived from the kernels (matches brief/audit — no disagreement):
- `untilized_cos`/`untilized_sin`: compute is **locked producer** (untilize helper push) *and* **locked consumer**
  (tilize helper wait+pop); writer is a second **locked consumer** (`wait_front(Wt)` @
  `writer_rotary_embedding_interleaved_start_id.cpp:47,62`, never pops) plus a role-free raw in-place writer.
  Two locked consumers → 1P+1C cannot fit → multi-binding flag (genuine, not a brief over-read). Not stacked with
  a self-loop resolution: the compute P+C here are *real locked FIFO roles* of a multi-bound DFB.
  **Verification-round revision:** the validator additionally enforces (`program_spec.cpp:1441-1460`) that once
  any kernel self-loops a DFB, the producer and consumer *kernel sets* must be equal — compute P+C plus a
  writer-C-only binding is rejected. The legal expression of this census: the writer's role-free raw in-place
  write (a genuine producer-side touch) takes the PRODUCER label, so both kernels bind P+C
  ({writer, compute} == {writer, compute}) and the per-node 2P+2C census is admitted by
  `allow_instance_multi_binding` (census relaxes to ">=1 per role", `program_spec.cpp:1368-1377`). On Gen1 the
  flag lowers the DFB to a plain shared circular buffer where role labels and risc masks are inert
  (`program_spec.cpp:2876-2882`); endpoint collection order (kernels = reader, writer, compute...) makes the
  writer the representative first producer, so no `tensix_scope` is set (DM representative), which the Gen1
  lowering ignores anyway (`dataflow_buffer.cpp:1870-1889` early-returns before all Gen2 machinery).
- Alias legality: each pair shares total size (Wt × scalar_tile_size), the same kernel set {compute, writer},
  the same node coverage; strict two-member clique both directions.
- `trans_mat`, `scalar` (multi-tile), `retilized_*`: consumer `wait_front` without `pop_front` is intentional
  fill-once/read-many reuse — no "balancing".

### Variant: multi-tile — deltas from single-tile

- No `trans_mat`; add `rotated_input` (input_tile_size / 2*Wt, input_fmt; reader P, compute C) and
  `scalar` (scalar_tile_size / 1, Float16_b; reader P, compute C — compute waits, never pops).
- `cos_interm`/`sin_interm` use **cos_fmt/sin_fmt** (not input_fmt).
- Everything else (including the decode set) identical in shape.

### Runtime-arg schemas (named; per config — schemas differ across configs, names don't shift)

- reader (interleaved-in): `num_rows`, `start_id`, `cos_sin_start_id`, + `start_row_id` (prefill only).
- reader (in-sharded): `num_rows`, `cos_sin_start_id`, + `start_row_id` (prefill only).
- writer: `num_tiles`, + `start_id` (unless OUT_SHARDED), + `cos_sin_offset`/`Wt`/`Wbytes` (decode only).
- compute: none (CTAs only).
- No CRTAs, no varargs (audit: every read is a fixed distinct index).

### Named CTAs

- single-tile reader: `Ht`, `HtWt`; multi-tile reader: `scalar_value`, `Ht`, `Wt`, `HtWt`, and
  `half_Wt` (interleaved) / `half_Wt_size` (sharded — value is `half_Wt * input_tile_size`, keep the kernel's own
  name).
- single-tile compute: `num_rows`; multi-tile compute: `num_rows`, `Wt`, `half_Wt`.
- All legacy CB-index CTAs become `DFBBinding`s; `TensorAccessorArgs` CTA blocks become `TensorBinding`s.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| compute g1 + g2 (per variant; g2 only in interleaved configs) of `rotary_embedding[_single_tile].cpp`, differing only in the `num_rows` CTA (and, multi-tile only, the legacy g1/g2 config asymmetry) | `compute_g1`, `compute_g2` | `wu_g1` @ core_group_1, `wu_g2` @ core_group_2 (disjoint) | each binds C of `input`/`cos`/`sin`(/`rotated_input`/`scalar`/`trans_mat`), P of `out`, self-loops of the interms, and the decode set — legal same-role multi-KernelSpec bindings over disjoint node sets (no flag) |

Reader and writer are single KernelSpecs listed in both WUs (legacy: one descriptor over `all_cores`).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA 0/1/2 (interleaved; `:453`/`:867`), 0/1 (sharded; `:449`/`:863`) | `Buffer*` args (src/cos/sin) | `TensorBinding` → `TensorAccessor(tensor::src/cos/sin)` |
| writer RTA 0 (`:462`/`:876`) | `Buffer*` (dst) | `TensorBinding` (conditional on !OUT_SHARDED) → `TensorAccessor(tensor::dst)` |
| reader CTA slots 0-3 (single-tile) / 0-4 (multi-tile); writer CTA 0 + decode CTA block; compute CTA slots 0-7 (single-tile) / 0-8 (multi-tile) + 6-element decode block | magic CB indices | `DFBBinding`s (`dfb::*`) |
| reader CTA tails (`:324-340` single-tile, `:723-743` multi-tile); writer CTA tail (`:340`/`:743`) | `TensorAccessorArgs(*buf).append_to(cta)` + kernel `TensorAccessorArgs<N>()` offset chains | binding mechanism end-to-end |
| single-tile readers, accessor 3rd arg (`...start_id.cpp:103,106,109`; `..._sharded.cpp:95,98`) | `TensorAccessor(args, addr, get_tile_size(cb))` | 2-arg collapse `TensorAccessor(tensor::x)` — binding supplies page size |
| all remaining positional CTAs / RTAs | positional | named (see schemas above) |
| `override_runtime_arguments` address rewrites (`:958-969`) + `UpdateDynamicCircularBufferAddress` block (`:980-991`) | raw addresses into arg slots / CB fields | `tensor_args` entries (`src`,`cos`,`sin`,`dst`); borrowed DFBs refresh from their backing tensor_arg natively |

## override_runtime_arguments translation (custom concept)

Inventory of what the legacy override writes (`:903-992`), mirrored exactly:

| legacy write | Metal 2.0 |
|---|---|
| reader args src/cos/sin addresses per core (`:960-967`) | `tensor_args`: `src`, `cos`, `sin` |
| writer arg dst address per core (`:968-969`) | `tensor_args`: `dst` |
| `UpdateDynamicCircularBufferAddress` on c_0 / c_16 when sharded (`:980-991`) | same `src`/`dst` `tensor_args` (borrowed DFBs re-derive backing address) |
| decode only: reader `cos_sin_start_id` per core (`:971`) | `kernel_run_args[reader].runtime_arg_values["cos_sin_start_id"][core] = token-derived value` for every core (via `AddRuntimeArgsForNode` over the same `grid_to_cores` order) |
| decode only: writer `cos_sin_offset` per core (`:972`) | `kernel_run_args[writer].runtime_arg_values["cos_sin_offset"][core] = ...` |

- All four io `TensorParameter`s get a `TensorArgument` on **every** dispatch (the legacy override refreshed all
  four in every config). Prefill: `kernel_run_args` empty. No `dfb_run_overrides` (sizes never change).
- The recomputation (`Wt`, `compute_rotary_work_split`, `grid_to_cores`, the two token formulas) stays shared
  with the miss path exactly as legacy. Guards: the legacy override contains no TT_FATAL/TT_ASSERT — none to move.

## Applied Patterns

- [Self-loop DFB binding] + [Sync-free/single-ended CBs → self-loop]: `rotated_in_interm`, `cos_interm`,
  `sin_interm`, `retilized_cos`, `retilized_sin` — compute-only touchers.
- [Aliased DFBs]: `untilized_cos`+`untilized_cos_sync`, `untilized_sin`+`untilized_sin_sync`
  (`advanced_options.alias_with`, strict cliques).
- Multi-binding advanced option (genuine ≥2-locked-consumers case): `untilized_cos`, `untilized_sin`
  (`allow_instance_multi_binding = true`), decode configs only.
- [Conditional / optional DFB bindings]: entire decode DFB/CTA/RTA surface keyed to `DECODE_MODE`; dst
  binding + writer args keyed to `OUT_SHARDED`. Kernel-side reads of config-dead args gated with the same defines.
- [Multi-variant factories]: `create_program_artifacts` branches to single-tile / multi-tile builders.
- [Pass DFB handles directly to LLKs and kernel-lib helpers]: `compute_kernel_lib::tilize/untilize` NTTPs and all
  LLK calls take `dfb::name` via the constexpr `uint32_t` conversion (audit: donors already DFB-based). The
  compute kernels' runtime-variable `updated_cos_cb`/`updated_sin_cb` locals stay `uint32_t` (initialized from
  `dfb::cos`/`dfb::sin`, decode-reassigned to `dfb::retilized_*` under `#ifdef DECODE_MODE`); helper-local
  `DataflowBuffer(uint16_t id)` construction is the sanctioned runtime-id form (`dataflow_buffer.h:113`).
- [Caution: Porting a shared kernel] rung 2: fork `rotary_embedding_single_tile.cpp` →
  `rotary_embedding_single_tile_metal2.cpp` in this op's directory (lent kernel), pointer comment in the original.
  Fork binding vocabulary (kernel-role names, for hf reuse): dfb accessors `in`, `cos`, `sin`, `trans_mat`,
  `rotated_in_interm`, `cos_interm`, `sin_interm`, `out`; decode (behind `DECODE_MODE`): `untilized_cos`,
  `untilized_cos_sync`, `untilized_sin`, `untilized_sin_sync`, `retilized_cos`, `retilized_sin`; named CTA
  `num_rows`. No tensor/sem bindings in the compute kernel.
- [Unity-build hygiene]: all `KernelSpecName`/`DFBSpecName`/`TensorParamName` constants function-local in the two
  builder helpers (single .cpp, two builders + override in one anon-namespace scope).
- Constexpr metadata token form (whitelist rule 7): writer's `constexpr uint32_t out_tile_size =
  get_tile_size(cb_id_out)` (`writer_...start_id.cpp:81`) → `get_tile_size(dfb::out)` token form, stays constexpr.

## Deferred / Flagged

- New findings during planning: **none** beyond the Flags above. The endpoint census, aliasing, multi-binding,
  and concept all confirm the audit/brief — no disagreements to surface.
- Coverage note (for the report): the single-tile (Wt==1) path is exercised only by
  `test_rotary_embedding_decode_program_cache_reuse` (`X=32` rows @
  `tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding.py:480-483`); no single-tile
  prefill/interleaved pytest exists.
