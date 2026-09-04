# Port Plan — `data_movement/slice`

Port plan for `slice`, ported from `ProgramDescriptorFactoryConcept` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass:** `SliceTileProgramFactory` only — one of the op's five factories.
Per the recipe's atomic-unit note, a factory is the unit of a port; the other four stay on
`ProgramDescriptorFactoryConcept` and the op keeps building and running. The remaining four are
enumerated under [Deferred / Flagged](#deferred--flagged) for the next pass.

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` (all five factories define
  `create_descriptor(const SliceParams&, const SliceInputs&, Tensor&) -> ProgramDescriptor`).
- Factory methods live in a `program_factory_t` variant (`slice_device_operation.hpp:36-41`), **not**
  directly on the device-operation struct — so exception 3 (direct-descriptor shape) does **not** apply.
- Variants: five factories in one variant —
  `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`,
  `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory`.
- Custom `compute_program_hash`: **present** at `slice_device_operation.cpp:348-432` — left intact.
  Recorded so a later `TensorSpec` legality failure has a named suspect.

*(Target concept `CustomProgramSpecFactoryConcept` was chosen during the audit — see the brief's
TTNN factory analysis. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels — `SliceTileProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp` | `all_cores` | `{num_dims}` then `TensorAccessorArgs(*src0_buffer)` (`tile.cpp:64-65`) | `{"dfb_id_in", src0_cb_index}` (`:139`) | per core: `[start_id, num_tiles, id_per_dim×num_dims]` (`:100-130`) | `[Buffer* src0_buffer, num_unpadded_tiles_per_dim×num_dims, num_padded_tiles_per_dim×num_dims]` (`:141-145`) | none | absent → resolves **O2** (DM) | `ReaderConfigDescriptor{}` (`:146`) |
| writer | `slice/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `all_cores` | `{}` then `TensorAccessorArgs(*dst_buffer)` (`:151-152`) | `{"dfb_id_out", src0_cb_index}` (`:161`) | per core: `[Buffer* dst_buffer, num_tiles_per_core, num_tiles_written]` (`:168-182`) | none | none | absent → resolves **O2** (DM) | `WriterConfigDescriptor{}` (`:162`) |

`grep -n opt_level` over all five factory `.cpp`s and `.hpp`s returns **nothing** — no legacy kernel
sets one. Both kernels here are DM, so rule 1 does not fire and rule 2 (compute → explicit `O3`) does
not apply: legacy DM default `O2` == Metal 2.0 default `O2`, nothing to carry. **This factory builds
no compute `KernelSpec`, so the `opt_level` self-audit item has an empty denominator by construction.**

### CBs — `SliceTileProgramFactory`

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `src0_cb_index = 0` | `num_input_tiles(2) * single_tile_size` (`tile.cpp:54`) | `all_cores` | `datatype_to_dataformat_converter(input.dtype())` | `single_tile_size = tile_size(cb_data_format)` | **not set** |

Single-element `format_descriptors` → not an aliased CB. `.buffer` not set → not borrowed memory.
`.global_circular_buffer` not set → not a GlobalCircularBuffer.

### Semaphores

**none** — slice declares no semaphores anywhere in the op.

### Tensor accessors — `SliceTileProgramFactory`

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `tile.cpp:65` (`TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args)`) | `tensor_args.input` | **CRTA** slot 0 (`tile.cpp:143`) |
| `tile.cpp:152` (`TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)`) | `output` | RTA slot 0 (`tile.cpp:180`) |

Kernel-side: `TensorAccessor(src_args, src_addr)` (`reader_...:26`), `TensorAccessor(dst_args, dst_addr)`
(`writer_...:36`). Both two-arg — **no 3rd (page-size) argument anywhere**, consistent with the brief.

### Work split

- Driver: `split_work_to_cores(sub_core_grids | compute_with_storage_grid_size, num_unpadded_tiles)`
  (`tile.cpp:31-34`).
- `(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2)`.
- Both kernels are instantiated **once** over `all_cores`; the per-group difference is carried in
  **runtime** args (`num_tiles_per_core`), not per-group CTAs. So there is no multi-`KernelDescriptor`
  work split here and nothing for the demoting-CTA anti-pattern to bite on.
- `all_cores` includes **no-op cores** — cores in neither group get zero-filled arg vectors
  (`tile.cpp:110-111`, `:176`). They are part of the kernels' node set and must still receive values.

### Shared kernels

**none for this factory.** Both sources were censused with `grep -rl <filename> ttnn/cpp/ttnn/operations/`
and each hit disambiguated:

- `reader_unary_unpad_dims_interleaved_start_id.cpp` — bound only by `slice_program_factory_tile.cpp:135`.
  The `nlp_kv_cache_load_slice` hit is a **different file**
  (`reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp`), not this one.
- `writer_unary_interleaved_start_id.cpp` (slice's own copy, `slice/device/kernels/dataflow/`) — bound
  only by `slice_program_factory_tile.cpp:157`. **Same-basename trap confirmed and avoided:** the
  identically-named file under `eltwise/unary/device/kernels/dataflow/` is a *different file* bound by
  `SliceTileTensorArgsProgramFactory` (`tile_tensor_args.cpp:133`) and 14 other ops. This port does not
  touch it. Slice's own copy is neither borrowed, lent, nor intra-op shared → **converted in place**.

### Flags

- **Two unreferenced kernel files** in `device/kernels/dataflow/` — `strided_slice_reader_rm_interleaved_nd.cpp`
  and `strided_slice_writer_rm_interleaved.cpp`. No factory names them. Not audited, not ported, noted so
  the report makes clear what was *not* covered.
- **Cross-op host coupling — the port's dominant structural fact.** `ccl/mesh_partition` drives slice's
  factories directly (see [TTNN ProgramFactory](#ttnn-programfactory)). Recorded here because it is
  discovered at inventory time, not at construction time.
- **`#ifdef OUT_SHARDED` / `#ifdef BACKWARDS` in the writer are dead on every slice path** — no slice
  factory sets kernel `defines`. Both branches are left exactly as they are (scope discipline); only the
  `#else` path is ever compiled for slice.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: **`CustomProgramSpecFactoryConcept`**. Selected by the presence of
  `override_runtime_arguments` (`slice_program_factory_tile.cpp:189`). Not re-derived.
- **Custom `compute_program_hash`**: present at `slice_device_operation.cpp:348` — **leave intact**.
- **Implementation notes**:

  **1. The override body is shared across all five factories and with another op.** All five
  `override_runtime_arguments` are one-line delegations to `ttnn::prim::patch_slice_program_addresses`
  (`slice_program_factory_rm_sharded.cpp:354-413`), which `std::visit`s the factory variant and branches
  internally. Porting one factory means **peeling that factory's branch out** into a
  `ProgramRunArgs`-returning override on the factory itself, leaving the shared function intact for the
  four unported factories (and for MeshPartition's use of them).

  **2. `create_descriptor` cannot survive alongside `create_program_artifacts`.** From
  `ttnn/api/ttnn/operation_concepts.hpp:120-135`, both spec concepts require
  `!ProgramDescriptorFactoryConcept<T>`, and that concept is satisfied by the mere presence of
  `&T::create_descriptor`. A factory declaring both members classifies as a **descriptor** factory:
  `AllFactoriesValid` still passes (exactly one concept matches), the build is green, tests pass — and
  `create_program_artifacts` is never called. **A "keep both as a compatibility shim" approach is
  therefore a silent no-op port**, not a safe fallback. `create_descriptor` must be deleted.

  **3. Consequence — `ccl/mesh_partition` must move with this factory.**
  `mesh_partition_program_factory.cpp:126-134` calls `Factory::create_descriptor(...)` inside a
  `std::visit` **generic lambda**, which the compiler instantiates for *every* alternative of
  `SliceDeviceOperation::program_factory_t`. Deleting `create_descriptor` from any one factory therefore
  breaks that translation unit's compile — there is no slice factory that can be ported in isolation.
  Combined with (2), no in-directory shim exists. The invoker was consulted and authorized carrying
  MeshPartition along; the two files are:
  - `ccl/mesh_partition/device/mesh_partition_program_factory.cpp` — `create_at` builds the ported
    factory's Program via `MakeProgramFromSpec` + `SetProgramRunArgs` instead of `Program{descriptor}`;
    `override_runtime_arguments` routes the ported factory through `UpdateProgramRunArgs`.
  - `ccl/mesh_partition/device/mesh_partition_device_operation.hpp` — unchanged if the stored
    `program_factory_t` still discriminates correctly (it does).

  This is an out-of-op-directory edit and is **not** covered by the recipe's shared-kernel carve-out. It
  is recorded as a Handoff point in the port report.

---

## Planned Spec Shape — `SliceTileProgramFactory`

- **KernelSpecs** (2, 1:1 with legacy `KernelDescriptor`s):
  - `READER` — `unique_id "reader"`, source `reader_unary_unpad_dims_interleaved_start_id.cpp`
  - `WRITER` — `unique_id "writer"`, source `writer_unary_interleaved_start_id.cpp` (slice's own copy)
- **DataflowBufferSpecs** (1, 1:1 with the single legacy `CBDescriptor`):
  - `TILES` — `unique_id "tiles"`, `entry_size = single_tile_size`, `num_entries = num_input_tiles (2)`,
    `data_format_metadata = cb_data_format`. `tile_format_metadata` **not set** (legacy `.tile` unset).
    No `borrowed_from`, no `alias_with`, no multi-binding flag.
- **SemaphoreSpecs**: none — legacy declares none.
- **TensorParameters** (2, one per distinct originating tensor):
  - `INPUT` — `unique_id "input"`, `spec = input.tensor_spec()`, relaxations **none** (strict).
  - `OUTPUT` — `unique_id "output"`, `spec = output.tensor_spec()`, relaxations **none** (strict).
- **WorkUnitSpecs** (1): `{READER, WRITER}` over `all_cores` — both legacy kernels share one
  `core_ranges`, so one (kernels, nodes) pairing covers them.
- **Op-owned tensors**: none (audit Q1 blank; `descriptor` concept could not carry them).

### DFB endpoint census — re-derived, not transcribed

One CB (`src0_cb_index = 0`), two distinct touchers on every node:

| Kernel | Touch | Tag |
|---|---|---|
| reader | `reserve_back(1)` / `push_back(1)` (`reader_...:39,42`) | **locked PRODUCER** |
| writer | `wait_front(onepage)` / `pop_front(onepage)` (`writer_...:45,48`) | **locked CONSUMER** |

Two touchers, at most one locked to each role → **1P + 1C**, no flag, no self-loop. Agrees with the
brief's "legal 1:1". Both `KernelSpec`s run over the *same* `all_cores`, so this is the same-grid
two-toucher shape, not the disjoint-node work split — and 1P+1C is exactly right for it.

### Binding names

Names are taken from the kernels' own role vocabulary, not the legacy host locals (`src0_cb_index`,
`src_addr`, `dst_addr` would all reintroduce `cb`/positional flavour):

| Resource | Host spec name | Reader accessor | Writer accessor |
|---|---|---|---|
| DFB | `"tiles"` | `dfb::in0` (PRODUCER) | `dfb::out` (CONSUMER) |
| TensorParameter `input` | `"input"` | `tensor::src` | — |
| TensorParameter `output` | `"output"` | — | `tensor::dst` |

The writer's `dfb::out` / `tensor::dst` deliberately match the vocabulary of the already-existing
`_metal2` fork of the *eltwise* near-twin
(`eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`), so that if the
two copies are ever consolidated the names already agree. (This port does **not** consolidate them —
that is a separate change; see the report's Open items.)

---

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Both kernels are instantiated once over `all_cores`;
the per-core-group difference rides a runtime arg (`num_tiles_per_core`), never a per-group CTA. There
is nothing to preserve and nothing for
[Demoting per-group CTA to RTA](docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
to catch.

---

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `tile.cpp:143` — reader **CRTA** slot 0 | `reader_common.push_back(src0_buffer)` (`Buffer*`) | `TensorBinding{INPUT, "src"}` + `TensorArgument{INPUT, input}` |
| `tile.cpp:180` — writer RTA slot 0 | `emplace_runtime_args(core, {dst_buffer, …})` (`Buffer*`) | `TensorBinding{OUTPUT, "dst"}` + `TensorArgument{OUTPUT, output}` |
| `tile.cpp:65` — reader CTA slots 1..N | `TensorAccessorArgs(*src0_buffer).append_to(...)` | binding mechanism end-to-end; kernel's `TensorAccessorArgs<1>()` (`reader_...:14`) drops |
| `tile.cpp:152` — writer CTA slots 0..N | `TensorAccessorArgs(*dst_buffer).append_to(...)` | binding mechanism end-to-end; kernel's `TensorAccessorArgs<0>()` (`writer_...:20`) drops |
| `tile.cpp:139` — reader named CTA | `{"dfb_id_in", src0_cb_index}` | `DFBBinding{TILES, "in0", PRODUCER}` (rule 2: a *named* CTA carrying a CB index still becomes a DFB binding, never a named arg) |
| `tile.cpp:161` — writer named CTA | `{"dfb_id_out", src0_cb_index}` | `DFBBinding{TILES, "out", CONSUMER}` |
| `tile.cpp:64` — reader CTA slot 0 | positional `{num_dims}` | **named** CTA `{"num_dims", num_dims}` |
| `tile.cpp:125-126` — reader RTA slots 0,1 | positional `reader_args[0]`, `[1]` | **named** RTAs `start_id`, `num_tiles` |
| `tile.cpp:180` — writer RTA slots 1,2 | positional | **named** RTAs `num_pages`, `start_id` |

No buffer-address RTA folds a host-computed offset into its base — the audit's Offset-base-pointer gate
cleared, and re-confirmed here: `tile.cpp:97,119,125` fold `start_offset` into `start_id`, a **tile
index**, not an address. Nothing to abort on.

### Retained as varargs (reported per whitelist rule 4)

| Kernel | Block | Count bound by | Why it stays a vararg |
|---|---|---|---|
| reader | `id_per_dim` (RTA) | `num_dims` (a CTA) | Read as `id_per_dim[j]` inside `for (j < num_dims)` (`reader_...:44-52`) — an indexed-collection element. Count is not a source literal (varies across instantiations). → `num_runtime_varargs = num_dims` |
| reader | `num_unpadded_tiles` + `num_padded_tiles` (CRTA) | `2 × num_dims` | Same: read as `[j]` inside the same loop (`:46,48`). → `num_common_runtime_varargs = 2 * num_dims` |

Everything else is named. `start_id` / `num_tiles` (reader) and `num_pages` / `start_id` (writer) are
distinct fields read once each — **named**, per the caution's trap (1).

**Kernel-side consequence — the reader's odometer.** Legacy takes a *writable* pointer into its own RTA
buffer (`tt_l1_ptr uint32_t* id_per_dim = (tt_l1_ptr uint32_t*)(get_arg_addr(2));`, `reader_...:23`) and
mutates it (`id_per_dim[j]++`, `= 0`). The Metal 2.0 vararg API exposes **values only** —
`get_vararg(idx)` is generated as `get_arg_val<uint32_t>(named_rta_words + idx)`
(`tt_metal/jit_build/genfiles.cpp:457`); there is no `get_vararg_addr`. The port therefore copies the
block into a local array at kernel entry and mutates the local. `num_dims` is a CTA, so the array is a
fixed-size local, not a VLA. **This is behaviour-preserving, not a fix**: the host re-supplies the whole
block on every dispatch (cache-miss `SetProgramRunArgs` and cache-hit `override_runtime_arguments`
alike, exactly as legacy's `slice_tile_dynamic_args` re-emitted those slots), so the legacy write-back
was never read across dispatches. Recorded as friction, not as an improvement.

---

## Applied Patterns

- **[Removing pybound legacy factory entry points]** — `slice_nanobind.cpp:168-179`. Delete **only** the
  `.def_static("create_descriptor", ...)`, keeping the enclosing
  `nb::class_<SliceTileProgramFactory>(mod, "SliceTileProgramFactory")`. Deleting the whole `nb::class_`
  would break `import ttnn` outright: `ttnn/ttnn/__init__.py:635-640` and
  `ttnn/ttnn/operations/data_movement.py:548` re-export the symbol at **module scope**. Downstream
  caller `models/experimental/ops/descriptors/data_movement/slice.py:54` breaks at *call* time; that is
  the user-visible surface change, and it goes in the report rather than the diff.
- **[Pass DFB handles directly to LLKs / kernel-lib helpers]** — not needed. Neither kernel passes a DFB
  id across the op boundary; the only cross-boundary calls are `noc.*` framework primitives that take
  the `DataflowBuffer` object itself.
- **[Caution: Avoid varargs unless absolutely necessary]** — applied and documented above; two genuine
  indexed-collection blocks retained, everything else named.
- **Self-loop / two-toucher / aliased / conditional / multi-variant patterns**: none apply.

---

## Deferred / Flagged

- **New findings during planning:**
  1. **Concept shadowing by `create_descriptor` is a silent-no-op hazard the recipe does not name.**
     See TTNN ProgramFactory note 2. Worth a recipe entry — it is the natural first idea anyone has when
     an out-of-op consumer blocks a port, and it fails green.
  2. **`slice_tile_dynamic_args` is shared with the unported `SliceTileTensorArgsProgramFactory`.** It
     lives in `slice_program_factory_tile.cpp:198-281` and is called from
     `slice_program_factory_rm_sharded.cpp:407` for *both* tile factories. The port must leave it in
     place (the tensor-args factory still needs it) while no longer using it for this factory. It is a
     free function in the op's own directory, so this is bookkeeping, not a shared-kernel case.
  3. **`#52651` is referenced in-code** (`slice_program_factory_tile.hpp:29`,
     `slice_program_factory_rm_sharded.cpp:396`) — a known divergent-partition cache-hit bug. Preserved,
     not fixed; the ported override refreshes exactly the same set the legacy one did.

- **Remaining factories for a later pass** (each a complete sub-port of its own):
  | Factory | Kernels | Notable |
  |---|---|---|
  | `SliceRmProgramFactory` | 2 owned | RM reader hardcodes its DFB index in-kernel (no host CTA to drop); vararg blocks at runtime-computed addresses |
  | `SliceRmShardedProgramFactory` | 1 owned | Two **borrowed-memory** DFBs, two **DM self-loops** (Gen1-legal, Quasar-uplift debt); hosts the shared `patch_slice_program_addresses` |
  | `SliceRmStrideProgramFactory` | 4 owned | Runtime kernel-source selection on rank (4D vs ND) — all four sources must convert together |
  | `SliceTileTensorArgsProgramFactory` | 1 owned + 1 **borrowed** | Binds the existing `writer_unary_interleaved_start_id_metal2.cpp` fork (rung 1); `c_1` needs a self-loop |
