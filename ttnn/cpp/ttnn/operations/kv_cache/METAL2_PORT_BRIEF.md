# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/kv_cache/`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `63ca139b420 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources` *(carry this line into the port report's Provenance section)*

Two factories on one device operation, ported together:
- `UpdateCacheMultiCoreProgramFactory` (`device/update_cache_multi_core_program_factory.cpp`) — the UPDATE/decode path (reader + writer + compute, up to two compute kernels for two core groups).
- `FillCacheMultiCoreProgramFactory` (`device/fill_cache_multi_core_program_factory.cpp`) — the FILL/prefill path (reader + a borrowed unary writer).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` (both factories — `create_descriptor` returning a `ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** (both factories) — each carries an `override_runtime_arguments` (`update_cache_multi_core_program_factory.cpp:436`, `fill_cache_multi_core_program_factory.cpp:262`). Translate each into a method returning `ProgramRunArgs` (see the recipe's *translating `override_runtime_arguments`* step).
- **Custom hash:** present — `UpdateKVCacheOperation::compute_program_hash` at `update_cache_device_operation.cpp:160`. **Leave it exactly as-is.** It hashes `op_type` + the two tensors and deliberately excludes `batch_idx`/`update_idx`/`batch_offset` and `compute_kernel_config`, which is why the override exists (below).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation` · `get_dynamic_runtime_args`. A custom hash and an `override_runtime_arguments` are present but do **not** gate.

## Construct — to do

**Tensor bindings** (per binding — express each as a `TensorParameter` / `TensorBinding`; drop the `Buffer*` `emplace_runtime_args` delivery and the address re-application in the override):

- `cache` (the in-place cache = output) — **Case 1** (via `TensorAccessor`), both factories, all configs. Kernels build `TensorAccessor(tensor::cache)` and address by `page_id`. The reader reads it (`reader_update:38`), the update writer writes it (`writer_update:44`), the fill writer (donor) writes it.
- `input` — **Case 1** (via `TensorAccessor`) in the **interleaved** config (`reader_update:43`, `reader_fill:29`); **clean / borrowed-memory DFB** in the **sharded** config — the input CB is `.buffer = src_buffer` and the reader only `reserve_back`/`push_back`s under `INPUT_SHARDED`. Express the sharded input CB with `DataflowBufferSpec::borrowed_from(tensor::input)`; there is no `TensorAccessor` on that path. (This is the existing `set_globally_allocated_address` + `UpdateDynamicCircularBufferAddress` pair — `update_cache_...cpp:489-492`, `fill_cache_...cpp:298-301` — translated mechanically.)

**Override translation (both factories):** after binding the tensors, the tensor **addresses** refresh through the typed channel automatically on cache hit. The translated `override_runtime_arguments` must therefore re-apply only the **non-address, hash-excluded** per-dispatch scalars:
- update: per-core `cache_start_id`, plus op-wide `Wbytes`, `tile_update_offset`, `batch_read_offset`.
- fill: per-core `cache_start_id`.

Keep `compute_update_cache_dynamic_args` (`update_cache_multi_core_program_factory.hpp:52`) and `compute_fill_cache_start_ids` (`fill_cache_multi_core_program_factory.hpp:41`) as the single shared source of truth for the work-split + formulas between the create path and the override — the core order and values must not drift.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes one; nothing to drop.

**CB endpoints:**
- **All legal 1:1, bind normally**, except the interm CB below:
  - update: `c_0` (cache), `c_1` (input), `c_26` (untilized_input), `c_16` (output) — one producer, one consumer each.
  - fill: `c_0` (input reused as pass-through output) — reader produces, donor writer consumes.
  - sharded configs: the input CB (`c_1` update / `c_0` fill) is borrowed-memory — bind `borrowed_from(tensor::input)`, still 1 producer + 1 consumer.
- **The interm0/interm1 CB (`c_24`/`c_25`) is a legacy aliased CB → port as Aliased DFBs.** `create_descriptor` builds **one** `CBDescriptor` with **two** `CBFormatDescriptor`s (`update_cache_multi_core_program_factory.cpp:235-261`) — two distinct `buffer_index`es (`c_24`, `c_25`) sharing one L1 region, which is what makes the writer's data flow correct. This is the recipe's **Pattern: Aliased DFBs (legacy aliased CBs)** (`shared/port_patterns.md`): declare **one `DataflowBufferSpec` per `buffer_index`** (two specs) and set each spec's `advanced_options.alias_with` to name the other (a strict clique). The two already satisfy the legality rules — identical `num_entries * entry_size` (both derive from the one `CBDescriptor`'s `total_size`) and the same kernel set (`{compute, writer}`). Under this model each aliased DFB is a **clean 1P+1C**, *not* a multi-binding:
  - `c_24` — compute **PRODUCER** (`untilize` output) + writer **CONSUMER** (`wait_front`/`pop_front`, including the raw in-place L1 poke at `writer_update:60-68`, which is a write through the writer's own consumer binding).
  - `c_25` — writer **PRODUCER** (`reserve_back`/`push_back`) + compute **CONSUMER** (`tilize` input).

  The distinct `buffer_index`es settle this as **Aliased DFBs**, not Same-FIFO aliasing, *even though* the writer walks both views at matching L1 addresses — see the "converse trap" note in `port_patterns.md` ("the disambiguator is the index, not the runtime pointer values"). **Don't** split them into independent non-aliased DFBs (changes the L1 footprint), and **don't** reach for the multi-binding advanced option.

## Watch for

- **CB endpoints (aliased DFBs):** the `c_24`/`c_25` interm CB is a legacy aliased CB — port it as two `alias_with` DFBs per the Construct section (each a clean 1P+1C, **no** multi-binding flag). The writer's raw L1 poke (`writer_update:60-68`) is a write through the writer's own `c_24` consumer binding, not a separate tensor access — it adds no endpoint and is not a TensorParameter binding.
- **Cross-op / shared kernels:** the fill factory instantiates the cross-family donor `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`. A `_metal2` fork **already exists** beside it (`writer_unary_interleaved_start_id_metal2.cpp`) — **bind the existing fork; do not re-fork and do not convert the legacy file in place.** Other ops also borrow this writer (sunset list tracked in issue #52228 — that is a coordination/sunset list, **not** authorization to convert the kernel in place). Ignore the unrelated `_wh.cpp` variant. `compute/update_cache.cpp` uses `kernel_lib/` helpers (shared-lib, no fork needed).
- **RTA varargs:** none — name every runtime arg.
- **Anything else:**
  - The donor `writer_unary_interleaved_start_id.cpp` and the op's own kernels are already Device 2.0 — this is a binding-layer port, not an idiom rewrite. The update kernels use `get_tile_size(cb_id)` and the donor uses `get_local_cb_interface(cb_id).fifo_page_size` — both **sanctioned** free functions; a Metal 2.0 port may move those lookups onto the DFB object (kernel-side whitelist rule 7), but that is optional, not required.
  - Two dead-ish includes in `update_cache_device_operation.hpp` (`global_circular_buffer.hpp:13`, `program_descriptor_patching.hpp:14`) are noted in the audit's Misc — not port work, leave them for the ops team unless they block a clean build.
