# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `f9451e2a21d 2026-09-09 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

## What you are porting

One directory, **three DeviceOperations**, **eight factories**, **eleven kernels**. All eleven kernels
are owned by this op, referenced by a factory, and Device 2.0 clean. Port them together — they share
the kernel tree and the same host-side structure.

| DeviceOperation | Factories | Kernels |
|---|---|---|
| `PagedFillCacheDeviceOperation` | `PagedFillCacheProgramFactory`, `PagedFillCacheMeshWorkloadFactory` | `reader_fill_cache_interleaved.cpp`, `writer_fill_cache_interleaved.cpp` (no compute kernel) |
| `PagedUpdateCacheDeviceOperation` | `PagedUpdateCacheProgramFactory`, `PagedUpdateCacheMeshWorkloadFactory` | `reader_update_cache_interleaved_start_id.cpp`, `writer_update_cache_interleaved_start_id.cpp`, `compute/update_cache.cpp` |
| `PagedFusedUpdateCacheDeviceOperation` | `PagedTiledFusedUpdateCacheProgramFactory`, `PagedTiledFusedUpdateCacheMeshWorkloadFactory`, `PagedRowMajorFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` | `reader_paged_fused_update_cache_interleaved_start_id.cpp`, `writer_paged_fused_update_cache_interleaved_start_id.cpp`, `compute/paged_fused_update_cache.cpp` (tiled); `reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp`, `compute/paged_row_major_fused_update_cache.cpp` (row-major) |

Each `*MeshWorkloadFactory` delegates its program build to its single-device sibling, and the two
`fill_cache` factories share one `override_runtime_arguments` body — so there are **four distinct
program bodies** and **four distinct cache-hit patch bodies** to translate, not eight of each (the two
fused patch bodies are thin wrappers over one shared `patch_runtime_args` helper).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` — every one of the eight factories defines `create_descriptor`
  returning `tt::tt_metal::ProgramDescriptor`. No `create_workload_descriptor` anywhere.
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** for all eight factories — `descriptor` plus
  an `override_runtime_arguments` on every factory. The readiness sheet's own `Porting Target` column
  agrees on all eight rows.
- **Custom hash:** present on all three device-ops
  ([`device/fill_cache/paged_fill_cache_device_operation.cpp:216-225`](device/fill_cache/paged_fill_cache_device_operation.cpp#L216-L225),
  [`device/update_cache/paged_update_cache_device_operation.cpp:314-338`](device/update_cache/paged_update_cache_device_operation.cpp#L314-L338),
  [`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:372-390`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L372-L390)).
  **Leave each exactly as it is.** Each hash deliberately *excludes* scalar attributes
  (`update_idxs`, `batch_offset`, `batch_idx_fallback`, `noop`) that the matching
  `override_runtime_arguments` then re-applies on every cache hit. The exclusion and the re-application
  are two halves of one design; if you translate the override, keep it re-applying the same set.
- **Pybound `create_descriptor`:** none. [`paged_cache_nanobind.cpp:23-165`](paged_cache_nanobind.cpp#L23-L165)
  binds only the three user-facing op functions, so this port has **no user-visible API change** to
  report.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation`
  that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args` (deprecated hook). A
  custom hash and an `override_runtime_arguments` are **not** in this list — both are present here and
  neither gates.

**One open question, and it gates only three of the eight factories** (audit, *Question for the
user*). `PagedUpdateCacheMeshWorkloadFactory` and the two mesh variants of the fused op return an
**empty `ProgramDescriptor`** for a mesh coordinate outside `mesh_coords`
([`device/update_cache/paged_update_cache_program_factory.cpp:448-453`](device/update_cache/paged_update_cache_program_factory.cpp#L448-L453),
[`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:544-549`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L544-L549),
[`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:542-548`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L542-L548))
— a structurally different program per coordinate, whereas both Metal 2.0 spec concepts are SPMD-shaped:
one `ProgramSpec` replicated across the mesh
([`../../../../../../ttnn/api/ttnn/operation_concepts.hpp:134-140`](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L134-L140)).

**Start with the five factories the question does not touch** — the plain `paged_fill_cache` and
`paged_update_cache` factories, `PagedFillCacheMeshWorkloadFactory`, and the two plain fused
factories. `PagedFillCache`'s
mesh variant is already in the recommended shape: its exclusion rides on a `noop` **runtime arg** that
its `override_runtime_arguments` re-derives per coordinate
([`device/fill_cache/paged_fill_cache_program_factory.cpp:399`](device/fill_cache/paged_fill_cache_program_factory.cpp#L399)),
which is exactly what a replicated spec plus per-coordinate `ProgramRunArgs` expresses.

**The recommended answer ports the other three too:** on an excluded coordinate, build the full program
and set an all-cores `has_work = 0`. Your kernels need no change — each already reads `has_work` first
and returns on 0
([`device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:16-20`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L16-L20)).
Under that answer `has_work` **changes role rather than disappearing**: from "is this core inside a
shard grid" (which the `WorkUnitSpec` narrowing in *Watch for*, below, makes always-true) to "is this chip inside
`mesh_coords`". Do not delete it.

**If the answer is no**, those three stay on the legacy `ProgramDescriptor` path and are **outside your
scope** — report them as deliberately not ported, not as a failed port. That mixed shape is supported:
the framework dispatches per variant alternative, so a `program_factory_t` may hold both a descriptor
factory and a spec factory
([`../../../../../../ttnn/api/ttnn/device_operation.hpp:239-269`](../../../../../../ttnn/api/ttnn/device_operation.hpp#L239-L269)).
It costs code duplication: each mesh variant currently delegates its build to the plain factory, so
that legacy `create_descriptor` body has to be retained once the plain factory returns a `ProgramSpec`.

## Construct — to do

### Tensor bindings (per binding)

Every legacy address delivery is the `Buffer*`-binding form: the factories push a `Buffer*` into
`KernelDescriptor::RTArgList` / `emplace_runtime_args`, never an `->address()` expression. So there is
**no stale-address hazard to repair** — just fifteen bindings to place, twelve of them as a
`TensorParameter` / `TensorBinding` and three as a `borrowed_from` DFB. **No Case 2 anywhere:** no kernel in the op does hand-rolled NoC arithmetic on a
tensor base, so you never need the `get_bank_base_address` bridge.

**`PagedFillCacheDeviceOperation`** — 5 bindings, all Case 1:

- `input_tensor` — **Case 1** → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::input)` in place of [`reader_fill_cache_interleaved.cpp:29`](device/kernels/dataflow/reader_fill_cache_interleaved.cpp#L29).
- `cache_tensor` (in-place output) — **Case 1** → replaces [`writer_fill_cache_interleaved.cpp:145`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L145).
- `page_table` — **Case 1** → replaces [`writer_fill_cache_interleaved.cpp:146`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L146).
- `batch_idx_tensor_opt` *(optional)* — **Case 1** → replaces [`writer_fill_cache_interleaved.cpp:101`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L101). Conditional binding — see *Watch for*.
- `valid_seq_len_tensor_opt` *(optional)* — **Case 1** → replaces [`writer_fill_cache_interleaved.cpp:122`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L122). Conditional binding — see *Watch for*.

**`PagedUpdateCacheDeviceOperation`** — 3 Case 1 + 1 clean:

- `cache_tensor` (in-place output) — **Case 1** → replaces [`reader_update_cache_interleaved_start_id.cpp:69`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L69) and [`writer_update_cache_interleaved_start_id.cpp:55`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L55).
- `update_idxs_tensor` *(optional)* — **Case 1** → replaces [`reader_update_cache_interleaved_start_id.cpp:74`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L74). No config split here: the `c_2` CB is never buffer-backed in this factory, so the tensor is always read through an accessor.
- `page_table` *(optional)* — **Case 1** → replaces [`reader_update_cache_interleaved_start_id.cpp:95`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L95).
- `input_tensor` — **clean** (borrowed-memory DFB) → `DataflowBufferSpec::borrowed_from` the
  `input_tensor` parameter, translating `.buffer = in1_buffer` at
  [`device/update_cache/paged_update_cache_program_factory.cpp:200-209`](device/update_cache/paged_update_cache_program_factory.cpp#L200-L209).
  The reader `reserve_back`/`push_back`s without writing and compute untilizes straight out of it — the
  borrowed DFB *is* the tensor access. Nothing else to do for this binding.

**`PagedFusedUpdateCacheDeviceOperation`** (both factories) — 2 Case 1 + 2 clean + 2 config-split:

- `cache_tensor1`, `cache_tensor2` (in-place outputs) — **Case 1**. Both reach the *same*
  `TensorAccessorArgs` slot, one per core group: the tiled reader gets `dst1_buffer` on `cores1[i]`
  and `dst2_buffer` on `cores2[i]`
  ([`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:438`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L438)
  and `:483`). Once you split the kernels per core group (see *Watch for*), each `KernelSpec` binds one
  of the two cache tensors and the choice becomes structural.
- `input_tensor1`, `input_tensor2` — **clean** (borrowed-memory DFBs `c_1` / `c_2`) → `borrowed_from`,
  translating [`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:203-222`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L203-L222)
  and [`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:208-227`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L208-L227).
- `update_idxs_tensor` *(optional)* — **Case 1 when DRAM, clean when L1-sharded.** The factory sets
  `.buffer = index_buffer_ptr` on the `c_3` CB only for a sharded tensor
  ([`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:267-279`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L267-L279)),
  and the reader then reads it out of the borrowed CB with no NoC read
  ([`device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:90-96`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L90-L96)).
  So: `borrowed_from` in the sharded config, `TensorParameter` + accessor in the DRAM config. The
  legacy code passes the address argument in **both** configs and builds the accessor unconditionally
  at [`device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:87`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L87)
  while using it only under `if constexpr (index_is_dram)` — the argument and the accessor are dead in
  the sharded config, so do not carry them over into it.
- `page_table` *(optional)* — **Case 1 when DRAM, clean when L1-sharded.** Same split, driven by
  `page_table_is_dram`: borrowed `c_4` CB at
  [`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:280-291`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L280-L291),
  and the reader either NoC-reads through the accessor or just offsets the borrowed CB's write pointer
  ([`device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:108-119`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L108-L119)).

**TensorParameter relaxation:** `none`. No relaxation, no `dynamic_tensor_shape`, no analysis doc to read.

**TensorAccessor 3rd arg:** none — no accessor in this op passes a 3rd argument. All 17 construction
sites are the 2-argument form, so there is nothing to drop.

### CB endpoints

Twenty-seven `(CB, config)` instances. **No multi-binding advanced option is needed anywhere, and no CB
is dead.** Twenty-four are plain 1P+1C FIFOs with nothing to do; the rest:

- **Self-loop `(c_1, all)`, `(c_2, use_batch_idx_tensor)`, `(c_3, use_valid_seq_len)` in `fill_cache`** —
  each is touched by the **writer only**, which `reserve_back(1)`s, takes `get_write_ptr()`, NoC-reads
  metadata in and reads it back through an L1 pointer, never `push_back`ing
  ([`device/kernels/dataflow/writer_fill_cache_interleaved.cpp:148-150`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L148-L150)
  for the page table; `:101-103` and `:122-124` for the other two). Bind the writer **PRODUCER and
  CONSUMER** on all three. Kernel code unchanged.
- **Conditional DFB on eight CBs** — the index CB and the page-table CB in `update_cache` (`c_2` / `c_3`)
  and in each fused factory (`c_3` / `c_4`), plus `fill_cache`'s batch_idx and valid_seq_len CBs. Each
  `DataflowBufferSpec` stays conditional on the same host flag the legacy factory already gates the
  allocation on. **Do not drop any of them and do not make them unconditional** — see the first
  *Watch for* item for the preprocessor work each one needs.

The `(is_paged_cache && !use_index_tensor)` config, which would have left a page-table CB with zero
endpoints, is **unreachable**: all three device-ops assert that paged mode requires an index tensor
([`device/update_cache/paged_update_cache_device_operation.cpp:149`](device/update_cache/paged_update_cache_device_operation.cpp#L149),
[`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:213`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L213)).
No dead-CB drop is warranted in this op; if the spec validator rejects a bindingless DFB during your
build, that is a sign you mis-scoped a conditional binding, not that you found a dead CB.

The full per-`(CB, config)`-per-node census, including which kernel takes which role on each CB, is in
`METAL2_PREPORT_AUDIT.md` under *Port-work summary → CB endpoints*. Read it before you write the
bindings; the CB-index-to-name mapping differs per kernel (the writers' compile-time arg 0 is the
*output* CB, not the cache CB) and it is easy to bind the wrong one from the kernel source alone.

## Watch for

- **Conditionally-bound resources — the largest single piece of work, and the gate must move to the
  preprocessor.** Eight CBs and four optional tensors exist only under a host-side flag, while the
  kernels declare `CircularBuffer` objects for them **unconditionally at function scope** and gate only
  the *use* behind `if constexpr`. `if constexpr` in a non-template function still performs name lookup
  on the discarded branch, so `dfb::cb_index` must exist at parse time or the off-path build fails to
  compile. Apply *Pattern: Conditional / optional resource bindings* from `port_patterns.md`: bind
  conditionally on the host, emit a matching define through `KernelSpec::compiler_options.defines`, and
  `#ifdef`-gate the kernel-side alias and every expression that names it. The declarations to convert:
  - [`device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:56-57`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L56-L57) — `cb_index`, `cb_page_table`, used under the `if constexpr (use_index_tensor)` at `:73` and the `if constexpr (is_paged_cache)` at `:94`
  - [`device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp:61-62`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L61-L62) — same pair
  - [`device/kernels/dataflow/writer_fill_cache_interleaved.cpp:92-93`](device/kernels/dataflow/writer_fill_cache_interleaved.cpp#L92-L93) — `cb_batch_idx`, `cb_valid_seq_len`
  - [`device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:69-70`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L69-L70) and [`device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp:66-67`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L66-L67)
  - [`device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:69-70`](device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L69-L70) and [`device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:73-74`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L73-L74)

  The four optional **tensors** — `update_idxs_tensor`, `page_table`, `batch_idx_tensor_opt`,
  `valid_seq_len_tensor_opt` — are the mandatory case rather than the preferred one: the factories pass
  `TensorAccessorArgs(nullptr)` for an absent one
  ([`device/fill_cache/paged_fill_cache_program_factory.cpp:261-264`](device/fill_cache/paged_fill_cache_program_factory.cpp#L261-L264),
  [`device/update_cache/paged_update_cache_program_factory.cpp:309-311`](device/update_cache/paged_update_cache_program_factory.cpp#L309-L311)),
  and an absent tensor emits no `tensor::` token at all, so there is nothing to bind even in principle.
  Note also that the define has to reach **every** kernel that names the resource, not just the one the
  legacy factory happened to send a flag to: `use_index_tensor` and `is_paged_cache` are read by the
  reader *and* the writer in all three update paths.

- **The fused factories select their input CB index at runtime, from a kernel placed over a core range
  wider than the CB it binds.** This is the second big piece of work, and it is forced rather than chosen:
  only one placement is expressible.

  Both fused factories place reader, writer and compute over `all_cores_bb` — the bounding box of the
  two input shard grids — while the input CBs `c_1` and `c_2` are scoped to `input1_cores` and
  `input2_cores`:
  - tiled: kernels at [`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:370`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L370), `:380`, `:389`; CBs at `:205` and `:215`
  - row-major: kernels at [`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:366`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L366), `:376`, `:386`; CBs at `:210` and `:220`

  Each kernel then picks between the two from a per-core runtime arg, `is_input1`:
  [`device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:32-35`](device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp#L32-L35),
  [`device/kernels/compute/paged_fused_update_cache.cpp:21-25`](device/kernels/compute/paged_fused_update_cache.cpp#L21-L25) and `:39-55`,
  [`device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:32-35`](device/kernels/dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L32-L35),
  [`device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:58-61`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L58-L61).
  Cores in the bounding box belonging to neither shard grid get a one-element runtime-arg vector
  `{!has_work}` and early-exit
  ([`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:524-530`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L524-L530),
  [`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:519-525`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L519-L525)),
  so one legacy kernel carries two different arg-vector lengths across cores.

  A `dfb::` token is a compile-time name, and a DFB has **no core range of its own** — its node set is
  derived from the union of its bound kernels' `WorkUnitSpec::target_nodes`
  ([`../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:57-59`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp#L57-L59)).
  So one `KernelSpec` over `all_cores_bb` binding both input DFBs would place `c_1` on `input2_cores`
  nodes as well, where the `borrowed_from` `input_tensor1` has no resident shard.

  The documented translation is the shape the *Anti-pattern: Demoting per-group CTA to RTA* entry in
  `port_patterns.md` states from the other direction: **two `WorkUnitSpec`s over the two disjoint node
  sets**, each with its own reader / writer / compute `KernelSpec` of the same source, differing only
  in which input DFB and which cache `TensorParameter` it binds. That makes the CB choice compile-time,
  dissolves `is_input1` into structure, and removes the ragged arg vectors. The device-op already
  guarantees the two grids are disjoint and equal in size
  ([`device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:352-357`](device/fused_update_cache/paged_fused_update_cache_device_operation.cpp#L352-L357)),
  which is exactly the non-overlapping-coverage condition the framework validates.

  **Two things to surface in the port report rather than doing silently:**
  1. The `unused_cores` instances disappear, and that is a **net gain, not a risk** — no decision
     needed, just a line in the port report. Those cores early-exit today and are absent afterwards,
     so no output moves; the op dispatches to fewer cores and stops reserving roughly **59–112 KB on
     each** of them. **This is not hypothetical: `unused_cores` holds 20 cores in the llama3-70b
     galaxy decode configuration** (the audit derives it under *Team-only → `unused_cores` in a
     shipping configuration*). It is empty in the unit tests, and empty in general whenever both input
     grids come straight from `nlp_create_qkv_heads_decode` onto a full-width rectangular grid, so do
     not expect a local test to show it. The only observable differences are a profiler trace showing
     fewer cores and an SRAM report showing less reserved — record both.
  2. `compute_kernel_hw_startup(in_cb, untilized_in_cb)` in the tiled compute kernel
     ([`device/kernels/compute/paged_fused_update_cache.cpp:36`](device/kernels/compute/paged_fused_update_cache.cpp#L36))
     currently takes a *runtime-selected* CB index. After the split it becomes a compile-time value
     again, which is what the LLK expects.

- **Cross-op / shared kernels:** **none — no coordination cost, no sunset list.** The op owns all 11
  kernel files, no other op in the tree instantiates any of them, and **no `_metal2` fork exists beside
  any of them**, so this port creates the first fork for each. There is also **no `paged_cache` copy
  under `ttnn/cpp/ttnn/operations/experimental/quasar/`** — and if you find one appear, do not read it:
  that tree holds deliberately hacky shortcut ports that carry idioms this recipe forbids.

  The only out-of-directory code the kernels call is `compute_kernel_lib::untilize` and
  `compute_kernel_lib::tilize`
  ([`../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:145-154`](../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp#L145-L154),
  [`../../../../../../ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:153-163`](../../../../../../ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp#L153-L163)).
  Both take their handles as `uint32_t` non-type template parameters already named `input_dfb` /
  `output_dfb` and construct `DataflowBuffer` objects internally
  ([`../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl:199-200`](../../../../../../ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl#L199-L200)).
  Pass the `dfb::` tokens straight through — the constexpr cast covers template-parameter position.
  **No donor-side change and no donor fork.**

- **RTA varargs:** **none.** Every runtime-arg read in the op is a distinct field at a fixed position,
  so **name every one**. Two shapes, both non-signal: constant indices in the `fill_cache` and
  `update_cache` kernels (`get_arg_val<uint32_t>(0)` … `(6)`), and a fixed run of
  `get_arg_val<uint32_t>(rt_args_idx++)` at the top of each of the six fused kernels. There is no
  counted loop over runtime args, no data-selected index, and no `get_common_arg_val` /
  `common_runtime_args` use anywhere in the op. **CTA varargs: also none** — no compile-time arg is
  read at a varying index.

- **The writers' CB-index-to-name mapping is deliberately confusing; check the factory, not the kernel
  name.** In `update_cache` and the tiled fused path the writer's compile-time arg 0 is called
  `cache_cb_id` inside the kernel but is wired to the **output** CB (`c_16`), while the reader's
  compile-time arg 0 with the same role name is the actual cache CB (`c_0`). Bind from the factory's
  compile-time-arg list, not from the kernel-side identifier.

- **`DataflowBuffer` exposes its own tile/format metadata, so eight `get_tile_size()` sites move onto the
  object.** The kernels currently read tile size through the Device 2.0 `CircularBuffer` wrapper (e.g.
  [`device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:63`](device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp#L63)).
  Moving them is port-stage work per the kernel-side whitelist, not a Device 2.0 change — **confirm the
  DFB equivalent rather than swapping blind.** Each kernel's `#include "api/dataflow/circular_buffer.h"`
  becomes the DFB header at the same time.

- **`my_x[noc_id]` / `my_y[noc_id]` in the three writers is not a Device 2.0 holdover — leave it
  alone.** The writers build a local NoC address for an intra-core L1→L1 copy out of the HAL's own-core
  coordinate globals, fed to a `UnicastEndpoint`, with `noc_id` from `noc.get_noc_id()`
  ([`device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp:114-131`](device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp#L114-L131)).
  That is the Device 2.0 path already; the Metal 2.0 port does not touch it.

- **Three semaphores, all the same plain `SemaphoreSpec` translation.** One per factory except
  `fill_cache`, which has none:
  [`device/update_cache/paged_update_cache_program_factory.cpp:246-252`](device/update_cache/paged_update_cache_program_factory.cpp#L246-L252),
  [`device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:259-265`](device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp#L259-L265),
  [`device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp:255-261`](device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp#L255-L261).
  Each serializes the `share_cache` cache-read ordering between consecutive worker cores; kernel-side it
  is already a `Semaphore<>` object. **It is allocated unconditionally, even when `share_cache` is
  false and no kernel ever touches it** — that is existing behavior; keep it unconditional so the port
  changes nothing. (It is recorded as an anomaly for the ops team in `METAL2_PREPORT_AUDIT.md`.)

- **Several dead compile-time args are wired from the host and will look like real bindings.** Do not
  build spec plumbing for them, and do not remove them either — they are the ops team's, not the
  port's. `max_blocks_per_seq` (six kernels), `log_base_2_of_page_size` (three readers, always literal
  `0`), and `log2_page_table_stick_size` (four kernels) are read into a variable and never used. The
  row-major fused compute kernel's compile-time args 0 and 1 plus its `is_input1` runtime arg are
  likewise dead
  ([`device/kernels/compute/paged_row_major_fused_update_cache.cpp:19-25`](device/kernels/compute/paged_row_major_fused_update_cache.cpp#L19-L25))
  — that kernel never touches the input CBs, because in the row-major path the **writer** drains them.
  The full list with sites is in `METAL2_PREPORT_AUDIT.md` under *Misc anomalies*.

- **The two fused writers disagree on whether to drain the index and page-table CBs.** The tiled writer
  `pop_front`s both
  ([`device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp:112`](device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp#L112), `:122`);
  the row-major writer `wait_front`s and never pops
  ([`device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:84`](device/kernels/dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp#L84), `:95`).
  Both work because each CB is filled and read exactly once per execution, so the FIFO never wraps.
  **Preserve each as-is** — do not "fix" the row-major one to match its sibling; that would be a
  functional change outside the port's scope.
