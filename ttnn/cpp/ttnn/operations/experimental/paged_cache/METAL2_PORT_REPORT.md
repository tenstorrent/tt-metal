# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

## Outcome

**`PORTED`** — **all four factories this recipe's procedure covers are ported.** The op's eight
factories split three ways, and the split is now clean:

- **`PORTED` (pass 1)** — `PagedUpdateCacheProgramFactory` and `PagedFillCacheProgramFactory` on
  `CustomProgramSpecFactoryConcept`, with five `_metal2` kernel forks.
- **`PORTED` (pass 2, this pass)** — `PagedTiledFusedUpdateCacheProgramFactory` and
  `PagedRowMajorFusedUpdateCacheProgramFactory`, same concept, with six more `_metal2` forks. These
  implement the answer pass 1 worked out to the audit's open **Question #1**, on both binding
  channels; see [Open items](#open-items-for-downstream) #1 for the design and
  [Friction](#friction) #4 for the one place the inherited design was incomplete.
- **Out of this procedure's scope** — the four `*MeshWorkloadFactory` factories. Pass 1 recorded them
  as blocked on a missing framework capability; **that capability has since merged** (`9fb0ed54794`,
  PR #54988: `MeshWorkloadSpecFactoryConcept`). They are therefore no longer *blocked* — but their
  target concept is that new one, and the port recipe bounds itself to
  `ProgramSpecFactoryConcept` / `CustomProgramSpecFactoryConcept` and names this exact case as one to
  stop on rather than improvise. Full detail, including the design work already done against the
  merged concept, in [Handoff points](#handoff-points) #1.

Together the four ported factories are every factory `select_program_factory` picks whenever
`mesh_coords` is `nullopt` — the default on all three public entry points, so this is the whole
common path across all three `DeviceOperation`s.

### What the remaining four factories are waiting on

One prerequisite, shared by all four, and it is **not** a capability gap any more:

| Remaining factories | Waiting on | Nature |
|---|---|---|
| `PagedUpdateCacheMeshWorkloadFactory`, `PagedFillCacheMeshWorkloadFactory`, `PagedTiledFusedUpdateCacheMeshWorkloadFactory`, `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` | **a port procedure for `MeshWorkloadSpecFactoryConcept`** | process, not framework. The concept and its adapter are on this branch (`9fb0ed54794`); what does not exist is a recipe section telling a porter how to target it, and the current recipe explicitly declines to have one improvised. |

**Built and tested — see [Verification performed](#verification-performed).** Unlike pass 1 (which
ran on a machine without `clang-20` and could only check statically), this pass built the tree and
ran the confirmed test set on hardware, with the Metal 2.0 legality checks forced on and proven live.
That also gives pass 1's two factories their first real execution.

## Provenance

- **Recipe docs (pass 2):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Recipe docs (pass 1):** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`

The working checkout carries no `metal_2.0` doc tree, so
`git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
prints nothing there. The hashes above are pinned from the sibling doc-branch checkout
`/localdev/edwinlee/Port_Recipe`, whose `ai/port/metal2_port.md` was verified byte-identical
(`diff -q`) to the recipe each pass was handed. Pass 2 ran against a **newer** doc revision than
pass 1; the recipe text itself is unchanged in every respect this port depends on, including the
coverage-boundary rule that keeps the four mesh factories out of scope.

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept`, as the audit chose, on all four ported factories. Each implements
`create_program_artifacts` plus a `ProgramRunArgs`-returning `override_runtime_arguments`.

Each `program_factory_t` variant is now **mixed-concept** — one or two alternatives on
`CustomProgramSpecFactoryConcept`, the rest on `ProgramDescriptorFactoryConcept`. `AllFactoriesValid`
permits this (each alternative satisfies exactly one concept) and the framework dispatches
per-factory at runtime, so `PagedFusedUpdateCacheDeviceOperation`'s four-alternative variant builds
and runs with two of its alternatives on each API.

**Cache-hit tensor-binding completeness** (the silent failure `ttnn_factory.md` warns about — on this
concept the framework refreshes *nothing* for you): each override returns a `TensorArgument` for
**every** `TensorParameter` the spec declares, in every configuration. Nothing is skipped.

- `PagedUpdateCacheProgramFactory::override_runtime_arguments` → `cache`, `input`, and (when present)
  `index`, `page_table`. The `input` entry is load-bearing beyond its own accessor: it is what
  refreshes the **borrowed-memory** `input` DFB's backing L1 address, the job the ported-from body
  did with `UpdateDynamicCircularBufferAddress`.
- `PagedFillCacheProgramFactory::override_runtime_arguments` → `input`, `cache`, `page_table`, and
  (when present) `batch_idx`, `valid_seq_len`.
- `PagedTiledFusedUpdateCacheProgramFactory::override_runtime_arguments` and the row-major sibling →
  `cache1`, `cache2`, `input1`, `input2`, and (when present) `index`, `page_table`. Four of the six
  are load-bearing beyond their own accessors: `input1` / `input2` refresh the borrowed-memory
  `src1` / `src2` DFBs' backing L1 addresses, and `index` / `page_table` refresh theirs **when those
  tensors are L1-sharded** — the four `UpdateDynamicCircularBufferAddress` calls the ported-from
  `patch_runtime_args` made (`paged_fused_update_cache_device_operation.cpp:73-81`). Note the
  ported-from code skipped the CB re-point for a *non*-sharded index / page table (its `.buffer` was
  `nullptr`); on this side the same conditionality lives in the DFB spec's `borrowed_from`, so the
  `TensorArgument` can be supplied unconditionally and the framework re-points only what borrows.

**Non-tensor refreshes mirror the ported-from set exactly, no more and no less:**

| ported-from override wrote | Metal 2.0 |
|---|---|
| `update_cache` reader `[1]`, writer `[1]`/`[2]` — only when `offsets` is non-empty | `cache_start_id` / `cache_tile_offset_B` named RTAs, under the same `offsets.empty()` guard |
| `update_cache` reader `[0]`/`[2]`/`[4]`, writer `[0]`, and the input CB re-point | `tensor_args` (4 entries) |
| `fill_cache` reader `[3]`, writer `[5]` (`noop`) | `noop` named RTA on both kernels |
| `fill_cache` writer `[4]` (`batch_idx_fallback`, scalar path only) | `batch_idx_fallback` named RTA, declared and refreshed only on the `!use_batch_idx_tensor` path |
| `fill_cache` reader `[0]`, writer `[0]`/`[1]`/`[4]`(tensor path)/`[6]` | `tensor_args` (3–5 entries) |
| both fused: reader `[3]`, writer `[2]`/`[3]` on `cores1[i]` **and** `cores2[i]` — only when `offsets` is non-empty | `cache_start_id` / `cache_tile_offset_B` named RTAs on the same two node lists, under the same `offsets.empty()` guard |
| both fused: reader `[2]`/`[4]`/`[6]`, writer `[1]`, and the four CB re-points | `tensor_args` (4–6 entries) |
| *(neither: `start_tile_id` / `start_row_num` / `num_rows` / `my_batch_idx` / `wait_to_start` / `send_*` / `has_work` / `is_input1`)* | **not** refreshed — identical, deliberately |

`UpdateProgramRunArgs` is a partial update, so everything omitted keeps its cache-miss value —
which is exactly the ported-from behaviour.

### Device-op-class edits

- **Pybind entry points removed: none.** `paged_cache_nanobind.cpp` binds only the three public entry
  points via `ttnn::bind_function` (`:48`, `:89`, `:134`); no `create_descriptor` was ever pybound, so
  the port makes **no user-visible API change**.
- **Custom `compute_program_hash`: left intact, untouched**, on all three DeviceOperations —
  `paged_update_cache_device_operation.cpp:313`, `paged_fill_cache_device_operation.cpp:207`,
  `paged_fused_update_cache_device_operation.cpp:371`.
- **No op-level device-operation-class code was edited** — no `validate_on_program_cache_miss`, no
  `compute_output_specs`, no `create_output_tensors`, no `select_program_factory`, no attribute
  parsing, no public entry point. The `TT_FATAL` census below confirms every guard is accounted for.
- One structural change *inside the port's own writeable surface*, forced by the split: in each ported
  factory `.cpp` the ported-from `create_descriptor` body moved verbatim into a helper-namespace free
  function, because the out-of-scope `*MeshWorkloadFactory` sibling still needs it —
  `build_paged_update_cache_descriptor`, `build_paged_fill_cache_descriptor`,
  `build_paged_tiled_fused_update_cache_descriptor`,
  `build_paged_row_major_fused_update_cache_descriptor`. Same for the ported-from `Program&`-mutating
  patch where it lived in the factory file (`patch_paged_update_cache_runtime_args`).
- **Pass 2 additionally moved two *factory method definitions* between files** (factory code, not
  op-level code, so inside the lane — but worth naming since the file changed):
  `PagedTiledFusedUpdateCacheProgramFactory::override_runtime_arguments` and its row-major sibling
  were defined in `paged_fused_update_cache_device_operation.cpp` and now live beside their own spec
  builds in the two factory `.cpp` files. The reason is mechanical: the Metal 2.0 override names spec
  resources (`TF_CACHE1_TENSOR`, `TF_READER_KERNEL`, …), and those names are declared in the
  factory's own helper namespace. What stays in the device-operation file is the ported-from
  index-addressed `patch_runtime_args` template and its arg-layout constants, which the two mesh
  factories still need; those two mesh hooks previously *delegated* to the single-device hooks and now
  call `patch_runtime_args` directly, because the signatures are no longer shared.

### Open items

See [Open items for downstream](#open-items-for-downstream).

---

## Handoff points

### 1. **The four `*MeshWorkloadFactory` factories: framework vehicle MERGED (`9fb0ed54794`, PR #54988), now waiting on a *port procedure* for `MeshWorkloadSpecFactoryConcept`.** *(owner: Metal 2.0 doc maintainers, for the procedure; then any porter)*

**Status, and it changed between the two passes.** Pass 1 recorded these four as a capitulation on
missing framework work: they need a per-mesh-coordinate `ProgramSpec` / `ProgramRunArgs`, and no
Metal 2.0 TTNN factory concept provided one. **That work has merged** — `9fb0ed54794` *"Let a spec
factory build a different program per mesh coordinate (#54988)"*, on this branch. So the capability
gap is closed, and the analysis pass 1 wrote is now directly actionable.

**What is left is not a capability gap but a missing procedure.** With #54988 merged the target
concept for these four is `MeshWorkloadSpecFactoryConcept`, and the port recipe scopes itself to
`ProgramSpecFactoryConcept` and `CustomProgramSpecFactoryConcept` — "the two single-program Metal 2.0
concepts, one spec stamped across the mesh" — then names precisely this situation:

> A brief naming any other target concept is outside this procedure — stop and report. The case to
> expect is a mesh-workload concept for a genuine multi-program op: the audit can clear one the day
> TTNN support lands, but no port procedure exists for it until someone writes it. Stopping is the
> correct outcome; improvising a multi-program port out of this recipe is not.

`ttnn_factory.md` says the same from the audit side: if the target is a mesh-workload concept, "note
that the port procedure does not cover it yet, so the porter will stop at its coverage boundary."
Pass 2 therefore stopped here deliberately rather than extrapolating a single-program procedure onto
a multi-program concept — which is the reading the invoker should sanity-check, since the invoker's
own framing was that the merge "allows the Mesh factories to be ported." It does, at the framework
level; the recipe is what withholds permission, not the API.

**The ask, then, is narrow:** a `MeshWorkloadSpecFactoryConcept` section in the port recipe (or an
explicit instruction to a porter that the general rules extend to it). Everything op-specific is
already worked out below and needs no rediscovery. Line numbers for
`mesh_device_operation_adapter.hpp` below are pass-1's, i.e. pre-#54988.

**Correcting this entry's original framing.** The first version of this report filed all four
factories under `ttnn_factory.md`'s *"multi-program / per-coord variation"* RED case. That is right
for three of them and **wrong for `fill_cache`**, and the distinction is worth keeping because the two
halves need different things from the framework:

| Factory | Programs differ per coordinate? | What is actually missing |
|---|---|---|
| `PagedFillCacheMeshWorkloadFactory` | **No** — the `ProgramSpec` is identical across coordinates: same kernels, same DFBs, same bindings, same core ranges. Only one **runtime-arg value** (`noop`) differs. By `ttnn_factory.md`'s own test (*"same kernels, same DFB shape, same bindings — only the tensor data differs"*) it is **single-program**. | per-coordinate **run args on the cache miss** |
| `PagedUpdateCacheMeshWorkloadFactory` | **Yes** — the program is *absent* on excluded coordinates | per-coordinate **`ProgramSpec`** (including "none here") |
| `PagedTiledFusedUpdateCacheMeshWorkloadFactory` | Yes, same shape | same |
| `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` | Yes, same shape | same |

**What was blocked, precisely.** `create_program_artifacts(attrs, tensor_args, tensor_return_value)`
takes no coordinate, and `ProgramSpecMeshWorkloadFactoryAdapter::create_mesh_workload`
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:912-921`) emplaces *the same* `artifacts.spec` for
every range in `tensor_coords`, then applies *the same* `artifacts.run_params` to every resulting
program via `SetProgramRunArgs`. Re-derived from `device_operation.hpp:296-364`: the miss path is
`create_mesh_workload_from_workload_factory` → cache insert → `enqueue_mesh_workload`, and
`apply_descriptor` is reached **only** from `handle_mesh_adapter_cache_hit` (`:282`). So on a miss the
uniform values are what the first dispatch executes. The two legacy idioms this defeats — neither of
which the port may normalise away (*"Preserve both behaviours as they are"*):

- **Empty-descriptor idiom** — `paged_update_cache_program_factory.cpp:1132-1139` (post-port;
  `:448-453` pre-port), tiled fused `:544-549`, RM fused `:547-552` (both unchanged). A coordinate
  outside `operation_attributes.mesh_coords` gets an **empty `ProgramDescriptor`**, and the descriptor
  adapter *skips adding a program for that coordinate entirely*
  (`mesh_device_operation_adapter.hpp:588-592`).
- **`noop`-RTA idiom** — `paged_fill_cache_program_factory.cpp:62-70` and `:879-890` (post-port;
  `:33-40` / `:348-359` pre-port). The cache-*hit* path was already correct
  (`override_runtime_arguments` receives the coordinate); only the miss dispatch was wrong, and it
  would perform a real cache fill on a coordinate the caller excluded.

**How #54988 closes both — verified against the merged code, not the PR description.** It replaced
the *"A future `MeshWorkloadSpecFactoryConcept` will…"* placeholder in `metal_v2_artifacts.hpp` with
the real thing (`ttnn/api/ttnn/metal_v2_artifacts.hpp:38-52`,
`ttnn/api/ttnn/operation_concepts.hpp:118-132`):

```cpp
static MeshWorkloadArtifacts create_mesh_workload_artifacts(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&,
    const ttnn::MeshCoordinateRangeSet& tensor_coords);
```

`MeshWorkloadArtifacts::programs` is a `std::vector<PerCoordProgram>`, each entry
`{range, spec, run_params}`. That gives:

1. **Per-coordinate `ProgramSpec`** — and, decisively for the empty-descriptor idiom, the new adapter
   requires each returned range to be *contained in* `tensor_coords`, forbids duplicate ranges, and
   requires at least one program — but does **not** require the ranges to *cover* `tensor_coords`.
   Omitting a range yields no program on those coordinates, which is exactly what the descriptor
   adapter's `desc.kernels.empty() → return` does today.
2. **Per-coordinate run args on the miss** — the new adapter applies
   `SetProgramRunArgs(program, run_params.at(range))` per range. The PR states this resolution in the
   comment it adds to the Custom adapter: *"The miss path applies `create_program_artifacts`' run
   args, not the override, so per-coordinate run args must come from a
   `MeshWorkloadSpecFactoryConcept` factory's `create_mesh_workload_artifacts`."*
3. The factory receives `tensor_coords` and can intersect it with `operation_attributes.mesh_coords`
   itself, so the framework needs no knowledge of the op's filter.
4. Cache-hit refresh is an optional
   `override_runtime_arguments(attrs, args, ret, const MeshCoordinateRange&)`, called once per range,
   with a `static_assert` that rejects a near-miss signature rather than leaving run args silently
   stale.

**The four ported factories are unaffected by #54988, confirmed post-merge.**
`CustomProgramSpecMeshWorkloadFactoryAdapter` still passes
`std::optional<MeshCoordinate>(coordinate_range.start_coord())`, and
`detail::HasSpecRuntimeArgsOverride` (`operation_concepts.hpp:111-115`) still keys only on the return
type — the range-instead-of-coordinate signature is confined to the new concept's *own* optional
override. `MeshWorkloadSpecFactoryConcept` is also declared mutually exclusive with the other four,
so the "exactly one concept per alternative" rule that `AllFactoriesValid` enforces still holds and
pass 1's two factories classify exactly as before. No edit was needed to either pass when it landed,
and pass 2's build confirms it.

**Two things to get right in the follow-up port:**

- **Range granularity must respect `noop` uniformity.** The hit-path override is called once per
  *range* and its result applies to every device the range covers. `fill_cache` must decompose
  `programs` so each range is wholly inside or wholly outside `mesh_coords`. The factory controls the
  decomposition, so this is easy — but a naive "one range = the whole mesh" is wrong.
- **`TT_FATAL(!artifacts.programs.empty())` vs. legacy's zero-program case.** `paged_update_cache`
  has **no** validation that `mesh_coords` is a non-empty subset of `tensor_coords`, unlike
  `paged_fill_cache_device_operation.cpp:165` which enforces exactly that. With an empty or
  fully-disjoint `mesh_coords`, legacy dispatches nothing while the new adapter would fatal.
  Nonsensical input, and no known caller does it, but it is a behaviour delta that wants a one-line
  ruling: accept the fatal, or relax the emptiness check.

**Severity — these factories are on a production path.** Not merely test-reachable:
**DeepSeek-V3 MLA** calls both with a **strict-subset** `mesh_coords`, in model code rather than tests
— `models/demos/deepseek_v3/tt/mla/mla1d.py:2356`, `:2364`, `:2374` (`paged_update_cache`) and
`:2138`, `:2146` (`paged_fill_cache`), each passing
`set(get_mesh_coords(mesh_shape, row_idx))`, which selects **one row** of the mesh
(`models/demos/deepseek_v3/utils/config_helpers.py:1222-1231`) — 4 of 32 coordinates on an `[8,4]`.
Since the device op requires only `mesh_coords ⊆ tensor_coords`, the filter really is strict and both
legacy branches genuinely fire in production. By contrast **Llama-3.2-1B never passes `mesh_coords`**
at all (`grep -rn mesh_coords models/experimental/llama32_1b_quasar/` is empty), so its whole captured
path is already on the two ported factories.

**Why it matters that this reached the port.** `ttnn_factory.md` → *Feasibility gate* names the
*"multi-program / per-coord variation"* case and says *"the port is blocked on framework work, not
porter-resolvable. Record RED and stop."* For `update_cache` and both fused mesh factories that gate
should have fired and did not — the audit filed the two idioms under *Watch for* (behaviour to
preserve) instead of running them through the factory-concept gate. For `fill_cache` the gate **as
worded** would not have caught it, because its programs do *not* differ; only a run-arg value does.
See [Friction](#friction) #2.

**Cost paid across both passes because of it:** eleven kernel forks (next entry) — every kernel source
in the op — which porting these four also lets us retire.

### 2. **Eleven intra-op `_metal2` kernel forks created — every kernel source in the op.** *(coordination signal; owner: this op's next porter)*

Because each `*MeshWorkloadFactory` binds the *same* kernel sources as the single-device sibling that
did convert, converting those sources in place would have broken the four out-of-scope factories. Per
*Caution: Porting a shared kernel* — **rung 2 (create the fork), intra-op shape** — each fork was
created beside its original, the original was left untouched apart from the mandated pointer comment,
and the originals keep serving the mesh factories.

| fork created (all under `device/kernels/`) | forked from | pass | remaining consumers of the original |
|---|---|---|---|
| `dataflow/reader_update_cache_interleaved_start_id_metal2.cpp` | `reader_update_cache_interleaved_start_id.cpp` | 1 | `PagedUpdateCacheMeshWorkloadFactory` |
| `dataflow/writer_update_cache_interleaved_start_id_metal2.cpp` | `writer_update_cache_interleaved_start_id.cpp` | 1 | `PagedUpdateCacheMeshWorkloadFactory` |
| `compute/update_cache_metal2.cpp` | `compute/update_cache.cpp` | 1 | `PagedUpdateCacheMeshWorkloadFactory` |
| `dataflow/reader_fill_cache_interleaved_metal2.cpp` | `reader_fill_cache_interleaved.cpp` | 1 | `PagedFillCacheMeshWorkloadFactory` |
| `dataflow/writer_fill_cache_interleaved_metal2.cpp` | `writer_fill_cache_interleaved.cpp` | 1 | `PagedFillCacheMeshWorkloadFactory` |
| `dataflow/reader_paged_fused_update_cache_interleaved_start_id_metal2.cpp` | `reader_paged_fused_update_cache_interleaved_start_id.cpp` | **2** | `PagedTiledFusedUpdateCacheMeshWorkloadFactory` |
| `dataflow/writer_paged_fused_update_cache_interleaved_start_id_metal2.cpp` | `writer_paged_fused_update_cache_interleaved_start_id.cpp` | **2** | `PagedTiledFusedUpdateCacheMeshWorkloadFactory` |
| `compute/paged_fused_update_cache_metal2.cpp` | `compute/paged_fused_update_cache.cpp` | **2** | `PagedTiledFusedUpdateCacheMeshWorkloadFactory` |
| `dataflow/reader_paged_row_major_fused_update_cache_interleaved_start_id_metal2.cpp` | `reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | **2** | `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` |
| `dataflow/writer_paged_row_major_fused_update_cache_interleaved_start_id_metal2.cpp` | `writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp` | **2** | `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` |
| `compute/paged_row_major_fused_update_cache_metal2.cpp` | `compute/paged_row_major_fused_update_cache.cpp` | **2** | `PagedRowMajorFusedUpdateCacheMeshWorkloadFactory` |

No `_metal2` fork existed beside any of them beforehand (checked locationally, by `ls` of each
original's directory — not by a tree-wide grep). No build-system change was needed: the op's kernels
are installed by a `file(GLOB_RECURSE …)` that already covers both directories, and no `sources.cmake`
entry changed because the port added no new host `.cpp`.

**Sunset trigger.** Porting the four mesh factories (handoff #1) retires **all eleven** originals: the
legacy copies are deleted and the forks take their names. Until then, **a fix to either copy of a pair
should be evaluated for the other.** This is now the whole kernel surface of the op duplicated inside
one directory — eleven pairs, ~2,900 lines — which is a real drift-discipline liability and the
strongest practical argument for writing the `MeshWorkloadSpecFactoryConcept` procedure and finishing
the op promptly. It is also worth noting as a **doc-evolution signal in its own right**: the
shared-kernel Caution's fork convention was designed for cross-op sharing, where the two consumers
have different owners and drift is the price of decoupling. Applied to the intra-op mesh/single-device
split it duplicates a whole op for a reason that is purely temporal, and the mitigation the Caution
offers (record the pair, evaluate fixes on both) scales poorly at eleven.

### 3. **`ttnn.experimental.paged_fill_cache`'s `noop` attribute is dead API surface.** *(owner: ops team; carried forward from the audit, confirmed during the port)*

`PagedFillCacheParams::noop` (`paged_fill_cache_device_operation_types.hpp:16`) is never set `true` by
any caller — the sole public entry point hardcodes `.noop = false`
(`paged_fill_cache_device_operation.cpp:237`). The functional noop path is driven entirely by the
mesh-coordinate test inside `paged_fill_cache_noop`. **The port preserves it**, and it is called out
here because it interacts with handoff point #1: once a per-coordinate mechanism exists, the attribute
and the coordinate test are two sources for one flag and should be reconciled together.

---

## Successes

**1. `dataflow_buffer_spec.hpp`'s PLACEMENT note stopped a wrong answer to the audit's open question
before any code was written.** Approaching the fused factories, the natural first move was the
audit's own first option — *"bind both `src1` and `src2` and rely on per-node existence."* The
header's one-line contract (*"PLACEMENT: Derived — the DFB's effective node set is the union of its
bound kernels' `WorkUnitSpec` `target_nodes`"*) makes it immediately clear that binding both on a
`KernelSpec` spanning `all_cores_bb` **places both DFBs across the whole bounding box**, where legacy
allocated each over only its own half — and that both are `borrowed_from` an L1-sharded tensor with no
shard on the other half. That is a different question from the one the audit posed, and reading the
declaring header answered it faster than any precedent would have. It is why the fused factories are
deferred rather than guessed at. (Recipe: *"Go to the headers first; they are ground truth."*)

**2. *Pattern: Conditional / optional DFB bindings* fired exactly where the brief predicted, on a
site the brief flagged and the recipe explained.** The brief's *"Kernels declare CB wrapper objects
unconditionally for conditionally-allocated CBs"* heads-up names
`reader_update_cache:56-57`, `writer_update_cache:61-62` and `writer_fill_cache:92-93` — harmless
under the legacy `uint32_t` CB id, a compile error under a named binding. The catalog's
*"Promote a CTA gate to a define"* paragraph is precisely this port's shape: `use_index_tensor`,
`is_paged_cache`, `use_batch_idx_tensor` and `use_valid_seq_len` were all CTAs gating
`if constexpr` blocks that name conditionally-bound resources, and all four became
`compiler_options.defines` + kernel-side `#ifdef`. The catalog's warning that the define must reach
**every** kernel that references the resource caught the `update_cache` case, where both the reader
*and* the writer touch `index` / `page_table` — hence one shared `conditional_defines` table fed to
both (`paged_update_cache_program_factory.cpp:762-768`, fed to both kernel specs at `:773` and `:839`).

**3. The `hw_config` section's insistence on porting *values, not spellings* caught a real
divergence.** `update_cache` resolves a full TTNN `ComputeKernelConfig` but hand-copies only
`fp32_dest_acc_en` onto its `ComputeConfigDescriptor` — the recipe's *"Check for a dropped field
before using the helper"* case, verbatim. Routing that through `to_compute_hardware_config` would have
substituted the helper's **high-performance** defaults for `math_fidelity`, `math_approx_mode` and
`dst_full_sync_en`, none of which this op ever applied. Building `ComputeGen1Config` directly
reproduces the legacy `ComputeConfigDescriptor` defaults exactly (verified field by field:
HiFi4 = HiFi4; `math_approx_mode=false` = `Precision::Precise`; `bfp8_pack_precise=false` =
`Precision::Approximate`; `dst_full_sync_en=false` = `double_buffer_dest=true`).

**4. The `opt_level` "absent line" check earned its place in the checklist.** `grep -n opt_level` over
both ported-from factories returns **nothing** — which reads as "nothing to carry across" and is
exactly the trap the recipe describes. `update_cache`'s compute kernel resolves to `O3` on a
`ComputeConfigDescriptor` and would have silently dropped to `O2` on `KernelSpec::compiler_options`.
Set explicitly at `paged_update_cache_program_factory.cpp:938`.

---

**Pass 2 — 4. `dataflow_buffer_spec.hpp`'s alias-group node-set rule caught a legality question before
it became a validator failure.** The tiled and row-major fused factories each carry an aliased
intermediate pair, and the header states the constraint at the field: *"Aliased DFBs must have the
same total size … All members must target the same node set (derived from their bound kernels'
WorkUnitSpecs)"* (`advanced_options.hpp:169-172`). That third clause is the one worth having read: it
made me check that the two members' *binding sets* — compute+writer for `untilized_cache`,
writer+compute for `untilized_cache2` — resolve to the same nodes, which they do only because the
factory declares a **single** `WorkUnitSpec` over `all_cores_bb`. Had I split the kernels into
per-core-set work units (the option the audit's Question #1 floated), the alias group would have
become illegal, and the failure would have arrived as a `TT_FATAL` from `program_spec.cpp` at first
dispatch with no obvious connection to the work-unit decision. Applies at
`paged_tiled_fused_update_cache_program_factory.cpp` (the `TF_UNTILIZED_CACHE_DFB` /
`TF_UNTILIZED_CACHE2_DFB` pair and the `WorkUnitSpec` below it).

**Pass 2 — 5. "Go to the headers first" is what resolved audit Question #1's tensor half, and a
precedent would have got it wrong.** The natural move, having found that a runtime ternary over
`dfb::src1` / `dfb::src2` works, is to reach for the same shape on the tensor channel. It cannot
compile, and the reason is only visible in the declarations: `DFBBindingToken` keeps its identity in
a runtime `uint16_t` member (`dataflow_buffer.h:90-101`), so every DFB token shares one type, whereas
host codegen emits `TensorBindingToken<cta_offset, addr_crta_offset>` — a **distinct type per
binding** (`genfiles.cpp:258-263`) — so two bindings on one kernel can never have a common type. Two
channels that look symmetric in the recipe's prose are asymmetric in the headers, and the asymmetry
decides the kernel shape (ternary vs. twice-instantiated generic lambda). The recipe's advice that
"the reflex to hunt for a precedent is the weaker one" was right here in a strong form: any precedent
would have shown one channel or the other, and copying it across would have produced either a
compile error or an unnecessary type-erased wrapper.

**Pass 2 — 6. Re-deriving the endpoint census instead of transcribing the brief's caught the one place
the two variants differ.** The recipe insists that endpoint dispositions are "mechanical enough to
*verify*, not transcribe". Re-derived from the six fused kernels, the census agrees with the audit on
the headline (all 1P+1C, no self-loop, no `allow_instance_multi_binding`) — but it does **not** agree
kernel-for-kernel: on the row-major path the input buffers' **consumer is the writer**, not compute,
because a row-major input needs no untilize step and the "untilized input" the writer reads *is* the
input buffer. A transcription that carried the tiled roles across (compute as consumer) would have
produced a row-major spec with compute bound to two DFBs it never touches and the writer bound to
none it does — the endpoint invariant would still have been satisfied on paper, one PRODUCER and one
CONSUMER each, so the validator would have passed it and the kernel would have hung waiting on a
buffer nobody drains. Checked mechanically at the end, too: a script re-extracted every
`DFBBinding` from both factories and confirmed 9 and 8 DFBs at exactly 1P+1C
([Verification performed](#verification-performed)).

## Friction

### Gaps

**1. (Pass 1 only — RESOLVED in pass 2.) The port could not be built or tested: the toolchain was
absent from the machine.** Kept because the two secondary `workspace_setup.md` observations below are
still worth acting on, and because the resolution is itself a data point: the same box **did** have
`clang-20` when pass 2 ran, so this was a transient environment gap rather than a fixture of the
bench. Everything pass 1 lists below as unverified is now verified — see
[Verification performed](#verification-performed).

At the time of pass 1, `./build_metal.sh` failed at CMake configure with
`The CMAKE_C_COMPILER: clang-20 is not a full path and was not found in the PATH`; only clang-14 is
installed (`/usr/bin/clang-14`, `/usr/lib/llvm-14`). A sibling checkout's `CMakeCache.txt` shows
`/usr/bin/llvm-ar-20`, so the box *did* carry clang-20 at some earlier point — it is gone now.
Installing a system toolchain was out of scope, and the invoker had asked to run the build and tests
themselves.

Two secondary things this surfaced, both worth a line in `workspace_setup.md`:

- **A `git worktree` does not inherit submodules.** The first configure failed with
  `Missing submodules. Run: git submodule update --init --recursive` — obvious in hindsight, but the
  recipe's *Workspace bootstrap* assumes a fresh clone and says nothing about worktrees, which is a
  natural way to isolate a port. Fixed by running the command inside the worktree.
- **The legality-check forcing step is wasted work if the build then fails.** The recipe orders
  *force → build → prove*, so the `skip_validation` scaffolding was applied to all nine grep-named
  sites before the build was attempted, then reverted when the build proved impossible. Cheap here,
  but a one-line "confirm you can build before you force" would save the round trip.

**Pass 1's verification was therefore entirely static.** The surfaces it flagged as highest-risk and
unverified — listed below as pass 1 left them — were all exercised in pass 2 and all pass:
1. Does the whole thing compile (host and the five JIT kernel forks)?
2. Does `ValidateProgramSpec` accept the `update_cache` alias group and the three `fill_cache`
   self-loops?
3. Do the ported factories still cache-hit and produce identical numerics on the second dispatch?

**2. `ttnn_factory.md`'s feasibility gate is not represented in the audit's own gate set, so a
listed-RED condition passed GREEN.** The per-coord-variation blocker (handoff point #1) is documented
clearly in `ttnn_factory.md` → *Feasibility gate*, which is the doc the audit's *final step* invokes.
But the audit's status summary is organised around Appendix A features + the named gates (Device 2.0,
Features, TTNN factory concept, Offset base pointers, TensorAccessor 3rd arg), and "TTNN factory
concept" was answered as *which* concept rather than *whether the op fits one*. The brief's *Watch
for* section then describes both mesh idioms accurately — it just files them as behaviour to
preserve.

*Suggestion:* make the **fit** half of the factory gate explicit and mechanical, alongside the
concept choice — a checkbox the auditor must answer with evidence, e.g. *"Does any factory's
`create_descriptor` read `mesh_dispatch_coordinate` for anything other than pass-through? If yes →
per-coord variation → RED."* That single question would have caught all four factories here: three
return an empty descriptor on it, and the fourth feeds it into `paged_fill_cache_noop`. As written,
an auditor can complete the *TTNN factory analysis* section fully and correctly without ever
evaluating the gate the section's own doc defines.

Two refinements, both learned after the fact:

- **Fixing the gate's *wording* is not enough — the mechanical check is doing real work.** The gate
  says *"the op's programs genuinely differ across mesh coordinates."* By that test `fill_cache`'s
  mesh factory is **not** a hit: its `ProgramSpec` is identical everywhere and only a runtime-arg
  *value* varies. So a careful auditor reading the gate as written could legitimately clear it and
  still be wrong, because the miss path applies one `ProgramRunArgs` mesh-wide. The
  `mesh_dispatch_coordinate` question above catches both shapes because it keys on *what the factory
  reads*, not on what differs in the result. Suggest the gate say "programs **or their run args**
  differ per coordinate."
- **The gate needs a third outcome now.** With
  [PR #54988](https://github.com/tenstorrent/tt-metal/pull/54988) in review, per-coord variation stops
  meaning RED and starts meaning *"target `MeshWorkloadSpecFactoryConcept` instead"*. The gate should
  route to that concept rather than halt the audit, and the readiness sheet needs the third value.

**Pass 2 — Gap: replacing an address RTA with a `TensorBinding` can lose *selection* information, and
the Dropped Plumbing table has no row for that.** The recipe treats a buffer-address RTA as pure
plumbing: the address goes away, a `TensorBinding` replaces it, nothing else changes. That holds when
the slot carries one tensor's address. It does **not** hold when the host wrote *different tensors'*
addresses into one slot on different nodes, which is what both fused factories do — reader RTA[2] and
writer RTA[1] carry `cache_tensor1` on `cores1` and `cache_tensor2` on `cores2`. There, the address
value was doing double duty: it was the address *and* it was the answer to "which cache tensor am I".
A `TensorBinding` is per-`KernelSpec`, so the port must bind both and the kernel must be told which to
use — and if no existing arg says so, the port has to **add** one.

Concretely: the row-major writer already had an `is_input1` runtime arg (it used it to pick an input
buffer) and needed nothing. The **tiled** writer had none, so
`writer_paged_fused_update_cache_interleaved_start_id_metal2.cpp` gains a named RTA `is_input1`, and
`paged_tiled_fused_update_cache_program_factory.cpp` declares and emits it. It is not a *demotion*
(nothing moved CTA→RTA) nor a smuggled address — but it also isn't in the recipe's vocabulary of
things a port does.

Worth noting how small the actual delta is, because it makes the shape easy to accept: the tiled
writer's runtime-arg **count is unchanged at 8.** The slot that carried the cache address is the slot
that now carries the selector — the address moved to the typed channel and the one bit of information
the address was implicitly encoding stayed behind in its place. Read that way it is less "the port
added an arg" than "the port split one overloaded slot into a binding plus the bit that chose it",
which is the same shape as the *Overloaded RTA slots* case the brief already describes for
`fill_cache` (`Buffer*` **or** meaningful scalar → conditional `TensorParameter` **plus** a named
scalar). The difference is that `fill_cache`'s two channels are selected by *configuration*, so the
brief could enumerate them, while here they are selected **per node** — and that is the case neither
the brief nor the recipe has a name for. Two doc
suggestions: (a) add a Dropped Plumbing row for "one address slot, several tensors, selected by node
→ several `TensorBinding`s **plus** a selector arg if the kernel has none"; (b) note it in the
`hw_config`-adjacent list of silent hazards, because the failure mode of getting it wrong is not a
compile error — bind both and forget the selector and the kernel writes the *wrong cache tensor* on
half the cores, which is exactly the shape of bug the fused tests would catch but a single-input smoke
test would not. Pass 1's own resolution of Question #1 (Open items #1) had this gap: it prescribed
"branch on `is_input1`" for both fused factories without noticing that one of the four kernels
involved has no such arg.

**Pass 2 — Gap: what to do with a per-node *short* runtime-arg list is decided by the validator, not
by the recipe.** Both fused factories give working cores 8/8/2 runtime args and every core in
`unused_cores` a **single** `{!has_work}`, which the kernels early-return on. The brief flagged this
and said to "decide up front how the short-arg nodes are expressed (supply the full named set with
don't-care values, or narrow the `KernelSpec`'s core range) rather than discovering it at validation
time" — but neither the brief nor the recipe says which is correct, and the two options are not
equivalent: narrowing the node set also narrows every DFB's derived placement, changing which cores
get buffers relative to legacy. The answer is in the validator:
`ValidateSetProgramRunArgs` requires every declared named RTA on **every** node the kernel runs on
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:296-324`), so the full-named-set option is the
only one that preserves the legacy program. Worth one sentence in the recipe's `KernelRunArgs`
bullet, since "a `runtime_arg_schema` is one schema for the whole `KernelSpec`" is stated but its
consequence — *and therefore every node must supply all of it* — is not.

**Pass 2 — Gap: a `borrowed_from` that is conditional on tensor configuration.** The recipe's
borrowed-memory bullet reads as a static property of a DFB ("Borrowed memory → set `borrowed_from`
= …"), and the conditional-binding pattern is written about *bindings* and `#ifdef`s. Both fused
factories need a third thing: the **same** DFB borrows the index / page-table tensor's L1 memory when
that tensor is sharded and is an ordinary L1 allocation when it is DRAM-interleaved — legacy said this
with a `CBDescriptor::buffer` that is `nullptr` on one path (`:117`, `:136`). Expressing it turned out
to be trivial (build the `DataflowBufferSpec`, then set `borrowed_from` inside an `if`), and nothing
kernel-side changes because the read through it was already `if constexpr`-gated. But it took a
detour through the conditional-binding pattern first, on the assumption that a per-config difference
in a DFB must be a per-config difference in its *binding*. It isn't: this one is a per-config
difference in the spec, with the binding unconditional. A line in the borrowed-memory bullet saying
`borrowed_from` may vary by configuration like any other spec field would have saved that.

**Pass 2 — Confusion: rule 2 and the brief's "don't remove dead args" collide on a dead *CB-index*
CTA, and rule 2 has to win.** The brief lists the row-major fused compute kernel's `in1_cb` / `in2_cb`
among dead compile-time args and says "The port does not remove them — dropping one is a functional
change to the arg schema." But those two args are **CB indices**, and kernel-side whitelist rule 2 is
categorical that a CB index becomes a DFB binding and "never a named argument." The kernel never
touches either buffer (a row-major input needs no untilize step), so there is no endpoint to declare
and binding them would invent one. Both instructions cannot be followed. Resolution taken: **drop
them** — rule 2 is about a channel that no longer exists, while the brief's rule is about preserving
a *scalar* schema, and the scalar schema is preserved intact because the dead **runtime** arg
`is_input1` is still declared and still read (`[[maybe_unused]]`). Recorded as a deliberate departure
in the plan's Dropped Plumbing table and in [Open items](#open-items-for-downstream) #2. The doc fix
is small: the brief's dead-arg instruction should say *dead scalar* args, since a dead CB index has no
legal Metal 2.0 spelling to carry forward.

### Confusion

**3. "Both members of a factory pair convert together" and "the atomic unit is one ProgramFactory"
point opposite ways when the pair straddles a framework gap, and the recipe does not adjudicate.**
The brief says *"Do not port one member of a factory pair without the other — they share the body, so
they convert together."* The recipe says a `program_factory_t` variant is *valid* with alternatives on
different concepts, and that a finished factory with the rest reported is a shippable deliverable.
Here the pair genuinely cannot convert together (#1), so one of the two had to give.

The resolution turned out to be already written down, just not where it was being looked for: the
*atomic-unit* note routes the **shared top-level entry point** case to *Caution: Porting a shared
kernel*, whose **intra-op** rung says the fork lands in your own directory and your op's other
factories are the "remaining consumers." It also says co-porting the sibling is *"only the simpler
move when that factory doesn't drag in shared kernels of its own"* — which quietly assumes the
sibling *can* be co-ported at all. It cost maybe twenty minutes of re-reading to be confident the
fork was sanctioned rather than an improvisation, mostly because a brief-level "do not split this
pair" reads as more categorical than a catalog rung.

*Suggestion:* one sentence in the shared-kernel Caution's intra-op bullet — *"if the sibling factory
is blocked on framework work, the fork is the answer; the brief's 'convert together' guidance assumes
both are portable."*

**4. The `_metal2` fork naming rule and the intra-op case pull in different directions.** *Name the
bindings for the kernel, not for your op* exists so a later consumer can reuse the fork. In the
intra-op case there is no later consumer — the fork's only future is to be deleted when the sibling
converts and to give up its name to the original. The rule was followed anyway (accessor names come
from the kernel's own vocabulary: `tensor::src` from `src_addr`, `dfb::in` from `cb_in`), and it cost
nothing, but it is worth noting that the guidance's stated *reason* does not apply here. One place it
bit for real: the `update_cache` writer's own vocabulary calls the **output** DFB `cache`
(`cache_cb_id` kernel-side is `output_cb_index` host-side — the brief flags this exact trap). Keeping
the kernel's word was the minimal-diff choice and is defensible — the buffer really does hold cache
data — but the host-side binding needed an explicit comment
(`paged_update_cache_program_factory.cpp:869-872`) to keep the mapping legible.

---

## Open items for downstream

### 1. The two **fused** single-device factories — audit Question #1, **RESOLVED in pass 1 and IMPLEMENTED in pass 2**

`PagedTiledFusedUpdateCacheProgramFactory` and `PagedRowMajorFusedUpdateCacheProgramFactory` are
**ported**. This entry is kept as the design record, since it is the reasoning a reviewer needs in
order to accept the two kernels' unusual shape, and since the DFB half's Quasar-debt note is a live
carry-forward. Read it as "what pass 2 implemented", not "what remains".

**Two corrections to what follows, found while implementing it.** (1) The prescription "branch on
`is_input1`" assumed all four affected kernels have that arg; the **tiled writer does not**, and the
port had to add it — see [Friction](#friction). (2) The claim that the approach "does not change the
arg schema" is therefore true of three kernels, not four.

Question #1 has two halves, on two different binding channels. Both resolve to the same strategy —
**bind both alternatives unconditionally and select at runtime from the existing `is_input1` arg** —
but they need different kernel-side mechanics, because the two token types are built differently.

#### DFB channel — resolved: bind both unconditionally, select the token at runtime

The approach, proposed by the op owner and validated against the headers and the implementation:
**bind both `src1` and `src2` to every fused `KernelSpec` unconditionally**, and let the kernel pick
which binding token to build its `DataflowBuffer` from, using the existing `is_input1` runtime arg.

**Kernel side is a pure token substitution — no logic restructuring at all.** `DFBBindingToken` is a
trivially-copyable `{uint16_t}` with no deleted copy constructor
(`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:84-97`), so a ternary over two tokens yields a
token, which feeds the *preferred* `DataflowBuffer(DFBBindingToken)` constructor (`:110`):

```cpp
// Legacy (reader_paged_fused_update_cache_interleaved_start_id.cpp:30-35, 67):
constexpr uint32_t input1_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t input2_cb_id = get_compile_time_arg_val(1);
uint32_t input_cb_id = input1_cb_id;
if (!is_input1) { input_cb_id = input2_cb_id; }
CircularBuffer cb_input(input_cb_id);

// Metal 2.0:
const DFBBindingToken input_dfb = is_input1 ? dfb::src1 : dfb::src2;
DataflowBuffer dfb_input(input_dfb);
```

No `.id` extraction, no `static_cast<uint16_t>`, no low-level constructor — it stays entirely in the
typed channel, so the `.id`-extraction anti-pattern does not apply. Per kernel:

- **both fused readers** — as above (`reader_paged_fused…:30-35, 67`; `reader_paged_row_major…:30-35, 67`).
- **RM fused writer** — identical shape for `untilized_input_cb_id`
  (`writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:59-62, 72`).
- **tiled fused compute** — needs *nothing new*. It already branches at runtime into **two
  compile-time-parameterised instantiations** (`compute/paged_fused_update_cache.cpp:39-56`), so the
  port just substitutes the tokens as NTTPs (`untilize<Wt, dfb::src1, …>` / `<Wt, dfb::src2, …>`) and
  passes the ternary to `compute_kernel_hw_startup`. `DFBBindingToken`'s `constexpr operator uint32_t()`
  covers template-parameter position, which the brief already confirmed for these donors.
- **RM fused compute** — does not touch the input DFBs at all (its `in1_cb`/`in2_cb`/`is_input1` are
  the dead `[[maybe_unused]]` args of *Misc anomalies* #4). Nothing to do.

**Host side is legal, and free.** Four things were checked rather than assumed:

1. **Endpoint invariant holds everywhere.** Each of `src1`/`src2` gets exactly one PRODUCER (the
   reader) and exactly one CONSUMER (compute on the tiled path, the writer on the RM path) on **every**
   node of `all_cores_bb`. No `allow_instance_multi_binding`, no self-loop.
2. **Zero L1 cost — this is the finding that makes the approach viable.** A borrowed DFB
   short-circuits allocation: `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2519-2521` takes
   `alloc_addr = dfb->borrowed_addr_` and never bumps the allocator. So widening a borrowed DFB's
   derived node set adds **no** L1 pressure. Had borrowed DFBs allocated, binding both over the
   bounding box would have doubled input-shard L1 on every core and the approach would have been dead.
3. **Validation passes.** Both borrowed-DFB checks are whole-buffer, never per-node — spec-time
   L1-residency plus `entry_size * num_entries <= compute_packed_buffer_size_bytes()`
   (`program_spec.cpp:1570-1615`), and attach-time `<= buffer->aligned_size_per_bank()`
   (`program_run_args.cpp:432-492`).
4. **DFB slot budget** — the fused factories use 9 buffer indices; Gen1 has 32 per node.

**And `is_input1` stays a runtime arg.** That is the decisive advantage over the previously-sketched
"split into per-core-set `KernelSpec`s" option, which would have forced this host-computed per-core
value onto a compile-time `#define` and changed the arg schema — a structural change the port is not
entitled to make. It also leaves the *variable per-core runtime-arg count* untouched (`unused_cores`
nodes get one arg and early-return), so no `runtime_arg_schema` reconciliation is needed either.

**The one honest delta from legacy, and it is Quasar debt.** Each borrowed DFB is now *configured* on
the half of `all_cores_bb` where legacy left it unconfigured — legacy allocated `c_1` only over
`input1_cores` and `c_2` only over `input2_cores`, validated disjoint at
`paged_fused_update_cache_device_operation.cpp:350`. On that half the DFB carries the one program-wide
borrowed base address (`AttachBorrowedDFBBuffers` latches a single value per DFB, not one per node),
which there points into the *other* input tensor's region. **On Gen1 this is inert**: the `is_input1`
guard means those nodes never touch it, and there is no allocation for it to collide with. It should
nonetheless be recorded as **Quasar-uplift debt** — on Gen2 a DFB's hardware footprint varies with
endpoint configuration, and a borrowed DFB whose backing tensor holds no shard on the node is
meaningless there. Same category as `allow_instance_multi_binding`'s self-documented Gen2 debt, and
worth a comment at the binding site saying so.

#### Tensor channel — resolved: bind both, branch on `is_input1` with two typed accessors

Question #1's second half: reader RTA[2] and writer RTA[1] carry `cache_tensor1` on `cores1` and
`cache_tensor2` on `cores2` (tiled `:438` vs `:483`; RM `:435` vs `:481`).

**Neither tensor is optional** — `cache_tensor1` and `cache_tensor2` are plain `Tensor` members of
`PagedFusedUpdateCacheInputs`; only `update_idxs_tensor` and `page_table` are `std::optional`. So this
is *not* a conditional-binding problem and has nothing in common with the optional-tensor `#ifdef`
pattern used in the two ported factories.

**"Bind both" is the shape of the answer here too, and on this channel it is completely free.** A
`TensorBinding`'s base address arrives as an implicit **CRTA** — a *common* runtime arg broadcast to
every node the kernel runs on (`tt_metal/jit_build/genfiles.cpp:200-203`; the CRTA buffer is
`[ user-named CRTAs | TensorBinding section | varargs ]`). So binding both puts *both* addresses on
*every* core, already correct everywhere. Unlike the DFB half there is **no placement delta and no
Quasar debt** — a `TensorParameter` has no placement concept and nothing is allocated. The only cost
is CTA payload: each binding contributes its static layout metadata as positional compile-time args,
so each kernel carries one extra accessor's worth. Code size, not behaviour.

**What does *not* transfer from the DFB half is the ternary.** Host codegen emits a **distinct type
per binding** (`genfiles.cpp:258-263`):

```cpp
namespace tensor {
using cache1_t = ::tensor_accessor::TensorBindingToken<12u, 0u>;   // <cta_offset, addr_crta_offset>
constexpr cache1_t cache1{};
using cache2_t = ::tensor_accessor::TensorBindingToken<25u, 4u>;
constexpr cache2_t cache2{};
}
```

Those two template parameters are the binding's slot offsets — where its layout metadata sits in the
CTA payload, and where its base-address CRTA sits in the CRTA section. Two distinct bindings on one
kernel necessarily occupy distinct slots, so **the types always differ** and
`is_input1 ? tensor::cache1 : tensor::cache2` can never compile. That is the whole asymmetry with the
DFB half: `DFBBindingToken` carries its identity in a runtime `uint16_t` member (one type for all
DFBs, hence the ternary works), while `TensorBindingToken` carries it in the type.

**Both candidate routes were checked; take the branch.**

- **Route A — `KernelAdvancedOptions::TensorBindingSequence`.** Declare
  `{sequence_name = "caches", members = {"cache1", "cache2"}}`; codegen emits `tensor::caches` as a
  `constexpr std::tuple` of tokens, and the kernel does `make_tensor_accessors(tensor::caches)` →
  `make_abstract_tensor_accessor_wrappers(...)` → a runtime-indexable
  `std::array<AbstractTensorAccessorWrapper, 2>`.

  **Verified viable — the earlier open question is closed.** `noc_traits.h:140-171` fully specializes
  `noc_traits_t<AbstractTensorAccessorWrapper>` with **both** `src_addr` and `dst_addr`, structurally
  identical to the `TensorAccessor<DSpecT>` specialization, so the wrapper serves as a NoC source
  *and* destination. And it covers everything these kernels need: `s0` is touched at exactly two
  sites — `noc.async_read(s0, …, {.page_id = curr_cache_id}, {})` in the reader
  (`reader_paged_fused…:163`) and `noc.async_write(…, s0, …, {}, {.page_id = curr_cache_id})` in the
  writer (`writer_paged_fused…:154`) — with no `get_bank_base_address` and no page-size queries.

  **But it costs per page.** The wrapper is type-erased through a stored function pointer
  (`GetNocAddrFn get_noc_addr_fn`), so each `get_noc_addr` is an **indirect call** the compiler cannot
  devirtualize (the array index is a runtime value), where the concrete accessor inlines the address
  arithmetic. Both call sites are **inside the per-tile loop** (`Wt` iterations × `num_heads`), so a
  Llama-shaped decode (`Wt = 2`, `num_heads = 8`) pays ~16 indirect calls per dispatch that legacy did
  not. Almost certainly noise beside the NoC transfers themselves — but it is a nonzero deviation from
  legacy codegen, and the port's contract is that performance is unchanged.

  Footgun if used: `make_abstract_tensor_accessor_wrappers` stores pointers *into* the tuple, so the
  tuple must outlive the array (`tensor_accessor.h:700`).

- **Route B — branch on `is_input1` with two typed accessors. Recommended.** Because the tokens are
  distinct *types*, this is not a workaround but ordinary C++ for heterogeneous types — a generic
  lambda or template binds either:

  ```cpp
  auto body = [&](const auto& s0) { /* the shared per-tile loop */ };
  if (is_input1) { body(TensorAccessor(tensor::cache1)); }
  else           { body(TensorAccessor(tensor::cache2)); }
  ```

  The selection is made **once** per invocation; inside the loop the accessor is concrete and fully
  inlined, giving codegen identical to legacy. The cost is two instantiations of the loop body — the
  same shape the tiled compute kernel already uses for its two `untilize<>` instantiations
  (`compute/paged_fused_update_cache.cpp:39-56`).

**The trade, stated plainly:** route A avoids a branch and adds a per-page indirect call; route B adds
a second instantiation and costs nothing at runtime. For a binary choice resolved once per
invocation, **route B wins**. Route A earns its keep when the binding count is genuinely variadic,
which is what `TensorBindingSequence` was built for.

### 2. Dead compile-time args — carried through, with one deliberate exception


The audit catalogued these as team-only anomalies; the port carried each across as a named arg rather
than dropping it, because removing one changes the arg schema. Now that they are *named*, they are
also easy to find and drop in a follow-up:

- `log_base_2_of_page_size` — `reader_update_cache_interleaved_start_id_metal2.cpp:23`; host value is
  a local initialised to `0` and never assigned (`paged_update_cache_program_factory.cpp:582`).
- `log2_page_table_stick_size` — read and unused in
  `reader_update_cache_interleaved_start_id_metal2.cpp:31` and
  `writer_fill_cache_interleaved_metal2.cpp:46`.
- `max_blocks_per_seq` — read and unused in both `update_cache` `_metal2` dataflow kernels, and in
  **all four** fused `_metal2` dataflow kernels. It *is* load-bearing in
  `validate_on_program_cache_miss` as a bound, but no kernel range-checks `virtual_block_id` against
  it before indexing `page_table_ptr[virtual_block_id]` — a missing on-device bound check worth the
  ops team's attention independently.
- `log_base_2_of_page_size` and `log2_page_table_stick_size` — likewise read and unused in both fused
  readers (host value `0` in both cases).

**The one exception: the row-major fused compute kernel's `in1_cb` / `in2_cb` were dropped, not
carried.** They are dead *CB indices*, not dead scalars, and kernel-side whitelist rule 2 forbids a CB
index becoming a named argument; the kernel touches neither buffer, so there is no DFB endpoint to
declare either. The scalar arg schema is unaffected — the dead runtime arg `is_input1` alongside them
is still declared and still read (`[[maybe_unused]]`) — so the host's per-node emission is unchanged.
Reasoning and the doc-fix suggestion are in [Friction](#friction); recorded in the plan's Dropped
Plumbing table for the row-major variant.

### 3. Naming and structure the port deliberately left alone

- **`update_cache`'s writer calls the output DFB `cache`.** Kernel-side `cache_cb_id` was host-side
  `output_cb_index` (`c_16`); the fork keeps the kernel's word and binds the `OUTPUT` spec under
  accessor `cache`. Semantically defensible (it holds re-tilized cache data), but it is the kind of
  thing a reader trips over — worth renaming when the mesh sibling converts and the fork takes over
  the original's name.
- **Two unbalanced FIFO `wait_front`s in the row-major fused writer** (`:84`, `:95`, no matching
  `pop_front`, unlike both sibling writers) — untouched, and unreached by this pass since the RM
  fused factory is deferred. Flagged again here because it will land in whoever ports it.

### 4. Test-coverage note — **the row-major fused factory has no test that reaches it**

**`PagedRowMajorFusedUpdateCacheProgramFactory` is ported but unreachable from the repo's test
suite.** `select_program_factory` picks it only when *both* fused inputs are `Layout::ROW_MAJOR`
(`paged_fused_update_cache_device_operation.cpp:150-155`), and the only pytest file for the op,
`tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py`, reaches that path
through a `row_major` parameter on its helper
(`run_test_paged_fused_update_cache_decode`, `:25`) that **defaults to `False` and is not overridden
by any of the file's three tests**. `grep -rn 'row_major' ` over the file returns exactly two hits:
the default at `:25` and the branch that consumes it at `:96`. So the row-major branch of the helper
is dead code as checked in, and the factory it selects has never been executed by CI.

That is a pre-existing gap, not one the port created — the legacy row-major factory was equally
untested — but it lands squarely on this port, because it means **half of pass 2's diff has no
regression net.** What pass 2 did about it:

- Drove the path directly, outside the repo suite, with a scratchpad script that imports the existing
  helper and calls it with `row_major=True` (plus the shapes the op's own validation requires on that
  path: `head_dim = 128` and eight padded heads, per
  `paged_fused_update_cache_device_operation.cpp:342-346`). Results in
  [Verification performed](#verification-performed). This is a smoke test, not coverage.
- **Recommended follow-up, and it is a one-line change:** add `row_major` to the
  `@pytest.mark.parametrize` set of `test_paged_fused_update_cache_decode`, constrained to the
  `num_heads`/`head_dim` combination the op accepts. The helper already handles the padding
  (`:96-103`) and the comparison, so nothing else is needed. The port did not make this change
  itself: adding test coverage for a previously-untested factory is a behaviour-revealing change of
  its own, and bundling it into a Metal 2.0 port is exactly what [§Scope discipline] forbids — if it
  fails, nobody can tell whether the port or the pre-existing factory is at fault.
- **Sweep coverage does not fill the gap either.**
  `tests/sweep_framework/sweeps/model_traced/paged_fused_update_cache_model_traced.py` replays traced
  model parameters, and every production caller found (`models/common/modules/attention/attention_1d.py:1125`,
  `models/demos/llama3_70b_galaxy/tt/llama_attention.py:898`) passes tiled inputs.

### 5. Test-coverage note — excluded mesh coordinates on the cache miss

Nothing in the confirmed test set exercises the **cache-miss dispatch on a mesh coordinate excluded
by `mesh_coords`** on the `fill_cache` path — which is precisely the behaviour handoff point #1's
`noop`-RTA idiom would silently change if someone ported that factory without the #54988 mechanism.
`test_paged_fill_cache_mesh_coords` and `test_paged_fill_cache_batched_mesh_coords`
(`tests/ttnn/nightly/.../test_paged_update_cache.py:946`, `:1300`) assert the *result*, so they would
catch it — but both take the `device` fixture, so on an N150 they run a 1x1 mesh where `mesh_coords`
can only be the full coordinate set and the excluded-coordinate branch never fires.

**This is the acceptance gate for the #54988 follow-up port, and it must not be run on a
single-device bench.** Two properties need a genuine multi-device mesh (T3K or Galaxy) to observe at
all, both on the **first** dispatch rather than a cached one:

1. `fill_cache` — an excluded coordinate's cache must be **unmodified** after the very first call
   (today it would be filled, then noop'd on every later call, so a test that dispatches twice
   before checking would pass a broken port).
2. `update_cache` — an excluded coordinate must have **no program dispatched** to it, not a program
   that early-returns; the observable is the cache contents, but the distinction matters for the
   `TT_FATAL(!artifacts.programs.empty())` edge case noted in handoff #1.

DeepSeek-V3 MLA is the natural end-to-end validator, since it is the production caller that passes a
strict-subset `mesh_coords` (see handoff #1).

---

## Verification performed

Denominators printed per the recipe's note, so a check that scanned nothing is distinguishable from a
check that found nothing.

### Pass 2 — static checks

`BASE = git merge-base origin/main HEAD = c6640d4f75f`. All sweeps below were re-run over the whole
port (both passes), not just pass 2's files.

| check | result |
|---|---|
| **Kernel ↔ host named-argument reconciliation**, all six pass-2 kernels: every `get_arg(args::…)` the kernel reads against every name the host declares in that `KernelSpec`'s `compile_time_args` + `runtime_arg_schema` | **6/6 exact match, both directions** — tiled reader 19/19, tiled writer 19/19, tiled compute 4/4, RM reader 19/19, RM writer 19/19, RM compute 4/4. No name read-but-not-emitted (a JIT `static_assert`) and none emitted-but-unread (dead schema) |
| **Kernel ↔ host binding-token reconciliation**, all six: every `dfb::` / `tensor::` / `sem::` name the kernel uses against every `accessor_name` the host binds on that kernel | **6/6 exact match, both directions** — dfb 5/6/7/5/7/4, tensor 4/2/0/4/2/0, sem 1/1/0/1/1/0 |
| **DFB endpoint census, re-extracted from the two factories' `DFBBinding`s** (not transcribed from the brief) | **17/17 DFBs at exactly 1P+1C** — tiled 9, row-major 8. No self-loop, no `allow_instance_multi_binding`, no DFB bound-but-undeclared or declared-but-unbound. Roles match the plan, *including* the row-major difference the brief did not spell out (the input DFBs' consumer is the writer, not compute) |
| **Alias-group legality**, tiled `c_24`/`c_25` and row-major `c_5`/`c_6` | all three header rules satisfied: mutual `alias_with`, equal `entry_size * num_entries`, and one `WorkUnitSpec` so both members derive the same node set (`advanced_options.hpp:167-172`) |
| `cb`-name sweep — `grep -rnE '[Cc][Bb]_\|_[Cc][Bb]\|[Cc][Bb]\|CB[A-Z]'` over the six new `_metal2` kernels | **0 hits / 6 files** (after rewording two comments that used the "CB-index" phrasing to explain what the constants became) |
| Same sweep plus `CircularBuffer` / `CBDescriptor` / `TensorAccessorArgs` / `buffer()->address` / `emplace_runtime_args` / `allow_instance_multi_binding` over both pass-2 factories' **Metal 2.0 regions** (tiled lines 594-1359, RM 592-1346) | **0 hits / 765 + 754 lines** (after rewording one comment per file that named `CBDescriptor::buffer`). Scoped to the Metal 2.0 regions deliberately: each file also retains its ported-from descriptor body, which is legacy CB code by design and must stay |
| Legacy-API constructs in the six new kernels — `CircularBuffer` / `get_compile_time_arg_val` / `get_arg_val` / `get_common_arg_val` / `get_vararg` / `TensorAccessorArgs` | **0 hits / 6 files** — every argument named, every resource from a binding token |
| `.id` extraction on a `dfb::` handle, temp `DataflowBuffer` wrappers at LLK call sites | **none.** The tiled compute kernel passes `dfb::src1` / `dfb::src2` straight into `compute_kernel_lib::untilize<>` as NTTPs and a runtime ternary over the two tokens into `compute_kernel_hw_startup` — the `constexpr operator uint32_t()` covers both positions |
| Varargs — `get_vararg` / `get_common_vararg` / `get_compile_time_vararg` / `compile_time_varargs` / `num_runtime_varargs` | **none** anywhere in the op |
| `opt_level` audit — `grep -n opt_level` over both pass-2 factories, paired against the compute `KernelSpec`s enumerated from the construction code | **2 lines / 2 compute specs.** Each factory builds exactly one compute `KernelSpec` and each carries an explicit `KernelBuildOptLevel::O3` (tiled `:1056`, RM `:1059`). The four DM specs correctly take Metal 2.0's `O2` default, which is also the legacy DM default |
| `hw_config` value diff vs. the ported-from configs | DM: `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` on all four dataflow kernels → the reader and writer default triples, reproduced by `ttnn::create_reader_datamovement_config` / `create_writer_datamovement_config`. Compute: legacy `ComputeConfigDescriptor{.fp32_dest_acc_en = …}` sets exactly one field, so only `enable_32_bit_dest` carries across and the other four `ComputeGen1Config` defaults are left alone (they coincide with the legacy descriptor's). `bfp_pack_precision_mode` untouched (legacy default). `unpack_modes` derived from the resolved data formats of the DFBs each compute kernel **consumes** — four candidates on the tiled path, two on the row-major one, since its compute kernel consumes neither input DFB |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census vs. `BASE`, per file | **no file's count dropped.** Two rose: `paged_fill_cache_program_factory.cpp` 3 → 6 (pass 1) and `paged_row_major_fused_update_cache_program_factory.cpp` 2 → **4** (pass 2 — the two shard-spec guards now appear in the retained descriptor body *and* the Metal 2.0 body, which performs the same `.value()` dereferences). The tiled variant is unchanged at 0, correctly: its ported-from body has no such guards, and the port does not add any |
| Ephemeral-doc citation — `.md` references in every changed + untracked `.cpp` / `.hpp` / `.h` | **0 hits / 33 files** (file list printed before trusting the result, per the recipe's denominator note) |
| Diff scope — `git diff --name-only $BASE \| grep '^tt_metal/'` | see the note below on the forced-legality scaffolding; **no other `tt_metal/` file is touched**, and every other changed path is inside the op directory |
| Forced-legality scaffolding in the diff | forced at all **9** sites `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` named, with one marker per file (not in `UpdateProgramRunArgs`, which fires on every cache hit), **proven live in the test log**, then reverted with `git checkout --`. Confirmed twice on the final tree: `git diff --stat $BASE -- tt_metal/` is **empty** (the two files are byte-identical to `BASE`), and the marker/`DO NOT COMMIT` grep over all 31 changed-or-untracked `.cpp`/`.hpp`/`.h` files returns **0 hits**. The strings do survive in `METAL2_PORT_REPORT.md`, which documents the procedure — the recipe's one-line `git diff \| grep` form flags that, so it is worth scoping the check to code files, as done here |
| Positional `compile_time_args` remaining in either ported factory | **none** — every entry is `{{name, value}}` (the only positional vectors left are `reader_compile_time_args` / `writer_compile_time_args` / `compute_kernel_args` inside the retained descriptor bodies, which are legacy by design) |
| Changed or added files outside the op directory | **none** |

### Pass 1 — static checks (unchanged, re-run)

| check | result |
|---|---|
| `cb`-name sweep over the 5 `_metal2` kernels | **0 hits / 5 files** |
| `cb`-name + `CBDescriptor` + `CircularBuffer` + `TensorAccessorArgs` + `buffer()->address` + `emplace_runtime_args` sweep over both Metal 2.0 factory regions (`paged_update_cache_program_factory.cpp:556-1128`, `paged_fill_cache_program_factory.cpp:435-878`) | **0 hits / 2 regions** after rewording two comments that named the legacy API |
| Ephemeral-doc citation — `.md` references in changed + untracked `.cpp/.hpp` | **0 hits / 14 files** |
| `.id` extraction on a `dfb::` handle | **none** (the one `.id` hit is `SemaphoreDescriptor::id` in the retained legacy body) |
| `allow_instance_multi_binding` | **not set anywhere** — the census is 1P+1C or self-loop throughout |
| Varargs (`get_vararg` / `compile_time_varargs` / `num_runtime_varargs`) | **none** — every argument is named |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census vs. `$BASE` | **no file's count dropped.** `paged_fill_cache_program_factory.cpp` 3 → **6** (the same three guards now also in the Metal 2.0 body); the three device-op `.cpp` files unchanged at 45 / 25 / 40 |
| `opt_level` audit | `grep -n opt_level` over both ported factories returns **one** line — `KernelBuildOptLevel::O3` at `paged_update_cache_program_factory.cpp:938`, paired with the **one** compute `KernelSpec` the factory builds. `fill_cache` builds **zero** compute specs; both its DM specs correctly take Metal 2.0's `O2` default, matching the legacy DM default |
| `hw_config` value diff | reader `RISCV_1 / NOC_0 / DM_DEDICATED_NOC`, writer `RISCV_0 / NOC_1 / DM_DEDICATED_NOC` — traced from `ReaderConfigDescriptor{}` → `ReaderDataMovementConfig` (`kernel_types.cpp:19-22`, with `preferred_noc_for_dram_read` = `NOC_0` on every Gen1 arch) and confirmed identical to `CreateReaderGen1DataMovementConfig()` / `CreateWriterGen1DataMovementConfig()`, which the TTNN helpers wrap. Compute: only `enable_32_bit_dest` carried across; all four remaining `ComputeGen1Config` defaults verified equal to the legacy `ComputeConfigDescriptor` defaults |
| DFB endpoint census re-derived from the kernels (not transcribed from the brief) | agrees with the audit: `update_cache` 8 × 1P+1C; `fill_cache` 1 × 1P+1C + 3 self-loop. Self-loop shape (one accessor name, PRODUCER + CONSUMER on one kernel) confirmed legal against `program_spec.cpp:296-370` |
| Alias-group legality (`update_cache` `c_24`/`c_25`) | mutual `alias_with`, equal `entry_size * num_entries`, same derived node set, neither borrows — all four rules at `program_spec.cpp:1619-1699` satisfied |
| Borrowed-DFB legality (`update_cache` `input`) | named `TensorParameter` exists, its `TensorSpec` is L1-resident (the op requires a sharded input), and `entry_size * num_entries` ≤ the tensor's packed size — the three checks at `program_spec.cpp:1570-1615` |

### Build and test — pass 2 (dynamic)

Machine had `clang-20` this time, so everything pass 1 could only reason about statically was actually
run. Watcher was on for every run (`TT_METAL_WATCHER=10`, exported once so no run paid a Watcher flip
mid-sequence), and the Metal 2.0 legality checks were forced at all 9 grep-named sites.

**Legality checks proven live before any green was trusted.** `grep -o 'METAL2_CHECKS_FORCED.*' | sort | uniq -c`:

```
26 METAL2_CHECKS_FORCED (program_run_args.cpp:565)
26 METAL2_CHECKS_FORCED (program_spec.cpp:2950)
```

Both translation units fresh, both firing — `program_spec.cpp` is `BuildProgramFromSpec`, the
spec-side choke point that `MakeProgramFromSpec` / `MakeMeshWorkloadFromSpec(s)` all funnel through,
and `program_run_args.cpp` is `SetProgramRunArgs`, the cache-miss apply. Exactly 2 per program
construction across 26 tests, so no spec in this port was validated with the checks bypassed.

**Build.** `./build_metal.sh --build-tests`, twice.

| build | result |
|---|---|
| Cold, pre-pass-2 tree (pass 1's code + the forced checks) | **SUCCESS.** Also the first time pass 1's factories were ever compiled |
| Incremental, with pass 2 | **SUCCESS**, `ttnn_op_experimental_paged_cache` recompiled, **zero diagnostics for the target**. Worth noting the bar: this build runs `-Werror` with `-Wextra -Wall -Wunused -Wunused-parameter -Wshadow -Wconversion -Wmissing-field-initializers` among others, so a stale unused local left behind by the descriptor→spec rewrite, or a designated initializer with fields out of declaration order, would have failed rather than warned |

**Tests — no-regression, measured against a real baseline.** The cold build compiled `paged_cache`
*before* pass 2's host edits landed, which handed the port something pass 1 never had: a binary with
the legacy fused factories in it. So the fused result below is a genuine before/after on the same
machine, same Watcher setting, same device.

| run | binary | result |
|---|---|---|
| `test_paged_fused_update_cache.py` | **pre-pass-2** (legacy fused factories) | **82 passed** in 98s |
| `test_paged_fused_update_cache.py` | **pass 2** (ported tiled fused factory) | **82 passed** in 93s — identical count, no skips, no xfails |
| row-major fused smoke (see below) | pass 2 | **26 passed** in 38s |
| `test_paged_cache_flexible_geometry.py` | pass 2 | **24 passed** in 22s |
| `test_paged_update_cache.py` (nightly) | pass 2 | **136 passed, 47 skipped** in 661s |
| `test_paged_cache_mask.py` | pass 2 | **1 skipped** — needs a multi-device mesh; this bench has one device (`ls /dev/tenstorrent` → a single entry), exactly as pass 1 predicted |

**Totals: 268 passed, 48 skipped, 0 failed, 0 errors.** Every one of the 47 nightly skips is a
pre-existing in-test `pytest.mark.skip` with an explicit reason (*"Test case covered by others"*,
*"just need to sanity-check a select test case for bfp4"*) rather than an environmental or
port-induced skip. No `0xdeadc0de`, no Watcher assertion, no NoC-idle complaint, and no hang in any
run — grepped for across all five logs.

**The two pass-1 factories ran for the first time here**, and pass, which retires the *"⚠ Not built,
not tested"* caveat that headed pass 1's report. Their coverage is the `flexgeo` and nightly files
above (160 passing cases including the index-tensor path, the batched fill path, and the
program-cache/cache-hit tests), so pass 1's borrowed-memory `input` DFB, its `c_24`/`c_25` alias
group, and `fill_cache`'s three writer self-loops are all now confirmed against a live validator
rather than by reading `program_spec.cpp`.

The tiled fused factory is therefore a measured no-regression, and the three cache-hit tests in that
file (`…_decode_program_caching`, `…_decode_attr_idxs_program_caching`) are what exercise the
translated `override_runtime_arguments` — they dispatch the same program at several positions, which
is precisely the second-dispatch-only failure mode a first-call pass would hide.

**The row-major fused factory had to be driven outside the repo suite**, because no test in it selects
that path ([Open items](#open-items-for-downstream) #4). Driven with a porter-side script that imports
the repo's own helper (so the correctness comparison is the repo's) and calls it with
`row_major=True`, across `paged_update` × `cache_idx` ∈ {0, 127, 1057} × `num_heads` ∈ {1, 8} ×
`cache_dtype` ∈ {bfloat16, bfloat8_b}, plus a repeated-dispatch case for the cache-hit path: **26
passed.** The script is **not** part of the diff — it lived in the porter's scratchpad and was
removed from the tree before the commit. Its content is reproduced in
[Open items](#open-items-for-downstream) #4's recommendation, which is to parametrize `row_major` in
the repo test instead.

### Confirmed test set

Located with a broad sweep
(`find tests -iname '*paged*' -o -iname '*update_cache*' -o -iname '*fill_cache*'`) and filtered.
There is **no C++ gtest coverage** for this op — pytest only.

| file | fixture | covers |
|---|---|---|
| `tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py` | `device` | the two **pass-1** factories (block-size / num-kv-heads overrides, negatives) |
| `tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py` | `device` | the two **pass-1** factories incl. the index-tensor path, the batched fill path, and the program-cache/cache-hit tests |
| `tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py` | `device` | the **tiled** pass-2 factory. Does **not** reach the row-major one — see [Open items](#open-items-for-downstream) #4 |
| `tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py` | `mesh_device` | the **out-of-scope** mesh path — no-regression only; meaningful coverage needs a multi-device mesh |
| `tests/sweep_framework/sweeps/model_traced/paged_{update,fill}_cache_model_traced.py` | sweep harness | optional breadth; every traced caller found passes tiled inputs |

**Excluded as a false positive:** `tests/tt_eager/python_api_testing/unit_testing/misc/test_update_cache.py`
drives `ttnn.update_cache` / `ttnn.fill_cache` — the separate `kv_cache` op, not this one.

**Before trusting any green, force the Metal 2.0 legality checks** (recipe: *Ensure the Metal 2.0
host-side legality checks are enabled*). TTNN sets `skip_validation` behind the factory concepts and
has got it wrong before; a pass with the checks bypassed is the most expensive false green available
here, because it means every spec mistake in this port is still sitting in it:

```bash
grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp
# make `skip_validation = false;` the first statement of EVERY function that grep names,
# and add one `log_warning(tt::LogMetal, "METAL2_CHECKS_FORCED");` per file (not in
# UpdateProgramRunArgs -- it fires on every cache hit and buries the log).
./build_metal.sh --build-tests
```

Run one test, then grep its log for `METAL2_CHECKS_FORCED`; **two** markers means both translation
units are fresh and the checks are live. **Revert the forcing and the markers before the PR** —
`tt_metal/impl/` is far outside a port's scope.

**Then, in order:**

```bash
export PYTHONPATH=$(pwd)
source python_env/bin/activate          # ./create_venv.sh first if there is no python_env

# 1. The two ported factories -- primary no-regression baseline.
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py -x -v
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py -x -v

# 2. The cache-hit path in isolation -- this is where the translated
#    override_runtime_arguments lives, and the failures it can produce are
#    second-dispatch-only, so they hide behind a first-call pass.
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py -v \
  -k "program_cache or program_caching or attr_idxs"

# 3. Untouched-by-this-pass factories -- confirm the split broke nothing.
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py -x -v
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py -x -v
```

**What to expect if something is wrong**, in the order the failures would surface:

1. **Compile.** Host: a `Table` built like a vector, or a designated initializer out of field order.
   Kernel (JIT, at first dispatch): a `dfb::` / `tensor::` / `args::` name the host did not emit, or
   an `#ifdef` gate that does not match its binding condition.
2. **`TT_FATAL` from `program_spec.cpp` at the first dispatch** — a spec-validation reject. The two
   constructs most worth suspecting are `update_cache`'s alias group (`untilized_cache` /
   `untilized_cache2`) and its borrowed `input` DFB, and `fill_cache`'s three writer self-loops.
3. **Wrong numerics on the second and later calls only** — a missing `TensorArgument` in an override.
   The completeness table under [Concept realized](#concept-realized) is the checklist.
4. **A `TensorSpec` legality failure on the second call, never the first** — that is the custom-hash
   failure mode, and it is a stop-and-report, not a fix: do not touch the hash. See
   `ttnn_factory.md` → *The cache key: leave the custom hash alone*.
