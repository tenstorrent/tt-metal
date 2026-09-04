# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/slice`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

> **Read this first — the port is incremental, and part of it is already done.** On
> `akertesz/slice-test` (PR #55433, draft) **two of the five factories are already ported** to
> `CustomProgramSpecFactoryConcept`: `SliceTileProgramFactory` (`8c8b9eea947`) and
> `SliceTileTensorArgsProgramFactory` (`aafc364bc0c`). The framework **dispatches per-factory**, so the
> op builds and runs with its factories on mixed concepts — you do not have to convert all five in one
> change, and this brief should not be read as requiring that. Reported verification for the first:
> Wormhole n150, legality checks forced on, `test_slice.py` 448 passed / 38 skipped, matching baseline.
> That branch also carries its own audit/brief/plan/report set. It cites recipe `1167faf7b42` against
> this brief's `4bd4bf42bfe` — but the later date is a divergent branch, and on content **this brief
> ran on the newer audit recipe**. The entire doc-tree delta between the two is one hunk in
> `metal2_audit.md`'s offset-base-pointer section; **`metal2_port.md` is byte-identical**, so the port
> recipe you follow is the same either way. Reconcile the documents before starting — see
> `METAL2_PREPORT_AUDIT.md` → *Post-audit reconciliation*, which also explains the one substantive
> finding difference (that branch already dropped the `TensorAccessor` 3rd args).

**Scope:** one DeviceOperation, **five** program factories, **ten** slice-owned kernels + **one**
cross-family donor kernel, **seven** CBs, **twelve** tensor bindings. Two kernel files in the op
directory are **unreferenced** and out of scope — do not touch
`device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` or
`device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`.

Which factory serves which config (`device/slice_device_operation.cpp:309-341`) — you need this to read
the per-config dispositions below:

| Config | Factory |
|---|---|
| `use_tensor_args == true` (TILE only) | `SliceTileTensorArgs` |
| RM, HEIGHT-sharded in **and** out, no step, W-begin L1-aligned | `SliceRmSharded` |
| RM, any `step != 1` | `SliceRmStride` (rank ≤ 4 and rank > 4 bind **different** kernels) |
| RM, otherwise (interleaved **or** BLOCK/WIDTH-sharded) | `SliceRm` |
| TILE, otherwise | `SliceTile` |

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` — all five factories declare
  `static tt::tt_metal::ProgramDescriptor create_descriptor(const SliceParams&, const SliceInputs&, Tensor&)`
  (`device/slice_program_factory_rm.hpp:26` and the four peers).
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** (no op-owned tensors) — selected by
  `Override runtime args method? == yes`, and confirmed by the readiness sheet's own `Porting Target`
  cell. All five factories define `override_runtime_arguments`:
  `…_rm.cpp:396`, `…_rm_sharded.cpp:415`, `…_rm_stride.cpp:178`, `…_tile.cpp:189`,
  `…_tile_tensor_args.cpp:195`. **Translate** these into `ProgramRunArgs`-returning methods; do not
  delete them.
- **`override_runtime_arguments` is unusual in shape — read this before you start on it.** All five
  one-line methods delegate to a single shared implementation,
  `patch_slice_program_addresses` (`device/slice_program_factory_rm_sharded.cpp:354-413`), which
  branches on the factory type and reaches **three different** refresh mechanisms:
  - `SliceRmSharded` → `apply_descriptor_runtime_args` over a CB-address-only descriptor, matching the
    two borrowed CBs **positionally** (src0, then c_16) — `:362-368`. Keep that order in the spec.
  - `SliceRm` / `SliceRmStride` → `patch_slot0`, a positional `GetRuntimeArgs` rewrite of arg slot 0,
    skipping cores whose slot holds 0 — `:372-380, 389`.
  - `SliceTile` / `SliceTileTensorArgs` → `apply_dynamic_runtime_args` over a `DynamicRuntimeArg`
    vector — `:394-409`, with the per-core half built by `slice_tile_dynamic_args`
    (`device/slice_program_factory_tile.cpp:198-281`), which **re-derives the work split** and must keep
    matching `create_descriptor`'s.
  All of that collapses once the tensors are typed bindings that the framework refreshes — most of
  what the function does is re-point addresses. The residue that genuinely belongs in the translated
  method is the tile factories' per-core scalar re-emission (`start_id`, `num_tiles`, `id_per_dim`,
  and the writer's `num_tiles` / `start_id`), which exists to fix #52651 (a divergent-partition hit
  leaving `num_pages = 0`). Note `tt::tt_metal::apply_dynamic_runtime_args` / `DynamicRuntimeArg` here
  is a **helper API**, not the deprecated device-op `get_dynamic_runtime_args` hook — the op has no
  such hook.
- **Custom hash:** present at `device/slice_device_operation.cpp:343`. **Leave it exactly as it is** —
  no rewrite, no trimming. It is deliberately over-keyed (see its comment at `:344-348` and issues
  `#53997` / `#47602`), and several factory comments depend on that keying being in place.
- **Pybound `create_descriptor` — delete it:** `slice_nanobind.cpp:168-179`, on `SliceTileProgramFactory`.
  This is a user-visible API change; give it its own entry in the port report. Leave the device-op's
  `create_output_tensors` / `compute_output_specs` bindings (`:156-166`) and the `SliceParams` /
  `SliceInputs` struct bindings (`:138-154`) alone — those are not descriptor internals.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): a `TensorParameter relaxation`
  that is neither `none` nor an analysis pointer (the cell reads **`none`** on all five rows) ·
  `get_dynamic_runtime_args` (the deprecated hook). A custom hash, an `override_runtime_arguments`, and
  a pybound `create_descriptor` are **not** in this list: none of them gate, and all three are present
  here.

## Construct — to do

**Tensor bindings** (12 — ten Case 1, two clean; **no Case 2 anywhere**, so no
`get_bank_base_address` bridge is needed):

- `SliceRm` · **`input`** — **Case 1**. Delivered today as a `Buffer*` binding at reader RTA 0
  (`…_rm.cpp:377`); kernel builds `TensorAccessor(src_args, src_addr, padded_stick_size)`
  (`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:43`). → express as
  `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::<name>)`. The RTA-0 address
  and the `TensorAccessorArgs` CTA plumbing (`…_rm.cpp:336`) both disappear. **Drop the 3rd arg** — see
  below.
- `SliceRm` · **`output`** — **Case 1**. `Buffer*` @ writer RTA 0 (`…_rm.cpp:385`) →
  `TensorAccessor(dst_args, dst_addr, page_size_override)`
  (`slice_writer_unary_stick_layout_interleaved_start_id.cpp:32`). Same treatment; **drop the 3rd arg**.
- `SliceRmSharded` · **`input`** — **clean** (borrowed-memory DFB read). There is **no address arg and
  no `TensorAccessor`** in this factory. CB `c_0` is borrowed from `input.buffer()`
  (`…_rm_sharded.cpp:282,290`) and the kernel reads through it. → port via
  `DataflowBufferSpec::borrowed_from`. No binding work item.
- `SliceRmSharded` · **`output`** — **clean**, same shape: CB `c_16` borrowed from `output.buffer()`
  (`…_rm_sharded.cpp:294,302`) → `borrowed_from`.
- `SliceRmStride` · **`input`** — **Case 1**. `Buffer*` @ reader RTA 0 (`…_rm_stride.cpp:128` for the
  4D path, `:147` for ND) → `TensorAccessor(src_args, src_addr)`
  (`reader_multicore_slice_4d.cpp:89`, `reader_multicore_slice_nd.cpp:94`). **No 3rd arg** — nothing to
  drop.
- `SliceRmStride` · **`output`** — **Case 1**. `Buffer*` @ writer RTA 0 (`:136` / `:160`) →
  `TensorAccessor(dst_args, dst_addr)` (`writer_multicore_slice_4d.cpp:72`, and the ND peer).
- `SliceTile` · **`input`** — **Case 1**, and note it rides a **CRTA**, not an RTA: `Buffer*` pushed
  into `emplace_common_runtime_args` at `…_tile.cpp:143` → `get_common_arg_val<uint32_t>(0)`
  (`reader_unary_unpad_dims_interleaved_start_id.cpp:15`) → `TensorAccessor(src_args, src_addr)` (`:26`).
- `SliceTile` · **`output`** — **Case 1**. `Buffer*` @ writer RTA 0 (`…_tile.cpp:180`) →
  `TensorAccessor(dst_args, dst_addr)` (slice's own
  `writer_unary_interleaved_start_id.cpp:36`).
- `SliceTileTensorArgs` · **`input`** / **`start_tensor`** / **`end_tensor`** — **three Case 1
  bindings**, all on CRTAs 0/1/2 (`…_tile_tensor_args.cpp:182,183,184`), each with its **own**
  `TensorAccessorArgs` block chained by `next_compile_time_args_offset()`
  (`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:17-19`) and its own accessor
  (`:33,44,45`). Bind all three; the chained-offset CTA arithmetic disappears with them.
- `SliceTileTensorArgs` · **`output`** — **Case 1**. `Buffer*` @ writer RTA 0
  (`…_tile_tensor_args.cpp:151,168`), consumed by the **donor** kernel (see *Watch for*).

None of the twelve is the silent-wrong stale-address hazard: every one already arrives as a `Buffer*`
binding, which the framework patches on cache hits. This is routine port work, not a correctness fix.

**TensorParameter relaxation:** **`none`.** Apply no relaxation. In particular do **not** set
`dynamic_tensor_shape` for the two 3rd-arg drops below — they are Class 2, not Class 1. *(Confirmed by
the op owner, 2026-09-04: the readiness sheet's `Provisional relaxation finding` cell reading
`needs fix, then none` is **stale** — the relaxations are fine and the value is `none`. Ignore that
cell; do not go looking for a pending fix.)*

**TensorAccessor 3rd arg:** drop the redundant page-size argument at **exactly two** sites, both in
`SliceRm`:

- `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:43` — drop `padded_stick_size` (RTA 1);
  the host-side source is `slice_program_factory_rm.cpp:86`.
- `slice_writer_unary_stick_layout_interleaved_start_id.cpp:32` — drop `page_size_override` (RTA 7);
  host-side source `slice_program_factory_rm.cpp:155`.

Both are **Class 2** — the value equals what Metal 2.0 supplies implicitly, so this is a pure no-op.
The other ten accessors in the op pass no 3rd argument.

> **Update: this item is already done on `akertesz/slice-test` (PR #55433), commit `87bd11a885e`.** It
> drops both 3rd args, removes the two now-dead host RTAs, and reindexes both kernels — exactly as
> specified below. If you are porting on top of that branch, **verify and skip**; the text is kept for
> a port that starts from an unpatched tree.
>
> The safety question is settled, so do not re-litigate it: for a BLOCK/WIDTH-sharded RM tensor the
> dropped value is `buffer->page_size()` while the implicit one is `buffer->aligned_page_size()`, and
> those coincide only when `shard_W · element_size` is alignment-aligned — but
> `has_subaligned_shard_row` (`slice.cpp:47-57`) plus `needs_rm_composite_input` / `_output`
> (`:60-86`) reshard any tensor that would violate that, so `SliceRmProgramFactory` never sees one via
> `ttnn::slice`. PR #55433 also adds `check_accessor_page_size` (`c65cafac4ee`), a `TT_FATAL` in
> `create_descriptor` that pins the invariant for the one route bypassing that guard
> (`MeshPartition`). **Carry that assertion into the ported factory** — the spec path is exactly such a
> bypass-visible route, and Metal 2.0 offers no page-size override to fall back on.

Once the args are gone, `padded_stick_size` / `page_size_override` become dead RTAs. Remove them from
the host arg lists too (`…_rm.cpp:91` and `:165`), and mind that both kernels read a **fixed positional
run** — every later index shifts.

**CB endpoints:** four of seven CBs are already legal 1:1; three need a self-loop. Nothing needs the
multi-binding advanced option, nothing is dead, nothing is config-conditional.

| Factory / config | CB | Do this |
|---|---|---|
| `SliceRm` | `c_0` (`…_rm.cpp:322`) | **plain 1:1** — reader PRODUCER, writer CONSUMER. No action beyond the ordinary binding. |
| `SliceRmSharded` | `c_0`, borrowed from `input.buffer()` (`…_rm_sharded.cpp:282,290`) | **self-loop** — one toucher, and it is *sync-free*: the reader only calls `dfb_in.get_write_ptr()` (`slice_reader_…_rm_sharded.cpp:41`), no FIFO ops at all. Bind the reader **PRODUCER and CONSUMER**. Legal on Gen1 for DM; record the Quasar debt. |
| `SliceRmSharded` | `c_16`, borrowed from `output.buffer()` (`:294,302`) | **self-loop** — one toucher, locked producer (`reserve_back` `:40` / `push_back` `:89`), nothing drains it. Bind the reader PRODUCER **and** CONSUMER. |
| `SliceRmStride` (rank ≤ 4) | `c_0` (`…_rm_stride.cpp:69`) | **plain 1:1** — `reader_multicore_slice_4d` PRODUCER, `writer_multicore_slice_4d` CONSUMER. |
| `SliceRmStride` (rank > 4) | `c_0` (same descriptor, different kernels) | **plain 1:1** — `reader_multicore_slice_nd` / `writer_multicore_slice_nd`. |
| `SliceTile` | `c_0` (`…_tile.cpp:53-60`) | **plain 1:1**. |
| `SliceTileTensorArgs` | `c_0` (`…_tile_tensor_args.cpp:56`) | **plain 1:1** — slice reader PRODUCER, **donor** writer CONSUMER. |
| `SliceTileTensorArgs` | `c_1`, the start/end staging scratchpad (`…_tile_tensor_args.cpp:65`) | **self-loop** — one toucher that already holds both roles: the reader runs `reserve_back`/`push_back`/`wait_front`/`pop_front` twice over it (`…_tensor_args.cpp:52-59,66,69-76,83`). Bind the reader PRODUCER **and** CONSUMER; the kernel body is unchanged. |

**RTA / CRTA varargs:** six kernels carry genuine variable-count blocks. Reach for the vararg mechanism
on these; **name everything else.**

| Kernel | Vararg site | What it is |
|---|---|---|
| `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `:32-34` — `get_arg_addr(14)` | three `num_dims`-length blocks (`num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim`); `num_dims` is **RTA 4**, a runtime value |
| `slice_reader_unary_unpad_dims_rm_sharded.cpp` | `:26-30` — `get_arg_addr(1)`, `get_arg_addr(1 + num_cores_read*2)`, `get_arg_addr(1 + num_cores_read*3)`, `chunk_start_id + 1` | noc-x/y pairs + per-core chunk descriptors, all sized off runtime `num_cores_read` (RTA 0). **The trickiest one in the op** — see the aliasing warning below |
| `reader_multicore_slice_nd.cpp` | `:73-87` | **five** consecutive `tensor_rank`-length blocks (`input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps`) |
| `writer_multicore_slice_nd.cpp` | `:73` | one `tensor_rank`-length block (`output_dims`) |
| `reader_unary_unpad_dims_interleaved_start_id.cpp` | `:17-18` — `get_common_arg_addr(1)`; `:23` — `get_arg_addr(2)` | a `2·num_dims` **CRTA** block **and** a `num_dims` **RTA** block. `num_dims` is a CTA (`:13`) — still a vararg: it varies across instantiations |
| `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `:25-26` — `get_common_arg_addr(3)`; `:92` — `get_common_arg_addr(3 + 2*num_dims)`; `:31` — `get_arg_addr(2)` | two CRTA blocks (the second at a **computed** start offset) + one RTA block |

**Name these — they are not varargs:** `reader_multicore_slice_4d.cpp:52-77` is a fixed run of **25**
distinct fields via `rt_args_idx++`, and `writer_multicore_slice_4d.cpp:52-61` is **9**. A sequential
counter over a fixed set is legacy positional plumbing, not a loop. Same for every scalar that precedes
a vararg block (`src_addr`, `num_dims`, `start_id`, `num_tiles`, `tensor_rank`, `element_size`,
`num_rows_for_this_core`, `start_row_for_this_core`) and for the `SliceRm` writer's entire 11-arg set —
don't let those ride the varargs.

**No CTA varargs.** Every `get_compile_time_arg_val` in the op uses a literal constant index, so
`KernelAdvancedOptions::compile_time_varargs` is not needed anywhere.

## Watch for

- **CB endpoints (multi-binding):** **none.** No CB reaches ≥3 touchers or doubles a FIFO role, and the
  hidden-second-writer face is structurally impossible here — **the op declares no semaphores at all**,
  so there is nothing to coordinate a raw co-fill. Every `get_write_ptr()` / `get_read_ptr()` in the op
  was attributed to a kernel that already holds that CB's role. Do not set the flag anywhere.
- **Cross-op / shared kernels:**
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — the **borrowed**
    donor bound by `SliceTileTensorArgs` (`…_tile_tensor_args.cpp:133`). **A `_metal2` fork already
    exists beside it** at
    `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — a true
    locational sibling, not an `experimental/quasar/` copy. → **Rung 1: bind it, don't re-fork.** Its
    interface is now *your* constraint, and it fits without change: DFB **`dfb::out`**, tensor
    **`tensor::dst`**, named args **`args::num_pages`**, **`args::start_id`** — which is exactly the
    `{dst_buffer, num_tiles_per_core, num_tiles_written}` the factory supplies today. It gates
    `#ifdef OUT_SHARDED` / `#ifdef BACKWARDS`; slice sets **no** `defines`, so neither fires, same as
    now. Do **not** edit that fork — it already has consumers. Do not add the pointer comment to the
    legacy original either; it already has one.
  - **Sunset list** (coordination and tracking only — **not** authorization to convert the legacy
    eltwise copy in place): ≥15 factories still bind that path, including `data_movement/concat`,
    `data_movement/reshape_on_device`, five `data_movement/tilize` factories,
    `eltwise/unary_backward/tanh_bw`, `embedding`, `examples/example` (×2),
    `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads` (+ `_boltz`).
    Tracked as **issue #52228**, which also records a **duplicate** fork at
    `copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — bind the
    eltwise-sited one, not that. Record the still-unmigrated set in the port report so the eventual
    last port can see it was last.
  - **Slice's own `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` is neither borrowed
    nor lent** — `SliceTileProgramFactory` is its only binder, verified tree-wide. **Convert it in
    place; no fork.** And resist redirecting `SliceTile` onto the shared `_metal2` fork instead: slice's
    copy exists precisely because it takes its DFB index from a **named** CTA
    (`get_named_compile_time_arg_val("dfb_id_out")`, `…_tile.cpp:161`) so the fusion infrastructure can
    remap it — a capability the shared fork does not have. Collapsing them would be a functional change.
- **RTA varargs:** the six sites in the table above — reach for the vararg mechanism rather than trying
  to name each element; name the fixed 4D runs.
- **`ccl/mesh_partition` calls slice's `create_descriptor` from outside the op directory — your
  signature change breaks its build.**
  `ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp` calls
  `SliceOp::validate_on_program_cache_miss` and `SliceOp::select_program_factory` (`:126-127`), then
  `Factory::create_descriptor(...)` on the selected slice factory (`:131`), and refreshes through
  `ttnn::prim::patch_slice_program_addresses` (`:155`) — which is *deliberately* shared for this
  purpose (see `device/slice_program_factory_rm_sharded.cpp:350-353` and
  `mesh_partition_device_operation.hpp:47`). That op is `legacy (MeshWorkload)` with
  `Is able to port? = no`, so it cannot co-migrate, and fixing it is **outside your scope**. The audit
  routes the decision to the ops team.

  > **Update: resolved and implemented on `akertesz/slice-test` (PR #55433), commit `8c8b9eea947`** —
  > you do **not** need to stop here, and you should not re-solve it. That commit adds
  > `template <typename T> concept IsSliceSpecFactory = requires { &T::create_program_artifacts; };`
  > to `mesh_partition_program_factory.cpp` and branches both call sites on it: `create_at` builds a
  > spec factory through `MakeProgramFromSpec` + `SetProgramRunArgs`, and `override_runtime_arguments`
  > routes it through `UpdateProgramRunArgs(program, Factory::override_runtime_arguments(...))`; the
  > descriptor path is untouched for the factories still on it. Because the concept keys on the **entry
  > point** rather than a factory-name list, **each further factory you port needs no edit there** —
  > and the branch retires when the last one converts. Two caveats to carry into your port report: that
  > out-of-op-directory change was **explicitly authorized by the invoker**, and it is **not
  > run-verified** (MeshPartition's tests are t3000/TG-only).
- **The `id_per_dim` vararg block is *written* by the kernel, not just read.**
  `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:80`,
  `reader_unary_unpad_dims_interleaved_start_id.cpp:45`,
  `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:124` all do `id_per_dim[j]++` in place,
  and the host seeds the block per core (`…_rm.cpp:153`, `…_tile.cpp:117-124`). Confirm the vararg
  mechanism hands you a **writable** L1 view before assuming read-only; the dimension walk depends on
  the mutation persisting across loop iterations.
- **`slice_reader_unary_unpad_dims_rm_sharded.cpp` aliases two pointers one word apart into the same
  interleaved stream.** `read_noc_x = get_arg_addr(1)` and `read_noc_y = get_arg_addr(2)` (`:26-27`)
  read x and y out of a single interleaved x/y RTA run, both strided by 2 (`:85`). Preserve the layout
  exactly; this is the easiest thing in the op to get subtly wrong when converting a block to varargs.
- **`SliceRmSharded` reads *remote* cores' L1 at the address of its *own* borrowed CB.** The kernel
  takes `l1_read_addr = dfb_in.get_write_ptr()` (`:41`) — a **local** pointer into its own borrowed
  input CB — then uses that same value as the `.addr` of reads aimed at other cores (`:65`, `:76`).
  That works only because a sharded CB sits at the same L1 offset on every core in the range. Keep
  `c_0` bound `borrowed_from` the input so the invariant holds; it is not visible at the call site.
- **Two kernels are already part-modernized — expect a binding-layer change, not an idiom rewrite.**
  `reader_unary_unpad_dims_interleaved_start_id.cpp` already uses
  `get_named_compile_time_arg_val("dfb_id_in")` (`:12`), takes its page size from
  `dfb_in0.get_entry_size()` (`:33`) rather than `get_tile_size(cb)`, and passes the DFB object straight
  into `noc.async_read(s0, dfb_in0, …)` (`:40`). Slice's own writer copy is the same shape. The host
  already emits `named_compile_time_args` for both (`…_tile.cpp:139,161`).
- **A Device 2.0 → Metal 2.0 breadcrumb in the donor: confirm, don't swap blind.** The *legacy* eltwise
  donor reads its page size via `get_local_cb_interface(cb_id_out).fifo_page_size`
  (`writer_unary_interleaved_start_id.cpp:27`) — a **sanctioned** free function, which is why the
  Device 2.0 gate is green. Whitelist rule 7 moves such lookups onto the object, and the `_metal2` fork
  **has already done it** (`dfb.get_entry_size()`). Since you are reusing the fork, there is nothing to
  change; just don't mistake the legacy line for work you owe.
- **`constexpr` vs `const` in the tensor-args reader.**
  `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:15-16` declares `tile_width` and
  `tile_height` as **`const`** (not `constexpr`) while reading them from `get_compile_time_arg_val(3)`
  / `(4)`; every other CTA in that file is `constexpr`. That distinction decides token-form vs
  member-getter — check which the named-CTA replacement needs before swapping.
- **Do not "fix" what the audit flagged as pre-existing.** `METAL2_PREPORT_AUDIT.md` → *Misc anomalies*
  lists genuine latent issues (a dead `compile_time_element_size` CTA in four kernels, dead RTAs in the
  4D stride writer, an end tensor that `SliceTileTensorArgs` reads and discards, dead `#ifdef` branches
  in slice's own writer copy). Those route to the ops team and are **not** in the port diff — the port
  keeps binding and emitting them. In particular the **end tensor is still read** by the kernel
  (`…_tensor_args.cpp:69-83`), so it stays a live `TensorParameter` binding even though its value is
  unused.
