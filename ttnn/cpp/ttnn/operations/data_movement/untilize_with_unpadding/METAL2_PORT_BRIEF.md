# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

> **RE-ISSUED 2026-09-23 after the re-audit.** This replaces the 2026-09-08 brief and its
> partial-port addendum (both in git history). The re-audit was run against the working tree, where
> **all five factories are already on `ProgramSpecFactoryConcept`** — three committed on this branch,
> two restored in the staging area after PR #56280 removed the blocker that forced the 3-of-5 revert.
>
> So this is not a "go and port it" brief. Most of what the original brief asked for is **done and
> verified**; what remains is a short, specific list. The full record is in
> `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Scope:** all five factories, no subset, nothing deferred. The 3-of-5 narrowing is **obsolete** —
see *Why 5/5 is now correct* below.

**Recipe docs:** the `metal_2.0/` doc tree is not on this branch, so the provenance command prints
nothing. Pinned by content: audit recipe blob `d0576d6d739`; newest `metal_2.0/` commit on
`akertesz/op-porting-recipe` is
`4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
*(Carry this line into the port report's Provenance section.)*

---

## Why 5/5 is now correct

The 2026-09-09 revert existed for exactly one reason: `untilize` codegen's non-tile-aligned L1
fallback (`build_native_equivalent`) selected a factory from
`UntilizeWithUnpaddingDeviceOperation` and called its `create_descriptor` — which a ported factory no
longer has. Two of the five factories were reachable on that path, so they went back to the
descriptor API.

**PR #56280 deleted `build_native_equivalent`.** `ttnn::untilize` now decides the Native tier itself
via `codegen_cb_plan_fits_live_l1()` (`untilize/untilize.cpp:178`) and routes to `untilize_native` as
a whole when no codegen plan fits. Verified independently: **no file outside this op directory
references any `UntilizeWithUnpadding*ProgramFactory`**, and no `create_descriptor` survives in the
op. The `if constexpr (requires { … })` + `TT_THROW` guard the revert needed is gone with it, so
there is nothing left to un-guard.

## Remaining work

Four items. Item 1 is the substantive one.

### 1. Take the shared-kernel sunset — six orphaned legacy kernels

The `_metal2` fork convention's closing move is that **the last consumer to migrate deletes the
legacy copy**. #56280's description names this port as that point: *"The untilize_with_unpadding port
(#51308) reuses the forks as-is and is the sunset point for the three legacy copies."* With the 5/5
restore in place, six legacy kernels have **zero binders repo-wide** (swept for both the full-path
and the split-literal `"…/dir/" "name.cpp"` spellings, excluding `experimental/quasar/**`):

| # | Delete | Newly orphaned by | Build file to edit with it |
|---|---|---|---|
| 1 | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | the staged BlockInterleaved restore | — (glob only) |
| 2 | `untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` | the staged BlockInterleaved restore | **`data_movement/CMakeLists.txt:88`** |
| 3 | `untilize/device/kernels/compute/untilize_wh.cpp` | the staged BlockInterleaved restore | — (glob only) |
| ~~4~~ | ~~`untilize/device/kernels/compute/untilize.cpp`~~ — **kept, see (c)** | the staged MultiCoreInterleaved restore (last *factory* binder) | — |
| 5 | `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | this branch's committed NDSharded port | — (glob only) |
| 6 | `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | this branch's committed Sharded port | **`ttnn/sources.cmake:173`** |

Rows 5–6 have been deletable since the committed 3-of-5 state; rows 1–4 become deletable with the
restore.

**Three things not to miss.** (a) Rows 2 and 6 are named **explicitly** in build files — a stale
explicit entry is a hard CMake error, not a warning. The others are covered only by `GLOB_RECURSE`
and need no build edit. Row 6 is a *replacement*, not a removal: `ttnn/sources.cmake` has no glob, and
the `…_blocks_metal2.cpp` fork this branch created was never added to the explicit list even though
its `…_start_id_metal2.cpp` sibling is there. A dev tree hides that (JIT resolves the path from the
checkout); an installed build would ship without the fork. (b) Each surviving `_metal2` fork carries a
note saying *"Until the last of them migrates and the original is retired, changes here likely belong
there too."* Once its original is gone that note is false; drop it in the same change. The reciprocal
pointer comments #56280 added to rows 1–3 die with the files. (c) Row 4 is **not** deletable. The
sweep looked only for factory bindings;
`tests/…/fused/parallel_sequential/test_parallel_sequential.py:1435` lists `untilize.cpp` in
`TestCrossOpCompilation.KERNEL_PATHS` and `open()`s it to read its source as fusion input, so deleting
it fails that suite. Repointing at the fork is not a safe substitution (named args and
`experimental/kernel_args.h` are not what that harness inlines), so the file and its fork's pointer
comment both stay. The sunset is five files.

### 2. Decide what happens to the unstaged BACKWARDS edit

`eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp` is modified
in the working tree and **not staged**: it rewrites the `#ifdef BACKWARDS` page-id walk (`:41-49`)
and adds a comment (`:8-9`) declaring the fork deliberately out of sync with its original.

- It is **latent, not live** — nothing in the tree defines `BACKWARDS` for this kernel. The only
  definer is `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:137`, which binds
  a different file. Both `#ifdef` branches are dead code in every current consumer.
- The bug it fixes is **real**: `dim` is `uint32_t`, so `dim > -third_dim` is `0 > huge` and the loop
  never executes. `METAL2_PORT_REPORT.md` *Handoff points* item 3 documents it in full.
- It is now a **cross-op** edit: since #56280, `data_movement/untilize`'s block factory binds this
  same fork (`untilize_multi_core_block_program_factory.cpp:60`).

**Pick one, and say which in the PR:** either drop it from this diff and route the correction to the
`eltwise/unary` kernel owners as a standalone change fixing the **legacy original and the fork
together** (cleaner, and what the port report already proposed — it also avoids creating a permanent
fork/original divergence marker), or keep it and put the `untilize` owners on the review as a
deliberate out-of-scope shared-kernel fix. The one outcome to avoid is it riding in unstaged and
being either committed unmentioned or silently lost.

Note that row 1 of the sunset deletes the *legacy original* carrying this bug — so if you take the
sunset, the legacy side resolves by removal and only the fork's form is left to decide.

### 3. Two dead CTAs the port can now remove for free

`…_multi_core_nd_sharded_program_factory.cpp` emits 16 named CTAs (`:191-207`) but
`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` reads 14 — `output_stick_size`
(`:192`) and `input_single_tile_size` (`:199`) are never read.

The 2026-09-08 audit routed this to the ops team because dropping a *positional* CTA meant
renumbering the kernel's reads. **That reason is gone**: with named args there is nothing to
renumber, so this is a two-line deletion in a factory this port rewrote anyway. Worth doing here.
(`output_stick_size`'s computation at `:167` becomes dead with it.)

### 4. Re-run BFLOAT8_B coverage on the block path

#56280 also changed `untilize_with_unpadding.cpp:95-105,123` so the output CB estimate and the
pending-L1 reservation are sized by the **output** dtype. For BFLOAT8_B inputs that is 2048 B/tile
instead of 1088 B, and the reservation is no longer silently zero — which makes `enough_space_height`
stricter and pushes BFLOAT8_B shapes that used to select **MultiCoreInterleaved** onto
**BlockInterleaved**.

The 5/5 verification recorded in `METAL2_PORT_REPORT.md` (`1195 passed / 10 skipped / 8 xfailed`)
predates that change, so the restored block factory's BFLOAT8_B coverage is **not** inherited from
it. Re-run `test_untilize_with_unpadding.py` and `test_untilize.py` on the current tree rather than
citing the old numbers.

---

## Already done — verify, don't redo

Each of these was a "to do" in the 2026-09-08 brief and is implemented in the working tree. Listed so
you can confirm rather than re-derive.

### TTNN factory analysis

- **Current concept:** `MetalV2` on all five factories (`descriptor` on `origin/main`, which is what
  the readiness sheet tracks — not a conflict, just two snapshots).
- **Op-owned tensors:** none. **Target concept:** `ProgramSpecFactoryConcept`, reached on all five.
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` other than `none` ·
  `get_dynamic_runtime_args`. Also absent, though none of them would have gated: a custom hash, an
  `override_runtime_arguments`, a pybound `create_descriptor`. **This port carries no user-visible
  API change.**

### Tensor bindings — all Case 1 or clean, no Case 2

No `get_bank_base_address` bridge is needed anywhere in this op; no kernel does hand-rolled NoC
arithmetic on a tensor base. There is not a single `->address()` expression left in the op directory.

- **Case 1 → `TensorBinding` + `TensorAccessor(tensor::name)`:** SingleCore input/output ·
  MultiCoreInterleaved input/output (`:109-112`, `:133-136`) · BlockInterleaved input/output
  (`:177-180`, `:202-205`) · Sharded output in the cross-shard-type (`:223`), HEIGHT→interleaved
  (`:288`) and W/B→interleaved (`:308`) configs · NDSharded output+input on the writer (`:183-190`)
  and input on the reader.
- **clean (borrowed-memory DFB read):** Sharded `SH_IN` ← input (`:163`) and, in the same-shard-type
  sharded-output configs only, `SH_SHARDED_OUT` ← output (`:188`).

### TensorAccessor 3rd arg — both sites dropped

`writer_unary_stick_layout_split_rows_multicore.cpp:34` and `writer_unary_unpad_cross_sharded.cpp:33`
both now read `TensorAccessor(tensor::dst)`, and the host CTAs that fed them are gone. A sweep of all
13 `TensorAccessor(` sites across the op and its donors finds every one taking only the binding
token. No site was Class 1, so **no `dynamic_tensor_shape`** is set.

The binding supplies `TensorAccessorArgs<…>::AlignedPageSize`
(`tt_metal/hw/inc/api/tensor/tensor_accessor.h:97`), which is the stride the allocator actually laid
the buffer out with — so in every reachable configuration it equals the value the two sites used to
compute by hand. The audit's classification table shows the per-config equality, including the
alignment argument the sharded configs need.

### CB endpoints — all dispositions implemented

- **Self-loop** on `SH_SHARDED_OUT` — the writer is bound both PRODUCER and CONSUMER
  (`…_multi_core_sharded_program_factory.cpp:249-257`). Its only toucher fills it by `get_write_ptr`
  and nothing drains it, because the DFB *is* the output buffer.
- **`borrowed_from`** on `SH_IN` (`:163`) and `SH_SHARDED_OUT` (`:188`).
- **Conditional DFBs** — `SH_SHARDED_OUT` only under `out_sharded && !cross_shard_type` (`:178`);
  the BlockInterleaved cliffrow input/output pair only when the split produced a cliff row
  (`…_block_interleaved…:142-144`).
- **No multi-binding advanced option, no dead DFB**, in any factory or configuration.

### RTA / CRTA varargs — both ported

- `writer_unary_stick_layout_split_rows_multicore.cpp:76-85` on `get_vararg(...)`, with the index
  base correctly reset to `0` (varargs live in their own section, so the legacy offset of 4 no longer
  applies), and per-core counts set through
  `advanced_options.num_runtime_varargs_per_node[core]` (`…_multi_core_interleaved…:268`). The four
  leading legacy args split correctly: `dst_addr` became the binding; `padded_X_size`,
  `start_stick_id`, `n_block_reps` are named RTAs.
- The ND-sharded writer's two shape loops on common runtime varargs, with
  `num_common_runtime_varargs = 2 * tensor_rank` (`…_multi_core_nd_sharded…:211`).
- **CTA varargs: none.**

---

## Watch for

- **The three `_metal2` forks now have a second consumer, and their arg names are frozen.**
  `reader_unary_interleaved_wh_multicore_metal2.cpp`,
  `writer_unary_stick_layout_wh_multicore_metal2.cpp` and `untilize_wh_metal2.cpp` are bound by
  **both** this op's BlockInterleaved factory and `untilize`'s block factory
  (`untilize_multi_core_block_program_factory.cpp:60,63,65`). The restored factory matches their
  vocabularies exactly — reader CTAs `num_tiles_per_2d`/`third_dim`/`total_tiles_per_row` and RTAs
  `start_id`/`single_block_size_row_arg`/`single_block_size_col_arg`; writer CTAs
  `total_num_rows`/`third_dim`/`tile_height`/`unpadded_X_size` plus 7 RTAs; compute CTAs
  `block_size_col`/`block_size_row`/`third_dim`. **Any rename now breaks `untilize`.**

- **Seven of the eight op-owned writers were converted in place, and that is correct.** The reverse
  ("lent") sweep the 2026-09-08 audit skipped has now been run over all eight:
  `writer_unary_stick_layout_wh_multicore.cpp` was the only shared one, and it is handled via the
  #56280 fork. The other seven have this op as their sole consumer, so in-place conversion needs no
  fork. Don't let the fork convention talk you into forking them retroactively.

- **Same-source kernel instances here are the *disjoint-node* variant, not a dual-instance
  work-split.** BlockInterleaved emits two readers and two writers (`:297-304`) and up to four
  compute instances (`:306-353`); MultiCoreInterleaved emits two compute instances (`:192-207`).
  Every one covers a **disjoint** node set via its own `WorkUnitSpec::target_nodes`, and each block
  set binds its **own** DFB pair. Each node sees exactly one instance per role → ordinary 1:1. Two
  `TT_FATAL`s keep the args and buffers from drifting: `block_size_row == set.block_tiles`
  (`:253-258`) and `single_sub_block_size_row_arg == set.block_tiles` (`:409-415`). Keep both.

- **`make_block_plan` reads live L1 occupancy**, so it is valid only on a program-cache miss — which
  is the only time `create_program_artifacts` runs. `untilize`'s block factory states this in a
  comment (`untilize_multi_core_block_program_factory.cpp:79-80`); the factory here does the same
  thing silently. A matching comment would help the next reader.

- **`get_tile_size(dfb::out)` at `writer_unary_unpad_width_16_sharded.cpp:23` must stay the free
  function.** It feeds a `static_assert` and `NOC_MAX_BURST_SIZE` template arguments, so it has to be
  a constant expression, which the `DataflowBuffer` member getter cannot yield. It is on the
  sanctioned list; the kernel's own comment (`:20-22`) records why. Don't "finish" the swap here.

- **One in-family donor call, and it needs nothing.** `writer_unary_stick_layout_split_rows_multicore.cpp:12`
  and `writer_unary_unpad_cross_sharded.cpp:10` include
  `ttnn/operations/data_movement/common/kernels/common.hpp` and call
  `tt::data_movement::common::noc_async_write_sharded(Noc, uint32_t l1_addr, AddrGenType, …)` —
  `Noc` by value and the accessor **by value**, Shape 1, ✓ excellent. The remaining `uint32_t
  l1_addr` is a DFB read pointer, not a resource handle.

- **The ND-sharded writer binds the *input* purely for shard geometry.** It builds
  `TensorAccessor(tensor::src)` and uses it only for `shard_pages(...)`; it never reads input data.
  That is a real second tensor binding on that kernel and is present (`…_nd_sharded…:187-190`) —
  don't prune it as redundant.

- **Stay out of `ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/`.** A copy of
  this op lives there carrying a deliberately hacky shortcut port, and quasar copies show up as
  apparent co-borrowers of six of the nine borrowed kernels. It is not a production port and not a
  precedent — its `_metal2` files are whole-op pre-port copies that do **not** count as forks to
  reuse, and it ships idioms this recipe forbids sitting inline with code that reads perfectly well.
  The audit excluded it from every table and from every count in the sunset list above; you should
  not read it either.

- **Two questions are open for the op owner** and are in the audit's *Questions* section, not here,
  because neither blocks: readiness-sheet factory-set confirmation, and whether a 1-byte output
  element size (e.g. `UINT8`) can reach the interleaved-input → 2D-sharded-output path. Neither
  changes any verdict above.
