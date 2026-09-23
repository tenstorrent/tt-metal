# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

> **RE-AUDIT, 2026-09-23.** This file replaces the 2026-09-08 audit and its post-port addendum; both
> are recoverable from git history. It was requested after PR
> [#56280](https://github.com/tenstorrent/tt-metal/pull/56280) landed on `main` and the working tree
> restored the **5-of-5** factory port that the 2026-09-09 revert had narrowed to 3 of 5.
>
> The audit is run against the **working tree** (`main` merged at `c3038aee531`, plus the staged
> restore of the two factories and one unstaged kernel edit), not against a pre-port snapshot.

One device-operation, five program factories. **All five are now on `create_program_artifacts`
(`MetalV2`)** in the working tree — three from commits already on this branch, two from the staged
restore:

- **`UntilizeWithUnpaddingDeviceOperation`** (`device/untilize_with_unpadding_device_operation.{hpp,cpp}`; params in `device/untilize_with_unpadding_device_operation_types.hpp`)
  - `UntilizeWithUnpaddingSingleCoreProgramFactory` — `…_single_core_program_factory.cpp` (committed)
  - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` — `…_multi_core_interleaved_program_factory.cpp` (**staged restore**)
  - `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` — `…_multi_core_sharded_program_factory.cpp` (committed)
  - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` — `…_multi_core_block_interleaved_program_factory.cpp` (**staged restore**)
  - `UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory` — `…_multi_core_nd_sharded_program_factory.cpp` (committed)

Every factory header declares only
`static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)` at line 14; no
`create_descriptor` survives anywhere in the op directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** the `metal_2.0/` doc tree is **not present on this branch**, so the provenance
command prints nothing. Pinned by content instead: audit recipe blob `d0576d6d739`
(`audit/metal2_audit.md`), identical on every branch that carries the tree; newest commit touching
`metal_2.0/` on `akertesz/op-porting-recipe` is
`4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
(The 2026-09-08 audit cited the same subject line at hash `9c1a0466220` — a rebase of the same commit,
not a different revision.)

---

## What changed since the 2026-09-08 audit

Six things moved. Four are clean wins, two are open items you have not actioned yet.

| # | Change | Effect on this audit |
|---|---|---|
| 1 | **`build_native_equivalent` deleted** by #56280 | The sole reason for the 3-of-5 revert is gone. `ttnn::untilize` now decides the codegen Native tier itself (`codegen_cb_plan_fits_live_l1`, `untilize/untilize.cpp:178`) and no longer reaches into this op's factories. **Restoring 5/5 is correct**, and the `if constexpr (requires { … create_descriptor })` guard the revert needed no longer exists to remove. |
| 2 | **Three `_metal2` forks now pre-exist**, created by #56280 | `reader_unary_interleaved_wh_multicore_metal2.cpp`, `writer_unary_stick_layout_wh_multicore_metal2.cpp`, `untilize_wh_metal2.cpp`. The 2026-09-08 audit said this port would create all three; it now **reuses** them, and the restored block factory does exactly that. Named-arg vocabularies match on all three — verified field by field. |
| 3 | **The sunset has come due and has not been taken** | Six legacy kernel copies now have **zero binders** repo-wide; four of them dropped to zero *because of the staged restore*. #56280's own description names this port as "the sunset point for the three legacy copies." **Nothing has been deleted.** See [Open item A](#open-item-a--six-orphaned-legacy-kernels-are-not-deleted). |
| 4 | **A shared fork is edited, unstaged** | `reader_unary_interleaved_wh_multicore_metal2.cpp` carries an uncommitted BACKWARDS-loop correction. That fork is bound by `data_movement/untilize` too. See [Open item B](#open-item-b--an-unstaged-behavioural-edit-to-a-fork-another-op-binds). |
| 5 | **`enough_space_height` got stricter for BFLOAT8_B** (#56280, `untilize_with_unpadding.cpp:95-105,123`) | Factory *selection* shifted: BFLOAT8_B inputs now size the output estimate at 2048 B/tile instead of 1088 B, so more BFLOAT8_B shapes route to **BlockInterleaved**. The restored block-factory port therefore sees traffic it did not see in the pre-merge 5/5 run. Verification heads-up, not a gate. |
| 6 | **Two prior Misc anomalies self-resolved** | The dead CTAs the old audit routed to the ops team (ND-sharded writer, MultiCoreInterleaved dead locals) are either gone or now trivially removable, because named args make CTA removal renumber-free. |

Everything else the 2026-09-08 audit found still holds and is re-stated below against the current
code, not carried over on trust.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `UntilizeWithUnpaddingDeviceOperation` → SingleCore · MultiCoreInterleaved · MultiCoreSharded · MultiCoreBlockInterleaved · MultiCoreNDSharded |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 17 bound kernels structurally Device 2.0; the single free-function call is sanctioned |
| *Prereqs* — Cross-op escapes | **Ok** — one in-family function-call escape, ✓ excellent shape; 9 borrowed kernel files (coordination cost, not a gate) |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no `get_compile_time_arg_val` at a varying index anywhere |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, all five factory rows (value supplied by the launching user in the 2026-09-08 session; no Drive connector here either — see *Gate detail* and *Questions*) |
| *TTNN Readiness* — Concept (current) | **`MetalV2`** on all five factories *in the working tree*; `descriptor` on `origin/main`, which is what the readiness sheet tracks — see *Gate detail* for why that is not a sheet conflict |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — no `create_workload_descriptor` in the op |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash`, no backdoor `attribute_values` / `to_hash` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — absent; base concept applies |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `untilize_with_unpadding_nanobind.cpp` binds only `ttnn::untilize_with_unpadding` |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** — reached on all five factories |
| *Port work* — Offset base pointer | **none** — no `->address()` expression anywhere in the op; every base rides a `TensorBinding` |
| *Port work* — Tensor bindings (per binding) | all **Case 1** or **clean** (two borrowed-memory DFB reads); no Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** on all five rows (clears; value supplied by the launching user) |
| *Port work* — TensorAccessor 3rd arg | **none remaining** — both Class-2 sites dropped; zero accessors in the op or its donors now pass a 3rd argument |
| *Port work* — CB endpoints | **legal** everywhere except one **self-loop** (`SH_SHARDED_OUT`, Sharded / same-shard-type sharded output), which is also a **conditional DFB** — both implemented |
| *Port work* — **Shared-kernel sunset** | **6 orphaned legacy kernels not yet deleted** — the one substantive item this re-audit adds |

**CB endpoints** are dispositions, not gates. Every CB in this op is either an ordinary
1-producer/1-consumer FIFO or the single-toucher `SH_SHARDED_OUT` that takes a self-loop. No
multi-binding advanced option is needed anywhere, and no CB is dead.

## Result

**GREEN → brief re-issued.** Every gate clears against the current tree: Device 2.0 ✓ · Feature
compatibility ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd argument ✓.

**Nothing is scoped out — the 3-of-5 narrowing is obsolete.** The reachability wall that forced it
(`build_native_equivalent` calling `create_descriptor` on a factory that no longer had one) was
removed by #56280; no code outside this op directory references any
`UntilizeWithUnpadding*ProgramFactory` any more. The restored 5/5 state is the correct state.

**Two open items remain, neither a gate:**

- **A — six orphaned legacy kernel copies are not deleted.** This is port work the fork convention
  assigns to the last consumer to migrate, and #56280 explicitly named this port as that point.
- **B — an unstaged behavioural edit sits on a `_metal2` fork that `data_movement/untilize` also
  binds.** It is latent-only today, but it is a cross-op change and it is not staged, so it is at
  risk of being committed silently or dropped entirely.

---

## Open item A — six orphaned legacy kernels are not deleted

With the 5/5 restore in place, **zero** program factories anywhere in the repo bind these six files.
Counts below are from a repo-wide sweep of both binding spellings (a full single-line path, and the
`"…/dir/" "name.cpp"` split-literal form the factories use), excluding `experimental/quasar/**`.

| # | Orphaned legacy kernel | Orphaned by | Fork that replaced it | Build-file entry to remove with it |
|---|---|---|---|---|
| 1 | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp` | **the staged restore** (BlockInterleaved) | `…_metal2.cpp` (#56280) | — (glob only) |
| 2 | `untilize_with_unpadding/device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` | **the staged restore** (BlockInterleaved) | `…_metal2.cpp` (#56280) | `ttnn/cpp/ttnn/operations/data_movement/CMakeLists.txt:88` |
| 3 | `untilize/device/kernels/compute/untilize_wh.cpp` | **the staged restore** (BlockInterleaved) | `untilize_wh_metal2.cpp` (#56280) | — (glob only) |
| ~~4~~ | ~~`untilize/device/kernels/compute/untilize.cpp`~~ — **not deletable, see below** | the staged restore (MultiCoreInterleaved was the last *factory* binder) | `untilize_metal2.cpp` | — (glob only) |
| 5 | `data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | this branch's committed NDSharded port | `…_metal2.cpp` | — (glob only) |
| 6 | `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | this branch's committed Sharded port | `…_metal2.cpp` (created here) | `ttnn/sources.cmake:173` |

Rows 1–4 are the ones the restore newly orphans; rows 5–6 have been orphaned since the committed
3-of-5 state and were also missed.

**Correction — row 4 survives; the sunset is five files, not six.** The sweep above searched only for
*factory* bindings, which is the wrong net for this one file. `untilize.cpp` has a live non-factory
consumer: `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1435`
lists it in `TestCrossOpCompilation.KERNEL_PATHS` and `open()`s the path to read its source text as
fusion input. Deleting it turns every "untilize" pair in that suite into a `FileNotFoundError`.
Repointing the entry at `untilize_metal2.cpp` is not a safe substitution — the fork's named-arg and
`experimental/kernel_args.h` form is not what that fusion harness is built to inline — so row 4 is
left in place and its fork keeps the pointer comment. Rows 1, 2, 3, 5, 6 are deleted.

The same net was too narrow in the other direction for row 6. `ttnn/sources.cmake` has **no** glob;
`TTNN_CORE_JIT_API_HEADERS` is fully explicit, and while the legacy copy was listed at `:173`, the
`…_blocks_metal2.cpp` fork this branch created was never added — even though its sibling
`writer_unary_stick_layout_interleaved_start_id_metal2.cpp` is listed two lines down. A dev tree hides
this (JIT resolves the path from the source checkout), which is why the branch's tests passed, but an
installed build would ship without the fork. The sunset therefore **replaces** `:173` rather than
deleting it.

**Why this matters and why it is not optional.** The `_metal2` fork convention exists so a shared
kernel can be migrated one consumer at a time; its closing move is that the last consumer to migrate
**deletes the legacy copy**. #56280's PR description states it directly: *"The untilize_with_unpadding
port (#51308) reuses the forks as-is and is the sunset point for the three legacy copies."* Leaving
them is not merely untidy — each orphan is a file that still compiles, still ships in the kernel file
set, and still reads as a live shared kernel to the next auditor or porter, who will then treat it as
a co-borrower constraint that does not exist.

**Two mechanical details the deletion needs.** Rows 2 and 6 are named **explicitly** in build files
(the `GLOB_RECURSE` in `data_movement/CMakeLists.txt` would have caught row 2 anyway; the explicit
entry is a redundant leftover, but a stale explicit entry is a hard CMake error, not a warning). Row
6's entry is replaced by its fork rather than removed, per the correction above. The others are
glob-only and need no build edit. Rows 1 and 3 also carry a "Ops ported to Metal 2.0
bind the fork" pointer comment added by #56280; that comment dies with the file, and the
corresponding *"changes here likely belong there too"* note in each surviving `_metal2` fork becomes
stale and should be dropped in the same change.

**Routing: PORT WORK** (this port's own diff), not an ops-team hand-off.

## Open item B — an unstaged behavioural edit to a fork another op binds

`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp`
is modified in the working tree and **not staged**. The edit rewrites the `#ifdef BACKWARDS` page-id
walk (`:41-49`) and adds a comment (`:8-9`) declaring the fork deliberately out of sync with its
legacy original.

Three facts, in the order that matters:

1. **It is latent, not live.** Nothing in the tree defines `BACKWARDS` for this kernel. The only
   `kernel_defines["BACKWARDS"]` in the repo is
   `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:137`, which binds a
   different file (`copy/device/kernels/reader_unary_start_id.cpp`). Both branches of the `#ifdef`
   are dead code in every current consumer, so the edit changes no observable behaviour today.
2. **The bug it fixes is real.** In the legacy form, `dim` is `uint32_t` and the condition is
   `dim > -third_dim`; `-third_dim` wraps to a huge unsigned value, so `0 > huge` is false and the
   loop never runs at all. `-start_id` wraps likewise. This is fully documented in
   `METAL2_PORT_REPORT.md` *Handoff points* item 3, which routed it to the `eltwise/unary` kernel
   owners.
3. **It is now a cross-op edit.** Since #56280, `data_movement/untilize`'s ported block factory binds
   this same fork (`untilize_multi_core_block_program_factory.cpp:60`). The 2026-09-08 report's
   reasoning for *not* touching the legacy original — *"that kernel is shared … so it needs its
   owner's review"* — applies with equal force to the fork now that it has a second consumer.

**What to decide before committing — and Open item A decides most of it.** The legacy original that
carries the same bug is orphan **row 1**, so taking the sunset deletes it rather than leaving it to
the `eltwise/unary` owners. That removes the two reasons to split the correction out: there is no
original left to fix alongside the fork, and no fork/original pair left to diverge. Keeping the edit
in this port's diff is then the coherent route, stated in the PR description as a shared-kernel fix
with the `untilize` owners on the review, since they are the fork's other consumer.

If the sunset is *not* taken, the earlier reasoning stands: drop the edit here and route the
correction to the `eltwise/unary` owners as a standalone change fixing the original and the fork
together. What should not happen either way is the third outcome — it rides in unstaged and is
either committed unmentioned or silently lost.

One consequence for the edit as written: its added comment (*"the BACKWARDS walk below is corrected
here and still carries the original's unsigned-comparison form there, so do not sync that block back
from the original"*) is a divergence marker pointing at a file the sunset deletes. It must come out
with the deletion; the corrected loop stays.

**Routing: FYI-P** (porter decision) **+ ops team** (`eltwise/unary` kernel owners) for the legacy
original, which carries the bug either way.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.**

  **Provenance, unchanged from the 2026-09-08 session and still a limitation.** The gate value
  (`Is able to port? = yes`) and the relaxation value (`TensorParameter relaxation = none`), for all
  five factory rows, were **supplied by the launching user** — *"Is able to port column is yes and
  tensor parameter relaxation column is none for all factories under this"*. This session likewise has
  no Drive connector, so the sheet could not be re-fetched. Every cross-checkable **primary** column
  was verified against the current code independently:

  | Column | Sheet value (as supplied) | Code evidence (working tree) | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | **`MetalV2`** in the working tree — five `create_program_artifacts` declarations at `device/factories/*_program_factory.hpp:14`, zero `create_descriptor`. On `origin/main` all five are still `descriptor`. **Not a conflict** — see below. | ✓ (see note) |
  | `Custom hash` | `no` | No `compute_program_hash` override on the device-op (`device/untilize_with_unpadding_device_operation.hpp:31-43`), no `attribute_values` / `to_hash` anywhere in the op. | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | Hook absent; a grep of the whole op directory returns zero hits. | ✓ |
  | `Override runtime args method?` | `no` | No `override_runtime_arguments` on any factory; zero textual occurrences now that both factories were rewritten. → base `ProgramSpecFactoryConcept`. | ✓ |
  | `Pybind descriptor` | `no` | `untilize_with_unpadding_nanobind.cpp` binds only `ttnn::untilize_with_unpadding`; no `create_descriptor` binding. Nothing for the port to delete ⇒ **no user-visible API change**. | ✓ |
  | `Secretly SPMD Workload?` | N/A | Only applies at `Concept == WorkloadDescriptor`. | ✓ |
  | `Op-owned tensors?` | `no` | No `buffers` vector constructed anywhere. Cross-column invariant holds. | ✓ |

  **The `Concept` mismatch is expected, not a broken sheet.** The recipe's "spreadsheet is broken"
  trigger is a primary-column conflict against the code the sheet describes. The sheet tracks `main`;
  this branch's port commits are not on `main` (`git log origin/main..HEAD` lists them all). So the
  sheet's `descriptor` is correct for `main` and the working tree's `MetalV2` is this in-flight port —
  two different snapshots, not a disagreement. The recipe's other `MetalV2` rule (*"the factory is
  already ported; report it as done, not blocked"*) is the one that applies to the working tree, and
  it is why this audit reads as a verification pass rather than a feasibility pass.

  Cross-column invariants hold: `get_dynamic_runtime_args == no` is consistent with the concept, and
  `Op-owned tensors? == no` is required on it.

  **Factory-set match — still only partially checkable.** The code has exactly five factories, fully
  enumerated in `program_factory_t` (`device/untilize_with_unpadding_device_operation.hpp:24-29`).
  The user's statement covers *"all factories under this"*, so the row set is asserted to match, but
  the individual rows were not in hand to confirm one-to-one. Unchanged from the prior audit; carried
  forward as a question rather than a finding.

- **Device 2.0 (every kernel used): GREEN.** All 17 currently-bound kernel files were re-scanned —
  8 op-owned writers (7 converted in place, 1 bound as the `_metal2` fork beside its original), plus
  9 borrowed. Across all of them there is **no** raw `noc_async_read` / `noc_async_write`, no
  `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no
  manual CB index management, no raw semaphore address (the op declares no semaphores at all), and
  **no remaining `CircularBuffer` wrapper** — every `get_read_ptr()` / `get_write_ptr()` in the set is
  a `DataflowBuffer` **method**, not a CB-index free function. There is no holdover table because
  there are no holdovers.

  One free function survives and is **sanctioned**:

  | File | Line | Call | Why it is not a violation |
  |---|---|---|---|
  | `device/kernels/dataflow/writer_unary_unpad_width_16_sharded.cpp` | 23 | `get_tile_size(dfb::out)` | Sanctioned free function. It must stay a constant expression — it feeds a `static_assert` and `NOC_MAX_BURST_SIZE` template arguments — which the `DataflowBuffer` member getter cannot yield. The kernel's own comment at `:20-22` records exactly this. Per the recipe, the sanctioned list does not turn on what object is in scope. |

  **Improvement since 2026-09-08:** the two kernels then flagged as still on the Device 2.0
  `CircularBuffer` wrapper (`writer_unary_unpad_sharded_to_interleaved.cpp` and the borrowed
  `writer_unary_stick_layout_interleaved_blocks.cpp`, the latter also passing `CircularBuffer&` to a
  file-local helper) are both on `DataflowBuffer` now — the second via its `_metal2` fork. That
  heads-up is retired.

- **Feature compatibility:** every Appendix A entry, in order. All `N/A` — the feature is *absent*, so
  the entry cannot fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, `using` alias, or `CreateGlobalCircularBuffer` call anywhere in the op. No `.global_circular_buffer` field (the arcane descriptor-API signal) — and no `CBDescriptor` at all survives, since all five factories now emit `DataflowBufferSpec`. No `remote_index(`, no `remote_cb_*` identifier, no `remote_circular_buffer.h` include, no `num_global_cb_receivers`. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | No `address_offset`, no `set_address_offset`, no four-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` — zero textual hits in the op directory. The borrowed-memory pattern that used to set `.buffer` is now `DataflowBufferSpec::borrowed_from` (`…_multi_core_sharded_program_factory.cpp:163` and `:188`), which is the mechanical translation, explicitly not this entry. |
  | GlobalSemaphore | **N/A** | The op uses no semaphores of any kind. A case-insensitive grep for `semaphore` across `device/` returns zero hits. |

- **CB endpoints (GATE-free): every CB legal or carrying an implemented disposition.** Census per CB,
  per node, per configuration; raw-pointer access counted as an endpoint alongside FIFO ops.

  | Factory | DFB | Config | Census on a node | Verdict | Disposition (as implemented) |
  |---|---|---|---|---|---|
  | SingleCore | `SC_IN` / `SC_OUT` | all | reader P + compute C / compute P + writer C | plain 1:1 | none |
  | MultiCoreInterleaved | `MCI_IN` / `MCI_OUT` | all | reader P + compute C / compute P + writer C | plain 1:1 | none |
  | BlockInterleaved | `BI_IN_FULL` / `BI_OUT_FULL` | full set | reader P + compute C / compute P + writer C | plain 1:1 | none |
  | BlockInterleaved | `BI_IN_CLIFFROW` / `BI_OUT_CLIFFROW` | cliffrow set | same pair, cliffrow cores only | plain 1:1 | **conditional DFB** — emitted only when `!cliffrow_set.empty()` (`…_block_interleaved…:142-144`) |
  | Sharded | `SH_IN` | all | reader P + compute C | plain 1:1 | **borrowed-memory** → `borrowed_from = SH_INPUT` (`:163`) |
  | Sharded | `SH_OUT` | all | compute P + writer C | plain 1:1 | none |
  | Sharded | `SH_SHARDED_OUT` | same-shard-type sharded output only | writer only — `reserve_back` + `get_write_ptr` + `push_back`; nothing drains it | **single-ended** | **self-loop** (writer bound both PRODUCER and CONSUMER, `:249-257`) + **borrowed-memory** (`:188`) + **conditional DFB** (`:178`) |
  | NDSharded | `ND_IN` / `ND_OUT` | all | reader P + compute C / compute P + writer C | plain 1:1 | none |

  Three things worth stating explicitly, because each is where a census usually goes wrong:

  1. **No hidden second writer anywhere.** Every kernel touching each DFB was scanned for a raw
     `get_write_ptr()` / `fifo_wr_ptr` co-fill by a non-FIFO-producer. There is none, and the shape
     that would coordinate one (a `reserve_done` / `write_done` semaphore pair) cannot exist here —
     the op declares no semaphores at all.
  2. **The same-source `KernelSpec` pairs are the *disjoint-node* variant, not the dual-instance
     work-split.** The BlockInterleaved factory emits two readers and two writers (full set,
     cliffrow set — `:297-304`) and up to four compute instances (`:306-353`), and the
     MultiCoreInterleaved factory emits two compute instances (`:192-207`). Every one covers a
     **disjoint** node set, expressed as separate `WorkUnitSpec::target_nodes`, and each block set
     binds its **own** DFB pair. Each node therefore sees exactly one instance of each role →
     ordinary 1:1, no assignment question. `buffer_set_for_core`
     (`data_movement/common/common.cpp:940`) resolves the per-core set, and the factory asserts the
     invariant twice — `block_size_row == set.block_tiles` (`:253-258`) and
     `single_sub_block_size_row_arg == set.block_tiles` (`:409-415`).
  3. **`SH_SHARDED_OUT` is single-ended, not dead.** Its one toucher is the writer, which fills it by
     write pointer and never drains it — the DFB *is* the output buffer
     (`borrowed_from = SH_OUTPUT`), so the data leaving is the point. Both writers that reach it do
     the same thing: `writer_unary_unpad_batch_rows_sharded.cpp:32` and
     `writer_unary_unpad_width_16_sharded.cpp:37`. The self-loop is implemented and the kernels are
     unchanged.

- **Offset base pointers: GREEN.** There is no fold to split out, and no construct that could carry
  one. A grep for `address()` across the entire op directory returns **zero** hits: every tensor base
  now reaches its kernel through a typed `TensorBinding`, resolved by the framework on every dispatch.
  Type 1 (raw offset arg), Type 2 (accessor-fed offset arg) and Type 4 (`ttnn::narrow`) are absent;
  Type 3 (`address_offset`) is the Appendix A row above and is absent.

  The one construct that superficially resembles a fold is **already the split-out form** the recipe
  describes as the fixed shape: the cross-shard writer receives a clean base via its binding and a
  **separate scalar** `col_byte_offset` runtime arg, which the kernel passes as
  `noc_async_write_sharded`'s `offset` argument (`writer_unary_unpad_cross_sharded.cpp:51`) — never
  added into the accessor's base.

  Cross-reference: the offset-base-pointer triage analysis
  (`analyses/2026-07-19_offset_base_pointers.md`, a dated prior) has no entry for this op, which
  agrees with the scan; the scan is what decides it.

- **TensorAccessor 3rd argument: GREEN — N/A on the current tree; both prior Class-2 sites are gone.**

  The 2026-09-08 audit found two sites passing a 3rd argument and classified both **Class 2**
  (redundant → drop). Both drops have been made:

  | Site | Was | Now |
  |---|---|---|
  | `device/kernels/dataflow/writer_unary_stick_layout_split_rows_multicore.cpp:34` | `TensorAccessor(dst_args, dst_addr, writer_page_size)` with a host-computed `writer_page_size` CTA (three branches, `…_multi_core_interleaved…` at HEAD `:110-118`) | `TensorAccessor(tensor::dst)` — the CTA is gone from the factory too |
  | `device/kernels/dataflow/writer_unary_unpad_cross_sharded.cpp:33` | `TensorAccessor(args, addr, shard_width_bytes)` | `TensorAccessor(tensor::dst)` |

  A fresh sweep of all 13 `TensorAccessor(` construction sites across the op's kernels and its
  donors finds **every one** now taking only the binding token. (The 13th,
  `writer_unary_stick_layout_wh_multicore.cpp:31`, is the orphaned legacy copy from
  [Open item A](#open-item-a--six-orphaned-legacy-kernels-are-not-deleted) — no factory binds it.)

  **Re-derivation of the Class-2 verdict for the site that actually needed one.** Two of the three
  configurations at the first site, and the second site, feed a **sharded** accessor, which uses the
  passed value verbatim with no realignment safety net — so "correct magnitude" is not sufficient
  there and the value must be shown to be alignment-aligned. It is:

  - *Interleaved output:* the value was `output.padded_shape()[-1] * element_size`, which for a
    ROW_MAJOR interleaved tensor is exactly `buffer->page_size()`.
  - *HEIGHT_SHARDED output:* the value was literally `dst_buffer->aligned_page_size()`.
  - *BLOCK/WIDTH_SHARDED output:* the value was `output.memory_config().shard_spec().shape[1] *
    element_size` — and `out_mem_config` is read from the **output tensor**, i.e. the spec
    `compute_output_specs` derived, where the shard width is rounded up to `tile_width` = 32
    (`device/untilize_with_unpadding_device_operation.cpp:437-450`). With a 2-byte minimum output
    element size (BFLOAT8_B is converted to BFLOAT16 on output, `untilize_output_dtype`), the page is
    at least 64 bytes and always a multiple of 64 — aligned under Blackhole DRAM (64), the strictest
    target, and *a fortiori* under L1 (16).

  So `page_size == aligned_page_size` in every reachable configuration and the drop is inert, as
  classified. **Stronger than last time, on one point:** `TensorAccessor`'s member is literally named
  `aligned_page_size` and is used verbatim as the stride
  (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:317,327`), with the binding supplying
  `TensorAccessorArgs<…>::AlignedPageSize` (`:97`). That is the value the allocator actually laid the
  buffer out with. So in the theoretical corner where the two *did* differ (a sub-64-byte sharded
  page), it is the **manual override that was wrong** and the drop is a fix, not a regression —
  which removes the only way this classification could have bitten. See *Questions* for the residual
  1-byte-dtype case.

---

## Port-work summary  *(mirrors the brief)*

- **Shared-kernel sunset — the outstanding item.** Delete the six orphaned legacy kernels in
  [Open item A](#open-item-a--six-orphaned-legacy-kernels-are-not-deleted), plus the two explicit
  build-file entries (`data_movement/CMakeLists.txt:88`, `ttnn/sources.cmake:173`) and the now-stale
  *"changes here likely belong there too"* notes in the surviving forks.

- **Tensor bindings** (per binding, per factory — classification varies by config in the Sharded
  factory). All **Case 1** or **clean**; **no Case 2 anywhere** — no kernel does hand-rolled NoC
  arithmetic on a tensor base. Every raw pointer in these kernels is a *DFB* pointer, not a tensor
  base.

  | Factory | Binding | Delivery | Kernel use | Case |
  |---|---|---|---|---|
  | SingleCore | input / output | `TensorBinding` | `TensorAccessor(tensor::src)` / `(tensor::dst)` | **Case 1** (done) |
  | MultiCoreInterleaved | input / output | `TensorBinding` (`:109-112`, `:133-136`) | `TensorAccessor` in both kernels | **Case 1** (done) |
  | BlockInterleaved | input / output | `TensorBinding` (`:177-180`, `:202-205`) | `TensorAccessor` in both forks | **Case 1** (done) |
  | Sharded | input | **borrowed-memory DFB** `SH_IN` (`:163`) | reader only `push_back`s — the DFB *is* the tensor access | **clean** (causal-link gate) |
  | Sharded | output — same-shard-type sharded | **borrowed-memory DFB** `SH_SHARDED_OUT` (`:188`) | writer fills it by `get_write_ptr` | **clean** (causal-link gate) |
  | Sharded | output — cross-shard-type / HEIGHT→interleaved / W-B→interleaved | `TensorBinding` (`:223`, `:288`, `:308`) | `TensorAccessor(tensor::dst)` | **Case 1** (done) |
  | NDSharded | output + input | `TensorBinding` (`:183-190`) | `TensorAccessor(tensor::dst)` / `(tensor::src)`, the latter for shard geometry only | **Case 1** (done) |

  **Urgency note, unchanged.** No base was ever delivered as `->address()` on this op — the legacy
  form was the `Buffer*`-binding shape, which the framework already patched on cache hits. This op
  never carried the silent-wrong stale-pointer hazard; the Case-1 conversion was routine.

- **TensorParameter relaxation:** `none` (sheet value supplied by the user). No relaxation applied,
  no analysis doc referenced.

- **TensorAccessor 3rd arg:** **none remaining** — both sites dropped along with the CTAs that fed
  them. No site was Class 1, so no `dynamic_tensor_shape` is set.

- **CB endpoints:** all implemented — self-loop on `SH_SHARDED_OUT`, `borrowed_from` on `SH_IN` and
  `SH_SHARDED_OUT`, conditional DFBs on `SH_SHARDED_OUT` and the BlockInterleaved cliffrow pair. No
  multi-binding advanced option anywhere; no dead DFB anywhere.

## Heads-ups  *(mirrors the brief)*

- **Shared `_metal2` forks now have a second consumer.** `reader_unary_interleaved_wh_multicore_metal2.cpp`,
  `writer_unary_stick_layout_wh_multicore_metal2.cpp` and `untilize_wh_metal2.cpp` are bound by
  **both** this op's BlockInterleaved factory and `data_movement/untilize`'s block factory
  (`untilize_multi_core_block_program_factory.cpp:60,63,65`). Their named-arg vocabularies are the
  forks' interface and are pinned — the restored factory matches them exactly (reader: CTAs
  `num_tiles_per_2d`/`third_dim`/`total_tiles_per_row`, RTAs
  `start_id`/`single_block_size_row_arg`/`single_block_size_col_arg`; writer: CTAs
  `total_num_rows`/`third_dim`/`tile_height`/`unpadded_X_size` + 7 RTAs; compute: CTAs
  `block_size_col`/`block_size_row`/`third_dim`). **Any rename now breaks `untilize`.** This is also
  why [Open item B](#open-item-b--an-unstaged-behavioural-edit-to-a-fork-another-op-binds) is a
  cross-op change.
- **`enough_space_height` moved under this branch's feet.** #56280 changed
  `untilize_with_unpadding.cpp:95-105,123` so the output CB estimate and the pending-L1 reservation
  are sized by the **output** dtype. For BFLOAT8_B inputs that is 2048 B/tile instead of 1088 B, and
  the reservation is no longer silently zero. Consequence: **BFLOAT8_B shapes that previously
  selected MultiCoreInterleaved now select BlockInterleaved.** The 5/5 verification recorded in
  `METAL2_PORT_REPORT.md` predates this change, so the restored block factory's BFLOAT8_B coverage
  should be re-run rather than inherited.
- **CB endpoints (multi-binding shapes to watch):** none. The census found no DFB with ≥3 touchers
  and none with two kernels locked to the same FIFO role, in any factory or configuration.
- **RTA / CRTA varargs:** two genuine variable-count blocks, both already ported onto the vararg
  mechanism. Detail below.
- **`make_block_plan` reads live L1 occupancy**, so it is valid only on a program-cache miss — which
  is the only time `create_program_artifacts` runs. `untilize`'s block factory states this in a
  comment (`untilize_multi_core_block_program_factory.cpp:79-80`); the restored factory here does the
  same thing without the comment. Pre-existing behaviour, faithfully preserved; worth a matching
  comment for the next reader.

### RTA varargs — detail

Both are **FYI-P**; Metal 2.0 supports RTA and CRTA varargs, so neither gates. Both are implemented.

1. **`writer_unary_stick_layout_split_rows_multicore.cpp:76-85` — RTA vararg (shape (a),
   variable-count loop).** The kernel loops `n_block_reps` times (itself a named RTA, a genuine
   runtime value) and pulls five values per group through a running `rt_arg_idx` advanced **inside**
   the loop body. Ported to `get_vararg(...)` with the index base reset to `0` (`:66-68`) — correct,
   because Metal 2.0 addresses varargs in a section separate from the named args, so the legacy
   offset of 4 no longer applies. Host side:
   `…_multi_core_interleaved_program_factory.cpp:225-269` builds the per-core payload and sets
   `advanced_options.num_runtime_varargs_per_node[core]` (`:268`), which is what a per-core-varying
   count requires.

   The four leading legacy args split correctly: `dst_addr` became the tensor binding, and
   `padded_X_size`, `start_stick_id`, `n_block_reps` are **named** RTAs — the fixed prefix precedes
   the vararg block, so none of them ride it.

2. **`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` — CRTA vararg (shape (a),
   CTA-bounded loop).** Two back-to-back loops read the output then the input padded shape, bounded
   by the CTA `tensor_rank`. A CTA-bounded count still varies across instantiations, so this is a
   vararg, not an unrolled name set. Ported as common runtime varargs with
   `advanced_options.num_common_runtime_varargs = 2 * tensor_rank`
   (`…_multi_core_nd_sharded_program_factory.cpp:211`), payload built at `:215-222`.

**Non-signal, correctly not converted:** `writer_unary_stick_layout_wh_multicore_metal2.cpp:72-77`
re-reads the same named args inside a `third_dim` loop — a fixed set of distinct fields read
repeatedly, which stays named. Likewise every other kernel in the op reads each argument a fixed
number of times.

**CTA varargs:** none. No kernel in the op or its donors calls `get_compile_time_arg_val` at a
varying index.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Exactly one function-call escape exists across all op-owned kernels,
and its shape is ✓ excellent. All other includes are `tt_metal/*` (`api/dataflow/*`, `api/tensor/*`,
`api/core_local_mem.h`, `experimental/kernel_args.h`) — donor class 1, no concern. There is **no** ⚠,
✗ or ⭐ entry, so the per-call detail section is omitted.

**Summary table — function-call escapes (one row per op kernel × donor file):**

| Op kernel | Donor file | Donor class | Functions called | Shape | Status |
|---|---|---|---|---|---|
| `writer_unary_stick_layout_split_rows_multicore.cpp:12` | `ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | `tt::data_movement::common::noc_async_write_sharded(Noc, uint32_t, AddrGenType, …)` | `Noc` by value (Device 2.0 native) + `TensorAccessor<DSpec>` by value (**Shape 1**) | **✓ excellent** |
| `writer_unary_unpad_cross_sharded.cpp:10` | same | 5 — in-family shared | same | same | **✓ excellent** |

The donor is on the Device 2.0 `Noc` object and takes the accessor by value, so the port constructs
`TensorAccessor(tensor::dst)` and passes it — no donor-side change, no fork of the header, no
`uint32_t sem_id` / `sem_addr` bridging problem, no old-style addr-gen (Shape 4). The remaining raw
`uint32_t l1_addr` parameter is a DFB read pointer, not a resource handle, so it is outside the shape
table.

**Borrowed kernel files (file-path kernel instantiation) — updated fork status.** The op instantiates
9 kernel files it does not own. `_metal2` fork status is a **locational** test;
`experimental/quasar/**` copies are excluded and do not count as forks. The co-borrower column is the
**sunset list**, not a must-port-together bundle and not authorization to convert anything in place.

| Kernel file the op binds | Owner | Fork status *(2026-09-08 → now)* | Other binders of the **legacy** copy (sunset list) |
|---|---|---|---|
| `eltwise/unary/…/reader_unary_interleaved_start_id_metal2.cpp` | eltwise/unary (cross-family) | existed → **reused** | 5 remaining: `reduction/topk`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `examples/example` (×2 factories), `examples/example_multiple_return` |
| `eltwise/unary/…/reader_unary_sharded_metal2.cpp` | eltwise/unary (cross-family) | existed → **reused** | 4 remaining: `experimental/slice_write` (×2), `untilize` ND-shard-identical factory, `sharded_partial/sharded_to_interleaved_partial` |
| `eltwise/unary/…/reader_unary_interleaved_wh_multicore_metal2.cpp` | eltwise/unary (cross-family) | **"this port creates the first" → created by #56280, reused here** | **0 — legacy orphaned; sunset now** (Open item A #1) |
| `data_movement/sharded/…/reader_unary_nd_sharded_blocks_metal2.cpp` | data_movement/sharded (in-family) | existed → reused | **0 — legacy orphaned; sunset now** (Open item A #5) |
| `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks_metal2.cpp` | `ttnn/cpp/ttnn/kernel/` (shared pool, class 3) | **created by this branch** | **0 — legacy orphaned; sunset now** (Open item A #6) |
| `untilize/…/compute/untilize_metal2.cpp` | data_movement/untilize (in-family) | existed → reused | **0 — legacy orphaned by the staged restore; sunset now** (Open item A #4). `fold` binds the fork, not the legacy copy. |
| `untilize/…/compute/untilize_wh_metal2.cpp` | data_movement/untilize (in-family) | **"this port creates the first" → created by #56280, reused here** | **0 — legacy orphaned; sunset now** (Open item A #3) |
| `untilize/…/compute/untilize_variable_num_blocks_metal2.cpp` | data_movement/untilize (in-family) | existed → reused | **1 remaining** — `untilize`'s ND-shard-identical factory. Legacy copy stays. |
| `ttnn/kernel/compute/eltwise_copy_metal2.cpp` | `ttnn/cpp/ttnn/kernel/` (shared pool, class 3) | existed → reused | 3 remaining: `data_movement/copy`, `sharded_partial/sharded_to_interleaved_partial`, and `sharded_partial/interleaved_to_sharded_partial` (which binds a different `sharded/…/eltwise_copy.cpp`) |

**The "lent" direction, swept this time.** The 2026-09-08 addendum admitted the op-owned-writer census
had only been run outward (kernels this op borrows) and never inward (kernels this op lends) — which
is how `writer_unary_stick_layout_wh_multicore.cpp` was missed. That sweep has now been run over all
8 op-owned writers: **`writer_unary_stick_layout_wh_multicore.cpp` was the only shared one**, and it
is handled (forked by #56280, legacy now orphaned). The other seven have this op as their sole
consumer, so converting them **in place** — which the port did — is correct and needs no fork:

`writer_unary_stick_layout_split_rows_multicore.cpp` · `…_nd_sharded.cpp` ·
`writer_unary_unpad_dims_split_rows.cpp` · `writer_unary_unpad_cross_sharded.cpp` ·
`writer_unary_unpad_batch_rows_sharded.cpp` · `writer_unary_unpad_width_16_sharded.cpp` ·
`writer_unary_unpad_sharded_to_interleaved.cpp`.

Also inventoried, though not borrowed: the compute kernels reach
`ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (donor class 2 — official shared kernel library).
Already fully `DataflowBuffer`-based; needs nothing from this port.

**A negative pointer, deliberately.** `ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/`
contains a copy of this op carrying a deliberately hacky shortcut port, and quasar copies appear as
apparent co-borrowers of six of the nine kernels above. That tree is **out of bounds** — not a
production port, its `_metal2` files are whole-op pre-port copies that do **not** count as forks to
reuse, and nothing in it is evidence about this op. It is excluded from every table here and from
every count in Open item A.

### Relaxation candidates

None. The op has no custom `compute_program_hash`, so there is no hash from which a candidate
relaxation could be mined.

### TTNN factory analysis

- **Current concept:** `MetalV2` in the working tree (`descriptor` on `origin/main`).
- **Op-owned tensors:** none.
- **MeshWorkload need:** none — no `create_workload_descriptor`, no `WorkloadDescriptor`.
- **Custom hash:** absent → no hash to preserve.
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent → base `ProgramSpecFactoryConcept`, no method to translate.
- **Pybind `create_descriptor`:** absent → nothing to delete, **no user-visible API change**.
- **Other risky pybind:** none. `untilize_with_unpadding_nanobind.cpp` exposes only the op function
  and its documented arguments.
- **Target concept:** **`ProgramSpecFactoryConcept`**, reached on all five factories.

## Misc anomalies  *(team-only, non-gating — route to the ops team; the port does not act on these)*

1. **Dead file — an orphaned shared-variables struct.** *(carried forward, unchanged)*
   `device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` defines
   `UntilizeWithUnpaddingMultiCoreSharedVariables`, and **nothing in the repository references either
   the header or the type** — only `data_movement/CMakeLists.txt:335` lists it in the header file
   set. A leftover from the pre-`ProgramDescriptor` cached-program era; it also drags in
   `<tt-metalium/host_api.hpp>` for no consumer. Now doubly dead — the concept it served is two
   migrations behind. Safe to delete on the ops track, along with its CMake line.

2. **Two dead compile-time args survive the port in the ND-sharded writer.** The factory emits 16
   named CTAs (`…_multi_core_nd_sharded_program_factory.cpp:191-207`) but the kernel reads 14 —
   `output_stick_size` (`:192`) and `input_single_tile_size` (`:199`) are never read
   (`writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp:23-37`). **The prior audit's reason
   for deferring this no longer applies:** it routed the removal to the ops team because dropping a
   positional CTA meant renumbering the kernel's reads. With named args there is nothing to
   renumber — deleting the two lines is a two-line, zero-risk change, and the factory was rewritten
   in this port anyway. Worth doing here rather than deferring.

3. **A dead compile-time arg on the W=16 sharded fast path.** *(carried forward)* The Sharded factory
   builds one CTA set for both same-shard-type writers (`…_multi_core_sharded_program_factory.cpp:259`)
   but `writer_unary_unpad_width_16_sharded.cpp` never reads `aligned_page_size`; it is live in the
   sibling `writer_unary_unpad_batch_rows_sharded.cpp`. Shared-vector convenience, not a bug.

4. **Unused debug include.** `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp` includes
   `api/debug/dprint.h` with zero `DPRINT` uses. *(The two borrowed kernels that previously shared
   this anomaly are no longer bound in their legacy form.)*

5. **A self-flagged uncertainty left in the code.** *(carried forward)*
   `…_multi_core_sharded_program_factory.cpp:78` reads *"I am not sure it is correct to ever use the
   shard_spec here"*, immediately above the `out_shard_spec` fallback that substitutes the **input's**
   shard spec when the output has none. The fallback is reachable — the interleaved-output branches
   take it — and `out_shard_spec` then drives `num_rows_block`, `block_row_size` and
   `last_block_row_size_unpadded`. Worth a deliberate answer from the op's owner. Not a portability
   issue; the port preserves the behaviour either way.

6. **`untilize_output_dtype` maps BFLOAT8_B but not BFLOAT4_B.** `data_movement/common/common.hpp:232`
   converts only `BFLOAT8_B → BFLOAT16`. A BFLOAT4_B tiled input would therefore be given a
   ROW_MAJOR BFLOAT4_B output spec, which block-float layout cannot express. Either BFLOAT4_B is
   unreachable here (in which case a `TT_FATAL` would say so) or the mapping is incomplete. Neither a
   gate nor port work — surfaced because the auditor reads every line. Also relevant to the element-size
   floor the 3rd-arg classification rests on; see *Questions* item 2.

7. **A legacy-only bug the sunset will bury.** The BACKWARDS walk in
   `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp:39-44` is genuinely broken (unsigned
   `dim > -third_dim` never enters the loop). It is dead code today, and Open item A deletes the file
   — which resolves it by removal for *this* kernel. Recorded so the removal is understood as
   deliberate rather than accidental. See [Open item B](#open-item-b--an-unstaged-behavioural-edit-to-a-fork-another-op-binds).

## Per-DeviceOperation attribution

Not applicable — the directory holds a single `DeviceOperation`. The per-factory and per-configuration
splits that *do* vary (the tensor-binding classification in the Sharded factory, the
`SH_SHARDED_OUT` disposition, the conditional cliffrow DFB pair) are attributed inline above.

## Questions for the user

1. **Readiness-sheet provenance — still open, and now with one more cell to re-read.** The two
   gate-bearing cells (`Is able to port? = yes`, `TensorParameter relaxation = none`, all five factory
   rows) are still taken from your 2026-09-08 launching message rather than fetched; no Drive
   connector is available in this session either. Every cross-checkable primary column was verified
   against the code and agrees. Two things you could close in one glance if you have the sheet open:
   the **factory-set match** (exactly five rows for this op, naming the five factories listed at the
   top), and whether the `Concept` cells are still `descriptor` — they should be, since the sheet
   tracks `main` and these commits are not on `main`, but confirming it removes the one ambiguity in
   the cross-check.

2. **Is a 1-byte output element size reachable?** The Class-2 verdict on the dropped 3rd argument
   rests on a 2-byte minimum output element size, which makes every BLOCK/WIDTH-sharded page a
   multiple of 64 bytes and therefore already aligned. I could not find a validation that *forces*
   that floor — `validate_on_program_cache_miss` constrains shard row alignment only on the
   ND-sharded input path (`device/untilize_with_unpadding_device_operation.cpp:376-384`), not on the
   interleaved-input → 2D-sharded-output path that the MultiCoreInterleaved writer serves. If a
   1-byte dtype (e.g. `UINT8`) can reach that path with a 32-wide shard on DRAM, the legacy manual
   value (32) and the framework's `aligned_page_size` (64) would differ. **This does not change the
   verdict** — the framework value is the one the allocator laid the buffer out with, so the drop
   would be a *fix* — but it would mean the legacy path was mis-addressing, which is worth knowing.
   Do you know whether UINT8 tiled input is supported by this op?

3. **Open item B is a decision, not a finding.** Do you want the BACKWARDS correction in this port's
   diff (cross-op, needs `untilize` owners on the review), or split out to the `eltwise/unary` kernel
   owners so the legacy original and the fork are fixed together? Either is defensible; leaving it
   unstaged is the one outcome that is not.

## Recipe notes

1. **The recipe has no shape for re-auditing an op whose port is already in the tree.** Its subjects
   are written as feasibility questions ("classify the site so the porter can drop it"), but the
   honest answer here is often "already done — verified". The recipe *does* anticipate the state in
   one line (`Concept == MetalV2` → *"report it as done, not blocked"*), but that line is scoped to a
   single sheet cell, not to the eleven subjects that follow. I ran each subject as a verification
   pass and said so; a short paragraph sanctioning that mode — and saying which subjects change
   meaning under it — would save the next auditor the same improvisation.

2. **"`Concept` mismatch ⇒ spreadsheet is broken" mis-fires on an in-flight port branch.** The
   cross-check rule says a primary-column conflict is one of only four triggers for a
   spreadsheet-broken GATE, and it rests on the auditor holding the code as independent evidence. But
   on a port branch the code is *ahead* of the sheet by construction: the sheet tracks `main`, and the
   whole point of the branch is that it diverges. A literal reading of the rule REDs every re-audit.
   The resolution is one sentence — *cross-check the `Concept` cell against the branch point, not the
   working tree* — and it belongs in the cross-check list, not in the auditor's judgement.

3. **The shared-kernel sunset has no home in the audit's output shape.** *Out-of-directory coupling*
   asks for the co-borrower set and says to label it a sunset list, and *Caution: Porting a shared
   kernel* owns the fork convention — but nothing asks the auditor the question that actually mattered
   here: **does this port take the last binder to zero, and if so, is the deletion in the diff?** That
   is a cheap, high-value check (one grep per borrowed kernel), it produces concrete PORT WORK, and it
   is invisible to every other subject. It is also exactly what was missed twice on this op. A line in
   Out-of-directory coupling — *"for each borrowed kernel, count the legacy copy's remaining binders
   after this port; zero means this port owns the deletion"* — would close it.

4. **The "lent" direction is still only implied.** The coupling subject is written entirely in terms
   of kernels the op *borrows*. Kernels the op *owns and lends* are what decides whether an in-place
   conversion is legal, and this op has eight of them with exactly one shared — a distinction the 2026-09-08
   audit got wrong precisely because the recipe never asks for the outward sweep. One sentence under
   *Borrowed kernel files* would fix it: *"also sweep the reverse direction — for each kernel the op
   owns, find every other op that binds it; an in-place Metal 2.0 conversion is legal only when the
   count is zero."*

5. **Prior recipe notes 2 and 3 (2026-09-08) still stand** and are not repeated here: the Class-2 row
   needs an explicit *"a sharded accessor passed `buffer->page_size()` is Class 2 only if that value
   is already alignment-aligned — show the alignment"* sub-case, and the CB-endpoint faces need the
   *"first, check whether the instances' core ranges intersect"* line ahead of face (c). Both
   false-fired again on this op in exactly the same way.
