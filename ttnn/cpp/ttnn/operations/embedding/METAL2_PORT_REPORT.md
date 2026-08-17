# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/embedding`

## Outcome

**`PORTED`** — `EmbeddingsRMProgramFactory` and `EmbeddingsTilizedIndicesProgramFactory` converted to
`ProgramSpecFactoryConcept`, together with the kernels they bind. The confirmed test set is green and
matches the pre-port baseline exactly (318 passed / 2 skipped, both before and after).

`EmbeddingsFusedProgramFactory` is **not ported** — the audit blocked it on a Type-2
offset-base-pointer wall (`embeddings_tilize.cpp:36`) and the brief scoped it out. It stays on
`ProgramDescriptorFactoryConcept` in the same `program_factory_t` variant; the framework dispatches
per-factory, so the op builds and runs with a mixed set of factory concepts.

## Provenance

- **Recipe docs (this port):** `0efcf9f88ae 2026-08-17 docs(metal_2.0): CTA varargs are in, and five columns read present-tense`
- **Audit docs (inherited):** `0efcf9f88ae 2026-08-17 docs(metal_2.0): CTA varargs are in, and five columns read present-tense`

## Verification performed

| check | result |
|---|---|
| Build (`./build_metal.sh -e --enable-fake-kernels-target --build-tests`) | SUCCESS, 0 errors |
| Confirmed test set, pre-port baseline | 318 passed, 2 skipped |
| Confirmed test set, post-port | 318 passed, 2 skipped — same set, no new failures |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` per-file census vs merge-base | no delta, in the op directory *and* in `ttnn/cpp/ttnn/kernel/` |
| `hw_config` values vs legacy | 2 `ReaderConfigDescriptor{}` → 2 reader helpers; 3 `WriterConfigDescriptor{}` → 3 writer helpers. No custom triples, no compute kernels |
| `opt_level` | legacy set none on either factory; both are DM-only, so legacy and Metal 2.0 both resolve to `O2`. Nothing set |
| Anti-pattern sweeps | 22 files in the op directory + 1 forked shared kernel scanned; details below |
| Ephemeral-doc citation sweep | 11 changed/new `.cpp`/`.hpp` files scanned, **0** `.md` citations |
| Untouched-scope check | `embeddings_fused_program_factory.{cpp,hpp}`, `embeddings_tilize.cpp`, `tilize_chunked.cpp`, `embedding_device_operation.{cpp,hpp}`, `embedding_nanobind.cpp`, `embedding.cpp` — byte-identical to merge-base |

Anti-pattern sweep results, as *hits / files scanned* (22 op-directory files + the 1 forked shared
kernel; every sweep ran over a non-zero denominator):

- buffer-address RTAs (`buffer()->address()`, `emplace_runtime_args`, bare `Buffer*`) in the two ported
  factories: **0**
- magic CB indices (`CBIndex::c_`, `*cb_index`) in the two ported factories: **0**
- `TensorAccessorArgs` in the ported kernels: **0**
- `cb`-shaped names across the ported surface: **0** after fixing 3 real leftovers (below)
- `.id` extraction on a `dfb::` handle: **0**
- `allow_instance_multi_binding`: **0** (no DFB in either factory reaches ≥3 touchers or two kernels
  locked to the same FIFO role)
- positional CTAs / RTAs / any vararg mechanism across the ported surface: **0**
- `CircularBuffer` / `CBDescriptor` / `CBFormatDescriptor` / `circular_buffer.h` / `.cbs` across the
  ported surface: **0**

The `cb`-name sweep found three genuine leftovers that the first draft carried, all renamed in one
pass: `input_cb_data_format` → `input_data_format`, `weights_cb_data_format` → `weights_data_format`,
`out_cb_size` → `out_dfb_total_size`.

### Extra verification: the branches no test reaches

**Every test in the confirmed set runs `EmbeddingsType::GENERIC`** — three of them say so explicitly,
the rest pass no `padding_idx`. And no test's `hidden_embedding_dim` comes near the 1 MB weight-row
threshold that selects the chunked writer (the largest is 16384, giving a 32 KB row). So the confirmed
set leaves these converted paths unexercised:

- `PADDED` / `BINARY` on both factories: the conditional `weight_cache` DFB, its self-loop bindings,
  the `pad_token` named RTA, and the whole forked `prepare_local_cache`.
- `use_chunked` on the RM factory: `embeddings_rm_writer_chunked.cpp` and its three CTAs.

Shipping those unverified would have made the "no behavior change" claim untestable, so I ran a
**pre/post bit-equality harness** over six configurations reaching all of them: RM PADDED, RM BINARY,
TILE-indices PADDED, TILE-indices BINARY, TILE-indices narrow-row (`ONLY_ONE_FACE_COLUMN`), and RM
chunked (`vocab=4, hidden=600000` → a 1.2 MB aligned weight row, `num_chunks > 1`). The same script
ran against the pre-port build and the post-port build; **all six outputs are bit-identical**.

The harness is a local scratch script, not a repo test — it is deliberately not added to the tree (see
*Open items* for the coverage note that belongs to the ops team).

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` on both in-scope factories, as the audit chose. No re-decision, nothing
surfaced back to the invoker on this axis. Neither factory has an `override_runtime_arguments`, so the
framework refreshes the tensor bindings on a cache hit and each factory writes exactly one method.

### Device-op-class edits

- **Pybind entry points removed:** none. `embedding_nanobind.cpp` exposes only `ttnn::embedding` and
  the `EmbeddingsType` enum — no `create_descriptor` was ever pybound, and no factory carried a
  pybind-hook-only parameter. The device-operation class is byte-identical to the merge-base.
- **Custom `compute_program_hash`:** none — the op uses the default reflection-based hash. Nothing to
  preserve or leave alone.

### Open items

- **Relaxation candidates:** none applied, and none obviously available. Both readers bake specific
  page sizes and row lengths into compile-time args, so they are not shape-agnostic in the way that
  would tolerate `dynamic_tensor_shape`.
- **The output `TensorParameter` is borrow-only in one configuration.** In the RM factory's
  height-sharded config no kernel binds `output`; the parameter exists solely because the output
  staging DFB names it in `borrowed_from`. The migration guide's validator note covers this case
  explicitly ("a `TensorParameter` named by a `DataflowBufferSpec::borrowed_from` counts as used even
  when no kernel binds it"), and it behaved as documented.

## Handoff points

### 1. `EmbeddingsTilizedIndicesProgramFactory` reads its pad token from the wrong runtime-arg slot — ops team

**Preserved, not fixed**, per the brief. This is the port's most important finding, and the port
reproduces it exactly.

- `embedding_ind_tilized.cpp` (pre-port `:42`) asks the shared cache setup for its pad token at
  runtime-arg slot **6**.
- The factory puts `col_offset % FACE_HEIGHT` in slot 6 (pre-port
  `embeddings_tilized_indices_program_factory.cpp:215`) and the real pad token in slot **7**
  (pre-port `:217`). Slot 7 is never read.
- The other two readers (`embeddings.cpp`, `embeddings_tilize.cpp`) put the pad token at slot 6 and are
  correct, so this is a single-factory off-by-one, presumably from the extra `starting_index` argument
  this factory carries.

**It is a numerics defect, not just a wasted argument** — worth stating plainly, because the audit's
wording ("indices that happen to equal it get substituted with the wrong weight row") understates it.
Measured on the **pre-port** build, `vocab=512, hidden=768, batch=2, sentence=64`, indices drawn from
`[0,16)`, `padding_idx=3`:

- row-major indices with `padding_idx=3` match the PyTorch reference exactly;
- TILE-layout indices with the same `padding_idx` **do not** — 7 of 128 output rows differ, all of them
  rows whose index token is `0`, and the data they receive **matches no row of the weight table** (not
  a wrong-but-real row: it reads as uninitialized). My guess is that this is the local cache being
  consulted on cores where the face-column index happens to equal an incoming token while the cached
  row was never populated for that value, but I did not chase the mechanism — diagnosing it is ops-team
  work and the port must not change the kernel's logic to find out.

**How the port preserves it.** The pad token now reaches `prepare_local_cache` as a *value* instead of
an argument index, so each reader passes what its own factory actually supplies:
`embeddings.cpp` passes `get_arg(args::pad_token)`; `embedding_ind_tilized.cpp` passes its
`starting_index` runtime arg, which is the same value legacy slot 6 carried. Legacy slot 7 is dropped
as dead plumbing (never read, so a zero-functional-change drop). The pre/post harness confirms the
TILE-indices PADDED output is bit-identical across the port.

**What the fix looks like**, when the ops team takes it: give the reader its own `pad_token` named
runtime arg fed from `pad_token.value()`, and pass that instead of `starting_index`. Under named
arguments this is a two-line change with no positional-slot bookkeeping — which is worth noting,
because under the legacy positional API the same fix had to renumber slots.

### 2. The shared-kernel fork needed a build-file edit outside the two sanctioned ones — docs / infra

Creating a `_metal2` fork of a kernel in the shared pool `ttnn/cpp/ttnn/kernel/` required a **third**
edit beyond the Caution's carve-out (add the fork, add the pointer comment): a line in
`ttnn/sources.cmake`'s `TTNN_CORE_JIT_API_HEADERS`, because that pool is enumerated explicitly rather
than globbed. Precedent for the same edit already exists on `main`
(`cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp`, line 164), so I followed it, but the
recipe's guidance says the opposite. See *Friction* for the doc fix.

No kernel-lib gaps, no boundary-rule assumption violations: nothing in this port needed to pass a
`sem::` or `tensor::` handle to a call site outside the op directory, and the op declares no
semaphores at all.

## Successes

- **[Conditional / optional DFB bindings]** fired exactly as written. The `weight_cache` DFB is bound
  only under `PADDED` / `BINARY`, and both readers reference `dfb::local_cache` from a point the
  compiler parses on every build (the `prepare_local_cache` call, unconditional in legacy). Without the
  pattern's preprocessor gate this is a name-lookup failure on the GENERIC build, which is the majority
  of the test set — so the warning saved a guaranteed break, not a hypothetical one. The pattern's
  "don't bind unconditionally" note also stopped me from taking the L1-wasting shortcut. Applied at
  `embeddings.cpp:37-41` and `embedding_ind_tilized.cpp:41-45`.
- **[Porting a shared kernel] rung 1, run locationally.** The Caution insists on `ls`-ing the
  original's directory rather than a tree-wide `_metal2` filename grep. That distinction mattered here:
  a tree-wide grep surfaces `_metal2` files under `experimental/quasar/**`, which are not forks of
  anything and are out of bounds. Running the check as specified returned the right answer (no fork
  beside either original → rung 2 twice).
- **The brief's "confirm rather than swap blind" heads-up on page-size metadata paid off.** It was
  tempting to have the forked stick writer take its write size from `dfb.get_entry_size()` instead of a
  runtime arg. In the RM non-chunked config the DFB's `entry_size` is `rounded_weight_page_size`
  (allocator-aligned) while the correct write size is the unaligned `output_page_size` — so that swap
  would have written aligned padding into the output tensor. `stick_size` stays a runtime arg, and the
  reason is now stated at the declaration in the fork.
- **"Print the denominator" caught a sweep that was scanning the wrong thing.** My ephemeral-doc sweep
  first ran with a stale working directory left by an earlier `cd`, so `git ls-files --others` resolved
  relative to a subdirectory: the file list was missing the new fork and nine paths failed to open,
  yet the check still printed `0 hits`. Printing the list is what exposed it. This is precisely the
  silent-false-GREEN shape the section describes, encountered live.
- **The [Two-toucher → assign 1P+1C] "re-derive, don't transcribe" instruction** was cheap to follow
  and the census agreed with the brief on every `(DFB, config)` row. No disagreement to report — but
  re-deriving is what makes that statement worth anything.

## Friction

### Gaps

- **Nothing warns you to capture the pre-port test baseline *before* editing a kernel.** Kernels are
  JIT-compiled from the source tree at dispatch, not from the built libraries, so the moment the first
  kernel source is converted a "baseline" run is already measuring the port. My first baseline attempt
  died mid-session with `'args' has not been declared` from a half-converted reader, and recovering
  meant `git stash -u`, a `tt-smi -r`, a clean baseline run, and `git stash pop`. The recipe's
  [Run tests] section notes that a selected-but-unconverted kernel can take down a whole pytest
  session, which is the same mechanism seen from the other side, but it does not draw the ordering
  conclusion. **Suggested fix:** one line under *Before you begin* — capture and record the confirmed
  test set's pre-port result before touching any kernel source, because kernel sources are read at
  dispatch time.
- **The shared-kernel Caution's build-system claim is too broad.** "No build-system change is needed
  for the new file. Op kernel sources are installed by per-family `file(GLOB_RECURSE kernels …)`
  patterns" holds for an op's own directory (`embedding/CMakeLists.txt` globs
  `device/kernels/*.cpp`/`*.hpp`, so `embeddings_common_metal2.hpp` needed nothing) but **not** for the
  shared pool `ttnn/cpp/ttnn/kernel/`, which `ttnn/sources.cmake` enumerates file by file.
  **Suggested fix:** qualify the sentence, and name `ttnn/cpp/ttnn/kernel/` as the known exception
  where one line is added to `TTNN_CORE_JIT_API_HEADERS`, citing the `generate_bcast_scalar_metal2.hpp`
  precedent.
- **A dead *runtime* arg has no prescribed disposition.** The construct step tells you to drop a dead
  CB "and any dead CTA carrying its index," and the plan template's *Dropped Plumbing* table covers
  named categories, but a runtime arg the kernel never reads is not addressed. This port has three
  (legacy slot 7 in the tilized reader, plus the never-read `output_page_size` CTA on the stick writer
  in both factories). I dropped them by analogy with the dead-CTA rule and recorded each with
  `file:line`. **Suggested fix:** state the rule once, generally: an argument of any dispatch kind that
  no selected kernel source reads is dropped, and recorded in the report.
- **`DataflowBufferSpec` has no `total_size`, and the mapping from the legacy pair is left implicit.**
  A legacy `CBDescriptor` carries `total_size` + `page_size`; the DFB carries `entry_size` +
  `num_entries`. The migration guide's `DataflowBufferSpec` section gives an example where the legacy
  code happened to be written as `num_pages * page_size`, so the mapping looks obvious — but ops like
  this one compute `total_size` directly (`output.buffer()->aligned_size_per_bank()`), and then
  `num_entries = total_size / page_size` is a *derivation* whose faithfulness depends on divisibility.
  I used integer division because that is how the legacy circular buffer derives its own page count,
  and noted the reasoning at the declaration. **Suggested fix:** one sentence in the guide, stating
  `num_entries = legacy total_size / legacy page_size` and flagging the non-divisible case as worth a
  look.

### Confusion

- **The brief framed an in-place-impossible shared *header* as a stop, and it is not one.** The brief
  says of `embeddings_common.hpp`: "If the CB→DFB conversion forces a signature or type change there,
  you cannot make it in place without touching the blocked kernel — treat that as an
  assumption-violation stop and raise it." The conversion *does* force exactly that (the header
  constructs a `CircularBuffer` from a buffer index and reads the pad token positionally), so read
  literally the port stops here. But the shared-kernel Caution's **intra-op** rung resolves it cleanly:
  fork the header beside the original inside the op's own directory, which needs no scope exception at
  all. **Suggested fix:** have the audit route an in-op shared *header* to the shared-kernel Caution
  rather than to the stop off-ramp; the "assumption violation" language should be reserved for the
  out-of-op call-site case it was written for.
- **The audit's RED/brief-scope split is easy to misread on arrival.** The invoker's instruction said
  the audit "cleared GREEN"; the audit itself reads *"RED at op level; subset … is clear"*, with a
  scoped brief. Reconciling those took a careful read of the audit's *Result* section plus its own
  recipe-note #2, which flags this same reconciliation problem from the auditor's side. The scoped
  brief is unambiguous once found — the status table's `Overall` row is what carries the surprise.

## Open items for downstream

### Shared kernel touches

| kernel path | rung taken | remaining unmigrated consumers |
|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | **created the fork** → `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp`; pointer comment added to the original (and one line to `ttnn/sources.cmake`) | `data_movement/concat` (`concat_program_factory.cpp:234`, row-major path), `data_movement/copy` (`copy_same_memory_config_program_factory.cpp:39`, row-major interleaved path) |
| `ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp` | **created the fork** → `…/embeddings_common_metal2.hpp`; pointer comment added to the original | `embeddings_tilize.cpp`, i.e. this op's own blocked `EmbeddingsFusedProgramFactory` |

The fork's interface, for whoever reaches it next: `dfb::out0`, `tensor::dst`, and named runtime args
`stick_size`, `num_sticks`, `start_id`; it still honours the `BACKWARDS` define. The legacy stick
writer retires when `concat` and `copy` have both migrated. `data_movement/slice` has its own
near-identically-named file and is **not** a consumer of either copy.

`embeddings_rm_writer_chunked.cpp` is bound only by this op's RM factory, so it was converted in
place — no fork, nothing to coordinate.

### Test coverage the verification step surfaced

- **No test in the confirmed set exercises `EmbeddingsType::PADDED` or `BINARY` on any factory.** That
  is why the pad-token slot defect above went unnoticed, and it means the local weight cache, the
  conditional DFB's live branch, and `prepare_local_cache` are all untested in CI. The nearest thing is
  the sweep `tests/sweep_framework/sweeps/data_movement/embedding/embedding_pytorch2.py`, which draws
  indices with `torch.randint` over 500-250k-row vocabularies, so it hits a 0-15 pad value only by
  chance.
- **No test exercises the RM chunked writer.** It needs an aligned weight row above 1 MB (hidden
  dimension above ~524288); the largest in the suite is 16384. Both gaps are covered for *this change*
  by the pre/post harness described above, but nothing in the tree keeps them covered.
- Adding either case is out of scope for a port, and both are cheap to write — the harness in this
  port's scratch work is a starting point.

### Per-op carry-over

- `EmbeddingsFusedProgramFactory` is the remaining factory. It comes back for a cheap re-audit once the
  ops team resolves how it addresses a column slice of the weights table without pre-offsetting the
  accessor base. When it ports, it will want the same `embeddings_common_metal2.hpp` fork, at which
  point the legacy header can be retired and the fork can take its name.
- The `stick_size` runtime arg on the shared writer carries the **same value on every node** in both
  factories (`output_page_size`). That makes it a common-runtime-arg candidate, which would cut
  per-node dispatch payload. Converting it changes dispatch semantics, so it is deliberately **not**
  part of this port; it belongs to the same later pass as a name-first restructure of the run-args
  loops (both factories still build their run args node-first through `AddRuntimeArgsForNode`).

### Non-gating anomalies left alone

Recorded so they are not lost, all confirmed still present and all deliberately untouched:

- **Dead CTA in the legacy stick writer, affecting its other consumers.** The legacy
  `writer_unary_stick_layout_interleaved_start_id.cpp` reads CTA slot 0 then hardcodes
  `TensorAccessorArgs<2>()`, so slot 1 is never read, yet all consumers dutifully fill it with a page
  size they *also* pass as a runtime arg. The Metal 2.0 fork drops the slot (no positional offsets to
  pad), but the legacy file still forces it on `concat` and `copy`. Routes to whoever owns
  `ttnn/cpp/ttnn/kernel/`.
- **Stale `// Grayskull Device Setup` banner** at `embeddings_rm_program_factory.cpp:34`, `embeddings_tilized_indices_program_factory.cpp:34`, and
  `embeddings_fused_program_factory.cpp:35`, over code that is not Grayskull-specific on an
  architecture that is no longer a target. Preserved verbatim.
- **Unused `api/debug/dprint.h` include** at `embedding_ind_tilized.cpp:12` — the file contains no
  `DPRINT`. Preserved; removing it is an unrelated cleanup.
- **`prepare_local_cache` reserves without committing**, by design (`embeddings_common_metal2.hpp`):
  the weight cache has no consumer to drain it. Likewise each reader reserves its index scratch page
  once and commits it only at the very end to leave the buffer balanced. Both are why the self-loop
  bindings are the right expression, and neither was "balanced" during the port.
- **A `GENERIC`-config argument with no consumer, now structurally impossible.** Pre-port, all three
  readers read the `c_2` buffer index unconditionally even under `GENERIC`, where the host allocated no
  such buffer — harmless as a `uint32_t`, but the audit flagged it as the kind of argument that stops
  being harmless once a binding is attached. In the ported readers the handle only exists on the builds
  where the host binds it, so the mismatch cannot recur.
