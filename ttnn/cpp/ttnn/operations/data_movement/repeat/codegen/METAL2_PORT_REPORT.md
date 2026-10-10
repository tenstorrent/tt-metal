# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/data_movement/repeat/codegen`

## Outcome

**PORTED (code-complete, verification pending on hardware)** — `RepeatCodegenProgramFactory` (the op's only
factory, all three internal branches: TILE / RM last-dim / RM higher-dim) converted from `create_descriptor` to
`create_program_artifacts` on `ProgramSpecFactoryConcept`, together with the five kernel entry points it binds
(three own kernels converted in place, two shared-pool kernels forked as `_metal2` siblings). **No build and no test
was run in this session** by the invoker's instruction — the orchestrator owns the N150 and runs the verification
listed under [Verification handed to the orchestrator](#verification-handed-to-the-orchestrator). The recipe's
anti-pattern self-audit was run and is clean (see [Self-audit](#anti-pattern-self-audit)).

## Provenance

- **Recipe docs (this port):** `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
  prints **nothing** in this worktree — the docs are an untracked overlay (not on the checked-out branch), so the
  version cannot be pinned from git here. The overlay was copied from the same revision the audit ran against
  (line below).
- **Audit docs (inherited):** `8c5559389da 2026-09-22 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Base:** `origin/main` @ `3a0efe4bd07` (branch `vsureshTT/metal2-repeat-codegen`).

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, exactly as the audit chose. `RepeatCodegenProgramFactory::create_program_artifacts(const
RepeatCodegenParams&, const RepeatCodegenInputs&, Tensor&)` returns `ProgramArtifacts{.spec, .run_params}`; no
`override_runtime_arguments`, no op-owned tensors. `program_factory_t = std::variant<RepeatCodegenProgramFactory>`
and `select_program_factory` are unchanged.

### Device-op-class edits
- Pybind entry points removed: **none** (no pybound `create_descriptor` existed).
- Custom `compute_program_hash`: **none** — default reflection hash, untouched.
- `repeat_codegen_device_operation.{hpp,cpp}` and `repeat_codegen_supported.{hpp,cpp}`: **byte-identical** to
  `origin/main`. The header `repeat_codegen_program_factory.hpp` changed only in the factory method signature and the
  `program_descriptors.hpp` → `ttnn/metal_v2_artifacts.hpp` include; `kRepeatCbDepth`, `RepeatCodegenParams` and
  `RepeatCodegenInputs` are unchanged (the header is also consumed by `repeat_codegen_supported.cpp`).

### Open items (factory layer)
- Relaxation candidates: none observed. `RepeatCodegenParams` hashes `stick_size` even on TILE (always 0 there) —
  pre-existing, documented in the struct, no behavioural effect; not a relaxation matter.
- The three TILE-reader sequencer params (`num_repeats`, `lower_pages`, `rep_dim_pages`) are per-node RTAs with the
  same value on every node — a CRTA candidate for a later, separate cleanup (dispatch-semantics change, deliberately
  not made here).

## Handoff points

none. No capitulation, no boundary-rule violation (no out-of-op call site needed a `sem::` / `tensor::` handle), no
kernel-lib gap, no framework gap hit, no pybind surface removed. The only writes outside the op directory are the two
sanctioned rung-2 fork edits (new `_metal2.cpp` + pointer comment) in `data_movement/common/kernels/codegen/`.

## Successes

- **[Caution: Porting a shared kernel](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  — census disambiguation.** The filename grep for `writer_interleaved.cpp` returned seven "consumers"
  (`transformer/sdpa/*`, `experimental/slice_write/*`, `reduction/sampling/*`, `untilize/codegen/*`); the "check the
  bound *path*, not the filename" instruction reduced that to the one real binder (`repeat_interleave/codegen`) and
  correctly excluded `untilize/codegen/kernels/reader_tile_interleaved_unified.cpp` as a same-named private copy.
  Also its "no build-system change is needed for the new file" note was verified true here:
  `ttnn/cpp/ttnn/operations/data_movement/CMakeLists.txt:22` already globs `common/kernels/codegen/*.cpp`.
- **Kernel-side whitelist rule 3 / §Dropped Plumbing — page-size 3rd argument.** The brief classified all three
  `TensorAccessor(..., page_size)` sites as Class 2 and they dropped cleanly; the recipe's wording ("a `page_size`
  value emitted **solely** to feed a `TensorAccessor`'s third constructor argument") is what flagged that
  `src_page_pitch` is *not* such a CTA — it also bounds the transfer size — so it was kept rather than dropped
  (`reader_tile_interleaved_unified_metal2.cpp:112-119`, factory `:141-145`).
- **CB→DFB whitelist §B/§C units check.** The brief's "confirm units before swapping" for
  `get_local_cb_interface(...).fifo_page_size << cb_addr_shift` and for `get_write_ptr()` paid off in the negative
  sense — both were confirmed from `internal/tt-1xx/dataflow_buffer.inl:42-48,118-124` (`get_entry_size()` applies
  the same shift; `get_write_ptr()` returns `fifo_wr_ptr` + 0 off-Quasar), so the L1→L1 self-copy in
  `reader_repeat_last_dim_rm.cpp:59,84-90` needed no change beyond the type swap.
- **§Construct — `AddRuntimeArgsForNode`.** Kept the legacy `for (core : cores_in_order)` loop and its `start += n`
  accumulation verbatim in all three branches; no loop inversion, no transposition risk.
- **§Hardware configuration.** Legacy `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` resolve to the role
  defaults, so the arch-agnostic `ttnn::create_reader/writer_datamovement_config(arch)` helpers apply on all six
  `KernelSpec`s with no custom Gen1 config and no Gen2 branch of my own.
- **§Compiler options.** `grep -n opt_level` on the legacy factory: no hits; both kernels are DM → legacy O2 ==
  Metal 2.0 default O2; no compute kernel, so no O3 line owed. Recorded rather than eyeballed.

## Friction

### Gaps
- **The recipe's build/test steps assume the porter owns a device.** This port ran under an orchestrator that
  serialises device access; the recipe has no "hand verification to the invoker" shape, so the commands are written
  out below under a heading the orchestrator asked for. A one-paragraph "if you cannot build/test, do X" note in
  §Verification would make that outcome first-class (and say whether `PORTED` may be claimed before tests run — I
  qualified it above instead).
- **Multi-sequencer shared kernel + generated `args::` names.** Neither the recipe nor the shared-kernel Caution
  covers a shared kernel whose *dead* `if constexpr` branches read arguments no current binder declares. Because
  `args::`/`dfb::`/`tensor::` are generated from the binding factory's schema and `if constexpr` still performs name
  lookup on the discarded branch, the fork cannot carry those branches unguarded. I dropped them (only `SEQ_REPEAT`
  and `SEQ_REPEAT_INTERLEAVE` survive, with a `static_assert` and a header comment explaining how to add one behind a
  `#ifdef`). A catalog note — "a pluggable-strategy shared kernel: carry only the strategies with binders, gate the
  rest with `compiler_options.defines`" — would save the next porter of `untilize/codegen` (same reader shape) the
  derivation.
- **Provenance command yields nothing for an untracked docs overlay.** The recipe anticipates "prints nothing" but
  only as "not a tracked doc-branch checkout"; the overlay workflow (docs copied in, never committed) is common
  enough in this porting effort to deserve its own sentence.

### Confusion
- **"Named argument names match the variables they were assigned to" vs. the existing lowercase named CTA.** The
  legacy TILE reader already read a lowercase named CTA `batch` into `constexpr uint32_t BATCH`. Applying the rule
  literally to the RM kernels would have produced `args::NUM_REPEATS`, `args::BATCH` beside `args::batch` on the
  shared fork. I standardised on lowercase snake_case arg names and left every kernel-side identifier unchanged
  (`NUM_REPEATS = get_arg(args::num_repeats)`), recording the table in the plan. A one-line convention ("arg names
  are lowercase; the kernel's constexpr identifier may stay uppercase") would remove the ambiguity.
- **Shared reader's 3rd-argument drop changes what `src_page_pitch` can influence.** Legacy: a non-zero override set
  *both* the accessor's addressing pitch and the transfer size. Fork: the binding fixes the accessor's pitch, so the
  override only bounds the transfer size. For every current binder (`0`) this is byte-identical; a hypothetical future
  binder passing a non-zero pitch would see different addressing. Documented in the fork header
  (`reader_tile_interleaved_unified_metal2.cpp:41-48`). The 3rd-argument subject might mention that dropping the
  argument can silently narrow the meaning of a *surviving* override CTA.

## Anti-pattern self-audit

Denominators: 9 code files under the op directory, 2 forks. Results (`scratchpad/selfaudit.txt` reproduced):

| check | result |
|---|---|
| buffer address / `emplace_runtime_args` / bare `Buffer*` in run-args | 0 — the only `Buffer*` uses are the two declarations feeding `aligned_page_size()` |
| magic CB indices / positional CTAs / `cb_id` / `buffer_index` | 0 |
| `TensorAccessorArgs<N>()` in ported kernels | 0 in code (2 hits are the fork header comments describing what was removed) |
| `cb` in a DFB name / `CircularBuffer` / `CBDescriptor` / `circular_buffer.h` | 0 in the factory, the forks and the three own kernels. Hits remain in `repeat_codegen_supported.cpp:38-85` (`rm_cb_fits_in_l1`, `cb_stick_elems`, comments) — off-limits device-op host code, untouched by design, routed to Open items. `kRepeatCbDepth` (header) is a depth constant the sweep does not match; also routed. |
| `.id` extraction at LLK sites | 0 (no LLK / kernel-lib call sites in these kernels) |
| conditional DFB bindings | none exist |
| CTA→RTA demotion | none (no compute kernel; the per-core count was already an RTA) |
| `allow_instance_multi_binding` | 0 |
| positional `compile_time_args` | 0 |
| varargs | 0 |
| `tt_metal/` in diff, `METAL2_CHECKS_FORCED` / `DO NOT COMMIT` | none |
| `.md` cited from code (9 files scanned) | 0 |
| TT_FATAL census (`3a0efe4bd07` vs working tree) | no output — both factory guards kept |
| `hw_config` | 6/6 `KernelSpec`s use the role-default helpers matching the legacy `Reader/WriterConfigDescriptor{}` |
| `opt_level` | no lines; correct — DM only, legacy unset (O2) |

## Verification handed to the orchestrator

Everything below was **not** run in this session. Run from the worktree root with the venv active.

1. **Baseline first, on the pre-port tree.** Kernels are JIT'd from the working tree, and this branch already
   carries the kernel edits, so the baseline must come from a checkout of the merge base:
   ```bash
   git -C /localdev/vsuresh/wt-m2-repeat-codegen merge-base origin/main HEAD    # expect 3a0efe4bd07
   # in a pristine checkout/worktree of 3a0efe4bd07 (or `git stash` here), same build:
   export TT_METAL_WATCHER=10
   pytest tests/ttnn/unit_tests/operations/data_movement/test_repeat.py -x -v
   pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_repeat_codegen_routing.py -x -v
   rm -rf ~/.cache/tt-metal-cache      # purge JIT cache between baseline and post-port runs
   ```
2. **Force and prove the Metal 2.0 legality checks** (recipe §Ensure the Metal 2.0 host-side legality checks are
   enabled) — working-tree scaffolding only, never committed:
   ```bash
   grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp
   # add `skip_validation = false;  // TEMP: force Metal 2.0 legality checks on. DO NOT COMMIT.` as the first
   # statement of every function listed, plus `log_warning(tt::LogMetal, "METAL2_CHECKS_FORCED");` once per file
   # (not in UpdateProgramRunArgs); rebuild; expect BOTH markers in the test log.
   ```
3. **Build** (background + log-reader subagent per the recipe):
   ```bash
   ./build_metal.sh --build-tests > /tmp/metal2_repeat_codegen_build.log 2>&1
   ```
4. **Post-port tests**, Watcher on, JIT cache purged:
   ```bash
   export TT_METAL_WATCHER=10
   pytest tests/ttnn/unit_tests/operations/data_movement/test_repeat.py -x -v \
       > /tmp/metal2_repeat_codegen_test_repeat.log 2>&1
   pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_repeat_codegen_routing.py -x -v \
       > /tmp/metal2_repeat_codegen_test_routing.log 2>&1
   grep -c METAL2_CHECKS_FORCED /tmp/metal2_repeat_codegen_test_repeat.log   # expect >= 2
   ```
   Branch coverage inside `test_repeat.py::test_repeat_codegen` (forced through
   `ttnn._ttnn.operations.data_movement.repeat_force_codegen`, bf16 + fp32):
   - TILE: `(1,1,32,32)x(2,1,1,1)`, `x(1,3,1,1)`, `x(1,1,2,1)`, `(1,1,32,64)x(1,1,1,2)`, rank-2 `(32,64)x(2,1)`
   - RM higher-dim: `(2,3,4,8)x(2,1,1,1)`, `(1,2,4,8)x(1,1,2,1)`
   - RM last-dim: `(1,1,4,8)x(1,1,1,2)`
   `test_pc_repeat_codegen` (TILE + RM last-dim) and `test_pc_with_different_shapes_in_sequence` exercise the
   program-cache-hit path (the `UpdateTensorArgs` refresh on this concept). `test_repeat` / `test_pc_repeat` route
   through `ttnn.repeat` and also land on codegen for gate-supported shapes. `test_repeat_codegen_routing.py` pins the
   fallback gate and a handful of forced RM cases.
5. **No C++ gtest** references this op (`grep -rl RepeatCodegen tests/` → the two pytest files only).
6. Remove the forcing scaffolding from `tt_metal/impl/metal2_host_api/*.cpp` afterwards; it must not reach the PR.

If a test that passed at baseline fails post-port, stop and report (structural spec error or arg-layout mismatch);
a `TensorSpec` legality failure only on the *second* dispatch would implicate the cache key — this op has no custom
hash, so that shape would point at the framework rather than the op.

## Open items for downstream

### Shared kernel touches
| kernel path | rung | fork created | pointer comment in original | remaining unmigrated consumers |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp` | 2 (create) | `common/kernels/codegen/reader_tile_interleaved_unified_metal2.cpp` | yes (lines 5-8) | `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp:38→113` (`seq_id = SEQ_REPEAT_INTERLEAVE`) |
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp` | 2 (create) | `common/kernels/codegen/writer_interleaved_metal2.cpp` | yes (lines 5-8) | `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp:40→124,188` (TILE and RM branches) |

Fork vocabulary the `repeat_interleave/codegen` port must adopt (read-only to it):
- reader: `tensor::src`, `dfb::in`, CTAs `seq_id` / `batch` / `src_page_pitch`, RTAs `num_pages` / `start_id` /
  `num_repeats` / `lower_pages` / `rep_dim_pages`. Its `SEQ_REPEAT_INTERLEAVE` branch is present and reads the same
  five RTAs (`sequencers.h::seq_repeat_interleave_init` takes exactly those), so that port should need no fork edit.
- writer: `tensor::dst`, `dfb::out`, CTAs `requested_write_size` / `batch`, RTAs `num_tiles` / `start_id`.
- Sunset: when `repeat_interleave/codegen` binds the forks, the two legacy originals have no binder left and can be
  retired (fork takes over the name). `untilize/codegen` has its own private copy and is not on this sunset list.

### Per-op carry-over
- **`untilize/codegen/kernels/reader_tile_interleaved_unified.cpp`** is a same-named private copy of the shared
  reader (likely a different `seq_id`); its port will meet the same dead-branch / generated-name issue — the fork here
  is the template for that decision.
- **`repeat_codegen_supported.cpp:38-85`** still speaks "CB" (`rm_cb_fits_in_l1`, `cb_stick_elems`,
  `projected_cb_bytes`, comments) and the shared depth constant is named **`kRepeatCbDepth`**
  (`repeat_codegen_program_factory.hpp:20`). Renaming both to DFB vocabulary is a mechanical follow-up that touches
  device-op-level host code, which this port may not edit; the header comment was updated to say "DFB entries" but
  the identifier was kept so `repeat_codegen_supported.cpp` stays byte-identical.
- **`reader_repeat_higherdim_rm.cpp:7`** header comment says it "mirrors the shared `reader_repeat_higherdim_rm.cpp`"
  (names itself; the shared file is `reader_tile_interleaved_unified.cpp`). Comment-only, pre-existing, preserved
  verbatim per whitelist rule 8.
- **`writer_repeat_rm.cpp:15-17`** header says both RM builders "pass [real_stick, aligned_page]"; the factory in
  fact passes `[aligned, aligned]` (`xfer_size == l1_stride`) on both RM branches. Stale but harmless; preserved
  (only the "CT slot 1" positional reference was reworded to `xfer_size`).

### Doc-evolution suggestions
- Catalog entry candidate: *Shared kernel with pluggable compile-time strategies* — carry only the strategies that
  have binders; each other strategy goes behind a `compiler_options.defines` `#ifdef`; `static_assert` the selector.
- §Verification: an explicit "verification delegated to the invoker" outcome and what `Outcome` should say.
- §Kernel-side whitelist rule 4: state the arg-name casing convention.

### Test coverage notes
- No test forces the codegen path on Blackhole-specific 64 B alignment for the RM last-dim sub-16 B stick
  specialisations (`stick_size == 2/4/8/12`); `(1,1,4,8)` bf16 gives `stick_size = 16` (the NOC L1→L1 branch) and
  fp32 gives 32. The RISC-copy branches are exercised only through `ttnn.repeat` routing when a W<8 bf16 RM repeat
  happens to be gate-supported. Pre-existing gap; unchanged by the port.
- `SEQ_REPEAT_INTERLEAVE` in the reader fork has no binder yet and is therefore untested until the
  `repeat_interleave/codegen` port lands.
