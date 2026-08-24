# Metal 2.0 Port Report — `moreh_nll_loss_step1`

## Outcome

**`PORTED`** — the op's single factory (`MorehNllLossStep1DeviceOperation::Factory`) is converted to
`ProgramSpecFactoryConcept`, together with all three kernel entry points it can bind
(`reader_moreh_nll_loss_step1.cpp`, `reader_moreh_nll_loss_step1_large.cpp`,
`writer_moreh_nll_loss_step1.cpp`). No factories remain on the legacy concept — the op has only this one.

Tests: **38 passed, 32 skipped, 62 deselected** post-port, **identical** to the pre-port baseline measured
on the same command and the same tree state. The 32 skips are `bfloat8_b` parametrizations the test file
skips itself; the 62 deselections are the `-k "not backward"` filter (`moreh_nll_loss_backward` is a
different device operation). Plus a targeted ad-hoc run for the large-algorithm path, which no repo test
reaches — see [Open items](#open-items-for-downstream).

## Provenance

- **Recipe docs (this port):** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
- Pre-port base (`git merge-base origin/main HEAD`): `9d839c902f5`

## TTNN ProgramFactory

### Concept realized

**`ProgramSpecFactoryConcept`** — as the audit chose. No disagreement arose, so nothing was re-decided.
`Factory::create_descriptor` became `Factory::create_program_artifacts` inside the existing nested
`Factory` struct; the op already had `using program_factory_t = std::variant<Factory>`, so the
direct-descriptor exception ([`ttnn_factory.md` exception 3]) did not apply. There is no
`override_runtime_arguments`, so the framework refreshes the tensor bindings on a cache hit and the port
writes exactly one method.

### Device-op-class edits

- **Pybind entry points removed: none.** `moreh_nll_loss_nanobind.cpp` binds only the user-facing
  `ttnn::moreh_nll_loss`; no factory or device-op internal was ever exposed, so the port removes no
  user-visible Python surface.
- **Custom `compute_program_hash`: none** — the op uses the default reflection-based hash. Nothing to
  leave intact, nothing touched.
- The only edit to `moreh_nll_loss_step1_device_operation.hpp` is the factory method's declaration and
  its includes (`<tt-metalium/program_descriptors.hpp>` → `"ttnn/metal_v2_artifacts.hpp"`).
  `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`,
  `create_output_tensors` and `moreh_nll_loss_step1_device_operation.cpp` are **byte-identical**.

### Open items

- No relaxation candidates were mined: the audit recorded `TensorParameter relaxation = none`, and there
  is no custom hash from which one could be inferred. All three `TensorParameter`s stay strict.
- No capability gap. The op needs nothing this concept lacks — single program, no op-owned tensors, no
  op-owned `GlobalSemaphore`s, no per-coord variation.

## Handoff points

**None.** Nothing in this port reached outside the op's own directory, and nothing needed a change the
recipe forbids.

Recorded so the absences are evidence rather than silence:

- **No boundary-rule assumption violation.** No call site required passing a `sem::` or `tensor::` handle
  out of the op. The only out-of-directory callees are `read_tile` / `read_value` / `read_line` /
  `get_tilized_idx` in `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, and all four take either
  `DataflowBuffer` by value or plain scalars. The donor is untouched.
- **No kernel-lib gap, no framework gap, no shared-kernel fork.** The op owns all three kernel `.cpp`
  files and no other op binds them, so there was no `_metal2` fork to reuse or create and no pointer
  comment to leave anywhere.
- **No capitulation.** Every construct this factory uses has a Metal 2.0 expression.

## Successes

- **[Patterns catalog — Two-toucher DFB → assign 1P+1C], step 3 ("re-derive, don't transcribe")** fired
  exactly as intended. I re-ran the endpoint census from the kernel sources rather than copying the
  brief's dispositions, and it **agreed in every particular** — three self-loops (`target` all configs,
  `weight` in B/C, `weight_scratch` in B), one plain 1P+1C (`output`), one dead drop (`c_24`), one
  config-conditional (`weight_scratch` in C). Cheap to run, and it converted "the brief says so" into
  "the kernel says so."
- **[Port recipe — Ensure the Metal 2.0 host-side legality checks are enabled]** earned its place. The
  grep found **nine** `skip_validation` sites, not the two the prose names as the choke points —
  `UpdateTensorArgs` among them. Forcing them all and proving it with the marker is what makes the green
  below mean something.
- **[cb_dfb_api_whitelist §A — `constexpr` metadata values]** decided both `get_tile_size` sites without
  a judgement call. Both legacy declarations were `const`, not `constexpr`
  (`writer_moreh_nll_loss_step1.cpp:25`, `reader_moreh_nll_loss_step1_large.cpp:37` pre-port), so both
  took the member-getter form. Reading the declaration is genuinely a one-second test.
- **[Port recipe — Scope discipline / the `cb_usage` trap]**, restated in the brief, caught the exact
  tidy-up I would otherwise have made. Having dropped `c_24`'s allocation, deleting its
  `intermed_num_tile * intermed_tile_size` term from the usage sum reads as obvious cleanup — and it
  would have moved the small/large algorithm threshold. The term is preserved verbatim at
  `..._program_factory.cpp:117-118` with a comment explaining why a size is computed for a buffer that
  no longer exists.
- **The sibling op's shape held up.** `moreh_nll_loss_backward` is already on Metal 2.0 with the same
  optional-`weight` + `weight_scratch` structure, and its self-loop bindings and `AddRuntimeArgsForNode`
  loop matched what the recipe prescribes independently. Useful corroboration, not a substitute for the
  recipe.

## Friction

### Gaps

- **The recipe has no rule for a CTA whose only job was positional padding for `TensorAccessorArgs`.**
  Both readers declare `constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;` and never
  use it; pre-port the *slot* was still load-bearing, because it kept `TensorAccessorArgs<1>` at its
  offset. The port deletes the accessor plumbing by mandate, so that justification evaporates and the CTA
  becomes purely dead — but [Dropped Plumbing] enumerates six categories and none of them is "a CTA the
  kernel reads and discards." I kept it (as a named CTA, with the kernel-side read preserved), on the
  grounds that 1:1 translation is the default and deleting is the deviation needing justification; the
  reasoning is written out in `METAL2_PORT_PLAN.md`. **Suggested addition:** a line under Dropped Plumbing
  saying which way to go when the port removes a dead argument's *reason for existing* but not the
  argument — and noting that the answer may differ from the identical-looking dead-RTA case, where this
  port's invoker chose deletion.
- **`Table` has no `emplace_back`, and the error is not obvious from the docs.** The recipe warns that
  `Table` is a map, not a vector, which covers `push_back`. What actually bites is the mirror: `Group<T>`
  *is* a vector and takes `push_back`, and the two types sit adjacent in the same initializer block, so
  the reflex is to reach for the wrong one in each. A one-line "`Group` → `push_back`, `Table` →
  `emplace`/`insert`/`operator[]`" pairing in the `Table`s-are-maps paragraph would settle it.

### Confusion

- **"Leave `cb_usage` byte-for-byte" reads as forbidding the rename the self-audit requires.** The brief
  and the audit both say to leave `cb_usage` (`..._program_factory.cpp:67-68`) byte-for-byte; the
  anti-pattern self-audit requires `grep -rnE '[Cc][Bb]_|…'` over the op directory to return **zero**,
  and `cb_usage` is a hit. The two are reconcilable — "byte-for-byte" is about the *arithmetic*, not the
  identifier — but only after noticing that the audit's warning predates the rename rule's scope. I
  renamed the local to `dfb_usage` and left every term of the sum identical (and likewise
  `weight_cb_tiles` → `weight_dfb_tiles`), and I am flagging it here so a reviewer diffing against the
  brief does not read the rename as the change the brief warned about. **Suggested wording for the
  audit-side guidance:** say "leave the *arithmetic* term-for-term" rather than "byte-for-byte," since a
  `cb`-named local is going to be renamed by the porter's own checklist.
- **Whitelist rule 7 can force a statement to move, and the rule does not say so.** In the large reader,
  the legacy `const uint32_t weight_tile_bytes = get_tile_size(cb_weight);` (`:37`) sat *above* the
  `DataflowBuffer` construction (`:54`). The member-getter form needs the object, so converting the line
  in place is impossible; the fix is to hoist the `DataflowBuffer dfb_weight_obj(dfb::weight);`
  construction above it (keeping exactly one object for the buffer, per [Same-FIFO aliasing]). Small and
  obvious once seen, but it is a *structural* edit that the "swap the call for a getter" framing does not
  prepare you for, and the tempting wrong move — keeping the free-function token form, which rule 7
  reserves for `constexpr` sites — is one keystroke away.

## Findings (behaviour preserved, not fixed)

Everything here is reproduced faithfully in the port. Listed for the ops team.

1. **Two dead RTAs — dropped, on the invoker's explicit instruction.** `element_size` (reader RTA idx 7,
   set at `..._program_factory.cpp:213` from the local at `:201`) and `target.element_size()` (idx 8, set
   at `:214`) were read by both readers (`:20-21` in each, pre-port) and never used. The invoker directed
   that these be **dropped rather than named**; both the host emission and the kernel-side reads are
   gone, and the host local `element_size` went with them. This is a deliberate deviation from strict
   binding-only translation, and it is zero-functional-change: a value nothing reads has no behaviour.
   The op now ships **five** live reader RTAs where it shipped nine.
2. **A declared-but-unused CTA remains, and is now dead outright.** `weight_has_value` is still emitted
   and still read-and-discarded by both readers. Pre-port it at least held a CTA offset open; post-port it
   holds nothing. It duplicates, through a second mechanism, exactly what the `WEIGHT` define already
   says. A follow-up that removes it should remove the kernel-side read in the same change. (Audit misc
   anomaly 7.)
3. **`compute_kernel_config` reaches this op only through a dead buffer's size.** `fp32_dest_acc_en` →
   `intermed_data_format` → `intermed_tile_size` → the usage sum → `use_large_algorithm` → *which reader
   kernel file compiles*. The op has no compute kernel, so there is no dest accumulation to configure;
   the parameter's only effect is to perturb the algorithm threshold. The port preserves this exactly —
   including keeping the `intermed_*` locals alive for a buffer it no longer allocates — but the coupling
   looks unintended, and a user changing `compute_kernel_config` can silently change which reader runs.
   (Audit misc anomaly 5, and the reason the `c_24` drop needed such care.)
4. **The `FP32_DEST_ACC_EN` define is dead.** Still emitted to the reader when `fp32_dest_acc_en`
   (`..._program_factory.cpp` post-port, in the reader defines block); neither reader nor any donor
   function they call consumes it. Its only in-header consumer, `fp32_dest_acc_cast`
   (`moreh_common.hpp:23-31`), is never called from this op. Preserved. (Audit misc anomaly 4.)
5. **A dead local pair in the large reader, converted rather than removed.**
   `reader_moreh_nll_loss_step1_large.cpp` computes `weight_tile_bytes` and `weight_element_size` and uses
   neither; `read_value` derives the same quantity internally (`moreh_common.hpp:709`). Kept, and the
   `get_tile_size` call converted to the DFB member getter per whitelist rule 7 — which is why the
   `DataflowBuffer` construction moved above it. (Audit misc anomaly 2.)
6. **`reduction` is an unused-but-hashed attribute.** `operation_attributes_t::reduction`
   (`..._device_operation.hpp:16`) is set by the caller but read nowhere in `step1`, while still feeding
   the default `compute_program_hash`. Two invocations differing only in `reduction` miss the cache and
   compile a second, byte-identical program. Untouched — the port never edits the cache key. (Audit misc
   anomaly 6.)
7. **The `/1024` element-size derivation in the donor is correct here but fragile.**
   `moreh_common.hpp:709` computes element size as `tile_size / 1024`, wrong for block-float formats. Safe
   in this op only because `validate_inputs` hard-asserts the weight tensor is `BFLOAT16`
   (`..._device_operation.cpp:26`). Donor code, out of scope, unchanged. (Audit misc anomaly 3.)
8. **Degenerate-config note, new in this port.** The `weight` buffer is declared on the legacy guard
   `weight_dfb_tiles > 0` while the `WEIGHT` define is emitted on `weight_has_value`. If a caller ever
   produced `weight_has_value == true` with `channel_size == 0`, legacy would define `WEIGHT` and index a
   buffer it never allocated (silent); the port would instead fail loudly at JIT with `dfb::weight`
   undeclared. The config is unreachable today (`channel_size` comes from a padded shape, so it is at
   least 32), and the port keeps the legacy guard rather than "fixing" the mismatch. Recorded because the
   failure *mode* differs even though the reachable behaviour does not.

## Verification evidence

- **Legality checks forced and proven live.** `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp`
  named **9** sites; all 9 were forced to `false` as the function's first statement, with one
  `METAL2_CHECKS_FORCED` marker per file (in `BuildProgramFromSpec` and `SetProgramRunArgs`), never in
  `UpdateProgramRunArgs`:

  | file | `skip_validation = false;` sites forced | `METAL2_CHECKS_FORCED` markers added |
  |---|---|---|
  | `tt_metal/impl/metal2_host_api/program_spec.cpp` | 4 (`BuildProgramFromSpec`, `MakeProgramFromSpec`, `MakeMeshWorkloadFromSpecs`, `MakeMeshWorkloadFromSpec`) | 1 |
  | `tt_metal/impl/metal2_host_api/program_run_args.cpp` | 5 (`SetProgramRunArgs`, `UpdateTensorArgs`, `MergeKernelRunArgsInto`, `UpdateProgramRunArgs`, `MergeProgramRunArgs`) | 1 |

  Marker counts observed in the logs — **both translation units fresh, checks running**:

  | run | `program_spec.cpp:2847` | `program_run_args.cpp:502` |
  |---|---|---|
  | main pytest run (`/tmp/nll_m2_test.log`) | **127** | **127** |
  | ad-hoc large-algorithm run (`/tmp/nll_large.log`) | **5** | **5** |

  **All of it reverted before finishing.** `git checkout --` on both files; the marker/force count in each
  is back to 0, `git diff --name-only <base> | grep '^tt_metal/'` names only the `third_party/umd`
  submodule pointer that was already modified in the working tree before this session began and is not
  part of the port, and `git diff <base> -- ttnn/cpp/ttnn/operations/moreh/ | grep -E
  'METAL2_CHECKS_FORCED|DO NOT COMMIT'` returns **0**.

- **Anti-pattern self-audit** — each sweep as *hits / files scanned*. Denominator for the op-directory
  sweeps is **6** `.cpp`/`.hpp` files; for the diff-scoped sweeps, **7** files.

  | check | result |
  |---|---|
  | buffer address in run-args (`buffer()->address()`, `emplace_runtime_args`, bare `Buffer*`) | **0 / 6** |
  | magic CB indices, `CBIndex`, `CBDescriptor`, `CBFormatDescriptor`, `.cbs`, `CircularBuffer` | **0 / 6** |
  | `TensorAccessorArgs<N>()` survivals | **0 / 6** |
  | `cb`-shaped names (`[Cc][Bb]_`, `_[Cc][Bb]\b`, `\b[Cc][Bb]\b`, `\bCB[A-Z]`) | **0 / 6** |
  | `.id` extraction at call sites | **0 / 6** |
  | `allow_instance_multi_binding` | **0 / 6** |
  | varargs of any kind (`get_vararg`, `num_runtime_varargs`, `compile_time_varargs`, …) | **0 / 6** |
  | legacy positional arg reads (`get_compile_time_arg_val`, `get_arg_val`, `get_common_arg_val`) | **0 / 6** |
  | all CTAs named | 1 `compile_time_args` site, named-pair form |
  | forced-legality scaffolding in the port's diff | **0 / 7** |
  | `.md` cited from code | **0 / 7** |
  | `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census vs base | **no diff** (device-op `.cpp` 6→6, factory 1→1) |

- **Conditional DFB bindings** follow [Pattern: Conditional / optional DFB bindings]: the host binds
  `weight` and `weight_scratch` conditionally, the matching `WEIGHT` flag rides on
  `KernelSpec::compiler_options.defines`, and every kernel-side reference to `dfb::weight`,
  `dfb::weight_scratch` and `tensor::weight` sits inside a pre-existing `#if defined(WEIGHT)` block. No
  binding was made unconditional as a workaround, and no new `#ifdef` was needed.

- **`hw_config` diffed before against after.** Legacy `ReaderConfigDescriptor{}` resolves to
  `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` and legacy `WriterConfigDescriptor{}` to
  `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` — the two role defaults exactly — so the port uses
  `ttnn::create_reader_datamovement_config(device->arch())` and
  `ttnn::create_writer_datamovement_config(device->arch())` respectively. No custom NOC, processor or
  `noc_mode` anywhere in the legacy factory, so nothing had to be replicated by hand. **No compute
  kernel exists**, so there is no compute `hw_config`, no `bfp_pack_precision_mode` and no `unpack_modes`
  question in this op at all.

- **`opt_level`**: `grep -n 'opt_level'` on the legacy factory returned **nothing**, so both DM kernels
  resolved to `O2`, which is Metal 2.0's `CompilerOptions` default; neither `KernelSpec` sets it. Counted
  from the construction code rather than the grep: the factory builds exactly **two** `KernelSpec`s, both
  data-movement (`grep -c 'ComputeHardwareConfig\|ComputeGen1Config\|to_compute_hardware_config'` on the
  factory → **0**). The compute-kernel `O3` rule has a **zero denominator** here — it does not apply,
  as distinct from having been skipped.

- **Test commands.**

  ```
  pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py -v -k "not backward"
  ```
  pre-port `38 passed, 32 skipped, 62 deselected` · post-port `38 passed, 32 skipped, 62 deselected`.

  `./build/test/ttnn/unit_tests_ttnn --gtest_list_tests | grep -ci nll` → **0**: this op has no C++ gtest
  coverage, so the pytest file is the whole baseline. Stated from the listing rather than from a grep
  over the test sources, so "none found" rests on the binary's own answer.

## Open items for downstream

- **Shared kernel touches: none.** The op owns all three kernel `.cpp` files, no other op binds them, no
  `_metal2` fork was reused or created, and no pointer comment was added anywhere. No sunset list.
- **Test coverage gap — the large-algorithm path is unexercised by the repo.** `use_large_algorithm`
  needs the buffer-usage sum to exceed available L1 (~1.4 MiB on Wormhole), i.e. a channel size around
  16k; the largest `C` in `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py` is 300.
  So `reader_moreh_nll_loss_step1_large.cpp` — one of the two reader entry points this port converts —
  is **never compiled by any test in the tree**, before or after the port.

  Because a port cannot be verified on a path nothing runs, I exercised it out-of-tree: shape
  `[4, 32768]` (so `weight_num_tile == 1024`, ≈2 MiB of usage), `reduction="mean"`, weight present,
  through `ttnn.operations.moreh.nll_loss`. It **passes** (`Max ATOL Delta 0.00175`, `Max RTOL Delta
  0.00400`, PCC gate 0.999), the JIT cache confirms `reader_moreh_nll_loss_step1_large` was actually
  compiled for that run, and the run carried 5 `METAL2_CHECKS_FORCED` marker pairs, so the spec was
  validated. The script is deliberately **not** committed — adding a test is outside a port's scope, and
  a ~2 MiB-per-core case wants an owner's judgement on where it belongs and what it costs in CI. A
  parametrization with `C ≈ 32768` added to the existing `test_moreh_nll_loss` would close the gap.
- **This factory only runs under `reduction == "mean"`.** `moreh_nll_loss.cpp:29-42` calls
  `prim::moreh_nll_loss_step1` on the mean branch alone; the `sum` and `none` branches go straight to
  `step2`. Worth knowing before reading a green result for this op, and worth knowing for whoever ports
  `step2`.
- **Sibling carry-over: `moreh_nll_loss_step2` is the natural next port** and is *not* covered here — it
  is a separate device operation with its own factory and its own six kernels, still on
  `ProgramDescriptor`. Two things from this port transfer directly: the `weight` / `weight_scratch`
  self-loop shape (step2 allocates the same scratch buffer with the same DRAM-alignment comment), and the
  three-place optional-`weight` consistency (host guard, `WEIGHT` define, kernel `#ifdef`). Unlike step1,
  step2 *does* have a compute kernel, so it will need the compute `hw_config`, `unpack_modes` and the
  explicit `O3` that this port had no occasion to exercise.
- **The `c_24` question the audit raised is now answered empirically.** The audit asked the ops team to
  confirm no planned or reverted compute kernel for `step1` would make the intermediate buffer live.
  Dropping it built and ran green across every reachable config, including the large path — which is the
  loud-failure direction the audit predicted, and it did not fire.
