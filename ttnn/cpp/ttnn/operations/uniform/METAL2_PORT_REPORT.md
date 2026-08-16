# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/uniform`

## Outcome

`PORTED` — the op's single factory (`UniformDeviceOperation`, formerly the direct `create_descriptor`) is on
`CustomProgramSpecFactoryConcept`. No factories remain on a legacy concept. Build clean; the confirmed test
baseline passes **101/101** (see [Verification](#verification)).

## Provenance

- **Recipe docs (this port):** `0a67b260fee 2026-08-16 docs(metal_2.0): four small fixes from the cold audit runs`
- **Audit docs (inherited):** `086a669ff5e 2026-08-15 docs(metal_2.0): two porter-facing gaps a blind cold read turned up`

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept`, as the audit chose. `override_runtime_arguments` was **translated**, not
deleted: it now returns a `ProgramRunArgs` (`device/uniform_program_factory.cpp:286`) applied via
`UpdateProgramRunArgs`.

**Tensor-arg completeness on the custom concept.** The op has exactly one `TensorParameter` (`OUTPUT`, the
in-place tensor), and the translated override returns a `TensorArgument` for it on every dispatch
(`uniform_program_factory.cpp:299`). None skipped. This is the faithful translation of the ported-from
override's `writer_args[0] = out_addr` (`:243` pre-port) — an address written as a *number* becoming a *binding*.

**The backdoor hash and the override are a matched pair, and both survive.** `attribute_values`
(`device/uniform_device_operation.hpp:28-29`) still excludes `from` / `to` / `seed` from the program hash, and
the translated override still supplies all three (`seed`, `f2u_from`, `f2u_to` in the compute kernel's
`runtime_arg_values`, built by the shared `add_uniform_run_args` helper). Verified against the brief's explicit
instruction to check this.

### Device-op-class edits

- **Pybind entry points removed:** none. `uniform_nanobind.cpp` binds only the user-facing `ttnn::uniform`; it
  never referenced `create_descriptor`. The file is byte-identical.
- **Custom `compute_program_hash`:** none. **Backdoor hash left intact, byte-identical**, at
  `device/uniform_device_operation.hpp:28-29`.
- **Forced restructure (see Friction → Gaps):** `create_descriptor` / `override_runtime_arguments` moved off the
  device-op struct into a nested `UniformDeviceOperation::ProgramFactory`, with
  `using program_factory_t = std::variant<ProgramFactory>;` added. Header-only; no behavioral surface changed.
  Everything else in the class (`validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`,
  `create_output_tensors`, `operation_attributes_t`, `tensor_args_t`) is untouched.

### Open items

- **Relaxation candidates:** none. Consistent with the audit's `TensorParameter relaxation == none`. The backdoor
  hash excludes only scalar op attributes, never a tensor property, so it implies no relaxation.
- **Capabilities not yet on this concept that the op would benefit from:** none observed. Single-program, no
  op-owned tensors, no op-owned `GlobalSemaphore`s.
- **Concept fit:** good. The one wiring friction is the `program_factory_t` restructure above.

## Handoff points

- **Shared-kernel forks created (informational, not an escalation).** See
  [Open items for downstream](#open-items-for-downstream) — the forks are within the sanctioned rung-2 carve-out
  and need no owner action today, but `rand`'s eventual porter inherits their binding vocabulary.
- **Readiness-sheet hold overridden on the invoker's instruction.** The brief's pre-flight check
  (`Is able to port? == no`, a deliberate family-wide hold on `CustomProgramSpecFactoryConcept` ops) was raised
  with the invoker before any code was written. The invoker confirmed this port **is** the test-out of that
  support and authorised proceeding. Recorded here so a later reader does not mistake a GREEN audit plus a
  completed port for evidence that the hold had lifted — as of this port it had not. The sheet row
  (`descriptor` / override `yes` / `CustomProgramSpecFactoryConcept` / `no`) remains the authority.
- **No boundary-rule assumption violations.** No out-of-op call site required a `sem::` or `tensor::` handle.
  The kernels' only out-of-op callees are LLKs (`init_sfpu`, `pack_tile`, `rand_tile*`, `Noc::async_write`) and
  the two that take a CB id take `dfb::intermed` through the sanctioned implicit conversion.
- **No kernel-lib gaps, no framework gaps, no capitulation.** No `GlobalCircularBuffer`, no
  `get_cb_tiles_*_ptr`, no compute-kernel Case 2 binding, no host-computed base-pointer offset.

## Successes

- **[CB→DFB API whitelist §B](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#b-size--layout-queries)
  answered a question the brief said had no clean answer.** The brief and audit both flagged
  `get_local_cb_interface(dst_cb_id).fifo_page_size` (`writer_uniform.cpp:26`) as a lookup with *"no direct
  `fifo_page_size` analog"* on `DataflowBuffer`, inferring `get_tile_size()` from the descriptor and asking the
  porter to confirm before swapping. Whitelist §B maps it directly: `fifo_page_size → get_entry_size()`. Checking
  the header confirmed it is an exact identity on a DM build — `get_entry_size()` is
  `address_units_to_bytes(fifo_page_size)` (`tt_metal/hw/inc/internal/tt-1xx/dataflow_buffer.inl:35-41`) and
  `cb_addr_shift == 0` off-TRISC (`tt_metal/hw/inc/internal/circular_buffer_interface.h:145-149`). Used at
  `device/kernels/writer_uniform_metal2.cpp:26`. The recipe's *"go to the headers first"* instruction is what
  turned an inference into a proof.

- **[Caution: Porting a shared kernel](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  fired exactly as designed — this port would have broken `rand` without it.** Both kernels live in `uniform`'s
  own directory, i.e. inside the writeable surface, and nothing about their paths hints that `rand` binds them.
  The caution names this *lent* shape explicitly ("the file sits inside your writeable surface, so converting it
  in place feels safe"), and its census procedure (`grep -rl <filename>`, then check each hit) turned up
  `rand_program_factory.cpp:27-28` in seconds. Rung 2 taken.

- **[Compiler options](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  caught a silent perf regression that nothing else would have.** `grep -n opt_level` over the op returns
  nothing, which reads as "no setting, nothing to do." The section's rule 2 says otherwise: an absent field on a
  `ComputeConfigDescriptor` still resolves to `O3`, while `CompilerOptions` defaults to `O2`. Set explicitly at
  `device/uniform_program_factory.cpp:234`. No build error, no test failure, and no validator would have flagged
  the drop.

- **[Two-toucher endpoint-assignment procedure](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  made the endpoint re-derivation a two-minute count** rather than a judgment call. Both DFBs re-derived
  independently of the brief; both agreed (`INTERMED` 1P+1C, `DST` self-loop). The recipe's instruction to
  *verify, not transcribe* cost almost nothing here and would have caught a brief over-read.

## Friction

### Gaps

- **An op with `create_descriptor` *directly on the device-op struct* is forced into a `program_factory_t`, and
  no doc says so.** `uniform` used the framework's direct-descriptor shortcut: `HasDirectDescriptor` is
  `&T::create_descriptor` present *and* `program_factory_t` absent (`ttnn/api/ttnn/operation_concepts.hpp:139`),
  and `resolve_program_factory` synthesises a `DirectDescriptorFactory` for it
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170-196`). **There is no spec-path analog** — no
  `HasDirectProgramArtifacts` — so the instant `create_descriptor` goes away, `DeviceOperationConcept`'s
  `HasDirectDescriptor || HasProgramFactoryType` disjunct (`operation_concepts.hpp:207-209`) forces the op to
  grow a `program_factory_t` plus a nested factory struct.

  Why this is a real gap rather than a triviality: it is a **device-op-class header edit**, and
  [Host-side: stay in the lane](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#host-side-stay-in-the-lane)
  says that file is off-limits with **two** documented exceptions (pybind removal; a pybind-hook-only parameter).
  This is a third, and it is *mandatory* — a porter reading the scope rules literally has no sanctioned move.
  The recipe's own gloss ("the program factory body is the port") arguably covers it, since the declarations
  being moved *are* the factory's entry points, but that requires the porter to reason past an explicit
  two-item list. Suggested fix: name it as a third exception in
  [`ttnn_factory.md` — Device-operation-class edits the port forces](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#device-operation-class-edits-the-port-forces),
  e.g. *"3. Introduce a `program_factory_t` when the ported-from op used the direct-descriptor form. The spec
  concepts have no direct-on-device-op shortcut, so `create_descriptor`'s removal forces a nested factory struct
  and a `program_factory_t` alias. Purely structural; nothing else in the class changes."* Realized at
  `device/uniform_device_operation.hpp:39-58`.

- **The audit brief's `fifo_page_size` breadcrumb contradicts the CB→DFB whitelist, and the brief is the document
  the porter is told to act on.** Detail in [Successes](#successes) — the substance landed right, but only
  because the recipe sends the porter to the whitelist and the headers independently. A porter who trusted the
  brief's *"`dfb::dst.get_tile_size()` should be the equivalent"* would have shipped a value that happens to be
  equal for this op (page size == tile size here) but is not the API-correct lookup, and would have carried the
  brief's stated uncertainty into the code. The audit doc's `Breadcrumb` step should consult
  [whitelist §B](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#b-size--layout-queries)
  before declaring a lookup unmapped; §B's last row is literally
  `get_local_cb_interface(...).fifo_page_size → same section B getters`.

- **The ephemeral-doc citation check doesn't see uncommitted port work.** The
  [anti-pattern self-audit](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)
  prescribes `git diff --name-only origin/main... | ...`. The three-dot form diffs *merge-base → HEAD*, i.e.
  committed history only — so run at the natural time (before committing the port) it inspects zero port files
  and passes vacuously. Every sibling check in that section is careful about this: the `TT_FATAL` census
  explicitly notes *"the right-hand side reads the working tree, so uncommitted port work counts."* Suggested
  fix: `git diff --name-only $(git merge-base origin/main HEAD) | ...` (two-dot against the merge-base commit),
  which reads the working tree.

### Confusion

- **Neither `create_descriptor` nor its two-arg / four-arg calling convention prepared me for the
  `override_runtime_arguments` *signature* change.**
  [`ttnn_factory.md`](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#the-custom-concept-customprogramspecfactoryconcept)
  does state it ("Only the **return type** is concept-enforced… the port has to change the method's *shape*"),
  but the ported-from method also **loses its first parameter** (`tt::tt_metal::Program& program`) — it no longer
  mutates a program, it returns a description. That is the single most consequential shape change on this path
  and it is implicit in the two side-by-side signatures rather than called out. One sentence — *"the `Program&`
  parameter goes away; the method describes rather than mutates"* — would land it faster.

- **The repo's pre-commit hooks include three Metal-2.0-aware checks, and no port doc mentions them.**
  `Detect legacy device operation classes in newly added files`, `Detect smuggled buffer-address runtime args in
  descriptor factories`, and `Detect ProgramDescriptor rebuilds inside override_runtime_arguments` all run on
  `git commit` and all passed here. Two of those overlap directly with
  [anti-pattern self-audit](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)
  items, so they are a free second opinion on the port — a porter would benefit from knowing they exist (and
  that a *failing* one is a real finding, not a lint nit). Also worth a heads-up that `clang-format` rejects the
  first commit and rewrites files, so the commit has to be re-staged and re-run; that surprised me mid-verify
  and meant re-confirming the build/tests against the reformatted tree.

- **`TT_KERNEL` exists and no port doc mentions it.** `tt_metal/hw/inc/experimental/kernel_args.h:44-47` defines
  a `TT_KERNEL` marker whose signature the JIT parses to generate `kernel_main()`
  (`tt_metal/jit_build/kernel_signature_parser.hpp`). The recipe and migration guide both show plain
  `void kernel_main()` + `get_arg(args::name)`, which is what this port used and which the parser explicitly
  supports ("Returns `std::nullopt` when the source contains no `TT_KERNEL` marker… fully backward compatible").
  So there was no wrong turn — but a porter who reads the header the recipe sends them to will meet an
  undocumented second style and have to decide. Worth one line saying `kernel_main()` is the port's form.

- **Near-miss avoided: the compute config is a hybrid the Style A / Style B fork doesn't quite name.**
  [Compute kernels](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels)
  splits on *"the op resolves a TTNN `ComputeKernelConfig`"* (Style A) vs *"sets a Metal `ComputeConfig`
  directly… with no TTNN `ComputeKernelConfig` feeding them"* (Style B). `uniform` does **both**: it calls
  `get_compute_kernel_config_args` and then hardcodes `fp32_dest_acc_en = true` over the resolved value
  (`uniform_program_factory.cpp:182` pre-port), and drops `packer_l1_acc` entirely. Read literally it is Style A,
  and Style A's prescription — `to_compute_hardware_config(device->arch(), config)` — would have **silently
  restored the user's `fp32_dest_acc_en`**, defeating a deliberate, commented override that the op says exists to
  stop generated values leaving `[from, to)`. Style B mechanics were used instead
  (`uniform_program_factory.cpp:244-252`). The section's own warning ("the two config structs default *opposite*
  ways") is what made the hazard visible, but the A/B fork could say that **any** field the op overrides after
  resolving a TTNN config puts it on Style B.

## Open items for downstream

- **Shared kernel touches — two forks created (rung 2).** Both kernels are *lent*: they live in `uniform`'s
  directory but `rand` binds them by file path.

  | Kernel | Rung taken | New file | Pointer comment landed in original |
  |---|---|---|---|
  | `device/kernels/writer_uniform.cpp` | **2 — created the fork** | `device/kernels/writer_uniform_metal2.cpp` | yes (`writer_uniform.cpp:5-9`) |
  | `device/kernels/compute_uniform.cpp` | **2 — created the fork** | `device/kernels/compute_uniform_metal2.cpp` | yes (`compute_uniform.cpp:5-9`) |

  **Remaining unmigrated consumer:** `ttnn/cpp/ttnn/operations/rand` — binds both originals at
  `device/rand_program_factory.cpp:27-28`, used at `:165` and `:181`. `rand` carries the identical
  readiness-sheet profile (`descriptor`, backdoor hash `yes`, override `yes`) and sits under the same
  family-wide hold, so it cannot co-migrate today. **When `rand` ports, the legacy originals can be deleted and
  the forks renamed to take their names** — this list is that sunset checklist.

  **Binding vocabulary `rand` will inherit** (named for the kernel's own role words, not `uniform`'s locals, per
  the caution):
  - `dfb::intermed` — Float32 staging DFB; compute PRODUCER, writer CONSUMER
  - `dfb::dst` — output-dtype staging DFB; writer PRODUCER **and** CONSUMER (self-loop)
  - `tensor::dst` — the output tensor
  - writer named args: `start_id`, `num_tiles`; compute named args: `seed`, `f2u_from`, `f2u_to`, `start_id`,
    `num_tiles`
  - writer `#ifdef`s the factory must feed: `OUTPUT_DTYPE_BFLOAT16` / `OUTPUT_DTYPE_FLOAT32`
  - `rand`'s factory is a near-clone of `uniform`'s, so it should fit these names without a fork edit. If it
    does not, that is a Handoff point for `rand`'s porter — **not** an edit to these forks.

- **Quasar-uplift debt this port deliberately incurs:** `dfb::dst` is a **DM self-loop**
  (`uniform_program_factory.cpp:204-213`), which Gen2 rejects. Sanctioned on Gen1 and left for the Quasar audit
  to find; recorded here so it is not a surprise.

- **Pre-existing findings, carried forward unchanged (the port fixed none of these):**
  - **The `1e-6f` endpoint epsilon does not scale with the range** — `uniform_program_factory.cpp:97-99`.
    `to - 1e-6f` in `float` is at or below one ULP for `to` above roughly 8.4, and exactly `to` for large `to`,
    so the documented half-open `[from, to)` silently becomes closed and the op can return exactly `to`. Nothing
    validates `to`'s magnitude (`uniform_device_operation.cpp:18` only checks `from < to`). A relative epsilon
    (`std::nextafter(to, -inf)`) would hold the contract. *Also raised by the audit; repeated here because this
    is the artifact the port's reviewer reads.*
  - **`fp32_dest_acc_en` and `packer_l1_acc` participate in the cache key but not in the program.**
    `get_compute_kernel_config_args` (`uniform_program_factory.cpp:165-166`) destructures all five fields;
    `enable_32_bit_dest` is hardcoded `true` and `packer_l1_acc` is never read — yet the whole
    `compute_kernel_config` attribute feeds the program hash (`uniform_device_operation.hpp:29`). Two distinct
    cache entries are kept for behaviourally identical configs. The `fp32_dest_acc_en` forcing is intentional and
    commented; the hash consequence and the dead `packer_l1_acc` look unintended.
  - **The writer's `default: break;` dtype arm emits no define** (`uniform_program_factory.cpp:190`), compiling
    a writer whose loop body performs no NOC write — a silent no-op rather than a diagnostic. Unreachable today
    because `validate_inputs` constrains the dtype; a `TT_THROW` in the `default` arm would keep it that way if
    that validation ever loosens. **Note the asymmetry:** the near-identical `rand` factory *does* `TT_THROW`
    here (`rand_program_factory.cpp:173-175`). Carried across verbatim rather than harmonised, per the porting
    invariant.
  - **The `dst` DFB costs a full Float32 tile of L1 for nothing under FLOAT32 output.** The writer reserves,
    peeks and pushes it in both configs, but only the BFLOAT16 branch writes bytes into it; under FLOAT32 it
    exists solely to carry its entry size (`writer_uniform_metal2.cpp:26`), which is a host-known constant that
    could have been a CTA. Carried across unchanged — the brief explicitly says not to drop it.

- **Test coverage note.** `uniform`'s only coverage lives at
  `tests/ttnn/nightly/unit_tests/operations/rand/test_uniform.py` — under the **`rand`** family directory, not a
  `uniform` one, and in the **nightly** tree rather than the primary one. Exactly the "family slug ≠ source
  family" trap the recipe's test-location section warns about; a porter guessing
  `tests/ttnn/unit_tests/operations/uniform/` finds nothing and could conclude the op is untested.
  `test_uniform_seed_distinguishes_cache_entries` is the load-bearing one for this port — it pins both halves of
  the hash-exclusion / override contract that `CustomProgramSpecFactoryConcept` carries.

## Verification

- **Build:** `./build_metal.sh --build-tests` — SUCCESS (exit 0, zero `error:` / `FAILED:` lines).
  Both forks are picked up by the existing `file(GLOB_RECURSE kernels device/kernels/*.cpp)` in
  `ttnn/cpp/ttnn/operations/uniform/CMakeLists.txt`; **no build-file edit was needed**, as the caution predicts.
- **Tests:** `pytest tests/ttnn/nightly/unit_tests/operations/rand/test_uniform.py -v` (the invoker-confirmed
  no-regression baseline) — **101 passed, 0 failed** (exit 0). Every parametrisation of all four test functions,
  across both dtypes, both `fp32_dest_acc_en` settings, and the seed/range sweep. Run twice: once before the
  `clang-format` pre-commit pass and once after, so the numbers above are for the tree as committed.

  The two that matter most for this concept both pass:
  - `test_uniform_callback` — asserts a changed `seed` does **not** grow the program cache, i.e. the cache-hit
    path is genuinely being taken and the translated override is what re-applies the seed.
  - `test_uniform_seed_distinguishes_cache_entries` — asserts, on cache hits only, that a changed `seed`
    changes the output, a changed `from`/`to` is re-applied (`5.0 <= out < 10.0`), and neither grows the cache.
    This is the direct test of the backdoor-hash / `override_runtime_arguments` pairing the brief flagged, on
    the in-place (input == output) path. It exercises the returned `TensorArgument` too: without it the output
    binding would freeze at the cache-miss address.
- **Anti-pattern self-audit:** all items clean.
  - no `buffer()->address()`, no `emplace_runtime_args`, no bare `Buffer*` in the ported code
  - no magic CB indices — both `KernelSpec`s have **empty** `compile_time_args`
  - no `TensorAccessorArgs<N>()` in either fork
  - `grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]'` over the ported files: **zero hits**
  - no `.id` extraction; `dfb::intermed` passed directly to `init_sfpu` / `pack_tile`
  - no conditional DFB bindings (both DFBs bound on every path; the `OUTPUT_DTYPE_*` define selects *code*, not
    *bindings*), no `allow_instance_multi_binding`, no varargs, no CTA→RTA demotion
  - `TT_FATAL` / `TT_ASSERT` / `TT_THROW` per-file census across the port: **no code file's count dropped**
  - no `.md` cited from any ported `.cpp` / `.hpp`
  - `opt_level`: compute explicit `O3`, writer left at the matching `O2` default
