# Metal 2.0 Port Report — `moreh/moreh_mean_backward`

## Outcome

**PORTED** — the single `descriptor` factory was converted to `MetalV2FactoryConcept`
(`create_program_artifacts`). Build/test verification is the **orchestrator's** (this porter did
not build or run tests, per orchestration constraints). Exact commands under *Verification (for the
orchestrator)* below.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as the audit specified. The legacy op was `HasDirectDescriptor`
(a `create_descriptor` directly on the device-op struct, no `program_factory_t`). The port:
- introduced a nested factory struct `MorehMeanBackwardOperation::MorehMeanBackwardProgramFactory`
  with `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`;
- declared `using program_factory_t = std::variant<MorehMeanBackwardProgramFactory>;` on the op
  struct (single alternative → framework auto-selects, no `select_program_factory`);
- removed `create_descriptor` from the op struct (and the now-unused
  `<tt-metalium/program_descriptors.hpp>` include; added `<variant>` and `ttnn/metal_v2_artifacts.hpp`).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** (op never had one).
- Pybind entry points removed: **none** (`moreh_mean_backward_nanobind.cpp` binds the host free
  function `ttnn::moreh_mean_backward`, not `create_descriptor`; no factory entry point was pybound).

### Open items
- **`output_grad` / `input_grad` are exact-match `TensorParameter`s** (strict, as default). The
  legacy factory deliberately used the `Buffer*`-overload RTA for `output_grad`/`input_grad` so the
  framework re-patches the address on every cache hit (documented in-code for AdamW-style training
  loops that pass a fresh `output_grad` each step; `program_factory.cpp:245-249`). The Metal 2.0
  `TensorBinding` preserves exactly this behavior — the adapter refreshes tensor args on cache hit —
  so the AdamW loop remains correct. No relaxation applied or needed.

## Handoff points

None. The port stayed entirely within the op directory. The three kernels' only out-of-op callees
are the shared moreh pool helpers (`ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`) and
LLK/HAL primitives, all reached via the sanctioned `dfb::name → uint32_t` implicit conversion or by
passing `DataflowBuffer` objects the kernel constructs locally — no shared-header signature change
was required.

## Successes

- **Self-loop DFB (recipe: Construct → sync-free/single-ended, and the accumulation reference).**
  `INTERMED` (legacy c_24) is filled and drained entirely inside the compute kernel
  (`moreh_mean_backward.cpp:52/58/63/75`). Binding it as **both** PRODUCER and CONSUMER on the
  compute `KernelSpec` (same accessor name `intermed`) satisfied the validator's "≥1 producer & ≥1
  consumer" rule with no kernel change — exactly as the brief's CB-endpoints disposition predicted.
- **Disjoint-node same-source split, not multi-binding (recipe: Two-toucher/§Anti-pattern).** The
  two compute instances cover disjoint core groups, so each shared DFB endpoint bound by both
  `COMPUTE_G1` and `COMPUTE_G2` contributes exactly one instance per node — no
  `allow_instance_multi_binding` needed. The brief called this out and the census confirmed it.
- **`dfb::name` implicit conversion to `uint32_t` (recipe kernel-side rule 2 + patterns catalog).**
  Passing `dfb::in` / `dfb::zero` / `dfb::scalar` / `dfb::intermed` / `dfb::out` directly into the
  LLK bcast/copy/`binary_op_init_common` calls, and passing the `DataflowBuffer` objects into the
  `*_init_short_with_dt` / `pack_tile_with_dt` moreh_common helpers, both worked without `.id`
  extraction or temp wrappers.
- **`get_tile_size(cb_id)` → `dfb.get_tile_size()` (recipe rule 7 / whitelist §A).** The device
  `DataflowBuffer` exposes a `constexpr get_tile_size()` member; the reader/writer switched to it
  cleanly since the magic cb-id constant is gone.

## Friction

### Gaps
- **The accumulation reference port is stale against the live API.** The recipe recommends
  `akertesz/porting-experiment-accumulation-jun10` as the shape reference, and it *was* invaluable
  for overall structure. But its API spellings no longer compile on this branch:
  it uses `ComputeHardwareConfig{.math_fidelity = ..., .unpack_to_dest_mode = ...}` (the live type is
  a `std::variant<ComputeGen1Config, ComputeGen2Config>` with renamed fields `fpu_math_fidelity` /
  `unpack_modes`, built via `to_compute_hardware_config`), and its factory method is named
  `create_program_spec` — but `MetalV2FactoryConcept` requires the method be named exactly
  `create_program_artifacts` (`ttnn/api/ttnn/operation_concepts.hpp:91`). Following the recipe +
  live headers (rather than the reference) was necessary. The current on-branch
  `experimental/quasar/fold` factory was a better spelling reference for the run-args / tensor-args
  idioms.

### Confusion
- **"KernelRunArgs may be omitted for a kernel with no RTAs" (recipe Construct) vs. "A KernelRunArgs
  must be specified for ALL kernels" (`program_run_args.hpp:90`).** These read as contradictory. The
  compute kernels here have no runtime args (CTAs only). Initially I provided empty
  `KernelRunArgs{.kernel = X}` entries for them. On the post-build follow-up I switched to
  **omitting** compute from `kernel_run_args` entirely, to match the proven-passing
  `moreh_group_norm` port (`moreh_group_norm_program_factory.cpp:605-608` lists only reader + writer).
  Worth reconciling the recipe/header wording — the working reference omits RTA-less kernels.

## Post-build follow-up (first on-device run — root cause of the 39/39 failure)

The orchestrator's first on-device run failed all 39 tests with JIT compile errors in the kernels
(`'args'/'dfb'/'tensor' not declared`, `'get_vararg' not declared`). Investigation (documented so the
next porter doesn't repeat it):

- Those symbols come from the build-injected `kernel_args_generated.h` / `kernel_bindings_generated.h`
  (on the emulator, emitted by `emit_metal2_namespaces`, gated on the kernel being a Metal 2.0 kernel).
  Their absence means the kernels were built as **legacy** (non-metal2).
- `tt_metal/impl/metal2_host_api/program_spec.cpp:3071` sets `is_metal2_kernel = true`
  **unconditionally** for every kernel `MakeProgramFromSpec` creates. So legacy kernels prove
  `MakeProgramFromSpec` never ran for this op — i.e. the op dispatched through the legacy
  `create_descriptor` path, not the MetalV2 adapter.
- The dispatch is concept-based (`device_operation.hpp:227` `std::visit`): a factory satisfying
  `MetalV2FactoryConcept` routes to `MetalV2MeshWorkloadFactoryAdapter` → `MakeProgramFromSpec`. My
  factory satisfies exactly that concept (has `create_program_artifacts`, no `create_descriptor`,
  `program_factory_t` variant) — verified against the header — so the wiring is correct **in source**.
- **Root cause (binary, not source):** the loaded/built `build_Release/.../_ttnncpp.so` (mtime
  `2026-07-24 23:08`) still exports the *pre-port* `MorehMeanBackwardOperation::create_descriptor` and
  contains **no** `create_program_artifacts` symbol for this op (nor, in that same lib, for
  moreh_group_norm). It is **older than the ported source** (`*_program_factory.cpp` /
  `*_device_operation.hpp` mtime `2026-07-25 00:22`). So the test ran the stale legacy factory, whose
  `create_descriptor` emitted a legacy program pointing at the kernel *paths*; the JIT then read the
  **new** metal2 kernel sources on disk and compiled them without generated headers → the reported
  errors. group_norm "passing" is consistent: that stale lib also predates its port, so it ran
  group_norm's old legacy factory + old behavior.
- **Fix:** a clean rebuild of the ttnn C++ library (`_ttnncpp.so`) so the ported factory is compiled
  in. The incremental build that produced the tested lib did not include this op's factory TU. After
  rebuild, confirm `nm -C build_Release/.../_ttnncpp.so | grep MorehMeanBackward` shows
  `...MorehMeanBackwardProgramFactory::create_program_artifacts` and **no** `create_descriptor`.
  Also confirm the loaded `_ttnn.so`/`_ttnncpp.so` is this worktree's fresh build (a
  `ttnn/ttnn/_ttnn.so.bak-oct-debug` is present in the tree, and the documented shared-python_env
  editable-.pth trap can load another worktree's `_ttnn.so`; verify with
  `python -c "import ttnn._ttnn as m; print(m.__file__)"`).
- **Source change made this pass:** aligned the run-args to the proven group_norm shape — compute
  kernels (no RTAs) are now omitted from `kernel_run_args` rather than given empty entries. This is
  the only code change; it does not affect the stale-binary root cause but removes any latent
  `SetProgramRunArgs` difference from the known-good reference.
- **Local-DFB "identical WorkUnitSpec membership" (migration guide troubleshooting) vs. the
  disjoint-group split.** Reader/writer are members of *both* work units while each compute instance
  is in only one, so a shared DFB's producer and consumer do not have literally-identical WU
  membership. The accumulation reference and this op both rely on the *per-node* reading (one
  producer + one consumer instance per node), which is the correct interpretation. Flagging in case
  the validator enforces the literal wording — see Open items for the orchestrator.

## Open items for downstream

- **Cross-op kernel touches:** none. All three kernels are op-owned; no shared-kernel file was
  modified or forked. The shared `moreh_common.hpp` helpers were left untouched (already
  `DataflowBuffer`-typed); a future family-wide Metal 2.0 syntax rewrite of those helpers is a
  separate port-together unit, not triggered by this op.
- **Validator watch (orchestrator):** if `MakeProgramFromSpec` rejects the reader/writer-in-two-WUs
  topology on the literal "identical WorkUnitSpec membership" rule (see Confusion), that is a
  framework/doc discrepancy, not a porter error — the structure mirrors the recommended accumulation
  reference. No workaround should be improvised; surface it.
- **RTA→CRTA tidy-up (not done, out of scope):** the reader's `num_dim` and the three vararg blocks
  hold the same value on every node (they don't depend on the core), so they are morally common
  runtime args / common varargs. Left as per-node RTAs/varargs to preserve legacy dispatch semantics
  faithfully; a later cleanup pass could demote them to CRTAs for dispatch efficiency.

## Verification (for the orchestrator)

Build (Metal + all TTNN test binaries):
```
./build_metal.sh --build-tests
```

The op's correctness tests (Python; the backward coverage lives alongside the forward in the moreh
`test_moreh_mean.py`). No C++ gtest and no dedicated `test_moreh_mean_backward.py` exists — coverage
is the `*_backward*` cases in:
```
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py -k backward -v
```
This exercises the no-regression baseline for this port:
- `test_moreh_mean_backward_ttnn_dtype` — dtype/shape/keepdim sweep
- `test_moreh_mean_backward_compute_kernel_options` — compute-kernel-config (fp32_dest_acc / fidelity) sweep
- `test_moreh_mean_backward_callback` — program-cache (cache-hit re-patch) behavior
- `test_moreh_mean_backward_create_input_grad` — with/without a preallocated `input_grad`

(The test set was discovered, not invoker-confirmed; the recipe's confirm-with-invoker checkpoint
could not be exercised in this delegated run. If a broader moreh_mean_backward test tree exists,
add it to the baseline.)
