# Metal 2.0 Port Report — moreh_layer_norm

## Outcome

**PORTED** — the single `ProgramFactory` (both the small and large algorithm branches, which are
runtime-selected within one factory) converted from the TTNN `descriptor` concept to
`MetalV2FactoryConcept` (`create_program_artifacts`). Build + test verification is the orchestrator's
(this porter did not build or run tests, per orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — as the audit decided. `program_factory_t` remains a single-alternative
`std::variant<ProgramFactory>`; the framework auto-routes the single alternative to the MetalV2
adapter (this op has no `select_program_factory`, and none was added — a single-factory op does not
need one).

### Device-op-class edits
- Custom `compute_program_hash` deleted: **none** — the op never had one (default reflection hash).
- Pybind entry points removed: **none** — the nanobind file binds only the op function
  (`ttnn::bind_function<"moreh_layer_norm">`); there was no pybind `create_descriptor` or
  factory-hook parameter to unwind. `moreh_layer_norm_nanobind.cpp` is unchanged.

The only device-op-class change is the mandatory factory-method signature swap in
`device/moreh_layer_norm_device_operation.hpp`: `ProgramFactory::create_descriptor` (returning
`tt::tt_metal::ProgramDescriptor`) → `create_program_artifacts` (returning
`ttnn::device_operation::ProgramArtifacts`), plus swapping the `<tt-metalium/program_descriptors.hpp>`
include for `"ttnn/metal_v2_artifacts.hpp"`. `validate`/`compute_output_specs`/`create_output_tensors`/
`invoke` are untouched.

### Open items
- **Relaxation candidates:** none applied. Tensor matching left strict (default).
- **RTA→CRTA cleanup (deferred, not port work):** the reader's `scaler`, `eps`, `mask_h`, `mask_w`
  RTAs and the writer's `mean_rstd_height`, `mean_rstd_width`, `normalized_dims` RTAs hold the **same
  value on every node** — they are morally CRTAs. Kept as per-node RTAs here (converting RTA→CRTA
  changes dispatch semantics and is out of scope for a syntax-swap port). A future cleanup pass could
  demote them.

## Handoff points

None — the port stayed entirely within the op directory. No `sem::`/`tensor::` boundary violations
(the op has no semaphores; all tensor accessors are consumed inside the op's own kernels). No
kernel-lib gaps: the shared header pools (`ttnn/kernel/dataflow/moreh_common.hpp`,
`ttnn/kernel/compute/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`) were
already Device 2.0 native (take `DataflowBuffer` / plain-`uint` args, and `reduce<>`'s CB-id NTTPs
accept `dfb::` handles) and needed no edits.

## Successes

- **DFB per-node invariant (recipe [Construct] + dataflow_buffer_spec.hpp:41-50).** The two compute
  KernelSpecs over disjoint `core_group_1/2` bind the reader-fed DFBs (input/scaler/eps/…) as
  CONSUMER over non-overlapping node sets. The header's explicit allowance — "you MAY bind more than
  one KernelSpec to a producer (or consumer) endpoint … non-overlapping node coverage … same kernel
  kind … identical binding-site parameters" — is exactly this case, so no `allow_instance_multi_binding`
  was needed (the audit predicted this; confirmed). The `reduce_op_multi_core_h` metal-v2 factory is
  the structural precedent for reader+writer-in-both-work-units + compute-group-split.
- **Preserve-multiplicity (recipe anti-pattern: Demoting per-group CTA to RTA).** `num_rows_per_core`
  stayed a per-group CTA across two compute KernelSpecs; was not demoted to an RTA on a single spec.
- **Conditional / optional DFB bindings (recipe rule 6 + migration guide Principle 2).** The
  optional resources (gamma/beta/mask_h/mask_w/mean/rstd/gamma_beta) are host-conditionally bound,
  matched by `#define`s, and `#ifdef`-gated in the kernels — including the file-scope
  `cb_gamma_beta_or_out` / `cb_outg` ternaries that reference the optional `c_30` DFB, which the
  recipe specifically warns must be `#ifdef`-gated rather than `if constexpr`-gated.

## Friction

### Gaps
- **`dfb::` as a non-type template parameter is undocumented but works.** The shared compute helper
  `compute_kernel_lib::reduce<PoolType, ReduceDim, uint32_t input_dfb_id, uint32_t scaler_dfb_id,
  uint32_t output_dfb_id>` takes CB ids as **NTTPs**, not runtime args. The recipe/migration guide
  document `dfb::name → uint32_t` only for *runtime* LLK call sites (`reduce_init(dfb::in, …)`); they
  don't mention NTTP use. It works because `DFBAccessor::operator uint32_t()` is `constexpr`
  (`dataflow_buffer.h:55`), making `dfb::xsum` a valid converted-constant-expression template
  argument — so `reduce<REDUCE_OP, REDUCE_DIM, dfb::xsum, dfb::scaler, dfb::ex>(…)` compiles unchanged.
  Worth a one-line note in the CB→DFB whitelist / patterns catalog, since a porter hitting a
  CB-id-NTTP helper has no documented assurance this is legal.

### Confusion
- **"Local DFB invariant: identical WorkUnitSpec membership" (migration guide troubleshooting) vs the
  header's per-node rule.** The troubleshooting line reads as if a DFB's producer and consumer
  kernels must have *identical* WorkUnitSpec membership. Here `reader` is in both WU_g1 and WU_g2 while
  `compute_g1` is only in WU_g1, so their membership *sets* differ — yet the port is correct because
  the authoritative header invariant (dataflow_buffer_spec.hpp:41-58) is a per-*node* rule (one P + one
  C instance per node), which holds. The precedent factory (`reduce_op_multi_core_h`) does the same.
  The troubleshooting wording could be tightened to say "co-located on the same nodes" rather than
  "identical membership".

## Open items for downstream

- **Cross-op kernel fork (coordination signal for the next moreh port).** The two **compute** kernel
  sources are borrowed by file path by the not-yet-ported legacy peer `moreh_group_norm`
  (`moreh_group_norm_program_factory.cpp:251-252`). Per the orchestration constraint they were
  **forked**, not modified in place:
  - `device/kernels/moreh_layer_norm_small_kernel.cpp` → fork `…_small_kernel_metal2.cpp` (legacy
    original untouched).
  - `device/kernels/moreh_layer_norm_large_kernel.cpp` → fork `…_large_kernel_metal2.cpp` (legacy
    original untouched).
  Remaining unmigrated consumer of the legacy originals: `moreh_group_norm`. **Sunset checklist:** when
  `moreh_group_norm` is ported to Metal 2.0, it can either adopt these `_metal2` forks or gain its own;
  once no legacy consumer remains, the two legacy compute originals can be deleted. The reader/writer
  sources are owned solely by this op and were converted in place (no fork).
- **Dead kernel variables (not cleaned — scope discipline).** `input_data_format` (both readers;
  relocated onto the DFB object per whitelist rule 7 but still unused), and `offs`/`onetile` in the
  small reader remain dead, exactly as in the legacy sources. The audit flagged these as an optional
  ops-team cleanup; not bundled into this port.
- **`moreh_group_norm` is a natural next port** — it shares the same compute kernels (now available as
  `_metal2` forks) and the same reader/writer *shape*, so much of this port's structure carries over.

## Files created / modified (all under the op directory)

Created:
- `device/kernels/moreh_layer_norm_small_kernel_metal2.cpp` — Metal 2.0 fork of the small compute
  kernel (dfb:: handles, named args, `#ifdef`-gated optionals). Legacy original untouched (group_norm).
- `device/kernels/moreh_layer_norm_large_kernel_metal2.cpp` — Metal 2.0 fork of the large compute
  kernel. Legacy original untouched.
- `METAL2_PORT_PLAN.md` — port plan (inventory + spec plan).
- `METAL2_PORT_REPORT.md` — this report.

Modified:
- `device/moreh_layer_norm_device_operation.hpp` — `create_descriptor` → `create_program_artifacts`;
  include swap (`program_descriptors.hpp` → `metal_v2_artifacts.hpp`).
- `device/moreh_layer_norm_program_factory.cpp` — rewritten to build `ProgramSpec` + `ProgramRunArgs`
  (DFB specs, kernel specs, tensor parameters, two work units, name-first RTA tables); repoints the
  compute source at the `_metal2` forks. All op parameter/shape/work-split/config computation preserved
  verbatim from the legacy factory.
- `device/kernels/reader_moreh_layer_norm_small.cpp` — converted in place (op-owned): dfb:: handles,
  `tensor::input/gamma/beta`, named args, metadata off the DFB object.
- `device/kernels/reader_moreh_layer_norm_large.cpp` — converted in place (op-owned).
- `device/kernels/writer_moreh_layer_norm.cpp` — converted in place (op-owned): `dfb::output/mean/rstd`,
  `tensor::output/mean/rstd`, named args, `#ifdef MEAN_HAS_VALUE`/`RSTD_HAS_VALUE`; `write_mean_rstd`
  now takes the `DataflowBuffer` object instead of a CB id.

Untouched legacy (shared with moreh_group_norm — DO NOT convert in place):
- `device/kernels/moreh_layer_norm_small_kernel.cpp`
- `device/kernels/moreh_layer_norm_large_kernel.cpp`

## Test command(s) — verification is the orchestrator's

Build (Metal + TTNN tests):
```bash
./build_metal.sh --build-tests
```

Correctness tests (this op's confirmed coverage — pytest; moreh_layer_norm has no dedicated C++
gtest). Run bfloat16/tiled shapes with/without gamma/beta and with/without mean/rstd outputs so both
the small and large algorithm branches and all conditional-binding combinations are exercised:
```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py -x -v
```

> **Test-set note:** a broad sweep (`find tests -iname '*moreh_layer_norm*' -name '*.py'`, excluding
> `moreh_layer_norm_backward`) found exactly one coverage file:
> `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py` (it lives under the
> **nightly** tree, not the plain `unit_tests` tree). No dedicated C++ gtest and no sweep entry. The
> porter did not build/run (per orchestration constraints) and did not get an explicit invoker test-set
> sign-off — the orchestrator should treat this file as the no-regression baseline and confirm it
> exercises both algorithm branches and the optional-arg combinations.

## Anti-pattern self-audit (static, porter-run)

Grep over the op directory (converted files only; legacy shared compute originals excluded):
- No `tensor.buffer()->address()`, no `TensorAccessorArgs`, no `get_arg_val`/`get_compile_time_arg_val`,
  no `tt::CBIndex`, no `CircularBuffer`/`CBDescriptor`/`ProgramDescriptor`/`create_descriptor` survive in
  the converted host factory or the converted/forked kernels.
- No `.id` extraction, no `get_vararg`, no `allow_instance_multi_binding`, no `ProducerOf`/`ConsumerOf`
  convenience factories (full designated-initializer `DFBBinding` form used throughout).
- All CTAs named; all RTAs named; hw_config carries the legacy resolved values (reader/writer default
  DM configs via the arch-agnostic helper; compute via `to_compute_hardware_config` = the four legacy
  knobs; `unpack_modes` = `UnpackToSrc` on the Float32 intermediate DFBs when `fp32_dest_acc_en`, the
  validator-required entry that legacy defaulted).
