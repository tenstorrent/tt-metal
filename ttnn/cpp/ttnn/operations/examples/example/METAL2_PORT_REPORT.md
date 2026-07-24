# Metal 2.0 Port Report — `examples/example` (`ExampleDeviceOperation`)

## Outcome

**`CAPITULATED`** — a grounded, in-scope stop (success-tier). Both factories (`SingleCore`, `MultiCore`)
bind three kernels that live **outside the op directory**, and every recipe-sanctioned way to give
those kernels Metal 2.0 entry points is closed by the orchestration scope constraint. No factory `.cpp`
was modified — a Metal 2.0 factory that binds unconverted (positional-arg) kernels does not build and is
not a deliverable. The full design blueprint is captured in `METAL2_PORT_PLAN.md`; the port is mechanical
to finish once the cross-op shared-kernel rewrite lands (see Handoff points).

No build or tests were run by the porter (per orchestration: the orchestrator builds and runs tests
serially). Verification commands are listed below for when the blocker is resolved.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

- **Concept targeted (from audit):** `MetalV2FactoryConcept` — not realized (port capitulated before construction).
- **Custom `compute_program_hash` deletion:** none required (op uses the default reflection-based hash).
- **Pybind entry points removed:** none — `example_nanobind.cpp` binds only the free function
  `composite_example`; there is no pybound `create_descriptor` to remove.
- **Device-op-class edits forced:** none. `select_program_factory`, `validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors` are all outside the factory body and were not touched.
- **Open items:** the concept fit is clean; the only obstacle is the out-of-directory kernel dependency
  (below), which is orthogonal to the concept.

## Handoff points

### 1. PORT CAPITULATION — cross-op shared kernels outside the writeable scope (blocker)

- **Op / factory:** `ttnn/cpp/ttnn/operations/examples/example` — `ExampleDeviceOperation::SingleCore`
  and `::MultiCore` (both).
- **Files / constructs that needed to change but could not:** all three kernels the factories bind, none
  of which is in the op directory:
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
    (positional `get_arg_val<uint32_t>(0..2)`, `TensorAccessorArgs<0>()`, `constexpr uint32_t cb_id_in0 = 0`).
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
    (positional `get_arg_val<uint32_t>(0..2)`, `get_compile_time_arg_val(0)`, `TensorAccessorArgs<1>()`).
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
    (hardcoded `cb_input = c_0`, `cb_output = c_2`, positional `get_arg_val<uint32_t>(0)`).
- **Why mechanical conversion failed:** the port's atomic unit is the factory + every kernel entry point
  it binds. A `MetalV2FactoryConcept` factory emits only named bindings; the kernels must read
  `tensor::`/`dfb::`/`args::` for the factory to launch them (a positional `get_compile_time_arg_val(0)`
  static_asserts at JIT once the host emits only named args). The three kernels are Device 2.0 but not
  Metal 2.0. Converting them means editing out-of-directory shared kernels — forbidden by the orchestration
  constraint ("only modify files under the op directory; do not edit shared kernels outside it"). Forking
  them into the op directory is forbidden by both the audit brief ("do NOT fork local copies into this op's
  directory") and the same constraint. No `_metal2` rewritten copies exist on this branch to consume.
  Both sanctioned cross-op paths (in-place co-migration / `_metal2` fork) are therefore closed. This is the
  recipe's most-common stop signal — "reaching past the op's own directory to make kernel changes."
- **Shared-kernel blast radius (co-migration coordination signal):**
  - `reader_unary_interleaved_start_id.cpp` — **19** consumer ops under `ttnn/cpp/ttnn/operations/`.
  - `writer_unary_interleaved_start_id.cpp` — **48** consumer ops.
  - `eltwise_sfpu.cpp` (compute) — **4** consumer ops.
  These counts are why in-place co-migration in this PR is not viable and why the eltwise/unary
  shared-kernel rewrite must be a separately owned, coordinated change.
- **What the off-rules change would have been (for a maintainer to evaluate the gap):** rewrite the three
  `eltwise/unary` kernels to Metal 2.0 (drop `src_addr`/`dst_addr` RTA reads and `TensorAccessorArgs<N>()`
  in favor of `TensorAccessor(tensor::input|output)`; replace the writer's `get_compile_time_arg_val(0)`
  CB index and the compute kernel's `cb_input`/`cb_output` with `dfb::` handles; name the `num_pages` /
  `start_id` / `num_tiles` args), OR land `_metal2`-suffixed forks of them under `eltwise/unary`. Once
  either exists, this op's factory port is a mechanical one-pass application of `METAL2_PORT_PLAN.md`
  (spec shape, dropped plumbing, and hw_config are all fully specified there).
- **Owner:** the `eltwise/unary` shared-kernel port / kernel-lib coordination — not the porter of this op.

## Successes

- **The scope-boundary rule fired exactly as designed.** `metal2_port.md` §"When the discipline doesn't
  fit" ("the stop-signal we see most often: if you find yourself reaching past the op's own directory to
  make kernel changes") and the shared-dataflow-kernel Caution
  (`port_patterns.md` §"Caution: Modifying a shared dataflow kernel") together made the correct action
  unambiguous: capitulate and hand off rather than edit or fork the shared kernels. The docs prevented a
  bundled cross-op kernel change that would have broken up to 48 sibling ops' CTA layouts.
- **The audit's "Watch for" item was accurate and load-bearing.** `METAL2_PORT_BRIEF.md` flagged all three
  kernels as broadly-shared `eltwise/unary` borrows with "do NOT fork / coordinate with the eltwise/unary
  shared-kernel port." That flag is exactly what materialized as the blocker.

## Friction

### Gaps
- **The audit cleared GREEN and issued detailed "Construct — to do" kernel-rewrite steps, yet the port is
  not completable in isolation.** The brief simultaneously tells the porter to rewrite the reader/writer
  kernels ("replace `src_addr = get_arg_val<uint32_t>(0)` ... with `TensorAccessor(tensor::name)`") *and*
  to not fork them and instead "coordinate with the eltwise/unary shared-kernel port" or "consume the
  already-rewritten kernels." When those rewritten kernels do not yet exist and the op has **zero**
  in-directory kernels of its own, the op is not independently portable. Suggestion: the audit's
  `Is able to port?` gate (or a new pre-port gate) should treat "every bound kernel is an unmigrated
  cross-op shared kernel" as a **sequencing dependency that blocks the brief** (or at least downgrades it
  to AMBER with an explicit "depends on eltwise/unary shared-kernel port" precondition), rather than GREEN.
  As written, a GREEN brief lands on a porter who then must capitulate on the first structural step.

### Confusion
- The recipe lists cross-op kernels as "porter-touchable with caution," while the orchestration constraint
  (which overrides) forbids touching anything outside the op directory. A porter reconciles this correctly
  by capitulating, but the two-sentence gap between "porter-touchable" (recipe) and "do not touch" (this
  run's constraint) is a place a less careful porter could wrongly edit a shared kernel. Not a doc defect
  per se — the constraint explicitly overrides — but worth a one-line note in the recipe that a bulk-port
  orchestrator may narrow the writeable surface below what the Caution allows.

## Open items for downstream

- **Cross-op kernel touches:** none made (capitulated). The three `eltwise/unary` kernels above remain
  unmodified and unforked; they are the coordination point for this op's eventual port.
- **Dead code in the op directory (non-port cleanup, route to ops team):**
  - Unreferenced/stale kernel files: `device/kernels/compute/eltwise_sfpu.cpp`,
    `device/kernels/dataflow/{blank,reader_binary_diff_lengths,reader_unary,writer_unary}.cpp` — no factory
    references them (both factories point at the `eltwise/unary` paths). Likely leftover tutorial
    scaffolding. Left untouched (out of port scope).
  - Unused attribute: `operation_attributes_t::some_other_attribute` (`device/example_device_operation.hpp:22`),
    set to `42` in `ttnn::prim::example` (`device/example_device_operation.cpp:42`) but never read. Dead;
    not hashed. Left untouched.
- **When unblocked:** finishing the port is a single mechanical pass from `METAL2_PORT_PLAN.md` — 3 KernelSpecs,
  2 DataflowBufferSpecs (both 1P+1C), 2 TensorParameters, no semaphores, the dropped-plumbing table, and the
  Style-B compute hw_config — applied identically to `SingleCore` and `MultiCore`.

## Verification commands (to be run by the orchestrator once the blocker is resolved and the port lands)

Build (Metal + all TTNN test binaries):
```bash
./build_metal.sh --build-tests
```

Run the op's tests (no-regression baseline — confirm this set with the invoker; located by broad sweep,
below, then filtered to this op):
```bash
# Locate the op's tests (this op is the tutorial "example" op):
find tests -iname '*example*' -name '*.py'

# C++ gtests for the example op:
./build/test/ttnn/unit_tests_ttnn --gtest_filter='*Example*'

# Python pytests (whichever of the located files exercise ttnn.example / composite_example):
pytest tests/ttnn/unit_tests/operations/ -k example -x -v
```
Because the port capitulated, no test set was confirmed with the invoker. The exact pytest path(s) must be
confirmed against the `find` sweep above before relying on them as the no-regression baseline.

**Build/test verification: to be performed by the orchestrator** (not run by the porter, per orchestration
constraints).
