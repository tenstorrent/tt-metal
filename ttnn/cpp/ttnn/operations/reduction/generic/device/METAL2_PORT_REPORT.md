# Port Report — reduction/generic op family

Second test-drive of the Metal 2.0 op-porting recipe, resumed after the catalog
fix on `7911c235273` restored the fork-with-`_metal2`-suffix path for cross-op
shared dataflow kernels. All four program factories — `ReduceMultiCoreWProgramFactory`,
`ReduceMultiCoreHProgramFactory` (interleaved + width-sharded branches),
`ReduceSingleCoreHwProgramFactory`, `WelfordReduceProgramFactory` (W/H/HW
variants) — ported to `ProgramSpecFactoryConcept`. Builds clean.
`tests/ttnn/unit_tests/operations/reduce/` runs 2260 passed / 63 skipped / 80
xfailed / 0 failed.

## Handoff points

### 1. None — boundary-rule assumptions held

No call site outside the op directory required passing `sem::name` or
`ta::name`. The porter's writeable surface (op directory + the two
`_metal2`-suffixed cross-op writer forks per the [shared-dataflow-kernel
Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel))
was sufficient to land the port.

Tagged "API: no escalations".

## Successes

### V1. Audit's borrowed-memory DFB classification carried into a working sharded port

[`port_op_to_metal2_audit.md` → Dynamic CircularBuffer LANDED entry](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_audit.md#dynamic-circularbuffer-cb-built-on-borrowed-buffer-memory--landed)
recognized the H factory's `use_width_sharding` branch as a borrowed-memory DFB
use case. The port landed in `reduce_op_multi_core_h_program_factory.cpp:165-200`
with `DataflowBufferSpec{.borrowed_from = H_INPUT_TENSOR}` and
`{.borrowed_from = H_OUTPUT_TENSOR}` for c_1 and c_3 respectively. No
`DFBRunParams` entries needed: the framework's per-Program `SetProgramRunParameters`
path attaches borrowed L1 addresses from the corresponding `TensorArg`
automatically. The width-sharded path completes end-to-end alongside the
interleaved path within the same `ReduceMultiCoreHProgramFactory`.

### V2. Plan template's per-variant breakdown survived contact with construction

[`port_op_to_metal2_recipe.md` → Appendix A](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/port_op_to_metal2_recipe.md#appendix-a--metal2_port_planmd-template)'s
"Multi-variant ops" structural convention (per-variant inventory + per-variant
Planned Spec Shape) carried cleanly into the construction step: each of the
seven variants (W, H interleaved, H width-sharded, HW, Welford W, Welford H,
Welford HW) became its own KernelSpec/WorkUnitSpec block in the factory `.cpp`
file. The factory code is grouped per-variant in a way that maps 1:1 to the
plan's organization, which kept the construction mechanical.

### V3. The shared-dataflow-kernel Caution's fork path is the right answer

[`metal2_port_patterns.md` → Caution: Modifying a shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel)
gave two paths: in-place co-migration, or fork-with-`_metal2`-suffix. The
co-migration path was unavailable (46 unmigrated consumers across the writer
families); the fork path worked cleanly. The forks live alongside their legacy
peers:
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  (referenced by 5 reduction-family variants: W, H interleaved, HW single-core,
  Welford W, Welford H)
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp`
  (referenced by H width-sharded)

The Caution's clarity that this is a *typical* answer during the bulk-port
window, not an exception, matched the situation. The sunset criterion ("delete
the legacy copy when the last unmigrated consumer ports") is recorded under
[Open items for downstream](#open-items-for-downstream).

### V4. Pass-DFB-handles-directly pattern collapsed the LLK-call-site decisions

[`metal2_port_patterns.md` → Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
made every LLK/helper call site straightforward. Examples:

- `compute_kernel_hw_startup(dfb::input, dfb::scaler, dfb::output)` in
  `kernels/compute/reduce.cpp:29`.
- `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, ...>(dfb::input, dfb::scaler, dfb::output, ...)`
  in `kernels/compute/reduce.cpp:31-49`.
- `dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f)`
  in `kernels/dataflow/reader_unary_reduce_universal_start_id.cpp:29` and
  `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:46`.
- `reduce_init<REDUCE_OP, REDUCE_DIM>(dfb::ineg, dfb::scaler, dfb::acc)` and
  the matching `reduce_tile<>` in `kernels/compute/reduce_w_neg.cpp:80-81`.

Without the implicit `DFBAccessor::operator uint32_t()` conversion, each of
these would have demanded a `.id` extraction or a temp wrapper. The pattern's
"there is no extraction; the conversion is automatic" framing aligned with
what the call sites required.

### V5. Self-loop and conditional-binding patterns carried the negate + do_scale combinatorics

The reduce-with-negate variants need ACC and INEG self-loops on the compute
kernel. The Welford W variant adds a self-loop on `scratch` (and conditionally
on `scaled`). All apply
[Pattern: Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding)
cleanly — the compute kernel binds the DFB as both PRODUCER and CONSUMER, the
host adds two bindings with the same `dfb_spec_name`. Examples:
`reduce_op_multi_core_w_program_factory.cpp:301-318` (W reduce negate),
`welford_reduce_program_factory.cpp:430-440` (Welford W scratch + scaled).

The conditional-binding pattern carries the rest:
- Negate-only DFBs (`acc`, `ineg`) added only when `operation_attributes.negate`.
- Welford W `scaled` DFB added only when `do_scale`.
- Welford HW partial/combined DFBs added only when `reduce_dim == HW`.

### V6. Dropped Plumbing enumeration vindicated again — borrowed-memory DFB run-params handling

The W factory port skipped writing per-core DFB-address run params for the
borrowed DFBs (H factory width-sharded path). The plan's
[Dropped Plumbing table](METAL2_PORT_PLAN.md#dropped-plumbing) called this
out explicitly ("borrowed-memory DFBs update from TensorArg"), so the porter
knew not to look for that plumbing.

## Friction

### Gaps

#### G1. Hash needs `tensor_args.logical_shape()`, not just `padded_shape()`

[`metal2_migration_guide.md`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_migration_guide.md)
and the recipe don't currently mention the cache-key shape requirement, but the
framework's `UpdateTensorArgs` (in
`tt_metal/impl/metal2_host_api/program_run_params.cpp:51-56`) requires *full
TensorSpec equality* (not just padded-shape match) between the cached program's
expected TensorParameter spec and the runtime tensor. The Welford
device-operation hashes `logical_shape()` alongside `padded_shape()` (per
`welford_reduce_device_operation.cpp:101`); the reduce device-operation did not
on initial port. Test failure: `test_std[dim=None-...]` runs a sequence of W
and H reductions where the W-output's logical_shape differs from input's even
when padded_shape doesn't — the cache hit fires but the validation rejects the
spec mismatch.

Fix landed at `reduce_op_device_operation.cpp:166-172`. What would help: the
migration guide's "TTNN Framework Integration" or "Design Principles" gain a
section: "compute_program_hash must include `tensor_args.logical_shape()` (in
addition to `padded_shape()` and any other shape components) because the
framework enforces full TensorSpec equality at cache-hit." Same note belongs
in the recipe (the "Plan the spec" or "Construct paired spec + run-params"
step) since this is a structural requirement, not a feature note.

#### G2. `unpack_to_dest_mode` entries required for FP32 DFBs consumed by compute kernels

The framework requires
([`program_spec.cpp:822-830`](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp))
an explicit `unpack_to_dest_mode` entry on the `ComputeConfiguration` for
every FP32 DFB consumed by a compute kernel when `fp32_dest_acc_en=true`. The
docs don't mention this. The error message is good — it explains both choices —
but absent docs, the porter writes the spec, fails verification, and back-fills.

In the reduction port, every compute kernel of every variant needs entries for
each FP32-when-the-input-is-FP32 DFB it consumes. The factories now compute
the entries dynamically:
`reduce_op_multi_core_w_program_factory.cpp:285-300`,
`reduce_op_multi_core_h_program_factory.cpp:417-433`,
`reduce_op_single_core_hw_program_factory.cpp:275-292`,
`welford_reduce_program_factory.cpp:454-477` (the Welford HW combined DFB is
always Float32 regardless of input dtype, separately handled).

What would help: catalog [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings)
or a new "Compute config with FP32 DFBs" pattern entry, explaining the rule
and showing the dynamic-build idiom.

#### G3. Intra-tensix self-loop DFBs need `disable_implicit_sync = true`

`dataflow_buffer.cpp:636-638` rejects intra-tensix DFBs with implicit sync
enabled. Implicit sync is *enabled by default* (`disable_implicit_sync = false`
on `DataflowBufferSpec`). Self-loop DFBs on compute kernels are intra-tensix.
The default-on-implicit-sync combined with the default-allow-self-loop produces
a runtime error on every self-loop DFB.

Fix: set `disable_implicit_sync = true` on each self-loop DFB declaration. The
[Pattern: Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding)
catalog entry should call this out as a required field for compute-self-loop
DFBs.

Sites: `reduce_op_multi_core_w_program_factory.cpp:165-180` (acc/ineg),
`reduce_op_multi_core_h_program_factory.cpp:266-279` (acc/ineg),
`reduce_op_single_core_hw_program_factory.cpp:167-181` (acc/ineg),
`welford_reduce_program_factory.cpp:185-200` (scratch + scaled).

#### G4. `if constexpr` on a CTA does not eliminate references to optionally-bound DFBs

The [Anti-pattern: Always-bind optional DFB + gate-uses-only](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-always-bind-optional-dfb--gate-uses-only)
recommends, as its "third option," "conditionally bind on the host + `if
constexpr` on a CTA gating *both declaration and uses* in the kernel." This
doesn't actually compile for the Welford W `scaled` DFB:

```cpp
if constexpr (do_scale) {
    DataflowBuffer cb_scaled_obj(dfb::scaled);  // dfb::scaled isn't defined when host doesn't bind it
    // ...
}
```

The framework's `kernel_args_generated.h` emits `dfb::scaled` only when the
SCALED DFB is bound. When `do_scale=false`, the binding is conditionally
omitted; `dfb::scaled` is then undefined in the generated header, and the
compiler errors out even though the code path is dead (C++ parsing rule:
`if constexpr` discards statements but names in the discarded statement still
have to be defined).

Workarounds:
- **Adopted**: use `#ifdef DO_SCALE` instead of `if constexpr (do_scale)`,
  with a host-side `defines["DO_SCALE"] = "1"` when do_scale is true. Each
  reference to `dfb::scaled` is fully gated out at the preprocessor level. See
  `welford_reduce_w.cpp:81-130` and the matching factory change at
  `welford_reduce_program_factory.cpp:152-156`.
- Alternative not taken: split into `welford_reduce_w_scaled.cpp` and
  `welford_reduce_w_unscaled.cpp` separate sources. More invasive; preferred
  the `#ifdef` since it stays in one file.

What would help: the catalog's [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings)
entry gain a note: "The kernel can use `if constexpr` to gate the *uses* of a
conditional DFB, but the *name* `dfb::<conditional>` must exist whenever the
kernel source mentions it — even inside discarded `if constexpr` branches.
When the DFB is bound on a subset of kernel-spec configurations and the kernel
source can't easily refer to it without the name being defined, use `#ifdef`
+ a host-side define instead of `if constexpr`."

#### G5. Welford W/H compute kernels need a CONSUMER on the scaler DFB unconditionally

The reader kernels (both `reader_unary_reduce_universal_start_id.cpp` and
`reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`)
unconditionally call `dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, ...>`,
which writes a scaler tile to the SCALER DFB. So SCALER always has a PRODUCER.

The Welford compute kernels only *read* the scaler when `do_scale` is true. If
the compute kernel only binds SCALER as CONSUMER when `do_scale`, the framework
rejects the spec: "DFB 'scaler' has no consumer" when `do_scale=false`.

Fix: always bind SCALER as CONSUMER on the compute kernel, and always
`wait_front(1)` on it in the kernel (even when do_scale=false). The tile is
produced and discarded; the L1 cost is one scalar tile. The kernel doesn't
otherwise touch the value when do_scale=false.

Sites: `welford_reduce_program_factory.cpp:404-411` (binding),
`welford_reduce_w.cpp:75`, `welford_reduce_h.cpp:52`, `welford_reduce_hw.cpp:80`
(unconditional wait_front).

What would help: the recipe (Construct paired spec + run-params) gain a note
on the producer/consumer balance rule: "Every DFB declared in the ProgramSpec
must have at least one PRODUCER and one CONSUMER. If a reader unconditionally
writes to a DFB that the compute kernel only sometimes reads, the compute
kernel must still bind the DFB as CONSUMER and consume the tile (e.g.,
`wait_front` + ignore the value)."

#### G6. Multi-variant compute kernels need explicit `runtime_arguments_schema`

Each Welford compute variant (W/H/HW) takes a single RTA (`NCHt` / `NCWt` /
`NC_per_core`). The factory's `make_compute` lambda needs to set
`runtime_arguments_schema.named_runtime_args` accordingly, *per variant*.
Missing this triggers `program_run_params.cpp:167`: "Kernel 'compute_g1' node
0-0 expects 0 named RTAs, but 1 were provided".

This is mechanically obvious in hindsight but it caught the port by surprise
because the W and H non-Welford reduce factories have *zero* RTAs on the
compute kernel (all CTAs), so it's tempting to leave the field unset on the
shared lambda.

Fix landed at `welford_reduce_program_factory.cpp:419-422`. What would help:
the catalog [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories)
entry gain a sub-bullet on the variant-specific RTA-schema requirement,
especially when shared `make_compute` helper sees the variant flag and needs to
branch on the schema name.

### Confusion

#### C1. Inferring DFB lifecycle from CB lifecycle in legacy code

Several legacy CBs are sized at runtime based on shape — e.g. the H factory's
borrowed-memory CB c_1 sized at `num_shard_tiles`, and the negate CBs sized at
`per_cb_total_size`. In Metal 2.0, sizes live in `DataflowBufferSpec`
(`entry_size`, `num_entries`), and `DFBRunParams` can override sizes per
execution.

The recipe's plan template explicitly lists DFB size in the inventory, but the
distinction between "size known at spec-construction time" (no DFBRunParams
needed) vs "size depends on a runtime parameter the spec doesn't know" (use
DFBRunParams override) is fuzzy. For this op, all sizes are computable at
spec-construction time from `tensor_args.shard_spec()`, so no DFBRunParams.

What would help: the recipe (or the migration guide DataflowBufferSpec section)
gain a one-line decision criterion: "If the size depends only on things the
spec construction has access to (input tensor's shard spec, attrs from
`operation_attributes`), set the size directly on the DataflowBufferSpec. Use
DFBRunParams only when the size needs to change between executions of the
*same* cached Program."

## Open items for downstream

### Cross-op kernel touches

- **`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`**
  - Path taken: **fork** with `_metal2` suffix
  - Fork path: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  - Remaining unmigrated consumers (as of May 2026): 33 in-tree `.cpp` files
    (grep `writer_unary_interleaved_start_id\.cpp ttnn/cpp/ttnn/`). Sunset:
    delete the legacy copy when all 33 port to Metal 2.0.
  - Drift discipline applies: any bug fix to the legacy copy during the bulk-port
    window should be evaluated for the `_metal2` copy.

- **`ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`**
  - Path taken: **fork** with `_metal2` suffix
  - Fork path: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp`
  - Remaining unmigrated consumers (as of May 2026): 13 in-tree `.cpp` files.
    Sunset: delete the legacy copy when all 13 port to Metal 2.0.
  - Drift discipline applies as above.

The Welford HW writer (`writer_welford_hw.cpp`) is op-local and was modified
in-place (no fork).

### Per-op carry-over

- Welford W/H reader is shared with non-Welford reduce. Both use the same
  in-op reader sources (`reader_unary_reduce_universal_start_id.cpp` for W
  branches, `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp`
  for H/HW branches). The in-op readers are co-migrated as a single set — no
  cross-op forks needed. This pattern (in-op reader, cross-op writer fork)
  is likely typical for reduction-family ops; the next porter can expect to
  hit the same shape.

- The legacy code has `*_neg.cpp` compute kernels for negate as separate
  sources (`reduce_w_neg.cpp`, `reduce_h_neg.cpp`, `reduce_hw_neg.cpp`). The
  port preserves this separation; the host selects between `reduce.cpp` and
  `reduce_*_neg.cpp` based on `operation_attributes.negate`. The negate kernels
  use the ACC/INEG self-loop DFBs that the non-negate kernels don't bind.

### Doc-evolution suggestions

- See G1-G6 (above) for the principal gaps. Most actionable: G2 and G3 (FP32
  unpack mode + intra-tensix self-loop disable_implicit_sync) are framework
  requirements that bite *every* multi-variant compute-kernel port. Recommend
  adding both to the catalog and the migration guide.

- C1: a short DFB-size-vs-DFBRunParams decision criterion would close a gap
  the recipe currently asks porters to infer.

### Test coverage notes

- `tests/ttnn/unit_tests/operations/reduce/` runs 2260 passed / 63 skipped /
  80 xfailed / 0 failed against this port.
- Test scope: every variant of the four factories is exercised (W, H
  interleaved, H width-sharded, HW single-core, Welford W, Welford H, Welford
  HW; all with and without negate / scaling / correction / fp32_dest_acc_en /
  post-mul). The first test-drive's `test_reduction_program_cache.py` and
  `test_reduction_h_interleaved.py` pass alongside the broader suite.
- Subagent did not run nightly or perf tests; those remain unverified.
- Subagent did not run any test that uses the dim=None or multi-dim path
  through `tt_transformers` or any model. Those are next-level verification.

### Open API ergonomics

- Welford device-operation hash includes both `logical_shape` and
  `padded_shape`; the reduce device-operation hash was missing
  `logical_shape` until this port fixed it. The framework's `UpdateTensorArgs`
  spec-equality contract is a structural requirement on every Metal 2.0
  device-operation's `compute_program_hash`. Worth a separate audit of all
  other in-tree device-operations on the Metal 2.0 path to make sure they
  hash all spec components.
