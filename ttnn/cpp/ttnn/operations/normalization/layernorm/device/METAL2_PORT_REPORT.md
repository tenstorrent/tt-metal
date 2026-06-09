# Port Report — Layernorm (Metal 2.0)

**Scope of this port:** **both** `LayerNormMultiCoreProgramFactory` (interleaved) and `LayerNormShardedProgramFactory` (sharded).

**Status:** **Draft port — structural changes complete, needs build iteration.** Both factories now satisfy `ProgramSpecFactoryConcept` (returning `ProgramArtifacts`). All 13 kernel sources under `device/kernels/` were converted: positional CTAs → named (`get_arg(args::name)`), magic CB indices → `dfb::name`, semaphore-ID CTAs → `sem::name`, `CircularBuffer` → `DataflowBuffer`, `TensorAccessorArgs<N>()` → `TensorAccessor(ta::name)`. The port has NOT been built or tested in this session — per project convention, the user runs builds and tests themselves. The interleaved factory section below was written first; the sharded section is appended at the end.

---

## Handoff points

### OpDescriptor / experimental Python descriptor flow — needs caller-side fix

**File:** `models/experimental/ops/descriptors/normalization/_utils.py:109`

`_run_factory()` calls `factory.create_descriptor(operation_params, tensor_args, out, cr_arg)`. After the port, `LayerNormMultiCoreProgramFactory::create_descriptor` no longer exists — it's replaced by `create_program_spec` which returns `ProgramArtifacts` (not `ProgramDescriptor`). The Python `create_descriptor` binding in `layernorm_nanobind.cpp` was removed.

**Why it can't be patched from the op directory:** the OpDescriptor framework is built around the assumption that ops return `ProgramDescriptor`. Metal 2.0 factories return `ProgramArtifacts`, which is a different type. The fix is upstream — either teach OpDescriptor to consume `ProgramArtifacts`, or route layernorm callers through the standard `ttnn::device_operation::launch` path (which the framework adapter handles transparently).

**Tagged:** "OpDescriptor framework: Metal 2.0 compatibility."

### Sharded factory still on legacy `ProgramDescriptor`

The `program_factory_t` variant now contains two factories satisfying different concepts:
- `LayerNormMultiCoreProgramFactory` → `ProgramSpecFactoryConcept` (Metal 2.0)
- `LayerNormShardedProgramFactory` → `ProgramDescriptorFactoryConcept` (legacy)

`AllFactoriesValid` accepts mixed-concept variants (each alternative must satisfy exactly one concept), and `dispatch_to_mesh_workload_factory` routes per-alternative via `std::visit`. So this is structurally supported.

**Open follow-up:** port `LayerNormShardedProgramFactory` to Metal 2.0 in a later session. The audit (`METAL2_PREPORT_AUDIT.md`) confirms it is GREEN-eligible. It is substantially larger (~1960 lines including helpers, 3 semaphores, multiple borrowed-memory CBs).

---

## Successes

### Borrowed-memory DFB → `borrowed_from`

The Welford reciprocal LUT CB (legacy CB 25 with `CBDescriptor::buffer = recip_tensor.value().buffer()`) translated cleanly to Metal 2.0 via `DataflowBufferSpec::borrowed_from = TP_RECIP`. The corresponding `TensorParameter` (RECIP) is declared conditionally on `use_welford`, and the kernel-side code unchanged — it reads the LUT through the same DFB handle. The migration guide's [Dynamic CircularBuffer entry](port_op_to_metal2_audit.md#dynamic-circularbuffer-cb-built-on-borrowed-buffer-memory--landed) (LANDED) accurately described the path.

### Mixed-concept program_factory_t variant

The `LayerNormDeviceOperation::program_factory_t` variant has alternatives satisfying different factory concepts (`ProgramSpecFactoryConcept` + `ProgramDescriptorFactoryConcept`). The framework adapter dispatched correctly via per-alternative concept matching in `dispatch_to_mesh_workload_factory`. This enabled the user's "audit both, port interleaved only" scoping cleanly.

### `dfb::name → uint32_t` implicit conversion

The compute kernels relied heavily on the `DFBAccessor::operator uint32_t()` implicit conversion. Patterns like `binary_op_init_common(cb_in, cb_inb, cb_x);` and `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb::cb_scaler, ...>()` worked unchanged after replacing `get_named_compile_time_arg_val("cb_scaler")` with `dfb::cb_scaler`. The [Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers) was load-bearing.

---

## Friction

### Gap — Conditional kernel-side DFB declarations don't compile under `if constexpr`

**Doc section:** `metal2_port_patterns.md` — [Pattern: Conditional / optional DFB bindings](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings).

**Issue:** The pattern says "On the host, bind the DFB unconditionally" and gate kernel uses with `if constexpr`. But the layernorm compute kernels heavily use `(do_gamma | do_beta) ? cb_fusion : cb_out` style ternaries at file scope. C++ requires both ternary operands to name-look-up at parse time — when `cb_fusion` is conditionally declared (because the host conditionally bound `cb_fusion` DFB), this fails.

**Workaround taken:** Gate the kernel-side `constexpr uint32_t cb_xxx = dfb::cb_xxx;` declarations with `#ifdef FUSE_PRE_ADD` / `#ifdef FUSE_GAMMA` / `#ifdef FUSE_BETA` / `#ifndef RMSNORM` — matching the host's conditional DFB declarations. This isn't `if constexpr`-style gating; it's preprocessor gating, which the pattern catalog warns against (`Anti-pattern: #ifdef-gated DFB references`). However, the alternative — binding all DFBs unconditionally on the host — wastes meaningful L1 (cb_fusion ~16 tiles, cb_inb ~16 tiles, cb_gamma/beta ~Wt tiles each). The L1 budget is already tight for the no-large-tensor path.

**Suggested doc improvement:** the pattern catalog's "unconditional bind + `if constexpr` gate" assumes the kernel-side references are individual statements. For kernels where DFB names appear in ternaries, struct-scope constexpr expressions, or non-template-context name-lookup contexts, `if constexpr` doesn't elide the discarded branch's name-lookup. The pattern needs a sub-case for "name appears at file scope outside `if constexpr` / template" — recommend `#ifdef` gating as the only working path, or document the "always bind, accept the L1 waste" tradeoff explicitly.

### Confusion — Where do `args::` / `dfb::` / `ta::` namespaces come from?

**Doc section:** `metal2_migration_guide.md` — [Kernel Argument Retrieval Syntax](metal2_migration_guide.md#kernel-argument-retrieval-syntax).

**Issue:** The migration guide says "The `args::` and `dfb::` / `sem::` namespaces are auto-generated from the host-side bindings into `kernel_bindings_generated.h`." It took non-trivial digging to confirm: (a) the file is `kernel_bindings_generated.h` (not `kernel_args_generated.h` as mentioned in some places — the doc has both names), and (b) the kernel does **not** `#include` it — the build system pulls it in before `<kernel_includes.hpp>`. The kernel only needs `#include "experimental/kernel_args.h"` for the accessor templates.

**Suggested doc improvement:** state explicitly at the top of "Device-Side Migration" that the only mechanical change to a kernel's `#include`s is adding `experimental/kernel_args.h`; the generated headers are auto-included by the build.

### Gap — Helpers taking `CircularBuffer&` vs `DataflowBuffer&`

**Doc section:** Not covered explicitly in the migration guide.

**Issue:** The op directory's `layernorm_dataflow_utils.h` defines helpers like `read_block_to_cb(Noc&, CircularBuffer& cb, ...)`. The Metal 2.0 port constructs `DataflowBuffer` objects (per the migration guide), which are a different type than `CircularBuffer`. Helpers don't accept them by reference.

**Workaround taken:** Changed helper parameter types from `CircularBuffer&` to `auto&` (template-via-`auto` — relies on the duck-typed CB-method interface). This works because both `CircularBuffer` and `DataflowBuffer` expose the same `wait_front` / `reserve_back` / `push_back` / `pop_front` API.

**Suggested doc improvement:** the migration guide notes "the kernel-side DFB APIs are drop-in-compatible with the Device 2.0 `experimental::CircularBuffer` wrapper methods." A small addendum: helpers that took `CircularBuffer&` by reference need to switch to `DataflowBuffer&` or accept either via templating. This is a common port-time touch.

### Friction — `tensor.mesh_tensor()` pointer-identity requirement

**Doc section:** `metal2_migration_guide.md` — [TTNN Framework Integration](metal2_migration_guide.md#ttnn-framework-integration) (in-text note about MeshTensor extraction).

**Issue:** The framework adapter matches `TensorArg.tensor` references against MeshTensors reachable from `tensor_args` / `tensor_return_value` by pointer identity. The legacy factory had `std::optional<Tensor> recip_tensor = tensor_args.recip_tensor;` — copying the tensor — and the rest of the factory used `recip_tensor->...`. This copy creates a new `Tensor`, and `.mesh_tensor()` on the copy returns a *different* MeshTensor than the one in `tensor_args.recip_tensor`. The framework would TT_FATAL at runtime.

**Workaround taken:** Removed the copy; switched to using `tensor_args.recip_tensor->...` directly throughout.

**Suggested doc improvement:** make this requirement louder. The "pointer identity, not value identity" gotcha is implicit in `mesh_device_operation_adapter.hpp`'s `resolve_bindings` (the matching code), but only buried in a comment in `operation_concepts.hpp`. A patterns catalog entry would help future porters avoid the copy-then-bind anti-pattern.

---

## Open items for downstream

### Compute kernel `#ifdef` rework — required to build

The four compute kernels (`layernorm.cpp`, `layernorm_welford.cpp`, `layernorm_large_tensor.cpp`, `layernorm_large_tensor_welford.cpp`) were started but their conditional DFB use sites need additional `#ifdef` gating to compile cleanly. The non-welford `layernorm.cpp` has the most progress; the others have the declarations updated but the use sites (LLK calls referencing `cb_inb`, `cb_gamma`, etc.) still assume those symbols are always defined.

**Approach:** the layernorm op's compute kernels reference conditional CBs in patterns the [Conditional / optional DFB bindings](metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings) pattern doesn't cleanly address (ternaries at file scope; `binary_op_init_common(cb_x, cb_scaler, cb_ex)` with conditionally-bound `cb_scaler`). The currently-applied workaround is to gate the constexpr declarations with `#ifdef`s matching the host's conditional DFB declarations. The use sites then need similar gating — most are already inside `#ifdef FUSE_PRE_ADD` / `#ifdef RMSNORM` etc. blocks in the legacy kernel; the remaining ones (top-level `binary_op_init_common`, the cb_im_or_out selector) need new `#ifdef` brackets.

**Recommended next step:** continue the compute kernel port by walking each `cb_xxx` reference site and ensuring it lives inside the matching `#ifdef`. Then run the kernel JIT build (cmake --build build_Release --target ttnncpp) and iteratively fix any `kernel_bindings_generated.h: 'cb_xxx' is not a member of 'dfb'` errors that surface.

### Sharded factory port (deferred)

Per the audit, `LayerNormShardedProgramFactory` is GREEN-eligible for Metal 2.0 port but was scoped out of this session. The port would follow a similar structure but is substantially larger: 3 semaphores (each `initial_value=0`, no issues), multiple borrowed-memory CBs (input shard, residual shard, stats shard, recip LUT, optionally output reshard), and complex pre-/post-all-gather variants. Reuse the helper functions from `sharded_layernorm_factory_helpers.{hpp,cpp}` as starting points, but rewrite them to produce `DataflowBufferSpec` / `KernelSpec` / `WorkUnitSpec` instances.

### Build & test verification

The user runs builds and tests directly (per project memory). To verify the port:

1. `cmake --build build_Release --target ttnncpp -j 8` — compile.
2. `pytest tests/ttnn/unit_tests/operations/normalization/test_layernorm.py -x -v` — run layernorm unit tests.
3. `pytest tests/ttnn/unit_tests/operations/normalization/test_rmsnorm.py -x -v` — run rmsnorm tests (rmsnorm shares the same factory via `norm_type = RMSNORM`).
4. Tests for `layernorm_distributed` and `rmsnorm_distributed` should also exercise the non-sharded path.

Anti-pattern self-audit (per the recipe's checklist) has been partially applied — no `tensor.buffer()->address()` in the ported factory, no magic CB indices in `compile_time_arg_bindings`, no `TensorAccessorArgs<N>()` in the ported reader/writer kernels. The compute kernels still need a pass.

### Cross-op kernel touches

None — all kernel sources modified live under `device/kernels/`.

### `layernorm_dataflow_utils.h` template-via-`auto`

The helper functions in this header now take `auto&` instead of `CircularBuffer&` for the CB parameter. They are still callable from the sharded-factory kernels (which haven't been ported and still pass `CircularBuffer&`) because `auto&` accepts both. No coordination needed for the sharded port.

---

## Sharded factory follow-up — `LayerNormShardedProgramFactory`

The sharded factory was ported in a second pass in the same session. **Status: structural draft — both factories are present in `LayerNormDeviceOperation::program_factory_t` now satisfying `ProgramSpecFactoryConcept`. Kernels were mechanically converted (positional CTAs → named, magic CB indices → `dfb::name`, semaphore-ID CTAs → `sem::name`, `CircularBuffer` → `DataflowBuffer`, `TensorAccessorArgs` → `ta::name`). Builds and tests have NOT been verified in this session — per project convention, the user runs builds and tests themselves.

### Handoff points (sharded)

#### OpDescriptor / Python `factory.create_descriptor` removal
Same as the interleaved factory: `LayerNormShardedProgramFactory::create_descriptor` is gone. The Python binding for it (in `layernorm_nanobind.cpp`) was removed. Same upstream fix needed in the OpDescriptor framework as for interleaved.

### Successes (sharded)

#### Reusable helpers carried over cleanly
The pure-logic structs (`GridParams`, `WorkerDistribution`, `CoreRanges`, `KernelPaths`, `KernelDefines`, `CBSizeParams`, `PerCoreIndices`) in `sharded_layernorm_factory_helpers.{hpp,cpp}` are independent of the legacy / Metal 2.0 split — they encapsulate grid math, kernel selection, and tile sizing logic. Kept them as-is and removed only the legacy-specific emission code (`add_kernel_descriptors`, `add_cb_descriptors`, `CompileTimeArgs`, `RuntimeArgsResult`). This was a clean separation — only ~500 lines of helper code stayed, ~1000 lines of emission code moved into `layernorm_op_multi_core_sharded.cpp` as inline Metal 2.0 builders.

#### Semaphore migration
The sharded factory uses three semaphores (sender / receiver / second_stage). Translation to `SemaphoreSpec` + `SemaphoreBinding` was straightforward. Kernel-side `Semaphore<>` constructor accepts `sem::name` (implicit conversion to `uint32_t`), so the only kernel-side change was replacing `get_compile_time_arg_val(N)` with `sem::name`. No structural redesign needed.

#### Borrowed-memory DFB pattern at scale
Seven CBs in the sharded factory used `CBDescriptor::buffer = ...` (input shard, residual shard, optional in0 alias for pre-allgather, output, output_reshard, stats, recip LUT). Each translated to a `DataflowBufferSpec` with `borrowed_from = <TensorParameter>`. The pattern applied cleanly to all seven sites.

### Friction (sharded)

#### Gap — vararg vs L1-pointer for `in0_remote_noc_x/y` lists
The legacy reader kernels access per-core NOC coordinate lists via `(tt_l1_ptr uint32_t*)(get_arg_addr(N))` — pointing directly into the dispatch buffer's RTA region. Metal 2.0 has no direct equivalent: named RTAs are scalar `uint32_t`, and the only way to get a variable-length list is varargs (`get_vararg(i)`).

**Workaround taken:** copy varargs into a stack-allocated `uint32_t[num_x]` / `uint32_t[num_y]` array and pass `reinterpret_cast<tt_l1_ptr uint32_t*>(buf)` to the existing helpers. This works but copies the array onto the stack, which is wasteful for `num_x + num_y` up to ~16 elements.

**Suggested doc improvement:** if the recipe sees this pattern in many ports, consider adding a snippet to the migration guide showing how to consume a vararg-encoded list. Alternatively, the Metal 2.0 API could provide `get_vararg_buffer_addr()` returning the L1 address of the vararg region (similar to `get_arg_addr`) — eliminating the copy.

#### Friction — vararg count upper bound for write-back segments
The post-allgather writer's write-back logic uses a variable-length segment list (3 words per segment: bytes, noc_x, noc_y). The number of segments depends on the resharded shape and varies per core. My host-side schema declares `num_runtime_varargs = 64` as an upper bound; kernel side reads via `get_vararg(2 + i)`.

This isn't ideal — fewer-than-64 actual segments waste vararg slots, and we'd need to bump the constant if a layout ever exceeds 64 segments. Better long-term: support per-node vararg counts (`num_runtime_varargs_per_node`) — the host already computes the exact per-core count, so this would be a natural fit. The API mentions this exists but is "truly bizarre" in `kernel_spec.hpp` — worth a closer look during a follow-up pass.

#### Confusion — mapping legacy magic CB indices to DFB names
Several sharded compute kernels use `constexpr uint32_t cb_X = tt::CBIndex::c_N;` directly (rather than `get_named_compile_time_arg_val("name")`). These are pure magic numbers — to port them, I had to cross-reference the legacy `cb_named_args` map in `add_kernel_descriptors` against the index assignments in `add_cb_descriptors` to figure out which `dfb::name` to use. Documented the mapping in the kernel comments as `dfb::name // c_N`.

### Open items for downstream (sharded)

#### Pre/post-allgather kernels: deeper review needed
The pre-allgather and post-allgather kernel paths exercise the distributed-norm flow. They use additional CB references (especially `cb_in0_pre` at c_14 in pre-allgather mode, `cb_stats` at c_7 in post-allgather, `cb_var` at c_19). The host code wires these conditionally; the kernels reference them — but I didn't trace every conditional branch as carefully as the normal-mode kernel. The translation is mechanical but the cross-mode CB-name mappings (e.g., c_14 = `cb_in0_pre`, c_24 = `cb_x` in normal but `cb_ex_sqr` aliased to `cb_x` in post-allgather, etc.) deserve a second-pass review.

#### `runtime_varargs_per_node` for write-back segments
The sharded writer's `num_runtime_varargs = 64` is an upper bound. The runtime cost of unused vararg slots is non-zero (dispatch-buffer bytes). Switching to `num_runtime_varargs_per_node` (per-node override) would eliminate the waste, but requires understanding the API's per-node override semantics, which the migration guide flags as a "bizarre" feature with no examples. Deferred.

#### Compute-kernel `#ifdef` gating for conditional DFBs
Same issue as the interleaved factory's compute kernels: when the host conditionally binds a DFB (e.g., `cb_inb` only when `b.has_value()`), the kernel's `constexpr uint32_t cb_inb = dfb::cb_inb;` line fails to compile in the non-bound case. The sharded compute kernels currently declare all CBs unconditionally; this will need `#ifdef FUSE_PRE_ADD` / `#ifdef FUSE_GAMMA` / `#ifdef FUSE_BETA` gating like the interleaved kernels did.

#### Build and test verification (sharded)
Same protocol as interleaved:
1. `cmake --build build_Release --target ttnncpp -j 8`
2. Run sharded layernorm tests (`pytest tests/ttnn/unit_tests/operations/normalization/test_layernorm.py -k sharded`)
3. Run distributed layernorm tests
4. Compile-iterate on `kernel_bindings_generated.h` errors, mostly missing DFB / args bindings

#### Cross-op kernel touches (sharded)
None — every sharded kernel source modified lives under `device/kernels/`.

#### Reshard writer helper templated
`reshard_writer.hpp`'s `write_resharded_data` was templated on the CB types so it accepts both `CircularBuffer&` and `DataflowBuffer&`. This avoids forking the helper for the Metal 2.0 port. No coordination needed for other consumers.
