# Port Report — `reduction/moe`

## Outcome

**`PORTED`.**

`MoeDeviceOperation::MoeProgramFactory` — the op's only factory — converted from
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to `ProgramSpecFactoryConcept`
(`create_program_artifacts`), together with all three kernel entry points it binds. No factory is
left on the legacy concept.

### Verification

Wormhole `n150`, `./build_metal.sh -e --enable-fake-kernels-target` (clean build, 0 errors).

All three confirmed test files pass — 9 tests, no regressions:

| test | result |
|---|---|
| `tests/ttnn/unit_tests/operations/reduce/test_moe.py` | 2 passed (`Wt=2` and `Wt=8`, 3 dispatches each) |
| `tests/ttnn/nightly/unit_tests/operations/reduction/test_reduction_ops.py::test_moe` | 6 passed (incl. scalar, non-4D-error, 0-volume, and preallocated-output paths) |
| `tests/ttnn/docs_examples/test_reduction_examples.py::test_moe` | 1 passed |

Every case runs the op more than once, so the program-cache-hit path — where the framework re-patches
the four `TensorBinding`s via `UpdateTensorArgs` instead of re-running the factory — is exercised, not
just the cache-miss build.

**`hw_config` before/after diff** (the silent-regression check; legacy values resolved through
`tt_metal/impl/program/program.cpp:415-430` and `tt_metal/impl/kernels/kernel_types.cpp:13-45`):

| kernel | legacy resolved | ported | match |
|---|---|---|---|
| reader | `ReaderConfigDescriptor{}` → `RISCV_1`, `NOC_0`, `DM_DEDICATED_NOC` | `create_reader_datamovement_config(arch)` → `RISCV_1`, `NOC_0`, `DM_DEDICATED_NOC` | ✓ |
| writer | `WriterConfigDescriptor{}` → `RISCV_0`, `NOC_1`, `DM_DEDICATED_NOC` | `create_writer_datamovement_config(arch)` → `RISCV_0`, `NOC_1`, `DM_DEDICATED_NOC` | ✓ |
| compute | `ComputeConfigDescriptor{}` → `HiFi4`, `fp32_dest_acc_en=false`, `dst_full_sync_en=false`, `unpack_to_dest_mode={}`, `bfp8_pack_precise=false`, `math_approx_mode=false` | `ComputeGen1Config{}` → `HiFi4`, `enable_32_bit_dest=false`, `double_buffer_dest=true` (= `!dst_full_sync_en`), `unpack_modes={}`, `bfp_pack_precision_mode=Approximate`, `sfpu_precision_mode=Precise` | ✓ |

The reader's and writer's NOCs are *not* interchangeable here and were checked individually rather
than assumed from the role names. No `unpack_modes` entry is required: `enable_32_bit_dest` is false,
so the Float32-consumer rule does not fire.

**`opt_level`:** reader and writer resolve to `O2` on both sides (Metal 2.0's `CompilerOptions`
default), so neither is set explicitly. The compute kernel resolves to `O3` on the legacy side and
would silently drop to `O2`, so it is set explicitly at `device/moe_program_factory.cpp:342`.

**Anti-pattern self-audit:** every item clean. Zero hits across the op directory for
`buffer()->address()`, `CircularBuffer` / `CBDescriptor` / `CBIndex`, `TensorAccessorArgs`,
`get_compile_time_arg_val` / `get_arg_val<` / `get_common_arg_val<`, `.id` extraction on a `dfb::`
handle, `allow_instance_multi_binding`, `get_vararg` / `num_runtime_varargs`, `ProgramDescriptor` /
`create_descriptor`, and cb-id-keyed free functions (the four surviving `get_tile_size` hits are
`DataflowBuffer` member calls, which is rule 7's required form). No `.md` path is cited from any
`.cpp` / `.hpp` / `.h` in the diff.

## Provenance

- **Recipe docs (this port):** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, exactly as the audit chose. Nothing about the op pushed back on the
fit: single program, one node, no op-owned tensors, no op-owned `GlobalSemaphore`s, no per-coord
variation. `ProgramArtifacts::op_owned_tensors` is left defaulted.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op already used the default reflection-based
  hash.
- **Pybind entry points removed:** none. `moe_nanobind.cpp` binds only the user-facing `ttnn::moe`
  function via `ttnn::bind_function`, so no pybind line referenced the vanishing `create_descriptor`
  and the file is untouched by this port.
- **One include removed** from `device/moe_device_operation.hpp`:
  `#include <tt-metalium/program_descriptors.hpp>`. The header declared no type the device-op class
  uses; it was reachable only because the factory header used to return a `ProgramDescriptor`. Left
  in place it would have been the last legacy-CB-API include in the op, against the whitelist's
  "post-port, no `CircularBuffer` references survive — sweep both sides, including unused
  `#include`s." Recorded here because it is a one-line edit to a device-op-class file rather than to
  the factory body.

### Open items

- **Relaxation candidates:** none identified. All four `TensorParameter`s are strict. The op's
  kernels bake in `Ht` / `Wt` / `K` as compile-time args and the device-op hashes the full attribute
  set, so a relaxed `TensorSpec` match would not obviously be tolerated. Nothing was flagged for this
  op family, so nothing was applied.
- **Capabilities not yet on this concept:** none needed.
- **Concept-fit friction:** none.

## Handoff points

**None.** No capitulation, no boundary-rule violation, no kernel-lib gap, no framework gap, no
removed pybind surface. Specifically:

- **No `sem::` / `tensor::` handle had to cross the op boundary.** The op allocates no semaphores, and
  both out-of-directory call sites take the buffer id as a `uint32_t` non-type template parameter —
  the shape `dfb::name`'s `constexpr` conversion already satisfies. Neither
  `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` nor `reduce_helpers_compute.hpp` needed a
  change, a fork, or a wrapper:
  - `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb::scale, PoolType::SUM, ReduceDim::REDUCE_ROW>()`
    — `device/kernels/dataflow/writer_unary_interleaved.cpp:21-22`
  - `compute_kernel_lib::reduce<pool_type, reduce_dim, in_dfb, scale_dfb, out_dfb, ReduceInputPolicy::WaitUpfrontNoPop>(…)`
    — `device/kernels/compute/moe.cpp:223-230`, reached from four `reduce_c<…>` call sites
- **No shared kernel was touched, forked, or reused.** The op owns all three kernel sources and no
  other op or test binds them, so no `_metal2` fork exists, none was created, and no pointer comment
  was added anywhere.

## Successes

- **[Kernel-side whitelist rule 7](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist) caught a line that
  compiles either way.** The writer's `const DataFormat data_format = get_dataformat(out_dfb_index);`
  and the reader's three `get_tile_size(<cb id>)` calls are cb-id-keyed free functions. Nothing in the
  build would have complained if the ported kernel had kept the free-function form by extracting an
  id from the handle — the rule is what made the object-getter rewrite (`dfb.get_tile_size()`) the
  default and the `.id` extraction unthinkable. Confirmed at the header that the two spellings read
  the same array: `tt_metal/hw/inc/api/dataflow/dataflow_api.h:280` returns
  `unpack_tile_size[operand]` and `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167` returns
  `unpack_tile_size[logical_dfb_id_]` on every non-PACK build, both behind the same
  `__has_include("chlkc_descriptors.h")` guard, so the swap is semantics-preserving on the
  data-movement path. Applies at `device/kernels/dataflow/reader_create_index_tensor.cpp:60-62` and
  `device/kernels/dataflow/writer_unary_interleaved.cpp:29`.
- **The [Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options) section's "answering *did it set
  one?* is the part that goes wrong" fired exactly as written.** `grep -n opt_level
  device/moe_program_factory.cpp` on the legacy factory returns nothing, and the instinct from that
  is "nothing to carry over." The legacy `ComputeConfigDescriptor` resolves to `O3` while Metal 2.0's
  type-agnostic `CompilerOptions` defaults to `O2`, so the compute `KernelSpec` needed an explicit
  `.compiler_options = {.opt_level = KernelBuildOptLevel::O3}`
  (`device/moe_program_factory.cpp:342`). Nothing in the build or the tests would have flagged the
  missing level.
- **The [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  pattern pre-empted a duplicate-symbol failure.** `ttnn_op_reduction` is unity-built
  (`ttnn/cpp/ttnn/operations/reduction/CMakeLists.txt:7`) and the already-ported `accumulation` and
  `ema` factories in the same target declare `ACCUM_READER` / `EMA_READER` and friends in their
  anonymous namespaces. Reading the pattern first is why this factory's constants went in as
  `MOE_READER` / `MOE_DFB_*` / `MOE_TENSOR_*` rather than the natural bare `READER` / `INPUT`, which
  would have collided on the first unity build.
- **The two-toucher / self-loop [endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  made the 13-buffer census mechanical.** Re-deriving rather than transcribing produced the same
  answer as the brief on every row (six 1:1, seven self-loops), including the two rows the brief
  warned about: `scale` is produced by the *writer* (`writer_unary_interleaved.cpp:21-22`), and
  `index`'s raw `get_write_ptr()` fill (`reader_create_index_tensor.cpp:23`) is one toucher, not two,
  because the same kernel brackets it with `reserve_back` / `push_back`. Reading that fill as a second
  endpoint is the mistake that would have produced a needless multi-binding flag.

## Friction

### Gaps

- **The recipe never mentions the `TT_KERNEL` entry-point convention, and a porter finds it while
  reading the one header the recipe tells them to add.** `experimental/kernel_args.h` is one of the
  two includes the [kernel-side whitelist](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist) sanctions,
  and lines 44-47 of that header define `TT_KERNEL` as "marks the named-arg entry point; the JIT
  generates `kernel_main()` from its signature" — an *alternative* to the `void kernel_main()` +
  `get_arg(args::name)` form the recipe prescribes, with real JIT support behind it
  (`tt_metal/jit_build/kernel_signature_parser.hpp`, `tt_metal/jit_build/genfiles.cpp:331-357`). The
  parser header documents it as optional and fully backward compatible, so this port used the
  recipe's form. But the recipe gives a porter nothing to decide with: they read the sanctioned
  header, find a second convention that looks newer, and have to infer from silence that it is not
  the target. **Suggestion:** one sentence in the whitelist's rule 4 saying the port targets
  `void kernel_main()` + `get_arg(args::name)` and that `TT_KERNEL` is out of scope for a port (with
  whatever the real reason is — not yet the convention, or a separate migration).
- **The recipe has no guidance for a legacy CB whose `total_size` is not a multiple of its own
  `page_size`.** `DataflowBufferSpec` expresses backing memory as `entry_size × num_entries`, so a
  legacy `CBDescriptor` with mismatched `total_size` / `page_size` has no 1:1 translation and the
  porter must choose which quantity to preserve. `c_5` input_transposed is such a CB
  (`total_size = Wt * tile_size(Float16_b)`, `page_size = input_tile_size`). The
  [Construct](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#construct-paired-spec--run-args) step says only "build with
  `entry_size`, `num_entries`", which reads as though the division always comes out even. Resolved by
  preserving total bytes (`num_entries = Wt * value_tile_size / input_tile_size`,
  `device/moe_program_factory.cpp:158`) on the scope-discipline argument that the alternative
  (`num_entries = Wt`) silently changes the FLOAT32 footprint and is therefore a fix, not a port.
  **Suggestion:** state that rule explicitly — preserve `entry_size × num_entries == legacy
  total_size`, never the page count the kernel appears to want — and say that a non-even division is
  a report item rather than a stop.
- **Dropping the legacy includes silently drops a transitive dependency, and the natural replacement
  is the one a pre-commit hook rejects.** The legacy factory reached
  `tt::tt_metal::datatype_to_dataformat_converter` and `tile_size` transitively through
  `<tt-metalium/host_api.hpp>` and `<tt-metalium/program_descriptors.hpp>`. Both go away with the
  port, so the ported factory needs an explicit include — and grepping for the declaration lands on
  `tt_metal/api/tt-metalium/experimental/tensor/tensor_types.hpp`, which the repo's
  `validate-metalium-includes` pre-commit hook rejects in favour of the TTNN forward header
  `<ttnn/tensor/types.hpp>`. Caught at commit time, not build time, because both headers compile.
  The [Construct](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#construct-paired-spec--run-args)
  step lists the Metal 2.0 headers a factory adds but says nothing about the includes the *removal*
  of the legacy API strands. **Suggestion:** a line noting that dropping `host_api.hpp` /
  `program_descriptors.hpp` can strand tensor-type and tile-size helpers, and that TTNN factories
  take the `<ttnn/tensor/...>` forward headers for those rather than the
  `tt-metalium/experimental/tensor/` originals.

### Confusion

- **"The port adds exactly two headers" is true but the count is off by one in practice, and the
  extra one is invisible.** The whitelist says the port adds `experimental/kernel_args.h` and
  `api/dataflow/dataflow_buffer.h`. This op's kernels already had the latter (they were on Device
  2.0), so only one include was actually added per kernel — fine. What briefly read as a
  contradiction: `kernel_args_generated.h` *itself* opens with
  `#include "experimental/kernel_args.h"` (`tt_metal/jit_build/genfiles.cpp:280-281`), and that
  generated header is auto-injected before the kernel source — so the include the porter adds by hand
  is already there transitively and is strictly redundant. It is still the right thing to write (the
  kernel should not depend on an injected header's include list), but the recipe presents it as
  *required* rather than as *good hygiene*, so a porter who notices the redundancy has to guess which
  reading is intended. One clause would settle it.
- **The `hw_config` variant nesting invites a spelling the codebase does not use.**
  `KernelSpec::hw_config` is `variant<DataMovementHardwareConfig, ComputeHardwareConfig>` and
  `ComputeHardwareConfig` is itself `variant<ComputeGen1Config, ComputeGen2Config>`, so
  `.hw_config = ComputeGen1Config{}` compiles via a double conversion — verified, this build is green
  with exactly that spelling (`device/moe_program_factory.cpp:353`). But every other ported factory in
  the tree writes the explicit outer wrap (`ComputeHardwareConfig{…}` — e.g.
  `ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward/device/moreh_dot_backward_program_factory.cpp:155`),
  and the [Compute kernels](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compute-kernels) section's examples show neither
  form at the assignment site — they show `std::get<ComputeGen1Config>(compute_hw).<field> = …`,
  which presupposes a variant the porter has already built. Both spellings work; the recipe should
  pick one so ported factories read alike.

## Open items for downstream

- **Shared kernel touches:** none. All three kernel sources are owned by this op and bound by no
  other factory, so there is no fork, no in-place bundled conversion, and no unmigrated-consumer list
  to carry forward.
- **`c_5` input_transposed is under-allocated for any input dtype wider than BFLOAT16.** The port
  preserves the legacy byte count exactly, so the latent mismatch survives unchanged:
  `device/moe_program_factory.cpp:154-160` sizes the buffer as `Wt` Float16_b tiles while
  `device/kernels/compute/moe.cpp:278` reserves `Wt` *input-format* entries. For FLOAT32 the entry
  count works out to `Wt/2`. Reachable only through an undocumented dtype — the nanobind docs restrict
  inputs to BFLOAT16 but `validate_on_program_cache_miss` never checks dtype, and the factory does
  contemplate FLOAT32 (`device/moe_program_factory.cpp:75-76`, `scalar_df`). The owner's call is
  whether to enforce the documented BFLOAT16-only contract in validation or to size the buffer from
  `input_tile_size`. Same finding as audit misc anomaly 1, restated here because the port had to make
  a translation choice around it.
- **Pre-existing comment drift in the compute kernel, deliberately left alone.** Eighteen
  Precondition / Postcondition comments in `device/kernels/compute/moe.cpp` name identifiers that no
  longer exist anywhere in the file — `in0_cb` / `in1_cb` / `in_cb` / `scale_cb` / `out_cb` (the
  parameters have been `*_dfb` since the Device 2.0 migration) at `:27-30`, `:66-68`, `:102-104`,
  `:129-132`, `:161-162`, `:189-190`, `:216-221`, and `cb_intermed0` / `cb_intermed1` at `:324`,
  `:329`, `:368`, `:373` (the buffers are `input_transposed` / `index_transposed`). This port touches
  none of those lines, so renaming them would be diff noise unrelated to the Metal 2.0
  transformation. Worth a one-line sweep by the op owner — a reader grepping the ported kernel for a
  buffer by the name a comment gives finds nothing.
- **Two unused includes in the compute kernel.** `#include "api/debug/dprint.h"` (`:18`) with no
  `DPRINT` in the file, and `#include "ckernel_sfpu.h"` (`:19`) whose only apparent purpose is the
  commented-out `// sfpu::_init_sfpu_config_reg();` at `:436`. Neither is a CB or argument construct,
  so neither is port work. Same as audit misc anomaly 5.
- **Two dead `constexpr uint32_t onetile = 1;` locals** at
  `device/kernels/dataflow/reader_create_index_tensor.cpp:47` and
  `device/kernels/dataflow/writer_unary_interleaved.cpp:18`, left exactly as found for the same
  reason. Same as audit misc anomaly 4. (By contrast the writer's dead
  `get_dataformat(out_dfb_index)` local *was* dropped — whitelist rule 7 forced that line to change
  because the cb id no longer exists, and rewriting a provably-dead statement into a member-getter
  call would have been worse than removing it.)
- **Test coverage note.** The confirmed test set has no C++ gtest — `ttnn.moe` is exercised only from
  Python. The op's cache behavior *is* covered: both
  `tests/ttnn/unit_tests/operations/reduce/test_moe.py` (three iterations per shape) and the nightly
  `test_reduction_ops.py::test_moe` (via the `_run_twice` helper in
  `tests/ttnn/nightly/unit_tests/operations/reduction/utility_functions.py:125-126`) invoke the op
  more than once, which is what exercises the `UpdateTensorArgs` cache-hit path the port's
  `TensorBinding`s now drive. No coverage gap was acted on.
- **Per-op carry-over.** Sibling `reduction/topk` shares this op's overall shape — one factory, an
  index-generating reader, a transposing/bitonic compute kernel, the same `reduce_helpers_*`
  kernel-lib donors — so the endpoint census and the self-loop-heavy binding layout here should
  transfer nearly unchanged when it is ported.
