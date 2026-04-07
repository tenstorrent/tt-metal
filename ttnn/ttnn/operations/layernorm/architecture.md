# Layernorm Architecture

## Goal

Implement a real C++ host/device `layer_norm` under `ttnn/cpp/ttnn/operations/normalization/layernorm` with:

- a public `ttnn::layer_norm(...)` wrapper in the normalization module,
- a `ttnn::prim::layer_norm(...)` device-op entrypoint,
- dedicated layernorm program-factory and kernel sources,
- normalization-module CMake integration, and
- normalization nanobind registration.

Phase 2a baseline is a single-device, tiled/interleaved, last-logical-dimension layernorm with same-shape output and optional affine `weight` / `bias`. The baseline math is:

`y = ((x - mean_last_dim(x)) * rsqrt(var_last_dim(x) + epsilon)) * weight + bias`

Current checked-in callers also expect `epsilon` spelling, optional `residual_input_tensor`, and an optional `program_config` surface (`tests/tt_eager/python_api_testing/sweep_tests/tt_lib_ops.py:643-645`, `tests/ttnn/python_api_testing/sweep_tests/ttnn_ops.py:1161`, `models/demos/deepseek_v3/reference/ttnn.md:147`). Phase 2a architecture keeps those as compatibility constraints, but distributed pre/post-all-gather stages remain out of scope for the first engineering cut.

## Classification

| Item | Decision |
| --- | --- |
| Op type | Fused normalization/reduction device op, not a composite host-only wrapper |
| Normalization axis | Last logical dimension only |
| Output contract | Same logical shape as input, tiled output layout |
| Baseline execution | Single device, interleaved tensors |
| Baseline kernel family | One interleaved last-dim program factory with reader / compute / writer kernels |
| Future extension axes | residual pre-add, sharded program configs, RMS/distributed variants |

## Planned Source Layout

| Planned file | Responsibility |
| --- | --- |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm.hpp` | Public C++ declaration for `ttnn::layer_norm(...)` |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm.cpp` | Host wrapper, zero-volume fast path, option forwarding |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.hpp` | Per-op binder declaration |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/layernorm_nanobind.cpp` | Python binding and docstring for `ttnn.layer_norm` |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_common.hpp` | Common enums/config helpers shared by layernorm-class users |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_utils.hpp/.cpp` | compute-kernel-config and program-config resolution helpers |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp/.cpp` | `LayerNormOperation`, validation, output spec/allocation, primitive entrypoint |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_program_factory.cpp` | Interleaved last-dim factory and runtime-arg override logic |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_layernorm.cpp` | Dense input/residual/affine reads plus width-tail helpers |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_layernorm.cpp` | Output writeback |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_kernel.cpp` | Main BF16/standard compute path |
| `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sfpu_kernel.cpp` | Optional fp32-dest / SFPU-specialized compute path |

`layernorm_common.hpp` is worth creating even in the first cut because current workspace code already includes it and expects a common primitive/config surface (`ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.cpp:6-7,38-52`).

## Public Surface

### Python-facing shape

The nanobind surface should be aligned to current callers, not copied blindly from `batch_norm`:

`ttnn.layer_norm(input, *, epsilon=1e-5, weight=None, bias=None, residual_input_tensor=None, memory_config=None, program_config=None, compute_kernel_config=None)`

Key decisions:

- Keep `input` positional and the rest keyword-only, following the `bind_function` pattern used by `batch_norm` (`ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm_nanobind.cpp:75-92`).
- Use the keyword name `epsilon`, not `eps`, because checked-in layernorm callers and docs consistently use `epsilon` (`tests/tt_eager/python_api_testing/sweep_tests/tt_lib_ops.py:634-645`, `tests/ttnn/nightly/unit_tests/operations/fused/test_single_core_fused_ops.py:47`, `models/demos/deepseek_v3/reference/ttnn.md:147`).
- Keep `residual_input_tensor` and `program_config` in the public shape as compatibility hooks, even if the first engineering cut validates only a subset of combinations.

### C++ boundary split

The public/device split should mirror the existing normalization pattern:

- `ttnn::layer_norm(...)` performs host-side guard logic and resolves optional defaults, analogous to `ttnn::batch_norm(...)` (`ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm.cpp:32-76`).
- `ttnn::prim::layer_norm(...)` constructs fully resolved operation attributes and launches `LayerNormOperation`, analogous to `ttnn::prim::batch_norm(...)` (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:162-181`).

Unlike `batch_norm`, the host wrapper should not precompute statistics with separate TTNN reductions. The layernorm statistics belong inside the dedicated device op so the implementation remains a real host/device op rather than a composite fallback.

## Phase Plan

### 1. Host wrapper

`ttnn::layer_norm(...)` should:

- reject unsupported layouts/types early,
- preserve the zero-volume fast path via `ttnn::clone(...)`, as `batch_norm` already does (`ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm.cpp:50-57`),
- resolve `memory_config.value_or(input.memory_config())`,
- resolve compute-kernel defaults through a layernorm-specific utility analogous to `batch_norm::utils::resolve_compute_kernel_config(...)` (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_utils.cpp:14-35`),
- forward the normalized option set to `ttnn::prim::layer_norm(...)`.

### 2. Primitive / device-op entrypoint

`ttnn::prim::layer_norm(...)` should build a concrete `operation_attributes_t` and `tensor_args_t`, then call `ttnn::device_operation::launch<LayerNormOperation>(...)`, following the `BatchNormOperation` boundary (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.hpp:12-65`, `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:163-181`).

Recommended `operation_attributes_t` fields for Phase 2a:

- `epsilon`
- resolved `memory_config`
- resolved `program_config`
- resolved `compute_kernel_config`
- `input_dtype`
- optional output `dtype`
- `LayerNormType` and `DistributedLayerNormStage` enums, but validate Phase 2a to `{LAYERNORM, NOT_DISTRIBUTED}` only

Recommended `tensor_args_t` fields for Phase 2a:

- `input`
- optional `weight`
- optional `bias`
- optional `residual_input_tensor`
- optional `output`

### 3. Device validation and output policy

Reuse the `BatchNormOperation` structure, but replace its channel/rank-4 rules with last-dim rules:

- all participating tensors must be device tensors with allocated buffers (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:17-30`),
- baseline implementation accepts tiled/interleaved tensors only (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:60-96`),
- `residual_input_tensor`, when present, must match input logical/padded shape,
- `weight` / `bias`, when present, must be broadcast-compatible with the last logical dimension,
- output spec must preserve input logical shape while using tiled page config and requested memory config (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:102-119`).

The program hash should include at least:

- `epsilon`,
- `memory_config`,
- `program_config`,
- resolved `compute_kernel_config`,
- input dtype/memory config,
- optional residual/affine presence and their dtype/memory metadata.

### 4. Program-factory selection

Phase 2a should start with one `InterleavedLastDimFactory` in `program_factory_t`, matching the single-factory `BatchNormOperation` pattern (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.hpp:36-58`).

However, keep the selection point explicit so the op can grow new factories later, similar to `MorehNormOperation::select_program_factory(...)` (`ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_device_operation.cpp:112-123`).

Recommended future split:

- `InterleavedLastDimFactory` for Phase 2a baseline
- `ShardedLastDimFactory` later if `program_config` requires sharded traversal
- distributed stage factories later if pre/post-all-gather layernorm is restored

### 5. Kernel pipeline

The device pipeline should be treated as a multi-phase last-dim normalization pipeline:

`read row tiles -> optional residual pre-add -> accumulate row statistics -> combine Welford partials / rsqrt(var + epsilon) -> normalize row tiles -> optional affine -> write output`

This matches the local kernel-library note that layernorm-class kernels are multi-step normalization pipelines with `E[x]`, `E[x^2]`, subtract-mean, and affine phases (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_plan.md:424-431`).

## Work Distribution

Use row-unit scheduling, not flat output-tile scheduling.

The work-unit definition should follow the `moreh_norm` last-dimension path:

- one work unit = one tile row across all `Wt` tiles of the last logical dimension,
- `num_units = outer_volume * Ht`,
- assign row units with `split_work_to_cores(grid, num_units)`,
- advance runtime `tile_offset` by `num_rows_per_core * Wt`.

This is directly grounded in `moreh_norm` last-dim scheduling (`ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp:32-40,52-59,151-189`).

Required invariants:

- never split a logical row across multiple cores,
- preserve row boundaries across both statistics and normalization phases,
- tail padding on the final width tile must be masked out of mean/variance computation,
- output tiles are written in the same flattened order as the input row traversal.

`program_config` should eventually control the same scheduling knobs, but the Phase 2a auto/default config can simply target the full `compute_with_storage_grid_size()` and row-split work distribution.

## Kernel Role Split

| Kernel | Responsibilities | Why |
| --- | --- | --- |
| Reader | Stream dense row-aligned operands: input row tiles, optional residual row tiles, optional affine tiles, width-tail mask tile, epsilon/constant tiles | For layernorm, `weight` and `bias` follow the width traversal and are dense streams, unlike batch-norm’s sparse per-channel broadcast operands |
| Compute | Optional residual pre-add, per-width partial stat generation, Welford combine, `rsqrt(var + epsilon)`, per-element normalization, optional affine application | This is the normalization-specific math core |
| Writer | Drain output tiles to destination tensor | Output DMA stays isolated and simple |

This intentionally differs from `batch_norm`, where the writer kernel also feeds sparse broadcast statistics and affine operands (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp:79-125`). For layernorm, the natural data movement is row-aligned, so the reader should own the dense operand streams.

## Rough CB Map

| CB purpose | Producer | Consumer | Notes |
| --- | --- | --- | --- |
| input row tiles | reader | compute | Dense row traversal across `Wt` |
| residual row tiles | reader | compute | Optional pre-add path |
| affine weight tiles | reader | compute | Optional, broadcast across outer rows but indexed by width tile |
| affine bias tiles | reader | compute | Optional, same traversal as weight |
| width mask tile | reader | compute | Tail masking pattern from `moreh_norm` |
| epsilon / constant tiles | reader | compute | Scalar or helper tiles reused within the row |
| per-tile stat partials | compute | compute | Interleaved mean / variance-or-M2 partials for Welford combine |
| combined row stats | compute | compute | Final `mean` and `invstd` for the current row |
| row replay / staging buffer | reader or compute | compute | Engineering choice: L1 staging or controlled reread path for wide rows |
| output tiles | compute | writer | Same-shape normalized output |
| temp scratch | compute | compute | Pre-add, normalize, affine, and rsqrt intermediates |

The best directly reusable helper here is `norm::kernel_util::compute::combine_welford_partials(...)`, which already combines partial means/variances and can optionally emit `1/sqrt(var + eps)` (`ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/combine_welford.h:24-155`).

## Helper / API Mapping

| Mechanism | Planned layernorm use | Local reference |
| --- | --- | --- |
| Thin public wrapper + zero-volume fast path | Base pattern for `ttnn::layer_norm(...)` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm.cpp:32-76` |
| Device-op attribute/tensor split | Base pattern for `LayerNormOperation` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.hpp:12-65` |
| Output-spec / output-allocation rule | Same-shape tiled output | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:102-119` |
| Cache-miss validation layering | Separate semantic checks from program-shape checks | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:33-96` |
| Compute-kernel-config resolution | Create `layernorm_utils.cpp` analogue | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_utils.cpp:14-35` |
| Runtime-arg setup + cache-hit override | Reuse factory pattern for initial setup and updates | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp:23-122,296-337` |
| Last-dim row scheduling | Use row units and `tile_offset += rows * Wt` | `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp:32-40,151-189` |
| Width-tail masking | Ignore padded columns in the last width tile | `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/reader_moreh_norm_w.cpp:30-36`, `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp:38-66` |
| Per-op nanobind pattern | Create `layernorm_nanobind.cpp` via `ttnn::bind_function` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm_nanobind.cpp:18-92` |
| Normalization module registration | Add `bind_layernorm_operation(mod)` to registrar | `ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp:9-13` |
| Runtime library integration | Add layernorm sources and kernel glob to normalization target | `ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:14-35` |
| Future common primitive/config surface | Preserve room for `LayerNormProgramConfig`, norm type, and stage enums | `ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.cpp:38-52`, `ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused_nanobind.cpp:25-41,96-105` |

## Integration Plan

### Normalization CMake

`ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt` must be extended to:

- include `layernorm/device/kernels/*` in the kernel glob,
- compile the new layernorm runtime/device `.cpp` files into `ttnn_op_normalization`.

The directly visible edit surface is `ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:14-35`.

### Nanobind registration

Source-tree registration should mirror `batch_norm`:

- add `layernorm/layernorm_nanobind.hpp`
- add `layernorm/layernorm_nanobind.cpp`
- include the binder in `normalization_nanobind.cpp`
- call `detail::bind_layernorm_operation(mod)` from `py_module(...)`

The current visible registrar only binds batch norm (`ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp:9-13`).

## Open Questions For Engineering

1. Should the first implementation stage row tiles in L1 for the normalization pass, or do an explicit reread after statistics are known? The architecture permits either; the choice depends on acceptable hidden-size / L1 tradeoffs.
2. Do we fully implement `residual_input_tensor` in the first cut, or expose it but validate-reject until tests are reconciled? Existing sweeps already call it (`tests/tt_eager/python_api_testing/sweep_tests/tt_lib_ops.py:643-645`, `tests/ttnn/python_api_testing/sweep_tests/ttnn_ops.py:1161`).
3. How strict should the first `program_config` support be? Checked-in docs and experimental code already expect the symbol, but Phase 2a only requires the default interleaved last-dim path (`models/demos/deepseek_v3/reference/ttnn.md:147`, `ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.cpp:45`).
4. What exact broadcast-compatibility rule should `weight` / `bias` validation enforce for low-rank tensors versus already-padded tiled tensors?
5. `normalization/CMakeLists.txt` covers runtime sources and kernel packaging, but it does not show where nanobind translation units are compiled. That wiring must be located before engineering starts adding the new binder.
6. `BatchNormOperation::compute_program_hash(...)` does not itself show shape fields. Before finalizing layernorm caching, verify where shape-sensitive identity is supplied in the wider device-operation runtime (`ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp:121-158`).
