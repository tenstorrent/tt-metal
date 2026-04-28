# Architecture: layernorm

## Classification

| Field | Value |
|-------|-------|
| Type | compute |
| Subtype | normalization with width-row reduction and optional affine |
| Complexity | high |
| Mode | Hybrid |

## Goal

`y = ((x - mean(x over normalized_shape)) / sqrt(var(x over normalized_shape) + eps)) * weight + bias`

Match `torch.nn.functional.layer_norm` at the public `ttnn.operations.layernorm.layernorm` surface while reusing `ttnn.layer_norm` and its dedicated on-device layernorm execution path. The new wrapper owns `normalized_shape` semantics, affine validation/flattening, and `eps=1e-5`; the delegated device path owns reader, compute, writer, and core scheduling.

## Parameters

| Name | Type | Required | Valid Range | Default | Description |
|------|------|----------|-------------|---------|-------------|
| `input_tensor` | `ttnn.Tensor` | yes | on-device, `TILE_LAYOUT`, `bfloat16` in Phase 2a | none | Input tensor to normalize |
| `normalized_shape` | `int` or trailing-dim sequence | yes | positive values matching trailing logical dims | none | Logical region to normalize |
| `weight` | `ttnn.Tensor` | no | trailing dims must match `normalized_shape` | `None` | Optional affine scale |
| `bias` | `ttnn.Tensor` | no | trailing dims must match `normalized_shape` | `None` | Optional affine bias |
| `eps` | `float` | no | `> 0` | `1e-5` | Numerical stability constant |

## Phase Plan

1. Canonicalize `normalized_shape` to a tuple, validate trailing-dim match, validate `weight`/`bias`, and reject `eps <= 0`.
2. Flatten the logical problem to a canonical width-oriented view: `normalized_width = prod(normalized_shape)`, `outer_height = logical_volume / normalized_width`.
3. Flatten `weight` and `bias` to the same normalized-width row shape used by the device affine path.
4. Dispatch to `ttnn.layer_norm` (backed by the dedicated device `layer_norm` primitive) with `epsilon=eps`, default non-sharded config, and no residual/RMSNorm extras.
5. Reuse the existing reader/compute/writer split for scaler generation, row-wise mean/variance, rsqrt, normalization, and optional affine application.

## API Mapping

| Phase | Mechanism | Type | API / Function | File:Line | Notes |
|-------|-----------|------|----------------|-----------|-------|
| Public export | Python op registration + golden hook | raw_api | `register_python_operation()` + `attach_golden_function()` | `ttnn/ttnn/decorators.py:913-1032` | Needed so `ttnn.get_golden_function(layernorm)` works; use the existing decorator path rather than ad-hoc export logic. |
| Public semantics | `normalized_shape` canonicalization + affine reshape | pattern | `fallback_ops.layer_norm()` | `ttnn/tt_lib/fallback_ops/fallback_ops.py:211-255` | USE PATTERN: exact reference for `int`/sequence handling and trailing-dim reshape semantics. |
| Package loading | Subpackage auto-discovery | pattern | `pkgutil.walk_packages(__path__)` | `ttnn/ttnn/operations/__init__.py:11-18` | Keep `ttnn.operations.layernorm` as a normal subpackage export. |
| Device dispatch | Top-level `ttnn.layer_norm` backed by the dedicated device primitive | pattern | `ttnn::layer_norm` | `build_Debug/include/ttnn/operations/normalization/layernorm/layernorm.hpp:16-25` | Reuse the purpose-built device primitive; do not re-express LayerNorm as a chain of generic ops. Host wrapper must override the primitive's `1e-12` default with `eps=1e-5`. |
| Shape-to-row translation | Flatten trailing normalized dims into width rows | pattern | row-flattening + contiguous row split | `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp:21-27,45-58,171-201` | NO HELPER: no checked-in helper maps `normalized_shape` to the primitive's `NCHt` / `Wt` model, so the wrapper must derive `outer_height` and `normalized_width` explicitly. |
| Reader constants | Reduce-scaler tile generation | helper | `generate_reduce_scaler()` | `ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp:11-40` | USE HELPER: matches the existing reader contract and preserves partial-last-tile support. |
| Reader constants | Epsilon broadcast tile generation | helper | `generate_bcast_col_scalar()` | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:12-23` | USE HELPER: exact helper already used by the layernorm reader. |
| Reader streaming | Block-synchronous tile reads | helper | `read_block_to_cb()` | `build_Debug/libexec/tt-metalium/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/layernorm_dataflow_utils.h:168-187` | USE HELPER: preserves full-block CB reservation/push rules that the compute kernel assumes. |
| Compute reduction | Mean / variance over width rows | helper | `numeric::row_wise_mean<>()` | `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h:206-224` | USE HELPER: the existing compute kernel already relies on this helper for both `E[x]` and `Var[x]`. |
| Optional layout conversion | Row-major tilize / untilize path | helper | `tilize_all_blocks_to_cb<>()`, `untilize_all_blocks_from_cb<>()` | `build_Debug/libexec/tt-metalium/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_compute_utils.h:49-62,86-100` | USE HELPER if scope expands later; keep disabled for the initial TILE-only public path. |
| Normalize + affine | Existing subtract / rsqrt / broadcast sequence | pattern | delegated layernorm compute chain | `build_Debug/libexec/tt-metalium/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp:173-345` | NO HELPER: there is no single higher-level helper for `x-mean -> square -> rsqrt -> gamma/beta`; preserve the proven kernel sequence instead of rewriting it. |

## Kernel Assignment

| Kernel / Stage | Responsibility |
|----------------|----------------|
| Host wrapper | Validate `normalized_shape`, `weight`, `bias`, and `eps`; flatten the normalized region to `normalized_width`; flatten affine tensors to that same row width; register the golden function; call `ttnn.layer_norm` with `eps=1e-5`. |
| Reader | Generate `cb_scaler` and `cb_eps`, stream width blocks of input into `cb_in`, and stream `gamma` / `beta` once per core for reuse across all assigned rows. |
| Compute | Run `mean -> subtract mean -> square -> variance -> add eps -> rsqrt -> normalize -> optional gamma -> optional beta`, retaining `cb_xmm` for the full row and reusing affine tiles across `NCHt`. |
| Writer | Emit tiled output with the original logical shape; do not enable row-major `UNTILIZE_OUT` in the initial public path. |

## Rough CB Layout

| CB | Purpose | Producer | Consumer |
|----|---------|----------|----------|
| `cb_in` | Width-row input tiles | reader | compute |
| `cb_scaler` | reduce-scaler tile(s) | reader helper | compute reduction helper |
| `cb_eps` | epsilon broadcast tile | reader helper | rsqrt prepass |
| `cb_gamma` | affine scale row, reused per core | reader | compute affine |
| `cb_beta` | affine bias row, reused per core | reader | compute affine |
| `cb_ex` | row mean tile | compute reduction helper | compute subtraction |
| `cb_xmm` | `x - mean` scratch retained across the row | compute | compute variance + normalize |
| `cb_xmm2` | squared residual scratch | compute | variance reduction |
| `cb_ex2` | variance tile | compute reduction helper | rsqrt prepass |
| `cb_ex2pe` | reciprocal stddev tile | compute | normalization broadcast |
| `cb_fusion` | normalized / affine intermediate | compute | compute affine |
| `cb_out` | final tiled output | compute | writer |
| `cb_in_rm`, `cb_out_rm` | optional row-major staging | reader / compute | compute / writer |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile row of the canonical width-oriented view |
| Canonical view | logical reshape or equivalent descriptor translation to `[1, 1, outer_height, normalized_width]`, where `normalized_width = prod(normalized_shape)` and `outer_height = logical_volume / normalized_width` |
| Grid strategy | flatten all leading dims into contiguous height rows, tile them, then split row ranges across cores with the existing width-reduction pattern |
| Per-core work | `NCHt = tile_rows_assigned_to_core`, `Wt = padded(normalized_width) / tile_width`; each core reads `NCHt * Wt` input tiles and writes `NCHt * Wt` output tiles |
| Remainder | keep the two-core-group split pattern for uneven row counts and preserve the existing partial-last-tile scaler path for logical widths that do not fully occupy the last tile |

## Edge Cases

| Condition | Expected Behavior |
|-----------|-------------------|
| `normalized_shape` is an `int` | Canonicalize to a one-element tuple before validation |
| `normalized_shape` does not match trailing logical dims | Raise `ValueError` before device dispatch |
| any `normalized_shape` value `<= 0` | Raise `ValueError` |
| `eps <= 0` | Raise `ValueError` |
| `weight` / `bias` trailing dims mismatch | Raise `ValueError` |
| multi-dim `normalized_shape` | Accept at the wrapper level, flatten to `normalized_width`, and require a tile-compatible canonical device view for on-device execution |
| unsupported layout / dtype / storage | Reject in Phase 2a rather than silently converting or falling back |
| residual, RMSNorm, sharded, distributed, activation, recip tensor knobs | Keep out of the public `layernorm` surface in this phase even though the delegated primitive has related internal variants |

## Assumptions

- The existing device primitive can consume a canonical width-oriented view without changing the physical tile contents.
- Affine tensors can be flattened to a single reusable width row before dispatch.
- Phase 2a remains on the default non-sharded path (`LayerNormDefaultProgramConfig` over sharded/distributed variants).

## Risks And Unknowns

- The best compute and reader references are build artifacts, so engineering must verify the checked-in runtime still matches them before changing bindings.
- The delegated primitive exposes `epsilon=1e-12` by default, so forwarding without an explicit override would violate the required public contract.
- Positive on-device golden coverage is width-only today; multi-dim `normalized_shape` is validated at the wrapper level but still needs device-path confirmation once engineering wires the canonical view.

## Component Sources

| Component | Source Reference | Role | Modifications |
|-----------|------------------|------|---------------|
| Public API contract | `ttnn/ttnn/operations/layernorm/references/discovery.md`, `tt_metal/third_party/tt_ops_code_gen/eval/golden_tests/layernorm/api_contract.md` | requirements | Add `normalized_shape` validation, `eps=1e-5`, and `ttnn.operations.layernorm` import surface |
| Python semantics | `ttnn/tt_lib/fallback_ops/fallback_ops.py:211-255` | semantic wrapper | Preserve int/sequence handling and affine shape checks, but dispatch to device instead of PyTorch |
| Device entry | `build_Debug/include/ttnn/operations/normalization/layernorm/layernorm.hpp:16-25` | device API | Reuse existing dedicated layernorm primitive |
| Reader path | `ttnn/ttnn/operations/layernorm/references/input_stage_analysis.md` | input stage | Reuse scaler/eps generation and one-time affine streaming |
| Compute path | `ttnn/ttnn/operations/layernorm/references/compute_core_analysis.md` | compute core | Reuse the existing row-wise layernorm sequence unchanged |
| Core split | `ttnn/ttnn/operations/layernorm/references/reduction_pattern_analysis.md` | work distribution | Reuse flattened contiguous row assignment logic for the canonical width-oriented view |

## Open Questions For Engineering

- Confirm whether the canonical `[1, 1, outer_height, normalized_width]` view can be expressed as metadata only, or whether the delegated primitive needs an explicit reshape helper before dispatch.
- Decide whether the wrapper should remain a thin Python registration layer or whether later phases need a checked-in nanobind surface for parity with other normalization ops.
- Verify that the default non-sharded program config is sufficient for the initial hardware scope before exposing any sharded or distributed variants.
