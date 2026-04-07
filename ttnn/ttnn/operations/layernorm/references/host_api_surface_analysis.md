# Host API Surface Analysis: `batch_norm.cpp`

## Overview

Reference file: `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm.cpp`

`ttnn::batch_norm` is a thin host wrapper that does three things before handing off to the device primitive:

1. Applies host-side input guards and a zero-volume fast path.
2. In training mode, derives normalization statistics on the host by composing existing TTNN ops.
3. Calls `ttnn::prim::batch_norm` with already-prepared tensors plus forwarded memory/kernel options. `batch_norm.cpp:32-76`, `batch_norm.hpp:11-22`

For a real `layernorm` host op under `ttnn/cpp/ttnn/operations/normalization/layernorm`, this is the closest local example of the intended split between public API orchestration and device-op launch.

## Public Signature

The exported C++ surface is:

| Parameter | Meaning in reference | Notes for layernorm |
| --- | --- | --- |
| `input` | Source tensor | Required primary operand. `batch_norm.hpp:11-12` |
| `running_mean`, `running_var` | Optional state tensors | Batch-norm-specific; likely not part of minimum layernorm API. `batch_norm.hpp:13-14` |
| `training` | Selects host-side stats path | Layernorm likely does not need training/inference divergence for per-input stats. `batch_norm.hpp:15`, `batch_norm.cpp:60-73` |
| `eps` | Numerical stabilizer | Directly relevant. `batch_norm.hpp:16`, `batch_norm.cpp:74-75` |
| `momentum` | Running-stats update factor | Batch-norm-specific, likely excluded from layernorm. `batch_norm.hpp:17`, `batch_norm.cpp:65-66` |
| `weight`, `bias` | Optional affine tensors | Strong precedent for optional affine parameters in layernorm. `batch_norm.hpp:18-19`, `batch_norm.cpp:74-75` |
| `output` | Optional preallocated output | Useful surface to preserve if layernorm should support output reuse. `batch_norm.hpp:20`, `batch_norm.cpp:74-75` |
| `memory_config` | Output placement override | Forwarded through all internal ops and the final primitive. `batch_norm.hpp:21`, `batch_norm.cpp:22-25`, `batch_norm.cpp:63-66`, `batch_norm.cpp:74-75` |
| `compute_kernel_config` | Device kernel policy override | Forwarded without host interpretation. Resolution happens lower in the stack. `batch_norm.hpp:22`, `batch_norm.cpp:61-66`, `batch_norm.cpp:74-75`, `device/batch_norm_utils.cpp:14-35` |

## Host Control Flow

### 1. Shape/degenerate guards

- The wrapper rejects tensors with rank `< 4`. `batch_norm.cpp:44-48`
- Zero-volume tensors bypass all normalization logic and return a `ttnn::clone` using the requested or inherited memory config. `batch_norm.cpp:50-57`

Implication for layernorm:

- Keep an explicit degenerate fast path at the host boundary; it avoids pushing empty work into the device program factory.
- Do not copy the rank-4 restriction directly. The manifest’s target behavior is last-dimension normalization with same-shape output, so downstream phases should derive the real rank policy from layernorm callers/tests rather than from this batch-norm-specific constraint.

### 2. Training path computes stats using existing ops

When `training == true`, the wrapper computes:

- `batch_mean = mean_NHW(input, ...)`
- `mean_sq = mean_NHW(square(input), ...)`
- `batch_var = mean_sq - square(batch_mean)`
- Updates running statistics through `ttnn::prim::running_statistics(...)`. `batch_norm.cpp:59-67`

`mean_NHW` itself is a host helper that reduces over `{2, 3}` and then over dimension `0`, preserving reduced dims. `batch_norm.cpp:18-26`

This is important mainly as a pattern boundary:

- The host API is allowed to compose existing TTNN ops to prepare operands for the real device primitive.
- The public op does not expose the internal statistics tensors to the caller; it materializes them transiently and then calls the primitive. `batch_norm.cpp:59-75`

For layernorm, the equivalent design question is whether mean/variance are computed:

- inside the layernorm device program itself, or
- by a host wrapper that composes separate reductions before a smaller primitive.

Given the requested scope explicitly calls for a real host/device layernorm with kernels, this reference argues for keeping public orchestration thin and moving the normalization math into the device op unless later references force a split.

### 3. Inference path reuses caller-provided statistics

When `training == false`, the wrapper requires both `running_mean` and `running_var`, then forwards them as the normalization statistics. `batch_norm.cpp:67-75`

This branch is batch-norm-specific and should not be treated as a generic normalization requirement.

## Host-to-Device Boundary

The final handoff is a single primitive call:

```cpp
return ttnn::prim::batch_norm(
    input, batch_mean, batch_var, eps, weight, bias, output, memory_config, compute_kernel_config);
```

`batch_norm.cpp:74-75`

Key architectural takeaways:

- The public `ttnn::batch_norm` symbol owns user-facing semantics.
- The device op is exposed under `ttnn::prim::batch_norm`, not directly as part of the public namespace. `device/batch_norm_device_operation.hpp:68-78`
- Host-side policy decisions stop at tensor preparation plus option forwarding; actual output-spec creation, validation, and launch plumbing are delegated to the device operation layer.

For layernorm, the same split is desirable:

- `ttnn::layer_norm(...)` as the public host wrapper.
- `ttnn::prim::layer_norm(...)` as the device-op entrypoint consumed by the host wrapper.

## Surface Patterns Worth Reusing

| Pattern | Evidence | Relevance to layernorm |
| --- | --- | --- |
| Optional affine `weight`/`bias` accepted as `std::optional<Tensor>` | `batch_norm.hpp:18-19`, `batch_norm.cpp:74-75` | Matches manifest scope for optional affine scale/shift. |
| Optional output tensor passthrough | `batch_norm.hpp:20`, `device/batch_norm_device_operation.hpp:30`, `batch_norm.cpp:74-75` | Useful if layernorm should support caller-provided output storage. |
| Memory config override with fallback to input memory config | `batch_norm.cpp:22`, `batch_norm.cpp:55`, `batch_norm.cpp:74-75` | Strong precedent for keeping output placement configurable. |
| Compute-kernel config accepted at public boundary but resolved later | `batch_norm.hpp:22`, `device/batch_norm_utils.cpp:14-35` | Lets layernorm keep a stable public signature while deferring hardware-specific defaults. |
| Early empty-tensor return | `batch_norm.cpp:50-57` | Worth preserving to avoid edge-case device launches. |

## Mismatches To Avoid Copying Blindly

| Batch-norm behavior | Why it does not map directly |
| --- | --- |
| Requires rank-4 tensors | Layernorm target is last-logical-dimension normalization, which is not inherently rank-4. `batch_norm.cpp:44-48` |
| Uses `running_mean`, `running_var`, `training`, `momentum` | These are tied to batch-statistics state management, not per-input layer normalization. `batch_norm.hpp:13-17`, `batch_norm.cpp:60-73` |
| Computes stats with `mean_NHW` reduction over fixed dimensions | Layernorm needs last-dimension reduction, not NHW reduction. `batch_norm.cpp:18-26` |

## Derivative Implementation Guidance

For `normalization/layernorm`, the host API should likely preserve the following shape:

- Required `input`
- Required/optional `eps`
- Optional affine `weight`
- Optional affine `bias`
- Optional `output`
- Optional `memory_config`
- Optional `compute_kernel_config`

Possible additions such as `residual_input_tensor` should be treated as separate scope decisions; this reference does not justify them either way.

## Risks And Unknowns

- This reference does not answer whether layernorm statistics should be computed inside one fused device program or across multiple device ops; it only shows where the public API boundary can sit.
- The optional `output` argument exists in the primitive boundary, but this file does not show aliasing constraints or shape checks for preallocated outputs.
- The current batch-norm host wrapper is specialized for rank-4 input and channel-style statistics, so only the orchestration pattern, not the math or validation policy, should be reused.

## Assumptions

- Assumed target layernorm API is the minimum same-shape, last-dimension normalization surface described in `reference_manifest.json`, not a training-stateful normalization API.
- Assumed downstream phases want a public wrapper plus a `ttnn::prim` device-op entrypoint because that separation is explicit in this reference and consistent with the requested host/device implementation scope.
