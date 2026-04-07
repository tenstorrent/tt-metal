# Device Validation And Launch Analysis: `batch_norm_device_operation.cpp`

## Overview

Reference file: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp`

`BatchNormOperation` is the device-op boundary between the public host wrapper and the program factory. It packages launch-time attributes, validates tensor contracts, defines output allocation/spec rules, computes the program-cache key, and invokes `ttnn::device_operation::launch`. `batch_norm_device_operation.hpp:12-65`, `batch_norm_device_operation.cpp:33-182`

For a new `layernorm` op, this file is the clearest local template for:

- what lives in `operation_attributes_t` versus `tensor_args_t`,
- which checks happen on every invocation versus only on program-cache misses,
- how output tensors are allocated or reused,
- and how the op enters the generic TTNN device-operation runtime.

## Operation Shape

### Attributes

`operation_attributes_t` holds:

| Field | Purpose | Evidence |
| --- | --- | --- |
| `eps` | Numerical stabilization constant | `batch_norm_device_operation.hpp:13-16`, `batch_norm_device_operation.cpp:174-178` |
| `memory_config` | Output memory layout/placement policy | `batch_norm_device_operation.hpp:15`, `batch_norm_device_operation.cpp:68-69`, `batch_norm_device_operation.cpp:106-108`, `batch_norm_device_operation.cpp:176` |
| `compute_kernel_config` | Fully resolved compute-kernel policy | `batch_norm_device_operation.hpp:16`, `batch_norm_device_operation.cpp:177` |
| `input_dtype` | Source dtype for output-spec defaulting and hashing | `batch_norm_device_operation.hpp:18`, `batch_norm_device_operation.cpp:98-99`, `batch_norm_device_operation.cpp:128` |
| `dtype` | Optional output dtype override | `batch_norm_device_operation.hpp:19`, `batch_norm_device_operation.cpp:98-99` |

`get_dtype()` returns `dtype` when present, otherwise `input_dtype`. `batch_norm_device_operation.cpp:98-100`

### Tensor args

`tensor_args_t` separates required and optional runtime tensors:

| Field | Role | Evidence |
| --- | --- | --- |
| `input` | Primary data tensor | `batch_norm_device_operation.hpp:24-26` |
| `batch_mean` | Required statistics tensor | `batch_norm_device_operation.hpp:26` |
| `batch_var` | Required statistics tensor | `batch_norm_device_operation.hpp:27` |
| `weight` | Optional affine scale | `batch_norm_device_operation.hpp:28` |
| `bias` | Optional affine shift | `batch_norm_device_operation.hpp:29` |
| `output` | Optional caller-provided output | `batch_norm_device_operation.hpp:30` |

For layernorm, this split is likely reusable even if the specific tensor list changes. If mean/variance are produced inside the same device program, they would disappear from `tensor_args_t`; if they are precomputed externally, this reference gives the expected layout.

## Validation Structure

### Shared per-tensor checks

The local helper `check_tensor_BN` enforces:

- tiled layout,
- dtype limited to `BFLOAT16` or `FLOAT32`,
- device storage,
- allocated device buffer,
- rank exactly `4`,
- channel dimension equality with the input tensor (`logical_shape()[1]`). `batch_norm_device_operation.cpp:17-30`

`validate_tensors` applies that helper to:

- `input`,
- required `batch_mean` and `batch_var`,
- optional `output`,
- optional `weight`,
- optional `bias`. `batch_norm_device_operation.cpp:33-58`

Important derivative lesson:

- The validation helper is written around the mathematical contract of batch norm, not around generic device-op mechanics.
- For layernorm, the helper should be rebuilt around last-dimension normalization semantics rather than copied. The rank-4 restriction and channel-index check are reference-specific.

### Extra cache-miss validation

`validate_on_program_cache_miss` adds stricter checks before program construction:

- `input`, `batch_mean`, `batch_var`, and optional affine tensors must be tiled and `INTERLEAVED`.
- The requested output `memory_config` must also be `INTERLEAVED`.
- After those layout checks, it calls `validate_tensors(...)`. `batch_norm_device_operation.cpp:60-96`

This implies a deliberate layering:

| Validation layer | Purpose |
| --- | --- |
| `validate_tensors` | Cheap semantic/input-shape checks that are always valid to run |
| `validate_on_program_cache_miss` | Additional constraints tied to the compiled program shape and memory layout |

The generic launcher uses `validate_on_program_cache_miss` on cache misses, and falls back to the same method on cache hits unless a specialized `validate_on_program_cache_hit` exists. `ttnn/api/ttnn/device_operation.hpp:232-236`, `ttnn/api/ttnn/device_operation.hpp:287-288`, `ttnn/api/ttnn/device_operation.hpp:387-392`

For layernorm, this is a strong design pattern:

- Put broad semantic invariants in `validate_tensors`.
- Put program-factory-specific assumptions such as interleaving, sharding restrictions, or tile-shape constraints in the cache-miss validation path.

## Output Spec And Allocation Policy

### Output spec

`compute_output_specs` preserves the input logical shape and creates a tiled output layout using:

- `operation_attributes.get_dtype()`,
- `PageConfig(Layout::TILE)`,
- `operation_attributes.memory_config`. `batch_norm_device_operation.cpp:102-109`

This matches the manifest’s target layernorm behavior of same-shape output.

### Output allocation / reuse

`create_output_tensors` reuses a provided `output` tensor when present; otherwise it allocates a device tensor from the computed spec on `input.device()`. `batch_norm_device_operation.cpp:111-119`

That is the main reusable output policy for layernorm:

- preserve exact input logical shape,
- keep tiled output,
- honor explicit output placement,
- allow optional output reuse if the op surface keeps that argument.

## Program Cache Identity

`compute_program_hash` builds the cache key from:

1. `attributes`
2. `input.dtype()`
3. `input.memory_config()`
4. dtype/memory-config metadata for `batch_mean`
5. dtype/memory-config metadata for `batch_var`
6. dtype/memory-config metadata for optional `weight`
7. dtype/memory-config metadata for optional `bias` `batch_norm_device_operation.cpp:121-154`

`operation_attributes_t::to_hash()` contributes:

- `eps`
- `memory_config`
- resolved output dtype (`get_dtype()`)
- resolved compute kernel config `batch_norm_device_operation.cpp:156-158`

Observations for derivative work:

- Shape does not appear explicitly in this custom hash; it is presumably captured elsewhere in the mesh-workload hash path or through program-factory internals. I would not rely on this file alone to infer the full cache identity surface.
- Optional affine presence affects the hash through optional dtype/memory metadata, so separate compiled programs may be cached for affine vs non-affine variants.
- Output tensor presence itself is not part of the hash; only its spec-driving attributes matter.

## Launch Path

The primitive entrypoint `ttnn::prim::batch_norm(...)` does the following:

1. Builds `operation_attributes_t` with:
   - `eps`
   - `memory_config.value_or(input.memory_config())`
   - resolved compute-kernel config via `batch_norm::utils::resolve_compute_kernel_config(...)`
   - `input.dtype()` `batch_norm_device_operation.cpp:163-178`
2. Packages the runtime tensors into `tensor_args_t`. `batch_norm_device_operation.cpp:179`
3. Calls `ttnn::device_operation::launch<BatchNormOperation>(...)`. `batch_norm_device_operation.cpp:181`

The generic launcher then:

- checks that the first tensor is a device tensor,
- creates output tensors before dispatch,
- resolves the mesh device from the first tensor,
- computes/queries the program cache,
- validates and either reuses cached programs or builds a new workload. `ttnn/api/ttnn/device_operation.hpp:430-507`, `ttnn/api/ttnn/device_operation.hpp:366-393`

For layernorm, the main reusable boundary is:

- `ttnn::prim::layer_norm(...)` should fully resolve `operation_attributes_t` before launch.
- The program factory should not need to reinterpret public options like `std::optional<MemoryConfig>`; it should receive concrete values.

## Contracts Relevant To Layernorm

| Contract in reference | Reuse status for layernorm |
| --- | --- |
| Device-op split into `operation_attributes_t` and `tensor_args_t` | Reuse directly. |
| Same-shape output spec with tiled layout | Reuse directly. |
| Optional output tensor reuse | Reuse if the public API keeps `output`. |
| Resolved compute-kernel config stored in attributes | Reuse directly. |
| Cache-miss validation layer distinct from generic validation | Reuse directly. |
| Required `batch_mean` / `batch_var` tensors | Only reuse if layernorm is architected as a two-stage stats-precompute + normalize flow. |
| Rank-4 / channel-dimension checks | Replace with last-dimension normalization rules. |
| Interleaved-only memory layout | Tentatively reuse for the first implementation unless later references establish a shard-aware design. |

## Risks And Unknowns

- This file does not show whether the generic launcher’s mesh-workload hash adds shape-dependent information beyond `compute_program_hash`; downstream design should verify that before assuming shape is irrelevant to caching.
- `operation_attributes_t` includes optional `dtype`, but the primitive builder in this reference only populates `input_dtype`; this suggests output-dtype override support is latent or handled elsewhere.
- The validation path is entirely interleaved-only. If layernorm later needs sharded inputs or distributed execution, this reference is not sufficient.

## Assumptions

- Assumed initial layernorm implementation will target single-device tiled/interleaved tensors, matching the manifest’s starting scope.
- Assumed same-shape output and optional affine tensors remain in scope, so output-spec and hashing patterns from this reference are directly relevant even though the statistics tensors likely are not.
