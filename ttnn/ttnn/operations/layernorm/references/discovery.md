# Layernorm Phase 0 Discovery

## Parsed Requirements Summary

- `op_name`: `layernorm`
- Requested scope: Phase 0 discovery only. No analysis, architecture, engineering, build, TDD, or reporting work.
- Implementation target: real C++ host/device layernorm under `ttnn/cpp/ttnn/operations/normalization/layernorm`
- Required integration points: normalization `CMakeLists.txt`, normalization nanobind module registration, per-op nanobind binding, and on-device kernels
- Available evidence in the local workspace:
  - public API/tests/docs already reference `ttnn.layer_norm(...)`
  - `ttnn/cpp/ttnn/operations/normalization/layernorm/` exists but is empty in the current working tree
  - local build artifacts and kernel-library notes mention `layernorm_*` and `rmsnorm_*` sources, but those source files are not present in the current working tree
- Inferred math target: normalize over the last logical dimension, preserve input shape, apply `epsilon`, and support optional affine `weight`/`bias`

## Candidate Operations Considered

| Candidate | Key Local Path(s) Inspected | Selected | Rationale |
|-----------|------------------------------|----------|-----------|
| `batch_norm` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/*` | Yes | Closest accessible TTNN normalization op with host wrapper, device operation, program factory, compute/dataflow kernels, nanobind binding, and normalization-module/CMake integration. |
| `moreh_norm` | `ttnn/cpp/ttnn/operations/moreh/moreh_norm/*` | Yes | Best accessible reference for last-dimension reduction scheduling and dimension-aware program-factory selection, which layernorm needs for reduction over the normalized axis. |
| `deepseek_grouped_gate` | `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/*` | No | Useful reduction-and-normalization mechanics exist, but the op is dominated by gating/top-k logic and is farther from the required normalization-module integration. |
| `layernorm` / `rmsnorm` in normalization module | referenced by local build outputs and `ttnn/cpp/ttnn/kernel_lib/*.md` | No | Mentioned locally, but the actual source files are missing from the current working tree, so they cannot serve as primary code references for this discovery run. |
| `dit_rms_norm_unary_fused` | `ttnn/cpp/ttnn/operations/experimental/transformer/dit_rms_norm_unary_fused/dit_rms_norm_unary_fused.cpp` | No | Confirms a `ttnn::prim::layer_norm(...)` call shape exists conceptually, but it includes missing `normalization/layernorm/*` headers, so it is evidence of intended API shape rather than an analyzable implementation reference. |

## Final Selected References

| Role | Operation | Path | Reason |
|------|-----------|------|--------|
| `host_api_surface` | `batch_norm` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm.cpp` | Closest host-side normalization wrapper in the target namespace; shows how non-prim API code prepares inputs, computes auxiliary statistics, and forwards to a prim/device op with optional affine parameters and memory/kernel config. |
| `device_validation_and_launch` | `batch_norm` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.cpp` | Shows device-op validation, output-spec creation, tensor creation, and `ttnn::device_operation::launch` plumbing inside the normalization module. |
| `device_program_factory` | `batch_norm` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp` | Closest accessible normalization program factory; allocates CBs for input/statistics/affine tensors, launches reader-compute-writer kernels, and produces same-shape normalized output. |
| `last_dim_reduction_pattern` | `moreh_norm` | `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp` | Best accessible pattern for reducing across the width/last-dimension path, including per-row unit splitting, runtime arguments, and compute-kernel setup for row reductions. |
| `operation_binding` | `batch_norm` | `ttnn/cpp/ttnn/operations/normalization/batch_norm/batch_norm_nanobind.cpp` | Shows per-op nanobind binding style, keyword argument layout, and docstring structure for a normalization operation exposed as a free function. |
| `module_registration` | `batch_norm` | `ttnn/cpp/ttnn/operations/normalization/normalization_nanobind.cpp` | Shows how normalization operations are registered into the module-level nanobind entrypoint. |
| `build_integration` | `batch_norm` | `ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt` | Shows normalization library source registration, kernel globbing, and install layout that layernorm must extend. |

## Planning Mode

`Hybrid`

Reason: no single accessible operation covers all needed concerns. `batch_norm` is the closest normalization/integration reference, while `moreh_norm` is needed to study last-dimension reduction mechanics for the normalized axis.

## Assumptions

- Initial implementation scope is single-device, non-distributed layernorm only. Distributed `layer_norm_pre_all_gather` / `layer_norm_post_all_gather` behavior is out of scope for this discovery run.
- Initial data-path assumption is on-device tiled input to tiled output. No row-major tilize/untilize references were selected because the request did not require row-major entry/exit support.
- Layernorm is assumed to normalize over the last logical dimension and preserve the full input shape.
- Optional affine `weight` and `bias` are in scope because local tests/docs call `ttnn.layer_norm` with them.
- Optional `residual_input_tensor` support is not assumed to be part of the minimum Phase 0 target, but local tests indicate it may need explicit scope reconciliation in later phases.
- Because the local `ttnn/cpp/ttnn/operations/normalization/layernorm/*` sources are absent, downstream phases must derive the new implementation from the selected references rather than editing an existing layernorm codepath.

## Risks

- Public local tests/docs already assume a `ttnn.layer_norm` API surface, while the current working tree lacks the corresponding normalization-layernorm source files. API compatibility will need explicit verification in later phases.
- `batch_norm` matches normalization math and integration style, but its reduction domain is channel/spatial-statistics oriented rather than last-dimension/token oriented.
- `moreh_norm` matches axis-reduction mechanics better, but it is a reduction op rather than an affine same-shape normalization op, so only parts of its program-factory structure are reusable.
- If later phases require row-major I/O, sharding defaults, distributed stats exchange, or fused residual behavior, additional references will need to be added beyond this Phase 0 set.
