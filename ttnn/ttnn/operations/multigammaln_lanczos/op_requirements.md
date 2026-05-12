# Operation Requirements: multigammaln_lanczos

## Definition

- **Formula**:

  ```
  multigammaln(x, p=4) = sum_{k=0..3} lgamma(x − 0.5·k) + 3·log(π)
  ```

  with `lgamma(a)` implemented via the Lanczos 6-term polynomial and zero-masked at the integer poles `a = 1` and `a = 2`. Mathematically identical to `torch.special.multigammaln(x, 4)`.

- **Inputs**:

  | Name | Role | Shape pattern | Description |
  |------|------|---------------|-------------|
  | `input_tensor` | source | Rank ≥ 2, `H % 32 == 0`, `W % 32 == 0` | fp32 TILE_LAYOUT tensor on device. Safe value domain: `[2.0, 10.0]`. |

- **Output**: Same shape, dtype, layout, and memory config as `input_tensor`.

- **Parameters**: None (the `p` argument is permanently fixed at 4; see `op_design.md` § Out of Scope).

- **PyTorch Reference**:

  ```python
  def reference(x: torch.Tensor) -> torch.Tensor:
      return torch.special.multigammaln(x, 4)
  ```

- **Import Path**: `from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos`

- **Function Signature**:

  ```python
  def multigammaln_lanczos(input_tensor: ttnn.Tensor) -> ttnn.Tensor
  ```

- **Validation** (rejects with `ValueError`):

  - `input_tensor.storage_type() != ttnn.StorageType.DEVICE`
  - `input_tensor.dtype != ttnn.float32`
  - `input_tensor.layout != ttnn.TILE_LAYOUT`
  - `len(input_tensor.shape) < 2`
  - `input_tensor.shape[-1] % 32 != 0` or `input_tensor.shape[-2] % 32 != 0`

## Phases

> **Non-regression rule**: Every phase must pass all tests from prior phases.
> **Accuracy**: PCC > 0.999 to pass initially. Agents tighten tolerances and note achieved values in changelog.
> **Checkbox protocol**: Agents mark `[x]` when a phase is complete and all tests pass.

### [x] Phase 0 — Core Implementation

- **Cores**: multi-core via `ttnn.split_work_to_cores(compute_with_storage_grid, total_tiles)`
- **Dtype**: float32 only
- **Layout**: TILE_LAYOUT only
- **Memory**: DRAM and L1 interleaved (kernels are layout-agnostic via `TensorAccessor`)
- **Compute config**: hard-coded `HiFi4` + `fp32_dest_acc_en=True` + `UnpackToDestFp32` on `cb_input_tiles` and `cb_accumulator`
- **Params**: none (p = 4 is permanent)
- **Test shapes**: `(1,1,32,32)`, `(1,1,32,256)`, `(1,1,256,32)`, `(1,1,64,128)`, `(1,1,128,64)`, `(2,4,64,128)`

### [x] Refinement 1 — Reuse DST across lgamma iterations

Try not to have a `cb_accumulator` round-trip between lgamma iterations.

Acceptance: all Phase 0 tests must continue passing; the precision-baseline
numbers (PCC, max_abs_err) must not regress.

### [ ] Refinement 2 — Expose `compute_kernel_config`

Goal: let callers override math fidelity, fp32 dest acc, and (optionally) the unpack-to-dest mode, instead of using the hard-coded HiFi4 + fp32 dest acc + UnpackToDestFp32 baseline.

- Change the entry-point signature to `multigammaln_lanczos(input_tensor, *, compute_kernel_config: Optional[ttnn.DeviceComputeKernelConfig] = None)`.
- Translate `compute_kernel_config` (or its absence) into the `ttnn.ComputeConfigDescriptor` fields in `multigammaln_lanczos_program_descriptor.py`.
- Defaults must preserve Phase-0 precision exactly: when `compute_kernel_config is None`, the program descriptor must produce the same config (HiFi4 / fp32_dest_acc / UnpackToDestFp32 on `cb_input_tiles` + `cb_accumulator`) that the verifier locked in.
- Lower-fidelity overrides (HiFi2, no fp32 dest acc) are caller-acknowledged precision tradeoffs and do NOT need to pass the Phase 0 precision baseline. Add tests that exercise at least one non-default config to confirm the plumbing works.
- **Note from verifier**: The numerical_stability analysis identifies `unpack_to_dest_mode[cb_accumulator]` as the single highest-leverage precision lever (~24× improvement when enabled, measured in this verification phase). Whatever API is chosen for `compute_kernel_config`, do not let callers accidentally turn this off — either keep it always-on, or surface a separate flag with a "don't disable unless you understand the cost" docstring. Reference: `numerical_stability.md` line 64 and the verification report's Precision Baseline.

### [ ] Refinement 3 — bfloat16 input support

Goal: accept `dtype=ttnn.bfloat16` end-to-end while preserving fp32 precision in the compute kernel.

- Relax the validator to allow bfloat16 in addition to float32.
- Reader/writer remain layout-aware via `TensorAccessor`; only the per-page byte count changes.
- The compute kernel reads bfloat16 from `cb_input_tiles` but runs the Lanczos polynomial in fp32 DST (fp32_dest_acc still on). Output dtype should match input dtype (`cb_output_tiles` page size and CB format follow `input_tensor.dtype`).
- Precision: expect a small additional error vs fp32 input due to the bf16 quantisation of `x` itself. Pass criterion: PCC ≥ 0.99 on the safe domain `[2.0, 10.0]`.
- **Note from verifier**: page sizes are taken from `input_tensor.buffer_page_size()` / `output_tensor.buffer_page_size()` (`multigammaln_lanczos_program_descriptor.py:39-40`), so most of the bfloat16 plumbing is already correct — only the entry-point validator and the `cb_accumulator` page size (still fp32, intentionally) need explicit handling. `cb_accumulator` MUST stay Float32 + UnpackToDestFp32 to keep the running-sum precision intact.

### [ ] Refinement 4 — Non-tile-aligned shapes

Goal: support `H` and/or `W` that are not multiples of 32 (matches PyTorch's elementwise semantics).

- Pad the input internally to tile boundaries (host-side or via a kernel-side tile-pad path).
- Mask the output so that the trailing-edge tiles do not include garbage data outside the logical shape.
- The Lanczos polynomial is undefined on `0` and produces NaN there, so the padding fill value matters — choose a safe-domain constant (e.g., `2.0`) for the padded elements and let the output mask discard them.
- **Note from verifier**: the validator at `multigammaln_lanczos.py:74` is the single rejection site. Both kernels assume `tile_id` indexes a clean tile-aligned interleaved layout (`multigammaln_lanczos_reader.cpp:34`, `multigammaln_lanczos_writer.cpp:32`). The host-pad approach is simpler than threading a tail-mask through the compute kernel.

### [ ] Refinement 5 — ROW_MAJOR_LAYOUT input support

Goal: accept `layout=ttnn.ROW_MAJOR_LAYOUT` and tilize internally rather than rejecting at validation.

- Use the `tilize_helpers_dataflow.hpp` helper family on the reader side to convert sticks to tiles inside the reader kernel.
- Compute and writer paths stay tile-based; only the reader changes.
- Alternative (cheaper to implement): tilize on the host before launch, untilize after.

### [ ] Refinement 6 — Memory pressure / sharded layouts

Goal: handle very large tensors that exceed the per-core DRAM read budget by using sharded inputs/outputs, and (optionally) reduce per-core L1 footprint.

- Current per-core L1 footprint is 24 KB (3 CBs × 2 pages × 4096 B) — minimal. The accumulator CB cannot be reduced below 2 pages without deadlocking the front+back ping-pong (see `op_design.md` K6 #3).
- Add a sharded input path: `cb_input_tiles` becomes a sharded CB pointing at a per-core slice of the input shard. `TensorAccessor` already supports both interleaved and sharded layouts compile-time.
- Output shard layout must match input shard layout to keep tile-id correspondence.
- **Note from verifier**: this is the lowest-priority refinement — the op is currently compute-bound, not memory-bound (see `data_transfer.md` § 7). Tackling this only makes sense if a real model surface needs sharded I/O.
