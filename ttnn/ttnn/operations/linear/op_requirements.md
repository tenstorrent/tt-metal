# Operation Requirements: linear

## Definition
- **Formula**: `output = input @ weight [+ bias[0, :]]`
- **Inputs**:

  | Name | Role | Shape pattern | Description |
  |------|------|---------------|-------------|
  | `input_tensor` | LHS matrix | `[1, 1, M, K]` | M, K divisible by 32 |
  | `weight_tensor` | RHS matrix | `[1, 1, K, N]` | K, N divisible by 32; K matches input |
  | `bias` (kw-only) | row-broadcast bias, optional | `[1, 1, 32, N]` | row 0 holds the bias values; rows 1-31 zero-padded |

- **Output**: `[1, 1, M, N]`, `bfloat16`, `TILE_LAYOUT`.
- **Parameters**:

  | Name | Type | Default | Range | Description |
  |------|------|---------|-------|-------------|
  | `bias` | `ttnn.Tensor \| None` | `None` | shape `[1, 1, 32, N]` if present | optional row-broadcast bias |
  | `memory_config` | `ttnn.MemoryConfig \| None` | `ttnn.DRAM_MEMORY_CONFIG` | DRAM interleaved | output memory config |

- **PyTorch Reference**:
  ```python
  def reference(x, w, bias=None):
      out = x @ w
      if bias is not None:
          # bias is [1, 1, 32, N]; only row 0 is the bias.
          out = out + bias[..., 0:1, :]
      return out
  ```
- **Import Path**: `from ttnn.operations.linear import linear`
- **Function Signature**:
  ```python
  def linear(
      input_tensor: ttnn.Tensor,
      weight_tensor: ttnn.Tensor,
      *,
      bias: ttnn.Tensor | None = None,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor:
  ```
- **Validation** (raises `ValueError` before any device work):
  1. Input/weight/bias dtype == `bfloat16`.
  2. Input/weight/bias layout == `TILE_LAYOUT`.
  3. Rank 4, leading dims `[1, 1, ...]`.
  4. `input.shape[-1] == weight.shape[-2]` (K match).
  5. `bias.shape[-1] == weight.shape[-1]` (N match).
  6. `bias.shape[-2] == 32`.
  7. M, K, N each divisible by 32.

## Phases

> **Non-regression rule**: Every phase must pass all tests from prior phases.
> **Accuracy**: PCC > 0.99 to pass initially. Agents tighten tolerances and note achieved values in changelog.
> **Checkbox protocol**: Agents mark `[x]` when a phase is complete and all tests pass.

### [x] Phase 0 — Core Implementation
- **Cores**: single (CoreRange (0,0)–(0,0))
- **Dtype**: bfloat16
- **Layout**: TILE
- **Memory**: DRAM-interleaved
- **Compute config**: `MathFidelity.HiFi4`, `fp32_dest_acc_en=True`, `packer_l1_acc=false` (hard-coded)
- **Params**: `bias` (optional row-broadcast), `memory_config` (output)
- **Test shapes**: `(32, 32, 32)`, `(64, 64, 64)`, `(32, 128, 64)`, `(96, 32, 128)`, `(128, 96, 64)`
- **Achieved precision**: PCC ≥ 0.99999775 across all baseline shapes (no-bias and bias)

### [ ] Refinement 1 — Multi-core
- Distribute the `[Mt, Nt]` output tile grid across multiple Tensix cores using the canonical 2D-multicast bmm pattern (`reader_bmm_tile_layout_in0_sender_*` / `reader_bmm_tile_layout_in1_sender_*` pattern).
- Add semaphores for sender/receiver coordination on input and weight blocks.
- Keep `cb_partials` per-core (single-core CB sizing logic for now); shrink as needed once block sizing is data-driven.
- **Note from verifier**: today's `linear_program_descriptor.py:74-75` hard-codes `CoreCoord(0, 0)`; replace with a chosen grid (e.g., compute device's worker grid, or accept a `core_grid` kwarg). Reader and writer kernels currently use `TensorAccessor` per tile — they will need sender/receiver split.
- **Note from verifier**: data_transfer.md flags this as the highest-impact bandwidth refinement. Without multicast, naive multi-core duplicates input by `num_blocks_x×` and weight by `num_blocks_y×`.

### [ ] Refinement 2 — Compute kernel config exposure
- Add `compute_kernel_config: ttnn.ComputeKernelConfig | None = None` kwarg to `linear()`.
- Plumb through to the program descriptor's `ComputeConfigDescriptor`.
- Continue defaulting to `MathFidelity.HiFi4 + fp32_dest_acc_en=True`.
- **Note from verifier (numerical_stability.md)**: HiFi4 is over-conservative for bf16-input matmul (HiFi2 covers full bf16 mantissa in 2 passes). On Wormhole B0, `HiFi4 + fp32_dest_acc_en=true` is a known-bad combination (#38306). Exposing the config lets users dial down to HiFi2/HiFi3 for throughput or to skirt the WH B0 bug.
- **Note from verifier**: also expose `packer_l1_acc` so the K-blocked path (Refinement 4) can opt into hardware L1 accumulation instead of software spill/reload.

### [ ] Refinement 3 — fp32 partials in bias path
- Promote `cb_partials` from bf16 to `Float32` when bias is present.
- Configure `UnpackToDestMode::UnpackToDestFp32` for `cb_partials` so the bias add re-unpacks at full fp32 precision.
- Doubles `cb_partials` L1 footprint from `Mt*Nt × 2 KiB` to `Mt*Nt × 4 KiB`. At 256×256×256 with `cb_partials = 64 tiles`, that's 256 KiB → cap on this without K-blocking.
- **Note from verifier (numerical_stability.md)**: today's `cb_partials` is bf16, costing one rounding-to-nearest-even per output sub-block before bias add. Promoting it to fp32 is the next precision lever.

### [ ] Refinement 4 — K-blocking (`num_k_blocks > 1`)
- Switch `MatmulBlockShape::of(...)` to use `in0_block_w < Kt` and `num_k_blocks = Kt / in0_block_w`.
- Resize input/weight CBs from full-block sizing (`Mt*Kt` / `Kt*Nt`) down to per-block sizing (`Mt*in0_block_w` / `in0_block_w*Nt`).
- Required to support large K (e.g., K > 512) without overflowing L1.
- Pairs naturally with `packer_l1_acc=true` to avoid software spill/reload through `cb_partials` in the no-bias path.

### [ ] Refinement 5 — bfloat8_b / bfp4_b inputs
- Accept block-float input/weight tensors. The `matmul_block` helper supports them at the LLK level; add validation paths and let `TensorAccessor` handle the smaller page sizes.
- May require tweaking the compute kernel's `compute_kernel_hw_startup` and CB data formats.

### [ ] Refinement 6 — Activation fusion
- Add `activation: str | None = None` kwarg accepting `"relu"`, `"gelu"`, `"silu"` (etc.).
- Wire to `matmul_block`'s `PostComputeFn` (no-bias path) or `add_bias_bcast_rows`'s `PostBiasFn` (bias path).
- Mirrors the canonical `bmm_large_block_zm_fused_bias_activation.cpp` SFPU functor pattern.

### [ ] Refinement 7 — Batched matmul (general rank)
- Extend to rank-4 with non-trivial leading dims (`[B0, B1, M, K]`), broadcasting matmul.
- Map onto the `MatmulBlockShape::batch` parameter (the helper supports `batch > 1` natively).
- Validation needs to widen to allow leading dims != `[1, 1]`.

### [ ] Refinement 8 — Non-tile-aligned shapes
- Allow M, K, N not divisible by 32.
- Padding/masking path for the partial last tile in each dim.
- Validation today rejects this outright (`linear.py:96-97`); refinement removes that gate and adds the corresponding kernel-side handling.

### [ ] Refinement 9 — ROW_MAJOR input acceptance
- Today `linear()` rejects ROW_MAJOR inputs and requires the caller to host-side `.to_layout(TILE_LAYOUT)` first.
- Add an in-kernel tilize stage (using `tilize_helpers_dataflow.hpp`) to accept ROW_MAJOR input/weight/bias, removing the host-side conversion roundtrip.

### [ ] Refinement 10 — Sharded I/O
- Accept block-sharded or width-sharded input and/or weight tensors.
- For resident operands the reader becomes a pure CB hand-off (no DRAM read).
- Pairs with multi-core (Refinement 1) — sharding determines the natural per-core work distribution.

### [ ] Refinement 11 — Memory pressure
- After all the above, audit L1 usage at the largest realistic shapes (multi-core, K-blocked, fp32 partials).
- Cap CB sizes, add overflow detection, and document maximum per-core tile counts.
