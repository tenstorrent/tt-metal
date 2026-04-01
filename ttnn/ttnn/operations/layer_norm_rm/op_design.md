# Operation Design: layer_norm_rm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused normalization + optional affine) |
| Goal | Layer-normalize each row of a row-major interleaved tensor in-kernel, with optional gamma/beta affine transform |
| Math | `output[..., i] = (x[..., i] - mean(x)) / sqrt(var(x) + eps) * gamma[i] + beta[i]` |
| Mode | Standalone |
| References | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `reduce_helpers_compute.hpp`, `binary_op_helpers.hpp`, `untilize_helpers.hpp`, `tilize_helpers_dataflow.hpp` |

**Algorithm steps** (per row of W elements):
1. `mean = sum(x) / W`
2. `centered = x - mean`
3. `var = sum(centered^2) / W`
4. `inv_std = rsqrt(var + epsilon)`
5. `normalized = centered * inv_std`
6. *(if gamma)* `scaled = normalized * gamma`
7. *(if beta)* `output = scaled + beta` *(or `normalized + beta` if no gamma)*

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| epsilon | float | No | > 0 | 1e-5 | CT (packed as uint32 bits) |
| has_gamma | bool | No | 0 or 1 | 0 | CT |
| has_beta | bool | No | 0 or 1 | 0 | CT |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[*, H, W]` — at least 2D. H and W must be multiples of 32 (tile-aligned). |
| Dtype | bfloat16 |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | DRAM interleaved |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, 1, W]` — W must match input width |
| Dtype | bfloat16 |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | DRAM interleaved |

### Beta (optional)

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, 1, W]` — W must match input width |
| Dtype | bfloat16 |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | Same as input |
| Dtype | bfloat16 |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | DRAM interleaved |

## Dataflow Strategy

All tensors arrive in ROW_MAJOR_LAYOUT, interleaved across DRAM banks. Compute requires tile format. The data pipeline is:

```
DRAM ──Reader──► cb_rm_in (RM sticks) ──Tilize──► cb_x (tiles)
                                                      │
                                         ┌─ reduce mean ──► cb_reduce (mean)
                                         │            │
                                         └── sub ─────► cb_centered
                                                           │
                                              ┌─ square ──► cb_sq ── reduce var ──► cb_reduce (var)
                                              │                                        │
                                              │              add eps + rsqrt ──────► cb_inv_std
                                              │                                        │
                                              └── mul ─────────────────────────────► cb_normed
                                                                                       │
                                                          (optional gamma/beta phases) │
                                                                                       ▼
                                         cb_rm_out (RM sticks) ◄──Untilize── final result
                                              │
                                          ──Writer──► DRAM
```

**Gamma/beta** (if present): read once by the reader, tilized once by compute at the start. Tilized gamma/beta tiles persist in CBs across all blocks.

**Processing granularity**: 1 tile-row block per iteration = 32 rows × Wt tiles. The compute kernel loops over `NC * Ht` blocks, where `NC = product(dims[:-2])`, `Ht = H / 32`, `Wt = W / 32`.

**No inter-Tensix communication**: single-core operation, all data flows through CBs within one Tensix core.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | 1 tile-row block (32 rows × W elements = Wt tiles) |
| Grid | Single core: `CoreCoord(0, 0)` |
| Per-core work | All blocks: `NC * Ht` tile-row blocks |
| Remainder | None (single core processes everything) |

**Derived constants**:
- `Wt = W / 32` (tiles per row)
- `Ht = H / 32` (tile-rows per batch)
- `NC = product(all dims except last two)` (batch count)
- `total_num_rows = NC * Ht * 32 = NC * H`
- `num_blocks = NC * Ht`
- `row_bytes = W * element_size` (bytes per RM stick)

## Circular Buffers

| CB ID | Name | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|-------|------|-----------|-----------|--------|----------|----------|----------|
| 0 | cb_rm_in | tile_size(dtype) | 2 × Wt | input dtype | Reader (read_sticks_for_tilize) | Tilize | Per-block, double-buffered |
| 1 | cb_gamma_rm | padded_row_bytes | 1 | input dtype | Reader (read_sticks_for_tilize) | Tilize (gamma) | Once at start (if gamma) |
| 2 | cb_beta_rm | padded_row_bytes | 1 | input dtype | Reader (read_sticks_for_tilize) | Tilize (beta) | Once at start (if beta) |
| 8 | cb_scaler | tile_size(bf16) | 1 | bfloat16 | Reader (prepare_reduce_scaler) | Reduce (mean + var) | Persistent (popped at end) |
| 9 | cb_eps | tile_size(bf16) | 1 | bfloat16 | Reader (prepare_reduce_scaler) | Add (eps) | Persistent (popped at end) |
| 16 | cb_rm_out | tile_size(dtype) | 2 × Wt | output dtype | Untilize | Writer (write_sticks_after_untilize) | Per-block, double-buffered |
| 24 | cb_x | tile_size(dtype) | Wt | input dtype | Tilize | Reduce mean, Sub mean | Per-block. Reused as output CB in affine phases. |
| 25 | cb_reduce | tile_size(dtype) | 1 | input dtype | Reduce | Sub mean / Add eps | Per-block (reused for mean and variance) |
| 26 | cb_centered | tile_size(dtype) | Wt | input dtype | Sub | Square, Mul inv_std | Per-block. Reused as output CB in gamma phase. |
| 27 | cb_sq | tile_size(dtype) | Wt | input dtype | Square | Reduce var | Per-block (freed after reduce var) |
| 28 | cb_inv_std | tile_size(dtype) | 1 | input dtype | Add+rsqrt | Mul inv_std | Per-block |
| 29 | cb_normed | tile_size(dtype) | Wt | input dtype | Mul inv_std | Gamma/Beta/Untilize | Per-block. Reused as output CB in beta phase. |
| 30 | cb_gamma_t | tile_size(dtype) | Wt | input dtype | Tilize (gamma) | Mul gamma | Persistent (if gamma, popped at end) |
| 31 | cb_beta_t | tile_size(dtype) | Wt | input dtype | Tilize (beta) | Add beta | Persistent (if beta, popped at end) |

**CB page size notes**:
- `tile_size(dtype)` = `ttnn.tile_size(input_tensor.dtype)` (2048 for bf16, 4096 for f32)
- `padded_row_bytes` = `ceil(row_bytes / (32 * elem_size)) * (32 * elem_size)` = `Wt * 32 * elem_size`
- cb_scaler and cb_eps are always bfloat16 regardless of input dtype (reduce scaler format requirement)

**CB sizing rationale**:
- cb_rm_in / cb_rm_out: 2×Wt pages for double-buffering between reader/compute and compute/writer
- cb_x, cb_centered, cb_sq, cb_normed: Wt pages — full tile-row block. Required because sequential compute helpers own all TRISCs; an intermediate CB between two helpers must hold the entire output of the first helper.
- cb_reduce, cb_inv_std: 1 page — REDUCE_ROW output is a single column-vector tile
- cb_gamma_t, cb_beta_t: Wt pages — tilized gamma/beta persist across all blocks for ROW broadcast
- cb_gamma_rm, cb_beta_rm: 1 page (1 row) — ROW granularity tilize for the single-row gamma/beta tensors

**Memory budget** (bfloat16, tile_size=2048):
- Without affine: `(2+2+1+1+1+1)*Wt*2048 + 4*2048` = `8*Wt*2048 + 8192`
- With gamma+beta: `(2+2+1+1+1+1+1+1)*Wt*2048 + 4*2048` = `10*Wt*2048 + 8192`
- Maximum W (with gamma+beta, 1.5MB L1): ~Wt=72, W≈2304

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference.

| Phase | Type | Function | File:Line | Template Params / Args | CB In | CB Out | Manages Own CB Ops |
|-------|------|----------|-----------|----------------------|-------|--------|---------------------|
| Tilize input | helper | `compute_kernel_lib::tilize` | tilize_helpers.hpp:130 | `<Wt, cb_rm_in, cb_x>` (num_blocks=1) | cb_rm_in (wait Wt, pop Wt) | cb_x (reserve Wt, push Wt) | Yes |
| Tilize gamma | helper | `compute_kernel_lib::tilize` | tilize_helpers.hpp:130 | `<Wt, cb_gamma_rm, cb_gamma_t>` (num_blocks=1, total_input_pages=1). Asymmetric mode for 1-row input. | cb_gamma_rm (wait 1, pop 1) | cb_gamma_t (reserve Wt, push Wt) | Yes |
| Tilize beta | helper | `compute_kernel_lib::tilize` | tilize_helpers.hpp:130 | `<Wt, cb_beta_rm, cb_beta_t>` (num_blocks=1, total_input_pages=1). Asymmetric mode for 1-row input. | cb_beta_rm (wait 1, pop 1) | cb_beta_t (reserve Wt, push Wt) | Yes |
| Reduce mean | helper | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` | reduce_helpers_compute.hpp:515 | input_policy=WaitUpfrontNoPop (tiles persist for sub phase) | cb_x (wait Wt upfront, no pop) + cb_scaler | cb_reduce (1 tile) | Yes |
| Sub mean | helper | `compute_kernel_lib::sub<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` | binary_op_helpers.hpp:290 | bcast_dim=COL. A already waited in reduce phase. | cb_x (no wait, pop Wt at end) + cb_reduce (wait 1, pop 1 per row) | cb_centered (Wt tiles) | Yes |
| Square | helper | `compute_kernel_lib::square<WaitUpfrontNoPop>` | binary_op_helpers.hpp:313 | input_policy=WaitUpfrontNoPop (centered tiles persist for mul inv_std) | cb_centered (wait Wt, no pop) | cb_sq (Wt tiles) | Yes |
| Reduce var | helper | `compute_kernel_lib::reduce<SUM, REDUCE_ROW>` | reduce_helpers_compute.hpp:515 | Default WaitAndPopPerTile | cb_sq (wait 1, pop 1 per tile) + cb_scaler | cb_reduce (1 tile) | Yes |
| Add eps + rsqrt | helper | `compute_kernel_lib::add<SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop>` | binary_op_helpers.hpp:277 | bcast_dim=SCALAR. B=eps persists. post_op=rsqrt_tile | cb_reduce (wait 1, pop 1) + cb_eps (wait 1, no pop) | cb_inv_std (1 tile) | Yes |
| Mul inv_std | helper | `compute_kernel_lib::mul<COL, NoWaitPopAtEnd, WaitAndPopPerTile>` | binary_op_helpers.hpp:303 | bcast_dim=COL. A already waited in square phase. | cb_centered (no wait, pop Wt at end) + cb_inv_std (wait 1, pop 1 per row) | cb_normed (Wt tiles) | Yes |
| Mul gamma | helper | `compute_kernel_lib::mul<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>` | binary_op_helpers.hpp:303 | bcast_dim=ROW. B=gamma persists. | cb_normed (wait 1, pop 1) + cb_gamma_t (wait Wt, no pop) | cb_x (Wt tiles, reused) | Yes |
| Add beta | helper | `compute_kernel_lib::add<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>` | binary_op_helpers.hpp:277 | bcast_dim=ROW. B=beta persists. Input depends on gamma presence. | (see phase table) + cb_beta_t (wait Wt, no pop) | (see phase table) | Yes |
| Untilize | helper | `compute_kernel_lib::untilize<Wt, cb_final, cb_rm_out>` | untilize_helpers.hpp:137 | cb_final depends on affine config (see Compute Phases) | cb_final (wait Wt, pop Wt) | cb_rm_out (Wt tiles) | Yes |
| Read input sticks | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in, TILE>` | tilize_helpers_dataflow.hpp:78 | TILE granularity | DRAM → cb_rm_in | — | Yes |
| Read gamma sticks | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_rm, ROW>` | tilize_helpers_dataflow.hpp:78 | ROW granularity, 1 row | DRAM → cb_gamma_rm | — | Yes |
| Read beta sticks | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_beta_rm, ROW>` | tilize_helpers_dataflow.hpp:78 | ROW granularity, 1 row | DRAM → cb_beta_rm | — | Yes |
| Write output sticks | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_rm_out>` | tilize_helpers_dataflow.hpp:108 | — | cb_rm_out → DRAM | — | Yes |
| Prepare reduce scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>` | reduce_helpers_dataflow.hpp:65 | scaler_f = 1.0f / W | — → cb_scaler | — | Yes |
| Prepare epsilon tile | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>` | reduce_helpers_dataflow.hpp:65 | scaler_f = epsilon | — → cb_eps | — | Yes |

## Compute Phases

Processing is per tile-row block (1 of `NC * Ht` blocks). Each block = 32 rows × Wt tiles.

**One-time setup** (before main loop):

| # | Operation | Helper | Input CB | Output CB | CB State After |
|---|-----------|--------|----------|-----------|----------------|
| S1 | Tilize gamma (if has_gamma) | `tilize<Wt, cb_gamma_rm, cb_gamma_t>(1, 1)` | cb_gamma_rm (1 row) | cb_gamma_t (Wt tiles) | cb_gamma_t holds Wt tiles persistently |
| S2 | Tilize beta (if has_beta) | `tilize<Wt, cb_beta_rm, cb_beta_t>(1, 1)` | cb_beta_rm (1 row) | cb_beta_t (Wt tiles) | cb_beta_t holds Wt tiles persistently |

**Per-block loop** (NC * Ht iterations):

| # | Operation | Helper | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|--------|-------------------------|-------------------|----------------|
| 1 | Tilize | `tilize<Wt, cb_rm_in, cb_x>(1)` | cb_rm_in (Wt pages, consumed) | cb_x (Wt tiles) | cb_x: Wt tiles live |
| 2 | Reduce mean | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(cb_x, cb_scaler, cb_reduce, {1, Wt})` | cb_x (Wt, **persist**) + cb_scaler | cb_reduce (1 tile = mean) | cb_x: Wt live; cb_reduce: 1 live |
| 3 | Sub mean | `sub<COL, NoWaitPopAtEnd, WaitAndPopPerTile>(cb_x, cb_reduce, cb_centered, {1, Wt})` | cb_x (Wt, **pop at end**) + cb_reduce (1, **pop**) | cb_centered (Wt tiles) | cb_x: freed; cb_reduce: freed; cb_centered: Wt live |
| 4 | Square | `square<WaitUpfrontNoPop>(cb_centered, cb_sq, {1, Wt})` | cb_centered (Wt, **persist**) | cb_sq (Wt tiles) | cb_centered: Wt live; cb_sq: Wt live |
| 5 | Reduce var | `reduce<SUM, REDUCE_ROW>(cb_sq, cb_scaler, cb_reduce, {1, Wt})` | cb_sq (Wt, **consumed**) + cb_scaler | cb_reduce (1 tile = variance) | cb_sq: freed; cb_centered: Wt live; cb_reduce: 1 live |
| 6 | Add eps + rsqrt | `add<SCALAR, WaitAndPop, WaitUpfrontNoPop>(cb_reduce, cb_eps, cb_inv_std, {1,1}, rsqrt_post_op)` | cb_reduce (1, **pop**) + cb_eps (**persist**) | cb_inv_std (1 tile) | cb_reduce: freed; cb_centered: Wt live; cb_inv_std: 1 live |
| 7 | Mul inv_std | `mul<COL, NoWaitPopAtEnd, WaitAndPopPerTile>(cb_centered, cb_inv_std, cb_normed, {1, Wt})` | cb_centered (Wt, **pop at end**) + cb_inv_std (1, **pop**) | cb_normed (Wt tiles) | cb_centered: freed; cb_inv_std: freed; cb_normed: Wt live |
| 8 | Mul gamma *(if has_gamma)* | `mul<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>(cb_normed, cb_gamma_t, cb_x, {1, Wt})` | cb_normed (Wt, **consumed**) + cb_gamma_t (**persist**) | cb_x (Wt tiles, reused) | cb_normed: freed; cb_x: Wt live |
| 9 | Add beta *(if has_beta)* | `add<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>(cb_in9, cb_beta_t, cb_out9, {1, Wt})` | cb_in9 + cb_beta_t (**persist**) → cb_out9 | (see below) | cb_out9: Wt live |
| 10 | Untilize | `untilize<Wt, cb_final, cb_rm_out>(1)` | cb_final (Wt, consumed) | cb_rm_out (Wt tile-pages) | All intermediates freed |

**Phase 9 CB routing** (depends on has_gamma):

| has_gamma | has_beta | Phase 9 input (cb_in9) | Phase 9 output (cb_out9) | cb_final for untilize |
|-----------|----------|------------------------|--------------------------|----------------------|
| false | false | — (skip) | — | cb_normed (CB29) |
| true | false | — (skip) | — | cb_x (CB24) |
| false | true | cb_normed (CB29) | cb_x (CB24) | cb_x (CB24) |
| true | true | cb_x (CB24) | cb_normed (CB29) | cb_normed (CB29) |

**Pattern**: `cb_final = (num_affine_ops == 1) ? cb_x : cb_normed` where `num_affine_ops = has_gamma + has_beta`.

**End-of-kernel cleanup** (after all blocks):

```
cb_pop_front(cb_scaler, 1);
cb_pop_front(cb_eps, 1);
if (has_gamma) cb_pop_front(cb_gamma_t, Wt);
if (has_beta) cb_pop_front(cb_beta_t, Wt);
```

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|----|-------------------|-------------------|---------------|
| 3 (sub mean) | SUB | All (2D [1, Wt]) | Col0 (REDUCE_ROW output) | COL |
| 6 (add eps) | ADD | Col0 (REDUCE_ROW output) | Scalar (single value) | SCALAR |
| 7 (mul inv_std) | MUL | All (2D [1, Wt]) | Col0 (derived from REDUCE_ROW) | COL |
| 8 (mul gamma) | MUL | All (2D [1, Wt]) | Row0 (gamma, [1, Wt] tiles) | ROW |
| 9 (add beta) | ADD | All (2D [1, Wt]) | Row0 (beta, [1, Wt] tiles) | ROW |

**Gamma/beta tile format note**: Gamma and beta are [1,1,1,W] tensors tilized from 1 RM row. The resulting tiles have valid data in row 0 only. ROW broadcast in the LLK replicates B's row across all H positions within the tile, so row 0 values are correctly broadcast to all 32 rows.

## Build Order

Incremental bring-up sequence for the implementer. Each stage adds one piece and verifies with DPRINT.

| Stage | What to add | Verify with |
|-------|-------------|-------------|
| 1. Passthrough | Reader → tilize → untilize → writer (no compute). Input = `torch.arange(W).repeat(H, 1)`. Output should equal input. | DPRINT cb_x tile[0] after tilize. Compare output tensor. |
| 2. Reduce mean | Add reduce<SUM, REDUCE_ROW> with scaler=1/W. Replace untilize input with cb_reduce. Output should be per-row means. | DPRINT cb_reduce tile. Input = `[[1,2,3,...,W], ...]`, expect mean = (W+1)/2. |
| 3. Sub + square | Add sub<COL> and square phases. Output the squared-centered values. | Input = constant row → centered = 0, squared = 0. Input = `[0, 1, ...]` → verify manually. |
| 4. Variance + inv_std | Add reduce var, add eps, rsqrt. Output inv_std tile. | Input = constant row → var = 0, inv_std = 1/sqrt(eps). |
| 5. Full normalize | Add mul inv_std. Connect untilize to cb_normed. Compare with `(x - mean) / sqrt(var + eps)` in PyTorch. | `F.layer_norm(input, [W])` — this is the pure normalization case. |
| 6. Gamma | Add gamma tilize + mul<ROW>. Test with gamma = 2.0 everywhere → output should be 2× normalized. | Compare with `F.layer_norm(input, [W], weight=gamma)`. |
| 7. Beta | Add beta tilize + add<ROW>. Test with beta = 1.0 everywhere → output should be normalized + 1. | Compare with `F.layer_norm(input, [W], weight=gamma, bias=beta)`. |

## Key Risks and Gotchas

1. **Sequential helper intermediate sizing**: cb_x, cb_centered, cb_sq, cb_normed must each hold Wt tiles. If Wt is too large, L1 will overflow. Maximum supported W ≈ 2304 (with gamma+beta, bfloat16).

2. **Epsilon tile format**: The epsilon tile is created via `prepare_reduce_scaler<cb_eps>(eps_float)`, which fills row 0 of each tile face. For `add<SCALAR>` broadcast, the LLK extracts from face 0 and broadcasts. This works because prepare_reduce_scaler fills face 0 row 0 with the epsilon value. If issues arise, replace with a full-tile fill (write epsilon to all 512 uint32 positions).

3. **Reduce scaler CB must be bfloat16**: Even if the input dtype differs, cb_scaler and cb_eps must use bfloat16 format. The reduce LLK expects bfloat16-packed scaler tiles.

4. **Gamma/beta ROW broadcast with partial tiles**: Gamma/beta are tilized from 1 row, producing tiles with valid data only in row 0. The ROW broadcast LLK replicates row 0 across all rows within the tile. This is the documented behavior of BroadcastType::ROW.

5. **cb_scaler persistence**: The reduce helper waits on cb_scaler but does NOT pop it. The same scaler tile is reused for both mean and variance reductions across all blocks. Pop only at the very end.

6. **cb_eps persistence**: The `add<SCALAR>` for epsilon uses `WaitUpfrontNoPop` for the B input policy. This ensures the epsilon tile persists across all blocks without re-creation.

7. **Affine CB routing**: The gamma and beta phases alternate between cb_x (CB24) and cb_normed (CB29) as input/output to avoid read-write conflicts on the same CB. The untilize source CB depends on the number of affine operations (see Compute Phases table).

8. **compute_kernel_hw_startup**: Must be called once at the start of the compute kernel before any helper. Use the 3-arg form: `compute_kernel_hw_startup(cb_rm_in, cb_scaler, cb_x)`. Each helper reconfigures unpack/pack registers via its reconfig_mode.

9. **Asymmetric tilize for gamma/beta**: Gamma/beta have only 1 row. Use the asymmetric tilize mode: `tilize<Wt, cb_gamma_rm, cb_gamma_t>(1, 1)` where the second argument is total_input_pages=1. The cb_gamma_rm page_size must be `padded_row_bytes` (ROW granularity).

10. **Scaler value**: The reduce scaler for mean and variance is `1.0f / W` where W is the **element count** in the last dimension (not Wt). This computes `sum(x_i * (1/W)) = mean(x)` via `reduce<SUM, REDUCE_ROW>`.
