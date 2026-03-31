# Operation Design: GroupNorm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute |
| Goal | Compute Group Normalization: divide channels into groups, normalize each group independently over spatial and channel dimensions |
| Math | `y[n,hw,c] = (x[n,hw,c] - mean[n,g]) / sqrt(var[n,g] + eps)` where `g = c // group_size`, `mean[n,g] = E[x]` over group, `var[n,g] = E[(x - mean)^2]` over group |
| Mode | Derivative (uses reduce, binary, copy helpers) |
| References | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| num_groups | uint32_t | yes | 1 to C, must divide C, (C / num_groups) must be divisible by 32 | - | CT |
| epsilon | float | no | > 0 | 1e-5 | CT |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | (N, 1, H\*W, C) where H\*W % 32 == 0 and C % 32 == 0 |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | same as input: (N, 1, H\*W, C) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved |

## Derived Quantities

| Symbol | Formula | Description |
|--------|---------|-------------|
| Ht | H\*W / 32 | Tile rows (spatial dimension) |
| Ct | C / 32 | Tile columns (channel dimension) |
| G | num_groups | Number of groups |
| group_size | C / G | Channels per group |
| Gt | group_size / 32 = Ct / G | Tile columns per group |
| reduce_count | H\*W \* group_size = Ht \* 32 \* Gt \* 32 | Elements reduced per group |
| inv_reduce_count | 1.0 / reduce_count | Reciprocal for mean/variance |
| total_tiles | N \* Ht \* Ct | Total tiles in input/output tensor |

## Dataflow Strategy

### High-Level Data Path

The operation processes each (batch, group) pair independently using a **two-pass** approach over the group's tiles:

1. **Statistics pass** (1 DRAM read): Simultaneously computes `sum(x)` and `sum(x^2)` per group using row-by-row streaming with accumulation. Derives mean and inverse standard deviation from these sums.
2. **Normalize pass** (1 DRAM read + 1 DRAM write): Reads group tiles again, subtracts mean, multiplies by inverse standard deviation, writes normalized output.

Total DRAM traffic: 2x input reads + 1x output write.

### Intra-Tensix Data Flow

```
DRAM ──[Reader/NoC0]──> cb_in ──[Compute]──> cb_out ──[Writer/NoC1]──> DRAM
                                    |
                           Compute-internal CBs:
                           cb_squared, cb_sum_accum,
                           cb_sumsq_accum, cb_mean,
                           cb_inv_std, cb_centered, cb_tmp
```

**Reader (BRISC)**: Reads tiles from DRAM in group-strided order. For group `g`, row `ht`, reads `Gt` contiguous tiles starting at page `n*Ht*Ct + ht*Ct + g*Gt`. Also prepares three scalar CBs at startup (reduce scaler, inv_count, epsilon).

**Compute (TRISC)**: Executes sequential helper phases for statistics and normalization. All helpers run on the 3 TRISCs (unpack + math + pack). Between sequential helpers, intermediate CBs must hold the full block produced by the previous helper.

**Writer (NCRISC)**: Writes tiles from cb_out to DRAM in the same group-strided order. Idle during the statistics pass (no tiles in cb_out).

### Inter-Tensix Communication

None. Single-core operation.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One (batch, group) pair at a time |
| Grid | Single core: CoreCoord(0, 0) |
| Per-core work | All N \* G groups, each requiring Ht \* Gt tiles per pass |
| Remainder | N/A (single core) |

### Processing Order

```
for n in [0, N):
    for g in [0, G):
        statistics_pass(n, g)   // 1 DRAM read of Ht*Gt tiles
        compute_statistics()     // single-tile ops, no DRAM
        normalize_pass(n, g)     // 1 DRAM read + 1 DRAM write of Ht*Gt tiles
```

## Circular Buffers

| CB ID | Name | Page Size | Num Pages | Format | Producer | Consumer | Purpose |
|-------|------|-----------|-----------|--------|----------|----------|---------|
| 0 | cb_in | tile_size(bf16) | Gt | bf16 | Reader | Compute | Input tiles. Gt pages for WaitUpfrontNoPop in statistics pass. |
| 8 | cb_scaler | tile_size(bf16) | 1 | bf16 | Reader | Compute | Reduce scaler (1.0). Persistent for entire op. |
| 9 | cb_inv_count | tile_size(bf16) | 1 | bf16 | Reader | Compute | 1/reduce_count scalar. Persistent for entire op. |
| 10 | cb_eps | tile_size(bf16) | 1 | bf16 | Reader | Compute | Epsilon scalar. Persistent for entire op. |
| 16 | cb_out | tile_size(bf16) | 2 | bf16 | Compute | Writer | Output tiles. Double-buffered for writer streaming. |
| 24 | cb_sum_accum | tile_size(bf16) | 2 | bf16 | Compute | Compute | Sum accumulator (statistics pass). 2 pages for Accumulate pattern. |
| 25 | cb_squared | tile_size(bf16) | Gt | bf16 | Compute | Compute | Squared tiles. Intermediate between square() and reduce() in statistics pass. Gt pages because sequential helpers cannot pipeline. |
| 26 | cb_sumsq_accum | tile_size(bf16) | 2 | bf16 | Compute | Compute | Sum-of-squares accumulator (statistics pass). 2 pages for Accumulate pattern. |
| 27 | cb_mean | tile_size(bf16) | 1 | bf16 | Compute | Compute | Mean tile. Persists from statistics computation through normalize pass. |
| 28 | cb_inv_std | tile_size(bf16) | 1 | bf16 | Compute | Compute | 1/sqrt(var+eps) tile. Persists through normalize pass. |
| 29 | cb_centered | tile_size(bf16) | Gt | bf16 | Compute | Compute | (x - mean) tiles. Intermediate between sub() and mul() in normalize pass. Gt pages because sequential helpers cannot pipeline. |
| 30 | cb_tmp | tile_size(bf16) | 1 | bf16 | Compute | Compute | Temporary for single-tile statistics sub-operations. |

**Total L1 for CBs**: (13 + 4\*Gt) \* tile_size(bf16). For Gt=1: 17 tiles = 34KB. For Gt=4: 29 tiles = 58KB.

**CB Page Count Rationale**:
- **cb_in (Gt pages)**: The statistics pass uses `WaitUpfrontNoPop` to keep tiles for both sum-reduce and square operations in the same row iteration. This requires all Gt tiles to be present.
- **cb_out (2 pages)**: Classic double-buffer for pipelining compute and writer.
- **cb_sum_accum, cb_sumsq_accum (2 pages each)**: The `Accumulate` pattern calls `cb_reserve_back` before `reload_accumulator_if_needed` (which pops). So the CB needs 1 page for the existing tile + 1 reserved page.
- **cb_squared, cb_centered (Gt pages each)**: Sequential compute helpers cannot pipeline. The producing helper (square / sub) must finish all Gt tiles before the consuming helper (reduce / mul) starts.
- **All other CBs (1 page)**: Single-tile scalars or temporaries.

## API Mapping

| Phase | Type | Function | File:Line | Template Params | CB In | CB Out | Manages Own CB Ops? |
|-------|------|----------|-----------|-----------------|-------|--------|---------------------|
| HW init | raw_api | compute_kernel_hw_startup | compute_kernel_hw_startup.h:41 | icb0=0, icb1=8, ocb=24 | - | - | N/A |
| Scaler prep | helper | dataflow_kernel_lib::prepare_reduce_scaler | reduce_helpers_dataflow.hpp:50 | cb_id=8, value=1.0f | - | 8 | Yes (reserve+push) |
| Scaler prep | helper | dataflow_kernel_lib::prepare_reduce_scaler | reduce_helpers_dataflow.hpp:50 | cb_id=9, value=inv_reduce_count | - | 9 | Yes (reserve+push) |
| Scaler prep | helper | dataflow_kernel_lib::prepare_reduce_scaler | reduce_helpers_dataflow.hpp:50 | cb_id=10, value=epsilon | - | 10 | Yes (reserve+push) |
| Stats: sum-reduce | helper | compute_kernel_lib::reduce | reduce_helpers_compute.hpp:431 | SUM, REDUCE_SCALAR, WaitUpfrontNoPop, INPUT_AND_OUTPUT | 0, 8 | 24 | Yes (wait/reserve/push; no pop due to WaitUpfrontNoPop) |
| Stats: square | helper | compute_kernel_lib::square | binary_op_helpers.hpp:314 | NoWaitNoPop, PerTile, INPUT_AND_OUTPUT | 0 | 25 | Yes for output; no wait/pop on input (NoWaitNoPop) |
| Stats: sumsq-reduce | helper | compute_kernel_lib::reduce | reduce_helpers_compute.hpp:431 | SUM, REDUCE_SCALAR, WaitAndPopPerTile, INPUT_AND_OUTPUT, Accumulate | 25, 8 | 26 | Yes (wait/pop/reserve/push + accumulate reload) |
| Stats: mean | helper | compute_kernel_lib::mul | binary_op_helpers.hpp:303 | SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop | 24, 9 | 27 | Yes; B not popped |
| Stats: mean-sq | helper | compute_kernel_lib::square | binary_op_helpers.hpp:314 | WaitUpfrontNoPop | 27 | 30 | Yes for output; input not popped |
| Stats: E[x^2] | helper | compute_kernel_lib::mul | binary_op_helpers.hpp:303 | SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop | 26, 9 | 24 | Yes; B not popped |
| Stats: variance | helper | compute_kernel_lib::sub | binary_op_helpers.hpp:290 | NONE, WaitAndPopPerTile, WaitAndPopPerTile | 24, 30 | 26 | Yes (both inputs popped) |
| Stats: var+eps | helper | compute_kernel_lib::add | binary_op_helpers.hpp:278 | SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop | 26, 10 | 30 | Yes; B not popped |
| Stats: rsqrt | helper | compute_kernel_lib::copy_tiles | copy_tile_helpers.hpp:148 | WaitAndPop, INPUT_AND_OUTPUT, post_op=rsqrt | 30 | 28 | Yes |
| Norm: center | helper | compute_kernel_lib::sub | binary_op_helpers.hpp:290 | SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop | 0, 27 | 29 | Yes; B not popped |
| Norm: scale | helper | compute_kernel_lib::mul | binary_op_helpers.hpp:303 | SCALAR, WaitAndPopPerTile, WaitUpfrontNoPop | 29, 28 | 16 | Yes; B not popped |
| Norm: pop mean | raw_api | cb_pop_front | api/compute/cb_api.h | cb=27, pages=1 | 27 | - | N/A |
| Norm: pop inv_std | raw_api | cb_pop_front | api/compute/cb_api.h | cb=28, pages=1 | 28 | - | N/A |
| Stats: pop cb_in | raw_api | cb_pop_front | api/compute/cb_api.h | cb=0, pages=Gt | 0 | - | N/A |

## Compute Phases

### Per-(batch, group) Execution

#### Statistics Pass (row-by-row, repeated Ht times)

| # | Operation | Helper | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|--------|-------------------------|-------------------|----------------|
| 1 | Reduce sum (WaitUpfrontNoPop) | reduce\<SUM, REDUCE_SCALAR, WaitUpfrontNoPop\> | cb_in (Gt tiles, waited), cb_scaler (1, persistent) | cb_sum_accum (1 tile, accumulate) | cb_in: Gt tiles still present (not popped). cb_sum_accum: accumulating. |
| 2 | Square (NoWaitNoPop) | square\<NoWaitNoPop\> | cb_in (Gt tiles, reuse from phase 1) | cb_squared (Gt tiles) | cb_in: still present. cb_squared: Gt tiles ready. |
| 3 | Reduce sumsq (Accumulate) | reduce\<SUM, REDUCE_SCALAR, WaitAndPopPerTile, Accumulate\> | cb_squared (Gt, consumed), cb_scaler (1, persistent) | cb_sumsq_accum (1 tile, accumulate) | cb_squared: empty. cb_sumsq_accum: accumulating. |
| 4 | Pop cb_in | cb_pop_front(cb_in, Gt) | cb_in (Gt tiles) | - | cb_in: empty. Reader can push next row. |

After Ht iterations: cb_sum_accum has 1 tile (total sum). cb_sumsq_accum has 1 tile (total sum of squares).

#### Statistics Computation (single-tile ops, no DRAM)

| # | Operation | Helper | Input CB | Output CB | CB State After |
|---|-----------|--------|----------|-----------|----------------|
| 5 | mean = sum \* inv_count | mul\<SCALAR\> | cb_sum_accum (consumed), cb_inv_count (persistent) | cb_mean (1) | cb_sum_accum: empty. cb_mean: has mean. |
| 6 | mean_sq = mean^2 | square\<WUPNoP\> | cb_mean (persistent) | cb_tmp (1) | cb_mean: persists. cb_tmp: has mean^2. |
| 7 | E[x^2] = sumsq \* inv_count | mul\<SCALAR\> | cb_sumsq_accum (consumed), cb_inv_count (persistent) | cb_sum_accum (1, reused) | cb_sumsq_accum: empty. cb_sum_accum: has E[x^2]. |
| 8 | var = E[x^2] - mean^2 | sub\<NONE\> | cb_sum_accum (consumed), cb_tmp (consumed) | cb_sumsq_accum (1, reused) | Both inputs freed. cb_sumsq_accum: has variance. |
| 9 | var + eps | add\<SCALAR\> | cb_sumsq_accum (consumed), cb_eps (persistent) | cb_tmp (1) | cb_tmp: has var + eps. |
| 10 | inv_std = rsqrt(var + eps) | copy_tiles + rsqrt | cb_tmp (consumed) | cb_inv_std (1) | cb_inv_std: has 1/sqrt(var + eps). |

After statistics: cb_mean (27) and cb_inv_std (28) persist for the normalize pass. All other intermediates are empty.

#### Normalize Pass (row-by-row, repeated Ht times)

| # | Operation | Helper | Input CB | Output CB | CB State After |
|---|-----------|--------|----------|-----------|----------------|
| 11 | centered = x - mean | sub\<SCALAR\> | cb_in (Gt, consumed), cb_mean (1, persistent) | cb_centered (Gt) | |
| 12 | output = centered \* inv_std | mul\<SCALAR\> | cb_centered (Gt, consumed), cb_inv_std (1, persistent) | cb_out (Gt) | Writer drains cb_out. |

After Ht iterations: all output tiles for this group written.

| 13 | Pop persistent mean | cb_pop_front(cb_mean, 1) | cb_mean | - | cb_mean: empty for next group. |
| 14 | Pop persistent inv_std | cb_pop_front(cb_inv_std, 1) | cb_inv_std | - | cb_inv_std: empty for next group. |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| Stats: mean | mul | Col0 (REDUCE_SCALAR output) | Row0 (prepare_reduce_scaler fills row 0 of faces) | SCALAR |
| Stats: mean_sq | square (self-mul) | Col0 (mean, from REDUCE_SCALAR chain) | same as A | NONE (self) |
| Stats: E[x^2] | mul | Col0 (REDUCE_SCALAR output) | Row0 (scaler tile) | SCALAR |
| Stats: variance | sub | Col0 (E[x^2]) | Col0 (mean^2) | NONE |
| Stats: var+eps | add | Col0 (variance) | Row0 (scaler tile) | SCALAR |
| Norm: center | sub | All (full input tile) | Col0 (mean, REDUCE_SCALAR chain) | SCALAR |
| Norm: scale | mul | All (centered tile) | Col0 (inv_std, REDUCE_SCALAR chain) | SCALAR |

**Note on valid regions**: REDUCE_SCALAR output has the scalar sum at tile position [0,0] with other positions zero. Binary ops with SCALAR broadcast read B[0,0] and replicate to all positions. The prepare_reduce_scaler helper fills row 0 of each face, which includes position [0,0]. Both formats are compatible with SCALAR broadcast.

## Build Order

1. **Passthrough**: Reader reads one group's tiles, compute copies to output (copy_tiles), writer writes. Verify with deterministic input (all 1s). DPRINT the first few tile values in reader, compute, and writer.

2. **Sum reduction**: Replace copy with `reduce<SUM, REDUCE_SCALAR>` for a single group. Verify sum against `torch.sum(input[:, :, :, 0:group_size])`. Use `torch.ones` input so expected sum = H\*W \* group_size.

3. **Sum-of-squares**: Add the square + reduce with accumulation loop. Verify sumsq against `torch.sum(input[:, :, :, 0:group_size] ** 2)`.

4. **Statistics chain**: Add the mean, variance, inv_std computation. Verify intermediate values with DPRINT against PyTorch reference for a known input (e.g., `torch.arange`-based).

5. **Normalize single group**: Add the sub(mean) + mul(inv_std) pass. Verify full GroupNorm output for `num_groups=1` against `torch.nn.functional.group_norm(input, 1)`.

6. **Multi-group**: Add the group loop. Verify for `num_groups > 1`.

7. **Multi-batch**: Add the batch loop. Verify for `N > 1`.

## Key Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| **Accumulator CB needs 2 pages**: The `Accumulate` pattern calls `cb_reserve_back` before `reload_accumulator_if_needed` (which pops the previous tile). If the CB has only 1 page, `cb_reserve_back` deadlocks because the CB is full. | cb_sum_accum and cb_sumsq_accum are sized to 2 pages. |
| **WaitUpfrontNoPop requires cb_in to hold Gt tiles**: The combined statistics pass waits for all Gt tiles upfront to reuse them for both sum-reduce and square. | cb_in is sized to Gt pages (not the usual 2). |
| **Manual cb_pop_front after WaitUpfrontNoPop + NoWaitNoPop chain**: Neither reduce (WaitUpfrontNoPop) nor square (NoWaitNoPop) pops cb_in. The compute kernel must manually pop Gt tiles after each row's combined statistics. | Explicit `cb_pop_front(cb_in, Gt)` at end of each row iteration. |
| **Sequential helper intermediates**: square/reduce and sub/mul are both all-TRISC helpers that cannot pipeline. The intermediate CB must hold the full block. | cb_squared and cb_centered sized to Gt pages. |
| **E[x^2] - E[x]^2 numerical stability**: Catastrophic cancellation possible for large input values. | Acceptable for bfloat16 precision. GroupNorm inputs are typically pre-normalized activations with moderate magnitudes. |
| **REDUCE_SCALAR output position**: The scalar sum is at tile position [0,0]. Subsequent SCALAR broadcast binary ops read from [0,0]. | Format is compatible. prepare_reduce_scaler also fills [0,0]. |
| **Scaler tile format for binary SCALAR broadcast**: prepare_reduce_scaler fills row 0 of each face. Binary SCALAR broadcast reads [0,0] from the B tile. | Row 0 includes position [0,0], so formats are compatible. |
| **rsqrt_tile_init() must precede rsqrt_tile()**: SFPU initialization protocol. | Included in the copy_tiles post_op lambda. |
| **Data format reconfig between reduce and binary helpers**: reduce_uninit() changes hardware state. | All helpers use `INPUT_AND_OUTPUT` reconfig mode by default, which reinitializes formats. |
| **group_size must be multiple of 32**: Groups must align to tile boundaries for correct reduce semantics. | Validated in program descriptor. Constraint: (C / num_groups) % 32 == 0. |
