# Last-Dim Reduction Pattern Analysis: `moreh_norm_program_factory_w_other.cpp`

## Overview

Primary reference: `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_program_factory_w_other.cpp`

Supporting references:

- `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/moreh_norm_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/reader_moreh_norm_w.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/moreh_norm_w_kernel.cpp`
- `ttnn/cpp/ttnn/operations/moreh/moreh_norm/device/ord_other/moreh_norm_w/kernels/writer_moreh_norm_w.cpp`

This factory is selected when `dim == input_rank - 1`, so it is the local reference for reducing over the last logical dimension. `moreh_norm_device_operation.cpp:112-123`

The math is not layernorm math; it computes specialized norms for `p == 0` or “other” cases using `SUM` or `MAX`. `moreh_norm_program_factory_w_other.cpp:112-121`

What is directly useful for layernorm is the scheduling model:

- work is partitioned by tile rows,
- each work unit consumes all `Wt` tiles along the last dimension for one row,
- tail width is masked explicitly using the original logical width,
- one output tile is produced per row unit. `moreh_norm_program_factory_w_other.cpp:35-39`, `moreh_norm_program_factory_w_other.cpp:151-189`, `reader_moreh_norm_w.cpp:30-49`, `moreh_norm_w_kernel.cpp:46-155`, `writer_moreh_norm_w.cpp:25-34`

## Why This Reference Matters For Layernorm

Layernorm over the last logical dimension needs a reduction domain that is “row-like”:

- for each logical row, consume all tiles spanning the final axis,
- compute row statistics,
- then normalize the original row against those statistics.

This reference already implements the first half of that scheduling problem:

- map one row unit to `Wt` input tiles,
- distribute row units across cores,
- deal with padded tail columns safely,
- and collapse each row unit into one reduced output tile.

## Program Selection Context

`MorehNormOperation::select_program_factory(...)` dispatches:

- `ProgramFactoryWOther` for `dim == input_rank - 1`
- `ProgramFactoryHOther` for `dim == input_rank - 2`
- `ProgramFactoryNCOther` otherwise `moreh_norm_device_operation.cpp:112-123`

That confirms the analyzed factory is intentionally the “reduce last dimension” path rather than a generic fallback.

## Work-Unit Definition

### Row-unit granularity

The factory derives:

- `H = padded_shape[-2]`
- `W = padded_shape[-1]`
- `Ht = H / TILE_HEIGHT`
- `Wt = W / TILE_WIDTH`
- `num_units = input.physical_volume() / H / W * Ht` `moreh_norm_program_factory_w_other.cpp:29-39`

Interpretation:

- `input.physical_volume() / H / W` is the product of all outer dimensions preceding the final `H x W` tile plane.
- Multiplying by `Ht` converts each tile-row within that plane into a separate work unit.

So one work unit is:

- one tile row in the reduction plane,
- spanning all `Wt` tiles along the last dimension.

This is the most important transferable pattern for layernorm. If layernorm normalizes over the last dimension, the natural work unit is likewise “one logical row across all width tiles.”

### Output correspondence

Reader consumes `Wt` input tiles per row unit. `reader_moreh_norm_w.cpp:41-48`

Writer emits exactly one output tile per row unit:

- `start_tile_idx = tile_offset / Wt`
- then writes one output tile per `row_idx`. `writer_moreh_norm_w.cpp:25-34`

That is a clean reduction mapping:

- input rows are tile-expanded along width,
- reduced output rows collapse width tiles back to one output tile each.

## Core Distribution Strategy

### Partitioning

The factory uses the full compute grid and calls:

```cpp
split_work_to_cores(grid, num_units)
```

to distribute row units across cores. `moreh_norm_program_factory_w_other.cpp:45-59`

Each core receives `num_units_per_core`, which here means “number of row units,” not number of tiles. `moreh_norm_program_factory_w_other.cpp:154-188`

### Tile offset progression

The runtime-setup loop carries a `tile_offset` measured in input tiles. `moreh_norm_program_factory_w_other.cpp:151-189`

After assigning `num_units_per_core` row units to a core, it advances:

```cpp
tile_offset += num_units_per_core * Wt;
```

`moreh_norm_program_factory_w_other.cpp:188`

This is the critical scheduling identity:

- one row unit consumes exactly `Wt` input tiles,
- so the next core’s starting input tile is previous offset plus `rows * Wt`.

For layernorm, this is more directly reusable than batch norm’s flat tile partitioning because it preserves row boundaries by construction.

## Dataflow

| Stage | Inputs consumed | Outputs produced | Notes |
| --- | --- | --- | --- |
| Reader | input tensor, logical `origin_w` | input tiles in `c_0`, constant-one tile in `c_1`, optional tail mask in `c_2` | Reader precomputes both constant and tail-mask auxiliaries once per core. `reader_moreh_norm_w.cpp:23-36` |
| Compute | `c_0`, `c_1`, optional `c_2` | reduced output tile in `c_16`, intermediates in `c_24`-`c_26` | Compute walks all `Wt` tiles for each row, merges them, then reduces within tile rows. `moreh_norm_w_kernel.cpp:46-155` |
| Writer | reduced output CB `c_16` | global output tensor | Writer emits one tile per row unit. `writer_moreh_norm_w.cpp:25-34` |

### End-to-end flow

1. Reader fills `c_1` with a constant one tile. `reader_moreh_norm_w.cpp:26-29`
2. If `origin_w` is not a multiple of 32, reader generates `mask_w` into `c_2`. `reader_moreh_norm_w.cpp:30-36`
3. Reader streams `num_rows_per_core * Wt` input tiles into `c_0`, preserving row-major order across the last dimension. `reader_moreh_norm_w.cpp:38-49`
4. Compute consumes one row at a time:
   - transforms each input tile into `f(x)` with optional masking on the last width tile,
   - merges all `Wt` tiles into one accumulated tile,
   - performs an intra-tile row reduction,
   - emits one reduced tile to `c_16`. `moreh_norm_w_kernel.cpp:46-155`
5. Writer drains one output tile per row to DRAM. `writer_moreh_norm_w.cpp:28-34`

## Tail Handling Pattern

This is the most directly relevant piece for layernorm correctness on padded widths.

### Host/runtime side

The factory passes `origin_w = input.logical_shape()[input_rank - 1]` into reader and compute runtime args. `moreh_norm_program_factory_w_other.cpp:40`, `moreh_norm_program_factory_w_other.cpp:166-185`

### Reader side

Reader computes:

- `do_mask_w = (origin_w % 32) != 0`
- `mask_w = origin_w % 32` when needed

and generates a mask tile into `c_2`. `reader_moreh_norm_w.cpp:30-36`

### Compute side

On the last width tile of each row (`col_idx == Wt - 1`), compute applies the width mask before transforming/reducing the tile. `moreh_norm_w_kernel.cpp:56-65`

That means padded elements in the final tile do not contribute to the reduction.

For layernorm, the same requirement exists:

- mean and variance over the last logical dimension must ignore padded columns in the last tile.

This reference provides a proven pattern for that tail masking.

## Circular Buffer Table

| CB | Meaning | Producer | Consumer | Evidence |
| --- | --- | --- | --- | --- |
| `c_0` | input row tiles | reader | compute | `moreh_norm_program_factory_w_other.cpp:81`, `reader_moreh_norm_w.cpp:16-24`, `moreh_norm_w_kernel.cpp:14-17` |
| `c_1` | constant one tile | reader | compute reduction helper | `moreh_norm_program_factory_w_other.cpp:82`, `reader_moreh_norm_w.cpp:26-29`, `moreh_norm_w_kernel.cpp:15`, `moreh_norm_w_kernel.cpp:133-134` |
| `c_2` | width mask tile | reader | compute | `moreh_norm_program_factory_w_other.cpp:83`, `reader_moreh_norm_w.cpp:31-36`, `moreh_norm_w_kernel.cpp:16`, `moreh_norm_w_kernel.cpp:42-44`, `moreh_norm_w_kernel.cpp:56-65` |
| `c_16` | reduced output tile | compute | writer | `moreh_norm_program_factory_w_other.cpp:84`, `moreh_norm_w_kernel.cpp:18-19`, `moreh_norm_w_kernel.cpp:139-155`, `writer_moreh_norm_w.cpp:17-18` |
| `c_24` | `f(x)` per tile | compute | compute | `moreh_norm_program_factory_w_other.cpp:85`, `moreh_norm_w_kernel.cpp:21-28`, `moreh_norm_w_kernel.cpp:48-85` |
| `c_25` | row accumulator across width tiles | compute | compute | `moreh_norm_program_factory_w_other.cpp:86`, `moreh_norm_w_kernel.cpp:27`, `moreh_norm_w_kernel.cpp:87-130` |
| `c_26` | reduced row result | compute | compute | `moreh_norm_program_factory_w_other.cpp:87`, `moreh_norm_w_kernel.cpp:28`, `moreh_norm_w_kernel.cpp:132-154` |

## Kernel Responsibilities

### Reader kernel

Responsibilities:

- load contiguous input tiles for each assigned row unit,
- materialize constant helper tiles,
- generate a width mask for the last tile when the logical width is not tile-aligned. `reader_moreh_norm_w.cpp:7-49`

The reader does not branch per mathematical mode beyond mask generation; all norm semantics are pushed into compute.

### Compute kernel

Responsibilities:

1. Consume `Wt` tiles for one row.
2. Transform each tile into `f(x)`:
   - `x != 0` when `p == 0`
   - `abs(x)` otherwise
   - with sign inversion special handling for `-inf` mode. `moreh_norm_program_factory_w_other.cpp:112-121`, `moreh_norm_w_kernel.cpp:48-85`
3. Combine width tiles into a single accumulator tile (`cb_cal`). `moreh_norm_w_kernel.cpp:87-130`
4. Invoke `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(..., REDUCE_ROW, ...)` to collapse within-tile rows. `moreh_norm_program_factory_w_other.cpp:112-117`, `moreh_norm_w_kernel.cpp:132-134`
5. Emit one output tile for the row. `moreh_norm_w_kernel.cpp:136-155`

Even though the math is different from layernorm, steps 1, 3, 4, and the tail-mask handling are directly relevant.

### Writer kernel

Responsibilities:

- map the input-tile offset to output-row tile index by dividing by `Wt`,
- write one reduced tile per row unit. `writer_moreh_norm_w.cpp:25-34`

This is a compact example of how last-dim reduction changes the output indexing relative to the input indexing.

## Runtime Argument Mapping

### Reader runtime args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | input base address | `moreh_norm_program_factory_w_other.cpp:166-173`, `reader_moreh_norm_w.cpp:9` |
| `1` | input is DRAM flag | `moreh_norm_program_factory_w_other.cpp:166-173`, `reader_moreh_norm_w.cpp:10` |
| `2` | `num_rows_per_core` | `moreh_norm_program_factory_w_other.cpp:169`, `reader_moreh_norm_w.cpp:11` |
| `3` | `Wt` | `moreh_norm_program_factory_w_other.cpp:170`, `reader_moreh_norm_w.cpp:12` |
| `4` | `tile_offset` | `moreh_norm_program_factory_w_other.cpp:171`, `reader_moreh_norm_w.cpp:13` |
| `5` | logical `origin_w` | `moreh_norm_program_factory_w_other.cpp:172`, `reader_moreh_norm_w.cpp:14` |

### Writer runtime args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | output base address | `moreh_norm_program_factory_w_other.cpp:176-178`, `writer_moreh_norm_w.cpp:11` |
| `1` | output is DRAM flag | `moreh_norm_program_factory_w_other.cpp:176-178`, `writer_moreh_norm_w.cpp:12` |
| `2` | `num_rows_per_core` | `moreh_norm_program_factory_w_other.cpp:177`, `writer_moreh_norm_w.cpp:13` |
| `3` | `Wt` | `moreh_norm_program_factory_w_other.cpp:177`, `writer_moreh_norm_w.cpp:14` |
| `4` | input-space `tile_offset` | `moreh_norm_program_factory_w_other.cpp:177`, `writer_moreh_norm_w.cpp:15` |

### Compute runtime args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | `num_rows_per_core` | `moreh_norm_program_factory_w_other.cpp:181-186`, `moreh_norm_w_kernel.cpp:9` |
| `1` | `Wt` | `moreh_norm_program_factory_w_other.cpp:182`, `moreh_norm_w_kernel.cpp:10` |
| `2` | logical `origin_w` | `moreh_norm_program_factory_w_other.cpp:183-184`, `moreh_norm_w_kernel.cpp:11` |

## Runtime-Update Pattern

`override_runtime_arguments(...)` only patches buffer addresses:

- reader arg `0` for input base address,
- writer arg `0` for output base address. `moreh_norm_program_factory_w_other.cpp:194-219`

All row partitioning metadata remains stable across cache hits, which implies the cached program shape is tied to the same row scheduling and tensor tilization.

For layernorm, this is a strong hint that row-partition descriptors should be part of compile-time/program-cache identity, while addresses remain runtime-patchable.

## Direct Layernorm Takeaways

| Pattern | Relevance |
| --- | --- |
| Partition by row units (`outer * Ht`) | Strong fit for last-dimension layernorm. |
| Consume all `Wt` tiles of a row together | Strong fit; mean/variance both need the entire final-axis span. |
| Carry `origin_w` into device kernels | Necessary to ignore padded columns. |
| Emit one reduced tile per row | Useful if layernorm computes row stats into an intermediate tensor. |
| Separate row scheduling from math mode | Lets layernorm reuse the scheduler while swapping the compute math. |

## Risks And Unknowns

- This reference only covers the reduction stage. Full layernorm also needs a second stage that rereads or retains the original row data to normalize with computed mean/variance.
- The output here is a reduced row tensor, not a same-shape tensor. Layernorm will need either:
  - a fused two-pass-in-one-program design, or
  - separate stats and normalization programs.
- `moreh_norm` uses `REDUCE_ROW` helpers and norm-specific transforms, so only the scheduling/tail-mask structure should be borrowed, not the reduction math itself.

## Assumptions

- Assumed layernorm target is reduction over the last logical dimension with tiled input, making this `WOther` path the closest local scheduling reference.
- Assumed initial layernorm will need explicit padded-width masking whenever the final logical dimension is not a multiple of 32.
