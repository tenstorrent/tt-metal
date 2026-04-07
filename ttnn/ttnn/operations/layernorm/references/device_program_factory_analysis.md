# Device Program Factory Analysis: `batch_norm_program_factory.cpp`

## Overview

Primary reference: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

Kernel sources referenced by the factory:

- `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp`
- `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`
- `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp` `batch_norm_program_factory.cpp:238-295`

This factory implements a fused three-kernel program:

- reader loads input tiles plus a scalar epsilon tile,
- writer loads broadcast-style per-channel statistics/affine tiles and drains the output,
- compute performs normalization and optional affine application. `reader_batch_norm.cpp:15-75`, `writer_batch_norm.cpp:13-130`, `batch_norm_kernel.cpp:137-205`, `batch_norm_sfpu_kernel.cpp:170-238`

For layernorm, this is the most relevant local example of a fused normalization program factory with:

- per-core tile-range assignment,
- broadcast reuse of statistics tiles,
- optional affine operands,
- and cache-safe runtime-argument overriding.

## Supported Variants

| Variant axis | How the factory handles it | Evidence |
| --- | --- | --- |
| `weight` present/absent | Compile-time flags remove weight reads and multiply path when absent | `batch_norm_program_factory.cpp:141-151`, `batch_norm_program_factory.cpp:213-227`, `batch_norm_program_factory.cpp:273-295`, `writer_batch_norm.cpp:30-36`, `writer_batch_norm.cpp:97-104`, `batch_norm_kernel.cpp:64-66`, `batch_norm_kernel.cpp:91-107` |
| `bias` present/absent | Compile-time flags remove bias reads and add path when absent | `batch_norm_program_factory.cpp:141-151`, `batch_norm_program_factory.cpp:213-227`, `batch_norm_program_factory.cpp:273-295`, `writer_batch_norm.cpp:30-36`, `writer_batch_norm.cpp:106-112`, `batch_norm_kernel.cpp:67-69`, `batch_norm_kernel.cpp:109-125` |
| Input dtype `FLOAT32` vs BF16 | Dataflow fill helpers and compute kernel choice adapt to dtype / fp32 dest accumulation | `batch_norm_program_factory.cpp:229-236`, `batch_norm_program_factory.cpp:255-295` |
| `fp32_dest_acc_en` on/off | Chooses `batch_norm_sfpu_kernel.cpp` vs `batch_norm_kernel.cpp` and changes unpack-to-dest mode for relevant CBs | `batch_norm_program_factory.cpp:255-271`, `batch_norm_program_factory.cpp:285-295` |

## Work-Unit Definition

### Logical work unit

The factory parallelizes across output tiles:

- `num_output_tiles = c.physical_volume() / tile_hw`
- `split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major=true)` partitions the output-tile range across the full compute grid. `batch_norm_program_factory.cpp:47-56`

Each core receives:

- `start_tile_id`
- `num_tiles_per_core`

and then processes a contiguous tile interval within the flattened output tensor. `batch_norm_program_factory.cpp:60-121`

### Broadcast granularity

Although work is assigned in flat output-tile ranges, the compute kernel groups tiles by `freq = cHt * cWt`, which is the number of spatial tiles per `(N, C)` slice. `batch_norm_program_factory.cpp:75`, `batch_norm_program_factory.cpp:114-118`

`tile_start = start_tile_id % freq` is passed into compute as the starting position within the current channel slice. `batch_norm_program_factory.cpp:114-118`

That means one compute “broadcast epoch” corresponds to a single `(n, c)` group:

- one `batch_mean` tile,
- one `batch_var` tile,
- optionally one `weight` tile,
- optionally one `bias` tile,
- reused across up to `freq` output tiles for that `(n, c)` channel band. `batch_norm_kernel.cpp:44-69`, `batch_norm_kernel.cpp:70-135`, `writer_batch_norm.cpp:80-125`

For layernorm, this broadcast-epoch structure is the key transferable idea, but the reuse axis will likely move from “channel over all spatial tiles” to “row stats over last-dimension tiles.”

## Dataflow

| Stage | Inputs consumed | Outputs produced | Notes |
| --- | --- | --- | --- |
| Reader | global input tensor, scalar `eps` runtime arg | `c_0` input tiles, `c_4` epsilon tile | Reader is responsible only for the main input stream and a one-tile scalar buffer. `reader_batch_norm.cpp:16-27`, `reader_batch_norm.cpp:45-74` |
| Writer | global `batch_mean`, `batch_var`, optional `weight`, optional `bias`, compute output CB | `c_1`, `c_3`, `c_5`, `c_6` populated from DRAM; output drained from `c_2` to global output | Writer acts as both a feeder of broadcast operands and the final output DMA stage. `writer_batch_norm.cpp:14-41`, `writer_batch_norm.cpp:79-129` |
| Compute | `c_0`, `c_1`, `c_3`, `c_4`, optional `c_5`, optional `c_6` | `c_7` denominator scratch, `c_8` optional temp, `c_2` final output | Compute derives `1/sqrt(var+eps)`, normalizes input, then applies optional affine transforms. `batch_norm_kernel.cpp:44-135`, `batch_norm_sfpu_kernel.cpp:47-168` |

### End-to-end flow

1. Reader pushes one epsilon tile and a stream of input tiles for the assigned range. `reader_batch_norm.cpp:45-54`, `reader_batch_norm.cpp:62-74`
2. Writer pushes one broadcast tile each for mean/variance and optional affine tensors per channel group. `writer_batch_norm.cpp:80-112`
3. Compute:
   - forms `den = rsqrt(batch_var + eps)`,
   - subtracts batch mean,
   - multiplies by `den`,
   - optionally multiplies by weight,
   - optionally adds bias,
   - packs the final tile into `c_2`. `batch_norm_kernel.cpp:44-125`, `batch_norm_sfpu_kernel.cpp:47-158`
4. Writer waits on `c_2` and writes output tiles back in the same flat order. `writer_batch_norm.cpp:114-124`

Synchronization is entirely through circular-buffer wait/reserve/push/pop operations; there are no explicit semaphores in this program.

## Circular Buffer Table

| CB | Factory meaning | Producer | Consumer | Evidence |
| --- | --- | --- | --- | --- |
| `c_0` | input tensor | reader | compute | `batch_norm_program_factory.cpp:171-173`, `reader_batch_norm.cpp:26-37`, `batch_norm_kernel.cpp:148-160` |
| `c_1` | batch mean | writer | compute | `batch_norm_program_factory.cpp:173-179`, `writer_batch_norm.cpp:29-45`, `batch_norm_kernel.cpp:149-160` |
| `c_2` | output tensor | compute | writer | `batch_norm_program_factory.cpp:180-181`, `writer_batch_norm.cpp:46-49`, `writer_batch_norm.cpp:114-124`, `batch_norm_kernel.cpp:150-157` |
| `c_3` | batch var | writer | compute | `batch_norm_program_factory.cpp:182-188`, `writer_batch_norm.cpp:50-53`, `batch_norm_kernel.cpp:152-154` |
| `c_4` | epsilon scalar tile | reader | compute | `batch_norm_program_factory.cpp:189-190`, `reader_batch_norm.cpp:26-27`, `reader_batch_norm.cpp:45-54`, `batch_norm_kernel.cpp:153-154` |
| `c_5` | weight | writer | compute | `batch_norm_program_factory.cpp:191-192`, `writer_batch_norm.cpp:54-57`, `batch_norm_kernel.cpp:155` |
| `c_6` | bias | writer | compute | `batch_norm_program_factory.cpp:193-194`, `writer_batch_norm.cpp:58-60`, `batch_norm_kernel.cpp:157` |
| `c_7` | denominator scratch `1/sqrt(var+eps)` | compute | compute | `batch_norm_program_factory.cpp:197-203`, `batch_norm_kernel.cpp:44-60` |
| `c_8` | temp affine/output staging | compute | compute | `batch_norm_program_factory.cpp:204-205`, `batch_norm_kernel.cpp:30-31`, `batch_norm_kernel.cpp:73-88`, `batch_norm_kernel.cpp:93-125` |

Buffer sizing is intentionally small:

- most inputs are double-buffered with `num_tiles_per_cb = 2`
- writer-side broadcast buffers also use two tiles, even though the compute kernel effectively consumes one broadcast tile at a time. `batch_norm_program_factory.cpp:166-205`

## Core Distribution Strategy

### Grid and partitioning

- The program uses the full `device->compute_with_storage_grid_size()` as the candidate core set. `batch_norm_program_factory.cpp:161-165`
- `split_work_to_cores(...)` partitions the flat output tile count across that grid in row-major order. `batch_norm_program_factory.cpp:49-56`
- `grid_to_cores(...)` converts the grid into an ordered core list matching that row-major split. `batch_norm_program_factory.cpp:56`

### Idle-core handling

If a core receives no tiles, the factory still installs zeroed runtime arguments for reader, writer, and compute kernels. `batch_norm_program_factory.cpp:63-72`

This matters because kernels are created over `all_device_cores`; zero-arg initialization prevents stale arguments on non-participating cores.

### Per-core state

For active cores:

- reader gets the flat range and full input-shape strides,
- writer gets the same flat range plus reduced-tensor strides,
- compute gets only range length plus per-channel repetition info. `batch_norm_program_factory.cpp:81-118`

## Kernel Responsibilities

### Reader kernel

Responsibilities:

- materialize a one-tile epsilon CB from a packed scalar,
- stream input tiles for the assigned flat interval,
- translate flat tile ranges into `(n, c, t)` traversal using `n_stride`, `c_stride`, and `HtWt`. `reader_batch_norm.cpp:16-27`, `reader_batch_norm.cpp:39-74`

Notable detail:

- epsilon is converted into a tile exactly once per core, not once per channel group. `reader_batch_norm.cpp:45-54`

### Writer kernel

Responsibilities:

- for each `(n, c)` group touched by the core, read one `batch_mean` tile and one `batch_var` tile, fill each tile with its first element, and push to CBs,
- optionally do the same for `weight` and `bias`,
- drain output tiles from `c_2` back to the destination tensor. `writer_batch_norm.cpp:79-129`

Important asymmetry:

- statistics and affine operands are fed by the writer kernel rather than the reader kernel.
- this keeps the reader focused on the dense input stream while the writer handles the sparse/broadcast operands and final writeback.

### Compute kernel

Both compute variants implement the same math:

1. `den = rsqrt(batch_var + eps)`
2. `normalized = (input - batch_mean) * den`
3. if present, `normalized *= weight`
4. if present, `normalized += bias` `batch_norm_kernel.cpp:44-125`, `batch_norm_sfpu_kernel.cpp:47-168`

The non-SFPU kernel uses `eltwise_binary` primitives with dest reuse. `batch_norm_kernel.cpp:75-85`

The SFPU kernel performs similar steps with explicit copy/add/sub/mul unary+binary SFPU building blocks. `batch_norm_sfpu_kernel.cpp:51-69`, `batch_norm_sfpu_kernel.cpp:83-157`

## Compile-Time Argument Mapping

### Reader compile-time args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | input CB id (`c_0`) | `batch_norm_program_factory.cpp:207-212`, `reader_batch_norm.cpp:26` |
| `1` | eps CB id (`c_4`) | `batch_norm_program_factory.cpp:207-212`, `reader_batch_norm.cpp:27` |
| appended | input tensor accessor args | `batch_norm_program_factory.cpp:211-212`, `reader_batch_norm.cpp:28-33` |

### Writer compile-time args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | `weight_has_value` flag | `batch_norm_program_factory.cpp:213-227`, `writer_batch_norm.cpp:30` |
| `1` | `bias_has_value` flag | `batch_norm_program_factory.cpp:213-227`, `writer_batch_norm.cpp:31` |
| `2` | batch-mean CB id (`c_1`) | `batch_norm_program_factory.cpp:216`, `writer_batch_norm.cpp:32` |
| `3` | output CB id (`c_2`) | `batch_norm_program_factory.cpp:217`, `writer_batch_norm.cpp:33` |
| `4` | batch-var CB id (`c_3`) | `batch_norm_program_factory.cpp:218`, `writer_batch_norm.cpp:34` |
| `5` | weight CB id (`c_5`) | `batch_norm_program_factory.cpp:219`, `writer_batch_norm.cpp:35` |
| `6` | bias CB id (`c_6`) | `batch_norm_program_factory.cpp:220`, `writer_batch_norm.cpp:36` |
| appended | accessor args for mean/output/var/weight/bias tensors | `batch_norm_program_factory.cpp:222-227`, `writer_batch_norm.cpp:37-60` |

### Compute compile-time args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | `weight_has_value` flag | `batch_norm_program_factory.cpp:273-275`, `batch_norm_kernel.cpp:141` |
| `1` | `bias_has_value` flag | `batch_norm_program_factory.cpp:274-275`, `batch_norm_kernel.cpp:142` |
| `2` | input CB (`c_0`) | `batch_norm_program_factory.cpp:276`, `batch_norm_kernel.cpp:148` |
| `3` | batch-mean CB (`c_1`) | `batch_norm_program_factory.cpp:277`, `batch_norm_kernel.cpp:149` |
| `4` | output CB (`c_2`) | `batch_norm_program_factory.cpp:278`, `batch_norm_kernel.cpp:150-151` |
| `5` | batch-var CB (`c_3`) | `batch_norm_program_factory.cpp:279`, `batch_norm_kernel.cpp:152` |
| `6` | eps CB (`c_4`) | `batch_norm_program_factory.cpp:280`, `batch_norm_kernel.cpp:153` |
| `7` | denominator scratch CB (`c_7`) | `batch_norm_program_factory.cpp:281`, `batch_norm_kernel.cpp:154` |
| `8` | weight CB (`c_5`) | `batch_norm_program_factory.cpp:282`, `batch_norm_kernel.cpp:155` |
| `9` | temp CB (`c_8`) | `batch_norm_program_factory.cpp:283`, `batch_norm_kernel.cpp:156` |
| `10` | bias CB (`c_6`) | `batch_norm_program_factory.cpp:284`, `batch_norm_kernel.cpp:157` |

## Runtime Argument Mapping

`set_or_update_runtime_arguments(...)` is the single source of truth for both initial setup and cache-hit updates. `batch_norm_program_factory.cpp:23-122`, `batch_norm_program_factory.cpp:296-309`, `batch_norm_program_factory.cpp:315-337`

### Reader runtime args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | packed epsilon scalar | `batch_norm_program_factory.cpp:76-83`, `reader_batch_norm.cpp:16` |
| `1` | input base address | `batch_norm_program_factory.cpp:81-84`, `reader_batch_norm.cpp:17` |
| `2` | `start_tile_id` | `batch_norm_program_factory.cpp:83-85`, `reader_batch_norm.cpp:18` |
| `3` | `num_tiles_per_core` | `batch_norm_program_factory.cpp:84-86`, `reader_batch_norm.cpp:19` |
| `4` | `cHtWt` | `batch_norm_program_factory.cpp:75`, `batch_norm_program_factory.cpp:86`, `reader_batch_norm.cpp:20` |
| `5` | `n_stride = aHt*aWt*aC*(aN>1)` | `batch_norm_program_factory.cpp:87`, `reader_batch_norm.cpp:21` |
| `6` | `c_stride = aHt*aWt*(aC>1)` | `batch_norm_program_factory.cpp:88`, `reader_batch_norm.cpp:22` |
| `7` | `N` | `batch_norm_program_factory.cpp:89`, `reader_batch_norm.cpp:23` |
| `8` | `C` | `batch_norm_program_factory.cpp:90`, `reader_batch_norm.cpp:24` |
| `9` | `Ht` | `batch_norm_program_factory.cpp:91` |
| `10` | `Wt` | `batch_norm_program_factory.cpp:92` |

The reader kernel currently consumes indices `0..8`; `Ht` and `Wt` are passed but unused in this specific kernel body. `reader_batch_norm.cpp:16-24`

### Writer runtime args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | batch-mean base address | `batch_norm_program_factory.cpp:97-99`, `writer_batch_norm.cpp:14` |
| `1` | batch-var base address | `batch_norm_program_factory.cpp:98-99`, `writer_batch_norm.cpp:15` |
| `2` | weight base address or `0` | `batch_norm_program_factory.cpp:95-101`, `writer_batch_norm.cpp:16` |
| `3` | bias base address or `0` | `batch_norm_program_factory.cpp:95-101`, `writer_batch_norm.cpp:17` |
| `4` | output base address | `batch_norm_program_factory.cpp:102`, `writer_batch_norm.cpp:18` |
| `5` | `start_tile_id` | `batch_norm_program_factory.cpp:103`, `writer_batch_norm.cpp:19` |
| `6` | `num_tiles_per_core` | `batch_norm_program_factory.cpp:104`, `writer_batch_norm.cpp:20` |
| `7` | `cHtWt` | `batch_norm_program_factory.cpp:105`, `writer_batch_norm.cpp:21` |
| `8` | reduced-tensor `n_stride = bHt*bWt*bC*(bN>1)` | `batch_norm_program_factory.cpp:106`, `writer_batch_norm.cpp:22` |
| `9` | reduced-tensor `c_stride = bHt*bWt*(bC>1)` | `batch_norm_program_factory.cpp:107`, `writer_batch_norm.cpp:23` |
| `10` | output `N` | `batch_norm_program_factory.cpp:108`, `writer_batch_norm.cpp:24` |
| `11` | output `C` | `batch_norm_program_factory.cpp:109`, `writer_batch_norm.cpp:25` |
| `12` | output `Ht` | `batch_norm_program_factory.cpp:110` |
| `13` | output `Wt` | `batch_norm_program_factory.cpp:111` |

### Compute runtime args

| Index | Meaning | Evidence |
| --- | --- | --- |
| `0` | `num_tiles_per_core` | `batch_norm_program_factory.cpp:117`, `batch_norm_kernel.cpp:138` |
| `1` | `freq = cHtWt` | `batch_norm_program_factory.cpp:115-118`, `batch_norm_kernel.cpp:139` |
| `2` | `counter = start_tile_id % cHtWt` | `batch_norm_program_factory.cpp:114-118`, `batch_norm_kernel.cpp:140` |

Interpretation:

- `freq` is the number of output tiles sharing the same broadcast stats tile.
- `counter` is the offset within the first such group for this core.

## Runtime-Update Strategy

The factory intentionally reuses the same runtime-argument generator in both `create(...)` and `override_runtime_arguments(...)`. `batch_norm_program_factory.cpp:23-122`, `batch_norm_program_factory.cpp:296-337`

That gives a clean derivative pattern for layernorm:

- build the program and compile-time args once,
- only patch addresses, tile ranges, and shape-derived strides on cache hits.

## Layernorm-Relevant Takeaways

| Reusable idea | Why it matters for layernorm |
| --- | --- |
| One fused reader/compute/writer program | Matches the requested real on-device layernorm implementation. |
| Runtime work split over flat output tiles | Can be reused if layernorm writes same-shape tiled outputs. |
| Broadcast tile reuse model in compute | Very relevant if layernorm computes or consumes one stats tile per logical row/group and reuses it across tiles in the normalized dimension. |
| Optional affine branches compiled out via flags | Direct fit for optional `weight` / `bias`. |
| Shared helper for create/update runtime args | Important for program-cache correctness. |

## Risks And Unknowns

- This factory assumes statistics tensors are already available in DRAM. A fused layernorm that computes mean/variance internally will need a different pipeline shape and likely additional temporary CBs.
- Work partitioning is over flat output tiles, not explicitly over “rows.” That is sufficient for batch norm because broadcast reuse is tied to channel groups; layernorm’s last-dimension reduction may want row-aware partitioning to avoid cross-row bookkeeping.
- The writer kernel assumes the same address-generation pattern is valid for `batch_mean`, `batch_var`, `weight`, and `bias`; this fits batch norm’s `(1, C, 1, 1)` broadcast structure but may not fit layernorm affine tensors if their physical layout differs.
- Reader runtime args include `Ht` and `Wt` that are unused in the current kernel, which suggests the argument contract already has some historical slack. Layernorm should keep its runtime ABI minimal unless those dimensions are genuinely needed.

## Assumptions

- Assumed first layernorm implementation will remain tiled, single-device, and same-shape-output, so the factory skeleton and cache-update pattern are directly reusable.
- Assumed optional affine tensors should compile out cleanly when absent, following this reference.
