# Capabilities: linear

> Last updated: 2026-05-06 by incremental-verifier (Phase 0)

| Dimension | Status | Details |
|-----------|--------|---------|
| **Data formats** | bfloat16 only | Entry point validates `dtype == ttnn.bfloat16` for input, weight, and bias (`linear.py:111-112`). float32, bfloat8_b, bfp4_b: rejected with `ValueError`. |
| **Layouts** | TILE only | Entry point validates `layout == ttnn.TILE_LAYOUT` (`linear.py:114-115`). No in-kernel tilize/untilize; ROW_MAJOR requires host-side `.to_layout(TILE_LAYOUT)` first (and is rejected outright by validation). |
| **Memory configs (input/weight/bias)** | DRAM-interleaved (assumed) | Validation does not gate the input memory config, but the program descriptor and reader assume DRAM-interleaved tile pages (`linear_program_descriptor.py` builds CBs sized to `Mt*Kt`/`Kt*Nt`/`Nt` tiles; `linear_reader.cpp` uses `TensorAccessor`). Sharded inputs are not on a tested path. |
| **Memory configs (output)** | DRAM-interleaved by default | `linear()` accepts `memory_config` kwarg; default is `ttnn.DRAM_MEMORY_CONFIG` (`linear.py:42`). Output tensor is allocated via `ttnn.allocate_tensor_on_device(...)` with whatever memory config the caller passes; no L1/sharded paths exercised. |
| **Core count** | Single core | Hard-coded `CoreRangeSet([CoreRange((0,0),(0,0))])` (`linear_program_descriptor.py:74-75`). No `split_work_to_cores`. |
| **Compute config** | Not exposed | `linear()` does not accept `compute_kernel_config`. Hard-coded `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True)` (`linear_program_descriptor.py:249-252`). `packer_l1_acc=false` and default `dst_full_sync_en` (= half-sync). |
| **Shape support** | Tile-aligned only | M, K, N must each be divisible by 32 (`linear.py:96-97`). No padding/masking path. Bias height fixed at exactly 32 (`linear.py:104-106`). |
| **Rank support** | Rank 4 only, leading dims `[1, 1, ...]` | Validation rejects rank != 4 and leading dims != `[1, 1]` (`linear.py:117-122`). PyTorch-style `[B, M, K]` or `[M, K]` inputs are rejected. |
| **Bias** | Optional, row-broadcast only | `bias` kwarg accepts a `[1, 1, 32, N]` tile-padded tensor with the bias values in row 0 (rows 1-31 are read-but-ignored by `add_tiles_bcast_rows`). N must match weight N. No elementwise / multi-row bias path. |
| **Activation fusion** | Not exposed | The `matmul_block` and `add_bias_bcast_rows` helpers support fused SFPU activation via `PostComputeFn` / `PostBiasFn` template params, but `linear()` does not expose this (no `activation` kwarg). |
| **Features vs PyTorch** | Partial | PyTorch `F.linear(x, w, b)` accepts arbitrary leading batch dims (`x: [..., in_features]`, `w: [out_features, in_features]`, transposes `w`, broadcasts batch). TTNN linear: 4D rank with leading `[1, 1]`, weight is **not** transposed (caller passes `[K, N]` shape directly), and bias must be tile-padded to height 32. PyTorch's batch-broadcasting matmul is not yet supported. |
