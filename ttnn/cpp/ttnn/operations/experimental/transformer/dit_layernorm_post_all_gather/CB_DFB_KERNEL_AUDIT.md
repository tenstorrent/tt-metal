# CB→DFB Kernel Audit: `dit_layernorm_post_all_gather`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/`

**Scope:** `device/kernels/compute/layernorm_post_allgather_welford.cpp`, `device/kernels/dataflow/reader_layernorm_postallgather_dit.cpp`, `device/kernels/dataflow/writer_layernorm_postallgather_dit.cpp`. Donor include: `ttnn/operations/normalization/kernel_util/compute/combine_welford.h` (scanned — clean).

## Overall verdict: GREEN

**Summary:** Post-allgather Welford LN. **Verified against the known finding:** unlike `dit_layernorm_pre_all_gather`, this compute kernel includes `combine_welford.h` — **not** `kernel_util/compute/memory.h` — and does **not** call `get_pointer_to_cb_data`/`get_tile_address`. `combine_welford.h` only pulls `api/` headers + `generic/bit.h` and is clean. All CBs are canonical Class 1 linear FIFOs driven through `CircularBuffer` objects (`reserve_back`/`push_back`/`wait_front`/`pop_front`). Step-4 litmus scans return **zero** hits across the kernel closure.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_inp` (c_0) | 1 | `layernorm_post_allgather_welford.cpp`, `reader_*` | Portable | input activations, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_stats` (c_1) | 1 | `layernorm_post_allgather_welford.cpp`, `reader_*` | Portable | per-device gathered stats, canonical FIFO | Portable | — |
| `cb_gamma` (c_2), `cb_beta` (c_3) | 1 | `layernorm_post_allgather_welford.cpp`, `reader_*` | Portable | weight FIFOs, cumulative `wait_front`/`pop_front` | Portable | — |
| `cb_eps` (c_4) | 1 | `layernorm_post_allgather_welford.cpp`, `reader_*` | Portable | broadcast epsilon, linear FIFO | Portable | — |
| `cb_stats_reduced` (c_5) | 1 | `layernorm_post_allgather_welford.cpp` (+`combine_welford.h`) | Portable | `combine_welford_partials` output [mean,var], canonical FIFO | Portable | — |
| `cb_recip_sqrt_var` (c_6) | 1 | `layernorm_post_allgather_welford.cpp` | Portable | 1/sqrt(var+eps) FIFO | Portable | — |
| `cb_intermediate` (c_7) | 1 | `layernorm_post_allgather_welford.cpp` | Portable | in-place-capable normalize stage via canonical reserve/push | Portable | — |
| `cb_out` (c_8) | 1 | `layernorm_post_allgather_welford.cpp`, `writer_*` | Portable | pack → output, `get_write_ptr()` as NoC addr | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
