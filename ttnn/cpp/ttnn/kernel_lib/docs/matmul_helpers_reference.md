# Matmul Compute Helper Reference (LLM)

One helper is provided: `matmul_block` for sub-blocked matmul with spill/reload.

For simple tile-at-a-time matmul (no sub-blocking), use inline `mm_init` + `matmul_tiles`
directly — see the programming example at
`tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp`.

## Include

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
```

Namespace: `compute_kernel_lib`.

## Dimension Notation

- `Mt = M / 32`, `Kt = K / 32`, `Nt = N / 32`
- M, K, N must all be multiples of 32
- A: shape `[batch, Mt, Kt]` in tiles
- B: shape `[batch, Kt, Nt]` in tiles
- C: shape `[batch, Mt, Nt]` in tiles

## compute_kernel_lib::matmul_block

Sub-blocked matmul with spill/reload for larger matrices. Uses `mm_block_init` +
`matmul_block` LLK with block-level CB waits and sub-block indexing. When the K
dimension is split across multiple blocks (`num_k_blocks > 1`), partial results
spill to an intermediate CB and are reloaded for accumulation on the next block.

```cpp
template <
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
    bool transpose = false,
    typename PostComputeFn = NoPostCompute>
void matmul_block(
    uint32_t block_w,            // K-dimension block size in tiles
    uint32_t in0_num_subblocks,  // sub-blocks along M dimension
    uint32_t in1_num_subblocks,  // sub-blocks along N dimension
    uint32_t num_k_blocks,       // blocks along K dimension
    uint32_t out_subblock_h,     // output sub-block height in tiles
    uint32_t out_subblock_w,     // output sub-block width in tiles
    uint32_t batch = 1,
    PostComputeFn post_compute = {});
```

### Derived quantities (computed internally)

```
out_num_tiles          = out_subblock_h * out_subblock_w
in0_subblock_num_tiles = out_subblock_h * block_w
in0_block_num_tiles    = in0_subblock_num_tiles * in0_num_subblocks
in1_per_core_w         = out_subblock_w * in1_num_subblocks
in1_block_num_tiles    = out_subblock_w * block_w * in1_num_subblocks
```

### CB Setup

```
in0_cb:    >= in0_block_num_tiles pages (full A block)
in1_cb:    >= in1_block_num_tiles pages (full B block)
out_cb:    >= total output tiles (for reservation tracking)
interm_cb: >= out_num_tiles pages (partial result spill, only when num_k_blocks > 1)
```

`out_cb` and `interm_cb` should share memory (overlapping address space). The output CB
only needs space once the final K-block is ready; until then, `interm_cb` uses the space
for partial results.

### PostComputeFn

Optional functor called on each output sub-block after the last K-block's matmul,
before tiles are packed. Use for fused SFPU activations (relu, gelu, etc.).

```cpp
struct ApplyRelu {
    ALWI void operator()(uint32_t num_tiles) const {
        for (uint32_t i = 0; i < num_tiles; i++) {
            relu_tile(i);
        }
    }
};
compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm,
    false, ApplyRelu>(..., ApplyRelu{});
```

### Compute Kernel Example

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    uint32_t num_k_blocks = get_compile_time_arg_val(7);
    uint32_t out_subblock_h = get_compile_time_arg_val(8);
    uint32_t out_subblock_w = get_compile_time_arg_val(9);
    uint32_t batch = get_compile_time_arg_val(11);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_interm = tt::CBIndex::c_24;

    compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm>(
        in0_block_w, in0_num_subblocks, in1_num_subblocks,
        num_k_blocks, out_subblock_h, out_subblock_w, batch);
}
```

### Runtime Asserts

The helper validates at runtime:
- All dimension parameters > 0
- `out_num_tiles <= DEST_AUTO_LIMIT` (DST register capacity — 16 tiles for FP16 full-sync,
  8 for FP32 full-sync or FP16 half-sync, 4 for FP32 half-sync)
- CB capacity >= required block sizes (in0, in1, out)
