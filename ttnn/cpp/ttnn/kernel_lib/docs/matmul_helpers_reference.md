# Matmul Compute Helper Reference (LLM)

Two helpers are provided: `matmul_block` for sub-blocked matmul with K-blocking, and
`add_bias_bcast_rows` for fused row-broadcast bias addition on matmul output.

For simple tile-at-a-time matmul (no sub-blocking), use inline `mm_init` + `matmul_tiles`
directly — see the programming example at
`tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp`.
There is no helper for this pattern because `matmul_tiles` has poor performance.

## Includes

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"  // only if using bias
```

Namespace: `compute_kernel_lib`.

## Dimension Notation

- `Mt = M / 32`, `Kt = K / 32`, `Nt = N / 32`
- M, K, N must all be multiples of 32
- A: shape `[batch, Mt, Kt]` in tiles
- B: shape `[batch, Kt, Nt]` in tiles
- C: shape `[batch, Mt, Nt]` in tiles

---

## compute_kernel_lib::matmul_block

Sub-blocked matmul with K-blocking. Uses `mm_block_init` + `matmul_block` LLK with
block-level CB waits and sub-block indexing. Supports two K-blocking strategies:

- `packer_l1_acc=false` (default): Software spill/reload via interm_cb
- `packer_l1_acc=true`: Hardware L1 accumulation (avoids spill/reload overhead)

```cpp
template <
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
    bool transpose = false,
    bool packer_l1_acc = false,
    bool pack_last_to_interm = false,
    bool pack_relu = false,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock>
void matmul_block(
    uint32_t block_w,            // K-dimension block size in tiles
    uint32_t in0_num_subblocks,  // sub-blocks along M dimension
    uint32_t in1_num_subblocks,  // sub-blocks along N dimension
    uint32_t num_k_blocks,       // blocks along K dimension
    uint32_t out_subblock_h,     // output sub-block height in tiles
    uint32_t out_subblock_w,     // output sub-block width in tiles
    uint32_t batch = 1,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {});
```

### PREREQUISITE

Caller must call `mm_block_init()` before invoking this helper. The helper does NOT
call `mm_block_init` internally.

### Template Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in0_cb` | — | Input CB for matrix A |
| `in1_cb` | — | Input CB for matrix B |
| `out_cb` | — | Output CB for result C |
| `interm_cb` | — | Intermediate CB for K-blocking spill/reload |
| `transpose` | false | Transpose B tiles before multiplication |
| `packer_l1_acc` | false | Use packer L1 accumulation instead of software spill/reload |
| `pack_last_to_interm` | false | Pack last K-block to interm_cb (for bias pipeline) |
| `pack_relu` | false | Enable PACK_RELU on last K-block (when !pack_last_to_interm) |
| `PostComputeFn` | NoPostCompute | Functor per sub-block on last K-block, before pack |
| `PreKBlockFn` | NoPreKBlock | Functor per K-block, before input CB waits |

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
in0_cb:    >= in0_block_num_tiles pages
in1_cb:    >= in1_block_num_tiles pages
out_cb:    >= total output tiles
interm_cb: >= out_num_tiles pages (only used when num_k_blocks > 1)
```

`out_cb` and `interm_cb` typically share L1 memory (overlapping address space).

### PostComputeFn

Functor called per output sub-block on the last K-block, after matmul, before pack.
Use for fused SFPU activations (relu, gelu, etc.).

```cpp
struct ApplyRelu {
    ALWI void operator()(uint32_t num_tiles) const {
        for (uint32_t i = 0; i < num_tiles; i++) relu_tile(i);
    }
};
compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm,
    false, false, false, false, ApplyRelu>(..., ApplyRelu{});
```

### PreKBlockFn

Functor called at the start of each K-block, before `cb_wait_front` for inputs.
Signature: `void operator()(uint32_t block, uint32_t num_k_blocks, bool is_last)`.
Use for per-K-block preprocessing such as in0_transpose or global CB pointer manipulation.

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

    mm_block_init(cb_in0, cb_in1, cb_out);

    compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm>(
        in0_block_w, in0_num_subblocks, in1_num_subblocks,
        num_k_blocks, out_subblock_h, out_subblock_w, batch);
}
```

---

## compute_kernel_lib::add_bias_bcast_rows

Row-broadcast bias addition on matmul output. Composes with `matmul_block` by reading
from the interm_cb that `matmul_block` packed to (when `pack_last_to_interm=true`).

```cpp
template <
    uint32_t partials_cb, uint32_t bias_cb, uint32_t out_cb,
    typename PostBiasFn = NoPostBias>
void add_bias_bcast_rows(
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t bias_width_tiles,
    PostBiasFn post_bias = {});
```

### PREREQUISITE

Caller must handle PACK_RELU configuration, pack format reconfig, and L1_ACC disable
before calling this helper.

### Template Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `partials_cb` | — | CB with matmul output (= interm_cb from matmul_block) |
| `bias_cb` | — | CB with bias tiles (row-broadcast, 1 tile per output column) |
| `out_cb` | — | Output CB for biased result |
| `PostBiasFn` | NoPostBias | Functor per sub-block after bias, before pack |

### CB Notes

- Waits for `bias_width_tiles` upfront, does NOT pop bias_cb (caller manages lifetime)
- Waits/pops partials_cb per sub-block
- Reserves/pushes out_cb per sub-block

### Matmul + Bias Composition Example

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"

void kernel_main() {
    // ... read compile-time args ...

    mm_block_init(cb_in0, cb_in1, cb_out);

    // Matmul: pack last K-block to interm for bias pipeline
    compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm,
        false, true, true>(  // packer_l1_acc=true, pack_last_to_interm=true
        block_w, in0_num_subblocks, in1_num_subblocks,
        num_k_blocks, out_subblock_h, out_subblock_w, batch);

    // Bias: read from interm, add bias with row broadcast, pack to out
    compute_kernel_lib::add_bias_bcast_rows<cb_interm, cb_bias, cb_out>(
        in0_num_subblocks, in1_num_subblocks,
        out_subblock_h, out_subblock_w, bias_width_tiles);
}
```

---

## Runtime Asserts

Both helpers validate at runtime:
- All dimension parameters > 0
- `out_num_tiles <= DEST_AUTO_LIMIT` (DST register capacity)
