# Matmul Compute Helper Reference (LLM)

Two helpers are provided: `matmul_tile` for simple tile-at-a-time matmul, and
`matmul_block` for sub-blocked matmul with spill/reload (used for performance).

## Includes

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp"   // simple tile-at-a-time
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"  // sub-blocked with spill/reload
```

Namespace: `compute_kernel_lib`.
Compute kernels MUST call `compute_kernel_hw_startup(in0_cb, in1_cb, out_cb)` before any
helper call. Use the **three-argument form** — srcA and srcB are different CBs.

## Dimension Notation

- `Mt = M / 32`, `Kt = K / 32`, `Nt = N / 32`
- M, K, N must all be multiples of 32
- A: shape `[batch, Mt, Kt]` in tiles — tile at `(b, mt, kt)` → linear index `b*Mt*Kt + mt*Kt + kt`
- B: shape `[batch, Kt, Nt]` in tiles — tile at `(b, kt, nt)` → linear index `b*Kt*Nt + kt*Nt + nt`
- C: shape `[batch, Mt, Nt]` in tiles — tile at `(b, mt, nt)` → linear index `b*Mt*Nt + mt*Nt + nt`

## CB Setup Requirements

```
in0_cb: tile-sized pages, >= 1 page (WaitPerTile) or >= Kt pages (WaitUpfront)
in1_cb: tile-sized pages, >= 1 page (WaitPerTile) or >= Kt*Nt pages (WaitUpfront)
out_cb: tile-sized pages, >= 1 page
```

All CBs use tiled data format (not row-major). in0_cb and in1_cb must differ from out_cb
(enforced by static_assert).

## Numerical Precision: fp32_dest_acc_en

For Kt > 4 (K > 128 elements), bf16 accumulation in the DEST register degrades precision
below typical test tolerances (rtol=0.05, atol=0.2). Enable fp32 DEST accumulation in the
program descriptor's `ComputeConfigDescriptor`:

```python
ComputeConfigDescriptor(math_fidelity="HiFi4", dst_full_sync_en=True, fp32_dest_acc_en=True)
```

With `fp32_dest_acc_en=True`, each partial product is accumulated in fp32 before the final
result is packed back to bf16, significantly reducing accumulated rounding error for large K.

## compute_kernel_lib::matmul_tile

Performs `C = A × B` tile-by-tile using `mm_init` + `matmul_tiles`. Loop order:
batch × Mt × Nt × Kt. One output tile is accumulated per (b, mt, nt) over all Kt steps.

```cpp
template <
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    InitUninitMode init_uninit_mode = InitAndUninit,
    WaitMode wait_mode = WaitPerTile,
    ReconfigureRegisterDatatypeMode reconfig_mode = UnpackAndPackReconfigure>
void matmul_tile(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch = 1);
```

## Compute Kernel Example

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_tile_helpers.hpp"
void kernel_main() {
    uint32_t Mt    = get_compile_time_arg_val(0);
    uint32_t Kt    = get_compile_time_arg_val(1);
    uint32_t Nt    = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    compute_kernel_lib::matmul_tile<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt);
}
```

## WaitMode Trade-offs

- `WaitPerTile` (default): CB depth 1 sufficient for in0 and in1. Reader and compute
  naturally pipeline tile-by-tile. Use for all standard cases.
- `WaitUpfront`: CB must hold the full Mt-row block (in0 >= Kt pages, in1 >= Kt*Nt pages).
  Reader must pre-load the full block before compute begins. Requires a hand-written reader.
- `NoWait`: caller guarantees all tiles are already in CBs. Skips all CB synchronization
  inside the helper.

## InitUninitMode Use Cases

- `InitAndUninit`: standalone matmul kernel. Most common case.
- `InitOnly`: matmul followed by an eltwise op in the same kernel. Init matmul first, then
  init the eltwise op (which will call its own init and uninit).
- `UninitOnly` / `Neither`: both are no-ops for matmul since there is no `mm_uninit` in
  the LLK API. Included for API symmetry. Use `Neither` for middle calls in a chain.

## ReconfigureRegisterDatatypeMode Use Cases

- `UnpackAndPackReconfigure` (default): always safe when the kernel switches between op types.
  Reconfigures both unpack (srcA, srcB) and pack (output) register formats before `mm_init`.
- `NoReconfigure`: use when the kernel only ever calls `matmul_tile` and no other op. Avoids
  redundant reconfiguration overhead.
- `UnpackReconfigure` / `PackReconfigure`: partial reconfiguration for mixed-precision cases.

## Static Asserts (matmul_tile)

The implementation enforces at compile time:
- `in0_cb != out_cb`
- `in1_cb != out_cb`
- `in0_cb < 32`, `in1_cb < 32`, `out_cb < 32`

---

## compute_kernel_lib::matmul_block

Sub-blocked matmul with spill/reload for larger matrices. Uses `mm_init` + `matmul_tiles`
with block-level CB waits and non-zero tile indices. When the K dimension is split across
multiple blocks (`num_blocks > 1`), partial results spill to an intermediate CB and are
reloaded for accumulation on the next block.

```cpp
template <
    uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t interm_cb,
    InitUninitMode init_uninit_mode = InitAndUninit,
    ReconfigureRegisterDatatypeMode reconfig_mode = UnpackAndPackReconfigure>
void matmul_block(
    uint32_t in0_block_w,            // inner block dimension in tiles
    uint32_t in0_num_subblocks,      // sub-blocks along M dimension
    uint32_t in0_block_num_tiles,    // total tiles per A block
    uint32_t in0_subblock_num_tiles, // tiles per A sub-block
    uint32_t in1_num_subblocks,      // sub-blocks along N dimension
    uint32_t in1_block_num_tiles,    // total tiles per B block
    uint32_t in1_per_core_w,         // tiles per B row
    uint32_t num_blocks,             // blocks along K dimension
    uint32_t out_subblock_h,         // output sub-block height in tiles
    uint32_t out_subblock_w,         // output sub-block width in tiles
    uint32_t out_subblock_num_tiles, // tiles per output sub-block
    uint32_t batch = 1);
```

### CB Setup (matmul_block)

```
in0_cb:    >= in0_block_num_tiles pages (full A block)
in1_cb:    >= in1_block_num_tiles pages (full B block)
out_cb:    >= total output tiles (for reservation tracking)
interm_cb: >= out_subblock_num_tiles pages (partial result spill)
```

`out_cb` and `interm_cb` should share memory (overlapping address space). The output CB
only needs space once the final K-block is ready; until then, `interm_cb` uses the space
for partial results.

### When to use matmul_block vs matmul_tile

- **matmul_tile**: simple, tile-at-a-time. Good for small matrices, prototyping, or when
  the full M×N output fits in DST. CB depth 1 is sufficient.
- **matmul_block**: sub-blocked with spill/reload. Required when the output is larger than
  DST capacity. Better performance through block-level CB operations. Used in production
  TTNN matmul kernels.

### Compute Kernel Example (matmul_block)

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);
    uint32_t num_blocks = get_compile_time_arg_val(7);
    uint32_t out_subblock_h = get_compile_time_arg_val(8);
    uint32_t out_subblock_w = get_compile_time_arg_val(9);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);
    uint32_t batch = get_compile_time_arg_val(11);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_interm = tt::CBIndex::c_24;

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    compute_kernel_lib::matmul_block<cb_in0, cb_in1, cb_out, cb_interm>(
        in0_block_w, in0_num_subblocks, in0_block_num_tiles, in0_subblock_num_tiles,
        in1_num_subblocks, in1_block_num_tiles, in1_per_core_w,
        num_blocks, out_subblock_h, out_subblock_w, out_subblock_num_tiles, batch);
}
```

### Sub-block dimension relationships

```
in0_block_num_tiles    = out_subblock_h * in0_block_w * in0_num_subblocks
in0_subblock_num_tiles = out_subblock_h * in0_block_w
in1_block_num_tiles    = out_subblock_w * in0_block_w * in1_num_subblocks
in1_per_core_w         = out_subblock_w * in1_num_subblocks
out_subblock_num_tiles = out_subblock_h * out_subblock_w
```
