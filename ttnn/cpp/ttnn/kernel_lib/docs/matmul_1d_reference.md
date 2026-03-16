# matmul_1d Helper Reference (LLM)

## Helpers

```cpp
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"          // compute kernels
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp" // reader/writer kernels
```

Compute namespace: `compute_kernel_lib`. Dataflow namespace: `dataflow_kernel_lib`.
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

## compute_kernel_lib::matmul_1d

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
void matmul_1d(uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch = 1);
```

## dataflow_kernel_lib::read_matmul_tiles

Reads A and B tiles from DRAM into CBs in the order consumed by `matmul_1d` (WaitPerTile).
For each `(b, mt, nt, kt)`: pushes A[b,mt,kt] then B[b,kt,nt].

```cpp
template <uint32_t in0_cb, uint32_t in1_cb>
void read_matmul_tiles(
    uint32_t in0_tensor_addr,
    uint32_t in1_tensor_addr,
    uint32_t Mt, uint32_t Nt, uint32_t Kt,
    uint32_t batch = 1,
    bool bcast_B = false);
```

`bcast_B = true`: B is not batched — the same B is used for all batch slices.

## dataflow_kernel_lib::write_matmul_tiles

Writes C output tiles from CB to DRAM in row-major tile order.

```cpp
template <uint32_t out_cb>
void write_matmul_tiles(
    uint32_t out_tensor_addr,
    uint32_t Mt, uint32_t Nt,
    uint32_t batch = 1);
```

## Full Single-Core Matmul Kernel Skeleton

```cpp
// reader kernel
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"
void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt       = get_arg_val<uint32_t>(2);
    uint32_t Kt       = get_arg_val<uint32_t>(3);
    uint32_t Nt       = get_arg_val<uint32_t>(4);
    uint32_t batch    = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    dataflow_kernel_lib::read_matmul_tiles<cb_in0, cb_in1>(in0_addr, in1_addr, Mt, Nt, Kt, batch);
}

// writer kernel
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt       = get_arg_val<uint32_t>(1);
    uint32_t Nt       = get_arg_val<uint32_t>(2);
    uint32_t batch    = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    dataflow_kernel_lib::write_matmul_tiles<cb_out>(out_addr, Mt, Nt, batch);
}

// compute kernel
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_helpers.hpp"
void kernel_main() {
    uint32_t Mt    = get_arg_val<uint32_t>(0);
    uint32_t Kt    = get_arg_val<uint32_t>(1);
    uint32_t Nt    = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
    compute_kernel_lib::matmul_1d<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt, batch);
}
```

## TensorAccessor Compile-Time Arg Layout

`read_matmul_tiles` creates two `TensorAccessor` objects using chained `TensorAccessorArgs`.
The program factory must insert their compile-time args in order after any named CB args:

```
CTA[0 .. N-1]:  accessor args for in0  (s0_args = TensorAccessorArgs<0>())
CTA[N .. M-1]:  accessor args for in1  (s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>())
```

`write_matmul_tiles` uses a single `TensorAccessor` at offset 0:
```
CTA[0 .. N-1]:  accessor args for out  (s_args = TensorAccessorArgs<0>())
```

## WaitMode Trade-offs

- `WaitPerTile` (default): CB depth 1 sufficient for in0 and in1. Reader and compute
  naturally pipeline tile-by-tile. Use for all standard cases. Compatible with
  `read_matmul_tiles()`.
- `WaitUpfront`: CB must hold the full Mt-row block (in0 >= Kt pages, in1 >= Kt*Nt pages).
  Reader must pre-load the full block before compute begins. `read_matmul_tiles()` does NOT
  support this mode — a hand-written reader is required.
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
- `NoReconfigure`: use when the kernel only ever calls `matmul_1d` and no other op. Avoids
  redundant reconfiguration overhead.
- `UnpackReconfigure` / `PackReconfigure`: partial reconfiguration for mixed-precision cases.

## Static Asserts

The implementation enforces at compile time:
- `in0_cb != out_cb`
- `in1_cb != out_cb`
- `in0_cb < 32`, `in1_cb < 32`, `out_cb < 32`
