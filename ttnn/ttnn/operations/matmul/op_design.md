# Operation Design: matmul (2D dual-multicast)

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + data_movement; fused on-device kernels — NOT a composite of `ttnn.matmul`) |
| Goal | Compute `C = A @ B` for activation `A (..., M, K)` and weight `B (K, N)` (Phase 0 shared 2D weight), distributed 2D across a Tensix core grid with dual orthogonal multicasts. |
| Math | `C[..., m, n] = sum_k A[..., m, k] * B[k, n]` — matches `torch.matmul`. |
| Mode | Hybrid (custom fused kernels; no ttnn-op delegation). |
| References | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`, `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`, `tech_reports/tensor_accessor/tensor_accessor.md`, `eval/golden_tests/matmul/feature_spec.py`, `.claude/references/precision_convention.md`. Reference impl studied: `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp` and `tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast/`. |

### Algorithm summary

The output tile-grid `Mt × Nt` (`Mt = ceil(M/32)`, `Nt = ceil(N/32)`) is mapped onto a 2D Tensix grid `GR × GC` (grid rows × grid cols):

- **Grid rows (Y axis) own row-blocks of M**; **grid columns (X axis) own column-blocks of N**.
- **First column of each grid-row (X=0)** reads its activation row-block from DRAM and **multicasts it ACROSS the row** (varying X). Every core in a grid-row shares the same activation row-block.
- **First row of each grid-column (Y=0)** reads its weight column-block from DRAM and **multicasts it DOWN the column** (varying Y). Every core in a grid-column shares the same weight column-block.
- Each core matmuls its received activation row-block against its received weight column-block, accumulating its `MxN` output block. Both multicasts stream **block-by-block along K**.

L1 is kept **bounded for arbitrarily large M, N, K** (an OOM is a correctness failure):
- **K**: streamed in K-blocks of `in0_block_w` tiles — in0/in1 CBs are sized per-K-block, never per-K. The accumulator block is K-independent.
- **M, N**: when a core's assigned output region exceeds the L1 budget, the core processes its region in **bounded output blocks** (`block_M_tiles × block_N_tiles`) via an outer per-core block loop. The whole grid iterates these blocks in lock-step so each multicast still serves all cores in its row/column.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (A) | `ttnn.Tensor` | yes | rank 2/3/4, `(..., M, K)`, TILE_LAYOUT, DRAM interleaved | — | tensor |
| `weight` (B) | `ttnn.Tensor` | yes | Phase 0: 2D `(K, N)` (or leading dims all 1). `B[-2] == A[-1]`. TILE_LAYOUT, DRAM interleaved | — | tensor |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (keyword-only) | any `math_fidelity` / `math_approx_mode`; `fp32_dest_acc_en ∈ {True, False}` | `default_compute_kernel_config()` (HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`) | host → CT/compute-config |

`math_fidelity` and `math_approx_mode` are **never gated** (any value accepted). `fp32_dest_acc_en` is a precision axis; `{dtype=float32, fp32_dest_acc_en=False}` is refused via an op-side EXCLUSION (maxed input demands maxed accumulator — see `.claude/references/precision_convention.md`).

## Tensors

### Input — activation A

| Property | Requirement |
|----------|-------------|
| Shape | `(..., M, K)`, rank 2/3/4 (leading batch dims allowed). `B = prod(leading dims)` total batches. |
| Dtype | Phase 0: `float32`. TARGET: `{float32, bfloat16, bfloat8_b}` (the `dtype` axis). |
| Layout | TILE_LAYOUT |
| Memory | DRAM interleaved |

### Input — weight B

| Property | Requirement |
|----------|-------------|
| Shape | Phase 0: 2D `(K, N)` shared across all of A's batches (`weight_batch="single"`). TARGET also: batched `(..., K, N)` matching A's leading dims (`weight_batch="batched"`, refinement). |
| Dtype | Phase 0: `float32`. TARGET: `{float32, bfloat16, bfloat8_b}` (independent `weight_dtype` axis). |
| Layout | TILE_LAYOUT |
| Memory | DRAM interleaved |

### Output — C

| Property | Value |
|----------|-------|
| Shape | `(..., M, N)` — A's leading dims with last dim swapped to N. |
| Dtype | activation dtype (`dtype`). Phase 0: `float32`. |
| Layout | TILE_LAYOUT |
| Memory | DRAM interleaved (`ttnn.DRAM_MEMORY_CONFIG`). |

## Dataflow Strategy

Data path (per output block, per K-block):

```
A in DRAM ── reader(NCRISC) ──► in0 CB ──► compute(TRISC, matmul_block) ──► out CB ──► writer(BRISC) ──► C in DRAM
B in DRAM ── reader(NCRISC) ──► in1 CB ──►        (K-accumulate via interm CB)
                 │
                 └── dual NoC multicast (mcast_pipe): in0 across the grid-row, in1 down the grid-column
```

All tensors are TILE_LAYOUT in and out → **no tilize/untilize** anywhere. Everything is tile-format end to end.

### Per-Tensix RISC split

| RISC | Kernel | Role |
|------|--------|------|
| NCRISC (RISCV_1) | reader | Reads A/B tiles from DRAM (only on sender cores), performs/participates in the two multicasts, fills `in0_cb` and `in1_cb`. One source, 4 KernelDescriptors keyed by per-group CT role flags `(is_in0_sender, is_in1_sender)`. |
| TRISC (unpack/math/pack) | compute | One `matmul_block(...)` per output block; internally loops K-blocks, spills/reloads via `interm_cb`, packs the final block to `out_cb`. Uniform across all cores. |
| BRISC (RISCV_0) | writer | Drains `out_cb` (one output block) and scatters tiles to the correct DRAM positions; skips out-of-range (ragged/phantom) tiles. Uniform across all cores. |

### Tensix-to-Tensix contract (dual multicast)

Two independent NoC-multicast axes, each a `mcast_pipe` Sender/Receiver pair with its own semaphore set:

| Axis | Sender | Receiver rectangle | Multicast content per K-block |
|------|--------|--------------------|-------------------------------|
| in0 (activation, ACROSS grid-row r) | core `(X=0, Y=r)` | `(X=1..GC-1, Y=r)` | `block_M_tiles × in0_block_w` tiles (contiguous in `in0_cb` slot) |
| in1 (weight, DOWN grid-column c) | core `(X=c, Y=0)` | `(X=c, Y=1..GR-1)` | `in0_block_w × block_N_tiles` tiles (contiguous in `in1_cb` slot) |

Semaphores (4 total, created on the union of all grid cores; cells are per-core so each row/column operates independently on the same IDs):

| Semaphore | Initial value | Role |
|-----------|---------------|------|
| `in0_data_ready` | 0 | sender → row receivers "data is in your `in0_cb`" (mcast_pipe `DATA_READY`, Flag). |
| `in0_consumer_ready` | 0 | row receivers → sender "my `in0_cb` slot is free" (mcast_pipe `CONSUMER_READY`, counter). **Host must init to 0** (a receiver ack can land before the sender ctor runs). |
| `in1_data_ready` | 0 | sender → column receivers, weight. |
| `in1_consumer_ready` | 0 | column receivers → sender, weight. |

**Ordering / deadlock-freedom:** every reader does the in0 (row) phase fully before the in1 (column) phase, each K-block. in0 handshakes are confined to a grid-row, in1 to a grid-column, and in0 has no dependency on in1, so the fixed in0→in1 order across all cores is acyclic. Corner `(0,0)` is both in0-sender (row 0) and in1-sender (col 0); first-column cores `(0,Y>0)` are in0-sender + in1-receiver; first-row cores `(X>0,0)` are in0-receiver + in1-sender; interior cores are both receivers.

**Landing-address invariant:** `mcast_pipe::SenderPipe::send(src_l1, dst_l1, size)` dictates the destination L1 address; every receiver lands the block at that same address. Because all cores configure identical CBs and push/pop in lock-step, `get_write_ptr(in0_cb)` is the same address on sender and receivers — pass it as both `src_l1` and `dst_l1`. The sender reads DRAM into its own `in0_cb` slot, then multicasts that slot to the receiver rectangle (sender excluded from the rect → EXCLUDE-source mcast, keeps its own copy).

**Degenerate axes:** if `GC == 1`, every core reads its own activation directly from DRAM (no in0 multicast, no `in0_*` pipe). If `GR == 1`, every core reads its own weight directly (no in1 multicast). The factory omits the empty Sender/Receiver KernelDescriptor groups.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output block = `block_M_tiles × block_N_tiles` output tiles, fully reduced over K. |
| Device grid | `(gx, gy) = device.compute_with_storage_grid_size()`. Map **N → X**, **M → Y**. |
| Block sizing | `out_subblock_h, out_subblock_w` chosen with `out_subblock_h*out_subblock_w ≤ DEST limit` (4 for fp32+`fp32_dest_acc_en`; see DEST table). `block_M_tiles`/`block_N_tiles` are multiples of the subblock dims and capped by the L1 budget (below). `in0_block_w = ttnn.find_max_divisor(Kt, 4)` (K-block width, divides `Kt` for tile-aligned K). `num_k_blocks = Kt / in0_block_w`. |
| Grid extent | `num_m_blocks = ceil(Mt/block_M_tiles)`, `num_n_blocks = ceil(Nt/block_N_tiles)`. `GR = min(gy, num_m_blocks)`, `GC = min(gx, num_n_blocks)`. |
| Per-core loop count (uniform) | `per_core_M_blocks = ceil(num_m_blocks/GR)`, `per_core_N_blocks = ceil(num_n_blocks/GC)`. Every core runs `B * per_core_M_blocks * per_core_N_blocks` output blocks in the **same order** (lock-step) so multicasts stay valid. |
| Per-core block → global block | grid-row `r` owns global block-rows `[r*per_core_M_blocks + local_mb]`; grid-col `c` owns `[c*per_core_N_blocks + local_nb]`. Loop nest on every kernel: `for b: for local_mb: for local_nb: { K-loop }`. |
| Common case | moderate shapes ⇒ `block_* = ceil(dim/grid)`, `num_*_blocks ≤ grid`, `per_core_*_blocks = 1` (one block per core; reduces to the classic production 2D-mcast matmul, no inner loop). |
| Remainder / ragged | Last block-row/col may have fewer valid tiles than `block_*_tiles` (`Mt`/`Nt` not divisible by `block_*_tiles`). Reader **clamps DRAM tile reads to the valid range** (no OOB NoC read); writer **skips tiles** whose global tile index `≥ Mt` or `≥ Nt`. Fully-phantom trailing iterations (only when `num_*_blocks > grid` and not divisible) participate in the multicast handshake (to keep counts uniform) but write nothing. |

**L1 budget (bounds the block size, in tiles of `tile_size(dtype)`):**

```
in0_cb  = block_M_tiles * in0_block_w * 2      (double-buffer)
in1_cb  = in0_block_w  * block_N_tiles * 2      (double-buffer)
out_cb  = block_M_tiles * block_N_tiles         (one block; ×2 optional for writer pipelining)
interm_cb = block_M_tiles * block_N_tiles       (K spill/reload; fp32 format when fp32_dest_acc_en)
require (in0_cb + in1_cb + out_cb + interm_cb) * tile_size(dtype) <= L1_MM_BUDGET   (~1 MB usable)
```

Shrink `block_M_tiles`/`block_N_tiles` (keeping subblock divisibility) until the budget holds. For fp32 (`tile_size=4096`) a safe default corner is `block_M_tiles=block_N_tiles=8`, `in0_block_w∈{1,2,4}`, `out_subblock 2×2`.

### DEST limit (from `fp32_dest_acc_en`)

| sync | 16-bit DEST | fp32 DEST (`fp32_dest_acc_en=True`) |
|------|-------------|-------------------------------------|
| half-sync (default) | 8 tiles | **4 tiles** |
| full-sync | 16 tiles | 8 tiles |

Phase 0 = fp32 + `fp32_dest_acc_en=True` + half-sync ⇒ subblock `out_subblock_h*out_subblock_w ≤ 4`. The kernel-lib honours `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`) automatically; the host must still choose subblock dims `≤` this so the compile-time block fits.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_in0_act` | 0 | `tile_size(A.dtype)` | `block_M_tiles * in0_block_w * 2` | A.dtype | reader (DRAM read on sender, mcast/receive) | compute (`matmul_block` in0) | one K-block per push, double-buffered (streaming) |
| `cb_in1_weight` | 1 | `tile_size(B.dtype)` | `in0_block_w * block_N_tiles * 2` | B.dtype | reader | compute (`matmul_block` in1) | one K-block per push, double-buffered |
| `cb_out` | 16 | `tile_size(out.dtype)` | `block_M_tiles * block_N_tiles` (×2 optional) | out.dtype (= A.dtype) | compute (`matmul_block` out) | writer | one full output block per push |
| `cb_interm` | 24 | `tile_size(interm_fmt)` | `block_M_tiles * block_N_tiles` | `float32` when `fp32_dest_acc_en` else out.dtype | compute (`matmul_block` spill) | compute (`matmul_block` reload) | K spill/reload, internal to `matmul_block` |

**CB sync (push = wait) per output block:**
- `cb_in0_act`: reader pushes `block_M_tiles*in0_block_w` tiles `num_k_blocks` times; `matmul_block` waits/pops `in0_block_num_tiles = block_M_tiles*in0_block_w` per K-block, `num_k_blocks` times. ✓
- `cb_in1_weight`: reader pushes `in0_block_w*block_N_tiles` tiles `num_k_blocks` times; `matmul_block` waits/pops `in1_block_num_tiles` per K-block, `num_k_blocks` times. ✓
- `cb_out`: `matmul_block` reserves the full block + pushes once; writer waits for `block_M_tiles*block_N_tiles` + pops once. ✓
- `cb_interm`: fully owned by `matmul_block` (spill on non-last K-block, reload on next). Distinct index from `cb_out` (required: in0/in1/out must be distinct CBs).

`cb_in0_act`, `cb_in1_weight`, `cb_out`, `cb_interm` are allocated on **all grid cores at identical indices/sizes** — this guarantees the multicast landing-address invariant.

## API Mapping

Every mechanism with an exact file:line reference. Line numbers are from the headers read during design; the implementer must confirm exact template-arg spellings against the header (the kernel-lib has evolved enum names).

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| in0 multicast send (sender cores) | helper | `mcast_pipe::SenderPipe<NOC,DATA_READY,PRE_HS,CONSUMER_READY,Flag>::send(src_l1,dst_l1,size)` | `mcast_pipe.hpp:156,173,176` | `NOC_ID`=reader NoC, `DATA_READY=in0_data_ready`, `PRE_HANDSHAKE=true`, `CONSUMER_READY=in0_consumer_ready`; ctor `(noc, McastRect{...}, ACK_EQUALS_FANOUT)` | reads `cb_in0_act` slot | writes receivers' `cb_in0_act` | rect excludes sender; `src_l1=dst_l1=get_write_ptr(cb_in0_act)`. Host inits `in0_consumer_ready=0`. |
| in0 multicast receive (receiver cores) | helper | `mcast_pipe::ReceiverPipe<DATA_READY,PRE_HS,CONSUMER_READY,Flag>::receive(sender_x,sender_y)` | `mcast_pipe.hpp:228,240,242` | same sem IDs as sender; ctor `(noc)`; `receive(sender_vx, sender_vy)` | — | lands into `cb_in0_act` slot | call `cb_reserve_back(cb_in0_act,...)` before `receive`, `cb_push_back` after |
| in1 multicast send/receive | helper | same `SenderPipe`/`ReceiverPipe` with `DATA_READY=in1_data_ready`, `CONSUMER_READY=in1_consumer_ready` | `mcast_pipe.hpp:156–251` | rect = column strip | `cb_in1_weight` | `cb_in1_weight` | same contract, down-column |
| mcast geometry | helper | `mcast_pipe::McastRect<NOC>{x0,y0,x1,y1}` | `mcast_pipe.hpp:99` | virtual NoC corners from `device.worker_core_from_logical_core(...)` (RT args) | — | — | normalizes corners per NoC |
| DRAM tile gather (sender reads) | raw_api | `TensorAccessor` + `noc_async_read` | `tech_reports/tensor_accessor/tensor_accessor.md`; CB fundamentals L208–230 | tile-id = `b*Mt*Kt + m_tile*Kt + k_tile` (A), `k_tile*Nt + n_tile` (B) | DRAM | `cb_in0_act`/`cb_in1_weight` | clamp tile ids to valid range for ragged/phantom blocks |
| matmul (K-loop, per output block) | helper | `matmul_block<...>(in0_buf,in1_buf,out_buf,interm_buf,shape)` | `matmul_block_helpers.hpp:334` | `last_block_target=Out`, `tile_order=OutputCBLayout::SubblockMajor` (default), `init_mode=Short`, `in0/in1_policy=WaitAndPopPerKBlock`, `packer_l1_acc=false` (Phase 0), `reconfig=INPUT_AND_OUTPUT` | `cb_in0_act`, `cb_in1_weight` | `cb_out` (final), `cb_interm` (spill) | helper owns CB reserve/push/wait/pop; in0/in1/out must be distinct CBs |
| matmul shape | helper | `MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, in0_block_k, num_k_blocks, batch=1)` | `matmul_block_helpers.hpp:157` | `in0_num_subblocks=block_M_tiles/out_subblock_h`, `in1_num_subblocks=block_N_tiles/out_subblock_w`, `in0_block_k=in0_block_w`, `num_k_blocks=Kt/in0_block_w`, `batch=1` (outer batch loop is in the kernel) | — | — | — |
| compute boot (once) | helper | `compute_kernel_hw_startup(...)` + `mm_block_init(...)` | `matmul_block_helpers.hpp:231–234` (prerequisite note) | boot before the block loop; `matmul_block` issues `mm_block_init_short` per call (`init_mode=Short`) | — | — | caller's one-time responsibility |
| DEST capacity | helper | `DEST_AUTO_LIMIT` / `get_dest_limit()` | `dest_helpers.hpp:103,89` | compile-time DEST tile cap from `DST_ACCUM_MODE`/sync | — | — | host must size subblock `≤` this |
| output scatter (writer) | raw_api | `TensorAccessor` + `noc_async_write` | `tech_reports/tensor_accessor/tensor_accessor.md` | reads `cb_out` SubblockMajor; tile-id = `b*Mt*Nt + m_tile*Nt + n_tile` | `cb_out` | DRAM | skip tiles with `m_tile≥Mt` or `n_tile≥Nt` |

### Helpers considered and rejected (for the raw-API entries)

- **DRAM tile gather (sender reads) — raw `TensorAccessor`+`noc_async_read`.**
  - `tilize_helpers.hpp` / `untilize_helpers.hpp`: rejected — both operands are already TILE_LAYOUT in DRAM; no RM↔tile conversion exists. (`tilize_helpers.hpp` only converts RM sticks to tiles.)
  - `mcast_pipe.hpp`: used for the *multicast* of the gathered block, but it multicasts an **already-resident L1 region** (`send(src_l1,...)`); it does not read DRAM. The DRAM→L1 gather (with the per-`(b, m_tile, n_tile, k_tile)` interleaved tile addressing and edge clamping) has no kernel-lib helper — `TensorAccessor` (`tech_reports/tensor_accessor/tensor_accessor.md`) is the designated primitive for DRAM tile addressing.
- **Output scatter (writer) — raw `TensorAccessor`+`noc_async_write`.**
  - `reblock_untilize_helpers.hpp` / `untilize_helpers.hpp`: rejected — output is TILE_LAYOUT, no untilize needed. `reblock_and_untilize` (`reblock_untilize_helpers.hpp`) gathers matmul subblock-major output into row-major *and untilizes*; here the output stays tiled, so its untilize stage is provably wrong (would byte-reinterpret tile data as RM sticks). The remaining work — scatter tiles to interleaved DRAM positions with edge skipping — is plain `TensorAccessor` addressing with no covering helper.

No raw compute APIs are used: the entire compute phase is `matmul_block` (no hand-rolled `matmul_tiles`/`tile_regs_*`/spill loop).

## Compute Phases

Per output block `(b, local_mb, local_nb)`, every core runs this sequence; the whole grid runs the blocks in lock-step.

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| boot | `compute_kernel_hw_startup` + `mm_block_init` | yes | — | — | matmul unpack/math state configured (once, before all blocks) |
| K-stream | reader fills `cb_in0_act` (`block_M_tiles*in0_block_w`) and `cb_in1_weight` (`in0_block_w*block_N_tiles`), once per K-block, via DRAM read (senders) + dual multicast | `mcast_pipe` (+ raw DRAM read) | DRAM | `cb_in0_act`, `cb_in1_weight` (one K-block each) | a K-block ready in each input CB |
| matmul | `matmul_block<...>(in0,in1,out,interm, MatmulBlockShape::of(...))` — internally loops `num_k_blocks`, waits/pops one K-block per iteration, spills/reloads `cb_interm`, packs final block | `matmul_block` | `cb_in0_act`, `cb_in1_weight` (consumed `num_k_blocks` times), `cb_interm` (internal) | `cb_out` (`block_M_tiles*block_N_tiles`, SubblockMajor) | inputs drained; `cb_out` holds the finished output block |
| writeback | writer waits `cb_out` full block, scatters tiles to DRAM (skip ragged/phantom), pops | raw (`TensorAccessor`) | `cb_out` | DRAM | `cb_out` free for next block |

After all blocks: every output tile with `m_tile < Mt`, `n_tile < Nt`, for every batch, has been written.

## Registry contract (op file)

The implementer turns `ttnn/ttnn/operations/matmul.py` (currently a nuked stub) into a package `ttnn/ttnn/operations/matmul/` that **exports** `matmul`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `default_compute_kernel_config` (consumed by `eval/golden_tests/matmul/axes.py`, `helpers.py`, `test_golden.py`).

### INPUT_TAGGERS (op-local)

```python
def tag_alignment(inputs, axes):
    (A_shape, B_shape) = inputs
    M, K, N = A_shape[-2], A_shape[-1], B_shape[-1]
    if M % 32 == 0 and K % 32 == 0 and N % 32 == 0:
        return "tile_aligned"
    if K % 32 != 0:   return "k_non_aligned"
    if N % 32 != 0:   return "n_non_aligned"
    return "m_non_aligned"            # precedence K > N > M

def tag_weight_batch(inputs, axes):
    B_shape = inputs[1]
    leading = B_shape[:-2]
    return "single" if (len(leading) == 0 or all(d == 1 for d in leading)) else "batched"

INPUT_TAGGERS = {"alignment": tag_alignment, "weight_batch": tag_weight_batch}
```

### SUPPORTED (Phase 0)

```python
SUPPORTED = {
    "dtype":            [ttnn.float32],
    "weight_dtype":     [ttnn.float32],
    "layout":           [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "alignment":        ["tile_aligned"],
    "weight_batch":     ["single"],
}
```

### EXCLUSIONS

```python
EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},  # maxed input demands maxed accumulator
]
```

(Inside Phase 0 this is redundant with `fp32_dest_acc_en=[True]`, but it is mandatory and future-proofs the `fp32_dest_acc_en` refinement.)

### default_compute_kernel_config

```python
def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
```

Single source of truth — both the entry point (resolves `compute_kernel_config=None` through it) and `axes.py`'s tagger read this factory. Do not hardcode the default elsewhere.

### validate() and entry point

`validate()` axes dict mirrors the harness: `{dtype: A.dtype, weight_dtype: B.dtype, layout: A.layout, fp32_dest_acc_en: bool(getattr(cfg or default_compute_kernel_config(), "fp32_dest_acc_en", True))}`, then apply `tag_alignment`/`tag_weight_batch` over `(A.shape, B.shape)`. Gate per-axis against `SUPPORTED` (raise `UnsupportedAxisValue`), then `EXCLUSIONS` (raise `ExcludedCell`) — both subclass `NotImplementedError`.

**Structural shape-contract checks (raise `ValueError`/`RuntimeError`, BEFORE the registry gate):**

| Check | Exception |
|-------|-----------|
| `A.rank < 2` | `ValueError` |
| `A.shape[-1] != B.shape[-2]` (K mismatch) | `ValueError` |
| B is neither 2D `(K,N)` (or leading dims all 1) nor a batched `(...,K,N)` whose leading dims equal A's | `ValueError` |

The public `matmul(input_tensor, weight, *, compute_kernel_config=None)` calls the shape checks then `validate()` as its first actions, resolves the config through `default_compute_kernel_config()`, allocates the `(..., M, N)` output (activation dtype, TILE_LAYOUT, DRAM), builds the ProgramDescriptor, and dispatches via `ttnn.generic_op([A, B, output], program_descriptor)` (output tensor last).

## Structural impossibilities (INVALID review)

Pipeline mode: `eval/golden_tests/matmul/feature_spec.py` already declares `INVALID = []` and I do not edit it. I concur it is correct: `TARGET["layout"]` is TILE-only, so the canonical `{bf8b, ROW_MAJOR}` rule is vacuous (no ROW_MAJOR cell exists in the cartesian). Every `dtype × weight_dtype × fp32_dest_acc_en` combination is meaningful (the two dtypes describe different tensors and don't structurally couple; `fp32 + fp32_dest_acc_en=False` is an EXCLUSION, not INVALID). **No additional INVALID candidates.**

## Key Risks and Gotchas

- **Lock-step is load-bearing.** Every core (reader/compute/writer) must run `B × per_core_M_blocks × per_core_N_blocks × num_k_blocks` iterations in the identical order. A mismatch desyncs the multicast handshake counts → hang (`cb_wait_front`/`consumer_ready` stall). `per_core_M_blocks`/`per_core_N_blocks` are computed with `ceil` so they're uniform across the grid even when work is ragged.
- **Multicast landing address.** CBs must be allocated identically (index, size, format) on all grid cores; `send`/`receive` rely on `get_write_ptr(cb)` being the same L1 address on sender and receivers. Do not give different CB sizes to sender vs receiver groups.
- **Host must init `*_consumer_ready` semaphores to 0** (not the sender ctor) — a remote receiver ack can arrive before the sender constructs its pipe (`mcast_pipe.hpp` host contract).
- **`fp32_dest_acc_en=True` halves DEST** → `out_subblock_h*out_subblock_w ≤ 4` (Phase 0). Size the subblock against `DEST_AUTO_LIMIT` from the start; an oversized subblock corrupts DEST silently.
- **`cb_interm` must be a distinct CB index from `cb_out`** (and from in0/in1) — `matmul_block` aliasing corrupts FIFO state. Phase 0 keeps it a separate full-block region (`float32` format for fp32 acc).
- **Output tile order.** `matmul_block` packs `cb_out` SubblockMajor (default); the writer must read in the matching subblock-major order and map each tile to its DRAM `(m_tile, n_tile)`. Reading in plain row-major order would transpose subblocks.
- **K-blocks bound L1, not the whole K.** `in0_block_w` is the per-K-block width; never size `cb_in0_act`/`cb_in1_weight` to the full `Kt`. Large K is just more `num_k_blocks` iterations.
- **Ragged/phantom edges (large M/N, and the non-aligned refinements).** Reader clamps DRAM reads to valid tile range (no OOB NoC read); writer skips out-of-range output tiles. Even tile-aligned shapes can have ragged *tile-block* edges when `block_*_tiles` doesn't divide `Mt`/`Nt`.
- **Non-tile-aligned M/K/N (refinement).** The `alignment` tagger labels the masking path (`k_non_aligned`/`n_non_aligned`/`m_non_aligned`). K is the load-bearing one: the last K-block's partial tile must contribute only valid elements (mask the invalid K columns/rows of in0/in1, or rely on guaranteed-zero tile padding). M/N partial tiles only affect unwritten output rows/columns. Phase 0 declares `alignment=["tile_aligned"]`; these are xfail until the masking refinement lands.
- **Batched weight (refinement).** Phase 0 weight is shared 2D (`weight_batch="single"`); the in1 read/multicast is batch-independent. For `weight_batch="batched"`, the in1 tile-id gains a `b*Kt*Nt` offset and the weight is re-read per batch.

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB (verified per-CB above).
- [x] DEST: subblock `≤` 4 tiles (fp32+`fp32_dest_acc_en`, half-sync) via `DEST_AUTO_LIMIT`.
- [x] `cb_interm` distinct CB from in0/in1/out; sized to full output block.
- [x] Page sizes = `tile_size(dtype)` (all CBs tile-format; no RM CBs).
- [x] All `cb_wait_front` on a given CB use the same page count (`matmul_block` fixes in0/in1 per-K-block counts; writer fixes the full-block count).
- [x] Helpers not wrapped with extra CB ops: `matmul_block` owns its CB lifecycle; `mcast_pipe` owns the handshake; reader only does `cb_reserve_back`/DRAM-read/`cb_push_back` around the pipe, writer only `cb_wait_front`/scatter/`cb_pop_front`.
- [x] `compute_kernel_hw_startup()` + `mm_block_init()` called once before any `matmul_block`.
- [x] No reduce scaler (matmul reduces internally via the FPU, not the reduce helper) — reduce-scaler checks N/A.
