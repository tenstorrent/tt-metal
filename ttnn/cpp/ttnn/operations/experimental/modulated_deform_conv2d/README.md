<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# `modulated_deform_conv2d` — fused modulated deformable convolution (DCNv2)

Design note for the DCNv2 device op.

One entry point, one device operation, one compute kernel (which owns the sampling geometry, the
bilinear columns and the conv's matmul) and two copies of one reader kernel:

| Entry point                                   | Frontend                                                |
| --------------------------------------------- | ------------------------------------------------------- |
| `ttnn.experimental.modulated_deform_conv2d`   | mmcv's `ModulatedDeformConv2dPack`, groups and deform_groups 1 |

---

## 1. Why a new op

A modulated deformable conv samples its input bilinearly at a learned offset from each kernel
position, scales each sample by a learned mask and multiplies the samples by the weight. Composed
from existing ttnn ops, it costs per call:

* a slice of the offset conv's output into offsets and mask, and a separate sigmoid,
* offsets converted to `grid_sample`'s normalized `[-1, 1]` coordinates and layout,
* one `ttnn.grid_sample` over all `K_h * K_w` sampling points per input-channel chunk, which
  gathers every corner from interleaved memory and writes the sampled `(B*H_out*W_out, K*K*C)`
  columns out,
* a mask multiply, reshapes and a tilize of those columns,
* a matmul per chunk and an add of the partial outputs.

The sampled columns are `K_h * K_w` times the input, and they round-trip through DRAM. In
BEVFormer-base, `GridSampleOperation` alone took 168 ms of the backbone's 557 ms.

`modulated_deform_conv2d` takes the offset conv's raw output and the weight, and never materializes
the columns: each core gathers the corners of its own output rows, builds their columns in L1 and
multiplies them by the weight there.

---

## 2. Public API

```python
out = ttnn.experimental.modulated_deform_conv2d(
    input,          # (B, H, W, C)               ROW_MAJOR bf16, INTERLEAVED or HEIGHT_SHARDED (L1)
    offset_mask,    # (B, H_out, W_out, 3*K*K)   ROW_MAJOR bf16, INTERLEAVED
    weight,         # (K*K*C, C_out)             TILE bf16, INTERLEAVED, rows ordered (kk, c)
    bias,           # (1, C_out) TILE bf16 INTERLEAVED, or None
    kernel_size=(K_h, K_w),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    relu=False,
    memory_config=None,  # the input's, or DRAM for a sharded input
)
# -> (1, 1, B*H_out*W_out, C_out) TILE bf16, interleaved: conv2d's output layout
```

`C` and `C_out` are multiples of 32. Every TILE operand uses the default, non-transposed 32x32 tile.

A height-sharded input must be in L1, in whole 32-row shards of whole rows, and the output must have
the input's height and width (stride 1 with "same" padding).

---

## 3. Offset and mask convention

`offset_mask` is what mmcv's `ModulatedDeformConv2dPack` offset conv emits, unsplit:

```text
channel 2*kk     y offset of kernel position kk, in input pixels
channel 2*kk + 1 x offset of kernel position kk, in input pixels
channel 2*K*K + kk mask logit of kernel position kk (the op applies the sigmoid)
kk = k_h * K_w + k_w
```

torchvision's form, a separate offset tensor and a mask after the sigmoid, is not taken.

Offsets and mask logits saturate at ±4096. The host validates that each input size plus its padding,
and the dilated kernel span, are at most 4096, so every sample coordinate is exact in fp32 and a
saturated offset lands where all four of its corners are outside the input. A non-finite offset
therefore samples zero, where mmcv and torchvision would give NaN.

---

## 4. Math

For output pixel `m = (b, h_o, w_o)`, kernel position `kk = (k_h, k_w)` and channel `c`:

```text
y      = h_o * stride_h - pad_h + k_h * dilation_h + offset_mask[m, 2*kk]
x      = w_o * stride_w - pad_w + k_w * dilation_w + offset_mask[m, 2*kk + 1]
mask   = sigmoid(offset_mask[m, 2*K*K + kk])
column[m, kk*C + c] = mask * bilinear(input[b, :, :, c], y, x)
out[m, :] = column[m, :] @ weight + bias          (then ReLU if relu)
```

Corners outside the input contribute zero, as in mmcv and torchvision.

Precision: the geometry is fp32. The columns are bfloat16 and the products run at HiFi2. Each pass's
reduction, over at most 256 channels of one kernel position, accumulates in fp32 dest; the sum across
passes and kernel positions accumulates in bfloat16 in L1.

---

## 5. Core decomposition

The unit of work is a **block** of 32 consecutive output rows `m`. A block's whole reduction, every
kernel position and channel, runs on the core that owns it — **no cross-core reduction**.

```text
blocks            = ceil(B * H_out * W_out / 32)
interleaved input   blocks spread evenly over the compute grid
sharded input       each core takes the blocks of its own shard; a core with an empty shard
                    still joins the weight multicast
```

Within a core the loop order is:

```text
for pass in 0 .. K*K * chunks:        kernel position kk, then channel chunk
    load the pass's (chunk x C_out) slice of the weight
    for block in the core's blocks:
        gather the block's corners for (kk, chunk)
        build its columns, multiply by the slice, accumulate into the block's output
after the last pass: bias, ReLU, write every block's output
```

Kernel positions run outermost so one weight slice serves all of a core's blocks before the next
replaces it. An input wider than one chunk (8 tiles, 256 channels) takes one pass per chunk at each
position, the chunk being the widest of at most 8 channel tiles that divides `C / 32`: the corner
blocks and two weight slices then stay small enough for L1.

Every core keeps the outputs of all its blocks in L1 across passes. That, not the columns, is what
bounds the op's L1 (§8).

---

## 6. Kernels

**Two readers per core.** One copy of `reader_deform_conv.cpp` runs on each data-movement RISC-V. The
first takes the core's even blocks, the second the odd ones. Each copy stages the geometry inputs of
its own blocks (the block's offset and mask rows), receives the block's geometry back from compute
and keeps it for every pass, then for each `(pass, block)` gathers the corners:

* the four corners of every row as one row-major block, 32 rows of
  `[corner 0 | corner 1 | corner 2 | corner 3]` chunk-wide input sticks,
* one scalar tile per corner, whose column 0 holds the row's bilinear weight times its mask.

A corner of zero weight, out of bounds, masked or past the block's end, is zeroed rather than read:
a stale NaN times a zero scalar would still be NaN. After the last pass each copy writes its own
blocks' outputs.

The reader with fewer blocks on the busiest cores also loads the weight slices and the bias.

**Compute** (`compute_deform_conv.cpp`):

* **Geometry**, on the SFPU in fp32, during the first pass and two blocks ahead of the block's
  corners: each sample's top-left corner page within its image and its four weights (bilinear
  coefficient times the mask's sigmoid). The data-movement RISC-Vs have no FPU, so the readers only
  move bits.
* **Columns**: the four corner blocks are tilized on the unpacker (fast tilize), multiplied by their
  scalar columns and summed in fp32 dest, on half-sync dest.
* **Matmul**: the block's columns times the pass's weight slice, in fp32 dest within the pass,
  accumulated into the block's output tiles in bfloat16 L1 across passes. The first pass overwrites.
* **Epilogue**: bias, ReLU, and the bfloat16 output handed to the block's reader.

**Weight multicast.** When the cores form one rectangle, one core reads each weight slice from DRAM
and multicasts it to the others, so a slice costs one DRAM read instead of one per core; per-core
reads otherwise congest the corner gather. A ragged core set reads the slices per core. Each core
claims the next slice's slot after its first block of the current pass, so the load overlaps the
gather.

**Sharded input.** With a height-sharded input both readers run in dynamic-NoC mode: corners of the
previous core's shard go over NoC 0, the next core's over NoC 1, the core's own shard is local L1.
The weight multicast is then unlinked and waits for its writes before the semaphore: a linked
multicast holds its path, and the other reader's reads crossing it would deadlock. Each reader drains
all of its NoC's transactions, the multicast receivers' semaphore atomics included, before it exits.

**Input band (interleaved input).** A core's blocks are consecutive output rows, which sample a band
of input rows: the kernel's rows plus a margin for the offsets. When it fits L1 the readers copy that
band once and read the corners inside it from there; corners outside it come from the input. Without
the band, every core's corner reads spread over every core's L1 saturate the NoC around the first
rows of cores.

---

## 7. Circular buffers

| CB | Role | Pages | Page |
| --- | --- | --- | --- |
| `c_0`, `c_1` | each reader's offset-mask staging arena | 16 | offset-mask row |
| `c_2`, `c_9` | each reader's corner blocks, 2 deep | `2 * 4 * chunk` | bf16 tile |
| `c_3`, `c_10` | each reader's scalar tiles, 2 deep | `2 * 4` | bf16 tile |
| `c_4` | a block's tilized corners | `4 * chunk` | bf16 tile |
| `c_5`, `c_21` | each reader's geometry inputs, one block | 5 | fp32 tile |
| `c_6`, `c_11` | each reader's blocks' corner pages, for the whole run | one per block | int32 tile |
| `c_7`, `c_12` | each reader's blocks' corner weights, for the whole run | 4 per block | bf16 tile |
| `c_8` | the kernel positions' offsets | 2 | fp32 tile |
| `c_22` | the geometry's fp32 intermediates | 5 | fp32 tile |
| `c_13`, `c_14` | the input band's two halves (interleaved input) | band / 2 | input row |
| `c_15` | weight slices, 2 deep | `2 * chunk * C_out/32` | bf16 tile |
| `c_16`, `c_17` | each reader's output, 1 or 2 blocks deep | `C_out/32` per block | bf16 tile |
| `c_18` | a block's columns of one pass | `chunk` | bf16 tile |
| `c_19` | every block's output accumulator | `blocks * C_out/32` | bf16 tile |
| `c_20` | the bias | `C_out/32` | bf16 tile |

---

## 8. L1 budget and program cache

The buffers are sized from the L1 the circular buffers can reach at the call, less the output's
per-bank bytes when the output goes to L1, rounded down to 32 KB. The budget picks the output depth
and whether the input band fits, and the op fails with the L1 it needs when the fixed buffers do not
fit:

```text
modulated_deform_conv2d: needs N KB of L1 per core, M KB free: ...
```

The budget is an operation attribute, so it keys the program cache: a cached program never assumes
more L1 than its call had, while small shifts in occupancy still reuse it. Other L1 tensors allocated
at the call can therefore change the program or make the op fail.

All buffer addresses are runtime arguments, so a cache hit binds the new call's tensors.

---

## 9. Constraints

| Property | Value | Why |
| --- | --- | --- |
| dtype | `bfloat16` (all tensors) | |
| layout | input and `offset_mask` `ROW_MAJOR`; weight and bias `TILE`, default 32x32 tiles | the readers address the input by stick; the kernels step through the weight as 32x32 tiles in row order |
| memory | `INTERLEAVED`; the input may be `HEIGHT_SHARDED` in L1 | |
| `C`, `C_out` | multiples of 32 | tile columns |
| `K_h * K_w` | at most 32 | the geometry tiles hold one kernel position per column |
| `H + pad_h`, `W + pad_w`, dilated span | at most 4096 | offsets saturate at ±4096 (§3) |
| groups, deform_groups | 1 | |
| sharded input | whole 32-row shards of whole rows, output of the input's H and W | each core computes its own shard's rows |
| architecture | not Quasar | the compute kernel calls the Blackhole and Wormhole SFPU functions directly |

---

## 10. Files

```
ttnn/cpp/ttnn/operations/experimental/modulated_deform_conv2d/
├── modulated_deform_conv2d.hpp / .cpp                    public API
├── modulated_deform_conv2d_nanobind.hpp / .cpp           Python binding
├── CMakeLists.txt / sources.cmake                        build wiring
└── device/
    ├── modulated_deform_conv2d_device_operation.hpp / .cpp   attrs, L1 budget, validation, output spec
    ├── modulated_deform_conv2d_program_factory.cpp           work split, CBs, multicast, kernels, RT args
    └── kernels/
        ├── dataflow/reader_deform_conv.cpp                   staging, corner gather, weight load, output
        └── compute/compute_deform_conv.cpp                   geometry, columns, matmul, epilogue

tests/ttnn/unit_tests/operations/experimental/test_modulated_deform_conv2d.py
```

The unit tests compare the op with a float64 host reference and with torchvision's `deform_conv2d`,
at BEVFormer's layer3 and layer4 shapes among others, and cover the sharded input, channel passes,
the weight multicast, program-cache hits on fresh buffers and with less L1, and the L1-too-small and
kernel-too-large errors.

## 11. BEVFormer integration

BEVFormer's ResNet101 backbone (base and small) runs each layer3 and layer4 bottleneck's conv2 as
`TTModulatedDeformConv2dPack` (`models/experimental/bevformer/tt/tt_resnet.py`), on
`TTModulatedDeformConv2dDevice` (`models/experimental/bevformer/tt/tt_modulated_deform_conv.py`):

* The offset conv is a conv2d with fp32 dest accumulation, whose bfloat16 ROW_MAJOR output is the
  op's `offset_mask` as it is: offsets in pixels, `(y, x)` per position, then the mask logits.
* The BatchNorm after conv2 is folded into the DCN's weight and bias, and the ReLU runs in the op.
* layer3's DCN takes its input height-sharded over the grid
  (`DCN_SHARDED_INPUT_STAGES` in `model_config.py`), and the offset conv reads the same shard, so the
  block input reaches L1 once. layer4 has few blocks per core and two channel passes, which do not
  pay for the shard; its DCN reads an interleaved L1 copy.

Numerics are pinned by `test_modulated_deform_conv2d.py` and BEVFormer's `test_resnet.py`,
`test_backbone_fpn.py` and `test_bevformer.py` for base and small.
