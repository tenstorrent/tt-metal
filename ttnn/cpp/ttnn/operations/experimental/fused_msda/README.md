<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# `fused_msda` — generic multi-scale deformable attention

Design note for the MSDA device op.

Two entry points, one device operation, one compute kernel (which owns both the
sampling geometry and the reduction), one writer kernel, two readers:

| Entry point                            | Frontend                                             |
| -------------------------------------- | ---------------------------------------------------- |
| `ttnn.experimental.fused_msda`         | V1 — caller supplies materialized sampling locations  |
| `ttnn.experimental.fused_msda_from_offsets` | V2 — locations derived on device from reference points + offsets |

A host-only query, `ttnn.experimental.can_use_fused_msda_from_offsets_l1_value`
(§2.4), tells a caller whether V2's `value` can be staged in L1.

---

## 1. Why a new op

`ttnn.experimental.multi_scale_deformable_attn` (`operations/experimental/multi_scale_deformable_attn/`)
is a **single-level** (`num_levels == 1`) fast path. Its contract is
`value (N, h, w, D)`, `grid (N, Q*P, 1, 2)` in `[-1, 1]`, `attn (N, Q, P)`
with `N = B * num_heads` already folded by the caller.

Consumers therefore still pay, per call, for:

* a per-level Python loop (`models/experimental/vadv2/tt/tt_utils.py:119`,
  `models/experimental/uniad/tt/ttnn_utils.py:238`),
* `split` / `permute` / `reshape` of `value` into `(B*H, H_l, W_l, D)` per level,
* `permute` / `reshape` of the grid and the attention weights per level,
* `stack` + `mul` + `sum` across levels when `L > 1`,
* a `reshape` + `permute` on the output.

`fused_msda` takes the flattened multi-level tensors directly, keeps the
`(heads, levels, points, xy)` semantics in *index arithmetic inside the reader*,
and reduces over `L * P` on-core. No `[B, Q, H, L, P, D]` intermediate is ever
materialized.

The single-level op is a separate contract and is left unchanged. Both remain
callable.

---

## 2. Public API

### 2.1 V1 — `fused_msda`

```python
out = ttnn.experimental.fused_msda(
    value,                      # (B, S, H, D)          ROW_MAJOR bf16   [canonical]
                                #   or (B, S, H*D)      ROW_MAJOR bf16   [packed]
    sampling_locations,         # (B, Q, H, L, P, 2)    ROW_MAJOR bf16   [canonical]
                                #   or (B, Q, H, L*P*2) ROW_MAJOR bf16   [packed]
    attention_weights,          # (B, Q, H, L, P)       ROW_MAJOR bf16   [canonical]
                                #   or (B, Q, H, L*P)   ROW_MAJOR bf16   [packed]
    spatial_shapes,             # host: [(H_0, W_0), ..., (H_{L-1}, W_{L-1})]
    *,
    align_corners=False,
    locations_in_grid_space=False,
    memory_config=None,
)
# -> (B, Q, H*D) ROW_MAJOR bf16
```

```text
B = batch          S = sum_l H_l * W_l      H = heads
Q = queries        L = levels               P = points per (head, level)
D = head_dim
```

`spatial_shapes` is a **host-side static attribute**, not a device tensor. It is
fixed by the feature-pyramid config, it is needed for address arithmetic in the
reader, and reading it from device would force a host sync. `level_start_index`
is derived from it on host (exclusive prefix sum of `H_l * W_l`) and shipped as
runtime args — callers never pass it.

### 2.2 V2 — `fused_msda_from_offsets`

```python
out = ttnn.experimental.fused_msda_from_offsets(
    value,                      # (B, S, H, D)          ROW_MAJOR bf16   [canonical]
                                #   or (B, S, H*D)      ROW_MAJOR bf16   [packed]
    reference_points,           # (B, Q, R, 2)          ROW_MAJOR bf16 or fp32
    sampling_offsets,           # (B, Q, H, L, P, 2)    ROW_MAJOR bf16   [canonical]
                                #   or (B, Q, H, L*P*2) ROW_MAJOR bf16   [packed]
    attention_weights,          # as V1
    spatial_shapes,             # as V1
    *,
    reference_mode="level",     # "level" | "pillar"
    align_corners=False,
    memory_config=None,
)
# -> (B, Q, H*D) ROW_MAJOR bf16
```

V2 is **only** a fused location generator. It computes, per `(b, q, h, l, p)`:

```text
loc = reference_points[b, q, r(l, p)] + sampling_offsets[b, q, h, l, p] * scale_l

scale_l = [1/W_l, 1/H_l]               align_corners = false
scale_l = [1/(W_l-1), 1/(H_l-1)]       align_corners = true, axis wider than 1
scale_l = 0                            align_corners = true, singleton axis
```

and then runs the identical bilinear + weight + reduce path as V1. `r(l, p)`
depends on `reference_mode` — see §7.

`sampling_offsets` are **raw**, in feature-map pixel units.
`scale_l` is the pixel spacing of the mapping in §3, so one offset unit moves
one pixel for either `align_corners`. The op never divides by the extent:
substituting `loc` into §3 gives `px = ref * W_l + off - 0.5` (or
`ref * (W_l-1) + off` with `align_corners`), which the SFPU evaluates from the
per-level constants (§3, §6). No BEVFormer-specific pre-folding is baked into
the op.

### 2.3 Output shape

`(B, Q, H*D)` ROW_MAJOR bf16 — chosen, not accidental. Every consumer in this
repo (BEVFormer, UniAD, VADv2) wants `(bs, num_queries, embed_dims)` immediately
after MSDA. The writer emits each head's `D` values as a `D*2`-byte NoC write at
byte offset `h * D * 2` inside query `(b, q)`'s output page, so the head
concatenation costs nothing and no post-op `reshape`/`permute` is needed.
`(B, Q, H, D)` is deliberately **not** offered; it would be a strictly worse
default and is one `ttnn.reshape` away for anyone who wants it.

### 2.4 L1 staging query — `can_use_fused_msda_from_offsets_l1_value`

```python
fits = ttnn.experimental.can_use_fused_msda_from_offsets_l1_value(
    value, reference_points, sampling_offsets, attention_weights, spatial_shapes,
    *, reference_mode="level", align_corners=False,
)
# -> bool
```

True when `value`, staged as ROW_MAJOR interleaved L1, and the bf16 output fit
below the L1 frontier together with this op's circular buffers, under
`usable_program_l1_capacity` (95%). Per-bank sizes are rounded up to the
allocator's DRAM-alignment granule; a `value` already in ROW_MAJOR L1 is not
counted again. `value` may still be TILE or in DRAM. The answer reflects the
allocator's current state, so call it right before the op and do not cache it.

---

## 3. Coordinate convention

`align_corners` and `locations_in_grid_space` together select the
normalized → pixel mapping, per level `l` with `(H_l, W_l)`:

| `locations_in_grid_space` | `align_corners` | `px` | `py` |
| --- | --- | --- | --- |
| `False` (MSDA/mmcv, `[0,1]`) | `False` | `x*W_l - 0.5` | `y*H_l - 0.5` |
| `False` | `True` | `x*(W_l-1)` | `y*(H_l-1)` |
| `True` (grid_sample, `[-1,1]`) | `False` | `(x+1)*W_l/2 - 0.5` | `(y+1)*H_l/2 - 0.5` |
| `True` | `True` | `(x+1)*(W_l-1)/2` | `(y+1)*(H_l-1)/2` |

Defaults (`False`, `False`) reproduce
`models/experimental/bevformer/reference/ms_deformable_attention.py:58` exactly:
`sampling_grids = 2*loc - 1` followed by `F.grid_sample(..., align_corners=False)`
at line 68.

`locations_in_grid_space=True` exists for callers that already hold a
grid_sample-space grid — for instance one whose sampling-offset projection has
the `2/[W,H]` scale and the `2*ref - 1` shift folded into its weights, a common
trick for saving a rescale per forward. Such a caller can feed the op directly.
It is a coordinate-space flag, not a model hook: no consumer in this repo needs
it today, since BEVFormer uses `fused_msda_from_offsets` instead (§11).

Both flags are resolved **on host**, into a per-level `(primary_scale,
secondary_scale, bias)` triple that the SFPU geometry applies as
`px = primary * primary_scale + secondary * secondary_scale + bias`
(`axis_constants` in the program factory). Nothing on device branches on them.
V2's two rows of that table follow from substituting `loc = ref + off * scale_l`
into the `[0, 1]` rows, which is why `secondary_scale` is 1 except on a
singleton `align_corners` axis.

Padding: **zeros only**. Any of the four bilinear neighbours that falls outside
`[0, W_l) x [0, H_l)` contributes 0, matching `padding_mode="zeros"`. The
in-bounds test is per corner, not per sample — a sample straddling the border
still contributes its in-bounds corners.

---

## 4. Math

For each `(b, q, h)`:

```text
acc[D] = 0
for l in 0..L-1:
    for p in 0..P-1:
        (px, py) = to_pixel(loc[b,q,h,l,p], H_l, W_l)
        x0 = floor(px); y0 = floor(py); dx = px - x0; dy = py - y0
        acc += attention_weights[b,q,h,l,p] * (
              (1-dx)*(1-dy) * V[b, base_l + y0    *W_l + x0    , h, :]
            +   dx  *(1-dy) * V[b, base_l + y0    *W_l + (x0+1), h, :]
            + (1-dx)*  dy   * V[b, base_l + (y0+1)*W_l + x0    , h, :]
            +   dx  *  dy   * V[b, base_l + (y0+1)*W_l + (x0+1), h, :])
out[b, q, h*D : (h+1)*D] = acc
```

`base_l = level_start_index[l] = sum_{k<l} H_k * W_k`.

---

## 5. Core decomposition

The unit of work is **one output tile**: up to 32 consecutive queries of a
single `(b, h)` pair.

```text
total_output_tiles = B * H * ceil(Q / 32)
```

Tiles are distributed with `split_work_to_cores` over
`compute_with_storage_grid_size()`. Queries and heads are independent, and the
whole `4 * L * P` reduction for a tile happens on the core that owns it — **no
cross-core reduction**.

`Q % 32 != 0` is supported: the trailing tile of each `(b, h)` carries
`v_rows = Q % 32` and the reader zeroes the attention lane of rows `>= v_rows`,
which makes their scalar zero, so they contribute nothing and the writer emits
only `v_rows` sticks.

`split_work_to_cores` is used rather than the spec's suggested
`core.x -> query blocks, core.y -> heads` because `B * H * ceil(Q/32)` rarely
factors onto the grid; a flat split load-balances the tail better and the
locality properties (reduction stays local) are identical.

---

## 6. Circular-buffer protocol

Three streams, not one. Per sampling point — one `(level, point)` pair — of an
output tile:

```text
reader  -> compute   geom_x, geom_y, attn_tile  [+ offset_x, offset_y for V2]
compute -> reader    x0, y0                     floor(px), floor(py) as bf16
reader  -> compute   4 x input_tiles            the gathered bilinear corners
                                                (one row-major block of all 4
                                                when D % 32 == 0, tilized by
                                                compute into tiled_input)
compute -> compute   4 x scalar_tile            attn * corner coefficient
```

| CB | Role | Pages | Page size |
| --- | --- | --- | --- |
| `c_0` `value_scratch` | reader-only L1 arena, one staged `D`-stick per row | 32 | `align(D*2)` |
| `c_1` `attn_scratch` | reader-only arena for attention weights | 32 (packed) or 32*L | `align(attn_stick)` |
| `c_2` `loc_scratch` | reader-only arena for sampling locations (V1) / offsets (V2) | 32 (packed) or 32*L*P | `align(loc_stick)` |
| `c_3` `input_tile` | reader → compute, the `V_corner` values: tiles, or a row-major `[NW \| NE \| SW \| SE]` block per point when `D % 32 == 0` | `8 * n_d_tiles` | 2048 |
| `c_4` `scalar_tile` | compute → compute, `attn * bilinear_coeff` per row | 8 | 2048 |
| `c_5` `output_scratch` | writer-only assembly of one head's `D`-stick | 1 | `align(D*2)` |
| `c_6` `ref_scratch` | **V2 only** — reference points arena, in the tensor's dtype. float32 builds a separate program, which keeps the point float32 through `geom_x` / `geom_y` | 32*R | `align(2*sizeof(ref))` (raw stick 4 B bf16, 8 B fp32) |
| `c_7` `geom_x`, `c_8` `geom_y` | reader → compute, the primary operand (location, or reference point for V2) | 4 | 2048, or 4096 when the reference point is float32 |
| `c_9` `offset_x`, `c_10` `offset_y` | reader → compute, the raw sampling offset (**V2**; allocated but unread on V1) | 4 | 2048 |
| `c_11` `attn_tile` | reader → compute, the attention weight | 4 | 2048 |
| `c_12` `x0`, `c_13` `y0` | compute → reader, the floored corner | 3 | 2048 |
| `c_14` `frac_x`, `c_15` `frac_y` | compute → compute, `px - floor(px)` | 2 | 2048 |
| `c_16` `output_tile` | compute → writer, the accumulator | `2 * n_d_tiles` | 2048 |
| `c_17` `tiled_input` | compute → compute, the tilized corners (only when `D % 32 == 0`) | `8 * n_d_tiles` | 2048 |

`n_d_tiles = ceil(D / 32)`. CB pressure for the BEVFormer shape
(`D=32, L=4, P=4, R=4`, packed) is ~126 KB, ~142 KB with float32 reference
points.

Deliberately **not** four separate `CB_VALUE_00/01/10/11` CBs and a separate
`CB_INTERP` / `CB_ATTN_WEIGHT`: the four corners are streamed **sequentially**
into the same `input_tile` CB, and `dx, dy` never cross to the reader — they
exist as `c_14` / `c_15` but stay inside the compute kernel. They are folded
into the scalar the reduction is already going to multiply by:

```text
scalar[row] = attention_weight * corner_coefficient
```

That is one scalar per `(row, level, point, corner)` instead of two interpolation
factors plus a weight, and it removes the bilinear blend from the reduction
entirely.

### Why the geometry pipe exists

Deriving `px`, `x0`, `dx` and the four corner weights is per-point float work
over 32 query rows, and the dataflow RISC has no FPU — every float operation
there costs ~140 cycles of soft-float emulation. The SFPU does it instead: the
reader ships the operands as column-0 tiles and gets `floor(px)` back, and
the only arithmetic left in the reader is the integer decode, the bounds test
and the page index. Those tiles are bf16, except `geom_x` / `geom_y` when
`reference_points` are float32. The SFPU multiplies that point, and a bf16 tile
would already have moved it in steps of up to 2^-8. Those two CBs are also
unpacked straight into the 32-bit destination (`UnpackToDestFp32`):
`copy_tile` through SrcA would keep only 10 mantissa bits.

Both sides run **one sampling point ahead** of the work they feed — the reader
pushes point `j+1`'s operand tiles before waiting for point `j`'s corners, and
the compute kernel solves point `j+1` before reducing point `j`. Without that
lookahead the two ping-pong, each idle through the other's turn. The CB depths
above are what make it legal: `scalar_tile` and `input_tile` hold two points'
worth, the reader ↔ compute geometry pipes three to four, `frac` two.

The floored corner crosses as **bf16**, which is exact for every integer up to
256 and only some beyond it, so each level's `H` and `W` must be at or below
256 — validated on host, see §9. `fp32_dest_acc_en` is required for the same
reason one step earlier: `px` reaches the feature map's extent and bf16's ulp
at 200 is 1.0, so a 16-bit destination would round `floor(px)` to the wrong
integer on the larger levels and collapse bilinear sampling to
nearest-neighbour.

### Tile packing

32 queries are packed vertically into a tile; `D` spans `n_d_tiles` tiles side by
side. `msda_tile_layout.hpp` maps row `r` to its `(lo, hi)` face-half byte
offsets. The compute kernel uses `mul_tiles_bcast<COL>`, which reads only column
0 of the `TL`/`BL` faces of the scalar tile, and the geometry only ever produces
a meaningful column 0 — so a query occupies one lane of a 32x32 tile throughout.

### Zero-fill contract

* **geometry tiles** (reader → compute): column 0 is written for all 32 rows on
  every emission; rows `>= v_rows` get `0` in that tile's format. A zero attn lane is what makes
  a tail row's scalar zero. Columns 1..31 are zeroed once per CB slot at reader
  startup, so an uninitialised L1 bit pattern never reaches the SFPU as a NaN.
* **input tile** (reader → compute): rows that are in range **and** in bounds
  hold the gathered value stick; every other row is explicitly zeroed. In the
  row-major block the same holds per corner slot: an in-range, in-bounds slot
  holds its stick, every other slot and every row `>= v_rows` is zeroed.
* **scalar tile** (compute → compute): column 0 for all 32 rows, from the
  fractions and the attention weight.

The input-tile zeroing is load-bearing: the scalar comes from a compute kernel
that cannot know which corners fell outside the feature map, so an un-zeroed
row would be multiplied by a live weight. A stale row is itself a plausible
sampled value, so the correlation survives it — which is why
`test_fused_msda_v1_masks_out_of_bounds_corners` asserts a per-element error ratio
alongside PCC.

---

## 7. Reference-point modes (V2)

Surveyed in this repo:

| Where | `reference_points` | Mapping |
| --- | --- | --- |
| `models/experimental/uniad/tt/ttnn_deformable_attention.py:134` | `(B, Q, L, 2)` | `ref[b,q,l] + off/[W_l,H_l]` |
| `models/experimental/vadv2/tt/tt_deformable_attention.py:137` | `(B, Q, L, 2)` | `ref[b,q,l] + off`, with `1/[W,H]` already folded into the Linear |
| `models/experimental/bevformer/reference/ms_deformable_attention.py:234` | `(B, Q, Z, 2)` | points grouped `(P//Z, Z)`; `ref[b,q,z] + off/[W_l,H_l]` |
| `models/experimental/uniad/tt/ttnn_deformable_attention.py:141` | `(B, Q, L, 4)` | `ref_xy + off/P * ref_wh * 0.5` |

So there is **no single reference-point convention**, which is exactly why V1
must not depend on one. V2 supports two modes and rejects the rest loudly:

* `reference_mode="level"` — `R == L`, `r(l, p) = l`. DINO-family decoders
  (UniAD, VADv2).
* `reference_mode="pillar"` — `R == Z`, `P % Z == 0`, `r(l, p) = p % Z`.
  BEVFormer spatial/temporal attention, where the reference points are the
  `num_points_in_pillar` z-anchors and the point axis is laid out as
  `(P//Z, Z)` — matching
  `sampling_offsets.view(bs, Q, H, L, P//Z, Z, 2)` in the reference.

With `L == 1` / `Z == 1` the two modes coincide.

The **4-D box mode is not supported** and is rejected with an explicit message
rather than silently reinterpreted — its offset scaling (`/P * ref_wh * 0.5`)
is a different formula, not a different layout.

---

## 8. Kernel responsibilities

```text
reader_msda_v1.cpp ─┐                                      compute_msda.cpp
                    ├─> fused_msda_reader_common.hpp <──>  + msda_geometry.hpp  ─> writer_msda.cpp
reader_msda_v2.cpp ─┘   (staging, gather, tile scatter)     (SFPU geometry, reduction)
```

**Reader** (both variants). Per output tile `(b, h, q_start, v_rows)`:

1. stage attention weights and locations/offsets (+ reference points for V2)
   for all `v_rows` rows into L1 arenas — one `noc.async_read` per page;
2. for each `(l, p)`: write the bf16 operands into column 0 of the geometry
   tiles and push them. No arithmetic — the reader moves bit patterns;
3. take `x0`, `y0` back from the SFPU, decode them with integer shifts, and
   form the four per-corner in-bounds flags;
4. when `D % 32 == 0`: reserve one block for all four corners and land each
   corner's `D`-wide stick straight in its slot of row `r`
   (`r * 4*D*2 + c * D*2`), zero the skipped slots while the reads are in
   flight, barrier, and push once. The reader copies no value data. Other `D`:
   for each corner, read the sticks into `value_scratch`, scatter them into
   `n_d_tiles` tile rows, zero the rows it skipped, and push. The stick is at
   page `(b*S + level_start[l] + cy*W_l + cx) * H + h` (canonical) or
   `b*S + ...` at byte offset `h*D*2` (packed).

The two readers differ only in which staged pair step 2 ships as the primary
and which as the secondary; compute additionally consumes the offset pipes on
V2. The primary pair is float32 when `reference_points` are float32.

**Compute** (shared), per sampling point, one point ahead of the reduction:

* `msda_geometry::point` — two `axis` windows on the SFPU produce `floor(px)`
  and `px - floor(px)` per axis; the floors go back to the reader, the
  fractions stay. Four `corner_weight` windows then form
  `attn * corner_coeff` — the scalar tile the reduction consumes. Out-of-bounds
  corners are *not* masked here: this kernel has no bounds information, which
  is why the reader zeroes the input rows it skipped (§6).
* for `D % 32 == 0`, one `tilize_block` per point turns the reader's row-major
  block into `4 * n_d_tiles` tiles in `tiled_input` on the unpacker, with L1
  accumulate off;
* the reduction — `4 * L * P` iterations of `mul_tiles_bcast<COL>(input,
  scalar)` packed into the output CB with `pack_reconfig_l1_acc(1)` after the
  first, so the accumulator lives in L1 on the owning core for the whole
  reduction. Nothing is written back to DRAM per sample.

**Writer** (shared). Waits on `n_d_tiles` accumulated tiles, gathers each query
row's `D` values across them into a stick, and writes it at
`page_id = b*Q + q`, `offset_bytes = h*D*2`. No SCA/BEV scatter logic, ever.

---

## 9. Constraints

| Property | Value | Why |
| --- | --- | --- |
| dtype | `bfloat16` (all tensors); `reference_points` may be `float32` | tile format of the FPU path; a float32 reference point stays float32 into the SFPU multiply, where a bf16 point in [0, 1] would move in steps of up to 0.8 px at `W = 200` |
| layout | `ROW_MAJOR` (all tensors) | reader/writer address by stick |
| memory | `INTERLEAVED` only | `TensorAccessor` page indexing |
| `D` | positive multiple of 16, and `D*2` a multiple of the device's buffer alignment when `H > 1` | a `D`-stick is scattered across `ceil(D/32)` tiles in 16-value face halves, and the writer's per-head byte offset `h*D*2` must be a legal NoC destination. `D = 16` gives a 32-B stride, which is fine where the alignment is 32 B but is rejected where it is 64 B; `D = 32` and up are unconstrained. Validated in the program factory, not assumed. |
| `L` | `1 <= L <= 8` | per-level geometry is held in a fixed reader array |
| `H_l`, `W_l` | `<= 256` | the SFPU floors the bilinear corner and hands it to the reader as bf16, which carries 8 significant bits. Past 256 an in-bounds corner index would round to a *different, still in-bounds* pixel — a silently wrong sample. Rejected in `derive_shapes`, pinned by `test_fused_msda_rejects_spatial_shape_beyond_bf16_exact_integers` |
| `Q`, `P`, `H`, `B` | any positive value | `Q % 32 != 0` handled by `v_rows` |
| `S` | must equal `sum_l H_l * W_l` | validated |
| padding | zeros | |

Known limitations, to revisit after profiling:

* Canonical rank-6 `sampling_locations` costs one 4-byte NoC read per sample
  point, and its DRAM pages are alignment-padded 8x. The packed rank-4 form
  (`(B, Q, H, L*P*2)`) is one read per `(b, q, h)` and no padding — prefer it.
  This is the "flatten the semantics into one channel dimension" idea from the
  design brief; both forms are accepted and tested for equality.
* Sharded inputs unsupported.
* `fp32` accumulate not exposed; the L1 accumulator runs at the pack format.
  `fp32_dest_acc_en` is on unconditionally, but that is about the geometry's
  destination register, not the reduction's accumulator.
* V2 does not accept pre-folded offsets (`2/[W,H]` baked into the Linear).
  Such callers use V1 with `locations_in_grid_space=True`, or keep their Linear
  unfolded; BEVFormer passes raw pixels (§11). A fused "pre-folded offsets" V2
  variant is future work.
* Packer L1 accumulation runs with `fp32_dest_acc_en`, a combination tt-metal
  lists as unsupported (#28800). Validated on Blackhole only.

---

## 10. Files

```
ttnn/cpp/ttnn/operations/experimental/fused_msda/
├── fused_msda.hpp / .cpp                       public API
├── fused_msda_nanobind.hpp / .cpp              Python binding
├── CMakeLists.txt / sources.cmake              build wiring
└── device/
    ├── fused_msda_device_operation.hpp / .cpp  attrs, validation, output spec
    ├── fused_msda_program_factory.cpp          CBs, kernels, work split, RT args
    └── kernels/
        ├── msda_tile_layout.hpp                tile face-row byte offsets
        ├── dataflow/fused_msda_reader_common.hpp
        ├── dataflow/reader_msda_v1.cpp
        ├── dataflow/reader_msda_v2.cpp
        ├── dataflow/writer_msda.cpp
        ├── compute/msda_geometry.hpp           sampling geometry on the SFPU
        └── compute/compute_msda.cpp

tests/ttnn/unit_tests/operations/experimental/test_fused_msda.py
models/experimental/bevformer/tests/pcc/test_fused_msda.py
models/experimental/bevformer/tests/perf/test_fused_msda_perf.py
```

## 11. BEVFormer integration

BEVFormer's `TTMSDeformableAttention`
(`models/experimental/bevformer/tt/tt_ms_deformable_attention.py`) computes its
core attention with `fused_msda_from_offsets` and `reference_mode="pillar"`.
Spatial cross-attention, temporal self-attention and the detection decoder build this
module. The decoder passes float32 reference points: on its 200x200 BEV map a bf16
point in [0, 1] moves in steps of up to 0.8 px, and each layer's refined points feed
the next.

It feeds the op the **packed** rank-4 forms. The `sampling_offsets` Linear emits
channels ordered `(head, level, point, xy)` and the joint softmax produces
`(bs, Q, heads, levels*points)`, so the reshape is a dimension split, not a
permute.

The module passes raw pixel offsets; the op's per-level constants account for
`scale_l`.

Before each call the module asks `can_use_fused_msda_from_offsets_l1_value`
(§2.4) and stages `value`, and the op's output with it, in L1 when they fit,
interleaved DRAM otherwise. The gather is bound by the DRAM endpoints' request
rate, so L1 is the fast path.

Numerics are pinned by
`models/experimental/bevformer/tests/pcc/test_fused_msda.py` plus
`test_ms_deformable_attention.py`, `test_spatial_cross_attention.py`,
`test_temporal_self_attention.py`, `test_encoder.py` and `test_decoder.py`. The float32 reference-point path is pinned by
`test_fused_msda_v2_float32_reference_points`.
