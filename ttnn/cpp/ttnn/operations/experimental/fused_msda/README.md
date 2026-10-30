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
    num_valid_queries=None,     # optional per-batch row count, see below
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

`num_valid_queries` serves rebatched callers whose tail rows are padding. It
takes one count per batch, each in `[0, Q]` with at least one nonzero; only
queries `[0, num_valid_queries[b])` are computed. The rows past it are never
written and hold undefined data, possibly NaN or Inf, so the caller must drop
them with a select such as `ttnn.where`, not a multiply by a 0/1 mask. The
counts are left out of the program hash and rewritten on every cache hit (§5),
so changing them per call does not recompile; a captured trace replays the
counts it was captured with.

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
single `(b, h)` pair. Queries and heads are independent, and the whole
`4 * L * P` reduction for a tile happens on the core that owns it — **no
cross-core reduction**.

The program and the tiles it runs are sized separately:

```text
slots per core = ceil(B * H * ceil(Q / 32) / cores)       fixed per program
active tiles   = sum_b H * ceil(q_end_b / 32)            per call
q_end_b        = num_valid_queries[b], or Q when not given
```

`split_work_to_cores` over `compute_with_storage_grid_size()` picks the cores
and the slot count from every query, so neither depends on `num_valid_queries`.
The active tiles are then spread evenly over those cores. Every core gets the
same number of slots, and `msda_runtime_args` zero-fills the unused ones, so
each core's runtime-arg vector has that fixed length.
`Kernel::set_runtime_args_count` only widens a core's reported count to the
longest vector in its kernel group; it does not write those slots. A shorter
vector on a cache hit would not match that count.

Each core takes a contiguous run of the tile list. The list is ordered `b`, then
query block, then head, so a core's consecutive tiles mostly share one query
block: `reference_points` is head-invariant, and the V2 reader stages it once
per query block rather than once per tile.

All runtime args, buffer addresses included, are plain values that
`ProgramFactory::override_runtime_arguments` rebuilds on every cache hit. That
is what lets `num_valid_queries` stay out of the program hash.

A partial block is supported wherever a batch's rows end: at `Q % 32`, or at
`num_valid_queries[b] % 32`. That tile carries `v_rows` below 32. Rows `>= v_rows`
are never gathered and never emitted: rows never mix, and the writer emits only
`v_rows` sticks.

`split_work_to_cores` is used rather than the spec's suggested
`core.x -> query blocks, core.y -> heads` because `B * H * ceil(Q/32)` rarely
factors onto the grid; a flat split load-balances the tail better and the
locality properties (reduction stays local) are identical.

---

## 6. Circular-buffer protocol

Three streams, not one. Per sampling point — one `(level, point)` pair — of an
output tile:

```text
reader  -> compute   geom_x, geom_y, attn_tile  operands; V2's offset in column 16
compute -> reader    corner                     the corner word, float32
reader  -> compute   4 x input_tiles            the gathered bilinear corners
                                                (one row-major block of all 4
                                                when D % 32 == 0, tilized by
                                                compute into tiled_input)
compute -> compute   4 x scalar_tile            attn * corner coefficient
```

The corner word (`kernels/msda_corner_word.hpp`) carries the NW corner's index
within its level and the four in-bounds flags, packed as an exact float32
integer. With the tiled geometry (below) the first two streams run once per
output block instead of once per point.

With the split gather (§8) a fourth, reader <-> writer, runs beside them:
`gather_mailbox` plus the `ready` / `done` semaphores, one point at a time. The
writer then also decodes the corner word at the reader's CB front, so the
reader pops the corner tile only after `done`.

| CB | Role | Pages | Page size |
| --- | --- | --- | --- |
| `c_0` `value_scratch` | reader-only L1 arena, one staged `D`-stick per row | 32 | `align(D*2)` |
| `c_1` `attn_scratch` | reader-only arena for attention weights | 32 (packed) or 32*L | `align(attn_stick)` |
| `c_2` `loc_scratch` | reader-only arena for sampling locations (V1) / offsets (V2) | 32 (packed) or 32*L*P | `align(loc_stick)` |
| `c_3` `input_tile` | reader → compute, the `V_corner` values: tiles, or a row-major `[NW \| NE \| SW \| SE]` block per point when `D % 32 == 0` | `8 * n_d_tiles` | 2048 |
| `c_4` `scalar_tile` | compute → compute, `attn * bilinear_coeff` per row | 8 | 2048 |
| `c_5` `output_scratch` | writer-only assembly of one head's `D`-stick | 1 | `align(D*2)` |
| `c_6` `ref_scratch` | **V2 only** — reference points arena, in the tensor's dtype. float32 builds a separate program, which keeps the point float32 through `geom_x` / `geom_y` | 32*R | `align(2*sizeof(ref))` (raw stick 4 B bf16, 8 B fp32) |
| `c_7` `geom_x`, `c_8` `geom_y` | reader → compute, the primary operand (location, or reference point for V2) in column 0 and, on V2, the raw sampling offset in column 16 (per-point geometry) | 4 | 2048, or 4096 when the reference point is float32 |
| `c_11` `attn_tile` | reader → compute, the attention weight (per-point geometry) | 4 | 2048 |
| `c_12` `corner` | compute → reader (and writer), the corner word: column 16 per point, or column 16 + j for point j of a tiled block | 3, or 2 tiled | 4096 (float32) |
| `c_9` `geom_rm` | reader → compute, one block's row-major `[offsets \| attn]` rows (tiled geometry) | 2 | 2048 |
| `c_10` `geom_operand` | compute scratch, `geom_rm` tilized (tiled geometry) | 2 | 2048 |
| `c_14` `ref_tile` | reader → compute, the block's reference points (tiled geometry) | 1 | 2048 |
| `c_15` `const` | reader → compute once, `M_off`, `M_ref`, `M_extent` and the 16 `E_j` (tiled geometry) | 19 | 2048 |
| `c_19` `weight` | compute → compute, a block's four weight tiles (tiled geometry) | 8 | 2048 |
| `c_16` `output_tile` | compute → writer, the accumulator | `2 * n_d_tiles` | 2048 |
| `c_17` `tiled_input` | compute → compute, the tilized corners (only when `D % 32 == 0`) | `8 * n_d_tiles` | 2048 |
| `c_18` `gather_mailbox` | reader → writer, one point's split-gather arguments (split gather only: `D % 32 == 0` on Blackhole) | 1 | 64 |

`n_d_tiles = ceil(D / 32)`. The tiled geometry replaces `c_1`, `c_2`, `c_7`,
`c_8` and `c_11` with `c_9`, `c_10`, `c_14`, `c_15` and `c_19`.
`derive_circular_buffer_plan` is the source of truth for the sizes, and
`circular_buffer_bytes_per_core` sums them for §2.4.

Deliberately **not** four separate `CB_VALUE_00/01/10/11` CBs and a separate
`CB_INTERP` / `CB_ATTN_WEIGHT`: the four corners are streamed **sequentially**
into the same `input_tile` CB, and `dx, dy` never leave the compute kernel's
destination registers. They are folded into the scalar the reduction is already
going to multiply by:

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
reader ships the operands as tiles (column 0, and on V2 the offset in column 16)
and gets the corner word back, with
the bounds tests and the in-level index already done, so the data-movement
RISCs are left with a subtract, a mask and a shift per row and one add per corner
for the page index.
Those tiles are bf16, except `geom_x` / `geom_y` when `reference_points` are
float32. The SFPU multiplies that point, and a bf16 tile
would already have moved it in steps of up to 2^-8. Those two CBs are also
unpacked straight into the 32-bit destination (`UnpackToDestFp32`):
`copy_tile` through SrcA would keep only 10 mantissa bits.

Both sides run **one sampling point ahead** of the work they feed — the reader
pushes point `j+1`'s operand tiles before it gathers point `j`, and the compute
kernel solves point `j+1` before reducing point `j`. Without that lookahead the
two ping-pong, each idle through the other's turn. Where the push sits depends
on the gather: with the reader-only gather it goes out before the reader waits
for point `j`'s corner word; with the split gather (§8) it goes out after point
`j` is posted to the writer, so the writer gathers while the reader builds the
tiles. That order cannot deadlock, because compute produces point `j`'s corner
word while solving `j`, before it waits for `j+1`'s tiles. The CB depths above
are what make the lookahead legal: `scalar_tile` and `input_tile` hold two
points' worth, the reader ↔ compute geometry pipes three to four.

The corner word crosses as **float32**, whose 24 significant bits hold it
exactly. `fp32_dest_acc_en` is required for the same reason one step earlier:
`px` reaches the feature map's extent and bf16's ulp at 200 is 1.0, so a 16-bit
destination would round `floor(px)` to the wrong integer on the larger levels
and collapse bilinear sampling to nearest-neighbour.

### Tiled geometry

On Blackhole, V2 with `L * P == 16`, at most 15 bf16 reference points and
packed offsets and weights hands the geometry over once per output block instead
of once per point (`kernels/msda_tiled_geometry.hpp` holds the layout). The
reader NoC-reads the block's offsets and attention rows into one row-major page
and rewrites the reference tile only when the query block changes. Compute
tilizes them and, in one window, solves `px | py` and the extents for all 16
points with matmuls against per-level constant matrices, then floors them and
builds the corner words and weights on the SFPU. It returns one corner tile per
block with point `j`'s word in column `16 + j`. Each
point's four scalar tiles are then moved to column 0 by a matmul against `E_j`.
The constants enter the FPU as bf16 matrix entries, so the factory falls back
to the per-point geometry when any is not bf16-exact, and the matmuls run at
HiFi4 because lower fidelities drop mantissa bits of SrcB, which holds the
data.

### Tile packing

32 queries are packed vertically into a tile; `D` spans `n_d_tiles` tiles side by
side. `msda_tile_layout.hpp` maps row `r` to its `(lo, hi)` face-half byte
offsets. The compute kernel uses `mul_tiles_bcast<COL>`, which reads only column
0 of the `TL`/`BL` faces of the scalar tile. A query occupies one SFPU lane of a
32x32 tile throughout: column 0, with column 16 (the same lane, one face to the
right) carrying a second operand: the V2 offset, or the corner word. The tiled
geometry puts point `j` of a block in column `j` and `16 + j`.

### Zero-fill contract

* **geometry tiles** (reader → compute, per-point geometry): column 0, and on
  V2 the offset in column 16 of `geom_x` / `geom_y`, are written for all 32 rows
  on every emission; rows `>= v_rows` get `0` in that tile's format. Every other
  column is zeroed once per CB slot at reader startup, so an uninitialised L1
  bit pattern never reaches the SFPU as a NaN.
* **input tile** (reader → compute; reader + writer with the split gather, each
  for its own rows): rows that are in range **and** in bounds hold the gathered
  value stick (per corner slot in the row-major block). On Blackhole every
  other row or slot keeps what the CB held: the reader zeroes the whole CB once
  at startup, and afterwards it only ever receives value sticks. Elsewhere the
  gathers also zero the out-of-bounds slots of rows `< v_rows`.
* **scalar tile** (compute → compute): column 0 for all 32 rows, from the
  fractions and the attention weight. With the tiled geometry, rows `>= v_rows`
  keep a previous block's operands, so their scalars are not zero. Rows never
  mix and the writer emits only rows `< v_rows`, so that is harmless.

Out-of-bounds corners are cancelled in compute: the SFPU zeroes the weight of
every corner whose bounds flags in the corner word are clear, and the gathers
skip those corners. On Blackhole they leave the slot unwritten, so that zero
weight is load-bearing: the skipped slot holds an earlier point's value stick.
It also cancels an inf or NaN stick, because Blackhole's FPU multiplies both by
zero to zero. That has not been measured on other architectures, so there the
gathers still zero the slot (`ZERO_SKIPPED_SLOTS` in `msda_gather_rows.hpp`).
A stale stick is itself a plausible sampled value, so the correlation survives
a missing mask — which is why
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
reader_msda_v2.cpp ─┘   (staging, gather, tile scatter)     (geometry, reduction)
                                 │                                                    │
                                 └──── msda_gather_rows.hpp (row-major gather) ───────┘
                                       rows [0, SPLIT_ROW) on the reader, the rest on the writer
```

**Reader** (both variants). Per output tile `(b, h, q_start, v_rows)`:

1. stage attention weights and locations/offsets (+ reference points for V2,
   unless the previous tile staged the same query block) for all `v_rows` rows
   into L1 arenas — one `noc.async_read` per page;
2. for each `(l, p)`: write the operands into column 0 (and the V2 offset into
   column 16) of the geometry tiles and push them (tiled geometry: one row-major page per block). No
   arithmetic — the reader moves bit patterns;
3. take the corner word back from compute and split it into the NW index and
   the four in-bounds flags (a subtract, a mask and a shift);
4. when `D % 32 == 0`: reserve one block for all four corners and land each
   corner's `D`-wide stick straight in its slot of row `r`
   (`r * 4*D*2 + c * D*2`), skip out-of-bounds corners and rows `>= v_rows`,
   barrier, and push once. The reader copies no value data; with the
   split gather it does this for rows `[0, SPLIT_ROW)` only and waits for the writer's
   rows before the push. Other `D`:
   for each corner, read the sticks into `value_scratch`, scatter them into
   `n_d_tiles` tile rows, skip the rest, and push. The stick is at
   page `(b*S + level_start[l] + cy*W_l + cx) * H + h` (canonical) or
   `b*S + ...` at byte offset `h*D*2` (packed).

The two readers differ only in which staged pair step 2 ships as the primary
and which as the secondary (column 16 of the geometry tiles on V2), and only V2
can take the tiled geometry. The primary pair is float32 when
`reference_points` are float32.

**Compute** (shared), per sampling point, one point ahead of the reduction:

* `msda_geometry::point` — one SFPU window floors `px` and `py`, packs the
  corner word for the gathers and forms `attn * corner_coeff` for the four
  corners — the scalar tiles the reduction consumes. With the tiled geometry
  `msda_geometry::block` does the same for a whole block (§6). The weight of
  every out-of-bounds corner is zeroed here, from the same flags the gathers
  use to skip it (§6).
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
With the split gather it first gathers rows `[SPLIT_ROW, 32)` of every point of
the block. It always receives the `value` accessor and address, and reads them
only then.

**Split gather** (`D % 32 == 0`, Blackhole). The reader's gather is bound by how
fast one RISC can decode corners and issue `D*2`-byte reads, and the writer RISC
has spare issue capacity and its own NoC. So both take part of every row-major
block: the reader rows `[0, SPLIT_ROW)`, the writer rows `[SPLIT_ROW, 32)`,
through `fused_msda_gather::gather_rows` (`msda_gather_rows.hpp`, which also
documents the handshake and how `SPLIT_ROW` was chosen).
Per point the reader posts the point's arguments to `gather_mailbox` and bumps
the `ready` semaphore, then (per-point geometry) pushes the next point's
geometry tiles while the writer gathers; the writer gathers its rows, barriers and bumps `done`; the
reader gathers its own rows and waits for `done` before it pops the corner tile
and pushes the block. The split row is measured, not derived: 16 / 16 with
`value` in L1 and 15 / 17 with `value` in DRAM. Both counters only grow, and
launch restarts them from their initial value.

Sharing one CB block between the reader and the writer follows conv2d's
`split_reader_cb_shared` mode, where the reader reserves the block, the writer
fills part of it and two semaphores hand it back. Here the hand-off is per
point rather than per block, because the writer needs each point's corner word
and level geometry: the reader posts them to a one-page mailbox and the
semaphores count points instead of flagging valid / invalid. Splitting by point
instead, with each RISC its own CBs as `grid_sample`'s split reader does, would
duplicate the input, corner and geometry pipes and make compute alternate
between them; the row split leaves compute unchanged and adds one 64 B CB.
Wormhole keeps the reader-only gather; the split has not been measured there.

---

## 9. Constraints

| Property | Value | Why |
| --- | --- | --- |
| dtype | `bfloat16` (all tensors); `reference_points` may be `float32` | tile format of the FPU path; a float32 reference point stays float32 into the SFPU multiply, where a bf16 point in [0, 1] would move in steps of up to 0.8 px at `W = 200` |
| layout | `ROW_MAJOR` (all tensors) | reader/writer address by stick |
| memory | `INTERLEAVED` only | `TensorAccessor` page indexing |
| `D` | positive multiple of 16, and `D*2` a multiple of the device's buffer alignment when `H > 1` | a `D`-stick is scattered across `ceil(D/32)` tiles in 16-value face halves, and the writer's per-head byte offset `h*D*2` must be a legal NoC destination. `D = 16` gives a 32-B stride, which is fine where the alignment is 32 B but is rejected where it is 64 B; `D = 32` and up are unconstrained. Validated in the program factory, not assumed. |
| `L` | `1 <= L <= 8` | per-level geometry is held in a fixed reader array |
| `H_l`, `W_l` | `<= 256` | the largest extent the op is tested at. The corner word's index field holds extents up to 361, and the tiled geometry falls back to the per-point path for an extent that is not bf16-exact. Rejected in `derive_shapes`, pinned by `test_fused_msda_rejects_spatial_shape_beyond_extent_limit` |
| `Q`, `P`, `H`, `B` | any positive value | `Q % 32 != 0` handled by `v_rows` |
| `num_valid_queries` | `None`, or `B` counts in `[0, Q]` with at least one nonzero | rows past a count are unwritten; the caller drops them with a select |
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
    ├── fused_msda_program_factory.cpp          CBs, kernels, work split, RT args (rebuilt on cache hits)
    └── kernels/
        ├── msda_tile_layout.hpp                tile face-row byte offsets
        ├── msda_corner_word.hpp                the corner word compute hands the gathers
        ├── msda_tiled_geometry.hpp             tiled geometry layout, shared with the factory
        ├── dataflow/fused_msda_reader_common.hpp
        ├── dataflow/reader_msda_v1.cpp
        ├── dataflow/reader_msda_v2.cpp
        ├── dataflow/writer_msda.cpp
        ├── dataflow/msda_gather_rows.hpp       row-range corner gather, shared by reader and writer
        ├── compute/msda_geometry.hpp           sampling geometry (SFPU; FPU matmuls when tiled)
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

Spatial cross-attention pads every camera to the longest camera's query count.
It passes each camera's valid count as `num_valid_queries`, so MSDA skips the
padding, and drops the unwritten rows with `ttnn.where(row_mask, ...)`. That
select has to stay a select: the dropped rows are undefined, not zero.

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
