# fold_store_faces (Perf 20)

The idea: store only faces 0 and 2 of an output tile whose real lanes all sit in columns [0, 16). Faces 1 and 3 are
tile padding. This covers every tile of the C_out fold (C_out <= 4) and, more generally, the last N tile wherever
`1 <= C_out % 32 <= 16`. That halves the output bytes of conv_out (P3 idx 35, P9 idx 33).

- `op_base/`: the real op's working tree at the round start, including the uncommitted `l1_small_bytes` /
  `prepare_conv2d_weights_host` edits.
- `op_copy/`: the candidate.
  - New `kernels/conv2d_3x3_left_faces.hpp` (`LeftFaces<tile_bytes, bfp>::write`) does two runs per tile:
    - bf16: `[0, 512)` and `[1024, 1536)`.
    - bfp8_b: `[0, 320)` (all exponents plus face 0's mantissas) and `[576, 832)`.
  - The writer uses it in the fold store, and in the interleaved K-blocked store for N tile `tensor_n_tiles - 1`.
  - `RoutedStore::write_left_faces` is the STORE_ROUTE version.
  - The descriptor sets `OUT_LEFT_FACES` / `OUT_LEFT_FACES_BFP` / `OUT_TILE_BYTES` when the output is not sharded
    and `1 <= C_out % 32 <= 16`.
- `integration.patch`: `diff -ruN op_base op_copy`. Apply it with `patch -p1` in `ttnn/ttnn/operations/conv2d_3x3/`
  (the dry run is clean).
- `test_fsf.py`, `run.sh`:
  `run.sh <label> <cases> base,f2 [FS_ORDER=alt FS_ITERS=3 FS_PAD=1 FS_POISON=1 FS_DTYPE=bf8]`.
  - `FS_PAD=1` takes a census of the padding lanes from the padded `to_torch`.
  - `FS_POISON=1` first allocates and frees a NaN tensor of the output's size. The output then reuses that address
    (`poison_same_addr`), so it starts as NaN.
  - Results go to `runs/<label>.{txt,jsonl}`.
  - Case keys:
    - `<n>`: FLUX bh2x4 idx (ROW_MAJOR).
    - `t<n>`: the same case on TILE input.
    - `x2`: the single-device TILE 1024x1024 128->3.
    - `r7`..`r10`: off-fold regimes (C_out 48 / 8, RM and TILE).

## Results (Blackhole p150b, 110 cores, device kernel us; logical output bit-identical to op_base in every row)

| case | base | f2 (two runs) | delta | runs |
|---|---|---|---|---|
| P3 idx 35, 1026x2048 128->3 (fold) | 1824 / 1854 | 1698 / 1657 | -7% / -11% | m1, fin |
| P9 idx 33, 514x1024 128->3 (fold) | 523 / 527 / 530 | 497 / 498 / 495 | -5.7% (median) | p9pad, m1, fin |
| x2, TILE 1024x1024 128->3 | 1297 (median of 3) | 1238 (median of 3) | -4.6% | x2r (swA: 1333 / 1222) |
| t35, TILE 1026x2048 128->3 | 5042 (median of 3) | 4836 (median of 3) | -4.1% | x2r (swA: 5034 / 4930) |
| t33, TILE 514x1024 128->3 | 741 | 730 | -1.5% (flat) | swA |
| r9, RM 514x1024 32->8 (routed store) | 286 / 283 | 256 / 258 | -9.5% | swB, fin |
| r7, RM 258x512 128->48 | 371 | 373 | flat | swB |
| r8, RM 514x1024 128->8 | 579 | 577 | flat | swB |
| r10, TILE 258x512 128->48 | 363 | 364 | flat | swB |
| bf8_b x2 / t33 / r10 | 1053 / 586 / 272 | 1051 / 587 / 273 | flat | bf8 |
| P4 idx 36 (not reached: C_out 128; routed-store refactor) | 544 | 545 | flat | swA |

Menu, P3 / P9:
- f2, two 512 B runs: 1698 / 498.
- f3, one run through face 2 (`[0, 1536)`, 75% of the tile): 1721 / 498. It is flat against f2 and was measured with
  an earlier op_copy revision that had a mode knob.
- Faces 0 and 2 of two tiles are never adjacent in DRAM. Interleaved page p sits next to page p + 8 in its bank, with
  face 1 or 3 in between, so no batching across tiles is possible.

## Padding semantics

- Today (op_base), every padding lane of the output reads exactly 0. This holds on P3 and P9 (FS_POISON census,
  `runs/fin.jsonl`, `runs/p9pad.jsonl`) and on the non-fold paths.
  - Fold: the final pass packs with edge masks. Faces 0 and 2 keep lanes [0, C_out), and faces 1 and 3 get mask 0.
  - Non-fold: the prepared weight and bias columns past C_out are zero.
- With the change, lanes [C_out, 16) of faces 0 and 2 are still written as 0. Faces 1 and 3 keep what the buffer held,
  which reads as NaN after the poison.
  - On bfp8_b, the exponents are written and the face 1 / 3 mantissas are stale (2^-127). A bfp8 face row shares one
    exponent, so stale bytes never share an exponent with an output lane.
- Nothing in the repo reads that padding:
  - tt_dit FLUX.2 VAE decoder conv_out goes through `Conv2d._forward_conv2d_3x3`. Then `ttnn.reshape` to
    (1, H, W, 3) (a view, no fill), `_all_gather_hw` (whole pages), and `fast_device_to_host(pre_transfer_fn=float_to_uint8)`
    (element-wise add / multiply / clamp, then untilize to the logical width and typecast).
  - ttnn leaves tile padding unspecified. Reductions, prod, topk, layernorm pre-all-gather, sort, slice and pad call
    `ttnn.fill_implicit_tile_padding` themselves, and `ttnn.reshape` always fills block-float padding.
  - This op does not trust its own input's padding lanes either (op_requirements.md: lanes past C_in may hold
    NaN / stale bytes).

## Domain

- Applies to every program with a DRAM-interleaved output and `1 <= C_out % 32 <= 16`: the fold, the K-blocked walk
  on RM and TILE input, the routed store, and bf16 and bfp8_b outputs.
- Of the 17 focus cases, only P3 and P9 are reached. The others have C_out % 32 == 0, so their kernels are unchanged
  apart from the routed store's `write0` refactor (P4 flat, bit-identical).
- Not reached:
  - Sharded outputs: the define is off, and a resident output is packed in place.
  - Panoptic: every C_out is a multiple of 32.
- No exceptions.
- L1 and blocking are unchanged; the blocking records are identical between base and f2.
- Raw NoC (ergonomics): sub-page runs use `TensorAccessor::get_noc_addr(page, offset)` plus a sized
  `noc_async_write`, because `noc_async_write_page` moves only whole pages.
