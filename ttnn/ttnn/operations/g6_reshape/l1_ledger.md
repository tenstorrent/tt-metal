# L1 Ledger: g6_reshape

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Symbols: `op_design.md` → Notation.
Block axes (from the Blocking Model): `out_atom` (folded `out_tile_col` × `out_row`, extent `block_atoms`,
i.e. `T = block_atoms·atom_rows` TRS), `half` (2), `face_row_elem` (16 elements = `F` bytes), `movers` (2 per core).

Inventory, before any sizing: three buffers per mover, each forced by a mechanism, none removable without
changing what the op can express.

- `cb_src_stage`: DRAM reads must land at an address ≡ source (mod 64), so a face-row cannot be read
  straight into its output position (odd/even row parity differs); something must hold source geometry.
- `cb_dst_image`: output runs are contiguous in output byte order, which the staging order is not; the
  element-run regime cannot write sub-face-row pieces to DRAM at all (16-B write congruence). Dropping it
  for `tile_face_row` only is the "store realization" perf lamp, not a default.
- `cb_move_plan`: copies must wait for the read barrier, the walk that discovers them runs before it;
  recording the walk (8 B per face-row) is cheaper than re-walking it.

No budget predicate is introduced beyond the BB-05 cap on `block_atoms`; every golden INPUTS shape is one
block per mover at a few KB per core.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_src_stage_m0` | 1 page of `stage_depth · 2T·S·(F + A)` B | the block's distinct source face-rows (`≤ 2T·S`, exactly `2T` in `tile_face_row`) + per-run phase pad `< A` | `{out_atom: spans → T, half: spans → 2, face_row_elem: spans → F, movers: one per mover}`; input preimage enters through `S` | input dtype (dataflow-only payload; never through DEST; no `fp32_dest_acc_en` exists) | mover 0 (NoC reads) | mover 0 (CPU loads) | `gather_block` → end of `permute_block`, every block | cannot share with `cb_dst_image_m0`: both live during `permute_block` (read one, write the other); cannot share with `cb_move_plan_m0`: plan is written during the same walk that issues reads into staging. Capacity > live set by the phase pad (DRAM read congruence) and by the `S` bound (element-run worst case) |
| `cb_dst_image_m0` | 1 page of `image_depth · 2T·F` B | the block's output face-rows, exactly `2T·F` (less where `trs_halves = 1`) | `{out_atom: spans → T, half: spans → 2, face_row_elem: spans → F, movers: one per mover}` | input dtype (dataflow-only) | mover 0 (CPU stores) | mover 0 (NoC writes) | `permute_block` → `writes_flushed` at the next block's `permute_block` | cannot alias staging (concurrent in `permute_block`); cannot alias the plan (tile table read by `store_block` while the image is being written out); capacity = live set |
| `cb_move_plan_m0` | 1 page of `16·S·T + 16·T` B | copy entries (`2T` in `tile_face_row`, `≤ 2T·S` element-run) × 8 B + tile entries (`≤ T`) × 16 B | `{out_atom: spans → T, half: spans → 2 (one entry per face-row), face_row_elem: streams (one entry per piece, not per element), movers: one per mover}` | UInt32 (raw words) | mover 0 | mover 0 | `gather_block` → `store_block`, every block | cannot share with staging (written while reads land) or image (tile table consumed while the image is written out); capacity > live set only by the `S` bound in `tile_face_row` (S = 1 there, so equal) |
| `cb_src_stage_m1` | as m0 | as m0 | as m0 | as m0 | mover 1 | mover 1 | as m0 | cannot share with any m0 buffer: the two movers run concurrently on disjoint atom ranges |
| `cb_dst_image_m1` | as m0 | as m0 | as m0 | as m0 | mover 1 | mover 1 | as m0 | as above (concurrent with mover 0) |
| `cb_move_plan_m1` | as m0 | as m0 | as m0 | as m0 | mover 1 | mover 1 | as m0 | as above (concurrent with mover 0) |

### Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `e` | 2 (bf16, Phase 0); 4 (fp32 / int32 / uint32, refinements) | SUPPORTED dtype |
| `F = 16·e` | 32 or 64 B | `e` |
| `A` | 64 B on Blackhole (32 on Wormhole) | `ttnn.get_dram_alignment()` (HAL) |
| `atom_rows = max(1, A/F)` | 1..2 | `A`, `F` |
| `S` | 1 if `W_in % 16 == 0 and W_out % 16 == 0`; 3 if `W_in ≥ 16`; `ceil(15/W_in) + 1 ≤ 16` otherwise | regime predicate + `W_in` |
| `block_atoms` | `min(max atoms over movers, floor(1347776 / (movers_per_core · atom_rows · bytes_per_trs)))` | BB-05 cap (host) |
| `T = block_atoms · atom_rows` | as above × `atom_rows` | `block_atoms` |
| `stage_depth`, `image_depth` | 1 (Phase 0) | host constants |
| `movers_per_core` | 2 (Phase 0) | host constant |

### Footprint

`bytes_per_trs = stage_depth·2·S·(F + A) + image_depth·2·F + 16·S + 16` (per mover)

`L1 per core = movers_per_core · T · bytes_per_trs`

| Term | Scales with |
|------|-------------|
| staging `2T·S·(F+A)` | `block_atoms`, `S` (regime / `W_in`), `stage_depth` |
| image `2T·F` | `block_atoms`, dtype, `image_depth` |
| plan `16·S·T + 16·T` | `block_atoms`, `S` |
| `× movers_per_core` | mover knob |

Worked values: `tile_face_row` bf16 on Blackhole (`S=1, F=32, A=64`): `bytes_per_trs = 192 + 64 + 32 = 288` → `576·T` B per core; the cap allows `T ≤ 2339`. Element-run fp32 with `W_in < 16` (`S=16, F=64`): `bytes_per_trs = 4096 + 128 + 272 = 4496` → cap `T ≤ 149`. Production bf16 maximum: the largest `G` is 2304 atoms (`(1,1,16,4608)→(16,1,4608)`), so `T ≤ 2·ceil(ceil(2304/P)/2)`; at `P ≥ 64` that is `T ≤ 36` → ≤ 21 KB per core, two orders of magnitude under 1347776.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (`tile_face_row`); ≤ 2 for face-rows straddling two output face-rows in different half-passes or blocks (`tile_element_run`) | output-stationary gather: each logical input face-row is the preimage of exactly one output face-row when both W are multiples of 16; consecutive reuse (`last fetched face-row`) removes repeats inside one half-pass; tile padding rows are never addressed | 0 |
| output | 1 | each output atom is owned by exactly one mover and written once from its image; padding rows never written | 0 |

Totals per tier (`tile_face_row`): DRAM read `N·e`, DRAM write `N·e`, cross-core NoC 0, core-local L1:
`N·e` landed by the NoC, `N·e` CPU-copied staging → image, `N·e` sent by the NoC. `tile_element_run`
adds at most the padding columns of each row's last face-row on both sides plus ≤ 1× of the straddling
input face-rows. Transaction count, not bytes, is the binding cost at production sizes: one read per
coalesced input face-column run, one write per merged output face run.

> Cheapest-traffic split considered: output atoms, column-major fold (tile column outer, logical row
> inner) — 0 B delta on every tier versus any other non-rejected split, and the fewest transactions.
> Implemented: the same split. It is the cheapest; no split is deferred for traffic reasons.
