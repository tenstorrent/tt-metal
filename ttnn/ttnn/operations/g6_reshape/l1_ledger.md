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
| `cb_src_stage_m0` | 1 page of `stage_depth · align_up(2T·S·(F + A), 64)` B | the block's distinct source face-rows (`≤ 2T·S`, exactly `2T` in `tile_face_row`) + per-run phase pad `< A` | `{out_atom: spans → T, half: spans → 2, face_row_elem: spans → F, movers: one per mover}`; input preimage enters through `S` | input dtype (dataflow-only payload; never through DEST; no `fp32_dest_acc_en` exists) | mover 0 (NoC reads) | mover 0 (CPU loads) | `gather_block` → end of `permute_block`, every block | cannot share with `cb_dst_image_m0`: both live during `permute_block` (read one, write the other); cannot share with `cb_move_plan_m0`: plan is written during the same walk that issues reads into staging. Capacity > live set by the phase pad (DRAM read congruence) and by the `S` bound (element-run worst case) |
| `cb_dst_image_m0` | 1 page of `image_depth · align_up(2T·F, 64)` B | the block's output face-rows, exactly `2T·F` (less where `trs_halves = 1`) | `{out_atom: spans → T, half: spans → 2, face_row_elem: spans → F, movers: one per mover}` | input dtype (dataflow-only) | mover 0 (CPU stores) | mover 0 (NoC writes) | `permute_block` → `writes_flushed` at the next block's `permute_block` | cannot alias staging (concurrent in `permute_block`); cannot alias the plan (tile table read by `store_block` while the image is being written out); capacity = live set |
| `cb_move_plan_m0` | 1 page of `align_up(16·S·T + 16·block_atoms, 64)` B | copy entries (`2T` in `tile_face_row`, `≤ 2T·S` element-run) × 8 B + tile entries (`≤ block_atoms`: an atom never straddles an output tile, so each tile entry holds ≥ 1 whole atom of the block) × 16 B | `{out_atom: spans → T, half: spans → 2 (one entry per face-row), face_row_elem: streams (one entry per piece, not per element), movers: one per mover}` | UInt32 (raw words) | mover 0 | mover 0 | `gather_block` → `store_block`, every block | cannot share with staging (written while reads land) or image (tile table consumed while the image is written out); capacity > live set only by the `S` bound in element-run (S = 1 in `tile_face_row`, so equal) and by ragged atoms |
| `cb_src_stage_m1` | as m0 | as m0 | as m0 | as m0 | mover 1 | mover 1 | as m0 | cannot share with any m0 buffer: the two movers run concurrently on disjoint atom ranges |
| `cb_dst_image_m1` | as m0 | as m0 | as m0 | as m0 | mover 1 | mover 1 | as m0 | as above (concurrent with mover 0) |
| `cb_move_plan_m1` | as m0 | as m0 | as m0 | as m0 | mover 1 | mover 1 | as m0 | as above (concurrent with mover 0) |

### Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `e` | 2 (bf16) or 4 (fp32), Phase 0; 4 (int32 / uint32, Refinement 1) | SUPPORTED dtype |
| `F = 16·e` | 32 or 64 B | `e` |
| `A` | 64 B on Blackhole (32 on Wormhole) | `ttnn.get_dram_alignment()` (HAL) |
| `atom_rows = max(1, A/F)` | 1..2 | `A`, `F` |
| `S` | 1 if `W_in % 16 == 0 and W_out % 16 == 0`; 3 if `W_in ≥ 16`; `ceil(15/W_in) + 1 ≤ 16` otherwise | regime predicate + `W_in` |
| `block_atoms` | `min(max atoms over movers, floor(1347776 / (movers_per_core · bytes_per_atom)), MAX_BLOCK_ATOMS)`, then lowered one atom at a time until the `align_up`-rounded total fits | BB-05 cap (host); `MAX_BLOCK_ATOMS` knob (default none = coarsest fit) |
| `T = block_atoms · atom_rows` | as above × `atom_rows` | `block_atoms` |
| `stage_depth`, `image_depth` | 1 (Phase 0) | host knobs `STAGE_DEPTH`, `IMAGE_DEPTH` |
| `movers_per_core` | 2 (Phase 0) | host knob `MOVERS_PER_CORE` (1..2) |

### Footprint

`bytes_per_atom = atom_rows·(stage_depth·2·S·(F + A) + image_depth·2·F + 16·S) + 16` (per mover)

`L1 per core = movers_per_core · (stage_depth · align_up(2T·S·(F+A), 64) + image_depth · align_up(2T·F, 64) + align_up(16·S·T + 16·block_atoms, 64))` ≈ `movers_per_core · block_atoms · bytes_per_atom`

One host helper (`_slot_bytes`) produces the three per-mover terms; the BB-05 cap, the CB sizes and
`describe_plan()["l1_bytes_per_core"]` all derive from it.

| Term | Scales with |
|------|-------------|
| staging `2T·S·(F+A)` | `block_atoms`, `S` (regime / `W_in`), `stage_depth` |
| image `2T·F` | `block_atoms`, dtype, `image_depth` |
| plan `16·S·T + 16·block_atoms` | `block_atoms`, `S` |
| `× movers_per_core` | mover knob |

Worked values: `tile_face_row` bf16 on Blackhole (`S=1, F=32, A=64, atom_rows=2`): `bytes_per_atom = 2·(192 + 64 + 16) + 16 = 560` per mover → 1120 B per atom per core; the cap allows `block_atoms ≤ 1203`. Element-run fp32 with `W_in < 16` (`S=16, F=64, atom_rows=1`): `bytes_per_atom = 4096 + 128 + 256 + 16 = 4496` → cap `block_atoms ≤ 149`. Production bf16 maximum: the largest `G` is 2304 atoms (`(1,1,16,4608)→(16,1,4608)`), ≤ 11 atoms per mover at `P = 110` → 12416 B per core, two orders of magnitude under 1347776.
Host plan values (110-core Blackhole grid; checked against the golden run's graph-capture L1 peak, see `verification_report.md`): `(1,1,16,4608)→(16,1,4608)` 12416 B (`block_atoms = 11, T = 22`), `(16,1,5120)→(1,1,16,5120)` 6784 B (`block_atoms = 6, T = 12`), `(1,16,6,256)→(1,1,16,1536)` 2304 B, `(16,1,64)→(1,1,16,64)` 1152 B.

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

## rm_row_gather (Refinement 1: ROW_MAJOR, any of bf16 / fp32 / int32 / uint32)

Same three buffers per mover, same CB slots (0..5), same forcing mechanisms; only the geometry changes.
Atom = one `A`-byte chunk of an output page slot (`align(W_out·e, A)` bytes); `EA = A/e` elements.
`S_rm = min(EA, ceil((EA − 1)/W_in) + 1)` input pieces per atom (1 for `[1,W]→[W,1]`, 16 for int32 `W_in = 1`).
`G = RM_READ_GAP_BYTES` (default `A`, bounded to `[0, A]`). One host helper (`_rm_slot_bytes`) feeds the BB-05
cap (via the shared `_fit_block_atoms`), the CB sizes and `describe_plan()["l1_bytes_per_core"]`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_src_stage_m{0,1}` | 1 page of `stage_depth · align_up(block_atoms·S_rm·(2A + max(A, G)) + A, 64)` B | the block's A-aligned covering read runs: a new run stages `< 3A` per piece, an extension `≤ G + 2A`; `+A` slack so the kernel can start the slot DRAM-aligned | `{out_atom: spans → block_atoms, elem: streams (one piece = one input-row run), movers: one per mover}` | input dtype (dataflow-only) | mover m (NoC reads) | mover m (CPU loads) | gather → end of permute, every block | cannot alias image (both live in permute) or plan (written while reads land); capacity > live set by the A-rounding of runs and the gap read-through |
| `cb_dst_image_m{0,1}` | 1 page of `image_depth · align_up(block_atoms·A + A, 64)` B | exactly `block_atoms` A-byte slot chunks in DRAM byte order (valid bytes + the slot's alignment padding, written as-is) + `A` alignment slack | `{out_atom: spans → block_atoms, movers: one per mover}` | input dtype | mover m (CPU stores) | mover m (NoC writes, one per (block, bank)) | permute → writes_flushed at the next block's permute | cannot alias staging (concurrent in permute); capacity = live set + slack |
| `cb_move_plan_m{0,1}` | 1 page of `align_up(8·S_rm·block_atoms, 64)` B | one copy entry per piece; no tile table (the store re-walks the bank-affine cursor) | `{out_atom: spans → block_atoms, elem: one entry per piece}` | UInt32 | mover m | mover m | gather → permute | cannot share with staging (written while reads land) |

Worked values (Blackhole, `A = 64`, 110 cores, `G = A`): `[1,4096]→[4096,1]` int32: 4096 atoms, ≤ 19 per mover,
`S_rm = 1` → 17920 B per core. `[1,3844]→[3844,1]`: 17024 B. `[8,1]→[8]` int32 (`S_rm = 16`, one atom): 6784 B.
`[4096,1]→[1,4096]` int32: 13312 B. All ≥ 75× under 1347776.

Data-movement budget (rm_row_gather): input — each needed input byte crosses DRAM once per covering run,
plus `< 2A` of rounding per run and up to `G` per read-through gap (for `[1,W]→[W,1]` a mover reads its
strided elements as ONE run: `≈ 8·e·atoms` bytes, i.e. every input byte is read once across the NB movers
sharing a bank stride); output — every output slot byte (valid + alignment padding) crosses DRAM exactly
once, one write per (block, bank). Cross-core traffic 0.
