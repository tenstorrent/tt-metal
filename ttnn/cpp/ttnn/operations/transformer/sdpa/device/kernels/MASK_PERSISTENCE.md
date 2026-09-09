# Persistent per-brick mask block in the neighborhood-SDPA reader

Branch `mask-persistence` (off `na-integration` 26e9c99), 2026-09-04. Commits `f548b5586d6`
(interior-only persistence + test) and `125a6656f5a` (signature-keyed persistence).

## Result

1080p DiffVAE stage-5 decode, 145 frames, 4x8 mesh, `bricked_sp_w_sharded`, the shipped
`DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1 DIFFVAE_NA_PER_BRICK_MASK=1`
(`test_decode_wsp_timing -k s34x60`, timed pass of the decode tree):

| | before | after |
|---|---:|---:|
| stage-5 attention (8 blocks) | 8055 ms | 3377 ms |
| decode | 12736 ms | 7883 ms |

Output is bit-identical: the same mask tiles are reused, never produced differently. Per-core
counters (DPRINT build): 95.9% of chunks skip all mask work, 4.1% rewrite the block once, none
generate tile-by-tile.

## The problem

Per work item (one 2-brick query chunk) the reader hands the compute kernel 2 x 168 = 336 mask
tiles (one per query brick per gathered key brick). It used to produce every one, every work item:
`classify_brick`, then a 2 KB DMA from the uploaded relative table (mixed tiles of an unclamped
brick), a 512-word memset (fully visible / fully masked), or `fill_mask_tile` (bricks whose window
clamps at a volume edge). Probes on the production decode: writing constant tiles instead
(`DIFFVAE_NA_MASK_MEMSET_ONLY=1`, wrong output) took attention from 8055 to 3230 ms; skipping the
K/V loads (`DIFFVAE_NA_SKIP_KV=1`) saved ~160 ms. So ~4.8 s of the 8 s was mask production, and the
flash compute itself was ~3.2 s.

## The observation

A mask tile is pure geometry. For two chunks, tile (row r, slot g) is the same whenever, on every
axis, (a) the gather origin sits at the same brick offset from the chunk, and (b) each brick's
window is shifted by the same amount at both ends of the brick -- 0 when the window is not clamped;
when it clamps at 0 or at `volume - window` the shift pins the brick's absolute position, and the
shifts of the sites in between follow. Interior chunks are the all-zero case; every T-edge chunk
along one W row shares another; and so on. Work items are assigned to cores W-fastest, so a core
sees long runs of chunks with the same signature.

## The mechanism

- `cb_mask` is sized to a WHOLE work item (`mask_tiles_per_kv_chunk * kv_chunk_count` pages) when
  the block fits `kernel_args::MAX_PERSISTENT_MASK_TILES` (512 tiles = 1 MB; production is 336, the
  unit tests 392-400, a 4-brick chunk would not fit and falls back to the double-buffered CB that is
  rewritten every chunk). The factory and the reader evaluate the same expression against that
  constant; there is deliberately no compile arg for the mode (adding one shifts the reader's
  TensorAccessorArgs chain). `cb_push_back` wraps the write pointer to base exactly when it reaches
  the limit and the CB is an exact multiple of the per-chunk push, so every work item writes the
  same L1 addresses.
- Per chunk the reader computes `per_brick_block_signature` (gather-origin offset per axis + per
  brick, per axis, the two window shifts + ghost flags; 1 + 2*bricks words) and compares it with the
  signature of the block resident in `cb_mask`. Equal: no mask work at all (K/V reads, barrier,
  pushes only). Different: write the block as before (table DMA / fill / memset per tile) and record
  the new signature. A brick beyond the resident tensor (the T-overhang of the last chunk) is a flag
  in the signature -- its rows are generated open and its keys lie inside the resident tensor
  because the planner clamps the gather. A brick below the volume (a low-edge halo brick) is not
  persistable; query bricks are never in the halo, so this never fires in practice.
- The chunk==stride path keeps its original, table-based `use_interior_table` /
  `mask_pages_hold_table` logic unchanged.
- Probes: `DIFFVAE_NA_MASK_MEMSET_ONLY=1` disables persistence (so it stays a floor on the write
  cost); `DIFFVAE_NA_SKIP_KV=1` and `DIFFVAE_NA_TABLE_ALWAYS=1` behave as before. With
  `TT_METAL_DPRINT_CORES=all` the reader prints per-core `mp items= skip= refill= gen=` counters at
  the end of its work; the factory logs the mask-CB decision (`neighborhood sdpa mask CB: ...`) at
  program creation.

## Why the first version saved nothing (worth remembering)

The first cut persisted only chunks whose bricks are all unclamped and whose gather origin is
canonical. On device it skipped 72% of production chunks -- and attention did not move (8131 ms
against 8055 ms before). That is not a contradiction; it is how the work is laid out.

**How work is distributed.** The stage-5 op has ~20,400 chunks per shard per block over ~130 cores,
handed out as contiguous ranges of 160-180 consecutive chunk indices. Chunk indices run row-major
over (T, H, W) with W fastest, so a core's range is a few consecutive W rows inside ONE T slice.
Whether a core's chunks are interior or edge is decided almost entirely by which T slice it holds.

**The edge slices are whole cores.** With a window reaching 5 sites and 2-site bricks in T, the
first and last 3 T-bricks clamp, so 4 of the 20 T-chunks are edge chunks: 20% of the chunks, but
concentrated on cores that own nothing else. The per-core counters (DPRINT build) showed it:

| core-instances (of 122,860) | items | skipped | generated |
|---:|---:|---:|---:|
| ~57,000 | 161-179 | all but 1 | 0 |
| ~23,500 | 161-179 | 0 | all |
| the rest | mixed | mostly | some |

**Why the total did not move.** An op finishes when its last core finishes. Before the change every
core generated 336 tiles per work item, so all cores were equally loaded. After it the interior
cores dropped to almost no mask work and idled, while the edge cores still generated 336 tiles per
work item exactly as before. The slowest core was unchanged, so the op was unchanged: skipping 72%
of the work removed 0% of the critical path. The probes agree -- `DIFFVAE_NA_TABLE_ALWAYS=1`, which
only changes edge bricks (table DMA instead of `fill_mask_tile`), took attention to 4563 ms
without touching an interior chunk, and the memset probe's 3230 ms is the floor once every core,
edge included, does only cheap writes.

**The T-overhang, the same lesson once more.** The signature version's first run landed at 5404 ms,
not the floor. The counters showed 2.5% of cores still generating every item: the cores owning the
LAST T slice. The volume is 39 T-bricks and the chunk is 2, so the 20th chunk's second brick lies
beyond the volume -- a ghost brick -- and the signature refused to persist any chunk containing one.
A tiny fraction of chunks, again concentrated on whole cores, again the critical path. Folding a
ghost flag into the signature (the ghost brick's rows are generated open, and its keys lie inside
the resident tensor because the planner clamps the gather) put those cores in skip mode too and
gave the final 3377 ms.

The general point: a fraction-of-work-saved number means nothing for a parallel op unless the
saving reaches the slowest cores. Count per core (`TT_METAL_DPRINT_CORES=all` and the reader's
counters), not in aggregate.

Two other traps on the way: `gather_is_canonical` can never pass for a multi-brick chunk (its
extent test assumes a single-brick gather; the origin-only `gather_origin_at_span_low` was split
out for that), and the unit tests' blocks were just over the original 384-tile budget, so the
persistence path was not being exercised by the tests meant to cover it.

## Tests

`models/tt_dit/tests/unit/test_neighborhood_sdpa.py`:
- `test_interior_table_per_brick_persistence` (new): the table test body under the forced (2,1,1)
  chunk on T-deep volumes ((24,24,24) for the (2,4,4) brick, (40,24,24) for (8,2,2)), unsharded and
  W-sharded with a negative origin, so every core sees edge -> interior -> edge sequences and the
  ghost T-overhang chunk. 4/4.
- `test_interior_table_matches_generated_masks` under the forced chunk 4/4; generation path
  (`test_matches_torch_reference`, forced chunk + per-brick) 12/12; default suite with no env 28/28
  (chunk==stride path unchanged).

## Remaining lever

The 4.1% of chunks that refill are on cores whose runs cross W-edge chunks (each W-edge brick has
its own signature: up to 25 rewrites of 336 tiles per core, ~7 per W row). A second resident
block, or ordering work so a core's run stays inside one W-regime, would remove those. The banded
flash (`BANDED_FLASH_*.md`, branch `banded-flash`) is orthogonal: it targets the ~3.2 s compute side
and measured slower than the default there.
