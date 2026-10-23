# L1 Ledger: distributed_rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes come from `op_design.md` → Blocking Model:
`tile_rows` (extent R), `width` (core slice S; streaming chunk C), `device` (G slots), `links` (per-link segment), `waves`
(count `num_waves`). Symbols: `xd = gd = id = min(2, num_waves)`, `sd = 2`, `T_in` / `T_out` / `T_g` = tile bytes of the
input / output / γ dtype (2048 bf16, 4096 fp32), `T32 = 4096`, `γ ∈ {0, 1}` (has_gamma), `r3 ∈ {0, 1}` (regime R3-X).

Page formats (Refinement 4): the formats marked † follow DEST through `dest_format(fp32_dest_acc_en)` in the program descriptor — `Float32`
with 32-bit DEST, `Float16_b` with 16-bit DEST (audit 2). The formats marked ‡ stay `Float32` whatever the flag (`STATS_FORMAT`): the reader
moves their column 0 / 128 B compact tails as 4-byte words (`col0_word`), and `cb_stats_slots` carries the fp32 wire partials (`UnpackToDestFp32` with 32-bit DEST only; one tile per tile-row, < 1 % of the footprint). `cb_x_tiles` / `cb_output` take the input dtype's page format (Float32 / Float16_b / Bfp8_b).

## Compute cores

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_tiles` | interleaved: `xd · R · WS` (`xd = num_waves` under the wave look-ahead, Refinement 3: the ring never wraps, so compute can square block w+1 at a fixed page offset behind block w); sharded: **the input shard itself** (`cb_descriptor_from_sharded_tensor`, zero-copy, no L1 of its own) | `R · WS` (one block; both passes read it) | `{tile_rows: spans → R, width: spans → WS, device: streams (one device), links: streams, waves: streams → depth xd}` | input dtype | reader | compute | load → end of γ pass (γ) / release (no γ) | none. Its live range covers both passes and overlaps every other block-scoped CB. The `xd − 1` extra block is the waves overlap mechanism (depth 1 in Phase 0) |
| `cb_reduce_scaler` | 1 (constant) | 1 | `{tile_rows: —, width: —, device: —, links: —, waves: —}` | bfloat16 | reader | compute | kernel | none: constant for the kernel, read by both reduces |
| `cb_sumsq_partial` † | `R` | `R` | `{tile_rows: spans → R, width: streams (accumulated over S), device: —, links: —, waves: streams}` | DEST (Float32 / Float16_b) | compute | compute | square → collapse | could alias `cb_inv_root` / `cb_inv_rms` (disjoint within a block). Not aliased: with `num_waves > 1`, block w+1's square overlaps block w's inv, so lifetimes overlap across waves; saving `R·T32` is below 1 % of the footprint |
| `cb_partial_col` (r3) ‡ | `R` | `R` | `{tile_rows: spans → R, width: — (collapsed), device: —, links: —, waves: streams}` | Float32 | compute | reader | collapse → gather send | not `cb_device_partial`: on the root both exist at once (own partial in flight to `cb_gather` while the combine writes `cb_device_partial`), and the producer / consumer pairs differ |
| `cb_gather` (r3) ‡ | `gd · G_w · Q` (`gd` = 1; 2 under the wave look-ahead, Refinement 3: block w+1's partials land while block w's are still being combined; block w+2's cannot, they need inv(w)) | `G_w · R` | `{tile_rows: spans → R, width: spans → G_w slices (one partial each), device: —, links: —, waves: streams → depth gd}` | Float32 | reader (root) | compute (root) | gather → core combine | allocated on every compute core because contributors write the root's CB address; on non-roots it is unused space. Not shared: it is a remote write target, so aliasing it with a local CB on a contributor would make a peer's write clobber live data |
| `cb_device_partial` ‡ | `Q` (rows a root owns per block: `R`, or `R / owners` under R3-RS) | `R` | `{tile_rows: spans → R, width: — (collapsed), device: — (this device), links: —, waves: streams}` | Float32 | compute | reader (root) | combine / collapse → publish | not packed straight into `cb_stats_slots`: that CB's producer is the reader (one producer per CB), and the device partial must be compacted to the wire format by the reader anyway |
| `cb_stats_slots` ‡ | `slot_depth · G · Q` (`slot_depth` = 1; 2 under the wave look-ahead, Refinement 3: block w+1's own slot is published while block w's region waits for its peers; every block pushes a whole `G · Q` region) | `G · R` | `{tile_rows: spans → R, width: —, device: spans → G slots, links: —, waves: streams}` | Float32 (`UnpackToDestFp32` iff 32-bit DEST) | reader (root) | compute (root) | publish → device sum | is its own staging: compaction (own slot) and expansion (peer slots) happen in place in its pages (ascending / descending word order), so no compact scratch CB exists |
| `cb_inv_root` (r3) ‡ | `Q` | `R` | `{tile_rows: spans → R, width: —, device: —, links: —, waves: streams}` | Float32 | compute (root) | reader (root) | finalize → distribute | not `cb_inv_rms`: one CB would then have two producers on the root (compute packs `cb_inv_root`, the reader pushes `cb_inv_rms` for every member including the root) |
| `cb_inv_rms` ‡ | `R` (depth 1, see Refinement 1) | `R` | `{tile_rows: spans → R, width: streams (read by every chunk), device: —, links: —, waves: streams → depth id}` | Float32 | R1-X: compute; R3-X: reader | compute | finalize / distribute → release | none: live through every chunk of the block. Depth `id` lets wave w+1's inv land early |
| `cb_gamma_tiles` (γ) | `WS` (resident); L1 fallback: `2 · C` (streamed per chunk) | `WS` | `{tile_rows: — (reuse-shared over rows), width: spans → WS, device: —, links: —, waves: — (resident)}` | γ dtype | writer | compute | kernel | none: resident for the kernel. Only row 0 of each page is filled (TILE γ: prefix read; ROW_MAJOR γ sticks: one whole-stick read into face 0 row 0, elements 16..31 then moved to face 1 row 0 in L1, Refinement 2); the full page is the FPU unpack unit |
| `cb_x_gamma` (γ) † | `R · C` | `R · C` | `{tile_rows: spans → R, width: spans → C, device: —, links: —, waves: streams}` | DEST (Float32 / Float16_b) | compute | compute | γ pass → inv pass | not in place in `cb_x_tiles`: the page formats differ (input dtype vs Float32; in place would round x·γ to bf16 before the inv multiply). Not packed into `cb_output`: it must exist **before** `inv` (the stall-shadow pass) and needs the fp32 width |
| `cb_output` | interleaved / foreign output placement: `sd · C`; output in the input's own shard placement: **the output shard itself** (zero-copy; compute packs strided into it) | `C` | `{tile_rows: streams (one tile-row per quantum), width: spans → C, device: —, links: —, waves: streams}` | output dtype | compute | writer | per chunk | none. `sd − 1` extra quantum = double buffering against the writer |

## Port cores

| Buffer | Capacity | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|--------|----------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| landing scratch (mesh L1 tensor, HEIGHT_SHARDED over the port cores; not a CB) | need `num_waves · G · seg_stride_max` bytes per port core; allocated as the smallest cached power-of-two bucket ≥ need (`_landing_bucket`, min 4 KiB), so the persistent set is ≤ ~log2(max/4 KiB) buffers, < 2× the largest | same (every (wave, device) slot is written once per call and read by the relay and the roots) | `{tile_rows: spans → rows of link l (128 B each), width: —, device: spans → G slots, links: one link per port, waves: spans → num_waves}` | fp32 words | roots (own slot, local NoC) + upstream peer port (relay) or every peer's same port (Ring direct sends, Refinement 5) (Fabric) | this port (relay only) + roots (NoC reads, per landed entry) | one call | none: it must have the **same address on every chip** (mesh allocation), and it outlives the program (cached across calls, ready-fenced). A program CB has neither property |
| fabric packet headers | ≤ 8 headers (pool, 144 B each) | ≤ 8 | `{all axes: —}` | — | port kernel | router | per call | the fabric's own pool (`PacketHeaderPool`), not op-allocated |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `R` | `1 ≤ R ≤ max_group_rows` | Work Distribution solve: largest R with `F(R) ≤ L1_budget`, capped by `BLOCK_ROWS_CAP`; `R_fit = 0` → `NotImplementedError` |
| `S` | `S ≤ Wt` (R1-X: = Wt) | the R1 / R3 selection; any Wt is admitted only if `F(1) ≤ L1_budget` |
| `C`, `nch`, `WS` | `nch = ceil(S / W_CHUNK_TILES)`, `C = ceil(S / nch) ≤ 64`, `WS = nch · C`, `S ≤ WS < S + nch` | host constant `W_CHUNK_TILES = 64` (`_chunking`). The `WS − V` pad tiles of each x row slot are zeroed once by the reader and never stored |
| `G` | `2 ≤ G ≤ mesh axis size` (≤ 8 on deployed meshes) | `validate()` (axis size ≥ 2) |
| `G_w` | `1 ≤ G_w ≤ R3_MAX_GROUP_CORES = 24` | `select_local` |
| `num_waves` | `ceil(max_group_rows / R)` | derived |
| `seg_stride_max` | `align_up(max_{w,l} rows(w, l) · 128, 64)`, rows(w, l) ≤ M_t | derived. M_t ≤ 128 over INPUTS, so ≤ 16 KiB |
| `L1_budget` | `largest_contiguous_bytes_free_per_bank` after the landing scratch is allocated | queried per call |

## Total per-core footprint

Compute core (bytes):

```
F(R) = R · per_row + fixed
per_row = xd · WS · T_in                                  # x block          (scales with R, S, waves)
        + T32 · (2 + slot_depth · G + id + r3 · (2 + gd · G_w))  # sumsq, device_partial, slots, inv, [partial_col, inv_root, gather]
        + γ · C · T32                                     # x·γ              (scales with R, C)
fixed   = 2048                                            # scaler
        + γ · WS · T_g                                    # resident γ slice (scales with WS)
        + sd · C · T_out                                  # output stream    (scales with C)
```

Terms that scale with each knob: R → everything in `per_row`; S (via WS) → x and γ; C → x·γ and output; G → slots;
`num_waves` → the depths `xd / gd / id` (1 → 2); the wave look-ahead (Refinement 3) → `xd = num_waves` (interleaved),
`gd = slot_depth = 2`; `G_w` → `cb_gather` (r3 only). (The root-only terms scale with `Q = R / owners`, not `R`.)

Worst INPUTS cell (fp32 x and γ, R1-X 4096×1024: R = 2, S = C = 32, G = 8): `per_row = 131072 + 4096·11 + 131072 = 307200`,
`F = 614400 + 2048 + 131072 + 262144 ≈ 1.01 MB`, below the ~1.3 MB Blackhole worker budget. Perf focus 640×1792 bf16
(R3-X, R = 2, S = C = 6, G = 4, G_w = 10, one wave): `per_row = 12288 + 4096·(2 + 4 + 1 + 2 + 10) + 24576 = 114688`,
`fixed = 2048 + 12288 + 24576 = 38912`, so `F = 268288 B ≈ 0.27 MB`.

Port core: `num_waves · G · seg_stride_max` (≤ 128 KiB at G = 8, M_t = 128, L = 1), rounded up to a power-of-two bucket.
The L1 allocator reserves this range on every core, so it is subtracted from `L1_budget` before the compute-core solve.
Verifier fix: the scratch cache used to keep one never-freed buffer per distinct size, which grew without bound over a
run of many shapes and shrank every later plan's budget (a 2×2048×4096 cell refused `block_rows = 1` mid-suite but fits
alone). Buckets bound it; a cached buffer freed externally is detected (`is_allocated()`) and replaced.

## Data-movement budget (chosen split, per device, per call)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 (read) | each core's block is resident in `cb_x_tiles` for both passes (`F(R)` fits; WAVES keeps it 1 at any size) | R3-X: none for x itself |
| out | 1 (write) | written once from `cb_output` | — |
| γ | `active_cores_sharing_a_slice` × slice prefixes (ROW_MAJOR sticks: one 64 B / 128 B stick per tile instead of a 544 B / 1088 B prefix) (R1-X: every active core reads all Wt prefixes; R3-X: each core its own slice) | per-core direct prefix reads (stepping stone; G-MC deferred) | — |
| per-row partial Σx² (stats) | 0 | lives in L1 end to end (ports' landing, roots' CBs) | **in-chip**: R3-X `(G_w − 1)` partial half-tiles (2 KiB) per tile-row to the root; each root `≤ 2 · R · 128 B` to ports and `(G − 1) · R · 128 B` back. **Fabric**: each device sends its own segment both ways (Ring) or along the line (Linear), and relays `floor(G/2) − 1` + `G − 2 − floor(G/2)` entries (Ring, relay) or up to `G − 2` (Linear); with Ring direct sends (Refinement 5, DIRECT_RING_SENDS) it sends its own segment once per peer (`G − 1` copies, multi-hop) and relays nothing: the same bytes leave the chip. It receives `(G − 1) · M_t · 128 B` |
| inv | 0 | computed on the root, consumed from L1 | R3-X: `(G_w − 1) · 128 B` per tile-row root → members (implementation: the compact wire form in the face-3 tail of the `cb_inv_rms` page, expanded to column 0 by each member's reader; was faces 0/2 = 2 KiB in the design — same CBs, 1/16 of the bytes) |

Totals at the perf focus (640×1792 bf16, QB2 1×4 ring, L = 2): DRAM `2 · 640 · 1792 · 2 B = 4.59 MB` (x + out) + γ prefixes
`100 cores · 6 tiles · 544 B ≈ 0.33 MB`. In-chip NoC ≈ `10 groups · 2 rows · 9 · 2 · 2 KiB ≈ 0.74 MB`. Fabric received per
device `3 · 20 · 128 B = 7.5 KiB`, latency-bound (2 hops to the farthest slot on the ring).

> Cheapest-traffic split considered: rows across cores with rows resident and a stats all-gather (R1-X): DRAM 1× x + 1× out, no in-chip combine traffic, `(G−1)·M_t·128 B` over Fabric. Implemented: R1-X **and** R3-X, selected by `select_local`. R3-X is not the cheapest in bytes: it adds `2·(G_w − 1)` half-tiles per tile-row on the NoC, with the same DRAM and Fabric bytes. It is built because R1-X leaves cores idle when `M_t < grid` (perf focus: 20 of 106 cores; decode: 1), an occupancy trade the regime table records, not a traffic one. Both keep x at one DRAM crossing.

## Implementation notes (ttnn-implementer)

- No CB was added, removed or resized relative to the inventory above. The compact 128 B stats / inv wire rows live in
  the face-3 tail (words 992..1023) of their own `cb_stats_slots` / `cb_inv_rms` pages: the column-0 consumers
  (lane-wise SFPU on faces 0/2, BroadcastDim::Col) never read face 3, so the same page holds both forms.
- Port cores: the landing scratch is exactly `num_waves · G · seg_stride_max` per port core, with
  `seg_stride_max = align_up(max_{w,l} seg_bytes(w, l), 64)`.

## Refinement 1 — sharded placements (HEIGHT_SHARDED / WIDTH_SHARDED, native zero-copy)

- **Shard-backed CBs.** `cb_x_tiles` is the input shard and `cb_output` the output shard (when the output carries the
  input's exact placement): `ttnn.cb_descriptor_from_sharded_tensor`, no L1 of their own, so `footprint()` drops the
  `xd · WS · T_in` and `sd · C · T_out` terms. The reader hands each block's resident `R_b · WS` pages to compute with
  no NoC traffic; compute squares the `V` valid tiles strided by `WS = S` (a ragged last shard's pad tiles are the
  caller's and never zeroed) and packs strided into the output shard (reserve / push once per block). A sharded input
  written to any other placement streams through `cb_output` and the writer's `TensorAccessor` as before.
- **Chunking.** Shard-backed: `C` = the largest divisor of `S` ≤ the chunk cap, `WS = S`, so every chunk is exactly `C`
  wide and every CB quantum is a whole number of `C` pages (pack / unpack addresses do not wrap inside a block).
- **R3-RS (WIDTH).** One row-group holds every row, so its rows are split over `owners` = the largest divisor of `R` ≤
  `G_w` (`R3_RS_OWNERS`). The root-only CBs scale with `Q = R / owners`, not `R`:
  `F(R) = R · per_row + Q · per_owned_row + fixed`, `per_row = xd·WS·T_in (interleaved) + T32·(1 + 1 + r3) + γ·C·T32`
  (sumsq, inv_rms, [partial_col], x·γ), `per_owned_row = T32 · (1 + G + r3 · (1 + G_w))` (device_partial, slots,
  [inv_root, gather]). Interleaved R3-X keeps one root (`owners = 1`, `Q = R`).
- **Depths.** `gd = id = 1` (was `min(2, num_waves)`): a member's partials of block b+1 exist only after its compute
  consumed inv(b), which an owner sends only after consuming gather(b), so a second slot never filled early. `xd`
  keeps `min(2, num_waves)` (the interleaved reader prefetches a block ahead).
- **L1 fallbacks** (`l1_fallbacks()`): when the default (resident γ, `C ≤ 64`) leaves no `R ≥ 1` — a 512 KB–1 MB
  shard pair leaves ~0.4 MB — γ is streamed (`2 · C` pages, `C` per chunk, re-read per block: row-0 prefixes only)
  and `C` halves until a block fits.
- **`cb_x_gamma` cycle padding.** A ragged block (`R_b < R`) pads `cb_x_gamma` to a whole `R · C` cycle per chunk
  (compute reserves / pushes, then waits / pops the pad), so every chunk's block starts at the ring base. This also
  closes a latent straddle on the interleaved path (ragged last wave with `nch ≥ 2` and γ).
- **Budget.** `L1_budget` = the free block at the BOTTOM of L1 (CBs end below the lowest occupied L1 address, and
  the allocator is bank-uniform), no longer `largest_contiguous_bytes_free_per_bank` (a hole above a low buffer
  over-reported CB space). The persistent landing scratch / global semaphores are moved up (`_settle_persistent`)
  when they are the lowest occupied L1 address and a higher free block can hold them: created under a live input
  shard, they otherwise capped every later program's CB region (measured 1.46 MB → 0.42 MB of CB space after one
  512 KB-shard call). Trade-off: moved under the then-live shards, they can split free L1 for a later borderline
  allocation (four fp32 32×4096 HEIGHT cells skip as INFEASIBLE_L1 at input creation in a sharded-only run).
- **Port cores** carry no CBs; when the shard grid covers logical row 0 they sit on the last free cores of the grid.

Data movement, sharded (per device, per call): x and out cross DRAM **0** times (resident shards). WIDTH adds the
in-chip all-to-all of partial half-tiles (`(G_w − 1) · M_t` × 2 KiB per core, spread over `owners` destinations) and
`(G_w − 1) · M_t · 128 B` of inv tails; HEIGHT adds nothing in-chip. Fabric bytes are unchanged.

## Refinement 3 — wave look-ahead (perf, 640×1792 ring)

- **Depths under the look-ahead** (`Plan.pipe`, chosen by the solve only where it fits): `cb_x_tiles` interleaved
  `num_waves · R · WS` (= the whole slice: no more L1 than one block holding every row, which is what the plan had
  before the block was split for the overlap), `cb_gather` `2 · G_w · Q`, `cb_stats_slots` `2 · G · Q`. Every other CB
  is unchanged. Waves forced by L1 (the solve's R < rows) never take the look-ahead on interleaved inputs: its x ring
  would not shrink with R. Sharded inputs keep the shard-backed x (no ring), so the look-ahead costs only the two
  regions there.
- **Data-movement budget**: unchanged (x read once, output written once, γ prefixes as before). The look-ahead
  reorders the same transfers; the exchange carries the same bytes in `num_waves` rounds.
- Refinement 5 (exchange transport): no CB added, resized or removed. Port cores carry `1 + (max mesh dim − 1)`
  persistent global semaphores (ready + one arrival counter per entry that can land on a port), up from 2 (4 B each,
  allocated with the landing scratch, `_new_sems`). Compute cores of single-root R3-X row-groups carry one more program
  semaphore (the inv multicast channel's Counter, allocated by `McastFamily.attach`).
