# Operation Design: g6_reshape

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — two dataflow kernels, **no compute kernel** (no value enters an unpacker, DEST or a packer: bit-exact by construction) |
| Goal | Return `input_tensor` under shape `e`: every logical value moved bit for bit to the position with the same row-major linear index, laid out again in the input's own layout (TILE in Phase 0), in ONE device program, reading only valid input bytes and writing only valid output bytes. |
| Math | `out[j] = x[i]` whenever `lin(j; e) = lin(i; d)`. Page view (leading dims folded into rows, `g` = global row): `Y[g_o, c_o] = X[L div W_in, L mod W_in]` with `L = g_o·W_out + c_o` |
| References | `eval/golden_tests/g6_reshape/{feature_spec,helpers,axes}.py`; `eval/op_template.py`; `ttnn/ttnn/operations/_op_contract.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template,ttnn-cb-memory-fundamentals}.md`; `.claude/references/generic_op_template/`; `ttnn/ttnn/operations/examples/master.md` entries `noc_placement`, `double_buffer`, `trid_double_issue`, `tile_reorder`, `split_reader`, `bank_stagger`, `read_coalescing`; `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380`; `tt_metal/hw/inc/internal/debug/sanitize.h:319-341,745`. `docs/explorations/g6_reshape/` is absent in this tree; this design works from the spec. |

The op is a bijection between the valid **face-row segments** (≤ 16 consecutive logical elements of one tile row inside one face) of the two tensors. One tensor — the **tile side** — is partitioned across cores and moved with coalesced face runs at fixed intra-tile offsets; the other — the **map side** — is addressed segment by segment through the index map. Which tensor is the tile side is the regime choice (§Regimes): the side whose tiles hold more valid rows gets the coalesced transfers.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, DRAM interleaved, rank 1..4, every dim ≥ 1; Phase 0 SUPPORTED: `bfloat16` × `TILE_LAYOUT` | — | geometry → CT; buffer address → common RT |
| `shape` | `list[int]` or `tuple[int]`, positional | yes | length 1..4; entries integers (not bool), each ≥ 1 or exactly one `-1`; product after resolving `-1` = N = element count of `input_tensor` | — | resolved shape `e` → CT (via the geometry symbols) |

### Registry and host validation

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | `{"dtype": tag_dtype, "layout": tag_layout}`; `tag_dtype(inputs, axes)` returns `inputs[0]["dtype"]`, `tag_layout(inputs, axes)` returns `inputs[0]["layout"]` (scenario dict `{"input_shape","shape","dtype","layout"}`) |
| `SUPPORTED` (Phase 0) | `{"dtype": [ttnn.bfloat16], "layout": [ttnn.TILE_LAYOUT]}` |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, shape)` | first line of `g6_reshape`. Builds the scenario dict from the call (`input_tensor.shape/dtype/layout`, `shape`), applies `INPUT_TAGGERS` in order, raises `UnsupportedAxisValue` for an axis value outside `SUPPORTED`, then `ExcludedCell` for an `EXCLUSIONS` match. No INVALID check. |
| `_resolve_shape(input_tensor, shape)` (right after `validate`) | `ValueError` for: `shape` not a list/tuple; length outside 1..4; a non-integer or bool entry; an entry `0`; an entry `< -1`; more than one `-1`; `N % prod(other entries) != 0` with a `-1`; product ≠ N; input rank outside 1..4; input not on device or not DRAM interleaved; `N ≥ 2**32` (index-width cap, §Mechanism caps). Returns `e`. |
| Output allocation | `ttnn.allocate_tensor_on_device(ttnn.Shape(e), input_tensor.dtype, input_tensor.layout, device, input_tensor.memory_config())` — the ONLY device allocation. An identity reshape (`e == d`) still dispatches the program (the output is a new tensor). |
| Dispatch | `ttnn.generic_op([input_tensor, output], descriptor)` — one program per call; no `to_layout`/`tilize`/`pad`/`slice`/`reshape`/host read of values |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `d = (d_0..d_{n-1})`, n ∈ 1..4, every `d_i ≥ 1` |
| Dtype | Phase 0: `bfloat16`; TARGET adds `int32`, `uint32` (ROW_MAJOR) and `float32` (TILE) |
| Layout | Phase 0: `TILE_LAYOUT` (32×32 tile = four 16×16 faces, face-major, row-major in a face) |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `e` = `shape` with `-1` resolved |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (DRAM interleaved) |

### Geometry symbols (host-derived once per call signature; single source: the program-descriptor builder)

| Symbol | Definition (same rule for `_in` and `_out`) |
|--------|---------------------------------------------|
| `W` | last logical dim |
| `H` | second-to-last logical dim; `1` for rank 1 (a rank-1 TILE tensor has padded shape `[32, 32]`, checked on host) |
| `IMG` | product of the dims before the last two; `1` for rank ≤ 2 |
| `Wt`, `Ht` | `ceil(W/32)`, `ceil(H/32)` — per image; tile padding is per image |
| `num_tiles` | `IMG·Ht·Wt`; host asserts it equals `tensor.buffer_num_pages()` (RuntimeError otherwise) |
| tile id | `t = (img·Ht + tr)·Wt + tc` |
| `v(tr)`, `vc(tc)` | valid rows / columns of a tile: `min(32, H − 32·tr)`, `min(32, W − 32·tc)` |
| global row, linear index | `g = img·H + 32·tr + r`; `L = g·W + c` |
| `esize` | `input_tensor.element_size()` (2 bf16, 4 fp32/int32/uint32) |
| `face_row_bytes`, `face_bytes`, `tile_bytes` | `16·esize`, `256·esize`, `1024·esize` |
| `face_off(r, c)` | `((r div 16)·2 + c div 16)·face_bytes + (r mod 16)·face_row_bytes + (c mod 16)·esize`, for `r, c ∈ [0, 32)` |
| `granule_bytes` | `max(ttnn.get_dram_alignment(), face_row_bytes)` — 64 on Blackhole for every TARGET dtype (`NOC_DRAM_READ_ALIGNMENT_BYTES = 64`, `noc_parameters.h:379`) |
| `runs(T)` | coalesced face runs of a whole tensor: `IMG · (Σ_{tr<Ht} (1 + [v(tr) > 16])) · (Σ_{tc<Wt} (1 + [vc(tc) > 16]))` |
| `aligned` | `W_in mod 16 == 0 ∧ W_out mod 16 == 0` |

## Blocking Model

### Work units

| Term | Definition |
|------|-----------|
| **Tile side** `T_s` | the output in `gather_*` regimes, the input in `scatter_aligned`. Its tiles are partitioned; transfers on this side are face runs at identical intra-tile offsets in DRAM and in L1. |
| **Map side** `T_m` | the other tensor; addressed per segment through the index map. Never partitioned on its own — each core touches exactly the map-side bytes its tile-side segments name. |
| **Strip** (the work unit) | one valid row `r < v(tr)` of one tile `t` of `T_s`. Order: `t` ascending, then `r` ascending (tile-major). Count `U = IMG_s · H_s · Wt_s`. |
| **Segment** | strip × column half `fh ∈ {0, 1}` with `16·fh < vc(tc)`: columns `[32·tc + 16·fh, +min(16, W_s − 32·tc − 16·fh))`, one face row of face `2·(r div 16) + fh`. |
| **Piece** | the maximal sub-range of a segment that is contiguous on the map side (ends at a map-side face-row end or row end). `aligned` ⇒ every segment is exactly one whole 16-element face row on both sides (proof: segment start `L ≡ 0 mod 16`, `W_in, W_out ≡ 0 mod 16` ⇒ map-side column `≡ 0 mod 16` and ≥ 16 columns left in that row). Otherwise up to `P = min(16, ceil(16 / W_m) + 2)` pieces per segment. |
| Strip `s` → `(t, r)` (closed form, host and kernel) | `img = s div (H·Wt)`; `s1 = s mod (H·Wt)`; `tr = min(s1 div (32·Wt), Ht − 1)`; `s2 = s1 − tr·32·Wt`; `tc = s2 div v(tr)`; `r = s2 mod v(tr)`. Kernels walk incrementally from the start strip: `r += 1`; at `r == v(tr)`: `r = 0`, `tc += 1`; at `tc == Wt`: `tc = 0`, `tr += 1`; at `tr == Ht`: `tr = 0`, `img += 1`. |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `t_s` — tile-side tile sequence (`img`, `tr`, `tc` folded: contiguous in tile-id order and treated identically) | independent — no logical value depends on another (the map is a bijection, no value spans an axis) | `block_tiles` | `min(max_core_tiles, BLOCK_TILES_CAP)` = the whole per-core run on every golden shape (1–3 tiles) | `BLOCK_SLOT_BYTES_CAP` (module constant) → `BLOCK_TILES_CAP = max(1, BLOCK_SLOT_BYTES_CAP div tile_bytes)` → `block_tiles` (CT) | contiguous strip runs over `n_cores` (Work Distribution) | knob-turn |
| `r` — valid row inside a tile-side tile | independent — each row's segments read/write only their own linear range | `block_rows(t)` = rows of tile `t` inside the block | all valid rows `v(tr)` of every tile the block covers; fewer only in a core's first/last tile (a core boundary may fall inside a tile) | derived from the per-core strip range (RT) | split across cores at strip granularity — this is what lets the core count exceed the tile count (`[16,1,64]→[1,1,16,64]`: 2 output tiles, 32 strips) | knob-turn |
| `fh` — column half of a strip | independent — two disjoint 16-column segments | `segments_per_strip` | `1 + [vc(tc) > 16]` — whole, never split | derived from `W_s` (CT) | not assigned across cores: a strip's two halves stay on one core | none — a face row is the DRAM-contiguous atom of the tile format |
| `e` — element inside a segment | independent | `SEGMENT_ELEMS = 16` | 16 (whole face row) | fixed by the tile format | not split | none |
| `t_m`, `r_m`, `c_m` — map-side tiles / rows / columns | independent; **not reuse-shared** — every map-side element is named by exactly one segment, so no operand is read twice | none: the map-side footprint of a block is exactly the segments its strips name; there is no extent to choose | equals the image of the block's strips | — (no knob by construction) | follows tile-side ownership (owner-moves) | the other `aligned` regime is this model with the roles of the two tensors swapped |
| staged granules (intermediate data of `gather_*`) | independent — each is consumed only by the copies of its own core | `staging_granules` (flush window) | `min(STAGING_GRANULES_CAP, staged_pieces_bound)` = one flush per block on every golden shape | `STAGING_GRANULES_CAP` (module constant) | never leaves its core; no core collects, coordinates or hands out work — every core runs the identical role on its own run | knob-turn |

### Knob sources (single source of truth)

All knobs are module-level constants of `ttnn/ttnn/operations/g6_reshape/g6_reshape_program_descriptor.py`, read as **module globals at build time** (never copied into another module at import), and every one of them is part of the descriptor memo key. Every quantity below them is derived once on the host; kernels receive only derived values.

| Knob | Phase 0 value | Derived quantities |
|------|---------------|--------------------|
| `MIN_STRIPS_PER_CORE` | 1 (full occupancy) | `n_cores = min(grid_cores, ceil(U / MIN_STRIPS_PER_CORE))` |
| `BLOCK_SLOT_BYTES_CAP` | 262144 | `BLOCK_TILES_CAP`, `block_tiles`, `num_blocks_this_core = ceil(core_tiles / block_tiles)`, `max_blocks_per_core`, `max_block_strips = min(max_core_strips, 32·block_tiles)` |
| `TILE_SLOT_DEPTH` | 2 | `slot_depth = min(TILE_SLOT_DEPTH, max_blocks_per_core)` |
| `STAGING_GRANULES_CAP` | 512 | `staged_pieces_bound = 2·max_block_strips·P` (`P = 1` for `gather_aligned`); `staging_granules = min(STAGING_GRANULES_CAP, staged_pieces_bound)` |
| `COPY_TABLE_CAP` | 1024 | `copy_entries = min(COPY_TABLE_CAP, staged_pieces_bound)` |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_tile_images` | `TILE_SLOT_DEPTH` → `slot_depth` | 2 → effective 1 on every golden shape (one block per core) | the reader fills block `b+1` while the writer drains block `b`; exists only for multi-block cores, so it is not paid where it cannot be used |
| `cb_granule_staging` | `STAGING_GRANULES_CAP`, `COPY_TABLE_CAP` → `staging_granules`, `copy_entries` | 512 / 1024 → effective `staged_pieces_bound` (≤ 228) on golden shapes | staged reads in flight between two flushes (barrier + assemble); one flush per block whenever the bound fits |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read congruence `(l1_dst ^ dram_src) & (NOC_DRAM_READ_ALIGNMENT_BYTES − 1) == 0` — 64 B on Blackhole, 32 B on Wormhole (`noc_parameters.h:379`; enforced at `sanitize.h:745`) | a direct map-side face-row read in `gather_aligned` (32-B bf16 face rows land at 32-B row stride: src/dst parity differs on 50 % of segments of the 9 production `gather_aligned` shapes) | per segment at run time: direct iff the congruence holds, else **staged** (read the `granule_bytes` granule containing the face row into `cb_granule_staging`, copy after the barrier). `gather_general` stages every piece. fp32 face rows are 64 B ⇒ always congruent | NoC alignment hang under the watcher; silently shifted rows without it |
| Same congruence for tile-side reads (`scatter_aligned` loads input face runs into input tile images) | none — always congruent: tile images sit at `cb_base + k·tile_bytes` with the CB base DRAM-aligned (`allocate_dfb(..., get_alignment(BufferType::DRAM))`, `tt_metal/impl/program/program.cpp:2505-2509`) and DRAM pages at bank base + `k·tile_bytes` | kernel asserts the congruence under `--dev` | — |
| DRAM write congruence `(l1_src ^ dram_dst) & 15 == 0` (`noc_parameters.h:380`) | every write | writes start at face-row boundaries (multiples of 32 B) on both sides; partial pieces are never written straight to DRAM — they go through the tile image | NoC error / hang |
| CB-bounds sanitizer: a NoC transfer that starts inside a CB must end inside it (`sanitize.h:319-341`) | staged granule reads | granule slots are whole pages of `cb_granule_staging`; the copy table occupies the CB tail and is touched by RISC-V loads/stores only | watcher `DebugSanitizeCBOutOfBounds` |
| `cb_tile_images` ring wrap (CB ring-wrap invariant, `ttnn-cb-memory-fundamentals.md`) | push / pop quantum | every reserve/push/wait/pop is the nominal `block_tiles`; a ragged last block narrows the transfers, never the counts; capacity `slot_depth·block_tiles` closes the ring | pointer drift → hang |
| Kernel L1 budget 1,347,776 B (BB-05) | `block_tiles`, `slot_depth`, `staging_granules`, `copy_entries` | `BLOCK_SLOT_BYTES_CAP`, `TILE_SLOT_DEPTH`, the two staging caps ⇒ ≤ 565,248 B on every shape, any N (`l1_ledger.md`) | OOM at launch for large N |
| 32-bit linear index / byte offset in the kernels | N | host refuses `N ≥ 2**32` with `ValueError` (no golden or production shape is within 10⁵× of it) | wrapped indices → wrong values |

### Regimes

Boundary: DRAM. Minimum = every logical input byte crosses once, every logical output byte crosses once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **`gather_aligned`** | **built** (Phase 0) — 9 of 14 production bf16 cases | TILE ∧ `aligned` ∧ `runs(in) > runs(out)` | `block_tiles` output tiles × their rows in the core's run × whole segments | output: minimum, one write per (tile, face, row run). input: every valid face row once; staged face rows cost a 64-B granule for 32 B ⇒ input bytes 1.5× minimum on BH bf16 (fp32: 1.0×, all direct). Transactions: `segs` reads + `runs(out)` writes (`[16,1,5120]→[1,1,16,5120]`: 5120 + 320) | per block: 1 reserve + 1 read barrier + 1 assemble pass + 1 push (reader), 1 wait + 1 write flush + 1 pop (writer); plus one DRAM round trip amortized over the block's segments. Intended frequency: once per core on every golden shape |
| **`scatter_aligned`** | **built** (Phase 0) — the other 5 production bf16 cases (`[1,1,H,4608]→[H,1,4608]`, `[1,H,64]→[H,1,64]`, `[1,1,6,256]→[1,1,1536]`) | TILE ∧ `aligned` ∧ `runs(in) ≤ runs(out)` | `block_tiles` input tiles × their rows in the core's run × whole segments | minimum on both tensors, no staging: input face runs land congruent in input tile images; output face rows are written straight from them. Transactions: `runs(in)` reads + `segs` writes (`[1,1,16,4608]→[16,1,4608]`: 288 + 4608, against 4608 + 4608 as a gather) | same fixed costs as `gather_aligned` (no assemble pass) |
| **`gather_general`** | **built** (Phase 0) — no production bf16 case, but `SUPPORTED` admits every bf16 TILE shape, and the float32 FLUX cases (`[1,W]→[1,1,W,1]`, W ∈ {185,196,207,256,512}) land here | TILE ∧ ¬`aligned` | as `gather_aligned` | input: each distinct granule containing a piece, once per flush window (last-granule memo) — above minimum by granule rounding; structural when `W_in < 16` (each row's few valid elements sit alone in a face row). output: face runs including the padding columns of partial face rows (`W_out = 1`: one write per 16 valid elements, 16× bytes, 1/16 the transactions of element writes) | as `gather_aligned`, plus granule reuse across consecutive strips (`W_out = 1`: 16 strips per granule) |
| `page_identity_copy` | deferred — no golden or production shape satisfies the predicate; `gather_aligned`/`scatter_aligned` are correct there at minimum bytes | TILE ∧ `H_in == H_out` ∧ `W_in == W_out` (then output page `t` ≡ input page `t` byte for byte) | `block_tiles` whole tiles | minimum in whole-tile transfers (1 read + 1 write per tile instead of face runs + segment transfers) | reachable: the tile side already moves face runs at identical offsets; this regime makes both sides tile-side with whole-tile runs |
| `rm_page_gather` | deferred — ROW_MAJOR enters `SUPPORTED` only together with int32/uint32 (spec rule 7); Phase 0 is TILE only | ROW_MAJOR | `block_pages` output pages (page = one output row, at its aligned-page stride) | input: one read per (output-page piece, input page) span, 64-B congruence handled by the same staging as `gather_general`; output: one write per output page | reachable: same reader/writer skeleton — page side = output pages held in the slot at aligned-page stride, map side = contiguous input spans; only the address function changes (no faces) |
| `rm_bank_run_writes` | deferred — successor of `rm_page_gather` for the 4 flagged int32 `[1,W]→[W,1]` cases (W = 4096, 3844, 1024, 1000); not reachable before it | ROW_MAJOR ∧ `W_out·esize < aligned_page_bytes(out)` | per core: its output pages grouped by DRAM bank (page `p` → bank `p mod num_banks`, slot `p div num_banks`) | output: one write per (bank, run of consecutive slots) — the alignment gap of each 64-B slot is written as part of the output buffer's own slot; p1 = 4096 four-byte pages → ~`num_banks` writes per core instead of 4096 total single-page writes; input: one contiguous read per core | the slot already holds pages at their aligned stride; this regime only orders the slot by bank and merges spans — layered on `rm_page_gather`, nothing rewritten |
| `whole_tiles_in_out` | rejected — dead end, superseded everywhere by the three built regimes | — | whole tiles | reads every padding row (`[16,1,5120]`: 2560 tiles = 5.24 MB for 160 KiB of values) and writes every padding row (`[16,1,4608]`: 4.7 MB for 144 KiB) — 16–32× minimum | — |
| `engine_untilize_regroup_tilize` | rejected — routes values through unpack/DEST/pack on a bit-exact contract (NaN payloads, denormals, int32 above 2²⁴ are not guaranteed to survive), reads whole padding tiles anyway, adds a compute kernel and two CBs, and still needs this design's index-map copies for the regroup | — | — | ≥ whole-tile traffic + two engine passes | — |
| `staged_through_dram_or_host` | rejected — forbidden by the contract (no further DRAM tensor, no host read) | — | — | 2× minimum | — |
| `cross_core_segment_routing` | rejected — DRAM → core A → NoC → core B → DRAM; each segment is already read and written by its owner, so the extra hop saves no DRAM byte | — | — | minimum DRAM + one NoC crossing per segment | — |
| `single_core` | rejected — Phase 0 is multi-core by contract; the strip split is free | — | — | minimum | — |

**Regime selection function** (host, exact, depends only on the shapes — never on the grid):

```
if layout != TILE:            # unreachable in Phase 0 (validate refuses it)
    regime = rm_page_gather / rm_bank_run_writes   (deferred)
elif not aligned:              regime = "gather_general"
elif runs(in) <= runs(out):    regime = "scatter_aligned"
else:                          regime = "gather_aligned"
```

Each regime compiles from the same two kernel sources with exactly one define: `G6_REGIME_GATHER_ALIGNED`, `G6_REGIME_SCATTER_ALIGNED`, `G6_REGIME_GATHER_GENERAL` (dead paths compiled out, BB-18). **Regime-pinned tests** (acceptance test asserts `g6_reshape_plan(...)["regime"]` before checking values): `gather_aligned` = the 9 production cases with `runs(in) > runs(out)`; `scatter_aligned` = the 5 others; `gather_general` = production geometries of the int32/float32 cases run in bf16 (`[1,185]→[1,1,185,1]`, `[1,77]→[77,1]`, `[1,4096]→[4096,1]`, `[8]→[8,1]`, `[8,1]→[8]`).

### Traffic ranking

Candidate splits ranked by DRAM crossings per tensor, then cross-core traffic, then transaction count (qualitative; no ns). `segs` = valid segments (equal on both sides when `aligned`).

| Rank | Split | input DRAM crossings | output DRAM crossings | Cross-core | Transactions | Verdict |
|------|-------|----------------------|-----------------------|------------|--------------|---------|
| 1 | **contiguous tile-side strip runs, owner moves; tile side = the tensor with fewer `runs`** | 1 × valid (`gather_aligned` staged face rows +32 B each) | 1 × valid | none | `segs + min(runs(in), runs(out))` | **chosen** — minimum bytes on both tensors, no combine, fills the grid at strip granularity |
| 2 | same split, tile side fixed to the output (gather only) | 1 × valid (+ staging) | 1 × valid | none | `segs + runs(out)` | equal bytes, ~2× the transactions on the 5 `scatter_aligned` shapes plus avoidable staging → superseded by the `runs` predicate |
| 3 | whole tile-side tiles per core (tile granularity) | 1 × valid | 1 × valid | none | as rank 1 | equal bytes, but the tile count caps the core count (2 cores for `[16,1,64]→[1,1,16,64]`) — strip runs subsume it |
| 4 | whole tiles in, whole tiles out | all input tiles incl. padding | all output tiles incl. padding | none | `num_tiles_in + num_tiles_out` | 16–32× bytes → rejected |
| 5 | split + cross-core routing | 1 × valid | 1 × valid | +1 NoC crossing per segment | 3 per segment | rejected |
| — | dependent-axis split + combine | — | — | — | — | not applicable: the op has no dependent axis — no logical value spans an axis |

**Operand-reuse check.** One input. Along the chosen split every core reads a disjoint set of input face rows (bijection), so the input is not reuse-shared by construction of the split; the only shared bytes are a `gather_general` granule straddling a core boundary (≤ 1 granule per boundary). No broadcast/multicast regime.

**Stall-shadow check.** (a) The `gather_*` reader blocks on its read barrier; the only work independent of the not-yet-landed direct reads is assembling staged pieces that already landed — legal in any order (pure moves are bit-identical under any reorder, no precision gate). Realizing it needs scoped barriers (staged reads on one transaction id, direct reads on another) → perf lamp "trid split". (b) The writer blocks on `cb_wait_front` for a whole block; nothing in its own role is independent of that block. The work it could absorb is half of the reader's map-side reads (catalog `split_reader`) → perf lamp. (c) `scatter_aligned`: the reader issues a handful of face runs and the writer waits for them; nothing is independent of the wait.

### Block schedule

```cpp
// gather_aligned / gather_general              // scatter_aligned
for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   for (block_idx ...) {
    gather_block(block_idx);    // reader                                  load_block(block_idx);     // reader
    assemble_block(block_idx);  // reader                                  scatter_block(block_idx);  // writer
    store_block(block_idx);     // writer                              }
}
```

| Block operation | Kernel | Block shape | What it does | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|-------------|--------------|--------------------|-----------------------------------|
| `gather_block` | reader (map side) | the block's strips: ≤ `block_tiles` output tiles × their rows in the run × segments | `cb_reserve_back(cb_tile_images, block_tiles)`; walk the block's segments; per piece compute the source (input tile id + byte offset) by the index map and the destination `slot + (t − block_first_tile)·tile_bytes + face_off(r, 16·fh) + piece offset`. `gather_aligned`: **direct** `noc_async_read` of the face row when the congruence holds, else **staged**. `gather_general`: every piece staged. Staged = read the granule containing the piece into the next staging page unless it equals the last staged granule (memo), and append `(staging_addr, dst_addr, bytes)` to the copy table. Ring or table full ⇒ flush (barrier + copies of the table, reset ring, table and memo) and continue — a flush may fall at any piece boundary | the reserved slot (`block_tiles` tile images) | 1 reserve per block; flushes = `ceil(staged pieces / min(staging_granules, copy_entries))` = 0 extra on golden shapes; TensorAccessor built once per kernel |
| `assemble_block` | reader | the block's copy table | `noc_async_read_barrier()` (it also invalidates the BH RISC-V L1 cache, `dataflow_api.h:1921-1932`); execute the copy table with RISC-V loads/stores (8 × `uint32` for a 32-B bf16 face row; element copies of `esize` for partial pieces); `cb_push_back(cb_tile_images, block_tiles)` | slot + staging | 1 barrier + 1 push per block |
| `store_block` | writer (tile side) | the block's tiles × valid faces × the run's rows | `cb_wait_front(cb_tile_images, block_tiles)`; per tile, per face (`hc = 1` only if `vc(tc) > 16`), the run's rows inside that face half form ONE `noc_async_write` of `rows·face_row_bytes` from `slot + (t − first)·tile_bytes + face_off(row0, 16·hc)` to output page `t` + the same offset; `noc_async_writes_flushed()`; `cb_pop_front(cb_tile_images, block_tiles)` | the slot being drained | ≤ 4 writes per tile (2 for a tile with ≤ 16 rows, 1 when `vc ≤ 16`); 1 flush + 1 pop per block; `noc_async_write_barrier()` once at kernel end (BB-03) |
| `load_block` | reader (tile side) | ≤ `block_tiles` input tiles × valid faces × the run's rows | `cb_reserve_back(cb_tile_images, block_tiles)`; per tile, per valid face, ONE `noc_async_read` of the run's rows in that face from input page `t` + `face_off(row0, 16·hc)` to `slot + (t − first)·tile_bytes` + the same offset (always congruent); `noc_async_read_barrier()`; `cb_push_back(cb_tile_images, block_tiles)` | the reserved slot | 1 reserve + 1 barrier + 1 push per block; ≤ 4 reads per tile |
| `scatter_block` | writer (map side) | the block's segments | `cb_wait_front(cb_tile_images, block_tiles)`; per segment compute the output tile id + byte offset of its (whole) destination face row by the index map; `noc_async_write` of `face_row_bytes` from the slot face row; `noc_async_writes_flushed()`; `cb_pop_front(cb_tile_images, block_tiles)` | the slot being drained | 1 wait + 1 flush + 1 pop per block; `noc_async_write_barrier()` at kernel end |

**Index map** (one header shared by both kernels, divisors are CT constants so division compiles to multiply-shift; the implementer may walk `L` incrementally instead):

| Direction | Map |
|-----------|-----|
| output segment → input pieces (`gather_*`) | `g_o = img·H_out + 32·tr + r`; `c0 = 32·tc + 16·fh`; `n = min(16, W_out − c0)`; `L = g_o·W_out + c0`. For `p ∈ [L, L+n)`: `g_i = p div W_in`, `c_i = p mod W_in`, `img_i = g_i div H_in`, `r_i = g_i mod H_in`; tile `(img_i·Ht_in + r_i div 32)·Wt_in + c_i div 32`, offset `face_off(r_i mod 32, c_i mod 32)`. A piece ends at `c_i mod 16 == 15`, at `c_i == W_in − 1`, or at `n`. |
| input segment → output face row (`scatter_aligned`) | `g_i = img·H_in + 32·tr + r`; `L = g_i·W_in + 32·tc + 16·fh`; `g_o = L div W_out`, `c_o = L mod W_out`; tile `((g_o div H_out)·Ht_out + (g_o mod H_out) div 32)·Wt_out + c_o div 32`, offset `face_off((g_o mod H_out) mod 32, c_o mod 32)` (`c_o mod 16 == 0` by `aligned`). |

Padding rule (spec rule 2): only logical positions are ever addressed as sources. A staged granule may contain padding bytes, but only piece bytes are copied. A partial destination segment (`n < 16`, `gather_general`) leaves face-row bytes `n..15` of the slot uninitialized; they land in output padding columns, which the contract leaves unspecified.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (`MIN_STRIPS_PER_CORE = 1`, full occupancy) | 10 of 14 production shapes put 1–7 strips on a core; per-core fixed costs (launch, RT args, accessor setup, one DRAM round trip, one write flush) may dominate, and per-core RT args cost dispatch time | `MIN_STRIPS_PER_CORE ∈ {4, 8, 16, 32}`; judge on unprofiled traced-replay time, not on profiled kernel time |
| Overlap (one block = whole per-core run) | the writer cannot issue its first write before every read of the core has landed | 2–4 blocks per core (`BLOCK_SLOT_BYTES_CAP = ceil(core_tiles/2)·tile_bytes`, `TILE_SLOT_DEPTH = 2`) on `[16,1,5120]→[1,1,16,5120]` and `[1,1,16,4608]→[16,1,4608]` |
| Trid split (catalog `trid_double_issue`; the stall-shadow reorder) | `assemble_block` waits for every read, including direct reads it does not depend on | staged reads on trid A, direct reads on trid B: `async_read_barrier_with_trid(A)` → assemble → barrier B → push (`local_copy_helpers_dataflow.hpp:147,158`) |
| Staged granule size | `gather_aligned` stages ~50 % of segments on BH bf16 and reads 64 B for 32 B | (a) read only the 32-B face row into the staging page at offset `src mod 64` (still congruent); (b) for output tiles with exactly one valid row, offset the tile image by 32 B in the slot so every segment of that tile is direct |
| Face-run vs span writes (catalog `tile_reorder`) | a 1–16-row output tile is written as 2 face runs; one larger write may beat two small ones | one write spanning the tile's first to last valid byte (padding rows in between are output padding) |
| Read face-pair coalescing | 1-row input tiles (`gather_aligned` on `[H,1,W]→[1,1,H,W]`) cost 2 reads of 32 B per strip | one 544-B read (face 0 row 0 … face 1 row 0) into staging, 2 copies: half the reads, 8.5× bytes on those tiles |
| Map-side run merging (catalog `read_coalescing`) | consecutive segments that are contiguous on the map side (identity-like regroups) are issued separately | merge contiguous map-side runs into one transfer |
| Split reader (catalog `split_reader`) | with one block per core the writer RISC-V idles during the whole gather | the writer issues half the block's map-side reads into a second slot it owns; only if the reader is issue-bound |
| Bank stagger (catalog `bank_stagger`) | every core starts its walk at a tile whose bank index is `start_tile mod num_banks`; neighbours may collide | rotate each core's walk start by `core_idx mod num_banks` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_tile_images` (tile side, `scatter_aligned`) | input face runs (≤ 512 B bf16) | `noc_async_read(TensorAccessor(input).get_noc_addr(t, face_off), slot_addr, bytes)` | identical intra-tile offset on both sides ⇒ always congruent |
| DRAM → `cb_tile_images` (map side, `gather_aligned` direct) | one 32-B face row | `noc_async_read` | only when `(dst ^ src) & (NOC_DRAM_READ_ALIGNMENT_BYTES − 1) == 0` |
| DRAM → `cb_granule_staging` (map side, staged) | one `granule_bytes` granule | `noc_async_read` of the granule containing the piece | last-granule memo |
| `cb_granule_staging` → `cb_tile_images` | piece bytes | RISC-V loads/stores after the barrier | reader-private |
| `cb_tile_images` → DRAM (tile side, `gather_*`) | output face runs | `noc_async_write(slot_addr, TensorAccessor(output).get_noc_addr(t, face_off), bytes)` | one write per (tile, face, row run) |
| `cb_tile_images` → DRAM (map side, `scatter_aligned`) | one face row per segment | `noc_async_write` | 16-B congruence holds for every face row |
| (no compute kernel) | — | — | bit-exact by construction |

Placement: reader on NCRISC / NoC0 (`ttnn.ReaderConfigDescriptor()`), writer on BRISC / NoC1 (`ttnn.WriterConfigDescriptor()`) — catalog `noc_placement`. No regime, built or deferred, uses inter-Tensix communication (no semaphores, no multicast).

**TARGET as regimes.**

| TARGET value | Regime(s) | Class |
|--------------|-----------|-------|
| `bfloat16` × `TILE_LAYOUT` | `gather_aligned`, `scatter_aligned`, `gather_general` | built (Phase 0) |
| `float32` × `TILE_LAYOUT` (FLUX `[1,W]→[1,1,W,1]`) | the same three with `esize = 4` (granule = face row = 64 B ⇒ `gather_aligned` all-direct); the 5 FLUX cases select `gather_general` | knob-turn: `esize` is a CT arg and every offset derives from it |
| `int32`, `uint32` × `ROW_MAJOR_LAYOUT` | `rm_page_gather` + `rm_bank_run_writes` | regime addition on the same reader/writer skeleton (new address function, no new pipeline); arrives with ROW_MAJOR in one change (rule 7) |

There is no placement axis in TARGET (every case is DRAM interleaved), so no sharded realization is designed. No index axis (`dim`) exists.

**Structural impossibilities:** none (agrees with `feature_spec.INVALID = []`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | strip of the tile-side tensor; a block = ≤ `block_tiles` consecutive tile-side tiles of the core's run |
| Grid | `device.compute_with_storage_grid_size()` (runtime query, never a literal); cores = `ttnn.num_cores_to_corerangeset(n_cores, grid, row_wise=True)` → ≤ 2 rectangular ranges (BB-17); reader, writer and both CBs on that one CoreRangeSet |
| Core count | `n_cores = min(grid_cores, ceil(U / MIN_STRIPS_PER_CORE))` — always ≤ U, so no core is idle |
| Per-core work | contiguous strip runs: the first `U mod n_cores` cores take `ceil(U / n_cores)` strips, the rest `floor(U / n_cores)`; host computes each core's `(start_tile, start_row)` by the closed form and its `num_blocks_this_core` |
| Remainder | strip counts differ by ≤ 1 across cores; ragged last block keeps nominal CB counts; tile geometry uses `ceil` and per-image `Ht` (`num_tiles = IMG·ceil(H/32)·ceil(W/32)`), never `floor(IMG·H/32)` |
| Production consequence (110-core grid) | `[16,1,5120]→[1,1,16,5120]` 2560 strips → 110 cores; `[1,1,16,4608]→[16,1,4608]` 2304 → 110; `[1,16,6,256]→[1,1,16,1536]` 768 → 110; `[2,1,5120]→[1,1,2,5120]` 320 → 110; `[16,1,64]→[1,1,16,64]` 32 → 32 (2 output tiles); `[1,1,6,256]→[1,1,1536]` 48 → 48 |
| Plan reporting (BB-08, BB-21) | `ttnn.operations.g6_reshape.g6_reshape_plan(input_tensor, shape) -> dict` (host only, no dispatch) with at least `regime`, `tile_side` (`"output"`/`"input"`), `work_units` (U), `n_cores`, `num_core_ranges`, `core_ranges`, `block_tiles`, `max_blocks_per_core`, `slot_depth`, `staging_granules`, `strips_per_core` (min, max) and `roles` (`reader`/`writer` role names; identical on every core) |
| Profiling zones (BB-21) | `DeviceZoneScopedN` around every piece of work and every wait: reader `G6_RD_RESERVE_WAIT`, `G6_RD_ISSUE`, `G6_RD_BARRIER`, `G6_RD_ASSEMBLE`, `G6_RD_PUSH`; writer `G6_WR_FRONT_WAIT`, `G6_WR_ISSUE`, `G6_WR_FLUSH`, `G6_WR_FINAL_BARRIER`. The macro is `(void(sizeof(name)))` without the profiler (`tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1047`) ⇒ the plain build is unchanged |
| Program description cache (BB-07, BB-16) | CT args: geometry of both tensors (`W, H, IMG, Wt, Ht`), `esize`, `block_tiles`, `slot_depth`, `staging_granules`, `copy_entries`, `granule_bytes`, CB indices, `TensorAccessorArgs`; regime define. Per-core RT args: `start_tile`, `start_row`, `num_strips`, `num_blocks_this_core`. **Common RT args: `[input_address, output_address]`** (same on every core). The Python side memoizes `(ProgramDescriptor, plan)` in a module dict keyed by `(tuple(d), tuple(input.padded_shape), tuple(e), dtype, layout, memory_config, device id, grid size, (MIN_STRIPS_PER_CORE, BLOCK_SLOT_BYTES_CAP, TILE_SLOT_DEPTH, STAGING_GRANULES_CAP, COPY_TABLE_CAP))` — never a buffer address. A hit rewrites only the two common RT args of both kernels; `custom_program_hash` may be set to a hash of the same key. No host read of tensor values; no allocation besides the output; the first (eager) call builds and compiles, so a traced call allocates only its output and builds nothing |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_tile_images` | 0 | `tile_bytes` (2048 B bf16) | `slot_depth · block_tiles` | spans `t_s` (`block_tiles`), `r` (all 32 row positions of a tile image), `fh`, `e`; streams over blocks; map-side axes never resident. Holds output tile images in `gather_*`, input tile images in `scatter_aligned` | input dtype as raw bytes (never unpacked) | reader | writer | whole kernel; ring of `slot_depth` block slots |
| `cb_granule_staging` (`gather_*` only; not allocated for `scatter_aligned`) | 1 | `granule_bytes` (64 B BH) | `staging_granules + ceil(8·copy_entries / granule_bytes)` | granule ring (one page per staged granule) + copy table in the tail (8 B per entry); live set = staged pieces between two flushes; streams over strips and blocks | raw bytes | reader (private scratch: `get_write_ptr` once, never reserved/pushed) | reader | whole kernel |

Full ledger, axis accounting and the data-movement budget: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_block` | ≤ `block_tiles` output tiles × run rows × segments | no compute helper applies (dataflow only): TensorAccessor + `noc_async_read` | input tensor (DRAM); `cb_granule_staging` ring | `cb_tile_images`, `block_tiles` reserved | slot reserved; direct reads and staged granules in flight |
| 2 | `assemble_block` | the block's copy table | RISC-V copies via `addr_to_l1_ptr` | `cb_granule_staging` (landed granules) | `cb_tile_images`, `block_tiles` | `block_tiles` pushed; staging reset |
| 3 | `store_block` | touched output tiles × valid faces × run rows | TensorAccessor + `noc_async_write` | `cb_tile_images`, wait `block_tiles` | output tensor (DRAM) | writes flushed; `block_tiles` popped |
| 1' | `load_block` | ≤ `block_tiles` input tiles × valid faces × run rows | TensorAccessor + `noc_async_read` | input tensor (DRAM) | `cb_tile_images`, `block_tiles` reserved → pushed after the barrier | `block_tiles` pushed |
| 2' | `scatter_block` | the block's segments | TensorAccessor + `noc_async_write` | `cb_tile_images`, wait `block_tiles` | output tensor (DRAM) | writes flushed; `block_tiles` popped |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all transfers | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | page = tile id, offset = `face_off(...)` (+ piece offset), or the granule-aligned offset | — | — | — |
| `gather_block`, `load_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, dst_l1, bytes)` | — | `cb_tile_images` / `cb_granule_staging` | — |
| `assemble_block`, `load_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block (+ once per flush) | — | — | frequency set by `block_tiles`, `staging_granules`, `copy_entries` |
| `assemble_block` | helper | `addr_to_l1_ptr` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | RISC-V word / element copies | `cb_granule_staging` | `cb_tile_images` | — |
| `store_block`, `scatter_block` | raw_api | `noc_async_write`, `noc_async_writes_flushed`, `noc_async_write_barrier` | `dataflow_api.h:876`, `:1980`, `:1951` | one write per (tile, face, row run) / per segment | `cb_tile_images` | — | `block_tiles` sets flush frequency |
| all | raw_api | `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`, `get_write_ptr`, `get_read_ptr` | `dataflow_api.h:406`, `:208`, `:479`, `:260`, `:325`, `:346` | nominal `block_tiles` | `cb_tile_images` | `cb_tile_images` | `block_tiles` |
| all | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012`, `:1047` | zone names above | — | — | — |
| (lamp: trid split) | helper | `dataflow_kernel_lib::set_read_trid`, `async_read_barrier_with_trid` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147`, `:158` | one trid per read class | — | — | — |

**Helpers considered and rejected** (cover every `raw_api` row above):

| Candidate | File:Line | Why it cannot express this op's block operation |
|-----------|-----------|-----------------------------------------------|
| `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163` | consumes whole input tiles through unpack → DEST → pack: padding tiles are read (regime `whole_tiles_in_out`), values pass the compute engine on a bit-exact contract, and the regroup between untilize and tilize still needs this design's index-map copies |
| `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206` | the mirror of the row above; same three reasons |
| `read_sticks_for_tilize`, `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:88`, `:130` | move whole row-major sticks of a ROW_MAJOR tensor; Phase 0 tensors are TILE and the move unit is a face-row segment inside a tile page — no stick exists on either side (candidates for `rm_page_gather`) |
| `copy_tile_column`, `fill_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:92`, `:70` | copy one *column* of one tile into a column of another; the source of a `gather_general` piece is a contiguous run of a *face row* of a granule, not a tile column, and the template needs a DataflowBuffer `dfb_id` for tile dims |
| `dataflow_kernel_lib::local_addr`, `set_read_state` / `read_with_state` | `local_copy_helpers_dataflow.hpp:57`, `:75`, `:95` | NoC self-copies need an issue + barrier per batch; a staged copy is 32 B (8 words) or a few elements, cheaper as RISC-V stores, and `read_with_state` fixes one size per loop while `gather_general` pieces vary. The implementer may use them for long pieces |
| `ttnn.Mcast` / `mcast/kernel/mcast_args.hpp` | — | no reuse-shared operand and no cross-core step in any regime |
| eltwise / reduce / matmul helpers | — | the op has no arithmetic |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input padding reaching a logical output | the golden suite poisons input padding with NaN (`fill_implicit_tile_padding`); `from_torch` zero padding hides a leak | only logical positions are addressed as sources; staged granules contribute only piece bytes; `aligned` regimes move whole valid face rows only |
| BH 64-B DRAM read congruence | 32-B face rows at 32-B row stride; src/dst parity differs on ~50 % of `gather_aligned` segments | runtime check per segment, staging otherwise; new kernels first run with `--dev` (NoC sanitizer, CB bounds) |
| Ordering of RISC-V copies and `cb_push_back` | the writer NoC-reads slot bytes the reader stored with RISC-V stores | BH L1 cache is write-through (`tt_metal/hw/inc/internal/tt-1xx/cache.h:7-15`) and stores issue in order; if `--dev` shows stale slot bytes, a `fence` before the push closes it. The reader's own barrier invalidates its cache before it loads staged bytes |
| Core boundary inside a tile | strip granularity: several cores write rows of one output tile (`[16,1,64]→[1,1,16,64]`: 16 cores per tile) | each core writes only its own rows' face runs; byte ranges are disjoint 32-B multiples. Two cores may write the two 32-B halves of one 64-B DRAM granule — byte-masked NoC writes; covered by the production `gather_aligned` shapes with core boundaries inside tiles |
| Program cache replaying a wrong map (BB-07) | `[16,1,64]→[1,1,16,64]` then `→[1,1,1,1024]` on one input share the input geometry | output geometry and the regime define are CT; the memo key holds `e` and every knob; acceptance test calls both targets and re-calls the first |
| Stale buffer address on a memo hit | the memo returns a descriptor built for another tensor | addresses live only in the two common RT args, rewritten on every call; acceptance test reuses a signature with fresh tensors |
| Rank-1 / rank-2 TILE tensors | rank 1 has `H = 1` and padded shape `[32, 32]` | derive `IMG, H, W` from the logical shape exactly as the geometry table says; assert `IMG·Ht·Wt == buffer_num_pages()` on both tensors |
| Staging overflow mid-segment (`gather_general`, `W_in < 16`) | up to 16 pieces per segment, one granule each before memo hits | flush at any piece boundary when the ring or the copy table is full; the multi-block knob test drives both caps to 4 |
| Regime predicate tie | `runs(in) == runs(out)` | deterministic: `scatter_aligned` (no staging) |
| Multi-block path untested by production shapes | full occupancy puts every golden shape at one block per core | acceptance test shrinks `BLOCK_SLOT_BYTES_CAP`, the staging caps and raises `MIN_STRIPS_PER_CORE` per regime and asserts `max_blocks_per_core > 1` |
| Transactions in flight at kernel exit (BB-03) | the next program starts right after | writer ends with `noc_async_write_barrier()`; reader's last block ends with a read barrier before its push |
| Kernel text size (BB-04, BB-18) | three regimes in two sources | one regime define per build; no compute kernel; geometry as CT constants keeps the address math small |
