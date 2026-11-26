# Operation Design: g6_permute

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic, no compute kernel) |
| Goal | Reorder the dims of one device tensor as `torch.permute(x, dims)`, as exactly one device program, every value moved bit-for-bit (bfloat16). Phase 0 builds the dims (0, 2, 1, 3) TILE bfloat16 DRAM class (2176 of 2203 traced calls). |
| Math | `out.shape[k] = x.shape[dims[k]]`; `out[o_0..o_{r-1}] = x[i_0..i_{r-1}]` with `i_{dims[k]} = o_k`. Phase 0: `out[n, b, a, :] = x[n, a, b, :]` (rows of W values move whole) |
| References | Spec: `docs/explorations/g6_permute/spec/g6_permute.txt` (the task text wins on conflict); math `docs/explorations/g6_permute/math.md` (F1 row swap); tile geometry `docs/explorations/g6_permute/tensix.md`; risks `docs/explorations/g6_permute/risks.md`; golden suite `eval/golden_tests/g6_permute/{feature_spec,helpers}.py`; blocking `.claude/references/blocking-model.md`; NoC alignment `.claude/references/noc-alignment-in-tt-metal.md`; catalog `ttnn/ttnn/operations/examples/{tile_reorder,double_buffer,noc_placement,bank_stagger,trid_double_issue,split_reader}/` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device, interleaved; Phase 0: rank 4, bfloat16, TILE, DRAM, `h_non_aligned` | — | buffer address RT; shape, dtype, layout CT |
| `dims` | sequence of `r` ints, positional | yes | each in `[-r, r-1]`; after wrapping negatives (`d + r`) a permutation of `range(r)`; Phase 0: `(0, 2, 1, 3)` | — | CT (selects regime and index map; in the program cache key) |

`dims` canonicalization: TARGET declares `dims` non-negative. The entry point wraps negative entries **before** the support check, so `(0, -2, 1, -1)` is the cell `(0, 2, 1, 3)` and gets the same program.

### Registry declarations (op file `ttnn/ttnn/operations/g6_permute/g6_permute.py`)

| Item | Decision |
|------|----------|
| Scenario dict built by `validate()` | `{"input_shape": tuple(int(d) for d in input_tensor.shape), "dims": tuple(int(d) for d in dims), "dtype": input_tensor.dtype, "layout": input_tensor.layout, "buffer": "dram" \| "l1"}`; a non-interleaved input gets `buffer = "<dram\|l1>_<memory_layout>"`, a value no SUPPORTED list holds, so it is refused by the per-axis check |
| `INPUT_TAGGERS` (order) | `dtype` → `scenario["dtype"]`; `layout` → `scenario["layout"]`; `buffer` → `scenario["buffer"]`; `rank` → `int(len(input_shape))`; `alignment` → W = shape[-1] not % 32: `"w_non_aligned"`; else H = shape[-2] not % 32: `"h_non_aligned"`; else `"tile_aligned"`; `dims` → canonical tuple (negatives wrapped) |
| `SUPPORTED` (Phase 0) | `dtype: [ttnn.bfloat16]`, `layout: [ttnn.TILE_LAYOUT]`, `buffer: ["dram"]`, `rank: [4]`, `alignment: ["h_non_aligned"]`, `dims: [(0, 2, 1, 3)]` |
| `EXCLUSIONS` (Phase 0) | `[]` |
| `validate(input_tensor, dims)` order | 1. structural dims checks → `ValueError` whose message contains the word `dims` (length ≠ rank; entry outside `[-r, r-1]`; not a permutation after wrapping); 2. per-axis SUPPORTED → `UnsupportedAxisValue`; 3. EXCLUSIONS → `ExcludedCell`. Step 1 runs first because `tag_dims` needs a well-formed `dims`. INVALID is never checked. |
| Entry point | `g6_permute(input_tensor, dims)`: first line `validate(...)`; then plan lookup (cached), output allocation, one `ttnn.generic_op` |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, dims) -> dict`, exported from `ttnn.operations.g6_permute` next to `g6_permute`. Keys: `regime` (`"rows_tile"`), `block_a`, `block_b`, `block_wt`, `num_blocks`, `num_cores`, `depth_in`, `depth_staging`, `regroup` (bool, `block_a > 1`), `cores` (list, one dict per core that runs kernels: `core` (x, y), `role` (`"rows_tile reader+writer"`), `start_block`, `num_blocks`). Host-only: no dispatch, no allocation. |

### Structural impossibilities (for a later `/golden-tests` pass; not edited here)

| Cell | Why impossible |
|------|----------------|
| `{rank: 4, dims: (0, 1, 4, 2, 5, 3)}` | a dims of length 6 on a rank-4 tensor is a validation error, never a result |
| `{rank: 6, dims: (0, 2, 1, 3)}`, `{rank: 6, dims: (0, 3, 1, 2)}` | same: dims length must equal rank |

The projected axes never build these cells; they matter only if TARGET ever gains a free axis.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Phase 0: `[N, A, B, W]`, any N, A ≥ 1; B % 32 ≠ 0, W % 32 = 0 (`h_non_aligned`). Production: N = 1, W = 64, (A, B) ∈ {(1,16), (16,1), (6,16), (16,6), (1,8), (8,1), (6,8), (8,6)} |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE (32×32 tiles, 16×16 faces) |
| Memory | DRAM interleaved (Phase 0) |
| Padding | rows `B..ceil32(B)-1` of every tile row hold anything; they never reach a logical output position |

### Output

| Property | Value |
|----------|-------|
| Shape | `[x.shape[d] for d in dims]`; Phase 0 `[N, B, A, W]` |
| Dtype | input's |
| Layout | input's |
| Memory | input's memory config (interleaved, same buffer type) |
| Padding | rows `A..ceil32(A)-1` of each output tile: unspecified (stale staging bytes may be written there) |

## Blocking Model

Index space of the Phase 0 map (all counts alignment-aware, per image):
`Bt = ceil(B/32)`, `At = ceil(A/32)`, `Wt = ceil(W/32)`.
Input page `in_page(n, a, bt, wt) = ((n·A + a)·Bt + bt)·Wt + wt`; output page `out_page(n, b, at, wt) = ((n·B + b)·At + at)·Wt + wt`.
A **row move** `(n, a, b, wt)`: input tile `in_page(n, a, b/32, wt)` row `rb = b % 32` → output tile `out_page(n, b, a/32, wt)` row `ra = a % 32`.
A tile row `r` is two face rows of 16 values (32 B each): half `h ∈ {0, 1}` lies at byte offset `face_off(r, h) = ((r / 16)·2 + h)·512 + (r % 16)·32`.

A **closed cell** `(n, at, bt, wt)` holds `la = min(32, A − 32·at)` input tiles and `lb = min(32, B − 32·bt)` output tiles; its `la·lb` row moves read every input tile of the cell once and complete every output tile of the cell. A **block** is a sub-rectangle of one cell: one image, `block_a` rows of `a` inside one output tile row, `block_b` rows of `b` inside one input tile row, `block_wt` tile columns. `block_a, block_b ∈ {32, 16, 8, 4, 2, 1}` (divisors of the tile height), so a group never crosses a tile row and, when < 32, never crosses the row-16 face band: a span is one contiguous run per face half.

Block grid: `num_blocks = N · ceil(A/block_a) · ceil(B/block_b) · ceil(Wt/block_wt)`, block index `block_idx = ((n·NA + ag)·NBg + bg)·NW + wtb` with `NA = ceil(A/block_a)`, `NBg = ceil(B/block_b)`, `NW = ceil(Wt/block_wt)`. Group `ag` covers `a ∈ [ag·block_a, min(A, (ag+1)·block_a))`; group `bg` covers `b ∈ [bg·block_b, min(B, (bg+1)·block_b))`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock | Why |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|-----|
| `n` (N, images) | independent — copied index; no result spans it (math.md, Axes) | `block_n` | 1 | host constant `BLOCK_N = 1` (implicit in the block grid) | spread: outermost axis of the flattened block grid, contiguous block ranges per core | knob-turn | a cell is per image; a block over two images holds two disjoint cells with no shared transfer, so it would amortize only the ~5 handshakes per block against ≥ 2 transfers + `la·lb·2` copies of work |
| `a` (A; output tile-row axis `at = a/32`, row `ra = a%32`) | independent — each `a` is one input tile and one row position of the output; no combine | `block_a` | selection rule below: 32 (closed cells) when cells ≥ grid; else halved toward 1 (Qwen: 1) | `select_plan()` on host → one CT arg | spread: second axis of the block grid; split inside a cell only to fill the grid | knob-turn | `block_a < la` turns one whole-tile write per output tile into 2 row-run writes per output tile per group; `block_a = 1` makes the regroup the identity (rows written straight from the input tile) |
| `b` (B; input tile-row axis `bt = b/32`, row `rb = b%32`) | independent — each `b` is one output tile and one row of an input tile | `block_b` | selection rule: 32 unless `block_a` already reached 1 and the grid is still under-filled (Qwen 303/304: 2; the other six: 1) | `select_plan()` → one CT arg | spread: third axis of the block grid | knob-turn | `block_b < lb` turns one whole-tile read into 2 row-run reads per input tile per group; each live input row still crosses DRAM once |
| `wt` (W tile columns, kept in place) | independent — the last dim is kept; tile column `wt` of x feeds only tile column `wt` of out | `block_wt` | 1 | host constant `BLOCK_WT = 1` → CT arg | spread: innermost axis of the block grid | knob-turn | different `wt` share no tile; a second column per block doubles both CB footprints and amortizes only per-block handshakes, which are < 1 % of a cell's work |
| `c` (the 32 values of one tile row, two 16-value face halves) | independent — moves whole with its row | none (indivisible unit) | whole: both halves of a row move together | face layout (`tt_metal/hw/inc/api/tile_constants.h:18-23`) | not assigned across cores | none | a half is the transfer unit (32 B); splitting halves across blocks would double every read and write |

**Selection rule (pinned; single source `select_plan(N, A, B, Wt, max_cores)` on host).** Walk the sequence `(32,32) → a halves: (16,32), (8,32), (4,32), (2,32), (1,32) → b halves: (1,16), (1,8), (1,4), (1,2), (1,1)`. The start `(32, 32)` is always taken; each next step is taken iff its `num_blocks ≤ max_cores`; the walk stops at the first step that exceeds `max_cores`. `max_cores = G = grid.x · grid.y` of `device.compute_with_storage_grid_size()` (Phase 0). `num_blocks` is non-decreasing along the sequence, so the result is the finest split that does not put two blocks on one core, or the closed cells when even those outnumber the cores. `a` is split before `b` because an `a`-split keeps input reads whole and few (reads gate the pipeline; writes are asynchronous) and at `block_a = 1` removes the regroup; a `b`-split adds read transactions.

**Intermediate data re-run.** The only intermediate is the block's staged output tiles (`block_b × block_wt` tiles): their axes `b, wt` are assigned with the block, so the core that regroups them writes them. No core collects, coordinates or hands out work; no stage computes on one core while others wait.

**Phase 0 plans of the production shapes** (G = 110; identical for any G in [96, 191]):

| Case | x shape | cells | (block_a, block_b) | num_blocks = cores | Per block |
|------|---------|-------|--------------------|--------------------|-----------|
| 301 | (1,1,16,64) | 2 | (1, 1) | 32 | 2 row-run reads of 32 B; 2 direct writes of 32 B |
| 302 | (1,16,1,64) | 2 | (1, 1) | 32 | 1 whole-tile read (span covers the 1 live row); 2 direct writes of 32 B |
| 303 | (1,6,16,64) | 2 | (1, 2) | 96 | 2 row-run reads of 64 B; 4 direct writes of 32 B (2 output tiles) |
| 304 | (1,16,6,64) | 2 | (1, 2) | 96 | 2 row-run reads of 64 B; 4 direct writes of 32 B |
| 305 | (1,1,8,64) | 2 | (1, 1) | 16 | 2 reads of 32 B; 2 writes of 32 B |
| 306 | (1,8,1,64) | 2 | (1, 1) | 16 | 1 whole-tile read; 2 writes of 32 B |
| 307 | (1,6,8,64) | 2 | (1, 1) | 96 | 2 reads of 32 B; 2 writes of 32 B |
| 308 | (1,8,6,64) | 2 | (1, 1) | 96 | 2 reads of 32 B; 2 writes of 32 B |

Every knob is a parameter: `block_a`, `block_b`, `block_wt`, `max_cores`, `depth_in`, `depth_staging` are each defined once on the host; CB page counts, loop bounds, `NA/NBg/NW`, `num_blocks` and per-core ranges are derived from them. No kernel contains a literal standing for any of them.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `depth_in` (host `MAX_DEPTH_IN = 2`) | `min(2, max_blocks_per_core)` (1 for every production shape) | the reader fetches block k+1 while the writer regroups and stores block k |
| `cb_output_staging` | `depth_staging` (host `MAX_DEPTH_STAGING = 2`) | `min(2, max_blocks_per_core)` when `block_a > 1`, else 0 (not allocated) | the regroup of block k+1 runs while block k's writes drain out of L1 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| BH DRAM read: local and NoC address equal mod 64 (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`; rule in `.claude/references/noc-alignment-in-tt-metal.md` §1) | row-run reads of a `b`-span (`block_b < lb`) | every run lands at the **same tile-relative offset** in a whole-tile slot of `cb_input_tiles`; slots are tile-sized (2048 B) on a CB base the allocator aligns to the DRAM alignment (`tt_metal/impl/program/program.cpp:2509`) | a compacted landing slot breaks the mod-64 match for odd `rb`: watcher stop, silent corruption without it |
| Direct DRAM→output-position gather is illegal: `rb % 2 ≠ ra % 2` puts source and destination 32 B apart mod 64 | any design that reads a face row straight into its output position | rejected (regime table X2); rows are regrouped in L1 | half the rows corrupt |
| BH DRAM write: equal mod 16 (`noc_parameters.h:380`) | row-run writes | runs start at `face_off(r, h)`, a multiple of 32 B, from sources at multiples of 32 B | — (satisfied by construction) |
| L1→L1 self-read: equal mod 16 (`noc_parameters.h:375-376`) | regroup copies of 32 B | source `face_off(rb, h)` and destination `face_off(ra, h)` are multiples of 32 | — (satisfied by construction) |
| Face geometry: a tile row = 2 face rows; rows < 16 in faces 0/1, ≥ 16 in faces 2/3 | `block_a`, `block_b` | restricted to {32, 16, 8, 4, 2, 1}; at 32 the whole live range of the tile is covered (one whole-tile transfer); below 32 a span lies in one face band (one run per half) | a span crossing row 16 is two runs per half; code that emits one run copies the wrong 512 B |
| Stateful single-packet self-read: `transfer_size ≤ NOC_MAX_BURST_SIZE` (`ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.inl:66`) | regroup copy size | 32 B (one face row) | static_assert |
| Whole-tile transfers are valid only between tiles in the same row layout | store in the direct path (`block_a = 1`) | the direct path **always** writes 2 row runs per output tile (source row `rb`, destination row `ra`), never a whole tile, even when the a-span covers all live rows | rows land at input positions in the output tile: wrong values |
| Tile shape 32×32 with 16×16 faces, bfloat16 (2 B) | all offsets | host asserts `tile.tile_shape == [32, 32]` and `face_shape == [16, 16]`; byte geometry (2048 / 512 / 32 B) derived from dtype and passed as CT args | offsets wrong for any other tile |

### Regimes

Selection: dispatch on `(canonical dims, layout, dtype)` after `validate()`; Phase 0 has one built regime, every other cell is refused by SUPPORTED. Minimum stated against the DRAM boundary: each live input value crosses DRAM once, each output value once.

| Regime | Status | Predicate | Block | Data movement (vs. minimum) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------|--------------------------|
| R1 `rows_tile` | **built** (Phase 0) | `dims == (0,2,1,3)` ∧ TILE ∧ bfloat16 (Phase 0 cell: DRAM, rank 4, `h_non_aligned`) | `(1 image, block_a rows of a, block_b rows of b, block_wt tile columns)` inside one closed cell | minimum: every live input row crosses DRAM once (a whole tile when the block's b-span covers the tile's live rows — its padding rows then cross too), every live output row once (a whole tile when the a-span covers the live rows); live rows cross core-local L1 once when `block_a > 1`, zero times at `block_a = 1`; no cross-core traffic. Domain: every shape of the class; no other regime applies because the last dim is kept, so the move is a pure row permutation that whole-row transfers express exactly | per block: 1 reserve + 1 push + 1 read barrier (reader), 1 wait + 1 pop + 1 read barrier + 1 flush (writer). Growing `block_a` to the cell (`la`) replaces `2·block_b` row-run writes per group by `block_b` whole-tile writes; growing `block_b` to `lb` replaces 2 row-run reads per input tile per group by 1 whole-tile read |
| R2 `blocks_rm` (FLUX p1, p2) | deferred — the two flagged ROW_MAJOR rank-6 cases (24 traced calls) need an element-granular (2 B) regroup that the NoC cannot express (mod-16 rule) and a RISC-V halfword shuffle, a new kernel pair; Phase 0 serves the 2176-call class first | `dims == (0,1,4,2,5,3)` ∧ ROW_MAJOR ∧ rank 6 | `(n, c, block_h rows of h; all w, i, j)`: `block_h·W·I` consecutive input pages (one contiguous slot run per DRAM bank) → `I·J` output-page runs of `block_h` pages | x: the contract stores each 4-B page in a 64-B DRAM slot (16× logical bytes, fixed at the boundary, BB-09): reading every slot once (8 bank-coalesced reads per block) or one 4-B read per page are the two forms; the logical minimum (2 MB for p1) is structurally unreachable. out: each 64-B page once | per block: 8 bank reads amortize the per-transaction cost of 4-B pages; `block_h` sets how many output pages one regroup completes |
| R3 `nhwc_rm` (Panoptic p4) | deferred — same positive reason as R2 (2-B granular transpose `[W, C] → [C, W]` per image row); 1 traced call per Panoptic pass | `dims == (0,3,1,2)` ∧ ROW_MAJOR | `(n, h, block_w of w; all c)`: `block_w` input pages (64 B) → `C` output runs of `block_w·2` B (`block_w` a multiple of 8 for the L1 write alignment) | minimum: x pages once, out pages once (spans when `block_w < W`), all in L1; output L1 pages count against 1347776 B (38 × 512 B per core) | `block_w = W` makes every output write a whole 512-B page |
| R4 `nhwc_tile_bf8` (Panoptic p3) | deferred — needs a compute kernel (bfloat8_b tile transpose and requantization with fresh shared exponents) plus R1's row regroup; reachable on top of R1: out tile `(c, ht, wt)` row `hh` = row `c` of `transpose(x tile (32·ht+hh, wt))`, i.e. R1's row move with `a := h`, `b := c` after a per-tile transpose | `dims == (0,3,1,2)` ∧ TILE ∧ bfloat8_b | closed cell `(ht, wt)`: 32 input tiles → 32 output tiles | minimum: x once, out once (L1); plus two compute passes in L1 (transpose to bfloat16, exact; requantize to bfloat8_b after the regroup) | the cell is the closure; no larger block saves transfers |
| R5 `nhwc_tile_bf16` | deferred — non-production cell of the cartesian TARGET; needs a bit-exact in-tile transpose (RISC-V halfword) instead of R4's compute pass; EXCLUSIONS candidate when TILE + (0,3,1,2) enter SUPPORTED for bfloat8_b | `dims == (0,3,1,2)` ∧ TILE ∧ bfloat16 | as R4 | as R4 without requantization | as R4 |
| R6 `rows_tile_bf8` | deferred — non-production; rows still move whole and keep their 16-value groups (bit-exact), but the per-face-row exponent bytes need byte moves (RISC-V), and the direct path cannot write 1-B exponents to DRAM at mismatched offsets; EXCLUSIONS candidate | `dims == (0,2,1,3)` ∧ TILE ∧ bfloat8_b | as R1 | as R1 | as R1 |
| R7 `rows_rm` | deferred — non-production; `(0,2,1,3)` on ROW_MAJOR is a pure page permutation (out page `(n,b,a)` = x page `(n,a,b)`), no regroup; EXCLUSIONS candidate | `dims == (0,2,1,3)` ∧ ROW_MAJOR | `block_pages` contiguous output pages | minimum: each page once | fewer barriers per page |
| X1 per-output-tile gather with whole input re-reads | rejected — reads every input tile `lb` times (≤ 32×); superseded by R1's closed cell, which reads each once | — | one output tile | x crosses DRAM up to 32× | — |
| X2 DRAM face-row gather straight into output positions | rejected — illegal for rows with `rb % 2 ≠ ra % 2` (BH mod-64 DRAM reads); superseded by R1's tile-relative reads + L1 regroup | — | — | — | — |
| X3 scatter every live row to DRAM for all shapes (no regroup) | rejected as a regime — 64 DRAM writes of 32 B per full tile against 1 tile write; R1 reaches it as the knob value `block_a = 1`, where it wins (few cells) | — | — | out in 32-B writes | — |
| X4 compute-engine transpose / `reshuffle_rows_tile` for bfloat16 | rejected — unpacker/DEST/packer path may flush denormals and alter NaN (risks.md R3); R1 is a byte-exact move | — | — | — | — |
| X5 cross-core row forwarding (one core reads an input tile, unicasts rows to the owners of its output tiles) | rejected — R1's b-split already reads each live row from DRAM exactly once; forwarding adds a full cross-core copy of the live bytes plus semaphores | — | — | +1 cross-core crossing of all live bytes | — |
| X6 host-side layout or slice transforms, or a second program | rejected — forbidden (spec "Layout handling", BB-01, BB-02) | — | — | +1 full pass | — |

### Traffic ranking

No axis is dependent (no result spans any axis, `math.md#axes-and-dependence`), so no split needs a combine and no "split the dependent axis + combine" candidate exists. **Operand-reuse check:** splitting across cores along `n`, `a`, `b` or `wt` — x varies along every one of them, and so does out: no operand is reuse-shared by construction of any split, no broadcast row. Within a cell, one input tile feeds `lb` output tiles; under the b-split each core reads disjoint rows of it, so the fan-out never re-reads bytes.

| Rank | Split | DRAM crossings (x / out) | Transactions | Cross-core |
|------|-------|--------------------------|--------------|------------|
| 1 | closed cells `(n, at, bt, wt)` across cores | 1 / 1 at tile granularity (padding rows of ragged tile rows included) | 1 per tile, read and write | 0 |
| 1 | a-split inside a cell | 1 / 1, out at live-row granularity | reads 1 per tile; writes 2 per output tile per a-group | 0 |
| 1 | b-split inside a cell | 1 / 1, x at live-row granularity | reads 2 per input tile per b-group; writes 1 per tile (or 2 runs at `block_a = 1`) | 0 |
| 2 | X5 row forwarding | 1 / 1 | as b-split + unicasts | all live bytes once more |
| 3 | X1 whole-tile re-reads | ≤ 32 / 1 | — | 0 |

All rank-1 splits cross each live row exactly once, so bytes do not separate them except for the padding rows of a ragged tile row (whole-tile transfers move them). The choice among them is the occupancy step: closed cells while they fill the grid (fewest transactions: whole-page transfers, `tile_reorder` catalog entry), then a, then b. Chosen: R1 with the selection rule above.

**Stall-shadow check.** The writer waits on `cb_input_tiles`; the reader waits on its read barrier. Nothing independent of the awaited data remains on a core with one block (every write needs the block's reads). With ≥ 2 blocks, the previous block's writes drain in hardware during the wait (flush placement below), and the reader's depth-2 CB lets block k+1's reads run under block k's regroup. The op being a pure move (row moves commute) permits starting the regroup on the first landed tile of a block instead of the whole block: recorded as the overlap lamp, not built in Phase 0 because no production shape has more than one input tile per block in the regroup path.

### Block schedule

```cpp
// per core: blocks [start_block, start_block + num_blocks_this_core)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(start_block + block_idx);     // reader : block's input tile rows -> cb_input_tiles
    regroup_block(start_block + block_idx);  // writer : row moves cb_input_tiles -> cb_output_staging (identity at block_a == 1)
    store_block(start_block + block_idx);    // writer : output tiles / row runs -> DRAM
}
```

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_block` | reader (NCRISC, NoC0) | `block_a × block_wt` input tiles; per tile the rows `[b0%32, b0%32 + live_b)`: one whole-tile read when they cover the tile's live rows `[0, lb)`, else 2 row runs (one per face half) at tile-relative offsets | the block's slots in `cb_input_tiles` until the writer pops them | 1 `cb_reserve_back` + 1 `noc_async_read_barrier` + 1 `cb_push_back` per block; all reads of a block issued before the one barrier. Ragged group (fewer live `a` than `block_a`): reads only live tiles, pushes the nominal `block_a·block_wt` pages |
| `regroup_block` | writer (BRISC, NoC1) | `live_a × live_b × block_wt × 2` copies of 32 B: source `slot(a, wt) + face_off(rb, h)` → destination `staging(b, wt) + face_off(ra, h)`; only live rows (`a < A`, `b < B`) | input slots (read), staging slots (written) | `set_read_state<32>` once per kernel; 1 `noc_async_read_barrier` per block, then 1 `cb_pop_front`. Skipped entirely when `block_a = 1` |
| `store_block` | writer | `block_b × block_wt` output tiles. Regroup path: whole-tile write from staging when the a-span covers `[0, la)`, else 2 row runs `[a0%32, a0%32 + live_a)` per tile. Direct path (`block_a = 1`): 2 runs of 32 B per output tile from `slot(a0, wt) + face_off(rb, h)` to `face_off(ra, h)` | staging slot (regroup) or input slot (direct) until flushed | Regroup path: 1 `noc_async_writes_flushed` per block placed **before** issuing this block's writes (it retires the previous block's writes, which drained during this regroup). Direct path: 1 flush after this block's writes, then 1 `cb_pop_front` (the writes read the input slot). 1 `noc_async_write_barrier` per kernel at the end (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (target cores) | Qwen plans put 16–96 cores on 4–6 transfers each; one DRAM round trip per core dominates, so beyond some count more cores add launch fan-out without shortening the critical path | `max_cores` = G vs G/2, G/4 on the 8 Qwen shapes (judge by unprofiled trace replay, lessons RB-053) |
| Split order a-before-b | 303/304 end at `(1, 2)`; `(2, 1)` trades 2 output runs for 2 copies + 2 more reads | b-first walk on 303, 304, 307, 308 |
| Overlap: block-granular handoff | closed cells push 32 tiles at once; the regroup waits for all of them | push `cb_input_tiles` per tile (or per k tiles) inside a block so the regroup starts on the first landed tile (pure move: row order is free) |
| Regroup issue rate | a closed cell is `2·la·lb` (≤ 2048) self-reads of 32 B on one RISC-V; the reader issues only `la` reads (`split_reader` catalog entry) | split the row moves of a block between reader and writer RISC-Vs (the staging buffer then needs a reader→writer handoff) |
| Whole-tile vs live-row reads for padded tile rows | closed cells read the ragged last tile row whole (e.g. `lb ≤ 16`: half the bytes are padding) | 2 row-run reads when `lb ≤ 16` |
| Bank stagger | every core walks its blocks with `wt` innermost from a start that may share a DRAM bank with other cores (`bank_stagger` catalog entry) | rotate each core's start inside its range |
| Barrier granularity on multi-block cores | the reader drains all reads per block (`trid_double_issue` catalog entry) | trid-scoped barrier per block with 2 blocks in flight |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x in DRAM | bfloat16 tiles, interleaved, page = tile (2048 B) | — | padding rows `rb ≥ lb` hold anything |
| `load_block` | tile slots of `cb_input_tiles` (whole tiles or row runs at tile-relative offsets) | reader: `noc_async_read` from `TensorAccessor::get_noc_addr(page, offset)` | mod-64 rule met by identical tile-relative offsets |
| `regroup_block` | tile slots of `cb_output_staging` | writer: self-aimed 32-B NoC reads (`dataflow_kernel_lib::set_read_state` / `read_with_state`) | only when `block_a > 1` |
| `store_block` | output tiles | writer: `noc_async_write` to `get_noc_addr(page, offset)`: whole tile or 32·k-B runs | writes may target the same output tile from several cores, always disjoint rows |
| out in DRAM | bfloat16 tiles, interleaved | — | output padding rows unspecified |

No compute kernel: no TRISC binary, no `compute_kernel_hw_startup`, no format conversion. Bytes move only through the NoC, so special values, -0 and denormals keep their bits.

**Placement axis.** TARGET's placement axis is `buffer ∈ {dram, l1}`, both interleaved; no sharded value exists, so no logical shard needs a physical realization. Per TARGET value:

| TARGET value outside SUPPORTED | Class for R1 | Why |
|-------------------------------|--------------|-----|
| `buffer: l1` | knob-turn | `TensorAccessor` already abstracts the bank type; L1 alignment (mod 16) is looser than DRAM's; the output's own L1 pages count with the op's CBs against 1347776 B |
| `alignment: tile_aligned`, `w_non_aligned` | knob-turn | B % 32 = 0 only means `lb = 32`; a non-aligned W only pads columns, which move with their row |
| `dtype: bfloat8_b` | scheme-change (R4 for (0,3,1,2) needs a compute kernel; R6 for (0,2,1,3) needs byte moves) | block-float groups and exponent bytes |
| `layout: ROW_MAJOR`, `rank: 6`, `dims: (0,1,4,2,5,3)`, `(0,3,1,2)` | scheme-change (new index maps: R2, R3, R4) | each dims value is its own index map and kernel pair |

When a refinement adds an axis value, cartesian cells of SUPPORTED that no production call reaches (e.g. ROW_MAJOR with (0,2,1,3)) are either built or listed in EXCLUSIONS; production cases are never excluded.

**Reachability of the deferred regimes.** Each regime is a `(reader, writer[, compute])` kernel set plus a `select_plan` branch; the host machinery is shared: block grid → `split_work_to_cores` → per-core `(start_block, num_blocks)`, cached descriptor, `describe_plan`. R4 reuses R1's regroup and store operations unchanged after a compute transpose. No deferred regime needs cross-core communication.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block (Blocking Model) |
| Grid | `G = grid.x · grid.y` from `device.compute_with_storage_grid_size()`; `max_cores = G` |
| Split | `ttnn.split_work_to_cores(grid, num_blocks, row_wise=True)` (row-wise spread across the DRAM-facing axis, `noc_placement` catalog entry); `num_cores = min(num_blocks, G)` |
| Per-core work | contiguous block range `[start_block, start_block + num_blocks_this_core)`; group 1 gets `q+1` blocks, group 2 `q` |
| Remainder | `NA = ceil(A/block_a)`, `NBg = ceil(B/block_b)`, `NW = ceil(Wt/block_wt)`, `Bt = ceil(B/32)`, `At = ceil(A/32)`, `Wt = ceil(W/32)`, all per image; ragged groups carry their live count, CB counts stay nominal |
| Kernel groups | both kernels on `all_cores` (one or two rectangles for any `num_blocks`, BB-17); cores outside get no kernel |
| Runtime args per core | input address, output address, `start_block`, `num_blocks_this_core` |

**Host caching (BB-07, BB-16).** The program descriptor for a key `(input logical shape, input padded shape, canonical dims, dtype, layout, buffer type, memory layout, grid size)` is built once and stored in a module-level dict; a repeat call patches only the two address runtime args and calls `generic_op`. `ProgramDescriptor.custom_program_hash` (bound at `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1141`; used in place of the structural hash at `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:79-80`) is set from the same key, so every value that changes the program (dims, shapes, derived knobs) is in it, and a call with other dims never replays this program (risks.md R10). `describe_plan` reads the same cached plan.

**Regime-pinned coverage.** Phase 0 has one regime; its knob-dependent paths (regroup vs direct, whole vs row-run transfers) depend on G. The acceptance shapes pin each path for any G in [64, 139] (every Wormhole and Blackhole worker grid): `(3,40,50,512)` 192 closed cells → regroup, whole-tile reads/writes, ragged `at`, `bt` with rows ≥ 16; `(1,32,20,640)` 20 cells → regroup with a-split runs (`block_a` 16, 8 or 4); `(1,70,3,64)` → regroup with 2- or 4-row runs at `at > 0`; Qwen shapes → direct path with row-run reads; `(1,5,47,32)` → row-run reads (`block_b` 4 or 2) in the row-16..31 face band; `(2,3,5,96)` → direct path with N > 1, Wt = 3.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `tile_bytes` (2048) | `depth_in · block_a · block_wt` | spans `a` (`block_a` slots) and `wt` (`block_wt`); `b` lives as rows inside each whole-tile slot (tile-relative offsets for the mod-64 rule); streams over `n` and blocks | Float16_b | reader | writer | `load_block` push → writer pop after the regroup barrier (regroup path) or after the store flush (direct path) |
| `cb_output_staging` | 1 | `tile_bytes` (2048) | `depth_staging · block_b · block_wt`; allocated only when `block_a > 1` | spans `b` (`block_b` slots) and `wt`; `a` lives as rows inside each slot; streams over `n` and blocks | Float16_b | writer | writer (private scratch: base address from `get_write_ptr` once; no reserve/push/wait/pop; slot `k % depth_staging` reused only after the flush that retires block k−depth_staging's writes) | `regroup_block` → `store_block` of one block |

Push count = wait count: `cb_input_tiles` is pushed and waited `block_a · block_wt` pages per block on every core, ragged or not; capacity is a whole multiple of that quantum, so the ring closes.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_a × block_wt` tiles, `block_b` rows each | raw NoC + TensorAccessor (helpers rejected below) | x (DRAM) | `cb_input_tiles`, `block_a·block_wt` pages reserved | pushed |
| 2 | `regroup_block` | `live_a × live_b × block_wt × 2` face-row copies | helper `local_copy_helpers_dataflow` | `cb_input_tiles`, `block_a·block_wt` pages waited | `cb_output_staging`, `block_b·block_wt` slots (scratch) | `cb_input_tiles` popped after the read barrier |
| 3 | `store_block` | `block_b × block_wt` output tiles (whole or 2 runs each) | raw NoC + TensorAccessor | staging slots (regroup) or `cb_input_tiles` slots (direct, still waited) | out (DRAM) | direct path: `cb_input_tiles` popped after the flush |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `cb_reserve_back`, `get_write_ptr`, `cb_push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406`, `:325`, `:208` | `(cb_input_tiles, block_a·block_wt)` | — | `cb_input_tiles` | page count = `block_a·block_wt` |
| `load_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `in_page(n,a,bt,wt)`, `offset = face_off(rb0, h)` or 0 | — | — | — |
| `load_block` | raw_api | `noc_async_read(src_noc_addr, dst_l1, size)`; `noc_async_read_barrier()` | `dataflow_api.h:547`, `:1921` | size = 2048 (whole tile) or `live_b·32` (row run) | x | `cb_input_tiles` | `block_b` sets the run length |
| `regroup_block` | helper | `dataflow_kernel_lib::set_read_state<face_row_bytes>(noc, src)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-75` | `face_row_bytes = 32`; `noc` = the writer's NoC object (coordinates via `local_addr(addr, noc.get_noc_id())`, `.inl:30-34`) | — | — | — |
| `regroup_block` | helper | `dataflow_kernel_lib::read_with_state<face_row_bytes>(noc, dst_addr, src_addr)` | `local_copy_helpers_dataflow.hpp:94-95`, impl `.inl:72-77` | per copy: `dst = staging(b,wt) + face_off(ra,h)`, `src = slot(a,wt) + face_off(rb,h)` | `cb_input_tiles` | `cb_output_staging` | loop extents `block_a`, `block_b`, `block_wt` |
| `regroup_block` | raw_api | `noc_async_read_barrier()`; `cb_wait_front`, `get_read_ptr`, `cb_pop_front` | `dataflow_api.h:1921`, `:479`, `:346`, `:260` | writer's NoC | `cb_input_tiles` | — | pop count `block_a·block_wt` |
| `store_block` | raw_api | `noc_async_write(src_l1, dst_noc_addr, size)` with `TensorAccessor::get_noc_addr(out_page, offset)` | `dataflow_api.h:876`, `tensor_accessor.h:165` | size 2048 (whole) or `live_a·32` (run) or 32 (direct) | staging / `cb_input_tiles` | out | `block_a` decides whole vs run vs direct |
| `store_block` | raw_api | `noc_async_writes_flushed()`; `noc_async_write_barrier()` | `dataflow_api.h:1980`, `:1951` | once per block; once per kernel | — | — | — |

**`read_with_state` with a varying source.** The helper's doc says the source should equal the programmed address (`local_copy_helpers_dataflow.hpp:92`), but the state it programs is the NoC coordinates and the size; every issue rewrites both addresses (`ncrisc_noc_read_with_state` writes `NOC_RET_ADDR_LO` and `NOC_TARG_ADDR_LO` per call, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:1508-1509`). A varying source inside the own core is therefore valid. If the implementer keeps to the documented contract instead, the same copy is `noc.async_read(UnicastEndpoint{}, ..., local_addr(src, noc.get_noc_id()), ...)` per copy (`.inl:17-28`) at a higher issue cost; measure it under the regroup-issue-rate lamp.

**Helpers considered and rejected for the raw_api rows (`load_block`, `store_block`, CB sync):**

| Candidate | File:Line | Why it cannot express the DRAM page / row-run transfer |
|-----------|-----------|--------------------------------------------------------|
| `local_copy_helpers_dataflow` (`local_addr`, `set_read_state`, `read_with_state`) | `local_copy_helpers_dataflow.inl:30-34`, `:62-70` | the source endpoint is always this core's own L1 (`my_x/my_y`); it cannot address a DRAM bank, and it has no write side |
| `l1_helpers` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:25-101` | address conversion, zero-fill and byte fills inside L1 only |
| `tile_column_dataflow` (`copy_tile_column`) | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:81-92` | RISC-V element stores of one tile column inside L1; R1 moves whole rows over the NoC |
| `dfb_helpers_dataflow` | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp:14-18` | tile-dimension queries only |

No kernel_lib dataflow helper issues interleaved DRAM page or sub-page transfers; the gap is a "row-run transfer" helper (`for_each_row_run(tile, r0, r1, live_rows, fn(offset, bytes))`) that the implementer writes once in a shared kernel header and uses in both kernels. No compute helper applies: the op has no compute phase (X4).

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Whole-tile write in the direct path | at `block_a = 1` the a-span often covers all live rows (`la = 1`, case 301), which looks like the whole-tile condition, but the input slot holds rows at input positions | the direct path always writes 2 runs per output tile; the whole-tile write exists only in the regroup path from staging |
| Popping `cb_input_tiles` before the direct writes left L1 | the reader refills the slot while the NoC still reads it: values from the next block, varying run to run (BB-06) | direct path: `noc_async_writes_flushed()` before `cb_pop_front` |
| Reusing a staging slot before its writes left L1 | same corruption in the regroup path | flush before issuing block k's writes retires block k−1; with `depth_staging = 2` slot k%2 was last written by block k−2, retired at block k−1 |
| Input padding reaching a live output row (risks.md R4) | whole-tile reads bring padding rows `rb ≥ lb` into L1 | regroup and direct writes iterate only `b < B` and `a < A`; acceptance test poisons padding with `inf` on regroup and direct shapes |
| Rows ≥ 16 in faces 2/3 | `face_off` adds 1024 B for the lower band; shapes with `B % 32 > 16` or A ≥ 16 hit it | single `face_off` definition shared by both kernels; acceptance shapes `(1,5,47,32)`, `(3,40,50,512)` cover it |
| mod-64 DRAM reads of row runs | a run landing at a different tile-relative offset than its source breaks for odd rows | runs land at the source's tile-relative offset in a whole-tile slot (Mechanism caps) |
| Self-read coordinates on the writer's NoC | NoC0 and NoC1 have different coordinate spaces; the writer runs on NoC1 | `local_addr(addr, noc.get_noc_id())`, never `my_x[0]` |
| Incomplete program-cache key (risks.md R10) | same shape, other dims would replay this index map once more maps are built | `custom_program_hash` from the full key (Work Distribution); acceptance test asserts a repeat call adds no cache entry and the result is right at new addresses |
| Plan depends on the grid | knob values (hence paths) differ between a 110- and a 130-core device | `describe_plan` reports the plan (BB-08); acceptance shapes pin every path for G in [64, 139] |
| Tiny programs, many calls (Qwen 2176 calls/step) | fixed per-program cost dominates; scattered kernel groups start late in traces | one kernel group on ≤ 2 rectangles (BB-17); no compute kernel and two small data-movement kernels keep kernel text small (BB-18) |
| Profiling zones (BB-21) | time outside a zone reads as waiting | one named zone per block operation and per blocking wait (`cb_wait_front`, read barrier, flush, final write barrier), compiled only in the zone build; `describe_plan["cores"]` dumps each core's role and block range |
