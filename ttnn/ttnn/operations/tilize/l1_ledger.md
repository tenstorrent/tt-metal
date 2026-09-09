# L1 Ledger: tilize

Schema and audits: `.claude/references/l1-footprint-discipline.md`.
Block semantics and axis names: `op_design.md` → Blocking Model.

Named block axes (all four appear in every row): `leading`, `tile_row`, `tile_col`,
`within_tile`.

## The buffer table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_rows` | `input_depth_rows * block_width_tiles` = `2 * block_width_tiles` | `block_width_tiles` pages — one tile-row of the block. `compute_kernel_lib::tilize` waits for and pops exactly `block_width_tiles` pages per iteration (`tilize_helpers.inl:233-259`) and `read_sticks_for_tilize` reserves/pushes exactly that many per tile-row (`tilize_helpers_dataflow.inl:110-127`), so one tile-row is the peak simultaneously-resident set. | `{leading: streams -> the block's row range walks images (never resident), tile_row: streams -> window = input_depth_rows tile-rows, tile_col: spans -> block_width_tiles pages, within_tile: spans -> one page IS one tile's row-major bytes (tile_h sticks x 32 elements, at an L1 stride of block_width_tiles*32*elem)}` | `input_tensor.dtype`. Phase 0 `Float16_b`. **Not `Float32` under 16-bit DEST**: `fp32_dest_acc_en` is set **iff** `input_tensor.dtype == float32`, so the page width and the DEST width move together by construction. This CB carries the *tensor's* bytes, not accumulator values — `tilize` performs no arithmetic, so there is no accumulation width to widen past the tensor's own format. | `reader` | `compute` | whole program (every block) | **Cannot share with `cb_output_tiles`.** Three independent reasons, any one sufficient: (1) **concurrent lifetime** — the two are pipelined against each other within a block (compute reads one while writing the other), and `compute_kernel_lib::tilize` `static_assert`s `input_dfb != output_dfb` (`tilize_helpers.inl:98-99`) precisely because tilize is not an in-place permutation; (2) **differing page format** whenever `dtype=` requests a cast (Phase 0 they match, but the sharing decision must hold across the dtype refinements); (3) **differing tile-descriptor semantics** — this page is row-major bytes, the other is a 4-face tile. Rule 3 patterns 1 and 2 were both attempted and are foreclosed by (1). |
| `cb_output_tiles` | `output_depth_rows * block_width_tiles` = `(write_rows_per_barrier + 1) * block_width_tiles` | `write_rows_per_barrier * block_width_tiles` pages — the batch of whole-tile-page writes in flight behind one `noc_async_write_barrier`. The `+1` tile-row of capacity is compute's overlap slot, not part of the live set. | `{leading: streams -> not resident, tile_row: spans -> a write_rows_per_barrier-deep window (this is the one axis whose live set genuinely spans more than one unit, and capacity scales with it), tile_col: spans -> block_width_tiles pages, within_tile: spans -> one page IS one output tile}` | `dtype` if given else `input_tensor.dtype`. Phase 0 `Float16_b`. Same DEST-width argument as above; `output_tensor.buffer_page_size()` is used verbatim so block-float (`Bfp8_b` = 1088 B, `Bfp4_b`) and tiny-tile page sizes are exact rather than computed. This is the CB whose format performs the value-preserving cast at pack time. | `compute` | `writer` | whole program (every block) | **Cannot share with `cb_input_rows`** — same three reasons, stated from this side. In particular the pipelining is an explicit design decision (the two depth knobs exist to make the stages overlap), which per Rule 3 pattern 3 is exactly the reason that must be *recorded* rather than left as an unexplained non-reuse. |

Two CBs is the inventory floor: each crosses a thread boundary (`reader`→`compute`,
`compute`→`writer`), and the one candidate for elimination — an in-place transform —
is forbidden by the helper's `static_assert`. There is **no** scratch buffer, no
intermediate between compute phases (there is only one compute phase), and no scaler,
mask or constant CB.

## Symbol table

Every non-block parameter appearing in a capacity expression, with its bound and the
predicate establishing it.

| Symbol | Bound | Predicate establishing the bound |
|--------|-------|----------------------------------|
| `tile_h` | `{1, 2, 4, 8, 16, 32}`; Phase 0 `= 32` | `tile=` validation: height must be a power-of-two fraction of 32 and width must be 32 (raises `ValueError` otherwise). Phase 0 additionally pins it via `SUPPORTED["tile_height"] = [32]`. |
| `block_width_tiles` | `1 <= block_width_tiles <= min(255, W_FIT)` | `W_FIT = clamp((budget - WRITE_BATCH_MIN_TILES*tb_out) // (2*tb_in + 2*tb_out), 1, FAST_TILIZE_WIDTH_CAP)` with `budget = ttnn.get_max_worker_l1_unreserved_size()` and `FAST_TILIZE_WIDTH_CAP = 255` from `can_use_fast_tilize`'s `block_width_tiles < 256` (`tilize_helpers.inl:77`). Under `low_l1=True` the additional cap `LOW_L1_WIDTH_CAP` (a host constant, independent of every tensor dimension) applies. `block_width_tiles = ceil(C / num_w_chunks_target)` with `num_w_chunks_target >= ceil(C / W_CAP)`, so the bound holds by construction. |
| `block_width_tail_tiles` | `1 <= block_width_tail_tiles <= block_width_tiles` | `= C - (num_w_chunks - 1) * block_width_tiles` with `num_w_chunks = ceil(C / block_width_tiles)`, which makes the tail non-empty and no wider than a full chunk. It does not enter either capacity expression (capacity is sized to the wider `block_width_tiles`). |
| `write_rows_per_barrier` | `1 <= write_rows_per_barrier <= WRITE_BATCH_MIN_TILES = 8` | `= max(1, ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))`; since `block_width_tiles >= 1`, the ceiling is at most `WRITE_BATCH_MIN_TILES`. |
| `input_depth_rows` | `= 2` (Phase 0); `>= 2` in general | Fixed host constant. Lower bound 2 is required by the overlap it buys and by `read_sticks_for_tilize`'s capacity assert `width_in_tiles <= cb_capacity` (`tilize_helpers_dataflow.inl:105-107`) plus `compute_kernel_lib::tilize`'s `get_dfb_num_pages(input_dfb) >= block_width_tiles` (`tilize_helpers.inl:220-222`). |
| `output_depth_rows` | `2 <= output_depth_rows <= 9` | `= write_rows_per_barrier + 1`, and `write_rows_per_barrier <= 8`. |
| `tb_in` | `= tile_h * 32 * element_size(in_dtype)`; `<= 32*32*4 = 4096` B | `tile_h <= 32` (above) and `element_size <= 4` over `TARGET["dtype"]` (widest is `uint32`/`int32`/`float32`). |
| `tb_out` | `= output_tensor.buffer_page_size()`; `<= 32*32*4 = 4096` B | Same: `tile_h <= 32`, and the widest format in `TARGET["output_dtype"]` is 4 B/element. Block-float outputs are *smaller* (`Bfp8_b` = 1088 B for a 32x32 tile). |
| `WRITE_BATCH_MIN_TILES` | `= 8`, a host constant | Named constant, single source. Chosen from `double_buffer/report.md` (plateau at ~4 transactions in flight, no gain past 8, `block=32` measurably worse). Not a tensor dimension. |
| `FAST_TILIZE_WIDTH_CAP` | `= 255`, a host constant | `can_use_fast_tilize` requires `block_width_tiles < 256` (`tilize_helpers.inl:77`). |
| `LOW_L1_WIDTH_CAP` | `= 4`, a host constant | Named constant, single source, **independent of every tensor dimension** — that independence is what makes the `low_l1=True` contract structural rather than a percentage. Yields 40 KB at bf16 / 80 KB at fp32 regardless of `W`. |
| `budget` | device-reported, `> 0` | `ttnn.get_max_worker_l1_unreserved_size()` (`ttnn/ttnn/device.py:20`) — read at runtime, never a literal. |
| `block_row_extent` | `1 <= block_row_extent <= R` | **Appears in no capacity expression.** It is the runtime `num_blocks` handed to `compute_kernel_lib::tilize`, which processes one tile-row per iteration with its own wait/push per iteration (`tilize_helpers.inl:233-259`), so it is an axis both CBs *stream over* — never one either spans. This is the reason the coarsest row extent (the core's whole row-group) is taken unconditionally: it costs zero L1. |
| `R`, `C`, `num_row_groups`, `num_w_chunks`, `num_images`, `rows_per_image` | tensor-derived, unbounded in principle | **None of these appears in any capacity expression.** They size the *work split*, not the footprint. Their absence from the table above is the whole `low_l1` proof. |

## Total per-core footprint

```
L1_per_core = input_depth_rows  * block_width_tiles * tb_in
            + output_depth_rows * block_width_tiles * tb_out

            = 2 * block_width_tiles * tb_in
            + (write_rows_per_barrier + 1) * block_width_tiles * tb_out
```

Closed-form upper bound, used as the `W_FIT` solve. Since
`block_width_tiles * write_rows_per_barrier <= block_width_tiles * (WRITE_BATCH_MIN_TILES/block_width_tiles + 1)
= WRITE_BATCH_MIN_TILES + block_width_tiles`:

```
L1_per_core <= 2*block_width_tiles*tb_in + (2*block_width_tiles + WRITE_BATCH_MIN_TILES)*tb_out
```

which inverts to
`W_FIT = (budget - WRITE_BATCH_MIN_TILES*tb_out) // (2*tb_in + 2*tb_out)`.

**Which terms scale with which knob.**

| Term | Scales with | Does **not** scale with |
|------|-------------|-------------------------|
| `2 * block_width_tiles * tb_in` | `block_width_tiles` (the `tile_col` extent), `input_depth_rows`, `tile_h`, `element_size(in_dtype)` | `block_row_extent`, `R`, `C`, `H`, `W`, `rank`, `num_images`, core count |
| `(write_rows_per_barrier+1) * block_width_tiles * tb_out` | `block_width_tiles`, `write_rows_per_barrier`, `tile_h`, output format width | the same list |

**No tensor dimension appears in the total, at either `low_l1` setting.** That is the
whole `low_l1` obligation discharged structurally rather than by a percentage:
`block_width_tiles <= W_FIT` is a constant of `(dtype, tile_h, device)`, and
`low_l1=True` merely lowers the cap to `LOW_L1_WIDTH_CAP`. Consequently
`low_l1_forcing_width` (`[1,1,32,8192]`, C=256) cannot OOM at either setting: at
float32 with `low_l1=False`, `W_FIT ~ 54`, `num_w_chunks = 64` for occupancy so
`block_width_tiles = 4`, and the footprint is `2*4*4096 + 2*4*4096 = 64 KB` — two
orders of magnitude inside the budget, while a full-width block would have been 1 MB
per side.

Worked footprints at the extremes (bf16 unless noted, `tb_in = tb_out = 2048`):

| Shape | `block_width_tiles` | `write_rows_per_barrier` | `L1_per_core` |
|-------|---------------------|--------------------------|---------------|
| `[1,1,2048,2048]` | 64 | 1 | `2*64*2048 + 2*64*2048` = **512 KB** |
| `[1,1,32,16384]` (perf focus) | 8 | 1 | `2*8*2048 + 2*8*2048` = **64 KB** |
| `[1,1,16384,32]` | 1 | 8 | `2*1*2048 + 9*1*2048` = **22 KB** |
| `[1,1,32,8192]` fp32, `low_l1=False` | 4 | 2 | `2*4*4096 + 3*4*4096` = **80 KB** |
| `[1,1,32,8192]` fp32, `low_l1=True` | `min(4, LOW_L1_WIDTH_CAP)` = 4 | 2 | **80 KB**, and independent of `W` |
| `[1,1,1,50304]` bf16, `low_l1=True` | `LOW_L1_WIDTH_CAP` = 4 (393 w-chunks, ~7 blocks/core) | 2 | **40 KB** — the widest grid in the suite, at the same footprint as the narrowest |

## Audit self-check

1. **Capacity vs live set, both directions.**
   *Over:* `cb_input_rows` capacity (`2*W`) exceeds its live set (`W`) — accounted:
   double buffering, the reader/compute overlap the catalog measures at 1.24x–1.99x.
   `cb_output_tiles` capacity (`(wrpb+1)*W`) exceeds its live set (`wrpb*W`) —
   accounted: one tile-row of compute overlap while the write batch drains.
   *Under:* the only axis either live set `spans` and capacity must scale with is
   `tile_col` (both scale with `block_width_tiles`) and, for `cb_output_tiles`, a
   `write_rows_per_barrier`-deep window of `tile_row` (capacity scales with
   `write_rows_per_barrier`). `block_row_extent` is tagged `streams` in both rows and
   correctly absent from both capacities — verified against the helper's per-tile-row
   wait/push, not assumed.
2. **Page format vs DEST width, both directions.** `fp32_dest_acc_en` is set **iff**
   the input dtype is `float32`, and both CBs carry the tensor's own format. No
   `Float32` page under 16-bit DEST; no 16-bit page under fp32 DEST. `tilize` performs
   no arithmetic, so there is no accumulation to truncate at a phase boundary.
   *(The fp32-output refinement additionally requires `Fp32Mode::Lossless` +
   `UnpackToDestFp32` — a correctness requirement of the LLK path, recorded in
   `op_design.md` → Key Risks, not a page-format finding.)*
3. **Disjoint lifetime, no justification.** No CB pair has disjoint lifetimes — both
   are live for the whole program and deliberately pipelined against each other. Both
   `Shares with / why not` cells are filled with three concrete reasons each, one of
   them a `static_assert` citation.
4. **Bounds and closed form.** Every symbol in the total appears in the symbol table
   with a bound and its predicate. The total is closed-form in
   `{input_depth_rows, output_depth_rows, block_width_tiles, tb_in, tb_out}`, and the
   `W_FIT` solve is the *inversion* of that closed form — a bounded extent selection
   over a stated expression, not a search that settled for a finer block. The
   inventory was minimized first (two CBs, the thread-boundary floor, with the
   in-place candidate ruled out by citation) before any budget expression was written.

---

# Data-movement budget

For the chosen split (`tile_row` x `tile_col`, `num_w_chunks` minimized subject to L1
fit and full occupancy). Named memory boundary: **DRAM**.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (ROW_MAJOR) | **1** | Each stick's column slice is read exactly once, by the one core that owns the block containing it. Block footprints do not overlap (no halo, no stencil) and no operand is shared across blocks (each input byte feeds exactly one output byte), so `blocking-model.md` §4's two re-fetch mechanisms are both absent. The residency decision that sets this: nothing needs to be held across blocks, so nothing is ever re-read. | **0** bytes. Zero semaphores, zero multicast declared. |
| output (TILE) | **1** | Each output tile page is written exactly once, by the one core that owns it. `block_id` → `(row_group, w_chunk)` is a bijection onto a partition of the `R x C` tile grid, so no tile is produced twice. | **0** bytes. |

**Totals per tier.**

| Tier | Bytes |
|------|-------|
| DRAM | `in_bytes + out_bytes` — the named-boundary minimum ("each input crosses once, each output crosses once"). The minimum is *reached*, not merely approached. |
| cross-core NoC | **0** |
| core-local L1 | `2*(in_bytes + out_bytes)` — each CB is written once and read once by its two owners. Not free (L1 bandwidth), but irreducible: the row-major bytes must land in L1 for the unpacker to permute them, and the tiled bytes must land in L1 for the packer to emit them. |

**Transaction shape**, which is what actually discriminates the splits here since the
byte counts are split-invariant:

| Direction | Transactions | Bytes each |
|-----------|--------------|------------|
| read | `R * tile_h * num_w_chunks` | `block_width_tiles * 32 * element_size(in_dtype)` (`= C*32*elem` when `num_w_chunks == 1`, i.e. a whole contiguous DRAM page) |
| write | `R * C` (one per output tile, independent of the split) | `tb_out` (one whole tile page) |

Read transactions are the only term the split moves, which is why `num_w_chunks` is
**minimized** rather than maximized: `max(w_chunks_for_l1, w_chunks_for_occupancy)`
takes the smallest value that both fits L1 and fills the grid, and never more.

> Cheapest-traffic split considered: **`tile_row` x `tile_col` with `num_w_chunks`
> minimized** — `in_bytes + out_bytes` across DRAM (the minimum), `0` cross-core, and
> the fewest read transactions of any split that fills the grid. Implemented: **that
> split**. Nothing deferred on traffic grounds.

For completeness, the two splits that move *strictly less* on the read side and why
they are not implemented: `tile_row`-only and `single_core_whole_tensor` both keep
whole-page reads (`R*tile_h` transactions, the floor) at identical byte counts — but
they reach `min(R, num_cores)` and `1` core respectively, which is `1` core on
`short_wide` where `R = 1`. They lose on **occupancy**, not on traffic, and
`width_split/report.md` measures that loss at up to **7.76x** on exactly this geometry
class. Both are `rejected` in the regime table with that reason; no structure is
foreclosed, since `num_w_chunks == 1` *is* the `tile_row`-only parameterization and the
built code produces it whenever it fits and fills the grid.
