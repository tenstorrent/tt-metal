# L1 Ledger: tilize

Schema and audits: `.claude/references/l1-footprint-discipline.md`.
Block semantics and axis names: `op_design.md` → Blocking Model.

Named block axes (all four appear in every row): `leading`, `tile_row`, `tile_col`,
`within_tile`.

## The buffer table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_rows` | `input_depth_rows * block_width_tiles` = `2 * block_width_tiles` | `block_width_tiles` pages — one tile-row of the block. `compute_kernel_lib::tilize` waits for and pops exactly `block_width_tiles` pages per iteration (`tilize_helpers.inl:233-259`) and `read_sticks_for_tilize` reserves/pushes exactly that many per tile-row (`tilize_helpers_dataflow.inl:110-127`), so one tile-row is the peak simultaneously-resident set. | `{leading: streams -> the block's row range walks images (never resident), tile_row: streams -> window = input_depth_rows tile-rows, tile_col: spans -> block_width_tiles pages, within_tile: spans -> one page IS one tile's row-major bytes (tile_h sticks x 32 elements, at an L1 stride of block_width_tiles*32*elem)}` | `input_tensor.dtype`. Phase 0 `Float16_b`. **Not `Float32` under 16-bit DEST**: `fp32_dest_acc_en` is set **iff** `input_tensor.dtype == float32`, so the page width and the DEST width move together by construction. This CB carries the *tensor's* bytes, not accumulator values — `tilize` performs no arithmetic, so there is no accumulation width to widen past the tensor's own format. | `reader` | `compute` | whole program (every block) | **Cannot share with `cb_output_tiles`.** Three independent reasons, any one sufficient: (1) **concurrent lifetime** — the two are pipelined against each other within a block (compute reads one while writing the other), and `compute_kernel_lib::tilize` `static_assert`s `input_dfb != output_dfb` (`tilize_helpers.inl:98-99`) precisely because tilize is not an in-place permutation; (2) **differing page format** whenever `dtype=` requests a cast (Phase 0 they match, but the sharing decision must hold across the dtype refinements); (3) **differing tile-descriptor semantics** — this page is row-major bytes, the other is a 4-face tile. Rule 3 patterns 1 and 2 were both attempted and are foreclosed by (1). |
| `cb_output_tiles` | `output_depth_batches * write_rows_per_barrier * block_width_tiles` = `2 * write_rows_per_barrier * block_width_tiles` | `write_rows_per_barrier * block_width_tiles` pages — the batch of whole-tile-page writes in flight behind one `noc_async_write_barrier`. The second batch of capacity is compute's overlap window, not part of the live set. **IMPLEMENTED capacity is `2 * wrpb * W`, not the design's `(wrpb + 1) * W`** — see "Deviations" below: the capacity must be an exact multiple of the write-batch quantum or a full batch straddles the FIFO wrap, which both CB endpoints refuse. | `{leading: streams -> not resident, tile_row: spans -> a write_rows_per_barrier-deep window (this is the one axis whose live set genuinely spans more than one unit, and capacity scales with it), tile_col: spans -> block_width_tiles pages, within_tile: spans -> one page IS one output tile}` | `dtype` if given else `input_tensor.dtype`. Phase 0 `Float16_b`. Same DEST-width argument as above; `output_tensor.buffer_page_size()` is used verbatim so block-float (`Bfp8_b` = 1088 B, `Bfp4_b`) and tiny-tile page sizes are exact rather than computed. This is the CB whose format performs the value-preserving cast at pack time. | `compute` | `writer` | whole program (every block) | **Cannot share with `cb_input_rows`** — same three reasons, stated from this side. In particular the pipelining is an explicit design decision (the two depth knobs exist to make the stages overlap), which per Rule 3 pattern 3 is exactly the reason that must be *recorded* rather than left as an unexplained non-reuse. |

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
| `block_width_tiles` | `1 <= block_width_tiles <= min(255, W_FIT)` | `W_FIT = clamp((budget - WRITE_BATCH_MIN_TILES*tb_out) // (2*tb_in + 2*tb_out), 1, FAST_TILIZE_WIDTH_CAP)` with `budget = ttnn.get_max_worker_l1_unreserved_size()` and `FAST_TILIZE_WIDTH_CAP = 255` from `can_use_fast_tilize`'s `block_width_tiles < 256` (`tilize_helpers.inl:77`). Under `low_l1=True` the additional cap `LOW_L1_WIDTH_CAP` (a host constant, independent of every tensor dimension) applies. `block_width_tiles` is the coarsest **divisor of `C`** that is `<= min(W_CAP, C / num_w_chunks_target)` with `num_w_chunks_target >= ceil(C / W_CAP)`, so the bound holds by construction. (The divisor constraint replaces the design's `ceil` — see "Deviations".) |
| `block_width_tail_tiles` | **does not exist in the implementation** | `block_width_tiles` divides `C` exactly, so `num_w_chunks = C / block_width_tiles` and every w-chunk is exactly `block_width_tiles` wide. There is no ragged column tail, hence no second CT instantiation of `compute_kernel_lib::tilize` and no second CB quantum. Forced by the wrap requirement — see "Deviations". |
| `write_rows_per_barrier` | `1 <= write_rows_per_barrier <= WRITE_BATCH_MIN_TILES = 4` | `= max(1, ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))`; since `block_width_tiles >= 1`, the ceiling is at most `WRITE_BATCH_MIN_TILES`. |
| `input_depth_rows` | `= 2` (Phase 0); `>= 2` in general. **Measured across {1,2,3,4} and flat at every value** on both shapes the design's overlap lamp names — kept at 2 as the smallest value that overlaps at all and the cheapest in L1 of those; still a live knob. Evidence and the bottleneck it implies (DRAM-bandwidth-bound, ~183 GB/s on `[1,1,2048,2048]`) are in `tilize_program_descriptor.INPUT_DEPTH_ROWS`. | Fixed host constant. Lower bound 2 is required by the overlap it buys and by `read_sticks_for_tilize`'s capacity assert `width_in_tiles <= cb_capacity` (`tilize_helpers_dataflow.inl:105-107`) plus `compute_kernel_lib::tilize`'s `get_dfb_num_pages(input_dfb) >= block_width_tiles` (`tilize_helpers.inl:220-222`). |
| `output_depth_batches` | `= 2` (a host constant) | Depth measured in WRITE BATCHES rather than tile-rows. Lower bound 2 is what buys the compute-side overlap window; being an integer count of batches is what keeps the capacity an exact multiple of the batch quantum (the wrap requirement below). Capacity in tile-rows is therefore `2 * write_rows_per_barrier`, bounded by `2 * WRITE_BATCH_MIN_TILES = 8`. |
| `tb_in` | `= tile_h * 32 * element_size(in_dtype)`; `<= 32*32*4 = 4096` B | `tile_h <= 32` (above) and `element_size <= 4` over `TARGET["dtype"]` (widest is `uint32`/`int32`/`float32`). |
| `tb_out` | `= output_tensor.buffer_page_size()`; `<= 32*32*4 = 4096` B | Same: `tile_h <= 32`, and the widest format in `TARGET["output_dtype"]` is 4 B/element. Block-float outputs are *smaller* (`Bfp8_b` = 1088 B for a 32x32 tile). |
| `WRITE_BATCH_MIN_TILES` | `= 4`, a host constant | Named constant, single source. **MEASURED on device, not taken from the catalog:** the design's lamp asked whether 8 sits past the knee and it does. `[1,1,16384,32]` (C == 1, so this constant *is* `write_rows_per_barrier`), median device kernel ns over 3 fresh-cache runs at 64/64 cores — wb=1 **24430** (the one-write-per-barrier trap), wb=2 22519, wb=4 **20723**, wb=8 22568, wb=16 22403. The plateau is at 4, which is exactly where `double_buffer/report.md` put it; 4 is 1.18x over the trap and 1.09x over the design's 8, and costs *less* L1. Harness: `tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_write_batch.py`. Not a tensor dimension. |
| `FAST_TILIZE_WIDTH_CAP` | `= 255`, a host constant | `can_use_fast_tilize` requires `block_width_tiles < 256` (`tilize_helpers.inl:77`). |
| `LOW_L1_WIDTH_CAP` | `= 4`, a host constant | Named constant, single source, **independent of every tensor dimension** — that independence is what makes the `low_l1=True` contract structural rather than a percentage. Yields 40 KB at bf16 / 80 KB at fp32 regardless of `W`. |
| `budget` | device-reported, `> 0` | `ttnn.get_max_worker_l1_unreserved_size()` (`ttnn/ttnn/device.py:20`) — read at runtime, never a literal. |
| `block_row_extent` | `1 <= block_row_extent <= R` | **Appears in no capacity expression.** It is the runtime `num_blocks` handed to `compute_kernel_lib::tilize`, which processes one tile-row per iteration with its own wait/push per iteration (`tilize_helpers.inl:233-259`), so it is an axis both CBs *stream over* — never one either spans. This is the reason the coarsest row extent (the core's whole row-group) is taken unconditionally: it costs zero L1. |
| `R`, `C`, `num_row_groups`, `num_w_chunks`, `num_images`, `rows_per_image` | tensor-derived, unbounded in principle | **None of these appears in any capacity expression.** They size the *work split*, not the footprint. Their absence from the table above is the whole `low_l1` proof. |

## Total per-core footprint

```
L1_per_core = input_depth_rows     * block_width_tiles * tb_in
            + output_depth_batches * write_rows_per_barrier * block_width_tiles * tb_out

            = 2 * block_width_tiles * tb_in
            + 2 * write_rows_per_barrier * block_width_tiles * tb_out
```

Closed-form upper bound, used as the `W_FIT` solve. Since
`block_width_tiles * write_rows_per_barrier <= block_width_tiles * (WRITE_BATCH_MIN_TILES/block_width_tiles + 1)
= WRITE_BATCH_MIN_TILES + block_width_tiles`:

```
L1_per_core <= 2*block_width_tiles*tb_in + 2*(block_width_tiles + WRITE_BATCH_MIN_TILES)*tb_out
```

which inverts to
`W_FIT = (budget - 2*WRITE_BATCH_MIN_TILES*tb_out) // (2*tb_in + 2*tb_out)`,
clamped into `[1, FAST_TILIZE_WIDTH_CAP]`. This is the expression
`tilize_program_descriptor.derive_plan` evaluates verbatim; the only change from
the design's form is the `2*` on the output depth (batches, not `wrpb + 1`) —
see "Deviations".

**Which terms scale with which knob.**

| Term | Scales with | Does **not** scale with |
|------|-------------|-------------------------|
| `2 * block_width_tiles * tb_in` | `block_width_tiles` (the `tile_col` extent), `input_depth_rows`, `tile_h`, `element_size(in_dtype)` | `block_row_extent`, `R`, `C`, `H`, `W`, `rank`, `num_images`, core count |
| `2 * write_rows_per_barrier * block_width_tiles * tb_out` | `block_width_tiles`, `write_rows_per_barrier`, `tile_h`, output format width | the same list |

**No tensor dimension appears in the total, at either `low_l1` setting.** That is the
whole `low_l1` obligation discharged structurally rather than by a percentage:
`block_width_tiles <= W_FIT` is a constant of `(dtype, tile_h, device)`, and
`low_l1=True` merely lowers the cap to `LOW_L1_WIDTH_CAP`. Consequently
`low_l1_forcing_width` (`[1,1,32,8192]`, C=256) cannot OOM at either setting: at
float32 with `low_l1=False`, `num_w_chunks = 64` for occupancy so
`block_width_tiles = 4`, and the footprint is 64 KB — an order of magnitude inside
the budget, while a full-width block would have been 1 MB per side.

Worked footprints, **read out of the built plan** on an 8x8 Wormhole grid
(`budget = 1464128` B, giving `W_FIT = 176` at bf16 and `87` at fp32 — both far
above what occupancy allows on any of these shapes, which is the point: L1 is not
the binding constraint here, the grid is). `bw` is `block_width_tiles`, `wrpb` is
`write_rows_per_barrier`; the L1 column is `TilizePlan.l1_per_core_bytes`, i.e.
the number the code actually allocates, not a hand recomputation:

| Shape | `bw` | `wrpb` | cores reached | `L1_per_core` |
|-------|------|--------|---------------|---------------|
| `[1,1,32,32]` single tile | 1 | 4 | 1 (only 1 tile exists) | **20 KB** |
| `[1,1,2048,2048]` square_large | 64 | 1 | 64 / 64 | **512 KB** |
| `[1,1,1024,1024]` square_large | 16 | 1 | 64 / 64 | **128 KB** |
| `[1,1,32,16384]` **perf focus** | 8 | 1 | 64 / 64 | **64 KB** |
| `[1,1,32,32768]` | 16 | 1 | 64 / 64 | **128 KB** |
| `[1,1,32,2048]` short_wide | 1 | 4 | 64 / 64 | **20 KB** |
| `[1,1,2048,64]` tall_narrow | 2 | 2 | 64 / 64 | **24 KB** |
| `[1,1,16384,32]` tall_narrow | 1 | 4 | 64 / 64 | **20 KB** |
| `[1,1,32,8192]` fp32, `low_l1=False` | 4 | 1 | 64 / 64 | **64 KB** |
| `[1,1,32,8192]` fp32, `low_l1=True` | `min(4, LOW_L1_WIDTH_CAP)` = 4 | 1 | 64 / 64 | **64 KB**, and independent of `W` |

The `low_l1` pair is the load-bearing row: at `C = 256` the two settings agree on
every single number, because `low_l1=False` was *already* dimension-independent —
which is also why `helpers.run_tilize`'s bit-identity A/B is satisfied by
construction rather than by care. The harness that produced this table is
`tests/ttnn/unit_tests/operations/tilize/probes/probe_002.py`.
| `[1,1,1,50304]` bf16, `low_l1=True` | `LOW_L1_WIDTH_CAP` = 4 (393 w-chunks, ~7 blocks/core) | 2 | **40 KB** — the widest grid in the suite, at the same footprint as the narrowest |

## Audit self-check

1. **Capacity vs live set, both directions.**
   *Over:* `cb_input_rows` capacity (`2*W`) exceeds its live set (`W`) — accounted:
   double buffering, the reader/compute overlap the catalog measures at 1.24x–1.99x.
   `cb_output_tiles` capacity (`2*wrpb*W`) exceeds its live set (`wrpb*W`) —
   accounted twice over: one whole batch of compute overlap while the previous
   batch drains, AND the wrap requirement, which needs the capacity to be an
   integer number of batch quanta (a `(wrpb+1)*W` capacity is not, and a full
   batch would straddle the FIFO limit — see "Deviations").
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
   `{input_depth_rows, output_depth_batches, write_rows_per_barrier,
   block_width_tiles, tb_in, tb_out}`, and the
   `W_FIT` solve is the *inversion* of that closed form — a bounded extent selection
   over a stated expression, not a search that settled for a finer block. The
   inventory was minimized first (two CBs, the thread-boundary floor, with the
   in-place candidate ruled out by citation) before any budget expression was written.
5. **Every capacity is a multiple of its transfer quantum.** Added by the
   implementation, because it is the audit the two CB endpoints enforce at
   runtime. `cb_input_rows` = `2 * W` pages against a `W`-page push/pop quantum;
   `cb_output_tiles` = `2 * wrpb * W` pages against a `wrpb * W`-page pop quantum
   and a `W`-page push quantum. Both hold for every reachable `(W, wrpb)` pair
   because `W` divides `C` exactly, so a single core never mixes two quanta.

## Deviations from `op_design.md`

Two, both recorded at their point of use in `tilize_program_descriptor.py`.

1. **`block_width_tiles` is a DIVISOR of `C`, not `ceil(C / num_w_chunks_target)`;
   `block_width_tail_tiles` therefore does not exist.**
   The ragged column tail violates a hard mechanism cap the design did not list:
   *neither CB endpoint may wrap mid-transfer.* On the producer side
   `llk_push_tiles` does `LLK_ASSERT(remaining >= num_words, "CB push_back:
   fifo_wr_ptr would exceed fifo_limit")` (`llk_io_pack.h`); on the consumer side
   `cb_pop_front` does `ASSERT(fifo_rd_ptr <= fifo_limit)` and wraps *only* on
   exact equality, under the comment "consumer always reads from contiguous
   memory, it cannot wrap" (`dataflow_api.h:267-272`). Together these require the
   CB's page count to be an exact multiple of **every** push/pop quantum used on
   that core. A core can legally be handed one full-width block and one tail
   block — `split_work_to_cores` hands out contiguous ranges of a linearization
   that crosses w-chunk boundaries — so a per-core mix of `block_width_tiles` and
   `block_width_tail_tiles` quanta is reachable, and supporting it would require
   sizing the capacity to `lcm(block_width_tiles, tail)` (550 pages at
   `C = 1572`). Constraining the extent to a divisor removes the mix at the cost
   of at most a coarser chunk count, and reproduces the design's entire worked
   table — `[1,1,32,16384]` -> 8, `[1,1,32,32768]` -> 16, `[1,1,64,12288]` -> 12,
   `[1,1,1024,1024]` -> 16, `[1,1,2048,2048]` -> 64, `[1,1,2048,64]` -> 2,
   `[1,1,32,2048]` -> 1, fp32 `[1,1,32,8192]` -> 4 — with one exception,
   `[1,1,1,50304]` (`C = 1572`), which becomes 12 x 131 chunks instead of
   25 x 63. Every geometry still reaches the full grid. The knob remains a live
   tunable at its coarsest correct value.

2. **`cb_output_tiles` depth is counted in write BATCHES (`2 * wrpb * W` pages),
   not tile-rows (`(wrpb + 1) * W`).** Same cap, other half: `(wrpb + 1) * W` is
   not a multiple of the `wrpb * W` pop quantum, so with `wrpb > 1` the read
   pointer walks to a position from which a full batch crosses `fifo_limit`. The
   writer additionally caps each batch by the contiguously readable span as a
   belt-and-braces guard (`tilize_writer.cpp`, "WRAP SAFETY"); with the
   batch-multiple capacity that cap is never the binding one for a full batch.
   Cost: `(wrpb - 1) * W` extra pages, i.e. 3 extra 2 KiB pages on the narrowest
   shape and **zero** on every block at least `WRITE_BATCH_MIN_TILES` wide (where
   `wrpb == 1` and `2*wrpb*W == (wrpb+1)*W`).

Additionally, `WRITE_BATCH_MIN_TILES` is **4**, not the design's 8 — not a
structural deviation but a measured knob value; the evidence is in its symbol-table
row above and the design's own "Write batch depth" lamp is what asked for it.

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

**Where the divisor constraint costs read transactions (verifier-added).** The
minimization above is over a *continuous* chunk count; Deviation 1 restricts
`block_width_tiles` to a **divisor of `C`**, so the realized `num_w_chunks` is
`C / block_width_tiles` and can only *overshoot* the target. On every shape whose
`C` is smooth the overshoot is zero (`C = 512 -> 64`, `1024 -> 64`, `384 -> 32`,
`64 -> 1`, `32 -> 2`: the whole worked table). It is non-zero exactly where `C` has
a large prime factor, and the one such shape in the suite is the logits row
`[1,1,1,50304]`, `C = 1572 = 2^2 * 3 * 131`: the target is 64 chunks, the coarsest
divisor `<= 1572/64 = 24` is **12**, so the split lands on **131** chunks of 12
tiles instead of 63 of 25. DRAM crossings are unchanged (still 1 in / 1 out, the
named-boundary minimum) and cross-core stays 0, but the **read transaction count is
2.08x the minimum-that-fills-the-grid** on that shape, at 768 B per read instead of
1600 B. So the closing claim below is exact on smooth `C` and an overshoot on rough
`C`; the honest form is "the fewest read transactions of any split that fills the
grid **and keeps one push/pop quantum per core**".

The escape, recorded so the perf pass does not have to re-derive it: the quantum
mix is only a problem *within one core*, and a `ProgramDescriptor` can carry two
disjoint core ranges with their own CBs and their own `block_width_tiles` CT arg —
full-width cores and tail cores, neither mixing quanta. That restores the ragged
tail the design specified without weakening the wrap invariant. It is a
work-distribution restructure, not a knob turn, so it is filed as a perf-refinement
lever rather than fixed here.

> Cheapest-traffic split considered: **`tile_row` x `tile_col` with `num_w_chunks`
> minimized** — `in_bytes + out_bytes` across DRAM (the minimum), `0` cross-core, and
> the fewest read transactions of any split that fills the grid subject to the
> one-quantum-per-core wrap invariant. Implemented: **that split**. Nothing deferred
> on traffic grounds; the single rough-`C` overshoot is quantified above and carried
> as a perf lever.

**Reconciled against the built code.** `num_w_chunks = C / block_width_tiles`
*exactly* (the divisor constraint, Deviation 1), so the transaction table above
holds unchanged with `num_w_chunks` no longer a `ceil`. Each stick's column slice
is read once by the one core owning it and each output tile page is written once
by the one core owning it — `block_id -> (row_group, w_chunk)` is a bijection onto
a partition of the `R x C` tile grid, and there is no halo, no shared operand and
no cross-core traffic (zero semaphores and zero multicast are declared on the
`ProgramDescriptor`). So DRAM crossings stay **1 in / 1 out**, the named-boundary
minimum, and the claim is structural: it is visible in the program (two buffers,
three kernels, no semaphores) rather than measured.

Measured device kernel time at the reached core count, for the two entries the
perf gate names (8x8 Wormhole, `WRITE_BATCH_MIN_TILES = 4`, fresh-cache medians):

| Shape | tiles | block | cores | read | ns | achieved GB/s |
|-------|-------|-------|-------|------|-----|---------------|
| `[1,1,32,16384]` **perf focus** | 512 | 1 x 8 | **64 / 64** | 512 B/stick | ~13900 | ~151 |
| `[1,1,16384,32]` transposed pair | 512 | 8 x 1 | **64 / 64** | 64 B/stick | ~21000 | ~100 |

Re-measured by the verification pass on the same box (`--profile`, two dispatches
each, `GenericOpDeviceOperation` rows): perf focus **13106 / 14133 ns** at 64/64
cores (~160 GB/s over the 2 MiB moved), transposed pair **21220 / 20268 ns** at
64/64 (~101 GB/s). Both reproduce the table. For scale, `double_buffer/report.md`
puts an untuned 64-core DRAM->DRAM stream at **190.8 GB/s** (≈ this part's DRAM
peak), so the residual headroom is ~1.19x on the perf-focus geometry and ~1.9x on
its transposed counterpart — the latter being the 64 B stick width, which the
blocking cannot coarsen.

The pair carries an identical tile count (2 MiB moved each way), both reach the
whole grid, and the 1.5x gap is therefore **transaction shape, not occupancy** —
which is exactly what the pair exists to make checkable. The narrow side's 64 B
reads are the tensor's own stick width and are not something the blocking can
coarsen: consecutive ROW_MAJOR sticks live on different DRAM banks, so they are
not coalescable into one transfer.

For completeness, the two splits that move *strictly less* on the read side and why
they are not implemented: `tile_row`-only and `single_core_whole_tensor` both keep
whole-page reads (`R*tile_h` transactions, the floor) at identical byte counts — but
they reach `min(R, num_cores)` and `1` core respectively, which is `1` core on
`short_wide` where `R = 1`. They lose on **occupancy**, not on traffic, and
`width_split/report.md` measures that loss at up to **7.76x** on exactly this geometry
class. Both are `rejected` in the regime table with that reason; no structure is
foreclosed, since `num_w_chunks == 1` *is* the `tile_row`-only parameterization and the
built code produces it whenever it fits and fills the grid.
