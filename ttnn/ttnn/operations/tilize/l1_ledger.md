# L1 Ledger: tilize

Schema and audits: `.claude/references/l1-footprint-discipline.md`.
Block semantics and axis names: `op_design.md` → Blocking Model.

Named block axes (all four appear in every row): `leading`, `tile_row`, `tile_col`,
`within_tile`.

## The buffer table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_rows` | `input_depth_rows * block_width_tiles` = `2 * block_width_tiles` | `block_width_tiles` pages — one tile-row of the block. `compute_kernel_lib::tilize` waits for and pops exactly `block_width_tiles` pages per iteration (`tilize_helpers.inl:233-259`) and `read_sticks_for_tilize` reserves/pushes exactly that many per tile-row (`tilize_helpers_dataflow.inl:110-127`), so one tile-row is the peak simultaneously-resident set. | `{leading: streams -> the block's row range walks images (never resident), tile_row: streams -> window = input_depth_rows tile-rows, tile_col: spans -> block_width_tiles pages, within_tile: spans -> one page IS one tile's row-major bytes (tile_h sticks x 32 elements, at an L1 stride of block_width_tiles*32*elem)}` | `input_tensor.dtype` — Phase 0 `Float16_b`; **Refinement 5** adds `Float32`, `Fp8_e4m3`, `UInt32`, `Int32`, `UInt16`, `UInt8`. **Never wider than the tensor**: this CB carries the *tensor's* bytes, not accumulator values — `tilize` performs no arithmetic, so there is no accumulation width to widen past the tensor's own format, and the page width therefore tracks `element_size(in_dtype)` exactly (1, 2 or 4 B). The DEST width moves with it rather than independently: `requires_fp32_dest_acc(in, out)` sets `fp32_dest_acc_en` iff the input is fp32, the relay is fp32 -> fp32, or the input is 8-bit integer (the LLK's Int8/UInt8 Src rule) — so a 16-bit DEST is never paired with a page that needs 32. **Refinement 5 also tags this CB** `UnpackToDestMode::UnpackToDestFp32`, and ONLY on the fp32 -> fp32 relay: the tag is what takes the datum to DEST at full fp32 instead of through SrcA's tf32, and on every other pair fast tilize is live and `static_assert`s the tag absent. Capacity-neutral — a tag, not a size. | `reader` | `compute` | whole program (every block) | **Refinement 1 — ZERO-COPY when the input shard IS the block** (`plan.input_native`): the CB is placed on the input buffer via `ttnn.cb_descriptor_from_sharded_tensor`, so its capacity is the shard's own bank size and it costs **no additional L1 at all** (the tensor was already resident). The page size is re-stated as `tb_in` — the shard's own paging is one ROW_MAJOR stick, and `block_width_tiles` sticks-worth of contiguous bytes is exactly one tile-row page group, because `block_width_tiles == shard_cols_tiles` on that path. The reader then issues no NoC read; it marks the block's pages available. | **Cannot share with `cb_output_tiles`.** Three independent reasons, any one sufficient: (1) **concurrent lifetime** — the two are pipelined against each other within a block (compute reads one while writing the other), and `compute_kernel_lib::tilize` `static_assert`s `input_dfb != output_dfb` (`tilize_helpers.inl:98-99`) precisely because tilize is not an in-place permutation; (2) **differing page format** whenever `dtype=` requests a cast (Phase 0 they match, but the sharing decision must hold across the dtype refinements); (3) **differing tile-descriptor semantics** — this page is row-major bytes, the other is a 4-face tile. Rule 3 patterns 1 and 2 were both attempted and are foreclosed by (1). |
| `cb_output_tiles` | `output_depth_batches * write_rows_per_barrier * block_width_tiles` = `2 * write_rows_per_barrier * block_width_tiles` | `write_rows_per_barrier * block_width_tiles` pages — the batch of whole-tile-page writes in flight behind one `noc_async_write_barrier`. The second batch of capacity is compute's overlap window, not part of the live set. **IMPLEMENTED capacity is `2 * wrpb * W`, not the design's `(wrpb + 1) * W`** — see "Deviations" below: the capacity must be an exact multiple of the write-batch quantum or a full batch straddles the FIFO wrap, which both CB endpoints refuse. | `{leading: streams -> not resident, tile_row: spans -> a write_rows_per_barrier-deep window (this is the one axis whose live set genuinely spans more than one unit, and capacity scales with it), tile_col: spans -> block_width_tiles pages, within_tile: spans -> one page IS one output tile}` | `dtype` if given else `input_tensor.dtype`. Phase 0 `Float16_b`; **Refinement 5** adds `Float32`, `Bfp8_b`, `Bfp4_b` and the four integer widths. `output_tensor.buffer_page_size()` is used verbatim so block-float (`Bfp8_b` = 1088 B, `Bfp4_b` = 576 B for a 32x32 tile) and tiny-tile page sizes are exact rather than computed. This is the CB whose format performs the value-preserving cast at pack time. **DECIDED, NOT INHERITED — the two CBs' page widths are now INDEPENDENT.** On the widening diagonal (`bf16 in -> fp32 out`) this page is 4096 B against a 2048 B input page: it doubles the bytes through the packer, doubles them again in every write of this CB, and doubles this CB's L1 — even though the DEST value is already the tensor's own bf16 value, widened losslessly at pack. It is accepted rather than avoided because the alternative is not available: the output page size IS the output tensor's, and `dtype=float32` is the caller asking for exactly those bytes. What it must NOT do is leak into the *solve*, and it does not — `W_FIT`'s denominator uses `tb_out` (this page) and `tb_in` (the input page) separately (`derive_plan`), so a widened output narrows `block_width_tiles` instead of overrunning the budget. The narrowing diagonal is the mirror image and strictly cheaper: `fp32 in -> bfp4 out` is 4096 B in against 576 B out. | `compute` | `writer` | whole program (every block) | **Refinement 1 — ZERO-COPY when the output shard IS the block** (`plan.output_native`): placed on the output buffer the same way, again costing no additional L1, and the program then carries **no writer kernel at all** — the packer has already written every tile into the output shard. Order matches because compute emits tile-rows left-to-right across the whole shard width, which is the order a TILE shard stores its pages in. | **Cannot share with `cb_input_rows`** — same three reasons, stated from this side. In particular the pipelining is an explicit design decision (the two depth knobs exist to make the stages overlap), which per Rule 3 pattern 3 is exactly the reason that must be *recorded* rather than left as an unexplained non-reuse. |
| `cb_input_rows`, **RETILE path** (Refinement 4, `plan.is_retile`) | **1** page of `tb_in` | none — the CB is never written, waited on, pushed or popped. | `{all four axes: absent}` — it carries no data on this path. | `input_tensor.dtype`, `tile = TileDescriptor(tile_h, 32)` | none | none | whole program | **Deliberately kept at one page rather than deleted.** A re-tile needs no row-major intermediate at all (the reader assembles output tiles directly — see the `cb_output_tiles` retile row), so the honest capacity is zero. It is one page and not zero because `kernel_main` is not a template: the DISCARDED branch of an `if constexpr` is still type-checked, and `read_sticks_for_tilize`'s `constexpr elem_size = get_tile_size(cb)/get_tile_hw(cb)` would be a division by zero against an unconfigured CB, so the ROW_MAJOR branches would not compile. One page (`<= 4 KB`) buys a well-formed JIT descriptor for dead code; the alternative is templating `kernel_main`'s whole body, which is churn for the same bytes. |
| `cb_output_tiles`, **RETILE path** (Refinement 4) | `output_depth_batches * write_rows_per_barrier * block_width_tiles` — unchanged | `write_rows_per_barrier * block_width_tiles` pages — unchanged | unchanged, except that `within_tile: spans` is now satisfied by the READER's face walk rather than by the packer | unchanged | **`reader`** (not `compute`) | `writer` | whole program | **The producer moves, the CB does not.** `retile_block` assembles each output tile IN PLACE in this CB out of `retile_copy_unit`-sized runs of the source tiles' faces, so the program carries no compute kernel and this CB's producer/consumer pair is reader -> writer. Still exactly one producer and one consumer, which is the invariant that matters. Because the input CB collapses to one page, the `W_FIT` solve's denominator drops the `input_depth_rows * tb_in` term on this path and the whole budget goes to the column extent — so the retile footprint is **strictly smaller** than the ROW_MAJOR one at the same `block_width_tiles`. |
| `cb_pad_row` (Refinement 2, `plan.pad_active` only) | **1** page of `block_row_bytes` = `block_width_tiles * 32 * element_size(in_dtype)` bytes — i.e. `tb_in / tile_h`, ONE row of the block, not one tile-row | the same 1 page. Nothing is ever pushed or popped: the reader seeds it once per kernel and thereafter only reads from it, so the live set IS the capacity. | `{leading: absent, tile_row: absent -> the fill is row-invariant, so ONE row serves every padded row of every block, tile_col: spans -> block_width_tiles*32 elements, within_tile: partial -> one ROW of a tile-row, not a tile}` | `input_tensor.dtype` — the fill is written into `cb_input_rows`, so it is encoded in the INPUT's format (`pad_fill_word`); the output cast happens later at pack time, exactly as it does for a real element. No `page_size`/DEST interaction: this CB never reaches the unpacker. | `reader` (seeds it) | `reader` (reads it as a local NoC source) | whole program; **allocated at all only when a pad region exists** | **Deliberately NOT shared, and deliberately not eliminated.** Sharing with `cb_input_rows` is foreclosed by concurrency: the reader reads this page as a NoC *source* while the same barrier writes `cb_input_rows` pages, and `cb_input_rows` is simultaneously being popped by compute — a single-producer/single-consumer CB cannot be both. Sharing with `cb_output_tiles` is worse (compute owns it). Eliminating it means filling each padded row with a RISC store loop over `block_row_bytes` (up to 32 KB, per row, on the critical path of whichever core owns a tail block) instead of one DM-engine transfer; `[1,1,1,50304]` at 31 pad rows per tile-row is the shape that makes that the difference. The page is `1/tile_h` of a `cb_input_rows` page, so at `tile_h = 32` this row adds **1.5%** to the footprint (see the total below). |
| `cb_input_rows_split` (Refinement 6, `plan.split_reader > 0` only) | `split_reader * block_width_tiles`, where `split_reader = min(max_extent - 1, floor(max_extent * SPLIT_READER_WRITER_SHARE_PCT / 100))` and `max_extent = ceil(R / num_row_groups)` — i.e. the WRITER's whole half-block, in tile-rows | `block_width_tiles` pages — one tile-row, exactly as `cb_input_rows`: `compute_kernel_lib::tilize` waits/pops one tile-row per iteration and `read_sticks_for_tilize` reserves/pushes one per tile-row. **The capacity is deliberately the whole half-block rather than a depth-2 window**, and that is a DEADLOCK argument, not a perf one: the writer must be able to finish its read and reach its store without waiting on compute, or it can block in `cb_reserve_back(cb_input_rows_split)` while compute blocks in `cb_reserve_back(cb_output_tiles)` — a cycle. | `{leading: streams, tile_row: spans -> split_reader tile-rows (the writer's share of the block), tile_col: spans -> block_width_tiles pages, within_tile: spans -> one page IS one tile's row-major bytes}` | `input_tensor.dtype`, identical to `cb_input_rows` in format, `page_size` and `TileDescriptor` — which is what makes the compute kernel's `NoReconfigure` correct across the two back-to-back `tilize` calls. On the fp32 -> fp32 relay it carries `UnpackToDestMode::UnpackToDestFp32` too; setting it on `cb_input_rows` alone trips the helper's own `Fp32Mode::Lossless` static_assert on the split call (`tilize_helpers.inl:122`). | `writer` | `compute` | whole program; **allocated at all only where the split is on** | **Not shared, and it is the reason the split needs a second CB at all.** `cb_input_rows` already has the reader as its producer, and a CB has exactly one producer — two pushers is silent UB, not a compile error. Sharing with `cb_output_tiles` is worse (opposite direction, different page format). Aliasing with `cb_pad_row` is foreclosed by the gate: the split is off on every padded plan, so the two are never both allocated — but they are also never both live, so the non-overlap is a consequence of the gate rather than an opportunity. Cost is bounded by the gate: the split only turns on at `block_row_bytes <= SPLIT_READER_MAX_ROW_BYTES` (256 B), so `block_width_tiles <= 8` at bf16 and this CB is at most `split_reader * 8 * tb_in` — 6 KB on the measured target `[1,1,16384,32]` (3 rows x 1 tile x 2048 B). |

Two CBs is the inventory floor for an UNPADDED call: each crosses a thread boundary (`reader`→`compute`,
`compute`→`writer`), and the one candidate for elimination — an in-place transform —
is forbidden by the helper's `static_assert`. There is no intermediate between compute
phases (there is only one compute phase) and no scaler or mask CB. A padded call adds
exactly ONE more — `cb_pad_row`, the constant CB — and it is one *row*, not one tile:
the fill is invariant along `tile_row`, so a single row serves every padded row of
every block, which is what keeps a constant buffer from being priced per tile.
A SPLIT-READER call (Refinement 6) adds one more, `cb_input_rows_split`, and it is
not an inventory failure but a direct consequence of the CB contract: the split's
whole point is that a SECOND kernel produces part of the block, and a CB has exactly
one producer. The two are mutually exclusive by the gate (the split is off on every
padded plan), so the inventory is never four.

## Symbol table

Every non-block parameter appearing in a capacity expression, with its bound and the
predicate establishing it.

| Symbol | Bound | Predicate establishing the bound |
|--------|-------|----------------------------------|
| `tile_h` | `{1, 2, 4, 8, 16, 32}`; Phase 0 `= 32` | `tile=` validation: height must be a power-of-two fraction of 32 and width must be 32 (raises `ValueError` otherwise). Phase 0 additionally pins it via `SUPPORTED["tile_height"] = [32]`. |
| `block_width_tiles` | `1 <= block_width_tiles <= min(255, W_FIT)` | `W_FIT = clamp((budget - WRITE_BATCH_MIN_TILES*tb_out) // (2*tb_in + 2*tb_out), 1, FAST_TILIZE_WIDTH_CAP)` with `budget = ttnn.get_max_worker_l1_unreserved_size()` and `FAST_TILIZE_WIDTH_CAP = 255` from `can_use_fast_tilize`'s `block_width_tiles < 256` (`tilize_helpers.inl:77`). Under `low_l1=True` the additional cap `LOW_L1_WIDTH_CAP` (a host constant, independent of every tensor dimension) applies. **Refinement 6:** `block_width_tiles` is `ceil(C / num_w_chunks_target)` — the design's own extent — clamped into `[1, min(W_CAP, C)]`, so the bound holds by construction. The leftover `C % block_width_tiles` columns become a tail chunk carried by a SECOND core range (see `block_width_tail_tiles`). The old coarsest-**divisor-of-`C`** rule survives on exactly one leg, a sub-row-paged source, where the width faces a constraint of its own — see below. **On the shard-driven plan (Refinement 1) it is not solved at all: `= shard_cols_tiles`, read off the shard spec.** The bound still holds — a shard column is a whole number of tiles by construction, and the non-native side's DEPTH knobs (not the block) are what shrink if the pair overruns the budget. `low_l1` is inert there: the shard, not `W_CAP`, fixes the extent. **On a sub-row-paged source it must additionally divide the page width in tiles**, so a block's row segment sits inside ONE source page; expressed as `_largest_divisor_at_most(gcd(C, page_width_tiles), width_limit)`. That is a constraint on the WIDTH itself, not on the per-core mix, so Refinement 6's tail cannot lift it and this leg keeps the divisor rule (`RAGGED_COLUMN_TAIL and input_pages_per_row == 1` is the gate). **Refinement 3:** `num_w_chunks_target` is no longer just the occupancy cut — it is `max(w_chunks_for_l1, ceil(num_cores * waves / R))` evaluated at the largest `waves <= PIPELINE_WAVES_PER_CORE` whose resulting `block_row_bytes >= MIN_BLOCK_ROW_BYTES`, falling back to `waves = 1`. The bound is unaffected (a larger target can only make the width SMALLER), and so is every capacity expression below — the wave rule spends L1, never asks for more. **Refinement 6** walks that `waves` ladder by HALVING (`cap, cap/2, ..., 2`) rather than decrementing, because `{1,2,4,8}` are the rungs `MIN_BLOCK_ROW_BYTES` was calibrated on and the ragged rule makes the intermediate rungs reachable for the first time. |
| `block_width_tail_tiles` | `0 <= block_width_tail_tiles < block_width_tiles`; `= C - (C // block_width_tiles) * block_width_tiles` | **Refinement 6 gave it an implementation.** 0 on smooth `C` (no tail group, one core range, byte-identical to Refinement 5) and on a sub-row-paged source (the divisor rule still binds there). Non-zero, it names a SECOND core range with its own `block_width_tiles` CT arg, its own `col_tile_offset` and its own CB sizes. The wrap requirement is untouched, because it was only ever a per-CORE requirement: `split_work_to_cores`' contiguous ranges are what could mix two quanta on one core, and the two families are a prefix/suffix split of the row-wise core order, so no core ever sees both. Its capacities are all `<=` the full group's (same depths, narrower block), so it never sets `L1_per_core`. |
| `split_reader` | `0 <= split_reader <= ceil(R / num_row_groups) - 1` | **Refinement 6.** The tile-rows of each block the WRITER kernel reads, and also `cb_input_rows_split`'s depth in tile-rows. `= min(max_extent - 1, floor(max_extent * SPLIT_READER_WRITER_SHARE_PCT / 100))` when every gate leg passes, else 0. The gate legs: the plain stick path only (not padded / retile / native input / sub-row-paged), a NON-native OUTPUT (a natively sharded output emits no writer kernel, so the split CB would have no producer — a hang, and the leg the static analyzer caught), `block_row_bytes <= SPLIT_READER_MAX_ROW_BYTES`, `max_extent >= 2`, and the extra CB fitting the same budget the column extent was solved against. That last check is what keeps this symbol out of the `W_FIT` inversion: the split is decided AFTER the width, and declines itself rather than narrowing the block. |
| `write_rows_per_barrier` | `1 <= write_rows_per_barrier <= WRITE_BATCH_MIN_TILES = 4` | `= max(1, ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))`; since `block_width_tiles >= 1`, the ceiling is at most `WRITE_BATCH_MIN_TILES`. |
| `input_depth_rows` | `= 2` (Phase 0); `>= 2` in general. **Measured across {1,2,3,4} and flat at every value** on both shapes the design's overlap lamp names — kept at 2 as the smallest value that overlaps at all and the cheapest in L1 of those; still a live knob. Evidence and the bottleneck it implies (DRAM-bandwidth-bound, ~183 GB/s on `[1,1,2048,2048]`) are in `tilize_program_descriptor.INPUT_DEPTH_ROWS`. | Fixed host constant. Lower bound 2 is required by the overlap it buys and by `read_sticks_for_tilize`'s capacity assert `width_in_tiles <= cb_capacity` (`tilize_helpers_dataflow.inl:105-107`) plus `compute_kernel_lib::tilize`'s `get_dfb_num_pages(input_dfb) >= block_width_tiles` (`tilize_helpers.inl:220-222`). |
| `output_depth_batches` | `= 2` (a host constant) | Depth measured in WRITE BATCHES rather than tile-rows. Lower bound 2 is what buys the compute-side overlap window; being an integer count of batches is what keeps the capacity an exact multiple of the batch quantum (the wrap requirement below). Capacity in tile-rows is therefore `2 * write_rows_per_barrier`, bounded by `2 * WRITE_BATCH_MIN_TILES = 8`. |
| `tb_in` | `= tile_h * 32 * element_size(in_dtype)`; `<= 32*32*4 = 4096` B | `tile_h <= 32` (above) and `element_size <= 4` over `TARGET["dtype"]` (widest is `uint32`/`int32`/`float32`). |
| `tb_out` | `= output_tensor.buffer_page_size()`; `<= 32*32*4 = 4096` B | Same: `tile_h <= 32`, and the widest format in `TARGET["output_dtype"]` is 4 B/element. Block-float outputs are *smaller* (`Bfp8_b` = 1088 B, `Bfp4_b` = 576 B for a 32x32 tile). **Refinement 5: `tb_out` and `tb_in` are independent** — the cast diagonal makes them differ by up to 4x in either direction (`bf16 -> fp32` widens 2048 -> 4096; `fp32 -> bfp4` narrows 4096 -> 576). Both appear separately in the `W_FIT` denominator, so the pair is solved rather than assumed equal, and the bound above still holds because each is independently `<= 4096`. |
| `WRITE_BATCH_MIN_TILES` | `= 4`, a host constant | Named constant, single source. **MEASURED on device, not taken from the catalog:** the design's lamp asked whether 8 sits past the knee and it does. `[1,1,16384,32]` (C == 1, so this constant *is* `write_rows_per_barrier`), median device kernel ns over 3 fresh-cache runs at 64/64 cores — wb=1 **24430** (the one-write-per-barrier trap), wb=2 22519, wb=4 **20723**, wb=8 22568, wb=16 22403. The plateau is at 4, which is exactly where `double_buffer/report.md` put it; 4 is 1.18x over the trap and 1.09x over the design's 8, and costs *less* L1. Harness: `tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_write_batch.py`. Not a tensor dimension. |
| `PIPELINE_WAVES_PER_CORE` | `= 4`, a host constant | The CAP on how many pipeline WAVES (one wave = one tile-row of the block, i.e. one reader push / one compute call / one writer wait) the column cut is allowed to buy. A core overlaps its DRAM reads against its DRAM writes only above one wave, and the tensor holds `R * num_w_chunks` tile-rows, so `waves_per_core = R * num_w_chunks / num_cores` — the wave demand and the occupancy demand are ONE expression on the column axis, this constant being the factor between them. Past 4 the pipe is full. |
| `SPLIT_READER_MAX_ROW_BYTES`, `SPLIT_READER_WRITER_SHARE_PCT` | `= 256` B and `= 38` %, host constants | Refinement 6's two split knobs. The first is the read-transaction size at or below which ONE data-movement RISC-V is issue-bound (established by a whole-op ablation, not assumed — see the constant's comment and `changelog.md`); at 0 the split is off everywhere and the plan is byte-identical to Refinement 5. The second is the writer's share of each block's tile-rows, swept on device (`tests/.../test_tilize_lever_split_reader.py`). Neither is a tensor dimension; the first bounds `block_width_tiles` on the split path, which is what bounds `cb_input_rows_split`. |
| `MIN_BLOCK_ROW_BYTES` | `= 512`, a host constant, in BYTES | The read-transaction floor that stops the wave trade: a wave is bought by HALVING the block width, so it only pays while the read stays large enough to amortize the NoC/DRAM per-transaction cost. Measured across five geometries — see the constant's own comment in `tilize_program_descriptor.py` and `tests/.../tilize/test_tilize_lever_pipeline_waves.py`. In bytes rather than tiles so it means the same thing at every element size. |
| `COMPUTE_SKIP_FORMAT_RECONFIG`, `COMPUTE_AMORTIZE_INIT` | `= True`, host constants | Compute-side per-call overhead knobs, both **capacity-neutral**: they change which `compute_kernel_lib::tilize` template arguments the kernel instantiates, not any CB size. `COMPUTE_AMORTIZE_INIT` is additionally gated at emission time on `max blocks per core > 1`, because the three extra instantiations grow the TRISC binary. |
| `FAST_TILIZE_WIDTH_CAP` | `= 255`, a host constant | `can_use_fast_tilize` requires `block_width_tiles < 256` (`tilize_helpers.inl:77`). |
| `LOW_L1_WIDTH_CAP` | `= 4`, a host constant | Named constant, single source, **independent of every tensor dimension** — that independence is what makes the `low_l1=True` contract structural rather than a percentage. Yields 40 KB at bf16 / 80 KB at fp32 regardless of `W`. |
| `budget` | device-reported, `> 0` | `ttnn.get_max_worker_l1_unreserved_size()` (`ttnn/ttnn/device.py:20`) — read at runtime, never a literal. **Refinement 1 correction:** that call reports the worker L1 *arena*, not what is free in it, and an L1-resident operand sits in the same arena the CBs are cut from. `derive_plan` now subtracts the per-core resident bytes of both operands (`_resident_l1_bytes`: the shard's bank size for a sharded tensor, `total / num_cores` for an L1-interleaved one, 0 for DRAM), floored at one page pair. Inert on `dram_to_dram`; it binds on `l1_to_l1` / `dram_to_l1`, and every sharded operand is L1-resident by definition. |
| `block_row_extent` | `1 <= block_row_extent <= R` | **Appears in no capacity expression.** It is the runtime `num_blocks` handed to `compute_kernel_lib::tilize`, which processes one tile-row per iteration with its own wait/push per iteration (`tilize_helpers.inl:233-259`), so it is an axis both CBs *stream over* — never one either spans. This is the reason the coarsest row extent (the core's whole row-group) is taken unconditionally: it costs zero L1. |
| `shard_rows_tiles`, `shard_cols_tiles`, `num_shard_rows`, `num_shard_cols` | shard-derived; `num_shard_rows * shard_rows_tiles == R` and `num_shard_cols * shard_cols_tiles == C` are CHECKED, and a partition failing either is refused as a plan driver | `shard_partition()` derives them from the shard spec against the tensor's padded 2-D view. Only `shard_cols_tiles` enters a capacity expression, and only as `block_width_tiles`; the rest size the work split. |
| `input_pages_per_row` | `>= 1`; `1` whenever a ROW_MAJOR page IS a whole row | `ceil(input_padded_W / page_width)`. `> 1` only for a width/block-cutting shard read through the accessor, which puts the reader on its strided branch. Appears in no capacity expression. |
| `pad_active` | `{0, 1}` | Derived from the two shapes, not from the request: 1 iff the output's padded grid reaches past the input's LOGICAL extent on any of the three axes (`in_num_images < num_images`, `in_rows_per_image < rows_per_image*tile_h`, `in_row_bytes < C*32*elem`). 0 on every tile-aligned call whether or not a padding argument was passed, which is what keeps the Phase 0 reader branch byte-identical. |
| `in_num_images`, `in_rows_per_image`, `in_row_bytes`, `rows_per_image_out`, `pad_word` | tensor-derived (the first three from the input's LOGICAL shape, left-padded to rank 2) | **None appears in any capacity expression.** They are the pad *boundary*, consumed as reader CT args; the only capacity the padding adds is the single `cb_pad_row` page, whose size is `block_row_bytes` and therefore already bounded by `block_width_tiles`. |
| `R`, `C`, `num_row_groups`, `num_w_chunks`, `num_images`, `rows_per_image` | tensor-derived, unbounded in principle | **None of these appears in any capacity expression.** They size the *work split*, not the footprint. Their absence from the table above is the whole `low_l1` proof. |

## Total per-core footprint

```
L1_per_core = input_depth_rows     * block_width_tiles * tb_in
            + split_reader         * block_width_tiles * tb_in         # cb_input_rows_split
            + output_depth_batches * write_rows_per_barrier * block_width_tiles * tb_out
            + [pad_active] * block_width_tiles * (tb_in / tile_h)      # cb_pad_row

            = (2 + split_reader) * block_width_tiles * tb_in
            + 2 * write_rows_per_barrier * block_width_tiles * tb_out
            + [pad_active] * block_width_tiles * (tb_in / tile_h)
```

`split_reader` is 0 on every plan whose read transaction is large enough not to be
RISC-V-issue-bound, and `[pad_active]` and `split_reader` are mutually exclusive by
the gate — so no call pays both. Where the split IS on, it is bounded by its own
gate: `block_row_bytes <= SPLIT_READER_MAX_ROW_BYTES` caps `block_width_tiles * tb_in
/ tile_h` at 256 B, so this term is at most `split_reader * 256 * tile_h` bytes
(6 KB on the measured target). The two-core-range plan does not appear here at all:
the tail range's block is NARROWER at the same depths, so `L1_per_core` is the full
range's, unchanged.

On the **RETILE path** (`plan.is_retile`, Refinement 4) the first term collapses to
one page, because the reader assembles output tiles directly into
`cb_output_tiles` and `cb_input_rows` carries no data:

```
L1_per_core (retile) = 1 * tb_in                                     # the dead-code stub
                     + 2 * write_rows_per_barrier * block_width_tiles * tb_out
```

and `derive_plan` drops `INPUT_DEPTH_ROWS * tb_in` from the `W_FIT` denominator to
match, so `W_FIT` roughly DOUBLES on the retile path. Tiny tiles cut both `tb_in`
and `tb_out` in proportion to `tile_h`, so the whole table below shrinks by
`32/tile_h` at a sub-32 output tile — **Refinement 4 only ever reduces this
footprint, on both of its halves.** Measured (`probes/probe_022.py`,
`probe_025.py`): `[1,1,32,2048]` is 20 KB at `tile_h=32`, 12 KB at 16, 2 KB at 1;
`[1,1,2048,64]` is 24 KB at 32 and 768 B at 1.

| Shape | path | `bw` | in pages | out pages | `L1_per_core` |
|-------|------|------|----------|-----------|---------------|
| `[1,1,2048,2048]` | ROW_MAJOR -> 32 | 16 | 32 | 32 | 131072 B |
| `[1,1,2048,2048]` | retile 32 -> 16 | **32** | **1** | 64 | **66560 B** |
| `[1,1,2048,2048]` | retile 16 -> 32 | 16 | **1** | 32 | **67584 B** |
| `[1,1,32,2048]` | ROW_MAJOR -> 16 | 2 | 4 | 8 | 12288 B |
| `[1,1,32,2048]` | ROW_MAJOR -> 1 | 8 | 16 | 16 | 2048 B |
| `[1,1,2048,64]` | ROW_MAJOR -> 1 | 2 | 4 | 8 | 768 B |

**Refinement 5 — what the dtype pair does to the footprint.** MEASURED on
`[1,1,2048,2048]`, ROW_MAJOR -> 32, `low_l1=False`, budget read from the device
(`probes/probe_041.py`):

| `dtype -> output_dtype` | `tb_in` | `tb_out` | `bw` | `L1_per_core` |
|-------------------------|---------|----------|------|---------------|
| `bf16 -> bf16` (Phase 0) | 2048 | 2048 | 16 | 131072 B |
| `bf16 -> fp32` (widening) | 2048 | 4096 | 16 | **196608 B** (1.5x) |
| `fp32 -> fp32` | 4096 | 4096 | 16 | **262144 B** (2x) |
| `fp32 -> bfp4` (narrowing) | 4096 | 576 | 16 | 149504 B |
| `uint8 -> uint8` | 1024 | 1024 | 16 | **65536 B** (0.5x) |

The informative column is `bw`, and it is CONSTANT at 16. On this shape the column
extent is fixed by the pipeline-wave rule, not by `W_FIT`: `block_width_tiles = 16`
keeps `block_row_bytes = 16*32*element_size >= MIN_BLOCK_ROW_BYTES` at every element
width, and `W_FIT` is nowhere near binding. So the widened page is NOT absorbed by a
narrower block here — it is paid, linearly, and `fp32 -> fp32` costs exactly 2x the
bfloat16 diagonal while `uint8` costs half. That is the trade taken deliberately
(see `cb_output_tiles`' page-format cell): the output page size IS the output
tensor's, and `dtype=float32` is the caller asking for those bytes. The solve still
protects the budget on shapes where `W_FIT` DOES bind — `tb_in` and `tb_out` enter its
denominator separately, so there a widened output narrows `bw` instead of overrunning.

No row scales with a tensor dimension, so the `low_l1` / `bounded_cb` claims are
unchanged: the pair moves `tb_in` / `tb_out`, which are functions of
`(tile_h, dtype)` only. 262144 B is the worst case over the whole cartesian at this
`bw`, well inside the arena.

Read out of the built plan (`probes/probe_026.py`), same as every other table here.
The retile rows show both effects at once: `in pages = 1`, and `bw` doubling from
16 to 32 on `32 -> 16` because the freed denominator lets the column extent grow.

`[pad_active]` is 0 or 1 — the pad row is allocated only where a pad region exists.
It scales with `block_width_tiles` like everything else and with **no tensor
dimension**, so the `low_l1` claim below is unchanged; at `tile_h = 32` it is
`1/64` of the two streaming terms' `2*tb_in + 2*tb_out` (1.5%), and it is folded
into the `W_FIT` denominator (`derive_plan` adds `in_page_bytes // tile_h` to
`denom` when `pad_active`) rather than spent behind the budget's back.

Closed-form upper bound, used as the `W_FIT` solve. Since
`block_width_tiles * write_rows_per_barrier <= block_width_tiles * (WRITE_BATCH_MIN_TILES/block_width_tiles + 1)
= WRITE_BATCH_MIN_TILES + block_width_tiles`:

```
L1_per_core <= 2*block_width_tiles*tb_in + 2*(block_width_tiles + WRITE_BATCH_MIN_TILES)*tb_out
```

which inverts to
`W_FIT = (budget - 2*WRITE_BATCH_MIN_TILES*tb_out) // (2*tb_in + 2*tb_out + [pad_active]*tb_in/tile_h)`,
clamped into `[1, FAST_TILIZE_WIDTH_CAP]`. This is the expression
`tilize_program_descriptor.derive_plan` evaluates verbatim; the only change from
the design's form is the `2*` on the output depth (batches, not `wrpb + 1`) —
see "Deviations".

**Which terms scale with which knob.**

| Term | Scales with | Does **not** scale with |
|------|-------------|-------------------------|
| `(2 + split_reader) * block_width_tiles * tb_in` | `block_width_tiles` (the `tile_col` extent), `input_depth_rows`, `split_reader`, `tile_h`, `element_size(in_dtype)` | `block_row_extent`, `R`, `C`, `H`, `W`, `rank`, `num_images`, core count. **`split_reader` is bounded by `ceil(R / num_row_groups) - 1`, which IS an `R`-dependent expression** — but `num_row_groups = min(R, ceil(num_cores / total_w_chunks))`, so `ceil(R / num_row_groups) = ceil(R * total_w_chunks / num_cores)` grows with `R` on a shape whose column axis cannot fill the grid (`C == 1`). The gate closes that: it also requires `block_row_bytes <= SPLIT_READER_MAX_ROW_BYTES`, AND it re-checks the resulting footprint against the same `budget` the width was solved against and returns 0 if it does not fit. So this term is dimension-bounded by an explicit budget test rather than by a structural argument — the one term in this table that is, and the reason the check exists. |
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

**Refinement 3 rebuilt this table**: `block_width_tiles` is now the coarsest
divisor that both fills the grid AND leaves the read at or above
`MIN_BLOCK_ROW_BYTES` while buying up to `PIPELINE_WAVES_PER_CORE` pipeline waves
(see the symbol table). Where the wave rule fires it makes the block NARROWER,
so every changed row is strictly SMALLER than it was; nothing grew.

| Shape | `bw` | `wrpb` | waves/core | read | cores reached | `L1_per_core` |
|-------|------|--------|-----------|------|---------------|---------------|
| `[1,1,32,32]` single tile | 1 | 4 | 1 | 64 B | 1 (only 1 tile exists) | **20 KB** |
| `[1,1,2048,2048]` square_large | 16 | 1 | 4 | 1024 B | 64 / 64 | **128 KB** (was 512 KB at `bw = 64`) |
| `[1,1,1024,1024]` square_mid | 8 | 1 | 2 | 512 B | 64 / 64 | **64 KB** (was 128 KB at `bw = 16`) |
| `[1,1,32,16384]` **perf focus** | 8 | 1 | 1 | 512 B | 64 / 64 | **64 KB** (unchanged — a 2nd wave would cost a 256 B read) |
| `[1,1,32,32768]` short_wide_wide | 8 | 1 | 2 | 512 B | 64 / 64 | **64 KB** (was 128 KB at `bw = 16`) |
| `[1,1,32,2048]` short_wide | 1 | 4 | 1 | 64 B | 64 / 64 | **20 KB** |
| `[1,1,2048,64]` full_width | 2 | 2 | 1 | 128 B | 64 / 64 | **24 KB** |
| `[1,1,16384,32]` tall_narrow | 1 | 4 | 8 | 64 B | 64 / 64 | **26 KB** (20 KB + 3 split pages; Refinement 6 turns the split reader on here) |
| `[1,1,32,8192]` fp32, `low_l1=False` | 4 | 1 | 1 | 512 B | 64 / 64 | **64 KB** |
| `[1,1,32,8192]` fp32, `low_l1=True` | `min(4, LOW_L1_WIDTH_CAP)` = 4 | 1 | 1 | 512 B | 64 / 64 | **64 KB**, and independent of `W` |

The two fp32 rows are derived rather than read (fp32 is Refinement 5): at
`elem = 4` the read is `128 * bw` bytes, so the 512 B floor needs `bw >= 4`, the
occupancy cut already gives exactly 4 at `C = 256`, and no wave factor clears
the floor — the row is unchanged from Phase 0 by construction.

The `low_l1` pair is the load-bearing row: at `C = 256` the two settings agree on
every single number, because `low_l1=False` was *already* dimension-independent —
which is also why `helpers.run_tilize`'s bit-identity A/B is satisfied by
construction rather than by care. The harness that produced this table is
`tests/ttnn/unit_tests/operations/tilize/probes/probe_002.py`.
| `[1,1,1,50304]` bf16, `low_l1=True` | `LOW_L1_WIDTH_CAP` = 4 (393 w-chunks, ~7 blocks/core) | 2 | **40 KB** — the widest grid in the suite, at the same footprint as the narrowest |

**Padded footprints (Refinement 2)**, read out of the built plan the same way
(`probes/probe_018.py`). The `+pad` column is `cb_pad_row` — the whole cost of
the padding path:

| Shape (`pad_mode="auto"`) | `bw` | `wrpb` | cores reached | streaming CBs | `+pad` | `L1_per_core` |
|-------|------|--------|---------------|---------------|--------|---------------|
| `[1,1,50,50]` hw tails | 1 | 4 | 4 (only 4 tiles exist) | 20 KB | 64 B | **20 KB** |
| `[1,1,1,2048]` single stick, 31 pad rows / tile-row | 1 | 4 | 64 / 64 | 20 KB | 64 B | **20 KB** |
| `[1,1,32,4090]` short_wide W tail | 2 | 2 | 64 / 64 | 24 KB | 128 B | **24 KB** |
| `[1,1,1,50304]` logits row, C=1572 | **13** + a 12-wide tail range | 1 | 64 / 64 | 104 KB | **832 B** | **105 KB** (was `bw = 12`, 768 B, 97 KB under the divisor rule; Refinement 6's tail) |
| `[8,1,249,2048]` H tail through the fold | 16 | 1 | 64 / 64 | 128 KB | 1024 B | **129 KB** (was 516 KB at `bw = 64`) |
| `[1,1,50,50]`, `low_l1=True` | 1 | 4 | 4 | 20 KB | 64 B | **20 KB** (identical to `False`) |
| `[1,1,1,50304]`, `low_l1=True` | 4 | 1 | 64 / 64 | 32 KB | 256 B | **32 KB** |

Refinement 3's wave rule reaches the padded path through the same solved column
cut, which is why `[8,1,249,2048]` (an `R = 64 x C = 64` grid, geometrically the
same as `square_large`) shrank by the same 4x. `cb_pad_row` shrinks with it — it
is one block ROW — so the padding overhead stays under 1% of the total.

The pad row is `<= 0.8%` of the total on every one of them, it holds no tensor
dimension, and the `low_l1` pair is again bit-for-bit the same footprint — the
padded path inherits the dimension-independence rather than re-arguing it.

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
   *`cb_pad_row`:* capacity == live set == 1 page, so there is nothing to over- or
   under-account. It is the one row whose live set does **not** scale with
   `tile_row`, and that is the load-bearing claim: the fill is invariant along
   that axis, so a per-tile-row (or per-tile) constant buffer would be `tile_h`x
   (or `tile_h * block_width_tiles`x) larger for no benefit.
2. **Page format vs DEST width, both directions.** `fp32_dest_acc_en` is set **iff**
   the input dtype is `float32`, and both CBs carry the tensor's own format. No
   `Float32` page under 16-bit DEST; no 16-bit page under fp32 DEST. `tilize` performs
   no arithmetic, so there is no accumulation to truncate at a phase boundary.
   *(The fp32-output refinement additionally requires `Fp32Mode::Lossless` +
   `UnpackToDestFp32` — a correctness requirement of the LLK path, recorded in
   `op_design.md` → Key Risks, not a page-format finding.)*
3. **Disjoint lifetime, no justification.** `cb_input_rows_split` and `cb_pad_row`
   are the one pair that is never co-allocated (the split gate excludes the padded
   path), but that is mutual exclusion by construction rather than a disjoint
   lifetime to exploit — aliasing them would save nothing, because no plan
   allocates both. Otherwise no CB pair has disjoint lifetimes — all
   are live for the whole program, and the two streaming CBs are deliberately
   pipelined against each other. Both `Shares with / why not` cells are filled with
   three concrete reasons each, one of them a `static_assert` citation.
   `cb_pad_row`'s cell is filled too: it cannot alias either streaming CB because
   the reader reads it as a NoC **source** in the same barrier that writes
   `cb_input_rows`, so their live ranges overlap exactly; and eliminating it in
   favour of a RISC store loop is priced there (up to 32 KB of per-word stores per
   padded row, on the critical path of whichever core owns a tail block).
4. **Bounds and closed form.** Every symbol in the total appears in the symbol table
   with a bound and its predicate. The total is closed-form in
   `{input_depth_rows, output_depth_batches, write_rows_per_barrier,
   block_width_tiles, tb_in, tb_out}`, and the
   `W_FIT` solve is the *inversion* of that closed form — a bounded extent selection
   over a stated expression, not a search that settled for a finer block. The
   inventory was minimized first (two CBs, the thread-boundary floor, with the
   in-place candidate ruled out by citation) before any budget expression was written.
   *`cb_pad_row`:* `block_row_bytes` = `block_width_tiles * 32 * elem`, bounded by
   `block_width_tiles <= FAST_TILIZE_WIDTH_CAP` and `elem <= 4`, so `<= 32 KB`; and
   it is inside the `W_FIT` inversion (`denom += tb_in / tile_h` when
   `pad_active`), not spent outside the budget.
5. **Every capacity is a multiple of its transfer quantum.** Added by the
   implementation, because it is the audit the two CB endpoints enforce at
   runtime. `cb_input_rows` = `2 * W` pages against a `W`-page push/pop quantum;
   `cb_output_tiles` = `2 * wrpb * W` pages against a `wrpb * W`-page pop quantum
   and a `W`-page push quantum. **Refinement 6 changed the reason, not the fact:**
   `W` no longer has to divide `C` — the leftover columns are a tail chunk — but a
   core never mixes two quanta because the two widths live on DISJOINT CORE
   RANGES, each with its own CB descriptors sized from its own `W`. The invariant
   is per-core and it is satisfied per-core. `cb_input_rows_split` is
   `split_reader * W` pages against the same `W`-page quantum, so it holds too.
   `cb_pad_row` has no quantum at all — nothing is pushed or popped on it, so the
   wrap invariant does not apply; the reader's local reads out of it are plain
   L1 addresses, never CB-relative ones. The padded reader branch pushes and waits
   `block_width_tiles` per tile-row exactly like the helper it replaces, including
   on a short last tile-row: the ACTUAL page count is always the full
   `block_width_tiles` because a tile-row's pages are whole tiles whether their
   rows came from the tensor or from the fill.

## Deviations from `op_design.md`

One remains; the first was RESOLVED by Refinement 6 and is kept here with its
resolution because the mechanism it describes is still live.

1. **RESOLVED (Refinement 6).** `block_width_tiles` is now
   `ceil(C / num_w_chunks_target)` — the design's own extent — and
   `block_width_tail_tiles` exists, carried by a second core range. The mechanism
   below is unchanged and is what the two-core-range shape respects: the wrap
   requirement is a **per-core** requirement, and the mix it forbids is a mix
   *within one core*. Splitting the grid's row-wise core order into a full-width
   prefix and a tail suffix means no core ever sees two quanta, so the ragged tail
   is legal without weakening anything. `RAGGED_COLUMN_TAIL = False` restores the
   divisor rule verbatim, and a sub-row-paged source still takes it unconditionally
   (there the divisor is a constraint on the width itself, not on the per-core mix).
   The original entry, for the record:

   **`block_width_tiles` is a DIVISOR of `C`, not `ceil(C / num_w_chunks_target)`;
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
   *(Refinement 6 measured exactly that exception: `[1,1,1,50304]` at 12 x 131
   took 31179 ns and the ragged rule's 13 + a 12-wide tail takes 28474 ns, at the
   same 64/64 cores. The `lcm(bw, tail)` capacity the entry priced is never needed,
   because the two widths never meet on one core.)*

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
| input, **L1 shard consumed natively** (Refinement 1) | **0** | The shard is already in this core's L1 and the CB is placed on it, so the reader issues no transfer of any kind. Not "one crossing made cheap" — no crossing. | **0** bytes. An accessor read of a core's own shard would have added `in_bytes` of cross-core NoC for no reason; that is exactly the non-implementation this refinement avoids. |
| output, **L1 shard produced natively** (Refinement 1) | **0** | The packer writes the tiles into the output shard's own L1; the program carries no writer kernel. | **0** bytes. |
| input, **padded** (Refinement 2) | **1**, and *less* than 1 in bytes | Each stick's column slice is still read exactly once — the pad does not add a re-fetch, it *removes* reads. A fully padded row (an H tail row, an all-pad tile column, an all-pad leading slice) issues **no DRAM transfer at all**: it is one local L1->L1 read out of `cb_pad_row`. So a padded call crosses DRAM with exactly the input's LOGICAL bytes, never its padded bytes. | **0** bytes of cross-core. The pad row's transfers are core-local (source and destination are both this core's L1), so they are L1 bandwidth, not NoC-between-cores. |
| output, **padded** | **1** | Unchanged: every output tile page, pad positions included, is written once by its owning core. The fill costs output bytes because the output genuinely IS larger — that is the caller's request, not overhead. | **0** bytes. |
| input (TILE), **retile** (Refinement 4) | **1** | Each `retile_copy_unit` run of the source is read exactly once, by the one core owning the output tile it lands in. The runs partition the input exactly (verified exhaustively for all 36 height pairs), so there is no re-fetch. **This is the row that matters**: the untilize-and-retilize round trip `op_design.md` ranks `rejected` would make this **2**, and the face walk makes it 1 — the named-boundary minimum, reached and not approached. | **0** bytes. Zero semaphores, zero multicast, and no compute kernel in the program at all. |
| output (TILE), **retile** | **1** | Unchanged: each output tile page is written once by its owning core, by the same writer kernel the ROW_MAJOR path uses. | **0** bytes. |
| input or output, **dtype cast** (Refinement 5) | **1 in / 1 out**, and the OUT side's bytes change | The cast is a PACK-TIME conversion inside the one compute call — it adds no pass, no intermediate buffer and no second crossing. What it does change is the size of the one output crossing: `bf16 -> fp32` writes 2x the bytes, `bf16 -> bfp4` writes 1/4. That is the caller's request (the output tensor genuinely is that size), not overhead, and the INPUT crossing is untouched at the input's own bytes. The `fp32 -> fp32` relay adds nothing either: `UnpackToDestFp32` and `Fp32Mode::Lossless` change the DEST datapath, which is on-core and crosses no boundary. | **0** bytes. |
| input or output, **empty tensor** (Refinement 5) | **0** | A zero-tile grid is handed to one core as zero blocks; every kernel's block loop runs no iterations, so no NoC transaction of any kind is issued against the zero-page buffers. The program still dispatches — one dispatch, no traffic. | **0** bytes. |
| input or output, **sharded but on the OTHER side's cut** (cross-spec, DRAM-sharded) | **1** | A genuinely non-local operand: this core's block needs bytes that live in another core's L1 (or in a DRAM bank), so the `TensorAccessor` leg is a real remote transfer and is the correct mechanism. | `<= ` the operand's bytes, once. The only cross-core traffic this op ever generates, and only where the two placements genuinely disagree. |

**Totals per tier.**

| Tier | Bytes |
|------|-------|
| DRAM | `in_bytes + out_bytes` on the interleaved plan — the named-boundary minimum ("each input crosses once, each output crosses once"). The minimum is *reached*, not merely approached. **On the shard-driven plan each L1-resident, natively-consumed side contributes 0**, so a same-spec L1 shard on both sides crosses DRAM zero times and the NoC zero times: the op becomes purely on-core. |
| cross-core NoC | **0** on every plan except the cross-spec / DRAM-sharded leg, where it is the non-local operand's bytes, once. |
| core-local L1 | `2*(in_bytes + out_bytes)` — each CB is written once and read once by its two owners. Not free (L1 bandwidth), but irreducible: the row-major bytes must land in L1 for the unpacker to permute them, and the tiled bytes must land in L1 for the packer to emit them. |

**Transaction shape**, which is what actually discriminates the splits here since the
byte counts are split-invariant:

| Direction | Transactions | Bytes each |
|-----------|--------------|------------|
| read | `R * tile_h * num_w_chunks`, or **0** when the input is consumed natively | `block_width_tiles * 32 * element_size(in_dtype)` (`= C*32*elem` when `num_w_chunks == 1`, i.e. a whole contiguous DRAM page) |
| write | `R * C` (one per output tile, independent of the split), or **0** when the output is produced natively | `tb_out` (one whole tile page) |
| read, **padded** (Refinement 2) | the same `R * tile_h * num_w_chunks` **total transactions**, re-partitioned: the ones whose row carries data stay DRAM reads (of `valid_bytes <= block_row_bytes`), and the fully padded ones become core-local L1->L1 reads of `block_row_bytes`. Plus, once per kernel, `log2(block_width_tiles)` seed transfers (`<= 8`) and one `TILE_WIDTH * elem` RISC store loop (`<= 128` B). | unchanged per transaction; the W tail adds no transaction at all (it is `< TILE_WIDTH*elem` bytes of in-place `fill_l1_range`, after the barrier) |

The padded transaction count is therefore **identical** to the unpadded one for the
same `(R, C, num_w_chunks)`, with a fraction of the reads redirected from DRAM to
local L1. `[1,1,1,2048]` is the extreme: `H = 1`, so 31 of every 32 reads never
touch DRAM. The one thing padding adds per kernel is the seed, and it is
deliberately `log2`-many DM transfers rather than a `block_row_bytes` store loop —
which is the whole reason `cb_pad_row` exists as a buffer rather than as a loop.

Read transactions are the only term the split moves, which is why `num_w_chunks` was
**minimized** rather than maximized at Phase 0: `max(w_chunks_for_l1,
w_chunks_for_occupancy)` takes the smallest value that both fits L1 and fills the
grid, and never more.

**Refinement 3 qualified that (measured).** Transaction COUNT is not the only thing
the split moves — it also moves the number of pipeline WAVES a core owns, and a core
with one wave reads, computes and writes strictly in series, so its DRAM read stream
and its DRAM write stream never overlap. Ablating `[1,1,32,16384]` (all payloads
stubbed 660 ns; + compute 1515; + reads 8241; + writes 9459; full op 13247) shows the
two halves are BALANCED (6726 ns of reads against 7944 ns of writes) and overlap by
only 2938 ns of the 14670 they would take in series. So the split now takes the
smallest `num_w_chunks` that fits L1, fills the grid AND gives each core up to
`PIPELINE_WAVES_PER_CORE` waves — but only while the resulting read stays at or above
`MIN_BLOCK_ROW_BYTES`, because each extra wave is bought by halving the read. Both
terms of that trade are measured, and the floor is set at the largest value that is
never the wrong call. DRAM crossings are untouched (still 1 in / 1 out): the wave
rule re-partitions the same bytes, it does not re-fetch any.

**RESOLVED by Refinement 6 — the ragged tail now lands, and the escape below was
taken.** `block_width_tiles` is `ceil(C / target)` and the leftover columns are one
tail chunk on a second core range, so the realized `num_w_chunks` no longer
overshoots: `[1,1,1,50304]` went from 131 chunks of 12 (768 B reads, and a busiest
core carrying 3 blocks against a 2.05 average) to 120 of 13 plus one 12-wide tail
(832 B reads, 2 blocks on the busiest core). Measured on device at 64/64 cores:
**31179 -> 28474 ns** on the padded witness and **37356 -> 35338** on the
tile-aligned one. DRAM crossings are untouched at 1 in / 1 out; only the
transaction SHAPE moved. The paragraphs immediately below are the original
statement of the problem and the escape, kept because the mechanism they describe
(the per-core one-quantum invariant) is what the two-range plan respects rather
than removes — and because a sub-row-paged source still takes the divisor rule,
where the overshoot they describe is still real.

**Refinement 6 also split the READ TRANSACTIONS across both data-movement
RISC-Vs on the issue-bound geometries.** Neither the crossing count nor the byte
count moves — the same sticks are read once each — but on a small read the cost is
the RISC-V's *issue* work, not the bytes, and a whole-op ablation of
`[1,1,16384,32]` (64 B sticks) put NCRISC on the critical path for the entire
kernel while BRISC's own payload was 14% of the wall. The writer kernel now reads
the trailing `split_reader` tile-rows of each block into `cb_input_rows_split`.
Per-tensor crossings: **still 1 in / 1 out** (each stick is read exactly once, by
exactly one of the two RISC-Vs); cross-core traffic: **still 0**. What changes is
which NoC carries a read — the writer's split reads go out on NoC1 alongside its
stores, which is why the writer's share is 38% and not 50%.

**Where the divisor constraint costs read transactions (verifier-added, now
historical for the interleaved path).** The
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
lever rather than fixed here. **Refinement 6 built exactly this** (`plan.groups`,
`ColumnGroup`, `col_tile_offset`); the paragraph stands as the design of what
shipped.

> Cheapest-traffic split considered: **`tile_row` x `tile_col` with `num_w_chunks`
> minimized** — `in_bytes + out_bytes` across DRAM (the minimum), `0` cross-core, and
> the fewest read transactions of any split that fills the grid subject to the
> one-quantum-per-core wrap invariant. Implemented: **that split**. Nothing deferred
> on traffic grounds; the single rough-`C` overshoot is quantified above and was
> closed by Refinement 6's two-core-range plan, so on the interleaved path the
> qualifier is now vacuous and the claim is simply "the fewest read transactions of
> any split that fills the grid".

**Reconciled against the built code.** `num_w_chunks = C // block_width_tiles`
plus a tail chunk of `C % block_width_tiles` columns when that remainder is
non-zero (Refinement 6); the two families' `(row_group, w_chunk)` ranges cover the
`R x C` tile grid exactly once, with the tail's `col_tile_offset` picking up
precisely where the full family's last chunk ends. Each stick's column slice
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
| `[1,1,32,16384]` **perf focus** | 512 | 1 x 8 | **64 / 64** | 512 B/stick | ~13500 | ~155 |
| `[1,1,16384,32]` transposed pair | 512 | 8 x 1 | **64 / 64** | 64 B/stick | ~20900 | ~100 |
| `[1,1,2048,2048]` square_large | 4096 | 4 x 16 | **64 / 64** | 1024 B/stick | ~87400 | ~192 |
| `[1,1,1024,1024]` square_mid | 1024 | 2 x 8 | **64 / 64** | 512 B/stick | ~22300 | ~188 |
| `[1,1,32,32768]` short_wide_wide | 1024 | 1 x 8 | **64 / 64** | 512 B/stick | ~24400 | ~172 |

**Calibration for those numbers (Refinement 3).** A production `ttnn.clone`
(pure whole-tile-page DRAM -> DRAM copy) on the SAME box and the same 64 cores
takes **15268 ns to move 2 MiB** (137 GB/s) and **87375 ns to move 16 MiB**
(192 GB/s). So 192 GB/s is this part's asymptotic ceiling and it is only reachable
at the large size; at 2 MiB the ramp and the ~660 ns dispatch floor cap a copy at
~137 GB/s. `[1,1,32,16384]` moves its 2 MiB in ~13500 ns = ~155 GB/s, i.e. **1.13x
faster than a plain copy of the same size** — the perf-focus shape is
data-movement-saturated, not under-tuned. `[1,1,2048,2048]` at ~192 GB/s is AT the
copy ceiling. The transposed pair at ~100 GB/s is the one genuinely off the
ceiling, and its 64 B read is why (Refinement 6).

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
