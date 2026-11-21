# Operation Design: padded_slice

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute: every output element is an input element, unchanged, or a zero of padding — spec R1) |
| Goal | Gather the block `input[s0:e0, s1:e1, s2:e2, s3:e3]` of a rank-4 `[N, H, W, C]` tensor as rows `[1, 1, n*h*w, width]` written straight into the caller's height-sharded L1 `memory_config`, each core filling its own shard from DRAM in one dispatch. |
| Math | `output[0, 0, r, j] = input[s0 + n_r, s1 + h_r, s2 + w_r, s3 + j]` for `j < c`, and `0` for `c <= j < width`, where `(n_r, h_r, w_r)` is the `r`-th position of the block `[n, h, w]` in row-major order and `width` = the shard width |
| References | `eval/specs/padded_slice/spec.py` (authoritative); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (ProgramSpec pattern); `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h`; `tt_metal/hw/inc/api/dataflow/noc.h`; catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `split_reader`, `read_coalescing`, `bank_stagger`, `trid_double_issue`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (worked numbers only; the op never reads the grid — its core set is the output shard grid, R10) | target page; run-time query `device.compute_with_storage_grid_size()` |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` with `l1_reserve_bytes = 0`; the op's program allocates **no** L1 (no DFB, no scratchpad), so no predicate in the op spends it — the output shard is allocated by `ttnn.allocate_tensor_on_device`, which refuses a shard that does not fit | target page |
| Host program | `ttnn.ProgramSpec` + `ttnn.generic_op` — `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (one compute thread each, C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); "R, W divide C" binds only on STRIDED rings — this op has no ring | target page, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ device operation (refused at kernel creation, target page); `bfloat8_b`/`bfloat4_b` (no path on this family — not in TARGET anyway); untagged NoC transfers: not used, every transfer is tagged with user trid 0 (below) | target page |

Worked numbers below are for this target (32 clusters). Nothing the op evaluates at run time uses them.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4 `[N, H, W, C]`; Phase 0: bfloat16, ROW_MAJOR, DRAM interleaved | — | tensor binding `input` (TensorAccessor) |
| `padded_slice_start` | list/tuple of 4 ints | yes | `0 <= start[d] <= end[d]` (R8: list or tuple) | — | host → common RT (`start_n`, `start_h`, `start_w`, `src_offset_bytes = start[3]*elem_bytes`) |
| `padded_slice_end` | list/tuple of 4 ints | yes | `end[d] <= shape[d]`; `end[d] > start[d]` (empty block refused) | — | host → common RT (`block_n` unused by kernel, `block_h`, `block_w`, `row_bytes = c*elem_bytes`) |
| `padded_slice_step` | None or 4 ints | no | None or all 1 (axis `step`: None/"ones"; anything else = "other", refused) | None | host only |
| `memory_config` (kw-only) | `ttnn.MemoryConfig` | yes | height-sharded L1 with a shard spec, shard width ≥ c (axis `to`: "L1/height" or "L1/height/padded") | — | host → output TensorSpec, core set, per-node RT (`first_row`, `num_rows`) |

Host-side checks, in this order (R6, R7, R8, R11, REJECTIONS):

| # | Check | Raise |
|---|-------|-------|
| 1 | `start`, `end` → tuples of `int` (lists accepted, R8); `len(start) == len(end) == input_tensor.rank` (X1) | `ValueError` |
| 2 | every `0 <= start[d]`, `end[d] <= shape[d]` (X2), `end[d] >= start[d]` (X3); `end[d] == start[d]` (empty block, unspecified) | `ValueError` / `NotImplementedError` for empty |
| 3 | `to = spec.canonical_to(memory_config, {"padded_slice_start": start, "padded_slice_end": end})` — itself raises `TypeError` for a non-MemoryConfig (X5) and `ValueError` for narrow shards (X6) | as raised |
| 4 | `step = spec.canonical_step(padded_slice_step, ...)` | — |
| 5 | `{memory_config, sharding} = eval.memory.memory_axes(input_tensor)` (R9) | — |
| 6 | tag `sliced_region`, `alignment`, `rank` with `INPUT_TAGGERS = AXES` on `(tuple(shape), start, end)` | — |
| 7 | `validate(axes)` against `SUPPORTED` (= PHASE0) / `EXCLUSIONS`: `to = "L1/height"` not in SUPPORTED for e.g. X4's `"L1"` → refused by name | `NotImplementedError` |

No index axis (`dim`) exists; nothing to canonicalize beyond the above.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[N, H, W, C]` (rank 4) |
| Dtype | Phase 0: bfloat16 (TARGET adds float32) |
| Layout | Phase 0: ROW_MAJOR — one page = one row of C elements, page id `((n*H + h)*W + w)` (TARGET adds TILE) |
| Memory | Phase 0: DRAM interleaved (TARGET adds L1 interleaved; sharded inputs are outside TARGET) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 1, n*h*w, width]`, `width = memory_config.shard_spec.shape[1]` (= c in Phase 0) |
| Dtype | input's |
| Layout | ROW_MAJOR |
| Memory | exactly the `memory_config` passed (height-sharded L1, its grid, shard shape, orientation — R5): `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, dtype, ttnn.ROW_MAJOR_LAYOUT, memory_config), device)`, the config object unmodified |

## Blocking Model

The op's index space is `out_row × channel`. `out_row` is the row-major flattening of the block's `(n, h, w)` — the output *is* that flattening, and every shard is a contiguous range of it — so the three leading axes fold into one honestly; the input address of a row is piecewise affine in `out_row` (consecutive page ids within a `w`-run, a jump at each `(n, h)` boundary). The caller's height-sharded config pins core-assignment and each core's extent (`shard_h` rows); what remains ours is how a shard is cut into blocks (one) and how a core's harts share it.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_row` (block rows; fold of `n,h,w`) | independent — each output row is one input row, no row reads another | `block_rows` | the whole shard: `block_rows = shard_h` → `num_blocks_this_core = 1` | host constant `MAX_BLOCK_ROWS = 0` (0 = whole shard) → common RT arg `block_rows` | pinned by the caller: shard `k` (k-th core of `shard_spec.grid` in `shard_spec.orientation` order) owns rows `[k*shard_h, min((k+1)*shard_h, rows))` | knob-turn (nothing to turn: a smaller block only adds barriers) |
| `n` (block images) | independent — folded into `out_row` | none of its own: its extent inside a block is induced by `block_rows` | induced | — (`block_n` is host-only) | via `out_row` | — |
| `h` (block image rows) | independent — folded into `out_row` | none of its own (induced) | induced | common RT `block_h` (address walk only) | via `out_row` | — |
| `w` (block image columns) | independent — folded into `out_row`; a `w`-run is a run of consecutive input page ids | none of its own (induced) | induced | common RT `block_w` (address walk only) | via `out_row` | — |
| `channel` (row width) | independent — elements of a row never interact | `row_bytes` (data) + `pad_bytes` (zeros) | whole row: `row_bytes = c*elem_bytes`, `pad_bytes = (width - c)*elem_bytes = 0` | host, from `start[3]`, `end[3]`, shard width → common RT args | not split: a height shard holds whole rows; splitting the row would fragment one page read into several | — (`channels`/padded are SUPPORTED growth, see Regimes) |
| `thread` (harts of one core) | independent — rows of a block are dealt to the core's reader harts | `gather_threads` (R) | 2 | host constant `GATHER_THREADS = 2` → kernel `num_threads` + CT arg `reader_threads` | thread `t` of R owns the contiguous sub-range `[lo_t, hi_t)` of each block, `lo_t = t*q + min(t, rem)`, `q = block_rows_this // R`, `rem = block_rows_this % R` | knob-turn (one constant, R ≤ 6) |

Every intermediate: none — rows go from DRAM to their final L1 location with no stage in between, so there is no second table to run.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| — (no DataflowBuffer exists) | transactions in flight per hart: not a buffer, set by the NoC's own back-pressure (`Noc::async_read<TXN_ID>` waits while fewer than half the trid counters are free, `noc.h:252-262`) and by `block_rows` (one barrier per block) | whole block in flight, one barrier per block | the read pipeline depth; the catalog (`double_buffer`) measures gains flattening from ~4 in flight, which a whole-shard block exceeds |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Interleaved page placement (input pages round-robin across DRAM banks) | one NoC read covers at most one input page, i.e. one row (`out_row` transfer quantum = 1 row) | the gather issues one read per row; rows are never coalesced | a multi-row read would read past the page into another tensor's bytes in that bank: wrong values |
| NoC read size vs. burst | none — `noc_async_read_impl` splits any length (`dataflow_api_raw.h:62-75`, `ncrisc_noc_fast_read_any_len`) | — | — (16 KiB bf16 / 32 KiB fp32 rows are legal single calls) |
| NoC alignment | none on Quasar: L1 and DRAM read alignment is 1 B (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:486-491`) | — | — (shard-row offsets and a mid-page `start[3]` offset need no staging) |
| Output shard row stride | the row stride in the shard is the output buffer's aligned page size, not `row_bytes + pad_bytes` | common RT `out_row_stride = output.buffer_aligned_page_size()` | a width whose bytes are not a multiple of 16 B would place rows at the wrong offsets |
| Rows of a block per thread | `hi_t - lo_t` may be 0 (block with fewer rows than R) | the thread's loop simply does nothing | none — no ring, so no equal-entries rule; an idle hart costs nothing |
| iDMA zero mode (padded regime only) | `async_write_zeros` reprograms write cmd buffer 0 until `write_zeros_l1_barrier()` (`noc.h:816-849`) | no NoC write is issued between the zero and its barrier (the gather issues only reads) | a NoC write in that window would carry zeros |
| 32-bit index fields | `first_row`, page ids, byte offsets fit `uint32_t` | host asserts `N*H*W < 2^32` and shard bytes `< 2^32` | wraparound → wrong rows |

### Regimes

The minimum is stated against DRAM → final L1 location: every byte of the block crosses DRAM once (read) and is written once, into its final place in the shard; nothing else moves.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **A. RM row gather into own shard** | **built** (Phase 0: `layout == ROW_MAJOR ∧ to == "L1/height"`; `sliced_region == "rows"` only because SUPPORTED says so — the kernel already takes `src_offset_bytes`, so `channels` is a SUPPORTED widening of this regime, and so are float32 (byte copy) and an L1 interleaved input (TensorAccessor args)) | `layout == ROW_MAJOR ∧ to == "L1/height"` | `block_rows × (row_bytes)`, `block_rows = shard_h` | **minimum**: block bytes cross DRAM once, land in the shard once; zero cross-core, zero intermediate L1 | one `async_read_barrier` per block (intended: once per core per hart) and one `(n,h,w)` decomposition per thread per block; pipeline fill/drain once. No inits, no handshakes exist. |
| **B. RM row gather, padded rows** | deferred — `to == "L1/height/padded"` is outside PHASE0; built **on top of** A by one block op `zero_block` before `gather_rows_block` (common RT `pad_bytes > 0` selects it) | `layout == ROW_MAJOR ∧ to == "L1/height/padded"` | same as A, row stride = shard width | minimum + one core-local zero write of each thread's range (`(hi_t-lo_t)*out_row_stride` B of L1, no NoC, no DRAM) — the shard memory holds stale data, so the zeros must be written | the zero is one iDMA transaction + one `write_zeros_l1_barrier` per thread per block |
| **C. TILE segment gather** | deferred — TILE input is outside PHASE0; built on A's kernel: the per-row read becomes `ceil`-many per-face-row segment reads (16 elements each, `noc` 1-B alignment makes a mid-segment `start[3]` legal) into the same shard offsets; tile indices per image use `ceil(W/32)`, `ceil(C/32)` | `layout == TILE ∧ to ∈ {L1/height, L1/height/padded}` | `block_rows × segments_per_row` | minimum DRAM bytes (only logical elements are read — R4), but `≈ ceil(c/16)+1` transactions per row instead of 1 | same as A |
| **D. TILE row untilize (staged)** | deferred — TILE outside PHASE0; it is the alternative to C for wide `c` (where C's segment count per row is large), and only a timed backend can rank transaction count against over-read; reachable because A/C leave the shard-gather contract unchanged — D adds a reader ring + compute untilize (`untilize_helpers.hpp`) + a local copy into the shard | `layout == TILE ∧ c_extent >= untilize_min_c` (threshold to measure) | tile-rows covering the block's `w`-range | reads whole tiles covering `[w0, w1) × [c0, c1)` (over-read up to `32/block_w` on the `w` axis) + one L1→L1 copy | per-block untilize init + ring handshakes |
| E. Staged gather (reader → DFB ring → writer copy into shard) | rejected — superseded by A: identical DRAM traffic plus one extra L1 write + read per byte, a ring and a second kernel for nothing (dead end: A is built instead of it) | — | — | +1 L1 hop per byte | — |
| F. Helper cores beyond the shard grid (read DRAM, NoC-write to the owner shard) | deferred — positive reason: it adds a cross-core crossing for every byte and converges on the owner core's L1 ingress, and the cheaper lever for the same limit (per-core issue rate when the caller pins 1–2 cores, R10) is more harts on the owner core (`gather_threads` up to 6), which adds no bytes; reopened only if a timed backend shows 6 harts on one owner core still issue-bound. Reachable: the per-node RT args (`first_row`, `num_rows`) already describe any row range, so a helper node is a different destination accessor, not a new walk | `num_shard_cores` small ∧ issue-bound | rows of an owner shard dealt to helpers | +1 cross-core crossing per byte | — |
| G. Split `channel` across harts or cores | rejected — superseded by A + `thread` split over rows: splitting a row doubles the transaction count per byte, and a height shard cannot be split across cores by width | — | — | same bytes, 2× transactions | — |
| H. Coalesced multi-row reads | rejected — structurally unreachable for an interleaved input (consecutive pages live in different banks; catalog `read_coalescing` rejects interleaved for this reason); sharded inputs are outside TARGET | — | — | — | — |

Regime selection (host, exact): `layout == ROW_MAJOR` → A (B when `pad_bytes > 0`); `layout == TILE` → C (D when its threshold is set and met). Phase 0 has one built row; the `pad_bytes`/`layout` selectors are common RT / CT values from day 1, so B and C are a code path added under an existing switch. When B, C or D is built, its cells are its regime-pinned tests (the golden suite's `to` and `layout` axes pin them; no grid dependence exists because the core set is the caller's).

### Traffic ranking

| Candidate split | DRAM crossings (input) | Cross-core | Core-local | Verdict |
|-----------------|-----------------------|------------|------------|---------|
| Owner-computes: each shard core reads its own rows into its own shard (A) | 1× block bytes | 0 | 0 (reads land in place) | **chosen — the minimum** |
| Same, but staged through a ring (E) | 1× | 0 | +1 write +1 read of every byte | rejected |
| Helper cores push into owner shards (F) | 1× | +1× block bytes, fan-in onto owner | +1 staging | deferred (above) |
| Split `channel` (G) | 1× | 0 | 0, but 2× transactions | rejected |
| Split the dependent axis + combine | n/a — the op has no dependent axis (no reduction, no scan) and no reuse-shared operand (one input; no core reads another core's rows) | — | — | — |

Operand-reuse check: the only operand, `input_tensor`, varies along `out_row` (each core reads different pages) — not reuse-shared by the split. Stall-shadow check: the only wait is the end-of-block read barrier; nothing in this op is independent of the reads it waits for (there is no compute), so nothing can be scheduled into it. In regime B the zero barrier precedes the gather because both write the same bytes; the alternative that removes that wait (zeroing only each row's tail, disjoint from the data bytes, interleaved with the reads) is lamp 3.

### Block schedule

```cpp
// per core (shard k), per hart t of R — logical schedule; one DM kernel realizes all of it
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // 1 in Phase 0
    zero_block(block_idx);          // regime B only (pad_bytes > 0)
    gather_rows_block(block_idx);   // DRAM pages → own shard rows, one barrier
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `zero_block` (B only) | hart t's rows `[lo_t, hi_t)` × `out_row_stride` bytes, contiguous in the shard | the output shard (caller-owned) | one `async_write_zeros` + one `write_zeros_l1_barrier` per hart per block |
| `gather_rows_block` | hart t's rows `[lo_t, hi_t)` × `row_bytes` | the output shard | accessor construction once per kernel; `(n,h,w)` decomposition of `first_row + block_start + lo_t` once per hart per block (then counters: `w` increments, page id +1; on a `w`-run wrap recompute the page id); one tagged `async_read` per row (the mechanism's quantum, cap 1); **one** `async_read_barrier<TXN_ID>({.trid=0})` per hart per block; tag hand-back once at kernel end |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| 1. Thread count (`GATHER_THREADS = 2`) | the op is bound by per-transaction issue on a caller-pinned core set (32–64 B pages; the traced stem call is 6,785 rows on one core); catalog `split_reader` measured 1.74× at 32 B / 1.64× at 64 B from a second issuer (WH), but on 32 cores × ≤ 64 rows each the second hart may buy nothing | `GATHER_THREADS ∈ {1, 2, 4, 6}` on the single-core stem config and on a golden 32-core config (model cycles on `ttsim-perf`) |
| 2. NoC per hart | both harts issue on the reader default NoC; with R ≥ 2 the second NoC is idle | hart `t` issues on NoC `t % 2` (`Noc(t % 2)`), same kernel |
| 3. Padded zeroing (regime B) | one zero of the whole range then a barrier serializes zeroing before the gather | per-row tail zeros (`pad_bytes` each, disjoint from data bytes) interleaved with the reads, no barrier between |

Overlap lamp: does not apply (no compute to overlap; one block is already the deepest read pipeline). Grid-synchronization lamp: does not apply (no collective; the core count is the caller's).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input in DRAM | RM pages, one row of C elements each (`input.buffer_aligned_page_size()` apart in its bank) | `TensorAccessor(tensor::input)` | page id `((s0+n_i)*H + s1+h_i)*W + s2+w_i` |
| DRAM → own shard | bytes `[src_offset_bytes, src_offset_bytes + row_bytes)` of the page | `noc.async_read<NocOptions::TXN_ID>(acc_in, out_shard, row_bytes, {.page_id, .offset_bytes = src_offset_bytes}, {.offset_bytes = local_row * out_row_stride}, {.trid = 0})` | destination is `LocalTensorAccessor<uint8_t>(tensor::output)` — this core's shard, written in place: no DFB, no NoC re-read, no writer |
| Shard (result) | RM rows of `width` elements, `out_row_stride` apart | caller's height-sharded L1 buffer | row `r` of the result is row `r - k*shard_h` of shard `k` (R5) |

Ring protocol: the kernel uses **no ring**. Every transfer carries the user transaction id (`NocOptions::TXN_ID`, `{.trid = 0}`), the barrier is on the same id, and the kernel's last statement is `noc_async_read_set_trid(0, noc.get_noc_id())` — so the kernel neither inherits a ring's tag from a previous implicit-sync program nor leaves one for a following untagged program (target page, *Ring protocols of a data-movement kernel*).

TARGET growth, classified against this op's axis characters:

| TARGET value outside PHASE0 | Class | What changes |
|-----------------------------|-------|--------------|
| `to = "L1/height/padded"` | knob-turn on the same placement (regime B: one block op added under `pad_bytes > 0`) | host `pad_bytes`, kernel `zero_block` |
| `sliced_region = "channels"` | knob-turn (common RT `src_offset_bytes = start[3]*elem_bytes`, `row_bytes = c*elem_bytes`; already threaded in Phase 0) | SUPPORTED only |
| `dtype = float32` | knob-turn (`elem_bytes = 4`; bytes are copied, never converted — R1) | SUPPORTED only |
| `memory_config = "L1"` (input) | knob-turn (TensorAccessor CT args describe an L1 interleaved tensor) | SUPPORTED only |
| `step = None` | host only | SUPPORTED only |
| `layout = TILE` | new address map (regime C), on the same kernel and shard contract | kernel path under a CT `input_is_tiled` |
| sharded input / width- or block-sharded destination | outside TARGET | — |

Placement: the destination's only TARGET sharding is height, which cuts the independent `out_row` axis — every core's work is self-contained, so the physical shard **is** the logical split; no placement value is a scheme-change.

Structural impossibilities: none to add to INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output row (an input page); a core's block = its whole shard |
| Grid | the output shard grid, `memory_config.shard_spec.grid` — never the device grid (R10); `WorkUnitSpec.target_nodes` = that core set |
| Per-core work | shard index `k` = position of the core in `ttnn.corerange_to_cores(grid, None, row_wise = (orientation == ROW_MAJOR))`; per-node RT `first_row = k*shard_h`, `num_rows = clamp(rows - k*shard_h, 0, shard_h)` |
| Remainder | a short last shard gets `num_rows < shard_h`; cores past the result get `num_rows = 0`. Row-major rows are not tiles; for regime C the per-image tile indices are `ceil(W/32)`, `ceil(C/32)` |
| Bound | data movement — per output row there is one NoC read of `row_bytes` (32 B – 16 KiB) and no arithmetic at all; with 32–64 B rows the limit is transaction issue on the cores the caller pinned, not DRAM bandwidth and not compute |
| Thread plan | (R, C, W) = (2, 0, 0), host constant `GATHER_THREADS = 2`: no compute kernel (nothing to compute) and no writer (reads land in the final location); two reader harts because the bound is transaction issue on a pinned core set and harts are the lever for it (catalog `split_reader`); lamp 1 measures 1/4/6 |
| Unit dealt to threads | rows, in contiguous sub-ranges of each block (`lo_t = t*q + min(t, rem)`); not a ring entry, so no round padding and no whole-round host split — a hart with 0 rows idles; the core set is the caller's and is not re-dealt |
| Tests the plan needs | (1) **threads ran**: a probe-only kwarg `_threads` (not in the public signature) runs `GATHER_THREADS ∈ {1, 2, 6}`; the test first allocates an L1 tensor with the same `memory_config`, fills it with a sentinel (e.g. 1e4), deallocates it, then calls the op (the allocator reuses the shard), and asserts exactness — a hart that did not run leaves a contiguous run of sentinel rows; it also asserts the built `ProgramSpec`'s gather kernel has `num_threads == R`. Use a shape with ≥ 6 rows per core (e.g. `(1,64,32,32)` block `(1,32,32,32)` → 32 rows/core). (2) **session**: in one device session, run a tt-metal program first (`ttnn.to_layout(<a bf16 TILE tensor in DRAM>, ttnn.ROW_MAJOR_LAYOUT)`) then the op; and the op then `ttnn.to_layout(<a bf16 TILE tensor>, ttnn.ROW_MAJOR_LAYOUT)`; check both results |

Worked Phase 0 values on this target (runner's rule: largest core count ≤ 32 dividing the rows; one row = one page):

| INPUTS entry (block) | rows | row bytes | shard cores | rows per core (`block_rows`) | DRAM bytes read |
|----------------------|------|-----------|-------------|------------------------------|-----------------|
| `(1,115,115,32)` → `(1,59,115,32)` | 6,785 | 64 | 23 | 295 | 434,240 |
| `(1,64,32,32)` → `(1,32,32,32)` | 1,024 | 64 | 32 | 32 | 65,536 |
| `(1,32,16,16)` → `(1,16,16,16)` | 256 | 32 | 32 | 8 | 8,192 |
| `(1,32,32,64)` h 16:32 | 512 | 128 | 32 | 16 | 65,536 |
| `(2,16,16,32)` n 1:2 | 256 | 64 | 32 | 8 | 16,384 |
| `(1,16,64,32)` w 16:48 | 512 | 64 | 32 | 16 | 32,768 |
| `(1,17,50,64)` h 3:12, w 7:40 | 297 | 128 | 27 | 11 | 38,016 |
| `(1,8,512,16)` h 2:6 | 2,048 | 32 | 32 | 64 | 65,536 |
| `(2,4,8,4096)` n 0:1 | 32 | 8,192 | 32 | 1 | 262,144 |
| `(1,2,8,8192)` h 1:2 | 8 | 16,384 | 8 | 1 | 131,072 |
| `(1,4,4,16)` h 0:2 | 8 | 32 | 8 | 1 | 256 |

The traced caller configs (R10) put 6,785 rows on one core and 512/128 rows on two: the same kernel, different per-node RT args.

## Circular Buffers

No DataflowBuffer, no scratchpad: the reads land in the output tensor's own shard. The only L1 the op touches is that tensor region, listed for the ledger:

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `output_shard` (tensor region, not a DFB; `LocalTensorAccessor(tensor::output)`) | — (tensor binding `output`) | `out_row_stride` = output aligned page size | `shard_h` (caller's) | spans `out_row` → `shard_h` and `channel` → `width`; harts stream over it | input dtype, bytes only (never through DEST) | `gather` (reader harts) | none in this program (the next op) | user trid 0 tagged reads + `async_read_barrier<TXN_ID>` (no ring credits) | whole program, and after |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `zero_block` (regime B, not Phase 0) | `[lo_t, hi_t) × out_row_stride` | no (dataflow NoC API) | — | `output_shard`, rows `[lo_t, hi_t)` | zeroed, barrier passed |
| 2 | `gather_rows_block` | `[lo_t, hi_t) × row_bytes` | no (dataflow NoC API) | input DRAM pages via `TensorAccessor` | `output_shard`, rows `[lo_t, hi_t)` at `local_row*out_row_stride` | complete after the hart's barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host: program | raw_api | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=GATHER_THREADS, hw_config=create_reader_datamovement_config(), tensor_bindings=[input, output], compile_time_args={"reader_threads": R}, runtime_arg_schema=RuntimeArgSchema(runtime_arg_names=["first_row","num_rows"], common_runtime_arg_names=[...]))`, `WorkUnitSpec(target_nodes=shard grid)`, `ttnn.generic_op` | pattern `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:72-158`; schema `ttnn/cpp/ttnn-nanobind/program_spec.cpp:632-648`; reader config `program_spec.cpp:470`; DM-only work unit legal `tt_metal/impl/metal2_host_api/program_spec/validation/placement/work_unit.cpp:62-99` | `dataflow_buffers=[]` | — | — | `GATHER_THREADS`, `block_rows` (common RT) |
| host: output | raw_api | `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(shape, dtype, layout, memory_config), device)` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:288-305` | config object unmodified (R5) | — | `output_shard` | — |
| host: core order | raw_api | `ttnn.corerange_to_cores(grid, None, row_wise)` | as used at `toy_thread_plan_program_artifacts.py:57` | `row_wise = orientation == ROW_MAJOR` | — | — | — |
| kernel: shard handle | raw_api | `LocalTensorAccessor<uint8_t>(tensor::output)` | `tt_metal/hw/inc/api/tensor/local_tensor_accessor.h:55-77`; NoC endpoint `tt_metal/hw/inc/internal/local_mem/noc_traits.h:104-127` | byte element type → offsets in bytes | — | `output_shard` | — |
| kernel: input | raw_api | `TensorAccessor(tensor::input)` | `tt_metal/hw/inc/api/tensor/noc_traits.h:135-170` (`page_id`, `offset_bytes` src args) | — | input | — | — |
| `gather_rows_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(acc_in, out_shard, row_bytes, {.page_id, .offset_bytes = src_offset_bytes}, {.offset_bytes}, {.trid = 0})` | `tt_metal/hw/inc/api/dataflow/noc.h:245-280` | trid 0 (user id) | input | `output_shard` | rows `[lo_t, hi_t)` from `block_rows`, R |
| `gather_rows_block` end | raw_api | `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:738-745` | once per hart per block | — | — | `block_rows` sets the frequency |
| kernel end | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558` | last statement | — | — | — |
| thread id | raw_api | `get_my_thread_id()` | `tt_metal/hw/inc/api/kernel_thread_globals.h:155` | with CT `reader_threads` (the plan's count, not `get_num_threads()`) | — | — | R |
| `zero_block` (B) | raw_api | `Noc::async_write_zeros(out_shard, bytes, {.offset_bytes})` + `write_zeros_l1_barrier()` | `noc.h:816-849`, `:904` | no NoC write between them | — | `output_shard` | — |

Helpers considered and rejected (applies to every `raw_api` row above): the kernel library's helpers are compute helpers (`tilize`, `untilize`, `reduce`, `eltwise_*`, `matmul_block`, `bias_add`, `reblock_untilize`, `sfpu_activation`) — this op has **no compute phase** (spec DEFINITION: "moves values and computes nothing"; GUIDANCE: "No compute"), so none has a phase to cover; `untilize_helpers.hpp` is named for regime D only. The one dataflow helper, `mcast/kernel/mcast_args.hpp`, moves one payload to many receivers; no byte here goes to more than one core (traffic ranking: no reuse-shared operand). Data movement is expressed through the `Noc` / `TensorAccessor` / `LocalTensorAccessor` API, which is the dataflow layer helpers would wrap.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Shard index ↔ core order (R5) | the result is checked by placement: row `r` must sit in shard `r // shard_h` on the config's k-th core | `k` from `corerange_to_cores(shard grid, row_wise = orientation == ROW_MAJOR)`; never from the device grid or `split_work_to_cores` |
| Result's memory config must equal the one given | `HOOKS["check"]` compares `output.memory_config()` with the cell's config, shard shape included | allocate the output from a `TensorSpec` carrying the passed `memory_config` object unchanged |
| Tag register | a previous implicit-sync program's tag would count our transfers for a ring | every read `NocOptions::TXN_ID` with `{.trid = 0}`, barrier on trid 0, `noc_async_read_set_trid(0, ...)` last; the session test |
| Read size vs. stride | `row_bytes` (what is read) ≠ DRAM aligned page size ≠ `out_row_stride` in general (C=16 bf16: 32 B read, 64 B DRAM page) | read exactly `row_bytes`; TensorAccessor owns the DRAM page stride; shard stride is the output's aligned page size |
| Kernel recompiles per shape | geometry as CT args would compile once per INPUTS entry on a slow simulator | geometry, offsets and row counts are common/per-node RT args; CT args are only `reader_threads` and the accessor args |
| `(n,h,w)` walk across shard and thread boundaries | a shard (and a hart's sub-range) can start mid `w`-run (e.g. 11 rows/core with `block_w = 33`) | decompose the hart's first row with div/mod once, then advance counters; test entry `(1,17,50,64)` exercises odd starts and odd runs |
| `canonical_to` needs the call's bounds | it reads `padded_slice_start`/`end` from its `arguments` dict to compare widths (X6) | pass `{"padded_slice_start": start, "padded_slice_end": end}` after the X1–X3 checks |
| Bounds passed as lists (R8) | the loose case passes lists | normalize to tuples of ints first |
| A DM-only work unit at launch | no example in the reference set runs without a compute kernel | validation counts kernels only (`work_unit.cpp:62-99`); if launch refuses it, add a compute kernel that returns immediately (no DFB, `fpu_math_fidelity` set) — the design is otherwise unchanged |
| Single-core shard of 434 KB (traced stem call) | L1 is the caller's; the op adds nothing | no op-owned L1; allocation failure is the allocator's error |
