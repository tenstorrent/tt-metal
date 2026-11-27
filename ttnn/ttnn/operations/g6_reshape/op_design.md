# Operation Design: g6_reshape

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic, bit-exact byte moves) |
| Goal | `torch.reshape` on the logical values of a TILE (Phase 0) tensor, re-laid-out in its own layout under the new shape, in **one** device program that reads every logical input value from DRAM once and writes every logical output value to DRAM once. |
| Math | `out[j] = x[i]` whenever `lin(j; e) = lin(i; d)`, `lin(i; d) = Σ_k i_k · Π_{l>k} d_l`; `e` = `shape` with one `-1` resolved to `N / Π(other entries)`. |
| References | `eval/golden_tests/g6_reshape/feature_spec.py` (TARGET / INPUTS / LOOSE_CASES), `eval/golden_tests/g6_reshape/helpers.py` (oracle, exact gate), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/noc-alignment-in-tt-metal.md`, `ttnn/ttnn/operations/examples/master.md` (entries `tile_reorder`, `double_buffer`, `trid_double_issue`, `split_reader`, `noc_placement`, `bank_stagger`), `l1_ledger.md` (this directory). No exploration bundle is installed (`docs/explorations/g6_reshape/` absent). |

### Hardware facts this design rests on (measured on this box, Blackhole)

| Fact | Value | Source |
|------|-------|--------|
| Worker grid | 11 × 10 = 110 cores | `device.compute_with_storage_grid_size()` (probe) |
| DRAM read congruence | src ≡ dst (mod **64**) | `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379` |
| DRAM write congruence | src ≡ dst (mod **16**) | `noc_parameters.h:380` |
| L1↔L1 congruence | mod 16 | `noc_parameters.h:375-376` |
| RISC-V L1 data cache | write-through, reader of NoC-written data must `invalidate_l1_cache()` | `tt_metal/hw/inc/internal/tt-1xx/cache.h:7-16` |
| `cb_wait_front` | does **not** invalidate the L1 cache | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:479-492` |
| TILE page | 32×32, 4 faces of 16×16, row-major inside a face; bf16 2048 B, fp32 4096 B; DRAM buffer base ≡ 0 (mod 64) | probe |
| Rank-1 TILE tensor `(W,)` | padded `[32, round_up(W,32)]` — one logical row | probe |
| RM int32 page `(4096,1)` | page 4 B, aligned page 64 B | probe |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1–4; Phase 0: bfloat16, TILE, DRAM interleaved | — | buffer address: RT |
| `shape` | `list[int]` or `tuple[int]` (positional) | yes | length 1–4; entries ≥ 1 except at most one `-1`; product (after resolution) = N | — | resolved output dims: CT |

Host-side validation (entry point, `ValueError`): `shape` not a list/tuple of ints; length outside 1–4; an entry `0` or `< -1`; more than one `-1`; `N % Π(other entries) != 0` when resolving `-1`; product ≠ N; input rank outside 1–4; input not DRAM interleaved (Phase 0 input requirement — not a TARGET axis). Order: `validate()` (registry refusals) is the **first line**; shape validation follows. Every host
validation message starts with `g6_reshape:` (the acceptance test matches it through `expect_error`).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `d = (d_0 … d_{n-1})`, n ∈ [1, 4], no alignment requirement on any dim |
| Dtype | Phase 0: `ttnn.bfloat16`. TARGET adds `ttnn.int32`, `ttnn.uint32` (with ROW_MAJOR), `ttnn.float32` (with TILE) |
| Layout | Phase 0: `TILE_LAYOUT`. TARGET adds `ROW_MAJOR_LAYOUT` (int32/uint32) |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `e` (`shape` with `-1` resolved) |
| Dtype | input dtype (no cast) |
| Layout | input layout (no conversion) |
| Memory | `input_tensor.memory_config()` (DRAM interleaved); allocated with `ttnn.allocate_tensor_on_device` — the only device allocation of the op |

### Registry axes (names as in `feature_spec.py` TARGET)

| Axis | Tagger | Reads | `validate()` reads | Phase 0 SUPPORTED | TARGET − SUPPORTED (refinements) |
|------|--------|-------|--------------------|-------------------|----------------------------------|
| `dtype` | `tag_dtype(inputs, axes)` | `inputs[0]["dtype"]` of the scenario dict | `input_tensor.dtype` | `ttnn.bfloat16` | `ttnn.int32`, `ttnn.uint32` (added together with `ROW_MAJOR_LAYOUT`), `ttnn.float32` (with `TILE_LAYOUT`) |
| `layout` | `tag_layout(inputs, axes)` | `inputs[0]["layout"]` | `input_tensor.layout` | `ttnn.TILE_LAYOUT` | `ttnn.ROW_MAJOR_LAYOUT` |

`EXCLUSIONS = []`. `validate(input_tensor, shape)` builds the scenario dict
`{"input_shape", "shape", "dtype", "layout"}` from the call, applies the taggers, and raises
`UnsupportedAxisValue` for a value outside SUPPORTED. The int32/uint32 row-to-column production cases
(including the four perf-flagged `[1, W]`) stay in TARGET and are refused as unsupported axis values until
the ROW_MAJOR refinement — never excluded, never tagged away. No shape-derived axis exists: every shape of
a supported (dtype, layout) runs.

### Shared index map (used by host, reader and writer)

For a tensor X with logical shape `s` (rank n) and element size `es`:

| Symbol | Definition |
|--------|-----------|
| `W_X` | `s[n-1]` |
| `H_X` | `s[n-2]` if n ≥ 2 else 1 |
| `B_X` | `Π s[0..n-3]` if n ≥ 3 else 1 |
| `Ht_X`, `Wt_X` | `ceil(H_X/32)`, `ceil(W_X/32)` — per image, alignment-aware |
| logical row `r` | `r ∈ [0, B_X·H_X)`, `b = r / H_X`, `h = r % H_X` |
| page of `(r, c)` | `(b·Ht_X + h/32)·Wt_X + c/32` |
| in-page byte offset of `(r, c)` | `(((h%32)/16)·2 + (c%32)/16)·256·es + (h%16)·16·es + (c%16)·es` |
| **line** `(r, g)` | the face-row holding columns `[16g, 16g+16) ∩ [0, W_X)` of row `r`: `line_bytes = 16·es` contiguous bytes at the page/offset of `(r, 16g)` |

A line is the largest unit that is contiguous in DRAM for an arbitrary reshape (one face row). Every
logical output value lives in exactly one output line; every move of this op is a move of lines or of
element segments of lines.

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### The op's work, stated in output terms

The op is a bijection between logical positions, so the work is indexed by the **output**: each output
line is produced from the input line(s) holding the same linear positions. Ownership is keyed on the
output because a 64 B DRAM word must have one writer (see Mechanism caps); input lines are read by
whichever core owns the output line they feed. The output work index space is
`(out_tile_col, out_image, out_row, face_col, line_elem)`. The first three are **flattened into one
granule axis** in column-major order — `u = (tc·B_out + b)·S + s` — which is honest because the op
treats every output position identically (a copy), and column-major is chosen so that consecutive
granules of one core read consecutive rows of the same input face (traffic ranking below).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_tile_col` (`tc`, extent `tensor_out_tile_cols = ceil(W_out/32)`) | independent — each output column group is a copy of disjoint input positions, no cross-block combine | flattened into `block_granules` (outermost component of `u`) | whole (all tile columns of the core's granule range) | `block_granules` (host → CT) | split across cores as part of `u` | knob-turn |
| `out_image` (`b`, extent `tensor_out_images = B_out`) | independent — same reason | flattened into `block_granules` (middle component of `u`) | whole | `block_granules` | split across cores as part of `u` | knob-turn |
| `out_row` (`h` within an image, extent `tensor_out_rows = H_out`) | independent — same reason | `rows_per_granule` (`R_g`, slab height) — also the ownership quantum | `R_g = rows_per_word` (= 2 for 2-byte dtypes, 1 for 4-byte); assemble regime with `W_out < 16`: `R_g = max(rows_per_word, min(16, next_pow2(ceil(16/W_out))))` | host constant → CT arg | slabs are the innermost component of `u`, split across cores | knob-turn |
| granule (`u`, flattened `tc × b × slab`, extent `total_granules = Wt_out·B_out·ceil(H_out/R_g)`) | independent | `block_granules` — granules per pipeline pass | the core's whole assignment (`max granules per core`), clamped by `BLOCK_L1_CAP` (see Mechanism caps); every production shape: one block per core | host → CT arg | contiguous ranges, `ttnn.split_work_to_cores(grid, total_granules, row_wise=True)`; grid = full `compute_with_storage_grid_size()` (runtime parameter, never a literal) | knob-turn |
| `face_col` (`fc`, extent `min(2, ceil((W_out − 32·tc)/16))` per tile column) | independent | `face_cols_per_pass` | 2 — whole tile width (both face columns of every granule in the block, `fc` walked outer inside the block) | fixed by the tile format; derived, not a literal in loops (`TILE_WIDTH / FACE_WIDTH`) | not split (both face columns of a granule have the same owner) | none needed |
| `line_elem` (extent 16 = `FACE_WIDTH`) | independent | `line_elems` | 16 — the whole line | fixed by the tile format (`FACE_WIDTH`) | not split: a line is the indivisible transfer of the direct regime; the assemble regime cuts it into element segments **inside** the block operation | none needed |

`out_row` Phase 0 value is the minimum legal granule (`rows_per_word`) — this is the **ownership quantum**,
not the block. The block (pipeline pass) is `block_granules` granules = the core's whole assignment, i.e.
the coarsest that fits. A fine ownership quantum costs nothing per granule (no handshake, no barrier is
per-granule) and buys load balance: `(16,1,5120)→(1,1,16,5120)` puts at most 48 lines on a core with
`R_g = 2` versus 64 with whole-tile granules (`R_g = 32`), and the production shapes with only 2 output
tiles still spread over 8–16 cores, the 48-tile shapes over 48–110. The assemble-regime raise for `W_out < 16` is a traffic
rule, not an occupancy rule: one input line feeds `ceil(16/W_out)` consecutive output rows, and a granule
at least that tall keeps each input line on one core (FLUX `(1,512)→(1,1,512,1)`: 32 reads for 32 input
lines instead of 128).

**Every stage's own data, re-tabled.** The scheme has two internal products: the staged source lines
(`cb_staged_lines`) and the writer's plan/assembly scratch (`cb_writer_scratch`). Both are indexed by the
same block (granule × row × face col × line), live on the core that owns the block, and are produced and
consumed by that core's two RISCs — no stage collects onto a designated core, so no core waits while
another computes.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_staged_lines` | `staging_depth` | `1` when every core has one block (all golden INPUTS), else `2` | at 2: the reader stages block `i+1` while the writer stores block `i` |
| `cb_writer_scratch` | none (writer-private, one block at a time) | 1 | — |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Single writer per 64 B DRAM word (a bf16 face holds 2 rows per 64 B; we do not rely on the DRAM controller merging two cores' 32 B partial writes to one word) | `rows_per_granule` | `R_g` is a multiple of `rows_per_word = max(1, 64 / line_bytes)`; slabs start at multiples of `R_g` within the image | two cores write the two halves of one 64 B word; correctness then depends on undocumented DRAM write-merge behaviour (risk of a lost half-word that varies run to run, BB-06) |
| Granule must not straddle a 32-row tile row | `rows_per_granule` | `R_g ∈ {1, 2, 4, 8, 16}` (a power of two dividing 32) | a slab spans two tile rows; the per-granule `(tile row, in-tile row)` math is wrong → wrong rows |
| DRAM read congruence mod 64 (`noc_parameters.h:379`) | placement of every read run in `cb_staged_lines` | run placed at `align_up(cursor, 64) + (S mod 64)`, `S` = the run's source NoC address (low bits, from the input accessor) | a bf16 face row at an odd in-face row (`S ≡ 32 mod 64`) lands 32 B off → silent corruption without the watcher |
| DRAM write congruence mod 16 (`noc_parameters.h:380`) | which lines may be written straight from staging | direct writes only under `P_direct` (every source line starts at element 0 of the output line); otherwise the assemble regime | a write whose source and destination differ mod 16 is corrupted |
| Staging / scratch must fit the op's L1 share (BB-05: 1 347 776 B) | `block_granules`, `staging_depth` | `BLOCK_L1_CAP = 1 MiB`: one block of the whole assignment if `staging_page_bytes(n_max) + writer_scratch_bytes(n_max) ≤ CAP` (depth 1); else depth 2 and the largest `block_granules` with `2·staging_page_bytes + writer_scratch_bytes ≤ CAP`. One granule needs ≤ 21 KB (worst case `R_g = 16`, `max_src_lines = 16`, fp32: 16 KB staging + 4.3 KB scratch), so `block_granules ≥ 1` always fits and **no shape is ever refused for L1** | out-of-memory at launch for large N |
| Single-packet stateful NoC APIs (if the realization uses them) | read/write run bytes | run ≤ `NOC_MAX_BURST_SIZE` (`noc_parameters.h:290`); longer merged runs use the multi-packet `noc_async_read` / `noc_async_write` | truncated transfer |

### Regimes

Named boundary for every row: **DRAM ↔ L1**. Minimum = each logical input line crosses it once (read) and
each logical output line crosses it once (write), at line (face-row) granularity; no cross-core traffic.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `face_row_direct` | **built** (Phase 0 primary — every bf16 production shape) | `layout == TILE and (W_in == W_out or (W_in % 16 == 0 and W_out % 16 == 0))` — then every output line is exactly one whole input line | `block_granules × min(R_g, H_out) rows × face_cols_per_pass × 16 elems` | **minimum**: reads exactly the source lines (merged into runs where consecutive in one input page), writes exactly the valid output lines; no cross-core traffic; no RISC-V copy. Transaction count = read runs + write runs (per-case numbers in `l1_ledger.md`) | per block: 1 `cb_reserve_back` + 1 `noc_async_read_barrier` + 1 `cb_push_back` (reader), 1 write-plan pass + 1 `cb_wait_front` + 1 `noc_async_writes_flushed` + 1 `cb_pop_front` (writer), and one DRAM read latency + one write latency of pipeline fill/drain. Intended: once per block = once per core on every golden shape |
| `face_row_assemble` | **built** (Phase 0 — correctness for every bf16 TILE shape outside `P_direct`; no bf16 production shape lands here, pinned by the acceptance test with FLUX geometry in bf16; FLUX float32 lands here after the dtype refinement) | `layout == TILE and not P_direct` | same block | reads every source line the block needs (deduped when consecutive), so an input line that feeds output lines of two adjacent cores is read by both (≤ 2× for lines on a core boundary; the `R_g` raise for `W_out < 16` keeps it at 1× for the FLUX shapes); writes = minimum; core-local: every logical element passes once through RISC-V loads/stores (`N·es` bytes) | as above, plus the per-block `invalidate_l1_cache()` before the first RISC-V load of staged data |
| `tile_page_copy` | deferred — no INPUTS shape satisfies the predicate (every production reshape changes the tile-row structure), and `face_row_direct` already moves such shapes at the minimum DRAM bytes with face-sized merged runs; this regime only collapses 4 face runs into one page transfer | `W_in == W_out and (B_in == 1 or H_in % 32 == 0) and (B_out == 1 or H_out % 32 == 0)` (identical tiled byte images) | `block_granules` whole tiles | minimum bytes, whole-page transactions | 1 read + 1 write per page instead of 4 + 4 |
| `rm_line_direct` / `rm_line_assemble` | deferred — ROW_MAJOR (int32/uint32) is outside Phase 0 SUPPORTED; reachable on the same walk/placer/writer structure with **line = RM page segment** (line bytes = the byte range of a page the block needs, page stride = aligned page size), direct when every segment satisfies `(c_in − c_out)·es ≡ 0 (mod 16)`, assemble otherwise | `layout == ROW_MAJOR` (+ the mod-16 test above for direct) | `block_granules` output pages | minimum bytes; source runs are sub-ranges of input pages rounded down to the 64 B parity | as for the TILE regimes |
| `rm_bank_run` (narrow RM output pages, e.g. int32 `[1,W]→[W,1]`, the flagged perf case) | deferred — ROW_MAJOR refinement; reachable without new structure: walk output pages in **bank-major** order and lay the assembly out at the output's aligned page stride, so the existing write-merge rule (contiguous in DRAM and in assembly) emits one write per same-bank page run | `layout == ROW_MAJOR and output aligned page size > W_out·es` | `block_granules` pages of one bank | writes the alignment gaps between pages too (64 B per 4 B page) — more bytes, `num_banks`-fold fewer transactions: `[1,4096]→[4096,1]` on 110 cores ≈ 1 read + 1 write per core instead of ≈ 38 + 38 | as above |
| `input_keyed_scatter` (cores own input tiles, write each line to its output position) | rejected — superseded by `face_row_direct` with column-major granules, which already reads dense input faces in one run (`(1,1,16,4608)→(16,1,4608)`: 492 read runs for 4608 lines); input ownership also breaks the single-writer word cap (`(16,1,5120)→(1,1,16,5120)`: output rows 2k and 2k+1 come from different input tiles) | — | — | same bytes | — |
| `gather_into_place` (read each input line straight into its slot of an output tile in L1, write whole tiles) | rejected — illegal on Blackhole: DRAM reads need src ≡ dst (mod 64), and a bf16 line whose input and output in-face rows differ in parity (half the lines of every production shape that changes the row grouping) cannot land in place; superseded by `face_row_direct`, which reads at source parity and moves the parity change to the write (mod 16) | — | — | — | — |
| `whole_tile_rw` (read every touched input tile whole, write every output tile whole) | rejected (dead end) — up to 32× the minimum bytes on 1-valid-row tiles (`(16,1,5120)` input: 5 MB read for 160 KB of values; `(16,1,4608)` output: 4.7 MB written for 147 KB); superseded by line-granular runs | — | — | ≤ 32× reads and writes | — |
| `staged_multi_pass` (untilize → reshape → tilize, or any pass through a further DRAM tensor or the host) | rejected — forbidden by the spec and BB-01; doubles DRAM crossings | — | — | 2× | — |
| `compute_engine_transpose` (row→column moves through unpack/DEST/pack) | rejected — values would pass through unpacker/DEST formats (int32 cannot, fp32 risks NaN/denormal canonicalization); the RISC-V/NoC byte moves above are exact | — | — | — | — |
| dependent-axis split + combine | not applicable — the op has no dependent axis (a bijective copy; no result spans an axis), so there is no combine to build | — | — | — | — |

**Regime selection function (host, exact):**

```
P_direct   = layout == TILE and (W_in == W_out or (W_in % 16 == 0 and W_out % 16 == 0))
regime     = "face_row_direct" if P_direct else "face_row_assemble"      # Phase 0 (TILE only)
```

Regime-pinned tests (acceptance test): every bf16 production shape pins `face_row_direct`;
`(1,185)→(1,1,185,1)` and `(1,512)→(1,1,512,1)` in bf16 pin `face_row_assemble`. The predicate depends
only on shapes, never on the grid, so a regime cannot change between devices.

### Traffic ranking

There is no dependent axis, so every candidate split is a split of independent output axes; the ranking
is over **bytes per tier and transaction count**, because a reshape moves each value exactly once and the
DRAM bytes of the good candidates are equal — what differs is how many transactions carry them.

| Rank | Candidate split | DRAM bytes (read / write) | Read transactions | Cross-core | Verdict |
|------|-----------------|---------------------------|-------------------|------------|---------|
| 1 | output-keyed granules, **column-major** `u = (tc·B_out + b)·S + s` | N·es / N·es (minimum at line granularity) | one per merged run; consecutive output rows of one tile column read consecutive rows of one input face (`(1,1,16,4608)→(16,1,4608)`: 492 runs for 4608 lines) | none | **chosen** |
| 2 | output-keyed granules, row-major (`b`, `h` outer, `tc` inner) | same | one per line where the input is dense (`(1,1,16,4608)→(16,1,4608)`: 4608) | none | rejected — 9× the read transactions on the dense-input shapes, same bytes |
| 3 | input-keyed tiles (`input_keyed_scatter`) | same | one per input face run | none | rejected — single-writer word cap |
| 4 | whole tiles (`whole_tile_rw`) | ≤ 32× | one per tile | none | rejected — bytes |
| 5 | DRAM-staged passes | 2× crossings | — | — | forbidden |

**Operand-reuse check.** The only operand is the input. Along the split axis (granules) it varies: each
granule reads disjoint input positions (bijection), so no operand is reuse-shared and there is no
broadcast regime. The one sharing that exists is transaction-level, not byte-level: an input face run
split by a core boundary costs one extra read transaction, no extra bytes (direct regime); in the
assemble regime an input line straddling a core boundary is read by both cores (bounded above).

**Occupancy (separate from the ranking).** Spread first: `split_work_to_cores` over the full grid. Then the
coarsest block: the whole assignment. Cores used = `min(110, total_granules)`; on the golden INPUTS this is
110 except the tiny shapes (8–48 cores, each doing ≤ 4 lines), where the granule count *is* the blocking
decision (minimum single-writer granule).

### Block schedule

```cpp
// per core: granules [granule_begin, granule_begin + granule_count), num_blocks_this_core = ceil(granule_count / block_granules)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    stage_source_lines(block_idx);     // reader (NCRISC, NoC0)
    plan_block_writes(block_idx);      // writer (BRISC) — pure index math, runs BEFORE the wait for staged data
    assemble_output_lines(block_idx);  // writer, face_row_assemble only — after the wait
    store_output_lines(block_idx);     // writer (BRISC, NoC1)
}
// writer: one noc_async_write_barrier() before returning (BB-03)
```

Walk order inside a block (shared by both kernels, one header): `for fc in {0,1}: for u in block: for h in
granule rows: output line (r_out = b·H_out + h, g_out = 2·tc + fc)`, skipping `fc` where `32·tc + 16·fc ≥ W_out`.
`fc` outer is what lets the column-major granules merge input face rows into one run.

| Operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------|------------------------|--------------------|-----------------------------------|
| `stage_source_lines` | all source lines of the block's output lines, in walk order | the staging page (reserved at start, pushed at end) | per block: 1 `cb_reserve_back`, one `noc_async_read` per merged run (a run grows while the next source line is the next bytes of the same input page; assemble regime skips a source line equal to the previous one), 1 `noc_async_read_barrier`, 1 `cb_push_back`. Each run lands at `align_up(cursor, 64) + (S mod 64)`, cursor = run end. No inits. |
| `plan_block_writes` | the block's output lines | write plan in `cb_writer_scratch` | per block: one replay of the walk and of the reader's run-placement arithmetic (identical merge/dedupe rules → identical staging addresses; staging page address from `get_read_ptr`, valid before the wait). Produces write runs `{src L1 addr, dst NoC addr, bytes}` (+ copy segments `{src, dst, bytes}` in the assemble regime). Runs **before** `cb_wait_front` — see stall-shadow. |
| `assemble_output_lines` | the block's output lines, each built at assembly slot `k` (k = line index in the block) from its element segments | staged page (waited, not popped); assembly region | per block: 1 `invalidate_l1_cache()` after `cb_wait_front`, then RISC-V copies of each segment (32-bit words when source, destination and length are 4 B aligned, else element-sized). Segments: from `pos = L0` while `pos < L0 + n_valid`: `r_in = pos / W_in`, `c_in = pos % W_in`, `len = min(16 − c_in%16, W_in − c_in, L0 + n_valid − pos)`, copy `len` elements from source line `(r_in, c_in/16)` element `c_in%16` to slot element `pos − L0`. |
| `store_output_lines` | the block's write runs | staged page until flushed | per block: 1 `cb_wait_front` (direct regime: here; assemble regime: before assembly), one `noc_async_write` per write run (direct: from the staging address of the source line; assemble: from the assembly slot; a run grows while the next output line is the next bytes of the same output page **and** the next bytes of the source buffer), 1 `noc_async_writes_flushed` (staging bytes have left L1 before the page is recycled), 1 `cb_pop_front`. Kernel end: 1 `noc_async_write_barrier`. |

**Stall-shadow.** Two stages wait: the reader on its read barrier (nothing independent of the outstanding
reads is left to it — its next work is the next block's reads, which the overlap lamp covers) and the
writer on the staged page. The writer's whole index computation — the walk, the reader's run placement,
every output NoC address — depends only on shapes, never on the staged bytes, so it is legal to compute it
before `cb_wait_front` returns; it is scheduled there (`plan_block_writes`). After the wait the writer only
copies (assemble regime) and issues writes. No floating-point order exists, so no precision gate applies.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | one block per core means the writer issues nothing until all of the core's reads have landed (fill/drain of a 48-read + 48-write core on `(16,1,5120)→(1,1,16,5120)`) | `block_granules = ceil(granule_count / k)`, k = 2–4, `staging_depth = 2`; with trid-tagged read batches (`noc_async_read_set_trid`, `dataflow_api.h:2558`; `noc_async_read_barrier_with_trid`, `:2602`) so the reader keeps reads in flight (catalog `trid_double_issue`, `double_buffer`) |
| Grid synchronization | tiny shapes (`(8,1,64)→(1,1,8,64)`, `(1,8,64)→(8,1,64)`, `(1,1,6,256)→(1,1,1536)`) give each core 1–4 lines; per-core RT-arg dispatch may outweigh the work | cap cores at `ceil(total_granules / 4)` (coarser `R_g`) on shapes with < 4 lines per core |
| Write shape (output-dense direct shapes) | on `(16,1,5120)→(1,1,16,5120)`, `(1,16,6,256)→(1,1,16,1536)`, `(16,1,1536)→(1,1,16,1536)` every output tile has 16 valid rows, so `face_row_direct` issues 32 B writes where one 1 KB write per tile would do (catalog `tile_reorder`: whole-page writes beat face writes on DRAM-bound moves) | route these shapes through the built `face_row_assemble` path (selector flip: RISC-V copies of 32 B lines into tile-image slots, then one write per face run) |
| Read shape (1-row input tiles) | `(16,1,5120)` input tiles hold one valid row: its two face rows sit 512 B apart and cost two 32 B reads | one spanning 544 B read per input row segment (one transaction, 17× the bytes) |
| RISC role split | reader issues all reads and writer all writes; on the 96-transaction core the two halves are serialized by the read barrier (catalog `split_reader`) | two lanes: NCRISC and BRISC each stage+store half of the core's granules (each its own staging region, reads and writes on its own NoC) |
| Bank stagger | cores' first reads may land on the same DRAM bank (catalog `bank_stagger`, 1.04–1.3× on BH) | rotate each core's walk start by `core_index % num_banks` within its block |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM input | TILE pages (faces of 16×16, row-major in a face) | interleaved, `TensorAccessor` (input `TensorAccessorArgs` on **both** kernels: the writer needs each run's `S mod 64` to replay placement) | input tile padding is never a source: runs contain only needed lines; the assemble regime copies only valid elements |
| DRAM → L1 | raw bytes, source lines at their source parity | reader on NCRISC, NoC0, `noc_async_read` of merged runs into `cb_staged_lines` | one read barrier per block |
| reader → writer | one staging page per block | `cb_staged_lines` push / wait, 1 page per block (nominal count even for a short last block) | |
| L1 → L1 (assemble only) | output lines at slot stride `line_bytes` | RISC-V loads/stores on BRISC into the assembly region of `cb_writer_scratch` | `invalidate_l1_cache()` once after the wait (`cache.h:12`) — write-through cache, stores need no flush |
| L1 → DRAM | output lines (direct: from staging; assemble: from slots) | writer on BRISC, NoC1, `noc_async_write` of merged runs | `noc_async_writes_flushed` before pop, `noc_async_write_barrier` at kernel end |
| DRAM output | TILE pages | interleaved `TensorAccessor` | output tile padding is never written (allowed to hold anything) |

No Tensix-to-Tensix traffic exists in any built or deferred regime (no dependent axis, no reuse-shared
operand). TARGET has no `memory_layout` axis, so no physical shard is realized; sharded inputs are refused
by the host check.

**TARGET axes as regime rows.**

| TARGET value | Lands in | Cost class |
|--------------|----------|------------|
| `dtype=bfloat16, layout=TILE` | `face_row_direct` / `face_row_assemble` | built |
| `dtype=float32, layout=TILE` | same regimes with `es = 4`: `line_bytes = 64`, `rows_per_word = 1`; FLUX shapes are `face_row_assemble` with `R_g = 16` (1 read + 1 write per core) | knob-turn (element size is a CT arg; nothing else changes) |
| `dtype=int32 / uint32, layout=ROW_MAJOR` | `rm_line_direct` / `rm_line_assemble`, `rm_bank_run` for narrow output pages | regime addition on the same loop nest and topology: new line map (page segments), bank-major walk order; no cross-core communication |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | granule: `R_g` consecutive rows (slab `s`) of one output image `b` in one output tile column `tc`; `u = (tc·B_out + b)·S + s`, `S = ceil(H_out / R_g)`; rows `[s·R_g, min((s+1)·R_g, H_out))`. Slabs never cross a tile row (`32 % R_g == 0`); only an image's last slab can be short; no empty granules. |
| Grid | `device.compute_with_storage_grid_size()` (110 cores here) — runtime parameter |
| Per-core work | `ttnn.split_work_to_cores(grid, total_granules, row_wise=True)` → `core_group_1` cores take `n1` consecutive granules, `core_group_2` cores `n2`; cores enumerated row-wise (`ttnn.corerange_to_cores(group, row_wise=True)`), group 1 first; RT args per core: input address, output address, `granule_begin`, `granule_count` |
| Remainder | handled by `split_work_to_cores` (n1 = n2 + 1); ragged last block keeps the nominal 1-page push/pop |
| Blocks | `num_blocks_this_core = ceil(granule_count / block_granules)`; 1 on every golden shape |
| Core ranges | `all_cores` from `split_work_to_cores` — 1–3 rectangles (BB-17); one reader + one writer kernel descriptor over `all_cores`, no compute kernel |

**Host-side program cache (BB-07, BB-16).** Everything that changes the program is a CT arg or a core
range: regime, `es`, `(B,H,W)` of input and output, `R_g`, `block_granules`, `staging_depth`, CB sizes,
accessor args — so the generic-op structural hash separates every shape pair (the LOOSE_CASES sequence
`(16,1,64)` → `[1,1,16,64]` then `[1,1,1,1024]`). The Python entry keeps a descriptor cache keyed on
`(input logical shape, resolved output shape, dtype, layout, input memory_config, device id)`; on a hit it
only rewrites the two buffer-address RT args per core and dispatches — no descriptor rebuild. Output
allocation is the only per-call device allocation; nothing is built after the first call of a shape.

**Plan report (BB-08, BB-21).** The op module exposes `describe_plan(input_tensor, shape) -> dict` with
`regime`, `rows_per_granule`, `total_granules`, `num_cores`, `block_granules`, `staging_depth`, per-CB bytes,
and the per-core role map (`{core: "stage(NCRISC)+store(BRISC)"}` for active cores). Kernels wrap every
phase and every wait in `DeviceZoneScopedN` zones (`kernel_profiler.hpp:1012`; compiles to nothing in the
plain build): reader `G6R_STAGE`, `G6R_RESERVE_WAIT`, `G6R_READ_WAIT`; writer `G6R_PLAN`, `G6R_STAGED_WAIT`,
`G6R_ASSEMBLE`, `G6R_STORE`, `G6R_FLUSH_WAIT`, `G6R_FINAL_BARRIER`.

## Circular Buffers

Full ledger (capacity vs live set, axis accounting, sharing, symbol table, footprint): `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_staged_lines` | 0 | `staging_page_bytes = 64 + block_granules · lines_per_granule · max_src_lines · 64` | `staging_depth` | spans granule (`block_granules`), rows in granule, face col, source lines per output line; 64 B per source line covers the line (32 or 64 B) plus its parity pad; +64 for the page base alignment | raw bytes (input dtype tag, never unpacked) | reader | writer | one block: reserve → reads → push → wait → writes flushed → pop |
| `cb_writer_scratch` | 1 | `writer_scratch_bytes = block_granules · lines_per_granule · (16 + A·(line_bytes + 12·max_src_lines))`, `A = 1` in `face_row_assemble`, else 0 | 1 | write plan (16 B per write run ≤ one per output line); assemble regime adds copy segments (12 B each) and assembly slots (`line_bytes` each) | raw bytes | writer | writer (private scratch: `get_write_ptr` only, no reserve/push/wait/pop) | one block (plan → assemble → store) |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `stage_source_lines` | source lines of `block_granules × min(R_g,H_out) × fc × 16` | no (raw NoC; see API Mapping) | DRAM input via accessor | `cb_staged_lines`, 1 page reserved | page pushed after one read barrier |
| 2 | `plan_block_writes` | the block's output lines | no (index math) | — (address of the next staged page from `get_read_ptr`) | `cb_writer_scratch` plan region | plan resident until `store_output_lines` ends |
| 3 | `assemble_output_lines` (assemble regime) | the block's output lines × segments | no (raw RISC-V copy) | `cb_staged_lines`, 1 page, waited | `cb_writer_scratch` assembly region | staged page still held (popped in #4) |
| 4 | `store_output_lines` | the block's write runs | no (raw NoC) | `cb_staged_lines` page (direct) or assembly region | DRAM output via accessor | staged page popped after `noc_async_writes_flushed` |

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. Paths: `DA` =
`tt_metal/hw/inc/api/dataflow/dataflow_api.h`, `TA` = `tt_metal/hw/inc/api/tensor/tensor_accessor.h`.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `stage_source_lines` | raw_api | `cb_reserve_back` | `DA:406` | `(cb_staged_lines, 1)` | — | `cb_staged_lines` | — |
| `stage_source_lines` | raw_api | `get_write_ptr` | `DA:325` | `(cb_staged_lines)` | — | `cb_staged_lines` | — |
| `stage_source_lines` | raw_api | `TensorAccessor::get_noc_addr` | `TA:165` | `(page_id, in_page_offset)`; `TensorAccessorArgs` at `tensor_accessor_args.h:21` | — | — | — |
| `stage_source_lines` | raw_api | `noc_async_read` | `DA:547` | `(src_noc_addr, l1_addr, run_bytes)` | DRAM | `cb_staged_lines` | run length follows `block_granules`·`R_g` merges |
| `stage_source_lines` | raw_api | `noc_async_read_barrier` | `DA:1921` | — | — | — | once per block |
| `stage_source_lines` | raw_api | `cb_push_back` | `DA:208` | `(cb_staged_lines, 1)` | — | `cb_staged_lines` | — |
| `plan_block_writes` | raw_api | `get_read_ptr`, `get_write_ptr` | `DA:346`, `DA:325` | `(cb_staged_lines)`, `(cb_writer_scratch)` | — | `cb_writer_scratch` | plan size ∝ `block_granules` |
| `assemble_output_lines` | raw_api | `cb_wait_front` | `DA:479` | `(cb_staged_lines, 1)` | `cb_staged_lines` | — | — |
| `assemble_output_lines` | raw_api | `invalidate_l1_cache` | `tt_metal/hw/inc/internal/tt-1xx/cache.h:12` | — | `cb_staged_lines` | — | once per block |
| `assemble_output_lines` | raw_api | RISC-V loads/stores (`volatile tt_l1_ptr uint32_t*` / element-sized) | — | segment `{src, dst, len}` | `cb_staged_lines` | `cb_writer_scratch` | — |
| `store_output_lines` | raw_api | `cb_wait_front` (direct regime) | `DA:479` | `(cb_staged_lines, 1)` | `cb_staged_lines` | — | — |
| `store_output_lines` | raw_api | `noc_async_write` (or `noc_async_write_one_packet_set_state` / `_with_state` for runs ≤ `NOC_MAX_BURST_SIZE`) | `DA:876` (`DA:982`, `DA:1008`) | `(src_l1, dst_noc_addr, run_bytes)` | `cb_staged_lines` / `cb_writer_scratch` | DRAM | — |
| `store_output_lines` | raw_api | `noc_async_writes_flushed` | `DA:1980` | — | — | — | once per block |
| `store_output_lines` | raw_api | `cb_pop_front` | `DA:260` | `(cb_staged_lines, 1)` | `cb_staged_lines` | — | — |
| `store_output_lines` | raw_api | `noc_async_write_barrier` | `DA:1951` | — | — | — | once per kernel |
| all | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | zone names above | — | — | — |

**Helpers considered and rejected** (the op has no compute phase; every candidate is a dataflow helper):

| Raw use | Candidate helper | File:Line of the mismatch | Why it cannot express this block shape |
|---------|------------------|---------------------------|----------------------------------------|
| DRAM line-run reads | `dataflow_kernel_lib::set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:75`, `:95` (source built by `local_addr`, `:57`) | self-aimed L1→L1 copies only: the source is this core's own L1 (`local_addr` returns `my_x/my_y`), and the size is a template constant; our source is a DRAM page at a per-run offset with per-run length |
| DRAM line-run reads | `read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-88` | reads whole RM sticks into a CB for a tilize compute phase; our input is TILE and must not be re-formatted (no compute, bit-exact) |
| Element assembly | `dataflow_kernel_lib::copy_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:91-92` | copies one whole 32-row column between two tiles of the same layout; our segments are 1–16-element runs between arbitrary line slots, and it does not invalidate the L1 cache before loading NoC-written data |
| Element assembly | NoC self-copy (`read_with_state`) | `local_copy_helpers_dataflow.hpp:95`; congruence rule `noc_parameters.h:375-376` | an L1→L1 NoC copy needs src ≡ dst (mod 16); element segments violate it by construction (that is what puts a shape in `face_row_assemble`) |
| Output line-run writes | `write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:129-130` | writes RM sticks produced by an untilize phase; our output is TILE lines from staging |
| Any phase | compute helpers (`tilize`, `untilize`, `eltwise_chain` copy, `reduce`, `matmul_block`) | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp:64-71`, `eltwise_chain.hpp` | every compute helper passes values through unpack → DEST → pack; the op must be bit-exact for every dtype (int32/uint32 and fp32 special values) and moves bytes only — there is no compute phase to map |

`compute_kernel_hw_startup()` does not apply: no compute kernel exists.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Reading a line straight into its output slot | BH DRAM reads need src ≡ dst (mod 64); a bf16 line at an odd in-face row has `S ≡ 32`. Without the watcher this corrupts silently, and only half the rows | staging placement at `align_up(cursor, 64) + (S mod 64)`; the parity change happens on the write (mod 16) |
| Reader/writer placement divergence | the writer recomputes every staging address; one different merge or dedupe decision shifts every later line of the block | one shared walk/placer header included by both kernels; identical walk order, merge rule (same input page, next byte), dedupe rule (assemble: skip a source line equal to the previous one), and `S` from the same input accessor |
| Input padding reaching a logical position | the golden suite poisons input padding with NaN (`ttnn.fill_implicit_tile_padding`) | runs merge only exactly-contiguous needed lines; the assemble regime copies only valid elements; a direct-regime tail line (`W_in == W_out`, `W % 16 != 0`) carries input padding columns only into output padding columns |
| Stale L1 cache in the assemble regime | `cb_wait_front` does not invalidate (`DA:479-492`); the writer's earlier loads of the same staging addresses (previous block) can hit | `invalidate_l1_cache()` after the wait, before the first load |
| Staging recycled while writes still read it | direct writes source the staging page; with depth 2 the reader may reserve it again after the pop | `noc_async_writes_flushed()` before `cb_pop_front` every block |
| In-flight writes at kernel exit (BB-03) | the next program starts immediately | writer ends with `noc_async_write_barrier()`; reader ends each block with a read barrier |
| Two writers on one 64 B DRAM word | bf16 rows 2k and 2k+1 of a face share a word | `R_g` multiple of `rows_per_word`, slabs aligned to `R_g` |
| Losing read coalescing | `(1,1,16,4608)→(16,1,4608)` and `(1,1,8,4608)→(8,1,4608)` coalesce only with column-major granules and `fc`-outer walk; row-major is still correct, just 9× the reads | granule order and walk order fixed here; a reorder is a measured change |
| Rank-1 and rank-3 views | `(W,)` is one row (`H = 1`, `B = 1`); `(16,1,4608)` is 16 images of one row | index map above derives `(B, H, W)` per tensor from the logical shape, tile counts with `ceil`, per image |
| `-1` and validation order | `[1,1,-1,1536]`, `[-1,1]` are golden cases; a `-1` must resolve exactly as torch | `validate()` first (registry), then shape parsing/resolution with `ValueError` |
| Element regime untested by production bf16 shapes | no bf16 golden INPUT reaches `face_row_assemble` | acceptance test pins it with `(1,185)→(1,1,185,1)` and `(1,512)→(1,1,512,1)` in bf16, plain and NaN-poisoned |
| Program cache key | generic-op hash covers CT args and core ranges, not RT values | every shape-dependent value is CT or a core range; the host descriptor cache key includes the resolved output shape |
| Trace capture (BB-16) | the traced LOOSE_CASE captures a call after an eager call of the same shape | the eager call builds and caches the program; the captured call is a cache hit that only patches addresses and allocates the output |
| Kernel image size (BB-04, BB-18) | both kernels include the walk/placer header | regime is a CT arg so only one path compiles; per-line work is a loop over the walk, never unrolled per shape; no size-optimising build flag |

### Structural impossibilities

None to add: `INVALID = []` is correct — every (dtype, layout) pair in TARGET is a legal tensor and every
shape with a matching element count is a legal reshape.
