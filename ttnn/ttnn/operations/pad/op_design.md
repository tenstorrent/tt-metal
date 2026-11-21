# Operation Design: pad

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel) |
| Goal | Grow a row-major interleaved tensor by `(before, after)` elements per dimension, every added element = `value` in the input's dtype; one `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec`. |
| Math | `out[i0+b0, i1+b1, i2+b2, i3+b3] = in[i0, i1, i2, i3]`; every other `out` element `= cast(value, dtype)`; `out.shape[d] = b_d + in.shape[d] + a_d` |
| References | `eval/specs/pad/spec.py` (governs); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (`toy_thread_plan`: host `ProgramSpec`, implicit / explicit / mixed ring protocols); `ttnn/ttnn/operations/toy_spec_single_tensor/` (a DM-only `ProgramSpec`: DM producer → DM consumer ring, no compute kernel); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4); predicates query `device.compute_with_storage_grid_size()` | target page + probe |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 − 0 = 3,963,328 B on quasar/ttsim (probe, 2026-11-21) | probe |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`); **NoC transfer alignment 1 B** on Quasar — src/dst need not be congruent (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:482-491`) | target page + header |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec`, `KernelSpec`, `DataflowBufferSpec`, `WorkUnitSpec`, `ProgramRunArgs`) dispatched by one `ttnn.generic_op` — reference `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}), 6 user DM harts (R + W ≤ 6); STRIDED ring: the larger endpoint thread count divisible by the smaller (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:1707-1726`), `num_entries % max(P, C) == 0` (`:1523`) | target page + source |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program-factory op (refused at kernel creation, target page); `bfloat8_b`/`bfloat4_b` (no path on quasar; not in TARGET anyway); DM self-loop of a DFB (one DM kernel as producer **and** consumer) is refused on Gen2 (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:637-653`) — so no DM kernel owns a private scratch ring; `MEM_ZEROS` region (WH/BH only). Library-gaps entries (`.claude/references/quasar-library-gaps.md`: `ReduceFp32Mode::Accurate`, `AccumulateViaAdd`, MAX+REDUCE_ROW Accumulate, SFPU post-op in reduce) concern compute helpers this op does not use; DPRINT from compute is moot (no compute kernel). | target page, library-gaps reference, source |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | ROW_MAJOR, interleaved, DRAM (Phase 0) / DRAM or L1 (TARGET), bfloat16 (Phase 0) / float32 (TARGET), rank 4 (Phase 0) / 2–4 (TARGET), every dim ≥ 1 | — | host |
| `padding` | any sequence of `(before, after)` int pairs, first dimension first (tuple of tuples, list of tuples) | yes | exactly `rank` pairs; every amount ≥ 0; Phase 0: every `before == 0` | — | host → RT (shape/offset args) |
| `value` | float | yes | Phase 0: 0.0; TARGET: 0.0, 1.0 | — (no default, as tt-metal) | host → RT `pad_bits` (element bit pattern) |

Host canonicalization (entry point, before `validate()`):

| Step | Rule |
|------|------|
| Padding parse (R3, X1–X4) | `padding` must be a sequence whose every item is a length-2 sequence of ints → else `TypeError` (X4: torch's flat form has int items). `len(padding) != rank` → `ValueError` (X1, X2). Any amount `< 0` → `ValueError` (X3). Normalize to `tuple((int(b), int(a)) for b, a in padding)`. |
| Axes | `memory_config`, `sharding` from `eval.memory.memory_axes(input_tensor)` (R6); `padding_side`, `alignment`, `rank` from the spec's `AXES` called on `(tuple(input_tensor.shape), padding)` (`INPUT_TAGGERS = AXES`); `dtype`, `layout` from the tensor; `pad_value` = `value`. `validate()` gates on them (R7, `NotImplementedError`). TILE layout is refused as an unsupported `layout`. There is no index axis to canonicalize. |
| Rank fold | Shape and pairs are folded to rank 4 by prepending `1`s and `(0, 0)` pairs: `(N, C, H, W)` and `(bn,an),(bc,ac),(bh,ah),(bw,aw)`. Kernels only ever see rank 4 (rank 2/3 is a host-only unlock). |
| Value | `pad_bits`: bfloat16 → upper 16 bits of `torch.tensor(value, dtype=torch.float32).to(torch.bfloat16)` (round-to-nearest-even, = the input dtype's conversion); float32 → the IEEE-754 32-bit pattern. Exact for 0.0 / 1.0. |
| Mechanism refusal | `entry_bytes > DFB_RING_MAX_BYTES` or `2 · entry_bytes > budget` → `NotImplementedError` (row too wide for one ring entry; see Mechanism caps / regime `row_chunked`). Not reachable by any INPUTS entry. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(N, C, H, W)` in Phase 0 (rank 2/3 folded to 4 later) |
| Dtype | bfloat16 (Phase 0); float32 (TARGET) |
| Layout | ROW_MAJOR (a page = one row of `W` elements, `in_row_bytes = W · elem_bytes`, stride in bank = `input.buffer_aligned_page_size()`) |
| Memory | interleaved; DRAM (Phase 0), L1 (TARGET) |

### Output

| Property | Value |
|----------|-------|
| Shape | `(bn+N+an, bc+C+ac, bh+H+ah, bw+W+aw)` = `(N_o, C_o, H_o, W_o)` |
| Dtype | same as input |
| Layout | ROW_MAJOR (page = one output row, `out_row_bytes = W_o · elem_bytes`) |
| Memory | interleaved, in the input's buffer (R4): `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(ttnn.Shape(out_shape), dtype, ttnn.ROW_MAJOR_LAYOUT, buffer_type=input_tensor.memory_config().buffer_type), device)` (`ttnn/cpp/ttnn-nanobind/tensor.cpp:306-325`) |

## Blocking Model

The op is a pure re-layout of rows. Its index space is output rows × elements of a row.
`O = N_o·C_o·H_o` output rows split into **data rows** (`D = N·C·H`, each the image of exactly one input row) and **pad rows** (`P = O − D`, every element = value). A data row = `[bw × value][input row][aw × value]`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| data rows (input rows `i ∈ [0, D)`, enumerated N·C·H row-major = input page order) | independent — each output data row depends on one input row only | `block_rows` (rows per reader block / ring reserve) | `max_core_data_rows` (whole per-core assignment) when it fits one ring (`≤ rows_fit`), else `rows_fit // buffer_depth` | host constant computed once in the program-artifacts function (`_choose_blocking`), passed as RT arg to reader; ring capacity derived from it | `ttnn.split_work_to_cores(grid, D, row_wise=True)` → contiguous `[data_start, data_start+data_count)` per core, over the full grid | knob-turn |
| pad rows (`q ∈ [0, P)`, enumerated in output-row order) | independent — a pad row depends on nothing but `value` | `pad_rows_per_barrier` (pad writes issued before one write barrier) | the core's whole pad assignment (`pad_count`): one barrier per core | same host function; RT arg to writer | `ttnn.split_work_to_cores(grid, P, row_wise=True)` → contiguous `[pad_start, pad_start+pad_count)` of the pad enumeration per core, over the full grid, independent of the data split | knob-turn |
| row elements (W / W_o) | independent — no element depends on another | `row_extent` = whole row (`W_o`); a ring entry is one whole output row | whole row; not split | `entry_bytes = align_up(out_row_bytes, ttnn.get_l1_alignment())`, host | not assigned across cores: a row is one page = one NoC transaction; splitting it multiplies transactions for the same bytes (Traffic ranking C) | scheme-change (`row_chunked`, deferred) |
| batch / channel (N, C) and H as separate axes | independent; folded into the two row enumerations (they are contiguous and treated identically by the op) | covered by the row-enumeration knobs | — | — | via the row splits | — |

No **dependent** axis exists (nothing is reduced, scanned or combined). No **reuse-shared** operand exists: the only tensor operand is the input, and it varies along the split axis (each core reads distinct rows). `value` is a scalar RT arg; its row-sized form (the pad-row template) is built in each core's own L1 by stores, never read from DRAM — so nothing needs a broadcast.

Per-stage re-run of the axes table (the scheme introduces two derived data sets):

| Stage data | Axis | Character | Block | Core assignment | Why |
|------------|------|-----------|-------|-----------------|-----|
| assembled data-row entries (ring `dfb_out_rows`) | data rows | independent | `block_rows` | same cores that read them (reader and writer of one core share the ring) | an entry is consumed on the core that assembled it; no cross-core hand-off |
| pad-row template (ring `dfb_pad_row`) | row elements | independent | one whole row, built once per core | every engaged core with `pad_count > 0` builds its own | building it is `W_o` L1 stores per core; distributing one copy over the NoC would cost more than building it, and no core idles while another builds |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `dfb_out_rows` | `buffer_depth` (blocks in flight) → `ring_blocks = 1 if num_blocks_max == 1 else buffer_depth` | `buffer_depth = 2`; `ring_blocks = 1` on every INPUTS entry (each core's assignment fits one block) | with >1 block per core: reader fills block k+1 while writer drains block k |
| `dfb_pad_row` | none (constant: one template entry per writer thread) | `max(R, W) = 1` entry | — (read-only after build) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Implicit-sync txn descriptor (writer side of `dfb_out_rows`): `num_entries / num_txn_ids ≤ 255` (`dataflow_buffer.cpp:1351-1360`) | ring entries → `block_rows · ring_blocks · max(R,W)` | `RING_ENTRY_CAP = 255` total entries (holds for any txn-id count ≥ 1) | `TT_FATAL` at program build |
| Implicit-sync divisibility: `num_entries % (num_txn_ids · W · tcs) == 0` (`dataflow_buffer.cpp:1333-1340`); host picks the smallest workable txn count, falling back to 1 (`:1448-1473`) | ring entries | `block_rows` rounded down to a multiple of `max(R,W)`; with W = 1 always satisfiable | `TT_FATAL` |
| STRIDED ring: `num_entries % max(P,C) == 0` (`:1523`), larger thread count divisible by smaller (`:1707-1726`) | thread plan, ring entries | plan restricted to `max(R,W) % min(R,W) == 0`; entries a multiple of `max(R,W)` | `TT_FATAL` |
| DFB capacity 16-bit (`:1590-1596`); `reserve_back(uint16_t)` (`dataflow_buffer.h:243`) | ring entries | subsumed by 255 | `TT_FATAL` |
| Ring size ≤ 1,048,560 B (target page, "on the compute side"; applied conservatively to these DM rings) and L1 budget | `block_rows · ring_blocks · entry_bytes`; per row `entry_bytes` | `rows_fit = min(RING_ENTRY_CAP, ring_byte_cap // entry_bytes)` with `ring_byte_cap = min(DFB_RING_MAX_BYTES, budget − template_bytes)`; a row with `entry_bytes > ring_byte_cap` is refused (`NotImplementedError`) | allocation failure, or a ring larger than the hardware ring |
| Implicit write transfers the whole **entry** (`get_entry_size()`), not the page (`tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:832-875`) | `entry_bytes` vs output page stride | `entry_bytes = align_up(out_row_bytes, l1_alignment)`; host asserts `entry_bytes ≤ output.buffer_aligned_page_size()` (alignment 16 or 64 ≥ 16, so it always holds) | each write spills into the **next output page** → silent wrong rows |
| `fill_l1_range<val_size>`: start and length multiples of `val_size` (`ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-151`) | head/tail fill ranges | `val_size = elem_bytes`; ranges are element-granular by construction | head/tail bytes silently under-filled |
| Explicit tagged read flow control (`noc.h:259-262`: waits while fewer than half the trid's transactions are free) | reads in flight per barrier (`block_rows`) | none needed — the API throttles | — |
| NoC transfer alignment = 1 B on Quasar (`noc_parameters.h:482-491`) | data-row landing offset `bw · elem_bytes`; row byte counts | none needed: any byte offset/size is legal | — (on WH/BH this would be a cap) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `rows_resident` — row split (data list + pad list over the grid), each core's whole data assignment one block | **built** (Phase 0) | `max_core_data_rows ≤ rows_fit` (true for every INPUTS entry: the tallest has 64 rows/core at 512 B entries where `rows_fit = 255`; the widest 1 row/core where `rows_fit = 63` — see note) | data: `block_rows = max_core_data_rows` × whole row; pad: `pad_count` × whole row | **minimum**: boundary = DRAM. Input crosses once (exactly `D · in_row_bytes`), output once (`D · entry_bytes + P · out_row_bytes`; `entry_bytes − out_row_bytes < 16` B per data row lands in the page's own alignment slack). Pad values never cross DRAM inward; no cross-core traffic. | one reserve/push + one read barrier per core (data); one write barrier per core (pad); ring wrap never happens |
| `rows_streamed` — same split and kernels, double-buffered blocks | **built** (same kernels, knob values differ; regime-pinned test forces it via a `block_rows` override) | `max_core_data_rows > rows_fit` | `block_rows = rows_fit // buffer_depth`; last block per core ragged (`last_block_rows`) | minimum (as above) | per block: one `reserve_back`/`push_back`, one tagged read barrier (pipeline drain), one implicit-txn window ack; bigger block → fewer drains, until the 255-entry / 1 MiB ring cap |
| `before` padding — head fill `[0, bw)`, read lands at offset `bw·elem_bytes`, `before` offsets in the row maps | **deferred** (knob-turn): the kernels and host carry `bn, bc, bh, bw` as RT args from Phase 0 (value 0, gated by `validate()` on `padding_side`); unlock = SUPPORTED change + tests. Reachable because Quasar's NoC has no alignment restriction (byte-offset landing is legal) | `padding_side == "before"` | as `rows_*` | minimum | — |
| `none` padding (copy) | **deferred** (knob-turn): same program with `P = 0`, `W_o = W` — one dispatch, a copy (never return the input: one native dispatch per call is mandatory) | `padding_side == "none"` | as `rows_*` | minimum (copy) | — |
| float32 / `pad_value` 1.0 / L1 placement / rank 2–3 | **deferred** (knob-turn each): `elem_bytes` CT arg, `pad_bits` RT arg, accessor + output `buffer_type` from the input, host rank fold | the respective SUPPORTED axes | as `rows_*` | minimum; R1 holds because rows are moved byte-exact by the NoC (no format conversion anywhere) | — |
| `row_chunked` — a row wider than one ring entry, read/written in column chunks | **deferred**: positive reason — no TARGET × INPUTS cell reaches it (largest row: `W_o = 8192`, ≤ 32 KiB in float32, vs the 1,048,560 B ring cap); structure keeps it reachable because the column chunk is just a second extent on the entry (`row_extent`), and Quasar NoC byte-alignment allows chunk offsets anywhere | `entry_bytes > ring_byte_cap` | (rows × column chunk) | minimum | — |
| contiguous output-row split (each core owns `[r0, r1)` of output rows, both kinds) | **rejected**: same DRAM bytes as `rows_*`, but cores owning only pad rows finish early while data-heavy cores read+write (up to 2× bytes imbalance, e.g. N-pad of `(2,4,128,512)`: last third all pad); superseded by the two-list split, which balances both kinds over the full grid at no extra traffic | — | — | minimum | — |
| column (W) split across cores | **rejected**: same bytes, k× the transactions per row; rows already fill the grid on every INPUTS entry with ≥ 32 rows; superseded by the row split | — | — | minimum bytes, more transactions | — |
| pad rows through the data ring (reader fills whole entries with value) | **rejected** (dead end): `W_o` L1 stores per pad row (a pad row's slot previously held data); superseded by the template (zero per-pad-row L1 work) | — | — | minimum DRAM, large core-local cost | — |
| implicit-sync reader with an input-row-sized ring, writer writes data part + tail separately | **rejected**: two writes per data row into one page, unordered w.r.t. each other (the implicit write sends the whole entry, overrunning into the tail) | — | — | 2× write transactions | — |
| dependent-axis split + combine | **n/a**: no dependent axis exists | — | — | — | — |

Note on `rows_fit` (quasar, budget 3,963,328 B): `rows_fit = min(255, min(1,048,560, budget − entry_bytes) // entry_bytes)` → 255 rows for `entry_bytes ≤ 4,112`; 63 rows at 16,384 B (W_o = 8192 bf16); 31 rows at 32,768 B (W_o = 8192 fp32). The widest INPUTS entry has 1 data row/core, the tallest (`(1,1,2048,256)`) 64 rows/core at 512 B. So `rows_streamed` is reached only through the override in its pinned test (or by larger shapes outside INPUTS, e.g. YOLO's 640×640 case: `D = 1920` → 60 rows/core, still resident).

Regime selection function (host, exact):

```
T = max(R, W)                                       # Phase 0: 1
entry_bytes = align_up(W_o * elem_bytes, ttnn.get_l1_alignment())
template_bytes = T * entry_bytes
budget = ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes
ring_byte_cap = min(DFB_RING_MAX_BYTES, budget - template_bytes)
rows_fit = min(RING_ENTRY_CAP, ring_byte_cap // entry_bytes) // T        # per writer/reader thread
if max_core_data_rows_per_thread <= rows_fit: block_rows, ring_blocks = max(1, max_core_data_rows_per_thread), 1   # rows_resident
else:                                         block_rows, ring_blocks = rows_fit // buffer_depth, buffer_depth      # rows_streamed
num_entries(dfb_out_rows) = T * ring_blocks * block_rows
```

`block_rows_override` (probe kwarg of the program-artifacts function, not of the public op) replaces `block_rows` and forces `ring_blocks = buffer_depth` — used by the regime-pinned test.

### Traffic ranking

| Rank | Split | DRAM crossings | Cross-core | Transactions | Verdict |
|------|-------|----------------|------------|--------------|---------|
| 1 | data rows and pad rows as two lists, each over the full grid | input ×1, output ×1 | none | 1 read / input row, 1 write / output row (minimum) | **chosen** |
| 1 (tie on bytes) | contiguous output-row ranges | input ×1, output ×1 | none | minimum | rejected on balance (tiebreaker), not on traffic |
| 3 | W split across cores | input ×1, output ×1 | none | k reads + k writes per row | rejected |
| — | dependent split + combine | — | — | — | does not exist |

The minimum is reached: every input byte crosses DRAM once, every output byte once, pad values never come from DRAM, nothing crosses between cores.

Stall-shadow check (two waits exist, both filled):

| Stage that waits | Waits on | Work scheduled into the stall | Why it is legal |
|------------------|----------|-------------------------------|-----------------|
| writer, before the first data block is pushed | reader's DRAM reads of block 0 | **all of the core's pad rows** are written first (they depend only on the template, which the reader builds before issuing any data read) | pad rows and data rows are disjoint output pages: write order between them is free. No arithmetic, no FP-order question. |
| reader, at each block's read barrier | DRAM read latency of `block_rows` reads | first-lap head/tail fills of the same block's entries are done **after issuing the reads and before the barrier** | fills touch bytes `[0, bw·e)` and `[(bw+W)·e, W_o·e)`; the read lands in `[bw·e, (bw+W)·e)`: disjoint bytes of the entry |

### Block schedule

Reader (per core; `T = 1` in Phase 0):

```cpp
build_pad_row_template();                       // iff pad_count > 0: reserve 1, fill W_o × value, push 1
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_data_block(block_idx);                 // reserve block_rows; issue rows_this_block tagged reads (one per input row,
                                                //   in_row_bytes → entry j at offset bw*elem_bytes);
    fill_row_margins_first_lap(block_idx);      // iff block_idx < ring_blocks: fill head [0,bw) and tail [bw+W, W_o) of each entry
                                                //   (later laps reuse the slot's margins: reads never touch them, writer only reads)
    publish_data_block(block_idx);              // one tagged read barrier; push rows_this_block
}
```

Writer (per core):

```cpp
store_pad_rows();                               // iff pad_count > 0: wait template; walk output rows from pad_start_out_row,
                                                //   skip data rows, one tagged write per pad row from the template
                                                //   (pad_rows_per_barrier = pad_count → ONE barrier); pop template
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    store_data_block(block_idx);                // rows_this_block implicit writes, one per entry, to page out_row(i)
}
release_trid();                                 // noc_async_write_set_trid(0, noc.get_noc_id()) — last statement
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `build_pad_row_template` | 1 row × `W_o` | template, for the whole kernel | once per core (iff `pad_count > 0`); `W_o` element stores |
| `load_data_block` | `rows_this_block` rows × `W` (into entries of `W_o`) | the block's entries until published | one `reserve_back` per block; reads issued back-to-back, no barrier inside |
| `fill_row_margins_first_lap` | `rows_this_block` × (`bw` + `aw`) elements | margins stay valid in their slots for the kernel's lifetime | only while `block_idx < ring_blocks` (first lap of the ring): at most `ring_blocks · block_rows` entries per core ever filled |
| `publish_data_block` | the block | — | one tagged read barrier + one `push_back` per block |
| `store_pad_rows` | `pad_count` rows × `W_o` | template | one `wait_front`, one write barrier, one `pop_front` per core |
| `store_data_block` | `rows_this_block` rows × `W_o` | — | no per-row or per-block kernel sync (implicit: DM0 ISR acks per txn window) |

Row maps (both kernels; counters advanced per row, divisions only at the start of a core's range):

| Map | Formula |
|-----|---------|
| data row → input page | `i` (the reader reads page `i` of the input accessor, `in_row_bytes` from offset 0) |
| data row → output page | `(n, c, h) = unravel(i, (N, C, H))`; `out_row(i) = ((n+bn)·C_o + (c+bc))·H_o + (h+bh)` |
| output row is data? | `(n_o, c_o, h_o) = unravel(r, (N_o, C_o, H_o))`; data iff `bn ≤ n_o < bn+N ∧ bc ≤ c_o < bc+C ∧ bh ≤ h_o < bh+H` |
| q-th pad row → output row (host only: `pad_start_out_row` per core) | recursive count over dims N, C, H: at dim d the output range splits into a before segment (`b_d` sub-boxes, all pad), a data segment (`I_d` sub-boxes, each with `pad_in_sub(d+1)` pads), an after segment (`a_d` sub-boxes, all pad); locate q, subtract, descend — O(rank), no per-row host loop |

The writer walks forward from `pad_start_out_row` with mixed-radix counters, skipping data rows (whole data runs of `H` rows may be skipped at once — implementer's choice), until `pad_count` pad rows are written.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | `rows_resident` makes each core's data one block: the writer cannot start data writes until every read of the core landed (only pad writes fill that window) | `block_rows = ceil(max_core_data_rows / 2)`, `ring_blocks = 2` (same ring bytes), on `(1,1,2048,256)` and `(2,4,128,512)` |
| Grid synchronization | small shapes leave ~2 data + ~8 pad rows per core (YOLO stem: D = 60, P = 260 over 32 cores); launch + per-core fixed cost may dominate | cap engaged cores so each owns ≥ 8 rows (e.g. 8 cores for the stem) |
| Thread plan | rows are small (40 B – 16 KiB): per-transaction cost, not bytes, may limit one reader hart (explicit sync; README: explicit needed 2+2 harts to reach the floor with tile-sized transfers, one per barrier) | plans (1,0,2) and (2,0,2) (`PAD_THREADS` constant; kernels already deal by thread id) |
| Margin fill | CPU stores (`fill_l1_range`) for template/margins; for `value == 0` the iDMA zero path exists | `noc.async_write_zeros` + `write_zeros_l1_barrier` (`noc.h:816-849`) when `pad_bits == 0` — no other NoC write may be issued between the two (`noc.h:833-838`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input rows DRAM/L1 → `dfb_out_rows` entry | raw row bytes (bf16 / fp32), `in_row_bytes` | reader: `noc.async_read<NocOptions::TXN_ID>(acc_in, dfb_out_rows, in_row_bytes, {.page_id = i}, {.offset_bytes = j·entry_bytes + bw·elem_bytes}, {.trid = 0})`, one tagged barrier per block | explicit sync (reader assembles entries) |
| margins of an entry | element bit pattern `pad_bits` | reader: `dataflow_kernel_lib::fill_l1_range<elem_bytes>` on `dfb_out_rows.get_write_ptr()` + offsets, first lap only | bytes disjoint from the read |
| template | `W_o × pad_bits` | reader: `fill_l1_range<elem_bytes>` over one `dfb_pad_row` entry; `push_back(1)` | built before any data read |
| `dfb_out_rows` → output page | `entry_bytes` (≥ `out_row_bytes`, < +16 B, into the page's alignment slack) | writer: implicit `noc.async_write<NocOptions::TXN_ID>(dfb_out_rows, acc_out, {}, {.page_id = out_row(i)})` | one write per output data row |
| template → output pad page | `out_row_bytes` | writer: `noc.async_write<NocOptions::TXN_ID>(CoreLocalMem<uint32_t>(dfb_pad_row.get_read_ptr()), acc_out, out_row_bytes, {}, {.page_id = r}, {.trid = 0})`; one `async_write_barrier<TXN_ID>({.trid = 0})` after the last | explicit sync on `dfb_pad_row` |

Placement as TARGET axes (`memory_config`, `sharding`): DRAM → L1 interleaved is a **knob-turn** (the accessor addresses either; the output follows the input's `buffer_type`). Sharded placements are not in TARGET (`sharding: ["interleaved"]`); a height-sharded RM input would cut the row axes, which are independent, so it would be a knob-turn too (consume the shard in place through a DFB `borrowed_from` the tensor, no NoC re-read). No cross-core contract is needed by any regime.

Structural impossibilities: none to add (INVALID = [] is right for this rectangle).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a data row (read + write) in the data list; a pad row (write) in the pad list; reader blocks are `block_rows` data rows |
| Grid | `device.compute_with_storage_grid_size()` (8 × 4 on quasar); engaged cores = union of the two splits' core sets (both fill row-major from (0,0)); a core with a zero count in one list skips that list's operations (and the template iff `pad_count == 0`, gated identically in both kernels) |
| Per-core work | data: `split_work_to_cores(grid, D, row_wise=True)` → `(data_start, data_count)`; pad: `split_work_to_cores(grid, P, row_wise=True)` → `(pad_start, pad_count)`, host converts `pad_start` → `pad_start_out_row` |
| Remainder | `split_work_to_cores` groups (counts differ by ≤ 1); within a core `num_blocks_this_core = ceil(data_count / block_rows)`, `last_block_rows = data_count − (num_blocks_this_core − 1)·block_rows`. The ragged last block reserves `block_rows` and pushes `last_block_rows` — legal because it is the core's final push and every earlier push was a whole block (block starts stay aligned to the ring; no wrap inside a push); the implicit writer issues exactly `data_count` writes (firmware credits the partial txn window at kernel exit). Row geometry is per-image and needs no ceil: RM rows are pages of any width. |
| Worked (quasar, 32 cores) | YOLO stem `(1,3,20,20)`+C(0,13): `D=60` (28 cores × 2, 4 × 1), `P=260` (4 × 9, 28 × 8), `entry_bytes=48`, `block_rows=2`, ring 2 entries (96 B). `(2,4,128,512)`+N(0,1): `D=1024` (32/core), `P=512` (16/core), `entry_bytes=1024`, ring 32 entries (32 KiB). `(1,1,2048,256)`+H(0,48): `D=2048` (64/core), `P=48`, `entry_bytes=512`, ring 64 entries (32 KiB). `(1,1,32,4096)`+W(0,4096): `D=32` (1/core), `P=0`, `entry_bytes=16,384`, ring 1 entry. |
| Bound | **data movement**: zero arithmetic per row; per data row `in_row_bytes` read + `out_row_bytes` written, per pad row `out_row_bytes` written |
| Thread plan | **(R, C, W) = (1, 0, 1)**: one reader hart, **no compute kernel** (nothing to compute: the NEOs have no work), one writer hart. Host constant `PAD_THREADS = (1, 1)` (reader, writer threads); counts passed as CT args (`reader_threads`, `writer_threads`); kernels deal by `get_my_thread_id()` so the lamp plans are a one-constant change: reader thread p takes core-local data rows `e ≡ p (mod R)`, writer thread w takes data entries `e ≡ w (mod W)` and pad rows `q ≡ w (mod W)`; `dfb_pad_row` holds `max(R,W)` template copies. Rule for this program: `max(R,W) % min(R,W) == 0`, `R + W ≤ 6`. |
| Unit dealt to compute threads | n/a (no compute kernel). Unit dealt to DM threads = the ring entry (one row), so no round padding exists. |
| Tests the plan needs | (1) **coverage**: call the program-artifacts entry directly with `pad_bits` of 1.0 (validate bypassed; the kernels are value-general) on `(2,4,128,512)`+N and `(4,8,47,256)`+all-after, plans `(1,0,1)` and `(1,0,2)`, and compare exactly — with value 0.0 a missing pad write is invisible on zero-initialized simulator DRAM; (2) **regime pin**: `block_rows_override=3` on `(2,4,128,512)`+N (11 blocks/core, ragged last block of 2, ring wraps) and on the YOLO stem; (3) **session**: in one device session run another program (e.g. `ttnn.add` of two tile tensors) then pad, then `ttnn.to_layout(output, ttnn.TILE_LAYOUT)` (a tt-metal program with untagged transfers) and compare the result to torch. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `dfb_out_rows` | 0 | `entry_bytes = align_up(W_o·elem_bytes, l1_alignment)` | `max(R,W) · ring_blocks · block_rows` | spans data rows (`block_rows` × `ring_blocks`) and the whole row; streams over the core's data list | input dtype (raw bytes; dataflow-only) | reader | writer | producer **explicit** (listed in reader's `disable_dfb_implicit_sync_for`; all reader transfers `{.trid = 0}`); consumer **implicit** (one tagged write per entry) | whole kernel |
| `dfb_pad_row` | 1 | `entry_bytes` | `max(R,W)` (Phase 0: 1) | constant: one template row per writer thread; spans the whole row | input dtype (raw bytes) | reader | writer | explicit both sides (listed in reader's and writer's `disable_dfb_implicit_sync_for`); writer's template writes `{.trid = 0}` | whole kernel (pushed once, popped once, only on cores with `pad_count > 0`) |

`data_format_metadata = input.dtype` on both (not required for DM-only rings, harmless). Sync counts: reader pushes `data_count` entries to `dfb_out_rows` (whole blocks + last partial), writer consumes `data_count` by implicit writes; reader pushes 1 template, writer `wait_front(1)` / `pop_front(1)` once — balanced on every core.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `build_pad_row_template` | 1 × `W_o` | `fill_l1_range` | — | `dfb_pad_row`, 1 (reserve → fill → push) | template visible to writer |
| 2 | `load_data_block` | `rows_this_block` × `W` | raw NoC | input accessor | `dfb_out_rows`, reserve `block_rows` | entries being filled |
| 3 | `fill_row_margins_first_lap` | `rows_this_block` × (`bw`+`aw`) | `fill_l1_range` | — | `dfb_out_rows` (same reserved entries) | margins valid for all later laps |
| 4 | `publish_data_block` | the block | raw NoC barrier + DFB push | — | `dfb_out_rows`, push `rows_this_block` | block visible to writer |
| 5 | `store_pad_rows` | `pad_count` × `W_o` | raw NoC | `dfb_pad_row`, wait 1 (held) | output accessor | template popped after the barrier |
| 6 | `store_data_block` | `rows_this_block` × `W_o` | raw NoC (implicit DFB write) | `dfb_out_rows`, consumed by implicit writes | output accessor | slots return to reader via ISR acks |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `build_pad_row_template`, `fill_row_margins_first_lap` | helper | `dataflow_kernel_lib::fill_l1_range<val_size>(start_addr, n_bytes, val)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-164` | `val_size = elem_bytes` (CT), `val = pad_bits`, addresses from `DataflowBuffer::get_write_ptr()` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:407`) | — | `dfb_pad_row` / `dfb_out_rows` | `n_bytes` = row extent / margin extent |
| ring credits | raw_api | `DataflowBuffer::reserve_back / push_back / wait_front / pop_front` | `dataflow_buffer.h:243-246` | `block_rows` / `rows_this_block` / 1 | | | `block_rows` |
| `load_data_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(src, dst, size_bytes, src_args, dst_args, noc_opts)` | `tt_metal/hw/inc/api/dataflow/noc.h:246-275`; DFB dst offset semantics `dataflow_buffer.h:585-611`; accessor page offset `tt_metal/hw/inc/api/tensor/noc_traits.h:135-142` | `size = in_row_bytes`, `{.page_id = i}`, `{.offset_bytes = j·entry_bytes + bw·elem_bytes}`, `{.trid = 0}` | input accessor | `dfb_out_rows` | reads per block = `block_rows` |
| `publish_data_block` | raw_api | `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:739` | | | | once per block |
| `store_pad_rows` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(CoreLocalMem<uint32_t>(addr), acc_out, size, {}, {.page_id}, {.trid = 0})`; `async_write_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:405-475`, `noc.h:760-765`; `tt_metal/hw/inc/api/core_local_mem.h:31`; in-tree pattern `ttnn/cpp/ttnn/operations/experimental/quasar/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp:55-60` | `size = out_row_bytes` | `dfb_pad_row` | output accessor | `pad_rows_per_barrier` |
| `store_data_block` | raw_api | implicit `Noc::async_write<NocOptions::TXN_ID>(dfb, acc_out, {}, {.page_id})` | `noc.h:937-942`; `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl:824-892` | — (size = entry) | `dfb_out_rows` | output accessor | — |
| `release_trid` | raw_api | `noc_async_write_set_trid(0, noc.get_noc_id())` (writer; reader's last transfer already carries trid 0, reset harmless) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2579` | | | | |
| host split | host | `ttnn.split_work_to_cores(grid, n, row_wise=True)` | `ttnn/ttnn/core.py:19` | n = D, then n = P | | | core count via grid |

Helpers considered and rejected (for the raw NoC entries): there is no compute kernel, so the compute helpers (`tilize`/`untilize` — data stays row-major and never becomes tiles; `eltwise_*`, `reduce`, `matmul_block` — no arithmetic) do not apply. `McastArgs` / `ttnn.Mcast` (`ttnn/cpp/ttnn/kernel_lib/mcast/`) — no reuse-shared operand (Blocking Model). `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile` (`l1_helpers.hpp:52-77`) fill whole tile entries with zero only; the template/margins need an arbitrary value over element ranges, which `fill_l1_range` covers. The kernel library has no row-copy / pad dataflow helper (the DM helpers in `l1_helpers.hpp` are address/fill utilities only), so the NoC calls are the block operation's realization.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A missing pad write looks correct | Phase 0 value is 0.0 and simulator DRAM starts zeroed | coverage test with `pad_bits` of 1.0 through the artifacts entry |
| Implicit write overruns into the next page | implicit writes send `entry_bytes`, not the page size | `entry_bytes = align_up(out_row_bytes, l1_alignment) ≤ output.buffer_aligned_page_size()`, asserted on host |
| Margins wrong after ring wrap | margins are filled on the first lap only | holds because reads never touch margins and the writer only reads; every push except the core's last is a whole block, so slot positions are `e mod num_entries`; regime-pinned test with `block_rows_override=3` exercises the wrap |
| Sticky transaction id | writer ends on implicit (ring-tagged) writes; the next tt-metal program's untagged writes would be counted for a ring | writer's last statement `noc_async_write_set_trid(0, noc.get_noc_id())`; every explicit transfer in both kernels carries `{.trid = 0}`; session test |
| Explicit producer + implicit consumer on one DM→DM ring | the mixed form is shown for DM→compute rings (`toy_thread_plan` `reader_mixed.cpp`), not DM→DM | per-side opt-out is the documented rule (`validation/resource/dfb.cpp:284-296`). If it fails on the simulator, fall back to an explicit writer on `dfb_out_rows` (`wait_front(block_rows)`, tagged writes, one barrier, `pop_front`) and list it in the writer's `disable_dfb_implicit_sync_for`. The structure does not change. |
| Pad-row ↔ output-row mapping off by a segment | the q-th pad row is found by a recursive count | the host function is shared by every core; test shapes pad several dims at once (`(4,8,47,256)` all dims, `(1,1,64,17)` H+W) |
| fill race on a shared 4-byte word | the tail starts at an odd element for bf16 (`W = 17, 47, 50`) | `fill_l1_range<elem_bytes>` stores element-wide at the range edges (`l1_helpers.hpp:150-163`), never into the read's bytes |
| Template DFB on cores with no pad rows | an unbalanced push/pop hangs at kernel exit | both kernels gate the template on the same RT arg `pad_count > 0` |
