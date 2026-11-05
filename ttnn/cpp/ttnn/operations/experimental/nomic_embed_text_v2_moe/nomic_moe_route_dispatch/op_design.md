# Operation Design: nomic_moe_route_dispatch

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + data movement + cross-core collective) |
| Goal | The front of a routed MoE layer in ONE `ttnn.generic_op` program: router scores, top-2, sort of the 2T slots into 32-row expert regions, and the gathered (tilized) activation matrix plus the metadata the expert ops read. Replaces `router_front_end` + `routing_metadata` + `nomic_moe_dispatch`. |
| Math | `p[t,:] = softmax_E(x[t,:] @ W)` (fp32); `sel[t,0..1]` = top-2 of `p[t,:]`, ties to the lower expert; `wgt[t,k] = p[t,sel[t,k]]`; `counts[e]`; `start_e = sum_{e'<e} 32*ceil(counts[e']/32)`; slot (t,k) with `sel[t,k]=e` -> row `start_e + #{slots of e with token < t}`; `dispatched[row] = x[t]` (bit-exact); `combine_index[k,t] = row`; `combine_weights[0,k,t,0] = wgt[t,k]`; `expert_counts[0,e] = counts[e]`. Rows holding no slot: `dispatched[r] = x[r mod T]`. |
| Mode | Derivative (behaviour of the three stock ops), new fused scheme |
| References | `ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/{router_front_end,routing_metadata,nomic_moe_dispatch,nomic_residual_layer_norm}/`; `models/experimental/nomic_embed_text_v2_moe/tt/{router,moe,experts}.py`; `eval/golden_tests/nomic_moe_route_dispatch/{feature_spec,helpers,test_regression}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,precision_convention}.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries cited per knob below) |

Model constants the op specializes on (single source: module constants in the op file): `HIDDEN = 768`,
`NUM_EXPERTS = 8`, `TOP_K = 2`, `TILE = 32`, `HIDDEN_TILES = 24`, `MAX_SPARE_TILE_ROWS = 3`,
`T_MIN_PAD, T_MAX = 1024, 4096` (on `T_pad`).

Derived shape quantities (host, alignment-aware, `ceil` everywhere):

| Symbol | Formula |
|--------|---------|
| `T` | `B * S` |
| `T_pad` | `32 * ceil(T / 32)` (per-image tile padding is never folded: Phase 0 has `S % 32 == 0`, so `T_pad = T`) |
| `tensor_token_tiles` | `B * ceil(S / 32)` |
| `R` | `32 * (ceil(2T / 32) + E - 1) + 32 * spare_tile_rows` |
| `tensor_out_tile_rows` | `R / 32` |
| `num_pad_rows` | `R - 2T` (every row not holding a slot), `<= 32*(E-1+3)+31 = 351` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor | yes | `(B,1,S,768)` bf16 TILE, `T_pad` in 1024..4096, `S % 32 == 0` (Phase 0) | - | buffer address RT |
| `weight` | ttnn.Tensor | yes | `(768, 8)` fp32 TILE DRAM interleaved (validate: rank 2, `E <= 32` else ValueError; `E != 8` refused) | - | buffer address RT |
| `spare_tile_rows` | int | no | 0..3 (non-int or out of range: ValueError) | 0 | enters only `R` (host) and the pad-run bounds (RT) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | any fidelity / approx; `fp32_dest_acc_en=False` refused | `default_compute_kernel_config()` = HiFi4, fp32 acc, approx off | compute config + `EXP_APPROX` define |
| `memory_config` | `ttnn.MemoryConfig` | no | buffer type L1 (Phase 0) / DRAM (TARGET); the op places `dispatched` with `output_memory_config(R, device, buffer_type)` | `None` -> L1 | - |

Exports (module `ttnn.operations.nomic_moe_route_dispatch`): `nomic_moe_route_dispatch`,
`default_compute_kernel_config`, `output_memory_config(rows, device, buffer_type=L1)`,
`circular_buffer_bytes(input_shape, device, spare_tile_rows=0)`, `INPUT_TAGGERS`, `SUPPORTED`,
`EXCLUSIONS`, `validate`, `plan`.

`output_memory_config(rows, device, L1)` **delegates** to
`ttnn.operations.nomic_moe_dispatch.output_memory_config(rows, device, L1)` (single source of truth:
the consumer grouped_matmul_w1 is tuned for exactly that layout); DRAM returns `ttnn.DRAM_MEMORY_CONFIG`.
The output shard geometry (`pieces_per_row`, `block_width_tiles`, shard count, blocks per core) is read
from `ttnn.operations.nomic_moe_dispatch.plan(R)`, never re-derived.

## Tensors

### Input

| Property | `input_tensor` (x) | `weight` (W) |
|----------|---------------------|--------------|
| Shape | `(B, 1, S, 768)` | `(768, 8)` |
| Dtype | bfloat16 | float32 |
| Layout | TILE | TILE (24 tile rows x 1 tile column; columns 8..31 are tile padding, never trusted) |
| Memory | Phase 0: `nomic_residual_layer_norm.input_memory_config(shape, device)` (ND-sharded L1, shard `[1,1,rps*32, ht*32]`, row-major round robin over the first `G*num_groups` cores of the 11x10 grid). TARGET: L1 / DRAM interleaved | DRAM interleaved |

### Output

| Tensor | Shape | Dtype | Layout | Memory |
|--------|-------|-------|--------|--------|
| `dispatched` | `(1, 1, R, 768)` | bfloat16 | TILE | `output_memory_config(R, device, L1)` = nomic_moe_dispatch's ND shard (`32 x w*32` shards, round robin over 110 cores). TARGET: DRAM interleaved |
| `combine_index` | `(2, T_pad)` | uint32 | ROW_MAJOR | DRAM interleaved (page k = row k, `T_pad*4` B) |
| `combine_weights` | `(1, 2, T_pad, 1)` | float32 | TILE | DRAM interleaved (page `k*T_pad/32 + tile_row`; column 0 = weight, columns 1..31 = 0, as routing_metadata writes) |
| `expert_counts` | `(1, 8)` | uint32 | ROW_MAJOR | DRAM interleaved (one 32 B page) |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Both the input and the main output arrive with
their placement **pinned by the caller**, so the core-assignment of two of the axes is read off the
shards; what remains a decision is the extent cut inside each core (one block, the whole shard) and
everything the scheme introduces between the two shards.

Plan parameters (host, `plan(x)`; single source per symbol):

| Symbol | Meaning | Source |
|--------|---------|--------|
| `G` | hidden slices a token row is cut into (group size) | x's `nd_shard_spec.shard_shape[-1] / 32` -> `ht`; `G = 24 / ht` |
| `ht` = `block_hidden_tiles` | hidden tiles a core holds of each token row | x's shard shape |
| `rps` = `block_token_tiles` | token tile rows a group holds (the shard height) | x's shard shape `[-2] / 32` |
| `num_groups` | token groups (`B * ceil(ceil(S/32) / rps)`) | x's shard spec |
| `core_token_tiles` | real tile rows of this group (last group of a batch may be short) | host, per core RT |
| `k` = `pieces_per_row`, `w` = `block_width_tiles` | output shard = 32 rows x `w` tiles, `k*w = 24` | `nomic_moe_dispatch.plan(R)` |
| `num_shards`, `max_blocks_per_core` | `tensor_out_tile_rows * k`, `ceil(num_shards / 110)` | `nomic_moe_dispatch.plan(R)` |
| `rows_reader` = `PUSH_TILE_ROWS_READER` | token tile rows whose rows the reader RISC pushes; the writer pushes the rest | op constant knob, Phase 0 `rps // 2` |

Per production shape (derived, recorded so the implementer can check `plan()`):

| (B,S) | T | R | G | ht | rps | groups | token cores | k | w | max_blocks_per_core | slice-piece writes per row |
|-------|---|---|---|----|-----|--------|-------------|---|---|---------------------|----------------------------|
| 8x512 | 4096 | 8416 | 3 | 8 | 4 | 32 | 96 | 2 | 12 | 5 | 4 |
| 6x512 | 3072 | 6368 | 2 | 12 | 2 | 48 | 96 | 2 | 12 | 4 | 2 |
| 8x384 | 3072 | 6368 | 2 | 12 | 2 | 48 | 96 | 2 | 12 | 4 | 2 |
| 4x512 | 2048 | 4320 | 3 | 8 | 2 | 32 | 96 | 3 | 8 | 4 | 3 |
| 8x256 | 2048 | 4320 | 3 | 8 | 2 | 32 | 96 | 3 | 8 | 4 | 3 |
| 2x512 | 1024 | 2272 | 2 | 12 | 1 | 32 | 64 | 3 | 8 | 2 | 4 |
| 4x256 | 1024 | 2272 | 2 | 12 | 1 | 32 | 64 | 3 | 8 | 2 | 4 |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `token` (T_pad/32 tile rows) | **dependent** for the slot rows (every row is an exclusive scan of per-expert counts over the WHOLE axis); independent for scores / selection / per-group counts | `block_token_tiles` (`rps`) | the whole shard height (`rps`, 1..4), one block | x's shard spec (`plan.rps`) | pinned by the x shard: group `g` = shard rows of `(b, sg)`; the cross-core scan is the count exchange (gather to coordinator + table multicast) | knob-turn for interleaved x (G = 1, tile rows split by `split_work_to_cores`); the combine already exists |
| `hidden` (24 tiles) | **dependent** for the logits (the matmul reduces over it); independent for the row copy (a row is copied slice by slice) | `block_hidden_tiles` (`ht`) | the whole slice (`ht` = 8 or 12), one block | x's shard spec (`plan.ht`) | pinned by the x shard: member `c` of a group holds slice `c`; combine = intra-group partial-logit exchange | G = 1 (whole rows) is the interleaved regime, a knob-turn |
| `expert` (E = 8, one tile column) | dependent (softmax normalizes over it, top-2 ranks over it) | `block_expert_tiles` | 1 (the whole axis: E = 8 <= 32, one tile) | constant `NUM_EXPERTS` -> 1 tile | not split: one tile holds every expert; splitting would add a combine for 8 values | none (E is a model constant) |
| `choice` (top-k = 2) | independent (each choice is a separate slot) | `block_choices` | 2 (whole axis) | constant `TOP_K` | not split: both choices of a token come from the same score row on the same core | none |
| `out_row` (R/32 output tile rows) | independent (each 32-row block tilizes alone) | `block_out_tile_rows` | 1 output tile row per tilize block (the dispatch shard height is 32 rows: mechanism cap of the pinned output layout) | `nomic_moe_dispatch.SHARD_TILE_ROWS` via `plan(R)` | pinned by the output shard map: shard `s` -> core `s % 110`, local slot `s / 110` | none (consumer-fixed layout) |
| `out_piece` (k pieces of w tiles) | independent (pieces of a row are tilized separately) | `block_width_tiles` (`w`) | `w = 24/k` (8 or 12), whole piece | `nomic_moe_dispatch.plan(R)` | pinned with `out_row` (shard = (row block, piece)) | none |
| `out_block` (a core's shards) | independent | `num_blocks_this_core` | ALL of the core's shards (`<= max_blocks_per_core`) in one tilize call | RT per core from `plan(R)` | output owner = every core of the 11x10 grid | sub-shard pipelining is perf lamp L3 |
| `slot` (2T slots + R-2T pad rows = R pushed rows) | independent once the table is known (each row is one copy) | (no block: rows are the transfer unit, each `<= w` tiles wide) | the core's whole token range in one push pass, split over the two DM RISCs by `rows_reader` | `PUSH_TILE_ROWS_READER` | owned by the core holding the token's slice | perf lamps L2, L7 |

**Stages the scheme introduces, re-tabled (their data has axes too):**

| Stage data | Axes | Assignment | Idle cores during it | Why not spread further |
|------------|------|-----------|----------------------|------------------------|
| W slices (24 tiles fp32) | hidden | G sender cores, one slice each, read once from DRAM, multicast to the grid | non-sender cores wait (compute untilizes x meanwhile: stall shadow) | the slice is the unit; G parallel senders already run. Perf R3.2: both RISCs of a sender core issue its DRAM reads (half the K tile pairs each, over their own NoCs; local semaphore `SEM_W_LOCAL`), W lands ~500 cycles earlier on 8x512 |
| partial logits (`rps` tiles per member) | token x expert | every member computes its own; every member sums all G (redundant, bit-identical) | none | the sum is `rps` tiles of 1 tile column; redundancy replaces a broadcast hop |
| scores / selection / counts | token x expert | softmax + top-2 split over the G members by 16-token half (Perf R3.1: partials reduce-scattered, code faces streamed to the group); selection on every member (redundant within a group) | none | selection is scalar RISC work on `rps*32` tokens; the leader-only alternative is regime `leader_select` (rejected) |
| count table (num_groups x 8 counts) | group x expert | coordinator **communication role only** (receives rows, multicasts table); the scan itself is done by EVERY core on its own copy | none; the coordinator does no exclusive computation | the scan is O(num_groups * 4) packed adds per core, cheaper than distributing results |
| pad-row list | out_row | every token core enumerates only the pad rows whose token it holds (closed-form per pad run) | none | already spread over token owners |
| owner landing (Perf R3.2) | - | every owner counts its own landed push writes (posted-write slave counter, 32 per owned block) | an owner waits only for the pushers that write its blocks | the global barrier (atomics to a coordinator + go multicast, ~1.5-1.9k cycles after the last landing) and the parked `ARRIVAL_PER_BLOCK` per-block atomics are deleted |

**Stall-shadow decisions** (scheme property; recorded here, realized by the schedule order):

| Wait | Who waits | Work scheduled into it | Legality |
|------|-----------|------------------------|----------|
| W multicast (DRAM latency + mcast) | compute on token cores | nothing since Refinement 4: the senders issue W first and the slices alternate over NOC0 / NOC1 (`W_SLICE_NOCS`), so W lands in ~3-6k cycles and the matmul runs first; `untilize_x_block` moved into the select shadow (was: untilize ran first, the whole-piece stage gather congested NOC0 under the W multicasts) | the untilize is needed only by the push, after the count table |
| W multicast | reader | reduce scalers + the `cb_mask_keep` tile are built | constants |
| intra-group partial exchange | reader | nothing independent remains on the reader; compute is idle for its duration (short: G-1 unicasts of `rps*2KB`) | - |
| global count exchange | reader (all token cores) | member `G-1` builds and writes `combine_weights` (depends only on its own selection); local per-slot ranks are computed before the counts are sent | combine_weights needs no global quantity |
| global count exchange | compute | nothing: all compute that does not need the table is done | - |
| owner landing | reader | `combine_index` / `combine_weights` writes issued right after the pushes, before the land wait | need only the table |

No reorder changes floating-point order relative to the precision baseline: the logits sum is a fixed
contributor order `(p_0 + p_1) + p_2`, identical on every member.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| every CB | `NUM_BLOCK_SLOTS` (1) | 1 block | Nothing to overlap: each core runs exactly one block per phase, and the two downstream gates (table, landing) are global barriers, so a second slot can never be filled before the first is drained. Sub-block pipelining is an algorithm change recorded as lamp L3/L10, not a depth turn. |
| `cb_x_tiles` | publish rounds (2) | 2 | The resident shard is published twice (untilize round, matmul round); each round pushes exactly the CB capacity so the ring closes. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| packed u16 count table (2 counts per word) | per-group per-expert count `<= rps*32 <= 128` | `rps <= 4` (nd_norm `MAX_ROWS_PER_SHARD`), validated on host | carry between packed halves corrupts counts |
| fp32 DEST, half sync (4 tiles) | matmul output subblock (`rps x 1`), logits chain slots (needs 3: sum, operand, fill) | `rps <= 4`; subblocking left to `matmul_block` | DEST overrun |
| output shard (32 rows) | `block_out_tile_rows` = 1 | pinned by `plan(R)` | wrong output layout |
| push transfer unit = one (row, slice, piece) intersection | `<= w*64 B <= 768 B`, 64 B granular | one-packet NoC write | split packets / misaligned L1 writes |
| L1 NoC alignment 16 B | count-table row = 16 B; piece offsets multiples of 64 B | row stride 16 B | dropped / merged bytes |
| DRAM write alignment (64 B) | `combine_index` segment offset = `first_token*4` | groups start on 32-token boundaries (128 B) | misaligned DRAM writes |
| pad-list capacity | pad rows a core pushes `<= num_pad_rows <= 351` | `cb_slot_rows` pad section sized `MAX_PAD_ROWS = 32*(E-1+3)+31` | overflow into the slot section |
| landed semaphore | arrivals `= 2 * num_token_cores <= 220` | 32-bit semaphore | - |
| W slice offset in `cb_router_weight` | in1 base = `c*ht` | `In1BaseOffsetFn` returns `c*ht` | wrong slice multiplied |

### Regimes

Selection predicate (host, `plan()`): `R_route = (in_placement == "nd_norm") and (out_buffer_type == L1) and (alignment == "tile_aligned")`.
Any other cell is refused by `validate()` in Phase 0 (SUPPORTED), so regime-pinned tests are the golden
cells of each regime as they are built.

| Regime | Status | Predicate | Block | Data movement (vs. minimum) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------|--------------------------|
| `R1 nd_push_l1` | **built** | `in_placement == nd_norm and out_buffer_type == L1 and alignment == tile_aligned` | per token core: `rps x ht` x tiles (whole shard); per owner: all `num_blocks_this_core` shards of `1 x w` tiles | Minimum named at the L1/NoC boundary: x crosses no DRAM (read in place twice from L1: untilize, matmul), W crosses DRAM once (G senders, one slice each) and the NoC once (multicast), every dispatched row crosses the NoC exactly once (one push per slice-piece intersection) and is tilized in place into the output shard (no output byte crosses the NoC after tilize). Above minimum: partial logits (`(G-1)*rps*2KB` per member, the price of the pinned hidden split), the count table gather + multicast (inherent global dependency), `R-2T` pad rows pushed (contract: finite x copies). | Fixed per-block cost paid ONCE per core per phase: matmul init + 2 data-format reconfigs, untilize/tilize init+uninit, softmax chain inits (5 phases), one CB handshake per CB, 2 global rendezvous. A smaller token block (e.g. 1 tile row) would multiply the inits by `rps` and add nothing: no stage can start before the global table. |
| `R2 interleaved_input` | **built** (Refinement 2: `_interleaved_ranges` = row-wise `split_work_to_cores` of the T_pad/32 tile rows, rps = the larger per-core count; the per-core reader/writer token split became a runtime arg so 1-tile-row cores still push on both RISCs). Was: deferred: TARGET `in_placement in {l1_interleaved, dram_interleaved}`; Phase 0 is the production placement. Reachable because G is a parameter: G = 1 makes every core a 1-member group (no partial exchange, W base offset 0, one slice = whole row, `k` writes per row), and the reader's sharded publish becomes a TensorAccessor tile read of the core's `split_work_to_cores` tile rows | `in_placement != nd_norm` | `core_token_tiles x 24` tiles per core | x crosses DRAM once (dram_interleaved) or the NoC once (l1_interleaved) instead of 0 | same as R1 |
| `R3 ragged_tokens` | **built** (Refinement 2: `group_ranges` clamps the last group's real tokens; the tile-padding slot words are zeroed and written by the RISC holding the core's last tokens). Was: deferred: TARGET alignment `ragged` (B = 1, T % 32 != 0). Reachable: the selection/count loops already run on a per-core RT real-token count; padded tokens get `combine_index = 0`, `combine_weights = 0`; pad rows map to `r mod T` real tokens | `alignment == ragged` | as R1 | as R1 | as R1 |
| `R4 dram_output` | **built** (Refinement 1; owner map kept as R1's round robin `shard % 110`, which is the same balanced per-core count as a row-wise `split_work_to_cores` and keeps the push addressing shared). Was: deferred: TARGET `out_buffer_type == DRAM` (scheme-change on the output side only: owners become a balanced `split_work_to_cores` of the R/32 x k shards, tilize into a 2-deep streaming CB and write tiles to DRAM interleaved; pushes target the owner's staging exactly as in R1). Not in Phase 0: the consumer is tuned for the L1 layout and the model keeps `dispatched` in L1 | `out_buffer_type == DRAM` | owner block = 1 shard, depth 2 | dispatched crosses DRAM once (write) | amortizes per-tile DRAM write issue |
| `R5 whole_row_members` | deferred: an intra-group exchange of untilized slices so each member pushes WHOLE output pieces (k writes per row instead of the slice-piece intersection count). Bytes-neutral on the push and adds `(G-1)/G` of each member's slice as cross-core traffic; it saves transactions only where `G != k` (8x512, 2x512, 4x256: 4 -> 2/3 writes per row) and is pure overhead where `G == k` (6x512, 8x384, 4x512, 8x256). Reachable: the push addressing is per (slice x piece) interval, and a whole-row member is the G = 1 case of it | `G != k` (if built) | as R1 | + `(G-1)/G * rps*32*ht*64 B` per member cross-core | fewer, larger writes |
| `R6 pull_gather` | rejected: superseded by R1. Owners would read their rows with an inverse permutation (gather_index) that first has to be built by an all-to-all of slot rows (stock routing_metadata's assembler step) and then distributed to every owner (stock dispatch's row multicast): one extra global exchange on the critical path for the same row bytes | - | - | + gather_index build (R*4 B all-to-all) + distribution | - |
| `R7 row_tensor_roundtrip` | rejected (dead end): the stock shape - untilize x into a row-major tensor, then gather from it - writes and re-reads 6.3 MB of rows through a tensor and needs gather_index; no good scheme passes through it | - | - | + T*1536 B write + read | - |
| `R8 permutation_matmul` | rejected: `dispatched = P @ x` with a one-hot P is exact but costs R*T*768*2 = 53 GFLOP | - | - | - | - |
| `R9 leader_select` | rejected: superseded by redundant selection. One member selects and broadcasts slot rows to the other members: adds one intra-group hop on the critical path to save `(G-1)` copies of a few-hundred-cycle scalar loop | - | - | + slot rows per member | - |

### Traffic ranking

Candidate splits, ranked by crossings per tier (8x512 numbers; bytes are totals over the chip):

| Rank | Split | DRAM | Cross-core NoC | Verdict |
|------|-------|------|----------------|---------|
| 1 | Token groups x hidden slices as pinned by the x shard (R1), push gather | W 96 KB once; metadata writes (combine_weights 1 MB, combine_index 32 KB) | rows 12.9 MB (R x 1536 B, inherent), partial logits 3 MB, W multicast 96 KB to 110 cores, table < 1 KB x 110 | **chosen**: x never leaves L1, every row crosses the NoC once |
| 2 | Same, pull gather (R6) | same | rows 12.9 MB + gather_index build and distribution | rejected: extra global exchange |
| 3 | Re-split tokens so each core holds whole rows (G = 1) from the pinned shard | same | + all of x (6.3 MB) re-shuffled across cores before anything starts | rejected: moves x once more than R1 to save 3 MB of partials |
| 4 | Split the expert axis across cores | same | + E-way combine of 8 values per token | rejected: one tile already holds every expert |
| 5 | Per-core DRAM re-read of W instead of multicast | W x 96 (3-4.5 MB) | - | rejected: DRAM-bound on the critical path |

The dependent token axis IS split across cores (by the pinned shard) and its combine (the count
exchange) is what R1 builds; this is the cheapest-traffic split available because the caller pinned
both shards.

### Block schedule

Logical schedule per core. Token-core phases run on token cores; `tilize_gathered_block` runs on all 110.

```cpp
// token core (group g, member c), one block = the whole shard
publish_x_block();                 // reader: cb_x_tiles round 1 (sharded, zero-copy)
receive_weight_slice();            // senders (first thing): read W slices, multicast, even slices on NOC0 / odd on NOC1
                                   // (writer); member c waits on the RISC whose NoC carries slice c
partial_logits_block();            // compute: x[rps x ht] @ W[ht x 1] -> rps fp32 tiles
exchange_partials_block();         // reader: send own partial to the G members, receive G-1
logits_block();                    // compute: (p_0 + p_1) + p_2, keep cols < E else -1e30
softmax_block();                   // compute: max, exp(x - max) on faces 0/2, 1/sum over E cols, mul, top-2 code -> scores
untilize_x_block();                // compute: x (or the stage gather, on the writer / NOC1) -> row-major staging
                                   // (Refinement 4: in the select shadow, off the W -> matmul critical path)
select_block();                    // reader + writer: decode the top-2 code, per-expert counts, per-slot local ranks
exchange_counts();                 // leader -> coordinator row; coordinator multicasts table
write_combine_weights_block();     // member G-1 (count-exchange shadow)
assign_rows_block();               // every member: starts, prefix, slot rows, pad list
write_combine_index_block();       // leader; coordinator writes expert_counts
push_rows_block();                 // reader + writer: one NoC write per (row, slice, piece)
owner_land();                     // wait until 32 * num_blocks posted push writes landed here (both NIUs)
// every core (owner)
tilize_gathered_block();           // compute: all owned shards, row-major -> tiles, into dispatched
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `publish_x_block` | `rps x ht` tiles | x shard (in place) | 2 pushes per core per call |
| `untilize_x_block` | `core_token_tiles` tile rows x `ht` tiles, two calls (reader half `rows_reader`, writer half) | x shard | untilize init/uninit once per call (2 per core) |
| `receive_weight_slice` | `ht` tiles of slice `c` (G slices land everywhere) | - | one DRAM read burst per sender, one multicast + one signal per sender |
| `partial_logits_block` | M = `core_token_tiles`, K = `ht`, N = 1 | W, x | one `matmul_block` call (one init, one reconfig) per core |
| `exchange_partials_block` | `core_token_tiles` tiles, faces 0 and 2 only (columns 0..15) | - | `G` unicast bursts + `G-1` semaphore incs per member, one barrier |
| `logits_block` | `core_token_tiles` tiles, contributor-major input `G x core_token_tiles` | `cb_mask_keep` | one chain per core |
| `softmax_block` | `core_token_tiles x 1` tiles (Ht = rows, Wt = 1) | scalers | one reduce(max), one chain, one reduce(sum)+recip, one chain per core |
| `select_block` | `core_token_tiles * 32` tokens x 8 experts (scalar) | scores | one pass |
| `exchange_counts` | `num_groups x 16 B` table | - | 1 posted unicast per leader (Perf R3.3: no barrier, no inc); 1 linked multicast (table + signal) by the coordinator |
| `write_combine_weights_block` | `2 x core_token_tiles` fp32 tiles | - | one zero-fill + `2*core_token_tiles` tile writes per group |
| `assign_rows_block` | 2 x tokens slot rows + pad list | table | one pass |
| `write_combine_index_block` | 2 segments of `core_tokens*4 B` | - | 2 writes per group |
| `push_rows_block` | the RISC's tokens: 2 slots + its pad rows per token, each split into slice-piece intersections | x row staging | one write per intersection, one barrier + one semaphore inc per RISC |
| `owner_land` | - | - | none (each owner polls its NIU posted-write slave counters) |
| `tilize_gathered_block` | `num_blocks_this_core` blocks of 1 x `w` tiles in ONE call | - | one tilize init/uninit per core |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 untilize placement | Untilize first hides the W fetch but delays the scores (and so every core's counts) if untilize outlasts the W fetch | untilize after `softmax_block` (inside the count-exchange wait), or split: half before matmul, half after |
| L2 push split over the two DM RISCs (`PUSH_TILE_ROWS_READER`) | The push is RISC-issue-bound (catalog `split_reader`); at `rps = 1` the default gives the reader nothing; with G=3,k=2 member 1 issues twice the writes of members 0/2 | `rows_reader = 0` (writer only), parity split of tokens, or member-aware split. **Built (Refinement 3)**: token-granular split (`PUSH_READER_SHARE`), and the member imbalance removed by the whole-piece push (`PUSH_WHOLE_PIECES`, host `_push_layout`): each output piece is pushed by the member holding most of it (ties split rows by parity), after it gathers the piece's other tiles from its peers' resident x shards into `cb_stage_tiles` and untilizes that range |
| L3 landing barrier (grid synchronization) | One global barrier makes every owner wait for the slowest pusher before any tilize starts (overlap lamp) | per-owner counted arrival: each pusher RISC atomically adds the number of pieces it wrote into each owner (owner expects `32 * intersections` per shard), owners tilize each shard as it completes. **Refinement 5**: built as `ARRIVAL_PER_BLOCK` (semaphore per owned block counting 64 B row granules, one atomic per (tile row, piece) run after the RISC's write barrier; owners publish blocks as they complete and interleave their combine_weights writes). Correct, parked off: 15-20 non-posted atomics a RISC at ~160-200 cycles each (2.8-3.4k cycles after every push) outweigh the ~2.4k-cycle barrier hop + ~1.5k tilize overlap (8x512 ~parity, rps <= 2 rows +1-2 us). Full-grid multicast atomics (one per RISC) measured 160+ us. The tail the barrier hid was the leader's 1 MB of combine_weights DRAM writes; `WEIGHTS_SPREAD` (member c writes every G-th tile) took it off (8x512 -1.3 us). **Perf R3.2**: superseded by owner-side landing: pushes are posted writes and each owner counts them on `NIU_SLV_POSTED_WR_REQ_RECEIVED` of both NIUs (only the pushes are posted, every owned block gets one write per row), so no atomics and no go; `ARRIVAL_PER_BLOCK` deleted (8x512 -0.5..-1.0 us) |
| L4 coordinator placement / count exchange shape | Last core (10,9) is a grid corner: longest unicast paths for leaders; two-hop gather + multicast | grid-center coordinator; all-gather by per-leader multicast of the 16 B row |
| L5 W delivery | 4 KB tiles multicast (half is padding) from G senders; 96 KB L1 per core | faces 0 and 2 only (half the bytes, two reads per tile); one sender for all slices |
| L6 combine_weights write bytes | full 4 KB tiles (1 MB at T=4096) match routing_metadata bit for bit in the padding columns | faces 0 and 2 only, if the consumers are confirmed to read column 0 only |
| L7 push issue order | lockstep cores walking tokens in the same order land on the same owners at once (measured on nomic_moe_dispatch, Refinement 3; catalog `bank_stagger`) | start each core's walk at token `(core_idx * stagger) mod core_tokens`. Built as `PUSH_STAGGER_TOKENS` (per group), measured flat on 8x512, default 0 |
| L8 NoC assignment | reader NOC0 / writer NOC1 default (catalog `noc_placement`) | both pushers on the NoC with fewer hops to the owner (path-aware). Measured (Refinement 3): swapping the RISCs' NoCs speeds the L1-out 8x512 push but regresses DRAM out by ~30 us; not adopted |
| L9 tilize/untilize under fp32 DEST | fp32 DEST (needed for the scores) halves the DEST tiles the untilize/tilize helpers get, which may push them onto their block-split or slower path | measure helper time at `w`, `ht` = 8 vs 12 against the standalone nomic_moe_dispatch tilize; the alternative is the helper's other path selection, not lowering the program's precision |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x into compute | bf16 tiles in the core's L1 shard | `cb_x_tiles` backed on x's buffer (`ttnn.cb_descriptor_from_sharded_tensor`), zero-copy; reader publishes `rps*ht` pages twice | never re-read over the NoC |
| x row-major staging | bf16 row-major, row = `ht*64 B` | `untilize` into `cb_x_rows_reader` (tile rows `[0, rows_reader)`) and `cb_x_rows_writer` (rest) | each pushing RISC owns its own staging CB |
| W | fp32, faces 0 and 2 only (Perf 2: the expert columns; half the bytes) | one sender per NoC (Perf 2: every slice a NoC carries goes out as ONE multicast, e.g. slices 0 and 2 on NOC0 at G = 3; full-grid multicasts on one NoC serialize per command): TensorAccessor reads of faces 0/2 of its slices into their compact slots of `cb_router_weight` (`nomic_moe_route_dispatch_w_layout.hpp`; compute reads them through the 1 KB-page view `cb_w_faces`), then ONE multicast to the 11x10 rectangle, **write barrier, then** the slice-c signal multicast (`ttnn.Mcast` fixed sender; one channel per slice, semaphore `sem_weight[c]`) | barrier-before-signal is what makes the W region quiescent grid-wide once any core has used every slice (needed by the `cb_router_weight` / `cb_gathered_rows` alias) |
| partial logits | fp32 tiles, faces 0 and 2 (expert columns 0..15) | compute packs member c's `rps` partial tiles straight into region `c` of its own `cb_partials` (Perf 2); the reader writes faces 0/2 to the higher-index members on NOC0 and the writer to the lower-index ones on NOC1 (no write wraps a row ring), each: write barrier, inc `sem_partials` on its peers | peers' addresses = the same CB address on every core |
| scores | fp32 code tiles (faces 0, 2; Perf 2: top-2 decided on the SFPU) | published one tile at a time; reader and writer read their tokens' rows locally (the writer peeks, the reader pops) | token i's row at word `(i < 16 ? 0 : 512) + (i % 16)*16`: col 0 = code (e1 = ctz, e2 = ctz(code & (code - 1)) & 7), col 2 = w1, col 4 = w2 |
| count table | 16 B row per group: 8 u16 counts | leader posts its row to the coordinator's `table` scratch at row `g` (Perf R3.3); the coordinator counts the rows on its `NIU_SLV_POSTED_WR_REQ_RECEIVED` (the only posted writes landing before the table; `land_base` advances by them), multicasts the table to the rectangle linked with `sem_table` set | coordinator = core 109 (last core) |
| slot rows | uint32 | reader computes all; keeps its half, pushes the writer half + writer pad list into `cb_slot_rows` | |
| rows | bf16 row pieces | for each pushed row r with token t, for each piece p overlapping slice c: one NoC write of `overlap_tiles*64 B` from `x_rows + t_local*ht*64 + (lo - c*ht)*64` to core `s % 110`, address `cb_gathered_rows + (s/110)*w*2048 + (r%32)*w*64 + (lo - p*w)*64`, with `s = (r/32)*k + p`, `[lo,hi) = [c*ht,(c+1)*ht) intersect [p*w,(p+1)*w)` | posted writes (Perf R3.2), capped on `NIU_MST_POSTED_WR_REQ_SENT`; no source barrier, the owner counts them |
| owner land | - | the reader polls both NIUs' `NIU_SLV_POSTED_WR_REQ_RECEIVED` until `32 * num_blocks_this_core` above its kernel-start value (Perf R3.2; replaced the coordinator go) | |
| dispatched | bf16 tiles in the core's output shards | reader waits for its owner landing count, pushes `num_blocks_this_core*w` pages of `cb_gathered_rows`; compute tilizes into `cb_dispatched` backed on the output buffer (zero-copy) | |
| metadata | uint32 / fp32 | leader: `combine_index` page k, offset `first_token*4`, each RISC its segment right after its pushes; `combine_weights` tiles `k*T_pad/32 + tile_row` (staging zeroed with `async_write_zeros`, column 0 written), the member's tiles alternated between the writer (NOC1, first) and the reader (NOC0) after their pushes (Perf 2: 1 MB of NOC0-only writes slowed the readers still pushing); coordinator: `expert_counts` | page sizes as RT args (no recompile per T) |

**Pad rows** (rows holding no slot, `R - 2T` of them; any x row is a valid copy). Built (Refinement 3): pad `p`
(in run order) belongs to push unit `p mod 2*num_groups` (one DM RISC of one token group) and carries that RISC's
`j mod risc_tokens`-th token, so the pads spread evenly whatever the routing. Planned: pad row `r` carries token `r mod T`. They form at
most 9 runs: the tail of each region `[start_e + counts_e, start_e + 32*ceil(counts_e/32))` and
`[regions_end, R)`. Each core intersects each run's (wrapping) token range with its own token range in
closed form and pushes those rows from its staging exactly like slot rows. Every output row is therefore
written exactly once per overlapping slice, by the cores holding that token.

**TARGET placements as regimes** (same rows as the regime table):

| TARGET value | Axis it cuts | Class | Regime |
|--------------|--------------|-------|--------|
| `in_placement = nd_norm` | token (dependent for routing, scan exists) x hidden (dependent for logits, exchange exists) | built | R1 |
| `in_placement = l1_interleaved / dram_interleaved` | token only (G = 1) | knob-turn (group size 1, reader source swap) | R2 |
| `alignment = ragged` | token extent | knob-turn (RT real-token count + padded-token zeros) | R3 |
| `out_buffer_type = DRAM` | output ownership | scheme-change on the output side | R4 |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | token core: its whole shard (`core_token_tiles x ht` tiles, one block); owner: all its output shards (one tilize call) |
| Grid | all 110 cores of the 11x10 compute grid run the program (every core owns output shards); token cores = x's shard cores (`G*num_groups`, the first cores row-major) |
| Per-core work | token core `i`: group `g = i / G`, member `c = i % G`, tokens `[first_token_g, first_token_g + core_tokens)`, `first_token_g = b*S + sg*rps*32`, `core_tokens = 32*min(rps, ceil(S/32) - sg*rps)` (with `g = b*SG + sg`, `SG = ceil(ceil(S/32)/rps)`); owner `j`: shards `s = j + 110*m`, `m < num_blocks_this_core = ceil((num_shards - j)/110)` |
| Roles | coordinator = core 109; W sender of slice `c` = core `108 - c`; roles may land on token cores (G*num_groups can reach 110) and are interleaved with that core's own token work so that a core never blocks on a wait before it has produced what others wait on |
| Remainder | token axis: last group of a batch may hold fewer than `rps` tile rows (`core_token_tiles` RT); output: `num_shards % 110` cores hold one extra shard (nominal CB counts are per-core RT, ring closes because each CB is used once) |
| Core mapping source | x: `ttnn.corerange_to_cores(x.memory_config().nd_shard_spec.grid, row_wise=True)` in shard order; output: `nomic_moe_dispatch.plan(R)` / its ND spec, shard `s` on core `s % 110` row-major |

Regime pinning: Phase 0 builds only R1; every golden cell with `in_placement = nd_norm`, `out_buffer_type = L1`,
`tile_aligned` exercises it on the 11x10 grid. `validate()` refuses a grid other than 11x10 (the output
layout is built for it).

## Circular Buffers

All CBs are declared on all 110 cores (identical addresses: remote writes address peers by the local
address). Sizes use `T4 = 4096` (fp32 tile), `T2 = 2048` (bf16 tile).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_x_tiles` | 0 | T2 | `rps*ht` | backed on x's shard: spans token x hidden of the block | bf16 | reader | compute | whole call (2 rounds) |
| `cb_router_weight` | 1 | T4 | 24 | all G slices land at slot `c*ht` (multicast targets one address); spans hidden | fp32 | reader | compute | until `partial_logits_block`; aliased with `cb_gathered_rows` |
| `cb_own_partial` | 2 | T4 | `rps` | spans token | fp32 | compute | reader | matmul -> exchange; aliased with `cb_masked_logits`, `cb_scores` |
| `cb_partials` | 3 | T4 | `G*rps` | contributor-major, spans token x contributor | fp32 (UnpackToDestFp32) | reader | compute | exchange -> `logits_block`; aliased with `weight_tiles` scratch |
| `cb_mask_keep` | 4 | T4 | 1 | constant: 1.0 in columns `< E`, 0 elsewhere | fp32 (UnpackToDestFp32) | reader | compute | whole call |
| `cb_masked_logits` | 5 | T4 | `rps` | spans token | fp32 | compute | compute | `logits_block` -> `softmax_block` (chain 2) |
| `cb_row_max` | 6 | T4 | `rps` | one REDUCE_ROW tile per tile row | fp32 | compute | compute | reduce max -> chain 2; aliased with `cb_recip_sum` |
| `cb_exps` | 7 | T4 | `rps` | spans token | fp32 | compute | compute | chain 2 -> chain 3 |
| `cb_recip_sum` | 8 | T4 | `rps` | one REDUCE_ROW tile per tile row | fp32 | compute | compute | reduce sum -> chain 3 |
| `cb_scores` | 9 | T4 | `rps` | spans token | fp32 | compute | reader | chain 3 -> `select_block` |
| `cb_max_scaler` | 10 | T2 | 1 | constant | bf16 | reader | compute | whole call |
| `cb_sum_scaler` | 11 | T2 | 1 | constant | bf16 | reader | compute | whole call |
| `cb_x_rows_reader` | 12 | T2 | `rows_reader*ht` | spans the reader's token rows x hidden | bf16 (row-major bytes) | compute | reader | untilize -> reader's pushes |
| `cb_x_rows_writer` | 13 | T2 | `(rps - rows_reader)*ht` | spans the writer's token rows x hidden | bf16 (row-major bytes) | compute | writer | untilize -> writer's pushes |
| `cb_slot_rows` | 14 | 4 B | `2*(rps-rows_reader)*32 + 2*MAX_PAD_ROWS + 2` | writer's slot rows (k-major), its pad (row, token) pairs, 2 counts | uint32 | reader | writer | `assign_rows_block` -> writer's pushes |
| `cb_gathered_rows` | 15 | T2 | `max_blocks_per_core*w` (capacity `max(that, 24*T4/T2)` through the alias) | spans the core's output blocks x piece width; remote pushers write it | bf16 (row-major bytes) | reader (publishes after `sem_go`) | compute | pushes -> tilize |
| `cb_dispatched` | 16 | T2 | `max_blocks_per_core*w` | backed on the output shards | bf16 | compute | (output tensor) | tilize |
| `reader_scratch` | 17 | 16 B | `num_groups + 4 + ...` (table rows, totals, drain word; selection ids/ranks `2*rps*32` words) | count table + selection state | uint32 | reader | reader (scratch, no handshake) | whole call |

Aliases (one `CBDescriptor` with several format descriptors): {`cb_router_weight`, `cb_gathered_rows`},
{`cb_own_partial`, `cb_masked_logits`, `cb_scores`}, {`cb_row_max`, `cb_recip_sum`},
{`cb_partials`, `weight_tiles` (reader scratch for combine_weights, `2*rps` tiles)}. Full ledger and
proofs of disjointness: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `publish_x_block` | `rps*ht` pages | raw (sharded publish) | - | `cb_x_tiles`, `rps*ht` | published (round 1; round 2 after compute pops) |
| 2 | `untilize_x_block` | 2 calls: `rows_reader` / `rps - rows_reader` tile rows x `ht` | `untilize<ht>` | `cb_x_tiles`, `rps*ht`, popped | `cb_x_rows_reader`, `cb_x_rows_writer` | x pages popped (ring closed) |
| 3 | `receive_weight_slice` | `ht` tiles | `ttnn.Mcast` / `McastArgs` sender/receiver | DRAM W | `cb_router_weight`, 24 pages pushed after `sem_weight[c]` | - |
| 4 | `partial_logits_block` | M=`core_token_tiles`, K=`ht`, N=1 | `matmul_block` | `cb_x_tiles` (round 2, popped), `cb_router_weight` (base `c*ht`, popped at end) | `cb_own_partial`, `rps` | - |
| 5 | `exchange_partials_block` | `rps` tiles x G | raw NoC (unicast faces + a landed-tile count write a peer, no barrier) | `cb_own_partial` / `cb_own_partial_west`, one page a packed tile (matmul runs one tile at a time in `TileOrder`, Perf R3.3) | `cb_partials`, G pages a tile once its own page is in and every peer's `sem_partials<c>` counter reached it; the writer sends every tile to every peer | - |
| 6 | `logits_block` | `rps` tiles | `eltwise_chain` | `cb_partials` (`G*rps`, popped), `cb_mask_keep` (no pop) | `cb_masked_logits`, `rps` | - |
| 7 | `softmax_block` | `rps x 1` tiles | `reduce` (MAX, ROW), `eltwise_chain` (sub bcast col + Exp), `reduce` (SUM, ROW, recip post-op), `mul` (bcast col) | `cb_masked_logits`, `cb_row_max`, `cb_exps`, `cb_recip_sum`, scalers | `cb_scores`, `rps` | scalers kept |
| 8 | `select_block` | tokens x 8 | raw RISC scalar | `cb_scores`, popped at end | `reader_scratch` | - |
| 9 | `exchange_counts` | table | raw NoC + `ttnn.Mcast` (table + signal) | scratch | scratch (table) | - |
| 10 | `write_combine_weights_block` | `2*core_token_tiles` tiles | raw NoC (zeros + tile writes) | scratch | DRAM `combine_weights` | - |
| 11 | `assign_rows_block` | slots + pad runs | raw RISC scalar | table | `cb_slot_rows` (writer half), scratch (reader half) | - |
| 12 | `write_combine_index_block` | 2 segments | raw NoC | scratch | DRAM `combine_index`, `expert_counts` | - |
| 13 | `push_rows_block` | the RISC's rows | raw NoC (one-packet writes) | `cb_x_rows_*`, `cb_slot_rows` (writer), popped at end | remote `cb_gathered_rows` | - |
| 14 | `owner_land` | - | raw NIU slave-counter reads (no helper exposes them) | - | - | - |
| 15 | `tilize_gathered_block` | `num_blocks_this_core` x `1 x w` | `tilize<w>` | `cb_gathered_rows`, `num_blocks_this_core*w` pushed after the owner landing count | `cb_dispatched` | - |

## API Mapping

> **Perf 1:** `logits_block` + `softmax_block` are no longer the helper rows below. They run as one raw-SFPI
> DEST window a tile (`kernels/nomic_moe_route_dispatch_softmax8.hpp`, exact fp32: contributor sum `(p_0 + p_1) + p_2`,
> 8-column row max / exp / sum / reciprocal / mul via lane rotates). `cb_mask_keep`, both scaler tiles and
> `cb_masked_logits` / `cb_row_max` / `cb_exps` / `cb_recip_sum` are gone. Justification and numbers: changelog `## Perf 1`.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_tiles, cb_router_weight, cb_own_partial)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | once, before any helper | - | - | - |
| `untilize_x_block` | helper | `compute_kernel_lib::untilize<ht, cb_x_tiles, cb_x_rows_*>` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:152-163` | `block_width_tiles = ht` (CT), `num_blocks` = tile rows (RT); `InitUninitMode::InitOnly` / `UninitOnly` across the two calls | `cb_x_tiles` | `cb_x_rows_reader`, `cb_x_rows_writer` | `ht`, `num_blocks` |
| `partial_logits_block` | helper | `compute_kernel_lib::matmul_block` + `MatmulBlockShape::of(in0_sb, 1, sb_h, 1, ht, 1)` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:366-398`, `:139`, `:169` | `In1BaseOffsetFn` returning `c*ht` (`:251`), `in1_policy` caller-managed (reader pushes 24 pages, compute waits/pops 24 around the call), reconfig `INPUT_AND_OUTPUT` (bf16 in0 on SrcB, fp32 in1 on SrcA) | `cb_x_tiles`, `cb_router_weight` | `cb_own_partial` | block = `core_token_tiles x ht`; subblock left to the helper/tuner (fp32 DEST: 4 tiles) |
| `logits_block` | helper | `eltwise_chain`: `CopyTile` p_0 -> D0, `CopyTile` p_1 -> D1, `AddBinary` (SFPU, fp32), `CopyTile` p_2 -> D1, `AddBinary`, `CopyTile` keep -> D1, `FillScalar(-inf)` -> D2, `Where<Float32>(D1, D0, D2)`, `PackTile` | `eltwise/api/chain.hpp:554,588,620`; `eltwise/binary/sfpu/basic.hpp:20`; `eltwise/generators/fill.hpp:20`; `eltwise/unary/special.hpp:37-38` | `IterationShape::tiles(core_token_tiles)`; partial input j read at tile offset `j*rps + i` | `cb_partials`, `cb_mask_keep` | `cb_masked_logits` | `core_token_tiles` |
| `softmax_block` (max) | helper | `reduce<PoolType::MAX, ReduceDim::REDUCE_ROW>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:673`, shape `:300` | `ReduceInputBlockShape::of(core_token_tiles, 1)`, `WaitUpfrontNoPop` | `cb_masked_logits`, `cb_max_scaler` | `cb_row_max` | rows |
| `softmax_block` (exp) | helper | `eltwise_chain(BinaryFpu<Sub, masked, max bcast Col>, Exp<EXP_APPROX>, PackTile)` | `eltwise/api/chain.hpp:568,620`; `eltwise/unary/math.hpp:22` | as stock `router_front_end` compute, shape `tiles(core_token_tiles)` | `cb_masked_logits`, `cb_row_max` | `cb_exps` | rows |
| `softmax_block` (sum) | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>` with post-op `recip_tile<FP32, Precise>` | `reduce_helpers_compute.hpp:673` | `WaitUpfrontNoPop` on `cb_exps` | `cb_exps`, `cb_sum_scaler` | `cb_recip_sum` | rows |
| `softmax_block` (mul) | helper | `mul<exps, recip bcast Col, scores>` | `eltwise/api/convenience.hpp:70` | `IterationShape::tiles(core_token_tiles)` | `cb_exps`, `cb_recip_sum` | `cb_scores` | rows |
| scalers | helper | `prepare_reduce_scaler<cb, PoolType::MAX / SUM, ReduceDim::REDUCE_ROW>(1.0f)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:70` | pool-type-aware overload, bf16 packed | - | `cb_max_scaler`, `cb_sum_scaler` | - |
| `receive_weight_slice`, table, go | helper | host `ttnn.Mcast` (`McastFixedSenderConfig`), device `McastArgs<>::sender(noc).send() / send_signal()`, `receiver(noc).receive() / receive_signal()` | `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp:110,133`; `mcast/kernel/mcast_args.hpp:41-76,220` | one channel per W slice + one for table + one for go | - | - | payload bytes |
| `tilize_gathered_block` | helper | `compute_kernel_lib::tilize<w, cb_gathered_rows, cb_dispatched>` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207` | `block_width_tiles = w` (CT), `num_blocks = num_blocks_this_core` (RT), symmetric pages, `Fp32Mode::Fast` (bf16 in: exact) | `cb_gathered_rows` | `cb_dispatched` | `w`, `num_blocks` |
| `exchange_partials_block`, `exchange_counts` (gather), `push_rows_block`, `land_barrier` (incs), metadata writes | raw_api | `Noc::async_write` (one-packet), `async_write_barrier`, `Semaphore::up / wait`, `async_write_zeros` | `tt_metal/hw/inc/api/dataflow/noc.h`; pattern: `routing_metadata/device/kernels/dataflow/routing_metadata_indices.cpp` | - | - | - | - |
| `select_block`, `assign_rows_block` | raw_api | scalar RISC code (order key on fp32 bits, top-2 with lower-index ties, packed u16 counts, exclusive scan) | pattern: `routing_metadata/device/kernels/dataflow/routing_metadata_select.hpp` (`order_key`, `top_k_of`) | - | `cb_scores` | scratch | - |

Helpers considered and rejected for the `raw_api` rows:

| Operation | Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|-----------|----------------------------------|
| `exchange_partials_block` | `ttnn.Mcast` | `mcast/host/mcast.hpp:98-133` | an intra-group all-to-all of G <= 3 adjacent members: G separate 2-point unicasts, not one sender to a rectangle; members of different groups would receive foreign partials |
| `exchange_counts` (gather half), `land_barrier` (incs) | `ttnn.Mcast` | `mcast/host/mcast.hpp:110-129` | many-to-one gather; the helper is one-to-many |
| `push_rows_block` | `ttnn.Mcast`; `local_copy_helpers_dataflow.hpp` | `mcast/host/mcast.hpp:133`; `kernel_lib/local_copy_helpers_dataflow.hpp` | each row goes to one data-dependent remote owner at a data-dependent offset (a scatter); the local-copy helper copies within one core |
| `select_block` / `assign_rows_block` | none in `kernel_lib/` | - | scalar integer routing logic on the DM RISC; no helper covers top-k on scores or a counting sort |
| `write_combine_*` | `tilize_helpers_dataflow.hpp` | `kernel_lib/tilize_helpers_dataflow.hpp` | the writes are column-0-only tiles and row segments built from scalar state, not a tilize of a CB |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| logits | Where(keep, logits, -inf) | `cb_partials` sums: All (cols >= 16 garbage, replaced) | `cb_mask_keep`: All | None |
| exp | Sub | `cb_masked_logits`: All | `cb_row_max`: Col0 (REDUCE_ROW out) | Col |
| normalize | Mul | `cb_exps`: All | `cb_recip_sum`: Col0 (REDUCE_ROW out) | Col |

## Precision

> **Perf 1:** the softmax now runs entirely in exact fp32 on the SFPU (it was FPU max / sub / sum / mul reading
> tf32 sources). Max |score - fp64| falls from 2.3e-3 to 1.8e-7 in the bench. Selection may differ from the old path
> on near-ties only. The exp is the accurate fp32 exp under either `math_approx_mode`.

| Step | Precision | Note |
|------|-----------|------|
| partial logits | fp32 DEST accumulation over `ht` tiles; W read through SrcA (tf32), x bf16 exact | stock runs K = 24 in one fp32 DEST pass |
| logits sum | SFPU fp32 adds in fixed contributor order, unpacked with UnpackToDestFp32 (exact) | identical bytes in on every member -> bit-identical logits, softmax and selection on every member |
| mask | `Where` select (NaN-safe): columns >= E become -inf regardless of W tile padding or uninitialized faces 1/3 | stock added a -inf mask, which is not NaN-safe |
| softmax | stock router_front_end helper path (reduce max, exp, sum, precise fp32 recip, mul) | the measured-accuracy baseline: weight error 3.3e-4..2.6e-3, reroutes 0.07..0.39% |
| selection | fp32 bits of `p`, order key, ties to the lower expert | as routing_metadata |
| rows | bf16 untilize / tilize through fp32 DEST: exact | bit-exact copies |

`fp32_dest_acc_en=False` is refused (SUPPORTED `[True]`); `math_fidelity` and `math_approx_mode` are
honored (`EXP_APPROX` define).

## Host contract

| Item | Decision |
|------|----------|
| `validate()` order | (1) ValueError for malformed calls: rank 4 with dim 1 == 1, last dim == W's dim 0, W rank 2 with `E <= 32`, `spare_tile_rows` an `int` (not bool) in 0..3, same device; (2) axes cell (`dtype`, `layout`, `alignment` via `tag_alignment`, `fp32_dest_acc_en`, `in_placement`, `out_buffer_type`) against SUPPORTED, then EXCLUSIONS; (3) contract refusals (UnsupportedAxisValue): `H != 768`, `E != 8`, `T_pad` outside 1024..4096, `B > 1 and S % 32 != 0`, grid != 11x10, `rps > 4`, `nomic_moe_dispatch.plan(R)` refusal |
| `in_placement` classification | the same vocabulary as `eval/golden_tests/.../axes.py`: interleaved L1 / DRAM, `nd_norm` iff x's memory config equals `nomic_residual_layer_norm.input_memory_config(shape, device)`, else `"other"` (refused) |
| L1 check | after allocating the four outputs (host allocation, no dispatch): read `ttnn.get_memory_view(device, L1)`, take the lowest allocated L1 address (it includes `dispatched`), require `ttnn.get_allocator_base_address(device, L1) + circular_buffer_bytes(...) <= lowest`; otherwise deallocate the outputs and raise UnsupportedAxisValue |
| `circular_buffer_bytes(input_shape, device, spare_tile_rows)` | closed form of `l1_ledger.md` Total for the nd_norm plan of `input_shape` (production placement) |
| entry point | `validate()` first; no tensor transformation, no readback; one `ttnn.generic_op([x, W, dispatched, combine_index, combine_weights, expert_counts], descriptor)` |
| trace safety | no runtime argument depends on values; only shapes, placements and addresses |
| INPUT_TAGGERS | `tag_alignment(inputs, axes)`: `"tile_aligned"` if `inputs[0][-2] % 32 == 0` else `"ragged"` |
| SUPPORTED (Phase 0) | `dtype [bfloat16]`, `layout [TILE]`, `alignment [tile_aligned]`, `fp32_dest_acc_en [True]`, `in_placement [nd_norm]`, `out_buffer_type [L1]` |
| EXCLUSIONS | `[]` |

Structural impossibilities: none beyond feature_spec's INVALID.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| members disagree on the selection | every member computes destination rows independently; a mismatch scatters different slices of a token to different rows | byte-identical `cb_partials` on every member (own partial reaches its own slot through the same send path), fixed add order, same kernel; garbage columns are masked with `Where` before any reduction |
| NaN from padding | W tile columns 8..31 and partial faces 1/3 are untrusted; a NaN in the row max poisons every probability | `Where(keep, logits, -inf)` select, not add |
| alias `cb_router_weight` / `cb_gathered_rows` | a push could land on a core whose W region is still being written or read | senders barrier before signalling (all destinations hold the data once any core sees the signal); a push requires the table, which requires every group's selection, which requires every slice's signal and every member's matmul (own matmul precedes own counts) |
| remote writes into CBs before the owner reserves them | peers write `cb_partials`, `table`, `cb_gathered_rows` at fixed addresses | each of those CBs is used exactly once per call and is not aliased with anything live at that time; the owner publishes only after the matching semaphore |
| ND shard core order | addressing assumes x shard i on the i-th core row-major of its grid and output shard s on core `s % 110` | derive both maps from the tensors' specs on the host and pass core NoC coordinates as RT args; regression test with non-uniform routing checks every row |
| program cache vs routing | data-dependent work must not reach the hash | all routing quantities live in L1; RT args depend only on shapes (golden `test_routing_change_warm_cache`) |
| skewed routing | one expert taking almost all slots concentrates the PUSH sources on nothing (sources are token owners) and the destinations on nothing (owners are fixed by row index); it only changes which rows are pad rows | owner and pusher load are routing-independent; regression tests cover dominant / dead experts |
| zero-count experts | empty region, `start_{e+1} = start_e` | scan handles 0; pad runs of an empty region are empty |
| landing count | coordinator must know exactly how many incs arrive | every token core's two DM RISCs inc exactly once each, pushed rows or not |
| deadlock through role interleaving | a role core (coordinator / sender) can be a token core | each core produces everything others wait on (its W slice, its counts, its pushes) before it blocks on the matching collective |
