# Operation Design: nomic_sdpa

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + softmax + matmul) |
| Goal | The bidirectional 12-head attention of nomic-embed-text-v2-moe on one Blackhole p300c (11x10 grid), in one `ttnn.generic_op` dispatch, faster than stock SDPA (perf focus 8x512: at most 58 us acceptance, 62.6 us model gate; 2x512 / 4x512 never slower than 24.5 / 38.2 us). |
| Math | `out[b, 0, i, 64h + d] = sum_{j < L_b} softmax_j(scale * q[b,h,i,:] . k[b,h,j,:]) * v[b,h,j,d]`, `L_b = key_lengths[b]` or `S` |
| Algorithm | Exact two-pass softmax per block of query tile-rows against the **whole resident key range** of one (b, h) pair: `scores = Q K^T`; `m = rowmax(scores)`; `P = exp(scale (scores - m))` in place; `[O | l] = P [V | 1]` (the row sum `l` comes out of the same matmul through a ones column appended to V); `out = O * (1 / l)`. No online rescaling (S <= 512). |
| Mode | Derivative (incumbent: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/`, streaming bf16-DEST path) |
| References | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp` (exp on pack thread :625, row-sum by packer L1-acc :647-660, normalize :812-920); `sdpa_program_factory.cpp:82,1365-1385`; `models/experimental/nomic_embed_text_v2_moe/tt/attention.py:653-663` (the model's call); `tt/model_config.py:52-71,228-243` (precision notes); host patterns: `ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/nomic_moe_combine/nomic_moe_combine.py:364-586`, `nomic_moe_combine_program_descriptor.py:91-128`, live-L1 check `nomic_moe_route_dispatch/nomic_moe_route_dispatch.py:447-454,561-581`; golden suite `eval/golden_tests/nomic_sdpa/` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `query` | ttnn.Tensor | yes | (B, 12, S, 64) bf16 TILE interleaved, L1 or DRAM | — | RT (address, common RT arg) |
| `key` | ttnn.Tensor | yes | same shape / dtype / layout / buffer type as `query` | — | RT (address) |
| `value` | ttnn.Tensor | yes | same as `query` | — | RT (address) |
| `scale` | float | no (kw) | finite, `> 0` (SupportRefusal otherwise: the max-subtraction identity `max(scale*s) = scale*max(s)` needs `scale > 0`) | 1.0 | CT (`scale == 1.0` selects the chain without the scale element; the bits otherwise) |
| `key_lengths` | list[int] \| None | no (kw) | `len == B`, each `1 <= L_b <= S` (ValueError otherwise) | None → `mask_mode = "none"` | RT (B common RT args, rewritten every call); Phase 0: given → `UnsupportedAxisValue(mask_mode=key_padding)` |
| `memory_config` | ttnn.MemoryConfig \| None | no (kw) | interleaved, L1 or DRAM | `ttnn.L1_MEMORY_CONFIG` | cache key (output allocation) |
| `compute_kernel_config` | ttnn.ComputeConfigDescriptor \| None | no (kw) | fidelity HiFi2/HiFi3/HiFi4, `fp32_dest_acc_en=False`; LoFi and fp32 DEST refused (SupportRefusal) | `default_compute_kernel_config()` = HiFi2, `fp32_dest_acc_en=False`, `math_approx_mode=True`, `dst_full_sync_en=False` | CT (descriptor) |

Module exports: `nomic_sdpa`, `default_compute_kernel_config()`, `circular_buffer_bytes(batch, seqlen)`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`.
Package: `ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/nomic_sdpa/` (imported as `ttnn.operations.nomic_sdpa`) (`__init__.py`, `nomic_sdpa.py`, `nomic_sdpa_program_descriptor.py`, `kernels/nomic_sdpa_reader.cpp`, `kernels/nomic_sdpa_compute.cpp`, `kernels/nomic_sdpa_writer.cpp`). io tensor order to `generic_op`: `[query, key, value, output]`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | q, k, v each (B, 12, S, 64); identical shapes (ValueError otherwise); rank 4 (ValueError) |
| Domain | heads == 12, head dim == 64, S <= 512, `B * ceil(S/32) * 32 <= 4096` (SupportRefusal otherwise) |
| Dtype | bfloat16 |
| Layout | TILE. Tile (b, h, st, dt) is page `((b*12 + h)*S_t + st)*2 + dt`, `S_t = ceil(S/32)`; padding rows `s >= S` are zero (producer contract, `nomic_qkv_rotary_heads`) |
| Memory | interleaved, all three in the same buffer type (L1 or DRAM; SupportRefusal if they differ) |
| Head-dim order | q and k share one order (the dot product is order-invariant); v is in reference order, so output columns are in reference order |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, 1, S, 768); tile (b, st, c) is page `(b*S_t + st)*24 + c`; head h owns `c = 2h, 2h+1` |
| Dtype | bfloat16 |
| Layout | TILE (padding rows of a ragged S must be finite — later refinement; tile-aligned in Phase 0) |
| Memory | interleaved, `memory_config` (default `ttnn.L1_MEMORY_CONFIG`); buffer type independent of the inputs' |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below is downstream of this section.

### Axes

The op's index space is `(b, h, query_row, key, head_dim)`. Tile units: a **query row** is one 32-row tile-row; a **key tile** is 32 keys; head dim is 2 tiles.

`b`, `h` and the query tile-row are flattened into one axis, **`query_row`** `g = (b*12 + h)*S_t + st` (0 <= g < `tensor_query_rows = 12*B*S_t`). They are treated identically (all independent) and are contiguous in q's page order (`page = 2g + dt`), so the flattening is honest. The **pair** `p = g / S_t` (= `b*12 + h`) is the unit that shares K and V.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `query_row` (b, h, query tile-row flattened) | **independent** — each query row's softmax and output depend only on its own scores | `q_block_rows` | `ceil(tensor_query_rows / num_cores)` (= the largest per-core assignment: 14 at 8x512, 7 at 4x512, 4 at 2x512; never above `Q_BLOCK_ROWS_CAP = 14`, the domain bound `ceil(1536/110)`) — one block = the whole of a core's rows inside one pair | host `plan(batch, seqlen)`: one constant → compute/reader/writer CT arg; every CB page count derives from it | contiguous balanced ranges over `num_cores = min(110, tensor_query_rows)` cores: core `c` owns `[core_row_start[c], core_row_start[c+1])` (first `rem` cores one row more, `ttnn.split_work_to_cores` semantics). The table is a **common-RT-arg table indexed by core** (`core_index = y*11 + x`), so the assignment is a host knob | knob-turn (any contiguous table, e.g. pair-aligned groups, runs unchanged) |
| `key` (key tiles of a pair) | **dependent** — the softmax max, the sum and the P·V contraction all span every key | `key_block_tiles` | `S_t` (whole key axis resident: K, V and one score row-block hold all keys) | host `plan`: = `S_t`, CT arg | not assigned across cores (whole on every core) — the residency predicate `S <= 512` makes the exact two-pass softmax the cheapest schedule; splitting needs a cross-core combine | scheme-change (regime `key-split + combine`, deferred) |
| `head_dim` (2 tiles) | **dependent** in Q·K^T (contracted), **independent** in P·V (output columns) | `head_dim_tiles` | 2 (whole) | validate() pins head dim 64; host constant 2, CT arg | not assigned across cores — 2 tiles; a split would re-read Q/K and add a combine for two tiles of work | none (fixed by the op contract) |
| `pv_cols` (P·V output tiles incl. the row-sum column) | derived from `head_dim` | `pv_out_tiles = head_dim_tiles + 1` | 3 (2 value tiles + 1 ones/sum tile) | derived from `head_dim_tiles` (never restated) | follows `query_row` | none |
| `pair` (b·12 + h) — the K/V sharing unit, not a separate block axis | **reuse-shared** along `query_row` — K and V do not vary with the query row, so every core holding rows of a pair needs the identical 64 KiB K + 64 KiB V (S=512) | none (a core loads one pair's K/V per **segment**) | — | — | implied by the `query_row` table: a core touches `<= ceil(rows/S_t) + 1` pairs (max 2 at the perf shapes, 4 at 32x128) | scheme-change (regime `row-split + pair multicast`, deferred) |

**Segment**: the maximal run of a core's rows inside one pair. A segment is the K/V residency unit; it is cut into `num_blocks_this_segment = ceil(segment_rows / q_block_rows)` blocks (== 1 everywhere in the domain because a core's rows never exceed `q_block_rows`; the loop stays general). A block's runtime row extent `block_rows <= q_block_rows` is a **runtime** value (ragged per core); CB capacities are **nominal** in `q_block_rows`.

#### Stages and their own data (re-running the table on intermediates)

| Stage | Data it produces | Axes of that data | Blocking / assignment | Why |
|-------|------------------|-------------------|----------------------|-----|
| `qk_block` | scores `block_rows x key_block_tiles` | query_row (indep.), key (indep. for the matmul output) | on the core that owns the rows; whole key range | all score columns of a row are needed by `row_max_block` on the same core |
| `row_max_block` | `block_rows` max tiles | query_row | same core, whole block | per-row; no cross-core data |
| `probs_block` | P (in place of scores) | query_row, key | same core | elementwise |
| `pv_block` | `[O | l]` `block_rows x 3` | query_row, pv_cols | same core | contraction over key stays local (key is whole) |
| `normalize_block` | out `block_rows x 2` | query_row, head_dim | same core | elementwise |

No stage computes on one designated core while others wait: there is no collector/coordinator role in the built regime; every core runs the full pipeline on its own rows.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_query` | `q_depth` | 1 (one nominal block) | Q is consumed by the **first** phase (`qk_block`); the next block's Q loads while this block runs max/exp/P·V/normalize. Depth 2 buys nothing extra. |
| `cb_key_t` | `kv_depth` | 1 (one pair) | K is popped after the segment's last `qk_block`; the next segment's K loads during max/exp/P·V/normalize of the current one. |
| `cb_value_aug` | `kv_depth` (same knob) | 1 (one pair) | V is popped after the segment's last `pv_block`; the next V loads during the next segment's `qk_block`/max/exp. |
| `cb_out` | `out_depth` (in blocks) | 1 block (`q_block_rows` rows) | compute never waits on the writer: a whole block's output fits; the writer drains it during the next block. |
| `cb_scores`, `cb_row_stats`, `cb_pv_out` | — (compute-internal, one block) | 1 block | intermediates between sequential helpers on the same TRISCs: they cannot pipeline, so they hold exactly one block. |
| `cb_reduce_scaler` | — | 1 tile | constant |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Domain predicate `12*B*S_t <= 1536` over 110 cores | `q_block_rows <= 14` | `q_block_rows = min(ceil(tensor_query_rows/num_cores), Q_BLOCK_ROWS_CAP=14)`; a larger per-core assignment (impossible in-domain) would run more blocks per segment | CBs sized for 14 rows overflow L1 budget assumptions |
| Residency (K, V and a score row-block hold the whole key range) | `key_block_tiles = S_t <= 16` | validate(): `S <= 512` else SupportRefusal (stock SDPA) | K/V/score CBs exceed the L1 budget |
| DEST capacity (bf16 DEST, half-sync: 8 tiles; full-sync: 16) — `dest_tiles` derived from the compute config | matmul subblock `h*w <= dest_tiles` (helper-internal; derived, not designed) | `qk`: `w` = largest divisor of `key_block_tiles` `<= dest_tiles`, `h` = largest divisor of `block_rows` with `h*w <= dest_tiles`; `pv`: `w = pv_out_tiles = 3`, `h` = largest divisor of `block_rows` `<= floor(dest_tiles/3)` | subblock overflows DEST: silent corruption |
| `matmul_block` shape legality: `in0_num_subblocks*h == block_rows`, `in1_num_subblocks*w == N` (`matmul_block_helpers.hpp:139-179`) | `w` must divide N | divisor rule above (for a prime N such as 13 in the ragged refinement, `w = 1` or `last_in1_subblock_w_valid` narrowing with N padded to a multiple of `w` in the CB, `:153-158`) | mismatched wait/pop counts → hang |
| CB ring-wrap invariant (block consumers index tiles linearly from `rd_ptr`) | every CB consumed as a multi-tile block (`cb_query`, `cb_key_t`, `cb_value_aug`, `cb_scores`, `cb_row_stats`, `cb_pv_out`) | **nominal quanta**: each block advances each such CB by its full capacity; a ragged block narrows only the helper shape and closes the ring with a pad push / pad pop of the remainder (see Circular Buffers) | a block straddling `fifo_limit` reads past the CB end: wrong values, no error |
| `cb_out` per-row quantum | rows of 2 tiles | capacity `q_block_rows*2` is a multiple of 2 | n/a (always closes) |
| Ones column of `cb_value_aug` | must hold bf16 1.0 (0x3F80) in slots `st*3 + 2` for every `st < key_block_tiles` | written once per depth slot on the reader's first reserve of that slot (ring returns to the same base every segment because pushes are nominal) | wrong row sums |
| Approximate exp input domain (`Exp<Approx::Fast>` → `exp_tile<true>` with `InputClamping::ClampToNegative`, `exp.h:80-85`) | none on extent; arguments are `scale*(s - m) <= 0` | clamping keeps very negative arguments at 0 | (clamp is on) |
| Common runtime args per kernel | `core_row_start` table `num_cores + 1 <= 111` + 4 addresses + `B <= 32` key lengths `<= 147` words | within the per-kernel runtime-arg limit | descriptor build error |
| Core index source | `core_index = get_absolute_logical_y()*11 + get_absolute_logical_x()` (`tt_metal/hw/inc/api/compute/common.h:122`; precedent `nomic_moe_combine_compute.cpp:75`) | program core range = the first `num_cores` cores in row-major order | a core reads another core's rows |

### Regimes

Named memory boundary for "minimum": **the io-tensor memory** (remote L1 banks over the NoC when `buffer = l1`; DRAM when `buffer = dram`). Minimum = q, k, v each cross it once and the output once: `4 * 12*B*S_t*2` tiles (24 MiB at 8x512).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **row-split, per-core K/V read** (two-pass exact softmax, whole key range resident) | **built** — the only built regime; covers the whole domain | domain: `S <= 512`, heads 12, head dim 64, `12*B*S_t <= 1536`, grid 11x10, CBs fit below live L1 | `q_block_rows x key_block_tiles` scores; Q block `q_block_rows x 2`; K, V whole pair (`key_block_tiles x 2`, `key_block_tiles x 3`) | q once, out once (minimum); K and V once **per segment**: `segments / pairs` = 2.00x at 8x512, 3.15x at 4x512, 4.88x at 2x512, 1.15x at 32x128 (exact counts in `l1_ledger.md`). Above minimum because a pair's rows straddle core boundaries; each extra read is one 64 KiB K + 64 KiB V | per block, once: six phase inits (matmul, reduce, sub/exp chain, matmul, recip chain, mul chain) and their CB handshakes; pipeline fill/drain of each phase. Phase 0 block = the whole per-core segment, so these are paid once per segment (1-2 per core at the perf shapes). Larger blocks are impossible (block = whole assignment) |
| **row-split + pair multicast** (cores sharing a pair receive K/V from one reader) | **built in Refinement 2** (multicast on top of the row split; groups = cores holding the pair at a segment `< kv_depth`, sender = the run's first core, no handshake, ring slot `p % kv_depth` via phantom slots; kernel_lib `SenderPipe` / `ReceiverPipe` from per-core RT rectangles, since the host `Mcast` cannot express irregular overlapping groups; see `l1_ledger.md` § Measured data movement, Refinement 2). Original deferral reason, falsified by measurement:  under L1 placement (every perf shape) each core's NoC **ingress** (Q + K + V of its segments, which bounds its load time) is byte-identical with and without multicast; multicast removes only source-bank egress (36 → 24 MiB at 8x512, spread over 110 banks: <= 0.33 MiB a bank) and adds a sender rendezvous per group. Its real win is DRAM placement (DRAM reads 36 → 24 MiB at 8x512), which the model selects only when q/k/v do not fit L1 — no perf case. Reachable: cores sharing a pair are **consecutive** in `core_row_start`, CB layouts and addresses are identical on every core (landing address = the reserved `cb_key_t`/`cb_value_aug` slot), so a group is a consecutive run, sender = its first core, `McastConfig(irregular_receiver_set_mode=ChainUnicast)` | same domain; selected by `buffer == dram` once built | same as built | K, V cross the io boundary once (minimum); cross-core: each receiver still ingests K/V once per segment | adds one handshake per group per segment |
| **pair-per-core** (whole (b, h) pairs per core) | **rejected** — superseded by the built regime: same minimum traffic only through leaving cores idle: makespan 16 rows/core vs 14 at 8x512 (96 of 110 cores busy), 16 vs 7 at 4x512 (48 busy), 16 vs 4 at 2x512 (24 busy) | — | `S_t x S_t` | minimum | — |
| **key-split + cross-core combine** (flash-decoding: a pair's keys split over cores, partial `(m, l, O)` combined) | **deferred** — applies only where `tensor_query_rows < 110` (1x128, 4x64, 2x37: 48-96 rows), where the built regime already gives every busy core one row (sub-microsecond compute) and the op time is dispatch + one K/V load, which a key split does not shorten (it adds a combine round). Reachable: the per-segment pipeline is unchanged; a combine stage (unicast of `block_rows x 4` partial tiles + semaphore) would append after `pv_block` | `tensor_query_rows < 110` once built | `q_block_rows x key_chunk` | Q re-read per key chunk; partials cross cores | amortizes the combine round trip |
| **online softmax over key chunks** (the incumbent's streaming flash) | **rejected** — superseded by the built regime inside `S <= 512`: K/V resident makes the exact two-pass cheaper (no running-max correction exps, no `alpha` rescale of partial O and sums). `S > 512` is refused to stock SDPA | — | `q_chunk x k_chunk` | same as built | — |
| **three-pass, scores not resident** (recompute Q·K^T for the exp pass) | **rejected** (dead end) — superseded by the built regime: recomputes the first matmul (one third of the FPU work) and re-reads K, to save a score block that fits (448 KiB at the largest shape) | — | — | K twice per segment | — |

No regime selection beyond the domain predicate: one regime is built.

### Traffic ranking

Counts are crossings of the io-tensor memory boundary (exact for the INPUTS shapes, from the balanced `core_row_start` table):

| Rank | Candidate split | q | k | v | out | Total at 8x512 (each tensor 6 MiB) | Occupancy (rows a core, makespan) |
|------|-----------------|---|---|---|-----|-----------------------|----------------|
| 1 | pair-per-core | 1 | 1 | 1 | 1 | 24 MiB | 16 (8x512), 16 (4x512), 16 (2x512) |
| 1 | row-split + pair multicast (deferred) | 1 | 1 (+ chain/mcast deliveries) | 1 (+) | 1 | 24 MiB at the source; receiver ingress = row-split's | 14 / 7 / 4 |
| 3 | **row-split, per-core K/V read (built)** | 1 | 2.00 | 2.00 | 1 | 36 MiB | 14 / 7 / 4 |
| 4 | key-split + combine | per key chunk | 1 | 1 | 1 + partials | > 36 MiB | under-filled shapes only |
| — | head-dim split | — | — | — | — | rejected: 2 tiles, adds a combine | — |

Chosen: **row-split, per-core K/V read**. The cheaper-traffic **pair-per-core** split is traded for occupancy explicitly: at 8x512 it costs 16/14 = +14% makespan for 12 MiB less NoC traffic that, under L1 placement, is spread across 110 banks and overlapped with compute (each core ingests at most 56 KiB Q + 2 x 128 KiB K/V); at 4x512 / 2x512 it would idle 62 / 86 cores. The multicast successor recovers the minimum at the source and is deferred for the reason in its row. The block extent does not enter this ranking: every operand is fetched once per segment regardless of `q_block_rows` (K/V are held across the segment's blocks), so bytes are constant in the extent and the extent ranking below decides it.

### Block schedule

```cpp
// per core: segments are derived from [core_row_start[c], core_row_start[c+1]) and S_t
for (uint32_t segment_idx = 0; segment_idx < num_segments_this_core; ++segment_idx) {
    load_kv_block(segment_idx);                   // reader: K (K^T order) + V (+ ones column) of the pair
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_segment; ++block_idx) {
        load_query_block(block_idx);              // reader
        qk_block(block_idx);                      // scores = Q K^T
        row_max_block(block_idx);                 // m = rowmax(scores)
        probs_block(block_idx);                   // P = exp(scale * (scores - m)), in place
        pv_block(block_idx);                      // [O | l] = P [V | 1]
        normalize_block(block_idx);               // out = O * (1 / l)
        store_block(block_idx);                   // writer: concatenated-heads pages
    }
    release_kv(segment_idx);                      // K popped after the last qk_block, V after the last pv_block
}
```

| Operation | Block shape (extent knobs) | Resident across it | Intended frequency of fixed costs |
|-----------|---------------------------|--------------------|-----------------------------------|
| boot | — | — | once a core: `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_query, cb_key_t, cb_scores)`; writer fills `cb_reduce_scaler` once; reader writes the ones column once per `cb_value_aug` depth slot |
| `load_kv_block` | K: `key_block_tiles x head_dim_tiles`; V: `key_block_tiles x pv_out_tiles` | — | once a segment: all K page reads issued, one barrier, one push; same for V |
| `load_query_block` | `block_rows x head_dim_tiles` (nominal `q_block_rows x 2`) | K, V | once a block: all Q page reads, one barrier, one push |
| `qk_block` | M = `block_rows`, K = `head_dim_tiles`, N = `key_block_tiles` | K (retained), V | one matmul init a block |
| `row_max_block` | `block_rows x key_block_tiles` → `block_rows` | scores (not popped), K, V | one reduce init a block |
| `probs_block` | `block_rows x key_block_tiles`, in place | V | one chain init a block (sub-bcast + [scale] + exp, one DEST window per tile) |
| `pv_block` | M = `block_rows`, K = `key_block_tiles`, N = `pv_out_tiles` | V (retained) | one matmul init a block |
| `normalize_block` | `block_rows` (recip of the sum column), then `block_rows x head_dim_tiles` | — | two chain inits a block |
| `store_block` | `block_rows x head_dim_tiles` | — | once a block: page writes issued per row, one write barrier a block |
| `release_kv` | — | — | once a segment |

All CB page formats are Float16_b (bf16 DEST), so no data-format reconfig is issued in the main loop: every helper call runs with its reconfig mode NONE / Disabled after boot.

**Stall-shadow check.**

| Waiting stage | Waits on | Independent work that can fill the window | Decision |
|---------------|----------|-------------------------------------------|----------|
| compute, first `qk_block` of the core | the first segment's K (64 KiB) | none on the compute side (every phase needs Q·K^T) | exposed once a core; overlap lamp L1 (key-chunked K delivery) |
| compute, `probs_block` of row r | `row_max_block` of row r (needs every key) | `qk_block` of rows r+1.. (FPU) — rows are independent, so exp of row group i may run on the SFPU (pack thread) while the FPU computes Q·K^T of row group i+1; permission comes from row independence, no floating-point reorder inside a row | Phase 0 runs the phases block-sequentially; the interleave is perf lamp L2 (the incumbent's `exp_packthread_tile`, `compute_streaming.hpp:625`) |
| compute, `pv_block` | V of the segment | V loads during `qk_block`/`row_max_block`/`probs_block` | already hidden by order (K before V) |
| reader, next segment | free space in `cb_key_t` / `cb_value_aug` (depth 1) | nothing on the reader | depth lamp L6 |
| writer | `cb_out` rows | nothing | — |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap: whole-K block before `qk_block` | the first segment's K load (64 KiB) is exposed; at 2x512 (4 rows a core) it is a visible fraction of the call | reader pushes K in key-chunks (e.g. 4 tiles) and `qk_block` runs N-chunked so compute starts on the first chunk |
| L2 overlap: SFPU exp serialized with FPU phases | exp is ~40% of the speed of light; on the math thread it adds to FPU time instead of overlapping | row-group interleave with the exp on the pack thread (`exp_packthread_tile`, `exp.h:103-133`) while Q·K^T of the next row group runs **Built in Refinement 3**: the partner is P·V of the previous group first (Q·K^T windows pack as much as the probs window, so the pack thread binds), scores and P in separate 2-group rings, and a DEST fence before the math-thread recip (`changelog.md`) |
| L3 exp form | helper `Exp<Approx::Fast>` uses `ClampToNegative`, per-face `iterations=8` | the incumbent's form: `exp_tile<true, false, InputClamping::None, 32>(dst, VectorMode::None)` + `PackRelu::Zero` on the output spec (`compute_streaming.hpp:617,625`); gate on all three precision profiles |
| L4 row-sum route | the ones column adds 50% to the P·V matmul (16 extra tile-muls a row at S=512) | `reduce<SUM, REDUCE_ROW, …, ReduceAlgorithm::AccumulateViaAdd>` over P (catalog `row_reduce_accumulate`: 2.9x over per-tile reduce at W=32) — fewer FPU passes, but a separate phase, a second unpack of P, and a sum from untruncated P (loses the HiFi2 truncation cancellation, see Precision); gate on precision |
| L5 normalize route | two eltwise chains + `cb_pv_out` round trip per block | normalize in `matmul_block`'s `PostComputeFn` on DEST (recip of the sum tile, SFPU multiply of the two O tiles — the sum tile already holds `l` in every column) and pack straight to `cb_out` laid out `block_rows x 3`, the writer skipping the third tile |
| L6 K/V depth | depth 1 hides the next segment's K/V only if the current segment is long enough; a 1-row first segment (frequent at 2x512 / 4x512) is ~1 us of compute vs ~1.5 us of K load | `kv_depth = 2` (+160 KiB at S_t = 16; 994 KiB total, fits 1055 KiB) |
| L7 core assignment | balanced contiguous ranges maximize occupancy but split pairs (4.88x K/V reads at 2x512) | pair-aligned groups (each core one pair; same makespan 4 rows at 2x512, 110 segments vs 117) — same kernels, different `core_row_start` table |
| L8 recip scope | recip over a full tile when only column 0 of the sum is needed by the bcast | `VectorMode::C` (catalog `sfpu_tile_scope`: 3.84x for column 0); small (one recip a row vs 16 exps) |
| L9 L1 bank contention | cores sharing a pair issue the same 64 K/V page reads in the same order at the same time | rotate each core's page-issue start by its position within the pair group |
| Grid synchronization | none: the built regime has no collective, barrier or multicast across cores | — |

Catalog entries that set the knobs (`ttnn/ttnn/operations/examples/master.md`): `compute_block_size` (one pass over the block instead of per tile-row: 1.65x, ~1.6 us fixed per extra pass → `q_block_rows` = whole per-core assignment); `matmul_output_subblock` (8-tile subblocks 1.46x vs 1x1 → bf16 DEST, subblocks derived to fill 8); `double_buffer`, `trid_double_issue` (many reads in flight per barrier → whole K / V / Q per barrier); `shared_input_reuse`, `mcast_topology` (the deferred multicast row); `row_reduce_accumulate` (L4); `sfpu_tile_scope` (L8); `compute_fusion` (sub and exp in one DEST window before the SFPU consumer). `read_coalescing` does not apply (interleaved pages are not contiguous).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| q, k, v (L1 banks or DRAM) → reader | bf16 tiles, 2048 B pages | `TensorAccessor` page reads (`noc_async_read_page`) on NoC0; one barrier per operand block | Q pages `2g + dt` for the block's rows; K pages `(p*S_t + st)*2 + dt` written to `cb_key_t` slot `dt*key_block_tiles + st` (K^T tile order for the matmul's in1); V page → `cb_value_aug` slot `st*3 + dt` |
| ones column | bf16 1.0 tiles | reader: `fill_l1_range` one tile (`l1_helpers.hpp:100-101`), then self-aimed local copies (`local_copy_helpers_dataflow.hpp:74-117`) into slots `st*3 + 2` | once per depth slot of `cb_value_aug` |
| reduce scaler | bf16 1.0 packed | writer: `calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>` (`reduce_helpers_dataflow.hpp:109-131`) | once a core |
| compute pipeline | bf16 tiles in CBs, bf16 DEST | helpers (API Mapping) | scores never leave L1; `cb_scores` holds one core's block only |
| compute → writer | `cb_out`, 2 tiles a row | writer `TensorAccessor` page writes on NoC1 to page `(b*S_t + st)*24 + 2h + dt`; one write barrier a block | concatenated-heads layout written directly |
| Pair multicast contract (built, Refinement 2; realized with the lower-level pipes, not `ttnn.Mcast`) | — | `ttnn.Mcast(device, McastConfig(noc, handshake=True, irregular_receiver_set_mode=ChainUnicast), receivers, receiver_group_size, McastExplicitFixedSenderConfig(...))`; sender `send(slot, slot, bytes)` of the reserved `cb_key_t` / `cb_value_aug` slot, receivers `receive_and_forward` then `push_back` (pattern `nomic_qkv_rotary_heads_reader.cpp:134-155`) | not built in Phase 0 |

TARGET axes against the scheme:

| TARGET axis value | Phase 0 | Cost class | Mechanism |
|-------------------|---------|------------|-----------|
| `buffer = l1` | supported | — | `TensorAccessor` over L1-interleaved buffers |
| `buffer = dram` | supported | — | same kernels, DRAM accessor (multicast successor deferred, see Regimes) |
| `mask_mode = key_padding` | refinement | knob-turn (structure-preserving) | per segment `key_tiles_valid = ceil(L_b/32)` (runtime, from the common-RT `key_lengths`) narrows N of `qk_block`, the reduce width, the exp width and K of `pv_block`; the reader reads only `key_tiles_valid` K/V tile-rows (pushes nominal); a `mask_boundary_block` adds a reader-made `cb_key_mask` tile (0 in columns `< L_b mod 32`, most-negative finite bf16 elsewhere) to the boundary score column **before** `row_max_block` (realization: in the `qk_block` `PostComputeFn` on DEST, or a chain with real CB credit). Masked keys then weigh exactly 0 in max, exp and both P·V columns |
| `alignment = ragged` | refinement | knob-turn | the same mechanism with `L_b = S`; `S_t = ceil(S/32)`; padded query rows read zero q rows → uniform softmax → finite |

Structural impossibilities: none beyond `feature_spec.py` (`INVALID = []`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_rows <= q_block_rows` consecutive query tile-rows of one pair |
| Grid | 11x10 (validate: SupportRefusal otherwise); program cores = first `num_cores = min(110, tensor_query_rows)` cores row-major |
| Per-core work | rows `[core_row_start[c], core_row_start[c+1])`; `base, rem = divmod(tensor_query_rows, num_cores)`; core c gets `base + (c < rem)` |
| Segments | from row `g`: pair `p = g / S_t`, segment end `min(core end, (p+1)*S_t)`; `b = p / 12`, `h = p % 12`, `st = g - p*S_t` |
| Remainder | ragged per-core row counts handled as runtime `block_rows`; tile geometry is per-image `S_t = ceil(S/32)` (never `floor(B*S/32)`) |
| Per-core values | none in per-core runtime args: `core_row_start` is a common-RT table indexed by `core_index`; all else is derived |

| Shape | S_t | pairs | rows | rows/core (max) | `q_block_rows` | segments (K/V loads) | max segments a core |
|-------|-----|-------|------|-----------------|----------------|----------------------|--------------------|
| 8x512 | 16 | 96 | 1536 | 14 | 14 | 192 | 2 |
| 4x512 | 16 | 48 | 768 | 7 | 7 | 151 | 2 |
| 2x512 | 16 | 24 | 384 | 4 | 4 | 117 | 2 |
| 32x128 | 4 | 384 | 1536 | 14 | 14 | 440 | 4 |
| 16x256 | 8 | 192 | 1536 | 14 | 14 | 275 | 3 |
| 8x384 | 12 | 96 | 1152 | 11 | 11 | 192 | 2 |
| 1x128 | 4 | 12 | 48 (48 cores) | 1 | 1 | 48 | 1 |

Regime-pinned tests: not required (one regime). The acceptance test covers `num_cores < 110` (1x32, 1x128), multi-segment cores (32x128, 16x256) and the perf shapes.

## Circular Buffers

Notation: `Q = q_block_rows`, `N = key_block_tiles`, `HD = head_dim_tiles = 2`, `PV = pv_out_tiles = 3`, tile = 2048 B (Float16_b).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_query` | 0 | 2048 | `Q*HD*q_depth` | spans query_row (block), head_dim; streams over segments | Float16_b | reader | compute | per block; popped by `qk_block` (+ pad) |
| `cb_key_t` | 1 | 2048 | `N*HD*kv_depth` | spans key, head_dim (K^T order `dt*N + st`); streams over segments | Float16_b | reader | compute | per segment |
| `cb_value_aug` | 2 | 2048 | `N*PV*kv_depth` | spans key, pv_cols (`st*3 + c`, c = 2 is the ones column) | Float16_b | reader | compute | per segment; ones slots permanent |
| `cb_scores` | 3 | 2048 | `Q*N` | spans query_row, key; holds scores, then P in place | Float16_b | compute | compute | `qk_block` → `pv_block` |
| `cb_row_stats` | 4 | 2048 | `Q` | spans query_row; holds the row max (`row_max_block` → `probs_block`), then `1/l` (`normalize_block`) — two disjoint lifetimes, one buffer | Float16_b | compute | compute | per block, two uses |
| `cb_pv_out` | 5 | 2048 | `Q*PV` | spans query_row, pv_cols | Float16_b | compute | compute | `pv_block` → `normalize_block` |
| `cb_out` | 6 | 2048 | `Q*HD*out_depth` | spans query_row, head_dim; writer drains per row | Float16_b | compute | writer | per block |
| `cb_reduce_scaler` | 7 | 2048 | 1 | constant (MAX/REDUCE_ROW scaler 1.0) | Float16_b | writer | compute | whole kernel |

CB index is any distinct slot; names are binding.

**Ring discipline (all block-consumed CBs).** Each block advances each of `cb_query`, `cb_scores`, `cb_row_stats` (per use), `cb_pv_out` by its **full capacity**; `cb_key_t` / `cb_value_aug` by their full capacity per segment. A ragged block (`block_rows < Q`) passes `block_rows` to the helpers and closes the ring with a pad: the producer reserves+pushes the remaining pages without data, the consumer pops them after the helper. For `cb_scores` the exact sequence (pointers return to base every block, P lands in exactly the slots the scores occupied):

| Step | Op on `cb_scores` | rd / wr after (capacity C = Q*N, live r = block_rows*N) |
|------|-------------------|-------------------------------------------------------|
| 1 | `qk_block` pushes r; compute pad-pushes C − r | rd 0, wr 0 (wrapped), occupancy C |
| 2 | `row_max_block` waits r (WaitUpfrontNoPop) | unchanged |
| 3 | `probs_block` per tile: wait, compute, **pop 1, then reserve 1**, pack, push (chain order: `chain.inl:2805-2820`) → in place | rd r, wr r |
| 4 | pad pop C − r | rd 0 |
| 5 | pad push C − r | wr 0, occupancy C |
| 6 | `pv_block` waits r, pops r (in0 WaitAndPopPerKBlock) | rd r |
| 7 | pad pop C − r | rd 0 |

When `r == C` (the full block — every core at 8x512 with a single segment) the pads are empty.

**Sync counts.**

| CB | Producer pushes / block (or segment) | Consumer waits / pops |
|----|--------------------------------------|-----------------------|
| `cb_query` | reader: `Q*HD` (block_rows*HD with data) | `qk_block` waits/pops `block_rows*HD`; pad pop `(Q - block_rows)*HD` |
| `cb_key_t` | reader: `N*HD` per segment | `qk_block` waits `N*HD` (retain); `release_kv` pops `N*HD` |
| `cb_value_aug` | reader: `N*PV` per segment | `pv_block` waits `N*PV` (retain); `release_kv` pops `N*PV` |
| `cb_scores` | 2 x C per block (table above) | 2 x C per block |
| `cb_row_stats` | `row_max_block` `block_rows` + pad; recip `block_rows` + pad = 2Q | `probs_block` waits/pops `block_rows` + pad; `normalize_block` waits/pops `block_rows` + pad = 2Q |
| `cb_pv_out` | `pv_block` `block_rows*PV` + pad = `Q*PV` | caller waits `block_rows*PV` (strided reads), pops `Q*PV` |
| `cb_out` | `normalize_block` 2 a row, `block_rows` rows | writer waits/pops 2 a row |
| `cb_reduce_scaler` | 1 once | waited once, never popped |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | boot | — | `compute_kernel_hw_startup<SrcOrder::Reverse>` | — | — | writer has pushed `cb_reduce_scaler` (1); compute waits it once |
| 1 | `load_kv_block` (reader) | K `N x HD`, V `N x PV` | dataflow (TensorAccessor) | q/k/v tensors | `cb_key_t` (N*HD), `cb_value_aug` (N*PV) | pushed, retained by compute across the segment |
| 2 | `load_query_block` (reader) | `block_rows x HD` | dataflow | query tensor | `cb_query` (Q*HD nominal) | pushed |
| 3 | `qk_block` | M=`block_rows`, K=HD, N=N | `matmul_block` (transpose) | `cb_query` (block, popped), `cb_key_t` (retained) | `cb_scores` (block_rows*N, TileRowMajor) + pad | scores resident |
| 4 | `row_max_block` | `block_rows x N` → `block_rows` | `reduce<MAX, REDUCE_ROW>` | `cb_scores` (WaitUpfrontNoPop), `cb_reduce_scaler` | `cb_row_stats` (block_rows) + pad | scores still resident |
| 5 | `probs_block` | grid `block_rows x N` | `eltwise_chain` | `cb_scores` (per tile wait/pop), `cb_row_stats` (Col mapping, upfront wait, pop at end) | `cb_scores` (per tile reserve/push, in place) + pads | P resident in the score slots |
| 6 | `pv_block` | M=`block_rows`, K=N, N=PV | `matmul_block` | `cb_scores` (P, popped), `cb_value_aug` (retained) | `cb_pv_out` (block_rows*PV, TileRowMajor) + pad | `[O | l]` resident |
| 7 | `normalize_block` (a) recip | grid `block_rows x 1`, column 2 of `cb_pv_out` (Strided, base 2, row stride PV) | `eltwise_chain` (`CopyTile` → `Recip`) | `cb_pv_out` (caller-waited, not popped) | `cb_row_stats` (block_rows) + pad | — |
| 7 | `normalize_block` (b) mul | grid `block_rows x HD`, columns 0..1 of `cb_pv_out` (Strided, base 0, row stride PV) | `eltwise_chain` (`BinaryFpu<Mul>` bcast Col) | `cb_pv_out` (caller-managed), `cb_row_stats` (Col mapping) | `cb_out` (2 a row) | caller pops `cb_pv_out` Q*PV |
| 8 | `store_block` (writer) | `block_rows x HD` | dataflow | `cb_out` | output tensor | — |
| 9 | `release_kv` | — | `cb_pop_front` | `cb_key_t` after the segment's last `qk_block`; `cb_value_aug` after its last `pv_block` | — | slots free for the reader |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60`; Reverse mandatory for matmul `matmul_block_helpers.hpp:100-106`; precedent mixing matmul + eltwise_chain after Reverse boot: `nomic_qkv_rotary_heads_compute.cpp:198` | `(cb_query, cb_key_t, cb_scores)` | — | — | — |
| `qk_block` | helper | `matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:366-398` | `transpose=true, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, in0_policy=WaitAndPopPerKBlock, in1_policy=WaitAndRetainOnLastBlock (:77), reconfig=NONE (:125)`; `MatmulBlockShape::of(block_rows/h, N/w, h, w, HD, 1)` (`:169-178`); interm = `cb_scores` placeholder (`:334-341`) | `cb_query`, `cb_key_t` | `cb_scores` | M = `block_rows` (≤ `q_block_rows`), N = `key_block_tiles`; `h, w` derived (mechanism caps) |
| `row_max_block` | helper | `reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:654-681`; policy enum `:121` | `<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_scores, cb_reduce_scaler, cb_row_stats, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::NONE>`; `ReduceInputBlockShape::of(block_rows, N)` (`:300-309`) | `cb_scores`, `cb_reduce_scaler` | `cb_row_stats` | `block_rows`, `N` |
| `probs_block` | helper | `eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:620`; `BinaryFpu` `:568`, `BroadcastDim::Col` `:330-335`, `PackTile` `:588`, `output()` `:442-450`; `Exp` `eltwise/unary/math.hpp:22`, `math.inl:29-32`; `MulUnary` `eltwise/unary/scalar.hpp:32` | `IterationShape::grid(block_rows, N)`; `BinaryFpu<Sub, input(cb_scores, PerTile, PerTile, Block), input(cb_row_stats, BroadcastDim::Col, Upfront, AtEnd, Col)>` → [`MulUnary{scale}` only when `scale != 1`] → `Exp<Approx::Fast>` (Exact when `math_approx_mode=False`) → `PackTile<output(cb_scores, PerTile, PerTile)>`; data-format reconfig Disabled | `cb_scores`, `cb_row_stats` | `cb_scores` (in place) | `block_rows`, `N` |
| `pv_block` | helper | `matmul_block` | `matmul_block_helpers.hpp:366-398` | `transpose=false, packer_l1_acc=false, Out, TileRowMajor, Short, in0=WaitAndPopPerKBlock, in1=WaitAndRetainOnLastBlock, reconfig=NONE`; `MatmulBlockShape::of(block_rows/h, 1, h, PV, N, 1)` | `cb_scores` (P), `cb_value_aug` | `cb_pv_out` | M = `block_rows`, K = `key_block_tiles`, N = `pv_out_tiles` |
| `normalize_block` (a) | helper | `eltwise_chain` | `chain.hpp:620`; `CopyTile` `:554`; `TileAddressing::Strided` `:286-291`; `Recip` `math.hpp:34`, `math.inl:55-58` | `IterationShape::grid(block_rows, 1)`; `CopyTile<input(cb_pv_out, None, None, Block, Disabled, Strided)>{…, StridedTileRange{2, PV}}` → `Recip<>` → `PackTile<output(cb_row_stats)>` | `cb_pv_out` | `cb_row_stats` | `block_rows` |
| `normalize_block` (b) | helper | `eltwise_chain` | `chain.hpp:568,588,620` | `IterationShape::grid(block_rows, HD)`; `BinaryFpu<Mul, input(cb_pv_out, None, None, Block, Disabled, Strided){base 0, row stride PV}, input(cb_row_stats, BroadcastDim::Col, Upfront, AtEnd, Col)>` → `PackTile<output(cb_out, PerTile, PerTile)>` | `cb_pv_out`, `cb_row_stats` | `cb_out` | `block_rows`, `head_dim_tiles` |
| `release_kv`, ring pads, `cb_pv_out` wait/pop | raw_api | `cb_wait_front` / `cb_pop_front` / `cb_reserve_back` / `cb_push_back` | `tt_metal/hw/inc/api/compute/cb_api.h` | counts in Circular Buffers | — | — | nominal capacities |
| reduce scaler (writer) | helper | `calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:109-131` | pool-type-aware overload, bf16 packed | — | `cb_reduce_scaler` | — |
| ones column (reader) | helper | `fill_l1_range<2>` then `read_with_state` local copies | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-101`; `local_copy_helpers_dataflow.hpp:74-117` | value 0x3F80 (bf16 1.0) | — | `cb_value_aug` ones slots | `key_block_tiles` copies |
| K/V/Q reads, output writes | raw_api | `TensorAccessor` + `noc_async_read_page` / `noc_async_write_page` + barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | one barrier per operand block | — | — | per-segment / per-block counts |

Helpers considered and rejected for the `raw_api` rows:

| raw_api row | Candidate helper | Mismatch (file:line) | Reason |
|-------------|------------------|----------------------|--------|
| ring pads / `release_kv` | `matmul_block` `InputPolicy` (`matmul_block_helpers.hpp:64-77`) | `WaitAndRetainOnLastBlock` "skip the pop on the last so an outer loop can reuse the data" (`:69-71`) | K/V are retained across a segment's blocks by the helper; the release after the last block is by design the caller's pop. Pads have no data, so no helper applies |
| K/V/Q/out page I/O | `mcast` `McastArgs` (`mcast/kernel/mcast_args.hpp:310-330`) | sends a raw L1 range from one core to receivers (`pipes/mcast_pipe.hpp:91-92`) | the built regime has no shared delivery; each core reads its own pages. The multicast is the deferred regime |
| boot | — | — | engine-wide init is the caller's by contract (`matmul_block_helpers.hpp:100-106`) |

Not used: row-sum via `reduce<SUM>` (replaced by the ones column of `pv_block`, lamp L4); `bias_add_helpers`, `reblock_untilize_helpers`, `tilize`/`untilize` (no bias, all TILE).

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `probs_block` | Sub | `cb_scores` All | `cb_row_stats` Col0 (REDUCE_ROW output) | Col |
| `normalize_block` (b) | Mul | `cb_pv_out` columns 0..1: All | `cb_row_stats` = `1/l`: All (the ones-column matmul replicates `l` across all 32 columns; Col0 is what the bcast reads) | Col |

## Support Contract and Validation

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `{"alignment": tag_alignment}`; `tag_alignment(inputs, axes) -> "tile_aligned" if inputs[0][-2] % 32 == 0 else "ragged"` |
| `SUPPORTED` (Phase 0) | `dtype: [ttnn.bfloat16]`, `buffer: ["l1", "dram"]`, `mask_mode: ["none"]`, `alignment: ["tile_aligned"]` |
| `EXCLUSIONS` | `[]` |
| `INVALID` | not declared in the op |
| Axis names | exactly TARGET's: `dtype`, `buffer` (from `query.memory_config().buffer_type`), `mask_mode` (`"none"` if `key_lengths is None` else `"key_padding"`, derived in validate()), `alignment` |

validate() order (first line of `nomic_sdpa`; malformed calls raise ValueError before any support check so the error is stable across refinements):

| # | Check | Raises |
|---|-------|--------|
| 1 | rank 4 for q, k, v; q, k, v shapes identical; `key_lengths` length == B; every length in `1..S` | ValueError |
| 2 | axes cell vs SUPPORTED / EXCLUSIONS | `UnsupportedAxisValue` / `ExcludedCell` |
| 3 | layout TILE, interleaved, q/k/v same buffer type; output `memory_config` interleaved | SupportRefusal |
| 4 | heads == 12, head dim == 64, `S <= 512`, `B*S_t*32 <= 4096` | SupportRefusal |
| 5 | grid 11x10 | SupportRefusal |
| 6 | `scale` finite and `> 0` | SupportRefusal |
| 7 | compute config: LoFi or `fp32_dest_acc_en=True` | SupportRefusal |
| 8 | `circular_buffer_bytes(B, S)` fits below the lowest live L1 allocation | SupportRefusal (checked before allocating the output and again after; on the second failure the output is deallocated first) |

## Host Call Path

| Item | Decision |
|------|----------|
| Cache | `_CALL_CACHE[key] -> _CallState(plan, descriptor, binding)`; `key = (B, S, q buffer_type, repr(memory_config), mask_mode, scale bits, compute-config key (fidelity, fp32_dest_acc_en, math_approx_mode, dst_full_sync_en), device.id())` |
| First call of a key | `plan(B, S)` (S_t, `tensor_query_rows`, `num_cores`, `core_row_start`, `q_block_rows`, subblock widths, CB specs), build the ProgramDescriptor once |
| Every call | validate() → live-L1 check → `ttnn.allocate_tensor_on_device` (output) → `bind_call`: write q/k/v addresses into the reader's common RT args, the output address into the writer's, `key_lengths` into reader + compute common RT args → `ttnn.generic_op([q, k, v, out], descriptor)` |
| Per-core runtime args | none (generic_op re-sends per-core args every call); per-core values come from the `core_row_start` common table and `core_index` |
| Live-L1 check | `ttnn.get_memory_view(device, ttnn.BufferType.L1)`: lowest allocated address; `get_allocator_base_address(device, L1) + circular_buffer_bytes(B, S) <= lowest` (`nomic_moe_route_dispatch.py:447-454`) |
| `circular_buffer_bytes(batch, seqlen)` | sum of the CB specs from the single `circular_buffer_specs(plan)` function the descriptor allocates from |
| `default_compute_kernel_config()` | `ttnn.ComputeConfigDescriptor(math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=True, dst_full_sync_en=False)`; `None` resolves through it |
| Target | <= 100 us host time a call (no descriptor rebuild after the first call of a key) |

## Precision

| Item | Decision |
|------|----------|
| Bar | `PROFILE_TOLERANCES` (`eval/golden_tests/nomic_sdpa/helpers.py`): real (0.99965, 0.0265), flat (0.99985, 0.0175), peaky (0.99925, 0.0375) — measured before any speed tuning |
| Default | HiFi2, bf16 DEST, approximate exp: the incumbent's numeric class at every step (bf16 scores, bf16 `s - m`, approximate exp, bf16 P, bf16-DEST P·V accumulation) |
| Gains over the incumbent | no online rescaling (no `exp(m_old - m_new)` corrections, no rescaled partial sums); the row sum `l` is computed by the same HiFi2 matmul from the same bf16 P as `O`, so the low-bit truncation of P (the model's measured HiFi2 bias, `model_config.py` SDPA notes) cancels in `O / l` instead of shrinking the output |
| **Phase 0 as built (verifier note)** | The implementer changed the default to **fp32 DEST** (HiFi2, approximate exp, every CB still bf16): the bf16-DEST default below missed the bar at S = 512. validate() honors fp32 DEST, refuses LoFi, and refuses bf16 DEST + approximate exp at S > 384 (`BF16_DEST_APPROX_EXP_MAX_SEQLEN`; measured to miss the bar at S = 512). bf16 DEST + exact exp is honored (it meets the bar at every measured shape). The rows below are the original plan |
| Caller configs | HiFi3 / HiFi4 honored (more exact, slower); `math_approx_mode=False` honored (accurate exp); LoFi refused (4% small outputs in the model); fp32 DEST refused in Phase 0 (all intermediate pages are 16-bit, matching bf16 DEST per the ledger's page-format audit) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| CB ring wrap on ragged blocks | `block_rows` differs per core (13/14 at 8x512, 1..4 at 2x512); helpers index a multi-tile block linearly from `rd_ptr` | nominal-quanta pad discipline (Circular Buffers); test shapes with ragged per-core rows (all perf shapes) |
| In-place exp deadlock / misplacement | `cb_scores` is full (occupancy C) when `probs_block` starts | per-tile pop precedes reserve in the chain (`chain.inl:2805-2820`); pad push after `qk_block` makes the rotation exactly in place; never use `ReservePolicy::None`/`PushPolicy::None` for the in-place pass (no credit, `l1-footprint-discipline.md` Rule 3.2) |
| SFPU state shared between exp and recip | `recip_tile_init` overwrites exp's programmable constants (`exp.h:52-59`) | each chain issues its element inits per call (once a block): `probs_block` re-inits exp after the previous block's recip |
| K^T tile order | `matmul_block` reads in1 as `[k][n]`; K arrives `[st][dt]` | reader writes K page `(st, dt)` to slot `dt*N + st`; the helper's `transpose=true` transposes within each tile |
| Ones column lost | `cb_value_aug` pushes must stay nominal so slots `st*3+2` are reused | reader writes the ones once per depth slot; compute never writes `cb_value_aug` |
| HiFi2 / bf16 DEST error over the bar | the bar's margin over the incumbent is small | measure all three profiles first; remedy order: HiFi3 (fidelity knob) → fp32 DEST with Float32 `cb_pv_out` / `cb_row_stats` (+112 KiB at Q = 14; `cb_scores` stays bf16, it is the matmul's bf16 in0) |
| SrcOrder after a Reverse boot | eltwise / reduce helpers follow a matmul boot | each helper issues its own short init (InitMode::Short / chain-owned init); precedent `nomic_qkv_rotary_heads_compute.cpp:198` |
| L1 budget | 834 KiB at 8x512 vs 1055 KiB free | per-call live-allocator check; `circular_buffer_bytes` is the single source |
| `core_index` vs program core range | cores beyond `num_cores` must not run (1x128 uses 48 cores) | program core range = first `num_cores` cores row-major; table indexed by `y*11 + x` |
| key_lengths compiled in | must be read per call | common RT args rewritten by `bind_call`; `mask_mode` (not the lengths) is in the cache key |
