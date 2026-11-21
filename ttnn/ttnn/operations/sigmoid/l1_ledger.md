# L1 Ledger: sigmoid

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model (`batch`, `h_tiles`, `w_tiles` folded into `page`; `neo`; `sync_block`; element-in-tile). Worked values are for `quasar`, Phase 0 (bfloat16, `fp32_dest_acc_en=True`, plan (2,4,2)).

**Inventory.** There are two rings and no other buffer. The whole expression (copy, clamp, `exp(-x)`, `+1`, reciprocal) stays in DEST in one chain pass, so there is no intermediate (Rule 3, pattern 4: fold into DEST). No constant or scaler buffer exists, because the `+1` and the Newton constant are SFPU immediates/registers. Each ring is the minimum the reader→compute→writer pipeline needs, and no budget predicate, safety fraction or search is introduced.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `BUFFER_DEPTH * block_tiles * C` = 2·4·4 = 32 | `block_tiles * C` = 16 being computed (one sync block per NEO) + `block_tiles * C` = 16 in flight from the readers | `{page: streams → window of BUFFER_DEPTH·block_tiles per NEO; batch/h_tiles/w_tiles: streams (folded into page); neo: spans → C; sync_block: spans → block_tiles; element-in-tile: spans → 1 whole tile (the page)}` | bfloat16 (the input tensor's dtype). The ring is a dataflow payload of the tensor's own pages. With `fp32_dest_acc_en=True` the 16-bit page is not an accumulation truncation, because no value has passed through DEST before it | reader | compute | whole program | **none.** It is live concurrently with `cb_output_tiles` (pipelined reader / compute / writer), its formats are tied to two different tensors, and in-place (Rule 3.2) is barred because the producer is a dataflow kernel and the output must reach the writer through a ring of its own (one producer / one consumer per ring). Capacity above the computing set is the double buffer (`BUFFER_DEPTH = 2`) |
| `cb_output_tiles` | `BUFFER_DEPTH * block_tiles * C` = 32 | `block_tiles * C` = 16 being packed + 16 draining to the writer | `{page: streams → window of BUFFER_DEPTH·block_tiles per NEO; batch/h_tiles/w_tiles: streams (folded into page); neo: spans → C; sync_block: spans → block_tiles; element-in-tile: spans → 1 whole tile}` | bfloat16 (the output tensor's dtype: an output ABI requirement). It is packed from fp32 DEST, and that pack is the op's one rounding to the output grid | compute | writer | whole program | **none.** It is concurrent with `cb_input_tiles` (pipelined). Packing straight into this destination (Rule 3.1) is already done, and no scratch sits in between. Capacity above the packing set is the double buffer |

## Symbol table

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `BUFFER_DEPTH` | 2 (Phase 0). Any integer ≥ 1 with the ring cap below | host constant in `sigmoid_program_artifacts.py` |
| `block_tiles` | ≤ `chain_max_block_v` (4 at fp32 half-sync DEST, 8 at 16-bit). Phase 0 4 | host `DEST_TILES[fp32_dest_acc_en]`. The chain clamps (`chain.inl:3339-3351`) |
| `C` | ∈ {1, 2, 4}. Phase 0 4 | `THREAD_PLAN` host constant, target page limits |
| `page_size` | bfloat16 2,048 B (Phase 0), float32 4,096 B | `tensor.buffer_page_size()` |
| ring cap | `BUFFER_DEPTH * block_tiles * C * page_size ≤ 1,048,560 B` per ring | target page (DFB compute-side limit). Host assert |
| per-core budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`, `l1_reserve_bytes = 0` = 3,963,328 B on quasar | run-time query (probe_001) |
| `num_tiles`, `core_tiles` | unbounded (tensor size) | **appear in no capacity expression**: both rings stream over `page` |

## Total per-core footprint

```
footprint = 2 * BUFFER_DEPTH * block_tiles * C * page_size
          = 2 * 2 * 4 * 4 * 2048 B = 131,072 B   (quasar, Phase 0: 3.3 % of 3,963,328 B)
```

Every term scales with the knobs `BUFFER_DEPTH`, `block_tiles` and `C` and with the dtype's `page_size`. No term scales with the tensor. The worst TARGET corner (float32, 16-bit DEST excluded by R1, so `block_tiles` = 4) is `2·2·4·4·4096 = 262,144 B`, so the op never needs a fits-in-L1 predicate.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 (an L1-interleaved input: 1 crossing of the L1-bank NoC instead) | each page is read once, into exactly one NEO's slice, and consumed from L1 once. The element-wise op needs nothing twice | none |
| `output` | 1 (L1 banks for an L1 input, R6) | each page is packed once and written once | none |

Total per tier, for `num_tiles` pages: DRAM (or L1 banks) `2 · num_tiles · page_size` bytes. For example `(1, 80, 8400)` bfloat16: 789 pages → 1,615,872 B read + 1,615,872 B written. Cross-core: 0 B. Core-local: each page is unpacked once and packed once.

> Cheapest-traffic split considered: flat `page` axis in contiguous runs (dealt in whole rounds of C). It is the minimum: each tensor crosses the expensive tier once, with no cross-core traffic. Implemented: the same split. The implemented split is the cheapest. The only cheaper placement is R-shard (zero NoC crossings for a pre-sharded tensor), deferred because `TARGET["sharding"]` holds `interleaved` only, so no cell needs it.
