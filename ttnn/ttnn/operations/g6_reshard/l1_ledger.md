# L1 Ledger: g6_reshard

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `tile_row`
(`block_rows_tiles`), `tile_col` (`block_cols_tiles`) — `op_design.md` § Blocking Model.

The op allocates **zero CBs**. Its only L1 is the shard of `out` (a tensor buffer, not a CB), which
counts against the 1347776 B the model leaves free per core (contract r3 §8: out's shards and every
scratch buffer of the op come out of it). in0's shard is the model's and is listed for reference only.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `out_shard` (tensor buffer of out, not a CB; one per target core) | `out_shard_rows_tiles * out_shard_cols_tiles` (P: 8 × 4 = 32; O1: 8 × 5 = 40) | the whole shard — it **is** the result; `core_rows_tiles * core_cols_tiles` valid pages, the rest unused tail slots | `{tile_row: spans → out_shard_rows_tiles (issuer blocks of block_rows_tiles each land in it), tile_col: spans → out_shard_cols_tiles (= block_cols_tiles)}` | Bfp8_b, the input's stored format; dataflow-only payload that never passes DEST (`fp32_dest_acc_en` not applicable) | issuer kernels of the same core (NCRISC/NOC0, BRISC/NOC1) via NoC reads, on disjoint row ranges | the next program of the model (no kernel of this program reads it) | allocated before the program, written once during it, returned | none: it is `out`, which must be a new tensor (rule 1, R8) and cannot alias in0's shard; no other buffer exists to share with |
| `in0_shard` (reference only — the model's tensor, **not** counted) | `in_shard_rows_tiles * in_shard_cols_tiles` (P: 7 × 3 = 21) | whole shard, resident before the call | `{tile_row: spans → in_shard_rows_tiles, tile_col: spans → in_shard_cols_tiles}` | Bfp8_b | the model's previous program | issuer kernels of up to 4 target cores (P), by NoC read | the whole program | not the op's allocation; read only, never written |

Inventory check (Rule 1, Rule 3): the minimum inventory is one buffer — the destination itself.
A staging CB on the target (pack-into-destination taken to its limit: reads land in `out_shard`),
an intermediate on a third core, or a DRAM/interleaved intermediate are each a buffer that should
not exist; all are `rejected` regimes in `op_design.md`. No budget solve, safety fraction or
blocking search exists, and none is needed.

### Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `out_shard_rows_tiles` | `≤ tensor_rows_tiles` in effect (a larger shard only adds unused slots) | `memory_config` shard height / 32; validated multiple of 32 |
| `out_shard_cols_tiles` | `≤ tensor_cols_tiles` in effect | `memory_config` shard width / 32; validated multiple of 32 |
| `page_bytes` | 1088 (bfloat8_b tile) in Phase 0; generally `tile_size(dtype)` | `input_tensor.buffer_aligned_page_size()`, asserted equal to out's and a multiple of `ttnn.get_l1_alignment()` |
| `out_shard_bytes = out_shard_rows_tiles * out_shard_cols_tiles * page_bytes` | `≤ L1_FREE_BYTES = 1347776` | host refusal (`RuntimeError`, BB-05) before allocation; P 34816 B, O1 43520 B, I0/I3/I5 34816 B |

### Total per-core footprint

    L1_op(core) = out_shard_rows_tiles * out_shard_cols_tiles * page_bytes      (target cores)
                = 0                                                              (all other cores)

It scales with the out shard shape only — fixed by `memory_config`, not by any op knob.
`block_rows_tiles`, `block_cols_tiles`, `issuers_per_core` and `runs_in_flight` scale **nothing**
in L1 (there is no staging), so every block knob is free to turn. P: 34816 B of 1347776 B (2.6 %).

## Data-movement budget

Chosen split: PULL — target cores read, two issuers each. Production pair P; bfloat8_b tiles of
1088 B; the tier crossed is source L1 → target L1 over the NoC.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| in0 | 0 | resident in L1 before the call (the model's tensor); read in place on its source cores, never staged | 1880 tiles = 2045440 B read once by their target cores (1024 runs over 256 source→target core pairs); 168 tiles = 182784 B copied by NoC loopback on cores that hold both ends |
| out | 0 | written once, straight into its own L1 shard | none beyond in0's line (the same bytes) |

Totals per tier (P): DRAM 0 B; cross-core NoC 2045440 B; core-local (loopback) 182784 B; every
tile moved exactly once. Other suite placements: I0 0 B cross-core / 2228224 B local; I3
1949696 / 278528; I5 2228224 / 0; O1 2181440 / 46784.

> Cheapest-traffic split considered: PULL, PUSH and BOTH-ENDS tie — each moves every tile across
> the NoC once (2045440 B cross-core, 0 DRAM in P). Implemented: PULL, which is the cheapest.
> PUSH and BOTH-ENDS are `deferred` regime rows because they land the same bytes on the same
> target cores; their only difference is the issue count per RISC (8 in PULL vs ≤ 4.7 in
> BOTH-ENDS), reachable with the same piece walk run from the source side.
