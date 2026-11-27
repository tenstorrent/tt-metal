# L1 Ledger: g6_reshard

Schema: `.claude/references/l1-footprint-discipline.md`. Blocking: `op_design.md` → Blocking Model
(axes `tile_row`, `tile_col`; regime `PULL`).

## Inventory

The op allocates **no circular buffer and no scratch**. Each run is a NoC read that lands at its final
address in out's shard (Rule 3, pattern 1: pack into the destination), and in0 is read in place on the core
that holds it. The rows below are the two tensor buffers the kernels touch, so the footprint has a written
justification. Only `out_shard` is the op's.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `out_shard` (tensor buffer, not a CB; allocated by the op) | `out_shard_row_tiles · out_shard_col_tiles` (the allocated shard, tail slots included) | `block_height_tiles · block_width_tiles` (= capacity except on a tail block row/col, where the placement leaves unused slots) | `{tile_row: spans → block_height_tiles (issuer bands are disjoint row slices of it), tile_col: spans → block_width_tiles}` | Bfp8_b (in0's dtype, 1088 B/tile; dataflow-only payload that never passes DEST, so no DEST-width rule applies) | the `num_issuers` pull kernels of this core, each writing its own disjoint row band (no FIFO, no handshake) | the next program (no consumer inside this op) | whole program; outlives it as the returned tensor | Nothing. It *is* the output, so it cannot alias `in0` (rule 1: a new tensor, in0 unchanged — I0 has equal placements and still needs it), and no other buffer exists to share with. The gap between capacity and live set is the shard padding the caller's placement fixes, not this op's choice. |
| `in0_shard` (the model's tensor; read-only; **not counted**) | `in_shard_row_tiles · in_shard_col_tiles` | — (not the op's allocation; read in place by remote pullers) | `{tile_row: streams → the pullers read row runs of it, tile_col: streams → runs ≤ in_shard_col_tiles}` | Bfp8_b | the program before this one | the pull kernels of every out-block core whose block overlaps it (remote NoC reads; ≤ 4 cores in production) | before and after the program | Not the op's buffer, so there is nothing to share; per the contract it lies outside the 1347776 B. |

No dataflow, compute or scratch CB rows: every candidate was eliminated before any budget was considered.

| Candidate buffer | Eliminated by |
|------------------|---------------|
| staging CB on the target (read into a CB, then copy to out) | pattern 1: the read lands in out's slot directly |
| staging CB on the source (copy shard into a CB, then send) | the source shard is already resident and addressable over the NoC |
| CB backed on out's shard (`cb_descriptor_from_sharded_tensor`) | the base address is a common RT arg; nothing would use its FIFO |
| per-run descriptor buffer (host-built run list in L1) | the run walk is derived on device from CT geometry |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `out_shard_row_tiles`, `out_shard_col_tiles` | out shard shape in tiles (block extents, caller-pinned) | `out_shard_row_tiles · out_shard_col_tiles · tile_stride_bytes ≤ 1347776` | host check before allocation, RuntimeError otherwise (BB-05). Production 8·4·1088 = 34816 B; O1 8·5·1088 = 43520 B |
| `block_height_tiles`, `block_width_tiles` | actual tiles of this core's out block | `≤ out_shard_row_tiles`, `≤ out_shard_col_tiles` | `min` clip against `tensor_row_tiles` / `tensor_col_tiles` |
| `band_height_tiles` | rows per issuer | `ceil(block_height_tiles / num_issuers)` | derived |
| `num_issuers` | pull kernels per core | `1 ≤ num_issuers ≤ 2` (data-movement RISCs) | host constant; contributes no L1 |
| `tile_stride_bytes` | aligned page size | 1088 (bfloat8_b) | `input_tensor.buffer_aligned_page_size()`, equal to out's, multiple of the L1 alignment (host assert) |
| `in_shard_row_tiles`, `in_shard_col_tiles` | in0 shard shape in tiles | any multiple-of-32 shard | the model's allocation; not in the op's footprint |

## Total per-core footprint

`L1_op(core) = out_shard_row_tiles · out_shard_col_tiles · tile_stride_bytes` on each out-block core, 0 on
every other core. It scales with the caller's out shard only. No term scales with `num_issuers`, the
transaction depth or the input placement. Production: 34816 B against the 1347776 B budget (2.6 %).

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `in0` | 0 | resident in L1 before the program; read in place by the pullers, every tile exactly once | production: 1880 tiles = 2,045,440 B core-to-core in 1024 runs over 256 core pairs; 168 tiles = 182,784 B by NoC loopback (source core = target core) |
| `out` | 0 | allocated in L1 in its final placement; every tile written once, by the read that moves it | none beyond the reads above (the read *is* the write) |

Per tier, production: DRAM 0 B; core-to-core NoC 2,045,440 B; core-local loopback 182,784 B; total L1
written 2,228,224 B (= one copy of the tensor). Other suite pairs: I0 0 B cross / 2,228,224 B loopback; I3
1,949,696 B cross; I5 2,228,224 B cross; O1 2,181,440 B cross.

> Cheapest-traffic split considered: by out block (`PULL`), by in block (`PUSH`) and the union of both
> ends (`HYBRID`) — all three move the minimum, delta 0 B on every tier. Implemented: `PULL`, which is
> therefore the cheapest-traffic split. `PUSH` and `HYBRID` are `deferred` rows in the regime table
> because `PULL` already serves every placement pair they would serve with identical bytes; they differ
> only in which end issues (production critical core 16 runs pull vs 14 push).
