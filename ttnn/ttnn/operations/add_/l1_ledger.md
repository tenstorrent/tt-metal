# L1 Ledger: add_

Regime: `sharded_resident` (Phase 0). Schema and audits: `.claude/references/l1-footprint-discipline.md`.
Block axes (from `op_design.md`, Blocking Model): `rows` (`block_row_tiles`), `cols` (`block_col_tiles`) and the core tile walk dealt to compute threads (`neo_block_tiles`, × `compute_threads`).

All three rings are **borrowed** from caller-owned shards (`borrowed_from`). They allocate no L1, so their capacity is the caller's storage seen through a ring, not op footprint.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_a_shard` | `shard_tiles = shard_row_tiles · tensor_col_tiles` (= `neo_block_tiles · compute_threads`), borrowed from `a`: 0 op-owned bytes | `core_tiles_k` (≤ `shard_tiles`): the whole operand is resident because it is the caller's tensor | `{rows: spans → block_row_tiles, cols: spans → block_col_tiles, core tile walk: spans → neo_block_tiles per compute thread × compute_threads}` | `a.dtype` (bfloat16): the caller's bytes | reader (credit only) | compute | whole program | **shares its bytes with `cb_result_shard`**: an in-place transform (Rule 3.2) with real credit on both rings. They stay two rings because one ring has one producer and one consumer: these bytes are consumed by compute's unpack and produced by compute's pack. |
| `cb_b_shard` | `shard_tiles`, borrowed from `b`: 0 op-owned bytes | `core_tiles_k` | `{rows: spans → block_row_tiles, cols: spans → block_col_tiles, core tile walk: spans → neo_block_tiles per compute thread × compute_threads}` | `b.dtype` (bfloat16) | reader (credit only) | compute | whole program | none: it is a different tensor's storage, read-only by R5, live for the whole program alongside both `a` rings |
| `cb_result_shard` | `shard_tiles`, borrowed from `a`: 0 op-owned bytes | `core_tiles_k` | `{rows: spans → block_row_tiles, cols: spans → block_col_tiles, core tile walk: spans → neo_block_tiles per compute thread × compute_threads}` | `a.dtype` (bfloat16) while `fp32_dest_acc_en` is on. Audit-2 "under" answered by an output ABI requirement: the page is the caller's tensor `a`, whose dtype the result must keep (R4), and the spec requires exactly one rounding to the operand dtype at store (DEFINITION, R6). No phase reads it back, so no accumulation is truncated. | compute | writer (credit only) | whole program | shares its bytes with `cb_a_shard` (see that row) |

Audit notes:
- Capacity vs live set: capacity = `shard_tiles` ≥ live set `core_tiles_k`. The gap exists only on a partly filled or empty shard (R8). The mechanism reason is that a borrowed ring must cover the whole shard because it cannot wrap at an offset, and it costs no op-owned L1.
- Disjoint lifetimes: none. All three rings are live for the whole program.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `shard_row_tiles` | `shard_spec.shape[0] / 32` | the caller's shard spec; TILE layout makes the shard height a multiple of 32 (validate step 6) |
| `tensor_col_tiles` | `ceil(W / 32)` = `shard_spec.shape[1] / 32` | height sharding holds the full padded width (validate step 6) |
| `shard_tiles` | `shard_row_tiles · tensor_col_tiles`; `shard_tiles · tile_bytes ≤` the host's accepted ring extent | the caller allocated it in L1 (≤ `L1_budget`). The ring extent is checked by the host validator (`dataflow_buffer.cpp:1600-1665`). Phase 0 max is 802,816 B. |
| `core_tiles_k` | `clamp(tensor_row_tiles − k·shard_row_tiles, 0, shard_row_tiles) · tensor_col_tiles ≤ shard_tiles` | shard map (R8) |
| `compute_threads` (C) | ∈ {1, 2, 4}, `C ∣ shard_tiles`, `C ≤ max_k core_tiles_k` | the clamp of `THREAD_PLAN[1]` (Mechanism caps) |
| `neo_block_tiles` | `shard_tiles / C ≤ 65,535` (uint16 credit count) | `shard_tiles ≤ L1_budget / tile_bytes` (≤ 1,935 bf16 tiles on quasar) |
| `L1_budget` | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` (3,963,328 B − reserve on quasar, probe 2026-11-21) | run-time query; used by **no** Phase 0 predicate because the op-owned footprint is 0 |

## Total per-core footprint

`op_owned_L1 = 0` bytes for every knob value (three borrowed rings). Kernel binaries are outside this ledger.
For reference, the deferred `interleaved_streaming` regime would own `3 · ring_depth_blocks · neo_block_tiles · C · tile_bytes`. That scales with `ring_depth_blocks` and `neo_block_tiles`, not with any op dimension.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor_a` (read) | 0 | resident shard consumed in place through `cb_a_shard` (borrowed) | 0 |
| `input_tensor_b` | 0 | resident shard consumed in place through `cb_b_shard` (borrowed) | 0 |
| result (= `input_tensor_a`, written) | 0 | packed into `a`'s own shard through `cb_result_shard` (borrowed) | 0 |

Totals per tier: DRAM 0 B. NoC 0 B. Core-local L1: `a` read once, `b` read once, `a` written once (3 × tensor bytes; bfloat16: 6 B per element), the minimum for an in-place binary op.

> Cheapest-traffic split considered: rows along the caller's height shards, consumed in place through borrowed rings: 0 B DRAM, 0 B NoC. Implemented: the same split. It is also the cheapest possible.
