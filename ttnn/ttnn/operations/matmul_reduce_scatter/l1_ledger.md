# L1 Ledger: matmul_reduce_scatter

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md`):
`scatter_block`, `m`, `n`, `k`, `segment`, `link`, `direction`. `X` = the block-invariant operand
(A for `scatter_dim=-1`, W for `-2`), `Y` = the other. `w_tile` = 2048 B (bf16) or 1088 B (bfloat8_b).
`acc_tile` = 4096 B if `fp32_dest_acc_en` else 2048 B. `seg_bytes = seg_tiles·2048`.

## Compute cores (the compute rectangle, `m_lines × n_lines` cores)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_operand` | R1 & X=A: `core_m_tiles·Kt`; else `operand_depth·core_m_tiles·k_block_tiles` | R1 & X=A: the whole capacity (resident, replayed per scatter block); else `core_m_tiles·k_block_tiles` + one prefetch K-block | `{scatter_block: streams (R1 X=A: spans — reused by all G blocks), m: spans → core_m_tiles, n: streams (A does not vary with n; multicast), k: spans → Kt (resident) / k_block_tiles·operand_depth (streamed), segment: n/a, link: n/a, direction: n/a}` | Float16_b (A bf16; Bfp8_b when activation is bfloat8_b, TARGET) | reader (NCRISC) | compute | call | none — concurrently live with `cb_weight_operand` (both feed every matmul K-block); capacity > live set only by the `operand_depth` prefetch window (double buffering, `double_buffer` catalog entry) |
| `cb_weight_operand` | R1 & X=W: `Kt·core_n_tiles`; else `operand_depth·k_block_tiles·core_n_tiles` | as above with W | `{scatter_block: streams (R1 X=W: spans), m: streams (W does not vary with m; multicast), n: spans → core_n_tiles, k: spans → Kt / k_block_tiles·operand_depth, segment: n/a, link: n/a, direction: n/a}` | Float16_b or Bfp8_b (follows W dtype; Bfp4_b TARGET) | writer (BRISC) | compute | call | none — concurrent with `cb_act_operand`; different page format (Bfp8_b) in the FOCUS case |
| `cb_partial_accum` | `core_m_tiles·core_n_tiles` | the whole capacity during the K loop (packer L1 accumulation in place) | `{scatter_block: streams (re-used per block), m: spans → core_m_tiles, n: spans → core_n_tiles, k: streams (folded in place), segment: n/a, link: n/a, direction: n/a}` | Float32 iff `fp32_dest_acc_en` else Float16_b (audit 2: follows DEST width) | compute | compute | one `matmul_block` (per scatter block) | not with `cb_partial_handoff`: matmul_block's TileRowMajor layout requires interm to be its own region (matmul_block_helpers.hpp, `interm_buf` note: "TileRowMajor: must be its OWN region"), and the handoff slot of the previous block is still being drained while this accumulates; not with the operand CBs (concurrent) |
| `cb_partial_handoff` | `handoff_depth·core_m_tiles·core_n_tiles` (backed by `handoff_l1`, globally allocated) | `core_m_tiles·core_n_tiles` being drained + one being packed | `{scatter_block: spans → handoff_depth blocks, m: spans → core_m_tiles, n: spans → core_n_tiles, k: n/a (final values), segment: streams (transport reads it segment by segment), link: streams, direction: streams}` | Float16_b (partials travel bf16; requirement allows it) | compute | reader (NCRISC) — remote transport readers only read bytes while the slot is fronted | call | none — it is the pipelining buffer between matmul and transport (explicit pipelining decision, depth knob); cannot alias `cb_partial_accum` (above) |

## Transport cores (`2L` ports + `2L` finals per chip, in the transport row(s))

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_xport_partial` | `2·xport_group·seg_tiles` | `xport_group·seg_tiles` + one prefetch group | `{scatter_block: streams, m: streams, n: streams, k: n/a, segment: spans → xport_group·seg_tiles, link: n/a (one link per core), direction: n/a (one direction per core)}` | Float16_b | reader | compute | call (relay ports, finals); absent on line-end ports (reader targets `cb_xport_sum`) | none — concurrently live with the arrival CBs (the add consumes all inputs of a group together); 2× = reader/compute overlap |
| `cb_xport_arrival_a` | `2·xport_group·seg_tiles` | as above | same as `cb_xport_partial` | Float16_b | reader | compute | call (relay ports; finals with a forward upstream) | none — concurrent second add operand |
| `cb_xport_arrival_b` | `2·xport_group·seg_tiles` | as above | same | Float16_b | reader | compute | call (finals only) | none — concurrent third add operand; not allocated on port cores |
| `cb_xport_sum` | `2·xport_group·seg_tiles` | as above | same | Float16_b (bf16 on the wire; the fp32 add happens in DEST before pack) | compute (relay/final) or reader (line-end port) | sender (port) / writer (final) | call | none — the in-place alternative (pack the sum back into `cb_xport_partial`) is rejected: the sender must hold the payload until the fabric flush while the reader already refills the partial CB (concurrent lifetimes) |

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|-------------------------------------|
| `core_m_tiles·core_n_tiles` | ≤ `CORE_BLOCK_MAX` = 64 | grid factorization in `_plan_blocking()`; a plan exceeding it raises `ValueError` (R4 needed); INPUTS max = 48 (G=2) |
| `Kt` | unbounded op dimension; appears **only** in the R1 resident CB | R1 predicate `RESIDENT` (resident bytes ≤ `L1_CB_BUDGET`); otherwise R2, whose capacities use `k_block_tiles` only |
| `k_block_tiles` | divisor of `Kt`, `operand_depth·k_block_tiles·core_y_tiles·w_tile ≤ STREAM_BUDGET` (384 KiB) | `_plan_blocking()` |
| `operand_depth`, `handoff_depth` | 2 (Phase 0) | host constants |
| `seg_tiles` | ≤ `max_payload // 2048` (≤ 7 at 14336 B) | live fabric config |
| `xport_group` | `max(1, min(8, 112 KiB // (2·seg_bytes)))` | reference sizing |
| `L1_CB_BUDGET` | per-core L1 available to CBs minus the `handoff_l1` shard | device query at plan time |

## Total per-core footprint

Compute core:

```
F_compute = A_pages·a_tile + W_pages·w_tile + core_m_tiles·core_n_tiles·(acc_tile + handoff_depth·2048)
  A_pages = core_m_tiles·Kt                              (R1, X=A)  | operand_depth·core_m_tiles·k_block_tiles
  W_pages = Kt·core_n_tiles                              (R1, X=W)  | operand_depth·k_block_tiles·core_n_tiles
```

Scales with: `core_m_tiles` (A, accum, handoff), `core_n_tiles` (W, accum, handoff), `Kt` (resident
operand only, R1), `k_block_tiles` and `operand_depth` (streamed operand), `handoff_depth` (handoff).

FOCUS (`640×2048×7168`, `-1`, R1 X=A, core 2×7, Kt 64, k_block_tiles 16, bf8b W, `fp32_dest_acc_en=False`):
`2·64·2048 + 2·16·7·1088 + 14·(2048 + 2·2048)` = 262,144 + 243,712 + 86,016 = **591,872 B**.
MiMo (`2048×2048×4096`, `-2`, core 2×12, Kt 64, bf8b W, fp32 acc): R1 needs 835,584 B of resident W
+ streamed A + 24·(4096+4096); the predicate decides R1 vs R2 on the live `L1_CB_BUDGET`.

Transport core: `F_xport = n_cbs·2·xport_group·seg_bytes ≤ 4·112 KiB` (finals 4 CBs, relay ports 3,
line-end ports 1). Constant in every block knob except `seg_tiles` (payload).

## Data-movement budget (chosen split R1, FOCUS case, per chip, G = 4 line, L = 2)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| W (15.6 MB, bf8b) | 1 | each scatter block needs a disjoint column slice; one injector per n-line reads it | multicast along each n-line to `m_lines` (10) receivers: 15.6 MB injected |
| A (2.6 MB) | 1 | R1: block-invariant, resident on every core after block 0 (R2: G = 4 crossings) | multicast along each m-line to `n_lines` (8) receivers once: 2.6 MB injected |
| partial P (9.2 MB, bf16) | 0 | never materialized in DRAM: parked in `cb_partial_handoff`, pulled by transport cores | 9.2 MB compute L1 → transport L1 (each tile read once) |
| relay scratch | 2 per received block: ≤ (G−1) blocks written by the fabric and read back (13.8 MB) | DRAM landing (R1); R5 lands in L1 instead | — |
| output (2.3 MB) | 1 | finals store the own block once | — |
| Fabric | busiest direction (G−1)·B = 6.9 MB per link pair | line topology minimum | — |

Totals per tier: DRAM 34.3 MB (66.9 us at 512 GB/s); cross-core NoC 9.2 MB hand-off + 18.2 MB
multicast injected; links 6.9 MB on the busiest direction (70.9 us at 2 × 48.5 GB/s).

> Cheapest-traffic split considered: R5 (R1 + L1 relay landing) — DRAM −13.8 MB (34.3 → 20.5 MB),
> NoC/links unchanged, plus one credit packet per `inc_every` segments on the reverse direction.
> Implemented: R1 (with R2 as the non-resident fallback). Deferred because in all three LOOSE_CASES the
> DRAM-scratch fused DRAM bound stays below the binding term (FOCUS 66.9 < links 70.9 us; GLM 89.3 <
> compute 118.9; MiMo 91.1 < links 129.7), so R1 does not move the roofline; the landing address is a
> per-stream runtime arg and every arrival read is already counter-gated, so R5 stays reachable.
