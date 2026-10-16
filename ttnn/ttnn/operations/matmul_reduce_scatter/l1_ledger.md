# L1 Ledger: matmul_reduce_scatter

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are those of `op_design.md` → Blocking Model:
`g` (device), `j` (entry/block), `m` (core_m), `n` (core_n), `k` (kb / Kt), `c` (transport chunk). `P = core_m·core_n`
(nominal per-core sub-block, tiles). DEST is fp32 on the matmul cores, always (see op_design.md, Perf lamps).

## Matmul cores (`grid_m·grid_n` cores per chip)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_k_window` | resident-xblock ∧ scatter -1: `core_m·Kt`; stream-xblock: `depth_in0·core_m·kb` (`depth_in0 = 3`); resident-xblock ∧ scatter -2 (A is the streamed operand): `d·core_m·kb`, `d` = the first of `RESIDENT_STREAM_DEPTHS = (3, 2)` at which a resident kb fits (Refinement 3) | resident: `core_m·Kt` (the whole K of this core's A rows, reused by all G entries). Stream: `3·core_m·kb` (one K-block computing, one landing, one prefetched by the line's sender: the loader issues K-block s+1's DRAM reads while K-block s multicasts). | `{g: streams → 1, j: streams (resident: spans, reused across all entries), m: spans → core_m, n: streams (independent of n), k: spans → Kt (resident) / kb·depth (stream), c: streams}` | A dtype: Float16_b or Bfp8_b (1088 B pages; derived from the A tensor since Refinement 1) | reader | compute | whole call | None. Live alongside every other CB for the whole call (the matmul of entry j+1 overlaps the finish of j). Residency is the regime predicate, not an alias. |
| `cb_weight_k_window` | resident-xblock ∧ scatter -2: `Kt·core_n`; stream-xblock: `depth_in1·kb·core_n` (`depth_in1 = 3`); resident-xblock ∧ scatter -1: `d·kb·core_n` (`d` as for A) | resident: `Kt·core_n`; stream: `3·kb·core_n` (one computing, one landing, one prefetched) | `{g: streams → 1, j: streams (resident: spans), m: streams (independent of m), n: spans → core_n, k: spans → Kt / kb·depth, c: streams}` | W dtype: Float16_b / Bfp8_b (1088 B) / Bfp4_b (576 B, Refinement 1) | reader | compute | whole call | None. Concurrent with `cb_act_k_window` (both are operands of every K-block). A different format means the two can never share. |
| `cb_partial_hold` | `depth_hold·P`: `hold_depth()` of the regime's schedule, 2 if any entry of the walk lags, else 1 (Refinement 3: the mirrored ring_balanced walks of the resident regime never lag, so they hold 1 slot) | `2·P` when lagging: entry e is held for its lagged arrival while entry e+1's matmul L1-accumulates into the second slot; `1·P` otherwise (each finish pops its slot before the next entry reserves) | `{g: streams, j: spans → 2 (lag 1 + current), m: spans → core_m, n: spans → core_n, k: streams (L1-accumulated in place across Kt/kb K-blocks; no K extent), c: streams}` | Float32 (DEST is fp32 and the slot is a packer-L1-accumulation target) | compute (matmul_block, caller-owned reserve/push) | compute (finish) | per entry, at most 2 live | None. It *is* the matmul's interm target (`Interm` + `caller_owns_pack_target`), so the separate spill CB was folded into it (Rule 3.4). It cannot alias `cb_finished`, which has a different format (Float32 vs Float16_b), a concurrent lifetime (finish reads hold while writing finished) and a different consumer core set (ports read finished remotely). |
| `cb_finished` (staging offset 0) | `depth_handoff·P` (`depth_handoff = 2`) | `2·P`: entry e being gathered by a port while entry e+1 is finished | `{g: streams, j: spans → 2, m: spans → core_m, n: spans → core_n, k: streams (absent), c: streams (ports read it chunk by chunk, chunk_tiles ≤ P)}` | Float16_b. A traveling partial may be bf16 per the spec; the fin output is bf16. | compute | writer (the port reads its bytes remotely under the writer's ack protocol, so the writer is the only CB consumer) | per entry until acked (start/relay) or written (fin) | Shares the **staging tensor allocation** with the two arrival CBs (disjoint `address_offset` ranges, never overlapping bytes). It cannot alias them: all three are live together. Fin output goes through this same CB, so there is no separate output CB (Rule 3.1, pack into the destination). |
| `cb_arrivals_fwd` (staging offset `depth_handoff·P·2048`) | `slots_fwd·P`, where `slots_fwd = max over halves` of the forward-travelling arrivals of a core. **Implementation: the max over every chip of the group as well (min 1)**, so every chip shares one staging layout and a port can address its downstream zone without per-chip offsets (line G=4: 3+3 instead of the per-chip G+1 = 5; ring: unchanged). | `slots_fwd·P`: every slot is written once by the fabric at an address fixed for the call, and may land before compute reaches it | `{g: spans → slots_fwd (one slot per upstream contribution: G-1 bounded), j: spans → slots_fwd, m: spans → core_m, n: spans → core_n, k: streams, c: streams (packets land chunk by chunk)}` | Float16_b | reader (credit; the bytes are written by the upstream port over the fabric) | compute | whole call | Staging allocation shared with `cb_finished` / `cb_arrivals_bwd` at disjoint offsets. It cannot be reused across slots inside a call: a slot reused before compute consumed it would need a credit round trip to the remote sender (regime `port-L1 landing with credits`, rejected). |
| `cb_arrivals_bwd` (staging offset after fwd) | `slots_bwd·P` | `slots_bwd·P` | `{g: spans → slots_bwd, j: spans → slots_bwd, m: spans → core_m, n: spans → core_n, k: streams, c: streams}` | Float16_b | reader | compute | whole call | As `cb_arrivals_fwd`. It is a separate CB because the fin add reads both arrival streams at once (two CB inputs of one chain). |

## Port cores (`2·L` per chip)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_port_chunks` | `2·port_group·chunk_tiles` (`port_group = 4`, `chunk_tiles ≤ 7`) | `2·port_group·chunk_tiles`: one batch being read in while one batch is sent (2× is the double buffer; the batch = `port_group` chunks) | `{g: streams, j: streams, m: streams, n: streams, k: streams (absent), c: spans → 2·port_group}` | Float16_b | reader (port) | writer (port sender) | whole call | The only CB on the core, so there is nothing to share. |

## Implementation deltas (ttnn-implementer)

| Item | Planner | Implemented | Why |
|---|---|---|---|
| `kb` | largest divisor of Kt that fits | largest divisor `<= KB_MAX = 16` that fits (then smaller, then larger) | the streamed operand needs several K-blocks per entry to overlap read / mcast / math: focus case kb 64 -> 212 us, 32 -> 160, 16 -> 159, 8 -> 164 (diagonal senders) |
| stream depth | 2 | 3 (stream regime); resident regime: 3, else 2 if only that fits (`RESIDENT_STREAM_DEPTHS`) | sender prefetch window (`PREFETCH_STEPS = 1`); depth 2 keeps PREFETCH 1 legal (cap 2 windows) |
| `depth_hold` | 2 | 2 if the schedule lags any entry, else 1 (`hold_depth()`) | the second slot exists only for the lag; the mirrored resident walks never lag (Refinement 3) |
| zone slots | per chip | max over the group's chips | one staging layout for all chips (see `cb_arrivals_fwd`) |
| transport strip | `choose_grid` key (P first) | row 0 preferred (`PREFER_PORT_ROW_STRIP`) | ports right under the Ethernet row: focus case 150-160 -> 137 us although P grows 12 -> 14 |
| ports | strip cells nearest the Ethernet row | each port at the strip cell nearest its link's Ethernet core (channel from `setup_fabric_connection`, no probe dispatch) | 137 -> 133 us |
| L1_budget | `L1_bank_size - allocator_base - 64 KiB` | `reports.get_device_info(mesh).l1_bank_size - 64 KiB` (the bank size is already past the allocator base); the cached grow-only staging tensor counts at its allocated size | the first estimate (`get_max_worker_l1_unreserved_size`) was ~70 KiB high and clashed CBs with the staging tensor |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `core_m`, `core_n` | `core_m ≤ ceil(block_m_tiles / min(ext_m, block_m_tiles))`, likewise n. For INPUTS on an 11×10 grid: `P ≤ 26` | `choose_grid()` minimizes `P` over strips and orientations |
| `Kt` | unbounded op dim | appears only under `resident-xblock`, whose predicate is `F(kb, True) ≤ L1_budget` |
| `kb` | `kb | Kt`, `kb ≤ Kt` | `choose_k_block()`: the largest divisor that satisfies `F ≤ L1_budget` (resident needs `kb ≥ min(Kt, 8)`) |
| `G` | 2 ≤ G ≤ 8 for every deployment (Galaxy axes 4 / 8, mocks 4 / 8) | mesh shape |
| `slots_fwd + slots_bwd` | ≤ G + 1. Line interior: `(G-p) + (p+1)`. Ring unbalanced: `G-1`. Ring balanced: ≤ G (per-half maxima, e.g. 2 + 2 at G=4). | `build_schedule()` (per-direction maxima over the two core halves) |
| `chunk_tiles` | ≤ 7 at 14 KiB payload (2048 B tiles) | `floor(get_tt_fabric_max_payload_size_bytes() / 2048)`, clamped to `P` |
| `port_group` | ≤ 8 (header pool 12/RISC) | constant 4 |
| `L` | ≤ (strip length)/2 | `get_forwarding_link_indices` + strip check |
| `L1_budget` | `L1_bank_size − allocator_base − 64 KiB` | device query at plan time |

## Total per-core footprint

Matmul core (`tA` = A tile bytes, `tW` = W tile bytes; `rA = resident ∧ scatter -1`, `rW = resident ∧ scatter -2`):

```
F(kb, resident, d) = tA·(rA ? core_m·Kt : d·core_m·kb)       // scales with core_m and kb (or Kt under rA)
                   + tW·(rW ? Kt·core_n : d·kb·core_n)       // scales with core_n and kb (or Kt under rW)
                   // d = 3 (stream) | first of RESIDENT_STREAM_DEPTHS = (3, 2) that fits (resident)
                + 4096·hold_depth·P                           // scales with P (hold_depth: 2 if the walk lags, else 1)
                + 2048·max(cached, (2 + slots_fwd + slots_bwd)·P)   // staging tensor (grow-only cache): P and G
```

Port core: `2048·2·port_group·chunk_tiles` (112 KiB at the defaults; independent of shape).

Worked examples on QB2 (11x10 grid, row transport strip, `L1_budget` ~1.33 MiB; verifier, recomputed from `choose_grid` / `choose_k_block` / `footprint`):

- Focus (640x2048x7168, scatter -1, G=4 ring balanced, W bf8b): grid 10x8 = 80 matmul cores, `core_m=2, core_n=7, P=14`, subblock 1x7, `slots_fwd = slots_bwd = 2`, resident A, `kb=16`: `256 KiB (A, core_m·Kt) + 357 KiB (W window, 3·16·7·1088 B) + 112 KiB (hold, 2·14·4 KiB) + 168 KiB (staging, (2+2+2)·14·2 KiB) = 893 KiB`.
- Focus (Refinement 3 update): the mirrored walk never lags, so `hold_depth = 1`: 893 − 56 = **837 KiB**, same plan (resident A, kb 16, depth 3).
- MiMo (2048x2048x4096, scatter -2, W bf8b): grid 8x11 = 88 cores (of the 99 rectangle cores: block_m = 16 tile rows do not split over 9 m-lines, so `core_m = 2` on 8 lines), `core_n=12, P=24`, `slots_fwd = slots_bwd = 2`. **Refinement 3: resident-xblock, kb=8, A window depth 3, hold depth 1** (measured `_LAST_PLAN`): `816 KiB (W resident, Kt·core_n·1088 B) + 96 KiB (A window, 3·2·8·2 KiB) + 96 KiB (hold, 1·24·4 KiB) + 288 KiB (staging, 6·24·2 KiB) = 1296 KiB` against the 1363 KiB budget (`l1_bank_size` 1427 KiB − 64 KiB). Before Refinement 3 the resident candidates were `816 + 192 (A, kb 16) / 96 (kb 8) + 192 (hold 2) + 288 = 1488 / 1392 KiB`, both over budget, so the case ran stream-xblock (`192 + 612 + 192 + 288 = 1284 KiB`) and W crossed DRAM G = 4 times. What closed the gap: the hold's second slot is unused when no entry lags (−96 KiB), plus the kb 8 / 16 choice inside `KB_MIN_RESIDENT`. Two other fitting resident plans measured the same within noise: kb 8 / depth 2 / hold 2 (1360 KiB) and kb 16 / depth 2 / hold 1 (1328 KiB). With **bf4b W** (576 B tiles, Refinement 1) the resident W is 432 KiB and the case was already resident.
- Wave split (Refinement 4, `ENTRIES_PER_BLOCK = W > 1`, resident regime only; parked at 1, so the plans above are unchanged). P is one wave's slot, `win_m·win_n`, and the streamed window is one wave's (`win_n` or `win_m`). The zone slots per direction scale by W (one slot per virtual block), and the resident kb cap is `KB_MAX·W`. Focus with W = 2: `P = 8`, kb 32, `256 KiB (A) + 3·32·4·1088 B = 408 KiB (W window) + 32 KiB (hold, 1·8·4 KiB) + (2+4+4)·8·2 KiB = 160 KiB (staging) = 856 KiB`. The DRAM traffic is unchanged: the streamed operand's waves are disjoint slices, and the resident operand is shared by every wave.
- **Perf 1 (DEST-resident K super-blocks):** the streamed operand's window depth becomes `max(depth, SUPER_BLOCK + SUPER_BLOCK_EXTRA) = 4` when `Kt/kb` is even and the footprint still fits (`super_block_plan()`, which re-runs `footprint()` with the deeper windows; else super-block 1 at the old depth). Focus: W window `4·16·7·1088 B = 476 KiB` → 837 + 119 = **956 KiB**. GLM (640x4096x6144, resident A 512 KiB, W 4·16·6·1088 B) ≈ 1112 KiB. MiMo: A window `4·2·8·2 KiB = 128 KiB` → 1296 + 32 = **1328 KiB** (budget 1363). Stream regime (K3 FFN, kb 12): both windows at depth 4.
- A shape that still misses: under the stream regime the hold keeps 2 slots whenever its (unmirrored) walk lags, and the resident regime falls back to the stream regime exactly as before once even depth 2 / kb = `min(Kt, 8)` misses.

## Data-movement budget (per chip, chosen split)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| A `(M, K)` | 1 (resident-xblock or scatter -2); G (stream-xblock ∧ scatter -1) | the A slice is reused by every entry under scatter -1 and kept in L1 when `F` fits. Under scatter -2 each entry needs different rows. | mcast once per m-line (scatter -1 resident) or once per entry |
| W `(K, N)` | 1 (scatter -1, or scatter -2 resident); G (stream-xblock ∧ scatter -2) | under scatter -1 each entry reads a different column stripe. Under scatter -2 every entry needs all of W. | mcast once per K-block per n-line |
| output block `(M·N/G)` | 1 write | the fin finish writes straight from `cb_finished` | — |
| partials (G blocks per chip) | **0** | They land in the next chip's matmul-core L1 zones (L1-landing). Own partials are handed off from L1. | on-chip: ports NoC-read `(G-1)` blocks per chip from the matmul cores. Fabric: per chip `(G-1)·B` sent in total. The busiest link direction carries `(G-1)/2·B/L` (ring) or `(G-1)·B/L` (line). |

Focus case (640×2048×7168, scatter -1, ring G=4, L=2, bf8b W): DRAM = 2.62 MB (A) + 15.6 MB (W) + 2.29 MB (out) = **20.5 MB**, about 40 µs at 512 GB/s. The roofline's fused-DRAM model counts 34.2 MB (66.9 µs) because it assumes relay scratch in DRAM. Fabric: 1.72 MB per link direction.
MiMo (2048×2048×4096, scatter -2, bf8b W), **resident-xblock since Refinement 3**: DRAM = 8.4 (A) + 1×8.9 (W) + 4.2 (out) = **21.5 MB** (was 48.2 MB under stream-xblock, 4×8.9 MB of W), about 42 µs at 512 GB/s. Measured on device: the W fill is spread over entry 0 (57.5 µs against 45.9 µs for each later, compute-bound entry); the stream regime's 35.6 MB of W was what held the case at 243-256 µs.

> Cheapest-traffic split considered: 2D (m,n) per block, entries walked in time, cross-block operand resident, arrivals landing in L1. That moves A and W once each, the output once, and 0 partial bytes through DRAM. Implemented: **the same split** (resident-xblock + L1-landing), which is the cheapest. The stream-xblock fallback is selected only when `F(kb, resident) > L1_budget`. Its delta is +(G-1)× the cross-block operand in DRAM, and it is reached by the same loop nest (the sender re-fetches, the window shrinks).
