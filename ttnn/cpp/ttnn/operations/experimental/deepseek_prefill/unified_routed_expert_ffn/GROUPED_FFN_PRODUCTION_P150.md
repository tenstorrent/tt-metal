# Grouped `unified_routed_expert_ffn`: full production comparison (legacy vs grouped), 8x4 P150 galaxy

Host `bh-glx-120-b09u08`, 32 x P150, 14 Gbps DDR, measured at `da953b98507` (this branch's op code plus the default-off experiment knobs of the `-exp` branch). Every arm below runs the SAME
harness, inputs, iteration count and reset procedure (`tt-smi -glx_reset` before each arm); the only
difference between the arms is the env override that selects the routed-expert FFN program
(`TT_MOE_FFN_ROW_GROUPS=0` legacy, `TT_MOE_FFN_ROW_GROUPS=5 TT_MOE_FFN_GRID_ROWS=10` grouped). Model
defaults are untouched. Runner scripts and full logs: `galaxy_logs/run_prod_*.sh`, `galaxy_logs/prod_*.log`
on the host; status lines in `galaxy_logs/prod_status.txt`.

## 1. Read-ahead experiment: tried, flat, not in this PR

The ablation in `GROUPED_FFN_EXPERIMENTS_P150.md` (branch `zbaczewski/moe-ffn-grouped-exp`) pointed at reading block k+1's weights
while block k is multicast and handshaken. It was implemented on `zbaczewski/optimizer/moe-ffn-readahead-2026-09-07`: the first version
hung because it barriered on NoC transaction id 0, the tag every untagged request on the NIU shares; with the x reads on their own id
it passes the full 185-case op test file (PCC min 0.9799) and measures flat against this op on one P150 (0.985-1.013 geomean over 46
cases, both layouts, both dtypes, also with a third CB slot). The reads were already hidden behind the writer and peer waits; the cost
the ablation exposed is NoC-0 bandwidth shared with the row multicasts, not read latency. Details, tables and raw data live on that
branch; this PR carries neither the read-ahead nor the experiment knobs.

## 2. MiniMax-M3, real weights, longbook_56320 (11 x 5120-token chunks), SP=8 / TP=4 (EP32)

`models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py`, bf4 expert weights, PREFILL_TPS_ITERS=3, per-layer
golden KV PCC after the timed blocks.

| arm | mode | WHOLE SEQUENCE tok/s (median) | LAST CHUNK tok/s (5120 @ 51200 cache) | KV PCC | rc |
|---|---|---|---|---|---|
| legacy | eager | 5732.5 | 5726.7 | layers 0-24: 0.98-0.999; layer 25: 0.000; 26-59: ~0.09 (assert -0.0004 < 0.88) | 1 (PCC assert after timing) |
| legacy | traced | 5744.1 | 5825.1 | min 0.891 (K 0.970 / V 0.891 / index_k 0.981) PASS | 0 |
| **grouped G5r10** | eager | **5766.6** (+0.6%) | **5837.7** (+1.9%) | min 0.890 (K 0.970 / V 0.890 / index_k 0.980) **PASS** — no layer-25 collapse | 0 |
| **grouped G5r10** | traced | **5764.9** (+0.4%) | **5806.8** (-0.3%) | min 0.890 PASS | 0 |


## 3. MiniMax-M3, 4-stage intragalaxy pipeline (4 x [2,4], 15 layers per stage, EP8 per stage)

`run_pipeline_prefill.sh` (served runner, 4 local ranks) + `prefill_producer` (2 users, 11 chunks of 5120, 10 requests round-robin for
timing, then 2 requests with golden KV PCC), same golden. Per-stage EP is 8 experts/chip-mesh: each chip holds 16 experts and does the
same expert-layers of FFN work per chunk as in the SP8/TP4 layout (15 layers x 16 experts vs 60 x 4), so the FFN share of the chunk
is unchanged by the pipeline layout.

Harness fixes needed to run this at all on this main snapshot (all in the shared runner / M3 runtime, none in the op; committed on the
read-ahead branch, see "Harness" in the PR description):

- the [2,4] tensor cache under the checkpoint dir is not writable by this user; `TT_CACHE_PATH` moved it to local disk;
- the bf16 shards are mmapped per tensor, which over NFS is page-fault bound (90 MB/s): the 854 GB of shards were copied to local disk;
- `ttnn` tensor-cache dumps are collective across the tt-run world, so a non-zero rank's dump completes only while rank 0 is also
  dumping (ranks 1-3 converted in lockstep with rank 0's own conversion and froze when it finished): the weight cache is now used by
  rank 0 only, ranks 1-3 convert from the local shards (~20 min per launch);
- the ranks reach the D2D setup at different times and `ttnn.distributed_context_barrier()` is per mesh under tt-run: a shared-filesystem
  barrier now precedes `build_d2d_pipeline_endpoints`;
- the shared runner passes `metadata_msg` to `prefill_chunk`, which the M3 runtime did not accept.

Readiness for the producer is "endpoints up" on all four ranks (rank 0 exports the H2D descriptor before the D2D setup, while peers
may still be loading).

| arm | dtype | ms/chunk per stage (rank 0/1/2/3) | bottleneck stage tok/s | end-to-end tok/s (110 chunks) | producer tok/s |
|---|---|---|---|---|---|
| legacy_bf4 | bf4 | 465 / 423 / 427 / 427 | 11003 | 10667 | 11657 |
| grouped_bf4 | bf4 | 392 / 392 / 396 / 396 | 12926 | 12605 | 13887 |
| legacy_bf8 | bf8 | 524 / 530 / 536 / 536 | 9551 | 9421 | 10435 |
| grouped_bf8 | bf8 | 426 / 431 / 436 / 436 | 11745 | 11605 | 12788 |

Steady state = per-stage `CHUNK_START` period over the 110-chunk timing pass (10 requests x 11 chunks of 5120, 2 users round-robin);
the pipeline throughput is set by the slowest stage. "producer tok/s" is the producer's own push-paced figure (it stops at its last push
and overstates by the pipeline fill), kept for reference only. Table script: `galaxy_logs/pp_table.py`; per-arm logs
`galaxy_logs/prod_pp_<arm>_runner.log`.

| dtype | legacy -> grouped, bottleneck stage | end-to-end |
|---|---|---|
| bf4 | 11003 -> 12926 tok/s (**+17.5%**) | 10667 -> 12605 (**+18.2%**) |
| bf8 | 9551 -> 11745 tok/s (**+23.0%**) | 9421 -> 11605 (**+23.2%**) |

Why the pipeline shows the gain the SP8/TP4 layout does not: per chip the FFN work is identical (15 layers x 16 experts here vs
60 x 4 there), but a PP stage processes a 5120-token chunk in ~430-530 ms against ~880 ms for the SP8/TP4 chunk, because the
stage's collectives (SP2 attention gathers, EP8 dispatch/combine) are far cheaper than SP8/EP32's. The FFN's share of the chunk
therefore doubles, and the op-level saving (15 x (4.4 - 1.9) ms bf4, 15 x (9.7 - 2.85) ms bf8 per stage) becomes 30-100 ms of a
400-530 ms chunk. On bf4 the legacy arm's first stage was also the slowest (465 vs 427 ms; it takes row-major token input from the
H2D path, the layout the legacy program is slowest on) and the grouped arm levels the four stages at 392-396 ms.

PP arms are timing-only: the multi-rank runner has no merged KV chunk table for the producer's golden readback (`PREFILL_MOCK_MIGRATION`
raises for `num_ranks > 1`). Model-level correctness of the grouped path in M3 rests on the SP8/TP4 traced golden KV PCC (0.890, equal
to legacy) in section 2 and the op-level tests; the PP arms ran the same weights and the same model code with only the FFN program
selection changed.

## 4. Kimi-K2.7, real weights, 61 layers, 11 x 5120 chunks, 10 iterations

`test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_perf[...L61-preload0-chunks_eleven-ten_iters-*]`
(timing only; accuracy of the grouped path at model level is the Kimi 5k PCC leg in GROUPED_FFN_GALAXY_P150.md).

| arm | mode | per-chunk median (s) / total | tok/s | rc |
|---|---|---|---|---|
| legacy | notrace | 1.051-1.058 s per 5120-tok chunk (all 11 chunks, stddev <= 0.010 s); sum 11.59 s per 56320-tok prefill; CI baseline band PASS | ~4860 per chunk, ~4860 whole | 0 |
| legacy | traced | see per-chunk table below; CI band: chunks 8 and 10 FASTER than the band's lower edge (0.726 vs [0.727,0.771], 0.795 vs [0.799,0.849]) -> rc=1 by being fast | see table | 1 (gate, not an error) |
| **grouped G5r10** | notrace | 1.053-1.056 s per chunk (unchanged: this path is host-bound, chunk time flat vs cache depth); CI band PASS | ~4860 | 0 |
| **grouped G5r10** | traced | **0.413 -> 0.727 s per chunk (legacy 0.492 -> 0.795): 9-19% faster on every chunk, sum 6.05 s vs 6.91 s per 56320-tok prefill (-12.5%)**; CI band "fails" on the fast side for all 11 chunks | ~9310 whole (legacy ~8150) | 1 (gate, faster than band) |


## 5. Summary

| where the grouped FFN runs | legacy -> grouped | what it says |
|---|---|---|
| op, one P150, 2.0-3.5x (rm A/B) | 2360 -> 999 us Kimi bf4; 1110 -> 493 us M3 bf4 | the op is 2-3.5x faster; tile input narrows this (legacy is 10-38% faster on tile input than in the rm A/B) |
| Kimi-K2.7 whole MoE layer, 8x4 Galaxy | 5.771 -> 4.418 ms (-23.4%) | ~1.0 ms of the remaining 4.4 ms is FFN; the rest is dispatch/combine/routing |
| Kimi-K2.7 prefill, traced | 6.91 -> 6.05 s per 56k (-12.5%) | 9-19% per chunk; the notrace path is host-bound and shows nothing |
| MiniMax-M3 SP8/TP4 (EP32), eager + traced | 5744 -> 5765 tok/s (flat) | 4 experts/chip, ~880 ms chunks: FFN is ~7% of the chunk |
| MiniMax-M3 4-stage PP, bf4 | 11003 -> 12926 tok/s (**+17.5%**) | same FFN work per chip, half the chunk time: the FFN share doubles |
| MiniMax-M3 4-stage PP, bf8 | 9551 -> 11745 tok/s (**+23.0%**) | bf8 FFN is 2x the bytes, so the share and the gain grow again |
| MiniMax-M3 4-stage PP, grouped bf8 vs legacy bf4 | 436 vs 465 ms per chunk at the bottleneck stage | grouped bf8 experts beat the legacy program on bf4 experts |
| read-ahead inside the grouped reader | 0.985-1.011x (flat) | hang fixed; the reads were already hidden, NoC-0 bandwidth is the cost |

Defaults stay legacy in the PR; the Kimi device-perf gates move by -23% / -27% and must be re-cut on the CI galaxy when the default flips
(`GROUPED_FFN_GALAXY_P150.md`, "Decisions").
