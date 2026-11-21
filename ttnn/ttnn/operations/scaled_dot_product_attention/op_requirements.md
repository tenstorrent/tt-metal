# Operation Requirements: scaled_dot_product_attention

## Definition
- **Formula**: `out[b,h,i,:] = Σ_j softmax_j(scale·Σ_d q[b,h,i,d]·k[b,h//g,j,d] + mask[i,j]) · v[b,h//g,j,:]`, `g = Hq // Hkv`, `mask = 0 / −inf (j > i)` when `is_causal`, none otherwise; `scale=None` → `1/sqrt(D)`.
- **PyTorch Reference**: `torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(Hq // Hkv, dim=1), v.repeat_interleave(Hq // Hkv, dim=1), is_causal=is_causal, scale=scale)`; the authoritative float64 form is `reference()` in `eval/specs/scaled_dot_product_attention/spec.py`.
- **Import Path**: `from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention`
- **Function Signature**: `scaled_dot_product_attention(input_tensor_q: ttnn.Tensor, input_tensor_k: ttnn.Tensor, input_tensor_v: ttnn.Tensor, *, is_causal: bool = True, scale: float = None, program_config=None, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (this run: Quasar on ttsim)**: ttsim has no clock. Every perf refinement below is gated on a **structural count** read off the op by a test or probe (`.claude/references/backends/ttsim.md`, *Performance measurement*), not on device-ns. Its gain in time stays **unconfirmed** until a timed backend (ttsim-perf) or hardware measures it. Never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config ∈ {DRAM}, sharding ∈ {interleaved}
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned}, batch ∈ {single}, kv_heads_mode ∈ {equal, grouped}
- **SUPPORTED op-specific axes**: is_causal ∈ {False, True}
- **Cores**: full grid by construction. NEO units (one query tile-row × D of one query head) are dealt C = 4 per cluster round, and whole rounds go to clusters (`split_work_to_cores`, `row_wise=True`). Thread plan `(R, C, W) = (4, 4, 2)` comes from one host constant. K/V are shared by a cluster's 4 NEOs through ALL rings.
- **Regimes**: `resident_kv` (one key chunk) on every Phase 0 shape; `streamed_kv_online` (online softmax over several key chunks), same kernels, regime-pinned by tests.
- **Compute config**: `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`. Any config is accepted (R5–R7); the exact exponential is always used.
- **Golden baseline**: 18 / 18 supported cells pass (14 cells + 4 numeric regression cases), 6 / 6 rejections, 210 xfail_expected, 0 loud (`golden_results/verifier_report.json`).

### [ ] Refinement 1 — Several sequences and ragged sequence lengths (key-padding mask + fit-driven thread plan)

**Goal**: add `"batched"` to `SUPPORTED["batch"]` and `"h_non_aligned"` to `SUPPORTED["alignment"]`, so that every INPUTS entry of TARGET runs at the Phase 0 precision corner. The kernels already carry the batch index from day 1 (`round → (b, c, head_round)`), and every `batched` cell except one passes today (verifier probe with SUPPORTED widened in-process). Two pieces of real work remain:

1. **Key-padding mask (rule R13)**. Without a mask, nothing keeps the padding columns of the last key tile out of the softmax. The non-causal `h_non_aligned` cells fail today: `(1,1,50,64)` rel-RMS 0.151, `(1,4,17,32)` PCC 0.9914 / rel-RMS 0.362, `(1,2,200,64)` rel-RMS 0.067, `(2,2,100,64)` rel-RMS 0.142. Following the design (Dataflow Strategy, "h_non_aligned is a knob-turn"), add a constant tile that is −inf on columns ≥ `S mod 32`. Add it in `score_block` to the last key tile of every row, whatever `is_causal` is. It reuses the diagonal-mask path (one more `cb_constants` entry; its index goes in `sdpa_geometry.hpp` next to `SCALER_TILE` / `DIAG_MASK_TILE`). The causal `h_non_aligned` cells already pass, because the diagonal masks every padding key for every real query row. The padding mask makes R13 hold by construction rather than by that coincidence. Padding query rows stay computed and unread; every one of them attends key 0, so no NaN can appear.
2. **`(2,1,32,4096)` (D = 4096, Dt = 128)**. The fit refuses it at plan (4, 4, 2): `out_accum` and `query_block` are each 2,097,152 B, over the 1,048,560 B ring limit even at `qc = kc = 1`. Measured by the verifier at plan **(1, 1, 1)**: it fits (1,601,536 B), and both causal values pass (PCC 0.999998 / 0.999999). The shape has one unit per kv head (`units_per_head = 1`), so C = 4 spends 3 NEOs on padding there anyway. Make C a per-call output of the host plan instead of a fixed value. Take the largest `C' ≤ THREAD_PLAN.C` (and R, W dividing it) for which the fit holds **and** `C' ≤ units_per_head`, following the target page: "a cluster needs at least C units of work for C threads to help". `THREAD_PLAN` stays the one named constant and becomes the ceiling. The same rule also stops padding NEOs on `(1,1,32,32)` (3 of 4 NEOs pad today), `(1,4,64,64)` and `(1,2,64,96)` (2 of 4).

**Verifier notes**: these are knob-turns on the planner's own Later-unlock column (`batch`: knob-turn; `h_non_aligned`: knob-turn; thread plan: one constant). No new topology. Ordered first because it is the only generality work that changes kernel code (the third mask tile and the `cb_constants` capacity, `const_tiles()`) and the host plan. Refinement 2 then widens dtypes and placement over the final kernel and plan. The design's deferred `wide_head_dim` regime (D blocked) is **not** needed for any INPUTS entry once C follows the fit; leave it deferred. If the per-call C makes a thread-plan test's assumption false (`test_each_neo_computes_its_own_unit` asserts `plan.C == THREAD_PLAN[1]`), restate the test against `plan.C` and keep one shape with `units_per_head ≥ 4` so it still shows four NEOs at work. Add regime-pinned tests for a ragged shape: non-causal `(1,2,200,64)` resident and at `k_chunk_tiles=3`, so the padded key tile is the whole last chunk.

**Done when**: `SUPPORTED["batch"]` holds `"batched"` and `SUPPORTED["alignment"]` holds `"h_non_aligned"`; every golden cell of those values at bfloat16 / `fp32_dest_acc_en=True` / DRAM passes, `(2,1,32,4096)` with both `is_causal` values included, with no EXCLUSIONS; the plan tests pass with the per-call C; prior phases stay green.

### [ ] Refinement 2 — Numerical configurability and queries in L1

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]`, `False` to `SUPPORTED["fp32_dest_acc_en"]`, and `"L1"` to `SUPPORTED["memory_config"]`. Keep the R4 refusal (float32 with `fp32_dest_acc_en=False`) in `validate()` in every phase. The verifier probed the first-phase shapes with SUPPORTED widened in-process, and all of them already pass:

- float32 / fp32 acc: PCC ≥ 0.999999, rel-RMS 0.0010–0.0021 (limit 0.999 / 0.01).
- bfloat16 / 16-bit acc: PCC ≥ 0.99994, rel-RMS 0.0052–0.0111 (limit 0.99 / 0.13).
- Q in L1: identical to DRAM.

So this refinement is mostly declaration plus the precision-matrix tests. It must still confirm that the internal ring formats follow `fp32_dest_acc_en` (bfloat16 rings with 16-bit DEST, 8 DEST tiles in half-sync) and the input dtype (`T_in` = 4,096 B for float32, which tightens the fit).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: the dtype and accumulator values land together, per the skill (fp32 and bf16 DEST in one refinement). `memory_config = L1` for the queries has no skill: it is a placement knob-turn. Q is interleaved and is read through its `TensorAccessor` wherever it is (no CB placement change), and the output stays interleaved in DRAM (rule R11, `ttnn.DRAM_MEMORY_CONFIG` in `run_attention`). It rides here because it only widens a SUPPORTED list. It runs after Refinement 1 so that the float32 cells (larger `T_in`, a smaller `kc_fit`) are checked against the final fit and thread plan, `(2,1,32,4096)` float32 included. With `fp32_dest_acc_en=False` the score / PV DEST sections may grow (`DEST_AUTO_LIMIT` follows the flag), so cover the streamed regime at both DEST widths in the regime-pinned tests. Nothing in this refinement is expected to land in EXCLUSIONS; if a cell fails, route it by its failure category (precision → stays failing and named here; a structural gap → EXCLUSIONS with its reason).

**Done when**: `SUPPORTED["dtype"]` = [bfloat16, float32], `SUPPORTED["fp32_dest_acc_en"]` = [True, False], `SUPPORTED["memory_config"]` = [DRAM, L1]; every golden cell in the widened rectangle passes, except the R4 cells, which are refused with `NotImplementedError`; the precision-matrix test covers (bfloat16, True), (bfloat16, False) and (float32, True) on a resident and a streamed shape; prior phases stay green.

### [ ] Refinement 3 — Read each K/V chunk once per kv head and multicast it to the clusters serving that head

**Type**: perf

**Goal**: the `kv_multicast_across_clusters` regime the design defers, which is the cheapest-traffic split in `l1_ledger.md`. Every cluster round re-reads its kv head's K and V from DRAM, so K/V cross DRAM `rounds_per_head` times per head. On `(1,2,512,64)` (16 tiles of tokens, two query heads on one kv head, 8 rounds on 8 clusters), at the Phase 0 configuration (bfloat16, `fp32_dest_acc_en=True`, DRAM, plan (4, 4, 2)), that is 512 K+V tiles (1,048,576 B) non-causal and 288 tiles (589,824 B) causal, against a minimum of 64 tiles (131,072 B). The fix is one injector reader per kv head: it reads each K/V chunk once and multicasts it into the same ALL-ring offset on every cluster of that head, gated by a semaphore pair per reader thread. The receivers' readers push the block without a DRAM read. The built per-cluster ALL ring is already the receive side, so only the reader's producer side changes. Pattern: `ttnn/ttnn/operations/examples/master.md` → `shared_input_reuse` (⭐⭐⭐ T3: read once + multicast; its measured win is far smaller than its read-count cut, because the injector serializes) and `mcast_topology` (a 1-D work split needs the operand broadcast to every cluster of the split). No SUPPORTED change. Free-selected region: the specification flags no `LOOSE_CASES` entry.

**Verifier notes**: a scheme-change (a reuse multicast), so it stands alone as one T3 lever for the whole phase. No Python `ProgramSpec` reference on Quasar exercises a cross-cluster multicast yet (`.claude/references/quasar_op_examples/` README, "Not covered"). Start from the multicast helper (host `ttnn.Mcast` + `attach()`, kernel `McastArgs` `sender(noc).send()` / `receiver(noc).receive()`) and from the Quasar matmul's semaphore pair (`unified_matmul_reader.cpp:106-119`). If the helper does not run on this part, that is the partial-tick blocker: name it and file `Refinement 3b`. Keep the explicit-sync / trid-0 ring protocol on every new transfer, and keep the one-session test (another program, the op, `ttnn.to_layout`) green. Causal rounds need different key extents; the injector reads up to the furthest round's diagonal, and each receiver pops what it does not compute, as today. Fold in the ledger's open item: re-state the data-movement budget per tier (DRAM, multicast NoC payload `(clusters_per_head − 1)·|K+V|`) once built.

**Done when**: a test or probe reads the K+V DRAM tile reads per call off the op on `(1,2,512,64)`: from 512 (non-causal) / 288 (causal) down to 64, i.e. `Skt·Dt` per tensor per kv head. Q and O DRAM traffic and the clusters engaged stay as they are. The golden suite, the regime-pinned, thread-plan and one-session tests stay green. No structural count regresses on the guard set: one representative per kernel path × regime × placement (resident and streamed, causal and not, a shape with one cluster per head such as `(1,1,128,64)`, and Q in L1 once Refinement 2 has landed). The changelog records the counts before and after and states that the gain in time is unconfirmed on ttsim.

### [ ] Refinement 4 — Per-chunk fixed costs of the streamed regime

**Type**: perf

**Goal**: the streamed regime (`k_chunk_tiles < Skt`) is the one the model's calls take (S = 4096–12288; the design's worked model call `(1,16,4096,128)` lands at `qc = 16`, `kc = 4`). It pays per-chunk fixed costs that the planner listed as perf lamps (`op_design.md`, *Perf lamps*):

1. **`rescale_accum_block` is its own pass over O.** For every chunk after the first it unpacks and packs every O tile once, and then `pv_block` unpacks and packs them again. Fuse the rescale into `pv_block`: reload `O_old`, multiply by α in DEST (column broadcast), and accumulate P·V on top in the same section.
2. **`row_max_block` / `row_sum_block` re-run `reduce_init` and `pack_init` once per query tile-row.** Hoist them to once per block where the section has no interleaved copy (chunk 0 always; chunk > 0 if the fold's copies can move ahead of the reduce).

Target: `(1,2,512,64)` at the Phase 0 configuration with `q_chunk_tiles=2`, `k_chunk_tiles=4` (the regime-pinned `qc2` / streamed knobs the tests already force). Levers: `ttnn/ttnn/operations/examples/master.md` → `compute_fusion` and `compute_block_size` (both ⭐⭐ T2). No SUPPORTED change. Free-selected region: no `LOOSE_CASES` entry is flagged.

**Verifier notes**: two T2 compute levers that touch the same kernel, so one phase. It runs after Refinement 3 because that one changes only the reader, and the two do not interact. Every state ring is rotated in place, so the fused section must still pop `O_old` before it reserves its successor (Key Risks, "In-place rotation"), and α must still be popped exactly once per chunk. This is a soft dependency on Refinement 2: run the count at both DEST widths once it has landed, since the 16-bit DEST doubles the section size.

**Done when**: a test or probe reads two counts off the op on the target (e.g. a test-only CT flag in the style of `tag_threads` that counts the compute kernel's `out_accum` pushes and its `reduce_init` calls per round):

- O-tile packs per chunk after the first: from `2·qc·Dt` to `qc·Dt` (8 → 4 at `qc = 2`, `Dt = 2`);
- reduce inits per `row_max_block` / `row_sum_block`: from `qc` to 1 where the section allows.

Results stay bit-identical across thread plans; the golden suite and the regime-pinned tests stay green. No count regresses on the guard set (resident and streamed, causal and not, both DEST widths). The changelog records before and after, and states that the time gain is unconfirmed on ttsim.

### [ ] Refinement 5 — Split the key axis across idle clusters when the query units under-fill the grid

**Type**: perf

**Goal**: the design's deferred `split_kv_across_clusters` regime. When `B·Hq·Sqt` units cannot fill `C·num_clusters` NEOs, split each unit's keys into `ksplit` slices on different clusters, and combine the partial `(m, l, O)` triples on the unit's owner cluster. The combine is the built `row_max` / `row_sum` / rescale block operations applied to a peer's triple. On `(1,2,512,64)` at the Phase 0 configuration there are 32 units (16 query tile-rows × 2 heads), dealt to 8 of the 32 clusters, so 96 of 128 NEOs are idle. Each NEO on the critical path walks all 16 key tile-rows non-causal, and the last q chunk walks 16 causal. Pattern: `ttnn/ttnn/operations/examples/master.md` → `tensix_all_reduce` / `tensix_all_reduce_ring_transport` (⭐⭐⭐ T3: a cross-core partial combine) and `width_split`. No SUPPORTED change. Free-selected region: no `LOOSE_CASES` entry is flagged.

**Verifier notes**: a scheme-change (a dependent axis split across cores), so it stands alone. It runs after Refinement 3 because it changes how the K/V readers are grouped. With a split, each cluster reads only its key slice, so Refinement 3's multicast group becomes "the clusters serving the same (kv head, key slice)". Building the multicast first keeps that a regrouping instead of a rewrite. Choose `ksplit` on the host as the smallest split that fills the grid; it is 1 whenever the q units already fill it, which the model's 4096–12288-token calls do. Causal: slices past a unit's diagonal are empty, so deal key slices so that no cluster gets only empty slices (the planner's causal-load-balance lamp). The added traffic is in the budget the ledger asks for: Q re-read `×ksplit`, plus `(ksplit−1)·qc·(Dt + 2)·T_acc` of NoC per unit and a semaphore round per unit. Record it in `l1_ledger.md`'s data-movement budget.

**Done when**: a test or probe reads off the op on `(1,2,512,64)`, non-causal and causal:

- clusters engaged: 8 → 32;
- key tile-rows walked by the busiest NEO: 16 → 4 at `ksplit = 4`.

Results stay within the cell's numeric requirement (the combine reorders the softmax sums), the golden suite and the thread-plan / one-session tests stay green, and `ksplit = 1` shapes are unchanged (no regression on the guard set: one shape per kernel path × regime, causal and not). The changelog records the counts before and after and states that the time gain is unconfirmed on ttsim.
