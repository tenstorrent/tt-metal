# Operation Design: scaled_dot_product_attention

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (two tiled matrix products with a row softmax between them, one program) |
| Goal | Prefill attention `softmax(scale·Q·Kᵀ + mask)·V` over `[B, Hq, S, D]` queries and `[B, Hkv, S, D]` keys/values (grouped heads), as **one** `ttnn.generic_op` dispatch of one `ttnn.ProgramSpec` on Quasar. |
| Math | `out[b,h,i,:] = Σ_j softmax_j(scale·Σ_d q[b,h,i,d]·k[b,h//g,j,d] + mask[i,j]) · v[b,h//g,j,:]`, `g = Hq // Hkv`; `mask = 0 / −inf (j > i)` when `is_causal`, none otherwise |
| Algorithm | FlashAttention form: a block of query tile-rows walks the keys in chunks, keeping a running row max `m`, a running row sum `l` and a running un-normalized output `O`, rescaled by `α = exp(m_old − m_new)` when the max grows; `out = O / l` at the end. With one key chunk (the resident regime) the rescale steps never run and this is the plain stable softmax. |
| Specification | `eval/specs/scaled_dot_product_attention/spec.py` (authoritative; RULES R1–R14, REJECTIONS X1–X6, NUMERIC) |
| References | `.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`, `.claude/references/quasar_op_examples/` (README, `toy_row_passes`, `toy_thread_plan`), `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A; Part C ALL/STRIDED), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/ttnn/operations/examples/master.md` (entries below), `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/compute/unified_matmul_compute.cpp` (raw Quasar matmul, ALL-bound operands — a different op, read for the API only) |

Catalog entries that informed knobs (`ttnn/ttnn/operations/examples/<name>/README.md`):

| Entry | Knob it informed |
|-------|------------------|
| `shared_input_reuse` (master.md:359) | K/V is the shared operand: shared within a cluster through an ALL ring (built); cross-cluster read-once + multicast is the deferred successor; its 1.71× (not 11×) result is why the multicast is a layering, not a rewrite |
| `compute_block_size` (master.md:220) | block operations take the whole `[q_chunk_tiles × k_chunk_tiles]` block per call, not a tile-row at a time |
| `double_buffer` / `trid_double_issue` (master.md:51,67) | K/V depth = one block per reader thread (R regions → R blocks in flight); Q depth 2 |
| `matmul_output_subblock` (master.md:159) | 32-bit DEST holds 4 tiles in half-sync: score/PV sub-blocks ≤ 4 tiles (inside the block op, the implementer's walk) |
| `row_reduce_accumulate` (master.md:245) | perf lamp on the row-sum realization |
| `compute_fusion` (master.md:201) | exp fused into the subtract chain (SFPU consumer of an FPU result: pays); no FPU-into-FPU fusion |
| `sfpu_tile_scope` (master.md:326) | perf lamp on `1/l` (only column 0 is meaningful) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator (`libttsim.so`, `TT_METAL_SIMULATOR`) | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe on this backend: `compute_with_storage_grid_size()` = 8-4) | target page; queried at run time |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by 4 NEOs + DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` = 3,963,328 − 0 = **3,963,328 B** on this target (probed 2026-11-21 via `scripts/tt-probe.sh`) | query |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`) | target page; probed |
| Host program | `ttnn.ProgramSpec` (Metal 2.0: `ttnn.experimental.{ProgramSpec, KernelSpec, DataflowBufferSpec, WorkUnitSpec, ProgramRunArgs}`), dispatched with one `ttnn.generic_op` — reference `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}), 6 user DM harts (R + W ≤ 6), on STRIDED rings R and W each divide C; ALL consumer fan-out ≤ 4 (`DataflowBuffer.md:147`); 24 active DFBs per NEO, 32 per cluster; ring ≤ 1,048,560 B compute-side; compute binary ≤ 24 KiB | target page *Cluster model / Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` and legacy program factories (refused at kernel creation). `matmul_block` helper (`matmul_block_helpers.hpp`): not used (the matmul is written with LLK calls); `ReduceFp32Mode::Accurate` (static_assert); `reduce<MAX, REDUCE_ROW>` with `Accumulate` (static_assert `reduce_helpers_compute.inl:802-806`); `reduce_helpers` indexed policies: not used (raw `reduce_tile` with stride-corrected indices); `eltwise/binary/sfpu/minmax.hpp`: not used (raw `binary_max_tile`); `pack_init` on every output-ring switch; `UnpackToDest` of a bfloat16 page into 32-bit DEST: not used (bfloat16 rings keep the default unpack); `fpu_math_fidelity` must be set on every compute kernel; `AddUnary/SubUnary/DivUnary` scalar SFPU (`binop_with_scalar.h:34-58`, non-Quasar only — `MulUnary` is fine); Quasar exp rejects `scale_en` (`ckernel_sfpu_exp.h:334`), so the scale cannot be folded into the exponential; the experimental 32-bit-DEST `reduce_block_max_row` (`llk_math_reduce_runtime_custom.h:76`) — not used (the ordinary `reduce_tile<MAX>` path is legal in 32-bit DEST); `bfloat8_b`/`bfloat4_b` (no path on the family) | target page + headers |

Every worked number below is for **quasar, 32 clusters, 3,963,328 B budget**. Every predicate the op evaluates calls the query, never the value.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_q` | `ttnn.Tensor` | yes | `[B, Hq, S, D]`, rank 4 (X1) | — | tensor binding |
| `input_tensor_k` | `ttnn.Tensor` | yes | `[B, Hkv, S, D]`, `Hq % Hkv == 0` (X2), same shape as V (X3), same S (X4), D (X5), B (X6) as Q; DRAM interleaved, TILE, Q's dtype (R10) | — | tensor binding |
| `input_tensor_v` | `ttnn.Tensor` | yes | as K | — | tensor binding |
| `is_causal` (kw-only) | bool | no | True / False | **True** | CT (`causal` compile-time arg; the mask ring and mask code exist only when 1) |
| `scale` (kw-only) | float / None | no | any float; `None → 1/sqrt(D)` (D = logical head dim of Q) | None | RT: common runtime arg `scale_bits` = fp32 bit pattern (`struct.unpack("I", struct.pack("f", scale))`) — so a new scale does not recompile |
| `program_config` (kw-only) | `ttnn.SDPAProgramConfig` / None | no | anything (R8) | None | **ignored**: read by nothing, sizes nothing (R8 "may use or ignore"; the result never depends on it) |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor` or device config (`WormholeComputeKernelConfig`) | no | any; fields read as attributes (R7) | `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` (R5) | CT: `fpu_math_fidelity`, `enable_32_bit_dest`, `sfpu_precision_mode` of `ComputeHardwareConfig` |

Entry-point rules: `compute_kernel_config=None` resolves through the one exported `default_compute_kernel_config()` (R5). Any config object is normalized into a `ttnn.ComputeConfigDescriptor` built from its three fields (R7: no type refusal; `packer_l1_acc`, `dst_full_sync_en`, `throttle_level` ignored), and that descriptor is mapped onto `ttnn.experimental.ComputeHardwareConfig(fpu_math_fidelity=…, enable_32_bit_dest=…, sfpu_precision_mode=Precise unless math_approx_mode, unpack_modes=…)` (the Quasar kernel config; `fpu_math_fidelity` always set explicitly). `validate()` reads neither `math_fidelity` nor `math_approx_mode` (R6). The exponential is always the exact one (`Approx::Exact`), whatever `math_approx_mode` / `exp_approx_mode` say (R8 allows; it is the more accurate choice).

`validate()` order: REJECTIONS X1–X6 (`ValueError`), R10 on K/V (`NotImplementedError` naming the parameter and what it is: placement via `eval.memory.memory_axes`, layout, dtype), then the axes (`INPUT_TAGGERS = AXES` imported from the specification; `memory_config`/`sharding` from `eval.memory.memory_axes(input_tensor_q)`, R9; `dtype`, `layout`, `is_causal`, `fp32_dest_acc_en`) against `SUPPORTED` (= PHASE0) → `NotImplementedError` (R14). R4 (float32 + `fp32_dest_acc_en=False`) is an explicit refusal kept in every phase. After the support check the host plan's fit predicate (Work Distribution) runs; a call that is in the rectangle but whose resident head dimension does not fit raises `NotImplementedError("head dimension D=… needs the wide-head-dim regime")` (regime table). No index axes exist (nothing to canonicalize).

## Tensors

### Input

| Property | Q | K, V |
|----------|---|------|
| Shape | `[B, Hq, S, D]` | `[B, Hkv, S, D]`, `Hq = g·Hkv` |
| Dtype | bfloat16 (Phase 0); float32 later | = Q's |
| Layout | TILE | TILE |
| Memory | interleaved, DRAM (Phase 0) or L1 (later) — read through `TensorAccessor` either way | interleaved DRAM only (R10) |

Tile geometry (alignment-aware from day 1, per image): `Sqt = ceil(S/32)` (= `Skt`, keys have Q's S), `Dt = ceil(D/32)`. Page of tile `(b, h, r, d)` of Q/out: `((b·Hq + h)·Sqt + r)·Dt + d`; of K/V: `((b·Hkv + c)·Skt + r)·Dt + d`. Never `floor(B·Hq·S/32)`.

### Output

| Property | Value |
|----------|-------|
| Shape | Q's shape `[B, Hq, S, D]` |
| Dtype | Q's dtype |
| Layout | TILE |
| Memory | interleaved **DRAM** whatever Q's buffer (R11): `ttnn.allocate_tensor_on_device(TensorSpec(shape, dtype, TILE, memory_config=ttnn.DRAM_MEMORY_CONFIG), device)` |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Axes

Plan constants used below: `(R, C, W) = THREAD_PLAN = (4, 4, 2)`; `g = Hq // Hkv`; `nqc = ceil(Sqt / q_chunk_tiles)`.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (B) | independent — no result spans sequences | `batch_per_round` | 1 — a round cannot span sequences because its NEOs share one K/V ring and K/V differ per sequence (mechanism, not a collapsed knob) | fixed by the scheme; `round → b` in every kernel | folded into the global round index `round = ((b·Hkv + c)·rounds_per_head + head_round)`; rounds dealt to clusters | knob-turn (PHASE0 refuses `batched` by axis only; indexing carries `b` from day 1) |
| `kv_head` (Hkv) | independent; **K/V are reuse-shared along `group_member` and `q_tokens`** — every query row of the group reads the whole K/V head | `kv_heads_per_round` | 1 — same reason as batch: a round's NEOs share one K/V stream | fixed by the scheme | part of the round index (above) | — |
| `group_member` (g query heads on one kv head) | independent — query heads never combine; K/V reuse-shared along it | `members_per_unit` | 1 per NEO unit; up to C units of a round are different members of the same kv head | host plan (`unit → (q_chunk, member)` = `(u // g, u % g)`, q-chunk-major) | the C NEOs of a cluster take C consecutive units of one kv head — members and q chunks of one head share one K/V read | knob-turn |
| `q_tokens` (Sqt query tile-rows) | independent — output rows never combine; K/V reuse-shared along it | `q_chunk_tiles` (tile-rows per NEO unit) | 1 on every PHASE0 shape — **derived**, not a literal: `clamp(floor(B·Hq·Sqt / (C·num_clusters)), 1, Sqt)` then clamped by the fit predicate; it rises with problem size (occupancy first, then coarsest) and every unit beyond occupancy would only cost a K/V re-read | host plan `SdpaPlan.q_chunk_tiles` (one source → CT arg) | units dealt to NEOs (C per round), whole rounds to clusters (`split_work_to_cores`, `row_wise=True`) | knob-turn |
| `k_tokens` (Skt key tile-rows) | **dependent** — the softmax (max, sum) and the second product reduce over it | `k_chunk_tiles` | `Skt` (whole head resident → one chunk) on every PHASE0 shape — derived as the coarsest that fits: `min(Skt, kc_fit(q_chunk_tiles))` | host plan `SdpaPlan.k_chunk_tiles` → CT arg | **not split across cores**: walked sequentially inside a NEO (cheap accumulate: online softmax) | cross-core split = scheme-change (deferred regime `split_kv_across_clusters`) |
| `d_head` (Dt) | dependent for Q·Kᵀ (contracted), independent for P·V (output columns) | `d_block_tiles` | `Dt` (whole) | host plan `SdpaPlan.d_block_tiles` → CT arg (Phase 0 asserts `== Dt`) | not split — a NEO unit owns all output columns of its rows | scheme-change (deferred `wide_head_dim`) |

Stage data, re-run through the same table (no stage is left on a designated core — there is no collector, coordinator or combine core):

| Stage data | Axes it has | Block | Who computes it |
|------------|-------------|-------|-----------------|
| scores / probabilities `S, P` | q_tokens × k_tokens | `[q_chunk_tiles × k_chunk_tiles]` per NEO unit | the NEO owning the unit (independent along q; dependent along k, consumed in the same NEO) |
| row state `m, l, α` | q_tokens | `[q_chunk_tiles]` | owning NEO |
| output accumulator `O` | q_tokens × d_head | `[q_chunk_tiles × Dt]` | owning NEO |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_q_block` | `Q_BUFFER_DEPTH` | 2 blocks per NEO | next round's Q lands while this round computes |
| `cb_k_block`, `cb_v_block` | `KV_BLOCKS_PER_READER` (blocks per producer region of the ALL ring) | 1 → R = 4 blocks in flight cluster-wide | each reader thread prefetches its own block (block n from thread n mod R), so up to R chunks are in flight while the NEOs compute |
| `cb_out` | `OUT_BUFFER_DEPTH` | 2 tiles per NEO | normalize/pack overlaps the writer's DRAM write |
| state / intermediate rings | none (full block) | — | sequential compute phases of one NEO cannot pipeline; full block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size ≤ `DFB_MAX_RING_BYTES` = 1,048,560 B (target page *Cluster model*; a part limit with no run-time query) | `k_chunk_tiles·d_block_tiles` (K, V rings: `R·kc·Dt·T_in`), `q_chunk_tiles·k_chunk_tiles` (S, P rings: `C·qc·kc·T_acc`), `q_chunk_tiles·Dt` (Q ring `C·Q_BUFFER_DEPTH·qc·Dt·T_in`, O ring `C·qc·Dt·T_acc`) | `kc_fit`/`qc` solve in Work Distribution; if `qc = kc = 1` still violates → `NotImplementedError` (wide-head-dim regime) | program build fails |
| ALL consumer fan-out ≤ 4 (`DataflowBuffer.md:147`) | compute threads C sharing K/V | `C ≤ 4` (plan validation) | program build refusal |
| Indexed unpack on a STRIDED ring (`llk_unpack_AB_matmul_api.h:191-194`, `llk_unpack_AB_reduce_api.h:122-125`) | every tile index handed to `matmul_tiles` / `reduce_tile` | index passed as `local_idx · ring.get_consumer_stride_tiles()` (1 on ALL rings, C on STRIDED rings) | — |
| Indexed window must not straddle the ring end | every ring read by index (`cb_q_block`, `cb_k_block`, `cb_v_block`, `cb_scores`, `cb_probs`) | capacity per NEO/region is an exact multiple of the nominal block, and **every push/pop is the nominal block** (`q_chunk_tiles·Dt`, `k_chunk_tiles·Dt`, `q_chunk_tiles·k_chunk_tiles`) even when the ragged/causal extent is smaller — only the work narrows | — |
| State rings rotated in place with exact capacity (`cb_row_max`, `cb_row_sum`, `cb_out_accum`) | — | every op that rotates a state ring pops the consumed tiles **before** reserving the produced ones (per DEST section) | deadlock (reserve on a full ring) or a front that drifts off the slice start (breaks the straddle rule) |
| 32-bit DEST = 4 tiles per section in half-sync (`ckernel_trisc_common.h:56-57,145-149`; `dest_helpers.hpp:98`) | sub-blocks inside score / PV / max / sum sections (not the block) | sub-block ≤ `DEST_AUTO_LIMIT` (4), ≤ 3 output tiles where a section also holds a mask or an α tile | DEST overrun — corrupt results |
| ALL ring with R producers: block n comes from producer n mod R, every producer pushes the same count (`unified_matmul_reader.cpp:10-12`, `DataflowBuffer.md` Part C ALL) | number of K/V blocks a cluster pushes | total padded to a multiple of R with credit-only blocks that the NEOs pop without computing | hang at kernel exit (unbalanced counters) |
| Compute binary ≤ 24,576 B (target page) | code volume of the compute kernel (not an extent) | `causal` is a compile-time arg: the mask code is compiled out when 0; raw inits shared between ops | build failure |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `resident_kv` — one key chunk per round (plain stable softmax) | **built** (every PHASE0 shape) | `k_chunk_tiles == Skt` (i.e. `kc_fit(q_chunk_tiles) ≥ Skt`) | per NEO `[q_chunk_tiles × Skt]` scores; K/V block `[Skt × Dt]` | Minimum (each input crosses DRAM once, output once) is met for Q and O; K/V cross DRAM **once per round** = `rounds_per_head` times per kv head (1 on shapes 1–4, 6; 3 on shape 5; 8 on shape 7), because rounds of one head run on different clusters | per round: one matmul init per score/PV block, one reduce init per max/sum block, one exp init, one pack_init per output-ring switch, one K/V ring handshake per chunk; larger `q_chunk_tiles` divides the K/V re-read count |
| `streamed_kv_online` — several key chunks, online rescale | **built** (same kernels; engaged when K/V of a head does not fit, e.g. S ≥ 4096 model calls) | `k_chunk_tiles < Skt` | per NEO `[q_chunk_tiles × k_chunk_tiles]`; K/V block `[k_chunk_tiles × Dt]`; `num_k_chunks = ceil(Skt / k_chunk_tiles)` | same crossings as `resident_kv` (K/V bytes per round do not depend on `kc`); causal rounds skip chunks past their diagonal | per chunk: the rescale ops (`α`, `O ·= α`, `l = α·l + …`) and one K/V handshake; bigger `kc` = fewer rescale passes |
| `kv_resident_across_rounds` — a cluster with several rounds of one kv head keeps K/V resident and reads it once | **deferred** — on the 32-cluster grid no INPUTS shape gives a cluster more than one round (max 8 rounds total, shape 7), and for large shapes `q_chunk_tiles` grows until rounds ≈ clusters, so the regime serves only L1-capped `q_chunk_tiles`; reachable: the ALL K/V ring's producer simply skips re-reading when the next round has the same `(b, c)` and `kc == Skt` | `kc == Skt ∧ rounds of the same (b,c) consecutive on a cluster` | as `resident_kv` | K/V crossings per head drop from `rounds_per_head` to `clusters_per_head` | the K/V read + handshake per round |
| `kv_multicast_across_clusters` — one injector reads each K/V chunk, multicasts it to every cluster serving that kv head | **deferred** — stepping stone built: the per-cluster ALL ring is the receive side; only the reader's producer side changes (DRAM read → multicast receive + semaphore). Not built in Phase 0 because no Python `ProgramSpec` reference on this part exercises a cross-cluster multicast (quasar_op_examples README, "Not covered: multicast between clusters") and its gain is bounded by injector serialization (catalog `shared_input_reuse`: 1.71× for an 11× read cut) | `clusters_per_head > 1` | as built | K/V DRAM crossings → **1 per head** (the minimum); adds NoC multicast payload `(clusters_per_head − 1)·|K+V|` and one semaphore round trip per chunk. Shape 7: K+V DRAM 1,048,576 B → 131,072 B | multicast + semaphore per chunk |
| `split_kv_across_clusters` — the dependent key axis split over clusters, partial `(m, l, O)` combined | **deferred** (scheme-change) — needed only where `B·Hq·Sqt < C·num_clusters` (short sequences, where the whole op is small); the model calls (4096–12288 tokens) fill the grid along q. Reachable: the combine is exactly the built `row_max/row_sum/rescale` block ops applied to a peer's `(m, l, O)` triple | `B·Hq·Sqt < C·num_clusters` | per cluster `[q_chunk_tiles × Skt/ksplit]` | K/V DRAM unchanged in total (each cluster reads its slice); adds Q re-read `×ksplit` and NoC traffic `(ksplit−1)·q_chunk_tiles·(Dt + 2)·T_acc` per unit + semaphores | one combine round per unit |
| `wide_head_dim` — D blocked (Q·Kᵀ accumulated over d-blocks in DEST, O split in d-blocks) | **deferred** — the only INPUTS entry it serves is `(2,1,32,4096)`, which is `batched` (outside PHASE0); a single-sequence call that fails the fit predicate is refused with `NotImplementedError`. Reachable: `d_block_tiles` is already a CT knob that Phase 0 pins to `Dt` | fit fails at `qc = kc = 1` (Dt·T sizes exceed `DFB_MAX_RING_BYTES` or the budget) | `[q_chunk_tiles × k_chunk_tiles]`, O in `[q_chunk_tiles × d_block_tiles]` | S recomputed per d-block or Q·Kᵀ streamed over d: +Q/K reads per d-block | — |
| `per_neo_kv_streams` — each NEO reads its own K/V copy through STRIDED rings | **rejected** — superseded by the ALL-shared K/V ring (4× fewer K/V DRAM crossings, same kernels otherwise) | — | — | K/V crossings ×C | — |
| `materialized_scores` — S·S scores to DRAM, softmax, then P·V | **rejected** — needs several dispatches or an S² DRAM round trip; superseded by the fused block pipeline | — | — | +2·B·Hq·S²·4 B DRAM | — |
| `two_pass_kv` — pass 1 over K for max/sum, pass 2 over K and V | **rejected** (dead end) — re-reads K per round; online softmax gets the same result in one pass | — | — | K crossings ×2 | — |

Regime selection is one function of the host plan (`streamed = plan.k_chunk_tiles < Skt`); both built regimes run the same kernels. **Regime-pinned tests are required**: the builder `create_program_artifacts(..., q_chunk_tiles=None, k_chunk_tiles=None, threads=THREAD_PLAN)` accepts knob overrides (clamped to legality, never above the fit); tests force (a) `streamed_kv_online` on `(1,2,512,64)` with `k_chunk_tiles ∈ {4, 3}` (3 → ragged last chunk), causal and not; (b) `q_chunk_tiles = 2` on the same shape (multi-row unit: full-mask tiles inside the diagonal chunk); (c) the resident default. Each compares against the spec's `reference`.

### Traffic ranking

Counts per round of one kv head (`T_in` = input tile bytes, 2,048 for bfloat16): Q block `C·qc·Dt`, K and V `Skt·Dt` each (causal: up to the round's last diagonal chunk), O `C·qc·Dt`.

| # | Candidate split | DRAM crossings | Cross-core | Verdict |
|---|-----------------|----------------|------------|---------|
| 1 | **q-units (member × q chunk) of one kv head dealt C per cluster, K/V shared via ALL ring** | Q 1×, O 1×, K/V `rounds_per_head`× | none | **chosen**: fills C NEOs per cluster with one K/V read; 4× fewer K/V bytes than #2 at identical compute parallelism |
| 2 | q-units dealt one per NEO with private K/V streams | Q 1×, O 1×, K/V `units_per_head`× | none | rejected (`per_neo_kv_streams`) |
| 3 | `(b, h)` per cluster, whole S | K/V once per query head | none | is #1 at its coarsest `q_chunk_tiles`; under-fills the grid on every PHASE0 shape (≤ 8 heads) |
| 4 | #1 + read-once/multicast of K/V across the clusters of a head | **K/V 1× (minimum)** | multicast payload `(clusters_per_head−1)·|K+V|` | cheapest traffic; deferred (`kv_multicast_across_clusters`) |
| 5 | split the dependent key axis over clusters + combine | K/V 1× per unit group, Q ×ksplit | partial `(m, l, O)` per unit | deferred (`split_kv_across_clusters`); only pays when #1 under-fills |
| 6 | split d_head | — | combine of partial scores | rejected: d is contracted in Q·Kᵀ; partial scores (S² sized) would cross the NoC |

Occupancy (not part of the ranking): PHASE0 engages 1, 1, 1, 4, 6, 2, 8 clusters (shapes in INPUTS order); every unit has its own NEO — the independent axes are fully spread; the remaining idle clusters are what regime 5 would use.

**Stall shadows.** (1) A NEO waits for K before Q·Kᵀ and for V before P·V: the reader issues K block n before V block n, so V's DRAM latency is hidden behind score/max/exp/sum of chunk n. (2) Row max → exp is a true dependence (nothing to fill). (3) The normalization `/l` is deferred past P·V — legal because `l` is a per-row scalar, `(P/l)·V = (P·V)/l`, and the online rescale is legal because `exp(s − m₂) = exp(s − m₁)·exp(m₁ − m₂)`; this lets P·V start before the row sum is final and is what makes one pass over K/V possible. It reorders floating-point operations relative to `reference` (normalize-after-product); the tolerance (PCC 0.999, rel-RMS 0.05) dwarfs the reorder (one rounding of `O/l` vs `P/l`).

### Block schedule

Per cluster: `num_rounds_this_core` rounds; in a round NEO t owns unit `t` (or NEO 0's unit when the round is short); per round `num_k_chunks(round)` K/V blocks are delivered; NEO t computes the prefix of chunks it needs (`chunk_idx ≤ last_chunk(t)`) and only pops the rest.

```cpp
for (uint32_t round_idx = 0; round_idx < num_rounds_this_core; ++round_idx) {
    load_q_block(round_idx);                                    // reader → cb_q_block (this NEO's unit)
    for (uint32_t chunk_idx = 0; chunk_idx < num_k_chunks(round_idx); ++chunk_idx) {
        load_kv_block(chunk_idx);                               // reader → cb_k_block, cb_v_block (ALL)
        if (chunk_idx <= last_chunk_of_this_neo(round_idx)) {
            score_block(chunk_idx);       // S = scale·Q·Kᵀ (+mask) → cb_scores; pops K
            row_max_block(chunk_idx);     // m_new (, α) → cb_row_max (, cb_rescale)
            exp_block(chunk_idx);         // P = exp(S − m_new) → cb_probs; pops S
            row_sum_block(chunk_idx);     // l = (α·l +) rowsum(P) → cb_row_sum
            rescale_accum_block(chunk_idx); // chunk_idx > 0 only: O ·= α (pops α)
            pv_block(chunk_idx);          // O (+)= P·V → cb_out_accum; pops P, V
        } else {
            skip_kv_block(chunk_idx);     // pop K and V without compute (causal tail of a round)
        }
    }
    normalize_block(round_idx);           // out = O · (1/l) → cb_out; pops O, l, m
    store_out_block(round_idx);           // writer
}
drain_padding_kv_blocks();                // pop the credit-only blocks that round the total to R
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_q_block` | `[qc × Dt]` tiles of one query head (NEO-private) | — | one reserve/push of the block per round per NEO; reads tagged trid 0, one barrier per block |
| `load_kv_block` | `[kc × Dt]` of K, then `[kc × Dt]` of V (shared by the C NEOs) | — | one reserve/push per block per chunk, by reader thread `n mod R` (n = cluster-wide block counter); one barrier per block. Causal: only key tile-rows `< min(Skt, round_last_q_tile + 1)` are read (the push stays the nominal block) — the resident regime's K/V reads shrink to the round's diagonal |
| `score_block` | `[qc × kc_eff]` output tiles, inner `Dt` | Q block (whole round), K block | one `matmul_init(…, transpose=1)` per block entry, re-issued only after the SFPU scale/mask of each DEST section; one pack_init(`cb_scores`); scale applied in DEST per section (no extra pass); mask only on tiles with `k_tile ≥ q_tile` (causal) |
| `row_max_block` | `[qc × kc_eff]` → `[qc]` | S block | one `reduce_init<MAX,ROW>` per block; one DEST section per q tile-row (reduce over kc_eff tiles + fold with m_old + α) |
| `exp_block` | `[qc × kc_eff]` | m_new | one sub-bcast-col + exp init per block; one pack_init(`cb_probs`) |
| `row_sum_block` | `[qc × kc_eff]` → `[qc]` | P block | one `reduce_init<SUM,ROW>` per block; one DEST section per q tile-row |
| `rescale_accum_block` | `[qc × Dt]` | α | one mul-bcast-col init per block; chunk 0 skips it |
| `pv_block` | `[qc × Dt]` output, inner `kc_eff` | P block, V block | one `matmul_init(…, transpose=0)` per block; per DEST section: reload O_old (chunk > 0), accumulate, pack |
| `normalize_block` | `[qc × Dt]` | 1/l | one recip pass over `[qc]`, one mul-bcast-col init per block; pack_init(`cb_out`) |
| `store_out_block` | `[qc × Dt]` (clamped rows rewrite the last real row) | — | implicit-sync writes, one per tile; tag handed back once at kernel end |

**Row clamp (ragged q chunk):** every unit has exactly `qc` rows; unit row i is query tile-row `min(q_tile_start + i, Sqt − 1)` — rows past the end recompute the last real row (reader loads it again, writer rewrites the same page with the same bytes), so every NEO pushes the nominal counts on every ring, which the strict-rotation writer and the ALL/STRIDED readers require (an implicit-sync writer cannot discard an entry). `kc_eff(chunk)` = `min(kc, Skt − chunk·kc, causal ? last_q_tile_of_unit + 1 − chunk·kc : ∞)` (`last_q_tile_of_unit` after the clamp). A ragged key extent narrows the work only; pushes/pops stay nominal.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (`resident_kv` block = whole head) | the first chunk's compute cannot start until the whole `[Skt × Dt]` K block has landed; with one round per cluster there is nothing else to overlap | `k_chunk_tiles ∈ {4, 8}` (streamed regime, prefetch depth R) vs `Skt` |
| Grid synchronization / plan | 4 NEOs per cluster on few clusters vs. the same NEO count spread over more clusters (more K/V reads, less per-cluster DM contention) | plan `(4,4,2)` vs `(2,4,2)` vs `(1,4,1)` vs `(1,1,1)` (structural count on ttsim: K/V tiles read per cluster; model cycles on ttsim-perf) |
| Causal load balance | units are q-chunk-major, so the cluster holding the last q chunks does up to Sqt× the work of the first | zig-zag pairing of q chunks (i, nqc−1−i) into one round |
| `q_chunk_tiles` | derived to 1 on PHASE0 shapes for occupancy; a larger unit halves K/V reads at half the NEOs | `q_chunk_tiles = 2` on `(1,2,512,64)` |
| Row sum realization | `reduce_tile<SUM>` per tile of the row | pairwise `add_tiles` into DEST then one reduce (catalog `row_reduce_accumulate`, 2.9× at W ≥ 4) |
| Rescale fused into `pv_block` | `rescale_accum_block` is one extra pass over `O` per chunk (streamed regime only) | reload O, multiply by α with `sfpu_mul_bcast_col` in DEST, then accumulate P·V in the same section |
| `1/l` scope | recip over full tiles though only column 0 is used | column-0-scoped SFPU (catalog `sfpu_tile_scope`) |
| Score sub-block shape | 4 output tiles per DEST section in 1×4 | 2×2 when `qc ≥ 2` (catalog `matmul_output_subblock`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_q_block` | Q dtype (bfloat16) tiles | reader, **explicit sync**: `reserve_back(qc·Dt)`, one `noc.async_read<TXN_ID>(acc_q, ring, tile_bytes, {.page_id}, {.offset_bytes = k·tile_bytes}, {.trid = 0})` per tile, `async_read_barrier<TXN_ID>({.trid=0})`, `push_back(qc·Dt)` | STRIDED: reader thread p serves NEOs p, p+R, … (R = C: thread t ↔ NEO t). A padded NEO reads NEO 0's pages |
| DRAM → `cb_k_block`, `cb_v_block` | input dtype | reader, explicit sync, whole block per push; block n (cluster-wide counter over all rounds, K then V of the same chunk) by reader thread `n mod R`; total padded to a multiple of R with credit-only blocks | **ALL** consumer: each NEO sees every entry (stride 1), K/V read once per cluster per round |
| reader → `cb_constants` | bfloat16 | reader writes the tiles itself once at kernel start (explicit sync). Entry 0: reduce scaler via `calculate_and_prepare_reduce_scaler<cb_constants, PoolType::MAX, ReduceDim::REDUCE_ROW>` (1.0 row-0 fill; the identical tile serves SUM). Causal only, entry 1: diagonal mask (`0` where `c ≤ r`, `−inf` = 0xFF80 elsewhere), entry 2: full mask (all `−inf`), both in tile face order (faces 16×16: f0 rows 0–15/cols 0–15, f1 rows 0–15/cols 16–31, f2 rows 16–31/cols 0–15, f3 rows 16–31/cols 16–31) | STRIDED, one copy per NEO. A reader thread pushes entry k to each of its NEOs before entry k+1 (strict rotation), so each NEO's slice reads scaler, diag, full |
| compute (per NEO) | Float32 pages for every internal ring (32-bit DEST), `UnpackToSrc` for all of them | raw LLK matmul/reduce with stride-corrected indices; eltwise chain for sub-bcast+exp, mul-bcast, recip; `pack_init` on every output-ring switch | see Block Operation Realization |
| `cb_out` → DRAM | output dtype | writer, **implicit sync**: one `noc.async_write<NocOptions::TXN_ID>(ring, acc_out, {}, {.page_id})` per tile, W = 2 threads, thread w drains NEOs w, w+W in rotation; last statement `noc_async_write_set_trid(0, noc.get_noc_id())` | padded NEOs rewrite NEO 0's pages with the same bytes |
| reader end | — | every transfer already tagged trid 0 (explicit kernel); `noc_async_read_set_trid(0, noc.get_noc_id())` as last statement anyway | ring protocols per target page |

Host: reader `hw_config.config_2xx = DataMovement2XXConfig(disable_dfb_implicit_sync_for=[every reader ring])` (as `toy_thread_plan_program_artifacts.py:68`); writer keeps implicit sync. Compute `ComputeHardwareConfig(fpu_math_fidelity, enable_32_bit_dest=fp32_dest_acc_en, sfpu_precision_mode, unpack_modes={each Float32 ring: UnpackMode.UnpackToSrc})` — bfloat16 rings keep the default unpack.

Placement as a TARGET axis: `memory_config = L1` for Q is a **knob-turn** (the `TensorAccessor` already addresses either buffer; no ring change); `sharding` has only `interleaved` in TARGET. `batched` is a knob-turn (round index already carries `b`). `h_non_aligned` is a knob-turn: a third mask tile (key padding: `−inf` on columns ≥ `S mod 32`), added on the last key tile of every row in `score_block` (R13, whatever `is_causal`), reusing the mask path; padding query rows are computed and written (nobody reads them; every padding row still attends key 0, so no NaN). `float32` is a knob-turn (tile bytes and ring formats follow the dtype; R4 refusal stays). `fp32_dest_acc_en = False` is a knob-turn (internal ring formats derive from the flag: bfloat16; DEST 8 tiles).

Deferred cross-cluster contracts (so the structure stays reachable): `kv_multicast_across_clusters` — the injector cluster's reader multicasts the filled ALL-ring block to the same ring offset on the receivers, gated by a `receivers_ready` / `data_ready` semaphore pair per reader thread (as the Quasar matmul does, `unified_matmul_reader.cpp:106-119`); receivers' readers push the block without a DRAM read. `split_kv_across_clusters` — each cluster pushes `(m, l, O)` of a unit to the owner cluster's state rings; the owner runs `row_max_block`/`row_sum_block`/`rescale_accum_block` on them.

### Structural impossibilities

None beyond the specification's (its INVALID is empty, and nothing in this design adds one).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | NEO unit = (`b`, kv head `c`, q chunk, group member): `q_chunk_tiles` query tile-rows × `Dt` of one query head. Round = C consecutive units of one `(b, c)`, units ordered q-chunk-major (`u → q_chunk = u // g, member = u % g`, `h = c·g + member`) |
| Grid | `device.compute_with_storage_grid_size()` (8×4 = 32 clusters on quasar); rounds dealt with `ttnn.split_work_to_cores(grid, total_rounds, row_wise=True)`; clusters engaged = `min(total_rounds, num_clusters)` |
| Per-core work | RT args `first_round`, `num_rounds`; every kernel derives `(b, c, head_round) = divmod chain of round`, its unit, `q_tile_start` (rows clamped to `Sqt − 1`), the round's `num_k_chunks` and its own last chunk from CT args (`C, R, W, g, Hq, Hkv, Sqt, Skt, Dt, qc, kc, nqc, rounds_per_head, causal`) |
| Remainder | `nqc = ceil(Sqt/qc)`, `units_per_head = g·nqc`, `rounds_per_head = ceil(units_per_head / C)` — the last round of each head is padded (a NEO without a unit recomputes NEO 0's unit; its writer rewrites identical bytes); `total_rounds = B·Hkv·rounds_per_head`, split two-group by `split_work_to_cores` (whole rounds, never units). Ragged q chunk: row clamp (rows past `Sqt − 1` recompute the last real row); ragged key chunk `kc_eff` narrows work only. All tile counts `ceil`, per image |
| Bound | **compute** — per K/V tile a cluster reads, each of its 4 NEOs performs `qc` tile-matmuls for Q·Kᵀ and `qc` for P·V plus SFPU exp / FPU reduces over `qc·kc` score tiles (several passes over the score block and two matmuls per unit) |
| Thread plan | `THREAD_PLAN = (R, C, W) = (4, 4, 2)` — the page's plan for a compute-bound op that reads more than it writes (three inputs, K/V the bulk); one host constant, its counts passed as CT args `reader_threads`, `compute_threads`, `writer_threads`; kernels size loops from them and `get_my_thread_id()` |
| Unit dealt to compute threads | the NEO unit (larger than a ring entry): every NEO takes the same number of entries per round on every ring a DM thread touches — padded last round per kv head (recompute NEO 0's unit); whole rounds of C units dealt to clusters. Rings are capacity-per-NEO (STRIDED) or capacity-per-producer-region (ALL); see Circular Buffers |
| Tests the plan needs | (1) `test_each_neo_computes_its_own_unit`: a test-only CT flag `tag_threads` makes each NEO pack `get_my_thread_id() + 1` into its output tiles; the test asserts the unit→NEO map of the host plan on a shape with ≥ C units (e.g. `(1,2,512,64)`, and `(1,4,64,64)` for padded rounds); (2) results identical for `threads=(1,1,1)`, `(2,4,2)`, `(4,4,2)`; (3) one device session: another program, then this op, then `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` — checks the ring-protocol/tag hand-back; (4) the regime-pinned tests above |

Fit predicate (host, one function, after the support check). With `T_in` = input tile bytes, `T_acc` = 4,096 when `fp32_dest_acc_en` else 2,048, `T_out` = output tile bytes, `budget = ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES`, `CONST_TILES = 3 if causal else 1`:

```
fixed(qc)  = C·Q_BUFFER_DEPTH·qc·Dt·T_in + C·CONST_TILES·T_in
           + C·3·qc·T_acc + C·qc·Dt·T_acc + C·OUT_BUFFER_DEPTH·T_out
per_kc(qc) = 2·R·KV_BLOCKS_PER_READER·Dt·T_in + 2·C·qc·T_acc
kc_fit(qc) = min( floor((budget − fixed(qc)) / per_kc(qc)),
                  floor(DFB_MAX_RING_BYTES / (R·KV_BLOCKS_PER_READER·Dt·T_in)),
                  floor(DFB_MAX_RING_BYTES / (C·qc·T_acc)) )
qc  = clamp(floor(B·Hq·Sqt / (C·num_clusters)), 1, Sqt)
while qc > 1 and (kc_fit(qc) < min(Skt, K_CHUNK_FLOOR_TILES) or ring caps on qc fail): qc -= 1
kc  = min(Skt, kc_fit(qc))
refuse (NotImplementedError, wide-head-dim) if kc < 1 or C·Q_BUFFER_DEPTH·qc·Dt·T_in > DFB_MAX_RING_BYTES
                                                 or C·qc·Dt·T_acc > DFB_MAX_RING_BYTES
```

`K_CHUNK_FLOOR_TILES = 4` keeps a coarse `q_chunk_tiles` from starving the key chunk below the granularity floor. Worked on quasar: every PHASE0 shape gives `qc = 1`, `kc = Skt` (resident); the largest, `(1,2,512,64)`, uses 1,204,224 B of 3,963,328 (l1_ledger.md). `S = 4096, D = 128, Hq = 16` (a model call): `qc = 16`, `kc_fit(16)` from the S/P ring cap `1,048,560/(4·16·4096) = 4` → streamed, `kc = 4`.

## Circular Buffers

Tile bytes: `T_in` (input dtype), `T_acc` (Float32 when `fp32_dest_acc_en`, else bfloat16), `T_out` (output dtype). Indices are any distinct slots; names are the identifiers.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_q_block` | 0 | `T_in` | `Q_BUFFER_DEPTH·qc·Dt·C` | spans q (qc) and d (Dt) per NEO; streams over rounds | input dtype | reader | compute | reader: explicit (reserve/push, tagged reads); compute: STRIDED consumer, indices × stride | per round |
| `cb_k_block` | 1 | `T_in` | `R·KV_BLOCKS_PER_READER·kc·Dt` | spans k (kc) and d (Dt); one block per producer region | input dtype | reader | compute | reader: explicit, block per push; compute: **ALL** consumer | per chunk (popped after `score_block`) |
| `cb_v_block` | 2 | `T_in` | `R·KV_BLOCKS_PER_READER·kc·Dt` | as K | input dtype | reader | compute | as K | per chunk (popped after `pv_block`) |
| `cb_constants` | 3 | `T_in` | `CONST_TILES·C` (`CONST_TILES` = 3 causal, 1 otherwise) | constant: scaler (+ diag mask, full mask) per NEO; index 0 needs no stride in `reduce_tile`; mask entries read by `copy_tile` | bfloat16 | reader | compute | reader explicit; STRIDED | whole kernel (popped at kernel end) |
| `cb_scores` | 4 | `T_acc` | `qc·kc·C` | spans q, k per NEO (row max then exp read it whole) | Float32 | compute | compute | — (compute-internal) | `score_block` → `exp_block` |
| `cb_probs` | 5 | `T_acc` | `qc·kc·C` | spans q, k per NEO (row sum and P·V read it whole) | Float32 | compute | compute | — | `exp_block` → `pv_block` |
| `cb_row_max` | 6 | `T_acc` | `qc·C` | spans q; rotated in place (pop m_old, push m_new) | Float32 | compute | compute | — | whole round |
| `cb_rescale` | 7 | `T_acc` | `qc·C` | spans q; α of this chunk | Float32 | compute | compute | — | `row_max_block` → `rescale_accum_block` (chunk > 0 only) |
| `cb_row_sum` | 8 | `T_acc` | `qc·C` | spans q; rotated in place (l, then 1/l at round end) | Float32 | compute | compute | — | whole round |
| `cb_out_accum` | 9 | `T_acc` | `qc·Dt·C` | spans q, d; rotated in place per chunk | Float32 | compute | compute | — | whole round |
| `cb_out` | 10 | `T_out` | `OUT_BUFFER_DEPTH·C` | streams (one tile at a time to the writer) | output dtype | compute | writer | writer: implicit sync, tagged | per round |

Push = wait for every ring: `cb_q_block` `qc·Dt` per round per NEO; K/V `kc·Dt` per block (credit-only padding blocks included on both sides); constants once; `cb_scores`/`cb_probs` `qc·kc` per computed chunk (nominal, even when `kc_eff < kc`); state rings one tile per row per rotation; `cb_out` `qc·Dt` per round per NEO (row clamp and padded units keep every NEO at the nominal count). Each ring has one producer kernel and one consumer kernel (compute-internal rings: the compute kernel on both sides, as `toy_row_passes`' `acc`). 11 rings ≤ 24 per NEO.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_q_block` | `[qc × Dt]` | dataflow (raw NoC) | DRAM Q | `cb_q_block`, `qc·Dt` | held by compute for the round, popped after the last chunk |
| 2 | `load_kv_block` | `[kc × Dt]` ×2 | dataflow | DRAM K, V | `cb_k_block`, `cb_v_block`, `kc·Dt` each | — |
| 3 | `score_block` | `[qc × kc_eff]`, inner `Dt` | raw matmul + SFPU | `cb_q_block` (held), `cb_k_block` (waited, popped at end), `cb_constants` (held) | `cb_scores`, `qc·kc` (nominal) | K popped |
| 4 | `row_max_block` | `[qc × kc_eff]` → `[qc]` | raw reduce + SFPU | `cb_scores` (held), `cb_constants` (held), `cb_row_max` (chunk > 0: popped per row) | `cb_row_max` `qc`, `cb_rescale` `qc` (chunk > 0) | S still held |
| 5 | `exp_block` | `[qc × kc_eff]` | **helper** (eltwise chain) | `cb_scores` (held, popped `qc·kc` at end), `cb_row_max` (held, Col broadcast) | `cb_probs`, `qc·kc` | S popped |
| 6 | `row_sum_block` | `[qc × kc_eff]` → `[qc]` | raw reduce + SFPU | `cb_probs` (held), `cb_constants`, `cb_row_sum` + `cb_rescale` (chunk > 0) | `cb_row_sum` `qc` | P held |
| 7 | `rescale_accum_block` (chunk > 0) | `[qc × Dt]` | **helper** (eltwise `mul`, Col broadcast, in place) | `cb_out_accum` (pop per tile), `cb_rescale` (held, popped at end) | `cb_out_accum` `qc·Dt` | α popped |
| 8 | `pv_block` | `[qc × Dt]`, inner `kc_eff` | raw matmul (+ copy reload) | `cb_probs` (held, popped `qc·kc` at end), `cb_v_block` (popped at end), `cb_out_accum` (chunk > 0: popped per section) | `cb_out_accum` `qc·Dt` | P, V popped |
| 9 | `normalize_block` | `[qc]` then `[qc × Dt]` | **helper** (eltwise `unary<Recip>` in place; `mul` Col broadcast) | `cb_row_sum` (rotated to 1/l, then held and popped), `cb_out_accum` (pop per tile), `cb_row_max` (popped) | `cb_out`, `qc·Dt` | round state empty |
| 10 | `store_out_block` | `[qc × Dt]` (clamped rows rewrite the last real row) | dataflow | `cb_out` | DRAM out | — |

Realization contracts the implementer must keep:

- **Stride rule**: every tile index passed to `matmul_tiles`, `reduce_tile` or `copy_tile` on a ring is `local_idx · ring.get_consumer_stride_tiles()` for matmul/reduce; `copy_tile`, eltwise binary and `pack_tile` take plain local indices.
- **Score section** (3): `matmul_init(cb_q_block, cb_k_block, /*transpose=*/1)` makes `matmul_tiles` multiply by the transposed K tile (in1 → SrcA, the only transposable source: `llk_unpack_matmul.h:121`); for each output tile accumulate `Dt` `matmul_tiles(Q[i·Dt+d], K[j·Dt+d])` into one DEST slot; then `binop_with_scalar_tile_init(); mul_unary_tile(slot, scale_bits)`; then, causal, for tiles with absolute `k_tile == q_tile` (diag) or `k_tile > q_tile` (full): copy the mask tile into a spare slot and `add_binary_tile(slot, mask_slot, slot)` — scale before mask, so the mask is never scaled (negative scales stay correct); pack the section to `cb_scores` at `i·kc + j`. Tiles past `kc_eff` are not computed.
- **Row max** (4): per q tile-row one DEST section: `reduce_tile<MAX, REDUCE_ROW>` over the row's `kc_eff` tiles into slot 0 (column 0 valid; −inf entries are ignored by max); chunk > 0: copy `m_old` (front of `cb_row_max`, then pop) into slot 1, `binary_max_tile(0, 1, 0)` → `m_new`, `sub_binary_tile(1, 0, 1)`, `exp_tile(1)` → α; pack `m_new` → `cb_row_max`, α → `cb_rescale` (pack_init per target switch). All copies precede the first SFPU op of the section.
- **Row sum** (6): per q tile-row: `reduce_tile<SUM, REDUCE_ROW>` over `kc_eff` tiles of `cb_probs` into slot 0; chunk > 0: copy `l_old` (pop) to slot 1 and α to slot 2, `mul_binary_tile(1, 2, 1)`, `add_binary_tile(0, 1, 0)`; pack → `cb_row_sum`.
- **P·V** (8): `matmul_init(cb_probs, cb_v_block, 0)`; per DEST section of ≤ 4 output tiles `(i, d…)`: chunk > 0 → `copy_tile` the `O_old` tiles from the front of `cb_out_accum`, pop them; then `Σ_{j<kc_eff} matmul_tiles(P[i·kc+j], V[j·Dt+d])` accumulates on top; reserve, pack, push.
- **pack_init** before every pack into a ring other than the previous one.
- **In-place rotation**: `cb_row_max`, `cb_row_sum`, `cb_out_accum` pop consumed tiles before reserving their successors; the eltwise convenience helpers' in-place form (`convenience.hpp`, "pass the input CB as the output CB") is used for `rescale_accum_block` / the recip of `normalize_block` only if it does so — otherwise those two ops are raw `mul_tiles_bcast_cols` / `recip_tile` with explicit wait-pop-reserve-push.
- **Causal skip**: a chunk past `last_chunk_of_this_neo` is popped (K and V, `kc·Dt` each) without compute; credit-only padding blocks at kernel end likewise. `chunk 0` is always computed by every NEO (key 0 is attended by every query), so `m` is finite from the first chunk on and `exp(−inf − m) = 0` (accurate fp32 exp maps −inf → 0: `ckernel_sfpu_exp.h:215`).

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(cb_q_block, cb_k_block, cb_scores)` once at kernel start | — | — | — |
| `score_block` (product) | raw_api | `matmul_init`, `matmul_tiles` | `tt_metal/hw/inc/api/compute/matmul.h:113`, `:172` (Quasar branches `:123-126`, `:180-184`) | `matmul_init(cb_q_block, cb_k_block, 1)`; `matmul_tiles(cb_q_block, cb_k_block, qi·sq, kj·1, slot)` | `cb_q_block`, `cb_k_block` | DEST | `q_chunk_tiles`, `k_chunk_tiles` (loop extents), `d_block_tiles` (inner) |
| `score_block` (scale) | raw_api | `binop_with_scalar_tile_init`, `mul_unary_tile` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:157`, `:61` (Quasar path `:61-73`) | `mul_unary_tile(slot, scale_bits)` | DEST | DEST | — |
| `score_block` (mask) | raw_api | `copy_tile`, `add_binary_tile_init`, `add_binary_tile` | `tile_move_copy.h:126`; `eltwise_binary_sfpu.h:272`, `:103` | mask slot = `copy_tile(cb_constants, 1 (diag) or 2 (full), slot)` | `cb_constants` | DEST | — |
| `score_block` / all packs | raw_api | `pack_init`, `pack_tile` | `tt_metal/hw/inc/api/compute/pack.h:32`, `:87` | `pack_tile(slot, cb, out_idx)` | DEST | ring | — |
| `row_max_block`, `row_sum_block` (reduce) | raw_api | `reduce_init`, `reduce_tile`, `reduce_uninit` | `tt_metal/hw/inc/api/compute/reduce.h:66`, `:147`, `:96` | `<PoolType::MAX / SUM, ReduceDim::REDUCE_ROW>`; `reduce_tile(cb_scores, cb_constants, idx·stride, 0, slot)` | `cb_scores` / `cb_probs`, `cb_constants` | DEST | `k_chunk_tiles` (tiles per row) |
| `row_max_block` (fold) | raw_api | `binary_max_tile_init`, `binary_max_tile`; `sub_binary_tile`; `exp_tile_init`, `exp_tile` | `binary_max_min.h:145`, `:126`; `eltwise_binary_sfpu.h:276`, `:131`; `eltwise_unary/exp.h:40`, `:85` | slots 0/1 | `cb_row_max` | `cb_row_max`, `cb_rescale` | — |
| `exp_block` | helper | `compute_kernel_lib::sub` / `eltwise_chain` with `Exp<Approx::Exact>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:61`; `eltwise/api/chain.hpp:330` (BroadcastDim), `:264-290` (`TileAddressing::Strided`, `StridedTileRange{0, kc}`); `eltwise/unary/math.hpp:22` (Exp) | A = `cb_scores` Strided `{base 0, row_stride kc}`, caller-managed wait/pop; B = `cb_row_max` Col broadcast, Col mapping; out `cb_probs`; shape `IterationShape::grid(qc, kc_eff)` | `cb_scores`, `cb_row_max` | `cb_probs` | `q_chunk_tiles`, `k_chunk_tiles` (grid + row stride) |
| `row_sum_block` (fold) | raw_api | `mul_binary_tile`, `add_binary_tile` (+ inits) | `eltwise_binary_sfpu.h:68`, `:103`, `:268`, `:272` | slots 0/1/2 | `cb_row_sum`, `cb_rescale` | `cb_row_sum` | — |
| `rescale_accum_block` | helper | `compute_kernel_lib::mul` | `eltwise/api/convenience.hpp:70` | A = `cb_out_accum` PerTile/PerTile, B = `cb_rescale` Col broadcast (Col mapping, held), out = `cb_out_accum` (in place); `IterationShape::grid(qc, Dt)` | `cb_out_accum`, `cb_rescale` | `cb_out_accum` | `q_chunk_tiles`, `d_block_tiles` |
| `pv_block` | raw_api | `matmul_init`, `matmul_tiles`, `copy_tile` | `matmul.h:113`, `:172`; `tile_move_copy.h:126` | `matmul_init(cb_probs, cb_v_block, 0)`; `matmul_tiles(cb_probs, cb_v_block, (i·kc+j)·sp, (j·Dt+d)·1, slot)` | `cb_probs`, `cb_v_block`, `cb_out_accum` | `cb_out_accum` | `q_chunk_tiles`, `k_chunk_tiles`, `d_block_tiles` |
| `normalize_block` | helper | `compute_kernel_lib::unary<Recip<>>`, `compute_kernel_lib::mul` | `eltwise/api/convenience.hpp:87`, `:70`; `eltwise/unary/math.hpp:34` | recip `cb_row_sum` in place over `qc`; mul A = `cb_out_accum` (pop per tile), B = `cb_row_sum` Col broadcast; out `cb_out`; `IterationShape::grid(qc, Dt)` | `cb_row_sum`, `cb_out_accum` | `cb_out` | `q_chunk_tiles`, `d_block_tiles` |
| scaler tile | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:109` | `<cb_constants, PoolType::MAX, ReduceDim::REDUCE_ROW>(dfb)` (1.0, row-0 fill — same tile SUM uses) | — | `cb_constants` | — |
| loads / stores | raw_api | `Noc::async_read<NocOptions::TXN_ID>`, `async_read_barrier<TXN_ID>`, `async_write<TXN_ID>`, `noc_async_{read,write}_set_trid` | as used in `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader.cpp:29-45`, `toy_row_passes/kernels/writer.cpp` | trid 0 (reader); implicit (writer) | DRAM / rings | rings / DRAM | block extents set the page loops |

**Helpers considered and rejected** (every `raw_api` compute entry):

| Raw op | Helper considered | Mismatch (file:line) | Concrete reason |
|--------|-------------------|----------------------|-----------------|
| matmul in `score_block`, `pv_block` | `matmul_block` (`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`) | — | the matmul is written with LLK calls, not with `matmul_block` |
| reduce MAX/SUM over a held block | `compute_kernel_lib::reduce<…, WaitUpfrontNoPop>` (`reduce_helpers_compute.hpp:673`) | — | `WaitAndPopPerTile` would consume S before `exp_block` needs it; the reduce is raw `reduce_tile` with `idx·stride` |
| max / α fold, l fold | `BinaryMax` (`eltwise/binary/sfpu/minmax.hpp:15`); `binary_sfpu<…>` | — | the fold mixes a reduce output already in DEST with a reloaded tile, which the chain (CB → DEST per element) cannot express without packing the reduce result out first |
| scale / mask in DEST after matmul | `eltwise_chain` with `MulUnary` (+ `DestReuseBinary` for the mask) | `ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl:3366` (the chain opens its own `tile_regs_acquire` window per walk step) | the score tile is produced in DEST by the raw matmul of the same section; a chain cannot continue a DEST window it did not open, so using it would need an extra pack + reload of every score tile |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| `exp_block` | sub (then exp) | `cb_scores`: All | `cb_row_max`: Col0 (REDUCE_ROW out) | Col |
| `rescale_accum_block` | mul | `cb_out_accum`: All | `cb_rescale`: Col0 (exp of a Col0-valid difference; other columns exp(0) = 1, unused) | Col |
| `normalize_block` | mul | `cb_out_accum`: All | `cb_row_sum` (1/l): Col0 | Col |
| `score_block` | SFPU add (mask), in DEST | score tile: All | mask tile: All | None |
| `row_max_block` / `row_sum_block` folds | SFPU max / sub / mul / add, in DEST | Col0 | Col0 | None (column 0 is the payload; other columns ignored) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Unstrided matmul/reduce indices under C = 4 | Q, S, P live in STRIDED rings | stride rule (Realization contracts); test at `threads=(1,1,1)` and `(4,4,2)` must match; Q index × C, K/V index × 1 (ALL) |
| ALL ring with R producers | K/V blocks must come from reader thread `n mod R` in order, with equal counts | cluster-wide block counter computed identically in reader and compute; credit-only padding blocks; `skip_kv_block` for causal tails |
| Causal ragged extents vs ring wrap | NEOs of one round need different chunk counts and `kc_eff`; partial pushes would move the ring front | nominal push/pop per block; only compute narrows; causal chunks past a NEO's diagonal are popped unread |
| Mask arithmetic | −inf must survive scale (scale applied first, mask added after) and exp must give 0 | mask added after `mul_unary_tile`; accurate exp maps −inf → 0 (`ckernel_sfpu_exp.h:215`); chunk 0 always computed so `m` is finite |
| Precision through FPU sources | FPU reads Float32 rings through 19-bit sources (tf32): S for max/sub, P for P·V, O on reload/rescale | consistent `m` (the tf32 max is also what the subtract uses); error ≈ 5e-4 relative, below bfloat16 output rounding; SFPU-exact alternatives named as lamps; NUMERIC PCC 0.999 / RMS 0.05 |
| Pack target | set by `pack_init` | `pack_init(ring)` before each pack into a different ring (S, m, α, l, P, O, out) |
| 24 KiB compute binary | matmul + reduce + 6 SFPU ops + chain in one kernel | causal code under a CT arg; no per-op template fan-out of the chain; if over, merge inits / drop the chain for raw bcast calls |
| Tag left in DM registers | reader/writer leave a ring's trid | reader tags everything trid 0; writer hands the tag back at end; one-session test after another program and before `ttnn.to_layout` |
| Regime only on some shapes | `streamed_kv_online` is never selected on PHASE0 shapes with the default fit | regime-pinned tests through knob overrides (Regimes) |
| Default config differs from tt-metal's | tt-metal's fork forces 16-bit DEST on Quasar | this design keeps 32-bit DEST (only the experimental block-max path needs 16-bit); `reduce_tile<MAX>` in 32-bit DEST is legal (`llk_math_reduce.h:435` bans only MIN) |
| `program_config` hints | traced calls pass `q_chunk_size=32/64`, grids larger than needed | ignored entirely (R8); never validated |
