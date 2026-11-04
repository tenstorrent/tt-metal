# Operation Requirements: nomic_residual_layer_norm

## Definition
- **Formula**: `y = x + residual` (or `y = x`). Per row over H = 768: `mean = Σy / 768`; `var = max(Σy²/768 − mean², 0)` (fp32); `out = (y − mean) · rsqrt(var + ε) · gamma + beta`, rounded to bf16.
- **PyTorch Reference**:
  ```python
  def reference(x, gamma, beta, residual=None, epsilon=1e-5):
      y = x.double() + (residual.double() if residual is not None else 0.0)
      return torch.nn.functional.layer_norm(
          y, (768,), gamma.double().reshape(-1), beta.double().reshape(-1), eps=epsilon
      ).to(torch.bfloat16)
  ```
- **Import Path**: `from ttnn.operations.nomic_residual_layer_norm import nomic_residual_layer_norm`
- **Function Signature**:
  ```python
  nomic_residual_layer_norm(
      input_tensor: ttnn.Tensor,            # (B, 1, S, 768) bf16 TILE, L1, input_memory_config's layout
      gamma: ttnn.Tensor,                   # (1, 1, 1, 768) float32 TILE, interleaved
      beta: ttnn.Tensor,                    # (1, 1, 1, 768) bfloat16 TILE, interleaved
      *,
      residual: ttnn.Tensor | None = None,  # x's shape, dtype and memory config
      epsilon: float = 1e-5,
      output_tensor: ttnn.Tensor | None = None,   # None (new) or x itself (in place, Refinement 1)
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,  # fp32_dest_acc_en must be True
  ) -> ttnn.Tensor
  ```
  Module exports: `input_memory_config`, `output_memory_config`, `default_compute_kernel_config`, `circular_buffer_bytes`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED placement**: [op_layout]. This is the ND-sharded L1 layout `input_memory_config`: groups of 3 consecutive row-major cores, one group per `rows_per_shard` tile rows of a batch, member c holding hidden third c.
- **SUPPORTED residual_mode**: [residual, no_residual]
- **SUPPORTED output_mode**: [new]
- **SUPPORTED seq_alignment** (tagger): [tile_aligned, seq_padded]
- **Shape gate**: 32 ≤ M ≤ 128, last dim 768, and `_plan()` finds `rows_per_shard ≤ 4` with ≤ 36 row groups. Every INPUTS shape passes the gate.
- **Cores**: multi-core, `3 · num_row_groups` cores (96 at every focus shape, up to 108). Regime `split3_exchange`: one fp32 all-gather of packed (S, Q) per block inside each group of 3.
- **Compute config**: HiFi4 default (fidelity honoured from the caller); fp32_dest_acc_en forced True; UnpackToDestFp32 on the statistics CBs.
- **Golden baseline**: 46 / 154 cells pass (supported_pass 46, xfail_expected 108, all loud categories 0).
- **Perf (device kernel, 1350 MHz)**: 19.04 / 11.71 / 8.26 µs at 8x512 / 4x512 / 2x512 residual, and 17.61 µs at 8x512 no residual. **Every focus shape is above its acceptance ceiling** (16.1 / 10.1 / 6.5 / 14.7 µs).

TARGET − SUPPORTED = {placement: model_layout, output_mode: in_place}. Both are in Refinement 1. Nothing is left in INVALID or omitted. Refinements 2–3 are the perf debt: they add nothing to SUPPORTED.

### [x] Refinement 1 — Model split layout placement + in-place output

**Goal**: add `"model_layout"` to `SUPPORTED["placement"]` and `"in_place"` to `SUPPORTED["output_mode"]`, natively in the kernels. This moves all 108 `xfail_expected` cells to passing:
- 36 × (op_layout, in_place)
- 36 × (model_layout, new)
- 36 × (model_layout, in_place)

1. **`model_layout`** is the model's current "split layout" (`models/experimental/nomic_embed_text_v2_moe/tt/common.py`: `sharded_norm_split`, `sharded_norm_configs`). It is BLOCK_SHARDED L1, COL_MAJOR, on 8 core rows × C core columns, with shard (32·bh, 96), `bh = ceil(M/11)`, `C = ceil(M/bh)`. Run the existing `split3_exchange` algorithm on it, with the plan read off that layout:
   - group = one grid column of 8 cores (G = HIDDEN_SPLIT = 8; hidden slice = the core's grid row);
   - `hidden_tiles = 3`;
   - row block ≤ `bh` (up to 12 at M = 128);
   - x / residual / output consumed zero-copy through `ttnn.cb_descriptor_from_sharded_tensor`;
   - the last core column may hold a partial shard (`C·bh > M`), processed at nominal block size as in Phase 0.

   `validate()` must recognise the layout and classify it as `"model_layout"`. Today every non-op layout is tagged `"other"`. `output_memory_config` keeps returning the input config, so the output is in the model layout too.
2. **`in_place`** (`output_tensor` is x itself) applies to both placements. Back `cb_output` with x's own shard and return x. It is legal because each block's x tiles are fully consumed by `residual_add_block` (or, with no residual, by normalize stage 1) before stage 4 packs any output tile of that block. `"provided"` (a non-x `output_tensor`) stays refused.

**Classification**: complicated sharding (difficulty tier 1). The cross-core combine already exists, so G = 8 is a knob-turn of `HIDDEN_SPLIT` in the Blocking Model. But the knob has never been turned, and turning it forces three structural changes:
- an **odd-width row sum**: `hidden_tiles = 3`, and the pairwise-halves sum `static_assert`s an even width;
- an **8-member all-gather**;
- **row blocking to fit L1**: one block at bh = 12 needs ~481 KB of CBs against the 256 KB budget.

`in_place` is a pure knob-turn (CB backing) and too small to stand alone, so it is bundled here.

**Verifier notes**:
- **L1 fit is part of this refinement, not a follow-up.**
  - A whole-shard block at M = 128 (bh = 12, G = 8, stat_tiles = 3) is `cb_y` 144 KB + `cb_stats_gather` 192 KB + the rest ≈ 481 KB.
  - `block_rows ≤ 4` (num_row_blocks = 3, stat_tiles = 1) brings it to ≈ 170 KB. `MAX_BLOCK_ROWS` / `num_row_blocks` already work for the op layout: the verifier ran it at 2 with acceptance 20/20, and the multi-round handshaked exchange is correct.
  - `circular_buffer_bytes()` and `l1_ledger.md` must cover the model-layout plan, ≤ 256 KB at M = 128.
- **Ledger finding folded in here.** `cb_stats_gather` ships full fp32 tiles that carry 1/8 useful lanes (perf lamp L4). At G = 8 it is the largest scratch buffer: 2·stat_tiles·8 tiles. Compacting the payload (only the face rows holding slots) is the natural way to allow bigger row blocks; the implementer decides.
- **Exchange topology.** Model-layout groups are 1 × 8 *rectangles*, so the Mcast helper's rotating sender / hardware multicast is legal there. It was not legal for the op layout's row-wrapping triples (implementer friction breadcrumb). Eight serialized fixed-sender chain rounds would put 8 rendezvous hops on the critical path. Pick the transport by measurement.
- **Build the path performantly** (it is the model's adoption path). It must fill its 88 cores at M = 128, use one exchange per row block, and keep the zero-copy shards. Never re-read a core's own shard through a `TensorAccessor`: that would mean the placement is tolerated, not implemented.
- **Layout recognition without a models/ import.** Replicate the 8 × C / bh formula in the op module, or import it if the dependency direction is acceptable. The golden `axes.py` compares memory configs for equality with `sharded_norm_configs(...)[0]`, so the config must match exactly.
- **No wrappers.** No `ttnn.to_memory_config` (or any reshard) wrapper is acceptable for `model_layout`. The prompt's Rules forbid it as a MUST, so a partial-tick escape hatch does not exist here.
- **Order.** This lands before the perf refinements: tier 1 sharding is the hardest generality item, and Refinements 2–3 must then hold the model-layout path in their no-regression guard set.

**Done when**:
- all 108 currently-`xfail_expected` cells pass, with every loud verifier category at 0;
- acceptance, regression (constant rows / offset rows) and precision-baseline tests pass, with the precision baseline extended to one model-layout shape per `bh` class;
- `circular_buffer_bytes()` ≤ 256 KB at M = 128 on both placements;
- the model-layout 8x512 device-kernel time is measured and recorded in the changelog. The bar is to beat the incumbent's 29.7 µs on the same layout.

**Outcome**:
- *Generality*: complete. SUPPORTED gains placement `model_layout` and output_mode `in_place`; golden 154/154 (all 108 former xfail cells pass).
- *L1*: `circular_buffer_bytes` ≤ 256 KB on both placements for every M (194 KB model / 256 KB op at M = 128).
- *Perf, model 8x512 residual*: measured 48.7 → **34.4 µs**, via row-block lookahead, pair-tile stats and the batched combine. **The 29.7 µs bar is missed by 4.7 µs.**
- *Op layout*: also faster, 19.04 → 18.03 µs at 8x512.
- *What binds now*: compute. The exchange is hidden behind the next block's phase A. Two costs dominate:
  - the G = 8 all-gather combine: each core unpacks 8 full fp32 tiles per 2 rows, ~17k of ~46k cycles;
  - the per-row SFPU collapse at 3 hidden tiles a row: 8.8k cycles.
- *Next*: (1) a reduce-scatter combine (each member sums 1/8 of the pair-tile bytes, then all-gathers the finalized slice); (2) lane-compacted statistics (SFPSHFT2).
- *Why not here*: both are exchange / packing scheme changes, which is Refinement 3's scope (its guard set includes model_layout).

### [x] Refinement 2 — Speed up the perf-flagged profile: per-tile compute cost

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags **8x512, residual, new output, op_layout, bf16, fp32_dest_acc_en=True, HiFi4** as the mandatory perf target. Its gates:
- acceptance ≤ 16.1 µs (incumbent 29.7 µs ÷ 1.85);
- stretch 12.0 µs;
- soft precision gate `pcc_threshold` 0.99999 with 256 KB of L1 reserved.

It measures **19.04 µs** today, and is **compute-bound**. Removing the gamma/beta reads changed nothing, and TRISC1 runs 18.7 µs. At 32 tiles a core, the cost is about 600 cycles a tile against the design's 400-cycle budget (≈ 14.4 µs of the 19.0 µs).

Cut the per-tile cost of the block's FPU stages. Levers to start from (the implementer owns the choice; see `ttnn/ttnn/operations/examples/master.md`):
- **Normalize fusion (lamp L6; `compute_fusion`, `eltwise_l1_vs_dest_accumulate`).** The 4 normalize stages each round-trip all T fp32 tiles of `cb_y` through L1. Fold mean and rstd into `a = rstd`, `b = −mean·rstd` in the finalize, and cut the number of fp32 pack/unpack passes.
- **Block-size co-tune (`compute_block_size`).** Co-tune `ELTWISE_BLOCK_TILES` and the phase-A pass structure (pairwise sum + mul-accumulate both re-read `cb_y`).
- **HIDDEN_SPLIT sweep (lamp L1, ⭐ T1).** The prompt asks for the layout to be chosen by measurement, and no sweep is recorded. Sweep the values the L1 budget and the even-width rule allow at the three focus shapes. Keep G = 3 unless a measurement beats it.

No SUPPORTED change.

**Verifier notes**:
- **Why this comes first among the perf refinements.** The per-tile slope is the dominant term at the flagged shape. The fixed intercept, ≈ 4.7 µs, is Refinement 3's.
- **Side effect on precision.** Fewer truncating fp32 → srcA re-reads should also pull the got/true ratio median (0.99867, design risk P3) toward 1. Record it, but it is not a gate.
- **Turning `MAX_BLOCK_ROWS` alone is a measured regression** (+3.5 µs) and is not this refinement's lever.

**Done when**:
- measured device-ns on 8x512 residual (op_layout) improves, moving toward ≤ 16.1 µs;
- its soft precision gate (PCC ≥ 0.99999, RMS ≤ 0.005) still holds;
- the golden suite is green (all loud categories 0);
- no regression across the config-spanning guard set. That set is one representative per distinct kernel path × placement × output mode: residual and no_residual; op_layout and model_layout; new and in_place; plus 2x512 and 4x512.

**Outcome**:
- *Measured (device kernel, µs, op_layout residual)*: 8x512 **18.10 → 18.08** (flat), 4x512 10.42 → 10.40, 2x512 **8.24 → 7.21** (−12.5%), 8x512 no residual 16.70 → 16.73; off the focus set, M = 33–48 and M = 80–96 drop 1–3 µs (5x512 / 6x512: 14.8 → 11.7). model_layout unchanged (8x512 34.3 µs). Precision unchanged (PCC 0.9999984, rel RMS 0.0022–0.0023, got/true median 0.9986–0.9987).
- *What landed*: the HIDDEN_SPLIT sweep (lamp L1) as a per-shape choice: `_choose_hidden_split` picks G ∈ (3, 2) from a measured (G, rows) cost table. G = 2 wins at 1-row shards and over G = 3's 3-row plans; G = 3 keeps M = 56–72 and M > 110, including the flagged 8x512 (G = 2 measures 18.4 there). G = 4 is slower (14.8 / 10.4 µs at 4x512 / 2x512). G = 6 does not build: 6 chain channels exhaust the semaphores.
- *Normalize fusion (lamp L6)*: measured a regression and parked as `SEEDED_NORMALIZE` (default off, correct both ways). It folds 4 stages into 2 seeded ELWMULs, `z = (−mean·rstd) + y·rstd` and `out = beta + z·gamma`. The seed costs as much as the ADD stage it removes (row bcast: NB 4.3k vs N3+N4 4.5k cycles), and the −mean·rstd seed must land in DEST as fp32. Through srcB it breaks constant rows (error up to 2.9); the fp32 unpack-to-dest column broadcast is slow (23.3 µs). Pre-expanding mean to full tiles to drop N1's column-broadcast unpack was also slower (18.5 µs). `ELTWISE_BLOCK_TILES` 2 is slower (18.9); `dst_full_sync_en` + 8 is slower (21.8).
- *What binds now (8x512 zones, 24.3k cycles)*: the per-tile stages are at the FPU floor of this formulation. Phase A + normalize ≈ 15.4k cycles against ≈ 13.5k of pure math (6.5 FPU passes a tile; three are HiFi4 multiplies at ≈ 84 cycles a tile). The only per-tile excess is the column-broadcast unpack of `y − mean` (≈ 1.3k), and every way of removing it adds a phase that costs more. The design's "400 cycles a tile" budget is below that floor. The remaining gap to 16.1 µs (≈ 2.6k cycles) is fixed cost: the exchange wait (TRISC0 3.9k), the collapse/pack (2.4–3.1k) and the finalize (≈ 2.5k).
- *Next*: Refinement 3's levers, the exchange overlap and the collapse/finalize scope. A cheaper per-tile pass needs a lower-fidelity product proven exact, which is not allowed here because fidelity is honoured from the caller.

### [x] Refinement 3 — Speed up the perf-flagged profile: fixed per-call cost and exchange overlap

**Type**: perf

**Goal**: the same flagged target, 8x512 residual new op_layout (≤ 16.1 µs acceptance, 12.0 µs stretch), attacked through the per-call fixed cost. A linear fit over 8 / 16 / 32 tiles a core gives ≈ 4.7 µs of fixed work against the design's ≈ 2.6 µs budget. That fixed work is:
- the packed-stats collapse/pack;
- the exchange rendezvous (3 serialized fixed-sender rounds);
- the finalize/extract;
- startup and reconfigs.

It dominates the 2x512 guard (8.26 µs against a 6.5 µs ceiling) and is ≈ 25% of the flagged shape.

Levers (see `ttnn/ttnn/operations/examples/master.md`):
- **Row-block software pipelining (lamp L2, ⭐⭐⭐ T3; `tensix_all_reduce`, `double_buffer`).** Issue phase A of block b+1 while block b's exchange is in flight, with `cb_stats_gather` / `cb_stats_local` holding two rounds. Measured: the knob-turn alone (`MAX_BLOCK_ROWS = 2`, sequential compute schedule) is +3.5 µs at 8x512, so the overlap must come from the schedule.
- **Exchange transport and payload (lamps L4 / L5; `tensix_all_reduce_ring_transport`).** Ship only the slot face rows, not full fp32 tiles. Compare concurrent against serialized rounds; the exchange-mode knob already exists.
- **Finalize/extract scope (lamp L3; `sfpu_tile_scope`).** The slot 2–3 extraction does an fp32 round trip through `cb_stats_final`.

This is one T3 lever plus T2 trims, so it is a whole phase. No SUPPORTED change.

**Verifier notes**:
- Filed after Refinement 2 because the per-tile term is larger at the flagged shape.
- **No hard dependency on Refinement 2.** The pipelined schedule restructures the block loop, while R2 changes the stages inside a block.
- Must keep the model layout's multi-round exchange (Refinement 1) correct, and preferably faster: its G = 8 rendezvous is the same fixed cost.

**Done when**:
- measured device-ns on 8x512 residual (op_layout) improves, moving toward ≤ 16.1 µs and then the 12.0 µs stretch;
- the 2x512 guard moves toward its 6.5 µs ceiling;
- soft precision gate holds, golden suite green;
- no regression across the config-spanning guard set (as in Refinement 2, including model_layout and in_place).

**Outcome**:
- *Measured (device kernel µs, residual, before → after)*:
  - op_layout: 8x512 **18.10 → 16.82**, 7x512 18.02 → 16.93, 4x512 10.36 → **9.98** (under its 10.1 ceiling), 2x512 7.17 → 7.05, 8x512 no residual 16.74 → **14.15** (under its 14.7 ceiling);
  - model_layout: 8x512 / 4x512 / 2x512 / 8x512 no residual / 7x512 unchanged (34.3 / 20.3 / 13.3 / 31.4 / 52.9); 4x288 (bh 4) 15.2 → 14.8.
  - Precision unchanged: PCC 0.9999984, rel RMS 0.00223–0.00228, got/true median 0.99861–0.99868.
- *What landed*:
  - **Row-block software pipelining (L2)** as a plan rule: `PIPELINE_BLOCK_ROWS = 2`. A plan the L1 rule leaves as one block is split into 2-row blocks wherever that gives at least two, so the existing lookahead schedule hides the exchange. At 8x512 the TRISC0 gather wait drops from 3.8k to 0.07k cycles. 1-row blocks lose wherever they appear (8x512 20.2, 6x512 G = 3 16.5 vs 14.8), and so does re-splitting plans that L1 already splits (model 8x512, 6 blocks of 2: 38.9 vs 34.3). The rule leaves both alone.
  - **Exchange transport (L5)** per plan: single-round op_layout plans use `concurrent_unicast` (2x512 7.30 → 7.00, 4x512 10.42 → 10.00 in the sweep), multi-round plans keep the serialized chain. On the model layout concurrent loses at G = 8 (4x288 17.0, 2x512 15.0), so `MODEL_EXCHANGE_MODE` stays serialized.
  - **Parked**: `FUSED_RECOMPUTE` (default off, correct), the lookahead recompute fused into normalize stage 1 as (x − mean) + r through `DestReuseBinary`. Measured slower: 17.4 vs 17.0 at 8x512, 35.0 vs 34.3 at model 8x512. The DEST → srcB move costs more than the pass it removes.
- *What binds now (8x512, 2 blocks, T1 zone cycles of ≈ 23k)*: compute. Per-tile streams ≈ 15.7k (add 1.0k, S + Q 4.1k, normalize 9.5k, lookahead recompute 1.1k), collapse 3.3k, finalize 3.5k. The exchange is hidden. At 2x512 / 4x512 (single-round), the exposed gather wait is 1.4k / 2.3k cycles of a 4 KB slot. That wait is rendezvous latency and inter-core skew, not bytes, so payload compaction (L4) would move about 60 cycles.
- *Next, not done here*:
  - (1) Drop the lookahead recompute by keeping y two blocks deep. That needs 284 KB of CBs at M = 128, over the 256 KB budget, unless normalize also streams through a smaller intermediate.
  - (2) Hide the single-round exchange by finalizing the mean ahead of the variance, so stage 1 overlaps the Q exchange. This costs a second rendezvous and split pair tiles, and was not measured.
  - (3) Cut the collapse: 4 full-tile SFPU row reduces per pair tile. Pre-folding the two rows' halves into one tile needs a custom one-face reduce.
