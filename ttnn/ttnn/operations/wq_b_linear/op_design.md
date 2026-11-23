# Operation Design: wq_b_linear

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused matmul: own reader / x-multicast / compute / writer kernels, one program dispatch) |
| Goal | `y = x @ W` for one chip's share of the DSA indexer wq_b projection, as fast as possible on the single problem `[1,1,160,1536] bf16 × [1536,4096] bf8_b → [1,1,160,4096] bf16 (L1)`. DRAM-bound: SOL 16,018 ns (7.18 MB at 448 GB/s, p100a). |
| Math | `y[0,0,m,n] = Σ_k x[0,0,m,k] · W[k,n]` — no bias, no activation |
| References | `eval/golden_tests/wq_b_linear/feature_spec.py` (TARGET / INPUTS / LOOSE_CASES + SOL), `eval/golden_tests/wq_b_linear/helpers.py` (oracle, TOLERANCES), `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/{host/mcast.hpp,kernel/mcast_args.hpp,kernel/pipes/mcast_pipe.hpp}`, `ttnn/ttnn/operations/toy_spec_mcast/toy_spec_mcast.py` + `ttnn/ttnn/operations/examples/shared_input_reuse/shared_input_reuse.py` (Python `ttnn.Mcast` wiring, `McastExplicitFixedSenderConfig`), `.claude/references/speed-of-light.md`, `.claude/tools/perf_calculator/` (op model + `dm_estimates.json`) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (x) | `ttnn.Tensor` | yes | `[1,1,160,1536]`, bf16, TILE, DRAM interleaved | — | buffer address RT (sender kernel) |
| `weight` (W) | `ttnn.Tensor` | yes | `[1536,4096]`, bf8_b, TILE, DRAM interleaved | — | buffer address RT (reader kernel) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (keyword-only) | `fp32_dest_acc_en=False`; any `math_fidelity` (HiFi2 tested) | `default_compute_kernel_config()` = `HiFi2, fp32_dest_acc_en=False, math_approx_mode=False` | compute-kernel config |

`default_compute_kernel_config()` is exported from the op package (the golden `axes.py` imports it); `None` resolves through it. `validate()` is the entry point's first line.

### Registry declarations (op file)

| Declaration | Value |
|-------------|-------|
| `INPUT_TAGGERS` | `{}` (one shape → no shape-derived axis) |
| `SUPPORTED` | `dtype: [bfloat16]`, `weight_dtype: [bfloat8_b]`, `layout: [TILE_LAYOUT]`, `fp32_dest_acc_en: [False]` — equals TARGET (one cell) |
| `EXCLUSIONS` | `[]` |
| `validate()` | `NotImplementedError` for any axis value outside SUPPORTED (axis names exactly as TARGET: `dtype`, `weight_dtype`, `layout`, `fp32_dest_acc_en`); `ValueError` if `x.shape != [1,1,160,1536]` or `W.shape != [1536,4096]`, or if x or W is not DRAM interleaved |
| Structural impossibilities | none (`INVALID = []` agrees) |

## Tensors

### Input

| Property | x (activation) | W (weight) |
|----------|----------------|------------|
| Shape | `[1,1,160,1536]` → `tensor_m_tiles = ceil(160/32) = 5`, `tensor_k_tiles = 48` | `[1536,4096]` → `tensor_k_tiles = 48`, `tensor_n_tiles = 128` |
| Dtype | bfloat16 (2048 B/tile) | bfloat8_b (1088 B/tile) |
| Layout | TILE | TILE |
| Memory | DRAM interleaved (page = tile, page id `m*tensor_k_tiles + k`) | DRAM interleaved (page id `k*tensor_n_tiles + n`, bank `page % num_dram_banks`) |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1,1,160,4096]` (`[..., M, N]`, `M = x.shape[-2]`, `N = W.shape[-1]`) |
| Dtype | bfloat16 (activation dtype) |
| Layout | TILE |
| Memory | **L1 interleaved** (`ttnn.L1_MEMORY_CONFIG`), allocated on host before the dispatch; page id `m*tensor_n_tiles + n` |

## Blocking Model

> **Perf 1 (2026-11-23):** compute no longer uses `cb_interm_acc`: the one `matmul_block` call packs every K-step with L1 accumulation straight into `cb_out_block` (`LastBlockTarget::Interm`, `TileRowMajor`, `caller_owns_pack_target`), which removed a per-K-step unpack-side drain. `block_k_tiles` is now 2 (`num_k_blocks` 24). Tables below that mention `cb_interm_acc`, the last-K-step reload, or kb 4 describe the Refinement-2 build; see `changelog.md` Perf 1.

Semantics: `.claude/references/blocking-model.md`. Tile index space is 3-D: `(m, k, n)` with extents `5 × 48 × 128`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `m` (output rows, 160 → 5 tiles) | independent — no cross-row dependency; **W is reuse-shared along it** (W does not vary with `m`), so cutting `m` re-reads (or must multicast) W per band | `block_m_tiles` | 5 = whole tensor (`tensor_m_tiles`) | host constant `block_m_tiles = tensor_m_tiles` | **not split** (1 band): every worker owns all 5 rows | scheme-change (2D: M bands + W multicast down grid columns — `deferred` row) |
| `n` (output cols, 4096 → 128 tiles) | independent — no cross-column dependency; **x is reuse-shared along it** (x does not vary with `n`) | `block_n_tiles` | 2 | host constant `block_n_tiles` (one source; `num_workers = tensor_n_tiles / block_n_tiles` derived) | **split**: `num_workers = 64` cores, worker `j` owns `block_n_tiles` columns chosen by the bank-spread column map (Work Distribution) | knob-turn (`block_n_tiles` 2 → 4 halves the workers; 2 → 1 needs 128 cores > grid) |
| `k` (contraction, 1536 → 48 tiles) | dependent — every output tile sums over all of `k` | `block_k_tiles` (helper `in0_block_k`); `num_k_blocks = tensor_k_tiles / block_k_tiles` derived | 4 → `num_k_blocks = 12` | host constant `block_k_tiles` | **not split across cores** (one core owns the whole `k` sum of its outputs → deterministic); streamed in `num_k_blocks` blocks per core | scheme-change (K-split groups + deterministic combine — `deferred` row); `block_k_tiles` itself is a knob-turn (2/4/6/8/12) |

The per-core block is the 3-D rectangle `block_m_tiles × block_k_tiles × block_n_tiles = 5 × 4 × 2` per K-step; the core's output block is `block_m_tiles × block_n_tiles = 5 × 2 = 10` tiles, driven through `num_k_blocks` K-steps in **one** `matmul_block` call (one init, one CB discipline). There is **one output block per core** (`num_blocks_this_core = 1`): the core's whole assignment is one block, the coarsest possible.

**Operand-reuse check (mechanical, per operand × chosen split `n`):**

| Operand | Varies along the split axis `n`? | Consequence |
|---------|----------------------------------|-------------|
| x `[m,k]` | no | reuse-shared by construction → **read once from DRAM on one sender core, multicast to all 64 workers** (built; the per-core DRAM re-read is `rejected`) |
| W `[k,n]` | yes | each worker reads only its own columns once — no reuse |
| y `[m,n]` | yes | each worker writes only its own tiles |

**Split search (op model, `/perf-calculator`).** Commands:

```
python3 .claude/tools/perf_calculator/op_model.py matmul 5 48 128 --fidelity HiFi2 --a-dtype bf16 --b-dtype bfp8_b --c-dtype bf16
python3 .claude/tools/perf_calculator/op_model.py matmul 5 48 128 --designs --c-mem l1 --fidelity HiFi2 --a-dtype bf16 --b-dtype bfp8_b --c-dtype bf16
```

(Model board: Blackhole p150, 11×10 workers, 8 banks. Target board p100a: 11×10, 7 banks — rank only.) Shortlist (`--designs`, 6,696 designs):

| Rank | us | Config |
|------|----|--------|
| 1 | 16.53 | 1 band, ksplit 1, 72 N-cores + 1 x-sender, kb 2, x mcast resident (no handshake), W from DRAM `bank_spread`, x NoC1 / W NoC0 / C split, rect 8×9 |
| 2 | 16.60 | same, **64 N-cores** (2 cols each) + 1 sender, rect 8×8 at logical (2,2)–(9,9), sender (10,9) |
| 3 | 17.22 | rank 1 with kb 4 |
| 4 | 17.39 | rank 2 with kb 4 ← **Phase 0** (tie with rank 1, 5%) |
| 5 | 18.15 | ksplit 3 (3 K-slice groups × 32 cores = 99), partials combined over NoC (2.6 MB) |

Curve (best us per core count, `--designs`): 17→51.7, 25→39.2, 33→27.3, 44→21.4, 50→21.2, 64→20.9, **65→16.6**, 73→16.5, 81→18.6, 99→18.2, 110→18.7. The curve flattens at 65; 73 is a tie with an uneven 128/72 column split. Lower bounds at 65: DRAM 16.6 us (model), compute 12.15 us.

**Phase 0 = rank-2/4 family: 64 workers + 1 sender (65 cores), `block_k_tiles = 4`.** Deviations from the literal rank 1, each a tie (<10%) with a stated reason:
- 64 vs 72 workers: 16.60 vs 16.53 (tie); 64 divides 128 evenly (`block_n_tiles = 2` on every core → no ragged columns), fewer cores (≈0.65 us/core launch cost not modelled).
- `block_k_tiles` 4 vs 2: 17.39 vs 16.60 (tie, 4.7%). The op model omits exactly the per-K-block fixed costs that the extent ranking prices (≈90 cycles CB handoff per block, per-K-block pack of all 10 output tiles into the L1-accumulation buffer, two `matmul_block` subblock passes). At `block_k_tiles = 2` a K-step is only 20 tile-matmuls (640 HiFi2 cycles) against a 10-tile pack — pack-bound; at 4 it is 1,280 cycles against the same pack. Perf lamp: measure 2.
- writer NoC 1 instead of `c_noc = split`: 17.55 vs 17.39 (tie); one NoC per kernel. Perf lamp.
- x sender reads and multicasts on one NoC (NoC1) instead of `a_read_noc = 0`: 17.92 vs 17.55 (tie, 2%); one sender kernel. Perf lamp.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_resident` | `x_depth_rounds` | `num_k_blocks` (12) — **whole x resident**, fixed by the scheme | The handshake-free multicast: every round lands in its own slot, so the sender never waits for a consumer ack (no `CONSUMER_READY` round trip, ~1,350 cycles/round at 64 receivers). Not a tunable — reducing it requires the handshake protocol (see Regimes). |
| `cb_w_block` | `w_depth_blocks` | `num_k_blocks` (12) — whole per-core W slice (96 tiles, 104 KB) | **Stall shadow**: the W reader never blocks on CB space, so it streams W from t=0 while compute waits for x round 0 (sender DRAM read + first multicast ≈ 1–2 us). With depth 2 the reader would stall after 2 blocks and DRAM would idle on the critical path. Lamp: 3–4 is likely equivalent. |
| `cb_out_block` | `out_depth_blocks` | 1 (one output block per core) | nothing to overlap — a core has exactly one output block; the store is the tail |
| `cb_interm_acc` | — | 1 block | holds the L1-accumulated partial of the one output block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST (16-bit, `fp32_dest_acc_en=False`, half-sync) — `ASSERT(out_subblock_h*out_subblock_w <= DEST_AUTO_LIMIT)` `matmul_block_helpers.inl:158`, `DEST_AUTO_LIMIT` `dest_helpers.hpp:103` (= 8) | output **subblock**, not the block | subblock `h × w` divides `block_m_tiles × block_n_tiles` and `h·w ≤ 8`. Block 5×2 → subblock **5×1** (largest legal; `in0_num_subblocks=1`, `in1_num_subblocks=2`). Derived by the implementer from the block, not a design knob. | ASSERT (dev) / DEST overrun → wrong tiles |
| `tensor_n_tiles % block_n_tiles == 0`, `tensor_k_tiles % block_k_tiles == 0` | `block_n_tiles`, `block_k_tiles` | host asserts divisibility (128 % 2, 48 % 4) | wait/pop count mismatch → hang |
| Multicast rectangle (`ttnn.Mcast`, one rectangle receiver group) | `num_workers` and grid geometry | receivers must form one full rectangle `rect_w × rect_h = num_workers`, `rect_w ≤ grid_x − 1` (sender column to its right), `rect_h ≤ grid_y`. Host picks `rect_w` = largest divisor of `num_workers` ≤ `min(mcast_rect_width_max, grid_x − 1)`; if none fits, doubles `block_n_tiles` and retries. | mcast to non-rectangle / off-grid cores → hang or corrupt L1 |
| Counter data-ready semaphore (`DataReadySignal::Counter`, `mcast_protocol.hpp:18`) | `num_k_blocks` (rounds) | 32-bit counter, 12 rounds — no clamp needed | — |
| `cb_x_resident` / `cb_w_block` full residency | `tensor_m_tiles × tensor_k_tiles`, `tensor_k_tiles × block_n_tiles` | bounded by `validate()`'s single-shape predicate (x is exactly `[1,1,160,1536]`); per-core total 636,928 B < usable L1 (~1.4 MB) | L1 allocation failure at program creation (not silent) |
| Multicast write payload per round | round bytes `block_m_tiles·block_k_tiles·2048 = 40,960 B` | `noc.async_write_multicast` (`mcast_pipe.inl:245–248`) issues the transfer; any size > NoC max burst is split by the NoC API — implementer confirms on the first run; if not, send a round as `ceil(40960/16384)` packets | truncated round → wrong x tiles |

### Regimes

Named boundary: **DRAM**. Minimum = each input crosses DRAM once (x 491,520 B + W 6,684,672 B = 7,176,192 B); output never touches DRAM (L1).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **N-split, x multicast resident** (1 band, 64 workers × `block_n_tiles` cols, 1 x-sender) | **built** | the only shape (`validate()`): `M=160, K=1536, N=4096`; grid fits `rect_w×rect_h + 1` sender column | per worker `5 × 4 × 2` per K-step, one `5 × 2` output block, `num_k_blocks = 12` | **minimum**: x 1× + W 1× from DRAM. Added NoC: x multicast 491,520 B delivered to 64 receivers (one tree, 12 rounds); y 1,310,720 B core→L1 bank writes | Bigger `block_k_tiles`: fewer CB handoffs / L1-acc pack passes / Counter semaphore increments (12 at kb=4) — at the cost of pipeline fill (first x round + first W block) and drain (last K-step). Bigger `block_n_tiles`: fewer cores (fewer launches), more compute per core (compute 11.4 us → 22.8 us at 4 → compute-bound). This domain is the whole op, so no other regime applies to any input. |
| N-split, every worker reads x from DRAM | **rejected** — superseded by the built multicast regime; dead end here | — | same | x crosses DRAM **64×** (31.5 MB): 4.4× the SOL DRAM bytes on a DRAM-bound op | — |
| N-split, x multicast with per-round consumer handshake (2-slot ring, `McastConfig.handshake=True`) | **rejected** — superseded: x fits L1 resident, so the ack round trip (~1,350 cycles × 12 rounds at 64 receivers) buys nothing | — | same | minimum + 12 × 64 acks | — |
| 2-D split: M bands on grid rows + N columns, x multicast along rows, W multicast down columns | **deferred** — op model ranks its best (5 bands × 8) at 22.3 us vs 16.6 us, and it is a lower bound (two trees on shared cores, retry backoff unmodelled, device 1.3–2.1× slower). Only pays when compute binds; here DRAM binds (compute 12.15 us < DRAM 16.6 us) | — | `1 × 4 × block_n_tiles` per band | minimum DRAM (W read once, multicast) + 2 trees of NoC traffic | Reachable: `block_m_tiles` is already a knob; adds a second `ttnn.Mcast` channel (column-major) for W (`mcast_topology` catalog entry). |
| K-split: `ksplit` K-slice groups, partial sums exchanged by output row and added (deterministic fixed-order combine) | **deferred** — op model 18.15 us (ksplit 3, 99 cores) vs 16.6–17.4 us; adds 2,621,440 B of partial NoC traffic + a combine stage + the bit-identical-repeat requirement, to cut compute that is not binding | — | `5 × 4 × 2` per K-step, `num_k_blocks/ksplit` per group | minimum DRAM + 2.6 MB NoC partials | Reachable: K is already a streamed, knob-sized axis; a K-slice is a `[k_begin, k_end)` range per group + a combine phase after `matmul_block` (fixed order → deterministic). |
| M-split with every band re-reading W from DRAM | **rejected** — W (93% of the bytes) crosses DRAM once per band | — | — | W × bands | — |

Regime selection: one built regime; no selection function needed (all other rows are not built).

### Traffic ranking

Bytes per tier for each candidate split (x = 0.49 MB, W = 6.68 MB, y = 1.31 MB in L1):

| Rank | Split | DRAM bytes | Cross-core NoC bytes | Notes |
|------|-------|-----------|---------------------|-------|
| 1 | **N-split + x multicast (chosen)** | 7.18 MB (min) | x tree 0.49 MB × 64 receivers; y 1.31 MB writes | fan-out 64 from one sender; W reads bank-spread |
| 2 | K-split + combine (ksplit 3) + x multicast | 7.18 MB (min) | + 2.62 MB partials | reduces per-core compute only |
| 3 | 2-D (M bands × N) + two multicasts | 7.18 MB (min) | x and W trees, overlapping on cores | NoC-tree contention |
| 4 | N-split, x read by every worker | 38.6 MB | y only | 5.4× DRAM |
| 5 | M-split, W per band | 6.68 MB × bands | — | W dominates |

Chosen: rank 1 — minimum DRAM bytes, no combine (tiebreaker: independent axis), and the op model agrees (16.6 us vs 18.15 / 22.3).

### Block schedule

```cpp
// per worker core (64 of them); one output block per core
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core /* = 1 */; ++block_idx) {
    stream_w_block_slice(block_idx);     // reader: W[k-step, my cols] for k_step = 0..num_k_blocks-1, from t=0
    receive_x_rounds(block_idx);         // writer: x round r arrives via multicast (Counter >= r+1) → push
    matmul_accumulate_block(block_idx);  // compute: one matmul_block over num_k_blocks K-steps, L1 accumulation
    store_block(block_idx);              // writer: 10 output tiles → L1-interleaved y
}
// x-sender core (1): for r in 0..num_k_blocks-1: load_x_round(r+1) overlapped with multicast_x_round(r)
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|----------------------------------|
| `load_x_round` (sender) | `block_m_tiles × block_k_tiles` = 5×4 tiles (40,960 B) of x, tile order `(m, k)` row-major within the round | whole x in sender's `cb_x_resident` region (rounds never reused) | 20 tile reads issued per round, **one read barrier per round**; round r+1's reads are issued *before* round r's multicast (stall-shadow: the multicast's write barrier and the next round's DRAM reads overlap) |
| `multicast_x_round` (sender) | same 40,960 B → same offset in every worker's `cb_x_resident` | — | one `send()` per round (12 total): one multicast + one Counter increment; no consumer handshake |
| `stream_w_block_slice` (reader) | `block_k_tiles × block_n_tiles` = 4×2 tiles per K-step, tile order `(k, n)` row-major within the block | all 12 K-steps fit (`w_depth_blocks = num_k_blocks`) | 8 tile reads + **one read barrier + one push per K-step** (batch 8 = DM-sweep rank 1) |
| `receive_x_rounds` (writer) | 20 tiles per round | x resident | one semaphore poll + one `push_back(20)` per round; no NoC traffic |
| `matmul_accumulate_block` (compute) | output `5×2`, K-steps of 4; subblock 5×1 (helper-internal) | output partial resident in `cb_interm_acc` (L1 accumulation) | **one** `compute_kernel_hw_startup` + **one** `matmul_block` call (one init); per K-step: 2 subblock passes, one L1-acc pack of 10 tiles; last K-step reloads the partial once and packs to `cb_out_block` |
| `store_block` (writer) | 10 tiles (5×2) | — | one `wait_front(10)`, 10 tile writes, **one** write barrier |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** — `block_k_tiles` = 4 | the op model ranks 2 first (smaller fill/drain); the extent argument (per-K-step pack/handoff) says 4 | `block_k_tiles` ∈ {2, 6, 8}; read compute-alone time (`/perf-measure` ablation) against the 15.6 us W stream |
| **Grid synchronization** — 64 workers | per-core launch/dispatch ≈ 0.65 us/core (not in the model); 64→72 is a tie; 33 cores → 27 us | 72 workers (uneven 128/72 columns) and 32 workers (`block_n_tiles = 4`, compute-bound) — expect 64 to hold |
| Writer NoC | op model `c_noc = split` is 1% better | alternate output tiles between NoC0/NoC1 in the writer |
| x sender read NoC | op model `a_read_noc = 0` (read on NoC0, multicast on NoC1) is 2% better | split the sender into a NoC0 reader kernel + NoC1 multicast kernel on the sender core's two RISCs (CB handoff) |
| W read coalescing | the bank-spread map puts a core's 2 columns `n, n+B` on one bank in adjacent slots → one 2,176 B read per k-row instead of two 1,088 B reads (`read_coalescing` catalog entry) | one `noc_async_read` of `2·page_size` from the first column's bank address |
| Column map | the op model's `bank_spread` (+15% vs `contiguous`) was computed for 8 banks; p100a has 7 | `contiguous` columns (`2j, 2j+1`) vs bank-spread on the target board |
| Multicast rectangle placement | the model's best rect sits away from the grid origin (logical (2,2)–(9,9), sender (10,9)); this depends on DRAM/NoC geometry | rectangle at the other grid corner / sender on the left |
| `w_depth_blocks` | full residency (12) is the safe stall-shadow choice; 3–4 may already cover the x-round-0 latency | 3, 4 |
| Linked multicast | Counter mode issues a write barrier per round (`mcast_pipe.inl:249–251`), so rounds are unlinked; model shows linked = unlinked here (x not critical) | a linked raw `noc_async_write_multicast` stream only if x arrival shows up on the critical path |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → sender L1 | bf16 tiles | sender kernel, `TensorAccessor` page reads (page id `m*48 + k`), 20 per round into the round's slot at `cb_x_resident` base + `r·40960`, tile order `(m, k_local)` | one sender core outside the receiver rectangle |
| sender L1 → 64 workers' L1 | bf16 tiles | `ttnn.Mcast(device, McastConfig(noc=NOC_1, handshake=False, data_ready=DataReadySignal.Counter), receivers=rect, receiver_group_size=num_workers, McastExplicitFixedSenderConfig([sender]))`; kernel `sender.send(src_l1, dst_l1, 40960)` with `src_l1 == dst_l1` (same CB address on all cores) | `cb_x_resident` is declared on the union {workers, sender} so its L1 address is identical everywhere |
| worker: multicast landed → compute | bf16 | writer kernel `receiver.receive(r)` (Counter ≥ r+1), then `cb_x_resident.push_back(20)` | the writer is idle until the store, so it owns x arrival and the W reader never waits on x |
| W DRAM → worker L1 | bf8_b tiles | reader kernel, `TensorAccessor` page reads (page id `k*128 + n`) for the core's columns, `(k, n_local)` order, NoC0 | the core's column list comes from host runtime args |
| compute | bf16 × bf8_b → 16-bit DEST → bf16 | `matmul_block` with packer L1 accumulation into `cb_interm_acc`, last K-step → `cb_out_block` (TileRowMajor) | |
| worker L1 → y (L1 interleaved) | bf16 tiles | writer, `TensorAccessor` page writes (page id `m*128 + n`), NoC1, one barrier | output tiles land in L1 banks across the chip |

TARGET has no placement axis; nothing is sharded. The Tensix-to-Tensix contracts for the deferred rows: 2-D adds a second `ttnn.Mcast` channel (column-major groups, W blocks) at `next_semaphore_id()`; K-split adds a partial-sum exchange (fixed-order unicast of `5×block_n_tiles` partials per K-slice group, semaphore-gated, added in a fixed slice order for determinism).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output block `block_m_tiles × block_n_tiles` (5×2 tiles) with its full K sum |
| Grid | `grid = device.compute_with_storage_grid_size()`. `num_workers = tensor_n_tiles / block_n_tiles` (64). `rect_w` = largest divisor of `num_workers` ≤ `min(mcast_rect_width_max=8, grid.x − 1)`, `rect_h = num_workers / rect_w` (8), require `rect_h ≤ grid.y`. Receiver rectangle (logical) `x ∈ [grid.x−1−rect_w, grid.x−2]`, `y ∈ [grid.y−rect_h, grid.y−1]`; sender `(grid.x−1, grid.y−1)`. On 11×10: rect (2,2)–(9,9), sender (10,9). Physical coordinates come from the Mcast helper / device queries — nothing hardcoded. |
| Per-core work | worker `j` (row-major index within the rectangle) owns columns `cols[j]` (length `block_n_tiles`), all 5 rows, all 48 K-tiles: 96 W tiles in, 240 x tiles via multicast, 10 y tiles out, 480 tile-matmuls |
| Column map (`bank_spread`) | `B = num_dram_banks` from the device (W's interleaving bank count — p100a 7, p150 8). Group columns by residue `n % B` (columns with equal residue sit on one bank in every k-row, adjacent slots). Worker `j` takes `block_n_tiles` unused columns from residue `j % B` in increasing order; columns left over (residue sizes not a multiple of `block_n_tiles`) are dealt in order to workers still short. Neighbouring workers read different banks; per k-row each bank serves `≈ num_workers / B` workers. Host asserts every column is assigned exactly once. |
| Remainder | none on this shape (`128 % 2 = 0`, `48 % 4 = 0`, `5` rows whole); geometry uses `ceil(H/32)` per image for `tensor_m_tiles` |
| Regime selection | single regime, no predicate |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_resident` | 0 | 2048 | `block_m_tiles · block_k_tiles · x_depth_rounds` = 5·4·12 = 240 | spans `m` (whole) and `k` (whole, resident — required by handshake-free multicast); streams nothing | Float16_b | worker: writer (`push_back` per received round; data written by the sender's multicast) · sender core: sender kernel (raw L1 staging, no push/pop) | worker: compute | whole program |
| `cb_w_block` | 1 | 1088 | `block_k_tiles · block_n_tiles · w_depth_blocks` = 4·2·12 = 96 | spans `k`-step × `n` block; depth = `num_k_blocks` (stall-shadow) | Bfp8_b | reader | compute | whole program |
| `cb_interm_acc` | 2 | 2048 | `block_m_tiles · block_n_tiles` = 10 | spans the output block `m × n`; streams `k` (accumulated in place) | Float16_b | compute | compute | K-steps 0..`num_k_blocks−1` |
| `cb_out_block` | 3 | 2048 | `block_m_tiles · block_n_tiles · out_depth_blocks` = 10 | spans `m × n`; streams `k` (written once at the last K-step) | Float16_b | compute | writer | last K-step → store |

`cb_x_resident` and `cb_w_block` exist on worker cores; `cb_x_resident` is also declared on the sender core (same index/size) so its address matches. `cb_interm_acc` is a matmul-internal FIFO (pack → reload inside the one compute kernel, the helper's documented interm contract).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_x_round` + `multicast_x_round` (sender) | 5×4 tiles | Mcast helper (`send`) + `TensorAccessor` reads | DRAM x | sender `cb_x_resident` slot r (raw), workers' `cb_x_resident` slot r | worker semaphore counter = r+1 |
| 2 | `stream_w_block_slice` (reader) | 4×2 tiles per K-step | `TensorAccessor` reads (dataflow) | DRAM W | `cb_w_block`, 8 pages per push, 12 pushes | full ring closed exactly (96 = capacity) |
| 3 | `receive_x_rounds` (writer) | 20 tiles per round | Mcast helper (`receive`) | multicast landing | `cb_x_resident`, 20 pages per push, 12 pushes | ring closed exactly (240 = capacity) |
| 4 | `matmul_accumulate_block` (compute) | out 5×2, K-step 4, 12 steps | `matmul_block` | `cb_x_resident` 20/step (wait+pop), `cb_w_block` 8/step (wait+pop) | `cb_interm_acc` (internal), `cb_out_block` 10 | x and W fully consumed; `cb_out_block` holds 10 tiles |
| 5 | `store_block` (writer) | 10 tiles | `TensorAccessor` writes | `cb_out_block` 10 (wait+pop) | y in L1 | empty |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | helper prerequisite | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_w_block, cb_out_block)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60`; Reverse mandated at `matmul_block_helpers.hpp:100-103` | once at top of compute `kernel_main` | — | — | — |
| `matmul_accumulate_block` | helper | `compute_kernel_lib::matmul_block<false /*transpose*/, true /*packer_l1_acc*/, LastBlockTarget::Out, OutputCBLayout::TileRowMajor>(x_buf, w_buf, out_buf, interm_buf, MatmulBlockShape::of(in0_num_subblocks=1, in1_num_subblocks=2, out_subblock_h=5, out_subblock_w=1, in0_block_k=block_k_tiles, num_k_blocks))` | decl `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:339-371`; shape `:139-170`; L1-acc Out path (accumulate blocks 0..n−2, reload once, pack last to out) `matmul_block_helpers.inl:407-418, 501-534`; TileRowMajor pack `.inl:35, 435, 476` | default `InitMode::Short`, `reconfig=INPUT_AND_OUTPUT` (one call → one init); input policies default `WaitAndPopPerKBlock` | `cb_x_resident`, `cb_w_block` | `cb_interm_acc` (interm), `cb_out_block` | `in0_block_k` (= `block_k_tiles`), `num_k_blocks`; output block `block_m_tiles × block_n_tiles` via the subblock counts (subblock dims derived: largest `h·w ≤ 8` dividing the block → 5×1) |
| x multicast channel (host) | helper | `ttnn.Mcast(device, ttnn.McastConfig(noc=NOC_1, handshake=False, data_ready=Counter), receivers, num_workers, ttnn.McastExplicitFixedSenderConfig([sender]))`; `mcast.attach(descriptor, "x_mcast", [sender_kernel, writer_kernel], first_semaphore_id)` | `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp:98-107, 124-126, 133-155`; Python usage `ttnn/ttnn/operations/toy_spec_mcast/toy_spec_mcast.py:148-153`, `examples/shared_input_reuse/shared_input_reuse.py:323-358` | | — | — | `num_workers` (= receiver group size) |
| `multicast_x_round` | helper | `McastArgs<ct,rt>().sender(noc).send<SourceL1Guard::CallerManaged>(slot_addr, slot_addr, round_bytes)` | `mcast/kernel/mcast_args.hpp:309-329`; `mcast/kernel/pipes/mcast_pipe.hpp:91-92`; Counter path barrier `mcast_pipe.inl:241-252` | `CallerManaged` is safe: each slot is written once and never reused | sender `cb_x_resident` (raw) | workers' `cb_x_resident` | `round_bytes = block_m_tiles·block_k_tiles·tile_bytes` |
| `receive_x_rounds` | helper | `McastArgs<ct,rt>().receiver(noc).receive(r)` then `cb_push_back(cb_x_resident, block_m_tiles·block_k_tiles)` | `mcast/kernel/pipes/mcast_pipe.hpp:163` | round index `r` (Counter compares ≥ r+1) | — | `cb_x_resident` | pages per push |
| `load_x_round`, `stream_w_block_slice`, `store_block` | raw_api (dataflow) | `TensorAccessor` + `noc_async_read_page` / `noc_async_write_page` + barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | batch = one block per barrier | DRAM / CBs | CBs / y | pages per barrier = block size |

**Helpers considered and rejected (raw_api rows):** the dataflow page loops are plain `TensorAccessor` transfers. `dfb_helpers_dataflow.hpp` / `local_copy_helpers_dataflow.hpp` were considered: the W reader needs an arbitrary per-core column list (bank-spread map) inside each K-step and the sender needs `(m, k)`-ordered slots at fixed L1 offsets for the multicast — neither is a contiguous page range, so a contiguous-range read helper does not express it. No compute phase uses raw APIs.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| `cb_x_resident` address must match on sender and workers | the multicast writes to `dst_l1 = src_l1` on every receiver | declare the CB on the union core range {workers ∪ sender} in one CB descriptor; pass `get_write_ptr(cb_x_resident)` (sender) as both src and dst |
| x round tile order vs `matmul_block`'s in0 layout | helper expects in0 per K-step as `block_m_tiles × block_k_tiles` row-major `(m, k)` | the sender reads round r as pages `m*48 + (r*4 + kk)` for `m` outer, `kk` inner, into consecutive slot tiles |
| W block tile order | helper expects in1 per K-step as `block_k_tiles × block_n_tiles` row-major `(k, n)` with `n` in the order of `cols[j]` | reader loops `kk` outer, column index inner; writer maps output column `nn` to `cols[j][nn]` with the same order |
| TileRowMajor output order | `cb_out_block` tile `i` = `(m = i / block_n_tiles, nn = i % block_n_tiles)` | writer writes page `m*128 + cols[j][nn]` |
| Precision (PCC ≥ 0.9998, 16-bit DEST) | DEST is 16-bit, so each K-step's L1 accumulation and each in-DEST accumulate round to bf16; hand-written reference reached 0.99990, stock 0.99985 | packing 16-bit DEST to a Float16_b interm is lossless, so `block_k_tiles` does not change precision; if PCC is short, the lever is a Float32 `cb_interm_acc` (L1 accumulation in fp32) — record, do not pre-empt |
| Determinism (test_regression) | repeat calls must be bit-identical | one core owns each output tile's whole K sum in a fixed K order; no cross-core adds |
| Counter semaphore reuse across calls | the semaphore must start at 0 each dispatch | semaphores come from `mcast.attach` (initial value 0 per program dispatch); the receiver compares against absolute round `r`, not a reset value |
| DRAM bank count for the column map | a wrong `B` only degrades bank balance (perf), never correctness | take `B` from the device (W's interleaved bank count); lamp measures contiguous vs bank-spread |
| W reader must not stall behind x | at start, x round 0 needs a DRAM read + multicast (~1–2 us) | x arrival is owned by the writer kernel, W reader is independent and `cb_w_block` holds all 12 K-steps |
| Output in L1 interleaved, CBs large (622 KB/worker) | the L1 output buffer (≈12 KB per L1 bank) and CB region must not overlap | allocate y with `ttnn.L1_MEMORY_CONFIG` before building the program; program creation validates CB vs. L1 buffer overlap |
| One dispatch | rule: exactly one device program per call | sender, reader, compute and writer kernels are all in one `ProgramDescriptor`, one `ttnn.generic_op` call |
