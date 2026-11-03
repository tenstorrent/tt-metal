# Operation Design: moe_down_proj

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (grouped matmul, per-row expert selection), one `ttnn.generic_op` dispatch |
| Goal | `y[r] = x[r] @ M_e^T` for every real row r of expert region e, for the Nomic Embed v2 MoE w2 (E = 8, H = 768, F = 3072), in one device program that reads `expert_counts` on the device only, at or under `max(T_compute/0.70, T_DRAM/0.60)` (90.8 us at t = 4096, 74.6 us below). |
| Math | `y[0,0,r,h] = sum_f x[0,0,r,f] * w[0,0,h, e(r)*F + f]` for real rows; `e(r)` from the 32-row-aligned region layout of `counts` |
| Mode | Hybrid: helper-built (`matmul_block` with packer L1 accumulation, `untilize`, `ttnn.Mcast`) plus one additive helper extension (pack-target base offset for `caller_owns_pack_target`) |
| References | `eval/golden_tests/moe_down_proj/feature_spec.py`, `eval/golden_tests/moe_down_proj/helpers.py`, `eval/prompts/moe_down_proj.txt`, `.claude/references/precision_convention.md`, `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp`, `tt_metal/hw/inc/api/compute/cb_api.h`, sibling op evidence `ttnn/ttnn/operations/linear_gelu/op_design.md` and `changelog.md` (same board, same 2D multicast topology, measured) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (x) | `ttnn.Tensor` | yes | (1, 1, R, 3072), R % 32 == 0, R <= 8416; bfloat8_b, TILE, interleaved | - | RT (buffer address) |
| `weight` (w) | `ttnn.Tensor` | yes | (1, 1, 768, 8 * 3072), bfloat8_b, TILE, interleaved | - | RT (buffer address) |
| `expert_counts` | `ttnn.Tensor` | yes | (1, 8), uint32, ROW_MAJOR, interleaved | - | RT (buffer address); values read on device only |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | any fidelity / approx flag; `fp32_dest_acc_en` must be True (gated) | `default_compute_kernel_config()` = LoFi, fp32_dest_acc_en=True, math_approx_mode=False | descriptor |
| `memory_config` | `ttnn.MemoryConfig` | no | interleaved L1 (Phase 0), interleaved DRAM (TARGET) | `input_tensor.memory_config()` | output allocation |

Entry-point rules:

| Rule | Decision |
|------|----------|
| First line | `validate(input_tensor, weight, expert_counts, compute_kernel_config=..., memory_config=...)` |
| Default config | `default_compute_kernel_config()` is an exported factory returning a fresh `ComputeConfigDescriptor(math_fidelity=LoFi, fp32_dest_acc_en=True, math_approx_mode=False)`. The entry resolves `cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()`. A caller config is honored as given; `fp32_dest_acc_en=False` is refused with `UnsupportedAxisValue`, never overridden. |
| Check order in `validate()` | structural `ValueError` checks, then model-constant refusals, then SUPPORTED per axis, then EXCLUSIONS |
| Malformed -> `ValueError` | x not rank 4 or `x.shape[:2] != (1, 1)` or `R % 32 != 0`; weight not rank 4 / `weight.shape[:2] != (1, 1)` / `weight.shape[-1] != E * x.shape[-1]` with `E = expert_counts.shape[-1]`; `expert_counts` not rank 2 / `shape[0] != 1` / dtype not uint32; tensors on different devices |
| Model constants -> `UnsupportedAxisValue` | `F != 3072`, `H != 768`, `E != 8`, `R > 8416` (t > 4096); weight dtype different from x dtype (axis `dtype`); weight or counts not interleaved |
| Forbidden on host | reading `expert_counts` (no `to_torch`, `cpu()`, sync); `ttnn.to_layout`, `ttnn.untilize`, `ttnn.slice`, `ttnn.reshape`, any composed ttnn op. Host work = validation, shape-only parameter derivation, `ttnn.allocate_tensor_on_device` for the output, one `ttnn.generic_op`. |
| Output | new tensor (1, 1, R, 768), bf16, ROW_MAJOR, interleaved, on `memory_config` |
| Program cache | every host-derived quantity is a function of shapes, grid and config only. The same program is correct for any counts; tests overwrite counts in place and re-run. |

### Registry declarations (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `{}` (R is always a tile multiple; no shape-derived axis) |
| Phase 0 `SUPPORTED` (implementer writes it) | dtype [bfloat8_b], layout [TILE], fp32_dest_acc_en [True], memory_layout [INTERLEAVED], buffer_type [L1], out_buffer_type [L1] |
| `EXCLUSIONS` | `[]` |
| Axis sources in `validate()` | dtype / layout / memory_layout / buffer_type from `input_tensor`; out_buffer_type from `memory_config or input_tensor.memory_config()`; fp32_dest_acc_en from the resolved config |

### TARGET axes vs. the Blocking Model

| TARGET value not in Phase 0 | Unlock class | What changes |
|-----------------------------|--------------|--------------|
| buffer_type = DRAM (x) | knob-turn | The x-channel senders read x through a placement-agnostic `TensorAccessor`. Each x tile is read by exactly one sender (its m-group's), so x crosses DRAM once (27.5 MB at R = 8416, in parallel with the 20 MB W stream). Gate flip plus a perf check. |
| out_buffer_type = DRAM | knob-turn | The writer's `TensorAccessor` is placement-agnostic. 12.9 MB written once; with `num_waves >= 2` wave p's store overlaps wave p+1's compute. |

No TARGET value is sharded, so no physical shard pins the blocking.

### Structural impossibilities

None beyond `feature_spec.py` INVALID.

## Tensors

### Input

| Property | input_tensor (x) | weight (w) | expert_counts |
|----------|------------------|------------|---------------|
| Shape | (1, 1, R, 3072) | (1, 1, 768, 24576) | (1, 8) |
| Dtype | bfloat8_b (1088 B per tile) | bfloat8_b | uint32 |
| Layout | TILE | TILE | ROW_MAJOR, one 32 B page |
| Memory | interleaved; L1 (Phase 0), DRAM (TARGET) | interleaved DRAM | interleaved DRAM |

### Output

| Property | Value |
|----------|-------|
| Shape | (1, 1, R, 768) |
| Dtype | bfloat16 |
| Layout | ROW_MAJOR; page = one row = 768 * 2 = 1536 B |
| Memory | interleaved, `memory_config` (default: x's) |
| Rows written | tile rows `[0, active_row_tiles)` only, all 32 rows of each (region padding rows hold computed, unspecified values). Rows `>= 32 * active_row_tiles` are never written. |

Tile geometry (single source: `moe_down_proj_plan.h`, shared by all three kernels; the host computes only the bounds):

| Symbol | Formula | Where |
|--------|---------|-------|
| `tensor_row_tiles` | `R / 32` | host CT arg |
| `tensor_k_tiles` | `3072 / 32 = 96` | constant |
| `tensor_n_tiles` | `768 / 32 = 24` | constant |
| `region_tiles[e]` | `ceil(counts[e] / 32)` | device |
| `region_start[e]` | `sum_{e' < e} region_tiles[e']` (tile rows) | device |
| `active_row_tiles` | `min(sum_e region_tiles[e], tensor_row_tiles)` (the clamp keeps every write inside y for any counts) | device |
| expert of tile row m | the e with `region_start[e] <= m < region_start[e] + region_tiles[e]` | device |
| x page | `m * 96 + k` | - |
| w page (n, e, k) | `n * 768 + e * 96 + k` (tile (n, e*96+k) holds `w[h, e*F+f]` for h in tile row n) | - |
| y write | row `32 * m + i`, byte offset `line_idx * core_n_tiles * 64`, length `core_n_tiles * 64` | - |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Op profile (what the scheme must respect)

| Quantity (t = 4096, R = 8416) | Value | Consequence |
|-------------------------------|-------|-------------|
| Tile products | `active_row_tiles * 24 * 96` = 259 * 2304 = 596,736 (layer 0) | About 5,400 per core at 110 cores; 16 cycles each at LoFi, so 64 us ideal. Compute binds at t = 4096. |
| w | 20.05 MB bfp8, DRAM, 8 experts x 2.5 MB, read regardless of t | Must cross DRAM once. At 448 GB/s that is 44.8 us, about 70% of the compute time, so the W stream must run continuously under compute. Below t = 4096 it binds. |
| x | 27.5 MB, L1 interleaved | Each x tile meets all 24 output columns. Unicast replication to the cores that split N saturates NoC0 at 8x or 12x (sibling op: 11x x unicast cost 2x end-to-end). x must be multicast. |
| y | 12.9 MB bf16 RM, L1 | Every core writes `core_n_tiles * 64` B segments of rows it owns; segment count is the writer cost. |
| Routing | counts on device, skewed up to 3.5x the mean expert | Any static expert-to-core assignment is 3.5x imbalanced. The row split must be computed on device from counts, and the multicast topology must not depend on counts. |
| x and y resident in L1 | 250 KB + 118 KB per bank at R = 8416 | CB budget about 800 KiB per core. |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| rows `m` (expert-sorted tile rows, flattened across regions) | independent: each output row depends only on its own x row and its expert's matrix | `core_row_tiles` (tile rows per core per wave; the M extent of one wave's block) | `ceil(ceil(active_row_tiles / num_waves) / num_m_groups)` on device; host bound `core_row_tiles_max = ceil(ceil(tensor_row_tiles / num_waves) / num_m_groups)` (15 at t = 4096) | device plan from counts (`moe_down_proj_plan.h`); bound from host `select_blocking()` as CT arg | across `num_m_groups` m-groups, contiguous: group g owns wave rows `[wave_start + g * core_row_tiles, ...)`. Not counts-dependent in topology, only in extent. | knob-turn |
| waves (time split of `m`) | independent (a partition of `m`) | `num_waves` | 1 or 2 from `select_blocking()` (2 at t >= 3072, 1 below) | host `select_blocking()`, CT arg | every core takes part in every wave | knob-turn |
| output columns `n` | independent; w is reuse-shared along `m` (its slice does not vary with the row) | `core_n_tiles` (N tiles per n-line) | `24 / num_n_lines` = 2 (L layout) or 3 (rectangle) | host `select_layout()`, CT arg | across `num_n_lines` n-lines; every core in an n-line holds the same n-slice | knob-turn |
| contraction `k` | dependent: every output element sums over all of it | `k_block_tiles` (K tiles per round) | 8 (every flagged shape) | host `select_blocking()` (largest divisor of 96 <= `K_BLOCK_MAX` = 8 that fits), CT arg | not split across cores: whole K per core, streamed in `num_k_rounds = 96 / k_block_tiles` rounds, accumulated in fp32 L1 partials | scheme-change (R7 K-split, rejected) |
| experts `e` | independent of each other; selects the matrix per row (not a data axis of x or y) | expert slots per round = `E` = 8 (nominal) | 8 slots per round, empty or out-of-wave experts skipped (no read, no transfer) | constant `NUM_EXPERTS` | every n-line streams every expert present in the wave | - |
| x across n-lines | reuse-shared: x row m is the same for all `num_n_lines` cores that share m | x multicast group size | `num_n_lines` | host layout | one x multicast group per m-group | - |
| w across m-groups | reuse-shared: an expert's n-slice is the same for all m-groups | W multicast group size | `num_m_groups` | host layout | one W multicast group per n-line | R2 prunes to the cores that need the expert |

Why `k_block_tiles = 8`. Each round packs and L1-accumulates every partial tile once (fp32, 4 KB). At LoFi a tile product is about 16 cycles, so with 8 K tiles per round the matmul spends about 128 cycles per partial per round, at or above the packer's fp32 L1-accumulate cost. At 4 the packer would bind. `K_BLOCK_MAX = 8` caps the fill (round 0 must arrive before compute starts), see lamp L1.

Why `num_waves = 2` at t >= 3072. At one wave the fp32 partials (`core_row_tiles_max * core_n_tiles * 4 KB` = 245 KB at t = 4096) plus two rounds of x and w exceed the budget at `k_block_tiles = 8`. Two waves halve the partials and let wave p's store overlap wave p+1's compute. The cost is at most `num_waves - 1` experts' w read twice (one expert can straddle a wave boundary): +2.5 MB DRAM at t >= 3072. `select_blocking()` takes the smallest `num_waves` at which `k_block_tiles = K_BLOCK_MAX` fits.

Why a wave is not cut finer by expert. K-major rounds keep every core's per-round work equal (its rows x its n-slice x one K block) regardless of which experts its rows belong to. An expert-major stream would make a core whose rows belong to expert 7 wait for experts 0..6 to be streamed, paced by their consumers (regime R5).

Stage-data re-check. The only internal datum is the fp32 partial block (`core_row_tiles x core_n_tiles`) of one core. It is finalized on the core that accumulated it, untilized by that core's compute and stored by that core's writer. There is no coordinating core. Every multicast sender (rotating, see Work Distribution) computes the same share as every other core; sending is a communication role, not a work assignment. No stage runs on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_round` | `X_DEPTH` (rounds) | 2 | round j+1's x multicast lands while round j computes |
| `cb_w_round` | `W_DEPTH` (rounds of `NUM_EXPERTS` chunk slots) | 2 | round j+1's w chunks land while round j computes; the DRAM stream runs up to one round ahead |
| `cb_partials` | none (wave accumulator) | 1 wave block | resident accumulator; recycled per wave |
| `cb_output_rows` | `OUT_DEPTH` (untilized tile-row blocks) | 2 | writer drains block b while compute untilizes block b+1 |
| `cb_counts_compute`, `cb_counts_writer` | none | 1 page each, constant | - |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| fp32 DEST, half-sync | matmul output subblock `sb_h * sb_w <= 4` | per call: largest area with `sb_w \| core_n_tiles`, `sb_h \| call_rows`, area <= 4. An odd sub-block is cut into an even part and a 1-row tail (two calls), so all but one tile row use area 4 (2x2 at `core_n_tiles` = 2). Re-init (`InitMode::Short`) only when the subblock shape changes. | DEST overrun, silently wrong tiles |
| `matmul_block` shape legality (`matmul_block_helpers.hpp:139-170`) | `in0_num_subblocks = call_rows / sb_h`, `in1_num_subblocks = core_n_tiles / sb_w` exact | as above | partial subblocks unpacked from the wrong tiles |
| `caller_owns_pack_target` pack offsets (`matmul_block_helpers.inl:443-448, 484-489`) | a sub-block packs into rows `[r0, r0 + rows)` of `cb_partials` | helper extension `out_base_offset = r0 * core_n_tiles` (see API Mapping) | every sub-block packs onto row 0 of the partials |
| CB ring contiguity (`ttnn-cb-memory-fundamentals.md`, ring-wrap invariant) | a matmul's in0 / in1 window must not straddle the ring end | every round pushes and pops the nominal `core_row_tiles_max * k_block_tiles` (x) and `NUM_EXPERTS * core_n_tiles * k_block_tiles` (w) pages, empty slots included; capacity is `DEPTH x` that | a straddling window reads wrapped garbage |
| `active_row_tiles` vs y | rows written | device clamp `active_row_tiles = min(sum region_tiles, tensor_row_tiles)` | malformed counts write past y |
| `core_row_tiles <= core_row_tiles_max` | per-core rows vs CB capacity | holds by construction (`active_row_tiles <= tensor_row_tiles`) | CB overflow |
| `cb_partials` precision | fp32 page format, `fp32_dest_acc_en = True` only | `validate()` refuses `False` | bf16 partials truncate the K accumulation every round |
| Multicast destination address | x / w land at the same L1 offset on every receiver | `cb_x_round`, `cb_w_round` identical index and size on every participating core; slot address derived from the uniform push sequence | data lands outside the CB |
| x / w rotating sender rounds | all cores must count channel rounds identically | channel round = global step index from the plan (skipped chunks consume no round) | senders and receivers disagree, hang |
| L1 budget | `core_row_tiles_max`, `k_block_tiles`, `num_waves` | `select_blocking()` checks `footprint <= CB_BUDGET_BYTES` (800 KiB), raises `RuntimeError` naming the grid if no `(num_waves <= MAX_WAVES = 4, k_block_tiles >= 1)` fits (unreachable at R <= 8416 on p100a and WH) | CB clash with the L1 x / y buffers |

### Regimes

Minimum (DRAM boundary): w crosses DRAM once (20.05 MB), x once (0 in Phase 0: L1), y once (12.9 MB to L1). Cross-core minimum given the 2D split: x multicast once per m-group, w multicast once per n-line.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 line-lockstep 2D, K-streamed, static multicasts** (rows split contiguously over m-groups on device; n split over n-lines; every round j: x rows of the m-group multicast along the m-group, every in-wave expert's w chunk (`k_block_tiles x core_n_tiles`) multicast along the n-line by a rotating sender; per core one `matmul_block` per expert sub-block, packer L1 accumulation into fp32 partials; per wave untilize + RM store) | **built** | every supported call | per wave per core: in0 `core_row_tiles x k_block_tiles` per round, in1 `k_block_tiles x core_n_tiles` per expert per round, partials `core_row_tiles x core_n_tiles` | w: 1 DRAM crossing + at most `num_waves - 1` straddling experts again (+2.5 MB at 2 waves); multicast to `num_m_groups - 1` receivers. Above minimum: every core receives every in-wave expert's chunk, about 8x the w bytes it uses (R2). x: read once by one sender per m-group, multicast to `num_n_lines - 1` receivers. y: once. | `k_block_tiles`: fewer rounds, so fewer fp32 L1-accumulate packs of every partial (one per round), fewer matmul calls / reconfigs (one set per sub-block per round), fewer channel rendezvous (1 x + up to 8 w per round). Longer fill. `core_row_tiles` (fewer waves): fewer w straddle re-reads, fewer fills and untilize passes, but larger partials. |
| R2 range-pruned w multicast (R1, but expert e's chunk goes only to the n-line cores whose rows touch e: a runtime sub-rectangle of the line) | deferred: `ttnn.Mcast` groups are static equal consecutive groups (`mcast.hpp:121-137`), so R2 needs a raw semaphore protocol with per-expert ready counters and runtime fan-out. It saves only receiver ejection bytes (w receive per core about 1.9 MB -> about 0.3 MB); DRAM bytes and link bytes are unchanged, and the predicted per-core receive in R1 (about 4.6k tiles, 5.0 MB over at least 68 us of compute, about 74 GB/s across two NoCs) is under the ejection capacity. Reachable: the plan already computes each core's expert set, and the w slot layout is unchanged. | - | as R1 | w receive per core: `(experts touching the core) x 96 x core_n_tiles` tiles instead of `(in-wave experts) x 96 x core_n_tiles` | as R1 |
| R3 x unicast (each core reads its x rows from L1 itself) | rejected: superseded by R1's x multicast. x NoC reads `num_n_lines`x = 330 MB at 12 lines, about 1.1-1.6 GB of link traffic in 70 us, above NoC0 capacity. Sibling op measured 2x end-to-end for 11x. | - | - | x: `num_n_lines` L1 crossings | - |
| R4 w unicast (each core reads its experts' w from DRAM) | rejected: superseded by R1. DRAM w = `(num_m_groups + 7) x 2.5 MB` = 40 MB at 9 groups, 89 us at peak. | - | - | w: about 2x DRAM | - |
| R5 expert-major schedules (a: all cores process expert e together, rows of e split over m-groups; b: flattened rows with an expert-major w stream) | rejected: (a) per-expert rounding costs +8% to +17% compute at real counts (layer 3: 28 vs 24 rows per core at 11 groups) and small experts' phases are DRAM-bound with idle cores; (b) a core with late-expert rows waits for every earlier expert to be streamed and consumed. Superseded by R1 (K-major rounds, equal per-round work). | - | - | same as R1 | - |
| R6 dynamic expert-stationary (cores allocated to experts in proportion to counts, each holding a w slice) | rejected: superseded by R1. Small experts need their whole 2.5 MB read by one or two cores, and large experts need an M split that re-reads w per M part. | - | - | w DRAM concentrated on few cores, or re-read | - |
| R7 K-split + cross-core combine (dependent axis) | rejected: superseded by R1. R1 fills 108 of 110 cores at every flagged shape. Where rows under-fill the grid (t < 300), the op is bound by the fixed 20 MB w stream, which a K-split does not shorten. The combine adds `(G_k - 1) x 6,200 tiles x 4 KB` >= 25 MB of fp32 partial traffic at G_k = 2. | - | - | adds fp32 partial NoC traffic | - |
| R8 1D row split (every core computes full rows, N unsplit) | rejected: superseded by R1. Each core needs 2.5 MB of w per expert it touches, about 0.4 w tiles per tile product. | - | - | w receive 2.5-7.5 MB per core | - |

R1's domain is every supported call. R2 changes only who receives w chunks. R3 to R8 are superseded on bytes or balance.

### Traffic ranking

Per t = 4096 call (R = 8416, 108 cores, 2 waves). Structural only.

| Candidate split | DRAM crossings | Cross-core NoC | Occupancy (separate) | Rank |
|-----------------|---------------|----------------|----------------------|------|
| 2D + both multicasts + range-pruned w (R2) | w 22.5 MB; x 0 (L1) | x 27.5 MB read once, multicast to 11 (+1); w multicast to about 1.4 cores per expert chunk | 108 / 110 | 1 |
| 2D + both multicasts, static line groups (R1, built) | w 22.5 MB | x as above; w 22.5 MB multicast to 8 receivers per line (about 1.9 MB received per core) | 108 / 110 | 2 |
| expert-major 2D (R5a) | w 20 MB | same as R1 | rounding: busiest +8% to +17% | 3 |
| 2D x unicast (R3) | w 22.5 MB | x 330 MB unicast | 108 / 110 | 5 |
| 2D w unicast (R4) | w 40 MB | x multicast | 108 / 110 | 6 |
| K-split + combine (R7) | w 20 MB | + >= 25 MB fp32 partials | no gain (grid full) | 4 |
| 1D rows (R8) | w 20 MB | w 2.5-7.5 MB received per core | 110 / 110 | 7 |

Chosen: R1. R2 is cheaper only in receiver-side w bytes, with no DRAM or link difference; it is deferred with that reason.

### Stall-shadow check

| Waiting stage | Waits for | Independent work scheduled into the stall | Legality |
|---------------|-----------|-------------------------------------------|----------|
| compute, round j, expert slot s | w chunk (j, s) | the sub-blocks of slots < s in the same round: compute waits cumulatively per slot (`wait_front((s+1) * w_chunk_tiles)`), so expert 0's sub-block starts before expert 1's chunk lands | no arithmetic reordered |
| compute, round 0 of a wave | x round 0 + w chunk (0, first slot) | with `num_waves >= 2`: nothing on compute; the reader prefetches wave p+1 round 0 into the free `X_DEPTH` / `W_DEPTH` slots while compute untilizes wave p | the partial accumulation order (K rounds ascending) is unchanged |
| writer, wave p store | untilized blocks of wave p | wave p+1's matmuls run on compute while the writer drains wave p | elementwise store, order-free |
| the last wave's untilize + store | the last round | nothing independent remains. Every output depends on the last K round, so the last wave's drain is exposed; `num_waves` bounds it to `1 / num_waves` of the output (lamp L2) | - |
| w sender | DRAM read of its chunk | the sender issues its next owned chunk's DRAM read before the current chunk's handshake (lookahead, lamp L4; sibling Perf 2: -2.3% at 8x512, -9% at small M) | data movement only |
| x sender | consumer-ready from its m-group | the next round's x reads for the next owned round | data movement only |

Reordering the K rounds per core (for example a per-line start offset to spread DRAM banks) would change the fp32 summation order. It is not done in Phase 0, and any such reorder is gated on the precision baseline.

### Block schedule

```cpp
// every kernel runs the same plan, derived from counts on device (moe_down_proj_plan.h)
for (uint32_t wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
    plan_wave(wave_idx);                                     // rows, expert sub-blocks, in-wave expert slots
    open_partials(wave_idx);                                 // compute
    for (uint32_t round_idx = 0; round_idx < num_k_rounds; ++round_idx) {
        deliver_x_round(wave_idx, round_idx);                // reader, x channel
        deliver_w_round(wave_idx, round_idx);                // reader, w channel, one chunk per in-wave expert slot
        accumulate_round(wave_idx, round_idx);               // compute, one matmul per expert sub-block
        release_round(wave_idx, round_idx);                  // compute pops nominal x / w pages
    }
    untilize_wave(wave_idx);                                 // compute
    store_wave(wave_idx);                                    // writer
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `plan_wave` | 8 counts -> `active_row_tiles`, this core's rows, sub-blocks `(expert, r0, rows)`, in-wave expert slots, channel round indices | `cb_counts_*` | once per wave per kernel; counts read once per kernel (reader: two 32 B DRAM reads, one per counts CB) |
| `open_partials` | `core_row_tiles_max x core_n_tiles` fp32 | - | one `reserve_back` of the nominal page count per wave |
| `deliver_x_round` | `core_row_tiles_max x k_block_tiles` slot; `core_row_tiles x k_block_tiles` filled | the other `X_DEPTH` slot | one channel round per round per m-group: sender issues `core_row_tiles * k_block_tiles` x reads then one barrier, one multicast `send()` (+ the strip `send()` in the L layout); receivers one `receive()`; every core one `push_back` of the nominal pages. An m-group with no rows still completes the channel round (signal only). |
| `deliver_w_round` | per in-wave expert: `k_block_tiles x core_n_tiles` chunk in slot e | the other round's slots | one channel round per in-wave expert per round per n-line: sender issues `core_n_tiles` runs of `k_block_tiles` contiguous w pages, one barrier, one `send()`; receivers one `receive()`. Every core pushes one chunk per slot, all 8 slots (empty and out-of-wave slots pushed with no transfer). |
| `accumulate_round` | per sub-block: in0 `rows x k_block_tiles`, in1 `k_block_tiles x core_n_tiles`, out `rows x core_n_tiles` into partials rows `[r0, r0+rows)` | partials (whole wave), w round region | one `matmul_block` per sub-block (two when an odd sub-block is cut for subblocking); matmul short-init + data-format reconfig on the first call of each wave and when the subblock shape changes, none otherwise; packer L1 accumulate on for rounds > 0; one cumulative `wait_front` on `cb_w_round` per sub-block |
| `release_round` | nominal x and w round pages | - | compute pops the x remainder (`(core_row_tiles_max - core_row_tiles) * k_block_tiles`) and `NUM_EXPERTS * w_chunk_tiles` w pages once per round |
| `untilize_wave` | `core_row_tiles` tile rows x `core_n_tiles`, fp32 tiles -> bf16 RM | - | one `push_back` of the nominal partials pages, then one `untilize` call over `core_row_tiles` blocks (init + reconfig once per wave), then pop of the remainder |
| `store_wave` | per tile row: 32 row segments of `core_n_tiles * 64` B | - | per block: one `wait_front(core_n_tiles)`, 32 writes, one `noc_async_writes_flushed`, one `pop_front`; one barrier at kernel end |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap / pack: `k_block_tiles` | 8 is set from an assumed fp32 L1-accumulate pack cost. If the packer binds, larger K blocks pay; if the fill dominates (round 0 = 120 x tiles + up to 8 x 16 w tiles), smaller K blocks pay | `k_block_tiles` in {4, 6, 8, 12} with `num_waves` re-solved; zones: compute busy vs pack busy |
| L2 overlap: `num_waves` | 2 waves cost one straddling expert's w (+2.5 MB DRAM, about +5.6 us of DRAM time at t <= 3072 where DRAM binds) and buy store overlap plus halved partials | `num_waves` in {1 (k_block 4), 2, 3} at t = 4096, 3072, 2048 |
| L3 grid synchronization: w handshake granularity | up to 8 w rendezvous per round per line; at small t most compute is tiny and the rendezvous dominate | one multicast per round of all in-wave slots (one rendezvous) vs per-chunk, at t = 1024 and t = 4096 |
| L4 w sender lookahead | the w stream is DRAM-bound below t = 4096 and only about 30% under compute at t = 4096; a sender that reads only after its turn leaves DRAM idle | rolling lookahead 1 or 2 chunks per sender with per-chunk trid barriers (sibling Perf 2 `w_delivery_pipeline`), `W_DEPTH` 2 vs 3 |
| L5 data-movement RISC split | both channels share NCRISC; a sender blocked on a handshake stalls the other channel | x channel on NCRISC / NoC0, w channel on BRISC / NoC1 with the writer (catalog `split_reader`; sibling measured NoC1 DRAM reads slower) |
| L6 store issue cost | 32 * `core_row_tiles` segment writes of 128 B per wave per core | stateful `noc_async_write_one_packet_with_state` writes; gathering an m-group's segments into full 1536 B rows on one core |
| L7 x sender bank order | the 12 cores of an m-group are fed by one sender, but all m-group senders read rows that start at the same k, so they hit the same L1 banks at the same time | start each sender's k walk at `(group * stride) mod k_block_tiles` (catalog `bank_stagger`) |
| L8 subblock shape | area-4 2x2 plus a 1-row tail vs uniform 1x2 (no tail call); unpack-bound at LoFi is possible | both, at t = 4096 |

### Catalog evidence

| Knob | Entry | What it set |
|------|-------|-------------|
| x and w multicast on a 2-D split | `mcast_topology`, `shared_input_reuse` | broadcast each operand along the axis it does not vary with; two channels on one grid at disjoint semaphore ids; rotating senders spread DRAM reads |
| one barrier per chunk read | `double_buffer` | the senders issue a whole chunk of reads before one barrier |
| w sender lookahead (L4) | `trid_double_issue` | per-chunk trid barriers keep reads in flight across a wait |
| subblock area 4 at fp32 DEST | `matmul_output_subblock` | size, not shape, sets the win; fp32 DEST caps at 4 |
| packer L1 accumulation for partials | `eltwise_l1_vs_dest_accumulate` | accumulate onto the partials with the packer instead of reloading them each round |
| sender placement | `noc_placement` | rotating senders spread over the line instead of one column of DRAM readers |
| sender bank order (L7) | `bank_stagger` | measured 1.32x on BH when all cores walk the same banks |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| D0 counts -> `cb_counts_compute`, `cb_counts_writer` | uint32 x 8 (32 B page) | reader: `TensorAccessor` read of counts page 0 into each CB, one barrier, `push_back(1)` on each | compute reads the 8 values with `read_tile_value(cb_counts_compute, 0, e)` (`cb_api.h:202`), identical on all three TRISCs. Writer waits its own CB. Reader parses its own landing copy before the push. |
| D1 x -> `cb_x_round` | bfp8_b tiles | x channel (`ttnn.Mcast`, rotating sender within the m-group): sender reads x pages `m * 96 + j * k_block_tiles + kk` for its group's rows into in0 position `(r_local * k_block_tiles + kk)` of the round slot, one barrier, `send(src, dst, core_row_tiles * k_block_tiles * 1088)`; receivers `receive(round)` | the sender's own copy is its slot (sender is a participant) |
| D2 w -> `cb_w_round` | bfp8_b tiles (untransposed as stored) | w channel (`ttnn.Mcast`, rotating sender within the n-line, channel round = global chunk index): sender reads pages `n * 768 + e * 96 + j * k_block_tiles + kk` for the line's `core_n_tiles` n into in1 position `slot_base(e) + kk * core_n_tiles + n_local`, one barrier, `send()` | in1 is K x N row-major; each stored tile is `w[h, f]` with h on rows, so compute transposes each in1 tile (`transpose = true`) |
| C1 matmul + K accumulation | DEST fp32 -> fp32 L1 partials | `matmul_block<transpose=true, packer_l1_acc=true, Interm, TileRowMajor, caller_owns_pack_target=true, seeded = (round_idx > 0)>` per sub-block, `out_base_offset = r0 * core_n_tiles` | x is in0 (SrcB) and w is in1 (SrcA), the same operand-to-source mapping as `ttnn.matmul(x, w^T)`. LoFi truncation therefore lands on the same operand as stock. |
| C2 untilize | fp32 tiles -> bf16 RM | `untilize<core_n_tiles, cb_partials, cb_output_rows>(core_row_tiles)` with `UnpackAndPackReconfigure` | one tile row (32 rows x `core_n_tiles * 32`) per block |
| W1 `cb_output_rows` -> y | bf16 RM | writer, `TensorAccessor` on y; for tile row m and row i: `noc_async_write(block_l1 + i * seg_bytes, y.get_noc_addr(32 * m + i, line_idx * seg_bytes), seg_bytes)`, `seg_bytes = core_n_tiles * 64` | one flush per block before `pop_front`; final barrier |
| R2-regime w delivery (deferred) | bfp8_b | raw `noc_async_write_multicast` to the line's sub-rectangle `[first core touching e, last core touching e]`, per-expert ready counters on the sender, data-ready counter per expert on receivers | slot layout and push sequence identical to R1, so R2 replaces only the multicast destination and handshake |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one wave's `core_row_tiles x core_n_tiles` output block per core, accumulated over `num_k_rounds` rounds |
| Grid | `grid = device.compute_with_storage_grid_size()` (p100a 11 x 10). Host `select_layout(grid, tensor_row_tiles)` returns the n-lines (ordered core lists) and m-groups. Never literal. |
| Per-core work | core (line l, group g): rows `[wave_start + g * core_row_tiles, min(wave_start + (g+1) * core_row_tiles, wave_end))` of every wave, columns `[l * core_n_tiles, (l+1) * core_n_tiles)` |
| Remainder | wave rows: `wave_row_tiles = ceil(active_row_tiles / num_waves)`, last wave shorter; last m-group of a wave may have fewer rows or none. All on device; push/pop counts stay nominal. |

Layout family (`select_layout`). Score = `ceil(tensor_row_tiles / num_m_groups) * (24 / num_n_lines)` (busiest core's output tiles); lowest score wins, ties keep the rectangle.

| Layout | n-lines | m-groups | Channels | p100a |
|--------|---------|----------|----------|-------|
| Rect | `num_n_lines` = largest divisor of 24 <= `grid.y`; line l = physical row y = l, x = 0..gx-1 | `num_m_groups = grid.x`; group g = column x = g, y = 0..num_n_lines-1 | w: one `Mcast` over the rectangle, RowMajor groups of `grid.x`; x: one `Mcast`, ColumnMajor groups of `num_n_lines` | 8 lines x 11 groups = 88 cores, `core_n_tiles` 3, score 72 at R = 8416 |
| L | `num_n_lines = grid.x + 1` (must divide 24); lines 0..gx-1 = columns x = l, y = 0..h-1 with `h = grid.y - 1`; line gx = strip y = gy-1, x = 0..h-1 (requires h <= grid.x) | `num_m_groups = h`; group g = row y = g (x = 0..gx-1) plus strip core (g, gy-1) | w_A: rect `(0..gx-1, 0..h-1)` ColumnMajor groups of h, rotating; w_B: strip, one group of h, rotating; x_A: rect, RowMajor groups of gx, rotating; x_B: strip cores, groups of 1, `McastRotatingSenderConfig(sender_cores = rect, sender_order = RowMajor)` so group g's senders are row g, rotating in step with x_A | 12 lines x 9 groups = 108 cores, `core_n_tiles` 2, score 60 at R = 8416. Selected on p100a. |

In the L layout the x sender of round s in m-group g sends twice: the row multicast (x_A) and the strip copy (x_B), both from the same slot. Both channels use the same round counter and order, so the same core sends on both.

Concrete values (p100a, L layout, `CB_BUDGET_BYTES` = 800 KiB, `K_BLOCK_MAX` = 8):

| t | tensor_row_tiles | num_waves | k_block_tiles | core_row_tiles_max | busiest rows (layer 0 / scaled counts) | footprint |
|---|------------------|-----------|---------------|--------------------|-----------------------------------------|-----------|
| 4096 | 263 | 2 | 8 | 15 | 2 x 15 (T = 259) | 670,784 B |
| 3072 | 199 | 2 | 8 | 12 | 2 x 11 | 593,984 B |
| 2048 | 135 | 1 | 8 | 15 | 15 | 670,784 B |
| 1024 | 71 | 1 | 8 | 8 | 8 | 491,584 B |
| 1 | 8 | 1 | 8 | 1 | 1 (2 groups active) | 312,384 B |

Single regime (R1) with two layouts. Regime-pinned tests (implementer, private override of `select_layout` in the program-descriptor builder, not in the public API): both layouts on p100a, at t = 1 (most groups empty), t = 37, t = 1858, t = 4096 layer 3, one-expert and zeros-between counts.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_x_round` | 0 | bfp8_b tile (1088 B) | `X_DEPTH * core_row_tiles_max * k_block_tiles` | spans m (whole wave block), k (one round); streams over n (shared by all sub-blocks) and over rounds | Bfp8_b | reader | compute | whole kernel, recycled per round |
| `cb_w_round` | 1 | bfp8_b tile | `W_DEPTH * NUM_EXPERTS * core_n_tiles * k_block_tiles` | spans n (line slice), k (one round), the expert slots; streams over m and over rounds | Bfp8_b | reader | compute | whole kernel, recycled per round |
| `cb_partials` | 2 | fp32 tile (4096 B) | `core_row_tiles_max * core_n_tiles` | spans m and n of the wave block; k folded by packer L1 accumulation; streams over waves | Float32 | compute (pack) | compute (unpack, untilize) | per wave |
| `cb_output_rows` | 3 | bf16 tile (2048 B) | `OUT_DEPTH * core_n_tiles` | one untilized tile row, double-buffered; streams over m | Float16_b | compute | writer | per wave |
| `cb_counts_compute` | 4 | 32 B | 1 | constant | UInt32 | reader | compute | whole kernel |
| `cb_counts_writer` | 5 | 32 B | 1 | constant | UInt32 | reader | writer | whole kernel |

Every CB has the same index and size on every core of the layout, so multicast slot addresses agree.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `plan_wave` | counts -> plan | op-local header `moe_down_proj_plan.h` (pure integer code, shared by the 3 kernels) | `cb_counts_compute` / `cb_counts_writer` (1, fronted, never popped until kernel end) | - | counts stay fronted |
| 1 | `open_partials` | `core_row_tiles_max x core_n_tiles` | `cb_reserve_back` | - | `cb_partials` (reserved, write pointer fixed) | reserved |
| 2 | `deliver_x_round` | `core_row_tiles_max x k_block_tiles` | `ttnn.Mcast` / `McastArgs` + `TensorAccessor` | x tensor | `cb_x_round` (+nominal round) | one round fronted |
| 3 | `deliver_w_round` | 8 slots x `k_block_tiles x core_n_tiles` | `ttnn.Mcast` / `McastArgs` + `TensorAccessor` | w tensor | `cb_w_round` (+1 chunk per slot) | one round fronted |
| 4 | `accumulate_round` | per sub-block `rows x k_block_tiles` x `k_block_tiles x core_n_tiles` | helper `matmul_block` (+ extension) | `cb_x_round` (rows x k, popped per call); `cb_w_round` (cumulative wait, NoWaitNoPop) | `cb_partials` (rows `[r0, r0+rows)`, packed with L1 acc) | x front advanced by the sub-block |
| 5 | `release_round` | nominal round pages | `cb_pop_front` | `cb_x_round` (remainder), `cb_w_round` (8 chunks) | - | both rounds freed |
| 6 | `untilize_wave` | `core_row_tiles x core_n_tiles` | `cb_push_back` + helper `untilize` | `cb_partials` (nominal push, popped per block, remainder popped) | `cb_output_rows` (`core_n_tiles` per block) | partials empty |
| 7 | `store_wave` | 32 rows x `core_n_tiles * 64` B per block | dataflow `TensorAccessor` | `cb_output_rows` | y | drained |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_round, cb_w_round, cb_partials)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60`; Reverse requirement `matmul_block_helpers.hpp:100-106` | once at top of `kernel_main` | - | - | - |
| `plan_wave` (compute) | raw_api | `read_tile_value(cb_counts_compute, 0, e)` | `tt_metal/hw/inc/api/compute/cb_api.h:202-216` | e = 0..7 after `cb_wait_front(cb_counts_compute, 1)` | `cb_counts_compute` | - | - |
| `accumulate_round` | helper | `matmul_block<true /*transpose*/, true /*packer_l1_acc*/, LastBlockTarget::Interm, OutputCBLayout::TileRowMajor, InitMode::Short (first call of a wave or subblock shape change) / None, InputPolicy::WaitAndPopPerKBlock, InputPolicy::NoWaitNoPop, NoPostCompute, NoPreKBlock, NoPostKBlock, 0, NoKBlockInnerDimFn, NoIn0Source, ExpertSlotOffset, true /*caller_owns_pack_target*/, round_idx > 0 /*seeded_pack_target*/, NoneActivation, INPUT_AND_OUTPUT (first) / NONE>` | `matmul_block_helpers.hpp:342-375`; caller-owned target contract `matmul_block_helpers.inl:132-145`; in1 offset `.inl:266`; pack offsets `.inl:443-448, 484-489` | `MatmulBlockShape::of(rows / sb_h, core_n_tiles / sb_w, sb_h, sb_w, k_block_tiles, 1)`; `in1_per_core_w = out_row_width = core_n_tiles`; `out_buf = interm_buf = cb_partials`; `ExpertSlotOffset{slot * core_n_tiles * k_block_tiles}`; `out_base_offset = r0 * core_n_tiles` (extension) | `cb_x_round`, `cb_w_round` | `cb_partials` | `rows` (sub-block of `core_row_tiles`), `k_block_tiles`, `core_n_tiles`; `sb_h`, `sb_w` derived per call, not design knobs |
| pack-target base offset (extension) | helper (additive) | runtime arg `uint32_t out_base_offset = 0` on `matmul_block`, added to `row_base` at both caller-owned pack sites; asserted zero unless `caller_owns_pack_target` | `matmul_block_helpers.inl:443-448, 484-489` | default 0: no behavior change for existing callers | - | - | - |
| `untilize_wave` | helper | `untilize<core_n_tiles, cb_partials, cb_output_rows, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(core_row_tiles)` | `untilize_helpers.hpp:145-154`; reconfig rule `untilize_helpers.hpp:55-62` | per wave | `cb_partials` | `cb_output_rows` | `core_n_tiles` (CT), `core_row_tiles` (RT) |
| `deliver_x_round` / `deliver_w_round` | helper (dataflow + host) | host `ttnn.Mcast(device, McastConfig(noc, handshake=True), receivers, group_size, McastRotatingSenderConfig(...), order)` + `attach`; kernel `McastArgs<CT, RT>()`, `optional_sender(noc)->send(src, dst, bytes)`, `optional_receiver(noc)->receive(round)` | `mcast/host/mcast.hpp:83-137`; `mcast/kernel/mcast_args.hpp:18-56, 218` | channels per layout (Work Distribution) | - | `cb_x_round`, `cb_w_round` | `k_block_tiles`, `core_row_tiles` (bytes per send) |
| x / w / counts reads | raw_api (dataflow) | `TensorAccessor` page reads, `noc_async_read_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` | - | - | `cb_x_round`, `cb_w_round`, `cb_counts_*` | - |
| `store_wave` | raw_api (dataflow) | `TensorAccessor::get_noc_addr(page, offset)` + `noc_async_write`, `noc_async_writes_flushed`, final `noc_async_write_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` | - | `cb_output_rows` | - | - |

Helpers considered and rejected:

| Candidate | Phase | File:Line | Why it cannot express the planned shape |
|-----------|-------|-----------|-----------------------------------------|
| `matmul_block` `LastBlockTarget::OutWithUntilize` | untilize fused into the last K block | `matmul_block_helpers.hpp:53-62` | requires SubblockMajor and the full K inside one call; here the K sum spans `num_k_rounds` separate calls accumulating in L1, and the last round's pack must add onto the L1 partials, which an untilizing pack cannot do. |
| `matmul_block` with `num_k_blocks = num_k_rounds` (one call per sub-block over all K) | K accumulation | `matmul_block_helpers.hpp:139-170` | a core with rows of 2 or more experts would consume all of expert e's K blocks before expert e+1's, which forces an expert-major w stream (regime R5b) or buffering a full-K w slice per expert. |
| `reblock_and_untilize` | untilize | `reblock_untilize_helpers.hpp` | gathers SubblockMajor output; the partials are already TileRowMajor, so plain `untilize` covers it. |
| `In0SourceFn` for sub-block selection | in0 rows of a sub-block | `matmul_block_helpers.hpp:217-224` | selects a CB, not an offset; sub-blocks are consumed in row order with `WaitAndPopPerKBlock`, so the front of `cb_x_round` is always the next sub-block. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Host reads counts | breaks trace safety and the contract | host derives nothing from counts. Every kernel reads counts on device. Acceptance test overwrites counts in place and re-runs the cached program. |
| Wrong operand on SrcA at LoFi | LoFi truncates one source's mantissa; swapping x and w changes the error and can miss rel RMS <= 2.0e-2 (stock 1.90e-2) | x = in0 (SrcB), w = in1 (SrcA), matching `ttnn.matmul(x, w^T)`. Precision baseline test on golden inputs. |
| Missing in1 transpose | w tiles hold `w[h, f]` (h on rows); in1 needs K on rows | `transpose = true` on `matmul_block` (unpack-side SrcA transpose) |
| Global step order | rotating senders and two channels on one RISC deadlock if any core walks channel rounds in a different order | one plan function: for each wave, for each round, x round then w chunks in slot order, skipping out-of-wave and empty experts identically on every core |
| Sub-block packs onto row 0 | `caller_owns_pack_target` folds only the M-subblock base (`.inl:443-448`) | additive `out_base_offset` extension; test with counts whose region boundaries fall inside a core's rows (t1858 multiples-of-32 + 1, layer 3) |
| Ring straddle in x / w windows | per-call windows of varying size | nominal pushes / pops per round; capacity a whole number of rounds |
| Zero-row cores and groups | t = 1 leaves 7 of 9 m-groups empty; zero-row cores still sit in their n-line's w channel | every core runs every channel round; an empty m-group's x round is signal-only; compute issues no matmul and untilizes 0 blocks |
| Malformed counts | sum of padded regions above R | device clamp of `active_row_tiles`; never writes past row R |
| x_B rotating sender with one-core groups | the sibling op found the rotating `Mcast` refuses a one-core group with in-group rotation | x_B uses a separate sender grid (`sender_cores` = rect, partitioned per group). If `ttnn.Mcast` still refuses it, fall back to the Rect layout on that grid (the selection function keeps both) and record the gap. |
| fp32 partial unpack | the untilize unpacks fp32 partials through SrcA (tf32), then packs bf16 | tf32 keeps 10 mantissa bits, more than bf16's 7; the error is far below the bfp8 input error. Precision baseline confirms. |
| L1 budget with x and y in L1 | 368 KB per bank at R = 8416 plus CBs | `CB_BUDGET_BYTES` = 800 KiB in the solve; the implementer confirms headroom against the allocator at R = 8416 and lowers the constant if CB allocation clashes |
| Single-dispatch rule | composing `ttnn.matmul` per expert, or slicing / untilizing on host | one `ttnn.generic_op`; output allocated with `ttnn.allocate_tensor_on_device` |

## Tests required from the implementer (besides the acceptance test)

| Test | Content |
|------|---------|
| `test_moe_down_proj_perf.py` | device kernel duration for the 9 LOOSE_CASES rows (production config), reported in ns against stock and target |
| precision baseline | PCC and rel RMS vs fp64 on golden inputs for t = 4096 layers 0-5 and t = 1024 |
| regime-pinned layouts | both layouts forced through the private override, shapes listed under Work Distribution |
