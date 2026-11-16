# Operation Design: wq_b_linear

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (DRAM-bound matmul: multicast activation broadcast + streamed weight + tiled matmul, one device program) |
| Goal | `y = x @ W` for the single DSA-indexer wq_b shape, as close as possible to the 16,018 ns DRAM speed of light on Blackhole p100a. One `ttnn.generic_op` dispatch per call. |
| Math | `y[0,0,m,n] = Σ_k x[0,0,m,k] · W[k,n]`, m∈[0,160), n∈[0,4096), k∈[0,1536). No bias, no activation. |
| References | `eval/golden_tests/wq_b_linear/feature_spec.py` (TARGET, LOOSE_CASES, SOL build-up); `.claude/references/speed-of-light.md`; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`; `ttnn/cpp/ttnn/kernel_lib/mcast/` (host `mcast.hpp`, kernel `mcast_args.hpp`, `pipes/mcast_pipe.hpp`); `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp` (trid reads); catalog `ttnn/ttnn/operations/examples/master.md` entries `width_split`, `shared_input_reuse`, `mcast_topology`, `double_buffer`, `trid_double_issue`, `noc_placement`, `bank_stagger`, `matmul_output_subblock`, `compute_block_size`, `eltwise_l1_vs_dest_accumulate` |

Module exports (`ttnn/ttnn/operations/wq_b_linear/__init__.py`): `wq_b_linear`, `default_compute_kernel_config`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`. `eval/golden_tests/wq_b_linear/axes.py` imports all of them.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (x) | `ttnn.Tensor` | yes | see Tensors | — | RT (buffer address) |
| `weight` (W) | `ttnn.Tensor` | yes | see Tensors | — | RT (buffer address) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor`, keyword-only | no | `fp32_dest_acc_en=False`; any `math_fidelity` (tests use HiFi2) | `None` → `default_compute_kernel_config()` = `ComputeConfigDescriptor(math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)` | passed through as the compute kernel's `config` |

Registry declarations (op file, inline):

| Declaration | Value |
|-------------|-------|
| `INPUT_TAGGERS` | `{}` (one shape, no shape-derived axis) |
| `SUPPORTED` | `{"dtype": [ttnn.bfloat16], "weight_dtype": [ttnn.bfloat8_b], "layout": [ttnn.TILE_LAYOUT], "fp32_dest_acc_en": [False]}` — equals TARGET |
| `EXCLUSIONS` | `[]` |
| `INVALID` | not declared in the op (lives in `feature_spec.py`, `[]`) |

`validate()` is the first line of `wq_b_linear()`:

| Check | Axis / contract | Raises |
|-------|-----------------|--------|
| `x.dtype ∉ SUPPORTED["dtype"]`, `W.dtype ∉ SUPPORTED["weight_dtype"]`, `x.layout` or `W.layout ∉ SUPPORTED["layout"]`, resolved `fp32_dest_acc_en ∉ SUPPORTED["fp32_dest_acc_en"]` | SUPPORTED axes | `NotImplementedError` |
| `list(x.shape) != [1, 1, 160, 1536]` or `list(W.shape) != [1536, 4096]` | shape contract | `ValueError` |
| x or W not (`buffer_type == DRAM` and `memory_layout == INTERLEAVED`) | placement contract | `ValueError` |
| `num_cores < MIN_CORES` (16) on the device grid | L1 bound predicate (see ledger) | `RuntimeError` |

## Tensors

### Input

| Property | x (activation) | W (weight) |
|----------|----------------|------------|
| Shape | `[1, 1, 160, 1536]` | `[1536, 4096]` |
| Dtype | bfloat16 (tile 2048 B) | bfloat8_b (tile 1088 B) |
| Layout | TILE | TILE |
| Memory | DRAM interleaved | DRAM interleaved |
| Tile grid | `tensor_m_tiles = ceil(160/32) = 5` (batch dims 1·1 fold into m), `tensor_k_tiles = ceil(1536/32) = 48`; page `(m,k)` = `m·tensor_k_tiles + k` | `tensor_k_tiles = 48`, `tensor_n_tiles = ceil(4096/32) = 128`; page `(k,n)` = `k·tensor_n_tiles + n` |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 1, 160, 4096]` = `x.shape[:-1] + [W.shape[-1]]` |
| Dtype | bfloat16 (= x dtype) |
| Layout | TILE; page `(m,n)` = `m·tensor_n_tiles + n` |
| Memory | L1 interleaved (`ttnn.L1_MEMORY_CONFIG`), allocated with `ttnn.allocate_tensor_on_device` before the program is built |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| m (output rows; batch dims folded in) | independent — no result spans m; W is reuse-shared along m (W does not vary with m) | `block_m_tiles` | `tensor_m_tiles` = 5 (whole axis) | host, derived from x shape | not split: every active core owns all 5 m-tiles. Reason: 5 m-tiles cannot fill a grid, and cutting m makes the 6.68 MB W reuse-shared across m-cores (regime R5) | scheme-change (R5: W multicast along m) |
| n (output columns) | independent — each output column is self-contained; x is reuse-shared along n (x does not vary with n) | `core_n_tiles` (this core's share; per-core block extent = whole share) | `ceil(128 / num_cores)` for core_group_1, `floor` for core_group_2 → 2 / 1 on p100a (110 cores) | host: `ttnn.split_work_to_cores(grid, tensor_n_tiles, row_wise=True)` with `grid = device.compute_with_storage_grid_size()` | contiguous n ranges, row-major over the full device grid; `n_start` per core is the prefix sum of `core_n_tiles` | knob-turn (grid size / core count) |
| k (contraction) | dependent — every output tile sums over all of k | block extent `tensor_k_tiles` (whole k resident per core); streaming quantum `k_block_tiles` (= `matmul_block`'s `in0_block_k`) | block: 48; `k_block_tiles` = 8 → `num_k_blocks = tensor_k_tiles / k_block_tiles` = 6 | host constant `K_BLOCK_TILES`, clamped to the largest divisor of `tensor_k_tiles` ≤ it; every dependent (`num_k_blocks`, x round size, trid count, CB push quanta) is derived from it | not split across cores; within a core the k-sum is the helper's sequential K-blocked accumulation | scheme-change (R4: K-remainder split + deterministic combine) |

Re-run on the intermediate data each stage produces:

| Stage | Data it produces | Axes of that data | Assignment across cores |
|-------|------------------|-------------------|-------------------------|
| x broadcast | full replica of x (m × k) in every active core's `cb_x_resident` | m (5), k (48) | k cut into `num_k_blocks` rounds; round `r` is read from DRAM and multicast by injector `r mod num_x_injectors` (rotating sender, `num_x_injectors` = 6 cores). This is a communication role; injectors also own their normal n-share of the matmul |
| W stream | this core's W slice (k × core_n) in `cb_w_stream` | k, n | each core reads only its own n-range; no core reads for another |
| matmul accumulate | partial-then-final y block (m × core_n) in `cb_y_tiles` | m, n (k accumulated in place) | per core, its own n-range; no stage runs on one core while others wait |
| y store | y tiles in L1 interleaved | m, n | each core writes its own `5 × core_n_tiles` tiles |

Every knob is a parameter: `K_BLOCK_TILES`, the device-derived grid / `num_cores` / `core_n_tiles`, `num_x_injectors`, `w_reads_in_flight`, subblock factors and all CB capacities are computed on the host from those sources and passed as CT/RT args. No kernel contains a literal tile count.

### As built (Phase 0, supersedes the values below where they differ)

The implementer re-tuned four design values on p100a (measured in `agent_logs/ttnn-implementer_breadcrumbs.jsonl`); the tables and prose below still state the planned values.

| Knob | Planned | As built | Evidence |
|------|---------|----------|----------|
| `K_BLOCK_TILES` | 8 (`num_k_blocks` = 6) | 6 (`num_k_blocks` = 8) | kb=4: 32.6 µs, kb=16: 30.7 µs, kb=6 best |
| `w_reads_in_flight` | `min(num_k_blocks, 15)` = 6 | `W_READS_IN_FLIGHT` = 1 | 6 → 1 measured faster |
| `num_x_injectors` | 6 | `max(2, min(num_k_blocks, grid_x))` = 8 (row 9, x = 3..10) | follows `num_k_blocks` |
| x multicast protocol | `McastConfig(handshake=False, data_ready=Counter)` | `McastConfig(handshake=True, data_ready=Flag)` (`X_MCAST_LINKED`) | Counter selects unlinked multicast writes; bare broadcast 26.3 → 19.1 µs with Flag |

### As built (Refinement 1, supersedes Phase 0 where they differ)

| Knob | Phase 0 | Refinement 1 | Evidence (p100a, full op, median of 9) |
|------|---------|--------------|----------|
| `K_BLOCK_TILES` | 6 (`num_k_blocks` = 8) | 3 (`num_k_blocks` = 16) | kb=6: 25.1 µs, kb=4: 24.0–24.4, kb=3: 23.6–23.9, kb=2: 24.5 µs. Finer K-blocks shorten the compute tail after the last W block on 2-column cores; Phase 0's kb result was taken with the round count tied to kb |
| `X_ROUND_K_BLOCKS` (new) | rounds = K-blocks | 4 → `num_x_rounds` = 4 (raised so rounds ≤ `MAX_X_MCAST_CHANNELS` = 8) | rounds decouple the multicast quantum from the compute K-block; writer still pushes per K-block |
| `X_MCAST_CHANNELS` (new) | one rotating channel, Flag + handshake (26.5 µs) | `per_round`: one fixed-sender Flag channel per round, no handshake (each fires once per launch, so the flag cell is never reused) | the rotating handshake costs ≈1.5 µs/round (109 ready atomics converge on the next injector) |
| `X_ROUNDS_CONCURRENT` (new) | chained | chained (`False`) | all injectors at once: 39–88 µs, erratic (both Counter/unlinked and Flag/linked) |
| `X_INJECTOR_PLACEMENT` (new) | last row | `diagonal` over the last `num_x_rounds` rows, distinct columns | x-only 17.7 → 16.6 µs at the same round size: last-row injectors share row links for their prefetch |
| `num_x_injectors` | `max(2, min(num_k_blocks, grid_x))` | `num_x_rounds` (per_round) | one injector per round |
| `W_READS_IN_FLIGHT` | 1 | 1 | 2: 27.7 µs, 3: 29.3 µs (kb=6); 2 at kb=3: 24.6 µs |

### As built (Refinement 2, supersedes Refinement 1 where they differ)

| Knob | Refinement 1 | Refinement 2 | Evidence (p100a, device ns) |
|------|--------------|--------------|----------|
| n split | `split_work_to_cores` over 110 cores: 18 × 2 + 92 × 1 | `CORE_N_TILES = 2`: 65 cores (host `_core_n_counts` / `_split_n`) | full 23.94 → 22.99 µs; W-reader max/p50 1.09 → 1.03 |
| `W_BANK_COALESCE` (new) | contiguous columns | runs of `num_dram_banks` (7) two-column cores own `(n, n + 7)`: one 2176 B single-bank read per k row; the 2 leftover columns go to single-column cores | full 22.99 → 22.39 µs (median of 9). The leftover pair on one core read per tile and trailed by 0.8 µs |
| `CORE_PLACEMENT` (new) | row-major | row-major | `spread`: +0.8 µs (full) |
| `W_READS_IN_FLIGHT_G1` (new) | — | 0 (= `W_READS_IN_FLIGHT`) | deeper heavy-core reads: no gain (w_only 21.2 → 21.5–23.3 µs) |
| `W_NOC1_K_BLOCKS` / `W_NOC1_AFTER_X` (new) | — | 0 (parked) | W-only 20.4 → 17.9 µs at 6 blocks on NoC1, but in the full op the x broadcast holds BRISC/NoC1: 24–31 µs issued at start, 22.7–25.2 µs issued after the x phase |
| `CORE_N_TILES = 3` | — | not taken | 27.1 µs: compute (720 tile-matmuls) binds |

### As built (Refinement 3: R4 built, parked behind `K_REMAINDER_SPLIT = False`)

R4 `n_split_k_remainder_combine` is implemented in the shared kernels; at the default the Refinement 2 split runs.
With `K_REMAINDER_SPLIT = True`: `C = min(grid cores, tensor_n_tiles)` cores each own column `i`; the
`R = tensor_n_tiles − C` leftover columns are cut into `S = min(num_k_blocks, C // R)` contiguous K-slices
(K-block boundaries `s · num_k_blocks // S`; p100a: 6 slices of 2–3 K-blocks on 108 cores). The regime falls back
to the `CORE_N_TILES` split when `S < 2`.

| Piece | Realization |
|-------|-------------|
| slice W | reader reads the slice into `cb_w_slice` first (own trid), then the main column |
| slice x | `cb_x_slice` = second index on `cb_x_resident`'s allocation; writer pushes every K-block on both indices, compute pops the `kb_start` K-blocks before the slice |
| slice matmul | a second `matmul_block` call (same template as the main one) into `cb_partial`, nested in the main call's `PostKBlockFn` after main K-block `min(slice_end − 1, num_k_blocks − 1 − SLICE_EARLY_K_BLOCKS)`; the functor restores the main unpack/math/pack state afterwards |
| send | slice `s > 0`: writer, after its x phase, unicasts tile `t` to the owner's `cb_landing` slot `t·(S−1) + s − 1`, write barrier, then `noc_semaphore_inc` on the owner's `LANDING_SEMAPHORE_ID` |
| combine | owner (slice 0), in the same functor after main K-block `num_k_blocks − 1 − COMBINE_EARLY_K_BLOCKS`: `eltwise_chain` copy of its own partial into `cb_y_tiles[5..10)`, then per output tile one `eltwise_chain` of `S − 1` copies with `L1Accumulation::AddToExisting` (slice order, so bit-identical). `matmul_block` leaves packer L1 accumulation on, so the combine first issues `pack_reconfig_l1_acc(0)` |
| store | owner writes 10 tiles: its column, then the leftover column |

| Knob | Value | Evidence (p100a, device ns, median of 5 calls) |
|------|-------|----------|
| `K_REMAINDER_SPLIT` | False | A/B, 3 runs each: R2 split 22.40 / 22.60 / 22.32 µs, R4 22.70 / 22.78 / 22.34 µs |
| `SLICE_EARLY_K_BLOCKS` / `COMBINE_EARLY_K_BLOCKS` | 9 / 6 | slice at its K end and combine after the main loop: 24.0–24.2 µs (owners trail by 2 µs waiting for partials and combining); 7/5: 22.9; 9/6: 22.5; 11/8: 22.7; 13/7: 22.5 |
| `W_READS_IN_FLIGHT` (R4) | 1 | 2: 25.0 µs |
| `K_BLOCK_TILES` (R4) | 3 | 4: 23.8 µs |

Why R4 does not win here: per-core W drops 104,448 → 62,016 B and compute 480 → 285 tile-matmuls, so the gap from
W end to compute end shrinks from ≈ 1.35 µs to ≈ 0.6 µs. But the W stream on 110 cores of single-tile 1088 B reads
ends at NCRISC p50 20.2–20.6 µs, against 19.3 µs on Refinement 2's 65 cores of 2176 B bank-coalesced reads. The
binding term is still W transfer efficiency at the NoC0 links next to DRAM, not per-core imbalance.

### As built (Perf 1: y store split across both NoCs)

| Knob | Value | Evidence (p100a, device kernel ns, 9 calls per arm) |
|------|-------|----------|
| `Y_READER_SHARE_TENTHS` | 4 | 0 (writer stores all): median 22,681; 4: median 22,441. The reader (NCRISC, NoC0, idle once its W stream ends) stores the last `round(0.4 · 5·core_n)` y slots in column-major order (`kernels/wq_b_linear_y_store.hpp`), peeking `cb_y_tiles` without popping; the writer stores the rest, waits on `Y_DONE_SEMAPHORE_ID` (id 2; x channels now start at id 3) and stays the only consumer. Leftover-column tiles (R4) stay on the writer. 10/10 on NoC0: 23.7 µs |

`x_cb_k_blocks` / `w_cb_k_blocks` are not independent knobs in the code: the writer and reader reserve all `num_k_blocks` blocks up front (whole-K residency), so the CB capacities are written in `num_k_blocks` directly. A ring depth below `num_k_blocks` needs wrap-aware slot math in both kernels.

Verifier ablation (p100a, device kernel ns, 3 calls each): full 26.6 µs; x broadcast + y store only (no W reads, no compute) 25.4 µs; sync-only (one-page multicast per round, no W reads, no compute) 14.7 µs; W-only (one-page x) 20.8 µs; compute-only (one-page x) 15.7 µs. The serialized 8-round x broadcast is the critical path, not the W DRAM stream. See `verification_report.md`.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_resident` | `x_cb_k_blocks` | `num_k_blocks` (6 rounds = all of x) | Write-once landing slot per multicast round, so the broadcast needs **no receiver→sender ready handshake** (`McastConfig(handshake=False)`); injectors never wait on consumers. Bounded by the shape contract (x is 491,520 B). |
| `cb_w_stream` | `w_cb_k_blocks` | `num_k_blocks` (whole per-core W slice, ≤ 96 tiles on p100a) | The W reader never waits on compute: every W read for the core can be in flight at once, which is what the DRAM-bound stream needs (Little's law: ~448 GB/s × ~1.5 µs ≈ 670 KB in flight grid-wide ≈ 6 KB/core; a per-K-block barrier would hold ~4 KB/core on 1-column cores). |
| `cb_y_tiles` | `y_cb_blocks` | 1 block (`block_m_tiles × core_n_tiles_max`) | Single per-core block, accumulated in place over all K-blocks (packer L1 accumulation); one push at the end. |
| W read transaction window | `w_reads_in_flight` (K-blocks with a distinct read trid) | `min(num_k_blocks, 15)` = 6 | All K-blocks issued before the first wait; compute is released per K-block via `async_read_barrier_with_trid`. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `matmul_block` uniform K-block (`MatmulBlockShape.in0_block_k`, matmul_block_helpers.hpp:144) | `k_block_tiles` must divide `tensor_k_tiles` | host: `k_block_tiles = max d ≤ K_BLOCK_TILES with tensor_k_tiles % d == 0` | the last K-block indexes past x's k-columns / W's k-rows → wrong results |
| DEST capacity per subblock (`ASSERT(out_subblock_h * out_subblock_w <= DEST_AUTO_LIMIT)`, matmul_block_helpers.inl:157; `DEST_AUTO_LIMIT` = 8 for 16-bit half-sync, dest_helpers.hpp:93-103) | subblock factors (not the block) | host rule: `(sb_h, sb_w)` = max `sb_h·sb_w ≤ dest_limit(fp32_dest_acc_en, dst_full_sync_en)` with `sb_h | block_m_tiles`, `sb_w | core_n_tiles`; ties → larger `sb_h`. p100a: (5,1) for both groups | device ASSERT / DEST overflow |
| `caller_owns_pack_target` legality (matmul_block_constraints.hpp:28) | none on extents; fixes the template combo | `TileRowMajor + packer_l1_acc=true + LastBlockTarget::Interm` | static_assert / deadlock |
| NoC read transaction IDs (`NOC_MAX_TRANSACTION_ID 0xF`, tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:15; trid 0 = untagged, local_copy_helpers_dataflow.hpp:141) | K-blocks in flight with distinct trids | `w_reads_in_flight = min(num_k_blocks, 15)`; beyond that, trid `1 + (b mod w_reads_in_flight)` and block `b` is issued only after block `b − w_reads_in_flight` is retired | two in-flight blocks share a trid → a block is pushed to compute before its bytes land |
| Rotating-sender multicast (mcast.cpp:63-68, receivers must be a rectangle: mcast.cpp:84-87) | `num_x_injectors ≥ 2`; receiver set = one rectangle | `num_x_injectors = clamp(num_k_blocks, 2, grid_x)`; receivers = the full device grid rectangle | host `TT_FATAL` |
| Counter signal ordering (Counter is monotonic, `receive(round)` = `wait_min(round+1)`, mcast_pipe.inl:423-424; no handshake → no per-round sender gating) | rounds must complete in order | every core walks rounds `0..num_k_blocks-1` in order and the round-`r` injector sends only after its own counter reached `r` (it has received rounds `< r`) | a later round's increment satisfies an earlier round's wait → compute reads an x slot before it lands |
| Multicast landing address (mcast_pipe.hpp:16-17: landing address identical on all receivers) | `cb_x_resident` placement | one CB descriptor, same index and capacity, on the full receiver rectangle | data lands at a different offset per core → wrong results |
| L1 residency (ledger) | `core_n_tiles_max` | `num_cores ≥ MIN_CORES = 16` ⇒ `core_n_tiles_max ≤ 8` (≤ 3 on every shipping WH/BH grid) | CB allocation exceeds L1 |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `n_split_x_bcast_resident`** | **built** | the whole input space (validate's shape/placement contract; one cell) | per core `(m: block_m_tiles=5, n: core_n_tiles, k: tensor_k_tiles=48)`, K streamed in `k_block_tiles` | DRAM boundary minimum = each input crosses DRAM once, output never (L1): **reached** — W 6,684,672 B ×1, x 491,520 B ×1, y 0. Above the minimum on the NoC: x replicated into every other core (491,520 B inbound per core), y 1,310,720 B unicast L1 writes. | The block is already the whole per-core assignment (one block per core). Per-kernel fixed costs paid once: `compute_kernel_hw_startup`, one data-format reconfig, one `matmul_block_init`. A bigger `k_block_tiles` pays fewer per-K-step costs: one multicast round + counter inc_multicast, one trid wait + CB push on W, one x CB push, one L1-acc pack pass per subblock (`num_k_blocks × in1_num_subblocks` packs of `block_m_tiles × sb_w` tiles); it costs a longer exposed last-K-block compute tail (`5 × k_block_tiles × core_n_tiles` tile-matmuls after the last W byte lands). |
| R2 `x_per_core_dram` | rejected — superseded by R1 (a stepping stone whose successor R1 already builds) | — | same as R1 | x crosses DRAM `num_cores` times: 110 × 491,520 B = 54.1 MB, 7.5× the entire SOL byte budget | — |
| R3 `x_streamed_handshake` (cb_x double-buffered, per-round ready ack) | rejected — superseded by R1 on the entire input space: the shape contract bounds x to 491,520 B, which is resident | — | R1 with `x_cb_k_blocks = 2` | same DRAM; adds a receiver→sender ready round trip per round on every core and blocks the next round's sender on the slowest consumer | — |
| R4 `n_split_k_remainder_combine` | built in Refinement 3, parked behind `K_REMAINDER_SPLIT = False` (22.7 vs 22.4 µs; see As built (Refinement 3)). Original deferral — R1's busiest-core terms sit under the binding DRAM term on p100a (busiest core: 480 tile-matmuls = 11.4 µs ideal HiFi2; NoC inbound ≤ 595 KB ≈ 6.9 µs; DRAM 16.0 µs), and R4 only adds bytes. Build it when the compute-only ablation of a `core_n_tiles=2` core exceeds the DRAM-only time. Reachable: R1 keeps each core's W/x stream parameterized by `(n_start, core_n_tiles)` and K-blocks; R4 adds a second `(k_start, k_count)` slice per core, one partial CB + one semaphore and a fixed-order combine stage — the x broadcast and W stream are unchanged. | — | each core: one full column `(5, 1, 48)` + one remainder K-slice `(5, 1, k_block_tiles)`; the 18 remainder columns (128 − 110) are cut into `48/8 = 6` K-slices over 108 cores | DRAM unchanged (minimum). Adds cross-core partials: 18 columns × 5 partials × 5 tiles × 2048 B = 921,600 B, combined in fixed slice order on the column owner (bit-identical across calls). Buys: busiest core W 104,448 B → ~60,928 B, compute 480 → 280 tile-matmuls. | — |
| R5 `mn_2d_w_bcast_along_m` | deferred — cheapest cross-core-traffic split (see Traffic ranking), but its saving is in a tier that does not bind on either split, and it worsens the binding tier's transfer shape: W DRAM reads move from 110 consumer cores (~4–6.5 GB/s each) to 22 injectors (~20 GB/s each, ~30 KB in flight each), and every W byte takes an extra multicast hop before compute. Its compute-balance gain is also R4's, which keeps W direct. Reachable: R1's x broadcast becomes the per-m-row channel and R1's W reader becomes the column-segment injector; requires a grid dimension divisible by `tensor_m_tiles` (true on 11×10 and 13×10, false on 8×8 / 8×7). | — | per core `(m: 1, n: ceil(128/22)=6, k: 48)`; groups: m = `y mod 5`, n-group = `(x, y div 5)` | DRAM unchanged. Cross-core: W 6.68 MB × 4 receivers = 26.7 MB + x 98 KB × 21 × 5 = 10.3 MB ≈ 37 MB delivered vs R1's ≈ 55 MB; per-core inbound ≈ 411 KB vs R1's ≤ 595 KB | — |
| R6 `k_split_all_n` (each core a K-slice for all n) | rejected — superseded by R1: every core produces a full 5 × 128-tile partial (1.31 MB) → ~144 MB of partials to combine across 110 cores | — | `(5, 128, 48/110)` | DRAM minimum, cross-core 100× R1's | — |
| R7 `w_dram_resharded` | rejected — the contract fixes W as DRAM interleaved; re-laying it out is a second device dispatch | — | — | — | — |

Single built regime: no selection predicate, no regime-pinned tests.

### Traffic ranking

| Rank | Split | DRAM crossings (bytes) | Cross-core traffic | Busiest core | Verdict |
|------|-------|------------------------|--------------------|--------------|---------|
| 1 (cross-core bytes) | R5: m × n, W multicast along m, x along n | 7,176,192 B (minimum) | ≈ 37 MB delivered (W ×4, x ×21) | 288 tile-matmuls; inbound ≈ 411 KB; DRAM issue on 22 injectors | deferred (W behind a hop, DRAM issue concentrated) |
| 2 | **R1: n only, x multicast to all** | 7,176,192 B (minimum) | ≈ 53.6 MB x delivered + 1.31 MB y | 480 tile-matmuls; inbound ≤ 595 KB; DRAM issue on all 110 cores | **built** |
| 3 | R4: n + K-remainder, combine | 7,176,192 B | R1 + 921,600 B partials | 280 tile-matmuls | deferred |
| 4 | R6: k for all n, combine | 7,176,192 B | ≈ 144 MB partials | 2,791 tile-matmuls on combine owners | rejected |
| 5 | R2: n only, x re-read per core | 60.8 MB (x ×110) | 1.31 MB | — | rejected |

All candidate splits that do not re-read an input tie on the most expensive tier. R1 is chosen because it keeps the binding DRAM stream direct (each W byte goes DRAM → its consumer once, read by every core in parallel) and its cross-core extra (x replication) is on a separate NoC (NoC1) from the W stream (NoC0) and below the DRAM term per core. Occupancy (110/110 cores) is not the reason.

### Block schedule

Per core, `num_blocks_this_core = 1` (block = the whole assignment). Logical schedule:

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_x_block(block_idx);           // writer: x broadcast, num_k_blocks rounds into cb_x_resident
    load_w_block(block_idx);           // reader: W slice into cb_w_stream, retired per K-block
    matmul_accumulate_block(block_idx);// compute: one matmul_block call over num_k_blocks K-steps
    store_y_block(block_idx);          // writer: block_m_tiles × core_n_tiles tiles → L1-interleaved y
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_x_block` | `(m: 5, k: 48)` delivered as `num_k_blocks` rounds of `(5, k_block_tiles)` | `cb_x_resident` slots are write-once; nothing is evicted | Injectors: one DRAM prefetch of their round(s) issued at kernel start (before any wait), one read barrier. Per round: exactly one multicast of `5·k_block_tiles` tiles + one counter `inc_multicast` (no ready handshake); every core: one counter wait + one `cb_push_back(5·k_block_tiles)` per round. One `cb_reserve_back(5·48)` per kernel. |
| `load_w_block` | `(k: 48, n: core_n_tiles)` issued as `num_k_blocks` trid-tagged batches of `(k_block_tiles, core_n_tiles)` | `cb_w_stream` holds the whole slice | One `cb_reserve_back(48·core_n_tiles)` per kernel; all batches issued back-to-back before the first wait; per K-block one `async_read_barrier_with_trid` + one `cb_push_back(k_block_tiles·core_n_tiles)`. No full `noc_async_read_barrier` inside the stream. |
| `matmul_accumulate_block` | `(m: 5, n: core_n_tiles, k: 48)`, K streamed in `k_block_tiles` | output block accumulates in place in `cb_y_tiles` (packer L1 accumulation; K-block 0 overwrites, later K-blocks add) | `compute_kernel_hw_startup` once; `matmul_block_init` + reconfig once (single helper call); per K-block one wait/pop on each input; one L1-acc pack per subblock per K-block; one `cb_reserve_back` before and one `cb_push_back` after the call. |
| `store_y_block` | `(m: 5, n: core_n_tiles)` | — | One `cb_wait_front(5·core_n_tiles)`, `5·core_n_tiles` page writes, one write barrier per kernel. |

Stall-shadow check:

| Stage that waits | Waits on | Work scheduled into the stall |
|------------------|----------|-------------------------------|
| injector before sending round `r` | rounds `< r` completing (counter ≥ r) | its own DRAM prefetch of round `r` is issued at kernel start, before the wait |
| compute, per K-block | W K-block (x arrives ahead: x rounds serialize at ≈1 µs each, W K-blocks arrive at ≈ 16/6 µs each) | nothing independent remains — every matmul depends on a W K-block. Summation over k is associative, so a per-core K-start rotation is algebraically legal; it is not used because the n-split already staggers banks (page `k·128 + n`, consecutive cores start on consecutive pages) and because x rounds arrive in global order. |
| writer `store_y_block` | the whole accumulated block | nothing: all of the core's other work is done |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap: `k_block_tiles = 8` | Larger K-steps leave a longer exposed last-K-block compute tail (≈1.9 µs ideal on 2-column cores); smaller ones raise per-K-step pack/handshake counts and may make the L1-acc pack bind against math | `k_block_tiles ∈ {4, 6, 12, 16}` |
| Grid synchronization: 110 cores with 1–2 columns each | Every core receives all 491,520 B of x for 1–2 output columns; core_group_1 (18 cores × 2 columns) sets the busiest-core time while 92 cores idle at the end | 64 cores × 2 columns (even split, same busiest-core work, 46 fewer x receivers) |
| `num_x_injectors = 6` rotating (one per round) | Mcast rounds serialize anyway; several DRAM prefetchers compete with the W stream at start-up | 1 fixed injector; 2–3 rotating |
| x injector reads on NoC1 (BRISC default) | catalog `noc_placement`: reads on NoC1 are slower for spread placements | issue the x prefetch on NoC0 |
| Subblock (5,1) under 16-bit half-sync DEST | 2-column cores re-unpack the 5 x-tiles per in1 subblock; catalog `matmul_output_subblock` measures subblock size as the lever | `dst_full_sync_en=True` → (5,2) on core_group_1 |
| Placement of core_group_1 (2-column cores) in the first grid rows | The heavier cores are clustered; their W reads share NoC0 links | spread core_group_1 (e.g. diagonal / every 6th core) |
| `w_reads_in_flight = num_k_blocks` | Many outstanding reads per core may contend at DRAM banks with no gain | `w_reads_in_flight ∈ {2, 3}` (catalog `trid_double_issue`: depth 3–4 pays) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| W DRAM → `cb_w_stream` | bfp8_b tiles (1088 B pages) | reader kernel (NCRISC, NoC0), `TensorAccessor` page reads `(k, n_start..n_start+core_n_tiles-1)` for every k; batch per K-block tagged with `set_read_trid`, retired with `async_read_barrier_with_trid` | catalog `trid_double_issue`, `double_buffer` |
| x DRAM → injector's own `cb_x_resident` slot | bf16 tiles (2048 B) | writer kernel (BRISC, NoC1) on injector cores; `TensorAccessor` page reads `(m, r·k_block_tiles .. +k_block_tiles-1)` for m ∈ 0..4, written as `[m][k]` row-major inside the slot (the helper's in0 K-block layout) | slot `r` address = CB base + `r · 5 · k_block_tiles · 2048` |
| injector slot → every core's slot (Tensix→Tensix) | bf16 | `ttnn.Mcast(device, McastConfig(noc=NOC_1, handshake=False, data_ready=DataReadySignal.Counter), receivers=full grid rectangle, receiver_group_size=grid cores, McastRotatingSenderConfig(sender_cores=<injector cores>, sender_order=RowMajor))`, attached to the writer `KernelDescriptor`; device: `McastArgs<…>().sender(noc).send(src=dst=slot_r, size)` on the round's injector, `receiver(noc).receive(r)` elsewhere | sender is inside the rectangle; src == dst skips loopback (mcast_protocol.hpp:37 `MulticastIncludeSource`). Catalog `shared_input_reuse`, `mcast_topology` |
| `cb_x_resident`, `cb_w_stream` → `cb_y_tiles` | bf16 × bfp8_b → bf16 (16-bit DEST) | compute: one `matmul_block` call, packer L1 accumulation in place | — |
| `cb_y_tiles` → y (L1 interleaved) | bf16 tiles | writer kernel (BRISC, NoC1), `TensorAccessor` page writes `(m, n_start..)` | 10 tiles on the busiest core |

TARGET mapping: `feature_spec.py` TARGET has no placement axis — x/W placement (DRAM interleaved) and y placement (L1 interleaved) are contract, not support axes. The only regime boundaries are R4/R5 (same TARGET cell, perf-only).

R4 Tensix-to-Tensix contract (deferred, kept reachable): each remainder-slice core packs its `(5, 1)` partial into a dedicated partial CB and unicasts it to the column owner's partial-landing slot `s` (slot index = K-slice index), then increments the owner's counter semaphore; the owner waits for `slices − 1` arrivals and sums its own partial + slots in ascending `s` order (fixed order → bit-identical).

R5 contract (deferred): two `ttnn.Mcast` channels at disjoint semaphore ids (catalog `mcast_topology`): column-segment groups (5 cores) carry W from the segment's injector; two-row groups (m-row `y mod 5`) carry that m-row of x.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one per-core block `(5, core_n_tiles, 48)` — a contiguous n range |
| Grid | `grid = device.compute_with_storage_grid_size()`; `num_cores, all_cores, core_group_1, core_group_2, n_g1, n_g2 = ttnn.split_work_to_cores(grid, tensor_n_tiles, row_wise=True)`. p100a: 110 cores, group_1 = 18 cores × 2 columns, group_2 = 92 × 1. Peer coordinates come from `device.worker_core_from_logical_core` (inside `ttnn.Mcast`); nothing is hardcoded. |
| Per-core work | `core_n_tiles` columns starting at `n_start` (row-major prefix sum); all 5 m-tiles; all 48 k-tiles |
| Remainder | `tensor_n_tiles mod num_cores` cores get one extra column (core_group_1); tile counts use `ceil` (`tensor_m_tiles = ceil(160/32)`, `tensor_n_tiles = ceil(4096/32)`, `tensor_k_tiles = ceil(1536/32)`) |
| Grid larger than `tensor_n_tiles` (e.g. 13×10 = 130) | split_work_to_cores uses 128 cores; the multicast receiver rectangle and `cb_x_resident` stay on the full grid; cores with no columns run only the x receive loop (reader/compute kernels are placed on `all_cores` only) |
| x injectors | the last `num_x_injectors` cores of the grid in row-major order (one grid row on p100a: (5..10, 9)) — a row line (catalog `noc_placement`), and in core_group_2 when two groups exist |
| Kernel placement | reader: `all_cores`; compute: two `KernelDescriptor`s (core_group_1, core_group_2) with per-group CT args (`core_n_tiles`, subblock factors); writer: full grid (it carries the Mcast channel) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_resident` | 0 | 2048 | `block_m_tiles · k_block_tiles · x_cb_k_blocks` = 5·8·6 = 240 | spans m and k (write-once multicast slots, one per round); does not scale with n (replica) | Float16_b | writer | compute | whole kernel |
| `cb_w_stream` | 1 | 1088 | `k_block_tiles · w_cb_k_blocks · core_n_tiles_max` = 8·6·2 = 96 | spans k and n; m absent | Bfp8_b | reader | compute | whole kernel |
| `cb_y_tiles` | 16 | 2048 | `block_m_tiles · core_n_tiles_max` = 10 | spans m and n; k accumulated in place | Float16_b | compute | writer | from first pack to store |

`cb_x_resident` exists on the full grid rectangle (multicast landing); `cb_w_stream` and `cb_y_tiles` on `all_cores`. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_x_block` | `(5, 48)` as 6 rounds of `(5, 8)` | `McastArgs` sender/receiver pipes | DRAM x (injectors only) | `cb_x_resident`: reserve 240 once, push 40 per round | 240 pages pushed, consumed by compute |
| 2 | `load_w_block` | `(48, core_n)` as 6 batches of `(8, core_n)` | trid read helpers + `TensorAccessor` | DRAM W | `cb_w_stream`: reserve `48·core_n` once, push `8·core_n` per K-block | `48·core_n` pushed |
| 3 | `matmul_accumulate_block` | `(5, core_n, 48)`, `in0_block_k = 8`, `num_k_blocks = 6` | `matmul_block` | `cb_x_resident` wait/pop 40 per K-block; `cb_w_stream` wait/pop `8·core_n` per K-block | `cb_y_tiles`: caller reserves `5·core_n` before, pushes `5·core_n` after | both inputs fully popped; `cb_y_tiles` holds the final block |
| 4 | `store_y_block` | `(5, core_n)` | `TensorAccessor` writes | `cb_y_tiles` wait/pop `5·core_n` | L1-interleaved y | — |

CB sync: `cb_x_resident` push 6×40 = 240 = compute pops 6×40; `cb_w_stream` push 6×`8·core_n` = pops 6×`8·core_n`; `cb_y_tiles` push `5·core_n` = writer wait/pop `5·core_n`. All pushes in one invocation stay ≤ capacity; CB pointers start fresh each dispatch.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (compute boot) | helper-prereq | `compute_kernel_hw_startup<SrcOrder::Reverse>(icb0, icb1, ocb)` | tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60 | `(cb_x_resident, cb_w_stream, cb_y_tiles)`; Reverse is mandatory for matmul (matmul_block_helpers.hpp:100-103) | — | — | — |
| `matmul_accumulate_block` | helper | `compute_kernel_lib::matmul_block` | ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:339-371 | `<transpose=false, packer_l1_acc=true, LastBlockTarget::Interm, OutputCBLayout::TileRowMajor, InitMode::Short, WaitAndPopPerKBlock, WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock, 0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, caller_owns_pack_target=true>`; args `(in0=cb_x_resident, in1=cb_w_stream, out=cb_y_tiles, interm=cb_y_tiles, MatmulBlockShape::of(block_m_tiles/sb_h, core_n_tiles/sb_w, sb_h, sb_w, k_block_tiles, num_k_blocks))`; legality matmul_block_constraints.hpp:28; L1-acc overwrite on K-block 0 / accumulate after: matmul_block_helpers.inl:414, 467 (`llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)`) | `cb_x_resident`, `cb_w_stream` | `cb_y_tiles` | `in0_block_k` = `k_block_tiles`, `num_k_blocks`, block width `core_n_tiles`, block height `block_m_tiles`; subblock (`sb_h`, `sb_w`) derived by the host rule, not a design knob |
| `matmul_accumulate_block` | helper | `MatmulBlockShape::of` | matmul_block_helpers.hpp:160 | see above | — | — | as above |
| `load_x_block` (host) | helper | `ttnn.Mcast`, `ttnn.McastConfig`, `ttnn.McastRotatingSenderConfig`, `Mcast.attach(descriptor, "x_mcast", [writer_kernel], first_semaphore_id)` | ttnn/cpp/ttnn-nanobind/mcast_host.cpp:108-200; host API ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp:98-121, 133-157 | `McastConfig(noc=NOC_1, handshake=False, data_ready=DataReadySignal.Counter)`; rotating sender grid = injector cores | — | — | `num_x_injectors`, round size = `5·k_block_tiles` tiles |
| `load_x_block` (device) | helper | `dataflow_kernel_lib::McastArgs<CT, RT>`; `.sender(noc).send(src_l1, dst_l1, size)`; `.receiver(noc).receive(round)`; `.should_send(round)` | ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:309-329, 180-181, 217-219; pipes/mcast_pipe.hpp:91-92, 163 | `size = 5·k_block_tiles·2048`, `src_l1 = dst_l1 = slot_r` | DRAM x | `cb_x_resident` | `k_block_tiles` |
| `load_x_block` (injector prefetch), `load_w_block`, `store_y_block` | raw_api | `TensorAccessor` page reads/writes (`noc.async_read` / `noc.async_write` on page ids) | tt_metal/hw/inc/api/tensor/tensor_accessor.h:71; tt_metal/hw/inc/api/dataflow/noc.h | page ids as in Dataflow Strategy | DRAM x / DRAM W / `cb_y_tiles` | `cb_x_resident` / `cb_w_stream` / y | page counts derive from `k_block_tiles`, `core_n_tiles` |
| `load_w_block` | helper | `dataflow_kernel_lib::set_read_trid(noc, trid)`, `dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid)` | ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147, 158 | trid `1 + (b mod w_reads_in_flight)` | DRAM W | `cb_w_stream` | `w_reads_in_flight` |

Raw-API entries — helpers considered and rejected:

| Raw entry | Helper considered | Mismatch (file:line) | Reason |
|-----------|-------------------|----------------------|--------|
| `TensorAccessor` page reads for W / x prefetch | `tilize_helpers_dataflow.hpp`, `local_copy_helpers_dataflow.hpp` (`read_with_state`) | local_copy_helpers_dataflow.hpp:11-23 (self-aimed L1→L1 reads only); `read_with_state` programs one fixed source (local_copy_helpers_dataflow.hpp:55-60) | the sources are interleaved DRAM pages in 7+ banks, one page per tile; no kernel-lib helper wraps interleaved-DRAM tile reads — `TensorAccessor` is the addressing layer itself |
| `TensorAccessor` page writes for y | none in `ttnn/cpp/ttnn/kernel_lib/` targets interleaved tile writes | — | same: `TensorAccessor` is the mechanism |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Counter rounds out of order | No handshake + rotating senders: if round `r`'s injector sent before round `r−1` completed, receivers' `wait_min(r)` would be satisfied by the wrong round | every core walks rounds in order; the injector of round `r` calls `send` only after its own loop has received/sent rounds `< r` (its counter ≥ r). Its DRAM prefetch is issued earlier, so the ordering costs no DRAM time |
| No-handshake multicast relies on dispatch ordering | Data may land in a receiver before its kernel starts | safe because all program config (semaphore init = 0, CB config) is written before any core's go signal and the previous program has finished on all cores; the receiver touches the slot only after its counter wait. Fallback if a debugger ever shows a stale counter: `handshake=True` (adds a per-round ack round trip; x is resident so it overlaps) |
| `cb_x_resident` slot math | Injectors compute `slot_r = base + r·5·k_block_tiles·2048` from the CB base; receivers push in round order | one `cb_reserve_back(cb_x_resident, 240)` at kernel start gives the base on every core; identical descriptor on the full rectangle ⇒ identical base |
| in0 tile order inside a K-block | `matmul_block` indexes in0 as `[m][k_in_block]` (`in0_index` advances by 1 per k, by `sb_h·in0_block_k` per subblock, matmul_block_helpers.inl:194-195, 376, 498) while x's DRAM pages are `m·48 + k` | injectors write slot `r` as 5 rows × `k_block_tiles` tiles, row-major |
| in1 tile order | helper advances `in1_index += in1_per_core_w` per k (matmul_block_helpers.inl:377) → `[k][n]` | reader writes W batch `b` as `k_block_tiles` rows × `core_n_tiles` tiles |
| `interm_buf` = `out_buf` = `cb_y_tiles` | With `Interm` + `caller_owns_pack_target` the helper packs every K-block (incl. last) to `interm` in place; `out_buf` is not reserved (TileRowMajor path) | pass `cb_y_tiles` for both; compute does `cb_reserve_back(cb_y_tiles, 5·core_n)` before the call and `cb_push_back(cb_y_tiles, 5·core_n)` after |
| K-accumulation precision | `fp32_dest_acc_en=False`: DEST and the L1-acc buffer are 16-bit; 48 k-tiles accumulate in bf16 | PCC gate 0.9998 against fp64; a hand-written op with the same config reached 0.99990; keep the K-order fixed (deterministic) |
| `cb_w_stream` sized to core_group_1 | one descriptor for all cores; 1-column cores use half | 52,224 B unused on core_group_2 cores; L1 is not binding (ledger) |
| Grid portability | grids from 8×7 (WH) to 13×10 (BH p150) | everything derives from `compute_with_storage_grid_size()`; `num_cores ≥ 16` enforced; > 128-core grids handled by zero-column receive-only cores |
| Determinism (test_regression.py) | — | n-only split: every output tile is produced by one core in a fixed K order; no cross-core reduction in R1 |

Structural impossibilities: none beyond `INVALID = []`.
