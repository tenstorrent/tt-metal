# Operation Design: scaled_dot_product_attention

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + softmax + matmul) |
| Goal | Compute `O = softmax((Q @ K^T) * scale + mask) @ V` as a single fused on-device kernel using Flash-Attention-1-style online softmax so memory stays O(D) per query tile-row independent of S_kv. |
| Math | `O[b, h, i, :] = sum_j softmax_j((Q[b,h,i,:] · K[b,h_kv,j,:]) * scale + mask[b,*,i,j]) * V[b,h_kv,j,:]` |
| Mode | Derivative (fused multi-stage kernel built on `kernel_lib` helpers) |
| References | `tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_compute_utils.hpp` (algorithmic reference for online softmax), `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp` (matmul primitive), `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (row reductions), `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp` (eltwise primitives) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `Q` | `ttnn.Tensor` (B, H, S_q, D), bfloat16, TILE | yes | rank=4; B,H ≥ 1; S_q,D multiples of 32 | — | runtime tensor |
| `K` | `ttnn.Tensor` (B, H, S_kv, D), bfloat16, TILE | yes | rank=4; matches Q in B,H,D; S_kv multiple of 32 | — | runtime tensor |
| `V` | `ttnn.Tensor` (B, H, S_kv, D), bfloat16, TILE | yes | rank=4; matches K exactly | — | runtime tensor |
| `attention_mask` | `Optional[ttnn.Tensor]` | no | bfloat16, TILE, shape (B, 1, S_q, S_kv) or (B, H, S_q, S_kv) | `None` | runtime tensor (optional) |
| `scale` | `Optional[float]` | no | finite positive | `1/sqrt(D)` | host-computed → CT scalar (`scale_bits` = bf16-packed bit pattern) |

Phase 0 SUPPORTED (declared in op file):

| Axis | Phase 0 values |
|------|----------------|
| `dtype` | `ttnn.bfloat16` |
| `layout` | `ttnn.TILE_LAYOUT` |
| `alignment` | `tile_aligned` |
| `attention_kind` | `self`, `cross` |
| `mask_mode` | `none`, `causal` |
| `scale_mode` | `auto`, `explicit` |

EXCLUSION: `{"mask_mode": "causal", "attention_kind": "cross"}` — a causal mask on a rectangular S_q × S_kv block is well-defined math but not a real workload, and this kernel's causal path makes no special-case assumption (mask comes in as a generic additive tensor). The exclusion is declared to keep refinements honest.

INVALID: none (TILE-only op; canonical bf8b+ROW_MAJOR rule is vacuous). Declared in `eval/golden_tests/scaled_dot_product_attention/feature_spec.py`, not in the op file.

## Tensors

### Input

| Property | Q | K | V | attention_mask |
|----------|---|---|---|----------------|
| Shape | (B, H, S_q, D) | (B, H, S_kv, D) | (B, H, S_kv, D) | (B, 1, S_q, S_kv) or (B, H, S_q, S_kv), optional |
| Dtype | bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| Layout | TILE | TILE | TILE | TILE |
| Memory | interleaved (DRAM or L1) | interleaved | interleaved | interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, H, S_q, D) (same as Q) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | interleaved (default DRAM; respects `memory_config` kwarg) |

## Dataflow Strategy

**Algorithmic shape**: one query tile-row at a time. Each query tile-row spans 32 consecutive sequence positions of Q (one tile-height) and the full `D` head dimension (`Dt = D/32` tiles wide). The output for that query tile-row is also (32, D), produced as `Dt` tiles. There are `Qt = S_q/32` such query tile-rows per (B, H) head-batch combination, so the total work pool is `B * H * Qt` independent query tile-rows distributed across cores.

**Per-query-tile-row data path** (Flash-Attention-1 / online softmax):

```
reader (NCRISC, DRAM → L1)         compute (TRISC, L1 → L1)            writer (BRISC, L1 → DRAM)
──────────────────────────         ────────────────────────            ──────────────────────────
push Q row    → cb_q_row           wait Q row
push scalers  → cb_scalers         (one-shot init)
for k in [0, Kt):                  for k in [0, Kt):
  push K[k]   → cb_k_row             A. matmul Q @ K[k]^T → cb_scores
  push V[k]   → cb_v_row             B. eltwise scale+mask in DEST
  push M[k]   → cb_mask (opt)        C. reduce<MAX,REDUCE_ROW> → cb_row_max
                                     D. running cur_max via max(prev,row)
                                     E. sub<COL> cb_scores − cb_cur_max,
                                        eltwise_chain → exp → cb_attn
                                     F. reduce<SUM,REDUCE_ROW> → cb_row_sum
                                     G. correction = exp(prev_max−cur_max)
                                     H. update cb_sum_acc, cb_out_acc
                                     I. matmul cb_attn @ V[k] → cb_partial
                                     J. add corrected prev_out + partial → cb_out_acc

                                   final divide cb_out_acc / cb_sum_acc → cb_output
                                                                       wait cb_output (Dt tiles)
                                                                       NoC write → DRAM
```

All inter-RISC handoffs are through CBs in L1 of the same Tensix core. No inter-Tensix communication (no multicast, no semaphores, no fabric) — each core processes its own query tile-rows independently. Output tensor lives in interleaved DRAM (default) or L1 if requested.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One query tile-row = (32 query positions, D) = `Dt` output tiles. |
| Total work | `total_rows = B * H * (S_q / 32)`. |
| Grid | Compact rectangle from `ttnn.num_cores_to_corerangeset(min(total_rows, compute_with_storage_grid_size().x * y), grid, row_wise=True)`. |
| Per-core split | `ttnn.split_work_to_cores(grid, total_rows) → (num_cores, all_cores, core_group_1, core_group_2, rows_per_core_1, rows_per_core_2)` — every core's runtime args supply `start_row_id` and `num_rows`. |
| Per-core formula | core *c* with `(num_rows_c, start_row_id_c)` iterates `for row_id in [start_row_id_c, start_row_id_c + num_rows_c)`. Each `row_id` decomposes as `b = row_id / (H * Qt); h = (row_id / Qt) % H; qt = row_id % Qt`. |
| Remainder | Handled by `split_work_to_cores` standard pattern: `core_group_1` cores get `rows_per_core_1` rows; `core_group_2` cores get `rows_per_core_2` (= `rows_per_core_1` − 1) rows. No per-core conditional logic needed — runtime args drive the loop bound. |
| Inter-core | None. No multicast, no semaphores. |

## Circular Buffers

CB-index convention: 0–7 inputs, 8–15 special (scalers, side data), 16–23 outputs, 24–31 intermediates. All CBs are sized per-core (each Tensix has its own copy).

`Dt = D / 32`, `Kt = S_kv / 32`. Tile = 32 × 32 × 2 bytes = 2048 B for bfloat16.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_q_row` | 0 | `tile_size(bf16)` = 2048 B | `Dt` | bfloat16 | reader | compute | One Q tile-row persists for the entire K-loop (consumed once per query tile-row, at the end). Sized = block, not double-buffered, because Q is held across `Kt` matmul invocations via `InputPolicy::WaitAndRetainOnLastBlock`. |
| `cb_k_row` | 1 | 2048 B | `2 * Dt` | bfloat16 | reader | compute | Streaming, one K tile-row pushed per K-iteration. Double-buffered so reader can overlap with compute. |
| `cb_v_row` | 2 | 2048 B | `2 * Dt` | bfloat16 | reader | compute | Streaming, one V tile-row pushed per K-iteration. Double-buffered. |
| `cb_mask` | 3 | 2048 B | `2` | bfloat16 | reader (optional) | compute | Streaming, one mask tile per K-iteration when `HAS_MASK`. Page count = 2 for double-buffer; when `HAS_MASK==0`, kernel skips wait/pop and the host omits the CB descriptor (the CB ID is still defined for symbol parity). |
| `cb_max_scaler` | 8 | 2048 B | `1` | bfloat16 | reader (one-shot) | compute (persistent) | Reader emits one bf16 scaler tile (value 1.0) for the MAX row-reduce at startup. Compute does `cb_wait_front` once; helper never pops. Lifetime = whole kernel. |
| `cb_sum_scaler` | 9 | 2048 B | `1` | bfloat16 | reader (one-shot) | compute (persistent) | Reader emits one bf16 scaler tile (value 1.0) for the SUM row-reduce at startup. The helper uses `compute_uses_reduce_tile=true` so the layout is row-0 fill (reduce-LLK), NOT col-0 fill (matmul). Lifetime = whole kernel. |
| `cb_scale` | 10 | 2048 B | `1` | bfloat16 | reader (one-shot) | compute (persistent) | Reader emits one scalar tile filled with `scale_value` (1/sqrt(D) or user-provided). Used for SCALAR-broadcast multiply on scores. Lifetime = whole kernel. |
| `cb_scores` | 24 | 2048 B | `2` | bfloat16 | compute (matmul Q@K^T phase) | compute (next eltwise phase) | One score tile per K-iteration, lifetime = within one iteration. Page count 2 lets the helper double-buffer the matmul output. |
| `cb_attn` | 25 | 2048 B | `2` | bfloat16 | compute (after exp) | compute (second matmul + sum-reduce) | Attention-weights tile (post-exp). Page count 2 for double-buffer. Consumed by both `reduce<SUM,REDUCE_ROW>` (WaitUpfrontNoPop) and `matmul_block` (S@V) within the same K-iteration before being popped — the helper sequence is "wait, reduce (no pop), matmul (waits + pops)". |
| `cb_row_max` | 26 | 2048 B | `2` | bfloat16 | compute (MAX reduce) | compute (running-max update) | The fresh row-max from the current K-iteration's reduce. Single-iteration lifetime. |
| `cb_row_sum` | 27 | 2048 B | `2` | bfloat16 | compute (SUM reduce on cb_attn) | compute (sum-acc update) | The fresh row-sum-of-exp from the current K-iteration. Single-iteration lifetime. |
| `cb_max_acc` | 28 | 2048 B | `1` | bfloat16 | compute (initialized on k=0, updated in-place every k) | compute (next iteration; final divide phase reads but never consumes) | Running cur_max column-vector tile. **In-place updated**: compute pops + reserves the single slot each iteration (compute exclusively owns this CB — no reader/writer touches it, so pop+pack-back is a safe slot-reuse pattern; see `binary_op_in_place` doc on the CB-ownership rule at `binary_op_helpers.hpp:341-351`). Persists across the entire K-loop. |
| `cb_max_prev` | 29 | 2048 B | `1` | bfloat16 | compute (copy of `cb_max_acc` before update) | compute (correction-factor computation) | Snapshot of previous-iteration max, needed to compute `correction = exp(prev_max - cur_max)`. Lifetime = within one K-iteration (k ≥ 1). |
| `cb_correction` | 30 | 2048 B | `2` | bfloat16 | compute (eltwise sub+exp) | compute (sum-acc and out-acc updates) | Column-vector tile `exp(prev_max - cur_max)`. Lifetime = within one K-iteration (k ≥ 1). Page count 2 because two consumers (sum-acc update reads it once, out-acc update reads it once) — held with `HeldStream` lifecycle across both consumers. |
| `cb_sum_acc` | 31 | 2048 B | `1` | bfloat16 | compute (initialized on k=0, updated in-place every k) | compute (final-divide phase) | Running cur_sum column-vector tile. In-place updated, persists across K-loop. |
| `cb_partial_out` | 22 | 2048 B | `2 * Dt` | bfloat16 | compute (S@V matmul phase) | compute (out-acc update) | Per-K-iteration partial output (attention_weights @ V[k]) = Dt tiles. Page count 2*Dt for double-buffer between matmul writer and out-acc-update reader. |
| `cb_out_acc` | 21 | 2048 B | `Dt` | bfloat16 | compute (initialized on k=0, updated in-place every k) | compute (final-divide phase) | Running output accumulator = Dt tiles. In-place updated, persists across K-loop. Caller-owned CB (no reader/writer touches it). |
| `cb_output` | 16 | 2048 B | `2 * Dt` | bfloat16 | compute (final divide phase) | writer | Final output for one query tile-row = Dt tiles. Page count 2*Dt for double-buffer between compute and writer. |

CB-sync invariant (verified per CB):

| CB | Producer pushes per iteration | Consumer waits per iteration | Match |
|----|-------------------------------|------------------------------|-------|
| `cb_q_row` | reader: Dt at row start | compute: Dt for matmul Q@K^T (helper retains via `WaitAndRetainOnLastBlock`; manual pop after K-loop) | ✓ |
| `cb_k_row` | reader: Dt per K iter | compute: Dt per K iter (matmul wait+pop) | ✓ |
| `cb_v_row` | reader: Dt per K iter | compute: Dt per K iter (matmul wait+pop) | ✓ |
| `cb_mask` | reader: 1 per K iter (when HAS_MASK) | compute: 1 per K iter (eltwise add wait+pop) | ✓ |
| `cb_max_scaler` | reader: 1 once | compute: 1 once (held — never popped; final manual pop at kernel end) | ✓ |
| `cb_sum_scaler` | reader: 1 once | compute: 1 once (held) | ✓ |
| `cb_scale` | reader: 1 once | compute: 1 once (held) | ✓ |
| `cb_scores` | compute: 1 per K iter | compute: 1 per K iter (sub+exp chain wait+pop) | ✓ |
| `cb_attn` | compute: 1 per K iter | compute: 1 per K iter (waited twice within the iter, popped once after S@V) | ✓ |
| `cb_row_max` | compute: 1 per K iter | compute: 1 per K iter | ✓ |
| `cb_row_sum` | compute: 1 per K iter | compute: 1 per K iter | ✓ |
| `cb_max_acc` | compute: 1 on k=0; 1 per k≥1 (after pop) | compute: 1 every k iter (read for update) + 1 in final divide (held — popped at end) | ✓ |
| `cb_max_prev` | compute: 1 per k≥1 | compute: 1 per k≥1 | ✓ |
| `cb_correction` | compute: 1 per k≥1 | compute: 1 per k≥1 (consumed by sum-acc and out-acc updates within the iter — held until out-acc update, then popped) | ✓ |
| `cb_sum_acc` | compute: 1 on k=0; 1 per k≥1 (after pop) | compute: 1 every k iter + 1 in final divide (popped at end) | ✓ |
| `cb_partial_out` | compute: Dt per K iter | compute: Dt per K iter (out-acc update consumes) | ✓ |
| `cb_out_acc` | compute: Dt on k=0; Dt per k≥1 (after pop) | compute: Dt every k iter + Dt in final divide (popped at end) | ✓ |
| `cb_output` | compute: Dt per query row | writer: Dt per query row | ✓ |

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference verified against the helper headers.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|-----------------------|--------------------------|---------------------------|--------------|
| Reader: emit MAX-reduce scaler tile | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_max_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:94-101` | `cb_id=cb_max_scaler, PoolType=MAX, ReduceDim=REDUCE_ROW, reduce_factor=SUM_AND_MAX_REDUCE_FACTOR`. Pool-type-aware (MAX → row-0 fill). | — | `cb_max_scaler` | Reader does this once at startup before main loop. Helper picks row-0 fill for MAX (correct layout for reduce LLK). |
| Reader: emit SUM-reduce scaler tile | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_sum_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, 1, /*compute_uses_reduce_tile=*/true>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:94-101` | `compute_uses_reduce_tile=true` forces row-0 fill even for SUM+REDUCE_ROW (default is col-0 / matmul layout). Compute uses `reduce<>` helper which expects row-0 fill for non-MAX REDUCE_ROW only when `compute_uses_reduce_tile=true`; but the SDPA compute kernel uses `compute_kernel_lib::reduce<>` which internally picks the matmul path for SUM+REDUCE_ROW — so we must set `compute_uses_reduce_tile=false` to keep col-0 fill. **Decision**: use the helper's default (`compute_uses_reduce_tile=false`) since `compute_kernel_lib::reduce<SUM, REDUCE_ROW>` follows the matmul path and expects col-0 fill (see `reduce_helpers_dataflow.hpp:54-60`). | — | `cb_sum_scaler` | One-shot. |
| Reader: emit scale tile | raw_api | manual fill of bf16 tile with `scale_value` (1/sqrt(D) or user) | tile fill via `noc_async_read_one_packet_set_state` is wrong here — use `dataflow::fill_cb_tile` or manual write through `get_write_ptr`. | one-shot scalar tile, value `scale_value` packed as bf16. Tile is intended as SCALAR-broadcast operand for the score-scaling multiply. | — | `cb_scale` | Helpers considered and rejected: <ul><li>`prepare_reduce_scaler` (`reduce_helpers_dataflow.hpp:64-67`) — explicitly documented as "ONLY be used for [reduce LLK] purpose — not for arbitrary constant tiles" at `reduce_helpers_dataflow.hpp:21-23`. Using it for a non-reduce scaler tile violates the documented contract and would route through the row-0/col-0 fill patterns which are wrong for SCALAR-broadcast.</li><li>`eltwise_fill` (`eltwise_fill.hpp`) — compute-side helper that fills DEST then packs; does not fit dataflow-kernel use. Would force a per-iteration compute pack and an extra CB push/pop cycle.</li></ul> Concrete reason: no dataflow-side "fill arbitrary bf16 constant tile" helper exists in the current kernel_lib. Reader writes a 32×32 tile of `scale_value` via `cb_reserve_back(cb_scale, 1); ptr = get_write_ptr(cb_scale); fill 1024 bf16 values; cb_push_back(cb_scale, 1)`. |
| Reader: stream tiles (Q, K, V, mask) | raw_api | `TensorAccessor` + `noc_async_read_tile` + `noc_async_read_barrier` per `cb_reserve_back / cb_push_back` window | `tech_reports/tensor_accessor/tensor_accessor.md` | TensorAccessor constructed per tensor from `TensorAccessorArgs<>()` CT args | DRAM tensor | `cb_q_row`, `cb_k_row`, `cb_v_row`, `cb_mask` | Helpers considered and rejected: no kernel_lib helper wraps the DRAM-tile streaming loop (the dataflow helpers cover scaler emission and tile-row push patterns for reduce, not generic tensor streaming). Standard pattern from `tech_reports/tensor_accessor/tensor_accessor.md`. |
| Compute: HW init | raw_api | `compute_kernel_hw_startup(cb_q_row, cb_k_row, cb_scores)` then `mm_init(cb_q_row, cb_k_row, cb_scores)` | `api/compute/matmul.h` (mm_init), `api/compute/compute_kernel_hw_startup.h` | First statement of `MAIN()` per D5 contract documented in `eltwise_chain.hpp:51-58`. | — | — | Required ONCE at kernel boot. Cannot be re-issued mid-kernel — MMIO-unsafe per `compute_kernel_hw_startup.h:26-30`. The first matmul is `matmul_block` which expects `mm_init` (matmul-first init); subsequent reconfigs are owned by the helpers themselves. |
| Compute: Q @ K[k]^T matmul | helper | `compute_kernel_lib::matmul_block<transpose=true, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock /*in0*/, InputPolicy::WaitAndPopPerKBlock /*in1*/, MaskScalePostCompute>(cb_q_row, cb_k_row, cb_scores, cb_q_row /*interm placeholder*/, MatmulBlockShape::of(1, 1, 1, 1, Dt, 1), MaskScalePostCompute{...})` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:790-823` | M=1, N=1, K=Dt, single subblock, num_k_blocks=1, transpose B (K rows are treated as K^T columns), in0 retained (Q reused across all Kt iterations), in1 popped each K iter. `MaskScalePostCompute` is a caller-defined `PostComputeFn` (functor signature `void(uint32_t out_subblock_num_tiles)`) that in DEST applies scale × score + mask: multiply by SCALAR-broadcast `cb_scale`, then (if HAS_MASK) add `cb_mask`. Helper docstring `matmul_block_helpers.hpp:486-489` explicitly documents this as the SDPA-mask pattern. interm_buf = `cb_q_row` per the helper's "num_k_blocks==1 → pass any same-type CB as placeholder" idiom (`matmul_block_helpers.hpp:606-612`). | `cb_q_row` (Dt held), `cb_k_row` (Dt per iter), `cb_scale` (1 held), `cb_mask` (1 per iter, when HAS_MASK) | `cb_scores` | Helper manages reserve/push on `cb_scores`, wait/pop on `cb_k_row`, and DEST. `WaitAndRetainOnLastBlock` on in0 with num_k_blocks=1 → helper waits Dt tiles on `cb_q_row` once at first call but never pops, exactly what we need to retain Q across the K-loop. Manual `cb_pop_front(cb_q_row, Dt)` at end of K-loop. |
| Compute: Q @ K[k]^T — score scaling + mask add (post-compute hook) | helper-composed | `MaskScalePostCompute` functor body: SCALAR-broadcast multiply in DEST, then SCALAR-broadcast add (mask path). Lives inside the matmul helper's `PostComputeFn` slot — runs after matmul completes, before pack. | `matmul_block_helpers.hpp:245-249` (PostComputeFn contract) + `eltwise_chain.hpp` for the DEST-resident ops (uses `mul_unary_tile` + `add_binary_tile` raw LLK calls inside the functor — DEST-internal, not chain-routable because the operands are already in DEST/CB with custom timing). | — | `cb_scores` (via helper) | Concrete reason raw LLK inside the post-compute functor (rather than a chain helper for this sub-phase): the `PostComputeFn` is invoked with DEST already holding matmul partials and the tile_regs_acquire window already open. `eltwise_chain` owns its own dst-sync window (`eltwise_chain.hpp:14-20`) so it cannot be called inside another op's open window. The post-compute hook is explicitly designed for raw DEST-resident operations — see the chain's caller-init contract section (`eltwise_chain.hpp:25-50`). Helper considered: `compute_kernel_lib::unary` / `compute_kernel_lib::binary_op` (`eltwise_convenience.hpp:79-92`) — both open their own DST window and reserve/push the output CB; both incompatible with the open DEST window inside `PostComputeFn`. file:line of incompatibility: `eltwise_chain.hpp:14-20` ("chain owns ... modern dst-sync window"). |
| Compute: row-max reduce over `cb_scores` | helper | `compute_kernel_lib::reduce<PoolType::MAX, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitAndPopPerTile>(cb_scores, cb_max_scaler, cb_row_max, ReduceInputBlockShape::of(1, 1, 1))` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:400-415` | rows=1, cols=1, batches=1 (one score tile). Policy `WaitAndPopPerTile` waits cb_scores, computes max along W, packs to cb_row_max, pops cb_scores. | `cb_scores`, `cb_max_scaler` (held) | `cb_row_max` | Pool-type-aware: MAX uses row-0 scaler fill, which `calculate_and_prepare_reduce_scaler<PoolType::MAX, REDUCE_ROW>` already produced. Wait scaler tile once at kernel start (compute owns the first cb_wait_front; helper waits internally each call but does not pop). |
| Compute: running cur_max = max(cb_max_acc, cb_row_max) | helper | `compute_kernel_lib::binary_sfpu<BinaryMax<>, cb_max_acc, cb_row_max, cb_max_acc>(/*n_tiles=*/1)` for k ≥ 1; for k == 0, `compute_kernel_lib::copy<cb_row_max, cb_max_acc>(1)` to initialize. | `ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp:97-106` (binary_sfpu), `eltwise_convenience.hpp:109-117` (copy). `BinaryMax` from `eltwise_binary_sfpu.hpp`. | streaming case: n_tiles=1. In-place pattern: pass `cb_max_acc` as both first input and output (`eltwise_convenience.hpp:24-30` doc — "In-place = pass the input CB as the output CB"). For k=0 we need to also retain `cb_row_max` as the snapshot for `cb_max_prev` on the NEXT iteration — copy semantics handle this; k ≥ 1 path copies `cb_max_acc` → `cb_max_prev` first (helper `copy`), then runs the binary_sfpu max in-place. | `cb_max_acc` (in-place self), `cb_row_max` | `cb_max_acc` (in-place); `cb_max_prev` (snapshot for k≥1) | Sequence per K iteration: <br/>(a) if k ≥ 1: `copy<cb_max_acc, cb_max_prev>(1)` — snapshot prev_max before update.<br/>(b) if k == 0: `copy<cb_row_max, cb_max_acc>(1)` — initialize cur_max.<br/>(c) if k ≥ 1: `binary_sfpu<BinaryMax<>, cb_max_acc, cb_row_max, cb_max_acc>(1)` — running max in-place. <br/> CB-ownership rule for in-place satisfied: compute exclusively owns `cb_max_acc`. |
| Compute: cb_scores − cb_max_acc → cb_attn (col-broadcast), then exp in-place | helper | `compute_kernel_lib::eltwise_chain(EltwiseShape::single(), BinaryFpu<cb_scores, cb_max_acc, BinaryFpuOp::Sub, BroadcastDim::Col, BinaryDataFormatReconfig::Input, Streaming, HeldStream, OperandKind::Scalar>{}, Exp<>{}, PackTile<cb_attn, Dst::D0, OutStreaming>{})` | `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp:160-171` (the exact softmax-style sub+exp chain composition example) | One tile, col-broadcast (cb_max_acc is a column vector — value in col 0 replicated). A streams (pop after), B (`cb_max_acc`) is held (no pop). Chain fuses sub + exp under one tile_regs window — exactly the planner-skill-section recommendation to fuse `(x − max) → exp` into one chain. | `cb_scores`, `cb_max_acc` (held) | `cb_attn` | Helper directly matches softmax sub+exp pattern documented in chain header. Single chain owns wait/pop on cb_scores (pop after), wait on cb_max_acc (held — kept for downstream reuse), reserve/push on cb_attn. |
| Compute: row-sum reduce over `cb_attn` (without popping) | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(cb_attn, cb_sum_scaler, cb_row_sum, ReduceInputBlockShape::of(1, 1, 1))` | `reduce_helpers_compute.hpp:400-415` | `WaitUpfrontNoPop` waits cb_attn but does not pop — the same tile is consumed by the downstream S@V matmul next. Documented exactly in the helper's softmax example at `reduce_helpers_compute.hpp:363-368`. | `cb_attn` (waited, not popped), `cb_sum_scaler` (held) | `cb_row_sum` | Helper handles all DEST + CB reserve/push on cb_row_sum. |
| Compute: correction = exp(cb_max_prev − cb_max_acc) (k ≥ 1 only) | helper | `compute_kernel_lib::eltwise_chain(EltwiseShape::single(), BinaryFpu<cb_max_prev, cb_max_acc, BinaryFpuOp::Sub, BroadcastDim::None, BinaryDataFormatReconfig::Input, Streaming, HeldStream>{}, Exp<>{}, PackTile<cb_correction, Dst::D0, OutStreaming>{})` | `eltwise_chain.hpp:160-171` | Both column vectors of the same shape — no broadcast needed (BroadcastDim::None). A=cb_max_prev streams (pop after), B=cb_max_acc held. | `cb_max_prev` (pop), `cb_max_acc` (held) | `cb_correction` | Helper owns wait/pop on cb_max_prev, wait on cb_max_acc, reserve/push on cb_correction. |
| Compute: sum-acc update — cb_sum_acc = cb_sum_acc * cb_correction + cb_row_sum (k ≥ 1) or cb_sum_acc = cb_row_sum (k == 0) | helper | k=0: `compute_kernel_lib::copy<cb_row_sum, cb_sum_acc>(1)`. k≥1: chain `BinaryFpu<cb_sum_acc, cb_correction, Mul, BroadcastDim::None, Reconfig::Input, Streaming, HeldStream>` + Pack to scratch, then `BinaryFpu<scratch, cb_row_sum, Add>` + Pack to `cb_sum_acc`. Two-stage because `eltwise_chain` does not support a single chain with two binary FPU ops on different CB pairs. | `eltwise_chain.hpp:160-171` (binary FPU + pack); `eltwise_convenience.hpp:50-65` (binary_add/mul wrappers) | Sum-acc update is fundamentally a 3-CB binary fused op (`(sum_acc * corr) + row_sum`). Concrete reason for not using a single 1-stage helper: no `compute_kernel_lib` helper currently composes "binary FPU × binary FPU" into one chain across different CB pairs — `eltwise_chain` element constraints require a single `BinaryFpu` per chain (only one DEST-resident binary FPU), and `binary_dest_reuse` is for DEST-DEST chaining (not what's needed). Documented at `eltwise_chain.hpp:280-290` (only one `is_binary_fpu_op_v` element per chain). | `cb_sum_acc` (in-place self) and `cb_correction` (held); then `cb_sum_acc` (in-place) and `cb_row_sum` | `cb_sum_acc` (in-place) | Stage 1: `binary_op<MUL, A=cb_sum_acc, B=cb_correction>(1)` packs to a 1-tile intermediate CB (`cb_sum_scratch`, idx 11, 1 page, format bf16). Stage 2: `binary_op<ADD, A=cb_sum_scratch, B=cb_row_sum>(1)` packs to cb_sum_acc. **Decision: add `cb_sum_scratch` (idx 11, 1 page, bf16) to the CB table.** See CB list addendum below. |
| Compute: S @ V matmul (cb_attn @ cb_v_row → cb_partial_out) | helper | `compute_kernel_lib::matmul_block<transpose=false, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndPopPerKBlock>(cb_attn, cb_v_row, cb_partial_out, cb_attn /*interm*/, MatmulBlockShape::of(1, Dt, 1, 1, 1, 1))` | `matmul_block_helpers.hpp:790-823` | M=1, N=Dt, K=1, num_k_blocks=1. cb_attn is the in0 (1 tile) and DOES get popped here (previous reduce did not pop). interm placeholder = cb_attn (idiom for num_k_blocks==1). | `cb_attn` (1 tile), `cb_v_row` (Dt tiles) | `cb_partial_out` (Dt tiles) | Helper handles all CB ownership. Init mode `Short` issues `mm_block_init_short` to re-program for this matmul shape (post-QK-matmul state is the prior matmul shape M=1,N=1,K=Dt). |
| Compute: out-acc update — cb_out_acc = cb_out_acc * cb_correction (col bcast) + cb_partial_out (k ≥ 1) or cb_out_acc = cb_partial_out (k == 0) | helper | k=0: `compute_kernel_lib::copy<cb_partial_out, cb_out_acc>(Dt)` (sized n_tiles=Dt). k≥1: same two-stage pattern as sum-acc but Dt tiles. Stage 1: `binary_op<MUL, A=cb_out_acc, B=cb_correction, BroadcastDim::Col, B_policy=HeldStream>(Dt)` → cb_out_scratch (Dt pages). Stage 2: `binary_op<ADD, A=cb_out_scratch, B=cb_partial_out>(Dt)` → cb_out_acc. | `eltwise_convenience.hpp:50-65, 109-117`; `binary_op_helpers.hpp:262-264` | Same reason as sum-acc: needs 3-CB fused FMA; not expressible as one chain. Adds `cb_out_scratch` (idx 20, Dt pages, bf16). **Decision: add `cb_out_scratch` (idx 20, Dt pages, bf16) to CB table.** | `cb_out_acc` (in-place) + `cb_correction` (held — broadcast col); `cb_out_scratch` + `cb_partial_out` | `cb_out_acc` (in-place) | After this stage, `cb_correction` is popped (last consumer). |
| Compute: final divide — cb_out_acc / cb_sum_acc → cb_output (after K-loop) | helper | Two-stage: (a) `compute_kernel_lib::unary<Recip<>, cb_sum_acc, cb_sum_acc>(1)` (in-place recip on cb_sum_acc). (b) `compute_kernel_lib::binary_op<MUL, BroadcastDim::Col, A_policy=Streaming, B_policy=HeldStream>(cb_out_acc, cb_sum_acc, cb_output, BinaryInputBlockShape::of(1, Dt))` | `eltwise_convenience.hpp:79-92` (unary), `binary_op_helpers.hpp:262-264` (binary_op with bcast) | (a) 1-tile reciprocal in place. (b) Dt-tile multiply with col-broadcast (cb_sum_acc is column vector). | `cb_out_acc` (pop), `cb_sum_acc` (pop after) | `cb_output` | Final consumer of cb_out_acc and cb_sum_acc — these CBs get popped here. Helper handles all CB ops. |
| Writer: drain cb_output → DRAM | raw_api | `TensorAccessor` + `noc_async_write_tile` + `noc_async_write_barrier` per tile | `tech_reports/tensor_accessor/tensor_accessor.md` | Standard writer streaming pattern | `cb_output` | DRAM tensor | No helper covers generic tensor write streaming. |

**Persistent CB summary** (CBs whose lifetime spans the whole kernel, popped only at the very end):
- `cb_max_scaler`, `cb_sum_scaler`, `cb_scale` — manual `cb_pop_front(cb_*, 1)` at kernel end.

**CB table addendum** (intermediates added by sum-acc and out-acc updates):

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_sum_scratch` | 11 | 2048 B | `1` | bfloat16 | compute (sum-acc update stage 1) | compute (sum-acc update stage 2) | Within one K-iteration. |
| `cb_out_scratch` | 20 | 2048 B | `Dt` | bfloat16 | compute (out-acc update stage 1) | compute (out-acc update stage 2) | Within one K-iteration. |

## Compute Phases

Sequential phase execution per query tile-row. `Kt = S_kv / 32`. K-loop iterates `k = 0 .. Kt − 1`.

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | One-shot HW init + wait scaler tiles | raw (mm_init + compute_kernel_hw_startup) | `cb_max_scaler` (1, fronted), `cb_sum_scaler` (1, fronted), `cb_scale` (1, fronted) | — | scalers held, ready for K-loop |
| 1 | Wait Q-row (Dt tiles) — done implicitly by matmul helper's `WaitAndRetainOnLastBlock` | helper (implicit in phase 2) | `cb_q_row` (Dt, fronted) | — | Q tiles retained for whole K-loop |
| 2a (per k) | Q @ K[k]^T with fused scale + mask (post-compute) | `matmul_block` (helper) | `cb_q_row` (Dt held), `cb_k_row` (Dt streaming), `cb_scale` (1 held), `cb_mask` (1 streaming if HAS_MASK) | `cb_scores` (1 tile) | `cb_k_row` popped (Dt), `cb_mask` popped (1 if HAS_MASK), `cb_scores` pushed (1) |
| 2b (per k) | Row-max reduce over `cb_scores` | `reduce<MAX, REDUCE_ROW>` (helper) | `cb_scores` (1, popped after) | `cb_row_max` (1 tile) | `cb_scores` popped, `cb_row_max` pushed |
| 2c (per k) | Snapshot prev_max (k ≥ 1) | `copy` (helper) | `cb_max_acc` (1, retained) | `cb_max_prev` (1 tile) | `cb_max_prev` pushed |
| 2d (per k) | Running cur_max update — k=0: copy cb_row_max → cb_max_acc; k≥1: in-place max(cb_max_acc, cb_row_max) | `copy` / `binary_sfpu<BinaryMax<>>` (helper) | `cb_row_max` (1, popped), `cb_max_acc` (1, in-place) | `cb_max_acc` (1 tile, updated) | `cb_row_max` popped, `cb_max_acc` slot rewritten in place |
| 2e (per k) | (cb_scores − cb_max_acc) → exp → cb_attn | `eltwise_chain` (helper: sub<COL> + Exp) | `cb_scores` (was popped in 2b; produced anew? — see correction below), `cb_max_acc` (1 held) | `cb_attn` (1 tile) | — |

**Correction on phase 2b / 2e ordering**: the cb_scores tile must persist from phase 2a (matmul) through phase 2e (sub+exp) — it cannot be popped in 2b. Use `WaitUpfrontNoPop` on the MAX reduce so cb_scores stays fronted. Phase 2e's `eltwise_chain` then waits + pops cb_scores normally. **Corrected sequence**:

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 2a (per k) | Q @ K[k]^T with fused scale + mask | `matmul_block` | `cb_q_row` (Dt held), `cb_k_row` (Dt stream), `cb_scale` (held), `cb_mask` (stream) | `cb_scores` (1) | cb_k_row/cb_mask popped, cb_scores pushed |
| 2b (per k) | Row-max reduce (no pop) | `reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>` | `cb_scores` (1, no pop), `cb_max_scaler` (held) | `cb_row_max` (1) | cb_scores still fronted |
| 2c (per k) | Snapshot prev_max if k ≥ 1 | `copy` | `cb_max_acc` (1, held) | `cb_max_prev` (1) | — |
| 2d (per k) | cur_max update (k=0 copy, k≥1 in-place max) | `copy` or `binary_sfpu<BinaryMax<>>` | `cb_row_max` (1, popped), `cb_max_acc` (1, in-place when k≥1) | `cb_max_acc` (1) | cb_row_max popped, cb_max_acc updated |
| 2e (per k) | sub<COL> + Exp → cb_attn | `eltwise_chain` (BinaryFpu Sub + Exp + PackTile) | `cb_scores` (1, popped), `cb_max_acc` (1, held) | `cb_attn` (1) | cb_scores popped, cb_attn pushed |
| 2f (per k) | Row-sum reduce (no pop on cb_attn — needed for next matmul) | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` | `cb_attn` (1, no pop), `cb_sum_scaler` (held) | `cb_row_sum` (1) | cb_attn still fronted |
| 2g (per k, k≥1) | Compute correction = exp(prev_max − cur_max) | `eltwise_chain` (Sub + Exp + Pack) | `cb_max_prev` (1, popped), `cb_max_acc` (1, held) | `cb_correction` (1) | cb_max_prev popped, cb_correction pushed |
| 2h (per k) | sum-acc update | k=0: `copy`; k≥1: two-stage `binary_op<MUL>` then `binary_op<ADD>` via `cb_sum_scratch` | `cb_sum_acc` (in-place self), `cb_correction` (held), `cb_row_sum` (popped) | `cb_sum_acc` (1) | cb_row_sum popped; cb_correction still fronted (consumer in 2j) |
| 2i (per k) | S @ V matmul (consumes cb_attn) | `matmul_block` | `cb_attn` (1, popped), `cb_v_row` (Dt, popped) | `cb_partial_out` (Dt) | cb_attn + cb_v_row popped, cb_partial_out pushed |
| 2j (per k) | out-acc update | k=0: `copy` Dt; k≥1: two-stage `binary_op<MUL, BroadcastDim::Col>` via `cb_out_scratch` then `binary_op<ADD>` | `cb_out_acc` (in-place, Dt), `cb_correction` (popped after), `cb_partial_out` (popped) | `cb_out_acc` (Dt) | cb_correction + cb_partial_out popped; cb_out_acc updated |
| 3 | After K-loop: pop cb_q_row | raw `cb_pop_front(cb_q_row, Dt)` | `cb_q_row` (Dt) | — | cb_q_row drained |
| 4 | Final divide stage A: reciprocal of cb_sum_acc | `unary<Recip<>>` in-place | `cb_sum_acc` (1, in-place) | `cb_sum_acc` (1) | cb_sum_acc holds 1/sum |
| 5 | Final divide stage B: cb_out_acc × (1/cb_sum_acc) with col bcast | `binary_op<MUL, BroadcastDim::Col>` | `cb_out_acc` (Dt, popped), `cb_sum_acc` (1, popped) | `cb_output` (Dt) | cb_out_acc + cb_sum_acc popped, cb_output pushed |
| 6 | Per-kernel-end cleanup | raw `cb_pop_front` on `cb_max_scaler`, `cb_sum_scaler`, `cb_scale` (each 1) | — | — | all CBs drained at kernel exit |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 2a (post-compute scale) | MUL scalar | `cb_scores` All [32,32] | `cb_scale` All [32,32] (broadcast scalar — same value in all positions) | SCALAR |
| 2a (post-compute mask add) | ADD | `cb_scores` All [32,32] | `cb_mask` All [32,32] | NONE |
| 2e (sub before exp) | SUB | `cb_scores` All [32,32] | `cb_max_acc` Col0 only (REDUCE_ROW output, valid in col 0) | COL |
| 2g (sub prev−cur max) | SUB | `cb_max_prev` Col0 | `cb_max_acc` Col0 | NONE (both are col vectors of same shape) |
| 2h (sum acc mul stage 1) | MUL | `cb_sum_acc` Col0 | `cb_correction` Col0 | NONE |
| 2h (sum acc add stage 2) | ADD | `cb_sum_scratch` Col0 | `cb_row_sum` Col0 | NONE |
| 2j (out acc mul stage 1) | MUL | `cb_out_acc` All [32,32] (each of Dt tiles, full data) | `cb_correction` Col0 (broadcast col across each of Dt tiles) | COL |
| 2j (out acc add stage 2) | ADD | `cb_out_scratch` All | `cb_partial_out` All | NONE |
| 5 (final divide) | MUL | `cb_out_acc` All (per tile) | `cb_sum_acc` Col0 (broadcast col across each of Dt tiles) | COL |

Reduce output regions:
- `cb_row_max`, `cb_max_acc`, `cb_max_prev`: REDUCE_ROW with MAX → Col0 valid (`reduce_helpers_dataflow.hpp:46-49`).
- `cb_row_sum`, `cb_sum_acc`: REDUCE_ROW with SUM via `compute_kernel_lib::reduce` → Col0 valid (the helper routes SUM+REDUCE_ROW through matmul which produces col-0 layout per the dataflow scaler comment `reduce_helpers_dataflow.hpp:54-60`).

## Key Risks and Gotchas

| # | Risk | Mitigation in design |
|---|------|----------------------|
| 1 | Q must persist across all K iterations — losing it = wrong matmul on iteration ≥ 1. | `matmul_block` with `InputPolicy::WaitAndRetainOnLastBlock` on in0 + `num_k_blocks=1` → helper waits once, never pops. Manual `cb_pop_front(cb_q_row, Dt)` after K-loop. CB sized to exactly `Dt` pages (not double-buffered) since reader pushes Q once per query tile-row, before the K-loop. |
| 2 | `cb_scores` produced by phase 2a must survive phase 2b (reduce) so phase 2e can consume it. | Phase 2b uses `ReduceInputPolicy::WaitUpfrontNoPop` — exactly the softmax pattern documented at `reduce_helpers_compute.hpp:363-368`. |
| 3 | `cb_attn` produced by phase 2e must survive phase 2f (reduce) so phase 2i can consume it. | Phase 2f uses `ReduceInputPolicy::WaitUpfrontNoPop`; phase 2i `matmul_block` does the wait+pop on cb_attn. |
| 4 | `cb_correction` consumed by BOTH phase 2h (sum-acc update) and phase 2j (out-acc update) — must persist between them. | `cb_correction` page count = 2. Phase 2h uses `BinaryInputPolicy::WaitUpfrontNoPop` on B; phase 2j uses `WaitAndPopPerTile` on B (HeldStream → pops the final consumer). Producer-push count = 1; consumer-wait count = 1 (held across two consumers, popped after the second). |
| 5 | Reduce scaler CB must be bfloat16 packed format. | `cb_max_scaler` and `cb_sum_scaler` are `bfloat16`, 1 tile each (2048 B). Both use `calculate_and_prepare_reduce_scaler<>` pool-type-aware overload (`reduce_helpers_dataflow.hpp:94-101`), NOT the legacy untyped `prepare_reduce_scaler<cb>` overload. |
| 6 | Pool-type-aware fill layout: SUM+REDUCE_ROW via `compute_kernel_lib::reduce` uses matmul path (col-0 fill); MAX+REDUCE_ROW uses reduce-LLK path (row-0 fill). | `cb_max_scaler` uses `calculate_and_prepare_reduce_scaler<…, PoolType::MAX, ReduceDim::REDUCE_ROW>` (row-0 fill auto-picked). `cb_sum_scaler` uses `calculate_and_prepare_reduce_scaler<…, PoolType::SUM, ReduceDim::REDUCE_ROW>` (col-0 fill auto-picked — `compute_uses_reduce_tile` defaults to `false`, matching `compute_kernel_lib::reduce`'s matmul routing for SUM+REDUCE_ROW). |
| 7 | DEST register budget: max 8 tiles bf16 / 4 tiles fp32. | All helper calls operate on ≤ 2 tiles in DEST at a time (single matmul tile + mask tile in phase 2a; single sub+exp pair in phase 2e). Two-stage out-acc update means each binary_op processes ≤ DEST_AUTO_LIMIT tiles at a time (helper handles chunking). No phase exceeds the budget. |
| 8 | In-place CB pattern (`cb_max_acc`, `cb_sum_acc`, `cb_out_acc`) requires compute-exclusive CB ownership. | Reader does NOT touch these CBs. Writer does NOT touch these CBs. Documented at `binary_op_helpers.hpp:341-351`. |
| 9 | First matmul shape (M=1, N=1, K=Dt) ≠ second matmul shape (M=1, N=Dt, K=1). | Both use `matmul_block` with `InitMode::Short` (default) — helper re-issues `mm_block_init_short` per call, restoring matmul state from the prior call's reconfig. Boot-time `mm_init` covers the very first invocation. |
| 10 | Mask is optional. | `HAS_MASK` is a compile-time arg (1 or 0). When 0: reader does not push to `cb_mask` (host omits the CB descriptor), `MaskScalePostCompute` functor's `if constexpr (HAS_MASK)` branch is elided, and the K-loop never waits on cb_mask. CB index 3 is still reserved in the global enum. |
| 11 | Causal mask correctness for cross-attention. | EXCLUSION declared: `{"mask_mode": "causal", "attention_kind": "cross"}`. Kernel treats mask as a generic additive tensor — caller passes a real causal mask when they want one. |
| 12 | Scale handling for `auto` vs `explicit`. | Both paths reduce to a single bf16 value packed into the `cb_scale` tile by the reader. The host computes `scale_value = scale_arg if scale_arg is not None else 1.0/math.sqrt(D)` and passes it as a fp32-bit CT arg; reader bit-casts to fp32, converts to bf16, fills the tile. |
| 13 | Output normalization stability when `cur_sum` is very small (numerical risk). | Phase 4 uses `Recip<>` (the SFPU reciprocal). For exp-scaled scores with mask of -inf, sums can be near zero only when an entire row is masked — same condition where torch produces NaN. PCC tolerance handles small precision drift. |
| 14 | Cross-attention with S_q != S_kv: Qt and Kt differ; total work uses Qt. | Work-distribution formula uses `Qt = S_q / 32`. K-loop runs `Kt = S_kv / 32` iterations. Both passed as CT args. |
| 15 | The Q tile-row's address calculation depends on (b, h, qt) → tile-id mapping. | Reader CT args include B, H, Qt, Kt, Dt. Per `row_id`, reader computes `b = row_id / (H*Qt); h = (row_id / Qt) % H; qt = row_id % Qt`. Q tile index range: `((b*H + h)*Qt + qt) * Dt + d` for d in [0, Dt). Same pattern for K, V, and mask. |

## Structural impossibilities (post-pipeline note)

`feature_spec.py` already declares `INVALID = []` and explains the reason (TILE-only op, no ROW_MAJOR layout values in TARGET, so the canonical bf8b+ROW_MAJOR rule is vacuous). The planner has no additional structural-impossibility candidates to file.
