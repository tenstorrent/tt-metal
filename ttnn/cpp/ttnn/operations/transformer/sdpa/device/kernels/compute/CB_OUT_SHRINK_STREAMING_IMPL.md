# Single-chip streaming SDPA `cb_out` shrink — implementation notes

Companion to [`CB_OUT_SHRINK_STREAMING.md`](./CB_OUT_SHRINK_STREAMING.md) (design)
and [`OUT_SUBBLOCK_STREAMING.md`](./OUT_SUBBLOCK_STREAMING.md) (host subblock picker).

This document records what actually landed and the gotchas we hit while moving
from the single up-front reserve + bulk drain to per-group reserves + row-grouped
drain.

## What changed

### 1. Host — `sdpa_program_factory.cpp`

- `out0_t` is now chosen based on `use_streaming_compute`. Streaming path:
  `out0_t = streaming_out0_groups * effective_qktv_h * vDHt`, with
  `streaming_out0_groups = 2` — pure ping-pong (1 pending SALAD slot + 1
  matmul-in-flight slot), which is the minimum that preserves the reserve
  invariant `cb_size >= qktv_h*vDHt + cur_h*vDHt`. No writer-slack slot: perf
  measurements below show the writer keeps up without it.
  `effective_qktv_h` mirrors the kernel's qktv_h bump (promotes `h=1` to `h=2`
  when `2*out_out_subblock_w <= dst_size` and `Sq_chunk_t >= 2`).
- Non-streaming paths keep the original `out0_t = Sq_chunk_t * vDHt` sizing
  untouched.
- Writer compile-time args gained two entries — `use_streaming_compute` (21)
  and `out_out_subblock_h` (22). `TensorAccessorArgs` now starts at index 23.

### 2. Kernel — `compute_streaming.hpp`

- Removed the single up-front `cb_reserve_back(out_cb, qktv_output_num_tiles)`.
- Iter 0 (`q_subblock = 0`): reserves one slot, `qktv_h * vDHt`.
- Main loop (`q_subblock = 1 .. total_v_row_groups - 1`): reserves
  `(qktv_h + cur_h) * vDHt` before the matmul. **This is the subtle part:** at
  the point where we call `cb_reserve_back`, the previous group's slot is
  already written at offset `[0, qktv_h*vDHt)` but not yet pushed, and the new
  matmul will write at offset `[qktv_h*vDHt, qktv_h*vDHt + cur_h*vDHt)`.
  `cb_reserve_back(N)` only guarantees the first `N` tiles ahead of `wr_ptr`
  are free; we must reserve the full span we're about to touch, otherwise the
  matmul write (at offset `qktv_h*vDHt`) can overwrite unacked consumer data
  when `wr_ptr` wraps.
- SALAD / normalize / push ordering is unchanged; the "pending slot" pattern
  of the original code already matches per-group reserves.

### 3. Writer — `dataflow_common.hpp` + `writer_interleaved.cpp`

- Added `write_block_row_grouped`. Drains `total_rows` rows from the CB (= what
  compute pushes, always `Sq_chunk_t`) but only issues NoC writes for the first
  `write_rows` rows (= `out_row_tile_count`, capped at `valid_Sqt` for padded
  chunks). Padding rows are popped but not written — this is required because
  compute always pushes the full `Sq_chunk_t` regardless of padding, so popping
  only the valid rows would leave padding tiles in the CB and deadlock compute
  on `cb_reserve_back`.
- Calls `noc_async_writes_flushed()` before each `cb_pop_front` so compute can
  safely reuse the L1 slot for the next group.
- `writer_interleaved.cpp` dispatches to `write_block_row_grouped` when
  `use_streaming_compute` is set. `sbh` is the host's `out_subblock_h` (not the
  kernel's bumped `qktv_h`); `cb_wait_front` is a `>=` check so smaller-sbh
  drains still unblock correctly when compute pushes larger groups.

## Results (Blackhole, `wan2_2_{1,4}xGLX_analog`, bf16)

### Correctness
- All 18 accuracy configs pass (q ∈ {224, 256, 288} × k ∈ {128, 256, 512} ×
  {`wan2_2_4xGLX_analog`, `wan2_2_1xGLX_analog`}). PCC ≥ 0.9997, RMSE < 4e-2.
- All 9+9 perf configs complete on both shapes. The row-grouped drain pops the
  full `Sq_chunk_t` per group (padded chunks included) so compute can't deadlock
  on `cb_reserve_back` when the writer has fewer valid rows to emit.

### L1 footprint

`cb_out` with 2-deep sizing is a fixed `2 * qktv_h * vDHt = 16` tiles = **32 KB**
(bf16) regardless of `Sq_chunk_t`, so the savings grow with the Q chunk size.

Best configs from the perf sweep:

| Config (best) | Sq_chunk_t | Baseline tiles | Baseline L1 | 2-deep tiles | 2-deep L1 | Δ |
|---------------|------------|----------------|-------------|--------------|-----------|---|
| 4xGLX q224-k512 | 7 | 28 | 56 KB | 16 | 32 KB | **−24 KB (−43%)** |
| 1xGLX q288-k512 | 9 | 36 | 72 KB | 16 | 32 KB | **−40 KB (−56%)** |

### Kernel duration (best configs)

| Shape  | Best cfg    | Duration | Math util |
|--------|-------------|----------|-----------|
| 4xGLX  | q224-k512   | 0.164 ms | 57.5%     |
| 1xGLX  | q288-k512   | 2.162 ms | 69.9%     |

No writer-induced `cb_reserve_back` stalls observed — the writer keeps up with
MM2 without the extra slack slot, so triple-buffering is unnecessary.

## References (same as design doc)
- `sdpa_program_factory.cpp` — CB allocation + writer CT args.
- `compute_streaming.hpp` — per-group reserves in Phase 2 matmul body.
- `dataflow_common.hpp::write_block_row_grouped` — row-grouped drain.
- `writer_interleaved.cpp` — dispatches streaming vs non-streaming drain.
