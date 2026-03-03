# SDPA Performance Testing Manual

## Test File

`tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py`

## Key Tests

| Test | Purpose | Takes `device` fixture? |
|------|---------|------------------------|
| `test_sdpa_sweep_perf_impl` | Single config run (used as subprocess by perf table) | Yes |
| `test_sdpa_accuracy` | PCC + RMSE check against PyTorch reference | Yes |
| `test_sdpa_determinism` | 10 runs, checks exact bitwise match | Yes |
| `test_sdpa_create_perf_table` | Sweeps all chunk combos, prints ranked table | **No** (spawns subprocesses) |

## Running the Perf Table

```bash
# Single shape:
pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table[wan_1xGLX_analog] -s

# Both shapes:
pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_create_perf_table -s
```

Requires tracy profiling (`python3 -m tracy`). The test spawns profiled subprocesses via `run_device_profiler` for each (q_chunk, k_chunk) combo.

## Running a Single Config

```bash
pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_sweep_perf_impl[wan_1xGLX_analog-k256-q288-bf16] -s
```

Test ID format: `[{shape_id}-k{k_chunk}-q{q_chunk}-bf16]`

## Shapes and Chunk Sizes

| Shape ID | b | nh | s | d |
|----------|---|-----|------|-----|
| wan_1xGLX_analog | 1 | 10 | 9472 | 128 |
| wan_4xGLX_analog | 1 | 10 | 2368 | 128 |

- Q chunk sizes: 224, 256, 288
- K chunk sizes: 128, 256, 512

## Which Configs Hit `sdpa_standard_v2`

Selection is in `sdpa_program_factory.cpp` (~line 328). The `use_streaming_compute` flag requires ALL of:
- `is_causal=False` (test uses False)
- No mask, no attention_sink, no sliding_window, not chunked
- `qk_out_subblock_h <= 2`
- `Sk_chunk_t % (8 / qk_out_subblock_h) == 0`

With d=128 (vDHt=4) and fp32_dest_acc_en=False (dst_size=8), subblock_h is computed by `determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, 8)` in `sdpa_subblock_utils.hpp`.

| q_chunk | k_chunk | Sq_t | Sk_t | subblock (h,w) | Hits v2? |
|---------|---------|------|------|----------------|----------|
| 224 | 128 | 7 | 4 | (7, 1) | **No** (h=7) |
| 224 | 256 | 7 | 8 | (1, 8) | Yes |
| 224 | 512 | 7 | 16 | (1, 8) | Yes |
| 256 | 128 | 8 | 4 | (2, 4) | Yes |
| 256 | 256 | 8 | 8 | (2, 4) | Yes |
| 256 | 512 | 8 | 16 | (2, 4) | Yes |
| 288 | 128 | 9 | 4 | (3, 2) | **No** (h=3) |
| 288 | 256 | 9 | 8 | (1, 8) | Yes |
| 288 | 512 | 9 | 16 | (1, 8) | Yes |

7 of 9 configs hit v2. Confirm at runtime with `TT_METAL_LOGGER_LEVEL=INFO` — logs `use_streaming_compute: true/false`.

## Running Only v2-Eligible Configs

Use `run_v2_perf.py` in the repo root. It filters to the 7 v2-eligible combos and produces the same perf table.

```bash
python3 run_v2_perf.py
```

## A/B Comparison (v2 vs old `sdpa_standard`)

To force the old path for comparison:

1. Edit `sdpa_program_factory.cpp` (~line 328):
   ```cpp
   const bool use_streaming_compute = false;  // HACK: force old path
   ```
2. Rebuild: `./build_metal.sh`
3. Clear JIT cache: `rm -rf built/tt-metal-cache*`
4. Run: `python3 run_v2_perf.py`
5. **Revert the hack** and rebuild when done.

The rebuild step MUST use `./build_metal.sh` (not just `cmake --build`), otherwise the installed kernel copies won't update and you'll get stale results.

## Utilization Model

Math utilization measures what fraction of peak matmul throughput is spent on useful work:

```
util = useful_FLOPs / (wall_time × peak_TFLOPS)
useful_FLOPs = 4 × s² × d × nh × b
```

The `4 × s² × d` comes from two matmuls per (Q, K) pair in `sdpa_inner_loop_step` (`compute_common.hpp`): Q@KT (2×s²×d) and QKT@V (2×s²×d). Three factors reduce utilization below 100%.

### 1. Core parallelization waste

110 cores are assigned hierarchically (`sdpa_program_factory.cpp:254–258`):

```
batch_parallel = min(B, num_cores)                                   = 1
nh_parallel    = min(num_cores / batch_parallel, NQH)                = 10
q_parallel     = min(num_cores / (batch_parallel × nh_parallel), q_num_chunks)
               = min(11, q_num_chunks)
```

Each head gets `q_parallel` cores. Each core processes `q_per_core = ceil(q_num_chunks / q_parallel)` Q chunks, iterating over all `k_num_chunks` K chunks per Q chunk. Wall time is gated by the busiest core.

Two forms of waste:
- **Idle cores**: when `q_num_chunks < 11`, only `q_num_chunks × 10` cores are active. The rest contribute nothing but the utilization denominator counts all 110 cores' capacity.
- **Load imbalance**: when `q_num_chunks` isn't divisible by `q_parallel`, the last core(s) per head process fewer Q chunks while wall time is set by the busiest.

```
core_waste = 1 − (nh × q_num_chunks) / (num_cores × q_per_core)
```

### 2. Padding waste

Each `sdpa_inner_loop_step` processes one (Q chunk, K chunk) pair. Both matmuls use the full chunk tile counts:
- **Phase 1 — Q@KT**: Sq_chunk_t × Sk_chunk_t output tiles, inner dim DHt
- **Phase 2 — QKT@V**: Sq_chunk_t × vDHt output tiles, inner dim Sk_chunk_t

The host pads s to `ceil(s / chunk_size) × chunk_size`. The reader (`read_chunk_with_padding` in `dataflow_common.hpp`) zero-fills tiles beyond the valid sequence length. Both matmuls pay full cost for zero-padded tiles.

```
Sqt = Skt = s / 32          (tile-aligned for these shapes)
q_chunks = ceil(Sqt / Sq_chunk_t)
k_chunks = ceil(Skt / Sk_chunk_t)
pad_waste = 1 − (Sqt × Skt) / (q_chunks × Sq_chunk_t × k_chunks × Sk_chunk_t)
```

### 3. Algorithmic overhead

Softmax (sub_exp, reduce_max, row_sum), SALAD corrections (exp_max_diff, rescaling), and normalization (matmul_reduce, recip, multiply) use SFPU/pack cycles that don't count as matmul FLOPs. This is roughly constant across chunk configs and accounts for the remaining gap to 100%.

## Results (2026-03-03, Blackhole 110 cores)

`old` = `sdpa_standard`, `v2` = `sdpa_standard_v2`.

### wan_1xGLX_analog (b=1, nh=10, s=9472, d=128) — 459.36 GFLOPs

Sqt = Skt = 296 tiles. q_parallel = 11 for all configs (q_num_chunks ≥ 33).

| Rank | Sq_chunk_t | Sk_chunk_t | sbh | sbw | q_chunks | k_chunks | Core waste | Pad waste | Dur v2 (ms) | Util v2 | Util old | vs old |
|------|------------|------------|-----|-----|----------|----------|------------|-----------|-------------|---------|----------|--------|
| 1 | 9 | 16 | 1 | 8 | 33 | 19 | 0.0% | 3.0% | 2.329 | 64.8% | 55.9% | +15.9% |
| 2 | 7 | 16 | 1 | 8 | 43 | 19 | 2.3% | 4.2% | 2.458 | 61.4% | 52.4% | +17.2% |
| 3 | 9 | 8 | 1 | 8 | 33 | 37 | 0.0% | 0.3% | 2.629 | 57.5% | 52.2% | +10.2% |
| 4 | 7 | 8 | 1 | 8 | 43 | 37 | 2.3% | 1.7% | 2.761 | 54.7% | 49.1% | +11.4% |
| 5 | 8 | 16 | 2 | 4 | 37 | 19 | 15.9% | 2.6% | 2.857 | 52.9% | 46.4% | +14.0% |
| 6 | 8 | 8 | 2 | 4 | 37 | 37 | 15.9% | 0.0% | 3.062 | 49.3% | 43.8% | +12.6% |
| 7 | 8 | 4 | 2 | 4 | 37 | 74 | 15.9% | 0.0% | 3.660 | 41.3% | 37.2% | +11.0% |

Ranks 1–4 (sbh=1) outperform ranks 5–7 (sbh=2) despite sbh=1 requiring more subblock iterations. The dominant effect is that Sq_chunk_t=8 (q=256) gives q_num_chunks=37, which distributes badly across 11 cores: 9 cores get 4 Q chunks, 1 gets 1, and 1 is idle — 15.9% core waste. In contrast, Sq_chunk_t=9 gives q_num_chunks=33=11×3, a perfect fit with 0% core waste.

Rank 7 (Sk_chunk_t=4) is slowest despite 0% padding waste: k_chunks=74 means 74 K iterations per Q chunk — the per-iteration overhead (matmul init, sub_exp, reduce, SALAD) dominates.

### wan_4xGLX_analog (b=1, nh=10, s=2368, d=128) — 28.71 GFLOPs

Sqt = Skt = 74 tiles. q_parallel = q_num_chunks for all configs (all ≤ 11), so q_per_core = 1 and there is no load imbalance among active cores — only idle cores.

| Rank | Sq_chunk_t | Sk_chunk_t | sbh | sbw | q_chunks | k_chunks | Core waste | Pad waste | Dur v2 (ms) | Util v2 | Util old | vs old |
|------|------------|------------|-----|-----|----------|----------|------------|-----------|-------------|---------|----------|--------|
| 1 | 7 | 16 | 1 | 8 | 11 | 5 | 0.0% | 11.1% | 0.187 | 50.4% | 43.7% | +15.3% |
| 2 | 7 | 8 | 1 | 8 | 11 | 10 | 0.0% | 11.1% | 0.214 | 44.0% | 40.0% | +10.0% |
| 3 | 8 | 16 | 2 | 4 | 10 | 5 | 9.1% | 14.4% | 0.227 | 41.6% | 36.5% | +14.0% |
| 4 | 8 | 8 | 2 | 4 | 10 | 10 | 9.1% | 14.4% | 0.236 | 39.9% | 36.0% | +10.8% |
| 5 | 9 | 16 | 1 | 8 | 9 | 5 | 18.2% | 15.5% | 0.249 | 37.9% | 33.1% | +14.5% |
| 6 | 9 | 8 | 1 | 8 | 9 | 10 | 18.2% | 15.5% | 0.266 | 35.5% | 32.3% | +9.9% |
| 7 | 8 | 4 | 2 | 4 | 10 | 19 | 9.1% | 9.9% | 0.266 | 35.4% | 32.3% | +9.6% |

Sq_chunk_t=7 (q=224) ranks best: it is the only chunk size that gives q_num_chunks=11, using all 110 cores with 0% core waste. Sq_chunk_t=9 (q=288) gives q_num_chunks=9 — 20 cores idle, 18.2% core waste — compounding with the worst padding waste (15.5%) for a double hit.

k=256 and k=512 both pad K to the same total (80 tiles) because ceil(74/16)×16 = ceil(74/8)×8 = 80. Larger K chunks buy no benefit here.

### Why wan_4xGLX_analog has worse utilization

Both shapes share the factor 37 (9472=256×37, 2368=64×37), but the shorter sequence compounds waste on both axes:

**Core waste**: s=2368 produces fewer Q chunks (9–11 vs 33–43), so q_parallel can fall below 11, leaving entire cores idle. wan_1x always uses all 110 cores.

**Padding waste**: Skt=74 = 2×37 does not divide evenly by any Sk_chunk_t (74%4=2, 74%8=2, 74%16=10) or Sq_chunk_t (74%7=4, 74%8=2, 74%9=2). Every config wastes tiles on both Q and K. In contrast, Skt=296 = 8×37 = 4×74 divides evenly by Sk_chunk_t=4 and 8, and Sq_chunk_t=8.

Combined: wan_4x configs face 9–18% core waste plus 10–16% padding waste, vs wan_1x's 0–16% core waste and 0–4% padding waste.

v2 delivers +10–17% math utilization improvement over old across all configs.

## Gotcha: Stale Builds

`cmake --build` alone does NOT update the installed kernel copies that the JIT compiler uses. Always:
1. `./build_metal.sh` (runs cmake install)
2. `rm -rf built/tt-metal-cache*` (clears JIT cache)

Without both steps, you'll measure the old binary and get misleading A/B numbers.
