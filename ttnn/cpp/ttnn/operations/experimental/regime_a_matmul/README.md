# ttnn.experimental.regime_a_matmul

DRAM-bandwidth-optimal matmul for **Regime A** (`M < N`) on Blackhole: `out[M,N] = in0[M,K] @ in1[K,N]`,
bf16 in/out, HiFi2 math, fp32 accumulation. Purpose-built for the low-arithmetic-intensity skinny
matmuls (FLUX/LTX transformer weights) where the operand-read bandwidth, not compute, is the wall.

## Why it exists

For `M < N` the weight `in1[K,N]` is the large operand. Reading it at ~full DRAM bandwidth requires
each worker to read from a **bank-adjacent** DRAM channel (the `reader == consumer` pattern) and to
split reads across both NoCs. This op width-shards `in1` across the 8 DRAM banks and places one worker
per bank per slice, alternating NOC0/NOC1, so the in1 read sustains ~500 GB/s. `in0` (small) is
delivered by an 8-wide **ring all-gather**; split-K partial sums are reduced up a linear chain.

On the target shapes it reaches **79–94% of peak DRAM bandwidth** and reproduces the tuned C++
prototype's kernel time within **1%** (see `tools/mm_sweep/GOLDEN_PARITY_SUITE.md`).

## API

```python
# Weight must be DRAM width-sharded across the 8 banks via this helper (layout depends on (K,N) only):
w_mem = ttnn.create_regime_a_weight_memory_config(list(weight.shape), ttnn.bfloat16, device)
in1   = ttnn.from_torch(weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=w_mem)
in0   = ttnn.from_torch(act,    layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)  # DRAM interleaved

# config=None -> auto-select; or pin it for reproducibility:
cfg = ttnn.RegimeAMatmulConfig(k_slices=12, n_slices=1, m_slices=1, k_block_tiles=2, n_subblock_tiles=1)
out = ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)   # config optional
```

Config knobs (all in tiles / slice counts): `k_slices` (Pk, split-K depth), `n_slices` (Ns),
`m_slices` (Sm), `k_block_tiles` (kb), `n_subblock_tiles` (nsb). The grid is `8 × Pk × Ns × Sm` cores.
`config=None` selects via the ported FLUX/LTX picker (lookup table + cost-model fallback).

## Structure

- `device/regime_a_matmul_plan.hpp` — **pure host planner** (no tt_metal deps): given tile dims + a
  config + the device's bank-adjacent core assignments, produces one canonical `ExecutionPlan`
  (padded geometry, per-core bank/NoC/ring/reduction assignment, CB/L1 sizing, kernel-arg values).
  Unit-tested offline in `tools/mm_sweep/regime_a_plan_test.cpp` (273 checks, no hardware).
- `device/regime_a_matmul_config.{hpp,cpp}` — public `RegimeAMatmulConfig`, the device adapter
  (`make_and_build_plan`), the auto-selector (`auto_select_config`), and the in1 weight-sharding helper.
- `device/regime_a_matmul_program_factory.cpp` — translates the plan into circular buffers, semaphores,
  split-NoC kernel placement, and per-core runtime args.
- `device/kernels/` — three focused kernels, no experimental mode matrix:
  - `in1_reader.cpp` — DRAM-sharded in1 reader (`reader == consumer`), rotated shard order, optional
    M-split forward.
  - `in0_ring_reduce_writer.cpp` — in0 ring all-gather + split-K reduction chain + output write.
  - `compute.cpp` — forked from `minimal_matmul` (HiFi2, fp32 acc, `IN0_KSLICE_RESIDENT` + `REDUCE_K`).

## Constraints (v1)

- Blackhole only; `Mt = ceil(M/32)` in 1..8 is the tuned range (works beyond, less optimal).
- bf16 in/out, HiFi2, fp32 accumulation. No bias / activation / transpose / fusion / batching.
- `in1` must be DRAM width-shardable across 8 banks (device must expose ≥ 8 DRAM banks).
- **Non-divisible dims**: currently handled by padding the inputs to bank alignment (K→mult-of-8 tiles,
  N→mult-of-8 tiles) with zeros — correct (PCC ≥ 0.999) but reads the pad tiles. The efficient
  **balanced floor/ceil tail** (read only valid tiles, local zero-fill; recovers the ~5% padding-waste
  gap measured on the non-divisible oracle shape) is the main planned follow-up. The planner already
  computes per-core valid extents (`CorePlan.valid_k/valid_m/valid_n`) for this.

## Tests

`tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul.py` — random BF16 vs Torch, PCC ≥ 0.999:
Pk=1, split-K reduction, Ns>1, Sm>1, golden Mt=1/2/4/8, non-divisible, program-cache replay,
column-dependent layout, and `config=None` auto-selection.
