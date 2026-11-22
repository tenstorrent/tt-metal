# ttnn.bringup.offset_cumsum

## What it does
Fork of `ttnn.experimental.deepseek_prefill.offset_cumsum` (source
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum` @ `67ca5f3af48`, synced to `73027b6e6ff`,
unchanged through `bdc576fa652`): MoE routing bookkeeping. Takes this chip's expert histogram [E] (UINT32, from
`masked_bincount`), all-gathers it along `cluster_axis`, and returns four UINT32 tensors:
`global_dispatch_offsets` [1, E] (where this chip writes each expert's tokens in the destination buffer),
`total_counts_per_expert` [1, E], `expert_region_offsets` [1, E] (tile-aligned exclusive prefix sum within each chip's
expert group), `all_global_dispatch_offsets` [devices on axis, E] (every device's offsets row).

This fork adds: on a 1-device dispatch axis the internal `ttnn.all_gather` is skipped (it rejects a single-device
axis) and the local histogram goes straight to the prefix sum; the 4th output then has one row. Multi-device axes run the
source path unchanged.

Everything else as `ttnn.experimental.deepseek_prefill.offset_cumsum`.

## Consider it instead of
- `ttnn.experimental.deepseek_prefill.offset_cumsum` when the dispatch group is a single chip (1xN mesh dispatching
  along its size-1 axis).
- `ttnn.all_gather` + `ttnn.cumsum` + host-side region math: one op, results stay on device for `dispatch`,
  `unified_routed_expert_moe` and `combine`.

## Options
Realistic (use by default): every option below.

- `input_tensor`: 1D UINT32 [E] histogram.
- `cluster_axis`: required, dispatch axis.
- `num_links`: required, links for the all_gather.
- `experts_per_chip`: required, expert-group stride for the region offsets.
- `memory_config`: required, for intermediates and outputs.
- `use_l1_small_for_semaphores=False`: all_gather semaphores in L1_SMALL.
No fork-only option: the fork change is automatic on a 1-device axis. Returns 4 tensors (callers unpack
`offsets, counts, region_offsets, _`).

Precision only: per-step checks or a measured end-to-end gap (rule 7): none.

## Measured
- Accuracy: exact on all outputs (every model case in `tests/cases.py` is `"exact": True`; reported exact on 2x2 in
  `models/demos/mimo_v2_6_d_p_2x2/bringup/BREADCRUMBS.md` line 482 and
  `models/demos/glm53_flash_d_p/bringup/BREADCRUMBS.md` line 2467).
- Speed: not measured in this tree (the source's #57859 claims its 4-output kernel is 2-3x faster, quoted in
  `CHANGELOG.md`).
- Source tests carried (`tests/source_baseline.json`): 25 / 25 pass on original and fork.

## Limits
- Return arity changed from 3 to 4 with the #57859 sync (`CHANGELOG.md`); every caller unpacks four.
- Carried source tests run the ring 4x1 and 2x2 cases only; 8- and 32-device cases and the 2x4 L1_SMALL/cache test are
  not run (`tests/source.yaml`). The 1-device axis is covered only by the fork's model cases.
- The xing40 4x2 model case needs 8 chips. No model case for deepseek_v41_flash_d_p's call.

## How to call
```python
offsets, counts, region_offsets, _ = ttnn.bringup.offset_cumsum(
    hist, cluster_axis=0, num_links=1, experts_per_chip=32, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
```
(`models/demos/gemma4_a4b_d_p/tt/experts.py`). Tests: `ttnn/ttnn/bringup/offset_cumsum/tests/`
(`test_offset_cumsum.py`, `cases.py`, `source.yaml`).

## Used by
- ernie45_d_p (`tt/moe_unified.py`, 1x4; forked for P3.1)
- gemma4_a4b_d_p, mimo_v2_6_d_p (`tt/experts.py`, 1x4)
- mimo_v2_6_d_p_2x2, glm53_flash_d_p, hy4_preview_d_p (`tt/experts.py`, 2x2, axis 0)
- xing40_a4b_d_p (`tt/experts.py`, 4x2, axis 0)
- deepseek_v41_flash_d_p (`tt/experts.py`, axis 0)

## History
`CHANGELOG.md`.
