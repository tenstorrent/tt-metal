# ttnn.bringup.dispatch

## What it does
Fork of `ttnn.experimental.deepseek_prefill.dispatch` (source
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch` @ `67ca5f3af48`, synced to `bdc576fa652`): MoE
token dispatch. Each token row is copied to the chip hosting each of its top-k experts, into a flat per-chip buffer
(experts packed back to back, regions tile-aligned, positions from `offset_cumsum`), plus a metadata row
`[linearized_mesh_coord, token_idx, topk_idx]` per copy for `combine`. Local writes over NoC, remote ones over fabric.

This fork adds: a 1-device dispatch axis. When the dispatch axis has one device (or `num_links == 0`) all fabric setup is
skipped and the kernels' no-fabric build runs; the source TT_FATALs "No neighbors found". Multi-device axes run the
source's fabric path unchanged.

Everything else as `ttnn.experimental.deepseek_prefill.dispatch`.

## Consider it instead of
- `ttnn.experimental.deepseek_prefill.dispatch` when the dispatch group is a single chip (e.g. a 1xN mesh dispatching
  along its size-1 axis). On a multi-chip axis the two are the same program.
- An all-gather of tokens plus per-expert gathers / `ttnn.experimental.deepseek_prefill.extract`: dispatch moves only
  the routed rows, sized by device-resident offsets.

## Options
Positional: `input_tensor` (1, S, H) BFLOAT16 or FP8_E4M3 (Blackhole), TILE (packer converts) or ROW_MAJOR (byte copy,
dtype must equal output); `indices_tensor` (1, S, K); `expert_offsets_tensor` (1, E); `expert_dispatch_table_tensor`
(1, E), chip id or -1.
Keyword (required): `dispatch_group_size`, `experts_per_chip`, `num_routed_experts`, `num_experts_per_tok`,
`metadata_len` (3, or 3 + emb/128 with scales), `max_dispatch_buffer_token_size`.
Keyword (optional):
- `padding_config=None`: per-device [real_tokens, pad_side]; bounds the token loop.
- `scales_tensor=None`: per-token fp8 scales copied into the metadata tail (Blackhole).
- `memory_config=None`, `subdevice_id=None`.
- `cluster_axis=None`: dispatch mesh axis; only 0 / None pass validation, FABRIC_2D needs 0 explicitly.
- `num_links=1`: fabric links; 0 also disables fabric (fork gate).
- `topology=Topology.Linear`.
- `use_l1_small_for_semaphores=False`: the GlobalSemaphores in L1_SMALL instead of L1.
- `fp8_output=False`: FP8_E4M3 buffer (Blackhole).
- `fp8_scaled_input=False`: fp8 ROW_MAJOR input + scales_tensor path (Blackhole).
- `num_workers_per_sender=2`: worker cores per sender (TILE and ROW_MAJOR input).
No fork-only option: the fork change is automatic on a 1-device axis.
Returns `(dispatched_buffer (1, 1, max_dispatch_buffer_token_size, H), metadata (1, 1, max_..., metadata_len))`.

## Measured
- Accuracy: exact. Every model case (`tests/cases.py`, `"exact": True`) compares bit for bit; reported exact on 2x2
  (`models/demos/mimo_v2_6_d_p_2x2/bringup/BREADCRUMBS.md` line 482, `models/demos/glm53_flash_d_p/bringup/BREADCRUMBS.md`
  line 2467).
- Speed: deepseek_v41_flash_d_p, 2x2, layer 0, chunk 2048, sync-instrumented step time: dispatch 1.8 ms
  (`models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md` line 1638-1641). ernie45_d_p 1x4, 28 layers, one
  50k->55k chunk: router + dispatch + combine together 100 ms (`models/demos/ernie45_d_p/bringup/BREADCRUMBS.md`
  line 120). Not measured on its own otherwise.
- Source tests carried (`tests/source_baseline.json`): 80 / 82 pass on original and fork; the same 2 fail on both.

## Limits
- Known failures (both original and fork, `tests/source.yaml`): the 2 gptoss_120b fp8_scaled_in cases (emb 2880 not a
  multiple of the 128-wide scale block).
- fp8 output / fp8 input / scales: Blackhole only. ROW_MAJOR input needs input dtype == output dtype.
- Carried source tests run only the 4-device cases (ring 4x1) on a 4-chip box; 8- and 32-device cases never run here.
  The 1-device axis is covered only by the fork's model cases.
- The xing40 4x2 model case needs 8 chips. No model case for deepseek_v41_flash_d_p's call (ROW_MAJOR byte copy, axis 0).

## How to call
```python
buf, meta = ttnn.bringup.dispatch(
    input_tensor=x, indices_tensor=idx, expert_offsets_tensor=offsets, expert_dispatch_table_tensor=table,
    dispatch_group_size=1, experts_per_chip=32, num_routed_experts=128, num_experts_per_tok=8, metadata_len=3,
    max_dispatch_buffer_token_size=41952, cluster_axis=0, num_links=1, topology=ttnn.Topology.Linear,
)
```
(the gemma4_a4b_d_p 1x4 call, `models/demos/gemma4_a4b_d_p/tt/experts.py`). Tests:
`ttnn/ttnn/bringup/dispatch/tests/` (`test_dispatch.py`, `cases.py`, `source.yaml`).

## Used by
- ernie45_d_p (`tt/moe_unified.py`, 1x4, dispatch group 1; forked for P3.1)
- gemma4_a4b_d_p, mimo_v2_6_d_p (`tt/experts.py`, 1x4, group 1)
- mimo_v2_6_d_p_2x2, glm53_flash_d_p, hy4_preview_d_p (`tt/experts.py`, 2x2, group 2, axis 0, fabric)
- xing40_a4b_d_p (`tt/experts.py`, 4x2, group 4, axis 0, fabric)
- deepseek_v41_flash_d_p (`tt/experts.py` `_dispatch`, axis 0, ROW_MAJOR x | routing pieces)

## History
`CHANGELOG.md`.
