# ttnn.bringup.combine

## What it does
Fork of `ttnn.experimental.deepseek_prefill.combine` (source
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine` @ `67ca5f3af48`, synced to `bdc576fa652`): the inverse
of `dispatch`. For each valid row of each local expert's region in the expert-output buffer it reads the dispatch
metadata (origin chip, token, top-k slot) and writes the row to `out[token, topk_slot]` on the origin chip (NoC locally,
fabric remotely). Output `(1, 1, S, K, H)`; slots of experts outside this dispatch group stay zero (`init_zeros`) or
uninitialized. No routing weight is applied (the kernels have none, despite the docstring's "weighted"): the caller
does the weighted top-k sum.

This fork adds: a 1-device dispatch axis. With one device on the axis (or `num_links == 0`) all fabric setup is skipped
and the kernels' no-fabric build runs; the source TT_FATALs "No neighbors found". Multi-device axes run the source's
fabric path unchanged.

Everything else as `ttnn.experimental.deepseek_prefill.combine`.

## Consider it instead of
- `ttnn.experimental.deepseek_prefill.combine` when the dispatch group is a single chip. On a multi-chip axis the two
  are the same program.
- A per-expert `ttnn.experimental.deepseek_prefill.insert` / scatter back plus an all-to-all: combine sends only the
  routed rows back, bounded by the device-resident counts.

## Options
Positional: `dispatched_buffer` (1, 1, N, H) expert outputs (BFLOAT16, or BFLOAT8_B TILE as the model cases pass),
`dispatched_metadata` (1, 1, N, 3) INT32, `expert_token_counts`, `expert_region_offsets` (1, 1, E) INT32 / UINT32.
Keyword (required): `dispatch_group_size`, `experts_per_chip`, `num_experts_per_tok`, `seq_len_per_chip`.
Keyword (optional):
- `memory_config=None`: interleaved L1 or DRAM; defaults to the buffer's.
- `subdevice_id=None`.
- `cluster_axis=None`: combine mesh axis (docstring says default 0; binding default is None).
- `num_links=1`: fabric links; 0 also disables fabric (fork gate).
- `topology=Topology.Linear` (Linear or Ring).
- `init_zeros=True`: zero the output first.
- `use_l1_small_for_semaphores=False` (bound, not in the docstring).
- `use_fp8_combine=False`: fp8_e4m3 output (Blackhole).
No fork-only option: the fork change is automatic on a 1-device axis.

## Measured
- Accuracy: exact, including a BFLOAT8_B buffer (unpacked to bf16 losslessly) (`CHANGELOG.md`; every model case in
  `tests/cases.py` is `"exact": True`); reported exact on 2x2
  (`models/demos/mimo_v2_6_d_p_2x2/bringup/BREADCRUMBS.md` line 482, `models/demos/glm53_flash_d_p/bringup/BREADCRUMBS.md`
  line 2467).
- Speed: deepseek_v41_flash_d_p, 2x2, layer 0, chunk 2048, sync-instrumented step time: combine 1.4 ms
  (`models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md` lines 1638-1641). ernie45_d_p 1x4, 28 layers, one
  50k->55k chunk: router + dispatch + combine together 100 ms (`models/demos/ernie45_d_p/bringup/BREADCRUMBS.md`
  line 120). Not measured on its own otherwise.
- Source tests carried (`tests/source_baseline.json`): 117 / 117 pass on original and fork.

## Limits
- Output memory must be interleaved. fp8 output Blackhole only.
- Carried source tests cover 2x2 and 4x1 (ring and line) on a 4-chip box; 4x2 / 8x1 / 8x4 cases never run here
  (`tests/source.yaml`). The 1-device axis is covered only by the fork's model cases.
- The xing40 4x2 model case needs 8 chips. No model case for deepseek_v41_flash_d_p's call.

## How to call
```python
out = ttnn.bringup.combine(
    buf, meta, counts, region_offsets,
    dispatch_group_size=1, experts_per_chip=32, num_experts_per_tok=8, seq_len_per_chip=5120,
    cluster_axis=0, num_links=1, topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG, init_zeros=True,
)  # -> [1, 1, S, K, H]
```
(the gemma4_a4b_d_p 1x4 call, `models/demos/gemma4_a4b_d_p/tt/experts.py`). Tests:
`ttnn/ttnn/bringup/combine/tests/` (`test_combine.py`, `cases.py`, `source.yaml`).

## Used by
- ernie45_d_p (`tt/moe_unified.py`, 1x4, group 1; forked for P3.1)
- gemma4_a4b_d_p, mimo_v2_6_d_p (`tt/experts.py`, 1x4, group 1)
- mimo_v2_6_d_p_2x2, glm53_flash_d_p, hy4_preview_d_p (`tt/experts.py`, 2x2, group 2, axis 0, fabric)
- xing40_a4b_d_p (`tt/experts.py`, 4x2, group 4, axis 0, init_zeros)
- deepseek_v41_flash_d_p (`tt/experts.py` `_combine`, axis 0, init_zeros)

## History
`CHANGELOG.md`.
