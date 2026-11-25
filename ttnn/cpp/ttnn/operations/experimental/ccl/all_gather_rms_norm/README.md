# all_gather_rms_norm

`ttnn.experimental.all_gather_rms_norm`: a fused all-gather + RMSNorm / LayerNorm over one mesh axis, as one program per
device on Blackhole.

Every device of a group along `cluster_axis` holds its `(..., M, W_local)` slice of the hidden dimension; every device
returns the full normalized `(..., M, G * W_local)` row, with `G` the group size. RMSNorm takes an optional gamma;
LayerNorm (`norm_type="layer_norm"`) takes gamma and an optional bias (`beta`). The statistics are exchanged first (one
small packet per row around the group), each device normalizes only its own slice, and the normalized slices go
straight into the peers' final output pages, so the gathered tensor is never materialized before the norm.

The call replaces the chain the MLA layer runs today: `all_gather` of the reduce-scattered activation, then `rms_norm`
(the q_a latent) or `layer_norm` (the indexer k).

## What it was generated on

The op was generated, not hand-written, by the `tt_ops_code_gen` evaluation pipeline, then ported to a native C++ op:

- **Generation**: one run of the pipeline on 2026-11-23 (planner, implementer, verifier, a refinement queue: sharded
  inputs, transport, overlap and a LayerNorm critical path, then two perf rounds), starting from a single-chip `rms_norm`
  as the reference (the `--from-base` mode), with Claude Opus 5.5 as the agent. A later refinement added the L1 interleaved
  output the MLA q_a norm asks for. The generated op was a Python `ttnn.generic_op` program descriptor.
- **Base tree**: `malimpic/distributed_rms_norm_all_gather_gen` at `38dcbf23172` (the `llk_helper_library` lineage,
  where `rms_norm` and `layer_norm` are removed for the agents), kernel helper library of that lineage.
- **Test suite it was held to**: the all_gather_rms_norm golden suite of `tt_ops_code_gen` (`malimpic/agrn-layernorm-golden`;
  `d58b8179` for the generation run, `d347afba` with the output-placement axis for the L1 interleaved refinement): the cartesian of dtype, fp32 DEST, gamma / beta format, norm type, memory placement (DRAM
  interleaved, L1 interleaved or sharded output), topology, links and 39 per-device shapes, run through a tiered subset
  (smoke / gate / full).
- **Hardware**: a Blackhole QuietBox 2 (4 chips, p300c), opened as a 1x4 (and 4x1) mesh under `FABRIC_2D_TORUS_XY`, which
  closes the box's snake cycle into a 4-device ring (the size of a Galaxy TP axis). Galaxy was not available: the
  8-device rings and the Galaxy fabric are untested.
- **Native port**: the generic op was migrated to this C++ op by the `native_migrate` flow (5 repair rounds), checked
  against the generated op: golden outcomes no worse, device time within 5% + 0.2 us per case, host call time lower
  (a hot call of the focus case: 186 us against 3944 us for the Python route).

## What it was optimized for

The focus cases are the two real call sites of the DeepSeek-V3-family MLA layer at TP=4, per-device shapes:

| call site | per-device shape | norm | gamma / beta | measured (fastest chip, ring, 2 links) |
|---|---|---|---|---|
| MLA q_a latent (q_lora_rank 1536 / TP4), 640-row chunk | `(1, 1, 640, 384)` bf16 | RMSNorm | gamma bf16, ROW_MAJOR sticks | 28.4 us (DRAM output), 35.9 us (L1 interleaved output) |
| MLA indexer k (index_head_dim 128 / TP4), 640-row chunk | `(1, 1, 640, 32)` bf16 | LayerNorm | gamma + beta bf16 | 20.8 us |

Both use the model's compute config (HiFi2, 16-bit DEST). The other shapes in the golden suite (the Kimi, GLM and
Mistral q_a widths, 2048-row box chunks, the dense-FFN norms, decode and non-aligned shapes) were swept for correctness,
not tuned. Numbers were measured on the QuietBox 2 described above and depend on the device being otherwise idle.

## API

```python
out = ttnn.experimental.all_gather_rms_norm(
    input_tensor,            # TILE, float32 / bfloat16 / bfloat8_b, rank 2-4, DRAM interleaved (or L1 HEIGHT/WIDTH sharded)
    cluster_axis=1,
    gamma=None, beta=None,   # full-W weights: TILE (1,1,1,W) or ROW_MAJOR (1,1,ceil(W/32),32); beta: layer_norm only
    norm_type="rms_norm",    # or "layer_norm"
    epsilon=1e-6,
    topology=ttnn.Topology.Ring,
    num_links=2,
    compute_kernel_config=None,   # ttnn.ComputeConfigDescriptor; default HiFi4, fp32 DEST
    memory_config=None,           # output: DRAM interleaved (default), L1 interleaved, or a shard spec for a sharded input
    output_tensor=None,
)
```

Calls outside the supported contract raise `NotImplementedError` naming the unsupported value.

## Known limits

- **Precision**: LayerNorm without a bias, with bf16 input and 16-bit DEST, exceeded the relative-RMS limit (0.015) on
  two of the 39 shapes of the generation suite (`(1, 1, 1, 384)`: up to 0.017; `(1, 1, 2048, 1280)`: 0.0151). The same shapes pass with a bias,
  which is what the indexer uses.
- **L1 interleaved output** is about 26% slower than DRAM on the focus shape: with about 110 L1 banks a fabric packet
  carries four single-page writes (8 KiB) instead of about 14 KiB.
- **Sharded inputs** (L1 HEIGHT / WIDTH sharded) are supported and were validated in the generation suite, but the
  nightly test here covers interleaved placements only, the placements the model uses.
- **LayerNorm at 4096 rows per device** (indexer k, width 32) returns wrong values on this port (PCC 0.96); 640 to 3200 rows
  pass. Not investigated; the MLA call site uses 640 rows. RMSNorm passes at 4096 rows.
- **Row counts** other than the tested ones (32, 160, 640, 1024 to 3200, 2048, 4096 rows per device) are only partly tested.
- **Not tested**: Galaxy (8x4) rings, Wormhole, mesh groups larger than 4.

## Tests

`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_all_gather_rms_norm_bh.py`: the gate subset of the
generation suite, run by the "ccl nightly tests" job.
