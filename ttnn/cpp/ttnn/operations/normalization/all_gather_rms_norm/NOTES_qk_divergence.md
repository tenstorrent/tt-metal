# all_gather_rms_norm — LTX accuracy investigation: the Q/K head-split path is the sole bad actor

**Status:** Block-norm fused path is a **byte-perfect** drop-in for the wan baseline. The model accuracy
regression when fusing comes **only from the Q/K (head-split, `num_heads > 1`) norms.** Holding this finding
here so we don't re-litigate it.

Context: replacing the wan `wan_fused_rmsnorm_pre/post_allgather` path in LTX-2
(`models/tt_dit/layers/normalization.py::DistributedRMSNorm.forward`) with the fused
`ttnn.all_gather_rms_norm`, on `bh_2x4sp1tp0` (mesh (2,4), sp_axis=1, tp_axis=0, TP=2). A/B'd via env:
`LTX_FUSED_AGRMS=1` fuses the block norms (`num_heads==1`); `LTX_FUSED_AGRMS_QK=1` additionally fuses the
Q/K norms (`num_heads==16`). Proxy metric: output `.mp4` file size (good ≈ 3.2 MB, garbage ≈ 12 MB).

## The two fixes/finds, in order

### 1. Block-norm accumulation order (FIXED → bit-identical to wan)
The fused op and wan compute the same RMS-norm math but **accumulated x² in a different order**, and float
add is non-associative, so the fp32 sum differed by ~1 ulp and rounded to an adjacent bf16 value on ~0.1%
of elements (per-norm diff: max 0.0625, mean 5e-5, PCC 1.0). Root cause:
- **wan** (`fused_distributed_rmsnorm/.../compute/rmsnorm_pre_allgather.cpp`): computes x² per column-tile
  and **L1-accumulates all Wt tiles element-wise onto a single tile** (`pack_tile<true>` +
  `llk_pack_reconfig_l1_acc(1)`), *then* one `REDUCE_ROW`. → sums **across tiles first**, then columns.
- **ours** (`all_gather_rms_norm/.../compute/all_gather_rms_norm_compute.cpp::reduce_x2`, before the fix):
  stored the Wt x² tiles separately and let the reduce library sum **per-tile columns first, then tiles**.

Ruled out as *not* the difference: `use_legacy_rsqrt` (both `false`); scaler placement (wan scales
`1/total_W` in *post*, we scale in *pre*) — **bit-identical** because `total_W` is a power of two
(4096, 128) so the multiply is an exact exponent shift.

**Fix:** rewrote `reduce_x2` to mirror wan exactly (L1-accumulate x² into one tile, then one `REDUCE_ROW`).
Result on the model (`DUMP_NORM_DIR` dump, idx 0–11): **12/12 norm outputs bit-identical to wan**
(`max = 0.00000`, in and out), block *and* Q/K alike.

### 2. Bisect: only the Q/K path corrupts the model
Same prompt/seed, byte counts of the produced video:

| Run | env | video bytes |
|---|---|---|
| wan baseline | (none) | 3,220,993 |
| **block-fused, Q/K = wan** | `LTX_FUSED_AGRMS=1` | **3,220,993** ← byte-identical to wan |
| both fused | `LTX_FUSED_AGRMS=1 LTX_FUSED_AGRMS_QK=1` | 12,383,333 |

**Conclusion (the thing to hold): the block-norm fused path is a byte-perfect drop-in; the entire
divergence comes from the Q/K head-split fused path alone.**

## What is (and isn't) different about the Q/K norms
- **Reduction math is identical** to the block norm: `reduce_factor = shape[-1] * ring_size` (full gathered
  width) regardless of `num_heads`; wan pools all heads into one E[x²] too. So Q/K is *not* a per-head
  reduction — `num_heads` is layout-only.
- The **only** Q/K-specific code is the **writer's head-split output scatter**
  (`all_gather_rms_norm_writer.cpp`): `out_id = h*per_head_stride + gr*head_dim_tiles + e`, with
  `per_head_stride = m_tiles*head_dim_tiles`, `head_dim_tiles = Wt/num_heads`. For `num_heads==1` this
  collapses to a contiguous write (the block path). `max out_id = num_heads*m_tiles*head_dim_tiles - 1` =
  output tile count − 1 → **in-bounds** for the shapes checked.

## Open thread (being narrowed)
A full per-norm fingerprint scan (`DUMP_NORM_FP`, fp64 sum/sumsq per norm, wan vs both-fused) shows **every
norm output bit-identical through idx 0–280** (~10 denoise steps; the run timed out there). So the Q/K
norm *output tensors* are correct early. Reconciling "Q/K is the sole cause" with "Q/K outputs match":
1. a Q/K norm in a **late** step (idx > 280) diverges, **or**
2. the Q/K op writes the correct output but has a **side-effect** (out-of-output writes / semaphores /
   fabric) that corrupts a downstream non-norm tensor — invisible to a norm-output fingerprint.

Next: full-coverage scan (raised pytest `--timeout`) to find the first divergent norm; if zero norms ever
diverge yet the video is 12 MB, it's conclusively a side-effect → instrument the Q/K writer's address range.

## How to reproduce
```bash
# block-only (matches wan byte-for-byte):
LTX_FUSED_AGRMS=1 SEED=0 OUTPUT_PATH=/tmp/blockonly.mp4 \
  pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_fast_av.py -k bh_2x4sp1tp0 -s
# both (garbage video, isolates Q/K):
LTX_FUSED_AGRMS=1 LTX_FUSED_AGRMS_QK=1 SEED=0 OUTPUT_PATH=/tmp/both.mp4 \
  pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_fast_av.py -k bh_2x4sp1tp0 -s
# per-norm fingerprint (compare wan vs fused, find first divergence):
DUMP_NORM_FP=/tmp/fp.txt ...  # + raise --timeout; diff fp_wan.txt vs fp_fused.txt by (idx,tag)
```
`tt-smi -r` before runs; the first JIT run may time out (re-run uses the cache).
