import importlib
import os
import sys

import ttnn

torch = importlib.import_module("torch")
sys.path.insert(0, "ttnn/ttnn/operations/layer_norm_tile/perf_experiments/affine_mcast")
import affine_mcast as am
from ttnn.operations.layer_norm_tile import layer_norm_tile as op
from eval.golden_tests.layer_norm_tile.helpers import make_affine, pytorch_layer_norm

dev = ttnn.open_device(device_id=0)
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
orig = op.create_program_descriptor


def run(shape, layout, abuf, variant):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    r = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16()
    b = torch.randn(shape[-1]).bfloat16()
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tr = ttnn.from_torch(
        r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tg = make_affine(g, dev, dtype=ttnn.bfloat16, layout=layout, buffer_type=abuf)
    tb = make_affine(b, dev, dtype=ttnn.bfloat16, layout=layout, buffer_type=abuf)
    op.create_program_descriptor = orig if variant == "baseline" else am._make_create_compact(True, "last")
    out = ttnn.to_torch(op.layer_norm(tx, tg, tb, tr, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    return out, pytorch_layer_norm(x, g, b, r)


for shape, layout, abuf in [
    ((8, 1, 512, 768), ttnn.ROW_MAJOR_LAYOUT, ttnn.BufferType.L1),
    ((8, 1, 512, 768), ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM),
    ((3, 1, 500, 768), ttnn.ROW_MAJOR_LAYOUT, ttnn.BufferType.DRAM),
]:
    a, ref = run(shape, layout, abuf, "baseline")
    c, _ = run(shape, layout, abuf, "compact_late")
    pcc = lambda u: torch.corrcoef(torch.stack([u.float().flatten(), ref.float().flatten()]))[0, 1].item()
    rel = lambda u: ((u.float() - ref.float()).pow(2).mean().sqrt() / ref.float().std()).item()
    print(
        f"EXACT {shape} {layout} {abuf}: bit_exact={torch.equal(a, c)} PCC base={pcc(a):.6f} cand={pcc(c):.6f} relRMS base={rel(a):.5f} cand={rel(c):.5f}"
    )
ttnn.close_device(dev)
