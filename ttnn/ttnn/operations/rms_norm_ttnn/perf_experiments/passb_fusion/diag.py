"""passb_fusion — correctness diagnostic: WHERE does the fused pass B differ?

Single core, one tile-row, 8 width tiles: the focus geometry without the combine.
Prints the fused-vs-baseline error laid out by (tile, face-row, face-col) so a
broadcast/index bug shows as a structured block rather than a number.
"""

import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

from pathlib import Path

import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config

HERE = Path(__file__).resolve().parent
_ML = ttnn.TensorMemoryLayout


def _cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def run_one(device, kdir, shape, shard, grid, gamma_t, x_t):
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    PD.KERNEL_DIR = kdir
    mc = shard_config(shard, grid, _ML.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    kw = {"epsilon": 1e-12, "compute_kernel_config": _cfg(), "memory_config": x.memory_config()}
    if gamma_t is not None:
        kw["weight"] = ttnn.from_torch(gamma_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm_ttnn(x, **kw)
    got = ttnn.to_torch(out).float()
    return got


def main():
    import torch  # function-local: ttnn/ttnn may not import torch globally (pre-commit)

    W = int(os.environ.get("DIAG_W", "256"))
    H = int(os.environ.get("DIAG_H", "32"))
    mode = os.environ.get("DIAG_GAMMA", "rand")  # rand | ones | ramp | none
    shape = (1, 1, H, W)
    torch.manual_seed(0)
    x_t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    if mode == "none":
        g_t = None
    elif mode == "ones":
        g_t = torch.ones(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
    elif mode == "global":
        g_t = (torch.arange(W, dtype=torch.float32) + 1.0).reshape(1, 1, 1, W).to(torch.bfloat16)
    elif mode == "ramp":
        g_t = (torch.arange(W, dtype=torch.float32) % 32 + 1).reshape(1, 1, 1, W).to(torch.bfloat16)
    else:
        g_t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)

    sw = int(os.environ.get("DIAG_SHARD_W", str(W)))
    gx = int(os.environ.get("DIAG_GRID_X", "1"))
    gy = int(os.environ.get("DIAG_GRID_Y", "1"))
    kd = os.environ.get("DIAG_VARIANT", "k_fuse")
    blk = os.environ.get("DIAG_BLK", "")
    saved_pc = PD._PC_NONE
    if blk:
        PD._PC_NONE = PD._PC_NONE._replace(subblock_w=int(blk))
    device = ttnn.open_device(device_id=0)
    saved = PD.KERNEL_DIR
    try:
        ref = torch_rms_norm_ttnn(x_t.float(), epsilon=1e-12, weight=None if g_t is None else g_t.float())
        base = run_one(device, HERE / "k_base", shape, [H, sw], (gx, gy), g_t, x_t)
        fuse = run_one(device, HERE / kd, shape, [H, sw], (gx, gy), g_t, x_t)
    finally:
        PD.KERNEL_DIR = saved
        PD._PC_NONE = saved_pc
        ttnn.close_device(device)

    def rel(a):
        return float(((a - ref).pow(2).mean().sqrt()) / (ref.pow(2).mean().sqrt() + 1e-30))

    print(f"RESULT shape={shape} gamma={mode} blk={blk} relrms(base)={rel(base):.6f} relrms(fuse)={rel(fuse):.6f}")
    b = base[0, 0]
    f = fuse[0, 0]
    err = (f - b).abs() / (b.abs() + 1e-6)
    # per 16x16 face block: max relative error
    print("RESULT per-face max-rel-err (rows = face-row, cols = 16-wide face across W):")
    for fr in range(0, H, 16):
        row = []
        for fc in range(0, min(W, 256), 16):
            row.append(err[fr : fr + 16, fc : fc + 16].max().item())
        print("RESULT   fr%-3d " % fr + " ".join(f"{v:7.3f}" for v in row))
    print("RESULT per-SHARD (core) relrms fuse-vs-base:")
    for i in range(0, W, sw):
        d = f[:, i : i + sw] - b[:, i : i + sw]
        n = b[:, i : i + sw]
        v = float(d.pow(2).mean().sqrt() / (n.pow(2).mean().sqrt() + 1e-30))
        print(f"RESULT   shard {i//sw:3d} cols[{i}:{i+sw}] relrms={v:.6f}")
    # where exactly does it differ?  print the ratio f/b over the bad columns,
    # for one row of each face-row.
    bad = (err.max(dim=0).values > 1e-3).nonzero().flatten().tolist()
    print(f"RESULT bad columns (max-rel-err > 1e-3): n={len(bad)} first={bad[:8]} last={bad[-8:] if bad else []}")
    badr = (err.max(dim=1).values > 1e-3).nonzero().flatten().tolist()
    print(f"RESULT bad rows: n={len(badr)} {badr[:40]}")
    if bad:
        c0 = bad[0]
        for r in (0, 1, 15, 16, 17, 31):
            if r < H:
                print(
                    f"RESULT r{r:3d} ratio f/b cols[{c0}:{c0+16}] "
                    + " ".join(f"{(f[r, c] / (b[r, c] + 1e-12)).item():7.3f}" for c in range(c0, min(c0 + 16, W)))
                )
