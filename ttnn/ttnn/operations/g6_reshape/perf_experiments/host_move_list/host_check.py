# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device) check of the move lists: emulate reads / zero-copy writes / copies / image writes
byte for byte on tilized pages and compare with torch.reshape. Also prints RT-word statistics.
Usage: python3 host_check.py"""

import sys
from pathlib import Path

torch = __import__("torch")  # experiment bench: global torch import is fine here (not part of ttnn)

sys.path.insert(0, str(Path(__file__).parent))
import hml  # noqa: E402

TILE, FACE = 32, 16


def tilize_pages(t, elem):
    shape = t.shape
    W = shape[-1]
    H = shape[-2] if len(shape) >= 2 else 1
    B = int(torch.tensor(shape[:-2]).prod()) if len(shape) > 2 else 1
    Ht, Wt = -(-H // TILE), -(-W // TILE)
    x = torch.zeros(B, Ht * TILE, Wt * TILE, dtype=t.dtype)
    x[:, :H, :W] = t.reshape(B, H, W)
    x = x.reshape(B, Ht, 2, FACE, Wt, 2, FACE).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    raw = x.reshape(-1).view(torch.uint8)
    return bytearray(raw.numpy().tobytes()), (B, H, W, Ht, Wt)


def untilize(buf, dims, dtype, out_shape):
    B, H, W, Ht, Wt = dims
    x = torch.frombuffer(bytes(buf), dtype=torch.uint8).view(dtype).reshape(B, Ht, Wt, 2, 2, FACE, FACE)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, Ht * TILE, Wt * TILE)[:, :H, :W]
    return x.reshape(out_shape)


def emulate(in_shape, out_shape, dtype, variant):
    elem = torch.tensor([], dtype=dtype).element_size()
    t = torch.randn(in_shape).to(dtype)
    src, _ = tilize_pages(t, elem)
    _, ddims = tilize_pages(torch.zeros(out_shape, dtype=dtype), elem)
    dst = bytearray(len(_))
    page = TILE * TILE * elem

    class G:
        x, y = 13, 10

    v = hml.VARIANTS[variant]
    plan = hml.BASE._make_plan(tuple(in_shape), tuple(out_shape), elem, G, 64)
    nmov = len(plan.roles)
    for ci in range(plan.num_cores):
        per = []
        for m in range(nmov):
            _, b, c = plan.roles[m][ci]
            per.append([p for a in range(b, b + c) for p in hml._atom_pieces(plan, a)])
        if v.split:
            per = hml._split(per)
        for m in range(nmov):
            pr = hml._mover_prog(per[m], v.gap, v.policy)
            st = bytearray(pr.stage_bytes + 128)
            im = bytearray(pr.image_bytes + 128)
            for pg, off, ln, so in pr.reads:
                assert (so - off) % 64 == 0
                st[so : so + ln] = src[pg * page + off : pg * page + off + ln]
            for pg, off, ln, so in pr.zc:
                assert (so - off) % 16 == 0
                dst[pg * page + off : pg * page + off + ln] = st[so : so + ln]
            for so, io, ln in pr.copies:
                im[io : io + ln] = st[so : so + ln]
            for pg, off, ln, io in pr.imgw:
                assert (io - off) % 16 == 0
                dst[pg * page + off : pg * page + off + ln] = im[io : io + ln]
    got = untilize(dst, ddims, dtype, out_shape)
    exp = t.reshape(out_shape)
    ok = torch.equal(got.view(torch.uint8) if False else got, exp)
    return ok


CASES = {
    "q553": ((1, 1, 6, 256), (1, 1, 1536), torch.bfloat16),
    "q508": ((1, 8, 64), (8, 1, 64), torch.bfloat16),
    "q494": ((8, 1, 64), (1, 1, 8, 64), torch.bfloat16),
    "q490": ((1, 8, 6, 256), (1, 1, 8, 1536), torch.bfloat16),
    "f120": ((1, 256), (1, 1, 256, 1), torch.float32),
    "q349": ((1, 16, 6, 256), (1, 1, 16, 1536), torch.bfloat16),
}

if __name__ == "__main__":
    for name, (i, o, d) in CASES.items():
        for var in ("hml_bank", "hml_bank_split", "hml_bank_1read", "hml_bank_img", "hml_bank_zc", "hml_bank_zc_1read"):
            ok = emulate(i, o, d, var)
            st = hml.plan_stats(i, o, torch.tensor([], dtype=d).element_size(), var)
            print(f"{name:5s} {var:16s} exact={ok} {st}")
