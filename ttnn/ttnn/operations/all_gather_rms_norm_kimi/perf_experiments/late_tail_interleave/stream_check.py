"""Host-only check of the per-chunk stream plan (no device): a fake ttnn module, then the copy's planning functions."""
import os, sys, types, importlib.util
from types import SimpleNamespace as NS

fake = types.ModuleType("ttnn")


class _N:
    def __getattr__(self, k):
        return _N()

    def __call__(self, *a, **k):
        return _N()


for n in ("NOC", "UnpackToDestMode", "TensorMemoryLayout", "BufferType"):
    setattr(fake, n, _N())
sys.modules["ttnn"] = fake
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pd", os.path.join(here, "op_copy", "all_gather_rms_norm_kimi_program_descriptor.py")
)
pd = importlib.util.module_from_spec(spec)
sys.modules["pd"] = pd
spec.loader.exec_module(pd)


def geo(Mt, block_h, Wl_t=12, G=4, L=2, chunk=6):
    nb = -(-Mt // block_h)
    g = pd.Geometry(
        mesh_device=None,
        G=G,
        L=L,
        rows=1,
        cols=G,
        Mt=Mt,
        Wl_t=Wl_t,
        Wt=G * Wl_t,
        block_w=Wl_t,
        block_h=block_h,
        nb=nb,
        last_h=Mt - (nb - 1) * block_h,
        chunk_tiles=chunk,
        tile_bytes=2048,
        max_payload=14336,
        gx=11,
        gy=10,
        placement={},
        compute=[],
        row_base=[],
    )
    return g


def show(g, p=1):
    for l in range(g.L):
        for d in (0, 1):
            ch = pd.stream_chunks(g, pd.port_stream(g, p, l, d))
            thr, incs = pd.stream_incs(g, pd.port_stream(g, p, l, d))
            tail = ch[-10:]
            print(
                f"p{p} l{l} d{d} n={len(ch)} incs={incs} tail:",
                " ".join(f"{c.q}.{c.m}c{c.c}{'*' if c.inc else ''}@{t}" for c, t in zip(tail, thr[-10:])),
            )
    w = pd.xg_ahead_table(g, p)
    a = [(w[i // 4] >> (8 * (i % 4))) & 0xFF for i in range(g.G * g.nb)]
    print("ahead:", a)
    # invariants: each block's chunks appear in order exactly once per stream; every remote block of chip p is in exactly one upstream stream
    for l in range(g.L):
        for d in (0, 1):
            ch = pd.stream_chunks(g, pd.port_stream(g, p, l, d))
            seen = {}
            for c in ch:
                assert seen.get((c.entry, c.m), -1) + 1 == c.c
                seen[(c.entry, c.m)] = c.c
            assert pd.stream_incs(g, pd.port_stream(g, p, l, d))[0][-1] == sum(x.inc for x in ch)
            assert all(seen[k] + 1 == g.nchunks(k[1]) for k in seen)


for args in [dict(Mt=20, block_h=1), dict(Mt=64, block_h=3)]:
    print(args, "ILV", pd.TAIL_INTERLEAVE_CHUNKS, "ROUND_INC", pd.TAIL_ROUND_INC)
    show(geo(**args))
