"""Per-core tail-write policy for the bank-local split (host only): for each core of class u, the NoC1 (writer) and
NoC0 (reader) endpoint among the bank's NoC0 / NoC1 endpoints with the fewest hops on that NoC, and the reader's
tile share. Physical model of THIS board (noc_model.py). usage: policy.py [P1|P2|P3]"""
import sys

sys.path.insert(0, sys.path[0])
from noc_model import layout, path, PX, EP0, EP1


def hops(src, dst, noc):
    return len(path(src, dst, noc, "YX")) - 1


def policy(kind="P2", nc_pct=36, n_tiles=512, lay=None):
    """lay: {(lx, ly): (u, t0, w)} (default: the te_gate layout)."""
    out = {}
    for (lx, ly), (u, t0, w) in (lay or layout(n_tiles)).items():
        if not w:
            continue
        c = (PX[lx], ly + 2)
        h1 = {e: hops(c, ep[u], 1) for e, ep in ((0, EP0), (1, EP1))}
        h0 = {e: hops(c, ep[u], 0) for e, ep in ((0, EP0), (1, EP1))}
        wr_ep = min(h1, key=lambda e: (h1[e], e != 1))
        rd_ep = min(h0, key=lambda e: (h0[e], e != 0))
        H1, H0 = h1[wr_ep], h0[rd_ep]
        if kind[:2] == "P1":
            nc = (w * nc_pct) // 100
        elif kind[:2] == "P2":
            nc = w if H0 < H1 else (0 if H1 < H0 else (w * nc_pct) // 100)
        else:  # P3: tiles inversely proportional to the path length
            nc = round(w * H1 / (H0 + H1)) if H0 + H1 else (w * nc_pct) // 100
        for mod in kind[2:].split("_"):  # debugging modifiers: r0 / r1 / w0 / w1 force an endpoint, nN force nc
            if mod[:1] == "r":
                rd_ep = int(mod[1:])
            elif mod[:1] == "w":
                wr_ep = int(mod[1:])
            elif mod[:1] == "n":
                nc = min(w, int(mod[1:]))
        out[(lx, ly)] = dict(u=u, w=w, nc=nc, wr_ep=wr_ep, rd_ep=rd_ep, H1=H1, H0=H0)
    return out


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "P2"
    pol = policy(kind)
    for ly in range(10):
        print(
            f"{ly + 2:3d} "
            + " ".join(
                f"{p['u']}:{p['nc']}/{p['w']} w{p['wr_ep']}r{p['rd_ep']} {p['H1']:2d}|{p['H0']:2d}"
                if (p := pol.get((lx, ly)))
                else "-" * 16
                for lx in range(11)
            )
        )
