"""Host-only NoC model of the bank-local tail write stream (no device): per core its bank class (pd.bank_layout on a
fake geometry), its NoC1 (writer) / NoC0 (reader) paths to the bank's endpoints, per-link loads, and a max-min
fair progressive-filling estimate of each core's finish time. usage: noc_model.py [nc_share] [noc1_order XY|YX]"""
import sys, types, collections

sys.path.insert(0, ".")
import importlib

pd = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul_program_descriptor")

PX = [1, 3, 4, 5, 6, 7, 10, 12, 13, 14, 15]  # logical x -> physical x (this board)
NX, NY = 17, 12
HOME = ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))
EP0 = {0: (0, 11), 1: (0, 2), 2: (0, 9), 3: (0, 5), 4: (9, 11), 5: (9, 3), 6: (9, 8), 7: (9, 6)}
EP1 = {0: (0, 1), 1: (0, 10), 2: (0, 4), 3: (0, 7), 4: (9, 1), 5: (9, 10), 6: (9, 4), 7: (9, 7)}


def layout(n_tiles=512, gx=11, gy=10):
    g = types.SimpleNamespace(tensor_n_tiles=n_tiles, b_banks=8, bank_home=HOME, swiglu=False)
    return pd.bank_layout(g, gx, gy, "row")


def path(src, dst, noc, order):
    """links ((x,y),dir) of a dimension-ordered route on the torus. noc0: +x,+y (X first); noc1: -x,-y."""
    (x, y), (dx, dy) = src, dst
    step = 1 if noc == 0 else -1
    links = []

    def walk_x():
        nonlocal x
        while x != dx:
            links.append(((x, y), "x"))
            x = (x + step) % NX

    def walk_y():
        nonlocal y
        while y != dy:
            links.append(((x, y), "y"))
            y = (y + step) % NY

    if noc == 0 or order == "XY":
        walk_x()
        walk_y()
    else:
        walk_y()
        walk_x()
    links.append(((dx, dy), "eject%d" % noc))
    return links


def model(nc_pct=36, order="YX", link_gbs=86.4, eject_gbs=None, tile=2048, rows=16, swap=False):
    lay = layout()
    flows = []  # (core, bytes, links)
    for (lx, ly), (u, t0, w) in lay.items():
        if not w:
            continue
        core = (PX[lx], ly + 2)
        n_nc = (w * nc_pct) // 100
        n_wr = w - n_nc
        flows.append((core, n_wr * rows * tile, path(core, EP1[u], 1, order), u))
        if n_nc:
            flows.append((core, n_nc * rows * tile, path(core, EP0[u], 0, order), u))
    cap = collections.defaultdict(lambda: link_gbs * 1e3)  # bytes/us
    if eject_gbs:
        for f in flows:
            cap[f[2][-1]] = eject_gbs * 1e3
    rem = [f[1] for f in flows]
    done = [None] * len(flows)
    t = 0.0
    active = set(range(len(flows)))
    while active:
        # max-min fair rates by progressive filling
        rate = {}
        capl = {}
        users = collections.defaultdict(set)
        for i in active:
            for l in flows[i][2]:
                users[l].add(i)
        capl = {l: cap[l] for l in users}
        unfixed = set(active)
        while unfixed:
            best = min((capl[l] / len(users[l] & unfixed), l) for l in users if users[l] & unfixed)
            r, l = best
            for i in list(users[l] & unfixed):
                rate[i] = r
                unfixed.discard(i)
                for l2 in flows[i][2]:
                    capl[l2] -= r
        dt = min(rem[i] / rate[i] for i in active)
        t += dt
        for i in list(active):
            rem[i] -= rate[i] * dt
            if rem[i] <= 1e-6:
                done[i] = t
                active.discard(i)
    end = collections.defaultdict(float)
    for f, d in zip(flows, done):
        end[f[0]] = max(end[f[0]], d)
    return end, flows


if __name__ == "__main__":
    pct = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    order = sys.argv[2] if len(sys.argv) > 2 else "YX"
    eject = float(sys.argv[3]) if len(sys.argv) > 3 else None
    end, flows = model(pct, order, eject_gbs=eject)
    print(f"model nc_pct={pct} noc1={order} eject={eject}: max {max(end.values()):.1f} us")
    for y in range(2, 12):
        print(f"{y:3d} " + " ".join(f"{end[(x, y)]:5.1f}" for x in PX))
