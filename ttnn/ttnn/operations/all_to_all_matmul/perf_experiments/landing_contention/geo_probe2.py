# Host-only: landing row hops (NoC1 north->west: eth column first, then the destination row westward) for the current
# head->core map vs an assignment that minimises them.
import importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache

HERE = Path("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention").resolve()
spec = importlib.util.spec_from_file_location("a2amm_lc_geo", HERE / "base" / "all_to_all_matmul_program_descriptor.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)
NX = 17
cache = _FabricMeshCache()
try:
    mesh = cache.get((1, 4), payload_bytes=7232)
    L = 2
    chips = pd._topology_plan(mesh, 1, L)
    grid = mesh.compute_with_storage_grid_size()
    port_cells = {xy for ch in chips.values() for xy in ch["ports"].values()}
    cells = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in port_cells]
    ethx, physx = {}, {}
    for coord, ch in chips.items():
        eth_list, t2p = pd._chip_maps(mesh, coord)
        for d in (0, 1):
            peer = ch["next"] if d == 0 else ch["prev"]
            for l in range(L):
                ethx[(ch["p"], d, l)] = eth_list[pd._eth_channel(mesh, coord, peer, ch["links"][d][l])][0]
        for c in cells:
            v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c))
            physx[(ch["p"], c)] = t2p.get(v.x, v.x)
    pos = {ch["p"]: ch for ch in chips.values()}
    for H, Hl in ((96, 24), (128, 32), (64, 16)):
        cost = np.zeros((H, len(cells)))
        for h in range(H):
            s, j = h // Hl, h % Hl
            for t in range(4):
                if t == s:
                    continue
                d = pos[s]["route"][t]
                ex = ethx[(t, 1 - d, j % L)]
                for ci, c in enumerate(cells):
                    cost[h, ci] += (ex - physx[(t, c)]) % NX
        assign = pd._split_heads(cells, H)
        cur = sum(cost[h, cells.index(c)] for c, hs in assign for h in hs)
        caps = [len(hs) for _, hs in assign] + [0] * (len(cells) - len(assign))
        # capacity copies: optimal cells keep the same per-core head counts (largest count on the cheapest cells)
        ncap = max(caps)
        cols = [ci for ci in range(len(cells)) for _ in range(ncap)]
        r, cidx = linear_sum_assignment(cost[:, cols])
        best = cost[r, cidx].sum()
        print(
            f"H={H}: cells {len(cells)} landing row hops per head per dest: current {cur/H/3:.2f}  optimal {best/H/3:.2f}"
        )
    print("ethx", ethx)
finally:
    cache.close()
