# Host-only geometry dump for the focus ring: eth / port / compute physical coordinates and the landing row hops.
import os, importlib.util, sys
from pathlib import Path
import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache

HERE = Path("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention").resolve()
spec = importlib.util.spec_from_file_location("a2amm_lc_geo", HERE / "base" / "all_to_all_matmul_program_descriptor.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)
cache = _FabricMeshCache()
try:
    mesh = cache.get((1, 4), payload_bytes=7232)
    chips = pd._topology_plan(mesh, 1, 2)
    grid = mesh.compute_with_storage_grid_size()
    print("grid", grid.x, grid.y)
    for coord, ch in chips.items():
        eth_list, t2p = pd._chip_maps(mesh, coord)

        def phys(xy):
            v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
            return (t2p.get(v.x, v.x), v.y)

        print("chip", coord, "p", ch["p"], "route", ch["route"], "links", ch["links"])
        for (d, l), xy in sorted(ch["ports"].items()):
            peer = ch["next"] if d == 0 else ch["prev"]
            chan = pd._eth_channel(mesh, coord, peer, ch["links"][d][l])
            print(f"  dir {d} link {l}: eth chan {chan} phys {eth_list[chan]}  port logical {xy} phys {phys(xy)}")
        if coord == list(chips)[0]:
            cols = sorted({phys((x, 0))[0] for x in range(grid.x)})
            rows = sorted({phys((0, y))[1] for y in range(grid.y)})
            print("  logical x -> phys x", [phys((x, 0))[0] for x in range(grid.x)])
            print("  logical y -> phys y", [phys((0, y))[1] for y in range(grid.y)])
finally:
    cache.close()
