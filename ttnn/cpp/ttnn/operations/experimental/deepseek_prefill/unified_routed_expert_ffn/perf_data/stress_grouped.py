import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttnn
from loguru import logger
from moe_bench_common import MultiExpertCase, open_dev
from test_grouped import DISTS

dist = sys.argv[1] if len(sys.argv) > 1 else "kimi_u"
dtype = sys.argv[2] if len(sys.argv) > 2 else "bf8"
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cfgs = [
    dict(G=0),
    dict(G=10, rows=10),
    dict(G=5, rows=10),
    dict(G=8, rows=8),
    dict(G=4, rows=8),
    dict(G=5, rows=10, mmax=8),
    dict(G=10, rows=10, mmax=8),
]
if os.environ.get("STRESS_CFGS"):
    import json

    cfgs = json.loads(os.environ["STRESS_CFGS"])
dev = open_dev()
model, counts = DISTS[dist]
case = MultiExpertCase(dev, counts, model, dtype_key=dtype, x_row_major=True)
for rd in range(rounds):
    for c in cfgs:
        ffn = dict(
            num_row_groups=c.get("G", 0),
            grid_rows=c.get("rows", 0),
            grid_cols=0,
            per_core_m_max=c.get("mmax", 0),
            weight_cb_depth=0,
            col_strided=0,
            down_split=int(os.environ.get("STRESS_DOWN_SPLIT", "1")),
            lpt_fixed_cost_tiles=0,
        )
        case.expert.ffn_kwargs = ffn
        t0 = time.time()
        sys.stdout.write(f"round {rd} cfg {c} ... ")
        sys.stdout.flush()
        for i in range(4):
            case.run()
        ttnn.synchronize_device(dev)
        res = case.check()
        ok = all(r["ok"] for r in res)
        print(
            f"{'OK' if ok else 'PCC-FAIL'} ({time.time()-t0:.1f}s)"
            + ("" if ok else "  " + str([(r["expert"], round(r["pcc"], 4), r["nan"]) for r in res]))
        )
        sys.stdout.flush()
ttnn.close_mesh_device(dev)
print("STRESS DONE")
