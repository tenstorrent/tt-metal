"""Per-tile-row / per-tile-col error map for repeated dispatches of one grouped config."""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttnn
from moe_bench_common import MultiExpertCase, open_dev, torch_expert_ref
from test_grouped import DISTS

dist, dtype = sys.argv[1], sys.argv[2]
G, rows, mmax = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
n = int(sys.argv[6]) if len(sys.argv) > 6 else 6
dev = open_dev()
model, counts = DISTS[dist]
case = MultiExpertCase(dev, counts, model, dtype_key=dtype, x_row_major=True)
case.expert.ffn_kwargs = dict(
    num_row_groups=G,
    grid_rows=rows,
    grid_cols=0,
    per_core_m_max=mmax,
    weight_cb_depth=int(os.environ.get("DEPTH", "0")),
    col_strided=0,
    down_split=int(os.environ.get("DS", "1")),
    lpt_fixed_cost_tiles=0,
)
refs = {e: torch_expert_ref(case.inputs[e], case.weights[e], case.act) for e, c in enumerate(counts) if c > 0}
R = rows // G
chunk_M = (mmax or 4) * R
for it in range(n):
    case.run()
    ttnn.synchronize_device(dev)
    out = ttnn.to_torch(case.out, mesh_composer=ttnn.ConcatMeshToTensor(case.out.device(), dim=0))
    bad = []
    for e, c in enumerate(counts):
        if c <= 0:
            continue
        got = out[case.offsets[e] : case.offsets[e] + c].float()
        ref = refs[e]
        err = (got - ref).abs()
        scale = ref.abs().max().item() + 1e-6
        rel = err / scale
        rows_bad = (rel.max(dim=1).values > 0.05).nonzero().flatten().tolist()
        if rows_bad:
            tile_rows = sorted(set(r // 32 for r in rows_bad))
            for tr in tile_rows:
                sub = rel[tr * 32 : (tr + 1) * 32]
                cols_bad = (sub.max(dim=0).values > 0.05).nonzero().flatten().tolist()
                tcols = sorted(set(cc // 32 for cc in cols_bad))
                nrows_bad = int((sub.max(dim=1).values > 0.05).sum())
                bad.append(
                    (
                        e,
                        tr,
                        tr // chunk_M,
                        (tr % chunk_M) // max(1, (mmax or 4)),
                        nrows_bad,
                        len(tcols),
                        tcols[:8],
                        (tcols[-1] if tcols else None),
                    )
                )
    print(f"dispatch {it}: {'clean' if not bad else ''}")
    for b in bad:
        print(
            f"   expert {b[0]:2d} tile-row {b[1]:2d} (chunk {b[2]}, row-in-group {b[3]}) bad_rows={b[4]:2d} bad_tile_cols={b[5]:3d} first={b[6]} last={b[7]}"
        )
    sys.stdout.flush()
ttnn.close_mesh_device(dev)
