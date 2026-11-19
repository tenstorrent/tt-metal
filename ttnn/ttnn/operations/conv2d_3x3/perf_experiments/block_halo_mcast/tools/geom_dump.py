# Probe body (run through scripts/tt-probe.sh): build the given focus / block cases through op_base with the program
# descriptor wrapped, and dump geometry, blocking and per-core TILE slab bands (input tile-rows) as JSON lines.
import json, os, sys
import ttnn

sys.path.insert(0, "/localdev/pmilojevic/workspaces/opgen")
from ttnn.operations.conv2d_3x3.perf_experiments.block_halo_mcast import test_bhm as T

op, pd = T._MODS["base"]
cases = os.environ.get("GEOM_CASES", "aspp_convs_1").split(",")
out_path = "/localdev/pmilojevic/workspaces/opgen/ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_halo_mcast/runs/geom.jsonl"
device = ttnn.open_device(device_id=0)
try:
    for case in cases:
        rec = {}
        real = op.create_program_descriptor

        def wrapped(*a, geometry, blocking, **kw):
            g, b = geometry, blocking
            rec.update(
                case=case,
                m_ranges=pd.core_m_ranges(g, b.core_m_tiles),
                bm=b.block_m_tiles,
                slab_tile_rows=b.slab_tile_rows,
                ub=b.untilize_batch_rows,
                k_cols=g.k_cols,
                c_in_tiles=g.c_in_tiles,
                H=g.input_height,
                W=g.input_width,
                d=g.dilation,
                p=g.padding,
                stride=g.stride,
                in0_tile_bytes=g.in0_tile_bytes,
                remote=g.remote_output,
                block_rows=g.block_rows,
                groups=b.num_m_groups,
                footprint=b.footprint_bytes,
                shared_slab_bytes=b.shared_slab_bytes,
            )
            bands = []
            for start, actual in rec["m_ranges"]:
                o_first = start * pd.TILE
                o_last = min((start + actual) * pd.TILE, g.num_out_px) - 1
                band_len = pd._block_band_len(g, o_first, o_last)
                f = pd._block_first_tap(g, o_first)
                bl = []
                for kh in range(3):
                    v_lo = f + kh * g.dilation * g.input_width
                    lo, hi = max(v_lo, 0), min(v_lo + band_len, g.num_in_px)
                    bl.append([lo // pd.TILE, pd._div_up(hi, pd.TILE)] if lo < hi else None)
                bands.append(bl)
            rec["bands"] = bands
            rec["l1_budget"] = pd.l1_cb_budget_bytes(0)
            return real(*a, geometry=geometry, blocking=blocking, **kw)

        op.create_program_descriptor = wrapped
        T.op, T.pd = op, pd
        try:
            if case.startswith("block:"):
                T._run_block(case.split(":", 1)[1], device)
            else:
                T._run_focus(T.FOCUS[case], device)
        finally:
            op.create_program_descriptor = real
        print("GEOM", json.dumps(rec))
        open(out_path, "a").write(json.dumps(rec) + "\n")
finally:
    ttnn.close_device(device)
