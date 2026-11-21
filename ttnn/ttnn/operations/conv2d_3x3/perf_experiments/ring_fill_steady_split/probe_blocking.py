"""Host-side blocking / L1-slack probe for the ring cases (run through scripts/tt-probe.sh conv2d_3x3 < this file).
Prints each case's blocking with 0 / 1 / 2 extra staging chunks counted in the L1 solve."""


def main():
    import importlib

    import torch
    import ttnn

    E = "ttnn.operations.conv2d_3x3.perf_experiments.ring_fill_steady_split.op_copy."
    op = importlib.import_module(E + "conv2d_3x3")
    pd = importlib.import_module(E + "conv2d_3x3_program_descriptor")
    _BH = {
        33: (514, 1024, 128, 3),
        35: (1026, 2048, 128, 3),
        36: (514, 1024, 32, 128),
        37: (258, 512, 64, 256),
        38: (130, 256, 128, 512),
        39: (514, 1024, 64, 128),
        40: (258, 512, 128, 256),
        41: (1026, 2048, 32, 128),
        42: (514, 1024, 64, 256),
        43: (258, 512, 128, 512),
        44: (1026, 2048, 64, 128),
        45: (514, 1024, 128, 256),
        46: (1026, 2048, 64, 256),
        47: (514, 1024, 128, 512),
    }

    class Stop(Exception):
        pass

    got = {}

    def cap(*a, **k):
        got["b"] = k.get("blocking")
        got["g"] = k.get("geometry") if "geometry" in k else (a[1] if len(a) > 1 else None)
        raise Stop()

    op.create_program_descriptor = cap
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    orig_stage = pd._rm_stage_bytes
    for extra in (0, 1, 2):

        def stage(g, _e=extra):
            nstg, chunk_px = pd._rm_stage(g)
            return (nstg + _e) * (chunk_px + pd.RM_MAX_DRAM_BANKS - 1) * g.c_in_p * 2

        pd._rm_stage_bytes = stage
        for f in dir(pd):
            o = getattr(pd, f)
            if hasattr(o, "cache_clear"):
                o.cache_clear()
        for idx, (H, W, ci, co) in _BH.items():
            x = torch.zeros(1, 1, H * W, ci, dtype=torch.bfloat16)
            tt_in = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w = torch.zeros(co, ci, 3, 3, dtype=torch.bfloat16)
            b = torch.zeros(co, dtype=torch.bfloat16)
            prep = dict(
                stride=1,
                padding=(0, 0, 1, 1),
                dilation=1,
                input_dtype=ttnn.bfloat16,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weight_dtype=ttnn.bfloat16,
                input_memory_config=tt_in.memory_config(),
                device=device,
            )
            tw = op.prepare_conv2d_weights(w, **prep)
            tb = op.prepare_conv2d_bias(b, **prep)
            try:
                op.conv2d_3x3(
                    tt_in,
                    tw,
                    input_height=H,
                    input_width=W,
                    bias=tb,
                    padding=(0, 0, 1, 1),
                    stride=1,
                    compute_kernel_config=cfg,
                )
            except Stop:
                pass
            bl = got["b"]
            budget = pd.l1_cb_budget_bytes(0, 65536)
            print(
                f"extra={extra} idx={idx} {H}x{W} {ci}->{co} bm={bl.block_m_tiles} bk={bl.block_k_tiles} mblk={bl.max_m_blocks_per_core} "
                f"ring_px={bl.rm_ring_px} win_kb={bl.rm_ring_px*ci*2//1024} stage=({bl.rm_stage_n},{bl.rm_stage_px}) res={bl.weights_resident} "
                f"ng={bl.n_groups} khr={bl.khr_ring_rows} fp={bl.footprint_bytes} budget={budget} slack={budget-bl.footprint_bytes} depth={bl.im2col_rm_depth_rows}"
            )
            for t in (tt_in, tw, tb):
                ttnn.deallocate(t)
    ttnn.close_device(device)


main()
