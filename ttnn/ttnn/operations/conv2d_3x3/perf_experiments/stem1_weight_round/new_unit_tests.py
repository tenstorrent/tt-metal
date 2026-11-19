# The unit tests the graduation adds to tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_height_sharded.py
# (unit_tests.patch), appended there verbatim; test_wr_unit.py runs them against op_grad/ before graduation.


# Slab-direct on the whole grid: no spare core, so no weight servers, and round 0 comes over the sender reader's NoC0
# multicast (SD_R0_NOC0) while any later round rides the writer channel. ((H, W, C_in), C_out, padding, stride, dilation,
# forced M-block tile-rows or 0): the op's own blocking (one M-block per core, the channel's only round on NoC0), and two
# M-blocks per core (forced; round 1 over the channel after the NoC0 round 0), streamed and not streamed.
SLAB_DIRECT_GRID_SHAPES = [
    ((256, 512, 8), 64, 1, 2, 1, 0),
    ((256, 512, 8), 64, 1, 2, 1, 5),
    ((128, 256, 8), 64, 1, 2, 1, 1),
]


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    "case",
    SLAB_DIRECT_GRID_SHAPES,
    ids=[f"H{c[0][0]}_W{c[0][1]}_Ci{c[0][2]}_Co{c[1]}_s{c[3]}_bm{c[5]}" for c in SLAB_DIRECT_GRID_SHAPES],
)
def test_conv2d_3x3_height_sharded_slab_direct_whole_grid(case, with_bias, device, monkeypatch):
    torch.manual_seed(13)
    *case_hw, force_bm = case
    H, W, c_in = case_hw[0]
    grid = device.compute_with_storage_grid_size()
    shard_rows = -(-H * W // (grid.x * grid.y))
    config = _row_major_config(device, H * W, c_in, ttnn.ROW_MAJOR_LAYOUT, shard_rows)
    if config.shard_spec.grid.num_cores() != grid.x * grid.y:
        pytest.skip("the input shards do not cover the whole grid")
    real_select = _pd.select_blocking
    real_servers = _pd._weight_servers
    seen = []

    def blocks(g, grid_, l1_budget):
        b = real_select(g, grid_, l1_budget)
        seen.append((g, b))
        if not force_bm:
            return b
        assert b.core_m_tiles % force_bm == 0, (b.core_m_tiles, force_bm)
        fit = _pd._footprint(g, b.core_m_tiles, force_bm, b.block_k_tiles, b.block_n_tiles)
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=force_bm,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    servers = []

    def recording_servers(*args, **kwargs):
        out = real_servers(*args, **kwargs)
        servers.append(out)
        return out

    monkeypatch.setattr(_op_module, "select_blocking", blocks)
    monkeypatch.setattr(_pd, "_weight_servers", recording_servers)
    _run(
        tuple(case_hw),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        config,
        device,
        with_bias=with_bias,
        relu=True,
        dispatches=2,
    )
    g, b = seen[0]
    assert _pd.slab_direct_capable(g, b.block_k_tiles)
    assert servers and all(s is None for s in servers), "the whole grid holds input shards: no weight server"
