# The unit test the graduation adds (unit_tests.patch), exec'd into test_conv2d_3x3_height_sharded's namespace by
# test_r0_unit.py (R0_UNIT_NEW=1): the whole-grid slab-direct test on 64x128x8 -> 64 s2, one tile-row per core on 64
# cores, where round 0's NoC0 multicast lands before the late-starting receivers reach their slot reserve (no ack).
SLAB_DIRECT_GRID_SHAPES_NEW = [((64, 128, 8), 64, 1, 2, 1, 0)]


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    "case",
    SLAB_DIRECT_GRID_SHAPES_NEW,
    ids=[f"H{c[0][0]}_W{c[0][1]}_Ci{c[0][2]}_Co{c[1]}_s{c[3]}_bm{c[5]}" for c in SLAB_DIRECT_GRID_SHAPES_NEW],
)
def test_conv2d_3x3_height_sharded_slab_direct_whole_grid_new(case, with_bias, device, monkeypatch):
    test_conv2d_3x3_height_sharded_slab_direct_whole_grid(case, with_bias, device, monkeypatch)
