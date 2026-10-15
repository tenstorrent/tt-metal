# Perf-only A/B: generated tilize vs production ttnn.tilize on small L1-sharded cases (no asserts).
import pytest
import ttnn

from ttnn.operations.tilize import tilize as new_tilize

PROD = ttnn.tilize


@pytest.mark.parametrize("impl", ["prod", "new"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.uint8])
@pytest.mark.parametrize("shape, grid", [([1, 1, 64, 64], (2, 2)), ([1, 1, 256, 256], (4, 4))])
def test_ab(device, impl, dtype, shape, grid):
    import torch  # not at module level under ttnn/ (repo hook)

    n_y, n_x = grid
    H, W = shape[-2:]
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_x - 1, n_y - 1))})
    spec = ttnn.ShardSpec(cores, [H // n_y, W // n_x], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)
    t = torch.randint(0, 100, shape).to(torch.float32 if dtype != ttnn.bfloat16 else torch.bfloat16)
    if dtype == ttnn.uint8:
        t = t.to(torch.uint8)
    x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    fn = PROD if impl == "prod" else new_tilize
    for _ in range(3):
        y = fn(x, memory_config=mem)
    ttnn.synchronize_device(device)
