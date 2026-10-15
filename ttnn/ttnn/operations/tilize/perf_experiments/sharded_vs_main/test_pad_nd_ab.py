# Perf-only: padded tilize from a 2-core ND-sharded input ([3,100,158] -> [3,128,160]). No asserts.
import ttnn

from ttnn.operations.tilize import tilize as new_tilize


def test_pad_nd(device):
    import torch  # not at module level under ttnn/ (repo hook)

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    in_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=ttnn.NdShardSpec([2, 64, 96], grid))
    out_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=ttnn.NdShardSpec([2, 96, 96], grid))
    x = ttnn.from_torch(
        torch.rand([3, 100, 158], dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_mem,
    )
    new_tilize(x, out_mem, output_padded_shape=[3, 128, 160], pad_value=0.0)
    ttnn.synchronize_device(device)
