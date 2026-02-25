import ttnn
import torch

device = ttnn.open_device(device_id=0)

# Create block sharded input
input_torch = torch.randn(1, 1024, 160, dtype=torch.bfloat16)
block_sharded_config = ttnn.create_sharded_memory_config(
    shape=(1, 1024, 160),
    core_grid=ttnn.CoreGrid(x=5, y=8),
    strategy=ttnn.ShardStrategy.BLOCK,
    use_height_and_width_as_shard_shape=False,
)
input_ttnn = ttnn.from_torch(
    input_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=block_sharded_config,
)

# Request DRAM output - but get invalid DRAM+BLOCK_SHARDED
output_dram = ttnn.mean(input_ttnn, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print(f"Output memory config: {output_dram.memory_config()}")
# Output: memory_layout=TensorMemoryLayout::BLOCK_SHARDED, buffer_type=BufferType::DRAM  <-- INVALID!

ttnn.close_device(device)
