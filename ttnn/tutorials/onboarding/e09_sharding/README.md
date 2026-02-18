# E07: Sharding

Understand sharded vs interleaved memory layouts.

## Goal

Learn sharded memory layouts and when to use them:
- Understand height, width, and block sharding
- Compare sharded vs interleaved memory
- Optimize data locality for multi-core operations

## Reference

- `tt_metal/programming_examples/shard_data_rm/`

## Key Concepts

### Sharding Types
- **Height sharding**: Rows distributed across cores
- **Width sharding**: Columns distributed across cores
- **Block sharding**: 2D blocks across core grid

### Interleaved vs Sharded
- **Interleaved**: Pages round-robin across DRAM banks, any core can access
- **Sharded**: Data in local L1, no remote reads needed

### Memory Config
- Use `ttnn.ShardSpec` to define shard shape and orientation
- Use `ttnn.MemoryConfig` with appropriate `TensorMemoryLayout`

## Common Pitfalls

1. **Shard size mismatch** - Dimensions must evenly divide
2. **L1 overflow** - Shards must fit in core's L1
3. **Wrong sharding type** - Choose based on access pattern
