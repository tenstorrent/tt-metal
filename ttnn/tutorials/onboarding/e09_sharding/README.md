# E09: Sharding

Understand sharded vs interleaved memory layouts for multi-core operations.

## Goal

Compare sharded and interleaved tensor memory layouts and their impact on multi-core performance.

## Key Concepts

- Interleaved: Tensor pages distributed round-robin across banks
- Sharded: Tensor partitions stored locally in each core's L1
- Height sharding, width sharding, block sharding
- Trade-offs between data locality and flexibility

## Reference

- `tt_metal/programming_examples/shard_data_rm/` - Sharding example

## Workflow

1. Implement the multi-core kernel with sharded inputs
2. Compare against interleaved version
3. Profile with Tracy
4. Document when to use each approach
