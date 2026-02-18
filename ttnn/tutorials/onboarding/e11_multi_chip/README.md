# E11: Multi-Chip

Distribute computation across multiple chips.

## Goal

Learn multi-device programming:
- Set up device mesh
- Distribute tensors across chips
- Use collective communication (CCLs)

## Reference

- `tt_metal/programming_examples/distributed/`

## Key Concepts

### Device Mesh
- Create mesh of devices with `ttnn.open_mesh_device`
- Access individual devices by coordinate
- Mesh shape must match physical topology

### Tensor Distribution
- Shard tensors across devices
- Each device holds a portion of the data

### Collective Communications (CCLs)
- **All-Gather**: Collect shards from all devices
- **Reduce-Scatter**: Sum and distribute results
- **All-Reduce**: Sum across all devices

### Ethernet Communication
- Chips connect via Ethernet links
- CCLs use these links for data transfer

## Common Pitfalls

1. **Mesh shape mismatch** - Must match physical topology
2. **CCL dimension** - Gather/scatter dim must match sharding
3. **Memory pressure** - All-gather creates full copies
