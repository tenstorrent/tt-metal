# Collective Communication Library

The Tenstorrent Metalium Collective Communication Library is a collection of high-level multi-chip
data movement operations. These data movement operations work with a variety of topologies and
configurations as well as deployment scenarios (N300, T3000, T7000, T7000 cluster).

In this context, a configuration is a combination of memory layout, data types, kernel args (e.g. dim for
all-reduce), allocation

# Supported Operations


## All Gather

### Configurations
For the time being, input and output configurations are expected to match

Tested Configurations:
* Layouts: {Row-Major, Tile}
* Memory Allocation: {Interleaved, Sharded (width)}
* Memory Locations: DRAM, L1
* Datatypes: BFloat16, Bfloat8_b

Unsupported Configurations:
* Sharding: {Block, Height}
* Row-Major + Bfloat8_b
  * Bfloat8_b must be in tile format

### Topologies
Supported Topologies:
* Ring

Future Tologies:
* Linear
* Mesh
* Torus (2d, 3d)

# Future Operations
* All Reduce
* Reduce
* Scatter
* Gather Scatter
* Scatter
* Point-to-Point Send/Receive
