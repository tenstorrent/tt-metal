## Repository Overview

This is the **tt-metal** repository, containing TT-Metalium (low-level programming model) and TT-NN (high-level neural network operations library) for Tenstorrent AI accelerators. The codebase implements tensor operations and kernels that run on Tenstorrent hardware, which uses a unique architecture based on Tensix cores with tile-based computing (32×32 tiles).

## Key Architecture Concepts

### Hardware Architecture
- **Tensix cores**: Each contains 5 RISC-V CPUs (2 data movement, 1 unpack, 1 math, 1 pack), 1.5MB SRAM (L1), NoC interfaces, FPU (matrix unit), and SFPU (vector unit)
- **Tile-based computing**: Native 32×32 element tiles for efficient matrix operations
- **Network-on-Chip (NoC)**: Two bidirectional NoCs for data movement between cores and DRAM

### Programming Model
- **Three kernel types per Tensix core**: Reader (data input via NoC0), Compute (FPU/SFPU operations), Writer (data output via NoC1)
- **Circular buffers**: Inter-kernel communication mechanism in SRAM
- **SPMD execution**: Single Program Multiple Data across cores (most common pattern)

Refer to `METALIUM_GUIDE.md` for detailed architecture explanations.

## Building the Code

### Standard Build
```bash
./build_metal.sh --debug
```

### Device-side kernels
Kernels build at runtime. DO NOT build when changing kernel code.

## Running Tests


### Testing rules

Always use pytest, avoid raw python calls with main functions
Use the defined device fixture, do not open the device yourself
Put all parameters of the test into pytest parametrizations (even if only one of them), avoid hard coding parameters inside of functions

Python environment is sourced with
`source python_env/bin/activate`

Run the tests with

`pytest <test_file_path.py>`

You should be attentive of device hangs when running tests, especially for new / just implemented operations and kernels
Watch the bash output at all times, if pytest console output stops, suspect a hang.
If a hang occurs, kill the process with

`pkill -9 -f pytest || true`

After that, reset the device with

`tt-smi -r`

## Code Organization

### Key Directories
- `tt_metal/`: Core TT-Metalium implementation (C++ low-level APIs, kernels, hardware abstraction)
- `ttnn/`: TT-NN high-level operations library (Python/C++ APIs)
- `models/`: Model implementations and demos
- `tests/`: Test suites (unit tests, sweep tests, integration tests)
- `tech_reports/`: Architecture documentation and optimization guides
- `tt_metal/programming_examples/`: Example kernels and operations

### Important Files
- `METALIUM_GUIDE.md`: Comprehensive architecture and programming guide
- `README.md`: High-level project overview
- `INSTALLING.md`: Installation instructions
- `build_metal.sh`: Main build script

## Kernel Development

### Kernel Types
1. **Reader kernels** (`tt_metal/kernels/dataflow/`): Read data from DRAM/other cores via NoC
2. **Compute kernels** (`tt_metal/kernels/compute/`): Execute FPU/SFPU operations
3. **Writer kernels** (`tt_metal/kernels/dataflow/`): Write results to DRAM/other cores

### Circular Buffers
Circular buffers (CBs): SRAM-based page queues for producer-consumer synchronization between kernels (Reader/Compute/Writer) within and across Tensix cores.
API calls:
- cb_reserve_back(cb_id, num_pages): Producer blocks until space available for writing num_pages
- cb_push_back(cb_id, num_pages): Producer publishes written num_pages to consumer
- cb_wait_front(cb_id, num_pages): Consumer blocks until num_pages available for reading
- cb_pop_front(cb_id, num_pages): Consumer frees processed num_pages
Typical pattern: cb_wait_front → process → cb_pop_front → cb_reserve_back → write → cb_push_back


### Compute APIs
- **Abstraction layer** for hardware compatibility across generations (Wormhole, Blackhole)

## Documentation and Resources. USE THESE PROACTIVELY.

### Documentation in tt-metal repo

**METALIUM_GUIDE.md**: Comprehensive architecture and programming guide covering hardware fundamentals (Tensix processors, tile-based operations, NoC), programming model (reader/compute/writer kernels, circular buffers), and multi-chip scaling.
Reference when learning core architecture or developing low-level kernels.

**tech_reports/tensor_accessor/tensor_accessor.md**: TensorAccessor utility
Reference for mapping logical tensor indices to physical memory locations across distributed banks
Reference when you need to understand how kernels access tensor data.

**tech_reports/tensor_layouts/tensor_layouts.md**: Explanation of tensor layouts (row-major vs tiled) and memory layouts (interleaved vs sharded) in TT-Metal.
Reference when dealing with tensor creation, configuring data organization, or understanding page-to-bank mapping.

**tech_reports/tensor_sharding/tensor_sharding.md**: Guide to dividing tensors across memory banks using four strategies (height, width, block, ND sharding) for performance optimization
Reference when you need to understand how tensors could be sharded between cores for better data locality.


## Creating New TTNN Operations

Two workflows are available for creating new TTNN operations:

### Standard C++ Workflow
For production operations requiring full C++ TTNN registration:
`ttnn/experimental/claude_ttnn_agents/references/ttnn-operation-workflow.md`

Pipeline: analyzer → planner → scaffolder → factory-builder → kernel-designer → kernel-writer

### Generic Op Workflow (Python-based)
For rapid prototyping using `ttnn.generic_op()` and ProgramDescriptor APIs:
`ttnn/experimental/claude_ttnn_agents/references/ttnn-generic-op-workflow.md`

Pipeline: analyzer → planner → (generic_op_builder || kernel-designer) → kernel-writer

Key difference: `generic_op_builder` and `kernel-designer` run in **parallel**, and `kernel-writer` waits for both to complete.

### Additional Resources
- `ttnn/experimental/claude_ttnn_agents/subagent_breakdown.md` - Detailed agent breakdown
