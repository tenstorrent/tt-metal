# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **tt-metal** repository, containing TT-Metalium (low-level programming model) and TT-NN (high-level neural network operations library) for Tenstorrent AI accelerators. The codebase implements tensor operations and kernels that run on Tenstorrent hardware, which uses a unique architecture based on Tensix cores with tile-based computing (32×32 tiles).

## Key Architecture Concepts

### Hardware Architecture
- **Tensix cores**: Each contains 5 RISC-V CPUs (2 data movement, 1 unpack, 1 math, 1 pack), 1.5MB SRAM (L1), NoC interfaces, FPU (matrix unit), and SFPU (vector unit)
- **Tile-based computing**: Native 32×32 element tiles for efficient matrix operations
- **No cache hierarchy**: Explicit SRAM management and DMA operations required
- **Network-on-Chip (NoC)**: Two bidirectional NoCs for data movement between cores and DRAM

### Programming Model
- **Three kernel types per Tensix core**: Reader (data input via NoC0), Compute (FPU/SFPU operations), Writer (data output via NoC1)
- **Circular buffers**: Inter-kernel communication mechanism in SRAM
- **SPMD execution**: Single Program Multiple Data across cores (most common pattern)
- **Fast dispatch**: Asynchronous command queuing (enabled by default, critical for performance)

Refer to `METALIUM_GUIDE.md` for detailed architecture explanations.

## Building the Code

### Standard Build
```bash
./build_metal.sh -b Debug
```

### Clean Build Artifacts
```bash
./build_metal.sh --clean
```

## Running Tests

### Python API Tests
```bash
pytest tests/python_api_testing/unit_testing/ -vvv
pytest tests/python_api_testing/sweep_tests/pytests/ -vvv
```

### TTNN Unit Tests
```bash
# Run a single test file
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py

# Run all tests in a directory
pytest tests/ttnn/unit_tests/operations/
```

## Environment Variables

### Essential Variables
```bash
export PYTHONPATH=/path/to/tt-metal  # Required for Python imports
export ARCH_NAME=wormhole_b0         # or grayskull, blackhole
```

### Debugging and Development
```bash
# Enable operation logging
export TT_LOGGER_TYPES=Op
export TT_LOGGER_LEVEL=DEBUG

# Make ops blocking with logging (useful for debugging)
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'
```

For kernel-level debugging (hangs, CB issues, device errors), use the `ttnn-riscv-debugger` agent.

## Development Workflow

### Virtual Environment Setup
```bash
# Create and activate Python environment
./create_venv.sh
source python_env/bin/activate
```

### Device Reset
If device hangs or misbehaves:
```bash
tt-smi -tr 0  # Reset device 0
```

## Debugging Guide

### Host-Side Debugging
```bash
# Build with debug symbols
./build_metal.sh -b Debug

!!!IMPORTANT!!! Unless Release builds are explicitly asked for, you should work ONLY with Debug builds.

# Run with GDB
gdb --args <generated_binary>
gdb --args python <python_file>
```

### Device-Side / Kernel Debugging

For kernel issues (hangs, incorrect results, CB deadlocks), use the `ttnn-riscv-debugger` agent.

**When to invoke**:
- Test hangs or times out
- Device errors or assertions
- Incorrect kernel output
- Suspected circular buffer issues

**How to invoke**: Provide the symptom and let the agent handle watcher/DPRINT:
```
Symptom: "test_avgpool2d hangs"
Test: "pytest tests/ttnn/.../test_avgpool2d.py::test_run_avg_pool2d"
Hint (optional): "bug likely in compute kernel"
```

The agent autonomously enables watcher, interprets core states, analyzes kernel code with grep/sed verification, forms hypotheses, and proposes verified fixes.

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
- `CONTRIBUTING.md`: Contribution guidelines and workflows
- `build_metal.sh`: Main build script

## Kernel Development

### Kernel Types
1. **Reader kernels** (`tt_metal/kernels/dataflow/`): Read data from DRAM/other cores via NoC
2. **Compute kernels** (`tt_metal/kernels/compute/`): Execute FPU/SFPU operations
3. **Writer kernels** (`tt_metal/kernels/dataflow/`): Write results to DRAM/other cores
**NOTE on "Reader" / "Writer" terminology**
"Reader" kernel refers to kernel running on RISCV_0 (BRISC) which reads data via NOC0. "Writer" kernel refers to kernel running on RISCV_1 (NCRISC) which writes data via NOC1. See "Reader and Writer Kernel Naming Convention" section below for details and caveats.

### Circular Buffers
Circular buffers (CBs): SRAM-based page queues for producer-consumer synchronization between kernels (Reader/Compute/Writer) within and across Tensix cores.
API calls:
- cb_reserve_back(cb_id, num_pages): Producer blocks until space available for writing num_pages
- cb_push_back(cb_id, num_pages): Producer publishes written num_pages to consumer
- cb_wait_front(cb_id, num_pages): Consumer blocks until num_pages available for reading
- cb_pop_front(cb_id, num_pages): Consumer frees processed num_pages
Typical pattern: cb_wait_front → process → cb_pop_front → cb_reserve_back → write → cb_push_back

For CB synchronization bugs (deadlocks, hangs), use the `ttnn-riscv-debugger` agent.

### Reader and Writer Kernel Naming Convention

Naming is based on FUNCTION, not RISC-V core assignment. However, naming conventions often align with core assignments:

#### Standard Configuration
- Reader kernels: Run on RISCV_0 (BRISC), use NOC0, fetch data from DRAM/other cores into L1 circular buffers
- Writer kernels: Run on RISCV_1 (NCRISC), use NOC1, send computed results from L1 circular buffers to DRAM/other cores

#### Important Caveats
1. Naming ≠ Actual Function: A kernel named "writer" (because it runs on RISCV_1) may actually READ data in certain patterns. The name reflects the core assignment convention, not necessarily the operation.
2. Split Reader Pattern: Optimization where activation reading is split between RISCV_0 and RISCV_1. Both cores perform reading, but the RISCV_1 instance may still be called a "writer" by convention. Note: NOC1 reads are slower than NOC0 reads, so split reader is justified only when the operation is RISC-bound (cores busy issuing instructions or computing coordinates) or compute-bound, and activation data is large enough that halving transfer time significantly reduces compute kernel stall time.
3. Writer Kernels Can Read: Writer kernels frequently read data, semaphores, and synchronization primitives, not just write data.
4. Flexible Assignment: While NOC0/RISCV_0 for readers and NOC1/RISCV_1 for writers is standard, assignments can vary (e.g., based on preferred_noc_for_dram_read). Both NoCs can be used simultaneously for the same data movement to increase bandwidth.
5. NOC Preference Functions: tt::tt_metal::detail::preferred_noc_for_dram_read() and preferred_noc_for_dram_write() determine optimal NOC for a given architecture. These functions calculate wrapped distances based on source/destination coordinates and grid size, selecting the NOC with shorter distance. The two NOCs traverse the chip in opposite directions, enabling
quasi-full-duplex operation.

### Compute APIs
- **Abstraction layer** for hardware compatibility across generations (Grayskull, Wormhole, Blackhole)
- Use provided compute APIs (`add_tiles`, `sin_tile`, etc.) instead of low-level implementations
- Different hardware has different vector widths and instruction sets

### Data Flow
Typical pattern: NoC0 → Unpacker → FPU/SFPU → Packer → NoC1

## Common Pitfalls

1. **Fast vs Slow Dispatch**: Always use fast dispatch (default) for production. Slow dispatch is debugging-only.
2. **Tile Alignment**: Tensors must align to 32×32 tiles (dimensions padded to multiples of 32)
3. **Memory Management**: No automatic caching—explicit SRAM allocation and DMA required
4. **SPMD Work Distribution**: Use `tt::tt_metal::split_work_to_cores()` to distribute work evenly
5. **Kernel Arguments**: Must set runtime args for ALL cores, even if some do no work (set args to no-op)
6. **Security**: Avoid command injection, validate inputs, especially in device-side code

## Documentation and Resources. USE THESE PROACTIVELY.

### Documentation in tt-metal repo

**METALIUM_GUIDE.md**: Comprehensive architecture and programming guide covering hardware fundamentals (Tensix processors, tile-based operations, NoC), programming model (reader/compute/writer kernels, circular buffers), and multi-chip scaling.
Reference when learning core architecture or developing low-level kernels.

**tech_reports/prog_examples/multicast/multicast.md**: Tutorial on efficiently broadcasting data tiles from one core to multiple receiver cores using NoC multicast primitives and semaphore synchronization. Reference when implementing parallel data distribution patterns.

**tech_reports/prog_examples/NoC_tile_transfer/NoC_tile_transfer.md**: Guide to direct L1-to-L1 inter-core communication via NoC, covering semaphore-based synchronization and circular buffer management.
Reference when building multi-core pipelines requiring core-to-core data transfers.

**tech_reports/tensor_accessor/tensor_accessor.md**: TensorAccessor utility
Reference for mapping logical tensor indices to physical memory locations across distributed banks
Reference when you need to understand how kernels access tensor data.

**tech_reports/tensor_layouts/tensor_layouts.md**: Explanation of tensor layouts (row-major vs tiled) and memory layouts (interleaved vs sharded) in TT-Metal.
Reference when dealing with tensor creation, configuring data organization, or understanding page-to-bank mapping.

**tech_reports/tensor_sharding/tensor_sharding.md**: Guide to dividing tensors across memory banks using four strategies (height, width, block, ND sharding) for performance optimization
Reference when you need to understand how tensors could be sharded between cores for better data locality.

### Online Documentation
- API Reference: https://docs.tenstorrent.com/tt-metal/latest/
- TT-NN Reference: https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html
- TT-Metalium Reference: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/index.html

### Deep Wiki Integration
For high-level architectural questions, hardware details, design patterns, and best practices, use the Deep Wiki MCP tool with repository path "tenstorrent/tt-metal". Deep Wiki has comprehensive indexing of architecture docs and tech reports. For best results, ask Deep Wiki one question at a time, starting from high-level ones down to implementation details as you discover you need more information.
For example:
"What are reader and writer kernels in tt-metal and what are their primary responsibilities?" -> (after learning that reader and writer run on RISC-V cores) "Which RISC-V core numbers (RISCV_0, RISCV_1, etc) are reader and writer kernels assigned to run on?" -> (after learning there is a "split reader" pattern) "What is the split reader pattern and how does it relate to reader and writer kernel assignments?"
To better understand how kernels work, you can also ask DeepWiki about what certain low-level (e.g. compute, data movement or transform) functions called in kernels do.
For example:
"What does the reduce_tile_math function do?"
"What are the inputs and outputs of unpack_tilizeA_B_block?"

## Creating New TTNN Operations

To create a new operation based on an existing reference, use the agents in `.claude/agents/`:

```
1. ttnn-operation-analyzer  → Analyze reference op → {ref}_analysis.md
2. ttnn-operation-planner   → Design new op       → {new}_spec.md  [USER REVIEW]
3. ttnn-operation-scaffolder → Stages 1-3         → API, validation, registration
4. ttnn-factory-builder     → Stages 4-6         → Program factory, stub kernels
```

See `.claude/subagent_breakdown.md` for detailed workflow and https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html for official docs.

## Debugging TTNN Operations

For kernel-level bugs in TTNN operations:

```
ttnn-riscv-debugger  → Debug kernel issues → Propose fix
```

Invoke with symptom only (e.g., "test hangs"). The agent autonomously:
1. Enables watcher and captures core states
2. Analyzes kernel code with grep/sed verification
3. Forms and tests hypotheses (observe → hypothesize → experiment)
4. Proposes verified fixes with diffs

## Performance Considerations

- **Tile sizes**: Operations on 32×32 tiles are native and most efficient
- **NoC bandwidth**: Minimize data movement, use sharded memory when beneficial
- **SRAM capacity**: 1.5MB per Tensix—keep intermediate results in SRAM when possible
- **Operator fusion**: Less critical than on CPU/GPU due to abundant on-chip SRAM
- **Tracy Profiler**: Use for performance analysis (enabled by default, disable with `--disable-profiler`)
