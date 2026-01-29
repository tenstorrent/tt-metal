---
name: ttnn-analyze-op
description: Get guidance on analyzing TTNN operation implementations. Use when you need to understand how an operation works, document its architecture, or prepare to implement a similar operation. Provides structured analysis steps for program factories and kernels.
---

# TTNN Operation Analysis Guide

Use this skill when you need to deeply understand how a TTNN operation is implemented - its kernels, data flow, memory patterns, and core distribution.

## When to Use

- Understanding an existing operation's implementation
- Documenting how an operation works
- Debugging operation behavior
- Learning patterns before implementing a new operation
- Reviewing operation code

## Quick Start

To analyze an operation, you need its program factory file:
```
ttnn/cpp/ttnn/operations/{category}/{operation_name}/device/{operation_name}_program_factory.cpp
```

---

## Analysis Checklist

### 1. Initial Reconnaissance

**Goal**: Map out all the operation's components.

- [ ] Read the program factory file
- [ ] Identify all kernel files (reader, compute, writer)
- [ ] Note any variant implementations (sharded vs interleaved, different data types)
- [ ] List conditional compilation paths

**Key questions**:
- What kernels does this operation use?
- Are there multiple program variants?
- What tensor layouts are supported?

### 2. Work Unit Definition

**Goal**: Understand what constitutes one unit of work.

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile / block / row / element |
| **Unit size** | e.g., "1 tile", "Wt tiles (row)", "N×H tiles" |
| **Total units** | Formula (e.g., `N * C * Ht * Wt`) |
| **Loop structure** | e.g., "outer NC, inner HW" |

**Key questions**:
- What does the inner loop iterate over?
- How are work units assigned to cores?

### 3. Tensor Format and Layout

**Goal**: Document input/output tensor requirements.

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | [N, H, W, C] | [N, H, W, C] |
| **Dimension convention** | NHWC / NCHW | NHWC / NCHW |
| **Tensor layout** | TILE_LAYOUT / ROW_MAJOR | TILE_LAYOUT / ROW_MAJOR |
| **Memory layout** | INTERLEAVED / SHARDED | INTERLEAVED / SHARDED |
| **Buffer type** | DRAM / L1 | DRAM / L1 |
| **Data type** | BFLOAT16 / FLOAT32 / etc | BFLOAT16 / FLOAT32 / etc |

**For sharded tensors, also document**:
- Shard shape
- Core grid
- Shard orientation (row-major / column-major)

### 4. Data Flow Mapping

**Goal**: Trace data from input to output through all stages.

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 | CB_in | reserve_back, push_back |
| 2 | Compute | CB_in | CB_out | wait_front, pop_front, reserve_back, push_back |
| 3 | Writer | CB_out | DRAM/L1 | wait_front, pop_front |

**Important caveats**:
- "Reader" kernel runs on RISCV_0 - may also write to other cores
- "Writer" kernel runs on RISCV_1 - may also read data (split reader pattern)
- Names reflect core assignment convention, not actual function

### 5. Circular Buffer Configuration

**Goal**: Document all CBs and their purpose.

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer |
|-------|------|---------|----------|------------|-----------|----------|----------|
| c_0 | cb_input | Input staging | 2 tiles | 1 tile | Double | Reader | Compute |
| c_16 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer |
| c_24 | cb_intermed | Intermediate | 4 tiles | 2 tiles | Double | Compute | Compute |

**Purpose values**: Input staging, Output staging, Scratchpad, Accumulator, Intermediate, Scaler
**Buffering types**:
- Single: capacity == block_size (no overlap possible)
- Double: capacity == 2 × block_size (producer/consumer overlap)
- Multi: capacity > 2 × block_size

### 6. Index Calculations

**Goal**: Understand how logical tensor indices map to physical memory.

**Look for**:
- TensorAccessor usage (modern pattern)
- InterleavedAddrGen (legacy pattern)
- Manual index calculations

**Document**:
- How tile indices are computed from loop variables
- Any coordinate transformations (transpose, permute)
- Page offset calculations

### 7. Memory Access Patterns

**Goal**: Characterize read/write ordering.

**Read Pattern**:
- Sequential / Strided / Tiled / Random
- Access source (DRAM, L1, other cores via NoC)

**Write Pattern**:
- Sequential / Strided / Tiled
- Access destination (DRAM, L1, other cores via NoC)

### 8. Core Distribution Strategy

**Goal**: Document how work is split across cores.

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D / 2D |
| **Grid dimensions** | rows × cols |
| **Total cores** | N |
| **Work per core** | e.g., "tiles / num_cores" |
| **Load balancing** | equal / two-groups / weighted |
| **Remainder handling** | extra to first N cores / last core / ignored |

**Look for**: `split_work_to_cores()` usage which creates two core groups.

### 9. Argument Classification

**Goal**: Separate compile-time from runtime arguments.

**Compile-Time Arguments** (affect kernel structure):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_block | uint32_t | Tiles per processing block |

**Runtime Arguments** (vary per invocation):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer address |
| 1 | dst_addr | uint32_t | Destination buffer address |
| 2 | num_tiles | uint32_t | Tiles to process |

**Rule of thumb**: User-facing parameters that change per call → runtime args.

### 10. Kernel Specifications

**Goal**: Document each kernel's responsibilities.

| Kernel | Core | NOC | Input | Output | Key Operations |
|--------|------|-----|-------|--------|----------------|
| reader | RISCV_0 | NOC0 | DRAM | CB_in | noc_async_read, cb_push_back |
| compute | RISCV_2 | N/A | CB_in | CB_out | unpack, math ops, pack |
| writer | RISCV_1 | NOC1 | CB_out | DRAM | cb_wait_front, noc_async_write |

**For each kernel, note**:
- File path
- Key loop structure
- Any special synchronization (semaphores, barriers)

---

## Output Template

When documenting an operation, use this structure:

```markdown
# {Operation Name} Implementation Analysis

## Overview
- What the operation does
- Path to program factory
- Supported variants

## Work Unit Definition
[Work Unit Table]

## Tensor Format and Layout
### Input Tensor(s)
[Tensor Format Table]

### Output Tensor(s)
[Tensor Format Table]

## Data Flow
[Data Flow Table]
[Step-by-step description]

## Circular Buffer Configuration
[CB Table]

## Core Distribution
[Core Distribution Table]

## Arguments
### Compile-Time
[Compile-Time Arguments Table]

### Runtime
[Runtime Arguments Table]

## Kernels
### Reader
- File: {path}
- Key logic: {description}

### Compute
- File: {path}
- Key logic: {description}

### Writer
- File: {path}
- Key logic: {description}

## Implementation Notes
[Special optimizations, edge cases, gotchas]
```

---

## Research Resources

When analyzing an operation, consult these resources:

| Resource | Use Case |
|----------|----------|
| `METALIUM_GUIDE.md` | Core architecture concepts |
| `tech_reports/tensor_layouts/` | Tensor layout patterns |
| `tech_reports/tensor_sharding/` | Sharding strategies |
| `tech_reports/prog_examples/multicast/` | Multicast patterns |
| `tech_reports/tensor_accessor/` | TensorAccessor API |
| DeepWiki (`tenstorrent/tt-metal`) | Specific function/API questions |

**DeepWiki query examples**:
- "What does `reduce_tile_math` do in tt-metal?"
- "How does `split_work_to_cores` work?"
- "What is the split reader pattern?"

---

## Common Patterns to Identify

### Streaming Pattern
- Single-buffered or double-buffered CBs
- Process one tile/block at a time
- Low memory footprint

### Accumulation Pattern
- Larger CB capacity to hold partial results
- Multiple passes over input
- Used in reductions

### Split Reader Pattern
- Both RISCV_0 and RISCV_1 read input data
- Used when operation is compute-bound
- Note: NoC1 reads are slower than NoC0

### Multicast Pattern
- One core broadcasts to many
- Uses semaphores for synchronization
- Common in attention/matmul operations

---

## Workflow

1. **Start**: Read the program factory file
2. **Map**: Identify all kernel files and CBs
3. **Trace**: Follow data from input through each kernel to output
4. **Document**: Fill in tables section by section
5. **Research**: Use DeepWiki/docs for unfamiliar functions
6. **Review**: Verify CB push/pop balance, arg indices match
