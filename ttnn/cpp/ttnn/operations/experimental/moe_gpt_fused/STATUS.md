# moe_gpt_fused - Fused MoE Compute Op for GPT-OSS

## Overview

`moe_gpt_fused` is a fused Mixture-of-Experts compute operation for the GPT-OSS model's
throughput expert path. It performs the full expert computation (W0/W1 matmul, SwiGLU
activation, W2 matmul) in a single op on Tenstorrent hardware, avoiding intermediate
DRAM round-trips.

### Dimensions (current test configuration)
- H = 2880 (hidden size), 90 tiles
- N = 2880 (intermediate size), 90 tiles
- E = 4 experts per device
- tokens = 32 (1 chunk, 1 tile row)
- Weights: bfloat4_b, HEIGHT_SHARDED on DRAM (12 banks)

### Activation: SwiGLU
```
gate_clamped = clamp(gate, max=7.0)
up_clamped   = clamp(up, min=-7.0, max=7.0)
result       = (up_clamped + 1) * gate_clamped * sigmoid(1.702 * gate_clamped)
```

## Architecture

The op uses three types of cores:

### Tilize Cores (3 cores: CoreRange({5,0},{5,2}))
- Receive WIDTH_SHARDED ROW_MAJOR input from L1, shard shape [32, 960] per core
- Core 0 is the "drain" core: gathers tiles from cores 1-2, then pushes all 90 tiles
  to every matmul core's c_1 circular buffer
- Address exchange: dm0 on ring_core_id 0 writes c_1 address to drain's semaphore,
  drain reads it to know where to send tiles

### Matmul Cores (12 cores, DRAM-bank-aligned)
- Logical coords: (3,6),(0,0),(0,4),(0,5),(4,0),(7,3),(4,1),(6,3),(4,6),(4,2),(4,4),(4,5)
- Each core reads weight tiles from its aligned DRAM bank (direct bank addressing via
  `get_noc_addr_from_bank_id`, NOT InterleavedAddrGen)
- Ring-based all-to-all: activation tiles rotate through the ring so each core computes
  its portion of the matmul for all experts
- Kernels per matmul core:
  - `matmul_dm0`: reads weights from DRAM, manages ring activation transfer
  - `matmul_dm1`: writes untilized W2 output to combine cores
  - `matmul_compute`: performs matmul + SwiGLU + second matmul
- c_14 buffer holds ALL experts' untilized output (32*E pages) since dm1 drains after
  the ring completes for all experts

### Combine Cores (12 cores: CoreRange({1,0},{3,3}), 3 cols x 4 rows)
- Receive untilized W2 output via `combine_dm1` kernel
- Output is BLOCK_SHARDED ROW_MAJOR on L1
- Each height shard = one expert, shard shape [32, 960]
- Total output shape: [E*tokens, H] = [128, 2880]

## File Structure

```
ttnn/cpp/ttnn/operations/experimental/moe_gpt_fused/
  moe_gpt_fused.hpp / .cpp              - Top-level op interface
  moe_gpt_fused_nanobind.hpp / .cpp     - Python bindings
  device/
    moe_gpt_fused_device_operation.hpp / .cpp     - Device op definition
    moe_gpt_fused_device_operation_types.hpp      - Op parameter types
    moe_gpt_fused_program_factory.hpp / .cpp      - Core assignment & program creation
    kernels/
      matmul_dm0.cpp        - Weight reader + ring activation data mover
      matmul_dm1.cpp        - Output writer (untilize + send to combine cores)
      matmul_compute.cpp    - Matmul + SwiGLU compute kernel
      gather_reader.cpp     - Tilize core: drain/gather reader
      gather_compute.cpp    - Tilize core: tilize compute
      gather_writer.cpp     - Tilize core: tile writer
      combine_dm1.cpp       - Combine core: receives output tiles
      swiglu_sfpu.h         - SwiGLU SFPU implementation
      moe_gpt_fused_ring_common.h  - Ring protocol shared definitions
```

## Test

```
tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_fused.py
```

- `test_moe_gpt_fused`: basic test with PCC check against torch reference (threshold 0.984)
- `test_moe_gpt_fused_with_metadata`: tests metadata tensor plumbing (per-expert token counts)
- Weight preparation uses shared helpers from `models/demos/gpt_oss/tt/experts_throughput/weights.py`

## Completed Milestones

| Milestone | Description | Commit |
|-----------|-------------|--------|
| M1 | Untilize output - basic matmul with untilized output | 97a41965c1d |
| M2 | DRAM input - read activations from DRAM interleaved buffers | c0b6e966b13 |
| M2.5 | Broadcast input - ring-based all-to-all activation broadcast | 85825cce1b4 |
| M4 | Integration - end-to-end fused op with W0/W1 + SwiGLU + W2 | e5fcd382c4d |
| M5 | Combine cores - untilize W2 output + write to BLOCK_SHARDED L1 on combine cores | 660564eaf2d |
| M6 | Tilize cores - convert WIDTH_SHARDED ROW_MAJOR input to TILE on 3 tilize cores | fece981d39b |

All milestones achieve PCC >= 0.984 for all experts.

## Known Issues / Lessons Learned

- **DRAM ROW_MAJOR page size**: `InterleavedAddrGen` and `TensorAccessor` produce garbage
  data for ROW_MAJOR DRAM reads with non-tile page sizes (e.g. 5760 bytes). TILE-sized
  pages (2048 bytes) work correctly. Root cause unknown. Workaround: use TILE layout for
  DRAM interleaved buffers, or use direct bank addressing for weight reads.
- **BLOCK_SHARDED grids**: grid dimensions must match shard dimensions (e.g. 3 columns for
  width_shard_dim=3).

## Remaining Work

1. **Multi-device testing** - Currently tested on single device only. Need to validate the
   op works correctly in the multi-device GPT-OSS expert throughput path with
   `all_to_all_dispatch` / `all_to_all_combine`.

2. **Variable total_tokens (>32)** - Currently hardcoded to 32 tokens (1 tile row). Need to
   support variable token counts, likely requiring multi-chunk processing where each chunk
   is 32 tokens.

3. **Weight loading from model checkpoint** - Currently uses random weights. Need to integrate
   with the model's weight loading path so that the fused op can use real checkpoint weights
   via the existing `prepare_w0_w1_tensor` / `prepare_w2_tensor` helpers.

4. **Sparse routing** - Currently all tokens go to all experts. Need to integrate with the
   actual top-k routing so only routed tokens are processed per expert, using the metadata
   tensor for per-expert token counts.

5. **Performance profiling** - Measure and optimize throughput vs the unfused expert path.
