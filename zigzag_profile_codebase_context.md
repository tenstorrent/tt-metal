# Context: Ring Joint SDPA Codebase

> **Related**: This document supports Phase 1 (✅ COMPLETED) and Phase 2 implementation.
> See `zigzag_profile_vision.md` for the full roadmap.

This document captures all the context gathered while analyzing the ring_joint_sdpa implementation for creating a single-device profiling op.

---

## 1. File Locations

### Core Implementation
```
ttnn/cpp/ttnn/operations/transformer/sdpa/device/
├── ring_joint_sdpa_device_operation.hpp
├── ring_joint_sdpa_device_operation.cpp
├── ring_joint_sdpa_device_operation_types.hpp
├── ring_joint_sdpa_program_factory.hpp
├── ring_joint_sdpa_program_factory.cpp
└── kernels/
    ├── compute/
    │   ├── ring_joint_sdpa.cpp
    │   ├── compute_common.hpp
    │   └── compute_streaming.hpp
    └── dataflow/
        ├── ring_joint_reader.cpp
        ├── ring_joint_writer.cpp
        ├── fused_op_receiver.hpp
        └── dataflow_common.hpp
```

### Python Bindings
```
ttnn/cpp/ttnn/operations/transformer/sdpa/
├── sdpa_nanobind.cpp
├── sdpa.hpp
└── sdpa.cpp
```

### Existing Tests
```
models/tt_dit/tests/unit/test_ring_joint_attention.py
models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py
tests/nightly/t3000/ccl/test_ring_joint_attention.py
tests/nightly/tg/ccl/test_ring_joint_attention.py
```

### Related Ops
```
ttnn/cpp/ttnn/operations/transformer/sdpa/device/
├── ring_distributed_sdpa_program_factory.cpp  # Has explicit zigzag chunk assignment
├── joint_sdpa_program_factory.cpp
└── sdpa_program_factory.cpp
```

---

## 2. Key Data Structures

### RingJointSDPAParams (ring_joint_sdpa_device_operation_types.hpp)
```cpp
struct RingJointSDPAParams {
    std::optional<std::string> joint_strategy;
    std::optional<float> scale;
    bool is_causal = false;
    bool is_balanced = false;
    std::size_t logical_n = 0;
    std::size_t ring_size = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes;
    experimental::prim::RingAttentionAllGatherAsyncInputs all_gather_tensor_args;
    CoreCoord ccl_core_grid_offset;
    // ...
};
```

### RingJointSDPAInputs
```cpp
struct RingJointSDPAInputs {
    Tensor input_q;
    Tensor input_k;
    Tensor input_v;
    std::optional<Tensor> joint_q;
    std::optional<Tensor> joint_k;
    std::optional<Tensor> joint_v;
    Tensor gathered_k;
    Tensor gathered_v;
};
```

---

## 3. Kernel Compile-Time Arguments

### Reader Kernel (ring_joint_reader.cpp)
```cpp
constexpr uint32_t B = get_compile_time_arg_val(0);
constexpr uint32_t NH = get_compile_time_arg_val(1);
constexpr uint32_t NHK = get_compile_time_arg_val(2);
constexpr uint32_t DHt = get_compile_time_arg_val(3);
constexpr uint32_t vDHt = get_compile_time_arg_val(4);
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
constexpr uint32_t local_padded_N = get_compile_time_arg_val(7);
constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(8);
constexpr uint32_t padded_Nt = get_compile_time_arg_val(9);
constexpr uint32_t logical_n = get_compile_time_arg_val(10);
constexpr uint32_t logical_nt = get_compile_time_arg_val(11);
constexpr uint32_t Lt = get_compile_time_arg_val(12);
constexpr uint32_t L = get_compile_time_arg_val(13);
constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(14);
constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(15);
constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(16);
constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(17);
constexpr uint32_t num_q_chunks = get_compile_time_arg_val(18);
constexpr uint32_t ring_size = get_compile_time_arg_val(19);
constexpr uint32_t is_causal = get_compile_time_arg_val(20);
constexpr uint32_t is_balanced = get_compile_time_arg_val(21);
// Followed by TensorAccessorArgs for q, k, v, gathered_k, gathered_v, joint_q, joint_k, joint_v
// Then semaphore IDs
```

### Compute Kernel (ring_joint_sdpa.cpp)
```cpp
constexpr bool is_causal = get_compile_time_arg_val(37) == 1;
constexpr bool is_balanced = get_compile_time_arg_val(38) == 1;
```

---

## 4. Causal + Balanced Logic in Kernels

### Reader Kernel Skip Logic (ring_joint_reader.cpp:159, 189-191)
```cpp
// Skip entire ring iteration for causal non-balanced
const bool ring_iter_does_work = (ring_iter_processes_KV_chunks || (do_joint_kv && L != 0)) &&
                                 !(is_causal && ring_index < ring_id && !is_balanced);

// Skip early Q chunks when processing later ring_ids in balanced mode
if (q_chunk < half_sequence && is_balanced && ring_index < ring_id) {
    continue;
}
```

### Compute Kernel Adjust Logic (ring_joint_sdpa.cpp:221-227)
```cpp
bool causality = (ring_iter == 0 ? is_causal : false);

uint32_t iter_num_kv_chunks = num_kv_chunks;
if (is_causal && is_balanced && ring_index > ring_id) {
    iter_num_kv_chunks /= 2;
}
bool balancing = (ring_index >= ring_id ? false : is_balanced);

sdpa_ring<...>(..., causality, balancing);
```

### Key Variables
- `ring_index`: This device's position in the ring (0 to ring_size-1), fixed per device
- `ring_id`: ID of the device whose KV we're currently processing, varies per ring iteration
- `ring_iter`: Iteration counter (0 to ring_size-1)
- `half_sequence`: `num_q_chunks / 2`, used for balanced skip logic

---

## 5. Fused Op Receiver (fused_op_receiver.hpp)

Handles bidirectional ring all-gather synchronization.

```cpp
struct RingSDPAOpReceiver {
    uint32_t ring_size = 0;
    uint32_t ring_index = 0;
    std::array<volatile tt_l1_ptr uint32_t*, 2> signal_op_semaphore_addr_ptrs = {};
    std::array<uint32_t, 2> received_inputs = {};  // [from_forward, from_backward]
    std::array<uint32_t, 2> expected_inputs = {};
    uint32_t curr_dir = 0;  // 0=forward, 1=backward
    uint32_t curr_transfer_idx = 0;

    uint32_t get_next_ring_id_and_sync() {
        // First iteration: returns local ring_index
        // Subsequent iterations: alternates between directions
        // Direction 0 (forward): ring_id = (ring_index + count) % ring_size
        // Direction 1 (backward): ring_id = (ring_index - count + ring_size) % ring_size
    }
};
```

### Direction Semantics
- Direction 0: Receiving from forward device (ring_id goes forward from ring_index)
- Direction 1: Receiving from backward device (ring_id goes backward from ring_index)

### Linear Topology Expected Inputs
For device at `ring_index`:
- `from_forward = ring_size - 1 - ring_index`
- `from_backward = ring_index`

---

## 6. Balanced Chunk Order Implementation

Found in `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py`:

```python
def create_balanced_chunk_order(rp_factor):
    """Create balanced chunk order for sequence reordering.

    For rp_factor=4, creates 2*4=8 chunks with order: 0,7,1,6,2,5,3,4
    This interleaves chunks from start and end to balance workload.
    """
    num_chunks = 2 * rp_factor
    balanced_order = []
    left = 0
    right = num_chunks - 1

    for i in range(num_chunks):
        if i % 2 == 0:
            balanced_order.append(left)
            left += 1
        else:
            balanced_order.append(right)
            right -= 1

    return balanced_order


def reorder_tensor_chunks(tensor, chunk_order, seq_dim=2):
    """Reorder tensor chunks along sequence dimension according to chunk_order."""
    # ... splits tensor into chunks and reorders according to chunk_order


def reverse_reorder_tensor_chunks(tensor, chunk_order, seq_dim=2):
    """Reverse the chunk reordering to restore original order."""
    # ... creates inverse permutation and applies it
```

### Usage Pattern (test_ring_joint_mla.py:196-205)
```python
# Apply balanced reordering if requested
chunk_order = None
if is_balanced:
    rp_factor = submesh.shape[rp_axis]
    chunk_order = create_balanced_chunk_order(rp_factor)

    padded_Q = reorder_tensor_chunks(padded_Q, chunk_order, seq_dim=2)
    padded_K = reorder_tensor_chunks(padded_K, chunk_order, seq_dim=2)
    padded_V = reorder_tensor_chunks(padded_V, chunk_order, seq_dim=2)
```

---

## 7. Python API (sdpa_nanobind.cpp:329-412)

```python
ttnn.transformer.ring_joint_scaled_dot_product_attention(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    persistent_output_buffer_k,
    persistent_output_buffer_v,
    logical_n,
    program_config,
    scale=None,
    compute_kernel_config=None,
    dim,
    multi_device_global_semaphore,
    num_links,
    cluster_axis,
    mesh_device,
    topology,
    subdevice_id=None,
    ccl_core_grid_offset,
    use_column_major_ccl=False,
    is_causal=False,
    is_balanced=False,
    joint_tensor_q=None,
    joint_tensor_k=None,
    joint_tensor_v=None,
    joint_strategy=None,
) -> Tuple[Tensor, Optional[Tensor], Tensor]
```

---

## 8. Circular Buffers Used

From reader/compute/writer kernels:
```cpp
constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
// ... intermediate buffers for QK, out, statistics, etc.
constexpr uint32_t cb_out = tt::CBIndex::c_16;
```

---

## 9. Ring Distributed SDPA (for reference)

`ring_distributed_sdpa_program_factory.cpp` has explicit zigzag chunk assignment:

```cpp
// Lines 78-80
const uint32_t chunk_1 = ring_id;
const uint32_t chunk_2 = (2 * ring_size) - ring_id - 1;
```

This shows device `ring_id` gets chunks `[ring_id, 2*ring_size - ring_id - 1]`.

---

## 10. Test Utilities

### fa_rand (tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py)
```python
def fa_rand(b, nh, seq_len, d):
    """Generate random tensor for flash attention testing."""
    return torch.randn(b, nh, seq_len, d)
```

### Comparison Functions
```python
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
# comp_pcc(actual, expected, threshold) -> (passed, pcc_value)
```

### Mesh Mappers
```python
ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=shard_dims)
ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=concat_dims)
```

---

## 11. Verified Ring ID Arrival Order

### Linear Topology, ring_size=2
| Device | expected_inputs | arrival_order |
|--------|-----------------|---------------|
| 0 | (1, 0) | [0, 1] |
| 1 | (0, 1) | [1, 0] |

### Linear Topology, ring_size=4
| Device | expected_inputs | arrival_order |
|--------|-----------------|---------------|
| 0 | (3, 0) | [0, 1, 2, 3] |
| 1 | (2, 1) | [1, 2, 0, 3] |
| 2 | (1, 2) | [2, 3, 1, 0] |
| 3 | (0, 3) | [3, 2, 1, 0] |

### Linear Topology, ring_size=8
| Device | expected_inputs | arrival_order |
|--------|-----------------|---------------|
| 0 | (7, 0) | [0, 1, 2, 3, 4, 5, 6, 7] |
| 1 | (6, 1) | [1, 2, 0, 3, 4, 5, 6, 7] |
| 2 | (5, 2) | [2, 3, 1, 4, 0, 5, 6, 7] |
| 3 | (4, 3) | [3, 4, 2, 5, 1, 6, 0, 7] |
| 4 | (3, 4) | [4, 5, 3, 6, 2, 7, 1, 0] |
| 5 | (2, 5) | [5, 6, 4, 7, 3, 2, 1, 0] |
| 6 | (1, 6) | [6, 7, 5, 4, 3, 2, 1, 0] |
| 7 | (0, 7) | [7, 6, 5, 4, 3, 2, 1, 0] |

---

## 12. Complete Chunk Assignment

### ring_size=2
- `balanced_order = [0, 3, 1, 2]`
- Device 0: chunks [0, 3], KV arrival: [(0, [0,3]), (1, [1,2])]
- Device 1: chunks [1, 2], KV arrival: [(1, [1,2]), (0, [0,3])]

### ring_size=4
- `balanced_order = [0, 7, 1, 6, 2, 5, 3, 4]`
- Device 0: chunks [0, 7], KV arrival: 0→1→2→3
- Device 1: chunks [1, 6], KV arrival: 1→2→0→3
- Device 2: chunks [2, 5], KV arrival: 2→3→1→0
- Device 3: chunks [3, 4], KV arrival: 3→2→1→0
