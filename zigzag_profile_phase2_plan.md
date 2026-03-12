# Plan: Phase 2 - Implement ring_joint_sdpa_profile Op

## STATUS: COMPLETE (2026-03-12)

All 32 tests pass:
- 26 Phase 1 helper function tests
- 6 Phase 2 device tests (ring_size=2,4 with all ring_index values)
- PCC > 0.99 against PyTorch reference

## Context

**Problem**: Phase 1 established test infrastructure for validating single-device profiling of `ring_joint_sdpa`. Now we need to implement the actual profiling op that:
- Runs on a single device
- Reads pre-staged KV from DRAM (instead of ring all-gather)
- Reuses the exact compute kernel
- Removes all synchronization overhead

**Goal**: Implement `ring_joint_sdpa_profile` op that passes the Phase 1 tests.

---

## Architecture Overview

The profiling op is a **simplified version** of `ring_joint_sdpa`:

| Component | ring_joint_sdpa | ring_joint_sdpa_profile |
|-----------|-----------------|-------------------------|
| Device operation | Full CCL params | Simplified (no all_gather) |
| Reader kernel | `fused_op_receiver` sync | Direct DRAM reads |
| Writer kernel | Ring sync | Direct DRAM writes |
| Compute kernel | `ring_joint_sdpa.cpp` | **Reuse exactly** |
| Program factory | CCL setup | Simplified |

---

## Implementation Steps

### Step 1: Create Device Operation Types

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_device_operation_types.hpp`

```cpp
struct RingJointSDPAProfileParams {
    std::optional<std::string> joint_strategy;
    std::optional<float> scale;
    bool is_causal = false;
    bool is_balanced = false;
    std::size_t logical_n = 0;
    std::size_t ring_size = 0;
    std::size_t ring_index = 0;  // NEW: which device we're simulating
    tt::tt_metal::MemoryConfig output_memory_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // NO all_gather params - key simplification
};

struct RingJointSDPAProfileInputs {
    Tensor input_q;      // Local Q
    Tensor input_k;      // Local K
    Tensor input_v;      // Local V
    Tensor gathered_k;   // Pre-staged full KV in arrival order
    Tensor gathered_v;
    std::optional<Tensor> joint_q;
    std::optional<Tensor> joint_k;
    std::optional<Tensor> joint_v;
};
```

### Step 2: Create Device Operation

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_device_operation.hpp`

```cpp
struct RingJointSDPAProfileDeviceOperation {
    using operation_attributes_t = RingJointSDPAProfileParams;
    using tensor_args_t = RingJointSDPAProfileInputs;
    using spec_return_value_t = RingJointSDPAResultSpec;  // Reuse
    using tensor_return_value_t = RingJointSDPAResult;    // Reuse
    using program_factory_t = std::variant<RingJointSDPAProfileProgramFactory>;

    static void validate_on_program_cache_miss(...);
    static spec_return_value_t compute_output_specs(...);
    static tensor_return_value_t create_output_tensors(...);
    static tt::stl::hash::hash_t compute_program_hash(...);
};
```

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_device_operation.cpp`

Key differences from `ring_joint_sdpa_device_operation.cpp`:
- No `RingAttentionAllGatherAsync` validation
- No CCL core grid overlap check
- Validate `ring_index < ring_size`
- Simpler hash computation (no all_gather hash)

### Step 3: Create Program Factory

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_program_factory.hpp/cpp`

Key differences from `ring_joint_sdpa_program_factory.cpp`:
- No CCL worker setup
- No `sdpa_fused_op_signaler`
- `ring_index` passed as compile-time arg (not from device topology)
- Use simplified reader/writer kernels

### Step 4: Create Simplified Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_profile_reader.cpp`

Changes from `ring_joint_reader.cpp`:

1. **Remove includes**:
   ```cpp
   // REMOVE: #include "fused_op_receiver.hpp"
   ```

2. **Remove runtime args for sync**:
   ```cpp
   // REMOVE: is_chain_participant, is_injector, is_sink, chain_*
   // REMOVE: prev_physical_x/y, next_physical_x/y
   // REMOVE: fused_op_receiver construction
   ```

3. **Add ring_index as compile-time arg**:
   ```cpp
   constexpr uint32_t ring_index = get_compile_time_arg_val(22);  // New position
   ```

4. **Replace `fused_op_receiver.get_next_ring_id_and_sync()` with pre-computed arrival order**:
   ```cpp
   // Pre-compute arrival order at compile time or pass as runtime args
   // For ring_iter, compute ring_id using same logic as Python helper
   uint32_t ring_id = get_ring_id_for_iter(ring_index, ring_size, ring_iter);
   ```

5. **Remove semaphore operations**:
   ```cpp
   // REMOVE: sender_semaphore_addr, receiver_semaphore_addr, valid_semaphore_addr
   // REMOVE: noc_semaphore_wait, noc_semaphore_inc
   ```

6. **Read from gathered_k/v for all iterations** (not just ring_iter > 0):
   - The gathered buffer is pre-staged with all KV in arrival order
   - Offset calculation: `ring_iter * local_padded_Nt`

### Step 5: Create Simplified Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_profile_writer.cpp`

Changes from `ring_joint_writer.cpp`:

1. **Remove includes**:
   ```cpp
   // REMOVE: #include "fused_op_receiver.hpp"
   ```

2. **Remove fused_op_receiver**:
   ```cpp
   // REMOVE: RingSDPAOpReceiver construction and get_next_ring_id_and_sync()
   ```

3. **Add ring_index as compile-time arg**

4. **Simplify the main loop**: Just write output, no sync needed

### Step 6: Reuse Compute Kernel

**No changes needed** to `ring_joint_sdpa.cpp`. The compute kernel already receives:
- `ring_index` via compile-time arg
- `ring_id` via reader pushing to CB
- `is_causal`, `is_balanced` flags

The profiling reader will push the same data to CBs, so compute works unchanged.

### Step 7: Add Python Bindings

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`

Add new binding (simpler signature):
```cpp
ttnn::bind_registered_operation(
    mod,
    ttnn::transformer::ring_joint_sdpa_profile,
    profile_doc,
    ttnn::nanobind_overload_t{
        [](const ProfileOperationType& self,
           const ttnn::Tensor& input_tensor_q,
           const ttnn::Tensor& input_tensor_k,
           const ttnn::Tensor& input_tensor_v,
           const ttnn::Tensor& gathered_k,
           const ttnn::Tensor& gathered_v,
           std::size_t ring_size,
           std::size_t ring_index,
           std::size_t logical_n,
           const SDPAProgramConfig& program_config,
           bool is_causal,
           bool is_balanced,
           std::optional<float> scale,
           std::optional<DeviceComputeKernelConfig> compute_kernel_config,
           const std::optional<ttnn::Tensor>& joint_tensor_q,
           const std::optional<ttnn::Tensor>& joint_tensor_k,
           const std::optional<ttnn::Tensor>& joint_tensor_v,
           const std::optional<std::string>& joint_strategy) {
            return self(...);
        },
        // ... arg bindings
    });
```

**File**: `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`

Add declaration for the profile operation.

---

## Files to Create

| File | Based On |
|------|----------|
| `ring_joint_sdpa_profile_device_operation_types.hpp` | Simplified from `ring_joint_sdpa_device_operation_types.hpp` |
| `ring_joint_sdpa_profile_device_operation.hpp` | New, simpler |
| `ring_joint_sdpa_profile_device_operation.cpp` | Simplified from `ring_joint_sdpa_device_operation.cpp` |
| `ring_joint_sdpa_profile_program_factory.hpp` | New |
| `ring_joint_sdpa_profile_program_factory.cpp` | Simplified from `ring_joint_sdpa_program_factory.cpp` |
| `kernels/dataflow/ring_joint_profile_reader.cpp` | Modified from `ring_joint_reader.cpp` |
| `kernels/dataflow/ring_joint_profile_writer.cpp` | Modified from `ring_joint_writer.cpp` |

## Files to Modify

| File | Change |
|------|--------|
| `sdpa_nanobind.cpp` | Add `ring_joint_sdpa_profile` binding |
| `sdpa.hpp` | Add declaration |
| `sdpa.cpp` | Add implementation wrapper |

---

## Key Implementation Details

### Ring ID Arrival Order (must match Python test helpers)

The reader kernel needs to compute `ring_id` for each `ring_iter`. Options:

**Option A: Compile-time array** (preferred for simplicity)
```cpp
// Pass as compile-time args or compute in kernel
constexpr uint32_t ring_id_order[MAX_RING_SIZE] = {...};
uint32_t ring_id = ring_id_order[ring_iter];
```

**Option B: Compute in kernel** (matches Python logic exactly)
```cpp
uint32_t get_ring_id_for_iter(uint32_t ring_index, uint32_t ring_size, uint32_t ring_iter) {
    if (ring_iter == 0) return ring_index;
    // ... bidirectional alternating logic
}
```

### Gathered KV Buffer Layout

Pre-staged by Python test (already implemented in Phase 1):
```
[local_kv (ring_iter=0) | arrival_1_kv | arrival_2_kv | ...]
```

Reader reads from offset: `ring_iter * local_padded_Nt * DHt * tile_bytes`

---

## Verification

1. **Build**: `./build_metal.sh`

2. **Run Phase 1 tests** (enable op calls):
   ```bash
   python -m pytest tests/ttnn/unit_tests/operations/sdpa/test_ring_joint_sdpa_profile.py -v
   ```

3. **Test all configurations**:
   - ring_size=2, ring_index=0,1
   - ring_size=4, ring_index=0,1,2,3

---

## Success Criteria

- [x] Op compiles without errors
- [x] All 26 Phase 1 tests pass with actual op (not just PyTorch reference)
- [x] Output matches PyTorch reference with PCC > 0.99
- [x] Can run on single-device system
