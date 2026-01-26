# Reduce Helper Assertions Plan

## Overview

Add static and runtime assertions to reduce helpers to catch invalid inputs early.

## Assertions to Implement

| Check | Location | Type | Priority |
|-------|----------|------|----------|
| CB IDs < NUM_CIRCULAR_BUFFERS | `reduce_helpers_compute.inl` | Runtime | High |
| PoolType != MIN | `reduce_helpers_compute.inl` | Static | High |
| InputPolicy is valid | `reduce_helpers_compute.inl` | Static | High |
| ReconfigPolicy is valid | `reduce_helpers_compute.inl` | Static | High |
| AccumulateT is valid | `reduce_helpers_compute.inl` | Static | High |
| PostReduceOp is callable | `reduce_helpers_compute.inl` | Static | High |
| InputBlockShape non-zero | `reduce_helpers_compute.inl` | Runtime | Medium |
| InputMemoryLayout row_stride valid | `reduce_helpers_compute.inl` | Runtime | Low |
| Accumulator CB < NUM_CIRCULAR_BUFFERS | `reduce_helpers_compute.inl` | Runtime | Medium |

## Details

### 1. CB IDs < NUM_CIRCULAR_BUFFERS (Runtime, High Priority)

**What:** Validate that `input_cb`, `scaler_cb`, and `output_cb` parameters are < NUM_CIRCULAR_BUFFERS (32).

**Where:** At `reduce()` function entry point in `reduce_helpers_compute.inl`.

**Note:** Uses the `NUM_CIRCULAR_BUFFERS` constant. Same pattern already exists in `reduce_helpers_dataflow.inl:23`.

```cpp
ASSERT(input_cb < NUM_CIRCULAR_BUFFERS);
ASSERT(scaler_cb < NUM_CIRCULAR_BUFFERS);
ASSERT(output_cb < NUM_CIRCULAR_BUFFERS);
```

### 2. PoolType != MIN (Static, High Priority)

**What:** The reduce helper only supports `SUM`, `AVG`, and `MAX`. `MIN` exists in the enum but is not implemented.

**Where:** At `reduce()` function template in `reduce_helpers_compute.inl`.

```cpp
static_assert(reduce_type != PoolType::MIN, "PoolType::MIN is not supported for reduce operations");
```

### 3. InputPolicy is valid (Static, High Priority)

**What:** Validate that the `InputPolicy` template parameter is a valid input policy type (StreamingPolicy, StreamingBatchedPolicy, PreloadedPolicy, or PersistentPolicy).

**Where:** At `reduce()` function template in `reduce_helpers_compute.inl`.

**Note:** Uses the `is_input_policy_v` type trait defined in `reduce_helper_policies.hpp`.

```cpp
static_assert(reduce_policies::is_input_policy_v<InputPolicy>,
              "InputPolicy must be a valid input policy (StreamingPolicy, StreamingBatchedPolicy, PreloadedPolicy, or PersistentPolicy)");
```

### 4. ReconfigPolicy is valid (Static, High Priority)

**What:** Validate that the `ReconfigPolicy` template parameter is a valid reconfig policy type (ReconfigNonePolicy, ReconfigInputPolicy, ReconfigOutputPolicy, or ReconfigBothPolicy).

**Where:** At `reduce()` function template in `reduce_helpers_compute.inl`.

**Note:** Uses the `is_reconfig_policy_v` type trait defined in `reduce_helper_policies.hpp`.

```cpp
static_assert(reduce_policies::is_reconfig_policy_v<ReconfigPolicy>,
              "ReconfigPolicy must be a valid reconfig policy (ReconfigNonePolicy, ReconfigInputPolicy, ReconfigOutputPolicy, or ReconfigBothPolicy)");
```

### 5. AccumulateT is valid (Static, High Priority)

**What:** Validate that the `AccumulateT` template parameter is a valid accumulation type (NoAccumulation or Accumulate).

**Where:** At `reduce()` function template in `reduce_helpers_compute.inl`.

**Note:** Uses the `is_accumulation_type_v` type trait defined in `reduce_helpers_compute.hpp`.

```cpp
static_assert(is_accumulation_type_v<AccumulateT>,
              "AccumulateT must be a valid accumulation type (NoAccumulation or Accumulate)");
```

### 6. PostReduceOp is callable (Static, High Priority)

**What:** Validate that the `PostReduceOp` template parameter is callable with a `uint32_t` argument.

**Where:** At `reduce()` function template in `reduce_helpers_compute.inl`.

**Note:** Uses the `is_post_reduce_op_v` type trait defined in `reduce_helpers_compute.hpp`.

```cpp
static_assert(is_post_reduce_op_v<PostReduceOp>,
              "PostReduceOp must be callable with a uint32_t argument");
```

### 7. InputBlockShape non-zero (Runtime, Medium Priority)

**What:** Validate that `InputBlockShape` dimensions are non-zero.

**Where:** At `reduce()` entry point in `reduce_helpers_compute.inl`.

```cpp
ASSERT(input_block_shape.rows > 0, "InputBlockShape rows must be > 0");
ASSERT(input_block_shape.cols > 0, "InputBlockShape cols must be > 0");
ASSERT(input_block_shape.batches > 0, "InputBlockShape batches must be > 0");
```

### 8. InputMemoryLayout row_stride valid (Runtime, Low Priority)

**What:** If explicit row_stride is provided, validate it is >= cols.

**Where:** At `reduce()` entry point in `reduce_helpers_compute.inl`.

**Note:** The current `InputMemoryLayout` only has `row_stride` (no `batch_stride`).

```cpp
if (input_memory_layout.row_stride != 0) {
    ASSERT(input_memory_layout.row_stride >= input_block_shape.cols,
           "row_stride must be >= cols");
}
```

### 9. Accumulator CB < NUM_CIRCULAR_BUFFERS (Runtime, Medium Priority)

**What:** When accumulation is used, validate the accumulator CB ID is < NUM_CIRCULAR_BUFFERS.

**Where:** At `reduce()` entry point when `Accumulate` is provided in `reduce_helpers_compute.inl`.

```cpp
if constexpr (is_accumulate_v<AccumulateT>) {
    ASSERT(accumulate.config.cb_accumulator < NUM_CIRCULAR_BUFFERS);
}
```
