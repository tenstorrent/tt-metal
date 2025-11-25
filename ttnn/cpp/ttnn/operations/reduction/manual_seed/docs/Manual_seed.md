# TTNN Manual Seed

## Overview

The TTNN Manual Seed operation is a device-level utility designed to initialize random number generator (RNG) seeds on Tenstorrent hardware cores. This operation provides flexible control over seed initialization across compute cores, enabling deterministic and reproducible random number generation in subsequent operations.

### Parameters

- **seeds**: Either a scalar `uint32_t` value or a `Tensor` of `uint32_t` values representing the seed(s) to initialize the random number generator(s)
- **device**: The target device on which to set the seed(s). Required when seeds is a scalar value. Can be a single device or multi-chip mesh device
- **user_ids**: Optional parameter that specifies which core(s) should receive the seed(s). Can be:
  - A scalar `uint32_t` representing a single core ID (valid range: 0-31)
  - A `Tensor` of `uint32_t` values representing multiple core IDs
  - Omitted (None) to apply the seed to all cores
- **sub_core_grids**: Optional custom core range set for advanced use cases

The operation supports multiple seeding strategies, from setting a single seed across all cores to fine-grained control where individual cores receive specific seeds based on user-provided mappings.

### Input Combinations

The following table shows all valid combinations of input parameters and their behavior:

| Seeds Type | User IDs Type | Device Type | Behavior |
|------------|---------------|-------------|----------|
| `uint32_t` | None | Single or Multi-chip | Push seed to all cores on all devices (single device or all devices in mesh_device) |
| `uint32_t` | `uint32_t` (core ID) | Single or Multi-chip | Push seed to the same core_id on each device in mesh_device (or single device based on input) |
| `uint32_t` | 1D Tensor (`uint32_t`) | Derived from tensor | Set same seed to all core IDs listed in user_ids tensor on the device(s) where the user_ids tensor is placed |
| 1D Tensor (`uint32_t`) | 1D Tensor (`uint32_t`) | Derived from tensors | Elements in seeds tensor correspond to user_ids elements at matching indices; seeds are pushed to the relevant device(s) where the user_ids tensor is placed |

**Notes:**
- When seeds and user_ids are both tensors, they must have the same shape and volume
- Scalar user_ids values must be in the range [0, 31]
- When seeds is a scalar, the device parameter must be explicitly provided
- When seeds is a tensor, the device is derived from the tensor's placement

## Brief Functional Description

The TTNN Manual Seed operation configures the random number generator state on device cores by setting seed values. It operates directly on the device hardware without producing tensor outputs, making it a configuration operation rather than a data transformation operation.

The operation provides four distinct strategies for seed distribution:

1. **Single Seed to All Cores**: Set the same seed value across all available compute cores
2. **Single Seed to Single Core**: Set a seed to one specific core identified by a core ID
3. **Single Seed to Set of Cores**: Set the same seed to multiple cores specified by a tensor of core IDs
4. **Multiple Seeds to Set of Cores**: Map different seeds to different cores based on paired tensors

### Usage Details

- Supported tensor types: `uint32` only
- Supported tensor layout: `ROW_MAJOR` only
- Tensor inputs must be 1-dimensional
- When providing tensor seeds and user_ids, they must have the same shape and volume
- Scalar user_ids represents a single core ID and must be in range [0, 31]
- Device must be provided when seeds is a scalar value
- When seeds is provided as a tensor, user_ids must also be a tensor (or omitted)
- Cannot provide user_ids as a scalar when seeds is a tensor

## Strategy Comparison Overview

The TTNN Manual Seed operation provides four distinct strategies for initializing random number generator seeds across device cores. Each strategy is optimized for different use cases and offers varying levels of control over seed distribution.

| Strategy                           | Seeds Input | User IDs Input | Description                                             | Use Case                                                    |
| ---------------------------------- | ----------- | -------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| **Single Seed to All Cores**       | uint32_t    | None           | Sets the same seed to all compute cores on the device   | Uniform random initialization across all cores              |
| **Single Seed to Single Core**     | uint32_t    | uint32_t       | Sets a seed to one specific core identified by core ID  | Targeted initialization of a single core                    |
| **Single Seed to Set of Cores**    | uint32_t    | Tensor         | Sets the same seed to multiple cores from a tensor list | Uniform seeding for a subset of cores                       |
| **Multiple Seeds to Set of Cores** | Tensor      | Tensor         | Maps different seeds to different cores via tensors     | Fine-grained control with unique seeds per core             |

## Strategy Descriptions

### Single Seed to All Cores

This strategy sets the same seed value across all available compute cores on the device. It is the simplest and most straightforward seeding approach.

**Invocation Pattern**: `seeds` as scalar, `user_ids` omitted

#### Overview:

1. **Core Grid Calculation**:
   * Determines all available compute cores from the device
   * Can be overridden with custom `sub_core_grids` parameter

2. **Kernel Deployment**:
   * A single compute kernel is created with the seed value as a compile-time argument
   * The kernel is deployed to all cores in the computed core grid

3. **Seed Initialization**:
   * Each core executes the kernel, which calls `rand_tile_init(seed)`
   * This initializes the hardware random number generator with the specified seed

4. **No Runtime Arguments**:
   * All configuration is done at compile time
   * No data movement or runtime parameter passing is required

#### Implementation Details:

```cpp
// Compute kernel: manual_seed_set_seed.cpp
void MAIN {
    constexpr uint32_t seed = get_compile_time_arg_val(0);
    rand_tile_init(seed);
}
```

---

### Single Seed to Single Core

This strategy sets a seed to one specific compute core, identified by a core ID provided as a scalar value. It targets a single core while leaving others unmodified.

**Invocation Pattern**: `seeds` as scalar, `user_ids` as scalar (core ID in range 0-31)

#### Overview:

1. **Core Selection**:
   * Computes the full device core grid
   * Selects the specific core based on the provided `user_ids` scalar value (core ID)
   * Core ID is converted to physical core coordinates

2. **Targeted Kernel Deployment**:
   * Compute kernel is created with the seed as a compile-time argument
   * Kernel is deployed only to the selected core

3. **Seed Initialization**:
   * Only the targeted core executes the kernel
   * Calls `rand_tile_init(seed)` to initialize RNG state

4. **No Runtime Arguments**:
   * Similar to the all-cores strategy, everything is compile-time configured

#### Implementation Details:

```cpp
// Core selection logic
const auto& cores = corerange_to_cores(core_grid, num_cores, true);
const auto& core_chosen = cores.at(operation_attributes.user_ids.value_or(0));

// Same kernel as Single Seed to All Cores
// Deployed to: core_chosen only
```

---

### Single Seed to Set of Cores

This strategy sets the same seed value to multiple cores specified by a tensor of core IDs. It combines the uniformity of a single seed with the selectivity of core targeting. Unlike the "Single Seed to All Cores" strategy which seeds all available cores, this strategy only seeds the cores whose IDs are listed in the user_ids tensor.

**Invocation Pattern**: `seeds` as scalar, `user_ids` as tensor (containing core IDs)

#### Overview:

1. **Core Grid Setup**:
   * Computes the device core grid
   * All cores in the grid are assigned kernels (both reader and compute)

2. **Data Movement - User IDs**:
   * A circular buffer (CB) is created for the user_ids tensor
   * Each core has a reader kernel that loads the user_ids tensor from DRAM to L1
   * The user_ids tensor contains core IDs that should receive the seed

3. **Core Matching and Seed Initialization**:
   * Each core's compute kernel reads the user_ids tensor from the CB
   * The kernel iterates through the user_ids to check if its core ID matches any entry
   * If a match is found, the core calls `rand_tile_init(seed)` to initialize RNG
   * Non-matching cores skip initialization and remain unmodified

4. **Runtime Arguments**:
   * User_ids tensor buffer address (updated on cache hit)
   * Number of IDs in the tensor (updated on cache hit)

#### Implementation Details:

**Reader Kernel:**
```cpp
// reader_manual_seed_read_user_id.cpp
void kernel_main() {
    const uint32_t user_ids_tensor_buffer_addr = get_arg_val<uint32_t>(0);

    // Read user_ids tile from DRAM to circular buffer
    cb_reserve_back(user_ids_cb_index, one_tile);
    const uint32_t l1_write_addr_index = get_write_ptr(user_ids_cb_index);
    noc_async_read_tile(0, user_ids_tensor_dram, l1_write_addr_index);
    noc_async_read_barrier();
    cb_push_back(user_ids_cb_index, one_tile);
}
```

**Compute Kernel:**
```cpp
// manual_seed_single_seed_receive_user_id.cpp
void MAIN {
    const uint32_t number_of_ids = get_arg_val<uint32_t>(0);
    constexpr uint32_t core_id = get_compile_time_arg_val(0);
    constexpr uint32_t seed = get_compile_time_arg_val(2);

    // Read user_ids from circular buffer
    cb_wait_front(user_ids_cb_index, one_tile);
    uint32_t* user_ids = nullptr;
    cb_get_tile(user_ids_cb_index, 0, &user_ids);
    user_ids += metadata_fields;  // Skip tile metadata

    // Check if this core should be seeded
    for (uint32_t i = 0; i < number_of_ids; i++) {
        if (core_id == user_ids[i]) {
            rand_tile_init(seed);
            break;
        }
    }

    cb_pop_front(user_ids_cb_index, one_tile);
}
```

---

### Multiple Seeds to Set of Cores

This strategy provides maximum flexibility by allowing different seeds to be assigned to different cores through paired tensors. Each core receives a unique seed based on the mapping provided by matching core IDs to their corresponding seed values.

**Invocation Pattern**: `seeds` as tensor, `user_ids` as tensor (must have same shape and volume)

#### Overview:

1. **Core Grid Setup**:
   * Computes the device core grid
   * All cores in the grid receive both reader and compute kernels

2. **Data Movement - Dual Tensors**:
   * Two circular buffers are created:
     * CB 0: user_ids tensor (containing core IDs)
     * CB 1: seeds tensor (containing corresponding seed values)
   * Each core's reader kernel loads both tensors from DRAM to L1
   * Tensors must have identical shapes and volumes (1-dimensional)

3. **Core Matching and Unique Seed Initialization**:
   * Each core's compute kernel reads both user_ids and seeds tensors
   * Iterates through user_ids to find its core ID
   * When found, retrieves the corresponding seed from the seeds tensor at the same index
   * Calls `rand_tile_init(seed)` with the matched seed value
   * Cores not listed in user_ids remain unmodified

4. **Runtime Arguments**:
   * User_ids tensor buffer address
   * Seeds tensor buffer address
   * Number of IDs in the tensors

#### Implementation Details:

**Reader Kernel:**
```cpp
// reader_manual_seed_read_all_data.cpp
void kernel_main() {
    const uint32_t user_ids_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t seeds_tensor_buffer_addr = get_arg_val<uint32_t>(1);

    // Read user_ids tile from DRAM
    cb_reserve_back(user_ids_cb_index, one_tile);
    const uint32_t l1_write_addr_index = get_write_ptr(user_ids_cb_index);
    noc_async_read_tile(0, user_ids_tensor_dram, l1_write_addr_index);
    noc_async_read_barrier();
    cb_push_back(user_ids_cb_index, one_tile);

    // Read seeds tile from DRAM
    cb_reserve_back(seeds_cb_index, one_tile);
    const uint32_t seeds_l1_write_addr_index = get_write_ptr(seeds_cb_index);
    noc_async_read_tile(0, seeds_tensor_dram, seeds_l1_write_addr_index);
    noc_async_read_barrier();
    cb_push_back(seeds_cb_index, one_tile);
}
```

**Compute Kernel:**
```cpp
// manual_seed_receive_all_data.cpp
void MAIN {
    const uint32_t number_of_ids = get_arg_val<uint32_t>(0);
    constexpr uint32_t core_id = get_compile_time_arg_val(0);

    // Read user_ids from circular buffer
    cb_wait_front(user_ids_cb_index, one_tile);
    uint32_t* user_ids = nullptr;
    cb_get_tile(user_ids_cb_index, 0, &user_ids);
    user_ids += metadata_fields;

    // Find matching core ID and get corresponding seed
    for (uint32_t i = 0; i < number_of_ids; i++) {
        if (core_id == user_ids[i]) {
            // Read seeds from circular buffer
            cb_wait_front(seeds_cb_index, one_tile);
            uint32_t* seeds = nullptr;
            cb_get_tile(seeds_cb_index, 0, &seeds);
            seeds += metadata_fields;
            const uint32_t seed = seeds[i];

            // Initialize RNG with matched seed
            rand_tile_init(seed);

            cb_pop_front(seeds_cb_index, one_tile);
            break;
        }
    }

    cb_pop_front(user_ids_cb_index, one_tile);
}
```

---

## Program Factory Selection Logic

The operation automatically selects the appropriate program factory based on the input arguments:

| Seeds Type | User IDs Type | Selected Strategy |
|------------|---------------|-------------------|
| uint32_t   | None (omitted) | Single Seed to All Cores |
| uint32_t   | uint32_t (core ID) | Single Seed to Single Core |
| uint32_t   | Tensor (core IDs) | Single Seed to Set of Cores |
| Tensor     | Tensor (core IDs) | Multiple Seeds to Set of Cores |

Note: When seeds is provided as a Tensor, user_ids must also be provided as a Tensor with matching shape and volume, or the operation will fail validation.

© Tenstorrent AI ULC 2025
