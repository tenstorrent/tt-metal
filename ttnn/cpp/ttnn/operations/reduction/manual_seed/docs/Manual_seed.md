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

The implementation uses a single compute kernel deployed across the entire core grid. The seed value is passed as a compile-time argument, and each core executes `rand_tile_init(seed)` to initialize its RNG state. No data movement or runtime arguments are required, making this the most efficient strategy.

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

The program factory converts the scalar core ID into physical core coordinates by mapping it to the device's core grid. The same compute kernel used in "Single Seed to All Cores" is deployed, but only to the single selected core. The seed is passed as a compile-time argument, ensuring minimal overhead.

---

### Single Seed to Set of Cores

This strategy sets the same seed value to multiple cores specified by a tensor of core IDs. It combines the uniformity of a single seed with the selectivity of core targeting. Unlike the "Single Seed to All Cores" strategy which seeds all available cores, this strategy only seeds the cores whose IDs are listed in the user_ids tensor.

**Invocation Pattern**: `seeds` as scalar, `user_ids` as tensor (containing core IDs)

#### Overview:

1. **Core Grid Setup**:
   * Computes the device core grid
   * All cores in the grid are assigned kernels (both reader and compute)

2. **Kernel Deployment**:
   * A single reader kernel and single compute kernel are created for the entire core grid
   * Each core receives the same kernels but with different runtime arguments
   * The seed is passed as a compile-time argument to the compute kernel

3. **Data Movement and Processing**:
   * Each core's reader kernel loads the user_ids tensor from DRAM to L1
   * The reader kernel checks if its core_id (passed at runtime) matches any ID in the tensor
   * The match result is communicated to the compute kernel via circular buffer

4. **Seed Initialization**:
   * The compute kernel reads the match result from the CB message
   * If matched, it calls `rand_tile_init(seed)` to initialize RNG
   * Non-matching cores skip initialization and remain unmodified

5. **Runtime Arguments**:
   * User_ids tensor buffer address (updated on cache hit)
   * Core ID for each core (unique per core)

#### Implementation Details:

**Program Factory:**

The program factory creates a single reader kernel and a single compute kernel that are deployed across the entire core grid. This is more efficient than creating individual kernels per core. The seed value is passed as a compile-time argument to the compute kernel. Each core receives unique runtime arguments consisting of the user_ids tensor buffer address and its core ID.

**Reader Kernel (`reader_manual_seed_read_user_id.cpp`):**

Each core's reader kernel performs the following operations:
1. Reads the user_ids tensor from DRAM into L1 memory using NoC operations
2. Iterates through the user_ids array to check if its core ID (received as runtime argument) matches any entry
3. Writes the match result (boolean) to a dedicated circular buffer for kernel communication
4. Pushes the communication entry to make it available to the compute kernel

The circular buffer communication mechanism provides a structured way for the reader kernel (running on the data movement processor) to pass information to the compute kernel.

**Compute Kernel (`manual_seed_single_seed_receive_user_id.cpp`):**

The compute kernel waits for data from the reader kernel via `cb_wait_front()` on the kernel communication circular buffer. It then reads the match result using `read_tile_value()`. If the match is positive, it initializes the RNG by calling `rand_tile_init(seed)` with the seed value received as a compile-time argument. After processing, it pops the communication entry with `cb_pop_front()`. Non-matching cores skip initialization entirely.

---

### Multiple Seeds to Set of Cores

This strategy provides maximum flexibility by allowing different seeds to be assigned to different cores through paired tensors. Each core receives a unique seed based on the mapping provided by matching core IDs to their corresponding seed values.

**Invocation Pattern**: `seeds` as tensor, `user_ids` as tensor (must have same shape and volume)

#### Overview:

1. **Core Grid Setup**:
   * Computes the device core grid
   * All cores in the grid receive both reader and compute kernels

2. **Kernel Deployment**:
   * A single reader kernel and single compute kernel are created for the entire core grid
   * Each core receives the same kernels but with different runtime arguments
   * No compile-time arguments are passed to the compute kernel

3. **Data Movement and Processing**:
   * Each core's reader kernel loads both user_ids and seeds tensors from DRAM to L1
   * The reader kernel checks if its core_id (passed at runtime) matches any ID in user_ids
   * If matched, it retrieves the corresponding seed from the seeds tensor at the same index
   * Both the match result and the seed value are communicated to the compute kernel via circular buffer
   * Tensors must have identical shapes and volumes (1-dimensional)

4. **Seed Initialization**:
   * The compute kernel reads the match result from the CB message
   * If matched, it reads the seed value from the CB message
   * Calls `rand_tile_init(seed)` with the matched seed value
   * Cores not listed in user_ids remain unmodified

5. **Runtime Arguments**:
   * User_ids tensor buffer address
   * Seeds tensor buffer address
   * Core ID for each core (unique per core)

#### Implementation Details:

**Program Factory:**

Similar to the "Single Seed to Set of Cores" strategy, the factory creates one reader kernel and one compute kernel for the entire core grid. However, the compute kernel receives no compile-time arguments since the seed values vary per core. Each core receives runtime arguments containing both tensor buffer addresses (user_ids and seeds) along with its core ID.

**Reader Kernel (`reader_manual_seed_read_all_data.cpp`):**

Each core's reader kernel executes a more complex workflow:
1. Reads both the user_ids tensor and seeds tensor from DRAM into separate L1 memory regions using NoC operations
2. Iterates through the user_ids array to find if its core ID matches any entry
3. If a match is found, retrieves the corresponding seed value from the seeds tensor at the same index
4. Writes both the match result (boolean at index 0) and the seed value (at index 1) to the kernel communication circular buffer
5. Pushes the communication entry to make it available to the compute kernel

The circular buffer communication pattern allows passing both control information (whether to initialize) and data (the seed value) from the reader to the compute kernel in a single structured message.

**Compute Kernel (`manual_seed_receive_all_data.cpp`):**

The compute kernel waits for data from the reader kernel via `cb_wait_front()` on the kernel communication circular buffer. It reads the match result from index 0 using `read_tile_value()`. If positive, it reads the seed value from index 1 and calls `rand_tile_init(seed)` to initialize the RNG with the core-specific seed. After processing, it pops the communication entry with `cb_pop_front()`. This allows each core to receive a unique seed value while using the same kernel code.

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
