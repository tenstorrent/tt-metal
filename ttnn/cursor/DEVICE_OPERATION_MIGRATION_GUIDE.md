# Migrate Device Operation to TMP (Template Metaprogramming) Pattern

**Task**: Migrate a device operation from the old vector-based structure to the new TMP device operation pattern that eliminates unnecessary heap allocations.

**Instructions**:
1. Replace `<PROVIDED DEVICE OPERATION>` with the actual operation name you're migrating (e.g., 'Embedding', 'Unary', 'Dropout', etc.)
2. Locate the old device operation structure in the codebase
3. Follow each section below systematically to create the new TMP structure
4. Update all call sites to use the new prim registration
5. Run tests to verify the migration

---

## Old Device Operation vs TMP Device Operation

### Old Operation Structure

The old operation is represented by a fairly simple structure. The main issue is that inputs and outputs are processed as vectors, so even for unary operations we do heap allocations for inputs and outputs.

**Example - Old Unary Operation:**

```cpp
struct Unary {
    const std::vector<UnaryWithParam> op_chain;
    const MemoryConfig output_mem_config;
    bool fp32_dest_acc_en;
    bool preserve_fp32_precision;
    DataType output_dtype;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<Tensor>> &optional_output_tensors) const;

    std::vector<tt::tt_metal::Shape> compute_output_shapes(
        const std::vector<Tensor>& input_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;

    UnaryOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor>& input_tensors) const;

    const operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors) const;
};
```

### New TMP Device Operation Structure

The new TMP Device Operation is more verbose but allows precise input/output types, which eliminates unnecessary heap allocations.

**Example - New Unary Operation:**

```cpp
struct operation_attributes_t {
    const std::vector<UnaryWithParam> op_chain;
    const DataType output_dtype = DataType::INVALID;
    const MemoryConfig output_memory_config;
    const bool fp32_dest_acc_en = false;
    const bool preserve_fp32_precision = false;
};

struct tensor_args_t {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;
using shape_return_value_t = ttnn::Shape;

struct UnaryDeviceOperation {
    using operation_attributes_t = unary::operation_attributes_t;
    using tensor_args_t = unary::tensor_args_t;
    using shape_return_value_t = unary::shape_return_value_t;
    using tensor_return_value_t = unary::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::UnaryProgramFactory,
        program::UnaryShardedProgramFactory
    >;

    static program_factory_t select_program_factory(
        const operation_attributes_t&,
        const tensor_args_t&);

    static void validate_on_program_cache_hit(
        const operation_attributes_t&,
        const tensor_args_t&);

    static void validate_on_program_cache_miss(
        const operation_attributes_t&,
        const tensor_args_t&);

    static shape_return_value_t compute_output_shapes(
        const operation_attributes_t&,
        const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&,
        const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const std::vector<UnaryWithParam>& op_chain,
        DataType output_dtype,
        const MemoryConfig& output_memory_config,
        bool fp32_dest_acc_en,
        bool preserve_fp32_precision,
        const std::optional<Tensor>& preallocated_output);
};
```

---

## Migration Process

### Step 1: Create `operation_attributes_t`

Use the old Device Operation structure to find members for `operation_attributes_t`. Extract all const member variables that represent operation configuration (not tensor arguments).

**Example - Old Embeddings Operation:**

```cpp
struct Embeddings {
    const MemoryConfig output_mem_config;       // ← Include
    const bool tilized;                         // ← Include
    const EmbeddingsType embeddings_type;       // ← Include
    const std::optional<uint32_t> pad_token;    // ← Include
    const DataType output_dtype;                // ← Include

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
};
```

**Resulting `operation_attributes_t`:**

```cpp
struct operation_attributes_t {
    MemoryConfig output_mem_config;
    bool tilized;
    EmbeddingsType embeddings_type;
    std::optional<uint32_t> pad_token;
    DataType output_dtype;
};
```

### Step 2: Create `tensor_args_t`

Use the Operation's `invoke` method signature to determine `tensor_args_t`. Include all Tensor parameters (both required and optional).

**Example - Embedding invoke signature:**

```cpp
struct Embedding {
    static inline Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_arg,              // ← Include
        const Tensor& weight_arg,                    // ← Include
        const std::optional<int>& pad_token = std::nullopt,
        const Layout& layout = ttnn::ROW_MAJOR_LAYOUT,
        EmbeddingsType embeddings_type = EmbeddingsType::GENERIC,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt)  // ← Include
```

**Resulting `tensor_args_t`:**

```cpp
struct tensor_args_t {
    Tensor input_tensor_arg;
    Tensor weight_arg;
    std::optional<Tensor> optional_output_tensor;
};
```

### Step 3: Define Return Types

#### For operations with a single return value:

```cpp
using shape_return_value_t = ttnn::Shape;
using tensor_return_value_t = Tensor;
```

**OR** (for newer operations using TensorSpec):

```cpp
using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;
```

#### For operations returning multiple tensors:

```cpp
using shape_return_value_t = std::vector<ttnn::Shape>;
using tensor_return_value_t = std::vector<Tensor>;
```

**OR** (using tuple):

```cpp
using shape_return_value_t = std::tuple<ttnn::Shape, ttnn::Shape>;
using tensor_return_value_t = std::tuple<Tensor, Tensor>;
```

**Note**: Prefer `spec_return_value_t` (TensorSpec) for newer operations as it provides more complete tensor metadata. Use `shape_return_value_t` only if maintaining compatibility with older patterns.

### Step 4: Implement `select_program_factory`

This method returns a `std::variant` listing all possible program factory types. The selection logic is usually straightforward based on tensor properties or operation attributes.

**Example:**

```cpp
UnaryDeviceOperation::program_factory_t UnaryDeviceOperation::select_program_factory(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        return program::UnaryShardedProgramFactory{};
    } else {
        return program::UnaryProgramFactory{};
    }
}
```

#### Step 4a: Mesh Workload Factories (When `mesh_coords` Filtering is Needed)

If your operation supports `mesh_coords` filtering (i.e., only executing on specific mesh coordinates), you need to create separate mesh workload factories. This is required because the infrastructure's `MeshWorkloadFactoryAdapter` creates programs for ALL tensor coordinates, ignoring `mesh_coords`.

**Pattern**: Create a separate factory that only implements `MeshWorkloadFactoryConcept` (not `ProgramFactoryConcept`), following the dropout example.

**Example - Dropout Pattern:**

```cpp
// In program factory header:
struct DropoutProgramFactory {
    using shared_variables_t = DropoutSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(...);
    static void override_runtime_arguments(...);
};

struct DropoutMeshWorkloadFactory {
    using shared_variables_t = DropoutSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};
```

**Implementation:**

```cpp
// In program factory .cpp:
DropoutMeshWorkloadFactory::cached_mesh_workload_t DropoutMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {

    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    // Filter coordinates based on mesh_coords if provided
    const std::optional<std::set<ttnn::MeshCoordinate>>& mesh_coords_opt = operation_attributes.mesh_coords;

    // Create programs for each coordinate in tensor_coords (filtered by mesh_coords if provided)
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            // Skip this coordinate if mesh_coords is provided and this coordinate is not in the set
            if (mesh_coords_opt.has_value()) {
                const auto& mesh_coords_set = mesh_coords_opt.value();
                if (mesh_coords_set.find(mesh_coord) == mesh_coords_set.end()) {
                    continue;  // Skip this coordinate
                }
            }

            // Create a program for this specific coordinate using the base factory
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = DropoutProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void DropoutMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    DropoutProgramFactory program_factory;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

        ttnn::device_operation::mesh_device_operation_utils::apply_override_runtime_arguments(
            program_factory,
            program,
            shared_variables,
            operation_attributes,
            *(coordinate_range.begin()),
            tensor_args,
            tensor_return_value);
    }
}
```

**Update variant and select_program_factory:**

```cpp
// In device operation header:
using program_factory_t = std::variant<
    program::DropoutProgramFactory,
    program::DropoutMeshWorkloadFactory>;

// In device operation .cpp:
DropoutDeviceOperation::program_factory_t DropoutDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Use mesh workload factory when mesh_coords is provided to enable coordinate filtering
    if (args.mesh_coords.has_value()) {
        return program::DropoutMeshWorkloadFactory{};
    } else {
        return program::DropoutProgramFactory{};
    }
}
```

**Why this pattern?** The infrastructure's `dispatch_to_mesh_workload_factory` checks `ProgramFactoryConcept` first. If your factory satisfies both concepts, it gets wrapped by the adapter (which doesn't filter by `mesh_coords`). By using a separate factory that only implements `MeshWorkloadFactoryConcept`, the infrastructure uses the direct path, giving you control over coordinate filtering.

**Real examples:**
- `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.{hpp,cpp}`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_program_factory.{hpp,cpp}`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.{hpp,cpp}`
- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_*_fused_update_cache_program_factory.{hpp,cpp}`

### Step 5: Implement Validation

Unless the old device operation provides a separate `validate_on_cache_hit`, implement `validate_on_cache_miss` and call it from `validate_on_cache_hit`.

**Example:**

```cpp
void DropoutDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void DropoutDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args) {
    // Copy validation logic from old operation's validate method
    // Adapt to use args and tensor_args instead of input_tensors vector
}
```

### Step 6: Register Prim

Register and call the device operation prim. **Do not call its `invoke` directly** - always use the registered prim.

**Registration (typically in the device operation header file):**

```cpp
namespace ttnn::prim {
constexpr auto dropout =
    ttnn::register_operation<
        "ttnn::prim::dropout",
        ttnn::operations::experimental::dropout::DropoutDeviceOperation
    >();
}  // namespace ttnn::prim
```

**Usage (replace old operation::run calls):**

```cpp
// OLD:
return operation::run(Embeddings{...}, input_tensors);

// NEW:
return ttnn::prim::embedding(input_tensor, weight, ...);
```

### Step 7: Create Program Factory

To define `shared_variables_t` and `override_runtime_arguments`, find the lambda that updates runtime arguments in the old program factory. It's usually called `override_runtime_args_callback`.

**Example - Old program factory lambda:**

```cpp
auto override_runtime_args_callback = [
    num_cores_x,
    num_cores_y,
    reader_kernel_id,
    writer_kernel_id,
    cores,
    device
](
    const Program &program,
    const std::vector<Buffer *> &input_buffers,
    const std::vector<Buffer *> &output_buffers) {

    auto output_dram_buffer = output_buffers.at(0);
    auto input_dram_buffer = input_buffers.at(0);
    auto weights_dram_buffer = input_buffers.at(1);

    for (const auto &core : cores) {
        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = input_dram_buffer->address();
            runtime_args[1] = weights_dram_buffer->address();
        }
        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_dram_buffer->address();
        }
    }
};
```

**Lambda capture → `shared_variables_t`:**

The lambda's capture list becomes `shared_variables_t`:

```cpp
struct shared_variables_t {
    uint32_t num_cores_x;
    uint32_t num_cores_y;
    KernelHandle reader_kernel_id;
    KernelHandle writer_kernel_id;
    std::vector<CoreCoord> cores;
    // Note: 'device' is typically not needed in shared_variables_t
};
```

**Lambda body → `override_runtime_arguments`:**

```cpp
void EmbeddingProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {

    // Buffers are not provided, so we take them from output/input tensors
    auto output_dram_buffer = output.buffer();
    auto input_dram_buffer = tensor_args.input_tensor_arg.buffer();
    auto weights_dram_buffer = tensor_args.weight_arg.buffer();

    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    for (const auto &core : cores) {
        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = input_dram_buffer->address();
            runtime_args[1] = weights_dram_buffer->address();
        }
        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_dram_buffer->address();
        }
    }
}
```

**Note**: If variables like `num_cores_x` and `num_cores_y` are not used in the override method, remove them from `shared_variables_t`.

### Step 8: Implement `compute_program_hash`

Hash is based on operation attributes and input arguments and helps reuse already compiled programs.

Note, the default legacy implementation hashes both operation attributes and input tensors.
Tensor hash includes:
 - Storage variant index (0 or 1)
 - Logical shape dimensions (vector of uint32_t)
- Data type (DataType enum)
- Page config variant index (0 or 1)
- Tile configuration (from page config)
- Memory layout (TensorMemoryLayout enum)
- Buffer type (BufferType enum)
- Shard spec (optional ShardSpec)
- ND shard spec (optional NdShardSpec)
- Created with ND shard spec flag (bool)
- Alignment dimensions (vector of uint32_t)

**If the legacy operation does not implement `compute_program_hash`**,
Do not implement `compute_program_hash` in the newly-migrated operation, because we want to mimic how the legacy was hashed.

**If the legacy operation implements `compute_program_hash`**,
Include everything that was in the legacy into the new hash. Make sure to include `tensor_args` as a whole.

**What must be included in hash:**
Anything that affects the compiled kernel binary:
- Setup of Circular Buffers
- Kernels and cores used
- Compile-time arguments/defines
- Operation attributes that affect program structure

**What to exclude from hash:**
Anything that has no effect on compiled binaries (runtime arguments)
- Buffer addresses
- Offsets
- Number of tiles to process (runtime arguments)

**Example migration:**

**Old:**

```cpp
const tt::tt_metal::operation::Hash Unary::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {

    const auto& input_tensor = input_tensors.at(0);
    const auto& input_shape = input_tensor.legacy_shape();

    tt::tt_metal::operation::Hash hash = operation::hash_operation<Unary>(
        compute_volume(input_shape),
        input_tensor.dtype(),
        std::get<DeviceStorage>(input_tensor.storage()).memory_config(),
        this->output_mem_config);

    for (const auto& unary_with_param_op : this->op_chain) {
        hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.op_type);
        if (unary_with_param_op.has_parameter()) {
            hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.params);
        }
    }

    return hash;
}
```

**New:**

```cpp
tt::stl::hash::hash_t UnaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args) {

    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.legacy_shape();

    auto program_factory = select_program_factory(args, tensor_args);

    operation::Hash hash = operation::hash_operation<UnaryDeviceOperation>(
        args,
        program_factory.index(),  // Include program factory variant index
        input_tensor.dtype(),     // impacts program in program factory
        std::get<DeviceStorage>(input_tensor.storage()).memory_config(),
        compute_volume(input_shape)); // core groups depend on volume

    for (const auto& unary_with_param_op : args.op_chain) {
        hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.op_type);
        if (unary_with_param_op.has_parameter()) {
            hash = tt::stl::hash::hash_objects(hash, unary_with_param_op.params); // impacts defines
        }
    }

    return hash;
}
```

---

## Examples

Great examples of operations migrated to TMP:

**Dropout operation**: `ttnn/cpp/ttnn/operations/experimental/dropout`
- Demonstrates mesh workload factory pattern for `mesh_coords` filtering
- PRs:
  - https://github.com/tenstorrent/tt-metal/pull/11793
  - https://github.com/tenstorrent/tt-metal/pull/11956

**Paged Cache operations**: `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/`
- `update_cache/`, `fill_cache/`, `fused_update_cache/` all demonstrate mesh workload factory pattern
- Shows how to handle multiple factory variants (tiled vs row_major for fused_update_cache)

**Send Async operations**: `ttnn/cpp/ttnn/operations/experimental/ccl/send_recv_async/send_async/device/`
- Demonstrates another mesh workload factory example
- PRs:
  - https://github.com/tenstorrent/tt-metal/pull/33005

---

## Building and Testing

### Building the Project

To build the project with debug symbols (without Python bindings for faster builds during development):

```bash
./build_metal.sh -c -e --debug
```

### Running Tests

After building, activate the Python environment and run the specific test file:

```bash
source python_env/bin/activate
pytest tests/ttnn/unit_tests/operations/<operation_name>/
```

### Resetting Device After Failure

If a test fails and leaves the device in a bad state, reset it:

```bash
tt-smi -r
```

---

## Migration Checklist

Use this checklist to ensure all steps are completed:

- [ ] **Step 1**: Created `operation_attributes_t` struct with all const configuration members
- [ ] **Step 2**: Created `tensor_args_t` struct with all Tensor parameters from invoke signature
- [ ] **Step 3**: Defined `tensor_return_value_t` and `spec_return_value_t` appropriately
- [ ] **Step 4**: Implemented `compute_output_specs`
- [ ] **Step 5**: [Optional] Implemented `create_output_tensors` (if legacy had it)
- [ ] **Step 6**: Implemented `select_program_factory` returning correct variant type
- [ ] **Step 6a**: [If needed] Created separate mesh workload factory for `mesh_coords` filtering support
- [ ] **Step 7**: Implemented `validate_on_program_cache_miss`
- [ ] **Step 8**: [Optional] Implemented `validate_on_program_cache_hit` (if legacy had it)
- [ ] **Step 9**: Registered prim in `ttnn::prim` namespace
- [ ] **Step 10**: Updated all call sites to use prim instead of direct invoke or `operation::run`
- [ ] **Step 11**: Created program factory with:
  - [ ] `shared_variables_t` struct (from lambda captures)
  - [ ] `create` method (from old `create_program`)
  - [ ] `override_runtime_arguments` method (from lambda body)
- [ ] **Step 12**: [Optional] Implemented `compute_program_hash` (if legacy had it)
- [ ] **Step 13**: Removed old device operation code (after verification)
- [ ] **Step 14**: Relevant test pass
- [ ] **Step 15**: Code compiles without warnings

---

## Common Pitfalls

1. **Forgetting to register the prim**: Always register in `ttnn::prim` namespace and use it instead of direct calls
2. **Including runtime-only values in hash**: Only hash compile-time constants that affect program structure
3. **Not including values that affect the program structure in hash**: Every parameter that has an effect on program structure must be taken into account in the hash
4. **Redundant tensors in tensor_args_t**: Do not add redundant arguments like `preallocated_output`, if legacy operation did not handle that explicitly in `create_output_tensors`
5. **Mesh workload factories not working with `mesh_coords`**: If your operation supports `mesh_coords` filtering, you MUST create a separate mesh workload factory. The infrastructure's `MeshWorkloadFactoryAdapter` will create programs for ALL tensor coordinates, ignoring `mesh_coords`. Only factories that implement `MeshWorkloadFactoryConcept` (and NOT `ProgramFactoryConcept`) will use the direct path that allows coordinate filtering.

---

## File Structure

After migration, your operation should have this structure:

```
ttnn/cpp/ttnn/operations/<operation>/
├── device/
│   ├── <operation>_device_operation.hpp      # Main device operation struct
│   ├── <operation>_device_operation.cpp      # Implementation
│   ├── <operation>_device_operation_types.hpp # operation_attributes_t, tensor_args_t, return types
│   ├── <operation>_program_factory.hpp       # Program factory structs
│   ├── <operation>_program_factory.cpp       # Program factory implementation
│   └── kernels/                              # Kernel files (if any)
├── <operation>.hpp                           # Public API wrapper
├── <operation>.cpp                           # Public API implementation
└── <operation>_pybind.cpp                    # Python bindings (if any)
```
