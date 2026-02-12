# Onboarding Workshop

Workshop for kernel/op development in tt-metal.

## Architecture

```
Python API (ttnn.matmul_add)
    ↓
C++ Operation (invoke → device_operation::launch)
    ↓
Device Operation (create buffers, setup program)
    ↓
Program Factory (create CBs, configure kernels)
    ↓
Kernels (reader → compute → writer)
```

## Key Patterns

### 1. Operation Registration (e03)

```cpp
// Header: define struct with invoke()
struct MatmulAdd {
    static Tensor invoke(const Tensor& a, const Tensor& b, const Tensor& c);
};

// Register in ttnn namespace
namespace ttnn {
constexpr auto matmul_add = register_operation<"ttnn::matmul_add", MatmulAdd>();
}
```

### 2. Python Bindings (e03)

```cpp
NB_MODULE(_module_name, mod) {
    ttnn::bind_registered_operation(
        mod,
        ttnn::matmul_add,
        R"doc(Docstring)doc",
        ttnn::nanobind_overload_t{
            [](const decltype(ttnn::matmul_add)& self,
               const Tensor& a, const Tensor& b, const Tensor& c) {
                return self(a, b, c);
            },
            nb::arg("a"), nb::arg("b"), nb::arg("c")});
}
```

### 3. Device Operation (e04)

```cpp
struct MatmulAddOperation {
    struct operation_attributes_t {};  // Compile-time config
    struct tensor_args_t { Tensor a, b, c; };  // Input tensors

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        struct shared_variables_t { /* cached data */ };
        static cached_program_t create(...);
        static void override_runtime_arguments(...);
    };

    static program_factory_t select_program_factory(...);
    static void validate_on_program_cache_miss(...);
    static spec_return_value_t compute_output_specs(...);
    static tensor_return_value_t create_output_tensors(...);
};
```

### 4. Kernel Pattern (e04)

**Reader** (RISCV_1 - data movement):
```cpp
void kernel_main() {
    // Get runtime args
    auto a_addr = get_arg_val<uint32_t>(0);

    // Wait for CB space, read from DRAM, push to CB
    cb_reserve_back(cb_a, 1);
    noc_async_read(...);
    noc_async_read_barrier();
    cb_push_back(cb_a, 1);
}
```

**Compute** (TRISC - math):
```cpp
void kernel_main() {
    // Init compute
    mm_init(cb_a, cb_b, cb_out);

    // Wait for data, compute, push result
    tile_regs_acquire();
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    matmul_tiles(cb_a, cb_b, 0, 0, 0);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    tile_regs_commit();

    // Pack output
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
```

**Writer** (RISCV_0 - data movement):
```cpp
void kernel_main() {
    // Wait for compute result, write to DRAM
    cb_wait_front(cb_out, 1);
    noc_async_write(...);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
```

## Key APIs

### Circular Buffers
- `cb_reserve_back(cb, n)` - Reserve n tiles for writing
- `cb_push_back(cb, n)` - Signal n tiles written
- `cb_wait_front(cb, n)` - Wait for n tiles available
- `cb_pop_front(cb, n)` - Release n tiles after reading

### Tile Registers (16 available)
- `tile_regs_acquire()` - Acquire tile registers
- `tile_regs_commit()` - Signal compute done
- `tile_regs_wait()` - Wait for commit
- `tile_regs_release()` - Release registers

### NOC (Network on Chip)
- `noc_async_read(src_addr, dst_local, size)` - DRAM → L1
- `noc_async_write(src_local, dst_addr, size)` - L1 → DRAM
- `noc_async_read_barrier()` / `noc_async_write_barrier()` - Wait for completion

### Compute Operations
- `matmul_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx)` - Matrix multiply
- `add_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx)` - Element-wise add
- `pack_tile(src_idx, cb_out)` - Pack from register to CB

## Common Pitfalls

1. **Forgetting barriers** - Always call `noc_async_*_barrier()` after NOC operations
2. **CB deadlock** - Ensure reader/compute/writer stay synchronized
3. **Tile register exhaustion** - Only 16 registers, acquire/release properly
4. **Re-init after switching ops** - Call `mm_init_short()` after binary ops in matmul loop

## Build Commands

```bash
./build_metal.sh                           # Build tt-metal
cmake --build build -- onboarding          # Build exercises
cmake --build build -- onboarding-clean    # Clean exercises
ttnn/tutorials/onboarding/run.sh "e04 and solution"  # Test
```

## Reference Code

- `ttnn/cpp/ttnn/operations/eltwise/binary/` - Simple operation pattern
- `ttnn/cpp/ttnn/operations/matmul/` - Complex device operation
- `tt_metal/programming_examples/matmul/` - Kernel examples
