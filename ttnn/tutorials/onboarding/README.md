# Onboarding Workshop

Learn kernel/op development in tt-metal through hands-on exercises.

## How It Works

Each exercise has:
- `reference` - PyTorch implementation (the "what")
- `exercise` - Your implementation (fill in the blanks)
- `solution` - Working implementation (peek if stuck)
- `test` - Validates your code against reference

For C++ exercises (e03+), you'll find `exercise_cpp/` and `solution_cpp/` folders.

## Workflow

1. **Learn** - Read the exercise README (goal + references to study)
2. **Code** - Fill in `exercise` (and `exercise_cpp/` for C++)
3. **Test** - `run.sh "eXX and exercise"`

**Hints:** `reference` shows expected behavior, `solution` has the answer.

## Build Commands

All commands run from project root (`tt-metal/`):

```bash
# Build tt-metal (required once)
./build_metal.sh

# Build onboarding exercises (required for e03+)
cmake --build build -- onboarding

# To clean when iterating on C++ code
cmake --build build -- onboarding-clean

# Test exercise
./ttnn/tutorials/onboarding/run.sh "e02 and exercise"
# Test solution
./ttnn/tutorials/onboarding/run.sh "e02 and solution"
```

## Curriculum

### Fundamentals (e01-e04)

| # | Name | Goal |
|---|------|------|
| [e01](e01_build_run/) | build_run | Build tt-metal, verify setup |
| [e02](e02_ttnn_basics/) | ttnn_basics | Get familiar with ttnn API, pytest |
| [e03](e03_python_bindings/) | python_bindings | Operation registration + Python bindings |
| [e04](e04_matmul_add/) | matmul_add | Write custom kernels (core exercise) |

### Debugging & Profiling (e05-e06)

| # | Name | Goal |
|---|------|------|
| [e05](e05_debugging/) | debugging | Debug kernels with DPRINT and tt-triage |
| [e06](e06_profiling/) | profiling | Calculate theoretical peak, measure with Tracy |

### Memory & Data Layout (e07-e09)

| # | Name | Goal |
|---|------|------|
| [e07](e07_l1_vs_dram/) | l1_vs_dram | Memory hierarchy |
| [e08](e08_tile_layout/) | tile_layout | Tiled vs row major |
| [e09](e09_sharding/) | sharding | Sharded memory layout |

### Scaling (e10-e11)

| # | Name | Goal |
|---|------|------|
| [e10](e10_multi_core/) | multi_core | Work splitting, multicast, semaphores |
| [e11](e11_multi_chip/) | multi_chip | Multi-device programming |

### Advanced Optimization (e12-e14)

| # | Name | Goal |
|---|------|------|
| [e12](e12_matmul/) | matmul | Deep dive into matmul variants |
| [e13](e13_kernel_fusion/) | kernel_fusion | Fuse CCLs into matmul |
| [e14](e14_pipelining/) | pipelining | Double buffering |
