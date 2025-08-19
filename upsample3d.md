# Detailed Implementation Plan for 3D Upsample Operation in TTNN

## Overview
Implement a 3D upsample operation (`upsample3d`) in TTNN that supports nearest-neighbor interpolation for 5D tensors (N, D, H, W, C) in row-major DRAM interleaved layout.

## 1. Core Tensor Shape Requirements

**Input tensor shape**: `[N, D, H, W, C]`
- N = batch size
- D = depth dimension
- H = height dimension
- W = width dimension
- C = channels

**Output tensor shape**: `[N, D*scale_d, H*scale_h, W*scale_w, C]`
- Each dimension is scaled by its respective scale factor
- For nearest neighbor: each input voxel is replicated `scale_d × scale_h × scale_w` times

## 2. File Structure and Components to Create

### 2.1 Core Operation Files
```
ttnn/cpp/ttnn/operations/pool/upsample3d/
├── CMakeLists.txt                   # Build configuration
├── upsample3d.hpp                    # Public API interface
├── upsample3d.cpp                    # Implementation logic
├── upsample3d_pybind.cpp            # Python bindings
└── device/
    ├── upsample3d_op.hpp            # Device operation header
    ├── upsample3d_op.cpp            # Device operation implementation
    ├── upsample3d_program_factory_multicore_interleaved.cpp  # Main factory
    └── kernels/
        └── dataflow/
            ├── reader_upsample3d_interleaved.cpp    # Reader kernel
            └── writer_upsample3d_interleaved.cpp    # Writer kernel
```

### 2.2 Integration Files to Modify

**Build System Integration:**
- Update `ttnn/cpp/ttnn/operations/pool/CMakeLists.txt` to add upsample3d source files
- Update `ttnn/CMakeLists.txt` to include upsample3d pybind files

**Python Bindings Integration:**
- Update `ttnn/cpp/ttnn-pybind/__init__.cpp` to include upsample3d pybind header and call py_module function
- Create `ttnn/python/ttnn/operations/pool/upsample3d.py` wrapper for Python API

**C++ Header Integration:**
- Add upsample3d operation to appropriate ttnn header includes for C++ usage

## 3. Implementation Details

### 3.1 UpSample3D Operation Structure
```cpp
struct UpSample3D {
    const int scale_factor_d_;  // New: depth scale factor
    const int scale_factor_h_;
    const int scale_factor_w_;
    const std::string mode_;    // Only "nearest" initially
};
```

### 3.2 Memory Layout Strategy

**Work Unit Definition**:
- A "stick" is one page in memory containing `C` elements (the innermost dimension)
- For 5D tensor `[N, D, H, W, C]` in row-major layout, the tensor is stored as 2D with:
  - First dimension: `N × D × H × W` (flattened outer dimensions)
  - Second dimension: `C` (channels - one page per stick)
- Total work units = `N × D × H × W` sticks, each stick is one page distributed across memory banks
- Each work unit processes one input stick (C elements) and generates `scale_d × scale_h × scale_w` output sticks

**Memory Access Pattern**:
- Input: `[N, D, H, W, C]` → Physical volume = `N × D × H × W` pages of `C` elements each
- Output: `[N, D*scale_d, H*scale_h, W*scale_w, C]` → Each input page generates `scale_d × scale_h × scale_w` output pages
- Pages are distributed across memory banks in round-robin fashion

### 3.3 Kernel Implementation Strategy

**Reader Kernel**:
- Reads one input page at a time: `C` elements (one stick)
- Takes linear page index as input and reads corresponding page from interleaved DRAM
- No coordinate conversion needed - works directly with flattened page indices

**Writer Kernel**:
- For each input page, writes `scale_d × scale_h × scale_w` output pages
- Each output page contains the same `C` channel values as the input page
- Handles coordinate mapping from input `(n, d, h, w)` to all corresponding output positions:
  ```
  input_page[n×D×H×W + d×H×W + h×W + w] →
  output_page[n×D'×H'×W' + d'×H'×W' + h'×W' + w']
  where: d' ∈ [d×scale_d, (d+1)×scale_d)
         h' ∈ [h×scale_h, (h+1)×scale_h)
         w' ∈ [w×scale_w, (w+1)×scale_w)
  ```

### 3.4 Core Algorithm (Nearest Neighbor)

For input page at coordinate `(n, d, h, w)` containing `C` elements:
1. Calculate output page coordinates:
   - `d_start = d * scale_d`, `d_end = (d+1) * scale_d`
   - `h_start = h * scale_h`, `h_end = (h+1) * scale_h`
   - `w_start = w * scale_w`, `w_end = (w+1) * scale_w`

2. Replicate input page to all corresponding output pages:
   ```
   input_page_idx = n×D×H×W + d×H×W + h×W + w

   for d' in [d_start, d_end):
     for h' in [h_start, h_end):
       for w' in [w_start, w_end):
         output_page_idx = n×D'×H'×W' + d'×H'×W' + h'×W' + w'
         copy_page(input_page[input_page_idx], output_page[output_page_idx])
   ```

## 4. Minimal Implementation Strategy (Test-Driven Development)

### Step 1: Python API Stub + Test
**Implementation:**
- Create basic Python binding that just exists and is callable
- Return "not implemented" error for now

**Test:** `test_upsample3d_api_exists()`
**Build & Test:** Verify API exists and can be called

### Step 2: Parameter Validation Stub + Test
**Implementation:**
- Add parameter validation in Python binding
- Parse scale factors, validate tensor dimensionality
- Still return "not implemented"

**Test:** `test_upsample3d_parameter_validation()`
**Build & Test:** Verify parameter validation works

### Step 3: C++ Operation Registration + Test
**Implementation:**
- Create minimal `UpSample3D` struct with scale factors only
- Register operation in TTNN (stub invoke method)
- Return dummy tensor with correct output shape

**Test:** `test_upsample3d_operation_registered()`, `test_upsample3d_output_shape_computation()`
**Build & Test:** Verify operation is registered and shape computation works

### Step 4: Device Operation + Coordinate Logic + Test
**Implementation:**
- Add `validate()` method to `UpSample3D`
- Check 5D tensor, positive scale factors, "nearest" mode
- Implement coordinate mapping functions (no actual kernel execution)
- Add buffer size calculations
- Test mathematical correctness of coordinate transformations
- Stub `create_program()` method

**Test:** `test_upsample3d_input_validation()`, `test_upsample3d_struct_creation()`, `test_upsample3d_coordinate_mapping()`, `test_upsample3d_buffer_sizing()`
**Build & Test:** Verify validation logic and coordinate math works correctly

### Step 5: Minimal Program Factory + Test
**Implementation:**
- Create program factory that creates empty program
- Add dummy reader/writer kernel files that compile but do nothing
- Calculate work distribution correctly

**Test:** `test_upsample3d_program_creation()`, `test_upsample3d_work_distribution()`
**Build & Test:** Verify program creation and work splitting works

### Step 6: Kernel Compilation + Test
**Implementation:**
- Create actual reader/writer kernel files with real logic
- Implement circular buffer setup
- Add runtime argument setup

**Test:** `test_upsample3d_kernels_compile()`, `test_upsample3d_kernel_argument_setup()`
**Build & Test:** Verify kernels compile and link successfully

### Step 7: Minimal Working Implementation + Test
**Implementation:**
- Implement actual upsampling logic in kernels
- Start with very simple case (small tensors, scale factor 2)

**Test:** `test_upsample3d_small_tensor()`
**Build & Test:** Verify basic functionality works for simple cases

### Step 8: Full Implementation + Test
**Implementation:**
- Handle all scale factor combinations
- Optimize for larger tensors
- Add edge case handling

**Test:** `test_upsample3d_various_scale_factors()`, `test_upsample3d_coordinate_correctness()`
**Build & Test:** Verify full functionality against PyTorch reference

### Testing Protocol for Each Step:
1. **Write minimal test first** (if not already written)
2. **Implement minimal code** to pass the test
3. **Build the project** (`make` or equivalent)
4. **Run the specific test** for this step
5. **If test fails:** Debug and refine implementation (do NOT change test)
6. **Repeat build/test cycle** until test passes
7. **Only then move to next step**

### Key Principles:
- **Never skip a test** - each test must pass before moving forward
- **Implement only what's needed** to pass the current test
- **Build frequently** - catch compilation issues early
- **Trust the tests** - if test fails, fix implementation not test
- **One step at a time** - don't jump ahead even if you think you know what's needed

## 5. Key Technical Considerations

**Memory Requirements**:
- Output tensor size = `N × D×scale_d × H×scale_h × W×scale_w × C`
- Can be significantly larger than input (cubic growth with scale factors)

**Coordinate Mapping**:
- Page-based indexing: input page = `n×D×H×W + d×H×W + h×W + w`
- Output page = `n×D'×H'×W' + d'×H'×W' + h'×W' + w'` where `D'=D×scale_d`, `H'=H×scale_h`, `W'=W×scale_w`
- Each input page maps to `scale_d × scale_h × scale_w` output pages

**Work Distribution**:
- Distribute `N × D × H × W` work units (pages) across available cores
- Each core processes multiple input pages and generates corresponding output pages

**Validation Requirements**:
- Verify input tensor is 5D with expected shape
- Ensure scale factors are positive integers
- Validate only "nearest" mode initially

This implementation leverages the existing 2D upsample infrastructure while extending it to handle the additional depth dimension and 3D coordinate transformations required for 3D upsampling.

## 6. Call Stack Flow and Testing Points for Upsample3D

### Complete Call Flow
```
Python Test
    ↓
ttnn.upsample3d(input_tensor, (scale_d, scale_h, scale_w))  # Python API call
    ↓
upsample3d_pybind.cpp::bind_upsample3d()                     # Python binding
    ↓
ttnn::upsample3d (registered operation)                     # C++ operation registration
    ↓
upsample3d.cpp::ExecuteUpSample3D::invoke()                  # C++ implementation
    ↓
tt::tt_metal::operation::run(UpSample3D{...})               # TT-Metal operation runner
    ↓
upsample3d_op.cpp::UpSample3D::create_program()             # Device operation
    ↓
upsample3d_program_factory_multicore_interleaved()          # Program factory
    ↓
┌─────────────────────────────────────────────┐
│ Kernel Creation & Configuration             │
├─────────────────────────────────────────────┤
│ reader_upsample3d_interleaved.cpp           │  # Reader kernel
│ writer_upsample3d_interleaved.cpp           │  # Writer kernel
│ Work distribution across cores              │
│ Circular buffer setup                       │
│ Runtime argument configuration              │
└─────────────────────────────────────────────┘
    ↓
Device Execution (Hardware)
```

### Individual Testing Points

**1. Python API Layer (test_upsample3d.py)**
```python
# Test: Python function exists and accepts correct parameters
def test_upsample3d_api_exists():
    assert hasattr(ttnn, 'upsample3d')

# Test: Parameter validation at Python level
def test_upsample3d_parameter_validation():
    # Test invalid scale factors, wrong tensor dimensions, etc.
```

**2. Python Binding Layer (upsample3d_pybind.cpp)**
```cpp
// Test: C++ binding compilation and parameter forwarding
def test_upsample3d_pybind_compilation():
    # Verify that pybind module compiles and links correctly

def test_upsample3d_argument_parsing():
    # Test that Python arguments are correctly parsed to C++ types
```

**3. Operation Registration (upsample3d.hpp)**
```cpp
// Test: Operation is properly registered in TTNN
def test_upsample3d_operation_registered():
    # Verify ttnn::upsample3d exists and is callable
```

**4. C++ Implementation Layer (upsample3d.cpp)**
```cpp
// Test: ExecuteUpSample3D::invoke logic
def test_upsample3d_invoke():
    # Test scale factor parsing from variant<int, Array3D>
    # Test memory config handling
    # Test parameter validation

// Test: Shape computation without execution
def test_upsample3d_output_shape_computation():
    input_shape = [2, 4, 8, 16, 32]  # N, D, H, W, C
    scale_factors = (2, 3, 4)        # scale_d, scale_h, scale_w
    expected_output = [2, 8, 24, 64, 32]
    # Verify compute_output_specs() returns correct shape
```

**5. Device Operation Layer (upsample3d_op.cpp)**
```cpp
// Test: UpSample3D struct and validation
def test_upsample3d_struct_creation():
    op = UpSample3D{2, 3, 4, "nearest"}
    # Test struct creation and member access

def test_upsample3d_input_validation():
    # Test validate() method with various invalid inputs:
    # - Non-5D tensors
    # - Invalid modes
    # - Zero/negative scale factors
```

**6. Program Factory Layer (upsample3d_program_factory_multicore_interleaved.cpp)**
```cpp
// Test: Program creation without execution
def test_upsample3d_program_creation():
    # Create program object, verify kernels are created
    # Test work splitting logic
    # Test circular buffer sizing

def test_upsample3d_work_distribution():
    # Test various input sizes and core counts
    # Verify work is evenly distributed
    # Test edge cases (more cores than work units)

def test_upsample3d_coordinate_mapping():
    input_coord = (1, 2, 3, 4)  # n=1, d=2, h=3, w=4
    expected_output_coords = [...]  # All corresponding output coordinates
    # Test mapping function without actual kernel execution
```

**7. Kernel Compilation Layer**
```cpp
// Test: Kernel compilation and linking
def test_upsample3d_kernels_compile():
    # Verify reader and writer kernels compile successfully
    # Test with different compile-time arguments

def test_upsample3d_kernel_argument_setup():
    # Test runtime argument calculation
    # Verify buffer addresses, work counts, start indices
```

**8. Memory and Buffer Management**
```cpp
// Test: Buffer allocation and page calculations
def test_upsample3d_buffer_sizing():
    # Test input/output buffer size calculations
    # Verify page count computations

def test_upsample3d_circular_buffer_config():
    # Test CB sizing for different input shapes
    # Verify alignment requirements
```

**9. End-to-End Functional Tests (PyTorch Reference Validation)**
```python
def test_upsample3d_small_tensor():
    # Create 5D input tensor [N, D, H, W, C]
    input_tensor = create_5d_tensor([1, 2, 2, 2, 4])
    scale_factors = (2, 2, 2)  # scale_d, scale_h, scale_w

    # TTNN result
    ttnn_result = ttnn.upsample3d(input_tensor, scale_factors)

    # PyTorch reference (need to convert NDHWC -> NCDHW for PyTorch)
    torch_input = input_tensor.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
    torch_result = F.interpolate(torch_input, scale_factor=scale_factors, mode='nearest')
    torch_result = torch_result.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

    # Compare results
    assert torch.equal(ttnn_result, torch_result)

def test_upsample3d_coordinate_correctness():
    # Create input with unique values at each (d,h,w) position
    input_tensor = create_unique_valued_5d_tensor([1, 2, 3, 4, 8])
    scale_factors = (2, 3, 2)

    ttnn_result = ttnn.upsample3d(input_tensor, scale_factors)

    # Verify each output position contains correct input value using nearest neighbor logic
    for n in range(1):
        for d_out in range(4):  # 2*2
            for h_out in range(9):  # 3*3
                for w_out in range(8):  # 4*2
                    # Calculate source coordinates for nearest neighbor
                    d_in = d_out // 2
                    h_in = h_out // 3
                    w_in = w_out // 2

                    expected_value = input_tensor[n, d_in, h_in, w_in, :]
                    actual_value = ttnn_result[n, d_out, h_out, w_out, :]
                    assert torch.equal(expected_value, actual_value)

def test_upsample3d_various_scale_factors():
    test_cases = [
        ([1, 2, 3, 4, 8], (1, 1, 1)),    # No scaling
        ([1, 2, 3, 4, 8], (2, 1, 3)),    # Mixed scale factors
        ([1, 3, 2, 2, 4], (3, 4, 2)),    # Different combinations
        ([2, 1, 4, 6, 16], (1, 2, 1)),   # Batch size > 1
    ]

    for input_shape, scale_factors in test_cases:
        input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

        # TTNN result
        ttnn_result = ttnn.upsample3d(input_tensor, scale_factors)

        # PyTorch reference
        torch_input = input_tensor.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
        torch_result = F.interpolate(torch_input, scale_factor=scale_factors, mode='nearest')
        torch_result = torch_result.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

        # Validate results match
        pcc_passed, pcc_message = assert_with_pcc(torch_result, ttnn_result, pcc=0.99999)
        assert torch.equal(ttnn_result, torch_result), f"Failed for shape {input_shape}, scales {scale_factors}"
```

This layered testing approach allows verification of each component independently, making debugging much easier and ensuring each layer works correctly before moving to the next level.
