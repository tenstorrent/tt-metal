# TTNN Operation Parameter Tracing

## Overview

The TTNN Operation Parameter Tracing feature allows you to serialize all `ttnn` operation parameters and return values into human-readable JSON files. This is extremely useful for debugging, analyzing operation behavior, and understanding data flow through your models.

When enabled, every `ttnn` operation call (such as `ttnn.add`, `ttnn.matmul`, `ttnn.from_torch`, etc.) will automatically:
- Capture all input arguments (positional and keyword)
- Capture the return value
- Serialize tensor metadata (shape, dtype, layout, storage_type) into JSON
- Optionally serialize tensor values (disabled by default to reduce file size)
- Save everything to a uniquely named JSON file

## Features

- **Automatic Tracing**: No code changes required - just enable via command-line flag
- **Complete Data Capture**: Captures both `ttnn.Tensor` and `torch.Tensor` objects
- **Human-Readable Format**: All data is serialized as JSON
- **Efficient Default**: By default, only tensor metadata is serialized (shape, dtype, etc.) - not values
- **Optional Value Serialization**: Can enable full tensor value serialization when needed
- **Sequential Numbering**: Operations are numbered sequentially for easy tracking
- **Timestamped Files**: Each trace file includes a timestamp for chronological ordering
- **Zero Performance Impact**: When disabled, tracing has zero overhead (single boolean check)

## Quick Start

### Basic Usage

Enable tracing by adding the `--trace-params` flag to your pytest command:

```bash
pytest tests/ttnn/unit_tests/base_functionality/test_python_tracer.py -s --trace-params
```

### Running Your Own Code

You can also enable tracing in your own Python scripts:

```python
import ttnn
import ttnn.operation_tracer

# Enable tracing
ttnn.operation_tracer._ENABLE_TRACE = True

# Optionally set a custom trace directory
ttnn.CONFIG.root_report_path = "/path/to/trace/output"

# Your ttnn operations will now be traced
device = ttnn.open_device(device_id=0)
a = ttnn.rand(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.rand(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
c = ttnn.add(a, b)
```

## Command-Line Flags

### `--trace-params`

**Purpose**: Enables operation parameter tracing

**Usage**:
```bash
pytest <test_file> --trace-params
```

**Behavior**:
- Automatically enables tracing for all `ttnn` operations
- By default, only tensor metadata is serialized (shape, dtype, layout, storage_type) - **not tensor values**
- Trace files are saved to `{root_report_path}/operation_parameters/`
- If `root_report_path` is not set, defaults to `generated/ttnn/operation_parameters/`
- Trace files are automatically cleaned up after tests complete

### `--trace-params-with-values`

**Purpose**: Enables full tensor value serialization (in addition to metadata)

**Usage**:
```bash
pytest <test_file> --trace-params --trace-params-with-values
```

**Behavior**:
- Must be used together with `--trace-params`
- Serializes complete tensor values in addition to metadata
- **Warning**: This will significantly increase file size and serialization time for large tensors
- Use only when you need to inspect actual tensor values for debugging

## Trace File Format

### File Naming Convention

Trace files follow this naming pattern:
```
{operation_number}_{operation_name}_{timestamp}.json
```

**Examples**:
- `1_ttnn_rand_20260115_104616_123456.json`
- `2_ttnn_rand_20260115_104616_234567.json`
- `3_ttnn_add_20260115_104616_345678.json`

Where:
- `{operation_number}`: Sequential number starting from 1
- `{operation_name}`: Fully qualified operation name (dots replaced with underscores)
- `{timestamp}`: Format: `YYYYMMDD_HHMMSS_microseconds`

### JSON Structure

Each trace file contains a JSON object with the following structure. Note that the `values` field is only included when `--trace-params-with-values` is used:

**Default (metadata only)**:
```json
{
  "operation_number": 3,
  "operation_name": "ttnn.add",
  "timestamp": "20260115_104616_345678",
  "args": [
    {
      "position": 0,
      "value": {
        "type": "ttnn.Tensor",
        "shape": [2, 3],
        "dtype": "bfloat16",
        "original_shape": [2, 3],
        "original_dtype": "bfloat16",
        "layout": "TILE",
        "storage_type": "DEVICE_STORAGE"
      }
    },
    {
      "position": 1,
      "value": {
        "type": "ttnn.Tensor",
        "shape": [2, 3],
        "dtype": "bfloat16"
      }
    }
  ],
  "kwargs": {
    "dtype": null,
    "memory_config": null
  },
  "num_tensors": 2,
  "return_value": {
    "type": "ttnn.Tensor",
    "shape": [2, 3],
    "dtype": "bfloat16"
  }
}
```

**With values** (when `--trace-params-with-values` is used):
```json
{
  "operation_number": 3,
  "operation_name": "ttnn.add",
  "timestamp": "20260115_104616_345678",
  "args": [
    {
      "position": 0,
      "value": {
        "type": "ttnn.Tensor",
        "shape": [2, 3],
        "dtype": "float32",
        "values": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "original_shape": [2, 3],
        "original_dtype": "bfloat16",
        "layout": "TILE",
        "storage_type": "DEVICE_STORAGE"
      }
    },
    {
      "position": 1,
      "value": {
        "type": "ttnn.Tensor",
        "shape": [2, 3],
        "dtype": "float32",
        "values": [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
      }
    }
  ],
  "kwargs": {
    "dtype": null,
    "memory_config": null
  },
  "num_tensors": 2,
  "return_value": {
    "type": "ttnn.Tensor",
    "shape": [2, 3],
    "dtype": "float32",
    "values": [[0.8, 1.0, 1.2], [1.4, 1.6, 1.8]]
  }
}
```

### Field Descriptions

- **`operation_number`**: Sequential number of the operation (1, 2, 3, ...)
- **`operation_name`**: Fully qualified name of the operation (e.g., `"ttnn.add"`, `"ttnn.from_torch"`)
- **`timestamp`**: When the operation was executed
- **`args`**: Array of positional arguments
  - Each element has `position` (0-indexed) and `value` (serialized argument)
- **`kwargs`**: Dictionary of keyword arguments
  - Keys are argument names, values are serialized arguments
- **`num_tensors`**: Total count of tensor objects (both `ttnn.Tensor` and `torch.Tensor`) in args, kwargs, and return value
- **`return_value`**: Serialized return value (if the operation returns a value)

### Tensor Serialization

Tensors are serialized with the following fields. **By default, only metadata is included** (shape, dtype, layout, storage_type). The `values` field is only included when `--trace-params-with-values` is used.

#### `ttnn.Tensor` Objects

**Default (metadata only)**:
```json
{
  "type": "ttnn.Tensor",
  "shape": [2, 3],
  "dtype": "bfloat16",
  "original_shape": [2, 3],
  "original_dtype": "bfloat16",
  "layout": "TILE",
  "storage_type": "DEVICE_STORAGE"
}
```

**With values** (when `--trace-params-with-values` is used):
```json
{
  "type": "ttnn.Tensor",
  "shape": [2, 3],
  "dtype": "float32",
  "values": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
  "original_shape": [2, 3],
  "original_dtype": "bfloat16",
  "layout": "TILE",
  "storage_type": "DEVICE_STORAGE"
}
```

- **`type`**: Always `"ttnn.Tensor"`
- **`shape`**: Shape of the tensor as a list
- **`dtype`**: Data type as a string
- **`values`**: (Optional) Nested list containing all tensor values - only included when `--trace-params-with-values` is used
- **`original_shape`**: Original shape before any conversions (if available)
- **`original_dtype`**: Original dtype before conversion (if available)
- **`layout`**: Tensor layout (e.g., `"TILE"`, `"ROW_MAJOR"`) - if available
- **`storage_type`**: Storage type (e.g., `"DEVICE_STORAGE"`, `"HOST_STORAGE"`) - if available

#### `torch.Tensor` Objects

**Default (metadata only)**:
```json
{
  "type": "torch.Tensor",
  "shape": [2, 3],
  "dtype": "bfloat16"
}
```

**With values** (when `--trace-params-with-values` is used):
```json
{
  "type": "torch.Tensor",
  "shape": [2, 3],
  "dtype": "float32",
  "values": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
}
```

- **`type`**: Always `"torch.Tensor"`
- **`shape`**: Shape of the tensor as a list
- **`dtype`**: Data type as a string
- **`values`**: (Optional) Nested list containing all tensor values - only included when `--trace-params-with-values` is used

**Note**: When `--trace-params-with-values` is used, `bfloat16` tensors are automatically converted to `float32` for JSON serialization, as NumPy does not natively support `bfloat16`.

## Trace File Location

Trace files are saved to:
```
{root_report_path}/operation_parameters/
```

Where `root_report_path` is determined in this order:
1. `ttnn.CONFIG.operation_parameter_log_dir` (if set)
2. `ttnn.CONFIG.root_report_path` (if set)
3. Default: `generated/ttnn/operation_parameters/`

### Setting Custom Trace Directory

#### In Python Code

```python
import ttnn

# Set custom trace directory
ttnn.CONFIG.root_report_path = "/path/to/my/traces"
# Or specifically:
ttnn.CONFIG.operation_parameter_log_dir = "/path/to/my/traces"
```

#### In Tests (using pytest tmp_path)

```python
def test_my_operation(tmp_path):
    # Override the log directory
    original_path = ttnn.CONFIG.root_report_path
    ttnn.CONFIG.root_report_path = str(tmp_path)

    # Your operations here...

    # Restore original path
    ttnn.CONFIG.root_report_path = original_path
```

## Supported Operations

Tracing works for **all** `ttnn` operations, including:

- **Element-wise operations**: `ttnn.add`, `ttnn.subtract`, `ttnn.multiply`, etc.
- **Tensor creation**: `ttnn.rand`, `ttnn.zeros`, `ttnn.ones`, `ttnn.full`, etc.
- **Data movement**: `ttnn.from_torch`, `ttnn.to_device`, `ttnn.from_device`, etc.
- **Linear algebra**: `ttnn.matmul`, `ttnn.linear`, etc.
- **Convolutions**: `ttnn.conv2d`, etc.
- **Reductions**: `ttnn.sum`, `ttnn.max`, etc.
- **And many more...**

## Examples

**Note**: Actual trace file examples are available in the `examples/` folder in this directory. These JSON files demonstrate the exact format and structure of trace files generated by the tracing system.

### Example 1: Basic Addition

```python
import ttnn
import ttnn.operation_tracer

# Enable tracing
ttnn.operation_tracer._ENABLE_TRACE = True

device = ttnn.open_device(device_id=0)
a = ttnn.rand(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.rand(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
c = ttnn.add(a, b)
```

This will create 3 trace files:
1. `1_ttnn_rand_*.json` - First `rand` operation
2. `2_ttnn_rand_*.json` - Second `rand` operation
3. `3_ttnn_add_*.json` - `add` operation (with both input tensors and output tensor)

### Example 2: from_torch and to_device

```python
import torch
import ttnn
import ttnn.operation_tracer

# Enable tracing
ttnn.operation_tracer._ENABLE_TRACE = True

device = ttnn.open_device(device_id=0)
torch_tensor = torch.randn([2, 3], dtype=torch.bfloat16)
tensor_host = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
tensor_device = ttnn.to_device(tensor_host, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

This will create 2 trace files:
1. `1_ttnn_from_torch_*.json` - Contains the input `torch.Tensor` and output `ttnn.Tensor`
2. `2_ttnn_to_device_*.json` - Contains the input `ttnn.Tensor` and output `ttnn.Tensor`

### Example 3: Running Tests with Tracing

```bash
# Run a specific test with tracing enabled (metadata only - default)
pytest tests/ttnn/unit_tests/base_functionality/test_python_tracer.py::test_operation_parameter_tracing -s --trace-params

# Run with full tensor value serialization
pytest tests/ttnn/unit_tests/base_functionality/test_python_tracer.py::test_operation_parameter_tracing -s --trace-params --trace-params-with-values

# Run all tests in a file with tracing
pytest tests/ttnn/unit_tests/base_functionality/test_python_tracer.py -s --trace-params
```

## Performance Considerations

- **When Disabled**: Tracing has **zero performance impact** - only a single boolean check per operation
- **When Enabled (metadata only - default)**:
  - Operations are wrapped but the check is still very cheap (single boolean)
  - Serialization happens **after** the operation completes, so it doesn't block execution
  - Only metadata is serialized (shape, dtype, etc.) - very fast and produces small files
  - Minimal overhead, suitable for production debugging
- **When Enabled (with values)**:
  - Tensor data is moved to CPU and converted to numpy, which adds significant overhead
  - Large tensors will result in very large JSON files
  - Serialization time scales with tensor size
  - **Not recommended for production use**

**Recommendation**:
- Use `--trace-params` (metadata only) for general debugging and production analysis
- Use `--trace-params --trace-params-with-values` only when you specifically need to inspect tensor values
- Do not enable value serialization in performance-critical paths

## Limitations

1. **Large Tensors (with values)**: When using `--trace-params-with-values`, very large tensors will create very large JSON files. Consider the disk space implications. The default (metadata only) avoids this issue.

2. **Data Type Conversions** (with values only):
   - Some precision may be lost when converting `bfloat16` to `float32`
   - Only relevant when `--trace-params-with-values` is used

3. **Device Tensors** (with values only):
   - Device tensors are moved to CPU before value serialization, which may change their representation slightly
   - Only relevant when `--trace-params-with-values` is used

4. **Memory Configurations**: Memory configuration information is not fully serialized (only basic tensor metadata).

## Troubleshooting

### No trace files are created

1. **Check that tracing is enabled**:
   ```python
   import ttnn.operation_tracer
   print(ttnn.operation_tracer._ENABLE_TRACE)  # Should be True
   ```

2. **Check the trace directory**:
   ```python
   print(ttnn.CONFIG.root_report_path)  # Should point to your desired directory
   ```

3. **Verify operations are being called**: Make sure you're actually calling `ttnn` operations after enabling tracing.

### Trace files are empty or incomplete

- This usually indicates an error during serialization. Check the logs for any error messages.
- Some tensor types may not be fully supported. Check the tensor's `dtype` and `storage_type`.

### Recursion errors

- The tracer includes recursion guards, but if you see recursion errors, it may indicate a bug in the serialization logic. Please report this as an issue.

### Performance issues

- If tracing is causing significant slowdown, consider:
  - Only tracing specific operations (by conditionally enabling/disabling)
  - Using smaller test tensors
  - Tracing only when necessary (not in production code)

## Implementation Details

### Architecture

The tracing functionality is implemented in `ttnn/ttnn/operation_tracer.py`:

- **`wrap_function_for_tracing()`**: Wraps operations to inject tracing logic
- **`serialize_operation_parameters()`**: Serializes operation data to JSON
- **`serialize_value()`**: Recursively serializes values, handling tensors specially

Operations are wrapped at registration time in `ttnn/ttnn/decorators.py`:
- `register_cpp_operation()`: Wraps C++ operations
- `register_python_operation()`: Wraps Python operations

### Global State

The tracer uses module-level global variables:
- `_ENABLE_TRACE`: Boolean flag to enable/disable tracing
- `_SERIALIZE_TENSOR_VALUES`: Boolean flag to control whether tensor values are serialized (default: `False`)
- `_OPERATION_COUNTER`: Sequential counter for operation numbering
- `_IS_SERIALIZING`: Flag to prevent recursion during serialization
- `_TRACE_PARAMS_IN_ARGV`: Cached check for `--trace-params` flag

### Integration with Pytest

The `conftest.py` file adds pytest command-line options:
- `--trace-params`: Enables tracing (metadata only by default)
- `--trace-params-with-values`: Enables full tensor value serialization (must be used with `--trace-params`)

When `--trace-params` is used, `pytest_configure()` sets `ttnn.operation_tracer._ENABLE_TRACE = True`.
When `--trace-params-with-values` is used, tensor values are also serialized.

## See Also

- **Test File**: `tests/ttnn/unit_tests/base_functionality/test_python_tracer.py` - Contains comprehensive examples
- **Implementation**: `ttnn/ttnn/operation_tracer.py` - Core tracing logic
- **Integration**: `ttnn/ttnn/decorators.py` - Operation wrapping
- **Pytest Integration**: `conftest.py` - Command-line flags

## Contributing

If you encounter issues or have suggestions for improving the tracing functionality, please:
1. Check existing issues
2. Create a new issue with:
   - Description of the problem or enhancement
   - Steps to reproduce (if applicable)
   - Expected vs. actual behavior
   - Relevant trace files (if applicable)
