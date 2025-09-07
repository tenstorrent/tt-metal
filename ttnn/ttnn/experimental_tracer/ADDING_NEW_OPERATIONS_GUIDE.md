# Adding New Operations to Unit Test Generation Framework

This guide provides step-by-step instructions for adding support for new operations to the PyTorch unit test generation framework.

## Overview

The unit test generation framework consists of several components that work together:
- **Individual Unittest Classes**: Handle single operation instances
- **Group Unittest Classes**: Combine multiple operations of the same type for batch testing
- **Combiner Classes**: Define how operations are fused together
- **Configuration**: Registers operations and their combiners

## Step-by-Step Process

### Step 1: Add Required Imports

First, ensure the operation's backend class is imported from `tracer_backend_utils`:

```python
from tracer_backend_utils import (
    # ... existing imports
    AtenYourNewOperation,  # Add your new operation import
)
```

**Example from our recent work:**
```python
from tracer_backend_utils import (
    ConvAttrs,
    PoolAttrs,
    AtenConvolution,
    AtenAddm,
    # ... other imports
    AtenSigmoid,           # ← Added
    AtenSoftmax,           # ← Added
    AtenView,              # ← Added
    AtenPermute,           # ← Added
)
```

### Step 2: Create Individual Unittest Class

Create a class that inherits from `UnitTestOperation` and handles single operation instances:

```python
class YourOperationUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["YourOperationUnittest"]:
        if operation.function_call_name == "torch.ops.aten.your_operation":
            your_op = operation.to_operation(AtenYourOperation)
            return YourOperationUnittest(your_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this operation unit test."""
        group_unit_test = YourOperationGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()
```

**Real Example - SigmoidUnittest:**
```python
class SigmoidUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]]):
        self.input_shapes = input_shapes
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    @staticmethod
    def parse_from_operation(operation: Operation) -> Optional["SigmoidUnittest"]:
        if operation.function_call_name == "torch.ops.aten.sigmoid":
            sigmoid_op = operation.to_operation(AtenSigmoid)
            return SigmoidUnittest(sigmoid_op.input_shapes)
        return None

    def generate_code(self, indent="") -> str:
        """Generate the code for this sigmoid unit test operation."""
        group_unit_test = SigmoidGroupUnittest([self.input_shapes])
        return group_unit_test.generate_code()
```

### Step 3: Create Group Unittest Class

Create a class that handles multiple operations of the same type for batch testing:

```python
class YourOperationGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)}
            for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this operation unit test."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_your_operation(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.your_operation(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.your_operation(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""
```

**Real Example - SigmoidGroupUnittest:**
```python
class SigmoidGroupUnittest(UnitTestOperation):
    def __init__(self, input_shapes_list: List[Optional[Dict[int, Any]]]):
        self.input_shape_list = [shapes for shapes in input_shapes_list if shapes is not None]
        self.input_shape_list = [shapes for shapes in self.input_shape_list if len(shapes) >= 1]
        self.input_shape_list = [
            {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)} for shapes in self.input_shape_list
        ]
        HEADER_IMPORTS.add("from tests.ttnn.utils_for_testing import assert_with_pcc")

    def generate_code(self) -> str:
        """Generate the code for this sigmoid unit test operation."""
        return f"""

@pytest.mark.parametrize(
    "input_shape",
    (
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
    )
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_sigmoid(device, input_shape, dtype, layout):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.sigmoid(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.sigmoid(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
"""
```

### Step 4: Create Combiner Class

Create a combiner class that defines how multiple operations are fused together:

```python
class YourOperationCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple your_operation operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            op.input_shapes for op in operations if isinstance(op, YourOperationUnittest)
        ]
        return YourOperationGroupUnittest(combined_shapes)
```

**Real Example - SigmoidCombiner:**
```python
class SigmoidCombiner(UnitTestOperationCombiner):
    @staticmethod
    def combine(operations: List[UnitTestOperation]) -> UnitTestOperation:
        """Combine multiple sigmoid operations into a single one."""
        if not operations:
            raise ValueError("No operations to combine.")

        combined_shapes = [
            sigmoid_op.input_shapes for sigmoid_op in operations if isinstance(sigmoid_op, SigmoidUnittest)
        ]
        return SigmoidGroupUnittest(combined_shapes)
```

### Step 5: Update Configuration

Add your new operation to both configuration lists in `PytorchLayerUnitTestGraphConfig.__post_init__()`:

```python
def __post_init__(self):
    if self.register_unit_test_operations is None:
        self.register_unit_test_operations = [
            # ... existing operations
            YourOperationUnittest,  # ← Add here
        ]
    if self.group_unit_test_operations is None:
        self.group_unit_test_operations = {
            # ... existing mappings
            YourOperationUnittest: YourOperationCombiner,  # ← Add here
        }
```

**Real Example:**
```python
def __post_init__(self):
    if self.register_unit_test_operations is None:
        self.register_unit_test_operations = [
            ConvolutionUnittest,
            AddmUnittest,
            # ... other operations
            SigmoidUnittest,           # ← Added
            SoftmaxUnittest,           # ← Added
            ViewUnittest,              # ← Added
            # ... more operations
        ]
    if self.group_unit_test_operations is None:
        self.group_unit_test_operations = {
            ConvolutionUnittest: ConvolutionCombiner,
            AddmUnittest: AddmCombiner,
            # ... other mappings
            SigmoidUnittest: SigmoidCombiner,         # ← Added
            SoftmaxUnittest: SoftmaxCombiner,         # ← Added
            ViewUnittest: ViewCombiner,               # ← Added
            # ... more mappings
        }
```

## Key Design Patterns

### 1. Shape Processing
Most operations need to process and normalize input shapes:
```python
# Convert torch.Size to list and normalize to 4D if needed
self.input_shape_list = [
    {k: list(v) for k, v in shapes.items() if isinstance(v, torch.Size)}
    for shapes in self.input_shape_list
]
```

### 2. Parameter Generation
Use this pattern to generate pytest parameters:
```python
# Remove duplicates while preserving order
{''.join(set(f'        {shape[0]},' for shape in self.input_shape_list if 0 in shape))}
```

### 3. Multi-Input Operations
For operations with multiple inputs (like add, mul, div):
```python
def generate_code(self) -> str:
    return f"""
@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    (
{''.join(set(f'        ({shape[0]}, {shape[1]}),' for shape in self.input_shape_list if 0 in shape and 1 in shape))}
    )
)
"""
```

### 4. Special Operations
Some operations require special handling:

**TorchOnes (output-based):**
```python
class TorchOnesUnittest(UnitTestOperation):
    def __init__(self, output_shapes: Optional[List[Any]]):  # ← Note: output_shapes
        self.output_shapes = output_shapes
```

**Operations with additional parameters:**
```python
class CatUnittest(UnitTestOperation):
    def __init__(self, input_shapes: Optional[Dict[int, Any]], dim: Optional[int] = None):
        self.input_shapes = input_shapes
        self.dim = dim  # ← Additional parameter
```

## Testing Your Implementation

1. **Add a few test operations** to ensure your classes work correctly
2. **Check for linting errors** using the linting tools
3. **Verify the generated test code** compiles and runs
4. **Test with actual operation graphs** from your models

## Common Pitfalls

1. **Missing imports**: Always add the backend operation import
2. **Shape normalization**: Ensure shapes are properly converted from torch.Size to lists
3. **Parameter deduplication**: Use `dict.fromkeys()` to remove duplicates while preserving order
4. **Layout compatibility**: Some operations only work with specific layouts (TILE_LAYOUT vs ROW_MAJOR_LAYOUT)
5. **TTNN API differences**: PyTorch and TTNN APIs may have slight differences in parameter names/order

## File Locations

- **Main implementation**: `generate_pytorch_unittest_graph.py`
- **Backend operations**: `tracer_backend_utils.py`
- **Generated tests**: Output directory specified when running the generator

## Summary

The process involves creating three classes (Individual, Group, Combiner) and updating the configuration. Each class has a specific role in the pipeline:

1. **Individual**: Parses single operations from the graph
2. **Group**: Generates batch test cases for multiple operations
3. **Combiner**: Defines how operations are fused together
4. **Configuration**: Registers everything with the framework

Following this pattern ensures consistency and maintainability across all operation types.
