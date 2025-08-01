# PyTorch Unit Test Graph Generator

`generate_pytorch_unittest_graph.py` is a Python module designed to generate unit tests for PyTorch operations. It processes computational graphs, extracts operations, and creates parameterized unit tests for validating the correctness of traced models.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Supported Operations](#supported-operations)
- [Contributions](#contributions)

---

## Overview

This module is part of the `experimental_tracer` suite and focuses on generating unit tests for PyTorch operations. It supports operations like convolution, matrix multiplication, and max pooling, and provides tools to validate their behavior using parameterized tests.

---

## Features

1. **Unit Test Generation**:
   - Generates parameterized unit tests for PyTorch operations such as convolution, matrix multiplication, and max pooling.
   - Supports grouping operations for efficient testing.

2. **Operation Parsing**:
   - Parses operations from computational graphs using `OperationGraph` and maps them to corresponding unit test classes.

3. **Code Export**:
   - Exports generated unit tests as Python code, ready to be executed with `pytest`.

4. **Combiner Logic**:
   - Combines multiple operations of the same type into grouped unit tests for better organization and reduced redundancy.

---

## Usage

### Generating Unit Tests
1. **Import the Module**:
   - Ensure following [Getting Started](README.md#getting-started) and make your CWD as `experimental_tracer`.
    ```bash
      tt-metal/
      ├── ttnn/
      │   ├── ttnn/
      │   │   ├── experimental_tracer/
    ```

2. **Create a Unit Test Graph**:
   - Use the `PytorchLayerUnitTestGraph` class to process an `OperationGraph` and generate unit tests.

   Example:
   ```python
    from generate_pytorch_unittest_graph import PytorchLayerUnitTestGraph, PytorchLayerUnitTestGraphConfig
    from tracer_backend import OperationGraph
    import torch.nn as nn
    from tracer_backend import trace_torch_model

    class ExampleModel(nn.Module): # DEFINE A SIMPLE MODEL
        def __init__(self):
            # resnet block
            super(ExampleModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.pool(x)
            return x

    operation_graph = trace_torch_model(
        ExampleModel(), [[1, 3, 1280, 800]],
    ) # Trace the model to get the operation graph

    config = PytorchLayerUnitTestGraphConfig(operation_graph=operation_graph) # Create a configuration for the unit test graph
    unit_test_graph = PytorchLayerUnitTestGraph(config) # Initialize the unit test graph with the configuration
    unit_test_graph.dump_to_python_file("unit_tests.py", format_code=True) # Generate the unit tests and save them to a Python file
   ```

    output:
    ```python
    # unit_tests.py
    # Auto-generated PyTorch code
    import ttnn
    from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d
    import torch
    from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import (
        run_conv,
        torch_tensor_map,
    )
    import pytest


    @pytest.mark.parametrize(
        ...
    )
    def test_maxpool2d(
        ...
    ):
        ...


    @pytest.mark.parametrize(
        ...
    )
    ...
    def test_conv(
        ...
    ):
        ...
    ```

3. **Run the Generated Tests**:
   - Execute the generated `unit_tests.py` file using `pytest`:
        > Make sure you have tt_metal dependencies installed.
     ```bash
     pytest unit_tests.py
     ```

### Output
- The generated unit tests are saved in the specified Python file and include parameterized tests for operations like convolution, matrix multiplication, and max pooling.

---

## Supported Operations

The following PyTorch operations are supported for unit test generation:
- **Convolution** (`torch.ops.aten.convolution`)
- **Matrix Multiplication** (`torch.ops.aten.addmm`)
- **Max Pooling** (`torch.ops.aten.max_pool2d_with_indices`)

For each operation, the module generates parameterized tests with configurable attributes such as input shapes, kernel sizes, strides, and padding.

---

## Contributions

We welcome contributions to improve the functionality of `generate_pytorch_unittest_graph.py`. Below are some ways you can contribute:

**Adding Support for New Operations**:
- If you want to add support for new PyTorch operations, follow these steps in the [`generate_pytorch_unittest_graph.py`](generate_pytorch_unittest_graph.py) file:
   - Extend the `UnitTestOperation` class to support additional PyTorch operations.
   - Implement parsing logic and unit test generation for the new operation.
   - Create a combiner class if necessary to group similar operations.
   - Create a corresponding group class for the new operation.
   - Add default values to the `PytorchLayerUnitTestGraphConfig` class to include the new operation in the unit test generation process.


Feel free to submit pull requests or open issues for discussion. Happy coding!
