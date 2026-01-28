# Sample Tracer

`sample_tracer.py` is a Python tool for tracing PyTorch models, generating computational graphs, and exporting them in multiple formats. It supports debugging, testing, and analysis, making it an essential utility for deep learning developers and researchers.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Future Enhancements](#future-enhancements)
- [How It Works](#how-it-works)
- [Supported Models](#supported-models)
- [Contributions](#contributions)


## Getting Started

### Prerequisites
1. **Python Environment**: Ensure you have Python 3.8 or later installed.
2. **Dependencies**: Install the required Python libraries:
    > Note: While no additional TT libraries are required, some files from the `tt-metal/models/demos` directory are necessary for the script to function.

   ```bash
   pip install torch torchvision transformers ultralytics torchinfo networkx xlsxwriter black
   ```

## Usage
1. Clone the repository and navigate to the directory containing `sample_tracer.py`:

   ```bash
   git clone git@github.com:tenstorrent/tt-metal.git
   export TT_METAL_HOME=$PWD/tt-metal/
   export PYTHONPATH=$PYTHONPATH:"${TT_METAL_HOME}"
   cd tt-metal/ttnn/ttnn/experimental_tracer/
   ```

2. Directory Structure: Below is the relevant directory structure for the `sample_tracer.py` script:

   ```bash
      tt-metal/
      ├── ttnn/
      │   ├── ttnn/
      │   │   ├── experimental_tracer/
      │   │   │   ├── sample_tracer.py          # Main script for tracing
      │   │   │   ├── tracer_backend.py         # Core tracing logic
      │   │   │   ├── generate_pytorch_graph.py # PyTorch code generation
      │   │   │   ├── generate_pytorch_excel_graph.py # Excel export
      │   │   │   ├── generate_pytorch_unittest_graph.py # Unit test generation
      │   │   │   └── ...
      │   └── ...
      └── ...
   ```

3. Run the script with the desired model and input configurations:

   ```bash
   python sample_tracer.py --model yolov8s --input-shape 1 3 640 640
   python sample_tracer.py --model yolov12x  --input-shape 1 3 4320 7680 --disable-torch-summary --no-infer
   ```

   - Replace `yolov8s` with any supported model from the `allowed_modes` list (defined in `sample_tracer.py`).
   - Specify input shapes using the `--input-shape` argument (e.g., `1 3 640 640` for batch size 1, 3 channels, and 640x640 resolution).
   - Run with --disable-torch-summary and --no-infer for a faster run without generating summary on the CLI

4. Outputs saved in the current working directory:
   - **`graph.py`**: Executable PyTorch code representing the traced model.
   - **`graph.xlsx`**: Excel report with detailed operation metadata.
   - **`test.py`**: Unit test code for validating the traced operations on TTNN.
      > To run this, you need to install tt-metal dependencies.
   - **`operation_graph_viz.json`**: visualization input to https://netron.app

## Features

### Summary
`sample_tracer.py` leverages the following features from its integrated files:
1. **Model Tracing**:
   - Uses `trace_torch_model` from `tracer_backend.py` to trace PyTorch models and generate an `OperationGraph`.
   - Captures tensor operations, metadata, and computational graphs.

2. **Graph Export**:
   - **Python Code**: Converts the traced graph into executable PyTorch code using `PytorchGraph` from `generate_pytorch_graph.py`.
   - **Excel Report**: Exports the graph as a detailed Excel sheet using `PytorchExcelGraph` from `generate_pytorch_excel_graph.py`.
   - **Unit Test Generation**:
      - Generates parameterized unit tests for operations (e.g., convolution, addition, pooling) using `PytorchLayerUnitTestGraph` from `generate_pytorch_unittest_graph.py`.

4. **Model Support**:
   - Supports a wide range of models, including YOLO (v4, v7, v8), ResNet50, MobileNetV2, EfficientNet, and Transformers (e.g., ViT, Swin).

5. **Visualization**:
   - Optionally dumps the graph as JSON for visualization with tools like Netron.

## Future Enhancements

1. **Expanded Model Support**:
   - Add support for additional models and architectures, including custom user-defined models.

2. **Enhanced Visualization**:
   - Integrate with advanced visualization tools for real-time graph inspection.

3. **Add Wrapped Operator Support**:

   - [x] Convolution
   - [x] Addm (Matrix multiplication with bias addition)
   - [x] Maxpool with indices
   - [ ] Batch matmul
   - [ ] Binary Ops
     - [ ] Add
     - [ ] Multiply
     - [ ] Divide
     - [ ] ...
   - [ ] Unary Ops
     - [ ] Sigmoid
     - [ ] Softmax
     - [ ] ...
   - [ ] ...



## How It Works

### Workflow
1. **Model Selection**:
   - Choose a model from the `allowed_modes` list.
   - Specify input shapes and data types.

2. **Tracing**:
   - The script traces the model using `trace_torch_model` and generates an `OperationGraph`.

3. **Export**:
   - The traced graph is exported as Python code, an Excel report, and unit test code.

4. **Validation**:
   - Use the generated unit tests (`test.py`) to validate the correctness of the traced operations.

## Supported Models

The following models are supported out of the box:
- YOLO (v4, v7, v8)
- ResNet50
- MobileNetV2
- EfficientNet (B0, B3, B4)
- Sentence-BERT
- VGG-UNet
- SegFormer
- Vision Transformers (ViT, Swin)

For custom models, ensure they are compatible with PyTorch and provide the necessary input shapes.

## Contributions

We welcome contributions to enhance the functionality of `sample_tracer.py` and its associated tools. Below are some ways you can contribute:

1. **Adding Unit Test Generators for Operations**:
   - To contribute to adding unit test generators for specific operations (e.g., batch matmul, binary/unary ops), refer to the [`generate_pytorch_unittest_graph.py`](generate_pytorch_unittest_graph.py) file in the same directory. This file contains the logic for generating parameterized unit tests. For detailed instructions, please refer to [`README_generate_pytorch_unittest_graph.md`](README_generate_pytorch_unittest_graph.md)

2. **Expanding Model Support**:
   - Add support for new models or architectures by updating the `allowed_modes` list in `sample_tracer.py`.
   - Ensure compatibility with PyTorch and provide necessary input shape configurations.

## Conclusion

`sample_tracer.py` is a tool for tracing PyTorch models, generating computational graphs, and exporting them in multiple formats for debugging, testing, and analysis. With its support for a wide range of models and its extensible design, it is an useful utility for developers and researchers working with deep learning frameworks.

We welcome contributions and feedback to improve this tool further. Happy tracing!
