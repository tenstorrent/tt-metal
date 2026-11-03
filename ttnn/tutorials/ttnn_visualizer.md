# TT-NN Visualizer

In this tutorial, we’ll explore the **TT-NN Visualizer** – a powerful tool designed to help developers understand and optimize models running on Tenstorrent hardware. This tool offers intuitive, in-depth insights into your neural network’s execution flow, memory usage, and performance characteristics.

**Main features include:**

* A detailed, searchable list of all operations used within the model.
* Interactive graph visualizations of operations and data flow.
* L1, DRAM, and circular buffer memory plots with interactivity.
* Tensor-level insights, including shape, layout, type, and memory placement.
* Complete overview of all buffers used during the model run.
* Core-level input/output visualization with sharding and tiling details.
* L1 memory usage over time, including peak memory visualization.
* Hierarchical view of device operations with associated memory buffers.
* High-level operation flow graph for the full model.
* Ability to load reports from local files or remote servers via SSH.
* Support for running multiple instances of the tool simultaneously.

TT-NN Visualizer gives you a comprehensive overview of how your model utilizes hardware resources. It helps identify optimization opportunities, debug bottlenecks, and better understand your model’s execution at the system level.

For more details, visit the official [TT-NN Visualizer GitHub repository](https://github.com/tenstorrent/ttnn-visualizer). You can also watch the full walkthrough video [here](https://youtu.be/lHtcD8cHoes?feature=shared).

---

## Prerequisites

The visualization workflow is divided into two stages:

1. **Model Profiling and Data Collection**
2. **Visualization and Analysis with TT-NN Visualizer**

To collect profiling data, you’ll need the [tt-metal](https://github.com/tenstorrent/tt-metal) project. The visualizer expects two sets of files:

* A **memory report**
* A **performance report**

You can generate both by enabling profiling during model execution. Clone and build `tt-metal`:

```bash
./build_metal.sh
```

Then, install `ttnn-visualizer` via pip:

```bash
pip install ttnn-visualizer
```

For installation from source or system requirements, see the [getting started guide](https://docs.tenstorrent.com/ttnn-visualizer/src/installing.html).

> [!NOTE]
> You can run the visualizer on your local machine and either connect remotely to your Tenstorrent system via SSH or copy the generated profiling files to your local machine for offline analysis.

---

## Running TT-NN Visualizer

Once installed, launch TT-NN Visualizer using:

```bash
ttnn-visualizer
```

This starts a local server at [http://localhost:8000](http://localhost:8000). Open this address in a browser (preferably Chrome). You’ll be greeted by the visualizer’s homepage:

![TT-NN Visualizer Homepage](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/1_ttnn_visualizer_homepage.jpg?raw=true)

Initially, you’ll only see the **Reports** tab active. Once the memory and performance reports are uploaded, all other tabs become available.

---

## Model Profiling

In this tutorial, we’ll profile MobileNetV2 (224x224 input) trained on ImageNet on a Wormhole device. Follow the model README prerequisites before running the commands from the tt-metal repository root:

[`tt-metal/models/demos/vision/classification/mobilenetv2`](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/vision/classification/mobilenetv2)

> [!NOTE]
> This tutorial uses the predefined MobileNetV2 model as an example, but you can profile any model by wrapping it in a pytest test case and following the same steps. For more information on creating custom test cases, refer to the [tt-metal documentation](https://github.com/tenstorrent/tt-metal).

### Generating the Memory Report

TT-NN uses configuration options for profiling. These can be set either through:

* A configuration file (`TTNN_CONFIG_PATH`)
* Inline overrides (`TTNN_CONFIG_OVERRIDES`)

We’ll use a config file for flexibility:

1. **Create a setup file** called `vis.setup` and paste in the following:

```json
{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "ttnn_visualizer_tutorial",
    "enable_graph_report": true,
    "enable_graph_python_stack_traces": true,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}
```

Each configuration option has a specific purpose:

* **enable_fast_runtime_mode** - Must be disabled to enable logging.
* **enable_logging** - Synchronizes main thread after every operation and logs the operation. Inside a metal trace capture the synchronization is skipped (it is not allowed there), so the logged timings cover dispatch only and device errors surface when the trace is executed.
* **report_name** (*optional*) - Name of the report used by TT-NN Visualizer. If not provided, the `ttnn_graph_report` pytest fixture derives one from the test id. The graph, buffer and tensor report flags below only write a report under pytest; a plain script generates one with `ttnn.graph.full_graph_capture(path)` followed by `ttnn.graph_report.import_report(path, output_dir)`. `enable_comparison_mode` still runs golden comparisons in any process, but its results only reach `db.sqlite` through the fixture.
* **enable_graph_python_stack_traces** (*optional*) - When true, outermost Python `begin_graph_capture` records Python call stacks. `ttnn.graph.full_graph_capture` always records Python stacks regardless of this flag.
* **enable_graph_report** (*required* for the graph view) - Captures the operation graph, which is what writes the graph data in `db.sqlite`. With it disabled the visualizer has no graph to show; *enable_comparison_mode* on its own still writes comparison records, but nothing else does.
* **enable_detailed_buffer_report** (if *report_name* and *enable_graph_report* are set) - Enable to visualize the detailed buffer report after every operation.
* **enable_detailed_tensor_report** (if *report_name* and *enable_graph_report* are set) - Records the Python source file and line of the operations that produce, last use and deallocate each tensor (the tensor lifetime view). It does not dump tensor values.
* **enable_comparison_mode** (if *report_name* is set) - Enable to test the output of operations against their golden implementation. Operations inside a metal trace capture are not compared, since their inputs and outputs cannot be read back there.

> [!NOTE]
> This config file corresponds to the recommended setup in TT-NN Visualizer docs, feel free to adjust it to your needs.

2. **Set the path** to the following file in your environment:

```bash
export TTNN_CONFIG_PATH=path/to/vis.setup
```

> [Note]
> Ensure all required global variables from `tt-metal` are also exported.

3. Run the profiling by simply running pytest:

```bash
pytest models/demos/vision/classification/mobilenetv2/tests/pcc/test_mobilenetv2.py::test_mobilenetv2
```

At the start of execution, the configuration log shows the selected settings. A shortened example is:

```text
... | DEBUG | ttnn:<module>:... - Initial ttnn.CONFIG:
Config{...,enable_fast_runtime_mode=false,...,enable_logging=true,enable_graph_report=true,...,root_report_path=generated/ttnn/reports,report_name=ttnn_visualizer_tutorial,...}
```

The report directory uses the sanitized report name and a timestamp, for example `generated/ttnn/reports/ttnn_visualizer_tutorial_nov02_1200/`. The exact path is available as `ttnn.CONFIG.report_path` in the test process. Once execution completes, that directory contains:

* `config.json` (single-process; distributed runs use `config_<rank>_of_<world_size>.json` when world_size > 1)
* `db.sqlite`
* `cluster_descriptor.yaml` (single-process; distributed runs add `cluster_descriptor_<rank>_of_<world_size>.yaml`)
* `physical_chip_mesh_coordinate_mapping.yaml` (single-process; distributed runs add `physical_chip_mesh_coordinate_mapping_<rank>_of_<world_size>.yaml`)

Upload this entire directory to TT-NN Visualizer under the **Memory reports** section.

For the mesh coordinate mapping YAML to be generated, you must have `TT_METAL_HOME` set.

---

### Generating Performance Reports

For the performance report, we'll use the `tracy profiler`. If you're using the same terminal session, unset the previous configuration to avoid regenerating the memory report:

```bash
unset TTNN_CONFIG_PATH
# Or if set inline:
unset TTNN_CONFIG_OVERRIDES
```

Run profiling using Tracy:

```bash
python -m tracy -p -r -v -m pytest models/demos/vision/classification/mobilenetv2/tests/pcc/test_mobilenetv2.py::test_mobilenetv2
```

Tracy prints the report CSV path. The timestamp and checkout path vary by run:

```text
... | INFO | ... - OPs csv generated at: /path/to/tt-metal/generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

This diredtory contains the following output files:

* `ops_perf_results_<timestamp>.csv`
* `device_profile_log.txt`
* `<name>.tracy` (Tracy file)

Upload this entire directory to TT-NN Visualizer as the **Performance report**.

---

## Result Analysis

### Uploading Reports

Once both report directories are uploaded, all analysis tabs will become available:

![Upload](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/2_upload_files.jpg?raw=true)

> [!NOTE]
> If successful, a message will appear at the bottom of the page indicating both reports have been synchronized.

---

### Operations Tab

The operations tab provides a complete, interactive list of all operations in your model:

* Filter/search operations by name.
* View input and output tensors per operation.
* See Python-level execution time.
* Click “Memory Details” to inspect memory layout for each operation.

![Operations Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/3_operations_tab.jpg?raw=true)

Memory Details offer the following breakdown:

* Per-core tensor placement (L1, DRAM).
* Tile layout and memory reuse.
* Operation-to-buffer relationships.

![Operation Memory Details](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/4_operation_details.jpg?raw=true)

---

### Tensors Tab

The Tensors tab provides detailed insights into all tensors used throughout your model's execution. View comprehensive tensor information including:

* Shape, datatype, layout (e.g., row-major or tiled).
* Placement (L1 or DRAM).
* Sharding details.
* Tensor movement between operations.

![Tensors Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/5_tensors_tab.jpg?raw=true)

You can also filter tensors by high memory usage, making it easy to identify optimization candidates.

![High Usage Tensors](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/6_tensors_high_usage.jpg?raw=true)

---

### Buffers Tab

Visualize all memory buffers used during execution:

* Table and chart views available.
* See allocation location (L1, DRAM).
* Correlate buffers with operations and tensor flow.
* Understand buffer reuse and lifetimes.

Useful for estimating memory headroom or pinpointing inefficient allocations.

![Buffers Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/7_buffers_tab.jpg?raw=true)

---

### Graph Tab

Visual representation of the model:

* Shows operations as nodes and tensor flow as edges.
* Click nodes for details on inputs, outputs, and execution.
* Zoom and pan to explore subnetworks or specific paths.
* Helpful for understanding overall model structure and execution paths.

![Graph Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/8_graph_tab.jpg?raw=true)

---

### Performance Tab

Here you’ll find advanced profiling data:

* Operation runtime (ms) and execution order.
* Number of cores used per op.
* FLOPs and utilization analysis.
* Charts showing runtime distribution per operation category.
* Identify runtime bottlenecks or underutilized operations.

Toggle **Matmul Optimization Analysis** for hints about suboptimal matrix ops.

![Performance Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/9_performance_tab.jpg?raw=true)

Use this tab to:

* Optimize kernel configurations.
* Increase parallelism.
* Understand memory and compute utilization in detail.

Visualize performance on graphs.

![Performance Graph](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/10_performance_graph.jpg?raw=true)

> [!NOTE]
> A comprehensive performance report analysis guide can be found in the official [tt-perf-report](https://github.com/tenstorrent/tt-perf-report) repository.

---

## Recap

To summarize:

1. **Build tt-metal**:

   ```bash
   ./build_metal.sh
   ```
2. **Install and Launch TT-NN Visualizer**:

   ```bash
   pip install ttnn-visualizer
   ttnn-visualizer
   ```
3. **Generate Profiling Data**:

   * Memory Report: Use `pytest` with config
   * Performance Report: Use `tracy`
4. **Upload Report Directories** to the visualizer
5. **Explore Model Details** using:

   * **Operations**: See execution flow and memory per operation
   * **Tensors**: Inspect data types, layout, and sharding
   * **Buffers**: Analyze memory allocation
   * **Graph**: Visualize the model’s structure
   * **Performance**: Find and fix performance bottlenecks

TT-NN Visualizer gives you everything you need to deeply understand your model’s interaction with the hardware — and where it can be improved.
