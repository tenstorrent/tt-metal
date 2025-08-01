# TTNN Visualizer

In this tutorial, we’ll explore the **TTNN Visualizer** – a powerful tool designed to help developers understand and optimize models running on Tenstorrent hardware. This tool offers intuitive, in-depth insights into your neural network’s execution flow, memory usage, and performance characteristics.

**Main features include:**

* A detailed, searchable list of all operations used within the model
* Interactive graph visualizations of operations and data flow
* L1, DRAM, and circular buffer memory plots with interactivity
* Tensor-level insights, including shape, layout, type, and memory placement
* Complete overview of all buffers used during the model run
* Core-level input/output visualization with sharding and tiling details
* L1 memory usage over time, including peak memory visualization
* Hierarchical view of device operations with associated memory buffers
* High-level operation flow graph for the full model
* Ability to load reports from local files or remote servers via SSH
* Support for running multiple instances of the tool simultaneously

TTNN Visualizer gives you a comprehensive overview of how your model utilizes hardware resources. It helps identify optimization opportunities, debug bottlenecks, and better understand your model’s execution at the system level.

For more details, visit the official [ttnn-visualizer GitHub repository](https://github.com/tenstorrent/ttnn-visualizer). You can also watch the full walkthrough video [here](https://youtu.be/lHtcD8cHoes?feature=shared).

---

## Prerequisites

The visualization workflow is divided into two stages:

1. **Model profiling and data collection**
2. **Visualization and analysis using TTNN Visualizer**

To collect profiling data, you’ll need the [tt-metal](https://github.com/tenstorrent/tt-metal) project. The visualizer expects two sets of files:

* A **memory report**
* A **performance report**

You can generate both by enabling profiling during model execution. To do this, clone and build `tt-metal` with the profiler enabled:

```bash
./build_metal.sh -p
```

Then, install `ttnn-visualizer` via pip:

```bash
pip install ttnn-visualizer
```

For installation from source or system requirements, see the [getting started guide](https://github.com/tenstorrent/ttnn-visualizer/blob/main/docs/getting-started.md).

> **NOTE:**
> You can run the visualizer on your local machine and either connect remotely to your Tenstorrent system via SSH or copy the generated profiling files to your local machine for offline analysis.

---

## Running TTNN Visualizer

Once installed, launch the application using:

```bash
ttnn-visualizer
```

This starts a local server at [http://localhost:8000](http://localhost:8000). Open this address in a browser (preferably Chrome). You’ll be greeted by the visualizer’s homepage:

![TTNN Visualizer Homepage](images/1_ttnn-visualizer-homepage.jpg)

Initially, you’ll only see the **Reports** tab active. Once the memory and performance reports are uploaded, all other tabs become available.

---

## Model Profiling

In this tutorial, we’ll profile the YOLOv4 model (320x320 input) trained on the COCO dataset. The model can be found in:

[`tt-metal/models/demos/yolov4`](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4)

### Generating the Memory Report

TTNN uses configuration options for profiling. These can be set either through:

* A configuration file (`TTNN_CONFIG_PATH`)
* Inline overrides (`TTNN_CONFIG_OVERRIDES`)

We’ll use a config file for flexibility:

1. **Create a setup file** called `vis.setup` and paste in the following:

```json
{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "ttnn_visualizer_tutorial",
    "enable_graph_report": false,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}
```

2. **Set the path** to this file in your environment:

```bash
export TTNN_CONFIG_PATH=path/to/vis.setup
```

> **Note:**
> Ensure all required global variables from `tt-metal` are also exported.

Now run the profiling:

```bash
pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```

Early in the logs, you’ll see something like:

```bash
Config{...root_report_path=generated/ttnn/reports/4042956046390500517}
```

This is the memory report output directory. After the run completes, it will contain:

* `config.json`
* `db.sqlite`

Upload this directory to the visualizer under the **Memory reports** section.

---

### Generating the Performance Report

Clear the previous config before continuing:

```bash
unset TTNN_CONFIG_PATH
# Or if set inline:
unset TTNN_CONFIG_OVERRIDES
```

Run profiling using Tracy:

```bash
python -m tracy -p -r -v -m pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```

At the end, the tool will output the path to a directory containing:

* `ops_perf_results_<timestamp>.csv`
* `device_profile_log.txt`
* `<name>.tracy` (Tracy file)

Upload this entire directory to the visualizer as the **Performance report**.

---

## Result Analysis

### Uploading Reports

Once both report directories are uploaded, all analysis tabs will become available:

![Upload](2_upload_files.jpg)

---

### Operations Tab

This tab provides a complete, interactive list of all operations in your model:

* Filter/search operations by name
* View input and output tensors per operation
* See Python-level execution time
* Click “Memory Details” to inspect memory layout for each operation

![Operations Tab](3_Operations_tab.jpg)

Memory Details offers a breakdown of:

* Per-core tensor placement (L1, DRAM)
* Tile layout and memory reuse
* Operation-to-buffer relationships

![Operation Memory Details](4_operation_details.jpg)

---

### Tensors Tab

This tab displays all tensors used in the model:

* Shape, datatype, layout (e.g., row-major or tiled)
* Placement (L1 or DRAM)
* Sharding details
* Producer/consumer operations

![Tensors Tab](5_Tensors_tab.jpg)

You can also filter tensors by high memory usage, making it easy to identify optimization candidates.


![High Usage Tensors](6_Tensors_high_usage)

---

### Buffers Tab

Visualize all memory buffers used during execution:

* Table and chart views available
* See allocation location (L1, DRAM)
* Correlate buffers with operations and tensor flow
* Understand buffer reuse and lifetimes

Useful for estimating memory headroom or pinpointing inefficient allocations.

![Buffers Tab](7_Buffers_tab.jpg)

---

### Graph Tab

Visual representation of the model:

* Shows operations as nodes and tensor flow as edges
* Click nodes for details on inputs, outputs, and execution
* Zoom and pan to explore subnetworks or specific paths
* Helpful for understanding overall model structure and execution paths

![Graph Tab](8_graph_tab.jpg)

---

### Performance Tab

Here you’ll find advanced profiling data:

* Operation runtime (ms) and execution order
* Number of cores used per op
* FLOPs and utilization analysis
* Charts showing runtime distribution per operation category
* Identify runtime bottlenecks or underutilized operations

Toggle **Matmul Optimization Analysis** to get hints about suboptimal matrix ops.

![Performance Tab](9_Performance_tab.jpg)

Use this tab to:

* Optimize kernel configurations
* Increase parallelism where needed
* Understand memory and compute utilization in detail

---

## Recap

To summarize:

1. **Build tt-metal with profiler support**:

   ```bash
   ./build_metal.sh -p
   ```
2. **Install and launch TTNN Visualizer**:

   ```bash
   pip install ttnn-visualizer
   ttnn-visualizer
   ```
3. **Generate profiling data**:

   * Memory Report: Use `pytest` with config
   * Performance Report: Use `tracy`
4. **Upload the report directories** to the visualizer
5. **Explore model details** using:

   * **Operations**: See execution flow and memory per operation
   * **Tensors**: Inspect data types, layout, and sharding
   * **Buffers**: Analyze memory allocation
   * **Graph**: Visualize the model’s structure
   * **Performance**: Find and fix performance bottlenecks

TTNN Visualizer gives you everything you need to deeply understand your model’s interaction with the hardware — and where it can be improved.
