# TT-NN Visualizer

In this tutorial, we’ll explore the **TT-NN Visualizer** – a powerful tool designed to help developers understand and optimize models running on Tenstorrent hardware. This tool offers intuitive, in-depth insights into your neural network’s execution flow, memory usage, and performance characteristics.

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

TT-NN Visualizer gives you a comprehensive overview of how your model utilizes hardware resources. It helps identify optimization opportunities, debug bottlenecks, and better understand your model’s execution at the system level.

For more details, visit the official [ttnn-visualizer GitHub repository](https://github.com/tenstorrent/ttnn-visualizer). You can also watch the full walkthrough video [here](https://youtu.be/lHtcD8cHoes?feature=shared).

---

## Prerequisites

The visualization workflow is divided into two stages:

1. **Model profiling and data collection**
2. **Visualization and analysis using TT-NN Visualizer**

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

![TTNN Visualizer Homepage](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/1_ttnn_visualizer_homepage.jpg?raw=true)

Initially, you’ll only see the **Reports** tab active. Once the memory and performance reports are uploaded, all other tabs become available.

---

## Model Profiling

In this tutorial, we’ll profile the YOLOv4 model (320x320 input) trained on the COCO dataset. The model can be found in:

[`tt-metal/models/demos/yolov4`](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4)

> **NOTE**
> This tutorial uses the predefined YOLOv4 model as an example, but you can profile any model by wrapping it in a pytest test case and following the same steps. For more information on creating custom test cases, refer to the [tt-metal documentation](https://github.com/tenstorrent/tt-metal).

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
    "enable_graph_report": false,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}
```

Here's what each configuration option does:

* **enable_fast_runtime_mode** - Must be disabled to enable logging,
* **enable_logging** - Synchronizes main thread after every operation and logs the operation,
* **report_name** (*optional*) - Name of the report used by the visualizer. If not provided, no data will be dumped to disk,
* **enable_detailed_buffer_report** (if *report_name* is set) - Enable to visualize the detailed buffer report after every operation,
* **enable_graph_report** (if *report_name* is set) - Enable to visualize the graph after every operation,
* **enable_detailed_tensor_report** (if *report_name* is set) - Enable to visualize the values of input and output tensors of every operation,
* **enable_comparison_mode** (if *report_name* is set) - Enable to test the output of operations against their golden implementation.

> **NOTE:**
> This config file corresponds to the recommended setup in TTNN Visualizer docs, but feel free to adjust it to your needs.

2. **Set the path** to this file in your environment:

```bash
export TTNN_CONFIG_PATH=path/to/vis.setup
```

> **Note:**
> Ensure all required global variables from `tt-metal` are also exported.

Now run the profiling by simply running pytest:

```bash
pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```

At the start of execution, you should see logs similar to:

```bash
(python_env) /root/tt-metal$ pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
2025-08-01 09:20:51.664 | DEBUG    | ttnn:<module>:73 - Loading ttnn configuration from /root/tt-metal/vis.setup
2025-08-01 09:20:51.665 | DEBUG    | ttnn:<module>:83 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false, \
   enable_fast_runtime_mode=false,throw_exception_on_fallback=false,enable_logging=true,enable_graph_report=false,enable_detailed_buffer_report=true, \
   enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false, \
   comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=ttnn_visualizer_tutorial,4042956046390500517}
2025-08-01 09:20:51.754 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.758 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:192)
2025-08-01 09:20:51.758 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.761 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.764 | info     |   SiliconDriver | Harvesting mask for chip 0 is 0x80 (NOC0: 0x80, simulated harvesting mask: 0x0). (cluster.cpp:295)
2025-08-01 09:20:51.776 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.836 | info     |   SiliconDriver | Opening local chip ids/pci ids: {0}/[4] and remote chip ids {} (cluster.cpp:157)
```

In the configuration output, look for the report path at the end:

```bash
Config{...root_report_path=generated/ttnn/reports,report_name=ttnn_visualizer_tutorial,4042956046390500517}
```

The final number (`4042956046390500517`) indicates the memory report output directory. Once execution completes, navigate to `generated/ttnn/reports/4042956046390500517/` which will contain:

* `config.json`
* `db.sqlite`

Upload this entire directory to the visualizer under the **Memory reports** section.

---

### Generating the Performance Report

For the performance report, we'll use the `tracy profiler`. If you're using the same terminal session, first unset the previous configuration to avoid regenerating the memory report:

```bash
unset TTNN_CONFIG_PATH
# Or if set inline:
unset TTNN_CONFIG_OVERRIDES
```

Run profiling using Tracy:

```bash
python -m tracy -p -r -v -m pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```

At the end, the tool will output the path to a directory looking like:

```bash
2025-08-01 10:51:02.731 | INFO     | tt_metal.tools.profiler.process_ops_logs:generate_reports:905 - OPs csv generated at: /root/tt-metal/generated/profiler/reports/2025_08_01_10_51_02/ops_perf_results_2025_08_01_10_51_02.csv
```

containing:

* `ops_perf_results_<timestamp>.csv`
* `device_profile_log.txt`
* `<name>.tracy` (Tracy file)

Upload this entire directory to the visualizer as the **Performance report**.

---

## Result Analysis

### Uploading Reports

Once both report directories are uploaded, all analysis tabs will become available:

![Upload](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/2_upload_files.jpg?raw=true)

> **NOTE:**
> If everything went according to plan, you should see a message at the bottom of the page that both reports have been synchronized.

---

### Operations Tab

This tab provides a complete, interactive list of all operations in your model:

* Filter/search operations by name,
* View input and output tensors per operation,
* See Python-level execution time,
* Click “Memory Details” to inspect memory layout for each operation.

![Operations Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/3_operations_tab.jpg?raw=true)

Memory Details offers a breakdown of:

* Per-core tensor placement (L1, DRAM),
* Tile layout and memory reuse,
* Operation-to-buffer relationships.

![Operation Memory Details](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/4_operation_details.jpg?raw=true)

---

### Tensors Tab

The Tensors tab provides detailed insights into all tensors used throughout your model's execution. View comprehensive tensor information including:

* Shape, datatype, layout (e.g., row-major or tiled),
* Placement (L1 or DRAM),
* Sharding details,
* Tensor movement between operations.

![Tensors Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/5_tensors_tab.jpg?raw=true)

You can also filter tensors by high memory usage, making it easy to identify optimization candidates.

![High Usage Tensors](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/6_tensors_high_usage.jpg?raw=true)

---

### Buffers Tab

Visualize all memory buffers used during execution:

* Table and chart views available,
* See allocation location (L1, DRAM),
* Correlate buffers with operations and tensor flow,
* Understand buffer reuse and lifetimes.

Useful for estimating memory headroom or pinpointing inefficient allocations.

![Buffers Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/7_buffers_tab.jpg?raw=true)

---

### Graph Tab

Visual representation of the model:

* Shows operations as nodes and tensor flow as edges,
* Click nodes for details on inputs, outputs, and execution,
* Zoom and pan to explore subnetworks or specific paths,
* Helpful for understanding overall model structure and execution paths.

![Graph Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/8_graph_tab.jpg?raw=true)

---

### Performance Tab

Here you’ll find advanced profiling data:

* Operation runtime (ms) and execution order,
* Number of cores used per op,
* FLOPs and utilization analysis,
* Charts showing runtime distribution per operation category,
* Identify runtime bottlenecks or underutilized operations.

Toggle **Matmul Optimization Analysis** to get hints about suboptimal matrix ops.

![Performance Tab](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/9_performance_tab.jpg?raw=true)

Use this tab to:

* Optimize kernel configurations,
* Increase parallelism where needed,
* Understand memory and compute utilization in detail.

Visualize performance on graphs.

![Performance Graph](https://github.com/mgajewskiTT/ttnn-tutorials-images/blob/main/media/ttnn_visualizer/10_performance_graph.jpg?raw=true)

> **NOTE**
> A comprehensive performance report analysis guide can be found in the official [tt-perf-report](https://github.com/tenstorrent/tt-perf-report) repository.

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

TT-NN Visualizer gives you everything you need to deeply understand your model’s interaction with the hardware — and where it can be improved.
