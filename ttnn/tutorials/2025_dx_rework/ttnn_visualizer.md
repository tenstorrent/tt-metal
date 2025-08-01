# TTNN Visualizer

In this example we will explore ttnn visualizer tool that allows you to easily and intuitively explore neural network models. The main features of this tool include:
- Detailed list of all operations used within the model,
- Interactive graph visualization of operations
- Detailed and interactive L1, DRAM, and circular buffer memory plots
- Filterable list of tensor details
- Overview of all buffers for the entire model run
- Visualization of input and output tensors with core tiling and sharding details
- Visualize inputs/outputs per tensor or tensor allocation across each core
- Detailed insights into L1 peak memory consumption, with an interactive graph of allocation over time
- Navigate a tree of device operations with associated buffers and circular buffers
- Operation flow graph for a holistic view of model execution
- Load reports via the local file system or through an SSH connection
- Supports multiple instances of the application running concurrently.

TTNN Visualizer has been specially prepared to give you a complete overview of the model's properties - from the operations used, through memory allocation, to the achieved performance. It lets you easily se what performance optimizations can be done.

You can find all the information in the official [tool repository](https://github.com/tenstorrent/ttnn-visualizer).
There is also a video version of this tutorial available at this [link](https://youtu.be/lHtcD8cHoes?feature=shared).

## Prerequisites

Using ttnn visualizer is divided into two parts:
- Model profiling and data collection,
- Data visualization and analysis.

For the first stage of data collection, the [tt-metal](https://github.com/tenstorrent/tt-metal) project will be required. Visualizer operates on two sets of files – memory report and performance report – to obtain them we use profiling. We need to build the project for profiling, to do this please follow the [installing instructions](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md) in tt-metal repository or just build project as:

```bash
./build_metal.sh -p
```

To install ttnn-visualizer simply use `pip` command like:

```bash
pip install ttnn-visualizer
```
For more information about system requirements or installing from source please refer to the [ttnn-visualizer repository](https://github.com/tenstorrent/ttnn-visualizer/blob/main/docs/getting-started.md).

> **NOTE**
>
> If you are connecting to the remote server to use Tenstorrent systems you can install the visualizer on your local machine and connect to remote server through it or just download profiling output files to your local machine and use them locally.

## Running TTNN-visualizer

After successful installation we can start the tool by simply typing in the terminal:

```
ttnn-visualizer
```

This results in creating start endpoint at [`localhost:8000`](http://localhost:8000/) go ahead and open it in your browser (we recommend Chrome browser for this action). You should see homepage:

![1_ttnn-visualizer-homepage](images/1_ttnn-visualizer-homepage.jpg)

By default we start at `Reports` tab, which allows you to upload necessary files from local machine or connect directly via Remote sync with your machine. In this example we will use the first option. Currently other tabs are unavailable they will be functional after uploading profiling reports. Keep in mind that memory report unlocks tabs from Operations to Graph and Performance tab is unlocked with performance report.

## Model profiling

In this tutorial we will profile yolov4 model on 320x320 images from COCO dataset, you can find model in [tt-metal/models/demos/convnet_mnist](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4).

### Obtaining memory report

To start with we will obtain memory report. For this we first need to set configuration options. There are two ways to do that:
- Create setup file and place all config there - set with `TTNN_CONFIG_PATH`,
- Explicitly set options in terminal with `TTNN_CONFIG_OVERRIDES`.

For simplicity and ease of change we will create a setup file and set `TTNN_CONFIG_PATH` globlal variable as the path to it.

1. Create `vis.setup` file
2. Open file with editor of your choice and paste general config:

```
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

You can find explanation of all options in [ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer) repository docs.

3. Set `TTNN_CONFIG_PATH`
    - Open terminal
    - Set variable

    ```bash
    export TTNN_CONFIG_PATH=path/to/vis.setup
    ```
> **NOTE**
>
> Remember to set also all necessary global variables described in `tt-metal` repository!

We are ready for profiling. Start profiling with:

```bash
pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```

Ath the very top you should see the logs looking like:

```bash
(python_env) /root/tt-metal$ pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0][resolution0-103-1-act_dtype0-weight_dtype0-device_params0]
2025-08-01 09:20:51.664 | DEBUG    | ttnn:<module>:73 - Loading ttnn configuration from /root/tt-metal/vis.setup
2025-08-01 09:20:51.665 | DEBUG    | ttnn:<module>:83 - Initial ttnn.CONFIG:
Config{cache_path=/root/.cache/ttnn,model_cache_path=/root/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false, \
    enable_fast_runtime_mode=false,throw_exception_on_fallback=false,enable_logging=true,enable_graph_report=false,enable_detailed_buffer_report=true, \
    enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false, \
    comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=ttnn_visualizer_tutorial,generated/ttnn/reports/4042956046390500517}
2025-08-01 09:20:51.754 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.758 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:192)
2025-08-01 09:20:51.758 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.761 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.764 | info     |   SiliconDriver | Harvesting mask for chip 0 is 0x80 (NOC0: 0x80, simulated harvesting mask: 0x0). (cluster.cpp:295)
2025-08-01 09:20:51.776 | info     |   SiliconDriver | Opened PCI device 4; KMD version: 1.34.0; API: 1; IOMMU: disabled (pci_device.cpp:197)
2025-08-01 09:20:51.836 | info     |   SiliconDriver | Opening local chip ids/pci ids: {0}/[4] and remote chip ids {} (cluster.cpp:157)
```

In the `Config{` at the very end you can see a path to memory report directory - in this case `generated/ttnn/reports/4042956046390500517`.

After the whole run is over in the output direcotry two new files will be created `config.json` and `db.sqlite`. After uploading direcotry to the ttnn-visualizer as memory report first four tabs will be available to use.

> **NOTE**
>
> In this tutorial we use predefined model, but you can use any model you want just close it in pytest test case - for more information refer to docs.


### Obtaining performance report

For the second part to obtain report of the model we need to use `tracy profiler`. If you are using the same terminal we need to unset previous configuration since we don't want to generate everything again:

```bash
unset TTNN_CONFIG_PATH
```
if you set `TTNN_CONFIG_OVERRIDES` use:

```bash
unset TTNN_CONFIG_OVERRIDES
```

Profiling starts with:

```bash
python -m tracy -p -r -v -m pytest models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]
```
In this case the path with output files is displayed at the very end of the program like:

```bash
2025-08-01 10:51:02.593 | INFO     | tt_metal.tools.profiler.process_ops_logs:generate_reports:643 - OPs' perf analysis is finished! Generating reports ...
2025-08-01 10:51:02.593 | INFO     | tt_metal.tools.profiler.process_ops_logs:generate_reports:661 - Copying runtime artifacts
2025-08-01 10:51:02.652 | INFO     | tt_metal.tools.profiler.process_ops_logs:generate_reports:670 - Generating OPs CSV
2025-08-01 10:51:02.731 | INFO     | tt_metal.tools.profiler.process_ops_logs:generate_reports:905 - OPs csv generated at: /root/tt-metal/generated/profiler/reports/2025_08_01_10_51_02/ops_perf_results_2025_08_01_10_51_02.csv
```
Output should contain two `csv` files and one `.tracy` file. To ttnn-visualizer we pass the whole: `/root/tt-metal/generated/profiler/reports/2025_08_01_10_51_02/` directory.

## Result analysis

### Upload files

Start by uploading both directories to appropriate placeholders on homepage. IF everything was done correctly another tabs should now be avalible to use.

![2_uplad_files](2_upload_files.jpg)

### Operations

Operations tab let's you browse through all operations used in model. You can filter them to look for te one that you are interested in the most, expand them to see on what exact input data operation was working on and what was the Python execution time of it.

![3_Operations_tab](3_Operations_tab.jpg)

Near each operation Memory detail button is visible. By clicking it you will be transferred to memory page site that allows you to see in more details memory allocation characteristics. You are able to see the exact adress space on each core tensor is occupying, what underneath operations are used and how they alocate new memory or use existing one.

![4_Operation_details](4_operation_details.jpg)

### Tensors

Tensors tab allow you to see all the tensors used in model (provide Tensor level view). You are able to explore their datatype, layout or memory layout and placement (DRAM or L1). If the tensor would be sharded additional information would be displayed here.

![5_Tensors_tab][5_Tensors_tab.jpg]

Useful you can see high consuming tensors – potential optimization:

![6_Tensors_high_usage](6_Tensors_high_usage)

### Buffers

Buffers tab can give you information memory allocation, how tensors are allocated – which memory type and places in memory,

![7_Buffers_tab](7_Buffers_tab.jpg)

### Graph

![8_graph_tab](8_graph_tab.jpg)

### Performance

![9_Performance_tab](9_Performance_tab.jpg)
