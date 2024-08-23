# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math
import pathlib

from bokeh.embed import components
from bokeh.models import Plot, ColumnDataSource, LinearAxis, CustomJSTickFormatter, NumeralTickFormatter, Rect, Range1d
from bokeh.models.tools import WheelZoomTool, PanTool, ResetTool, ZoomInTool, ZoomOutTool, HoverTool
from bokeh.palettes import Category20
from bokeh.plotting import figure
from flask import Flask, render_template, session
from loguru import logger
import numpy as np
import pandas as pd

import ttnn
import ttnn.database

ttnn.CONFIG.enable_logging = False

logger.info(f"Visualizer ttnn.CONFIG {ttnn.CONFIG}")

BUFFER_TO_COLOR_INDEX = {}
COLORS = Category20[20]


def get_report_path():
    report_path = session.get("report_path")
    if report_path is not None:
        return pathlib.Path(report_path)
    raise ValueError("report_path is not set in session")


def shorten_stack_trace(stack_trace, num_lines=12):
    if stack_trace is None:
        return None
    stack_trace = stack_trace.split("\n")[:num_lines]
    stack_trace = "\n".join(stack_trace)
    return stack_trace


def red_to_green_spectrum(percentage):
    percentage_difference = 1.0 - percentage
    red_color = int(min(255, percentage_difference * 8 * 255))
    green_color = int(min(255, percentage * 2 * 255))
    color = f"#{red_color:02X}{green_color:02X}{0:02X}"
    return color


def tensor_comparison_record_to_percentage(record):
    if record.matches:
        percentage = 1
    elif record.actual_pcc < 0:
        percentage = 0
    elif record.actual_pcc >= record.desired_pcc:
        return 1.0
    else:
        percentage = record.actual_pcc * 0.9 / record.desired_pcc
    return percentage


def get_actual_pccs(table_name, operation_id):
    report_path = get_report_path()
    output_tensor_records = ttnn.database.query_output_tensors(report_path, operation_id=operation_id)
    output_tensor_records = sorted(output_tensor_records, key=lambda tensor: tensor.output_index)

    if not output_tensor_records:
        return "No output tensors"

    actual_pccs = []
    for output_tensor_record in output_tensor_records:
        tensor_comparison_record = ttnn.database.query_tensor_comparison_record(
            report_path, table_name, tensor_id=output_tensor_record.tensor_id
        )
        if tensor_comparison_record is None:
            continue
        actual_pccs.append(tensor_comparison_record.actual_pcc)

    if not actual_pccs:
        return "Comparison N/A"

    actual_pccs = ", ".join([f"{pcc:.5f}" for pcc in actual_pccs])
    return f"Actual PCCs: {actual_pccs}"


def comparison_color(table_name, operation_id):
    report_path = get_report_path()
    output_tensor_records = ttnn.database.query_output_tensors(report_path, operation_id=operation_id)
    output_tensor_records = sorted(output_tensor_records, key=lambda tensor: tensor.output_index)

    if not output_tensor_records:
        return "white"

    percentages = []
    for output_tensor_record in output_tensor_records:
        tensor_comparison_record = ttnn.database.query_tensor_comparison_record(
            report_path, table_name, tensor_id=output_tensor_record.tensor_id
        )
        if tensor_comparison_record is None:
            continue
        percentages.append(tensor_comparison_record_to_percentage(tensor_comparison_record))

    if not percentages:
        return "grey"

    percentage = sum(percentages) / len(percentages)
    return red_to_green_spectrum(percentage)


app = Flask(__name__)
app.secret_key = "BAD_SECRET_KEY"


@app.route("/")
def root():
    report_dirs = ttnn.CONFIG.root_report_path.glob("*")
    reports = []
    for report_dir in report_dirs:
        with open(report_dir / ttnn.database.CONFIG_PATH) as f:
            config = json.load(f)
        reports.append((report_dir.name, config["report_name"]))
    return render_template("root.html", root_report_path=ttnn.CONFIG.root_report_path, reports=reports)


@app.route("/apis")
def apis():
    @dataclasses.dataclass
    class Api:
        python_fully_qualified_name: str
        is_experimental: bool
        is_cpp_operation: bool
        golden_function: callable

        @classmethod
        def from_registered_operation(cls, operation):
            return cls(
                python_fully_qualified_name=operation.python_fully_qualified_name,
                is_experimental=operation.is_experimental,
                is_cpp_operation=operation.is_cpp_operation,
                golden_function=operation.golden_function,
            )

    apis = [Api.from_registered_operation(api) for api in ttnn.query_registered_operations(include_experimental=True)]

    df = pd.DataFrame(apis)
    df.sort_values(by=["is_experimental", "is_cpp_operation", "python_fully_qualified_name"], inplace=True)
    df["has_fallback"] = df["golden_function"].apply(lambda golden_function: golden_function is not None)
    return render_template(
        "apis.html",
        apis=df.to_html(
            index=False,
            justify="center",
            columns=[
                "python_fully_qualified_name",
                "is_cpp_operation",
                "is_experimental",
                "has_fallback",
            ],
        ),
    )


@app.route("/operations/", defaults={"report_hash": None})
@app.route("/operations/<report_hash>")
def operations(report_hash):
    if report_hash is None:
        report_path = get_report_path()
    else:
        report_path = ttnn.CONFIG.root_report_path / report_hash
        session["report_hash"] = report_hash
        session["report_path"] = str(report_path)
        with open(report_path / ttnn.database.CONFIG_PATH) as f:
            config = json.load(f)
        session["report_name"] = config["report_name"]

    operations = list(ttnn.database.query_operations(get_report_path()))

    def load_captured_graph(operation_id):
        captured_graph = ttnn.database.query_captured_graph(get_report_path(), operation_id=operation_id)
        output = ttnn.graph.pretty_format(captured_graph)
        output = output.replace(" ", "&nbsp;")
        output = output.replace("\n", "<br>")
        return output

    return render_template(
        "operations.html",
        operations=operations,
        comparison_color=comparison_color,
        get_actual_pccs=get_actual_pccs,
        load_captured_graph=load_captured_graph,
    )


@app.route("/operations_with_l1_buffer_report")
def operations_with_l1_buffer_report():
    operations = list(ttnn.database.query_operations(get_report_path()))

    l1_reports = {}
    stack_traces = {}
    for operation in operations:
        l1_reports[operation.operation_id] = create_summarized_l1_buffer_plot(operation.operation_id)
        stack_trace = ttnn.database.query_stack_trace(get_report_path(), operation_id=operation.operation_id)
        stack_traces[operation.operation_id] = shorten_stack_trace(stack_trace)

    return render_template(
        "operations_with_l1_buffer_report.html",
        operations=operations,
        l1_reports=l1_reports,
        stack_traces=stack_traces,
    )


def create_summarized_l1_buffer_plot(operation_id):
    glyph_y_location = 0
    glyph_height = 1

    buffers = list(ttnn.database.query_buffers(get_report_path(), operation_id))
    if len(buffers) == 0:
        return "", "There are no L1 Buffers!"
    device_ids = set(buffer.device_id for buffer in buffers)
    if len(device_ids) != 1:
        return "", "Cannot visualize buffer plot for multiple devices!"
    device_id = device_ids.pop()
    device = ttnn.database.query_device_by_id(get_report_path(), device_id)

    l1_size = device.worker_l1_size

    memory_glyph_y_location = [glyph_y_location]
    memory_glyph_x_location = [l1_size // 2]
    memory_height = [glyph_height]
    memory_width = [l1_size]
    memory_color = ["white"]
    memory_line_color = ["black"]

    memory_data_source = ColumnDataSource(
        dict(
            glyph_y_location=memory_glyph_y_location,
            glyph_x_location=memory_glyph_x_location,
            glyph_height=memory_height,
            glyph_width=memory_width,
            color=memory_color,
            line_color=memory_line_color,
        )
    )

    buffers_glyph_y_location = []
    buffers_glyph_x_location = []
    buffers_height = []
    buffers_width = []
    buffers_color = []
    buffers_max_size_per_bank = []
    buffers_address = []

    for buffer in ttnn.database.query_buffers(get_report_path(), operation_id):
        if (buffer.device_id, buffer.address, buffer.buffer_type) not in BUFFER_TO_COLOR_INDEX:
            BUFFER_TO_COLOR_INDEX[(buffer.device_id, buffer.address, buffer.buffer_type)] = len(BUFFER_TO_COLOR_INDEX)

        buffers_address.append(buffer.address)
        buffers_max_size_per_bank.append(buffer.max_size_per_bank)
        buffers_glyph_y_location.append(glyph_y_location)
        buffers_glyph_x_location.append(buffer.address + buffer.max_size_per_bank // 2)
        buffers_height.append(glyph_height)
        buffers_width.append(buffer.max_size_per_bank)
        buffers_color.append(
            COLORS[BUFFER_TO_COLOR_INDEX[(buffer.device_id, buffer.address, buffer.buffer_type)] % len(COLORS)]
        )

    buffers_glyph_x_location = np.asarray(buffers_glyph_x_location)
    buffers_glyph_y_location = np.asarray(buffers_glyph_y_location)
    buffers_height = np.asarray(buffers_height)
    buffers_width = np.asarray(buffers_width)
    buffers_data_source = ColumnDataSource(
        dict(
            glyph_y_location=buffers_glyph_y_location,
            glyph_x_location=buffers_glyph_x_location,
            glyph_height=buffers_height,
            glyph_width=buffers_width,
            color=buffers_color,
            address=buffers_address,
            max_size_per_bank=buffers_max_size_per_bank,
        )
    )

    plot = Plot(title=None, width=800, height=100, min_border=0, toolbar_location="below")

    xaxis = LinearAxis()
    plot.x_range = Range1d(0, l1_size)
    plot.add_layout(xaxis, "below")
    plot.xaxis.axis_label = "L1 Address Space"
    plot.xaxis.formatter = NumeralTickFormatter(format="0000000")

    memory_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color="line_color",
        fill_color="color",
    )
    plot.add_glyph(memory_data_source, memory_glyph)

    buffer_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color="black",
        fill_color="color",
    )
    buffer_renderer = plot.add_glyph(buffers_data_source, buffer_glyph)

    plot.add_tools(
        WheelZoomTool(),
        PanTool(),
        ResetTool(),
        ZoomInTool(),
        ZoomOutTool(),
        HoverTool(
            renderers=[buffer_renderer],
            tooltips=[("Address", "@address"), ("Max Size Per Bank", "@max_size_per_bank")],
        ),
    )
    return components(plot)


def create_detailed_l1_buffer_plot(operation_id):
    buffers = list(ttnn.database.query_buffers(get_report_path(), operation_id))
    device_ids = set(buffer.device_id for buffer in buffers)
    if len(buffers) == 0:
        return "", "There are no L1 Buffers!"
    if len(device_ids) != 1:
        return "", "Cannot visualize buffer plot for multiple devices!"
    device_id = device_ids.pop()
    device = ttnn.database.query_device_by_id(get_report_path(), device_id)
    l1_size = device.worker_l1_size

    core_grid = [device.num_y_cores, device.num_x_cores]
    num_cores = math.prod(core_grid)
    core_glyph_height = 100
    core_glyph_width = 10
    core_glyph_y_offset = core_glyph_height + 20
    core_glyph_x_offset = core_glyph_width + 10

    cores_y = []
    cores_x = []
    cores_glyph_y_location = []
    cores_glyph_x_location = []
    for core_y in range(core_grid[0]):
        for core_x in range(core_grid[1]):
            cores_y.append(core_y)
            cores_x.append(core_x)
            cores_glyph_y_location.append(core_y * core_glyph_y_offset)
            cores_glyph_x_location.append(core_x * core_glyph_x_offset)

    cores_glyph_y_location = np.asarray(cores_glyph_y_location)
    cores_glyph_x_location = np.asarray(cores_glyph_x_location)
    cores_height = np.full((num_cores,), core_glyph_height)
    cores_width = np.full((num_cores,), core_glyph_width)
    cores_data_source = ColumnDataSource(
        dict(
            glyph_y_location=cores_glyph_y_location,
            glyph_x_location=cores_glyph_x_location,
            glyph_height=cores_height,
            glyph_width=cores_width,
            core_y=cores_y,
            core_x=cores_x,
        )
    )

    buffer_pages_glyph_y_location = []
    buffer_pages_glyph_x_location = []
    buffer_pages_height = []
    buffer_pages_width = []
    buffer_pages_color = []

    num_buffer_pages = 0
    for buffer_page in ttnn.database.query_buffer_pages(get_report_path(), operation_id):
        if (buffer_page.device_id, buffer_page.address, buffer_page.buffer_type) not in BUFFER_TO_COLOR_INDEX:
            BUFFER_TO_COLOR_INDEX[(buffer_page.device_id, buffer_page.address, buffer_page.buffer_type)] = len(
                BUFFER_TO_COLOR_INDEX
            )

        buffer_page_glyph_y_location = buffer_page.core_y * core_glyph_y_offset + core_glyph_height // 2
        buffer_page_glyph_y_location = (
            buffer_page_glyph_y_location - (l1_size - buffer_page.page_address) / (l1_size) * core_glyph_height
        )
        buffer_page_glyph_x_location = buffer_page.core_x * core_glyph_x_offset

        buffer_page_glyph_height = buffer_page.page_size / l1_size * core_glyph_height

        buffer_pages_glyph_y_location.append(buffer_page_glyph_y_location)
        buffer_pages_glyph_x_location.append(buffer_page_glyph_x_location)
        buffer_pages_height.append(buffer_page_glyph_height)
        buffer_pages_width.append(core_glyph_width)
        buffer_pages_color.append(
            COLORS[
                BUFFER_TO_COLOR_INDEX[(buffer_page.device_id, buffer_page.address, buffer_page.buffer_type)]
                % len(COLORS)
            ]
        )
        num_buffer_pages += 1
    if num_buffer_pages == 0:
        return (
            "",
            "Detailed L1 Buffer Report is not Available! Set  TTNN_CONFIG_OVERRIDES='{\"enable_detailed_buffer_report\": true}' in your environment",
        )

    buffer_pages_glyph_x_location = np.asarray(buffer_pages_glyph_x_location)
    buffer_pages_glyph_y_location = np.asarray(buffer_pages_glyph_y_location)
    buffer_pages_height = np.asarray(buffer_pages_height)
    buffer_pages_width = np.asarray(buffer_pages_width)
    buffer_pages_data_source = ColumnDataSource(
        dict(
            glyph_y_location=buffer_pages_glyph_y_location,
            glyph_x_location=buffer_pages_glyph_x_location,
            glyph_height=buffer_pages_height,
            glyph_width=buffer_pages_width,
            color=buffer_pages_color,
        )
    )

    plot = Plot(title=None, width=800, height=800, min_border=0, toolbar_location="below")

    plot.y_range = Range1d(-100, 1200)
    plot.x_range = Range1d(-10, 250)

    xaxis = LinearAxis()
    plot.add_layout(xaxis, "below")

    yaxis = LinearAxis()
    plot.add_layout(yaxis, "left")

    plot.yaxis.axis_label = "Core Y"
    plot.xaxis.axis_label = "Core X"

    plot.yaxis.ticker.desired_num_ticks = 1
    plot.yaxis.formatter = CustomJSTickFormatter(
        code=f"""
        return "";
    """
    )
    plot.xaxis.ticker.desired_num_ticks = 1
    plot.xaxis.formatter = CustomJSTickFormatter(
        code=f"""
        return "";
    """
    )

    core_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color="black",
        fill_color="white",
    )
    core_renderer = plot.add_glyph(cores_data_source, core_glyph)

    buffer_page_glyph = Rect(
        y="glyph_y_location",
        x="glyph_x_location",
        height="glyph_height",
        width="glyph_width",
        line_color=None,
        fill_color="color",
    )
    plot.add_glyph(buffer_pages_data_source, buffer_page_glyph)

    plot.add_tools(
        WheelZoomTool(),
        PanTool(),
        ResetTool(),
        ZoomInTool(),
        ZoomOutTool(),
        HoverTool(
            renderers=[core_renderer],
            tooltips=[("Core", "(@core_y, @core_x)")],
        ),
    )
    return components(plot)


@app.route("/operation_buffer_report/<operation_id>")
def operation_buffer_report(operation_id):
    operation, previous_operation, next_operation = ttnn.database.query_operation_by_id_together_with_previous_and_next(
        get_report_path(), operation_id=operation_id
    )

    current_summarized_l1_report_script, current_summarized_l1_report_div = create_summarized_l1_buffer_plot(
        operation_id
    )
    current_detailed_l1_report_script, current_detailed_l1_report_div = create_detailed_l1_buffer_plot(operation_id)

    if previous_operation is not None:
        previous_summarized_l1_report_script, previous_summarized_l1_report_div = create_summarized_l1_buffer_plot(
            previous_operation.operation_id
        )
    else:
        previous_summarized_l1_report_script, previous_summarized_l1_report_div = "", ""

    def get_tensor_color(tensor):
        if (tensor.device_id, tensor.address, tensor.buffer_type) not in BUFFER_TO_COLOR_INDEX:
            return "white"
        color_index = BUFFER_TO_COLOR_INDEX[(tensor.device_id, tensor.address, tensor.buffer_type)] % len(COLORS)
        return COLORS[color_index]

    input_tensor_records = ttnn.database.query_input_tensors(get_report_path(), operation_id=operation_id)
    input_tensor_records = sorted(input_tensor_records, key=lambda tensor: tensor.input_index)
    input_tensors = [
        ttnn.database.query_tensor_by_id(get_report_path(), tensor_id=tensor.tensor_id)
        for tensor in input_tensor_records
    ]

    output_tensor_records = ttnn.database.query_output_tensors(get_report_path(), operation_id=operation_id)
    output_tensor_records = sorted(output_tensor_records, key=lambda tensor: tensor.output_index)
    output_tensors = [
        ttnn.database.query_tensor_by_id(get_report_path(), tensor_id=tensor.tensor_id)
        for tensor in output_tensor_records
    ]

    stack_trace = ttnn.database.query_stack_trace(get_report_path(), operation_id=operation_id)
    stack_trace = shorten_stack_trace(stack_trace)

    return render_template(
        "operation_buffer_report.html",
        operation=operation,
        previous_operation=previous_operation,
        next_operation=next_operation,
        current_summarized_l1_report_script=current_summarized_l1_report_script,
        current_summarized_l1_report_div=current_summarized_l1_report_div,
        previous_summarized_l1_report_script=previous_summarized_l1_report_script,
        previous_summarized_l1_report_div=previous_summarized_l1_report_div,
        current_detailed_l1_report_script=current_detailed_l1_report_script,
        current_detailed_l1_report_div=current_detailed_l1_report_div,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        get_tensor_color=get_tensor_color,
        stack_trace=stack_trace,
    )


@app.route("/operation_graph_report/<operation_id>")
def operation_graph_report(operation_id):
    operation, previous_operation, next_operation = ttnn.database.query_operation_by_id_together_with_previous_and_next(
        get_report_path(), operation_id=operation_id
    )

    # graph = ttnn.database.load_graph(operation_id)

    # import graphviz

    # graphviz_graph = graphviz.Digraph()
    # for node in graph:
    #     attributes = graph.nodes[node]
    #     print(attributes)
    #     node_name = attributes["name"]
    #     graphviz_graph.node(name=f"{node}", label=node_name)
    #     for child in graph[node]:
    #         graphviz_graph.edge(f"{node}", f"{child}")

    # return graphviz_graph.pipe(format="svg").decode("utf-8")
    svg_file = get_report_path() / ttnn.database.GRAPHS_PATH / f"{operation_id}.svg"
    if not svg_file.exists():
        return "Graph not found! Was TTNN_CONFIG_OVERRIDES='{\"enable_graph_report\": true}' set?"
    with open(svg_file) as f:
        graph_svg = f.read()
    return render_template(
        "operation_graph_report.html",
        operation=operation,
        previous_operation=previous_operation,
        next_operation=next_operation,
        graph_svg=graph_svg,
    )


@app.route("/operation_tensor_report/<operation_id>")
def operation_tensor_report(operation_id):
    operation, previous_operation, next_operation = ttnn.database.query_operation_by_id_together_with_previous_and_next(
        get_report_path(), operation_id=operation_id
    )

    operation_arguments = list(ttnn.database.query_operation_arguments(get_report_path(), operation_id=operation_id))

    input_tensor_records = ttnn.database.query_input_tensors(get_report_path(), operation_id=operation_id)
    input_tensor_records = sorted(input_tensor_records, key=lambda tensor: tensor.input_index)
    input_tensors = [
        ttnn.database.query_tensor_by_id(get_report_path(), tensor_id=tensor.tensor_id)
        for tensor in input_tensor_records
    ]

    output_tensor_records = ttnn.database.query_output_tensors(get_report_path(), operation_id=operation_id)
    output_tensor_records = sorted(output_tensor_records, key=lambda tensor: tensor.output_index)
    output_tensors = [
        ttnn.database.query_tensor_by_id(get_report_path(), tensor_id=tensor.tensor_id)
        for tensor in output_tensor_records
    ]

    def query_producer_operation_id(tensor_record):
        return ttnn.database.query_producer_operation_id(get_report_path(), tensor_record.tensor_id)

    def query_consumer_operation_ids(tensor_record):
        return ttnn.database.query_consumer_operation_ids(get_report_path(), tensor_record.tensor_id)

    def display_operation_name(operation_id):
        operation = ttnn.database.query_operation_by_id(get_report_path(), operation_id)
        return f"{operation_id}: {operation.name}"

    def load_golden_input_tensors(table_name):
        golden_input_tensors = {}
        for input_tensor_record in input_tensor_records:
            tensor_comparison_record = ttnn.database.query_tensor_comparison_record(
                get_report_path(), table_name, tensor_id=input_tensor_record.tensor_id
            )

            if tensor_comparison_record is None:
                continue
            tensor_record = ttnn.database.query_tensor_by_id(
                get_report_path(), tensor_id=tensor_comparison_record.golden_tensor_id
            )
            golden_input_tensors[input_tensor_record.input_index] = (
                tensor_record,
                tensor_comparison_record,
            )
        return golden_input_tensors

    def load_golden_output_tensors(table_name):
        golden_output_tensors = {}
        for output_tensor_record in output_tensor_records:
            tensor_comparison_record = ttnn.database.query_tensor_comparison_record(
                get_report_path(), table_name, tensor_id=output_tensor_record.tensor_id
            )

            if tensor_comparison_record is None:
                continue
            tensor_record = ttnn.database.query_tensor_by_id(
                get_report_path(), tensor_id=tensor_comparison_record.golden_tensor_id
            )
            golden_output_tensors[output_tensor_record.output_index] = (
                tensor_record,
                tensor_comparison_record,
            )
        return golden_output_tensors

    global_golden_input_tensors = load_golden_input_tensors("global_tensor_comparison_records")
    local_golden_output_tensors = load_golden_output_tensors("local_tensor_comparison_records")
    global_golden_output_tensors = load_golden_output_tensors("global_tensor_comparison_records")

    def display_tensor_comparison_record(record):
        percentage = tensor_comparison_record_to_percentage(record)
        bgcolor = red_to_green_spectrum(percentage)
        return f"""
            <td bgcolor="{bgcolor}">
                Desired PCC = {record.desired_pcc}<br>Actual PCC = {record.actual_pcc}
            </td>
        """

    def plot_tensor(tensor_record):
        import torch

        if tensor_record is None:
            return "", "", ""

        file_name = ttnn.database.get_tensor_file_name_by_id(get_report_path(), tensor_record.tensor_id)
        tensor = ttnn.database.load_tensor_by_id(get_report_path(), tensor_record.tensor_id)
        if tensor is None:
            return "", "", ""

        if isinstance(tensor, ttnn.Tensor):
            tensor = ttnn.to_torch(tensor)

        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        tensor = tensor.detach().numpy()

        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        elif tensor.ndim == 2:
            tensor = tensor
        elif tensor.ndim == 3:
            tensor = tensor[0]
        elif tensor.ndim == 4:
            tensor = tensor[0, 0]
        else:
            raise ValueError(f"Unsupported tensor shape {tensor.shape}")

        tensor = tensor[:1024, :1024]

        plot = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], height=400, width=400)
        plot.x_range.range_padding = plot.y_range.range_padding = 0

        # must give a vector of image data for image parameter
        plot.image(image=[tensor], x=0, y=0, dw=tensor.shape[-1], dh=tensor.shape[-2], palette="Viridis10")

        script, div = components(plot)
        return script, div, file_name

    def get_tensor_attributes(tensor_record):
        if tensor_record is None:
            return ""
        output = "<table>"
        output += f"""
            <tr><th>Shape</th><th>{tensor_record.shape}</th></tr>
            <tr><th>Dtype<t/h><th>{tensor_record.dtype}</th></tr>
            <tr><th>Layout</th><th>{tensor_record.layout}</th></tr>
        """
        if tensor_record.memory_config is not None:
            tensor_memory_config = tensor_record.memory_config.replace("(", "(<br>").replace(",", ",<br>")
            output += f"""
                <tr><th>Device<t/h><th>{tensor_record.device_id}</th></tr>
                <tr><th>Memory Config</th><th>{tensor_memory_config}</th></tr>
            """
        output += "</table>"
        return output

    def get_tensor_statistics(tensor_record):
        if tensor_record is None:
            return ""

        tensor = ttnn.database.load_tensor_by_id(get_report_path(), tensor_record.tensor_id)
        if tensor is None:
            return ""

        if isinstance(tensor, ttnn.Tensor):
            tensor = ttnn.to_torch(tensor)

        tensor = tensor.double()

        statistics = {
            "Min": tensor.min().item(),
            "Max": tensor.max().item(),
            "Mean": tensor.mean().item(),
            "Std": tensor.std().item(),
            "Var": tensor.var().item(),
        }

        output = "<table>"
        for key, value in statistics.items():
            output += f"<tr><th>{key}</th><th>{value}</th></tr>"
        output += "</table>"
        return output

    stack_trace = ttnn.database.query_stack_trace(get_report_path(), operation_id=operation_id)
    stack_trace = shorten_stack_trace(stack_trace)

    return render_template(
        "operation_tensor_report.html",
        operation=operation,
        previous_operation=previous_operation,
        next_operation=next_operation,
        operation_arguments=operation_arguments,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        global_golden_input_tensors=global_golden_input_tensors,
        local_golden_output_tensors=local_golden_output_tensors,
        global_golden_output_tensors=global_golden_output_tensors,
        display_tensor_comparison_record=display_tensor_comparison_record,
        plot_tensor=plot_tensor,
        get_tensor_attributes=get_tensor_attributes,
        get_tensor_statistics=get_tensor_statistics,
        stack_trace=stack_trace,
        query_producer_operation_id=query_producer_operation_id,
        query_consumer_operation_ids=query_consumer_operation_ids,
        display_operation_name=display_operation_name,
    )


@app.route("/operation_stack_trace/<operation_id>")
def operation_stack_trace(operation_id):
    operation, previous_operation, next_operation = ttnn.database.query_operation_by_id_together_with_previous_and_next(
        get_report_path(), operation_id=operation_id
    )
    stack_trace = ttnn.database.query_stack_trace(get_report_path(), operation_id=operation_id)
    return render_template(
        "operation_stack_trace.html",
        operation=operation,
        previous_operation=previous_operation,
        next_operation=next_operation,
        stack_trace=stack_trace,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
