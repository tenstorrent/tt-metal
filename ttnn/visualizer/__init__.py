# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import sqlite3

from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Rect, Range1d, WheelZoomTool
from flask import Flask, render_template
import numpy as np

import ttnn


@dataclasses.dataclass
class DeviceInfo:
    device_id: int
    l1_num_banks: int
    l1_bank_size: int
    num_y_compute_cores: int
    num_x_compute_cores: int
    address_at_first_l1_bank: int
    address_at_first_l1_cb_buffer: int
    num_banks_per_storage_core: int
    num_compute_cores: int
    num_storage_cores: int
    total_l1_memory: int
    total_l1_for_tensors: int
    total_l1_for_interleaved_buffers: int
    total_l1_for_sharded_buffers: int
    cb_limit: int


@dataclasses.dataclass
class Operation:
    operation_id: int
    name: str


@dataclasses.dataclass
class BufferPage:
    operation_id: int
    address: int
    device_id: int
    core_y: int
    core_x: int
    bank_id: int
    page_index: int
    page_address: int
    page_size: int
    buffer_type: int


app = Flask(__name__)


def query_operations():
    sqlite_connection = sqlite3.connect(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM operations")
    for row in cursor.fetchall():
        operation = Operation(*row)
        yield operation

    sqlite_connection.close()


def query_buffer_pages(operation_id):
    sqlite_connection = sqlite3.connect(ttnn.database.DATABASE_FILE)
    cursor = sqlite_connection.cursor()

    cursor.execute("SELECT * FROM buffer_pages WHERE operation_id = ?", (operation_id,))
    for row in cursor.fetchall():
        yield BufferPage(*row)

    sqlite_connection.close()


ADDRESS_TO_ID = {}
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black"]


@app.route("/")
def root():
    return render_template("index.html")


@app.route("/operations")
def operations():
    operations = [(operation_index, operation) for operation_index, operation in enumerate(query_operations())]
    return render_template("operations.html", operations=operations)


@app.route("/l1_visualizer/<operation_id>")
def l1_visualizer(operation_id):
    core_grid = [10, 12]
    core_height = 100
    core_width = 10
    core_height_offset = core_height + 10
    core_width_offset = core_width + 10

    cores_y = []
    cores_x = []
    for y in range(core_grid[0]):
        for x in range(core_grid[1]):
            cores_y.append(y * core_height_offset)
            cores_x.append(x * core_width_offset)

    cores_x = np.asarray(cores_x)
    cores_y = np.asarray(cores_y)
    cores_h = np.asarray([core_height for _ in range(len(cores_y))])
    cores_w = np.asarray([core_width for _ in range(len(cores_x))])
    cores = ColumnDataSource(dict(core_x=cores_x, core_y=cores_y, core_w=cores_w, core_h=cores_h))

    buffers_y = []
    buffers_x = []
    buffers_h = []
    buffers_w = []
    buffers_colors = []

    for buffer_page in query_buffer_pages(operation_id):
        if buffer_page.address not in ADDRESS_TO_ID:
            ADDRESS_TO_ID[buffer_page.address] = len(ADDRESS_TO_ID)

        core_y = buffer_page.core_y * core_height_offset + core_height // 2
        core_x = buffer_page.core_x * core_width_offset

        y = core_y
        one_mb = 1024 * 1024
        y = core_y - (one_mb - buffer_page.page_address) / (one_mb) * core_height
        x = core_x
        h = buffer_page.page_size / (1024 * 1024) * core_height

        buffers_y.append(y)
        buffers_x.append(x)
        buffers_h.append(h)
        buffers_w.append(core_width)
        buffers_colors.append(COLORS[ADDRESS_TO_ID[buffer_page.address] % len(COLORS)])

    buffers_x = np.asarray(buffers_x)
    buffers_y = np.asarray(buffers_y)
    buffers_h = np.asarray(buffers_h)
    buffers_w = np.asarray(buffers_w)
    buffers = ColumnDataSource(
        dict(buffer_x=buffers_x, buffer_y=buffers_y, buffer_w=buffers_w, buffer_h=buffers_h, color=buffers_colors)
    )

    plot = Plot(title=None, width=1000, height=1000, min_border=0, toolbar_location="below")
    plot.add_tools(WheelZoomTool())
    plot.y_range = Range1d(-100, 1200)
    plot.x_range = Range1d(-10, 250)

    glyph = Rect(x="core_x", y="core_y", width="core_w", height="core_h", line_color="black", fill_color="white")
    plot.add_glyph(cores, glyph)

    glyph = Rect(x="buffer_x", y="buffer_y", width="buffer_w", height="buffer_h", line_color=None, fill_color="color")
    plot.add_glyph(buffers, glyph)

    xaxis = LinearAxis()
    plot.add_layout(xaxis, "below")

    yaxis = LinearAxis()
    plot.add_layout(yaxis, "left")

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    return file_html(plot, CDN, "my plot")


@app.route("/operation_graph/<operation_id>")
def operation_graph(operation_id):
    svg_file = ttnn.REPORTS_PATH / "graphs" / f"{operation_id}.svg"
    if not svg_file.exists():
        return "Graph not found! Did you set TTNN_ENABLE_GRAPH_REPORT=True in your environment?"
    with open(svg_file) as f:
        content = f.read()
    return content


@app.route("/operation_codegen/<operation_id>")
def operation_codegen(operation_id):
    # codegen_file = ttnn.REPORTS_PATH / "codegen" / f"{operation_id}.py"
    # with open(codegen_file) as f:
    #     content = f.read()
    content = "# WIP"
    return render_template("operation_codegen.html", content=content)


if __name__ == "__main__":
    app.run(debug=True)
