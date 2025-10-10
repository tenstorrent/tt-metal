#!/usr/bin/env python3
"""
Interactive Topology Visualizer with Enhanced UI
- Uses actual physical coordinates (shelf, rack_x, rack_y, adapter)
- Easy element selection with hover highlighting
- Selectable ethernet connections
- Side panel with detailed info
- Multiple views: Physical rack layout, logical mesh
"""

import yaml
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("ERROR: plotly not installed. Install with: pip install plotly")
    sys.exit(1)


def load_cluster_descriptor(yaml_path):
    """Load physical cluster descriptor"""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def load_mesh_graph_descriptor(yaml_path):
    """Load mesh graph descriptor"""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def parse_physical_topology(cluster_desc):
    """Parse physical topology with actual rack coordinates"""
    chips = {}
    connections = []
    boards = defaultdict(lambda: {"chips": [], "type": "unknown"})
    chip_to_board = {}

    # Parse chips with physical locations
    if "arch" in cluster_desc:
        for chip_id, arch in cluster_desc["arch"].items():
            chip_id = int(chip_id)
            # Get physical location [shelf, rack_x, rack_y, adapter]
            location = cluster_desc.get("chips", {}).get(str(chip_id), [0, 0, 0, 0])
            chips[chip_id] = {
                "id": chip_id,
                "arch": arch,
                "shelf": location[0] if len(location) > 0 else 0,
                "rack_x": location[1] if len(location) > 1 else 0,
                "rack_y": location[2] if len(location) > 2 else 0,
                "adapter": location[3] if len(location) > 3 else 0,
                "location": location,
            }

    # Parse board information
    if "boards" in cluster_desc:
        for board_idx, board_info in enumerate(cluster_desc["boards"]):
            if isinstance(board_info, list):
                board_id = None
                board_type = "unknown"
                board_chips = []

                for item in board_info:
                    if isinstance(item, dict):
                        if "board_id" in item:
                            board_id = item["board_id"]
                        if "board_type" in item:
                            board_type = item["board_type"]
                        if "chips" in item:
                            board_chips = item["chips"]

                if board_id is None:
                    board_id = f"board_{board_idx}"

                boards[board_id] = {"chips": board_chips, "type": board_type}

                for chip_id in board_chips:
                    chip_to_board[chip_id] = board_id

    # Infer boards from chip_to_boardtype if needed
    if not boards and "chip_to_boardtype" in cluster_desc:
        chip_to_board_type = cluster_desc["chip_to_boardtype"]
        board_type_to_chips = defaultdict(list)

        for chip_id, board_type in chip_to_board_type.items():
            board_type_to_chips[board_type].append(int(chip_id))

        for board_idx, (board_type, chip_list) in enumerate(board_type_to_chips.items()):
            board_id = f"{board_type}_{board_idx}"
            boards[board_id] = {"chips": sorted(chip_list), "type": board_type}

            for chip_id in chip_list:
                chip_to_board[chip_id] = board_id

    # Parse ethernet connections
    if "ethernet_connections" in cluster_desc:
        for conn_idx, conn in enumerate(cluster_desc["ethernet_connections"]):
            if len(conn) == 2:
                src = conn[0]
                dst = conn[1]
                connections.append(
                    {
                        "id": conn_idx,
                        "src_chip": src["chip"],
                        "src_chan": src["chan"],
                        "dst_chip": dst["chip"],
                        "dst_chan": dst["chan"],
                    }
                )

    return chips, connections, boards, chip_to_board


def create_interactive_ui(chips, connections, boards, chip_to_board, mesh_desc=None, output_file="topology_ui.html"):
    """Create interactive UI with multiple views and selection panel"""

    # Prepare data
    chip_data = []
    for chip_id, chip_info in sorted(chips.items()):
        chip_data.append(
            {
                "id": chip_id,
                "arch": chip_info["arch"],
                "shelf": chip_info["shelf"],
                "rack_x": chip_info["rack_x"],
                "rack_y": chip_info["rack_y"],
                "adapter": chip_info["adapter"],
                "board": chip_to_board.get(chip_id, "unknown"),
                "board_type": boards.get(chip_to_board.get(chip_id, ""), {}).get("type", "unknown"),
            }
        )

    connection_data = []
    for conn in connections:
        src_chip = chips.get(conn["src_chip"], {})
        dst_chip = chips.get(conn["dst_chip"], {})
        connection_data.append(
            {
                "id": conn["id"],
                "src_chip": conn["src_chip"],
                "src_chan": conn["src_chan"],
                "dst_chip": conn["dst_chip"],
                "dst_chan": conn["dst_chan"],
                "src_board": chip_to_board.get(conn["src_chip"], "unknown"),
                "dst_board": chip_to_board.get(conn["dst_chip"], "unknown"),
                "same_board": chip_to_board.get(conn["src_chip"]) == chip_to_board.get(conn["dst_chip"]),
            }
        )

    # Create HTML with custom UI
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Interactive Topology Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}

        #main-container {{
            display: flex;
            width: 100%;
            height: 100%;
        }}

        #visualization {{
            flex: 1;
            background: #f5f5f5;
            position: relative;
        }}

        #sidebar {{
            width: 350px;
            background: #2c3e50;
            color: white;
            padding: 20px;
            overflow-y: auto;
            box-shadow: -2px 0 10px rgba(0,0,0,0.3);
        }}

        #sidebar h2 {{
            margin-top: 0;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}

        #sidebar h3 {{
            margin-top: 20px;
            color: #3498db;
            font-size: 16px;
        }}

        .info-section {{
            background: #34495e;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}

        .info-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
        }}

        .info-label {{
            font-weight: bold;
            color: #3498db;
        }}

        .info-value {{
            color: #ecf0f1;
        }}

        #view-selector {{
            margin-bottom: 20px;
        }}

        .view-btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}

        .view-btn:hover {{
            background: #2980b9;
        }}

        .view-btn.active {{
            background: #27ae60;
        }}

        #filter-section {{
            margin-bottom: 20px;
        }}

        .filter-group {{
            margin: 10px 0;
        }}

        .filter-label {{
            display: block;
            margin-bottom: 5px;
            color: #3498db;
            font-size: 14px;
        }}

        .filter-checkbox {{
            margin-right: 8px;
        }}

        #stats {{
            background: #34495e;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}

        .stat-item {{
            margin: 8px 0;
            font-size: 14px;
        }}

        .stat-value {{
            color: #3498db;
            font-weight: bold;
            font-size: 18px;
        }}

        #selection-info {{
            min-height: 200px;
        }}

        .no-selection {{
            color: #95a5a6;
            font-style: italic;
            text-align: center;
            padding: 40px 20px;
        }}

        .chip-badge {{
            display: inline-block;
            background: #3498db;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 12px;
        }}

        .connection-badge {{
            display: inline-block;
            background: #e74c3c;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="main-container">
        <div id="visualization">
            <div id="plot-3d"></div>
            <div id="plot-2d"></div>
        </div>

        <div id="sidebar">
            <h2>🎛️ Topology Control Panel</h2>

            <div id="view-selector">
                <h3>View Mode</h3>
                <button class="view-btn active" onclick="switchView('3d')">3D Rack View</button>
                <button class="view-btn" onclick="switchView('2d')">2D Schematic</button>
            </div>

            <div id="stats">
                <h3>📊 System Stats</h3>
                <div class="stat-item">Chips: <span class="stat-value">{len(chips)}</span></div>
                <div class="stat-item">Boards: <span class="stat-value">{len(boards)}</span></div>
                <div class="stat-item">Connections: <span class="stat-value">{len(connections)}</span></div>
            </div>

            <div id="filter-section">
                <h3>🔍 Filters</h3>
                <div class="filter-group">
                    <label class="filter-label">
                        <input type="checkbox" class="filter-checkbox" id="show-chips" checked onchange="updateView()">
                        Show Chips
                    </label>
                    <label class="filter-label">
                        <input type="checkbox" class="filter-checkbox" id="show-connections" checked onchange="updateView()">
                        Show Connections
                    </label>
                    <label class="filter-label">
                        <input type="checkbox" class="filter-checkbox" id="show-labels" checked onchange="updateView()">
                        Show Labels
                    </label>
                </div>
            </div>

            <div id="selection-info">
                <h3>📌 Selected Element</h3>
                <div class="no-selection">Click on a chip or connection to see details</div>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const chipsData = {json.dumps(chip_data)};
        const connectionsData = {json.dumps(connection_data)};

        let currentView = '3d';
        let selectedElement = null;

        // Create 3D visualization
        function create3DView() {{
            const chips = [];
            const chipX = [], chipY = [], chipZ = [], chipText = [], chipHover = [], chipCustomData = [];

            chipsData.forEach(chip => {{
                chipX.push(chip.rack_x);
                chipY.push(chip.rack_y);
                chipZ.push(chip.shelf);
                chipText.push(chip.id.toString());
                chipHover.push(
                    `<b>Chip ${{chip.id}}</b><br>` +
                    `Arch: ${{chip.arch}}<br>` +
                    `Board: ${{chip.board_type}}<br>` +
                    `Shelf: ${{chip.shelf}}, Rack: (${{chip.rack_x}}, ${{chip.rack_y}})`
                );
                chipCustomData.push({{
                    type: 'chip',
                    ...chip
                }});
            }});

            const chipTrace = {{
                type: 'scatter3d',
                x: chipX,
                y: chipY,
                z: chipZ,
                mode: 'markers+text',
                marker: {{
                    size: 20,
                    color: 'rgb(52, 152, 219)',
                    line: {{ color: 'white', width: 3 }},
                    symbol: 'square'
                }},
                text: chipText,
                textposition: 'middle center',
                textfont: {{ size: 12, color: 'white', family: 'Arial Black' }},
                hovertext: chipHover,
                hoverinfo: 'text',
                name: 'Chips',
                customdata: chipCustomData
            }};

            // Create connection lines
            const connectionTraces = [];
            connectionsData.forEach(conn => {{
                const srcChip = chipsData.find(c => c.id === conn.src_chip);
                const dstChip = chipsData.find(c => c.id === conn.dst_chip);

                if (srcChip && dstChip) {{
                    const color = conn.same_board ? 'rgba(52, 152, 219, 0.6)' : 'rgba(149, 165, 166, 0.6)';
                    const width = conn.same_board ? 4 : 3;

                    connectionTraces.push({{
                        type: 'scatter3d',
                        x: [srcChip.rack_x, dstChip.rack_x],
                        y: [srcChip.rack_y, dstChip.rack_y],
                        z: [srcChip.shelf, dstChip.shelf],
                        mode: 'lines',
                        line: {{ color: color, width: width }},
                        hovertext: `<b>Connection ${{conn.id}}</b><br>` +
                                  `Chip ${{conn.src_chip}}[ch${{conn.src_chan}}] ↔ ` +
                                  `Chip ${{conn.dst_chip}}[ch${{conn.dst_chan}}]<br>` +
                                  `Type: ${{conn.same_board ? 'Intra-board' : 'Inter-board'}}`,
                        hoverinfo: 'text',
                        showlegend: false,
                        customdata: [{{ type: 'connection', ...conn }}]
                    }});
                }}
            }});

            const layout = {{
                title: '<b>3D Physical Rack Layout</b><br><sub>Click elements to select</sub>',
                scene: {{
                    xaxis: {{ title: 'Rack X' }},
                    yaxis: {{ title: 'Rack Y' }},
                    zaxis: {{ title: 'Shelf' }},
                    camera: {{
                        eye: {{ x: 1.5, y: 1.5, z: 1.2 }}
                    }}
                }},
                hovermode: 'closest',
                showlegend: false,
                margin: {{ l: 0, r: 0, b: 0, t: 50 }}
            }};

            const config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            }};

            Plotly.newPlot('plot-3d', [chipTrace, ...connectionTraces], layout, config);

            // Add click event handler
            document.getElementById('plot-3d').on('plotly_click', function(data) {{
                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    if (point.customdata) {{
                        selectElement(point.customdata);
                    }}
                }}
            }});
        }}

        // Create 2D schematic view
        function create2DView() {{
            // Create 2D schematic based on logical layout
            const chipX = [], chipY = [], chipText = [], chipHover = [], chipCustomData = [];

            // Arrange chips in grid based on ID
            chipsData.forEach((chip, idx) => {{
                const cols = Math.ceil(Math.sqrt(chipsData.length));
                const row = Math.floor(idx / cols);
                const col = idx % cols;

                chipX.push(col * 3);
                chipY.push(-row * 3);
                chipText.push(chip.id.toString());
                chipHover.push(
                    `<b>Chip ${{chip.id}}</b><br>` +
                    `Arch: ${{chip.arch}}<br>` +
                    `Board: ${{chip.board_type}}`
                );
                chipCustomData.push({{
                    type: 'chip',
                    ...chip
                }});
            }});

            const chipTrace = {{
                type: 'scatter',
                x: chipX,
                y: chipY,
                mode: 'markers+text',
                marker: {{
                    size: 40,
                    color: 'rgb(52, 152, 219)',
                    line: {{ color: 'white', width: 3 }}
                }},
                text: chipText,
                textposition: 'middle center',
                textfont: {{ size: 14, color: 'white', family: 'Arial Black' }},
                hovertext: chipHover,
                hoverinfo: 'text',
                name: 'Chips',
                customdata: chipCustomData
            }};

            const layout = {{
                title: '<b>2D Schematic View</b><br><sub>Click elements to select</sub>',
                xaxis: {{ visible: false }},
                yaxis: {{ visible: false }},
                hovermode: 'closest',
                showlegend: false,
                margin: {{ l: 20, r: 20, b: 20, t: 50 }}
            }};

            const config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            }};

            Plotly.newPlot('plot-2d', [chipTrace], layout, config);

            document.getElementById('plot-2d').on('plotly_click', function(data) {{
                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    if (point.customdata) {{
                        selectElement(point.customdata);
                    }}
                }}
            }});
        }}

        // Handle element selection
        function selectElement(element) {{
            selectedElement = element;
            updateSelectionInfo();
        }}

        // Update selection info panel
        function updateSelectionInfo() {{
            const infoDiv = document.getElementById('selection-info');

            if (!selectedElement) {{
                infoDiv.innerHTML = '<h3>📌 Selected Element</h3><div class="no-selection">Click on a chip or connection to see details</div>';
                return;
            }}

            let html = '<h3>📌 Selected Element</h3><div class="info-section">';

            if (selectedElement.type === 'chip') {{
                html += `
                    <div class="info-row">
                        <span class="info-label">Type:</span>
                        <span class="info-value"><span class="chip-badge">CHIP</span></span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Chip ID:</span>
                        <span class="info-value">${{selectedElement.id}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Architecture:</span>
                        <span class="info-value">${{selectedElement.arch}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Board Type:</span>
                        <span class="info-value">${{selectedElement.board_type}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Board ID:</span>
                        <span class="info-value">${{selectedElement.board}}</span>
                    </div>
                    <h3>📍 Physical Location</h3>
                    <div class="info-row">
                        <span class="info-label">Shelf:</span>
                        <span class="info-value">${{selectedElement.shelf}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Rack X:</span>
                        <span class="info-value">${{selectedElement.rack_x}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Rack Y:</span>
                        <span class="info-value">${{selectedElement.rack_y}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Adapter:</span>
                        <span class="info-value">${{selectedElement.adapter}}</span>
                    </div>
                `;
            }} else if (selectedElement.type === 'connection') {{
                html += `
                    <div class="info-row">
                        <span class="info-label">Type:</span>
                        <span class="info-value"><span class="connection-badge">CONNECTION</span></span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Connection ID:</span>
                        <span class="info-value">${{selectedElement.id}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Source:</span>
                        <span class="info-value">Chip ${{selectedElement.src_chip}} [ch${{selectedElement.src_chan}}]</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Destination:</span>
                        <span class="info-value">Chip ${{selectedElement.dst_chip}} [ch${{selectedElement.dst_chan}}]</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Scope:</span>
                        <span class="info-value">${{selectedElement.same_board ? 'Intra-board' : 'Inter-board'}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Source Board:</span>
                        <span class="info-value">${{selectedElement.src_board}}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Dest Board:</span>
                        <span class="info-value">${{selectedElement.dst_board}}</span>
                    </div>
                `;
            }}

            html += '</div>';
            infoDiv.innerHTML = html;
        }}

        // Switch between views
        function switchView(view) {{
            currentView = view;

            // Update button states
            document.querySelectorAll('.view-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');

            // Show/hide plots
            if (view === '3d') {{
                document.getElementById('plot-3d').style.display = 'block';
                document.getElementById('plot-2d').style.display = 'none';
            }} else {{
                document.getElementById('plot-3d').style.display = 'none';
                document.getElementById('plot-2d').style.display = 'block';
            }}
        }}

        // Update view based on filters
        function updateView() {{
            // Implement filter logic here
            console.log('Updating view with filters');
        }}

        // Initialize
        window.onload = function() {{
            create3DView();
            create2DView();
            document.getElementById('plot-2d').style.display = 'none';
        }};

        // Handle window resize
        window.onresize = function() {{
            Plotly.Plots.resize('plot-3d');
            Plotly.Plots.resize('plot-2d');
        }};
    </script>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"✅ Interactive UI saved to: {output_file}")
    print(f"   Open in browser: file://{Path(output_file).absolute()}")
    print(f"\n✨ Features:")
    print(f"   • Click chips or connections to see details in sidebar")
    print(f"   • Uses actual physical coordinates (shelf, rack_x, rack_y)")
    print(f"   • Switch between 3D rack view and 2D schematic")
    print(f"   • Larger click targets for easier selection")
    print(f"   • Detailed info panel for selected elements")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Topology Visualizer with Enhanced UI",
        epilog="""
Features:
  • Easy element selection with sidebar details
  • Uses physical rack coordinates (shelf, rack_x, rack_y)
  • Selectable ethernet connections
  • Multiple view modes (3D rack, 2D schematic)

Examples:
  ./visualize_topology_interactive_ui.py --cluster t3k_phys_topology.yaml \\
      --mesh t3k_generated_mesh.yaml
        """,
    )

    parser.add_argument("--cluster", required=True, help="Path to cluster descriptor YAML")
    parser.add_argument("--mesh", help="Path to mesh graph descriptor YAML")
    parser.add_argument("--output", default="topology_ui.html", help="Output file (default: topology_ui.html)")

    args = parser.parse_args()

    print(f"📂 Loading cluster descriptor: {args.cluster}")
    try:
        cluster_desc = load_cluster_descriptor(args.cluster)
        chips, connections, boards, chip_to_board = parse_physical_topology(cluster_desc)
        print(f"   Found {len(chips)} chips in {len(boards)} boards")
        print(f"   Found {len(connections)} ethernet connections")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    mesh_desc = None
    if args.mesh:
        print(f"\n📂 Loading mesh graph descriptor: {args.mesh}")
        try:
            mesh_desc = load_mesh_graph_descriptor(args.mesh)
        except Exception as e:
            print(f"⚠️  Warning: {e}")

    print(f"\n🎨 Generating interactive UI...")
    success = create_interactive_ui(chips, connections, boards, chip_to_board, mesh_desc, args.output)

    if success:
        print(f"\n🎉 Done! Open the file and click on elements to explore.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
