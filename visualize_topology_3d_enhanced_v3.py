#!/usr/bin/env python3
"""
Enhanced 3D Topology Visualizer v3
- Straight lines within same board
- Curved lines between different boards
- Battery-style grid layout for boards
- Click for coordinates
"""

import yaml
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import math

try:
    import plotly.graph_objects as go
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
    """Parse physical topology and organize by boards"""
    chips = {}
    connections = []
    boards = defaultdict(lambda: {"chips": [], "type": "unknown"})
    chip_to_board = {}  # Track which board each chip belongs to

    # Parse chips
    if "arch" in cluster_desc:
        for chip_id, arch in cluster_desc["arch"].items():
            chip_id = int(chip_id)
            chips[chip_id] = {
                "id": chip_id,
                "arch": arch,
                "location": cluster_desc.get("chips", {}).get(str(chip_id), [0, 0, 0, 0]),
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

                # Map chips to boards
                for chip_id in board_chips:
                    chip_to_board[chip_id] = board_id

    # If no board info, infer from chip_to_boardtype
    if not boards and "chip_to_boardtype" in cluster_desc:
        chip_to_board_type = cluster_desc["chip_to_boardtype"]
        board_type_to_chips = defaultdict(list)

        for chip_id, board_type in chip_to_board_type.items():
            board_type_to_chips[board_type].append(int(chip_id))

        for board_idx, (board_type, chip_list) in enumerate(board_type_to_chips.items()):
            board_id = f"{board_type}_{board_idx}"
            boards[board_id] = {"chips": sorted(chip_list), "type": board_type}

            # Map chips to boards
            for chip_id in chip_list:
                chip_to_board[chip_id] = board_id

    # Parse ethernet connections
    if "ethernet_connections" in cluster_desc:
        for conn in cluster_desc["ethernet_connections"]:
            if len(conn) == 2:
                src = conn[0]
                dst = conn[1]
                connections.append(
                    {"src_chip": src["chip"], "src_chan": src["chan"], "dst_chip": dst["chip"], "dst_chan": dst["chan"]}
                )

    return chips, connections, boards, chip_to_board


def get_battery_grid_layout(num_boards):
    """
    Arrange boards in a battery-style grid (2x2, 2x3, 2x4, etc.)
    Like battery cells in a pack
    """
    if num_boards == 1:
        return {0: (0, 0, 0)}
    elif num_boards == 2:
        # Side by side (1x2)
        spacing = 10
        return {0: (0, 0, 0), 1: (spacing, 0, 0)}
    elif num_boards == 4:
        # 2x2 grid (like 4 AA batteries)
        spacing = 10
        return {0: (0, 0, 0), 1: (spacing, 0, 0), 2: (0, spacing, 0), 3: (spacing, spacing, 0)}
    elif num_boards <= 6:
        # 2x3 grid
        spacing = 10
        cols = 3
        positions = {}
        for i in range(num_boards):
            row = i // cols
            col = i % cols
            positions[i] = (col * spacing, row * spacing, 0)
        return positions
    elif num_boards <= 8:
        # 2x4 grid (your T3K case)
        spacing = 10
        cols = 4
        positions = {}
        for i in range(num_boards):
            row = i // cols
            col = i % cols
            positions[i] = (col * spacing, row * spacing, 0)
        return positions
    else:
        # General grid for larger systems
        cols = int(np.ceil(np.sqrt(num_boards)))
        spacing = 10
        positions = {}
        for i in range(num_boards):
            row = i // cols
            col = i % cols
            positions[i] = (col * spacing, row * spacing, 0)
        return positions


def get_chip_position_in_board(chip_idx, num_chips_in_board, board_type):
    """Calculate relative position of chip within its board"""
    board_type = board_type.lower()

    if "n300" in board_type or num_chips_in_board == 2:
        # N300: 2 chips side by side
        return chip_idx * 2.5, 0, 1
    elif "n150" in board_type or "p150" in board_type or num_chips_in_board == 1:
        # N150/P150: 1 chip centered
        return 0, 0, 1
    elif "galaxy" in board_type or num_chips_in_board == 32:
        # Galaxy: 8x4 grid
        cols = 8
        row = chip_idx // cols
        col = chip_idx % cols
        return col * 1.5, row * 1.5, 1
    elif num_chips_in_board == 8:
        # T3K: 4x2 grid
        cols = 4
        row = chip_idx // cols
        col = chip_idx % cols
        return col * 2, row * 2, 1
    else:
        # Generic grid layout
        cols = int(np.ceil(np.sqrt(num_chips_in_board)))
        row = chip_idx // cols
        col = chip_idx % cols
        return col * 2, row * 2, 1


def get_board_dimensions(board_type, num_chips):
    """Get dimensions for board 3D box"""
    board_type = board_type.lower()

    if "n300" in board_type or num_chips == 2:
        return 6, 3, 2.5  # width, depth, height
    elif "n150" in board_type or "p150" in board_type or num_chips == 1:
        return 3, 3, 2.5
    elif "galaxy" in board_type or num_chips == 32:
        return 13, 7, 2.5
    elif num_chips == 8:
        return 10, 6, 2.5
    else:
        cols = int(np.ceil(np.sqrt(num_chips)))
        rows = int(np.ceil(num_chips / cols))
        return cols * 2 + 1, rows * 2 + 1, 2.5


def create_curved_line(p1, p2, num_points=50, lift_factor=2.0):
    """Create a curved line between two points"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Calculate distance
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Control point at midpoint but lifted
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    mid_z = max(z1, z2) + dist * lift_factor * 0.12  # Lift proportional to distance

    # Generate points along Bezier curve
    t = np.linspace(0, 1, num_points)

    # Quadratic Bezier curve
    x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * mid_x + t**2 * x2
    y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * mid_y + t**2 * y2
    z = (1 - t) ** 2 * z1 + 2 * (1 - t) * t * mid_z + t**2 * z2

    return x, y, z


def chips_on_same_board(chip1, chip2, chip_to_board):
    """Check if two chips are on the same board"""
    return chip_to_board.get(chip1) == chip_to_board.get(chip2)


def create_3d_box_mesh(center, size, color="lightblue", opacity=0.3, name="", customdata=None):
    """Create a 3D box using mesh3d"""
    cx, cy, cz = center
    dx, dy, dz = size

    vertices = np.array(
        [
            [cx - dx / 2, cy - dy / 2, cz - dz / 2],
            [cx + dx / 2, cy - dy / 2, cz - dz / 2],
            [cx + dx / 2, cy + dy / 2, cz - dz / 2],
            [cx - dx / 2, cy + dy / 2, cz - dz / 2],
            [cx - dx / 2, cy - dy / 2, cz + dz / 2],
            [cx + dx / 2, cy - dy / 2, cz + dz / 2],
            [cx + dx / 2, cy + dy / 2, cz + dz / 2],
            [cx - dx / 2, cy + dy / 2, cz + dz / 2],
        ]
    )

    faces = [
        [0, 1, 2],
        [0, 2, 3],  # bottom
        [4, 5, 6],
        [4, 6, 7],  # top
        [0, 1, 5],
        [0, 5, 4],  # front
        [2, 3, 7],
        [2, 7, 6],  # back
        [0, 3, 7],
        [0, 7, 4],  # left
        [1, 2, 6],
        [1, 6, 5],  # right
    ]

    i, j, k = zip(*faces)

    if customdata is None:
        customdata = [[cx, cy, cz]]

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name=name,
        hovertemplate=f"<b>{name}</b><br>Center: ({cx:.2f}, {cy:.2f}, {cz:.2f})<br><extra></extra>",
        showlegend=True,
        legendgroup=name,
        customdata=customdata,
    )


def create_board_wireframe(center, size, color="blue"):
    """Create wireframe edges for a board box"""
    cx, cy, cz = center
    dx, dy, dz = size

    vertices = [
        [cx - dx / 2, cy - dy / 2, cz - dz / 2],
        [cx + dx / 2, cy - dy / 2, cz - dz / 2],
        [cx + dx / 2, cy + dy / 2, cz - dz / 2],
        [cx - dx / 2, cy + dy / 2, cz - dz / 2],
        [cx - dx / 2, cy - dy / 2, cz + dz / 2],
        [cx + dx / 2, cy - dy / 2, cz + dz / 2],
        [cx + dx / 2, cy + dy / 2, cz + dz / 2],
        [cx - dx / 2, cy + dy / 2, cz + dz / 2],
    ]

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical
    ]

    lines = []
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        lines.append(
            go.Scatter3d(
                x=[v1[0], v2[0]],
                y=[v1[1], v2[1]],
                z=[v1[2], v2[2]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return lines


def create_enhanced_3d_viz(
    chips, connections, boards, chip_to_board, mesh_desc=None, output_file="topology_3d_battery.html"
):
    """Create enhanced 3D visualization with battery layout"""

    traces = []
    chip_positions = {}

    # Color schemes
    board_colors = {
        "n150": "lightblue",
        "n300": "lightgreen",
        "p150": "lightcoral",
        "p300": "lightsalmon",
        "galaxy": "lightgoldenrodyellow",
        "t3k": "lightcyan",
        "unknown": "lightgray",
    }

    # Get battery-style grid layout
    board_list = list(boards.items())
    board_positions = get_battery_grid_layout(len(board_list))

    # Process each board
    for board_idx, (board_id, board_info) in enumerate(board_list):
        board_type = board_info["type"].lower()
        board_chips = board_info["chips"]
        num_chips = len(board_chips)

        if num_chips == 0:
            continue

        # Get board position from battery grid
        board_x, board_y, board_z = board_positions[board_idx]

        # Get board dimensions
        board_width, board_depth, board_height = get_board_dimensions(board_type, num_chips)

        # Get board color
        color_key = next((k for k in board_colors.keys() if k in board_type), "unknown")
        board_color = board_colors[color_key]

        # Create board box
        board_center = (board_x + board_width / 2 - 0.5, board_y + board_depth / 2 - 0.5, board_height / 2)
        board_trace = create_3d_box_mesh(
            board_center,
            (board_width, board_depth, board_height),
            color=board_color,
            opacity=0.2,
            name=f'{board_info["type"]} ({num_chips} chips)',
            customdata=[[board_center[0], board_center[1], board_center[2]]],
        )
        traces.append(board_trace)

        # Add board wireframe
        wireframe_traces = create_board_wireframe(
            board_center, (board_width, board_depth, board_height), color="darkblue"
        )
        traces.extend(wireframe_traces)

        # Add chips inside the board
        chip_x_coords = []
        chip_y_coords = []
        chip_z_coords = []
        chip_texts = []
        chip_hovertexts = []
        chip_customdata = []

        for idx, chip_id in enumerate(board_chips):
            if chip_id in chips:
                # Get chip position relative to board
                rel_x, rel_y, rel_z = get_chip_position_in_board(idx, num_chips, board_type)

                # Absolute position
                abs_x = board_x + rel_x
                abs_y = board_y + rel_y
                abs_z = rel_z

                chip_positions[chip_id] = (abs_x, abs_y, abs_z)

                chip_x_coords.append(abs_x)
                chip_y_coords.append(abs_y)
                chip_z_coords.append(abs_z)
                chip_texts.append(str(chip_id))
                chip_hovertexts.append(
                    f"<b>Chip {chip_id}</b><br>"
                    f"Arch: {chips[chip_id]['arch']}<br>"
                    f"Board: {board_info['type']}<br>"
                    f"Position: ({abs_x:.2f}, {abs_y:.2f}, {abs_z:.2f})"
                )
                chip_customdata.append([abs_x, abs_y, abs_z, chip_id])

        # Add all chips in this board
        if chip_x_coords:
            chip_trace = go.Scatter3d(
                x=chip_x_coords,
                y=chip_y_coords,
                z=chip_z_coords,
                mode="markers+text",
                marker=dict(size=15, color="darkblue", line=dict(color="white", width=2), symbol="square"),
                text=chip_texts,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                hovertext=chip_hovertexts,
                hoverinfo="text",
                name="Chips",
                showlegend=False,
                customdata=chip_customdata,
            )
            traces.append(chip_trace)

    # Add ethernet connections
    # Separate intra-board (straight) from inter-board (curved)
    for conn in connections:
        src = conn["src_chip"]
        dst = conn["dst_chip"]

        if src in chip_positions and dst in chip_positions:
            src_pos = chip_positions[src]
            dst_pos = chip_positions[dst]

            # Check if chips are on same board
            same_board = chips_on_same_board(src, dst, chip_to_board)

            if same_board:
                # Straight line for intra-board connections
                line_color = "blue"
                line_width = 2
                line_dash = "solid"

                edge_trace = go.Scatter3d(
                    x=[src_pos[0], dst_pos[0]],
                    y=[src_pos[1], dst_pos[1]],
                    z=[src_pos[2], dst_pos[2]],
                    mode="lines",
                    line=dict(color=line_color, width=line_width, dash=line_dash),
                    hovertemplate=(
                        f"<b>Intra-board Link</b><br>"
                        f"Chip {src}[ch{conn['src_chan']}] ↔ "
                        f"Chip {dst}[ch{conn['dst_chan']}]<br>"
                        f"<extra></extra>"
                    ),
                    showlegend=False,
                )
                traces.append(edge_trace)
            else:
                # Curved line for inter-board connections
                x_curve, y_curve, z_curve = create_curved_line(src_pos, dst_pos)

                mid_idx = len(x_curve) // 2
                mid_x, mid_y, mid_z = x_curve[mid_idx], y_curve[mid_idx], z_curve[mid_idx]

                edge_trace = go.Scatter3d(
                    x=x_curve,
                    y=y_curve,
                    z=z_curve,
                    mode="lines",
                    line=dict(color="gray", width=3),
                    hovertemplate=(
                        f"<b>Inter-board Link</b><br>"
                        f"Chip {src}[ch{conn['src_chan']}] ↔ "
                        f"Chip {dst}[ch{conn['dst_chan']}]<br>"
                        f"Position: ({mid_x:.2f}, {mid_y:.2f}, {mid_z:.2f})"
                        f"<extra></extra>"
                    ),
                    showlegend=False,
                    customdata=[[mid_x, mid_y, mid_z]],
                )
                traces.append(edge_trace)

    # Add mesh overlay if provided
    if mesh_desc and "Mesh" in mesh_desc:
        for mesh in mesh_desc["Mesh"]:
            mesh_id = mesh.get("id", 0)
            dev_topo = mesh.get("device_topology", [])
            board_name = mesh.get("board", "unknown")

            if len(dev_topo) >= 2:
                rows, cols = dev_topo[0], dev_topo[1]
                mesh_chip_count = rows * cols

                mesh_chip_ids = sorted(chips.keys())[:mesh_chip_count]

                if mesh_chip_ids and all(cid in chip_positions for cid in mesh_chip_ids):
                    mesh_positions = [chip_positions[cid] for cid in mesh_chip_ids]
                    xs = [p[0] for p in mesh_positions]
                    ys = [p[1] for p in mesh_positions]
                    zs = [p[2] for p in mesh_positions]

                    min_x, max_x = min(xs) - 1, max(xs) + 1
                    min_y, max_y = min(ys) - 1, max(ys) + 1
                    min_z, max_z = min(zs) - 0.5, max(zs) + 2

                    mesh_center = ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
                    mesh_size = (max_x - min_x, max_y - min_y, max_z - min_z)

                    mesh_trace = create_3d_box_mesh(
                        mesh_center,
                        mesh_size,
                        color="red",
                        opacity=0.15,
                        name=f"Logical Mesh {mesh_id}<br>({rows}×{cols} = {board_name})",
                        customdata=[[mesh_center[0], mesh_center[1], mesh_center[2]]],
                    )
                    traces.append(mesh_trace)

                    mesh_wireframe = create_board_wireframe(mesh_center, mesh_size, color="red")
                    traces.extend(mesh_wireframe)

    # Layout
    layout = go.Layout(
        title=dict(text="<b>3D Topology - Battery Grid Layout (Click for coordinates)</b>", font=dict(size=20)),
        scene=dict(
            xaxis=dict(title="X", showgrid=True, gridcolor="lightgray"),
            yaxis=dict(title="Y", showgrid=True, gridcolor="lightgray"),
            zaxis=dict(title="Z (Height)", showgrid=True, gridcolor="lightgray"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
            aspectmode="data",
        ),
        hovermode="closest",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig = go.Figure(data=traces, layout=layout)

    # Add JavaScript for click handling
    click_js = """
    <script>
    var myPlot = document.getElementById('plotly-div');

    myPlot.on('plotly_click', function(data){
        var pt = data.points[0];
        var coord_text = '';

        if (pt.customdata) {
            var x = pt.customdata[0];
            var y = pt.customdata[1];
            var z = pt.customdata[2];
            coord_text = 'Coordinates: (' + x.toFixed(2) + ', ' + y.toFixed(2) + ', ' + z.toFixed(2) + ')';

            if (pt.customdata.length > 3) {
                var chip_id = pt.customdata[3];
                coord_text = 'Chip ' + chip_id + ' - ' + coord_text;
            }
        } else {
            coord_text = 'Coordinates: (' + pt.x.toFixed(2) + ', ' + pt.y.toFixed(2) + ', ' + pt.z.toFixed(2) + ')';
        }

        console.log('Clicked:', coord_text);

        var update = {
            title: '<b>3D Topology - Battery Grid Layout</b><br><span style="font-size:14px;color:blue;">' + coord_text + '</span>'
        };
        Plotly.relayout('plotly-div', update);

        alert(coord_text);
    });

    myPlot.on('plotly_doubleclick', function(data){
        var update = {
            title: '<b>3D Topology - Battery Grid Layout (Click for coordinates)</b>'
        };
        Plotly.relayout('plotly-div', update);
    });
    </script>
    """

    html_content = fig.to_html(
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["toImage"],
            "modeBarButtonsToAdd": ["hoverclosest", "hovercompare"],
        },
        div_id="plotly-div",
    )

    html_content = html_content.replace("</body>", click_js + "</body>")

    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"✅ Enhanced 3D visualization saved to: {output_file}")
    print(f"   Open in browser: file://{Path(output_file).absolute()}")
    print(f"\n🎮 Controls:")
    print(f"   • Left-click + drag: Rotate")
    print(f"   • Scroll: Zoom in/out")
    print(f"   • Right-click + drag: Pan")
    print(f"   • Click element: See coordinates")
    print(f"   • Double-click: Reset view")
    print(f"\n✨ Features:")
    print(f"   • Battery-style grid layout (2x4 for your T3K)")
    print(f"   • Straight BLUE lines for chips on same board")
    print(f"   • Curved GRAY lines for chips on different boards")
    print(f"   • Click any element for coordinates")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced 3D Topology Visualizer v3 - Battery Grid Layout",
        epilog="""
Features:
  • Battery-style grid layout (like battery cells in a pack)
  • Straight lines within same board
  • Curved lines between different boards
  • Click for coordinates

Examples:
  ./visualize_topology_3d_enhanced_v3.py --cluster t3k_phys_topology.yaml \\
      --mesh t3k_generated_mesh.yaml
        """,
    )

    parser.add_argument("--cluster", required=True, help="Path to cluster descriptor YAML")
    parser.add_argument("--mesh", help="Path to mesh graph descriptor YAML")
    parser.add_argument(
        "--output", default="topology_3d_battery.html", help="Output file (default: topology_3d_battery.html)"
    )

    args = parser.parse_args()

    print(f"📂 Loading cluster descriptor: {args.cluster}")
    try:
        cluster_desc = load_cluster_descriptor(args.cluster)
        chips, connections, boards, chip_to_board = parse_physical_topology(cluster_desc)
        print(f"   Found {len(chips)} chips in {len(boards)} boards")
        print(f"   Found {len(connections)} ethernet connections")

        for board_id, board_info in boards.items():
            print(f"   • {board_info['type']}: {len(board_info['chips'])} chips")
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
            if "Mesh" in mesh_desc:
                for mesh in mesh_desc["Mesh"]:
                    topo = mesh.get("device_topology", [])
                    print(f"   Mesh {mesh.get('id', 0)}: {topo[0]}×{topo[1]}")
        except Exception as e:
            print(f"⚠️  Warning: {e}")

    print(f"\n🎨 Generating battery-grid 3D visualization...")
    success = create_enhanced_3d_viz(chips, connections, boards, chip_to_board, mesh_desc, args.output)

    if success:
        print(f"\n🎉 Done!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
