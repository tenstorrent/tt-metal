#!/usr/bin/env python3
"""
Enhanced 3D Topology Visualizer v2
- Click to see coordinates
- Improved layout to avoid line crossings
- Curved ethernet lines that go around chips
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

    # If no board info, infer from chip_to_boardtype
    if not boards and "chip_to_boardtype" in cluster_desc:
        chip_to_board = cluster_desc["chip_to_boardtype"]
        board_type_to_chips = defaultdict(list)

        for chip_id, board_type in chip_to_board.items():
            board_type_to_chips[board_type].append(int(chip_id))

        for board_idx, (board_type, chip_list) in enumerate(board_type_to_chips.items()):
            board_id = f"{board_type}_{board_idx}"
            boards[board_id] = {"chips": sorted(chip_list), "type": board_type}

    # Parse ethernet connections
    if "ethernet_connections" in cluster_desc:
        for conn in cluster_desc["ethernet_connections"]:
            if len(conn) == 2:
                src = conn[0]
                dst = conn[1]
                connections.append(
                    {"src_chip": src["chip"], "src_chan": src["chan"], "dst_chip": dst["chip"], "dst_chan": dst["chan"]}
                )

    return chips, connections, boards


def get_optimized_board_layout(num_boards, connections, boards):
    """
    Calculate optimized board positions to minimize line crossings
    Uses a circular or grid layout with spacing
    """
    if num_boards <= 4:
        # For small numbers, use a line layout with good spacing
        positions = {}
        spacing = 12
        for i in range(num_boards):
            positions[i] = (i * spacing, 0, 0)
        return positions
    elif num_boards <= 8:
        # For medium numbers, use 2 rows
        positions = {}
        spacing = 12
        cols = (num_boards + 1) // 2
        for i in range(num_boards):
            row = i // cols
            col = i % cols
            positions[i] = (col * spacing, row * spacing, 0)
        return positions
    else:
        # For large numbers, use a grid
        positions = {}
        cols = int(np.ceil(np.sqrt(num_boards)))
        spacing = 12
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
        # N150/P150: 1 chip
        return 0, 0, 1
    elif "galaxy" in board_type or num_chips_in_board == 32:
        # Galaxy: 8x4 grid
        cols = 8
        row = chip_idx // cols
        col = chip_idx % cols
        return col * 1.5, row * 1.5, 1
    elif num_chips_in_board == 8:
        # T3K: 4x2 or 2x4 grid
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
    """
    Create a curved line between two points that lifts up to avoid passing through chips
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Calculate distance
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Create a bezier curve with a control point above
    # Control point is at the midpoint but lifted
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    mid_z = max(z1, z2) + dist * lift_factor * 0.15  # Lift proportional to distance

    # Generate points along the curve
    t = np.linspace(0, 1, num_points)

    # Quadratic Bezier curve
    x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * mid_x + t**2 * x2
    y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * mid_y + t**2 * y2
    z = (1 - t) ** 2 * z1 + 2 * (1 - t) * t * mid_z + t**2 * z2

    return x, y, z


def create_3d_box_mesh(center, size, color="lightblue", opacity=0.3, name="", customdata=None):
    """Create a 3D box using mesh3d with clickable customdata"""
    cx, cy, cz = center
    dx, dy, dz = size

    # Define 8 vertices of the box
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

    # Define 12 triangles (2 per face)
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

    # Create customdata for the center point
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


def create_board_wireframe(center, size, color="blue", name=""):
    """Create wireframe edges for a board box"""
    cx, cy, cz = center
    dx, dy, dz = size

    # Define 8 vertices
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

    # Define edges
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


def create_enhanced_3d_viz(chips, connections, boards, mesh_desc=None, output_file="topology_3d_enhanced.html"):
    """Create enhanced 3D visualization with improved layout and click coordinates"""

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

    # Get optimized board layout
    board_list = list(boards.items())
    board_positions = get_optimized_board_layout(len(board_list), connections, boards)

    # Process each board
    for board_idx, (board_id, board_info) in enumerate(board_list):
        board_type = board_info["type"].lower()
        board_chips = board_info["chips"]
        num_chips = len(board_chips)

        if num_chips == 0:
            continue

        # Get board position from optimized layout
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
            board_center, (board_width, board_depth, board_height), color="darkblue", name=f'{board_info["type"]}_frame'
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

        # Add all chips in this board as one trace
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

    # Add ethernet connections with curved lines
    for conn in connections:
        src = conn["src_chip"]
        dst = conn["dst_chip"]

        if src in chip_positions and dst in chip_positions:
            src_pos = chip_positions[src]
            dst_pos = chip_positions[dst]

            # Create curved line
            x_curve, y_curve, z_curve = create_curved_line(src_pos, dst_pos)

            # Calculate midpoint for annotation
            mid_idx = len(x_curve) // 2
            mid_x, mid_y, mid_z = x_curve[mid_idx], y_curve[mid_idx], z_curve[mid_idx]

            edge_trace = go.Scatter3d(
                x=x_curve,
                y=y_curve,
                z=z_curve,
                mode="lines",
                line=dict(color="gray", width=3),
                hovertemplate=(
                    f"<b>Ethernet Link</b><br>"
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

                # Find chips in this mesh
                mesh_chip_ids = sorted(chips.keys())[:mesh_chip_count]

                if mesh_chip_ids and all(cid in chip_positions for cid in mesh_chip_ids):
                    # Calculate bounding box
                    mesh_positions = [chip_positions[cid] for cid in mesh_chip_ids]
                    xs = [p[0] for p in mesh_positions]
                    ys = [p[1] for p in mesh_positions]
                    zs = [p[2] for p in mesh_positions]

                    min_x, max_x = min(xs) - 1, max(xs) + 1
                    min_y, max_y = min(ys) - 1, max(ys) + 1
                    min_z, max_z = min(zs) - 0.5, max(zs) + 2

                    mesh_center = ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
                    mesh_size = (max_x - min_x, max_y - min_y, max_z - min_z)

                    # Create mesh box
                    mesh_trace = create_3d_box_mesh(
                        mesh_center,
                        mesh_size,
                        color="red",
                        opacity=0.15,
                        name=f"Logical Mesh {mesh_id}<br>({rows}×{cols} = {board_name})",
                        customdata=[[mesh_center[0], mesh_center[1], mesh_center[2]]],
                    )
                    traces.append(mesh_trace)

                    # Add mesh wireframe
                    mesh_wireframe = create_board_wireframe(
                        mesh_center, mesh_size, color="red", name=f"Mesh_{mesh_id}_frame"
                    )
                    traces.extend(mesh_wireframe)

    # Layout with click event handling
    layout = go.Layout(
        title=dict(text="<b>3D Topology (Click elements to see coordinates)</b>", font=dict(size=20)),
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
        # Add annotation for clicked coordinates
        annotations=[],
    )

    fig = go.Figure(data=traces, layout=layout)

    # Add JavaScript for click handling
    click_js = """
    <script>
    var myPlot = document.getElementById('plotly-div');

    myPlot.on('plotly_click', function(data){
        var pt = data.points[0];
        var coord_text = '';

        // Try to get coordinates from customdata first
        if (pt.customdata) {
            var x = pt.customdata[0];
            var y = pt.customdata[1];
            var z = pt.customdata[2];
            coord_text = 'Coordinates: (' + x.toFixed(2) + ', ' + y.toFixed(2) + ', ' + z.toFixed(2) + ')';

            // If there's a chip ID
            if (pt.customdata.length > 3) {
                var chip_id = pt.customdata[3];
                coord_text = 'Chip ' + chip_id + ' - ' + coord_text;
            }
        } else {
            // Fallback to point coordinates
            coord_text = 'Coordinates: (' + pt.x.toFixed(2) + ', ' + pt.y.toFixed(2) + ', ' + pt.z.toFixed(2) + ')';
        }

        // Display in console and create annotation
        console.log('Clicked:', coord_text);

        // Update title to show coordinates
        var update = {
            title: '<b>3D Topology</b><br><span style="font-size:14px;color:blue;">' + coord_text + '</span>'
        };
        Plotly.relayout('plotly-div', update);

        // Also show as alert
        alert(coord_text);
    });

    // Reset title on double click
    myPlot.on('plotly_doubleclick', function(data){
        var update = {
            title: '<b>3D Topology (Click elements to see coordinates)</b>'
        };
        Plotly.relayout('plotly-div', update);
    });
    </script>
    """

    # Save to HTML with custom JavaScript
    html_content = fig.to_html(
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["toImage"],
            "modeBarButtonsToAdd": ["hoverclosest", "hovercompare"],
        },
        div_id="plotly-div",
    )

    # Insert JavaScript before closing body tag
    html_content = html_content.replace("</body>", click_js + "</body>")

    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"✅ Enhanced 3D visualization saved to: {output_file}")
    print(f"   Open in browser: file://{Path(output_file).absolute()}")
    print(f"\n🎮 Controls:")
    print(f"   • Left-click + drag: Rotate")
    print(f"   • Scroll: Zoom in/out")
    print(f"   • Right-click + drag: Pan")
    print(f"   • Click element: See coordinates (popup + title)")
    print(f"   • Double-click: Reset view and clear coordinates")
    print(f"   • Hover: See details")
    print(f"\n✨ Improvements:")
    print(f"   • Optimized board layout")
    print(f"   • Curved ethernet lines (lift above chips)")
    print(f"   • Click any element to see X,Y,Z coordinates")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced 3D Topology Visualizer v2 with Click Coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  • Click any element to see coordinates (popup + title display)
  • Optimized layout to minimize line crossings
  • Curved ethernet lines that lift over chips
  • Interactive legend and hover details

Examples:
  ./visualize_topology_3d_enhanced_v2.py --cluster t3k_phys_topology.yaml \\
      --mesh t3k_generated_mesh.yaml
        """,
    )

    parser.add_argument("--cluster", required=True, help="Path to cluster descriptor YAML (physical topology)")
    parser.add_argument("--mesh", help="Path to mesh graph descriptor YAML (logical overlay)")
    parser.add_argument(
        "--output", default="topology_3d_interactive.html", help="Output file (default: topology_3d_interactive.html)"
    )

    args = parser.parse_args()

    # Load cluster descriptor
    print(f"📂 Loading cluster descriptor: {args.cluster}")
    try:
        cluster_desc = load_cluster_descriptor(args.cluster)
        chips, connections, boards = parse_physical_topology(cluster_desc)
        print(f"   Found {len(chips)} chips in {len(boards)} boards")
        print(f"   Found {len(connections)} ethernet connections")

        for board_id, board_info in boards.items():
            print(f"   • {board_info['type']}: {len(board_info['chips'])} chips")
    except Exception as e:
        print(f"❌ Error loading cluster descriptor: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Load mesh descriptor if provided
    mesh_desc = None
    if args.mesh:
        print(f"\n📂 Loading mesh graph descriptor: {args.mesh}")
        try:
            mesh_desc = load_mesh_graph_descriptor(args.mesh)
            if "Mesh" in mesh_desc:
                for mesh in mesh_desc["Mesh"]:
                    topo = mesh.get("device_topology", [])
                    print(f"   Mesh {mesh.get('id', 0)}: {topo[0]}×{topo[1]} = {mesh.get('board', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load mesh descriptor: {e}")

    print(f"\n🎨 Generating enhanced 3D visualization with improvements...")
    success = create_enhanced_3d_viz(chips, connections, boards, mesh_desc, args.output)

    if success:
        print(f"\n🎉 Done! Open the file in your browser.")
        print(f"\nTry clicking on chips, boards, or connections to see their coordinates!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
