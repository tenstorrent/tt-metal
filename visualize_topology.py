#!/usr/bin/env python3
"""
3D Interactive Topology Visualizer
Visualizes physical chip topology and mesh graph descriptor overlay
"""

import yaml
import argparse
import sys
from pathlib import Path

try:
    import plotly.graph_objects as go
    import networkx as nx

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Install with: pip install plotly networkx")

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_cluster_descriptor(yaml_path):
    """Load physical cluster descriptor (old format)"""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def load_mesh_graph_descriptor(yaml_path):
    """Load mesh graph descriptor (new format)"""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def parse_physical_topology(cluster_desc):
    """Parse physical topology from cluster descriptor"""
    chips = {}
    connections = []

    # Parse chips
    if "arch" in cluster_desc:
        for chip_id, arch in cluster_desc["arch"].items():
            chips[int(chip_id)] = {
                "id": int(chip_id),
                "arch": arch,
                "location": cluster_desc.get("chips", {}).get(chip_id, [0, 0, 0, 0]),
            }

    # Parse ethernet connections
    if "ethernet_connections" in cluster_desc:
        for conn in cluster_desc["ethernet_connections"]:
            if len(conn) == 2:
                src = conn[0]
                dst = conn[1]
                connections.append(
                    {"src_chip": src["chip"], "src_chan": src["chan"], "dst_chip": dst["chip"], "dst_chan": dst["chan"]}
                )

    return chips, connections


def create_3d_interactive_viz(chips, connections, mesh_desc=None, output_file="topology_3d.html"):
    """Create 3D interactive visualization using Plotly"""
    if not HAS_PLOTLY:
        print("ERROR: Plotly not installed. Cannot create 3D visualization.")
        return False

    # Infer 3D layout from chip locations
    chip_positions = {}
    for chip_id, chip_info in chips.items():
        loc = chip_info["location"]
        # [shelf, rack_x, rack_y, adapter]
        x = loc[1] if len(loc) > 1 else chip_id % 4
        y = loc[2] if len(loc) > 2 else (chip_id // 4) % 2
        z = loc[0] if len(loc) > 0 else 0  # shelf
        chip_positions[chip_id] = (x, y, z)

    # Create chip nodes
    chip_ids = list(chips.keys())
    x_pos = [chip_positions[cid][0] for cid in chip_ids]
    y_pos = [chip_positions[cid][1] for cid in chip_ids]
    z_pos = [chip_positions[cid][2] for cid in chip_ids]

    chip_text = [f"Chip {cid}<br>Arch: {chips[cid]['arch']}<br>Location: {chips[cid]['location']}" for cid in chip_ids]

    # Create scatter plot for chips
    chip_trace = go.Scatter3d(
        x=x_pos,
        y=y_pos,
        z=z_pos,
        mode="markers+text",
        marker=dict(size=20, color="lightblue", line=dict(color="darkblue", width=2)),
        text=[str(cid) for cid in chip_ids],
        textposition="middle center",
        hovertext=chip_text,
        hoverinfo="text",
        name="Chips",
    )

    # Create connection lines
    edge_traces = []
    for conn in connections:
        src = conn["src_chip"]
        dst = conn["dst_chip"]

        if src in chip_positions and dst in chip_positions:
            src_pos = chip_positions[src]
            dst_pos = chip_positions[dst]

            edge_trace = go.Scatter3d(
                x=[src_pos[0], dst_pos[0]],
                y=[src_pos[1], dst_pos[1]],
                z=[src_pos[2], dst_pos[2]],
                mode="lines",
                line=dict(color="gray", width=3),
                hovertext=f"Link: {src}[ch{conn['src_chan']}] ↔ {dst}[ch{conn['dst_chan']}]",
                hoverinfo="text",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

    # Create mesh overlay if provided
    mesh_traces = []
    if mesh_desc and "Mesh" in mesh_desc:
        for mesh in mesh_desc["Mesh"]:
            mesh_id = mesh.get("id", 0)
            dev_topo = mesh.get("device_topology", [])
            if len(dev_topo) >= 2:
                rows, cols = dev_topo[0], dev_topo[1]

                # Draw mesh grid overlay
                for r in range(rows):
                    for c in range(cols):
                        chip_idx = r * cols + c
                        if chip_idx < len(chip_ids):
                            # Highlight mesh chips
                            pass

    # Combine all traces
    data = [chip_trace] + edge_traces + mesh_traces

    # Layout
    layout = go.Layout(
        title="3D Physical Topology Visualization",
        scene=dict(
            xaxis=dict(title="Rack X"),
            yaxis=dict(title="Rack Y"),
            zaxis=dict(title="Shelf"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        hovermode="closest",
        showlegend=True,
    )

    fig = go.Figure(data=data, layout=layout)

    # Save to HTML
    fig.write_html(output_file)
    print(f"✅ 3D visualization saved to: {output_file}")
    print(f"   Open in browser: file://{Path(output_file).absolute()}")

    return True


def create_2d_topology_viz(chips, connections, mesh_desc=None, output_file="topology_2d.png"):
    """Create 2D topology visualization using Matplotlib"""
    if not HAS_MATPLOTLIB:
        print("ERROR: Matplotlib not installed. Cannot create 2D visualization.")
        return False

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create layout (simple grid based on chip count)
    num_chips = len(chips)
    cols = int(num_chips**0.5) + 1
    rows = (num_chips + cols - 1) // cols

    chip_positions = {}
    chip_ids = sorted(chips.keys())

    for idx, chip_id in enumerate(chip_ids):
        row = idx // cols
        col = idx % cols
        chip_positions[chip_id] = (col * 2, -row * 2)

    # Draw connections first (behind chips)
    for conn in connections:
        src = conn["src_chip"]
        dst = conn["dst_chip"]

        if src in chip_positions and dst in chip_positions:
            src_pos = chip_positions[src]
            dst_pos = chip_positions[dst]

            # Draw arrow
            arrow = FancyArrowPatch(
                src_pos, dst_pos, arrowstyle="<->", mutation_scale=15, linewidth=1.5, color="gray", alpha=0.6
            )
            ax.add_patch(arrow)

            # Add channel labels
            mid_x = (src_pos[0] + dst_pos[0]) / 2
            mid_y = (src_pos[1] + dst_pos[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                f"ch{conn['src_chan']}↔{conn['dst_chan']}",
                fontsize=6,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
            )

    # Draw chips
    for chip_id, pos in chip_positions.items():
        x, y = pos

        # Draw chip box
        rect = FancyBboxPatch(
            (x - 0.4, y - 0.4),
            0.8,
            0.8,
            boxstyle="round,pad=0.05",
            facecolor="lightblue",
            edgecolor="darkblue",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Add chip label
        ax.text(x, y, str(chip_id), ha="center", va="center", fontsize=12, fontweight="bold")

        # Add arch label
        arch = chips[chip_id]["arch"]
        ax.text(x, y - 0.6, arch, ha="center", va="top", fontsize=7, style="italic", color="darkblue")

    # Overlay mesh descriptor if provided
    if mesh_desc and "Mesh" in mesh_desc:
        for mesh in mesh_desc["Mesh"]:
            mesh_id = mesh.get("id", 0)
            dev_topo = mesh.get("device_topology", [])
            board = mesh.get("board", "unknown")

            if len(dev_topo) >= 2:
                rows, cols = dev_topo[0], dev_topo[1]

                # Draw mesh grid overlay
                mesh_chip_ids = chip_ids[: rows * cols]
                if mesh_chip_ids:
                    # Find bounding box
                    mesh_positions = [chip_positions[cid] for cid in mesh_chip_ids]
                    xs = [p[0] for p in mesh_positions]
                    ys = [p[1] for p in mesh_positions]

                    min_x, max_x = min(xs) - 0.7, max(xs) + 0.7
                    min_y, max_y = min(ys) - 0.7, max(ys) + 0.7

                    # Draw mesh boundary
                    mesh_rect = plt.Rectangle(
                        (min_x, min_y),
                        max_x - min_x,
                        max_y - min_y,
                        fill=False,
                        edgecolor="red",
                        linewidth=3,
                        linestyle="--",
                        label=f"Mesh {mesh_id} ({board}): {rows}×{cols}",
                    )
                    ax.add_patch(mesh_rect)

    # Set axis properties
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.1)
    ax.set_title("Physical Topology with Mesh Overlay", fontsize=16, fontweight="bold")
    ax.axis("off")

    if mesh_desc:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ 2D visualization saved to: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TT-Metal physical topology and mesh graph descriptors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize physical topology only
  ./visualize_topology.py --cluster t3k_phys_topology.yaml

  # Visualize with mesh overlay
  ./visualize_topology.py --cluster t3k_phys_topology.yaml \\
      --mesh tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml

  # Generate both 2D and 3D
  ./visualize_topology.py --cluster t3k_phys_topology.yaml --both
        """,
    )

    parser.add_argument("--cluster", required=True, help="Path to cluster descriptor YAML (physical topology)")
    parser.add_argument("--mesh", help="Path to mesh graph descriptor YAML (logical overlay)")
    parser.add_argument(
        "--output-3d", default="topology_3d.html", help="Output file for 3D visualization (default: topology_3d.html)"
    )
    parser.add_argument(
        "--output-2d", default="topology_2d.png", help="Output file for 2D visualization (default: topology_2d.png)"
    )
    parser.add_argument("--both", action="store_true", help="Generate both 2D and 3D visualizations")
    parser.add_argument(
        "--3d", dest="three_d", action="store_true", help="Generate 3D visualization (default if no --both)"
    )

    args = parser.parse_args()

    # Load cluster descriptor
    print(f"📂 Loading cluster descriptor: {args.cluster}")
    try:
        cluster_desc = load_cluster_descriptor(args.cluster)
        chips, connections = parse_physical_topology(cluster_desc)
        print(f"   Found {len(chips)} chips and {len(connections)} connections")
    except Exception as e:
        print(f"❌ Error loading cluster descriptor: {e}")
        sys.exit(1)

    # Load mesh descriptor if provided
    mesh_desc = None
    if args.mesh:
        print(f"📂 Loading mesh graph descriptor: {args.mesh}")
        try:
            mesh_desc = load_mesh_graph_descriptor(args.mesh)
            print(f"   Loaded mesh descriptor")
        except Exception as e:
            print(f"⚠️  Warning: Could not load mesh descriptor: {e}")

    print("\n🎨 Generating visualizations...")

    success = False

    # Generate visualizations
    if args.both:
        success_3d = create_3d_interactive_viz(chips, connections, mesh_desc, args.output_3d)
        success_2d = create_2d_topology_viz(chips, connections, mesh_desc, args.output_2d)
        success = success_3d or success_2d
    elif args.three_d:
        success = create_3d_interactive_viz(chips, connections, mesh_desc, args.output_3d)
    else:
        # Default: try 3D first, fallback to 2D
        success = create_3d_interactive_viz(chips, connections, mesh_desc, args.output_3d)
        if not success:
            print("\n📊 Falling back to 2D visualization...")
            success = create_2d_topology_viz(chips, connections, mesh_desc, args.output_2d)

    if success:
        print("\n🎉 Visualization complete!")
    else:
        print("\n❌ Failed to generate visualization. Please install dependencies:")
        print("   pip install plotly networkx matplotlib")
        sys.exit(1)


if __name__ == "__main__":
    main()
