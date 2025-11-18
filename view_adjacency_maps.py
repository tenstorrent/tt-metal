#!/usr/bin/env python3
"""
Visualization script for logical and physical adjacency maps.
Shows the topology with mesh_host_rank_id and MPI host rank information.
"""

import json
import sys
import argparse
from typing import Dict, List, Any
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import numpy as np


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def compute_degree_based_grid_layout(G: nx.Graph) -> Dict[str, tuple]:
    """
    Compute grid layout where nodes are placed based on their degree:
    - Degree 2 nodes → corners
    - Degree 3 nodes → edges
    - Degree 4 nodes → centers

    Uses Manhattan distance to optimize placement.
    """
    node_degrees = dict(G.degree())

    # Group nodes by degree
    degree_2_nodes = [n for n, d in node_degrees.items() if d == 2]
    degree_3_nodes = [n for n, d in node_degrees.items() if d == 3]
    degree_4_nodes = [n for n, d in node_degrees.items() if d == 4]
    other_nodes = [n for n, d in node_degrees.items() if d not in [2, 3, 4]]

    # Estimate grid dimensions
    # For a rectangular grid: corners = 4, edges = 2*(rows+cols-2), centers = (rows-2)*(cols-2)
    total_nodes = len(G.nodes())

    # Try to find grid dimensions that match the degree distribution
    # Start with a reasonable estimate
    if len(degree_2_nodes) >= 4:
        # We have at least 4 corners, so it's likely a rectangular grid
        # Estimate: sqrt(total_nodes) gives approximate square dimensions
        est_dim = int(np.ceil(np.sqrt(total_nodes)))
        rows = est_dim
        cols = est_dim

        # Refine based on degree counts
        # For a grid: corners=4, edges=2*(rows+cols-2), centers=(rows-2)*(cols-2)
        # Try to match the observed counts
        best_rows, best_cols = rows, cols
        best_score = float("inf")

        for r in range(2, int(np.sqrt(total_nodes)) + 5):
            for c in range(2, int(np.sqrt(total_nodes)) + 5):
                expected_corners = 4
                expected_edges = 2 * (r + c - 2)
                expected_centers = (r - 2) * (c - 2) if r > 2 and c > 2 else 0
                total_grid = r * c

                if total_grid < total_nodes:
                    continue

                # Score based on how well it matches
                score = (
                    abs(len(degree_2_nodes) - expected_corners)
                    + abs(len(degree_3_nodes) - expected_edges)
                    + abs(len(degree_4_nodes) - expected_centers)
                    + abs(total_grid - total_nodes)
                )

                if score < best_score:
                    best_score = score
                    best_rows, best_cols = r, c

        rows, cols = best_rows, best_cols
    else:
        # Fallback: estimate from total nodes
        rows = int(np.ceil(np.sqrt(total_nodes)))
        cols = int(np.ceil(total_nodes / rows))

    # Classify grid positions
    corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    edge_positions = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in corner_positions:
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    edge_positions.append((r, c))

    center_positions = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center_positions.append((r, c))

    # All positions
    all_positions = corner_positions + edge_positions + center_positions

    # Initialize position mapping
    pos = {}
    used_positions = set()

    # Helper function to compute Manhattan distance between two positions
    def manhattan_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # Helper function to find best position for a node given its neighbors' positions
    def find_best_position(node, candidate_positions, neighbor_positions):
        if not neighbor_positions:
            # No neighbors placed yet, return first available position
            return candidate_positions[0] if candidate_positions else None

        best_pos = None
        best_score = float("inf")

        for pos_candidate in candidate_positions:
            if pos_candidate in used_positions:
                continue

            # Score: minimize sum of Manhattan distances to neighbors
            total_dist = sum(manhattan_dist(pos_candidate, n_pos) for n_pos in neighbor_positions)

            if total_dist < best_score:
                best_score = total_dist
                best_pos = pos_candidate

        return best_pos

    # Assign degree-2 nodes to corners first
    unassigned_corners = degree_2_nodes.copy()
    available_corners = corner_positions.copy()

    # Start with one corner node at (0, 0)
    if unassigned_corners:
        start_node = unassigned_corners.pop(0)
        pos[start_node] = (0, 0)
        used_positions.add((0, 0))
        available_corners.remove((0, 0))

    # Assign remaining corners
    while unassigned_corners and available_corners:
        best_node = None
        best_pos = None
        best_score = float("inf")

        for node in unassigned_corners:
            neighbor_positions = [pos[n] for n in G.neighbors(node) if n in pos]
            if neighbor_positions:
                for corner_pos in available_corners:
                    total_dist = sum(manhattan_dist(corner_pos, n_pos) for n_pos in neighbor_positions)
                    if total_dist < best_score:
                        best_score = total_dist
                        best_node = node
                        best_pos = corner_pos
            else:
                # No neighbors placed, assign to first available corner
                best_node = node
                best_pos = available_corners[0]
                break

        if best_node and best_pos:
            pos[best_node] = best_pos
            used_positions.add(best_pos)
            unassigned_corners.remove(best_node)
            available_corners.remove(best_pos)
        else:
            break

    # Assign degree-3 nodes to edges
    unassigned_edges = degree_3_nodes.copy()
    available_edges = [p for p in edge_positions if p not in used_positions]

    # Use BFS-like approach to place edge nodes near their neighbors
    queue = list(pos.keys())

    while queue and unassigned_edges:
        current = queue.pop(0)
        neighbors = [n for n in G.neighbors(current) if n not in pos]

        for neighbor in neighbors:
            if neighbor not in unassigned_edges:
                continue

            neighbor_positions = [pos[n] for n in G.neighbors(neighbor) if n in pos]
            best_pos = find_best_position(neighbor, available_edges, neighbor_positions)

            if best_pos:
                pos[neighbor] = best_pos
                used_positions.add(best_pos)
                if best_pos in available_edges:
                    available_edges.remove(best_pos)
                unassigned_edges.remove(neighbor)
                queue.append(neighbor)

    # Assign remaining edge nodes
    while unassigned_edges and available_edges:
        node = unassigned_edges.pop(0)
        neighbor_positions = [pos[n] for n in G.neighbors(node) if n in pos]
        best_pos = find_best_position(node, available_edges, neighbor_positions)
        if best_pos:
            pos[node] = best_pos
            used_positions.add(best_pos)
            available_edges.remove(best_pos)
        elif available_edges:
            # Fallback: assign to first available
            pos[node] = available_edges.pop(0)
            used_positions.add(pos[node])

    # Assign degree-4 nodes to centers
    unassigned_centers = degree_4_nodes.copy()
    available_centers = [p for p in center_positions if p not in used_positions]

    # Use BFS-like approach for centers
    queue = list(pos.keys())

    while queue and unassigned_centers:
        current = queue.pop(0)
        neighbors = [n for n in G.neighbors(current) if n not in pos]

        for neighbor in neighbors:
            if neighbor not in unassigned_centers:
                continue

            neighbor_positions = [pos[n] for n in G.neighbors(neighbor) if n in pos]
            best_pos = find_best_position(neighbor, available_centers, neighbor_positions)

            if best_pos:
                pos[neighbor] = best_pos
                used_positions.add(best_pos)
                if best_pos in available_centers:
                    available_centers.remove(best_pos)
                unassigned_centers.remove(neighbor)
                queue.append(neighbor)

    # Assign remaining center nodes
    while unassigned_centers and available_centers:
        node = unassigned_centers.pop(0)
        neighbor_positions = [pos[n] for n in G.neighbors(node) if n in pos]
        best_pos = find_best_position(node, available_centers, neighbor_positions)
        if best_pos:
            pos[node] = best_pos
            used_positions.add(best_pos)
            available_centers.remove(best_pos)
        elif available_centers:
            pos[node] = available_centers.pop(0)
            used_positions.add(pos[node])

    # Assign other nodes (degree != 2, 3, 4) to remaining positions
    unassigned_other = other_nodes.copy()
    available_other = [p for p in all_positions if p not in used_positions]

    while unassigned_other and available_other:
        node = unassigned_other.pop(0)
        neighbor_positions = [pos[n] for n in G.neighbors(node) if n in pos]
        best_pos = find_best_position(node, available_other, neighbor_positions)
        if best_pos:
            pos[node] = best_pos
            used_positions.add(best_pos)
            available_other.remove(best_pos)
        elif available_other:
            pos[node] = available_other.pop(0)
            used_positions.add(pos[node])

    # Convert to (x, y) format where x is column and y is -row (for top-to-bottom display)
    final_pos = {node: (col, -row) for node, (row, col) in pos.items()}

    # Fill any remaining unassigned nodes with nearby positions
    for node in G.nodes():
        if node not in final_pos:
            # Find nearest used position and place nearby
            if pos:
                # Find a nearby empty spot
                max_row = max(r for r, c in pos.values()) if pos else 0
                max_col = max(c for r, c in pos.values()) if pos else 0
                for r in range(max_row + 2):
                    for c in range(max_col + 2):
                        if (r, c) not in used_positions:
                            final_pos[node] = (c, -r)
                            used_positions.add((r, c))
                            break
                    else:
                        continue
                    break
            else:
                final_pos[node] = (0, 0)

    return final_pos


def visualize_logical_adjacency_map(data: Dict[str, Any], output_file: str = None):
    """Visualize logical adjacency map."""
    logical_map = data.get("logical_adjacency_map", {})

    fig, axes = plt.subplots(1, len(logical_map), figsize=(30, 20))
    if len(logical_map) == 1:
        axes = [axes]

    for idx, (mesh_id_str, mesh_data) in enumerate(logical_map.items()):
        ax = axes[idx] if len(logical_map) > 1 else axes[0]
        G = nx.Graph()

        nodes = mesh_data.get("nodes", [])
        node_labels = {}
        node_colors = []

        # Build mapping from fabric_node_id to mesh_rank from nodes themselves
        mesh_rank_mapping = {}

        for node in nodes:
            fabric_node = node.get("fabric_node_id", {})
            chip_id = fabric_node.get("chip_id", -1)
            mesh_id_val = fabric_node.get("mesh_id", -1)
            mesh_host_rank = node.get("mesh_host_rank_id", -1)
            mpi_rank = node.get("mpi_host_rank_id", -1)

            # Build mapping from nodes
            fabric_key = f"{mesh_id_val}_{chip_id}"
            mesh_rank_mapping[fabric_key] = mesh_host_rank

            node_id = f"M{mesh_id_str}_C{chip_id}"
            G.add_node(node_id)

            # Create label showing fabric_node_id and associated mesh_host_rank_id
            label = f"FabricNodeID:\nM{mesh_id_val}_C{chip_id}\nMeshRank:{mesh_host_rank}"
            if mpi_rank >= 0:
                label += f"\nMPI:{mpi_rank}"
            node_labels[node_id] = label

            # Color by MPI rank
            if mpi_rank >= 0:
                node_colors.append(mpi_rank)
            else:
                node_colors.append(-1)

            # Add edges
            for neighbor in node.get("neighbors", []):
                neighbor_chip_id = neighbor.get("chip_id", -1)
                neighbor_id = f"M{mesh_id_str}_C{neighbor_chip_id}"
                if neighbor_id in G.nodes():
                    G.add_edge(node_id, neighbor_id)

        # Use degree-based grid layout
        pos = compute_degree_based_grid_layout(G)

        # Draw nodes with colors based on MPI rank
        unique_ranks = sorted(set(node_colors))
        color_map = plt.cm.tab20
        node_color_list = [
            color_map(unique_ranks.index(r) / max(len(unique_ranks), 1)) if r >= 0 else "gray" for r in node_colors
        ]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, node_size=800, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1)
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=6)

        ax.set_title(f"Logical Adjacency Map - Mesh {mesh_id_str}", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")

        # Add fabric_node_id to mesh_rank mapping information
        if mesh_rank_mapping:
            mapping_text = "Fabric Node ID → Mesh Rank:\n"
            for fabric_key, rank_val in sorted(mesh_rank_mapping.items()):
                mesh_id_val, chip_id_val = fabric_key.split("_")
                mapping_text += f"  M{mesh_id_val}_C{chip_id_val} → Rank {rank_val}\n"
            ax.text(
                0.02,
                0.02,
                mapping_text.rstrip(),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            )

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved logical adjacency map visualization to {output_file}")
    else:
        plt.show()


def visualize_physical_adjacency_map(data: Dict[str, Any], output_file: str = None):
    """Visualize physical adjacency map."""
    physical_map = data.get("physical_adjacency_map", {})

    fig, axes = plt.subplots(1, len(physical_map), figsize=(30, 20))
    if len(physical_map) == 1:
        axes = [axes]

    for idx, (mesh_id_str, mesh_data) in enumerate(physical_map.items()):
        ax = axes[idx] if len(physical_map) > 1 else axes[0]
        G = nx.Graph()

        nodes = mesh_data.get("nodes", [])
        node_labels = {}
        node_colors = []
        host_groups = {}

        # Build mapping from asic_id to mesh_rank from nodes themselves
        asic_rank_mapping = {}

        for node in nodes:
            asic_id = node.get("asic_id", -1)
            host_name = node.get("host_name", "unknown")
            mesh_host_rank = node.get("mesh_host_rank_id", -1)
            mpi_rank = node.get("mpi_host_rank_id", -1)

            # Build mapping from nodes
            asic_rank_mapping[asic_id] = mesh_host_rank

            node_id = f"ASIC{asic_id}"
            G.add_node(node_id)

            # Group by host
            if host_name not in host_groups:
                host_groups[host_name] = []
            host_groups[host_name].append(node_id)

            # Create label showing asic_id and associated mesh_host_rank_id
            label = f"ASICID:\n{asic_id}\nMeshRank:{mesh_host_rank}"
            label += f"\nHost:{host_name}"
            if mpi_rank != 4294967295:
                label += f"\nMPI:{mpi_rank}"
            node_labels[node_id] = label

            # Color by MPI rank
            if mpi_rank != 4294967295:  # UINT32_MAX
                node_colors.append(mpi_rank)
            else:
                node_colors.append(-1)

            # Add edges
            for neighbor in node.get("neighbors", []):
                neighbor_asic_id = neighbor.get("asic_id", -1)
                neighbor_id = f"ASIC{neighbor_asic_id}"
                if neighbor_id in G.nodes():
                    G.add_edge(node_id, neighbor_id)

        # Use degree-based grid layout
        pos = compute_degree_based_grid_layout(G)

        # Draw nodes
        unique_ranks = sorted(set(node_colors))
        color_map = plt.cm.tab20
        node_color_list = [
            color_map(unique_ranks.index(r) / max(len(unique_ranks), 1)) if r >= 0 else "gray" for r in node_colors
        ]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color_list, node_size=600, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.8)
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=5)

        ax.set_title(f"Physical Adjacency Map - Mesh {mesh_id_str}", fontsize=14, fontweight="bold")
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")

        # Add host information
        host_info = "\n".join([f"{host}: {len(nodes)} ASICs" for host, nodes in host_groups.items()])
        ax.text(
            0.02,
            0.98,
            f"Hosts:\n{host_info}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Add asic_id to mesh_rank mapping information
        if asic_rank_mapping:
            mapping_text = "ASIC ID → Mesh Rank:\n"
            for asic_id_val, rank_val in sorted(asic_rank_mapping.items()):
                mapping_text += f"  ASIC {asic_id_val} → Rank {rank_val}\n"
            ax.text(
                0.02,
                0.02,
                mapping_text.rstrip(),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            )

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved physical adjacency map visualization to {output_file}")
    else:
        plt.show()


def print_summary(data: Dict[str, Any], map_type: str):
    """Print summary statistics."""
    map_key = f"{map_type}_adjacency_map"
    adj_map = data.get(map_key, {})

    print(f"\n{'='*60}")
    print(f"{map_type.upper()} ADJACENCY MAP SUMMARY")
    print(f"{'='*60}")

    for mesh_id_str, mesh_data in adj_map.items():
        nodes = mesh_data.get("nodes", [])
        print(f"\nMesh {mesh_id_str}:")
        print(f"  Total nodes: {len(nodes)}")

        # Count by MPI rank
        mpi_ranks = {}
        mesh_host_ranks = {}
        for node in nodes:
            mpi_rank = node.get("mpi_host_rank_id", -1)
            mesh_host_rank = node.get("mesh_host_rank_id", -1)

            if mpi_rank not in mpi_ranks:
                mpi_ranks[mpi_rank] = 0
            mpi_ranks[mpi_rank] += 1

            if mesh_host_rank not in mesh_host_ranks:
                mesh_host_ranks[mesh_host_rank] = 0
            mesh_host_ranks[mesh_host_rank] += 1

        print(f"  Nodes by MPI rank: {dict(sorted(mpi_ranks.items()))}")
        print(f"  Nodes by mesh_host_rank: {dict(sorted(mesh_host_ranks.items()))}")

        # Count edges
        total_edges = sum(len(node.get("neighbors", [])) for node in nodes) // 2
        print(f"  Total edges: {total_edges}")


def main():
    parser = argparse.ArgumentParser(description="Visualize adjacency maps")
    parser.add_argument("--logical", type=str, required=True, help="Path to logical adjacency map JSON file (required)")
    parser.add_argument(
        "--physical", type=str, required=True, help="Path to physical adjacency map JSON file (required)"
    )
    parser.add_argument(
        "--output-logical",
        type=str,
        default="logical_adjacency_map.png",
        help="Output file for logical map visualization (PNG). Default: logical_adjacency_map.png",
    )
    parser.add_argument(
        "--output-physical",
        type=str,
        default="physical_adjacency_map.png",
        help="Output file for physical map visualization (PNG). Default: physical_adjacency_map.png",
    )
    parser.add_argument("--no-plot", action="store_true", help="Only print summary, do not generate plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively instead of saving to files")

    args = parser.parse_args()

    # Determine output mode
    output_logical = None if args.show else args.output_logical
    output_physical = None if args.show else args.output_physical

    # Load and process logical map
    try:
        logical_data = load_json(args.logical)
        print_summary(logical_data, "logical")
        if not args.no_plot:
            visualize_logical_adjacency_map(logical_data, output_logical)
    except FileNotFoundError:
        print(f"Error: Could not find {args.logical}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing logical map: {e}")
        sys.exit(1)

    # Load and process physical map
    try:
        physical_data = load_json(args.physical)
        print_summary(physical_data, "physical")
        if not args.no_plot:
            visualize_physical_adjacency_map(physical_data, output_physical)
    except FileNotFoundError:
        print(f"Error: Could not find {args.physical}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing physical map: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
