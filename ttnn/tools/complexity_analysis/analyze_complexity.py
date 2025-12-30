#!/usr/bin/env python3
"""
Operation Complexity Analysis Tool

Analyzes TT-NN operations by extracting complexity metrics from program factory files.
Generates ranked complexity scores and buckets operations for refactoring prioritization.
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from statistics import mean, median


@dataclass
class ProgramFactoryMetrics:
    """Metrics extracted from a single program factory file."""

    file_path: str
    operation_name: str
    lines_of_code: int
    create_kernel_count: int
    create_circular_buffer_count: int
    set_runtime_args_count: int
    create_semaphore_count: int
    conditional_branches: int  # if/else/switch statements
    # CCL-specific metrics
    create_mesh_workload_count: int
    global_semaphore_count: int
    fabric_api_calls: int
    command_stream_builders: int


@dataclass
class OperationComplexity:
    """Aggregated complexity metrics for an operation."""

    operation_name: str
    operation_category: str
    program_factory_count: int
    total_lines_of_code: int
    total_kernels: int
    total_circular_buffers: int
    total_runtime_args: int
    total_semaphores: int
    total_conditionals: int
    kernel_file_count: int
    # CCL-specific totals
    total_mesh_workloads: int
    total_global_semaphores: int
    total_fabric_api_calls: int
    total_command_builders: int
    is_ccl_operation: bool
    normalized_loc: float
    normalized_kernels: float
    normalized_cbs: float
    normalized_runtime_args: float
    normalized_semaphores: float
    normalized_mesh_workloads: float
    normalized_global_semaphores: float
    complexity_score: float
    complexity_bucket: str


class ComplexityAnalyzer:
    """Analyzes operation complexity from program factory files."""

    def __init__(self, operations_root: Path):
        self.operations_root = operations_root
        self.program_factories: List[ProgramFactoryMetrics] = []
        self.operation_complexities: List[OperationComplexity] = []

    def find_program_factories(self) -> List[Path]:
        """Find all program factory .cpp files."""
        factories = []
        for path in self.operations_root.rglob("*program_factory*.cpp"):
            factories.append(path)
        return sorted(factories)

    def extract_operation_name(self, file_path: Path) -> Tuple[str, str]:
        """Extract operation name and category from file path.

        Returns: (operation_name, category)
        """
        # Path structure: operations/{category}/.../..._program_factory.cpp
        parts = file_path.parts
        try:
            ops_idx = parts.index("operations")
            if ops_idx + 1 < len(parts):
                category = parts[ops_idx + 1]
            else:
                category = "unknown"

            # Extract operation name from filename
            filename = file_path.stem
            # Remove common suffixes
            name = filename.replace("_program_factory", "").replace("program_factory", "")

            # Try to get more specific operation name from path
            if "device" in parts:
                device_idx = parts.index("device")
                if device_idx > 0:
                    # Use parent directory name
                    parent_name = parts[device_idx - 1]
                    if parent_name != category:
                        name = f"{category}/{parent_name}"
                    else:
                        name = category
            else:
                name = category

            return name, category
        except (ValueError, IndexError):
            return "unknown", "unknown"

    def count_lines_of_code(self, file_path: Path) -> int:
        """Count non-empty, non-comment lines."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            loc = 0
            in_block_comment = False

            for line in lines:
                stripped = line.strip()

                # Skip empty lines
                if not stripped:
                    continue

                # Handle block comments
                if "/*" in stripped:
                    in_block_comment = True
                if "*/" in stripped:
                    in_block_comment = False
                    continue
                if in_block_comment:
                    continue

                # Skip single-line comments
                if stripped.startswith("//"):
                    continue

                loc += 1

            return loc
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return 0

    def count_pattern(self, file_path: Path, pattern: str) -> int:
        """Count occurrences of a regex pattern in file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            matches = re.findall(pattern, content)
            return len(matches)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return 0

    def count_conditionals(self, file_path: Path) -> int:
        """Count conditional branches (if/else/switch)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Count if statements (excluding else if)
            if_count = len(re.findall(r"\bif\s*\(", content))
            # Count else statements
            else_count = len(re.findall(r"\belse\s*\{", content))
            # Count switch statements
            switch_count = len(re.findall(r"\bswitch\s*\(", content))

            return if_count + else_count + switch_count
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return 0

    def analyze_program_factory(self, file_path: Path) -> ProgramFactoryMetrics:
        """Extract metrics from a single program factory file."""
        op_name, category = self.extract_operation_name(file_path)

        loc = self.count_lines_of_code(file_path)
        kernels = self.count_pattern(file_path, r"CreateKernel\s*\(")
        cbs = self.count_pattern(file_path, r"CreateCircularBuffer\s*\(")
        runtime_args = self.count_pattern(file_path, r"SetRuntimeArgs\s*\(")
        semaphores = self.count_pattern(file_path, r"CreateSemaphore\s*\(")
        conditionals = self.count_conditionals(file_path)

        # CCL-specific metrics
        mesh_workloads = self.count_pattern(file_path, r"create_mesh_workload\s*\(")
        global_semaphores = self.count_pattern(file_path, r"(create_global_semaphore|GlobalSemaphore)\s*\(")
        fabric_apis = self.count_pattern(file_path, r"(fabric::|mesh_graph|MeshWorkload|MeshDevice)")
        command_builders = self.count_pattern(
            file_path, r"(command_stream_builders|command_lowering|worker_builder|CCLCommand)"
        )

        return ProgramFactoryMetrics(
            file_path=str(file_path.relative_to(self.operations_root.parent.parent)),
            operation_name=op_name,
            lines_of_code=loc,
            create_kernel_count=kernels,
            create_circular_buffer_count=cbs,
            set_runtime_args_count=runtime_args,
            create_semaphore_count=semaphores,
            conditional_branches=conditionals,
            create_mesh_workload_count=mesh_workloads,
            global_semaphore_count=global_semaphores,
            fabric_api_calls=fabric_apis,
            command_stream_builders=command_builders,
        )

    def count_kernel_files(self, operation_path: Path) -> int:
        """Count kernel .cpp files for an operation."""
        kernel_files = list(operation_path.rglob("kernels/**/*.cpp"))
        return len(kernel_files)

    def aggregate_operation_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics per operation."""
        operation_data = defaultdict(lambda: {"factories": [], "category": "", "kernel_files": 0})

        # Group by operation name
        for factory in self.program_factories:
            op_name = factory.operation_name
            operation_data[op_name]["factories"].append(factory)
            operation_data[op_name]["category"] = (
                factory.operation_name.split("/")[0] if "/" in factory.operation_name else factory.operation_name
            )

        # Count kernel files per operation
        for op_name in operation_data.keys():
            # Try to find the operation directory
            category = operation_data[op_name]["category"]
            op_path = self.operations_root / category
            if op_path.exists():
                operation_data[op_name]["kernel_files"] = self.count_kernel_files(op_path)

        return operation_data

    def normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a metric to 0-1 range."""
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def is_ccl_operation(self, op_name: str, factories: List[ProgramFactoryMetrics]) -> bool:
        """Determine if an operation is CCL-based."""
        # Check if operation name contains ccl
        if "ccl" in op_name.lower():
            return True
        # Check if any factory uses CCL APIs
        for factory in factories:
            if (
                factory.create_mesh_workload_count > 0
                or factory.global_semaphore_count > 0
                or factory.fabric_api_calls > 5
                or factory.command_stream_builders > 0
            ):
                return True
        return False

    def compute_complexity_scores(self):
        """Compute complexity scores for all operations."""
        operation_data = self.aggregate_operation_metrics()

        # Collect all values for normalization
        all_locs = []
        all_kernels = []
        all_cbs = []
        all_runtime_args = []
        all_semaphores = []
        all_mesh_workloads = []
        all_global_semaphores = []

        for op_name, data in operation_data.items():
            total_loc = sum(f.lines_of_code for f in data["factories"])
            total_kernels = sum(f.create_kernel_count for f in data["factories"])
            total_cbs = sum(f.create_circular_buffer_count for f in data["factories"])
            total_runtime_args = sum(f.set_runtime_args_count for f in data["factories"])
            total_semaphores = sum(f.create_semaphore_count for f in data["factories"])
            total_mesh_workloads = sum(f.create_mesh_workload_count for f in data["factories"])
            total_global_semaphores = sum(f.global_semaphore_count for f in data["factories"])

            all_locs.append(total_loc)
            all_kernels.append(total_kernels)
            all_cbs.append(total_cbs)
            all_runtime_args.append(total_runtime_args)
            all_semaphores.append(total_semaphores)
            all_mesh_workloads.append(total_mesh_workloads)
            all_global_semaphores.append(total_global_semaphores)

        # Compute min/max for normalization
        min_loc, max_loc = min(all_locs), max(all_locs)
        min_kernels, max_kernels = min(all_kernels), max(all_kernels)
        min_cbs, max_cbs = min(all_cbs), max(all_cbs)
        min_runtime_args, max_runtime_args = min(all_runtime_args), max(all_runtime_args)
        min_semaphores, max_semaphores = min(all_semaphores), max(all_semaphores)
        min_mesh_workloads, max_mesh_workloads = min(all_mesh_workloads), max(
            all_mesh_workloads
        ) if all_mesh_workloads else (0, 0)
        min_global_semaphores, max_global_semaphores = min(all_global_semaphores), max(
            all_global_semaphores
        ) if all_global_semaphores else (0, 0)

        # Compute complexity for each operation
        for op_name, data in operation_data.items():
            factories = data["factories"]
            total_loc = sum(f.lines_of_code for f in factories)
            total_kernels = sum(f.create_kernel_count for f in factories)
            total_cbs = sum(f.create_circular_buffer_count for f in factories)
            total_runtime_args = sum(f.set_runtime_args_count for f in factories)
            total_semaphores = sum(f.create_semaphore_count for f in factories)
            total_conditionals = sum(f.conditional_branches for f in factories)
            total_mesh_workloads = sum(f.create_mesh_workload_count for f in factories)
            total_global_semaphores = sum(f.global_semaphore_count for f in factories)
            total_fabric_apis = sum(f.fabric_api_calls for f in factories)
            total_command_builders = sum(f.command_stream_builders for f in factories)

            is_ccl = self.is_ccl_operation(op_name, factories)

            # Normalize metrics
            norm_loc = self.normalize_metric(total_loc, min_loc, max_loc)
            norm_kernels = self.normalize_metric(total_kernels, min_kernels, max_kernels)
            norm_cbs = self.normalize_metric(total_cbs, min_cbs, max_cbs)
            norm_runtime_args = self.normalize_metric(total_runtime_args, min_runtime_args, max_runtime_args)
            norm_semaphores = self.normalize_metric(total_semaphores, min_semaphores, max_semaphores)
            norm_mesh_workloads = (
                self.normalize_metric(total_mesh_workloads, min_mesh_workloads, max_mesh_workloads)
                if max_mesh_workloads > 0
                else 0.0
            )
            norm_global_semaphores = (
                self.normalize_metric(total_global_semaphores, min_global_semaphores, max_global_semaphores)
                if max_global_semaphores > 0
                else 0.0
            )

            # Weighted complexity score - different weights for CCL operations
            if is_ccl:
                # CCL operations: higher weight on mesh workloads and global semaphores
                complexity_score = (
                    0.25 * norm_loc
                    + 0.15 * norm_kernels
                    + 0.15 * norm_cbs
                    + 0.10 * norm_runtime_args
                    + 0.05 * norm_semaphores
                    + 0.20 * norm_mesh_workloads
                    + 0.10 * norm_global_semaphores
                )
            else:
                # Standard operations: original formula
                complexity_score = (
                    0.3 * norm_loc
                    + 0.25 * norm_kernels
                    + 0.2 * norm_cbs
                    + 0.15 * norm_runtime_args
                    + 0.1 * norm_semaphores
                )

            # Determine bucket
            if complexity_score < 0.2:
                bucket = "Simple"
            elif complexity_score < 0.5:
                bucket = "Standard"
            elif complexity_score < 0.8:
                bucket = "Complex"
            else:
                bucket = "Very Complex"

            op_complexity = OperationComplexity(
                operation_name=op_name,
                operation_category=data["category"],
                program_factory_count=len(factories),
                total_lines_of_code=total_loc,
                total_kernels=total_kernels,
                total_circular_buffers=total_cbs,
                total_runtime_args=total_runtime_args,
                total_semaphores=total_semaphores,
                total_conditionals=total_conditionals,
                kernel_file_count=data["kernel_files"],
                total_mesh_workloads=total_mesh_workloads,
                total_global_semaphores=total_global_semaphores,
                total_fabric_api_calls=total_fabric_apis,
                total_command_builders=total_command_builders,
                is_ccl_operation=is_ccl,
                normalized_loc=norm_loc,
                normalized_kernels=norm_kernels,
                normalized_cbs=norm_cbs,
                normalized_runtime_args=norm_runtime_args,
                normalized_semaphores=norm_semaphores,
                normalized_mesh_workloads=norm_mesh_workloads,
                normalized_global_semaphores=norm_global_semaphores,
                complexity_score=complexity_score,
                complexity_bucket=bucket,
            )

            self.operation_complexities.append(op_complexity)

        # Sort by complexity score (descending)
        self.operation_complexities.sort(key=lambda x: x.complexity_score, reverse=True)

    def compute_program_factory_scores(self) -> List[Tuple[ProgramFactoryMetrics, float]]:
        """Compute complexity scores for individual program factories."""
        if not self.program_factories:
            return []

        # Collect all values for normalization
        all_locs = [f.lines_of_code for f in self.program_factories]
        all_kernels = [f.create_kernel_count for f in self.program_factories]
        all_cbs = [f.create_circular_buffer_count for f in self.program_factories]
        all_runtime_args = [f.set_runtime_args_count for f in self.program_factories]
        all_semaphores = [f.create_semaphore_count for f in self.program_factories]
        all_mesh_workloads = [f.create_mesh_workload_count for f in self.program_factories]
        all_global_semaphores = [f.global_semaphore_count for f in self.program_factories]

        min_loc, max_loc = min(all_locs), max(all_locs)
        min_kernels, max_kernels = min(all_kernels), max(all_kernels)
        min_cbs, max_cbs = min(all_cbs), max(all_cbs)
        min_runtime_args, max_runtime_args = min(all_runtime_args), max(all_runtime_args)
        min_semaphores, max_semaphores = min(all_semaphores), max(all_semaphores)
        min_mesh_workloads, max_mesh_workloads = min(all_mesh_workloads), max(
            all_mesh_workloads
        ) if all_mesh_workloads else (0, 0)
        min_global_semaphores, max_global_semaphores = min(all_global_semaphores), max(
            all_global_semaphores
        ) if all_global_semaphores else (0, 0)

        program_scores = []
        for factory in self.program_factories:
            is_ccl = (
                factory.create_mesh_workload_count > 0
                or factory.global_semaphore_count > 0
                or factory.fabric_api_calls > 5
                or factory.command_stream_builders > 0
                or "ccl" in factory.operation_name.lower()
            )

            norm_loc = self.normalize_metric(factory.lines_of_code, min_loc, max_loc)
            norm_kernels = self.normalize_metric(factory.create_kernel_count, min_kernels, max_kernels)
            norm_cbs = self.normalize_metric(factory.create_circular_buffer_count, min_cbs, max_cbs)
            norm_runtime_args = self.normalize_metric(
                factory.set_runtime_args_count, min_runtime_args, max_runtime_args
            )
            norm_semaphores = self.normalize_metric(factory.create_semaphore_count, min_semaphores, max_semaphores)
            norm_mesh_workloads = (
                self.normalize_metric(factory.create_mesh_workload_count, min_mesh_workloads, max_mesh_workloads)
                if max_mesh_workloads > 0
                else 0.0
            )
            norm_global_semaphores = (
                self.normalize_metric(factory.global_semaphore_count, min_global_semaphores, max_global_semaphores)
                if max_global_semaphores > 0
                else 0.0
            )

            if is_ccl:
                score = (
                    0.25 * norm_loc
                    + 0.15 * norm_kernels
                    + 0.15 * norm_cbs
                    + 0.10 * norm_runtime_args
                    + 0.05 * norm_semaphores
                    + 0.20 * norm_mesh_workloads
                    + 0.10 * norm_global_semaphores
                )
            else:
                score = (
                    0.3 * norm_loc
                    + 0.25 * norm_kernels
                    + 0.2 * norm_cbs
                    + 0.15 * norm_runtime_args
                    + 0.1 * norm_semaphores
                )

            program_scores.append((factory, score))

        # Sort by score (descending)
        program_scores.sort(key=lambda x: x[1], reverse=True)
        return program_scores

    def analyze(self):
        """Run the complete analysis."""
        print("Finding program factory files...")
        factories = self.find_program_factories()
        print(f"Found {len(factories)} program factory files")

        print("Extracting metrics from program factories...")
        for factory_path in factories:
            metrics = self.analyze_program_factory(factory_path)
            self.program_factories.append(metrics)

        print("Computing complexity scores...")
        self.compute_complexity_scores()

        print(f"Analyzed {len(self.operation_complexities)} operations")

    def save_json_report(self, output_path: Path):
        """Save detailed JSON report."""
        report = {
            "program_factories": [asdict(f) for f in self.program_factories],
            "operations": [asdict(op) for op in self.operation_complexities],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Saved JSON report to {output_path}")

    def generate_markdown_report(self, output_path: Path):
        """Generate human-readable markdown report."""
        with open(output_path, "w") as f:
            f.write("# TT-NN Operation Complexity Ranking\n\n")
            f.write("This report ranks all TT-NN operations by complexity to guide refactoring efforts.\n\n")

            # Summary statistics
            f.write("## Summary Statistics\n\n")
            buckets = defaultdict(int)
            for op in self.operation_complexities:
                buckets[op.complexity_bucket] += 1

            f.write("| Bucket | Count |\n")
            f.write("|--------|-------|\n")
            for bucket in ["Simple", "Standard", "Complex", "Very Complex"]:
                f.write(f"| {bucket} | {buckets[bucket]} |\n")
            f.write("\n")

            # Bucket breakdown
            f.write("## Complexity Buckets\n\n")
            for bucket in ["Simple", "Standard", "Complex", "Very Complex"]:
                ops_in_bucket = [op for op in self.operation_complexities if op.complexity_bucket == bucket]
                if not ops_in_bucket:
                    continue

                f.write(f"### {bucket} ({len(ops_in_bucket)} operations)\n\n")
                f.write("| Operation | Score | LOC | Kernels | CBs | Program Factories |\n")
                f.write("|-----------|-------|-----|---------|-----|------------------|\n")

                for op in sorted(ops_in_bucket, key=lambda x: x.complexity_score, reverse=True):
                    f.write(
                        f"| {op.operation_name} | {op.complexity_score:.3f} | "
                        f"{op.total_lines_of_code} | {op.total_kernels} | "
                        f"{op.total_circular_buffers} | {op.program_factory_count} |\n"
                    )
                f.write("\n")

            # Full ranking
            f.write("## Complete Ranking (Most Complex First)\n\n")
            f.write("| Rank | Operation | Category | Score | Bucket | LOC | Kernels | CBs | Factories |\n")
            f.write("|------|-----------|----------|-------|--------|-----|---------|-----|----------|\n")

            for rank, op in enumerate(self.operation_complexities, 1):
                f.write(
                    f"| {rank} | {op.operation_name} | {op.operation_category} | "
                    f"{op.complexity_score:.3f} | {op.complexity_bucket} | "
                    f"{op.total_lines_of_code} | {op.total_kernels} | "
                    f"{op.total_circular_buffers} | {op.program_factory_count} |\n"
                )

            # Characteristics
            f.write("\n## Complexity Characteristics\n\n")
            f.write("### Simple (0.0 - 0.2)\n")
            f.write("- Single kernel or no kernels\n")
            f.write("- Minimal circular buffers (0-2)\n")
            f.write("- No semaphores\n")
            f.write("- Low LOC (< 500)\n")
            f.write("- Examples: move, simple data movement operations\n\n")

            f.write("### Standard (0.2 - 0.5)\n")
            f.write("- 2-3 kernels\n")
            f.write("- Moderate CB usage (3-5)\n")
            f.write("- Basic runtime args\n")
            f.write("- Medium LOC (200-1000)\n")
            f.write("- Examples: element-wise operations, simple reductions\n\n")

            f.write("### Complex (0.5 - 0.8)\n")
            f.write("- Multiple program variants\n")
            f.write("- Sharding support\n")
            f.write("- 4-7 kernels\n")
            f.write("- Higher LOC (1000-2000)\n")
            f.write("- Examples: matmul variants, normalization ops\n\n")

            f.write("### Very Complex (0.8 - 1.0)\n")
            f.write("- Multicast operations\n")
            f.write("- Advanced synchronization (semaphores)\n")
            f.write("- 8+ kernels\n")
            f.write("- Very high LOC (2000+)\n")
            f.write("- Complex runtime arg setup\n")
            f.write("- Examples: matmul_mcast_1d, transformer operations\n\n")

            # CCL Operations Highlight
            ccl_ops = [op for op in self.operation_complexities if op.is_ccl_operation]
            if ccl_ops:
                f.write("## CCL Operations Complexity\n\n")
                f.write("CCL (Collective Communication Library) operations use specialized APIs:\n")
                f.write("- `create_mesh_workload` for multi-device coordination\n")
                f.write("- `GlobalSemaphore` for cross-device synchronization\n")
                f.write("- Fabric/mesh APIs for device-to-device communication\n")
                f.write("- Command stream builders for complex communication patterns\n\n")
                f.write(
                    "These operations are ranked using adjusted weights that emphasize mesh workloads and global semaphores.\n\n"
                )
                f.write("| Operation | Score | LOC | Kernels | Mesh Workloads | Global Semaphores | Fabric APIs |\n")
                f.write("|-----------|-------|-----|---------|----------------|-------------------|-------------|\n")
                for op in sorted(ccl_ops, key=lambda x: x.complexity_score, reverse=True):
                    f.write(
                        f"| {op.operation_name} | {op.complexity_score:.3f} | "
                        f"{op.total_lines_of_code} | {op.total_kernels} | "
                        f"{op.total_mesh_workloads} | {op.total_global_semaphores} | "
                        f"{op.total_fabric_api_calls} |\n"
                    )
                f.write("\n")

            # Program Factory Ranking
            f.write("## Program Factory Ranking (Individual Programs)\n\n")
            f.write("This section ranks all individual program factories by complexity.\n")
            f.write("Useful for identifying which specific program variants are most complex.\n\n")
            program_scores = self.compute_program_factory_scores()
            f.write("| Rank | Program Factory | Operation | Score | LOC | Kernels | CBs | CCL APIs |\n")
            f.write("|------|-----------------|-----------|-------|-----|---------|-----|----------|\n")
            for rank, (factory, score) in enumerate(program_scores[:100], 1):  # Top 100
                ccl_indicators = []
                if factory.create_mesh_workload_count > 0:
                    ccl_indicators.append(f"MW:{factory.create_mesh_workload_count}")
                if factory.global_semaphore_count > 0:
                    ccl_indicators.append(f"GS:{factory.global_semaphore_count}")
                if factory.fabric_api_calls > 0:
                    ccl_indicators.append(f"F:{factory.fabric_api_calls}")
                ccl_str = ", ".join(ccl_indicators) if ccl_indicators else "-"
                filename = factory.file_path.split("/")[-1]
                f.write(
                    f"| {rank} | {filename} | {factory.operation_name} | {score:.3f} | "
                    f"{factory.lines_of_code} | {factory.create_kernel_count} | "
                    f"{factory.create_circular_buffer_count} | {ccl_str} |\n"
                )
            f.write(f"\n*Showing top 100 of {len(program_scores)} program factories*\n\n")

            # Top Programs Per Operation
            f.write("## Top Programs Per Operation\n\n")
            f.write("For each operation, shows the most complex program factories.\n\n")
            operation_data = defaultdict(list)
            for factory, score in program_scores:
                operation_data[factory.operation_name].append((factory, score))

            # Show top 3 programs for operations with multiple factories
            for op in sorted(self.operation_complexities, key=lambda x: x.complexity_score, reverse=True):
                if op.program_factory_count > 1:
                    programs = sorted(operation_data[op.operation_name], key=lambda x: x[1], reverse=True)[:3]
                    f.write(f"### {op.operation_name} (Score: {op.complexity_score:.3f})\n\n")
                    f.write("| Program Factory | Score | LOC | Kernels | CBs |\n")
                    f.write("|-----------------|-------|-----|---------|-----|\n")
                    for factory, score in programs:
                        filename = factory.file_path.split("/")[-1]
                        f.write(
                            f"| {filename} | {score:.3f} | {factory.lines_of_code} | "
                            f"{factory.create_kernel_count} | {factory.create_circular_buffer_count} |\n"
                        )
                    f.write("\n")

        print(f"Saved markdown report to {output_path}")


def main():
    """Main entry point."""
    # Find operations directory
    script_dir = Path(__file__).parent
    operations_root = script_dir.parent.parent / "cpp" / "ttnn" / "operations"

    if not operations_root.exists():
        print(f"Error: Operations directory not found at {operations_root}")
        return 1

    analyzer = ComplexityAnalyzer(operations_root)
    analyzer.analyze()

    # Save reports
    output_dir = script_dir
    analyzer.save_json_report(output_dir / "complexity_report.json")
    analyzer.generate_markdown_report(output_dir / "complexity_ranking.md")

    return 0


if __name__ == "__main__":
    exit(main())
