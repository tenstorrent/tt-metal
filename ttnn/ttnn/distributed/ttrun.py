#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt-run - MPI process launcher for TT-Metal and TTNN distributed applications."""

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

TT_RUN_PREFIX = "[tt-run]"
DEFAULT_CACHE_DIR_PATTERN = "{home}/.cache/{hostname}_rank{rank}"
DEFAULT_LD_LIBRARY_PATH = "{home}/build/lib"
INTERRUPTED_EXIT_CODE = 130  # 128 + SIGINT
PRETTY_PRINT_THRESHOLD = 10  # Minimum args to trigger multi-line formatting


class RankBinding(BaseModel):
    """Binding between MPI rank to target MeshId and MeshHostRankId as defined in the mesh graph descriptor."""

    rank: int = Field(..., ge=0, description="MPI rank (must be >= 0)")
    mesh_id: int = Field(..., ge=0, description="`MeshId` defines the mesh to which the rank belongs")
    mesh_host_rank: Optional[int] = Field(None, ge=0, description="Host rank within the mesh")
    env_overrides: Dict[str, str] = Field(default_factory=dict, description="Environment variable overrides")


class TTRunConfig(BaseModel):
    """Rank binding YAML specification consumed by `tt-run`."""

    rank_bindings: List[RankBinding] = Field(..., min_length=1, description="Rank to fabric bindings")
    global_env: Dict[str, str] = Field(default_factory=dict, description="Global environment variables for all ranks")
    mesh_graph_desc_path: str = Field(..., description="Path to mesh graph descriptor")
    mock_cluster_rank_binding: Dict[int, Path] = Field(
        default_factory=dict, description="Mock cluster rank binding configuration"
    )

    @field_validator("rank_bindings")
    def validate_ranks(cls, bindings: List[RankBinding]) -> List[RankBinding]:
        """Ensure ranks are unique and contiguous starting from 0"""
        ranks = [b.rank for b in bindings]

        if len(ranks) != len(set(ranks)):
            raise ValueError("Duplicate ranks found in bindings")

        sorted_ranks = sorted(ranks)
        if sorted_ranks != list(range(len(ranks))):
            raise ValueError(f"Ranks must be contiguous from 0. Got: {sorted_ranks}")

        return bindings

    @field_validator("mesh_graph_desc_path")
    def validate_mesh_graph_exists(cls, path: str) -> str:
        """Ensure mesh graph descriptor file exists"""
        mesh_path = Path(path).expanduser().resolve()
        if not mesh_path.is_file():
            raise ValueError(f"Mesh graph descriptor not found: {mesh_path}")
        return str(mesh_path)


def parse_binding_config(yaml_path: Path, mock_cluster_rank_binding: Optional[Path] = None) -> TTRunConfig:
    """Parse YAML configuration file with schema validation."""
    if not yaml_path.exists():
        raise ValueError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    try:
        config = TTRunConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")

    # Parse mock cluster rank binding configuration
    if mock_cluster_rank_binding:
        with open(mock_cluster_rank_binding, "r") as f:
            mock_data = yaml.safe_load(f)

        # Validate mock cluster rank binding configuration
        for rank, path in mock_data["rank_to_cluster_mock_cluster_desc"].items():
            if not Path(path).expanduser().resolve().is_file():
                raise ValueError(f"Mock cluster rank binding configuration file not found: {path}")

        config.mock_cluster_rank_binding = mock_data["rank_to_cluster_mock_cluster_desc"]

    return config


@dataclass
class TracyConfig:
    output_root: Path
    base_port: int
    extra_args: List[str]


def get_rank_environment(
    binding: RankBinding,
    config: TTRunConfig,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Get all environment variables for a specific rank.

    Args:
        binding: Rank binding configuration
        config: Global configuration

    Returns:
        Dictionary of environment variables for this rank
    """
    # Handle TT_METAL_CACHE with rank-specific suffix to prevent cache conflicts/collisions between ranks (multi-process safety).
    hostname = os.uname().nodename

    if "TT_METAL_CACHE" in os.environ:
        user_cache_path = os.environ["TT_METAL_CACHE"]
        base_path = user_cache_path
        logger.warning(
            f"{TT_RUN_PREFIX} User-provided TT_METAL_CACHE '{user_cache_path}' "
            f"will be modified with rank suffix for multi-process safety"
        )
    else:
        # Use default pattern when TT_METAL_CACHE is not set
        base_path = f"{Path.home()}/.cache"

    # Apply consistent rank suffix pattern to both user-provided and default paths
    cache_path = f"{base_path}_{hostname}_rank{binding.rank}"

    env = {
        "TT_METAL_CACHE": cache_path,
        "TT_MESH_ID": str(binding.mesh_id),
        "TT_MESH_GRAPH_DESC_PATH": config.mesh_graph_desc_path,
        "TT_METAL_HOME": os.environ.get("TT_METAL_HOME", str(Path.home())),
        "PYTHONPATH": os.environ.get("PYTHONPATH", str(Path.home())),
        # 26640: TODO - Investigate why this needs to be set for multi-host CI environments
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", DEFAULT_LD_LIBRARY_PATH.format(home=str(Path.home()))),
    }

    # Add TT_MESH_HOST_RANK only if mesh_host_rank is set
    if binding.mesh_host_rank is not None:
        env["TT_MESH_HOST_RANK"] = str(binding.mesh_host_rank)

    if config.mock_cluster_rank_binding:
        env["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = config.mock_cluster_rank_binding[binding.rank]

    # Apply environment variables with expansion and proper precedence
    # Global environment variables first
    env.update({k: os.path.expandvars(v) for k, v in config.global_env.items()})
    # Rank-specific overrides last (higher precedence)
    env.update({k: os.path.expandvars(v) for k, v in binding.env_overrides.items()})

    if extra_env:
        env.update(extra_env)

    return env


def build_rank_environment_args(
    binding: RankBinding,
    config: TTRunConfig,
    extra_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build environment variable arguments for mpirun.

    Args:
        binding: Rank binding configuration
        config: Global configuration

    Returns:
        List of ["-x", "KEY=value"] arguments for mpirun
    """
    env_args = []
    env = get_rank_environment(binding, config, extra_env)

    for key, value in env.items():
        env_args.extend(["-x", f"{key}={value}"])

    return env_args


def build_mpi_command(
    config: TTRunConfig,
    program: List[str],
    mpi_args: Optional[List[str]] = None,
    tracy_config: Optional[TracyConfig] = None,
) -> List[str]:
    """Build OpenMPI command with per-rank environment variables."""
    # Find mpirun-ulfm executable, fall back to mpirun if not found
    mpi_launcher = shutil.which("mpirun-ulfm")
    if not mpi_launcher:
        logger.warning(f"{TT_RUN_PREFIX} mpirun-ulfm not found in PATH, falling back to mpirun")
        mpi_launcher = "mpirun"

    cmd = [mpi_launcher]

    # Check if --bind-to is already specified in mpi_args
    bind_to_already_specified = False
    if mpi_args:
        for i, arg in enumerate(mpi_args):
            if arg == "--bind-to":
                bind_to_already_specified = True
                break

    # Add --bind-to none only if not already specified
    if not bind_to_already_specified:
        cmd.extend(["--bind-to", "none"])

    if mpi_args:
        cmd.extend(mpi_args)

    # Build per-rank application contexts
    for i, binding in sorted(enumerate(config.rank_bindings), key=lambda x: x[1].rank):
        if i > 0:
            cmd.append(":")

        cmd.extend(["-np", "1"])
        tracy_env: Dict[str, str] = {}
        rank_program = program
        if tracy_config:
            rank_program, tracy_env = wrap_program_with_tracy(program, binding, tracy_config)
        cmd.extend(build_rank_environment_args(binding, config, tracy_env))
        cmd.extend(rank_program)

    return cmd


def wrap_program_with_tracy(
    program: List[str], binding: RankBinding, tracy_config: TracyConfig
) -> Tuple[List[str], Dict[str, str]]:
    """Return the tracy-wrapped command and any extra env for a rank."""

    rank_output_dir = (tracy_config.output_root / f"rank{binding.rank}").resolve()
    rank_output_dir.mkdir(parents=True, exist_ok=True)
    port = tracy_config.base_port + binding.rank

    tracy_cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "--port",
        str(port),
        "-o",
        str(rank_output_dir),
    ]

    if tracy_config.extra_args:
        tracy_cmd.extend(tracy_config.extra_args)

    tracy_cmd.extend(program)

    extra_env = {
        "TT_METAL_PROFILER_DIR": str(rank_output_dir),
        "TRACY_PORT": str(port),
    }

    return tracy_cmd, extra_env


def print_command(cmd: List[str], prefix: str = TT_RUN_PREFIX) -> None:
    """Pretty print a command for readability."""
    if len(cmd) > PRETTY_PRINT_THRESHOLD:
        logger.info(f"{prefix} Command:")
        parts = []
        current_part = ["mpirun"]

        for arg in cmd[1:]:
            if arg == ":":
                parts.append(" ".join(current_part))
                current_part = [":"]
            else:
                current_part.append(arg)

        if current_part:
            parts.append(" ".join(current_part))

        logger.info(f"{prefix} Command: " + " ".join(parts))

    else:
        logger.info(f"{prefix} Command: " + " ".join(cmd))


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--rank-binding",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Rank binding configuration file (YAML)",
)
@click.option("--dry-run", is_flag=True, help="Print command without executing")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option(
    "--mpi-args",
    callback=lambda ctx, param, value: shlex.split(value) if value else None,
    help="Additional MPI arguments (quoted)",
)
@click.option(
    "--mock-cluster-rank-binding",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Mock cluster rank binding configuration file (YAML)",
)
@click.option(
    "--profile-with-tracy",
    is_flag=True,
    default=False,
    help="Wrap each rank command with `python -m tracy` (unique port/output per rank).",
)
@click.option(
    "--tracy-output-root",
    required=False,
    type=click.Path(path_type=Path),
    help="Base directory for Tracy artifacts (defaults to $TT_METAL_HOME/generated/profiler/ttrun).",
)
@click.option(
    "--tracy-base-port",
    type=int,
    default=8086,
    show_default=True,
    help="Base port for Tracy capture; each rank increments this by its MPI rank.",
)
@click.option(
    "--tracy-extra-args",
    callback=lambda ctx, param, value: shlex.split(value) if value else [],
    help="Additional arguments to pass through to `python -m tracy` (quoted).",
)
@click.pass_context
def main(
    ctx: click.Context,
    rank_binding: Path,
    dry_run: bool,
    verbose: bool,
    mpi_args: Optional[List[str]],
    mock_cluster_rank_binding: Optional[Path],
    profile_with_tracy: bool,
    tracy_output_root: Optional[Path],
    tracy_base_port: int,
    tracy_extra_args: List[str],
) -> None:
    """tt-run - MPI process launcher for TT-Metal and TTNN distributed applications

    tt-run is a lightweight wrapper around `mpirun` that simplifies launching
    TT-Metal and TT-NN distributed applications by automatically mapping
    MPI ranks to target MeshId and MeshHostRankId as defined in the mesh graph descriptor.

    \b
    Quick Start:
        # Launch with rank binding configuration
        tt-run --rank-binding rank_binding.yaml ./my_app

        # Launch on multiple hosts with rankfile
        tt-run --rank-binding binding.yaml --mpi-args "--rankfile hosts.txt" ./my_app

    \b
    Rank Binding YAML Example:
        rank_bindings:
          - rank: 0                  # MPI rank
            mesh_id: 0               # Mesh ID
            mesh_host_rank: 0        # Host rank within the mesh
            env_overrides:
              RANK_0_ENV_VAR: "value"
          - rank: 1
            mesh_id: 0
            mesh_host_rank: 1        # Host rank within the mesh
            env_overrides:
              RANK_1_ENV_VAR: "value"
        global_env:                  # Environment variables for all ranks
          TT_LOGGER_LEVEL: Debug
          PYTHONPATH: "${HOME}/my_project:${PYTHONPATH}"  # Supports env var expansion
        mesh_graph_desc_path: "path/to/mesh_graph.yaml"  # Required

    \b
    Examples:
        # Single host, multiple processes
        tt-run --rank-binding rank_binding.yaml ./my_app

        # Multi-host with rankfile
        tt-run --rank-binding binding.yaml --mpi-args "--rankfile hosts.txt" ./my_app

        # With additional MPI args
        tt-run --rank-binding binding.yaml --mpi-args "--bind-to core" ./my_app

        # Dry run to see command
        tt-run --rank-binding binding.yaml --dry-run ./my_app

    \b
    Environment Variables:
        The following variables are automatically set for each rank:
        - TT_MESH_ID: Mesh identifier
        - TT_MESH_HOST_RANK: Host rank within the mesh
        - TT_METAL_CACHE: Per-rank cache directory
        - TT_METAL_HOME: TT-Metal installation directory
        - PYTHONPATH: Python module search path
        - LD_LIBRARY_PATH: Library search path
        - TT_MESH_GRAPH_DESC_PATH: Path to mesh graph descriptor
        Default values for the following environment variables will be used if not set when calling tt-run:
        - TT_METAL_HOME: User's home directory
        - PYTHONPATH: User's home directory
        - LD_LIBRARY_PATH: `<USER_HOME>/build/lib`

    \b
    Mock testing:

    For Control plane internal testing, we can use a mock cluster descriptor to initialize control plane without
    any hardware dependencies. To enable mock cluster, use the --mock-cluster-rank-binding flag to specify the mock cluster descriptor mapping file.
    The mock cluster descriptor mapping file is a YAML file that maps each rank to a mock cluster descriptor file.

    Mock Cluster Rank Binding YAML Example:
        rank_to_cluster_mock_cluster_desc:
          - rank: 0
            filename: "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_rank_0.yaml"
          - rank: 1
            filename: "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_rank_1.yaml"

    See examples/ttrun/ for example configuration files.

    \b
    Documentation:
        For comprehensive usage guide, design patterns (SPMD Big-Mesh and Multi-Mesh),
        and integration with MGD 2.0, see:
        tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md
        Section 2.4: Distributed Process Launch with tt-run
    """
    program = list(ctx.args)
    try:
        config = parse_binding_config(rank_binding, mock_cluster_rank_binding)
    except (ValueError, ValidationError) as e:
        raise click.ClickException(f"Configuration error: {e}")

    if not program:
        raise click.ClickException("No program specified. Please provide a program to run.")

    # Validate program executable exists
    program_path = Path(program[0])
    if not program_path.exists() and not shutil.which(program[0]):
        raise click.ClickException(f"Program not found: {program[0]}")

    # Build MPI command
    tracy_config = None
    if profile_with_tracy:
        tracy_root = tracy_output_root
        if tracy_root is None:
            default_root = Path(os.environ.get("TT_METAL_HOME", str(Path.home()))) / "generated/profiler/ttrun"
            tracy_root = default_root
        tracy_root = tracy_root.expanduser().resolve()
        tracy_root.mkdir(parents=True, exist_ok=True)
        if tracy_base_port <= 0:
            raise click.ClickException("--tracy-base-port must be a positive integer")
        tracy_config = TracyConfig(output_root=tracy_root, base_port=tracy_base_port, extra_args=tracy_extra_args)

    mpi_cmd = build_mpi_command(config, program, mpi_args, tracy_config)

    if verbose or dry_run:
        print_command(mpi_cmd)

    if dry_run:
        return

    try:
        result = subprocess.run(mpi_cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully with proper exit code (128 + SIGINT)
        logger.error(f"{TT_RUN_PREFIX} Interrupted")
        sys.exit(INTERRUPTED_EXIT_CODE)
    except OSError as e:
        raise click.ClickException(f"Error launching mpirun: {e}")


if __name__ == "__main__":
    main()
