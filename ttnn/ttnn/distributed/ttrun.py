#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt-run - MPI process launcher for TT-Metal and TTNN distributed applications."""

import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import click
import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

TT_RUN_PREFIX = "[tt-run]"
DEFAULT_CACHE_DIR_PATTERN = "{home}/.cache/{hostname}_rank{rank}"
INTERRUPTED_EXIT_CODE = 130  # 128 + SIGINT
PRETTY_PRINT_THRESHOLD = 10  # Minimum args to trigger multi-line formatting


class RankBinding(BaseModel):
    """Binding between MPI rank to target MeshId and HostRankId as defined in the mesh graph descriptor."""

    rank: int = Field(..., ge=0, description="MPI rank (must be >= 0)")
    mesh_id: int = Field(..., ge=0, description="`MeshId` defines the mesh to which the rank belongs")
    env_overrides: Dict[str, str] = Field(default_factory=dict, description="Environment variable overrides")


class TTRunConfig(BaseModel):
    """Rank binding YAML specification consumed by `tt-run`."""

    rank_bindings: List[RankBinding] = Field(..., min_length=1, description="Rank to fabric bindings")
    global_env: Dict[str, str] = Field(default_factory=dict, description="Global environment variables for all ranks")
    mesh_graph_desc_path: str = Field(..., description="Path to mesh graph descriptor")

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


def parse_binding_config(yaml_path: Path) -> TTRunConfig:
    """Parse YAML configuration file with schema validation."""
    if not yaml_path.exists():
        raise ValueError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    try:
        return TTRunConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


def get_rank_environment(
    binding: RankBinding, config: TTRunConfig, mesh_to_host_rank_id: Dict[int, int]
) -> Dict[str, str]:
    """Get all environment variables for a specific rank.

    Args:
        binding: Rank binding configuration
        config: Global configuration

    Returns:
        Dictionary of environment variables for this rank
    """
    env = {
        "TT_METAL_CACHE": os.environ.get(
            "TT_METAL_CACHE",
            DEFAULT_CACHE_DIR_PATTERN.format(home=str(Path.home()), hostname=os.uname().nodename, rank=binding.rank),
        ),  # Need to explicitly configure this because kernel cache is not multi-process safe (#21089)
        "TT_MESH_ID": str(binding.mesh_id),
        "TT_HOST_RANK": str(mesh_to_host_rank_id[binding.mesh_id]),
        "TT_MESH_GRAPH_DESC_PATH": config.mesh_graph_desc_path,
    }
    mesh_to_host_rank_id[binding.mesh_id] += 1

    # Apply environment variables with expansion and proper precedence
    # Global environment variables first
    env.update({k: os.path.expandvars(v) for k, v in config.global_env.items()})
    # Rank-specific overrides last (higher precedence)
    env.update({k: os.path.expandvars(v) for k, v in binding.env_overrides.items()})

    return env


def build_rank_environment_args(
    binding: RankBinding, config: TTRunConfig, mesh_to_host_rank_id: Dict[int, int]
) -> List[str]:
    """Build environment variable arguments for mpirun.

    Args:
        binding: Rank binding configuration
        config: Global configuration

    Returns:
        List of ["-x", "KEY=value"] arguments for mpirun
    """
    env_args = []
    env = get_rank_environment(binding, config, mesh_to_host_rank_id)

    for key, value in env.items():
        env_args.extend(["-x", f"{key}={value}"])

    return env_args


def build_mpi_command(config: TTRunConfig, program: List[str], mpi_args: Optional[List[str]] = None) -> List[str]:
    """Build OpenMPI command with per-rank environment variables."""
    # Find mpirun-ulfm executable, fall back to mpirun if not found
    mpi_launcher = shutil.which("mpirun-ulfm")
    if not mpi_launcher:
        logger.warning(f"{TT_RUN_PREFIX} mpirun-ulfm not found in PATH, falling back to mpirun")
        mpi_launcher = "mpirun"

    cmd = [mpi_launcher]

    if mpi_args:
        cmd.extend(mpi_args)

    # Build per-rank application contexts
    mesh_to_host_rank_id = defaultdict(int)
    for i, binding in enumerate(config.rank_bindings):
        if i > 0:
            cmd.append(":")

        cmd.extend(["-np", "1"])
        cmd.extend(build_rank_environment_args(binding, config, mesh_to_host_rank_id))
        cmd.extend(program)

    return cmd


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

        logger.info(" \\\n    ".join(parts))
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
@click.pass_context
def main(ctx: click.Context, rank_binding: Path, dry_run: bool, verbose: bool, mpi_args: Optional[List[str]]) -> None:
    """tt-run - MPI process launcher for TT-Metal and TTNN distributed applications

    tt-run is a lightweight wrapper around `mpirun` that simplifies launching
    TT-Metal and TT-NN distributed applications by automatically mapping
    MPI ranks to target MeshId and HostRankId as defined in the mesh graph descriptor.

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
            env_overrides:
              RANK_0_ENV_VAR: "value"
          - rank: 1
            mesh_id: 0
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
        - TT_HOST_RANK: Host rank within the mesh
        - TT_METAL_CACHE: Per-rank cache directory
        - TT_METAL_HOME: TT-Metal installation directory
        - PYTHONPATH: Python module search path
        - TT_MESH_GRAPH_DESC_PATH: Path to mesh graph descriptor

    See examples/ttrun/ for example configuration files.
    """
    program = ctx.args
    try:
        config = parse_binding_config(rank_binding)
    except (ValueError, ValidationError) as e:
        raise click.ClickException(f"Configuration error: {e}")

    if not program:
        raise click.ClickException("No program specified. Please provide a program to run.")

    # Validate program executable exists
    program_path = Path(program[0])
    if not program_path.exists() and not shutil.which(program[0]):
        raise click.ClickException(f"Program not found: {program[0]}")

    # Build MPI command
    mpi_cmd = build_mpi_command(config, program, mpi_args)

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
