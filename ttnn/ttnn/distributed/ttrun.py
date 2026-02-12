#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt-run - MPI process launcher for TT-Metal and TTNN distributed applications."""

import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import click
import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

TT_RUN_PREFIX = "[tt-run]"
DEFAULT_CACHE_DIR_PATTERN = "{home}/.cache/{hostname}_rank{rank}"
DEFAULT_LD_LIBRARY_PATH = "{home}/build/lib"
INTERRUPTED_EXIT_CODE = 130  # 128 + SIGINT
PRETTY_PRINT_THRESHOLD = 10  # Minimum args to trigger multi-line formatting

# Store the original working directory at module load time to preserve it
# across mpirun process launches (critical for SLURM/sbatch environments)
ORIGINAL_CWD = Path.cwd().resolve()


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
        """Ensure mesh graph descriptor file exists.

        Uses resolve_path() to search multiple locations for relative paths.
        """
        resolved = resolve_path(path, description="Mesh graph descriptor", must_be_file=True)
        return str(resolved)


def get_search_paths() -> List[Optional[Path]]:
    """Get the ordered list of paths to search for relative file resolution.

    Search order:
    1. TT_METAL_HOME - If environment variable is set (explicit user configuration)
    2. ORIGINAL_CWD - Launch directory (critical for SLURM/sbatch where mpirun
       may change the working directory on remote nodes)
    3. Current working directory - Fallback for local execution

    Returns:
        List of paths to search, with None entries for unset optional paths.
    """
    return [
        Path(os.environ["TT_METAL_HOME"]).expanduser() if os.environ.get("TT_METAL_HOME") else None,
        ORIGINAL_CWD,
        Path.cwd(),
    ]


def resolve_path(
    path: Union[str, Path],
    description: str = "file",
    must_exist: bool = True,
    must_be_file: bool = False,
) -> Path:
    """Resolve a path by searching multiple locations.

    For absolute paths, validates existence if required and returns as-is.
    For relative paths, searches locations from get_search_paths() in order.

    Args:
        path: The path to resolve (can be relative or absolute)
        description: Human-readable description for error messages
        must_exist: Raise ValueError if path doesn't exist (default: True)
        must_be_file: Require path to be a file, not directory (default: False)

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If must_exist=True and path not found in any search location
    """
    expanded_path = Path(path).expanduser()

    # Absolute paths: validate and return
    if expanded_path.is_absolute():
        resolved = expanded_path.resolve()
        if must_exist:
            if must_be_file and not resolved.is_file():
                raise ValueError(f"{description} not found: {resolved}")
            elif not must_be_file and not resolved.exists():
                raise ValueError(f"{description} not found: {resolved}")
        return resolved

    # Relative paths: search multiple locations
    search_paths = get_search_paths()
    check_fn = Path.is_file if must_be_file else Path.exists

    for base_path in search_paths:
        if base_path is None:
            continue
        candidate = (base_path / expanded_path).resolve()
        if check_fn(candidate):
            logger.debug(f"{TT_RUN_PREFIX} Resolved {description}: {path} -> {candidate}")
            return candidate

    # Path not found
    if must_exist:
        searched = [str(p) for p in search_paths if p is not None]
        raise ValueError(
            f"{description} not found: {path}\n"
            f"Searched in: {searched}\n"
            f"Tip: Use an absolute path or ensure the file exists relative to the launch directory."
        )

    # Best-effort resolution when existence check is not required
    return (ORIGINAL_CWD / expanded_path).resolve()


def parse_binding_config(yaml_path: Path, mock_cluster_rank_binding: Optional[Path] = None) -> TTRunConfig:
    """Parse YAML configuration file with schema validation.

    Resolves all relative paths in the configuration against the launch directory
    to ensure proper operation in SLURM/sbatch environments.
    """
    # Resolve the yaml_path first
    resolved_yaml_path = resolve_path(yaml_path, description="Configuration file", must_be_file=True)

    logger.debug(f"{TT_RUN_PREFIX} Loading configuration from: {resolved_yaml_path}")
    logger.debug(f"{TT_RUN_PREFIX} Original CWD: {ORIGINAL_CWD}")

    with open(resolved_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    try:
        config = TTRunConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")

    # Parse mock cluster rank binding configuration
    if mock_cluster_rank_binding:
        resolved_mock_path = resolve_path(
            mock_cluster_rank_binding, description="Mock cluster rank binding configuration", must_be_file=True
        )
        with open(resolved_mock_path, "r") as f:
            mock_data = yaml.safe_load(f)

        # Validate and resolve mock cluster rank binding configuration paths
        resolved_mock_bindings = {}
        for rank, path in mock_data["rank_to_cluster_mock_cluster_desc"].items():
            resolved_path = resolve_path(
                path, description=f"Mock cluster descriptor for rank {rank}", must_be_file=True
            )
            resolved_mock_bindings[rank] = resolved_path

        config.mock_cluster_rank_binding = resolved_mock_bindings

    return config


# Environment variable prefixes that should be automatically passed through to MPI processes
ENV_PASSTHROUGH_PREFIXES = (
    "TT_",  # TT-Metal/TTNN variables
    "ARCH_",  # Architecture variables (e.g., ARCH_NAME)
    "WH_",  # Wormhole-specific variables (e.g., WH_ARCH_YAML)
    "TTNN_",  # TTNN-specific variables (e.g., TTNN_CONFIG_OVERRIDES)
)


def get_rank_environment(binding: RankBinding, config: TTRunConfig) -> Dict[str, str]:
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
        # Use launch directory for cache when TT_METAL_CACHE is not set.
        # This ensures the cache is on the shared filesystem (NFS) visible to all nodes.
        base_path = f"{ORIGINAL_CWD}/.cache"

    # Apply consistent rank suffix pattern to both user-provided and default paths
    cache_path = f"{base_path}_{hostname}_rank{binding.rank}"

    # Start with automatic pass-through of TT-related environment variables
    # This ensures variables like ARCH_NAME, WH_ARCH_YAML, TTNN_CONFIG_OVERRIDES are propagated
    env = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_PASSTHROUGH_PREFIXES):
            env[key] = value

    # Use ORIGINAL_CWD as the default for TT_METAL_HOME when not explicitly set.
    # This assumes the launch directory is on a shared filesystem (NFS) visible to all nodes.
    default_tt_metal_home = os.environ.get("TT_METAL_HOME", str(ORIGINAL_CWD))

    # Set/override core tt-run managed variables
    env.update(
        {
            "TT_METAL_CACHE": cache_path,
            "TT_MESH_ID": str(binding.mesh_id),
            "TT_MESH_GRAPH_DESC_PATH": config.mesh_graph_desc_path,
            "TT_METAL_HOME": default_tt_metal_home,
            "TT_METAL_RUNTIME_ROOT": os.environ.get("TT_METAL_RUNTIME_ROOT", default_tt_metal_home),
            # 26640: TODO - Investigate why this needs to be set for multi-host CI environments
            "LD_LIBRARY_PATH": os.environ.get(
                "LD_LIBRARY_PATH", DEFAULT_LD_LIBRARY_PATH.format(home=str(ORIGINAL_CWD))
            ),
            # Pass the original CWD to subprocesses so they can resolve relative paths correctly
            "TT_RUN_ORIGINAL_CWD": str(ORIGINAL_CWD),
        }
    )

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

    return env


def build_rank_environment_args(binding: RankBinding, config: TTRunConfig) -> List[str]:
    """Build environment variable arguments for mpirun.

    Args:
        binding: Rank binding configuration
        config: Global configuration

    Returns:
        List of ["-x", "KEY=value"] arguments for mpirun
    """
    env_args = []
    env = get_rank_environment(binding, config)

    for key, value in env.items():
        env_args.extend(["-x", f"{key}={value}"])

    return env_args


def build_mpi_command(
    config: TTRunConfig, program: List[str], mpi_args: Optional[List[str]] = None, debug_gdbserver: bool = False
) -> List[str]:
    """Build OpenMPI command with per-rank environment variables."""
    # Check if running in SLURM interactive session
    if os.environ.get("SLURM_JOB_ID") is not None and os.environ.get("SLURM_STEP_ID") is not None:
        logger.warning(f"{TT_RUN_PREFIX} SLURM interactive session detected, using mpirun")
        mpi_launcher = "mpirun"
    else:
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

    if debug_gdbserver:
        cmd.extend(["--tag-output"])

    # Build per-rank application contexts
    for i, binding in sorted(enumerate(config.rank_bindings), key=lambda x: x[1].rank):
        if i > 0:
            cmd.append(":")

        cmd.extend(["-np", "1"])
        cmd.extend(build_rank_environment_args(binding, config))
        program_to_run = program
        if debug_gdbserver:
            port = 20000 + binding.rank
            echo_part = f'echo "Rank {binding.rank} on $(hostname) listening on :{port}";'
            gdbserver_part = f"exec gdbserver :{port}"
            quoted_program_args = " ".join(shlex.quote(arg) for arg in program)
            cmd_str = f"{echo_part} {gdbserver_part} {quoted_program_args}"
            program_to_run = ["bash", "-c", cmd_str]
        cmd.extend(program_to_run)

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
    type=click.Path(path_type=Path),
    required=True,
    help="Rank binding configuration file (YAML). Relative paths are resolved against the launch directory.",
)
@click.option("--dry-run", is_flag=True, help="Print command without executing")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option(
    "--mpi-args",
    callback=lambda ctx, param, value: shlex.split(value) if value else None,
    help="Additional MPI arguments (quoted)",
)
@click.option("--debug-gdbserver", is_flag=True, help="Launch each process with gdbserver for remote debugging")
@click.option(
    "--mock-cluster-rank-binding",
    required=False,
    type=click.Path(path_type=Path),
    help="Mock cluster rank binding configuration file (YAML). Relative paths are resolved against the launch directory.",
)
@click.option(
    "--skip-executable-check", is_flag=True, help="Skip the check if program executable exists on the local host"
)
@click.option(
    "--multihost",
    is_flag=True,
    help="Enable recommended MPI settings for multi-host clusters (TCP transport, tagged output, etc.)",
)
@click.option(
    "--tcp-interface",
    type=str,
    default=None,
    help="Network interface for MPI TCP communication (e.g., 'eth0', 'cnx1'). Implies --multihost.",
)
@click.pass_context
def main(
    ctx: click.Context,
    rank_binding: Path,
    dry_run: bool,
    verbose: bool,
    mpi_args: Optional[List[str]],
    debug_gdbserver: bool,
    mock_cluster_rank_binding: Optional[Path],
    skip_executable_check: bool,
    multihost: bool,
    tcp_interface: Optional[str],
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
    Understanding --rank-binding vs MPI Host Options:
        tt-run's --rank-binding and MPI's host/rankfile options serve complementary purposes:

        --rank-binding (tt-run):
            Configures TT-Metal mesh topology. Maps MPI ranks to:
            - mesh_id: Which TT-Metal mesh the rank belongs to
            - mesh_host_rank: Position within the mesh
            - env_overrides: Per-rank environment variables (e.g., TT_VISIBLE_DEVICES)
            This is about TT-Metal's logical device organization.

        --mpi-args "--host ..." or "--rankfile ...":
            Configures MPI process placement. Tells mpirun:
            - Which physical cluster nodes to spawn processes on
            - How to distribute ranks across those nodes
            This is about physical cluster topology.

        For multi-host setups, you typically need BOTH:
            tt-run --rank-binding mesh_config.yaml \\
                   --mpi-args "--host nodeA,nodeB --map-by rankfile:file=/etc/mpirun/rankfile" \\
                   ./my_app

        The rank-binding configures what each MPI rank "sees" in terms of TT-Metal devices,
        while the MPI host options control where those ranks physically execute.

    \b
    Examples:
        # Single host, multiple processes
        tt-run --rank-binding rank_binding.yaml ./my_app

        # Multi-host with recommended MPI settings
        tt-run --rank-binding binding.yaml --multihost --mpi-args "--rankfile hosts.txt" ./my_app

        # Multi-host with specific network interface (e.g., ConnectX NIC)
        tt-run --rank-binding binding.yaml --tcp-interface cnx1 --mpi-args "--rankfile hosts.txt" ./my_app

        # With additional MPI args
        tt-run --rank-binding binding.yaml --mpi-args "--bind-to core" ./my_app

        # Dry run to see command
        tt-run --rank-binding binding.yaml --dry-run ./my_app

    \b
    Environment Variables:
        The following variables are automatically set for each rank:
        - TT_MESH_ID: Mesh identifier
        - TT_MESH_HOST_RANK: Host rank within the mesh
        - TT_METAL_CACHE: Per-rank cache directory (defaults to `<LAUNCH_DIR>/.cache_<hostname>_rank<N>`)
        - TT_METAL_HOME: TT-Metal installation directory
        - PYTHONPATH: Python module search path
        - PYTHONHOME: Python installation directory
        - LD_LIBRARY_PATH: Library search path
        - TT_MESH_GRAPH_DESC_PATH: Path to mesh graph descriptor
        - TT_RUN_ORIGINAL_CWD: Directory where tt-run was launched (for subprocess path resolution)

        Default values for the following environment variables will be used if not set when calling tt-run:
        - TT_METAL_HOME: Launch directory (where tt-run was invoked)
        - TT_METAL_RUNTIME_ROOT: Same as TT_METAL_HOME
        - PYTHONPATH: Launch directory
        - PYTHONHOME: Launch directory
        - LD_LIBRARY_PATH: `<LAUNCH_DIR>/build/lib`

        This assumes the launch directory is on a shared filesystem (e.g., NFS) visible to all
        cluster nodes, which is the common setup for SLURM environments.

        Additionally, all environment variables with the following prefixes are automatically
        passed through to MPI processes:
        - TT_*: TT-Metal/TTNN variables
        - ARCH_*: Architecture variables (e.g., ARCH_NAME)
        - WH_*: Wormhole-specific variables (e.g., WH_ARCH_YAML)
        - TTNN_*: TTNN-specific variables (e.g., TTNN_CONFIG_OVERRIDES)

        You can also specify additional environment variables in the rank binding YAML using
        the `global_env` field (for all ranks) or `env_overrides` field (per-rank).

    \b
    Path Resolution (SLURM/sbatch compatibility):
        Relative paths for --rank-binding, --mock-cluster-rank-binding, and mesh_graph_desc_path
        are resolved by searching multiple locations in order:

        1. TT_METAL_HOME - If the environment variable is set (explicit user configuration).
        2. Launch directory - The directory where tt-run was originally invoked. This is
           captured at module load time and is critical for SLURM/sbatch environments where
           mpirun may change the working directory when spawning processes on remote nodes.
        3. Current working directory - Fallback to the current directory at resolution time.

        The first location where the file is found will be used. If the file is not found
        in any location, an error is raised listing all searched paths.

        This behavior ensures that commands like:
            tt-run --rank-binding tests/config/bindings.yaml ./my_app
        work correctly when launched from a tt-metal directory on an NFS mount, even when
        mpirun spawns processes on remote cluster nodes with different working directories.

        Use --verbose to see path resolution diagnostics.

    \b
    Multi-Host Mode (--multihost):
        The --multihost flag enables recommended MPI settings for multi-host cluster environments.
        This adds the following MPI arguments:

        - --mca btl self,tcp: Use TCP byte transfer layer for inter-node communication
        - --mca btl_tcp_if_exclude lo: Exclude loopback interface (can't route inter-node traffic)
        - --tag-output: Prefix output lines with rank information for easier debugging

        If --tcp-interface is specified (e.g., --tcp-interface cnx1), it uses btl_tcp_if_include
        instead of btl_tcp_if_exclude to explicitly select the network interface.

        These settings help avoid common MPI issues in multi-host environments:
        - Stale process connections from other nodes
        - Network interface selection problems
        - Output interleaving from multiple ranks

        Note: Only loopback (lo) is excluded by default. Loopback must be excluded because it
        can only communicate with the local machine, making it useless for inter-node MPI traffic.
        Other interfaces like docker0 are not excluded to remain compatible with Docker containers
        and other virtualized environments where network interface naming varies. For explicit
        interface control, use --tcp-interface.

        Example:
            tt-run --multihost --rank-binding config.yaml --mpi-args "--host nodeA,nodeB" ./my_app
            tt-run --tcp-interface cnx1 --rank-binding config.yaml --mpi-args "--rankfile hosts.txt" ./my_app

    \b
    Debugging with --debug-gdbserver:
        This flag launches each MPI rank under gdbserver for remote debugging, ideal for multi-machine setups.
        - Each rank runs 'gdbserver :PORT ./program args' where PORT = 20000 + rank.
        - It prints "Rank X on HOST listening on :PORT" and waits for attachment.
        - Use with --mpi-args for host mapping (e.g., "--host nodeA,nodeB").

        Prerequisites:
        - Build program with -g -O0 for debug symbols.
        - Passwordless SSH between workstation and nodes.
        - gdbserver installed on all nodes.

        Usage:
            tt-run --rank-binding binding.yaml --debug-gdbserver --mpi-args "--host nodeA,nodeB" ./program arg1 arg2

        Attachment Steps:
        1. Note host and port from output (e.g., Rank 0 on nodeA :20000).
        2. Set up SSH tunnels (one per rank):
           ssh -L 20000:localhost:20000 nodeA  # For rank 0
           ssh -L 20001:localhost:20001 nodeB  # For rank 1
        3. Attach locally (one gdb per rank):
           gdb ./program
           (gdb) target remote localhost:20000
           (gdb) break main
           (gdb) continue

        Tips:
        - Conditional breaks: break foo if atoi(getenv("OMPI_COMM_WORLD_RANK")) == 1
        - Non-stop: set non-stop on
        - Ignore signals: handle SIGPIPE nostop noprint pass
        - MPI issues: Add --mca pml ob1 --mca btl tcp,self to --mpi-args
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
    program = ctx.args

    # Log diagnostic information for path resolution debugging
    if verbose:
        logger.info(f"{TT_RUN_PREFIX} Path Resolution Diagnostics:")
        logger.info(f"{TT_RUN_PREFIX}   Original CWD (at launch): {ORIGINAL_CWD}")
        logger.info(f"{TT_RUN_PREFIX}   Current CWD: {Path.cwd()}")
        logger.info(f"{TT_RUN_PREFIX}   rank-binding input: {rank_binding}")
        logger.info(f"{TT_RUN_PREFIX}   TT_METAL_HOME env: {os.environ.get('TT_METAL_HOME', '<not set>')}")
        logger.info(f"{TT_RUN_PREFIX}   HOME env: {os.environ.get('HOME', '<not set>')}")
        logger.info(f"{TT_RUN_PREFIX}   PYTHONPATH env: {os.environ.get('PYTHONPATH', '<not set>')}")
        logger.info(f"{TT_RUN_PREFIX}   PYTHONHOME env: {os.environ.get('PYTHONHOME', '<not set>')}")
        logger.info(f"{TT_RUN_PREFIX}   LD_LIBRARY_PATH env: {os.environ.get('LD_LIBRARY_PATH', '<not set>')}")
        if os.environ.get("SLURM_JOB_ID"):
            logger.info(f"{TT_RUN_PREFIX}   SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            logger.info(f"{TT_RUN_PREFIX}   SLURM_SUBMIT_DIR: {os.environ.get('SLURM_SUBMIT_DIR', '<not set>')}")

    try:
        config = parse_binding_config(rank_binding, mock_cluster_rank_binding)
    except (ValueError, ValidationError) as e:
        raise click.ClickException(f"Configuration error: {e}")

    if verbose:
        logger.info(f"{TT_RUN_PREFIX}   Resolved mesh_graph_desc_path: {config.mesh_graph_desc_path}")

    if not program:
        raise click.ClickException("No program specified. Please provide a program to run.")

    # Validate program executable exists
    if not skip_executable_check:
        program_path = Path(program[0])
        if not program_path.exists() and not shutil.which(program[0]):
            raise click.ClickException(f"Program not found: {program[0]}")

    # Build multihost MPI args if requested
    # --tcp-interface implies --multihost
    if tcp_interface:
        multihost = True

    effective_mpi_args = list(mpi_args) if mpi_args else []

    if multihost:
        # Recommended MPI settings for multi-host clusters:
        # - Use TCP for byte transfer layer (reliable for multi-host)
        # - Exclude loopback (can't route inter-node traffic)
        # - Tag output with rank info for easier debugging
        # Note: Only exclude 'lo' to remain compatible with Docker containers
        # and other virtualized environments. Users needing specific interface
        # control should use --tcp-interface.
        multihost_args = [
            "--mca",
            "btl",
            "self,tcp",
            "--mca",
            "btl_tcp_if_exclude",
            "lo",
            "--tag-output",
        ]

        if tcp_interface:
            # If a specific interface is requested, use include instead of exclude
            multihost_args = [
                "--mca",
                "btl",
                "self,tcp",
                "--mca",
                "btl_tcp_if_include",
                tcp_interface,
                "--tag-output",
            ]

        # Prepend multihost args so user-provided --mpi-args can override if needed
        effective_mpi_args = multihost_args + effective_mpi_args

        if verbose:
            logger.info(f"{TT_RUN_PREFIX} Multihost mode enabled with args: {' '.join(multihost_args)}")

    # Build MPI command
    mpi_cmd = build_mpi_command(
        config, program, effective_mpi_args if effective_mpi_args else None, debug_gdbserver=debug_gdbserver
    )

    if verbose or dry_run:
        print_command(mpi_cmd)

    if dry_run:
        return

    if debug_gdbserver:
        logger.info(f"{TT_RUN_PREFIX} GDBServer mode: Each rank starts gdbserver on port 20000 + rank")
        logger.info(f"{TT_RUN_PREFIX} After ranks print their host and port, set up SSH tunnels:")
        logger.info(f"{TT_RUN_PREFIX} ssh -L <local_port>:localhost:<remote_port> <remote_host>")
        logger.info(f"{TT_RUN_PREFIX} Then locally run: gdb {program[0]}")
        logger.info(f"{TT_RUN_PREFIX} (gdb) target remote localhost:<local_port>")
        logger.info(f"{TT_RUN_PREFIX} (gdb) break main")
        logger.info(f"{TT_RUN_PREFIX} (gdb) continue")

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
