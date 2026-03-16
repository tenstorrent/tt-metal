#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""tt-run - MPI process launcher for TT-Metal and TTNN distributed applications."""

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Union

import click
import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

TT_RUN_PREFIX = "[tt-run]"
DEFAULT_LD_LIBRARY_PATH = "{home}/build/lib"
INTERRUPTED_EXIT_CODE = 130  # 128 + SIGINT
PRETTY_PRINT_THRESHOLD = 10  # Minimum args to trigger multi-line formatting

# Store the original working directory at module load time to preserve it
# across mpirun process launches (critical for SLURM/sbatch environments)
ORIGINAL_CWD = Path.cwd().resolve()


class RankfileSyntax(Enum):
    """MPI rankfile syntax variants for different mpirun versions."""

    MAP_BY_RANKFILE_FILE = "map_by_rankfile_file"  # --map-by rankfile:file=<path> (OpenMPI 5.x / PRRTE)
    RANKFILE = "rankfile"  # --rankfile <path> (older OpenMPI)
    MCA_RMAPS_RANKFILE_PATH = "mca_rmaps_rankfile_path"  # -mca rmaps_rankfile_path <path> (fallback)


def get_mpi_launcher() -> str:
    """Get the MPI launcher executable name.

    Returns:
        'mpirun-ulfm' if available, otherwise 'mpirun'. In SLURM interactive sessions, always returns 'mpirun'.
    """
    # Check if running in SLURM interactive session
    if os.environ.get("SLURM_JOB_ID") is not None and os.environ.get("SLURM_STEP_ID") is not None:
        logger.warning(f"{TT_RUN_PREFIX} SLURM interactive session detected, using mpirun")
        return "mpirun"

    # Find mpirun-ulfm executable, fall back to mpirun if not found
    mpi_launcher = shutil.which("mpirun-ulfm")
    if not mpi_launcher:
        logger.warning(f"{TT_RUN_PREFIX} mpirun-ulfm not found in PATH, falling back to mpirun")
        return "mpirun"

    return mpi_launcher


def build_rankfile_args(syntax: RankfileSyntax, rankfile: Path) -> List[str]:
    """Build MPI command-line arguments for rankfile based on syntax variant.

    This is a pure function that constructs the appropriate MPI arguments for a given
    rankfile syntax variant. It does not perform any I/O or subprocess calls.

    Args:
        syntax: The rankfile syntax variant to use
        rankfile: Path to the rankfile

    Returns:
        List of MPI command-line arguments (e.g., ["--map-by", "rankfile:file=/path/to/rankfile"])

    Examples:
        >>> build_rankfile_args(RankfileSyntax.MAP_BY_RANKFILE_FILE, Path("/tmp/rankfile"))
        ['--map-by', 'rankfile:file=/tmp/rankfile']
        >>> build_rankfile_args(RankfileSyntax.RANKFILE, Path("/tmp/rankfile"))
        ['--rankfile', '/tmp/rankfile']
    """
    rankfile_str = str(rankfile.resolve())

    if syntax == RankfileSyntax.MAP_BY_RANKFILE_FILE:
        return ["--map-by", f"rankfile:file={rankfile_str}"]
    elif syntax == RankfileSyntax.RANKFILE:
        return ["--rankfile", rankfile_str]
    elif syntax == RankfileSyntax.MCA_RMAPS_RANKFILE_PATH:
        return ["--mca", "rmaps_rankfile_path", rankfile_str]
    else:
        raise ValueError(f"Unknown rankfile syntax: {syntax}")


def detect_rankfile_syntax(mpi_launcher: str, subprocess_run=subprocess.run) -> RankfileSyntax:
    """Detect which rankfile syntax variant the MPI launcher supports.

    Runs `{mpi_launcher} --help` and parses the output to determine which rankfile
    syntax is supported. Prefers `--map-by rankfile:file=` (OpenMPI 5.x), then
    `--rankfile` (older OpenMPI), then falls back to MCA parameter.

    Args:
        mpi_launcher: Path or name of MPI launcher (e.g., "mpirun", "mpirun-ulfm")
        subprocess_run: Subprocess run function (injectable for testing)

    Returns:
        RankfileSyntax enum indicating which syntax to use
    """
    try:
        result = subprocess_run(
            [mpi_launcher, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        help_text = result.stdout + result.stderr

        # Check for --map-by rankfile:file= syntax (OpenMPI 5.x / PRRTE)
        if "--map-by" in help_text and "rankfile" in help_text.lower():
            # Look for rankfile:file= pattern in help text
            if "rankfile:file" in help_text or "rankfile:file=" in help_text:
                return RankfileSyntax.MAP_BY_RANKFILE_FILE

        # Check for --rankfile option (older OpenMPI)
        if "--rankfile" in help_text:
            return RankfileSyntax.RANKFILE

        # Fallback to MCA parameter
        return RankfileSyntax.MCA_RMAPS_RANKFILE_PATH

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning(
            f"{TT_RUN_PREFIX} Failed to detect rankfile syntax from {mpi_launcher}: {e}. "
            f"Falling back to MCA parameter syntax."
        )
        return RankfileSyntax.MCA_RMAPS_RANKFILE_PATH


def inject_rankfile_mpi_args(
    rankfile: Path,
    base_mpi_args: List[str],
    mpi_launcher: str,
    detect_fn=detect_rankfile_syntax,
) -> List[str]:
    """Inject rankfile MPI arguments into base MPI args.

    Auto-detects the correct rankfile syntax for the MPI launcher and prepends
    the appropriate arguments to base_mpi_args.

    Args:
        rankfile: Path to the rankfile
        base_mpi_args: Existing MPI arguments to prepend to
        mpi_launcher: MPI launcher executable name/path (for detection)
        detect_fn: Function to detect rankfile syntax (injectable for testing)

    Returns:
        New list with rankfile args prepended: [rankfile_args..., ...base_mpi_args]
    """
    syntax = detect_fn(mpi_launcher)
    rankfile_args = build_rankfile_args(syntax, rankfile)
    return rankfile_args + base_mpi_args


def find_generate_rank_bindings_executable() -> Path:
    """Find the generate_rank_bindings executable.

    Searches for the executable in standard locations:
    1. TT_METAL_HOME/build/tools/scaleout/generate_rank_bindings
    2. Current directory relative paths

    Returns:
        Path to the generate_rank_bindings executable

    Raises:
        FileNotFoundError: If the executable cannot be found
    """
    # Try TT_METAL_HOME first
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if tt_metal_home:
        candidate = Path(tt_metal_home) / "build" / "tools" / "scaleout" / "generate_rank_bindings"
        if candidate.exists():
            return candidate.resolve()

    # Try relative to ORIGINAL_CWD
    candidate = ORIGINAL_CWD / "build" / "tools" / "scaleout" / "generate_rank_bindings"
    if candidate.exists():
        return candidate.resolve()

    # Try current directory
    candidate = Path("build") / "tools" / "scaleout" / "generate_rank_bindings"
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"generate_rank_bindings executable not found. "
        f"Searched: TT_METAL_HOME/build/tools/scaleout/generate_rank_bindings, "
        f"{ORIGINAL_CWD}/build/tools/scaleout/generate_rank_bindings, "
        f"build/tools/scaleout/generate_rank_bindings"
    )


def get_generate_rank_bindings_output_paths(output_dir: Path) -> tuple[Path, Path]:
    """Get the output paths for generate_rank_bindings.

    This is a pure function that returns the expected output paths.

    Args:
        output_dir: Base output directory (typically tt-run-generated/)

    Returns:
        Tuple of (rank_bindings.yaml path, rankfile path)
    """
    rank_bindings_path = output_dir / "rank_bindings.yaml"
    rankfile_path = output_dir / "rankfile"
    return (rank_bindings_path, rankfile_path)


def parse_rankfile(rankfile_path: Path) -> Dict[int, str]:
    """Parse OpenMPI rankfile to extract rank -> hostname mapping.

    Args:
        rankfile_path: Path to rankfile

    Returns:
        Dictionary mapping rank (int) -> hostname (str)
    """
    rank_to_host: Dict[int, str] = {}
    with open(rankfile_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse format: rank N=hostname slot=X
            # Match: rank <number>=<hostname> slot=<number>
            match = re.match(r"rank\s+(\d+)=([^\s]+)\s+slot=\d+", line)
            if match:
                rank = int(match.group(1))
                hostname = match.group(2)
                rank_to_host[rank] = hostname
    return rank_to_host


def rankfile_needs_oversubscribe(rankfile_path: Path) -> bool:
    """Check if rankfile requires --oversubscribe (multiple ranks per host).

    Args:
        rankfile_path: Path to rankfile

    Returns:
        True if any host has more than one rank assigned (requires oversubscription)
    """
    rank_to_host = parse_rankfile(rankfile_path)
    if not rank_to_host:
        return False

    # Count ranks per host
    host_to_rank_count: Dict[str, int] = {}
    for hostname in rank_to_host.values():
        host_to_rank_count[hostname] = host_to_rank_count.get(hostname, 0) + 1

    # Check if any host has more than one rank
    return any(count > 1 for count in host_to_rank_count.values())


def build_phase2_mock_mapping(
    rankfile_path: Path,
    phase1_hosts: Optional[List[str]],
    phase1_mock_rank_to_desc: Optional[Dict[int, Path]],
) -> Optional[Dict[int, Path]]:
    """Build Phase 2 mock cluster descriptor mapping from Phase 1 mapping.

    Maps Phase 2 ranks (from generated rank_bindings.yaml) to Phase 1 mock descriptors
    based on host assignment. If Phase 1 used hosts, Phase 2 ranks on the same host
    get the Phase 1 rank's descriptor for that host. If Phase 1 used mock (all localhost),
    Phase 2 ranks map directly to Phase 1 ranks.

    Args:
        rankfile_path: Path to generated rankfile (maps Phase 2 ranks to hosts)
        phase1_hosts: List of hosts used in Phase 1 (None if mock cluster)
        phase1_mock_rank_to_desc: Phase 1 mock descriptor mapping (rank -> path)

    Returns:
        Dictionary mapping Phase 2 rank -> mock descriptor path, or None if no mock cluster
    """
    if not phase1_mock_rank_to_desc:
        return None

    # Parse rankfile to get Phase 2 rank -> host mapping
    phase2_rank_to_host = parse_rankfile(rankfile_path)

    # Build Phase 2 rank -> Phase 1 rank -> mock descriptor mapping
    phase2_mock_mapping: Dict[int, Path] = {}

    if phase1_hosts:
        # Phase 1 used hosts: map by host (first rank on each host gets that host's Phase 1 rank's descriptor)
        # Build host -> Phase 1 rank mapping (first rank on each host)
        host_to_phase1_rank: Dict[str, int] = {}
        for phase1_rank, host in enumerate(phase1_hosts):
            if host not in host_to_phase1_rank:
                host_to_phase1_rank[host] = phase1_rank

        # Map Phase 2 ranks to Phase 1 ranks by host
        for phase2_rank, hostname in phase2_rank_to_host.items():
            if hostname in host_to_phase1_rank:
                phase1_rank = host_to_phase1_rank[hostname]
                if phase1_rank in phase1_mock_rank_to_desc:
                    phase2_mock_mapping[phase2_rank] = phase1_mock_rank_to_desc[phase1_rank]
            else:
                # Host not in Phase 1 hosts - shouldn't happen, but fallback to rank 0
                logger.warning(
                    f"{TT_RUN_PREFIX} Phase 2 rank {phase2_rank} on host {hostname} "
                    f"not found in Phase 1 hosts. Using Phase 1 rank 0 mock descriptor."
                )
                if 0 in phase1_mock_rank_to_desc:
                    phase2_mock_mapping[phase2_rank] = phase1_mock_rank_to_desc[0]
    else:
        # Phase 1 used mock (all localhost): map Phase 2 ranks directly to Phase 1 ranks
        # This assumes Phase 2 ranks are in the same order as Phase 1 ranks
        for phase2_rank in sorted(phase2_rank_to_host.keys()):
            # Map Phase 2 rank to Phase 1 rank directly (modulo if Phase 2 has more ranks)
            phase1_rank = phase2_rank % len(phase1_mock_rank_to_desc)
            if phase1_rank in phase1_mock_rank_to_desc:
                phase2_mock_mapping[phase2_rank] = phase1_mock_rank_to_desc[phase1_rank]

    return phase2_mock_mapping


def build_generate_rank_bindings_mpi_cmd(
    executable: Path,
    mgd_path: Path,
    hosts: Optional[List[str]],
    output_dir: Path,
    mock_rank_to_desc: Optional[Dict[int, Path]] = None,
    mpi_args: Optional[List[str]] = None,
) -> List[str]:
    """Build MPI command for running generate_rank_bindings.

    This is a pure function that constructs the MPI command without executing it.

    Args:
        executable: Path to generate_rank_bindings executable
        mgd_path: Path to mesh graph descriptor
        hosts: List of hostnames (for real cluster) or None (for mock)
        output_dir: Output directory for generated files
        mock_rank_to_desc: Optional dict mapping rank -> mock cluster descriptor path
        mpi_args: Optional list of additional MPI arguments (e.g., ["--allow-run-as-root"])

    Returns:
        List of command-line arguments for mpirun

    Raises:
        ValueError: If neither hosts nor mock_rank_to_desc is provided
    """
    mpi_launcher = get_mpi_launcher()
    cmd = [mpi_launcher]

    # Always enable tagged output for easier debugging (prefixes output with rank info)
    cmd.extend(["--tag-output"])

    # Add user-provided MPI args (e.g., --allow-run-as-root for Docker containers)
    if mpi_args:
        cmd.extend(mpi_args)

    if mock_rank_to_desc:
        # Mock cluster: all processes on localhost
        # Use per-rank -np 1 segments to set per-rank env vars (similar to legacy_flow)
        # Use --oversubscribe to allow more processes than available slots (needed for mock clusters)
        cmd.extend(["--oversubscribe"])
        # Don't specify --host for mock clusters - MPI will default to localhost
        # This avoids "All nodes which are allocated for this job are already filled" errors

        # Build per-rank segments with : separator
        for i, rank in enumerate(sorted(mock_rank_to_desc.keys())):
            if i > 0:
                cmd.append(":")
            desc_path = mock_rank_to_desc[rank]
            cmd.extend(["-np", "1"])
            cmd.extend(["-x", f"TT_METAL_MOCK_CLUSTER_DESC_PATH={desc_path.resolve()}"])
            cmd.append(str(executable.resolve()))
            cmd.extend(["--mesh-graph-descriptor", str(mgd_path.resolve())])
            # Note: generate_rank_bindings doesn't accept --output-dir, it hardcodes output to "tt-run-generated/"

        # Return early for mock mode (already added executable and args per rank)
        return cmd

    elif hosts:
        # Real cluster: one process per host
        np = len(hosts)
        hosts_str = ",".join(hosts)
        cmd.extend(["--host", hosts_str])
        cmd.extend(["-np", str(np)])
        cmd.append(str(executable.resolve()))
        cmd.extend(["--mesh-graph-descriptor", str(mgd_path.resolve())])
        # Note: generate_rank_bindings doesn't accept --output-dir, it hardcodes output to "tt-run-generated/"
    else:
        raise ValueError("Either hosts or mock_rank_to_desc must be provided")

    return cmd


def run_generate_rank_bindings(cmd: List[str], cwd: Path, subprocess_run=subprocess.run) -> int:
    """Run generate_rank_bindings command via subprocess.

    Args:
        cmd: Command to run (from build_generate_rank_bindings_mpi_cmd)
        cwd: Working directory for the command
        subprocess_run: Subprocess run function (injectable for testing)

    Returns:
        Exit code from the subprocess
    """
    result = subprocess_run(cmd, cwd=cwd)
    return result.returncode if hasattr(result, "returncode") else result


def run_phase1_generate_rank_bindings(
    mgd_path: Path,
    hosts: Optional[List[str]],
    output_dir: Path,
    subprocess_run=subprocess.run,
    sleep_secs: int = 5,
    mock_rank_to_desc: Optional[Dict[int, Path]] = None,
    mpi_args: Optional[List[str]] = None,
) -> tuple[Path, Path]:
    """Run Phase 1: generate_rank_bindings to produce rank_bindings.yaml and rankfile.

    Orchestrates the Phase 1 MPI call, waits for file sync, and validates outputs.

    Args:
        mgd_path: Path to mesh graph descriptor
        hosts: List of hostnames (for real cluster) or None (for mock)
        output_dir: Output directory (typically tt-run-generated/)
        subprocess_run: Subprocess run function (injectable for testing)
        sleep_secs: Seconds to sleep after Phase 1 for file sync (default 5)
        mock_rank_to_desc: Optional dict mapping rank -> mock cluster descriptor path
        mpi_args: Optional list of additional MPI arguments (e.g., ["--allow-run-as-root"])

    Returns:
        Tuple of (rank_bindings.yaml path, rankfile path)

    Raises:
        FileNotFoundError: If generate_rank_bindings executable not found
        RuntimeError: If Phase 1 fails or outputs are missing
    """
    executable = find_generate_rank_bindings_executable()
    cmd = build_generate_rank_bindings_mpi_cmd(executable, mgd_path, hosts, output_dir, mock_rank_to_desc, mpi_args)

    logger.info(f"{TT_RUN_PREFIX} Phase 1: Running generate_rank_bindings...")
    logger.debug(f"{TT_RUN_PREFIX} Phase 1 command: {' '.join(cmd)}")

    exit_code = run_generate_rank_bindings(cmd, cwd=ORIGINAL_CWD, subprocess_run=subprocess_run)

    if exit_code != 0:
        raise RuntimeError(f"generate_rank_bindings failed with exit code {exit_code}. " f"Command: {' '.join(cmd)}")

    # Wait for file sync (NFS, shared storage)
    if sleep_secs > 0:
        logger.info(f"{TT_RUN_PREFIX} Waiting {sleep_secs} seconds for file sync...")
        time.sleep(sleep_secs)

    # Validate outputs exist
    rank_bindings_path, rankfile_path = get_generate_rank_bindings_output_paths(output_dir)

    if not rank_bindings_path.exists():
        raise RuntimeError(
            f"Phase 1 output not found: {rank_bindings_path}. " f"generate_rank_bindings may have failed silently."
        )

    if not rankfile_path.exists():
        raise RuntimeError(
            f"Phase 1 output not found: {rankfile_path}. " f"generate_rank_bindings may have failed silently."
        )

    logger.info(f"{TT_RUN_PREFIX} Phase 1 complete. Generated: {rank_bindings_path}, {rankfile_path}")

    return (rank_bindings_path, rankfile_path)


def get_local_network_interfaces() -> List[str]:
    """Get list of network interface names on the local host.

    Returns:
        List of interface names (e.g., ['lo', 'eth0', 'cnx1'])
    """
    try:
        # /sys/class/net contains symlinks to all network interfaces
        net_path = Path("/sys/class/net")
        if net_path.exists():
            return [p.name for p in net_path.iterdir()]
    except (OSError, PermissionError) as exc:
        # Best-effort enumeration: on non-Linux or restricted environments, fall back to empty list
        logger.debug(f"{TT_RUN_PREFIX} Failed to enumerate network interfaces: {exc}")
    return []


def validate_network_interface(interface: str, verbose: bool = False) -> None:
    """Warn if the specified network interface doesn't exist on the local host.

    This is a best-effort check - we can only validate the local host, not remote
    MPI hosts. The warning helps catch typos and misconfiguration early.

    Args:
        interface: Network interface name to validate (e.g., 'eth0', 'cnx1')
        verbose: If True, log additional diagnostic information
    """
    local_interfaces = get_local_network_interfaces()

    if not local_interfaces:
        # Can't determine interfaces (non-Linux or permission issue), skip check
        if verbose:
            logger.debug(f"{TT_RUN_PREFIX} Unable to enumerate network interfaces, skipping validation")
        return

    if interface not in local_interfaces:
        logger.warning(
            f"{TT_RUN_PREFIX} Network interface '{interface}' not found on local host. "
            f"Available interfaces: {', '.join(sorted(local_interfaces))}. "
            f"Note: This check only validates the local host; the interface may exist on remote MPI hosts."
        )
    elif verbose:
        logger.info(f"{TT_RUN_PREFIX} Network interface '{interface}' found on local host")


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
    mesh_graph_desc_path: Path = Field(..., description="Path to mesh graph descriptor")
    mock_cluster_rank_binding: Dict[int, Path] = Field(
        default_factory=dict, description="Mock cluster rank binding configuration (rank -> resolved path)"
    )

    model_config = {"arbitrary_types_allowed": True}

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

    @field_validator("mesh_graph_desc_path", mode="before")
    def validate_mesh_graph_exists(cls, path: Union[str, Path]) -> Path:
        """Ensure mesh graph descriptor file exists.

        Uses resolve_path() to search multiple locations for relative paths.
        """
        return resolve_path(path, description="Mesh graph descriptor", must_be_file=True)


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

    # Track TT_METAL_HOME for fallback warning
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    tt_metal_home_checked = False

    for base_path in search_paths:
        if base_path is None:
            continue
        candidate = (base_path / expanded_path).resolve()
        if check_fn(candidate):
            # Warn if TT_METAL_HOME was set but we found the file elsewhere (fallback occurred)
            if tt_metal_home and tt_metal_home_checked and str(base_path) != tt_metal_home:
                logger.debug(
                    f"{TT_RUN_PREFIX} {description} not found in TT_METAL_HOME ({tt_metal_home}), "
                    f"using fallback location: {candidate}"
                )
            else:
                logger.debug(f"{TT_RUN_PREFIX} Resolved {description}: {path} -> {candidate}")
            return candidate
        # Track if we checked TT_METAL_HOME
        if tt_metal_home and str(base_path) == str(Path(tt_metal_home).expanduser()):
            tt_metal_home_checked = True

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
        resolved_mock_bindings: Dict[int, Path] = {}

        # Check if this is a mapping file (has rank_to_cluster_mock_cluster_desc key) or a cluster descriptor file
        # Check the key in the YAML data, not the filename
        if mock_data and isinstance(mock_data, dict) and "rank_to_cluster_mock_cluster_desc" in mock_data:
            # Mapping format: rank -> cluster descriptor path
            for rank_str, path in mock_data["rank_to_cluster_mock_cluster_desc"].items():
                # Convert rank to int (YAML may parse numeric keys as strings)
                rank = int(rank_str)
                resolved_path = resolve_path(
                    path, description=f"Mock cluster descriptor for rank {rank}", must_be_file=True
                )
                resolved_mock_bindings[rank] = resolved_path
        else:
            # Cluster descriptor format: treat as single entry with rank 0
            resolved_mock_bindings[0] = resolved_mock_path

        config.mock_cluster_rank_binding = resolved_mock_bindings

    return config


# Environment variable prefixes that should be automatically passed through to MPI processes
ENV_PASSTHROUGH_PREFIXES = (
    "TT_",  # TT-Metal/TTNN variables
    "ARCH_",  # Architecture variables (e.g., ARCH_NAME)
    "WH_",  # Wormhole-specific variables (e.g., WH_ARCH_YAML)
    "TTNN_",  # TTNN-specific variables (e.g., TTNN_CONFIG_OVERRIDES)
    "DEBUG_",  # Generic debug toggles used by tests/demos
    "DEEPSEEK_",  # DeepSeek model vars (e.g., DEEPSEEK_V3_HF_MODEL, DEEPSEEK_V3_CACHE)
    "MESH_",  # Mesh config (e.g., MESH_DEVICE)
)

# Environment variables that should NOT be passed through even if they match ENV_PASSTHROUGH_PREFIXES.
# These are either:
# 1. Explicitly managed by tt-run and derived from rank bindings (not parent environment)
# 2. Should only be set via rank binding env_overrides (e.g., TT_VISIBLE_DEVICES)
#
# TT_VISIBLE_DEVICES: Controls which PCIe devices are visible to a process. This must be set
# per-rank via env_overrides in rank bindings to ensure each MPI process sees only its assigned
# devices. Passing through from the parent environment would override per-rank device assignments
# configured by cluster descriptors and rank bindings, causing incorrect device visibility.
# See: tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md Section 5.2
#      scripts/scaleout/README_generate_cluster_descriptors.md
#
# Note: TT_METAL_HOME, TT_METAL_RUNTIME_ROOT, and TT_METAL_CACHE are NOT blocklisted because
# they are read from the parent environment (with fallbacks) and should be passed through to
# support NFS-based distributed workloads where all MPI ranks share the same python_venv.
ENV_BLOCKLIST = frozenset(
    {
        # Managed by tt-run - values derived from rank bindings, not parent environment
        "TT_MESH_ID",  # Mesh identifier from rank binding
        "TT_MESH_HOST_RANK",  # Host rank within mesh from rank binding
        "TT_MESH_GRAPH_DESC_PATH",  # Path to mesh graph descriptor from config
        "TT_RUN_ORIGINAL_CWD",  # Always set to ORIGINAL_CWD by tt-run
        "TT_METAL_MOCK_CLUSTER_DESC_PATH",  # Mock cluster path for testing
        # Should only come from rank binding env_overrides
        "TT_VISIBLE_DEVICES",  # Per-rank device visibility - must be set via rank bindings
    }
)


def get_rank_environment(binding: RankBinding, config: TTRunConfig) -> Dict[str, str]:
    """Get all environment variables for a specific rank.

    Args:
        binding: Rank binding configuration
        config: Global configuration

    Returns:
        Dictionary of environment variables for this rank
    """
    # Start with automatic pass-through of TT-related environment variables
    # This ensures variables like ARCH_NAME, WH_ARCH_YAML, TTNN_CONFIG_OVERRIDES are propagated
    # Variables in ENV_BLOCKLIST are excluded even if they match prefixes
    env = {}
    passthrough_vars = []
    blocked_vars = []
    for key, value in os.environ.items():
        if key.startswith(ENV_PASSTHROUGH_PREFIXES):
            if key in ENV_BLOCKLIST:
                blocked_vars.append(key)
            else:
                env[key] = value
                passthrough_vars.append(key)

    if passthrough_vars:
        logger.debug(
            f"{TT_RUN_PREFIX} Auto-propagating {len(passthrough_vars)} environment variables "
            f"with prefixes {ENV_PASSTHROUGH_PREFIXES}: {', '.join(sorted(passthrough_vars))}"
        )

    if blocked_vars:
        logger.debug(
            f"{TT_RUN_PREFIX} Blocked {len(blocked_vars)} environment variables from pass-through "
            f"(managed by tt-run or rank bindings): {', '.join(sorted(blocked_vars))}"
        )

    # Use ORIGINAL_CWD as the default for TT_METAL_HOME when not explicitly set.
    # This assumes the launch directory is on a shared filesystem (NFS) visible to all nodes.
    default_tt_metal_home = os.environ.get("TT_METAL_HOME", str(ORIGINAL_CWD))

    # Set/override core tt-run managed variables
    # Note: Path objects are converted to str here at the env var boundary
    env.update(
        {
            "TT_MESH_ID": str(binding.mesh_id),
            "TT_MESH_GRAPH_DESC_PATH": str(config.mesh_graph_desc_path),
            "TT_METAL_HOME": default_tt_metal_home,
            "TT_METAL_RUNTIME_ROOT": os.environ.get("TT_METAL_RUNTIME_ROOT", default_tt_metal_home),
            "PYTHONPATH": os.environ.get("PYTHONPATH", str(ORIGINAL_CWD)),
            # 26640: TODO - Investigate why this needs to be set for multi-host CI environments
            "LD_LIBRARY_PATH": os.environ.get(
                "LD_LIBRARY_PATH", DEFAULT_LD_LIBRARY_PATH.format(home=str(ORIGINAL_CWD))
            ),
            # Pass the original CWD to subprocesses so they can resolve relative paths correctly
            "TT_RUN_ORIGINAL_CWD": str(ORIGINAL_CWD),
        }
    )

    # Pass critical shell/user environment variables.
    # HOME and USER are required by OpenMPI for process management and state files.
    # PATH enables finding executables (e.g., pytest in venv).
    # VIRTUAL_ENV enables venv-aware execution on remote hosts.
    for var in ("HOME", "USER", "PATH", "VIRTUAL_ENV"):
        if os.environ.get(var):
            env[var] = os.environ[var]

    # PYTHONHOME: Only pass through if explicitly set. Do NOT default to ORIGINAL_CWD.
    # Setting PYTHONHOME incorrectly causes Python to look for its standard library
    # in the wrong location, resulting in "ModuleNotFoundError: No module named 'encodings'".
    # When using a virtualenv, PYTHONHOME should not be set - Python determines the
    # correct paths from the executable location.
    if os.environ.get("PYTHONHOME"):
        env["PYTHONHOME"] = os.environ["PYTHONHOME"]

    # Add TT_MESH_HOST_RANK only if mesh_host_rank is set
    if binding.mesh_host_rank is not None:
        env["TT_MESH_HOST_RANK"] = str(binding.mesh_host_rank)

    if config.mock_cluster_rank_binding:
        env["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(config.mock_cluster_rank_binding[binding.rank])

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
    mpi_launcher = get_mpi_launcher()
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

    # Always enable tagged output for easier debugging (prefixes output with rank info)
    cmd.extend(["--tag-output"])

    if mpi_args:
        cmd.extend(mpi_args)

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


def legacy_flow(
    ctx: click.Context,
    rank_binding: Path,
    dry_run: bool,
    verbose: bool,
    mpi_args: Optional[List[str]],
    debug_gdbserver: bool,
    mock_cluster_rank_binding: Optional[Path],
    skip_executable_check: bool,
    bare: bool,
    tcp_interface: Optional[str],
    rankfile: Optional[Path] = None,
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

        # Multi-host with rankfile (multihost MPI settings are default)
        tt-run --rank-binding binding.yaml --mpi-args "--rankfile hosts.txt" ./my_app

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
        - TT_METAL_HOME: TT-Metal installation directory
        - PYTHONPATH: Python module search path
        - LD_LIBRARY_PATH: Library search path
        - TT_MESH_GRAPH_DESC_PATH: Path to mesh graph descriptor
        - TT_RUN_ORIGINAL_CWD: Directory where tt-run was launched (for subprocess path resolution)
        - HOME: Passed through (required by OpenMPI for process management)
        - USER: Passed through (required by OpenMPI for process identity)
        - PATH: Passed through from caller (enables venv tools like pytest on remote hosts)
        - VIRTUAL_ENV: Passed through from caller (enables venv-aware execution)
        - PYTHONHOME: Passed through only if explicitly set (do not set when using virtualenvs)

        Default values for the following environment variables will be used if not set when calling tt-run:
        - TT_METAL_HOME: Launch directory (where tt-run was invoked)
        - TT_METAL_RUNTIME_ROOT: Same as TT_METAL_HOME
        - PYTHONPATH: Launch directory
        - LD_LIBRARY_PATH: `<LAUNCH_DIR>/build/lib`

        This assumes the launch directory is on a shared filesystem (e.g., NFS) visible to all
        cluster nodes, which is the common setup for SLURM environments.

        Additionally, all environment variables with the following prefixes are automatically
        passed through to MPI processes:
        - TT_*: TT-Metal/TTNN variables
        - ARCH_*: Architecture variables (e.g., ARCH_NAME)
        - WH_*: Wormhole-specific variables (e.g., WH_ARCH_YAML)
        - TTNN_*: TTNN-specific variables (e.g., TTNN_CONFIG_OVERRIDES)

        Exception: The following TT_* variables are BLOCKED from automatic pass-through because
        they are managed by tt-run or should only be set via rank binding env_overrides:
        - TT_VISIBLE_DEVICES: Must be set per-rank via env_overrides in rank bindings to ensure
          correct device visibility. Cluster descriptors and rank bindings configure this per-rank.
        - TT_MESH_ID, TT_MESH_HOST_RANK, TT_MESH_GRAPH_DESC_PATH: Derived from rank bindings/config
        - TT_RUN_ORIGINAL_CWD, TT_METAL_MOCK_CLUSTER_DESC_PATH: Set by tt-run internally

        Note: TT_METAL_HOME, TT_METAL_RUNTIME_ROOT, and TT_METAL_CACHE ARE passed through from
        the parent environment to support NFS-based distributed workloads where all MPI ranks
        share the same python_venv from the launch directory.

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
    Tagged Output:
        tt-run always enables --tag-output, which prefixes each output line with rank
        information (e.g., [1,0]<stdout>:). This makes it easier to identify which rank
        produced each line of output when debugging distributed applications.

    \b
    Multi-Host MPI Settings (default):
        tt-run applies recommended MPI settings for multi-host clusters by default:

        - --mca btl self,tcp: Use TCP byte transfer layer for inter-node communication
        - --mca btl_tcp_if_exclude docker0,lo: Exclude Docker bridge and loopback interfaces

        If --tcp-interface is specified (e.g., --tcp-interface cnx1), it uses btl_tcp_if_include
        instead to explicitly select the network interface.

        Use --bare to disable these settings (e.g., single-host or special setups).

        These settings help avoid common MPI issues in multi-host environments:
        - Stale process connections from other nodes
        - Network interface selection problems (docker0, lo can't route inter-node traffic)

        Example:
            tt-run --rank-binding config.yaml --mpi-args "--host nodeA,nodeB" ./my_app
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

    # Apply default multihost MPI args unless --bare
    if tcp_interface and not bare:
        # Validate the interface exists on the local host (best-effort check)
        validate_network_interface(tcp_interface, verbose=verbose)

    effective_mpi_args = list(mpi_args) if mpi_args else []

    if not bare:
        # Recommended MPI settings for multi-host clusters:
        # - Use TCP for byte transfer layer (reliable for multi-host)
        # - Exclude loopback and docker0 (can't route inter-node traffic)
        # Note: Exclude both 'lo' (loopback) and 'docker0' (Docker bridge) by default.
        # These interfaces cannot route traffic between hosts and can cause MPI
        # process discovery issues if selected. For specific interface control,
        # use --tcp-interface.
        multihost_args = [
            "--mca",
            "btl",
            "self,tcp",
            "--mca",
            "btl_tcp_if_exclude",
            "docker0,lo",
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
            ]

        # Prepend multihost args so user-provided --mpi-args can override if needed
        effective_mpi_args = multihost_args + effective_mpi_args

        if verbose:
            logger.info(f"{TT_RUN_PREFIX} Using multihost MPI args: {' '.join(multihost_args)}")

    # Inject rankfile args if provided (auto-detect MPI syntax)
    # This happens after multihost args so rankfile comes right before user args
    # Check if user already specified rankfile in mpi_args to avoid conflicts
    if rankfile:
        # Check for existing rankfile-related args in user's mpi_args
        rankfile_keywords = ["--rankfile", "--map-by", "rankfile:file=", "rmaps_rankfile_path"]
        has_existing_rankfile = False
        if mpi_args:
            mpi_args_str = " ".join(mpi_args)
            for keyword in rankfile_keywords:
                if keyword in mpi_args_str:
                    has_existing_rankfile = True
                    break

        if has_existing_rankfile:
            logger.warning(
                f"{TT_RUN_PREFIX} Rankfile argument already present in --mpi-args. "
                f"Skipping rankfile injection from parameter. "
                f"To use the rankfile parameter, remove rankfile-related args from --mpi-args."
            )
        else:
            mpi_launcher = get_mpi_launcher()
            # Detect rankfile syntax once
            rankfile_syntax = detect_rankfile_syntax(mpi_launcher)
            rankfile_args = build_rankfile_args(rankfile_syntax, rankfile)
            effective_mpi_args = rankfile_args + effective_mpi_args
            if verbose:
                logger.info(f"{TT_RUN_PREFIX} Injected rankfile: {rankfile}")

            # Check if rankfile requires oversubscription (multiple ranks per host)
            # Add --oversubscribe if needed and not already present
            if rankfile_needs_oversubscribe(rankfile):
                has_oversubscribe = False
                if effective_mpi_args:
                    has_oversubscribe = "--oversubscribe" in effective_mpi_args
                if not has_oversubscribe:
                    # Insert --oversubscribe right after rankfile args (before other args)
                    rankfile_args_len = len(rankfile_args)
                    effective_mpi_args = (
                        effective_mpi_args[:rankfile_args_len]
                        + ["--oversubscribe"]
                        + effective_mpi_args[rankfile_args_len:]
                    )
                    if verbose:
                        logger.info(f"{TT_RUN_PREFIX} Added --oversubscribe (rankfile has multiple ranks per host)")

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


def new_mode_flow(
    ctx: click.Context,
    mesh_graph_descriptor: Path,
    hosts: Optional[List[str]],
    dry_run: bool,
    verbose: bool,
    mpi_args: Optional[List[str]],
    debug_gdbserver: bool,
    mock_cluster_rank_binding: Optional[Path],
    skip_executable_check: bool,
    bare: bool,
    tcp_interface: Optional[str],
) -> None:
    """New mode flow for ttrun using mesh graph descriptor.

    This function implements the new mode of ttrun that uses --mesh-graph-descriptor
    instead of --rank-binding. It runs generate_rank_bindings (Phase 1) to produce
    rank_bindings.yaml and rankfile, then calls legacy_flow (Phase 2) with those files.

    Args:
        ctx: Click context
        mesh_graph_descriptor: Path to mesh graph descriptor file
        hosts: List of hostnames (required unless mock_cluster_rank_binding is provided)
        dry_run: If True, print command without executing
        verbose: If True, show detailed diagnostics
        mpi_args: Additional MPI arguments
        debug_gdbserver: If True, launch with gdbserver for debugging
        mock_cluster_rank_binding: Optional mock cluster rank binding configuration
        skip_executable_check: If True, skip program executable validation
        bare: If True, disable tt-run defaults
        tcp_interface: Network interface for MPI TCP communication
    """
    program = ctx.args

    if not program:
        raise click.ClickException("No program specified. Please provide a program to run.")

    # Resolve mesh_graph_descriptor path
    resolved_mgd = resolve_path(mesh_graph_descriptor, description="Mesh graph descriptor", must_be_file=True)

    if verbose:
        logger.info(f"{TT_RUN_PREFIX} New mode: Mesh Graph Descriptor = {resolved_mgd}")

    # Parse mock cluster mapping if provided
    mock_rank_to_desc: Optional[Dict[int, Path]] = None
    if mock_cluster_rank_binding:
        resolved_mock_path = resolve_path(
            mock_cluster_rank_binding, description="Mock cluster rank binding configuration", must_be_file=True
        )
        with open(resolved_mock_path, "r") as f:
            mock_data = yaml.safe_load(f)

        mock_rank_to_desc = {}

        # Check if this is a mapping file (has rank_to_cluster_mock_cluster_desc key) or a cluster descriptor file
        # Check the key in the YAML data, not the filename
        if mock_data and isinstance(mock_data, dict) and "rank_to_cluster_mock_cluster_desc" in mock_data:
            # Mapping format: rank -> cluster descriptor path
            for rank, path in mock_data["rank_to_cluster_mock_cluster_desc"].items():
                resolved_path = resolve_path(
                    path, description=f"Mock cluster descriptor for rank {rank}", must_be_file=True
                )
                mock_rank_to_desc[int(rank)] = resolved_path
        else:
            # Cluster descriptor format: treat as single entry with rank 0
            mock_rank_to_desc[0] = resolved_mock_path

        if verbose:
            logger.info(f"{TT_RUN_PREFIX} Mock cluster: {len(mock_rank_to_desc)} ranks")

    # Output directory: tt-run-generated/ relative to ORIGINAL_CWD
    output_dir = ORIGINAL_CWD / "tt-run-generated"
    output_dir.mkdir(exist_ok=True)

    if verbose:
        logger.info(f"{TT_RUN_PREFIX} Phase 1 output directory: {output_dir}")

    # Phase 1: Run generate_rank_bindings
    try:
        rank_bindings_path, rankfile_path = run_phase1_generate_rank_bindings(
            resolved_mgd,
            hosts,
            output_dir,
            subprocess_run=subprocess.run,
            sleep_secs=5,
            mock_rank_to_desc=mock_rank_to_desc,
            mpi_args=mpi_args,
        )
    except (FileNotFoundError, RuntimeError) as e:
        raise click.ClickException(f"Phase 1 (generate_rank_bindings) failed: {e}")

    # Phase 2: Use phase2_mock_mapping.yaml from generate_rank_bindings (cluster descriptors used during allocation).
    # The C++ tool writes this file; ttrun only reads it.
    phase2_mock_binding_path: Optional[Path] = None
    if mock_rank_to_desc:
        generated_phase2_mock_path = output_dir / "phase2_mock_mapping.yaml"
        if generated_phase2_mock_path.exists():
            phase2_mock_binding_path = generated_phase2_mock_path
            if verbose:
                with open(phase2_mock_binding_path, "r") as f:
                    phase2_data = yaml.safe_load(f) or {}
                phase2_mock_mapping = phase2_data.get("rank_to_cluster_mock_cluster_desc", {})
                logger.info(
                    f"{TT_RUN_PREFIX} Phase 2 mock mapping: {len(phase2_mock_mapping)} ranks from "
                    f"generate_rank_bindings"
                )

    # Log Phase 2-only command for re-runs without re-running generate_rank_bindings
    def _path_for_display(p: Path) -> str:
        try:
            return str(p.relative_to(ORIGINAL_CWD))
        except ValueError:
            return str(p)

    phase2_parts = ["tt-run", "--rank-binding", _path_for_display(rank_bindings_path)]
    if phase2_mock_binding_path:
        phase2_parts.extend(["--mock-cluster-rank-binding", _path_for_display(phase2_mock_binding_path)])
    mpi_launcher = get_mpi_launcher()
    rankfile_args = inject_rankfile_mpi_args(rankfile_path, mpi_args or [], mpi_launcher)
    if rankfile_needs_oversubscribe(rankfile_path) and "--oversubscribe" not in (mpi_args or []):
        rankfile_args = ["--oversubscribe"] + rankfile_args
    mpi_args_str = " ".join(shlex.quote(a) for a in rankfile_args)
    phase2_parts.extend(["--mpi-args", shlex.quote(mpi_args_str)])
    phase2_parts.extend(["--"] + [str(a) for a in ctx.args])
    logger.info(f"{TT_RUN_PREFIX} To re-run only Phase 2 (skip generate_rank_bindings): {' '.join(phase2_parts)}")

    legacy_flow(
        ctx,
        rank_binding=rank_bindings_path,
        dry_run=dry_run,
        verbose=verbose,
        mpi_args=mpi_args,
        debug_gdbserver=debug_gdbserver,
        mock_cluster_rank_binding=phase2_mock_binding_path,  # Pass Phase 2 mock mapping
        skip_executable_check=skip_executable_check,
        bare=bare,
        tcp_interface=tcp_interface,
        rankfile=rankfile_path,  # Pass generated rankfile
    )


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "--rank-binding",
    type=click.Path(path_type=Path),
    required=False,
    help="Rank binding configuration file (YAML). Relative paths are resolved against the launch directory.",
)
@click.option(
    "--mesh-graph-descriptor",
    type=click.Path(path_type=Path),
    required=False,
    help="Mesh graph descriptor file. When provided, enables new mode (mutually exclusive with --rank-binding). "
    "Requires --hosts unless --mock-cluster-rank-binding is provided.",
)
@click.option(
    "--hosts",
    type=str,
    required=False,
    callback=lambda ctx, param, value: [h.strip() for h in value.split(",")] if value else None,
    help="Comma-separated list of hostnames for MPI processes (e.g., 'node1,node2,node3'). "
    "Required for new mode (--mesh-graph-descriptor) unless --mock-cluster-rank-binding is provided. "
    "Not used in legacy mode (--rank-binding).",
)
@click.option("--dry-run", is_flag=True, help="Print command without executing")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show path resolution diagnostics, environment propagation, and MPI command details",
)
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
    help="Mock cluster rank binding configuration file (YAML). Relative paths are resolved against the launch directory. "
    "When used with new mode (--mesh-graph-descriptor), makes --hosts optional.",
)
@click.option(
    "--skip-executable-check", is_flag=True, help="Skip the check if program executable exists on the local host"
)
@click.option(
    "--bare",
    is_flag=True,
    help="Disable tt-run defaults (TCP transport, interface exclusions). Use for single-host or special setups.",
)
@click.option(
    "--tcp-interface",
    type=str,
    default=None,
    help="Network interface for MPI TCP communication (e.g., 'eth0', 'cnx1'). Uses btl_tcp_if_include instead of default exclusions.",
)
@click.pass_context
def main(
    ctx: click.Context,
    rank_binding: Optional[Path],
    mesh_graph_descriptor: Optional[Path],
    hosts: Optional[List[str]],
    dry_run: bool,
    verbose: bool,
    mpi_args: Optional[List[str]],
    debug_gdbserver: bool,
    mock_cluster_rank_binding: Optional[Path],
    skip_executable_check: bool,
    bare: bool,
    tcp_interface: Optional[str],
) -> None:
    """tt-run - MPI process launcher for TT-Metal and TTNN distributed applications

    tt-run operates in two modes:
        - Legacy mode: Use --rank-binding (see legacy_flow function for detailed documentation)
        - New mode: Use --mesh-graph-descriptor (mutually exclusive with --rank-binding)

    The two modes are mutually exclusive - you must specify exactly one.

    \b
    Quick Start:
        # Legacy mode
        tt-run --rank-binding rank_binding.yaml ./my_app

        # New mode (not yet implemented)
        tt-run --mesh-graph-descriptor mesh_graph.yaml --hosts node1,node2 ./my_app
        # Or with mock cluster (makes --hosts optional):
        tt-run --mesh-graph-descriptor mesh_graph.yaml --mock-cluster-rank-binding mock.yaml ./my_app

    For detailed documentation on legacy mode, see the legacy_flow function docstring.
    """
    # Check for mutually exclusive options
    if rank_binding is not None and mesh_graph_descriptor is not None:
        raise click.ClickException(
            "--rank-binding and --mesh-graph-descriptor are mutually exclusive. " "Please use only one of them."
        )

    if rank_binding is None and mesh_graph_descriptor is None:
        raise click.ClickException(
            "Either --rank-binding (legacy mode) or --mesh-graph-descriptor (new mode) must be specified."
        )

    # Legacy mode: use --rank-binding
    if rank_binding is not None:
        # Warn if new mode options are used with legacy mode
        if hosts is not None:
            logger.warning(
                f"{TT_RUN_PREFIX} --hosts is ignored in legacy mode (--rank-binding). "
                "Use --mesh-graph-descriptor to enable new mode."
            )
        legacy_flow(
            ctx,
            rank_binding,
            dry_run,
            verbose,
            mpi_args,
            debug_gdbserver,
            mock_cluster_rank_binding,
            skip_executable_check,
            bare,
            tcp_interface,
        )
        return

    # New mode: --mesh-graph-descriptor is provided
    # Validate required arguments for new mode
    if mesh_graph_descriptor is not None:
        # --hosts is required unless --mock-cluster-rank-binding is provided
        if mock_cluster_rank_binding is None and hosts is None:
            raise click.ClickException(
                "--hosts is required for new mode (--mesh-graph-descriptor) "
                "unless --mock-cluster-rank-binding is provided."
            )

        new_mode_flow(
            ctx,
            mesh_graph_descriptor,
            hosts,
            dry_run,
            verbose,
            mpi_args,
            debug_gdbserver,
            mock_cluster_rank_binding,
            skip_executable_check,
            bare,
            tcp_interface,
        )
        return


if __name__ == "__main__":
    main()
