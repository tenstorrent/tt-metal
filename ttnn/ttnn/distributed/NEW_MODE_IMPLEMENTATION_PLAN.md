# ttrun New Mode Implementation Plan

## Design Principle

**Modular and testable**: All logic is split into small, single-responsibility functions. Pure functions (no I/O) where possible; I/O functions accept injectable dependencies (e.g. `subprocess_run`) so unit tests can mock them.

## Overview

The new mode of `ttrun` simplifies launching distributed applications by requiring only:
- **`--mesh-graph-descriptor`** (MGD): Path to the mesh graph descriptor file
- **`--hosts`**: List of hostnames for MPI processes

The flow is a two-phase process:
1. **Phase 1**: Run `generate_rank_bindings` on the hosts with the MGD file → produces rankfile + rank_bindings.yaml
2. **Phase 2**: Call the legacy flow with the generated files → launches the user's program

---

## Current Architecture

### Existing Pieces to Reuse

| Component | Location | Purpose |
|-----------|----------|---------|
| `generate_rank_bindings` | `build/tools/scaleout/generate_rank_bindings` | C++ executable that runs PSD discovery, topology mapping, outputs rank_bindings.yaml + rankfile |
| `legacy_flow` | `ttrun.py` | Parses rank_bindings YAML, builds MPI command with per-rank env vars, launches program |
| `parse_binding_config` | `ttrun.py` | Parses rank binding YAML into TTRunConfig |
| `build_mpi_command` | `ttrun.py` | Builds mpirun command with per-rank environments |
| `resolve_path` | `ttrun.py` | Path resolution for files |

### generate_rank_bindings Details

- **CLI**: `generate_rank_bindings --mesh-graph-descriptor <path> [--physical-grouping-descriptor <path>]`
- **Must run under MPI** (e.g., `mpirun -np N --host node1,node2,... ./generate_rank_bindings -m mgd.textproto`)
- **Outputs** (written by rank 0 to `tt-run-generated/`):
  - `tt-run-generated/rank_bindings.yaml` – rank bindings in tt-run format
  - `tt-run-generated/rankfile` – OpenMPI rankfile format
- **Process count**: Need to align with number of hosts (one process per host for PSD discovery)

### legacy_flow Current Signature

```python
def legacy_flow(
    ctx: click.Context,
    rank_binding: Path,           # Path to rank_bindings.yaml
    dry_run: bool,
    verbose: bool,
    mpi_args: Optional[List[str]],   # User can pass --rankfile here manually
    debug_gdbserver: bool,
    mock_cluster_rank_binding: Optional[Path],
    skip_executable_check: bool,
    bare: bool,
    tcp_interface: Optional[str],
) -> None:
```

- **Does NOT** have a dedicated `rankfile` parameter – rankfile is passed via `mpi_args` (e.g. `--mpi-args "--rankfile path"`)

---

## Proposed Architecture

### 1. Adapt legacy_flow for Reuse – Rankfile Injection

**Goal**: Make `legacy_flow` callable with pre-generated rank_bindings + rankfile so new mode can pass generated files directly. When a rankfile is provided, tt-run must **auto-inject** the rankfile into MPI arguments using the correct syntax for the installed mpirun. The user does not specify the syntax; tt-run detects it.

**Interface change – Add optional `rankfile` parameter**
- Add `rankfile: Optional[Path] = None` to `legacy_flow`
- When `rankfile` is provided: call `inject_rankfile_mpi_args(rankfile, effective_mpi_args, mpi_launcher)` to prepend the correct rankfile args
- Keeps existing `--rank-binding` CLI flow unchanged (rankfile=None)
- New mode can call: `legacy_flow(..., rank_binding=generated_yaml_path, rankfile=generated_rankfile_path, ...)`

**MPI rankfile syntax varies by mpirun version** – tt-run must auto-detect:

| Syntax | Form | Used by |
|--------|------|---------|
| `--map-by rankfile:file=<path>` | `["--map-by", "rankfile:file=<path>"]` | OpenMPI 5.x / PRRTE (recommended, modern) |
| `--rankfile <path>` | `["--rankfile", "<path>"]` | Older OpenMPI (deprecated but still supported) |
| `-mca rmaps_rankfile_path <path>` | `["--mca", "rmaps_rankfile_path", "<path>"]` | Some older versions |

**Auto-determination (not user-specified)**:
1. Run `{mpi_launcher} --help` and parse output
2. Prefer `--map-by rankfile:file=` if help mentions it (OpenMPI 5.x / PRRTE)
3. Else use `--rankfile` if documented (older OpenMPI)
4. Else fall back to `-mca rmaps_rankfile_path`
5. Cache the result per process to avoid repeated help parsing (optional optimization)

**Implementation modules**:
- `get_mpi_launcher() -> str` – extract from `build_mpi_command`; returns `mpirun-ulfm` or `mpirun` (used by both build_mpi_command and inject_rankfile_mpi_args)
- `RankfileSyntax` – Enum: `MAP_BY_RANKFILE_FILE`, `RANKFILE`, `MCA_RMAPS_RANKFILE_PATH`
- `detect_rankfile_syntax(mpi_launcher: str, subprocess_run) -> RankfileSyntax` – queries mpirun `--help`, returns supported syntax
- `build_rankfile_args(syntax: RankfileSyntax, rankfile: Path) -> List[str]` – **pure**, returns args for given syntax
- `inject_rankfile_mpi_args(rankfile: Path, base_mpi_args: List[str], mpi_launcher: str, detect_fn=detect_rankfile_syntax) -> List[str]` – detects, builds, prepends; inject `detect_fn` for tests

**Call order in legacy_flow** (when rankfile provided):
1. `mpi_launcher = get_mpi_launcher()`
2. `effective_mpi_args = inject_rankfile_mpi_args(rankfile, base_mpi_args, mpi_launcher)` — returns `[rankfile_args...] + base_mpi_args` (prepends rankfile args)

### 2. New Mode Flow (`new_mode_flow`)

```
new_mode_flow(mesh_graph_descriptor, hosts, program, ...)
│
├─ Step 1: Resolve paths
│   ├─ Resolve mesh_graph_descriptor path
│   ├─ Find generate_rank_bindings executable (build/tools/scaleout/generate_rank_bindings)
│   └─ Output dir: tt-run-generated/ (same as generate_rank_bindings default)
│
├─ Step 2: First MPI call – run generate_rank_bindings
│   IMPORTANT: Must explicitly pass --host with the hosts list (do NOT rely on -np alone).
│   generate_rank_bindings writes to tt-run-generated/ (relative to CWD).
│
│   mpirun --host <host1>,<host2>,... -np <len(hosts)> --tag-output \
│     ./build/tools/scaleout/generate_rank_bindings \
│     --mesh-graph-descriptor <mgd_path>
│
│   Example: hosts=["nodeA","nodeB","nodeC"] →
│     mpirun --host nodeA,nodeB,nodeC -np 3 --tag-output ...
│   │
│   └─ Check exit code; on failure, report and exit
│
├─ Step 3: Wait 5 seconds
│   Sleep 5 seconds to allow generated files to sync (e.g. NFS, shared storage) before Phase 2.
│
├─ Step 4: Read generated outputs from tt-run-generated/
│   Path: ORIGINAL_CWD / tt-run-generated/ (generate_rank_bindings writes here relative to CWD)
│   ├─ rank_bindings.yaml → tt-run-generated/rank_bindings.yaml
│   └─ rankfile → tt-run-generated/rankfile
│
└─ Step 5: Call legacy_flow (or run_with_config) with generated files
    legacy_flow(
        ctx,
        rank_binding=tt-run-generated/rank_bindings.yaml,
        rankfile=tt-run-generated/rankfile,   # NEW param
        mpi_args=...,                     # Don't duplicate rankfile
        ...
    )
```

### 3. First vs Second MPI Command – Process Count and Host Mapping (Critical)

The two MPI commands use **different** process counts and host assignments. Do not conflate them.

| | Phase 1 (generate_rank_bindings) | Phase 2 (user program via legacy_flow) |
|---|----------------------------------|---------------------------------------|
| **Process count** | `np = len(hosts)` (real) or `np = len(rank_to_cluster_mock_cluster_desc)` (mock) | `np = len(rank_bindings)` (from generated YAML) |
| **Host assignment** | Real: `--host host1,host2,...`. Mock: all localhost, per-rank `TT_METAL_MOCK_CLUSTER_DESC_PATH` | `--rankfile` + `rankfile` content (rank → host/slot mapping) |
| **Source** | User's `--hosts` or mock mapping file | Generated `rank_bindings.yaml` + `rankfile` from Phase 1 |

**Phase 1**: One discovery process per host. Used for PSD discovery across the cluster.
- `mpirun --host nodeA,nodeB,nodeC -np 3 ./generate_rank_bindings ...`

**Phase 2**: Number of processes = number of ranks in the generated rank_bindings. The rankfile specifies which host and slot each rank uses.
- `mpirun --rankfile <path> ...` (or `--map-by rankfile:file=<path>`)
- legacy_flow's `build_mpi_command` iterates over `config.rank_bindings` – one `-np 1` segment per rank
- The rankfile maps rank 0 → host X slot Y, rank 1 → host Z slot W, etc.

**Example**: 3 hosts, but MGD defines 4 ranks (e.g. 2 on host0, 1 each on host1/host2).
- Phase 1: `-np 3`, `--host host0,host1,host2`
- Phase 2: 4 processes, placed per rankfile (rank 0,1 on host0; rank 2 on host1; rank 3 on host2)

### 4. Phase 1 MPI Command – Host Assignment (Required)

**Requirement**: The first MPI call (generate_rank_bindings) MUST pass the hosts list explicitly via `--host`.

- OpenMPI: `mpirun --host node1,node2,node3 -np 3 ...`
- The hosts list comes from `--hosts` (already a `List[str]` in new mode)
- Join with comma: `",".join(hosts)`
- Do NOT omit `--host` – without it, mpirun may launch processes on localhost only regardless of `-np`

### 5. Open Questions / Decisions

1. **`-np` for generate_rank_bindings** (Phase 1)
   - `np = len(hosts)` – one process per host for PSD discovery
   - Confirm with generate_rank_bindings behavior and docs

2. **Output directory**
   - generate_rank_bindings writes to `tt-run-generated/` (relative to CWD)
   - New mode should ensure CWD is stable (e.g. ORIGINAL_CWD)
   - Consider: allow override via `--output-dir` or similar?

3. **generate_rank_bindings executable path**
   - Likely: `{TT_METAL_HOME}/build/tools/scaleout/generate_rank_bindings`
   - Or search like other tools (resolve_path)

4. **Physical Grouping Descriptor (PGD)**
   - generate_rank_bindings supports optional `--physical-grouping-descriptor`
   - New mode: add optional `--physical-grouping-descriptor` to ttrun new mode?

5. **Mock cluster / single-host**
   - When `--mock-cluster-rank-binding` is used: **`--hosts` is not required**
   - See "Mock Cluster Path" section below for Phase 1 behavior

---

## Modular Design (Testable)

All logic MUST be split into small, pure or injectable functions so each can be unit tested in isolation.

### Module 1: generate_rank_bindings helpers (new_mode)

| Function | Signature | Pure? | Purpose | Unit Test |
|----------|------------|-------|---------|-----------|
| `find_generate_rank_bindings_executable` | `() -> Path` | No (uses resolve_path, env) | Locate `generate_rank_bindings` binary | Mock resolve_path, check returned path |
| `build_generate_rank_bindings_mpi_cmd` | `(executable: Path, mgd_path: Path, hosts: List[str], output_dir: Path, mock_rank_to_desc: Optional[Dict[int, Path]] = None) -> List[str]` | **Yes** | Build Phase 1 mpirun command. With hosts: `-np` = len(hosts), `--host` = hosts. With mock: `-np` = len(mock_rank_to_desc), `--host localhost`, per-rank `-x TT_METAL_MOCK_CLUSTER_DESC_PATH=<path>`. | Assert `--host` present, `-np` correct; with mock, assert per-rank env vars |
| `get_generate_rank_bindings_output_paths` | `(output_dir: Path) -> tuple[Path, Path]` | **Yes** | Return (rank_bindings.yaml path, rankfile path). Default output_dir = tt-run-generated/ | Assert paths inside output_dir |
| `run_generate_rank_bindings` | `(cmd: List[str], cwd: Path, subprocess_run) -> int` | No (I/O) | Run cmd via subprocess_run, return exit code. **Inject subprocess.run** for tests. | Mock subprocess_run, assert called with correct cmd |
| `run_phase1_generate_rank_bindings` | `(mgd_path, hosts, output_dir, subprocess_run, sleep_secs=5, mock_rank_to_desc: Optional[Dict[int, Path]] = None) -> tuple[Path, Path]` | No | Orchestrates: build cmd (using hosts or mock_rank_to_desc), run, sleep sleep_secs (default 5), validate outputs exist. With mock: np=len(mock_rank_to_desc), no --hosts, per-rank TT_METAL_MOCK_CLUSTER_DESC_PATH. | Mock subprocess_run, mock filesystem; use sleep_secs=0 in tests; test both hosts and mock paths |

### Module 2: legacy_flow adaptation and rankfile injection

| Function | Signature | Pure? | Purpose | Unit Test |
|----------|------------|-------|---------|-----------|
| `get_mpi_launcher` | `() -> str` | No (shutil.which) | Return mpirun-ulfm or mpirun; extract from build_mpi_command | Mock shutil.which |
| `RankfileSyntax` | Enum: `MAP_BY_RANKFILE_FILE`, `RANKFILE`, `MCA_RMAPS_RANKFILE_PATH` | - | Syntax variants for different mpirun versions | - |
| `detect_rankfile_syntax` | `(mpi_launcher: str, subprocess_run) -> RankfileSyntax` | No (I/O) | Run `{launcher} --help`, parse output, return supported syntax | Mock subprocess_run; assert correct syntax for sample help output |
| `build_rankfile_args` | `(syntax: RankfileSyntax, rankfile: Path) -> List[str]` | **Yes** | Return MPI args for given syntax and path | Assert output for each syntax variant |
| `inject_rankfile_mpi_args` | `(rankfile: Path, base_mpi_args: List[str], mpi_launcher: str, detect_fn=detect_rankfile_syntax) -> List[str]` | No | Detect syntax, build args, return rankfile_args + base_mpi_args | Inject mock detect_fn; assert correct args prepended |
| `legacy_flow` | (existing + `rankfile: Optional[Path] = None`) | No | When rankfile provided: call inject_rankfile_mpi_args before user mpi_args | Test with rankfile=None vs provided |

### Module 3: new_mode_flow orchestration

| Function | Signature | Pure? | Purpose | Unit Test |
|----------|------------|-------|---------|-----------|
| `new_mode_flow` | (existing params) | No | Resolve paths → run_phase1 → legacy_flow. Thin orchestrator. | Mock run_phase1, legacy_flow; verify call order and args |

### Dependency Injection for Testability

- **subprocess.run**: Pass as parameter (e.g. `subprocess_run=subprocess.run`) or use a wrapper. In tests: `subprocess_run=MagicMock()`.
- **resolve_path**: Already a top-level function; tests can monkeypatch.
- **Path / filesystem**: Use `pathlib.Path`; tests use `tmp_path` fixture.

### Function Naming: No Leading Underscore

Use **public** names (e.g. `find_generate_rank_bindings_executable`, `build_generate_rank_bindings_mpi_cmd`) so they can be imported and tested directly. Do not use `_private` for functions that need unit tests.

---

## Implementation Tasks

### Phase 1: Adapt legacy_flow and add rankfile injection ✅ COMPLETE
- [x] Extract `get_mpi_launcher() -> str` from build_mpi_command logic
- [x] Add `RankfileSyntax` enum and `detect_rankfile_syntax(mpi_launcher, subprocess_run) -> RankfileSyntax`
- [x] Add `build_rankfile_args(syntax, rankfile) -> List[str]` (pure)
- [x] Add `inject_rankfile_mpi_args(rankfile, base_mpi_args, mpi_launcher, detect_fn=...) -> List[str]`
- [x] Add `rankfile: Optional[Path] = None` to legacy_flow
- [x] When rankfile provided: call `inject_rankfile_mpi_args(rankfile, base_mpi_args, get_mpi_launcher())` to prepend rankfile args (auto-detect mpirun syntax)
- [x] Update docstring; ensure user does not pass rankfile in --mpi-args when using new rankfile param (or document that tt-run's rankfile param takes precedence)
- [x] Added conflict detection: skip injection if rankfile args already present in user's --mpi-args

### Phase 2: Implement modular generate_rank_bindings helpers ✅ COMPLETE
- [x] `find_generate_rank_bindings_executable() -> Path` – locate executable (searches TT_METAL_HOME, ORIGINAL_CWD, current dir)
- [x] `build_generate_rank_bindings_mpi_cmd(executable, mgd_path, hosts, output_dir, mock_rank_to_desc) -> List[str]` – **pure**, returns cmd list (supports both hosts and mock)
- [x] `get_generate_rank_bindings_output_paths(output_dir) -> tuple[Path, Path]` – **pure**
- [x] `run_generate_rank_bindings(cmd, cwd, subprocess_run) -> int` – inject subprocess_run
- [x] `run_phase1_generate_rank_bindings(mgd_path, hosts, output_dir, subprocess_run, sleep_secs=5, mock_rank_to_desc)` – orchestrator; sleeps 5s after Phase 1 for NFS/sync

### Phase 3: Implement new_mode_flow ✅ COMPLETE
- [x] Resolve mesh_graph_descriptor path
- [x] When mock_cluster_rank_binding provided: parse mapping file, get rank count and rank→descriptor paths
- [x] Call `run_phase1_generate_rank_bindings` (inject subprocess.run, sleep_secs=5); with mock, use np=len(mock_ranks) and per-rank TT_METAL_MOCK_CLUSTER_DESC_PATH
- [x] Read rank_bindings.yaml and rankfile from tt-run-generated/ (use `get_generate_rank_bindings_output_paths`)
- [x] Call `legacy_flow` with generated rank_binding and rankfile
- [x] Error handling and cleanup

### Phase 4: Unit Tests (all modular functions) ✅ IN PROGRESS

See [TTRUN_TEST_PLAN.md](TTRUN_TEST_PLAN.md) for full test plan. **Critical: ttrun tests must never launch MPI; use `--dry-run` and mock `subprocess.run`.**

- [x] `test_build_rankfile_args` – for each RankfileSyntax variant
- [x] `test_inject_rankfile_mpi_args` – with mock detect_fn, each syntax
- [x] `test_get_mpi_launcher` – verify launcher detection
- [x] `test_legacy_flow_rankfile_conflict` – verify conflict detection
- [ ] `test_build_generate_rank_bindings_mpi_cmd` – assert `--host`, hosts list, `-np`, executable, mgd arg
- [ ] `test_build_generate_rank_bindings_mpi_cmd_mock` – with mock_rank_to_desc: assert np=len(mock), each rank has TT_METAL_MOCK_CLUSTER_DESC_PATH set to corresponding descriptor
- [ ] `test_get_generate_rank_bindings_output_paths` – assert paths correct
- [ ] `test_detect_rankfile_syntax` – mock subprocess_run with sample --help output
- [ ] `test_find_generate_rank_bindings_executable` – mock resolve_path
- [ ] `test_run_phase1_generate_rank_bindings` – mock subprocess_run and fs
- [ ] `test_new_mode_flow` – mock run_phase1 and legacy_flow

---

## Code Reuse Summary

| Reused As-Is | Reused With Mod |
|--------------|-----------------|
| `parse_binding_config` | `legacy_flow` – add rankfile param |
| `build_mpi_command` | |
| `resolve_path` | |
| `get_rank_environment`, `build_rank_environment_args` | |
| multihost MPI args logic | |
| `generate_rank_bindings` executable | Run via subprocess with mpirun |

---

## Mock Cluster Path

When `--mock-cluster-rank-binding` is provided:

- **`--hosts` is not required** – all processes run on localhost
- **Phase 1** (first MPI command): Launch `generate_rank_bindings` with:
  - **Process count**: `np = len(rank_to_cluster_mock_cluster_desc)` (number of rankings in the mapping file)
  - **Host assignment**: All on localhost (e.g. `--host localhost` or equivalent; no multi-host)
  - **Per-rank env**: Each rank must receive `TT_METAL_MOCK_CLUSTER_DESC_PATH` pointing to its corresponding mock cluster descriptor from the mapping file (rank 0 → first descriptor, rank 1 → second, etc.)
- **Phase 2**: Same as normal flow – call legacy_flow with generated rank_bindings.yaml and rankfile
- **Testing**: Unit tests must verify that Phase 1 is invoked with the correct `np`, that each rank’s env includes the correct `TT_METAL_MOCK_CLUSTER_DESC_PATH`, and that the mock mapping file format is parsed correctly

---

## Next Steps

1. **Review this plan** – confirm flow, resolve open questions
2. **Implement Phase 1** – adapt legacy_flow with rankfile param
3. **Implement Phase 2–3** – new_mode_flow implementation
4. **Test** – unit tests and manual runs
