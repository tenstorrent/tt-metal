# tt-run Test Plan

## Core Principle: No MPI Launch, Dry-Run Only

**tt-run tests must never launch MPI or any real subprocess.** All tests either:

1. **Run with `--dry-run`** – tt-run prints the constructed command and exits without executing it; `subprocess.run` must not be called
2. **Test pure functions in isolation** – e.g. `build_generate_rank_bindings_mpi_cmd`, `inject_rankfile_mpi_args`, `parse_binding_config` – no subprocess
3. **Mock subprocess and I/O** – when testing functions that would call `subprocess.run`, inject a mock; assert the mock was called with expected arguments (or not called at all in dry-run)

---

## Test Categories

### 1. Command-Line and Mode Selection (Existing + Extensions)

| Test | What to verify | Dry-run / Mock |
|------|----------------|----------------|
| `test_mutually_exclusive_options` | `--rank-binding` and `--mesh-graph-descriptor` cannot be used together | CLI only |
| `test_no_mode_specified` | Either mode must be specified | CLI only |
| `test_new_mode_not_implemented` | New mode raises "not yet implemented" | CLI only |
| `test_new_mode_requires_hosts` | New mode without `--hosts` and without `--mock-cluster-rank-binding` fails | CLI only |
| `test_new_mode_with_hosts` | New mode with `--hosts` proceeds to "not yet implemented" (not "hosts required") | CLI only |
| `test_new_mode_with_mock_cluster_no_hosts` | New mode with `--mock-cluster-rank-binding` proceeds without `--hosts` | CLI only |
| `test_legacy_mode_ignores_hosts` | Legacy mode accepts `--hosts` but ignores it; dry-run succeeds | **`--dry-run` + mock subprocess** |

---

### 2. Legacy Flow – Dry-Run Only

| Test | What to verify | How |
|------|----------------|-----|
| `test_legacy_flow_dry_run` | With `--dry-run`, `subprocess.run` is **never** called | Patch `subprocess.run`, assert `mock_subprocess.assert_not_called()` |
| `test_legacy_flow_dry_run_host_count` | Printed MPI command contains `-np N` where `N = len(rank_bindings)` | Parse `result.output`, assert `-np` value matches rank count |
| `test_legacy_flow_dry_run_host_list` | With `--mpi-args "--host A,B,C"`, printed command includes host list | Assert host string in output |
| `test_legacy_flow_dry_run_rankfile` | When rankfile is passed via `--mpi-args`, it appears in output | Assert `rankfile:file=` or `--rankfile` in output |
| `test_legacy_flow_dry_run_per_rank_env` | Each rank gets `TT_MESH_ID`, `TT_MESH_GRAPH_DESC_PATH`, etc. | Parse output or test `build_mpi_command` / `get_rank_environment` directly |

**Implementation note**: All legacy-flow tests must use `--dry-run`. Do **not** run tt-run without `--dry-run` in any test that reaches the execution path.

---

### 3. Pure Functions (No Subprocess, No MPI)

| Test | Function | What to verify |
|------|----------|----------------|
| `test_build_mpi_command_basic` | `build_mpi_command` | Command contains mpirun, `--tag-output`, `--bind-to` |
| `test_build_mpi_command_rank_count` | `build_mpi_command` | `-np` segments or total process count = `len(config.rank_bindings)` |
| `test_build_rankfile_args_map_by` | `build_rankfile_args` | For `MAP_BY_RANKFILE_FILE`: returns `["--map-by", "rankfile:file=<path>"]` |
| `test_build_rankfile_args_rankfile` | `build_rankfile_args` | For `RANKFILE`: returns `["--rankfile", "<path>"]` |
| `test_build_rankfile_args_mca` | `build_rankfile_args` | For `MCA_RMAPS_RANKFILE_PATH`: returns `["--mca", "rmaps_rankfile_path", "<path>"]` |
| `test_inject_rankfile_mpi_args` | `inject_rankfile_mpi_args` | With mock detect_fn: asserts rankfile args prepended to base_mpi_args; test each syntax |
| `test_detect_rankfile_syntax` | `detect_rankfile_syntax` | Mock subprocess_run returning sample `--help` output; assert MAP_BY_RANKFILE_FILE or RANKFILE returned |
| `test_get_generate_rank_bindings_output_paths` | `get_generate_rank_bindings_output_paths` | Returns `(output_dir/rank_bindings.yaml, output_dir/rankfile)` |
| `test_parse_binding_config_mock` | `parse_binding_config` | With mock mapping file, `mock_cluster_rank_binding` dict has correct rank→path |

---

### 4. Phase 1 Helpers (When Implemented)

These test the **command construction** only. `subprocess.run` is **never** called; we assert on the command list that would be passed.

| Test | Function | What to verify |
|------|----------|----------------|
| `test_build_generate_rank_bindings_mpi_cmd_hosts` | `build_generate_rank_bindings_mpi_cmd` | `-np` = len(hosts), `--host` = hosts joined by comma |
| `test_build_generate_rank_bindings_mpi_cmd_mock` | `build_generate_rank_bindings_mpi_cmd` | `-np` = len(mock_rank_to_desc), `--host localhost`, per-rank `TT_METAL_MOCK_CLUSTER_DESC_PATH` |
| `test_build_generate_rank_bindings_mpi_cmd_executable` | `build_generate_rank_bindings_mpi_cmd` | Executable path and `--mesh-graph-descriptor` arg present |
| `test_find_generate_rank_bindings_executable` | `find_generate_rank_bindings_executable` | Returns path; mock `resolve_path` to avoid filesystem |

---

### 5. Orchestration (Mocked Phase 1 and Legacy)

| Test | What to verify | How |
|------|----------------|-----|
| `test_run_phase1_generate_rank_bindings` | Correct cmd passed to `subprocess_run` | Inject `subprocess_run=MagicMock()`, assert call args |
| `test_run_phase1_generate_rank_bindings_mock` | With mock mapping: `np` and env vars correct | Same pattern, verify cmd contains `-np N` and per-rank `-x TT_METAL_MOCK_CLUSTER_DESC_PATH=...` |
| `test_new_mode_flow_dry_run` | When new mode supports dry-run: no subprocess, printed Phase 1 + Phase 2 commands | Mock `run_phase1` and `legacy_flow` or assert `subprocess.run` not called |

---

### 6. YAML and Path Resolution (Existing)

| Test | What to verify |
|------|----------------|
| `test_valid_rank_binding_yaml` | Parse valid YAML, rank count, mesh_graph_desc_path |
| `test_invalid_rank_binding_duplicate_ranks` | Duplicate ranks raise `ValueError` |
| `test_resolve_absolute_path` | Absolute path resolved correctly |
| `test_resolve_path_not_found` | Missing path raises `ValueError` |

---

### 7. Environment Variables (Existing)

| Test | What to verify |
|------|----------------|
| `test_get_rank_environment` | `TT_MESH_ID`, `TT_MESH_GRAPH_DESC_PATH`, `TT_MESH_HOST_RANK`, `env_overrides`, `global_env` |
| `test_get_rank_environment_with_mock` | When `mock_cluster_rank_binding` present: `TT_METAL_MOCK_CLUSTER_DESC_PATH` set per rank |

---

## Test Execution Guardrails

1. **CI / pytest**
   - All tests in `tests/ttnn/distributed/test_ttrun.py` must pass without network or MPI runtime
   - No `mpirun` or `generate_rank_bindings` binary required
   - Use `tmp_path` for temp files; no reliance on real cluster configs

2. **Subprocess mocking**
   - Any test that can reach code calling `subprocess.run` must patch it:
     ```python
     with patch.object(subprocess, "run") as mock_run:
         result = runner.invoke(main, ["--dry-run", ...])
         mock_run.assert_not_called()  # dry-run must not run anything
     ```
   - For tests of `run_generate_rank_bindings` or `run_phase1_generate_rank_bindings`: inject `subprocess_run=MagicMock()` and assert on the command passed, not on real execution

3. **Dry-run as default for integration-style tests**
   - Any test that invokes `main()` with a real executable (e.g. `echo`, `true`) must use `--dry-run` so tt-run never actually runs it

---

## Summary Checklist

- [ ] No test launches MPI
- [ ] No test runs `mpirun` or `generate_rank_bindings` for real
- [ ] Legacy flow tests use `--dry-run` and mock `subprocess.run`
- [ ] Pure functions tested without any subprocess
- [ ] Phase 1 helper tests assert on command construction only
- [ ] Host count (`-np`), host list (`--host`), and per-rank env vars validated in dry-run output or via unit tests
