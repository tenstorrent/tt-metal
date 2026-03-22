<!-- SUMMARY: Architecture for ULFM rank reinitialization and work redistribution in tt-metal multihost -->
<!-- KEYWORDS: ULFM, MPI, multihost, rank-failure, reinitialization, tt-smi, fault-tolerant, work-redistribution -->
<!-- SOURCE: codebase investigation + ULFM spec knowledge -->
<!-- SCOPE: tt_metal/distributed/multihost, ttrun.py, .github workflows, conftest fixtures -->
<!-- USE WHEN: designing fault-tolerant multihost test infrastructure or implementing rank respawn -->

# ULFM Rank Reinitialization & Work Redistribution Architecture

**Target repo:** `tenstorrent/tt-metal` (branch `nsexton/0-multihost-triage`)
**Date:** 2026-03-21
**Author:** BrAIn (AI SW Infrastructure)

---

## Table of Contents

1. [Failure Taxonomy](#1-failure-taxonomy)
2. [Current State Assessment](#2-current-state-assessment)
3. [Rank Reinitialization Architecture](#3-rank-reinitialization-architecture)
4. [Required Code Changes](#4-required-code-changes)
5. [Recommended Implementation Phases](#5-recommended-implementation-phases)
6. [Ergonomics and Usability](#6-ergonomics-and-usability)
7. [Risks and Limitations](#7-risks-and-limitations)
8. [Port Laddering, Inspector, Watcher, and Triage Interactions](#8-port-laddering-inspector-watcher-and-triage-interactions)
9. [Watcher-to-ULFM Bridge, Multihost Defaults, and Dispatch Timeout Semantics](#9-watcher-to-ulfm-bridge-multihost-defaults-and-dispatch-timeout-semantics)

---

## Implementation Status

> ⚠️ **Scope of this branch**: This branch implements ULFM *detection* (Layers 1–3: revocation,
> shrink, fast-fail exit code propagation). Rank respawn, hardware reset, and work redistribution
> (Phase 2–3) are **NOT YET IMPLEMENTED**. Items marked ⏳ in the table below are future work.

Status of bugs/gaps identified in this document against branch `nsexton/0-multihost-triage`:

```
Bug/Gap                              Status   Commit / Notes
───────────────────────────────────  ───────  ─────────────────────────────────────
is_revoked() using MPI_Comm_test_   ✅ Fixed  3345ccce — uses revoked_.load() now
  inter (Section 2.1)
No MPI_Finalize watchdog             ✅ Fixed  5bb8ec27 — SIGALRM in init_env()
  (Section 2.3 / 7.5)
No FailurePolicy enum                ✅ Fixed  2747ecf5 — FAST_FAIL / FAULT_TOLERANT
  (Section 2.2)
mpi_fault.py missing                 ✅ Fixed  2747ecf5 — created with ulfm_guard()
  (Section 2.6 item 7)
No MPI_ERRORS_RETURN                 ✅ Fixed  base branch — set in constructors
  (Section 2.1)
No MPIX_Comm_agree wrapper           ✅ Fixed  33c867c6 — MPIContext::agree()
  (Section 2.6 item 1)
Thread safety of                     ✅ Fixed  33c867c6 — comm_mutex_ added
  revoke_and_shrink() (Section 7.6)
No failed_ranks() accessor           ✅ Fixed  33c867c6 — queries MPIX_Comm_failure_
  (Section 6.2)                                  get_acked directly
mpi_fault.py error code constants    ✅ OK     Verified correct for mpi4py + ULFM
No MPI_Comm_spawn usage              ⏳ Phase 3 — not yet implemented
  (Section 2.6 item 2)
No hardware reset from MPI context   ⏳ Phase 2 — not yet implemented
  (Section 2.6 item 4)
No work redistribution               ⏳ Phase 3 — not yet implemented
  (Section 2.6 item 8)
ttrun.py exit code interpretation    ⏳ Phase 2 — partially done, needs more
  (Section 2.6 item 7)
```

---

## 1. Failure Taxonomy

### 1.1 Failure Modes

```
Failure Mode           ULFM Detectable?  Affected Ranks      Recovery Path
─────────────────────  ────────────────  ──────────────────  ─────────────────────
Clean exit (exit(0))   Yes (ERR_PROC)    Just the exiting    Shrink + continue
Segfault / SIGABRT     Yes (ERR_PROC)    Just the crashing   Shrink + hw reset + respawn
OOM (SIGKILL by OOM)   Yes (ERR_PROC)    Just the killed     Shrink + hw reset + respawn
Hang (infinite loop)   No (no signal)    Appears alive       Timeout-based detection
Network partition      Partial           Split groups        ULFM may see ERR_PROC for
                                                             unreachable ranks; timeout
NFS stale handle       No (app-level)    Process alive but   App-level retry; remount
                                         I/O failing
Device firmware crash  No (app-level)    Process alive but   tt-smi -r + process restart
                                         device unusable
Power/hardware fault   Yes (TCP drop)    All ranks on node   Shrink node's ranks + respawn
```

### 1.2 ULFM Detection Mechanism

ULFM detects rank death through the underlying transport layer (TCP connection drop, heartbeat timeout). When a rank's MPI process terminates abnormally:

1. The next MPI operation targeting that rank returns `MPIX_ERR_PROC_FAILED`
2. Any collective involving the dead rank returns `MPIX_ERR_PROC_FAILED`
3. A revoked communicator returns `MPIX_ERR_REVOKED` (MPI error code 78 in OpenMPI 5.x)

**What ULFM cannot detect:**
- Application-level hangs (rank is alive, MPI stack is fine, but the app is stuck)
- Device-only failures (TT device firmware crash, but host process is still running)
- NFS/filesystem failures (process alive, MPI operational, but I/O fails)
- Partial network degradation (slow but not disconnected)

### 1.3 Node-Level vs Rank-Level Failures

In tt-metal's multihost setup, each physical node typically runs **one MPI rank** (one `tt-run` process per node, each controlling one or more TT devices). This means:

- **Node death** = rank death (1:1 mapping in typical deployments)
- **Device failure on a node** does NOT kill the MPI rank (the host process may still be alive but unable to use its devices)
- The `distributed_reset.sh` pattern (parallel-ssh to all nodes to run `tt-smi -r`) shows the existing assumption: reset is node-level, not rank-level

---

## 2. Current State Assessment

### 2.1 What's Implemented

**File: `tt_metal/distributed/multihost/mpi_distributed_context.cpp`**

The MPI distributed context already has substantial ULFM infrastructure:

- **`OMPI_HAS_ULFM` compile-time check** (lines 16-21): Detects whether OpenMPI was built with ULFM by checking for `MPIX_ERR_PROC_FAILED`. The ULFM build is at `/opt/openmpi-v5.0.7-ulfm` (installed from `tenstorrent/ompi` releases).

- **`MPI_ERRORS_RETURN` error handler** (lines 206, 214): All communicators are created with `MPI_ERRORS_RETURN`, meaning MPI calls return error codes instead of aborting. This is the prerequisite for fault tolerance.

- **`revoke_and_shrink()`** (lines 493-528): The core ULFM recovery routine:
  1. Calls `MPIX_Comm_revoke(comm_)` to poison the communicator (all pending/future operations on it will fail with `MPI_ERR_REVOKED`)
  2. Calls `MPIX_Comm_shrink(comm_, &new_comm)` to create a new communicator excluding dead ranks
  3. Updates internal state (`comm_`, `group_`, `rank_`, `size_`) in-place
  4. Frees the old communicator

  **Key design choice:** The shrink modifies the `MPIContext` object in-place rather than returning a new one. This means all code holding a reference to the context object automatically sees the repaired communicator. This is elegant but has thread-safety implications (noted in the comment on line 43 of ulfm_tests.cpp).

- **`is_revoked()`**: Checks if the current communicator has been revoked. **UPDATE:** Now correctly uses `revoked_.load(std::memory_order_acquire)` on an `std::atomic<bool>` flag that is set in `handle_rank_failure()` and `revoke_and_shrink()`, and cleared after successful shrink (commit `3345ccce`).

- **`supports_fault_tolerance()`** (line 222): Returns `OMPI_HAS_ULFM` at compile time.

- **`MPIDistributedException`** (lines 105-123): Carries rank, MPI error code, and human-readable error string. Used by `mpi_check()` to throw on any MPI error.

**File: `tt_metal/api/tt-metalium/distributed_context.hpp`**

The abstract `DistributedContext` interface exposes:
- `supports_fault_tolerance()` — virtual, returns bool
- `revoke_and_shrink()` — virtual, modifies communicator in-place
- `is_revoked()` — virtual, checks communicator state

**File: `tt_metal/distributed/multihost/single_host_context.cpp`**

The single-host implementation stubs out fault tolerance:
- `supports_fault_tolerance()` returns `false`
- `revoke_and_shrink()` throws
- `is_revoked()` returns `false`

### 2.2 What the FAST_FAIL Path Does Today

**UPDATE:** A `FailurePolicy` enum (`FAST_FAIL` / `FAULT_TOLERANT`) is now implemented (commit `2747ecf5`). The `mpi_check_ctx()` function dispatches to `handle_rank_failure()` which either calls `_exit(70)` (FAST_FAIL) or throws `MPIRankFailureException` (FAULT_TOLERANT). Callers can switch via `ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT)`.

The fault tolerance tests (`ulfm_tests.cpp`) demonstrate the recovery pattern:
1. Catch `DistributedException` from a failed collective
2. Call `ctx->revoke_and_shrink()` on survivors
3. Continue with the shrunken communicator

But this pattern is only exercised in tests — there's no production code path that catches MPI errors and attempts recovery.

### 2.3 The MPI_Finalize Watchdog

**UPDATE:** A SIGALRM watchdog is now implemented (commit `5bb8ec27`). The `init_env()` function arms a 30-second alarm before `MPI_Finalize()` in the atexit handler. If MPI_Finalize does not complete in time, `mpi_finalize_alarm_handler` calls `_exit(70)`. Additionally, a `std::set_terminate` handler revokes `MPI_COMM_WORLD` on uncaught exceptions before calling `_exit(70)`.

### 2.4 Existing ULFM Tests

**File: `tests/tt_metal/multihost/fault_tolerance_tests/ulfm_tests.cpp`**

Two tests exist:

1. **`ShrinkAfterRankFailure`**: Kills rank 1 via `SIGKILL`, survivors detect failure at barrier, call `revoke_and_shrink()`, verify shrunken communicator has N-1 ranks.

2. **`DisableBrokenBlock`**: Kills one rank in a "block" (pair of ranks sharing a machine), then uses `split()` to separate healthy blocks from broken ones. Demonstrates the pattern of excluding an entire node when one of its ranks dies.

Run via: `mpirun-ulfm --with-ft ulfm -np 8 ./build/test/tt_metal/fault_tolerance_tests`

### 2.5 Existing Reset Infrastructure

**Hardware reset patterns found in the repo:**

1. **`tt-smi -r`** — The primary device reset command. Used in:
   - `.github/actions/ensure-bh-links-online/action.yml`: Retry loop (up to 10 attempts) of `tt-smi -r` + health check
   - `.github/workflows/ttnn-run-sweeps.yaml`: Per-runner reset (`tt-smi -r` for standard, `tt-smi -glx_reset_auto` for Galaxy 6U topology)
   - `tests/scale_out/4x_bh_quietbox/distributed_reset.sh`: Synchronized reset across 4 nodes via `parallel-ssh`

2. **`distributed_reset.sh`** (`tests/scale_out/4x_bh_quietbox/distributed_reset.sh`): Uses NFS barrier + `parallel-ssh` to synchronize `tt-smi -r` across cluster nodes. Critical insight: reset skew between nodes causes link failures, so resets must be synchronized.

3. **`validate_cluster_health.py`** (`tests/scale_out/4x_bh_quietbox/validate_cluster_health.py`): Retry loop with automatic reset on failure (up to 10 attempts). Calls `distributed_reset.sh` between attempts.

4. **`ensure-bh-links-online` GitHub Action** (`.github/actions/ensure-bh-links-online/action.yml`): Health check loop for Blackhole devices: `tt-smi -r` + `test_system_health` binary.

### 2.6 Gaps — What's Missing (Updated)

1. ~~**No `MPIX_Comm_agree` usage**~~ — **FIXED** (commit `33c867c6`). `MPIContext::agree()` wraps `MPIX_Comm_agree` for survivor consensus.

2. **No `MPI_Comm_spawn` usage** — No ability to spawn replacement ranks into the communicator. *(Phase 3)*

3. ~~**No failure policy abstraction**~~ — **FIXED** (commit `2747ecf5`). `FailurePolicy` enum with `FAST_FAIL` / `FAULT_TOLERANT` modes.

4. **No hardware reset from MPI context** — No mechanism for surviving MPI ranks to trigger `tt-smi -r` on the failed node. *(Phase 2)*

5. ~~**No `MPI_Finalize` timeout**~~ — **FIXED** (commit `5bb8ec27`). SIGALRM watchdog + std::set_terminate handler.

6. ~~**`is_revoked()` implementation is suspect**~~ — **FIXED** (commit `3345ccce`). Uses `revoked_.load(std::memory_order_acquire)` on an atomic flag.

7. ~~**No Python-layer fault awareness**~~ — **PARTIALLY FIXED** (commit `2747ecf5`). `mpi_fault.py` provides `ulfm_guard` context manager with ULFM error detection. `ttrun.py` exit code interpretation still needs work. *(Phase 2)*

8. **No work redistribution** — After `revoke_and_shrink()`, the test code verifies the shrunken communicator works, but no mechanism exists to reassign the dead rank's work. *(Phase 3)*

---

## 3. Rank Reinitialization Architecture

### 3a. Hardware Reset

#### Current Reset Infrastructure

The repo already has the building blocks for remote hardware reset:

1. **`distributed_reset.sh`** (`tests/scale_out/4x_bh_quietbox/distributed_reset.sh`):
   ```bash
   parallel-ssh -i -H "sjc1-tt-qb-01 sjc1-tt-qb-02 sjc1-tt-qb-03 sjc1-tt-qb-04" \
       "cd /nfs/$USER/tt-smi && source .venv/bin/activate && \
        touch $BARRIER_DIR/$(hostname) && \
        while [ $(ls $BARRIER_DIR | wc -l) -lt 4 ]; do sleep 0.01; done && \
        tt-smi -r"
   ```

2. **`ensure-bh-links-online` action** (`.github/actions/ensure-bh-links-online/action.yml`):
   - Retry loop: `tt-smi -r` + `test_system_health`, up to 10 attempts with 5s sleep

3. **`validate_cluster_health.py`** (`tests/scale_out/4x_bh_quietbox/validate_cluster_health.py`):
   - Orchestrates reset + validation in a retry loop (MAX_RETRIES=10)

#### Proposed Remote Reset Architecture

When rank on node `f10cs04` crashes and surviving ranks on `mpirun-host` need to trigger device reset:

**Option A: SSH-based reset (recommended for Phase 1)**

The surviving rank (typically rank 0 or the lowest-numbered survivor) SSHes to the failed node and runs `tt-smi -r`. Prerequisites:
- Passwordless SSH between mpirun-host and all worker nodes (already required for mpirun itself)
- `tt-smi` installed and accessible on all nodes
- Knowledge of which hostname maps to which MPI rank (available from the rank binding YAML)

```
Surviving ranks:
  1. MPIX_Comm_agree() — all survivors agree a failure occurred
  2. Rank 0 (or designated "leader") SSHes to failed node:
     ssh f10cs04 "tt-smi -r && sleep 3 && tt-smi -r"  # double reset for link stability
  3. Leader broadcasts reset completion to survivors
  4. MPI_Comm_spawn() or reconnect respawned rank
```

**Option B: Out-of-band reset daemon (Phase 3 / aspirational)**

A lightweight daemon on each node listens for reset requests over a separate channel (ZMQ, gRPC, or simple TCP socket). This avoids SSH overhead and works even if the node's SSH daemon is unresponsive (but the OS is still up).

**Critical timing consideration:** Device reset via `tt-smi -r` takes 3-10 seconds for Wormhole, potentially 10-30 seconds for Blackhole. The `distributed_reset.sh` adds a 3-second stabilization delay. The `ensure-bh-links-online` action uses a 5-second delay between retries. Plan for **15-30 seconds minimum** for a reliable reset-and-reinitialize cycle.

**Synchronized reset problem:** When one node resets, it may disrupt Ethernet links to other nodes in the fabric. The existing `distributed_reset.sh` solves this by resetting ALL nodes simultaneously using an NFS barrier. For fault-tolerant operation, we need a more targeted approach:
- Reset only the failed node's devices
- Surviving nodes may need to re-establish fabric links after the failed node resets
- This may require running `test_system_health` or equivalent after reset

### 3b. Process Respawn

#### MPI_Comm_spawn Pattern

OpenMPI 5.x with ULFM supports `MPI_Comm_spawn`, but with important caveats:

```
Proposed flow:
  1. Detect failure (MPIX_ERR_PROC_FAILED from collective)
  2. MPIX_Comm_revoke(old_comm) — poison the old communicator
  3. MPIX_Comm_shrink(old_comm, &shrunk_comm) — get survivors-only communicator
  4. MPIX_Comm_agree(shrunk_comm, &decision) — all survivors agree on recovery action
  5. SSH to failed node: tt-smi -r (hardware reset)
  6. MPI_Comm_spawn(shrunk_comm, program, args, ..., &intercomm) — spawn replacement on failed node
  7. MPI_Intercomm_merge(intercomm, ..., &new_comm) — merge survivors + respawned into one communicator
  8. Continue with new_comm (size == original size)
```

**Challenges with MPI_Comm_spawn in tt-metal:**

1. **Environment setup:** The spawned process needs the full tt-metal environment (TT_MESH_ID, TT_MESH_HOST_RANK, TT_MESH_GRAPH_DESC_PATH, LD_LIBRARY_PATH, etc.). `MPI_Comm_spawn` supports passing `MPI_Info` keys for environment, but it's fragile.

2. **Device state:** The respawned process starts fresh — it needs to re-initialize its TT devices, which means going through the full device open/init sequence. The `MeshDevice` and `SystemMesh` initialization is not designed for this.

3. **Application state:** The respawned rank has no knowledge of the application's state (model weights loaded, pipeline stage, iteration count). This requires either:
   - Checkpointing (save state to shared storage)
   - Re-broadcasting from survivors
   - Starting the respawned rank from the beginning of the current work unit

4. **Rank numbering:** After `MPI_Intercomm_merge`, the respawned process may get a different rank number than the dead one. The application needs to handle rank remapping.

#### MPI_Comm_connect/accept Pattern (Alternative)

Instead of `MPI_Comm_spawn`, the respawned process could be launched externally (by ttrun.py or a supervisor) and connect back:

```
Proposed flow:
  1. Survivors detect failure, shrink communicator
  2. ttrun.py (or supervisor) detects rank exit, resets hardware, relaunches process
  3. Relaunched process publishes a "port" via MPI_Open_port() or a shared file
  4. Survivors call MPI_Comm_connect() to the new process's port
  5. New process calls MPI_Comm_accept()
  6. MPI_Intercomm_merge() to get unified communicator
```

This is **more practical for tt-metal** because:
- ttrun.py already manages the launch environment (all env vars, rank bindings)
- The relaunched process gets the correct environment automatically
- No need to pass complex MPI_Info to MPI_Comm_spawn

**However**, this requires MPI to still be operational after the failure, which is not guaranteed if MPI_Finalize has been called or the MPI runtime itself is corrupted.

#### Feasibility Assessment

```
Approach                          Feasibility  Complexity  Time to Implement
────────────────────────────────  ───────────  ──────────  ─────────────────
Shrink-only (no respawn)          High         Low         2-4 weeks
Shrink + external relaunch        Medium       Medium      6-8 weeks
  (ttrun.py supervisor loop)
Shrink + MPI_Comm_spawn           Low-Medium   High        8-12 weeks
Shrink + connect/accept           Low          High        10-14 weeks
```

**Recommendation: Start with shrink-only (Phase 1), then add external relaunch (Phase 2).**

### 3c. Work Redistribution

#### How tt-metal Distributes Work Across Ranks

In the current architecture, work distribution happens at two layers:

1. **C++ layer (`MPIContext`):** The `DistributedContext` provides the communication primitives. It does NOT assign work — it's a transport layer.

2. **Application/test layer:** Tests and applications decide how to partition work. The typical pattern:
   - Each rank owns a subset of devices (defined by rank binding YAML)
   - Each rank runs the same code (SPMD) operating on its local devices
   - Collectives (all_reduce, barrier) synchronize across ranks

3. **Mesh topology layer:** `MeshDevice` maps physical devices to a logical mesh grid. Each rank sees its portion of the mesh via `TT_MESH_ID` and `TT_MESH_HOST_RANK`.

#### Work Redistribution Patterns

**Pattern 1: Checkpoint/Restart (most practical)**

Before each "work unit" (iteration, batch, test case), surviving ranks checkpoint minimal state to shared storage (NFS). After shrink + optional respawn:
- If respawned: new rank loads checkpoint and joins at the current iteration
- If no respawn: surviving ranks pick up the dead rank's remaining work

This is the standard pattern for fault-tolerant distributed training and maps well to tt-metal's use cases.

**Pattern 2: Degraded-mode continuation**

After `revoke_and_shrink()`, the application continues with fewer ranks/devices. This works for:
- Tests that can tolerate partial results
- Inference workloads where throughput degrades but correctness is maintained
- Pipeline stages where the dead rank's stage can be skipped or absorbed

The `DisableBrokenBlock` test already demonstrates this pattern: surviving ranks split into healthy/unhealthy blocks and continue with only healthy blocks.

**Pattern 3: Work stealing (aspirational)**

A work queue shared via MPI or shared memory, where idle ranks pull work items. This requires significant application-level changes and is not recommended for Phase 1.

#### Which Layer Handles Work Assignment

- **For C++ tests:** Work assignment is in the test binary itself. The test framework (GTest + multihost fixtures) would need a "fault-tolerant test runner" mode.
- **For Python tests (pytest):** Work assignment is in the test script. A pytest fixture could provide the "continue after rank loss" behavior.
- **For training workloads:** Work assignment is in the training loop. The `tt-train` examples would need checkpoint/restart support.

---

## 4. Required Code Changes

### 4.1 Core MPI Context Changes

```
File: tt_metal/distributed/multihost/mpi_distributed_context.cpp
Change type: Modify existing + new functions
What:
  1. Fix is_revoked() — replace MPI_Comm_test_inter() with a proper revocation check.
     Use MPIX_Comm_is_revoked() if available in OpenMPI 5.0.7-ulfm, or attempt
     MPI_Comm_test_inter on a dup'd comm and check for MPI_ERR_REVOKED.
  2. Add agree() method — wraps MPIX_Comm_agree() for survivor consensus.
  3. Add MPI_Finalize watchdog — register a SIGALRM handler before MPI_Finalize()
     in the atexit callback; if MPI_Finalize doesn't complete within 30s, force exit.
  4. Add spawn_replacement() method — wraps MPI_Comm_spawn + MPI_Intercomm_merge
     for Phase 2 respawn capability.
  5. Add FailurePolicy enum and set_failure_policy() — allow callers to choose
     FAST_FAIL (current behavior) vs FAULT_TOLERANT (catch and recover).
Complexity: Medium
```

```
File: tt_metal/distributed/multihost/mpi_distributed_context.hpp
Change type: Modify existing
What:
  1. Add agree() declaration
  2. Add spawn_replacement() declaration
  3. Add FailurePolicy enum: { FAST_FAIL, FAULT_TOLERANT }
  4. Add set_failure_policy() / get_failure_policy()
  5. Add failed_ranks() — returns list of ranks known to be dead
Complexity: Low
```

```
File: tt_metal/api/tt-metalium/distributed_context.hpp
Change type: Modify existing
What:
  1. Add virtual agree() to DistributedContext interface
  2. Add FailurePolicy to the interface
  3. Add virtual failed_ranks()
Complexity: Low
```

```
File: tt_metal/distributed/multihost/single_host_context.cpp
Change type: Modify existing
What:
  1. Stub out new methods (agree(), spawn_replacement(), etc.) with appropriate
     error messages or no-ops
Complexity: Low
```

### 4.2 ttrun.py Changes

```
File: ttnn/ttnn/distributed/ttrun.py
Change type: Modify existing
What:
  1. Interpret mpirun exit codes: MPI rank failure produces specific exit codes
     (typically 1-255, with signal numbers encoded). Map these to structured
     output (which rank failed, signal, etc.).
  2. Add --fault-tolerant flag: When set, ttrun.py enters a supervisor loop:
     - Launch mpirun
     - On non-zero exit, parse output for rank failure info
     - Optionally reset hardware on failed node (SSH + tt-smi -r)
     - Optionally relaunch the failed rank only
  3. Add --max-retries option for the supervisor loop
  4. Add structured exit code conventions:
     - 0: success
     - 1: application error (all ranks ran, test failed)
     - 2: infrastructure error (rank died)
     - 130: interrupted (SIGINT)
     - 137: rank killed by OOM (SIGKILL = 128+9)
Complexity: Medium-High
```

### 4.3 New Files

```
File: tt_metal/distributed/multihost/fault_policy.hpp (NEW)
Change type: New file
What:
  Define the FailurePolicy enum, a FaultContext structure (tracking which ranks
  are dead, recovery state), and a FaultHandler interface:
    class FaultHandler {
      virtual void on_rank_failure(Rank failed, const ContextPtr& ctx) = 0;
      virtual void on_recovery_complete(const ContextPtr& new_ctx) = 0;
    };
Complexity: Medium
```

```
File: tt_metal/distributed/multihost/hardware_reset.hpp (NEW)
File: tt_metal/distributed/multihost/hardware_reset.cpp (NEW)
Change type: New files
What:
  Encapsulate remote hardware reset logic:
    class HardwareResetter {
      // Maps rank -> hostname (populated from rank binding config)
      void reset_node(const std::string& hostname);
      bool wait_for_health(const std::string& hostname, int timeout_s);
    };
  Implementation: fork+exec SSH to target node, run tt-smi -r, wait, run health check.
Complexity: Medium
```

```
File: ttnn/ttnn/distributed/mpi_fault.py (NEW)
Change type: New file
What:
  Python-layer fault handling utilities:
    - parse_mpi_failure_output(stdout, stderr) -> FailureInfo
    - interpret_exit_code(returncode) -> ExitReason
    - class RankSupervisor: manages relaunch of individual ranks
Complexity: Medium
```

### 4.4 Test Infrastructure Changes

```
File: tests/tt_metal/multihost/fault_tolerance_tests/ulfm_tests.cpp
Change type: Modify existing
What:
  1. Add test for agree() consensus
  2. Add test for MPI_Finalize watchdog (verify process doesn't hang)
  3. Add test for FailurePolicy switching
  4. Add test for spawn_replacement() (Phase 2)
Complexity: Medium
```

```
File: tests/tt_metal/multihost/common/multihost_test_tools.hpp
Change type: Modify existing
What:
  1. Add fault-tolerant test macros: EXPECT_EQ_SURVIVING_RANKS (like EXPECT_EQ_ALL_RANKS
     but accounts for dead ranks)
  2. Add skip_if_no_ulfm() helper
Complexity: Low
```

```
File: tests/tt_metal/multihost/run_fault_tolerance_tests.sh
Change type: Modify existing
What:
  Add new test invocations for agree(), watchdog, and policy tests
Complexity: Low
```

### 4.5 Workflow Changes

```
File: .github/workflows/multi-host-physical.yaml
Change type: Modify existing (future)
What:
  Add a fault-tolerance test job that runs the ULFM tests on physical multi-host
  hardware. Currently fault tolerance tests are only run with run_fault_tolerance_tests.sh
  which uses -np 8 on a single node. For proper validation, they need to run
  across actual separate hosts.
Complexity: Medium (requires dedicated multi-host runner time)
```

```
File: .github/actions/ensure-bh-links-online/action.yml
Change type: No change needed
What:
  This action already implements the retry loop pattern (tt-smi -r + health check
  up to 10 times). The hardware_reset.cpp module should follow the same pattern.
Complexity: N/A
```

### 4.6 CMake Changes

```
File: tt_metal/distributed/CMakeLists.txt
Change type: Modify existing
What:
  1. Add new source files (fault_policy.hpp, hardware_reset.cpp) to the build
  2. No new dependencies needed — SSH exec is via fork/exec, no library dependency
Complexity: Low
```

---

## 5. Recommended Implementation Phases

### Phase 1: Defensive Foundations (2-4 weeks)

**Goal:** Make the existing system more robust without adding respawn capability.

1. **Fix `is_revoked()`** — Replace the `MPI_Comm_test_inter` hack with a correct implementation.

2. **Add `MPIX_Comm_agree()` wrapper** — Essential for survivors to reach consensus before taking recovery actions. Without this, survivors may disagree on what happened and take inconsistent actions.

3. **Add `MPI_Finalize` watchdog** — Register a SIGALRM in the `atexit` handler. If `MPI_Finalize()` doesn't complete in 30 seconds, call `_exit(1)`. This prevents zombie processes in CI.

4. **Add `FailurePolicy` enum** — Allow code to switch between fast-fail (throw immediately) and fault-tolerant (return error code, let caller decide).

5. **Improve ttrun.py exit code handling** — Parse mpirun exit codes to distinguish "test failed" from "rank died." Report structured failure info.

6. **Add new ULFM tests** — Test agree(), watchdog, and policy switching.

**Impact:** CI processes stop hanging when a rank dies. Developers get clear error messages instead of cryptic MPI errors.

### Phase 2: Shrink-and-Continue + Supervised Relaunch (4-8 weeks)

**Goal:** Enable tests to continue after rank loss, and add supervised process relaunch.

1. **`hardware_reset.hpp/.cpp`** — Remote reset capability (SSH + tt-smi -r) triggered by surviving ranks.

2. **`mpi_fault.py`** — Python fault handling library for ttrun.py.

3. **ttrun.py supervisor mode** — `--fault-tolerant` flag that enters a retry loop on rank failure:
   - Detect which rank failed
   - Reset hardware on the failed node
   - Relaunch the entire job (simple restart) or just the failed rank (advanced)

4. **Shrink-and-continue test pattern** — A GTest fixture that wraps the `catch DistributedException -> revoke_and_shrink()` pattern, allowing tests to opt into degraded-mode continuation.

5. **`DisableBrokenBlock` pattern as a library** — Extract the block-disabling logic from `ulfm_tests.cpp` into a reusable utility.

**Impact:** CI jobs recover from transient rank failures automatically. Tests can be marked as "degradable" and produce partial results instead of total failure.

### Phase 3: Rank Respawn + Full Work Redistribution (8-16 weeks, aspirational)

**Goal:** Spawn replacement ranks and redistribute work.

1. **`MPI_Comm_spawn` integration** — Spawn a replacement process on the recovered node, merge it into the communicator.

2. **Rank state restoration** — Checkpoint/restore mechanism for the respawned rank to catch up:
   - Broadcast model weights from survivors
   - Replay work queue from checkpoint
   - Or: simply restart the work unit from the beginning on all ranks

3. **Work queue abstraction** — A shared work queue that surviving ranks can pull from, allowing dynamic rebalancing.

4. **Out-of-band reset daemon** — Replace SSH-based reset with a lightweight daemon for faster, more reliable remote reset.

5. **Python-layer fault-tolerant fixtures** — pytest fixtures that enable `@pytest.mark.fault_tolerant` for Python multihost tests.

**Impact:** True fault-tolerant execution — a rank dies, gets replaced, work continues with minimal delay. This is the end state for production-grade multihost deployment.

---

## 6. Ergonomics and Usability

### 6.1 How Test Authors Opt Into Fault Tolerance

**C++ tests (GTest):**

```cpp
// Option 1: Environment variable
// Set TT_FAULT_TOLERANT=1 before running
// The test fixture checks this and wraps collectives in try/catch

// Option 2: Test fixture base class
class FaultTolerantTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = DistributedContext::get_current_world();
        ctx_->set_failure_policy(FailurePolicy::FAULT_TOLERANT);
    }
    // Provides: safe_barrier(), safe_allreduce(), etc.
    // These catch DistributedException and call revoke_and_shrink()
};

// Option 3: GTest parameterization
// INSTANTIATE_TEST_SUITE_P(FaultTolerant, MyTest,
//     ::testing::Values(FailurePolicy::FAST_FAIL, FailurePolicy::FAULT_TOLERANT));
```

**Recommendation:** Option 2 (fixture base class) for Phase 1, Option 3 for Phase 2.

**Python tests (pytest):**

```python
# Option 1: Environment variable
# TT_FAULT_TOLERANT=1 pytest tests/multihost/

# Option 2: Decorator (Phase 3)
@pytest.mark.fault_tolerant
def test_my_distributed_op():
    ...

# Option 3: Fixture
@pytest.fixture
def fault_tolerant_context():
    ctx = ttnn.distributed.get_context()
    ctx.set_failure_policy("fault_tolerant")
    yield ctx
    ctx.set_failure_policy("fast_fail")
```

### 6.2 API Surface for "My Test Supports Rank Loss"

The minimal API a test author needs:

```cpp
// C++ — check if we lost ranks
if (ctx->size() < expected_world_size) {
    // We lost ranks — adjust test expectations
    GTEST_SKIP() << "Rank loss detected, skipping validation";
    // OR: adjust expected results for fewer ranks
}

// C++ — query which ranks failed
auto dead = ctx->failed_ranks();  // returns vector<Rank>
```

```python
# Python — check if we lost ranks
world = ttnn.distributed.get_context()
if world.size < expected_size:
    pytest.skip("Rank loss detected")
```

### 6.3 How ttrun.py Surfaces Rank-Loss Events to CI

**Proposed output format:**

```
[tt-run] RANK FAILURE DETECTED
[tt-run]   Failed rank: 3
[tt-run]   Signal: SIGKILL (9)
[tt-run]   Node: f10cs04
[tt-run]   Time: 2026-03-21T14:32:01Z
[tt-run]   Action: hardware reset + relaunch (attempt 1/3)
```

**Structured exit codes:**

```
Exit Code  Meaning
─────────  ──────────────────────────────
0          All ranks completed successfully
1          Application error (test assertion failed, all ranks alive)
2          Rank failure — at least one rank died (infrastructure error)
3          Rank failure — recovery attempted but failed
130        Interrupted (SIGINT / Ctrl+C)
```

These exit codes allow CI to distinguish "test bug" (exit 1) from "infrastructure flake" (exit 2), enabling smarter retry policies.

### 6.4 Environment Variables

```
Variable                     Purpose                              Default
──────────────────────────   ──────────────────────────────────   ───────
TT_FAULT_TOLERANT            Enable fault-tolerant mode            0
TT_FAULT_MAX_RETRIES         Max rank-failure retries              3
TT_FAULT_RESET_TIMEOUT_S     Hardware reset timeout (seconds)      30
TT_FAULT_FINALIZE_TIMEOUT_S  MPI_Finalize watchdog timeout (s)     30
```

---

## 7. Risks and Limitations

### 7.1 ULFM's Own Limitations

1. **`MPI_ANY_SOURCE` with dead ranks:** Using `MPI_ANY_SOURCE` in receive operations can block indefinitely if the expected sender is dead. ULFM eventually detects this and returns `MPIX_ERR_PROC_FAILED`, but the timeout can be long (minutes). **Current codebase status:** `MPI_ANY_SOURCE` is NOT used in `mpi_distributed_context.cpp` — all recv/irecv take explicit source ranks. This is good.

2. **Non-blocking operations:** Outstanding `MPI_Isend`/`MPI_Irecv` to dead ranks will fail when tested/waited. The `MPIRequest::wait()` and `MPIRequest::test()` methods throw `MPIDistributedException` on error, which is the correct behavior.

3. **Communicator revocation is irreversible:** Once `MPIX_Comm_revoke()` is called, the communicator is permanently poisoned. All pending operations on it fail. The only path forward is `MPIX_Comm_shrink()` to get a new communicator.

4. **`MPIX_Comm_shrink()` is a collective:** ALL surviving ranks must call it. If one survivor doesn't (e.g., it's in a different code path), the shrink will hang. This requires careful coordination, which is why `MPIX_Comm_agree()` is essential.

5. **Multiple simultaneous failures:** ULFM can handle multiple ranks failing simultaneously, but `MPIX_Comm_shrink()` may need to be called multiple times if new failures occur during recovery.

### 7.2 Hardware Reset Timing

- `tt-smi -r` takes 3-10 seconds for Wormhole devices
- Blackhole devices may take longer (observed 10-30 second range in CI)
- After reset, Ethernet links need time to stabilize (3-5 seconds)
- Total realistic reset-and-reinit cycle: **15-45 seconds**
- During this time, surviving ranks are blocked waiting
- This is acceptable for fault-tolerant test runs but may be too slow for production inference

### 7.3 Network Partition vs Dead Rank

ULFM cannot distinguish between "rank's node crashed" and "network partition isolating the rank's node":

- **Dead rank:** Node is down, process is gone. Reset and respawn is appropriate.
- **Partitioned rank:** Node is up, process is running, but unreachable. Resetting the node could corrupt in-flight device operations.

**Mitigation:** Before resetting, attempt to SSH to the node and check if the process is still running. If yes, it's likely a partition — try to kill the orphaned process before resetting.

### 7.4 State Consistency After Rank Loss

When a rank dies mid-operation:
- **Device state:** The TT devices on the failed node may have in-flight operations, corrupted buffers, or hung firmware. `tt-smi -r` is the only reliable way to clear this.
- **Fabric state:** The TT Fabric connections to the dead node's devices will be broken. Surviving nodes' Fabric state needs to be updated to exclude the dead node.
- **Application state:** Any in-progress computation on the dead rank is lost. The application must either:
  - Restart the entire work unit
  - Use checkpointing to resume from a known-good state
  - Accept partial results (degraded mode)

### 7.5 MPI_Finalize Hang Risk

**STATUS: FIXED** (commit `5bb8ec27`)

The atexit handler now arms a 30-second SIGALRM watchdog before calling `MPI_Finalize()`. If it does not complete in time, `mpi_finalize_alarm_handler` writes a diagnostic to stderr and calls `_exit(70)`. Additionally, `std::set_terminate` is installed to revoke `MPI_COMM_WORLD` and `_exit(70)` on uncaught exceptions, preventing hangs from non-MPI failures (ESTALE, OOM, etc.).

### 7.6 Thread Safety of revoke_and_shrink()

**STATUS: FIXED** (commit `33c867c6`)

A `mutable std::mutex comm_mutex_` has been added to `MPIContext`. The `revoke_and_shrink()` method holds `comm_mutex_` during the critical section where it frees the old communicator and updates `comm_`, `group_`, `rank_`, `size_`. This prevents data races with concurrent reads.

Note: A `std::shared_mutex` was considered but rejected because MPI operations are not re-entrant and the contention profile does not benefit from shared locking. A plain `std::mutex` protecting only the mutation (not every MPI call) is the minimal safe approach.

### 7.7 OpenMPI 5.x ULFM Maturity

The ULFM implementation in OpenMPI 5.0.7 (the version tt-metal uses via `tenstorrent/ompi` releases) is relatively mature — ULFM has been in development since ~2012 and was formally standardized in the MPI 4.1 proposal. However:

- `MPI_Comm_spawn` with ULFM may have edge cases
- `MPIX_Comm_agree` is critical and well-tested
- `MPIX_Comm_shrink` is the most reliable recovery primitive
- The `--with-ft ulfm` runtime flag is required to enable fault detection

### 7.8 Single-Node Testing Gap

The existing ULFM tests run with `-np 8` on a **single node** (no `--hostfile`). This tests the MPI-level fault tolerance but does NOT test:
- Cross-node failure detection
- Network-level failure modes
- Remote hardware reset
- NFS-related failures

Phase 2 should include multi-host ULFM tests on physical hardware.

---

## Appendix A: File Reference

All file paths verified against the repository:

### Core MPI Implementation
- `tt_metal/distributed/multihost/mpi_distributed_context.cpp` — MPI context implementation with ULFM
- `tt_metal/distributed/multihost/mpi_distributed_context.hpp` — MPI context header
- `tt_metal/distributed/multihost/distributed_context.cpp` — Factory that selects MPI vs single-host
- `tt_metal/distributed/multihost/single_host_context.cpp` — Non-MPI fallback
- `tt_metal/distributed/multihost/single_host_context.hpp` — Non-MPI fallback header
- `tt_metal/api/tt-metalium/distributed_context.hpp` — Abstract interface for distributed contexts

### Build Configuration
- `tt_metal/distributed/CMakeLists.txt` — ULFM MPI detection and linking
- `tt_metal/CMakeLists.txt` — RPATH configuration for ULFM library
- `install_dependencies.sh` — ULFM installation from `tenstorrent/ompi` .deb package

### Test Infrastructure
- `tests/tt_metal/multihost/fault_tolerance_tests/ulfm_tests.cpp` — Existing ULFM tests
- `tests/tt_metal/multihost/fault_tolerance_tests/main.cpp` — Test main
- `tests/tt_metal/multihost/fault_tolerance_tests/CMakeLists.txt` — Test build
- `tests/tt_metal/multihost/common/multihost_test_tools.hpp` — Test utilities
- `tests/tt_metal/multihost/run_fault_tolerance_tests.sh` — Test runner
- `tests/tt_metal/multihost/mpirun_wrapper.sh` — mpirun-ulfm finder

### Process Launch
- `ttnn/ttnn/distributed/ttrun.py` — MPI process launcher (tt-run)

### Hardware Reset
- `tests/scale_out/4x_bh_quietbox/distributed_reset.sh` — Synchronized multi-node reset via parallel-ssh
- `tests/scale_out/4x_bh_quietbox/validate_cluster_health.py` — Reset + validation retry loop
- `.github/actions/ensure-bh-links-online/action.yml` — BH device reset + health check action

### CI Workflows
- `.github/workflows/multi-host-physical.yaml` — Multi-host test workflow (T3K, Galaxy)
- `.github/workflows/ttnn-run-sweeps.yaml` — Uses tt-smi reset
- `tests/scripts/multihost/run_dual_t3k_tests.sh` — Dual T3K test suite
- `tests/scripts/multihost/run_dual_galaxy_tests.sh` — Dual Galaxy test suite
- `tests/scripts/multihost/setup_shared_venv.sh` — Multi-host venv setup with race condition handling

### Proposed New Files
- `tt_metal/distributed/multihost/fault_policy.hpp` — FailurePolicy enum and FaultHandler interface
- `tt_metal/distributed/multihost/hardware_reset.hpp` — Remote hardware reset abstraction
- `tt_metal/distributed/multihost/hardware_reset.cpp` — Remote hardware reset implementation
- `ttnn/ttnn/distributed/mpi_fault.py` — Python fault handling utilities

---

## Appendix B: ULFM API Quick Reference

```
Function                          Purpose
────────────────────────────────  ─────────────────────────────────────────────
MPIX_Comm_revoke(comm)            Poison communicator — all ops on it will fail
MPIX_Comm_shrink(comm, &new)      Create new comm excluding dead ranks (collective)
MPIX_Comm_agree(comm, &flag)      All survivors agree on a boolean value (collective)
MPIX_ERR_PROC_FAILED              Error code: operation failed because peer is dead
MPIX_ERR_REVOKED                  Error code: communicator was revoked
MPI_Comm_spawn(...)               Spawn new process(es) and get intercommunicator
MPI_Intercomm_merge(inter, &intra) Merge parent+child into single communicator
MPI_Comm_connect/accept(...)      Connect independently-launched processes
```

Runtime flags for OpenMPI ULFM:
- `--with-ft ulfm` — Enable ULFM fault detection at runtime
- `mpirun-ulfm` — The ULFM-enabled mpirun binary (at `/opt/openmpi-v5.0.7-ulfm/bin/mpirun-ulfm`)

---

## 8. Port Laddering, Inspector, Watcher, and Triage Interactions

This section documents how the Inspector RPC port laddering, the Watcher server, and the triage toolchain interact — and how these intersect with the ULFM fault-tolerance architecture on branch `nsexton/0-multihost-triage`.

### 8.1 What Port Laddering Is and Why It Was Needed

**Problem:** In multihost MPI deployments, each rank runs its own Metal runtime, which starts an Inspector RPC server (Cap'n Proto over TCP). Without port laddering, every rank on the same physical host attempts to bind to the same default port (50051), causing `bind()` failures (`EADDRINUSE`). This crashes ranks at startup — exactly the class of failure ULFM is designed to handle, but one that should never happen in the first place.

**Solution — rank-aware port offset:**

The effective Inspector RPC port is computed as:

```
effective_port = base_port + mpi_rank
```

where `base_port` defaults to **50051** and `mpi_rank` is read from the process environment at `RunTimeOptions` construction time.

**C++ implementation** (`tt_metal/llrt/rtoptions.cpp`, lines 444-459):

```cpp
uint16_t RunTimeOptions::get_effective_inspector_rpc_server_port() const {
    int rank = this->cached_mpi_rank_;
    uint32_t base = inspector_settings.rpc_server_port;
    if (rank >= 0) {
        uint32_t port = base + static_cast<uint32_t>(rank);
        if (port > 65535) {
            TT_THROW("Inspector RPC port overflow: base_port={} + rank={} ...", base, rank);
        }
        return static_cast<uint16_t>(port);
    }
    return static_cast<uint16_t>(base);
}
```

**Python implementation** (`tools/triage/inspector_data.py`, lines 169-217):

The triage tool mirrors the same logic via `_get_rank_from_env()` and applies the offset when the default port (50051) is in use:

```python
_DEFAULT_INSPECTOR_RPC_PORT = 50051

def _get_rank_from_env() -> int:
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "PMIX_RANK", "TT_MESH_HOST_RANK"):
        val = os.environ.get(var)
        if val is not None:
            try: return int(val)
            except: continue
    return -1
```

**Key properties:**

- The offset is **global across all MPI ranks**, not per-node. Rank 3 always gets port 50054 regardless of which host it runs on. This is safe because ranks on different hosts bind to different network interfaces; on the same host the offset prevents collisions.
- The rank is cached once at `RunTimeOptions` construction (`cached_mpi_rank_`, line 390 in `rtoptions.cpp`) and never re-read — thread-safe by design.
- The rank detection precedence is identical in C++ (`get_rank_from_env()` at line 319) and Python (`_get_rank_from_env()` at line 172 of `inspector_data.py`): `OMPI_COMM_WORLD_RANK` > `PMI_RANK` > `SLURM_PROCID` > `PMIX_RANK` > `TT_MESH_HOST_RANK`.
- The base port can be overridden via `TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS` (env var, parsed at line 1271 of `rtoptions.cpp`).
- Default: `inspector_settings.rpc_server_port = 50051` (defined in `rtoptions.hpp`, line 113).

### 8.2 The Watcher/Inspector Architecture

The Watcher and Inspector are two distinct but complementary debug subsystems in tt-metal. They serve different purposes and have different failure semantics.

#### 8.2.1 Inspector

The Inspector is a **software-level instrumentation system** that records program/kernel metadata, dispatch events, and mesh workload information. It exposes this data via a Cap'n Proto RPC server so that the triage tool can query it live.

Key files:
- `tt_metal/impl/debug/inspector/inspector.hpp` — Static API (`Inspector::is_enabled()`, `Inspector::serialize_rpc()`, etc.)
- `tt_metal/impl/debug/inspector/inspector.cpp` — Implementation; delegates to `inspector::Data`
- `tt_metal/impl/debug/inspector/rpc_server_controller.cpp` — Cap'n Proto RPC server lifecycle (start/stop/poll loop)
- `tt_metal/impl/debug/inspector/data.hpp` / `data.cpp` — In-memory data store + serialization
- `tt_metal/api/tt-metalium/experimental/inspector.hpp` — Public API header

The Inspector RPC server runs in a dedicated thread (`RpcServerController::run_server()`, line 71 of `rpc_server_controller.cpp`). It uses Cap'n Proto's two-party RPC protocol with a KJ async event loop. The server polls for stop requests and sleeps 1ms between events when idle.

**Enabled by:** `TT_METAL_INSPECTOR=1` + `TT_METAL_INSPECTOR_RPC=1` environment variables.

**Port binding:** Uses `get_effective_inspector_rpc_server_port()` (the port-laddered value) to bind. If binding fails, `RpcServerController::start()` throws `std::runtime_error` (line 46 of `rpc_server_controller.cpp`).

#### 8.2.2 Watcher

The Watcher is a **hardware-level polling system** that periodically reads device mailboxes to detect asserts, NOC sanitization violations, paused cores, link status issues, and other firmware-level problems.

Key files:
- `tt_metal/impl/debug/watcher_server.hpp` — `WatcherServer` class (pimpl pattern)
- `tt_metal/impl/debug/watcher_server.cpp` — Full implementation: `poll_watcher_data()` loop, `Dump()`, log file management
- `tt_metal/impl/debug/watcher_device_reader.hpp` / `.cpp` — Per-device, per-core data reading and interpretation
- `tt_metal/tools/watcher_dump/watcher_dump.cpp` — Standalone dump tool (reads watcher data without the server)

The Watcher server runs in a dedicated thread (`poll_watcher_data()`, line 513 of `watcher_server.cpp`). It wakes at a configurable interval (`TT_METAL_WATCHER_INTERVAL`), acquires `watch_mutex_`, reads all device mailboxes, writes findings to the watcher log file, and checks for assert/pause/NOC errors.

**Critical distinction from Inspector:** The Watcher does **not** use any network ports. It communicates with devices via direct memory-mapped I/O (through UMD). It writes to a local log file (`generated/watcher/watcher.log`). There is no port laddering concern for the Watcher itself.

**Enabled by:** `TT_METAL_WATCHER=1` (defaults to on in debug builds, off in release).

#### 8.2.3 How They Relate

```
Inspector                                 Watcher
─────────                                ────────
Software metadata (programs, kernels)     Hardware state (asserts, NOC, waypoints)
Cap'n Proto RPC server (TCP port)         Local file I/O only
Port-laddered (base + rank)               No port needed
Queried by triage tool                    Log file parsed by triage tool
Records dispatch events                   Detects firmware-level hangs
```

Both are initialized by `MetalContext` and run as background threads within the Metal runtime process. The Inspector can reference Watcher kernel IDs via `Inspector::get_kernel_path_from_watcher_kernel_id()` (line 98 of `inspector.hpp`).

### 8.3 Failure Detection Chain: Watcher → Dispatch Timeout → Triage → ULFM

The chain from hardware error detection to ULFM-level response involves multiple layers:

#### Step 1: Watcher Detects Error

The `WatcherDeviceReader::Core::Dump()` method (line 546 of `watcher_device_reader.cpp`) reads device mailboxes and checks:
- Assert status (`DumpAssertStatus()`)
- NOC sanitize status (`DumpNocSanitizeStatus()`)
- Pause flags (`DumpPauseStatus()`)
- L1 corruption (`DumpL1Status()`)
- Ethernet link status (`DumpEthLinkStatus()`)

If a check fails (e.g., assert tripped, NOC violation), it throws `std::runtime_error`. In test mode (`get_test_mode_enabled()`), the watcher catches this, sets `server_killed_due_to_error_ = true`, and breaks from the poll loop (lines 547-554 of `watcher_server.cpp`). In non-test mode, the exception propagates and crashes the process.

#### Step 2: Dispatch Timeout Detection

Separately, when the dispatch system (command queue) times out waiting for a completion event, `SystemMemoryManager` calls `MetalContext::on_dispatch_timeout_detected()` (line 560 of `system_memory_manager.cpp`).

This method (`metal_context.cpp`, lines 715-740) does two things:
1. **Serializes Inspector RPC data** to disk if `serialize_on_dispatch_timeout` is enabled (default: `true`, `rtoptions.hpp` line 115). This writes binary Cap'n Proto snapshots to `<logs_dir>/generated/inspector/` so triage can load them even after the process dies.
2. **Executes a user-specified command** from `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` — typically `./tools/tt-triage.py`. This launches the triage tool in-process via `std::system()`.

#### Step 3: Triage Tool Connects to Inspector

When triage runs (either manually or via dispatch timeout command), `inspector_data.py` attempts to connect to the Inspector RPC server in this fallback order:
1. **Live RPC connection** — connects to `host:effective_port` (port-laddered)
2. **Serialized RPC data** — loads binary Cap'n Proto files from the inspector log directory
3. **Parsed inspector logs** — last resort, text log parsing

The triage tool then runs analysis scripts (`run_checks.py`, `dispatcher_data.py`, etc.) that depend on Inspector data.

#### Step 4: How This Relates to ULFM

Currently, the Watcher and Inspector are **per-rank, local-only** systems. There is **no cross-rank failure notification** from Watcher/Inspector to ULFM:

- If rank 2's watcher detects an assert, rank 2 may crash (exception propagation) or log the error (test mode). But rank 0 and rank 1 learn about rank 2's death only through:
  - **ULFM failure detection** — MPI operations on the surviving ranks return `MPIX_ERR_PROC_FAILED`, triggering `handle_rank_failure()` in `mpi_distributed_context.cpp` (line 154).
  - **PRRTE/orte abort propagation** — if `prte_abort_on_non_zero_status=1` (set by ttrun.py via `_get_abort_on_failure_mca_param()`), the MPI launcher terminates all ranks when one exits non-zero.

The watcher has no mechanism to proactively signal other MPI ranks. It is a purely local diagnostic tool.

### 8.4 ttrun.py Interaction with Watcher/Inspector

`ttnn/ttnn/distributed/ttrun.py` is the MPI process launcher. Its interactions with the watcher/inspector subsystems are **indirect** — through environment variable propagation and process management.

#### 8.4.1 Environment Setup

ttrun.py sets up rank-scoped paths for each MPI rank:

```python
RANK_SCOPED_PATH_ENV_VARS = frozenset({"TT_METAL_LOGS_PATH", "TT_METAL_JIT_SCRATCH"})
```

`apply_rank_scoped_paths()` (line 562) appends `<hostname>_rank_<N>` to these paths, creating per-rank directories:

```
/shared/logs/myhost_rank_0/generated/inspector/
/shared/logs/myhost_rank_1/generated/inspector/
```

This rank scoping is **critical for port laddering correctness** — each rank's Inspector writes its serialized data to its own rank-scoped directory, and the triage tool uses the same rank detection logic to find the right directory.

#### 8.4.2 ULFM Launcher Selection

ttrun.py prefers `mpirun-ulfm` over plain `mpirun` (`build_mpi_command()`, line 768):

```python
mpi_launcher = shutil.which("mpirun-ulfm")
if not mpi_launcher:
    mpi_launcher = "mpirun"
```

When using `mpirun-ulfm`, it adds `--with-ft ulfm` to enable ULFM fault detection (line 777).

#### 8.4.3 What ttrun.py Does NOT Do

- Does **not** set `TT_METAL_INSPECTOR=1` or `TT_METAL_INSPECTOR_RPC=1` — the user must enable these.
- Does **not** parse watcher output or monitor Inspector RPC health.
- Does **not** set `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` — this must be configured by the user or CI scripts.
- Does **not** know about Inspector ports — the port laddering is entirely handled within the Metal runtime and triage tool.
- Does **not** handle timeout-triggered triage. It has a `TT_RUN_TIMEOUT` wall-clock kill mechanism (lines 1326-1335) but this just kills the entire process group — no triage is run.

#### 8.4.4 XIP Dump Disable

ttrun.py disables XIP ELF dumps by default in multihost mode (line 690):

```python
if "TT_METAL_DISABLE_XIP_DUMP" not in env:
    env["TT_METAL_DISABLE_XIP_DUMP"] = "1"
```

This prevents ESTALE NFS errors when multiple ranks write `.xip.elf` files simultaneously — another class of infrastructure crash that predated the port laddering fix.

### 8.5 Implications for ULFM Architecture

#### 8.5.1 Port Laddering Prevents a Class of Startup Crashes

Before port laddering, multihost startup was fragile: if Inspector RPC was enabled and multiple ranks landed on the same host, the `EADDRINUSE` crash would cascade through the MPI job. Port laddering eliminates this, making Inspector-enabled multihost runs viable.

#### 8.5.2 No Watcher-to-ULFM Bridge Exists

The biggest architectural gap: the Watcher detects hardware-level problems (asserts, NOC violations, link failures) that would be extremely valuable for ULFM decision-making, but this information **never reaches the ULFM layer**. Currently:

```
Watcher detects assert → throws exception → rank crashes → ULFM detects dead rank
```

A more intelligent flow would be:

```
Watcher detects assert → notifies local ULFM handler → ULFM decides: revoke+shrink or fast-fail
```

This would allow FAULT_TOLERANT mode to make informed decisions (e.g., "rank 2 had a firmware assert on core [1,3] — this is recoverable after device reset" vs. "rank 2 had an L1 corruption — this device is broken").

#### 8.5.3 Inspector Serialization is One-Shot

`on_dispatch_timeout_detected()` sets `dispatch_timeout_detection_processed_ = true` and never resets it (line 719 of `metal_context.cpp`). After a timeout, no subsequent timeout will trigger serialization. In a FAULT_TOLERANT ULFM scenario where a rank survives and continues work after `revoke_and_shrink()`, this flag would need to be reset.

#### 8.5.4 Triage Assumes Live Process

The triage tool's RPC connection path assumes the target rank's Metal runtime is still running. In ULFM FAST_FAIL mode, the process calls `_exit(70)` immediately — no time for triage. In FAULT_TOLERANT mode, the process survives but may be mid-recovery. The serialized-data fallback path is the reliable option for post-mortem analysis.

### 8.6 Required Changes for Robust Multihost Triage

#### 8.6.1 Reset Dispatch Timeout Flag After Recovery

**File:** `tt_metal/impl/context/metal_context.cpp` (line 719)
**Change:** After a successful `revoke_and_shrink()`, reset `dispatch_timeout_detection_processed_ = false` so the serialization/triage pipeline can fire again on a subsequent timeout.
**Complexity:** Low — single boolean reset, but need to ensure thread safety with `dispatch_timeout_detection_mutex_`.

#### 8.6.2 Watcher Error Reporting to ULFM Layer

**Files:**
- `tt_metal/impl/debug/watcher_server.cpp` (lines 547-553, error handling in `poll_watcher_data`)
- `tt_metal/distributed/multihost/mpi_distributed_context.cpp` (ULFM handlers)

**Change:** Instead of throwing `std::runtime_error` that crashes the process, the Watcher could set a "hardware fault detected" flag that the ULFM handler reads. In FAULT_TOLERANT mode, this would allow the rank to participate in `agree()` with useful diagnostic information before `revoke_and_shrink()`.

**Complexity:** Medium — requires a new callback/flag interface between WatcherServer and the distributed context layer. The watcher runs in its own thread, so the flag must be atomic or mutex-protected.

#### 8.6.3 Automatic Inspector Enable in ttrun.py

**File:** `ttnn/ttnn/distributed/ttrun.py` (around line 690, in `get_rank_environment()`)
**Change:** Consider auto-setting `TT_METAL_INSPECTOR=1` and `TT_METAL_INSPECTOR_RPC=1` in multihost mode, or at minimum logging a warning when they are not set.
**Complexity:** Low — but may impact performance (Inspector has non-trivial overhead).

#### 8.6.4 Triage Summary for CI Hang Reports

**File:** `tools/triage/triage.py` (the `--triage-summary-path` flag, line 26)
**Change:** Ensure the triage summary file includes the MPI rank that generated it and the port used. In multihost CI, multiple ranks may write triage summaries simultaneously — they need distinct filenames (rank-scoped).
**Complexity:** Low — the flag already exists, just needs rank-aware filename generation.

#### 8.6.5 Port Overflow Guard for Large Clusters

**File:** `tt_metal/llrt/rtoptions.cpp` (line 449)
**Change:** The current guard throws on overflow (`base_port + rank > 65535`). For clusters with >15,000 ranks (unlikely but possible with Slurm), the base port should be configurable to a much lower value. Consider logging a warning at startup when `base_port + world_size` would overflow.
**Complexity:** Low — add a check at `RunTimeOptions` construction using the MPI world size if available.

<!-- TODO: verify — need to check whether MetalContext has access to MPI world size at construction time, or if this check belongs in ttrun.py -->

---

## 9. Watcher-to-ULFM Bridge, Multihost Defaults, and Dispatch Timeout Semantics

This section covers three interrelated design areas that bridge the gap between per-rank debug infrastructure (Watcher, Inspector, dispatch timeout) and the ULFM fault-tolerance layer. The goal: when hardware goes wrong on one rank, the distributed system should react intelligently rather than just crashing.

### 9.1 Watcher-to-ULFM Callback Bridge

#### 9.1.1 Current Error Throw Path

The Watcher runs in a dedicated thread (`WatcherServer::Impl::poll_watcher_data()`, `watcher_server.cpp` line 513). On each poll cycle it acquires `watch_mutex_`, iterates over all attached devices, and calls `WatcherDeviceReader::Dump()` for each device. `Dump()` delegates to per-core `WatcherDeviceReader::Core::Dump()` (`watcher_device_reader.cpp` line 546), which calls a series of check methods:

```
Core::Dump()                          (watcher_device_reader.cpp:546)
  ├── ValidateKernelIDs()             (watcher_device_reader.cpp:552)
  ├── DumpWaypoints()                 (watcher_device_reader.cpp:569)
  ├── DumpL1Status()                  (watcher_device_reader.cpp:574) → TT_THROW on L1[0] corruption (:643)
  ├── DumpNocSanitizeStatus(noc)      (watcher_device_reader.cpp:576-581) → TT_THROW on NOC violations (:746)
  ├── DumpAssertStatus()              (watcher_device_reader.cpp:582-583) → TT_THROW on tripped asserts (:755, :769, :782)
  ├── DumpPauseStatus()               (watcher_device_reader.cpp:585-586) → TT_THROW on pause errors (:803)
  ├── DumpEthLinkStatus()             (watcher_device_reader.cpp:589-591) → TT_THROW on link failures
  ├── DumpStackUsage()                (watcher_device_reader.cpp:621-622) → TT_THROW on stack overflow
  └── DumpRingBuffer()                (watcher_device_reader.cpp:624-625)
```

When any Dump method throws `std::runtime_error` (via `TT_THROW`), the exception propagates up to `poll_watcher_data()` (line 547), which has this catch:

```cpp
// watcher_server.cpp lines 547-554
try {
    dump();
} catch (const std::runtime_error& e) {
    if (rtoptions.get_test_mode_enabled()) {
        server_killed_due_to_error_ = true;  // atomic<bool>
        break;                                // exit poll loop
    }
    throw;  // propagate — crashes the process
}
```

**Two behaviors depending on test mode:**
- **Test mode** (`TT_METAL_WATCHER_TEST_MODE=1`): The Watcher thread silently sets `server_killed_due_to_error_` and stops polling. The main thread can check `killed_due_to_error()` later. No exception reaches the main thread.
- **Non-test mode** (default): The exception propagates from the Watcher thread. Since `std::thread` terminates the process on unhandled exceptions, this effectively calls `std::terminate()` — the rank dies, and ULFM detects it via TCP drop.

**The gap:** In neither case does the Watcher communicate the *nature* of the error to anyone before the process dies. The ULFM layer on surviving ranks sees a generic "rank died" event (`MPIX_ERR_PROC_FAILED`), with zero diagnostic information about *why* it died.

#### 9.1.2 Proposed Callback Interface

The design goal is to let the Watcher thread notify the ULFM layer *before* the process terminates, so the system can make policy-driven decisions.

**Option A: Direct callback registration (recommended)**

```cpp
// New file or addition to watcher_server.hpp
namespace tt::tt_metal {

// Callback signature: invoked on the Watcher thread when a hardware error is detected.
// The callback receives a structured error description and must be safe to call from
// the Watcher thread (i.e., must not acquire watch_mutex_).
struct WatcherError {
    ChipId device_id;
    CoreCoord core;
    std::string error_type;     // "assert", "noc_sanitize", "l1_corruption", "eth_link", "pause", "stack_overflow"
    std::string error_message;  // full diagnostic string from TT_THROW
    bool recoverable;           // hint: is this likely recoverable after tt-smi -r?
};

using WatcherErrorCallback = std::function<void(const WatcherError&)>;

class WatcherServer {
public:
    // ... existing API ...

    // Register a callback that fires when the Watcher detects a hardware error.
    // Only one callback is supported. Registering a new one replaces the old one.
    // Pass nullptr to clear.
    void set_error_callback(WatcherErrorCallback cb);

    // Query: has the error callback been invoked since last clear?
    bool has_pending_error() const;

    // Get the last error (if any). Returns nullopt if no error pending.
    std::optional<WatcherError> last_error() const;
};

}  // namespace tt::tt_metal
```

**Integration into `poll_watcher_data()`:**

```cpp
// Modified catch block in poll_watcher_data()
try {
    dump();
} catch (const std::runtime_error& e) {
    WatcherError error = parse_watcher_exception(e);  // extract structured info from message

    if (error_callback_) {
        error_callback_(error);  // notify ULFM layer BEFORE deciding to throw/exit
    }

    last_error_ = error;  // store for query

    if (rtoptions.get_test_mode_enabled()) {
        server_killed_due_to_error_ = true;
        break;
    }
    throw;  // still crashes, but callback had a chance to act first
}
```

**Option B: Atomic flag + condition variable (simpler, less flexible)**

Instead of a callback, the Watcher sets an atomic `WatcherError` struct and signals a condition variable. The ULFM integration layer polls or waits on this. This avoids any callback invocation on the Watcher thread but requires the consumer to actively check.

**Recommendation:** Option A. The callback pattern is more flexible and allows the ULFM handler to immediately initiate a revoke before the Watcher thread throws and terminates.

#### 9.1.3 Integration with FailurePolicy

The callback's behavior should be policy-driven:

```
FailurePolicy     Watcher Error Callback Action
────────────────  ─────────────────────────────────────────────────────────
FAST_FAIL         Log error with full diagnostics, then allow throw to proceed
                  (rank dies, ULFM detects via TCP drop — same as today,
                  but with better diagnostics in stderr/logfile)

FAULT_TOLERANT    1. Log error with full diagnostics
                  2. Set an atomic "hardware_fault_detected" flag
                  3. Call MPIX_Comm_revoke(comm_) to poison the communicator
                     (survivors will see MPIX_ERR_REVOKED on their next MPI call)
                  4. Optionally: call Inspector::serialize_rpc() for post-mortem
                  5. Throw — but now surviving ranks already know why
```

**Critical detail:** In FAULT_TOLERANT mode, the callback should call `MPIX_Comm_revoke()` *before* the process dies. This is safe — `MPIX_Comm_revoke()` is designed to be callable from any thread, and calling it multiple times on the same communicator is a no-op (returns `MPI_ERR_REVOKED` on the second call).

The advantage: surviving ranks get `MPIX_ERR_REVOKED` immediately rather than waiting for the TCP layer to detect the dead rank (which can take seconds to minutes depending on heartbeat configuration).

#### 9.1.4 Thread Safety Considerations

The Watcher thread is separate from the main thread and any MPI communication threads. Key constraints:

1. **`watch_mutex_`** is held during `dump()`. The callback MUST NOT acquire `watch_mutex_` (deadlock). This means the callback cannot call any Watcher methods that take the mutex (e.g., `get_lock()`, `clear_log()`).

2. **`comm_mutex_`** in `MPIContext` protects communicator state. The callback CAN acquire this — there's no lock ordering conflict since `watch_mutex_` and `comm_mutex_` are in independent classes with no nesting.

3. **MPI thread safety:** `MPIX_Comm_revoke()` is safe to call from any thread when MPI is initialized with `MPI_THREAD_MULTIPLE`. The current codebase initializes MPI with `MPI_THREAD_MULTIPLE` (verified in `mpi_distributed_context.cpp`). Notably, calling `MPIX_Comm_revoke()` from the Watcher callback while a collective is in progress on another thread is safe: the collective immediately returns `MPIX_ERR_REVOKED`, unblocking all waiters.

4. **Callback lifetime:** The callback holds a reference to the `MPIContext`. The `MPIContext` outlives the `WatcherServer` (both are owned by `MetalContext`, and `MetalContext` destroys them in the correct order: watcher first via `detach_devices()`, then MPI context at process exit via `atexit`). However, if the callback fires during `MetalContext` shutdown, the `MPIContext` may already be in a torn-down state. The callback should check `is_revoked()` or use a `weak_ptr` guard.

5. **`set_exception_message()`** in `WatcherServer` is already mutex-protected (`exception_message_mutex_`). The callback should be invoked AFTER `set_exception_message()` so that the error string is available to the callback consumer.

#### 9.1.5 Registration Point

The callback should be registered when `MetalContext` initializes its `WatcherServer`. In `metal_context.cpp` around line 240 (after `watcher_server_ = std::make_unique<WatcherServer>(*this->env_)`):

```cpp
// After watcher_server_ creation (metal_context.cpp ~line 240):
auto dist_ctx = get_distributed_context_ptr();
if (dist_ctx && dist_ctx->supports_fault_tolerance()) {
    watcher_server_->set_error_callback([weak_ctx = std::weak_ptr(dist_ctx)](const WatcherError& err) {
        auto ctx = weak_ctx.lock();
        if (!ctx) return;  // context already destroyed

        log_error(tt::LogMetal, "Watcher error on device {}: {} — {}",
                  err.device_id, err.error_type, err.error_message);

        // In FAULT_TOLERANT mode, proactively revoke to notify survivors
        auto* mpi_ctx = dynamic_cast<MPIContext*>(ctx.get());
        if (mpi_ctx && mpi_ctx->get_failure_policy() == FailurePolicy::FAULT_TOLERANT) {
            MPIX_Comm_revoke(mpi_ctx->comm());
        }
    });
}
```

### 9.2 Inspector/Watcher Multihost Defaults

#### 9.2.1 Current State: Defaults Are Identical (and Underserved)

After thorough investigation, **no defaults are gated on multihost**. There is no `is_multihost()` check in `rtoptions.cpp`, `rtoptions.hpp`, `metal_context.cpp`, or anywhere in the initialization path. The defaults are:

```
Setting                              Default Value          Set by Env Var                           Multihost-Aware?
───────────────────────────────────  ─────────────────────  ───────────────────────────────────────  ────────────────
Inspector enabled                    true                   TT_METAL_INSPECTOR=0/1                   No
Inspector RPC enabled                true                   TT_METAL_INSPECTOR_RPC=0/1               No
Inspector RPC port                   50051                  TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS     No (port laddering applies rank offset)
Serialize on dispatch timeout        true                   TT_METAL_INSPECTOR_SERIALIZE_ON_...       No
Watcher enabled                      false                  TT_METAL_WATCHER=1                       No
Watcher interval                     0 (no polling)         TT_METAL_WATCHER_INTERVAL=<ms>           No
Dispatch timeout                     0.0 (no timeout)       TT_METAL_OPERATION_TIMEOUT_SECONDS=<s>   No
Dispatch timeout command             "" (none)              TT_METAL_DISPATCH_TIMEOUT_COMMAND_...     No
Watcher test mode                    false                  TT_METAL_WATCHER_TEST_MODE=1             No
```

**Critical observation:** Inspector and its RPC server are enabled by default (`enabled = true`, `rpc_server_enabled = true` in `InspectorSettings` at `rtoptions.hpp` lines 108, 114). This means every Metal process starts an Inspector RPC server thread — including multihost ranks. Port laddering makes this work, but users may not realize they're running Inspector RPC servers on every rank.

**The Watcher is OFF by default.** This is the bigger problem for multihost: when a device fault occurs, the Watcher is not running to detect it. The dispatch timeout is also OFF (0.0s), so hangs are only detected by the MPI-level heartbeat or ttrun.py's wall-clock timeout (`TT_RUN_TIMEOUT`).

**ttrun.py does NOT set any of these.** Confirmed: `ttrun.py` does not set `TT_METAL_WATCHER`, `TT_METAL_INSPECTOR`, `TT_METAL_INSPECTOR_RPC`, or `TT_METAL_OPERATION_TIMEOUT_SECONDS`. The only debug-related env vars it sets are `TT_METAL_DISABLE_XIP_DUMP=1` (line 690) and rank-scoped log paths.

#### 9.2.2 Proposed Uniform Defaults

The goal: multihost runs should have the same fault-detection capabilities as single-host runs, without requiring users to set a matrix of env vars.

**Proposal: ttrun.py sets sensible defaults when launching MPI jobs.**

```python
# In ttrun.py get_rank_environment() or apply_rank_scoped_paths():

def _apply_multihost_debug_defaults(env: dict) -> None:
    """Set sensible debug defaults for multihost runs if not already configured."""

    # Watcher: enable by default in multihost for fault detection
    if "TT_METAL_WATCHER" not in env:
        env["TT_METAL_WATCHER"] = "1"

    # Watcher interval: 2000ms is a reasonable default (same as CI scripts)
    if "TT_METAL_WATCHER_INTERVAL" not in env:
        env["TT_METAL_WATCHER_INTERVAL"] = "2000"

    # Dispatch timeout: 300s (5 min) default to catch hangs
    if "TT_METAL_OPERATION_TIMEOUT_SECONDS" not in env:
        env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "300"

    # Inspector is already on by default (enabled=true, rpc=true in rtoptions.hpp),
    # so no need to set it. But ensure serialize-on-timeout is on:
    if "TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT" not in env:
        env["TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT"] = "1"

    # Dispatch timeout command: run triage automatically
    if "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE" not in env:
        env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"] = "./tools/tt-triage.py"
```

**Call site:** Inside `ttrun.py`'s environment setup, gated on multihost mode:

```python
# In build_mpi_command() or get_rank_environment():
if is_multihost_run:  # determined by presence of --hostfile or world_size > 1
    _apply_multihost_debug_defaults(env)
```

#### 9.2.3 Opt-In / Opt-Out

Users can override any default by setting the env var explicitly before calling ttrun:

```bash
# Disable watcher in multihost (override the new default):
TT_METAL_WATCHER=0 tt-run --hostfile hosts.txt my_test

# Set a shorter dispatch timeout:
TT_METAL_OPERATION_TIMEOUT_SECONDS=60 tt-run --hostfile hosts.txt my_test

# Disable all debug defaults:
TT_RUN_NO_DEBUG_DEFAULTS=1 tt-run --hostfile hosts.txt my_test
```

The `TT_RUN_NO_DEBUG_DEFAULTS` escape hatch skips `_apply_multihost_debug_defaults()` entirely for performance-sensitive runs.

#### 9.2.4 Interaction with Port Laddering

Enabling Inspector RPC on all ranks in multihost is already safe thanks to port laddering (`base_port + rank`). No changes needed to the port assignment. The only consideration:

- **Same-host co-location:** If multiple ranks run on the same host (e.g., 2 ranks per node for dual-chip nodes), port laddering prevents collisions because the offset is global rank, not per-node rank.
- **Firewall rules:** If the cluster firewall restricts ports, the Inspector RPC port range (`50051` to `50051 + world_size - 1`) must be open. This is already a requirement.

#### 9.2.5 Performance Impact

Enabling Watcher adds overhead:
- **Memory-mapped reads** from device mailboxes every `WATCHER_INTERVAL` ms
- **DMA disabled** while Watcher is running (`rtoptions.set_disable_dma_ops(true)` at `watcher_server.cpp` line 138)
- Typical overhead: 1-5% throughput reduction depending on workload

For production inference, users should set `TT_METAL_WATCHER=0` explicitly. For CI and development, the fault-detection benefit outweighs the overhead.

### 9.3 `on_dispatch_timeout_detected` — Idempotent vs One-Shot

#### 9.3.1 Current Implementation

`MetalContext::on_dispatch_timeout_detected()` is defined at `metal_context.cpp` line 715:

```cpp
void MetalContext::on_dispatch_timeout_detected() {
    std::lock_guard<std::mutex> lock(dispatch_timeout_detection_mutex_);

    if (!dispatch_timeout_detection_processed_) {
        dispatch_timeout_detection_processed_ = true;
        log_error(tt::LogMetal, "Timeout detected");
        if (rtoptions().get_serialize_inspector_on_dispatch_timeout()) {
            log_info(tt::LogMetal, "Serializing Inspector RPC data");
            Inspector::serialize_rpc();
        }
        std::string command = rtoptions().get_dispatch_timeout_command_to_execute();
        if (!command.empty()) {
            log_info(tt::LogMetal, "Executing command: {}", command);
            int result = std::system(command.c_str());
            if (result != 0) {
                log_warning(tt::LogMetal, "Timeout command '{}' returned non-zero exit code: {}",
                            command, WEXITSTATUS(result));
            }
        }
    }
}
```

**State it mutates:**
- `dispatch_timeout_detection_mutex_` (std::mutex) — guards the critical section
- `dispatch_timeout_detection_processed_` (bool) — set to `true` on first call, never reset

**Current behavior is one-shot:** The `dispatch_timeout_detection_processed_` flag ensures that only the *first* call to `on_dispatch_timeout_detected()` actually serializes Inspector data and runs the triage command. Subsequent calls are no-ops.

**Call sites (3 total):**

1. `system_memory_manager.cpp` line 560 — fetch queue wait timeout
2. `system_memory_manager.cpp` line 613 — completion queue wait timeout
3. `llrt.cpp` line 337 — RISC core completion timeout

All three call sites follow the same pattern: call `on_dispatch_timeout_detected()`, then `TT_THROW(...)`. The throw propagates up the call stack and typically crashes the rank.

**Thread safety:** The mutex protects concurrent calls from multiple command queue threads (each CQ has its own thread). However, the method calls `std::system()` inside the mutex, which means a long-running triage command blocks other timeout handlers. This is acceptable for one-shot but problematic for idempotent mode.

#### 9.3.2 Idempotent Design

**Use case:** Fault-tolerant / relaunch scenarios. A rank hits a dispatch timeout, serializes diagnostic data, but does NOT die — it participates in `revoke_and_shrink()` and continues with new work. If a second timeout occurs on the same rank, the diagnostic pipeline should fire again.

**Design:**

```cpp
void MetalContext::on_dispatch_timeout_detected() {
    // Atomic guard: only one thread enters the critical section at a time,
    // but the flag resets after completion to allow future invocations.
    std::lock_guard<std::mutex> lock(dispatch_timeout_detection_mutex_);

    if (dispatch_timeout_detection_in_progress_) {
        // Another thread is already handling a timeout — skip to avoid re-entrancy.
        return;
    }
    dispatch_timeout_detection_in_progress_ = true;

    // Increment counter for telemetry
    dispatch_timeout_count_++;

    log_error(tt::LogMetal, "Dispatch timeout detected (occurrence #{})", dispatch_timeout_count_);

    if (rtoptions().get_serialize_inspector_on_dispatch_timeout()) {
        Inspector::serialize_rpc();
    }

    std::string command = rtoptions().get_dispatch_timeout_command_to_execute();
    if (!command.empty()) {
        int result = std::system(command.c_str());
        // ... error handling ...
    }

    // In FAULT_TOLERANT mode, notify ULFM layer
    auto dist_ctx = get_distributed_context_ptr();
    if (dist_ctx && dist_ctx->supports_fault_tolerance()) {
        // Callback or direct flag set — see Section 9.1
        notify_ulfm_dispatch_timeout(dist_ctx);
    }

    dispatch_timeout_detection_in_progress_ = false;
    // NOTE: dispatch_timeout_detection_processed_ is NOT set — allows future calls.
}
```

**Key changes from current:**
- Replace `dispatch_timeout_detection_processed_` (permanent latch) with `dispatch_timeout_detection_in_progress_` (transient re-entrancy guard)
- Add `dispatch_timeout_count_` (atomic counter) for telemetry
- The `in_progress_` flag is cleared at the end, allowing future invocations
- Re-entrant calls from other threads while one is in progress are skipped (not queued)
- After `revoke_and_shrink()`, no explicit reset is needed — the method is always callable

**What "safely called multiple times" means:**
- Inspector serialization is safe to call repeatedly (it overwrites the same files)
- The triage command (`std::system()`) is safe to call repeatedly (it reads current state)
- MPI revocation is idempotent (`MPIX_Comm_revoke()` returns `MPI_ERR_REVOKED` on second call)
- The mutex prevents concurrent execution — two threads hitting timeout simultaneously results in one executing the handler, the other skipping

#### 9.3.3 One-Shot ULFM Propagation Design

**Use case:** All ranks should terminate if one rank errors out. When rank 2 hits a dispatch timeout, it triggers `MPIX_Comm_revoke()` to propagate to all ranks, causing them all to fail their next MPI operation.

**Design:**

```cpp
void MetalContext::on_dispatch_timeout_detected() {
    // One-shot: use std::call_once for maximum safety
    std::call_once(dispatch_timeout_once_flag_, [this]() {
        log_error(tt::LogMetal, "Dispatch timeout detected — initiating one-shot ULFM propagation");

        // Step 1: Serialize diagnostics before revocation (revocation may trigger
        //         rapid shutdown of other ranks)
        if (rtoptions().get_serialize_inspector_on_dispatch_timeout()) {
            Inspector::serialize_rpc();
        }

        // Step 2: Run triage command
        std::string command = rtoptions().get_dispatch_timeout_command_to_execute();
        if (!command.empty()) {
            int result = std::system(command.c_str());
        }

        // Step 3: Revoke communicator to propagate failure to all ranks
        auto dist_ctx = get_distributed_context_ptr();
        if (dist_ctx && dist_ctx->supports_fault_tolerance()) {
            auto* mpi_ctx = dynamic_cast<distributed::multihost::MPIContext*>(dist_ctx.get());
            if (mpi_ctx && !mpi_ctx->is_revoked()) {
                MPIX_Comm_revoke(mpi_ctx->comm());
                // All surviving ranks will get MPIX_ERR_REVOKED on their next MPI call,
                // which dispatches to handle_rank_failure() with their active policy.
            }
        }
    });
}
```

**How `MPIX_Comm_revoke()` propagates to all ranks:**

1. Rank 2 calls `MPIX_Comm_revoke(comm_)`.
2. OpenMPI ULFM runtime broadcasts a revocation message to all ranks in `comm_`.
3. On each surviving rank, the next MPI operation returns `MPIX_ERR_REVOKED`.
4. `mpi_check_ctx()` detects `MPIX_ERR_REVOKED`, calls `handle_rank_failure()`.
5. `handle_rank_failure()` acts according to the rank's local `FailurePolicy`:
   - `FAST_FAIL`: `_exit(70)` — all ranks terminate.
   - `FAULT_TOLERANT`: throws `MPIRankFailureException` — caller decides.

**This is the "poison pill" pattern:** one rank's dispatch timeout kills the entire job cleanly.

#### 9.3.4 Policy-Driven Behavior

The choice between idempotent and one-shot should be controlled by `FailurePolicy`:

```cpp
void MetalContext::on_dispatch_timeout_detected() {
    auto dist_ctx = get_distributed_context_ptr();
    bool is_fault_tolerant = dist_ctx && dist_ctx->supports_fault_tolerance();

    auto* mpi_ctx = is_fault_tolerant
        ? dynamic_cast<distributed::multihost::MPIContext*>(dist_ctx.get())
        : nullptr;

    FailurePolicy policy = mpi_ctx
        ? mpi_ctx->get_failure_policy()
        : FailurePolicy::FAST_FAIL;

    switch (policy) {
        case FailurePolicy::FAST_FAIL:
            // One-shot: serialize diagnostics, revoke communicator, let all ranks die.
            on_dispatch_timeout_one_shot(mpi_ctx);
            break;

        case FailurePolicy::FAULT_TOLERANT:
            // Idempotent: serialize diagnostics, set error flag, throw to caller.
            // Caller catches, calls revoke_and_shrink(), continues with reduced world.
            on_dispatch_timeout_idempotent(mpi_ctx);
            break;
    }
}
```

**For single-host runs** (no MPI context): The behavior is unchanged — serialize Inspector data and run the triage command. The one-shot guard (`dispatch_timeout_detection_processed_`) is appropriate since there's no recovery path.

#### 9.3.5 Race Conditions: Two Ranks Hit Dispatch Timeout Simultaneously

**Scenario:** Rank 0 and Rank 2 both hit dispatch timeouts at nearly the same time (e.g., both waiting on a hung device).

**FAST_FAIL (one-shot) mode:**
1. Rank 0 calls `on_dispatch_timeout_detected()` → serializes Inspector → calls `MPIX_Comm_revoke()`
2. Rank 2 calls `on_dispatch_timeout_detected()` → serializes Inspector → calls `MPIX_Comm_revoke()`
3. Both revocations succeed (idempotent at the MPI level — second call returns `MPI_ERR_REVOKED`)
4. Both ranks throw, both eventually `_exit(70)` or crash
5. Surviving ranks (1, 3, ...) get `MPIX_ERR_REVOKED` and also terminate

**No race hazard.** `MPIX_Comm_revoke()` is designed to be called concurrently by multiple ranks. The MPI runtime deduplicates the revocation internally.

**FAULT_TOLERANT (idempotent) mode:**
1. Rank 0 calls `on_dispatch_timeout_detected()` → serializes Inspector → throws to caller
2. Rank 2 calls `on_dispatch_timeout_detected()` → serializes Inspector → throws to caller
3. Both callers catch the exception, both eventually call `revoke_and_shrink()`
4. `revoke_and_shrink()` is a collective — ALL surviving ranks must call it

**Potential issue:** If only ranks 0 and 2 hit the timeout, but ranks 1 and 3 do not, the shrink call will hang because it requires all survivors. This is the fundamental coordination problem with FAULT_TOLERANT mode.

**Mitigation:** After calling `on_dispatch_timeout_detected()`, the timeout-detecting rank should ALSO revoke the communicator:

```cpp
// In the catch handler after on_dispatch_timeout_detected():
if (policy == FailurePolicy::FAULT_TOLERANT) {
    // Revoke forces all ranks to see ERR_REVOKED, which eventually causes
    // all of them to enter their error handlers and call revoke_and_shrink().
    MPIX_Comm_revoke(mpi_ctx->comm());
    mpi_ctx->revoke_and_shrink();  // collective — waits for all survivors
}
```

This ensures all ranks enter the recovery path, not just the ones that hit the timeout.

### 9.4 Implementation Roadmap

Changes ordered by dependency and increasing complexity:

```
Step  Change                                    File(s)                                              Complexity  Dependencies
────  ────────────────────────────────────────  ──────────────────────────────────────────────────  ──────────  ────────────
1     Add WatcherError struct and callback API   tt_metal/impl/debug/watcher_server.hpp               Low         None
      to WatcherServer                           tt_metal/impl/debug/watcher_server.cpp

2     Modify poll_watcher_data() to invoke       tt_metal/impl/debug/watcher_server.cpp               Low         Step 1
      callback before throw/break                (lines 547-554)

3     Add get_failure_policy() accessor to       tt_metal/distributed/multihost/                      Low         None
      MPIContext (currently set_failure_policy    mpi_distributed_context.hpp
      exists but no getter)

4     Register Watcher callback in MetalContext   tt_metal/impl/context/metal_context.cpp              Medium      Steps 1, 2, 3
      initialization                             (around line 240)

5     Add multihost debug defaults to ttrun.py   ttnn/ttnn/distributed/ttrun.py                       Low         None
                                                 (get_rank_environment / apply_rank_scoped_paths)

6     Add TT_RUN_NO_DEBUG_DEFAULTS escape hatch  ttnn/ttnn/distributed/ttrun.py                       Low         Step 5

7     Split on_dispatch_timeout_detected into    tt_metal/impl/context/metal_context.hpp               Medium      Step 3
      idempotent and one-shot variants           tt_metal/impl/context/metal_context.cpp
                                                 (lines 715-740)

8     Wire FailurePolicy into dispatch timeout   tt_metal/impl/context/metal_context.cpp               Medium      Steps 3, 7
      to select idempotent vs one-shot

9     Add MPIX_Comm_revoke() call in             tt_metal/impl/context/metal_context.cpp               Low         Step 7
      on_dispatch_timeout_one_shot()

10    Reset dispatch_timeout_detection_processed  tt_metal/impl/context/metal_context.cpp               Low         Step 7
      after revoke_and_shrink() (for idempotent  tt_metal/distributed/multihost/
      mode)                                      mpi_distributed_context.cpp

11    Add tests: Watcher callback fires on       tests/tt_metal/multihost/                             Medium      Steps 1-4
      simulated error                            fault_tolerance_tests/

12    Add tests: dispatch timeout in             tests/tt_metal/multihost/                             Medium      Steps 7-10
      FAULT_TOLERANT mode triggers               fault_tolerance_tests/
      revoke_and_shrink()
```

**Total estimated effort:** 3-5 developer-weeks for steps 1-10, plus 1-2 weeks for tests (steps 11-12).

**Risk areas:**
- Step 4 (callback registration) requires careful lifetime management — the distributed context must outlive the Watcher callback. Using `weak_ptr` mitigates this.
- Step 9 (revoke from dispatch timeout) changes the failure behavior of single-rank timeout from "just this rank dies" to "entire job fails." This is intentional for FAST_FAIL but must be clearly documented.
- Steps 7-8 (idempotent vs one-shot split) requires rethinking the `std::system()` call inside the mutex — a long-running triage command in idempotent mode blocks other timeout handlers. Consider running the triage command asynchronously (e.g., `std::async`).
