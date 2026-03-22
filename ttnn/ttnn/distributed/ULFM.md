# ULFM Support in tt-metal Multihost

User Level Failure Mitigation (ULFM) is an MPI extension that lets surviving ranks **detect and react to** remote rank failures instead of hanging forever at the next collective. Without it, a single segfault or OOM on one node stalls every other node until a system timeout kills the job.

tt-metal uses a multi-layer defence to turn "hang until timeout" into "detect, log, exit fast":

1. **C++ ULFM detection** — catches failures at the MPI call site
2. **`std::set_terminate` handler** — catches uncaught C++ exceptions including JIT/thread-pool failures
3. **`MPI_Finalize` watchdog** — 30-second SIGALRM; prevents atexit from hanging if a remote rank died before calling finalize
4. **ORTE/PRRTE abort-on-non-zero** — catches crashes that exit non-zero before any MPI call
5. **Process-level timeout** — last-resort `TT_RUN_TIMEOUT` watchdog
6. **Python mpi4py wrapper** — ULFM support for Python-level collectives


## How It Works

### Layer 1 — C++ ULFM Error Detection

File: `tt_metal/distributed/multihost/mpi_distributed_context.cpp`

Every MPI call in `MPIContext` goes through the `MPI_CHECK_CTX` macro. When OpenMPI returns a ULFM error code (`MPIX_ERR_PROC_FAILED`, `MPIX_ERR_PROC_FAILED_PENDING`, `MPIX_ERR_REVOKED`), the macro:

1. Acknowledges failures via `MPIX_Comm_failure_ack`
2. Queries which world-ranks died via `MPIX_Comm_failure_get_acked`
3. Revokes the communicator with `MPIX_Comm_revoke` to unblock all surviving ranks
4. Prints a structured diagnostic to stderr
5. Dispatches to the configured `FailurePolicy`

Under `FAST_FAIL` (the default), the process calls `_exit(70)` immediately — no destructors, no `MPI_Finalize`, no deadlock risk. Under `FAULT_TOLERANT`, it throws `MPIRankFailureException` so application code can attempt recovery.

### Layer 2 — `std::set_terminate` Handler

File: `tt_metal/distributed/multihost/mpi_distributed_context.cpp`

Layer 1 only fires when MPI itself returns an error code. Non-MPI fatal errors — filesystem ESTALE during JIT compilation, OOM, exceptions in thread-pool workers — kill the rank without going through an MPI call. The surviving ranks hang at the next collective forever.

`init_env()` installs a `std::set_terminate` handler via `std::set_terminate(mpi_terminate_handler)`. This fires for:
- Any uncaught C++ exception (including ones thrown in `std::async`/thread-pool workers)
- Explicit `std::terminate()` calls

The handler calls `MPIX_Comm_revoke(MPI_COMM_WORLD)` (if ULFM is available) to unblock surviving ranks, then calls `_exit(70)`.

**Limitation**: exceptions caught by Python/pybind11 are not "uncaught" in the C++ sense and won't trigger this handler. Layer 3 covers that case.

### Layer 3 — `MPI_Finalize` Watchdog

File: `tt_metal/distributed/multihost/mpi_distributed_context.cpp`

The most common hang pattern: a rank hits a non-MPI error (e.g. ESTALE), the exception is caught by pytest/pybind11, pytest runs test teardown, the rank's process eventually exits normally, and the `atexit` handler calls `MPI_Finalize()`. `MPI_Finalize` is a collective — it blocks waiting for all other ranks to also call it. The other ranks are still running normally and are nowhere near `MPI_Finalize`. The job hangs for the entire CI step timeout (often 30–60 minutes).

The atexit handler now arms a `SIGALRM` watchdog before calling `MPI_Finalize`:

```cpp
signal(SIGALRM, mpi_finalize_alarm_handler);
alarm(MPI_FINALIZE_TIMEOUT_SECS);  // default: 30 seconds
MPI_Finalize();
alarm(0);  // disarm if finalize completed normally
```

If `MPI_Finalize` does not complete within 30 seconds, `SIGALRM` fires and calls `_exit(70)`.

### Layer 4 — ORTE/PRRTE Abort on Non-Zero Exit

File: `ttnn/ttnn/distributed/ttrun.py`

If a rank crashes (segfault, unhandled exception) before ever entering an MPI collective, ULFM has nothing to detect. The safety net is the MPI runtime's built-in abort propagation:

- OpenMPI 4.x: `--mca orte_abort_on_non_zero_status 1`
- OpenMPI 5.x / PRRTE: `--mca prte_abort_on_non_zero_status 1`

`ttrun.py` detects the OpenMPI major version at runtime via `_detect_openmpi_major_version()` and sets the correct flag.

### Layer 5 — Process-Level Timeout

File: `ttnn/ttnn/distributed/ttrun.py`

Set `TT_RUN_TIMEOUT=<seconds>` in the environment. `ttrun.py` calls `proc.wait(timeout=N)` and on expiry sends `SIGKILL` then exits with code 124.

This catches anything Layers 1–4 miss — infinite loops, deadlocks in non-MPI code, ranks stuck in device I/O.

### Layer 6 — Python mpi4py Wrapper

File: `ttnn/ttnn/distributed/mpi_fault.py`

For Python-level MPI usage:

- `install_ulfm_handler(comm)` sets `ERRORS_RETURN` on the communicator so MPI errors come back as return codes instead of aborting
- `ulfm_guard(comm, operation_name, policy)` is a context manager that catches ULFM errors and dispatches per policy
- `MPIRankFailureError` is the Python exception raised in fault-tolerant mode


## Exit Codes

```
Code   Meaning                          Source
70     EX_SOFTWARE — ULFM fast-fail     Layer 1 (_exit(70) after detecting rank failure)
124    Timeout                           Layer 3 (TT_RUN_TIMEOUT expired, SIGKILL sent)
```

**CI tooling**: grep for exit code 70 to identify ULFM-initiated shutdowns. Exit 124 means the timeout watchdog fired. Both indicate a remote rank failure or hang — check the failing rank's stderr for the root cause.


## Switching to Fault-Tolerant Mode

The default `FAST_FAIL` policy is intentional — it gives CI clean, fast exits. `FAULT_TOLERANT` mode is for future resilient applications that want to survive rank losses and continue with a reduced communicator.

### C++

```cpp
#include "tt_metal/distributed/multihost/mpi_distributed_context.hpp"

using namespace tt::tt_metal::distributed::multihost;

// Switch to fault-tolerant mode (throws instead of _exit):
ctx->set_failure_policy(FailurePolicy::FAULT_TOLERANT);
// Throws TT_THROW if ULFM is not available in the build.

// In your communication loop:
try {
    ctx->barrier();
} catch (const MPIRankFailureException& e) {
    // e.failed_ranks() — comma-separated list of dead world-ranks, e.g. "2, 5"
    // e.error_code()   — raw MPI error code
    // e.rank()         — detecting rank (this rank)

    ctx->revoke_and_shrink();  // MUST call before any further MPI ops
    // ctx now has a new communicator excluding the dead ranks.
    // Resize your data structures, update rank mappings, continue.
}
```

### Python

```python
from ttnn.distributed.mpi_fault import install_ulfm_handler, ulfm_guard, MPIRankFailureError
from mpi4py import MPI

comm = MPI.COMM_WORLD
install_ulfm_handler(comm)

try:
    with ulfm_guard(comm, "allreduce", policy="fault_tolerant"):
        comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
except MPIRankFailureError as e:
    # e.failed_ranks — list of dead world-ranks
    # e.rank         — detecting rank
    # e.error_code   — raw MPI error code
    new_comm = comm.Shrink()  # mpi4py ULFM shrink — new comm without dead ranks
    # Rebuild data structures, continue.
```

### Key points

- `set_failure_policy(FAULT_TOLERANT)` / `policy="fault_tolerant"` must be set **before** the first collective that might fail
- After catching a failure, you **must** call `revoke_and_shrink()` (C++) or `comm.Shrink()` (Python) before any further MPI operations on that communicator
- The shrunk communicator has new rank numbers — update any rank-indexed data structures
- If `OMPI_HAS_ULFM` is not defined at compile time, `set_failure_policy(FAULT_TOLERANT)` throws — you cannot use fault-tolerant mode without a ULFM-enabled build


## Relevant Files

- `tt_metal/distributed/multihost/mpi_distributed_context.hpp` — C++ context class, `FailurePolicy` enum, `MPIRankFailureException`
- `tt_metal/distributed/multihost/mpi_distributed_context.cpp` — ULFM detection logic, `handle_rank_failure`, `revoke_and_shrink`, `MPI_CHECK_CTX` macro
- `ttnn/ttnn/distributed/ttrun.py` — `mpirun` launcher, `--with-ft ulfm` flag, ORTE/PRRTE abort parameter, `TT_RUN_TIMEOUT` watchdog
- `ttnn/ttnn/distributed/mpi_fault.py` — Python mpi4py ULFM wrapper: `install_ulfm_handler`, `ulfm_guard`, `MPIRankFailureError`


## Runtime Requirements

- **ULFM-enabled OpenMPI**: must be built with `--with-ft=ulfm` (the tt-metal Docker container includes this)
- **ULFM launcher**: `mpirun-ulfm` must be available on `$PATH` — `ttrun.py` selects it automatically
- **mpirun flags**: `--with-ft ulfm` must be passed to `mpirun` — `ttrun.py` does this automatically
- **Compile-time define**: `OMPI_HAS_ULFM` controls whether C++ ULFM code paths are compiled in; without it, ULFM detection is a no-op and `FAULT_TOLERANT` policy is rejected at runtime


## Known Limitations

- **mpi4py ULFM bindings**: `comm.Revoke()`, `comm.Get_failed()`, `comm.Shrink()` may not exist if mpi4py was linked against a non-ULFM OpenMPI. `mpi_fault.py` degrades gracefully — `install_ulfm_handler` becomes a no-op, and `ulfm_guard` falls back to standard error handling.
- **PRRTE watchdog gap**: `prte_abort_on_non_zero_status` only fires when the rank's process actually exits. A rank stuck in an infinite loop (no crash, no exit) will **not** trigger it. Use `TT_RUN_TIMEOUT` as the backstop.
- **No automated ULFM tests**: there is no CI test harness for ULFM paths yet. Manual test plan is tracked in PR #40457.
- **Shrink overhead**: `revoke_and_shrink` creates a new communicator, which is an expensive collective across all surviving ranks. Do not call it in a hot loop.


## Testing ULFM Locally (Manual)

### Verify ULFM is enabled in the mpirun command

```bash
# ttrun.py prints the full mpirun command to stderr. Look for:
#   --with-ft ulfm
#   mpirun-ulfm (or mpirun with ULFM support)
python ttnn/ttnn/distributed/ttrun.py --help  # check available flags
```

### Test TT_RUN_TIMEOUT

```bash
# Run a program that hangs, confirm it gets killed after 10s:
TT_RUN_TIMEOUT=10 python ttnn/ttnn/distributed/ttrun.py -n 2 -- python -c "import time; time.sleep(999)"
# Expected: SIGKILL after 10s, exit code 124
echo $?  # should print 124
```

### Simulate a rank crash

```bash
# crash_rank0.py — rank 0 segfaults, rank 1 should detect and exit 70
cat > /tmp/crash_rank0.py << 'EOF'
from mpi4py import MPI
import ctypes, time

rank = MPI.COMM_WORLD.Get_rank()
if rank == 0:
    time.sleep(1)
    ctypes.string_at(0)  # segfault
else:
    # This barrier will detect rank 0's death via ULFM
    MPI.COMM_WORLD.Barrier()
EOF

python ttnn/ttnn/distributed/ttrun.py -n 2 -- python /tmp/crash_rank0.py
# Expected: rank 1 logs ULFM diagnostic, exits 70
```
