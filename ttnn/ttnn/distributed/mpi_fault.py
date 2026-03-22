# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""ULFM (User Level Fault Mitigation) helpers for mpi4py-based Python MPI programs.

This module provides a thin wrapper around mpi4py's ULFM extensions so that
Python test scripts and applications can detect MPI rank failures and react
cleanly — without needing to implement the ULFM protocol themselves.

Quick start
-----------
::

    from ttnn.distributed.mpi_fault import install_ulfm_handler, ulfm_guard

    install_ulfm_handler()          # once, at program start

    with ulfm_guard(comm, "allreduce"):
        comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)

Two failure policies
--------------------
``fast_fail`` (default)
    On rank failure the process prints a structured diagnostic to stderr,
    revokes the communicator (if ULFM bindings are available), and exits
    with code 70 (``EX_SOFTWARE``).  This mirrors the C++ ULFM handler in
    ``tt_metal/distributed/mesh_command_queue.cpp``.

``fault_tolerant``
    Instead of exiting, raises ``MPIRankFailureError``.  The caller should
    catch the exception, call ``comm.Shrink()`` to get a new communicator
    without the dead rank, and continue work on the new communicator.

Requirements
------------
- mpi4py (optional — module degrades gracefully if not installed)
- OpenMPI built with ULFM support for full functionality
- Without ULFM, the module still installs ERRORS_RETURN and catches
  standard MPI exceptions, but cannot identify specific failed ranks
  or revoke communicators.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from typing import Generator

# ---------------------------------------------------------------------------
# Guarded mpi4py import — mpi4py may not be installed in all environments
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI

    _MPI_AVAILABLE = True
except ImportError:
    MPI = None  # type: ignore[assignment]
    _MPI_AVAILABLE = False

# ULFM-specific MPI error codes.  These constants are only present in mpi4py
# builds linked against an ULFM-enabled MPI library (e.g. OpenMPI with
# --enable-mpi-fault-tolerance).
_ULFM_ERROR_CODES: set[int] = set()
if _MPI_AVAILABLE:
    for _attr in ("ERR_PROC_FAILED", "ERR_PROC_FAILED_PENDING", "ERR_REVOKED"):
        if hasattr(MPI, _attr):
            _ULFM_ERROR_CODES.add(getattr(MPI, _attr))


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class MPIRankFailureError(RuntimeError):
    """Raised by ``ulfm_guard`` when a rank failure is detected and
    ``policy == 'fault_tolerant'``.

    The caller should revoke and shrink the communicator before continuing.

    To switch from fast-fail to fault-tolerant mode::

        with ulfm_guard(comm, "allreduce", policy="fault_tolerant"):
            comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)

    Then catch this exception::

        try:
            with ulfm_guard(comm, "allreduce", policy="fault_tolerant"):
                comm.Allreduce(...)
        except MPIRankFailureError:
            new_comm = comm.Shrink()
            # continue work on new_comm
    """

    def __init__(self, rank: int, error_code: int, operation: str, failed_ranks: Optional[list[int]] = None):
        self.rank = rank
        self.error_code = error_code
        self.operation = operation
        self.failed_ranks = failed_ranks or []
        super().__init__(
            f"Rank {rank}: MPI rank failure detected during '{operation}' "
            f"(error_code={error_code}, failed_ranks={self.failed_ranks})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_ulfm_handler(comm=None) -> None:
    """Install ERRORS_RETURN on the given communicator so MPI errors raise
    ``MPI.Exception`` instead of aborting the process.

    This is the prerequisite for ``ulfm_guard`` to work — without it, any MPI
    error (including rank failure) causes an immediate ``MPI_Abort``.

    Args:
        comm: An mpi4py communicator.  Defaults to ``MPI.COMM_WORLD``.
    """
    if not _MPI_AVAILABLE:
        return

    if comm is None:
        comm = MPI.COMM_WORLD

    # ERRORS_RETURN tells the MPI runtime to return error codes to the caller
    # instead of invoking MPI_Abort.  This is essential for ULFM: without it
    # the process would be killed before we ever get a chance to inspect the
    # error code and handle the failure gracefully.
    comm.Set_errhandler(MPI.ERRORS_RETURN)


@contextmanager
def ulfm_guard(
    comm,
    operation_name: str = "collective",
    policy: str = "fast_fail",
) -> Generator[None, None, None]:
    """Context manager that catches ULFM rank-failure errors from mpi4py.

    Usage::

        with ulfm_guard(comm, "allreduce"):
            comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)

    If a ULFM error is detected:
    - ``policy="fast_fail"`` (default): prints a diagnostic and exits with
      code 70, mirroring the C++ handler.
    - ``policy="fault_tolerant"``: raises ``MPIRankFailureError`` so the
      caller can shrink the communicator and continue.

    Non-ULFM MPI exceptions propagate normally.

    Args:
        comm: The mpi4py communicator to guard.
        operation_name: Human-readable name for the operation (for diagnostics).
        policy: ``"fast_fail"`` or ``"fault_tolerant"``.
    """
    if not _MPI_AVAILABLE:
        # No mpi4py — just run the body with no protection
        yield
        return

    try:
        yield
    except MPI.Exception as exc:
        error_code = exc.Get_error_code()

        # Check if this is a ULFM error (rank failure / communicator revoked).
        # If ULFM constants aren't available in this mpi4py build, the set is
        # empty and we fall through to re-raise.
        if _ULFM_ERROR_CODES and error_code in _ULFM_ERROR_CODES:
            rank = comm.Get_rank()
            if policy == "fault_tolerant":
                # Let the caller handle recovery (shrink communicator, etc.)
                failed = _try_get_failed_ranks(comm)
                raise MPIRankFailureError(rank, error_code, operation_name, failed) from exc
            else:
                # fast_fail: mirror the C++ handle_rank_failure behavior
                _ulfm_fast_fail(comm, rank, error_code, operation_name)
        else:
            # Not a ULFM error — propagate normally
            raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_get_failed_ranks(comm) -> list[int]:
    """Best-effort identification of failed ranks via MPIX_Comm_get_failed.

    mpi4py exposes ULFM functions as methods on the communicator in
    ULFM-enabled builds.  If unavailable, we return an empty list — the
    caller still knows *something* failed, just not *who*.
    """
    if not hasattr(comm, "Get_failed"):
        return []

    try:
        # comm.Get_failed() returns a Group containing the failed ranks.
        # We translate that to a list of rank integers.
        failed_group = comm.Get_failed()
        return list(range(failed_group.Get_size()))
    except Exception:
        # Any failure here is non-critical — we're already in error handling
        return []


def _ulfm_fast_fail(comm, rank: int, error_code: int, operation_name: str) -> None:
    """Print a structured diagnostic and exit.

    Mirrors the C++ ``handle_rank_failure`` in
    ``tt_metal/distributed/mesh_command_queue.cpp``:
    1. Identify failed ranks (best-effort via MPIX)
    2. Revoke the communicator so other ranks unblock
    3. Print diagnostic to stderr
    4. Exit with code 70 (EX_SOFTWARE)
    """
    # Step 1: Try to identify which ranks actually failed
    failed_ranks = _try_get_failed_ranks(comm)

    # Step 2: Revoke the communicator so that any ranks blocked in MPI calls
    # will receive ERR_REVOKED and can exit cleanly.  comm.Revoke() is only
    # available in ULFM-enabled mpi4py builds.
    if hasattr(comm, "Revoke"):
        try:
            comm.Revoke()
        except Exception:
            # Revoke failed — continue to diagnostic anyway.  This can happen
            # if the communicator is already revoked or the runtime is in a
            # bad state.
            pass

    # Step 3: Structured diagnostic to stderr (matches C++ format)
    _ulfm_error_name = {
        getattr(MPI, "ERR_PROC_FAILED", -1): "ERR_PROC_FAILED",
        getattr(MPI, "ERR_PROC_FAILED_PENDING", -1): "ERR_PROC_FAILED_PENDING",
        getattr(MPI, "ERR_REVOKED", -1): "ERR_REVOKED",
    }
    error_name = _ulfm_error_name.get(error_code, f"UNKNOWN({error_code})")

    comm_size = -1
    try:
        comm_size = comm.Get_size()
    except Exception:
        # If we cannot query the communicator size, keep the sentinel -1.
        # This is a best-effort diagnostic only and should not affect handling.
        ...

    print(
        f"\n{'='*72}\n"
        f"[ULFM] Rank {rank}: MPI rank failure detected\n"
        f"  Operation : {operation_name}\n"
        f"  Error     : {error_name} ({error_code})\n"
        f"  Comm size : {comm_size}\n"
        f"  Failed    : {failed_ranks if failed_ranks else 'unknown'}\n"
        f"  Action    : fast_fail — revoking communicator and exiting\n"
        f"{'='*72}",
        file=sys.stderr,
        flush=True,
    )

    # Step 4: Exit with EX_SOFTWARE (70) — same as the C++ handler
    sys.exit(70)
