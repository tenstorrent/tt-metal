# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib

import ttnn


@contextlib.contextmanager
def tensor_prefetcher_session(device):
    """Open a Tensor prefetcher Start/Stop window. Stop runs even on test failure so
    the next test sees a clean device, replacing the explicit
    ``start -> ... -> stop -> synchronize`` callers used to spell out at every test site.

    A test that hoists its weights (``fetch_weights`` / ``queue_tensor_prefetcher_request``)
    ahead of the matmuls that consume them can fail with requests still sitting on the DRISC
    senders -- and not only from a rejected matmul, but from any op that throws in between.
    A clean stop cannot retire those: its sentinel queues behind the orphaned requests, so
    the kernel blocks on a full GCB and never reaches it, and the wait hangs and buries the
    error that caused it. The exception path therefore force-stops (abandoning the kernels)
    and skips the device sync, which would hang for the same reason. Forcing leaves DRISC
    kernels running, so the device has to be closed or reset before another session opens --
    fine here, because this path only runs while the caller is already unwinding.

    ``BaseException`` rather than ``Exception``: pytest's own outcomes (``pytest.fail`` /
    ``pytest.skip``) and Ctrl-C do not derive from ``Exception``, and each of them leaves the
    senders in exactly the same state.
    """
    ttnn.experimental.start_tensor_prefetcher(device)
    try:
        yield
    except BaseException:
        # Already stopped if the failure came out of ``LinearDecode.forward``, which retires
        # the senders itself; stopping twice is a no-op. Suppressed so a failure in the
        # teardown cannot mask the error being unwound.
        with contextlib.suppress(Exception):
            ttnn.experimental.stop_tensor_prefetcher(device, force=True)
        raise
    ttnn.experimental.stop_tensor_prefetcher(device)
    ttnn.synchronize_device(device)
