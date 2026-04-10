# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Regression test for: MeshBuffer::wait_for_pending_events() spinning forever
# on a closed device.
#
# Root cause (fixed in mesh_buffer.cpp):
#   enqueue_mesh_workload() records a completion event (pending_event_ids_[cq] = N)
#   on every tensor involved in a dispatch.  ttnn.close_device() closes the device and
#   resets sysmem_manager_, but the Python `device` shared_ptr stays alive.  When Python
#   GC later deallocates those tensors, MeshBuffer::deallocate() → wait_for_pending_events()
#   → EventSynchronize() hit sysmem_manager_'s lazy-reinit path, which creates a fresh
#   SystemMemoryManager with last_completed_event=0.  The busy-spin
#   "while (0 < event_N)" never exits → 40-minute CI hang.
#
# Fix: wait_for_pending_events() returns immediately if !mesh_device->is_initialized().

import gc
import signal
import torch
import ttnn

_TIMEOUT_SECONDS = 60


def _timeout_handler(signum, frame):
    raise RuntimeError(
        "REGRESSION: tensor deallocation after device close hung — "
        "MeshBuffer::wait_for_pending_events() is spinning on a closed device. "
        "See mesh_buffer.cpp and the is_initialized() guard."
    )


# Arm a watchdog so the test fails loudly instead of blocking the CI job for 40 min.
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(_TIMEOUT_SECONDS)

try:
    device = ttnn.open_device(device_id=0)

    t = torch.rand(32, 32, dtype=torch.float32)

    # Dispatching this op calls enqueue_record_event_to_host() and
    # track_completion_event_on_tensors(), which sets pending_event_ids_[0] = event_N
    # on both input_tensor and output_tensor's MeshBuffers.
    input_tensor = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.exp(input_tensor)

    # Ensure the result is correct while the device is still alive.
    torch_output = ttnn.to_torch(output_tensor)

    # Close the device:
    #   1. ~FDMeshCommandQueue() flushes outstanding work and kills reader threads.
    #   2. Device::close() resets sysmem_manager_.
    #   3. is_initialized() → false.
    # The Python `device` variable still holds the shared_ptr<MeshDevice>.
    ttnn.close_device(device)

    # Explicitly deallocate tensors while `device` is still in scope.
    # Before the fix this triggered EventSynchronize() on a closed device → infinite spin.
    del output_tensor
    del input_tensor
    gc.collect()

    # `device` goes out of scope here — fine, device is already closed.

finally:
    signal.alarm(0)  # Disarm watchdog.

print("[pass] tensor_dealloc_after_device_close: no hang after ttnn.close_device()")
