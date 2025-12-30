# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of UnifiedKernelBuilder.

This demonstrates how to use the unified kernel API for both local-only
operations and multicast operations.
"""

import ttnn
from ttnn.unified_kernel import UnifiedKernelBuilder


def example_local_op(device, input_tensor, output_tensor):
    """
    Example: Local-only operation (3-way split: Reader/Compute/Writer).

    The kernel source should include unified_common.h and use read_tile/write_tile:

    ```cpp
    #include "unified_common.h"
    #include "compute_kernel_api/common.h"
    #include "compute_kernel_api/eltwise_binary.h"

    KERNEL_MAIN {
        INIT_ARGUMENTS

        // For compute kernel: initialize compute operations
        #ifdef COMPILE_FOR_TRISC
        binary_op_init_common(in0_cb, in1_cb, out_cb);
        add_tiles_init(in0_cb, in1_cb);
        #endif

        for (uint32_t i = 0; i < n_tiles; i++) {
            // read_tile() adapts based on processor (BRISC vs TRISC)
            auto tile0 = read_tile(in0, i);
            auto tile1 = read_tile(in1, i);

            #ifdef COMPILE_FOR_TRISC
            // Compute on tiles
            tile_regs_acquire();
            add_tiles(tile0, tile1, 0);
            tile_regs_commit();
            tile_regs_wait();
            write_tile(0, out, i);  // write_tile() adapts based on processor
            tile_regs_release();
            #endif
        }
    }
    ```

    Note: The builder automatically sets COMPILE_FOR_BRISC/TRISC/NCRISC based on
    config type (ReaderConfigDescriptor/ComputeConfigDescriptor/WriterConfigDescriptor).
    """
    max_core = ttnn.CoreCoord(7, 7)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    builder = (
        UnifiedKernelBuilder("path/to/kernel.cpp", math_fidelity=ttnn.MathFidelity.HiFi4)
        .add_runtime_arg("n_tiles", 100)
        .add_tensor("in0", input_tensor)
        .add_tensor("in1", input_tensor)  # Example: binary op
        .add_tensor("out", output_tensor)
        .set_core_grid(all_cores)
    )

    program = builder.build(device)
    return ttnn.generic_op([input_tensor, input_tensor, output_tensor], program)


def example_mcast_op(device, input_tensor, output_tensor):
    """
    Example: Multicast operation.

    The kernel source uses role defines (MCAST_SENDER/MCAST_RECEIVER):

    ```cpp
    #include "unified_kernel/unified_kernel_api.h"
    #include "api/dataflow/dataflow_api.h"

    void kernel_main() {
        INIT_ARGUMENTS

        #if defined(MCAST_SENDER) && MCAST_SENDER == 1
        // Sender: read tile and multicast to receivers
        cb_reserve_back(CB_IN0, 1);
        uint32_t tile_addr = get_write_ptr(CB_IN0);
        // ... read tile into tile_addr ...
        cb_push_back(CB_IN0, 1);
        uint32_t tile_size = get_tile_size(CB_IN0);
        mcast_tile(tile_addr, tile_size, receivers);
        #elif defined(MCAST_RECEIVER) && MCAST_RECEIVER == 1
        // Receiver: receive tile, compute, write
        uint32_t tile_addr = receive_tile(in0);
        // ... compute on tile ...
        cb_wait_front(CB_OUT0, 1);
        uint32_t out_addr = get_read_ptr(CB_OUT0);
        // ... write result ...
        cb_pop_front(CB_OUT0, 1);
        #endif
    }
    ```

    Note: Role defines (MCAST_SENDER/MCAST_RECEIVER) are orthogonal to processor
    defines (COMPILE_FOR_BRISC/NCRISC). A multicast sender can run on BRISC or NCRISC.
    """
    # Define sender and receiver cores
    sender_core = ttnn.CoreCoord(0, 0)
    receiver_range = ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(3, 3))
    receivers = ttnn.CoreRangeSet([receiver_range])

    builder = (
        UnifiedKernelBuilder("path/to/mcast_kernel.cpp")
        .add_tensor("in0", input_tensor)
        .add_tensor("out", output_tensor)
        .add_mcast_group("receivers", receivers, sender=sender_core)
        .set_core_grid(receivers.merge(ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])))
    )

    program = builder.build(device)
    return ttnn.generic_op([input_tensor, output_tensor], program)
