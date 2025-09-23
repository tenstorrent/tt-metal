/*
 * TTNN Slice Operation - Multi-Core Writer Kernel (4D Support)
 *
 * This kernel handles the output data writing phase of the slice operation for multi-core
 * execution, writing sliced tensor data from circular buffer to output tensor memory.
 * Supports 1D, 2D, 3D, and 4D tensors with work distribution across cores.
 *
 * Key Responsibilities:
 * - Read sliced data from circular buffer (produced by reader kernel)
 * - Write assigned portion of output data to DRAM using InterleavedAddrGenFast
 * - Handle different tensor dimensions with proper address calculations
 * - Support different data types with proper element size handling
 * - Process assigned rows for this core based on work distribution
 *
 * Architecture:
 * - Uses InterleavedAddrGenFast for efficient DRAM address generation
 * - Processes data row-by-row to match reader kernel output
 * - Simple sequential write pattern for optimal memory controller utilization
 * - Multi-core work distribution: each core writes a subset of output rows
 *
 * Memory Management:
 * - DRAM alignment: 32-byte boundaries for memory controller optimization
 * - L1 alignment: 16-byte boundaries for L1 cache efficiency
 * - Circular buffer: Double buffering synchronized with reader kernel
 *
 * Data Type Support:
 * - Element size determined at compile time for performance
 * - Dynamic element size passed as runtime argument for flexibility
 *
 * Performance Optimizations:
 * - Minimal synchronization overhead with reader kernel
 * - Efficient memory write operations using NOC async transfers
 * - Sequential access patterns optimized for DRAM controllers
 * - Parallel processing with load balancing across multiple cores
 *
 * Author: claude (based on pad operation template and single-core 4D slice)
 * Compatible with: TTNN framework, ROW_MAJOR_LAYOUT tensors, 1D-4D dimensions, multi-core execution
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Runtime arguments - optimized like production writer
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t stick_size_offset = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(3);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(4);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(5);
    uint32_t start_id = get_arg_val<uint32_t>(6);

    // Compile-time arguments
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Use TensorAccessor like production
    const auto s0 = TensorAccessor(dst_args, dst_addr, stick_size);

    // Batch processing like production writer
    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;

    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
        cb_wait_front(cb_id_in, num_read_per_barrier);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in);

        for (uint32_t i = 0; i < num_read_per_barrier && sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            uint64_t dst_noc_addr = get_noc_addr(i_stick, s0);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            l1_read_addr += stick_size_offset;
            i_stick += 1;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_in, num_read_per_barrier);
    }
}
