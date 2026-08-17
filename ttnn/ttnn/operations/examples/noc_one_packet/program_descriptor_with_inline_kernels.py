# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data-movement example: the `max_page_size` TEMPLATE ARGUMENT on `noc_async_write`.

`noc_async_write` is declared with a defaulted template parameter:

    template <uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, ...>
    inline void noc_async_write(uint32_t src, uint64_t dst, uint32_t size, ...) {
        if constexpr (max_page_size <= NOC_MAX_BURST_SIZE) {
            noc_async_write_one_packet<...>(src, dst, size, ...);   // cheap path
        } else {
            ncrisc_noc_fast_write_any_len<...>(...);                // generic path
        }
    }

The default is `NOC_MAX_BURST_SIZE + 1`, so the `if constexpr` is FALSE and a plain
call always compiles to the GENERIC path — the one that carries a
chunk-at-a-time loop for transfers larger than one burst. A page that is *provably*
one packet (a burst is 16 KB on this class of part) still pays for that generality.
Naming the size as the template argument selects the one-packet path instead.

Both paths issue the same NoC transaction and move the same bytes; the difference is
the SOFTWARE cost the issuing RISC-V pays per call, before the transfer is even
programmed. This example isolates that cost.

ISOLATION. The concept is per-call issue overhead on the data-movement core, so:

  * NO compute. There is one data-movement kernel and nothing else.
  * Input and output are BOTH L1-sharded, so no DRAM bandwidth or bank contention
    is in the measurement.
  * The traffic is a RING SHIFT: core k writes its input shard, page by page, into
    core (k+1)'s output shard. Every core issues exactly the same number of writes
    to exactly one remote core, one hop away.
  * No semaphores and no circular buffers: nothing reads the destination during the
    kernel, so a single write barrier at the end is the only synchronization.

That makes the issuing RISC-V *the whole kernel* — its per-call cost IS the kernel
duration. This matters for reading the result: the same change measured inside an op
where the issuing core is NOT on the critical path moves the stage and not the wall.

    variant="generic"    : noc_async_write(src, dst, PAGE_BYTES)
    variant="one_packet" : noc_async_write<PAGE_BYTES>(src, dst, PAGE_BYTES)

The kernels are otherwise byte-identical — same loop, same addresses, same barrier —
so the delta is attributable to the template argument alone. Sweep `--page-bytes` to
see the fixed per-call cost become a larger fraction as the payload shrinks, and
`--pages-per-core` to see it accumulate.
"""

import ttnn

# Baseline first: the plain call every kernel writes by default.
VARIANTS = ("generic", "one_packet", "generic_runtime_size")

SUPPORTED_DTYPES = (ttnn.bfloat16, ttnn.float32)

# A NoC page must start on a 16 B boundary in L1; every page size the CLI accepts is
# a multiple of 64 B, which satisfies that with room to spare.
_L1_ALIGN_BYTES = 16


_RING_WRITE_KERNEL = """
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// One data-movement kernel, no compute. Each core walks its own input shard and
// writes it page by page into the NEXT core's output shard, then barriers once.
//
// USE_ONE_PACKET selects which overload of noc_async_write is instantiated. Both
// branches are the same statement with the same arguments; only the template
// argument (and therefore the code path inside dataflow_api.h) differs.
void kernel_main() {
    constexpr uint32_t PAGE_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t PAGES_PER_CORE = get_compile_time_arg_val(1);
    constexpr uint32_t KERNEL_ITERS = get_compile_time_arg_val(2);
    constexpr uint32_t USE_ONE_PACKET = get_compile_time_arg_val(3);

    constexpr uint32_t DEST_SPREAD = get_compile_time_arg_val(4);
    constexpr uint32_t NUM_DESTS = get_compile_time_arg_val(5);
    constexpr uint32_t RUNTIME_SIZE = get_compile_time_arg_val(6);

    const uint32_t src_base = get_arg_val<uint32_t>(0);
    const uint32_t dst_base = get_arg_val<uint32_t>(1);
    // runtime args 2.. : NUM_DESTS (x, y) pairs, this core's ring order.
    const uint32_t dst_xy_base = 2;

    for (uint32_t iter = 0; iter < KERNEL_ITERS; ++iter) {
        for (uint32_t page = 0; page < PAGES_PER_CORE; ++page) {
            const uint32_t offset = page * PAGE_BYTES;
            // DEST_SPREAD=0: every page to ONE destination -- the transfers queue at
            // that core's NIU and the issue cost is hidden behind the drain.
            // DEST_SPREAD=1: page p goes to the p-th destination round-robin, so the
            // transfers drain CONCURRENTLY and the issuing core's per-call software
            // cost is what is left exposed.
            const uint32_t d = DEST_SPREAD ? (page % NUM_DESTS) : 0;
            // RUNTIME_SIZE: the transfer size arrives as a runtime arg, so the generic
            // path's `while (len_bytes > NOC_MAX_BURST_SIZE)` chunk loop and its size
            // arithmetic CANNOT be folded away at compile time. This is the control that
            // shows what the constexpr size is worth, independent of the template arg.
            const uint32_t bytes = RUNTIME_SIZE ? get_arg_val<uint32_t>(1 + 2 * NUM_DESTS + 1) : PAGE_BYTES;
            const uint32_t dst_x = get_arg_val<uint32_t>(dst_xy_base + 2 * d);
            const uint32_t dst_y = get_arg_val<uint32_t>(dst_xy_base + 2 * d + 1);
            const uint64_t dst = get_noc_addr(dst_x, dst_y, dst_base + offset);
            if constexpr (USE_ONE_PACKET) {
                noc_async_write<PAGE_BYTES>(src_base + offset, dst, PAGE_BYTES);
            } else if constexpr (RUNTIME_SIZE) {
                noc_async_write(src_base + offset, dst, bytes);
            } else {
                noc_async_write(src_base + offset, dst, PAGE_BYTES);
            }
        }
        noc_async_write_barrier();
    }
}
"""


def _grid_cores(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y


def _ordered_cores(device, num_cores):
    """`num_cores` cores filled row-major. Identical placement for both variants."""
    grid = device.compute_with_storage_grid_size()
    if num_cores > grid.x * grid.y:
        raise ValueError(f"requested {num_cores} cores, grid has {grid.x * grid.y}")
    return [(k % grid.x, k // grid.x) for k in range(num_cores)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def create_sharded_memory_config(device, num_cores, pages_per_core, page_elems, dtype):
    """Height-shard a (num_cores*pages_per_core, page_elems) tensor one shard per core."""
    cores = _ordered_cores(device, num_cores)
    return ttnn.create_sharded_memory_config(
        shape=(pages_per_core, page_elems),
        core_grid=_core_range_set(cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def validate(input_tensor, output_tensor, *, num_cores, pages_per_core):
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"noc_one_packet: dtype must be one of {SUPPORTED_DTYPES}")
    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError("noc_one_packet: input and output dtype must match")
    if list(input_tensor.shape) != list(output_tensor.shape):
        raise ValueError("noc_one_packet: input and output shape must match")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("noc_one_packet: tensors must be ROW_MAJOR (this is a page copy)")
    rows = list(input_tensor.shape)[0]
    if rows != num_cores * pages_per_core:
        raise ValueError(
            f"noc_one_packet: rows ({rows}) must equal num_cores*pages_per_core " f"({num_cores}*{pages_per_core})"
        )


def create_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    variant,
    num_cores,
    pages_per_core,
    kernel_iters=1,
    dest_spread=False,
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if kernel_iters < 1:
        raise ValueError("kernel_iters must be positive")
    validate(input_tensor, output_tensor, num_cores=num_cores, pages_per_core=pages_per_core)

    device = input_tensor.device()
    cores = _ordered_cores(device, num_cores)
    page_bytes = input_tensor.buffer_aligned_page_size()
    if page_bytes % _L1_ALIGN_BYTES:
        raise ValueError(f"page_bytes ({page_bytes}) must be a multiple of {_L1_ALIGN_BYTES}")

    virtual = [(lambda c: (c.x, c.y))(device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))) for x, y in cores]

    # Ring shift: core k -> core k+1. Every core issues the same work, so no core is
    # idle and the busiest-core duration is every core's duration.
    runtime_args = ttnn.RuntimeArgs()
    for index, (x, y) in enumerate(cores):
        # This core's destinations in ring order: next, next+1, ... Under
        # dest_spread the kernel walks them; otherwise it only uses the first.
        args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
        for step in range(num_cores):
            dx, dy = virtual[(index + 1 + step) % num_cores]
            args += [dx, dy]
        args.append(page_bytes)  # only read by the generic_runtime_size variant
        runtime_args[x][y] = args

    kernel = ttnn.KernelDescriptor(
        kernel_source=_RING_WRITE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_core_range_set(cores),
        compile_time_args=[
            page_bytes,
            pages_per_core,
            kernel_iters,
            1 if variant == "one_packet" else 0,
            1 if dest_spread else 0,
            num_cores,
            1 if variant == "generic_runtime_size" else 0,
        ],
        runtime_args=runtime_args,
        # Writes go out on the writer's NoC, which is the preferred NoC for writes.
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])


def noc_one_packet(
    input_tensor,
    output_tensor,
    *,
    variant="one_packet",
    num_cores,
    pages_per_core,
    kernel_iters=1,
    dest_spread=False,
):
    """Run the ring-shift copy in place on `output_tensor`; returns it."""
    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        num_cores=num_cores,
        pages_per_core=pages_per_core,
        kernel_iters=kernel_iters,
        dest_spread=dest_spread,
    )
    ttnn.generic_op([input_tensor, output_tensor], descriptor)
    return output_tensor
