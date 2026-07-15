# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-core data-movement benchmark: reusing a shared input stream across cores ("read once, share many").

A fixed 2×`grid_x` rectangle of worker cores each need the SAME multi-MB input `X` — a large shared
matrix `[R, C]`, which is far larger than L1, so it is streamed through L1 in fixed-size chunks. `X`
lives in DRAM, interleaved.
Every core does the SAME trivial job — fold the whole stream into one running tile-sum (kept in the DEST
accumulator, one write-back at the end) — so the only thing that differs between the two variants is HOW
each core gets the stream:

  per_core_dram (baseline): every core reads the whole stream from DRAM itself. With N cores that is
                 `N × R × C` of DRAM read traffic for one stream's worth of unique data.
  mcast (optimized): the top-left injector reads each chunk from DRAM ONCE and NoC-multicasts it to all
                 the other cores (per-chunk semaphore handshake — the production forwarding pattern), so
                 DRAM sees the stream once and the copies travel core-to-core.

Output is one tile per core (`N` tiles) — negligible next to the multi-MB read — so the kernel is
READ-bound and the measured delta is the read strategy. Geometry is fixed (one 2×`grid_x` rectangle used
by both variants), so core placement is not a hidden variable (contrast `reader_placement`, which
isolates exactly that: WHERE a fixed-work line sits). See README.md.
"""

import ttnn

TILE = 32
CB_IN = 0  # one input chunk at a time (chunk_tiles), streamed into L1
CB_ZERO = 2  # a single all-zero tile (built on-device), the +0 operand for the acc_to_dest add
CB_OUT = 16  # per-core output: the running tile-sum (1 tile)
# The multicast semaphores are owned/allocated by ttnn.Mcast2D (base_sem_id=0 -> data_ready + consumer_ready).

VARIANTS = ("per_core_dram", "mcast")

# ---- Reader (baseline): stream every chunk of the shared input from DRAM into cb_in. ----
_READER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t x_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = 0;
    constexpr auto x_args = TensorAccessorArgs<2>();

    const uint32_t tile_bytes = get_tile_size(cb_in);
    const auto x_acc = TensorAccessor(x_args, x_addr, tile_bytes);

    for (uint32_t c = 0; c < num_chunks; ++c) {
        cb_reserve_back(cb_in, chunk_tiles);
        uint32_t w = get_write_ptr(cb_in);
        const uint32_t base = c * chunk_tiles;
        for (uint32_t i = 0; i < chunk_tiles; ++i) { noc_async_read_tile(base + i, x_acc, w + i * tile_bytes); }
        noc_async_read_barrier();
        cb_push_back(cb_in, chunk_tiles);
    }
}
"""

# ---- Injector (mcast): read each chunk once, broadcast it to the worker rectangle via the mcast_pipe
# SenderPipe helper, feed own compute. Injector is a corner of the rect and broadcasts from the same
# cb_in slot it read into (src==dst) so the pipe self-excludes (fan-out = the other 21 cores). ----
_SENDER_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/1>();  // mcast CT block at 1..; dest rect at RT 1..
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(SCALARS + 1);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(SCALARS + 2);
    constexpr auto in_args = TensorAccessorArgs<SCALARS + 3>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);  // RT 1.. = dest rect, consumed by mc.sender()
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;

    Noc noc;
    CircularBuffer cin(cb_in);
    const auto in = TensorAccessor(in_args, x_addr);
    auto pipe = mc.sender(noc);  // in_rect corner, PRE_HANDSHAKE flow control from the wire

    for (uint32_t c = 0; c < num_chunks; ++c) {
        cin.reserve_back(chunk_tiles);  // own slot free (compute drained an earlier chunk)
        const uint32_t dst = cin.get_write_ptr();
        const uint32_t base = c * chunk_tiles;
        for (uint32_t i = 0; i < chunk_tiles; ++i) {
            noc.async_read(in, cin, tile_bytes, {.page_id = base + i}, {.offset_bytes = i * tile_bytes});
        }
        noc.async_read_barrier();
        if constexpr (mc.active) {
            pipe.send(dst, dst, chunk_bytes);  // wait 21 acks, broadcast chunk (self excluded), signal
        }
        cin.push_back(chunk_tiles);  // hand the chunk to the injector's own compute
    }
}
"""

# ---- Receiver (mcast): per chunk, ack (slot free) + wait for the broadcast via ReceiverPipe, feed compute. ----
_RECEIVER_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr auto mc = McastArgs</*CT=*/1, /*RT=*/0>();  // sender coords at RT 0..
    constexpr uint32_t SCALARS = mc.next_compile_time_args_offset();
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(SCALARS + 0);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(SCALARS + 1);

    Noc noc;
    CircularBuffer cin(cb_in);
    auto pipe = mc.receiver(noc);

    for (uint32_t c = 0; c < num_chunks; ++c) {
        cin.reserve_back(chunk_tiles);  // slot free -> write_ptr is the landing address; ack must follow this
        pipe.receive();                 // ack sender (slot free), wait VALID; chunk now in cb_in
        cin.push_back(chunk_tiles);
    }
}
"""

# ---- Compute (shared by both variants, all cores): running tile-sum over the whole stream -> 1 tile. ----
# The running sum lives in the fp32 DEST accumulator across the ENTIRE stream, folded in with
# add_tiles(acc_to_dest) so the sum never round-trips through a bf16 Src register (16-bit data, exact
# fp32 accumulation), packed to L1 ONCE at the end — a cheap stream of FPU adds, negligible next to the
# multi-MB read, so it can't mask the read strategy.
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

// TRUE fp32 accumulation with bf16 data: add_tiles(..., acc_to_dest) adds A+B INTO the fp32 DEST adder
// without reading the running sum back through a Src register (which is what truncated the earlier
// binary_dest_reuse path to bf16 and stalled it at ~256). We add in[t] + 0 -> DEST, so DEST accumulates
// the whole bf16 stream exactly in fp32. The +0 operand is a zero tile built on-device (in[0]-in[0]),
// so no extra input tensor is needed.
void kernel_main() {
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = 0, cb_zero = 2, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out);

    // Build a zero tile: cb_zero[0] = in[0] - in[0]. (Peek the first chunk; do not pop it — the
    // accumulation loop below still consumes it.)
    cb_wait_front(cb_in, chunk_tiles);
    sub_tiles_init(cb_in, cb_in);
    tile_regs_acquire();
    sub_tiles(cb_in, cb_in, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_zero, 1);
    pack_tile(0, cb_zero);
    cb_push_back(cb_zero, 1);
    tile_regs_release();
    cb_wait_front(cb_zero, 1);

    // acc_to_dest=true => DEST += in[t] + 0, accumulating the whole stream in the fp32 DEST adder.
    add_tiles_init(cb_in, cb_zero, /*acc_to_dest=*/true);
    tile_regs_acquire();  // DEST starts zeroed; running sum stays here across the whole stream
    for (uint32_t c = 0; c < num_chunks; ++c) {
        if (c > 0) { cb_wait_front(cb_in, chunk_tiles); }  // chunk 0 is already fronted
        for (uint32_t t = 0; t < chunk_tiles; ++t) {
            add_tiles(cb_in, cb_zero, t, 0, 0);  // DEST += in[t] (+ 0)
        }
        cb_pop_front(cb_in, chunk_tiles);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_zero, 1);
}
"""

# ---- Writer (shared, all cores): write this core's single running-sum tile to its output slot. ----
_WRITER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t my_index = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);

    cb_wait_front(cb_out, 1);
    noc_async_write_tile(my_index, out_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
"""


def _worker_grid(device):
    """A fixed 2 × grid_x rectangle of worker cores; injector = top-left (0,0), inside the grid."""
    grid = device.compute_with_storage_grid_size()
    cols = grid.x
    cores = [(x, y) for y in range(2) for x in range(cols)]  # row-major
    return cores, 2, cols


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def create_program_descriptor(x_in, output, *, variant, chunk_rows):
    """`x_in` is the shared input [R, C] in DRAM; it is streamed in chunks of `chunk_rows` tile-rows
    × (C/32) cols. Each core folds the whole stream into one output tile."""
    if variant not in VARIANTS:
        raise ValueError(f"shared_input_reuse: variant must be one of {VARIANTS}, got {variant!r}")
    device = x_in.device()
    cores, rows, cols_grid = _worker_grid(device)
    n = len(cores)

    d_cols = list(x_in.shape)[1] // TILE  # input width in tiles
    row_tiles = list(x_in.shape)[0] // TILE  # input rows in tiles
    if row_tiles % chunk_rows:
        raise ValueError(f"shared_input_reuse: row tiles {row_tiles} must be a multiple of chunk_rows {chunk_rows}")
    chunk_tiles = chunk_rows * d_cols
    num_chunks = row_tiles // chunk_rows
    tile_bytes = x_in.buffer_aligned_page_size()
    all_crs = _crs(cores)

    # Double-buffered: the reader / injector fetches chunk c+1 while compute sums chunk c, so the
    # (cheap) tile-sum hides behind the fetch and the kernel stays read/mcast-bound.
    cb_in = ttnn.CBDescriptor(
        total_size=2 * chunk_tiles * tile_bytes,
        core_ranges=all_crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=x_in.dtype, page_size=tile_bytes)],
    )
    out_bytes = output.buffer_aligned_page_size()
    cb_out = ttnn.CBDescriptor(
        total_size=out_bytes,
        core_ranges=all_crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=output.dtype, page_size=out_bytes)
        ],
    )
    # Scratch: a single zero tile (built on-device) used as the +0 operand of the acc_to_dest add.
    cb_zero = ttnn.CBDescriptor(
        total_size=tile_bytes,
        core_ranges=all_crs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_ZERO, data_format=x_in.dtype, page_size=tile_bytes)
        ],
    )

    x_ct = ttnn.TensorAccessorArgs(x_in).get_compile_time_args()
    out_ct = ttnn.TensorAccessorArgs(output).get_compile_time_args()
    x_addr, out_addr = x_in.buffer_address(), output.buffer_address()

    # Compute + writer are IDENTICAL across variants and all cores — only the read path differs.
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_crs,
        compile_time_args=[chunk_tiles, num_chunks],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi2),
    )
    writer_rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        writer_rt[cx][cy] = [out_addr, i]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=_WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_crs,
        compile_time_args=[*out_ct],
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    if variant == "per_core_dram":
        reader_rt = ttnn.RuntimeArgs()
        for cx, cy in cores:
            reader_rt[cx][cy] = [x_addr]
        reader_kernel = ttnn.KernelDescriptor(
            kernel_source=_READER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=all_crs,
            compile_time_args=[chunk_tiles, num_chunks, *x_ct],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(
            kernels=[reader_kernel, compute_kernel, writer_kernel], semaphores=[], cbs=[cb_in, cb_zero, cb_out]
        )

    # mcast: injector = cores[0] = (0,0), a corner of the 2xC rect; the mcast_pipe SenderPipe broadcasts
    # to the rect and self-excludes (fan-out = the other 21). ttnn.Mcast2D emits the semaphores + the
    # McastArgs wire (compile-time block + per-core runtime args) the kernels decode.
    injector = cores[0]
    receivers = cores[1:]
    mc = ttnn.Mcast2D(device, all_crs, ttnn.CoreCoord(*injector), ttnn.McastConfig(handshake=True, base_sem_id=0))
    semaphores = mc.owned_semaphores()

    sender_ct = [CB_IN, *mc.compile_time_args(), chunk_tiles, tile_bytes, num_chunks, *x_ct]
    sender_rt = ttnn.RuntimeArgs()
    sender_rt[injector[0]][injector[1]] = [x_addr, *mc.runtime_args(ttnn.CoreCoord(*injector))]
    sender_kernel = ttnn.KernelDescriptor(
        kernel_source=_SENDER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_crs([injector]),
        compile_time_args=sender_ct,
        runtime_args=sender_rt,
        config=ttnn.ReaderConfigDescriptor(),  # NCRISC (the shared writer runs on BRISC — no conflict)
    )

    recv_ct = [CB_IN, *mc.compile_time_args(), chunk_tiles, num_chunks]
    recv_rt = ttnn.RuntimeArgs()
    for cx, cy in receivers:
        recv_rt[cx][cy] = list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))
    receiver_kernel = ttnn.KernelDescriptor(
        kernel_source=_RECEIVER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_crs(receivers),
        compile_time_args=recv_ct,
        runtime_args=recv_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(
        kernels=[sender_kernel, receiver_kernel, compute_kernel, writer_kernel],
        semaphores=semaphores,
        cbs=[cb_in, cb_zero, cb_out],
    )


def shared_input_reuse(x_in, output, *, variant="mcast", chunk_rows):
    """A 2×grid_x rectangle of cores each fold the shared input stream `x_in` into one output tile; `mcast`
    reads each chunk once on the top-left injector and broadcasts it. `output` holds one tile per core.
    Output is the identical (correct) running tile-sum on every core for both variants; only how each
    core obtains the stream differs."""
    descriptor = create_program_descriptor(x_in, output, variant=variant, chunk_rows=chunk_rows)
    return ttnn.generic_op([x_in, output], descriptor)
