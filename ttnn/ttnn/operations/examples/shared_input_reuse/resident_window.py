# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-shot broadcast of a small shared window that every Tensix core keeps resident in L1.

The second situation of this example. Every Tensix core streams its OWN rows of a large input from DRAM
and writes its own output rows (a DRAM-bound read + write over the grid), and every core ALSO needs the
same small vector `w` of W elements, resident in L1 for the whole run:

    out[r, c] = x[r, c] * w[c]        x: [R, W] (Layout::TILE, bf16, DRAM TensorMemoryLayout::INTERLEAVED)
                                      w: [1, W] -> one tile-row of W/32 tiles (row 0 valid), bf16, DRAM

x is split by tile-rows across a rectangle of Tensix cores (a contiguous range of tile ids per core). The
window is W/32 tiles = W/32 x 2048 bytes in L1 on every Tensix core. The variants differ ONLY in how w
gets into each core's L1:

  per_core        every Tensix core reads all of w from DRAM itself, on its reader, after its first
                  x block (so the first x block's DRAM latency overlaps the w read).
  oneshot_mcast   one injector (the rectangle's top-left Tensix core) reads w from DRAM in chunks and
                  NoC-multicasts each chunk to the whole rectangle, then multicasts a semaphore SET
                  (value = chunks landed). No handshake: the window's L1 is written once per launch,
                  so receivers never have to say "slot free". Receivers push each chunk to compute as
                  soon as the semaphore says it landed, also after their first x block.
  lockstep_mcast  the same multicast with an ack-then-broadcast handshake per chunk (the injector waits
                  for every receiver's "ready" before each chunk), and every Tensix core fetches the
                  whole window BEFORE its first x block — the stream waits for the broadcast.

Compute (mul_tiles_bcast_rows) and writer are identical in every variant. See README.md.
"""

import ttnn

TILE = 32
CB_X = 0  # x blocks streamed from DRAM (2 blocks deep)
CB_W = 1  # the resident window: all W/32 tiles, written once per launch
CB_OUT = 16  # output blocks (2 blocks deep)
SEM_VALID = 0  # receivers: number of window chunks that have landed (set by the injector's multicast)
SEM_ACK = 1  # injector (lockstep only): number of receivers ready for the next chunk

WINDOW_VARIANTS = ("per_core", "oneshot_mcast", "lockstep_mcast")

_READER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_w = 1;

void kernel_main() {
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t wt = get_compile_time_arg_val(1);           // window tiles (W / 32)
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(2);  // window tiles per chunk / multicast
    constexpr uint32_t sem_valid_id = get_compile_time_arg_val(3);
    constexpr uint32_t sem_ack_id = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w_addr = get_arg_val<uint32_t>(a++);
    const uint32_t start_tile = get_arg_val<uint32_t>(a++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t is_injector = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t lo_x = get_arg_val<uint32_t>(a++);  // rectangle corners, NoC (virtual) coordinates
    [[maybe_unused]] const uint32_t lo_y = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t hi_x = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t hi_y = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t num_dests = get_arg_val<uint32_t>(a++);  // receivers = rectangle - injector
    [[maybe_unused]] const uint32_t inj_x = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t inj_y = get_arg_val<uint32_t>(a++);

    const uint32_t tile_bytes = get_tile_size(cb_x);
    const auto x_acc = TensorAccessor(x_args, x_addr, tile_bytes);
    const auto w_acc = TensorAccessor(w_args, w_addr, tile_bytes);
    constexpr uint32_t num_chunks = (wt + chunk_tiles - 1) / chunk_tiles;

    auto read_x_block = [&](uint32_t t0) {
        const uint32_t n = (num_tiles - t0 < block_tiles) ? (num_tiles - t0) : block_tiles;
        cb_reserve_back(cb_x, n);
        const uint32_t dst = get_write_ptr(cb_x);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read_tile(start_tile + t0 + i, x_acc, dst + i * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, n);
    };

    auto get_window = [&]() {
#if defined(WINDOW_PER_CORE)
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t n = (wt - c * chunk_tiles < chunk_tiles) ? (wt - c * chunk_tiles) : chunk_tiles;
            cb_reserve_back(cb_w, n);
            const uint32_t dst = get_write_ptr(cb_w);
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_read_tile(c * chunk_tiles + i, w_acc, dst + i * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, n);
        }
#else
        const uint32_t valid_addr = get_semaphore(sem_valid_id);
        volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
        [[maybe_unused]] const uint32_t ack_addr = get_semaphore(sem_ack_id);
        [[maybe_unused]] volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_addr);
        // NoC 0 multicasts from the low corner to the high one; NoC 1 the other way round.
        const uint32_t sx = noc_index == 0 ? lo_x : hi_x, sy = noc_index == 0 ? lo_y : hi_y;
        const uint32_t ex = noc_index == 0 ? hi_x : lo_x, ey = noc_index == 0 ? hi_y : lo_y;
        for (uint32_t c = 0; c < num_chunks; ++c) {
            const uint32_t n = (wt - c * chunk_tiles < chunk_tiles) ? (wt - c * chunk_tiles) : chunk_tiles;
            cb_reserve_back(cb_w, n);
            const uint32_t dst = get_write_ptr(cb_w);  // same L1 address on every Tensix core
            if (is_injector) {
                for (uint32_t i = 0; i < n; ++i) {
                    noc_async_read_tile(c * chunk_tiles + i, w_acc, dst + i * tile_bytes);
                }
                noc_async_read_barrier();
#if defined(WINDOW_LOCKSTEP)
                noc_semaphore_wait(ack, num_dests);  // every receiver said "ready for chunk c"
                noc_semaphore_set(ack, 0);
#endif
                if (num_dests > 0) {
                    noc_async_writes_flushed();  // the previous chunk's semaphore multicast has read the word
                    *valid = c + 1;
                    // Data, then the "chunks landed" word: same NoC, multicast VC, linked -> arrive in order.
                    noc_async_write_multicast(
                        dst, get_noc_multicast_addr(sx, sy, ex, ey, dst), n * tile_bytes, num_dests, true);
                    noc_semaphore_set_multicast(valid_addr, get_noc_multicast_addr(sx, sy, ex, ey, valid_addr), num_dests);
                }
            } else {
#if defined(WINDOW_LOCKSTEP)
                noc_semaphore_inc(get_noc_addr(inj_x, inj_y, ack_addr), 1);
#endif
                noc_semaphore_wait_min(valid, c + 1);
            }
            cb_push_back(cb_w, n);  // compute may start on the part of the window that landed
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
#endif
    };

#if defined(WINDOW_LOCKSTEP)
    get_window();  // the whole broadcast first, then the stream
    for (uint32_t t0 = 0; t0 < num_tiles; t0 += block_tiles) {
        read_x_block(t0);
    }
#else
    read_x_block(0);  // first x block in flight, then the window, then the rest of the stream
    get_window();
    for (uint32_t t0 = block_tiles; t0 < num_tiles; t0 += block_tiles) {
        read_x_block(t0);
    }
#endif
}
"""

_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/bcast.h"

void kernel_main() {
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t wt = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_out = 16;
    const uint32_t start_tile = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    mul_bcast_rows_init(cb_x, cb_w);
    uint32_t w_ready = 0;  // window tiles known to be at the front of cb_w (never popped)
    for (uint32_t t0 = 0; t0 < num_tiles; t0 += block_tiles) {
        const uint32_t n = (num_tiles - t0 < block_tiles) ? (num_tiles - t0) : block_tiles;
        if (w_ready < wt) {  // first pass over the window: wait only for the columns this block needs
            const uint32_t last_col = (start_tile + t0 + n - 1) % wt;
            const uint32_t need = (t0 + n >= wt) ? wt : last_col + 1;
            if (need > w_ready) {
                cb_wait_front(cb_w, need);
                w_ready = need;
            }
        }
        cb_wait_front(cb_x, n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles_bcast_rows(cb_x, cb_w, i, (start_tile + t0 + i) % wt, i);  // x * w (row 0 broadcast)
        }
        tile_regs_commit();
        cb_pop_front(cb_x, n);
        cb_reserve_back(cb_out, n);
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, n);
    }
}
"""

_WRITER_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_out = 16;
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    for (uint32_t t0 = 0; t0 < num_tiles; t0 += block_tiles) {
        const uint32_t n = (num_tiles - t0 < block_tiles) ? (num_tiles - t0) : block_tiles;
        cb_wait_front(cb_out, n);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write_tile(start_tile + t0 + i, out_acc, src + i * tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, n);
    }
}
"""


def window_rectangle(device, num_cores=None):
    """The squarest cols x rows rectangle of Tensix cores with cols * rows == num_cores (default: the
    whole compute grid), anchored at logical (0, 0). Returns (cols, rows)."""
    grid = device.compute_with_storage_grid_size()
    if num_cores is None:
        return grid.x, grid.y
    best = None
    for cols in range(1, grid.x + 1):
        if num_cores % cols:
            continue
        rows = num_cores // cols
        if rows <= grid.y and (best is None or abs(cols - rows) < abs(best[0] - best[1])):
            best = (cols, rows)
    if best is None:
        raise ValueError(f"no rectangle of {num_cores} Tensix cores fits the {grid.x}x{grid.y} grid")
    return best


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def create_window_program_descriptor(x, w, output, *, variant, num_cores=None, chunk_tiles=32, block_tiles=8):
    """out = x * w (w broadcast down the rows). `x`/`output` [R, W] and `w` [1, W] are Layout::TILE bf16
    in DRAM; x's tile-rows are split over a rectangle of `num_cores` Tensix cores. `chunk_tiles` is the
    window chunk per DRAM read batch / multicast; `block_tiles` the x block per CB push."""
    if variant not in WINDOW_VARIANTS:
        raise ValueError(f"shared_input_reuse window: variant must be one of {WINDOW_VARIANTS}, got {variant!r}")
    device = x.device()
    cols, rows = window_rectangle(device, num_cores)
    cores = [(cx, cy) for cy in range(rows) for cx in range(cols)]  # row-major; cores[0] = injector
    n = len(cores)

    r, wdim = list(x.shape)
    if r % TILE or wdim % TILE or list(w.shape)[-1] != wdim:
        raise ValueError(f"x must be [R, W] with R, W multiples of 32 and w [1, W]; got {x.shape}, {w.shape}")
    ht, wt = r // TILE, wdim // TILE
    if ht < n:
        raise ValueError(f"need at least one tile-row per Tensix core: {ht} tile-rows < {n} Tensix cores")
    tile_bytes = x.buffer_aligned_page_size()
    all_crs = _crs(cores)

    def cb(index, pages, tensor):
        return ttnn.CBDescriptor(
            total_size=pages * tile_bytes,
            core_ranges=all_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=tensor.dtype, page_size=tile_bytes)
            ],
        )

    cbs = [cb(CB_X, 2 * block_tiles, x), cb(CB_W, wt, w), cb(CB_OUT, 2 * block_tiles, output)]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_VALID, core_ranges=all_crs, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_ACK, core_ranges=all_crs, initial_value=0),
    ]

    def virt(xy):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
        return c.x, c.y

    lo, hi = virt((0, 0)), virt((cols - 1, rows - 1))
    inj = virt(cores[0])

    base, extra = divmod(ht, n)
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    row = 0
    for i, (cx, cy) in enumerate(cores):
        my_rows = base + (1 if i < extra else 0)
        start, count = row * wt, my_rows * wt
        row += my_rows
        reader_rt[cx][cy] = [
            x.buffer_address(),
            w.buffer_address(),
            start,
            count,
            int(i == 0),
            *lo,
            *hi,
            n - 1,
            *inj,
        ]
        compute_rt[cx][cy] = [start, count]
        writer_rt[cx][cy] = [output.buffer_address(), start, count]

    define = {"per_core": "WINDOW_PER_CORE", "oneshot_mcast": "WINDOW_ONESHOT", "lockstep_mcast": "WINDOW_LOCKSTEP"}
    x_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    w_ct = ttnn.TensorAccessorArgs(w).get_compile_time_args()
    out_ct = ttnn.TensorAccessorArgs(output).get_compile_time_args()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=all_crs,
            compile_time_args=[block_tiles, wt, chunk_tiles, SEM_VALID, SEM_ACK, *x_ct, *w_ct],
            defines=[(define[variant], "1")],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=all_crs,
            compile_time_args=[block_tiles, wt],
            runtime_args=compute_rt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=all_crs,
            compile_time_args=[block_tiles, *out_ct],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def scale_rows_by_window(x, w, output, *, variant="oneshot_mcast", num_cores=None, chunk_tiles=32):
    """out = x * w on a rectangle of `num_cores` Tensix cores; `variant` picks how w reaches L1."""
    descriptor = create_window_program_descriptor(
        x, w, output, variant=variant, num_cores=num_cores, chunk_tiles=chunk_tiles
    )
    return ttnn.generic_op([x, w, output], descriptor)
