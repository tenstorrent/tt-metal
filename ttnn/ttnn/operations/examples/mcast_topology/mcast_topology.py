# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-core operand DELIVERY on a 2-D (block-sharded) work split: redundant DRAM reads vs 1-D mcasts.

This example measures **only the delivery**. There is no compute kernel: the cores fetch the operand
slices a tiled matmul `C[M,N] = A[M,K] @ B[K,N]` would need for their share of the output, and stop.
The matmul is never performed, so nothing about the Matrix Unit can leak into the number.

The work split is **identical in both variants** and is fixed 2-D: a `Gr x Gc` core rectangle where
grid ROWS carry M and grid COLUMNS carry N, so core `(x=c, y=r)` owns output block `[M_r, N_c]` and
needs exactly two slices — `A[M_r, :]` and `B[:, N_c]`. Every core ends up holding the same bytes in
both variants; only HOW those bytes arrive differs:

  per_core_dram (baseline): every core reads both of its own slices straight from DRAM. Because the
      split is 2-D, that is redundant by construction — all `Gc` cores in a grid row want the SAME
      `A[M_r, :]`, and all `Gr` cores in a grid column want the SAME `B[:, N_c]`. DRAM therefore
      serves each A slice `Gc` times and each B slice `Gr` times.

  mcast_1d_pair (optimized): an operand is broadcast along the axis it does NOT vary with. `A` does
      not vary along a grid ROW, so column 0 of each row reads `A[M_r, :]` once and multicasts it
      across its row (`Mcast1D(PerRow)`). `B` does not vary down a grid COLUMN, so row 0 of each
      column reads `B[:, N_c]` once and multicasts it down (`Mcast1D(PerColumn)`). Each slice crosses
      DRAM exactly once; the copies travel core-to-core.

The topology follows from the split, and the naming inverts in a way worth internalising: a **2-D**
work split needs **1-D** multicasts (a source per line, feeding its own row / column), whereas a
**1-D** work split — M only, every core needing the whole of B — is the one that needs a **2-D**
multicast (a single injector feeding a whole rectangle). "More sharded" does not mean "bigger
broadcast"; it means each operand travels a shorter, narrower path, and the grid dimension an operand
does not travel along is the one carrying the other operand.

Correctness is checked without computing anything: each core writes a few PROBE tiles straight out of
the operand CBs it filled, and the test asserts tile-exactly that those are the tiles that core was
supposed to receive. That proves the delivery — every landing address, every slice mapping — while
keeping the written bytes negligible next to the transferred bytes, so the kernel stays
delivery-bound. See README.md.
"""

import ttnn

TILE = 32

CB_A = 0  # A block this core receives: [Mloc, Kt] tiles, m-major
CB_B = 1  # B block this core receives: [Kt, Nloc] tiles, k-major

VARIANTS = ("per_core_dram", "mcast_1d_pair")

# Probe tiles each core writes out of its filled CBs, purely to prove delivery:
#   0 -> first tile of the A block  (A[m0, 0])
#   1 -> first tile of the B block  (B[0, n0])
#   2 -> last  tile of the B block  (B[Kt-1, n0+Nloc-1])
PROBES = 3


# =====================================================================================
# per_core_dram (baseline) — the SAME 2-D split, but every core pulls both of its own slices
# straight from DRAM. No semaphores, no cross-core traffic; DRAM serves each A slice Gc times
# and each B slice Gr times.
# =====================================================================================
_PER_CORE_DRAM_READER = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_b = 1;
    constexpr uint32_t Mloc = get_compile_time_arg_val(0);
    constexpr uint32_t Nloc = get_compile_time_arg_val(1);
    constexpr uint32_t Kt   = get_compile_time_arg_val(2);
    constexpr uint32_t Nt   = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr auto a_args = TensorAccessorArgs<5>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t m0     = get_arg_val<uint32_t>(2);
    const uint32_t n0     = get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer a_buf(cb_a), b_buf(cb_b);
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);

    // A[M_r, :] -> [Mloc, Kt], m-major. Every core in this grid ROW reads these same tiles.
    a_buf.reserve_back(Mloc * Kt);
    for (uint32_t i = 0; i < Mloc; ++i) {
        for (uint32_t k = 0; k < Kt; ++k) {
            noc.async_read(a, a_buf, tile_bytes, {.page_id = (m0 + i) * Kt + k},
                           {.offset_bytes = (i * Kt + k) * tile_bytes});
        }
    }

    // B[:, N_c] -> [Kt, Nloc], k-major. Every core in this grid COLUMN reads these same tiles.
    b_buf.reserve_back(Kt * Nloc);
    for (uint32_t k = 0; k < Kt; ++k) {
        for (uint32_t n = 0; n < Nloc; ++n) {
            noc.async_read(b, b_buf, tile_bytes, {.page_id = k * Nt + (n0 + n)},
                           {.offset_bytes = (k * Nloc + n) * tile_bytes});
        }
    }
    noc.async_read_barrier();
    a_buf.push_back(Mloc * Kt);
    b_buf.push_back(Kt * Nloc);
}
"""


# =====================================================================================
# mcast_1d_pair — cut M across grid rows and N across grid columns. TWO 1-D multicasts.
#
# One reader source, specialized per core by two compile-time flags. Each core is a sender or a
# receiver on each of the two independent lines, so there are four (A-role x B-role) combinations
# and every core hosts exactly ONE reader kernel:
#     (x=0,y=0)  A-sender (across row 0)   + B-sender (down column 0)
#     (x=0,y>0)  A-sender (across row y)   + B-receiver
#     (x>0,y=0)  A-receiver                + B-sender (down column x)
#     (x>0,y>0)  A-receiver                + B-receiver
# =====================================================================================
_1D_PAIR_READER = r"""
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
    constexpr uint32_t cb_a = 0, cb_b = 1;
    constexpr uint32_t A_SENDS = get_compile_time_arg_val(0);  // this core injects A across its ROW
    constexpr uint32_t B_SENDS = get_compile_time_arg_val(1);  // this core injects B down its COLUMN

    // Two independent mcast families on one grid: the PerRow family (A) then the PerColumn family
    // (B). Each self-parses a fixed 6-word CT block and a 6-word RT block, laid out back to back.
    constexpr auto mc_a = McastArgs</*CT=*/2, /*RT=*/2>();  // RT 0,1 = a_addr,b_addr
    constexpr auto mc_b = McastArgs<mc_a.next_compile_time_args_offset(), mc_a.next_runtime_args_offset()>();
    constexpr uint32_t S = mc_b.next_compile_time_args_offset();
    constexpr uint32_t Mloc = get_compile_time_arg_val(S + 0);
    constexpr uint32_t Nloc = get_compile_time_arg_val(S + 1);
    constexpr uint32_t Kt   = get_compile_time_arg_val(S + 2);
    constexpr uint32_t Nt   = get_compile_time_arg_val(S + 3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(S + 4);
    constexpr auto a_args = TensorAccessorArgs<S + 5>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t scalars = mc_b.next_runtime_args_offset();
    const uint32_t m0 = get_arg_val<uint32_t>(scalars + 0);  // this core's first output tile-row
    const uint32_t n0 = get_arg_val<uint32_t>(scalars + 1);  // this core's first output tile-col

    Noc noc;
    CircularBuffer a_buf(cb_a), b_buf(cb_b);

    // ---- A: shared ACROSS the grid row -> Mcast1D(PerRow) ----
    a_buf.reserve_back(Mloc * Kt);
    if constexpr (A_SENDS) {
        const auto a = TensorAccessor(a_args, a_addr);
        const uint32_t a_dst = a_buf.get_write_ptr();
        for (uint32_t i = 0; i < Mloc; ++i) {
            for (uint32_t k = 0; k < Kt; ++k) {
                noc.async_read(a, a_buf, tile_bytes, {.page_id = (m0 + i) * Kt + k},
                               {.offset_bytes = (i * Kt + k) * tile_bytes});
            }
        }
        noc.async_read_barrier();
        auto pipe = mc_a.sender(noc);
        if constexpr (mc_a.has_receivers) { pipe.send(a_dst, a_dst, Mloc * Kt * tile_bytes); }
    } else {
        auto pipe = mc_a.receiver(noc);
        pipe.receive();
    }
    a_buf.push_back(Mloc * Kt);

    // ---- B: shared DOWN the grid column -> Mcast1D(PerColumn) ----
    b_buf.reserve_back(Kt * Nloc);
    if constexpr (B_SENDS) {
        const auto b = TensorAccessor(b_args, b_addr);
        const uint32_t b_dst = b_buf.get_write_ptr();
        for (uint32_t k = 0; k < Kt; ++k) {
            for (uint32_t n = 0; n < Nloc; ++n) {
                noc.async_read(b, b_buf, tile_bytes, {.page_id = k * Nt + (n0 + n)},
                               {.offset_bytes = (k * Nloc + n) * tile_bytes});
            }
        }
        noc.async_read_barrier();
        auto pipe = mc_b.sender(noc);
        if constexpr (mc_b.has_receivers) { pipe.send(b_dst, b_dst, Kt * Nloc * tile_bytes); }
    } else {
        auto pipe = mc_b.receiver(noc);
        pipe.receive();
    }
    b_buf.push_back(Kt * Nloc);
}
"""


# =====================================================================================
# Probe writer — IDENTICAL source for both variants. NOT compute: it copies three already-delivered
# tiles straight out of the operand CBs so the test can prove every core got the right block. Three
# tiles per core is negligible next to the transferred operands, so the kernel stays delivery-bound.
# =====================================================================================
_PROBE_WRITER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Mloc   = get_compile_time_arg_val(0);
    constexpr uint32_t Nloc   = get_compile_time_arg_val(1);
    constexpr uint32_t Kt     = get_compile_time_arg_val(2);
    constexpr uint32_t PROBES = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    constexpr uint32_t cb_a = 0, cb_b = 1;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot     = get_arg_val<uint32_t>(1);  // this core's probe slot index

    const uint32_t tile_bytes = get_tile_size(cb_a);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    cb_wait_front(cb_a, Mloc * Kt);
    cb_wait_front(cb_b, Kt * Nloc);
    const uint32_t a_base = get_read_ptr(cb_a);
    const uint32_t b_base = get_read_ptr(cb_b);

    // 0: A[m0, 0]   1: B[0, n0]   2: B[Kt-1, n0+Nloc-1]
    noc_async_write_tile(slot * PROBES + 0, out, a_base);
    noc_async_write_tile(slot * PROBES + 1, out, b_base);
    noc_async_write_tile(slot * PROBES + 2, out, b_base + (Kt * Nloc - 1) * tile_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_a, Mloc * Kt);
    cb_pop_front(cb_b, Kt * Nloc);
}
"""


# =====================================================================================
# Host: layout, work split, program construction
# =====================================================================================


def _divisors_desc(n):
    return [d for d in range(n, 0, -1) if n % d == 0]


def largest_divisor_at_most(n, cap):
    """Largest d dividing n with d <= cap (1 always qualifies)."""
    for d in _divisors_desc(n):
        if d <= cap:
            return d
    return 1


def layout(device, mt, nt):
    """THE work split — one geometry, shared by both variants so only the transport differs.

    A `Gr x Gc` rectangle with grid ROWS carrying M and grid COLUMNS carrying N: `Gr | Mt` and
    `Gc | Nt`, each as large as the device grid allows.
    """
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    gr = largest_divisor_at_most(mt, gy)
    gc = largest_divisor_at_most(nt, gx)
    return {"grid": (gx, gy), "rows": gr, "cols": gc, "cores": gr * gc}


def core_assignment(device, mt, nt):
    """(x, y) -> (m0, n0, Mloc, Nloc, slot): the output block each core fetches operands for.

    Identical for both variants — that is the point: same cores, same slices, same resident bytes.
    """
    lay = layout(device, mt, nt)
    rows, cols = lay["rows"], lay["cols"]
    mloc, nloc = mt // rows, nt // cols
    return {(x, y): (y * mloc, x * nloc, mloc, nloc, y * cols + x) for y in range(rows) for x in range(cols)}


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _cb(index, num_tiles, tile_bytes, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_tiles * tile_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
    )


def validate(a, b, variant):
    if variant not in VARIANTS:
        raise ValueError(f"mcast_topology: variant must be one of {VARIANTS}, got {variant!r}")
    for name, t in (("A", a), ("B", b)):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"mcast_topology: {name} must be bfloat16 TILE_LAYOUT")
    if list(a.shape)[1] != list(b.shape)[0]:
        raise ValueError(f"mcast_topology: inner dims must match, got A={a.shape} B={b.shape}")


def _probe_writer(all_crs, mloc, nloc, kt, out_ct, out_addr, assign):
    rt = ttnn.RuntimeArgs()
    for (cx, cy), (_, _, _, _, slot) in assign.items():
        rt[cx][cy] = [out_addr, slot]
    return ttnn.KernelDescriptor(
        kernel_source=_PROBE_WRITER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=all_crs,
        compile_time_args=[mloc, nloc, kt, PROBES, *out_ct],
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )


def create_program_descriptor(a, b, probes, *, variant):
    validate(a, b, variant)
    device = a.device()
    mt, kt = list(a.shape)[0] // TILE, list(a.shape)[1] // TILE
    nt = list(b.shape)[1] // TILE
    tile_bytes = a.buffer_aligned_page_size()
    assign = core_assignment(device, mt, nt)
    lay = layout(device, mt, nt)
    rows, cols = lay["rows"], lay["cols"]
    all_crs = _rect(0, 0, cols - 1, rows - 1)

    a_ct = ttnn.TensorAccessorArgs(a).get_compile_time_args()
    b_ct = ttnn.TensorAccessorArgs(b).get_compile_time_args()
    o_ct = ttnn.TensorAccessorArgs(probes).get_compile_time_args()
    a_addr, b_addr, o_addr = a.buffer_address(), b.buffer_address(), probes.buffer_address()

    any_core = next(iter(assign))
    _, _, mloc, nloc, _ = assign[any_core]

    if variant == "per_core_dram":
        semaphores = []
        rt = ttnn.RuntimeArgs()
        for (cx, cy), (m0, n0, _, _, _) in assign.items():
            rt[cx][cy] = [a_addr, b_addr, m0, n0]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_PER_CORE_DRAM_READER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=all_crs,
                compile_time_args=[mloc, nloc, kt, nt, tile_bytes, *a_ct, *b_ct],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
    else:
        # TWO 1-D families on the same grid, on disjoint semaphore ids:
        #   A rides PerRow    (sender = column 0 of each row)  -> base_sem_id 0
        #   B rides PerColumn (sender = row 0 of each column)  -> base_sem_id 2
        mc_a = ttnn.Mcast1D(
            device, all_crs, ttnn.Mcast1DShape.PerRow, 0, ttnn.McastConfig(handshake=True, base_sem_id=0)
        )
        mc_b = ttnn.Mcast1D(
            device, all_crs, ttnn.Mcast1DShape.PerColumn, 0, ttnn.McastConfig(handshake=True, base_sem_id=2)
        )
        semaphores = [*mc_a.owned_semaphores(), *mc_b.owned_semaphores()]

        groups = {
            (1, 1): [(0, 0)],
            (1, 0): [(0, y) for y in range(1, rows)],
            (0, 1): [(x, 0) for x in range(1, cols)],
            (0, 0): [(x, y) for y in range(1, rows) for x in range(1, cols)],
        }
        kernels = []
        for (a_sends, b_sends), members in groups.items():
            if not members:
                continue
            rt = ttnn.RuntimeArgs()
            for cx, cy in members:
                core = ttnn.CoreCoord(cx, cy)
                m0, n0, _, _, _ = assign[(cx, cy)]
                rt[cx][cy] = [a_addr, b_addr, *mc_a.runtime_args(core), *mc_b.runtime_args(core), m0, n0]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_1D_PAIR_READER,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=_crs(members),
                    compile_time_args=[
                        a_sends,
                        b_sends,
                        *mc_a.compile_time_args(),
                        *mc_b.compile_time_args(),
                        mloc,
                        nloc,
                        kt,
                        nt,
                        tile_bytes,
                        *a_ct,
                        *b_ct,
                    ],
                    runtime_args=rt,
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )

    writer = _probe_writer(all_crs, mloc, nloc, kt, o_ct, o_addr, assign)
    cbs = [
        _cb(CB_A, mloc * kt, tile_bytes, a.dtype, all_crs),
        _cb(CB_B, kt * nloc, tile_bytes, a.dtype, all_crs),
    ]
    return ttnn.ProgramDescriptor(kernels=[*kernels, writer], semaphores=semaphores, cbs=cbs)


def num_cores(device, mt, nt):
    return layout(device, mt, nt)["cores"]


def mcast_topology(a, b, *, variant="mcast_1d_pair"):
    """Deliver the operands a tiled `A @ B` would need to every participating core — no compute.

    Both variants use the SAME 2-D work split, so each core ends up holding the same slices;
    `per_core_dram` has every core read them from DRAM itself (redundantly, Gc x for A and Gr x
    for B), while `mcast_1d_pair` reads each slice once per line and broadcasts it.
    Returns the per-core PROBE tiles, which prove what each core actually received.
    """
    validate(a, b, variant)
    device = a.device()
    mt = list(a.shape)[0] // TILE
    nt = list(b.shape)[1] // TILE
    slots = num_cores(device, mt, nt)
    probes = ttnn.allocate_tensor_on_device(
        ttnn.Shape([slots * PROBES * TILE, TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(a, b, probes, variant=variant)
    return ttnn.generic_op([a, b, probes], descriptor)
