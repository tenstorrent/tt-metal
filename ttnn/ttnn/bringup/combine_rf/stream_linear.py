# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""stream_linear — y [M, N] = x [M, K] @ W [K, N] for M = 32 x (grid rows) rows (K3: 320; bf16 x, bf8 W) as a 2D
matmul whose operands are read from DRAM by every core: ttnn's 2D matmul reads the weights through its 12 top-row
cores (one column block each, 2.4 MB at K3's latent down) and runs at ~45% of the DRAM floor at 320 rows
(zz_scratch_k3_dense_mm.py).

  grid     GY rows x GX columns; core (cx, cy): output row tile cy x PN(cx) tile columns from n0(cx), in NP passes of
           PNP(cx) columns, each accumulated over all of K in the fp32 DST and packed once
  x        resident: a row's x (Kt tiles) in every core of the row; each core of the row reads ~Kt / GX tiles of it from
           DRAM at the start, multicasts them to the row (NOC0) and, once landed, credits every row member's x count;
           the compute sees x once per pass (the CB is exactly Kt tiles: its FIFO wraps onto the same data)
  W        a ring along each column: step i = p NBLK + j (pass p, K block j of KB tiles), piece = KB x PNP tiles of
           column cx's pass-p columns, owner (cx, i % GY) - read from DRAM one own piece ahead (three staging slots),
           multicast (loopback) once every core of the column freed ring slot i % NB, then the slot's valid semaphore
           (= i + 1). The receivers (NCRISC) credit steps up to NB ahead, as far as the ring has room.
  stacked  W: [GX NP Kt 32, PNPX 32] bf8 TILE (column cx, pass p: its PNP(cx) columns, zero padded to PNPX)
  y        bf16 TILE, DRAM interleaved (NCRISC, at the end)
History (K3 latent down, ttnn 117 us): 1D (x multicast to the whole grid) 210 us; 2D with x and W rings and ttnn's
compute kernel 220 us; own compute, x + W rings 133-230 us - ~1 us of credit / valid latency per step with both a row
and a column ring per step.
"""

import os

import torch

import ttnn

# semaphores: x pieces arrived (0), W free [1, 1 + NB), W valid [1 + NB, 1 + 2 NB)
_SEND = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t KT = get_compile_time_arg_val(0), KB = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4), NB = get_compile_time_arg_val(5);
constexpr uint32_t GY = get_compile_time_arg_val(6), WTB = get_compile_time_arg_val(7), NXP = get_compile_time_arg_val(8);
constexpr uint32_t GX = get_compile_time_arg_val(9);
constexpr auto w_args = TensorAccessorArgs<10>();
constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr uint32_t WB = KB * PNPX * WTB, NS = NP * NBLK, SF = 1, SV = 1 + NB;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0), cx = get_arg_val<uint32_t>(1), cy = get_arg_val<uint32_t>(2);
    const uint32_t cxv = get_arg_val<uint32_t>(3), cy0 = get_arg_val<uint32_t>(4), cy1 = get_arg_val<uint32_t>(5);
    const uint32_t x_addr = get_arg_val<uint32_t>(6), rx0 = get_arg_val<uint32_t>(7), ry = get_arg_val<uint32_t>(8);
    const uint32_t rx1 = get_arg_val<uint32_t>(9), x0 = get_arg_val<uint32_t>(10), xn = get_arg_val<uint32_t>(11);
    // (then GX: my row's x coordinates)
    const auto wg = TensorAccessor(w_args, w_addr, WTB);
    const auto xg = TensorAccessor(x_args, x_addr, 2048);
    const uint32_t xres = get_write_ptr(0), wring = get_write_ptr(1), xstage = get_write_ptr(8);
    const uint32_t wstage = get_write_ptr(9), words = get_write_ptr(10);
    // my 1 / GX of my row's x: DRAM -> staging -> every row member's resident x; then credit each one's x count
    for (uint32_t t = 0; t < xn; ++t) noc_async_read(xg.get_noc_addr(cy * KT + x0 + t), xstage + t * 2048, 2048);
    noc_async_read_barrier();
    noc_async_write_multicast_loopback_src(xstage, get_noc_multicast_addr(rx0, ry, rx1, ry, xres + x0 * 2048), xn * 2048, GX);
    noc_async_write_barrier();  // (landed everywhere before the counts)
    for (uint32_t m = 0; m < GX; ++m) noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(12 + m), ry, get_semaphore(0)), 1);
    // own W pieces: steps cy, cy + GY, ..; the piece of step i = (pass i / NBLK, block i % NBLK), slot (own #) % 3
    auto read_w = [&](uint32_t i, uint32_t slot) {
        if constexpr ((PROBE & 1) == 0) {
            const uint32_t p0 = ((cx * NP + i / NBLK) * KT + (i % NBLK) * KB) * PNPX;
            for (uint32_t t = 0; t < KB * PNPX; ++t) noc_async_read(wg.get_noc_addr(p0 + t), wstage + slot * WB + t * WTB, WTB);
        }
    };
    if (cy < NS) read_w(cy, 0);
    volatile tt_l1_ptr uint32_t* wv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);
    uint32_t wuse[NB];
    for (uint32_t s = 0; s < NB; ++s) wuse[s] = 0;
    uint32_t wo = 0;  // (own W pieces sent)
    for (uint32_t i = cy; i < NS; i += GY) {
        const uint32_t s = i % NB;
        noc_async_read_barrier();  // (this piece: read one own piece ago)
        if (i + GY < NS) {
            noc_async_writes_flushed();  // (the next slot's previous piece, three own pieces ago, has left)
            read_w(i + GY, (wo + 1) % 3);
        }
        wuse[s] += 1;
        if constexpr ((PROBE & 2) == 0) {
            noc_semaphore_wait_min(sem(SF + s), GY * wuse[s]);
            noc_async_write_multicast_loopback_src(
                wstage + (wo % 3) * WB, get_noc_multicast_addr(cxv, cy0, cxv, cy1, wring + s * WB), (PROBE & 8) ? 64 : WB, GY);
            wv[(wo & 3) * 4] = i + 1;  // (a ring of value words, 16 B apart: no flush before reuse)
            noc_semaphore_set_multicast_loopback_src(
                words + (wo & 3) * 16, get_noc_multicast_addr(cxv, cy0, cxv, cy1, get_semaphore(SV + s)), GY);
        }
        ++wo;
    }
    noc_async_write_barrier();
}
"""

_RECV = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t KB = get_compile_time_arg_val(0), PNP = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4), NB = get_compile_time_arg_val(5);
constexpr uint32_t GX = get_compile_time_arg_val(6), GY = get_compile_time_arg_val(7), NT = get_compile_time_arg_val(8);
constexpr uint32_t KT = get_compile_time_arg_val(9), NXP = get_compile_time_arg_val(10);
constexpr auto y_args = TensorAccessorArgs<11>();
constexpr auto x_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
constexpr uint32_t NS = NP * NBLK, SF = 1, SV = 1 + NB;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0), cy = get_arg_val<uint32_t>(1), n0 = get_arg_val<uint32_t>(2);
    const uint32_t cxv = get_arg_val<uint32_t>(3), x_addr = get_arg_val<uint32_t>(4);
    const auto yg = TensorAccessor(y_args, y_addr, 2048);
    uint32_t c = 0, r = 0;
    while (r < NS) {
        // credit W step c (slots r .. c free: compute popped up to c - NB)
        if (c < NS && c < r + NB && cb_pages_reservable_at_back(1, KB * PNPX * (c - r + 1))) {
            if constexpr ((PROBE & 2) == 0) {
                noc_semaphore_inc(get_noc_addr(cxv, get_arg_val<uint32_t>(10 + c % GY), get_semaphore(SF + c % NB)), 1);
            }
            ++c;
        }
        // receive step r: its x piece resident, its W piece valid
        if (r < c) {
            invalidate_l1_cache();
            if ((PROBE & 2) != 0 || (*sem(0) >= GX && *sem(SV + r % NB) >= r + 1)) {
                cb_reserve_back(0, KB);
                cb_push_back(0, KB);
                cb_reserve_back(1, KB * PNPX);
                cb_push_back(1, KB * PNPX);
                ++r;
            }
        }
    }
    for (uint32_t p = 0; p < NP; ++p) {  // (y: NP passes x PNP tiles)
        cb_wait_front(4, PNP);
        const uint32_t l1 = get_read_ptr(4);
        for (uint32_t t = 0; t < PNP; ++t) noc_async_write(l1 + t * 2048, yg.get_noc_addr(cy * NT + n0 + p * PNP + t), 2048);
        noc_async_write_barrier();
        cb_pop_front(4, PNP);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();  // (no unacknowledged atomic past the kernel: the next program's barrier)
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
constexpr uint32_t KB = get_compile_time_arg_val(0), PNP = get_compile_time_arg_val(1), PNPX = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NP = get_compile_time_arg_val(4);
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 4);  // (matmul: in1 -> SrcA, in0 -> SrcB)
    matmul_block_init(0, 1, 0, PNP, 1, KB);
    for (uint32_t p = 0; p < NP; ++p) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < NBLK; ++j) {
            cb_wait_front(0, KB);
            cb_wait_front(1, KB * PNPX);
            if constexpr ((PROBE & 4) == 0) {
                for (uint32_t k = 0; k < KB; ++k) matmul_block(0, 1, k, k * PNPX, 0, false, PNP, 1, KB);
            }
            cb_pop_front(0, KB);
            cb_pop_front(1, KB * PNPX);
        }
        tile_regs_commit();
        cb_reserve_back(4, PNP);
        tile_regs_wait();
        for (uint32_t t = 0; t < PNP; ++t) pack_tile(t, 4);
        tile_regs_release();
        cb_push_back(4, PNP);
    }
}
"""


def column_split(Nt, GX, unit):
    """Per-column output tiles: multiples of `unit` (the pass count), as even as possible."""
    assert Nt % unit == 0, (Nt, unit)
    base, rem = divmod(Nt // unit, GX)
    return [(base + (1 if i < rem else 0)) * unit for i in range(GX)]


def stack_weight(w, pns, NP):
    """W [K, N] torch -> [GX NP K, PNPX 32]: column cx, pass p at rows (cx NP + p) K .. (zero padded to PNPX)."""
    K, _ = w.shape
    pnpx = max(pns) // NP
    out = torch.zeros(len(pns) * NP * K, pnpx * 32, dtype=w.dtype)
    n0 = 0
    for c, pn in enumerate(pns):
        pnp = pn // NP
        for p in range(NP):
            r = (c * NP + p) * K
            out[r : r + K, : pnp * 32] = w[:, (n0 + p * pnp) * 32 : (n0 + (p + 1) * pnp) * 32]
        n0 += pn
    return out


def stream_linear_prepare(mesh_device, *, M, K, N, kb=16, nb=7, np_=None, fidelity=None, w_dtype=ttnn.bfloat8_b):
    """Returns (run, layout): run(x [M, K] bf16 TILE DRAM, upload_stacked(mesh, W, layout)) -> y [M, N] bf16 TILE DRAM."""
    grid = mesh_device.compute_with_storage_grid_size()
    Mt, Kt, Nt = M // 32, K // 32, N // 32
    GX, GY = grid.x, Mt
    assert M % 32 == 0 and Mt <= grid.y, "one output row tile per grid row"
    assert 1 + 2 * nb <= 16, nb
    full = os.environ.get("SL_FULL_SYNC", "0") == "1"
    dst = 8 if full else 4  # (a pass's width: the fp32 DST tiles)
    NP = np_ or 1
    while (Nt % NP) or max(column_split(Nt, GX, NP)) // NP > dst:
        NP += 1
    pns = column_split(Nt, GX, NP)
    pnpx = max(pns) // NP
    assert Kt % kb == 0, (Kt, kb)
    nblk = Kt // kb
    wtb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}[w_dtype]
    fid = fidelity if fidelity is not None else getattr(ttnn.MathFidelity, os.environ.get("SL_FID", "HiFi2"))
    wb = kb * pnpx * wtb
    xb_, xr_ = divmod(Kt, GX)  # (x pieces: column cx reads xcnt[cx] tiles of its row's x from xoff[cx])
    xcnt = [xb_ + (1 if i < xr_ else 0) for i in range(GX)]
    xoff = [sum(xcnt[:i]) for i in range(GX)]
    xpt = max(xcnt)
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GX - 1, GY - 1))])
    vv = {
        (cx, cy): mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(cx, cy))
        for cx in range(GX)
        for cy in range(GY)
    }
    n0s = [sum(pns[:i]) for i in range(GX)]
    groups = {}
    for cx, pn in enumerate(pns):
        groups.setdefault(pn, []).append(cx)
    st = ttnn.KernelDescriptor.SourceType
    state = {}
    layout = dict(pns=pns, NP=NP)

    def build(x, w, y):
        program = ttnn.ProgramDescriptor()
        fmt = lambda i, dt, page: [ttnn.CBFormatDescriptor(i, dt, page)]  # noqa: E731
        program.cbs = [  # (the multicast targets first, same sizes everywhere: their addresses match across the grid)
            ttnn.CBDescriptor(total_size=Kt * 2048, core_ranges=allc, format_descriptors=fmt(0, ttnn.bfloat16, 2048)),
            ttnn.CBDescriptor(total_size=nb * wb, core_ranges=allc, format_descriptors=fmt(1, w_dtype, wtb)),
            ttnn.CBDescriptor(
                total_size=NP * pnpx * 2048, core_ranges=allc, format_descriptors=fmt(4, ttnn.bfloat16, 2048)
            ),
            ttnn.CBDescriptor(total_size=xpt * 2048, core_ranges=allc, format_descriptors=fmt(8, ttnn.bfloat16, 2048)),
            ttnn.CBDescriptor(total_size=3 * wb, core_ranges=allc, format_descriptors=fmt(9, w_dtype, wtb)),
            ttnn.CBDescriptor(total_size=128, core_ranges=allc, format_descriptors=fmt(10, ttnn.uint32, 128)),
        ]
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=allc, initial_value=0) for i in range(1 + 2 * nb)
        ]
        x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
        w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
        y_ct = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
        dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
            processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
        )  # noqa: E731
        probe = [("PROBE", os.environ.get("SL_PROBE", "0"))]
        kernels = []
        for pn, cols in groups.items():
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, 0), ttnn.CoreCoord(cx, GY - 1)) for cx in cols])
            pnp = pn // NP
            sd, rv, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for cx in cols:
                for cy in range(GY):
                    v = vv[(cx, cy)]
                    sd[cx][cy] = [
                        int(w.buffer_address()),
                        cx,
                        cy,
                        v.x,
                        vv[(cx, 0)].y,
                        vv[(cx, GY - 1)].y,
                        int(x.buffer_address()),
                        vv[(0, cy)].x,
                        v.y,
                        vv[(GX - 1, cy)].x,
                        xoff[cx],
                        xcnt[cx],
                    ] + [vv[(i, cy)].x for i in range(GX)]
                    rv[cx][cy] = [
                        int(y.buffer_address()),
                        cy,
                        n0s[cx],
                        v.x,
                        int(x.buffer_address()),
                        cx,
                        0,
                        0,
                        0,
                        0,
                    ] + [vv[(cx, i)].y for i in range(GY)]
                    cp[cx][cy] = []
            comp = ttnn.ComputeConfigDescriptor()
            comp.math_fidelity = fid
            comp.fp32_dest_acc_en = True
            comp.dst_full_sync_en = full
            comp.math_approx_mode = False
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=_SEND,
                    source_type=st.SOURCE_CODE,
                    core_ranges=crs,
                    defines=probe,
                    compile_time_args=[Kt, kb, pnpx, nblk, NP, nb, GY, wtb, 1, GX] + w_ct + x_ct,
                    runtime_args=sd,
                    config=dm("RISCV_0", ttnn.NOC.NOC_0),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_RECV,
                    source_type=st.SOURCE_CODE,
                    core_ranges=crs,
                    defines=probe,
                    compile_time_args=[kb, pnp, pnpx, nblk, NP, nb, GX, GY, Nt, Kt, 1] + y_ct + x_ct,
                    runtime_args=rv,
                    config=dm("RISCV_1", ttnn.NOC.NOC_1),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_COMPUTE,
                    source_type=st.SOURCE_CODE,
                    core_ranges=crs,
                    defines=probe,
                    compile_time_args=[kb, pnp, pnpx, nblk, NP],
                    runtime_args=cp,
                    config=comp,
                ),
            ]
        program.kernels = kernels
        R, C = tuple(mesh_device.shape)
        mesh_desc = ttnn.MeshProgramDescriptor()
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = program
        return mesh_desc

    def run(x, w, out_memory_config=None):
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([M, N]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh_device,
            out_memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        k = tuple(int(t.buffer_address()) for t in (x, w, y))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(x, w, y)
        ttnn.generic_op([x, w, y], state["desc"])
        return y

    return run, layout


def upload_stacked(mesh_device, w, layout, dtype=ttnn.bfloat8_b):
    """W [K, N] torch -> the stacked device weight (replicated over the mesh)."""
    return ttnn.from_torch(
        stack_weight(w, layout["pns"], layout["NP"]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
