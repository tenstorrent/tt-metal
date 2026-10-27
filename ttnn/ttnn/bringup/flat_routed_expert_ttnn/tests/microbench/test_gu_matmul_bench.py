# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One-core microbenchmark of the flat expert's gate/up K loop (se3_compute): x bfp8 [RT x K] tiles and gate/up weights
bfp4 [K x CT] tiles resident in L1, LoFi, NREP passes of one RT x CT output block over all of K, timed on the device's
wall clock (UNPACK thread after the inputs are resident -> PACK thread after the last pack). Reports cycles per
32x32x32 tile product against the LoFi peak (4096 FLOP / cycle / core: 16 cycles a tile product).

Variants (BenchCase): the DST shape (RT x CT) and mode (fp32 / bf16 DEST, half-sync), the activation on the PACK
thread's SFPU (silu(gate) * up over (gate, up) column pairs, as se3_compute's act_dst), and the K stream: resident
(one wait for all of K) or streamed (CB wait / pop per K-block of KB tiles, the reader re-pushing the resident blocks:
the flat op's per-block handshakes without data movement). Every case checks its last pass against torch (PCC)."""

from dataclasses import dataclass

import pytest
import torch

import ttnn

PEAK_CYC = 16  # 2 * 32^3 FLOP / 4096 FLOP per cycle (LoFi, one core)

_READ = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t RT = get_compile_time_arg_val(0), CT = get_compile_time_arg_val(1), KB = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NREP = get_compile_time_arg_val(4), STREAM = get_compile_time_arg_val(5);
constexpr auto x_args = TensorAccessorArgs<6>();
constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr uint32_t K = KB * NBLK, XB = RT * KB, WB = KB * CT;
void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1);
    const uint32_t xs = get_tile_size(0), ws = get_tile_size(1);
    const auto x = TensorAccessor(x_args, x_addr, xs);
    const auto w = TensorAccessor(w_args, w_addr, ws);
    // block j of x: [RT x KB] row-major (tile (r, k) of x at page r * K + j * KB + k); of w: [KB x CT] (natural order)
    auto load = [&](uint32_t j) {
        uint32_t l1 = get_write_ptr(0);
        for (uint32_t r = 0; r < RT; ++r)
            for (uint32_t k = 0; k < KB; ++k, l1 += xs) noc_async_read(x.get_noc_addr(r * K + j * KB + k), l1, xs);
        l1 = get_write_ptr(1);
        for (uint32_t t = 0; t < WB; ++t, l1 += ws) noc_async_read(w.get_noc_addr(j * WB + t), l1, ws);
        noc_async_read_barrier();
    };
    if constexpr (!STREAM) {
        for (uint32_t j = 0; j < NBLK; ++j) {
            cb_reserve_back(0, XB);
            cb_reserve_back(1, WB);
            load(j);
            cb_push_back(0, XB);
            cb_push_back(1, WB);
        }
        return;
    }
    // streamed: the CBs hold all of K (capacity NBLK blocks), so block j sits at the same L1 every pass: pass 0 loads
    // it, later passes only reserve + push (the handshakes of a streamed K without the data movement)
    for (uint32_t rep = 0; rep < NREP; ++rep) {
        for (uint32_t j = 0; j < NBLK; ++j) {
            cb_reserve_back(0, XB);
            cb_reserve_back(1, WB);
            if (rep == 0) load(j);
            cb_push_back(0, XB);
            cb_push_back(1, WB);
        }
    }
}
"""

_WRITE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t NOUT = get_compile_time_arg_val(0), NREP = get_compile_time_arg_val(1);
constexpr auto o_args = TensorAccessorArgs<2>();
void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const auto o = TensorAccessor(o_args, o_addr, 2048);
    for (uint32_t rep = 0; rep < NREP; ++rep) {
        cb_wait_front(4, NOUT);
        if (rep == NREP - 1) {  // (the last pass, for the PCC check)
            const uint32_t l1 = get_read_ptr(4);
            for (uint32_t t = 0; t < NOUT; ++t) noc_async_write(l1 + t * 2048, o.get_noc_addr(t), 2048);
            noc_async_write_barrier();
        }
        cb_pop_front(4, NOUT);
    }
}
"""

_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#if defined(GB_ACT) && defined(TRISC_PACK)
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
constexpr uint32_t RT = get_compile_time_arg_val(0), CT = get_compile_time_arg_val(1), KB = get_compile_time_arg_val(2);
constexpr uint32_t NBLK = get_compile_time_arg_val(3), NREP = get_compile_time_arg_val(4), STREAM = get_compile_time_arg_val(5);
constexpr uint32_t XB = RT * KB, WB = KB * CT, NT = RT * CT;
#ifdef GB_ACT
constexpr uint32_t NOUT = RT * (CT / 2 + CT % 2);  // h of each (gate, up) pair; an odd last column packed raw
#else
constexpr uint32_t NOUT = NT;
#endif
void kernel_main() {
    const uint32_t ts = get_arg_val<uint32_t>(0);  // L1 words: start (UNPACK), end (PACK)
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 4);  // (matmul: in1 -> SrcA, in0 -> SrcB)
    matmul_block_init(0, 1, 0, CT, RT, KB);
#ifdef GB_ACT
    silu_tile_init_pack();
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
#endif
    if constexpr (STREAM) {
        cb_wait_front(0, XB);
        cb_wait_front(1, WB);
    } else {
        cb_wait_front(0, XB * NBLK);
        cb_wait_front(1, WB * NBLK);
    }
    UNPACK((reinterpret_cast<volatile uint32_t*>(ts)[0] = static_cast<uint32_t>(ckernel::read_wall_clock())));
    for (uint32_t rep = 0; rep < NREP; ++rep) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < NBLK; ++j) {
            uint32_t xo = j * XB, wo = j * WB;
            if constexpr (STREAM) {
                cb_wait_front(0, XB);
                cb_wait_front(1, WB);
                xo = 0;
                wo = 0;
            }
            for (uint32_t k = 0; k < KB; ++k) matmul_block(0, 1, xo + k, wo + k * CT, 0, false, CT, RT, KB);
            if constexpr (STREAM) {
                cb_pop_front(0, XB);
                cb_pop_front(1, WB);
            }
        }
        tile_regs_commit();
        cb_reserve_back(4, NOUT);
        tile_regs_wait();
#ifdef GB_ACT
        PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
        for (uint32_t r = 0; r < RT; ++r)
            for (uint32_t c = 0; c + 1 < CT; c += 2) silu_tile_pack(r * CT + c);
        for (uint32_t r = 0; r < RT; ++r)
            for (uint32_t c = 0; c + 1 < CT; c += 2) {
                const uint32_t t = r * CT + c;
                PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
                                       (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE), t, t + 1, t, VectorMode::RC)));
            }
        PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t c = 0; c + 1 < CT; c += 2) pack_tile(r * CT + c, 4);
            if constexpr (CT % 2) pack_tile(r * CT + CT - 1, 4);
        }
#else
        for (uint32_t t = 0; t < NT; ++t) pack_tile(t, 4);
#endif
        tile_regs_release();
        cb_push_back(4, NOUT);
    }
    PACK((reinterpret_cast<volatile uint32_t*>(ts)[1] = static_cast<uint32_t>(ckernel::read_wall_clock())));
}
"""


@dataclass(frozen=True)
class BenchCase:
    name: str
    rt: int  # output row tiles a pass
    ct: int  # output column tiles a pass (gate / up columns)
    fp32: bool  # fp32 DEST (half-sync: 4 tiles) else bf16 (8 tiles)
    act: bool = False  # silu(gate) * up on the PACK thread's SFPU
    stream: bool = False  # CB handshakes per K-block
    k: int = 192  # K tiles (GLM-5.3 H 6144)
    kb: int = 8  # K-block tiles (the flat op's kblk)
    rep: int = 32  # passes


CASES = (
    [
        BenchCase(f"{rt}x{ct} {'fp32' if f else 'bf16'}{' act' if a else ''}{' stream' if s else ''}", rt, ct, f, a, s)
        for rt, ct, f in [(2, 2, True), (1, 3, True), (4, 2, False)]
        for a in (False, True)
        for s in (False, True)
    ]
    + [  # K-block size under the streamed handshakes (the flat op's kblk is 8)
        BenchCase(f"{rt}x{ct} fp32 stream kb{kb}", rt, ct, True, False, True, kb=kb)
        for rt, ct in [(2, 2), (1, 3)]
        for kb in (4, 16, 24)
    ]
    + [  # shape sensitivity, resident: which operand streams per tile product (in1 when ct >= rt, else in0)
        BenchCase(f"{rt}x{ct} fp32 shape", rt, ct, True) for rt, ct in [(1, 1), (1, 4), (4, 1)]
    ]
)


def _program(c: BenchCase, x, w, out, ts):
    core = ttnn.CoreCoord(0, 0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    nblk = c.k // c.kb
    xb, wb = c.rt * c.kb, c.kb * c.ct
    nout = c.rt * (c.ct // 2 + c.ct % 2) if c.act else c.rt * c.ct
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    ra, wa, ca = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    ra[0][0] = [int(x.buffer_address()), int(w.buffer_address())]
    wa[0][0] = [int(out.buffer_address())]
    ca[0][0] = [int(ts.buffer_address())]
    cta = [c.rt, c.ct, c.kb, nblk, c.rep, int(c.stream)]
    program = ttnn.ProgramDescriptor()
    program.cbs = [
        ttnn.CBDescriptor(
            total_size=xb * nblk * 1088,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat8_b, 1088)],
        ),
        ttnn.CBDescriptor(
            total_size=wb * nblk * 576,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.bfloat4_b, 576)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * nout * 2048,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(4, ttnn.bfloat16, 2048)],
        ),
    ]
    program.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READ,
            source_type=st,
            core_ranges=crs,
            compile_time_args=cta
            + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(w).get_compile_time_args()),
            runtime_args=ra,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITE,
            source_type=st,
            core_ranges=crs,
            compile_time_args=[nout, c.rep] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wa,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE,
            source_type=st,
            core_ranges=crs,
            compile_time_args=cta,
            defines=[("GB_ACT", "1")] if c.act else [],
            runtime_args=ca,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=c.fp32, dst_full_sync_en=False
            ),
        ),
    ]
    return program, nout


def _reference(c: BenchCase, xq, wq):
    """The last pass's packed tiles, in pack order (row r: h of each pair, then an odd last column raw)."""
    acc = xq[: c.rt * 32, : c.k * 32].float() @ wq[: c.k * 32, : c.ct * 32].float()
    tiles = []
    for r in range(c.rt):
        t = lambda col: acc[r * 32 : (r + 1) * 32, col * 32 : (col + 1) * 32]  # noqa: E731
        if c.act:
            for p in range(0, c.ct - 1, 2):
                tiles.append(torch.nn.functional.silu(t(p)) * t(p + 1))
            if c.ct % 2:
                tiles.append(t(c.ct - 1))
        else:
            tiles.extend(t(col) for col in range(c.ct))
    return tiles


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_gu_matmul_bench(device, case):
    c = case
    torch.manual_seed(0)
    dram = ttnn.DRAM_MEMORY_CONFIG
    x_host = torch.randn(c.rt * 32, c.k * 32) * 0.5
    w_host = torch.randn(c.k * 32, c.ct * 32) * 0.05
    x = ttnn.from_torch(x_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    w = ttnn.from_torch(w_host, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    xq, wq = ttnn.to_torch(x), ttnn.to_torch(w)  # (the quantized values the device multiplies)
    nout = c.rt * (c.ct // 2 + c.ct % 2) if c.act else c.rt * c.ct
    out = ttnn.from_torch(
        torch.zeros(32, nout * 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
    )
    core = ttnn.CoreCoord(0, 0)
    ts_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    ts = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ts_mc,
    )
    program, _ = _program(c, x, w, out, ts)
    if hasattr(device, "shape"):
        md = ttnn.MeshProgramDescriptor()
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))] = program
        run = lambda: ttnn.generic_op([x, w, out, ts], md)  # noqa: E731
    else:
        run = lambda: ttnn.generic_op([x, w, out, ts], program)  # noqa: E731
    cyc = []
    for _ in range(4):  # (first run compiles; the median of the rest)
        run()
        ttnn.synchronize_device(device)
        t = ttnn.to_torch(ts).flatten()
        cyc.append((int(t[1]) - int(t[0])) & 0xFFFFFFFF)
    cyc = sorted(cyc[1:])[1]
    prods = c.rep * c.rt * c.ct * c.k
    per = cyc / prods
    got = ttnn.to_torch(out).float()
    ref = _reference(c, xq, wq)
    pccs = [_pcc(got[:, i * 32 : (i + 1) * 32], r) for i, r in enumerate(ref)]
    print(
        f"[gubench] | {c.name} | cycles {cyc} | tile products {prods} | {per:.2f} cyc/product | "
        f"{PEAK_CYC / per:.1%} of LoFi | {cyc / 1.35e3:.1f} us @1.35 GHz | min PCC {min(pccs):.5f} |",
        flush=True,
    )
    assert min(pccs) > (0.99 if c.fp32 else 0.97), pccs
