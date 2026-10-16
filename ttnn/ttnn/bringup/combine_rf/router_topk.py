# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""router_topk — Kimi K3's router selection as one generic_op per chip, writing the joint routing rows directly:
from the scores s = sigmoid(logits) and the biased scores b = s + e_score_correction_bias ([S, E] fp32 TILE, from
ttnn ops), each token's top K by b (the bias only selects), weights s / (sum of the picked s + 1e-20) * route_scale,
out as one row-major [S, 2 K] uint16 row per token: the K expert ids, then the K weights' bf16 bits (round to nearest
even) -- combine_plan's joint rows (row_pages + joint) and the fabric all-gather's input, with no layout / concat ops.

Cores: per row tile tr (32 tokens) SPLIT cores, each SPT = 32 / SPLIT tokens; each reads its row tile's E / 32 tiles of
both inputs (fp32 tiles: four 16 x 16 faces) and scans its tokens' E values keeping the top K (replace-the-minimum).
Order of the K picks: as found (the planner and the combine do not care).
"""

import ttnn

_TOPK = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t K = get_compile_time_arg_val(0), ET = get_compile_time_arg_val(1), SPT = get_compile_time_arg_val(2);
constexpr auto s_args = TensorAccessorArgs<3>();
constexpr auto b_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
constexpr auto o_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
constexpr uint32_t TB = 4096;  // fp32 tile bytes

FORCE_INLINE float f32(uint32_t u) {
    union { uint32_t u; float f; } v{u};
    return v.f;
}
FORCE_INLINE uint32_t u32(float f) {
    union { float f; uint32_t u; } v{f};
    return v.u;
}
FORCE_INLINE uint32_t key(uint32_t u) { return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }  // fp32 order -> uint order
FORCE_INLINE uint32_t bf16_rne(float f) {
    const uint32_t u = u32(f);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

void kernel_main() {
    const uint32_t s_addr = get_arg_val<uint32_t>(0), b_addr = get_arg_val<uint32_t>(1), o_addr = get_arg_val<uint32_t>(2);
    const uint32_t tr = get_arg_val<uint32_t>(3), r0 = get_arg_val<uint32_t>(4);  // row tile, first token row in it
    const uint32_t scale_bits = get_arg_val<uint32_t>(5);
    const uint32_t l1_off = get_arg_val<uint32_t>(6), ntok = get_arg_val<uint32_t>(7);  // (two RISCs: their halves)
    const auto sg = TensorAccessor(s_args, s_addr, TB);
    const auto bg = TensorAccessor(b_args, b_addr, TB);
    const auto og = TensorAccessor(o_args, o_addr, 32 * 4 * K);  // (a page: one row tile's 32 tokens' rows)
    // only this RISC's tokens' rows: per token, per tile, its two 16-value face rows (64 B each) of both inputs
    constexpr uint32_t RB = ET * 2 * 64;  // one token's row of one input
    const uint32_t l1 = ((get_write_ptr(0) + 63) & ~63u) + l1_off;
    const uint32_t s_l1 = l1, b_l1 = l1 + ntok * RB, o_l1 = b_l1 + ntok * RB;
    for (uint32_t i = 0; i < ntok; ++i) {
        const uint32_t r = r0 + i, off = ((r >> 4) * 2 * 256 + (r & 15) * 16) * 4;
        for (uint32_t t = 0; t < ET; ++t) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_read(sg.get_noc_addr(tr * ET + t) + off + h * 1024, s_l1 + i * RB + (t * 2 + h) * 64, 64);
                noc_async_read(bg.get_noc_addr(tr * ET + t) + off + h * 1024, b_l1 + i * RB + (t * 2 + h) * 64, 64);
            }
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const float scale = f32(scale_bits);
    for (uint32_t i = 0; i < ntok; ++i) {
        const uint32_t r = r0 + i;
        uint32_t tk[K], ti[K];
        uint32_t n = 0, mpos = 0, mkey = 0;
        for (uint32_t t = 0; t < ET * (__PROBE__ != 2); ++t) {
            for (uint32_t h = 0; h < 2; ++h) {  // (the row's two faces in tile t: columns t * 32 + h * 16 ..)
                const uint32_t* bv = reinterpret_cast<const uint32_t*>(b_l1 + i * RB + (t * 2 + h) * 64);
                for (uint32_t c = 0; c < 16; ++c) {
                    const uint32_t kk = key(bv[c]);
                    if (n == K && kk <= mkey) continue;  // (the common case: one compare)
                    const uint32_t e = t * 32 + h * 16 + c;
                    if (n < K) {
                        tk[n] = kk;
                        ti[n] = e;
                        if (n == 0 || kk < mkey) mkey = kk, mpos = n;
                        ++n;
                    } else if (kk > mkey) {
                        tk[mpos] = kk;
                        ti[mpos] = e;
                        mkey = tk[0], mpos = 0;
                        for (uint32_t j = 1; j < K; ++j) {
                            if (tk[j] < mkey) mkey = tk[j], mpos = j;
                        }
                    }
                }
            }
        }
        float w[K], sum = 0.0f;
        if constexpr (__PROBE__ == 1) {  // (probe: no weight math)
            volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(o_l1 + i * 64);
            for (uint32_t j = 0; j < K; ++j) o[j] = static_cast<uint16_t>(ti[j]), o[K + j] = 0;
            noc_async_write(o_l1 + i * 64, og.get_noc_addr(tr) + r * 4 * K, 4 * K);
            continue;
        }
        for (uint32_t j = 0; j < K; ++j) {
            const uint32_t e = ti[j], t = e >> 5, cc = e & 31;
            const uint32_t* sv = reinterpret_cast<const uint32_t*>(s_l1 + i * RB + (t * 2 + (cc >> 4)) * 64 + (cc & 15) * 4);
            w[j] = f32(sv[0]);
            sum += w[j];
        }
        const float inv = scale / (sum + 1e-20f);
        volatile tt_l1_ptr uint16_t* o = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(o_l1 + i * 64);
        for (uint32_t j = 0; j < K; ++j) {
            o[j] = static_cast<uint16_t>(ti[j]);
            o[K + j] = static_cast<uint16_t>(bf16_rne(w[j] * inv));
        }
        noc_async_write(o_l1 + i * 64, og.get_noc_addr(tr) + r * 4 * K, 4 * K);
    }
    noc_async_write_barrier();
}
"""


def router_topk_prepare(mesh_device, *, S, E, K, route_scale=1.0, split=8):
    """run(s, b) -> [S / 32, 32 * 2 K] uint16 row-major: 2 KB pages of 32 tokens' joint rows (s / b: [S, E] fp32
    TILE, DRAM interleaved); combine_plan tokens_per_page=32 reads it (and its fabric all-gather)."""
    import struct

    assert S % 32 == 0 and E % 32 == 0 and 32 % split == 0 and 4 * K == 64
    ET, nt, spt = E // 32, S // 32, 32 // split
    grid = mesh_device.compute_with_storage_grid_size()
    assert nt * split <= grid.x * grid.y
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(nt * split)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    assert spt <= 16 and spt % 2 == 0
    per_risc = (spt // 2) * (2 * ET * 128 + 64)  # (each RISC: its tokens' row pieces and output rows)
    l1 = 2 * per_risc + 128
    scale_bits = struct.unpack("<I", struct.pack("<f", route_scale))[0]
    st = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    dm = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default)
    state = {}

    def build(s, b, out):
        rt0, rt1 = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            base = [int(s.buffer_address()), int(b.buffer_address()), int(out.buffer_address()), i // split]
            rt0[c.x][c.y] = base + [(i % split) * spt, scale_bits, 0, spt // 2]
            rt1[c.x][c.y] = base + [(i % split) * spt + spt // 2, scale_bits, per_risc, spt // 2]
        ct = [K, ET, spt] + list(ttnn.TensorAccessorArgs(s).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(b).get_compile_time_args()) + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        program = ttnn.ProgramDescriptor()
        program.cbs = [ttnn.CBDescriptor(total_size=l1, core_ranges=crs,
                                         format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, l1)])]
        import os

        src = _TOPK.replace("__PROBE__", {"noweights": "1", "noscan": "2"}.get(os.environ.get("TOPK_PROBE", ""), "0"))
        dm1 = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default)
        program.kernels = [
            ttnn.KernelDescriptor(kernel_source=src, source_type=st, core_ranges=crs, compile_time_args=ct, runtime_args=rt0,
                                  config=dm),
            ttnn.KernelDescriptor(kernel_source=src, source_type=st, core_ranges=crs, compile_time_args=ct, runtime_args=rt1,
                                  config=dm1),
        ]
        R, C = tuple(mesh_device.shape)
        mesh_desc = ttnn.MeshProgramDescriptor()
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = program
        return mesh_desc

    def run(s, b):
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([S // 32, 32 * 2 * K]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, mesh_device,
                                             ttnn.DRAM_MEMORY_CONFIG)
        k = tuple(int(t.buffer_address()) for t in (s, b, out))
        if state.get("key") != k:
            state["key"], state["desc"] = k, build(s, b, out)
        ttnn.generic_op([s, b, out], state["desc"])
        return out

    return run
