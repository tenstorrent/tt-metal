# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""flat_gather - lay a FlatRoutedExpert's weights out ON THE DEVICE from expert matrices already in DRAM as bfp tiles
(e.g. TtMoe's routed-expert ttnn cache): no host dequantize / pack / requantize (~4 min a K3 layer of 896 experts).

  flat_face_map(H, I, E, lay)  host, once per layout: for every page (tile) of each laid-out tensor (w_gu, w_d, w_rd -
      _bank_sharded's width-sharded DRAM tensors, page = row-major tile index t * banks + b) and each of its 4 faces
      (16 x 16), the source face it holds: ((s * NT + tile) * 4 + face), or -1 (padding: zeros). Sources s = e * 3 + m
      (local expert e; m 0 Wg [H, I], 1 Wu [H, I], 2 Wd [I, H]), tile = the source's row-major tile index, NT its tiles.
      Derived by running the flat op's own flat_weight_regions and _bank_sharded's packing on index tensors (each
      element 1 + its (source, tile, face, position)), every face checked to be one source face, in order - so any
      layout (whole tiles, 16-column units, half-width pairs) maps, face by face.
  gather(mesh, srcs, face_map, dst)  device: one data-movement program a chip (the same everywhere: lockstep L1 / DRAM
      addresses) - each dest page: the source tiles its faces come from read into L1, the faces copied (a bfp face:
      its 16 shared exponent bytes - the tile's first 64 B hold every face's - and its mantissa bytes, contiguous a
      face after them), the page written. srcs: the NS source tensors in s order, DRAM interleaved, the dst's dtype.
"""

from __future__ import annotations

import torch

import ttnn

TPP = 64  # (dest pages a face-map table page holds: 64 x 4 words = 1 KB)

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t NS = get_compile_time_arg_val(0);   // source tensors
constexpr uint32_t NT = get_compile_time_arg_val(1);   // tiles a source
constexpr uint32_t TB = get_compile_time_arg_val(2);   // bytes a tile (page)
constexpr uint32_t EB = get_compile_time_arg_val(3);   // shared exponent bytes a face (the tile's first 4 EB)
constexpr uint32_t MB = get_compile_time_arg_val(4);   // mantissa bytes a face (after the exponents, face-major)
constexpr uint32_t TP = get_compile_time_arg_val(5);   // dest pages a table page
constexpr auto src_args = TensorAccessorArgs<6>();
constexpr auto tab_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
constexpr auto dst_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();

FORCE_INLINE void copy_bytes(uint32_t dst, uint32_t src, uint32_t n) {  // (L1 to L1, word-aligned, n % 4 == 0)
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < n / 4; ++i) d[i] = s[i];
}
FORCE_INLINE void zero_bytes(uint32_t dst, uint32_t n) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < n / 4; ++i) d[i] = 0;
}

void kernel_main() {
    const uint32_t tab_addr = get_arg_val<uint32_t>(0), dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t q0 = get_arg_val<uint32_t>(2), nq = get_arg_val<uint32_t>(3), np_ = get_arg_val<uint32_t>(4);
    const auto tab = TensorAccessor(tab_args, tab_addr, TP * 16);
    const auto dst = TensorAccessor(dst_args, dst_addr, TB);
    // L1: the table page | 4 source tile slots | 2 dest tiles
    const uint32_t base = get_write_ptr(0);
    const uint32_t tl1 = base, sl1 = base + TP * 16, dl1 = sl1 + 4 * TB;
    uint32_t wbuf = 0;
    for (uint32_t q = q0; q < q0 + nq; ++q) {  // (table page q: dest pages q TP ..)
        noc_async_read(tab.get_noc_addr(q), tl1, TP * 16);
        noc_async_read_barrier();
        volatile tt_l1_ptr int32_t* e = reinterpret_cast<volatile tt_l1_ptr int32_t*>(tl1);
        for (uint32_t i = 0; i < TP; ++i) {
            const uint32_t p = q * TP + i;
            if (p >= np_) break;
            int32_t tile_of[4];  // (the slot each face's source tile went to: distinct tiles read once)
            int32_t slot_tile[4] = {-1, -1, -1, -1};
            uint32_t ns = 0;
            for (uint32_t f = 0; f < 4; ++f) {
                const int32_t c = e[i * 4 + f];
                tile_of[f] = -1;
                if (c < 0) continue;
                const int32_t g = c >> 2;  // (s NT + tile)
                uint32_t k = 0;
                while (k < ns && slot_tile[k] != g) ++k;
                if (k == ns) {
                    slot_tile[ns++] = g;
                    const uint32_t s = static_cast<uint32_t>(g) / NT, t = static_cast<uint32_t>(g) % NT;
                    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(s), TB);
                    noc_async_read(src.get_noc_addr(t), sl1 + k * TB, TB);
                }
                tile_of[f] = static_cast<int32_t>(k);
            }
            const uint32_t d = dl1 + wbuf * TB;
            noc_async_write_barrier();  // (this dest buffer's previous write out)
            noc_async_read_barrier();
            for (uint32_t f = 0; f < 4; ++f) {
                const int32_t c = e[i * 4 + f];
                if (c < 0) {
                    zero_bytes(d + f * EB, EB);
                    zero_bytes(d + 4 * EB + f * MB, MB);
                    continue;
                }
                const uint32_t sf = static_cast<uint32_t>(c) & 3, s0 = sl1 + static_cast<uint32_t>(tile_of[f]) * TB;
                copy_bytes(d + f * EB, s0 + sf * EB, EB);
                copy_bytes(d + 4 * EB + f * MB, s0 + 4 * EB + sf * MB, MB);
            }
            noc_async_write(d, dst.get_noc_addr(p), TB);
            wbuf ^= 1;
        }
    }
    noc_async_write_barrier();
}
"""

# (bytes a bfp tile: 4 faces' shared exponents, then each face's mantissas)
_BFP = {ttnn.bfloat4_b: (576, 16, 128), ttnn.bfloat8_b: (1088, 16, 256)}


def _ids(s: int, NT: int, R: int, C: int) -> torch.Tensor:
    """Source s's [R, C] as index values: 1 + ((s NT + tile) 4 + face) 256 + position in the face (int64)."""
    r = torch.arange(R).view(-1, 1)
    c = torch.arange(C).view(1, -1)
    tile = (r // 32) * (C // 32) + (c // 32)
    face = ((r % 32) // 16) * 2 + (c % 32) // 16
    pos = (r % 16) * 16 + (c % 16)
    return 1 + ((s * NT + tile) * 4 + face) * 256 + pos


def _packed(regions: list, banks: int) -> torch.Tensor:
    """_bank_sharded's host assembly of one device's regions -> [tiles, banks, 32, 32] (page t banks + b)."""
    per = -(-len(regions) // banks)
    regions = list(regions) + [torch.zeros_like(regions[0])] * (per * banks - len(regions))
    host = torch.cat(
        [torch.cat([regions[b + h * banks] for h in range(per)]).reshape(-1, 32) for b in range(banks)], dim=1
    )
    T = host.shape[0] // 32
    return host.view(T, 32, banks, 32).permute(0, 2, 1, 3)


def flat_face_map(H: int, I: int, E: int, lay: dict) -> dict:
    """{"w_gu" / "w_d" / "w_rd": int32 [pages, 4]} - each dest page's 4 faces' sources (module docstring)."""
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import flat_weight_regions

    assert (H // 32) * (I // 32) == (I // 32) * (H // 32)
    NT = (H // 32) * (I // 32)
    assert (E * 3 * NT * 4) * 256 + 256 < 2**31, "index values past int32"
    W_l = [
        (
            _ids(3 * e, NT, H, I).to(torch.int32),
            _ids(3 * e + 1, NT, H, I).to(torch.int32),
            _ids(3 * e + 2, NT, I, H).to(torch.int32),
        )
        for e in range(E)
    ]
    regions, dregs, rregs = flat_weight_regions(W_l, lay)
    del W_l
    out = {}
    pos = torch.arange(256, dtype=torch.int64)
    for name, regs in (("w_gu", regions), ("w_d", dregs), ("w_rd", rregs)):
        if regs is None:
            continue
        tiles = _packed(regs, lay["banks"])  # [T, banks, 32, 32]
        T, B = tiles.shape[0], tiles.shape[1]
        faces = (
            tiles.reshape(T, B, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5).reshape(T * B, 4, 256).to(torch.int64)
        )  # (page, face, position)
        first = faces[:, :, 0]
        pad = first == 0
        assert bool((faces[pad] == 0).all()), f"{name}: a face partly padding"
        want = first.unsqueeze(-1) + pos
        ok = (faces == want) | pad.unsqueeze(-1)
        assert bool(ok.all()), f"{name}: a face is not one source face in order (a transposed or split face)"
        code = torch.where(pad, torch.full_like(first, -1), (first - 1) // 256)
        out[name] = code.to(torch.int32)
    return out


def gather(mesh, srcs: list, face_map: torch.Tensor, dst, num_cores: int | None = None):
    """dst (a laid-out FlatRoutedExpert tensor) <- its faces from srcs (module docstring); face_map: int32 [pages, 4]."""
    tb, eb, mb = _BFP[dst.dtype]
    assert all(s.dtype == dst.dtype for s in srcs), "sources and dest: one bfp dtype"
    shape = tuple(srcs[0].shape)
    NT = (shape[-2] // 32) * (shape[-1] // 32)
    pages = face_map.shape[0]
    nq = -(-pages // TPP)
    tab = torch.full((nq * TPP, 4), -1, dtype=torch.int32)
    tab[:pages] = face_map
    tab_t = ttnn.from_torch(
        tab.view(nq, TPP * 4).view(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    grid = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][: num_cores or grid.x * grid.y]
    per = -(-nq // len(cores))
    used = [(c, i * per, min(per, nq - i * per)) for i, c in enumerate(cores) if i * per < nq]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in used])
    rt = ttnn.RuntimeArgs()
    for c, q0, n in used:
        rt[c.x][c.y] = [int(tab_t.buffer_address()), int(dst.buffer_address()), q0, n, pages]
    ct = (
        [len(srcs), NT, tb, eb, mb, TPP]
        + list(ttnn.TensorAccessorArgs(srcs[0]).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
    )
    prog = ttnn.ProgramDescriptor()
    prog.cbs = [
        ttnn.CBDescriptor(
            total_size=TPP * 16 + 6 * tb,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, TPP * 16 + 6 * tb)],
        )
    ]
    prog.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            common_runtime_args=[int(s.buffer_address()) for s in srcs],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
    ]
    R, C = tuple(mesh.shape)
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
    ttnn.generic_op(list(srcs) + [tab_t, dst], md)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(tab_t)
