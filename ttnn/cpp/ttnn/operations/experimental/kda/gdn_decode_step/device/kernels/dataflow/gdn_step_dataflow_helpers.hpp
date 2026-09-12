// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Dataflow helpers shared by the gdn_decode_step reader/writer kernels.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace gdn_step_df {

template <typename Accessor>
inline void read_tiles_at(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t dst_tile) {
    const uint32_t entry = dfb.get_entry_size();
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_read(acc, dfb, entry, {.page_id = first_page + t}, {.offset_bytes = (dst_tile + t) * entry});
    }
}

template <typename Accessor>
inline void read_tiles(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count) {
    dfb.reserve_back(count);
    read_tiles_at(acc, dfb, noc, first_page, count, 0);
    noc.async_read_barrier();
    dfb.push_back(count);
}

// Read the head's [q | k | v] tiles of a row tensor laid out [q(Nk*Dk) | k(Nk*Dk) | v(Nv*Dv) | ...] into one DFB.
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void read_head_row(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h) {
    dfb.reserve_back(2 * Kt + Vt);
    read_tiles_at(acc, dfb, noc, hk * Kt, Kt, 0);
    read_tiles_at(acc, dfb, noc, Nk * Kt + hk * Kt, Kt, Kt);
    read_tiles_at(acc, dfb, noc, 2 * Nk * Kt + h * Vt, Vt, 2 * Kt);
    noc.async_read_barrier();
    dfb.push_back(2 * Kt + Vt);
}

// Write the head's [q | k | v] tiles held in `dfb` back to a row tensor (q/k only when `write_qk`).
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void write_head_row(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, bool write_qk) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    dfb.wait_front(Ct);
    const uint32_t entry = dfb.get_entry_size();
    if (write_qk) {
        for (uint32_t t = 0; t < Kt; ++t) {
            noc.async_write(dfb, acc, entry, {.offset_bytes = t * entry}, {.page_id = hk * Kt + t});
            noc.async_write(dfb, acc, entry, {.offset_bytes = (Kt + t) * entry}, {.page_id = Nk * Kt + hk * Kt + t});
        }
    }
    for (uint32_t t = 0; t < Vt; ++t) {
        noc.async_write(dfb, acc, entry, {.offset_bytes = (2 * Kt + t) * entry}, {.page_id = 2 * Nk * Kt + h * Vt + t});
    }
    noc.async_write_barrier();
    dfb.pop_front(Ct);
}

FORCE_INLINE uint32_t tile_elem_index(uint32_t row, uint32_t col) {
    // 32x32 tile stored as four 16x16 faces: f0 (r<16,c<16), f1 (r<16,c>=16), f2, f3
    return ((row < 16 ? 0u : 2u) + (col < 16 ? 0u : 1u)) * 256u + (row & 15u) * 16u + (col & 15u);
}

// Broadcast one fp32 value over a whole fp32 tile in `dfb` (slot must be reserved; lock covers 1 entry).
inline void fill_scalar_tile(DataflowBuffer& dfb, uint32_t value) {
    auto lock = dfb.scoped_write_lock(1);
    auto p32 = lock.template get_ptr<volatile uint32_t>();
    for (uint32_t i = 0; i < 1024; ++i) {
        p32[i] = value;
    }
}

// Read tile `page` of a tensor and return element (row 0, col) as fp32 bits (bf16 source is widened).
template <bool src_fp32, typename Accessor>
FORCE_INLINE uint32_t
load_tile_scalar(const Accessor& acc, DataflowBuffer& staging, Noc& noc, uint32_t page, uint32_t col) {
    constexpr uint32_t src_bytes = src_fp32 ? 4096 : 2048;
    noc.async_read(acc, staging, src_bytes, {.page_id = page}, {.offset_bytes = 0});
    noc.async_read_barrier();
    auto lock = staging.scoped_write_lock(1);
    const uint32_t idx = tile_elem_index(0, col);
    if constexpr (src_fp32) {
        auto p = lock.template get_ptr<volatile uint32_t>();
        return p[idx];
    } else {
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        return static_cast<uint32_t>(p16[idx]) << 16;
    }
}

// Read tile 0 of a [.., H]-wide tensor and broadcast element (row 0, col) over an fp32 tile pushed into `dfb`.
template <bool src_fp32, typename Accessor>
inline void load_head_scalar(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t col) {
    dfb.reserve_back(1);
    const uint32_t value = load_tile_scalar<src_fp32>(acc, dfb, noc, 0, col);
    fill_scalar_tile(dfb, value);
    dfb.push_back(1);
}

// bf16 column mask: 1.0 in row 0 (column 0), 0 elsewhere.
inline void build_row0_mask(DataflowBuffer& dfb, Noc& noc) {
    dfb.reserve_back(1);
    noc.async_write_zeros(dfb, dfb.get_entry_size());
    noc.write_zeros_l1_barrier();
    {
        auto lock = dfb.scoped_write_lock(1);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        p16[0] = 0x3F80;
    }
    dfb.push_back(1);
}

// ---- row-0-only transfers: a 32x32 tile stores row 0 as two 32-byte face rows at byte offsets 0 and entry/4.
// Only the token row matters for the decode inputs, so moving 64 B instead of a whole tile cuts the per-core
// traffic ~30x. Tiles are zero-filled first so the untouched rows are finite (they are masked, not ignored).
template <typename Accessor>
inline void read_row0_at(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t dst_tile) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;  // bytes of one face row: 32 for bf16, 64 for fp32
    const uint32_t base = dst_tile * entry;
    noc.async_read(acc, dfb, seg, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = base});
    noc.async_read(acc, dfb, seg, {.page_id = page, .offset_bytes = entry / 4}, {.offset_bytes = base + entry / 4});
}

template <typename Accessor>
inline void write_row0_at(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t src_tile) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;
    const uint32_t base = src_tile * entry;
    noc.async_write(dfb, acc, seg, {.offset_bytes = base}, {.page_id = page, .offset_bytes = 0});
    noc.async_write(dfb, acc, seg, {.offset_bytes = base + entry / 4}, {.page_id = page, .offset_bytes = entry / 4});
}

inline void zero_reserved(DataflowBuffer& dfb, Noc& noc, uint32_t tiles) {
    noc.async_write_zeros(dfb, tiles * dfb.get_entry_size());
    noc.write_zeros_l1_barrier();
}

// Row 0 of `count` consecutive tiles (tiles zero-filled first when `zero`).
template <typename Accessor>
inline void read_tiles_r0(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, bool zero = true) {
    dfb.reserve_back(count);
    if (zero) {
        zero_reserved(dfb, noc, count);
    }
    for (uint32_t t = 0; t < count; ++t) {
        read_row0_at(acc, dfb, noc, first_page + t, t);
    }
    noc.async_read_barrier();
    dfb.push_back(count);
}

// Row 0 of the head's [q | k | v] tiles of a [q | k | v | ...] row tensor.
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void read_head_row_r0(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, bool zero = true) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    dfb.reserve_back(Ct);
    if (zero) {
        zero_reserved(dfb, noc, Ct);
    }
    for (uint32_t t = 0; t < Kt; ++t) {
        read_row0_at(acc, dfb, noc, hk * Kt + t, t);
        read_row0_at(acc, dfb, noc, Nk * Kt + hk * Kt + t, Kt + t);
    }
    for (uint32_t t = 0; t < Vt; ++t) {
        read_row0_at(acc, dfb, noc, 2 * Nk * Kt + h * Vt + t, 2 * Kt + t);
    }
    noc.async_read_barrier();
    dfb.push_back(Ct);
}

// Row 0 of the head's [q | k | v] tiles held in `dfb` back to a row tensor (q/k only when `write_qk`).
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void write_head_row_r0(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, bool write_qk) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    dfb.wait_front(Ct);
    if (write_qk) {
        for (uint32_t t = 0; t < Kt; ++t) {
            write_row0_at(acc, dfb, noc, hk * Kt + t, t);
            write_row0_at(acc, dfb, noc, Nk * Kt + hk * Kt + t, Kt + t);
        }
    }
    for (uint32_t t = 0; t < Vt; ++t) {
        write_row0_at(acc, dfb, noc, 2 * Nk * Kt + h * Vt + t, 2 * Kt + t);
    }
    noc.async_write_barrier();
    dfb.pop_front(Ct);
}

// Copy row 0 of source tile `page` into row `dst_row` of tile `dst_tile` in `dfb` (two face-row segments).
template <typename Accessor>
inline void pack_row_from(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t dst_tile, uint32_t dst_row) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;    // one face row: 32 B for bf16
    const uint32_t esz = entry / 1024;  // bytes per element
    const uint32_t base = dst_tile * entry;
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = 0},
        {.offset_bytes = base + tile_elem_index(dst_row, 0) * esz});
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = entry / 4},
        {.offset_bytes = base + tile_elem_index(dst_row, 16) * esz});
}

// Copy row `src_row` of source tile `page` into row `dst_row` of tile `dst_tile` (two face-row segments). Source and
// destination rows must have the same parity so both segments keep their 64 B alignment class.
template <typename Accessor>
inline void pack_row_from_row(
    const Accessor& acc,
    DataflowBuffer& dfb,
    Noc& noc,
    uint32_t page,
    uint32_t src_row,
    uint32_t dst_tile,
    uint32_t dst_row) {
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = entry / 64;    // one face row: 32 B for bf16
    const uint32_t esz = entry / 1024;  // bytes per element
    const uint32_t base = dst_tile * entry;
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = tile_elem_index(src_row, 0) * esz},
        {.offset_bytes = base + tile_elem_index(dst_row, 0) * esz});
    noc.async_read(
        acc,
        dfb,
        seg,
        {.page_id = page, .offset_bytes = tile_elem_index(src_row, 16) * esz},
        {.offset_bytes = base + tile_elem_index(dst_row, 16) * esz});
}

// Build the packed [q | k | v] tile of value head h for user row b into tile `dst_tile` (zeroed first): channel chunk c
// goes to row 2c + (b & 1) (same parity as the source row -> 64 B aligned segments).
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void pack_head_tile_user(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, uint32_t b, uint32_t dst_tile) {
    const uint32_t par = b & 1u;
    for (uint32_t c = 0; c < Kt; ++c) {
        pack_row_from_row(acc, dfb, noc, hk * Kt + c, b, dst_tile, 2 * c + par);
        pack_row_from_row(acc, dfb, noc, Nk * Kt + hk * Kt + c, b, dst_tile, 2 * (Kt + c) + par);
    }
    for (uint32_t c = 0; c < Vt; ++c) {
        pack_row_from_row(acc, dfb, noc, 2 * Nk * Kt + h * Vt + c, b, dst_tile, 2 * (2 * Kt + c) + par);
    }
}

// B=1 convenience (user row 0): chunk c in row 2c.
template <uint32_t Kt, uint32_t Vt, uint32_t Nk, typename Accessor>
inline void pack_head_tile(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t hk, uint32_t h, uint32_t dst_tile) {
    pack_head_tile_user<Kt, Vt, Nk>(acc, dfb, noc, hk, h, 0, dst_tile);
}

// Selector tiles for user row b: sel[c] has a single 1.0 at (row b, col 2c + parity(b)), so sel[c] @ P puts packed row
// 2c + parity(b) of P into row b. Also the row mask e_b (1.0 at (row b, col 0)).
inline void build_user_selectors(DataflowBuffer& sel, DataflowBuffer& mask, Noc& noc, uint32_t b, uint32_t Ct) {
    sel.reserve_back(Ct);
    zero_reserved(sel, noc, Ct);
    {
        auto lock = sel.scoped_write_lock(Ct);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t c = 0; c < Ct; ++c) {
            p16[c * 1024 + tile_elem_index(b, 2 * c + (b & 1u))] = 0x3F80;
        }
    }
    sel.push_back(Ct);
    mask.reserve_back(1);
    zero_reserved(mask, noc, 1);
    {
        auto lock = mask.scoped_write_lock(1);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        p16[tile_elem_index(b, 0)] = 0x3F80;
    }
    mask.push_back(1);
}

// Write rows [r0, r0 + nr) of `count` consecutive L1 tiles to the same rows of the destination tiles. Rows are written
// as whole face-row spans; r0 must be even and nr even (or r0 == 0, nr == 1 -> rows 0..1, row 1 being zero padding),
// so every DRAM destination address is 64 B aligned.
template <typename Accessor>
inline void write_rows(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t r0, uint32_t nr) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;  // one face row
    const uint32_t r1 = (nr == 1) ? r0 + 2 : r0 + nr;
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t base = t * entry;
        // rows below 16 live in faces 0/1, rows >= 16 in faces 2/3; write each face span separately
        for (uint32_t lo = r0; lo < r1;) {
            const uint32_t hi = (lo < 16) ? (r1 < 16 ? r1 : 16) : r1;
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = tile_elem_index(lo, half * 16) * esz;
                noc.async_write(
                    dfb,
                    acc,
                    seg * (hi - lo),
                    {.offset_bytes = base + off},
                    {.page_id = first_page + t, .offset_bytes = off});
            }
            lo = hi;
        }
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}

}  // namespace gdn_step_df
