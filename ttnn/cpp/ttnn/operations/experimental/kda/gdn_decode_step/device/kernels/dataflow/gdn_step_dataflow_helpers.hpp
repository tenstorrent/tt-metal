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

}  // namespace gdn_step_df
