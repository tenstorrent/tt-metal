// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for dfb_helpers_compute.hpp
// Do not include directly - include dfb_helpers_compute.hpp instead

#include "api/compute/cb_api.h"
#ifdef ARCH_QUASAR
// ckernel::trisc::tile_counters — the per-Tensix HW tile-counter registers that hold each DFB's
// capacity in entries/pages (the same source DataflowBuffer's reserve/wait/pop assert against).
#include "ckernel_trisc_common.h"
#endif

namespace compute_kernel_lib {

constexpr uint32_t DATUM_SHIFT = 10;  // 32x32 = 1024 datums
constexpr uint32_t EXP_SHIFT = 6;    // 64 exponents for block formats

ALWI constexpr uint32_t get_full_tile_size_impl(DataFormat format) {
    switch (format) {
        case DataFormat::UInt8:
#ifndef ARCH_QUASAR
        case DataFormat::Lf8:
#endif
        case DataFormat::Int8:
            return (1 << DATUM_SHIFT);
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
            return (1 << (DATUM_SHIFT + 1));
        case DataFormat::Float32:
        case DataFormat::Int32:
#ifndef ARCH_QUASAR
        case DataFormat::UInt32:
#endif
            return (1 << (DATUM_SHIFT + 2));
#ifndef ARCH_QUASAR
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b:
            return (1 << DATUM_SHIFT) + (1 << EXP_SHIFT);
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b:
            return (1 << (DATUM_SHIFT - 1)) + (1 << EXP_SHIFT);
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b:
            return (1 << (DATUM_SHIFT - 2)) + (1 << EXP_SHIFT);
#endif
        default:
            return 0;
    }
}

template <DataFormat format>
ALWI constexpr uint32_t get_full_tile_size() {
    return get_full_tile_size_impl(format);
}

ALWI uint32_t get_full_tile_size(DataFormat format) {
    return get_full_tile_size_impl(format);
}

ALWI constexpr bool is_block_float_format(uint32_t format) {
    return format == 2 || format == 3 || format == 11 || format == 6 || format == 7 || format == 15;
}

// Returns the DFB's capacity in entries (pages) — used by the debug capacity ASSERTs in the
// tilize/untilize/reduce helpers (both `>=` and the `% tiles_per_bulk == 0` check in reduce).
ALWI uint32_t get_dfb_num_pages(uint32_t dfb_id) {
#ifdef ARCH_QUASAR
    // Quasar compute kernels track DFB state in g_dfb_interface, NOT cb_interface (the latter is
    // undefined on the Quasar TRISC link). The buffer's capacity in entries/pages lives in the HW
    // tile-counter register — exactly the value DataflowBuffer's reserve_back/wait_front/pop_front
    // assert against (see dataflow_buffer.inl), so read it the same way. A max sentinel would be
    // wrong for the `num_pages % tiles_per_bulk == 0` check in reduce_helpers_compute.inl. There is
    // no compile-time per-DFB capacity descriptor (the JIT emits tile format/dims only, not ring
    // depth), and even WH/BH derive this at runtime from cb_interface — so this is the runtime
    // equivalent. tile_counters is a per-Tensix register shared across TRISCs, set at DFB init.
    const LocalDFBInterface& dfb = get_local_dfb_interface(dfb_id);
    const dfb::PackedTileCounter ptc = dfb.tc_slots[dfb.tc_idx].packed_tile_counter;
    const uint8_t tc_id = dfb::get_counter_id(ptc);
    return ckernel::trisc::tile_counters[tc_id].f.buf_capacity;
#else
    auto& cb = get_local_cb_interface(dfb_id);
    return cb.fifo_size / cb.fifo_page_size;
#endif
}

#ifndef ARCH_QUASAR
// CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT value (= 4): fifo_page_size is stored in
// 16-byte units on WH/BH, so shifting left by 4 converts to bytes.
constexpr uint32_t DFB_COMPUTE_ADDR_SHIFT = 4; // almeet

template <DataFormat format>
ALWI bool is_valid_dfb_tile_page_size(uint32_t dfb_id) {
    uint32_t tile_size = get_full_tile_size<format>();
    uint32_t page_size_bytes = get_local_cb_interface(dfb_id).fifo_page_size << DFB_COMPUTE_ADDR_SHIFT;
    return page_size_bytes == tile_size;
}

ALWI bool is_valid_dfb_tile_page_size(uint32_t dfb_id, DataFormat format) {
    uint32_t tile_size = get_full_tile_size(format);
    uint32_t page_size_bytes = get_local_cb_interface(dfb_id).fifo_page_size << DFB_COMPUTE_ADDR_SHIFT;
    return page_size_bytes == tile_size;
}
#endif  // !ARCH_QUASAR

template <uint32_t dfb_id>
constexpr uint32_t dfb_l1_format() {
#if defined(UCK_CHLKC_PACK)
    return pack_dst_format[dfb_id];
#else
    return unpack_src_format[dfb_id];
#endif
}

template <uint32_t dfb_id>
constexpr bool dfb_has_32x32_tiles() {
#if defined(UCK_CHLKC_PACK)
    constexpr uint32_t tile_r_dim = pack_tile_r_dim[dfb_id];
    constexpr uint32_t tile_c_dim = pack_tile_c_dim[dfb_id];
#else
    constexpr uint32_t tile_r_dim = unpack_tile_r_dim[dfb_id];
    constexpr uint32_t tile_c_dim = unpack_tile_c_dim[dfb_id];
#endif
    return tile_r_dim == 32 && tile_c_dim == 32;
}

}  // namespace compute_kernel_lib
