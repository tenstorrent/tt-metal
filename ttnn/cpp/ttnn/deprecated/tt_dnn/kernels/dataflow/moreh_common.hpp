// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <cstdint>
#include <cstring>

#include "tt-metalium/constants.hpp"
#include "dataflow_api.h"
#include "noc/noc_parameters.h"

constexpr std::uint32_t NOC_MINIMUM_READ_SIZE = 32;  // 32 Bytes

static inline float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t uint32_data = ((uint32_t)bfloat_val) << 16;
    float f;
    std::memcpy(&f, &uint32_data, sizeof(f));
    return f;
}

static inline uint16_t float_to_bfloat16(float val) {
    union {
        float f;
        uint32_t u;
    } ret;
    ret.f = val;
    return uint16_t(ret.u >> 16);
}

#if defined(FP32_DEST_ACC_EN)
using FP32_DEST_ACC_FTYPE = float;
FORCE_INLINE FP32_DEST_ACC_FTYPE fp32_dest_acc_cast(uint16_t val) { return bfloat16_to_float(val); }
FORCE_INLINE FP32_DEST_ACC_FTYPE fp32_dest_acc_cast(float val) { return val; }
#else
using FP32_DEST_ACC_FTYPE = uint16_t;
FORCE_INLINE FP32_DEST_ACC_FTYPE fp32_dest_acc_cast(uint16_t val) { return val; }
FORCE_INLINE FP32_DEST_ACC_FTYPE fp32_dest_acc_cast(float val) {
    union {
        float f;
        uint32_t u;
    } ret;
    ret.f = val;
    return FP32_DEST_ACC_FTYPE(ret.u >> 16);
}
#endif

union Scalar {
    float f;
    uint32_t u;
};

class ArgFetcher {
private:
    int arg_idx = 0;

public:
    template <typename T>
    T get_next_arg_val() {
        return get_arg_val<T>(arg_idx++);
    }
};

template <typename AddrGen>
FORCE_INLINE void noc_async_read_tile_helper(tt::CBIndex cb, uint32_t num_tiles, uint32_t tile_idx, AddrGen addr_gen) {
    cb_reserve_back(cb, num_tiles);
    uint32_t addr = get_write_ptr(cb);
    noc_async_read_tile(tile_idx, addr_gen, addr);
    noc_async_read_barrier();
    cb_push_back(cb, num_tiles);
}

template <typename AddrGen>
FORCE_INLINE void noc_async_write_tile_helper(tt::CBIndex cb, uint32_t num_tiles, uint32_t tile_idx, AddrGen addr_gen) {
    cb_wait_front(cb, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb);
    noc_async_write_tile(tile_idx, addr_gen, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb, num_tiles);
}

template <typename T = uint16_t>
FORCE_INLINE void generate_bcast_scaler(uint32_t cb_scaler, uint32_t scaler) {
    union {
        float f;
        uint32_t u;
    } u = {};
    u.u = scaler;
    cb_reserve_back(cb_scaler, 1);
    auto ptr = reinterpret_cast<T*>(get_write_ptr(cb_scaler));

    for (int j = 0; j < 1024; j++) {
        ptr[j] = T(0);
    }

    for (int k = 0; k < 4; k++) {
        for (int j = 0; j < 16; j++) {
            if constexpr (std::is_same_v<T, uint16_t>) {
                ptr[k * 256 + j] = T(u.u >> 16);
            } else {
                ptr[k * 256 + j] = T(u.u);
            }
        }
    }

    cb_push_back(cb_scaler, 1);
}

template <typename T>
FORCE_INLINE void process_data(int cb_id, uint32_t value, int32_t num_of_elems) {
    T* ptr = reinterpret_cast<T*>(get_write_ptr(cb_id));
    for (int j = 0; j < num_of_elems; j++) {
        ptr[j] = static_cast<T>(value);
    }
}

template <>
FORCE_INLINE void process_data<uint16_t>(int cb_id, uint32_t value, int32_t num_of_elems) {
    uint16_t* ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id));
    for (int j = 0; j < num_of_elems; j++) {
        ptr[j] = static_cast<uint16_t>(value >> 16);
    }
}

FORCE_INLINE void fill_cb_with_value(uint32_t cb_id, uint32_t value, int32_t num_of_elems = 1024) {
    cb_reserve_back(cb_id, 1);
    const DataFormat data_format = get_dataformat(cb_id);
    switch ((uint)data_format & 0x1F) {
        case ((uint8_t)DataFormat::Float32):
        case ((uint8_t)DataFormat::Int32):
        case ((uint8_t)DataFormat::UInt32): process_data<uint32_t>(cb_id, value, num_of_elems); break;
        case ((uint8_t)DataFormat::Float16_b):
        default: process_data<uint16_t>(cb_id, value, num_of_elems); break;
    }
    cb_push_back(cb_id, 1);
}

FORCE_INLINE uint32_t get_tilized_idx(uint32_t h_idx, uint32_t w_idx, uint32_t tile_height, uint32_t tile_width) {
    const auto half_tile_height = tile_height / 2;
    const auto half_tile_width = tile_width / 2;

    if (h_idx < half_tile_height && w_idx < half_tile_width) {
        return h_idx * half_tile_width + w_idx;
    }

    if (h_idx < half_tile_height && w_idx >= half_tile_width) {
        return h_idx * half_tile_width + (w_idx % half_tile_width) + half_tile_height * half_tile_width;
    }

    if (h_idx >= half_tile_height && w_idx < half_tile_width) {
        return (h_idx % half_tile_width) * half_tile_width + w_idx + half_tile_height * tile_width;
    }

    return (h_idx % half_tile_height) * half_tile_width + (w_idx % half_tile_width) +
           half_tile_height * (tile_width + half_tile_width);
}

FORCE_INLINE uint32_t get_gamma_beta_tile_idx(uint32_t input_tile_idx, uint32_t HtWt, uint32_t C, uint32_t tile_width) {
    const auto c_idx = (input_tile_idx / HtWt) % C;
    const auto tile_idx = c_idx / tile_width;
    return tile_idx;
}

FORCE_INLINE uint32_t get_tilized_gamma_beta_idx_in_tile(
    uint32_t input_tile_idx, uint32_t HtWt, uint32_t C, uint32_t tile_height, uint32_t tile_width) {
    const auto c_idx = (input_tile_idx / HtWt) % C;
    const auto w_idx_in_tile = c_idx % tile_width;
    const auto tilized_idx_in_tile = get_tilized_idx(0, w_idx_in_tile, tile_height, tile_width);
    return tilized_idx_in_tile;
}

template <typename T>
FORCE_INLINE T get_mask_value(const Scalar& scalar, bool is_one) {
    if constexpr (std::is_same_v<T, uint16_t>) {
        return T(scalar.u >> 16);
    } else {
        return T(scalar.u);
    }
}

template <typename T>
FORCE_INLINE void fill_subtile_mask_h(
    T* ptr, uint32_t w, uint32_t subtile_offset, uint32_t active_height, const Scalar& one, const Scalar& zero) {
    uint32_t h = 0;
    // Fill active elements with one
    for (; h < active_height; h++) {
        ptr[h * 16 + w + subtile_offset] = get_mask_value<T>(one, true);
    }
    // Fill remaining elements with zero
    for (; h < 16; h++) {
        ptr[h * 16 + w + subtile_offset] = get_mask_value<T>(zero, false);
    }
}

template <typename T>
FORCE_INLINE void fill_subtile_mask_w(
    T* ptr, uint32_t h, uint32_t subtile_offset, uint32_t active_width, const Scalar& one, const Scalar& zero) {
    uint32_t w = 0;
    // Fill active elements with one
    for (; w < active_width; w++) {
        ptr[h * 16 + w + subtile_offset] = get_mask_value<T>(one, true);
    }
    // Fill remaining elements with zero
    for (; w < 16; w++) {
        ptr[h * 16 + w + subtile_offset] = get_mask_value<T>(zero, false);
    }
}

template <typename T = uint16_t>
FORCE_INLINE void generate_mask_h(uint32_t cb_mask, uint32_t mask_h) {
    Scalar one = {};
    Scalar zero = {};

    if constexpr (std::is_same_v<T, int32_t>) {
        one.u = 1;
        zero.u = 0;
    } else {
        one.f = 1.0f;
        zero.f = 0.0f;
    }

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<T*>(get_write_ptr(cb_mask));

    // Calculate mask heights for top and bottom subtiles
    const uint32_t mask_h_top = (mask_h >= 16) ? 16 : mask_h;        // subtiles 0, 1
    const uint32_t mask_h_bottom = (mask_h < 16) ? 0 : mask_h - 16;  // subtiles 2, 3

    // Define subtile offsets: [0, 256, 512, 768] for subtiles [0, 1, 2, 3]
    constexpr uint32_t subtile_offsets[4] = {0, 256, 512, 768};

    for (uint32_t w = 0; w < 16; w++) {
        // sub tile 0 (top-left)
        fill_subtile_mask_h<T>(ptr, w, subtile_offsets[0], mask_h_top, one, zero);

        // sub tile 1 (top-right)
        fill_subtile_mask_h<T>(ptr, w, subtile_offsets[1], mask_h_top, one, zero);

        // sub tile 2 (bottom-left)
        fill_subtile_mask_h<T>(ptr, w, subtile_offsets[2], mask_h_bottom, one, zero);

        // sub tile 3 (bottom-right)
        fill_subtile_mask_h<T>(ptr, w, subtile_offsets[3], mask_h_bottom, one, zero);
    }

    cb_push_back(cb_mask, 1);
}

template <typename T = uint16_t>
FORCE_INLINE void generate_mask_w(uint32_t cb_mask, uint32_t mask_w) {
    Scalar one = {};
    Scalar zero = {};

    if constexpr (std::is_same_v<T, int32_t>) {
        one.u = 1;
        zero.u = 0;
    } else {
        one.f = 1.0f;
        zero.f = 0.0f;
    }

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<T*>(get_write_ptr(cb_mask));

    // Calculate mask widths for left and right subtiles
    const uint32_t mask_w_left = (mask_w >= 16) ? 16 : mask_w;      // subtiles 0, 2
    const uint32_t mask_w_right = (mask_w < 16) ? 0 : mask_w - 16;  // subtiles 1, 3

    // Define subtile offsets: [0, 256, 512, 768] for subtiles [0, 1, 2, 3]
    constexpr uint32_t subtile_offsets[4] = {0, 256, 512, 768};

    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0 (top-left)
        fill_subtile_mask_w<T>(ptr, h, subtile_offsets[0], mask_w_left, one, zero);

        // sub tile 1 (top-right)
        fill_subtile_mask_w<T>(ptr, h, subtile_offsets[1], mask_w_right, one, zero);

        // sub tile 2 (bottom-left)
        fill_subtile_mask_w<T>(ptr, h, subtile_offsets[2], mask_w_left, one, zero);

        // sub tile 3 (bottom-right)
        fill_subtile_mask_w<T>(ptr, h, subtile_offsets[3], mask_w_right, one, zero);
    }

    cb_push_back(cb_mask, 1);
}

FORCE_INLINE void generate_mask_h_w(
    uint32_t cb_mask_h_w, uint32_t mask_h, uint32_t mask_w, uint32_t single_tile_size = 2048) {
    Scalar one = {};
    Scalar zero = {};

    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

    cb_reserve_back(cb_mask_h_w, 2);

    // mask_h
    // first tile ptr
    auto mask_h_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask_h_w));
    for (uint32_t w = 0; w < 16; w++) {
        // sub tile 0
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) {
                mask_h_0 = 16;
            }
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                mask_h_ptr[h * 16 + w] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w] = u16_zero;
            }
        }

        // sub tile 1
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) {
                mask_h_0 = 16;
            }
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                mask_h_ptr[h * 16 + w + 256] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 256] = u16_zero;
            }
        }

        // sub tile 2
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                mask_h_ptr[h * 16 + w + 512] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 512] = u16_zero;
            }
        }

        // sub tile 3
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                mask_h_ptr[h * 16 + w + 768] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }

    // mask_w
    // second tile ptr
    auto mask_w_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask_h_w) + single_tile_size);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) {
                mask_w_0 = 16;
            }
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                mask_w_ptr[h * 16 + w] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w] = u16_zero;
            }
        }

        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                mask_w_ptr[h * 16 + w + 256] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 256] = u16_zero;
            }
        }

        // sub tile 2
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) {
                mask_w_0 = 16;
            }
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                mask_w_ptr[h * 16 + w + 512] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 512] = u16_zero;
            }
        }

        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                mask_w_ptr[h * 16 + w + 768] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }

    cb_push_back(cb_mask_h_w, 2);
}

FORCE_INLINE void generate_mask_h_w_if_needed(uint32_t cb_mask_h_w, uint32_t origin_h, uint32_t origin_w) {
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const uint32_t mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const uint32_t mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_h || do_mask_w) {
        const uint32_t mask_tile_bytes = get_tile_size(cb_mask_h_w);
        generate_mask_h_w(cb_mask_h_w, mask_h, mask_w, mask_tile_bytes);
    }
}

FORCE_INLINE void mask_tile_hw(uint32_t l1_addr, uint32_t mask_h = 32, uint32_t mask_w = 32) {
    Scalar zero = {};
    zero.f = 0.0f;
    const auto u16_zero = uint16_t(zero.u >> 16);

    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w] = u16_zero;
            }
        }
        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_0 = (mask_h >= 16) ? 16 : mask_h;
            uint32_t w = (h >= mask_h_0) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 256] = u16_zero;
            }
        }
        // sub tile 2
        {
            uint32_t mask_w_0 = (mask_w >= 16) ? 16 : mask_w;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_0;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 512] = u16_zero;
            }
        }
        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t w = (h >= mask_h_1) ? 0 : mask_w_1;
            for (; w < 16; w++) {
                ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }
}

FORCE_INLINE void mask_tile_if_need(uint32_t l1_addr, uint32_t origin_h, uint32_t origin_w) {
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const uint32_t mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const uint32_t mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_h || do_mask_w) {
        mask_tile_hw(l1_addr, mask_h, mask_w);
    }
}

FORCE_INLINE void generate_mask_tiles(
    uint32_t cb_mask, uint32_t mask_h, uint32_t mask_w, uint32_t single_tile_size = 2048) {
    constexpr uint32_t num_mask_tiles = 3;
    Scalar one = {};
    Scalar zero = {};

    one.f = 1.0f;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

    cb_reserve_back(cb_mask, num_mask_tiles);

    // mask_h
    // first tile ptr
    auto mask_h_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask));
    for (uint32_t w = 0; w < 16; w++) {
        // sub tile 0
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) {
                mask_h_0 = 16;
            }
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                mask_h_ptr[h * 16 + w] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w] = u16_zero;
            }
        }

        // sub tile 1
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16) {
                mask_h_0 = 16;
            }
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                mask_h_ptr[h * 16 + w + 256] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 256] = u16_zero;
            }
        }

        // sub tile 2
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                mask_h_ptr[h * 16 + w + 512] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 512] = u16_zero;
            }
        }

        // sub tile 3
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                mask_h_ptr[h * 16 + w + 768] = u16_one;
            }
            for (; h < 16; h++) {
                mask_h_ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }

    // mask_w
    // second tile ptr
    auto mask_w_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask) + single_tile_size);
    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) {
                mask_w_0 = 16;
            }
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                mask_w_ptr[h * 16 + w] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w] = u16_zero;
            }
        }

        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                mask_w_ptr[h * 16 + w + 256] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 256] = u16_zero;
            }
        }

        // sub tile 2
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16) {
                mask_w_0 = 16;
            }
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                mask_w_ptr[h * 16 + w + 512] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 512] = u16_zero;
            }
        }

        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                mask_w_ptr[h * 16 + w + 768] = u16_one;
            }
            for (; w < 16; w++) {
                mask_w_ptr[h * 16 + w + 768] = u16_zero;
            }
        }
    }

    // mask_hw
    // third tile ptr
    auto mask_hw_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask) + (single_tile_size * 2));
    for (uint32_t i = 0; i < 1024; ++i) {
        if (mask_h_ptr[i] == mask_w_ptr[i]) {
            mask_hw_ptr[i] = mask_h_ptr[i];
        } else {
            mask_hw_ptr[i] = u16_zero;
        }
    }
    cb_push_back(cb_mask, num_mask_tiles);
}

uint32_t get_tilized_idx(uint32_t h, uint32_t w) {
    using namespace tt::constants;
    h = h % TILE_HEIGHT;
    w = w % TILE_WIDTH;
    uint32_t idx = 0;
    if (w >= FACE_WIDTH) {
        w -= FACE_WIDTH;
        idx += FACE_HEIGHT * FACE_WIDTH;
    }
    if (h >= FACE_WIDTH) {
        h -= FACE_WIDTH;
        idx += FACE_HEIGHT * TILE_WIDTH;
    }

    idx += h * FACE_WIDTH + w;
    return idx;
}

void get_noc_offset(uint32_t h, uint32_t w, uint32_t element_size, uint32_t& noc_offset) {
    using namespace tt::constants;
    noc_offset = 0;

    // compute h, w in tile
    h = h - (h / TILE_HEIGHT) * TILE_HEIGHT;
    w = w - (w / TILE_WIDTH) * TILE_WIDTH;

    const bool is_even_face = (w < FACE_HEIGHT);
    const bool is_odd_face = !is_even_face;

    const uint32_t face_width_bytes = FACE_WIDTH * element_size;

    if (h < FACE_WIDTH && is_even_face) {
        noc_offset += h * face_width_bytes + w * element_size;  // face 0
    } else if (h < FACE_WIDTH && is_odd_face) {
        noc_offset += (FACE_HEIGHT + h) * face_width_bytes + (w - FACE_WIDTH) * element_size;  // face 1
    } else if (h >= FACE_WIDTH && is_even_face) {
        noc_offset += (FACE_HEIGHT + h) * face_width_bytes + w * element_size;  // face 2
    } else if (h >= FACE_WIDTH && is_odd_face) {
        noc_offset += (2 * FACE_HEIGHT + h) * face_width_bytes + (w - FACE_WIDTH) * element_size;  // face 3
    }

    const uint32_t noc_offset_align = (noc_offset / NOC_MINIMUM_READ_SIZE) * NOC_MINIMUM_READ_SIZE;

    noc_offset = noc_offset_align;
}

template <typename T>
volatile tt_l1_ptr T* get_read_ptr(uint32_t cb_id) {
    auto l1_write_addr = get_read_ptr(cb_id);
    auto l1_ptr = reinterpret_cast<volatile tt_l1_ptr T*>(l1_write_addr);
    return l1_ptr;
}

template <typename T>
volatile tt_l1_ptr T* get_write_ptr(uint32_t cb_id) {
    auto l1_write_addr = get_write_ptr(cb_id);
    auto l1_ptr = reinterpret_cast<volatile tt_l1_ptr T*>(l1_write_addr);
    return l1_ptr;
}

// It reads values from one tile.
template <typename T>
void read_tile(
    uint32_t cb_id,
    T addrgen,
    uint32_t noc_id,
    uint32_t size = 0,
    uint32_t offset = 0,
    bool do_reserve = true,
    bool do_push_back = true) {
    constexpr uint32_t onetile = 1;

    if (do_reserve) {
        cb_reserve_back(cb_id, onetile);
    }

    // If the size is 0, it reads one tile.
    if (size == 0) {
        size = get_tile_size(cb_id);
    }

    auto l1_write_addr = get_write_ptr(cb_id);
    auto noc_addr = get_noc_addr(noc_id, addrgen, offset);
    noc_async_read(noc_addr, l1_write_addr, size);

    noc_async_read_barrier();

    if (do_push_back) {
        cb_push_back(cb_id, onetile);
    }
}

template <typename T>
void read_value(
    uint32_t cb_id,
    T addrgen,
    uint32_t noc_id,
    uint32_t tilized_idx = 0,
    bool do_reserve = true,
    bool do_push_back = true) {
    constexpr uint32_t onetile = 1;

    if (do_reserve) {
        cb_reserve_back(cb_id, onetile);
    }

    uint32_t size = get_tile_size(cb_id) / 1024;

    auto l1_write_addr = get_write_ptr(cb_id);
    auto noc_addr = get_noc_addr(noc_id, addrgen);
    noc_async_read(noc_addr + tilized_idx * size, l1_write_addr + tilized_idx * size, size);
    noc_async_read_barrier();

    if (do_push_back) {
        cb_push_back(cb_id, onetile);
    }
}

// clang-format off
/**
 * Reads values from a tilized tensor with shape (1, W).
 * The assumption is that only the first row of the tensor contains useful data.
 *
 * Return value: None
 *
 * | Argument                     | Description                             | Data type | Valid range                    | required |
 * |------------------------------|-----------------------------------------|-----------|--------------------------------|----------|
 * | cb_id                        | Destination CB for the read data        | uint32_t  | Any valid CB ID                | True     |
 * | cb_scratch_id                | CB to use as scratch storage            | uint32_t  | Any valid CB ID                | True     |
 * | addrgen                      | Address generator object                | AddrGen   | N/A                            | True     |
 * | num_tiles                    | Number of tiles to read                 | uint32_t  | Any uint32_t number            | True     |
 * | do_reserve                   | Whether to reserve space in the CB      | bool      | true or false                  | False    |
 * | do_push_back                 | Whether to push the data back to the CB | bool      | true or false                  | False    |
 */
// clang-format on
template <typename T>
void read_line(
    uint32_t cb_id,
    uint32_t cb_scratch_id,
    T addrgen,
    uint32_t num_tiles,
    bool do_reserve = true,
    bool do_push_back = true) {
    using namespace tt::constants;
    if (do_reserve) {
        cb_reserve_back(cb_id, num_tiles);
    }

    auto tile_bytes = get_tile_size(cb_id);
    auto element_bytes = tile_bytes / (TILE_HEIGHT * TILE_WIDTH);
    auto valid_elements_bytes = FACE_WIDTH * element_bytes;

    // We want to read all valid elements, but may need to read more from DRAM,
    // because DRAM has larger read size than L1 on some architectures.
    auto noc_read_size_bytes = std::max(valid_elements_bytes, static_cast<uint32_t>(NOC_DRAM_READ_ALIGNMENT_BYTES));

    uint32_t l1_write_addr = get_write_ptr(cb_id);
    for (uint32_t i = 0; i < num_tiles * 2; ++i) {
        uint32_t noc_id = i / 2;
        uint32_t noc_offset = 0;
        if (noc_id * 2 != i) {
            noc_offset += (FACE_HEIGHT * FACE_WIDTH) * element_bytes;
        }
        auto src_noc_addr = get_noc_addr(noc_id, addrgen, noc_offset);

        if (noc_read_size_bytes == valid_elements_bytes) {
            // DRAM and L1 read sizes are aligned, so we can read directly into the destination CB.
            noc_async_read(src_noc_addr, l1_write_addr, noc_read_size_bytes);
            noc_async_read_barrier();
        } else {
            // DRAM has larger read size than L1, so there will be some padding in data read from DRAM.
            // Need to use scratch CB to read from DRAM, then copy valid parts to the destination CB.
            auto scratch_l1_write_addr = get_write_ptr(cb_scratch_id);
            noc_async_read(src_noc_addr, scratch_l1_write_addr, noc_read_size_bytes);
            noc_async_read_barrier();
            auto scratch_l1_noc_read_addr = get_noc_addr(scratch_l1_write_addr);
            // Now copy only the valid elements to the destination CB.
            noc_async_read(scratch_l1_noc_read_addr, l1_write_addr, valid_elements_bytes);
            noc_async_read_barrier();
        }

        l1_write_addr += valid_elements_bytes;
    }

    if (do_push_back) {
        cb_push_back(cb_id, num_tiles);
    }
}
