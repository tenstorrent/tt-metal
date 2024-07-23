// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void fill_cb_with_value(uint32_t cb_id, uint32_t value, int32_t num_of_elems = 1024) {
    cb_reserve_back(cb_id, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_id));
    for (int j = 0; j < num_of_elems; j++) {
        ptr[j] = uint16_t(value >> 16);
    }
    cb_push_back(cb_id, 1);
}

void generate_mask_h_w(uint32_t cb_mask_h_w, uint32_t mask_h, uint32_t mask_w, uint32_t single_tile_size = 2048) {
    union {
        float f;
        uint32_t u;
    } one;
    one.f = 1.0f;
    union {
        float f;
        uint32_t u;
    } zero;
    zero.f = 0.0f;

    const auto u16_one = uint16_t(one.u >> 16);
    const auto u16_zero = uint16_t(zero.u >> 16);

    cb_reserve_back(cb_mask_h_w, 2);

    // mask_h
    // first tile ptr
    auto mask_h_ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask_h_w));
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
    auto mask_w_ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask_h_w) + single_tile_size);
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

void generate_mask_w(uint32_t cb_mask, uint32_t mask_w) {
    union {
        float f;
        uint32_t u;
    } one;
    one.f = 1.0f;
    union {
        float f;
        uint32_t u;
    } zero;
    zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask));

    for (uint32_t h = 0; h < 16; h++) {
        // sub tile 0
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16)
                mask_w_0 = 16;
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            uint32_t mask_w_0 = mask_w;
            if (mask_w_0 >= 16)
                mask_w_0 = 16;
            uint32_t w = 0;
            for (; w < mask_w_0; w++) {
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            uint32_t mask_w_1 = (mask_w < 16) ? 0 : mask_w - 16;
            uint32_t w = 0;
            for (; w < mask_w_1; w++) {
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for (; w < 16; w++) {
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}

void generate_mask_h(uint32_t cb_mask, uint32_t mask_h) {
    union {
        float f;
        uint32_t u;
    } one;
    one.f = 1.0f;
    union {
        float f;
        uint32_t u;
    } zero;
    zero.f = 0.0f;

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask));

    for (uint32_t w = 0; w < 16; w++) {
        // sub tile 0
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16)
                mask_h_0 = 16;
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                ptr[h * 16 + w] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 1
        {
            uint32_t mask_h_0 = mask_h;
            if (mask_h_0 >= 16)
                mask_h_0 = 16;
            uint32_t h = 0;
            for (; h < mask_h_0; h++) {
                ptr[h * 16 + w + 256] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w + 256] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 2
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                ptr[h * 16 + w + 512] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w + 512] = uint16_t(zero.u >> 16);
            }
        }

        // sub tile 3
        {
            uint32_t mask_h_1 = (mask_h < 16) ? 0 : mask_h - 16;
            uint32_t h = 0;
            for (; h < mask_h_1; h++) {
                ptr[h * 16 + w + 768] = uint16_t(one.u >> 16);
            }
            for (; h < 16; h++) {
                ptr[h * 16 + w + 768] = uint16_t(zero.u >> 16);
            }
        }
    }

    cb_push_back(cb_mask, 1);
}
