// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common kernel functions used for debugging
#pragma once
#include "debug/dprint.h"
namespace tt::data_movement::common {
inline void print_bf16_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

inline void print_f32_pages(uint32_t l1_addr, uint32_t elts_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr) + start * elts_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < elts_per_page; ++j, ++ptr) {
            DPRINT << F32(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

inline void print_u8_pages(uint32_t l1_addr, uint32_t bytes_per_page, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint8_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_addr) + start * bytes_per_page;
    for (uint32_t page = 0; page < npages; ++page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < bytes_per_page; ++j, ++ptr) {
            DPRINT << SETW(2) << HEX() << "0x" << (uint32_t)*ptr << " ";
        }
        DPRINT << DEC();  // revert to decimal representation
        DPRINT << ENDL();
    }
}
}  // namespace tt::data_movement::common
