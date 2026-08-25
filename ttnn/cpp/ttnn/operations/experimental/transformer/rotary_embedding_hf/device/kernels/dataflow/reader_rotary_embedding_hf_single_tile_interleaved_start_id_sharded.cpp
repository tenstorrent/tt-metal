// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline void fill_rotate_half_trans_mat_bfp8(DataflowBuffer& dfb) {
    constexpr uint32_t onetile = 1;
    dfb.reserve_back(onetile);
    volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb.get_write_ptr());
    for (uint32_t i = 0; i < 1088; ++i) {
        p[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        p[16 + r] = 127;
        p[320 + r * 16 + r] = 0x40;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        p[32 + r] = 127;
        p[576 + r * 16 + r] = 0xC0;
    }
    dfb.push_back(onetile);
}

inline void fill_rotate_half_trans_mat_bf16(DataflowBuffer& dfb) {
    constexpr uint32_t onetile = 1;
    constexpr uint16_t one_bf16 = 0x3F80;
    constexpr uint16_t neg_one_bf16 = 0xBF80;
    constexpr uint32_t face_elems = 16 * 16;

    dfb.reserve_back(onetile);
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dfb.get_write_ptr());
    for (uint32_t i = 0; i < 4 * face_elems; ++i) {
        tile[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[1 * face_elems + r * 16 + r] = one_bf16;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[2 * face_elems + r * 16 + r] = neg_one_bf16;
    }
    dfb.push_back(onetile);
}

inline void fill_rotate_half_trans_mat_fp32(DataflowBuffer& dfb) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t one_fp32 = 0x3F800000;
    constexpr uint32_t neg_one_fp32 = 0xBF800000;
    constexpr uint32_t face_elems = 16 * 16;

    dfb.reserve_back(onetile);
    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    for (uint32_t i = 0; i < 4 * face_elems; ++i) {
        tile[i] = 0;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[1 * face_elems + r * 16 + r] = one_fp32;
    }
    for (uint32_t r = 0; r < 16; ++r) {
        tile[2 * face_elems + r * 16 + r] = neg_one_fp32;
    }
    dfb.push_back(onetile);
}

void kernel_main() {
    Noc noc;

    uint32_t num_rows = get_arg(args::num_rows);
    uint32_t start_row_id = get_arg(args::start_row_id);
    uint32_t cos_sin_start_id = get_arg(args::cos_sin_start_id);

    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t HtWt = get_arg(args::HtWt);

    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_input(dfb::in);
    DataflowBuffer dfb_cos(dfb::cos);
    DataflowBuffer dfb_sin(dfb::sin);
    DataflowBuffer dfb_trans_mat(dfb::trans_mat);

    dfb_input.reserve_back(num_rows);
    dfb_input.push_back(num_rows);

    const uint32_t cos_tile_bytes = dfb_cos.get_tile_size();
    const auto s1 = TensorAccessor(tensor::cos);

    const uint32_t sin_tile_bytes = dfb_sin.get_tile_size();
    const auto s2 = TensorAccessor(tensor::sin);

    const uint32_t trans_mat_tile_size = dfb_trans_mat.get_tile_size();
    if (trans_mat_tile_size == 4096) {
        fill_rotate_half_trans_mat_fp32(dfb_trans_mat);
    } else if (trans_mat_tile_size == 2048) {
        fill_rotate_half_trans_mat_bf16(dfb_trans_mat);
    } else {
        fill_rotate_half_trans_mat_bfp8(dfb_trans_mat);
    }

    uint32_t cos_sin_curr_id = cos_sin_start_id;
    uint32_t ht = start_row_id;
    for (uint32_t i = 0; i < num_rows; ++i) {
        dfb_sin.reserve_back(onetile);
        {
            uint32_t sin_l1_write_addr = dfb_sin.get_write_ptr();
            noc.async_read(
                s2, CoreLocalMem<uint32_t>(sin_l1_write_addr), sin_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
        }
        dfb_sin.push_back(onetile);

        dfb_cos.reserve_back(onetile);
        {
            uint32_t cos_l1_write_addr = dfb_cos.get_write_ptr();
            noc.async_read(
                s1, CoreLocalMem<uint32_t>(cos_l1_write_addr), cos_tile_bytes, {.page_id = cos_sin_curr_id}, {});
            noc.async_read_barrier();
        }
        dfb_cos.push_back(onetile);
        cos_sin_curr_id++;

        ht++;
        if (ht == Ht) {
            ht = 0;
            cos_sin_curr_id -= HtWt;
        }
    }
}
