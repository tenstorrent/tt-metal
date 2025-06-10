// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void fill_cb_with_value(uint32_t cb_id, uint32_t value_bits) {
    cb_reserve_back(cb_id, 1);

    // Get the actual tile size for this CB
    uint32_t tile_size_bytes = get_tile_size(cb_id);

#if defined FP32_DEST_ACC_EN
    // FP32 mode - value_bits is already the correct 32-bit representation
    uint32_t num_elements = tile_size_bytes / sizeof(uint32_t);
    auto ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t j = 0; j < num_elements; j++) {
        ptr[j] = value_bits;
    }
#else
    // FP16/BFLOAT16 mode - value_bits contains the 16-bit representation in lower bits
    uint32_t num_elements = tile_size_bytes / sizeof(uint16_t);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id));
    uint16_t value_16bit = static_cast<uint16_t>(value_bits & 0xFFFF);
    for (uint32_t j = 0; j < num_elements; j++) {
        ptr[j] = value_16bit;
    }
#endif

    cb_push_back(cb_id, 1);
}

void kernel_main() {
    // Runtime arguments - tensor-tensor-scalar case
    uint32_t src0_addr = get_arg_val<uint32_t>(0);         // predicate tensor
    uint32_t src1_addr = get_arg_val<uint32_t>(1);         // value_true tensor
    uint32_t value_false_bits = get_arg_val<uint32_t>(2);  // scalar value_false
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(3);

    // Compile time arguments
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);  // predicate CB
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(3);  // value_true CB (tensor)
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(5);  // value_false CB (scalar)

    // Setup for predicate tensor (always needed)
    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    // Setup for value_true tensor (tensor)
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    constexpr bool src1_is_dram = get_compile_time_arg_val(2) == 1;
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};

    for (uint32_t i = 0; i < num_tiles_per_core; i++) {
        // Read predicate tensor
        cb_reserve_back(cb_id_in0, 1);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read(get_noc_addr(i, s0), l1_write_addr_in0, src0_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        // Read value_true tensor
        cb_reserve_back(cb_id_in1, 1);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read(get_noc_addr(i, s1), l1_write_addr_in1, src1_tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, 1);

        // Fill value_false CB with scalar
        fill_cb_with_value(cb_id_in2, value_false_bits);
    }
}
