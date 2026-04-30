// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
#if defined(REDUCE_METAL2_NAMED_ARGS) || defined(REDUCE_MCW_NAMED_ARGS)
    constexpr uint32_t scaler_bits = get_named_compile_time_arg_val("scaler_bits");
    constexpr uint32_t src_args_config = get_named_compile_time_arg_val("src_args_config");
    constexpr uint32_t src_page_size = get_named_compile_time_arg_val("src_page_size");
#else
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();
#endif

    constexpr uint32_t cb_id_in2 = 2;
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2, REDUCE_OP, REDUCE_DIM>(scaler_f);

    constexpr uint32_t cb_id_in0 = 0;

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);

#if defined(REDUCE_METAL2_NAMED_ARGS) || defined(REDUCE_MCW_NAMED_ARGS)
    constexpr bool src_is_dram = (src_args_config & static_cast<uint32_t>(tensor_accessor::ArgConfig::IsDram)) != 0;
    const auto tensor_accessor =
        TensorAccessor(tensor_accessor::make_interleaved_dspec<src_is_dram>(), src_addr, src_page_size);
#else
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);
#endif
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_in0.reserve_back(onetile);
        noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
}
