// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_reduce_universal_start_id.cpp. Identical dataflow: fill the reduce
// scaler DFB once (prepare_reduce_scaler), then stream input tiles into the input DFB. CB indices →
// dfb:: bindings, the source TensorAccessor → tensor:: binding (src_addr runtime arg gone), scaler_bits
// → named CTA, count/start → named RTAs. The legacy copy is retained for not-yet-ported reduce factories.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);

    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);

    const auto tensor_accessor = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer cb_in0(dfb::in);
    const uint32_t tile_bytes = cb_in0.get_entry_size();

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_in0.reserve_back(1);
        noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(1);
    }
}
