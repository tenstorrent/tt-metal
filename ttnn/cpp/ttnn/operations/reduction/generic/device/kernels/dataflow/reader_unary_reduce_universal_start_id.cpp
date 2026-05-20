// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_reduce_universal_start_id.cpp.
//
// Differences from the legacy reader:
//   - Named arg retrieval via get_arg(args::name)
//   - Named DFB bindings (dfb::input, dfb::scaler) replace magic CB indices
//   - TensorBinding (ta::input) replaces TensorAccessorArgs + buffer-address RTA
//   - get_tile_size(cb_id) -> cb_obj.get_tile_size() (Device 2.0 cleanup)
//
// Host bindings expected (per the W factory's KernelSpec):
//   compile_time_arg_bindings: { {"scaler_bits", ...} }
//   runtime_arguments_schema.named_runtime_args: { "num_tiles", "start_id" }
//   dfb_bindings: { INPUT (CONSUMER, name="input"), SCALER (PRODUCER, name="scaler") }
//   tensor_bindings: { INPUT_TENSOR (name="input") }

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);
    auto num_tiles = get_arg(args::num_tiles);
    auto start_id = get_arg(args::start_id);

    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    // prepare_reduce_scaler is a template on the CB id; dfb::scaler converts implicitly.
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);

    DataflowBuffer cb_in0(dfb::input);
    auto tensor_accessor = TensorAccessor(ta::input);

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = cb_in0.get_tile_size();

    Noc noc;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_in0.reserve_back(onetile);
        noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
}
