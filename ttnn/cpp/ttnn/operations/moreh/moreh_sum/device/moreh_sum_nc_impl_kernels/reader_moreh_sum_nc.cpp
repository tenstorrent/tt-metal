// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline uint32_t get_read_tile_id(uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size) {
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

void kernel_main() {
    // runtime args
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);
    const auto start_id = get_arg(args::start_id);
    const auto dim = get_arg(args::dim);
    const auto reduce_tile_size = get_arg(args::reduce_tile_size);
    const auto inner_tile_size = get_arg(args::inner_tile_size);

    constexpr uint32_t onetile = 1;

#ifdef USE_FPU
    // Only the float factory accumulates on the FPU and needs a zero tile staged for it; the int32
    // factory binds no zero buffer, so this block is preprocessed away there.
    dataflow_kernel_lib::prepare_zero_tile<dfb::zero>();
#endif

    const auto input_addrg = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer dfb_in0_obj(dfb::input);
    const auto in0_tile_bytes = dfb_in0_obj.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        auto read_tile_id = (dim == 0) ? (i) : (get_read_tile_id(i, reduce_tile_size, inner_tile_size));
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            dfb_in0_obj.reserve_back(onetile);
            noc.async_read(input_addrg, dfb_in0_obj, in0_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0_obj.push_back(onetile);
            read_tile_id += inner_tile_size;
        }
    }
}
