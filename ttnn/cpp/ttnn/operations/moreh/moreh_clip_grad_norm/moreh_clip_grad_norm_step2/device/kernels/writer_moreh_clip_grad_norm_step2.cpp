// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_output = 16;

    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(output_args, output_addr);

    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb_output(cb_id_output);
    const auto output_tile_bytes = get_tile_size(cb_id_output);

    cb_output.wait_front(onetile);
    noc.async_write(cb_output, s, output_tile_bytes, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    cb_output.pop_front(onetile);

}  // void kernel_main()
