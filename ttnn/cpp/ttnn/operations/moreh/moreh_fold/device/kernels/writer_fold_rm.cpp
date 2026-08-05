// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t output_cb_page_size = get_arg(args::output_cb_page_size);
    const uint32_t start_id = get_arg(args::start_id);
    const uint32_t num_units_per_core = get_arg(args::num_units_per_core);

    constexpr int onetile = 1;

    const auto s = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer output_dfb(dfb::output);

    for (uint32_t i = start_id; i < start_id + num_units_per_core; i++) {
        output_dfb.wait_front(onetile);
        noc.async_write(output_dfb, s, output_cb_page_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        output_dfb.pop_front(onetile);
    }
}
