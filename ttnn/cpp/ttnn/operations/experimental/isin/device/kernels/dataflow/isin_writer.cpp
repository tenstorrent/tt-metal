// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../isin_common.hpp"

namespace {

}

void kernel_main() {
    constexpr auto ctas = get_ctas();

    const uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    const auto output_addr_gtor = TensorAccessor{ctas.output_args, output_buffer_address, ctas.elements_stick_size};

    for (uint32_t output_subchunk_id = 0; output_subchunk_id < ctas; ++output_subchunk_id) {
        const uint32_t output_subchunk_size = std::min(
            ctas.elements_size - output_subchunk_id * ctas.single_fetch_subchunk_size, ctas.single_fetch_subchunk_size);
        const uint32_t output_l1_write_addr = get_write_addr(ctas.output_cb);
        write_to_dram<output_number_type, decltype(output_addr_gtor)>(
            ctas.output_cb, output_addr_gtor, output_subchunk_id, output_subchunk_size);
    }
}
