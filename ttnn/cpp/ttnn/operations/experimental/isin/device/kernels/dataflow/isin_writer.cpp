// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include "isin_common.hpp"

void kernel_main() {
    constexpr auto ctas = get_ctas();

    const uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t subchunks_per_core = get_arg_val<uint32_t>(1);
    const uint32_t subchunks_offset = get_arg_val<uint32_t>(2);
    const auto output_addr_gtor = TensorAccessor{ctas.output_accessor_args, output_buffer_address, ctas.elements_size};

    constexpr uint32_t output_element_size = ctas.elements_tensor_datum_size;

    /*
        this loop goes over the same range as the reader with its elements subchunk loop
        elements' and output's values have a 1-to-1 correspondence
    */
    for (uint32_t output_subchunk_id = subchunks_offset,
                  output_offset = subchunks_offset * ctas.single_fetch_subchunk_size;
         output_offset < ((subchunks_offset + subchunks_per_core) * ctas.single_fetch_subchunk_size);
         ++output_subchunk_id, output_offset += ctas.single_fetch_subchunk_size) {
        const uint32_t output_subchunk_size =
            std::min(ctas.elements_size - output_offset, ctas.single_fetch_subchunk_size);

        write_to_dram(ctas.output_cb, output_addr_gtor, output_offset, output_subchunk_size, output_element_size);
    }
}
