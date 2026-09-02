// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#include "../common.hpp"

void kernel_main() {
    Noc noc;

    constexpr auto output_stick_size_bytes = get_arg(args::output_stick_size_bytes);

    const auto start_stick_id = get_arg(args::start_stick_id);
    const auto sticks_for_core = get_arg(args::sticks_for_core);

    const auto output_addr_gtor = TensorAccessor(tensor::output);

    using output_std_type = std_type_t<get_dataformat(dfb::output)>;

    const auto input_and_output_chunk_size = get_arg(args::input_and_output_chunk_size);

    // read sticks (or chunks of them) and write them to output
    for (uint32_t stick_id = start_stick_id; stick_id < start_stick_id + sticks_for_core; ++stick_id) {
        for (uint32_t offset_bytes = 0; offset_bytes < output_stick_size_bytes;
             offset_bytes += input_and_output_chunk_size * sizeof(output_std_type)) {
            const uint32_t chunk_write_bytes =
                std::min(output_stick_size_bytes - offset_bytes, input_and_output_chunk_size * sizeof(output_std_type));
            write_to_output(noc, dfb::output, output_addr_gtor, offset_bytes, chunk_write_bytes, stick_id);
        }
    }
}
