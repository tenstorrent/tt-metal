// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"

namespace {

// this function is supposed to write either a whole stick or part of it (76800 elements)
template <bool is_dram, typename AddrGen>
FORCE_INLINE void write_to_output(
    const uint32_t& cb, const AddrGen& addr_gtor, const uint32_t& offset_bytes, const uint32_t& chunk_size_bytes, const uint32_t& stick_id) {
    cb_wait_front(cb, ONE_PAGE);
    const uint64_t destination_noc_address = get_noc_addr(stick_id, addr_gtor);
    const uint32_t l1_read_address = get_read_ptr(cb);

    noc_async_write(l1_read_address, destination_noc_address + offset_bytes, chunk_size_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb, ONE_PAGE);
}

}  // namespace

void kernel_main() {
    constexpr auto ctas{get_ctas()};

    const uint32_t output_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t sticks_for_core = get_arg_val<uint32_t>(2);

    const auto output_addr_gtor{
        get_interleaved_addr_gen<ctas.output_tensor_is_dram, ctas.is_output_stick_size_bytes_pow2_min_32>(
            output_buffer_address, ctas.output_stick_size_bytes, ctas.output_stick_size_bytes_log2)};
    using output_addr_gtor_type = decltype(output_addr_gtor);

    using output_std_type = std_type_t<get_dataformat(ctas.output_cb)>;

    const uint32_t input_and_output_chunk_size = get_arg_val<uint32_t>(3);

    // read sticks (or chunks of them) and write them to output
    for (uint32_t stick_id = start_stick_id; stick_id < start_stick_id + sticks_for_core; ++stick_id) {
        for (uint32_t offset_bytes = 0; offset_bytes < ctas.output_stick_size_bytes; offset_bytes += input_and_output_chunk_size * sizeof(output_std_type)) {
            const uint32_t chunk_write_bytes = std::min(
                ctas.output_stick_size_bytes - offset_bytes, input_and_output_chunk_size * sizeof(output_std_type));
            write_to_output<ctas.output_tensor_is_dram, output_addr_gtor_type>(
                ctas.output_cb, output_addr_gtor, offset_bytes, chunk_write_bytes, stick_id);
        }
    }
}
