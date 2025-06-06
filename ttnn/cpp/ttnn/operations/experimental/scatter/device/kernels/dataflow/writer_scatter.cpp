// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"

namespace {

template <bool is_dram, typename AddrGen>
FORCE_INLINE void write_to_output(
    const uint32_t& cb, const AddrGen& addr_gtor, const uint32_t& stick_size_bytes, const uint32_t& stick_id) {
    cb_wait_front(cb, ONE_PAGE);  // read a whole input row
    const uint64_t destination_noc_address = get_noc_addr(stick_id, addr_gtor);
    const uint32_t l1_read_address = get_read_ptr(cb);

    noc_async_write(destination_noc_address, l1_read_address, stick_size_bytes);
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

    for (uint32_t stick_id = start_stick_id; stick_id < start_stick_id + sticks_for_core; ++stick_id) {
        write_to_output<ctas.output_tensor_is_dram, output_addr_gtor_type>(
            ctas.output_cb, output_addr_gtor, ctas.output_stick_size_bytes, stick_id);
    }
}
