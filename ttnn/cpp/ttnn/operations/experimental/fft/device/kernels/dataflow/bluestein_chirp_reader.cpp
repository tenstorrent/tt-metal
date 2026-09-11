// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "bluestein_streaming_common.h"
#include "experimental/kernel_args.h"

namespace {

void zero_lanes(DataflowBuffer& buffer, uint32_t begin) {
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer.get_write_ptr());
    for (uint32_t i = begin; i < bluestein_streaming::chunk_elements; ++i) {
        words[i] = 0;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t base_chunk = get_arg(args::base_chunk);
    const uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr uint32_t N = get_arg(args::chirp_size);
    constexpr bool HAS_IMAG = get_arg(args::has_imag) != 0;

    const auto input_real = TensorAccessor(tensor::input_real);
    const auto input_imag = TensorAccessor(tensor::input_imag);
    const auto chirp_real = TensorAccessor(tensor::chirp_real);
    const auto chirp_imag = TensorAccessor(tensor::chirp_imag);
    DataflowBuffer a_r(dfb::a_r);
    DataflowBuffer a_i(dfb::a_i);
    DataflowBuffer t_r(dfb::t_r);
    DataflowBuffer t_i(dfb::t_i);
    Noc noc;

    for (uint32_t i = 0; i < num_chunks; ++i) {
        const uint32_t chunk = base_chunk + i;
        const uint32_t valid = bluestein_streaming::valid_elements(N, chunk);
        const uint32_t bytes = valid * bluestein_streaming::element_bytes;
        const uint32_t offset = chunk * bluestein_streaming::chunk_elements * bluestein_streaming::element_bytes;
        a_r.reserve_back(1);
        a_i.reserve_back(1);
        t_r.reserve_back(1);
        t_i.reserve_back(1);

        // Every tensor is a single physical row. TensorAccessor supplies the
        // padded page stride; transfers never read past the logical chirp tail.
        // Full-chunk offsets are 4096-byte aligned, including the final tail.
        if (valid != 0) {
            noc.async_read(input_real, a_r, bytes, {.page_id = 0, .offset_bytes = offset}, {});
            if constexpr (HAS_IMAG) {
                noc.async_read(input_imag, a_i, bytes, {.page_id = 0, .offset_bytes = offset}, {});
            }
            noc.async_read(chirp_real, t_r, bytes, {.page_id = 0, .offset_bytes = offset}, {});
            noc.async_read(chirp_imag, t_i, bytes, {.page_id = 0, .offset_bytes = offset}, {});
            noc.async_read_barrier();
        }

        zero_lanes(a_r, valid);
        zero_lanes(a_i, HAS_IMAG ? valid : 0u);
        zero_lanes(t_r, valid);
        zero_lanes(t_i, valid);
        a_r.push_back(1);
        a_i.push_back(1);
        t_r.push_back(1);
        t_i.push_back(1);
    }
}
