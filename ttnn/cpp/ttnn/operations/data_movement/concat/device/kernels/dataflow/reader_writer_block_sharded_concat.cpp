// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tuple>
#include <utility>
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/concat/device/kernels/dataflow/concat_reader_common.hpp"

// Dual-RISC block-sharded concat kernel.
// Used as both reader (NCRISC/NOC0) and writer (BRISC/NOC1) with the host
// splitting copies between the two to double NOC bandwidth.
//
// Each copy moves a rectangular region: num_rows rows of copy_size bytes, with independent source
// and destination strides. The source is a shard of one input, usually on another core, named by
// its shard id; the destination is this core's own output shard. The per-copy arrays are sized to
// the busiest core and padded, so only the first num_copies entries are read.
void kernel_main() {
    const uint32_t num_copies = get_arg(args::num_copies);

    // The tensor binding sequence carries its own length, so the host passes no tensor count.
    constexpr uint32_t num_inputs = std::tuple_size_v<decltype(tensor::inputs)>;
    constexpr auto input_indices = std::make_integer_sequence<uint32_t, num_inputs>();

    Noc noc;
    const LocalTensorAccessor<uint32_t> output(tensor::output);

    for (uint32_t c = 0; c < num_copies; c++) {
        const uint32_t src_offset = get_arg(args::src_offset)[c];
        const uint32_t src_stride = get_arg(args::src_stride)[c];
        const uint32_t dst_offset = get_arg(args::dst_offset)[c];
        const uint32_t dst_stride = get_arg(args::dst_stride)[c];
        const uint32_t copy_size = get_arg(args::copy_size)[c];
        const uint32_t num_rows = get_arg(args::num_rows)[c];
        const uint32_t shard_id = get_arg(args::shard_id)[c];

        // The input index is only known at run time; with_input runs the body for the matching
        // compile-time index. The accessor is built inside, so only one input's is on the stack.
        concat_reader::with_input(get_arg(args::input_index)[c], input_indices, [&](auto I) {
            const auto input = TensorAccessor(std::get<decltype(I)::value>(tensor::inputs));
            // Looked up once per copy, then each row only adds an offset. Valid only on this
            // RISC's NoC, which is why the lookup takes it.
            const auto src = ShardView(input).shard(shard_id, noc.get_noc_id());
            uint32_t src_row_offset = src_offset;
            uint32_t dst_row_offset = dst_offset;
            for (uint32_t row = 0; row < num_rows; row++) {
                noc.async_read(
                    src, output, copy_size, {.offset_bytes = src_row_offset}, {.offset_bytes = dst_row_offset});
                src_row_offset += src_stride;
                dst_row_offset += dst_stride;
            }
        });
    }

    noc.async_read_barrier();
}
