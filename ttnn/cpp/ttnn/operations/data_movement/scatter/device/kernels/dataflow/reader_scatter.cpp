// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "../common.hpp"

#include <array>

namespace {

// copies source stick to destination stick (first phase of scatter)
template <typename number_type>
FORCE_INLINE void copy_input_to_output(
    const DataflowBuffer& input_dfb, const DataflowBuffer& output_dfb, const uint32_t& input_chunk_size) {
    const uint32_t input_l1_read_addr = input_dfb.get_read_ptr();
    const uint32_t output_l1_write_addr = output_dfb.get_write_ptr();
    volatile tt_l1_ptr number_type* input_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr number_type* output_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);
    for (uint32_t index_in_input_chunk = 0; index_in_input_chunk < input_chunk_size; ++index_in_input_chunk) {
        output_l1_write_ptr[index_in_input_chunk] = input_l1_read_ptr[index_in_input_chunk];
    }
}

template <typename number_type>
FORCE_INLINE number_type perform_reduction(
    number_type input, number_type source_value, ScatterReductionType scatter_reduction_type, DataFormat data_format) {
    switch (scatter_reduction_type) {
        case ScatterReductionType::ADD: {
            return input + source_value;
        }
        case ScatterReductionType::MULTIPLY: {
            return input * source_value;
        }
        case ScatterReductionType::AMAX: {
            return std::max(input, source_value);
        }
        case ScatterReductionType::AMIN: {
            return std::min(input, source_value);
        }
        case ScatterReductionType::INVALID: {
            return source_value;
        }
        default: {
            return source_value;
        }
    }
}

// performs scatter on data loaded to dfb with load_to_dfb
template <typename number_type, typename index_type>
FORCE_INLINE void scatter_along_chunk(
    const DataflowBuffer& input_dfb,
    const DataflowBuffer& index_dfb,
    const DataflowBuffer& source_dfb,
    const DataflowBuffer& output_dfb,
    const uint32_t& input_stick_size,
    const uint32_t& input_offset,
    const uint32_t& input_chunk_size,
    const uint32_t& index_chunk_size,
    const ScatterReductionType& scatter_reduction_type = ScatterReductionType::INVALID) {
    const uint32_t input_l1_read_addr = input_dfb.get_read_ptr();
    const uint32_t index_l1_read_addr = index_dfb.get_read_ptr();
    const uint32_t source_l1_read_addr = source_dfb.get_read_ptr();
    const uint32_t output_l1_write_addr = output_dfb.get_write_ptr();
    volatile tt_l1_ptr number_type* input_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr index_type* index_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr index_type*>(index_l1_read_addr);
    volatile tt_l1_ptr number_type* source_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(source_l1_read_addr);
    volatile tt_l1_ptr number_type* output_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr number_type*>(output_l1_write_addr);

    // each index from the index chunk is checked whether it points
    // to any of the elements in the current output range (defined by
    // partial stick length and offset)
    for (uint32_t index_in_index_chunk = 0; index_in_index_chunk < index_chunk_size; ++index_in_index_chunk) {
        volatile index_type& index_value = index_l1_read_ptr[index_in_index_chunk];
        if (index_value < input_offset || index_value >= input_offset + input_chunk_size) {
            continue;
        }
        if (index_value >= input_stick_size) {
            continue;
        }
        volatile number_type& source_value = source_l1_read_ptr[index_in_index_chunk];
        const uint32_t& output_index = index_value - input_offset;
        output_l1_write_ptr[output_index] = perform_reduction<number_type>(
            output_l1_write_ptr[output_index], source_value, scatter_reduction_type, input_dfb.get_dataformat());
    }
}

}  // namespace

void kernel_main() {
    Noc noc;

    constexpr auto input_stick_size = get_arg(args::input_stick_size);
    constexpr auto index_stick_size = get_arg(args::index_stick_size);
    constexpr auto source_stick_size = get_arg(args::source_stick_size);
    constexpr auto input_rank = get_arg(args::input_rank);

    const auto start_stick_id = get_arg(args::start_stick_id);
    const auto sticks_for_core = get_arg(args::sticks_for_core);
    // for the outer input/output loop (DRAM accesses per stick: input_row_elem_num / 76800)
    const auto input_and_output_chunk_size = get_arg(args::input_and_output_chunk_size);
    // for the inner index/source loop (DRAM accesses per stick per single input/output loop: index_row_elem_num /
    // 76800)
    const auto index_chunk_size = get_arg(args::index_chunk_size);
    const auto source_chunk_size = get_arg(args::source_chunk_size);
    const auto scatter_reduction_type = static_cast<ScatterReductionType>(get_arg(args::scatter_reduction_type));

    const auto input_addr_gtor = TensorAccessor(tensor::input);
    const auto index_addr_gtor = TensorAccessor(tensor::index);
    const auto source_addr_gtor = TensorAccessor(tensor::source);

    using input_std_type = std_type_t<get_dataformat(dfb::input)>;
    using index_std_type = std_type_t<get_dataformat(dfb::index)>;

    constexpr uint32_t N = input_rank - 1;
    // generate 2 stick shape counters
    const auto input_dims{make_shape_array_from_runtime_args<N>(0)};
    const auto index_dims{make_shape_array_from_runtime_args<N>(N)};

    const auto index_strides = make_strides<N>(index_dims);

    std::array<uint32_t, N> coord{from_id<N>(start_stick_id, input_dims)};

    DataflowBuffer input_dfb(dfb::input);
    DataflowBuffer output_dfb(dfb::output);
    DataflowBuffer index_dfb(dfb::index);
    DataflowBuffer source_dfb(dfb::source);

    for (uint32_t input_stick_id = start_stick_id; input_stick_id < start_stick_id + sticks_for_core;
         ++input_stick_id) {
        // process input/output chunks sequentially
        for (uint32_t input_offset = 0; input_offset < input_stick_size; input_offset += input_and_output_chunk_size) {
            const uint32_t input_chunk_length = std::min(input_stick_size - input_offset, input_and_output_chunk_size);

            // first phase: copy input data to output
            load_to_dfb(
                noc,
                dfb::input,
                input_addr_gtor,
                input_offset * sizeof(input_std_type),
                input_chunk_length * sizeof(input_std_type),
                input_stick_id);
            input_dfb.wait_front(ONE_PAGE);
            output_dfb.reserve_back(ONE_PAGE);

            copy_input_to_output<input_std_type>(input_dfb, output_dfb, input_chunk_length);

            if (in_bounds<N>(coord, index_dims)) {
                const uint32_t index_stick_id = to_id<N>(coord, index_strides);
                // second phase: load index and source data chunk-by-chunk and scatter
                for (uint32_t index_offset = 0, source_offset = 0; index_offset < index_stick_size;
                     index_offset += index_chunk_size, source_offset += source_chunk_size) {
                    // if stick is chunked, the last chunk is usually smaller
                    const uint32_t index_chunk_length = std::min(index_stick_size - index_offset, index_chunk_size);
                    const uint32_t source_chunk_length = std::min(source_stick_size - source_offset, source_chunk_size);

                    load_to_dfb(
                        noc,
                        dfb::index,
                        index_addr_gtor,
                        index_offset * sizeof(index_std_type),
                        index_chunk_length * sizeof(index_std_type),
                        index_stick_id);
                    // source tensor is sliced beforehand to match index tensor's dimensions, therefore their stick ids
                    // map 1:1
                    load_to_dfb(
                        noc,
                        dfb::source,
                        source_addr_gtor,
                        source_offset * sizeof(input_std_type),
                        source_chunk_length * sizeof(input_std_type),
                        index_stick_id);
                    index_dfb.wait_front(ONE_PAGE);
                    source_dfb.wait_front(ONE_PAGE);
                    scatter_along_chunk<input_std_type, index_std_type>(
                        input_dfb,
                        index_dfb,
                        source_dfb,
                        output_dfb,
                        input_stick_size,
                        input_offset,
                        input_chunk_length,
                        index_chunk_length,
                        scatter_reduction_type);
                    source_dfb.pop_front(ONE_PAGE);
                    index_dfb.pop_front(ONE_PAGE);
                }
            }

            // third phase: push to the output dfb
            output_dfb.push_back(ONE_PAGE);
            input_dfb.pop_front(ONE_PAGE);
        }
        next_inplace<N>(coord, input_dims);
    }
}
