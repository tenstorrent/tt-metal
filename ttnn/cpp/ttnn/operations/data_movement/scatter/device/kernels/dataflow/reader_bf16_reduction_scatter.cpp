// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "../scatter_bf16_reduction_common.hpp"
#include "dprint.h"

#include <array>

namespace {

FORCE_INLINE static float bfloat16_to_float(uint16_t bfloat_val) {
    uint32_t uint32_data = ((uint32_t)bfloat_val) << 16;
    float f;
    std::memcpy(&f, &uint32_data, sizeof(f));
    return f;
}

FORCE_INLINE std::uint16_t fp32_to_bf16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));

    std::uint32_t lsb = (bits >> 16) & 1u;
    std::uint32_t rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;

    return static_cast<std::uint16_t>(bits >> 16);
}

FORCE_INLINE float perform_reduction(float input, uint16_t source_value, ScatterReductionType scatter_reduction_type) {
    float fp32_source_value = bfloat16_to_float(source_value);
    switch (scatter_reduction_type) {
        case ScatterReductionType::ADD: {
            return input + fp32_source_value;
        }
        case ScatterReductionType::MULTIPLY: {
            return input * fp32_source_value;
        }
        case ScatterReductionType::AMAX: {
            return std::max(input, fp32_source_value);
        }
        case ScatterReductionType::AMIN: {
            return std::min(input, fp32_source_value);
        }
        case ScatterReductionType::INVALID: {
            return fp32_source_value;
        }
        default: {
            return fp32_source_value;
        }
    }
}

// performs scatter on data loaded to cb with load_to_cb
template <typename index_type>
FORCE_INLINE void scatter_along_chunk(
    const uint32_t& input_cb,
    const uint32_t& index_cb,
    const uint32_t& source_cb,
    const uint32_t& output_cb,
    const uint32_t& fp32_temp_cb,
    const uint32_t& input_stick_size,
    const index_type& input_offset,
    const uint32_t& input_chunk_size,
    const uint32_t& index_chunk_size,
    const ScatterReductionType& scatter_reduction_type = ScatterReductionType::INVALID) {
    const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
    const uint32_t index_l1_read_addr = get_read_ptr(index_cb);
    const uint32_t source_l1_read_addr = get_read_ptr(source_cb);
    const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
    const uint32_t fp32_temp_l1_write_addr = get_write_ptr(fp32_temp_cb);
    volatile tt_l1_ptr uint16_t* input_l1_read_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_read_addr);
    volatile tt_l1_ptr index_type* index_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr index_type*>(index_l1_read_addr);
    volatile tt_l1_ptr uint16_t* source_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(source_l1_read_addr);
    volatile tt_l1_ptr uint16_t* output_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(output_l1_write_addr);
    volatile tt_l1_ptr float* fp32_temp_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr float*>(fp32_temp_l1_write_addr);

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
        volatile uint16_t& source_value = source_l1_read_ptr[index_in_index_chunk];
        const index_type& output_index = index_value - input_offset;
        fp32_temp_l1_write_ptr[output_index] =
            perform_reduction(fp32_temp_l1_write_ptr[output_index], source_value, scatter_reduction_type);
    }
}

// copies source stick to destination stick (first phase of scatter)
FORCE_INLINE void copy_input_to_fp32_temp(uint32_t input_cb, uint32_t fp32_temp_cb, uint32_t input_chunk_size) {
    const uint32_t input_l1_read_addr = get_read_ptr(input_cb);
    const uint32_t fp32_temp_l1_write_addr = get_write_ptr(fp32_temp_cb);
    volatile tt_l1_ptr uint16_t* input_l1_read_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_read_addr);
    volatile tt_l1_ptr float* fp32_temp_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr float*>(fp32_temp_l1_write_addr);
    for (uint32_t index_in_input_chunk = 0; index_in_input_chunk < input_chunk_size; ++index_in_input_chunk) {
        fp32_temp_l1_write_ptr[index_in_input_chunk] = bfloat16_to_float(input_l1_read_ptr[index_in_input_chunk]);
    }
}

FORCE_INLINE void copy_fp32_temp_to_output(uint32_t fp32_temp_cb, uint32_t output_cb, uint32_t chunk_size) {
    const uint32_t fp32_temp_l1_read_addr = get_read_ptr(fp32_temp_cb);
    const uint32_t output_l1_write_addr = get_write_ptr(output_cb);
    volatile tt_l1_ptr float* fp32_temp_l1_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr float*>(fp32_temp_l1_read_addr);
    volatile tt_l1_ptr uint16_t* output_l1_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(output_l1_write_addr);

    for (uint32_t copy_i = 0; copy_i < chunk_size; ++copy_i) {
        output_l1_write_ptr[copy_i] = fp32_to_bf16(fp32_temp_l1_read_ptr[copy_i]);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto ctas{get_ctas()};

    const uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t index_buffer_address = get_arg_val<uint32_t>(1);
    const uint32_t source_buffer_address = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);
    const uint32_t sticks_for_core = get_arg_val<uint32_t>(4);
    // for the outer input/output loop (DRAM accesses per stick: input_row_elem_num / 76800)
    const uint32_t input_and_output_chunk_size = get_arg_val<uint32_t>(5);
    // for the inner index/source loop (DRAM accesses per stick per single input/output loop: index_row_elem_num /
    // 76800)
    const uint32_t index_chunk_size = get_arg_val<uint32_t>(6);
    const uint32_t source_chunk_size = get_arg_val<uint32_t>(7);
    const auto scatter_reduction_type = static_cast<ScatterReductionType>(get_arg_val<uint32_t>(8));

    const auto input_addr_gtor = TensorAccessor(ctas.input_args, input_buffer_address, ctas.input_stick_size_bytes);
    const auto index_addr_gtor = TensorAccessor(ctas.index_args, index_buffer_address, ctas.index_stick_size_bytes);
    const auto source_addr_gtor = TensorAccessor(ctas.source_args, source_buffer_address, ctas.source_stick_size_bytes);

    using input_std_type = std_type_t<get_dataformat(ctas.input_cb)>;
    using index_std_type = std_type_t<get_dataformat(ctas.index_cb)>;

    constexpr uint32_t N = ctas.input_rank - 1;
    // generate 2 stick shape counters
    const auto input_dims{make_shape_array_from_runtime_args<N>(9)};
    const auto index_dims{make_shape_array_from_runtime_args<N>(9 + N)};

    const auto index_strides = make_strides<N>(index_dims);

    std::array<uint32_t, N> coord{from_id<N>(start_stick_id, input_dims)};

    for (uint32_t input_stick_id = start_stick_id; input_stick_id < start_stick_id + sticks_for_core;
         ++input_stick_id) {
        // process input/output chunks sequentially
        for (uint32_t input_offset = 0; input_offset < ctas.input_stick_size;
             input_offset += input_and_output_chunk_size) {
            const uint32_t input_chunk_length =
                std::min(ctas.input_stick_size - input_offset, input_and_output_chunk_size);

            // first phase: copy input data to output
            load_to_cb(
                ctas.input_cb,
                input_addr_gtor,
                input_offset * sizeof(input_std_type),
                input_chunk_length * sizeof(input_std_type),
                input_stick_id);
            cb_wait_front(ctas.input_cb, ONE_PAGE);
            cb_reserve_back(ctas.fp32_temp_cb, ONE_PAGE);

            copy_input_to_fp32_temp(ctas.input_cb, ctas.fp32_temp_cb, input_chunk_length);

            if (in_bounds<N>(coord, index_dims)) {
                const uint32_t index_stick_id = to_id<N>(coord, index_strides);
                // DPRINT << "INSIDE " << index_stick_id << ENDL();
                // second phase: load index and source data chunk-by-chunk and scatter
                for (uint32_t index_offset = 0, source_offset = 0; index_offset < ctas.index_stick_size;
                     index_offset += index_chunk_size, source_offset += source_chunk_size) {
                    // if stick is chunked, the last chunk is usually smaller
                    const uint32_t index_chunk_length =
                        std::min(ctas.index_stick_size - index_offset, index_chunk_size);
                    const uint32_t source_chunk_length =
                        std::min(ctas.source_stick_size - source_offset, source_chunk_size);

                    load_to_cb(
                        ctas.index_cb,
                        index_addr_gtor,
                        index_offset * sizeof(index_std_type),
                        index_chunk_length * sizeof(index_std_type),
                        index_stick_id);
                    // source tensor is sliced beforehand to match index tensor's dimensions, therefore their stick ids
                    // map 1:1
                    load_to_cb(
                        ctas.source_cb,
                        source_addr_gtor,
                        source_offset * sizeof(input_std_type),
                        source_chunk_length * sizeof(input_std_type),
                        index_stick_id);
                    cb_wait_front(ctas.index_cb, ONE_PAGE);
                    cb_wait_front(ctas.source_cb, ONE_PAGE);
                    scatter_along_chunk<index_std_type>(
                        ctas.input_cb,
                        ctas.index_cb,
                        ctas.source_cb,
                        ctas.output_cb,
                        ctas.fp32_temp_cb,
                        ctas.input_stick_size,
                        input_offset,
                        input_chunk_length,
                        index_chunk_length,
                        scatter_reduction_type);
                    cb_pop_front(ctas.source_cb, ONE_PAGE);
                    cb_pop_front(ctas.index_cb, ONE_PAGE);
                }
            }

            cb_pop_front(ctas.input_cb, ONE_PAGE);
            cb_push_back(ctas.fp32_temp_cb, ONE_PAGE);
            cb_wait_front(ctas.fp32_temp_cb, ONE_PAGE);
            cb_reserve_back(ctas.output_cb, ONE_PAGE);

            // third phase: push to the output cb with fp32->bf16 conversion
            copy_fp32_temp_to_output(ctas.fp32_temp_cb, ctas.output_cb, input_chunk_length);
            cb_pop_front(ctas.fp32_temp_cb, ONE_PAGE);
            cb_push_back(ctas.output_cb, ONE_PAGE);
        }
        next_inplace<N>(coord, input_dims);
    }
}
