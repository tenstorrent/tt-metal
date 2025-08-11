
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../isunique_common.hpp"

#include <numeric>

namespace {

template <typename addr_gen_type>
FORCE_INLINE void load_to_cb(
    const uint32_t& cb,
    const addr_gen_type& addr_gtor,
    const uint32_t& offset_bytes,
    const uint32_t& chunk_size_bytes,
    const uint32_t& stick_id) {
    cb_reserve_back(cb, ONE_PAGE);
    const uint64_t source_noc_address = get_noc_addr(stick_id, addr_gtor);
    const uint32_t l1_write_address = get_write_ptr(cb);

    noc_async_read(source_noc_address + offset_bytes, l1_write_address, chunk_size_bytes);
    noc_async_read_barrier();

    cb_push_back(cb, ONE_PAGE);
}

template <typename addr_gen_type>
FORCE_INLINE void write_to_dram(
    const uint32_t& cb,
    const addr_gen_type& addr_gtor,
    const uint32_t& offset_bytes,
    const uint32_t& chunk_size_bytes,
    const uint32_t& stick_id) {}

template <typename input_number_type, typename index_hint_number_type>
FORCE_INLINE void sort_chunk(
    const uint32_t& input_l1_read_addr,
    const uint32_t& index_hint_l1_read_addr,
    const uint32_t& chunk_id,
    const uint32_t& chunk_size) {
    constexpr uint32_t input_pod_size = sizeof(input_number_type);
    volatile tt_l1_ptr input_number_type* input_chunk_begin_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_type*>(input_l1_read_addr);
    volatile tt_l1_ptr input_number_type* input_chunk_end_ptr =
        reinterpret_cast<volatile tt_l1_ptr input_number_type*>(input_l1_read_addr + input_pod_size * chunk_size);
    constexpr uint32_t index_hint_pod_size = sizeof(input_number_type);
    volatile tt_l1_ptr index_hint_number_type* index_hint_chunk_begin_ptr =
        reinterpret_cast<volatile tt_l1_ptr index_hint_number_type*>(index_hint_l1_read_addr);
    volatile tt_l1_ptr index_hint_number_type* index_hint_chunk_end_ptr =
        reinterpret_cast<volatile tt_l1_ptr index_hint_number_type*>(
            index_hint_l1_read_addr + index_hint_pod_size * chunk_size);
    const uint32_t iota_start = chunk_id * chunk_size;
    std::iota(index_hint_chunk_begin_ptr, index_hint_chunk_end_ptr, iota_start);
    std::sort(index_hint_chunk_begin_ptr, index_hint_chunk_end_ptr, [&](const uint32_t& left, const uint32_t& right) {
        return input_chunk_begin_ptr[left] < input_chunk_begin_ptr[right];
    });
    std::sort(input_chunk_begin_ptr, input_chunk_end_ptr);
}

// PHASE 0
template <
    // bool is_input_dram,
    // bool is_index_hint_dram,
    typename input_number_type,
    typename index_hint_number_type,
    typename input_addr_gen_type,
    typename index_hint_addr_gen_type>
FORCE_INLINE void sort_chunks(
    const uint32_t& input_cb,
    const uint32_t& index_hint_cb,
    const uint32_t& core_id,
    const uint32_t& num_chunks,
    const uint32_t& chunk_size,
    const input_addr_gen_type& input_addr_gen,
    const index_hint_addr_gen_type& index_hint_addr_gen) {
    const uint32_t first_chunk_id = ;
    const uint32_t last_chunk_id = ;
    for (uint32_t chunk_id = first_chunk_id; chunk_id < last_chunk_id; ++chunk_id) {
        load_to_cb(input_cb, input_addr_gen, );
        load_to_cb(index_hint_cb, index_hint_addr_gen);
        const uint32_t input_l1_addr = ;
        const uint32_t index_hint_l1_addr = ;
        sort_chunk<input_number_type, index_hint_number_type>(input_l1_addr, index_hint_l1_addr, chunk_id, chunk_size);
        write_to_dram();
        write_to_dram();
    }
}

template <typename input_number_type, typename index_hint_number_type>
FORCE_INLINE void process_pair(
    const uint32_t& input_l1_read_addr, const uint32_t& index_hint_l1_read_addr, const uint32_t& chunk_size) {
    //
}

FORCE_INLINE void hit_barrier() {
    //
}

template <
    // bool is_input_dram,
    // bool is_index_hint_dram,
    typename input_number_type,
    typename index_hint_number_type,
    typename input_addr_gen_type,
    typename index_hint_addr_gen_type>
FORCE_INLINE void process_pairs(
    const uint32_t& core_id,
    const uint32_t& num_chunks,
    const uint32_t& chunk_size,
    const input_addr_gen_type& input_addr_gen,
    const index_hint_addr_gen_type& index_hint_addr_gen) {
    // EulerScheduler scheduler{};
    ChunkIDPair current_chunk_id_pair;
    for () {
        process_pair(chunk_size);
        hit_barrier();
    }
}

template <
    // bool is_input_dram,
    // bool is_index_hint_dram,
    typename input_number_type,
    typename index_hint_number_type,
    typename input_addr_gen_type,
    typename index_hint_addr_gen_type,
    typename output_addr_gen_type>
FORCE_INLINE void recover_ordering(
    const uint32_t& input_cb,
    const uint32_t& index_hint_cb,
    const uint32_t& output_cb,
    const uint32_t& core_id,
    const uint32_t& num_chunks,
    const uint32_t& chunk_size,
    const input_addr_gen_type& input_addr_gen,
    const index_hint_addr_gen_type& index_hint_addr_gen,
    const output_addr_gen_type& output_addr_gen) {
    //
}

}  // namespace

void kernel_main() {
    constexpr auto ctas = get_ctas();

    const uint32_t input_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t index_hint_buffer_address = get_arg_val<uint32_t>(1);
    // const uint32_t first_occurrences_buffer_address = get_arg_val<uint32_t>(2);

    const auto input_addr_gtor = TensorAccessor{ctas.input_args, input_buffer_address, ctas.input_stick_size};
    // const auto input_addr_gtor{
    //     get_interleaved_addr_gen<ctas.input_tensor_is_dram, ctas.is_input_stick_size_bytes_pow2_min_32>(
    //         input_buffer_address, ctas.input_stick_size_bytes, ctas.input_stick_size_bytes_log2)};
    // const auto index_hint_addr_gtor{
    //     get_interleaved_addr_gen<ctas.index_hint_tensor_is_dram, ctas.is_index_hint_stick_size_bytes_pow2_min_32>(
    //         index_hint_buffer_address, ctas.index_hint_stick_size_bytes, ctas.index_hint_stick_size_bytes_log2)};
    // const auto first_occurrence_addr_gtor{
    //     get_interleaved_addr_gen<ctas.first_occurrences_tensor_is_dram,
    //     ctas.is_first_occurrences_stick_size_bytes_pow2_min_32>(
    //         first_occurrences_buffer_address, ctas.first_occurrences_stick_size_bytes,
    //         ctas.first_occurrences_stick_size_bytes_log2)};

    phase0();
    phase1();
    phase2();
}
