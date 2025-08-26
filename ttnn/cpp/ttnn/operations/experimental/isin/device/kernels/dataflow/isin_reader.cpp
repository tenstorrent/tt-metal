
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../isin_common.hpp"

#include <algorithm>
#include <numeric>

namespace {

// template <typename elements_number_type, DataType dt>
// FORCE_INLINE int32_t compare(const elements_number_type& left, const elements_number_type& right) {
//     return left - right;
// }

// template <>
// FORCE_INLINE int32_t compare<uint16_t, DataType::BFLOAT16>(const uint16_t& left, const uint16_t& right) {
//     const float float_left = *static_cast<float*>(static_cast<uint32_t>(left) << (8 * sizeof(uint16_t)));
//     const float float_right = *static_cast<float*>(static_cast<uint32_t>(right) << (8 * sizeof(uint16_t)));
//     return compare<float, DataType::FLOAT>(float_left, float_right);
// }

// template <typename elements_number_type>
// FORCE_INLINE void apply_inverse_tagged(elements_number_type* a, std::size_t n, index_hint_number_type* p) {
//     for (std::size_t i = 0; i < n; ++i) {
//         if (p[i] >= n) {
//             continue;  // visited
//         }
//         std::size_t j = i;
//         elements_number_type tmp = std::move(a[i]);  // hold dest slot's future value
//         while (p[j] < n) {                           // follow cycle
//             std::size_t src = p[j];
//             std::swap(tmp, a[src]);  // pull value from source
//             p[j] += n;               // mark visited
//             j = src;
//         }
//         a[i] = std::move(tmp);  // place last carried value
//     }
//     for (std::size_t i = 0; i < n; ++i) {
//         p[i] -= n;  // untag
//     }
// }

// template <typename elements_number_type>
// FORCE_INLINE void apply_direct_tagged(elements_number_type* a, std::size_t n, index_hint_number_type* p) {
//     for (std::size_t i = 0; i < n; ++i) {
//         if (p[i] >= n || p[i] == i) {  // visited or fixed
//             if (p[i] < n && p[i] == i) {
//                 p[i] += n;
//             }
//             continue;
//         }
//         std::size_t j = i;
//         elements_number_type tmp = std::move(a[i]);
//         while (p[j] < n && p[j] != i) {
//             std::size_t to = p[j];
//             std::swap(tmp, a[to]);  // push value forward
//             index_hint_number_type nxt = p[j];
//             p[j] += n;  // mark visited
//             j = nxt;
//         }
//         a[i] = std::move(tmp);
//         p[j] += n;  // close the cycle
//     }
//     for (std::size_t i = 0; i < n; ++i) {
//         p[i] -= n;  // untag
//     }
// }

// template <typename test_elements_number_type, DataType dt>
// FORCE_INLINE void sort_chunk(const uint32_t& test_elements_l1_read_addr, const uint32_t& subchunk_size) {
//     constexpr uint32_t test_elements_pod_size = sizeof(test_elements_number_type);
//     volatile tt_l1_ptr test_elements_number_type* test_elements_chunk_begin_ptr =
//         reinterpret_cast<volatile tt_l1_ptr test_elements_number_type*>(test_elements_l1_read_addr);
//     volatile tt_l1_ptr test_elements_number_type* test_elements_chunk_end_ptr =
//         reinterpret_cast<volatile tt_l1_ptr test_elements_number_type*>(
//             test_elements_l1_read_addr + test_elements_pod_size * subchunk_size);
//     std::sort(test_elements_chunk_begin_ptr, test_elements_chunk_end_ptr);
// }

// template <typename elements_number_type, DataType dt>
// FORCE_INLINE void sort_chunk(
//     const uint32_t& elements_l1_read_addr, const uint32_t& index_hint_l1_read_addr, const uint32_t& subchunk_size) {
//     constexpr uint32_t elements_pod_size = sizeof(elements_number_type);
//     volatile tt_l1_ptr elements_number_type* elements_chunk_begin_ptr =
//         reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(elements_l1_read_addr);
//     volatile tt_l1_ptr elements_number_type* elements_chunk_end_ptr =
//         reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(
//             elements_l1_read_addr + elements_pod_size * subchunk_size);
//     constexpr uint32_t index_hint_pod_size = sizeof(elements_number_type);
//     volatile tt_l1_ptr index_hint_number_type* index_hint_chunk_begin_ptr =
//         reinterpret_cast<volatile tt_l1_ptr index_hint_number_type*>(index_hint_l1_read_addr);
//     volatile tt_l1_ptr index_hint_number_type* index_hint_chunk_end_ptr =
//         reinterpret_cast<volatile tt_l1_ptr index_hint_number_type*>(
//             index_hint_l1_read_addr + index_hint_pod_size * subchunk_size);
//     std::iota(index_hint_chunk_begin_ptr, index_hint_chunk_end_ptr, 0);
//     std::sort(index_hint_chunk_begin_ptr, index_hint_chunk_end_ptr, [&](const uint32_t& left, const uint32_t& right)
//     {
//         return compare<elements_number_type, dt>(elements_chunk_begin_ptr[left], elements_chunk_begin_ptr[right]);
//     });
//     apply_inverse_tagged(elements_chunk_begin_ptr, subchunk_size, index_hint_chunk_begin_ptr);
// }

template <typename elements_number_type>
FORCE_INLINE void isin_subchunks(
    const uint32_t& elements_l1_read_addr,
    const uint32_t& test_elements_l1_read_addr,
    const uint32_t& output_l1_write_addr,
    const uint32_t& elements_subchunk_size,
    const uint32_t& test_elements_subchunk_size) {
    volatile tt_l1_ptr elements_number_type* elements_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(elements_l1_read_addr);
    volatile tt_l1_ptr elements_number_type* test_elements_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr elements_number_type*>(test_elements_l1_read_addr);
    volatile tt_l1_ptr output_number_type* output_subchunk_ptr =
        reinterpret_cast<volatile tt_l1_ptr output_number_type*>(output_l1_write_addr);
    for (uint32_t elements_index = 0; elements_index < elements_subchunk_size; ++elements_index) {
        for (uint32_t test_elements_index = 0; test_elements_index < test_elements_subchunk_size;
             ++test_elements_index) {
            if (elements_subchunk_ptr[elements_index] == test_elements_subchunk_ptr[test_elements_index]) {
                output_subchunk_ptr[elements_index] = 0x1;
                break;
            }
        }
    }
}

// FORCE_INLINE void recover_ordering(
//     const uint32_t& output_l1_addr,
//     const uint32_t& index_hint_l1_read_addr,
//     const uint32_t& output_l1_write_addr,
//     const uint32_t& subchunk_size) {
//     constexpr uint32_t output_pod_size = sizeof(output_number_type);
//     volatile tt_l1_ptr output_number_type* output_chunk_begin_ptr =
//         reinterpret_cast<volatile tt_l1_ptr output_number_type*>(output_l1_addr);
//     volatile tt_l1_ptr output_number_type* output_chunk_end_ptr =
//         reinterpret_cast<volatile tt_l1_ptr output_number_type*>(output_l1_addr + output_pod_size * subchunk_size);
//     constexpr uint32_t index_hint_pod_size = sizeof(index_hint_number_type);
//     volatile tt_l1_ptr index_hint_number_type* index_hint_chunk_begin_ptr =
//         reinterpret_cast<volatile tt_l1_ptr index_hint_number_type*>(index_hint_l1_read_addr);
//     volatile tt_l1_ptr index_hint_number_type* index_hint_chunk_end_ptr =
//         reinterpret_cast<volatile tt_l1_ptr index_hint_number_type*>(
//             index_hint_l1_read_addr + index_hint_pod_size * subchunk_size);

//     apply_direct_tagged(output_chunk_begin_ptr, subchunk_size, index_hint_chunk_begin_ptr);
// }

FORCE_INLINE void prefill_output(const uint32_t& output_l1_write_addr, const uint32_t& output_subchunk_size) {
    volatile tt_l1_ptr output_number_type* output_chunk_begin_ptr =
        reinterpret_cast<volatile tt_l1_ptr output_number_type*>(output_l1_write_addr);
    for (uint32_t i = 0; i < output_subchunk_size; ++i) {
        output_chunk_begin_ptr[i] = 0x0;
    }
}

}  // namespace

void kernel_main() {
    constexpr auto ctas = get_ctas();

    const uint32_t elements_buffer_address = get_arg_val<uint32_t>(0);
    const uint32_t test_elements_buffer_address = get_arg_val<uint32_t>(1);

    constexpr auto elements_dataformat = get_dataformat(ctas.elements_cb);
    using elements_number_type = std_type_t<elements_dataformat>;

    const auto elements_addr_gtor =
        TensorAccessor{ctas.elements_accessor_args, elements_buffer_address, ctas.elements_size};
    const auto test_elements_addr_gtor =
        TensorAccessor{ctas.test_elements_accessor_args, test_elements_buffer_address, ctas.test_elements_size};

    constexpr uint32_t elements_start_subchunk_id = 0;
    constexpr uint32_t test_elements_start_subchunk_id = 0;

    for (uint32_t elements_subchunk_id = elements_start_subchunk_id, elements_offset = 0;
         elements_offset < ctas.elements_size;
         ++elements_subchunk_id, elements_offset += ctas.single_fetch_subchunk_size) {
        const uint32_t elements_subchunk_size =
            std::min(ctas.elements_size - elements_offset, ctas.single_fetch_subchunk_size);
        load_to_cb<elements_number_type, decltype(elements_addr_gtor)>(
            ctas.elements_cb, elements_addr_gtor, elements_subchunk_id, elements_subchunk_size);
        cb_wait_front(ctas.elements_cb, ONE_PAGE);
        cb_reserve_back(ctas.output_cb, ONE_PAGE);
        const uint32_t elements_l1_read_addr = get_read_ptr(ctas.elements_cb);
        const uint32_t output_l1_write_addr = get_write_ptr(ctas.output_cb);
        prefill_output(output_l1_write_addr, elements_subchunk_size);

        for (uint32_t test_elements_subchunk_id = test_elements_start_subchunk_id, test_elements_offset = 0;
             test_elements_offset < ctas.test_elements_size;
             ++test_elements_subchunk_id, test_elements_offset += ctas.single_fetch_subchunk_size) {
            DPRINT << test_elements_subchunk_id << ENDL();
            const uint32_t test_elements_subchunk_size =
                std::min(ctas.test_elements_size - test_elements_offset, ctas.single_fetch_subchunk_size);
            load_to_cb<elements_number_type, decltype(test_elements_addr_gtor)>(
                ctas.test_elements_cb, test_elements_addr_gtor, test_elements_offset, test_elements_subchunk_size);
            cb_wait_front(ctas.test_elements_cb, ONE_PAGE);
            const uint32_t test_elements_l1_read_addr = get_read_ptr(ctas.test_elements_cb);

            isin_subchunks<elements_number_type>(
                elements_l1_read_addr,
                test_elements_l1_read_addr,
                output_l1_write_addr,
                elements_subchunk_size,
                test_elements_subchunk_size);

            cb_pop_front(ctas.test_elements_cb, ONE_PAGE);
        }

        cb_push_back(ctas.output_cb, ONE_PAGE);
        cb_pop_front(ctas.elements_cb, ONE_PAGE);
    }
}
