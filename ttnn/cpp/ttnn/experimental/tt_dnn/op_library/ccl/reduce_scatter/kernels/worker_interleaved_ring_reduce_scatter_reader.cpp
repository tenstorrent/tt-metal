// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tuple>

#include "dataflow_api.h"
#include "debug/assert.h"
#include "tensix_types.h"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "ttnn/cpp/ttnn/experimental/tt_dnn/op_library/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/cpp/ttnn/experimental/tt_dnn/op_library/ccl/shared_with_host/hetergeneous_data_structs.hpp"

using tt::tt_metal::ccl::coord_t;
using tt::tt_metal::ccl::WorkerXY;

struct reduce_scatter_reader_common_args_t {
    reduce_scatter_reader_common_args_t(uint32_t& arg_idx) :
        src_addr(get_arg_val<uint32_t>(arg_idx++)),
        num_transfers(get_arg_val<uint32_t>(arg_idx++)),
        full_chunk_num_pages(get_arg_val<uint32_t>(arg_idx++)),
        page_size(get_arg_val<uint32_t>(arg_idx++)),

        my_ring_idx(get_arg_val<uint32_t>(arg_idx++)),
        ring_size(get_arg_val<uint32_t>(arg_idx++)),
        sem_addr(get_arg_val<uint32_t>(arg_idx++)),

        is_clockwise_direction(get_arg_val<uint32_t>(arg_idx++) == 1),
        half_cb_n_pages(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_noc0_core_x(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_noc0_core_y(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_semaphore_address(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_buffer_address(get_arg_val<uint32_t>(arg_idx++)),
        num_concurrent_workers(get_arg_val<uint32_t>(arg_idx++)),

        input_tensor_shape(tt::tt_metal::ccl::coord_from_args(arg_idx)),
        tensor_slice_shape(tt::tt_metal::ccl::coord_from_args(arg_idx)),
        worker_slice_shape(tt::tt_metal::ccl::coord_from_args(arg_idx)),
        worker_slice_offset(tt::tt_metal::ccl::coord_from_args(arg_idx)),
        total_eltwise_kernel_num_pages(get_arg_val<uint32_t>(arg_idx++))
         {
        ASSERT(full_chunk_num_pages > 0);
        ASSERT(page_size > 0);
        ASSERT(ring_size > 0);
        ASSERT(half_cb_n_pages > 0);
    }

    const uint32_t src_addr;
    const uint32_t num_transfers;
    const uint32_t full_chunk_num_pages;
    const uint32_t page_size;
    uint32_t my_ring_idx;
    const uint32_t ring_size;
    const uint32_t sem_addr;

    const bool is_clockwise_direction;

    const uint32_t half_cb_n_pages;
    const uint32_t edm_core_noc0_core_x;
    const uint32_t edm_core_noc0_core_y;
    const uint32_t edm_core_semaphore_address;
    const uint32_t edm_core_buffer_address;
    const uint32_t num_concurrent_workers;

    coord_t input_tensor_shape;
    coord_t tensor_slice_shape;
    coord_t worker_slice_shape;
    coord_t worker_slice_offset;
    uint32_t total_eltwise_kernel_num_pages;
};
#ifdef RM_INTERLEAVED
constexpr bool rm_interleaved_addr_gen_mode = true;
#else
constexpr bool rm_interleaved_addr_gen_mode = false;
#endif

template <bool dram, bool row_major>
struct interleaved_addr_gen_t {
    using type = InterleavedAddrGen<dram>;
};
template <>
struct interleaved_addr_gen_t<false, true> {
    using type = InterleavedAddrGen<false>;
};
template <>
struct interleaved_addr_gen_t<true, true> {
    using type = InterleavedAddrGen<true>;
};
template <>
struct interleaved_addr_gen_t<false, false> {
    using type = InterleavedAddrGenFast<false>;
};
template <>
struct interleaved_addr_gen_t<true, false> {
    using type = InterleavedAddrGenFast<true>;
};

template <bool is_sharded, bool src_is_dram>
struct reduce_scatter_reader_unique_args_t : public reduce_scatter_reader_common_args_t {
    using src_addr_gen_t = typename interleaved_addr_gen_t<src_is_dram, rm_interleaved_addr_gen_mode>::type;

    reduce_scatter_reader_unique_args_t(uint32_t& arg_idx, const DataFormat in0_df) :
        reduce_scatter_reader_common_args_t(arg_idx) {
        this->s = {
            .bank_base_address = this->src_addr,
            .page_size = page_size
#if defined TILE_INTERLEAVED
            ,
            .data_format = in0_df
#endif
        };
    }

    src_addr_gen_t s;

    void dprint() const {
        DPRINT << "RSR args:"
               << "\n\tsrc_addr=" << src_addr << "\n\tnum_transfers=" << num_transfers << "\n\tpage_size=" << page_size
               << "\n\tfull_chunk_num_pages=" << full_chunk_num_pages << "\n\tmy_ring_idx=" << my_ring_idx
               << "\n\tsem_addr=" << sem_addr << "\n\tis_clockwise_direction=" << (uint32_t)is_clockwise_direction
               << "\n\thalf_cb_n_pages=" << half_cb_n_pages << "\n\tring_size=" << ring_size
               << "\n\tedm_core_noc0_core_x=" << edm_core_noc0_core_x
               << "\n\tedm_core_noc0_core_y=" << edm_core_noc0_core_y
               << "\n\tedm_core_semaphore_address=" << edm_core_semaphore_address
               << "\n\tedm_core_buffer_address=" << edm_core_buffer_address << "\n";
    }
};

template <bool src_is_dram>
struct reduce_scatter_reader_unique_args_t<true, src_is_dram> : public reduce_scatter_reader_common_args_t {
    reduce_scatter_reader_unique_args_t(uint32_t& arg_idx, const DataFormat in0_df) :
        reduce_scatter_reader_common_args_t(arg_idx),
        shard_num_pages(get_arg_val<uint32_t>(arg_idx++)),
        num_l1_cores(get_arg_val<uint32_t>(arg_idx++)),
        l1_cores_ptr(reinterpret_cast<WorkerXY*>(get_arg_addr(arg_idx))) {
        arg_idx += this->num_l1_cores;
    }

    const uint32_t shard_num_pages;
    const uint32_t num_l1_cores;
    const WorkerXY* const l1_cores_ptr;

    void dprint() const {}
};

using advance_to_next_transfer_slice_result_t = std::tuple<
    uint32_t,  // ring_index
    uint32_t   // slice_base_page_offset
    >;
template <bool is_sharded>
advance_to_next_transfer_slice_result_t advance_to_next_transfer_slice(
    uint32_t const ring_size,
    uint32_t const curr_ring_idx,
    uint32_t const slice_base_page_offset,
    coord_t const& input_tensor_shape,
    coord_t const& tensor_slice_shape,
    bool const is_clockwise_direction) {
    bool const sliced_only_on_width = tensor_slice_shape.x < input_tensor_shape.x && tensor_slice_shape.y == input_tensor_shape.y;
    uint32_t single_ring_idx_stride =
        sliced_only_on_width ? tensor_slice_shape.x : tensor_slice_shape.y * input_tensor_shape.x;
    uint32_t n_minus_one_ring_indices_stride = sliced_only_on_width
                                                   ? tensor_slice_shape.x * (ring_size - 1)
                                                   : tensor_slice_shape.y * input_tensor_shape.x * (ring_size - 1);

    if constexpr (!is_sharded) {
        if (is_clockwise_direction) {
            if (curr_ring_idx == 0) {
                return advance_to_next_transfer_slice_result_t{
                    ring_size - 1,
                    slice_base_page_offset + n_minus_one_ring_indices_stride,
                };
            } else {
                return advance_to_next_transfer_slice_result_t{
                    curr_ring_idx - 1,
                    slice_base_page_offset - single_ring_idx_stride,
                };
            }
        } else {
            if (curr_ring_idx == ring_size - 1) {
                return advance_to_next_transfer_slice_result_t{
                    0,
                    slice_base_page_offset - n_minus_one_ring_indices_stride,
                };
            } else {
                return advance_to_next_transfer_slice_result_t{
                    curr_ring_idx + 1,
                    slice_base_page_offset + single_ring_idx_stride,
                };
            }
        }
    }
}

void kernel_main() {
    constexpr bool is_sharded = get_compile_time_arg_val(0) == 1;

    // Currently meaningless when `is_sharded=true`
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;

    uint32_t arg_idx = 0;

    constexpr uint32_t to_dm_sender_short_circuit_cb = tt::CB::c_out1;
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
    const DataFormat in0_df = get_dataformat(cb_id_in0);
    auto args = reduce_scatter_reader_unique_args_t<is_sharded, src_is_dram>(arg_idx, in0_df);

    ASSERT(args.half_cb_n_pages >= args.full_chunk_num_pages);

    bool width_sliced = args.tensor_slice_shape.x <= args.input_tensor_shape.x;

    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.sem_addr);
    const uint64_t eth_receiver_l1_base_noc_addr =
        get_noc_addr(args.edm_core_noc0_core_x, args.edm_core_noc0_core_y, args.edm_core_buffer_address);
    const uint64_t eth_receiver_l1_semaphore_noc_addr =
        get_noc_addr(args.edm_core_noc0_core_x, args.edm_core_noc0_core_y, args.edm_core_semaphore_address);

    uint32_t total_cb_pages_pushed = 0;
    uint32_t total_cb_pages_pushed_to_math = 0;

    // For the first timestep, there is no other input to reduce with, so we just send it straight to the input CB
    // of the output data movement kernel - short-circuiting past the (reducer) math kernel
    // For tile => shape in tiles
    // For RM => shape in elements
    uint32_t start_ring_index = args.my_ring_idx;
    while (args.worker_slice_offset.x < args.tensor_slice_shape.x &&
           args.worker_slice_offset.y < args.tensor_slice_shape.y) {
        // Need to reset back to the start ring index because the last iteration of the tranfers read chunks
        // loop won't increment after the last iteration since the increment is within the loop body
        args.my_ring_idx = start_ring_index;
        uint32_t curr_ring_slice_start_page_offset =
            width_sliced ? args.tensor_slice_shape.x * start_ring_index
                         : args.tensor_slice_shape.y * start_ring_index * args.input_tensor_shape.x;

        auto const& next_slice_offset = advance_slice_row_major(
            args.worker_slice_offset, args.worker_slice_shape, args.tensor_slice_shape, args.num_concurrent_workers);
        bool last_slice_of_worker = next_slice_offset.x >= args.tensor_slice_shape.x ||
                                    next_slice_offset.y >= args.tensor_slice_shape.y;

        const uint32_t worker_relative_start_offset_into_slice =
            args.worker_slice_offset.x + (args.worker_slice_offset.y * args.input_tensor_shape.x);
        const uint32_t starting_tile_id = curr_ring_slice_start_page_offset + worker_relative_start_offset_into_slice;
        uint32_t curr_tile_id = starting_tile_id;

        coord_t valid_worker_slice_shape = coord_t(
            std::min(args.worker_slice_shape.x, args.tensor_slice_shape.x - args.worker_slice_offset.x),
            std::min(args.worker_slice_shape.y, args.tensor_slice_shape.y - args.worker_slice_offset.y));

        bool last_page_of_worker = false;
        uint32_t const worker_slice_n_pages = valid_worker_slice_shape.x * valid_worker_slice_shape.y;
        ASSERT(
            (args.num_transfers - 1) * worker_slice_n_pages + total_cb_pages_pushed_to_math <=
            args.total_eltwise_kernel_num_pages);
        {
            coord_t offset_into_worker_slice = {0, 0};
            for (uint32_t p = 0; p < worker_slice_n_pages; p += args.full_chunk_num_pages) {
                uint32_t n_pages = std::min(args.full_chunk_num_pages, worker_slice_n_pages - p);
                ASSERT(!last_page_of_worker);
                read_chunk_from_output_tensor_v2(
                    curr_tile_id,
                    offset_into_worker_slice,
                    valid_worker_slice_shape,
                    // In tiles for tile layout
                    args.input_tensor_shape,
                    to_dm_sender_short_circuit_cb,
                    args.s,
                    n_pages,
                    args.page_size,
                    last_page_of_worker);
                total_cb_pages_pushed += n_pages;
                if (n_pages < args.half_cb_n_pages) {
                    uint32_t num_filler_pages = args.half_cb_n_pages - n_pages;
                    push_filler_pages_to_cb(to_dm_sender_short_circuit_cb, num_filler_pages);
                    ASSERT(args.half_cb_n_pages > n_pages);
                    ASSERT(p + n_pages == worker_slice_n_pages);
                    total_cb_pages_pushed += num_filler_pages;
                }
            }
        }

        for (uint32_t i = 1; i < args.num_transfers; ++i) {
            bool last_transfer = i == args.num_transfers - 1;
            coord_t offset_into_worker_slice = {0, 0};
            std::tie(args.my_ring_idx, curr_ring_slice_start_page_offset) = advance_to_next_transfer_slice<is_sharded>(
                args.ring_size,
                args.my_ring_idx,
                curr_ring_slice_start_page_offset,
                args.input_tensor_shape,
                args.tensor_slice_shape,
                args.is_clockwise_direction);
            ASSERT(last_page_of_worker);
            last_page_of_worker = false;
            curr_tile_id = curr_ring_slice_start_page_offset + worker_relative_start_offset_into_slice;

            for (uint32_t p = 0; p < worker_slice_n_pages; p += args.full_chunk_num_pages) {
                uint32_t n_pages = std::min(args.full_chunk_num_pages, worker_slice_n_pages - p);
                ASSERT(n_pages > 0);
                // Fetch from input tensor
                read_chunk_from_output_tensor_v2(
                    curr_tile_id,
                    offset_into_worker_slice,
                    valid_worker_slice_shape,
                    // In tiles for tile layout
                    args.input_tensor_shape,
                    cb_id_in1,
                    args.s,
                    n_pages,
                    args.page_size,
                    last_page_of_worker);
                uint64_t eth_receiver_l1_curr_noc_addr = eth_receiver_l1_base_noc_addr;

                // Fetch from EDM
                noc_semaphore_wait(receiver_read_semaphore_addr_ptr, 1);
                noc_semaphore_set(receiver_read_semaphore_addr_ptr, 0);
                fetch_chunk(cb_id_in0, n_pages, args.page_size, eth_receiver_l1_base_noc_addr);
                total_cb_pages_pushed_to_math += n_pages;
                total_cb_pages_pushed += n_pages;

                bool last_worker_message_to_edm = last_transfer && last_slice_of_worker && (p + n_pages >= worker_slice_n_pages);
                if (!last_worker_message_to_edm) {
                    noc_semaphore_inc(
                        eth_receiver_l1_semaphore_noc_addr,
                        tt::tt_metal::ccl::EriscDataMoverWorkerSignal::NEXT_MESSAGE_AVAILABLE);
                }
                if (n_pages < args.half_cb_n_pages) {
                    uint32_t num_filler_pages = args.half_cb_n_pages - n_pages;
                    push_filler_pages_to_cb(cb_id_in0, num_filler_pages);
                    push_filler_pages_to_cb(cb_id_in1, num_filler_pages);
                    total_cb_pages_pushed_to_math += num_filler_pages;
                    total_cb_pages_pushed += num_filler_pages;
                }
            }
            ASSERT(last_page_of_worker);
        }

        args.worker_slice_offset = next_slice_offset;
    }

    ASSERT(args.total_eltwise_kernel_num_pages >= total_cb_pages_pushed_to_math);
    DEBUG_STATUS("DRN1");
    // The host code currently doesn't know how to accuractly count the exact number of pages pushed through the
    // math reduce op so it instead provides a known safe lower bound which may be more than actually required by the
    // op. It passes this number to sender and receiver, who will push/pop junk pages to/from the math op to ensure
    // it will complete
    for (; total_cb_pages_pushed_to_math < args.total_eltwise_kernel_num_pages; total_cb_pages_pushed_to_math++) {
        push_filler_pages_to_cb(cb_id_in0, 1);
        push_filler_pages_to_cb(cb_id_in1, 1);
    }

    noc_semaphore_inc(
        eth_receiver_l1_semaphore_noc_addr,
        tt::tt_metal::ccl::EriscDataMoverWorkerSignal::TERMINATE_IMMEDIATELY);
    DEBUG_STATUS("DONE");
}
