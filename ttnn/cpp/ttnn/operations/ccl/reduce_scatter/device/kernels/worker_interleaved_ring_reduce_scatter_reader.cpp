// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "dataflow_api.h"
#include "debug/assert.h"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"

using ttnn::ccl::coord_t;
using ttnn::ccl::WorkerXY;
using tt::tt_metal::TensorMemoryLayout;

struct reduce_scatter_reader_common_args_t {
    reduce_scatter_reader_common_args_t(std::size_t& arg_idx, DataFormat in0_df) :
        input_tensor_addr(get_arg_val<uint32_t>(arg_idx++)),
        output_tensor_addr(get_arg_val<uint32_t>(arg_idx++)),
        num_transfers(get_arg_val<uint32_t>(arg_idx++)),
        full_chunk_num_pages(get_arg_val<uint32_t>(arg_idx++)),
        page_size(get_arg_val<uint32_t>(arg_idx++)),

        my_ring_idx(get_arg_val<uint32_t>(arg_idx++)),
        ring_size(get_arg_val<uint32_t>(arg_idx++)),
        sem_addr(get_semaphore(get_arg_val<uint32_t>(arg_idx++))),

        is_clockwise_direction(get_arg_val<uint32_t>(arg_idx++) == 1),
        half_cb_n_pages(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_noc0_core_x(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_noc0_core_y(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_semaphore_address(get_arg_val<uint32_t>(arg_idx++)),
        edm_core_buffer_address(get_arg_val<uint32_t>(arg_idx++)),
        num_concurrent_workers(get_arg_val<uint32_t>(arg_idx++)),

        input_tensor_shape(ttnn::ccl::coord_from_args(arg_idx)),
        tensor_slice_shape(ttnn::ccl::coord_from_args(arg_idx)),
        worker_slice_shape(ttnn::ccl::coord_from_args(arg_idx)),
        worker_slice_offset(ttnn::ccl::coord_from_args(arg_idx)),
        total_eltwise_kernel_num_pages(get_arg_val<uint32_t>(arg_idx++)),
        requires_last_input_from_other_sender(get_arg_val<uint32_t>(arg_idx++)),
        in0_df(in0_df)
        {
        ASSERT(full_chunk_num_pages > 0);
        ASSERT(page_size > 0);
        ASSERT(ring_size > 0);
        ASSERT(half_cb_n_pages > 0);
    }

    const uint32_t input_tensor_addr;
    const uint32_t output_tensor_addr;
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
    bool requires_last_input_from_other_sender;
    DataFormat in0_df;
};
#ifdef ROW_MAJOR_LAYOUT
constexpr bool row_major_layout = true;
#else
constexpr bool row_major_layout = false;
#endif

template <TensorMemoryLayout layout, bool src_is_dram> struct source_tensor_addrgen {
#ifdef ROW_MAJOR_LAYOUT
    using type = InterleavedAddrGen<src_is_dram>;
#else
    using type = InterleavedAddrGenFast<src_is_dram>;
#endif
};
template <bool src_is_dram> struct source_tensor_addrgen<TensorMemoryLayout::WIDTH_SHARDED, src_is_dram> { using type = tt::tt_metal::address_generators::DefaultWidthShardedAddressGenerator; };
template <bool src_is_dram> struct source_tensor_addrgen<TensorMemoryLayout::HEIGHT_SHARDED, src_is_dram> { using type = tt::tt_metal::address_generators::DefaultHeightShardedAddressGenerator; };
template <bool src_is_dram> struct source_tensor_addrgen<TensorMemoryLayout::BLOCK_SHARDED, src_is_dram> { using type = tt::tt_metal::address_generators::DefaultBlockShardedAddressGenerator; };


constexpr bool is_sharded = get_compile_time_arg_val(0) == 1;

// Currently meaningless when `is_sharded=true`
constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;
static constexpr tt::tt_metal::TensorMemoryLayout input_tensor_memory_layout =
    static_cast<tt::tt_metal::TensorMemoryLayout>(get_compile_time_arg_val(2));

constexpr bool dest_is_dram = get_compile_time_arg_val(3) == 1;
static constexpr tt::tt_metal::TensorMemoryLayout output_tensor_memory_layout =
    static_cast<tt::tt_metal::TensorMemoryLayout>(get_compile_time_arg_val(4));

constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(5);
/*
 * Readback accumulation is a mode that may be enabled for line reductions. This
 * option tells the worker that it must perform a second pass of the data and
 * and emit to the output tensor again:
 * The first pass would be a partial accumulated for a given direction (in a line reduce-scatter)
 * and the second one would be to accumulate that  b
 */
constexpr bool is_line_reduce_scatter = get_compile_time_arg_val(6) != 0;
// TODO: clean this up
#ifdef SHARDED_MEM_LAYOUT
static constexpr bool is_sharded_mode = true;
static constexpr uint32_t input_tensor_shard_grid_height = get_compile_time_arg_val(7);
static constexpr uint32_t input_tensor_shard_grid_width = get_compile_time_arg_val(8);
static constexpr uint32_t input_tensor_shard_grid_start_y_logical = get_compile_time_arg_val(9);
static constexpr uint32_t input_tensor_shard_grid_start_x_logical = get_compile_time_arg_val(10);
static constexpr uint32_t input_tensor_shard_pages_per_shard_y = get_compile_time_arg_val(11);
static constexpr uint32_t input_tensor_shard_pages_per_shard_x = get_compile_time_arg_val(12);
static constexpr bool input_tensor_shard_grid_transposed = get_compile_time_arg_val(13) != 0;


static constexpr uint32_t output_tensor_shard_grid_height = get_compile_time_arg_val(14);
static constexpr uint32_t output_tensor_shard_grid_width = get_compile_time_arg_val(15);
static constexpr uint32_t output_tensor_shard_grid_start_y_logical = get_compile_time_arg_val(16);
static constexpr uint32_t output_tensor_shard_grid_start_x_logical = get_compile_time_arg_val(17);
static constexpr uint32_t output_tensor_shard_pages_per_shard_y = get_compile_time_arg_val(18);
static constexpr uint32_t output_tensor_shard_pages_per_shard_x = get_compile_time_arg_val(19);
static constexpr bool output_tensor_shard_grid_transposed = get_compile_time_arg_val(20) != 0;

#else
static constexpr bool is_sharded_mode = false;
static constexpr uint32_t input_tensor_shard_grid_height = 0;
static constexpr uint32_t input_tensor_shard_grid_width = 0;
static constexpr uint32_t input_tensor_shard_grid_start_y_logical = 0;
static constexpr uint32_t input_tensor_shard_grid_start_x_logical = 0;
static constexpr uint32_t input_tensor_shard_pages_per_shard_y = 0;
static constexpr uint32_t input_tensor_shard_pages_per_shard_x = 0;
static constexpr bool input_tensor_shard_grid_transposed = 0;

static constexpr uint32_t output_tensor_shard_grid_height = 0;
static constexpr uint32_t output_tensor_shard_grid_width = 0;
static constexpr uint32_t output_tensor_shard_grid_start_y_logical = 0;
static constexpr uint32_t output_tensor_shard_grid_start_x_logical = 0;
static constexpr uint32_t output_tensor_shard_pages_per_shard_y = 0;
static constexpr uint32_t output_tensor_shard_pages_per_shard_x = 0;
static constexpr bool output_tensor_shard_grid_transposed = 0;
#endif

struct sharded_tensor_ct_args_t {
    const uint32_t tensor_shard_grid_height;
    const uint32_t tensor_shard_grid_width;
    const uint32_t tensor_shard_grid_start_y_logical;
    const uint32_t tensor_shard_grid_start_x_logical;
    const uint32_t tensor_shard_pages_per_shard_y;
    const uint32_t tensor_shard_pages_per_shard_x;
    const bool tensor_shard_grid_transposed;
};

static constexpr auto input_sharded_tensor_ct_args = sharded_tensor_ct_args_t{
    input_tensor_shard_grid_height,
    input_tensor_shard_grid_width,
    input_tensor_shard_grid_start_y_logical,
    input_tensor_shard_grid_start_x_logical,
    input_tensor_shard_pages_per_shard_y,
    input_tensor_shard_pages_per_shard_x,
    input_tensor_shard_grid_transposed
};

static constexpr auto output_sharded_tensor_ct_args = sharded_tensor_ct_args_t{
    output_tensor_shard_grid_height,
    output_tensor_shard_grid_width,
    output_tensor_shard_grid_start_y_logical,
    output_tensor_shard_grid_start_x_logical,
    output_tensor_shard_pages_per_shard_y,
    output_tensor_shard_pages_per_shard_x,
    output_tensor_shard_grid_transposed
};

struct sharded_tensor_rt_args_t {
    uint32_t shard_grid_nrows;
    const uint32_t* shard_grid_row_map;
    uint32_t shard_grid_ncols;
    const uint32_t* shard_grid_col_map;

    static sharded_tensor_rt_args_t from_args(std::size_t& arg_idx) {
        sharded_tensor_rt_args_t args;
        args.shard_grid_nrows = get_arg_val<uint32_t>(arg_idx++);
        args.shard_grid_row_map = reinterpret_cast<const uint32_t*>(get_arg_addr(arg_idx));
        arg_idx += args.shard_grid_nrows;
        args.shard_grid_ncols = get_arg_val<uint32_t>(arg_idx++);
        args.shard_grid_col_map = reinterpret_cast<const uint32_t*>(get_arg_addr(arg_idx));
        arg_idx += args.shard_grid_ncols;
        return args;
    }
};



template <tt::tt_metal::TensorMemoryLayout tensor_memory_layout, bool tensor_is_dram>
auto build_sharded_tensor_address_generator(sharded_tensor_ct_args_t const& ct_args, sharded_tensor_rt_args_t const &sharded_tensor_rt_args, reduce_scatter_reader_common_args_t const& args, uint32_t tensor_base_addr) -> typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type {
    ASSERT(is_sharded_mode);
    return typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type(
        tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(
            sharded_tensor_rt_args.shard_grid_nrows, sharded_tensor_rt_args.shard_grid_row_map, sharded_tensor_rt_args.shard_grid_ncols, sharded_tensor_rt_args.shard_grid_col_map),
        typename tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<tensor_memory_layout>::type(
            ct_args.tensor_shard_pages_per_shard_y,
            ct_args.tensor_shard_pages_per_shard_x,
            ct_args.tensor_shard_grid_height,
            ct_args.tensor_shard_grid_width,
            ct_args.tensor_shard_grid_start_y_logical,
            ct_args.tensor_shard_grid_start_x_logical,
            ct_args.tensor_shard_grid_transposed),
        args.page_size,
        tensor_base_addr);
}

template <tt::tt_metal::TensorMemoryLayout tensor_memory_layout, bool tensor_is_dram>
auto build_source_address_generator(std::size_t &arg_idx, reduce_scatter_reader_common_args_t const& args, uint32_t tensor_base_addr) -> typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type {
    if constexpr (tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        if constexpr (row_major_layout) {
            return typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type{tensor_base_addr, args.page_size};
        } else {
            return typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type{tensor_base_addr, args.page_size, args.in0_df};
        }
    } else if constexpr (
        tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
        tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
        tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        ASSERT(is_sharded_mode);
        auto sharded_tensor_rt_args = sharded_tensor_rt_args_t::from_args(arg_idx);

        return build_sharded_tensor_address_generator<tensor_memory_layout, tensor_is_dram>(input_sharded_tensor_ct_args, sharded_tensor_rt_args, args, tensor_base_addr);
    } else {
        ASSERT(false);
    }
}

template <tt::tt_metal::TensorMemoryLayout tensor_memory_layout, bool tensor_is_dram>
auto build_dest_address_generator(std::size_t &arg_idx, reduce_scatter_reader_common_args_t const& args, uint32_t tensor_base_addr) -> typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type {
    if constexpr (tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        if constexpr (row_major_layout) {
            return typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type{tensor_base_addr, args.page_size};
        } else {
            return typename source_tensor_addrgen<tensor_memory_layout, tensor_is_dram>::type{tensor_base_addr, args.page_size, args.in0_df};
        }
    } else if constexpr (
        tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
        tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
        tensor_memory_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        ASSERT(is_sharded_mode);

        sharded_tensor_rt_args_t sharded_tensor_rt_args;
        if constexpr (is_line_reduce_scatter) {
            sharded_tensor_rt_args = sharded_tensor_rt_args_t::from_args(arg_idx);
        }

        return build_sharded_tensor_address_generator<tensor_memory_layout, tensor_is_dram>(output_sharded_tensor_ct_args, sharded_tensor_rt_args, args, tensor_base_addr);
    } else {
        ASSERT(false);
    }
}

using advance_to_next_transfer_slice_result_t = std::tuple<
    uint32_t,  // ring_index
    uint32_t   // slice_base_page_offset
    >;
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

template <bool connected_to_producer>
struct signal_receiver {
    bool requires_last_input_from_other_sender;
    volatile uint32_t * noc_semaphore_address;

    FORCE_INLINE static signal_receiver build(bool requires_last_input_from_other_sender, std::size_t &arg_idx) {
        if constexpr (connected_to_producer) {
            if (requires_last_input_from_other_sender) {
                return {requires_last_input_from_other_sender, reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)))};
            } else {
                return {requires_last_input_from_other_sender, 0};
            }
        } else {
            return {requires_last_input_from_other_sender, 0};
        }
    }

    FORCE_INLINE void wait_min(uint32_t count) const {
        if constexpr (connected_to_producer) {
            if (requires_last_input_from_other_sender) {
                noc_semaphore_wait_min(noc_semaphore_address, count);
            }
        }
    }
};


void kernel_main() {
    std::size_t arg_idx = 0;

    constexpr uint32_t to_dm_sender_short_circuit_cb = tt::CB::c_out1;
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
    auto args = reduce_scatter_reader_common_args_t(arg_idx, get_dataformat(cb_id_in0));
    auto output_partial_signal_ready_receiver = signal_receiver<is_line_reduce_scatter>::build(args.requires_last_input_from_other_sender, arg_idx);

    auto s = build_source_address_generator<input_tensor_memory_layout, src_is_dram>(arg_idx, args, args.input_tensor_addr);

    auto d = build_dest_address_generator<output_tensor_memory_layout, dest_is_dram>(arg_idx, args, args.output_tensor_addr);


    bool width_sliced = args.tensor_slice_shape.x <= args.input_tensor_shape.x;

    volatile tt_l1_ptr uint32_t* receiver_read_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.sem_addr);

    uint32_t total_cb_pages_pushed = 0;
    uint32_t total_cb_pages_pushed_to_math = 0;

    ccl::edm::WorkerToEdmReader<ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED> reader(
        ttnn::ccl::WorkerXY(args.edm_core_noc0_core_x, args.edm_core_noc0_core_y),
        args.edm_core_buffer_address,
        num_buffers_per_channel,
        args.edm_core_semaphore_address,
        args.full_chunk_num_pages * args.page_size,
        receiver_read_semaphore_addr_ptr);

    // For the first timestep, there is no other input to reduce with, so we just send it straight to the input CB
    // of the output data movement kernel - short-circuiting past the (reducer) math kernel
    // For tile => shape in tiles
    // For RM => shape in elements
    std::size_t n_reads = 1;
    uint32_t start_ring_index = args.my_ring_idx;
    while (args.worker_slice_offset.x < args.tensor_slice_shape.x &&
           args.worker_slice_offset.y < args.tensor_slice_shape.y) {
        // Need to reset back to the start ring index because the last iteration of the tranfers read chunks
        // loop won't increment after the last iteration since the increment is within the loop body
        args.my_ring_idx = start_ring_index;
        uint32_t curr_ring_slice_start_page_offset =
            width_sliced ? args.tensor_slice_shape.x * start_ring_index
                         : args.tensor_slice_shape.y * start_ring_index * args.input_tensor_shape.x;

        auto const& next_slice_offset = advance_wrapped_slice_row_major(
            args.worker_slice_offset, args.worker_slice_shape, args.tensor_slice_shape, args.num_concurrent_workers);
        bool last_slice_of_worker = next_slice_offset.x >= args.tensor_slice_shape.x ||
                                    next_slice_offset.y >= args.tensor_slice_shape.y;

        const uint32_t worker_relative_start_offset_into_slice =
            args.worker_slice_offset.x + (args.worker_slice_offset.y * args.input_tensor_shape.x);
        const uint32_t starting_tile_id = curr_ring_slice_start_page_offset + worker_relative_start_offset_into_slice;
        uint32_t curr_tile_id = starting_tile_id;

        const uint32_t worker_relative_start_offset_into_output_slice =
            args.worker_slice_offset.x + (args.worker_slice_offset.y * args.tensor_slice_shape.x);
        uint32_t output_curr_tile_id = worker_relative_start_offset_into_output_slice;


        // Set the valid_worker_slice_shape
        coord_t valid_worker_slice_shape = args.worker_slice_shape;
        if (args.worker_slice_offset.y == args.tensor_slice_shape.y - 1) { // Worker is on last row of tensor_slice
            if (args.tensor_slice_shape.x - args.worker_slice_offset.x < args.worker_slice_shape.x) { // Worker is cutoff by the end of the tensor_slice
                valid_worker_slice_shape.x = args.tensor_slice_shape.x - args.worker_slice_offset.x;
            }
        }

        bool last_page_of_worker = false;
        uint32_t const worker_slice_n_pages = valid_worker_slice_shape.x * valid_worker_slice_shape.y;
        ASSERT(
            (args.num_transfers - 1) * worker_slice_n_pages + total_cb_pages_pushed_to_math <=
            args.total_eltwise_kernel_num_pages);

        if constexpr (!is_line_reduce_scatter) {
            // is_line_reduce_scatter enabled for linear only. In linear topology reduce scatter,
            // only the first chip in the line will forward and input without producing a partial reduction
            // output. Therefore we don't do this short-circuit when that mode is enabled
            uint32_t offset_into_worker_slice = 0;
            for (uint32_t p = 0; p < worker_slice_n_pages; p += args.full_chunk_num_pages) {

                uint32_t n_pages = std::min(args.full_chunk_num_pages, worker_slice_n_pages - p);
                ASSERT(args.half_cb_n_pages >= n_pages);
                ASSERT(!last_page_of_worker);
                read_wrapped_chunk_from_output_tensor(
                    curr_tile_id,
                    offset_into_worker_slice,
                    args.worker_slice_offset, // Offset into tensor slice
                    valid_worker_slice_shape,
                    // In tiles for tile layout
                    args.input_tensor_shape,
                    args.tensor_slice_shape,
                    to_dm_sender_short_circuit_cb,
                    s,
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
            uint32_t offset_into_worker_slice = 0;
            std::tie(args.my_ring_idx, curr_ring_slice_start_page_offset) = advance_to_next_transfer_slice(
                args.ring_size,
                args.my_ring_idx,
                curr_ring_slice_start_page_offset,
                args.input_tensor_shape,
                args.tensor_slice_shape,
                args.is_clockwise_direction);
            ASSERT(is_line_reduce_scatter || last_page_of_worker);
            last_page_of_worker = false;
            curr_tile_id = curr_ring_slice_start_page_offset + worker_relative_start_offset_into_slice;

            for (uint32_t p = 0; p < worker_slice_n_pages; p += args.full_chunk_num_pages) {
                uint32_t n_pages = std::min(args.full_chunk_num_pages, worker_slice_n_pages - p);
                ASSERT(args.half_cb_n_pages >= n_pages);
                ASSERT(n_pages > 0);
                // Fetch from input tensor

                // POTENTIALLY CONFUSING SEQUENCE AHEAD
                // Please note that we are NOT reading from the tensor twice. We just distinguish between the case
                // where our input is dependent on the other line direction and when it is not. In the former case,
                // we are waiting for the other direction to complete its output, which typically will be equidistant
                // from the very first read - in other words, we are likely to be waiting for a little bit of time.
                // During that time, we may as well try reading from EDM first, as it is likely to be ready sooner.
                bool read_from_output_tensor = is_line_reduce_scatter && last_transfer && output_partial_signal_ready_receiver.requires_last_input_from_other_sender;
                if (!read_from_output_tensor) {
                    read_wrapped_chunk_from_output_tensor(
                        curr_tile_id,
                        offset_into_worker_slice,
                        args.worker_slice_offset, // Offset into tensor slice
                        valid_worker_slice_shape,
                        // In tiles for tile layout
                        args.input_tensor_shape,
                        args.tensor_slice_shape,
                        cb_id_in1,
                        s,
                        n_pages,
                        args.page_size,
                        last_page_of_worker);
                }

                // Fetch from EDM
                bool last_worker_message_to_edm = last_transfer && last_slice_of_worker && (p + n_pages >= worker_slice_n_pages);
                reader.wait_for_payload_available();
                reader.fetch_payload_blocking(cb_id_in0, n_pages, args.page_size, last_worker_message_to_edm);

                if (read_from_output_tensor) {
                    // The last transfer for the reader, when in line reduce-scatter mode
                    // (i.e. when `is_line_reduce_scatter` is true), must fetch its
                    // local tensor input from the output tensor because the other direction
                    // of the line was designated be the first writer (of its partial result)
                    // to the output tensor while this kernel instance was designated to be the
                    // final accumulator
                    if (output_partial_signal_ready_receiver.requires_last_input_from_other_sender) {
                        output_partial_signal_ready_receiver.wait_min(n_reads++);
                    }

                    read_wrapped_chunk_from_output_tensor(
                        output_curr_tile_id,//curr_tile_id,
                        offset_into_worker_slice,
                        args.worker_slice_offset, // Offset into tensor slice
                        valid_worker_slice_shape,
                        // In tiles for tile layout
                        args.tensor_slice_shape, // output tensor shape
                        args.tensor_slice_shape,
                        cb_id_in1,
                        d,
                        n_pages,
                        args.page_size,
                        last_page_of_worker);
                }


                total_cb_pages_pushed_to_math += n_pages;
                total_cb_pages_pushed += n_pages;


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
    WAYPOINT("DRN1");
    // The host code currently doesn't know how to accuractly count the exact number of pages pushed through the
    // math reduce op so it instead provides a known safe lower bound which may be more than actually required by the
    // op. It passes this number to sender and receiver, who will push/pop junk pages to/from the math op to ensure
    // it will complete
    for (; total_cb_pages_pushed_to_math < args.total_eltwise_kernel_num_pages; total_cb_pages_pushed_to_math++) {
        push_filler_pages_to_cb(cb_id_in0, 1);
        push_filler_pages_to_cb(cb_id_in1, 1);
    }

    reader.close();
    WAYPOINT("DONE");
}
