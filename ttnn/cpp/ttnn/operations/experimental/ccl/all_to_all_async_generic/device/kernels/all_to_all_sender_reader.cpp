#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tile.h"
#include "dataflow_api.h"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
constexpr uint32_t current_device_id = get_compile_time_arg_val(2);
constexpr uint32_t ring_size = get_compile_time_arg_val(3);
constexpr uint32_t outer_dims_size = get_compile_time_arg_val(4);
constexpr uint32_t split_dim_size = get_compile_time_arg_val(5);
constexpr uint32_t inner_dims_size = get_compile_time_arg_val(6);
constexpr uint32_t last_dim_size = get_compile_time_arg_val(7);
constexpr uint32_t split_block_size = get_compile_time_arg_val(8);
constexpr uint32_t has_reader_tail = get_compile_time_arg_val(9);
constexpr uint32_t has_writer_tail = get_compile_time_arg_val(10);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t input_address = get_arg_val<address_t>(arg_idx++);
    constexpr auto input_tensor_args = TensorAccessorArgs<11>();
    auto input_addrgen = TensorAccessor(input_tensor_args, input_address, input_page_size);

    for (uint32_t device_id = 0; device_id < ring_size; ++device_id) {
        const uint32_t device_read_offset =
            (has_reader_tail) ? split_block_size * device_id + device_id / 2 : split_block_size * device_id;
        for (uint32_t o = 0; o < outer_dims_size; ++o) {
            for (uint32_t s = device_read_offset; s < device_read_offset + split_block_size; ++s) {
                for (uint32_t i = 0; i < inner_dims_size; ++i) {
                    const uint32_t tile_id = o * inner_dims_size * split_dim_size + s * inner_dims_size + i;
                    cb_reserve_back(cb0_id, 1);
                    address_t l1_write_addr = get_write_ptr(cb0_id);
                    if (has_reader_tail && device_id % 2 == 1) {
                        noc_async_read(
                            input_addrgen.get_noc_addr(tile_id, input_page_size / 2),
                            l1_write_addr,
                            input_page_size / 2);
                        noc_async_read_barrier();
                        noc_async_read(
                            input_addrgen.get_noc_addr(tile_id + inner_dims_size, 0),
                            l1_write_addr + input_page_size / 2,
                            input_page_size / 2);
                        noc_async_read_barrier();
                    } else {
                        if (has_writer_tail && current_device_id % 2 == 1 && i / last_dim_size > 0) {
                            noc_async_read(
                                input_addrgen.get_noc_addr(tile_id - last_dim_size, input_page_size / 2),
                                l1_write_addr,
                                input_page_size / 2);
                            noc_async_read(
                                input_addrgen.get_noc_addr(tile_id, 0),
                                l1_write_addr + input_page_size / 2,
                                input_page_size / 2);
                        } else {
                            noc_async_read_tile(tile_id, input_addrgen, l1_write_addr);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb0_id, 1);
                }
            }

            // partial tile reads
            if (has_reader_tail) {
                for (uint32_t i = 0; i < inner_dims_size; ++i) {
                    const uint32_t tile_id = o * inner_dims_size * split_dim_size +
                                             (device_read_offset + split_block_size) * inner_dims_size + i;
                    cb_reserve_back(cb0_id, 1);
                    address_t l1_write_addr = get_write_ptr(cb0_id);
                    noc_async_read(
                        input_addrgen.get_noc_addr(tile_id, (device_id % 2) * input_page_size / 2),
                        l1_write_addr,
                        input_page_size / 2);
                    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
                    noc_async_read(zeros_noc_addr, l1_write_addr + input_page_size / 2, input_page_size / 2);
                    noc_async_read_barrier();
                    cb_push_back(cb0_id, 1);
                }
            }
        }
    }
}
