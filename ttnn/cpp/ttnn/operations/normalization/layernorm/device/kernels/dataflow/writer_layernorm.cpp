#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    int i = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(i++);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t input_Wt = get_arg_val<uint32_t>(i++);
    const uint32_t output_Wt = get_arg_val<uint32_t>(i++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t one_tile = 1;
    constexpr auto output_args = TensorAccessorArgs<0>();

    const uint32_t output_tile_bytes = get_tile_size(cb_output);
    const auto output = TensorAccessor(output_args, output_addr, output_tile_bytes);

    const uint32_t num_tiles = num_rows_per_core * output_Wt;
    const uint32_t start_tile_idx = (tile_offset / input_Wt) * output_Wt;
    for (uint32_t tile_idx = start_tile_idx; tile_idx < start_tile_idx + num_tiles; ++tile_idx) {
        cb_wait_front(cb_output, one_tile);
        const uint32_t output_l1_read_addr = get_read_ptr(cb_output);
        noc_async_write_tile(tile_idx, output, output_l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output, one_tile);
    }
}
