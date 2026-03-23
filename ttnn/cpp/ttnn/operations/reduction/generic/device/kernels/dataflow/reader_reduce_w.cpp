#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t H = get_arg_val<uint32_t>(2);
    uint32_t W = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t scaler = get_arg_val<uint32_t>(6);
    uint32_t eps = get_arg_val<uint32_t>(7);
    uint32_t stick_size = get_arg_val<uint32_t>(8);
    uint32_t reduce_type = get_arg_val<uint32_t>(9);
    uint32_t origin_H = get_arg_val<uint32_t>(10);
    uint32_t origin_W = get_arg_val<uint32_t>(11);
    uint32_t padded_W = get_arg_val<uint32_t>(12);
    uint32_t padded_Wt = get_arg_val<uint32_t>(13);
    uint32_t has_padding = get_arg_val<uint32_t>(14);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_eps = 2;
    constexpr uint32_t cb_id_one = 3;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    cb_reserve_back(cb_id_scaler, onetile);
    cb_reserve_back(cb_id_eps, onetile);
    cb_reserve_back(cb_id_one, onetile);

    // Fill scaler with constant values
    auto scaler_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id_scaler));
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint16_t); ++i) {
        scaler_ptr[i] = scaler;
    }

    // Fill eps with constant values
    auto eps_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id_eps));
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint16_t); ++i) {
        eps_ptr[i] = eps;
    }

    // Fill one with constant 1.0 values (bf16 format: 0x3F80)
    auto one_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_id_one));
    for (uint32_t i = 0; i < tile_bytes / sizeof(uint16_t); ++i) {
        one_ptr[i] = 0x3F80;
    }

    cb_push_back(cb_id_scaler, onetile);
    cb_push_back(cb_id_eps, onetile);
    cb_push_back(cb_id_one, onetile);

    uint32_t l1_write_addr_in0;
    uint32_t curr_tile_id = 0;

    // Process tiles with padding awareness
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t wt = 0; wt < Wt; wt++) {
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                
                uint32_t tile_id;
                
                if (has_padding && wt >= (origin_W / 32)) {
                    // Handle padded region
                    if (padded_Wt > 0) {
                        // Use padded tile indexing
                        tile_id = n * H * padded_Wt + h * padded_Wt + wt;
                    } else {
                        // Zero tile for padding
                        memset(reinterpret_cast<void*>(l1_write_addr_in0), 0, tile_bytes);
                        cb_push_back(cb_id_in0, onetile);
                        continue;
                    }
                } else {
                    // Normal tile indexing
                    tile_id = n * H * Wt + h * Wt + wt;
                }
                
                noc_async_read_tile(tile_id, s, l1_write_addr_in0);
                noc_async_read_barrier();
                
                cb_push_back(cb_id_in0, onetile);
                curr_tile_id++;
            }
        }
    }
}