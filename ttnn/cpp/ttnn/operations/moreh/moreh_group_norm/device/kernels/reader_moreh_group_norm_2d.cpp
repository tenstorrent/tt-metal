// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

/**
 * @brief Generates a mask with bits set inside the valid range [start, end).
 *
 * This function modifies the `cb_mask` by setting bits that correspond to values
 * within the valid range [start, end). The values outside this range are considered
 * invalid and will not have their corresponding bits set.
 *
 * @param cb_mask The mask to be modified. Bits outside the range [start, end) will be set.
 * @param start The start of the valid range (inclusive).
 * @param end The end of the valid range (exclusive).
 */
FORCE_INLINE void generate_mask_w(uint32_t cb_mask, uint32_t start, uint32_t end) {
    start = start % TILE_WIDTH;
    end = (end + TILE_WIDTH - 1) % TILE_WIDTH + 1;
    Scalar one;
    Scalar zero;

    one.f = 1.0f;
    zero.f = 0.0f;

    uint16_t u16_one = uint16_t(one.u >> 16);
    uint16_t u16_zero = uint16_t(zero.u >> 16);

    cb_reserve_back(cb_mask, 1);
    auto ptr = reinterpret_cast<uint16_t *>(get_write_ptr(cb_mask));

    for (uint32_t h = 0; h < FACE_HEIGHT; h++) {
        // sub tile 0
        {
            uint32_t offset = h * FACE_WIDTH;
            if (start >= FACE_WIDTH) {
                for (uint32_t w = 0; w < FACE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

            } else {
                for (uint32_t w = 0; w < start; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

                uint32_t w_end = std::min(end, FACE_WIDTH);
                for (uint32_t w = start; w < w_end; w++) {
                    // valid
                    ptr[offset + w] = u16_one;
                }

                for (uint32_t w = w_end; w < FACE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }
            }
        }

        // sub tile 1
        {
            uint32_t offset = h * FACE_WIDTH + 256 - FACE_WIDTH;

            if (end <= FACE_WIDTH) {
                for (uint32_t w = FACE_WIDTH; w < TILE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

            } else {
                for (uint32_t w = FACE_WIDTH; w < start; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

                uint32_t w_start = std::max(start, FACE_WIDTH);
                for (uint32_t w = w_start; w < end; w++) {
                    // valid
                    ptr[offset + w] = u16_one;
                }

                for (uint32_t w = end; w < TILE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }
            }
        }

        // sub tile 2
        {
            uint32_t offset = h * FACE_WIDTH + 512;

            if (start >= FACE_WIDTH) {
                for (uint32_t w = 0; w < FACE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

            } else {
                for (uint32_t w = 0; w < start; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

                uint32_t w_end = std::min(end, FACE_WIDTH);
                for (uint32_t w = start; w < w_end; w++) {
                    // valid
                    ptr[offset + w] = u16_one;
                }

                for (uint32_t w = w_end; w < FACE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }
            }
        }

        // sub tile 3
        {
            uint32_t offset = h * FACE_WIDTH + 768 - FACE_WIDTH;

            if (end <= FACE_WIDTH) {
                for (uint32_t w = FACE_WIDTH; w < TILE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

            } else {
                for (uint32_t w = FACE_WIDTH; w < start; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }

                uint32_t w_start = std::max(start, FACE_WIDTH);
                for (uint32_t w = w_start; w < end; w++) {
                    // valid
                    ptr[offset + w] = u16_one;
                }

                for (uint32_t w = end; w < TILE_WIDTH; w++) {
                    // invalid
                    ptr[offset + w] = u16_zero;
                }
            }
        }
    }

    cb_push_back(cb_mask, 1);
}

template <typename T>
FORCE_INLINE void read_input(
    uint32_t cb_id, T addrgen, uint32_t read_start, uint32_t read_end, uint32_t nt, uint32_t Ct, uint32_t N) {
    constexpr uint32_t onetile = 1;
    const auto elem_size = get_tile_size(cb_id) >> 10;

    cb_reserve_back(cb_id, onetile);
    auto l1_write_addr = get_write_ptr(cb_id);

    uint32_t ct = read_start / TILE_WIDTH;
    auto noc_id = nt * Ct + ct;
    auto noc_addr = get_noc_addr(noc_id, addrgen);

    // read left face
    if (read_start % TILE_WIDTH < FACE_WIDTH) {
        // read [tile_start, tile_end)
        uint32_t tile_start = read_start;
        uint32_t tile_end = std::min(round_down(read_start, TILE_WIDTH) + FACE_WIDTH, read_end);

        // n loop
        uint32_t n_start = nt * TILE_HEIGHT;
        uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);

        auto noc_read_size = (tile_end - tile_start) * elem_size;
        for (uint32_t n = n_start; n < n_end; n++) {
            uint32_t tilized_idx = get_tilized_idx(n, tile_start);
            uint32_t offset = tilized_idx * elem_size;

            noc_async_read(noc_addr + offset, l1_write_addr + offset, noc_read_size);
        }  // n loop

    }  // read left face

    // read right face
    if ((read_end + TILE_WIDTH - 1) % TILE_WIDTH >= FACE_WIDTH) {
        // read [tile_start, tile_end)
        uint32_t tile_start = std::max(round_down(read_start, TILE_WIDTH) + FACE_WIDTH, read_start);
        uint32_t tile_end = read_end;

        // n loop
        uint32_t n_start = nt * TILE_HEIGHT;
        uint32_t n_end = std::min((nt + 1) * TILE_HEIGHT, N);
        auto noc_read_size = (tile_end - tile_start) * elem_size;
        for (uint32_t n = n_start; n < n_end; n++) {
            uint32_t tilized_idx = get_tilized_idx(n, tile_start);
            uint32_t offset = tilized_idx * elem_size;

            noc_async_read(noc_addr + offset, l1_write_addr + offset, noc_read_size);
        }  // n loop

    }  // read right face

    noc_async_read_barrier();

    cb_push_back(cb_id, onetile);
}

template <typename T, typename U, typename V>
void read_all_tiles(
    uint32_t cb_input,
    T input_addrgen,
    uint32_t cb_gamma,
    U gamma_addrgen,
    uint32_t cb_beta,
    V beta_addrgen,
    uint32_t c_start,
    uint32_t c_end,
    uint32_t nt,
    uint32_t Ct,
    uint32_t N,
    uint32_t cb_mask,
    bool read_gamma,
    bool read_beta,
    bool create_mask) {
    uint32_t read_start = c_start;
    uint32_t read_end = c_start;

    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);

    for (;;) {
        // compute read range in tiles
        read_start = read_end;
        read_end = std::min(round_up(read_start + 1, TILE_WIDTH), c_end);

        if (read_start >= c_end) {
            break;
        }

        // read [read_start, read_end]
        bool is_tile_end = (read_end % TILE_WIDTH == 0);
        bool is_last = (read_end == c_end);
        if (is_tile_end || is_last) {
            read_input(cb_input, input_addrgen, read_start, read_end, nt, Ct, N);

            // generate mask
            bool mask_required = create_mask && ((read_start % TILE_WIDTH != 0) || (read_end % TILE_WIDTH != 0));
            if (mask_required) {
                generate_mask_w(cb_mask, read_start, read_end);
            }

            uint32_t ct = read_start / TILE_WIDTH;
            // gamma[1, C]
            if (read_gamma) {
                uint32_t noc_id = ct;
                read_tile(cb_gamma, gamma_addrgen, noc_id, gamma_tile_bytes);
            }

            // beta[1, C]
            if (read_beta) {
                uint32_t noc_id = ct;
                read_tile(cb_beta, beta_addrgen, noc_id, beta_tile_bytes);
            }
        }
    }
}

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto gamma_addr = get_arg_val<uint32_t>(i++);
    const auto beta_addr = get_arg_val<uint32_t>(i++);

    const auto scaler = get_arg_val<uint32_t>(i++);
    const auto eps = get_arg_val<uint32_t>(i++);

    const auto unit_offset = get_arg_val<uint32_t>(i++);
    const auto num_units_per_core = get_arg_val<uint32_t>(i++);

    const auto N = get_arg_val<uint32_t>(i++);
    const auto C = get_arg_val<uint32_t>(i++);
    const auto num_groups = get_arg_val<uint32_t>(i++);

    // compile-time args
    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool gamma_has_value = get_compile_time_arg_val(1) == 1;
    constexpr bool gamma_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool beta_has_value = get_compile_time_arg_val(3) == 1;
    constexpr bool beta_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t onetile = 1;

    const auto Nt = div_up(N, TILE_HEIGHT);
    const auto Ct = div_up(C, TILE_WIDTH);

    uint32_t cb_id = tt::CB::c_in0;
    const auto cb_input = cb_id++;
    const auto cb_scaler = cb_id++;
    const auto cb_eps = cb_id++;
    const auto cb_gamma = cb_id++;
    const auto cb_beta = cb_id++;
    const auto cb_mask = cb_id++;
    const auto cb_zeros = cb_id++;

    fill_cb_with_value(cb_scaler, scaler);
    fill_cb_with_value(cb_eps, eps);
    fill_cb_with_zeros(cb_zeros);

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input_data_format = get_dataformat(cb_input);
    const auto input_elem_size = get_tile_size(cb_input) >> 10;

    const InterleavedAddrGenFast<input_is_dram> input_addrgen = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    // gamma
    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
    const auto gamma_data_format = get_dataformat(cb_gamma);
    const InterleavedAddrGenFast<gamma_is_dram> gamma_addrgen = {
        .bank_base_address = gamma_addr, .page_size = gamma_tile_bytes, .data_format = gamma_data_format};

    // beta
    const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
    const auto beta_data_format = get_dataformat(cb_beta);
    const InterleavedAddrGenFast<beta_has_value> beta_addrgen = {
        .bank_base_address = beta_addr, .page_size = beta_tile_bytes, .data_format = beta_data_format};

    for (uint32_t outer_idx = 0; outer_idx < num_units_per_core; ++outer_idx) {
        // input[N, C]
        // reshaped_input[N, num_groups, C / num_groups]
        auto unit_idx = unit_offset + outer_idx;
        auto group_idx = unit_idx % num_groups;
        auto nt = unit_idx / num_groups;

        // read range [c_start, c_end)
        uint32_t c_size_per_unit = C / num_groups;
        uint32_t c_start = c_size_per_unit * group_idx;
        uint32_t c_end = c_size_per_unit * (group_idx + 1);

        // read x For E[x]
        read_all_tiles(
            cb_input,
            input_addrgen,
            cb_gamma,
            gamma_addrgen,
            cb_beta,
            beta_addrgen,
            c_start,
            c_end,
            nt,
            Ct,
            N,
            cb_mask,
            false,
            false,
            true);

        // read x for x - E[x]
        read_all_tiles(
            cb_input,
            input_addrgen,
            cb_gamma,
            gamma_addrgen,
            cb_beta,
            beta_addrgen,
            c_start,
            c_end,
            nt,
            Ct,
            N,
            cb_mask,
            false,
            false,
            true);

        // read x for (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
        read_all_tiles(
            cb_input,
            input_addrgen,
            cb_gamma,
            gamma_addrgen,
            cb_beta,
            beta_addrgen,
            c_start,
            c_end,
            nt,
            Ct,
            N,
            cb_mask,
            gamma_has_value,
            beta_has_value,
            false);

    }  // num_units_per_core

}  // void kernel_main()
