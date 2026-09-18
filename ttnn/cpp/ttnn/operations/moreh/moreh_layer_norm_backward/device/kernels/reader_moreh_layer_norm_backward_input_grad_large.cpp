// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <typename T>
void read_mean_rstd(
    const Noc& noc,
    DFBBindingToken dfb_token,
    uint32_t tile_offset,
    uint32_t normalized_dims,
    uint32_t outer_idx,
    uint32_t height,
    uint32_t width,
    uint32_t Ht,
    uint32_t Wt,
    T addrg) {
    using namespace tt::constants;
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb(dfb_token);
    const uint32_t tile_bytes = dfb.get_tile_size();
    const auto dtype_bytes = tile_bytes / (TILE_HEIGHT * TILE_WIDTH);

    dfb.reserve_back(onetile);

    uint32_t l1_write_addr = dfb.get_write_ptr();
    CoreLocalMem<volatile uint16_t> l1_ptr(l1_write_addr);
    if (normalized_dims == 1) {
        for (uint32_t src_h = 0; src_h < 2; src_h++) {
            auto tile_idx = tile_offset + outer_idx;

            auto wt = tile_idx % Wt;
            auto nh = tile_idx / Wt;
            auto h = nh % height;
            auto n = nh / height;

            auto w = src_h * FACE_HEIGHT;

            auto tilized_idx = get_tilized_idx(h % TILE_HEIGHT, w);

            auto ht = h / TILE_HEIGHT;
            auto noc_id = n * Ht * Wt + ht * Wt + wt;

            auto src_idx = get_tilized_idx(0, src_h * FACE_WIDTH);

            noc.async_read(
                addrg,
                dfb,
                dtype_bytes * FACE_HEIGHT,
                {.page_id = noc_id, .offset_bytes = tilized_idx * dtype_bytes},
                {.offset_bytes = src_idx * dtype_bytes});

            noc.async_read_barrier();
        }

        // rotate data
        for (uint32_t i = 0; i < 16; i++) {
            l1_ptr[i * FACE_WIDTH] = l1_ptr[i];
            l1_ptr[i * FACE_WIDTH + 256 * 2] = l1_ptr[i + 256];
        }
    } else {
        auto idx = tile_offset + outer_idx;

        auto w = idx % width;
        auto nh = idx / width;
        auto h = nh % height;
        auto n = nh / height;

        auto tilized_idx = get_tilized_idx(h % TILE_HEIGHT, w % TILE_WIDTH);

        auto wt = w / TILE_WIDTH;
        auto ht = h / TILE_HEIGHT;

        auto noc_id = n * Ht * Wt + ht * Wt + wt;

        noc.async_read(
            addrg,
            dfb,
            dtype_bytes,
            {.page_id = noc_id, .offset_bytes = tilized_idx * dtype_bytes},
            {.offset_bytes = tilized_idx * dtype_bytes});

        noc.async_read_barrier();
        if (idx != 0) {
            l1_ptr[0] = l1_ptr[tilized_idx];
        }
    }

    dfb.push_back(onetile);
}

void kernel_main() {
    using namespace tt::constants;
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto num_inner = get_arg(args::num_inner);
    const auto tile_offset = get_arg(args::tile_offset);
    const auto n = get_arg(args::n);
    const auto recip_n = get_arg(args::recip_n);
    const auto mask_h = get_arg(args::mask_h);
    const auto mask_w = get_arg(args::mask_w);
    const auto normalized_dims = get_arg(args::normalized_dims);
    const auto mean_rstd_height = get_arg(args::mean_rstd_height);
    const auto mean_rstd_width = get_arg(args::mean_rstd_width);

    // GAMMA_HAS_VALUE / DO_MASK_H / DO_MASK_W arrive as preprocessor defines rather than as
    // arguments, because each selects whether the host binds a resource; a name the host did not bind
    // does not exist in this build, and even a discarded `if constexpr` branch would still look it up.

    const auto output_grad_addrg = TensorAccessor(tensor::output_grad);
    const auto input_addrg = TensorAccessor(tensor::input);
    const auto mean_addrg = TensorAccessor(tensor::mean);
    const auto rstd_addrg = TensorAccessor(tensor::rstd);

#ifdef GAMMA_HAS_VALUE
    const auto gamma_addrg = TensorAccessor(tensor::gamma);
#endif

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_n_recip_n(dfb::n_recip_n);
    fill_cb_with_value(dfb_scaler, scaler.u);
    fill_cb_with_value(dfb_n_recip_n, n);
    fill_cb_with_value(dfb_n_recip_n, recip_n);

#if defined(DO_MASK_H) || defined(DO_MASK_W)
    DataflowBuffer dfb_mask_h_w(dfb::mask_h_w);
    generate_mask_h_w(dfb_mask_h_w, mask_h, mask_w);
#endif

    uint32_t offs = 0;
    constexpr uint32_t onetile = 1;

    auto mean_rstd_Ht = (mean_rstd_height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    auto mean_rstd_Wt = (mean_rstd_width + TILE_WIDTH - 1) / TILE_WIDTH;

    Noc noc;
    DataflowBuffer dfb_output_grad(dfb::output_grad);
    DataflowBuffer dfb_input(dfb::input);
    const auto output_grad_tile_bytes = dfb_output_grad.get_tile_size();
    const auto input_tile_bytes = dfb_input.get_tile_size();
#ifdef GAMMA_HAS_VALUE
    DataflowBuffer dfb_gamma(dfb::gamma);
    const auto gamma_tile_bytes = dfb_gamma.get_tile_size();
#endif

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        uint32_t mean_rstd_tile_offset = tile_offset / num_inner;

        // mean
        read_mean_rstd(
            noc,
            dfb::mean,
            mean_rstd_tile_offset,
            normalized_dims,
            outer_idx,
            mean_rstd_height,
            mean_rstd_width,
            mean_rstd_Ht,
            mean_rstd_Wt,
            mean_addrg);

        // rstd
        read_mean_rstd(
            noc,
            dfb::rstd,
            mean_rstd_tile_offset,
            normalized_dims,
            outer_idx,
            mean_rstd_height,
            mean_rstd_width,
            mean_rstd_Ht,
            mean_rstd_Wt,
            rstd_addrg);

        // For Sum[dy] and Sum[y * dy]
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            // input (N, C, H, W)
            dfb_input.reserve_back(onetile);
            noc.async_read(
                input_addrg,
                dfb_input,
                input_tile_bytes,
                {.page_id = offs + inner_idx + tile_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(onetile);

            // output_grad (N, C, H, W)
            dfb_output_grad.reserve_back(onetile);
            noc.async_read(
                output_grad_addrg,
                dfb_output_grad,
                output_grad_tile_bytes,
                {.page_id = offs + inner_idx + tile_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_output_grad.push_back(onetile);

#ifdef GAMMA_HAS_VALUE
            // gamma (1, 1, 1, W)
            dfb_gamma.reserve_back(onetile);
            noc.async_read(gamma_addrg, dfb_gamma, gamma_tile_bytes, {.page_id = inner_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_gamma.push_back(onetile);
#endif  // GAMMA_HAS_VALUE
        }  // num_inner loop

        // For ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            // output_grad (N, C, H, W)
            dfb_output_grad.reserve_back(onetile);
            noc.async_read(
                output_grad_addrg,
                dfb_output_grad,
                output_grad_tile_bytes,
                {.page_id = offs + inner_idx + tile_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_output_grad.push_back(onetile);

#ifdef GAMMA_HAS_VALUE
            // gamma (1, 1, 1, W)
            dfb_gamma.reserve_back(onetile);
            noc.async_read(gamma_addrg, dfb_gamma, gamma_tile_bytes, {.page_id = inner_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_gamma.push_back(onetile);
#endif  // GAMMA_HAS_VALUE

            // input (N, C, H, W)
            dfb_input.reserve_back(onetile);
            noc.async_read(
                input_addrg,
                dfb_input,
                input_tile_bytes,
                {.page_id = offs + inner_idx + tile_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(onetile);
        }  // num_inner loop

        offs += num_inner;
    }  // num_rows_per_core loop
}  // void kernel_main()
