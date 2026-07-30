// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace ttnn::prim::sdpa_cb {

struct CBIds {
    static constexpr uint32_t inactive = std::numeric_limits<uint32_t>::max();

    uint32_t q_in = inactive;
    uint32_t k_in = inactive;
    uint32_t v_in = inactive;
    uint32_t mask_in = inactive;
    uint32_t attention_sink = inactive;
    uint32_t identity_scale_in = inactive;
    uint32_t page_table = inactive;
    uint32_t col_identity = inactive;
    uint32_t chunk_start_idx_compute = inactive;
    uint32_t chunk_start_idx_writer = inactive;
    uint32_t recip_scratch = inactive;
    uint32_t out = inactive;
    uint32_t qk_im = inactive;
    uint32_t out_im_A = inactive;
    uint32_t out_im_B = inactive;
    uint32_t max_A = inactive;
    uint32_t max_B = inactive;
    uint32_t sum_A = inactive;
    uint32_t sum_B = inactive;
    uint32_t exp_max_diff = inactive;
    uint32_t cu_window_seqlens = inactive;  // windowed mode only

    std::vector<uint32_t> reader_compile_time_args() const {
        return {q_in, k_in, v_in, mask_in, attention_sink, page_table, chunk_start_idx_compute, chunk_start_idx_writer};
    }

    std::vector<uint32_t> writer_compile_time_args() const {
        return {mask_in, identity_scale_in, col_identity, chunk_start_idx_writer, out, cu_window_seqlens};
    }

    std::vector<uint32_t> compute_compile_time_args() const {
        return {
            q_in,
            k_in,
            v_in,
            mask_in,
            attention_sink,
            identity_scale_in,
            col_identity,
            chunk_start_idx_compute,
            recip_scratch,
            out,
            qk_im,
            out_im_A,
            out_im_B,
            max_A,
            max_B,
            sum_A,
            sum_B,
            exp_max_diff};
    }
};

}  // namespace ttnn::prim::sdpa_cb
