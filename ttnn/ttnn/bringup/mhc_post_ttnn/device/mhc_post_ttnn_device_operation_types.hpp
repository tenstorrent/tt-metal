// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_post_ttnn {

// The op's attributes: everything the program depends on besides the tensors' specs. The compute config is the
// resolved one (the caller's, or default_compute_kernel_config()); the builder copies math_fidelity,
// fp32_dest_acc_en and math_approx_mode from it, as mhc_post_program_descriptor.py does.
struct MhcPostParams {
    tt::tt_metal::ComputeConfigDescriptor compute_config;
    // true (default): X'_j = post_j F + sum_i comb[i*n + j] X_i (comb^T, the op's original math).
    // false: X'_j = post_j F + sum_i comb[j*n + i] X_i (comb as stored; kernel define MHC_POST_COMB_DIRECT).
    bool comb_transposed = true;
    // post_k at post's column post_col + k, comb[i*n + j] at comb's column comb_col + i*n + j (fork option, P.19;
    // DM defines MHC_POST_POST_COL / MHC_POST_COMB_COL; 0 / 0 = the program it always was)
    uint32_t post_col = 0;
    uint32_t comb_col = 0;
    // Fork option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18): the op runs on the rectangle (0, 0) ..
    // (max_grid.x - 1, max_grid.y - 1) of the device's compute grid. Default (none) = the whole grid. Hashed when set.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
};

// the device's compute grid cut to max_grid (FIX.15)
inline tt::tt_metal::CoreCoord cut_grid(
    tt::tt_metal::CoreCoord g, const std::optional<tt::tt_metal::CoreCoord>& max_grid) {
    if (max_grid.has_value()) {
        g.x = std::max<std::size_t>(1, std::min<std::size_t>(g.x, max_grid->x));
        g.y = std::max<std::size_t>(1, std::min<std::size_t>(g.y, max_grid->y));
    }
    return g;
}

struct MhcPostInputs {
    Tensor input;                  // F   (..., T, C)      the sublayer output
    Tensor residual;               // X   (..., T, n*C)    the n residual streams
    Tensor post;                   // (..., T, n)          float32
    Tensor comb;                   // (..., T, n*n)        float32, comb[i*n + j]
    std::optional<Tensor> output;  // output_tensor (P.16; default absent = the op allocates X')
};

// The refusals keep the Python op's exception types: std::invalid_argument surfaces as ValueError (nanobind's
// built-in translation); UnsupportedAxisError is translated by the binding to
// ttnn.operations._op_contract.UnsupportedAxisValue.
struct ValueErrorCpp : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
struct UnsupportedAxisError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
