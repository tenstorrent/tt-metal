// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>

#include <tt_stl/reflection.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "mhc_post_ttnn_device_operation_types.hpp"  // cut_grid

namespace ttnn::operations::bringup::mhc_post_ttnn {

// ttnn.bringup.mhc_post(..., stats_weight=W) (deepseek_v41_flash_d_p P.14): the residual mix of mhc_post AND, in the
// same pass over the new streams X', this chip's partial row of the next mHC hc step:
//     stats[t, :]          = X'[t, :] @ W            (W (n*C, 32) fp32: the chip's slice of hc_fn^T, FPU matmul)
//     stats[t, n (n + 2)]  = sum_c X'[t, c]^2         (exact fp32 on the SFPU; replaces that column of X' @ W)
// = mhc_pre_xing_pack(X' @ W, X'). A separate program (kernels mhc_post_stats_*): the op without stats_weight builds
// the program it always did.
struct MhcPostStatsParams {
    tt::tt_metal::ComputeConfigDescriptor compute_config;
    bool comb_transposed = true;
    uint32_t post_col = 0;  // P.19: post / comb column offsets (writer defines MHC_POST_POST_COL / _COMB_COL; 0 = off)
    uint32_t comb_col = 0;
    // Fork option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18): the op runs on the rectangle (0, 0) ..
    // (max_grid.x - 1, max_grid.y - 1) of the device's compute grid. Default (none) = the whole grid. Hashed when set.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
    // Fork option collapse_pre (deepseek_v41_flash_d_p P.34, POST_COLLAPSE; false = off: every program, arg and hash as
    // before): the same pass also writes the next mHC step's inputs, y = sum_i pre_i X'_i (the collapse with the given
    // pre, tensor_args.collapse_pre columns 0..n-1; mhc_pre_xing(collapse_pre=)'s arithmetic on the bf16 X') and, on
    // each row's root after the partials' sum, that step's hc = DeepSeek's coefficients + Sinkhorn of the stats row
    // (mhc_pre_xing(deepseek=True)'s arithmetic). The scalars are runtime args patched on every call (not hashed).
    bool collapse = false;
    std::array<double, 3> hc_scale{};
    std::vector<double> hc_base;  // n (n + 2)
    double hc_inv_nc = 0.0;
    double hc_norm_eps = 1e-6;
    double hc_eps = 1e-6;
    uint32_t hc_sinkhorn_iters = 20;
    // Fork option norm_weight (deepseek_v41_flash_d_p P.35, NORM_IN_MHC; false = off: every program, arg and hash as
    // before; needs collapse): y_output holds rms_norm(y) * norm_weight instead of y, y = the bf16 collapse:
    // bf16(bf16(y) * rsqrt(sum bf16(y)^2 / C + rms_eps) * w), the statistic an exact fp32 SFPU sum (a spare column of
    // the partial rows, n (n + 2) + 1), rsqrt on each row's root, sent back to the row's units through the scratch.
    // The scalars are runtime args (not hashed).
    bool norm = false;
    double rms_inv_w = 0.0;
    double rms_eps = 1e-6;
};

// The collapse option's call arguments (P.34): the next step's pre, its coefficient scalars, the caller's buffers.
struct MhcPostCollapseArgs {
    Tensor collapse_pre;
    std::array<double, 3> scale{};
    std::vector<double> base;
    double inv_nc = 0.0;
    double norm_eps = 1e-6;
    double hc_eps = 1e-6;
    uint32_t sinkhorn_iters = 20;
    std::optional<Tensor> hc_output;
    std::optional<Tensor> y_output;
    // P.35 (norm): the norm's weight (..., 1, C) fp32 TILE (None = off) and epsilon
    std::optional<Tensor> norm_weight;
    double rms_eps = 1e-6;
};

struct MhcPostStatsInputs {
    Tensor input;     // F   (..., T, C)      fp32, or bf16 (P.19: its own CB, define MHC_POST_STATS_F16)
    Tensor residual;  // X   (..., T, n*C)    fp32
    Tensor post;      // (..., T, n)          fp32
    Tensor comb;      // (..., T, n*n)        fp32
    Tensor weight;    // W   (..., n*C, 32)   fp32, leading dims 1
    // P.16 (default absent = the op allocates): the caller's X', stats and scratch buffers
    std::optional<Tensor> output;
    std::optional<Tensor> stats_output;
    std::optional<Tensor> stats_scratch;
    // P.34 (collapse): the pre y collapses with (fp32 (..., T, n..32), columns 0..n-1) and the caller's hc / y buffers
    std::optional<Tensor> collapse_pre;
    std::optional<Tensor> hc_output;
    std::optional<Tensor> y_output;
    // P.35 (norm): the norm's weight, (..., 1, C) fp32 TILE
    std::optional<Tensor> norm_weight;
};

// The work plan (host): column blocks of `cols_per_block` tiles; a unit = one block of one token tile-row, units
// block-major (u = b * row_tiles + r) split contiguously over the grid. Each unit's partial row goes to a DRAM scratch
// page; the root core of each row sums its `blocks` partials in block order (see the program factory).
struct MhcPostStatsPlan {
    uint32_t n = 0, col_tiles = 0, row_tiles = 0, cols_per_block = 0, blocks = 0, units = 0;
};
MhcPostStatsPlan mhc_post_stats_plan(
    const MhcPostStatsInputs& in, const std::optional<tt::tt_metal::CoreCoord>& max_grid = std::nullopt);

struct MhcPostStatsProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MhcPostStatsParams& operation_attributes,
        const MhcPostStatsInputs& tensor_args,
        std::vector<Tensor>& outputs);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MhcPostStatsParams& operation_attributes,
        const MhcPostStatsInputs& tensor_args,
        std::vector<Tensor>& outputs,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct MhcPostStatsDeviceOperation {
    using operation_attributes_t = MhcPostStatsParams;
    using tensor_args_t = MhcPostStatsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;  // [X', stats, scratch] (+ [hc, y] with collapse)
    using program_factory_t = std::variant<MhcPostStatsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::mhc_post_ttnn

namespace ttnn::prim::bringup {
std::vector<ttnn::Tensor> mhc_post_stats(
    const ttnn::Tensor& input,
    const ttnn::Tensor& residual,
    const ttnn::Tensor& post,
    const ttnn::Tensor& comb,
    const ttnn::Tensor& weight,
    const tt::tt_metal::ComputeConfigDescriptor& compute_config,
    bool comb_transposed,
    uint32_t post_col = 0,
    uint32_t comb_col = 0,
    const std::optional<ttnn::Tensor>& output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& stats_output = std::nullopt,
    const std::optional<ttnn::Tensor>& stats_scratch = std::nullopt,
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt,
    const std::optional<ttnn::operations::bringup::mhc_post_ttnn::MhcPostCollapseArgs>& collapse = std::nullopt);
}  // namespace ttnn::prim::bringup
