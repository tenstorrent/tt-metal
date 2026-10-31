// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    using tensor_return_value_t = std::vector<Tensor>;  // [X', stats, scratch]
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
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt);
}  // namespace ttnn::prim::bringup
